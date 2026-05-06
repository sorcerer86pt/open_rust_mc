//! 17×17 PWR fuel assembly demo.
//!
//! First geometry exercising the recursive-geometry machinery in
//! anger: a Westinghouse-pattern 17×17 fuel assembly built as a
//! `RectLattice` of pin universes (264 UO2 fuel pins + 24 guide tubes
//! + 1 instrument tube), wrapped by a reflective box on x and y and
//! vacuum on z.
//!
//! Stack depth at every transport step is 3:
//!   root → assembly lattice → pin/GT cell.
//!
//! This is the proof-of-life demo for tasks #1–#10. A real comparison
//! against an OpenMC reference is left for a follow-up; the goal here
//! is "produces a sensible k-eff with non-trivial physics on a
//! geometry that the flat code path could not represent."
//!
//! Usage:
//!   pwr_assembly <data_dir> [--rank K] [--batches N] [--particles N] [--seeds S]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::lattice::RectLattice;
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::universe::{Universe, UniverseId};
use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
use open_rust_mc::hdf5_reader;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider;

#[derive(Parser, Debug)]
#[command(
    name = "pwr_assembly",
    about = "17×17 PWR fuel assembly k-inf demo (recursive geometry)"
)]
struct Args {
    /// Directory containing nuclide HDF5 files.
    data_dir: PathBuf,

    #[arg(short, long, default_value_t = 5)]
    rank: usize,

    #[arg(short, long, default_value_t = 50)]
    batches: u32,

    #[arg(short, long, default_value_t = 15)]
    inactive: u32,

    #[arg(short, long, default_value_t = 10000)]
    particles: u32,

    #[arg(short, long, default_value_t = 1)]
    seeds: u32,

    /// Use reflective z boundaries instead of vacuum (gives k_inf
    /// rather than a leakage-dominated k_eff).
    #[arg(long, default_value_t = false)]
    reflective_z: bool,

    /// Open a window showing the assembly XY cross-section before
    /// running transport. Close the window (Esc / X) to start the
    /// eigenvalue calculation. Requires the `preview` feature; build
    /// with `cargo run --release --features preview --bin pwr_assembly`.
    #[arg(long, default_value_t = false)]
    preview: bool,

    /// Lattice size N (N×N pin grid). Default 17 (Westinghouse layout).
    /// For N != 17 the layout is all-fuel (no guide tubes) — useful
    /// for fast 3×3 mini-core stability tests.
    #[arg(long, default_value_t = 17)]
    shape: usize,

    /// Write an HDF5 statepoint at end-of-run (FIRST seed only).
    #[arg(long)]
    statepoint: Option<PathBuf>,

    /// Resume from a previously-written statepoint (FIRST seed only).
    #[arg(long)]
    restart_from: Option<PathBuf>,
}

/// Same nuclide list as pwr_pincell.
const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3), // 0  fuel: 900K
    ("U238.h5", 236.006, 2.49, 3), // 1  fuel: 900K
    ("O16.h5", 15.858, 0.0, 3),    // 2  fuel O16: 900K
    ("H1.h5", 0.999, 0.0, 2),      // 3  water: 600K
    ("Zr90.h5", 89.132, 0.0, 2),   // 4  clad: 600K
    ("Zr91.h5", 90.130, 0.0, 2),   // 5  clad: 600K
    ("Zr92.h5", 91.126, 0.0, 2),   // 6  clad: 600K
    ("Zr94.h5", 93.120, 0.0, 2),   // 7  clad: 600K
    ("O16.h5", 15.858, 0.0, 2),    // 8  water O16: 600K
];

// ── Geometry ────────────────────────────────────────────────────────

const PITCH: f64 = 1.260; // cm, pin-cell pitch
const FUEL_OR: f64 = 0.4096; // cm, fuel outer radius
const CLAD_IR: f64 = 0.4180; // cm, clad inner radius
const CLAD_OR: f64 = 0.4750; // cm, clad outer radius

/// Universe ids assigned by `setup_geometry`.
const U_ROOT: u32 = 0;
const U_PIN: u32 = 1;
const U_GT: u32 = 2;

/// Pin layout: `true` = guide tube / instrument tube, `false` = fuel
/// pin. Returns the standard Westinghouse 17×17 layout when
/// `shape == 17`, otherwise an all-fuel grid (no guide tubes) — the
/// usual choice for non-standard mini-core sizes.
fn pin_layout(shape: usize) -> Vec<Vec<bool>> {
    let mut l = vec![vec![false; shape]; shape];
    if shape == 17 {
        let positions: &[(usize, usize)] = &[
            (2, 5), (2, 8), (2, 11),
            (3, 3), (3, 13),
            (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
            (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
            (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
            (13, 3), (13, 13),
            (14, 5), (14, 8), (14, 11),
        ];
        for &(r, c) in positions {
            l[r][c] = true;
        }
    }
    l
}

/// Build the full assembly Geometry.
///
/// Cell layout (global indices):
///   0 fuel_pin: inside fuel cylinder              → Material(0) UO2
///   1 fuel_gap: outside fuel cyl, inside clad ir  → Void
///   2 fuel_clad: outside clad ir, inside clad or  → Material(1) Zr
///   3 fuel_water: outside clad or                 → Material(2) H2O
///   4 gt_water_in: inside clad ir                 → Material(2) H2O
///   5 gt_clad: outside clad ir, inside clad or    → Material(1) Zr
///   6 gt_water_out: outside clad or               → Material(2) H2O
///   7 root_inside_box: inside the assembly box    → Lattice(0)
///   8 root_outside_box: complement                → Void
fn setup_geometry(reflective_z: bool, shape: usize) -> Geometry {
    let lat_half = (shape as f64) * PITCH / 2.0; // 17 × 1.26 / 2 = 10.71 cm
    let z_half = lat_half;
    let pin_center = PITCH / 2.0; // pin center in element-local coords

    let z_bc = if reflective_z {
        BoundaryCondition::Reflective
    } else {
        BoundaryCondition::Vacuum
    };

    // Surfaces — 0..=2 in element-local frame, 3..=8 in world frame.
    let surfaces = vec![
        // 0: fuel outer cylinder — element-local center (pitch/2, pitch/2)
        Surface::CylinderZ {
            center_x: pin_center,
            center_y: pin_center,
            radius: FUEL_OR,
            bc: BoundaryCondition::Transmission,
        },
        // 1: clad inner cylinder
        Surface::CylinderZ {
            center_x: pin_center,
            center_y: pin_center,
            radius: CLAD_IR,
            bc: BoundaryCondition::Transmission,
        },
        // 2: clad outer cylinder
        Surface::CylinderZ {
            center_x: pin_center,
            center_y: pin_center,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
        // 3: -X box plane (reflective)
        Surface::PlaneX {
            x0: -lat_half,
            bc: BoundaryCondition::Reflective,
        },
        // 4: +X box plane
        Surface::PlaneX {
            x0: lat_half,
            bc: BoundaryCondition::Reflective,
        },
        // 5: -Y box plane
        Surface::PlaneY {
            y0: -lat_half,
            bc: BoundaryCondition::Reflective,
        },
        // 6: +Y box plane
        Surface::PlaneY {
            y0: lat_half,
            bc: BoundaryCondition::Reflective,
        },
        // 7: -Z box plane
        Surface::PlaneZ {
            z0: -z_half,
            bc: z_bc,
        },
        // 8: +Z box plane
        Surface::PlaneZ {
            z0: z_half,
            bc: z_bc,
        },
    ];

    let cells = vec![
        // 0: fuel
        Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_temperature(900.0),
        // 1: fuel-gap (helium → treat as void)
        Cell::new(
            CellId(1),
            cell::between(0, 1), // outside surf 0, inside surf 1
            CellFill::Void,
        ),
        // 2: fuel-clad
        Cell::new(CellId(2), cell::between(1, 2), CellFill::Material(1)).with_temperature(600.0),
        // 3: fuel-water (everything outside the clad in this pin)
        Cell::new(CellId(3), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
        // 4: GT inner water (inside the tube wall)
        Cell::new(CellId(4), cell::inside(1), CellFill::Material(2)).with_temperature(600.0),
        // 5: GT clad
        Cell::new(CellId(5), cell::between(1, 2), CellFill::Material(1)).with_temperature(600.0),
        // 6: GT outer water
        Cell::new(CellId(6), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
        // 7: root cell — the entire assembly box, filled with the lattice
        Cell::new(
            CellId(7),
            cell::intersect_all(vec![
                cell::outside(3), // x > -lat_half
                cell::inside(4),  // x < +lat_half
                cell::outside(5), // y > -lat_half
                cell::inside(6),  // y < +lat_half
                cell::outside(7), // z > -z_half
                cell::inside(8),  // z < +z_half
            ]),
            CellFill::Lattice(0),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-lat_half, -lat_half, -z_half),
            Vec3::new(lat_half, lat_half, z_half),
        )),
        // 8: outside the assembly box
        Cell::new(
            CellId(8),
            cell::Region::Union(
                Box::new(cell::Region::Union(
                    Box::new(cell::inside(3)),
                    Box::new(cell::outside(4)),
                )),
                Box::new(cell::Region::Union(
                    Box::new(cell::inside(5)),
                    Box::new(cell::outside(6)),
                )),
            ),
            CellFill::Void,
        ),
    ];

    let universes = vec![
        // 0: root
        Universe::new(UniverseId(U_ROOT), vec![7, 8]),
        // 1: pin
        Universe::new(UniverseId(U_PIN), vec![0, 1, 2, 3]),
        // 2: guide tube
        Universe::new(UniverseId(U_GT), vec![4, 5, 6]),
    ];

    // Build the lattice element → universe map (row-major, [ix][iy][iz=0]).
    let layout = pin_layout(shape);
    let mut lattice_universes = Vec::with_capacity(shape * shape);
    for iz in 0..1 {
        let _ = iz;
        for iy in 0..shape {
            for ix in 0..shape {
                let id = if layout[iy][ix] {
                    UniverseId(U_GT)
                } else {
                    UniverseId(U_PIN)
                };
                lattice_universes.push(id);
            }
        }
    }

    let lattices = vec![RectLattice {
        origin: Vec3::new(-lat_half, -lat_half, -z_half),
        // z pitch large but finite — single z element covers the whole
        // box, so distance_to_grid in z reports a far crossing that the
        // box surfaces always preempt.
        pitch: Vec3::new(PITCH, PITCH, 2.0 * z_half),
        shape: [shape, shape, 1],
        universes: lattice_universes,
        material_overrides: None,
    }];

    Geometry::new(surfaces, cells, universes, lattices, UniverseId(U_ROOT))
        .expect("assembly geometry must validate")
}

// ── Preview rendering (gated on the `preview` feature) ─────────────

#[cfg(feature = "preview")]
fn run_preview(geometry: &Geometry, materials: &[Material]) {
    use open_rust_mc::geometry::cell::CellFill;
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use rust_mc_sim::preview::{
        LegendEntry, MaterialPalette, Viewport, auto_color_from_name, show_window,
    };

    // Build the per-material colour palette by name lookup. Falls back
    // to the default index palette for any material we haven't tagged.
    let fallback = MaterialPalette::default();
    let palette = MaterialPalette {
        colors: materials
            .iter()
            .enumerate()
            .map(|(i, m)| {
                auto_color_from_name(&m.name).unwrap_or_else(|| {
                    fallback.colors.get(i).copied().unwrap_or(fallback.void)
                })
            })
            .collect(),
        void: fallback.void,
    };
    let legend: Vec<LegendEntry> = materials
        .iter()
        .enumerate()
        .map(|(i, m)| {
            LegendEntry::new(
                m.name.clone(),
                palette.colors.get(i).copied().unwrap_or(palette.void),
            )
        })
        .collect();

    let lat_half = (SHAPE as f64) * PITCH / 2.0;
    let initial = Viewport::square_centered(lat_half * 1.05, 0.0, 900);

    // Render closure: per pixel, sample world coords through the
    // recursive find_cell, look up the deepest material, look up the
    // colour. Empty stack → void colour.
    let render = move |vp: &Viewport| -> Vec<u32> {
        let w = vp.width as usize;
        let h = vp.height as usize;
        let dx = (vp.x_max - vp.x_min) / vp.width as f64;
        let dy = (vp.y_max - vp.y_min) / vp.height as f64;
        let mut buf = vec![0u32; w * h];
        for py in 0..vp.height {
            let world_y = vp.y_max - (py as f64 + 0.5) * dy;
            for px in 0..vp.width {
                let world_x = vp.x_min + (px as f64 + 0.5) * dx;
                let pos = Vec3::new(world_x, world_y, vp.z_slice);
                let color = match find_cell_recursive(pos, geometry) {
                    Some(stack) => {
                        let deepest_idx =
                            stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                        match geometry.cells[deepest_idx].fill {
                            CellFill::Material(m) => palette
                                .colors
                                .get(m as usize)
                                .copied()
                                .unwrap_or(palette.void),
                            _ => palette.void,
                        }
                    }
                    None => palette.void,
                };
                let [r, g, b] = color;
                buf[(py as usize) * w + (px as usize)] =
                    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        }
        buf
    };

    show_window(
        initial,
        "17×17 PWR assembly — recursive geometry preview",
        legend,
        render,
    );
}

#[cfg(not(feature = "preview"))]
fn run_preview(_geometry: &Geometry, _materials: &[Material]) {
    eprintln!(
        "  --preview requires the `preview` feature. Re-run with:\n\
         cargo run --release --features preview --bin pwr_assembly -- ..."
    );
}

// ── Materials (same as pwr_pincell) ──────────────────────────────────

fn setup_materials() -> Vec<Material> {
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(0.000719, 0); // U-235
    fuel.add_nuclide(0.022482, 1); // U-238
    fuel.add_nuclide(0.046402, 2); // O-16

    let mut clad = Material::new("Zircaloy", 600.0);
    clad.add_nuclide(0.022932, 4);
    clad.add_nuclide(0.004996, 5);
    clad.add_nuclide(0.007636, 6);
    clad.add_nuclide(0.007740, 7);

    let mut water = Material::new("H2O", 600.0);
    water.add_nuclide(0.049486, 3);
    water.add_nuclide(0.024743, 8);

    vec![fuel, clad, water]
}

fn load_thermal(data_dir: &std::path::Path) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    let h2o_thermal: Option<Arc<ThermalScatteringData>> = if h2o_path.exists() {
        match hdf5_reader::load_thermal_scattering(&h2o_path) {
            Ok(tsl) => {
                println!(
                    "  S(α,β): loaded {} ({} temperatures)",
                    tsl.name,
                    tsl.temp_labels.len()
                );
                Some(Arc::new(tsl))
            }
            Err(e) => {
                eprintln!("  WARN: failed to load S(α,β): {e}");
                None
            }
        }
    } else {
        println!("  S(α,β): c_H_in_H2O.h5 not found — using free-gas for H");
        None
    };
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    if let Some(ref tsl) = h2o_thermal {
        thermal[3] = Some(Arc::clone(tsl));
    }
    thermal
}

fn load_svd(args: &Args) -> (xs_provider::SvdXsProvider, usize, f64) {
    println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
    let t0 = Instant::now();

    let mut kernels = Vec::new();
    for (_nuc_idx, &(filename, awr, nu_bar, nuc_temp_idx)) in NUCLIDE_SPECS.iter().enumerate() {
        let path = args.data_dir.join(filename);
        if !path.exists() {
            eprintln!("  WARN: {} not found — using zero kernel", path.display());
            kernels.push(xs_provider::NuclideKernels {
                elastic: None,
                total_table: None,
                total_xs_raw: None,
                missing_xs: None,
                pointwise_xs: None,
                inelastic: None,
                n2n: None,
                n3n: None,
                fission: None,
                capture: None,
                awr,
                nu_bar_const: nu_bar,
                nu_bar_table: None,
                delayed_nu_bar_table: None,
                discrete_levels: vec![],
                inelastic_cdf: None,
                discrete_level_angles: vec![],
                has_continuum_inelastic: false,
                elastic_angle: None,
                fission_energy_dist: None,
                inelastic_continuum_edist: None,
                n2n_edist: None,
                n3n_edist: None,
                urr_tables: None,
                photon_products: Vec::new(),
            });
        } else {
            kernels.push(xs_provider::load_nuclide(
                &path,
                args.rank,
                nuc_temp_idx,
                awr,
                nu_bar,
            ));
        }
    }

    let thermal = load_thermal(&args.data_dir);
    let xs_mem: usize = kernels.iter().map(|k| k.svd_memory_bytes()).sum();
    let provider = xs_provider::SvdXsProvider {
        nuclides: kernels,
        thermal,
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded in {load_ms:.0} ms  |  XS memory: {:.1} KB ({} nuclides)",
        xs_mem as f64 / 1024.0,
        NUCLIDE_SPECS.len()
    );
    (provider, xs_mem, load_ms)
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    println!("========================================================");
    println!("  17×17 PWR Fuel Assembly — Recursive Geometry Demo");
    println!("========================================================");
    let shape = args.shape;
    println!(
        "  Lattice  : {shape}×{shape} pins (pitch {PITCH:.3} cm)"
    );
    let layout = pin_layout(shape);
    let n_gt = layout.iter().flatten().filter(|&&b| b).count();
    let n_pin = shape * shape - n_gt;
    println!("  Layout   : {n_pin} fuel pins + {n_gt} guide tubes");
    println!(
        "  Bounds   : x,y reflective at ±{:.3} cm; z {} at ±{:.3} cm",
        (shape as f64) * PITCH / 2.0,
        if args.reflective_z {
            "reflective"
        } else {
            "vacuum"
        },
        (shape as f64) * PITCH / 2.0
    );
    println!(
        "  Stack    : depth 3 at every transport step (root → lattice → pin/GT)"
    );

    let geometry = setup_geometry(args.reflective_z, shape);
    let materials = setup_materials();

    println!(
        "  Geometry : {} surfaces, {} cells, {} universes, {} lattice",
        geometry.surfaces.len(),
        geometry.cells.len(),
        geometry.universes.len(),
        geometry.lattices.len()
    );

    if args.preview {
        println!(
            "\n  Opening preview window — close it (Esc / X) to start transport."
        );
        run_preview(&geometry, &materials);
        // Preview-only mode: exit cleanly without running transport.
        return;
    }

    let (xs_provider, xs_mem, load_ms) = load_svd(&args);
    let _ = (xs_mem, load_ms);

    // Optional restart bank from a previous statepoint.
    let restart_bank = args.restart_from.as_ref().map(|path| {
        let bank = open_rust_mc::transport::statepoint::read_source_bank(path)
            .unwrap_or_else(|e| panic!("failed to load restart bank from {path:?}: {e}"));
        println!("  Resuming from {} ({} sites)", path.display(), bank.len());
        bank
    });

    let inactive = args.inactive.min(args.batches.saturating_sub(1));
    let total_histories = (args.batches - inactive) as u64 * args.particles as u64;

    let mut k_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
    let mut std_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
    let mut t_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);

    for seed in 0..args.seeds {
        let config = SimConfig {
            batches: args.batches,
            inactive,
            particles_per_batch: args.particles,
            seed: seed as u64,
            auto_inactive: None,
            verbose: true,
            parallel: true,
            tallies: Default::default(),
            statepoint_path: if seed == 0 { args.statepoint.clone() } else { None },
            survival_biasing: None,
            initial_source_bank: if seed == 0 {
                restart_bank.clone()
            } else {
                None
            },
            weight_window: None,
        };

        if args.seeds > 1 {
            print!("  Seed {seed}: ");
            let _ = std::io::stdout().flush();
        } else {
            println!();
        }

        let t1 = Instant::now();
        let (results, _) =
            simulate::run_eigenvalue_with_geometry(&config, &geometry, &materials, &xs_provider);
        let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let active: Vec<f64> = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.k_eff)
            .collect();
        let n_active = active.len() as f64;
        let k_mean = active.iter().sum::<f64>() / n_active;
        let k_var =
            active.iter().map(|&k| (k - k_mean).powi(2)).sum::<f64>() / (n_active * (n_active - 1.0));
        let k_std = k_var.sqrt();

        if args.seeds > 1 {
            println!(
                "k={k_mean:.5} +/- {k_std:.5}  {sim_ms:.0}ms  ({:.1} ns/particle)",
                sim_ms * 1e6 / total_histories as f64
            );
        }

        k_per_seed.push(k_mean);
        std_per_seed.push(k_std);
        t_per_seed.push(sim_ms);
    }

    let n = k_per_seed.len() as f64;
    let k_mean = k_per_seed.iter().sum::<f64>() / n;
    let k_std = if k_per_seed.len() > 1 {
        // Multi-seed: cross-seed stderr.
        let var = k_per_seed
            .iter()
            .map(|&k| (k - k_mean).powi(2))
            .sum::<f64>()
            / (n * (n - 1.0));
        var.sqrt()
    } else {
        // Single seed: use the within-seed batch-to-batch stderr.
        std_per_seed.first().copied().unwrap_or(0.0)
    };
    let total_ms: f64 = t_per_seed.iter().sum();
    let total_ns_per_particle =
        total_ms * 1e6 / (args.seeds as u64 * total_histories) as f64;

    println!();
    println!("============================================================");
    println!("  RESULT — {shape}×{shape} PWR Assembly (SVD rank={})", args.rank);
    println!("============================================================");
    println!(
        "  k{}            = {:.5} +/- {:.5}",
        if args.reflective_z { "_inf" } else { "_eff" },
        k_mean,
        k_std
    );
    println!("  Histories      = {} per seed × {} seeds", total_histories, args.seeds);
    println!("  Total sim time = {:.0} ms", total_ms);
    println!("  ns/particle    = {:.1} (averaged over all seeds)", total_ns_per_particle);
}
