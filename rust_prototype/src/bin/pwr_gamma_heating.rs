//! PWR pin cell gamma-heating estimate — coupled neutron + photon
//! transport on a shared CSG geometry.
//!
//! # Pipeline
//!
//!  1. Runs a neutron k-eigenvalue simulation on the standard PWR pin
//!     cell (UO₂ fuel + Zr clad + H₂O, 1.26 cm pitch, reflective
//!     lattice) with the same `TableXsProvider` path that
//!     `pwr_pincell` uses. Captures are tallied per cell via the new
//!     `BatchResult::captures_by_cell` field.
//!  2. Aggregates active-batch captures into a per-cell probability
//!     distribution `P(cell | capture)` — the real spatial
//!     distribution of `(n,γ)` events in the pin.
//!  3. Runs the photon driver (`transport_history_csg`) on the same
//!     CSG, with a per-cell `PhotonMaterial`. Source positions are
//!     sampled per-cell proportional to `P`, uniformly inside each
//!     cell's AABB with a cell-membership reject test. Source
//!     energies come from a two-line notional capture-γ spectrum
//!     (70 % × 1 MeV soft + 30 % × 5 MeV hard).
//!  4. Per-collision deposits are binned by containing cell; the
//!     binary prints the fraction of source energy landing in
//!     fuel / gap / clad / water.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin pwr_gamma_heating -- \
//!     data/endfb-vii.1-hdf5/neutron \
//!     --photon-data data/endfb-vii.1-hdf5/photon \
//!     --n-neutron-batches 40 --n-neutron-inactive 10 \
//!     --n-neutron-particles 5000 \
//!     --n-photon 50000
//! ```
//!
//! # Caveats
//!
//! - Photon source energy is a two-line notional spectrum, not the
//!   per-nuclide cascade HDF5 spectra. Mean energy (2.2 MeV) is
//!   realistic; shape is simplified.
//! - No photon production from fission γs or inelastic scattering γs
//!   (these add another ~4 % of reactor power, mostly to fuel).
//! - Capture-only photon source: other absorptions ((n,α), (n,p))
//!   produce different secondary distributions not modelled.
//!
//! The capture *spatial* distribution is now real data, not a uniform
//! stub. That is the key coupling this binary demonstrates.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::{Aabb, Vec3, ray};
use open_rust_mc::hdf5_reader;
use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::material::PhotonMaterial;
use open_rust_mc::photon::transport::transport_history_csg;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::rng::Rng;
use open_rust_mc::transport::simulate::{self, SimConfig};
use open_rust_mc::transport::xs_provider::{self, TableXsProvider};

// ── Pin cell dimensions (same as bin/pwr_pincell.rs) ────────────────
const FUEL_OR: f64 = 0.4096;
const CLAD_IR: f64 = 0.4180;
const CLAD_OR: f64 = 0.4750;
const PITCH: f64 = 1.2600;

// ── Photon material atom densities (per-element, photon XS) ─────────
const UO2_MOL_DENSITY: f64 = 2.319e-2;
const ZR_ATOM_DENSITY: f64 = 4.324e-2;
const H2O_MOL_DENSITY: f64 = 2.474e-2;

// ── Notional capture-γ source (70 % 1 MeV + 30 % 5 MeV) ─────────────
const SOURCE_E_SOFT_EV: f64 = 1.0e6;
const SOURCE_E_HARD_EV: f64 = 5.0e6;
const SOURCE_HARD_FRACTION: f64 = 0.30;

// ── Nuclide specs matching pwr_pincell's NUCLIDE_SPECS ──────────────
// (filename, AWR, fallback nu-bar, temp_idx after sort).
const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3), // 0  fuel  900K
    ("U238.h5", 236.006, 2.49, 3), // 1  fuel  900K
    ("O16.h5", 15.858, 0.0, 3),    // 2  fuel  900K
    ("H1.h5", 0.999, 0.0, 2),      // 3  water 600K
    ("Zr90.h5", 89.132, 0.0, 2),   // 4  clad  600K
    ("Zr91.h5", 90.130, 0.0, 2),   // 5  clad  600K
    ("Zr92.h5", 91.126, 0.0, 2),   // 6  clad  600K
    ("Zr94.h5", 93.120, 0.0, 2),   // 7  clad  600K
    ("O16.h5", 15.858, 0.0, 2),    // 8  water 600K
];

struct Args {
    neutron_data: PathBuf,
    photon_data: PathBuf,
    n_neutron_batches: u32,
    n_neutron_inactive: u32,
    n_neutron_particles: u32,
    n_photon: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        neutron_data: PathBuf::new(),
        photon_data: PathBuf::new(),
        n_neutron_batches: 40,
        n_neutron_inactive: 10,
        n_neutron_particles: 5_000,
        n_photon: 50_000,
    };
    let mut it = std::env::args().skip(1);
    let Some(neutron_data) = it.next() else {
        return Err(
            "usage: pwr_gamma_heating <neutron_data_dir> --photon-data <photon_dir> \
             [--n-neutron-batches N] [--n-neutron-inactive N] \
             [--n-neutron-particles N] [--n-photon N]"
                .to_string(),
        );
    };
    args.neutron_data = PathBuf::from(neutron_data);

    while let Some(a) = it.next() {
        match a.as_str() {
            "--photon-data" => {
                args.photon_data =
                    PathBuf::from(it.next().ok_or("missing value for --photon-data")?);
            }
            "--n-neutron-batches" => {
                args.n_neutron_batches = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-neutron-inactive" => {
                args.n_neutron_inactive = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-neutron-particles" => {
                args.n_neutron_particles = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--n-photon" => {
                args.n_photon = it
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    if args.photon_data.as_os_str().is_empty() {
        return Err("--photon-data is required".to_string());
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(2);
        }
    };

    let (surfaces, cells) = setup_geometry();
    let materials_n = setup_neutron_materials();

    // ── Phase 1: neutron transport ─────────────────────────────────
    println!("── Phase 1: neutron (n,γ) capture tally ──");
    let t0 = Instant::now();
    let xs = match load_table_xs(&args.neutron_data) {
        Ok(p) => p,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(1);
        }
    };
    println!("  XS load: {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);

    let config = SimConfig {
        batches: args.n_neutron_batches,
        inactive: args.n_neutron_inactive,
        particles_per_batch: args.n_neutron_particles,
        seed: 1,
        auto_inactive: None,
    };
    let t1 = Instant::now();
    let (batch_results, k_eff) =
        simulate::run_eigenvalue(&config, &surfaces, &cells, &materials_n, &xs);
    let neutron_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Aggregate captures per cell across ACTIVE batches only (source
    // converged, fission-rate-normalised estimator).
    let mut captures_per_cell = vec![0.0_f64; cells.len()];
    let mut active_batches = 0_usize;
    for br in &batch_results {
        if br.active {
            active_batches += 1;
            for (i, c) in br.captures_by_cell.iter().enumerate() {
                captures_per_cell[i] += *c;
            }
        }
    }
    let total_captures: f64 = captures_per_cell.iter().sum();
    println!(
        "  Neutron: k = {:.5}, {} active batches, {} captures, {:.1} s",
        k_eff,
        active_batches,
        total_captures as u64,
        neutron_ms / 1000.0
    );
    println!("  Captures by cell (normalised):");
    for (i, c) in captures_per_cell.iter().enumerate() {
        let label = cell_label(i);
        let frac = if total_captures > 0.0 {
            c / total_captures
        } else {
            0.0
        };
        println!(
            "    cell {i} [{label:<5}]: {:>8.0} captures  ({:>6.3} %)",
            c,
            100.0 * frac
        );
    }

    if total_captures == 0.0 {
        eprintln!("No captures recorded — cannot seed photon source.");
        return ExitCode::from(1);
    }

    // ── Phase 2: photon source + transport ─────────────────────────
    println!("\n── Phase 2: photon transport from (n,γ) source ──");
    let materials_p = match load_photon_materials(&args.photon_data) {
        Ok(m) => m,
        Err(m) => {
            eprintln!("{m}");
            return ExitCode::from(1);
        }
    };

    // Build cumulative probability per cell for roulette sampling.
    let mut cdf = Vec::with_capacity(cells.len());
    let mut cum = 0.0;
    for c in &captures_per_cell {
        cum += c / total_captures;
        cdf.push(cum);
    }

    let t2 = Instant::now();
    let mut deposited_per_cell = vec![0.0_f64; cells.len()];
    let mut escaped_energy = 0.0_f64;
    let mut orphan_deposit = 0.0_f64;
    let mut total_source_energy = 0.0_f64;

    for i in 0..args.n_photon {
        let mut rng = Rng::new(0xB0F1_0000 + i as u64, 1);

        // Pick source cell by capture-weighted CDF.
        let xi = rng.uniform();
        let cell_src = cdf.iter().position(|c| xi < *c).unwrap_or(cells.len() - 1);

        // Sample a position uniformly inside that cell (reject in AABB).
        let Some(pos) = sample_in_cell(cell_src, &surfaces, &cells, &mut rng) else {
            // Fallback: skip this history if the AABB is degenerate.
            continue;
        };

        let (dx, dy, dz) = rng.isotropic_direction();
        let e_src = if rng.uniform() < SOURCE_HARD_FRACTION {
            SOURCE_E_HARD_EV
        } else {
            SOURCE_E_SOFT_EV
        };
        total_source_energy += e_src;

        let r = transport_history_csg(
            pos,
            Vec3::new(dx, dy, dz),
            e_src,
            &surfaces,
            &cells,
            &materials_p,
            1_000.0, // 1 keV cutoff
            &mut rng,
        );

        escaped_energy += r.energy_escaped;
        for (p, e) in &r.deposits {
            if let Some(idx) = ray::find_cell(*p, &surfaces, &cells) {
                deposited_per_cell[idx] += e;
            } else {
                orphan_deposit += e;
            }
        }
    }
    let photon_s = t2.elapsed().as_secs_f64();

    println!(
        "  Photon: {} histories, {:.2} s, {:.0} hist/s, total source {:.3e} eV",
        args.n_photon,
        photon_s,
        args.n_photon as f64 / photon_s,
        total_source_energy
    );

    // ── Final report ───────────────────────────────────────────────
    println!("\n── Energy deposition by region ──");
    println!(
        "  {:<6} {:>16} {:>10}",
        "region", "deposited (eV)", "fraction"
    );
    let mut total_dep = 0.0;
    for (i, e) in deposited_per_cell.iter().enumerate() {
        total_dep += e;
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            cell_label(i),
            e,
            100.0 * e / total_source_energy
        );
    }
    if orphan_deposit > 0.0 {
        println!(
            "  {:<6} {:>16.3e} {:>9.3} %",
            "orphan",
            orphan_deposit,
            100.0 * orphan_deposit / total_source_energy
        );
    }
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "escape",
        escaped_energy,
        100.0 * escaped_energy / total_source_energy
    );
    let sum = total_dep + orphan_deposit + escaped_energy;
    println!(
        "  {:<6} {:>16.3e} {:>9.3} %",
        "sum",
        sum,
        100.0 * sum / total_source_energy
    );

    if escaped_energy / total_source_energy > 1.0e-3 {
        eprintln!(
            "WARNING: reflective lattice leaked {:.2} % of source energy",
            100.0 * escaped_energy / total_source_energy
        );
    }

    ExitCode::SUCCESS
}

// ── Geometry (same as pwr_pincell) ──────────────────────────────────

fn cell_label(idx: usize) -> &'static str {
    match idx {
        0 => "fuel",
        1 => "gap",
        2 => "clad",
        3 => "water",
        _ => "?",
    }
}

fn setup_geometry() -> (Vec<Surface>, Vec<Cell>) {
    let half = PITCH / 2.0;
    let z_half = half;
    let surfaces = vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: FUEL_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_IR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: -half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneX {
            x0: half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: -half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: -z_half,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: z_half,
            bc: BoundaryCondition::Reflective,
        },
    ];
    let box_aabb = Aabb::new(
        Vec3::new(-half, -half, -z_half),
        Vec3::new(half, half, z_half),
    );
    let cells = vec![
        Cell::new(
            CellId(0),
            cell::intersect_all(vec![cell::inside(0), cell::outside(7), cell::inside(8)]),
            CellFill::Material(0),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-FUEL_OR, -FUEL_OR, -z_half),
            Vec3::new(FUEL_OR, FUEL_OR, z_half),
        ))
        .with_temperature(900.0),
        Cell::new(
            CellId(1),
            cell::intersect_all(vec![
                cell::outside(0),
                cell::inside(1),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Void,
        )
        .with_aabb(Aabb::new(
            Vec3::new(-CLAD_IR, -CLAD_IR, -z_half),
            Vec3::new(CLAD_IR, CLAD_IR, z_half),
        )),
        Cell::new(
            CellId(2),
            cell::intersect_all(vec![
                cell::outside(1),
                cell::inside(2),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Material(1),
        )
        .with_aabb(Aabb::new(
            Vec3::new(-CLAD_OR, -CLAD_OR, -z_half),
            Vec3::new(CLAD_OR, CLAD_OR, z_half),
        ))
        .with_temperature(600.0),
        Cell::new(
            CellId(3),
            cell::intersect_all(vec![
                cell::outside(2),
                cell::outside(3),
                cell::inside(4),
                cell::outside(5),
                cell::inside(6),
                cell::outside(7),
                cell::inside(8),
            ]),
            CellFill::Material(2),
        )
        .with_aabb(box_aabb)
        .with_temperature(600.0),
    ];
    (surfaces, cells)
}

fn setup_neutron_materials() -> Vec<Material> {
    let mut fuel = Material::new("UO2", 900.0);
    fuel.add_nuclide(0.000719, 0);
    fuel.add_nuclide(0.022482, 1);
    fuel.add_nuclide(0.046402, 2);

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

fn load_table_xs(data_dir: &Path) -> Result<TableXsProvider, String> {
    let mut tables = Vec::with_capacity(NUCLIDE_SPECS.len());
    for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS {
        let path = data_dir.join(filename);
        if !path.exists() {
            return Err(format!("missing neutron file: {}", path.display()));
        }
        tables.push(xs_provider::load_nuclide_table(
            &path,
            nuc_temp_idx,
            awr,
            nu_bar,
        ));
    }
    let thermal = load_thermal(data_dir);
    Ok(TableXsProvider {
        nuclides: tables,
        thermal,
    })
}

fn load_thermal(data_dir: &Path) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    if h2o_path.exists()
        && let Ok(tsl) = hdf5_reader::load_thermal_scattering(&h2o_path)
    {
        thermal[3] = Some(Arc::new(tsl));
    }
    thermal
}

fn load_photon_materials(data_dir: &Path) -> Result<Vec<Option<PhotonMaterial>>, String> {
    let load = |name: &str| -> Result<PhotonElement, String> {
        PhotonElement::from_hdf5(&data_dir.join(name))
            .map_err(|e| format!("failed to load {name}: {e}"))
    };
    let h = load("H.h5")?;
    let o1 = load("O.h5")?;
    let o2 = load("O.h5")?;
    let zr = load("Zr.h5")?;
    let u = load("U.h5")?;

    let uo2 = PhotonMaterial::new(vec![(UO2_MOL_DENSITY, u), (2.0 * UO2_MOL_DENSITY, o1)]);
    let clad = PhotonMaterial::mono(ZR_ATOM_DENSITY, zr);
    let h2o = PhotonMaterial::new(vec![(2.0 * H2O_MOL_DENSITY, h), (H2O_MOL_DENSITY, o2)]);

    Ok(vec![Some(uo2), Some(clad), Some(h2o)])
}

/// Sample a point uniformly inside the given cell via AABB rejection.
/// Returns `None` if the cell has infinite or inside-out AABB.
fn sample_in_cell(
    cell_idx: usize,
    surfaces: &[Surface],
    cells: &[Cell],
    rng: &mut Rng,
) -> Option<Vec3> {
    let aabb = &cells[cell_idx].aabb;
    let lx = aabb.max.x - aabb.min.x;
    let ly = aabb.max.y - aabb.min.y;
    let lz = aabb.max.z - aabb.min.z;
    if !(lx.is_finite() && ly.is_finite() && lz.is_finite()) || lx <= 0.0 || ly <= 0.0 || lz <= 0.0
    {
        return None;
    }
    for _ in 0..200 {
        let p = Vec3::new(
            aabb.min.x + rng.uniform() * lx,
            aabb.min.y + rng.uniform() * ly,
            aabb.min.z + rng.uniform() * lz,
        );
        if ray::find_cell(p, surfaces, cells) == Some(cell_idx) {
            return Some(p);
        }
    }
    None
}

#[allow(dead_code)]
fn _flush() {
    let _ = std::io::stdout().flush();
}
