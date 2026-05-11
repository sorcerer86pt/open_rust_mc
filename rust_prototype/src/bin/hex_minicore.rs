//! Hex-lattice mini-core demo — first eigenvalue exercise of
//! `HexLattice` transport.
//!
//! Builds an N-ring hex array of identical UO₂ pins (1-ring = 7 pins,
//! 2-ring = 19 pins) inside a reflective box. Same nuclide / material
//! choices as pwr_pincell so the resulting k_inf is directly
//! comparable to the rectangular pin-cell baseline.
//!
//! Stack depth at every transport step is 3:
//!   root → hex_lattice → pin cell.
//!
//! Usage:
//!   hex_minicore <data_dir> [--rings N] [--rank K] [--batches N]
//!                           [--particles N] [--seeds S]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::lattice::{HexLattice, HexOrientation};
use open_rust_mc::geometry::shapes;
use open_rust_mc::geometry::surface::BoundaryCondition;
use open_rust_mc::geometry::universe::{Universe, UniverseId};
use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
use open_rust_mc::hdf5_reader;
use open_rust_mc::thermal::ThermalScatteringData;
use open_rust_mc::transport::dispatch::{CpuRunner, EigenvalueRunner};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::simulate::SimConfig;
use open_rust_mc::transport::xs_provider;

#[derive(Parser, Debug)]
#[command(
    name = "hex_minicore",
    about = "Hex-lattice fuel mini-core k-inf demo (recursive geometry)"
)]
struct Args {
    data_dir: PathBuf,

    /// Number of hex rings around the centre pin. 1 → 7 pins, 2 → 19 pins.
    #[arg(long, default_value_t = 1)]
    rings: usize,

    #[arg(short, long, default_value_t = 5)]
    rank: usize,

    #[arg(short, long, default_value_t = 50)]
    batches: u32,

    #[arg(short, long, default_value_t = 15)]
    inactive: u32,

    #[arg(short, long, default_value_t = 5000)]
    particles: u32,

    #[arg(short, long, default_value_t = 1)]
    seeds: u32,

    /// Use reflective z boundaries (k_inf) rather than vacuum.
    #[arg(long, default_value_t = true)]
    reflective_z: bool,
}

const NUCLIDE_SPECS: &[(&str, f64, f64, usize)] = &[
    ("U235.h5", 233.025, 2.43, 3),
    ("U238.h5", 236.006, 2.49, 3),
    ("O16.h5", 15.858, 0.0, 3),
    ("H1.h5", 0.999, 0.0, 2),
    ("Zr90.h5", 89.132, 0.0, 2),
    ("Zr91.h5", 90.130, 0.0, 2),
    ("Zr92.h5", 91.126, 0.0, 2),
    ("Zr94.h5", 93.120, 0.0, 2),
    ("O16.h5", 15.858, 0.0, 2),
];

const PITCH: f64 = 1.260;
const FUEL_OR: f64 = 0.4096;
const CLAD_IR: f64 = 0.4180;
const CLAD_OR: f64 = 0.4750;

const U_ROOT: u32 = 0;
const U_PIN: u32 = 1;
/// Universe used for the hex-lattice corner placeholders — slots
/// outside the ring radius. A single all-water cell so particles
/// that wander into the corner of the rectangular outer box just
/// stream through moderator until they hit a reflective wall.
const U_OUTSIDE_HEX: u32 = 2;

fn setup_materials() -> Vec<Material> {
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

fn load_thermal(data_dir: &std::path::Path) -> Vec<Option<Arc<ThermalScatteringData>>> {
    let h2o_path = data_dir.join("c_H_in_H2O.h5");
    let h2o_thermal = if h2o_path.exists() {
        match hdf5_reader::load_thermal_scattering(&h2o_path) {
            Ok(tsl) => Some(Arc::new(tsl)),
            Err(e) => {
                eprintln!("  WARN: failed to load S(α,β): {e}");
                None
            }
        }
    } else {
        None
    };
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; NUCLIDE_SPECS.len()];
    if let Some(ref tsl) = h2o_thermal {
        thermal[3] = Some(Arc::clone(tsl));
    }
    thermal
}

fn load_svd(args: &Args) -> (xs_provider::SvdXsProvider, f64) {
    println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
    let t0 = Instant::now();

    let mut kernels = Vec::new();
    for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS.iter() {
        let path = args.data_dir.join(filename);
        if !path.exists() {
            eprintln!("  WARN: {} not found — using zero kernel", path.display());
            kernels.push(xs_provider::NuclideKernels::empty(awr, nu_bar));
            continue;
        }
        kernels.push(xs_provider::load_nuclide(
            &path,
            args.rank,
            nuc_temp_idx,
            awr,
            nu_bar,
        ));
    }
    let thermal = load_thermal(&args.data_dir);
    let provider = xs_provider::SvdXsProvider {
        nuclides: kernels,
        thermal,
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Loaded {} nuclides in {load_ms:.0} ms",
        NUCLIDE_SPECS.len()
    );
    (provider, load_ms)
}

fn setup_geometry(rings: usize, reflective_z: bool) -> Geometry {
    // Hex outer boundary: 6 reflective planes at the edge-midpoint
    // directions of a flat-top outer hex, perpendicular distance
    // `inradius = (rings + 0.5) * pitch`. The lattice itself is
    // sized one ring larger so that float-precision points right
    // at the boundary (which can cube-round to ring N+1) still
    // resolve cleanly through the lattice's outer placeholder ring.
    let inradius = (rings as f64 + 0.5) * PITCH;
    let circumradius = inradius * 2.0 / 3.0_f64.sqrt();
    let z_half = inradius.max(1.0);
    let z_bc = if reflective_z {
        BoundaryCondition::Reflective
    } else {
        BoundaryCondition::Vacuum
    };

    // Cylinders for the pin (surfaces 0, 1, 2 at element-local origin)
    // + hex outer boundary (surfaces 3..=10 = 6 hex sides + 2 z planes).
    let mut surfaces = shapes::pin_cylinders(0.0, 0.0, &[FUEL_OR, CLAD_IR, CLAD_OR]);
    let outer = shapes::hex_boundary(
        rings,
        PITCH,
        HexOrientation::Y,
        BoundaryCondition::Reflective,
        z_half,
        z_bc,
        surfaces.len(),
    );
    surfaces.extend(outer.surfaces);

    let cells = vec![
        // 0: fuel
        Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_temperature(900.0),
        // 1: gap (helium → void)
        Cell::new(CellId(1), cell::between(0, 1), CellFill::Void),
        // 2: clad
        Cell::new(CellId(2), cell::between(1, 2), CellFill::Material(1)).with_temperature(600.0),
        // 3: water (outside clad)
        Cell::new(CellId(3), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
        // 4: root inside hex → HexLattice(0)
        Cell::new(CellId(4), outer.inside.clone(), CellFill::HexLattice(0)).with_aabb(Aabb::new(
            Vec3::new(-circumradius, -inradius, -z_half),
            Vec3::new(circumradius, inradius, z_half),
        )),
        // 5: outside the hex (complement of cell 4's region — kept
        // for partition completeness).
        Cell::new(
            CellId(5),
            cell::Region::Complement(Box::new(outer.inside)),
            CellFill::Void,
        ),
        // 6: all-water cell for the U_OUTSIDE_HEX universe.
        Cell::new(
            CellId(6),
            cell::Region::Union(Box::new(cell::inside(0)), Box::new(cell::outside(0))),
            CellFill::Material(2),
        )
        .with_temperature(600.0),
    ];

    let universes = vec![
        Universe::new(UniverseId(U_ROOT), vec![4, 5]),
        Universe::new(UniverseId(U_PIN), vec![0, 1, 2, 3]),
        Universe::new(UniverseId(U_OUTSIDE_HEX), vec![6]),
    ];

    // The hex outer-boundary inradius is `(rings + 0.5) * pitch`.
    // Float-precision points at that edge can round to ring index
    // `rings + 1` in the cube-coord cube-rounding step, so the
    // lattice itself is built one ring larger and the outer ring
    // filled with the all-water placeholder universe. Without this
    // padding, edge points map to a ring outside the lattice's
    // n_rings and find_element returns None, leaking the particle.
    let visible = rings;
    let n = visible + 1;
    let stride = 2 * n + 1;
    let mut hex_universes = vec![UniverseId(U_OUTSIDE_HEX); stride * stride];
    let mut n_pins = 0;
    for q in -(n as i32)..=(n as i32) {
        for r in -(n as i32)..=(n as i32) {
            let cube_s = -q - r;
            let ring = q
                .unsigned_abs()
                .max(r.unsigned_abs())
                .max(cube_s.unsigned_abs()) as usize;
            let qi = (q + n as i32) as usize;
            let ri = (r + n as i32) as usize;
            if ring <= visible {
                hex_universes[ri * stride + qi] = UniverseId(U_PIN);
                n_pins += 1;
            }
            // Else: stays U_OUTSIDE_HEX (placeholder ring beyond the
            // visible region; lives outside the hex outer boundary so
            // is never queried in practice — just here to keep
            // find_element from failing on edge-rounding).
        }
    }
    println!(
        "  Hex lattice: {visible} visible ring{} → {n_pins} fuel pins (lattice n_rings = {n} for edge-rounding)",
        if visible == 1 { "" } else { "s" }
    );

    let hex = HexLattice {
        center: Vec3::new(0.0, 0.0, 0.0),
        pitch_xy: PITCH,
        pitch_z: 2.0 * z_half,
        n_rings: n,
        n_axial: 1,
        orientation: HexOrientation::Y,
        universes: hex_universes,
        material_overrides: None,
    };

    Geometry::new(surfaces, cells, universes, vec![], UniverseId(U_ROOT))
        .expect("hex minicore geometry")
        .with_hex_lattices(vec![hex])
        .expect("hex lattices validated")
}

fn main() {
    let args = Args::parse();

    println!("========================================================");
    println!("  Hex-lattice mini-core (HexLattice recursive geometry)");
    println!("========================================================");
    println!(
        "  Rings    : {}   pitch {:.3} cm   reflective_z = {}",
        args.rings, PITCH, args.reflective_z
    );

    let geometry = setup_geometry(args.rings, args.reflective_z);
    let materials = setup_materials();
    println!(
        "  Geometry : {} surfaces, {} cells, {} universes, {} hex lattices",
        geometry.surfaces.len(),
        geometry.cells.len(),
        geometry.universes.len(),
        geometry.hex_lattices.len()
    );

    let (xs_provider, load_ms) = load_svd(&args);
    let _ = load_ms;

    let inactive = args.inactive.min(args.batches.saturating_sub(1));
    let total_histories = (args.batches - inactive) as u64 * args.particles as u64;

    let mut k_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
    let mut k_track_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
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
            statepoint_path: None,
            survival_biasing: None,
            initial_source_bank: None,
            weight_window: None,
            disable_delayed_neutrons: false,
            urr_equivalence: None,
        };

        if args.seeds > 1 {
            print!("  Seed {seed}: ");
            let _ = std::io::stdout().flush();
        }

        let runner = CpuRunner {
            geometry: &geometry,
            materials: &materials,
            xs_provider: &xs_provider,
        };
        let t1 = Instant::now();
        let outcome = runner.run(&config);
        let results = outcome.batches;
        let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let active: Vec<f64> = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.k_eff)
            .collect();
        let active_track: Vec<f64> = results
            .iter()
            .filter(|r| r.active)
            .map(|r| r.k_track)
            .collect();
        let n_active = active.len() as f64;
        let k_mean = active.iter().sum::<f64>() / n_active;
        let k_track_mean = active_track.iter().sum::<f64>() / n_active;
        k_per_seed.push(k_mean);
        k_track_per_seed.push(k_track_mean);
        t_per_seed.push(sim_ms);

        if args.seeds > 1 {
            println!("k_inf={k_mean:.5}  k_track={k_track_mean:.5}  {sim_ms:.0}ms");
        }
    }

    let n_seeds = k_per_seed.len() as f64;
    let k_mean = k_per_seed.iter().sum::<f64>() / n_seeds;
    let k_var =
        k_per_seed.iter().map(|k| (k - k_mean).powi(2)).sum::<f64>() / (n_seeds - 1.0).max(1.0);
    let k_std = k_var.sqrt();

    let kt_mean = k_track_per_seed.iter().sum::<f64>() / n_seeds;
    let kt_var = k_track_per_seed
        .iter()
        .map(|k| (k - kt_mean).powi(2))
        .sum::<f64>()
        / (n_seeds - 1.0).max(1.0);
    let kt_std = kt_var.sqrt();

    let total_t_ms: f64 = t_per_seed.iter().sum();
    let ns_per_p = total_t_ms * 1e6 / (total_histories * args.seeds as u64) as f64;

    println!("\n========================================================");
    println!(
        "  RESULT — hex mini-core ({} ring{}, SVD rank={})",
        args.rings,
        if args.rings == 1 { "" } else { "s" },
        args.rank
    );
    println!("========================================================");
    println!("  k_inf (collision)  = {k_mean:.5} +/- {k_std:.5}");
    println!("  k_inf (track-len)  = {kt_mean:.5} +/- {kt_std:.5}");
    println!(
        "  Histories  = {} per seed × {} seeds",
        total_histories, args.seeds
    );
    println!("  Total sim time = {total_t_ms:.0} ms");
    println!("  ns/particle    = {ns_per_p:.1}");
}
