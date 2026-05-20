//! Hex-lattice mini-core eigenvalue on GPU.
//!
//! Mirrors `hex_minicore` (same nuclides, same materials, same hex
//! geometry) but drives `transport_recursive_persistent` via the
//! dispatch `CudaRunner`. The k_inf result is directly comparable to
//! the CPU `hex_minicore` baseline — agreement within MC noise is
//! the validation that GPU hex transport works end-to-end.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_hex_minicore -- \
//!     <data_dir> [--rings N] [--rank K] [--batches N]
//!                [--inactive N] [--particles N] [--seeds S]
//!                [--max-events N]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: this binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_hex_minicore -- ...");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    cuda_main::run();
}

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;

    use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
    use open_rust_mc::geometry::lattice::{HexLattice, HexOrientation};
    use open_rust_mc::geometry::shapes;
    use open_rust_mc::geometry::surface::BoundaryCondition;
    use open_rust_mc::geometry::universe::{Universe, UniverseId};
    use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::hdf5_reader;
    use open_rust_mc::transport::dispatch::{CudaRunner, EigenvalueRunner};
    use open_rust_mc::transport::material::Material;
    use open_rust_mc::transport::simulate::SimConfig;
    use open_rust_mc::transport::xs_provider;
    use rust_mc_sim::Pcg64;

    #[derive(Parser, Debug)]
    #[command(
        name = "gpu_hex_minicore",
        about = "Hex-lattice mini-core k-inf on GPU (CudaRunner dispatch)"
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

        #[arg(short, long, default_value_t = 5000)]
        max_events: i32,

        /// Use reflective z boundaries (k_inf) rather than vacuum.
        #[arg(long, default_value_t = true)]
        reflective_z: bool,
    }

    // ── Same as hex_minicore ──────────────────────────────────────────
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

    fn setup_geometry(rings: usize, reflective_z: bool) -> Geometry {
        let inradius = (rings as f64 + 0.5) * PITCH;
        let circumradius = inradius * 2.0 / 3.0_f64.sqrt();
        let z_half = inradius.max(1.0);
        let z_bc = if reflective_z {
            BoundaryCondition::Reflective
        } else {
            BoundaryCondition::Vacuum
        };

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
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_temperature(900.0),
            Cell::new(CellId(1), cell::between(0, 1), CellFill::Void),
            Cell::new(CellId(2), cell::between(1, 2), CellFill::Material(1))
                .with_temperature(600.0),
            Cell::new(CellId(3), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
            Cell::new(CellId(4), outer.inside.clone(), CellFill::HexLattice(0)).with_aabb(
                Aabb::new(
                    Vec3::new(-circumradius, -inradius, -z_half),
                    Vec3::new(circumradius, inradius, z_half),
                ),
            ),
            Cell::new(
                CellId(5),
                cell::Region::Complement(Box::new(outer.inside)),
                CellFill::Void,
            ),
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

    /// Initial source: uniform inside a square inscribed in the hex
    /// inradius circle (every point guaranteed inside the hex), at
    /// 1 MeV. Sampled by particle index for reproducibility.
    fn make_initial_source_factory(
        rings: usize,
        reflective_z: bool,
    ) -> Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)>> {
        let inradius = (rings as f64 + 0.5) * PITCH;
        let z_half = inradius.max(1.0);
        // Use 0.7 * inradius as half-side: comfortably inside the hex
        // for any orientation and free from edge-precision issues.
        let half = inradius * 0.7;
        let z_extent = if reflective_z { z_half } else { z_half };
        Box::new(move |n: usize, seed: u64| -> Vec<(f64, f64, f64, f64)> {
            let mut rng = Pcg64::new(seed, 0);
            (0..n)
                .map(|_| {
                    let x = -half + 2.0 * half * rng.uniform();
                    let y = -half + 2.0 * half * rng.uniform();
                    let z = -z_extent + 2.0 * z_extent * rng.uniform();
                    (x, y, z, 1.0e6)
                })
                .collect()
        })
    }

    pub fn run() {
        let args = Args::parse();
        let inactive = args.inactive.min(args.batches.saturating_sub(1));
        let active = args.batches - inactive;
        let total_histories = active as u64 * args.particles as u64;

        println!("========================================================");
        println!("  GPU Hex-lattice mini-core (CudaRunner dispatch)");
        println!("========================================================");
        println!(
            "  Rings    : {}   pitch {:.3} cm   reflective_z = {}",
            args.rings, PITCH, args.reflective_z
        );
        println!(
            "  Batches  : {} ({} inactive + {} active)   particles/batch = {}",
            args.batches, inactive, active, args.particles
        );
        println!(
            "  SVD rank : {}   seeds = {}   max events = {}",
            args.rank, args.seeds, args.max_events
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

        // ── Load nuclear data ─────────────────────────────────────────
        println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
        let t0 = Instant::now();
        let mut kernels = Vec::new();
        for &(filename, awr, nu_bar, nuc_temp_idx) in NUCLIDE_SPECS.iter() {
            let path = args.data_dir.join(filename);
            kernels.push(std::sync::Arc::new(xs_provider::load_nuclide(
                &path,
                args.rank,
                nuc_temp_idx,
                awr,
                nu_bar,
            )));
        }
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Loaded {} nuclides in {load_ms:.0} ms",
            NUCLIDE_SPECS.len()
        );

        // ── Initialise GPU ────────────────────────────────────────────
        println!("\n── Initialising GPU ──");
        let t1 = Instant::now();
        let gpu = GpuTransportContext::new().expect("GpuTransportContext::new");
        let nuc_data = gpu
            .upload_nuclide_data(&kernels, args.rank)
            .expect("upload nuclides");
        let awrs: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.2).collect();
        let mat_data = gpu
            .upload_material_data(&materials, &awrs, &nu_bars)
            .expect("upload materials");

        // S(α,β) on H-1 (water at 600 K). NUCLIDE_SPECS index 3.
        let n_nuc = NUCLIDE_SPECS.len();
        let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
        let sab_data = if h2o_path.exists() {
            match hdf5_reader::load_thermal_scattering(&h2o_path) {
                Ok(tsl) => {
                    let t_idx = tsl.select_temperature(
                        600.0,
                        open_rust_mc::transport::sim_limits::SimLimits::default()
                            .sab_temperature_tolerance,
                    );
                    println!("  S(α,β): loaded c_H_in_H2O.h5, t_idx = {t_idx}");
                    gpu.upload_sab_data(&tsl, t_idx, 3, n_nuc).expect("upload S(α,β)")
                }
                Err(e) => {
                    eprintln!("  WARN: S(α,β) load failed: {e} — using empty");
                    gpu.upload_sab_data_empty(n_nuc).expect("empty S(α,β)")
                }
            }
        } else {
            println!("  S(α,β): c_H_in_H2O.h5 not found — free-gas only");
            gpu.upload_sab_data_empty(n_nuc).expect("empty S(α,β)")
        };
        let wmp_data = gpu
            .upload_wmp_data_empty(NUCLIDE_SPECS.len())
            .expect("upload empty WMP");

        let recursive = GpuRecursiveContext::build(&geometry, args.particles as usize)
            .expect("GpuRecursiveContext::build");
        let gpu_init_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!("  GPU ready in {gpu_init_ms:.0} ms");

        // Per-material kT: fuel at 900 K, clad+water at 600 K.
        const K_B: f64 = 8.617_333_262e-5;
        let mat_k_t: Vec<f64> = vec![900.0 * K_B, 600.0 * K_B, 600.0 * K_B];
        let sab_nuc_idx: i32 = 3; // H-1

        // ── Source iteration via CudaRunner ───────────────────────────
        println!("\n── Source iteration ──");

        let mut k_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);
        let mut t_per_seed: Vec<f64> = Vec::with_capacity(args.seeds as usize);

        for seed in 0..args.seeds {
            let config = SimConfig {
                batches: args.batches,
                inactive,
                particles_per_batch: args.particles,
                seed: seed as u64,
                auto_inactive: None,
                verbose: false,
                parallel: true,
                tallies: Default::default(),
                statepoint_path: None,
                survival_biasing: None,
                initial_source_bank: None,
                weight_window: None,
                disable_delayed_neutrons: false,
                urr_equivalence: None,
                gpu_refill_pool_factor: None,
                gpu_auto_refill: false,
            };

            if args.seeds > 1 {
                print!("  Seed {seed}: ");
                let _ = std::io::stdout().flush();
            }

            let runner = CudaRunner {
                recursive: &recursive,
                transport: &gpu,
                nuc_data: &nuc_data,
                mat_data: &mat_data,
                sab_data: &sab_data,
                wmp_data: &wmp_data,
                mat_k_t: &mat_k_t,
                sab_nuc_idx,
                max_events_per_history: args.max_events,
                fis_capacity: args.particles as usize * 4,
                initial_source: make_initial_source_factory(args.rings, args.reflective_z),
                buffers: std::cell::RefCell::new(None),
        refill: std::cell::RefCell::new(None),
            };

            let t_seed = Instant::now();
            let outcome = runner.run(&config);
            let sim_ms = t_seed.elapsed().as_secs_f64() * 1000.0;

            k_per_seed.push(outcome.k_eff);
            t_per_seed.push(sim_ms);

            if args.seeds > 1 {
                println!("k_inf={:.5}   {sim_ms:.0} ms", outcome.k_eff);
            } else {
                // Single-seed run: print per-batch trace.
                for b in &outcome.batches {
                    let active_marker = if b.active { " *" } else { "" };
                    println!(
                        "  Batch {:>3}: k = {:.5}  coll = {:>9}  fis = {:>7}  leak = {:>6}{active_marker}",
                        b.batch, b.k_eff, b.collisions, b.fissions, b.leakage
                    );
                }
            }
        }

        let n_seeds = k_per_seed.len() as f64;
        let k_mean = k_per_seed.iter().sum::<f64>() / n_seeds;
        let k_var =
            k_per_seed.iter().map(|k| (k - k_mean).powi(2)).sum::<f64>() / (n_seeds - 1.0).max(1.0);
        let k_std = k_var.sqrt();

        let total_t_ms: f64 = t_per_seed.iter().sum();
        let ns_per_p = total_t_ms * 1e6 / (total_histories * args.seeds as u64) as f64;

        println!("\n========================================================");
        println!(
            "  RESULT — GPU hex mini-core ({} ring{}, SVD rank={})",
            args.rings,
            if args.rings == 1 { "" } else { "s" },
            args.rank
        );
        println!("========================================================");
        println!(
            "  k_inf       = {k_mean:.5} +/- {k_std:.5}  ({} seed{})",
            args.seeds,
            if args.seeds == 1 { "" } else { "s" }
        );
        println!(
            "  Histories   = {} per seed × {} seeds",
            total_histories, args.seeds
        );
        println!("  Sim time    = {total_t_ms:.0} ms");
        println!("  ns/particle = {ns_per_p:.1}");
        println!("\n  Compare to CPU `hex_minicore` with same args.");
    }
}
