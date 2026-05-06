//! 17×17 PWR fuel assembly k_inf on GPU — task #22 stage 4.
//!
//! Drives `transport_recursive_persistent` through a full source
//! iteration (inactive + active batches with fission-bank propagation)
//! on the depth-3 Westinghouse-pattern assembly geometry from
//! `pwr_assembly`. Reports k_inf with active-batch mean ± std.
//!
//! This is the headline geometry from `resume.md`:
//!   * 264 UO2 fuel pins + 24 guide tubes + 1 instrument tube
//!   * pin pitch 1.26 cm, fuel/clad cylinders, reflective x/y/z box
//!   * 9 nuclides, S(α,β) on H-1, full SVD/PW/URR XS path
//!   * stack depth 3: root → assembly lattice → pin/GT cell
//!
//! CPU reference (resume.md): k_inf = 1.14958 ± 0.00318.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_assembly_keff -- \
//!     <data_dir> --rank 5 --batches 50 --inactive 15 --particles 10000

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: this binary requires the 'cuda' feature.");
    eprintln!(
        "Build with: cargo run --release --features cuda --bin gpu_assembly_keff -- ..."
    );
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

    use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
    use open_rust_mc::geometry::lattice::RectLattice;
    use open_rust_mc::geometry::surface::BoundaryCondition;
    use open_rust_mc::geometry::universe::{Universe, UniverseId};
    use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::material::Material;
    use open_rust_mc::transport::xs_provider;
    use rust_mc_sim::Pcg64;

    #[derive(Parser)]
    #[command(name = "gpu_assembly_keff")]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        rank: usize,
        #[arg(short, long, default_value_t = 50)]
        batches: u32,
        #[arg(short, long, default_value_t = 15)]
        inactive: u32,
        #[arg(short, long, default_value_t = 10_000)]
        particles: usize,
        #[arg(short, long, default_value_t = 1)]
        seeds: u32,
        #[arg(short, long, default_value_t = 5_000)]
        max_events: i32,
        /// Diagnostic: replace every GT element with a fuel pin (k_inf
        /// should converge to the pin-cell value ≈ 1.328 if the lattice
        /// element → universe mapping is uploaded correctly).
        #[arg(long, default_value_t = false)]
        no_gt: bool,
        /// Diagnostic: replace every fuel element with a guide tube
        /// (k_inf should be 0 — no fissionable material left).
        #[arg(long, default_value_t = false)]
        all_gt: bool,
    }

    // ── Geometry constants (mirrors pwr_assembly.rs) ──────────────────
    const PITCH: f64 = 1.260;
    const FUEL_OR: f64 = 0.4096;
    const CLAD_IR: f64 = 0.4180;
    const CLAD_OR: f64 = 0.4750;
    const SHAPE: usize = 17;
    const U_ROOT: u32 = 0;
    const U_PIN: u32 = 1;
    const U_GT: u32 = 2;

    /// Same nuclide list as pwr_assembly / pwr_pincell.
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

    /// Westinghouse 17×17: 24 guide tubes + 1 instrument tube. `true`
    /// = guide/instrument (water-filled), `false` = fuel pin.
    fn westinghouse_layout() -> [[bool; SHAPE]; SHAPE] {
        let mut l = [[false; SHAPE]; SHAPE];
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
        l
    }

    /// Build the recursive 17×17 assembly geometry. Mirrors the CPU
    /// `pwr_assembly::setup_geometry`. `reflective_z=true` gives k_inf
    /// (no axial leakage) — the comparison case for resume.md's CPU
    /// reference.
    fn setup_geometry(reflective_z: bool, no_gt: bool, all_gt: bool) -> Geometry {
        let lat_half = (SHAPE as f64) * PITCH / 2.0;
        let z_half = lat_half;
        let pin_center = PITCH / 2.0;
        let z_bc = if reflective_z {
            BoundaryCondition::Reflective
        } else {
            BoundaryCondition::Vacuum
        };

        let mut surfaces = open_rust_mc::geometry::shapes::pin_cylinders(
            pin_center,
            pin_center,
            &[FUEL_OR, CLAD_IR, CLAD_OR],
        );
        let outer_box = open_rust_mc::geometry::shapes::rect_box_split_bc(
            [lat_half, lat_half, z_half],
            BoundaryCondition::Reflective,
            z_bc,
            surfaces.len(),
        );
        surfaces.extend(outer_box.surfaces);

        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)).with_temperature(900.0),
            Cell::new(CellId(1), cell::between(0, 1), CellFill::Void),
            Cell::new(CellId(2), cell::between(1, 2), CellFill::Material(1)).with_temperature(600.0),
            Cell::new(CellId(3), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
            Cell::new(CellId(4), cell::inside(1), CellFill::Material(2)).with_temperature(600.0),
            Cell::new(CellId(5), cell::between(1, 2), CellFill::Material(1)).with_temperature(600.0),
            Cell::new(CellId(6), cell::outside(2), CellFill::Material(2)).with_temperature(600.0),
            Cell::new(CellId(7), outer_box.inside.clone(), CellFill::Lattice(0)).with_aabb(
                Aabb::new(
                    Vec3::new(-lat_half, -lat_half, -z_half),
                    Vec3::new(lat_half, lat_half, z_half),
                ),
            ),
            Cell::new(
                CellId(8),
                Region::Complement(Box::new(outer_box.inside)),
                CellFill::Void,
            ),
        ];
        let universes = vec![
            Universe::new(UniverseId(U_ROOT), vec![7, 8]),
            Universe::new(UniverseId(U_PIN), vec![0, 1, 2, 3]),
            Universe::new(UniverseId(U_GT), vec![4, 5, 6]),
        ];
        let layout = westinghouse_layout();
        let mut lat_universes = Vec::with_capacity(SHAPE * SHAPE);
        for iy in 0..SHAPE {
            for ix in 0..SHAPE {
                let mut is_gt = layout[iy][ix];
                if no_gt {
                    is_gt = false;
                }
                if all_gt {
                    is_gt = true;
                }
                let id = if is_gt {
                    UniverseId(U_GT)
                } else {
                    UniverseId(U_PIN)
                };
                lat_universes.push(id);
            }
        }
        let lattices = vec![RectLattice {
            origin: Vec3::new(-lat_half, -lat_half, -z_half),
            pitch: Vec3::new(PITCH, PITCH, 2.0 * z_half),
            shape: [SHAPE, SHAPE, 1],
            universes: lat_universes,
            material_overrides: None,
        }];
        Geometry::new(surfaces, cells, universes, lattices, UniverseId(U_ROOT))
            .expect("assembly geometry must validate")
    }

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

    /// Initial source: uniform inside the assembly box, 1 MeV.
    fn initial_source(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        let half = (SHAPE as f64) * PITCH / 2.0;
        let z_half = half;
        let mut rng = Pcg64::new(seed, 0);
        (0..n)
            .map(|_| {
                let x = -half + 2.0 * half * rng.uniform();
                let y = -half + 2.0 * half * rng.uniform();
                let z = -z_half + 2.0 * z_half * rng.uniform();
                (x, y, z, 1.0e6)
            })
            .collect()
    }

    /// Resample N source sites from a non-empty fission bank.
    fn normalize_bank(
        bank: &[(f64, f64, f64, f64)],
        n: usize,
        seed: u64,
    ) -> Vec<(f64, f64, f64, f64)> {
        if bank.is_empty() {
            return initial_source(n, seed);
        }
        let mut rng = Pcg64::new(seed, 0);
        (0..n)
            .map(|_| {
                let idx = (rng.uniform() * bank.len() as f64) as usize;
                bank[idx.min(bank.len() - 1)]
            })
            .collect()
    }

    pub fn run() {
        let args = Args::parse();
        let inactive = args.inactive.min(args.batches.saturating_sub(1));
        let active = args.batches - inactive;

        println!("=== GPU 17×17 PWR assembly k_inf — recursive geometry ===");
        println!("  data dir   : {}", args.data_dir.display());
        println!(
            "  geometry   : depth-3 Westinghouse 17x17 (264 fuel + 24 GT + 1 IT)"
        );
        println!("  particles  : {}/batch", args.particles);
        println!(
            "  batches    : {} ({} inactive + {} active)",
            args.batches, inactive, active
        );
        println!("  SVD rank   : {}", args.rank);
        println!("  seeds      : {}", args.seeds);

        // ── Load nuclear data ─────────────────────────────────────────
        println!("\n── Loading nuclear data ──");
        let t0 = Instant::now();
        let mut kernels = Vec::new();
        for &(fname, awr, nu_bar, t_idx) in NUCLIDE_SPECS {
            let path = args.data_dir.join(fname);
            println!("  loading {fname}...");
            kernels.push(xs_provider::load_nuclide(
                &path, args.rank, t_idx, awr, nu_bar,
            ));
        }
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  loaded in {load_ms:.0} ms");

        // ── Initialise GPU + upload data ──────────────────────────────
        println!("\n── Initialising GPU ──");
        let t1 = Instant::now();
        let gpu = GpuTransportContext::new().expect("GpuTransportContext::new");
        let nuc_data = gpu
            .upload_nuclide_data(&kernels, args.rank)
            .expect("upload nuclides");
        let materials = setup_materials();
        let awrs: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.2).collect();
        let mat_data = gpu
            .upload_material_data(&materials, &awrs, &nu_bars)
            .expect("upload materials");

        // S(α,β) on H-1
        let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
        let sab_data = if h2o_path.exists() {
            match open_rust_mc::hdf5_reader::load_thermal_scattering(&h2o_path) {
                Ok(tsl) => {
                    let t_idx = tsl.select_temperature(600.0, 0.5);
                    println!("  S(α,β): loaded c_H_in_H2O.h5, t_idx = {t_idx}");
                    gpu.upload_sab_data(&tsl, t_idx).expect("upload S(α,β)")
                }
                Err(e) => {
                    eprintln!("  WARN: S(α,β) load failed: {e} — using empty");
                    gpu.upload_sab_data_empty().expect("empty S(α,β)")
                }
            }
        } else {
            println!("  S(α,β): c_H_in_H2O.h5 not found — free-gas only");
            gpu.upload_sab_data_empty().expect("empty S(α,β)")
        };
        let wmp_data = gpu
            .upload_wmp_data_empty(NUCLIDE_SPECS.len())
            .expect("upload empty WMP");

        let geom = setup_geometry(true, args.no_gt, args.all_gt);
        if args.no_gt {
            println!("  [diagnostic] --no-gt: every lattice element is a fuel pin");
        }
        if args.all_gt {
            println!("  [diagnostic] --all-gt: every lattice element is a guide tube");
        }
        println!(
            "  geometry: {} surfaces, {} cells, {} universes, {} lattices",
            geom.surfaces.len(),
            geom.cells.len(),
            geom.universes.len(),
            geom.lattices.len()
        );
        let rec = GpuRecursiveContext::build(&geom, args.particles)
            .expect("GpuRecursiveContext::build");

        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
        println!("  GPU ready in {gpu_ms:.0} ms");

        // ── Per-material kT ────────────────────────────────────────────
        const K_B: f64 = 8.617_333_262e-5; // eV/K
        let mat_k_t: Vec<f64> = vec![900.0 * K_B, 600.0 * K_B, 600.0 * K_B];
        let sab_nuc_idx: i32 = 3; // H-1

        // ── Source iteration ──────────────────────────────────────────
        println!(
            "\n── Source iteration ({} batch{} per seed) ──",
            args.batches,
            if args.batches > 1 { "es" } else { "" }
        );

        let mut all_seeds_k_means = Vec::with_capacity(args.seeds as usize);
        let total_t0 = Instant::now();

        for seed in 0..args.seeds {
            let seed_offset = seed as u64 * 1_000_000;
            let mut source = initial_source(args.particles, 0xCAFEBEEF + seed as u64);
            let mut k_active: Vec<f64> = Vec::with_capacity(active as usize);

            println!("\n  Seed {seed}:");
            let seed_t0 = Instant::now();

            for batch in 1..=args.batches {
                let batch_seed = seed_offset + batch as u64 * 1_000;
                let rng_seeds: Vec<(u64, u64)> = (0..args.particles)
                    .map(|i| {
                        let p = Pcg64::for_particle(batch_seed, i as u64);
                        (p.state(), p.stream())
                    })
                    .collect();

                let result = rec
                    .transport_recursive(
                        &gpu,
                        &nuc_data,
                        &mat_data,
                        &sab_data,
                        &wmp_data,
                        &source,
                        &rng_seeds,
                        &mat_k_t,
                        sab_nuc_idx,
                        args.max_events,
                        args.particles * 4,
                    )
                    .expect("transport_recursive failed");

                let k_eff = result.k_eff;
                let active_marker = if batch > inactive { " *" } else { "" };
                println!(
                    "    Batch {:>3}: k = {:.5}  bank = {:>6}  coll = {:>9}  fis = {:>7}  leak = {:>6}{active_marker}",
                    batch,
                    k_eff,
                    result.fission_bank.len(),
                    result.n_collisions,
                    result.n_fissions,
                    result.n_leakage
                );
                let _ = std::io::stdout().flush();

                if batch > inactive {
                    k_active.push(k_eff);
                }
                source = normalize_bank(&result.fission_bank, args.particles, batch_seed);
            }

            let seed_ms = seed_t0.elapsed().as_secs_f64() * 1000.0;
            let k_mean: f64 = k_active.iter().sum::<f64>() / k_active.len() as f64;
            let k_var: f64 = k_active
                .iter()
                .map(|k| (k - k_mean).powi(2))
                .sum::<f64>()
                / (k_active.len() - 1).max(1) as f64;
            let k_se = (k_var / k_active.len() as f64).sqrt();

            println!(
                "    → seed {seed}: k_inf = {:.5} ± {:.5} ({:.0} ms, {:.1} ns/p)",
                k_mean,
                k_se,
                seed_ms,
                seed_ms * 1e6 / (args.batches as f64 * args.particles as f64)
            );
            all_seeds_k_means.push(k_mean);
        }

        let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

        // ── Aggregate across seeds ───────────────────────────────────
        let k_grand_mean: f64 =
            all_seeds_k_means.iter().sum::<f64>() / all_seeds_k_means.len() as f64;
        let k_grand_var: f64 = all_seeds_k_means
            .iter()
            .map(|k| (k - k_grand_mean).powi(2))
            .sum::<f64>()
            / (all_seeds_k_means.len() - 1).max(1) as f64;
        let k_grand_se = (k_grand_var / all_seeds_k_means.len() as f64).sqrt();

        println!("\n{}", "=".repeat(64));
        println!("RESULT — 17×17 PWR assembly k_inf (recursive GPU transport)");
        println!("{}", "=".repeat(64));
        if args.seeds > 1 {
            println!(
                "  k_inf  = {:.5} ± {:.5}   ({} seeds × {} active batches × {} particles)",
                k_grand_mean,
                k_grand_se,
                args.seeds,
                active,
                args.particles
            );
            for (i, k) in all_seeds_k_means.iter().enumerate() {
                println!("    seed {i}: k = {:.5}", k);
            }
        } else {
            println!(
                "  k_inf  = {:.5}   ({} active batches × {} particles)",
                k_grand_mean, active, args.particles
            );
        }
        println!("  CPU reference (resume.md): k_inf = 1.14958 ± 0.00318");
        let delta_pcm = (k_grand_mean - 1.14958) * 1e5;
        println!("  Δ vs CPU = {delta_pcm:+.0} pcm");
        println!(
            "  total wall = {:.0} ms ({:.1} ns/particle)",
            total_ms,
            total_ms * 1e6
                / (args.seeds as f64 * args.batches as f64 * args.particles as f64)
        );
    }
}
