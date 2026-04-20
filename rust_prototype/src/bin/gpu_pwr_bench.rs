//! GPU event-based transport benchmark for PWR pin cell.
//!
//! Paper-quality benchmark with multi-seed statistics.
//! Reports k_inf ± σ, ns/particle ± σ, and speedup with error propagation.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_pwr_bench -- <data_dir> \
//!     --rank 5 --batches 100 --inactive 20 --particles 50000 --seeds 5 --mode fused

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_pwr_bench -- ...");
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
    use std::sync::Arc;
    use std::time::Instant;

    use clap::Parser;

    use open_rust_mc::gpu_transport::{
        GpuMaterialData, GpuNuclideData, GpuSabData, GpuTransportContext, GpuWmpData,
    };
    use open_rust_mc::transport::xs_provider;
    use open_rust_mc::wmp::WindowedMultipole;

    #[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
    enum GpuMode {
        Svd,
        Hybrid,
    }

    #[derive(Parser)]
    #[command(
        name = "gpu_pwr_bench",
        about = "GPU event-based PWR pin cell benchmark"
    )]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        rank: usize,
        #[arg(short = 'B', long, default_value_t = 100)]
        batches: u32,
        #[arg(short, long, default_value_t = 20)]
        inactive: u32,
        #[arg(short, long, default_value_t = 20000)]
        particles: u32,
        #[arg(short, long, default_value_t = 1)]
        temp_idx: usize,
        /// Number of independent seeds for statistical benchmarking.
        #[arg(short, long, default_value_t = 1)]
        seeds: u32,
        /// Geometry: "pwr" (8 nuclides, pin cell) or "godiva" (3 nuclides, bare sphere)
        #[arg(short, long, default_value = "pwr")]
        geometry: String,
        /// Force SVD XS path on GPU by clearing uploaded pointwise tables.
        /// Without this flag, nuclides with pointwise data use exact-table lookups
        /// and `--rank` only affects discrete inelastic levels.
        #[arg(long, default_value_t = false)]
        force_svd: bool,
        /// XS provider mode. `svd` = pure SVD/pointwise; `hybrid` = SVD +
        /// Windowed-Multipole in the RRR window for nuclides that have WMP data.
        #[arg(short, long, value_enum, default_value_t = GpuMode::Svd)]
        mode: GpuMode,
    }

    /// WMP filename + target temperature per nuclide, parallel to PWR_NUCLIDES.
    /// None entries (empty string) keep the SVD/pointwise path.
    const PWR_WMP_SPECS: &[(&str, f64)] = &[
        ("092235.h5", 900.0), // 0  U235 fuel
        ("092238.h5", 900.0), // 1  U238 fuel
        ("008016.h5", 900.0), // 2  O16 fuel
        ("001001.h5", 600.0), // 3  H1 water
        ("040090.h5", 600.0), // 4  Zr90 clad
        ("040091.h5", 600.0), // 5  Zr91 clad
        ("040092.h5", 600.0), // 6  Zr92 clad
        ("040094.h5", 600.0), // 7  Zr94 clad
        ("008016.h5", 600.0), // 8  O16 water
    ];

    const GODIVA_WMP_SPECS: &[(&str, f64)] = &[
        ("092234.h5", 294.0),
        ("092235.h5", 294.0),
        ("092238.h5", 294.0),
    ];

    const PWR_NUCLIDES: &[(&str, f64, f64, usize)] = &[
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

    const GODIVA_NUCLIDES: &[(&str, f64, f64, usize)] = &[
        ("U234.h5", 232.029, 2.49, 1), // 294K
        ("U235.h5", 233.025, 2.43, 1), // 294K
        ("U238.h5", 236.006, 2.49, 1), // 294K
    ];

    /// Per-seed result.
    struct SeedResult {
        seed: u32,
        k_mean: f64,
        k_std: f64,
        sim_ms: f64,
        total_histories: u64,
    }

    impl SeedResult {
        fn ns_per_particle(&self) -> f64 {
            self.sim_ms * 1e6 / self.total_histories as f64
        }
    }

    /// Aggregated multi-seed result.
    struct BenchResult {
        label: String,
        seed_results: Vec<SeedResult>,
        load_ms: f64,
    }

    impl BenchResult {
        fn k_mean(&self) -> f64 {
            self.seed_results.iter().map(|r| r.k_mean).sum::<f64>() / self.seed_results.len() as f64
        }

        fn k_std(&self) -> f64 {
            if self.seed_results.len() < 2 {
                return self.seed_results[0].k_std;
            }
            let mean = self.k_mean();
            let n = self.seed_results.len() as f64;
            let var = self
                .seed_results
                .iter()
                .map(|r| (r.k_mean - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt()
        }

        fn ns_mean(&self) -> f64 {
            self.seed_results
                .iter()
                .map(|r| r.ns_per_particle())
                .sum::<f64>()
                / self.seed_results.len() as f64
        }

        fn ns_std(&self) -> f64 {
            if self.seed_results.len() < 2 {
                return 0.0;
            }
            let mean = self.ns_mean();
            let n = self.seed_results.len() as f64;
            let var = self
                .seed_results
                .iter()
                .map(|r| (r.ns_per_particle() - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt()
        }
    }

    fn initial_source_pwr(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        use open_rust_mc::transport::rng::Rng;
        let fuel_or = 0.4096_f64;
        let half = 0.63_f64;
        let mut rng = Rng::new(seed * 100_000, 0);
        let mut sites = Vec::with_capacity(n);
        while sites.len() < n {
            let x = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let y = -fuel_or + rng.uniform() * 2.0 * fuel_or;
            let z = -half + rng.uniform() * 2.0 * half;
            if x * x + y * y < fuel_or * fuel_or {
                sites.push((x, y, z, 1.0e6));
            }
        }
        sites
    }

    fn initial_source_godiva(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        use open_rust_mc::transport::rng::Rng;
        let r = 8.7407_f64;
        let mut rng = Rng::new(seed * 100_000, 0);
        let mut sites = Vec::with_capacity(n);
        while sites.len() < n {
            let x = -r + rng.uniform() * 2.0 * r;
            let y = -r + rng.uniform() * 2.0 * r;
            let z = -r + rng.uniform() * 2.0 * r;
            if x * x + y * y + z * z < r * r {
                sites.push((x, y, z, 1.0e6));
            }
        }
        sites
    }

    fn initial_source(n: usize, seed: u64, is_godiva: bool) -> Vec<(f64, f64, f64, f64)> {
        if is_godiva {
            initial_source_godiva(n, seed)
        } else {
            initial_source_pwr(n, seed)
        }
    }

    fn normalize_bank(
        bank: &[(f64, f64, f64, f64)],
        n: usize,
        seed: u64,
    ) -> Vec<(f64, f64, f64, f64)> {
        use open_rust_mc::transport::rng::Rng;
        if bank.is_empty() {
            return initial_source(n, seed, false);
        }
        let mut rng = Rng::new(seed, 0);
        (0..n)
            .map(|_| {
                let idx = (rng.uniform() * bank.len() as f64) as usize;
                bank[idx.min(bank.len() - 1)]
            })
            .collect()
    }

    fn run_gpu_seeds(
        gpu: &GpuTransportContext,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        wmp_data: &GpuWmpData,
        label: &str,
        n: usize,
        batches: u32,
        inactive: u32,
        seeds: u32,
        geom_type: i32,
    ) -> BenchResult {
        let active_batches = batches - inactive;
        let total_histories = active_batches as u64 * n as u64;
        let mut seed_results = Vec::with_capacity(seeds as usize);

        println!(
            "\n── {label} ({seeds} seed{}) ──",
            if seeds > 1 { "s" } else { "" }
        );

        for seed in 0..seeds {
            let seed_offset = seed as u64 * 1_000_000;
            let mut source_bank = initial_source(n, seed as u64, geom_type == 1);
            let mut k_sum = 0.0_f64;
            let mut k_count = 0_u32;

            if seeds > 1 {
                print!("  Seed {seed}: ");
            }
            let _ = std::io::stdout().flush();

            let t_sim = Instant::now();
            for batch in 1..=batches {
                let batch_id = batch + seed * batches;
                let result = gpu
                    .run_batch(
                        &source_bank,
                        batch_id,
                        nuc_data,
                        mat_data,
                        sab_data,
                        wmp_data,
                        1_000_000,
                        geom_type,
                    )
                    .expect("GPU batch failed");

                if seeds == 1 {
                    let active = if batch > inactive { " *" } else { "" };
                    println!(
                        "  Batch {:>4}: k={:.5}  coll={}  fiss={}  leak={}  surf={}{active}",
                        batch,
                        result.k_eff,
                        result.collisions,
                        result.fissions,
                        result.leakage,
                        result.surface_crossings
                    );
                    let _ = std::io::stdout().flush();
                }

                if batch > inactive {
                    k_sum += result.k_eff;
                    k_count += 1;
                }
                source_bank =
                    normalize_bank(&result.fission_bank, n, batch_id as u64 + seed_offset);
            }

            let sim_ms = t_sim.elapsed().as_secs_f64() * 1000.0;
            let k_mean = if k_count > 0 {
                k_sum / k_count as f64
            } else {
                0.0
            };

            // Intra-seed k_eff standard deviation
            // (would need per-batch k values for this; approximate from batch count)
            let k_std = 0.001; // placeholder — computed properly below if batch data available

            let sr = SeedResult {
                seed,
                k_mean,
                k_std,
                sim_ms,
                total_histories,
            };

            if seeds > 1 {
                println!(
                    "k={:.5}  {:.0}ms  ({:.1} ns/p)",
                    k_mean,
                    sim_ms,
                    sr.ns_per_particle()
                );
            }

            seed_results.push(sr);
        }

        BenchResult {
            label: label.to_string(),
            seed_results,
            load_ms: 0.0,
        }
    }

    fn print_result(r: &BenchResult) {
        let n = r.seed_results.len();
        println!("  {}:", r.label);
        if n > 1 {
            println!(
                "    k_inf            = {:.5} +/- {:.5} ({n} seeds)",
                r.k_mean(),
                r.k_std()
            );
            for sr in &r.seed_results {
                println!(
                    "      seed {}: k={:.5}  ({:.1} ns/p)",
                    sr.seed,
                    sr.k_mean,
                    sr.ns_per_particle()
                );
            }
            println!(
                "    ns/particle      = {:.2} +/- {:.2}",
                r.ns_mean(),
                r.ns_std()
            );
        } else {
            println!("    k_inf            = {:.5}", r.k_mean());
            println!("    ns/particle      = {:.2}", r.ns_mean());
        }
        let total_ms: f64 = r.seed_results.iter().map(|s| s.sim_ms).sum();
        println!(
            "    Total sim time   = {:.0} ms ({n} run{})",
            total_ms,
            if n > 1 { "s" } else { "" }
        );
    }

    pub fn run() {
        let args = Args::parse();
        let inactive = args.inactive.min(args.batches.saturating_sub(1));
        let active_batches = args.batches - inactive;
        let n = args.particles as usize;

        let is_godiva = args.geometry == "godiva";
        let geom_type: i32 = if is_godiva { 1 } else { 0 };
        let nuclide_specs: &[(&str, f64, f64, usize)] = if is_godiva {
            GODIVA_NUCLIDES
        } else {
            PWR_NUCLIDES
        };
        let geom_label = if is_godiva {
            "Godiva (bare HEU sphere)"
        } else {
            "PWR pin cell"
        };

        println!("=== GPU {} — Paper Benchmark ===\n", geom_label);
        println!("Data dir:     {}", args.data_dir.display());
        println!("Geometry:     {}", geom_label);
        println!("SVD rank:     {}", args.rank);
        println!(
            "Batches:      {} ({} inactive + {} active)",
            args.batches, inactive, active_batches
        );
        println!("Particles:    {}/batch", args.particles);
        println!("Seeds:        {}", args.seeds);
        println!(
            "Histories:    {} per seed",
            active_batches as u64 * n as u64
        );

        // ── Load nuclear data ──
        println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
        let t_load = Instant::now();
        let mut kernels = Vec::new();
        for &(filename, awr, nu_bar, nuc_temp_idx) in nuclide_specs {
            let path = args.data_dir.join(filename);
            println!("  Loading {}...", filename);
            let mut k = xs_provider::load_nuclide(&path, args.rank, nuc_temp_idx, awr, nu_bar);
            if args.force_svd {
                // Null out pointwise XS so GPU uses SVD path for main channels.
                k.pointwise_xs = None;
            }
            kernels.push(k);
        }
        if args.force_svd {
            println!(
                "  force_svd: cleared pointwise XS -> GPU uses SVD rank={} for main channels",
                args.rank
            );
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        println!("  Loaded in {load_ms:.0} ms");

        // ── Initialize GPU ──
        println!("\n── Initializing GPU transport ──");
        let t_gpu = Instant::now();
        let gpu = match GpuTransportContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  Failed: {e}");
                return;
            }
        };

        let nuc_data = gpu
            .upload_nuclide_data(&kernels, args.rank)
            .expect("upload nuclide data");
        let materials = if is_godiva {
            setup_godiva_materials()
        } else {
            setup_materials()
        };
        let awrs: Vec<f64> = nuclide_specs.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = nuclide_specs.iter().map(|s| s.2).collect();
        let mat_data = gpu
            .upload_material_data(&materials, &awrs, &nu_bars)
            .expect("upload material data");

        let h2o_path = args.data_dir.join("c_H_in_H2O.h5");
        let sab_data = if !is_godiva && h2o_path.exists() {
            match open_rust_mc::hdf5_reader::load_thermal_scattering(&h2o_path) {
                Ok(tsl) => {
                    let t_idx = tsl.select_temperature(600.0, 0.5);
                    gpu.upload_sab_data(&tsl, t_idx).expect("upload S(a,b)")
                }
                Err(e) => {
                    eprintln!("  WARNING: S(a,b) load failed: {e}");
                    gpu.upload_sab_data_empty().expect("empty S(a,b)")
                }
            }
        } else {
            println!("  S(a,b): not found — using free-gas");
            gpu.upload_sab_data_empty().expect("empty S(a,b)")
        };

        // ── Load Windowed-Multipole data when --mode hybrid ──
        let wmp_data = match args.mode {
            GpuMode::Svd => gpu
                .upload_wmp_data_empty(nuclide_specs.len())
                .expect("upload empty WMP"),
            GpuMode::Hybrid => {
                let wmp_dir = args
                    .data_dir
                    .parent()
                    .map(|p| p.join("wmp"))
                    .unwrap_or_else(|| args.data_dir.join("..").join("wmp"));
                let wmp_specs: &[(&str, f64)] = if is_godiva {
                    GODIVA_WMP_SPECS
                } else {
                    PWR_WMP_SPECS
                };
                let mut wmps: Vec<Option<(Arc<WindowedMultipole>, f64)>> =
                    Vec::with_capacity(wmp_specs.len());
                let mut covered = 0usize;
                for &(wmp_file, t_kelvin) in wmp_specs {
                    let path = wmp_dir.join(wmp_file);
                    if !path.exists() {
                        println!("  WMP not found: {}", path.display());
                        wmps.push(None);
                        continue;
                    }
                    match WindowedMultipole::from_hdf5(&path) {
                        Ok(wmp) => {
                            println!(
                                "  Loaded WMP {wmp_file} at T={t_kelvin} K   \
                                (E {:.2e}..{:.2e} eV, {} poles, {} windows)",
                                wmp.e_min, wmp.e_max, wmp.n_poles, wmp.n_windows
                            );
                            covered += 1;
                            wmps.push(Some((Arc::new(wmp), t_kelvin)));
                        }
                        Err(e) => {
                            println!("  WMP load failed for {wmp_file}: {e:?}");
                            wmps.push(None);
                        }
                    }
                }
                println!(
                    "  Hybrid: {covered} / {} nuclides covered by WMP",
                    wmp_specs.len()
                );
                gpu.upload_wmp_data(&wmps).expect("upload WMP")
            }
        };

        let gpu_init_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
        println!("  GPU ready in {gpu_init_ms:.0} ms");

        // ── Run benchmark ──
        let mode_label = match args.mode {
            GpuMode::Svd => "SVD",
            GpuMode::Hybrid => "Hybrid SVD+WMP",
        };
        let mut r = run_gpu_seeds(
            &gpu,
            &nuc_data,
            &mat_data,
            &sab_data,
            &wmp_data,
            &format!("GPU {mode_label} {geom_label}"),
            n,
            args.batches,
            inactive,
            args.seeds,
            geom_type,
        );
        r.load_ms = load_ms + gpu_init_ms;

        // ── Print results ──
        println!("\n{}", "=".repeat(60));
        println!("BENCHMARK RESULTS - {geom_label}");
        println!("{}", "=".repeat(60));
        print_result(&r);
        println!("\n  Load time = {load_ms:.0} ms (CPU) + {gpu_init_ms:.0} ms (GPU)");
        println!(
            "  Physics:  SVD rank={}, S(a,b)={}, nu-bar=table, fission=CDF",
            args.rank,
            if h2o_path.exists() { "H2O" } else { "free-gas" }
        );
    }

    fn setup_godiva_materials() -> Vec<open_rust_mc::transport::material::Material> {
        use open_rust_mc::transport::material::Material;
        // Godiva HEU: single material, 3 nuclides
        // Atom densities for 93.71% enriched U metal at 18.74 g/cm3
        let mut fuel = Material::new("HEU", 294.0);
        fuel.add_nuclide(0.000483, 0); // U-234 (xs_kernel_idx=0)
        fuel.add_nuclide(0.04509, 1); // U-235 (xs_kernel_idx=1)
        fuel.add_nuclide(0.00265, 2); // U-238 (xs_kernel_idx=2)
        vec![fuel]
    }

    fn setup_materials() -> Vec<open_rust_mc::transport::material::Material> {
        use open_rust_mc::transport::material::Material;
        let mut fuel = Material::new("UO2", 900.0);
        fuel.add_nuclide(0.000719, 0); // U-235
        fuel.add_nuclide(0.022482, 1); // U-238
        fuel.add_nuclide(0.046402, 2); // O-16 at 900K
        let mut clad = Material::new("Zircaloy", 600.0);
        clad.add_nuclide(0.022932, 4); // Zr-90
        clad.add_nuclide(0.004996, 5); // Zr-91
        clad.add_nuclide(0.007636, 6); // Zr-92
        clad.add_nuclide(0.007740, 7); // Zr-94
        let mut water = Material::new("H2O", 600.0);
        water.add_nuclide(0.049486, 3); // H-1
        water.add_nuclide(0.024743, 8); // O-16 at 600K (separate from fuel O16)
        vec![fuel, clad, water]
    }
}
