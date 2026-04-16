//! GPU event-based transport benchmark for PWR pin cell.
//!
//! Compares CPU vs GPU eigenvalue transport, same physics simplifications
//! on both sides for fair comparison. Reports k_eff, timing, and speedup.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_pwr_bench -- <data_dir> [--rank K] [--batches N] [--particles N]

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
    use std::time::Instant;

    use clap::Parser;

    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::xs_provider;

    #[derive(Parser)]
    #[command(name = "gpu_pwr_bench", about = "GPU event-based PWR pin cell benchmark")]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        rank: usize,
        #[arg(short, long, default_value_t = 50)]
        batches: u32,
        #[arg(short, long, default_value_t = 10)]
        inactive: u32,
        #[arg(short, long, default_value_t = 10000)]
        particles: u32,
        #[arg(short, long, default_value_t = 1)]
        temp_idx: usize,
        /// Run mode: "fused" (optimized), "split" (3-kernel), "both" (compare)
        #[arg(short, long, default_value = "both")]
        mode: String,
    }

    /// Nuclide specs: (filename, AWR, fallback nu-bar) — same as pwr_pincell.rs
    const NUCLIDE_SPECS: &[(&str, f64, f64)] = &[
        ("U235.h5", 233.025, 2.43),
        ("U238.h5", 236.006, 2.49),
        ("O16.h5",  15.858,  0.0),
        ("H1.h5",    0.999,  0.0),
        ("Zr90.h5", 89.132,  0.0),
        ("Zr91.h5", 90.130,  0.0),
        ("Zr92.h5", 91.126,  0.0),
        ("Zr94.h5", 93.120,  0.0),
    ];

    /// Create initial source in fuel region (rejection sampling).
    fn initial_source(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
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
                sites.push((x, y, z, 1.0e6)); // 1 MeV fission source
            }
        }
        sites
    }

    /// Normalize fission bank to N particles.
    fn normalize_bank(bank: &[(f64, f64, f64, f64)], n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        use open_rust_mc::transport::rng::Rng;

        if bank.is_empty() {
            return initial_source(n, seed);
        }
        let mut rng = Rng::new(seed, 0);
        (0..n).map(|_| {
            let idx = (rng.uniform() * bank.len() as f64) as usize;
            bank[idx.min(bank.len() - 1)]
        }).collect()
    }

    pub fn run() {
        let args = Args::parse();
        let active_batches = args.batches - args.inactive;
        let n = args.particles as usize;

        println!("=== GPU Event-Based PWR Pin Cell Benchmark ===\n");
        println!("Data dir:     {}", args.data_dir.display());
        println!("SVD rank:     {}", args.rank);
        println!("Batches:      {} ({} inactive + {} active)",
                 args.batches, args.inactive, active_batches);
        println!("Particles:    {}/batch", args.particles);
        println!("Max events:   1,000,000 per particle");

        // ── Load nuclear data ──
        println!("\n── Loading nuclear data (SVD, rank={}) ──", args.rank);
        let t_load = Instant::now();

        let mut kernels = Vec::new();
        for &(filename, awr, nu_bar) in NUCLIDE_SPECS {
            let path = args.data_dir.join(filename);
            println!("  Loading {}...", filename);
            kernels.push(xs_provider::load_nuclide(&path, args.rank, args.temp_idx, awr, nu_bar));
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

        let nuc_data = gpu.upload_nuclide_data(&kernels, args.rank)
            .expect("upload nuclide data");

        // Material data — same as pwr_pincell.rs
        let materials = setup_materials();
        let awrs: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.1).collect();
        let nu_bars: Vec<f64> = NUCLIDE_SPECS.iter().map(|s| s.2).collect();
        let mat_data = gpu.upload_material_data(&materials, &awrs, &nu_bars)
            .expect("upload material data");

        let gpu_init_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
        println!("  GPU ready in {gpu_init_ms:.0} ms");

        let run_split = args.mode == "split" || args.mode == "both";
        let run_fused = args.mode == "fused" || args.mode == "both";

        // ── Helper: run one mode ──
        fn run_mode(
            label: &str,
            gpu: &GpuTransportContext,
            nuc_data: &open_rust_mc::gpu_transport::GpuNuclideData,
            mat_data: &open_rust_mc::gpu_transport::GpuMaterialData,
            n: usize,
            batches: u32,
            inactive: u32,
            fused: bool,
        ) -> (f64, f64, f64) {
            println!("\n── {label} ──");
            let mut source_bank = initial_source(n, 0);
            let mut k_sum = 0.0_f64;
            let mut k_count = 0_u32;
            let active_batches = batches - inactive;

            let t_sim = Instant::now();
            for batch in 1..=batches {
                let result = if fused {
                    gpu.run_batch_fused(&source_bank, batch, nuc_data, mat_data, 1_000_000)
                } else {
                    gpu.run_batch(&source_bank, batch, nuc_data, mat_data, 1_000_000)
                }.expect("GPU batch failed");

                let active = if batch > inactive { " *" } else { "" };
                println!("  Batch {:>4}: k={:.5}  coll={}  fiss={}  leak={}  surf={}{active}",
                         batch, result.k_eff, result.collisions, result.fissions,
                         result.leakage, result.surface_crossings);
                let _ = std::io::stdout().flush();

                if batch > inactive {
                    k_sum += result.k_eff;
                    k_count += 1;
                }
                source_bank = normalize_bank(&result.fission_bank, n, batch as u64);
            }

            let sim_ms = t_sim.elapsed().as_secs_f64() * 1000.0;
            let total_histories = active_batches as u64 * n as u64;
            let ns_pp = sim_ms * 1e6 / total_histories as f64;
            let k_mean = if k_count > 0 { k_sum / k_count as f64 } else { 0.0 };

            println!("\n  {label}:");
            println!("    k_inf       = {k_mean:.5}");
            println!("    ns/particle = {ns_pp:.2}");
            println!("    sim time    = {sim_ms:.0} ms");

            (k_mean, ns_pp, sim_ms)
        }

        let mut split_ns = 0.0;
        let mut fused_ns = 0.0;

        if run_split {
            let (_, ns, _) = run_mode("3-kernel (split)", &gpu, &nuc_data, &mat_data,
                                       n, args.batches, args.inactive, false);
            split_ns = ns;
        }
        if run_fused {
            let (_, ns, _) = run_mode("Fused + compaction", &gpu, &nuc_data, &mat_data,
                                       n, args.batches, args.inactive, true);
            fused_ns = ns;
        }

        if run_split && run_fused {
            println!("\n{}", "=".repeat(60));
            println!("COMPARISON");
            println!("{}", "=".repeat(60));
            println!("  Split:    {split_ns:.2} ns/particle");
            println!("  Fused:    {fused_ns:.2} ns/particle");
            println!("  Speedup:  {:.2}x", split_ns / fused_ns);
        }
    }

    fn setup_materials() -> Vec<open_rust_mc::transport::material::Material> {
        use open_rust_mc::transport::material::Material;

        let mut fuel = Material::new("UO2", 900.0);
        fuel.add_nuclide(0.00072, 0);  // U-235
        fuel.add_nuclide(0.02219, 1);  // U-238
        fuel.add_nuclide(0.04582, 2);  // O-16

        let mut clad = Material::new("Zircaloy", 600.0);
        clad.add_nuclide(0.02189, 4);  // Zr-90
        clad.add_nuclide(0.00477, 5);  // Zr-91
        clad.add_nuclide(0.00729, 6);  // Zr-92
        clad.add_nuclide(0.00739, 7);  // Zr-94

        let mut water = Material::new("H2O", 600.0);
        water.add_nuclide(0.04937, 3);  // H-1
        water.add_nuclide(0.02469, 2);  // O-16

        vec![fuel, clad, water]
    }
}
