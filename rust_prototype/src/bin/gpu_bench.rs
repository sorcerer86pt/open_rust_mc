// SPDX-License-Identifier: MIT
//! GPU SVD reconstruction benchmark.
//!
//! Loads real nuclear data, reconstructs cross-sections on GPU,
//! and compares throughput with CPU. Supports single-nuclide (U-235)
//! and multi-nuclide PWR pin cell (8 nuclides) modes.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_bench -- <data_dir> [--rank K] [--particles N] [--pwr]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: This binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_bench -- ...");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    cuda_main::run();
}

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;

    use open_rust_mc::gpu::GpuSvdContext;
    use open_rust_mc::transport::xs_provider;

    #[derive(Parser)]
    #[command(name = "gpu_bench", about = "GPU SVD reconstruction benchmark")]
    struct Args {
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        rank: usize,
        #[arg(short, long, default_value_t = 1_000_000)]
        particles: usize,
        #[arg(short, long, default_value_t = 1)]
        temp_idx: usize,
        /// Run PWR pin cell multi-nuclide benchmark (8 nuclides × 6 reactions)
        #[arg(long)]
        pwr: bool,
    }

    /// PWR pin cell nuclide specs (same as pwr_pincell.rs)
    const PWR_NUCLIDES: &[(&str, f64, f64)] = &[
        ("U235.h5", 233.025, 2.43),
        ("U238.h5", 236.006, 2.49),
        ("O16.h5", 15.858, 0.0),
        ("H1.h5", 0.999, 0.0),
        ("Zr90.h5", 89.132, 0.0),
        ("Zr91.h5", 90.130, 0.0),
        ("Zr92.h5", 91.126, 0.0),
        ("Zr94.h5", 93.120, 0.0),
    ];

    pub fn run() {
        let args = Args::parse();

        println!("=== GPU SVD Reconstruction Benchmark ===\n");

        if args.pwr {
            run_pwr_benchmark(&args);
        } else {
            run_single_nuclide_benchmark(&args);
        }
    }

    fn run_single_nuclide_benchmark(args: &Args) {
        // Load nuclear data (U-235 fission)
        let u235_path = args.data_dir.join("U235.h5");
        println!("Loading {} (rank={})...", u235_path.display(), args.rank);
        let kernels =
            xs_provider::load_nuclide(&u235_path, args.rank, args.temp_idx, 233.025, 2.43);

        let fission_rxn = kernels.fission.as_ref().expect("U-235 must have fission");
        let (basis, coeffs, n_e, rank) = match fission_rxn {
            xs_provider::ReactionKernel::Svd { kernel, coeffs } => {
                (kernel.basis_f64(), coeffs, kernel.n_energy(), kernel.rank())
            }
            xs_provider::ReactionKernel::Table { .. } => {
                panic!("gpu_bench expects SVD U-235 fission kernel")
            }
        };
        let fission = fission_rxn;

        println!("  N_E = {n_e}, rank = {rank}");
        println!(
            "  Basis: {} elements ({:.1} MB f64)",
            basis.len(),
            basis.len() as f64 * 4.0 / 1e6
        );

        // Generate random energy indices
        let energy_indices = generate_random_indices(args.particles, n_e);

        // CPU baseline
        println!("\nCPU reconstruction ({} particles)...", args.particles);
        let t0 = Instant::now();
        let cpu_results: Vec<f64> = energy_indices
            .iter()
            .map(|&idx| fission.reconstruct_at_index(idx as usize))
            .collect();
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cpu_ns = cpu_ms * 1e6 / args.particles as f64;
        println!("  CPU: {cpu_ms:.1} ms ({cpu_ns:.1} ns/particle)");

        // GPU
        println!("\nInitializing GPU...");
        let gpu = match GpuSvdContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  Failed to initialize GPU: {e}");
                return;
            }
        };

        // Warmup
        let _ = gpu.reconstruct_batch(
            basis,
            coeffs,
            &energy_indices[..1000.min(args.particles)],
            n_e,
            rank,
        );

        println!("GPU reconstruction ({} particles)...", args.particles);
        let t1 = Instant::now();
        let gpu_results = gpu
            .reconstruct_batch(basis, coeffs, &energy_indices, n_e, rank)
            .expect("GPU reconstruction failed");
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let gpu_ns = gpu_ms * 1e6 / args.particles as f64;
        println!("  GPU: {gpu_ms:.1} ms ({gpu_ns:.1} ns/particle)");

        // Comparison
        print_comparison(
            "U-235 fission",
            cpu_ns,
            gpu_ns,
            &cpu_results,
            &gpu_results,
            args.particles,
        );
    }

    fn run_pwr_benchmark(args: &Args) {
        println!("PWR pin cell mode: 8 nuclides, all reactions\n");

        // Load all nuclides
        let mut all_kernels = Vec::new();
        for &(filename, awr, nu_bar) in PWR_NUCLIDES {
            let path = args.data_dir.join(filename);
            if !path.exists() {
                eprintln!("  WARNING: {} not found, skipping", path.display());
                continue;
            }
            let k = xs_provider::load_nuclide(&path, args.rank, args.temp_idx, awr, nu_bar);
            all_kernels.push((filename, k));
        }

        // Initialize GPU
        println!("\nInitializing GPU...");
        let gpu = match GpuSvdContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  Failed to initialize GPU: {e}");
                return;
            }
        };

        let n_particles = args.particles;
        let mut total_cpu_ns = 0.0;
        let mut total_gpu_ns = 0.0;
        let mut total_reactions = 0_u32;

        println!(
            "\n{:<12} {:<10} {:>10} {:>10} {:>8} {:>10}",
            "Nuclide", "Reaction", "CPU ns/p", "GPU ns/p", "Speedup", "Max err"
        );
        println!("{}", "-".repeat(70));

        for (filename, kernels) in &all_kernels {
            let nuclide_name = filename.strip_suffix(".h5").unwrap_or(filename);

            // Benchmark each reaction that has a kernel
            let reactions: Vec<(&str, Option<&xs_provider::ReactionKernel>)> = vec![
                ("elastic", kernels.elastic.as_ref()),
                ("fission", kernels.fission.as_ref()),
                ("capture", kernels.capture.as_ref()),
                ("inelast", kernels.inelastic.as_ref()),
                ("n2n", kernels.n2n.as_ref()),
            ];

            for (rxn_name, kernel_opt) in reactions {
                let kernel = match kernel_opt {
                    Some(k) => k,
                    None => continue,
                };

                let (basis, coeffs, n_e, rank) = match kernel {
                    xs_provider::ReactionKernel::Svd { kernel: k, coeffs: c } => {
                        (k.basis_f64(), c, k.n_energy(), k.rank())
                    }
                    xs_provider::ReactionKernel::Table { .. } => continue,
                };

                let energy_indices = generate_random_indices(n_particles, n_e);

                // CPU
                let t0 = Instant::now();
                let cpu_results: Vec<f64> = energy_indices
                    .iter()
                    .map(|&idx| kernel.reconstruct_at_index(idx as usize))
                    .collect();
                let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let cpu_ns = cpu_ms * 1e6 / n_particles as f64;

                // GPU
                let _ = gpu
                    .reconstruct_batch(basis, coeffs, &energy_indices, n_e, rank)
                    .expect("GPU failed");
                // Timed run (after warmup from above)
                let t1 = Instant::now();
                let gpu_results = gpu
                    .reconstruct_batch(basis, coeffs, &energy_indices, n_e, rank)
                    .expect("GPU failed");
                let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
                let gpu_ns = gpu_ms * 1e6 / n_particles as f64;

                // Max relative error
                let max_err = cpu_results
                    .iter()
                    .zip(gpu_results.iter())
                    .map(|(&c, &g)| (c - g).abs() / c.max(1e-30))
                    .fold(0.0_f64, f64::max);

                let speedup = cpu_ns / gpu_ns;
                println!(
                    "{:<12} {:<10} {:>10.1} {:>10.1} {:>7.1}x {:>10.2e}",
                    nuclide_name, rxn_name, cpu_ns, gpu_ns, speedup, max_err
                );

                total_cpu_ns += cpu_ns;
                total_gpu_ns += gpu_ns;
                total_reactions += 1;
            }
        }

        println!("{}", "-".repeat(70));
        let avg_cpu = total_cpu_ns / total_reactions as f64;
        let avg_gpu = total_gpu_ns / total_reactions as f64;
        let avg_speedup = avg_cpu / avg_gpu;
        println!(
            "{:<12} {:<10} {:>10.1} {:>10.1} {:>7.1}x",
            "AVERAGE",
            &format!("({total_reactions} rxn)"),
            avg_cpu,
            avg_gpu,
            avg_speedup
        );

        println!("\nPWR collision cost estimate ({n_particles} particles):");
        println!(
            "  CPU: {:.1} ns/particle (sum of all nuclides/reactions)",
            total_cpu_ns
        );
        println!(
            "  GPU: {:.1} ns/particle (sum of all nuclides/reactions)",
            total_gpu_ns
        );
        println!("  Speedup: {:.1}x", total_cpu_ns / total_gpu_ns);
    }

    fn generate_random_indices(n: usize, n_e: usize) -> Vec<i32> {
        let mut rng_state = 42_u64;
        (0..n)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as usize % n_e) as i32
            })
            .collect()
    }

    fn print_comparison(
        label: &str,
        cpu_ns: f64,
        gpu_ns: f64,
        cpu_results: &[f64],
        gpu_results: &[f64],
        n: usize,
    ) {
        let speedup = cpu_ns / gpu_ns;
        println!("\n{}", "=".repeat(50));
        println!("  {label}:");
        println!("  CPU:     {cpu_ns:.1} ns/particle");
        println!("  GPU:     {gpu_ns:.1} ns/particle");
        println!("  Speedup: {speedup:.1}x");

        // Verify correctness
        let mut max_err = 0.0_f64;
        for i in 0..n.min(10) {
            let err = (gpu_results[i] - cpu_results[i]).abs() / cpu_results[i].max(1e-30);
            max_err = max_err.max(err);
            if i < 5 {
                println!(
                    "  [{i}] CPU={:.6e}  GPU={:.6e}  err={:.2e}",
                    cpu_results[i], gpu_results[i], err
                );
            }
        }
        println!(
            "  Max relative error (first {}): {:.2e}",
            n.min(10),
            max_err
        );
    }
}
