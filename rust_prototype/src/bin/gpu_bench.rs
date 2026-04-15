//! GPU SVD reconstruction benchmark.
//!
//! Loads real nuclear data, reconstructs cross-sections on GPU,
//! and compares throughput with CPU.
//!
//! Usage:
//!   cargo run --release --features cuda --bin gpu_bench -- <data_dir> [--rank K] [--particles N]

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
    }

    pub fn run() {
        let args = Args::parse();

        println!("=== GPU SVD Reconstruction Benchmark ===\n");

        // Load nuclear data (U-235 fission)
        let u235_path = args.data_dir.join("U235.h5");
        println!("Loading {} (rank={})...", u235_path.display(), args.rank);
        let kernels = xs_provider::load_nuclide(
            &u235_path, args.rank, args.temp_idx, 233.025, 2.43,
        );

        let fission = kernels.fission.as_ref().expect("U-235 must have fission");
        let basis = fission.kernel.basis_f32();
        let coeffs = &fission.coeffs;
        let n_e = fission.kernel.n_energy();
        let rank = fission.kernel.rank();

        println!("  N_E = {n_e}, rank = {rank}");
        println!("  Basis: {} elements ({:.1} MB f32)",
                 basis.len(), basis.len() as f64 * 4.0 / 1e6);

        // Generate random energy indices (simulating particle lookups)
        let mut rng_state = 42_u64;
        let energy_indices: Vec<i32> = (0..args.particles)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as usize % n_e) as i32
            })
            .collect();

        // ── CPU baseline ──
        println!("\nCPU reconstruction ({} particles)...", args.particles);
        let t0 = Instant::now();
        let cpu_results: Vec<f64> = energy_indices.iter()
            .map(|&idx| fission.reconstruct_at_index(idx as usize))
            .collect();
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let cpu_ns = cpu_ms * 1e6 / args.particles as f64;
        println!("  CPU: {cpu_ms:.1} ms ({cpu_ns:.1} ns/particle)");

        // ── GPU ──
        println!("\nInitializing GPU...");
        let gpu = match GpuSvdContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  Failed to initialize GPU: {e}");
                eprintln!("  Make sure CUDA toolkit is installed and a GPU is available.");
                return;
            }
        };

        // Warmup
        let _ = gpu.reconstruct_batch(basis, coeffs, &energy_indices[..1000], n_e, rank);

        println!("GPU reconstruction ({} particles)...", args.particles);
        let t1 = Instant::now();
        let gpu_results = gpu.reconstruct_batch(basis, coeffs, &energy_indices, n_e, rank)
            .expect("GPU reconstruction failed");
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let gpu_ns = gpu_ms * 1e6 / args.particles as f64;
        println!("  GPU: {gpu_ms:.1} ms ({gpu_ns:.1} ns/particle)");

        // ── Comparison ──
        let speedup = cpu_ms / gpu_ms;
        println!("\n{}", "=".repeat(50));
        println!("  CPU:     {cpu_ns:.1} ns/particle");
        println!("  GPU:     {gpu_ns:.1} ns/particle");
        println!("  Speedup: {speedup:.1}x");

        // Verify correctness (first 10 values)
        let mut max_err = 0.0_f64;
        for i in 0..args.particles.min(10) {
            let err = (gpu_results[i] - cpu_results[i]).abs() / cpu_results[i].max(1e-30);
            max_err = max_err.max(err);
            if i < 5 {
                println!("  [{i}] CPU={:.6e}  GPU={:.6e}  err={:.2e}", cpu_results[i], gpu_results[i], err);
            }
        }
        println!("  Max relative error (first {}): {:.2e}", args.particles.min(10), max_err);
    }
}
