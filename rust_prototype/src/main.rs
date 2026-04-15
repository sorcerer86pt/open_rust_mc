//! open_rust_mc — SVD cross-section reconstruction engine.
//!
//! Two modes:
//!   hdf5  — Read an OpenMC HDF5 file, decompose via SVD, reconstruct, compare.
//!   npy   — Load pre-computed SVD factors from Python, benchmark reconstruction.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use open_rust_mc::compare;
use open_rust_mc::decompose;
use open_rust_mc::hdf5_reader::NuclideData as Hdf5Nuclide;
use open_rust_mc::kernel;
use open_rust_mc::loader::SvdFactors;
use open_rust_mc::table::PointwiseTable;

#[derive(Parser)]
#[command(name = "open_rust_mc", about = "SVD cross-section reconstruction engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Full pipeline: read HDF5, decompose, reconstruct, validate.
    Hdf5 {
        /// Path to OpenMC HDF5 nuclide file (e.g. U235.h5).
        path: PathBuf,
        /// ENDF reaction MT number (18=fission, 102=capture, 2=elastic).
        #[arg(short, long, default_value_t = 18)]
        mt: u32,
        /// Maximum SVD rank to test.
        #[arg(short, long, default_value_t = 7)]
        max_rank: usize,
    },
    /// Explore the structure of an HDF5 file.
    Explore {
        /// Path to HDF5 file.
        path: PathBuf,
    },
    /// Benchmark using pre-computed numpy SVD factors.
    Npy {
        /// Directory containing the .npy output files.
        #[arg(short, long, default_value = "")]
        dir: PathBuf,
        /// File prefix (e.g. "jeff33_").
        #[arg(short, long, default_value = "")]
        prefix: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Hdf5 { path, mt, max_rank } => run_hdf5(&path, mt, max_rank),
        Command::Explore { path } => run_explore(&path),
        Command::Npy { dir, prefix } => run_npy(&dir, &prefix),
    }
}

// ─── HDF5 full pipeline ────────────────────────────────────────────────────

fn run_hdf5(path: &PathBuf, mt: u32, max_rank: usize) {
    println!("=== open_rust_mc — HDF5 Full Pipeline ===\n");
    println!("File: {}", path.display());
    println!("Reaction: MT={mt}");

    // 1. Read HDF5
    let t0 = Instant::now();
    let data = match Hdf5Nuclide::from_hdf5(path, mt) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read HDF5: {e}");
            std::process::exit(1);
        }
    };
    let read_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let n_e = data.n_energy();
    let n_t = data.n_temp();
    println!("\n  Read in {read_ms:.0} ms: N_E={n_e}, N_T={n_t}");
    println!("  Temperatures: {:?}", data.temp_labels);
    println!(
        "  Energy range: {:.4e} – {:.4e} eV",
        data.energies[0],
        data.energies[n_e - 1]
    );

    // 2. Build matrices
    let log_matrix = data.to_log_matrix();
    let lin_matrix = data.to_linear_matrix();

    // 3. SVD (via faer)
    let t1 = Instant::now();
    let svd_result = decompose::svd(&log_matrix, n_e, n_t);
    let svd_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("\n  SVD computed in {svd_ms:.0} ms (rank = {})", svd_result.rank);
    println!("  Singular values:");
    for (j, &s) in svd_result.s.iter().enumerate() {
        let ratio = s / svd_result.s[0];
        println!("    σ_{} = {:.6e}  (σ_{}/σ_1 = {:.6e})", j + 1, s, j + 1, ratio);
    }

    // Cumulative energy
    let s2_total: f64 = svd_result.s.iter().map(|s| s * s).sum();
    let mut cum = 0.0_f64;
    println!("\n  Cumulative energy:");
    for (j, &s) in svd_result.s.iter().enumerate() {
        cum += s * s;
        println!("    k={}: {:.8}%", j + 1, cum / s2_total * 100.0);
    }

    // 4. Reconstruct and compare at various ranks
    let effective_max = max_rank.min(svd_result.rank);
    for k in 2..=effective_max {
        let recon_log = svd_result.reconstruct_log(k);

        // Convert to linear scale
        let recon_lin: Vec<f64> = recon_log.iter().map(|&v| 10.0_f64.powf(v)).collect();

        // Compare
        let results = compare::compare(
            &lin_matrix,
            &recon_lin,
            &data.energies,
            n_e,
            n_t,
            &data.temp_labels,
        );
        compare::print_report(&results, k);
    }

    // 5. Benchmark reconstruction speed
    println!("\n{}", "=".repeat(80));
    println!("RECONSTRUCTION BENCHMARK");
    println!("{}", "=".repeat(80));

    let iters = 100_u32;
    let table = PointwiseTable::from_vecs(data.energies.clone(), data.xs_per_temp[0].clone());

    for k in [3, 4, 5, 6].iter().copied().filter(|&k| k <= effective_max) {
        // Build kernel from SVD result
        let kern = build_kernel_from_svd(&svd_result, &data.energies, k);
        let coeffs = kern.temp_coeffs(0);
        let mut buf = vec![0.0_f64; n_e];

        // Manual FMA
        let t0 = Instant::now();
        for _ in 0..iters {
            kern.reconstruct_log(&coeffs, &mut buf);
        }
        let manual_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        // faer SIMD
        let t1 = Instant::now();
        for _ in 0..iters {
            kernel::reconstruct_log_faer(&kern, &coeffs, &mut buf);
        }
        let faer_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        // Table lookup
        let mut tbl_buf = vec![0.0_f64; n_e];
        let t2 = Instant::now();
        for _ in 0..iters {
            table.batch_lookup(&data.energies, &mut tbl_buf);
        }
        let table_us = t2.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        let speedup = table_us / manual_us;
        println!(
            "\n  k={k}: manual={manual_us:.0}µs  faer={faer_us:.0}µs  \
             table={table_us:.0}µs  speedup={speedup:.1}x"
        );
        println!(
            "        {:.2} ns/pt (manual)  {:.2} ns/pt (table)  kernel={:.1} KB",
            manual_us * 1000.0 / n_e as f64,
            table_us * 1000.0 / n_e as f64,
            kern.memory_bytes() as f64 / 1024.0,
        );
    }
}

/// Build an `SvdKernel` from a `decompose::SvdResult` (no numpy files needed).
fn build_kernel_from_svd(
    svd: &decompose::SvdResult,
    energies: &[f64],
    k: usize,
) -> open_rust_mc::kernel::SvdKernel {
    let rank = k.min(svd.rank);
    let n_e = svd.n_e;
    let n_t = svd.n_t;

    // Pre-multiply basis[i,j] = U[i,j] * S[j]
    let mut basis = vec![0.0_f64; n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }

    // V^T truncated to rank
    let mut vt_coeffs = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt_coeffs[j * n_t + t] = svd.vt[j * n_t + t];
        }
    }

    open_rust_mc::kernel::SvdKernel::new(
        basis,
        vt_coeffs,
        energies.to_vec().into(),
        rank,
        n_e,
        n_t,
    )
}

// ─── Explore HDF5 structure ────────────────────────────────────────────────

fn run_explore(path: &PathBuf) {
    let file = hdf5_pure::File::open(path).expect("cannot open HDF5 file");
    println!("HDF5 file: {}\n", path.display());
    explore_group(&file.root(), "", 0);
}

fn explore_group(group: &hdf5_pure::Group<'_>, path: &str, depth: usize) {
    let indent = "  ".repeat(depth);

    if let Ok(attrs) = group.attrs() {
        for (k, v) in &attrs {
            println!("{indent}  @{k} = {v:?}");
        }
    }

    if let Ok(datasets) = group.datasets() {
        for name in &datasets {
            let ds_path = if path.is_empty() { name.clone() } else { format!("{path}/{name}") };
            if let Ok(ds) = group.dataset(name) {
                let shape = ds.shape().unwrap_or_default();
                let dtype = ds.dtype().map(|d| format!("{d:?}")).unwrap_or_else(|_| "?".into());
                println!("{indent}  [{ds_path}] shape={shape:?} dtype={dtype}");
            }
        }
    }

    if let Ok(subgroups) = group.groups() {
        for name in &subgroups {
            let sub_path = if path.is_empty() { name.clone() } else { format!("{path}/{name}") };
            println!("{indent}  {sub_path}/");
            if let Ok(sg) = group.group(name) {
                explore_group(&sg, &sub_path, depth + 1);
            }
        }
    }
}

// ─── NPY benchmark mode ───────────────────────────────────────────────────

fn run_npy(dir: &PathBuf, prefix: &str) {
    let dir = if dir.as_os_str().is_empty() {
        default_output_dir()
    } else {
        dir.clone()
    };

    println!("=== open_rust_mc — NPY Benchmark Mode ===\n");

    let t0 = Instant::now();
    let factors = match SvdFactors::load(&dir, prefix) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to load: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let n_e = factors.energies.len();
    let full_rank = factors.s.len();
    let n_t = factors.vt.dim().1;

    println!("Loaded in {load_ms:.1} ms: N_E={n_e}, rank={full_rank}, N_T={n_t}");
    println!("Singular values: {:?}\n", &factors.s);

    let iters = 100_u32;

    for k in (2..=full_rank.min(7)).collect::<Vec<_>>() {
        let kern = factors.clone().into_kernel(k);
        let coeffs = kern.temp_coeffs(0);
        let mut buf = vec![0.0_f64; n_e];

        let t0 = Instant::now();
        for _ in 0..iters {
            kern.reconstruct_log(&coeffs, &mut buf);
        }
        let manual_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        let t1 = Instant::now();
        for _ in 0..iters {
            kernel::reconstruct_log_faer(&kern, &coeffs, &mut buf);
        }
        let faer_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        println!(
            "k={k}: manual={manual_us:.0}µs ({:.2}ns/pt)  faer={faer_us:.0}µs ({:.2}ns/pt)  mem={:.1}KB",
            manual_us * 1000.0 / n_e as f64,
            faer_us * 1000.0 / n_e as f64,
            kern.memory_bytes() as f64 / 1024.0,
        );
    }
}

fn default_output_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("HOME or USERPROFILE must be set");
    PathBuf::from(home)
        .join("madman_svd_experiment")
        .join("outputs")
}
