//! Pareto benchmark: SVD accuracy vs reconstruction speed, per rank.
//!
//! For each rank k in {2, 3, 4, 5, 6}:
//!   - Build the rank-k SVD kernel for each (nuclide, reaction) pair.
//!   - Compute RMSE of reconstructed log10(sigma) vs ground-truth HDF5 data,
//!     pooled across all temperatures and all energy points.
//!   - Also report mean relative error and max absolute log10 error.
//!   - Benchmark: ns/lookup = time for `energy_index + reconstruct_single`
//!     averaged over N random energy queries.
//!
//! Also benchmarks the pointwise table (log-log interp, binary search) as
//! the baseline "table" point on the Pareto plot.
//!
//! Output: CSV to stdout with one row per (rank, nuclide, reaction) and a
//! "table" row per (nuclide, reaction). Columns:
//!   kind,rank,nuclide,mt,n_e,n_t,rmse_log10,max_abs_log10,mean_rel_err,ns_per_lookup,mem_bytes
//!
//! Usage:
//!   cargo run --release --bin pareto_bench -- <data_dir>

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use open_rust_mc::decompose;
use open_rust_mc::hdf5_reader::NuclideData;
use open_rust_mc::kernel::SvdKernel;
use open_rust_mc::table::PointwiseTable;
use open_rust_mc::transport::rng::Rng;

const BENCH_LOOKUPS: usize = 2_000_000;
const RANKS: &[usize] = &[2, 3, 4, 5, 6];

/// (filename, mt, label)
const TARGETS: &[(&str, u32, &str)] = &[
    ("U235.h5", 2,   "U235_elastic"),
    ("U235.h5", 18,  "U235_fission"),
    ("U235.h5", 102, "U235_capture"),
    ("U238.h5", 2,   "U238_elastic"),
    ("U238.h5", 18,  "U238_fission"),
    ("U238.h5", 102, "U238_capture"),
    ("U234.h5", 2,   "U234_elastic"),
    ("U234.h5", 102, "U234_capture"),
];

fn build_kernel(svd: &decompose::SvdResult, energies: &[f64], k: usize) -> SvdKernel {
    let rank = k.min(svd.rank);
    let mut basis = vec![0.0_f64; svd.n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..svd.n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }
    let mut vt_coeffs = vec![0.0_f64; rank * svd.n_t];
    for j in 0..rank {
        for t in 0..svd.n_t {
            vt_coeffs[j * svd.n_t + t] = svd.vt[j * svd.n_t + t];
        }
    }
    SvdKernel::new(basis, vt_coeffs, energies.to_vec().into(), rank, svd.n_e, svd.n_t)
}

/// Compute accuracy metrics over all (E, T) points.
fn accuracy(kernel: &SvdKernel, data: &NuclideData) -> (f64, f64, f64) {
    let n_e = kernel.n_energy();
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut sum_rel = 0.0_f64;
    let mut n_points = 0_usize;
    let mut buf = vec![0.0_f64; n_e];

    for t in 0..data.n_temp() {
        let coeffs = kernel.temp_coeffs(t);
        kernel.reconstruct_linear(&coeffs, &mut buf);
        for i in 0..n_e {
            let truth = data.xs_per_temp[t][i];
            if truth <= 0.0 { continue; }
            let recon = buf[i].max(1e-30);
            let d = recon.log10() - truth.log10();
            sum_sq += d * d;
            if d.abs() > max_abs { max_abs = d.abs(); }
            sum_rel += ((recon - truth) / truth).abs();
            n_points += 1;
        }
    }

    if n_points == 0 { return (0.0, 0.0, 0.0); }
    let rmse = (sum_sq / n_points as f64).sqrt();
    let mean_rel = sum_rel / n_points as f64;
    (rmse, max_abs, mean_rel)
}

/// Generate uniform-log random energies within the grid range.
fn random_log_energies(grid: &[f64], n: usize) -> Vec<f64> {
    let e_min = grid[0].max(1e-5);
    let e_max = grid[grid.len() - 1].min(2.0e7);
    let log_min = e_min.ln();
    let log_max = e_max.ln();
    let mut rng = Rng::new(0xDEAD_BEEF_0000_0001, 0);
    (0..n).map(|_| {
        let u = rng.uniform();
        (log_min + u * (log_max - log_min)).exp()
    }).collect()
}

/// Time SVD lookup (hash index + reconstruct_single, linear scale).
fn bench_svd_lookup(kernel: &SvdKernel, energies: &[f64]) -> f64 {
    let coeffs = kernel.temp_coeffs(0);
    // Warmup
    let mut acc = 0.0_f64;
    for &e in energies.iter().take(1000) {
        let idx = kernel.energy_index(e);
        acc += kernel.reconstruct_single(idx, &coeffs);
    }
    black_box(acc);

    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for &e in energies {
        let idx = kernel.energy_index(black_box(e));
        acc += kernel.reconstruct_single(idx, &coeffs);
    }
    black_box(acc);
    let elapsed = t0.elapsed();
    (elapsed.as_nanos() as f64) / (energies.len() as f64)
}

/// Time pointwise table lookup (binary search / hash + log-log interp).
fn bench_table_lookup(table: &PointwiseTable, energies: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &e in energies.iter().take(1000) {
        acc += table.lookup(e);
    }
    black_box(acc);

    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for &e in energies {
        acc += table.lookup(black_box(e));
    }
    black_box(acc);
    let elapsed = t0.elapsed();
    (elapsed.as_nanos() as f64) / (energies.len() as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: pareto_bench <data_dir>");
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[0]);

    // CSV header to stdout so shell redirection captures only data.
    println!("kind,rank,nuclide,mt,n_e,n_t,rmse_log10,max_abs_log10,mean_rel_err,ns_per_lookup,mem_bytes");

    for (filename, mt, label) in TARGETS {
        let path = data_dir.join(filename);
        eprintln!("\n=== {label} ({filename} MT={mt}) ===");

        let data = match NuclideData::from_hdf5(&path, *mt) {
            Ok(d) => d,
            Err(e) => { eprintln!("  SKIP: {e}"); continue; }
        };
        let n_e = data.n_energy();
        let n_t = data.n_temp();

        // Full SVD once, reuse truncations.
        let log_mat = data.to_log_matrix();
        let svd = decompose::svd(&log_mat, n_e, n_t);

        // Random energies for timing (independent of rank).
        let queries = random_log_energies(&data.energies, BENCH_LOOKUPS);

        // Table baseline: use temp index 0 XS as the reference table.
        let xs_col = data.xs_per_temp[0].clone();
        let table = PointwiseTable::from_vecs(data.energies.clone(), xs_col);
        let table_ns = bench_table_lookup(&table, &queries);
        let table_mem = n_e * 8; // xs values; energy grid shared per nuclide
        println!("table,0,{label},{mt},{n_e},{n_t},0,0,0,{:.3},{}",
                 table_ns, table_mem);
        eprintln!("  table:   ns/lookup = {:.2}", table_ns);

        for &k in RANKS {
            if k > svd.rank { continue; }
            let kernel = build_kernel(&svd, &data.energies, k);
            let (rmse, max_abs, mean_rel) = accuracy(&kernel, &data);
            let ns = bench_svd_lookup(&kernel, &queries);
            let mem = kernel.memory_bytes();
            println!("svd,{k},{label},{mt},{n_e},{n_t},{:.3e},{:.3e},{:.3e},{:.3},{}",
                     rmse, max_abs, mean_rel, ns, mem);
            eprintln!("  k={k}: rmse_log10 = {:.2e}  ns/lookup = {:.2}  mean_rel = {:.2e}",
                      rmse, ns, mean_rel);
        }
    }
}
