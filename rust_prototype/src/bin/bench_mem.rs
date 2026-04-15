//! Memory and speed comparison: SVD kernel vs. pointwise tables.
//!
//! Loads one or more nuclide HDF5 files and reports:
//!   - Per-nuclide: table size vs SVD kernel size at each rank
//!   - Aggregate: total memory for N nuclides
//!   - Speed: reconstruction throughput vs table lookup throughput
//!   - Projections: estimated savings for a full reactor model (~400 nuclides)
//!
//! Usage:
//!   open-rust-mc-bench-mem <U235.h5> [U238.h5] [Pu239.h5] ...

use std::path::PathBuf;
use std::time::Instant;

use open_rust_mc::decompose;
use open_rust_mc::hdf5_reader::NuclideData;
use open_rust_mc::kernel::SvdKernel;
use open_rust_mc::table::PointwiseTable;

fn main() {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("Usage: bench_mem <U235.h5> [U238.h5] ...");
        std::process::exit(1);
    }

    println!("=== open_rust_mc — Memory & Speed Report ===\n");

    let mt = 18_u32; // fission
    let mut total_table_bytes = 0_usize;
    let mut total_svd_bytes = [0_usize; 6]; // k=2..7
    let mut nuclide_reports: Vec<NuclideReport> = Vec::new();

    for path in &paths {
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "?".into());

        print!("Loading {name}... ");
        let t0 = Instant::now();
        let data = match NuclideData::from_hdf5(path, mt) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("\n  SKIP ({e})");
                continue;
            }
        };
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let n_e = data.n_energy();
        let n_t = data.n_temp();

        // Table memory: energy grid + xs per temperature (all temps)
        let table_bytes = n_e * n_t * 2 * 8; // energy + xs per temp
        total_table_bytes += table_bytes;

        // SVD
        let log_mat = data.to_log_matrix();
        let t1 = Instant::now();
        let svd = decompose::svd(&log_mat, n_e, n_t);
        let svd_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Benchmark one temperature
        let table = PointwiseTable::from_vecs(data.energies.clone(), data.xs_per_temp[0].clone());
        let iters = 100_u32;

        println!("done ({load_ms:.0}ms load, {svd_ms:.0}ms SVD, N_E={n_e}, N_T={n_t})");

        let mut rank_reports = Vec::new();
        for (ki, &k) in [2_usize, 3, 4, 5, 6, 7].iter().enumerate() {
            if k > svd.rank { continue; }

            let kern = build_kernel(&svd, &data.energies, k);
            let svd_bytes = kern.memory_bytes();
            total_svd_bytes[ki] += svd_bytes;

            let coeffs = kern.temp_coeffs(0);
            let mut buf = vec![0.0_f64; n_e];

            // SVD throughput
            let t0 = Instant::now();
            for _ in 0..iters {
                kern.reconstruct_log(&coeffs, &mut buf);
            }
            let svd_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

            // Table throughput
            let mut tbl_buf = vec![0.0_f64; n_e];
            let t1 = Instant::now();
            for _ in 0..iters {
                table.batch_lookup(&data.energies, &mut tbl_buf);
            }
            let tbl_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

            rank_reports.push(RankReport {
                k,
                svd_bytes,
                svd_us,
                tbl_us,
            });
        }

        nuclide_reports.push(NuclideReport {
            name,
            n_e,
            n_t,
            table_bytes,
            singular_values: svd.s.clone(),
            ranks: rank_reports,
        });
    }

    // ── Print report ─────────────────────────────────────────────────────

    println!("\n{}", "=".repeat(90));
    println!("PER-NUCLIDE SUMMARY");
    println!("{}", "=".repeat(90));

    for nr in &nuclide_reports {
        println!("\n  {} (N_E={}, N_T={})", nr.name, nr.n_e, nr.n_t);
        println!("  Table: {:.1} KB", nr.table_bytes as f64 / 1024.0);
        println!("  σ values: {:?}", &nr.singular_values[..nr.singular_values.len().min(4)]);
        println!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
                 "k", "SVD KB", "Table KB", "Ratio", "SVD µs", "Speedup");
        for rr in &nr.ranks {
            let ratio = nr.table_bytes as f64 / rr.svd_bytes as f64;
            let speedup = rr.tbl_us / rr.svd_us;
            println!("  {:>4}  {:>10.1}  {:>10.1}  {:>9.1}x  {:>8.0}  {:>7.1}x",
                     rr.k,
                     rr.svd_bytes as f64 / 1024.0,
                     nr.table_bytes as f64 / 1024.0,
                     ratio, rr.svd_us, speedup);
        }
    }

    // ── Aggregate ────────────────────────────────────────────────────────

    let n_nuclides = nuclide_reports.len();
    println!("\n{}", "=".repeat(90));
    println!("AGGREGATE ({n_nuclides} nuclide(s) loaded)");
    println!("{}", "=".repeat(90));
    println!("  Total table memory:  {:.1} MB", total_table_bytes as f64 / 1024.0 / 1024.0);
    for (ki, &k) in [2_usize, 3, 4, 5, 6, 7].iter().enumerate() {
        if total_svd_bytes[ki] == 0 { continue; }
        let ratio = total_table_bytes as f64 / total_svd_bytes[ki] as f64;
        println!("  SVD k={k} memory:     {:.1} MB  ({ratio:.1}x smaller)",
                 total_svd_bytes[ki] as f64 / 1024.0 / 1024.0);
    }

    // ── Projections ──────────────────────────────────────────────────────

    if !nuclide_reports.is_empty() {
        let avg_table = total_table_bytes as f64 / n_nuclides as f64;
        let avg_svd_k5 = if total_svd_bytes[3] > 0 {
            total_svd_bytes[3] as f64 / n_nuclides as f64
        } else {
            total_svd_bytes[2] as f64 / n_nuclides as f64
        };

        println!("\n{}", "=".repeat(90));
        println!("PROJECTIONS (typical reactor model)");
        println!("{}", "=".repeat(90));
        for &n in &[50, 200, 400] {
            let tbl_mb = avg_table * n as f64 / 1024.0 / 1024.0;
            let svd_mb = avg_svd_k5 * n as f64 / 1024.0 / 1024.0;
            println!("  {n:>4} nuclides × ~50 reactions:  table={:.0} MB  SVD(k=5)={:.0} MB  ratio={:.1}x",
                     tbl_mb * 50.0, svd_mb * 50.0, tbl_mb / svd_mb);
        }
    }

    // ── Process memory ───────────────────────────────────────────────────

    #[cfg(windows)]
    print_process_memory();
}

struct NuclideReport {
    name: String,
    n_e: usize,
    n_t: usize,
    table_bytes: usize,
    singular_values: Vec<f64>,
    ranks: Vec<RankReport>,
}

struct RankReport {
    k: usize,
    svd_bytes: usize,
    svd_us: f64,
    tbl_us: f64,
}

fn build_kernel(svd: &decompose::SvdResult, energies: &[f64], k: usize) -> SvdKernel {
    let rank = k.min(svd.rank);
    let n_e = svd.n_e;
    let n_t = svd.n_t;

    let mut basis = vec![0.0_f64; n_e * rank];
    for j in 0..rank {
        let s_j = svd.s[j];
        for i in 0..n_e {
            basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
        }
    }

    let mut vt_coeffs = vec![0.0_f64; rank * n_t];
    for j in 0..rank {
        for t in 0..n_t {
            vt_coeffs[j * n_t + t] = svd.vt[j * n_t + t];
        }
    }

    SvdKernel::new(basis, vt_coeffs, energies.to_vec().into(), rank, n_e, n_t)
}

#[cfg(windows)]
fn print_process_memory() {
    // Use Windows API to get working set size
    use std::mem::MaybeUninit;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            ppsmemCounters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    unsafe {
        let mut pmc = MaybeUninit::<ProcessMemoryCounters>::zeroed().assume_init();
        pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
        let handle = GetCurrentProcess();
        if K32GetProcessMemoryInfo(handle, &mut pmc, pmc.cb) != 0 {
            println!("\n  Process memory:");
            println!("    Working set:      {:.1} MB", pmc.WorkingSetSize as f64 / 1024.0 / 1024.0);
            println!("    Peak working set: {:.1} MB", pmc.PeakWorkingSetSize as f64 / 1024.0 / 1024.0);
            println!("    Pagefile usage:   {:.1} MB", pmc.PagefileUsage as f64 / 1024.0 / 1024.0);
        }
    }
}
