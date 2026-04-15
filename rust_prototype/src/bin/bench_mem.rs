//! Memory and speed comparison: SVD kernel vs. pointwise tables.
use std::path::PathBuf;
use std::time::Instant;
use std::fs;
use std::io::{self, Write};

use open_rust_mc::decompose;
use open_rust_mc::hdf5_reader::NuclideData;
use open_rust_mc::kernel::SvdKernel;
use open_rust_mc::table::PointwiseTable;

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() {
        eprintln!("Usage: bench_mem <folder_or_file.h5> [--rank K]");
        std::process::exit(1);
    }

    let mut paths = Vec::new();
    let mut target_rank = 5;
    
    // Robust argument parsing
    let mut i = 0;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--rank" => {
                if i + 1 < raw_args.len() {
                    target_rank = raw_args[i+1].parse().unwrap_or(5);
                    i += 2;
                } else { i += 1; }
            }
            path_str => {
                let p = PathBuf::from(path_str);
                if p.is_dir() {
                    if let Ok(entries) = fs::read_dir(p) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                            if path.extension().and_then(|s| s.to_str()) == Some("h5") && !filename.starts_with("c_") {
                                paths.push(path);
                            }
                        }
                    }
                } else if p.exists() {
                    paths.push(p);
                }
                i += 1;
            }
        }
    }

    println!("=== open_rust_mc — Memory Report (Target Rank: {target_rank}) ===\n");
    io::stdout().flush().unwrap();

    let mut total_table_bytes = 0_usize;
    let mut total_svd_bytes = 0_usize;
    let mut n = 0;

    for path in &paths {
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "?".into());
        print!("Loading {:<10}... ", name);
        io::stdout().flush().unwrap();

        let data = match NuclideData::from_hdf5(path, 18_u32) { 
            Ok(d) => d,
            Err(_) => match NuclideData::from_hdf5(path, 2_u32) { 
                Ok(d) => d, 
                Err(_) => { println!("SKIP"); continue; }
            }
        };

        let n_e = data.n_energy();
        let n_t = data.n_temp();
        let log_mat = data.to_log_matrix();
        let svd = decompose::svd(&log_mat, n_e, n_t);
        let kern = build_kernel(&svd, &data.energies, target_rank);
        
        let table_bytes = n_e * n_t * 16;
        let svd_bytes = kern.memory_bytes();
        
        total_table_bytes += table_bytes;
        total_svd_bytes += svd_bytes;
        n += 1;

        println!("done. SVD is {:.1}x smaller", table_bytes as f64 / svd_bytes as f64);
        io::stdout().flush().unwrap();
    }

    if n > 0 {
        let tbl_mb = total_table_bytes as f64 / 1_048_576.0;
        let svd_mb = total_svd_bytes as f64 / 1_048_576.0;
        
        println!("\nAGGREGATE ESTIMATE (Total library):");
        // FIXED: Added {n} to the format string below
        println!("  {n:>4} nuclides × ~50 reactions:  table={:.0} MB  SVD(k={})={:.0} MB  ratio={:.1}x",
                 tbl_mb * 50.0, target_rank, svd_mb * 50.0, tbl_mb / svd_mb);
        io::stdout().flush().unwrap();
    }
}

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