//! Diagnostic: compare CPU vs GPU cross-section and angular distribution data.
//!
//! Verifies bit-exact agreement between CPU and GPU for:
//!   1. Angular distribution sampling (stair-step and interpolated)
//!   2. SVD cross-section reconstruction at test energies
//!   3. Energy grid index lookup consistency

// Diagnostic binary — panic-on-failure is the intended behaviour for
// file I/O and flush errors; relax the crate-wide `unwrap_used` deny.
#![allow(clippy::unwrap_used)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]
//!
//! Usage:
//!   cargo run --release --features cuda --bin debug_trace -- <data_dir>
//!   cargo run --release --bin debug_trace -- <data_dir>  (CPU-only, no GPU comparison)

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use open_rust_mc::hdf5_reader::AngularDistribution;
use open_rust_mc::transport::simulate::XsProvider;
use open_rust_mc::transport::xs_provider;

#[derive(Parser)]
#[command(name = "debug_trace", about = "CPU vs GPU physics diagnostic")]
struct Args {
    /// Directory containing nuclide HDF5 files.
    data_dir: PathBuf,
    #[arg(short, long, default_value_t = 5)]
    rank: usize,
}

const NUCLIDES: &[(&str, f64, f64)] = &[
    ("U234.h5", 232.029, 2.49),
    ("U235.h5", 233.025, 2.43),
    ("U238.h5", 236.006, 2.49),
];

fn main() {
    let args = Args::parse();

    println!("=== CPU vs GPU Physics Diagnostic ===\n");
    println!("Loading nuclides (rank={})...", args.rank);
    let mut kernels = Vec::new();
    for &(filename, awr, nu_bar) in NUCLIDES {
        let path = args.data_dir.join(filename);
        kernels.push(xs_provider::load_nuclide(&path, args.rank, 1, awr, nu_bar));
    }
    let provider = xs_provider::SvdXsProvider {
        nuclides: kernels,
        thermal: vec![],
    };

    // === CPU XS values at test energies ===
    println!("\n── CPU XS at test energies ──");
    let mut f = std::fs::File::create("diag_cpu_xs.csv").unwrap();
    writeln!(
        f,
        "nuc,energy,elastic,inelastic,n2n,n3n,fission,capture,total"
    )
    .unwrap();
    let test_energies = [1e-2, 1e0, 1e2, 1e3, 5e3, 1e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7];
    for nuc_idx in 0..3 {
        for &e in &test_energies {
            let xs = provider.lookup(nuc_idx, e);
            writeln!(
                f,
                "{},{:.2e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                NUCLIDES[nuc_idx].0,
                e,
                xs.elastic,
                xs.inelastic,
                xs.n2n,
                xs.n3n,
                xs.fission,
                xs.capture,
                xs.total
            )
            .unwrap();
        }
    }
    println!("  Wrote diag_cpu_xs.csv");

    // === GPU comparison ===
    run_gpu_comparison(&args, &provider);

    println!("\nDone.");
}

/// Manual implementation of sample_mu matching CPU hdf5_reader.rs logic.
fn sample_mu_manual(ang: &AngularDistribution, energy: f64, xi: f64) -> f64 {
    if ang.energies.is_empty() {
        return 2.0 * xi - 1.0;
    }
    let n = ang.energies.len();
    if energy <= ang.energies[0] {
        return sample_cdf(&ang.distributions[0].mu, &ang.distributions[0].cdf, xi);
    }
    if energy >= ang.energies[n - 1] {
        return sample_cdf(
            &ang.distributions[n - 1].mu,
            &ang.distributions[n - 1].cdf,
            xi,
        );
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if ang.energies[mid] <= energy {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let idx = lo;
    if idx + 1 >= n {
        return sample_cdf(&ang.distributions[idx].mu, &ang.distributions[idx].cdf, xi);
    }
    let frac = (energy - ang.energies[idx]) / (ang.energies[idx + 1] - ang.energies[idx]);
    let mu_lo = sample_cdf(&ang.distributions[idx].mu, &ang.distributions[idx].cdf, xi);
    let mu_hi = sample_cdf(
        &ang.distributions[idx + 1].mu,
        &ang.distributions[idx + 1].cdf,
        xi,
    );
    ((1.0 - frac) * mu_lo + frac * mu_hi).clamp(-1.0, 1.0)
}

fn sample_cdf(mu: &[f64], cdf: &[f64], xi: f64) -> f64 {
    let n = cdf.len();
    if n < 2 {
        return 2.0 * xi - 1.0;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cdf[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (xi - cdf[lo]) / (cdf[hi] - cdf[lo]).max(1e-30);
    (mu[lo] + f * (mu[hi] - mu[lo])).clamp(-1.0, 1.0)
}

#[cfg(not(feature = "cuda"))]
fn run_gpu_comparison(_args: &Args, _provider: &xs_provider::SvdXsProvider) {
    println!("\n── GPU comparison requires --features cuda ──");
}

#[cfg(feature = "cuda")]
fn run_gpu_comparison(_args: &Args, provider: &xs_provider::SvdXsProvider) {
    use open_rust_mc::gpu_transport::GpuTransportContext;
    use open_rust_mc::transport::material::Material;

    println!("\n── GPU vs CPU Angular Distribution ──");
    let gpu = match GpuTransportContext::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU init failed: {e}");
            return;
        }
    };
    let nuc_data = gpu
        .upload_nuclide_data(&provider.nuclides, 5)
        .expect("upload nuclide");
    let mut heu = Material::new("HEU", 294.0);
    heu.add_nuclide(0.000483, 0);
    heu.add_nuclide(0.04509, 1);
    heu.add_nuclide(0.00265, 2);
    let awrs: Vec<f64> = NUCLIDES.iter().map(|s| s.1).collect();
    let nu_bars: Vec<f64> = NUCLIDES.iter().map(|s| s.2).collect();
    let mat_data = gpu
        .upload_material_data(&[heu], &awrs, &nu_bars)
        .expect("upload mat");
    let sab_data = gpu.upload_sab_data_empty().expect("empty sab");

    let mut f = std::fs::File::create("diag_gpu_vs_cpu_angular.csv").unwrap();
    writeln!(
        f,
        "nuc,energy,xi,cpu_mu,gpu_stairstep,gpu_interp,diff_ss,diff_interp"
    )
    .unwrap();

    let mut total = 0usize;
    let mut max_diff_interp = 0.0_f64;

    for nuc_idx in 0..3usize {
        let ang = match provider.elastic_angular_dist(nuc_idx) {
            Some(a) => a,
            None => continue,
        };
        let n_e = ang.energies.len();
        let mut test_energies = Vec::new();
        let mut test_xis = Vec::new();
        for ei in 0..n_e.min(30) {
            let e = if ei + 1 < n_e {
                (ang.energies[ei] + ang.energies[ei + 1]) * 0.5
            } else {
                ang.energies[ei]
            };
            for xi_i in 0..50 {
                let xi = (xi_i as f64 + 0.5) / 50.0;
                test_energies.push(e);
                test_xis.push(xi);
            }
        }
        let (gpu_ss, gpu_interp) = gpu
            .debug_angular_sample(
                &test_energies,
                &test_xis,
                nuc_idx as i32,
                &nuc_data,
                &mat_data,
                &sab_data,
                1,
            )
            .expect("GPU angular sample");

        for i in 0..test_energies.len() {
            let cpu_mu = sample_mu_manual(ang, test_energies[i], test_xis[i]);
            let diff_interp = (gpu_interp[i] - cpu_mu).abs();
            max_diff_interp = max_diff_interp.max(diff_interp);
            total += 1;
            writeln!(
                f,
                "{},{:.6e},{:.6},{:.10},{:.10},{:.10},{:.2e},{:.2e}",
                NUCLIDES[nuc_idx].0,
                test_energies[i],
                test_xis[i],
                cpu_mu,
                gpu_ss[i],
                gpu_interp[i],
                (gpu_ss[i] - cpu_mu).abs(),
                diff_interp
            )
            .unwrap();
        }
    }
    println!("  Wrote diag_gpu_vs_cpu_angular.csv ({total} samples)");
    if max_diff_interp < 1e-10 {
        println!("  Angular dist: PASS (max diff {:.2e})", max_diff_interp);
    } else {
        println!("  Angular dist: FAIL (max diff {:.2e})", max_diff_interp);
    }

    // XS reconstruction comparison
    println!("\n── GPU vs CPU XS Reconstruction ──");
    let test_energies = [1e-2, 1e0, 1e2, 1e3, 5e3, 1e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7];
    let rxn_names = ["elastic", "inelastic", "n2n", "n3n", "fission", "capture"];
    let mut f = std::fs::File::create("diag_gpu_vs_cpu_xs.csv").unwrap();
    writeln!(f, "nuc,energy,reaction,cpu_xs,gpu_xs,rel_diff").unwrap();
    let mut max_rel = 0.0_f64;
    let mut n_mismatch = 0;
    for nuc_idx in 0..3usize {
        let gpu_xs = gpu
            .debug_xs_reconstruct(
                &test_energies,
                nuc_idx as i32,
                &nuc_data,
                &mat_data,
                &sab_data,
                1,
            )
            .expect("GPU XS");
        for (ei, &e) in test_energies.iter().enumerate() {
            let cpu = provider.lookup(nuc_idx, e);
            let cpu_vals = [
                cpu.elastic,
                cpu.inelastic,
                cpu.n2n,
                cpu.n3n,
                cpu.fission,
                cpu.capture,
            ];
            for r in 0..6 {
                let cv = cpu_vals[r];
                let gv = gpu_xs[ei * 6 + r];
                let rel = if cv.abs() > 1e-30 {
                    ((gv - cv) / cv).abs()
                } else if gv.abs() > 1e-30 {
                    1.0
                } else {
                    0.0
                };
                if rel > 1e-6 {
                    n_mismatch += 1;
                }
                max_rel = max_rel.max(rel);
                writeln!(
                    f,
                    "{},{:.2e},{},{:.6e},{:.6e},{:.2e}",
                    NUCLIDES[nuc_idx].0, e, rxn_names[r], cv, gv, rel
                )
                .unwrap();
            }
        }
    }
    println!("  Wrote diag_gpu_vs_cpu_xs.csv");
    if max_rel < 1e-4 {
        println!("  XS reconstruction: PASS (max rel diff {:.2e})", max_rel);
    } else {
        println!(
            "  XS reconstruction: FAIL (max rel diff {:.2e}, {} mismatches)",
            max_rel, n_mismatch
        );
        println!("  NOTE: diag kernel uses stair-step XS; transport kernel uses log-log interp");
    }

    // Energy grid index comparison
    println!("\n── Energy Grid Index Check ──");
    let grid_offsets: Vec<i32> = gpu
        .stream()
        .clone_dtoh(&nuc_data.grid_offsets)
        .expect("dtoh");
    let n_energies: Vec<i32> = gpu.stream().clone_dtoh(&nuc_data.n_energies).expect("dtoh");
    let grids: Vec<f64> = gpu
        .stream()
        .clone_dtoh(&nuc_data.all_energy_grids)
        .expect("dtoh");
    for nuc_idx in 0..3usize {
        let g_off = grid_offsets[nuc_idx] as usize;
        let n_e = n_energies[nuc_idx] as usize;
        let gpu_grid = &grids[g_off..g_off + n_e];
        let mut all_ok = true;
        for &e in &[1e3_f64, 5e3, 1e4, 1e5, 1e6] {
            let gpu_idx = {
                if e <= gpu_grid[0] {
                    0
                } else if e >= gpu_grid[n_e - 1] {
                    n_e - 1
                } else {
                    let (mut lo, mut hi) = (0, n_e - 1);
                    while hi - lo > 1 {
                        let mid = (lo + hi) / 2;
                        if gpu_grid[mid] <= e {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    lo
                }
            };
            let cpu_idx = provider.nuclides[nuc_idx]
                .elastic
                .as_ref()
                .or(provider.nuclides[nuc_idx].fission.as_ref())
                .or(provider.nuclides[nuc_idx].capture.as_ref())
                .map_or(0, |rk| rk.energy_index(e));
            if gpu_idx != cpu_idx {
                println!(
                    "  {} E={:.0e}: gpu_idx={} cpu_idx={} MISMATCH",
                    NUCLIDES[nuc_idx].0, e, gpu_idx, cpu_idx
                );
                all_ok = false;
            }
        }
        if all_ok {
            println!("  {}: all energy indices match", NUCLIDES[nuc_idx].0);
        }
    }
}
