//! χ sampling diagnostic.
//!
//! Loads U-235's fission energy distribution and samples it under
//! three schemes:
//!  - CPU `TabularEnergyDist::sample_with_xi`         (quadratic, ref)
//!  - linear-CDF       (the pre-PDF GPU implementation)
//!  - GPU-style quadratic — the same formula now in transport.cu's
//!    `sample_eout_bin`, ported back to Rust here for bit-for-bit
//!    comparison against the CPU.
//!
//! For each scheme reports ⟨E_out⟩, ⟨E_out²⟩, and the full sampled
//! distribution histogram. The two quadratic implementations should
//! be bit-identical — if they differ, something in the GPU port
//! diverges from the CPU.

use std::path::PathBuf;

use open_rust_mc::hdf5_reader::{read_fission_energy_dist, TabularEnergyDist};
use open_rust_mc::transport::rng::Rng;

fn data_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return PathBuf::from(v);
    }
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("data/endfb-vii.1-hdf5/neutron").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("data/endfb-vii.1-hdf5/neutron")
}

/// CPU quadratic — verbatim port of `TabularEnergyDist::sample_with_xi`
/// in `hdf5_reader.rs`. Sample_with_xi is private; this is a faithful
/// duplicate for the diagnostic.
fn sample_cpu_quadratic(xi: f64, dist: &TabularEnergyDist) -> f64 {
    let n = dist.cdf.len();
    if n < 2 {
        return dist.e_out.first().copied().unwrap_or(1.0e6);
    }
    let idx = match dist
        .cdf
        .binary_search_by(|c| c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => i,
        Err(i) => {
            if i > 0 {
                i - 1
            } else {
                0
            }
        }
    };
    let idx = idx.min(n - 2);
    let cdf_lo = dist.cdf[idx];
    let cdf_hi = dist.cdf[idx + 1];
    let e_lo = dist.e_out[idx];
    let e_hi = dist.e_out[idx + 1];
    let de = e_hi - e_lo;
    if (cdf_hi - cdf_lo).abs() < 1e-15 {
        return e_lo.max(1e-5);
    }
    if dist.pdf.len() == n && de > 0.0 {
        let p_lo = dist.pdf[idx];
        let p_hi = dist.pdf[idx + 1];
        let m = (p_hi - p_lo) / de;
        let dc = xi - cdf_lo;
        let e = if m.abs() < 1e-30 {
            if p_lo.abs() < 1e-30 {
                e_lo
            } else {
                e_lo + dc / p_lo
            }
        } else {
            let disc = p_lo * p_lo + 2.0 * m * dc;
            if disc < 0.0 {
                e_lo
            } else {
                e_lo + (disc.sqrt() - p_lo) / m
            }
        };
        return e.max(1e-5);
    }
    let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
    (e_lo + frac * de).max(1e-5)
}

/// Linear-CDF (pre-PDF GPU). Histogram approximation.
fn sample_linear(xi: f64, eo: &[f64], cd: &[f64]) -> f64 {
    let n = eo.len();
    if n <= 1 {
        return eo[0];
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cd[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (xi - cd[lo]) / (cd[hi] - cd[lo]).max(1e-30);
    eo[lo] + f * (eo[hi] - eo[lo])
}

/// Faithful port of the GPU's `sample_eout_bin` (quadratic with PDF).
fn sample_gpu_quadratic(xi: f64, eo: &[f64], cd: &[f64], pd: &[f64]) -> f64 {
    let n = eo.len();
    if n <= 1 {
        return eo[0];
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cd[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let e_lo = eo[lo];
    let e_hi = eo[hi];
    let cdf_lo = cd[lo];
    let cdf_hi = cd[hi];
    let de = e_hi - e_lo;
    if (cdf_hi - cdf_lo).abs() < 1e-15 {
        return e_lo;
    }
    if de > 0.0 {
        let p_lo = pd[lo];
        let p_hi = pd[hi];
        let dc = xi - cdf_lo;
        if p_lo > 0.0 || p_hi > 0.0 {
            let m = (p_hi - p_lo) / de;
            if m.abs() < 1e-30 {
                if p_lo > 0.0 {
                    return e_lo + dc / p_lo;
                }
            } else {
                let disc = p_lo * p_lo + 2.0 * m * dc;
                if disc >= 0.0 {
                    return e_lo + (disc.sqrt() - p_lo) / m;
                }
            }
        }
    }
    let f = (xi - cdf_lo) / (cdf_hi - cdf_lo).max(1e-30);
    e_lo + f * de
}

fn moments(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn run_bin(dist: &TabularEnergyDist, label: &str) {
    println!("\n=== {label} ===");
    println!(
        "  e_out range = [{:.3e}, {:.3e}] eV, {} pts; pdf.len = {}",
        dist.e_out[0],
        dist.e_out.last().unwrap(),
        dist.e_out.len(),
        dist.pdf.len()
    );

    let mut rng = Rng::new(42, 0);
    let n = 200_000;
    let mut s_cpu = Vec::with_capacity(n);
    let mut s_lin = Vec::with_capacity(n);
    let mut s_gpu = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.uniform().clamp(1e-12, 1.0 - 1e-12);
        s_cpu.push(sample_cpu_quadratic(xi, dist).max(1e-5));
        s_lin.push(sample_linear(xi, &dist.e_out, &dist.cdf).max(1e-5));
        s_gpu.push(sample_gpu_quadratic(xi, &dist.e_out, &dist.cdf, &dist.pdf).max(1e-5));
    }
    let (mean_cpu, std_cpu) = moments(&s_cpu);
    let (mean_lin, std_lin) = moments(&s_lin);
    let (mean_gpu, std_gpu) = moments(&s_gpu);
    println!("  CPU quadratic : ⟨E⟩ = {:.4e}  σ = {:.4e}", mean_cpu, std_cpu);
    println!("  Linear-CDF    : ⟨E⟩ = {:.4e}  σ = {:.4e}", mean_lin, std_lin);
    println!("  GPU quadratic : ⟨E⟩ = {:.4e}  σ = {:.4e}", mean_gpu, std_gpu);

    // Bit-for-bit check on the first 8 samples.
    println!("  First 8 samples (same ξ stream, fresh rng):");
    let mut rng2 = Rng::new(42, 0);
    for k in 0..8 {
        let xi = rng2.uniform().clamp(1e-12, 1.0 - 1e-12);
        let v_cpu = sample_cpu_quadratic(xi, dist);
        let v_gpu = sample_gpu_quadratic(xi, &dist.e_out, &dist.cdf, &dist.pdf);
        let v_lin = sample_linear(xi, &dist.e_out, &dist.cdf);
        let cpu_gpu_diff = (v_cpu - v_gpu).abs();
        println!(
            "    [{k}] ξ={:.4}  cpu={:.4e}  gpu={:.4e}  Δ={:.2e}  lin={:.4e}",
            xi, v_cpu, v_gpu, cpu_gpu_diff, v_lin
        );
    }
}

fn main() {
    let data_dir = data_dir();
    let path = data_dir.join("U235.h5");
    println!("Loading {}", path.display());
    let edist = read_fission_energy_dist(&path)
        .expect("U235 χ load")
        .expect("U235 must have χ");
    println!(
        "U-235 χ: {} incident-energy bins, range [{:.3e}, {:.3e}] eV",
        edist.energies.len(),
        edist.energies[0],
        edist.energies.last().unwrap()
    );

    // Pick representative E_in indices.
    let n = edist.energies.len();
    for &idx in &[0_usize, n / 4, n / 2, 3 * n / 4, n - 1] {
        run_bin(
            &edist.distributions[idx],
            &format!("E_in = {:.3e} eV (bin {idx})", edist.energies[idx]),
        );
    }
}
