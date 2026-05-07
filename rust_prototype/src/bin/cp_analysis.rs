//! CP/PARAFAC analysis probe.
//!
//! Builds the σ(E, T, ℓ) 3-tensor for one or more nuclides from raw
//! HDF5 data, decomposes it via greedy rank-1 power iteration, and
//! reports per-rank reconstruction error. Compares the joint CP
//! representation memory cost to the per-level rank-1 SVD baseline
//! we currently ship.
//!
//! Usage:
//!   cargo run --release --bin cp_analysis -- <data_dir>
//!
//! Prints a table per nuclide of the form:
//!
//!     rank  rel_L2     max_abs    mem_KB    vs_per_level_svd
//!        1  3.7e-02    1.2e+00    32.4     0.18×  (smaller is better)
//!        2  ...
//!     ...

use std::path::PathBuf;

use clap::Parser;

use open_rust_mc::cp_decompose::{
    CpDecomposition, cp_greedy_rank1, max_abs_error, relative_l2_error,
};
use open_rust_mc::hdf5_reader::NuclideFileReader;

#[derive(Parser)]
#[command(name = "cp_analysis", about = "CP/PARAFAC probe on σ(E, T, ℓ)")]
struct Args {
    /// Directory containing OpenMC HDF5 nuclide files.
    data_dir: PathBuf,
    /// Nuclides to analyse (comma-separated). Default: Zr clad + U-238.
    #[arg(long, default_value = "Zr90,Zr91,Zr92,Zr94,U238")]
    nuclides: String,
    /// Maximum CP rank to fit.
    #[arg(long, default_value_t = 10)]
    max_rank: usize,
    /// Power-iteration cap per rank component.
    #[arg(long, default_value_t = 200)]
    max_iter: usize,
    /// Optional: log-decimate the per-level data to this many energy
    /// points before decomposition (matches the CDF convention; faster
    /// to fit and what would actually be deployed). 0 disables.
    #[arg(long, default_value_t = 200)]
    decimate_e: usize,
}

fn main() {
    let args = Args::parse();
    let nuclide_names: Vec<&str> = args.nuclides.split(',').collect();

    println!("=== CP/PARAFAC analysis on σ(E, T, ℓ) ===\n");
    println!(
        "  decimation: {}\n  max_rank:   {}\n",
        if args.decimate_e == 0 {
            "full grid".to_string()
        } else {
            format!("{} log-spaced points", args.decimate_e)
        },
        args.max_rank
    );

    for nuclide in &nuclide_names {
        let h5_path = args.data_dir.join(format!("{nuclide}.h5"));
        if !h5_path.exists() {
            println!("  SKIP {} (not found)", h5_path.display());
            continue;
        }
        analyse_nuclide(nuclide, &h5_path, &args);
    }
}

fn analyse_nuclide(name: &str, path: &std::path::Path, args: &Args) {
    println!("── {name} ─────────────────────────────────────────────");
    let reader = match NuclideFileReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            println!("  open failed: {e}");
            return;
        }
    };

    // Discover discrete-level MTs from HDF5 metadata.
    let levels = reader.discrete_levels(0.0); // awr unused for MT discovery
    let level_mts: Vec<u32> = levels
        .iter()
        .map(|l| l.mt)
        .filter(|&mt| (51..=91).contains(&mt))
        .collect();
    if level_mts.is_empty() {
        println!("  no discrete inelastic levels — skipping");
        return;
    }

    // Read each level's xs_per_temp on the per-nuclide union grid.
    let mut per_level: Vec<Vec<Vec<f64>>> = Vec::with_capacity(level_mts.len()); // [l][t][e]
    let mut kept_mts: Vec<u32> = Vec::with_capacity(level_mts.len());
    let mut energies: Vec<f64> = Vec::new();
    let mut temperatures: Vec<f64> = Vec::new();
    for &mt in &level_mts {
        match reader.read_reaction(mt) {
            Ok(d) if d.n_energy() > 0 && d.n_temp() > 0 => {
                if energies.is_empty() {
                    energies = d.energies.clone();
                    temperatures = d.temperatures.clone();
                }
                per_level.push(d.xs_per_temp);
                kept_mts.push(mt);
            }
            _ => {}
        }
    }
    if per_level.is_empty() {
        println!("  could not read any level reactions");
        return;
    }

    let n_e_full = energies.len();
    let n_t = temperatures.len();
    let n_l = per_level.len();
    println!(
        "  shape (full): n_e = {n_e_full}, n_t = {n_t}, n_l = {n_l} \
         (MTs {} - {})",
        kept_mts.first().unwrap(),
        kept_mts.last().unwrap()
    );

    // Optional log-decimation (matches the InelasticCdf convention).
    let (n_e, decimated): (usize, Vec<Vec<Vec<f64>>>) =
        if args.decimate_e > 0 && args.decimate_e < n_e_full {
            let dec = decimate_per_level(&per_level, &energies, args.decimate_e);
            (args.decimate_e, dec)
        } else {
            (n_e_full, per_level.clone())
        };
    println!("  shape (decomposed): n_e = {n_e}, n_t = {n_t}, n_l = {n_l}");

    // Build the σ(E, T, ℓ) 3-tensor flat: tensor[i * n_t * n_l + t * n_l + l]
    let mut tensor = vec![0.0_f64; n_e * n_t * n_l];
    for l in 0..n_l {
        for t in 0..n_t {
            for i in 0..n_e {
                tensor[i * n_t * n_l + t * n_l + l] = decimated[l][t][i].max(0.0);
            }
        }
    }

    // Tensor norm for normalised error reporting (and a sanity print).
    let tensor_l2: f64 = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("  ||σ||_F = {:.4e}", tensor_l2);

    // Decompose at max_rank, reuse for every truncation k <= max_rank.
    let cp = cp_greedy_rank1(&tensor, n_e, n_t, n_l, args.max_rank, args.max_iter, 1e-9);

    // Reference: per-level rank-1 SVD memory (current production for
    // synthesised nuclides — n_l rank-1 SVDs).
    let per_level_svd_bytes = n_l * (n_e + n_t) * std::mem::size_of::<f64>();

    println!("\n  rank  rel_L2     max_abs       mem_KB     vs_per_level_svd");
    println!("  ----  ---------  ----------    --------   ----------------");
    for k in 1..=cp.rank {
        let recon = cp.reconstruct(k);
        let l2 = relative_l2_error(&tensor, &recon);
        let abs_err = max_abs_error(&tensor, &recon);
        // Truncated CP memory: k components, each (n_e + n_t + n_l)
        // doubles plus one σ scalar.
        let cp_bytes = k * (n_e + n_t + n_l + 1) * std::mem::size_of::<f64>();
        let ratio = cp_bytes as f64 / per_level_svd_bytes as f64;
        println!(
            "  {:>4}  {:>9.2e}  {:>10.2e}    {:>7.1}    {:>5.2}× (CP/SVD)",
            k,
            l2,
            abs_err,
            cp_bytes as f64 / 1024.0,
            ratio
        );
    }
    println!(
        "  per-level rank-1 SVD baseline memory: {:.1} KB ({n_l} levels × \
         (n_e + n_t) × 8 bytes)",
        per_level_svd_bytes as f64 / 1024.0
    );
    println!();
}

/// Log-decimate the (n_l, n_t, n_e) per-level XS to a smaller energy
/// axis with `n_dec` log-spaced points spanning the union grid range.
/// Linear interpolation in raw E between bracketing samples — same
/// convention used by InelasticCdf.
fn decimate_per_level(
    per_level: &[Vec<Vec<f64>>],
    energies: &[f64],
    n_dec: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n_l = per_level.len();
    let n_t = per_level[0].len();
    let mut e_min = f64::INFINITY;
    let mut e_max = f64::NEG_INFINITY;
    for &e in energies {
        if e > 0.0 {
            if e < e_min {
                e_min = e;
            }
            if e > e_max {
                e_max = e;
            }
        }
    }
    let log_e_min = e_min.log10();
    let log_e_max = e_max.log10();

    let bsearch = |e: f64| -> (usize, f64) {
        if e <= energies[0] {
            return (0, 0.0);
        }
        if e >= energies[energies.len() - 1] {
            return (energies.len() - 1, 0.0);
        }
        let mut lo = 0usize;
        let mut hi = energies.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if energies[mid] <= e {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = energies[hi] - energies[lo];
        let alpha = if span > 0.0 {
            (e - energies[lo]) / span
        } else {
            0.0
        };
        (lo, alpha)
    };

    let mut out = vec![vec![vec![0.0; n_dec]; n_t]; n_l];
    for ed in 0..n_dec {
        let frac = if n_dec == 1 {
            0.0
        } else {
            ed as f64 / (n_dec - 1) as f64
        };
        let log_e = log_e_min + frac * (log_e_max - log_e_min);
        let e = 10f64.powf(log_e);
        let (idx, alpha) = bsearch(e);
        let nxt = (idx + 1).min(energies.len() - 1);
        for l in 0..n_l {
            for t in 0..n_t {
                let lo = per_level[l][t][idx].max(0.0);
                let hi = per_level[l][t][nxt].max(0.0);
                out[l][t][ed] = lo + alpha * (hi - lo);
            }
        }
    }
    out
}

#[allow(dead_code)]
fn dump(_cp: &CpDecomposition) {
    // placeholder — keeps CpDecomposition referenced for the binary
}
