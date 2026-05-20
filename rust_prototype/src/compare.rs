// SPDX-License-Identifier: MIT
//! Comparison between original and SVD-reconstructed cross-sections.

/// Error statistics for a single comparison.
#[derive(Debug, Clone)]
pub struct ErrorStats {
    pub label: String,
    pub n_points: usize,
    pub max_rel_err: f64,
    pub mean_rel_err: f64,
    pub p99_rel_err: f64,
}

/// Per-region error breakdown.
pub struct RegionalErrors {
    pub thermal: ErrorStats,   // E < 1 eV
    pub resonance: ErrorStats, // 1 eV – 25 keV
    pub fast: ErrorStats,      // > 25 keV
    pub overall: ErrorStats,
}

/// Compute relative error statistics between original and reconstructed.
///
/// Both are in linear scale (barns). `energies` is the energy grid for
/// regional breakdown.
pub fn compare(
    original: &[f64],
    reconstructed: &[f64],
    energies: &[f64],
    n_e: usize,
    n_t: usize,
    temp_labels: &[String],
) -> Vec<(String, RegionalErrors)> {
    assert_eq!(original.len(), n_e * n_t);
    assert_eq!(reconstructed.len(), n_e * n_t);

    let mut results = Vec::new();

    for t in 0..n_t {
        let label = if t < temp_labels.len() {
            temp_labels[t].clone()
        } else {
            format!("T[{t}]")
        };

        let mut errs_thermal = Vec::new();
        let mut errs_resonance = Vec::new();
        let mut errs_fast = Vec::new();
        let mut errs_all = Vec::new();

        for i in 0..n_e {
            let orig = original[i * n_t + t];
            let recon = reconstructed[i * n_t + t];
            let rel_err = if orig.abs() > 1e-30 {
                ((orig - recon) / orig).abs()
            } else {
                0.0
            };

            errs_all.push(rel_err);
            let e = energies[i];
            if e < 1.0 {
                errs_thermal.push(rel_err);
            } else if e < 25_000.0 {
                errs_resonance.push(rel_err);
            } else {
                errs_fast.push(rel_err);
            }
        }

        results.push((
            label,
            RegionalErrors {
                thermal: compute_stats("Thermal (<1eV)", &errs_thermal),
                resonance: compute_stats("Resonance (1eV-25keV)", &errs_resonance),
                fast: compute_stats("Fast (>25keV)", &errs_fast),
                overall: compute_stats("Overall", &errs_all),
            },
        ));
    }
    results
}

fn compute_stats(label: &str, errors: &[f64]) -> ErrorStats {
    if errors.is_empty() {
        return ErrorStats {
            label: label.to_string(),
            n_points: 0,
            max_rel_err: 0.0,
            mean_rel_err: 0.0,
            p99_rel_err: 0.0,
        };
    }

    let n = errors.len();
    let max = errors.iter().copied().fold(0.0_f64, f64::max);
    let mean = errors.iter().sum::<f64>() / n as f64;

    let mut sorted = errors.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_idx = ((n as f64) * 0.99).ceil() as usize;
    let p99 = sorted[p99_idx.min(n - 1)];

    ErrorStats {
        label: label.to_string(),
        n_points: n,
        max_rel_err: max,
        mean_rel_err: mean,
        p99_rel_err: p99,
    }
}

/// Print a formatted comparison report.
pub fn print_report(comparisons: &[(String, RegionalErrors)], k: usize) {
    println!("\n{}", "=".repeat(80));
    println!("SVD RECONSTRUCTION COMPARISON (k={k})");
    println!("{}", "=".repeat(80));

    for (label, regional) in comparisons {
        println!("\n  --- {label} ---");
        for stats in [
            &regional.thermal,
            &regional.resonance,
            &regional.fast,
            &regional.overall,
        ] {
            if stats.n_points == 0 {
                continue;
            }
            println!(
                "    {:<25} max={:.2e}  mean={:.2e}  P99={:.2e}  (n={})",
                stats.label,
                stats.max_rel_err,
                stats.mean_rel_err,
                stats.p99_rel_err,
                stats.n_points
            );
        }
    }
}
