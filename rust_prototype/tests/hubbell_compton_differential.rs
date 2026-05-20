#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
// SPDX-License-Identifier: MIT
//! Hubbell-1975 differential Compton cross-section validation.
//!
//! The bound-electron incoherent differential cross section is
//! (per unit `μ = cos θ`, up to normalisation):
//!
//! ```text
//!   dσ_inc/dμ ∝ k²(μ) · (k + 1/k − 1 + μ²) · S(x(μ), Z) / Z
//! ```
//!
//! where `k = 1/(1 + α(1 − μ))`, `α = E/m_e c²`,
//! `x = (E/hc) · √((1 − μ)/2)` in inverse Ångström, and `S(x, Z)` is
//! the tabulated incoherent scattering function (Hubbell et al.,
//! J. Phys. Chem. Ref. Data 4, 471 (1975)).
//!
//! This integration test:
//!   1. Samples `N_MC = 200 000` Compton events on Pb at 100 keV
//!      using the production Compton kernel (KN + S(x,Z)/Z rejection).
//!   2. Histograms sampled `μ` into 40 uniform bins on [−1, 1].
//!   3. Computes the analytic PDF from the formula above, normalised
//!      over the same 40 bins by Simpson integration.
//!   4. Asserts bin-by-bin agreement to within statistical tolerance
//!      (chi-squared reduced χ² / ν ≤ 2 at 40 degrees of freedom,
//!      standard acceptance criterion for MC sampler validation).

use std::path::PathBuf;

use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::compton::{HC_EV_ANGSTROM, M_E_C2_EV, compton_scatter};
use open_rust_mc::transport::rng::Rng;

fn photon_path(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest
        .parent()?
        .join("data/endfb-vii.1-hdf5/photon")
        .join(name);
    if p.exists() { Some(p) } else { None }
}

/// Linear interpolation on a strictly monotonic grid. Clamps below /
/// above endpoints. Used only in the test for reading the tabulated
/// `S(x, Z)` values at arbitrary `x`; must match the kernel's own
/// linear interpolation so the test validates the sampler rather than
/// interpolation-vs-interpolation.
fn interp_linear(grid: &[f64], vals: &[f64], x: f64) -> f64 {
    if x <= grid[0] {
        return vals[0];
    }
    let last = grid.len() - 1;
    if x >= grid[last] {
        return vals[last];
    }
    let idx = grid.partition_point(|v| *v < x);
    let x_lo = grid[idx - 1];
    let x_hi = grid[idx];
    let y_lo = vals[idx - 1];
    let y_hi = vals[idx];
    let t = (x - x_lo) / (x_hi - x_lo);
    y_lo + t * (y_hi - y_lo)
}

/// Analytic differential Compton PDF shape at photon energy `e_ev`
/// on element `elem`, evaluated at `mu`.
fn compton_differential_shape(elem: &PhotonElement, e_ev: f64, mu: f64) -> f64 {
    let mu = mu.clamp(-1.0, 1.0);
    let alpha = e_ev / M_E_C2_EV;
    let k = 1.0 / (1.0 + alpha * (1.0 - mu));
    let half_1m_mu = (0.5 * (1.0 - mu)).max(0.0);
    let x = (e_ev / HC_EV_ANGSTROM) * half_1m_mu.sqrt();
    let s_of_x = interp_linear(
        &elem.incoherent_scattering_factor.x,
        &elem.incoherent_scattering_factor.value,
        x,
    );
    let s_over_z = s_of_x / elem.z as f64;
    k * k * (k + 1.0 / k - 1.0 + mu * mu) * s_over_z
}

#[test]
fn compton_angular_distribution_matches_hubbell_shape_on_pb_100kev() {
    let Some(path) = photon_path("Pb.h5") else {
        eprintln!("skipping: Pb.h5 not present");
        return;
    };
    let pb = PhotonElement::from_hdf5(&path).expect("load Pb");
    let energy = 100_000.0_f64;

    // Sample N events and histogram mu in 40 uniform bins.
    let n_bins = 40_usize;
    let n_mc = 200_000_usize;
    let bin_width = 2.0 / n_bins as f64;
    let mut hist = vec![0u64; n_bins];
    let mut rng = Rng::new(0xDEADBEEF, 1);
    for _ in 0..n_mc {
        let out = compton_scatter(&pb, energy, &mut rng);
        let b = ((out.mu + 1.0) / bin_width) as usize;
        let b = b.min(n_bins - 1);
        hist[b] += 1;
    }

    // Build analytic expected counts per bin using Simpson integration
    // of the differential shape.
    let n_sub = 101_usize; // odd for Simpson
    let mut analytic = vec![0.0_f64; n_bins];
    for bin in 0..n_bins {
        let mu_lo = -1.0 + bin as f64 * bin_width;
        let h = bin_width / (n_sub as f64 - 1.0);
        let mut s = 0.0;
        for i in 0..n_sub {
            let mu = mu_lo + i as f64 * h;
            let w = if i == 0 || i == n_sub - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            s += w * compton_differential_shape(&pb, energy, mu);
        }
        analytic[bin] = s * h / 3.0;
    }
    let total_analytic: f64 = analytic.iter().sum();
    // Scale analytic to N_MC.
    let expected: Vec<f64> = analytic
        .iter()
        .map(|a| a / total_analytic * n_mc as f64)
        .collect();

    // Pearson chi-squared / dof. Expected bins must exceed 5 per
    // standard Pearson validity (they all do here given N_MC = 200k
    // and the smoothness of the distribution).
    let mut chi2 = 0.0;
    let mut bins_used = 0u32;
    for (o, e) in hist.iter().zip(expected.iter()) {
        if *e > 5.0 {
            let d = *o as f64 - e;
            chi2 += d * d / e;
            bins_used += 1;
        }
    }
    let dof = bins_used.saturating_sub(1) as f64;
    let reduced = chi2 / dof.max(1.0);

    // At 40 dof, reduced χ²/ν ≤ 2 corresponds to p ≈ 1e-3 — a
    // comfortable pass threshold for a well-seeded MC sampler while
    // still rejecting gross shape mismatch.
    assert!(
        reduced <= 2.0,
        "Compton angular distribution χ²/ν = {reduced:.3} (χ² = {chi2:.1}, dof = {dof}), bins_used = {bins_used}"
    );

    // Spot-check a few bins at the ±15 % level to guarantee no
    // spectacular outlier hid inside a low-weight χ²:
    for i in 0..n_bins {
        let o = hist[i] as f64;
        let e = expected[i];
        if e > 200.0 {
            let rel = ((o - e) / e).abs();
            assert!(
                rel < 0.15,
                "bin {i} (μ ≈ {}): sampled {o}, expected {e:.1}, rel {rel:.3}",
                -1.0 + (i as f64 + 0.5) * bin_width
            );
        }
    }
}

/// Same test but at 500 keV on Carbon, where the bound correction
/// is much smaller and the distribution is closer to free Klein-
/// Nishina. A second energy / element validates that the test isn't
/// accidentally passing due to a fixed-E coincidence.
#[test]
fn compton_angular_distribution_matches_hubbell_shape_on_c_500kev() {
    let Some(path) = photon_path("C.h5") else {
        eprintln!("skipping: C.h5 not present");
        return;
    };
    let c = PhotonElement::from_hdf5(&path).expect("load C");
    let energy = 500_000.0_f64;

    let n_bins = 40_usize;
    let n_mc = 200_000_usize;
    let bin_width = 2.0 / n_bins as f64;
    let mut hist = vec![0u64; n_bins];
    let mut rng = Rng::new(0xBADF00D, 1);
    for _ in 0..n_mc {
        let out = compton_scatter(&c, energy, &mut rng);
        let b = ((out.mu + 1.0) / bin_width) as usize;
        let b = b.min(n_bins - 1);
        hist[b] += 1;
    }

    let n_sub = 101_usize;
    let mut analytic = vec![0.0_f64; n_bins];
    for bin in 0..n_bins {
        let mu_lo = -1.0 + bin as f64 * bin_width;
        let h = bin_width / (n_sub as f64 - 1.0);
        let mut s = 0.0;
        for i in 0..n_sub {
            let mu = mu_lo + i as f64 * h;
            let w = if i == 0 || i == n_sub - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            s += w * compton_differential_shape(&c, energy, mu);
        }
        analytic[bin] = s * h / 3.0;
    }
    let total_analytic: f64 = analytic.iter().sum();
    let expected: Vec<f64> = analytic
        .iter()
        .map(|a| a / total_analytic * n_mc as f64)
        .collect();

    let mut chi2 = 0.0;
    let mut bins_used = 0u32;
    for (o, e) in hist.iter().zip(expected.iter()) {
        if *e > 5.0 {
            let d = *o as f64 - e;
            chi2 += d * d / e;
            bins_used += 1;
        }
    }
    let dof = bins_used.saturating_sub(1) as f64;
    let reduced = chi2 / dof.max(1.0);
    assert!(
        reduced <= 2.0,
        "C 500 keV Compton χ²/ν = {reduced:.3} (bins_used = {bins_used})"
    );
}
