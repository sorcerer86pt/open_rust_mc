//! Coherent (Rayleigh) scattering kernel.
//!
//! Elastic — the photon keeps its energy, only its direction changes.
//!
//! Differential cross section (per atom):
//!   `dσ_coh/dΩ = (r_e²/2)(1 + cos²θ) · F(x, Z)²`
//! with `x = sin(θ/2)/λ`, `λ = hc/E`.
//!
//! Sampling (OpenMC / PENELOPE method):
//!   1. Compute kinematic limit `x_max = E/hc` (at θ = π).
//!   2. Sample `x²` from the cumulative form-factor-squared
//!      distribution on `[0, x_max²]`. The HDF5 file stores
//!      `∫₀ˣ² F²(x', Z) dx'²` pre-tabulated as
//!      `coherent_integrated_form_factor` (shape `[2, N]` with row 0
//!      the `x²` grid and row 1 the cumulative). Invert the CDF.
//!   3. Convert `x² → μ`: `μ = 1 − 2 x² (hc/E)² = 1 − 2 x² λ²`.
//!   4. Thomson acceptance: accept with probability `(1 + μ²)/2`.
//!      The form factor is already folded into step 2; this step
//!      applies the remaining `(1+cos²θ)/2` factor of the
//!      unpolarized differential.
//!
//! Anomalous amplitude correction `f'(E) + i f''(E)` is negligible
//! above ~100 keV and small below; deferred to a future commit.
//!
//! # References
//! - OpenMC src/element.cpp `Element::rayleigh_scatter`
//! - PENELOPE-2018 §2.1 (Salvat)
//! - Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975)

use crate::photon::compton::HC_EV_ANGSTROM;
use crate::photon::data::{PhotonElement, ScatteringFactor};
use crate::transport::rng::Rng;

/// Outcome of a coherent scattering event.
#[derive(Debug, Clone, Copy)]
pub struct CoherentOutcome {
    /// Scattering cosine `cos θ ∈ [-1, 1]`.
    pub mu: f64,
}

/// Sample a coherent scattering event at incoming photon energy
/// `energy_in` (eV). Returns only `μ` — the photon energy is
/// unchanged by elastic scattering.
pub fn coherent_scatter(elem: &PhotonElement, energy_in: f64, rng: &mut Rng) -> CoherentOutcome {
    // Kinematic limits on x = sin(θ/2)/λ.
    //   x_min = 0 at θ = 0  (forward, μ = 1)
    //   x_max = 1/λ = E/hc at θ = π  (backward, μ = -1)
    let lambda = HC_EV_ANGSTROM / energy_in; // Å
    let x_max = energy_in / HC_EV_ANGSTROM; // 1/Å
    let x_max_sq = x_max * x_max;

    // Cumulative form-factor-squared at x²_max. The stored CDF is
    // F²-integrated in x², so we interpolate at x²_max.
    let iff = &elem.coherent_integrated_form_factor;
    let cdf_max = interp_linear(iff, x_max_sq);

    loop {
        // Sample x² from the cumulative via inverse-CDF.
        let xi = rng.uniform();
        let target = xi * cdf_max;
        let x_sq = invert_cdf(iff, target);
        // Guard rail: kinematic clamp (in case interpolation slightly
        // overshoots x_max² due to rounding).
        let x_sq = x_sq.min(x_max_sq);

        // Convert to μ.
        //   sin²(θ/2) = x² · λ² = x² · (hc/E)²
        //   μ = 1 - 2 sin²(θ/2) = 1 - 2 x² λ²
        let mu = 1.0 - 2.0 * x_sq * lambda * lambda;
        let mu = mu.clamp(-1.0, 1.0);

        // Thomson acceptance (1 + μ²)/2 ∈ [1/2, 1].
        let accept = 0.5 * (1.0 + mu * mu);
        if rng.uniform() < accept {
            return CoherentOutcome { mu };
        }
    }
}

/// Invert a monotonic-non-decreasing tabulated CDF at target value `y`.
///
/// `factor.x` is the x² grid; `factor.value` is the cumulative F².
/// Binary search for the bin containing `y`, then linear interpolation
/// in x² within the bin.
fn invert_cdf(factor: &ScatteringFactor, y: f64) -> f64 {
    if factor.value.is_empty() {
        return 0.0;
    }
    if y <= factor.value[0] {
        return factor.x[0];
    }
    let last = factor.value.len() - 1;
    if y >= factor.value[last] {
        return factor.x[last];
    }
    let idx = factor.value.partition_point(|v| *v < y);
    let y_lo = factor.value[idx - 1];
    let y_hi = factor.value[idx];
    let x_lo = factor.x[idx - 1];
    let x_hi = factor.x[idx];
    let denom = y_hi - y_lo;
    if denom <= 0.0 {
        return x_lo;
    }
    let t = (y - y_lo) / denom;
    x_lo + t * (x_hi - x_lo)
}

/// Linear interpolation of a `ScatteringFactor` at query `x`.
fn interp_linear(factor: &ScatteringFactor, x: f64) -> f64 {
    if factor.x.is_empty() {
        return 0.0;
    }
    if x <= factor.x[0] {
        return factor.value[0];
    }
    let last = factor.x.len() - 1;
    if x >= factor.x[last] {
        return factor.value[last];
    }
    let idx = factor.x.partition_point(|v| *v < x);
    let x_lo = factor.x[idx - 1];
    let x_hi = factor.x[idx];
    let y_lo = factor.value[idx - 1];
    let y_hi = factor.value[idx];
    let t = (x - x_lo) / (x_hi - x_lo);
    y_lo + t * (y_hi - y_lo)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::double_comparisons,
    clippy::doc_lazy_continuation,
    clippy::too_many_arguments
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn load(name: &str) -> Option<PhotonElement> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let p = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon")
            .join(name);
        if p.exists() {
            Some(PhotonElement::from_hdf5(&p).unwrap())
        } else {
            None
        }
    }

    /// `μ ∈ [-1, 1]`.
    #[test]
    fn mu_in_unit_interval() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5");
            return;
        };
        let mut rng = Rng::new(1, 1);
        for energy_kev in [1.0, 10.0, 100.0, 1_000.0] {
            let e = energy_kev * 1_000.0;
            for _ in 0..5_000 {
                let o = coherent_scatter(&pb, e, &mut rng);
                assert!((-1.0..=1.0).contains(&o.mu), "μ = {} at {e}", o.mu);
            }
        }
    }

    /// Coherent scattering is strongly forward-peaked: `<μ>` close to
    /// 1 at high energy on low-Z. At high E, x_max is large so F(x, Z)
    /// collapses toward zero except at very small x, forcing small
    /// θ (μ → 1).
    #[test]
    fn forward_peaked_at_high_energy() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5");
            return;
        };
        let mut rng = Rng::new(2, 1);
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += coherent_scatter(&c, 1.0e6, &mut rng).mu;
        }
        let mean = sum / n as f64;
        assert!(mean > 0.99, "high-E <μ> on C = {mean}, expected > 0.99");
    }

    /// At low energy (x_max small) coherent scattering is almost
    /// Thomson-like (isotropic in 1+μ²), so `<μ>` is close to 0
    /// modulo the form-factor correction. On Pb at 1 keV the form
    /// factor is near Z over the full angular range, so we're close
    /// to pure Thomson and `<μ>` is small.
    #[test]
    fn low_energy_pb_near_thomson_limit() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5");
            return;
        };
        let mut rng = Rng::new(3, 1);
        let n = 20_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let mu = coherent_scatter(&pb, 1_000.0, &mut rng).mu;
            sum += mu;
            sum_sq += mu * mu;
        }
        let mean = sum / n as f64;
        // Thomson <μ> = 0 exactly. Accept within 0.1 for sampling noise
        // and the residual form-factor asymmetry.
        assert!(mean.abs() < 0.1, "low-E Pb <μ> = {mean}, expected ≈ 0");
        // Thomson <μ²> = 2/5 = 0.4 exactly.
        let mean_sq = sum_sq / n as f64;
        assert!(
            (mean_sq - 0.4).abs() < 0.05,
            "low-E Pb <μ²> = {mean_sq}, expected ≈ 0.4"
        );
    }

    mod cdf_inversion {
        use super::super::*;

        #[test]
        fn endpoints_clamped() {
            let sf = ScatteringFactor {
                x: vec![0.0, 1.0, 2.0, 3.0],
                value: vec![0.0, 2.0, 5.0, 9.0],
            };
            assert_eq!(invert_cdf(&sf, -1.0), 0.0);
            assert_eq!(invert_cdf(&sf, 9.0), 3.0);
            assert_eq!(invert_cdf(&sf, 100.0), 3.0);
        }

        #[test]
        fn midpoint_inversion_linear() {
            let sf = ScatteringFactor {
                x: vec![0.0, 2.0],
                value: vec![0.0, 10.0],
            };
            assert_eq!(invert_cdf(&sf, 5.0), 1.0);
        }

        #[test]
        fn flat_segment_returns_lower() {
            // Flat CDF segment (y_hi == y_lo); any target in this
            // range returns the lower x.
            let sf = ScatteringFactor {
                x: vec![0.0, 1.0, 2.0, 3.0],
                value: vec![0.0, 2.0, 2.0, 5.0],
            };
            // y=2.0 lies on the flat segment boundary; we return the
            // first x where y >= 2.0 which is x = 1.0.
            assert_eq!(invert_cdf(&sf, 2.0), 1.0);
        }
    }
}
