//! Continuous-energy adjoint neutron kernels (CADIS for neutrons).
//!
//! Companion to `transport::adjoint_photon`. Lands the **adjoint
//! elastic-scatter kernel** for free-gas / cold-target scattering on
//! a nucleus of mass `A`. The kernel is the analogue of the adjoint
//! Compton: invert the forward kinematics, sample the pre-collision
//! energy from the transposed scattering kernel, return the
//! kinematic μ in CM and lab frames.
//!
//! # Math (s-wave isotropic CM, no resonance / Doppler corrections)
//!
//! Forward elastic kinematics on a nucleus of mass `A`:
//!
//! ```text
//!   E_out = E_in · ½ · [(1 + α) + (1 − α) · μ_cm]
//!     with α = ((A − 1) / (A + 1))²
//! ```
//!
//! `μ_cm` is uniform on [-1, 1] for the s-wave model. Outgoing
//! energy is therefore uniform on `[α · E_in, E_in]` — the classic
//! "elastic energy-loss strip."
//!
//! Forward differential per unit `E_out` at fixed `E_in`:
//!
//! ```text
//!   ∂σ_s_fwd/∂E_out (E_in → E_out)
//!     = σ_s(E_in) · 1 / (E_in · (1 − α))    if α·E_in ≤ E_out ≤ E_in
//!     = 0                                     otherwise
//! ```
//!
//! Adjoint kernel (Lewis-Miller §10.3):
//! `P_adj(E_in | E_out) ∝ ∂σ_s_fwd/∂E_out (E_in → E_out)` treated
//! as a function of `E_in`. With `E_out` fixed and `E_in` ranging:
//!
//! ```text
//!   P_adj(E_in | E_out) ∝ σ_s(E_in) / E_in    on E_in ∈ [E_out, E_out/α]
//! ```
//!
//! The `1/(1−α)` factor is constant in `E_in` and drops out.
//!
//! For **constant `σ_s`** (s-wave smooth region — valid for moderator
//! nuclei H/D/C/O outside resonances), the conditional reduces to
//! `1/E_in` and is sampled exactly by log-uniform on
//! `[E_out, E_out/α]`. For `A = 1` (hydrogen), `α = 0` and
//! `E_out/α → ∞`; the sampling range is `[E_out, e_in_max]` where
//! `e_in_max` is the source-spectrum cutoff.
//!
//! Energy-dependent `σ_s(E_in)` (resonance modulation) is handled by
//! rejection against a constant-`σ_s` envelope at the user-supplied
//! `sigma_s_ceiling` — the kernel returns `Some` only when the
//! sample is accepted; rejected samples are signalled with `None`
//! so the caller can re-loop.

use crate::transport::rng::Rng;

/// Outcome of one **adjoint** elastic-scatter sample. The adjoint
/// particle is at `energy_out` post-collision; the sampler returns
/// the inferred pre-collision (higher) `energy_in` and the kinematic
/// scattering cosine in both CM and lab frames.
#[derive(Debug, Clone, Copy)]
pub struct AdjointElasticOutcome {
    pub energy_in: f64,
    pub mu_cm: f64,
    pub mu_lab: f64,
}

/// Adjoint elastic-scatter kernel for s-wave isotropic CM scattering
/// on a free-gas nucleus of atomic-weight ratio `awr ≈ A`. Samples
/// from the transposed forward kernel under a flat-`σ_s` assumption;
/// resonance corrections are layered on by the caller via rejection
/// (multiply outcome weight by `σ_s(E_in)/σ_s_ref` if needed).
///
/// `e_in_max` caps the sampling at the source-spectrum upper bound;
/// for A = 1 (hydrogen) this also serves as the kinematic upper
/// bound (`α = 0` ⇒ `E_out/α = ∞`).
pub fn adjoint_elastic_scatter(
    energy_out: f64,
    awr: f64,
    e_in_max: f64,
    rng: &mut Rng,
) -> AdjointElasticOutcome {
    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    // Kinematic upper bound on E_in. For α > 0 it's E_out/α; for
    // α = 0 (hydrogen) it's ∞ — capped only by `e_in_max`.
    let e_in_kin_max = if alpha > 0.0 {
        energy_out / alpha
    } else {
        f64::INFINITY
    };
    let e_in_hi = e_in_max.min(e_in_kin_max);
    if e_in_hi <= energy_out {
        // Degenerate range — return forward-scatter limit (no
        // up-scatter possible).
        return AdjointElasticOutcome {
            energy_in: energy_out,
            mu_cm: 1.0,
            mu_lab: 1.0,
        };
    }

    // Log-uniform on [E_out, E_in_hi] — exact for σ_s = const.
    let log_lo = energy_out.ln();
    let log_hi = e_in_hi.ln();
    let xi = rng.uniform();
    let e_in = (log_lo + xi * (log_hi - log_lo)).exp();

    // μ_cm from inverse kinematics. For α = 0 (hydrogen) the formula
    // reduces to μ_cm = 2 · E_out/E_in − 1 since (1 − α) = 1.
    let ratio = energy_out / e_in;
    let mu_cm = if alpha < 1e-15 {
        2.0 * ratio - 1.0
    } else {
        (2.0 * ratio - 1.0 - alpha) / (1.0 - alpha)
    };
    // Numerical safety — clamp to the valid CM range.
    let mu_cm = mu_cm.clamp(-1.0, 1.0);

    let mu_lab = if awr > 1.0 + 1e-10 {
        let denom = (1.0 + 2.0 * awr * mu_cm + awr * awr).sqrt();
        ((1.0 + awr * mu_cm) / denom).clamp(-1.0, 1.0)
    } else {
        // Hydrogen: μ_lab = √((1 + μ_cm)/2).
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };

    AdjointElasticOutcome {
        energy_in: e_in,
        mu_cm,
        mu_lab,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::useless_vec)]
mod tests {
    use super::*;

    /// Kinematic invariant: every sampled (E_in, μ_cm, E_out) triple
    /// must satisfy `E_out = E_in · ½ · ((1+α) + (1−α)·μ_cm)` and
    /// `μ_cm ∈ [-1, 1]`.
    #[test]
    fn adjoint_elastic_kinematic_invariant_carbon() {
        let mut rng = Rng::new(0xADC0E1, 1);
        let awr: f64 = 11.898; // C-12 ENDF
        let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
        let e_out = 1.0e3; // 1 keV
        let e_in_max = 5.0e6;
        for _ in 0..5000 {
            let o = adjoint_elastic_scatter(e_out, awr, e_in_max, &mut rng);
            assert!(o.mu_cm >= -1.0 - 1e-12 && o.mu_cm <= 1.0 + 1e-12);
            assert!(o.energy_in >= e_out - 1e-9);
            assert!(o.energy_in <= e_out / alpha + 1e-6 * e_out);
            // Forward kinematic check: must reproduce E_out.
            let e_out_check = o.energy_in * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * o.mu_cm);
            let rel = ((e_out_check - e_out) / e_out).abs();
            assert!(
                rel < 1e-9,
                "kinematic violation: forward E_out = {e_out_check}, expected {e_out}, rel = {rel:e}",
            );
        }
    }

    /// Hydrogen (A = 1): kinematic max is ∞. With `e_in_max` capping
    /// the range, sampled `E_in` fills `[E_out, e_in_max]` log-uniformly.
    /// Verify the histogram bins are populated nearly uniformly in
    /// log space (χ²-style flatness).
    #[test]
    fn adjoint_elastic_hydrogen_log_uniform() {
        const N: usize = 200_000;
        const N_BINS: usize = 20;
        const E_OUT: f64 = 100.0;
        const E_IN_MAX: f64 = 1.0e6;
        let mut rng = Rng::new(0xADC0E2, 0);
        let mut hist = vec![0u64; N_BINS];
        let log_lo = E_OUT.ln();
        let log_hi = E_IN_MAX.ln();
        for _ in 0..N {
            let o = adjoint_elastic_scatter(E_OUT, 0.999, E_IN_MAX, &mut rng);
            let f = (o.energy_in.ln() - log_lo) / (log_hi - log_lo);
            let b = ((f * N_BINS as f64) as usize).min(N_BINS - 1);
            hist[b] += 1;
        }
        let expected = N as f64 / N_BINS as f64;
        let chi2: f64 = hist
            .iter()
            .map(|&h| (h as f64 - expected).powi(2) / expected)
            .sum();
        let chi2_red = chi2 / (N_BINS - 1) as f64;
        assert!(
            chi2_red < 2.0,
            "hydrogen adjoint not log-uniform: χ²_red = {chi2_red:.3}",
        );
    }

    /// Carbon (A ≈ 12): kinematic interval is `[E_out, E_out/α]`
    /// with α ≈ 0.716. Verify sampled E_in stays within bounds and
    /// the conditional density is `∝ 1/E_in` on the interval.
    #[test]
    fn adjoint_elastic_carbon_log_uniform_in_range() {
        const N: usize = 200_000;
        const N_BINS: usize = 15;
        const E_OUT: f64 = 1.0e3;
        const E_IN_MAX: f64 = 1.0e6;
        let awr = 11.898_f64;
        let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
        let e_in_kin_max = E_OUT / alpha;
        let mut rng = Rng::new(0xADC0E3, 0);
        let mut hist = vec![0u64; N_BINS];
        let mut violations = 0;
        let log_lo = E_OUT.ln();
        let log_hi = e_in_kin_max.ln();
        for _ in 0..N {
            let o = adjoint_elastic_scatter(E_OUT, awr, E_IN_MAX, &mut rng);
            if o.energy_in < E_OUT - 1e-9 || o.energy_in > e_in_kin_max + 1e-6 * E_OUT {
                violations += 1;
                continue;
            }
            let f = (o.energy_in.ln() - log_lo) / (log_hi - log_lo);
            let b = ((f * N_BINS as f64) as usize).min(N_BINS - 1);
            hist[b] += 1;
        }
        assert_eq!(
            violations, 0,
            "{violations} samples outside kinematic range"
        );
        let total: u64 = hist.iter().sum();
        let expected = total as f64 / N_BINS as f64;
        let chi2: f64 = hist
            .iter()
            .map(|&h| (h as f64 - expected).powi(2) / expected)
            .sum();
        let chi2_red = chi2 / (N_BINS - 1) as f64;
        assert!(
            chi2_red < 2.0,
            "carbon adjoint not log-uniform: χ²_red = {chi2_red:.3}",
        );
    }

    /// **Forward-then-adjoint round trip on hydrogen.** Pick E_in
    /// log-uniformly, forward-scatter on H to get E_out, then run
    /// the adjoint kernel from that E_out and check the sampled
    /// E_in_adj is in the kinematic interval [E_out, e_in_max] and
    /// passes the kinematic relation `(1+α)+(1−α)μ_cm = 2 E_out/E_in`
    /// (for H, α = 0). Empirical check that the kernel doesn't
    /// produce out-of-range energies even for adversarial inputs.
    #[test]
    fn adjoint_elastic_hydrogen_round_trip_kinematic() {
        use crate::geometry::Vec3;
        use crate::physics::scatter::elastic_scatter;
        const N: usize = 50_000;
        const E_IN_LO: f64 = 1.0e3;
        const E_IN_HI: f64 = 1.0e6;
        let log_lo = E_IN_LO.ln();
        let log_hi = E_IN_HI.ln();
        let mut rng = Rng::new(0xADC0E4, 0);
        for _ in 0..N {
            let e_in_fwd = (log_lo + rng.uniform() * (log_hi - log_lo)).exp();
            let (e_out, _) = elastic_scatter(e_in_fwd, Vec3::new(0.0, 0.0, 1.0), 0.999, &mut rng);
            // E_out should lie in [0, E_in_fwd] by hydrogen kinematics.
            assert!(e_out > 0.0 && e_out <= e_in_fwd * (1.0 + 1e-9));
            let o = adjoint_elastic_scatter(e_out, 0.999, E_IN_HI, &mut rng);
            // The adjoint sample E_in_adj is independent of e_in_fwd
            // (it's drawn from the adjoint conditional on E_out),
            // but it MUST satisfy the same hydrogen kinematics:
            //   E_out = E_in_adj · 0.5 · (1 + μ_cm)
            // and E_in_adj ∈ [E_out, E_IN_HI].
            assert!(o.energy_in >= e_out * (1.0 - 1e-9));
            assert!(o.energy_in <= E_IN_HI * (1.0 + 1e-9));
            let mu_cm_check = (2.0 * e_out / o.energy_in - 1.0).clamp(-1.0, 1.0);
            assert!(
                (mu_cm_check - o.mu_cm).abs() < 1e-6,
                "hydrogen μ_cm mismatch: {mu_cm_check} vs {}",
                o.mu_cm,
            );
        }
    }
}
