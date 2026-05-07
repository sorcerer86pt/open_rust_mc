//! MoC analytic segment integrator (multigroup, flat source).
//!
//! Along a characteristic in a flat-source region (FSR) with constant
//! Σ_t,g and isotropic source Q_f,g, the angular flux ψ_g (per
//! steradian) satisfies
//!
//! ```text
//!   dψ_g/ds = -Σ_t,g · ψ_g + q_g,    q_g = Q_f,g / (4π)
//! ```
//!
//! Closed-form solution over a segment of length `l`, with `τ = Σ_t·l`:
//!
//! ```text
//!   ψ_out = ψ_in · exp(-τ) + (q/Σ_t) · (1 - exp(-τ))
//!   ψ_avg = ψ_in · F + (q/Σ_t) · (1 - F),  F = (1 - exp(-τ))/τ
//! ```
//!
//! The track-length estimator for the FSR scalar flux is
//!
//! ```text
//!   φ_f,g = 4π · ⟨ψ⟩_segments_in_f
//!         = 4π · (Σ_segments l · ψ_avg) / (Σ_segments l)
//! ```
//!
//! `solve_segment` returns ψ_out and the track-length contribution
//! `l · ψ_avg` so the solver can atomic-add into per-FSR accumulators.

#[derive(Debug, Clone, Copy)]
pub struct SegmentResult {
    pub psi_out: f64,
    /// `l · ψ_avg` — the contribution this segment makes to the
    /// track-length flux numerator for this (FSR, group).
    pub track_psi: f64,
}

/// Numerically stable `(1 - exp(-τ))/τ`.
///
/// Tail expansion `1 - τ/2 + τ²/6 - τ³/24 + …` for small τ avoids the
/// catastrophic cancellation in the direct formula.
#[inline]
pub fn exp_m1_over(tau: f64) -> f64 {
    if tau.abs() < 1e-4 {
        // Horner-evaluated truncated series to ~8 sig figs at τ=1e-4.
        1.0 - tau * (0.5 - tau * (1.0 / 6.0 - tau * (1.0 / 24.0)))
    } else {
        (1.0 - (-tau).exp()) / tau
    }
}

/// Solve the MoC ODE across one flat segment.
///
/// `sigma_t` is Σ_t,g (cm⁻¹), strictly positive.
/// `q_per_sr` is the per-steradian source `Q_f,g/(4π)`.
/// `length` is the segment length (cm).
/// `psi_in` is the angular flux entering the segment.
#[inline]
pub fn solve_segment(sigma_t: f64, q_per_sr: f64, length: f64, psi_in: f64) -> SegmentResult {
    debug_assert!(sigma_t > 0.0, "Σ_t must be positive");
    debug_assert!(length >= 0.0, "segment length must be non-negative");
    let tau = sigma_t * length;
    let f = exp_m1_over(tau);
    let q_over_t = q_per_sr / sigma_t;
    let psi_avg = psi_in * f + q_over_t * (1.0 - f);
    let exp_neg_tau = if tau.abs() < 1e-4 {
        1.0 - tau * (1.0 - tau * (0.5 - tau * (1.0 / 6.0)))
    } else {
        (-tau).exp()
    };
    let psi_out = psi_in * exp_neg_tau + q_over_t * (1.0 - exp_neg_tau);
    SegmentResult {
        psi_out,
        track_psi: length * psi_avg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_m1_over_zero_is_one() {
        assert!((exp_m1_over(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn exp_m1_over_small_matches_direct_formula() {
        // For τ in the regime where direct evaluation is stable
        // (around 1e-4 to 0.1), the series and direct should agree.
        for &tau in &[1e-4_f64, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 10.0] {
            let direct = (1.0 - (-tau).exp()) / tau;
            let series = exp_m1_over(tau);
            let rel = ((direct - series) / direct).abs();
            assert!(
                rel < 1e-8,
                "τ={tau}: direct={direct}, series={series}, rel={rel}"
            );
        }
    }

    #[test]
    fn segment_with_zero_source_is_pure_attenuation() {
        // ψ_in = 1, no source, length such that τ = ln(2) → ψ_out = 0.5.
        let sigma_t = 1.0;
        let length = std::f64::consts::LN_2;
        let r = solve_segment(sigma_t, 0.0, length, 1.0);
        assert!((r.psi_out - 0.5).abs() < 1e-12);
    }

    #[test]
    fn segment_in_steady_state_holds_psi_constant() {
        // q/Σ_t represents the steady-state angular flux. Set ψ_in
        // equal to it: ψ_out should equal ψ_in regardless of length.
        let sigma_t = 0.7;
        let q_per_sr = 0.42;
        let psi_eq = q_per_sr / sigma_t;
        for &length in &[0.01, 0.1, 1.0, 10.0, 100.0] {
            let r = solve_segment(sigma_t, q_per_sr, length, psi_eq);
            assert!(
                (r.psi_out - psi_eq).abs() < 1e-12,
                "ψ_out={} should equal ψ_eq={} at length {length}",
                r.psi_out,
                psi_eq
            );
            // ψ_avg also equals ψ_eq so track_psi = length · ψ_eq.
            assert!((r.track_psi - length * psi_eq).abs() < 1e-12);
        }
    }

    #[test]
    fn segment_track_psi_matches_definite_integral() {
        // Integrate ψ(s) analytically from 0 to l and compare:
        //   ∫₀ˡ ψ(s) ds = ψ_in · (1-e^(-τ))/Σ_t  +  (q/Σ_t) · (l - (1-e^(-τ))/Σ_t)
        let sigma_t = 0.4;
        let q = 0.3;
        let l = 2.0;
        let psi_in = 0.7;
        let r = solve_segment(sigma_t, q, l, psi_in);
        let tau = sigma_t * l;
        let one_minus_e = 1.0 - (-tau).exp();
        let analytic = psi_in * one_minus_e / sigma_t + (q / sigma_t) * (l - one_minus_e / sigma_t);
        assert!(
            (r.track_psi - analytic).abs() < 1e-12,
            "track={}, analytic={}",
            r.track_psi,
            analytic
        );
    }

    #[test]
    fn segment_zero_length_is_no_op() {
        let r = solve_segment(1.0, 0.5, 0.0, 0.7);
        assert!((r.psi_out - 0.7).abs() < 1e-15);
        assert!(r.track_psi.abs() < 1e-15);
    }
}
