//! Pair production kernel (nuclear + triplet channels), Bethe-Heitler
//! sampling with in-flight-then-at-rest positron annihilation.
//!
//! Kinematics:
//!   - Threshold for pair production in nuclear field:
//!     `2 m_e c² = 1.022 MeV`.
//!   - Threshold for triplet (electron-field) pair:
//!     `4 m_e c² = 2.044 MeV`. OpenMC folds triplet into the nuclear
//!     channel and ignores the recoil electron; we do the same.
//!
//! Energy partition (Bethe-Heitler, unscreened — valid within ~5 %
//! from threshold to ~50 MeV photon energy, which covers all reactor
//! and typical shielding applications). Let
//!   `ε = T_-/(E − 2 m_e c²)  ∈ [0, 1]`.
//! The Bethe-Heitler shape is symmetric about `ε = 0.5` and given by
//!   `f(ε) ∝ ε² + (1 − ε)² + (2/3) ε (1 − ε)`.
//! Sample by rejection from the uniform envelope (peak value `1` at
//! ε=0 or 1, minimum `2/3` at ε=1/2).
//!
//! Angular distribution: both leptons are sharply forward-peaked at
//! high photon energies (`⟨θ⟩ ≈ m_e c² / E`). The first-pass
//! implementation emits both along the incoming photon direction
//! (μ = 1). Near-threshold angular spread can be added as a refinement
//! when/if we need kerma → dose accuracy below ~5 MeV.
//!
//! Positron annihilation under the kerma approximation (no electron
//! transport): the positron deposits its KE locally, then annihilates
//! at rest emitting two back-to-back 511 keV photons isotropically.
//!
//! # References
//! - Heitler, *The Quantum Theory of Radiation* (3rd ed., 1954) §26
//! - Bethe & Heitler, Proc. Roy. Soc. A 146, 83 (1934)
//! - Tsai, Rev. Mod. Phys. 46, 815 (1974) — high-energy screening
//! - Motz, Olsen & Koch, Rev. Mod. Phys. 41, 581 (1969) — review
//! - OpenMC src/physics.cpp `sample_pair_production`

use crate::transport::rng::Rng;

/// Electron rest-mass energy in eV (CODATA-2018).
pub const M_E_C2_EV: f64 = 510_998.95;

/// Pair-production threshold in eV (`2 m_e c²`).
pub const PAIR_THRESHOLD_EV: f64 = 2.0 * M_E_C2_EV;

/// At-rest annihilation photon energy in eV (`m_e c²`).
pub const ANNIHILATION_ENERGY_EV: f64 = M_E_C2_EV;

/// Outcome of a single pair-production event.
#[derive(Debug, Clone)]
pub struct PairOutcome {
    /// Electron kinetic energy in eV (deposited locally under kerma).
    pub electron_kinetic: f64,
    /// Positron kinetic energy in eV (deposited locally under kerma).
    pub positron_kinetic: f64,
    /// Electron `cos θ` relative to incoming photon direction.
    pub mu_electron: f64,
    /// Positron `cos θ` relative to incoming photon direction.
    pub mu_positron: f64,
    /// Annihilation photon energies in eV. At-rest approximation:
    /// two 511 keV photons back-to-back, orientations set by the
    /// caller (transport loop samples an isotropic axis).
    pub annihilation_photons: Vec<f64>,
}

impl PairOutcome {
    /// Total local energy deposition (electron + positron kinetic).
    pub fn local_deposition(&self) -> f64 {
        self.electron_kinetic + self.positron_kinetic
    }
}

/// Sample a pair-production event at incoming photon energy `energy_in`
/// (eV). Returns `None` when below threshold.
pub fn pair_produce(energy_in: f64, rng: &mut Rng) -> Option<PairOutcome> {
    if energy_in < PAIR_THRESHOLD_EV {
        return None;
    }
    let t_total = energy_in - PAIR_THRESHOLD_EV;

    let epsilon = sample_bethe_heitler_epsilon(rng);

    let electron_ke = epsilon * t_total;
    let positron_ke = (1.0 - epsilon) * t_total;

    Some(PairOutcome {
        electron_kinetic: electron_ke,
        positron_kinetic: positron_ke,
        // Forward-peaked approximation. The exact angle of each lepton
        // is ~ m_e c²/E and symmetric about the incoming direction.
        mu_electron: 1.0,
        mu_positron: 1.0,
        annihilation_photons: vec![ANNIHILATION_ENERGY_EV, ANNIHILATION_ENERGY_EV],
    })
}

/// Sample `ε ∈ [0, 1]` from the Bethe-Heitler shape by rejection
/// from a uniform envelope.
///
/// `f(ε) = ε² + (1−ε)² + (2/3) ε(1−ε)` has
/// `f(0) = f(1) = 1`, `f(1/2) = 2/3`,
/// so `f ≤ 1` and rejection with the uniform envelope has 83 %
/// efficiency on average (accept probability = `<f> = 5/6`).
fn sample_bethe_heitler_epsilon(rng: &mut Rng) -> f64 {
    loop {
        let eps = rng.uniform();
        let xi = rng.uniform();
        let f_val = eps * eps + (1.0 - eps) * (1.0 - eps) + (2.0 / 3.0) * eps * (1.0 - eps);
        if xi < f_val {
            return eps;
        }
    }
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

    #[test]
    fn below_threshold_returns_none() {
        let mut rng = Rng::new(1, 1);
        assert!(pair_produce(500_000.0, &mut rng).is_none());
        assert!(pair_produce(PAIR_THRESHOLD_EV - 1.0, &mut rng).is_none());
    }

    #[test]
    fn at_threshold_kinetic_energies_vanish() {
        let mut rng = Rng::new(2, 1);
        // Exactly at threshold, both leptons have zero KE.
        let out = pair_produce(PAIR_THRESHOLD_EV, &mut rng).expect("at threshold");
        assert!(out.electron_kinetic.abs() < 1e-9);
        assert!(out.positron_kinetic.abs() < 1e-9);
    }

    /// Energy conservation per event:
    /// `E_in = T_- + T_+ + 2 m_e c²` (the two 511 keV annihilation
    /// photons carry the rest-mass energy back out).
    #[test]
    fn energy_conservation_strict() {
        let mut rng = Rng::new(42, 1);
        for energy_mev in [1.1, 2.0, 5.0, 20.0, 100.0] {
            let energy = energy_mev * 1.0e6;
            for _ in 0..500 {
                let out = pair_produce(energy, &mut rng).expect("above threshold");
                let total = out.electron_kinetic
                    + out.positron_kinetic
                    + out.annihilation_photons.iter().sum::<f64>();
                assert!(
                    (total - energy).abs() < 1e-6,
                    "energy mismatch at {energy_mev} MeV: in={energy}, out={total}"
                );
            }
        }
    }

    /// `<ε>` = 1/2 by Bethe-Heitler symmetry (f(ε) = f(1−ε)).
    #[test]
    fn mean_epsilon_is_one_half() {
        let mut rng = Rng::new(7, 1);
        let n = 200_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_bethe_heitler_epsilon(&mut rng);
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 5e-3, "<ε> = {mean}, expected 0.5");
    }

    /// `ε ∈ [0, 1]` always.
    #[test]
    fn epsilon_within_unit_interval() {
        let mut rng = Rng::new(123, 1);
        for _ in 0..10_000 {
            let e = sample_bethe_heitler_epsilon(&mut rng);
            assert!((0.0..=1.0).contains(&e), "ε = {e} outside [0, 1]");
        }
    }

    /// Both leptons have non-negative kinetic energy and sum to the
    /// kinetic budget `E − 2 m_e c²`.
    #[test]
    fn partition_is_consistent() {
        let mut rng = Rng::new(99, 1);
        let energy = 5.0e6;
        let t_total = energy - PAIR_THRESHOLD_EV;
        for _ in 0..1_000 {
            let out = pair_produce(energy, &mut rng).unwrap();
            assert!(out.electron_kinetic >= 0.0);
            assert!(out.positron_kinetic >= 0.0);
            let sum = out.electron_kinetic + out.positron_kinetic;
            assert!((sum - t_total).abs() < 1e-6);
        }
    }

    /// Always two annihilation photons, each 511 keV.
    #[test]
    fn annihilation_is_two_511_kev_photons() {
        let mut rng = Rng::new(5, 1);
        for _ in 0..100 {
            let out = pair_produce(3.0e6, &mut rng).unwrap();
            assert_eq!(out.annihilation_photons.len(), 2);
            for &e in &out.annihilation_photons {
                assert_eq!(e, ANNIHILATION_ENERGY_EV);
            }
        }
    }

    /// `<ε²>` matches the analytic Bethe-Heitler integral.
    /// ∫₀¹ ε² [ε² + (1−ε)² + (2/3)ε(1−ε)] dε / ∫₀¹ [ε² + (1−ε)² + (2/3)ε(1−ε)] dε
    #[test]
    fn epsilon_second_moment_matches_analytic() {
        // Analytic:
        // ∫ε²[ε² + (1-ε)² + (2/3)ε(1-ε)] dε
        //   = ∫[ε⁴ + ε²(1-ε)² + (2/3)ε³(1-ε)] dε
        //   = 1/5 + [1/3 - 2/4 + 1/5] + (2/3)[1/4 - 1/5]
        //   = 0.2 + 1/30 + (2/3)(0.05)
        //   = 0.2 + 0.0333 + 0.0333 = 0.2667
        // ∫[ε² + (1-ε)² + (2/3)ε(1-ε)] dε
        //   = 1/3 + 1/3 + (2/3)(1/2 - 1/3) = 2/3 + (2/3)(1/6) = 2/3 + 1/9
        //   = 7/9 ≈ 0.7778
        // <ε²> = 0.2667 / 0.7778 ≈ 0.3429
        let analytic_mean_eps2 = 0.2667 / 0.7778;

        let mut rng = Rng::new(31, 1);
        let n = 200_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let e = sample_bethe_heitler_epsilon(&mut rng);
            sum += e * e;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - analytic_mean_eps2).abs() < 5e-3,
            "<ε²> = {mean}, analytic = {analytic_mean_eps2}"
        );
    }
}
