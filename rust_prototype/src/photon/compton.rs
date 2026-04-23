//! Compton (incoherent) scattering kernel.
//!
//! Phase 1 (this file): free-electron Klein-Nishina sampling with
//! bound-electron `S(x, Z)/Z` rejection. Implements the algorithm used
//! by OpenMC (`src/physics.cpp::sample_compton_angle` and
//! `Element::compton_scatter`) and PENELOPE-2018 §2.3 without Doppler
//! broadening. Outgoing `(E', μ)` lie on the free-electron Klein-Nishina
//! kinematic curve and conserve energy-momentum for a free electron at
//! rest. The bound-electron correction modifies the *angular*
//! distribution by rejecting events with `S(x, Z)/Z` probability below
//! 1 at low momentum transfer, but leaves the free kinematics intact.
//!
//! Phase 2 (future commit) will add Doppler broadening: select a
//! Compton shell, sample `p_z` from `Jᵢ(|p_z|)`, and solve the Doppler
//! quadratic for `E'(p_z, θ)`. That modifies the outgoing energy
//! around the free-KN value.
//!
//! # Algorithm (PENELOPE §2.3.3 / OpenMC)
//!
//! 1. `α = E / m_e c²`, `κ = 1 + 2α`.
//! 2. Decompose the Klein-Nishina differential on `k = E'/E ∈ [1/κ, 1]`:
//!    envelope `f_env(k) ∝ 1/k + k`, with composite weights
//!    `a₁ = ln κ` (for the `1/k` part) and
//!    `a₂ = ½(1 - 1/κ²)` (for the `k` part).
//! 3. In one draw: branch with probability `a₁/(a₁+a₂)`, sample `k`
//!    from the chosen component, compute `μ = 1 − (1−k)/(αk)`.
//! 4. Rejection in one combined test:
//!    accept if `ξ < [1 − (1−μ²)/(k + 1/k)] × S(x, Z)/Z`, where
//!    `x = E sin(θ/2) / hc` in inverse Ångström (Hubbell 1975).
//! 5. Return the outgoing photon energy `E' = E · k` and `μ`; the
//!    electron kinetic energy is `E − E'` (kerma approximation — no
//!    binding deducted in phase 1, added in phase 2 alongside Doppler).
//!
//! # References
//! - OpenMC source `src/physics.cpp` (master reference for the choice
//!   of rejection envelope and S/Z combined test)
//! - PENELOPE-2018 manual §2.3 (Salvat), Nuclear Energy Agency
//! - Klein & Nishina, Z. Phys. 52, 853 (1929)
//! - Koblinger, Nucl. Sci. Eng. 56, 218 (1975) — composite sampling
//! - Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975) —
//!   definition of `x` and tabulated `S(x, Z)`

use crate::photon::data::{PhotonElement, ScatteringFactor};
use crate::transport::rng::Rng;

// --- Physical constants ----------------------------------------------------

/// Electron rest-mass energy, eV. CODATA-2018: 510998.95 eV.
pub const M_E_C2_EV: f64 = 510_998.95;

/// `h c` in eV·Å. CODATA-2018 exact: 12398.419843320... eV·Å.
pub const HC_EV_ANGSTROM: f64 = 12_398.419_843_320_025;

// --- Types -----------------------------------------------------------------

/// Outcome of a single Compton scattering event.
#[derive(Debug, Clone, Copy)]
pub struct ComptonOutcome {
    /// Outgoing photon energy in eV.
    pub energy_out: f64,
    /// Scattering cosine `cos θ ∈ [-1, 1]`.
    pub mu: f64,
    /// Kinetic energy of the recoil electron in eV, phase-1
    /// approximation: `E_in − E_out` (no shell binding subtracted).
    /// Phase 2 with Doppler broadening will subtract the selected
    /// shell's `B_i`.
    pub electron_kinetic: f64,
}

// --- Public API ------------------------------------------------------------

/// Sample a Compton scattering event at incoming photon energy
/// `energy_in` (eV) on the element `elem` using the provided `rng`.
///
/// Returns `(E', μ, T_e)` where `E'` is on the free-electron
/// Klein-Nishina kinematic curve and the angular distribution is
/// modulated by the bound-electron `S(x, Z)/Z` rejection.
pub fn compton_scatter(
    elem: &PhotonElement,
    energy_in: f64,
    rng: &mut Rng,
) -> ComptonOutcome {
    let alpha = energy_in / M_E_C2_EV;
    let (k, mu) = sample_kn_with_bound_rejection(alpha, energy_in, elem, rng);
    let energy_out = energy_in * k;
    ComptonOutcome {
        energy_out,
        mu,
        electron_kinetic: energy_in - energy_out,
    }
}

// --- Internals -------------------------------------------------------------

/// Sample `(k = E'/E, μ = cos θ)` from Klein-Nishina modulated by
/// bound-electron `S(x, Z)/Z`. Composite envelope + one-draw rejection.
fn sample_kn_with_bound_rejection(
    alpha: f64,
    energy_in: f64,
    elem: &PhotonElement,
    rng: &mut Rng,
) -> (f64, f64) {
    let kappa = 1.0 + 2.0 * alpha;
    let kappa_inv = 1.0 / kappa;
    let kappa_inv_sq = kappa_inv * kappa_inv;

    let a1 = kappa.ln();
    let a2 = 0.5 * (1.0 - kappa_inv_sq);
    let p_branch_1 = a1 / (a1 + a2);

    let z = elem.z as f64;
    let hc_inv = energy_in / HC_EV_ANGSTROM;

    loop {
        let xi_branch = rng.uniform();
        let xi_sample = rng.uniform();
        let xi_reject = rng.uniform();

        // Sample k from the envelope ∝ 1/k + k on [1/κ, 1].
        let k = if xi_branch < p_branch_1 {
            // 1/k component: k = κ^(-ξ) = exp(-ξ · ln κ).
            (-xi_sample * a1).exp()
        } else {
            // k component: k = √(1/κ² + ξ · (1 - 1/κ²)).
            (kappa_inv_sq + xi_sample * (1.0 - kappa_inv_sq)).sqrt()
        };

        // Recover μ from the Compton shift relation
        //   1/k = 1 + α(1 − μ)  ⇒  μ = 1 − (1 − k)/(α · k).
        let mu = 1.0 - (1.0 - k) / (alpha * k);

        // Hubbell's momentum-transfer variable
        //   x [1/Å] = (E/hc) · sin(θ/2) = (E/hc) · √((1 − μ)/2)
        let x = hc_inv * (0.5 * (1.0 - mu)).sqrt();

        let s_of_x = interp_linear(&elem.incoherent_scattering_factor, x);
        let s_over_z = s_of_x / z;

        // Klein-Nishina shape factor
        //   accept_prob_KN(k, μ) = 1 − (1 − μ²)/(k + 1/k) ∈ [0, 1].
        let kn_accept = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);

        if xi_reject < kn_accept * s_over_z {
            return (k, mu);
        }
    }
}

/// Linear interpolation of a tabulated factor at query `x`.
///
/// Clamps to the tabulation endpoints outside the range (OpenMC
/// convention: `S(x, Z)` is extended by its last value at large x,
/// which is `Z`). The input `factor.x` must be strictly monotonically
/// increasing from 0.
fn interp_linear(factor: &ScatteringFactor, x_query: f64) -> f64 {
    if factor.x.is_empty() {
        return 0.0;
    }
    if x_query <= factor.x[0] {
        return factor.value[0];
    }
    let last = factor.x.len() - 1;
    if x_query >= factor.x[last] {
        return factor.value[last];
    }
    // Find idx such that factor.x[idx - 1] <= x_query < factor.x[idx].
    let idx = factor.x.partition_point(|v| *v < x_query);
    let x_lo = factor.x[idx - 1];
    let x_hi = factor.x[idx];
    let y_lo = factor.value[idx - 1];
    let y_hi = factor.value[idx];
    let t = (x_query - x_lo) / (x_hi - x_lo);
    y_lo + t * (y_hi - y_lo)
}

// --- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn photon_path(filename: &str) -> Option<PathBuf> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon")
            .join(filename);
        if path.exists() { Some(path) } else { None }
    }

    fn load(filename: &str) -> Option<PhotonElement> {
        let path = photon_path(filename)?;
        Some(PhotonElement::from_hdf5(&path).expect("load photon data"))
    }

    /// `k ∈ [1/κ, 1]` is the Klein-Nishina kinematic support.
    #[test]
    fn k_within_kinematic_bounds() {
        let Some(h) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xC0FFEE, 1);
        let energy = 1.0e6;
        let alpha = energy / M_E_C2_EV;
        let k_min = 1.0 / (1.0 + 2.0 * alpha);

        for _ in 0..20_000 {
            let out = compton_scatter(&h, energy, &mut rng);
            let k = out.energy_out / energy;
            assert!(
                (k_min - 1e-12..=1.0 + 1e-12).contains(&k),
                "k = {k} outside [{k_min}, 1]"
            );
        }
    }

    /// `μ ∈ [-1, 1]`.
    #[test]
    fn mu_within_unit_interval() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xBEEF, 1);
        for energy_mev in [0.1, 1.0, 10.0] {
            let energy = energy_mev * 1.0e6;
            for _ in 0..10_000 {
                let out = compton_scatter(&c, energy, &mut rng);
                assert!(
                    (-1.0 - 1e-12..=1.0 + 1e-12).contains(&out.mu),
                    "μ = {} outside [-1, 1] at E = {energy_mev} MeV",
                    out.mu
                );
            }
        }
    }

    /// The sampled `(k, μ)` pair must satisfy the Compton shift relation
    /// `1/k − 1 = α(1 − μ)` exactly (no stochastic noise — it's a
    /// kinematic identity).
    #[test]
    fn mu_k_consistent_with_compton_shift() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(42, 1);
        let energy = 2.0e6;
        let alpha = energy / M_E_C2_EV;

        for _ in 0..5_000 {
            let out = compton_scatter(&c, energy, &mut rng);
            let k = out.energy_out / energy;
            // μ = 1 − (1 − k)/(αk) should match the returned μ exactly.
            let mu_from_k = 1.0 - (1.0 - k) / (alpha * k);
            assert!(
                (out.mu - mu_from_k).abs() < 1e-12,
                "μ inconsistency: sampled {}, from k = {k} → {mu_from_k}",
                out.mu
            );
        }
    }

    /// `T_e = E − E'` (phase-1 kerma): electron kinetic energy equals
    /// the photon energy loss.
    #[test]
    fn electron_kinetic_is_photon_energy_loss() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(1234, 1);
        let energy = 5.0e5;

        for _ in 0..1_000 {
            let out = compton_scatter(&c, energy, &mut rng);
            let expected = energy - out.energy_out;
            assert!((out.electron_kinetic - expected).abs() < 1e-12);
            assert!(out.electron_kinetic >= 0.0);
        }
    }

    /// Compton forward-peaks with increasing photon energy. At low `α`
    /// a free-electron Klein-Nishina is Thomson-symmetric with `<μ> = 0`,
    /// but the bound-electron `S(x, Z)/Z` rejection preferentially
    /// removes *forward* scatters (small `x`, small momentum transfer,
    /// where `S → 0`) and so biases `<μ>` negative. At high `α` the
    /// Klein-Nishina differential itself forward-peaks strongly and
    /// `<μ> → 1`. Verify strict monotone increase 10 keV → 1 MeV →
    /// 10 MeV on Carbon.
    #[test]
    fn mean_mu_monotone_increasing_in_energy() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(7, 1);
        let n = 30_000;

        let mean_mu = |energy: f64, rng: &mut Rng| -> f64 {
            let mut s = 0.0;
            for _ in 0..n {
                s += compton_scatter(&c, energy, rng).mu;
            }
            s / n as f64
        };

        let mu_10kev = mean_mu(1.0e4, &mut rng);
        let mu_1mev = mean_mu(1.0e6, &mut rng);
        let mu_10mev = mean_mu(1.0e7, &mut rng);

        assert!(
            mu_10kev < mu_1mev && mu_1mev < mu_10mev,
            "<μ> not monotone in E: 10keV={mu_10kev}, 1MeV={mu_1mev}, 10MeV={mu_10mev}"
        );
        // Low-α bound Compton: forward scattering suppressed by S/Z → 0
        // at small x, so <μ> lands negative.
        assert!(
            mu_10kev < 0.0,
            "expected bound-suppressed <μ> < 0 at 10 keV on C, got {mu_10kev}"
        );
        // At 10 MeV the Klein-Nishina shape dominates; scattering is
        // strongly forward-peaked. Analytic free-KN <μ> at α ≈ 19.6
        // lands around 0.6; threshold 0.5 accepts sampling noise at
        // N = 30k without approving gross errors.
        assert!(
            mu_10mev > 0.5,
            "10 MeV forward peaking too weak: <μ> = {mu_10mev}"
        );
    }

    /// Bound-electron effect on a heavy element at low photon energy:
    /// `S(x, Z)/Z` rejection suppresses small-`x` (small-angle, i.e.
    /// forward) scatters where the tightly bound inner electrons
    /// cannot absorb the tiny momentum transfer. Relative to the
    /// free-electron Klein-Nishina reference, the surviving sample
    /// is biased toward backward scatter (more negative `<μ>`).
    ///
    /// We construct a matched "free-electron Pb" by overwriting the
    /// S-factor with the constant `Z` (so S/Z ≡ 1 everywhere, no
    /// rejection). Same seeded RNG stream for both, so differences
    /// come only from the S/Z factor.
    #[test]
    fn bound_rejection_shifts_mu_backward_at_low_energy() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut pb_free = PhotonElement::from_hdf5(&photon_path("Pb.h5").unwrap())
            .expect("load Pb");
        let z = pb_free.z as f64;
        pb_free
            .incoherent_scattering_factor
            .value
            .iter_mut()
            .for_each(|v| *v = z);

        let mut rng1 = Rng::new(99, 1);
        let mut rng2 = Rng::new(99, 1); // same stream for fair comparison
        let n = 40_000;
        let energy = 1.0e4;

        let mean = |elem: &PhotonElement, rng: &mut Rng| -> f64 {
            let mut s = 0.0;
            for _ in 0..n {
                s += compton_scatter(elem, energy, rng).mu;
            }
            s / n as f64
        };

        let mu_bound = mean(&pb, &mut rng1);
        let mu_free = mean(&pb_free, &mut rng2);
        assert!(
            mu_bound < mu_free - 0.05,
            "bound rejection should push <μ> backward on Pb at 10 keV: \
             bound={mu_bound}, free={mu_free}"
        );
    }

    /// At high photon energy the bound-electron rejection is
    /// saturated (`S/Z → 1` for all physically reachable `x`), so the
    /// sampled `(k, μ)` distribution must match the analytic free
    /// Klein-Nishina differential. Verify `<μ>` and `<μ²>` on Hydrogen
    /// at 10 MeV against numerically-integrated KN moments.
    #[test]
    fn high_energy_matches_analytic_klein_nishina() {
        let Some(h) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let energy = 1.0e7;
        let alpha = energy / M_E_C2_EV;

        // Analytic KN <μ^p> for p = 1, 2: integrate
        //   ∫ μ^p · dσ/dμ dμ / σ_total
        // with dσ/dμ ∝ k²(k + 1/k − 1 + μ²),
        // k = 1/(1 + α(1 − μ)).
        // Do it with Simpson's rule on a fine grid.
        let analytic_moments = |alpha: f64, p: u32| -> f64 {
            let n = 10_001_usize; // odd for Simpson's
            let h_step = 2.0 / (n as f64 - 1.0);
            let pdf_num = |mu: f64| -> f64 {
                let k = 1.0 / (1.0 + alpha * (1.0 - mu));
                k * k * (k + 1.0 / k - 1.0 + mu * mu)
            };
            let mut sum_num = 0.0;
            let mut sum_den = 0.0;
            for i in 0..n {
                let mu = -1.0 + i as f64 * h_step;
                let w = if i == 0 || i == n - 1 {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                let f = pdf_num(mu);
                sum_num += w * mu.powi(p as i32) * f;
                sum_den += w * f;
            }
            sum_num / sum_den
        };

        let mu1_analytic = analytic_moments(alpha, 1);
        let mu2_analytic = analytic_moments(alpha, 2);

        let mut rng = Rng::new(0xFACE, 1);
        let n_samples = 200_000_usize;
        let mut sum_mu = 0.0;
        let mut sum_mu2 = 0.0;
        for _ in 0..n_samples {
            let out = compton_scatter(&h, energy, &mut rng);
            sum_mu += out.mu;
            sum_mu2 += out.mu * out.mu;
        }
        let mu1_sampled = sum_mu / n_samples as f64;
        let mu2_sampled = sum_mu2 / n_samples as f64;

        // Expected sampling SEM for 200k samples: σ/√N ≈ 0.5/450 ≈
        // 1.1e-3, so tolerate 5e-3 (≈ 4 σ to keep the test flake-free).
        assert!(
            (mu1_sampled - mu1_analytic).abs() < 5e-3,
            "<μ>: sampled {mu1_sampled}, analytic {mu1_analytic}"
        );
        assert!(
            (mu2_sampled - mu2_analytic).abs() < 5e-3,
            "<μ²>: sampled {mu2_sampled}, analytic {mu2_analytic}"
        );
    }

    /// `interp_linear` endpoint / midpoint behaviour.
    mod interp {
        use super::super::*;

        fn sf(x: Vec<f64>, value: Vec<f64>) -> ScatteringFactor {
            ScatteringFactor { x, value }
        }

        #[test]
        fn below_range_clamps_to_first() {
            let f = sf(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
            assert_eq!(interp_linear(&f, 0.5), 10.0);
            assert_eq!(interp_linear(&f, 1.0), 10.0);
        }

        #[test]
        fn above_range_clamps_to_last() {
            let f = sf(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]);
            assert_eq!(interp_linear(&f, 3.0), 30.0);
            assert_eq!(interp_linear(&f, 100.0), 30.0);
        }

        #[test]
        fn midpoint_interpolates_linearly() {
            let f = sf(vec![0.0, 2.0], vec![0.0, 10.0]);
            assert_eq!(interp_linear(&f, 1.0), 5.0);
            assert_eq!(interp_linear(&f, 1.5), 7.5);
        }

        #[test]
        fn empty_returns_zero() {
            let f = sf(vec![], vec![]);
            assert_eq!(interp_linear(&f, 0.0), 0.0);
            assert_eq!(interp_linear(&f, 1.0), 0.0);
        }
    }
}
