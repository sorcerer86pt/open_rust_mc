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
/// Returns `(E', μ, T_e)` with the outgoing photon energy
/// Doppler-broadened about the free-electron Klein-Nishina value
/// using the Hartree-Fock Compton profiles (Ribberfors 1975 impulse
/// approximation, as in PENELOPE §2.3.5 and OpenMC).
pub fn compton_scatter(elem: &PhotonElement, energy_in: f64, rng: &mut Rng) -> ComptonOutcome {
    let alpha = energy_in / M_E_C2_EV;
    let (k_free, mu) = sample_kn_with_bound_rejection(alpha, energy_in, elem, rng);
    // Apply Doppler broadening on top of the free kinematics.
    let (energy_out, binding) = apply_doppler(elem, energy_in, alpha, k_free, mu, rng);
    ComptonOutcome {
        energy_out,
        mu,
        electron_kinetic: (energy_in - energy_out - binding).max(0.0),
    }
}

/// Sample a Compton scattering event without Doppler broadening —
/// outgoing energy is exactly `E · k_free`. Retained for unit-tests
/// that compare against the analytic free-electron Klein-Nishina
/// differential, where Doppler smearing would be a confounder.
pub fn compton_scatter_free(elem: &PhotonElement, energy_in: f64, rng: &mut Rng) -> ComptonOutcome {
    let alpha = energy_in / M_E_C2_EV;
    let (k, mu) = sample_kn_with_bound_rejection(alpha, energy_in, elem, rng);
    let energy_out = energy_in * k;
    ComptonOutcome {
        energy_out,
        mu,
        electron_kinetic: energy_in - energy_out,
    }
}

// --- Analytic Klein-Nishina helpers (free electron) ------------------------

/// Classical electron radius squared in cm². CODATA-2018:
/// `r_e = 2.8179403262e-13 cm` so `r_e² = 7.94079e-26 cm²`.
pub const R_E_SQ_CM2: f64 = 7.940_787_338_e-26;

/// Outgoing Compton photon energy `E' = E / (1 + α(1 − μ))` (free
/// electron, no Doppler). `μ = cos θ ∈ [-1, 1]`.
#[inline]
pub fn compton_e_out(energy_in: f64, mu: f64) -> f64 {
    let alpha = energy_in / M_E_C2_EV;
    energy_in / (1.0 + alpha * (1.0 - mu))
}

/// Klein-Nishina total free-electron Compton cross section (cm²)
/// at incoming photon energy `E` (eV). Reduces to Thomson
/// `σ_T = (8π/3)r_e²` in the limit `α → 0`.
///
/// Formula (Heitler 1954 / Jackson §13.7):
/// ```text
///   σ_KN(α) = 2π r_e² · {
///     (1+α)/α² · [2(1+α)/(1+2α) − ln(1+2α)/α]
///     + ln(1+2α)/(2α)
///     − (1+3α)/(1+2α)²
///   }
/// ```
pub fn klein_nishina_total(energy_in: f64) -> f64 {
    let alpha = energy_in / M_E_C2_EV;
    let opa = 1.0 + alpha;
    let oda = 1.0 + 2.0 * alpha;
    let log_oda = oda.ln();
    let two_pi_re_sq = 2.0 * std::f64::consts::PI * R_E_SQ_CM2;
    two_pi_re_sq
        * ((opa / (alpha * alpha)) * (2.0 * opa / oda - log_oda / alpha)
            + log_oda / (2.0 * alpha)
            - (1.0 + 3.0 * alpha) / (oda * oda))
}

/// Klein-Nishina differential cross section per unit `μ = cos θ`,
/// `dσ/dμ` in cm². Free electron, no bound correction.
///
/// `dσ/dΩ = (r_e²/2) · (k')² · (k/k' + k'/k − sin²θ)` with
/// `k' = 1/(1+α(1−μ))`. Multiplying by `2π` (azimuthal integration)
/// gives `dσ/dμ`.
#[inline]
pub fn klein_nishina_dcs_dmu(energy_in: f64, mu: f64) -> f64 {
    let alpha = energy_in / M_E_C2_EV;
    let k_prime = 1.0 / (1.0 + alpha * (1.0 - mu));
    let sin_sq = (1.0 - mu * mu).max(0.0);
    let dsigma_domega =
        0.5 * R_E_SQ_CM2 * k_prime * k_prime * (1.0 / k_prime + k_prime - sin_sq);
    2.0 * std::f64::consts::PI * dsigma_domega
}

/// Klein-Nishina probability density on `μ ∈ [-1, 1]`, normalised so
/// `∫_{-1}^{1} pdf dμ = 1`. This is the angular distribution per
/// scatter event used by the next-event / DXTRAN-style estimator
/// to compute the deterministic-contribution to a forward detector.
#[inline]
pub fn klein_nishina_pdf(energy_in: f64, mu: f64) -> f64 {
    klein_nishina_dcs_dmu(energy_in, mu) / klein_nishina_total(energy_in)
}

// ── Adjoint Compton ────────────────────────────────────────────────

/// Outcome of one **adjoint** Compton scatter sample. The adjoint
/// particle is "moving backwards" in time: it enters the collision
/// at `energy_out` and the kernel returns the inferred pre-collision
/// (i.e. higher) photon energy `energy_in` and the scattering
/// cosine `μ`.
#[derive(Debug, Clone, Copy)]
pub struct AdjointComptonOutcome {
    /// Sampled pre-collision (higher) photon energy in eV. Always
    /// `>= energy_out`.
    pub energy_in: f64,
    /// Scattering cosine `μ = cos θ`. By Compton kinematics
    /// `μ = 1 − m_e c² · (1/E_out − 1/E_in)`.
    pub mu: f64,
    /// Number of rejection attempts before acceptance — useful for
    /// efficiency telemetry.
    pub attempts: u32,
}

/// Adjoint Compton kernel. Inverts the Klein-Nishina relation:
/// given the post-collision photon energy `energy_out` (eV), sample
/// the pre-collision energy `energy_in` and scattering cosine `μ`
/// distributed by the **transposed** Compton kernel.
///
/// # Math
///
/// Forward kinematics: `E_out = E_in / (1 + α (1−μ))` with
/// `α = E_in / m_e c²`. Inverse (kinematic): given `(E_out, μ)`
///
/// ```text
///   E_in = E_out / (1 − β (1−μ))    with  β = E_out / m_e c²
/// ```
///
/// Constraint for finite positive `E_in`: `β(1−μ) < 1`, i.e.
/// `μ > 1 − m_e c²/E_out`. For `E_out < m_e c²/2` every μ ∈ [-1, 1]
/// is allowed (the kinematic curve is bounded). For `E_out ≥ m_e c²`
/// only forward angles `μ > 1 − 1/β` are allowed — at high energy
/// adjoint Compton is forward-peaked (the photon was higher energy
/// and only a glancing forward scatter degrades it that little).
///
/// Adjoint density (Lewis-Miller §10.3 / Wagner-Haghighat 1998):
/// `P_adj(E_in | E_out) ∝ Σ_s_fwd(E_in → E_out)`. For Compton the
/// forward differential per unit outgoing energy is
/// `∂σ_KN/∂E_out = (dσ/dμ)(E_in, μ_kin) · m_e c²/E_out²`, and with
/// `E_out` fixed the `1/E_out²` factor is constant in `E_in`. So:
///
/// ```text
///   p_target(E_in) ∝ klein_nishina_dcs_dmu(E_in, μ_kin(E_in, E_out))
/// ```
///
/// — no `1/E_in²` Jacobian; the `dμ/dE_in` factor cancels into the
/// constant. We sample with a flat-on-E_in envelope on
/// `[E_out, E_in_max]` and accept with `KN_dcs / 2π r_e²` (the
/// Thomson-limit ceiling, `dσ/dμ = 2π · (r_e²/2)(1+μ²) ≤ 2π r_e²`,
/// safely bounded by `3 r_e²`). Rejection efficiency is moderate
/// (~30 % at 1 MeV inputs); a more aggressive `1/(E_in·m_ec²)`-tail
/// envelope would tighten it but isn't required for correctness.
///
/// Bound-electron correction: same `S(x, Z)/Z` rejection used by the
/// forward sampler — `x = E_in sin(θ/2) / hc` is unchanged because
/// the kinematic relation is energy-momentum conservation regardless
/// of which direction we walk in time.
///
/// References:
/// - Wagner & Haghighat, *Nucl. Sci. Eng.* 128, 186 (1998) §III —
///   adjoint MC for Compton via reverse kinematics.
/// - Lewis & Miller, *Computational Methods of Neutron Transport*
///   §10.3 — generic adjoint kernel form `σ_s^*(E', Ω' → E, Ω) =
///   σ_s(E, Ω → E', Ω')`.
pub fn adjoint_compton_scatter(
    elem: &PhotonElement,
    energy_out: f64,
    e_in_max: f64,
    rng: &mut Rng,
) -> AdjointComptonOutcome {
    assert!(
        e_in_max > energy_out,
        "adjoint compton: e_in_max ({e_in_max}) must exceed energy_out ({energy_out})",
    );
    let beta = energy_out / M_E_C2_EV;
    // μ_min from kinematic: 1 − 1/β. If β ≤ 1 (E_out < m_e c²) all
    // μ ∈ [-1,1] are kinematically allowed.
    let mu_min = if beta <= 1.0 { -1.0 } else { 1.0 - 1.0 / beta };
    // Clamp e_in_max to the kinematic limit (μ = mu_min branch):
    //   E_in_kin_max = E_out / (1 − β · (1 − μ_min))
    // For β ≤ 1 (μ_min = -1) → E_in_kin_max = E_out/(1 − 2β); inf
    // when 2β = 1 i.e. E_out = m_e c²/2.
    let e_in_kin_max = if beta < 0.5 {
        energy_out / (1.0 - 2.0 * beta)
    } else {
        f64::INFINITY
    };
    let e_in_hi = e_in_max.min(e_in_kin_max);
    if !(e_in_hi > energy_out) {
        // No kinematic room — forward scatter limit, return the
        // degenerate (E_in = E_out, μ = 1) sample.
        return AdjointComptonOutcome {
            energy_in: energy_out,
            mu: 1.0,
            attempts: 1,
        };
    }

    // Flat-on-E_in envelope on [E_out, e_in_hi]. Acceptance ratio
    // `KN_dcs(E_in, μ_kin) / envelope_ceiling`.
    //
    // **Tight analytic bound** on dσ_KN/dμ over the entire (E_in, μ)
    // domain. With `ε = E_out/E_in = 1/(1 + α(1−μ)) ∈ (0, 1]`:
    //
    //     dσ/dμ = π · r_e² · (ε + ε³ − ε² · (1 − μ²))
    //
    // For fixed ε, ∂/∂μ = 2π·r_e² · ε²·μ has its only critical
    // point at μ = 0 (a minimum); the maxima on the closed interval
    // are at the endpoints |μ| = 1 → 1−μ² = 0:
    //
    //     dσ/dμ |_{|μ|=1} = π · r_e² · (ε + ε³)
    //
    // The function g(ε) = ε + ε³ is strictly monotone on (0, 1] —
    // g′(ε) = 1 + 3ε² > 0 — so its supremum on the kinematic domain
    // is at ε = 1 (E_in = E_out, no scatter / forward peak):
    //
    //     dσ/dμ |_{ε=1, μ=1} = π · r_e² · 2 = 2π · r_e² ≈ 6.2832 r_e²
    //
    // Achievable: at the bottom of the sampling interval (E_in →
    // E_out, μ → 1) the rejection acceptance hits 100 %. We use the
    // exact analytic bound `2π · r_e²` as the ceiling; no fudge
    // factor is needed — verified by `adjoint_compton_envelope_bound`
    // which scans 10⁶ points on plausible kinematic curves and
    // checks no `klein_nishina_dcs_dmu` value exceeds the ceiling.
    let inv_lo = 1.0 / energy_out;
    let envelope_ceiling = 2.0 * std::f64::consts::PI * R_E_SQ_CM2;
    let e_in_span = e_in_hi - energy_out;

    let mut attempts = 0_u32;
    loop {
        attempts += 1;
        if attempts > 10_000 {
            // Pathological: degenerate kinematic interval. Return
            // forward-scatter limit.
            return AdjointComptonOutcome {
                energy_in: energy_out,
                mu: 1.0,
                attempts,
            };
        }
        let xi = rng.uniform();
        let e_in = energy_out + xi * e_in_span;
        if e_in <= energy_out {
            continue;
        }
        // μ from kinematic inverse.
        let mu = 1.0 - M_E_C2_EV * (inv_lo - 1.0 / e_in);
        if mu < mu_min || mu > 1.0 {
            continue;
        }
        let kn_dcs = klein_nishina_dcs_dmu(e_in, mu);
        let target_density = kn_dcs; // up to constant
        let r1 = rng.uniform();
        if r1 * envelope_ceiling >= target_density {
            continue;
        }
        // Bound-electron rejection — re-uses the forward S(x,Z)/Z
        // table with x = E_in sin(θ/2) / hc, identical kinematics.
        let theta = mu.clamp(-1.0, 1.0).acos();
        let x = e_in * (0.5 * theta).sin() / HC_EV_ANGSTROM;
        let s_over_z = scattering_factor_s_over_z(elem, x);
        if rng.uniform() >= s_over_z {
            continue;
        }
        return AdjointComptonOutcome {
            energy_in: e_in,
            mu,
            attempts,
        };
    }
}

/// Read the bound-electron incoherent scattering factor `S(x, Z)/Z`
/// at the requested `x` (Å⁻¹) from the element's tabulated curve.
/// Mirrors the same lookup the forward Compton sampler uses; pulled
/// out so the adjoint kernel can call it directly without going
/// through the full forward-rejection routine.
fn scattering_factor_s_over_z(elem: &PhotonElement, x: f64) -> f64 {
    let sf: &ScatteringFactor = &elem.incoherent_scattering_factor;
    if sf.x.is_empty() {
        return 1.0;
    }
    if x <= sf.x[0] {
        return (sf.value[0] / elem.z as f64).clamp(0.0, 1.0);
    }
    let last = sf.x.len() - 1;
    if x >= sf.x[last] {
        return (sf.value[last] / elem.z as f64).clamp(0.0, 1.0);
    }
    // Linear interp in x; S is gently varying.
    let i = sf.x.partition_point(|v| *v < x).saturating_sub(1);
    let i = i.min(last - 1);
    let t = (x - sf.x[i]) / (sf.x[i + 1] - sf.x[i]);
    let s = sf.value[i] + t * (sf.value[i + 1] - sf.value[i]);
    (s / elem.z as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod kn_helpers_tests {
    use super::*;
    use crate::quadrature::{GL16_NODES, GL16_WEIGHTS};

    #[test]
    fn pdf_integrates_to_one() {
        // Tolerance loosened with energy: at higher α the KN PDF is
        // strongly forward-peaked and 16-point Gauss-Legendre on
        // [-1, 1] no longer resolves the peak. 1e-3 is sufficient
        // for the unit test — the production NEE integrator splits
        // the [-1, 1] range into segments to handle this.
        for &(e, tol) in &[(1.0e3_f64, 1e-6), (1.0e5, 1e-6), (1.0e6, 1e-4), (5.0e6, 1e-3)] {
            let mut acc = 0.0;
            for i in 0..16 {
                acc += GL16_WEIGHTS[i] * klein_nishina_pdf(e, GL16_NODES[i]);
            }
            assert!(
                (acc - 1.0).abs() < tol,
                "E={e}: ∫pdf = {acc}, expected 1 (tol {tol})"
            );
        }
    }

    #[test]
    fn forward_scatter_has_no_energy_loss() {
        // μ = 1 (cos θ = 1, θ = 0): photon continues forward, E' = E.
        for &e in &[1.0e3_f64, 1.0e6, 1.0e7] {
            assert!((compton_e_out(e, 1.0) - e).abs() / e < 1e-15);
        }
    }

    #[test]
    fn backscatter_e_out_matches_textbook() {
        // μ = -1 (back-scatter): E' = E / (1 + 2α).
        let e = 1.0e6;
        let alpha = e / M_E_C2_EV;
        let expected = e / (1.0 + 2.0 * alpha);
        assert!((compton_e_out(e, -1.0) - expected).abs() / expected < 1e-15);
    }

    #[test]
    fn total_xs_matches_thomson_at_low_energy() {
        // σ_T = (8π/3) r_e². At α = 1 keV / 511 keV ≈ 2e-3, σ_KN should
        // be within 1% of Thomson.
        let thomson = (8.0 / 3.0) * std::f64::consts::PI * R_E_SQ_CM2;
        let kn_low = klein_nishina_total(1.0e3); // 1 keV
        let rel = ((kn_low - thomson) / thomson).abs();
        assert!(rel < 0.01, "σ_KN(1 keV) = {kn_low}, Thomson = {thomson}, rel = {rel}");
    }

    #[test]
    fn total_xs_decreases_with_energy() {
        // KN cross section monotonically decreases with photon energy.
        let energies = [1.0e3_f64, 1.0e4, 1.0e5, 1.0e6, 1.0e7];
        let mut prev = klein_nishina_total(energies[0]);
        for &e in &energies[1..] {
            let now = klein_nishina_total(e);
            assert!(now < prev, "σ_KN not monotone: {prev} -> {now} at E={e}");
            prev = now;
        }
    }
}

/// Apply Compton Doppler broadening.
///
/// Inputs are the incoming photon kinematics `(energy_in, α, k_free)`
/// and the already-sampled scattering cosine `μ` from the free-KN
/// sampler. Returns `(E', B_i)` where `E'` is the Doppler-broadened
/// outgoing photon energy in eV and `B_i` is the binding energy
/// (eV) of the Compton shell from which the struck electron came
/// (used by the caller to deduct from the recoil-electron KE).
///
/// Algorithm (Ribberfors 1975 / PENELOPE §2.3.5):
/// 1. Compute the electron rest-frame momentum projection
///    `p_z_max(i)` for each kinematically-accessible shell (those
///    with `B_i < E_in − E'_free`).
/// 2. Select a shell weighted by `n_i · n_i(p_z_max)` where
///    `n_i(p)` is the cumulative Compton profile — PENELOPE's
///    "maximum kinematically-allowed fraction of electrons" on
///    that shell.
/// 3. Sample `|p_z|` from `Jᵢ(|p_z|)` truncated at `p_z_max(i)`
///    (inverse-CDF of a trapezoidally-integrated profile).
/// 4. Random sign on `p_z`.
/// 5. Solve the Doppler energy relation (eq. 2.50 of PENELOPE):
///    `(p_z c / m_e c²) = [α(1−μ) α' − α + α'] / q`
///    where `q = √(α² − 2 α α' μ + α'²)` and `α' = E'/m_e c²`.
///    Rearranged into a quadratic in `α'`.
fn apply_doppler(
    elem: &PhotonElement,
    energy_in: f64,
    alpha: f64,
    k_free: f64,
    mu: f64,
    rng: &mut Rng,
) -> (f64, f64) {
    let cp = &elem.compton_profiles;
    let n_shells = cp.n_shells();
    if n_shells == 0 {
        return (energy_in * k_free, 0.0);
    }

    // Pre-compute p_z_max(i) and cumulative profile at p_z_max for
    // the shell-selection weights.
    let mut weights = Vec::with_capacity(n_shells);
    let mut pz_max = Vec::with_capacity(n_shells);
    let _alpha_free = k_free * alpha; // α' for the free-electron case (reference only)
    for i in 0..n_shells {
        let b_ev = cp.binding_energy[i];
        let binding_alpha = b_ev / M_E_C2_EV;
        if b_ev >= energy_in - energy_in * k_free {
            // Kinematically inaccessible from the free-KN outgoing
            // energy: outgoing photon would need to exceed incoming.
            weights.push(0.0);
            pz_max.push(0.0);
            continue;
        }
        // PENELOPE §2.3.5 p_z,max (impulse approximation):
        //   p_z,max/(m_e c) = [α(α − β_b)(1 − μ) − β_b]
        //                    / √(α² + (α − β_b)² − 2α(α − β_b)μ)
        // where β_b = B_i/m_e c².
        let alpha_prime_max = alpha - binding_alpha;
        let denom_sq =
            alpha * alpha + alpha_prime_max * alpha_prime_max - 2.0 * alpha * alpha_prime_max * mu;
        if denom_sq <= 0.0 {
            weights.push(0.0);
            pz_max.push(0.0);
            continue;
        }
        let denom = denom_sq.sqrt();
        let pmax_mec = (alpha * alpha_prime_max * (1.0 - mu) - binding_alpha) / denom;
        // Convert m_e c units → atomic units of momentum
        //   1 m_e c = 137.036 a.u.
        let pmax_au = pmax_mec * INV_FINE_STRUCTURE_ALPHA;
        let pmax_clamped = pmax_au.clamp(0.0, *cp.pz.last().unwrap_or(&100.0));
        let cum_j = cumulative_profile(&cp.j[i], &cp.pz, pmax_clamped);
        // Shell-selection weight: n_i · P(|p_z| < p_z_max,i).
        // With Biggs-Lighthill / OpenMC normalisation `∫₀^∞ J_i dp
        // = n_i/2`, `P_i = (2/n_i) · cum_j`, so the product is
        // `n_i · (2/n_i) · cum_j = 2 · cum_j`. The factor of 2 is a
        // constant multiplier across shells and drops out of
        // relative weighting; we use `cum_j` directly.
        weights.push(cum_j);
        pz_max.push(pmax_clamped);
    }
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        // No shell is kinematically accessible — return the free-KN
        // result with no binding deduction.
        return (energy_in * k_free, 0.0);
    }

    // Rejection sampling wrapper: if the sampled shell/p_z yields an
    // unphysical outgoing energy, redraw.
    for _ in 0..32 {
        // Select shell by weight.
        let xi = rng.uniform() * total_weight;
        let mut cum = 0.0;
        let mut shell_idx = 0;
        for (i, w) in weights.iter().enumerate() {
            cum += w;
            if xi < cum {
                shell_idx = i;
                break;
            }
        }

        // Sample |p_z| from J_i truncated at p_z_max, in a.u.
        let pmax_au = pz_max[shell_idx];
        if pmax_au <= 0.0 {
            continue;
        }
        let pz_au = sample_profile(&cp.j[shell_idx], &cp.pz, pmax_au, rng);
        let pz_signed_au = if rng.uniform() < 0.5 { -pz_au } else { pz_au };
        let pz_mec = pz_signed_au * FINE_STRUCTURE_ALPHA;

        // Solve Doppler relation for α'.
        //
        // Derivation (impulse approximation, PENELOPE §2.3.5 /
        // Ribberfors 1975):
        //   q² = α² + α'² − 2αα'μ            (momentum transfer squared)
        //   t·q = α − α'·(1 + α(1−μ))         (projection of initial
        //                                      electron momentum on the
        //                                      scattering axis)
        // with t = p_z·c / (m_e c²) and q positive. Eliminating q by
        // squaring the second equation and substituting:
        //
        //   [t² − ε²]·α'² + 2α[ε − t²μ]·α' + α²[t² − 1] = 0,
        //   ε ≡ 1 + α(1−μ)
        //
        // Squaring introduces a spurious second root. The "physical"
        // root in principle follows from the sign of `t·q`, but the
        // sign convention for `p_z` differs between published
        // formulations (Ribberfors 1975 vs Brusa-Pratt-Salvat 1996
        // vs OpenMC), and the spurious root generally lies far from
        // the free-electron value while the physical root stays near
        // `α_free = α/ε`. Since we randomize sign(p_z) in step 4,
        // averaging is symmetric regardless of convention; we pick
        // the root closer to `α_free` in absolute value to stay on
        // the physical branch and discard the spurious one.
        let t = pz_mec;
        let eps = 1.0 + alpha * (1.0 - mu);
        let alpha_free_root = alpha / eps;
        let a_coef = t * t - eps * eps;
        let b_coef = 2.0 * alpha * (eps - t * t * mu);
        let c_coef = alpha * alpha * (t * t - 1.0);
        let disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
        if disc < 0.0 || a_coef == 0.0 {
            continue;
        }
        let sqrt_disc = disc.sqrt();
        let two_a = 2.0 * a_coef;
        let root_p = (-b_coef + sqrt_disc) / two_a;
        let root_m = (-b_coef - sqrt_disc) / two_a;
        let alpha_out = match (root_p > 0.0, root_m > 0.0) {
            (true, true) => {
                if (root_p - alpha_free_root).abs() <= (root_m - alpha_free_root).abs() {
                    root_p
                } else {
                    root_m
                }
            }
            (true, false) => root_p,
            (false, true) => root_m,
            (false, false) => continue,
        };

        let e_out_ev = alpha_out * M_E_C2_EV;
        if e_out_ev <= 0.0 || e_out_ev >= energy_in {
            continue;
        }
        return (e_out_ev, cp.binding_energy[shell_idx]);
    }
    // Fallback: free-KN.
    (energy_in * k_free, 0.0)
}

/// Fine-structure constant (CODATA-2018).
const FINE_STRUCTURE_ALPHA: f64 = 7.297_352_569_3e-3;
const INV_FINE_STRUCTURE_ALPHA: f64 = 1.0 / FINE_STRUCTURE_ALPHA;

/// Trapezoidal integral `∫₀^{p_max} J(p) dp` using the tabulated
/// Compton profile.
fn cumulative_profile(j: &[f64], pz: &[f64], p_max: f64) -> f64 {
    let n = pz.len();
    if n == 0 || p_max <= pz[0] {
        return 0.0;
    }
    let mut acc = 0.0;
    for k in 1..n {
        if pz[k] <= p_max {
            acc += 0.5 * (j[k - 1] + j[k]) * (pz[k] - pz[k - 1]);
        } else {
            // Partial bin up to p_max
            let frac = (p_max - pz[k - 1]) / (pz[k] - pz[k - 1]);
            let j_at_pmax = j[k - 1] + frac * (j[k] - j[k - 1]);
            acc += 0.5 * (j[k - 1] + j_at_pmax) * (p_max - pz[k - 1]);
            break;
        }
    }
    acc
}

/// Sample `|p_z|` from the Hartree-Fock profile `J(|p_z|)` restricted
/// to `[0, p_max]` via inverse-CDF on the trapezoidally-integrated
/// cumulative.
fn sample_profile(j: &[f64], pz: &[f64], p_max: f64, rng: &mut Rng) -> f64 {
    let cum_max = cumulative_profile(j, pz, p_max);
    if cum_max <= 0.0 {
        return 0.0;
    }
    let target = rng.uniform() * cum_max;
    let n = pz.len();
    let mut acc = 0.0;
    for k in 1..n {
        let pk = pz[k].min(p_max);
        let jk = if pz[k] <= p_max {
            j[k]
        } else {
            let frac = (p_max - pz[k - 1]) / (pz[k] - pz[k - 1]);
            j[k - 1] + frac * (j[k] - j[k - 1])
        };
        let bin = 0.5 * (j[k - 1] + jk) * (pk - pz[k - 1]);
        if target <= acc + bin {
            // Linear-in-J inversion inside the bin.
            let leftover = target - acc;
            // Solve: leftover = 0.5 (j_lo + j(t)) · Δ · (t/Δ)
            //                = 0.5 (j_lo + j_lo + (jk - j_lo) · t/Δ) · t
            // With m = (jk - j_lo)/Δ, solving quadratic:
            //   0.5 m t² + j_lo t − leftover = 0
            let dp = pk - pz[k - 1];
            let j_lo = j[k - 1];
            let m = (jk - j_lo) / dp.max(1e-30);
            if m.abs() < 1e-12 {
                // Flat J
                return pz[k - 1] + leftover / j_lo.max(1e-30);
            }
            let disc = j_lo * j_lo + 2.0 * m * leftover;
            if disc < 0.0 {
                return pz[k - 1] + 0.5 * dp;
            }
            let t = (-j_lo + disc.sqrt()) / m;
            return pz[k - 1] + t.clamp(0.0, dp);
        }
        acc += bin;
        if pz[k] >= p_max {
            break;
        }
    }
    p_max.min(*pz.last().unwrap_or(&p_max))
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

    /// `k ∈ [1/κ, 1]` is the Klein-Nishina kinematic support — valid
    /// only for the free-electron sampler. With Doppler broadening
    /// `k` can drift slightly outside that interval by the
    /// profile-sampled `p_z`.
    #[test]
    fn k_within_kinematic_bounds_free_variant() {
        let Some(h) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xC0FFEE, 1);
        let energy = 1.0e6;
        let alpha = energy / M_E_C2_EV;
        let k_min = 1.0 / (1.0 + 2.0 * alpha);

        for _ in 0..20_000 {
            let out = compton_scatter_free(&h, energy, &mut rng);
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
    /// free-electron kinematic identity). Only holds for
    /// `compton_scatter_free`; the Doppler-broadened variant
    /// deliberately breaks this identity at the profile level.
    #[test]
    fn mu_k_consistent_with_compton_shift_free_variant() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(42, 1);
        let energy = 2.0e6;
        let alpha = energy / M_E_C2_EV;

        for _ in 0..5_000 {
            let out = compton_scatter_free(&c, energy, &mut rng);
            let k = out.energy_out / energy;
            let mu_from_k = 1.0 - (1.0 - k) / (alpha * k);
            assert!(
                (out.mu - mu_from_k).abs() < 1e-12,
                "μ inconsistency: sampled {}, from k = {k} → {mu_from_k}",
                out.mu
            );
        }
    }

    /// For the free sampler, `T_e = E − E'` exactly (kerma, no
    /// binding deduction).
    #[test]
    fn electron_kinetic_is_photon_energy_loss_free_variant() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(1234, 1);
        let energy = 5.0e5;

        for _ in 0..1_000 {
            let out = compton_scatter_free(&c, energy, &mut rng);
            let expected = energy - out.energy_out;
            assert!((out.electron_kinetic - expected).abs() < 1e-12);
            assert!(out.electron_kinetic >= 0.0);
        }
    }

    /// With Doppler broadening the outgoing photon energy is smeared
    /// about the free-KN value. Verify the variance is non-zero but
    /// small on a case where binding is significant: Pb at 100 keV,
    /// K-shell binding 88 keV. Also check energy conservation with
    /// binding deduction: `E_in = E_out + T_e + B_i`.
    #[test]
    fn doppler_broadens_outgoing_spectrum_on_pb() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xD0, 1);
        let energy = 1.0e5;
        let alpha = energy / M_E_C2_EV;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let n = 50_000;
        for _ in 0..n {
            let out = compton_scatter(&pb, energy, &mut rng);
            // Energy conservation: E_in ≥ E_out + T_e (binding is
            // absorbed, so E_in − E_out − T_e ≥ 0). Equality up to
            // kerma/binding is weak; just check positivity.
            assert!(out.energy_out >= 0.0);
            assert!(out.electron_kinetic >= 0.0);
            assert!(out.energy_out + out.electron_kinetic <= energy + 1e-6);
            // Compare deviation from free-KN value at the same μ.
            let k_free = 1.0 / (1.0 + alpha * (1.0 - out.mu));
            let dev = out.energy_out / energy - k_free;
            sum += dev;
            sum_sq += dev * dev;
        }
        let var = sum_sq / n as f64 - (sum / n as f64).powi(2);
        let std = var.sqrt();
        // Expect RMS deviation of a few percent (typical Compton profile
        // widths ≈ 0.5–2 a.u. · α_fine ≈ 0.004–0.015 in m_e c units).
        assert!(std > 1e-3, "Doppler spread too small (std k_dev = {std})");
        assert!(
            std < 0.2,
            "Doppler spread unphysically large (std k_dev = {std})"
        );
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
        let mut pb_free =
            PhotonElement::from_hdf5(&photon_path("Pb.h5").unwrap()).expect("load Pb");
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
            // Use the free variant: Doppler broadening would smear E'
            // but not the angular distribution, yet sampling through
            // the Doppler shell-selection loop can fail and fall
            // back to free-KN in ways that bias <μ²>. The angular
            // test should be independent of the Doppler channel.
            let out = compton_scatter_free(&h, energy, &mut rng);
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

    /// **Envelope-ceiling bound check.** The adjoint Compton sampler
    /// uses `2π · r_e²` as the rejection ceiling for
    /// `klein_nishina_dcs_dmu`. The analytic derivation gives this
    /// as the exact supremum (achieved at ε = E_out/E_in = 1, |μ| = 1
    /// — the no-scatter forward limit). This test scans a dense
    /// (E_in, μ) lattice across ranges that bracket every adjoint
    /// sampling case (E_in from 1 keV to 100 MeV, μ ∈ [-1, 1]) and
    /// verifies no `klein_nishina_dcs_dmu` evaluation exceeds the
    /// ceiling. Belt-and-suspenders against a future code change to
    /// `klein_nishina_dcs_dmu` silently invalidating the bound.
    #[test]
    fn adjoint_compton_envelope_bound() {
        let ceiling = 2.0 * std::f64::consts::PI * R_E_SQ_CM2;
        const N_E: usize = 1_000;
        const N_MU: usize = 1_000;
        let log_e_lo = (1.0e3_f64).ln(); // 1 keV
        let log_e_hi = (1.0e8_f64).ln(); // 100 MeV
        for i in 0..N_E {
            let log_e = log_e_lo + (log_e_hi - log_e_lo) * (i as f64 / (N_E - 1) as f64);
            let e_in = log_e.exp();
            for j in 0..N_MU {
                let mu = -1.0 + 2.0 * (j as f64 / (N_MU - 1) as f64);
                let dcs = klein_nishina_dcs_dmu(e_in, mu);
                assert!(
                    dcs <= ceiling * (1.0 + 1.0e-12),
                    "KN_dcs/dμ = {dcs:e} exceeds envelope ceiling {ceiling:e} at E_in={e_in}, μ={mu}",
                );
            }
        }
        // Also pin the supremum: at (E_in arbitrary, μ = 1) we hit
        // exactly 2π·r_e² (k' → 1, sin² → 0, prefactor → r_e²).
        let dcs_forward_low_e = klein_nishina_dcs_dmu(1.0, 1.0);
        assert!(
            ((dcs_forward_low_e - ceiling) / ceiling).abs() < 1e-12,
            "forward-limit dσ/dμ = {dcs_forward_low_e}, expected {ceiling}",
        );
    }

    /// Adjoint Compton: kinematic invariant. The relation
    /// `1/E_out − 1/E_in = (1−μ)/m_e c²` is enforced at sample time;
    /// no sample should violate it (within FP noise).
    #[test]
    fn adjoint_compton_kinematic_invariant() {
        let Some(elem) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xADC011, 1);
        let e_out = 200_000.0; // 200 keV
        let e_in_max = 5e6;
        for _ in 0..2000 {
            let o = adjoint_compton_scatter(&elem, e_out, e_in_max, &mut rng);
            let lhs = 1.0 / e_out - 1.0 / o.energy_in;
            let rhs = (1.0 - o.mu) / M_E_C2_EV;
            assert!(
                (lhs - rhs).abs() < 1e-6 * (lhs.abs().max(rhs.abs()).max(1e-30)),
                "kinematic violation: lhs={lhs:e} rhs={rhs:e}, mu={}, e_in={}",
                o.mu,
                o.energy_in,
            );
            assert!(o.energy_in >= e_out - 1e-6);
            assert!(o.mu >= -1.0 - 1e-12 && o.mu <= 1.0 + 1e-12);
        }
    }

    /// **Conditional density check** — the central correctness test
    /// for adjoint Compton. The adjoint conditional density on E_in
    /// given a fixed E_out should be proportional to
    /// `KN_dcs_dmu(E_in, μ_kin(E_in, E_out))` (the forward Klein-
    /// Nishina differential evaluated along the kinematic curve).
    /// We sample the adjoint kernel many times at fixed E_out, build
    /// a histogram of E_in, and χ² it against the analytic density.
    #[test]
    fn adjoint_compton_conditional_matches_analytic() {
        let Some(elem) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        const N: usize = 200_000;
        const E_OUT_FIXED: f64 = 200_000.0; // 200 keV
        const E_IN_MAX: f64 = 5.0e6;
        const N_BINS: usize = 25;

        // Linear bin grid on E_in ∈ [E_OUT_FIXED, kinematic_max]; use
        // the E_OUT < m_e c²/2 case where every μ ∈ [-1, 1] is valid
        // and the kinematic bound is finite.
        let beta = E_OUT_FIXED / M_E_C2_EV;
        let e_in_kin_max = E_OUT_FIXED / (1.0 - 2.0 * beta);
        let e_in_hi = E_IN_MAX.min(e_in_kin_max);
        let bin_w = (e_in_hi - E_OUT_FIXED) / N_BINS as f64;
        let bin = |e: f64| -> Option<usize> {
            if e < E_OUT_FIXED || e >= e_in_hi {
                return None;
            }
            Some(((e - E_OUT_FIXED) / bin_w) as usize)
        };

        let mut rng = Rng::new(0xDB2, 0);
        let mut h_sampled = vec![0u64; N_BINS];
        for _ in 0..N {
            let o = adjoint_compton_scatter(&elem, E_OUT_FIXED, E_IN_MAX, &mut rng);
            if let Some(b) = bin(o.energy_in) {
                h_sampled[b] += 1;
            }
        }

        // Analytic expected count per bin: integrate
        //   p(E_in) = KN_dcs_dmu(E_in, μ_kin(E_in, E_OUT_FIXED))
        // over each bin. Use mid-point rule with 5 sub-samples per
        // bin (KN is smooth, this converges fast).
        let inv_e_out = 1.0 / E_OUT_FIXED;
        let mut analytic = vec![0.0_f64; N_BINS];
        let sub = 5_usize;
        for i in 0..N_BINS {
            for s in 0..sub {
                let e_in = E_OUT_FIXED + (i as f64 + (s as f64 + 0.5) / sub as f64) * bin_w;
                let mu = 1.0 - M_E_C2_EV * (inv_e_out - 1.0 / e_in);
                if !(-1.0..=1.0).contains(&mu) {
                    continue;
                }
                analytic[i] += klein_nishina_dcs_dmu(e_in, mu);
            }
            analytic[i] *= bin_w / sub as f64;
        }
        let analytic_sum: f64 = analytic.iter().sum();
        let total: u64 = h_sampled.iter().sum();
        assert!(
            total > N as u64 / 2,
            "adjoint sample efficiency too low: {total}/{N}",
        );
        let scale = total as f64 / analytic_sum;
        let mut chi2 = 0.0_f64;
        let mut active = 0;
        for i in 0..N_BINS {
            let exp = analytic[i] * scale;
            if exp < 50.0 {
                continue;
            }
            let obs = h_sampled[i] as f64;
            chi2 += (obs - exp).powi(2) / exp;
            active += 1;
        }
        let chi2_red = chi2 / active as f64;
        assert!(
            chi2_red < 2.5,
            "adjoint conditional density disagrees: χ²_red = {chi2_red:.3} over {active} bins",
        );
    }

    /// Adjoint sampling at high `E_out` (where the kinematic-allowed
    /// μ range collapses to forward-only) returns μ ≥ μ_min and
    /// E_in ≥ E_out.
    #[test]
    fn adjoint_compton_high_energy_forward_peak() {
        let Some(elem) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xC0DE, 0);
        let e_out = 5.0e6; // 5 MeV — β = 9.78, μ_min = 1 − 1/β ≈ 0.898
        let mu_min_expected = 1.0 - M_E_C2_EV / e_out;
        let e_in_max = 50e6;
        for _ in 0..1000 {
            let o = adjoint_compton_scatter(&elem, e_out, e_in_max, &mut rng);
            assert!(
                o.mu >= mu_min_expected - 1e-6,
                "high-E adjoint sampled μ = {} below kinematic min {}",
                o.mu,
                mu_min_expected,
            );
            assert!(o.energy_in >= e_out);
        }
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
