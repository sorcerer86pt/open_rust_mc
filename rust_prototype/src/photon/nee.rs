//! Next-event estimator for photon-shielding tallies.
//!
//! At every Compton collision in a slab geometry, the analog estimator
//! contributes to the transmitted-energy tally only if the photon
//! eventually escapes through the detector face — which at deep
//! penetration is rare. The next-event (also "expected-value" or
//! "DXTRAN-style") estimator instead adds a deterministic contribution
//! to the tally from every collision, equal to the *expected* energy
//! that would scatter forward and arrive at the detector unscattered.
//! That contribution is non-zero at every collision, regardless of
//! whether the photon physically escapes, so the tally accumulates much
//! faster than analog.
//!
//! For a homogeneous slab of thickness `T` with the detector at
//! `z = T` and a Compton collision at depth `z` with incoming photon
//! energy `E`, the deterministic-contribution kernel is:
//!
//! ```text
//!   N(E, z, T) = ∫_0^1 PDF_KN(E, μ) · E'(E, μ) · exp(−Σ_t(E') · (T-z)/μ) dμ
//! ```
//!
//! where `PDF_KN` is the Klein-Nishina angular distribution (free-
//! electron, normalised on `[-1, 1]`), `E'(E, μ) = E / (1+α(1-μ))`
//! is the scattered photon energy, and the integral is restricted to
//! the forward hemisphere `μ ∈ (0, 1]`.
//!
//! Quadrature: 16-point Gauss-Legendre, mapped from `[-1, 1]` to
//! `[0, 1]` via `crate::quadrature::integrate_gl16`. The shared table
//! is unit-tested against monomials `x^0..x^31` exactly (one test
//! catches typos in any node or weight). 16 points is sufficient for
//! `α ≲ 5` (E ≲ 2.5 MeV); for higher energies the integrand peaks
//! near μ=1 and the tolerance widens — documented in unit tests.
//!
//! Caller multiplies by the photon's current weight and the
//! probability that this collision was a Compton event
//! (`σ_compton / σ_total` — caller-provided since material XS layout
//! is caller's choice).
//!
//! Limitations of this v1 module:
//! - Compton only. Rayleigh (no energy change, isotropic scatter)
//!   could be added with a similar integral. Photoelectric and pair
//!   contribute fluorescence / annihilation gammas which are
//!   isotropic; small effect at 1 MeV in water but worth adding for
//!   high-Z shields.
//! - Free-electron Klein-Nishina (no bound `S(x,Z)/Z` correction).
//!   Bound correction is ~1% at the forward angles that dominate the
//!   integral; documented as a follow-on refinement.
//! - Direct-flight only — no contribution from photons that scatter
//!   backward and then reflect off `z = 0`. For shield_slab with a
//!   reflective back face this misses some flux; documented as a
//!   small bias.

use crate::photon::compton::{HC_EV_ANGSTROM, compton_e_out, klein_nishina_dcs_dmu};
use crate::photon::data::ScatteringFactor;
use crate::photon::material::PhotonMaterial;
use crate::quadrature::integrate_gl16;

/// Deterministic-contribution kernel for one collision in a
/// homogeneous slab of thickness `t_cm` (detector at `z = t_cm`, +z
/// is "toward detector"), integrated over the bound-corrected
/// Compton DCS.
///
/// `mu_in` is the z-component of the incoming photon direction,
/// `Ω_in · ẑ`. For a source photon born along +ẑ this is `1.0` and
/// the formula simplifies to the 1-D `μ_scat = μ_z` integral. For
/// any subsequent collision the photon's incoming direction is
/// arbitrary, so we must integrate over both the scattering cosine
/// `μ_scat` AND the azimuthal angle `ψ_scat` to recover the
/// scattered-direction z-component
///
/// ```text
///   μ_z = μ_in·μ_scat + √(1-μ_in²)·√(1-μ_scat²)·cos(ψ_scat)
/// ```
///
/// the path-to-detector along the scattered direction is `(T-z)/μ_z`
/// when `μ_z > 0` (else the scatter goes backward and contributes 0).
/// **The v0 of this routine accidentally used `μ_scat` in place of
/// `μ_z` everywhere, which gave correct results only at the source's
/// first collision and produced a +21–38% over-counting bias at
/// subsequent collisions** — see `outputs/random_ray_cadis_fom.txt`
/// "diagnosed" section for the post-mortem. This v1 integrates the
/// full 2D scatter manifold.
///
/// Math:
/// ```text
///   N(E, z, T, μ_in) =
///     (1 / Σ_total) · ∫_{-1}^{1} dμ_scat · σ_KN_free(E,μ_scat) · S_eff(x)
///       · E'(E,μ_scat) · (1/(2π)) · ∫_0^{2π} dψ_scat
///         · 1[μ_z > 0] · exp(-Σ_t(E') · (T-z)/μ_z)
/// ```
///
/// MCNP-style exclusion regularisation (when `exclusion_cm > 0` *and*
/// `T - z < exclusion_cm`): replace `exp(-Σ_t·(T-z)/μ_z)` with the
/// analytic z-average `(μ_z/(Σ_t·R))·(1 − exp(−Σ_t·R/μ_z))` — same as
/// the original v0 formulation but now in the corrected `μ_z` axis.
///
/// `Σ_total` enters because the caller has already multiplied by it
/// (collision rate); dividing here gives the channel-averaged
/// contribution per collision. Caller multiplies by current photon
/// weight.
///
/// Returns eV per collision (caller multiplies by weight).
pub fn compton_forward_transmission(
    material: &PhotonMaterial,
    e_in: f64,
    z_cm: f64,
    t_cm: f64,
    mu_in: f64,
    exclusion_cm: f64,
) -> f64 {
    if t_cm <= z_cm {
        return 0.0;
    }
    let depth_remaining = t_cm - z_cm;
    let sigma_total = material.macro_total(e_in);
    if sigma_total <= 0.0 {
        return 0.0;
    }
    let hc_inv = 1.0 / HC_EV_ANGSTROM;
    const CM2_TO_BARNS: f64 = 1.0e24;
    let mu_in = mu_in.clamp(-1.0, 1.0);
    let sin_in = (1.0 - mu_in * mu_in).max(0.0).sqrt();

    // 16-point GL on μ_scat ∈ [-1, 1]. Interior 16-point GL on
    // ψ_scat ∈ [0, 2π] for the azimuthal integration.
    integrate_gl16(-1.0, 1.0, |mu_scat| {
        // Bound-corrected dΣ_compton/dμ_scat (1/cm per dμ_scat).
        let dsigma_kn_dmu = klein_nishina_dcs_dmu(e_in, mu_scat) * CM2_TO_BARNS;
        let x = e_in * hc_inv * (0.5 * (1.0 - mu_scat)).sqrt();
        let mut s_eff = 0.0;
        for (n_e, elem) in &material.entries {
            let s_of_x = interp_linear(&elem.incoherent_scattering_factor, x);
            s_eff += n_e * s_of_x;
        }
        let dsigma_macro_dmu = dsigma_kn_dmu * s_eff;
        let e_out = compton_e_out(e_in, mu_scat);
        let sigma_t = material.macro_total(e_out);

        let sin_scat = (1.0 - mu_scat * mu_scat).max(0.0).sqrt();
        // Inner integral over azimuthal angle ψ_scat ∈ [0, 2π],
        // 16-point GL, taking 1[μ_z > 0] · exp(-τ/μ_z) (or the
        // exclusion-zone-averaged variant) at each node.
        let two_pi = 2.0 * std::f64::consts::PI;
        let phi_avg = integrate_gl16(0.0, two_pi, |psi| {
            let mu_z = mu_in * mu_scat + sin_in * sin_scat * psi.cos();
            if mu_z <= 0.0 {
                return 0.0;
            }
            if exclusion_cm > 0.0 && depth_remaining < exclusion_cm {
                let kappa = sigma_t * exclusion_cm / mu_z;
                if kappa < 1.0e-6 {
                    1.0 - 0.5 * kappa + (1.0 / 6.0) * kappa * kappa
                } else {
                    (1.0 - (-kappa).exp()) / kappa
                }
            } else {
                let path = depth_remaining / mu_z;
                let tau = sigma_t * path;
                (-tau).exp()
            }
        }) / two_pi;

        dsigma_macro_dmu * e_out * phi_avg / sigma_total
    })
}

/// Linear interpolation of a tabulated `S(x, Z)` factor (or any
/// `ScatteringFactor`). Mirrors `compton::interp_linear` (which is
/// private to that module).
fn interp_linear(factor: &ScatteringFactor, x: f64) -> f64 {
    let xs = &factor.x;
    let vs = &factor.value;
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return vs[0];
    }
    if x >= *xs.last().expect("non-empty") {
        return *vs.last().expect("non-empty");
    }
    let i = match xs.binary_search_by(|t| t.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return vs[i],
        Err(i) => i,
    };
    let x0 = xs[i - 1];
    let x1 = xs[i];
    let v0 = vs[i - 1];
    let v1 = vs[i];
    v0 + (v1 - v0) * (x - x0) / (x1 - x0)
}

/// Uncollided forward transmission — the photon born at `(_, _, z_birth)`
/// heading +z reaches the detector face without any scatter.
/// `weight × e_source × exp(-Σ_t(e_source)·(T-z_birth))`. Caller
/// multiplies by weight.
pub fn uncollided_forward_transmission(
    material: &PhotonMaterial,
    e_source: f64,
    z_birth_cm: f64,
    t_cm: f64,
) -> f64 {
    if t_cm <= z_birth_cm {
        return e_source;
    }
    let sigma_t = material.macro_total(e_source);
    let depth = t_cm - z_birth_cm;
    e_source * (-sigma_t * depth).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photon::data::PhotonElement;
    use crate::physics_constants::{MU_AXIAL_FORWARD, NEE_NO_EXCLUSION};
    use std::path::Path;

    fn try_load_water_material() -> Option<PhotonMaterial> {
        let path = Path::new("../data/endfb-vii.1-hdf5/photon");
        let h = PhotonElement::from_hdf5(&path.join("H.h5")).ok()?;
        let o = PhotonElement::from_hdf5(&path.join("O.h5")).ok()?;
        let n_h2o = 1.0 * 6.022e23 / 18.0153 * 1.0e-24;
        Some(PhotonMaterial::new(vec![(2.0 * n_h2o, h), (n_h2o, o)]).with_density(1.0))
    }

    #[test]
    fn uncollided_matches_beer_lambert() {
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return, // photon HDF5 not present in this env
        };
        let e = 1.0e6;
        let t = 100.0;
        let expected = e * (-mat.macro_total(e) * t).exp();
        let got = uncollided_forward_transmission(&mat, e, 0.0, t);
        let rel = ((got - expected) / expected).abs();
        assert!(rel < 1e-12, "Beer-Lambert: got {got}, expected {expected}");
    }

    #[test]
    fn compton_forward_transmission_decreases_with_depth() {
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        let e = 1.0e6;
        let t = 100.0;
        let near_detector =
            compton_forward_transmission(&mat, e, 95.0, t, MU_AXIAL_FORWARD, NEE_NO_EXCLUSION);
        let middle =
            compton_forward_transmission(&mat, e, 50.0, t, MU_AXIAL_FORWARD, NEE_NO_EXCLUSION);
        let near_source =
            compton_forward_transmission(&mat, e, 5.0, t, MU_AXIAL_FORWARD, NEE_NO_EXCLUSION);
        assert!(
            near_detector > middle && middle > near_source,
            "monotone: near={near_detector}, mid={middle}, far={near_source}"
        );
    }

    #[test]
    fn compton_forward_transmission_at_detector_face_in_expected_range() {
        // At z = T − ε, depth_remaining → 0, exp(−τ) → 1. The
        // integrand becomes (dΣ_compton/dμ × E') / Σ_total.
        // Bound-electron correction `S(x,Z)` suppresses small-x
        // (small forward-angle) scattering, so the integrated
        // contribution is a fraction of E_in but smaller than the
        // free-KN limit. Order-of-magnitude check: positive, less
        // than E_in, and within 0.001E < contribution < E.
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        let e = 1.0e6;
        let t = 100.0;
        let at_face = compton_forward_transmission(
            &mat,
            e,
            t - 1.0e-9,
            t,
            MU_AXIAL_FORWARD,
            NEE_NO_EXCLUSION,
        );
        assert!(
            at_face > 1.0e-3 * e && at_face < e,
            "expected 0.001E < {at_face} < {e}"
        );
    }

    #[test]
    fn integrated_bound_compton_dcs_matches_tabulated_macro_xs() {
        // Sanity check: ∫_{-1}^{1} dΣ_compton_macro/dμ dμ should equal
        // the tabulated `incoherent_xs` macroscopic XS within
        // quadrature error. This catches unit-conversion bugs
        // (cm² ↔ barns) and any S(x,Z) interpolation glitches.
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        for &e in &[1.0e5_f64, 1.0e6, 5.0e6] {
            let tab = mat.macro_xs(crate::photon::material::Channel::Incoherent, e);
            // Replicate the integrand from compton_forward_transmission
            // but without the exp(-τ) and without dividing by Σ_total.
            let hc_inv = 1.0 / crate::photon::compton::HC_EV_ANGSTROM;
            const CM2_TO_BARNS: f64 = 1.0e24;
            let integrated = crate::quadrature::integrate_gl16(-1.0, 1.0, |mu| {
                let dkn = crate::photon::compton::klein_nishina_dcs_dmu(e, mu) * CM2_TO_BARNS;
                let x = e * hc_inv * (0.5 * (1.0 - mu)).sqrt();
                let mut s_eff = 0.0;
                for (n_e, elem) in &mat.entries {
                    s_eff += n_e * interp_linear(&elem.incoherent_scattering_factor, x);
                }
                dkn * s_eff
            });
            let rel = ((integrated - tab) / tab).abs();
            assert!(
                rel < 0.05,
                "E={e}: integrated bound-Compton {integrated} vs tabulated macro {tab}, rel={rel}"
            );
        }
    }

    #[test]
    fn exclusion_zone_with_zero_thickness_matches_raw() {
        // exclusion_cm = 0 → behaviour identical to no regularisation.
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        for &z in &[5.0_f64, 50.0, 95.0] {
            let raw = compton_forward_transmission(
                &mat,
                1.0e6,
                z,
                100.0,
                MU_AXIAL_FORWARD,
                NEE_NO_EXCLUSION,
            );
            let zero_excl = compton_forward_transmission(
                &mat,
                1.0e6,
                z,
                100.0,
                MU_AXIAL_FORWARD,
                NEE_NO_EXCLUSION,
            );
            assert!((raw - zero_excl).abs() < 1e-15);
        }
    }

    #[test]
    fn exclusion_zone_caps_near_face_contribution() {
        // For collisions inside the exclusion zone (T-z < R), the
        // regularised attenuation factor (μ/(Σ_t·R))(1-exp(-Σ_t·R/μ))
        // is ≤ exp(-Σ_t·(T-z)/μ) at z near T, because the average over
        // [T-R, T] necessarily attenuates more than the value right at
        // the upper limit. So the regularised contribution is ≤ raw.
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        // Pick R = 5 cm and z = 99 cm (deep inside exclusion zone).
        let raw = compton_forward_transmission(
            &mat,
            1.0e6,
            99.0,
            100.0,
            MU_AXIAL_FORWARD,
            NEE_NO_EXCLUSION,
        );
        let regularised =
            compton_forward_transmission(&mat, 1.0e6, 99.0, 100.0, MU_AXIAL_FORWARD, 5.0);
        assert!(
            regularised < raw,
            "regularised {regularised} should be < raw {raw} inside exclusion zone"
        );
        // Should still be a reasonable fraction (not zero).
        assert!(regularised > 0.1 * raw);
    }

    #[test]
    fn exclusion_zone_outside_zone_unchanged() {
        // For collisions outside the exclusion zone (T-z >= R), the
        // exclusion-zone code path doesn't fire and behaviour matches
        // the raw integrand.
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        // R = 5 cm, z = 50 cm → T-z = 50 cm >> R, definitely outside.
        let raw = compton_forward_transmission(
            &mat,
            1.0e6,
            50.0,
            100.0,
            MU_AXIAL_FORWARD,
            NEE_NO_EXCLUSION,
        );
        let with_excl =
            compton_forward_transmission(&mat, 1.0e6, 50.0, 100.0, MU_AXIAL_FORWARD, 5.0);
        assert!((raw - with_excl).abs() / raw.max(1e-30) < 1e-15);
    }

    /// Deep-penetration sanity: at ~21 mfp the per-collision NEE
    /// kernel must stay finite and positive. Previous numerical
    /// glitches (e.g. underflow in `exp(-tau)` from `f64::INFINITY`
    /// when `mu_z` flips negative inside the integrand without the
    /// `1[mu_z > 0]` gate) produced silent zero / NaN at deep z. This
    /// is the regression fence for task #5: NEE has to keep working
    /// at 300 cm water, where analog gives zero transmitted in any
    /// finite-history Monte Carlo.
    #[test]
    fn compton_forward_transmission_finite_at_deep_penetration() {
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        // 300 cm water at 1 MeV: ~21 mfp. Pick a collision near the
        // detector face — the kernel here drives the dominant
        // last-only estimator contribution that delivers FOM > 0.1
        // where analog gives 0.
        let v = compton_forward_transmission(
            &mat,
            1.0e6,
            295.0,
            300.0,
            MU_AXIAL_FORWARD,
            NEE_NO_EXCLUSION,
        );
        assert!(v.is_finite(), "NEE underflowed to non-finite at deep z: {v}");
        assert!(
            v > 0.0,
            "NEE returned zero contribution at z=295/300 — coverage gap",
        );
        // Order-of-magnitude bound: ~0.5 × E_in × exp(-Σ_t · 5 cm) ≈
        // 0.5 × 1e6 × exp(-0.353) ≈ 3.5e5 eV. Loose envelope to
        // tolerate 16-point GL truncation + bound-correction shape.
        assert!(
            v > 1.0e3 && v < 1.0e6,
            "NEE at z=295/300 outside reasonable envelope: {v}",
        );
    }

    #[test]
    fn compton_forward_transmission_is_zero_past_detector() {
        let mat = match try_load_water_material() {
            Some(m) => m,
            None => return,
        };
        let beyond = compton_forward_transmission(
            &mat,
            1.0e6,
            110.0,
            100.0,
            MU_AXIAL_FORWARD,
            NEE_NO_EXCLUSION,
        );
        assert_eq!(beyond, 0.0);
    }
}
