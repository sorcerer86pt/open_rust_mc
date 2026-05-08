//! Predictor-corrector depletion (CE/LI — Constant flux Extrapolation
//! / Linear Interpolation, Isotalo-Aarnio 2011).
//!
//! Given a beginning-of-step (BOC) composition `N₀`, BOC flux `φ₀`,
//! and a callback that produces the end-of-step flux `φ̂` from a
//! candidate composition (a transport solve under the hood), this
//! routine performs:
//!
//!   1. **Predictor (CE):** integrate from `t` to `t + Δt` with
//!      constant `A₀ = A(N₀, φ₀)` →  `N̂ = exp(A₀ · Δt) · N₀`.
//!   2. **Flux update:** call `flux_at(N̂)` to get the EOC flux `φ̂`.
//!   3. **Corrector (LI):** rebuild `Â = A(N̂, φ̂)` (linear-
//!      interpolation flavour: average the BOC and EOC matrices),
//!      integrate again with the averaged matrix:
//!     `N(t + Δt) = exp(½ (A₀ + Â) · Δt) · N₀`.
//!
//! The corrector exists to remove the leading error from the
//! constant-`A` predictor (~Δt² instead of Δt). It costs one extra
//! transport solve per step, which is the dominant runtime expense.
//! For coarse time steps (days–weeks at PWR power) the corrector
//! changes most isotopics by 1 to 5 %.

use crate::depletion::chain::DepletionChain;
use crate::depletion::cram::{CramOrder, cram};
use crate::depletion::matrix::{TransmutationInputs, build_transmutation_matrix};

/// The result of one predictor-corrector step. `predicted` is the
/// CE estimate (cheap, single matrix solve); `corrected` is the
/// CE/LI estimate after one corrector pass.
#[derive(Debug, Clone)]
pub struct DepletionStep {
    pub predicted: Vec<f64>,
    pub corrected: Vec<f64>,
    /// EOC flux returned by the user-supplied callback. Useful for
    /// tracking flux evolution across a multi-step run.
    pub eoc_flux: f64,
}

/// One CE/LI step. `n0` is the BOC composition (atom densities or
/// raw atom counts — CRAM is linear so the choice is consistent
/// throughout). `order` selects the CRAM approximation order
/// (16 is the default for PWR-typical Δt; 48 for stiff activation /
/// shutdown decay calcs). `flux_at` is the user's transport-solve
/// callback: given a candidate composition, return the one-group
/// flux for the next step. Pass `|_| flux_boc` to skip the
/// transport solve and run constant-flux CE/LI (corrector and
/// predictor agree by construction).
pub fn deplete_ce_li<F>(
    chain: &DepletionChain,
    n0: &[f64],
    flux_boc: f64,
    dt_seconds: f64,
    order: CramOrder,
    mut flux_at: F,
) -> DepletionStep
where
    F: FnMut(&[f64]) -> f64,
{
    assert_eq!(
        n0.len(),
        chain.len(),
        "n0 length {} does not match chain length {}",
        n0.len(),
        chain.len()
    );

    // Predictor: A₀ · Δt, then CRAM.
    let a0 = build_transmutation_matrix(
        &TransmutationInputs {
            chain,
            flux: flux_boc,
        },
        dt_seconds,
    );
    let predicted = cram(order, &a0, n0);

    // EOC flux from user callback (the transport solve).
    let flux_eoc = flux_at(&predicted);

    // Corrector: average matrix from BOC and EOC. Building two
    // matrices and averaging is exactly equivalent to building one
    // matrix at the average flux when the chain's reaction rates
    // are linear in flux (which they are here) — but we keep the
    // explicit average so non-linear extensions slot in cleanly.
    let a_eoc = build_transmutation_matrix(
        &TransmutationInputs {
            chain,
            flux: flux_eoc,
        },
        dt_seconds,
    );
    let avg: Vec<f64> = a0
        .iter()
        .zip(a_eoc.iter())
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let corrected = cram(order, &avg, n0);

    DepletionStep {
        predicted,
        corrected,
        eoc_flux: flux_eoc,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::depletion::chain::{DecayBranch, NuclideEntry};
    use std::collections::HashMap;

    /// Xe-135 equilibrium with a *constant* fission source — the
    /// classical textbook test. We model only I-135 + Xe-135 + Cs-135,
    /// driving fission-product production via a fixed external rate
    /// (so U-235 burnup is decoupled and the analytical equilibrium
    /// is exactly reachable).
    #[test]
    fn xe135_equilibrium_matches_analytical_constant_source() {
        // I-135 → Xe-135 → Cs-135. Production rates are folded into
        // the matrix as decay-from-virtual-parent terms via a fake
        // "Σ_f source" nuclide that sits at huge constant inventory
        // and decays into I-135 / Xe-135 with rates set to match
        // (γ_I + γ_Xe) · Σ_f · φ. This keeps the chain linear.
        let lambda_i = 2.926_400e-5_f64;
        let lambda_xe = 2.106_530e-5_f64;
        let xe_capture_rate = 2.65e6 * 1.0e-24 * 3.0e14; // σ φ in s⁻¹

        // Source rates (atoms / cm³ / s) for I-135 and Xe-135.
        let r_i = 1.0e10_f64; // arbitrary; equilibrium ratio is independent
        let r_xe = 0.04 * r_i; // γ_Xe / γ_I ≈ 0.04 for U-235 thermal

        // Analytical equilibrium:
        //   N_I^eq  = R_I / λ_I
        //   N_Xe^eq = (R_I + R_Xe) / (λ_Xe + σ_a φ)
        // (the Xe gets all the I that decays, plus its own direct yield)
        let n_i_eq = r_i / lambda_i;
        let n_xe_eq = (r_i + r_xe) / (lambda_xe + xe_capture_rate);

        // Encode the source as a "S" nuclide with huge inventory and
        // a tiny effective decay constant chosen so its decay rate
        // delivers exactly r_i + r_xe atoms per second per cm³ of S.
        let mut chain = DepletionChain::new();
        let s_inventory = 1.0e30; // arbitrary large
        let lambda_s = (r_i + r_xe) / s_inventory;
        let frac_i = r_i / (r_i + r_xe);
        let frac_xe = 1.0 - frac_i;
        chain.add_nuclide(NuclideEntry {
            name: "S".into(),
            zaid: 999_999,
            decay_constant: lambda_s,
            decay_branches: vec![
                DecayBranch {
                    daughter_zaid: 53135,
                    branch_ratio: frac_i,
                },
                DecayBranch {
                    daughter_zaid: 54135,
                    branch_ratio: frac_xe,
                },
            ],
        });
        chain.add_nuclide(NuclideEntry {
            name: "I-135".into(),
            zaid: 53135,
            decay_constant: lambda_i,
            decay_branches: vec![DecayBranch {
                daughter_zaid: 54135,
                branch_ratio: 1.0,
            }],
        });
        chain.add_nuclide(NuclideEntry {
            name: "Xe-135".into(),
            zaid: 54135,
            decay_constant: lambda_xe,
            decay_branches: vec![DecayBranch {
                daughter_zaid: 55135,
                branch_ratio: 1.0,
            }],
        });
        chain.add_nuclide(NuclideEntry {
            name: "Cs-135".into(),
            zaid: 55135,
            decay_constant: 0.0,
            decay_branches: vec![],
        });
        // Xe-135 (n,γ) → off-chain.
        chain.add_reaction(54135, 102, 2.65e6, Some(HashMap::new()));

        let s_idx = chain.index_of_zaid(999_999).unwrap();
        let i_idx = chain.index_of_zaid(53135).unwrap();
        let xe_idx = chain.index_of_zaid(54135).unwrap();

        // Burn for 5 days — well past Xe-135 equilibration timescale
        // (~1 / λ_Xe ≈ 13 h). Substep into 30 corrector passes.
        let total_seconds = 5.0 * 86_400.0;
        let n_steps = 30;
        let dt = total_seconds / n_steps as f64;
        let flux = 3.0e14;
        let mut composition = vec![0.0_f64; chain.len()];
        composition[s_idx] = s_inventory;
        for _ in 0..n_steps {
            let step = deplete_ce_li(&chain, &composition, flux, dt, CramOrder::Cram16, |_| flux);
            composition = step.corrected;
        }

        let rel_err_i = (composition[i_idx] - n_i_eq).abs() / n_i_eq;
        let rel_err_xe = (composition[xe_idx] - n_xe_eq).abs() / n_xe_eq;
        assert!(
            rel_err_i < 1e-4,
            "N_I-135 = {:.4e}, expected {:.4e}, rel_err = {:.2e}",
            composition[i_idx],
            n_i_eq,
            rel_err_i
        );
        assert!(
            rel_err_xe < 1e-4,
            "N_Xe-135 = {:.4e}, expected {:.4e}, rel_err = {:.2e}",
            composition[xe_idx],
            n_xe_eq,
            rel_err_xe
        );
    }

    /// Pure-decay system: corrector should agree with predictor to
    /// machine precision (no flux dependence, both A's are
    /// identical).
    #[test]
    fn pure_decay_corrector_equals_predictor() {
        let mut chain = DepletionChain::new();
        chain.add_nuclide(NuclideEntry {
            name: "I-135".into(),
            zaid: 53135,
            decay_constant: 2.93e-5,
            decay_branches: vec![DecayBranch {
                daughter_zaid: 54135,
                branch_ratio: 1.0,
            }],
        });
        chain.add_nuclide(NuclideEntry {
            name: "Xe-135".into(),
            zaid: 54135,
            decay_constant: 2.106e-5,
            decay_branches: vec![],
        });
        let n0 = vec![1.0e21, 0.0];
        let step = deplete_ce_li(&chain, &n0, 0.0, 3600.0, CramOrder::Cram16, |_| 0.0);
        assert!((step.predicted[0] - step.corrected[0]).abs() < 1e-6);
        assert!((step.predicted[1] - step.corrected[1]).abs() < 1e-6);
    }

    /// With a constant-flux callback, the corrector matrix is
    /// identical to the predictor matrix and the corrected
    /// composition equals the predicted one. Smoke test that the
    /// callback path is wired correctly.
    #[test]
    fn constant_flux_callback_makes_corrector_match_predictor() {
        let mut chain = DepletionChain::new();
        chain.add_nuclide(NuclideEntry {
            name: "U-238".into(),
            zaid: 92238,
            decay_constant: 0.0,
            decay_branches: vec![],
        });
        chain.add_nuclide(NuclideEntry {
            name: "U-239".into(),
            zaid: 92239,
            decay_constant: 4.92e-4, // ~23.5 min
            decay_branches: vec![],
        });
        chain.add_reaction(92238, 102, 2.7, None);
        let n0 = vec![1.0e22, 0.0];
        let dt = 86_400.0;
        let flux = 1.0e14;
        let step = deplete_ce_li(&chain, &n0, flux, dt, CramOrder::Cram16, |_| flux);
        for i in 0..n0.len() {
            assert!(
                (step.predicted[i] - step.corrected[i]).abs() / step.predicted[i].abs().max(1.0)
                    < 1e-12,
                "i={i}: predicted={} corrected={}",
                step.predicted[i],
                step.corrected[i]
            );
        }
    }
}
