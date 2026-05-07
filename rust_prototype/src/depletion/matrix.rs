//! Build the transmutation matrix `A` from a `DepletionChain` plus a
//! flux magnitude (n / cm² / s) plus per-nuclide one-group reaction
//! cross-sections (already on the chain).
//!
//! For nuclide `i`:
//!   A[i, i]  = − λ_i  − Σ_r σ_{i,r} · φ
//!   A[j, i]  += λ_i · b_{i→j}   (decay branches)
//!              + σ_{i,r} · φ · y_{r, i→j}   (reaction yields)
//!
//! Multiply by `Δt` before passing to `cram::cram16` (CRAM expects
//! `exp(M)` for `M = A·Δt`).

use crate::depletion::chain::DepletionChain;

/// Inputs to the transmutation matrix: the chain plus the
/// one-group flux magnitude in `n / (cm² · s)`. Reaction
/// cross-sections live on `chain.reactions`.
pub struct TransmutationInputs<'a> {
    pub chain: &'a DepletionChain,
    /// Flux in n/(cm² s). For multi-group, the caller must collapse
    /// to one group ahead of time.
    pub flux: f64,
}

/// Build `A · Δt` for use with `cram::cram16`. Row-major `n × n`
/// where `n = chain.len()`. Pure decay terms scale with `dt`;
/// reaction-rate terms scale with `flux · dt`.
pub fn build_transmutation_matrix(inputs: &TransmutationInputs<'_>, dt_seconds: f64) -> Vec<f64> {
    let n = inputs.chain.len();
    let mut a = vec![0.0_f64; n * n];

    // Decay terms.
    for (i, entry) in inputs.chain.nuclides.iter().enumerate() {
        if entry.decay_constant == 0.0 {
            continue;
        }
        let lambda_dt = entry.decay_constant * dt_seconds;
        a[i * n + i] -= lambda_dt;
        for branch in &entry.decay_branches {
            if let Some(j) = inputs.chain.index_of_zaid(branch.daughter_zaid) {
                a[j * n + i] += lambda_dt * branch.branch_ratio;
            }
            // Daughter not in chain: silently dropped — the parent
            // still leaves at full rate, but the daughter is
            // untracked. Standard simplification for short-chain
            // benchmarks.
        }
    }

    // Reaction terms (barns × flux × dt). 1 barn = 1e-24 cm².
    const BARN_CM2: f64 = 1.0e-24;
    let phi_dt = inputs.flux * dt_seconds;
    for ((parent_zaid, _mt), rxn) in &inputs.chain.reactions {
        let Some(parent_idx) = inputs.chain.index_of_zaid(*parent_zaid) else {
            continue;
        };
        let rate = rxn.xs_barns * BARN_CM2 * phi_dt;
        a[parent_idx * n + parent_idx] -= rate;
        for (&daughter_zaid, &yield_per_rxn) in &rxn.yields {
            if let Some(j) = inputs.chain.index_of_zaid(daughter_zaid) {
                a[j * n + parent_idx] += rate * yield_per_rxn;
            }
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depletion::chain::{DecayBranch, DepletionChain, NuclideEntry};

    #[test]
    fn pure_decay_diagonal_is_minus_lambda_dt() {
        let mut chain = DepletionChain::new();
        chain.add_nuclide(NuclideEntry {
            name: "X".into(),
            zaid: 53135,
            decay_constant: 1.0e-3,
            decay_branches: vec![],
        });
        let a = build_transmutation_matrix(
            &TransmutationInputs {
                chain: &chain,
                flux: 0.0,
            },
            60.0,
        );
        assert!((a[0] - (-1.0e-3 * 60.0)).abs() < 1e-15);
    }

    #[test]
    fn decay_branch_populates_off_diagonal() {
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
        let dt = 3600.0; // 1 hour
        let a = build_transmutation_matrix(
            &TransmutationInputs {
                chain: &chain,
                flux: 0.0,
            },
            dt,
        );
        // 2x2 row-major: [[A_00, A_01], [A_10, A_11]]
        assert!((a[0] - (-2.93e-5 * dt)).abs() < 1e-15);
        assert!((a[2] - (2.93e-5 * dt)).abs() < 1e-15);
        assert!((a[3] - (-2.106e-5 * dt)).abs() < 1e-15);
        assert_eq!(a[1], 0.0);
    }

    #[test]
    fn n_gamma_rate_uses_microscopic_xs_times_flux() {
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
            decay_constant: 0.0,
            decay_branches: vec![],
        });
        // 2.7 b U-238 (n,γ) one-group XS, flux 1e14, dt = 1 day.
        chain.add_reaction(92238, 102, 2.7, None);
        let dt = 86_400.0_f64;
        let flux = 1.0e14_f64;
        let a = build_transmutation_matrix(
            &TransmutationInputs {
                chain: &chain,
                flux,
            },
            dt,
        );
        let expected_rate_dt = 2.7 * 1.0e-24 * flux * dt;
        assert!((a[0] - (-expected_rate_dt)).abs() < 1e-15);
        // U-239 row: A[1, 0] = +rate_dt.
        assert!((a[1 * 2 + 0] - expected_rate_dt).abs() < 1e-15);
    }
}
