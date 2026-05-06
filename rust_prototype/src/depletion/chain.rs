//! Depletion chain — nuclide list, decay constants, decay branches,
//! and per-nuclide one-group reaction cross-sections.
//!
//! The chain is the static structure that, combined with a flux
//! magnitude and per-nuclide one-group XS, gives the transmutation
//! matrix `A` (see `matrix.rs`). Reaction yields default to the
//! single-daughter convention (e.g. `(n,γ)` from `Z, A` → `Z, A+1`)
//! but explicit yield maps are honoured when set on `ReactionXs`.

use std::collections::HashMap;

/// One nuclide in the chain. `index` is the row/column in the
/// transmutation matrix (`matrix::build_transmutation_matrix`
/// fills it from `nuclides[i]`'s `index`). `decay_constant` is in
/// `s⁻¹` (`ln 2 / T_half`); pass `0.0` for stable nuclides.
#[derive(Debug, Clone)]
pub struct NuclideEntry {
    pub name: String,
    /// Z·1000 + A; used as the canonical lookup key.
    pub zaid: u32,
    /// Decay constant in s⁻¹. Stable → 0.0.
    pub decay_constant: f64,
    /// Decay branches summing to <= 1.0. The unused fraction (e.g.
    /// branchless decay to nothing tracked) leaves the chain.
    pub decay_branches: Vec<DecayBranch>,
}

/// One branch of a decay. `daughter_zaid` is the ZAID of the
/// resulting nuclide; `branch_ratio` is the fraction of decays that
/// follow this branch. Unbranched decays (single daughter) have one
/// `DecayBranch` with `branch_ratio = 1.0`.
#[derive(Debug, Clone, Copy)]
pub struct DecayBranch {
    pub daughter_zaid: u32,
    pub branch_ratio: f64,
}

/// One reaction's one-group microscopic cross-section (barns) on
/// one parent nuclide, plus the per-daughter yield map.
///
/// Yields default to the standard ENDF convention:
///   - `(n,γ)`  (MT=102): single daughter at `(Z, A+1)`
///   - `(n,2n)` (MT=16):  single daughter at `(Z, A−1)`
///   - `(n,3n)` (MT=17):  single daughter at `(Z, A−2)`
///   - `(n,α)`  (MT=107): two daughters at `(Z−2, A−4)` and `(2, 4)`
///   - `(n,p)`  (MT=103): single daughter at `(Z−1, A)`
///   - `(n,fission)` (MT=18): caller provides the fission-product
///     yield map directly via `yields`.
///
/// `yields` keys are daughter ZAIDs; values are the integer or
/// fractional production count per parent reaction. When `yields`
/// is empty the reaction is treated as a pure removal channel
/// (parent leaves the chain via this MT but no daughter is tracked).
#[derive(Debug, Clone)]
pub struct ReactionXs {
    pub mt: u32,
    pub xs_barns: f64,
    pub yields: HashMap<u32, f64>,
}

/// A depletion chain — the catalog of nuclides plus their decay /
/// reaction data. Construct with `DepletionChain::new` then `add_*`.
#[derive(Debug, Clone, Default)]
pub struct DepletionChain {
    pub nuclides: Vec<NuclideEntry>,
    /// `zaid → index in nuclides` lookup.
    pub index_of: HashMap<u32, usize>,
    /// `(zaid, mt) → ReactionXs`. Per-nuclide one-group reaction
    /// cross-sections collected for matrix construction.
    pub reactions: HashMap<(u32, u32), ReactionXs>,
}

impl DepletionChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a nuclide. Returns its `index` slot. Subsequent
    /// `add_reaction` / `add_decay` calls for the same nuclide must
    /// use a matching ZAID.
    pub fn add_nuclide(&mut self, mut entry: NuclideEntry) -> usize {
        let index = self.nuclides.len();
        let zaid = entry.zaid;
        // Stable nuclides shouldn't carry stale branches.
        if entry.decay_constant == 0.0 {
            entry.decay_branches.clear();
        }
        self.nuclides.push(entry);
        self.index_of.insert(zaid, index);
        index
    }

    /// Register a reaction for nuclide `zaid`. `yields` is filled
    /// with the standard ENDF mapping when empty for a recognized
    /// MT; otherwise the caller's map is used.
    pub fn add_reaction(&mut self, zaid: u32, mt: u32, xs_barns: f64, yields: Option<HashMap<u32, f64>>) {
        let yields = yields.unwrap_or_else(|| default_yields_for(zaid, mt));
        self.reactions.insert(
            (zaid, mt),
            ReactionXs {
                mt,
                xs_barns,
                yields,
            },
        );
    }

    /// Number of tracked nuclides.
    pub fn len(&self) -> usize {
        self.nuclides.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nuclides.is_empty()
    }

    /// Index of `zaid`, or `None` if not in the chain.
    pub fn index_of_zaid(&self, zaid: u32) -> Option<usize> {
        self.index_of.get(&zaid).copied()
    }
}

/// Default daughter ZAID(s) for a given (parent, MT) pair, per
/// ENDF mass / charge bookkeeping. Returns an empty map for MTs we
/// don't have a hard-coded rule for (e.g. fission — caller supplies).
fn default_yields_for(parent_zaid: u32, mt: u32) -> HashMap<u32, f64> {
    let z = parent_zaid / 1000;
    let a = parent_zaid % 1000;
    let mut out = HashMap::new();
    match mt {
        102 => {
            out.insert(z * 1000 + a + 1, 1.0); // (n,γ)
        }
        16 => {
            out.insert(z * 1000 + a - 1, 1.0); // (n,2n)
        }
        17 => {
            out.insert(z * 1000 + a - 2, 1.0); // (n,3n)
        }
        103 => {
            out.insert((z - 1) * 1000 + a, 1.0); // (n,p)
        }
        107 => {
            out.insert((z - 2) * 1000 + a - 4, 1.0); // (n,α) heavy fragment
            out.insert(2 * 1000 + 4, 1.0); // alpha
        }
        18 => {
            // Fission: caller must provide explicit yields. Empty
            // map → pure removal (parent leaves, no daughters).
        }
        _ => {}
    }
    out
}

/// Convenience builder for a small fission-product yield map. Used
/// in the Xe-135 equilibrium test and demos. ZAIDs and yields
/// taken from ENDF/B-VIII for thermal U-235 fission.
pub fn u235_thermal_iodine_xenon_yields() -> HashMap<u32, f64> {
    let mut out = HashMap::new();
    // Cumulative thermal-fission yield of I-135 from U-235:
    // 0.06309 (ENDF/B-VIII.0). Xe-135 is dominantly produced via
    // I-135 β-decay so its independent yield is small (~0.00256
    // direct); we use the cumulative-into-I-135 convention here.
    out.insert(53135, 0.06309); // I-135
    // Xe-135 direct (independent) yield:
    out.insert(54135, 0.00256);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_yields_n_gamma_canonical() {
        // U-238 (n,γ) → U-239
        let y = default_yields_for(92238, 102);
        assert_eq!(y.get(&92239).copied(), Some(1.0));
        assert_eq!(y.len(), 1);
    }

    #[test]
    fn default_yields_n_2n_canonical() {
        // U-238 (n,2n) → U-237
        let y = default_yields_for(92238, 16);
        assert_eq!(y.get(&92237).copied(), Some(1.0));
    }

    #[test]
    fn add_nuclide_returns_index_and_indexes_zaid() {
        let mut chain = DepletionChain::new();
        let idx_a = chain.add_nuclide(NuclideEntry {
            name: "I-135".into(),
            zaid: 53135,
            decay_constant: 2.93e-5, // ~6.57 h
            decay_branches: vec![DecayBranch {
                daughter_zaid: 54135,
                branch_ratio: 1.0,
            }],
        });
        let idx_b = chain.add_nuclide(NuclideEntry {
            name: "Xe-135".into(),
            zaid: 54135,
            decay_constant: 2.106e-5, // ~9.14 h
            decay_branches: vec![DecayBranch {
                daughter_zaid: 55135,
                branch_ratio: 1.0,
            }],
        });
        assert_eq!(idx_a, 0);
        assert_eq!(idx_b, 1);
        assert_eq!(chain.index_of_zaid(53135), Some(0));
        assert_eq!(chain.index_of_zaid(54135), Some(1));
        assert_eq!(chain.len(), 2);
    }
}
