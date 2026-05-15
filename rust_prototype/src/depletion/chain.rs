//! Depletion chain (static; pairs with flux + per-nuclide one-group
//! XS to build `A` in `matrix.rs`).

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NuclideEntry {
    pub name: String,
    /// `Z·1000 + A`; canonical lookup key.
    pub zaid: u32,
    /// `ln 2 / T_half`; 0.0 = stable.
    pub decay_constant: f64,
    /// `Σ branch_ratio ≤ 1.0`; remainder leaves the chain untracked.
    pub decay_branches: Vec<DecayBranch>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecayBranch {
    pub daughter_zaid: u32,
    pub branch_ratio: f64,
}

/// Default ENDF yields when `yields` is empty:
/// `(n,γ)→(Z,A+1)`, `(n,2n)→(Z,A−1)`, `(n,3n)→(Z,A−2)`,
/// `(n,α)→(Z−2,A−4)+(2,4)`, `(n,p)→(Z−1,A)`. MT=18 caller-supplied.
/// Empty `yields` AND non-default MT = pure removal channel.
#[derive(Debug, Clone)]
pub struct ReactionXs {
    pub mt: u32,
    pub xs_barns: f64,
    pub yields: HashMap<u32, f64>,
}

#[derive(Debug, Clone, Default)]
pub struct DepletionChain {
    pub nuclides: Vec<NuclideEntry>,
    pub index_of: HashMap<u32, usize>,
    pub reactions: HashMap<(u32, u32), ReactionXs>,
}

impl DepletionChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_nuclide(&mut self, mut entry: NuclideEntry) -> usize {
        let index = self.nuclides.len();
        let zaid = entry.zaid;
        if entry.decay_constant == 0.0 {
            entry.decay_branches.clear();
        }
        self.nuclides.push(entry);
        self.index_of.insert(zaid, index);
        index
    }

    /// `yields` defaults to ENDF mapping for recognized MT when empty.
    pub fn add_reaction(
        &mut self,
        zaid: u32,
        mt: u32,
        xs_barns: f64,
        yields: Option<HashMap<u32, f64>>,
    ) {
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
