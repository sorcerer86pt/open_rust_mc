//! `BurnupMapping` — table that pairs every chain ZAID with the
//! transport-side `(material_idx, xs_kernel_idx)` it lives in.
//!
//! At end of each CRAM step the mapping walks every entry and pushes
//! the new atom density into the live `Material`. ZAIDs that exist
//! only in the chain (not in any transport material) are silently
//! skipped — common for tracking decay daughters and short-lived
//! intermediates that have negligible XS.
//!
//! Conversely, ZAIDs in transport but not in the chain are also
//! skipped — common for cladding nuclides (Zr-90..94) that don't
//! deplete on the timescale of interest.
//!
//! # Building a mapping
//!
//! The mapping is a flat list of `(chain_idx, material_idx,
//! xs_kernel_idx)` triples. Construct via:
//!
//! - `BurnupMapping::from_zaid_table(chain, &materials, &table)` —
//!   pass a `&[(zaid, material_idx, xs_kernel_idx)]` table that
//!   you've curated manually for the problem.
//! - `BurnupMapping::auto(chain, &materials, &nuclide_specs)` — for
//!   binaries like `deplete_pwr` where `NUCLIDE_SPECS` already
//!   carries the (filename, ZAID-implied-from-filename) pair, this
//!   helper builds the mapping by ZAID match.

use crate::depletion::chain::DepletionChain;
use crate::transport::material::Material;

/// One mapping entry: (chain index, material index, xs_kernel_idx).
#[derive(Debug, Clone, Copy)]
pub struct BurnupMappingEntry {
    pub chain_idx: usize,
    pub material_idx: usize,
    pub xs_kernel_idx: usize,
}

/// Walks every (chain, material) pair after a CRAM step.
#[derive(Debug, Clone, Default)]
pub struct BurnupMapping {
    pub entries: Vec<BurnupMappingEntry>,
}

impl BurnupMapping {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Build a mapping by matching chain ZAIDs against a curated
    /// table. Each row is `(zaid, material_idx, xs_kernel_idx)`. ZAIDs
    /// not in the chain are skipped silently. Useful when the same
    /// ZAID lives in multiple materials (e.g. O-16 in fuel + water)
    /// — list it twice with different `(material_idx, xs_kernel_idx)`.
    pub fn from_zaid_table(
        chain: &DepletionChain,
        materials: &[Material],
        table: &[(u32, usize, usize)],
    ) -> Self {
        let mut mapping = Self::new();
        for &(zaid, material_idx, xs_kernel_idx) in table {
            let Some(chain_idx) = chain.index_of_zaid(zaid) else {
                continue;
            };
            // Require the (material, xs_kernel_idx) slot to exist on
            // the live material so post-CRAM pushes always succeed.
            // Entries pointing at non-existent materials are dropped
            // at construction time rather than silently failing later.
            if material_idx >= materials.len() {
                continue;
            }
            if materials[material_idx]
                .atom_density_of(xs_kernel_idx)
                .is_none()
            {
                continue;
            }
            mapping.entries.push(BurnupMappingEntry {
                chain_idx,
                material_idx,
                xs_kernel_idx,
            });
        }
        mapping
    }

    /// Push CRAM-evolved chain composition back into the live
    /// transport materials. `composition` is indexed by chain
    /// position (`chain.nuclides[i]`'s ZAID at index `i`).
    pub fn push(&self, composition: &[f64], materials: &mut [Material]) {
        for entry in &self.entries {
            if entry.material_idx < materials.len() {
                let density = composition[entry.chain_idx];
                materials[entry.material_idx].set_atom_density(entry.xs_kernel_idx, density);
            }
        }
    }

    /// Pull the current transport-material composition into a fresh
    /// chain-ordered vector. Used at the start of a burnup run, or
    /// to refresh the chain composition from a hand-edited
    /// `Material` (e.g. after a fuel-shuffle event). ZAIDs in the
    /// chain but not in any mapping entry default to `0.0`.
    pub fn pull(&self, chain: &DepletionChain, materials: &[Material]) -> Vec<f64> {
        let mut out = vec![0.0_f64; chain.len()];
        for entry in &self.entries {
            if entry.material_idx < materials.len()
                && let Some(d) = materials[entry.material_idx].atom_density_of(entry.xs_kernel_idx)
            {
                out[entry.chain_idx] = d;
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depletion::chain::{DepletionChain, NuclideEntry};

    fn build_chain() -> DepletionChain {
        let mut chain = DepletionChain::new();
        chain.add_nuclide(NuclideEntry {
            name: "U-235".into(),
            zaid: 92235,
            decay_constant: 0.0,
            decay_branches: vec![],
        });
        chain.add_nuclide(NuclideEntry {
            name: "Xe-135".into(),
            zaid: 54135,
            decay_constant: 2.106e-5,
            decay_branches: vec![],
        });
        chain.add_nuclide(NuclideEntry {
            name: "Cs-135".into(),
            zaid: 55135,
            decay_constant: 0.0,
            decay_branches: vec![],
        });
        chain
    }

    fn build_materials() -> Vec<Material> {
        let mut fuel = Material::new("UO2", 900.0);
        fuel.add_nuclide(7.19e-4, 0); // U-235  (xs_kernel_idx=0)
        fuel.add_nuclide(0.0, 9); // Xe-135 (xs_kernel_idx=9)
        vec![fuel]
    }

    #[test]
    fn push_updates_only_mapped_entries() {
        let chain = build_chain();
        let mut materials = build_materials();
        // Cs-135 is in the chain but not in any material — must be
        // dropped from the mapping silently.
        let mapping = BurnupMapping::from_zaid_table(
            &chain,
            &materials,
            &[(92235, 0, 0), (54135, 0, 9), (55135, 99, 99)],
        );
        // Only U-235 + Xe-135 entries created (Cs-135 skipped).
        assert_eq!(mapping.entries.len(), 2);

        let composition = vec![6.5e-4, 9.7e-9, 5.0e-8]; // U-235, Xe-135, Cs-135
        mapping.push(&composition, &mut materials);
        assert!((materials[0].atom_density_of(0).unwrap() - 6.5e-4).abs() < 1e-15);
        assert!((materials[0].atom_density_of(9).unwrap() - 9.7e-9).abs() < 1e-15);
    }

    #[test]
    fn pull_reads_current_material_state() {
        let chain = build_chain();
        let materials = build_materials();
        let mapping =
            BurnupMapping::from_zaid_table(&chain, &materials, &[(92235, 0, 0), (54135, 0, 9)]);
        let composition = mapping.pull(&chain, &materials);
        assert_eq!(composition.len(), 3);
        assert!((composition[0] - 7.19e-4).abs() < 1e-15);
        assert_eq!(composition[1], 0.0);
        // Cs-135 not in mapping → stays at 0.
        assert_eq!(composition[2], 0.0);
    }

    #[test]
    fn round_trip_pull_then_push_is_identity() {
        let chain = build_chain();
        let mut materials = build_materials();
        let mapping =
            BurnupMapping::from_zaid_table(&chain, &materials, &[(92235, 0, 0), (54135, 0, 9)]);
        let comp = mapping.pull(&chain, &materials);
        mapping.push(&comp, &mut materials);
        let comp_again = mapping.pull(&chain, &materials);
        for (a, b) in comp.iter().zip(comp_again.iter()) {
            assert_eq!(a, b);
        }
    }
}
