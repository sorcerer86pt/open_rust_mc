//! Depletion-chain JSON loader.
//!
//! The in-memory `DepletionChain` is rich (HashMaps, Option fields,
//! ZAID-based references) but the serialized form should be flat,
//! diff-friendly, and copy-paste-able from a paper or an OpenMC
//! chain export. This module defines that flat schema and the
//! conversion helpers between it and `DepletionChain`.
//!
//! # Schema
//!
//! ```json
//! {
//!   "name": "pwr_basic",
//!   "description": "Partial PWR depletion chain for k_eff feedback",
//!   "nuclides": [
//!     {
//!       "name": "U-235",
//!       "zaid": 92235,
//!       "decay_constant": 0.0,
//!       "branches": []
//!     },
//!     {
//!       "name": "I-135",
//!       "zaid": 53135,
//!       "decay_constant": 2.9264e-5,
//!       "branches": [
//!         {"daughter": 54135, "ratio": 1.0}
//!       ]
//!     }
//!   ],
//!   "reactions": [
//!     {
//!       "parent": 92235,
//!       "mt": 18,
//!       "xs_barns": 583.5,
//!       "yields": {"53135": 0.06309, "54135": 0.00256}
//!     },
//!     {
//!       "parent": 54135,
//!       "mt": 102,
//!       "xs_barns": 2.65e6,
//!       "yields": {}
//!     }
//!   ]
//! }
//! ```
//!
//! `yields` is `daughter_zaid → atoms produced per parent reaction`.
//! Empty `yields` means the reaction is a pure removal channel
//! (parent leaves the chain, no daughter tracked). When `yields` is
//! omitted entirely on an `(n,γ)` / `(n,2n)` / `(n,3n)` / `(n,p)` /
//! `(n,α)` reaction, the standard ENDF mass/charge bookkeeping
//! defaults are used (see `chain::default_yields_for`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::depletion::chain::{DecayBranch, DepletionChain, NuclideEntry};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChainSpec {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub nuclides: Vec<ChainSpecNuclide>,
    #[serde(default)]
    pub reactions: Vec<ChainSpecReaction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChainSpecNuclide {
    pub name: String,
    pub zaid: u32,
    #[serde(default)]
    pub decay_constant: f64,
    #[serde(default)]
    pub branches: Vec<ChainSpecBranch>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChainSpecBranch {
    pub daughter: u32,
    pub ratio: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChainSpecReaction {
    pub parent: u32,
    pub mt: u32,
    pub xs_barns: f64,
    /// Daughter-ZAID → yield per parent reaction. Keys are stored
    /// as strings in JSON (JSON object-key constraint); values are
    /// the per-reaction multiplicities (1.0 for single daughters,
    /// fractional for fission yields, 2.0 for `(n,α)` heavy fragment
    /// + alpha when the chain tracks both, etc.).
    ///
    /// Three valid forms:
    /// - **field omitted** → use the default ENDF daughter inferred
    ///   from `(parent, mt)` (e.g. `(n,γ)` → `(Z, A+1)` at yield 1.0)
    /// - **`"yields": {}`** → pure removal channel (parent leaves
    ///   the chain via this MT, no daughter tracked — useful for
    ///   `(n,γ)` on a poison whose daughter is inert in this chain)
    /// - **`"yields": {"<daughter_zaid>": <multiplicity>, ...}`** →
    ///   explicit daughter list
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yields: Option<HashMap<String, f64>>,
}

impl ChainSpec {
    /// Build a `ChainSpec` from a JSON file on disk.
    pub fn from_file(path: &Path) -> Result<Self, ChainLoadError> {
        let text = fs::read_to_string(path).map_err(ChainLoadError::Io)?;
        Self::from_str(&text)
    }

    /// Build a `ChainSpec` from a JSON string. Named `from_json_str`
    /// to avoid shadowing the standard `FromStr::from_str` trait
    /// method, which has a different return type (`Result<Self,
    /// Self::Err>` with a custom `Err` associated type).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Self, ChainLoadError> {
        serde_json::from_str(text).map_err(ChainLoadError::Json)
    }

    /// Materialize into a runtime `DepletionChain`. Reactions whose
    /// `yields` map is empty fall through to `chain.add_reaction`'s
    /// default ENDF inference (see `default_yields_for`).
    pub fn build(&self) -> DepletionChain {
        let mut chain = DepletionChain::new();
        for n in &self.nuclides {
            chain.add_nuclide(NuclideEntry {
                name: n.name.clone(),
                zaid: n.zaid,
                decay_constant: n.decay_constant,
                decay_branches: n
                    .branches
                    .iter()
                    .map(|b| DecayBranch {
                        daughter_zaid: b.daughter,
                        branch_ratio: b.ratio,
                    })
                    .collect(),
            });
        }
        for r in &self.reactions {
            // Three-way semantics on the JSON `yields` field:
            //   - omitted entirely (None): use ENDF defaults
            //   - present and empty (Some({})): explicit pure removal
            //   - present with entries: explicit daughter list
            let yields = match &r.yields {
                None => None,
                Some(map) => {
                    let mut yields_int: HashMap<u32, f64> = HashMap::with_capacity(map.len());
                    for (k, v) in map {
                        if let Ok(zaid) = k.parse::<u32>() {
                            yields_int.insert(zaid, *v);
                        }
                    }
                    Some(yields_int)
                }
            };
            chain.add_reaction(r.parent, r.mt, r.xs_barns, yields);
        }
        chain
    }
}

#[derive(Debug)]
pub enum ChainLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for ChainLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "chain I/O error: {e}"),
            Self::Json(e) => write!(f, "chain JSON parse error: {e}"),
        }
    }
}

impl std::error::Error for ChainLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const PARTIAL_XE_JSON: &str = r#"{
        "name": "partial_xe",
        "description": "Xe poisoning chain — minimal for k_eff feedback demos.",
        "nuclides": [
            {"name": "U-235", "zaid": 92235, "decay_constant": 0.0, "branches": []},
            {"name": "I-135",  "zaid": 53135, "decay_constant": 2.9264e-5,
             "branches": [{"daughter": 54135, "ratio": 1.0}]},
            {"name": "Xe-135", "zaid": 54135, "decay_constant": 2.10653e-5,
             "branches": [{"daughter": 55135, "ratio": 1.0}]},
            {"name": "Cs-135", "zaid": 55135, "decay_constant": 0.0, "branches": []}
        ],
        "reactions": [
            {"parent": 92235, "mt": 18, "xs_barns": 583.5,
             "yields": {"53135": 0.06309, "54135": 0.00256}},
            {"parent": 54135, "mt": 102, "xs_barns": 2.65e6, "yields": {}}
        ]
    }"#;

    #[test]
    fn round_trip_partial_xe_chain() {
        let spec = ChainSpec::from_str(PARTIAL_XE_JSON).expect("parse");
        assert_eq!(spec.nuclides.len(), 4);
        assert_eq!(spec.reactions.len(), 2);
        let chain = spec.build();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain.index_of_zaid(92235), Some(0));
        assert_eq!(chain.index_of_zaid(54135), Some(2));
        // Fission yields preserved.
        let rxn = chain
            .reactions
            .get(&(92235, 18))
            .expect("U-235 fission entry");
        assert!((rxn.yields[&53135] - 0.06309).abs() < 1e-12);
    }

    /// Smoke-load `chains/pwr_actinides.json` from the repo and check
    /// it parses, builds a non-empty chain with the expected
    /// nuclide / reaction count, and the dominant fission-product
    /// poisons are wired up properly. Guards against future edits to
    /// the JSON file silently breaking the schema.
    #[test]
    fn pwr_actinides_chain_loads_and_has_expected_shape() {
        // Repo root is two levels above the rust_prototype crate.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("chains")
            .join("pwr_actinides.json");
        if !path.exists() {
            // Skip in environments where the chains/ directory isn't
            // checked out (e.g. crate published independently).
            return;
        }
        let spec = ChainSpec::from_file(&path).expect("parse pwr_actinides.json");
        assert_eq!(spec.name, "pwr_actinides");
        let chain = spec.build();
        // 12 actinides + 5 fission products = 17 nuclides expected.
        assert_eq!(chain.len(), 17, "actinide chain nuclide count");
        // All key actinides present.
        for &z in &[92235u32, 92238, 94239, 94240, 94241, 94242, 95241] {
            assert!(
                chain.index_of_zaid(z).is_some(),
                "actinide ZAID {z} missing from chain",
            );
        }
        // Dominant FP poisons.
        assert!(chain.index_of_zaid(54135).is_some(), "Xe-135 missing");
        assert!(chain.index_of_zaid(62149).is_some(), "Sm-149 missing");
        // U-235 fission yields the expected daughters.
        let u235_fission = chain
            .reactions
            .get(&(92235u32, 18u32))
            .expect("U-235 (n,fission) reaction");
        assert!(
            u235_fission.yields.contains_key(&53135),
            "U-235 → I-135 yield"
        );
        assert!(
            u235_fission.yields.contains_key(&54135),
            "U-235 → Xe-135 yield"
        );
        assert!(
            u235_fission.yields.contains_key(&61149),
            "U-235 → Pm-149 yield"
        );
        // Xe-135 (n,γ) is a pure removal channel (no daughter tracked).
        let xe_capture = chain
            .reactions
            .get(&(54135u32, 102u32))
            .expect("Xe-135 (n,γ) reaction");
        assert!(
            xe_capture.yields.is_empty(),
            "Xe-135 (n,γ) should be pure removal in this chain",
        );
    }

    /// Drive the loaded actinides chain through one CRAM step at PWR
    /// thermal flux. Verifies the 17-nuclide system solves cleanly
    /// (no NaN / negative densities) and produces qualitatively
    /// correct behaviour: U-235 depletes a tiny amount, U-238 →
    /// U-239 → Np-239 → Pu-239 buildup chain populates.
    #[test]
    fn pwr_actinides_chain_solves_one_cram_step_correctly() {
        use crate::depletion::cram::CramOrder;
        use crate::depletion::deplete_ce_li;

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("chains")
            .join("pwr_actinides.json");
        if !path.exists() {
            return;
        }
        let spec = ChainSpec::from_file(&path).expect("parse");
        let chain = spec.build();

        // Initial PWR fuel composition (atoms / b·cm).
        let mut n0 = vec![0.0_f64; chain.len()];
        n0[chain.index_of_zaid(92235).expect("U-235 idx")] = 7.19e-4;
        n0[chain.index_of_zaid(92238).expect("U-238 idx")] = 2.2482e-2;

        // Burn one day at 3e14 n/cm²/s. CRAM-16, constant flux.
        let dt = 86_400.0_f64;
        let flux = 3.0e14_f64;
        let step = deplete_ce_li(&chain, &n0, flux, dt, CramOrder::Cram16, |_| flux);
        let n_t = &step.corrected;

        // Sanity: no NaN, no large negatives. CRAM is unconditionally
        // stable on left-half-plane matrices, so any residue near
        // zero is just numerical noise — clip below 1e-30.
        for (i, &x) in n_t.iter().enumerate() {
            assert!(x.is_finite(), "NaN/Inf at chain idx {i}");
            assert!(x > -1e-30, "large negative density at chain idx {i}: {x}",);
        }

        // U-235 mass decreased.
        let u235_idx = chain.index_of_zaid(92235).unwrap();
        assert!(
            n_t[u235_idx] < n0[u235_idx],
            "U-235 should deplete: {} -> {}",
            n0[u235_idx],
            n_t[u235_idx],
        );
        // U-239 grew from 0.
        let u239_idx = chain.index_of_zaid(92239).unwrap();
        assert!(
            n_t[u239_idx] > 0.0,
            "U-239 should populate from U-238 (n,γ): {}",
            n_t[u239_idx],
        );
        // Np-239 grew from 0 via U-239 → Np-239 decay.
        let np239_idx = chain.index_of_zaid(93239).unwrap();
        assert!(
            n_t[np239_idx] > 0.0,
            "Np-239 should populate from U-239 decay: {}",
            n_t[np239_idx],
        );
        // Pu-239 grew from 0 via Np-239 → Pu-239 decay.
        let pu239_idx = chain.index_of_zaid(94239).unwrap();
        assert!(
            n_t[pu239_idx] > 0.0,
            "Pu-239 should populate via Np-239 decay: {}",
            n_t[pu239_idx],
        );
        // Xe-135 grew from 0 via U-235 fission yield + I-135 decay.
        let xe_idx = chain.index_of_zaid(54135).unwrap();
        assert!(
            n_t[xe_idx] > 0.0,
            "Xe-135 should populate via fission yield: {}",
            n_t[xe_idx],
        );
    }

    #[test]
    fn omitted_yields_uses_endf_default() {
        let json = r#"{
            "name": "ngamma_default",
            "description": "",
            "nuclides": [
                {"name": "U-238", "zaid": 92238, "decay_constant": 0.0, "branches": []},
                {"name": "U-239", "zaid": 92239, "decay_constant": 0.0, "branches": []}
            ],
            "reactions": [
                {"parent": 92238, "mt": 102, "xs_barns": 2.7}
            ]
        }"#;
        let chain = ChainSpec::from_str(json).expect("parse").build();
        let rxn = chain
            .reactions
            .get(&(92238, 102))
            .expect("U-238 (n,γ) entry");
        // Default daughter from ENDF: (Z, A+1) = U-239.
        assert!((rxn.yields[&92239] - 1.0).abs() < 1e-15);
    }
}
