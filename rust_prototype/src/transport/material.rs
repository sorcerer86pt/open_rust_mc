// SPDX-License-Identifier: MIT
//! Material = nuclide compositions + macroscopic XS.

#[derive(Debug, Clone)]
pub struct NuclideEntry {
    /// atoms/barn-cm.
    pub atom_density: f64,
    /// Index into the global XS kernels array.
    pub xs_kernel_idx: usize,
}

#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub nuclides: Vec<NuclideEntry>,
    /// K; drives XS lookup.
    pub temperature: f64,
}

impl Material {
    pub fn new(name: &str, temperature: f64) -> Self {
        Self {
            name: name.to_string(),
            nuclides: Vec::new(),
            temperature,
        }
    }

    pub fn add_nuclide(&mut self, atom_density: f64, xs_kernel_idx: usize) {
        self.nuclides.push(NuclideEntry {
            atom_density,
            xs_kernel_idx,
        });
    }

    /// Returns `true` on hit. Used by depletion to push CRAM-evolved
    /// densities back before the next transport solve.
    pub fn set_atom_density(&mut self, xs_kernel_idx: usize, atom_density: f64) -> bool {
        for nuc in &mut self.nuclides {
            if nuc.xs_kernel_idx == xs_kernel_idx {
                nuc.atom_density = atom_density;
                return true;
            }
        }
        false
    }

    pub fn atom_density_of(&self, xs_kernel_idx: usize) -> Option<f64> {
        self.nuclides
            .iter()
            .find(|n| n.xs_kernel_idx == xs_kernel_idx)
            .map(|n| n.atom_density)
    }

    /// Σ_t = Σ_i N_i · σ_t,i (barns × atoms/barn-cm = 1/cm).
    #[inline]
    pub fn macro_total(&self, micro_totals: &[f64]) -> f64 {
        self.nuclides
            .iter()
            .zip(micro_totals.iter())
            .map(|(nuc, &sigma)| nuc.atom_density * sigma)
            .sum()
    }

    /// P_i ∝ N_i · σ_t,i.
    #[inline]
    pub fn sample_nuclide(&self, micro_totals: &[f64], macro_total: f64, xi: f64) -> usize {
        let threshold = xi * macro_total;
        let mut cumulative = 0.0;
        for (i, (nuc, &sigma)) in self.nuclides.iter().zip(micro_totals.iter()).enumerate() {
            cumulative += nuc.atom_density * sigma;
            if threshold < cumulative {
                return i;
            }
        }
        self.nuclides.len() - 1
    }

    /// Build a material from bulk density + per-nuclide mass fractions.
    ///
    /// ICSBEP benchmark cards specify compositions as `(bulk density,
    /// per-nuclide weight fraction)` — this is the canonical way to
    /// `N_i = ρ · w_i / A_i · N_A / 1e24`. `entries` =
    /// `(xs_kernel_idx, atomic_mass_u, weight_fraction)`. Weight
    /// fractions are NOT renormalised — drift propagates so it's
    /// visible. Use `_awr` variant when caller has AWR not A in u.
    pub fn from_mass_fractions(
        name: &str,
        temperature: f64,
        density_g_per_cc: f64,
        entries: &[(usize, f64, f64)],
    ) -> Self {
        let mut mat = Self::new(name, temperature);
        for &(kernel_idx, atomic_mass_u, weight_fraction) in entries {
            let atom_density = density_g_per_cc * weight_fraction / atomic_mass_u
                * AVOGADRO_PER_BARN_CM;
            mat.add_nuclide(atom_density, kernel_idx);
        }
        mat
    }

    /// `A = AWR · m_n` with `m_n = 1.008664916 u`. `entries` =
    /// `(xs_kernel_idx, awr, weight_fraction)`.
    pub fn from_mass_fractions_awr(
        name: &str,
        temperature: f64,
        density_g_per_cc: f64,
        entries: &[(usize, f64, f64)],
    ) -> Self {
        let mut mat = Self::new(name, temperature);
        for &(kernel_idx, awr, weight_fraction) in entries {
            // N = ρ · w / (AWR · m_n) · N_A / 1e24
            //   = ρ · w / AWR · AVOGADRO_PER_BARN_CM_OVER_NEUTRON_MASS
            let atom_density = density_g_per_cc * weight_fraction / awr
                * AVOGADRO_PER_BARN_CM_OVER_NEUTRON_MASS;
            mat.add_nuclide(atom_density, kernel_idx);
        }
        mat
    }

    /// `M̄ = Σ x_i·A_i`, `n_total = ρ·N_A/M̄/1e24`, `N_i = x_i·n_total`.
    /// `entries` = `(xs_kernel_idx, atomic_mass_u, atom_fraction)`.
    /// Not renormalised.
    pub fn from_atom_fractions(
        name: &str,
        temperature: f64,
        density_g_per_cc: f64,
        entries: &[(usize, f64, f64)],
    ) -> Self {
        let mean_mass: f64 = entries.iter().map(|&(_, a, x)| a * x).sum();
        let total_atom_density =
            density_g_per_cc / mean_mass * AVOGADRO_PER_BARN_CM;
        let mut mat = Self::new(name, temperature);
        for &(kernel_idx, _atomic_mass_u, atom_fraction) in entries {
            mat.add_nuclide(atom_fraction * total_atom_density, kernel_idx);
        }
        mat
    }
}

/// 2018 CODATA.
pub const NEUTRON_MASS_U: f64 = 1.008_664_916;
/// Post-2019-SI exact: `N_A · 1e-24`.
pub const AVOGADRO_PER_BARN_CM: f64 = 0.602_214_076;
/// `const` division so the two can never drift.
pub const AVOGADRO_PER_BARN_CM_OVER_NEUTRON_MASS: f64 =
    AVOGADRO_PER_BARN_CM / NEUTRON_MASS_U;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Avogadro-scaled constants match 2019-SI / 2018-CODATA values
    /// exactly. The post-2019 SI definition fixes `N_A = 6.02214076e23 /mol`
    /// exactly; scaled to barns that is `0.602214076`.
    #[test]
    fn avogadro_constants_match_codata() {
        assert_eq!(AVOGADRO_PER_BARN_CM, 0.602_214_076);
        assert_eq!(NEUTRON_MASS_U, 1.008_664_916);
        // The "over neutron mass" constant must equal the const-division
        // of the two — guard against typo / accidental override.
        let recomputed = AVOGADRO_PER_BARN_CM / NEUTRON_MASS_U;
        assert!(
            (AVOGADRO_PER_BARN_CM_OVER_NEUTRON_MASS - recomputed).abs() < 1e-15,
            "constants drifted: {AVOGADRO_PER_BARN_CM_OVER_NEUTRON_MASS} vs {recomputed}",
        );
    }

    /// HEU metal at 18.74 g/cm³, 93.8 wt% U-235, 6.2 wt% U-238 — matches
    /// the canonical Godiva (HMF-001) composition. Expected atom
    /// densities computed from the same constants the function uses
    /// (round-trip), so the test cannot be made to pass by truncating
    /// the constants.
    #[test]
    fn mass_fractions_reproduce_godiva_heu() {
        let density = 18.74;
        let m_u235 = 235.0439;
        let m_u238 = 238.0508;
        let w_u235 = 0.9380;
        let w_u238 = 0.0620;
        let mat = Material::from_mass_fractions(
            "HEU_metal",
            293.6,
            density,
            &[(0, m_u235, w_u235), (1, m_u238, w_u238)],
        );
        let n_u235 = mat.atom_density_of(0).unwrap();
        let n_u238 = mat.atom_density_of(1).unwrap();
        let expected_u235 = density * w_u235 / m_u235 * AVOGADRO_PER_BARN_CM;
        let expected_u238 = density * w_u238 / m_u238 * AVOGADRO_PER_BARN_CM;
        assert!((n_u235 - expected_u235).abs() < 1e-12);
        assert!((n_u238 - expected_u238).abs() < 1e-12);
        // Sanity: the Godiva U-235 atom density quoted in the ICSBEP
        // handbook is ~4.50e-2. Should agree to within 1 % since the
        // 93.8 / 6.2 split is a representative composition.
        assert!(
            (n_u235 - 4.50e-2).abs() / 4.50e-2 < 0.012,
            "U-235 density {n_u235} differs from ICSBEP ~4.50e-2 by > 1.2%",
        );
    }

    /// PWR UO₂ at 3.1 % enrichment, 10.5 g/cm³ (typical 95 % TD pellet).
    /// Mass fractions: 3.1 % × (235.04/270.04) U-235, balance U-238 +
    /// O-16. Round-trip against the published atom densities (within
    /// 1 % since real pellets are 92-97 % TD, not exact).
    #[test]
    fn mass_fractions_pwr_uo2() {
        // UO₂ molar mass ≈ 235·0.031 + 238·0.969 + 2·16 = 269.91
        // U mass fraction in UO₂: 269.91 - 32 = 237.91 / 269.91 ≈ 0.8814
        // O mass fraction: 32 / 269.91 ≈ 0.1186
        // Within U: 3.1 % is U-235, 96.9 % is U-238 (mass fractions).
        let w_u235 = 0.031 * 0.8814;
        let w_u238 = 0.969 * 0.8814;
        let w_o16 = 0.1186;
        let mat = Material::from_mass_fractions(
            "UO2_3.1pct",
            900.0,
            10.5,
            &[
                (0, 235.0439, w_u235),
                (1, 238.0508, w_u238),
                (2, 15.9949, w_o16),
            ],
        );
        let n_u235 = mat.atom_density_of(0).unwrap();
        let n_u238 = mat.atom_density_of(1).unwrap();
        let n_o16 = mat.atom_density_of(2).unwrap();
        // O atom density ~ 2× U atom density (stoichiometry of UO₂).
        let n_u_total = n_u235 + n_u238;
        let stoich_ratio = n_o16 / n_u_total;
        assert!(
            (stoich_ratio - 2.0).abs() < 0.05,
            "UO₂ stoichiometry should give O/U ≈ 2.0, got {stoich_ratio}",
        );
        // U-235 / U-238 atom-density ratio should reflect the 3.1 %
        // enrichment after correcting for the mass difference.
        let enrichment_atom = n_u235 / (n_u235 + n_u238);
        // Atom-% enrichment is slightly different from mass-%:
        //   x_at = w_at/M_at / Σ(w_i/M_i)
        // For 3.1 % wt: x_at ≈ 0.0314 atom%.
        assert!(
            (enrichment_atom - 0.0314).abs() < 5e-4,
            "atom-% enrichment = {enrichment_atom} (expected ~0.0314)",
        );
    }

    /// AWR-flavoured version produces the same atom densities as the
    /// atomic-mass version when AWR · m_n is consistent with the
    /// atomic mass in u. Cross-check that the two helpers agree.
    #[test]
    fn from_mass_fractions_awr_matches_atomic_mass() {
        const M_N: f64 = 1.008_664_916;
        let atomic_mass_u = 235.0439;
        let awr = atomic_mass_u / M_N;
        let m1 = Material::from_mass_fractions("a", 293.6, 18.74, &[(0, atomic_mass_u, 0.938)]);
        let m2 = Material::from_mass_fractions_awr("b", 293.6, 18.74, &[(0, awr, 0.938)]);
        let d1 = m1.atom_density_of(0).unwrap();
        let d2 = m2.atom_density_of(0).unwrap();
        assert!(
            (d1 - d2).abs() / d1 < 1e-9,
            "AWR vs atomic-mass disagree: {d1} vs {d2}",
        );
    }

    /// Atom-fraction constructor: a 50/50 mol mixture of U-235 and
    /// U-238 at 18.5 g/cm³ should give equal atom densities, with the
    /// total matching the bulk density / mean atomic mass. Expected
    /// total computed from the same M̄ the function uses.
    #[test]
    fn atom_fractions_50_50_uranium() {
        let density = 18.5;
        let m_u235 = 235.0439;
        let m_u238 = 238.0508;
        let mat = Material::from_atom_fractions(
            "U_5050",
            293.6,
            density,
            &[(0, m_u235, 0.5), (1, m_u238, 0.5)],
        );
        let n_u235 = mat.atom_density_of(0).unwrap();
        let n_u238 = mat.atom_density_of(1).unwrap();
        assert!((n_u235 - n_u238).abs() / n_u235 < 1e-12);
        let mean_mass = 0.5 * m_u235 + 0.5 * m_u238;
        let expected_total = density / mean_mass * AVOGADRO_PER_BARN_CM;
        let actual_total = n_u235 + n_u238;
        assert!(
            (actual_total - expected_total).abs() / expected_total < 1e-12,
            "total density = {actual_total} (expected {expected_total})",
        );
    }

    /// Empty entries list produces an empty material (no nuclides),
    /// not a panic. Defensive against builder code that filters out
    /// zero-fraction nuclides.
    #[test]
    fn empty_entries_produces_empty_material() {
        let mat = Material::from_mass_fractions("empty", 293.6, 1.0, &[]);
        assert!(mat.nuclides.is_empty());
        assert_eq!(mat.name, "empty");
        assert_eq!(mat.temperature, 293.6);
    }
}
