//! Photon material composition and macroscopic cross-section
//! aggregation across a multi-element target.
//!
//! A `PhotonMaterial` is a list of `(atom_density, PhotonElement)`
//! entries where atom density is in atoms/(barn·cm) (the standard
//! Monte Carlo unit that yields macroscopic cross sections in
//! inverse cm when multiplied by microscopic cross sections in barns).
//!
//! # Reaction channel sampling at a collision
//! 1. Compute macroscopic channel XS at `E`:
//!    `Σ_ch(E) = Σ_i N_i · σ_ch,i(E)` for each of the five channels.
//! 2. Total `Σ_tot = Σ_coh + Σ_inc + Σ_pe + Σ_pp_nuc + Σ_pp_el`.
//! 3. Free-flight distance `d = -ln ξ / Σ_tot`.
//! 4. At collision, sample channel by cumulative fractions, then
//!    pick the specific element by weighted `N_i · σ_ch,i / Σ_ch`.

use crate::photon::data::PhotonElement;
use crate::photon::photoelectric::interpolate_log_log;

/// Photon reaction channel identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Coherent,
    Incoherent,
    Photoelectric,
    PairProductionNuclear,
    PairProductionElectron,
}

/// A photon target — a homogeneous mixture of elements at specified
/// atom densities. Elements are stored by value; for large materials
/// consider `Arc<PhotonElement>` instead (not needed yet).
pub struct PhotonMaterial {
    pub entries: Vec<(f64, PhotonElement)>,
    /// Mass density (g/cm³) used to convert the Katz-Penfold CSDA
    /// electron range from g/cm² to cm. Set via
    /// [`PhotonMaterial::with_density`]. When zero (default) the
    /// transport driver disables electron-range displacement and
    /// falls back to kerma (deposit-at-collision).
    pub density_g_per_cm3: f64,
}

impl PhotonMaterial {
    /// Construct a material from `(atoms_per_barn_cm, element)` pairs.
    /// Density defaults to zero (kerma mode); use
    /// [`PhotonMaterial::with_density`] to enable electron-range
    /// displacement.
    pub fn new(entries: Vec<(f64, PhotonElement)>) -> Self {
        Self {
            entries,
            density_g_per_cm3: 0.0,
        }
    }

    /// Construct a single-element material at a given atom density.
    pub fn mono(atom_density: f64, element: PhotonElement) -> Self {
        Self {
            entries: vec![(atom_density, element)],
            density_g_per_cm3: 0.0,
        }
    }

    /// Set the mass density (g/cm³) used for the Katz-Penfold CSDA
    /// electron range. Chain after [`PhotonMaterial::new`] or
    /// [`PhotonMaterial::mono`].
    pub fn with_density(mut self, density_g_per_cm3: f64) -> Self {
        self.density_g_per_cm3 = density_g_per_cm3;
        self
    }

    /// Katz-Penfold CSDA electron range (cm) at kinetic energy
    /// `e_kin_ev`. Returns 0 when density is unset — the caller
    /// treats that as "kerma mode, no displacement".
    ///
    /// Reference: Katz & Penfold, Rev. Mod. Phys. 24, 28 (1952).
    /// `R(E) [g/cm²] ≈ 0.412 · E^(1.265 − 0.0954 ln E)` below 2.5 MeV,
    /// `R(E) [g/cm²] ≈ 0.530 · E − 0.106` above. Valid 10 keV-20 MeV.
    /// Not material-specific (mass-range invariance); divide by ρ to
    /// get cm.
    pub fn electron_range_cm(&self, e_kin_ev: f64) -> f64 {
        if self.density_g_per_cm3 <= 0.0 || e_kin_ev <= 1.0e4 {
            return 0.0;
        }
        let e_mev = e_kin_ev * 1.0e-6;
        let r_g_per_cm2 = if e_mev < 2.5 {
            let exp = 1.265 - 0.0954 * e_mev.ln();
            0.412 * e_mev.powf(exp)
        } else {
            0.530 * e_mev - 0.106
        };
        r_g_per_cm2 / self.density_g_per_cm3
    }

    /// Macroscopic channel cross section at energy `E` in cm⁻¹.
    pub fn macro_xs(&self, channel: Channel, energy: f64) -> f64 {
        let mut sigma = 0.0;
        for (n, elem) in &self.entries {
            let sigma_micro = match channel {
                Channel::Coherent => interpolate_log_log(&elem.energy, &elem.coherent_xs, energy),
                Channel::Incoherent => {
                    interpolate_log_log(&elem.energy, &elem.incoherent_xs, energy)
                }
                Channel::Photoelectric => {
                    interpolate_log_log(&elem.energy, &elem.photoelectric_xs, energy)
                }
                Channel::PairProductionNuclear => {
                    interpolate_log_log(&elem.energy, &elem.pair_production_nuclear_xs, energy)
                }
                Channel::PairProductionElectron => {
                    interpolate_log_log(&elem.energy, &elem.pair_production_electron_xs, energy)
                }
            };
            sigma += n * sigma_micro;
        }
        sigma
    }

    /// Macroscopic total cross section in cm⁻¹.
    pub fn macro_total(&self, energy: f64) -> f64 {
        self.macro_xs(Channel::Coherent, energy)
            + self.macro_xs(Channel::Incoherent, energy)
            + self.macro_xs(Channel::Photoelectric, energy)
            + self.macro_xs(Channel::PairProductionNuclear, energy)
            + self.macro_xs(Channel::PairProductionElectron, energy)
    }

    /// Mean free path in cm at energy `E`.
    pub fn mean_free_path(&self, energy: f64) -> f64 {
        1.0 / self.macro_total(energy)
    }

    /// Sample a reaction channel at collision energy `E`. Uses cumulative
    /// fractions of the five macroscopic channel XS.
    pub fn sample_channel(&self, energy: f64, xi: f64) -> Channel {
        let coh = self.macro_xs(Channel::Coherent, energy);
        let inc = self.macro_xs(Channel::Incoherent, energy);
        let pe = self.macro_xs(Channel::Photoelectric, energy);
        let pp_n = self.macro_xs(Channel::PairProductionNuclear, energy);
        let pp_e = self.macro_xs(Channel::PairProductionElectron, energy);
        let total = coh + inc + pe + pp_n + pp_e;
        let target = xi * total;

        let mut cum = coh;
        if target < cum {
            return Channel::Coherent;
        }
        cum += inc;
        if target < cum {
            return Channel::Incoherent;
        }
        cum += pe;
        if target < cum {
            return Channel::Photoelectric;
        }
        cum += pp_n;
        if target < cum {
            return Channel::PairProductionNuclear;
        }
        Channel::PairProductionElectron
    }

    /// Given a selected channel, sample which element participates in
    /// the collision with probability `N_i σ_ch,i / Σ_ch`.
    pub fn sample_element(&self, channel: Channel, energy: f64, xi: f64) -> usize {
        let per_element: Vec<f64> = self
            .entries
            .iter()
            .map(|(n, elem)| {
                let sigma = match channel {
                    Channel::Coherent => {
                        interpolate_log_log(&elem.energy, &elem.coherent_xs, energy)
                    }
                    Channel::Incoherent => {
                        interpolate_log_log(&elem.energy, &elem.incoherent_xs, energy)
                    }
                    Channel::Photoelectric => {
                        interpolate_log_log(&elem.energy, &elem.photoelectric_xs, energy)
                    }
                    Channel::PairProductionNuclear => {
                        interpolate_log_log(&elem.energy, &elem.pair_production_nuclear_xs, energy)
                    }
                    Channel::PairProductionElectron => {
                        interpolate_log_log(&elem.energy, &elem.pair_production_electron_xs, energy)
                    }
                };
                n * sigma
            })
            .collect();
        let total: f64 = per_element.iter().sum();
        if total <= 0.0 {
            return 0;
        }
        let target = xi * total;
        let mut cum = 0.0;
        for (i, s) in per_element.iter().enumerate() {
            cum += s;
            if target < cum {
                return i;
            }
        }
        self.entries.len() - 1
    }
}

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

    fn load(name: &str) -> Option<PhotonElement> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let p = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon")
            .join(name);
        if p.exists() {
            Some(PhotonElement::from_hdf5(&p).unwrap())
        } else {
            None
        }
    }

    /// Water at standard density: 3.3428e22 molecules/cm³ = 3.3428e-2
    /// molecules/(barn·cm). Atom density: 2× that for H, 1× for O.
    fn make_water() -> Option<PhotonMaterial> {
        let h = load("H.h5")?;
        let o = load("O.h5")?;
        let molecule_density = 3.3428e-2; // molecules / (barn·cm)
        Some(PhotonMaterial::new(vec![
            (2.0 * molecule_density, h),
            (1.0 * molecule_density, o),
        ]))
    }

    #[test]
    fn water_macro_total_at_1mev() {
        let Some(water) = make_water() else {
            eprintln!("skipping: H or O data not present");
            return;
        };
        // NIST XCOM: water mass attenuation at 1 MeV is ≈ 0.0707 cm²/g.
        // Macro XS = ρ × μ/ρ = 1.0 g/cm³ × 0.0707 cm²/g = 0.0707 cm⁻¹.
        let sigma = water.macro_total(1.0e6);
        assert!(
            (sigma - 0.0707).abs() / 0.0707 < 0.03,
            "water macro total at 1 MeV = {sigma} cm⁻¹, NIST ≈ 0.0707, rel err {}",
            ((sigma - 0.0707) / 0.0707).abs()
        );
    }

    #[test]
    fn water_macro_total_at_100kev() {
        let Some(water) = make_water() else {
            eprintln!("skipping: H or O data not present");
            return;
        };
        // NIST XCOM: water at 100 keV is ≈ 0.1707 cm²/g = 0.1707 cm⁻¹
        // (density 1.0 g/cm³).
        let sigma = water.macro_total(1.0e5);
        assert!(
            (sigma - 0.1707).abs() / 0.1707 < 0.03,
            "water macro total at 100 keV = {sigma} cm⁻¹, NIST ≈ 0.1707"
        );
    }

    #[test]
    fn channel_fractions_sum_to_total() {
        let Some(water) = make_water() else {
            eprintln!("skipping");
            return;
        };
        for e_kev in [10.0, 100.0, 1_000.0, 10_000.0] {
            let e = e_kev * 1_000.0;
            let coh = water.macro_xs(Channel::Coherent, e);
            let inc = water.macro_xs(Channel::Incoherent, e);
            let pe = water.macro_xs(Channel::Photoelectric, e);
            let pp_n = water.macro_xs(Channel::PairProductionNuclear, e);
            let pp_e = water.macro_xs(Channel::PairProductionElectron, e);
            let total = water.macro_total(e);
            let sum = coh + inc + pe + pp_n + pp_e;
            assert!((sum - total).abs() < 1e-12);
        }
    }

    #[test]
    fn sampled_channels_match_fractions_at_1mev() {
        let Some(water) = make_water() else {
            eprintln!("skipping");
            return;
        };
        // At 1 MeV water is ~100 % Compton.
        let inc = water.macro_xs(Channel::Incoherent, 1.0e6);
        let total = water.macro_total(1.0e6);
        let frac_inc = inc / total;
        assert!(
            frac_inc > 0.97,
            "Compton fraction at 1 MeV should dominate, got {frac_inc}"
        );

        let n = 50_000;
        let mut inc_count = 0;
        for i in 0..n {
            // Deterministic cover of [0, 1] — the channel fractions
            // should fall out directly in the sampling counts.
            let xi = (i as f64 + 0.5) / n as f64;
            if water.sample_channel(1.0e6, xi) == Channel::Incoherent {
                inc_count += 1;
            }
        }
        let sampled_frac = inc_count as f64 / n as f64;
        assert!(
            (sampled_frac - frac_inc).abs() < 0.01,
            "sampled {sampled_frac} vs fraction {frac_inc}"
        );
    }

    #[test]
    fn mean_free_path_reasonable_for_lead_at_1mev() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5");
            return;
        };
        // Lead density 11.34 g/cm³, A = 207.2 → atom density ≈ 3.296e-2 /barn·cm
        let pb_mat = PhotonMaterial::mono(3.296e-2, pb);
        let mfp = pb_mat.mean_free_path(1.0e6);
        // At 1 MeV Pb has μ/ρ ≈ 0.0684 cm²/g → μ ≈ 0.776 cm⁻¹
        // → mfp ≈ 1.29 cm.
        assert!(
            mfp > 1.15 && mfp < 1.45,
            "Pb mfp at 1 MeV = {mfp} cm, expected ~1.29"
        );
    }
}
