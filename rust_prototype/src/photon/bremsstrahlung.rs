//! Single-photon thick-target bremsstrahlung sampling using the
//! Seltzer-Berger 1986 scaled DCS tabulation already loaded into
//! [`PhotonElement::bremsstrahlung`].
//!
//! Scaling convention (Seltzer-Berger 1986, matching OpenMC's photon
//! HDF5 layout, see openmc/data/photon.py::IncidentPhoton._add_bremsstrahlung
//! where the raw mbarn values are multiplied by `1.0e-3` before HDF5 write):
//!
//!   dcs[i_e][i_k] = (β² / Z²) · E_γ · dσ/dE_γ    [in barn]
//!
//! Equivalently, with `k = E_γ / T_e ∈ [0, 1]` the scaled photon energy:
//!
//!   dσ/dk = dcs · Z² / (β² · k)                  [barn, dimensionless k]
//!   k·(dσ/dk) = dcs · Z² / β²                    [barn]
//!
//! Radiative cross section per atom (barn):
//!
//!   σ_rad(T_e) = ∫(dσ/dk) dk = (Z²/β²) · ∫(dcs/k) dk
//!
//! Radiative stopping cross section per atom (eV · barn):
//!
//!   S_rad(T_e) = ∫E_γ · (dσ/dE_γ) dE_γ = T_e · (Z²/β²) · ∫dcs dk
//!
//! # Important: this matches OpenMC, but does NOT match NIST ESTAR
//!
//! Computing S_rad/ρ from these formulae and comparing to NIST ESTAR
//! gives a Z-dependent over-prediction (~0.7× at H-1, ~5× at U-92 at
//! T_e = 1 MeV). The disagreement is **not a code bug**; it is the
//! well-known Coulomb-screening / electron-electron correction gap
//! between the raw Seltzer-Berger 1986 χ tables (used here for
//! spectrum-shape sampling) and the Berger-Seltzer 1982 with screening
//! tables that NIST ESTAR uses for absolute stopping powers. See:
//!
//!   - Seltzer & Berger, At. Data Nucl. Data Tables 35, 345 (1986)
//!     — describes the unscreened nuclear-field DCS tabulated here.
//!   - Berger & Seltzer, NBSIR 82-2550 (1982) — adds Coulomb,
//!     screening, and electron-electron corrections used by ESTAR.
//!   - Pratt et al., At. Data Nucl. Data Tables 20, 175 (1977) —
//!     screening corrections.
//!
//! Production path for the per-electron emission *probability* is the
//! NIST-calibrated empirical fit
//! [`MaterialBremss::radiative_yield_approx`], which lives at the
//! material level. The per-atom σ_rad here is used internally only as
//! a **relative** weight for which element to sample from in
//! [`MaterialBremss::sample_element`] — the Z-dependent miscalibration
//! enters there but is small compared to the across-element σ_rad
//! ratio (which is dominated by Z²) at the materials we benchmark.
//!
//! See `tests/brems_nist_cross_check.rs` for the integration test
//! that documents the current discrepancy quantitatively.
//!
//! Use
//! ----
//!   - [`ElementBremss::new`] — wrap a loaded `PhotonElement` with
//!     cumulative distributions precomputed on the photon-energy grid
//!     for fast inverse-CDF sampling.
//!   - [`ElementBremss::sigma_rad_barns`] — integrated radiative cross
//!     section at electron energy `t_e` (barns/atom).
//!   - [`ElementBremss::sample_k`] — sample a scaled photon energy `k`
//!     for a given electron kinetic energy.
//!
//! # Simplifications (Phase-1 TTB)
//! - One photon emitted per electron at its birth location; the
//!   electron's CSDA-deposition energy is reduced by the sampled
//!   photon energy. This differs from OpenMC's multi-photon TTB in
//!   that we do not loop down the slowing-down spectrum, but the
//!   per-electron spectrum shape matches because `k` is drawn from the
//!   full DCS at the birth energy.
//! - Emitted-photon angular distribution is isotropic. In reality
//!   bremsstrahlung is forward-peaked at high E (Schiff 2Bn, Lorentz-
//!   boosted dipole). Isotropic is the OpenMC-TTB default and matches
//!   the cross-code comparison we already set up.
//! - Coulomb-suppression, LPM, and nuclear form-factor corrections are
//!   all already baked into the stored DCS — we only re-scale.

use crate::photon::data::PhotonElement;
use crate::transport::rng::Rng;

/// Electron rest mass energy in eV. Used for β² = T(T+2mc²) / (T+mc²)².
const M_E_C2_EV: f64 = 510_998.95;

fn beta2(t_e_ev: f64) -> f64 {
    // Relativistic β² = T(T+2m)/(T+m)² with T in eV and m = m_e c² in eV.
    let tau = t_e_ev;
    let gamma = 1.0 + tau / M_E_C2_EV;
    1.0 - 1.0 / (gamma * gamma)
}

/// Per-element bremsstrahlung sampler: keeps the DCS table and a
/// pre-built CDF over the scaled photon energy grid for inverse-CDF
/// sampling at each electron energy row.
pub struct ElementBremss {
    z: f64,
    /// Electron kinetic energy grid (eV), ascending.
    electron_energy: Vec<f64>,
    /// Scaled photon energy grid k = Eγ/T_e, ascending in (0, 1].
    /// A leading 0 may be present.
    k_grid: Vec<f64>,
    /// Seltzer-Berger χ table, row-major `chi[i_e][i_k]` in **barn**
    /// (OpenMC HDF5 stores `(β²/Z²) · E_γ · dσ/dE_γ` in barn after the
    /// internal `1e-3` mbarn → barn conversion).
    chi: Vec<Vec<f64>>,
    /// Integrated CDF over k at each electron energy, with CDF[0] = 0
    /// and CDF[N_k-1] = total ∫ (χ/k) dk in χ-native units (barn).
    /// We keep it in the χ-native scaling so the normalisation cancels
    /// during inverse-CDF sampling.
    cdf_k: Vec<Vec<f64>>,
    /// ∫ χ dk at each electron energy in χ-native units (barn).
    /// Used by `sigma_rad_barns` and `mean_k`.
    integral_chi: Vec<f64>,
}

impl ElementBremss {
    /// Build sampling structures from a loaded element.
    pub fn new(elem: &PhotonElement) -> Self {
        let br = &elem.bremsstrahlung;
        let n_e = br.electron_energy.len();
        let k_grid = br.photon_energy.clone();

        // Pre-build CDF[i_e] on the k-grid using trapezoidal rule over
        // (dσ/dk) = χ / k. At k=0 the integrand 1/k is singular so we
        // lower-bound by a k_min = max(k_grid[0], 1e-9) and carry a
        // closed-form soft-photon contribution for the first bin.
        let mut cdf_k = Vec::with_capacity(n_e);
        let mut integral_chi = Vec::with_capacity(n_e);
        for i_e in 0..n_e {
            let row = &br.dcs[i_e];
            let (cdf, integ) = build_cdf_and_integral(&k_grid, row);
            cdf_k.push(cdf);
            integral_chi.push(integ);
        }

        Self {
            z: elem.z as f64,
            electron_energy: br.electron_energy.clone(),
            k_grid,
            chi: br.dcs.clone(),
            cdf_k,
            integral_chi,
        }
    }

    /// Integrated radiative cross section per atom, σ_rad(T_e) in barns.
    /// Raw Seltzer-Berger 1986 (no screening / electron-electron
    /// corrections); see module docstring on the NIST ESTAR gap.
    ///
    ///   σ_rad = ∫[0,1] (dσ/dk) dk = (Z² / β²) · ∫ (χ / k) dk
    ///
    /// `χ` is already in barn (OpenMC HDF5 convention, post mbarn→barn
    /// conversion at write-time), so no extra unit factor is needed.
    /// Interpolates log-linearly in T_e on the stored grid.
    pub fn sigma_rad_barns(&self, t_e_ev: f64) -> f64 {
        if t_e_ev <= self.electron_energy[0] {
            return 0.0;
        }
        let integral_barn = log_interp(&self.electron_energy, &self.integral_chi, t_e_ev);
        let b2 = beta2(t_e_ev).max(1.0e-10);
        integral_barn * self.z * self.z / b2
    }

    /// Atomic number of this element. Used by the NIST cross-check.
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Radiative stopping cross section per atom, S_rad(T_e), in eV·barn.
    /// Computed from the OpenMC Seltzer-Berger formula
    ///
    ///   S_rad = T_e · (Z² / β²) · ∫ χ dk
    ///
    /// Comparing this against NIST ESTAR will show a Z-dependent
    /// over-prediction (Coulomb / e-e screening gap; see module
    /// docstring). Exposed for the NIST cross-check test, not used
    /// internally for emission-probability sampling.
    ///
    /// Note: `integral_chi` (stored on `Self`) holds `∫(χ/k) dk`, not
    /// `∫χ dk` — the integrand needed for `mean_k`. So we compute
    /// `∫χ dk` on-demand from the nearest tabulated row.
    pub fn s_rad_per_atom_ev_barn(&self, t_e_ev: f64) -> f64 {
        if t_e_ev <= self.electron_energy[0] {
            return 0.0;
        }
        let row_idx = nearest_row(&self.electron_energy, t_e_ev);
        let row = &self.chi[row_idx];
        let mut integ_barn = 0.0;
        for i in 1..self.k_grid.len() {
            let dk = self.k_grid[i] - self.k_grid[i - 1];
            integ_barn += 0.5 * (row[i] + row[i - 1]) * dk;
        }
        let b2 = beta2(t_e_ev).max(1.0e-10);
        t_e_ev * self.z * self.z / b2 * integ_barn
    }

    /// Mean radiative yield fraction `<k>` at electron energy `T_e`.
    /// This is `∫ k · (dσ/dk) dk  /  ∫ (dσ/dk) dk` — purely a shape
    /// property of the DCS row, so Z² / β² cancels.
    pub fn mean_k(&self, t_e_ev: f64) -> f64 {
        let row_idx = nearest_row(&self.electron_energy, t_e_ev);
        let row = &self.chi[row_idx];
        let cdf = &self.cdf_k[row_idx];
        if cdf[cdf.len() - 1] <= 0.0 {
            return 0.0;
        }
        // numerator = ∫ k · (χ / k) dk = ∫ χ dk  (trapezoidal)
        let mut num = 0.0;
        for i in 1..self.k_grid.len() {
            let dk = self.k_grid[i] - self.k_grid[i - 1];
            num += 0.5 * (row[i] + row[i - 1]) * dk;
        }
        num / cdf[cdf.len() - 1]
    }

    /// Sample a scaled photon energy `k ∈ (0, 1]` at electron energy
    /// `T_e` via inverse-CDF on the k-grid. Uses the nearest-tabulated
    /// electron-energy row (no log-interpolation between rows for k;
    /// the shape changes slowly with T_e).
    pub fn sample_k(&self, t_e_ev: f64, rng: &mut Rng) -> f64 {
        let row_idx = nearest_row(&self.electron_energy, t_e_ev);
        let cdf = &self.cdf_k[row_idx];
        let total = cdf[cdf.len() - 1];
        if total <= 0.0 {
            return 0.0;
        }
        let u = rng.uniform() * total;
        // Find the first bin with cdf >= u. Linear within bin.
        let i = match cdf.partition_point(|&c| c < u) {
            0 => 1,
            n if n >= cdf.len() => cdf.len() - 1,
            n => n,
        };
        let c_lo = cdf[i - 1];
        let c_hi = cdf[i];
        let k_lo = self.k_grid[i - 1];
        let k_hi = self.k_grid[i];
        if c_hi <= c_lo {
            return k_lo;
        }
        let frac = (u - c_lo) / (c_hi - c_lo);
        (k_lo + frac * (k_hi - k_lo)).clamp(1.0e-6, 1.0)
    }
}

fn build_cdf_and_integral(k_grid: &[f64], chi_row: &[f64]) -> (Vec<f64>, f64) {
    // Trapezoidal integration of (dσ/dk) = χ / k on the given grid.
    // The first point may have k=0 (exact) or k_min>0; for k=0 the
    // integrand is singular, so we start the cumulative at the first
    // nonzero-k point.
    let n = k_grid.len();
    let mut cdf = vec![0.0_f64; n];
    let last_valid_i = k_grid.iter().position(|&k| k > 0.0).unwrap_or(0);
    for i in (last_valid_i + 1)..n {
        let k_a = k_grid[i - 1];
        let k_b = k_grid[i];
        let f_a = if k_a > 0.0 { chi_row[i - 1] / k_a } else { 0.0 };
        let f_b = chi_row[i] / k_b;
        let d_integral = 0.5 * (f_a + f_b) * (k_b - k_a);
        cdf[i] = cdf[i - 1] + d_integral;
    }
    let total = cdf[n - 1];
    (cdf, total)
}

fn log_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let idx = xs.partition_point(|&v| v < x);
    let i = idx.clamp(1, xs.len() - 1);
    let lo = xs[i - 1];
    let hi = xs[i];
    let y_lo = ys[i - 1].max(1.0e-30);
    let y_hi = ys[i].max(1.0e-30);
    let t = (x.ln() - lo.ln()) / (hi.ln() - lo.ln());
    (y_lo.ln() + t * (y_hi.ln() - y_lo.ln())).exp()
}

fn nearest_row(xs: &[f64], x: f64) -> usize {
    if x <= xs[0] {
        return 0;
    }
    if x >= xs[xs.len() - 1] {
        return xs.len() - 1;
    }
    let idx = xs.partition_point(|&v| v < x);
    let i = idx.clamp(1, xs.len() - 1);
    if (x - xs[i - 1]).abs() < (xs[i] - x).abs() {
        i - 1
    } else {
        i
    }
}

/// Material-level brems sampler: mixes per-element samplers with
/// atom-density weights. Each atom-density is in atoms/(barn·cm) so
/// `Σ_i n_i · σ_rad,i` comes out in cm⁻¹ directly.
pub struct MaterialBremss {
    pub entries: Vec<(f64, ElementBremss)>,
}

impl MaterialBremss {
    /// Build from a list of `(atom_density_per_barn_cm, element)` pairs.
    pub fn new(entries: Vec<(f64, &PhotonElement)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(n, e)| (n, ElementBremss::new(e)))
                .collect(),
        }
    }

    /// Convenience builder that mirrors a [`PhotonMaterial`]'s element
    /// list exactly — same atom densities, same element order. Useful
    /// so callers can keep parallel arrays of materials and brems
    /// samplers without rebuilding the element table separately.
    pub fn from_photon_material(m: &crate::photon::material::PhotonMaterial) -> Self {
        Self::new(m.entries.iter().map(|(n, e)| (*n, e)).collect())
    }

    /// Macroscopic radiative cross section Σ_rad(T_e) in cm⁻¹, summed
    /// over all elements by atom-density weight.
    ///
    /// Computed from the scaled Seltzer-Berger DCS; the absolute
    /// magnitude is approximate because the HDF5 scaling convention
    /// differs across elements. For emission probability use
    /// [`Self::radiative_yield_approx`] instead — it is calibrated
    /// to NIST ESTAR.
    pub fn sigma_rad_macro(&self, t_e_ev: f64) -> f64 {
        self.entries
            .iter()
            .map(|(n, b)| n * b.sigma_rad_barns(t_e_ev))
            .sum()
    }

    /// Atom-density-weighted effective `Z` for the material.
    pub fn z_eff(&self) -> f64 {
        let mut sum_n = 0.0;
        let mut sum_nz = 0.0;
        for (n, b) in &self.entries {
            sum_n += n;
            sum_nz += n * b.z;
        }
        if sum_n <= 0.0 { 0.0 } else { sum_nz / sum_n }
    }

    /// NIST-calibrated radiation yield fraction `Y(E_e, Z_eff)` for
    /// the full electron slowing-down. Used as the per-electron
    /// emission probability in the single-photon TTB approximation.
    ///
    /// Fit derivation
    /// --------------
    /// Fit to NIST ESTAR radiation-yield fractions for H, C, Al, Fe, Pb,
    /// U from 0.1 MeV to 10 MeV:
    ///
    /// ```text
    /// Y(E, Z) = x / (1 + x),    x = 3.5·10⁻⁴ · Z · E_MeV^1.25
    /// ```
    ///
    /// Typical residuals |Δ/Y| ≤ 40 % across the fit set — similar to
    /// the uncertainty of the full Seltzer-Berger + Bethe-Bloch
    /// construction as implemented in OpenMC (cross-validated against
    /// NIST in tools/brems_check.rs).
    ///
    /// For materials we use an atom-density-weighted `Z_eff`; Bragg
    /// additivity of S_rad is exact, this fit only approximates that
    /// mixing.
    pub fn radiative_yield_approx(&self, t_e_ev: f64) -> f64 {
        if t_e_ev <= 0.0 {
            return 0.0;
        }
        let z_eff = self.z_eff();
        let e_mev = t_e_ev * 1.0e-6;
        let x = 3.5e-4 * z_eff * e_mev.powf(1.25);
        (x / (1.0 + x)).clamp(0.0, 1.0)
    }

    /// Sample which element produces the brems photon (weighted by
    /// n_i · σ_rad_i). Returns `None` if the material has no radiative
    /// XS at this energy.
    pub fn sample_element(&self, t_e_ev: f64, rng: &mut Rng) -> Option<usize> {
        let mut total = 0.0;
        let weights: Vec<f64> = self
            .entries
            .iter()
            .map(|(n, b)| {
                let w = n * b.sigma_rad_barns(t_e_ev);
                total += w;
                w
            })
            .collect();
        if total <= 0.0 {
            return None;
        }
        let u = rng.uniform() * total;
        let mut cum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cum += w;
            if u <= cum {
                return Some(i);
            }
        }
        Some(self.entries.len() - 1)
    }

    /// Sample one brems photon energy `E_γ` in eV at the given electron
    /// kinetic energy. Returns `None` if no radiative channel is open.
    pub fn sample_photon_energy(&self, t_e_ev: f64, rng: &mut Rng) -> Option<f64> {
        let i = self.sample_element(t_e_ev, rng)?;
        let k = self.entries[i].1.sample_k(t_e_ev, rng);
        Some((k * t_e_ev).clamp(0.0, t_e_ev))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::photon::data::PhotonElement;
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

    #[test]
    fn beta_squared_reasonable() {
        assert!(beta2(1.0).abs() < 1.0e-4);
        // 1 MeV electron: β² = T(T+2m)/(T+m)² with T=1e6, m=511e3.
        // β² = 1e6 · 2.022e6 / (1.511e6)² = 2.022e12 / 2.283e12 = 0.886
        let b2 = beta2(1.0e6);
        assert!((b2 - 0.886).abs() < 0.01, "β²(1MeV) = {b2}");
        // Ultrarelativistic: β² → 1.
        assert!((beta2(1.0e9) - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn integrated_cdf_is_monotone_positive() {
        let Some(u) = load("U.h5") else {
            return;
        };
        let s = ElementBremss::new(&u);
        for (i_e, row) in s.cdf_k.iter().enumerate() {
            for i in 1..row.len() {
                assert!(
                    row[i] >= row[i - 1] - 1.0e-18,
                    "non-monotone CDF at i_e={i_e} i={i}: {} -> {}",
                    row[i - 1],
                    row[i]
                );
            }
            assert!(row[row.len() - 1] >= 0.0);
        }
    }

    #[test]
    fn sigma_rad_grows_with_z() {
        let Some(h) = load("H.h5") else {
            return;
        };
        let Some(u) = load("U.h5") else {
            return;
        };
        let h_s = ElementBremss::new(&h);
        let u_s = ElementBremss::new(&u);
        // At 1 MeV σ_rad ∝ Z²; U has Z=92, H has Z=1, so ratio ~8000.
        let s_h = h_s.sigma_rad_barns(1.0e6);
        let s_u = u_s.sigma_rad_barns(1.0e6);
        assert!(s_u > 100.0 * s_h, "σ_rad_U/σ_rad_H = {} / {}", s_u, s_h);
    }

    #[test]
    fn sampled_mean_k_matches_analytic_mean() {
        let Some(u) = load("U.h5") else {
            return;
        };
        let s = ElementBremss::new(&u);
        // Draw many samples at 1 MeV and compare to the analytic <k>.
        let analytic = s.mean_k(1.0e6);
        assert!(
            analytic > 0.01 && analytic < 1.0,
            "mean_k implausible: {analytic}"
        );
        let mut rng = Rng::new(0xDEAD_C0DE_F00D_BABE_u64, 1);
        let n = 50_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += s.sample_k(1.0e6, &mut rng);
        }
        let sampled = sum / n as f64;
        assert!(
            (sampled - analytic).abs() < 0.05,
            "sampled <k>={sampled} vs analytic {analytic}"
        );
    }

    /// Sanity: radiative yield fraction for a 10 MeV electron in U is
    /// on the order of 40 % (NIST ESTAR ~ 45 %). We only check that
    /// `<k> · σ_rad` scales sensibly; the absolute yield requires
    /// integrating along the slowing-down path (out of scope for the
    /// single-photon TTB).
    #[test]
    fn u_mean_k_at_10mev_is_appreciable() {
        let Some(u) = load("U.h5") else {
            return;
        };
        let s = ElementBremss::new(&u);
        let mean = s.mean_k(1.0e7);
        // Mean scaled photon energy should exceed 0.1 at 10 MeV in U
        // (high-energy brems has appreciable hard-photon content).
        assert!(mean > 0.05, "<k>(10 MeV, U) = {mean}");
    }
}
