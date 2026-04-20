//! Hybrid SVD + Windowed-Multipole cross-section provider.
//!
//! Wraps an existing `SvdXsProvider` and, for nuclides that have WMP data
//! loaded, intercepts lookups inside `[E_min^WMP, E_max^WMP]` and replaces
//! the SVD-reconstructed (elastic, fission, capture) with the WMP
//! evaluation at the nuclide's target temperature.
//!
//! Non-WMP channels (inelastic, n2n, n3n) always come from the underlying
//! SVD provider; outside the WMP window the SVD path is used unchanged.
//! URR handling is delegated to the SVD provider — URR data stays
//! applicable in the region above `E_max^WMP`.

use std::sync::Arc;

use crate::hdf5_reader::{AngularDistribution, DiscreteLevelInfo, EnergyDistribution};
use crate::physics::collision::MicroXs;
use crate::thermal::ThermalScatteringData;
use crate::transport::simulate::XsProvider;
use crate::transport::xs_provider::SvdXsProvider;
use crate::wmp::WindowedMultipole;

/// Hybrid provider: SVD everywhere, overridden by WMP inside the resolved
/// resonance window for nuclides that carry WMP data.
pub struct HybridSvdWmpXsProvider {
    inner: SvdXsProvider,
    /// One entry per nuclide; `Some(wmp, T_kelvin)` if WMP applies.
    wmps: Vec<Option<(Arc<WindowedMultipole>, f64)>>,
}

impl HybridSvdWmpXsProvider {
    /// Wrap an `SvdXsProvider`. `wmps` length must equal the number of
    /// nuclides in the inner provider; `None` entries keep the SVD path.
    pub fn new(inner: SvdXsProvider, wmps: Vec<Option<(Arc<WindowedMultipole>, f64)>>) -> Self {
        assert_eq!(
            inner.nuclides.len(),
            wmps.len(),
            "wmps length must match nuclide count"
        );
        Self { inner, wmps }
    }

    /// Count how many nuclides actually have WMP coverage — useful for logs.
    pub fn covered_nuclides(&self) -> usize {
        self.wmps.iter().filter(|w| w.is_some()).count()
    }

    /// Memory budget report for the hybrid architecture.
    ///
    /// Reports both the current in-solver memory (full SVD basis + WMP
    /// payload — what we actually carry because the full SVD basis is
    /// retained for inelastic/n2n/n3n fall-through and because we have
    /// not yet rebuilt the elastic/fission/capture kernels on a
    /// smooth-only energy grid) and a measured projection of the
    /// smooth-only layout: for each nuclide with WMP coverage, the
    /// elastic/fission/capture kernel bytes are reduced in proportion
    /// to the fraction of the kernel's energy grid that falls outside
    /// the WMP window. This yields the engine-level number that
    /// matches the representation-byte total reported in the hybrid
    /// memory table, computed from live data rather than an offline
    /// script.
    pub fn memory_report(&self) -> HybridMemoryReport {
        let mut current_svd = 0_usize;
        let mut smooth_only_svd = 0_usize;
        let mut wmp_payload = 0_usize;

        for (i, nuc) in self.inner.nuclides.iter().enumerate() {
            // Full-grid bytes for every reaction
            let k_el = nuc.elastic.as_ref().map_or(0, |r| r.kernel.memory_bytes());
            let k_fis = nuc.fission.as_ref().map_or(0, |r| r.kernel.memory_bytes());
            let k_cap = nuc.capture.as_ref().map_or(0, |r| r.kernel.memory_bytes());
            let k_in = nuc
                .inelastic
                .as_ref()
                .map_or(0, |r| r.kernel.memory_bytes());
            let k_2n = nuc.n2n.as_ref().map_or(0, |r| r.kernel.memory_bytes());
            let k_3n = nuc.n3n.as_ref().map_or(0, |r| r.kernel.memory_bytes());
            let k_tt = nuc.total_table.as_ref().map_or(0, |t| t.memory_bytes());
            let k_dl: usize = nuc
                .discrete_levels
                .iter()
                .map(|l| l.kernel.as_ref().map_or(0, |r| r.kernel.memory_bytes()))
                .sum();

            let full_nuc = k_el + k_fis + k_cap + k_in + k_2n + k_3n + k_tt + k_dl;
            current_svd += full_nuc;

            match self.wmps[i].as_ref() {
                None => {
                    // No WMP — full SVD everywhere.
                    smooth_only_svd += full_nuc;
                }
                Some((wmp, _)) => {
                    // Fraction of elastic/fission/capture grid outside
                    // [E_min^WMP, E_max^WMP]. Use the elastic kernel's
                    // grid as a representative since all three share it.
                    let frac_smooth = kernel_smooth_fraction(nuc, wmp.e_min, wmp.e_max);
                    let efc = k_el + k_fis + k_cap;
                    // For elastic/fission/capture: only smooth grid points need basis.
                    smooth_only_svd += (efc as f64 * frac_smooth).round() as usize;
                    // Inelastic, n2n, n3n, discrete levels stay full-grid.
                    smooth_only_svd += k_in + k_2n + k_3n + k_tt + k_dl;
                }
            }

            if let Some((wmp, _)) = self.wmps[i].as_ref() {
                // WMP payload bytes (poles + windows + curvefit + broaden_poly)
                wmp_payload += wmp_payload_bytes(wmp);
            }
        }

        HybridMemoryReport {
            current_svd_bytes: current_svd,
            smooth_only_svd_bytes: smooth_only_svd,
            wmp_payload_bytes: wmp_payload,
        }
    }
}

pub struct HybridMemoryReport {
    pub current_svd_bytes: usize,
    pub smooth_only_svd_bytes: usize,
    pub wmp_payload_bytes: usize,
}

impl HybridMemoryReport {
    pub fn current_total(&self) -> usize {
        self.current_svd_bytes + self.wmp_payload_bytes
    }
    pub fn smooth_only_total(&self) -> usize {
        self.smooth_only_svd_bytes + self.wmp_payload_bytes
    }
}

/// Fraction of a nuclide's elastic/fission/capture energy grid points
/// that lie outside `[e_lo, e_hi]`. Uses the elastic kernel's grid if
/// present, else fission, else capture; returns 1.0 if no kernel.
fn kernel_smooth_fraction(
    nuc: &crate::transport::xs_provider::NuclideKernels,
    e_lo: f64,
    e_hi: f64,
) -> f64 {
    let grid = nuc
        .elastic
        .as_ref()
        .or(nuc.fission.as_ref())
        .or(nuc.capture.as_ref())
        .map(|r| r.kernel.energies());
    match grid {
        None => 1.0,
        Some(g) => {
            let outside = g.iter().filter(|&&e| e < e_lo || e > e_hi).count();
            if g.is_empty() {
                1.0
            } else {
                outside as f64 / g.len() as f64
            }
        }
    }
}

fn wmp_payload_bytes(w: &crate::wmp::WindowedMultipole) -> usize {
    let poles = w.n_poles * 4 * 16; // complex128
    let windows = w.n_windows * 2 * 4; // i32
    let curvefit = w.n_windows * (w.fit_order + 1) * 3 * 8; // f64
    let broaden = w.n_windows; // u8
    poles + windows + curvefit + broaden
}

impl XsProvider for HybridSvdWmpXsProvider {
    fn lookup(&self, nuclide_idx: usize, energy: f64) -> MicroXs {
        let mut xs = self.inner.lookup(nuclide_idx, energy);

        if let Some((wmp, t_kelvin)) = self.wmps[nuclide_idx].as_ref()
            && energy >= wmp.e_min
            && energy <= wmp.e_max
        {
            let (sig_s, sig_a, sig_f) = wmp.evaluate(energy, *t_kelvin);
            // Floor negative values at zero — WMP with truncated pole sets
            // can produce tiny negative tails between resonances.
            let elastic = sig_s.max(0.0);
            let fission = sig_f.max(0.0);
            let capture = (sig_a - fission).max(0.0);
            // Recompute total from partials.
            let total = elastic + xs.inelastic + xs.n2n + xs.n3n + fission + capture;
            xs.elastic = elastic;
            xs.fission = fission;
            xs.capture = capture;
            xs.total = total;
        }
        xs
    }

    fn discrete_level_info(&self, nuclide_idx: usize) -> Vec<DiscreteLevelInfo> {
        self.inner.discrete_level_info(nuclide_idx)
    }

    fn discrete_level_xs(&self, nuclide_idx: usize, energy: f64) -> Vec<f64> {
        self.inner.discrete_level_xs(nuclide_idx, energy)
    }

    fn has_continuum_inelastic(&self, nuclide_idx: usize) -> bool {
        self.inner.has_continuum_inelastic(nuclide_idx)
    }

    fn elastic_angular_dist(&self, nuclide_idx: usize) -> Option<&AngularDistribution> {
        self.inner.elastic_angular_dist(nuclide_idx)
    }

    fn discrete_level_angles(&self, nuclide_idx: usize) -> &[Option<AngularDistribution>] {
        self.inner.discrete_level_angles(nuclide_idx)
    }

    fn fission_energy_dist(&self, nuclide_idx: usize) -> Option<&EnergyDistribution> {
        self.inner.fission_energy_dist(nuclide_idx)
    }

    fn apply_urr(&self, nuclide_idx: usize, xs: &mut MicroXs, energy: f64, xi: f64) {
        // If we're inside WMP range, URR is not physically applicable
        // (the resonances are already explicit via poles). Skip it.
        if let Some((wmp, _)) = self.wmps[nuclide_idx].as_ref()
            && energy >= wmp.e_min
            && energy <= wmp.e_max
        {
            return;
        }
        self.inner.apply_urr(nuclide_idx, xs, energy, xi);
    }

    fn thermal_scattering(&self, nuclide_idx: usize) -> Option<&ThermalScatteringData> {
        self.inner.thermal_scattering(nuclide_idx)
    }
}
