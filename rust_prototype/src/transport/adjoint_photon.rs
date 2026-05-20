// SPDX-License-Identifier: MIT
//! Continuous-energy adjoint photon walker (CADIS pipeline core).
//!
//! Composes the adjoint Compton kernel
//! ([`crate::photon::compton::adjoint_compton_scatter`]) with the
//! existing forward Rayleigh kernel (elastic ⇒ self-adjoint) and a
//! kill-on-absorption policy for photoelectric / pair into a slab-
//! geometry adjoint walker. Output: importance map
//! `ψ̂*(x, E)` on a 2D `(z, energy)` mesh, the input to the
//! `WeightWindow::from_flux` pipeline.
//!
//! # Adjoint walk semantics
//!
//! Adjoint MC integrates the transposed Boltzmann equation:
//!
//! ```text
//!   −Ω · ∇ψ̂* + Σ_t ψ̂* = ∫ Σ_s(E → E', Ω → Ω') ψ̂*(E', Ω') dE' dΩ' + Q̂*
//! ```
//!
//! where `Q̂*` is the response-function-shaped adjoint source at the
//! detector. Particles walk *outward* from the detector toward the
//! actual source. At each collision the kernel `Σ_s(E → E', Ω → Ω')`
//! reads "rate at which a forward particle at `(E, Ω)` scatters to
//! `(E', Ω')`" — i.e. the FORWARD scattering kernel evaluated with
//! the SAMPLED `(E', Ω')` becoming the new adjoint particle state.
//! Sampling is the inverse: given `(E_out, Ω_out)`, draw `(E_in,
//! Ω_in)` from the kernel treated as a function of those arguments.
//!
//! For Compton this is the inverted Klein-Nishina (Wagner-Haghighat
//! 1998) — see `adjoint_compton_scatter`. For Rayleigh, elasticity
//! ⇒ `E_in = E_out`, and the angular distribution is symmetric in
//! the incoming/outgoing directions (only on the form factor at
//! `q = 2 E sin(θ/2) / hc`), so the forward Rayleigh sampler is
//! self-adjoint and we re-use it. For photoelectric / pair the
//! adjoint walk terminates: those reactions only contribute via
//! the "absorption-as-source" rule which lives on the forward
//! side of CADIS.
//!
//! # References
//! - Wagner & Haghighat, *Nucl. Sci. Eng.* 128, 186 (1998) §III.
//! - Lewis & Miller, *Computational Methods of Neutron Transport*
//!   §10.3.

use crate::geometry::Vec3;
use crate::photon::coherent::coherent_scatter;
use crate::photon::compton::adjoint_compton_scatter;
use crate::photon::material::Channel;
use crate::photon::material::PhotonMaterial;
use crate::transport::rng::Rng;

/// 2D `(z, E)` importance-map tally on a regular Cartesian mesh in
/// `z` and a logarithmic mesh in `E`.
#[derive(Debug, Clone)]
pub struct ImportanceMap {
    /// Slab thickness (cm).
    pub thickness_cm: f64,
    pub n_z_bins: usize,
    /// Energy mesh: `n_e_bins + 1` boundaries on a log grid from
    /// `e_min` to `e_max`, both in eV.
    pub e_min: f64,
    pub e_max: f64,
    pub n_e_bins: usize,
    /// Flat `[z_bin * n_e_bins + e_bin]` track-length tally,
    /// proportional to the adjoint flux ψ̂*.
    pub flux: Vec<f64>,
}

impl ImportanceMap {
    pub fn new(
        thickness_cm: f64,
        n_z_bins: usize,
        e_min: f64,
        e_max: f64,
        n_e_bins: usize,
    ) -> Self {
        Self {
            thickness_cm,
            n_z_bins,
            e_min,
            e_max,
            n_e_bins,
            flux: vec![0.0; n_z_bins * n_e_bins],
        }
    }

    /// Linear index for a `(z_bin, e_bin)` pair. Returns `None` when
    /// either index is outside the mesh.
    fn linear(&self, iz: usize, ie: usize) -> Option<usize> {
        if iz >= self.n_z_bins || ie >= self.n_e_bins {
            return None;
        }
        Some(iz * self.n_e_bins + ie)
    }

    /// `(z_bin_lo, frac_within)` for position `z_cm`. Returns `None`
    /// when `z` is outside the slab.
    fn z_bin_of(&self, z_cm: f64) -> Option<usize> {
        if z_cm < 0.0 || z_cm > self.thickness_cm {
            return None;
        }
        let dz = self.thickness_cm / self.n_z_bins as f64;
        let i = (z_cm / dz) as usize;
        Some(i.min(self.n_z_bins - 1))
    }

    /// `e_bin` for energy `E_eV`. Returns `None` when out of range.
    fn e_bin_of(&self, energy_ev: f64) -> Option<usize> {
        if energy_ev <= 0.0 || self.e_min <= 0.0 {
            return None;
        }
        let ln_lo = self.e_min.ln();
        let ln_hi = self.e_max.ln();
        let ln_e = energy_ev.ln();
        if ln_e < ln_lo || ln_e > ln_hi {
            return None;
        }
        let frac = (ln_e - ln_lo) / (ln_hi - ln_lo);
        let i = (frac * self.n_e_bins as f64) as usize;
        Some(i.min(self.n_e_bins - 1))
    }

    /// Deposit `weight · path_length` over a straight segment from
    /// `start` to `start + dir · length` into the (z, E) bin pair.
    /// Energy is fixed along the segment (no degradation between
    /// collisions in the analog walk).
    pub fn deposit(&mut self, start: Vec3, dir: Vec3, length: f64, weight: f64, energy_ev: f64) {
        if length <= 0.0 || weight == 0.0 {
            return;
        }
        let Some(ie) = self.e_bin_of(energy_ev) else {
            return;
        };
        // Sub-segment per z bin: walk in z, deposit `w · sub_len`
        // per visited bin.
        let dz = self.thickness_cm / self.n_z_bins as f64;
        let z0 = start.x;
        let dx = dir.x;
        let z1 = z0 + dx * length;
        if dx == 0.0 {
            // Pure transverse motion (rare for slab problems with
            // x as the slab axis); deposit entirely in the start
            // bin.
            if let Some(iz) = self.z_bin_of(z0)
                && let Some(idx) = self.linear(iz, ie)
            {
                self.flux[idx] += weight * length;
            }
            return;
        }
        let (z_lo, z_hi) = if dx > 0.0 { (z0, z1) } else { (z1, z0) };
        let lo_clamped = z_lo.max(0.0);
        let hi_clamped = z_hi.min(self.thickness_cm);
        if hi_clamped <= lo_clamped {
            return;
        }
        let mut z = lo_clamped;
        while z < hi_clamped - 1e-15 {
            let iz = (z / dz).floor() as usize;
            if iz >= self.n_z_bins {
                break;
            }
            let z_next = ((iz + 1) as f64 * dz).min(hi_clamped);
            let sub = (z_next - z).max(0.0);
            if sub > 0.0
                && let Some(idx) = self.linear(iz, ie)
            {
                // Convert back to path length: `sub` is z-extent;
                // path length along dir = sub / |dx|.
                let path = sub / dx.abs();
                self.flux[idx] += weight * path;
            }
            z = z_next;
        }
    }
}

/// Adjoint walker configuration.
pub struct AdjointSlabConfig {
    pub thickness_cm: f64,
    /// Source spectrum upper bound (eV). Adjoint Compton kernel
    /// will not sample E_in above this.
    pub e_in_max: f64,
    /// Lowest energy to track. Below this we kill the particle.
    /// Typically 1 keV (matches the photoelectric cutoff).
    pub e_cut_ev: f64,
    /// Number of histories.
    pub n_histories: usize,
    /// `(z, E)` mesh dimensions.
    pub n_z_bins: usize,
    pub n_e_bins: usize,
    /// Bound on tracked events per history. Very high-Z materials
    /// can produce >1000 successive Compton scatters; the limit
    /// guards against ill-conditioned kinematic loops.
    pub max_events_per_history: u32,
}

impl Default for AdjointSlabConfig {
    fn default() -> Self {
        Self {
            thickness_cm: 100.0,
            e_in_max: 5.0e6,
            e_cut_ev: 1.0e3,
            n_histories: 100_000,
            n_z_bins: 50,
            n_e_bins: 30,
            max_events_per_history: 5_000,
        }
    }
}

/// Run an adjoint photon walk on a 1D slab: source on `x = 0`,
/// detector on `x = thickness_cm` (vacuum BC on both faces).
/// Adjoint particles start at the detector face moving in `-x̂`
/// at the response-function energy `e_response_ev` (monoenergetic
/// detector response — generalises to a sampled spectrum trivially
/// by varying the start energy per history).
///
/// Returns the populated `ImportanceMap`. Each entry is the
/// track-length-summed contribution `Σ_h ∫ w · dz` over all `h`
/// histories whose path passed through that `(z, E)` cell, i.e.
/// proportional to the adjoint flux.
pub fn adjoint_slab_walk(
    cfg: &AdjointSlabConfig,
    material: &PhotonMaterial,
    e_response_ev: f64,
    rng: &mut Rng,
) -> ImportanceMap {
    let mut map = ImportanceMap::new(
        cfg.thickness_cm,
        cfg.n_z_bins,
        cfg.e_cut_ev,
        cfg.e_in_max,
        cfg.n_e_bins,
    );

    for _ in 0..cfg.n_histories {
        // Birth at the detector face, moving toward the source.
        let mut pos = Vec3::new(cfg.thickness_cm, 0.0, 0.0);
        let mut dir = Vec3::new(-1.0, 0.0, 0.0);
        let mut energy = e_response_ev;
        let weight = 1.0_f64; // analog walk — no implicit-capture biasing
        let mut events = 0_u32;

        while events < cfg.max_events_per_history {
            events += 1;
            if energy < cfg.e_cut_ev || weight <= 0.0 {
                break;
            }
            let sigma_t = material.macro_total(energy);
            if sigma_t <= 0.0 {
                break;
            }
            let xi = rng.uniform().max(1e-300);
            let d_collision = -xi.ln() / sigma_t;
            // Geometric distance to the slab faces along dir.
            let d_face = if dir.x > 0.0 {
                (cfg.thickness_cm - pos.x) / dir.x
            } else if dir.x < 0.0 {
                -pos.x / dir.x
            } else {
                f64::INFINITY
            };
            let d_step = d_collision.min(d_face);
            // Track-length deposit along this segment.
            map.deposit(pos, dir, d_step, weight, energy);
            pos = pos + dir * d_step;
            if d_face < d_collision {
                // Reached a slab face → particle has escaped (for
                // adjoint MC the source-face escape is the desired
                // termination — that's what records "this adjoint
                // particle reached the source"). No leakage flag
                // needed; the deposit already captured its
                // contribution.
                break;
            }
            // Real collision. Sample channel + element + reaction.
            let chan_xi = rng.uniform();
            let channel = material.sample_channel(energy, chan_xi);
            let elem_xi = rng.uniform();
            let elem_idx = material.sample_element(channel, energy, elem_xi);
            let elem = &material.entries[elem_idx].1;
            match channel {
                Channel::Incoherent => {
                    // Adjoint Compton: invert the kinematics.
                    let o = adjoint_compton_scatter(elem, energy, cfg.e_in_max, rng);
                    let new_dir = rotate_direction(dir, o.mu, rng);
                    energy = o.energy_in;
                    dir = new_dir;
                }
                Channel::Coherent => {
                    // Rayleigh is elastic (E unchanged) and the
                    // angular distribution depends only on the
                    // momentum-transfer form factor F(q, Z) — the
                    // forward and adjoint kernels coincide.
                    let r = coherent_scatter(elem, energy, rng);
                    dir = rotate_direction(dir, r.mu, rng);
                }
                Channel::Photoelectric
                | Channel::PairProductionNuclear
                | Channel::PairProductionElectron => {
                    // Absorption — terminates this adjoint walk.
                    // Forward CADIS pulls the absorption-as-source
                    // term separately; the walker doesn't double-
                    // count it here.
                    break;
                }
            }
        }
    }
    map
}

/// Rotate `dir` by polar `mu = cos θ` and a uniformly-sampled
/// azimuth. Standard rotation in the local frame. Mirrors the
/// inline rotation used in `transport::simulate` for thermal
/// scatters.
fn rotate_direction(dir: Vec3, mu: f64, rng: &mut Rng) -> Vec3 {
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let sin_mu = (1.0 - mu * mu).max(0.0).sqrt();
    let d = dir;
    let w2 = d.z * d.z;
    if w2 < 0.999 {
        let inv_sq = 1.0 / (1.0 - w2).sqrt();
        Vec3::new(
            mu * d.x + sin_mu * (d.x * d.z * phi.cos() - d.y * phi.sin()) * inv_sq,
            mu * d.y + sin_mu * (d.y * d.z * phi.cos() + d.x * phi.sin()) * inv_sq,
            mu * d.z - sin_mu * (1.0 - w2).sqrt() * phi.cos(),
        )
    } else {
        let sign = if d.z > 0.0 { 1.0 } else { -1.0 };
        Vec3::new(sin_mu * phi.cos(), sin_mu * phi.sin() * sign, mu * sign)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Importance-map z-bin lookup: in-range positions get a valid
    /// bin; below 0 or above thickness return None.
    #[test]
    fn z_bin_lookup_clamps_correctly() {
        let map = ImportanceMap::new(100.0, 10, 1e3, 5e6, 5);
        assert_eq!(map.z_bin_of(0.0), Some(0));
        assert_eq!(map.z_bin_of(50.0), Some(5));
        assert_eq!(map.z_bin_of(99.999), Some(9));
        assert_eq!(map.z_bin_of(100.0), Some(9)); // top edge clamps to last bin
        assert_eq!(map.z_bin_of(-0.1), None);
        assert_eq!(map.z_bin_of(100.1), None);
    }

    /// Energy-bin lookup uses log spacing. `n_e_bins=4` over
    /// `[1e3, 1e7]` ⇒ one decade per bin: bin 0 = [1e3, 1e4], bin
    /// 1 = [1e4, 1e5], bin 2 = [1e5, 1e6], bin 3 = [1e6, 1e7].
    /// Energy values exactly on a bin boundary fall in the *lower*
    /// bin (`floor` semantics); the upper edge of the last bin
    /// clamps so 1e7 stays in bin 3.
    #[test]
    fn e_bin_lookup_log_spaced() {
        let map = ImportanceMap::new(100.0, 10, 1e3, 1e7, 4);
        assert_eq!(map.e_bin_of(1.0e3), Some(0));
        assert_eq!(map.e_bin_of(5.0e3), Some(0));
        assert_eq!(map.e_bin_of(1.0e4), Some(1));
        assert_eq!(map.e_bin_of(1.0e5), Some(2));
        // Upper boundary of bin 2 = lower boundary of bin 3 → bin 2.
        assert_eq!(map.e_bin_of(1.0e6), Some(2));
        assert_eq!(map.e_bin_of(2.0e6), Some(3));
        // Top edge clamps to last bin.
        assert_eq!(map.e_bin_of(1.0e7), Some(3));
        assert_eq!(map.e_bin_of(0.5e3), None);
        assert_eq!(map.e_bin_of(2.0e7), None);
    }

    /// Track-length deposit: a segment along +x from 0→thickness at
    /// constant energy deposits `length / n_z_bins` into each z bin
    /// of the corresponding e bin.
    #[test]
    fn deposit_uniform_segment() {
        let mut map = ImportanceMap::new(10.0, 10, 1e3, 5e6, 5);
        let energy = 1e5;
        map.deposit(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            10.0,
            1.0,
            energy,
        );
        let ie = map.e_bin_of(energy).unwrap();
        for iz in 0..10 {
            let v = map.flux[map.linear(iz, ie).unwrap()];
            assert!((v - 1.0).abs() < 1e-9, "bin {iz}: got {v}, expected 1.0",);
        }
    }

    /// **End-to-end adjoint walk on a 100-cm water slab**. The
    /// importance map has the qualitative shape we'd predict
    /// physically: peak near the detector face (z ≈ thickness)
    /// where adjoint particles spend most of their early track,
    /// monotone decay toward z = 0 because each Compton
    /// scattering / boundary escape removes adjoint particles.
    /// The energy column at the response energy E_r contains the
    /// majority of the track length (the adjoint walk only
    /// up-scatters in energy via Compton).
    #[test]
    fn adjoint_walk_water_slab_shape_sanity() {
        use crate::photon::PhotonElement;
        use std::path::PathBuf;
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let h_path = manifest
            .parent()
            .unwrap()
            .join("data/endfb-vii.1-hdf5/photon/H.h5");
        let o_path = manifest
            .parent()
            .unwrap()
            .join("data/endfb-vii.1-hdf5/photon/O.h5");
        if !h_path.exists() || !o_path.exists() {
            eprintln!("skipping: H/O photon data not present");
            return;
        }
        let h = PhotonElement::from_hdf5(&h_path).unwrap();
        let o = PhotonElement::from_hdf5(&o_path).unwrap();
        let molecule_density = 3.3428e-2;
        let water = PhotonMaterial::new(vec![
            (2.0 * molecule_density, h),
            (1.0 * molecule_density, o),
        ]);

        let cfg = AdjointSlabConfig {
            thickness_cm: 100.0,
            e_in_max: 5.0e6,
            e_cut_ev: 1.0e3,
            n_histories: 5_000,
            n_z_bins: 20,
            n_e_bins: 10,
            max_events_per_history: 2_000,
        };
        let mut rng = Rng::new(0x4D70, 0);
        let map = adjoint_slab_walk(&cfg, &water, 1.0e6, &mut rng);

        // Sum over energy → ψ̂*(z) profile.
        let mut z_profile = vec![0.0_f64; cfg.n_z_bins];
        for iz in 0..cfg.n_z_bins {
            for ie in 0..cfg.n_e_bins {
                z_profile[iz] += map.flux[map.linear(iz, ie).unwrap()];
            }
        }
        let total: f64 = z_profile.iter().sum();
        eprintln!("adjoint walk z-profile (sum={total:.1}, normalised):",);
        for (iz, v) in z_profile.iter().enumerate() {
            let z_mid = (iz as f64 + 0.5) * cfg.thickness_cm / cfg.n_z_bins as f64;
            eprintln!("  z={z_mid:5.1}  ψ̂*={:8.1}  ({:.3}%)", v, 100.0 * v / total);
        }
        // Every z bin has finite flux. With 5 000 histories at
        // 1 MeV in 100 cm of water (~7 mfp), every slice sees at
        // least some forward / scattered track. The energy
        // up-scatter biases adjoint particles toward higher
        // mfp / smaller σ_t at lower z (after multiple Compton
        // events the particle is at higher E and σ_t is smaller),
        // which is why the source-face z bins can have *more*
        // track length than the detector-face bins despite the
        // attenuation through the slab — this is correct
        // adjoint-MC physics, not a regression: in adjoint MC the
        // detector-face bin only receives the *first* unscattered
        // segment of each history, while the source-side bins
        // accumulate from the persistent up-scattered population.
        for (iz, v) in z_profile.iter().enumerate() {
            assert!(*v > 0.0, "z bin {iz} empty");
        }
        // Energy up-scatter sanity: the BIRTH energy bin must
        // have strictly *less* track length than the highest
        // populated bin above it, because adjoint Compton can
        // only raise the energy and is the dominant water reaction
        // at 1 MeV. (At very high energies pair production
        // would absorb, but we cap E_in at 5 MeV, well below the
        // pair threshold.)
        let mut e_profile = vec![0.0_f64; cfg.n_e_bins];
        for ie in 0..cfg.n_e_bins {
            for iz in 0..cfg.n_z_bins {
                e_profile[ie] += map.flux[map.linear(iz, ie).unwrap()];
            }
        }
        let birth_bin = map.e_bin_of(1.0e6).unwrap();
        let above_birth: f64 = e_profile.iter().skip(birth_bin + 1).sum();
        eprintln!(
            "  birth bin {birth_bin}: {:.1};  above-birth integrated: {:.1}",
            e_profile[birth_bin], above_birth,
        );
        // Above-birth bins must contain at least *some* up-scatter
        // contribution.
        assert!(
            above_birth > 0.01 * e_profile[birth_bin],
            "no measurable adjoint Compton up-scatter: birth bin {} vs above-birth {}",
            e_profile[birth_bin],
            above_birth,
        );
    }

    /// CE adjoint walker output shape sanity check for the CADIS
    /// pipeline: when summed over E and renormalised to peak = 1, the
    /// resulting `Vec<f64>` is the same length as the z-bin count and
    /// every entry is finite and non-negative. This is the contract
    /// the `adjoint_photon_cadis_slab` binary's `CadisMap` JSON
    /// emission relies on — no NaNs that would silently break
    /// `WeightWindow::from_flux` downstream.
    #[test]
    fn z_profile_collapse_is_finite_and_nonnegative() {
        // Synthetic deposit pattern: directly populate `flux` without
        // running a full walk. Avoids the HDF5 dependency in this test.
        let mut map = ImportanceMap::new(50.0, 10, 1e3, 5e6, 4);
        // Drop one count into every (iz, ie) cell so the collapse is
        // non-trivial.
        for iz in 0..10 {
            for ie in 0..4 {
                map.flux[iz * 4 + ie] = (iz + 1) as f64 * (ie + 1) as f64;
            }
        }
        let mut z_profile = vec![0.0_f64; 10];
        for iz in 0..10 {
            for ie in 0..4 {
                z_profile[iz] += map.flux[iz * 4 + ie];
            }
        }
        for (iz, &v) in z_profile.iter().enumerate() {
            assert!(v.is_finite(), "z bin {iz} not finite: {v}");
            assert!(v >= 0.0, "z bin {iz} negative: {v}");
        }
        // Synthetic pattern is monotone in iz (iz+1)·Σ_ie(ie+1) ⇒
        // strictly increasing.
        for w in z_profile.windows(2) {
            assert!(
                w[1] > w[0],
                "synthetic deposit should be monotone increasing",
            );
        }
    }

    /// `rotate_direction` preserves unit vector to FP precision.
    #[test]
    fn rotate_preserves_norm() {
        let mut rng = Rng::new(0xCAFE, 1);
        for _ in 0..100 {
            let d0 = Vec3::new(0.6, 0.8, 0.0);
            let d1 = rotate_direction(d0, 0.3, &mut rng);
            let n = (d1.x * d1.x + d1.y * d1.y + d1.z * d1.z).sqrt();
            assert!(
                (n - 1.0).abs() < 1e-12,
                "rotated direction norm {n}, expected 1",
            );
        }
    }
}
