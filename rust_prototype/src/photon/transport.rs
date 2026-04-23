//! Fixed-source photon transport driver.
//!
//! Streams a photon through a homogeneous `PhotonMaterial` until it is
//! absorbed or escapes the medium. Handles all four interaction
//! channels (coherent, incoherent, photoelectric with full EADL
//! cascade, pair production with positron annihilation), banks
//! secondary photons (fluorescence, annihilation), and tracks local
//! energy deposition for use by a caller-supplied tally.
//!
//! The caller provides a closure `is_inside(pos) -> bool` that the
//! driver consults after each collision and between collisions. A
//! photon that steps outside the medium is considered "escaped" and
//! terminates with its remaining energy returned to the caller.
//!
//! # Simplifications
//! - **Kerma approximation**: electron kinetic energies are deposited
//!   locally. No electron transport, no secondary bremsstrahlung
//!   photons from the scattered electron. Accuracy ~5 % for typical
//!   shielding problems.
//! - **No Doppler broadening** on Compton — outgoing `(E', μ)` is on
//!   the free-electron kinematic curve. Smears the Compton edge by a
//!   few keV on high-Z; will be added in a future commit.
//! - Coherent keeps the photon energy and only deflects direction.
//! - Pair-production positrons stop locally and annihilate at rest;
//!   two 511 keV photons emitted back-to-back with an isotropic axis.

use crate::geometry::Vec3;
use crate::photon::coherent::coherent_scatter;
use crate::photon::compton::compton_scatter;
use crate::photon::material::{Channel, PhotonMaterial};
use crate::photon::pair::{pair_produce, ANNIHILATION_ENERGY_EV};
use crate::photon::photoelectric::{photoelectric_absorb, DEFAULT_PHOTON_CUTOFF_EV};
use crate::transport::rng::Rng;

/// Outcome of one complete source-photon history (including all
/// banked secondaries).
#[derive(Debug, Clone)]
pub struct HistoryResult {
    /// Total energy deposited locally inside the medium (eV).
    pub energy_deposited: f64,
    /// Total energy that escaped the medium (eV). If there are no
    /// geometric boundaries this is always zero.
    pub energy_escaped: f64,
    /// Number of collisions processed (source photon + all secondaries).
    pub n_collisions: u32,
    /// Per-collision record for tallies that need position-resolved
    /// deposition (e.g. pulse-height in a finite detector). Each
    /// entry is `(position, local_deposit_eV)`.
    pub deposits: Vec<(Vec3, f64)>,
}

/// A single photon track in the bank.
#[derive(Debug, Clone, Copy)]
struct BankEntry {
    pos: Vec3,
    dir: Vec3,
    energy: f64,
}

/// Drive a full photon history from a source particle through the
/// `material`. The `is_inside` closure tells the driver whether the
/// given position is inside the transport medium. A photon that
/// enters a region where `is_inside` returns `false` is terminated
/// with its remaining energy added to `energy_escaped`.
///
/// Set `energy_cutoff_ev` to the absorption threshold (e.g. 1 keV);
/// photons whose energy drops below are killed and their energy
/// deposited locally.
pub fn transport_history<F: Fn(Vec3) -> bool>(
    source_pos: Vec3,
    source_dir: Vec3,
    source_energy: f64,
    material: &PhotonMaterial,
    is_inside: F,
    energy_cutoff_ev: f64,
    rng: &mut Rng,
) -> HistoryResult {
    let mut result = HistoryResult {
        energy_deposited: 0.0,
        energy_escaped: 0.0,
        n_collisions: 0,
        deposits: Vec::new(),
    };
    let mut bank: Vec<BankEntry> = Vec::with_capacity(8);
    bank.push(BankEntry {
        pos: source_pos,
        dir: source_dir,
        energy: source_energy,
    });

    while let Some(start) = bank.pop() {
        transport_one(start, material, &is_inside, energy_cutoff_ev, rng, &mut bank, &mut result);
    }

    result
}

/// Transport a single photon track until termination. Banks secondaries.
fn transport_one<F: Fn(Vec3) -> bool>(
    start: BankEntry,
    material: &PhotonMaterial,
    is_inside: &F,
    energy_cutoff_ev: f64,
    rng: &mut Rng,
    bank: &mut Vec<BankEntry>,
    result: &mut HistoryResult,
) {
    let mut pos = start.pos;
    let mut dir = start.dir;
    let mut energy = start.energy;

    // Safety: cap collisions per track to prevent runaway loops from
    // any pathological interaction sequence.
    const MAX_COLLISIONS_PER_TRACK: u32 = 10_000;

    for _ in 0..MAX_COLLISIONS_PER_TRACK {
        if !is_inside(pos) {
            result.energy_escaped += energy;
            return;
        }
        if energy < energy_cutoff_ev {
            result.energy_deposited += energy;
            result.deposits.push((pos, energy));
            return;
        }

        let sigma_tot = material.macro_total(energy);
        if sigma_tot <= 0.0 {
            // Void — photon streams to infinity (boundary-escape if
            // any; otherwise we have to bail).
            result.energy_escaped += energy;
            return;
        }

        // Sample free-flight distance.
        let d = rng.exponential(sigma_tot);
        pos = pos + dir * d;
        if !is_inside(pos) {
            result.energy_escaped += energy;
            return;
        }

        // Collision.
        result.n_collisions += 1;
        let xi_ch = rng.uniform();
        let channel = material.sample_channel(energy, xi_ch);
        let xi_el = rng.uniform();
        let elem_idx = material.sample_element(channel, energy, xi_el);
        let elem = &material.entries[elem_idx].1;

        match channel {
            Channel::Coherent => {
                let out = coherent_scatter(elem, energy, rng);
                dir = deflect(dir, out.mu, rng);
                // Energy unchanged.
            }
            Channel::Incoherent => {
                let out = compton_scatter(elem, energy, rng);
                result.energy_deposited += out.electron_kinetic;
                result.deposits.push((pos, out.electron_kinetic));
                energy = out.energy_out;
                dir = deflect(dir, out.mu, rng);
            }
            Channel::Photoelectric => {
                let out = photoelectric_absorb(elem, energy, DEFAULT_PHOTON_CUTOFF_EV, rng);
                result.energy_deposited += out.local_deposition;
                result.deposits.push((pos, out.local_deposition));
                for ep in out.fluorescence_photons {
                    let (dx, dy, dz) = rng.isotropic_direction();
                    bank.push(BankEntry {
                        pos,
                        dir: Vec3::new(dx, dy, dz),
                        energy: ep,
                    });
                }
                return;
            }
            Channel::PairProductionNuclear | Channel::PairProductionElectron => {
                if let Some(out) = pair_produce(energy, rng) {
                    result.energy_deposited += out.local_deposition();
                    result.deposits.push((pos, out.local_deposition()));
                    // Two 511 keV back-to-back annihilation photons.
                    let (dx, dy, dz) = rng.isotropic_direction();
                    let ann_dir = Vec3::new(dx, dy, dz);
                    bank.push(BankEntry {
                        pos,
                        dir: ann_dir,
                        energy: ANNIHILATION_ENERGY_EV,
                    });
                    bank.push(BankEntry {
                        pos,
                        dir: -ann_dir,
                        energy: ANNIHILATION_ENERGY_EV,
                    });
                } else {
                    // Below threshold — deposit locally.
                    result.energy_deposited += energy;
                    result.deposits.push((pos, energy));
                }
                return;
            }
        }
    }
    // Exceeded collision cap — deposit remaining and warn silently.
    result.energy_deposited += energy;
    result.deposits.push((pos, energy));
}

/// Rotate `dir` by a scattering polar angle whose cosine is `mu` and
/// a uniform azimuthal angle. Standard Monte Carlo deflection:
/// preserve norm, rotate around the normal to `dir`.
pub fn deflect(dir: Vec3, mu: f64, rng: &mut Rng) -> Vec3 {
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Rotate using the "rotate around normal" formula. If the current
    // direction is near the z-axis use a simpler branch to avoid
    // division by a small sin_theta_dir.
    let u = dir.x;
    let v = dir.y;
    let w = dir.z;
    let sin_theta_dir = (1.0 - w * w).max(0.0).sqrt();
    if sin_theta_dir < 1.0e-8 {
        // Dir is ± z; use simple form.
        let sgn = if w >= 0.0 { 1.0 } else { -1.0 };
        Vec3::new(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            sgn * mu,
        )
    } else {
        let inv = 1.0 / sin_theta_dir;
        Vec3::new(
            u * mu + sin_theta * (u * w * cos_phi - v * sin_phi) * inv,
            v * mu + sin_theta * (v * w * cos_phi + u * sin_phi) * inv,
            w * mu - sin_theta * sin_theta_dir * cos_phi,
        )
    }
}

#[cfg(test)]
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

    fn water() -> Option<PhotonMaterial> {
        Some(PhotonMaterial::new(vec![
            (2.0 * 3.3428e-2, load("H.h5")?),
            (1.0 * 3.3428e-2, load("O.h5")?),
        ]))
    }

    /// A deflected direction is still a unit vector.
    #[test]
    fn deflect_preserves_norm() {
        let mut rng = Rng::new(1, 1);
        for _ in 0..1_000 {
            let mu = 2.0 * rng.uniform() - 1.0;
            let d = Vec3::new(0.3, 0.4, 0.8660254).normalized();
            let out = deflect(d, mu, &mut rng);
            let n = (out.x * out.x + out.y * out.y + out.z * out.z).sqrt();
            assert!((n - 1.0).abs() < 1.0e-9, "|deflected| = {n}");
        }
    }

    /// For μ = 1 (no scattering) `deflect` is the identity.
    #[test]
    fn deflect_by_mu_one_is_identity() {
        let mut rng = Rng::new(2, 1);
        let d = Vec3::new(0.1, 0.7, 0.7).normalized();
        let out = deflect(d, 1.0, &mut rng);
        assert!((out.x - d.x).abs() < 1e-9);
        assert!((out.y - d.y).abs() < 1e-9);
        assert!((out.z - d.z).abs() < 1e-9);
    }

    /// For μ = −1 (back-scatter) `deflect` reverses direction.
    #[test]
    fn deflect_by_mu_minus_one_reverses() {
        let mut rng = Rng::new(3, 1);
        let d = Vec3::new(0.1, 0.7, 0.7).normalized();
        let out = deflect(d, -1.0, &mut rng);
        assert!((out.x + d.x).abs() < 1e-9);
        assert!((out.y + d.y).abs() < 1e-9);
        assert!((out.z + d.z).abs() < 1e-9);
    }

    /// `deflect` along the z axis produces a direction whose z
    /// component equals `mu`.
    #[test]
    fn deflect_z_axis_yields_mu_as_cos_theta() {
        let mut rng = Rng::new(4, 1);
        let d = Vec3::new(0.0, 0.0, 1.0);
        for _ in 0..100 {
            let mu = 2.0 * rng.uniform() - 1.0;
            let out = deflect(d, mu, &mut rng);
            assert!(
                (out.z - mu).abs() < 1e-9,
                "z component {} != mu {}",
                out.z,
                mu
            );
        }
    }

    /// Energy conservation on full histories in an infinite medium
    /// (no escape): total deposited equals source energy within ~1 %
    /// (valence-binding accounting loss in the relaxation cascade).
    #[test]
    fn infinite_medium_energy_conservation() {
        let Some(water) = water() else {
            eprintln!("skipping: H.h5 or O.h5 not present");
            return;
        };
        let mut rng = Rng::new(42, 1);
        for _ in 0..200 {
            let source_e = 1.0e6;
            let r = transport_history(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &water,
                |_pos| true, // infinite medium
                1_000.0,     // 1 keV cutoff
                &mut rng,
            );
            assert_eq!(r.energy_escaped, 0.0);
            let rel_err = (r.energy_deposited - source_e).abs() / source_e;
            assert!(
                rel_err < 1.0e-2,
                "energy violation: deposited {} vs source {} (rel err {})",
                r.energy_deposited,
                source_e,
                rel_err
            );
        }
    }

    /// A photon hitting a zero-thickness slab must escape with all
    /// its energy intact.
    #[test]
    fn zero_thickness_means_full_escape() {
        let Some(water) = water() else {
            eprintln!("skipping");
            return;
        };
        let mut rng = Rng::new(1, 1);
        let source_e = 1.0e6;
        let r = transport_history(
            Vec3::new(0.0, 0.0, 1e-12), // just outside the slab
            Vec3::new(0.0, 0.0, 1.0),
            source_e,
            &water,
            |pos| pos.z >= 0.0 && pos.z <= 0.0, // zero-thickness
            1_000.0,
            &mut rng,
        );
        assert_eq!(r.energy_deposited, 0.0);
        assert!((r.energy_escaped - source_e).abs() < 1e-12);
    }

    /// Transport through a thick slab absorbs most of the energy,
    /// with a fraction backscattered out the entry face.
    /// At 100 keV in water, Compton dominates (~97 % of macro XS),
    /// so photons multi-scatter before degrading into the
    /// photoelectric-dominant regime. Some photons reflect out the
    /// entry face before absorption. A realistic pass threshold is
    /// > 60 % absorbed for 1 m of water, verifying that the physics
    /// is closing on the expected backscatter-modulated absorption.
    #[test]
    fn thick_slab_absorbs_majority_of_energy() {
        let Some(water) = water() else {
            eprintln!("skipping");
            return;
        };
        let mut rng = Rng::new(10, 1);
        let source_e = 1.0e5; // 100 keV
        let n = 500;
        let mut total_dep = 0.0;
        for _ in 0..n {
            let r = transport_history(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &water,
                |pos| pos.z >= 0.0 && pos.z <= 100.0, // 1 m slab
                1_000.0,
                &mut rng,
            );
            total_dep += r.energy_deposited;
        }
        let avg_dep = total_dep / n as f64;
        assert!(
            avg_dep / source_e > 0.6,
            "thick slab absorbed only {:.3} of source energy",
            avg_dep / source_e
        );
    }
}
