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

use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{self, Cell, Surface, Vec3};
use crate::geometry::cell::CellFill;
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

// ── CSG-aware transport ───────────────────────────────────────────────────

/// Drive a full photon history through a CSG geometry.
///
/// This is the multi-material counterpart to [`transport_history`]. The
/// transport medium is described by:
///   - `surfaces` — all quadric surfaces in the problem (indexed by
///     `HalfSpace::surface_idx`), each carrying its own boundary
///     condition (Vacuum, Reflective, Transmission),
///   - `cells` — boolean half-space regions with a `CellFill`
///     (material index, void, or nested universe),
///   - `materials` — per-cell `PhotonMaterial`, indexed by the
///     `CellFill::Material` id. An entry of `None` marks a material id
///     that is void (handy for e.g. an explicitly named vacuum
///     material rather than `CellFill::Void`).
///
/// Behaviour mirrors the neutron transport loop in
/// `transport::simulate::transport_particle`:
///   - sample a free flight in the current cell's macroscopic total,
///   - trace to the next surface crossing; if closer than the sampled
///     collision distance, handle the BC (leak / reflect / enter next
///     cell); otherwise collide and sample a channel.
///
/// A source outside the modelled geometry is returned immediately as
/// fully escaped energy.
pub fn transport_history_csg(
    source_pos: Vec3,
    source_dir: Vec3,
    source_energy: f64,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Option<PhotonMaterial>],
    energy_cutoff_ev: f64,
    rng: &mut Rng,
) -> HistoryResult {
    let mut result = HistoryResult {
        energy_deposited: 0.0,
        energy_escaped: 0.0,
        n_collisions: 0,
        deposits: Vec::new(),
    };

    // Find starting cell; if the source is outside the model the
    // photon is born already leaked and its energy escapes.
    let Some(start_cell) = geometry::ray::find_cell(source_pos, surfaces, cells) else {
        result.energy_escaped += source_energy;
        return result;
    };

    // (position, direction, energy, cell_idx). Secondaries are
    // spawned at a collision site inside the current cell, so they
    // inherit that cell; we re-resolve only when we cannot trust the
    // inherited index (e.g. on the very first step).
    let mut bank: Vec<(Vec3, Vec3, f64, usize)> =
        vec![(source_pos, source_dir, source_energy, start_cell)];

    while let Some((pos, dir, energy, cell_idx)) = bank.pop() {
        transport_one_csg(
            pos,
            dir,
            energy,
            cell_idx,
            surfaces,
            cells,
            materials,
            energy_cutoff_ev,
            rng,
            &mut bank,
            &mut result,
        );
    }

    result
}

#[allow(clippy::too_many_arguments)]
fn transport_one_csg(
    start_pos: Vec3,
    start_dir: Vec3,
    start_energy: f64,
    start_cell: usize,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Option<PhotonMaterial>],
    energy_cutoff_ev: f64,
    rng: &mut Rng,
    bank: &mut Vec<(Vec3, Vec3, f64, usize)>,
    result: &mut HistoryResult,
) {
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut energy = start_energy;
    let mut cell_idx = start_cell;

    // Shared cap across collisions + surface crossings. A stuck photon
    // (pathological geometry, grazing reflection loop) is terminated
    // with its remaining energy deposited locally.
    const MAX_EVENTS_PER_TRACK: u32 = 100_000;
    let mut void_streaks = 0_u32;

    for _ in 0..MAX_EVENTS_PER_TRACK {
        if energy < energy_cutoff_ev {
            result.energy_deposited += energy;
            result.deposits.push((pos, energy));
            return;
        }

        let material: Option<&PhotonMaterial> = match cells[cell_idx].fill {
            CellFill::Material(m) => materials.get(m as usize).and_then(|o| o.as_ref()),
            CellFill::Void => None,
            CellFill::Universe(_) => {
                // Nested universes not yet supported on the photon path.
                result.energy_escaped += energy;
                return;
            }
        };

        // Distance to the next surface in the current cell, if any.
        let trace = geometry::ray::trace_step(pos, dir, cell_idx, surfaces, cells);

        let sigma_tot = material.map(|m| m.macro_total(energy)).unwrap_or(0.0);

        // Void / zero-XS region: stream to the next surface.
        if sigma_tot <= 0.0 {
            void_streaks += 1;
            if void_streaks > 100 {
                result.energy_escaped += energy;
                return;
            }
            let Some(hit) = trace else {
                result.energy_escaped += energy;
                return;
            };
            if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx, &mut energy, result) {
                return;
            }
            continue;
        }
        void_streaks = 0;

        let dist_collision = rng.exponential(sigma_tot);

        match trace {
            Some(hit) if hit.distance < dist_collision => {
                if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx, &mut energy, result) {
                    return;
                }
            }
            _ => {
                // Collision inside the cell.
                pos = pos + dir * dist_collision;
                let material = material.expect("sigma_tot > 0 implies material");
                result.n_collisions += 1;

                let channel = material.sample_channel(energy, rng.uniform());
                let elem_idx = material.sample_element(channel, energy, rng.uniform());
                let elem = &material.entries[elem_idx].1;

                match channel {
                    Channel::Coherent => {
                        let out = coherent_scatter(elem, energy, rng);
                        dir = deflect(dir, out.mu, rng);
                    }
                    Channel::Incoherent => {
                        let out = compton_scatter(elem, energy, rng);
                        result.energy_deposited += out.electron_kinetic;
                        result.deposits.push((pos, out.electron_kinetic));
                        energy = out.energy_out;
                        dir = deflect(dir, out.mu, rng);
                    }
                    Channel::Photoelectric => {
                        let out =
                            photoelectric_absorb(elem, energy, DEFAULT_PHOTON_CUTOFF_EV, rng);
                        result.energy_deposited += out.local_deposition;
                        result.deposits.push((pos, out.local_deposition));
                        for ep in out.fluorescence_photons {
                            let (dx, dy, dz) = rng.isotropic_direction();
                            bank.push((pos, Vec3::new(dx, dy, dz), ep, cell_idx));
                        }
                        return;
                    }
                    Channel::PairProductionNuclear | Channel::PairProductionElectron => {
                        if let Some(out) = pair_produce(energy, rng) {
                            result.energy_deposited += out.local_deposition();
                            result.deposits.push((pos, out.local_deposition()));
                            let (dx, dy, dz) = rng.isotropic_direction();
                            let ann_dir = Vec3::new(dx, dy, dz);
                            bank.push((pos, ann_dir, ANNIHILATION_ENERGY_EV, cell_idx));
                            bank.push((pos, -ann_dir, ANNIHILATION_ENERGY_EV, cell_idx));
                        } else {
                            result.energy_deposited += energy;
                            result.deposits.push((pos, energy));
                        }
                        return;
                    }
                }
            }
        }
    }
    // Event budget exhausted — deposit remaining energy locally.
    result.energy_deposited += energy;
    result.deposits.push((pos, energy));
}

/// Apply the boundary condition at a surface hit. Returns `false` when
/// the track terminates (leak or unresolved neighbour); `true` when the
/// photon should continue (reflected or entered a new cell).
#[inline]
fn handle_boundary(
    hit: crate::geometry::RayHit,
    surfaces: &[Surface],
    pos: &mut Vec3,
    dir: &mut Vec3,
    cell_idx: &mut usize,
    energy: &mut f64,
    result: &mut HistoryResult,
) -> bool {
    let bc = surfaces[hit.surface_idx].boundary_condition();
    match bc {
        BoundaryCondition::Vacuum => {
            *pos = *pos + *dir * hit.distance;
            result.energy_escaped += *energy;
            false
        }
        BoundaryCondition::Reflective => {
            *pos = *pos + *dir * hit.distance;
            let n = surfaces[hit.surface_idx].normal_at(*pos);
            let d = *dir;
            *dir = d - n * (2.0 * d.dot(n));
            true
        }
        BoundaryCondition::Transmission => {
            let nudge = (hit.distance * 1e-8).max(1e-8);
            *pos = *pos + *dir * (hit.distance + nudge);
            match hit.next_cell_idx {
                Some(next) => {
                    *cell_idx = next;
                    true
                }
                None => {
                    result.energy_escaped += *energy;
                    false
                }
            }
        }
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

    // ── CSG driver tests ──────────────────────────────────────────────

    use crate::geometry::cell::{self, CellFill, CellId};
    use crate::geometry::surface::BoundaryCondition;
    use crate::geometry::{Cell, Surface};

    /// Build a 1 m water slab (0 ≤ z ≤ 100 cm) in CSG form with two
    /// transmission planes bounded laterally by vacuum (no x/y bounds
    /// — the planes z=0 and z=100 are vacuum, everything else in the
    /// surrounding cell is void that streams to infinity). Returns
    /// `(surfaces, cells, materials)` ready for `transport_history_csg`.
    fn water_slab_csg(
        water: PhotonMaterial,
        thickness_cm: f64,
    ) -> (Vec<Surface>, Vec<Cell>, Vec<Option<PhotonMaterial>>) {
        let surfaces = vec![
            Surface::PlaneZ {
                z0: 0.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneZ {
                z0: thickness_cm,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            // Slab: 0 < z < thickness
            Cell::new(
                CellId(0),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(0),
            ),
            // Outside below z=0
            Cell::new(CellId(1), cell::inside(0), CellFill::Void),
            // Outside above z=thickness
            Cell::new(CellId(2), cell::outside(1), CellFill::Void),
        ];
        let materials = vec![Some(water)];
        (surfaces, cells, materials)
    }

    /// Thick CSG water slab should absorb roughly the same fraction
    /// as the closure-based driver (same geometry, same physics). The
    /// tolerance covers different random-number consumption patterns.
    #[test]
    fn csg_slab_matches_closure_slab_water() {
        let Some(water_a) = water() else {
            eprintln!("skipping");
            return;
        };
        let Some(water_b) = water() else {
            eprintln!("skipping");
            return;
        };

        let source_e = 1.0e5;
        let thickness = 100.0;
        let n = 400;

        // CSG version
        let (surfaces, cells, materials) = water_slab_csg(water_a, thickness);
        let mut rng_csg = Rng::new(123, 1);
        let mut dep_csg = 0.0;
        for _ in 0..n {
            let r = transport_history_csg(
                Vec3::new(0.0, 0.0, 1e-6),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &surfaces,
                &cells,
                &materials,
                1_000.0,
                &mut rng_csg,
            );
            dep_csg += r.energy_deposited;
        }

        // Closure-based version (same slab)
        let mut rng_cl = Rng::new(123, 1);
        let mut dep_cl = 0.0;
        for _ in 0..n {
            let r = transport_history(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &water_b,
                |p| p.z >= 0.0 && p.z <= thickness,
                1_000.0,
                &mut rng_cl,
            );
            dep_cl += r.energy_deposited;
        }

        let frac_csg = dep_csg / (n as f64 * source_e);
        let frac_cl = dep_cl / (n as f64 * source_e);
        // Both should be in the physically reasonable range.
        assert!(
            frac_csg > 0.6,
            "CSG slab absorbed fraction {frac_csg} too low"
        );
        // And agree to ~5 % (finite-sample noise dominates).
        assert!(
            (frac_csg - frac_cl).abs() < 0.05,
            "CSG frac {frac_csg} vs closure frac {frac_cl}"
        );
    }

    /// Two back-to-back slabs (water in 0..50, lead in 50..60) —
    /// lead is far more attenuating at 100 keV, so the total escape
    /// past z=60 must be much lower than a pure-water slab of the
    /// same total length. Exercises per-cell material lookup.
    #[test]
    fn csg_water_plus_lead_attenuates_more_than_water_only() {
        let Some(water_a) = water() else {
            eprintln!("skipping");
            return;
        };
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5");
            return;
        };
        let Some(water_b) = water() else {
            eprintln!("skipping");
            return;
        };

        // Lead: 11.34 g/cm³, A = 207.2 → 3.296e-2 atoms/(barn·cm)
        let lead_mat = PhotonMaterial::mono(3.296e-2, pb);

        let surfaces = vec![
            Surface::PlaneZ { z0: 0.0, bc: BoundaryCondition::Vacuum },
            Surface::PlaneZ { z0: 50.0, bc: BoundaryCondition::Transmission },
            Surface::PlaneZ { z0: 60.0, bc: BoundaryCondition::Vacuum },
        ];
        let cells = vec![
            Cell::new(
                CellId(0),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(0), // water
            ),
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![cell::outside(1), cell::inside(2)]),
                CellFill::Material(1), // lead
            ),
            Cell::new(CellId(2), cell::inside(0), CellFill::Void),
            Cell::new(CellId(3), cell::outside(2), CellFill::Void),
        ];
        let materials = vec![Some(water_a), Some(lead_mat)];

        let source_e = 1.0e5;
        let n = 400;
        let mut rng = Rng::new(7, 1);
        let mut esc_mixed = 0.0;
        for _ in 0..n {
            let r = transport_history_csg(
                Vec3::new(0.0, 0.0, 1e-6),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &surfaces,
                &cells,
                &materials,
                1_000.0,
                &mut rng,
            );
            esc_mixed += r.energy_escaped;
        }

        // Pure 60 cm water reference.
        let surfaces_w = vec![
            Surface::PlaneZ { z0: 0.0, bc: BoundaryCondition::Vacuum },
            Surface::PlaneZ { z0: 60.0, bc: BoundaryCondition::Vacuum },
        ];
        let cells_w = vec![
            Cell::new(
                CellId(0),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(0),
            ),
            Cell::new(CellId(1), cell::inside(0), CellFill::Void),
            Cell::new(CellId(2), cell::outside(1), CellFill::Void),
        ];
        let materials_w = vec![Some(water_b)];

        let mut rng_w = Rng::new(7, 1);
        let mut esc_water = 0.0;
        for _ in 0..n {
            let r = transport_history_csg(
                Vec3::new(0.0, 0.0, 1e-6),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &surfaces_w,
                &cells_w,
                &materials_w,
                1_000.0,
                &mut rng_w,
            );
            esc_water += r.energy_escaped;
        }

        assert!(
            esc_mixed < esc_water,
            "water+lead escape {esc_mixed} not less than water-only {esc_water}"
        );
    }

    /// A closed box with all Reflective boundaries is an infinite
    /// medium in disguise: no energy can escape, so the full source
    /// energy must be deposited (up to the binding-loss tolerance of
    /// the relaxation cascade).
    #[test]
    fn csg_reflective_box_conserves_energy() {
        let Some(water) = water() else {
            eprintln!("skipping");
            return;
        };
        let half = 50.0;
        let surfaces = vec![
            Surface::PlaneX { x0: -half, bc: BoundaryCondition::Reflective },
            Surface::PlaneX { x0: half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: -half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: -half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: half, bc: BoundaryCondition::Reflective },
        ];
        let cells = vec![Cell::new(
            CellId(0),
            cell::intersect_all(vec![
                cell::outside(0),
                cell::inside(1),
                cell::outside(2),
                cell::inside(3),
                cell::outside(4),
                cell::inside(5),
            ]),
            CellFill::Material(0),
        )];
        let materials = vec![Some(water)];

        let mut rng = Rng::new(99, 1);
        let source_e = 1.0e6;
        for _ in 0..50 {
            let r = transport_history_csg(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                source_e,
                &surfaces,
                &cells,
                &materials,
                1_000.0,
                &mut rng,
            );
            assert_eq!(r.energy_escaped, 0.0);
            let rel_err = (r.energy_deposited - source_e).abs() / source_e;
            assert!(
                rel_err < 1.0e-2,
                "reflective box violated conservation: deposited {} vs source {}",
                r.energy_deposited,
                source_e
            );
        }
    }

    /// A photon source outside the modelled geometry must be reported
    /// as fully escaped with no interaction.
    #[test]
    fn csg_source_outside_geometry_escapes() {
        let Some(water) = water() else {
            eprintln!("skipping");
            return;
        };
        let (surfaces, cells, materials) = water_slab_csg(water, 10.0);
        let mut rng = Rng::new(1, 1);
        let r = transport_history_csg(
            Vec3::new(0.0, 0.0, -5.0), // below the slab; cell 1 is void → streams to -∞
            Vec3::new(0.0, 0.0, -1.0),
            1.0e6,
            &surfaces,
            &cells,
            &materials,
            1_000.0,
            &mut rng,
        );
        assert_eq!(r.n_collisions, 0);
        assert!((r.energy_escaped - 1.0e6).abs() < 1e-9);
    }
}
