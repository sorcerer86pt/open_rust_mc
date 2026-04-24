//! Track-integrated CSDA electron deposition on a CSG geometry, with
//! optional Highland-Gaussian multiple scattering.
//!
//! Replaces the point-deposit CSDA midrange approximation used by the
//! photon transport driver. The electron is launched from its birth
//! site with an energy budget equal to the Katz-Penfold range in
//! g/cm² (material-independent to first order); it then streams
//! through the CSG sub-step by sub-step, reflecting off reflective
//! BCs, streaming freely through void cells, and depositing energy in
//! dense cells in proportion to the mass-traversed (ρ · ds) fraction
//! of the total budget.
//!
//! Deposition model
//! ----------------
//! Range-budget CSDA:
//!
//!   - Total range: R_KP(E₀) [g/cm²], constant along the track.
//!   - In a cell with density ρ: each cm traversed spends ρ [g/cm²]
//!     of budget and deposits `(ρ · ds / R_KP) · E₀` eV of energy.
//!   - In a void (ρ = 0): no deposit, no budget cost — the electron
//!     drifts to the next surface.
//!
//! Uniform dE/dx in range-space matches the CSDA definition of range
//! (range = ∫ dE / (dE/ds)). Material transitions are handled exactly
//! because only the local ρ governs spending.
//!
//! Multiple scattering (optional)
//! ------------------------------
//! Each CSDA sub-step is also a multiple-scattering step. At the end
//! of each sub-step, the direction is perturbed by a Gaussian polar
//! angle drawn from the Highland-1975 (PDG review) width:
//!
//!     θ_0 = (13.6 MeV / β·p·c) × √(s/X₀) × (1 + 0.038 ln(s/X₀))
//!
//! The azimuth φ is uniform in [0, 2π). Radiation length X₀ is
//! computed per material via Tsai's formula with Bragg additivity on
//! the element list. This is a thin-layer Gaussian approximation to
//! Molière/Goudsmit-Saunderson; it underestimates the non-Gaussian
//! single-large-angle tails (<5 % of the distribution). Adequate for
//! reactor γ-heating where electrons stop within a few cell-widths.

use crate::geometry::cell::CellFill;
use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{self, Cell, Surface, Vec3};
use crate::photon::material::PhotonMaterial;
use crate::transport::rng::Rng;

/// Standard atomic weights for common elements. Covers the PWR
/// materials (H, O, Zr, U) exactly; falls back to `A ≈ 2·Z` for the
/// rest. Used only for radiation-length computation in multiple
/// scattering — a ~5 % error on A shifts X₀ by ~5 %, well within the
/// Highland-Gaussian approximation's own error band.
fn standard_atomic_weight(z: u32) -> f64 {
    match z {
        1 => 1.008,
        2 => 4.003,
        6 => 12.011,
        7 => 14.007,
        8 => 15.999,
        13 => 26.982,
        14 => 28.085,
        26 => 55.845,
        40 => 91.224,  // Zr
        50 => 118.710,
        82 => 207.2,   // Pb
        92 => 238.029, // U
        _ => 2.0 * z as f64,
    }
}

/// Tsai 1974 radiation length for a single element in g/cm² (PDG
/// 2022 review eq. 27.23). Uses the Z = 1..4 special values for L_rad
/// and L_rad' tabulated in the PDG review table 33.1 to match H and
/// Be correctly; for Z ≥ 5 the log-based approximations are accurate
/// to ~2 %.
fn x0_element_g_per_cm2(z: u32) -> f64 {
    if z == 0 {
        return f64::INFINITY;
    }
    let z_f = z as f64;
    let a = standard_atomic_weight(z);
    let (l_rad, l_rad_prime) = match z {
        1 => (5.31, 6.144),
        2 => (4.79, 5.621),
        3 => (4.74, 5.805),
        4 => (4.71, 5.924),
        _ => {
            let z13 = z_f.cbrt();
            let z23 = z13 * z13;
            ((184.15 / z13).ln(), (1194.0 / z23).ln())
        }
    };
    // f(Z) Coulomb correction, small (≤ 0.5 %) for Z ≤ 92.
    let alpha_z = z_f / 137.036;
    let alpha_z_sq = alpha_z * alpha_z;
    let f_z = alpha_z_sq
        * (1.0 / (1.0 + alpha_z_sq)
            + 0.20206
            - 0.0369 * alpha_z_sq
            + 0.0083 * alpha_z_sq * alpha_z_sq
            - 0.002 * alpha_z_sq.powi(3));
    let inv_x0 = 1.396e-3 / a * (z_f * z_f * (l_rad - f_z) + z_f * l_rad_prime);
    1.0 / inv_x0
}

/// Radiation length of a `PhotonMaterial` in cm, via Bragg additivity
/// of the per-element X₀ and the material's mass density. Returns
/// `f64::INFINITY` if the density is unset (kerma mode).
pub fn radiation_length_cm(material: &PhotonMaterial) -> f64 {
    if material.density_g_per_cm3 <= 0.0 {
        return f64::INFINITY;
    }
    // 1/X₀_mix = Σ (w_i / X₀_i), mass fractions w_i.
    let mut sum_nz_a = 0.0; // Σ n_i · A_i  (∝ mass content)
    let mut sum_weighted_inv_x0 = 0.0; // Σ n_i · A_i / X₀_i
    for (n, elem) in &material.entries {
        let z = elem.z;
        let a = standard_atomic_weight(z);
        let x0_i = x0_element_g_per_cm2(z);
        sum_nz_a += n * a;
        sum_weighted_inv_x0 += n * a / x0_i;
    }
    if sum_nz_a <= 0.0 {
        return f64::INFINITY;
    }
    let inv_x0_mix_g_per_cm2 = sum_weighted_inv_x0 / sum_nz_a;
    // Convert g/cm² → cm via mass density.
    1.0 / (inv_x0_mix_g_per_cm2 * material.density_g_per_cm3)
}

/// Highland 1975 Gaussian RMS multiple-scattering angle, in radians,
/// after straight-line path length `s_cm` in a material of radiation
/// length `x0_cm` for an electron of kinetic energy `t_ev`.
///
/// θ₀ = (13.6 MeV / βp·c) · √(s/X₀) · (1 + 0.038 ln(s/X₀))
///
/// The √(s/X₀) factor gives the usual diffusion-like growth with step;
/// the ln(s/X₀) tail is a small (≤ 30 %) correction that matters at
/// large s/X₀.
fn highland_theta0(t_ev: f64, s_cm: f64, x0_cm: f64) -> f64 {
    if s_cm <= 0.0 || !x0_cm.is_finite() || x0_cm <= 0.0 {
        return 0.0;
    }
    // βpc in eV. T(T+2m)/(T+m) for T, m in eV.
    const M_E_EV: f64 = 510_998.95;
    let bpc_ev = t_ev * (t_ev + 2.0 * M_E_EV) / (t_ev + M_E_EV);
    if bpc_ev <= 0.0 {
        return 0.0;
    }
    let bpc_mev = bpc_ev * 1.0e-6;
    let ratio = s_cm / x0_cm;
    let sqrt_ratio = ratio.sqrt();
    let log_term = (1.0 + 0.038 * ratio.ln()).max(0.0);
    (13.6 / bpc_mev) * sqrt_ratio * log_term
}

/// Sample a Gaussian polar angle `θ` with standard deviation `θ₀`
/// using the Box-Muller transform. Clamped to `[0, π]`.
fn sample_gaussian_theta(theta0: f64, rng: &mut Rng) -> f64 {
    if theta0 <= 0.0 {
        return 0.0;
    }
    // Box-Muller: draw two uniforms, produce one Gaussian.
    let u1 = rng.uniform().max(1.0e-300);
    let u2 = rng.uniform();
    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (theta0 * g.abs()).clamp(0.0, std::f64::consts::PI)
}

/// Rotate `dir` by polar angle `theta` (from dir) and azimuth `phi`,
/// returning the new unit direction. Replicates
/// `photon::transport::deflect` logic without pulling rng dependency.
fn rotate_direction(dir: Vec3, theta: f64, phi: f64) -> Vec3 {
    let mu = theta.cos();
    let sin_t = theta.sin();
    let cos_p = phi.cos();
    let sin_p = phi.sin();
    let u = dir.x;
    let v = dir.y;
    let w = dir.z;
    let sin_t_dir = (1.0 - w * w).max(0.0).sqrt();
    if sin_t_dir < 1.0e-8 {
        let sgn = if w >= 0.0 { 1.0 } else { -1.0 };
        Vec3::new(sin_t * cos_p, sin_t * sin_p, sgn * mu)
    } else {
        let inv = 1.0 / sin_t_dir;
        Vec3::new(
            u * mu + sin_t * (u * w * cos_p - v * sin_p) * inv,
            v * mu + sin_t * (v * w * cos_p + u * sin_p) * inv,
            w * mu - sin_t * sin_t_dir * cos_p,
        )
    }
}

/// Katz-Penfold CSDA electron range in g/cm², valid 10 keV-20 MeV.
///
/// Reference: Katz & Penfold, Rev. Mod. Phys. 24, 28 (1952).
/// R[g/cm²] = 0.412 · E^(1.265 − 0.0954 ln E) for E < 2.5 MeV,
/// R[g/cm²] = 0.530 · E − 0.106           for E ≥ 2.5 MeV.
/// Returns 0 below 10 keV (kerma regime — deposit locally).
#[inline]
pub fn katz_penfold_range_g_per_cm2(e_kin_ev: f64) -> f64 {
    if e_kin_ev <= 1.0e4 {
        return 0.0;
    }
    let e_mev = e_kin_ev * 1.0e-6;
    if e_mev < 2.5 {
        let exp = 1.265 - 0.0954 * e_mev.ln();
        0.412 * e_mev.powf(exp)
    } else {
        0.530 * e_mev - 0.106
    }
}

/// Bethe-Bloch-style instantaneous stopping power dE/d(ρs), in eV per
/// g/cm², computed analytically from the Katz-Penfold range formula.
///
/// Derivation: given R_KP(E) = 0.412 · E^α(E) with α(E) = 1.265 −
/// 0.0954 ln(E/MeV), differentiating gives
///   dR/dE = R_KP · (1.265 − 0.1908 ln E_MeV) / E
/// so dE/dR = E / [R_KP · (1.265 − 0.1908 ln E_MeV)].
///
/// This is NOT the true Bethe-Bloch collision stopping power (which
/// requires per-material I-value, Sternheimer density effect, etc.),
/// but it does recover the main Bragg-like shape — dE/ds rises as the
/// electron slows — while remaining self-consistent with the range
/// formula used elsewhere in this file. For E > 2.5 MeV the Katz-
/// Penfold range is linear, so dR/dE = 0.530 and dE/dR = 1/0.530.
#[inline]
pub fn instantaneous_de_per_dr(e_kin_ev: f64) -> f64 {
    let r_kp = katz_penfold_range_g_per_cm2(e_kin_ev);
    if r_kp <= 0.0 {
        return 0.0;
    }
    let e_mev = e_kin_ev * 1.0e-6;
    if e_mev < 2.5 {
        let factor = 1.265 - 0.1908 * e_mev.ln();
        if factor <= 0.0 {
            // Numerically degenerate at extreme E; fall back to the
            // uniform-budget slope.
            e_kin_ev / r_kp
        } else {
            e_kin_ev / (r_kp * factor)
        }
    } else {
        // R = 0.530 E − 0.106 (eV-units: R_gcm2 = 5.30e-7 E_eV − 0.106)
        // so dE/dR = 1/0.530 MeV per g/cm² = 1.887e6 eV per g/cm²
        1.887e6
    }
}

fn cell_density(cell: &Cell, materials: &[Option<PhotonMaterial>]) -> f64 {
    match cell.fill {
        CellFill::Material(m) => materials
            .get(m as usize)
            .and_then(|o| o.as_ref())
            .map(|mat| mat.density_g_per_cm3)
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Track-integrate a single electron through the CSG. Starts at
/// `birth_pos` with direction `birth_dir` and kinetic energy
/// `e_kin_ev`, stepping from cell to cell until the range budget is
/// exhausted or the track leaves through a vacuum boundary. Per-cell
/// deposition is accumulated into `per_cell_deposit[cell_idx]`.
///
/// `start_cell_idx` must be the cell containing `birth_pos` (the
/// caller already knows this from the photon's collision site). The
/// birth energy below 10 keV is deposited locally without transport.
pub fn track_integrate_electron_csg(
    birth_pos: Vec3,
    birth_dir: Vec3,
    e_kin_ev: f64,
    start_cell_idx: usize,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Option<PhotonMaterial>],
    per_cell_deposit: &mut [f64],
) {
    if e_kin_ev <= 0.0 || start_cell_idx >= cells.len() {
        return;
    }

    let r_total_g_per_cm2 = katz_penfold_range_g_per_cm2(e_kin_ev);
    if r_total_g_per_cm2 <= 0.0 {
        // Sub-10 keV electron: negligible range, deposit locally.
        per_cell_deposit[start_cell_idx] += e_kin_ev;
        return;
    }

    let mut pos = birth_pos;
    let mut dir = birth_dir;
    let mut cell_idx = start_cell_idx;
    let mut remaining_budget = r_total_g_per_cm2;
    let mut remaining_energy = e_kin_ev;

    // Safety cap: a pathological reflective BC with a zero-thickness
    // cell could in principle cause an infinite loop. 4096 steps is
    // orders of magnitude more than any physical electron history
    // needs (typical is < 10 cell transitions).
    const MAX_STEPS: u32 = 4096;

    for _ in 0..MAX_STEPS {
        if remaining_budget <= 0.0 || remaining_energy <= 0.0 {
            return;
        }

        let rho = cell_density(&cells[cell_idx], materials);
        let trace = geometry::ray::trace_step(pos, dir, cell_idx, surfaces, cells);

        if rho <= 0.0 {
            // Void / zero-density: stream to the next surface.
            let Some(hit) = trace else {
                // No bounding surface — dump remainder here and stop.
                per_cell_deposit[cell_idx] += remaining_energy;
                return;
            };
            if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx) {
                // Vacuum leak: electron exits the model carrying its
                // residual energy. Without an "escape" tally for
                // electrons, we drop it on the floor — reflective
                // outer BCs (the intended use) never take this path.
                return;
            }
            continue;
        }

        // Dense region: compute how far the budget lets us travel.
        let ds_budget_cm = remaining_budget / rho;

        let ds_to_surface = trace.map(|h| h.distance).unwrap_or(f64::INFINITY);

        if ds_budget_cm < ds_to_surface {
            // Electron stops inside the current cell — deposit all
            // residual energy here.
            per_cell_deposit[cell_idx] += remaining_energy;
            return;
        }

        // Otherwise traverse to the next surface, spending budget at
        // rate ρ and depositing energy proportionally.
        let Some(hit) = trace else {
            per_cell_deposit[cell_idx] += remaining_energy;
            return;
        };
        let cost = ds_to_surface * rho;
        let frac = (cost / r_total_g_per_cm2).min(remaining_energy / e_kin_ev);
        let e_dep = frac * e_kin_ev;
        per_cell_deposit[cell_idx] += e_dep;
        remaining_budget -= cost;
        remaining_energy -= e_dep;

        if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx) {
            return;
        }
    }

    // Step cap exceeded: deposit anything left at the current pos.
    if remaining_energy > 0.0 {
        per_cell_deposit[cell_idx] += remaining_energy;
    }
}

/// Track-integrate a single electron through the CSG with Highland
/// multiple-scattering deflections between straight-line sub-steps.
///
/// Arguments mirror [`track_integrate_electron_csg`] with three extra:
///   - `x0_per_cell` — radiation length (cm) indexed by cell index;
///     pass `f64::INFINITY` for void cells so MS is skipped.
///   - `ms_step_cm` — maximum straight-line step between deflections.
///     Typical values: 10 % of the smallest in-material CSDA range
///     in the problem. For the PWR pin cell, 0.005 cm is a good choice
///     (electrons in UO₂ have R ≈ 0.04 cm).
///   - `rng` — PRNG for scattering-angle samples.
#[allow(clippy::too_many_arguments)]
pub fn track_integrate_electron_csg_with_ms(
    birth_pos: Vec3,
    birth_dir: Vec3,
    e_kin_ev: f64,
    start_cell_idx: usize,
    surfaces: &[Surface],
    cells: &[Cell],
    materials: &[Option<PhotonMaterial>],
    x0_per_cell: &[f64],
    ms_step_cm: f64,
    rng: &mut Rng,
    per_cell_deposit: &mut [f64],
) {
    if e_kin_ev <= 0.0 || start_cell_idx >= cells.len() {
        return;
    }

    if katz_penfold_range_g_per_cm2(e_kin_ev) <= 0.0 {
        per_cell_deposit[start_cell_idx] += e_kin_ev;
        return;
    }

    let mut pos = birth_pos;
    let mut dir = birth_dir;
    let mut cell_idx = start_cell_idx;
    // Track the electron's current kinetic energy directly. Each
    // sub-step deposits `cost × dE/dR(E_current)` locally, where
    // cost = ρ · ds. The electron stops when E drops below the
    // 10 keV Katz-Penfold validity floor — any residual there is
    // deposited in one chunk (standard MC cutoff handling).
    let mut e_current = e_kin_ev;
    const E_CUTOFF_EV: f64 = 1.0e4;

    // Sub-step cap: ~4× the no-MS walker's bound since each iteration
    // advances by at most `ms_step_cm` rather than going straight to
    // the next surface.
    const MAX_STEPS: u32 = 65_536;

    for _ in 0..MAX_STEPS {
        if e_current <= E_CUTOFF_EV {
            per_cell_deposit[cell_idx] += e_current;
            return;
        }

        let rho = cell_density(&cells[cell_idx], materials);
        let trace = geometry::ray::trace_step(pos, dir, cell_idx, surfaces, cells);
        let ds_to_surface = trace.map(|h| h.distance).unwrap_or(f64::INFINITY);

        if rho <= 0.0 {
            // Void region: stream through, no MS, no deposit.
            let Some(hit) = trace else {
                per_cell_deposit[cell_idx] += e_current;
                return;
            };
            if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx) {
                return;
            }
            continue;
        }

        // Limit the step to: MS step, or the remaining CSDA range at
        // current energy, or the distance to the next surface.
        let r_current = katz_penfold_range_g_per_cm2(e_current);
        let ds_range_cm = r_current / rho;
        let ds_step = ms_step_cm.min(ds_range_cm).min(ds_to_surface);

        // Deposit Bethe-Bloch-style — dE = (ρ·ds) · (dE/dR)_{E_current}.
        // Capped at the electron's remaining energy so we never
        // deposit more than we carry.
        let cost = ds_step * rho;
        let de_per_dr = instantaneous_de_per_dr(e_current);
        let e_lost = (cost * de_per_dr).min(e_current);
        per_cell_deposit[cell_idx] += e_lost;
        e_current -= e_lost;

        // Three cases for what ended the step.
        if ds_step >= ds_range_cm - 1.0e-12 || e_current <= E_CUTOFF_EV {
            // Range exhausted or energy below floor — dump the rest
            // here (already counted if e_lost = e_current; e_current
            // is now 0 or near-zero).
            if e_current > 0.0 {
                per_cell_deposit[cell_idx] += e_current;
            }
            return;
        } else if ds_step >= ds_to_surface - 1.0e-12 && trace.is_some() {
            // Boundary hit — handle BC, no MS at interface.
            if let Some(hit) = trace {
                if !handle_boundary(hit, surfaces, &mut pos, &mut dir, &mut cell_idx) {
                    return;
                }
            }
        } else {
            // Interior sub-step: advance, then apply MS deflection.
            pos = pos + dir * ds_step;
            let x0 = x0_per_cell.get(cell_idx).copied().unwrap_or(f64::INFINITY);
            let theta0 = highland_theta0(e_current, ds_step, x0);
            let theta = sample_gaussian_theta(theta0, rng);
            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
            dir = rotate_direction(dir, theta, phi);
        }
    }
    if e_current > 0.0 {
        per_cell_deposit[cell_idx] += e_current;
    }
}

/// Handle a surface hit during electron streaming. Returns `false` if
/// the track terminates (vacuum leak or unresolved neighbour).
fn handle_boundary(
    hit: geometry::RayHit,
    surfaces: &[Surface],
    pos: &mut Vec3,
    dir: &mut Vec3,
    cell_idx: &mut usize,
) -> bool {
    let bc = surfaces[hit.surface_idx].boundary_condition();
    match bc {
        BoundaryCondition::Vacuum => false,
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
                None => false,
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, Cell, CellFill, CellId};
    use crate::geometry::surface::{BoundaryCondition, Surface};
    use crate::geometry::{Aabb, Vec3};
    use crate::photon::data::PhotonElement;
    use crate::photon::material::PhotonMaterial;
    use std::path::PathBuf;

    fn load_h() -> Option<PhotonElement> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let p = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon/H.h5");
        if p.exists() { Some(PhotonElement::from_hdf5(&p).unwrap()) } else { None }
    }

    #[test]
    fn range_decreases_below_threshold() {
        assert_eq!(katz_penfold_range_g_per_cm2(5_000.0), 0.0);
        assert!(katz_penfold_range_g_per_cm2(100_000.0) > 0.0);
    }

    #[test]
    fn x0_element_matches_pdg_references() {
        // PDG 2022 reference values (g/cm²):
        //   H:   63.04
        //   C:   42.70
        //   Pb:   6.37
        //   U:    6.00
        let x0_h = x0_element_g_per_cm2(1);
        let x0_c = x0_element_g_per_cm2(6);
        let x0_pb = x0_element_g_per_cm2(82);
        let x0_u = x0_element_g_per_cm2(92);
        assert!((x0_h - 63.04).abs() / 63.04 < 0.05, "X0(H) = {x0_h}");
        assert!((x0_c - 42.70).abs() / 42.70 < 0.05, "X0(C) = {x0_c}");
        assert!((x0_pb - 6.37).abs() / 6.37 < 0.05, "X0(Pb) = {x0_pb}");
        assert!((x0_u - 6.00).abs() / 6.00 < 0.05, "X0(U) = {x0_u}");
    }

    #[test]
    fn instantaneous_de_per_dr_grows_as_electron_slows() {
        // Bragg-like: dE/dR should increase as E decreases (electron
        // deposits energy faster per unit path when slow).
        let s_1mev = instantaneous_de_per_dr(1.0e6);
        let s_100kev = instantaneous_de_per_dr(1.0e5);
        let s_50kev = instantaneous_de_per_dr(5.0e4);
        assert!(s_100kev > s_1mev, "dE/dR(100 keV) > dE/dR(1 MeV): {s_100kev} vs {s_1mev}");
        assert!(s_50kev > s_100kev, "dE/dR(50 keV) > dE/dR(100 keV)");
        // Sanity: 1 MeV electron in solid (ρ=1 g/cm³, step 0.01 cm →
        // cost 0.01 g/cm², dE ≈ 0.01 × 1.4 MeV = 14 keV). Ballpark.
        let de = s_1mev * 0.01;
        assert!(de > 5_000.0 && de < 30_000.0, "dE over 0.01 g/cm² = {de} eV");
    }

    #[test]
    fn highland_theta_grows_with_path_length() {
        // X0 = 1 cm (high-Z), electron T = 1 MeV
        let theta_small = highland_theta0(1.0e6, 0.001, 1.0);
        let theta_big = highland_theta0(1.0e6, 0.01, 1.0);
        assert!(theta_big > theta_small);
        // Relativistic 1 MeV electron, thick target: significant MS.
        assert!(theta_small > 0.0 && theta_small < 1.0);
    }

    #[test]
    fn range_matches_known_values() {
        // At 1 MeV the formula gives ≈ 0.412·1^1.265 = 0.412 g/cm².
        // NIST ESTAR for water at 1 MeV is 0.437 g/cm² — 6 % off, OK
        // for Katz-Penfold.
        let r = katz_penfold_range_g_per_cm2(1.0e6);
        assert!((r - 0.412).abs() < 0.01, "R(1 MeV) = {r}");
        // At 5 MeV: 0.530 * 5 - 0.106 = 2.544 g/cm².
        let r = katz_penfold_range_g_per_cm2(5.0e6);
        assert!((r - 2.544).abs() < 0.01, "R(5 MeV) = {r}");
    }

    /// Single-cell slab: electron deposits 100 % of its energy in the
    /// host cell since its range is much smaller than the slab half
    /// width.
    #[test]
    fn single_cell_deposits_all_energy() {
        let Some(h) = load_h() else { return; };
        let water = PhotonMaterial::new(vec![(2.0 * 3.343e-2, h)]).with_density(1.0);
        // A 10 cm cube bounded by 6 reflective planes.
        let surfaces = vec![
            Surface::PlaneX { x0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneX { x0:  5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0:  5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0:  5.0, bc: BoundaryCondition::Reflective },
        ];
        let cells = vec![
            Cell::new(
                CellId(0),
                cell::intersect_all(vec![
                    cell::outside(0), cell::inside(1),
                    cell::outside(2), cell::inside(3),
                    cell::outside(4), cell::inside(5),
                ]),
                CellFill::Material(0),
            )
            .with_aabb(Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0))),
        ];
        let materials = vec![Some(water)];
        let mut deposit = vec![0.0_f64];

        track_integrate_electron_csg(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
            &surfaces, &cells, &materials,
            &mut deposit,
        );

        assert!((deposit[0] - 1.0e6).abs() < 1.0, "deposit[0] = {}", deposit[0]);
    }

    /// Two-cell slab separated at x=0: one dense (water), one void
    /// (vacuum). Electron starts in void and streams into the dense
    /// cell, where it deposits all its energy.
    #[test]
    fn void_cell_transits_without_loss() {
        let Some(h) = load_h() else { return; };
        let water = PhotonMaterial::new(vec![(2.0 * 3.343e-2, h)]).with_density(1.0);
        let surfaces = vec![
            Surface::PlaneX { x0:  0.0, bc: BoundaryCondition::Transmission },
            Surface::PlaneX { x0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneX { x0:  5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0:  5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: -5.0, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0:  5.0, bc: BoundaryCondition::Reflective },
        ];
        // Surface 0 is PlaneX at x=0; `inside(0)` is x < 0 (negative
        // half-space), `outside(0)` is x > 0. cell 0 on the left is
        // void; cell 1 on the right is water.
        let cells = vec![
            Cell::new(
                CellId(0),
                cell::intersect_all(vec![
                    cell::inside(0),
                    cell::outside(1), cell::inside(2),
                    cell::outside(3), cell::inside(4),
                    cell::outside(5), cell::inside(6),
                ]),
                CellFill::Void,
            )
            .with_aabb(Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(0.0, 5.0, 5.0))),
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![
                    cell::outside(0),
                    cell::outside(1), cell::inside(2),
                    cell::outside(3), cell::inside(4),
                    cell::outside(5), cell::inside(6),
                ]),
                CellFill::Material(0),
            )
            .with_aabb(Aabb::new(Vec3::new(0.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0))),
        ];
        let materials = vec![Some(water)];
        let mut deposit = vec![0.0_f64, 0.0_f64];

        // Start at x = -1 (in void), moving +x into water.
        track_integrate_electron_csg(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0e6,
            0,
            &surfaces, &cells, &materials,
            &mut deposit,
        );

        assert!(deposit[0] < 1.0, "void cell got {} eV (should be 0)", deposit[0]);
        assert!(
            (deposit[1] - 1.0e6).abs() < 1.0,
            "water cell got {} eV (should be 1 MeV)",
            deposit[1]
        );
    }
}
