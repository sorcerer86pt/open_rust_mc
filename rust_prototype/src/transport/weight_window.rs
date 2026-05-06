//! Cartesian-mesh weight windows — forward application path.
//!
//! Per-voxel `(w_lower, w_upper)` thresholds drive splitting and
//! Russian roulette so particle weight stays in a controlled band as
//! particles move through space:
//!
//! - `w > w_upper` → split into `N = ceil(w / w_survive)` copies of
//!   weight `w / N`. The primary keeps `w / N`; the remaining `N-1`
//!   are pushed onto the per-history `pending` stack.
//! - `w < w_lower` → Russian roulette with survival probability
//!   `w / w_survive`. On survival, weight is restored to `w_survive`.
//! - `w_lower ≤ w ≤ w_upper` → no-op.
//!
//! `w_survive` per voxel is the geometric mean of the bounds —
//! `sqrt(w_lower · w_upper)` — which keeps the unbiased mean weight
//! consistent with the standard CADIS / FW-CADIS convention.
//!
//! This module ships the *forward application* only. Window
//! generation (CADIS, FW-CADIS, manual tuning) is a much bigger
//! piece of work that requires a deterministic adjoint solver and
//! is not in scope here.

use crate::geometry::{Aabb, Vec3};
use crate::transport::particle::Particle;
use crate::transport::rng::Rng;

/// A Cartesian voxel mesh of weight-window bounds.
#[derive(Debug, Clone)]
pub struct WeightWindow {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub n: [usize; 3],
    /// Lower bound per voxel (flattened, x-major). `0.0` means
    /// "voxel is outside the active window" — apply() short-circuits.
    pub lower: Vec<f64>,
    /// Upper bound per voxel. Must satisfy `upper[i] > lower[i]`
    /// when `lower[i] > 0`; otherwise that voxel is treated as
    /// inactive.
    pub upper: Vec<f64>,
    /// Maximum number of split copies emitted per application.
    /// Caps runaway splits when a particle suddenly enters a
    /// high-importance region with very small `w_upper`.
    pub max_split: u32,
}

impl WeightWindow {
    /// Generate a weight window from a per-voxel flux estimate
    /// (forward CADIS-lite / flux-bootstrap convention).
    ///
    /// `flux` is the active-batch mesh flux (Σ_active w·d per voxel)
    /// in row-major `[ix][iy][iz]` order matching `MeshFluxTally`.
    /// Voxels with low flux are under-sampled — there `w_target` is
    /// high so analog-weight particles roulette out and surviving
    /// particles carry more weight per visit. Voxels with high flux
    /// get a low `w_target` so analog particles split and sampling
    /// per particle goes up.
    ///
    /// Concretely:
    /// ```text
    ///   w_target_v = w_ref · φ_max / max(φ_v, φ_floor)
    ///   w_lower_v  = w_target_v / sqrt(ratio)
    ///   w_upper_v  = w_target_v * sqrt(ratio)
    /// ```
    /// `w_ref` is the analog reference weight (typically 1.0);
    /// `ratio` is the upper/lower bound ratio (typically 5–10);
    /// `phi_floor` is a relative floor on φ (e.g. 1e-3 of φ_max) to
    /// avoid pathological splits where flux is essentially zero.
    /// Voxels with `φ_v <= phi_floor·φ_max` are flagged inactive
    /// (lower = 0) so the apply path leaves particles alone there.
    pub fn from_flux(
        aabb: &Aabb,
        n: [usize; 3],
        flux: &[f64],
        w_ref: f64,
        ratio: f64,
        phi_floor: f64,
    ) -> Self {
        let n_vox = n[0].max(1) * n[1].max(1) * n[2].max(1);
        assert_eq!(
            flux.len(),
            n_vox,
            "flux length {} doesn't match mesh n_vox {}",
            flux.len(),
            n_vox
        );
        let origin = [aabb.min.x, aabb.min.y, aabb.min.z];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0].max(1) as f64,
            (aabb.max.y - aabb.min.y) / n[1].max(1) as f64,
            (aabb.max.z - aabb.min.z) / n[2].max(1) as f64,
        ];
        let phi_max = flux.iter().cloned().fold(0.0_f64, f64::max);
        let cutoff = phi_max * phi_floor.max(0.0);
        let sqrt_ratio = ratio.max(1.0).sqrt();
        let mut lower = vec![0.0; n_vox];
        let mut upper = vec![0.0; n_vox];
        for (i, &phi) in flux.iter().enumerate() {
            if phi <= cutoff || phi_max <= 0.0 {
                // Inactive voxel — apply() short-circuits on lo == 0.
                continue;
            }
            let w_target = w_ref * phi_max / phi;
            lower[i] = w_target / sqrt_ratio;
            upper[i] = w_target * sqrt_ratio;
        }
        Self {
            origin,
            spacing,
            n,
            lower,
            upper,
            max_split: 8,
        }
    }

    /// Build a window with uniform bounds across an AABB.
    pub fn uniform(aabb: &Aabb, n: [usize; 3], lower: f64, upper: f64) -> Self {
        let n_vox = n[0].max(1) * n[1].max(1) * n[2].max(1);
        let origin = [aabb.min.x, aabb.min.y, aabb.min.z];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0].max(1) as f64,
            (aabb.max.y - aabb.min.y) / n[1].max(1) as f64,
            (aabb.max.z - aabb.min.z) / n[2].max(1) as f64,
        ];
        Self {
            origin,
            spacing,
            n,
            lower: vec![lower; n_vox],
            upper: vec![upper; n_vox],
            max_split: 8,
        }
    }

    /// Linear voxel index from (ix, iy, iz). No bounds check.
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.n[1] + iy) * self.n[2] + iz
    }

    pub fn n_voxels(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    /// Linear voxel index for a world-frame position, or `None` if
    /// the position falls outside the mesh. Does **not** check
    /// whether the voxel is flagged active — see `lookup` for that.
    #[inline]
    pub fn voxel_index(&self, pos: Vec3) -> Option<usize> {
        let ix = ((pos.x - self.origin[0]) / self.spacing[0]).floor() as isize;
        let iy = ((pos.y - self.origin[1]) / self.spacing[1]).floor() as isize;
        let iz = ((pos.z - self.origin[2]) / self.spacing[2]).floor() as isize;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix as usize >= self.n[0]
            || iy as usize >= self.n[1]
            || iz as usize >= self.n[2]
        {
            return None;
        }
        Some(self.index(ix as usize, iy as usize, iz as usize))
    }

    /// Look up `(w_lower, w_upper)` at a world-frame position.
    /// Returns `None` when the position is outside the mesh or in a
    /// voxel that's flagged inactive (lower == 0).
    #[inline]
    pub fn lookup(&self, pos: Vec3) -> Option<(f64, f64)> {
        let ix = ((pos.x - self.origin[0]) / self.spacing[0]).floor() as isize;
        let iy = ((pos.y - self.origin[1]) / self.spacing[1]).floor() as isize;
        let iz = ((pos.z - self.origin[2]) / self.spacing[2]).floor() as isize;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix as usize >= self.n[0]
            || iy as usize >= self.n[1]
            || iz as usize >= self.n[2]
        {
            return None;
        }
        let idx = self.index(ix as usize, iy as usize, iz as usize);
        let lo = self.lower[idx];
        let hi = self.upper[idx];
        if lo > 0.0 && hi > lo {
            Some((lo, hi))
        } else {
            None
        }
    }
}

/// Apply the weight window at the particle's current position.
///
/// On split the primary is mutated in place to the per-copy weight
/// and the additional copies are appended to `pending`. On roulette
/// kill the particle's status is set to `Dead`.
pub fn apply(particle: &mut Particle, ww: &WeightWindow, rng: &mut Rng, pending: &mut Vec<Particle>) {
    if !particle.is_alive() {
        return;
    }
    let (lo, hi) = match ww.lookup(particle.pos) {
        Some(b) => b,
        None => return,
    };
    let w_survive = (lo * hi).sqrt();
    let w = particle.weight;
    if w > hi {
        let n_split = ((w / w_survive).ceil() as u32).clamp(2, ww.max_split);
        let new_w = w / n_split as f64;
        particle.weight = new_w;
        for _ in 0..(n_split - 1) {
            let mut copy = particle.clone();
            copy.weight = new_w;
            pending.push(copy);
        }
    } else if w < lo {
        let p_survive = (w / w_survive).clamp(0.0, 1.0);
        if rng.uniform() < p_survive {
            particle.weight = w_survive;
        } else {
            particle.kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::coord::{Coord, CoordStack};
    use crate::geometry::UniverseId;
    use smallvec::smallvec;

    fn make_particle(pos: Vec3, weight: f64) -> Particle {
        let stack: CoordStack = smallvec![Coord::root(UniverseId(0), 0)];
        let mut p = Particle::with_stack(pos, Vec3::new(1.0, 0.0, 0.0), 1e6, stack);
        p.weight = weight;
        p
    }

    #[test]
    fn weight_in_band_is_no_op() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ww = WeightWindow::uniform(&aabb, [2, 2, 2], 0.25, 1.0);
        let mut p = make_particle(Vec3::new(0.5, 0.5, 0.5), 0.5);
        let mut pending = Vec::new();
        let mut rng = Rng::new(1, 0);
        apply(&mut p, &ww, &mut rng, &mut pending);
        assert!(p.is_alive());
        assert!((p.weight - 0.5).abs() < 1e-12);
        assert!(pending.is_empty());
    }

    #[test]
    fn high_weight_splits_into_multiple_copies() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ww = WeightWindow::uniform(&aabb, [1, 1, 1], 0.25, 1.0);
        // w_survive = sqrt(0.25 * 1.0) = 0.5; weight 4.0 → ceil(4 / 0.5) = 8 copies.
        let mut p = make_particle(Vec3::new(0.5, 0.5, 0.5), 4.0);
        let mut pending = Vec::new();
        let mut rng = Rng::new(1, 0);
        apply(&mut p, &ww, &mut rng, &mut pending);
        // Capped by max_split = 8 → exactly 8 copies (1 primary + 7 pending).
        assert_eq!(pending.len(), 7);
        let new_w = 4.0 / 8.0;
        assert!((p.weight - new_w).abs() < 1e-12);
        for c in &pending {
            assert!((c.weight - new_w).abs() < 1e-12);
        }
    }

    #[test]
    fn low_weight_rouletted_to_w_survive_or_killed() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ww = WeightWindow::uniform(&aabb, [1, 1, 1], 0.25, 1.0);
        // w = 0.05, w_survive = 0.5, p_survive = 0.05 / 0.5 = 0.1.
        // Run many trials and check the survival rate ≈ 10%.
        let trials = 5000;
        let mut survived = 0;
        let mut total_weight = 0.0;
        for i in 0..trials {
            let mut p = make_particle(Vec3::new(0.5, 0.5, 0.5), 0.05);
            let mut pending = Vec::new();
            let mut rng = Rng::new(7, i as u64);
            apply(&mut p, &ww, &mut rng, &mut pending);
            assert!(pending.is_empty());
            if p.is_alive() {
                survived += 1;
                total_weight += p.weight;
                assert!((p.weight - 0.5).abs() < 1e-12);
            }
        }
        let rate = survived as f64 / trials as f64;
        assert!((rate - 0.1).abs() < 0.02, "rate={rate}");
        // Mean weight is preserved: trials × 0.05 ≈ survived × 0.5
        let expected_total = trials as f64 * 0.05;
        let rel_err = (total_weight - expected_total).abs() / expected_total;
        assert!(rel_err < 0.10, "weight conservation off: {rel_err}");
    }

    #[test]
    fn position_outside_mesh_is_no_op() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let ww = WeightWindow::uniform(&aabb, [2, 2, 2], 0.25, 1.0);
        let mut p = make_particle(Vec3::new(-5.0, 0.5, 0.5), 0.01);
        let mut pending = Vec::new();
        let mut rng = Rng::new(1, 0);
        apply(&mut p, &ww, &mut rng, &mut pending);
        assert!(p.is_alive());
        assert!((p.weight - 0.01).abs() < 1e-12);
    }

    #[test]
    fn from_flux_inverts_flux_into_target() {
        use crate::geometry::Aabb;
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        // Two voxels: voxel 0 has flux=1.0, voxel 1 has flux=10.0.
        // φ_max = 10, w_ref = 1, ratio = 4 → sqrt_ratio = 2.
        //   voxel 0: target = 1 * 10 / 1   = 10  → lower = 5,    upper = 20
        //   voxel 1: target = 1 * 10 / 10  = 1   → lower = 0.5,  upper = 2
        let ww = WeightWindow::from_flux(
            &aabb,
            [2, 1, 1],
            &[1.0, 10.0],
            1.0,
            4.0,
            1e-12,
        );
        assert!((ww.lower[0] - 5.0).abs() < 1e-9);
        assert!((ww.upper[0] - 20.0).abs() < 1e-9);
        assert!((ww.lower[1] - 0.5).abs() < 1e-9);
        assert!((ww.upper[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn from_flux_zeros_inactive_voxels_below_floor() {
        use crate::geometry::Aabb;
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        // φ_max = 100, floor = 1% → cutoff = 1.0.
        let flux = vec![0.5, 100.0, 50.0];
        let ww = WeightWindow::from_flux(&aabb, [3, 1, 1], &flux, 1.0, 5.0, 0.01);
        // Voxel 0 below cutoff → inactive (lower == 0).
        assert_eq!(ww.lower[0], 0.0);
        assert_eq!(ww.upper[0], 0.0);
        // Voxels 1 and 2 active.
        assert!(ww.lower[1] > 0.0);
        assert!(ww.lower[2] > 0.0);
        // High-flux voxel (1) gets the smallest target.
        assert!(ww.lower[1] < ww.lower[2]);
    }

    #[test]
    fn from_flux_round_trip_with_apply() {
        use crate::geometry::Aabb;
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        // Two voxels, voxel 0 has 10× the flux of voxel 1.
        // w_ref = 1, ratio = 4 → voxel 0 target = 1 (low flux side
        // would be 1 not 10 here — wait flux[0] = 10, flux[1] = 1).
        // Re-explain: voxel 0 (high flux) has target 1, lower 0.5,
        // upper 2. A particle at weight 1 sits in the band → no-op.
        // Voxel 1 (low flux) has target 10, lower 5, upper 20. A
        // particle at weight 1 is below w_lower = 5 → roulette.
        let ww = WeightWindow::from_flux(
            &aabb,
            [2, 1, 1],
            &[10.0, 1.0],
            1.0,
            4.0,
            1e-12,
        );
        let mut rng = Rng::new(42, 0);

        // Particle in voxel 0 — no-op.
        let mut p = make_particle(Vec3::new(0.5, 0.5, 0.5), 1.0);
        let mut pending = Vec::new();
        apply(&mut p, &ww, &mut rng, &mut pending);
        assert!(p.is_alive());
        assert!((p.weight - 1.0).abs() < 1e-12);
        assert!(pending.is_empty());

        // Particle in voxel 1 — should fire roulette (weight 1 < lower 5).
        // Repeat trials and check the survival rate.
        let trials = 5000;
        let mut survived = 0;
        for i in 0..trials {
            let mut p = make_particle(Vec3::new(1.5, 0.5, 0.5), 1.0);
            let mut pending = Vec::new();
            let mut rng = Rng::new(99, i as u64);
            apply(&mut p, &ww, &mut rng, &mut pending);
            assert!(pending.is_empty());
            if p.is_alive() {
                survived += 1;
                // sqrt(5*20) = 10 = w_survive
                assert!((p.weight - 10.0).abs() < 1e-9);
            }
        }
        // p_survive = w / w_survive = 1 / 10 = 0.1 → ~10% survival
        let rate = survived as f64 / trials as f64;
        assert!(
            (rate - 0.1).abs() < 0.02,
            "rate {rate} should be ~0.1"
        );
    }

    #[test]
    fn inactive_voxel_lower_zero_is_no_op() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        let mut ww = WeightWindow::uniform(&aabb, [2, 1, 1], 0.25, 1.0);
        // Disable the second voxel.
        ww.lower[1] = 0.0;
        let mut p = make_particle(Vec3::new(1.5, 0.5, 0.5), 0.01);
        let mut pending = Vec::new();
        let mut rng = Rng::new(1, 0);
        apply(&mut p, &ww, &mut rng, &mut pending);
        assert!(p.is_alive());
        assert!((p.weight - 0.01).abs() < 1e-12);
    }
}
