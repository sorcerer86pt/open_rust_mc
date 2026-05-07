//! Optional MC tallies: surface currents and Cartesian mesh flux.
//!
//! Both are opt-in via `SimConfig.tallies`. When unset, the transport
//! loop skips every tally hook — zero cost on the hot path.
//!
//! - `SurfaceCurrentTally` accumulates J+ and J- (forward / backward
//!   crossings, signed by `dir · normal`) for a user-supplied set of
//!   surface indices. Net current is `J+ - J-`; total is `J+ + J-`.
//! - `MeshFluxTally` is a Cartesian voxel mesh; each flight segment
//!   contributes `w · d_voxel` (track length traversed within the voxel)
//!   to the flux estimator. Per-batch sum-of-squares is accumulated so
//!   the final mean and standard error can be derived over active
//!   batches.

use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{Aabb, Vec3};

/// Surface current tally over a user-tagged set of surface indices.
#[derive(Debug, Clone)]
pub struct SurfaceCurrentTally {
    /// Surface indices to tally. The position of each index in this
    /// vec is the tally bin id used in the result arrays.
    pub surfaces: Vec<usize>,
}

impl SurfaceCurrentTally {
    pub fn new(surfaces: Vec<usize>) -> Self {
        Self { surfaces }
    }

    /// Build a tally over every reflective-BC surface in the slice.
    /// Common case: outer pin / assembly box where the user wants
    /// J+/J- on every face. Surfaces are tagged in slice order.
    pub fn for_reflective_surfaces(surfaces: &[Surface]) -> Self {
        Self::for_bc_matching(surfaces, |bc| bc == BoundaryCondition::Reflective)
    }

    /// Build a tally over every non-Transmission surface — i.e. every
    /// surface that bounds the physical problem (`Reflective` or
    /// `Vacuum`). Use this for leakage / outflow currents on vacuum
    /// boundaries (e.g. Godiva sphere) and reflective faces (pin cell).
    pub fn for_boundary_surfaces(surfaces: &[Surface]) -> Self {
        Self::for_bc_matching(surfaces, |bc| {
            bc == BoundaryCondition::Reflective || bc == BoundaryCondition::Vacuum
        })
    }

    fn for_bc_matching<F>(surfaces: &[Surface], pred: F) -> Self
    where
        F: Fn(BoundaryCondition) -> bool,
    {
        let indices = surfaces
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if pred(s.boundary_condition()) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        Self { surfaces: indices }
    }

    /// Return the tally bin for a crossed surface, or `None` if that
    /// surface isn't tagged.
    #[inline]
    pub fn bin_for(&self, surface_idx: usize) -> Option<usize> {
        self.surfaces.iter().position(|&s| s == surface_idx)
    }

    pub fn n_bins(&self) -> usize {
        self.surfaces.len()
    }
}

/// Cartesian voxel mesh for flux tallies. Origin is the lower corner;
/// `n[i]` voxels along axis i with edge length `spacing[i]`.
#[derive(Debug, Clone)]
pub struct MeshFluxTally {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub n: [usize; 3],
}

impl MeshFluxTally {
    pub fn new(origin: [f64; 3], spacing: [f64; 3], n: [usize; 3]) -> Self {
        Self { origin, spacing, n }
    }

    /// Build a mesh that covers `aabb` exactly with `n[i]` voxels along
    /// axis i. Spacing per axis is the AABB extent divided by `n[i]`.
    /// Useful default for "tally flux throughout the geometry's
    /// fissile box" — the typical pin / assembly use case.
    pub fn from_aabb(aabb: &Aabb, n: [usize; 3]) -> Self {
        let origin = [aabb.min.x, aabb.min.y, aabb.min.z];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0].max(1) as f64,
            (aabb.max.y - aabb.min.y) / n[1].max(1) as f64,
            (aabb.max.z - aabb.min.z) / n[2].max(1) as f64,
        ];
        Self { origin, spacing, n }
    }

    pub fn n_voxels(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    /// Linear voxel index from (ix, iy, iz). No bounds check.
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.n[1] + iy) * self.n[2] + iz
    }

    /// Accumulate `w · d` flux contribution along a straight-line
    /// segment from `start` to `start + dir · length` into `acc`.
    /// Uses an Amanatides-Woo-style 3D DDA: each axis walks one voxel
    /// at a time, the smallest of the three `t_to_next` defines the
    /// next sub-segment. Voxels outside the mesh are skipped.
    #[inline]
    pub fn deposit(&self, start: Vec3, dir: Vec3, length: f64, weight: f64, acc: &mut [f64]) {
        if length <= 0.0 || weight == 0.0 {
            return;
        }

        // Walk axes independently. For each axis: current voxel,
        // direction step (-1 / 0 / +1), parametric t to next voxel
        // boundary, and parametric t increment per voxel.
        let p = [start.x, start.y, start.z];
        let d = [dir.x, dir.y, dir.z];

        let mut iv = [0_isize; 3];
        let mut step = [0_isize; 3];
        let mut t_next = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];

        for a in 0..3 {
            let local = (p[a] - self.origin[a]) / self.spacing[a];
            iv[a] = local.floor() as isize;
            if d[a] > 0.0 {
                step[a] = 1;
                let next_boundary = self.origin[a] + (iv[a] + 1) as f64 * self.spacing[a];
                t_next[a] = (next_boundary - p[a]) / d[a];
                t_delta[a] = self.spacing[a] / d[a];
            } else if d[a] < 0.0 {
                step[a] = -1;
                let next_boundary = self.origin[a] + iv[a] as f64 * self.spacing[a];
                t_next[a] = (next_boundary - p[a]) / d[a];
                t_delta[a] = -self.spacing[a] / d[a];
            } else {
                step[a] = 0;
                t_next[a] = f64::INFINITY;
                t_delta[a] = f64::INFINITY;
            }
        }

        let nx = self.n[0] as isize;
        let ny = self.n[1] as isize;
        let nz = self.n[2] as isize;

        let mut t = 0.0_f64;
        let mut safety = 0_u32;
        while t < length && safety < 10_000 {
            safety += 1;
            // Pick axis with smallest t_next.
            let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
                0
            } else if t_next[1] <= t_next[2] {
                1
            } else {
                2
            };
            let t_end = t_next[axis].min(length);
            let dt = t_end - t;
            if dt > 0.0
                && iv[0] >= 0
                && iv[0] < nx
                && iv[1] >= 0
                && iv[1] < ny
                && iv[2] >= 0
                && iv[2] < nz
            {
                let idx =
                    (iv[0] as usize * self.n[1] + iv[1] as usize) * self.n[2] + iv[2] as usize;
                acc[idx] += weight * dt;
            }
            t = t_end;
            iv[axis] += step[axis];
            t_next[axis] += t_delta[axis];
        }
    }
}

/// Bundle of optional tallies passed to the eigenvalue solver. None
/// of the fields are required; the transport loop tests each `Option`
/// once per particle and skips the tally otherwise.
#[derive(Debug, Clone, Default)]
pub struct Tallies {
    pub surface_current: Option<SurfaceCurrentTally>,
    pub mesh_flux: Option<MeshFluxTally>,
}

impl Tallies {
    pub fn n_surface_bins(&self) -> usize {
        self.surface_current.as_ref().map_or(0, |t| t.n_bins())
    }

    pub fn n_mesh_voxels(&self) -> usize {
        self.mesh_flux.as_ref().map_or(0, |m| m.n_voxels())
    }
}

/// Per-particle tally accumulators. Sized once at particle birth from
/// the `Tallies` config; zero-sized when the corresponding tally is
/// disabled.
#[derive(Debug, Clone)]
pub struct ParticleTallies {
    /// Forward-crossing weight per surface tally bin (dir · normal ≥ 0).
    pub surface_current_pos: Vec<f64>,
    /// Backward-crossing weight per surface tally bin (dir · normal < 0).
    pub surface_current_neg: Vec<f64>,
    /// Track-length flux per voxel (weight × distance through voxel).
    pub mesh_flux: Vec<f64>,
}

impl ParticleTallies {
    pub fn new(tallies: &Tallies) -> Self {
        Self {
            surface_current_pos: vec![0.0; tallies.n_surface_bins()],
            surface_current_neg: vec![0.0; tallies.n_surface_bins()],
            mesh_flux: vec![0.0; tallies.n_mesh_voxels()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_bin_lookup() {
        let t = SurfaceCurrentTally::new(vec![3, 7, 11]);
        assert_eq!(t.bin_for(3), Some(0));
        assert_eq!(t.bin_for(11), Some(2));
        assert_eq!(t.bin_for(5), None);
    }

    #[test]
    fn surface_tally_picks_only_reflective_bcs() {
        use crate::geometry::surface::{BoundaryCondition, Surface};
        let surfaces = vec![
            Surface::PlaneX {
                x0: 0.0,
                bc: BoundaryCondition::Transmission,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: 0.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let t = SurfaceCurrentTally::for_reflective_surfaces(&surfaces);
        assert_eq!(t.surfaces, vec![1, 3]);
    }

    #[test]
    fn mesh_from_aabb_covers_exactly() {
        use crate::geometry::Aabb;
        let aabb = Aabb::new(Vec3::new(-2.0, -1.0, 0.0), Vec3::new(2.0, 1.0, 4.0));
        let mesh = MeshFluxTally::from_aabb(&aabb, [4, 2, 8]);
        assert_eq!(mesh.origin, [-2.0, -1.0, 0.0]);
        assert!((mesh.spacing[0] - 1.0).abs() < 1e-12);
        assert!((mesh.spacing[1] - 1.0).abs() < 1e-12);
        assert!((mesh.spacing[2] - 0.5).abs() < 1e-12);
        assert_eq!(mesh.n_voxels(), 4 * 2 * 8);

        // A diagonal segment fully inside aabb should deposit weight ×
        // length even though it crosses many voxel boundaries.
        let mut acc = vec![0.0; mesh.n_voxels()];
        let dir = Vec3::new(1.0, 0.0, 0.0);
        mesh.deposit(Vec3::new(-1.5, 0.0, 1.0), dir, 3.0, 1.5, &mut acc);
        let total: f64 = acc.iter().sum();
        assert!((total - 4.5).abs() < 1e-9, "total={total}");
    }

    #[test]
    fn mesh_deposit_axial_segment_lands_in_one_voxel() {
        let mesh = MeshFluxTally::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4, 4, 4]);
        let mut acc = vec![0.0; mesh.n_voxels()];
        mesh.deposit(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            0.4,
            1.0,
            &mut acc,
        );
        // Segment lies entirely in voxel (0,0,0).
        assert!((acc[mesh.index(0, 0, 0)] - 0.4).abs() < 1e-12);
        let total: f64 = acc.iter().sum();
        assert!((total - 0.4).abs() < 1e-12);
    }

    #[test]
    fn mesh_deposit_crosses_two_voxels_and_total_equals_segment_length() {
        let mesh = MeshFluxTally::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4, 4, 4]);
        let mut acc = vec![0.0; mesh.n_voxels()];
        // Start at x=0.5, walk +x for length 1.0 — crosses from voxel
        // (0,0,0) to (1,0,0). First half (0.5 cm) in (0,0,0), second
        // half (0.5 cm) in (1,0,0).
        mesh.deposit(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            2.0, // weight to make accounting visible
            &mut acc,
        );
        assert!((acc[mesh.index(0, 0, 0)] - 1.0).abs() < 1e-12);
        assert!((acc[mesh.index(1, 0, 0)] - 1.0).abs() < 1e-12);
        let total: f64 = acc.iter().sum();
        assert!((total - 2.0).abs() < 1e-12); // weight × length
    }

    #[test]
    fn mesh_deposit_segment_outside_mesh_contributes_nothing() {
        let mesh = MeshFluxTally::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2, 2, 2]);
        let mut acc = vec![0.0; mesh.n_voxels()];
        mesh.deposit(
            Vec3::new(-5.0, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            10.0,
            1.0,
            &mut acc,
        );
        let total: f64 = acc.iter().sum();
        assert_eq!(total, 0.0);
    }

    #[test]
    fn diagonal_segment_total_equals_weighted_length() {
        let mesh = MeshFluxTally::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4, 4, 4]);
        let mut acc = vec![0.0; mesh.n_voxels()];
        let dir = Vec3::new(1.0, 1.0, 1.0);
        let inv = 1.0 / 3.0_f64.sqrt();
        let dir = Vec3::new(dir.x * inv, dir.y * inv, dir.z * inv);
        mesh.deposit(Vec3::new(0.1, 0.1, 0.1), dir, 2.0, 3.0, &mut acc);
        let total: f64 = acc.iter().sum();
        // Whole segment is inside the 4×4×4 mesh; total deposit equals weight × length.
        assert!((total - 6.0).abs() < 1e-9, "total={total}");
    }
}
