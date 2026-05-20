// SPDX-License-Identifier: MIT
//! Flat-source-region (FSR) mesh.
//!
//! Two modes share a single `FsrMesh` struct, distinguished by the
//! `FsrMeshKind` enum:
//!
//! - **Cartesian** (`from_geometry` / `uniform`): a regular voxel grid
//!   over an `Aabb`. `fsr_at(pos, _)` is O(1) integer division. Volume
//!   per FSR is the uniform `voxel_volume` constant. Suitable for
//!   shielding problems and rectangular geometries.
//!
//! - **Cell-based** (`cell_based`): one FSR per leaf-cell × lattice-
//!   element key. `fsr_at(_, stack)` reads the deepest stack frame's
//!   `(cell_idx, lattice_element)` and looks up a `HashMap`. Volume
//!   per FSR is either user-provided analytic or stochastically
//!   estimated by the solver from track lengths. Suitable for
//!   pin-cell / assembly geometries where Cartesian voxels would
//!   over-discretise.
//!
//! Both modes return per-FSR `material[]` + `active[]` arrays. The
//! solver caches these at construction so the hot path only needs
//! `fsr_at(pos, stack) -> Option<usize>`.

use std::collections::HashMap;

use crate::geometry::cell::CellFill;
use crate::geometry::coord::CoordStack;
use crate::geometry::ray::find_cell_recursive;
use crate::geometry::{Aabb, EffectiveFill, Geometry, Vec3};

#[derive(Debug, Clone)]
pub struct FsrMesh {
    pub aabb: Aabb,
    /// Material id per FSR. `u32::MAX` means inactive / void.
    pub material: Vec<u32>,
    /// True if the FSR is backed by a real material in the library.
    pub active: Vec<bool>,
    pub kind: FsrMeshKind,
}

#[derive(Debug, Clone)]
pub enum FsrMeshKind {
    Cartesian {
        n: [usize; 3],
        spacing: [f64; 3],
        voxel_volume: f64,
    },
    Cell {
        /// `(cell_idx, optional lattice element key)` → FSR id. The
        /// lattice key is a flat index into the lattice's element grid;
        /// `None` for cells outside any lattice.
        cell_to_fsr: HashMap<(u32, Option<i64>), usize>,
        /// Per-FSR volume in cm³. Zero / negative entries fall back
        /// to the solver's stochastic volume estimate.
        fsr_volume: Vec<f64>,
    },
}

impl FsrMesh {
    pub const VOID: u32 = u32::MAX;

    /// Cartesian voxel mesh sampled at voxel centroids.
    pub fn from_geometry(aabb: Aabb, n: [usize; 3], geom: &Geometry) -> Self {
        let n = [n[0].max(1), n[1].max(1), n[2].max(1)];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0] as f64,
            (aabb.max.y - aabb.min.y) / n[1] as f64,
            (aabb.max.z - aabb.min.z) / n[2] as f64,
        ];
        let voxel_volume = spacing[0] * spacing[1] * spacing[2];
        let total = n[0] * n[1] * n[2];
        let mut material = vec![Self::VOID; total];
        let mut active = vec![false; total];
        for ix in 0..n[0] {
            for iy in 0..n[1] {
                for iz in 0..n[2] {
                    let cx = aabb.min.x + (ix as f64 + 0.5) * spacing[0];
                    let cy = aabb.min.y + (iy as f64 + 0.5) * spacing[1];
                    let cz = aabb.min.z + (iz as f64 + 0.5) * spacing[2];
                    let c = Vec3::new(cx, cy, cz);
                    if let Some(stack) = find_cell_recursive(c, geom) {
                        let fill = geom.effective_material_idx(&stack);
                        if let EffectiveFill::Material(mat) = fill {
                            let idx = Self::cart_flat_index(n, ix, iy, iz);
                            material[idx] = mat;
                            active[idx] = true;
                        }
                    }
                }
            }
        }
        Self {
            aabb,
            material,
            active,
            kind: FsrMeshKind::Cartesian {
                n,
                spacing,
                voxel_volume,
            },
        }
    }

    /// Cartesian mesh with a fixed material id (no geometry query).
    /// Useful for analytic infinite-medium tests.
    pub fn uniform(aabb: Aabb, n: [usize; 3], mat: u32) -> Self {
        let n = [n[0].max(1), n[1].max(1), n[2].max(1)];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0] as f64,
            (aabb.max.y - aabb.min.y) / n[1] as f64,
            (aabb.max.z - aabb.min.z) / n[2] as f64,
        ];
        let voxel_volume = spacing[0] * spacing[1] * spacing[2];
        let total = n[0] * n[1] * n[2];
        Self {
            aabb,
            material: vec![mat; total],
            active: vec![true; total],
            kind: FsrMeshKind::Cartesian {
                n,
                spacing,
                voxel_volume,
            },
        }
    }

    /// Cell-based FSR mesh. One FSR per `(deepest cell index, lattice
    /// element key)` pair encountered while walking the geometry.
    /// `aabb` is the bounding region used for ray sampling — it should
    /// enclose the active geometry and may be larger.
    ///
    /// `analytic_volume` is an optional per-(cell, element) volume map.
    /// If `None` for an FSR, that entry stays at zero and the solver
    /// uses stochastic track-length volume estimation. Usual practice
    /// is to provide analytic volumes for simple shapes (slabs, pin
    /// cylinders) and let the solver estimate the rest.
    pub fn cell_based(
        aabb: Aabb,
        geom: &Geometry,
        sample_grid: [usize; 3],
        analytic_volume: Option<&HashMap<(u32, Option<i64>), f64>>,
    ) -> Self {
        let n = [
            sample_grid[0].max(1),
            sample_grid[1].max(1),
            sample_grid[2].max(1),
        ];
        let spacing = [
            (aabb.max.x - aabb.min.x) / n[0] as f64,
            (aabb.max.y - aabb.min.y) / n[1] as f64,
            (aabb.max.z - aabb.min.z) / n[2] as f64,
        ];
        let mut cell_to_fsr: HashMap<(u32, Option<i64>), usize> = HashMap::new();
        let mut material: Vec<u32> = Vec::new();
        let mut active: Vec<bool> = Vec::new();
        // Walk the sample grid to discover (cell, element) keys
        // exhaustively. Any key not seen here will not have an FSR
        // and will be treated as a non-tally region by the solver.
        for ix in 0..n[0] {
            for iy in 0..n[1] {
                for iz in 0..n[2] {
                    let cx = aabb.min.x + (ix as f64 + 0.5) * spacing[0];
                    let cy = aabb.min.y + (iy as f64 + 0.5) * spacing[1];
                    let cz = aabb.min.z + (iz as f64 + 0.5) * spacing[2];
                    let c = Vec3::new(cx, cy, cz);
                    let stack = match find_cell_recursive(c, geom) {
                        Some(s) => s,
                        None => continue,
                    };
                    let key = key_for_stack(&stack);
                    if cell_to_fsr.contains_key(&key) {
                        continue;
                    }
                    // Resolve the material at this stack; void cells
                    // get an inactive FSR slot so the solver doesn't
                    // try to look them up.
                    let fill = geom.effective_material_idx(&stack);
                    let (mat, is_active) = match fill {
                        EffectiveFill::Material(m) => (m, true),
                        EffectiveFill::Void => (Self::VOID, false),
                    };
                    // Sanity: deepest cell should not be a Universe /
                    // Lattice fill. find_cell_recursive descends, so
                    // this only matters as a defensive assert.
                    let last_cell = stack.last().expect("non-empty stack").cell_idx as usize;
                    debug_assert!(matches!(
                        geom.cells[last_cell].fill,
                        CellFill::Material(_) | CellFill::Void
                    ));
                    let id = material.len();
                    cell_to_fsr.insert(key, id);
                    material.push(mat);
                    active.push(is_active);
                }
            }
        }
        let mut fsr_volume = vec![0.0_f64; material.len()];
        if let Some(av) = analytic_volume {
            for (key, &id) in &cell_to_fsr {
                if let Some(&v) = av.get(key) {
                    fsr_volume[id] = v;
                }
            }
        }
        Self {
            aabb,
            material,
            active,
            kind: FsrMeshKind::Cell {
                cell_to_fsr,
                fsr_volume,
            },
        }
    }

    #[inline]
    pub fn cart_flat_index(n: [usize; 3], ix: usize, iy: usize, iz: usize) -> usize {
        (ix * n[1] + iy) * n[2] + iz
    }

    #[inline]
    pub fn n_fsrs(&self) -> usize {
        self.material.len()
    }

    /// FSR volume in cm³.
    ///
    /// Cartesian: the uniform voxel volume (same value for every FSR).
    /// Cell-based: the analytic volume if provided, else `0.0` —
    /// the solver substitutes its stochastic estimate when this is
    /// non-positive.
    #[inline]
    pub fn fsr_volume(&self, fsr: usize) -> f64 {
        match &self.kind {
            FsrMeshKind::Cartesian { voxel_volume, .. } => *voxel_volume,
            FsrMeshKind::Cell { fsr_volume, .. } => *fsr_volume.get(fsr).unwrap_or(&0.0),
        }
    }

    /// Locate the FSR id at a world-frame `(pos, stack)`. `stack` is
    /// only used for cell-based meshes; Cartesian ignores it.
    pub fn fsr_at(&self, pos: Vec3, stack: &CoordStack) -> Option<usize> {
        match &self.kind {
            FsrMeshKind::Cartesian { n, spacing, .. } => {
                let ix = ((pos.x - self.aabb.min.x) / spacing[0]).floor() as isize;
                let iy = ((pos.y - self.aabb.min.y) / spacing[1]).floor() as isize;
                let iz = ((pos.z - self.aabb.min.z) / spacing[2]).floor() as isize;
                if ix < 0
                    || iy < 0
                    || iz < 0
                    || ix as usize >= n[0]
                    || iy as usize >= n[1]
                    || iz as usize >= n[2]
                {
                    return None;
                }
                Some(Self::cart_flat_index(
                    *n,
                    ix as usize,
                    iy as usize,
                    iz as usize,
                ))
            }
            FsrMeshKind::Cell { cell_to_fsr, .. } => {
                let key = key_for_stack(stack);
                cell_to_fsr.get(&key).copied()
            }
        }
    }

    /// Cartesian-only voxel index at `pos`. Returns `None` for
    /// cell-based meshes.
    pub fn voxel_index(&self, pos: Vec3) -> Option<usize> {
        match &self.kind {
            FsrMeshKind::Cartesian { n, spacing, .. } => {
                let ix = ((pos.x - self.aabb.min.x) / spacing[0]).floor() as isize;
                let iy = ((pos.y - self.aabb.min.y) / spacing[1]).floor() as isize;
                let iz = ((pos.z - self.aabb.min.z) / spacing[2]).floor() as isize;
                if ix < 0
                    || iy < 0
                    || iz < 0
                    || ix as usize >= n[0]
                    || iy as usize >= n[1]
                    || iz as usize >= n[2]
                {
                    return None;
                }
                Some(Self::cart_flat_index(
                    *n,
                    ix as usize,
                    iy as usize,
                    iz as usize,
                ))
            }
            FsrMeshKind::Cell { .. } => None,
        }
    }

    /// Cartesian convenience: voxel volume (panics on cell-based).
    /// Use `fsr_volume(f)` for portable per-FSR access.
    pub fn voxel_volume(&self) -> f64 {
        match &self.kind {
            FsrMeshKind::Cartesian { voxel_volume, .. } => *voxel_volume,
            FsrMeshKind::Cell { .. } => {
                panic!("voxel_volume() called on cell-based mesh; use fsr_volume(f)")
            }
        }
    }

    /// Cartesian dimensions `[n_x, n_y, n_z]`. Returns `[0,0,0]` for
    /// cell-based meshes — callers that need this should branch on
    /// `kind` first.
    pub fn cartesian_n(&self) -> [usize; 3] {
        match &self.kind {
            FsrMeshKind::Cartesian { n, .. } => *n,
            FsrMeshKind::Cell { .. } => [0, 0, 0],
        }
    }

    /// Set per-FSR analytic volumes (cell-based only). Volumes ≤ 0
    /// are treated as "use stochastic estimate".
    pub fn set_fsr_volumes(&mut self, volumes: Vec<f64>) {
        match &mut self.kind {
            FsrMeshKind::Cell { fsr_volume, .. } => {
                assert_eq!(volumes.len(), fsr_volume.len(), "volume length mismatch");
                *fsr_volume = volumes;
            }
            FsrMeshKind::Cartesian { .. } => {
                panic!("set_fsr_volumes is only valid for cell-based meshes")
            }
        }
    }
}

/// Hash key for the deepest stack frame of a `CoordStack`. Encodes
/// `(cell_idx, optional flat lattice element index)` so cells inside
/// lattices get one FSR per element while cells outside lattices get
/// one FSR per cell.
fn key_for_stack(stack: &CoordStack) -> (u32, Option<i64>) {
    let last = stack.last().expect("empty stack");
    let cell_idx = last.cell_idx;
    let lat_key = if let Some((_, el)) = last.lattice {
        // Pack the [i32; 3] element coordinate into a single i64.
        // (-1 << 21..1 << 21 fits both axes; lattice extents stay well
        // below this in practice.)
        let packed = ((el[0] as i64) & 0xFFFF)
            | (((el[1] as i64) & 0xFFFF) << 16)
            | (((el[2] as i64) & 0xFFFF) << 32);
        Some(packed)
    } else if let Some((_, el)) = last.hex_lattice {
        let packed = ((el[0] as i64) & 0xFFFF)
            | (((el[1] as i64) & 0xFFFF) << 16)
            | (((el[2] as i64) & 0xFFFF) << 32);
        Some(packed)
    } else {
        None
    };
    (cell_idx, lat_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Surface;
    use crate::geometry::cell::{self, Cell, CellFill, CellId, Region};
    use crate::geometry::surface::BoundaryCondition;

    fn unit_box_geometry(mat: u32) -> Geometry {
        // 6 reflective planes forming a unit cube centred at origin.
        let surfaces = vec![
            Surface::PlaneX {
                x0: -0.5,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: 0.5,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -0.5,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: 0.5,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: -0.5,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: 0.5,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let region = cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]);
        let outside_region = Region::Complement(Box::new(cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ])));
        let cells = vec![
            Cell::new(CellId(0), region, CellFill::Material(mat)),
            Cell::new(CellId(1), outside_region, CellFill::Void),
        ];
        Geometry::flat(surfaces, cells).expect("flat geometry")
    }

    #[test]
    fn unit_box_voxels_resolve_to_material() {
        let geom = unit_box_geometry(0);
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        let mesh = FsrMesh::from_geometry(aabb, [4, 4, 4], &geom);
        assert_eq!(mesh.n_fsrs(), 64);
        // Volume of mesh is 1.0 → each voxel = 1/64.
        assert!((mesh.voxel_volume() - 1.0 / 64.0).abs() < 1e-12);
        for active in &mesh.active {
            assert!(*active, "every voxel should be inside the unit box");
        }
        for &m in &mesh.material {
            assert_eq!(m, 0);
        }
    }

    #[test]
    fn voxel_index_round_trips_through_centroid() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let mesh = FsrMesh::uniform(aabb, [2, 2, 2], 0);
        let mid = Vec3::new(0.5, 0.5, 0.5); // voxel (0,0,0)
        assert_eq!(mesh.voxel_index(mid), Some(0));
        let mid2 = Vec3::new(1.5, 1.5, 1.5); // voxel (1,1,1)
        let last = FsrMesh::cart_flat_index([2, 2, 2], 1, 1, 1);
        assert_eq!(mesh.voxel_index(mid2), Some(last));
    }

    #[test]
    fn voxel_outside_returns_none() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let mesh = FsrMesh::uniform(aabb, [2, 2, 2], 0);
        assert_eq!(mesh.voxel_index(Vec3::new(-1.0, 0.5, 0.5)), None);
        assert_eq!(mesh.voxel_index(Vec3::new(0.5, 2.0, 0.5)), None);
    }

    #[test]
    fn cell_based_unit_box_collapses_to_one_fsr() {
        // The unit-box geometry has one material cell. Cell-based
        // discovery should produce exactly one active FSR no matter
        // how dense the sample grid is.
        let geom = unit_box_geometry(0);
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        let mesh = FsrMesh::cell_based(aabb, &geom, [4, 4, 4], None);
        assert_eq!(mesh.n_fsrs(), 1);
        assert!(mesh.active[0]);
        assert_eq!(mesh.material[0], 0);
        // No analytic volume provided → 0.0; solver substitutes
        // stochastic estimate.
        assert_eq!(mesh.fsr_volume(0), 0.0);
    }

    #[test]
    fn cell_based_with_analytic_volume_returns_it() {
        let geom = unit_box_geometry(0);
        let aabb = Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        let mut analytic = HashMap::new();
        // The cell-based key for this single-cell, no-lattice geometry
        // is (0, None) — cell index 0 with no lattice element.
        analytic.insert((0_u32, None), 1.0);
        let mesh = FsrMesh::cell_based(aabb, &geom, [4, 4, 4], Some(&analytic));
        assert_eq!(mesh.n_fsrs(), 1);
        assert!((mesh.fsr_volume(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fsr_at_cartesian_returns_voxel_index() {
        let aabb = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let geom = unit_box_geometry(0); // unused for Cartesian fsr_at
        let mesh = FsrMesh::uniform(aabb, [2, 2, 2], 0);
        let mid = Vec3::new(0.5, 0.5, 0.5);
        let stack = find_cell_recursive(mid, &geom).unwrap_or_default();
        assert_eq!(mesh.fsr_at(mid, &stack), Some(0));
    }
}
