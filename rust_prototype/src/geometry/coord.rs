//! Coordinate stack for nested-universe geometry traversal.
//!
//! A particle in a recursive geometry doesn't live in a single
//! "current cell" — it lives in a stack of frames, where each frame
//! identifies a universe, the cell within that universe, optionally a
//! lattice element, and the translation from the parent frame's local
//! coordinates into this frame's local coordinates.
//!
//! For v1 lattices are axis-aligned and rotation-free, so `local =
//! parent_local - offset`. The `_dir` helpers exist so callers don't
//! bake that assumption in — when rotations land in task #15 the body
//! changes but the call sites don't.

use super::cell::CellFill;
use super::{Cell, LatticeId, Mat3, UniverseId, Vec3};
use smallvec::SmallVec;

/// One frame in a particle's coordinate stack.
///
/// A frame names which universe and which cell of that universe the
/// particle is in, optionally records the lattice element that hosted
/// the universe, and stores the translation + rotation from the
/// parent frame's local coordinates to this frame's local coordinates:
///
///   `this_local = rotation * (parent_local - offset)`
///   `this_local_dir = rotation * parent_local_dir`
///
/// When `rotation` is `None` (the common case) the math reduces to the
/// pure-translation form `this_local = parent_local - offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub universe: UniverseId,
    /// Index into the global `Geometry::cells` array.
    pub cell_idx: u32,
    /// `Some((lattice_id, [ix, iy, iz]))` if this frame is inside a lattice
    /// element (i.e. the parent cell's fill was `CellFill::Lattice`).
    pub lattice: Option<(LatticeId, [i32; 3])>,
    /// Translation from parent local frame: applied *before* `rotation`.
    pub offset: Vec3,
    /// Rotation from parent local frame to this frame. `None` is
    /// equivalent to `Some(Mat3::IDENTITY)` but cheaper.
    pub rotation: Option<Mat3>,
}

impl Coord {
    /// Build a root-universe frame with no offset, no rotation, and no lattice.
    pub fn root(universe: UniverseId, cell_idx: u32) -> Self {
        Self {
            universe,
            cell_idx,
            lattice: None,
            offset: Vec3::new(0.0, 0.0, 0.0),
            rotation: None,
        }
    }
}

/// A stack of coordinate frames, deepest last.
///
/// Inline-allocated up to depth 4 (root → assembly lattice → pin
/// lattice → cell). Deeper geometries spill to the heap silently.
pub type CoordStack = SmallVec<[Coord; 4]>;

/// Helpers for reading information off the deepest frame.
pub trait CoordStackExt {
    fn deepest(&self) -> &Coord;
    fn deepest_cell_idx(&self) -> usize;

    /// Index into `materials` for the deepest cell, or `None` if the
    /// deepest cell is `Void` or a non-material fill (Universe/Lattice
    /// — which would mean the descent stopped early, a bug).
    fn material_idx(&self, cells: &[Cell]) -> Option<u32>;

    /// Transform a world-frame position into the local frame of the
    /// deepest coordinate.
    fn local_pos(&self, world_pos: Vec3) -> Vec3;

    /// Transform a world-frame direction into the local frame of the
    /// deepest coordinate. Identity for v1 (no rotations).
    fn local_dir(&self, world_dir: Vec3) -> Vec3;
}

impl CoordStackExt for CoordStack {
    #[inline]
    fn deepest(&self) -> &Coord {
        self.last().expect("CoordStack must never be empty")
    }

    #[inline]
    fn deepest_cell_idx(&self) -> usize {
        self.deepest().cell_idx as usize
    }

    #[inline]
    fn material_idx(&self, cells: &[Cell]) -> Option<u32> {
        match cells[self.deepest_cell_idx()].fill {
            CellFill::Material(m) => Some(m),
            _ => None,
        }
    }

    #[inline]
    fn local_pos(&self, world_pos: Vec3) -> Vec3 {
        let mut local = world_pos;
        for frame in self {
            local = local - frame.offset;
            if let Some(r) = frame.rotation {
                local = r.transform(local);
            }
        }
        local
    }

    #[inline]
    fn local_dir(&self, world_dir: Vec3) -> Vec3 {
        let mut dir = world_dir;
        for frame in self {
            if let Some(r) = frame.rotation {
                dir = r.transform(dir);
            }
        }
        dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn root_frame_has_zero_offset() {
        let c = Coord::root(UniverseId(0), 7);
        assert_eq!(c.universe, UniverseId(0));
        assert_eq!(c.cell_idx, 7);
        assert_eq!(c.offset, Vec3::new(0.0, 0.0, 0.0));
        assert!(c.lattice.is_none());
    }

    #[test]
    fn local_pos_subtracts_offsets_in_order() {
        // Root frame at world origin; child frame offset by (1,2,3); grandchild by (10,0,0).
        let stack: CoordStack = smallvec![
            Coord::root(UniverseId(0), 0),
            Coord {
                universe: UniverseId(1),
                cell_idx: 1,
                lattice: Some((LatticeId(0), [1, 0, 0])),
                offset: Vec3::new(1.0, 2.0, 3.0),
                rotation: None,
            },
            Coord {
                universe: UniverseId(2),
                cell_idx: 2,
                lattice: None,
                offset: Vec3::new(10.0, 0.0, 0.0),
                rotation: None,
            },
        ];

        let world = Vec3::new(15.0, 5.0, 7.0);
        let local = stack.local_pos(world);
        // 15 - 0 - 1 - 10 = 4; 5 - 0 - 2 - 0 = 3; 7 - 0 - 3 - 0 = 4
        assert_eq!(local, Vec3::new(4.0, 3.0, 4.0));
    }

    #[test]
    fn local_dir_is_identity_when_no_rotation() {
        let stack: CoordStack = smallvec![Coord::root(UniverseId(0), 0)];
        let dir = Vec3::new(0.6, 0.8, 0.0);
        assert_eq!(stack.local_dir(dir), dir);
    }

    #[test]
    fn local_pos_applies_rotation_after_offset() {
        // 90° z rotation on the deeper frame. Root has no offset/rotation.
        // Particle at world (1, 0, 0), child frame offset (0, 0, 0) and
        // rotation 90° z. Expected local: rotate (1, 0, 0) → (0, 1, 0).
        let r = crate::geometry::Mat3::rotation_z(std::f64::consts::FRAC_PI_2);
        let stack: CoordStack = smallvec![
            Coord::root(UniverseId(0), 0),
            Coord {
                universe: UniverseId(1),
                cell_idx: 1,
                lattice: None,
                offset: Vec3::new(0.0, 0.0, 0.0),
                rotation: Some(r),
            },
        ];
        let world = Vec3::new(1.0, 0.0, 0.0);
        let local = stack.local_pos(world);
        assert!((local.x - 0.0).abs() < 1e-12);
        assert!((local.y - 1.0).abs() < 1e-12);
        assert!(local.z.abs() < 1e-12);

        let dir = Vec3::new(1.0, 0.0, 0.0);
        let local_dir = stack.local_dir(dir);
        assert!((local_dir.x - 0.0).abs() < 1e-12);
        assert!((local_dir.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deepest_returns_last_frame() {
        let stack: CoordStack = smallvec![
            Coord::root(UniverseId(0), 1),
            Coord::root(UniverseId(5), 9),
        ];
        assert_eq!(stack.deepest().universe, UniverseId(5));
        assert_eq!(stack.deepest_cell_idx(), 9);
    }
}
