// SPDX-License-Identifier: MIT
//! Coordinate stack for nested-universe geometry.
//!
//! `this_local = rotation * (parent_local - offset)`; when
//! `rotation = None` it reduces to pure translation.

use super::cell::CellFill;
use super::{Cell, HexLatticeId, LatticeId, Mat3, UniverseId, Vec3};
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub universe: UniverseId,
    /// Index into `Geometry::cells`.
    pub cell_idx: u32,
    /// `(lattice_id, [ix, iy, iz])`. Mutually exclusive with `hex_lattice`.
    pub lattice: Option<(LatticeId, [i32; 3])>,
    /// `[q, r, z]`; axial coords (cube `s = -q-r`), `z` = axial layer.
    pub hex_lattice: Option<(HexLatticeId, [i32; 3])>,
    pub offset: Vec3,
    /// `None` ≡ identity, cheaper.
    pub rotation: Option<Mat3>,
}

impl Coord {
    pub fn root(universe: UniverseId, cell_idx: u32) -> Self {
        Self {
            universe,
            cell_idx,
            lattice: None,
            hex_lattice: None,
            offset: Vec3::new(0.0, 0.0, 0.0),
            rotation: None,
        }
    }
}

/// Deepest last. Inline up to depth 4 (root → assembly → pin → cell);
/// spills to heap silently for deeper.
pub type CoordStack = SmallVec<[Coord; 4]>;

pub trait CoordStackExt {
    fn deepest(&self) -> &Coord;
    fn deepest_cell_idx(&self) -> usize;
    /// `None` on Void or non-Material fill (the latter is a descent bug).
    fn material_idx(&self, cells: &[Cell]) -> Option<u32>;
    fn local_pos(&self, world_pos: Vec3) -> Vec3;
    /// Identity in v1 (no rotations).
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
                hex_lattice: None,
                offset: Vec3::new(1.0, 2.0, 3.0),
                rotation: None,
            },
            Coord {
                universe: UniverseId(2),
                cell_idx: 2,
                lattice: None,
                hex_lattice: None,
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
                hex_lattice: None,
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
        let stack: CoordStack =
            smallvec![Coord::root(UniverseId(0), 1), Coord::root(UniverseId(5), 9),];
        assert_eq!(stack.deepest().universe, UniverseId(5));
        assert_eq!(stack.deepest_cell_idx(), 9);
    }
}
