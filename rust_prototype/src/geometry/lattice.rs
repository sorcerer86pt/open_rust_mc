//! Lattice — regular arrays of universes.
//!
//! Rectangular and hexagonal lattices for repeated geometry (e.g., fuel
//! assemblies in a reactor core). Stub for Phase 2.

use super::{UniverseId, Vec3};

/// A rectangular lattice of universes.
#[derive(Debug, Clone)]
pub struct RectLattice {
    /// Lower-left corner of the lattice.
    pub origin: Vec3,
    /// Pitch in each direction.
    pub pitch: Vec3,
    /// Number of elements in each direction.
    pub shape: [usize; 3],
    /// Universe IDs filling each lattice position, row-major.
    pub universes: Vec<UniverseId>,
}

impl RectLattice {
    /// Find which lattice element a point falls in.
    pub fn find_element(&self, pos: Vec3) -> Option<(usize, usize, usize)> {
        let rel = pos - self.origin;
        let ix = (rel.x / self.pitch.x).floor() as isize;
        let iy = (rel.y / self.pitch.y).floor() as isize;
        let iz = (rel.z / self.pitch.z).floor() as isize;

        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
        if ix >= self.shape[0] || iy >= self.shape[1] || iz >= self.shape[2] {
            return None;
        }
        Some((ix, iy, iz))
    }

    /// Get the universe ID at a lattice position.
    pub fn universe_at(&self, ix: usize, iy: usize, iz: usize) -> UniverseId {
        let idx = iz * self.shape[1] * self.shape[0] + iy * self.shape[0] + ix;
        self.universes[idx]
    }

    /// Get the local coordinate within a lattice element.
    pub fn local_position(&self, pos: Vec3, ix: usize, iy: usize, iz: usize) -> Vec3 {
        Vec3::new(
            pos.x - self.origin.x - (ix as f64) * self.pitch.x,
            pos.y - self.origin.y - (iy as f64) * self.pitch.y,
            pos.z - self.origin.z - (iz as f64) * self.pitch.z,
        )
    }
}
