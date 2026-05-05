//! Lattice — regular arrays of universes.
//!
//! Rectangular and hexagonal lattices for repeated geometry (e.g., fuel
//! assemblies in a reactor core). Stub for Phase 2.

use super::{UniverseId, Vec3};

/// Unique identifier for a lattice within a `Geometry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatticeId(pub u32);

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

    /// Distance from `pos` along `dir` to the next grid plane bounding
    /// the element identified by `current`.
    ///
    /// Both `pos` and `dir` are expressed in the lattice's parent
    /// universe frame (i.e. the same frame `find_element` operates in).
    /// `current` is `[ix, iy, iz]` — typically the element returned by
    /// `find_element` for `pos`.
    ///
    /// Returns `f64::INFINITY` if `dir` is zero on every axis (caller
    /// should treat that as "no crossing"). Negative direction
    /// components shoot toward the lower grid plane; positive
    /// components toward the upper one.
    ///
    /// Convention: at the boundary itself (distance zero) the function
    /// reports the distance to the *opposite* plane, not 0 — otherwise
    /// a particle sitting exactly on a grid line would never advance.
    pub fn distance_to_grid(&self, pos: Vec3, dir: Vec3, current: [i32; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for axis in 0..3 {
            let (p, d) = match axis {
                0 => (pos.x - self.origin.x, dir.x),
                1 => (pos.y - self.origin.y, dir.y),
                _ => (pos.z - self.origin.z, dir.z),
            };
            let pitch = match axis {
                0 => self.pitch.x,
                1 => self.pitch.y,
                _ => self.pitch.z,
            };
            if d == 0.0 {
                continue;
            }
            // Plane index: when moving in +d direction, target the upper
            // plane of the current element (current[axis] + 1); when
            // moving in -d direction, target the lower plane (current[axis]).
            let target = if d > 0.0 {
                (current[axis] + 1) as f64 * pitch
            } else {
                current[axis] as f64 * pitch
            };
            let t = (target - p) / d;
            // Reject zero/negative distances — particle is on or past
            // the plane on this axis. Take the next plane in this case.
            let t = if t <= 0.0 {
                let next_target = if d > 0.0 {
                    (current[axis] + 2) as f64 * pitch
                } else {
                    (current[axis] - 1) as f64 * pitch
                };
                (next_target - p) / d
            } else {
                t
            };
            if t > 0.0 && t < best {
                best = t;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_2x2() -> RectLattice {
        RectLattice {
            origin: Vec3::new(0.0, 0.0, 0.0),
            pitch: Vec3::new(1.0, 1.0, 1.0),
            shape: [2, 2, 1],
            universes: vec![UniverseId(0); 4],
        }
    }

    #[test]
    fn distance_to_right_plane_from_element_zero() {
        // Particle in element (0,0,0) at x=0.3, heading +x with pitch 1.0.
        // Next x-plane is at x=1.0, distance 0.7.
        let lat = unit_2x2();
        let d = lat.distance_to_grid(
            Vec3::new(0.3, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            [0, 0, 0],
        );
        assert!((d - 0.7).abs() < 1e-12, "d = {d}");
    }

    #[test]
    fn distance_to_left_plane_negative_direction() {
        // Particle in element (1,0,0) at x=1.3, heading -x.
        // Lower plane of element 1 is at x=1.0, distance 0.3.
        let lat = unit_2x2();
        let d = lat.distance_to_grid(
            Vec3::new(1.3, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            [1, 0, 0],
        );
        assert!((d - 0.3).abs() < 1e-12, "d = {d}");
    }

    #[test]
    fn diagonal_takes_minimum_axis_distance() {
        // Particle at (0.4, 0.1, 0.5) heading (+1,+1,0)/sqrt(2) (unit). Pitch 1.
        // Distance to x=1: (1-0.4)/(1/√2) = 0.6√2 ≈ 0.849
        // Distance to y=1: (1-0.1)/(1/√2) = 0.9√2 ≈ 1.273
        // Min = 0.6√2.
        let lat = unit_2x2();
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        let d = lat.distance_to_grid(
            Vec3::new(0.4, 0.1, 0.5),
            Vec3::new(inv_sqrt2, inv_sqrt2, 0.0),
            [0, 0, 0],
        );
        assert!((d - 0.6 * 2.0_f64.sqrt()).abs() < 1e-12, "d = {d}");
    }

    #[test]
    fn pure_z_motion_with_unit_z_pitch() {
        // Particle in element (0,0,0) at z=0.0, heading +z, pitch 1.0.
        // Convention: at the boundary, report distance to the opposite plane (1.0).
        let lat = unit_2x2();
        let d = lat.distance_to_grid(
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            [0, 0, 0],
        );
        assert!((d - 1.0).abs() < 1e-12, "d = {d}");
    }

    #[test]
    fn dir_zero_on_all_axes_returns_infinity() {
        let lat = unit_2x2();
        let d = lat.distance_to_grid(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            [0, 0, 0],
        );
        assert!(d.is_infinite());
    }
}
