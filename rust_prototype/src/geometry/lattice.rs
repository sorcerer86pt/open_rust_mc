//! Lattice — regular arrays of universes.
//!
//! Rectangular and hexagonal lattices for repeated geometry (e.g., fuel
//! assemblies in a reactor core). Stub for Phase 2.

use std::collections::HashMap;

use super::{UniverseId, Vec3};

/// Unique identifier for a lattice within a `Geometry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatticeId(pub u32);

/// Per-element override of a cell's static material fill.
///
/// Map from global cell index (into `Geometry::cells`) to the
/// overriding material index. When a particle's deepest stack frame
/// is in this lattice element, transport uses the overriding
/// material in place of `cell.fill`.
pub type MaterialOverrideMap = HashMap<usize, u32>;

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
    /// Optional per-element material overrides. When `Some`, the Vec
    /// has length `shape[0] * shape[1] * shape[2]`, addressed in the
    /// same row-major order as `universes`. The map at each element
    /// rebinds specific cells (by global cell index) to a different
    /// material index, letting one pin universe be reused at many
    /// lattice positions with different enrichments / burnup tiers.
    /// `None` means no overrides anywhere in the lattice (the common
    /// case; checked once in the hot path and short-circuited).
    pub material_overrides: Option<Vec<MaterialOverrideMap>>,
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

    /// Row-major linear index for an element. Use with `universes` and
    /// `material_overrides` indexing.
    #[inline]
    pub fn linear_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.shape[1] * self.shape[0] + iy * self.shape[0] + ix
    }

    /// Resolve an element from a signed `[ix, iy, iz]` (the form
    /// stored on `Coord.lattice`) to its linear index. Returns
    /// `None` if any axis is out of range — should not happen for a
    /// well-formed `Coord` produced by `find_cell_recursive`.
    #[inline]
    pub fn linear_index_signed(&self, ix: i32, iy: i32, iz: i32) -> Option<usize> {
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
        if ix >= self.shape[0] || iy >= self.shape[1] || iz >= self.shape[2] {
            return None;
        }
        Some(self.linear_index(ix, iy, iz))
    }

    /// Look up the overriding material for `cell_idx` at lattice
    /// element `[ix, iy, iz]`, if any. Returns `None` when this
    /// lattice has no overrides at all, the element index is out of
    /// range, or the element has no override for this cell.
    #[inline]
    pub fn material_override(&self, element: [i32; 3], cell_idx: usize) -> Option<u32> {
        let overrides = self.material_overrides.as_ref()?;
        let idx = self.linear_index_signed(element[0], element[1], element[2])?;
        let elem_overrides = overrides.get(idx)?;
        elem_overrides.get(&cell_idx).copied()
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

// ── Hexagonal lattice ──────────────────────────────────────────────

/// Orientation of a hex lattice. Matches OpenMC's convention:
/// `Y` = flat-top hexagons (top/bottom edges horizontal), used by
/// VVER-1000 and most water-cooled hex reactors. `X` = pointy-top
/// hexagons (vertex at top), used by some sodium-cooled fast
/// reactors and matches the standard "Red Blob Games" pointy-top
/// orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexOrientation {
    Y,
    X,
}

/// A hexagonal lattice of universes.
///
/// Internally indexed by axial coordinates `(q, r, z)` with the
/// constraint `q + r + s = 0` (the third cube coordinate `s` is
/// implicit). The valid axial range is `|q|, |r|, |s| ≤ n_rings`.
/// Universes are stored row-major over a doubled `(2*n_rings+1) ×
/// (2*n_rings+1)` grid per axial layer; the corner cells outside the
/// hex (where ring > n_rings) carry an arbitrary `UniverseId(0)`
/// placeholder that never gets queried.
#[derive(Debug, Clone)]
pub struct HexLattice {
    /// Centre of the hex lattice (in world frame, before this
    /// lattice's own offset is subtracted).
    pub center: Vec3,
    /// Centre-to-centre distance to any of the 6 in-plane neighbours.
    /// For a flat-top hex (`Y`) this equals √3 × side_length.
    pub pitch_xy: f64,
    /// Axial pitch (centre-to-centre between z layers).
    pub pitch_z: f64,
    /// Number of hex rings (ring 0 = centre only, ring N = 6N
    /// elements). Total in-plane elements = `1 + 3*N*(N+1)`.
    pub n_rings: usize,
    /// Number of axial layers.
    pub n_axial: usize,
    pub orientation: HexOrientation,
    /// Universe at each `(q, r, z)`, stored row-major over the
    /// doubled axial grid.
    pub universes: Vec<UniverseId>,
    /// Optional per-element material overrides, same convention as
    /// `RectLattice`.
    pub material_overrides: Option<Vec<MaterialOverrideMap>>,
}

impl HexLattice {
    /// Total in-plane elements for an N-ring lattice.
    #[inline]
    pub fn elements_per_slice(n_rings: usize) -> usize {
        1 + 3 * n_rings * (n_rings + 1)
    }

    /// Hex side length: `pitch_xy / √3` for both orientations
    /// (centre-to-centre distance equals √3 × side_length whether
    /// the hex is pointy-top or flat-top).
    #[inline]
    pub fn side_length(&self) -> f64 {
        self.pitch_xy / 3.0_f64.sqrt()
    }

    /// Linear index into the doubled-axial `universes` Vec.
    #[inline]
    pub fn axial_index(&self, q: i32, r: i32, z: i32) -> Option<usize> {
        let n = self.n_rings as i32;
        if q < -n || q > n || r < -n || r > n {
            return None;
        }
        if z < 0 || z >= self.n_axial as i32 {
            return None;
        }
        let qi = (q + n) as usize;
        let ri = (r + n) as usize;
        let zi = z as usize;
        let stride = (2 * self.n_rings + 1) * (2 * self.n_rings + 1);
        Some(zi * stride + ri * (2 * self.n_rings + 1) + qi)
    }

    /// Cartesian centre of element `(q, r)` in the lattice frame
    /// (relative to `self.center`).
    pub fn element_center_local(&self, q: i32, r: i32) -> Vec3 {
        let s = self.side_length();
        let (q, r) = (q as f64, r as f64);
        match self.orientation {
            // Flat-top: q-axis along world x; r-axis tilted up-right.
            // Centre offsets follow Red Blob Games' flat-top axial
            // convention (size = side length s):
            //   x = 1.5 * s * q
            //   y = √3 * s * (r + q/2)
            HexOrientation::Y => Vec3::new(
                1.5 * s * q,
                3.0_f64.sqrt() * s * (r + q * 0.5),
                0.0,
            ),
            // Pointy-top: q-axis tilted right, r-axis along world y.
            //   x = √3 * s * (q + r/2)
            //   y = 1.5 * s * r
            HexOrientation::X => Vec3::new(
                3.0_f64.sqrt() * s * (q + r * 0.5),
                1.5 * s * r,
                0.0,
            ),
        }
    }

    /// Find which element contains `world_pos`. Uses the standard
    /// pixel-to-hex round via cube coordinates.
    pub fn find_element(&self, world_pos: Vec3) -> Option<(i32, i32, i32)> {
        let p = world_pos - self.center;
        let s = self.side_length();

        // Cartesian → fractional axial.
        let (qf, rf) = match self.orientation {
            HexOrientation::Y => {
                // Flat-top inverse:
                //   q = (2/3 * x) / s
                //   r = (-x/3 + √3/3 * y) / s
                let q = (2.0 / 3.0 * p.x) / s;
                let r = (-p.x / 3.0 + (3.0_f64.sqrt() / 3.0) * p.y) / s;
                (q, r)
            }
            HexOrientation::X => {
                // Pointy-top inverse:
                //   q = (√3/3 * x - y/3) / s
                //   r = (2/3 * y) / s
                let q = ((3.0_f64.sqrt() / 3.0) * p.x - p.y / 3.0) / s;
                let r = (2.0 / 3.0 * p.y) / s;
                (q, r)
            }
        };

        // Cube rounding.
        let xf = qf;
        let zf = rf;
        let yf = -xf - zf;
        let (mut x, mut y, mut z) = (xf.round(), yf.round(), zf.round());
        let (xd, yd, zd) = ((x - xf).abs(), (y - yf).abs(), (z - zf).abs());
        if xd > yd && xd > zd {
            x = -y - z;
        } else if yd > zd {
            y = -x - z;
        } else {
            z = -x - y;
        }
        let _ = y;
        let q = x as i32;
        let r = z as i32;

        // Bounds check via cube ring radius.
        let cube_s = -q - r;
        let ring = q.unsigned_abs().max(r.unsigned_abs()).max(cube_s.unsigned_abs()) as usize;
        if ring > self.n_rings {
            return None;
        }

        // z layer: lattice z range is `[-n_axial/2, +n_axial/2) * pitch_z`
        // around `center.z`; index 0 is the lowest layer.
        let dz = p.z / self.pitch_z + (self.n_axial as f64) * 0.5;
        let zi = dz.floor() as i32;
        if zi < 0 || zi >= self.n_axial as i32 {
            return None;
        }

        Some((q, r, zi))
    }

    /// Get the universe at axial coordinate `(q, r, z)`.
    pub fn universe_at(&self, q: i32, r: i32, z: i32) -> UniverseId {
        let idx = self
            .axial_index(q, r, z)
            .expect("axial index out of range");
        self.universes[idx]
    }

    /// Lattice-frame local position of `world_pos` within element
    /// `(q, r, z)`: subtracts the element centre. Useful for the
    /// recursive descent's offset calculation.
    pub fn local_position(&self, world_pos: Vec3, q: i32, r: i32, z: i32) -> Vec3 {
        let centre_xy = self.element_center_local(q, r);
        let centre_z =
            (z as f64 - (self.n_axial as f64) * 0.5 + 0.5) * self.pitch_z;
        let world_local = world_pos - self.center;
        Vec3::new(
            world_local.x - centre_xy.x,
            world_local.y - centre_xy.y,
            world_local.z - centre_z,
        )
    }

    /// Distance along `local_dir` from `local_pos` (in lattice frame)
    /// until the particle crosses out of element `current`. Six in-plane
    /// edges plus the two axial planes; min positive is reported.
    pub fn distance_to_grid(
        &self,
        local_pos: Vec3,
        local_dir: Vec3,
        current: [i32; 3],
    ) -> f64 {
        let d_perp = self.pitch_xy * 0.5;
        let elem_center = self.element_center_local(current[0], current[1]);
        let pos_rel = Vec3::new(
            local_pos.x - elem_center.x,
            local_pos.y - elem_center.y,
            0.0,
        );

        // Six outward edge normals (unit vectors). For flat-top (Y),
        // edges are perpendicular to angles 30°/90°/150°/210°/270°/330°.
        // For pointy-top (X), 0°/60°/120°/180°/240°/300°.
        let sqrt3_2 = 3.0_f64.sqrt() * 0.5;
        let normals: [Vec3; 6] = match self.orientation {
            HexOrientation::Y => [
                Vec3::new(sqrt3_2, 0.5, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-sqrt3_2, 0.5, 0.0),
                Vec3::new(-sqrt3_2, -0.5, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(sqrt3_2, -0.5, 0.0),
            ],
            HexOrientation::X => [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, sqrt3_2, 0.0),
                Vec3::new(-0.5, sqrt3_2, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(-0.5, -sqrt3_2, 0.0),
                Vec3::new(0.5, -sqrt3_2, 0.0),
            ],
        };

        let mut best = f64::INFINITY;
        for n in &normals {
            let denom = n.x * local_dir.x + n.y * local_dir.y;
            if denom <= 0.0 {
                continue;
            }
            let projection = n.x * pos_rel.x + n.y * pos_rel.y;
            let t = (d_perp - projection) / denom;
            if t > 0.0 && t < best {
                best = t;
            }
        }

        // Axial planes.
        if local_dir.z.abs() > 0.0 {
            let centre_z =
                (current[2] as f64 - (self.n_axial as f64) * 0.5 + 0.5) * self.pitch_z;
            let half_z = self.pitch_z * 0.5;
            let target_z = if local_dir.z > 0.0 {
                centre_z + half_z
            } else {
                centre_z - half_z
            };
            let t = (target_z - local_pos.z) / local_dir.z;
            if t > 0.0 && t < best {
                best = t;
            }
        }

        best
    }

    /// Look up the overriding material for `cell_idx` at hex element
    /// `[q, r, z]`. `None` if no override applies.
    #[inline]
    pub fn material_override(&self, element: [i32; 3], cell_idx: usize) -> Option<u32> {
        let overrides = self.material_overrides.as_ref()?;
        let idx = self.axial_index(element[0], element[1], element[2])?;
        let elem_overrides = overrides.get(idx)?;
        elem_overrides.get(&cell_idx).copied()
    }
}

#[cfg(test)]
mod hex_tests {
    use super::*;

    fn unit_lattice(orientation: HexOrientation) -> HexLattice {
        let n_rings = 2;
        let n_axial = 1;
        let stride = (2 * n_rings + 1) * (2 * n_rings + 1);
        HexLattice {
            center: Vec3::new(0.0, 0.0, 0.0),
            pitch_xy: 1.0,
            pitch_z: 100.0,
            n_rings,
            n_axial,
            orientation,
            universes: vec![UniverseId(0); stride * n_axial],
            material_overrides: None,
        }
    }

    #[test]
    fn elements_per_slice_matches_ring_formula() {
        assert_eq!(HexLattice::elements_per_slice(0), 1);
        assert_eq!(HexLattice::elements_per_slice(1), 7);
        assert_eq!(HexLattice::elements_per_slice(2), 19);
        assert_eq!(HexLattice::elements_per_slice(8), 1 + 3 * 8 * 9); // 217
    }

    #[test]
    fn round_trip_element_center_flat_top() {
        let lat = unit_lattice(HexOrientation::Y);
        for &(q, r) in &[(0, 0), (1, 0), (-1, 1), (2, -1), (0, 2), (-2, 0)] {
            let centre = lat.element_center_local(q, r);
            let world = centre + lat.center;
            let elem = lat.find_element(world).expect("inside");
            assert_eq!((elem.0, elem.1), (q, r), "flat-top centre ({q},{r})");
        }
    }

    #[test]
    fn round_trip_element_center_pointy_top() {
        let lat = unit_lattice(HexOrientation::X);
        for &(q, r) in &[(0, 0), (1, 0), (-1, 1), (2, -1), (0, 2), (-2, 0)] {
            let centre = lat.element_center_local(q, r);
            let world = centre + lat.center;
            let elem = lat.find_element(world).expect("inside");
            assert_eq!((elem.0, elem.1), (q, r), "pointy-top centre ({q},{r})");
        }
    }

    #[test]
    fn distance_to_grid_centre_along_edge_normal() {
        // Particle at the centre of element (0,0,0) heading toward
        // an edge normal direction. For flat-top, edge normals are
        // at 30°/90°/etc. Heading +y (90°) for flat-top should cross
        // the top edge at distance pitch/2.
        let lat = unit_lattice(HexOrientation::Y);
        let d = lat.distance_to_grid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), [0, 0, 0]);
        assert!(
            (d - lat.pitch_xy * 0.5).abs() < 1e-12,
            "expected pitch/2 = {}, got {}",
            lat.pitch_xy * 0.5,
            d
        );

        // For pointy-top, edge normal at 0° (+x), heading +x crosses
        // right edge at distance pitch/2.
        let lat = unit_lattice(HexOrientation::X);
        let d = lat.distance_to_grid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), [0, 0, 0]);
        assert!(
            (d - lat.pitch_xy * 0.5).abs() < 1e-12,
            "expected pitch/2 = {}, got {}",
            lat.pitch_xy * 0.5,
            d
        );
    }

    #[test]
    fn find_element_inside_first_ring_flat_top() {
        // For flat-top with pitch 1.0, the +x neighbour of (0,0,0)
        // sits at (1.5*s, √3*s/2) where s = 1/√3, i.e.
        // (√3/2, 0.5). World point at that position should land in
        // element (1, 0, 0).
        let lat = unit_lattice(HexOrientation::Y);
        let s = lat.side_length();
        let neighbour = Vec3::new(1.5 * s, 3.0_f64.sqrt() * s * 0.5, 0.0);
        let elem = lat.find_element(neighbour).expect("inside");
        assert_eq!((elem.0, elem.1), (1, 0));
    }

    #[test]
    fn find_element_outside_returns_none() {
        let lat = unit_lattice(HexOrientation::Y);
        // A point far from the lattice (ring 10).
        let s = lat.side_length();
        let far = Vec3::new(20.0 * s, 0.0, 0.0);
        assert!(lat.find_element(far).is_none());
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
            material_overrides: None,
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
