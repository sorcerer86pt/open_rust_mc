//! Geometry engine — CSG with BVH acceleration.
//!
//! Design:
//!   - Surfaces as `enum` (zero-cost dispatch, no vtable)
//!   - Cells as boolean combinations of surface half-spaces
//!   - BVH over cell AABBs for O(log n) cell lookup
//!   - Ray tracing: find nearest surface crossing along a direction

pub mod aabb;
pub mod bvh;
pub mod cell;
pub mod coord;
pub mod lattice;
pub mod ray;
#[cfg(test)]
mod recursive_smoke;
pub mod scene;
pub mod shapes;
pub mod surface;
pub mod universe;

pub use aabb::Aabb;
pub use cell::{Cell, CellId};
pub use coord::{Coord, CoordStack, CoordStackExt};
pub use lattice::{HexLattice, HexLatticeId, LatticeId, RectLattice};
pub use ray::{Ray, RayHit};
pub use scene::{EffectiveFill, Geometry, GeometryError};
pub use surface::{Surface, SurfaceId};
pub use universe::{Universe, UniverseId};

/// 3D position vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let inv_len = 1.0 / self.length();
        Self {
            x: self.x * inv_len,
            y: self.y * inv_len,
            z: self.z * inv_len,
        }
    }

    #[inline]
    pub fn component_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    #[inline]
    pub fn component_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// 3×3 matrix used for universe / lattice-element rotations.
///
/// Stored row-major: `rows[i].x/y/z` is the i-th row's
/// x/y/z component. A `Mat3` representing a rotation matrix is
/// orthogonal — its transpose equals its inverse — so applying
/// `transpose()` is a cheap way to undo a rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub rows: [Vec3; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        rows: [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
    };

    /// Rotation around the Z axis by `angle_rad` (right-hand rule).
    /// `rotation_z(π/2)` maps `+x` to `+y`.
    #[inline]
    pub fn rotation_z(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            rows: [
                Vec3::new(c, -s, 0.0),
                Vec3::new(s, c, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    /// Rotation around the Y axis by `angle_rad` (right-hand rule).
    #[inline]
    pub fn rotation_y(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            rows: [
                Vec3::new(c, 0.0, s),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-s, 0.0, c),
            ],
        }
    }

    /// Rotation around the X axis by `angle_rad` (right-hand rule).
    #[inline]
    pub fn rotation_x(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            rows: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, c, -s),
                Vec3::new(0.0, s, c),
            ],
        }
    }

    /// Transpose. For an orthogonal matrix this is also the inverse.
    #[inline]
    pub fn transpose(&self) -> Self {
        let r = &self.rows;
        Self {
            rows: [
                Vec3::new(r[0].x, r[1].x, r[2].x),
                Vec3::new(r[0].y, r[1].y, r[2].y),
                Vec3::new(r[0].z, r[1].z, r[2].z),
            ],
        }
    }

    /// Apply this matrix to a column vector: `M · v`.
    #[inline]
    pub fn transform(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0].dot(v),
            self.rows[1].dot(v),
            self.rows[2].dot(v),
        )
    }

    /// Matrix-matrix product. `self * other` is "apply `other` first,
    /// then `self`" — same convention as `transform` on column vectors.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let other_t = other.transpose();
        Self {
            rows: [
                Vec3::new(
                    self.rows[0].dot(other_t.rows[0]),
                    self.rows[0].dot(other_t.rows[1]),
                    self.rows[0].dot(other_t.rows[2]),
                ),
                Vec3::new(
                    self.rows[1].dot(other_t.rows[0]),
                    self.rows[1].dot(other_t.rows[1]),
                    self.rows[1].dot(other_t.rows[2]),
                ),
                Vec3::new(
                    self.rows[2].dot(other_t.rows[0]),
                    self.rows[2].dot(other_t.rows[1]),
                    self.rows[2].dot(other_t.rows[2]),
                ),
            ],
        }
    }
}

impl Default for Mat3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod mat3_tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3, tol: f64) -> bool {
        (a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol && (a.z - b.z).abs() < tol
    }

    #[test]
    fn identity_is_a_noop() {
        let v = Vec3::new(0.7, -1.3, 4.2);
        assert_eq!(Mat3::IDENTITY.transform(v), v);
    }

    #[test]
    fn rotation_z_pi_over_two_sends_x_to_y() {
        let r = Mat3::rotation_z(std::f64::consts::FRAC_PI_2);
        assert!(approx(
            r.transform(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0),
            1e-12
        ));
        assert!(approx(
            r.transform(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(-1.0, 0.0, 0.0),
            1e-12
        ));
        assert!(approx(
            r.transform(Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(0.0, 0.0, 1.0),
            1e-12
        ));
    }

    #[test]
    fn rotation_z_60_degrees_round_trip() {
        let theta = std::f64::consts::PI / 3.0;
        let fwd = Mat3::rotation_z(theta);
        let back = Mat3::rotation_z(-theta);
        let v = Vec3::new(0.6, -0.4, 0.2);
        assert!(approx(back.transform(fwd.transform(v)), v, 1e-12));
    }

    #[test]
    fn transpose_inverts_rotation() {
        let r = Mat3::rotation_y(0.7);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let round = r.transpose().transform(r.transform(v));
        assert!(approx(round, v, 1e-12));
    }

    #[test]
    fn matmul_composes_rotations() {
        let a = Mat3::rotation_z(0.4);
        let b = Mat3::rotation_z(0.3);
        let composed = a.mul(&b);
        let direct = Mat3::rotation_z(0.7);
        let v = Vec3::new(0.3, 0.7, -0.2);
        assert!(approx(composed.transform(v), direct.transform(v), 1e-12));
    }
}
