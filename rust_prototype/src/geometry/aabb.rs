//! Axis-Aligned Bounding Box — the core of BVH acceleration.
//!
//! AABB-ray intersection is branchless and SIMD-friendly.
//! This is the first test in the BVH traversal: if the ray doesn't
//! hit the box, skip the entire subtree.

use super::Vec3;

/// Axis-Aligned Bounding Box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// An infinitely large AABB (used for infinite surfaces like planes).
    pub const INFINITE: Self = Self {
        min: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        max: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
    };

    /// Create a new AABB from min and max corners.
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Merge two AABBs into one that encloses both.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.component_min(other.min),
            max: self.max.component_max(other.max),
        }
    }

    /// Test if a point is inside the AABB.
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Surface area (used for SAH cost in BVH construction).
    #[inline]
    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Center of the AABB.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Ray-AABB intersection using the slab method.
    ///
    /// Returns `true` if the ray hits the box at any distance > 0.
    /// This is the hot path for BVH traversal — branchless and SIMD-friendly.
    #[inline]
    pub fn ray_intersects(&self, origin: Vec3, dir: Vec3) -> bool {
        // Inverse direction (precomputed per ray in production)
        let inv_d = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        self.ray_intersects_inv(origin, inv_d)
    }

    /// Ray-AABB intersection with precomputed inverse direction.
    #[inline]
    pub fn ray_intersects_inv(&self, origin: Vec3, inv_dir: Vec3) -> bool {
        let t1x = (self.min.x - origin.x) * inv_dir.x;
        let t2x = (self.max.x - origin.x) * inv_dir.x;
        let t1y = (self.min.y - origin.y) * inv_dir.y;
        let t2y = (self.max.y - origin.y) * inv_dir.y;
        let t1z = (self.min.z - origin.z) * inv_dir.z;
        let t2z = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
        let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

        tmax >= tmin.max(0.0)
    }

    /// Ray-AABB intersection returning (tmin, tmax) for ordered traversal.
    #[inline]
    pub fn ray_interval(&self, origin: Vec3, inv_dir: Vec3) -> (f64, f64) {
        let t1x = (self.min.x - origin.x) * inv_dir.x;
        let t2x = (self.max.x - origin.x) * inv_dir.x;
        let t1y = (self.min.y - origin.y) * inv_dir.y;
        let t2y = (self.max.y - origin.y) * inv_dir.y;
        let t1z = (self.min.z - origin.z) * inv_dir.z;
        let t2z = (self.max.z - origin.z) * inv_dir.z;

        let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
        let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

        (tmin, tmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_contains() {
        let b = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(b.contains(Vec3::new(0.0, 0.0, 0.0)));
        assert!(!b.contains(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn aabb_ray_hit() {
        let b = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Ray from outside, heading toward box
        assert!(b.ray_intersects(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
        // Ray from outside, heading away
        assert!(!b.ray_intersects(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)));
        // Ray from inside
        assert!(b.ray_intersects(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn aabb_ray_miss() {
        let b = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Ray passes above the box
        assert!(!b.ray_intersects(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
    }
}
