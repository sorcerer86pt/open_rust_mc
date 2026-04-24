//! Surface definitions — enum dispatch, no vtables.
//!
//! Each surface divides space into a positive and negative half-space.
//! The `evaluate` method returns > 0 for the positive side, < 0 for negative.
//! The `distance` method returns the distance along a ray to the surface.

use super::{Aabb, Vec3};

/// Unique identifier for a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceId(pub u32);

/// Boundary condition applied at a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Particle passes through (internal surface).
    Transmission,
    /// Particle reflects specularly.
    Reflective,
    /// Particle is killed (leaves the geometry).
    Vacuum,
}

/// All supported surface types — enum dispatch compiles to a jump table.
///
/// Adding a new surface type is just adding a variant + match arms.
/// No heap allocation, no dynamic dispatch overhead.
#[derive(Debug, Clone)]
pub enum Surface {
    /// Plane: Ax + By + Cz = D
    Plane {
        normal: Vec3, // (A, B, C), unit normal
        offset: f64,  // D
        bc: BoundaryCondition,
    },
    /// Axis-aligned plane: x = x0
    PlaneX { x0: f64, bc: BoundaryCondition },
    /// Axis-aligned plane: y = y0
    PlaneY { y0: f64, bc: BoundaryCondition },
    /// Axis-aligned plane: z = z0
    PlaneZ { z0: f64, bc: BoundaryCondition },
    /// Sphere: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = R^2
    Sphere {
        center: Vec3,
        radius: f64,
        bc: BoundaryCondition,
    },
    /// Cylinder along Z: (x-x0)^2 + (y-y0)^2 = R^2
    CylinderZ {
        center_x: f64,
        center_y: f64,
        radius: f64,
        bc: BoundaryCondition,
    },
    /// Cylinder along X: (y-y0)^2 + (z-z0)^2 = R^2
    CylinderX {
        center_y: f64,
        center_z: f64,
        radius: f64,
        bc: BoundaryCondition,
    },
    /// Cylinder along Y: (x-x0)^2 + (z-z0)^2 = R^2
    CylinderY {
        center_x: f64,
        center_z: f64,
        radius: f64,
        bc: BoundaryCondition,
    },
    /// Double-cone aligned with the Z axis, apex at (x0, y0, z0):
    /// (x-x0)² + (y-y0)² = r²(z-z0)²  where `r_sq = tan²(half-angle)`.
    /// The negative half-space is the interior of both sheets; cell
    /// regions typically intersect with a `PlaneZ` to pick one sheet.
    ConeZ {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryCondition,
    },
    /// Double-cone aligned with the X axis, apex at (x0, y0, z0):
    /// (y-y0)² + (z-z0)² = r²(x-x0)².
    ConeX {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryCondition,
    },
    /// Double-cone aligned with the Y axis, apex at (x0, y0, z0):
    /// (x-x0)² + (z-z0)² = r²(y-y0)².
    ConeY {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryCondition,
    },
}

/// Coincidence tolerance (same as OpenMC).
const COINCIDENCE_TOL: f64 = 1.0e-12;

impl Surface {
    /// Evaluate the surface equation at point `p`.
    /// Returns > 0 for positive half-space, < 0 for negative.
    #[inline]
    pub fn evaluate(&self, p: Vec3) -> f64 {
        match self {
            Self::Plane { normal, offset, .. } => normal.dot(p) - offset,
            Self::PlaneX { x0, .. } => p.x - x0,
            Self::PlaneY { y0, .. } => p.y - y0,
            Self::PlaneZ { z0, .. } => p.z - z0,
            Self::Sphere { center, radius, .. } => {
                let d = p - *center;
                d.dot(d) - radius * radius
            }
            Self::CylinderZ {
                center_x,
                center_y,
                radius,
                ..
            } => {
                let dx = p.x - center_x;
                let dy = p.y - center_y;
                dx.mul_add(dx, dy * dy) - radius * radius
            }
            Self::CylinderX {
                center_y,
                center_z,
                radius,
                ..
            } => {
                let dy = p.y - center_y;
                let dz = p.z - center_z;
                dy.mul_add(dy, dz * dz) - radius * radius
            }
            Self::CylinderY {
                center_x,
                center_z,
                radius,
                ..
            } => {
                let dx = p.x - center_x;
                let dz = p.z - center_z;
                dx.mul_add(dx, dz * dz) - radius * radius
            }
            Self::ConeZ { x0, y0, z0, r_sq, .. } => {
                let dx = p.x - x0;
                let dy = p.y - y0;
                let dz = p.z - z0;
                dx.mul_add(dx, dy * dy) - r_sq * dz * dz
            }
            Self::ConeX { x0, y0, z0, r_sq, .. } => {
                let dx = p.x - x0;
                let dy = p.y - y0;
                let dz = p.z - z0;
                dy.mul_add(dy, dz * dz) - r_sq * dx * dx
            }
            Self::ConeY { x0, y0, z0, r_sq, .. } => {
                let dx = p.x - x0;
                let dy = p.y - y0;
                let dz = p.z - z0;
                dx.mul_add(dx, dz * dz) - r_sq * dy * dy
            }
        }
    }

    /// Compute the distance from point `p` along direction `d` to this surface.
    /// Returns `None` if the ray doesn't intersect or only intersects behind.
    #[inline]
    pub fn distance(&self, p: Vec3, dir: Vec3) -> Option<f64> {
        match self {
            Self::Plane { normal, offset, .. } => {
                let denom = normal.dot(dir);
                if denom.abs() < COINCIDENCE_TOL {
                    return None;
                }
                let t = (offset - normal.dot(p)) / denom;
                if t > COINCIDENCE_TOL { Some(t) } else { None }
            }
            Self::PlaneX { x0, .. } => {
                if dir.x.abs() < COINCIDENCE_TOL {
                    return None;
                }
                let t = (x0 - p.x) / dir.x;
                if t > COINCIDENCE_TOL { Some(t) } else { None }
            }
            Self::PlaneY { y0, .. } => {
                if dir.y.abs() < COINCIDENCE_TOL {
                    return None;
                }
                let t = (y0 - p.y) / dir.y;
                if t > COINCIDENCE_TOL { Some(t) } else { None }
            }
            Self::PlaneZ { z0, .. } => {
                if dir.z.abs() < COINCIDENCE_TOL {
                    return None;
                }
                let t = (z0 - p.z) / dir.z;
                if t > COINCIDENCE_TOL { Some(t) } else { None }
            }
            Self::Sphere { center, radius, .. } => sphere_intersect(p, dir, *center, *radius),
            Self::CylinderZ {
                center_x,
                center_y,
                radius,
                ..
            } => cylinder_z_intersect(p, dir, *center_x, *center_y, *radius),
            Self::CylinderX {
                center_y,
                center_z,
                radius,
                ..
            } => {
                // Rotate coordinates: X-cylinder is Z-cylinder in rotated frame
                let p_rot = Vec3::new(p.y, p.z, p.x);
                let d_rot = Vec3::new(dir.y, dir.z, dir.x);
                cylinder_z_intersect(p_rot, d_rot, *center_y, *center_z, *radius)
            }
            Self::CylinderY {
                center_x,
                center_z,
                radius,
                ..
            } => {
                let p_rot = Vec3::new(p.x, p.z, p.y);
                let d_rot = Vec3::new(dir.x, dir.z, dir.y);
                cylinder_z_intersect(p_rot, d_rot, *center_x, *center_z, *radius)
            }
            Self::ConeZ { x0, y0, z0, r_sq, .. } => {
                cone_z_intersect(p, dir, *x0, *y0, *z0, *r_sq)
            }
            Self::ConeX { x0, y0, z0, r_sq, .. } => {
                // X-axis cone = Z-axis cone with (y, z, x) → (x, y, z).
                let p_rot = Vec3::new(p.y, p.z, p.x);
                let d_rot = Vec3::new(dir.y, dir.z, dir.x);
                cone_z_intersect(p_rot, d_rot, *y0, *z0, *x0, *r_sq)
            }
            Self::ConeY { x0, y0, z0, r_sq, .. } => {
                // Y-axis cone: swap y and z so y becomes the axis.
                let p_rot = Vec3::new(p.x, p.z, p.y);
                let d_rot = Vec3::new(dir.x, dir.z, dir.y);
                cone_z_intersect(p_rot, d_rot, *x0, *z0, *y0, *r_sq)
            }
        }
    }

    /// Get the boundary condition for this surface.
    #[inline]
    pub fn boundary_condition(&self) -> BoundaryCondition {
        match self {
            Self::Plane { bc, .. }
            | Self::PlaneX { bc, .. }
            | Self::PlaneY { bc, .. }
            | Self::PlaneZ { bc, .. }
            | Self::Sphere { bc, .. }
            | Self::CylinderZ { bc, .. }
            | Self::CylinderX { bc, .. }
            | Self::CylinderY { bc, .. }
            | Self::ConeZ { bc, .. }
            | Self::ConeX { bc, .. }
            | Self::ConeY { bc, .. } => *bc,
        }
    }

    /// Normal vector at a point on the surface.
    #[inline]
    pub fn normal_at(&self, p: Vec3) -> Vec3 {
        match self {
            Self::Plane { normal, .. } => *normal,
            Self::PlaneX { .. } => Vec3::new(1.0, 0.0, 0.0),
            Self::PlaneY { .. } => Vec3::new(0.0, 1.0, 0.0),
            Self::PlaneZ { .. } => Vec3::new(0.0, 0.0, 1.0),
            Self::Sphere { center, .. } => (p - *center).normalized(),
            Self::CylinderZ {
                center_x, center_y, ..
            } => Vec3::new(p.x - center_x, p.y - center_y, 0.0).normalized(),
            Self::CylinderX {
                center_y, center_z, ..
            } => Vec3::new(0.0, p.y - center_y, p.z - center_z).normalized(),
            Self::CylinderY {
                center_x, center_z, ..
            } => Vec3::new(p.x - center_x, 0.0, p.z - center_z).normalized(),
            Self::ConeZ { x0, y0, z0, r_sq, .. } => {
                // ∇f = (2(x-x0), 2(y-y0), -2·r²(z-z0)); dropping the
                // factor of 2 before normalising.
                Vec3::new(p.x - x0, p.y - y0, -r_sq * (p.z - z0)).normalized()
            }
            Self::ConeX { x0, y0, z0, r_sq, .. } => {
                Vec3::new(-r_sq * (p.x - x0), p.y - y0, p.z - z0).normalized()
            }
            Self::ConeY { x0, y0, z0, r_sq, .. } => {
                Vec3::new(p.x - x0, -r_sq * (p.y - y0), p.z - z0).normalized()
            }
        }
    }

    /// Compute an AABB for this surface (may be infinite for planes).
    pub fn aabb(&self) -> Aabb {
        match self {
            Self::Sphere { center, radius, .. } => {
                let r = Vec3::new(*radius, *radius, *radius);
                Aabb {
                    min: *center - r,
                    max: *center + r,
                }
            }
            Self::CylinderZ {
                center_x,
                center_y,
                radius,
                ..
            } => Aabb {
                min: Vec3::new(center_x - radius, center_y - radius, f64::NEG_INFINITY),
                max: Vec3::new(center_x + radius, center_y + radius, f64::INFINITY),
            },
            // Planes and other infinite surfaces get infinite AABBs
            _ => Aabb::INFINITE,
        }
    }
}

// ── Ray-sphere intersection ────────────────────────────────────────────────

#[inline]
fn sphere_intersect(p: Vec3, dir: Vec3, center: Vec3, radius: f64) -> Option<f64> {
    let oc = p - center;
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let inv_2a = 0.5 / a;

    // Try the nearer intersection first
    let t1 = (-b - sqrt_disc) * inv_2a;
    if t1 > COINCIDENCE_TOL {
        return Some(t1);
    }

    // If we're inside the sphere, take the far intersection
    let t2 = (-b + sqrt_disc) * inv_2a;
    if t2 > COINCIDENCE_TOL {
        return Some(t2);
    }

    None
}

// ── Ray-cylinder (Z-aligned) intersection ──────────────────────────────────

#[inline]
fn cylinder_z_intersect(p: Vec3, dir: Vec3, cx: f64, cy: f64, r: f64) -> Option<f64> {
    // Project to 2D (ignore z)
    let ox = p.x - cx;
    let oy = p.y - cy;

    let a = dir.x.mul_add(dir.x, dir.y * dir.y);
    if a < COINCIDENCE_TOL {
        return None; // Ray parallel to cylinder axis
    }

    let b = 2.0 * ox.mul_add(dir.x, oy * dir.y);
    let c = ox.mul_add(ox, oy * oy) - r * r;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let inv_2a = 0.5 / a;

    let t1 = (-b - sqrt_disc) * inv_2a;
    if t1 > COINCIDENCE_TOL {
        return Some(t1);
    }

    let t2 = (-b + sqrt_disc) * inv_2a;
    if t2 > COINCIDENCE_TOL {
        return Some(t2);
    }

    None
}

// ── Ray-cone (Z-aligned double cone) intersection ─────────────────────────

#[inline]
fn cone_z_intersect(p: Vec3, dir: Vec3, x0: f64, y0: f64, z0: f64, r_sq: f64) -> Option<f64> {
    // f(p + t·d) = (Δx + t·dx)² + (Δy + t·dy)² − r²(Δz + t·dz)²
    //            = A·t² + B·t + C
    let dx = p.x - x0;
    let dy = p.y - y0;
    let dz = p.z - z0;

    let a = dir.x.mul_add(dir.x, dir.y * dir.y) - r_sq * dir.z * dir.z;
    let b = 2.0 * (dx.mul_add(dir.x, dy * dir.y) - r_sq * dz * dir.z);
    let c = dx.mul_add(dx, dy * dy) - r_sq * dz * dz;

    quadratic_smallest_positive(a, b, c)
}

/// Smallest `t > COINCIDENCE_TOL` satisfying `a·t² + b·t + c = 0`, or
/// `None` if there's no such root. Handles the degenerate linear case
/// when `|a| < COINCIDENCE_TOL`, which for cones happens when the ray
/// is parallel to a generatrix.
#[inline]
fn quadratic_smallest_positive(a: f64, b: f64, c: f64) -> Option<f64> {
    if a.abs() < COINCIDENCE_TOL {
        // Linear: b·t + c = 0.
        if b.abs() < COINCIDENCE_TOL {
            return None;
        }
        let t = -c / b;
        return if t > COINCIDENCE_TOL { Some(t) } else { None };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let inv_2a = 0.5 / a;
    let t1 = (-b - sqrt_disc) * inv_2a;
    let t2 = (-b + sqrt_disc) * inv_2a;
    // We want the smallest t > COINCIDENCE_TOL. Note that with a < 0
    // (can happen for cones depending on ray direction) the two roots
    // swap ordering, so check both.
    let (tmin, tmax) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
    if tmin > COINCIDENCE_TOL {
        Some(tmin)
    } else if tmax > COINCIDENCE_TOL {
        Some(tmax)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_evaluate_inside_outside() {
        let s = Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 5.0,
            bc: BoundaryCondition::Vacuum,
        };
        // Inside
        assert!(s.evaluate(Vec3::new(1.0, 0.0, 0.0)) < 0.0);
        // On surface
        assert!((s.evaluate(Vec3::new(5.0, 0.0, 0.0))).abs() < 1e-10);
        // Outside
        assert!(s.evaluate(Vec3::new(6.0, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn sphere_distance_from_outside() {
        let s = Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 5.0,
            bc: BoundaryCondition::Vacuum,
        };
        let p = Vec3::new(-10.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let t = s.distance(p, d).expect("should intersect");
        assert!((t - 5.0).abs() < 1e-10);
    }

    #[test]
    fn sphere_distance_from_inside() {
        let s = Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 5.0,
            bc: BoundaryCondition::Vacuum,
        };
        let p = Vec3::new(0.0, 0.0, 0.0); // at center
        let d = Vec3::new(1.0, 0.0, 0.0);
        let t = s.distance(p, d).expect("should intersect");
        assert!((t - 5.0).abs() < 1e-10);
    }

    #[test]
    fn plane_x_distance() {
        let s = Surface::PlaneX {
            x0: 3.0,
            bc: BoundaryCondition::Vacuum,
        };
        let p = Vec3::new(0.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let t = s.distance(p, d).expect("should intersect");
        assert!((t - 3.0).abs() < 1e-10);
    }

    #[test]
    fn cylinder_z_distance() {
        let s = Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: 1.0,
            bc: BoundaryCondition::Vacuum,
        };
        let p = Vec3::new(-5.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let t = s.distance(p, d).expect("should intersect");
        assert!((t - 4.0).abs() < 1e-10);
    }

    #[test]
    fn cone_z_45deg_side_hit() {
        // Double-cone apex at origin, 45° half-angle → r² = tan²(45°) = 1.
        // Particle at (2, 0, -5) heading +z hits at t = 3 (position
        // (2, 0, -2), where f = 2² + 0 − 1·(-2)² = 0).
        let s = Surface::ConeZ {
            x0: 0.0, y0: 0.0, z0: 0.0, r_sq: 1.0,
            bc: BoundaryCondition::Transmission,
        };
        let p = Vec3::new(2.0, 0.0, -5.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        let t = s.distance(p, d).expect("should hit");
        assert!((t - 3.0).abs() < 1.0e-10, "t = {t}");
        // Sign check: (2,0,-5) is outside because f = 4 − 25 = −21, so
        // evaluate is negative. Approach from negative half-space.
        assert!(s.evaluate(p) < 0.0);
    }

    #[test]
    fn cone_z_miss_when_inside_generatrix_parallel() {
        // 45° cone apex at origin. Ray parallel to a generatrix
        // direction (1, 0, 1)/√2 starting at (1, 0, 0). With A = 0
        // (exactly parallel), the quadratic degenerates to linear.
        let s = Surface::ConeZ {
            x0: 0.0, y0: 0.0, z0: 0.0, r_sq: 1.0,
            bc: BoundaryCondition::Transmission,
        };
        let p = Vec3::new(1.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 1.0).normalized();
        // f at p: 1 − 0 = 1 > 0 (outside), heading roughly along the
        // cone surface → they diverge; expect no hit or a distant one.
        let _ = s.distance(p, d); // just ensure no panic on the linear branch
    }

    #[test]
    fn cone_x_equivalent_under_axis_swap() {
        // ConeX with same parameters as ConeZ should match when we
        // swap the test ray through the same rotation.
        let s_z = Surface::ConeZ {
            x0: 0.0, y0: 0.0, z0: 0.0, r_sq: 1.0,
            bc: BoundaryCondition::Transmission,
        };
        let s_x = Surface::ConeX {
            x0: 0.0, y0: 0.0, z0: 0.0, r_sq: 1.0,
            bc: BoundaryCondition::Transmission,
        };
        // Z-test: ray at (2, 0, -5) heading +z hits at t = 3.
        // X-equivalent: ray at (-5, 0, 2) heading +x.
        let t_z = s_z.distance(Vec3::new(2.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let t_x = s_x.distance(Vec3::new(-5.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(t_z.is_some() && t_x.is_some());
        assert!((t_z.unwrap() - t_x.unwrap()).abs() < 1.0e-10);
    }

    #[test]
    fn cone_normal_points_outward() {
        // 45° cone apex at origin. At point (1, 0, 1) on the +z sheet,
        // outward normal should point roughly in the radial-outward
        // direction with a slope matching the cone opening.
        let s = Surface::ConeZ {
            x0: 0.0, y0: 0.0, z0: 0.0, r_sq: 1.0,
            bc: BoundaryCondition::Transmission,
        };
        let n = s.normal_at(Vec3::new(1.0, 0.0, 1.0));
        // Should be in the x>0 direction and have a z component.
        assert!(n.x > 0.5, "n.x = {}", n.x);
        // For 45° cone at (1,0,1), ∇f = (2, 0, -2), normalised to (1/√2, 0, -1/√2).
        assert!((n.x - 1.0 / (2.0_f64).sqrt()).abs() < 1.0e-10);
        assert!((n.z + 1.0 / (2.0_f64).sqrt()).abs() < 1.0e-10);
    }

    #[test]
    fn godiva_sphere() {
        // Godiva: sphere R=8.7407 cm
        let s = Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        };
        // Particle at origin heading outward
        let t = s
            .distance(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0))
            .expect("should hit");
        assert!((t - 8.7407).abs() < 1e-10);
    }
}
