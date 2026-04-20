//! Ray tracing — find the nearest surface crossing along a direction.
//!
//! This is the inner loop of particle transport: given a particle's
//! position and direction, find how far it can travel before hitting
//! a surface, and which cell it enters on the other side.

use super::{Cell, Surface, Vec3};

/// A ray: position + direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    /// Precomputed 1/dir for AABB tests.
    pub inv_dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        Self {
            origin,
            dir,
            inv_dir,
        }
    }
}

/// Result of a ray-geometry intersection.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Distance to the surface.
    pub distance: f64,
    /// Index of the surface that was hit.
    pub surface_idx: usize,
    /// Which cell the particle enters after crossing.
    pub next_cell_idx: Option<usize>,
}

/// Find the nearest surface crossing from a position along a direction.
///
/// Tests all surfaces in `surface_indices` (the surfaces bounding the
/// current cell) and returns the closest hit.
pub fn find_nearest_surface(
    pos: Vec3,
    dir: Vec3,
    surfaces: &[Surface],
    surface_indices: &[usize],
) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;

    for &idx in surface_indices {
        if let Some(t) = surfaces[idx].distance(pos, dir) {
            let is_closer = best.as_ref().is_none_or(|b| t < b.distance);
            if is_closer {
                best = Some(RayHit {
                    distance: t,
                    surface_idx: idx,
                    next_cell_idx: None, // resolved later
                });
            }
        }
    }

    best
}

/// Find which cell contains a given point.
///
/// Evaluates all surfaces once, then tests each cell's boolean region.
/// The BVH accelerates this by skipping cells whose AABB doesn't
/// contain the point.
pub fn find_cell(pos: Vec3, surfaces: &[Surface], cells: &[Cell]) -> Option<usize> {
    // Pre-evaluate all surfaces at this point
    let evals: Vec<f64> = surfaces.iter().map(|s| s.evaluate(pos)).collect();

    // Test each cell
    for (idx, cell) in cells.iter().enumerate() {
        // Quick AABB rejection
        if !cell.aabb.contains(pos) {
            continue;
        }
        if cell.contains(&evals) {
            return Some(idx);
        }
    }

    None
}

/// Full ray trace step: find distance to nearest surface and next cell.
///
/// This is the complete geometry step in particle transport:
/// 1. Find nearest surface crossing
/// 2. Move particle to the surface (with small nudge)
/// 3. Find which cell the particle is now in
pub fn trace_step(
    pos: Vec3,
    dir: Vec3,
    current_cell_idx: usize,
    surfaces: &[Surface],
    cells: &[Cell],
) -> Option<RayHit> {
    // Get the surfaces bounding the current cell
    let cell = &cells[current_cell_idx];
    let mut surface_indices = Vec::new();
    cell.region.surface_indices(&mut surface_indices);
    surface_indices.sort_unstable();
    surface_indices.dedup();

    // Find nearest surface
    let mut hit = find_nearest_surface(pos, dir, surfaces, &surface_indices)?;

    // Move to the crossing point (with a small nudge across)
    let cross_point = pos + dir * (hit.distance + 1e-10);

    // Find the cell on the other side
    hit.next_cell_idx = find_cell(cross_point, surfaces, cells);

    Some(hit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, CellFill, CellId};
    use crate::geometry::surface::BoundaryCondition;

    #[test]
    fn trace_godiva() {
        // Godiva: single sphere, R=8.7407
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];

        let cells = vec![
            // Fuel: inside the sphere
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // Outside: outside the sphere
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];

        // Particle at origin, heading +x
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);

        // Should be in cell 0 (fuel)
        let cell_idx = find_cell(pos, &surfaces, &cells).expect("should find cell");
        assert_eq!(cell_idx, 0);

        // Trace to the surface
        let hit = trace_step(pos, dir, 0, &surfaces, &cells).expect("should hit");
        assert!((hit.distance - 8.7407).abs() < 1e-8);
        assert_eq!(hit.surface_idx, 0);

        // After crossing, should be in the void (cell 1)
        assert_eq!(hit.next_cell_idx, Some(1));
    }

    #[test]
    fn trace_pincell() {
        // Simple pin cell: fuel cylinder + water
        let surfaces = vec![
            // 0: fuel cylinder R=0.4096
            Surface::CylinderZ {
                center_x: 0.0,
                center_y: 0.0,
                radius: 0.4096,
                bc: BoundaryCondition::Transmission,
            },
            // 1-4: reflective box (pitch=1.26)
            Surface::PlaneX {
                x0: -0.63,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: 0.63,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -0.63,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: 0.63,
                bc: BoundaryCondition::Reflective,
            },
        ];

        let cells = vec![
            // Fuel: inside cylinder
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // Water: outside cylinder, inside box
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![
                    cell::outside(0), // outside fuel
                    cell::outside(1), // x > -0.63
                    cell::inside(2),  // x < 0.63
                    cell::outside(3), // y > -0.63
                    cell::inside(4),  // y < 0.63
                ]),
                CellFill::Material(1),
            ),
        ];

        // Particle in fuel at origin
        let pos = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(find_cell(pos, &surfaces, &cells), Some(0));

        // Particle in water
        let pos_water = Vec3::new(0.5, 0.0, 0.0);
        assert_eq!(find_cell(pos_water, &surfaces, &cells), Some(1));

        // Trace from fuel center heading +x: should hit fuel cylinder at R=0.4096
        let hit =
            trace_step(pos, Vec3::new(1.0, 0.0, 0.0), 0, &surfaces, &cells).expect("should hit");
        assert!((hit.distance - 0.4096).abs() < 1e-8);
        assert_eq!(hit.next_cell_idx, Some(1)); // enters water
    }
}
