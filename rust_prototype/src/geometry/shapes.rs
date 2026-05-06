//! Geometry shape builders for common patterns.
//!
//! Each builder returns the `Surface`s it generated and an
//! "inside-the-shape" `Region` that uses surface indices starting at
//! the caller-supplied `surface_offset`. The caller appends the
//! surfaces to its own `Vec<Surface>` (in order, starting at the
//! offset they passed) and slots the region into a `Cell`.
//!
//! Pattern:
//! ```ignore
//! let mut surfaces: Vec<Surface> = vec![/* ...pin cylinders... */];
//! let (box_surfaces, box_region) =
//!     shapes::rect_box([0.63, 0.63, 0.63], BC::Reflective, surfaces.len());
//! surfaces.extend(box_surfaces);
//! let outer_cell = Cell::new(CellId(7), box_region, CellFill::Lattice(0));
//! ```
//!
//! No magic surface indices, no copy-paste of the same 6 PlaneX/PlaneY/PlaneZ
//! literal block in every binary.

use crate::geometry::cell::Region;
use crate::geometry::lattice::HexOrientation;
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::Vec3;

/// Result of a shape builder: the new surfaces and the
/// inside-the-shape Region using indices `[surface_offset, surface_offset + surfaces.len())`.
#[derive(Debug, Clone)]
pub struct Shape {
    pub surfaces: Vec<Surface>,
    pub inside: Region,
}

/// Axis-aligned rectangular box centred at origin with half-extents
/// `[half_x, half_y, half_z]`. Generates 6 planes (PlaneX±, PlaneY±,
/// PlaneZ±) all carrying `bc`, and the region "inside the box".
///
/// Surface indices are assigned in this order, starting at
/// `surface_offset`:
///   0: PlaneX(-half_x)
///   1: PlaneX(+half_x)
///   2: PlaneY(-half_y)
///   3: PlaneY(+half_y)
///   4: PlaneZ(-half_z)
///   5: PlaneZ(+half_z)
pub fn rect_box(half: [f64; 3], bc: BoundaryCondition, surface_offset: usize) -> Shape {
    let surfaces = vec![
        Surface::PlaneX { x0: -half[0], bc },
        Surface::PlaneX { x0: half[0], bc },
        Surface::PlaneY { y0: -half[1], bc },
        Surface::PlaneY { y0: half[1], bc },
        Surface::PlaneZ { z0: -half[2], bc },
        Surface::PlaneZ { z0: half[2], bc },
    ];
    let s = surface_offset;
    // outside(-half) ∩ inside(+half) on each axis = inside the box.
    let inside = crate::geometry::cell::intersect_all(vec![
        crate::geometry::cell::outside(s),
        crate::geometry::cell::inside(s + 1),
        crate::geometry::cell::outside(s + 2),
        crate::geometry::cell::inside(s + 3),
        crate::geometry::cell::outside(s + 4),
        crate::geometry::cell::inside(s + 5),
    ]);
    Shape { surfaces, inside }
}

/// Outward unit normals for the 6 sides of a regular hex centred at
/// origin. Order is counter-clockwise starting at the smallest
/// positive angle. Z is always 0.
///
/// For `HexOrientation::Y` (flat-top) the side midpoints are at
/// 30°, 90°, 150°, 210°, 270°, 330° from +x.
/// For `HexOrientation::X` (pointy-top) they're at 0°, 60°, 120°,
/// 180°, 240°, 300°.
pub fn hex_side_normals(orientation: HexOrientation) -> [Vec3; 6] {
    let s30 = 0.5_f64;
    let c30 = 3.0_f64.sqrt() * 0.5;
    match orientation {
        HexOrientation::Y => [
            Vec3::new(c30, s30, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-c30, s30, 0.0),
            Vec3::new(-c30, -s30, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(c30, -s30, 0.0),
        ],
        HexOrientation::X => [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(s30, c30, 0.0),
            Vec3::new(-s30, c30, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-s30, -c30, 0.0),
            Vec3::new(s30, -c30, 0.0),
        ],
    }
}

/// Hex-shaped 3D region: 6 reflective hex sides plus 2 z-planes.
/// Sized to enclose an `n_rings` hex tessellation of the given
/// `pitch` (centre-to-centre distance) — perpendicular inradius is
/// `(n_rings + 0.5) * pitch` so every point inside the region maps
/// to a valid axial cell with `ring ≤ n_rings + 1` (the +1 buffers
/// the cube-rounding tie at the edge — see hex_minicore for why).
///
/// Surface indices in order, starting at `surface_offset`:
///   0..=5: 6 hex side planes (with `xy_bc`)
///   6: PlaneZ(-z_half) (with `z_bc`)
///   7: PlaneZ(+z_half) (with `z_bc`)
pub fn hex_boundary(
    n_rings: usize,
    pitch: f64,
    orientation: HexOrientation,
    xy_bc: BoundaryCondition,
    z_half: f64,
    z_bc: BoundaryCondition,
    surface_offset: usize,
) -> Shape {
    let inradius = (n_rings as f64 + 0.5) * pitch;
    let normals = hex_side_normals(orientation);
    let mut surfaces: Vec<Surface> = normals
        .iter()
        .map(|&n| Surface::Plane {
            normal: n,
            offset: inradius,
            bc: xy_bc,
        })
        .collect();
    surfaces.push(Surface::PlaneZ {
        z0: -z_half,
        bc: z_bc,
    });
    surfaces.push(Surface::PlaneZ {
        z0: z_half,
        bc: z_bc,
    });
    let s = surface_offset;
    let inside = crate::geometry::cell::intersect_all(vec![
        crate::geometry::cell::inside(s),
        crate::geometry::cell::inside(s + 1),
        crate::geometry::cell::inside(s + 2),
        crate::geometry::cell::inside(s + 3),
        crate::geometry::cell::inside(s + 4),
        crate::geometry::cell::inside(s + 5),
        crate::geometry::cell::outside(s + 6),
        crate::geometry::cell::inside(s + 7),
    ]);
    Shape { surfaces, inside }
}

/// Concentric cylinders along Z centred at `(center_x, center_y)`.
/// Used as the building block for fuel pins, guide tubes, control rods.
/// Returns N `CylinderZ` surfaces with `bc = Transmission`. The
/// caller composes the cell regions via `cell::inside`,
/// `cell::between`, `cell::outside`.
pub fn pin_cylinders(center_x: f64, center_y: f64, radii: &[f64]) -> Vec<Surface> {
    radii
        .iter()
        .map(|&r| Surface::CylinderZ {
            center_x,
            center_y,
            radius: r,
            bc: BoundaryCondition::Transmission,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::Region;

    fn evaluate_region(region: &Region, surfaces: &[Surface], pos: Vec3) -> bool {
        let mut evals = vec![0.0; surfaces.len()];
        for (i, s) in surfaces.iter().enumerate() {
            evals[i] = s.evaluate(pos);
        }
        region.contains(&evals)
    }

    #[test]
    fn rect_box_inside_origin_outside_corner() {
        let shape = rect_box([1.0, 1.0, 1.0], BoundaryCondition::Reflective, 0);
        assert_eq!(shape.surfaces.len(), 6);
        assert!(evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(0.0, 0.0, 0.0),
        ));
        assert!(!evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(2.0, 0.0, 0.0),
        ));
        assert!(!evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(0.0, 2.0, 0.0),
        ));
    }

    #[test]
    fn rect_box_offset_indices_align_correctly() {
        // surface_offset = 4 (caller has 4 prior surfaces).
        let shape = rect_box([1.0, 1.0, 1.0], BoundaryCondition::Reflective, 4);
        // The region's surface indices are 4..=9. Build a synthetic
        // surfaces vec with 4 dummy + the 6 box planes.
        let mut all_surfaces: Vec<Surface> = (0..4)
            .map(|i| Surface::PlaneX {
                x0: 100.0 + i as f64,
                bc: BoundaryCondition::Vacuum,
            })
            .collect();
        all_surfaces.extend(shape.surfaces);
        assert!(evaluate_region(
            &shape.inside,
            &all_surfaces,
            Vec3::new(0.0, 0.0, 0.0),
        ));
        assert!(!evaluate_region(
            &shape.inside,
            &all_surfaces,
            Vec3::new(2.0, 0.0, 0.0),
        ));
    }

    #[test]
    fn hex_boundary_origin_inside_corners_outside() {
        let pitch = 1.0;
        let rings = 1;
        let shape = hex_boundary(
            rings,
            pitch,
            HexOrientation::Y,
            BoundaryCondition::Reflective,
            5.0,
            BoundaryCondition::Reflective,
            0,
        );
        assert_eq!(shape.surfaces.len(), 8); // 6 sides + 2 z planes.

        // Inradius = (1 + 0.5) * 1.0 = 1.5; circumradius = 1.5 * 2/sqrt(3) ≈ 1.732.
        // Origin is well inside.
        assert!(evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(0.0, 0.0, 0.0),
        ));
        // (3, 3, 0) is well outside the hex.
        assert!(!evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(3.0, 3.0, 0.0),
        ));
        // (1.4, 0, 0) is inside (within circumradius and inside the
        // hex shape on this axis).
        assert!(evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(1.4, 0.0, 0.0),
        ));
        // (0, 1.6, 0) is just outside the hex's top edge (inradius = 1.5).
        assert!(!evaluate_region(
            &shape.inside,
            &shape.surfaces,
            Vec3::new(0.0, 1.6, 0.0),
        ));
    }

    #[test]
    fn hex_side_normals_point_outward() {
        // For both orientations, every normal must satisfy
        // `dot(normal, edge_midpoint_position) > 0` — i.e. the normal
        // points away from the origin, in the same hemisphere as the
        // edge midpoint.
        for orientation in [HexOrientation::Y, HexOrientation::X] {
            let normals = hex_side_normals(orientation);
            for (i, n) in normals.iter().enumerate() {
                let len2 = n.x * n.x + n.y * n.y + n.z * n.z;
                assert!((len2 - 1.0).abs() < 1e-12, "normal {i} not unit");
                // z component is 0 (hex is in xy plane).
                assert_eq!(n.z, 0.0);
            }
            // Adjacent normals should be 60° apart → dot product = 0.5.
            for i in 0..6 {
                let next = (i + 1) % 6;
                let d = normals[i].x * normals[next].x + normals[i].y * normals[next].y;
                assert!(
                    (d - 0.5).abs() < 1e-9,
                    "adjacent dot {} != 0.5 (got {d})",
                    i
                );
            }
        }
    }

    #[test]
    fn pin_cylinders_returns_correct_count_and_radii() {
        let cylinders = pin_cylinders(0.0, 0.0, &[0.41, 0.42, 0.475]);
        assert_eq!(cylinders.len(), 3);
        for (i, s) in cylinders.iter().enumerate() {
            match s {
                Surface::CylinderZ {
                    center_x,
                    center_y,
                    radius,
                    bc,
                } => {
                    assert_eq!(*center_x, 0.0);
                    assert_eq!(*center_y, 0.0);
                    assert!((*radius - [0.41, 0.42, 0.475][i]).abs() < 1e-12);
                    assert!(matches!(bc, BoundaryCondition::Transmission));
                }
                _ => panic!("expected CylinderZ"),
            }
        }
    }
}
