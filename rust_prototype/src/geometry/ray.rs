//! Ray tracing — find the nearest surface crossing along a direction.
//!
//! This is the inner loop of particle transport: given a particle's
//! position and direction, find how far it can travel before hitting
//! a surface, and which cell it enters on the other side.

use super::cell::CellFill;
use super::coord::{Coord, CoordStack};
use super::surface::BoundaryCondition;
use super::{Cell, Geometry, LatticeId, Mat3, Surface, UniverseId, Vec3};
use smallvec::SmallVec;

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

/// Maximum CoordStack depth before `find_cell_recursive` gives up.
/// Real-world geometries top out around 4–5; the limit catches infinite
/// recursion bugs (a Universe-fill cell whose universe contains itself).
pub const MAX_COORD_DEPTH: usize = 16;

/// Resolve a world-space point into a full coordinate stack through
/// the recursive geometry.
///
/// Walks down from the root universe, picking the cell that contains
/// the point at each level, and recursing through `CellFill::Universe`
/// or `CellFill::Lattice` until a `Material` or `Void` cell is reached.
///
/// Returns `None` if no cell in the root (or any descended) universe
/// contains the point — i.e. the particle has leaked out of the
/// geometry.
pub fn find_cell_recursive(world_pos: Vec3, geom: &Geometry) -> Option<CoordStack> {
    let mut stack = CoordStack::new();
    let mut current_universe = geom.root_universe;
    let mut next_offset = Vec3::new(0.0, 0.0, 0.0);
    let mut next_rotation: Option<Mat3> = None;
    let mut next_lattice: Option<(LatticeId, [i32; 3])> = None;
    let mut next_hex_lattice: Option<(crate::geometry::HexLatticeId, [i32; 3])> = None;
    let mut local_pos = world_pos;

    // Pre-allocate a single evals buffer shared across descents — surfaces
    // re-evaluate at each new local_pos, but we keep the Vec around to
    // avoid per-iteration allocs.
    let mut evals: Vec<f64> = vec![0.0; geom.surfaces.len()];

    loop {
        if stack.len() >= MAX_COORD_DEPTH {
            return None;
        }

        // Apply this frame's offset (translation), then rotation, to
        // transform from parent local coords to this frame's local
        // coords: `this_local = R · (parent_local − offset)`.
        local_pos = local_pos - next_offset;
        if let Some(r) = next_rotation {
            local_pos = r.transform(local_pos);
        }

        // Refresh only the surface evaluations relevant to this universe.
        // Cells in this universe's `cell_indices` only ever reference
        // surfaces in `universe_surfaces[u]`, so stale entries for
        // surfaces in other universes are harmless. For nested
        // geometries this is typically a 2–10x reduction in surface
        // evaluations per descent.
        let u_idx = current_universe.0 as usize;
        for &s_idx in &geom.universe_surfaces[u_idx] {
            evals[s_idx] = geom.surfaces[s_idx].evaluate(local_pos);
        }

        // Cell-find: BVH if this universe has finite AABBs on all its
        // cells; linear scan otherwise.
        let universe = &geom.universes[u_idx];
        let cell_idx = if let Some(bvh) = &geom.universe_bvhs[u_idx] {
            bvh.find_cell_with_evals(local_pos, &evals, &geom.cells)?
        } else {
            universe.cell_indices.iter().copied().find(|&idx| {
                let cell = &geom.cells[idx];
                cell.aabb.contains(local_pos) && cell.contains(&evals)
            })?
        };

        stack.push(Coord {
            universe: current_universe,
            cell_idx: cell_idx as u32,
            lattice: next_lattice,
            hex_lattice: next_hex_lattice,
            offset: next_offset,
            rotation: next_rotation,
        });

        match geom.cells[cell_idx].fill {
            CellFill::Material(_) | CellFill::Void => return Some(stack),
            CellFill::Universe(u) => {
                current_universe = UniverseId(u);
                next_offset = Vec3::new(0.0, 0.0, 0.0);
                next_rotation = geom.cells[cell_idx].rotation;
                next_lattice = None;
                next_hex_lattice = None;
            }
            CellFill::Lattice(l) => {
                let lattice_id = LatticeId(l);
                let lattice = geom.lattice(lattice_id);
                let (ix, iy, iz) = lattice.find_element(local_pos)?;
                let element_universe = lattice.universe_at(ix, iy, iz);
                let local_in_element = lattice.local_position(local_pos, ix, iy, iz);
                let element_offset = local_pos - local_in_element;

                current_universe = element_universe;
                next_offset = element_offset;
                next_rotation = geom.cells[cell_idx].rotation;
                next_lattice = Some((lattice_id, [ix as i32, iy as i32, iz as i32]));
                next_hex_lattice = None;
            }
            CellFill::HexLattice(h) => {
                let hex_id = crate::geometry::HexLatticeId(h);
                let hex = geom.hex_lattice(hex_id);
                let (q, r, z) = hex.find_element(local_pos)?;
                let element_universe = hex.universe_at(q, r, z);
                let local_in_element = hex.local_position(local_pos, q, r, z);
                let element_offset = local_pos - local_in_element;

                current_universe = element_universe;
                next_offset = element_offset;
                next_rotation = geom.cells[cell_idx].rotation;
                next_lattice = None;
                next_hex_lattice = Some((hex_id, [q, r, z]));
            }
        }
    }
}

/// Outcome of a recursive ray-trace step: distance, optional surface,
/// re-resolved coordinate stack at the new position, and the boundary
/// condition of the surface hit (if any).
///
/// `next_stack` is `None` if the particle leaks out of the geometry
/// after the step (no cell contains the new world position).
#[derive(Debug, Clone)]
pub struct RecursiveHit {
    pub distance: f64,
    /// Surface that was crossed; `None` if the crossing was a lattice
    /// grid line.
    pub surface_idx: Option<usize>,
    pub next_stack: Option<CoordStack>,
    pub bc: BoundaryCondition,
}

/// Compute the local position of the particle at every frame on the
/// stack. `out[i]` is `world_pos` transformed into frame `i`'s local
/// coordinates by walking offsets and rotations from the root down;
/// the deepest frame is `out.last()`.
fn local_positions(stack: &CoordStack, world_pos: Vec3) -> SmallVec<[Vec3; 4]> {
    let mut acc = world_pos;
    let mut out: SmallVec<[Vec3; 4]> = SmallVec::new();
    for frame in stack {
        acc = acc - frame.offset;
        if let Some(r) = frame.rotation {
            acc = r.transform(acc);
        }
        out.push(acc);
    }
    out
}

/// Same idea as `local_positions` but for the direction vector. Only
/// the rotation cascade applies — translations don't affect direction.
fn local_directions(stack: &CoordStack, world_dir: Vec3) -> SmallVec<[Vec3; 4]> {
    let mut acc = world_dir;
    let mut out: SmallVec<[Vec3; 4]> = SmallVec::new();
    for frame in stack {
        if let Some(r) = frame.rotation {
            acc = r.transform(acc);
        }
        out.push(acc);
    }
    out
}

/// Recursive ray-trace step.
///
/// Considers three sources of crossings:
///   1. Surfaces bounding the deepest cell's region (in deepest local
///      frame).
///   2. Surfaces bounding any *parent* cell's region — these let the
///      particle leave a sub-universe via the parent's boundary.
///   3. Grid planes of any lattice frame on the stack — when the
///      particle moves between lattice elements.
///
/// Takes the minimum positive distance, advances `world_pos` by that
/// plus a small nudge, and calls `find_cell_recursive` to resolve the
/// new coordinate stack at the new world position.
///
/// `world_dir` must be unit-length.
pub fn trace_step_recursive(
    stack: &CoordStack,
    world_pos: Vec3,
    world_dir: Vec3,
    geom: &Geometry,
) -> Option<RecursiveHit> {
    if stack.is_empty() {
        return None;
    }

    let locals = local_positions(stack, world_pos);
    // Fast path: when no frame on the stack carries a rotation, every
    // frame's local_dir equals world_dir and we can skip the per-frame
    // direction-cascade entirely. Rotation-free geometries (Godiva,
    // PWR pin-cell, the 17×17 assembly demo) take this path.
    let any_rotation = stack.iter().any(|c| c.rotation.is_some());
    let local_dirs = if any_rotation {
        Some(local_directions(stack, world_dir))
    } else {
        None
    };

    let mut best_dist = f64::INFINITY;
    let mut best_surface: Option<usize> = None;

    // Source (1) + (2): every cell on the stack contributes its surface
    // boundaries, evaluated in that cell's own local frame (so
    // rotations on parent cells correctly transform the ray).
    for (depth, coord) in stack.iter().enumerate() {
        let cell = &geom.cells[coord.cell_idx as usize];
        let mut surface_indices = Vec::new();
        cell.region.surface_indices(&mut surface_indices);
        surface_indices.sort_unstable();
        surface_indices.dedup();

        let local_pos = locals[depth];
        let local_dir_d = local_dirs.as_ref().map(|v| v[depth]).unwrap_or(world_dir);
        if let Some(hit) =
            find_nearest_surface(local_pos, local_dir_d, &geom.surfaces, &surface_indices)
            && hit.distance < best_dist
        {
            best_dist = hit.distance;
            best_surface = Some(hit.surface_idx);
        }
    }

    // Source (3): lattice grid planes. Surfaces win ties — if a grid
    // line coincides with a reflective surface (common when a 2x2
    // lattice sits inside a reflective box at the same world
    // coordinates), float rounding may put the grid distance an ULP
    // below the surface distance, and treating that as a Transmission
    // grid crossing instead of a Reflective surface hit silently leaks
    // the particle. The COINCIDENCE_TOL break here picks the surface.
    const COINCIDENCE_TOL: f64 = 1e-9;
    for (depth, coord) in stack.iter().enumerate() {
        // Parent-frame coordinates: depth==0 → world, otherwise the
        // depth-1 entry in the precomputed locals/local_dirs.
        let parent_xy = || -> (Vec3, Vec3) {
            if depth == 0 {
                (world_pos, world_dir)
            } else {
                (
                    locals[depth - 1],
                    local_dirs
                        .as_ref()
                        .map(|v| v[depth - 1])
                        .unwrap_or(world_dir),
                )
            }
        };
        if let Some((lattice_id, current)) = coord.lattice {
            let (parent_local, parent_dir) = parent_xy();
            let lattice = geom.lattice(lattice_id);
            let d = lattice.distance_to_grid(parent_local, parent_dir, current);
            if d + COINCIDENCE_TOL < best_dist {
                best_dist = d;
                best_surface = None;
            }
        }
        if let Some((hex_id, current)) = coord.hex_lattice {
            let (parent_local, parent_dir) = parent_xy();
            let hex = geom.hex_lattice(hex_id);
            let d = hex.distance_to_grid(parent_local, parent_dir, current);
            if d + COINCIDENCE_TOL < best_dist {
                best_dist = d;
                best_surface = None;
            }
        }
    }

    if !best_dist.is_finite() {
        return None;
    }

    // Advance and re-resolve.
    let nudge = 1e-10;
    let new_world = world_pos + world_dir * (best_dist + nudge);
    let next_stack = find_cell_recursive(new_world, geom);

    let bc = best_surface
        .map(|idx| geom.surfaces[idx].boundary_condition())
        .unwrap_or(BoundaryCondition::Transmission);

    Some(RecursiveHit {
        distance: best_dist,
        surface_idx: best_surface,
        next_stack,
        bc,
    })
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
    use crate::geometry::coord::CoordStackExt;
    use crate::geometry::lattice::RectLattice;
    use crate::geometry::surface::BoundaryCondition;
    use crate::geometry::universe::{Universe, UniverseId};

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

    #[test]
    fn flat_godiva_unchanged_under_recursive_find() {
        // The recursive path on a single-universe geometry must agree
        // with the existing flat find_cell.
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let geom = Geometry::flat(surfaces, cells).expect("flat construction");

        // Inside: depth 1 stack ending at cell 0 (fuel).
        let stack =
            find_cell_recursive(Vec3::new(0.0, 0.0, 0.0), &geom).expect("origin is in fuel");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.deepest_cell_idx(), 0);
        assert_eq!(stack.deepest().universe, geom.root_universe);

        // Outside: depth 1 stack ending at cell 1 (void).
        let stack = find_cell_recursive(Vec3::new(20.0, 0.0, 0.0), &geom).expect("outside is void");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.deepest_cell_idx(), 1);
    }

    #[test]
    fn one_pin_in_universe() {
        // Root universe holds a bounding box cell whose fill is a pin
        // universe. The pin universe is a fuel cylinder + water.
        // Surfaces (in their own frame, not pin-universe-shifted because
        // the pin universe sits at the root with zero offset):
        //   0: fuel cylinder R=0.4 around z-axis
        //   1..4: bounding box at +/- 1 in x and y (root cell)
        let surfaces = vec![
            Surface::CylinderZ {
                center_x: 0.0,
                center_y: 0.0,
                radius: 0.4,
                bc: BoundaryCondition::Transmission,
            },
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];

        let cells = vec![
            // 0: pin-universe fuel (inside cylinder)
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // 1: pin-universe water (outside cylinder)
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            // 2: root bounding box, filled with pin universe (universe id 1)
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                ]),
                CellFill::Universe(1),
            ),
            // 3: outside the box (root level void)
            Cell::new(
                CellId(3),
                cell::Region::Union(
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(1)),
                        Box::new(cell::outside(2)),
                    )),
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(3)),
                        Box::new(cell::outside(4)),
                    )),
                ),
                CellFill::Void,
            ),
        ];

        let universes = vec![
            Universe::new(UniverseId(0), vec![2, 3]), // root: bounding-box cell + outside
            Universe::new(UniverseId(1), vec![0, 1]), // pin: fuel + water
        ];

        let geom = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect("geometry construction");

        // (0,0,0): inside box, inside cylinder -> fuel.
        let stack = find_cell_recursive(Vec3::new(0.0, 0.0, 0.0), &geom).expect("found");
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0].cell_idx, 2); // root bounding box
        assert_eq!(stack[0].universe, UniverseId(0));
        assert_eq!(stack[1].cell_idx, 0); // pin fuel
        assert_eq!(stack[1].universe, UniverseId(1));

        // (0.6, 0, 0): inside box, outside cylinder -> water.
        let stack = find_cell_recursive(Vec3::new(0.6, 0.0, 0.0), &geom).expect("found");
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0].cell_idx, 2);
        assert_eq!(stack[1].cell_idx, 1); // pin water

        // (5, 0, 0): outside the bounding box -> void at root.
        let stack = find_cell_recursive(Vec3::new(5.0, 0.0, 0.0), &geom).expect("found");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].cell_idx, 3); // root void
    }

    #[test]
    fn two_by_two_lattice() {
        // Root cell holds a 2×2 RectLattice. Lattice has two pin
        // universes — A (fissile) and B (moderator) — alternating in
        // columns: A in col 0, B in col 1.
        //
        // Lattice origin at (-1, -1, -big), pitch 1.0 — so the four
        // elements occupy [-1,0)×[-1,0), [0,1)×[-1,0), [-1,0)×[0,1),
        // [0,1)×[0,1) in the xy plane. Cylinders are centered at
        // pin-universe origin (0, 0) — `RectLattice::local_position`
        // is element-CENTRE-relative (OpenMC convention), so a pin
        // surface defined at the pin universe's local origin sits at
        // the centre of every lattice element automatically.
        let surfaces = vec![
            // 0: cylinder R=0.3 at element-local (0, 0)
            Surface::CylinderZ {
                center_x: 0.0,
                center_y: 0.0,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            // 1..4: root bounding box at +/- 1
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];

        let cells = vec![
            // 0: pin-A fuel (inside cylinder) -> material 0
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // 1: pin-A water (outside cylinder) -> material 1
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            // 2: pin-B "fuel slot" (inside cylinder) -> material 1 (water; this universe is moderator-only)
            Cell::new(CellId(2), cell::inside(0), CellFill::Material(1)),
            // 3: pin-B water -> material 1
            Cell::new(CellId(3), cell::outside(0), CellFill::Material(1)),
            // 4: root cell holding the lattice
            Cell::new(
                CellId(4),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                ]),
                CellFill::Lattice(0),
            ),
            // 5: root void (outside the box)
            Cell::new(
                CellId(5),
                cell::Region::Union(
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(1)),
                        Box::new(cell::outside(2)),
                    )),
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(3)),
                        Box::new(cell::outside(4)),
                    )),
                ),
                CellFill::Void,
            ),
        ];

        let universes = vec![
            Universe::new(UniverseId(0), vec![4, 5]), // root
            Universe::new(UniverseId(1), vec![0, 1]), // pin A (fissile)
            Universe::new(UniverseId(2), vec![2, 3]), // pin B (moderator)
        ];

        let lattices = vec![RectLattice {
            origin: Vec3::new(-1.0, -1.0, -1e6),
            pitch: Vec3::new(1.0, 1.0, 2e6),
            shape: [2, 2, 1],
            // row-major: [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
            // col 0 = A, col 1 = B → [A, B, A, B]
            universes: vec![UniverseId(1), UniverseId(2), UniverseId(1), UniverseId(2)],
            material_overrides: None,
        }];

        let geom = Geometry::new(surfaces, cells, universes, lattices, UniverseId(0))
            .expect("geometry construction");

        // Helper: assert (lattice_index, deepest_cell, deepest_material).
        let cells_ref = &geom.cells;
        let assert_at = |x: f64, y: f64, lattice_xy: [i32; 2], deep_cell: u32, mat: u32| {
            let stack = find_cell_recursive(Vec3::new(x, y, 0.0), &geom)
                .unwrap_or_else(|| panic!("({x},{y}) not found"));
            assert_eq!(stack.len(), 2, "({x},{y}) stack depth");
            assert_eq!(stack[0].cell_idx, 4, "({x},{y}) parent");
            assert_eq!(stack[1].cell_idx, deep_cell, "({x},{y}) deepest cell");
            let (_, idxs) = stack[1].lattice.expect("({x},{y}) should be in a lattice");
            assert_eq!([idxs[0], idxs[1]], lattice_xy, "({x},{y}) lattice element");
            assert_eq!(
                stack.material_idx(cells_ref),
                Some(mat),
                "({x},{y}) material"
            );
        };

        // Element (0,0): A pin centered at world (-0.5, -0.5).
        // (-0.5,-0.5) → element-local (0.5, 0.5) = pin center → fuel (cell 0, mat 0).
        assert_at(-0.5, -0.5, [0, 0], 0, 0);
        // (-0.9,-0.9) → element-local (0.1,0.1), distance from (0.5,0.5) ~ 0.566 > 0.3 → A water (cell 1, mat 1).
        assert_at(-0.9, -0.9, [0, 0], 1, 1);

        // Element (1,0): B pin centered at world (0.5, -0.5).
        // (0.5,-0.5) → element-local (0.5, 0.5) → "fuel slot" (cell 2, mat 1 — water).
        assert_at(0.5, -0.5, [1, 0], 2, 1);
        // (0.9,-0.9) → element-local (0.9, 0.1), out of cylinder → B water (cell 3, mat 1).
        assert_at(0.9, -0.9, [1, 0], 3, 1);

        // Element (0,1): A pin centered at world (-0.5, 0.5).
        assert_at(-0.5, 0.5, [0, 1], 0, 0);

        // Element (1,1): B pin centered at world (0.5, 0.5).
        assert_at(0.5, 0.5, [1, 1], 2, 1);
    }

    #[test]
    fn trace_recursive_through_lattice_grid() {
        // 2x2 lattice, pin universes share a single fuel cylinder R=0.3
        // centered at element-local (0.5, 0.5). Particle starts at
        // (0.1, -0.5, 0) heading +x. It should:
        //   1. Cross out of element (0,0) at x=0 (lattice grid).
        //   2. Land in element (1,0).
        let surfaces = vec![
            Surface::CylinderZ {
                center_x: 0.0,
                center_y: 0.0,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                ]),
                CellFill::Lattice(0),
            ),
            Cell::new(
                CellId(3),
                cell::Region::Union(
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(1)),
                        Box::new(cell::outside(2)),
                    )),
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(3)),
                        Box::new(cell::outside(4)),
                    )),
                ),
                CellFill::Void,
            ),
        ];
        let universes = vec![
            Universe::new(UniverseId(0), vec![2, 3]),
            Universe::new(UniverseId(1), vec![0, 1]),
        ];
        let lattices = vec![RectLattice {
            origin: Vec3::new(-1.0, -1.0, -1e6),
            pitch: Vec3::new(1.0, 1.0, 2e6),
            shape: [2, 2, 1],
            universes: vec![UniverseId(1); 4],
            material_overrides: None,
        }];
        let geom =
            Geometry::new(surfaces, cells, universes, lattices, UniverseId(0)).expect("geometry");

        // Start at world (-0.9, -0.9, 0): element-local (0.1, 0.1).
        // Cylinder at (0.5, 0.5) R=0.3; ray heading +x at y=0.1 misses
        // the cylinder (closest approach distance 0.4 > 0.3).
        let pos0 = Vec3::new(-0.9, -0.9, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);

        let stack0 = find_cell_recursive(pos0, &geom).expect("start");
        // No surface crossing on this ray; the only crossing is the
        // x=0 lattice grid plane at distance 0.9.
        let hit = trace_step_recursive(&stack0, pos0, dir, &geom).expect("hit");
        assert!(
            (hit.distance - 0.9).abs() < 1e-8,
            "expected grid crossing at d=0.9, got {}",
            hit.distance
        );
        assert!(
            hit.surface_idx.is_none(),
            "grid crossing should not report a surface"
        );

        let next_stack = hit.next_stack.expect("re-resolved");
        assert_eq!(next_stack.len(), 2);
        // After the grid crossing we should be in element (1,0).
        assert_eq!(next_stack[1].lattice.expect("lattice").1, [1, 0, 0]);
    }

    #[test]
    fn cell_rotation_rotates_universe_fill() {
        // Pin universe contains a fuel cylinder centered at pin-local
        // (1, 0, 0) with R=0.3. Two scenes share the pin universe but
        // differ in whether the parent cell carries a 90° rotation
        // around z when descending into the pin.
        //
        // No rotation: world point (1.0, 0.0, 0.0) lands inside the
        // fuel; (0.0, -1.0, 0.0) misses (it'd map to pin-local
        // (0, -1, 0), distance √2 from the cylinder centre).
        //
        // 90° rotation around z (R · (1,0,0) = (0,1,0)): the world
        // point that *now* maps to pin-local (1, 0, 0) is the one
        // satisfying R · world = (1,0,0), i.e. world (0, -1, 0). So
        // the previously-outside point is now in fuel and vice versa.
        let surfaces = vec![
            // 0: pin-local cylinder at (1,0) R=0.3
            Surface::CylinderZ {
                center_x: 1.0,
                center_y: 0.0,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            // 1..4: world bounding box at ±2
            Surface::PlaneX {
                x0: -2.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: 2.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -2.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 2.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];

        let make_geometry = |rotation: Option<crate::geometry::Mat3>| {
            let cells = vec![
                // 0 pin fuel
                Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
                // 1 pin water
                Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
                // 2 root cell — bounding box, fills with pin universe
                {
                    let mut c = Cell::new(
                        CellId(2),
                        cell::intersect_all(vec![
                            cell::outside(1),
                            cell::inside(2),
                            cell::outside(3),
                            cell::inside(4),
                        ]),
                        CellFill::Universe(1),
                    );
                    if let Some(r) = rotation {
                        c = c.with_rotation(r);
                    }
                    c
                },
                // 3 outside the box
                Cell::new(
                    CellId(3),
                    cell::Region::Union(
                        Box::new(cell::Region::Union(
                            Box::new(cell::inside(1)),
                            Box::new(cell::outside(2)),
                        )),
                        Box::new(cell::Region::Union(
                            Box::new(cell::inside(3)),
                            Box::new(cell::outside(4)),
                        )),
                    ),
                    CellFill::Void,
                ),
            ];
            let universes = vec![
                Universe::new(UniverseId(0), vec![2, 3]),
                Universe::new(UniverseId(1), vec![0, 1]),
            ];
            Geometry::new(
                surfaces.clone(),
                cells,
                universes,
                Vec::new(),
                UniverseId(0),
            )
            .expect("geometry")
        };

        // Without rotation: (1, 0, 0) → pin-local (1, 0, 0) → fuel.
        let no_rot = make_geometry(None);
        let stack = find_cell_recursive(Vec3::new(1.0, 0.0, 0.0), &no_rot).expect("found");
        assert_eq!(
            stack.last().expect("non-empty").cell_idx,
            0,
            "no-rotation: (1,0,0) should be in fuel"
        );
        let stack = find_cell_recursive(Vec3::new(0.0, -1.0, 0.0), &no_rot).expect("found");
        assert_eq!(
            stack.last().expect("non-empty").cell_idx,
            1,
            "no-rotation: (0,-1,0) should be in water"
        );

        // 90° rotation around z: (0,-1,0) → pin-local (1,0,0) → fuel.
        // (1,0,0) → pin-local (0,1,0) → water.
        let r = crate::geometry::Mat3::rotation_z(std::f64::consts::FRAC_PI_2);
        let rot = make_geometry(Some(r));
        let stack = find_cell_recursive(Vec3::new(0.0, -1.0, 0.0), &rot).expect("found");
        assert_eq!(
            stack.last().expect("non-empty").cell_idx,
            0,
            "rotated: (0,-1,0) should now be in fuel"
        );
        let stack = find_cell_recursive(Vec3::new(1.0, 0.0, 0.0), &rot).expect("found");
        assert_eq!(
            stack.last().expect("non-empty").cell_idx,
            1,
            "rotated: (1,0,0) should now be in water"
        );

        // The deepest Coord must record the rotation so subsequent
        // trace_step_recursive calls can transform direction vectors.
        let stack = find_cell_recursive(Vec3::new(0.0, -1.0, 0.0), &rot).expect("found");
        let deepest = stack.last().expect("non-empty");
        assert!(deepest.rotation.is_some(), "rotation propagated to coord");
    }

    #[test]
    fn trace_recursive_hits_pin_cylinder() {
        // 2x2 lattice as before. Particle at element (0,0) center
        // (-0.5, -0.5) heading +x. It's inside the fuel cylinder; first
        // crossing should be the cylinder surface.
        let surfaces = vec![
            Surface::CylinderZ {
                center_x: 0.0,
                center_y: 0.0,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                ]),
                CellFill::Lattice(0),
            ),
            Cell::new(
                CellId(3),
                cell::Region::Union(
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(1)),
                        Box::new(cell::outside(2)),
                    )),
                    Box::new(cell::Region::Union(
                        Box::new(cell::inside(3)),
                        Box::new(cell::outside(4)),
                    )),
                ),
                CellFill::Void,
            ),
        ];
        let universes = vec![
            Universe::new(UniverseId(0), vec![2, 3]),
            Universe::new(UniverseId(1), vec![0, 1]),
        ];
        let lattices = vec![RectLattice {
            origin: Vec3::new(-1.0, -1.0, -1e6),
            pitch: Vec3::new(1.0, 1.0, 2e6),
            shape: [2, 2, 1],
            universes: vec![UniverseId(1); 4],
            material_overrides: None,
        }];
        let geom =
            Geometry::new(surfaces, cells, universes, lattices, UniverseId(0)).expect("geometry");

        let pos0 = Vec3::new(-0.5, -0.5, 0.0); // element (0,0) center → cylinder center → fuel
        let dir = Vec3::new(1.0, 0.0, 0.0);

        let stack0 = find_cell_recursive(pos0, &geom).expect("start");
        assert_eq!(stack0[1].cell_idx, 0); // fuel cell
        let hit = trace_step_recursive(&stack0, pos0, dir, &geom).expect("hit");
        assert!(
            (hit.distance - 0.3).abs() < 1e-8,
            "cylinder edge at d=0.3, got {}",
            hit.distance
        );
        assert!(hit.surface_idx == Some(0), "should report cylinder surface");

        let next = hit.next_stack.expect("re-resolved");
        assert_eq!(next[1].cell_idx, 1); // pin water cell
    }
}
