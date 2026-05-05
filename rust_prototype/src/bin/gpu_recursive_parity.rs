//! CPU↔GPU parity test for the recursive cell-find primitives.
//!
//! Builds a small recursive geometry (the 2×2 lattice from
//! `recursive_smoke`), uploads it to the GPU, and runs
//! `find_cell_recursive` on N random world points on both CPU and
//! GPU. Asserts that the deepest cell index agrees on every point.
//!
//! This is the proof-of-life test for task #19 — confirms the
//! device-side data layout and recursive descent reproduce the CPU
//! result before the harder transport-loop integration lands.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: this binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_recursive_parity");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    cuda_main::run();
}

#[cfg(feature = "cuda")]
mod cuda_main {
    use std::time::Instant;

    use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
    use open_rust_mc::geometry::lattice::RectLattice;
    use open_rust_mc::geometry::ray::{find_cell_recursive, trace_step_recursive};
    use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
    use open_rust_mc::geometry::universe::{Universe, UniverseId};
    use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
    use open_rust_mc::gpu_recursive::GpuRecursiveContext;
    use rust_mc_sim::Pcg64;

    /// Build the same 2×2 lattice the CPU smoke test uses: pin
    /// universe with fuel cylinder + water, lattice of 4 identical
    /// pin universes, reflective box at world ±1. Only the deepest
    /// cell index is compared, so the chosen materials don't matter.
    fn build_geometry() -> Geometry {
        let surfaces = vec![
            // 0: cylinder at element-local (0.5, 0.5) R=0.3
            Surface::CylinderZ {
                center_x: 0.5,
                center_y: 0.5,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            // 1..4: outer box at ±1
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
            // 5..6: z planes for box closure
            Surface::PlaneZ {
                z0: -10.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: 10.0,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let cells = vec![
            // 0: pin fuel
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // 1: pin water
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            // 2: root cell — bounding box, fills with lattice
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                    cell::outside(5),
                    cell::inside(6),
                ]),
                CellFill::Lattice(0),
            )
            .with_aabb(Aabb::new(
                Vec3::new(-1.0, -1.0, -10.0),
                Vec3::new(1.0, 1.0, 10.0),
            )),
            // 3: outside box
            Cell::new(
                CellId(3),
                Region::Union(
                    Box::new(Region::Union(
                        Box::new(cell::inside(1)),
                        Box::new(cell::outside(2)),
                    )),
                    Box::new(Region::Union(
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
            origin: Vec3::new(-1.0, -1.0, -10.0),
            pitch: Vec3::new(1.0, 1.0, 20.0),
            shape: [2, 2, 1],
            universes: vec![UniverseId(1); 4],
            material_overrides: None,
        }];
        Geometry::new(surfaces, cells, universes, lattices, UniverseId(0))
            .expect("geometry")
    }

    fn build_assembly_geometry() -> Geometry {
        // Mirrors src/bin/pwr_assembly.rs: 17×17 Westinghouse layout,
        // pin and guide-tube universes, reflective box at world ±10.71.
        const PITCH: f64 = 1.260;
        const FUEL_OR: f64 = 0.4096;
        const CLAD_IR: f64 = 0.4180;
        const CLAD_OR: f64 = 0.4750;
        const SHAPE: usize = 17;
        let lat_half = (SHAPE as f64) * PITCH / 2.0;
        let z_half = lat_half;
        let pin_center = PITCH / 2.0;
        let layout = {
            let mut l = [[false; 17]; 17];
            let positions: &[(usize, usize)] = &[
                (2, 5), (2, 8), (2, 11),
                (3, 3), (3, 13),
                (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
                (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
                (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
                (13, 3), (13, 13),
                (14, 5), (14, 8), (14, 11),
            ];
            for &(r, c) in positions {
                l[r][c] = true;
            }
            l
        };
        let surfaces = vec![
            Surface::CylinderZ {
                center_x: pin_center,
                center_y: pin_center,
                radius: FUEL_OR,
                bc: BoundaryCondition::Transmission,
            },
            Surface::CylinderZ {
                center_x: pin_center,
                center_y: pin_center,
                radius: CLAD_IR,
                bc: BoundaryCondition::Transmission,
            },
            Surface::CylinderZ {
                center_x: pin_center,
                center_y: pin_center,
                radius: CLAD_OR,
                bc: BoundaryCondition::Transmission,
            },
            Surface::PlaneX { x0: -lat_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneX { x0:  lat_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0: -lat_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneY { y0:  lat_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0: -z_half, bc: BoundaryCondition::Reflective },
            Surface::PlaneZ { z0:  z_half, bc: BoundaryCondition::Reflective },
        ];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::between(0, 1), CellFill::Void),
            Cell::new(CellId(2), cell::between(1, 2), CellFill::Material(1)),
            Cell::new(CellId(3), cell::outside(2), CellFill::Material(2)),
            Cell::new(CellId(4), cell::inside(1), CellFill::Material(2)),
            Cell::new(CellId(5), cell::between(1, 2), CellFill::Material(1)),
            Cell::new(CellId(6), cell::outside(2), CellFill::Material(2)),
            Cell::new(
                CellId(7),
                cell::intersect_all(vec![
                    cell::outside(3),
                    cell::inside(4),
                    cell::outside(5),
                    cell::inside(6),
                    cell::outside(7),
                    cell::inside(8),
                ]),
                CellFill::Lattice(0),
            )
            .with_aabb(Aabb::new(
                Vec3::new(-lat_half, -lat_half, -z_half),
                Vec3::new(lat_half, lat_half, z_half),
            )),
            Cell::new(
                CellId(8),
                Region::Union(
                    Box::new(Region::Union(
                        Box::new(cell::inside(3)),
                        Box::new(cell::outside(4)),
                    )),
                    Box::new(Region::Union(
                        Box::new(cell::inside(5)),
                        Box::new(cell::outside(6)),
                    )),
                ),
                CellFill::Void,
            ),
        ];
        let universes = vec![
            Universe::new(UniverseId(0), vec![7, 8]),
            Universe::new(UniverseId(1), vec![0, 1, 2, 3]),
            Universe::new(UniverseId(2), vec![4, 5, 6]),
        ];
        let mut lattice_universes = Vec::with_capacity(SHAPE * SHAPE);
        for iy in 0..SHAPE {
            for ix in 0..SHAPE {
                let id = if layout[iy][ix] { UniverseId(2) } else { UniverseId(1) };
                lattice_universes.push(id);
            }
        }
        let lattices = vec![RectLattice {
            origin: Vec3::new(-lat_half, -lat_half, -z_half),
            pitch: Vec3::new(PITCH, PITCH, 2.0 * z_half),
            shape: [SHAPE, SHAPE, 1],
            universes: lattice_universes,
            material_overrides: None,
        }];
        Geometry::new(surfaces, cells, universes, lattices, UniverseId(0))
            .expect("assembly geometry")
    }

    pub fn run() {
        let n = 200_000;
        println!("GPU↔CPU recursive cell-find parity test\n");

        // Test 1: 2×2 lattice (depth 2 stacks).
        println!("=== Test 1 (find_cell): 2×2 lattice ===");
        run_one("2×2 lattice", &build_geometry(), n);

        // Test 2: 17×17 assembly (depth 2 stacks; matches pwr_assembly).
        println!("\n=== Test 2 (find_cell): 17×17 PWR assembly ===");
        run_one("17×17 assembly", &build_assembly_geometry(), n);

        // Test 3+4: trace_step parity on the same two geometries.
        println!("\n=== Test 3 (trace_step): 2×2 lattice ===");
        run_trace("2×2 lattice", &build_geometry(), 50_000);
        println!("\n=== Test 4 (trace_step): 17×17 PWR assembly ===");
        run_trace("17×17 assembly", &build_assembly_geometry(), 50_000);
    }

    fn run_trace(label: &str, geom: &Geometry, n: usize) {
        println!("  geometry  : {label}");
        println!(
            "  surfaces  : {}, cells: {}, universes: {}, lattices: {}",
            geom.surfaces.len(),
            geom.cells.len(),
            geom.universes.len(),
            geom.lattices.len()
        );
        println!("  particles : {n} random (pos, dir) pairs");

        // Sample box from the lattice cell's AABB.
        let (lo, hi) = geom
            .cells
            .iter()
            .find(|c| matches!(c.fill, CellFill::Lattice(_)))
            .map(|c| (c.aabb.min, c.aabb.max))
            .unwrap_or((Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));

        let mut rng = Pcg64::new(0xBEEF, 0);
        let mut positions: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        let mut directions: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let x = lo.x + (hi.x - lo.x) * rng.uniform();
            let y = lo.y + (hi.y - lo.y) * rng.uniform();
            let z = lo.z + (hi.z - lo.z) * rng.uniform();
            // Reject points outside the lattice cell so every history
            // starts in a depth-2 frame.
            if find_cell_recursive(Vec3::new(x, y, z), geom)
                .map(|s| s.len() >= 2)
                .unwrap_or(false)
            {
                let (dx, dy, dz) = rng.isotropic_direction();
                positions.push((x, y, z));
                directions.push((dx, dy, dz));
                i += 1;
            }
        }

        // CPU pass.
        let t0 = Instant::now();
        let cpu_results: Vec<(f64, i32, i32, i32)> = positions
            .iter()
            .zip(directions.iter())
            .map(|(&(x, y, z), &(dx, dy, dz))| {
                let stack = match find_cell_recursive(Vec3::new(x, y, z), geom) {
                    Some(s) => s,
                    None => return (1e300, -1, 1 /* vacuum */, -1),
                };
                let hit = match trace_step_recursive(
                    &stack,
                    Vec3::new(x, y, z),
                    Vec3::new(dx, dy, dz),
                    geom,
                ) {
                    Some(h) => h,
                    None => return (1e300, -1, 1, -1),
                };
                let surf = hit.surface_idx.map(|s| s as i32).unwrap_or(-1);
                let bc = match hit.bc {
                    open_rust_mc::geometry::surface::BoundaryCondition::Transmission => 0,
                    open_rust_mc::geometry::surface::BoundaryCondition::Vacuum => 1,
                    open_rust_mc::geometry::surface::BoundaryCondition::Reflective => 2,
                };
                let next_cell = hit
                    .next_stack
                    .as_ref()
                    .and_then(|s| s.last().map(|c| c.cell_idx as i32))
                    .unwrap_or(-1);
                (hit.distance, surf, bc, next_cell)
            })
            .collect();
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "\nCPU trace_step_recursive: {cpu_ms:.1} ms ({:.1} ns/event)",
            cpu_ms * 1e6 / n as f64
        );

        // GPU pass.
        println!("\nBuilding GPU context...");
        let ctx = GpuRecursiveContext::build(geom, n).expect("gpu context");
        let t0 = Instant::now();
        let gpu_results = ctx
            .trace_step_batch(&positions, &directions)
            .expect("gpu trace_step_batch");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "GPU trace_step_batch:    {gpu_ms:.1} ms ({:.1} ns/event)",
            gpu_ms * 1e6 / n as f64
        );

        // Compare. Distance: relative tolerance 1e-9 (different
        // multiplication order between CPU and GPU). Surface idx,
        // BC, next-cell idx: bit-exact.
        let mut mismatches: usize = 0;
        let mut surf_mismatches: usize = 0;
        let mut bc_mismatches: usize = 0;
        let mut next_mismatches: usize = 0;
        let mut max_dist_rel_err: f64 = 0.0;
        for (cpu, gpu) in cpu_results.iter().zip(gpu_results.iter()) {
            let rel = ((cpu.0 - gpu.distance).abs() / cpu.0.abs().max(1e-12)).abs();
            if rel > max_dist_rel_err {
                max_dist_rel_err = rel;
            }
            if rel > 1e-9 {
                mismatches += 1;
            }
            if cpu.1 != gpu.surface_idx {
                surf_mismatches += 1;
            }
            if cpu.2 != gpu.bc {
                bc_mismatches += 1;
            }
            if cpu.3 != gpu.next_deepest_cell {
                next_mismatches += 1;
            }
        }
        println!("\n  distance max-rel-err   : {max_dist_rel_err:.3e}");
        println!(
            "  distance disagree (>1e-9): {mismatches} / {n}  ({:.4}%)",
            mismatches as f64 / n as f64 * 100.0
        );
        println!(
            "  surface idx disagree   : {surf_mismatches} / {n}  ({:.4}%)",
            surf_mismatches as f64 / n as f64 * 100.0
        );
        println!(
            "  bc disagree            : {bc_mismatches} / {n}  ({:.4}%)",
            bc_mismatches as f64 / n as f64 * 100.0
        );
        println!(
            "  next-cell disagree     : {next_mismatches} / {n}  ({:.4}%)",
            next_mismatches as f64 / n as f64 * 100.0
        );
    }

    fn run_one(label: &str, geom: &Geometry, n: usize) {
        println!(
            "  geometry  : {label}");
        println!(
            "  surfaces  : {}, cells: {}, universes: {}, lattices: {}",
            geom.surfaces.len(),
            geom.cells.len(),
            geom.universes.len(),
            geom.lattices.len()
        );
        println!("  points    : {n} random points");

        // Sample box: pick the bounding box of the root lattice cell
        // (or fall back to xyz ∈ [-1.5, 1.5]).
        let (lo, hi) = geom
            .cells
            .iter()
            .find(|c| matches!(c.fill, CellFill::Lattice(_)))
            .map(|c| (c.aabb.min, c.aabb.max))
            .unwrap_or((Vec3::new(-1.5, -1.5, -1.0), Vec3::new(1.5, 1.5, 1.0)));
        // Inflate slightly so we also get out-of-lattice "void" points.
        let pad = 0.2 * (hi.x - lo.x).max(hi.y - lo.y).max(hi.z - lo.z).max(1.0);
        let lo = Vec3::new(lo.x - pad, lo.y - pad, (lo.z + pad).max(-pad));
        let hi = Vec3::new(hi.x + pad, hi.y + pad, (hi.z - pad).min(pad));

        let mut rng = Pcg64::new(0x9_1234, 0);
        let points: Vec<(f64, f64, f64)> = (0..n)
            .map(|_| {
                (
                    lo.x + (hi.x - lo.x) * rng.uniform(),
                    lo.y + (hi.y - lo.y) * rng.uniform(),
                    lo.z + (hi.z - lo.z) * rng.uniform(),
                )
            })
            .collect();

        // CPU pass.
        let t0 = Instant::now();
        let cpu_results: Vec<i32> = points
            .iter()
            .map(|&(x, y, z)| {
                find_cell_recursive(Vec3::new(x, y, z), geom)
                    .and_then(|s| s.last().map(|c| c.cell_idx as i32))
                    .unwrap_or(-1)
            })
            .collect();
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("\nCPU find_cell_recursive: {cpu_ms:.1} ms ({:.1} ns/pt)", cpu_ms * 1e6 / n as f64);

        // GPU pass.
        println!("\nBuilding GPU context...");
        let ctx = GpuRecursiveContext::build(geom, n).expect("gpu context");
        let t0 = Instant::now();
        let gpu_results = ctx.find_cell_batch(&points).expect("gpu find_cell_batch");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("GPU find_cell_batch:    {gpu_ms:.1} ms ({:.1} ns/pt)", gpu_ms * 1e6 / n as f64);

        // Compare.
        let mut mismatches: Vec<(usize, i32, i32, (f64, f64, f64))> = Vec::new();
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            if cpu != gpu {
                mismatches.push((i, *cpu, *gpu, points[i]));
            }
        }

        // Cell-index histograms (for sanity check the points span everything).
        let mut hist = std::collections::BTreeMap::<i32, usize>::new();
        for &c in &cpu_results {
            *hist.entry(c).or_insert(0) += 1;
        }
        println!("\nCPU deepest-cell histogram:");
        for (k, v) in &hist {
            println!("  cell {k:>3} : {v}");
        }

        println!(
            "\n=== {} / {n} points ({:.4}%) DISAGREE ===",
            mismatches.len(),
            mismatches.len() as f64 / n as f64 * 100.0
        );
        if !mismatches.is_empty() {
            println!("\nFirst 10 mismatches:");
            for (i, cpu, gpu, (x, y, z)) in mismatches.iter().take(10) {
                println!(
                    "  [{i:>6}] world=({x:+.4}, {y:+.4}, {z:+.4})  cpu={cpu}  gpu={gpu}"
                );
            }
            std::process::exit(1);
        }
        println!("\nAll {n} points agree. GPU recursive cell-find parity confirmed.");
    }
}
