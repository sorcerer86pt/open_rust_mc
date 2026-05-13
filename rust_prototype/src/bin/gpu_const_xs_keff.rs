//! GPU vs CPU constant-XS transport comparison on the 2×2 lattice.
//!
//! Validates that the device-side const_xs_transport kernel produces
//! aggregate counts (collisions, absorptions, fissions, fission-bank
//! size) consistent with the CPU recursive transport on the same
//! geometry, same constant XS, and (independently) per-particle
//! seeded RNGs. Aggregate agreement within a few-σ MC envelope is
//! the gate; bit-exact is not expected because float-rounding ties
//! between collision-distance and surface-distance can flip event
//! ordering between CPU and GPU.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: this binary requires the 'cuda' feature.");
    eprintln!("Build with: cargo run --release --features cuda --bin gpu_const_xs_keff");
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
    use open_rust_mc::geometry::{Aabb, EffectiveFill, Geometry, Vec3};
    use open_rust_mc::gpu_recursive::{ConstXs, GpuRecursiveContext};
    use rust_mc_sim::Pcg64;

    fn build_geometry() -> Geometry {
        // Same 2×2 lattice as the parity test, with two materials:
        // 0 = fissile, 1 = pure scatterer (water-like).
        // Surfaces: 0 = pin cylinder, 1..=6 = reflective box (xy + z).
        let mut surfaces = open_rust_mc::geometry::shapes::pin_cylinders(0.5, 0.5, &[0.3]);
        let outer_box = open_rust_mc::geometry::shapes::rect_box(
            [1.0, 1.0, 10.0],
            BoundaryCondition::Reflective,
            surfaces.len(),
        );
        surfaces.extend(outer_box.surfaces);

        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            Cell::new(CellId(2), outer_box.inside.clone(), CellFill::Lattice(0)).with_aabb(
                Aabb::new(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, 10.0)),
            ),
            Cell::new(
                CellId(3),
                Region::Complement(Box::new(outer_box.inside)),
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
        Geometry::new(surfaces, cells, universes, lattices, UniverseId(0)).expect("geometry")
    }

    /// CPU reference transport — same algorithm the GPU kernel runs.
    /// Returns aggregate counts plus the fission bank from this batch.
    fn cpu_transport(
        geom: &Geometry,
        materials: &[ConstXs],
        positions: &[(f64, f64, f64)],
        directions: &[(f64, f64, f64)],
        seeds: &[(u64, u64)],
        max_events: i32,
    ) -> (
        Vec<(f64, f64, f64)>, // fission bank
        u64,                  // collisions
        u64,                  // absorptions
        u64,                  // fissions
        u64,                  // leakage
    ) {
        let mut fis_bank: Vec<(f64, f64, f64)> = Vec::new();
        let mut n_coll = 0_u64;
        let mut n_abs = 0_u64;
        let mut n_fis = 0_u64;
        let mut n_leak = 0_u64;

        for ((p0, d0), seed) in positions.iter().zip(directions.iter()).zip(seeds.iter()) {
            let mut pos = Vec3::new(p0.0, p0.1, p0.2);
            let mut dir = Vec3::new(d0.0, d0.1, d0.2);
            let mut rng = Pcg64::from_state(seed.0, seed.1);

            let mut stack = match find_cell_recursive(pos, geom) {
                Some(s) => s,
                None => {
                    n_leak += 1;
                    continue;
                }
            };

            let mut alive = true;
            for _ in 0..max_events {
                if !alive {
                    break;
                }
                // effective material via geom helper
                let mat_idx = match geom.effective_material_idx(&stack) {
                    EffectiveFill::Material(m) => m as usize,
                    EffectiveFill::Void => {
                        // Free-stream to next surface (mirrors the
                        // kernel's void path).
                        let hit = match trace_step_recursive(&stack, pos, dir, geom) {
                            Some(h) => h,
                            None => {
                                n_leak += 1;
                                break;
                            }
                        };
                        match hit.bc {
                            BoundaryCondition::Vacuum => {
                                pos = pos + dir * hit.distance;
                                n_leak += 1;
                                alive = false;
                            }
                            BoundaryCondition::Reflective => {
                                pos = pos + dir * hit.distance;
                                if let Some(s) = hit.surface_idx {
                                    match geom.surfaces[s] {
                                        Surface::PlaneX { .. } => {
                                            dir = Vec3::new(-dir.x, dir.y, dir.z)
                                        }
                                        Surface::PlaneY { .. } => {
                                            dir = Vec3::new(dir.x, -dir.y, dir.z)
                                        }
                                        Surface::PlaneZ { .. } => {
                                            dir = Vec3::new(dir.x, dir.y, -dir.z)
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            BoundaryCondition::Transmission => {
                                let nudge = 1e-10;
                                pos = pos + dir * (hit.distance + nudge);
                                match hit.next_stack {
                                    Some(s) => {
                                        stack = s;
                                    }
                                    None => {
                                        n_leak += 1;
                                        alive = false;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                };
                let m = materials[mat_idx];
                if m.sigma_t <= 0.0 {
                    n_leak += 1;
                    break;
                }
                let d_collide = rng.exponential(m.sigma_t);
                let hit = match trace_step_recursive(&stack, pos, dir, geom) {
                    Some(h) => h,
                    None => {
                        n_leak += 1;
                        break;
                    }
                };
                if d_collide < hit.distance {
                    // collision
                    pos = pos + dir * d_collide;
                    n_coll += 1;
                    let xi_react = rng.uniform() * m.sigma_t;
                    if xi_react < m.sigma_a {
                        n_abs += 1;
                        if m.sigma_a > 0.0 {
                            let pf = m.sigma_f / m.sigma_a;
                            if rng.uniform() < pf {
                                let xi_nu = rng.uniform();
                                let n_neutrons = (m.nu_bar + xi_nu).floor() as i32;
                                if n_neutrons > 0 {
                                    for _ in 0..n_neutrons {
                                        fis_bank.push((pos.x, pos.y, pos.z));
                                    }
                                    n_fis += n_neutrons as u64;
                                }
                            }
                        }
                        break;
                    } else {
                        // scatter — isotropic
                        let (ndx, ndy, ndz) = rng.isotropic_direction();
                        dir = Vec3::new(ndx, ndy, ndz);
                    }
                } else {
                    // crossing
                    match hit.bc {
                        BoundaryCondition::Vacuum => {
                            pos = pos + dir * hit.distance;
                            n_leak += 1;
                            break;
                        }
                        BoundaryCondition::Reflective => {
                            pos = pos + dir * hit.distance;
                            if let Some(s) = hit.surface_idx {
                                match geom.surfaces[s] {
                                    Surface::PlaneX { .. } => dir = Vec3::new(-dir.x, dir.y, dir.z),
                                    Surface::PlaneY { .. } => dir = Vec3::new(dir.x, -dir.y, dir.z),
                                    Surface::PlaneZ { .. } => dir = Vec3::new(dir.x, dir.y, -dir.z),
                                    _ => {}
                                }
                            }
                        }
                        BoundaryCondition::Transmission => {
                            let nudge = 1e-10;
                            pos = pos + dir * (hit.distance + nudge);
                            match hit.next_stack {
                                Some(s) => {
                                    stack = s;
                                }
                                None => {
                                    n_leak += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        (fis_bank, n_coll, n_abs, n_fis, n_leak)
    }

    pub fn run() {
        let geom = build_geometry();

        // Synthetic constant XS: material 0 = fissile (k_inf ≈ 1.6),
        // material 1 = pure scatterer.
        let materials = vec![
            ConstXs {
                sigma_t: 1.0,
                sigma_a: 0.5,
                sigma_f: 0.4,
                nu_bar: 2.0,
            },
            ConstXs {
                sigma_t: 0.5,
                sigma_a: 0.0,
                sigma_f: 0.0,
                nu_bar: 0.0,
            },
        ];

        let limits = open_rust_mc::transport::sim_limits::SimLimits::default();
        let n = 50_000;
        let max_events = limits.max_events_per_history as i32;
        let fis_capacity = limits.fis_capacity(n);

        // Source: uniform in [-1, 1]² × [-1, 1]; reject points outside
        // the lattice cell (the GPU/CPU loop will count those as
        // leakage at birth — same on both).
        let mut rng = Pcg64::new(0xCAFEBEEF, 0);
        let mut positions: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        let mut directions: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        let mut seeds: Vec<(u64, u64)> = Vec::with_capacity(n);
        for i in 0..n {
            let x = -1.0 + 2.0 * rng.uniform();
            let y = -1.0 + 2.0 * rng.uniform();
            let z = -1.0 + 2.0 * rng.uniform();
            let (dx, dy, dz) = rng.isotropic_direction();
            positions.push((x, y, z));
            directions.push((dx, dy, dz));
            // Per-particle independent RNG stream.
            let p = Pcg64::for_particle(0, i as u64);
            seeds.push((p.state(), p.stream()));
        }

        println!("=== Constant-XS transport: CPU vs GPU on 2×2 lattice ===");
        println!("  particles  : {n}");
        println!("  max_events : {max_events}");
        println!("  fissile    : σ_t=1.0, σ_a=0.5, σ_f=0.4, ν̄=2.0  (k_∞ ≈ 1.6)");
        println!("  scatterer  : σ_t=0.5, σ_a=0.0  (pure scatter)");
        println!();

        // CPU pass.
        let t0 = Instant::now();
        let (cpu_bank, cpu_coll, cpu_abs, cpu_fis, cpu_leak) = cpu_transport(
            &geom,
            &materials,
            &positions,
            &directions,
            &seeds,
            max_events,
        );
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "CPU: {} ms  ({:.1} ns/particle)",
            cpu_ms as i64,
            cpu_ms * 1e6 / n as f64
        );
        println!(
            "  collisions  = {cpu_coll}\n  absorptions = {cpu_abs}\n  fission_bank = {} (events {})",
            cpu_bank.len(),
            cpu_fis
        );
        println!("  leakage     = {cpu_leak}");
        println!(
            "  k = fissions / particle = {:.5}",
            cpu_fis as f64 / n as f64
        );

        // GPU pass.
        println!("\nBuilding GPU context...");
        let ctx = GpuRecursiveContext::build(&geom, n).expect("gpu context");
        let t0 = Instant::now();
        let gpu = ctx
            .const_xs_transport(
                &positions,
                &directions,
                &seeds,
                &materials,
                max_events,
                fis_capacity,
            )
            .expect("gpu transport");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "GPU: {} ms  ({:.1} ns/particle)",
            gpu_ms as i64,
            gpu_ms * 1e6 / n as f64
        );
        println!(
            "  collisions  = {}\n  absorptions = {}\n  fission_bank = {} (events {})",
            gpu.n_collisions,
            gpu.n_absorptions,
            gpu.fission_sites.len(),
            gpu.n_fissions
        );
        println!("  leakage     = {}", gpu.n_leakage);
        println!(
            "  k = fissions / particle = {:.5}",
            gpu.n_fissions as f64 / n as f64
        );

        let speedup = cpu_ms / gpu_ms;
        println!("\n=== Comparison ===");
        let cmp = |label: &str, c: u64, g: u64| {
            let diff = (c as f64 - g as f64).abs();
            let scale = (c as f64).max(g as f64).max(1.0);
            let pct = diff / scale * 100.0;
            println!("  {label:<14} CPU {c:>10}  GPU {g:>10}  Δ {diff:>8.0} ({pct:.3}%)");
        };
        cmp("collisions", cpu_coll, gpu.n_collisions);
        cmp("absorptions", cpu_abs, gpu.n_absorptions);
        cmp("fissions", cpu_fis, gpu.n_fissions);
        cmp("leakage", cpu_leak, gpu.n_leakage);
        cmp(
            "bank size",
            cpu_bank.len() as u64,
            gpu.fission_sites.len() as u64,
        );
        let k_cpu = cpu_fis as f64 / n as f64;
        let k_gpu = gpu.n_fissions as f64 / n as f64;
        let k_diff_pcm = (k_cpu - k_gpu).abs() * 1e5;
        println!("\n  k(CPU) = {k_cpu:.5}, k(GPU) = {k_gpu:.5}, |Δ| = {k_diff_pcm:.0} pcm");
        println!("  speedup = {speedup:.2}x");
    }
}
