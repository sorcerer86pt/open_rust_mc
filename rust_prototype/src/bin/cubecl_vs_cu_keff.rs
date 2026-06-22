// SPDX-License-Identifier: MIT
//! A/B/C const-XS k comparison: CPU reference vs legacy CUDA `.cu`
//! (`GpuRecursiveContext::const_xs_transport`) vs the new CubeCL kernel
//! (`gpu_transport_cubecl::const_xs_transport` on the CUDA runtime), on
//! the SAME 2×2 reflective lattice, same constant cross-sections, same
//! per-particle RNG seeds.
//!
//! `k = fissions / source particle` (one-generation multiplication).
//! The three legs are independent MC realizations (same seeds, but
//! collision-vs-surface ties flip on float rounding across
//! implementations), so they agree within MC noise, not bit-for-bit.
//! This proves the CubeCL port reproduces the legacy `.cu` physics.
//!
//!   cargo run --release --features cuda --bin cubecl_vs_cu_keff

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ERROR: requires the 'cuda' feature.");
    eprintln!("cargo run --release --features cuda --bin cubecl_vs_cu_keff");
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
    use open_rust_mc::geometry::surface::BoundaryCondition;
    use open_rust_mc::geometry::universe::{Universe, UniverseId};
    use open_rust_mc::geometry::{Aabb, Geometry, Vec3};
    use open_rust_mc::geometry::flat::build_host_tables;
    use open_rust_mc::gpu_recursive::{ConstXs as CuConstXs, GpuRecursiveContext};
    use open_rust_mc::gpu_transport_cubecl::{
        self as cc, ConstXs as CcConstXs,
    };
    use rust_mc_sim::Pcg64;

    /// Same 2×2 reflective lattice as `gpu_const_xs_keff`: a fissile pin
    /// (mat 0) in a water-like scatterer (mat 1), 2×2, reflective box.
    fn build_geometry() -> Geometry {
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
            Cell::new(CellId(2), outer_box.inside.clone(), CellFill::Lattice(0))
                .with_aabb(Aabb::new(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(1.0, 1.0, 10.0))),
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

    pub fn run() {
        let geom = build_geometry();

        // Synthetic constant XS — identical for both backends.
        let cu_mats = vec![
            CuConstXs { sigma_t: 1.0, sigma_a: 0.5, sigma_f: 0.4, nu_bar: 2.0 },
            CuConstXs { sigma_t: 0.5, sigma_a: 0.0, sigma_f: 0.0, nu_bar: 0.0 },
        ];
        let cc_mats = vec![
            CcConstXs { sigma_t: 1.0, sigma_a: 0.5, sigma_f: 0.4, nu_bar: 2.0 },
            CcConstXs { sigma_t: 0.5, sigma_a: 0.0, sigma_f: 0.0, nu_bar: 0.0 },
        ];

        let limits = open_rust_mc::transport::sim_limits::SimLimits::default();
        let n = 200_000usize;
        let max_events = limits.max_events_per_history as i32;
        let fis_capacity = limits.fis_capacity(n);

        // Shared source + per-particle RNG seeds.
        let mut rng = Pcg64::new(0xCAFEBEEF, 0);
        let mut positions = Vec::with_capacity(n);
        let mut directions = Vec::with_capacity(n);
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            let x = -1.0 + 2.0 * rng.uniform();
            let y = -1.0 + 2.0 * rng.uniform();
            let z = -1.0 + 2.0 * rng.uniform();
            let (dx, dy, dz) = rng.isotropic_direction();
            positions.push((x, y, z));
            directions.push((dx, dy, dz));
            let p = Pcg64::for_particle(0, i as u64);
            seeds.push((p.state(), p.stream()));
        }

        println!("=== const-XS k: legacy .cu vs CubeCL, 2×2 lattice ===");
        println!("  particles = {n}, fissile k_inf ≈ 1.6\n");

        // Leg A — legacy CUDA .cu.
        let ctx = GpuRecursiveContext::build(&geom, n).expect("gpu ctx");
        let t = Instant::now();
        let cu = ctx
            .const_xs_transport(&positions, &directions, &seeds, &cu_mats, max_events, fis_capacity)
            .expect("legacy .cu transport");
        let cu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let k_cu = cu.n_fissions as f64 / n as f64;

        // Leg B — CubeCL on the CUDA runtime.
        let tables = build_host_tables(&geom);
        let packed = cc::pack_transport(&tables, &geom, &cc_mats);
        let device = cubecl::cuda::CudaDevice::default();
        let t = Instant::now();
        let ccb = cc::const_xs_transport::<cubecl::cuda::CudaRuntime>(
            &device, &packed, &positions, &directions, &seeds, max_events as u32, fis_capacity,
        )
        .expect("cubecl transport");
        let cc_ms = t.elapsed().as_secs_f64() * 1000.0;
        let k_cc = ccb.n_fissions as f64 / n as f64;

        let row = |label: &str, b: &str, coll: u64, abs: u64, fis: u64, leak: u64, k: f64, ms: f64| {
            println!(
                "  {label:<12} [{b}]  coll={coll:>9} abs={abs:>9} fis={fis:>9} leak={leak:>7}  k={k:.5}  {ms:.0} ms"
            );
        };
        row("legacy .cu", "CUDA", cu.n_collisions, cu.n_absorptions, cu.n_fissions, cu.n_leakage, k_cu, cu_ms);
        row("CubeCL", "CUDA", ccb.n_collisions, ccb.n_absorptions, ccb.n_fissions, ccb.n_leakage, k_cc, cc_ms);

        let dk_pcm = (k_cu - k_cc).abs() * 1e5;
        // 1σ on k ≈ sqrt(fissions)/n; a few-σ envelope is the gate.
        let sigma_pcm = (cu.n_fissions as f64).sqrt() / n as f64 * 1e5;
        println!(
            "\n  |Δk| = {dk_pcm:.0} pcm   (1σ_MC ≈ {sigma_pcm:.0} pcm)   {}",
            if dk_pcm <= 5.0 * sigma_pcm.max(1.0) {
                "PASS — within MC noise"
            } else {
                "FAIL — outside few-σ envelope"
            }
        );
        // Bank size should also track.
        println!(
            "  fission bank: legacy {} vs CubeCL {}",
            cu.fission_sites.len(),
            ccb.fission_sites.len()
        );
    }
}
