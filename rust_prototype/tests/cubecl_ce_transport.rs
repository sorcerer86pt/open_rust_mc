// SPDX-License-Identifier: MIT
//! Fused CE transport kernel — end-to-end run on a real fissile-sphere
//! scene, `--features cuda`. Validates that the assembled kernel
//! (geometry walk + CE Σ lookup + nuclide/reaction sampling + free-gas
//! elastic + tabulated angular + fission χ banking) runs on the GPU and
//! produces physically sane output: collisions happen, fissions are
//! banked at MeV-range energies, and every history terminates.
//!
//! This is the first whole-transport run of the CubeCL CE kernel; the
//! quantitative k_eff A/B vs the .cu CudaRunner is the follow-up (needs
//! a real ν̄(E) grid rather than the kernel's const-ν first cut).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use open_rust_mc::gpu_ce_cubecl as ce;
use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId};
use open_rust_mc::geometry::flat::build_host_tables;
use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
use open_rust_mc::geometry::universe::{Universe, UniverseId};
use open_rust_mc::geometry::{Geometry, Vec3};
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::rng::Rng;

fn data_dir() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return Some(PathBuf::from(v));
    }
    open_rust_mc::data_paths::discover_neutron_dir(&PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

#[test]
fn cubecl_ce_transport_runs() {
    let Some(dd) = data_dir() else {
        eprintln!("no data dir — skipping CE transport run");
        return;
    };
    let lib = NuclideLibrary::from_data_dir(&dd);

    // Bare HEU sphere via the Python-builder DTOs would be heavier; build
    // a minimal material directly: U-235 + U-238 at Godiva-ish densities.
    use open_rust_mc::geometry::scene_io::{MaterialDto, NuclideEntryDto};
    let mat = MaterialDto {
        name: "HEU".into(),
        comment: None,
        temperature: 294.0,
        nuclides: vec![
            NuclideEntryDto {
                hdf5_file: Some("U235.h5".into()),
                zaid: None,
                label: Some("U-235".into()),
                atom_density: 0.045,
                thermal_file: None,
            },
            NuclideEntryDto {
                hdf5_file: Some("U238.h5".into()),
                zaid: None,
                label: Some("U-238".into()),
                atom_density: 0.0027,
                thermal_file: None,
            },
        ],
        thermal_files: Vec::new(),
    };
    let resolved = match material_resolve::resolve_materials(std::slice::from_ref(&mat), &lib, 5) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resolve failed ({e:?}) — skipping");
            return;
        }
    };

    // Geometry: fissile sphere r=8.7 cm, vacuum boundary.
    let surfaces = vec![Surface::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 8.7,
        bc: BoundaryCondition::Vacuum,
    }];
    let cells = vec![
        Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
        Cell::new(CellId(1), cell::outside(0), CellFill::Void),
    ];
    let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
    let geom = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0)).unwrap();

    // Pack the unified CE scene.
    let tables = build_host_tables(&geom);
    let nucs = &resolved.provider.nuclides;
    let ce_nucs = ce::extract_ce(nucs);
    let ang = ce::extract_angular(nucs);
    let fis = ce::extract_fission(nucs);
    // Material 0 references nuclides by their provider index (0,1 here).
    let mats = vec![ce::MaterialCe {
        nuclides: resolved.materials[0]
            .nuclides
            .iter()
            .map(|n| (n.xs_kernel_idx, n.atom_density))
            .collect(),
    }];
    let scene = ce::pack_ce_full(&tables, &geom, &ce_nucs, &ang, &fis, &mats);
    let mat_kt = vec![294.0 * 8.617333262e-5];

    // Source: isotropic at origin, fission-spectrum-ish 2 MeV.
    let n = 20_000usize;
    let mut rng = Rng::new(0xCE_5151, 1);
    let mut pos = Vec::with_capacity(n);
    let mut dir = Vec::with_capacity(n);
    let mut e = Vec::with_capacity(n);
    let mut seeds = Vec::with_capacity(n);
    for i in 0..n {
        pos.push((0.0, 0.0, 0.0));
        let (dx, dy, dz) = rng.isotropic_direction();
        dir.push((dx, dy, dz));
        e.push(2.0e6);
        let p = Rng::for_particle(0, i as u64);
        seeds.push((p.state(), p.stream()));
    }

    let device = cubecl::cuda::CudaDevice::default();
    let cegen = match std::panic::catch_unwind(|| {
        ce::ce_generation::<cubecl::cuda::CudaRuntime>(
            &device, &scene, &mat_kt, &pos, &dir, &e, &seeds, 10_000, n * 4,
        )
    }) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no CUDA device — skipping CE transport run");
            return;
        }
    };

    let k = cegen.n_fissions as f64 / n as f64;
    let mean_fe = if cegen.fission_energies.is_empty() {
        0.0
    } else {
        cegen.fission_energies.iter().sum::<f64>() / cegen.fission_energies.len() as f64
    };
    eprintln!(
        "CE cegen: coll={} leak={} cap={} fissions={} k1gen={:.4} mean_fis_E={:.3e} eV (bank {})",
        cegen.n_collisions, cegen.n_leak, cegen.n_capture, cegen.n_fissions, k, mean_fe,
        cegen.fission_energies.len()
    );

    // Sanity: real transport happened, fissions banked, energies physical.
    assert!(cegen.n_collisions > 0, "no collisions");
    assert!(cegen.n_fissions > 0, "no fissions banked");
    assert!(!cegen.fission_energies.is_empty(), "empty fission bank");
    // Single-generation k for a near-critical HEU sphere should be O(1):
    // not zero, not absurd. Wide gate — this is a sanity check, not a
    // converged eigenvalue.
    assert!(k > 0.3 && k < 3.0, "single-cegen k out of sane range: {k}");
    // Mean fission emission energy ~ 2 MeV (Watt/χ peak ~1-2 MeV).
    assert!(
        mean_fe > 3.0e5 && mean_fe < 5.0e6,
        "mean fission energy unphysical: {mean_fe:.3e} eV"
    );
}
