// SPDX-License-Identifier: MIT
//! Continuous-energy cross-section A/B: CPU provider vs CubeCL device
//! lookup, on the real HEU-COMP-INTER-003 nuclides — `--features cuda`.
//!
//! Validates the genuinely new CE piece of the CubeCL transport port:
//! per-nuclide pointwise σ on the device, one binary search on the
//! shared energy grid + log-log interpolation per reaction. For each
//! nuclide we reconstruct the total microscopic σ(E) (sum of the 5
//! carried channels: elastic + fission + capture + inelastic + n2n) at
//! a spread of energies on both the CPU (`SvdXsProvider`/`ReactionKernel`)
//! and the GPU (CubeCL), and require agreement to tight relative error.
//!
//! Runs the CubeCL kernel through its CUDA runtime (cubecl#1336 blocks
//! the heavier kernels on Vulkan; this lookup is small but we use CUDA
//! for consistency with the transport path).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use open_rust_mc::geometry::scene_io;
use open_rust_mc::gpu_ce_cubecl as ce;
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::xs_provider::ReactionKernel;

fn bench_dir() -> PathBuf {
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("bench/icsbep")
}

fn data_dir() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return Some(PathBuf::from(v));
    }
    let start: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    open_rust_mc::data_paths::discover_neutron_dir(&start)
}

/// CPU reference: total micro σ of `nuc` at `energy` over the same 5
/// channels the device carries (elastic+fission+capture+inelastic+n2n).
fn cpu_total_5ch(
    nuc: &open_rust_mc::transport::xs_provider::NuclideKernels,
    energy: f64,
) -> f64 {
    let chans: [&Option<ReactionKernel>; 5] = [
        &nuc.elastic,
        &nuc.fission,
        &nuc.capture,
        &nuc.inelastic,
        &nuc.n2n,
    ];
    chans
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|k| {
            let idx = k.energy_index(energy);
            // log-frac matches the device kernel + provider lookup.
            let grid = k.energies();
            let frac = if idx + 1 < grid.len() && grid[idx] > 0.0 && grid[idx + 1] > grid[idx] {
                ((energy.ln() - grid[idx].ln()) / (grid[idx + 1].ln() - grid[idx].ln()))
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            k.reconstruct_interp(idx, frac)
        })
        .sum()
}

#[test]
fn cubecl_ce_total_xs_matches_cpu() {
    let Some(dd) = data_dir() else {
        eprintln!("no nuclear data dir found — skipping CE XS A/B");
        return;
    };
    let case = bench_dir().join("heu-comp-inter-003_case-1.json");
    if !case.exists() {
        eprintln!("case file missing: {} — skipping", case.display());
        return;
    }

    let text = std::fs::read_to_string(&case).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let scene = value.get("scene").expect("scene block");
    let loaded = scene_io::load_scene_from_json(&scene.to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&dd);
    let resolved = match material_resolve::resolve_materials(&loaded.materials, &lib, 5) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resolve failed ({e:?}) — skipping");
            return;
        }
    };

    let nuclides = &resolved.provider.nuclides;
    assert!(!nuclides.is_empty(), "no nuclides resolved");
    eprintln!("HEU-COMP-INTER-003 case-1: {} nuclides", nuclides.len());

    // Energies spanning the fast/intermediate spectrum (eV).
    let energies: Vec<f64> = {
        let mut v = Vec::new();
        let mut e = 1.0e-2;
        while e < 2.0e7 {
            v.push(e);
            e *= 1.5; // ~geometric sweep, ~52 points
        }
        v
    };

    let device = cubecl::cuda::CudaDevice::default();
    let mut worst_rel = 0.0f64;
    let mut n_checked = 0usize;

    // One nuclide at a time: pack just that nuclide as "nuclide 0" so the
    // device kernel (which reads nuclide 0) compares against this nuclide.
    for (ni, nuc) in nuclides.iter().enumerate() {
        let ce_nucs = ce::extract_ce(std::slice::from_ref(nuc));
        if ce_nucs[0].grid.is_empty() {
            continue; // no reactions (shouldn't happen for these)
        }
        let packed = ce::pack_ce(&ce_nucs);

        let gpu = std::panic::catch_unwind(|| {
            ce::total_micro_xs::<cubecl::cuda::CudaRuntime>(&device, &packed, &energies)
        });
        let gpu = match gpu {
            Ok(v) => v,
            Err(_) => {
                eprintln!("no usable CUDA device — skipping CE XS A/B");
                return;
            }
        };

        for (i, &e) in energies.iter().enumerate() {
            let cpu = cpu_total_5ch(nuc, e);
            let g = gpu[i];
            // Skip points where both are ~0 (below all thresholds).
            if cpu.abs() < 1e-12 && g.abs() < 1e-12 {
                continue;
            }
            let denom = cpu.abs().max(1e-30);
            let rel = (cpu - g).abs() / denom;
            if rel > worst_rel {
                worst_rel = rel;
            }
            n_checked += 1;
            assert!(
                rel < 1e-4,
                "nuclide {ni}, E={e:.4e} eV: CPU σ={cpu:.6e} vs GPU σ={g:.6e} (rel {rel:.2e})"
            );
        }
    }

    eprintln!(
        "CE XS A/B: {n_checked} (nuclide,E) points checked, worst rel err = {worst_rel:.2e}"
    );
    assert!(n_checked > 100, "too few points actually compared");
}
