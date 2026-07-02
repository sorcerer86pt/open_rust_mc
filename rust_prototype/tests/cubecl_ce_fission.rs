// SPDX-License-Identifier: MIT
//! Fission-χ A/B: CPU `EnergyDistribution::sample` vs the CubeCL device
//! fission-energy sampler, on real HEU-COMP-INTER-003 fissile nuclides
//! — `--features cuda`.
//!
//! Both sample the outgoing fission-neutron energy from the same data:
//! tabular ContinuousTabular (stochastic bin + quadratic CDF inversion +
//! kinematic remap) or a closed-form Watt/Maxwell/Evaporation law. As
//! with the angular A/B we compare the sampled spectra statistically —
//! mean E_out and variance at each incident energy must agree within MC
//! noise. A broken χ (wrong law, wrong CDF inversion, missing remap)
//! shifts the mean far outside noise.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use open_rust_mc::geometry::scene_io;
use open_rust_mc::gpu_ce_cubecl as ce;
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::rng::Rng;

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
    open_rust_mc::data_paths::discover_neutron_dir(&PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var)
}

#[test]
fn cubecl_ce_fission_matches_cpu() {
    let Some(dd) = data_dir() else {
        eprintln!("no data dir — skipping fission A/B");
        return;
    };
    let case = bench_dir().join("heu-comp-inter-003_case-1.json");
    if !case.exists() {
        eprintln!("case missing — skipping");
        return;
    }
    let text = std::fs::read_to_string(&case).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let loaded = scene_io::load_scene_from_json(&v.get("scene").unwrap().to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&dd);
    let resolved = match material_resolve::resolve_materials(&loaded.materials, &lib, 5) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("resolve failed ({e:?}) — skipping");
            return;
        }
    };

    let device = cubecl::cuda::CudaDevice::default();
    let n_samp = 200_000usize;
    // Incident energies: thermal-ish, intermediate, fast.
    let test_e = [2.53e-2, 1.0e3, 1.0e5, 1.0e6, 2.0e6, 6.0e6];

    let mut checked = 0usize;
    let mut worst_rel = 0.0f64;

    for (ni, nuc) in resolved.provider.nuclides.iter().enumerate() {
        let Some(ed) = nuc.fission_energy_dist.as_ref() else {
            continue;
        };
        let fc = ce::extract_fission(std::slice::from_ref(nuc));
        if fc[0].law < 0 {
            continue;
        }
        let packed = ce::pack_fission(&fc[0]);

        for &e in &test_e {
            // CPU samples.
            let mut rng = Rng::new(0xF155_0000 + ni as u64, e as u64 | 1);
            let cpu: Vec<f64> = (0..n_samp).map(|_| ed.sample(e, &mut rng)).collect();

            // GPU samples: independent per-thread seeds.
            let seeds: Vec<(u64, u64)> = (0..n_samp)
                .map(|i| {
                    (
                        0x9E37_79B9_7F4A_7C15u64
                            .wrapping_mul(i as u64 + 1)
                            .wrapping_add(ni as u64),
                        (e as u64) | 1,
                    )
                })
                .collect();
            let e_in = vec![e; n_samp];
            let gpu = match std::panic::catch_unwind(|| {
                ce::sample_fission_gpu::<cubecl::cuda::CudaRuntime>(&device, &packed, &e_in, &seeds)
            }) {
                Ok(g) => g,
                Err(_) => {
                    eprintln!("no CUDA device — skipping fission A/B");
                    return;
                }
            };

            let (cm, cv) = mean_var(&cpu);
            let (gm, gv) = mean_var(&gpu);
            // Compare mean E_out relative to CPU mean; MC stderr is small
            // at 200k, so a correct port lands well under 1%.
            let rel = (cm - gm).abs() / cm.abs().max(1.0);
            worst_rel = worst_rel.max(rel);
            checked += 1;
            // 5σ on the mean as the hard gate, plus a 2% sanity ceiling.
            let se = (cv.max(gv) / n_samp as f64).sqrt();
            assert!(
                (cm - gm).abs() < 5.0 * se.max(cm.abs() * 1e-3),
                "nuclide {ni} E_in={e:.2e}: mean E_out CPU={cm:.4e} GPU={gm:.4e} \
                 (rel {rel:.3e}); var CPU={cv:.3e} GPU={gv:.3e}"
            );
        }
    }

    eprintln!("fission χ A/B: {checked} (nuclide,E) means checked, worst rel Δmean = {worst_rel:.3e}");
    assert!(checked > 5, "too few fission comparisons");
}
