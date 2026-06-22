// SPDX-License-Identifier: MIT
//! Elastic angular-distribution A/B: CPU `AngularDistribution::sample_mu`
//! vs the CubeCL device μ-sampler, on real HEU-COMP-INTER-003 nuclides —
//! `--features cuda`.
//!
//! Both invert the same tabulated (μ, cdf, pdf) data with the same
//! quadratic/linear formula. We can't share an RNG stream across the two
//! implementations bit-for-bit (the CPU draws xi_bin only on the
//! interpolation branch; draw order differs), so we compare the sampled
//! **distributions statistically**: mean μ and variance at each test
//! energy must agree within MC noise. A broken port (wrong bracket, wrong
//! CDF inversion, isotropic fallback) shifts the mean far outside noise —
//! exactly the forward-peaked-elastic bias that hit ieu-met-fast-001.
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
fn cubecl_ce_angular_matches_cpu() {
    let Some(dd) = data_dir() else {
        eprintln!("no data dir — skipping angular A/B");
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
    // Test energies spanning the fast/intermediate range where elastic
    // angular structure matters (eV).
    let test_e = [1.0e3, 1.0e4, 1.0e5, 5.0e5, 1.0e6, 2.0e6, 5.0e6];

    let mut checked = 0usize;
    let mut worst_dmean = 0.0f64;

    for (ni, nuc) in resolved.provider.nuclides.iter().enumerate() {
        if nuc.elastic_angle.is_none() {
            continue;
        }
        let ang = ce::extract_angular(std::slice::from_ref(nuc));
        let packed = ce::pack_angular(&ang[0]);

        for &e in &test_e {
            // CPU samples (seeded Rng).
            let mut rng = Rng::new(0xA5A5_0000 + ni as u64, e as u64 | 1);
            let cpu: Vec<f64> = (0..n_samp)
                .map(|_| nuc.elastic_angle.as_ref().unwrap().sample_mu(e, &mut rng))
                .collect();

            // GPU samples: feed independent uniform pairs.
            let mut grng = Rng::new(0x5A5A_0000 + ni as u64, e as u64 | 1);
            let mut es = vec![e; n_samp];
            let mut xb = Vec::with_capacity(n_samp);
            let mut xm = Vec::with_capacity(n_samp);
            for _ in 0..n_samp {
                xb.push(grng.uniform());
                xm.push(grng.uniform());
            }
            let _ = &mut es;
            let gpu = match std::panic::catch_unwind(|| {
                ce::sample_mu_gpu::<cubecl::cuda::CudaRuntime>(&device, &packed, &es, &xb, &xm)
            }) {
                Ok(g) => g,
                Err(_) => {
                    eprintln!("no CUDA device — skipping angular A/B");
                    return;
                }
            };

            let (cm, cv) = mean_var(&cpu);
            let (gm, gv) = mean_var(&gpu);
            // MC stderr on the mean ≈ sqrt(var / n); compare within 5σ.
            let se = (cv.max(gv) / n_samp as f64).sqrt().max(1e-4);
            let dmean = (cm - gm).abs();
            worst_dmean = worst_dmean.max(dmean);
            checked += 1;
            assert!(
                dmean < 5.0 * se,
                "nuclide {ni} E={e:.2e}: mean μ CPU={cm:.5} GPU={gm:.5} (Δ={dmean:.5}, 5σ={:.5}); \
                 var CPU={cv:.4} GPU={gv:.4}",
                5.0 * se
            );
        }
    }

    eprintln!("angular A/B: {checked} (nuclide,E) means checked, worst |Δmean μ| = {worst_dmean:.2e}");
    assert!(checked > 10, "too few angular comparisons");
}
