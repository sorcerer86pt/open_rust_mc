//! CUDA backend ICSBEP regression — `--features cuda` only.
//!
//! Reuses `material_resolve` and the JSON case format from
//! `icsbep_runs.rs` (companion file) but routes the eigenvalue loop
//! through `dispatch::CudaRunner`. Limited to metal cases that fit
//! the device's `max_nuc = 4` per-material upload constraint
//! (HEU-MF-001 Godiva, U233-MF-001 Jezebel-23). Solution and Pu
//! benchmarks need a wider `max_nuc` or sparse-material upload
//! before they can run here.

use open_rust_mc::geometry::scene_io;
use open_rust_mc::transport::dispatch::{CudaRunner, EigenvalueRunner};
use open_rust_mc::transport::material::Material;
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::simulate::SimConfig;
use open_rust_mc::gpu_transport::GpuTransportContext;
use open_rust_mc::gpu_recursive::GpuRecursiveContext;
use std::path::{Path, PathBuf};

const K_B_EV_PER_K: f64 = 8.617_333_262e-5;

fn bench_dir() -> PathBuf {
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("bench/icsbep")
}

fn data_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return PathBuf::from(v);
    }
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("data/endfb-vii.1-hdf5/neutron").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("data/endfb-vii.1-hdf5/neutron")
}

fn run_case_cuda(
    case_file: &Path,
    batches: u32,
    inactive: u32,
    particles: u32,
    seed: u64,
    rank: usize,
) -> (f64, f64, f64, f64) {
    let text = std::fs::read_to_string(case_file).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let benchmark = &value["benchmark"];
    let scene = &value["scene"];
    let k_ref = benchmark["k_eff_reference"].as_f64().unwrap();
    let sigma_exp = benchmark["k_eff_sigma"].as_f64().unwrap();

    let loaded = scene_io::load_scene_from_json(&scene.to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&data_dir());
    let resolved =
        material_resolve::resolve_materials(&loaded.materials, &lib, rank).unwrap();

    // Per-nuclide AWR + ν̄ constants — the device kernel reads these
    // from flat arrays indexed by `xs_kernel_idx`.
    let awrs: Vec<f64> = resolved.provider.nuclides.iter().map(|n| n.awr).collect();
    let nu_bars: Vec<f64> = resolved
        .provider
        .nuclides
        .iter()
        .map(|n| n.nu_bar_const)
        .collect();

    // Per-material kT in eV. `dispatch::CudaRunner` indexes this by
    // material idx in the same order as `resolved.materials`.
    let mat_k_t: Vec<f64> = resolved
        .materials
        .iter()
        .map(|m| m.temperature * K_B_EV_PER_K)
        .collect();

    // Pick the first nuclide with thermal data as the S(α,β) target.
    // Device kernel supports exactly one thermal nuclide; multi-thermal
    // materials need a follow-up to the CUDA upload format.
    let sab_nuc_idx: i32 = resolved
        .provider
        .thermal
        .iter()
        .position(|t| t.is_some())
        .map_or(-1, |i| i as i32);

    // Build GPU contexts.
    let gpu =
        GpuTransportContext::new().expect("GpuTransportContext::new (no CUDA device?)");
    let nuc_data = gpu
        .upload_nuclide_data(&resolved.provider.nuclides, rank)
        .expect("upload nuclides");
    let mat_data = gpu
        .upload_material_data(&resolved.materials, &awrs, &nu_bars)
        .expect("upload materials");
    let sab_data = if sab_nuc_idx >= 0 {
        // `material_resolve` stashes thermal data on the provider —
        // pick the one at the target index and upload at the case
        // temperature.
        let arc = resolved.provider.thermal[sab_nuc_idx as usize]
            .as_ref()
            .expect("sab arc");
        let t_idx = arc.select_temperature(loaded.materials[0].temperature, 0.5);
        gpu.upload_sab_data(arc, t_idx).expect("upload S(α,β)")
    } else {
        gpu.upload_sab_data_empty().expect("upload empty S(α,β)")
    };
    let wmp_data = gpu
        .upload_wmp_data_empty(resolved.provider.nuclides.len())
        .expect("upload empty WMP");

    let rec =
        GpuRecursiveContext::build(&loaded.geometry, particles as usize).expect("GpuRecursiveContext::build");

    let mut cfg = SimConfig::default();
    cfg.batches = batches;
    cfg.inactive = inactive;
    cfg.particles_per_batch = particles;
    cfg.seed = seed;
    cfg.verbose = false;

    let materials = resolved.materials.clone();
    let cells = loaded.geometry.cells.clone();
    let geometry = loaded.geometry.clone();

    let runner = CudaRunner {
        recursive: &rec,
        transport: &gpu,
        nuc_data: &nuc_data,
        mat_data: &mat_data,
        sab_data: &sab_data,
        wmp_data: &wmp_data,
        mat_k_t: &mat_k_t,
        sab_nuc_idx,
        max_events_per_history: 10_000,
        fis_capacity: (particles as usize) * 4,
        initial_source: Box::new(move |n, s| {
            let sites = open_rust_mc::transport::simulate::initial_source(
                n, &geometry, &cells, s,
            );
            sites
                .iter()
                .map(|fs| (fs.pos.x, fs.pos.y, fs.pos.z, fs.energy))
                .collect()
        }),
    };
    let _ = materials;
    let outcome = runner.run(&cfg);

    let active: Vec<f64> = outcome
        .batches
        .iter()
        .skip(inactive as usize)
        .map(|b| b.k_eff)
        .collect();
    let n = active.len() as f64;
    let mean = active.iter().sum::<f64>() / n;
    let variance = active
        .iter()
        .map(|k| (k - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let stderr = (variance / n).sqrt();
    (mean, stderr, k_ref, sigma_exp)
}

fn report(case: &str, k_calc: f64, sigma_calc: f64, k_ref: f64, sigma_exp: f64) -> bool {
    let delta = k_calc - k_ref;
    let pcm = delta * 1.0e5;
    let sigma_c = (sigma_calc * sigma_calc + sigma_exp * sigma_exp).sqrt();
    let n_sigma = if sigma_c > 0.0 {
        delta.abs() / sigma_c
    } else {
        f64::INFINITY
    };
    let pass_stat = n_sigma <= 3.0;
    let pass_phys = pcm.abs() <= 500.0;
    let verdict = match (pass_stat, pass_phys) {
        (true, true) => "PASS",
        (false, _) => "FAIL(stat)",
        (_, false) => "FAIL(phys)",
    };
    println!(
        "  [CUDA {case}] k_calc = {k_calc:.5} ± {sigma_calc:.5}   k_ref = {k_ref:.5} ± {sigma_exp:.5}   \
         Δ = {pcm:+.0} pcm   {n_sigma:.2}σ   [{verdict}]"
    );
    pass_stat && pass_phys
}

/// HMF-001 Godiva on CUDA. 3 nuclides (U-234/235/238), no S(α,β),
/// fits the device's max_nuc=4 constraint.
#[test]
#[ignore = "ICSBEP regression (CUDA) — opt in via --ignored. Requires `--features cuda` and a working CUDA device."]
fn cuda_heu_met_fast_001_godiva() {
    let case = bench_dir().join("heu-met-fast-001_case-1.json");
    let (k, sigma, k_ref, sigma_exp) = run_case_cuda(&case, 80, 20, 5_000, 42, 15);
    let pass = report("HEU-MET-FAST-001.case-1", k, sigma, k_ref, sigma_exp);
    assert!(pass, "Godiva CUDA case failed under dual criterion");
}

/// U233-MF-001 Jezebel-23 on CUDA. 4 nuclides (U-233/234/235/238),
/// no S(α,β), fits max_nuc=4 exactly.
#[test]
#[ignore = "ICSBEP regression (CUDA) — opt in via --ignored. Requires `--features cuda` and a working CUDA device."]
fn cuda_u233_met_fast_001() {
    let case = bench_dir().join("u233-met-fast-001.json");
    let (k, sigma, k_ref, sigma_exp) = run_case_cuda(&case, 80, 20, 5_000, 42, 15);
    let pass = report("U233-MET-FAST-001", k, sigma, k_ref, sigma_exp);
    assert!(pass, "U-233 Jezebel-23 CUDA case failed under dual criterion");
}
