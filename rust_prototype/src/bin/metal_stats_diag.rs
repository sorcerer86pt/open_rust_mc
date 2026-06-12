// SPDX-License-Identifier: MIT
//! CPU-vs-GPU integrated-tally diagnostic for the metal hot bias.
//!
//! Runs Godiva (HEU-MET-FAST-001 case-1) eigenvalue on both backends
//! at matched seed and prints active-batch means for: k_eff,
//! leakage_frac, fissions, absorptions, collisions, surface_crossings,
//! thermal_scatters. The +500-700 pcm GPU↔CPU gap on fast-metal
//! benchmarks must be expressible as a divergence in at least one of
//! these tallies — k_eff is a function of (multiplications −
//! absorptions − leakage) per source neutron. The point of this
//! binary is to localise the divergence.
//!
//! Usage:
//!   cargo run --release --features cuda --bin metal_stats_diag

#![allow(dead_code)]

use std::path::PathBuf;

use open_rust_mc::geometry::scene_io;
use open_rust_mc::transport::dispatch::{CpuRunner, EigenvalueRunner};
use open_rust_mc::transport::material_resolve;
use open_rust_mc::transport::nuclides::NuclideLibrary;
use open_rust_mc::transport::simulate::SimConfig;

#[cfg(feature = "cuda")]
use open_rust_mc::gpu_recursive::GpuRecursiveContext;
#[cfg(feature = "cuda")]
use open_rust_mc::gpu_transport::GpuTransportContext;
#[cfg(feature = "cuda")]
use open_rust_mc::transport::dispatch::CudaRunner;

const K_B_EV_PER_K: f64 = 8.617_333_262e-5;

fn workspace_root() -> PathBuf {
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    loop {
        if p.join("bench").join("icsbep").exists() {
            return p;
        }
        if !p.pop() {
            panic!("no bench/icsbep");
        }
    }
}

fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("ICSBEP_DATA_DIR") {
        return PathBuf::from(p);
    }
    let root = workspace_root();
    open_rust_mc::data_paths::discover_neutron_dir(&root)
        .unwrap_or_else(|| root.join("data/endfb-viii.1-hdf5/neutron"))
}

#[derive(Default)]
struct Active {
    n: u32,
    k_sum: f64,
    leak_sum: u64,
    abs_sum: u64,
    fis_sum: u64,
    coll_sum: u64,
    surf_sum: u64,
    therm_sum: u64,
    // for σ
    k_sq_sum: f64,
    // Per-reaction tallies (GPU populates; CPU leaves 0).
    el_sum: u64,
    inel_sum: u64,
    /// Subset of `inel_sum` whose sampled channel was MT=91
    /// (continuum inelastic). Used for the engine-vs-OpenMC
    /// MT=91 isolation diagnostic on Godiva. GPU side always
    /// reports 0 — the CPU number is the one to read.
    inel_cont_sum: u64,
    cap_sum: u64,
    e_fis_in: f64,
    e_el_in: f64,
    e_inel_in: f64,
    e_inel_out: f64,
    // Squared accumulators for σ(E_at_reaction). Added after
    // `bin/nu_lookup_compare` confirmed ν(E) parity — the metal hot
    // bias has to come from a higher moment of the E-at-reaction
    // distribution, not its mean.
    e_fis_in_sq: f64,
    e_el_in_sq: f64,
    e_inel_in_sq: f64,
    // Σ |Q| over inelastic events. ⟨|Q|⟩ = q_inel_sum / inel_sum is
    // the per-event CM-frame excitation energy. A CPU↔GPU gap here
    // localises the spectrum-hardening bias to level selection.
    q_inel_sum: f64,
    // (n,2n)/(n,3n) explicit tallies — both backends populate now.
    // The reconciliation residual is useless in S(α,β) systems
    // (thermal scatter swamps it), so these read the real rate.
    n2n_sum: u64,
    n3n_sum: u64,
    nxn_out_sum: u64,
    e_nxn_out_sum: f64,
    e_nxn_out_sq: f64,
    // Real histories transported (Σ over active batches of n+refilled).
    // The correct per-source denominator — using nominal n over-reports
    // every rate by the refill factor when PHYSOR-F refill is active.
    hist_sum: u64,
}

impl Active {
    fn add(&mut self, b: &open_rust_mc::transport::simulate::BatchResult) {
        self.n += 1;
        self.k_sum += b.k_eff;
        self.k_sq_sum += b.k_eff * b.k_eff;
        self.leak_sum += b.leakage as u64;
        self.abs_sum += b.absorptions as u64;
        self.fis_sum += b.fissions as u64;
        self.coll_sum += b.collisions as u64;
        self.surf_sum += b.surface_crossings as u64;
        self.therm_sum += b.thermal_scatters as u64;
        self.el_sum += b.n_elastic;
        self.inel_sum += b.n_inelastic;
        self.inel_cont_sum += b.n_inelastic_continuum;
        self.cap_sum += b.n_capture;
        self.e_fis_in += b.e_fis_in_sum;
        self.e_el_in += b.e_el_in_sum;
        self.e_inel_in += b.e_inel_in_sum;
        self.e_inel_out += b.e_inel_out_sum;
        self.e_fis_in_sq += b.e_fis_in_sq_sum;
        self.e_el_in_sq += b.e_el_in_sq_sum;
        self.e_inel_in_sq += b.e_inel_in_sq_sum;
        self.q_inel_sum += b.q_inel_sum;
        self.n2n_sum += b.n_n2n;
        self.n3n_sum += b.n_n3n;
        self.nxn_out_sum += b.n_nxn_out;
        self.e_nxn_out_sum += b.e_nxn_out_sum;
        self.e_nxn_out_sq += b.e_nxn_out_sq;
        self.hist_sum += b.source_histories;
    }

    fn report(&self, label: &str, particles_per_batch: u64) {
        let n = self.n as f64;
        // Real histories transported (n + refilled under PHYSOR-F refill),
        // falling back to nominal n×ppb for older batches that don't
        // report source_histories. Using nominal n over-reports every
        // rate by the refill factor — see the refill-normalization fix.
        let n_source = if self.hist_sum > 0 {
            self.hist_sum as f64
        } else {
            n * particles_per_batch as f64
        };
        let mean = self.k_sum / n;
        let var = (self.k_sq_sum / n - mean * mean).max(0.0);
        let stderr = (var / n).sqrt();
        println!("--- {label} active-batch means (over {} batches, {:.0} source histories) ---", self.n, n_source);
        println!("  k_eff             : {mean:.5} ± {stderr:.5}   ({:+.0} pcm vs 1.0)", (mean - 1.0) * 1e5);
        println!("  leakage / source  : {:.4}    ({} / {:.0})", self.leak_sum as f64 / n_source, self.leak_sum, n_source);
        // Two columns: `captures` follows the codebase convention
        // (BatchResult.absorptions = capture events only); `abs (OpenMC)`
        // = captures + fissions, the broader definition OpenMC's
        // "absorption" tally uses. Showing both prevents the "GPU 0 vs
        // CPU 0.04" or "GPU 0.42 vs CPU 0.04" mismatches we saw earlier.
        let abs_omc_style = self.abs_sum + self.fis_sum;
        println!("  captures / source : {:.4}    ({})", self.abs_sum as f64 / n_source, self.abs_sum);
        println!("  abs (OpenMC-def)  : {:.4}    ({} = captures + fissions)", abs_omc_style as f64 / n_source, abs_omc_style);
        println!("  fissions / source : {:.4}    ({})", self.fis_sum as f64 / n_source, self.fis_sum);
        println!("  collisions / src  : {:.2}    ({})", self.coll_sum as f64 / n_source, self.coll_sum);
        println!("  surf cross / src  : {:.2}    ({})", self.surf_sum as f64 / n_source, self.surf_sum);
        println!("  thermal scat / src: {:.4}   ({})", self.therm_sum as f64 / n_source, self.therm_sum);
        // Spectrum-hardening diagnostic tallies (GPU populates; CPU
        // leaves 0). When everything is zero this whole block is
        // suppressed.
        if self.el_sum + self.inel_sum + self.cap_sum > 0 {
            let el = self.el_sum as f64;
            let inel = self.inel_sum as f64;
            let fis = self.fis_sum as f64;
            let cap = self.cap_sum as f64;
            println!("  ─ Per-reaction breakdown (events per source neutron) ─");
            println!("    elastic   / src : {:.4}    ({} events)", el / n_source, self.el_sum);
            println!("    inelastic / src : {:.4}    ({} events)", inel / n_source, self.inel_sum);
            if self.inel_cont_sum > 0 {
                let cont = self.inel_cont_sum as f64;
                let disc = inel - cont;
                println!(
                    "      MT=91 (cont)  : {:.4}    ({} events, {:.1}% of inel)",
                    cont / n_source, self.inel_cont_sum, 100.0 * cont / inel,
                );
                println!(
                    "      MT=51-90 (dis): {:.4}    ({} events, {:.1}% of inel)",
                    disc / n_source, self.inel_sum - self.inel_cont_sum, 100.0 * disc / inel,
                );
            }
            println!("    fission   / src : {:.4}    ({} events)", fis / n_source, self.fis_sum);
            println!("    capture   / src : {:.4}    ({} events)", cap / n_source, self.cap_sum);
            let sum = el + inel + fis + cap;
            let recon = sum / self.coll_sum as f64;
            println!("    (n2n+n3n+...) / src = collisions − (el+inel+fis+cap) = {:.4}   reconciliation {:.4} of total coll",
                     (self.coll_sum as f64 - sum) / n_source, recon);
            // Explicit (n,2n)/(n,3n) tally — the reconciliation residual
            // above is useless in S(α,β) systems (thermal scatter swamps
            // it), so read the real rate here. The CPU transports the
            // secondaries IN-GENERATION; the GPU banks them into the
            // FISSION source (so they enter the k numerator) — a count
            // gap or k gap here localises that methodological split.
            let n2n = self.n2n_sum as f64;
            let n3n = self.n3n_sum as f64;
            println!("  ─ (n,2n) / (n,3n) explicit tally ─");
            println!("    (n,2n)    / src : {:.6}    ({} events)", n2n / n_source, self.n2n_sum);
            println!("    (n,3n)    / src : {:.6}    ({} events)", n3n / n_source, self.n3n_sum);
            // Secondary neutrons added per source (1·n2n + 2·n3n). On the
            // GPU these are banked into the fission source → +k numerator;
            // on the CPU they transport in-generation → 0 direct k.
            let banked = n2n + 2.0 * n3n;
            println!("    extra-n bank / src : {:.6}   (1·n2n + 2·n3n — GPU routes to fission source, CPU in-generation)",
                     banked / n_source);
            if self.nxn_out_sum > 0 {
                let m = self.e_nxn_out_sum / self.nxn_out_sum as f64;
                let m2 = self.e_nxn_out_sq / self.nxn_out_sum as f64;
                let s = (m2 - m * m).max(0.0).sqrt();
                println!("    nxn ⟨E_out⟩    : {:.4e} eV   σ = {:.4e}   ({} outgoing n)", m, s, self.nxn_out_sum);
            }
            // σ(E_at_reaction) = sqrt(⟨E²⟩ − ⟨E⟩²). After nu_lookup_compare
            // proved ν(E) parity, the only way the GPU can have higher
            // ⟨ν⟩ at lower ⟨E_in fission⟩ than OpenMC is a wider /
            // higher-tail E_in distribution. σ_fis is the direct test.
            println!("  ─ Mean + σ at reaction (eV) ─");
            if self.fis_sum > 0 {
                let m = self.e_fis_in / fis;
                let m2 = self.e_fis_in_sq / fis;
                let s = (m2 - m * m).max(0.0).sqrt();
                println!(
                    "    fission:   ⟨E_in⟩ = {:.4e}   σ(E_in) = {:.4e}   σ/⟨E⟩ = {:.3}",
                    m, s, s / m
                );
            }
            if self.el_sum > 0 {
                let m = self.e_el_in / el;
                let m2 = self.e_el_in_sq / el;
                let s = (m2 - m * m).max(0.0).sqrt();
                println!(
                    "    elastic:   ⟨E_in⟩ = {:.4e}   σ(E_in) = {:.4e}   σ/⟨E⟩ = {:.3}",
                    m, s, s / m
                );
            }
            if self.inel_sum > 0 {
                let m_in = self.e_inel_in / inel;
                let m_out = self.e_inel_out / inel;
                let m2_in = self.e_inel_in_sq / inel;
                let s_in = (m2_in - m_in * m_in).max(0.0).sqrt();
                let q_mean = self.q_inel_sum / inel;
                println!(
                    "    inelastic: ⟨E_in⟩ = {:.4e}   σ(E_in) = {:.4e}   ⟨E_out⟩ = {:.4e}   ⟨ΔE⟩ = {:.4e} eV ({:+.2}% loss)   ⟨|Q|⟩ = {:.4e}",
                    m_in, s_in, m_out, m_in - m_out, (m_in - m_out) / m_in * 100.0, q_mean,
                );
            }
        }
    }
}

fn diff_pcm(cpu: f64, gpu: f64) -> f64 {
    (gpu - cpu) * 1e5
}

fn diff_pct(cpu: f64, gpu: f64) -> f64 {
    if cpu.abs() < 1e-12 {
        0.0
    } else {
        (gpu - cpu) / cpu * 100.0
    }
}

fn report_delta(cpu: &Active, gpu: &Active, particles_per_batch: u64) {
    // Real transported histories per leg (n + refilled), so refilled
    // GPU rates are comparable to CPU rates rather than inflated by the
    // refill factor.
    let nps = if cpu.hist_sum > 0 { cpu.hist_sum as f64 } else { (cpu.n as f64) * particles_per_batch as f64 };
    let nps_g = if gpu.hist_sum > 0 { gpu.hist_sum as f64 } else { (gpu.n as f64) * particles_per_batch as f64 };
    println!("\n=== Δ (GPU − CPU) ===");
    let cpu_k = cpu.k_sum / cpu.n as f64;
    let gpu_k = gpu.k_sum / gpu.n as f64;
    println!("  Δ k_eff      : {:+.0} pcm", diff_pcm(cpu_k, gpu_k));
    let cpu_leak = cpu.leak_sum as f64 / nps;
    let gpu_leak = gpu.leak_sum as f64 / nps_g;
    println!("  Δ leakage/src: {:+.4}  ({:+.2}%)   cpu={:.4}  gpu={:.4}", gpu_leak - cpu_leak, diff_pct(cpu_leak, gpu_leak), cpu_leak, gpu_leak);
    let cpu_abs = cpu.abs_sum as f64 / nps;
    let gpu_abs = gpu.abs_sum as f64 / nps_g;
    println!("  Δ abs/src    : {:+.4}  ({:+.2}%)   cpu={:.4}  gpu={:.4}", gpu_abs - cpu_abs, diff_pct(cpu_abs, gpu_abs), cpu_abs, gpu_abs);
    let cpu_fis = cpu.fis_sum as f64 / nps;
    let gpu_fis = gpu.fis_sum as f64 / nps_g;
    println!("  Δ fis/src    : {:+.4}  ({:+.2}%)   cpu={:.4}  gpu={:.4}", gpu_fis - cpu_fis, diff_pct(cpu_fis, gpu_fis), cpu_fis, gpu_fis);
    let cpu_col = cpu.coll_sum as f64 / nps;
    let gpu_col = gpu.coll_sum as f64 / nps_g;
    println!("  Δ coll/src   : {:+.2}   ({:+.2}%)   cpu={:.2}  gpu={:.2}", gpu_col - cpu_col, diff_pct(cpu_col, gpu_col), cpu_col, gpu_col);
    let cpu_surf = cpu.surf_sum as f64 / nps;
    let gpu_surf = gpu.surf_sum as f64 / nps_g;
    println!("  Δ surf/src   : {:+.2}   ({:+.2}%)   cpu={:.2}  gpu={:.2}", gpu_surf - cpu_surf, diff_pct(cpu_surf, gpu_surf), cpu_surf, gpu_surf);
    // S(α,β) thermal scatter — the GPU side used to be hardcoded 0; now
    // a real counter, so this Δ is meaningful. A large gap means the
    // backends disagree on bound-vs-free scattering in the reflector.
    let cpu_therm = cpu.therm_sum as f64 / nps;
    let gpu_therm = gpu.therm_sum as f64 / nps_g;
    println!("  Δ therm/src  : {:+.4} ({:+.2}%)   cpu={:.4}  gpu={:.4}", gpu_therm - cpu_therm, diff_pct(cpu_therm, gpu_therm), cpu_therm, gpu_therm);
    // (n,2n)/(n,3n) rate Δ — the channel the Be-reflector hypothesis is
    // about. cpu/gpu should agree on the RATE (same XS); the k effect of
    // the GPU's bank-into-fission routing shows up in Δ k_eff above.
    let cpu_n2n = cpu.n2n_sum as f64 / nps;
    let gpu_n2n = gpu.n2n_sum as f64 / nps_g;
    println!("  Δ n2n/src    : {:+.6} ({:+.2}%)   cpu={:.6}  gpu={:.6}", gpu_n2n - cpu_n2n, diff_pct(cpu_n2n, gpu_n2n), cpu_n2n, gpu_n2n);
    let cpu_n3n = cpu.n3n_sum as f64 / nps;
    let gpu_n3n = gpu.n3n_sum as f64 / nps_g;
    println!("  Δ n3n/src    : {:+.6} ({:+.2}%)   cpu={:.6}  gpu={:.6}", gpu_n3n - cpu_n3n, diff_pct(cpu_n3n, gpu_n3n), cpu_n3n, gpu_n3n);
}

fn main() {
    // Optional argv[1] = case stem (e.g. "ieu-met-fast-001_case-3") or
    // full JSON path. Defaults to heu-met-fast-001_case-1 (Godiva) so
    // historical invocations keep working.
    let case_arg: Option<String> = std::env::args().nth(1);
    let case_file = match case_arg {
        Some(s) if s.ends_with(".json") => std::path::PathBuf::from(s),
        Some(stem) => workspace_root()
            .join("bench/icsbep")
            .join(format!("{stem}.json")),
        None => workspace_root()
            .join("bench/icsbep")
            .join("heu-met-fast-001_case-1.json"),
    };
    eprintln!("case: {}", case_file.display());

    // Optional argv[2..] = `b=NN i=NN p=NN s=NN` to bump statistics
    // past the 80b × 5k Godiva default. Anything not set keeps the
    // legacy value.
    let mut arg_b: u32 = 80;
    let mut arg_i: u32 = 20;
    let mut arg_p: u32 = 5_000;
    let mut arg_s: u64 = 42;
    // SVD rank for XS reconstruction. 15 = iteration default (~200 pcm
    // residual on Godiva-class fast metal — the "memory vs precision"
    // figure in the paper documents the curve). Bump to 30+ for
    // reference-grade comparisons against OpenMC / MCNP where the
    // engine-vs-data-compression bias must be sub-MC-noise.
    let mut arg_r: usize = 15;
    // Sweep-faithful GPU knobs. The B200 ICSBEP sweep ran with
    // `gpu_auto_refill=true` (PHYSOR 2022 Optimization F population
    // refill) and 5M particles; the diag's historical 5k / no-refill
    // default is NOT representative — the same case swung ~620 pcm
    // between the two configs. `refill=<f>` pins an explicit pool
    // factor; `auto_refill` lets the device-attribute heuristic pick.
    let mut arg_refill: Option<f64> = None;
    let mut arg_auto_refill = false;
    // OpenMC reference JSON (per-case). Defaults to the legacy Godiva
    // path for back-compat; pass `openmc=outputs/openmc_<case>.json`
    // (produced by scripts/openmc_scene_runner.py) to grade any scene.
    let mut arg_openmc: Option<String> = None;
    // (n,2n)/(n,3n) secondary routing mode for the GPU. Default 0 =
    // in-generation transport (the correct fix). `bank_as_fission`
    // selects the legacy mode 1 (over-credits k — for regression A/B);
    // `no_nxn_bank` selects mode 2 (drop — isolation arm).
    let mut arg_nxn_mode: i32 = 0;
    // GPU max event-pipeline steps per batch. The historical hardcode
    // (10000) truncates the long many-bounce thermal tail in
    // low-absorption reflectors (Be), losing ~8% of thermal scatters
    // and biasing k low. Raise via `maxev=N` to test/fix.
    let mut arg_maxev: i32 = 10_000;
    for a in std::env::args().skip(2) {
        if let Some(v) = a.strip_prefix("b=") {
            arg_b = v.parse().unwrap_or(arg_b);
        } else if let Some(v) = a.strip_prefix("i=") {
            arg_i = v.parse().unwrap_or(arg_i);
        } else if let Some(v) = a.strip_prefix("p=") {
            arg_p = v.parse().unwrap_or(arg_p);
        } else if let Some(v) = a.strip_prefix("s=") {
            arg_s = v.parse().unwrap_or(arg_s);
        } else if let Some(v) = a.strip_prefix("r=") {
            arg_r = v.parse().unwrap_or(arg_r);
        } else if let Some(v) = a.strip_prefix("refill=") {
            arg_refill = v.parse().ok();
        } else if a == "auto_refill" {
            arg_auto_refill = true;
        } else if let Some(v) = a.strip_prefix("openmc=") {
            arg_openmc = Some(v.to_string());
        } else if a == "no_nxn_bank" {
            arg_nxn_mode = 2; // drop
        } else if a == "bank_as_fission" {
            arg_nxn_mode = 1; // legacy bank-as-fission
        } else if let Some(v) = a.strip_prefix("maxev=") {
            arg_maxev = v.parse().unwrap_or(arg_maxev);
        }
    }
    eprintln!("settings: batches={arg_b} inactive={arg_i} particles={arg_p} seed={arg_s} rank={arg_r} refill={arg_refill:?} auto_refill={arg_auto_refill} nxn_mode={arg_nxn_mode} maxev={arg_maxev}");
    let text = std::fs::read_to_string(&case_file).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();

    let loaded = scene_io::load_scene_from_json(&value["scene"].to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&data_dir());
    let resolved = material_resolve::resolve_materials(&loaded.materials, &lib, arg_r).unwrap();

    let mut cfg = SimConfig::default();
    cfg.batches = arg_b;
    cfg.inactive = arg_i;
    cfg.particles_per_batch = arg_p;
    cfg.seed = arg_s;
    cfg.verbose = false;
    // Sweep-faithful GPU population control (CPU path ignores these).
    cfg.gpu_refill_pool_factor = arg_refill;
    cfg.gpu_auto_refill = arg_auto_refill;

    let inactive = cfg.inactive;
    let ppb = cfg.particles_per_batch as u64;

    println!("=== CPU run ===");
    let cpu_runner = CpuRunner {
        geometry: &loaded.geometry,
        materials: &resolved.materials,
        xs_provider: &resolved.provider,
    };
    let cpu_outcome = cpu_runner.run(&cfg);
    let mut cpu_act = Active::default();
    for b in cpu_outcome.batches.iter().skip(inactive as usize) {
        cpu_act.add(b);
    }
    cpu_act.report("CPU", ppb);

    #[cfg(feature = "cuda")]
    {
        println!("\n=== GPU run ===");
        let awrs: Vec<f64> = resolved.provider.nuclides.iter().map(|n| n.awr).collect();
        let nu_bars: Vec<f64> = resolved
            .provider
            .nuclides
            .iter()
            .map(|n| n.nu_bar_const)
            .collect();
        let mat_k_t: Vec<f64> = resolved
            .materials
            .iter()
            .map(|m| m.temperature * K_B_EV_PER_K)
            .collect();
        let sab_nuc_idx: i32 = resolved
            .provider
            .thermal
            .iter()
            .position(|t| t.is_some())
            .map_or(-1, |i| i as i32);

        let gpu = GpuTransportContext::new().expect("GPU init");
        let nuc_data = gpu
            .upload_nuclide_data(&resolved.provider.nuclides, arg_r)
            .expect("upload nuclides");
        let q_n2ns: Vec<f64> = resolved.provider.nuclides.iter().map(|n| n.q_n2n).collect();
        let q_n3ns: Vec<f64> = resolved.provider.nuclides.iter().map(|n| n.q_n3n).collect();
        let mat_data = gpu
            .upload_material_data(&resolved.materials, &awrs, &nu_bars, &q_n2ns, &q_n3ns)
            .expect("upload materials");
        let n_nuc = resolved.provider.nuclides.len();
        let sab_data = if sab_nuc_idx >= 0 {
            let arc = resolved.provider.thermal[sab_nuc_idx as usize]
                .as_ref()
                .expect("sab");
            let t_idx = arc.select_temperature(
                loaded.materials[0].temperature,
                open_rust_mc::transport::sim_limits::SimLimits::default()
                    .sab_temperature_tolerance,
            );
            gpu.upload_sab_data(arc, t_idx, sab_nuc_idx as usize, n_nuc)
                .expect("upload S(α,β)")
        } else {
            gpu.upload_sab_data_empty(n_nuc).expect("empty S(α,β)")
        };
        let wmp_data = gpu
            .upload_wmp_data_empty(resolved.provider.nuclides.len())
            .expect("empty WMP");

        let rec = GpuRecursiveContext::build(&loaded.geometry, cfg.particles_per_batch as usize)
            .expect("GpuRecursiveContext");

        let geometry = loaded.geometry.clone();
        let cells = loaded.geometry.cells.clone();
        let runner = CudaRunner {
            recursive: &rec,
            transport: &gpu,
            nuc_data: &nuc_data,
            mat_data: &mat_data,
            sab_data: &sab_data,
            wmp_data: &wmp_data,
            mat_k_t: &mat_k_t,
            sab_nuc_idx,
            max_events_per_history: arg_maxev,
            fis_capacity: (cfg.particles_per_batch as usize) * 4,
            initial_source: Box::new(move |n, s| {
                let sites = open_rust_mc::transport::simulate::initial_source(
                    n, &geometry, &cells, s,
                );
                sites
                    .iter()
                    .map(|fs| (fs.pos.x, fs.pos.y, fs.pos.z, fs.energy))
                    .collect()
            }),
            buffers: std::cell::RefCell::new(None),
        refill: std::cell::RefCell::new(None),
            nxn_mode: arg_nxn_mode,
        };
        let gpu_outcome = runner.run(&cfg);
        let mut gpu_act = Active::default();
        for b in gpu_outcome.batches.iter().skip(inactive as usize) {
            gpu_act.add(b);
        }
        gpu_act.report("GPU", ppb);

        report_delta(&cpu_act, &gpu_act, ppb);

        // ── OpenMC reference (optional) ─────────────────────────
        // The companion script `scripts/openmc_godiva_tallies.py`
        // writes `outputs/openmc_godiva_tallies.json` from OpenMC
        // running the IDENTICAL HDF5 library on the IDENTICAL
        // Godiva geometry. If present, fold its k / leakage /
        // reaction-rate aggregates into the comparison so the
        // CPU↔GPU↔OpenMC three-way is visible at a glance.
        // Per-case OpenMC reference. `openmc=<path>` overrides the
        // legacy Godiva default — grading HMF-058 against Godiva's
        // OpenMC k (the old hardcode) is meaningless. Generate a
        // matching reference with scripts/openmc_scene_runner.py.
        let openmc_path = match &arg_openmc {
            Some(p) => std::path::PathBuf::from(p),
            None => workspace_root().join("outputs").join("openmc_godiva_tallies.json"),
        };
        if let Ok(text) = std::fs::read_to_string(&openmc_path) {
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            let k_omc = v["k_mean"].as_f64().unwrap_or(f64::NAN);
            let sigma_seeds = v["sigma_seeds"].as_f64().unwrap_or(f64::NAN);
            let particles = v["particles"].as_u64().unwrap_or(0);
            let seeds = v["seeds"].as_u64().unwrap_or(0);
            let batches = v["batches"].as_u64().unwrap_or(0);
            let inactive_o = v["inactive"].as_u64().unwrap_or(0);
            let active_batches = batches - inactive_o;
            let n_source = (seeds * active_batches * particles) as f64;
            println!("\n=== OpenMC reference ({}, {} seeds × {} active batches × {} particles = {:.0} active histories) ===",
                     openmc_path.display(), seeds, active_batches, particles, n_source);
            println!("  k_eff             : {k_omc:.5} ± {sigma_seeds:.5} (seed σ)   ({:+.0} pcm vs 1.0)", (k_omc - 1.0) * 1e5);

            // Leakage current per source — OpenMC tally "leakage", current score.
            if let Some(leak_t) = v["tallies_seed_mean"].get("leakage") {
                if let Some(mean) = leak_t["mean"].as_array().and_then(|a| a.first()).and_then(|x| x.as_f64()) {
                    println!("  leakage / source  : {mean:.4}    (OpenMC current score, per source particle)");
                }
            }
            // Production / source = ν·Σ_f integrated.
            if let Some(nuf) = v["tallies_seed_mean"].get("rate_nu-fission") {
                if let Some(arr) = nuf["mean"].as_array() {
                    let total: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
                    println!("  ν·fissions / src  : {total:.4}    (OpenMC nu-fission summed across nuclides)");
                }
            }
            if let Some(abs_t) = v["tallies_seed_mean"].get("rate_absorption") {
                if let Some(arr) = abs_t["mean"].as_array() {
                    let total: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
                    println!("  absorptions / src : {total:.4}");
                }
            }
            if let Some(fis_t) = v["tallies_seed_mean"].get("rate_fission") {
                if let Some(arr) = fis_t["mean"].as_array() {
                    let total: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
                    println!("  fissions / src    : {total:.4}");
                }
            }
            // Per-reaction OpenMC tallies for direct A/B against GPU
            // diagnostic counters. OpenMC reports per-nuclide per
            // reaction; we sum across nuclides for the macro rate.
            println!("  ─ Per-reaction OpenMC (rate × N_source equiv) ─");
            for (label, tname) in [
                ("elastic     / src ", "rate_elastic"),
                ("inelastic   / src ", "rate_scatter"),  // scatter score ≈ MT≠2
                ("MT=4 (inel) / src ", "rate_MT4"),
                ("MT=91 (cont)/ src ", "rate_MT91"),
                ("(n,γ)       / src ", "rate_(n,gamma)"),
                ("(n,2n)      / src ", "rate_(n,2n)"),
                ("(n,3n)      / src ", "rate_(n,3n)"),
                ("absorpt     / src ", "rate_absorption"),
            ] {
                if let Some(t) = v["tallies_seed_mean"].get(tname) {
                    if let Some(arr) = t["mean"].as_array() {
                        let total: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
                        println!("    {label}: {total:.4}");
                    }
                }
            }
            // Energy-resolved totals → compute ⟨E⟩ at fission, scatter
            // from the rate_by_energy tally. Bins are
            //   [0.0, 1e-1, 1e3, 1e5, 1e6, 2e6, 5e6, 2e7]
            // and the tally is laid out as
            //   mean[bin * n_scores + score]
            // with scores order = total, fission, absorption, scatter.
            if let Some(t) = v["tallies_seed_mean"].get("rate_by_energy") {
                if let (Some(arr), Some(groups_arr)) =
                    (t["mean"].as_array(), v["energy_groups_MeV"].as_array())
                {
                    let groups: Vec<f64> = groups_arr
                        .iter()
                        .filter_map(|g| g.as_f64())
                        .collect();
                    let centers: Vec<f64> = groups
                        .windows(2)
                        .map(|w| 0.5 * (w[0] + w[1]) * 1e6) // back to eV
                        .collect();
                    let n_scores = 4;
                    let n_bins = centers.len();
                    let mut total = vec![0.0_f64; n_bins];
                    let mut fission = vec![0.0_f64; n_bins];
                    let mut scatter = vec![0.0_f64; n_bins];
                    for (i, val) in arr.iter().filter_map(|x| x.as_f64()).enumerate() {
                        let bin = i / n_scores;
                        let score = i % n_scores;
                        if bin >= n_bins {
                            break;
                        }
                        match score {
                            0 => total[bin] = val,
                            1 => fission[bin] = val,
                            3 => scatter[bin] = val,
                            _ => {}
                        }
                    }
                    // σ from bin midpoints — coarse (only 7 bins) but
                    // directly comparable to the GPU's σ at fission.
                    let sigma_e = |rates: &[f64]| -> (f64, f64) {
                        let den: f64 = rates.iter().sum();
                        if den <= 0.0 {
                            return (0.0, 0.0);
                        }
                        let m: f64 =
                            rates.iter().zip(&centers).map(|(r, e)| r * e).sum::<f64>() / den;
                        let m2: f64 = rates
                            .iter()
                            .zip(&centers)
                            .map(|(r, e)| r * e * e)
                            .sum::<f64>()
                            / den;
                        (m, (m2 - m * m).max(0.0).sqrt())
                    };
                    let (m_fis, s_fis) = sigma_e(&fission);
                    let (m_sc, s_sc) = sigma_e(&scatter);
                    let (m_tot, s_tot) = sigma_e(&total);

                    // Fine-binned fission σ from the 100-bin log-spaced
                    // tally added in `scripts/openmc_godiva_tallies.py`.
                    // When present this is the actually-faithful OpenMC σ
                    // — the coarse 7-bin σ above is biased upward by wide
                    // bin widths and shouldn't be used for the A/B.
                    let fine = (|| {
                        let edges = v["fine_fission_groups_eV"].as_array()?;
                        let rates_t = v["tallies_seed_mean"].get("fission_by_energy_fine")?;
                        let rates_arr = rates_t["mean"].as_array()?;
                        let edges: Vec<f64> =
                            edges.iter().filter_map(|x| x.as_f64()).collect();
                        let rates: Vec<f64> =
                            rates_arr.iter().filter_map(|x| x.as_f64()).collect();
                        if edges.len() < 2 || rates.len() != edges.len() - 1 {
                            return None;
                        }
                        // Per-bin contribution to ⟨E⟩ and ⟨E²⟩ via the
                        // analytic average of E and E² over a flat-rate
                        // bin [e_lo, e_hi]: ∫E dE / (e_hi − e_lo) =
                        // (e_lo + e_hi) / 2 and ∫E² dE = (e_lo² + e_lo·e_hi
                        // + e_hi²) / 3. Flat-within-bin is the same
                        // assumption the histogram inherently makes; this
                        // beats the midpoint approximation for σ when the
                        // bin width is non-negligible.
                        let mut den = 0.0;
                        let mut sum_e = 0.0;
                        let mut sum_e2 = 0.0;
                        for (i, &r) in rates.iter().enumerate() {
                            if r <= 0.0 {
                                continue;
                            }
                            let lo = edges[i];
                            let hi = edges[i + 1];
                            let m_bin = 0.5 * (lo + hi);
                            let m2_bin = (lo * lo + lo * hi + hi * hi) / 3.0;
                            den += r;
                            sum_e += r * m_bin;
                            sum_e2 += r * m2_bin;
                        }
                        if den <= 0.0 {
                            return None;
                        }
                        let m = sum_e / den;
                        let m2 = sum_e2 / den;
                        let s = (m2 - m * m).max(0.0).sqrt();
                        Some((m, s))
                    })();

                    println!("  ⟨E⟩ and σ(E) from rate_by_energy (7-bin coarse, midpoints):");
                    println!(
                        "    fission : ⟨E⟩ = {:.4e}   σ = {:.4e}   σ/⟨E⟩ = {:.3}",
                        m_fis, s_fis, if m_fis > 0.0 { s_fis / m_fis } else { 0.0 }
                    );
                    println!(
                        "    scatter : ⟨E⟩ = {:.4e}   σ = {:.4e}",
                        m_sc, s_sc
                    );
                    println!(
                        "    total   : ⟨E⟩ = {:.4e}   σ = {:.4e}",
                        m_tot, s_tot
                    );

                    // Direct A/B vs GPU when the GPU run populated
                    // squared sums. CPU run leaves these at 0, which
                    // sigma_from_sums returns as (0,0) — suppressed.
                    let gpu_sigma = |a: &Active| -> (f64, f64, f64) {
                        if a.fis_sum == 0 {
                            return (0.0, 0.0, 0.0);
                        }
                        let n = a.fis_sum as f64;
                        let m = a.e_fis_in / n;
                        let m2 = a.e_fis_in_sq / n;
                        let s = (m2 - m * m).max(0.0).sqrt();
                        (m, s, if m > 0.0 { s / m } else { 0.0 })
                    };
                    let (cpu_m, cpu_s, cpu_r) = gpu_sigma(&cpu_act);
                    let (gpu_m, gpu_s, gpu_r) = gpu_sigma(&gpu_act);
                    if gpu_s > 0.0 || cpu_s > 0.0 {
                        // Pick the most faithful OpenMC σ available.
                        let (omc_m, omc_s, omc_label) = match fine {
                            Some((m, s)) => (m, s, "fine 100-bin"),
                            None => (m_fis, s_fis, "coarse 7-bin midpoint (BIASED HIGH)"),
                        };
                        println!("\n  ─ σ(E_in) at fission, three-way ─");
                        if cpu_s > 0.0 {
                            println!(
                                "    CPU    : ⟨E⟩ = {:.4e}   σ = {:.4e}   σ/⟨E⟩ = {:.3}",
                                cpu_m, cpu_s, cpu_r
                            );
                        }
                        if gpu_s > 0.0 {
                            println!(
                                "    GPU    : ⟨E⟩ = {:.4e}   σ = {:.4e}   σ/⟨E⟩ = {:.3}",
                                gpu_m, gpu_s, gpu_r
                            );
                        }
                        println!(
                            "    OpenMC : ⟨E⟩ = {:.4e}   σ = {:.4e}   σ/⟨E⟩ = {:.3}   ({omc_label})",
                            omc_m, omc_s, if omc_m > 0.0 { omc_s / omc_m } else { 0.0 }
                        );
                        if gpu_s > 0.0 {
                            println!(
                                "    Δσ_fis (GPU − OpenMC) = {:+.3e}   ({:+.2}% of OpenMC σ)",
                                gpu_s - omc_s,
                                if omc_s > 0.0 { (gpu_s - omc_s) / omc_s * 100.0 } else { 0.0 }
                            );
                        }
                        if cpu_s > 0.0 && gpu_s > 0.0 {
                            println!(
                                "    Δσ_fis (GPU − CPU)    = {:+.3e}   ({:+.2}% of CPU σ)",
                                gpu_s - cpu_s,
                                if cpu_s > 0.0 { (gpu_s - cpu_s) / cpu_s * 100.0 } else { 0.0 }
                            );
                        }
                        println!(
                            "    → If GPU σ ≈ CPU σ but both differ from OpenMC, the bias"
                        );
                        println!(
                            "      is a Rust-engine effect shared by both backends. If GPU"
                        );
                        println!(
                            "      σ differs from CPU σ, the bias is GPU-only and lives"
                        );
                        println!(
                            "      in event ordering / float-rounding / kernel layout."
                        );
                    }
                }
            }

            println!("\n=== Δ vs OpenMC ===");
            let cpu_k = cpu_act.k_sum / cpu_act.n as f64;
            let gpu_k = gpu_act.k_sum / gpu_act.n as f64;
            println!("  k_eff Δ        : CPU {:+.0} pcm   GPU {:+.0} pcm   (GPU − CPU = {:+.0} pcm)",
                     (cpu_k - k_omc) * 1e5, (gpu_k - k_omc) * 1e5, (gpu_k - cpu_k) * 1e5);
            if let Some(leak_t) = v["tallies_seed_mean"].get("leakage") {
                if let Some(leak_omc) = leak_t["mean"].as_array().and_then(|a| a.first()).and_then(|x| x.as_f64()) {
                    let nps = if cpu_act.hist_sum > 0 { cpu_act.hist_sum as f64 } else { (cpu_act.n as f64) * ppb as f64 };
                    let nps_g = if gpu_act.hist_sum > 0 { gpu_act.hist_sum as f64 } else { (gpu_act.n as f64) * ppb as f64 };
                    let cpu_leak = cpu_act.leak_sum as f64 / nps;
                    let gpu_leak = gpu_act.leak_sum as f64 / nps_g;
                    println!("  leakage/src Δ  : CPU {:+.4}   GPU {:+.4}   (cpu={:.4} gpu={:.4} omc={:.4})",
                             cpu_leak - leak_omc, gpu_leak - leak_omc, cpu_leak, gpu_leak, leak_omc);
                }
            }

            // ── Absorption-bucket reconciliation ─────────────────────
            // OpenMC reports the narrow (n,γ) MT=102 score separately
            // from charged-particle absorption ((n,α)/(n,p)/…), while
            // our engine lumps ALL non-fission absorption into a single
            // "capture" bucket (total − el − inel − n2n − n3n − fis).
            // Comparing our "capture" against OpenMC's bare "(n,γ)" looks
            // like a large gap that is purely a labeling artifact. The
            // apples-to-apples quantities are (a) total absorption and
            // (b) capture-equivalent = OpenMC(absorption − fission) =
            // (n,γ) + charged. Print the full decomposition so the
            // mismatch can't be misread as a physics difference.
            {
                let sum_tally = |name: &str| -> Option<f64> {
                    v["tallies_seed_mean"][name]["mean"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_f64()).sum::<f64>())
                };
                if let (Some(omc_abs), Some(omc_fis)) =
                    (sum_tally("rate_absorption"), sum_tally("rate_fission"))
                {
                    let omc_ngamma = sum_tally("rate_(n,gamma)").unwrap_or(f64::NAN);
                    let omc_charged = omc_abs - omc_fis - omc_ngamma;
                    let omc_cap_equiv = omc_abs - omc_fis; // (n,γ) + charged
                    let nps = if cpu_act.hist_sum > 0 { cpu_act.hist_sum as f64 } else { (cpu_act.n as f64) * ppb as f64 };
                    let nps_g = if gpu_act.hist_sum > 0 { gpu_act.hist_sum as f64 } else { (gpu_act.n as f64) * ppb as f64 };
                    let cpu_cap = cpu_act.abs_sum as f64 / nps;
                    let gpu_cap = gpu_act.abs_sum as f64 / nps_g;
                    let cpu_totabs = (cpu_act.abs_sum + cpu_act.fis_sum) as f64 / nps;
                    let gpu_totabs = (gpu_act.abs_sum + gpu_act.fis_sum) as f64 / nps_g;
                    println!("  ─ absorption-bucket reconciliation (read THIS before quoting a capture gap) ─");
                    println!("    OpenMC: total_abs {omc_abs:.4} = fission {omc_fis:.4} + (n,γ) {omc_ngamma:.4} + charged {omc_charged:.4}");
                    println!("    OpenMC capture-equiv ((n,γ)+charged = total_abs − fis) : {omc_cap_equiv:.4}");
                    println!("    our \"capture\" bucket (ALL non-fission abs)            : CPU {cpu_cap:.4}   GPU {gpu_cap:.4}");
                    println!("      → Δ capture-equiv : CPU {:+.4}   GPU {:+.4}   (compare vs capture-equiv, NOT bare (n,γ))",
                             cpu_cap - omc_cap_equiv, gpu_cap - omc_cap_equiv);
                    println!("    total absorption (cap + fis)                          : CPU {cpu_totabs:.4}   GPU {gpu_totabs:.4}   omc {omc_abs:.4}");
                    println!("      → Δ total-abs     : CPU {:+.4}   GPU {:+.4}",
                             cpu_totabs - omc_abs, gpu_totabs - omc_abs);
                }
            }
        } else {
            println!("\n(OpenMC reference JSON not found at {})", openmc_path.display());
            println!("To generate: (in WSL + conda env with openmc installed)");
            println!("    python scripts/openmc_godiva_tallies.py");
        }
    }

    #[cfg(not(feature = "cuda"))]
    println!("\n(CUDA feature disabled; run with `--features cuda`)");
}
