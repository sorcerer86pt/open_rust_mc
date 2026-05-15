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
    workspace_root()
        .join("data")
        .join("endfb-vii.1-hdf5")
        .join("neutron")
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
        self.cap_sum += b.n_capture;
        self.e_fis_in += b.e_fis_in_sum;
        self.e_el_in += b.e_el_in_sum;
        self.e_inel_in += b.e_inel_in_sum;
        self.e_inel_out += b.e_inel_out_sum;
        self.e_fis_in_sq += b.e_fis_in_sq_sum;
        self.e_el_in_sq += b.e_el_in_sq_sum;
        self.e_inel_in_sq += b.e_inel_in_sq_sum;
        self.q_inel_sum += b.q_inel_sum;
    }

    fn report(&self, label: &str, particles_per_batch: u64) {
        let n = self.n as f64;
        let n_source = n * particles_per_batch as f64;
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
            println!("    fission   / src : {:.4}    ({} events)", fis / n_source, self.fis_sum);
            println!("    capture   / src : {:.4}    ({} events)", cap / n_source, self.cap_sum);
            let sum = el + inel + fis + cap;
            let recon = sum / self.coll_sum as f64;
            println!("    (n2n+n3n+...) / src = collisions − (el+inel+fis+cap) = {:.4}   reconciliation {:.4} of total coll",
                     (self.coll_sum as f64 - sum) / n_source, recon);
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
    let nps = (cpu.n as f64) * particles_per_batch as f64;
    let nps_g = (gpu.n as f64) * particles_per_batch as f64;
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
}

fn main() {
    let case_file = workspace_root()
        .join("bench/icsbep")
        .join("heu-met-fast-001_case-1.json");
    let text = std::fs::read_to_string(&case_file).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();

    let loaded = scene_io::load_scene_from_json(&value["scene"].to_string()).unwrap();
    let lib = NuclideLibrary::from_data_dir(&data_dir());
    let resolved = material_resolve::resolve_materials(&loaded.materials, &lib, 15).unwrap();

    let mut cfg = SimConfig::default();
    cfg.batches = 80;
    cfg.inactive = 20;
    cfg.particles_per_batch = 5_000;
    cfg.seed = 42;
    cfg.verbose = false;

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
            .upload_nuclide_data(&resolved.provider.nuclides, 15)
            .expect("upload nuclides");
        let mat_data = gpu
            .upload_material_data(&resolved.materials, &awrs, &nu_bars)
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
            max_events_per_history: 10_000,
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
        let openmc_path = workspace_root().join("outputs").join("openmc_godiva_tallies.json");
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
                ("elastic   / src ", "rate_elastic"),
                ("inelastic / src ", "rate_scatter"),  // scatter score ≈ MT≠2
                ("(n,γ)     / src ", "rate_(n,gamma)"),
                ("(n,2n)    / src ", "rate_(n,2n)"),
                ("(n,3n)    / src ", "rate_(n,3n)"),
                ("absorpt   / src ", "rate_absorption"),
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
                    let nps = (cpu_act.n as f64) * ppb as f64;
                    let nps_g = (gpu_act.n as f64) * ppb as f64;
                    let cpu_leak = cpu_act.leak_sum as f64 / nps;
                    let gpu_leak = gpu_act.leak_sum as f64 / nps_g;
                    println!("  leakage/src Δ  : CPU {:+.4}   GPU {:+.4}   (cpu={:.4} gpu={:.4} omc={:.4})",
                             cpu_leak - leak_omc, gpu_leak - leak_omc, cpu_leak, gpu_leak, leak_omc);
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
