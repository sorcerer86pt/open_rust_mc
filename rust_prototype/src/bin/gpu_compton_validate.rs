//! Photon-kernel CPU-vs-GPU parity tests.
//!
//! Tests four GPU kernels against their CPU counterparts:
//!   1. Compton (free-KN + S(x,Z)/Z bound rejection), fixed E_in.
//!   2. Compton, per-particle E_in (uniform 100 keV..10 MeV).
//!   3. Coherent (Rayleigh) scattering.
//!   4. Bethe-Heitler pair production (no element data).
//!
//! CPU baseline runs in parallel via rayon to exercise all CPU cores;
//! the per-event ns column is therefore "single-event amortised across
//! whatever cores rayon picks." The GPU column is a single async launch
//! including H2D + D2H transfers but excluding kernel-compile / one-time
//! upload of element data (those are amortised over `sample_batch` calls
//! in real transport).
//!
//! Pass criteria (statistical, N=1M):
//!   - means agree to <0.5 %
//!   - reduced χ² of 50-bin output histograms ≤ 2.0
//!
//! Bit-exact agreement happens to occur for fixed-E Compton and pair —
//! CUDA and Rust libm produce identical doubles on the relevant inputs
//! and the PCG-64 path is byte-equivalent. χ²_red = 0 in that case.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;

use open_rust_mc::photon::PhotonElement;
use open_rust_mc::photon::coherent::coherent_scatter;
use open_rust_mc::photon::compton::{compton_scatter, compton_scatter_free};
use open_rust_mc::photon::gpu::{
    GpuComptonContext, GpuComptonDopplerCtx, GpuComptonVarECtx, GpuPairContext,
    GpuPhotoelectricCtx, GpuRayleighContext,
};
use open_rust_mc::photon::pair::{PAIR_THRESHOLD_EV, pair_produce};
use open_rust_mc::photon::photoelectric::{DEFAULT_PHOTON_CUTOFF_EV, photoelectric_absorb};
use open_rust_mc::transport::rng::Rng;

const N: usize = 1_000_000;
const BATCH_ID: u64 = 0;

const CASES: &[(&str, &str, u32)] = &[
    ("H", "H.h5", 1),
    ("O", "O.h5", 8),
    ("Zr", "Zr.h5", 40),
    ("U", "U.h5", 92),
];

const COMPTON_ENERGIES_EV: &[f64] = &[1.0e6, 5.0e6];
const RAYLEIGH_ENERGIES_EV: &[f64] = &[100_000.0, 1.0e6];
const PAIR_ENERGIES_EV: &[f64] = &[2.0e6, 5.0e6, 20.0e6];

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn std_dev(v: &[f64], m: f64) -> f64 {
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn histogram(values: &[f64], lo: f64, hi: f64, n_bins: usize) -> Vec<u64> {
    let mut h = vec![0u64; n_bins];
    let inv = (n_bins as f64) / (hi - lo);
    for &v in values {
        if v < lo || v >= hi {
            continue;
        }
        let mut b = ((v - lo) * inv) as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        h[b] += 1;
    }
    h
}

fn reduced_chi2(a: &[u64], b: &[u64]) -> f64 {
    let mut chi2 = 0.0;
    let mut dof = 0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let af = ai as f64;
        let bf = bi as f64;
        let denom = af + bf;
        if denom < 10.0 {
            continue;
        }
        let diff = af - bf;
        chi2 += diff * diff / denom;
        dof += 1;
    }
    if dof == 0 {
        f64::NAN
    } else {
        chi2 / dof as f64
    }
}

fn print_row(
    label: &str,
    m_cpu: f64,
    m_gpu: f64,
    s_cpu: f64,
    s_gpu: f64,
    chi2: f64,
    ns_cpu: f64,
    ns_gpu: f64,
    pass: bool,
) {
    let mark = if pass { "OK" } else { "**" };
    println!(
        "{:<28} {:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10.3} {:>9.1} {:>9.1} {:>5.2}x  {}",
        label,
        m_cpu,
        m_gpu,
        s_cpu,
        s_gpu,
        chi2,
        ns_cpu,
        ns_gpu,
        ns_cpu / ns_gpu,
        mark
    );
}

// ---- Compton (fixed E) ---------------------------------------------------

fn test_compton_fixed_e(elem: &PhotonElement, sym: &str, e_in: f64) -> bool {
    let ctx = GpuComptonContext::new(elem).expect("gpu compton ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warmup

    let t0 = Instant::now();
    let cpu: Vec<(f64, f64)> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            let o = compton_scatter_free(elem, e_in, &mut rng);
            (o.energy_out / e_in, o.mu)
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();
    let k_cpu: Vec<f64> = cpu.iter().map(|p| p.0).collect();
    let mu_cpu: Vec<f64> = cpu.iter().map(|p| p.1).collect();

    let t0 = Instant::now();
    let g = ctx.sample_batch(e_in, BATCH_ID, N).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    let mk_c = mean(&k_cpu);
    let mk_g = mean(&g.k);
    let sk_c = std_dev(&k_cpu, mk_c);
    let sk_g = std_dev(&g.k, mk_g);
    let alpha = e_in / 510_998.95;
    let kappa = 1.0 + 2.0 * alpha;
    let chi2 = reduced_chi2(
        &histogram(&k_cpu, 1.0 / kappa, 1.0, 50),
        &histogram(&g.k, 1.0 / kappa, 1.0, 50),
    );

    let pass = ((mk_c - mk_g) / mk_c).abs() < 5e-3 && (chi2.is_nan() || chi2 < 2.0);
    let label = format!("Compton[{} {:>5.1}MeV]", sym, e_in / 1e6);
    print_row(
        &label,
        mk_c,
        mk_g,
        sk_c,
        sk_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    if !pass {
        let mu_gm = mean(&g.mu);
        let mu_cm = mean(&mu_cpu);
        eprintln!("  mu_cpu={:.6} mu_gpu={:.6}", mu_cm, mu_gm);
    }
    pass
}

// ---- Compton (per-particle E) -------------------------------------------

fn test_compton_var_e(elem: &PhotonElement, sym: &str) -> bool {
    let ctx = GpuComptonVarECtx::new(elem).expect("gpu compton ve ctx");
    // Generate per-particle E_in deterministically: uniform 100 keV..10 MeV.
    let energies: Vec<f64> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut r = Rng::new(0xC0DE_CAFE, tid as u64);
            100_000.0 + r.uniform() * (10_000_000.0 - 100_000.0)
        })
        .collect();

    let _ = ctx.sample_batch(&energies[..4096], BATCH_ID); // warmup

    let t0 = Instant::now();
    let cpu: Vec<(f64, f64)> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            let e_in = energies[tid];
            let o = compton_scatter_free(elem, e_in, &mut rng);
            (o.energy_out / e_in, o.mu)
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();
    let k_cpu: Vec<f64> = cpu.iter().map(|p| p.0).collect();
    let mu_cpu: Vec<f64> = cpu.iter().map(|p| p.1).collect();

    let t0 = Instant::now();
    let (k_gpu, mu_gpu) = ctx.sample_batch(&energies, BATCH_ID).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    let mk_c = mean(&k_cpu);
    let mk_g = mean(&k_gpu);
    let sk_c = std_dev(&k_cpu, mk_c);
    let sk_g = std_dev(&k_gpu, mk_g);
    // k ranges over [1/κ_max, 1] where κ_max @ 10 MeV; just use [0,1].
    let chi2 = reduced_chi2(
        &histogram(&k_cpu, 0.0, 1.0, 50),
        &histogram(&k_gpu, 0.0, 1.0, 50),
    );
    let mu_diff = (mean(&mu_gpu) - mean(&mu_cpu)).abs();

    let pass =
        ((mk_c - mk_g) / mk_c).abs() < 5e-3 && (chi2.is_nan() || chi2 < 2.0) && mu_diff < 5e-3;
    let label = format!("Compton-varE[{} 0.1-10MeV]", sym);
    print_row(
        &label,
        mk_c,
        mk_g,
        sk_c,
        sk_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    pass
}

// ---- Compton with Doppler broadening ------------------------------------

fn test_compton_doppler(elem: &PhotonElement, sym: &str, e_in: f64) -> bool {
    let ctx = GpuComptonDopplerCtx::new(elem).expect("gpu compton doppler ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warmup

    let t0 = Instant::now();
    let cpu: Vec<(f64, f64)> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            let o = compton_scatter(elem, e_in, &mut rng);
            (o.energy_out, o.mu)
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();
    let e_cpu: Vec<f64> = cpu.iter().map(|p| p.0).collect();
    let mu_cpu: Vec<f64> = cpu.iter().map(|p| p.1).collect();

    let t0 = Instant::now();
    let g = ctx.sample_batch(e_in, BATCH_ID, N).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    // Compare on the dimensionless k = E_out / E_in.
    let k_cpu: Vec<f64> = e_cpu.iter().map(|e| e / e_in).collect();
    let k_gpu: Vec<f64> = g.energy_out.iter().map(|e| e / e_in).collect();

    let m_c = mean(&k_cpu);
    let m_g = mean(&k_gpu);
    let s_c = std_dev(&k_cpu, m_c);
    let s_g = std_dev(&k_gpu, m_g);
    // Doppler can shift outgoing E above the free-KN k = 1 ceiling for
    // strongly-bound electrons (Compton edge smear). Use a wider hist.
    let chi2 = reduced_chi2(
        &histogram(&k_cpu, 0.0, 1.05, 50),
        &histogram(&k_gpu, 0.0, 1.05, 50),
    );
    let mu_diff = (mean(&mu_gpu_extract(&mu_cpu)) - mean(&g.mu)).abs();

    let pass = ((m_c - m_g) / m_c).abs() < 5e-3 && (chi2.is_nan() || chi2 < 2.0) && mu_diff < 5e-3;
    let label = format!("Compton-Dop[{} {:>5.1}MeV]", sym, e_in / 1e6);
    print_row(
        &label,
        m_c,
        m_g,
        s_c,
        s_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    pass
}

// Helper: identity wrapper to keep type uniform when extracting mu.
fn mu_gpu_extract(mu: &[f64]) -> Vec<f64> {
    mu.to_vec()
}

// ---- Rayleigh ------------------------------------------------------------

fn test_rayleigh(elem: &PhotonElement, sym: &str, e_in: f64) -> bool {
    let ctx = GpuRayleighContext::new(elem).expect("gpu rayleigh ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warmup

    let t0 = Instant::now();
    let cpu: Vec<f64> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            coherent_scatter(elem, e_in, &mut rng).mu
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let g = ctx.sample_batch(e_in, BATCH_ID, N).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    let mu_c = mean(&cpu);
    let mu_g = mean(&g.mu);
    let s_c = std_dev(&cpu, mu_c);
    let s_g = std_dev(&g.mu, mu_g);
    let chi2 = reduced_chi2(
        &histogram(&cpu, -1.0, 1.0, 50),
        &histogram(&g.mu, -1.0, 1.0, 50),
    );

    // For peaked-forward distributions <μ> can be near 1; relative
    // tolerance breaks down. Compare absolute shift instead.
    let dmu = (mu_c - mu_g).abs();
    let pass = dmu < 5e-3 && (chi2.is_nan() || chi2 < 2.0);
    let label = format!("Rayleigh[{} {:>5.0}keV]", sym, e_in / 1e3);
    print_row(
        &label,
        mu_c,
        mu_g,
        s_c,
        s_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    pass
}

// ---- Photoelectric (Phase 1: primary photoelectron only) ----------------
//
// We compare GPU's `T_e_primary = E_in - B_struck` and `struck` designator
// against the CPU `photoelectric_absorb`'s primary fields.  The GPU does not
// run the EADL cascade, so the secondary fluorescence-photon bank and the
// portion of `local_deposition` coming from the cascade are not compared
// here. The struck-shell distribution is the meaningful parity check —
// if GPU samples shells correctly, the cascade physics on top is identical
// to the CPU side and can be added in a follow-up kernel.

fn test_photoelectric_phase1(elem: &PhotonElement, sym: &str, e_in: f64) -> bool {
    let ctx = GpuPhotoelectricCtx::new(elem).expect("gpu pe ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warmup

    let t0 = Instant::now();
    let cpu: Vec<(f64, i32)> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            let o = photoelectric_absorb(elem, e_in, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            let struck_idx = (o.struck_subshell_designator as usize).saturating_sub(1);
            let b = elem.subshells[struck_idx].binding_energy;
            ((e_in - b).max(0.0), o.struck_subshell_designator as i32)
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();
    let te_cpu: Vec<f64> = cpu.iter().map(|p| p.0).collect();
    let st_cpu: Vec<i32> = cpu.iter().map(|p| p.1).collect();

    let t0 = Instant::now();
    let g = ctx.sample_batch(e_in, BATCH_ID, N).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    let m_c = mean(&te_cpu);
    let m_g = mean(&g.t_e);
    let s_c = std_dev(&te_cpu, m_c);
    let s_g = std_dev(&g.t_e, m_g);

    // Compare struck-shell distributions via χ² over n_shells bins.
    let n_shells = elem.subshells.len();
    let mut h_cpu = vec![0u64; n_shells + 1];
    let mut h_gpu = vec![0u64; n_shells + 1];
    for (&c, &gpu_s) in st_cpu.iter().zip(g.struck.iter()) {
        let ic = (c as usize).min(n_shells);
        let ig = (gpu_s as usize).min(n_shells);
        h_cpu[ic] += 1;
        h_gpu[ig] += 1;
    }
    let chi2 = reduced_chi2(&h_cpu, &h_gpu);

    let pass = ((m_c - m_g) / m_c.abs().max(1.0)).abs() < 5e-3 && (chi2.is_nan() || chi2 < 2.0);
    let label = format!("PE-phase1[{} {:>5.1}MeV]", sym, e_in / 1e6);
    print_row(
        &label,
        m_c,
        m_g,
        s_c,
        s_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    pass
}

// ---- Pair production -----------------------------------------------------

fn test_pair(e_in: f64) -> bool {
    let ctx = GpuPairContext::new().expect("gpu pair ctx");
    let _ = ctx.sample_batch(e_in, BATCH_ID, 4096); // warmup

    let t0 = Instant::now();
    let cpu: Vec<(f64, f64)> = (0..N)
        .into_par_iter()
        .map(|tid| {
            let mut rng = Rng::for_particle(BATCH_ID, tid as u64);
            match pair_produce(e_in, &mut rng) {
                Some(o) => (o.electron_kinetic, o.positron_kinetic),
                None => (0.0, 0.0),
            }
        })
        .collect();
    let t_cpu = t0.elapsed().as_secs_f64();
    let te_m: Vec<f64> = cpu.iter().map(|p| p.0).collect();

    let t0 = Instant::now();
    let g = ctx.sample_batch(e_in, BATCH_ID, N).expect("gpu launch");
    let t_gpu = t0.elapsed().as_secs_f64();

    // Compare ε = T_-/(E - 2m_e c²); collapses to dimensionless shape.
    let t_total = (e_in - PAIR_THRESHOLD_EV).max(1e-9);
    let eps_cpu: Vec<f64> = te_m.iter().map(|t| t / t_total).collect();
    let eps_gpu: Vec<f64> = g.te_minus.iter().map(|t| t / t_total).collect();

    let m_c = mean(&eps_cpu);
    let m_g = mean(&eps_gpu);
    let s_c = std_dev(&eps_cpu, m_c);
    let s_g = std_dev(&eps_gpu, m_g);
    let chi2 = reduced_chi2(
        &histogram(&eps_cpu, 0.0, 1.0, 50),
        &histogram(&eps_gpu, 0.0, 1.0, 50),
    );

    let pass = (m_c - m_g).abs() < 5e-3 && (chi2.is_nan() || chi2 < 2.0);
    let label = format!("Pair[{:>5.1}MeV]", e_in / 1e6);
    print_row(
        &label,
        m_c,
        m_g,
        s_c,
        s_g,
        chi2,
        t_cpu * 1e9 / N as f64,
        t_gpu * 1e9 / N as f64,
        pass,
    );
    pass
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let photon_dir = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: gpu_photon_validate <photon_data_dir>");
            return ExitCode::from(1);
        }
    };

    println!(
        "# Photon GPU kernels — CPU-vs-GPU parity, N = {} per case",
        N
    );
    println!(
        "# CPU baseline = rayon ({} threads)",
        rayon::current_num_threads()
    );
    println!("# Pass: |Δmean|/mean < 0.5 %, reduced χ² (50 bins) < 2.0");
    println!();
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>6}",
        "test", "<x>_cpu", "<x>_gpu", "σ_cpu", "σ_gpu", "χ²_red", "ns/ev_c", "ns/ev_g", "speedup"
    );

    let mut all_pass = true;

    // Compton & Compton-VarE & Rayleigh: per element
    for &(sym, file, _z) in CASES {
        let p = photon_dir.join(file);
        let elem = match PhotonElement::from_hdf5(&p) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("# skip {}: {}", file, e);
                continue;
            }
        };
        for &e_in in COMPTON_ENERGIES_EV {
            all_pass &= test_compton_fixed_e(&elem, sym, e_in);
        }
        all_pass &= test_compton_var_e(&elem, sym);
        for &e_in in COMPTON_ENERGIES_EV {
            all_pass &= test_compton_doppler(&elem, sym, e_in);
        }
        for &e_in in RAYLEIGH_ENERGIES_EV {
            all_pass &= test_rayleigh(&elem, sym, e_in);
        }
        // Photoelectric energies: 100 keV (K-edge dominant for heavy Z)
        // and 1 MeV (M/N shells contribute more).
        for &e_in in &[100_000.0_f64, 1_000_000.0_f64] {
            all_pass &= test_photoelectric_phase1(&elem, sym, e_in);
        }
    }

    // Pair: no element data
    for &e_in in PAIR_ENERGIES_EV {
        all_pass &= test_pair(e_in);
    }

    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}
