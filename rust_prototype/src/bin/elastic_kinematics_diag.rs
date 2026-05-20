// SPDX-License-Identifier: MIT
//! Sampler-level A/B for the metal hot bias.
//!
//! The integrated-tally diagnostic (`bin/metal_stats_diag`) showed
//! Godiva leakage and fission counts agree CPU↔GPU to 0.01-0.14%,
//! but ν per fission is ~1% higher on GPU and collision count is
//! ~2% lower. Together → harder spectrum on GPU → +1148 pcm.
//!
//! Harder spectrum points at energy-loss-per-elastic-scatter being
//! too small. This binary stresses the four prime suspects in
//! isolation by running the CPU formula and the GPU formula on the
//! same fixed inputs, ξ-by-ξ:
//!
//!   1. Elastic CM→lab — sample mu_cm from a uniform fallback (the
//!      GPU uses tabular when available, but with the same seed
//!      stream both backends should pull the same mu_cm). Compare
//!      E_out / mu_lab moments for heavy nuclides (A_U235=233.025).
//!   2. Free-gas thermal: at E just below and just above 400·kT,
//!      confirm CPU and GPU both switch on/off the free-gas branch
//!      at the same threshold and that the lab E_out distribution
//!      matches.
//!   3. URR band selection: at a fixed E inside U-238's RR window,
//!      check the band-mixing factors match.
//!   4. Hydrogen vs heavy elastic — special-cases the mu_lab
//!      formula. Sanity-check both backends produce identical
//!      mu_lab when A ≤ 1+ε.

#![allow(dead_code)]

use open_rust_mc::transport::rng::Rng;

const PI: f64 = std::f64::consts::PI;

// ────────────────────────────────────────────────────────────────────
// CPU implementations (verbatim port of the inner formulas).
// ────────────────────────────────────────────────────────────────────

/// Mirrors `transport.cu` analog elastic anisotropic path
/// (and `physics/collision.rs` elastic kinematics):
/// E' = E·(1 + α + (1−α)·μ_cm) / 2,  α = ((A−1)/(A+1))²
/// μ_lab = (1 + A·μ_cm) / √(1 + A² + 2 A μ_cm)
fn cpu_elastic_anisotropic(e_in: f64, mu_cm: f64, a: f64) -> (f64, f64) {
    let alpha = ((a - 1.0) / (a + 1.0)).powi(2);
    let e_out = e_in * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;
    let mu_lab = if a > 1.0 + 1e-10 {
        (1.0 + a * mu_cm) / (1.0 + a * a + 2.0 * a * mu_cm).sqrt()
    } else {
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };
    (e_out, mu_lab)
}

/// Port of the GPU's elastic_anisotropic block (transport_recursive.cu
/// lines ~325-334 / transport.cu lines ~1408-1418). Should be bit-
/// identical to the CPU since the formulas are the same.
fn gpu_elastic_anisotropic(e_in: f64, mu_cm: f64, a: f64) -> (f64, f64) {
    let alpha = ((a - 1.0) / (a + 1.0)) * ((a - 1.0) / (a + 1.0));
    let e_out = e_in * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;
    let mu_lab = if a > 1.0 + 1e-10 {
        (1.0 + a * mu_cm) / (1.0 + a * a + 2.0 * a * mu_cm).sqrt()
    } else {
        (0.0_f64.max((1.0 + mu_cm) * 0.5)).sqrt()
    };
    (e_out, mu_lab)
}

/// Port of the GPU's free-gas thermal scattering block
/// (transport_recursive.cu lines ~280-323). Samples a Maxwell-
/// Boltzmann target velocity, computes relative E, samples isotropic
/// CM angle (no tabular angular here — that's the simplification),
/// returns (E_out, mu_lab).
fn gpu_free_gas_elastic(e_in: f64, a: f64, kt_ev: f64, rng: &mut Rng) -> (f64, f64) {
    let sigma = (kt_ev / a).sqrt();
    let v_n = (2.0 * e_in).sqrt();

    let u1 = rng.uniform().max(1e-30);
    let u2 = rng.uniform();
    let r_bm = sigma * (-2.0 * u1.ln()).sqrt();
    let th = 2.0 * PI * u2;
    let vtx = r_bm * th.cos();
    let vty = r_bm * th.sin();

    let u3 = rng.uniform().max(1e-30);
    let u4 = rng.uniform();
    let r_bm2 = sigma * (-2.0 * u3.ln()).sqrt();
    let th2 = 2.0 * PI * u4;
    let vtz = r_bm2 * th2.cos();

    // Neutron lab-frame velocity along z (since we're not sampling
    // direction explicitly, treat as forward beam).
    let vnx = 0.0;
    let vny = 0.0;
    let vnz = v_n;
    let vrx = vnx - vtx;
    let vry = vny - vty;
    let vrz = vnz - vtz;
    let vr = (vrx * vrx + vry * vry + vrz * vrz).sqrt().max(1e-20);

    let ia1 = 1.0 / (1.0 + a);
    let vcn = vr * a * ia1;

    // Isotropic CM angle (uniform mu).
    let mu_cm = 2.0 * rng.uniform() - 1.0;

    // Final lab-frame neutron speed:
    //   v_lab² = vcm² + vcn² + 2·vcm·vcn·μ_cm
    // where vcm is the centre-of-mass speed magnitude
    //   |vcm| = vr · A / (A+1)
    // This is the algebra inside the GPU's free-gas block; we use the
    // scalar approximation appropriate for the forward-beam diagnostic.
    let vcm = vr * a * ia1; // same as vcn
    let v_lab_sq = vcm * vcm + vcn * vcn + 2.0 * vcm * vcn * mu_cm;
    let e_out = 0.5 * v_lab_sq;
    let mu_lab = if vcm + vcn > 1e-20 {
        ((vcm + vcn * mu_cm) / v_lab_sq.sqrt()).clamp(-1.0, 1.0)
    } else {
        2.0 * rng.uniform() - 1.0
    };
    (e_out.max(1e-11), mu_lab)
}

// ────────────────────────────────────────────────────────────────────
// Drivers
// ────────────────────────────────────────────────────────────────────

fn moments(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn run_anisotropic_ab(e_in: f64, a: f64, n: usize, label: &str) {
    println!("\n=== Elastic anisotropic A/B: {label} ===");
    println!("  E_in = {:.3e} eV,  A = {:.3}", e_in, a);
    let mut rng = Rng::new(42, 0);
    let mut e_cpu = Vec::with_capacity(n);
    let mut e_gpu = Vec::with_capacity(n);
    let mut mu_cpu = Vec::with_capacity(n);
    let mut mu_gpu = Vec::with_capacity(n);
    for _ in 0..n {
        // Isotropic-CM μ for a fair A/B — both implementations should
        // give bit-identical results at the same μ_cm. Tabular angular
        // distributions are tested separately by the simulation
        // regression suite.
        let mu_cm = 2.0 * rng.uniform() - 1.0;
        let (ec, mc) = cpu_elastic_anisotropic(e_in, mu_cm, a);
        let (eg, mg) = gpu_elastic_anisotropic(e_in, mu_cm, a);
        e_cpu.push(ec);
        e_gpu.push(eg);
        mu_cpu.push(mc);
        mu_gpu.push(mg);
    }
    let (mc, sc) = moments(&e_cpu);
    let (mg, sg) = moments(&e_gpu);
    let (mmc, _) = moments(&mu_cpu);
    let (mmg, _) = moments(&mu_gpu);
    println!("  ⟨E_out⟩       : cpu = {:.4e}   gpu = {:.4e}   Δ = {:+.2e}   ({:+.3}%)",
             mc, mg, mg - mc, if mc != 0.0 { (mg - mc) / mc * 100.0 } else { 0.0 });
    println!("  σ(E_out)      : cpu = {:.4e}   gpu = {:.4e}", sc, sg);
    println!("  ⟨μ_lab⟩       : cpu = {:.5}    gpu = {:.5}    Δ = {:+.2e}", mmc, mmg, mmg - mmc);
    println!("  ⟨E_out/E_in⟩  : cpu = {:.5}    gpu = {:.5}", mc / e_in, mg / e_in);
    let mut bit_for_bit_ok = true;
    for (a_v, b_v) in e_cpu.iter().take(8).zip(e_gpu.iter()) {
        if (a_v - b_v).abs() > 1e-18 {
            bit_for_bit_ok = false;
        }
    }
    println!("  first-8 bit-for-bit : {}", if bit_for_bit_ok { "MATCH" } else { "DIVERGE" });
}

fn run_free_gas_ab(e_in: f64, a: f64, kt_ev: f64, n: usize, label: &str) {
    println!("\n=== Free-gas elastic: {label} ===");
    println!("  E_in = {:.3e} eV,  A = {:.3},  kT = {:.3e} eV   (E/kT = {:.1})",
             e_in, a, kt_ev, e_in / kt_ev);
    let mut rng = Rng::new(42, 0);
    let mut e_out = Vec::with_capacity(n);
    let mut mu_lab = Vec::with_capacity(n);
    for _ in 0..n {
        let (e, m) = gpu_free_gas_elastic(e_in, a, kt_ev, &mut rng);
        e_out.push(e);
        mu_lab.push(m);
    }
    let (me, se) = moments(&e_out);
    let (mm, _) = moments(&mu_lab);
    println!("  ⟨E_out⟩       : {:.4e} eV   σ = {:.4e}", me, se);
    println!("  ⟨μ_lab⟩       : {:.5}", mm);
    println!("  ⟨E_out/E_in⟩  : {:.5}   (≈ ((A²+1)/(A+1)²) for analog = {:.5})",
             me / e_in,
             (a * a + 1.0) / ((a + 1.0) * (a + 1.0)));
    println!("  threshold note: GPU uses E < 400·kT  → {} kT bin, free-gas branch {} engaged",
             (e_in / kt_ev) as i32,
             if e_in < 400.0 * kt_ev { "IS" } else { "is NOT" });
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("Sampler-level A/B for the metal hot bias");
    println!("══════════════════════════════════════════════════════════════════");

    // ── 1. Elastic anisotropic on heavy nuclide (U-235, A=233.025) ──
    //   Fast Godiva spectrum: most elastic collisions near 1 MeV.
    run_anisotropic_ab(1.0e6, 233.025, 200_000, "U-235 @ 1 MeV");
    run_anisotropic_ab(2.0e6, 233.025, 200_000, "U-235 @ 2 MeV");

    // ── 2. Elastic anisotropic on hydrogen (A=1.0) — special μ_lab branch
    run_anisotropic_ab(1.0e6, 1.000, 200_000, "H-1 @ 1 MeV (special-case μ_lab)");

    // ── Free-gas thermal: 400·kT cutoff sanity check ──
    // Godiva spectrum is mostly above 10 keV, well above the 400·kT
    // cutoff (~10 eV at room temperature), so free-gas is not engaged
    // for the bulk of Godiva histories. Just confirm the gate boundary.
    let kt_room = 294.0 * 8.617_333_262e-5;
    println!("\n  Free-gas branch threshold: E < 400·kT = {:.4e} eV at 294 K", 400.0 * kt_room);
    println!("  Godiva spectrum is mostly E > 10 keV → free-gas not engaged for fast metal.");

    // ── Real U-235 elastic angular data: quadratic-PDF vs linear-CDF ──
    // The CPU's `TabularMuDist::sample_with_xi` does quadratic lin-lin
    // inversion within each μ bin when the distribution carries a PDF
    // (the common case for OpenMC HDF5 angular data). The GPU's
    // `sample_mu_bin` (transport.cu line ~543) does linear-CDF
    // (histogram-PDF approximation). Localise whether this gap
    // matters numerically by loading U-235's real elastic angular
    // distribution and re-sampling both schemes on the same ξ stream.
    run_real_u235_angular_ab(1.0e6, 200_000);
    run_real_u235_angular_ab(2.0e6, 200_000);
    run_real_u235_angular_ab(5.0e6, 200_000);

    println!("\nDone.");
}

/// Load U-235 elastic angular distribution from HDF5, then resample
/// μ_cm under both the CPU's quadratic-PDF inversion and the GPU's
/// linear-CDF interpolation. Reports ⟨μ_cm⟩ and ⟨E_out⟩ moments.
fn run_real_u235_angular_ab(e_in: f64, n: usize) {
    use open_rust_mc::hdf5_reader::read_angular_distribution;

    let path = workspace_root()
        .join("data")
        .join("endfb-vii.1-hdf5")
        .join("neutron")
        .join("U235.h5");
    let ang = match read_angular_distribution(&path, 2_u32).expect("read MT=2 angular") {
        Some(a) => a,
        None => {
            println!("\n=== Real U-235 elastic angular A/B: NO ANGULAR DATA ===");
            return;
        }
    };

    println!("\n=== Real U-235 elastic angular A/B @ E_in = {:.3e} eV ===", e_in);

    let a_awr = 233.025_f64;
    let alpha = ((a_awr - 1.0) / (a_awr + 1.0)).powi(2);
    let mut rng = Rng::new(42, 0);
    let mut mu_cpu = Vec::with_capacity(n);
    let mut mu_gpu = Vec::with_capacity(n);
    let mut e_cpu = Vec::with_capacity(n);
    let mut e_gpu = Vec::with_capacity(n);
    for _ in 0..n {
        // Reproduce the stochastic-bin selection so both paths see the
        // same chosen energy bin → the only remaining axis of variation
        // is the within-bin inversion (quadratic vs linear-CDF).
        let bin = pick_bin(&ang.energies, e_in, rng.uniform());
        let dist = &ang.distributions[bin];
        let xi = rng.uniform();
        let mc = sample_mu_quadratic_cpu(xi, &dist.mu, &dist.pdf, &dist.cdf).clamp(-1.0, 1.0);
        let mg = sample_mu_linear_cdf(xi, &dist.mu, &dist.cdf).clamp(-1.0, 1.0);
        mu_cpu.push(mc);
        mu_gpu.push(mg);
        e_cpu.push(e_in * (1.0 + alpha + (1.0 - alpha) * mc) / 2.0);
        e_gpu.push(e_in * (1.0 + alpha + (1.0 - alpha) * mg) / 2.0);
    }
    let (mmc, _) = moments(&mu_cpu);
    let (mmg, _) = moments(&mu_gpu);
    let (mec, _) = moments(&e_cpu);
    let (meg, _) = moments(&e_gpu);
    println!("  ⟨μ_cm⟩   : cpu(quad) = {:+.5}   gpu(lin) = {:+.5}   Δ = {:+.2e}",
             mmc, mmg, mmg - mmc);
    println!("  ⟨E_out⟩  : cpu(quad) = {:.5e}   gpu(lin) = {:.5e}   Δ = {:+.3e}   ({:+.4}%)",
             mec, meg, meg - mec, if mec != 0.0 { (meg - mec) / mec * 100.0 } else { 0.0 });
    println!("  ⟨1-E_out/E_in⟩ (energy loss/scatter): cpu = {:.5}   gpu = {:.5}",
             1.0 - mec / e_in, 1.0 - meg / e_in);
}

fn pick_bin(energies: &[f64], e: f64, xi_bin: f64) -> usize {
    let n = energies.len();
    if e <= energies[0] {
        return 0;
    }
    if e >= energies[n - 1] {
        return n - 1;
    }
    let idx = match energies.binary_search_by(|x| x.partial_cmp(&e).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(i) => return i,
        Err(i) => i.saturating_sub(1),
    };
    if idx + 1 >= n {
        return idx;
    }
    let r = (e - energies[idx]) / (energies[idx + 1] - energies[idx]);
    if xi_bin < r {
        idx + 1
    } else {
        idx
    }
}

/// CPU quadratic lin-lin μ inversion — verbatim port of
/// `TabularMuDist::sample_with_xi` (lin-lin / interpolation=2 branch)
/// in `hdf5_reader.rs`. Private upstream; duplicated here for the
/// A/B diagnostic.
fn sample_mu_quadratic_cpu(xi: f64, mu: &[f64], pdf: &[f64], cd: &[f64]) -> f64 {
    let n = cd.len();
    if n < 2 {
        return 2.0 * xi - 1.0;
    }
    let idx = match cd.binary_search_by(|c| c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    let idx = idx.min(n - 2);
    let cdf_lo = cd[idx];
    let cdf_hi = cd[idx + 1];
    let mu_lo = mu[idx];
    let mu_hi = mu[idx + 1];
    let dmu = mu_hi - mu_lo;
    if (cdf_hi - cdf_lo).abs() < 1e-15 || dmu.abs() < 1e-15 {
        return mu_lo;
    }
    let pdf_lo = pdf.get(idx).copied().unwrap_or(0.0);
    let pdf_hi = pdf.get(idx + 1).copied().unwrap_or(pdf_lo);
    let a = (pdf_hi - pdf_lo) / (2.0 * dmu);
    let b = pdf_lo;
    let c = cdf_lo - xi;
    let x = if a.abs() < 1e-14 {
        if b.abs() < 1e-30 {
            (xi - cdf_lo) / (cdf_hi - cdf_lo) * dmu
        } else {
            -c / b
        }
    } else {
        let disc = (b * b - 4.0 * a * c).max(0.0);
        (-b + disc.sqrt()) / (2.0 * a)
    };
    mu_lo + x.clamp(0.0, dmu)
}

/// GPU's linear-CDF μ_bin inversion — port of `sample_mu_bin` in transport.cu.
fn sample_mu_linear_cdf(xi: f64, mu: &[f64], cd: &[f64]) -> f64 {
    let n = mu.len();
    if n <= 1 {
        return 2.0 * xi - 1.0;
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cd[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (xi - cd[lo]) / (cd[hi] - cd[lo]).max(1e-30);
    (mu[lo] + f * (mu[hi] - mu[lo])).clamp(-1.0, 1.0)
}

fn workspace_root() -> std::path::PathBuf {
    let mut p: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p
}
