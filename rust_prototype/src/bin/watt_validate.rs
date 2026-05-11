//! Validate the Watt fission-spectrum sampler from first principles.
//!
//! The Watt PDF is f(E) = K · exp(-E/a) · sinh(√(b·E)) for E > 0.
//! We compute:
//!   * the analytical 0th, 1st, 2nd moments via Simpson quadrature on
//!     a dense log-uniform grid, with K determined by ∫f(E)dE = 1.
//!   * the empirical 1st and 2nd moments from N draws of the engine's
//!     Watt sampler.
//!
//! If the sampler implements the PDF correctly, empirical and analytic
//! moments must agree to MC noise (∼ 1/√N).

use open_rust_mc::hdf5_reader::WattLaw;
use open_rust_mc::transport::rng::Rng;

fn watt_pdf_unnormalised(e: f64, a: f64, b: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    (-e / a).exp() * (b * e).sqrt().sinh()
}

/// Composite-trapezoid integration of `f` over a log-uniform grid
/// from `lo` to `hi`. The Watt distribution is well-behaved; we
/// don't need adaptive quadrature.
fn integrate<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, n: usize) -> f64 {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let h = (log_hi - log_lo) / (n as f64);
    let mut sum = 0.0;
    let mut prev_e = lo;
    let mut prev_v = f(lo) * prev_e; // dE = E · d(lnE) since dE/dlnE = E
    for i in 1..=n {
        let log_e = log_lo + (i as f64) * h;
        let e = log_e.exp();
        let v = f(e) * e;
        sum += 0.5 * (prev_v + v) * h;
        prev_v = v;
        let _ = prev_e;
        prev_e = e;
    }
    sum
}

fn analytic_moments(a: f64, b: f64) -> (f64, f64) {
    // Watt is rapidly decaying in E; integrate from 1e-3 a to 60 a
    // (e^-60 < 1e-26 — safe upper bound).
    let lo = 1e-3 * a;
    let hi = 60.0 * a;
    let n = 200_000;
    let z = integrate(|e| watt_pdf_unnormalised(e, a, b), lo, hi, n);
    let m1 = integrate(|e| e * watt_pdf_unnormalised(e, a, b), lo, hi, n);
    let m2 = integrate(|e| e * e * watt_pdf_unnormalised(e, a, b), lo, hi, n);
    (m1 / z, m2 / z)
}

fn empirical_moments(a: f64, b: f64, n: usize, seed: u64) -> (f64, f64) {
    let law = WattLaw {
        a_energies: vec![1.0],
        a_values: vec![a],
        b_energies: vec![1.0],
        b_values: vec![b],
        u: -1.0e15, // disable u cutoff for pure-Watt validation
    };
    let mut rng = Rng::new(seed, 0);
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for _ in 0..n {
        let e = law.sample(1.0, &mut rng); // E_in is irrelevant when a, b are constants
        s1 += e;
        s2 += e * e;
    }
    (s1 / n as f64, s2 / n as f64)
}

fn main() {
    println!("Watt sampler first-principles validation");
    println!("PDF: χ(E) = K · exp(-E/a) · sinh(√(b·E))");
    println!();
    // Three test cases: Cranberg U-235 prompt, ENDF delayed-soft,
    // U-233 ENDF/B-VII.1 mid-range Watt parameters.
    let cases = &[
        ("Cranberg U-235 prompt", 0.988e6, 2.249e-6),
        ("Delayed soft", 0.400e6, 2.249e-6),
        ("U-233 ENDF mid-range", 0.977e6, 2.546e-6),
    ];
    let n = 1_000_000;
    println!(
        "{:<28}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
        "case", "a (MeV)", "b (/MeV)", "⟨E⟩_analytic", "⟨E⟩_sample", "rel err", "⟨E²⟩ rel err"
    );
    for &(name, a, b) in cases {
        let (m1_a, m2_a) = analytic_moments(a, b);
        let (m1_e, m2_e) = empirical_moments(a, b, n, 0xC07D);
        let err1 = (m1_e - m1_a) / m1_a;
        let err2 = (m2_e - m2_a) / m2_a;
        println!(
            "{:<28}  {:>10.4}  {:>12.4}  {:>14.6e}  {:>14.6e}  {:>+14.4e}  {:>+14.4e}",
            name,
            a / 1.0e6,
            b * 1.0e6,
            m1_a,
            m1_e,
            err1,
            err2
        );
    }
    println!();
    println!("Pass criterion: |rel err| < 5/√N ≈ {:.3e} for N = {}", 5.0 / (n as f64).sqrt(), n);
}
