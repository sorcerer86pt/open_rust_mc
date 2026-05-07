//! Gauss-Legendre quadrature tables — single source of truth.
//!
//! Hardcoded transcribed tables are a correctness risk: a typo in any
//! single digit will silently bias every integral that uses them. This
//! module addresses that with two safeguards:
//!
//! 1. **One copy, used everywhere.** Photon NEE, MoC quadrature in
//!    random-ray (when added), any future numerical-integration callers
//!    all import from here.
//! 2. **Validated against known integrals at test time.** The
//!    `tests` module integrates `1`, `x`, `x²`, ..., `x^(2N-1)` —
//!    every polynomial that an N-point Gauss-Legendre rule should
//!    integrate exactly — and asserts the result to machine precision.
//!    A typo in any node or weight breaks one of those tests.
//!
//! Source for the 16-point table values: Abramowitz & Stegun (1964),
//! *Handbook of Mathematical Functions*, Table 25.4 (corrected
//! edition). Cross-checked against the `gauss_quad` crate
//! (https://docs.rs/gauss-quad) and Wolfram Alpha for spot-checks of
//! `LegendreP[15, x]` roots.

/// 16-point Gauss-Legendre nodes on `[-1, 1]`.
pub const GL16_NODES: [f64; 16] = [
    -0.989_400_934_991_649_9,
    -0.944_575_023_073_232_5,
    -0.865_631_202_387_831_8,
    -0.755_404_408_355_003_0,
    -0.617_876_244_402_643_8,
    -0.458_016_777_657_227_4,
    -0.281_603_550_779_258_8,
    -0.095_012_509_837_637_44,
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_8,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003_0,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_5,
    0.989_400_934_991_649_9,
];

/// 16-point Gauss-Legendre weights on `[-1, 1]`. Sum to 2.
pub const GL16_WEIGHTS: [f64; 16] = [
    0.027_152_459_411_754_09,
    0.062_253_523_938_647_89,
    0.095_158_511_682_492_78,
    0.124_628_971_255_533_87,
    0.149_595_988_816_576_73,
    0.169_156_519_395_002_54,
    0.182_603_415_044_923_60,
    0.189_450_610_455_068_50,
    0.189_450_610_455_068_50,
    0.182_603_415_044_923_60,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_73,
    0.124_628_971_255_533_87,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_89,
    0.027_152_459_411_754_09,
];

/// Integrate `f` over `[a, b]` with 16-point Gauss-Legendre.
/// Exact for polynomials of degree ≤ 31.
#[inline]
pub fn integrate_gl16<F: Fn(f64) -> f64>(a: f64, b: f64, f: F) -> f64 {
    let half_w = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let mut acc = 0.0;
    for i in 0..16 {
        let x = mid + half_w * GL16_NODES[i];
        acc += GL16_WEIGHTS[i] * f(x);
    }
    half_w * acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-point GL is exact for polynomials up to degree 31. Test all
    /// monomials `x^k` for `k = 0..=31` against the analytic integral
    /// over `[-1, 1]`. A typo in any node or weight will fail one of
    /// these.
    #[test]
    fn gl16_integrates_polynomials_exactly_up_to_degree_31() {
        for k in 0..=31_usize {
            // ∫_{-1}^{1} x^k dx = 0 for odd k, 2/(k+1) for even k.
            let analytic = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            let mut got = 0.0;
            for i in 0..16 {
                got += GL16_WEIGHTS[i] * GL16_NODES[i].powi(k as i32);
            }
            let abs_err = (got - analytic).abs();
            // Polynomials up to degree 31 should be exact to machine
            // precision; some catastrophic cancellation at high k
            // pushes us to ~1e-13.
            assert!(
                abs_err < 1e-12,
                "k={k}: got {got}, analytic {analytic}, |err|={abs_err}"
            );
        }
    }

    #[test]
    fn gl16_weights_sum_to_two() {
        let sum: f64 = GL16_WEIGHTS.iter().sum();
        assert!((sum - 2.0).abs() < 1e-14);
    }

    #[test]
    fn gl16_nodes_are_symmetric() {
        // Gauss-Legendre nodes on [-1, 1] are symmetric about 0.
        for i in 0..8 {
            let lo = GL16_NODES[i];
            let hi = GL16_NODES[15 - i];
            assert!((lo + hi).abs() < 1e-14);
            // And matching weights.
            assert!((GL16_WEIGHTS[i] - GL16_WEIGHTS[15 - i]).abs() < 1e-14);
        }
    }

    #[test]
    fn integrate_gl16_against_analytic_unit_interval() {
        // ∫_0^1 x dx = 1/2
        let i1 = integrate_gl16(0.0, 1.0, |x| x);
        assert!((i1 - 0.5).abs() < 1e-14);
        // ∫_0^1 x^7 dx = 1/8
        let i7 = integrate_gl16(0.0, 1.0, |x| x.powi(7));
        assert!((i7 - 0.125).abs() < 1e-14);
        // ∫_0^π sin(x) dx = 2 — not polynomial, so within quadrature
        // error of ~1e-9 for this smooth integrand.
        let isin = integrate_gl16(0.0, std::f64::consts::PI, f64::sin);
        assert!((isin - 2.0).abs() < 1e-9);
    }
}
