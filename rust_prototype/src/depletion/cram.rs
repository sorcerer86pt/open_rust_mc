// SPDX-License-Identifier: MIT
// Pusa 2016 CRAM coefficients: full f64 precision is the algorithm's
// correctness contract.
#![allow(clippy::excessive_precision)]
//! CRAM `exp(A)·n` evaluator, IPF form (Pusa 2016, OpenMC
//! `openmc/deplete/cram.py`):
//!
//! ```text
//! y = n
//! for k in 1..=K/2:
//!     solve (A − θ_k I) · w = y
//!     y = y + 2 · Re(α_k · w)
//! y *= α₀
//! ```
//!
//! IPF is multiplicative, NOT additive — don't substitute the
//! additive PF α_k. Order 16 (8 solves) is sufficient at PWR Δt
//! < 1e-10; order 48 (24 solves) for extreme stiffness or geologic
//! Δt. Dense complex LU below; swap to sparse for `N ≳ 200`.

use num_complex::Complex64;

/// `k/2` complex solves per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CramOrder {
    #[default]
    Cram16,
    Cram48,
}

/// IPF multiplicative end-of-loop scale (NOT the "limit at infinity").
pub const CRAM16_ALPHA0: f64 = 2.124_853_710_495_223_7e-16;
pub const CRAM48_ALPHA0: f64 = 2.258_038_182_743_983e-47;

/// 8 conjugate-pair folded poles. Each contributes
/// `2·Re(α_k·(A−θ_k I)⁻¹·n)`. From OpenMC `cram.py`, Pusa 2016
/// Table 1 IRA form.
pub const CRAM16_THETA: [Complex64; 8] = [
    Complex64::new(3.509_103_608_414_918, 8.436_198_985_884_374),
    Complex64::new(5.948_152_268_951_177, 3.587_457_362_018_322),
    Complex64::new(-5.264_971_343_442_647, 1.622_022_147_316_793_0e1),
    Complex64::new(1.419_375_897_185_666, 1.092_536_348_449_672_0e1),
    Complex64::new(6.416_177_699_099_435, 1.194_122_393_370_139),
    Complex64::new(4.993_174_737_717_997, 5.996_881_713_603_942),
    Complex64::new(-1.413_928_462_488_886, 1.349_772_569_889_275_0e1),
    Complex64::new(-1.084_391_707_869_699e1, 1.927_744_616_718_165_0e1),
];

/// Eight residues paired with `CRAM16_THETA`.
pub const CRAM16_ALPHA: [Complex64; 8] = [
    Complex64::new(5.464_930_576_870_210e3, -3.797_983_575_308_356e4),
    Complex64::new(9.045_112_476_907_548e1, -1.115_537_522_430_261e3),
    Complex64::new(2.344_818_070_467_641e2, -4.228_020_157_070_496e2),
    Complex64::new(9.453_304_067_358_312e1, -2.951_294_291_446_048e2),
    Complex64::new(7.283_792_954_673_409e2, -1.205_646_080_220_011e5),
    Complex64::new(3.648_229_059_594_851e1, -1.155_509_621_409_682e2),
    Complex64::new(2.547_321_630_156_819e1, -2.639_500_283_021_502e1),
    Complex64::new(2.394_538_338_734_709e1, -5.650_522_971_778_156),
];

/// 24 (folded) complex poles for CRAM-48. Same source / convention
/// as CRAM-16, an order higher.
pub const CRAM48_THETA: [Complex64; 24] = [
    Complex64::new(-4.465_731_934_165_702e1, 6.233_225_190_695_437e1),
    Complex64::new(-5.284_616_241_568_964, 4.057_499_381_311_059e1),
    Complex64::new(-8.867_715_667_624_458, 4.325_515_754_166_724e1),
    Complex64::new(3.493_013_124_279_215, 3.281_615_453_173_585e1),
    Complex64::new(1.564_102_508_858_634e1, 1.558_061_616_372_237e1),
    Complex64::new(1.742_097_597_385_893e1, 1.076_629_305_714_420e1),
    Complex64::new(-2.834_466_755_180_654e1, 5.492_841_024_648_724e1),
    Complex64::new(1.661_569_367_939_544e1, 1.316_994_930_024_688e1),
    Complex64::new(8.011_836_167_974_721, 2.780_232_111_309_410e1),
    Complex64::new(-2.056_267_541_998_229, 3.794_824_788_914_354e1),
    Complex64::new(1.449_208_170_441_839e1, 1.799_988_210_051_809e1),
    Complex64::new(1.853_807_176_907_916e1, 5.974_332_563_100_539),
    Complex64::new(9.932_562_704_505_182, 2.532_823_409_972_962e1),
    Complex64::new(-2.244_223_871_767_187e1, 5.179_633_600_312_162e1),
    Complex64::new(8.590_014_121_680_897e-1, 3.536_456_194_294_350e1),
    Complex64::new(-1.286_192_925_744_479e1, 4.600_304_902_833_652e1),
    Complex64::new(1.164_596_909_542_055e1, 2.287_153_304_140_217e1),
    Complex64::new(1.806_076_684_783_089e1, 8.368_200_580_099_821),
    Complex64::new(5.870_672_154_659_249, 3.029_700_159_040_121e1),
    Complex64::new(-3.542_938_819_659_747e1, 5.834_381_701_800_013e1),
    Complex64::new(1.901_323_489_060_250e1, 1.194_282_058_271_408),
    Complex64::new(1.885_508_331_552_577e1, 3.583_428_564_427_879),
    Complex64::new(-1.734_689_708_174_982e1, 4.883_941_101_108_207e1),
    Complex64::new(1.316_284_237_125_190e1, 2.042_951_874_827_759e1),
];

/// 24 residues paired with `CRAM48_THETA`.
pub const CRAM48_ALPHA: [Complex64; 24] = [
    Complex64::new(6.387_380_733_878_774e2, -6.743_912_502_859_256e2),
    Complex64::new(1.909_896_179_065_730e2, -3.973_203_432_721_332e2),
    Complex64::new(4.236_195_226_571_914e2, -2.041_233_768_918_671e3),
    Complex64::new(4.645_770_595_258_726e2, -1.652_917_287_299_683e3),
    Complex64::new(7.765_163_276_752_433e2, -1.783_617_639_907_328e4),
    Complex64::new(1.907_115_136_768_522e3, -5.887_068_595_142_284e4),
    Complex64::new(2.909_892_685_603_256e3, -9.953_255_345_514_560e3),
    Complex64::new(1.944_772_206_620_450e2, -1.427_131_226_068_449e3),
    Complex64::new(1.382_799_786_972_332e5, -3.256_885_197_214_938e6),
    Complex64::new(5.628_442_079_602_433e3, -2.924_284_515_884_309e4),
    Complex64::new(2.151_681_283_794_220e2, -1.121_774_011_188_224e3),
    Complex64::new(1.324_720_240_514_420e3, -6.370_088_443_140_973e4),
    Complex64::new(1.617_548_476_343_347e4, -1.008_798_413_156_542e6),
    Complex64::new(1.112_729_040_439_685e2, -8.837_109_731_680_418e1),
    Complex64::new(1.074_624_783_191_125e2, -1.457_246_116_408_180e2),
    Complex64::new(8.835_727_765_158_191e1, -6.388_286_188_419_360e1),
    Complex64::new(9.354_078_136_054_179e1, -2.195_424_319_460_237e2),
    Complex64::new(9.418_142_823_531_573e1, -6.719_055_740_098_035e2),
    Complex64::new(1.040_012_390_717_851e2, -1.693_747_595_553_868e2),
    Complex64::new(6.861_882_624_343_235e1, -1.177_598_523_430_493e1),
    Complex64::new(8.766_654_491_283_722e1, -4.596_464_999_363_902e3),
    Complex64::new(1.056_007_619_389_650e2, -1.738_294_585_524_067e3),
    Complex64::new(7.738_987_569_039_419e1, -4.311_715_386_228_984e1),
    Complex64::new(1.041_366_366_475_571e2, -2.777_743_732_451_969e2),
];

/// Compute `exp(A) · n` via CRAM at the given order (IPF form).
/// `A` is `n × n` row-major (real). Returns the real result vector.
/// Allocates `O(n²)` complex scratch once per pole.
///
/// Panics if `a` is not square or `n.len()` does not match `a`'s
/// dimension.
pub fn cram(order: CramOrder, a: &[f64], n_in: &[f64]) -> Vec<f64> {
    let (theta, alpha, alpha0) = match order {
        CramOrder::Cram16 => (
            CRAM16_THETA.as_slice(),
            CRAM16_ALPHA.as_slice(),
            CRAM16_ALPHA0,
        ),
        CramOrder::Cram48 => (
            CRAM48_THETA.as_slice(),
            CRAM48_ALPHA.as_slice(),
            CRAM48_ALPHA0,
        ),
    };
    cram_with_coefficients(a, n_in, theta, alpha, alpha0)
}

/// Convenience wrapper for `cram(CramOrder::Cram16, ...)`. Kept as
/// a stable name for callers that don't need to choose order.
pub fn cram16(a: &[f64], n_in: &[f64]) -> Vec<f64> {
    cram(CramOrder::Cram16, a, n_in)
}

/// Convenience wrapper for `cram(CramOrder::Cram48, ...)`.
pub fn cram48(a: &[f64], n_in: &[f64]) -> Vec<f64> {
    cram(CramOrder::Cram48, a, n_in)
}

/// Generic IPF CRAM evaluator. Same algorithm regardless of order;
/// only the `(θ_k, α_k, α₀)` table changes.
fn cram_with_coefficients(
    a: &[f64],
    n_in: &[f64],
    theta: &[Complex64],
    alpha: &[Complex64],
    alpha0: f64,
) -> Vec<f64> {
    let n = n_in.len();
    assert_eq!(
        a.len(),
        n * n,
        "matrix length {} does not match vector length squared {}",
        a.len(),
        n * n
    );
    assert_eq!(
        theta.len(),
        alpha.len(),
        "theta and alpha arrays must have the same length"
    );

    let mut y: Vec<f64> = n_in.to_vec();
    let mut m: Vec<Complex64> = vec![Complex64::default(); n * n];

    for (theta_k, alpha_k) in theta.iter().zip(alpha.iter()) {
        // Build M = A - θ_k I (complex).
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = Complex64::new(a[i * n + j], 0.0);
            }
            m[i * n + i] -= *theta_k;
        }

        // Solve M · w = y (current running y, not the original n_in).
        let y_complex: Vec<Complex64> = y.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let w = solve_complex_dense(&mut m, &y_complex, n);

        // y += 2 · Re(α_k · w).
        for (yi, wi) in y.iter_mut().zip(w.iter()) {
            *yi += 2.0 * (alpha_k * wi).re;
        }
    }

    // Final IPF normalization: y *= α₀.
    for yi in y.iter_mut() {
        *yi *= alpha0;
    }

    y
}

/// Dense complex Gaussian elimination with partial pivoting. `m` is
/// `n × n` row-major (consumed / overwritten); `b` is the RHS. The
/// pivot scan picks the largest |M[k..n, k]| at each step.
///
/// Robust enough for the small (`n ≤ ~200`), well-conditioned
/// matrices that arise in depletion at finite Δt.
fn solve_complex_dense(m: &mut [Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut x = b.to_vec();

    // Forward elimination with partial pivoting.
    for k in 0..n {
        // Find pivot row.
        let mut pivot_row = k;
        let mut pivot_mag = m[k * n + k].norm();
        for i in (k + 1)..n {
            let mag = m[i * n + k].norm();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = i;
            }
        }
        if pivot_row != k {
            // Swap rows in M.
            for j in 0..n {
                m.swap(k * n + j, pivot_row * n + j);
            }
            x.swap(k, pivot_row);
        }
        if pivot_mag == 0.0 {
            // Singular — fill with zeros and bail. Caller's matrix
            // wasn't well-conditioned; CRAM never hits this for a
            // physical transmutation matrix at non-zero Δt.
            return vec![Complex64::default(); n];
        }

        // Eliminate column k below row k.
        let pivot = m[k * n + k];
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                let val = m[k * n + j] * factor;
                m[i * n + j] -= val;
            }
            let val = x[k] * factor;
            x[i] -= val;
        }
    }

    // Back substitution.
    for k in (0..n).rev() {
        let pivot = m[k * n + k];
        let mut sum = x[k];
        for j in (k + 1)..n {
            sum -= m[k * n + j] * x[j];
        }
        x[k] = sum / pivot;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1×1 case: `exp(-λ Δt) · n0 = n0 · e^{-λ Δt}`. Verifies that
    /// CRAM-16 reproduces a pure exponential to ~1e-14 precision.
    #[test]
    fn pure_decay_one_nuclide_matches_analytical() {
        // Half-life 1.0 s, decay constant ln(2). Δt = 0.5 s.
        let lambda = 2.0_f64.ln();
        let dt = 0.5_f64;
        let a = vec![-lambda * dt];
        let n0 = vec![1.0_f64];
        let n_t = cram16(&a, &n0);
        let expected = (-lambda * dt).exp();
        assert!(
            (n_t[0] - expected).abs() < 1e-13,
            "got {}, expected {}",
            n_t[0],
            expected
        );
    }

    /// 2×2 chain `A → B`, with decay constants `λ_A`, `λ_B`. Bateman
    /// closed form:
    ///   N_A(t) = N_A(0) · exp(-λ_A t)
    ///   N_B(t) = (λ_A · N_A(0) / (λ_B - λ_A)) · (exp(-λ_A t) - exp(-λ_B t))
    ///          + N_B(0) · exp(-λ_B t)
    /// The transmutation matrix is `A = [[-λ_A, 0], [+λ_A, -λ_B]]`
    /// (column-major flattened: A[A→A], A[A→B], A[B→A], A[B→B] in
    /// row-major terms — daughter index gets the source rate
    /// contribution). Verifies the off-diagonal coupling.
    #[test]
    fn two_nuclide_chain_matches_bateman_analytical() {
        let lambda_a = 1.0_f64;
        let lambda_b = 0.3_f64;
        let dt = 1.5_f64;

        // Row-major A (size 2), with row i giving the rate of dN_i/dt:
        //   dN_A/dt = -λ_A · N_A
        //   dN_B/dt = +λ_A · N_A − λ_B · N_B
        let a = vec![-lambda_a * dt, 0.0, lambda_a * dt, -lambda_b * dt];
        let n0 = vec![1.0_f64, 0.0_f64];
        let n_t = cram16(&a, &n0);

        let n_a_expected = (-lambda_a * dt).exp();
        // Standard Bateman, daughter from a single-step parent:
        //   N_B(t) = λ_A / (λ_B − λ_A) · (e^{-λ_A t} − e^{-λ_B t}).
        // Both numerator and denominator flip sign together when
        // λ_A and λ_B are reversed, so the expression is symmetric.
        let n_b_expected =
            lambda_a / (lambda_b - lambda_a) * ((-lambda_a * dt).exp() - (-lambda_b * dt).exp());

        assert!(
            (n_t[0] - n_a_expected).abs() < 1e-12,
            "N_A: got {}, expected {}",
            n_t[0],
            n_a_expected
        );
        assert!(
            (n_t[1] - n_b_expected).abs() < 1e-12,
            "N_B: got {}, expected {}",
            n_t[1],
            n_b_expected
        );
    }

    /// CRAM-48 must reproduce the same single-nuclide answer as
    /// CRAM-16 to better than ~1e-13 relative.
    #[test]
    fn cram48_matches_cram16_on_pure_decay() {
        let lambda = 2.0_f64.ln();
        let dt = 0.5_f64;
        let a = vec![-lambda * dt];
        let n0 = vec![1.0_f64];
        let n_16 = cram16(&a, &n0);
        let n_48 = cram48(&a, &n0);
        let expected = (-lambda * dt).exp();
        assert!((n_16[0] - expected).abs() < 1e-13);
        assert!((n_48[0] - expected).abs() < 1e-13);
        assert!(
            (n_16[0] - n_48[0]).abs() < 1e-13,
            "CRAM-16 and CRAM-48 disagree by {}",
            (n_16[0] - n_48[0]).abs()
        );
    }

    /// Stiff test: CRAM-48 should outperform CRAM-16 when the chain
    /// spans many decades of time scale within a single Δt.
    /// Construct a 1×1 problem with a large argument λ·dt = 50
    /// (~30 decades of decay in one step). At λ·dt = 50,
    /// exp(-50) ≈ 1.93e-22 — close to where CRAM-16 starts losing
    /// relative precision while CRAM-48 stays at machine epsilon.
    #[test]
    fn cram48_more_accurate_than_cram16_at_extreme_dt() {
        let arg = 50.0_f64;
        let a = vec![-arg];
        let n0 = vec![1.0_f64];
        let n_16 = cram16(&a, &n0)[0];
        let n_48 = cram48(&a, &n0)[0];
        let expected = (-arg).exp();
        let err_16 = (n_16 - expected).abs() / expected.max(1e-300);
        let err_48 = (n_48 - expected).abs() / expected.max(1e-300);
        // Both are valid — we just want the order-48 error to be
        // no worse than order-16 in this regime.
        assert!(
            err_48 <= err_16 || err_48 < 1e-12,
            "CRAM-48 error {err_48:.3e} should not exceed CRAM-16 error {err_16:.3e}"
        );
    }

    /// 3-nuclide chain `A → B → C` (stable C). Tests that two-step
    /// daughter buildup is captured. C must accumulate everything
    /// removed from A and B.
    #[test]
    fn three_nuclide_chain_conserves_mass_at_long_time() {
        let lambda_a = 2.0_f64;
        let lambda_b = 1.0_f64;
        let dt = 10.0_f64; // long enough that A and B are nearly fully decayed
        let a = vec![
            -lambda_a * dt,
            0.0,
            0.0,
            lambda_a * dt,
            -lambda_b * dt,
            0.0,
            0.0,
            lambda_b * dt,
            0.0,
        ];
        let n0 = vec![1.0_f64, 0.0_f64, 0.0_f64];
        let n_t = cram16(&a, &n0);
        let total: f64 = n_t.iter().sum();
        // Mass conservation: N_A + N_B + N_C = N_A(0) = 1
        assert!(
            (total - 1.0).abs() < 1e-10,
            "total mass {} not conserved (residual {})",
            total,
            total - 1.0,
        );
        // C should hold ~all the mass after 10/λ_B time constants.
        assert!(
            n_t[2] > 0.999,
            "N_C should approach 1.0 at long time; got {}",
            n_t[2]
        );
    }
}
