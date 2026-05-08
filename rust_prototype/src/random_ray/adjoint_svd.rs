//! SVD compression of the random-ray adjoint flux ψ*(r,g) for
//! weight-window storage (Evans 2020 method).
//!
//! Random-ray returns `phi[fsr * n_groups + g]` — a row-major
//! [n_fsrs × n_groups] matrix that is highly compressible by SVD when
//! the spatial structure correlates across groups (typical for
//! shielding problems where ψ* is dominated by a single attenuation
//! profile modulated per group).
//!
//! Cartesian-mesh problems with [n_x, n_y, n_z] FSRs and 1 group can
//! also be SVD-compressed by reshaping the flat array. Two reshapes
//! are useful and exposed:
//!
//! - `[n_x * n_y, n_z]` — separates streaming axis (z) from the
//!   transverse plane. The slab benchmark in `rr_cadis_slab` falls
//!   here trivially: the whole adjoint is rank-1 in this reshape
//!   because the slab is symmetric in x,y.
//! - `[n_x, n_y * n_z]` — separates one transverse axis from the
//!   rest. Useful for ducted / streaming geometries.
//!
//! The compressed representation stores `(U[m × k], S[k], V^T[k × n])`
//! at rank `k`. Memory is `(m + 1 + n) · k · 8` bytes vs the dense
//! `m · n · 8` bytes — a factor of `m·n / ((m+n+1)·k)` reduction.
//!
//! `compute_compression(m, n, k)` returns the byte counts so callers
//! can pick `k` for a target byte budget.

use serde::{Deserialize, Serialize};

use crate::decompose::{SvdResult, svd};

/// Whether the SVD factors live in linear space or log10 space.
///
/// `Log10` is the Evans 2020 recommendation for adjoint-flux
/// compression: low-flux voxels (where linear-space SVD reconstruction
/// inflates max-rel-err by orders of magnitude due to divide-by-near-
/// zero) are well-conditioned in log space because adjoint flux spans
/// many decades in shielding problems. Use `Linear` for matrices with
/// signed entries or values bounded away from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpaceMode {
    Linear,
    Log10,
}

/// SVD-compressed adjoint flux. Stores the truncation factors, the
/// original matrix shape, and the working space so `reconstruct` can
/// invert the transform automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjointSvd {
    /// Number of rows in the dense matrix (e.g. n_fsrs).
    pub n_rows: usize,
    /// Number of columns (e.g. n_groups).
    pub n_cols: usize,
    /// Truncation rank actually used (≤ min(n_rows, n_cols)).
    pub rank: usize,
    /// Working space the factors live in. `reconstruct` applies the
    /// inverse transform after the dense reconstruction so the
    /// returned matrix is always in linear units.
    #[serde(default = "default_space")]
    pub space: SpaceMode,
    /// Left singular vectors, row-major [n_rows × rank].
    pub u: Vec<f64>,
    /// Singular values (descending), length = rank.
    pub s: Vec<f64>,
    /// Right singular vectors transposed, row-major [rank × n_cols].
    pub vt: Vec<f64>,
}

fn default_space() -> SpaceMode {
    SpaceMode::Linear
}

/// Reconstruction error metrics for a given truncation rank.
#[derive(Debug, Clone, Copy)]
pub struct AdjointReconError {
    /// Max absolute error across all entries.
    pub max_abs: f64,
    /// Max relative error (|a - b| / max(|a|, eps)).
    pub max_rel: f64,
    /// L2 norm of the residual.
    pub frob_residual: f64,
    /// L2 norm of the original matrix.
    pub frob_orig: f64,
    /// `frob_residual / frob_orig` — the "energy retained" complement.
    pub frob_rel: f64,
}

impl AdjointSvd {
    /// Compute the SVD of an `n_rows × n_cols` row-major matrix in
    /// linear space, truncated to `rank`. Convenience for callers
    /// that don't need log-space.
    pub fn compress(matrix: &[f64], n_rows: usize, n_cols: usize, rank: usize) -> Self {
        Self::compress_with_space(matrix, n_rows, n_cols, rank, SpaceMode::Linear)
    }

    /// Compute the SVD in `space` (linear or log10) and truncate to
    /// `rank`. The factors are stored in `space`; `reconstruct`
    /// inverts the transform automatically so callers always see
    /// linear-space output.
    pub fn compress_with_space(
        matrix: &[f64],
        n_rows: usize,
        n_cols: usize,
        rank: usize,
        space: SpaceMode,
    ) -> Self {
        assert_eq!(
            matrix.len(),
            n_rows * n_cols,
            "matrix length {} doesn't match {} × {} = {}",
            matrix.len(),
            n_rows,
            n_cols,
            n_rows * n_cols
        );
        let work: Vec<f64> = match space {
            SpaceMode::Linear => matrix.to_vec(),
            SpaceMode::Log10 => matrix.iter().map(|v| v.max(1e-30).log10()).collect(),
        };
        let svd_result: SvdResult = svd(&work, n_rows, n_cols);
        let rank = rank.min(svd_result.rank).max(1);

        let mut u = vec![0.0_f64; n_rows * rank];
        for i in 0..n_rows {
            for j in 0..rank {
                u[i * rank + j] = svd_result.u[i * svd_result.rank + j];
            }
        }
        let s = svd_result.s[..rank].to_vec();
        let mut vt = vec![0.0_f64; rank * n_cols];
        for j in 0..rank {
            for c in 0..n_cols {
                vt[j * n_cols + c] = svd_result.vt[j * n_cols + c];
            }
        }

        Self {
            n_rows,
            n_cols,
            rank,
            space,
            u,
            s,
            vt,
        }
    }

    /// Reconstruct the dense `n_rows × n_cols` matrix at the stored
    /// rank, inverting the working-space transform so the result is
    /// always in linear units.
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            for c in 0..self.n_cols {
                let mut acc = 0.0_f64;
                for j in 0..self.rank {
                    let u_ij = self.u[i * self.rank + j];
                    let s_j = self.s[j];
                    let vt_jc = self.vt[j * self.n_cols + c];
                    acc = (u_ij * s_j).mul_add(vt_jc, acc);
                }
                out[i * self.n_cols + c] = acc;
            }
        }
        if self.space == SpaceMode::Log10 {
            for v in &mut out {
                *v = 10.0_f64.powf(*v);
            }
        }
        out
    }

    /// Bytes used by the compressed representation (factors only).
    /// Excludes serialization overhead.
    pub fn bytes_compressed(&self) -> usize {
        (self.u.len() + self.s.len() + self.vt.len()) * std::mem::size_of::<f64>()
    }

    /// Bytes that the dense matrix would consume.
    pub fn bytes_dense(&self) -> usize {
        self.n_rows * self.n_cols * std::mem::size_of::<f64>()
    }
}

/// Compute reconstruction error of `recon` relative to `orig`. Both
/// are row-major dense arrays of length `n_rows * n_cols`.
pub fn recon_error(orig: &[f64], recon: &[f64]) -> AdjointReconError {
    assert_eq!(orig.len(), recon.len());
    let eps = 1.0e-300_f64;
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut frob_residual = 0.0_f64;
    let mut frob_orig = 0.0_f64;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        let rel = d / a.abs().max(eps);
        if rel > max_rel {
            max_rel = rel;
        }
        frob_residual += d * d;
        frob_orig += a * a;
    }
    let frob_residual = frob_residual.sqrt();
    let frob_orig = frob_orig.sqrt();
    let frob_rel = if frob_orig > 0.0 {
        frob_residual / frob_orig
    } else {
        0.0
    };
    AdjointReconError {
        max_abs,
        max_rel,
        frob_residual,
        frob_orig,
        frob_rel,
    }
}

/// Theoretical byte counts for `(m × n)` dense vs rank-`k`
/// factored storage. Factored count includes one f64 per singular
/// value plus the U and V^T blocks.
pub fn compression_bytes(n_rows: usize, n_cols: usize, rank: usize) -> (usize, usize) {
    let dense = n_rows * n_cols * std::mem::size_of::<f64>();
    let factored = (n_rows * rank + rank + rank * n_cols) * std::mem::size_of::<f64>();
    (dense, factored)
}

/// Choice between dense and SVD-factored storage. Returned by
/// `pick_representation` so callers can serialize whichever is
/// smaller while staying inside an accuracy budget.
#[derive(Debug, Clone)]
pub enum AdjointRepr {
    /// Dense `[n_rows × n_cols]` row-major matrix. Use when the
    /// matrix is too small for SVD to break even, when no rank
    /// satisfies the accuracy tolerance, or when the caller asked
    /// for an exact representation.
    Dense {
        n_rows: usize,
        n_cols: usize,
        data: Vec<f64>,
    },
    /// Truncated SVD factors. Use when the SVD-factored byte count
    /// is strictly smaller than dense AND the Frobenius rel error
    /// at that rank is ≤ `frob_tol`.
    Svd(AdjointSvd),
}

impl AdjointRepr {
    pub fn bytes(&self) -> usize {
        match self {
            AdjointRepr::Dense { data, .. } => data.len() * std::mem::size_of::<f64>(),
            AdjointRepr::Svd(s) => s.bytes_compressed(),
        }
    }

    pub fn rank(&self) -> Option<usize> {
        match self {
            AdjointRepr::Dense { .. } => None,
            AdjointRepr::Svd(s) => Some(s.rank),
        }
    }

    /// Reconstruct the dense `[n_rows × n_cols]` matrix.
    pub fn reconstruct(&self) -> Vec<f64> {
        match self {
            AdjointRepr::Dense { data, .. } => data.clone(),
            AdjointRepr::Svd(s) => s.reconstruct(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            AdjointRepr::Dense { n_rows, n_cols, .. } => (*n_rows, *n_cols),
            AdjointRepr::Svd(s) => (s.n_rows, s.n_cols),
        }
    }
}

/// Which working space(s) the picker should consider.
#[derive(Debug, Clone, Copy)]
pub enum PickerSpace {
    /// Try linear-space SVD only.
    Linear,
    /// Try log10-space SVD only. Recommended only for matrices with
    /// strictly positive entries (e.g. flux fields).
    Log10,
    /// Try both and return whichever gives the smallest byte count
    /// while meeting the tolerance. Default for ψ*(r,g) — the
    /// high-dynamic-range case where log10 may help, balanced
    /// against the fact that the tolerance is in linear units.
    Both,
}

/// Pick the smallest representation of `matrix` that satisfies
/// `frob_tol` (relative Frobenius reconstruction error against the
/// linear-space original). For each rank `1..=max_rank` and each
/// candidate space, builds the SVD and tests whether its linear
/// reconstruction error meets the tolerance. Returns the SVD with
/// the smallest byte count that qualifies; falls back to dense if
/// none does.
pub fn pick_representation(
    matrix: &[f64],
    n_rows: usize,
    n_cols: usize,
    max_rank: usize,
    frob_tol: f64,
    space: PickerSpace,
) -> AdjointRepr {
    assert_eq!(matrix.len(), n_rows * n_cols);
    let max_rank = max_rank.min(n_rows).min(n_cols);
    let dense_bytes = n_rows * n_cols * std::mem::size_of::<f64>();

    let candidates: &[SpaceMode] = match space {
        PickerSpace::Linear => &[SpaceMode::Linear],
        PickerSpace::Log10 => &[SpaceMode::Log10],
        PickerSpace::Both => &[SpaceMode::Linear, SpaceMode::Log10],
    };

    let mut best: Option<AdjointSvd> = None;
    for &mode in candidates {
        for rank in 1..=max_rank {
            let svd = AdjointSvd::compress_with_space(matrix, n_rows, n_cols, rank, mode);
            let bytes = svd.bytes_compressed();
            if bytes >= dense_bytes {
                break;
            }
            let recon = svd.reconstruct();
            if recon_error(matrix, &recon).frob_rel > frob_tol {
                continue;
            }
            // Qualifies — keep if smaller than the running best.
            if best.as_ref().is_none_or(|b| bytes < b.bytes_compressed()) {
                best = Some(svd);
            }
            // For this space, no point going to higher rank — they
            // only get bigger and the reconstruction can only improve.
            break;
        }
    }

    match best {
        Some(svd) => AdjointRepr::Svd(svd),
        None => AdjointRepr::Dense {
            n_rows,
            n_cols,
            data: matrix.to_vec(),
        },
    }
}

/// Reshape utilities for using SVD on Cartesian-mesh adjoint flux.
/// `phi` is laid out as `[ix * (n_y * n_z) + iy * n_z + iz]` (the
/// `WeightWindow::index` convention). This module reshapes the flat
/// array into a 2D matrix without copying when possible.
pub mod reshape {
    /// `[n_x * n_y, n_z]` — streaming-axis reshape. Best for slab /
    /// duct geometries where ψ* is dominated by one direction.
    pub fn xy_z(phi: &[f64], n_x: usize, n_y: usize, n_z: usize) -> (usize, usize) {
        debug_assert_eq!(phi.len(), n_x * n_y * n_z);
        (n_x * n_y, n_z)
    }

    /// `[n_x, n_y * n_z]` — separate one transverse axis.
    pub fn x_yz(phi: &[f64], n_x: usize, n_y: usize, n_z: usize) -> (usize, usize) {
        debug_assert_eq!(phi.len(), n_x * n_y * n_z);
        (n_x, n_y * n_z)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Synthetic separable adjoint: ψ*(r,g) = a(r) · b(g). This is
    /// rank-1 by construction so SVD at rank 1 should reproduce it
    /// to machine epsilon.
    #[test]
    fn rank_one_separable_compresses_exactly_at_rank_one() {
        let n_rows = 50;
        let n_cols = 4;
        let mut a = vec![0.0_f64; n_rows];
        for i in 0..n_rows {
            // Exponential decay along rows — typical of slab adjoint.
            a[i] = (-(i as f64) * 0.1).exp();
        }
        let b = [1.0_f64, 0.7, 0.3, 0.1];

        let mut mat = vec![0.0_f64; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                mat[i * n_cols + j] = a[i] * b[j];
            }
        }

        let svd = AdjointSvd::compress(&mat, n_rows, n_cols, 1);
        let recon = svd.reconstruct();
        let err = recon_error(&mat, &recon);
        assert!(
            err.max_rel < 1e-12,
            "rank-1 separable: max_rel={}",
            err.max_rel
        );
        assert!(err.frob_rel < 1e-12);
        assert_eq!(svd.rank, 1);
    }

    /// Two-component matrix should be exact at rank 2.
    #[test]
    fn rank_two_separable_exact_at_rank_two() {
        let n_rows = 20;
        let n_cols = 6;
        let a1: Vec<f64> = (0..n_rows).map(|i| (-(i as f64) * 0.2).exp()).collect();
        let a2: Vec<f64> = (0..n_rows).map(|i| ((i as f64) * 0.05).sin()).collect();
        let b1 = [1.0, 0.5, 0.3, 0.1, 0.05, 0.01];
        let b2 = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5];

        let mut mat = vec![0.0_f64; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                mat[i * n_cols + j] = a1[i] * b1[j] + a2[i] * b2[j];
            }
        }

        let svd1 = AdjointSvd::compress(&mat, n_rows, n_cols, 1);
        let recon1 = svd1.reconstruct();
        let err1 = recon_error(&mat, &recon1);
        assert!(
            err1.frob_rel > 1e-3,
            "rank 1 should not be exact for rank-2 matrix"
        );

        let svd2 = AdjointSvd::compress(&mat, n_rows, n_cols, 2);
        let recon2 = svd2.reconstruct();
        let err2 = recon_error(&mat, &recon2);
        assert!(
            err2.max_rel < 1e-10,
            "rank-2 should be exact: max_rel={}",
            err2.max_rel
        );
    }

    #[test]
    fn compression_bytes_match_factor_storage() {
        let svd = AdjointSvd::compress(&vec![1.0; 100], 10, 10, 3);
        let (dense, factored) = compression_bytes(10, 10, svd.rank);
        assert_eq!(svd.bytes_dense(), dense);
        // bytes_compressed includes the s vector exactly once
        // matching the factored formula.
        assert_eq!(svd.bytes_compressed(), factored);
    }

    /// Singular spectrum should be monotonically decreasing.
    #[test]
    fn singular_values_descend() {
        let n_rows = 30;
        let n_cols = 5;
        let mut mat = vec![0.0_f64; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                mat[i * n_cols + j] = ((i + 1) as f64).ln() * (j + 1) as f64;
            }
        }
        let svd = AdjointSvd::compress(&mat, n_rows, n_cols, n_cols);
        for w in svd.s.windows(2) {
            assert!(w[0] >= w[1] - 1e-12, "non-monotone: {:?}", svd.s);
        }
    }

    /// Reshape doesn't change SVD properties — we just relabel which
    /// dimension is "rows".
    /// log10 SVD round-trips exactly for a rank-1 separable matrix
    /// in linear space (it's also rank-1 in log space because
    /// log(a · b) = log(a) + log(b) — but the basis is different).
    #[test]
    fn log10_compress_round_trips_at_full_rank() {
        let n_rows = 30;
        let n_cols = 4;
        let mut mat = vec![0.0_f64; n_rows * n_cols];
        // Geometric progression — spans 4 decades.
        for i in 0..n_rows {
            for j in 0..n_cols {
                mat[i * n_cols + j] = 10.0_f64.powf(i as f64 * 0.1) * (j + 1) as f64;
            }
        }
        let svd = AdjointSvd::compress_with_space(&mat, n_rows, n_cols, n_cols, SpaceMode::Log10);
        let recon = svd.reconstruct();
        let err = recon_error(&mat, &recon);
        // Full-rank log10 SVD reconstructs to within numerical
        // precision of 10^x · 10^-x ≈ machine eps amplified by the
        // dynamic range — set a generous bound.
        assert!(
            err.frob_rel < 1e-10,
            "log10 round-trip: frob_rel={}",
            err.frob_rel
        );
    }

    /// Picker should return Dense when the matrix is too small for
    /// SVD to break even.
    #[test]
    fn pick_returns_dense_for_tiny_matrix() {
        let mat = vec![1.0_f64, 2.0, 3.0, 4.0];
        let repr = pick_representation(&mat, 2, 2, 2, 1e-3, PickerSpace::Linear);
        match repr {
            AdjointRepr::Dense { .. } => {}
            AdjointRepr::Svd(_) => {
                panic!("2x2 matrix should not compress with SVD")
            }
        }
    }

    /// Picker should return SVD for a low-rank matrix that meets the
    /// tolerance and is large enough for SVD to win on bytes.
    #[test]
    fn pick_returns_svd_for_low_rank_large_matrix() {
        let n_rows = 200;
        let n_cols = 50;
        // Rank-1 separable matrix.
        let mut mat = vec![0.0_f64; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                mat[i * n_cols + j] = ((i + 1) as f64).sqrt() * (j + 1) as f64;
            }
        }
        let repr = pick_representation(&mat, n_rows, n_cols, 5, 1e-6, PickerSpace::Linear);
        match repr {
            AdjointRepr::Svd(s) => {
                assert_eq!(s.rank, 1, "should pick rank 1 for rank-1 matrix");
                assert!(s.bytes_compressed() < (n_rows * n_cols * 8));
            }
            AdjointRepr::Dense { .. } => panic!("should compress this rank-1 matrix"),
        }
    }

    /// Picker falls back to Dense if no rank meets the tolerance
    /// while also beating dense on bytes.
    #[test]
    fn pick_falls_back_to_dense_when_tolerance_unreachable() {
        // Random-ish 100x10 matrix; full rank up to min(100,10)=10.
        // Force tolerance impossibly tight (1e-20). Even rank=10
        // (full rank) will not satisfy this in floating point — and
        // even if it did, byte budget breaks even at rank 10 because
        // (100+10+1)·10·8 = 8880 > 100·10·8 = 8000. So we expect Dense.
        let mut mat = vec![0.0_f64; 100 * 10];
        for i in 0..100 {
            for j in 0..10 {
                let mut s = 0_u64;
                let prod = (i * 37 + j * 7 + 1) as u64;
                s = s.wrapping_mul(6364136223846793005).wrapping_add(prod);
                mat[i * 10 + j] = ((s >> 32) as f64 / u32::MAX as f64) - 0.5;
            }
        }
        let repr = pick_representation(&mat, 100, 10, 10, 1e-20, PickerSpace::Linear);
        assert!(matches!(repr, AdjointRepr::Dense { .. }));
    }

    /// Log-space picker handles a matrix with values spanning many
    /// decades — linear SVD would have huge max-rel-err on small
    /// entries; log SVD does not.
    #[test]
    fn log_space_picker_handles_wide_dynamic_range() {
        let n_rows = 100;
        let n_cols = 20;
        let mut mat = vec![0.0_f64; n_rows * n_cols];
        // ψ* spans 6 decades along rows (typical for shielding).
        for i in 0..n_rows {
            for j in 0..n_cols {
                let row_decay = 10.0_f64.powf(-(i as f64) * 0.06);
                let col_factor = (j + 1) as f64;
                mat[i * n_cols + j] = row_decay * col_factor;
            }
        }
        // Tolerance 1e-6: at this stringency log-space SVD should
        // pass at rank 1 (matrix is rank-1 in log space) while
        // linear-space SVD also passes at rank 1 because the
        // synthetic data is exactly rank-1 in linear space too. The
        // important assertion is that the *log* path returns
        // a rank-1 SVD.
        let repr = pick_representation(&mat, n_rows, n_cols, 5, 1e-6, PickerSpace::Log10);
        match repr {
            AdjointRepr::Svd(s) => {
                assert_eq!(s.space, SpaceMode::Log10);
                assert!(
                    s.rank <= 2,
                    "log-space rank should stay tiny, got {}",
                    s.rank
                );
            }
            AdjointRepr::Dense { .. } => {
                panic!("log-space picker should compress wide-dynamic-range data")
            }
        }
    }

    #[test]
    fn reshape_xy_z_for_slab_is_rank_one() {
        // Synthetic 1-group slab: ψ*(x,y,z) = f(z), independent of x,y.
        let n_x = 4_usize;
        let n_y = 4_usize;
        let n_z = 25_usize;
        let mut phi = vec![0.0_f64; n_x * n_y * n_z];
        for ix in 0..n_x {
            for iy in 0..n_y {
                for iz in 0..n_z {
                    let z = iz as f64;
                    // Increasing toward detector face.
                    phi[ix * n_y * n_z + iy * n_z + iz] = (z * 0.1).exp();
                }
            }
        }
        let (m, n) = reshape::xy_z(&phi, n_x, n_y, n_z);
        assert_eq!(m, n_x * n_y);
        assert_eq!(n, n_z);
        let svd = AdjointSvd::compress(&phi, m, n, 1);
        let recon = svd.reconstruct();
        let err = recon_error(&phi, &recon);
        assert!(
            err.frob_rel < 1e-12,
            "slab is rank 1 in xy_z reshape: {:?}",
            err
        );
    }
}
