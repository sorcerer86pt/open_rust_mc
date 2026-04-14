//! Load SVD factors and energy grids from numpy `.npy` files produced
//! by the Python analysis pipeline.

use std::path::Path;

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use crate::error::{SvdError, Result};
use crate::kernel::SvdKernel;

/// Raw SVD factors loaded from disk.
#[derive(Clone)]
pub struct SvdFactors {
    /// Left singular vectors: N_E × rank (energy basis).
    pub u: Array2<f64>,
    /// Singular values: length = rank.
    pub s: Vec<f64>,
    /// Right singular vectors transposed: rank × N_T (temperature coefficients).
    pub vt: Array2<f64>,
    /// Energy grid in eV: length = N_E.
    pub energies: Vec<f64>,
}

fn load_npy_2d(path: &Path) -> Result<Array2<f64>> {
    let file = std::fs::File::open(path).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: ndarray_npy::ReadNpyError::Io(e),
    })?;
    Array2::<f64>::read_npy(file).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: e,
    })
}

fn load_npy_1d(path: &Path) -> Result<Vec<f64>> {
    let file = std::fs::File::open(path).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: ndarray_npy::ReadNpyError::Io(e),
    })?;
    let arr = ndarray::Array1::<f64>::read_npy(file).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(arr.to_vec())
}

impl SvdFactors {
    /// Load SVD factors from a directory containing the numpy outputs.
    ///
    /// Expected files (with optional `prefix`):
    ///   - `{prefix}svd_U_u235.npy`
    ///   - `{prefix}svd_S_u235.npy`
    ///   - `{prefix}svd_Vt_u235.npy`
    ///   - `{prefix}energies_u235.npy`
    pub fn load(dir: &Path, prefix: &str) -> Result<Self> {
        let u = load_npy_2d(&dir.join(format!("{prefix}svd_U_u235.npy")))?;
        let s = load_npy_1d(&dir.join(format!("{prefix}svd_S_u235.npy")))?;
        let vt = load_npy_2d(&dir.join(format!("{prefix}svd_Vt_u235.npy")))?;
        let energies = load_npy_1d(&dir.join(format!("{prefix}energies_u235.npy")))?;

        let (n_e, rank_u) = u.dim();
        let rank_s = s.len();
        let (rank_vt, _n_t) = vt.dim();

        if rank_u != rank_s || rank_s != rank_vt {
            return Err(SvdError::DimensionMismatch {
                expected: format!("consistent rank (U cols={rank_u}, S len={rank_s}, Vt rows={rank_vt})"),
                got: "mismatched ranks".into(),
            });
        }
        if energies.len() != n_e {
            return Err(SvdError::DimensionMismatch {
                expected: format!("energies length = N_E = {n_e}"),
                got: format!("{}", energies.len()),
            });
        }

        Ok(Self { u, s, vt, energies })
    }

    /// Build a reconstruction kernel truncated to rank `k`.
    ///
    /// Pre-multiplies U × Σ into the basis matrix so reconstruction
    /// is a single matrix-vector product.
    pub fn into_kernel(self, k: usize) -> SvdKernel {
        let rank = k.min(self.s.len());
        let n_e = self.energies.len();
        let (_rank_vt, n_t) = self.vt.dim();

        // Pre-multiply: basis[i, j] = U[i, j] * S[j]
        // This moves the Σ multiply out of the hot path.
        let mut basis = vec![0.0_f64; n_e * rank];
        for j in 0..rank {
            let s_j = self.s[j];
            for i in 0..n_e {
                basis[i * rank + j] = self.u[[i, j]] * s_j;
            }
        }

        // V^T coefficients: rank × N_T, row-major
        let mut vt_coeffs = vec![0.0_f64; rank * n_t];
        for j in 0..rank {
            for t in 0..n_t {
                vt_coeffs[j * n_t + t] = self.vt[[j, t]];
            }
        }

        SvdKernel::new(basis, vt_coeffs, self.energies, rank, n_e, n_t)
    }
}
