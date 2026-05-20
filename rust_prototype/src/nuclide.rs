// SPDX-License-Identifier: MIT
//! Load nuclide cross-section data from `.npy` files (exported by Python).
//!
//! Expected files in `dir` with a given `prefix`:
//!   {prefix}A_raw_u235_mt18.npy  — N_E × N_T matrix (linear scale, barns)
//!   {prefix}A_log_u235_mt18.npy  — same in log₁₀ scale
//!   {prefix}energies_u235.npy    — energy grid (eV)
//!   {prefix}temperatures_u235.npy — temperature labels (string array)

use std::path::Path;

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use crate::error::{Result, SvdError};

/// Cross-section data for a single nuclide + reaction loaded from numpy files.
pub struct NuclideData {
    /// Cross-section matrix in linear scale (barns), N_E × N_T, row-major.
    pub a_raw: Vec<f64>,
    /// Cross-section matrix in log₁₀ scale, N_E × N_T, row-major.
    pub a_log: Vec<f64>,
    /// Energy grid in eV, length N_E.
    pub energies: Vec<f64>,
    /// Number of energy points.
    pub n_e: usize,
    /// Number of temperatures.
    pub n_t: usize,
}

impl NuclideData {
    /// Load from a directory containing numpy outputs from the Python pipeline.
    pub fn load(dir: &Path, prefix: &str) -> Result<Self> {
        let a_raw_nd = load_2d(&dir.join(format!("{prefix}A_raw_u235_mt18.npy")))?;
        let a_log_nd = load_2d(&dir.join(format!("{prefix}A_log_u235_mt18.npy")))?;
        let energies = load_1d(&dir.join(format!("{prefix}energies_u235.npy")))?;

        let (n_e, n_t) = a_raw_nd.dim();

        if a_log_nd.dim() != (n_e, n_t) {
            return Err(SvdError::DimensionMismatch {
                expected: format!("A_log shape ({n_e}, {n_t})"),
                got: format!("{:?}", a_log_nd.dim()),
            });
        }
        if energies.len() != n_e {
            return Err(SvdError::DimensionMismatch {
                expected: format!("energies length {n_e}"),
                got: format!("{}", energies.len()),
            });
        }

        // Flatten to row-major Vec<f64>
        let a_raw: Vec<f64> = a_raw_nd.into_raw_vec_and_offset().0;
        let a_log: Vec<f64> = a_log_nd.into_raw_vec_and_offset().0;

        Ok(Self {
            a_raw,
            a_log,
            energies,
            n_e,
            n_t,
        })
    }
}

fn load_2d(path: &Path) -> Result<Array2<f64>> {
    let file = std::fs::File::open(path).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: ndarray_npy::ReadNpyError::Io(e),
    })?;
    Array2::<f64>::read_npy(file).map_err(|e| SvdError::NpyLoad {
        path: path.display().to_string(),
        source: e,
    })
}

fn load_1d(path: &Path) -> Result<Vec<f64>> {
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
