// SPDX-License-Identifier: MIT
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SvdError {
    #[error("failed to load numpy file: {path}")]
    NpyLoad {
        path: String,
        #[source]
        source: ndarray_npy::ReadNpyError,
    },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("HDF5 error reading {path}: {detail}")]
    Hdf5 { path: String, detail: String },

    #[error("I/O error")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SvdError>;
