//! open_rust_mc — on-the-fly cross-section reconstruction kernel.
//!
//! Core idea: store the SVD factors (U, Σ, V^T) in cache-friendly layout,
//! reconstruct σ(E, T) via a k-wide FMA loop instead of binary-searching
//! a multi-GB pointwise table.
//!
//! Memory layout for the hot path:
//!   - `basis`: N_E × k column-major matrix (U × Σ pre-multiplied)
//!   - `coeffs`: k-element vector (V^T column for the current temperature)
//!   - reconstruction: σ(E_i) = Σ_{j=0}^{k-1} basis[i,j] * coeffs[j]

pub mod compare;
pub mod decompose;
pub mod error;
pub mod hdf5_reader;
pub mod kernel;
pub mod loader;
pub mod nuclide;
pub mod table;
