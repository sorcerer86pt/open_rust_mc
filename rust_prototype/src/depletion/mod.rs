//! Depletion (burnup) — Bateman + CRAM-16 + predictor-corrector.
//!
//! Solves the Bateman equation
//!   dN/dt = A · N
//! where `A` is the transmutation matrix (decay constants on the
//! diagonal, decay branches and reaction rates off-diagonal). For a
//! constant `A` over a time step `Δt`:
//!   N(t + Δt) = exp(A · Δt) · N(t)
//!
//! `A` is non-positive-definite in the sense that its eigenvalues
//! lie in the left half-plane (decay/removal). `exp(A · Δt)` is
//! evaluated via the **Chebyshev Rational Approximation Method**
//! (CRAM-16, Pusa 2016 — order 16 partial-fraction approximation,
//! double precision, accurate to ~1e-14 for arguments in `(-∞, 0]`).
//!
//! For varying flux (and hence varying `A`) over the step, the
//! predictor-corrector wrappers in `predictor_corrector.rs` solve
//! once with the BOC flux (predictor), update the flux from the
//! predicted composition via a fresh transport solve, then resolve
//! with the average flux (corrector).
//!
//! References:
//! - Pusa, "Rational Approximations to the Matrix Exponential in
//!   Burnup Calculations", J. Nucl. Sci. Tech. 2016.
//! - Isotalo & Aarnio, "Comparison of depletion algorithms for
//!   large systems of nuclides", Ann. Nucl. Energy 2011.
//! - OpenMC methods paper §depletion.

pub mod chain;
pub mod chain_io;
pub mod cram;
pub mod flux;
pub mod mapping;
pub mod matrix;
pub mod predictor_corrector;

pub use chain::{DecayBranch, DepletionChain, NuclideEntry, ReactionXs};
pub use chain_io::{ChainLoadError, ChainSpec};
pub use cram::cram16;
pub use mapping::{BurnupMapping, BurnupMappingEntry};
pub use flux::{
    mean_fissions_per_source, mean_flux_per_source, power_normalized_source,
    voxel_flux_per_source, E_PER_FISSION_J,
};
pub use matrix::{build_transmutation_matrix, TransmutationInputs};
pub use predictor_corrector::{deplete_ce_li, DepletionStep};
