//! Depletion: solves `dN/dt = A·N` via CRAM-16 (Pusa 2016). `A`
//! eigenvalues lie in the left half-plane; CRAM is accurate to
//! ~1e-14 in `(-∞, 0]`.
//!
//! Predictor-corrector handles flux variation: BOC solve, transport
//! resolve at the predicted composition, then average-flux corrector.
//!
//! Refs: Pusa, J. Nucl. Sci. Tech. 2016; Isotalo & Aarnio, Ann.
//! Nucl. Energy 2011; OpenMC methods §depletion.

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
pub use flux::{
    E_PER_FISSION_J, mean_fissions_per_source, mean_flux_per_source, power_normalized_source,
    voxel_flux_per_source,
};
pub use mapping::{BurnupMapping, BurnupMappingEntry};
pub use matrix::{TransmutationInputs, build_transmutation_matrix};
pub use predictor_corrector::{DepletionStep, deplete_ce_li};
