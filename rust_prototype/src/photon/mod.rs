//! Photon transport data and physics.
//!
//! Scope (Phase 1 — data): HDF5 reader for OpenMC photon-per-element files
//! (e.g., `photon/C.h5`) and the data structures a transport loop would
//! consume. Physics kernels (Compton, photoelectric, pair production,
//! coherent) and the photon transport loop itself are deferred to
//! subsequent phases.
//!
//! References:
//!   - OpenMC docs §3.5 "Photon Interaction Data"
//!   - ENDF/B-VII.1 photoatomic evaluations
//!   - Hubbell & Seltzer, NISTIR 5632 (attenuation coefficients)

pub mod data;
pub mod hdf5_reader;

pub use data::{
    AnomalousFactors, Bremsstrahlung, ComptonProfiles, PhotonElement, ScatteringFactor, Subshell,
    TabulatedFactor,
};
