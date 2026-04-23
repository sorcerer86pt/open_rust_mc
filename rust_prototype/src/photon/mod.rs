//! Photon transport data and physics.
//!
//! Scope of the data layer: HDF5 reader for OpenMC photon-per-element
//! files (e.g., `photon/C.h5`) and the data structures the physics
//! kernels will consume. Loaded fields cover everything needed by the
//! physics-correct algorithms we plan to implement:
//!
//!   - Five reaction cross sections (coherent, incoherent, photoelectric,
//!     pair production in nuclear and electron fields)
//!   - Coherent sampling auxiliaries: `F(x, Z)` form factor, cumulative
//!     `∫ F²` for direct sampling, and `f'(E) + i f''(E)` anomalous
//!     scattering factors
//!   - Incoherent sampling auxiliaries: `S(x, Z)` bound-electron
//!     rejection function and Hartree-Fock Compton profiles `Jᵢ(p_z)`
//!     for Doppler broadening
//!   - Photoelectric: per-subshell partial cross sections, binding
//!     energies, and EADL atomic relaxation transitions
//!   - Bremsstrahlung: Seltzer-Berger DCS on `(T_e, k)` grid and
//!     Sternheimer-Berger oscillator parameters for stopping power
//!
//! Physics kernels and the photon transport loop are implemented in
//! subsequent phases on top of this data layer.
//!
//! References:
//!   - OpenMC docs §3.5 "Photon Interaction Data"
//!   - OpenMC source: `src/element.cpp`, `src/photon.cpp` (algorithm
//!     reference for Compton, photoelectric, pair production)
//!   - PENELOPE-2018 manual §2 (Salvat) — gold-standard sampling algorithms
//!   - Hubbell et al., "Atomic form factors, incoherent scattering
//!     functions, and photon scattering cross sections",
//!     J. Phys. Chem. Ref. Data 4, 471 (1975)
//!   - Seltzer & Berger, "Bremsstrahlung energy spectra from electrons
//!     with kinetic energy 1 keV – 10 GeV",
//!     At. Data Nucl. Data Tables 35, 345 (1986)
//!   - Perkins et al., LLNL EADL/EEDL (UCRL-50400 vol. 30) — atomic
//!     relaxation transition data lineage

pub mod compton;
pub mod data;
pub mod hdf5_reader;
pub mod photoelectric;

pub use data::{
    AnomalousFactors, Bremsstrahlung, ComptonProfiles, PhotonElement, ScatteringFactor, Subshell,
    TabulatedFactor,
};
