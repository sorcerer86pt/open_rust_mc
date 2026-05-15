//! Photon transport data + physics. Loaded from OpenMC photon-per-
//! element `photon/*.h5`.
//!
//! Auxiliaries: coherent `F(x,Z)` + cumulative `∫F²`, anomalous
//! `f'+if''`; incoherent `S(x,Z)` + Hartree-Fock `Jᵢ(p_z)`;
//! photoelectric subshells + EADL relaxation; bremsstrahlung
//! Seltzer-Berger DCS + Sternheimer-Berger stopping power.
//!
//! Refs: OpenMC §3.5 + `src/element.cpp` / `src/photon.cpp`;
//! PENELOPE-2018 §2 (Salvat); Hubbell, J. Phys. Chem. Ref. Data 4
//! (1975); Seltzer & Berger, At. Data Nucl. Data Tables 35 (1986);
//! Perkins, LLNL EADL/EEDL UCRL-50400 v.30.

pub mod bremsstrahlung;
pub mod coherent;
pub mod compton;
pub mod data;
pub mod electron;
pub mod gpu;
pub mod hdf5_reader;
pub mod material;
pub mod nee;
pub mod pair;
pub mod photoelectric;
pub mod transport;

pub use data::{
    AnomalousFactors, Bremsstrahlung, ComptonProfiles, PhotonElement, ScatteringFactor, Subshell,
    TabulatedFactor,
};
