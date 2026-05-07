//! Random Ray Method (TRRM) — multigroup forward + adjoint solver.
//!
//! This module ships a CPU-only multigroup random-ray transport solver
//! whose target deliverable is a **real adjoint flux** ψ*(r,g) suitable
//! for FW-CADIS weight-window generation, feeding into the existing
//! `transport::weight_window::WeightWindow::from_flux` pipeline.
//!
//! Scope and explicit non-goals:
//!
//! - Cartesian voxel FSRs only. Cell-based FSRs with ray-stochastic
//!   volume estimation are a follow-on.
//! - Flat-source approximation (0th order). Linear source is a follow-on.
//! - Forward + adjoint k-eigenvalue and forward fixed-source. The adjoint
//!   is built by transposing the scattering matrix and swapping χ ↔ νΣ_f.
//! - "Mortal" rays — sampled fresh each batch, dead-zone Z + active-zone
//!   length D, terminated. The "immortal ray" persistent-state variant
//!   from Tramm & Siegel 2021 is the GPU-side follow-on.
//!
//! The module reuses `Geometry`, `find_cell_recursive`,
//! `trace_step_recursive`, and the existing surface-BC handling. It does
//! not touch the continuous-energy MC engine; both solvers are siblings.

pub mod cadis;
pub mod fsr;
pub mod integrator;
pub mod mgxs;
pub mod solver;

pub use cadis::weight_window_from_adjoint;
pub use fsr::FsrMesh;
pub use mgxs::{MaterialMgxs, MgxsLibrary, ScatterMatrix};
pub use solver::{AdjointFlag, RandomRaySolver, RaySolverConfig, SolverMode, SolverResult};
