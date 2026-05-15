//! Random Ray (TRRM) — multigroup forward + adjoint solver.
//! Deliverable: ψ*(r,g) for FW-CADIS via
//! `transport::weight_window::WeightWindow::from_flux`.
//!
//! Scope: Cartesian voxel FSRs, flat source, mortal rays.
//! Adjoint = transposed scatter + χ ↔ νΣ_f swap. Reuses `Geometry`,
//! `find_cell_recursive`, `trace_step_recursive`, surface BCs.

pub mod adjoint_svd;
pub mod cadis;
pub mod fsr;
pub mod integrator;
pub mod mgxs;
pub mod solver;

pub use adjoint_svd::{
    AdjointReconError, AdjointRepr, AdjointSvd, PickerSpace, SpaceMode, compression_bytes,
    pick_representation, recon_error,
};
pub use cadis::weight_window_from_adjoint;
pub use fsr::FsrMesh;
pub use mgxs::{MaterialMgxs, MgxsLibrary, ScatterMatrix};
pub use solver::{AdjointFlag, RandomRaySolver, RaySolverConfig, SolverMode, SolverResult};
