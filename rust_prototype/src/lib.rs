// SPDX-License-Identifier: MIT
//! open_rust_mc — pure-Rust MC neutron transport. SVD-compressed XS,
//! BVH+enum-dispatch CSG, SoA particles, rayon event-based transport.

pub mod compare;
pub mod cp_decompose;
pub mod decompose;
pub mod depletion;
pub mod error;
/// Per-material nuclide cap. Single source of truth — flows through
/// `MicroXs`, `simulate::MAX_NUCLIDES`, `GpuMaterialData`, and the
/// GPU NVRTC `-DMAX_NUC_PER_MAT` define. Bumping requires full
/// rebuild and re-checking sm_86 register pressure
/// (`nuc_t[128]` is ~128 32-bit regs out of 255 / thread).
///
/// Bumped 32 → 128 for HMF-069 (69 nuclides), Pu-soln (67), spent-
/// fuel casks (50-80).
pub const MAX_NUCLIDES_PER_MATERIAL: usize = 128;

pub mod geometry;
pub mod hardware_profile;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_per_nuclide;
#[cfg(feature = "cuda")]
pub mod gpu_random_ray;
#[cfg(feature = "cuda")]
pub mod gpu_recursive;
#[cfg(feature = "cuda")]
pub mod gpu_transport;
pub mod hdf5_reader;
pub mod kernel;
pub mod loader;
pub mod nuclide;
pub mod photon;
pub mod physics;
pub mod physics_constants;
pub mod quadrature;
pub mod random_ray;
pub mod table;
pub mod thermal;
pub mod transport;
pub mod wmp;
