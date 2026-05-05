//! open_rust_mc — Monte Carlo neutron transport engine.
//!
//! A pure-Rust Monte Carlo Engine with:
//!   - SVD-compressed cross-sections (cache-resident reconstruction)
//!   - BVH-accelerated CSG geometry (enum dispatch, no vtables)
//!   - SoA particle layout for SIMD vectorization
//!   - Event-based transport with rayon parallelism

pub mod compare;
pub mod cp_decompose;
pub mod decompose;
pub mod error;
pub mod geometry;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_transport;
#[cfg(feature = "cuda")]
pub mod gpu_recursive;
pub mod hdf5_reader;
pub mod kernel;
pub mod loader;
pub mod nuclide;
pub mod photon;
pub mod physics;
pub mod table;
pub mod thermal;
pub mod transport;
pub mod wmp;
