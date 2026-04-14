//! open_rust_mc — Monte Carlo neutron transport engine.
//!
//! A pure-Rust reimagining of OpenMC with:
//!   - SVD-compressed cross-sections (cache-resident reconstruction)
//!   - BVH-accelerated CSG geometry (enum dispatch, no vtables)
//!   - SoA particle layout for SIMD vectorization
//!   - Event-based transport with rayon parallelism

pub mod compare;
pub mod decompose;
pub mod error;
pub mod geometry;
pub mod hdf5_reader;
pub mod kernel;
pub mod loader;
pub mod nuclide;
pub mod physics;
pub mod table;
pub mod transport;
