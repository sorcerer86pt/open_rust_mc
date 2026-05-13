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
pub mod depletion;
pub mod error;
/// Per-material nuclide cap. The CPU transport hot path uses this to
/// size fixed-length micro-XS arrays (`simulate.rs::MAX_NUCLIDES`),
/// the GPU kernel receives the same number through an NVRTC
/// `-DMAX_NUC_PER_MAT=N` compile flag (`gpu_recursive.rs` and
/// `gpu_transport.rs`), and `Material::add_nuclide` callers must keep
/// each material below this threshold or transport will panic at the
/// first collision. Bumping requires a full rebuild — the constant
/// flows through `MicroXs`, `GpuMaterialData`, the device-side
/// `nuc_t[MAX_NUC_PER_MAT]` register array, and every downstream
/// fixed-size loop.
///
/// Bumped 32 → 128 to cover HMF-069 ("HEU part 2732", 69 nuclides) and
/// Pu solution benchmarks ("Plutonium nitrate solution", 67 nuclides)
/// with headroom for spent-fuel cask compositions (often 50-80 per
/// material). GPU register pressure: `nuc_t[128]` is 1024 bytes/thread
/// = ~128 32-bit registers, fitting inside sm_86's 255-register
/// per-thread budget. If kernel occupancy regresses, dial back to ~80.
pub const MAX_NUCLIDES_PER_MATERIAL: usize = 128;

pub mod geometry;
#[cfg(feature = "cuda")]
pub mod gpu;
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
