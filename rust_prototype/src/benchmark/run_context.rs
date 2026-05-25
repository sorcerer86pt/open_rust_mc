// SPDX-License-Identifier: MIT
//! Stage 1 — process-wide context for the benchmark pipeline.
//!
//! Built once at startup, passed as `Arc<RunContext>` to every
//! pipeline thread. Holds the long-lived resources (GPU contexts,
//! rayon pool, nuclide library) that survive across all cases in a
//! sweep.

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;

use crate::hardware_profile::HardwareProfile;
use crate::transport::nuclides::NuclideLibrary;

use super::run_args::RunArgs;

/// Long-lived per-run state. Cheap to clone (`Arc`-shared everywhere).
pub struct RunContext {
    pub hw: Arc<HardwareProfile>,
    pub rayon_pool: Arc<rayon::ThreadPool>,

    #[cfg(feature = "cuda")]
    pub gpu_t: Option<Arc<crate::gpu_transport::GpuTransportContext>>,
    /// Lazily built per geometry shape (Phase 4 — for now, one shared
    /// instance reused across cases). `RwLock` so the watchdog can
    /// rebuild it after a `cuDeviceReset`.
    #[cfg(feature = "cuda")]
    pub gpu_r: std::sync::RwLock<Option<Arc<crate::gpu_recursive::GpuRecursiveContext>>>,
    /// Stage 4 kernel launches go through this stream. Distinct from
    /// `stream_transfer` so Stage 3's H→D copies and Stage 4's
    /// kernels can actually overlap.
    #[cfg(feature = "cuda")]
    pub stream_compute: Option<Arc<CudaStream>>,
    /// Stage 3 H→D copies go through this stream. Stage 4 must
    /// `stream_compute.wait(upload_done_event)` before launching a
    /// kernel that reads buffers populated by `stream_transfer`.
    #[cfg(feature = "cuda")]
    pub stream_transfer: Option<Arc<CudaStream>>,

    /// Process-wide three-tier nuclide cache. Shared across all cases.
    pub nuclide_store: &'static crate::transport::nuclide_cache::TieredStore,

    /// Nuclide library (file paths, ZAID lookups). Constructed once
    /// from `data_dir`; cheap to share.
    pub lib: Arc<NuclideLibrary>,

    pub args: RunArgs,
    pub data_dir: PathBuf,
    pub bench_dir: PathBuf,

    /// Capacity of the slot pool. Sized at construction per §5.3.1
    /// from total VRAM and an initial bundle-size estimate.
    pub n_slots: usize,
}
