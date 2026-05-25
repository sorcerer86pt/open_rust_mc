// SPDX-License-Identifier: MIT
//! Slot pool and per-case bundle.
//!
//! Stage 3 produces a `CaseBundle` (resolved nuclides + GPU upload,
//! when applicable) and writes it into one of the `SlotArray` slots.
//! Stage 4 takes it out, runs transport, and marks the slot `Done`.
//! Bounded slot count gives the pipeline natural backpressure: when
//! every slot is full, Stage 3 blocks on `empty_notify` until Stage 4
//! finishes a case.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaEvent, CudaStream};

use crate::transport::material_resolve::ResolvedMaterials;

use super::case_loader::CaseSpec;

/// Routing target after the §5.4.0 rule fires. Carried on the slot
/// so the executor threads can claim only their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
}

/// Per-case GPU bundle handle. Owns the device buffers for the
/// duration of the case; drop frees VRAM (the in-engine LFU caches
/// may keep some `Arc`s alive longer for cross-case reuse).
///
/// `upload_done` is the `CudaEvent` recorded on `stream_transfer` at
/// the end of Stage 3's H→D copies. Stage 4 must `stream_compute.wait`
/// on it before launching kernels that read these buffers.
#[cfg(feature = "cuda")]
pub struct GpuBundleHandle {
    pub nuc_data: Arc<crate::gpu_transport::GpuNuclideData>,
    pub mat_data: crate::gpu_transport::GpuMaterialData,
    pub sab_data: Arc<crate::gpu_transport::GpuSabData>,
    pub wmp_data: crate::gpu_transport::GpuWmpData,
    pub upload_done: Option<CudaEvent>,
    /// The stream the upload went through. Stored so debugging can
    /// re-record on the same stream.
    pub stream_transfer: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuBundleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBundleHandle")
            .field("upload_done", &self.upload_done.is_some())
            .finish_non_exhaustive()
    }
}

/// Resolved + (optionally) uploaded case, ready for the executor.
pub struct CaseBundle {
    pub spec: CaseSpec,
    pub resolved: ResolvedMaterials,
    #[cfg(feature = "cuda")]
    pub gpu_data: Option<GpuBundleHandle>,
    pub load_start: Instant,
    pub load_end: Instant,
}

/// State machine for one slot in the `SlotArray`. Transitions:
/// `Empty → Loading → Ready → Running → Done → Empty`.
pub enum Slot {
    Empty,
    Loading { case_id: String },
    Ready(Box<CaseBundle>),
    Running { case_id: String },
    /// Result emitted; slot is awaiting reuse. ResultProcessor flips
    /// it back to `Empty` after consuming the `ExecutionResult`.
    Done,
}

/// Bounded pool of slots shared between Stage 3 (writer) and the two
/// Stage 4 executors (readers). Capacity defined by §5.3.1.
pub struct SlotArray {
    pub slots: Vec<Mutex<Slot>>,
    /// Signalled when any slot transitions to `Ready`. Both executors
    /// wait on this; the one whose backend matches the slot's runner
    /// claims it.
    pub ready_notify: Condvar,
    /// Signalled when any slot transitions to `Done` / `Empty`.
    /// Stage 3 waits on this when all slots are full.
    pub empty_notify: Condvar,
}

impl SlotArray {
    pub fn with_capacity(n: usize) -> Self {
        let slots = (0..n).map(|_| Mutex::new(Slot::Empty)).collect();
        Self {
            slots,
            ready_notify: Condvar::new(),
            empty_notify: Condvar::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}
