// SPDX-License-Identifier: MIT
//! Stage 4 — heterogeneous executors (CPU + GPU concurrent).
//!
//! Two executor threads run side-by-side, both pulling `Ready` slots
//! from the shared `SlotArray`. The slot's routing decision (made by
//! Stage 2 + revised by Stage 3 if VRAM is tight) determines which
//! executor claims it. See `docs/benchmark-pipeline-spec.md` §5.4.

use std::path::PathBuf;
use std::time::Instant;

use super::case_loader::{CaseSpec, ReferenceSource};

/// Identity + reference values lifted out of `CaseSpec`. Kept in
/// `ExecutionResult` so Stage 5 can compute pass/fail / log the case
/// without holding a back-reference to the (non-Clone, large)
/// `CaseSpec`. Lifted at run time via `ExecutionResult::summary_from`.
#[derive(Debug, Clone)]
pub struct CaseSummary {
    pub case_id: String,
    pub seq: usize,
    pub total: usize,
    pub k_ref: f64,
    pub sigma_exp: f64,
    pub source: ReferenceSource,
    pub source_path: PathBuf,
}

impl CaseSummary {
    /// Project a `CaseSpec` into its identity fields. `CaseSpec`
    /// itself owns `Arc<Geometry>` + `SimConfig` which aren't trivially
    /// cloneable — once Stage 3 binds them into a `CaseBundle` we
    /// only need their identity downstream.
    pub fn from_spec(spec: &CaseSpec) -> Self {
        Self {
            case_id: spec.case_id.clone(),
            seq: spec.seq,
            total: spec.total,
            k_ref: spec.k_ref,
            sigma_exp: spec.sigma_exp,
            source: spec.source.clone(),
            source_path: spec.source_path.clone(),
        }
    }
}

/// Per-case execution outcome. One per case, emitted on the
/// `ResultChannel` to Stage 5.
pub struct ExecutionResult {
    pub k_calc: f64,
    pub k_sigma: f64,
    /// Track-length k_eff (lower variance than collision k_eff;
    /// reported alongside for diagnostics).
    pub k_track: f64,
    pub k_track_sigma: f64,
    pub runtime_s: f64,
    pub load_s: f64,
    pub n_histories: u64,
    pub backend: BackendUsed,
    pub summary: CaseSummary,
    pub error: Option<String>,
}

/// Which executor ran the case. Recorded in the JSONL telemetry so
/// the §5.4.0 router can be calibrated post-hoc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendUsed {
    Cpu,
    Cuda,
    /// Stage 3 downgraded a `Gpu` hint to `Cpu` because the VRAM
    /// gate refused the upload.
    CpuVramDowngrade,
}

impl ExecutionResult {
    pub fn error(summary: CaseSummary, msg: String) -> Self {
        Self {
            k_calc: f64::NAN,
            k_sigma: f64::NAN,
            k_track: f64::NAN,
            k_track_sigma: f64::NAN,
            runtime_s: 0.0,
            load_s: 0.0,
            n_histories: 0,
            backend: BackendUsed::Cpu,
            summary,
            error: Some(msg),
        }
    }
}

/// Watchdog timeout marker — wraps `Instant::now() + case_timeout`
/// and lets the executor abort a wedged kernel cleanly.
#[derive(Debug, Clone, Copy)]
pub struct CaseDeadline {
    pub started: Instant,
    pub timeout_s: u64,
}

impl CaseDeadline {
    pub fn new(timeout_s: u64) -> Self {
        Self {
            started: Instant::now(),
            timeout_s,
        }
    }

    pub fn exceeded(&self) -> bool {
        self.started.elapsed().as_secs() >= self.timeout_s
    }
}

/// Run `work` on a worker thread with a wall-clock timeout. Returns
/// `Some(result)` when the worker completes inside the deadline,
/// `None` when the timeout fires first. On timeout the worker thread
/// keeps running in the background (rust threads can't be cancelled
/// safely); the watchdog reports ERROR(timeout) and triggers a
/// device-context rebuild so the next case starts clean.
///
/// `work` must be `Send + 'static`. Callers move ownership of any
/// required state into the closure; the parent thread blocks on a
/// channel recv with the deadline.
///
/// Used by the benchmark pipeline's Stage 4 to wrap `runner.run` and
/// the CPU `simulate::run_eigenvalue_with_geometry` calls. Timeout
/// recovery for the GPU path additionally calls
/// `GpuTransportContext::force_rebuild` to issue `cuDeviceReset` and
/// rebuild a fresh context.
pub fn run_with_timeout<T, F>(
    work: F,
    timeout: std::time::Duration,
) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);
    let handle = std::thread::Builder::new()
        .name("bench-watchdog-worker".to_string())
        .spawn(move || {
            let result = work();
            // Send-fail = parent already gave up; drop the result.
            let _ = tx.send(result);
        })
        .expect("spawn watchdog worker");

    match rx.recv_timeout(timeout) {
        Ok(result) => {
            // Worker completed in time; join (won't block — it's done).
            let _ = handle.join();
            Some(result)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            // Worker still running. We deliberately don't join — the
            // thread leaks into the background until process exit,
            // which is the price of not being able to cancel a CUDA
            // kernel mid-flight. The GPU-side recovery path calls
            // `force_rebuild` which issues `cuDeviceReset`; that DOES
            // cancel pending kernels (the leaked worker will see a
            // CUDA error and unwind).
            None
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // Worker panicked. Try to surface a stub; the caller will
            // see `None` and emit ERROR.
            None
        }
    }
}
