// SPDX-License-Identifier: MIT
//! Stage 4 — heterogeneous executors (CPU + GPU concurrent).
//!
//! Two executor threads run side-by-side, both pulling `Ready` slots
//! from the shared `SlotArray`. The slot's routing decision (made by
//! Stage 2 + revised by Stage 3 if VRAM is tight) determines which
//! executor claims it. See `docs/benchmark-pipeline-spec.md` §5.4.

use std::time::Instant;

use super::case_loader::CaseSpec;

/// Per-case execution outcome. One per case, emitted on the
/// `ResultChannel` to Stage 5.
pub struct ExecutionResult {
    pub case_id: String,
    pub seq: usize,
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
    pub spec: CaseSpec,
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
    pub fn error(case_id: String, seq: usize, spec: CaseSpec, msg: String) -> Self {
        Self {
            case_id,
            seq,
            k_calc: f64::NAN,
            k_sigma: f64::NAN,
            k_track: f64::NAN,
            k_track_sigma: f64::NAN,
            runtime_s: 0.0,
            load_s: 0.0,
            n_histories: 0,
            backend: BackendUsed::Cpu,
            spec,
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
