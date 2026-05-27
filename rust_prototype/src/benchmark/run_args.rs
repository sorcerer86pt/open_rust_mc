// SPDX-License-Identifier: MIT
//! CLI arguments for the benchmark runner.
//!
//! Mirrors the existing Python sweep (`icsbep_sweep.py`) arg surface
//! so the harness migration is a drop-in. clap-derived; populated by
//! `bin/benchmark_runner.rs` (Phase 4).

use std::path::PathBuf;

/// Backend selection at the run level. Per-case routing (`RunnerHint`)
/// can still override per case when this is `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerSelection {
    Cpu,
    Gpu,
    /// Heterogeneous — both executors active; per-case routing via
    /// `RunnerHint::Auto` heuristic (§5.4.0).
    Auto,
}

/// All CLI args. Most defaults match the historical Python sweep so
/// command lines transfer over without surprises.
#[derive(Debug, Clone)]
pub struct RunArgs {
    /// Directory of benchmark case JSONs.
    pub bench_dir: PathBuf,
    /// Nuclear-data root (ENDF/B HDF5 distribution).
    pub data_dir: PathBuf,
    /// Substring filter applied to `case_id`. None = all cases.
    pub filter: Option<String>,
    /// CSV output path; appended row-by-row, flushed per case.
    pub csv: Option<PathBuf>,
    /// JSONL telemetry path (`gpu_debug_metrics` snapshots, routing
    /// decisions, stage timings). None = no telemetry.
    pub telemetry: Option<PathBuf>,
    /// Per-case timeout (watchdog kicks at this elapsed time).
    pub case_timeout_s: u64,
    /// Stop-file path — when this file exists, finish the current
    /// case(s) and exit cleanly between cases.
    pub stop_file: PathBuf,
    /// Resume mode: skip cases already present in `csv`.
    pub resume: bool,
    /// Acceptance bound multiplier for pass/fail (matches the
    /// historical sweep — `delta_pcm.abs() <= bound × 3σ_exp`).
    pub n_sigma: f64,
    /// Backend selection at the run level.
    pub runner: RunnerSelection,
    /// Per-case settings (CLI overrides JSON `recommended_settings`).
    pub particles_per_batch: Option<u32>,
    pub batches: Option<u32>,
    pub inactive_batches: Option<u32>,
    pub seeds: Option<u32>,
    pub base_seed: u64,
    /// SVD rank (global default; per-case JSON can override).
    pub rank: usize,
    /// Slot pool size override (None = §5.3.1 VRAM-aware auto-sizing).
    pub n_slots: Option<usize>,
    /// Upper bound on VRAM-auto n_slots. Auto-computed value is clamped
    /// to [1, max_slots]. Ignored when `n_slots` is set explicitly.
    /// Default 4.
    pub max_slots: usize,
    /// CpuExecutor parallelism: how many cases the CpuExecutor runs
    /// concurrently. Default 1 (whole rayon pool to one case at a
    /// time). >1 splits the pool.
    pub n_cpu_executor_threads: usize,
    /// Emit incremental scatter plot every N completed cases.
    pub plot_every: usize,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            bench_dir: PathBuf::from("bench/icsbep"),
            data_dir: PathBuf::from("data/endfb-viii.1-hdf5/neutron"),
            filter: None,
            csv: None,
            telemetry: None,
            case_timeout_s: 3600,
            stop_file: PathBuf::from("outputs/STOP"),
            resume: false,
            n_sigma: 2.0,
            runner: RunnerSelection::Auto,
            particles_per_batch: None,
            batches: None,
            inactive_batches: None,
            seeds: None,
            base_seed: 42,
            rank: 15,
            n_slots: None,
            max_slots: 4,
            n_cpu_executor_threads: 1,
            plot_every: 10,
        }
    }
}
