// SPDX-License-Identifier: MIT
//! Orchestrator — spawns the six stages, wires the channels, joins.
//!
//! Public entry point is `Pipeline::run(ctx: Arc<RunContext>) ->
//! ExitCode`. Internally:
//!   * Stage 2 (CaseLoader): 1 thread, pushes `CaseSpec`s onto
//!     the bounded `LoadQueue`.
//!   * Stage 3 (DataLoader): 1-3 threads, consumes `CaseSpec`,
//!     resolves materials, uploads to GPU on `stream_transfer`,
//!     writes a `CaseBundle` into a `Ready` slot.
//!   * Stage 4 GpuExecutor: 1 thread, runs `CudaRunner::run` on
//!     `stream_compute`, sending `ExecutionResult` to Stage 5.
//!   * Stage 4 CpuExecutor: 1 coordinator thread; actual work via
//!     the shared rayon pool. Runs one case at a time by default.
//!   * Stage 5 (ResultProcessor): 1 thread, CSV + stdout + telemetry.
//!   * Stage 6 (Finalizer): runs on the main thread after the others
//!     join.
//!
//! Phase 1: scaffold only — `Pipeline::run` is unimplemented.
