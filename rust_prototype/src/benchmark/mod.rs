// SPDX-License-Identifier: MIT
//! Benchmark pipeline for ICSBEP regression sweeps.
//!
//! Six-stage in-process pipeline that replaces the per-case
//! subprocess loop in `bin/icsbep_bench.rs`. Spawns concurrent CPU
//! and GPU executors that pull from a shared `SlotArray`, with
//! Stage 3 (data load + GPU upload) running ahead of Stage 4
//! (transport) and gated on actual VRAM availability via
//! `cuMemGetInfo`.
//!
//! See `docs/benchmark-pipeline-spec.md` for the design.
//!
//! Phase 1 status: scaffold only. Every type below is `dead_code`
//! until Phase 2 wires it into a `Pipeline::run`.

#![allow(dead_code)]

pub mod case_bundle;
pub mod case_loader;
pub mod data_loader;
pub mod executor;
pub mod finalizer;
pub mod pipeline;
pub mod result_processor;
pub mod run_args;
pub mod run_context;
pub mod stats;
