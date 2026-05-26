// SPDX-License-Identifier: MIT
//! ICSBEP benchmark sweep — in-process, multi-stage pipeline driver.
//!
//! Replaces the per-case Python subprocess loop (`icsbep_sweep.py`)
//! with the threaded `benchmark::Pipeline`. Reads the same case JSONs,
//! produces the same per-case stdout / CSV output, and exits with
//! `0 / 1 / 2` per the historical contract (`all-pass / any-fail /
//! any-error`).
//!
//! See `docs/benchmark-pipeline-spec.md` for the architecture.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};

use open_rust_mc::benchmark::{
    pipeline::Pipeline,
    run_args::{RunArgs, RunnerSelection},
    run_context::RunContext,
};
use open_rust_mc::hardware_profile;
use open_rust_mc::transport::nuclide_cache;
use open_rust_mc::transport::nuclides::NuclideLibrary;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum RunnerArg {
    Cpu,
    Gpu,
    Auto,
}

impl From<RunnerArg> for RunnerSelection {
    fn from(value: RunnerArg) -> Self {
        match value {
            RunnerArg::Cpu => RunnerSelection::Cpu,
            RunnerArg::Gpu => RunnerSelection::Gpu,
            RunnerArg::Auto => RunnerSelection::Auto,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "benchmark_runner",
    about = "ICSBEP regression sweep — heterogeneous CPU/GPU pipeline",
    long_about = "Drives the in-process benchmark pipeline over a \
                  directory of ICSBEP case JSONs. Replaces the \
                  per-case subprocess loop in icsbep_sweep.py."
)]
struct Cli {
    /// Directory containing `*.json` case files.
    #[arg(long, default_value = "bench/icsbep")]
    bench_dir: PathBuf,

    /// Root directory of the OpenMC HDF5 nuclear-data distribution.
    /// Defaults to ENDF/B-VIII.1 under the workspace `data/` tree;
    /// override for alternate libraries.
    #[arg(long, default_value = "data/endfb-viii.1-hdf5/neutron")]
    data_dir: PathBuf,

    /// Substring filter applied to case file stem (no `.json`).
    #[arg(long)]
    filter: Option<String>,

    /// CSV output path. Appended row-by-row, flushed per case.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// JSONL telemetry path (per-case stage timings + routing).
    #[arg(long)]
    telemetry: Option<PathBuf>,

    /// Per-case timeout (watchdog kick threshold).
    #[arg(long, default_value_t = 3600)]
    case_timeout_s: u64,

    /// Stop-file path — when present, the sweep halts at the next
    /// case boundary.
    #[arg(long, default_value = "outputs/STOP")]
    stop_file: PathBuf,

    /// Resume from a previous CSV — already-completed case ids are
    /// skipped.
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Pass/fail envelope multiplier (`|Δk| ≤ max(150 pcm, n_sigma × σ)`).
    #[arg(long, default_value_t = 2.0)]
    n_sigma: f64,

    /// Backend selection at the run level.
    #[arg(long, value_enum, default_value_t = RunnerArg::Auto)]
    runner: RunnerArg,

    /// Override `particles_per_batch` for every case (CLI beats JSON).
    #[arg(long)]
    particles_per_batch: Option<u32>,

    /// Override `batches`.
    #[arg(long)]
    batches: Option<u32>,

    /// Override `inactive_batches`.
    #[arg(long)]
    inactive_batches: Option<u32>,

    /// Number of independent seeds per case (averaged for σ_calc).
    #[arg(long)]
    seeds: Option<u32>,

    /// Base RNG seed (per-case seeds derived deterministically).
    #[arg(long, default_value_t = 42)]
    base_seed: u64,

    /// Global SVD rank.
    #[arg(long, default_value_t = 15)]
    rank: usize,

    /// Slot pool capacity override (defaults to §5.3.1 auto-sizing).
    #[arg(long)]
    n_slots: Option<usize>,

    /// CpuExecutor parallelism (number of cases the CPU runs
    /// concurrently). Default 1 (whole rayon pool per case).
    #[arg(long, default_value_t = 1)]
    n_cpu_executor_threads: usize,

    /// Emit incremental scatter plot every N cases.
    #[arg(long, default_value_t = 10)]
    plot_every: usize,

    /// Use the sequential single-thread driver instead of the parallel
    /// one. Diagnostic / determinism mode.
    #[arg(long, default_value_t = false)]
    sequential: bool,
}

impl Cli {
    fn into_args(self) -> RunArgs {
        let Cli {
            bench_dir,
            data_dir,
            filter,
            csv,
            telemetry,
            case_timeout_s,
            stop_file,
            resume,
            n_sigma,
            runner,
            particles_per_batch,
            batches,
            inactive_batches,
            seeds,
            base_seed,
            rank,
            n_slots,
            n_cpu_executor_threads,
            plot_every,
            ..
        } = self;
        RunArgs {
            bench_dir,
            data_dir,
            filter,
            csv,
            telemetry,
            case_timeout_s,
            stop_file,
            resume,
            n_sigma,
            runner: runner.into(),
            particles_per_batch,
            batches,
            inactive_batches,
            seeds,
            base_seed,
            rank,
            n_slots,
            n_cpu_executor_threads,
            plot_every,
        }
    }
}

fn main() -> ExitCode {
    hardware_profile::log_startup_banner();
    let cli = Cli::parse();
    let sequential_mode = cli.sequential;
    let args = cli.into_args();

    let hw = Arc::new(hardware_profile::hardware_profile().clone());

    // Default slot pool: 4 — enough to keep both executors fed without
    // ballooning channel queues. Phase 4 plumbs in the dynamic sizing
    // from §5.3.1 (VRAM-aware on GPU builds).
    let n_slots = args.n_slots.unwrap_or(4);

    // Rayon pool sized to `cores - 6` per §5.4.1; floor at 2 so small
    // hosts still progress (the CPU executor blocks start-to-end on
    // its rayon install).
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let pool_threads = logical.saturating_sub(6).max(2);
    let rayon_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .thread_name(|i| format!("bench-rayon-{i}"))
            .build()
            .expect("rayon pool"),
    );

    let lib = Arc::new(NuclideLibrary::from_data_dir(&args.data_dir));
    let nuclide_store = nuclide_cache::shared();

    #[cfg(feature = "cuda")]
    let (gpu_t, gpu_r, stream_compute, stream_transfer) = {
        match open_rust_mc::gpu_transport::GpuTransportContext::shared() {
            Ok(ctx) => {
                let sc = ctx.new_compute_stream().ok();
                let st = ctx.new_transfer_stream().ok();
                (Some(ctx), std::sync::RwLock::new(None), sc, st)
            }
            Err(e) => {
                eprintln!("[gpu] context init failed: {e} — falling back to CPU");
                (None, std::sync::RwLock::new(None), None, None)
            }
        }
    };

    let bench_dir = args.bench_dir.clone();
    let data_dir = args.data_dir.clone();

    let ctx = Arc::new(RunContext {
        hw,
        rayon_pool,
        #[cfg(feature = "cuda")]
        gpu_t,
        #[cfg(feature = "cuda")]
        gpu_r,
        #[cfg(feature = "cuda")]
        stream_compute,
        #[cfg(feature = "cuda")]
        stream_transfer,
        nuclide_store,
        lib,
        args,
        data_dir,
        bench_dir,
        n_slots,
    });

    let state = if sequential_mode {
        Pipeline::run_sequential(ctx)
    } else {
        Pipeline::run_parallel(ctx)
    };

    match state {
        Ok(state) => {
            println!(
                "\nSummary: {} PASS, {} FAIL, {} ERROR ({} total)",
                state.passes,
                state.fails,
                state.errors,
                state.total(),
            );
            match state.exit_code() {
                0 => ExitCode::SUCCESS,
                1 => ExitCode::from(1),
                _ => ExitCode::from(2),
            }
        }
        Err(e) => {
            eprintln!("pipeline failed: {e}");
            ExitCode::from(3)
        }
    }
}
