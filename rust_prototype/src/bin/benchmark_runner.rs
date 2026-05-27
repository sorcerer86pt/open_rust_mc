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

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};

use open_rust_mc::benchmark::{
    pipeline::Pipeline,
    run_args::{RunArgs, RunnerSelection},
    run_context::RunContext,
};
use open_rust_mc::data_paths;
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

    /// OpenMC HDF5 nuclear-data directory. Accepts any of:
    ///   * a neutron directory (`.../endfb-viii.1-hdf5/neutron`)
    ///   * a library root (`.../endfb-viii.1-hdf5`) — the binary
    ///     auto-appends `neutron/`
    ///   * a workspace root containing `data/<lib>/neutron` (probed
    ///     in priority order VIII.1 → VIII.0 → VII.1)
    /// When omitted, walks up from the current working directory
    /// looking for `data/<lib>/neutron`.
    #[arg(long)]
    data_dir: Option<PathBuf>,

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

    /// Slot pool capacity exact override (bypasses VRAM-aware sizing).
    /// Use `--max-slots` to cap the auto value instead.
    #[arg(long)]
    n_slots: Option<usize>,

    /// Upper bound on VRAM-aware n_slots. Auto-computed value is clamped
    /// to [1, max_slots]. Ignored when `--n-slots` is set. Default 4.
    #[arg(long, default_value_t = 4)]
    max_slots: usize,

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

    /// Enable implicit-capture + Russian-roulette survival biasing
    /// (OpenMC defaults `w_min=0.25, w_survive=1.0`). k_eff unbiased,
    /// σ tightens, long-tail histories terminate in O(log w) RR
    /// rolls instead of running the event loop to
    /// `max_events_per_history = 1_000_000`. Required for ≥200k-
    /// particle GPU runs on small-VRAM cards (3080 / A1000). Mirrors
    /// `SimConfig::survival_biasing` on the CPU path.
    #[arg(long, default_value_t = false)]
    survival_bias: bool,
}

/// Heuristic: a neutron directory contains the canonical
/// `<Symbol><Mass>.h5` files (`H1.h5`, `U235.h5`). Probe with `H1.h5`
/// since every ENDF/B HDF5 distribution ships it.
fn looks_like_neutron_dir(path: &Path) -> bool {
    path.join("H1.h5").is_file() || path.join("U235.h5").is_file()
}

/// Resolve `--data-dir` into the absolute path of a neutron directory.
///
/// Accepts:
///   * `None` — discover via [`data_paths::discover_neutron_dir`]
///     from the current working directory.
///   * `Some(p)` where `p` is itself a neutron directory.
///   * `Some(p)` where `p/neutron` is a neutron directory (library
///     root form).
///   * `Some(p)` where `discover_neutron_dir(p)` finds one (workspace
///     root form).
fn resolve_data_dir(given: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = given {
        if looks_like_neutron_dir(&path) {
            return Ok(path);
        }
        let with_neutron = path.join("neutron");
        if looks_like_neutron_dir(&with_neutron) {
            return Ok(with_neutron);
        }
        if let Some(found) = data_paths::discover_neutron_dir(&path) {
            return Ok(found);
        }
        if path.is_dir() {
            return Err(format!(
                "--data-dir {} exists but contains no neutron HDF5 files \
                 (expected `H1.h5` either directly inside it or under a \
                 `neutron/` subdirectory)",
                path.display()
            ));
        }
        return Err(format!("--data-dir {} not found", path.display()));
    }
    let cwd =
        std::env::current_dir().map_err(|e| format!("failed to read current directory: {e}"))?;
    data_paths::discover_neutron_dir(&cwd).ok_or_else(|| {
        format!(
            "no ENDF/B HDF5 library discovered (walked up from {} looking for \
             data/endfb-viii.1-hdf5/neutron, data/endfb-viii.0-hdf5/neutron, \
             data/endfb-vii.1-hdf5/neutron). Pass --data-dir explicitly.",
            cwd.display()
        )
    })
}

impl Cli {
    fn into_args(self, data_dir: PathBuf) -> RunArgs {
        let Cli {
            bench_dir,
            data_dir: _,
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
            max_slots,
            n_cpu_executor_threads,
            plot_every,
            survival_bias,
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
            max_slots,
            n_cpu_executor_threads,
            plot_every,
            survival_biasing: survival_bias,
        }
    }
}

fn main() -> ExitCode {
    hardware_profile::log_startup_banner();
    let cli = Cli::parse();
    let sequential_mode = cli.sequential;
    let data_dir = match resolve_data_dir(cli.data_dir.clone()) {
        Ok(p) => {
            eprintln!("[data-dir] using {}", p.display());
            p
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let args = cli.into_args(data_dir);

    let hw = Arc::new(hardware_profile::hardware_profile().clone());

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

    // VRAM-aware slot pool sizing (§5.3.1). On GPU builds, compute how
    // many pre-uploaded bundles can coexist in VRAM without OOM; each
    // bundle's flat-pack DtoD copy is ~1.5 GB for 20-nuclide heavy-metal
    // cases. Peak concurrent bundles = n_slots + 3 (channel + running +
    // uploading + cache source). Falls back to 4 on CPU-only. Explicit
    // `--n-slots` overrides the auto value.
    #[cfg(feature = "cuda")]
    let n_slots = args.n_slots.unwrap_or_else(|| {
        if let Some(gpu) = gpu_t.as_ref() {
            let vram_n = gpu.vram_aware_pipeline_slots();
            let n = vram_n.min(args.max_slots).max(1);
            eprintln!(
                "[pipeline] n_slots={n} (VRAM-aware={vram_n}, max_slots={}; \
                 use --n-slots for exact override)",
                args.max_slots
            );
            n
        } else {
            args.max_slots.max(1)
        }
    });
    #[cfg(not(feature = "cuda"))]
    let n_slots = args.n_slots.unwrap_or(args.max_slots.max(1));

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
