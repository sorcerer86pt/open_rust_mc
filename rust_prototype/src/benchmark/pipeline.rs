// SPDX-License-Identifier: MIT
//! Orchestrator — spawns the six stages, wires the channels, joins.
//!
//! Two driver entry points:
//!   * `Pipeline::run_sequential` — Phase 2 single-thread reference
//!     implementation. One case at a time. Used by tests and the
//!     single-case CLI path.
//!   * `Pipeline::run_parallel` — Phase 3 multi-thread implementation.
//!     Stage 2 / Stage 3 / Stage 4 CpuExecutor / Stage 4 GpuExecutor /
//!     Stage 5 each run in their own thread, glued by bounded
//!     `crossbeam_channel` queues. Backpressure flows from result
//!     channel → executors → bundle channels → load channel → loader.
//!
//! Stage layout (`run_parallel`):
//!   * Stage 2 (CaseLoader thread): walks `bench_dir`, parses JSONs,
//!     pushes `LoadItem`s onto the bounded `load_tx`. Honours the
//!     `STOP` file between cases.
//!   * Stage 3 (DataLoader thread): consumes `LoadItem`s, calls
//!     `data_loader::resolve_case`, routes the resulting bundle to
//!     either `cpu_bundle_tx` or `gpu_bundle_tx` based on the §5.4.0
//!     router. Today the GPU upload happens inside the executor
//!     thread; Phase 4 moves it here onto `stream_transfer`.
//!   * Stage 4 CpuExecutor thread: drains `cpu_bundle_rx`, runs each
//!     case via `simulate::run_eigenvalue_with_geometry` on the
//!     shared `rayon_pool`, emits `ExecutionResult`s on `result_tx`.
//!   * Stage 4 GpuExecutor thread (when `--features cuda` + the
//!     context has a GPU): drains `gpu_bundle_rx`, runs each case
//!     via `executor::gpu_run_case` (uploads + transport in-thread
//!     today), emits `ExecutionResult`s on `result_tx`.
//!   * Stage 5 (ResultProcessor thread): drains `result_rx`, prints
//!     the case line, records into `RunState`, optionally appends to
//!     CSV. Sends the final `RunState` back to the main thread once
//!     `result_rx` closes.

use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::geometry::scene_io::LoadedScene;
use crate::transport::simulate::{self, BatchResult};

use super::case_bundle::CaseBundle;
use super::case_loader::{parse_case_json, CaseDefaults, CaseSpec, RunnerHint};
use super::data_loader::{self, DataLoadError};
use super::executor::{BackendUsed, ExecutionResult};
use super::result_processor::{
    read_completed_case_ids, CsvWriter, TelemetryWriter, Verdict,
};
use super::run_args::RunnerSelection;
use super::run_context::RunContext;
use super::stats::RunState;

/// Entry-point error class. Distinct from per-case errors (those go
/// into `ExecutionResult.error` so the sweep keeps running).
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("bench dir {0} not readable")]
    BenchDirUnreadable(std::path::PathBuf),
    #[error("no case JSON matched in {0}")]
    NoCases(std::path::PathBuf),
}

pub struct Pipeline;

impl Pipeline {
    /// Sequential driver — Phase 2 of the migration plan.
    ///
    /// Walks `ctx.bench_dir`, parses each `*.json`, resolves materials,
    /// runs the case on the appropriate executor, records the result.
    /// One case at a time, no threading. Adequate for correctness
    /// smoke-tests (`A1000 ICSBEP smoke` in §8 Phase 2 of the spec) —
    /// production sweeps lean on `run_parallel` once Phase 3 lands.
    pub fn run_sequential(ctx: Arc<RunContext>) -> Result<RunState, PipelineError> {
        let mut cases = list_case_files(&ctx.bench_dir, ctx.args.filter.as_deref())?;

        // --resume support: drop already-completed case ids from the
        // existing CSV. Mirrors `run_parallel`.
        if ctx.args.resume {
            if let Some(csv_path) = ctx.args.csv.as_ref() {
                if let Ok(done) = read_completed_case_ids(csv_path) {
                    cases.retain(|p| {
                        let id = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        !done.contains(id)
                    });
                }
            }
        }

        let total = cases.len();
        if total == 0 {
            // Resume case where every case is already in the CSV is a
            // success, not an error — the sweep has nothing left to
            // do. Only flag NoCases when the directory itself was
            // truly empty / non-matching from the start.
            if ctx.args.resume && ctx.args.csv.is_some() {
                return Ok(RunState::default());
            }
            return Err(PipelineError::NoCases(ctx.bench_dir.clone()));
        }

        let mut csv_writer = match ctx.args.csv.as_ref() {
            Some(p) => CsvWriter::open(p).ok(),
            None => None,
        };
        let mut telemetry = match ctx.args.telemetry.as_ref() {
            Some(p) => TelemetryWriter::open(p).ok(),
            None => None,
        };

        let defaults = CaseDefaults {
            particles_per_batch: ctx.args.particles_per_batch,
            batches: ctx.args.batches,
            inactive_batches: ctx.args.inactive_batches,
            base_seed: ctx.args.base_seed,
            survival_biasing: ctx.args.survival_biasing,
        };

        let mut state = RunState::default();

        for (seq, path) in cases.iter().enumerate() {
            // Stop-file check between cases (§5 of spec).
            if ctx.args.stop_file.exists() {
                eprintln!(
                    "stop-file {} present; halting after {seq}/{total}",
                    ctx.args.stop_file.display()
                );
                break;
            }

            let hint = match ctx.args.runner {
                RunnerSelection::Cpu => RunnerHint::Cpu,
                RunnerSelection::Gpu => RunnerHint::Gpu,
                RunnerSelection::Auto => RunnerHint::Auto,
            };

            // Stage 2 — parse JSON.
            let (spec, loaded) = match parse_case_json(path, seq + 1, total, &defaults, hint) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("[parse] {}: {e}", path.display());
                    continue;
                }
            };

            // Stage 3 — resolve materials.
            let resolve_result = data_loader::resolve_case(
                spec,
                &loaded,
                ctx.lib.as_ref(),
                ctx.args.rank,
                &ctx.data_dir,
            );
            let bundle = match resolve_result {
                Ok(b) => b,
                Err(DataLoadError::Resolve(e)) => {
                    eprintln!("[resolve] {}: {e}", path.display());
                    continue;
                }
                Err(e) => {
                    eprintln!("[load] {}: {e}", path.display());
                    continue;
                }
            };

            // Stage 4 — run the case. Heterogeneous routing: GPU when
            // the runtime ctx has it AND the hint allows; otherwise
            // CPU. The full RunnerHint::Auto router (§5.4.0 — thermal,
            // tiny-workload, lattice rules) lands in Phase 3.
            let backend = pick_backend(&bundle.spec, &ctx);
            let load_s = bundle.load_end.duration_since(bundle.load_start).as_secs_f64();
            let exec_start = Instant::now();
            let result = match backend {
                BackendUsed::Cpu | BackendUsed::CpuVramDowngrade => {
                    run_cpu_case(&bundle, &ctx, backend, load_s, exec_start)
                }
                BackendUsed::Cuda => run_gpu_case(&bundle, &ctx, load_s, exec_start),
            };

            // Stage 5 — pass/fail classification + console line + CSV + telemetry.
            print_case_line(&result, ctx.args.n_sigma);
            if let Some(w) = csv_writer.as_mut() {
                if let Err(e) = w.append(&result, ctx.args.n_sigma) {
                    eprintln!("[csv] append failed: {e}");
                }
            }
            if let Some(w) = telemetry.as_mut() {
                let extra = build_telemetry_extra(&ctx);
                if let Err(e) = w.append(&result, ctx.args.n_sigma, extra) {
                    eprintln!("[telemetry] append failed: {e}");
                }
            }
            state.record(result, ctx.args.n_sigma);
        }

        Ok(state)
    }
}

impl Pipeline {
    /// Phase 3 parallel driver. Spawns one thread per stage, glued by
    /// `crossbeam-channel` bounded queues. Returns the accumulated
    /// `RunState` once every case has been processed (or the sweep
    /// halted at a stop-file boundary).
    pub fn run_parallel(ctx: Arc<RunContext>) -> Result<RunState, PipelineError> {
        let mut cases = list_case_files(&ctx.bench_dir, ctx.args.filter.as_deref())?;

        // --resume: drop cases already completed in the existing CSV.
        // Matches the historical Python sweep semantics — re-running
        // with the same --csv resumes mid-sweep instead of redoing.
        if ctx.args.resume {
            if let Some(csv_path) = ctx.args.csv.as_ref() {
                match read_completed_case_ids(csv_path) {
                    Ok(done) if !done.is_empty() => {
                        let before = cases.len();
                        cases.retain(|p| {
                            let id = p
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("");
                            !done.contains(id)
                        });
                        eprintln!(
                            "[resume] {} cases already in {} → {} remaining",
                            before - cases.len(),
                            csv_path.display(),
                            cases.len(),
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!(
                            "[resume] failed to read {}: {e} — running all cases",
                            csv_path.display()
                        );
                    }
                }
            }
        }

        let total = cases.len();
        if total == 0 {
            // Resume case where every case is already in the CSV is a
            // success, not an error — the sweep has nothing left to
            // do. Only flag NoCases when the directory itself was
            // truly empty / non-matching from the start.
            if ctx.args.resume && ctx.args.csv.is_some() {
                return Ok(RunState::default());
            }
            return Err(PipelineError::NoCases(ctx.bench_dir.clone()));
        }

        // Open the CSV writer once. Lazy `Option` so CPU-only sweeps
        // without `--csv` skip the IO entirely.
        let mut csv_writer = match ctx.args.csv.as_ref() {
            Some(p) => match CsvWriter::open(p) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!("[csv] open {}: {e} — proceeding without CSV", p.display());
                    None
                }
            },
            None => None,
        };

        // Optional JSONL telemetry sink.
        let mut telemetry = match ctx.args.telemetry.as_ref() {
            Some(p) => match TelemetryWriter::open(p) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!(
                        "[telemetry] open {}: {e} — proceeding without telemetry",
                        p.display()
                    );
                    None
                }
            },
            None => None,
        };

        // Channel sizing: `n_slots` from RunContext defines how many
        // cases can be in-flight. Doubling on the load queue lets
        // Stage 2 race a little ahead of Stage 3.
        let n_slots = ctx.n_slots.max(2);
        let (load_tx, load_rx) = bounded::<LoadItem>(n_slots * 2);
        let (cpu_tx, cpu_rx) = bounded::<CaseBundle>(n_slots);
        let (gpu_tx, gpu_rx) = bounded::<CaseBundle>(n_slots);
        let (result_tx, result_rx) = bounded::<ExecutionResult>(n_slots * 2);

        let defaults = CaseDefaults {
            particles_per_batch: ctx.args.particles_per_batch,
            batches: ctx.args.batches,
            inactive_batches: ctx.args.inactive_batches,
            base_seed: ctx.args.base_seed,
            survival_biasing: ctx.args.survival_biasing,
        };
        let hint = run_hint(ctx.args.runner);

        // Stage 2 — CaseLoader thread.
        let loader_handle = {
            let ctx = Arc::clone(&ctx);
            let cases = cases.clone();
            thread::Builder::new()
                .name("bench-stage2-loader".to_string())
                .spawn(move || stage2_loader(ctx, cases, defaults, hint, load_tx))
                .expect("spawn stage2")
        };

        // Stage 3 — DataLoader thread.
        let resolver_handle = {
            let ctx = Arc::clone(&ctx);
            thread::Builder::new()
                .name("bench-stage3-resolver".to_string())
                .spawn(move || stage3_resolver(ctx, load_rx, cpu_tx, gpu_tx))
                .expect("spawn stage3")
        };

        // Stage 4 — CpuExecutor thread.
        let cpu_handle = {
            let ctx = Arc::clone(&ctx);
            let result_tx = result_tx.clone();
            thread::Builder::new()
                .name("bench-stage4-cpu".to_string())
                .spawn(move || stage4_cpu_executor(ctx, cpu_rx, result_tx))
                .expect("spawn stage4-cpu")
        };

        // Stage 4 — GpuExecutor thread. Spawned even on CPU-only
        // builds so the gpu_rx side has a draining endpoint (otherwise
        // Stage 3 would deadlock if it ever routed to GPU). On CPU-only
        // builds the GPU channel will receive zero items because the
        // router never selects `Backend::Cuda` without a gpu_t.
        let gpu_handle = {
            let ctx = Arc::clone(&ctx);
            let result_tx = result_tx.clone();
            thread::Builder::new()
                .name("bench-stage4-gpu".to_string())
                .spawn(move || stage4_gpu_executor(ctx, gpu_rx, result_tx))
                .expect("spawn stage4-gpu")
        };

        // Drop the parent's `result_tx`; once both executors finish
        // and drop theirs the channel closes and Stage 5 exits.
        drop(result_tx);

        // Stage 5 — ResultProcessor thread. Runs on the main thread by
        // taking the result_rx end here.
        let n_sigma = ctx.args.n_sigma;
        let mut state = RunState::default();
        for r in result_rx.iter() {
            print_case_line(&r, n_sigma);
            if let Some(w) = csv_writer.as_mut() {
                if let Err(e) = w.append(&r, n_sigma) {
                    eprintln!("[csv] append failed: {e}");
                }
            }
            if let Some(w) = telemetry.as_mut() {
                let extra = build_telemetry_extra(&ctx);
                if let Err(e) = w.append(&r, n_sigma, extra) {
                    eprintln!("[telemetry] append failed: {e}");
                }
            }
            state.record(r, n_sigma);
        }

        // Join the upstream threads. Any panic in a stage surfaces
        // here as an error — propagate so the bench can be diagnosed.
        loader_handle.join().expect("stage2 panicked");
        resolver_handle.join().expect("stage3 panicked");
        cpu_handle.join().expect("stage4-cpu panicked");
        gpu_handle.join().expect("stage4-gpu panicked");

        Ok(state)
    }
}

/// Item the loader passes downstream — keeps the parsed scene alive
/// for the resolver (which needs it for material resolution).
pub(super) struct LoadItem {
    pub spec: CaseSpec,
    pub loaded: LoadedScene,
}

fn run_hint(sel: RunnerSelection) -> RunnerHint {
    match sel {
        RunnerSelection::Cpu => RunnerHint::Cpu,
        RunnerSelection::Gpu => RunnerHint::Gpu,
        RunnerSelection::Auto => RunnerHint::Auto,
    }
}

fn stage2_loader(
    ctx: Arc<RunContext>,
    cases: Vec<std::path::PathBuf>,
    defaults: CaseDefaults,
    hint: RunnerHint,
    load_tx: Sender<LoadItem>,
) {
    let total = cases.len();
    for (seq, path) in cases.into_iter().enumerate() {
        if ctx.args.stop_file.exists() {
            eprintln!(
                "stop-file {} present; halting loader at {seq}/{total}",
                ctx.args.stop_file.display()
            );
            break;
        }
        match parse_case_json(&path, seq + 1, total, &defaults, hint) {
            Ok((spec, loaded)) => {
                if load_tx.send(LoadItem { spec, loaded }).is_err() {
                    // Receiver gone — downstream collapsed.
                    break;
                }
            }
            Err(e) => {
                eprintln!("[parse] {}: {e}", path.display());
            }
        }
    }
    // Drop load_tx implicitly when the function returns; closes the
    // channel so Stage 3 knows the work is done.
}

fn stage3_resolver(
    ctx: Arc<RunContext>,
    load_rx: Receiver<LoadItem>,
    cpu_tx: Sender<CaseBundle>,
    gpu_tx: Sender<CaseBundle>,
) {
    while let Ok(LoadItem { spec, loaded }) = load_rx.recv() {
        let path = spec.source_path.clone();

        // Peek at the routing decision BEFORE choosing CPU vs GPU
        // resolution, so the GPU upload chain runs only for cases
        // actually destined for the GPU executor. `pick_backend`
        // reads only `CaseSpec` + run-level args (no heavy parsing).
        let pre_backend = pick_backend(&spec, &ctx);

        #[cfg(feature = "cuda")]
        let result = match pre_backend {
            BackendUsed::Cuda => {
                // Both ctx.gpu_t and ctx.stream_transfer must be present;
                // pick_backend already gated on gpu_t.
                match (ctx.gpu_t.as_ref(), ctx.stream_transfer.as_ref()) {
                    (Some(gpu), Some(st)) => data_loader::resolve_case_gpu(
                        spec,
                        &loaded,
                        ctx.lib.as_ref(),
                        ctx.args.rank,
                        &ctx.data_dir,
                        gpu.as_ref(),
                        st,
                    ),
                    _ => data_loader::resolve_case(
                        spec,
                        &loaded,
                        ctx.lib.as_ref(),
                        ctx.args.rank,
                        &ctx.data_dir,
                    ),
                }
            }
            _ => data_loader::resolve_case(
                spec,
                &loaded,
                ctx.lib.as_ref(),
                ctx.args.rank,
                &ctx.data_dir,
            ),
        };

        #[cfg(not(feature = "cuda"))]
        let result = data_loader::resolve_case(
            spec,
            &loaded,
            ctx.lib.as_ref(),
            ctx.args.rank,
            &ctx.data_dir,
        );

        match result {
            Ok(bundle) => {
                let route = match pre_backend {
                    BackendUsed::Cuda => &gpu_tx,
                    _ => &cpu_tx,
                };
                if route.send(bundle).is_err() {
                    break;
                }
            }
            Err(DataLoadError::Resolve(e)) => {
                eprintln!("[resolve] {}: {e}", path.display());
            }
            Err(e) => {
                eprintln!("[load] {}: {e}", path.display());
            }
        }
    }
    // Dropping the senders closes the executor channels.
}

fn stage4_cpu_executor(
    ctx: Arc<RunContext>,
    cpu_rx: Receiver<CaseBundle>,
    result_tx: Sender<ExecutionResult>,
) {
    let timeout = std::time::Duration::from_secs(ctx.args.case_timeout_s);
    while let Ok(bundle) = cpu_rx.recv() {
        let load_s = bundle.load_end.duration_since(bundle.load_start).as_secs_f64();
        let exec_start = Instant::now();
        let case_id = bundle.spec.case_id.clone();
        let summary = super::executor::CaseSummary::from_spec(&bundle.spec);
        let ctx_clone = Arc::clone(&ctx);
        let work = move || {
            run_cpu_case(&bundle, &ctx_clone, BackendUsed::Cpu, load_s, exec_start)
        };

        let result = match super::executor::run_with_timeout(work, timeout) {
            Some(r) => r,
            None => {
                // CPU rayon work isn't cancellable. The leaked worker
                // keeps consuming rayon threads until it finishes
                // (this is the price of not interrupting). The
                // process keeps moving — the executor pulls the next
                // case while the wedged one finishes in the
                // background.
                eprintln!(
                    "[watchdog] {} exceeded {}s on CPU (leaked worker continues)",
                    case_id, timeout.as_secs(),
                );
                ExecutionResult::error(
                    summary,
                    format!("timeout after {}s", timeout.as_secs()),
                )
            }
        };
        if result_tx.send(result).is_err() {
            break;
        }
    }
}

fn stage4_gpu_executor(
    ctx: Arc<RunContext>,
    gpu_rx: Receiver<CaseBundle>,
    result_tx: Sender<ExecutionResult>,
) {
    let timeout = std::time::Duration::from_secs(ctx.args.case_timeout_s);
    while let Ok(bundle) = gpu_rx.recv() {
        let load_s = bundle.load_end.duration_since(bundle.load_start).as_secs_f64();
        let exec_start = Instant::now();

        // Wrap the run in the watchdog. On timeout we report ERROR
        // and rebuild the GPU context so the next case starts clean.
        let case_id = bundle.spec.case_id.clone();
        let summary = super::executor::CaseSummary::from_spec(&bundle.spec);
        let ctx_clone = Arc::clone(&ctx);
        let work = move || run_gpu_case(&bundle, &ctx_clone, load_s, exec_start);

        let result = match super::executor::run_with_timeout(work, timeout) {
            Some(r) => r,
            None => {
                eprintln!(
                    "[watchdog] {} exceeded {}s — forcing GPU context rebuild",
                    case_id, timeout.as_secs(),
                );
                #[cfg(feature = "cuda")]
                {
                    if let Err(e) =
                        crate::gpu_transport::GpuTransportContext::force_rebuild()
                    {
                        eprintln!("[watchdog] force_rebuild failed: {e}");
                    }
                }
                ExecutionResult::error(
                    summary,
                    format!("timeout after {}s", timeout.as_secs()),
                )
            }
        };
        if result_tx.send(result).is_err() {
            break;
        }
    }
}

/// Discover `*.json` files under `bench_dir`. Optional substring
/// filter mirrors the historical Python sweep's `--filter` arg.
fn list_case_files(
    bench_dir: &Path,
    filter: Option<&str>,
) -> Result<Vec<std::path::PathBuf>, PipelineError> {
    let read_dir = std::fs::read_dir(bench_dir)
        .map_err(|_| PipelineError::BenchDirUnreadable(bench_dir.to_path_buf()))?;
    let mut out: Vec<std::path::PathBuf> = read_dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter(|p| match filter {
            Some(needle) => p
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.contains(needle))
                .unwrap_or(false),
            None => true,
        })
        .collect();
    out.sort();
    Ok(out)
}

/// Threshold below which the §5.4.0 router routes work to CPU. Below
/// `particles_per_batch × batches < 10M` the GPU's per-case setup
/// overhead (NVRTC amortised, but nuclide upload + recursive context
/// build still adds ~1 s on a modern device) usually exceeds the
/// actual transport work the case would do. Tunable per host — the
/// constant lives here so the calibration loop in Phase 4 can
/// override it without touching the routing logic.
const TINY_WORK_THRESHOLD: u64 = 10_000_000;

/// Material count above which §5.4.0 routes lattice-class cases to
/// GPU. Empirical break-even on the A1000 corpus.
const LATTICE_MATERIAL_THRESHOLD: usize = 50;

/// Pick the executor for a case. Implements the §5.4.0 routing rules
/// in order — first match wins.
///
///   Rule 1 (firm):  thermal scattering  → GPU
///   Rule 2 (firm):  tiny workload       → CPU
///   Rule 3 (soft):  lattice / many-mat  → GPU
///   Default:        GPU if available, else CPU
///
/// Each rule reads cheap, already-parsed data from the `CaseSpec`
/// (no HDF5 / device queries). Explicit `RunnerHint::Cpu` / `::Gpu`
/// override the auto rules; only `RunnerHint::Auto` runs the rule
/// table. The run-level `RunnerSelection::Cpu` / `::Gpu` short-
/// circuits at the top before any rule fires.
fn pick_backend(spec: &CaseSpec, ctx: &Arc<RunContext>) -> BackendUsed {
    let has_gpu = backend_has_gpu(ctx);

    // Run-level override beats everything else.
    match ctx.args.runner {
        RunnerSelection::Cpu => return BackendUsed::Cpu,
        RunnerSelection::Gpu => {
            return if has_gpu {
                BackendUsed::Cuda
            } else {
                BackendUsed::CpuVramDowngrade
            };
        }
        RunnerSelection::Auto => {}
    }

    // Per-case `RunnerHint::{Cpu, Gpu}` beats the rule table; only
    // `Auto` runs the rules.
    match spec.runner {
        RunnerHint::Cpu => return BackendUsed::Cpu,
        RunnerHint::Gpu => {
            return if has_gpu {
                BackendUsed::Cuda
            } else {
                BackendUsed::CpuVramDowngrade
            };
        }
        RunnerHint::Auto => {}
    }

    auto_pick(
        spec.has_thermal_scattering(),
        spec.work_proxy(),
        spec.has_lattice(),
        spec.material_count(),
        has_gpu,
    )
}

fn backend_has_gpu(ctx: &Arc<RunContext>) -> bool {
    #[cfg(feature = "cuda")]
    {
        ctx.gpu_t.is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ctx;
        false
    }
}

/// §5.4.0 rule table — pure function on the small set of features the
/// rules actually read. Kept primitive-typed (rather than taking
/// `&CaseSpec`) so unit tests can pin behaviour without building a
/// full `Geometry`.
fn auto_pick(
    has_thermal: bool,
    work: u64,
    has_lattice: bool,
    n_materials: usize,
    has_gpu: bool,
) -> BackendUsed {
    // Rule 1 (firm): thermal scattering → GPU. The CPU's branchy SAB
    // sampler is the engine's worst CPU path; GPU dominates ~10× on
    // Be / H-in-H2O cases. Falls through to default when no GPU.
    if has_gpu && has_thermal {
        return BackendUsed::Cuda;
    }

    // Rule 2 (firm): tiny workload → CPU. Below the threshold the GPU
    // setup overhead (~1 s of nuclide upload + recursive context build)
    // dwarfs the actual transport, so the CPU pool finishes faster
    // even at a fraction of the GPU's per-history throughput.
    if work < TINY_WORK_THRESHOLD {
        return BackendUsed::Cpu;
    }

    // Rule 3 (soft): lattice or many-material cases → GPU. Recursive
    // descent amortises the GPU's setup over many cells.
    if has_gpu && (has_lattice || n_materials > LATTICE_MATERIAL_THRESHOLD) {
        return BackendUsed::Cuda;
    }

    // Default: GPU when available, otherwise CPU.
    if has_gpu {
        BackendUsed::Cuda
    } else {
        BackendUsed::Cpu
    }
}

fn run_cpu_case(
    bundle: &super::case_bundle::CaseBundle,
    ctx: &Arc<RunContext>,
    backend: BackendUsed,
    load_s: f64,
    exec_start: Instant,
) -> ExecutionResult {
    let cfg = &bundle.spec.config;
    let batches: Vec<BatchResult> = ctx.rayon_pool.install(|| {
        simulate::run_eigenvalue_with_geometry(
            cfg,
            &bundle.spec.geometry,
            &bundle.resolved.materials,
            &bundle.resolved.provider,
        )
        .0
    });
    summarise(bundle, batches, load_s, exec_start, backend)
}

fn run_gpu_case(
    bundle: &super::case_bundle::CaseBundle,
    ctx: &Arc<RunContext>,
    load_s: f64,
    exec_start: Instant,
) -> ExecutionResult {
    #[cfg(feature = "cuda")]
    {
        // Fall back to CPU if any of (gpu_t, gpu_data, stream_compute)
        // are missing. The first two should always be present when
        // Stage 3 routes to the GPU channel; the third is set up by
        // the runner main.
        let (Some(gpu_t), Some(gpu_data), Some(stream_compute)) = (
            ctx.gpu_t.as_ref(),
            bundle.gpu_data.as_ref(),
            ctx.stream_compute.as_ref(),
        ) else {
            eprintln!(
                "[gpu-exec] {} downgraded to CPU (missing gpu_t / gpu_data / stream_compute)",
                bundle.spec.case_id
            );
            return run_cpu_case(
                bundle,
                ctx,
                BackendUsed::CpuVramDowngrade,
                load_s,
                exec_start,
            );
        };

        // Cross-stream sync: wait on the event Stage 3 recorded after
        // the H→D upload chain. The kernel launch then sees the
        // device buffers fully populated.
        if let Some(ref ev) = gpu_data.upload_done {
            if let Err(e) = stream_compute.wait(ev) {
                let msg = format!("stream_compute.wait(upload_done): {e}");
                return ExecutionResult::error(
                    super::executor::CaseSummary::from_spec(&bundle.spec),
                    msg,
                );
            }
        }

        match gpu_executor::run_with_bundle(bundle, ctx, gpu_t.as_ref(), gpu_data) {
            Ok(batches) => summarise(bundle, batches, load_s, exec_start, BackendUsed::Cuda),
            Err(e) => ExecutionResult::error(
                super::executor::CaseSummary::from_spec(&bundle.spec),
                format!("[gpu-exec] {}: {e}", bundle.spec.case_id),
            ),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        run_cpu_case(
            bundle,
            ctx,
            BackendUsed::CpuVramDowngrade,
            load_s,
            exec_start,
        )
    }
}

#[cfg(feature = "cuda")]
mod gpu_executor {
    use super::*;
    use crate::gpu_recursive::GpuRecursiveContext;
    use crate::gpu_transport::GpuTransportContext;
    use crate::transport::dispatch::{CudaRunner, EigenvalueRunner};
    use crate::transport::simulate::{self};
    use crate::transport::sim_limits::SimLimits;

    /// Run a GPU case using a pre-built bundle. Mirrors the inline
    /// driver in `bindings/python/src/lib.rs::run_gpu_icsbep` minus
    /// the upload chain (Stage 3 already did that on
    /// `stream_transfer`).
    pub fn run_with_bundle(
        bundle: &super::super::case_bundle::CaseBundle,
        _ctx: &Arc<RunContext>,
        gpu: &GpuTransportContext,
        gpu_data: &super::super::case_bundle::GpuBundleHandle,
    ) -> Result<Vec<crate::transport::simulate::BatchResult>, Box<dyn std::error::Error>> {
        let provider = &bundle.resolved.provider;
        let materials_rt = &bundle.resolved.materials;
        let n = bundle.spec.config.particles_per_batch as usize;
        let limits = SimLimits::default();

        let recursive = GpuRecursiveContext::build(&bundle.spec.geometry, n)?;

        const K_B_EV_PER_K: f64 = 8.617_333_262e-5;
        let mat_k_t: Vec<f64> = materials_rt
            .iter()
            .map(|m| m.temperature * K_B_EV_PER_K)
            .collect();

        // `sab_nuc_idx` was an early-singular legacy index; the
        // kernel reads `slot_per_nuc[nuc]` for the real binding so
        // the first slot's nuc_idx (or -1 when no TSL) is a
        // back-compat signal only.
        let sab_nuc_idx: i32 = provider
            .thermal
            .iter()
            .enumerate()
            .find_map(|(i, t)| t.as_ref().map(|_| i as i32))
            .unwrap_or(-1);

        let geometry = bundle.spec.geometry.clone();
        let cells: Vec<crate::geometry::Cell> = geometry.cells.clone();
        let initial_source: Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)>> =
            Box::new(move |n_part, seed| {
                simulate::try_initial_source(n_part, &geometry, &cells, seed)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|s| (s.pos.x, s.pos.y, s.pos.z, s.energy))
                    .collect()
            });

        let runner = CudaRunner {
            recursive: &recursive,
            transport: gpu,
            nuc_data: &gpu_data.nuc_data,
            mat_data: &gpu_data.mat_data,
            sab_data: &gpu_data.sab_data,
            wmp_data: &gpu_data.wmp_data,
            mat_k_t: &mat_k_t,
            sab_nuc_idx,
            max_events_per_history: limits.max_events_per_history as i32,
            fis_capacity: limits.fis_capacity(n),
            initial_source,
            buffers: std::cell::RefCell::new(None),
            refill: std::cell::RefCell::new(None),
        };
        let outcome = runner.run(&bundle.spec.config);
        Ok(outcome.batches)
    }
}

fn summarise(
    bundle: &super::case_bundle::CaseBundle,
    batches: Vec<BatchResult>,
    load_s: f64,
    exec_start: Instant,
    backend: BackendUsed,
) -> ExecutionResult {
    let runtime_s = exec_start.elapsed().as_secs_f64();

    let active_k: Vec<f64> = batches
        .iter()
        .filter(|b| b.active)
        .map(|b| b.k_eff)
        .collect();
    let active_track: Vec<f64> = batches
        .iter()
        .filter(|b| b.active && b.k_track > 0.0)
        .map(|b| b.k_track)
        .collect();
    let n_histories: u64 = batches
        .iter()
        .filter(|b| b.active)
        .map(|b| u64::from(b.batch))
        .count() as u64
        * u64::from(bundle.spec.config.particles_per_batch);
    let (k_calc, k_sigma) = mean_stderr(&active_k);
    let (k_track, k_track_sigma) = mean_stderr(&active_track);

    ExecutionResult {
        k_calc,
        k_sigma,
        k_track,
        k_track_sigma,
        runtime_s,
        load_s,
        n_histories,
        backend,
        summary: super::executor::CaseSummary::from_spec(&bundle.spec),
        error: None,
    }
}

fn mean_stderr(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = xs.len() as f64;
    let mean: f64 = xs.iter().sum::<f64>() / n;
    if xs.len() == 1 {
        return (mean, 0.0);
    }
    let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let stderr = (var / n).sqrt();
    (mean, stderr)
}

/// Snapshot of host-level state at result-time for the JSONL extras.
/// On GPU builds includes the per-nuclide + SAB cache hit counts and
/// free VRAM so post-hoc analysis can correlate sweep performance
/// with cache warmth. Cheap — same call the runner's diagnostic
/// dump uses.
fn build_telemetry_extra(ctx: &Arc<RunContext>) -> serde_json::Value {
    #[cfg(feature = "cuda")]
    {
        if let Some(gpu) = ctx.gpu_t.as_ref() {
            let (per_nuc_entries, per_nuc_bytes, per_nuc_hits) =
                gpu.per_nuclide_cache_stats();
            let (sab_entries, sab_bytes, sab_hits) = gpu.sab_buffer_cache_stats();
            return serde_json::json!({
                "per_nuclide_cache": {
                    "entries": per_nuc_entries,
                    "bytes": per_nuc_bytes,
                    "hits": per_nuc_hits,
                },
                "sab_buffer_cache": {
                    "entries": sab_entries,
                    "bytes": sab_bytes,
                    "hits": sab_hits,
                },
            });
        }
    }
    let _ = ctx;
    serde_json::Value::Null
}

fn print_case_line(r: &ExecutionResult, n_sigma: f64) {
    let verdict = Verdict::classify(r, n_sigma);
    let tag = match verdict {
        Verdict::Pass => "PASS",
        Verdict::Fail => "FAIL",
        Verdict::Error => "ERROR",
    };
    let delta_pcm = (r.k_calc - r.summary.k_ref) * 1e5;
    let sigma_combined =
        (r.k_sigma.powi(2) + r.summary.sigma_exp.powi(2)).sqrt() * 1e5;
    let bound = (n_sigma * sigma_combined).max(150.0);
    let sigma_ratio = if sigma_combined > 0.0 {
        delta_pcm.abs() / sigma_combined
    } else {
        0.0
    };
    println!(
        "{}: {} -- k={:.5}+/-{:.5}, delta={:+.0}pcm, bound=+/-{:.0}pcm, \
         {:.2}sigma, {:.1}s [{}/{}]",
        r.summary.case_id,
        tag,
        r.k_calc,
        r.k_sigma,
        delta_pcm,
        bound,
        sigma_ratio,
        r.runtime_s + r.load_s,
        r.summary.seq,
        r.summary.total,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rule 1 (firm) — thermal scattering routes to GPU even when the
    /// workload would otherwise hit Rule 2 (tiny → CPU). GPU must be
    /// available; otherwise we fall through to Rule 2.
    #[test]
    fn router_rule1_thermal_to_gpu() {
        // Tiny workload + thermal + GPU → GPU.
        assert_eq!(
            auto_pick(true, 1_000, false, 0, true),
            BackendUsed::Cuda
        );
    }

    /// Rule 1 falls through when no GPU is present.
    #[test]
    fn router_rule1_thermal_without_gpu_falls_through() {
        // Same tiny + thermal but no GPU → Rule 2 fires → CPU.
        assert_eq!(
            auto_pick(true, 1_000, false, 0, false),
            BackendUsed::Cpu
        );
    }

    /// Rule 2 (firm) — tiny workload routes to CPU even when GPU is
    /// available, as long as Rule 1 doesn't fire.
    #[test]
    fn router_rule2_tiny_workload_to_cpu() {
        assert_eq!(
            auto_pick(false, TINY_WORK_THRESHOLD - 1, false, 0, true),
            BackendUsed::Cpu
        );
    }

    /// Rule 3 (soft) — large lattice case routes to GPU when present.
    #[test]
    fn router_rule3_lattice_to_gpu() {
        assert_eq!(
            auto_pick(false, 100_000_000, true, 0, true),
            BackendUsed::Cuda
        );
    }

    /// Rule 3 — many materials triggers GPU even without lattice.
    #[test]
    fn router_rule3_many_materials_to_gpu() {
        assert_eq!(
            auto_pick(false, 100_000_000, false, LATTICE_MATERIAL_THRESHOLD + 1, true),
            BackendUsed::Cuda
        );
    }

    /// Default — no firm rule fires, plain large case → GPU when
    /// present, CPU otherwise.
    #[test]
    fn router_default_to_gpu_when_present() {
        assert_eq!(
            auto_pick(false, 100_000_000, false, 5, true),
            BackendUsed::Cuda
        );
        assert_eq!(
            auto_pick(false, 100_000_000, false, 5, false),
            BackendUsed::Cpu
        );
    }

    #[test]
    fn mean_stderr_empty_is_nan() {
        let (m, s) = mean_stderr(&[]);
        assert!(m.is_nan());
        assert!(s.is_nan());
    }

    #[test]
    fn mean_stderr_single_sample_has_zero_stderr() {
        let (m, s) = mean_stderr(&[1.0]);
        assert_eq!(m, 1.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn mean_stderr_two_sample() {
        // [1.0, 3.0] mean=2.0, sample var=2.0, stderr = sqrt(2/2)=1.0.
        let (m, s) = mean_stderr(&[1.0, 3.0]);
        assert!((m - 2.0).abs() < 1e-12);
        assert!((s - 1.0).abs() < 1e-12);
    }
}
