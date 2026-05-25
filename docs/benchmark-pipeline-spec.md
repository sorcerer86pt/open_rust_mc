# Benchmark Pipeline — Architectural Specification (v2)

**Target:** `open_rust_mc` ICSBEP regression runner
**Replaces:** Original `BENCHMARK_PIPELINE_SPEC.md` (out-of-tree, in
`~/Downloads/`). Same goal, but corrected against the actual
codebase and hardened against the concurrency / CUDA-driver issues
raised in review.
**Root cause addressed:** The current sweep (`icsbep_bench`,
`icsbep_sweep.py`) restarts a fresh process per case, cold-initialises
the CUDA context, re-parses HDF5, re-uploads the GPU bundle, and
serialises every stage onto one thread. The 75 h projection on the
RTX 5090 (`results/icebsp_run_5090_full.txt`) is overhead, not
physics.

This spec is the result of a three-amigos review (Architect /
Implementer / Operator) plus Gemini's concurrency audit. It supersedes
the v1 wording wherever the two disagree.

---

## 1. Why v2

v1 was directionally correct but had three classes of errors:

1. **Invented APIs that don't exist.** `Geometry::from_json()`,
   `resolve_materials_with_cache()`, `KernelCache`,
   `ThermalCache`, `GpuTransportContext::assemble_bundle()` — none
   of these are real. v1 sketched against them anyway.
2. **Ignored CUDA driver concurrency semantics.** Stage 3
   pre-uploads "while Stage 4 runs" only works if the two stages use
   distinct CUDA streams. v1 assumed parallelism but didn't specify
   streams; on the default stream every H→D copy serialises with
   every kernel launch and the pipeline degenerates to the sequential
   path it was supposed to replace.
3. **Counted VRAM by slot count, not by bytes.** "n\_slots in flight"
   meant nothing while one Be S(α,β) bundle is 800 MB and one
   actinide upload is 1 GB. Two slots in flight ≠ headroom; two
   slots in flight = OOM on cases 4–5 of `heu-sol-therm-020`.
4. **One backend per run.** v1 treated CPU and GPU as mutually
   exclusive — pick one with `--runner`, the other half of the host
   sits idle. v2 makes them concurrent: CpuExecutor and GpuExecutor
   share the `SlotArray` and pull work in parallel, routed per
   `RunnerHint::{Cpu, Gpu, Auto}`. Small/cold cases (Godiva-class)
   finish on CPU while the GPU is mid-batch on the heavyweights.

v2 fixes these explicitly.

---

## 2. Three-amigos review

### 2.1 Architect — does the layering work?

**Verdict:** the six-stage shape (CaseLoader → DataLoader → SlotArray
→ Executor → ResultProcessor → Finalizer) is sound. The key
invariants:

- **Stage boundaries are channels with bounded capacity.** Backpressure
  is implicit; no stage can outrun its consumer beyond the queue
  depth. This makes the pipeline self-throttling.
- **Each stage owns exactly one external resource.** Stage 2 owns
  filesystem I/O. Stage 3 owns nuclear-data resolution + GPU H2D.
  Stage 4 owns kernel execution. Stage 5 owns CSV + stdout. No
  shared mutable state between stages other than the SlotArray and
  the result channel.
- **`RunContext` is a passive bag of `Arc`s.** All long-lived
  resources (`GpuTransportContext`, `TieredStore`, rayon pool) live
  there. Stages clone `Arc`s and hold them for the duration of the
  run.

**Concerns the architect signs off on:**

- **`EigenvalueRunner` stays.** The existing trait + `CudaRunner` /
  `CpuRunner` impls in `transport/dispatch.rs` are the right
  abstraction. Stage 4 wraps one or the other; no new trait needed.
- **`GpuRecursiveContext` is the event-based production context.**
  It already loads the 9 `gr_*` event kernels and runs the recursive
  geometry. Stage 4 holds an `Arc<GpuRecursiveContext>` and reuses it
  across cases.
- **`SimConfig` carries per-run settings.** Already in
  `transport/simulate.rs`. Stage 2 produces one per `CaseSpec`.

### 2.2 Implementer — can I build this from current code?

**Verdict:** roughly 60% of the named entities in v1 already exist
under different names; ~30% need thin adapter layers; ~10% are
genuinely new. The gap analysis below lists each.

**The single biggest implementation hazard** is migrating every
existing GPU operation off `ctx.default_stream()`. The current
codebase has ~30+ call sites that use `self.stream.clone_htod(...)`
or `self.stream.launch_builder(...)` on the implicit default stream.
The pipeline needs two named streams (compute, transfer). Either
**(a)** add a second stream to `GpuTransportContext` and route uploads
to it, or **(b)** pass a `&CudaStream` to each upload call. Option
(b) is more invasive but cleaner; option (a) requires careful
synchronisation at every site that today assumes "upload then launch"
runs in stream order.

### 2.3 Operator — what fails at 3am?

**Verdict:** v1 had no story for the failures we actually saw on
the 5090. v2 must define recovery for each.

**Failure modes seen on the 5090 run** (`results/icebsp_run_5090_full.txt`):

| Mode | What happened | v2 response |
|---|---|---|
| OOM during upload | `CUDA_ERROR_OUT_OF_MEMORY` after 1.3 s, case skipped | Stage 3 gates on `cuMemGetInfo` *before* issuing the upload. If predicted footprint > free × safety, block until Stage 4 releases the previous case. |
| Case wedged | One case stuck in transport kernel, no progress | Stage 4 wraps each `run()` in a watchdog; on timeout, `cudaDeviceReset()` + rebuild context + report ERROR. The current Python sweep had no such guard — when the 5090 wedged, the whole process was killed manually. |
| Sweep partially completed | `STOP_SWEEP` after 115 / 375 | CSV must flush per row; resume via `--resume` checks completed case-ids; no rerun lost work. |
| VRAM creep across cases | Free VRAM drifted down across cases | The fix landed in `8ce9369` (dynamic budget, defensive clear). Stage 3 inherits this. |

**Observability requirements** (new in v2):

- Per-case JSONL telemetry: timestamp, case-id, wall, stage timings
  (load, upload, exec, finalize), `gpu_debug_metrics()` snapshot
  (cache entries, bytes, hits, free VRAM, budget).
- Stop-file (`outputs/STOP`) honoured per-case boundary.
- Resume from CSV reads `case` column, skips already-completed.
- One-line stdout per case in the same format as the current Python
  sweep (graders / dashboards already parse this).

---

## 3. Gemini concurrency audit — integrated

The concurrency review (Gemini, see conversation log 2026-05-24)
identified three hard requirements v1 missed. All three are mandatory.

### 3.1 VRAM gating in Stage 3

**Problem.** v1 said "Stage 3 pre-uploads the next case while Stage 4
runs the current one." That's exactly the heu-sol-therm-020 OOM
scenario on the 5090 — two full bundles live in VRAM at the case
boundary, with one being a Be S(α,β) payload of ~800 MB on top of a
1 GB nuclide bundle.

**Resolution.** Before any `upload_*` call in Stage 3, query
`cuMemGetInfo`. Estimate the next bundle's footprint from the resolved
materials (sum of per-nuclide `approx_device_bytes` + SAB device
bytes + flat-pack overhead). If `predicted > free × 0.85`, block on
`SlotArray::empty_notify` until Stage 4 finishes a case and drops its
`Arc<GpuNuclideData>`. The slot pool stops being "n in flight" and
becomes "n in flight OR free VRAM allows, whichever binds first."

This *complements* the in-engine `defensive_clear_if_low_vram` shipped
in `8ce9369`. The defensive clear is a last-resort safety net inside
the upload path; the Stage 3 gate is a planner-level check that
avoids triggering the safety net.

### 3.2 Separate CUDA streams (compute vs transfer)

**Problem.** cudarc's `CudaContext::default_stream()` returns *the*
context-default stream. When Stage 3's H→D copies and Stage 4's
kernel launches both go through that stream, the driver serialises
them. The "Stage 3 pre-uploads while Stage 4 runs" claim becomes a
no-op — they run sequentially because they share the stream.

**Resolution.** Create two non-default streams at `RunContext`
construction time:

```rust
let stream_compute  = ctx.new_stream()?;   // Stage 4 kernel launches
let stream_transfer = ctx.new_stream()?;   // Stage 3 H→D copies
```

Cross-stream synchronisation via events:

```rust
// Stage 3, end of upload:
let upload_done = ctx.new_event(None)?;
upload_done.record(&stream_transfer)?;

// Stage 4, start of run that consumes that bundle:
stream_compute.wait(&upload_done)?;
```

**Hard rule:** the pipeline must never call `CudaContext::synchronize()`.
That collapses both streams into the implicit default in some cudarc
code paths and undoes the parallelism. All synchronisation must be
event-based.

**Migration cost:** every site in `gpu_transport.rs` /
`gpu_recursive.rs` / `gpu_per_nuclide.rs` that today does
`self.stream.clone_htod(...)` needs to take a `&CudaStream`
parameter. ~30 call sites. Mechanical change but pervasive — it's the
implementation hazard the implementer-amigo flagged in §2.2.

### 3.3 Dynamic VRAM budget — already landed

`bundle_cache_budget_bytes()` is no longer cached via `OnceLock`; it
recomputes from `cuMemGetInfo` on every call. The per-nuclide and SAB
caches share the budget via `per_nuclide_cache_budget_bytes` /
`sab_buffer_cache_budget_bytes` (sibling-aware). See commit `8ce9369`.

Stage 3 inherits this for free.

---

## 4. Entity map — what exists vs what's new

Cross-referenced against the actual `rust_prototype/src/` tree.

### 4.1 Reuse as-is

| v1 named it | Actual identity | Location |
|---|---|---|
| `HardwareProfile` / `hardware_profile()` | same | `hardware_profile.rs:log_startup_banner`, `HardwareProfile` struct |
| `GpuTransportContext::shared()` | same | `gpu_transport.rs:1022` — process-wide singleton via `OnceLock<Arc<GpuTransportContext>>` |
| `SimConfig` | same | `transport/simulate.rs:27` |
| `BatchResult` | same | `transport/simulate.rs:402` |
| `ResolvedMaterials` | same | `transport/material_resolve.rs:80` |
| `SceneDto` | same | `geometry/scene_io.rs:537` |
| `EigenvalueRunner` trait | same | `transport/dispatch.rs:78` |
| `CudaRunner` / `CpuRunner` | same | `transport/dispatch.rs:140` / `:92` |
| `GpuRecursiveContext` | same — this *is* the event-based context | `gpu_recursive.rs:371` |
| `GpuTransportContext::run_eigenvalue` style | `gpu_run_icsbep` in `bindings/python/src/lib.rs:3062` — extract verbatim into Stage 4 helper |
| `BatchResult` aggregation | `simulate::run_eigenvalue_with_geometry` for CPU; `CudaRunner::run` for GPU |

### 4.2 Reuse with renaming / re-export

| v1 named it | Actual identity | Adaptation |
|---|---|---|
| `Geometry::from_json()` | `scene_io::load_scene_from_json(text) -> LoadedScene` | Stage 2 calls this; uses `LoadedScene { geometry, materials, names }` directly. |
| `resolve_materials_with_cache()` | `resolve_materials_with_data_dir(materials, lib, rank, thermal_dir)` | Already cache-aware: thermal hits `material_resolve::thermal_cache` (private `OnceLock<RwLock<HashMap>>`); nuclide hits the `transport::nuclide_cache::TieredStore` singleton via `xs_provider::load_nuclide_with_policy`. No new wrapper needed. |
| `NuclideLibrary` | same | `transport/nuclides.rs:1265` — constructed once per run. |
| `TieredStore` | same, *but* process-wide singleton | `transport::nuclide_cache::TieredStore`, accessed via the private `cache()` function. Stage 1 needs a `pub fn shared() -> &'static TieredStore` accessor (small new export). |

### 4.3 New types (Stage scaffolding)

| Type | Module | Role |
|---|---|---|
| `RunContext` | `benchmark/run_context.rs` | Holds `Arc<HardwareProfile>`, `Arc<GpuTransportContext>` (when `cuda`), `Arc<GpuRecursiveContext>`, run-wide rayon pool, `RunArgs`, `data_dir`, `bench_dir`, the two streams from §3.2, slot pool size. |
| `RunArgs` | `benchmark/run_args.rs` | clap-derived CLI args. Mirrors the existing Python `icsbep_sweep.py` arg surface so the harness migration is a drop-in. |
| `CaseSpec` | `benchmark/case_loader.rs` | Output of Stage 2: `case_id`, `seq`, `total`, `k_ref`, `sigma_exp`, `source`, `Arc<Geometry>`, `Arc<SceneDto>`, `SimConfig`, `RunnerHint`. Also carries `has_thermal_scattering()` and `material_count()` helpers used by the §5.4.0 router. |
| `RunnerHint` | `benchmark/case_loader.rs` | `Cpu`, `Gpu`, `Auto`. `Auto` defers to §5.4.0 heuristic. Default `Auto` so users opt out of heterogeneous routing rather than opt in. |
| `CaseBundle` | `benchmark/case_bundle.rs` | Output of Stage 3: `CaseSpec` + `ResolvedMaterials` + `GpuBundleHandle` (when GPU). |
| `GpuBundleHandle` | `benchmark/case_bundle.rs` | Newtype wrapping the four existing GPU device-buffer structs (`GpuNuclideData`, `GpuMaterialData`, `GpuSabData`, `GpuWmpData`) + the per-nuclide cache `Arc`s. Carries the `CudaEvent` from §3.2 so Stage 4 can wait on transfer completion. |
| `Slot` enum | `benchmark/case_bundle.rs` | `Empty / Loading{case_id} / Ready(Box<CaseBundle>) / Running{case_id} / Done`. |
| `SlotArray` | `benchmark/case_bundle.rs` | `Vec<Mutex<Slot>>` + `Condvar`s. Sized by `RunContext::n_slots`. |
| `ExecutionResult` | `benchmark/executor.rs` | `case_id`, `seq`, `k_calc`, `k_sigma`, `k_track`, `k_track_sigma`, `runtime_s`, `load_s`, `n_histories`, the originating `CaseSpec`, `Option<Error>`. |
| `RunState` | `benchmark/stats.rs` | Pass/fail counters, accumulated results vector. |
| `Pipeline` | `benchmark/pipeline.rs` | The orchestrator. Spawns Stage 2 / Stage 3 / Stage 4 / Stage 5 threads, wires channels, joins them. |

### 4.4 Existing entities that need extension

| Entity | Change |
|---|---|
| `GpuTransportContext` | (a) Migrate singleton storage from `OnceLock<Arc<Self>>` to `OnceLock<RwLock<Option<Arc<Self>>>>` (enables watchdog rebuild — see §5.4.2). (b) Add `pub fn new_compute_stream() -> Arc<CudaStream>` / `pub fn new_transfer_stream() -> Arc<CudaStream>`. (c) Add `&CudaStream` parameter to every `upload_*` method (the existing default-stream signatures stay as back-compat wrappers). (d) Add `pub fn estimate_sab_device_bytes(slots: &[(Arc<ThermalScatteringData>, usize)], n_nuc: usize) -> usize` — sums each TSL's projected device footprint (`n_temps × (n_inc_E × header + Σ_inc(n_E_out × 8) + bragg_edges + …)`) without uploading. Stage 3 calls this for the pre-flight VRAM check. |
| `transport::nuclide_cache` | Add `pub fn shared() -> &'static TieredStore`. Today the singleton is reached only via the private `cache()`. |
| `transport::dispatch::CudaRunner` | Add `pub compute_stream: Arc<CudaStream>` field. Internal launches inside `CudaRunner::run` route through it instead of `self.recursive.stream` / `self.transport.stream`. The launch sites inside `GpuRecursiveContext::transport_recursive_with_buffers` need a corresponding `stream: &CudaStream` parameter (or a `with_stream(stream)` adapter) so the runner can thread `compute_stream` all the way down. |
| `bindings/python/src/lib.rs:3062` (`run_gpu_icsbep`) | Refactor into `gpu_run_eigenvalue_case(ctx, bundle, sim_config) -> EigenvalueOutcome`, callable from both Stage 4 *and* the existing Python entry point. The Python entry point becomes a single-case driver that allocates a one-shot Stage 4 instance. |

### 4.5 No new CUDA kernels needed

Despite the earlier "maybe a new kernel" concern, the pipeline is
purely host-side orchestration. All synchronisation between streams
goes through `CudaEvent` (record + wait), which is API, not kernel.
The four event-based kernels (`gr_init_stacks`, `gr_trace_and_sample`,
`gr_partition`, `gr_elastic_event`, …) stay as the production
transport path.

---

## 5. Stage-by-stage spec (corrected)

### 5.1 Stage 1 — `RunContext` (orchestrator)

Runs once at process start. All other stages receive an
`Arc<RunContext>`.

```rust
pub struct RunContext {
    // Hardware
    pub hw:          Arc<HardwareProfile>,
    pub rayon_pool:  Arc<rayon::ThreadPool>,    // N_cpu - 2 threads

    // GPU — None when --runner=cpu
    #[cfg(feature = "cuda")]
    pub gpu_t:       Option<Arc<GpuTransportContext>>,
    #[cfg(feature = "cuda")]
    pub gpu_r:       Option<Arc<GpuRecursiveContext>>,
    #[cfg(feature = "cuda")]
    pub stream_compute:  Option<Arc<CudaStream>>,
    #[cfg(feature = "cuda")]
    pub stream_transfer: Option<Arc<CudaStream>>,

    // Process-wide nuclide cache — shared across cases
    pub nuclide_store: &'static TieredStore,

    // Nuclide library (file paths, ZAID lookups). Constructed once.
    pub lib: Arc<NuclideLibrary>,

    // Run params
    pub args:       RunArgs,
    pub data_dir:   PathBuf,
    pub bench_dir:  PathBuf,
    pub n_slots:    usize,   // see §5.3.1
}
```

**Responsibilities:**

- Parse CLI args. Print hardware banner (`log_startup_banner()`).
- Call `GpuTransportContext::shared()` once. Build a fresh
  `GpuRecursiveContext` per geometry shape (see open question
  §7.1 — geometry-specific kernel cache).
- Create the two CUDA streams.
- Build `rayon_pool` with `N_cpu - 2` threads (leave 1 each for the
  GPU dispatch thread and the loader/result threads).
- Construct `NuclideLibrary` from `data_dir`.
- Compute `n_slots` per §5.3.1.
- Hand off to `Pipeline::run`.

### 5.2 Stage 2 — `CaseLoader`

One dedicated thread. Walks `bench_dir`, parses each
`bench/icsbep/*.json`, produces `CaseSpec`s, pushes onto a
`crossbeam_channel::bounded::<CaseSpec>(n_slots * 2)`.

Reuses `scene_io::load_scene_from_json` for the `scene` block and
`benchmark` block deserialisation logic that already lives in the
Python sweep (`run_icsbep_case` in `bindings/python/src/lib.rs:2782`)
— extract into a pure-Rust `parse_case_json(text) -> Result<CaseSpec, _>`.

The loader does *not* touch HDF5 or GPU. It produces `CaseSpec`s with
fully parsed scene + geometry only.

### 5.3 Stage 3 — `DataLoader`

One dedicated thread (or a small fixed pool of 2–3 for parallel
intra-case nuclide loading). Consumes from the `LoadQueue`, writes
`CaseBundle`s into the `SlotArray`. **Runs ahead of Stage 4 by
`n_slots` cases, but gated by §3.1 VRAM budget.**

#### 5.3.1 SlotArray sizing

```
n_slots = max(2, min(8, floor(0.7 × total_vram_bytes / avg_bundle_bytes)))
```

`avg_bundle_bytes` defaulted to 1 GB; updated from observed bundle
sizes after case 3 onward. On the 5090 (32 GB total, ~24 GB available
after driver / particle bank): `n_slots = 8`. On the A1000 (4 GB, ~3
GB available): `n_slots = 2`.

#### 5.3.2 Per-case load pipeline

For each `CaseSpec`:

1. **Resolve materials.** `resolve_materials_with_data_dir(...)`. On
   warm cache (typical after case ~5), each per-nuclide call is an
   `Arc` clone from `TieredStore::L1`. On cold cache, parses HDF5 +
   does SVD decomposition and inserts.

2. **(GPU only) Pre-flight VRAM check.** Query
   `gpu_t.ctx().mem_get_info()`. Estimate `predicted_bytes`:
   ```
   predicted = Σ nuc.approx_device_bytes()
             + gpu_t.estimate_sab_device_bytes(&sab_slots, n_nuc)
             + flat_pack_overhead(resolved.nuclides.len())
             + particle_bank_bytes(sim_config.particles_per_batch)
   ```
   `estimate_sab_device_bytes` is the new helper from §4.4 — without
   it the SAB contribution (often the *largest* per-case term on
   thermal cases — Be at 11 temperatures runs ~800 MB) is invisible
   to the gate. `flat_pack_overhead` is a constant ~`n_nuc × N_PARAMS
   × 8` for the pointer-array buffers in `GpuNuclideData`.

   If `predicted > free × 0.85`, block on `SlotArray::empty_notify`
   until Stage 4 drops a bundle.

3. **(GPU only) H→D upload on `stream_transfer`.** All
   `upload_nuclide_data` / `upload_material_data` /
   `upload_sab_data_multi_cached` calls take an explicit
   `&stream_transfer` argument. The existing in-engine LFU cache +
   defensive clear keep working — they're orthogonal to which stream
   the upload uses.

4. **(GPU only) Record completion event.**
   ```rust
   let upload_done = ctx.new_event(None)?;
   upload_done.record(&stream_transfer)?;
   bundle.gpu_data.as_mut().unwrap().upload_done = Some(upload_done);
   ```

5. **Write slot.** Transition `Loading{case_id}` → `Ready(bundle)`.
   Signal `ready_notify`.

### 5.4 Stage 4 — `Executor` (heterogeneous, CPU + GPU concurrent)

**Two executors run concurrently**, both consuming `Ready` slots from
the same `SlotArray`. The slot's `runner` field (set by Stage 2 and
possibly revised by Stage 3) determines which executor claims it.
Each executor calls the right `EigenvalueRunner` impl, emits an
`ExecutionResult` to the `ResultChannel`, and transitions the slot
to `Done`.

The goal is to keep the GPU busy *and* keep the otherwise-idle CPU
cores doing useful work — small ICSBEP cases where GPU setup
amortises poorly (Godiva-style 3-nuclide HEU at < 50 k particles)
finish faster on CPU than they spend uploading to the device. While
the GpuExecutor is mid-batch on a PWR-17×17 case, the CpuExecutor
finishes three Godivas off to the side.

#### 5.4.0 Routing rule — `RunnerHint::Auto`

Rule-based, not a single proxy. Each rule fires on data we have at
the end of Stage 2 (parsed but not yet HDF5-resolved). Rules are
ordered — the first match wins. Calibrating any one rule doesn't
disturb the others.

```rust
pub enum RunnerHint { Cpu, Gpu, Auto }

fn route(spec: &CaseSpec, has_gpu: bool) -> Backend {
    match spec.runner {
        RunnerHint::Cpu  => Backend::Cpu,
        RunnerHint::Gpu  => Backend::Cuda,
        RunnerHint::Auto => auto_pick(spec, has_gpu),
    }
}

fn auto_pick(spec: &CaseSpec, has_gpu: bool) -> Backend {
    // Rule 1 (firm): thermal scattering → GPU.
    //   CPU's branchy SAB sampling is the engine's worst CPU path;
    //   GPU dominates 10×+ on Be / H_in_H2O cases.
    let has_thermal = spec.scene.materials
        .iter()
        .any(|m| !m.thermal_files.is_empty());
    if has_thermal && has_gpu { return Backend::Cuda; }

    // Rule 2 (firm): tiny workload → CPU.
    //   Below ~10M particle-events the GPU's setup overhead (NVRTC
    //   amortized, but per-case nuclide upload still ~1 GB / 1 s on
    //   the 5090) is bigger than the CPU run itself on a modern
    //   16-core box. Threshold from `outputs/saturation_*.csv` —
    //   re-tune per platform.
    let work = (spec.config.particles_per_batch as u64)
             .saturating_mul(spec.config.batches as u64);
    if work < 10_000_000 { return Backend::Cpu; }

    // Rule 3: lattice / large-multi-material geometry → GPU.
    //   Assembly-class scenes (PWR-17×17, hex minicore) amortise
    //   even tiny-batch setups by the recursive descent the GPU
    //   does well.
    let lattice = !spec.scene.lattices.is_empty()
               || !spec.scene.hex_lattices.is_empty();
    let many_mat = spec.scene.materials.len() > 50;
    if (lattice || many_mat) && has_gpu { return Backend::Cuda; }

    // Default: GPU when available, otherwise CPU.
    if has_gpu { Backend::Cuda } else { Backend::Cpu }
}
```

**What each rule reads, why it's firm vs soft:**

| Rule | Reads | Firmness | Failure mode if wrong |
|---|---|---|---|
| 1 — thermal | `SceneDto.materials[].thermal_files` | Firm — no counterexamples in the 375-case ICSBEP corpus | If we wrongly route a thermal case to CPU, expect ~5–10× slowdown but still correct |
| 2 — tiny | `SimConfig.particles_per_batch × batches` | Firm — measured crossover from `outputs/saturation_*.csv` | Wrong → small case takes ~5 s of GPU setup before ~0.5 s of work |
| 3 — lattice / many-mat | `SceneDto.{lattices, hex_lattices, materials}` | Soft — geometry-complexity heuristic only | Wrong → modest slowdown, GPU still fine for borderline cases |
| Default | — | Hard fallback | GPU preferred when present; no downside vs current `--runner gpu` |

**Stage 3 override.** If the VRAM gate (§5.3.2) refuses the upload
(`predicted > free × 0.85` with no `Ready` slots to evict), Stage 3
downgrades the hint to `Cpu` and re-queues. This is the safety
valve for the `heu-sol-therm-020_case-4/5` OOM mode — on a tight
VRAM budget the case runs slow on CPU instead of erroring out.

**Calibration loop.** The JSONL telemetry records `routed_to`,
`wall_s`, `load_s`, `exec_s`. After N runs, a calibration script can
fit the thresholds (10M particle-events, 50 materials) from real
data per host. Out of scope for the MVP.

#### 5.4.1 Thread budget

For a host with `N` logical cores:

| Role | Threads | Notes |
|---|---|---|
| Stage 2 CaseLoader | 1 | JSON parse, light |
| Stage 3 DataLoader | 2 | HDF5 parse + SVD across nuclides via rayon child pool |
| Stage 4 GpuExecutor | 1 | Drives kernel launches; mostly blocks waiting on the GPU |
| Stage 4 CpuExecutor | 1 | Coordinator; the actual work happens in the shared rayon pool |
| Stage 5 ResultProcessor | 1 | CSV append + stdout |
| Shared rayon pool | `N − 6` | Parallel-history transport for CPU cases + nuclide load helpers |

On a 20-thread machine (MSI Home from CLAUDE.md): rayon pool = 14.
On a 32-thread workstation: 26. Below 8 cores the pool collapses to
`max(2, N − 6)`, the CpuExecutor still works but each CPU case ties
up the pool start-to-end (one CPU case in flight at a time).

#### 5.4.2 Slot claim protocol

The slot transitions are atomic; concurrent executors cannot both
claim the same slot.

```rust
// On both executor threads:
loop {
    let claim = slot_array.take_ready_for(my_backend);   // blocks
    let (slot_idx, bundle) = match claim {
        SlotClaim::Took(slot, bundle) => (slot, bundle),
        SlotClaim::Drain => break,  // Stage 5 saw STOP file
    };
    // … run + emit result …
}
```

`take_ready_for(backend)` skips slots tagged for the other backend.
The two executors never contend for the same slot.

#### 5.4.3 GpuExecutor body

```rust
// stream_compute borrowed from RunContext.
// §3.2 — wait on the upload event from Stage 3 before any kernel
// launch consumes the bundle's buffers.
if let Some(ref ev) = bundle.gpu_data.as_ref().unwrap().upload_done {
    stream_compute.wait(ev)?;
}

let runner = CudaRunner {
    recursive: &gpu_r,
    transport: &gpu_t,
    nuc_data:  &bundle.gpu_data.as_ref().unwrap().nuc_data,
    // … remaining existing fields (mat_data, sab_data, wmp_data,
    //   mat_k_t, sab_nuc_idx, max_events_per_history, fis_capacity,
    //   initial_source, buffers, refill) …
    compute_stream: Arc::clone(&stream_compute),  // new — see §4.4
};
let outcome = runner.run(&bundle.spec.config);

result_tx.send(ExecutionResult::from(outcome, &bundle.spec))?;
slot_array.mark_done(slot_idx);   // signals empty_notify
```

The CUDA context is **never** re-initialised between cases.
`GpuTransportContext::shared()` returns the same `Arc` every call —
modulo a watchdog rebuild (see §5.4.5).

#### 5.4.4 CpuExecutor body

```rust
// rayon_pool borrowed from RunContext. The CPU runner uses
// par_iter under the hood; we install the pool for the duration
// of this case so rayon picks our threads, not the global default.
rayon_pool.install(|| {
    let runner = CpuRunner {
        geometry:    &bundle.spec.geometry,
        materials:   &bundle.resolved.materials,
        xs_provider: &bundle.resolved.provider,
    };
    let outcome = runner.run(&bundle.spec.config);
    result_tx.send(ExecutionResult::from(outcome, &bundle.spec))?;
});

slot_array.mark_done(slot_idx);
```

The CpuExecutor consumes the whole rayon pool for the duration of
its case. If the routing heuristic puts two CPU cases adjacent in
time, the second one waits in `Ready` until the first finishes
(serialised on the CPU side). The GpuExecutor continues running its
own queue independently the entire time.

For CPU-only sweeps (`--runner cpu`), the GpuExecutor thread is not
spawned; every slot routes to CpuExecutor. Multiple CPU cases can
run in parallel only if the user explicitly sets
`--n-cpu-executor-threads K`, splitting the rayon pool into `K`
sub-pools of `(N − 6) / K` threads each. Default `K = 1` for
predictable per-case wall.

#### 5.4.5 Timeout watchdog

Each `runner.run(config)` call (on either executor) is wrapped in a
worker thread with a configurable timeout
(`RunArgs::case_timeout_s`, default 3600 s). On timeout:

- **GPU:** drop the current context, `cuDeviceReset`, build a fresh
  `GpuTransportContext` + `GpuRecursiveContext`, swap them into the
  shared slot. The CpuExecutor pauses on a barrier during the
  rebuild (it doesn't need the GPU but the reset is a context-wide
  event). The next case sees a cold context (loses cache state but
  the sweep doesn't die). Implementation requires changing the
  singleton storage — see §4.4.
- **CPU:** join with timeout, log hung threads, mark case `ERROR`,
  rebuild the rayon pool. The GpuExecutor keeps running unaffected.

Replaces the current "kill the subprocess" strategy.

**Singleton storage change.** Today `GpuTransportContext::shared`
uses `static SHARED: OnceLock<Arc<GpuTransportContext>>` — write-once,
no rebuild path. For watchdog recovery, migrate to:

```rust
static SHARED: OnceLock<RwLock<Option<Arc<GpuTransportContext>>>>
    = OnceLock::new();

pub fn shared() -> Result<Arc<Self>, Box<dyn Error>> {
    let cell = SHARED.get_or_init(|| RwLock::new(None));
    {
        let g = cell.read().expect("poisoned");
        if let Some(arc) = g.as_ref() { return Ok(Arc::clone(arc)); }
    }
    // First call (or post-reset): build under write lock.
    let mut g = cell.write().expect("poisoned");
    if let Some(arc) = g.as_ref() { return Ok(Arc::clone(arc)); }
    let fresh = Arc::new(Self::new()?);
    *g = Some(Arc::clone(&fresh));
    Ok(fresh)
}

pub fn force_rebuild() -> Result<Arc<Self>, Box<dyn Error>> {
    let cell = SHARED.get().ok_or("not initialised")?;
    { let mut g = cell.write().expect("poisoned"); *g = None; }
    // Old Arc dropped here, then `cuDeviceReset` invalidates remaining
    // device handles. The watchdog calls this from a single thread
    // while Stage 4 is held at a barrier (no concurrent reads).
    unsafe { cudarc::driver::sys::cuDeviceReset(); }
    Self::shared()
}
```

Read path cost is one `RwLock::read` acquisition per `shared()` call
(nanoseconds vs the per-case wall of seconds — negligible). Writes
happen only on first init and on watchdog rebuild.

`Option<Arc<_>>` lets us be in the "no context exists right now"
state during a rebuild — without it we'd leave a dangling `Arc`
pointing at memory the reset just invalidated. Stage 4's executor
treats `None` as "GPU in recovery; block on the watchdog completion
signal."

### 5.5 Stage 5 — `ResultProcessor`

One dedicated thread. Reads `ResultChannel`, computes pass/fail,
appends CSV row (flushed every row), prints one-line stdout in the
existing format, updates `RunState`. Honours `outputs/STOP` stop-file
between cases.

Output format matches the current Python sweep:
```
{case_id}: {PASS|FAIL|ERROR} -- k={k:.5}+/-{σ:.5}, delta={Δ:+.0}pcm, \
  bound=+/-{:.0}pcm, {σ_ratio:.2}sigma, {wall_s:.1}s [{seq}/{total}]
```

Incremental scatter plot every `RunArgs::plot_every` cases (default
10): writes `outputs/{run_name}_delta_k_scatter_partial.png`. Final
plot replaces the partial.

### 5.6 Stage 6 — `RunFinalizer`

Joins all stage threads. Prints summary table (`human_summary()`,
already exists). Writes final scatter + EALF correlation plots.
Releases the `Arc<RunContext>`; the GPU context drops when the last
ref goes (Python in-process callers keep theirs).

---

## 6. Failure-mode coverage

| Failure | Detected by | Recovery |
|---|---|---|
| OOM at upload | `mem_get_info` pre-check + driver error | Block in Stage 3 until Stage 4 releases a case; if still failing, `defensive_clear_if_low_vram` runs inside the upload; if *still* failing, propagate as `ExecutionResult::error`. |
| Case wedged (kernel hang) | Stage 4 watchdog timeout | `cudaDeviceReset` + rebuild GPU context. Report case `ERROR(timeout)`. |
| Single nuclide bigger than budget | `evict_to_budget` at upload time | Existing in-engine eviction handles it; cache holds 1 oversized entry by design. |
| HDF5 parse error | `resolve_materials_*` returns `ResolveError` | Stage 3 emits `ExecutionResult::error(case_id, e)`; Stage 4 skipped; sweep continues. |
| Panic in Stage 4 kernel call | `std::panic::catch_unwind` in executor | Report `ERROR(panic: {msg})`; sweep continues. |
| `outputs/STOP` appears | Stage 5 polls between cases | Drain in-flight, mark remaining as not-run, exit cleanly. |
| Power loss / kill -9 | N/A (host died) | `--resume` reads CSV's `case` column on next start. |

---

## 7. Open questions / deferred

1. **Per-geometry `GpuRecursiveContext` cache.** Each case has a
   recursive geometry; `GpuRecursiveContext::build(&geom, n)` uploads
   geometry tables. The PWR-17×17 assembly and a Godiva sphere need
   *different* `GpuRecursiveContext` instances. Currently the Python
   sweep rebuilds on every case. Open: cache by geometry hash, drop
   when not used. Likely cheap to build (no PTX compile, just table
   upload). Defer until profile shows it's a bottleneck.
2. **Multi-GPU.** Out of scope for v2. The architecture (single
   `RunContext`, one set of streams) assumes one device. Adding a
   second device is a separate spec.
3. **Validation-only kernels.** `k_find_cell_batch`,
   `k_trace_step_batch`, `k_multi_step_walk`, `k_const_xs_transport`
   are loaded into `GpuRecursiveContext` but only used by
   `gpu_recursive_parity` and `gpu_const_xs_keff` validation binaries.
   Open: split into a `GpuParityContext` so the production
   `GpuRecursiveContext` carries only the event-based kernels.
   Independent of pipeline.
4. **Refill pool sizing.** `CudaRunner::buffers` and `::refill` are
   `RefCell<Option<...>>` lazily built on first batch. Stage 4 needs
   to either keep one runner per slot or rebuild buffers when the
   batch shape changes between cases. Defer to implementation.

---

## 8. Migration plan

Four phases, each independently mergeable, each smoke-testable on
the A1000 before the 5090 sees them.

### Phase 1 — entities + stream plumbing (no behaviour change)

- New: `benchmark/{run_context, run_args, case_loader, case_bundle,
  stats, executor, result_processor, finalizer, pipeline}.rs`
  scaffolds. All initially marked `#[allow(dead_code)]` until
  wired in.
- `transport::nuclide_cache::pub fn shared()`.
- `GpuTransportContext::new_compute_stream` /
  `new_transfer_stream`.
- Add `&CudaStream` overloads of `upload_nuclide_data`,
  `upload_material_data`, `upload_sab_data_multi_cached`,
  `upload_wmp_data_*`. Default-stream versions stay as
  back-compat wrappers.
- `cargo test --features cuda --lib`: 451 / 451 stays green; no
  semantic change yet.

### Phase 2 — Stage 2 + Stage 3 wired, single-threaded driver

- Extract `parse_case_json` from the Python binding into
  `benchmark/case_loader.rs`.
- Wire `Pipeline::run` to call Stage 2 → Stage 3 → Stage 4 sequentially
  (no threads yet). Confirms each stage is correct in isolation.
- A1000 ICSBEP smoke: `heu-met-fast-001` 1–2 cases, `cargo run
  --bin benchmark_runner ...`. Same k_eff as Python sweep.

### Phase 3 — pipeline parallelism

- Spawn Stage 2, Stage 3, Stage 4 GpuExecutor, Stage 4 CpuExecutor,
  Stage 5 as separate threads.
- Wire `crossbeam-channel` for `LoadQueue` and `ResultChannel`.
- Implement `SlotArray` with proper notify/wait and the
  backend-filtered claim (`take_ready_for(backend)`).
- Stage 3 uses `stream_transfer`; GpuExecutor uses `stream_compute`;
  cross-stream sync via `CudaEvent`.
- Implement `RunnerHint::Auto` router (§5.4.0).
- A1000 smoke: a sweep of mixed cases (`heu-met-fast-001` =
  CPU-cheap + `heu-sol-therm-001` = GPU-required) with
  `--runner auto`. Confirm in telemetry that the two cases overlap
  on different backends (Cpu wall ⨯ Gpu wall has non-zero
  intersection).

### Phase 4 — production polish

- Watchdog timeout + recovery.
- Resume from CSV.
- Stop-file handling.
- Incremental plotting hook.
- 5090 dry run: 30-case subset of `bench/icsbep/`. Target: ≥ 2×
  speedup vs current Python sweep on the same cases. Compare against
  the `results/icebsp_run_5090_full.txt` baseline.
- Full 375-case 5090 sweep: target 25–35 h (vs 75 h projected for the
  serial path).

### Phase 5 — retire old runner

- `bin/icsbep_bench.rs` (subprocess loop) → thin shim that calls
  `benchmark_runner`.
- `examples/icsbep_sweep.py` deprecated (kept for back-compat). The
  Python `run_icsbep_case` single-case entry point stays since it
  drives Jupyter / interactive use; it just shares the Stage 4 logic
  via the extracted `gpu_run_eigenvalue_case`.

---

## 9. Acceptance criteria

The pipeline is "done" when:

1. `cargo test --features cuda --lib` is green.
2. `cargo run --release --features cuda --bin benchmark_runner --
   --bench-dir bench/icsbep --filter heu-met-fast --runner gpu`
   produces the same k_eff as the current Python sweep on the same
   cases, within MC noise.
3. The full 375-case sweep completes on the 5090 in < 35 h with no
   manual intervention (no OOM kills, no wedged-case restarts).
4. JSONL telemetry from the 5090 dry run satisfies
   `Σ load_s + Σ exec_s ≥ total_wall_s × 1.2` — i.e. accumulated
   per-stage work exceeds wall by at least 20%, proving Stage 3 and
   Stage 4 actually overlap. Anything below that would mean the
   stages are running sequentially and the pipeline is theatre.
5. The 2 OOMs that wedged the v1 sweep
   (`heu-sol-therm-020_case-4`, `_case-5`) either complete or
   surface as `ERROR(OOM)` without halting the sweep.

---

## Appendix A — file map for new module tree

```
rust_prototype/src/benchmark/
├── mod.rs
├── case_bundle.rs       # CaseBundle, GpuBundleHandle, Slot, SlotArray
├── case_loader.rs       # parse_case_json, CaseSpec, Stage 2 thread
├── data_loader.rs       # Stage 3 thread, VRAM-gated pre-upload
├── executor.rs          # Stage 4 (GPU + CPU), watchdog
├── finalizer.rs         # Stage 6 summary + final plots
├── pipeline.rs          # Pipeline orchestrator, channel wiring
├── result_processor.rs  # Stage 5 thread
├── run_args.rs          # clap-derived RunArgs
├── run_context.rs       # RunContext
└── stats.rs             # RunState, pass/fail accounting

rust_prototype/src/bin/
└── benchmark_runner.rs  # main: parse args, build RunContext, call Pipeline::run
```

`lib.rs` gains `pub mod benchmark;` (gated on neither cpu nor cuda
specifically — Stage 4 is the only cuda-conditional piece).

---

**Document status:** v2.0. Reviewed by Architect / Implementer /
Operator amigos and Gemini concurrency audit. Ready to drive Phase 1
implementation.
