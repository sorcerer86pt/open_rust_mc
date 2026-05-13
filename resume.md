# Session resume — 2026-05-13 (eod)

## State

Branch: `main`, working tree heavy with un-committed work (ICSBEP scene
expansion + sampler refactor + Python harness). Engine builds clean:
`cargo check` default and `cargo check --features cuda` both green;
`cargo check -p open-rust-mc-py [--features cuda]` green; lib tests
were green at the start of session (no engine-internal regressions
expected from this batch but not re-run end-to-end).

## What landed this session (in commit order this evening)

1. **Single-source-of-truth `MAX_NUCLIDES_PER_MATERIAL = 128`** at
   `src/lib.rs`. CPU imports it (`simulate.rs::MAX_NUCLIDES`); GPU
   receives it via NVRTC `-DMAX_NUC_PER_MAT=N` flag from
   `gpu_transport.rs::transport_kernel_options` and
   `gpu_recursive.rs::assemble_kernel_source`. `transport.cu` has no
   `#define` fallback (`#error`s if NVRTC forgets the flag). Bumped
   32 → 128 so HMF-069 (69 nuclides) and Pu nitrate solutions
   (67 nuclides) run.

2. **`SimLimits` (`src/transport/sim_limits.rs`)** — engine policy
   knobs (`max_events_per_history`, `fis_capacity_factor`,
   `sab_temperature_tolerance`, `initial_source_max_attempts_factor`)
   with TOML loader. Replaces 5_000 / 4× / 0.5 / 10_000 magic literals
   at every CudaRunner site + `try_initial_source` + SAB-tolerance
   sites in all binaries.

3. **Region-tree AABB walker + fissionability-aware sampler.**
   `Region::world_aabb(surfaces)` (geometry/cell.rs) does the textbook
   CSG recursive walk. `ResolvedMaterials::fissionable_materials()`
   marks any material with `nu_bar_const > 0` as fissionable.
   `try_initial_source_in_materials` builds a per-cell AABB+volume
   table, samples weighted by volume, accepts any draw in a
   fissionable cell — mirrors Serpent 2's default. Eliminates the
   `target_idx` + `smallest-volume-material` heuristics that broke on
   BWR cruciforms / PWR poisons / HFIR plates / CANDU spacers.
   `CudaRunner::run` now honours `config.initial_source_bank` (was a
   bug — GPU silently bypassed the fissionable pre-seed).

4. **Multi-slot S(α,β) on GPU** — `upload_sab_data_multi` packs N TSLs;
   per-nuclide `slot_per_nuc` lookup table replaces the legacy single
   `sab_nuc_idx` scalar. New params slots 123–129 (N_PARAMS = 130).
   Unlocks H-in-H₂O + D-in-D₂O + C-in-graphite in one run.

5. **Python bindings — `Runner` enum + `run_icsbep_case`**. Full
   GpuCuda × XsMode matrix dispatched through the existing
   `EigenvalueRunner` trait. `IcsbepResult` carries k_calc / k_ref /
   Δ / σ_ratio / bound / `passed` / runner / timing.

6. **`bench/icsbep/` cleanup + expansion** (376 cases → 374 scenes
   + 3 CLI-runner manifests moved to `bench/cli_runners/`):
   - `hmf-001_godiva.json` — new single-region Godiva scene (was a
     CLI-runner manifest; full Godiva-IV layered case still lives at
     `heu-met-fast-001_case-1.json`).
   - `internal_pwr_pincell.json` — new KARMA ACE7 pin cell
     (3.0 wt% U-235), `k_ref = 1.24206 ± 0.00025`.
   - `pwr_assembly_17x17.json` — new KARMA ACE7 17×17 assembly
     (264 fuel + 25 GT positions), `k_ref = 1.25827 ± 0.00029`.
     Engine produces 1.25120 at cheap settings ⇒ Δ = −707 pcm
     (within thermal-lattice cross-code disagreement).
   - All 373 scene-based JSONs got a `benchmark.recommended_settings`
     block (per-family tuned: `met-fast` 150/30/20000/5,
     `sol-therm` 150/50/15000/5, etc. — Lee et al. M&C 2011 PDF the
     reference for the PWR cases).

7. **Python harness `icsbep_sweep.py`** — start / stop / resume:
   - Per-case row written to CSV as soon as it completes (`fp.flush()`
     after each row), so a `kill -9` between cases loses nothing.
   - `--resume` reads existing CSV, skips completed cases.
   - `--stop-file outputs/STOP` checked between cases for graceful
     termination. Ctrl-C (SIGINT) also flushes and exits 0.
   - `--seeds N` (or JSON `recommended_settings.seeds`) runs each
     case N times, reports mean ± seed-to-seed stderr. Matches
     `tests/cuda_runs.rs::run_case_cuda_seeds` semantics.
   - Per-case settings precedence: JSON `recommended_settings` →
     CLI flags → built-in defaults.

8. **`preview_scene` binary** (`src/bin/preview_scene.rs`, gated on
   `preview` feature) — interactive XY-cross-section viewer for any
   scene JSON, mirrors `pwr_assembly --preview` plumbing. Walks
   `bench/icsbep/` and `data/` upward from CWD / exe path, so works
   from any subdirectory.

## Known issues / deferred

- **`preview_scene` rendering bug** for JSON-loaded lattices. The
  17×17 PWR assembly renders as one big annular ring (or quartered
  pins when surfaces shifted to `(pitch/2, pitch/2)`) instead of a
  proper 17×17 grid. Transport still produces the correct k_eff —
  this is visualization only. Engine's own `pwr_assembly --preview`
  binary (hand-built `Geometry`) works fine.

- **`internal_pwr_pincell`** at cheap settings shows Δ ≈ +2000 pcm vs
  the KARMA reference, but that's 3-batch / 300-particle sampling
  noise. At the JSON's `recommended_settings` it'll tighten well
  inside bound.

- **CANDU / SFR scenes** not built. Outline drafted in the chat log
  (simplified pin-cell variants using D₂O + Na coolants) — needs a
  follow-up session once the preview bug is closed so we can
  visually validate them.

- **`pwr_assembly.rs` binary's k_inf = 1.14958** disagrees with the
  KARMA paper's 1.25827 by ~8700 pcm. Probably the binary's hand-built
  geometry has the pin cylinders at `(pitch/2, pitch/2)` which, with
  the engine's element-CENTRE-relative lattice descent, puts them at
  element corners not centers. Not a regression introduced this
  session — flagged for separate audit.

- **Sweep error column = 0** on the full 374-scene corpus after this
  session's fixes. Pre-fix baseline was 51 ERRORs; the breakdown was
  (a) 43 rejection-sampling failures (fixed by region-tree AABB), (b)
  5 nuclide-cap overflows (fixed by MAX_NUC = 128), (c) 3 CLI-runner
  manifests (moved out of `bench/icsbep/`).

## Reproduce-from-cold

```powershell
# 0. Build the Python extension (one-time, ~1 min each).
cd rust_prototype/bindings/python
maturin develop --release                 # CPU only
maturin develop --release --features cuda # also enables Runner.GpuCuda

# 1. Smoke a single case to confirm the extension loads.
cd ../../..
python rust_prototype/bindings/python/examples/icsbep_run.py heu-met-fast-001_case-1 cpu

# 2. Sweep — production single-seed settings, all 374 scene cases.
#    Output: outputs/icsbep_full_<runner>.csv, durable per-case.
#    Stop gracefully from another shell: New-Item outputs/STOP -ItemType File
.\rust_prototype\bindings\python\examples\run_benchmark.ps1
#  (writes outputs/icsbep_full_gpu.csv when GPU build is present,
#   else outputs/icsbep_full_cpu.csv)

# 3. Resume after a stop:
python rust_prototype/bindings/python/examples/icsbep_sweep.py `
    --runner gpu --csv outputs/icsbep_full_gpu.csv `
    --stop-file outputs/STOP --resume

# 4. After the sweep finishes, commit the CSV + log:
git add outputs/icsbep_full_*.csv outputs/icsbep_full_*.log
git commit -m "icsbep: full sweep results (<machine>, <date>)"
```

`outputs/*` is gitignored by glob; results files need `git add -f`
if they should land in the repo, OR the user can keep them local-only
(the convention so far).
