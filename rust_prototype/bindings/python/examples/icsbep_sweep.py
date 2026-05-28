# SPDX-License-Identifier: MIT
"""Sweep every ICSBEP benchmark JSON in `bench/icsbep/`.

Runs each `*.json` through `run_icsbep_case`, optionally averaged over
N seeds. Per-case settings are taken from the JSON's
`benchmark.recommended_settings` block when present, but explicit CLI
flags always win — pass `--particles 20000` to force a light A1000
re-run regardless of what the JSON recommends for a 3080-class box.
Appends one CSV row per case so a partial run is always recoverable;
supports `--resume` and a `--stop-file` for graceful termination of
multi-hour production runs.

Per-case settings precedence (highest first):
    1. Explicit CLI flag (--batches, --inactive, --particles, --seeds)
    2. JSON `benchmark.recommended_settings.{batches,inactive,particles[,particles_gpu],seeds}`
    3. Built-in defaults (80, 20, 5000, 1)

The schema for the JSON override:
    {
      "benchmark": {
        ...,
        "recommended_settings": {
          "batches": 150,
          "inactive": 30,
          "particles": 20000,
          "seeds": 3
        }
      }
    }

Usage
-----
    # Light smoke (all cases, CPU, CLI-default cheap settings)
    python icsbep_sweep.py

    # Production single-seed GPU sweep (~3 h):
    python icsbep_sweep.py --runner gpu --batches 80 --inactive 20 --particles 5000

    # Paper-quality multi-seed GPU sweep (~12-15 h):
    python icsbep_sweep.py --runner gpu --batches 150 --inactive 30 `
        --particles 20000 --seeds 3 --csv outputs/icsbep_paper_gpu.csv `
        --stop-file outputs/STOP

    # Resume an interrupted run: skip cases already in the CSV
    python icsbep_sweep.py --csv outputs/icsbep_paper_gpu.csv --resume

    # Graceful stop from another shell:
    #   PowerShell: New-Item outputs\\STOP -ItemType File
    #   bash:       touch outputs/STOP
    # (SIGINT / Ctrl-C also flushes the partial CSV before exit.)

Multi-seed semantics
--------------------
Per case, the script runs `n_seeds` independent simulations with
consecutive seeds (`base_seed`, `base_seed + 1`, ...). The reported
k_calc is the mean over seeds; k_sigma is the seed-to-seed stderr
(sqrt(variance / n_seeds)). This is what tests/cuda_runs.rs uses for
ICSBEP regression and is more conservative than the within-batch
stderr a single run reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from collections import Counter

from open_rust_mc import (
    Runner,
    Settings,
    preload_nuclide_cache_weights,
    run_icsbep_case,
    gpu_debug_metrics,
)


def _walk_nuclide_weights(case_paths):
    """Walk every case JSON and tally `(zaid, temperature_k)` →
    appearance count across the corpus. Pre-warms the L1 nuclide
    cache so actinides + structurals stay resident even when a
    rare-nuclide case lands mid-sweep.

    Cases without a `scene` block (CLI-runner manifests) are
    silently skipped — they don't drive `run_icsbep_case` anyway.
    """
    counts = Counter()
    for case_path in case_paths:
        try:
            with open(case_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        scene = doc.get("scene")
        if not scene:
            continue
        for mat in scene.get("materials", []):
            temp_k = float(mat.get("temperature", 294.0))
            for nuc in mat.get("nuclides", []):
                zaid = nuc.get("zaid")
                if isinstance(zaid, int):
                    counts[(zaid, temp_k)] += 1
    return counts


@dataclass
class Row:
    case: str
    status: str  # "PASS", "FAIL", "ERROR"
    k_calc: float | None
    k_sigma: float | None
    k_ref: float | None
    sigma_exp: float | None
    delta_pcm: float | None
    bound_pcm: float | None
    sigma_ratio: float | None
    ref_source: str
    runtime_s: float
    n_seeds: int = 1
    batches: int = 0
    inactive: int = 0
    particles: int = 0
    gpu_refill_pool_factor: float | None = None
    gpu_auto_refill: bool = False
    error: str = ""


CSV_COLUMNS = [
    "case", "status", "k_calc", "k_sigma", "k_ref", "sigma_exp",
    "delta_pcm", "bound_pcm", "sigma_ratio", "ref_source",
    "runtime_s", "n_seeds", "batches", "inactive", "particles",
    "gpu_refill_pool_factor", "gpu_auto_refill", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", type=Path, default=None,
                   help="bench/icsbep directory (auto-discovered if omitted)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="ENDF HDF5 neutron directory (auto-discovered if omitted)")
    p.add_argument("--filter", type=str, default=None,
                   help="regex pattern; only case stems matching this are run")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of cases run after filtering")
    p.add_argument("--runner", choices=["cpu", "gpu"], default="cpu",
                   help="execution backend (default: cpu)")
    # Precedence (highest → lowest): explicit CLI flag > JSON
    # `benchmark.recommended_settings.<key>` > built-in default below.
    # CLI args default to ``None`` as a sentinel for "not passed"; the
    # built-in fallback is applied at use-time so it doesn't swallow
    # the JSON recommendation when the user didn't override.
    p.add_argument("--batches", type=int, default=None,
                   help="active+inactive batches per seed. CLI flag wins over "
                        "JSON benchmark.recommended_settings.batches; built-in default 80.")
    p.add_argument("--inactive", type=int, default=None,
                   help="inactive batches per seed. CLI wins over JSON; "
                        "built-in default 20.")
    p.add_argument("--particles", type=int, default=None,
                   help="particles per batch. CLI wins over JSON "
                        "(both `particles` and `particles_gpu`); built-in default 5000.")
    p.add_argument("--seeds", type=int, default=None,
                   help="number of seeds per case (mean ± seed-to-seed stderr). "
                        "CLI wins over JSON; built-in default 1.")
    p.add_argument("--base-seed", type=int, default=42,
                   help="first seed; subsequent seeds are base, base+1, base+2, ...")
    p.add_argument("--rank", type=int, default=15, help="SVD rank")
    p.add_argument("--gpu-refill-factor", type=float, default=None,
                   help="PHYSOR 2022 Optimization F — explicit refill pool factor "
                        "(e.g. 2.0 = source bank is 2x particles per batch). "
                        "GPU runner only; CPU ignores. Wins over --gpu-auto-refill.")
    p.add_argument("--gpu-auto-refill", action="store_true",
                   help="GPU runner only: let the engine pick a refill factor "
                        "automatically from device SM count + kernel reg count. "
                        "Ignored if --gpu-refill-factor is set or if runner=cpu.")
    # Survival biasing — default ON. Implicit capture +
    # Bernoulli-banked fission + Russian roulette terminates
    # long-tail histories that otherwise spin the GPU event loop to
    # `max_events_per_history = 1_000_000` on high-particle-count
    # runs (the 200k 3080 symptom). k_eff stays unbiased; σ tightens
    # ~10-15%. Pass `--no-survival-bias` for analog A/B work.
    sb = p.add_mutually_exclusive_group()
    sb.add_argument("--survival-bias", dest="survival_bias",
                    action="store_true", default=True,
                    help="enable implicit capture + RR (default)")
    sb.add_argument("--no-survival-bias", dest="survival_bias",
                    action="store_false",
                    help="opt out — pure analog tracking")
    p.add_argument("--csv", type=Path, default=None, help="save results to CSV file (appended row-by-row)")
    p.add_argument("--resume", action="store_true",
                   help="skip cases already present in --csv (case names matched on the `case` column)")
    p.add_argument("--stop-file", type=Path, default=None,
                   help="if this file exists between cases, finish the current case and exit cleanly")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop on first FAIL or ERROR")
    p.add_argument("--debug-metrics", type=Path, default=None,
                   help="path to write one JSON line per case with "
                        "GPU cache + arch + budget snapshots. Use the "
                        "delta in `sab_cache_hits` / `nuc_cache_hits` "
                        "across consecutive cases to see how many "
                        "uploads were avoided. CPU runs still write "
                        "rows but the GPU fields are null. The file is "
                        "opened append; pass a fresh path or rm it "
                        "between runs to avoid mixing telemetry.")
    return p.parse_args()


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "bench" / "icsbep").is_dir():
            return p
    raise SystemExit(f"could not locate bench/icsbep relative to {start}")


def read_completed_cases(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done: set[str] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            r = csv.DictReader(fp)
            for row in r:
                if "case" in row and row["case"]:
                    done.add(row["case"])
    except Exception as e:  # noqa: BLE001
        print(f"warning: failed to read {csv_path} for resume: {e}", file=sys.stderr)
    return done


def open_csv_for_append(csv_path: Path) -> tuple[object, csv.DictWriter]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    fp = csv_path.open("a", encoding="utf-8", newline="")
    w = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
    if is_new:
        w.writeheader()
        fp.flush()
    return fp, w


def write_row(writer: csv.DictWriter, fp, row: Row) -> None:
    d = asdict(row)
    out = {}
    for col in CSV_COLUMNS:
        v = d.get(col, "")
        if v is None:
            out[col] = ""
        elif isinstance(v, float):
            if col in ("k_calc", "k_sigma", "k_ref", "sigma_exp"):
                out[col] = f"{v:.6f}"
            elif col == "sigma_ratio":
                out[col] = f"{v:.3f}"
            elif col in ("delta_pcm", "bound_pcm"):
                out[col] = f"{v:.1f}"
            elif col == "runtime_s":
                out[col] = f"{v:.2f}"
            else:
                out[col] = f"{v}"
        else:
            out[col] = str(v)
    writer.writerow(out)
    fp.flush()


def case_settings(
    case_path: Path,
    args: argparse.Namespace,
    runner: Runner | None = None,
) -> tuple[Settings, int, int, int, int]:
    """Per-case settings. Precedence per knob (highest first):

        1. explicit CLI flag (e.g. ``--particles 20000``)
        2. JSON ``benchmark.recommended_settings.<key>``
        3. built-in default

    Returns (Settings, n_seeds, batches, inactive, particles) so the
    CSV row can record which numbers were actually used.

    Particle count uses a backend-aware fallback. CPU and GPU saturate
    at vastly different particle counts (CPU ~5k on an 8-thread laptop,
    3080 at 500k-1M per the saturation sweep) so a single number is
    always wrong for one of them. JSON schema (backward compatible):

        "recommended_settings": {
            "batches": 150,
            "inactive": 30,
            "particles": 20000,        # default / CPU sweet spot
            "particles_gpu": 500000,   # optional GPU override
            "seeds": 5
        }

    When ``particles_gpu`` is absent, the CPU value is used for both
    backends. When present and the runner is GPU, it overrides the
    CPU ``particles`` — but only when the CLI didn't pin
    ``--particles`` explicitly.
    """
    rec: dict = {}
    try:
        with case_path.open("r", encoding="utf-8") as fp:
            j = json.load(fp)
        rec = j.get("benchmark", {}).get("recommended_settings", {}) or {}
    except Exception:
        rec = {}

    def _pick(cli_val, json_key: str, builtin: int) -> int:
        """CLI > JSON > built-in default."""
        if cli_val is not None:
            return int(cli_val)
        if json_key in rec:
            return int(rec[json_key])
        return builtin

    batches = _pick(args.batches, "batches", 80)
    inactive = _pick(args.inactive, "inactive", 20)
    n_seeds = _pick(args.seeds, "seeds", 1)

    # Particles: explicit `--particles` always wins for both backends.
    # When not passed, GPU runs prefer JSON `particles_gpu` then fall
    # back to `particles` then to the built-in default; CPU goes
    # straight to `particles` → built-in.
    if args.particles is not None:
        particles = int(args.particles)
    else:
        particles_cpu = int(rec.get("particles", 5000))
        particles_gpu = int(rec.get("particles_gpu", particles_cpu))
        particles = particles_gpu if (runner is not None and runner is Runner.GpuCuda) else particles_cpu
    # Refill knobs are GPU-only. CPU runner silently ignores both
    # (kernel doesn't exist; SimConfig fields just sit unused on
    # the rayon path). Explicit `--gpu-refill-factor` wins over
    # `--gpu-auto-refill`; the engine's CudaRunner::run announces
    # what got picked when auto fires.
    #
    # Per-case JSON keys (both optional):
    #   "gpu_refill_pool_factor": 2.0   # explicit factor
    #   "gpu_auto_refill": true         # let engine pick
    # Precedence: explicit CLI flag > JSON > unset. The CLI sentinel
    # for `--gpu-refill-factor` is `None` (default), and
    # `--gpu-auto-refill` only flips the auto-refill bit when the
    # flag is present.
    is_gpu = runner is not None and runner is Runner.GpuCuda
    if not is_gpu:
        refill_factor = None
        auto_refill = False
    else:
        # Explicit CLI refill factor wins.
        if args.gpu_refill_factor is not None:
            refill_factor = float(args.gpu_refill_factor)
        elif "gpu_refill_pool_factor" in rec:
            refill_factor = float(rec["gpu_refill_pool_factor"])
        else:
            refill_factor = None
        # `store_true` flags can't distinguish "user passed --flag" from
        # "default False", so the CLI overrides JSON only when set.
        if args.gpu_auto_refill:
            auto_refill = True
        else:
            auto_refill = bool(rec.get("gpu_auto_refill", False))
    # OpenMC defaults: w_min=0.25, w_survive=1.0. CPU + GPU both
    # read the tuple directly from PySettings.survival_biasing.
    sb_tuple = (0.25, 1.0) if getattr(args, "survival_bias", False) else None
    settings = Settings(
        batches=batches,
        inactive=inactive,
        particles=particles,
        seed=args.base_seed,  # overwritten per-seed below
        gpu_refill_pool_factor=refill_factor,
        gpu_auto_refill=auto_refill,
        survival_biasing=sb_tuple,
    )
    return settings, n_seeds, batches, inactive, particles


def run_case_multi_seed(
    case_path: Path,
    data_dir: Path,
    base_settings: Settings,
    runner: Runner,
    rank: int,
    n_seeds: int,
    base_seed: int,
) -> tuple[Row, float]:
    """Run one case across N seeds; return aggregated Row + total wall time.
    Aggregation matches tests/cuda_runs.rs::run_case_cuda_seeds: per-seed
    k values averaged, σ = sqrt(seed_to_seed_variance / n_seeds). The
    pass envelope is recomputed from the aggregated σ."""
    assert n_seeds >= 1
    t0 = time.time()
    seed_ks: list[float] = []
    seed_ksigmas: list[float] = []
    k_ref = 0.0
    sigma_exp = 0.0
    ref_source = ""
    case_label = case_path.stem
    last_error: str | None = None
    for s in range(n_seeds):
        seed = base_seed + s
        settings = Settings(
            batches=base_settings.batches,
            inactive=base_settings.inactive,
            particles=base_settings.particles,
            seed=seed,
            gpu_refill_pool_factor=base_settings.gpu_refill_pool_factor,
            gpu_auto_refill=base_settings.gpu_auto_refill,
            survival_biasing=base_settings.survival_biasing,
        )
        # Per-seed banner: long cases (≥400s/seed) used to sit silent
        # between the pre-case banner and the post-case summary. With
        # this, the user sees exactly which seed is running. The Rust
        # eigenvalue ProgressBar then renders the per-batch progress
        # inside `run_icsbep_case` when stderr is a TTY.
        seed_t0 = time.time()
        print(
            f"  {case_path.stem} -- seed {s + 1}/{n_seeds} (seed={seed}) starting",
            flush=True,
        )
        try:
            r = run_icsbep_case(
                case_json=case_path,
                data_dir=data_dir,
                settings=settings,
                runner=runner,
                rank=rank,
            )
        except Exception as e:  # noqa: BLE001
            last_error = str(e).splitlines()[0][:200]
            print(
                f"  {case_path.stem} -- seed {s + 1}/{n_seeds} ERROR after "
                f"{time.time() - seed_t0:.1f}s: {last_error}",
                flush=True,
            )
            # On error, abandon the remaining seeds for this case.
            break
        print(
            f"  {case_path.stem} -- seed {s + 1}/{n_seeds} done in "
            f"{time.time() - seed_t0:.1f}s: k={r.k_eff:.5f} +/- {r.k_sigma:.5f}",
            flush=True,
        )
        seed_ks.append(r.k_eff)
        seed_ksigmas.append(r.k_sigma)
        case_label = r.case
        k_ref = r.k_ref
        sigma_exp = r.sigma_exp
        ref_source = r.ref_source

    runtime = time.time() - t0

    if last_error is not None or not seed_ks:
        return (
            Row(
                case=case_label,
                status="ERROR",
                k_calc=None, k_sigma=None, k_ref=None, sigma_exp=None,
                delta_pcm=None, bound_pcm=None, sigma_ratio=None,
                ref_source="",
                runtime_s=runtime,
                n_seeds=n_seeds,
                batches=base_settings.batches,
                inactive=base_settings.inactive,
                particles=base_settings.particles,
                gpu_refill_pool_factor=base_settings.gpu_refill_pool_factor,
                gpu_auto_refill=base_settings.gpu_auto_refill,
                error=last_error or "no seed produced a result",
            ),
            runtime,
        )

    # Multi-seed aggregation: mean across seeds, σ_mean = stderr of mean.
    n = len(seed_ks)
    mean = sum(seed_ks) / n
    if n > 1:
        var = sum((k - mean) ** 2 for k in seed_ks) / (n - 1)
        sigma_seed_stderr = math.sqrt(var / n)
    else:
        # Single-seed: fall back to the engine's within-batch stderr
        # so the bound is well-defined.
        sigma_seed_stderr = seed_ksigmas[0]

    sigma_combined = math.sqrt(sigma_seed_stderr * sigma_seed_stderr + sigma_exp * sigma_exp)
    delta = mean - k_ref
    delta_pcm = delta * 1.0e5
    bound_pcm = max(150.0, 2.0 * sigma_combined * 1.0e5)
    sigma_ratio = abs(delta) / sigma_combined if sigma_combined > 0 else 0.0
    passed = abs(delta_pcm) <= bound_pcm

    return (
        Row(
            case=case_label,
            status="PASS" if passed else "FAIL",
            k_calc=mean,
            k_sigma=sigma_seed_stderr,
            k_ref=k_ref,
            sigma_exp=sigma_exp,
            delta_pcm=delta_pcm,
            bound_pcm=bound_pcm,
            sigma_ratio=sigma_ratio,
            ref_source=ref_source,
            runtime_s=runtime,
            n_seeds=n,
            batches=base_settings.batches,
            inactive=base_settings.inactive,
            particles=base_settings.particles,
            gpu_refill_pool_factor=base_settings.gpu_refill_pool_factor,
            gpu_auto_refill=base_settings.gpu_auto_refill,
        ),
        runtime,
    )


def _find_data_dir(repo_root: Path) -> Path | None:
    # VIII.1 is the current default download
    # (`scripts/setup_nuclear_data.ps1`). Fall back to older libraries
    # only when VIII.1 isn't installed, so partial installs keep
    # working without flag tweaks.
    for lib in ("endfb-viii.1-hdf5", "endfb-viii.0-hdf5", "endfb-vii.1-hdf5"):
        candidate = repo_root / "data" / lib / "neutron"
        if candidate.is_dir():
            return candidate
    return None


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    bench_dir = args.bench_dir or repo_root / "bench" / "icsbep"
    data_dir = args.data_dir or _find_data_dir(repo_root)

    if not bench_dir.is_dir():
        print(f"bench dir not found: {bench_dir}", file=sys.stderr)
        return 2
    if data_dir is None or not data_dir.is_dir():
        print(
            f"data dir not found (looked under {repo_root / 'data'} for "
            "endfb-viii.1-hdf5 / endfb-viii.0-hdf5 / endfb-vii.1-hdf5)",
            file=sys.stderr,
        )
        return 2

    runner = Runner.GpuCuda if args.runner == "gpu" else Runner.Cpu
    pattern = re.compile(args.filter) if args.filter else None

    cases = sorted(bench_dir.glob("*.json"))
    if pattern:
        cases = [c for c in cases if pattern.search(c.stem)]
    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        print("no cases match the filter", file=sys.stderr)
        return 2

    completed: set[str] = set()
    if args.resume:
        if args.csv is None:
            print("--resume requires --csv to know what's already done", file=sys.stderr)
            return 2
        completed = read_completed_cases(args.csv)
        if completed:
            print(f"  resume: {len(completed)} case(s) already in {args.csv}, skipping those")
    cases = [c for c in cases if c.stem not in completed]
    if not cases:
        print("nothing to do — all cases already completed in CSV")
        return 0

    stop_file = args.stop_file
    stop_requested = {"flag": False}

    def _signal_stop(signum, _frame):
        stop_requested["flag"] = True

    try:
        signal.signal(signal.SIGINT, _signal_stop)
        signal.signal(signal.SIGTERM, _signal_stop)
    except (ValueError, AttributeError):
        pass

    print(f"Sweeping {len(cases)} case(s) on {args.runner.upper()} runner")
    # ``None`` here means "no explicit CLI flag passed" — display "auto"
    # so the user can see at a glance which knobs will fall through to
    # the JSON recommended_settings vs the built-in defaults.
    def _show(v, builtin):
        return f"{v} (explicit)" if v is not None else f"auto (JSON or {builtin})"
    print(
        f"  CLI overrides: batches={_show(args.batches, 80)}, "
        f"inactive={_show(args.inactive, 20)}, "
        f"particles={_show(args.particles, 5000)}, "
        f"seeds={_show(args.seeds, 1)}, "
        f"base_seed={args.base_seed}, rank={args.rank}"
    )
    sb_mode = "ON (default — implicit capture + RR)" if args.survival_bias else "OFF (analog, opt-out)"
    print(f"  Survival bias: {sb_mode}")
    print("  per-case settings: explicit CLI flag > JSON `benchmark.recommended_settings` > built-in default")
    # Backend-equivalence note: GPU has a refill-pool knob to keep SM
    # lanes filled during the batch tail (PHYSOR 2022 Optimization F).
    # CPU runs don't need it — rayon's work-stealing already grabs a
    # new history the instant any thread's current one dies, so the
    # batch-tail concern doesn't exist. JSONs that set
    # `gpu_auto_refill` / `gpu_refill_pool_factor` flow through to the
    # CSV column for traceability but have no CPU-side effect. State
    # it explicitly so a CPU-vs-GPU A/B reader knows the column isn't
    # silently broken.
    if args.runner == "cpu":
        print(
            "  CPU runner: any `gpu_auto_refill` / `gpu_refill_pool_factor` in JSON "
            "is ignored (no analog needed; rayon work-stealing saturates cores)."
        )

    # ── L1 nuclide-cache warm-start ────────────────────────────────
    # Walk the manifest once, count nuclide appearances, hand the
    # histogram to the engine. U-235 / O-16 / Fe-56 / U-238 land with
    # high preload weight; rare dosimetry nuclides start cold but
    # gain hits as cases visit them. Eviction picks losers by
    # (hits + preload) score under the LFU-with-recency policy.
    pre_t0 = time.time()
    nuc_counts = _walk_nuclide_weights(cases)
    if nuc_counts:
        weights = [
            (zaid, temp_k, count)
            for (zaid, temp_k), count in nuc_counts.items()
        ]
        n_loaded = preload_nuclide_cache_weights(
            data_dir=data_dir, weights=weights, rank=args.rank
        )
        top = sorted(nuc_counts.items(), key=lambda kv: -kv[1])[:5]
        print(
            f"  preload: {n_loaded}/{len(weights)} nuclide weights resolved "
            f"({time.time() - pre_t0:.1f}s). Top 5: "
            + ", ".join(f"Z={z}@{t:.0f}K*{c}" for (z, t), c in top)
        )
    if args.csv:
        print(f"  CSV (append, flushed per case): {args.csv}")
    if stop_file:
        print(f"  stop-file (create to terminate gracefully): {stop_file}")
    print()

    csv_fp = None
    csv_writer = None
    if args.csv:
        csv_fp, csv_writer = open_csv_for_append(args.csv)

    # Optional per-case observability dump. One JSON line per snapshot
    # (one "init" line before the loop + one "after_case" line per
    # case) — downstream tools can diff consecutive lines to see how
    # many SAB / nuclide cache hits each case took.
    debug_fp = None
    if args.debug_metrics is not None:
        args.debug_metrics.parent.mkdir(parents=True, exist_ok=True)
        debug_fp = open(args.debug_metrics, "a", encoding="utf-8")

    def _emit_debug(phase: str, case: str | None, wall_t: float, extra: dict | None = None) -> None:
        if debug_fp is None:
            return
        snap = gpu_debug_metrics() or {}
        record = {
            "ts": time.time(),
            "phase": phase,
            "case": case,
            "wall_t": wall_t,
            "metrics": snap,
        }
        if extra:
            record.update(extra)
        debug_fp.write(json.dumps(record) + "\n")
        debug_fp.flush()

    rows: list[Row] = []
    sweep_t0 = time.time()
    aborted = False
    # Initial-state snapshot — caches start empty; gpu_arch records
    # the NVRTC arch the kernels were just compiled against (from the
    # preload pass that ran above).
    _emit_debug(
        phase="init",
        case=None,
        wall_t=0.0,
        extra={"runner": args.runner, "n_cases": len(cases)},
    )

    try:
        for idx, case_path in enumerate(cases, 1):
            if stop_requested["flag"]:
                print(f"\n  stop requested (signal); exiting after {idx - 1} case(s).")
                aborted = True
                break
            if stop_file and stop_file.exists():
                print(f"\n  stop file {stop_file} detected; exiting after {idx - 1} case(s).")
                aborted = True
                break

            base_settings, n_seeds, batches, inactive, particles = case_settings(case_path, args, runner)
            # Per-case banner. Combined with the per-seed banner in
            # `run_case_multi_seed` and the Rust eigenvalue ProgressBar
            # inside `run_icsbep_case`, the user always sees where the
            # sweep is — even on cases that take 400+ s/seed.
            print(
                f"[{idx}/{len(cases)}] {case_path.stem} starting "
                f"({n_seeds}seed x {batches}b x {inactive}i x {particles}p)",
                flush=True,
            )
            row, _ = run_case_multi_seed(
                case_path=case_path,
                data_dir=data_dir,
                base_settings=base_settings,
                runner=runner,
                rank=args.rank,
                n_seeds=n_seeds,
                base_seed=args.base_seed,
            )
            rows.append(row)

            _emit_debug(
                phase="after_case",
                case=row.case,
                wall_t=row.runtime_s,
                extra={"status": row.status, "k_calc": row.k_calc, "k_sigma": row.k_sigma},
            )

            if csv_writer is not None:
                write_row(csv_writer, csv_fp, row)

            settings_tag = (
                f"({n_seeds}seed x {batches}b x {inactive}i x {particles}p)"
                if n_seeds > 1
                else f"({batches}b x {inactive}i x {particles}p)"
            )

            if row.status == "ERROR":
                print(
                    f"{row.case}: ERROR -- {row.error} ({row.runtime_s:.1f}s) {settings_tag}",
                    flush=True,
                )
            else:
                done_total = len(completed) + idx
                grand_total = len(completed) + len(cases)
                print(
                    f"{row.case}: {row.status} -- "
                    f"k={row.k_calc:.5f}+/-{row.k_sigma:.5f}, "
                    f"delta={row.delta_pcm:+.0f}pcm, "
                    f"bound=+/-{row.bound_pcm:.0f}pcm, "
                    f"{row.sigma_ratio:.2f}sigma, "
                    f"{row.runtime_s:.1f}s "
                    f"{settings_tag} "
                    f"[{done_total}/{grand_total}]",
                    flush=True,
                )

            if args.fail_fast and row.status != "PASS":
                print(f"\nfail-fast: stopping on {row.status} at case {row.case}")
                aborted = True
                break

    finally:
        if csv_fp is not None:
            csv_fp.close()
        if debug_fp is not None:
            debug_fp.close()

    sweep_dt = time.time() - sweep_t0

    n_pass = sum(1 for r in rows if r.status == "PASS")
    n_fail = sum(1 for r in rows if r.status == "FAIL")
    n_err = sum(1 for r in rows if r.status == "ERROR")
    print()
    suffix = " (aborted early)" if aborted else ""
    print(f"  Sweep this session in {sweep_dt:.1f} s ({sweep_dt/60:.1f} min){suffix}")
    print(f"  Result: {n_pass} PASS  |  {n_fail} FAIL  |  {n_err} ERROR  ({len(rows)} ran this session)")
    if completed:
        print(f"  + {len(completed)} cases carried over from prior session(s) in {args.csv}")

    if n_fail or n_err:
        print()
        print("  Non-passing cases this session:")
        for r in rows:
            if r.status == "PASS":
                continue
            if r.status == "FAIL":
                print(
                    f"    FAIL  {r.case:<40} delta={r.delta_pcm:+6.0f} pcm  "
                    f"bound=+/-{r.bound_pcm:5.0f}  k_calc={r.k_calc:.5f}  k_ref={r.k_ref:.5f}"
                )
            else:
                print(f"    ERROR {r.case:<40} {r.error}")

    if args.csv:
        print(f"\n  CSV: {args.csv}")
        # Fire-and-forget the delta-k vs EALF plotter as a separate
        # process. Decoupled from the sweep:
        #   - matplotlib import / draw failures never affect the
        #     sweep's exit code (sweep returns based on PASS/FAIL only)
        #   - non-blocking: the sweep returns immediately while the
        #     plot child writes its PNG in parallel
        #   - the PNG lands at outputs/icsbep/plots/<csv_stem>_delta_k_vs_ealf.png
        #     so the dashboard / publish step can grab it by glob
        # The plot script is itself idempotent; rerunning the sweep
        # overwrites the PNG with fresh data.
        import subprocess
        plot_script = Path(__file__).resolve().parents[4] / "scripts" / "plot_delta_k_vs_ealf.py"
        plot_dir = Path("outputs") / "icsbep" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_png = plot_dir / f"{args.csv.stem}_delta_k_vs_ealf.png"
        try:
            subprocess.Popen(
                [sys.executable, str(plot_script), str(args.csv), "--output", str(plot_png)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"  Plot: {plot_png}  (generating in background)")
        except OSError as e:  # plot script missing, python missing, etc.
            print(f"  Plot generation skipped: {e}", file=sys.stderr)

    # Graceful stop (Ctrl-C or stop-file) returns 0 — the partial CSV
    # is durable and `--resume` will pick up where we left off. Non-zero
    # is reserved for "ran to completion but some cases FAILed / ERRORed".
    if aborted:
        return 0
    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
