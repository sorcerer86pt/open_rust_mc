"""Sweep every ICSBEP benchmark JSON in `bench/icsbep/`.

Runs each `*.json` through `run_icsbep_case`, optionally averaged over
N seeds, with per-case settings either taken from the JSON's
`benchmark.recommended_settings` block (when present) or from the CLI
defaults. Appends one CSV row per case so a partial run is always
recoverable; supports `--resume` and a `--stop-file` for graceful
termination of multi-hour production runs.

Per-case settings precedence (highest → lowest):
    1. `benchmark.recommended_settings.{batches, inactive, particles, seeds}` in the JSON
    2. CLI flags (--batches, --inactive, --particles, --seeds)
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
    error: str = ""


CSV_COLUMNS = [
    "case", "status", "k_calc", "k_sigma", "k_ref", "sigma_exp",
    "delta_pcm", "bound_pcm", "sigma_ratio", "ref_source",
    "runtime_s", "n_seeds", "batches", "inactive", "particles", "error",
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
    p.add_argument("--batches", type=int, default=80,
                   help="CLI default; overridden by JSON benchmark.recommended_settings.batches")
    p.add_argument("--inactive", type=int, default=20,
                   help="CLI default; overridden by JSON recommended_settings.inactive")
    p.add_argument("--particles", type=int, default=5000,
                   help="CLI default; overridden by JSON recommended_settings.particles")
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds per case (mean ± seed-to-seed stderr); "
                        "CLI default, overridden by JSON recommended_settings.seeds")
    p.add_argument("--base-seed", type=int, default=42,
                   help="first seed; subsequent seeds are base, base+1, base+2, ...")
    p.add_argument("--rank", type=int, default=15, help="SVD rank")
    p.add_argument("--csv", type=Path, default=None, help="save results to CSV file (appended row-by-row)")
    p.add_argument("--resume", action="store_true",
                   help="skip cases already present in --csv (case names matched on the `case` column)")
    p.add_argument("--stop-file", type=Path, default=None,
                   help="if this file exists between cases, finish the current case and exit cleanly")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop on first FAIL or ERROR")
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
    """Per-case settings: JSON `benchmark.recommended_settings` overrides
    CLI args. Returns (Settings, n_seeds, batches, inactive, particles)
    so the row can record which numbers were actually used.

    Particle count uses a backend-aware override. CPU and GPU saturate
    at vastly different particle counts (CPU ~5k on an 8-thread laptop,
    3080 at 500k-1M per the saturation sweep) so a single number is
    always wrong for one of them. Schema (backward compatible):

        "recommended_settings": {
            "batches": 150,
            "inactive": 30,
            "particles": 20000,        # default / CPU sweet spot
            "particles_gpu": 500000,   # optional GPU override
            "seeds": 5
        }

    When `particles_gpu` is absent, the CPU value is used for both
    backends — same as today's behaviour. When present and the runner
    is GPU, it overrides the CPU `particles`.
    """
    rec: dict = {}
    try:
        with case_path.open("r", encoding="utf-8") as fp:
            j = json.load(fp)
        rec = j.get("benchmark", {}).get("recommended_settings", {}) or {}
    except Exception:
        rec = {}
    batches = int(rec.get("batches", args.batches))
    inactive = int(rec.get("inactive", args.inactive))
    particles_cpu = int(rec.get("particles", args.particles))
    particles_gpu = int(rec.get("particles_gpu", particles_cpu))
    if runner is not None and runner is Runner.GpuCuda:
        particles = particles_gpu
    else:
        particles = particles_cpu
    n_seeds = int(rec.get("seeds", args.seeds))
    settings = Settings(
        batches=batches,
        inactive=inactive,
        particles=particles,
        seed=args.base_seed,  # overwritten per-seed below
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
            # On error, abandon the remaining seeds for this case.
            break
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
        ),
        runtime,
    )


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    bench_dir = args.bench_dir or repo_root / "bench" / "icsbep"
    data_dir = args.data_dir or repo_root / "data" / "endfb-vii.1-hdf5" / "neutron"

    if not bench_dir.is_dir():
        print(f"bench dir not found: {bench_dir}", file=sys.stderr)
        return 2
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
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
    print(
        f"  CLI defaults: batches={args.batches}, inactive={args.inactive}, "
        f"particles={args.particles}, seeds={args.seeds}, base_seed={args.base_seed}, "
        f"rank={args.rank}"
    )
    print("  per-case settings: JSON `benchmark.recommended_settings` overrides CLI defaults")

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

    rows: list[Row] = []
    sweep_t0 = time.time()
    aborted = False

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

    # Graceful stop (Ctrl-C or stop-file) returns 0 — the partial CSV
    # is durable and `--resume` will pick up where we left off. Non-zero
    # is reserved for "ran to completion but some cases FAILed / ERRORed".
    if aborted:
        return 0
    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
