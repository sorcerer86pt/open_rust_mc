# SPDX-License-Identifier: MIT
"""Run OpenMC under ENDF/B-VIII.1 over the ICSBEP scenes that LANL did NOT
include in Nobre et al. 2025 Table LIX, and stamp each scene JSON with a
`local_validation.viii1` block containing the OpenMC reference k_eff.

Why this exists
---------------

`outputs/refs/scene_to_viii1_ref.csv` maps each of our 375 scene JSONs to a
LANL Table LIX row when one exists. 289 cases match, 86 do not — the
"orphans". For those, we have no public VIII.1 reference; the only way to
grade engine quality without scene-transcription drift is to run OpenMC on
the same JSON our engine consumes, under VIII.1.

Pipeline
--------

For each orphan (csv: `outputs/refs/orphans_to_run.csv`):
  * skip if the scene JSON already has `local_validation.viii1`
  * shell out to `scripts/openmc_scene_runner.py` (which translates our
    JSON → OpenMC objects, handles S(α,β) since we extended it)
  * parse `{k_mean, sigma_seeds, per_seed[].time_s}` from the runner output
  * stamp `local_validation.viii1` into the JSON in place
  * append a row to `outputs/openmc_orphans_viii1.csv` and fsync

Designed to be killed and restarted: progress is durable case-by-case in
the JSONs themselves, and the CSV/log are append-only. Drop a marker file
`outputs/STOP_OPENMC` to exit between cases without losing the in-flight
one.

Expected runtime
----------------

Paper-grade defaults (20k × 100 batches × 3 seeds) give ~10-20 min/case
for thermal-spectrum scenes and ~3-5 min/case for fast metal. Total for
the 85 ICSBEP orphans: ~14-20h. Resumable, so spread across multiple
sittings is fine.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORPHAN_CSV = REPO_ROOT / "outputs" / "refs" / "orphans_to_run.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "outputs" / "openmc_orphans_viii1.csv"
DEFAULT_OUT_LOG = REPO_ROOT / "outputs" / "openmc_orphans_viii1.log"
DEFAULT_STOP_FILE = REPO_ROOT / "outputs" / "STOP_OPENMC"
DEFAULT_XS = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-viii.1-hdf5/cross_sections.xml"

# Cases to skip wholesale. B03AT3V17 is our internal PWR pin-cell mock
# (not an ICSBEP benchmark), uses lattices that openmc_scene_runner.py
# doesn't translate. Listed by file-stem to keep the matcher cheap.
SKIP_FILES = {"B03AT3V17.json"}


def load_orphans(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def already_done(scene_path: Path) -> bool:
    with open(scene_path, encoding="utf-8") as fh:
        scene = json.load(fh)
    # Convention in this repo: local_validation lives under `benchmark`.
    lv = scene.get("benchmark", {}).get("local_validation", {})
    return bool(lv and "viii1" in lv and lv["viii1"].get("openmc_k_eff") is not None)


def run_one(
    scene_path: Path,
    runner: Path,
    cross_sections: str,
    particles: int,
    batches: int,
    inactive: int,
    seeds: int,
    work_root: Path,
) -> dict:
    """Invoke openmc_scene_runner.py on a single scene and return the
    aggregate dict. The runner writes a per-case JSON we read back."""
    tmp_out = work_root / f"{scene_path.stem}.openmc_viii1.json"
    cmd = [
        sys.executable,
        str(runner),
        str(scene_path),
        str(tmp_out),
        "--particles", str(particles),
        "--batches", str(batches),
        "--inactive", str(inactive),
        "--seeds", str(seeds),
        "--cross-sections", cross_sections,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed (rc={proc.returncode}) for {scene_path.name}\n"
            f"stderr:\n{proc.stderr[-2000:]}\n"
            f"stdout tail:\n{proc.stdout[-1000:]}"
        )
    with open(tmp_out, encoding="utf-8") as fh:
        agg = json.load(fh)
    agg["wallclock_s"] = dt
    return agg


def stamp_scene(scene_path: Path, agg: dict, cross_sections: str, args: argparse.Namespace) -> None:
    """In-place update: add benchmark.local_validation.viii1 to the scene JSON.

    Nests under `benchmark` (existing convention in this repo) so the VII.1
    block in cases that already had one is preserved as a sibling key.
    """
    with open(scene_path, encoding="utf-8") as fh:
        scene = json.load(fh)
    bench = scene.setdefault("benchmark", {})
    lv = bench.setdefault("local_validation", {})
    lv["viii1"] = {
        "_what_this_is": (
            "OpenMC-on-this-same-JSON reference k_eff under ENDF/B-VIII.1. "
            "Filled by scripts/openmc_orphans_viii1.py because Nobre et al. "
            "2025 Table LIX does not include this benchmark case. Used as "
            "the secondary grading target so the engine isn't punished for "
            "library bias it inherits from VIII.1."
        ),
        "openmc_k_eff": agg["k_mean"],
        "openmc_sigma_seeds": agg["sigma_seeds"],
        "openmc_library_path": cross_sections,
        "openmc_run_date": time.strftime("%Y-%m-%d"),
        "openmc_seed_count": args.seeds,
        "openmc_particles_per_seed": args.particles,
        "openmc_batches": args.batches,
        "openmc_inactive": args.inactive,
        "openmc_wallclock_s": round(agg.get("wallclock_s", 0.0), 1),
    }
    # Atomic write + ensure_ascii=False so legacy fields with em-dash / σ / ±
    # in other blocks don't get re-encoded into \uXXXX escapes on rewrite.
    tmp = scene_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, scene_path)


def append_csv_row(out_csv: Path, row: dict) -> None:
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def log(line: str, log_path: Path) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{stamp}] {line}"
    print(msg, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orphans-csv", default=str(DEFAULT_ORPHAN_CSV))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--out-log", default=str(DEFAULT_OUT_LOG))
    ap.add_argument("--stop-file", default=str(DEFAULT_STOP_FILE))
    ap.add_argument("--cross-sections", default=DEFAULT_XS)
    ap.add_argument("--runner", default=str(REPO_ROOT / "scripts" / "openmc_scene_runner.py"))
    ap.add_argument("--particles", type=int, default=20_000)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--inactive", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--filter", default=None,
                    help="Substring filter on scene file basename.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many successful runs (0 = no limit).")
    ap.add_argument("--work-dir", default="/tmp/openmc_orphans",
                    help="Scratch dir for runner per-case outputs.")
    args = ap.parse_args()

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    orphans = load_orphans(Path(args.orphans_csv))
    runner = Path(args.runner)
    cross_sections = args.cross_sections
    stop_file = Path(args.stop_file)
    out_csv = Path(args.out_csv)
    out_log = Path(args.out_log)

    log(f"Loaded {len(orphans)} orphan rows from {args.orphans_csv}", out_log)
    log(f"Stats: {args.particles}p x {args.batches}b ({args.inactive} inactive) x {args.seeds} seeds", out_log)
    log(f"Cross-sections: {cross_sections}", out_log)

    done = 0
    skipped = 0
    failed = 0
    for o in orphans:
        if args.filter and args.filter not in o["file"]:
            continue
        if o["file"] in SKIP_FILES:
            log(f"SKIP {o['file']} (in SKIP_FILES — internal mock)", out_log)
            skipped += 1
            continue
        if stop_file.exists():
            log(f"STOP file present, exiting before {o['file']}", out_log)
            break
        if args.limit and done >= args.limit:
            log(f"--limit {args.limit} reached, exiting", out_log)
            break

        scene_path = REPO_ROOT / "bench" / "icsbep" / o["file"]
        if not scene_path.exists():
            log(f"MISS {o['file']} (file not found)", out_log)
            failed += 1
            continue
        if already_done(scene_path):
            log(f"DONE {o['file']} (local_validation.viii1 already set)", out_log)
            skipped += 1
            continue

        log(f"RUN  {o['file']} ({o['case_id']})", out_log)
        try:
            agg = run_one(
                scene_path, runner, cross_sections,
                args.particles, args.batches, args.inactive, args.seeds,
                work_root,
            )
            stamp_scene(scene_path, agg, cross_sections, args)
            k_ref = float(o.get("k_handbook", 0) or 0)
            delta_pcm = (agg["k_mean"] - k_ref) * 1.0e5 if k_ref else None
            append_csv_row(out_csv, {
                "file": o["file"],
                "case_id": o["case_id"],
                "k_openmc_viii1": f"{agg['k_mean']:.6f}",
                "sigma_seeds": f"{agg['sigma_seeds']:.6f}",
                "k_handbook": o.get("k_handbook", ""),
                "delta_handbook_pcm": f"{delta_pcm:+.1f}" if delta_pcm is not None else "",
                "wallclock_s": f"{agg.get('wallclock_s', 0.0):.1f}",
                "particles": args.particles,
                "batches": args.batches,
                "seeds": args.seeds,
            })
            log(
                f"OK   {o['file']}  k={agg['k_mean']:.5f} ± {agg['sigma_seeds']:.5f} "
                f"({agg.get('wallclock_s', 0):.0f}s)",
                out_log,
            )
            done += 1
        except Exception as exc:
            log(f"FAIL {o['file']}: {exc}", out_log)
            failed += 1

    log(f"FINISHED  ok={done} skipped={skipped} failed={failed}", out_log)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
