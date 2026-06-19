# SPDX-License-Identifier: MIT
"""Re-grade an ICSBEP sweep CSV against the ENDF/B-VIII.1 acceptance target.

The sweep harness records the engine's `k_calc` per case. Older sweeps (and
sweeps run before the PyO3 binding learned the VIII.1 priority chain) graded
`k_calc` against the *experimental* handbook `k_eff_reference` — even when the
library in use (VIII.1) is known to shift that benchmark away from 1.0. That
manufactured false FAILs (e.g. graphite-reflected HEU-MET-FAST-019, which
VIII.1 itself predicts at k=1.0055, not 1.0000).

This script re-grades each recorded `k_calc` against the correct VIII.1 target,
resolved from the scene JSON with the SAME priority the engine now uses
(`bindings/python/src/lib.rs::run_icsbep_case` /
`tests/cuda_runs.rs::resolve_acceptance_target`):

    1. local_validation.viii1.lanl_k_eff   — LANL MCNP under VIII.1 (Table LIX)
    2. local_validation.viii1.openmc_k_eff — our OpenMC on this JSON under VIII.1
    3. local_validation.openmc_k_eff       — legacy VII.1 OpenMC block
    4. benchmark.k_eff_reference           — handbook experimental k

σ for tiers 1-3 is max(σ_pub, handbook_sigma) so the envelope never under-states
uncertainty. The pass envelope matches the harness: |Δ| ≤ max(150 pcm, 2σ_comb).

It does NOT re-run the engine; `k_calc` is taken verbatim from the input CSV.

Usage:
    python scripts/regrade_sweep_viii1.py \
        outputs/icsbep_full_gpu.csv outputs/icsbep_full_gpu_viii1graded.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "bench" / "icsbep"

# Clean column order for the output CSV (matches the current harness writer;
# the legacy input CSV has trailing spaces and a "dekta_pcm" typo we normalize).
OUT_COLUMNS = [
    "case", "status", "k_calc", "k_sigma", "k_ref", "sigma_exp",
    "delta_pcm", "bound_pcm", "sigma_ratio", "ref_source",
    "runtime_s", "n_seeds", "batches", "inactive", "particles",
    "gpu_refill_pool_factor", "gpu_auto_refill", "error_str",
]


def resolve_target(
    benchmark: dict, handbook_k: float, handbook_sigma: float
) -> tuple[float, float, str]:
    """Mirror of the engine's acceptance-target resolution (see module docstring)."""
    lv = benchmark.get("local_validation")
    if not lv:
        return handbook_k, handbook_sigma, "k_eff_reference (ICSBEP handbook)"
    viii1 = lv.get("viii1") or {}
    lanl = viii1.get("lanl_k_eff")
    if lanl is not None:
        s = viii1.get("lanl_sigma") or 0.0
        return lanl, max(s, handbook_sigma), "local_validation.viii1 (LANL Table LIX)"
    viii1_omc = viii1.get("openmc_k_eff")
    if viii1_omc is not None:
        s = viii1.get("openmc_sigma_seeds") or 0.0
        return viii1_omc, max(s, handbook_sigma), "local_validation.viii1 (OpenMC on this scene)"
    legacy = lv.get("openmc_k_eff")
    if legacy is not None:
        s = lv.get("openmc_k_sigma_seeds") or 0.001
        return legacy, max(s, handbook_sigma), "local_validation (legacy OpenMC)"
    return handbook_k, handbook_sigma, "k_eff_reference (ICSBEP handbook)"


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    """Strip whitespace from keys and map the legacy `dekta_pcm` typo."""
    clean = {k.strip(): v for k, v in row.items()}
    if "dekta_pcm" in clean and "delta_pcm" not in clean:
        clean["delta_pcm"] = clean.pop("dekta_pcm")
    return clean


def regrade(in_csv: Path, out_csv: Path) -> None:
    rows = [normalize_row(r) for r in csv.DictReader(in_csv.open())]
    changes: list[tuple[str, str, str, float]] = []
    out_rows: list[dict[str, str]] = []

    for r in rows:
        case = r["case"]
        old_status = r["status"]
        if old_status == "ERROR" or not r.get("k_calc"):
            out_rows.append({c: r.get(c, "") for c in OUT_COLUMNS})
            continue

        scene_path = BENCH_DIR / f"{case}.json"
        if not scene_path.exists():
            raise FileNotFoundError(f"scene JSON missing for case {case}: {scene_path}")
        benchmark = json.loads(scene_path.read_text(encoding="utf-8"))["benchmark"]
        handbook_k = benchmark["k_eff_reference"]
        handbook_sigma = benchmark["k_eff_sigma"]
        k_ref, sigma_exp, ref_source = resolve_target(benchmark, handbook_k, handbook_sigma)

        k_calc = float(r["k_calc"])
        k_sigma = float(r["k_sigma"])
        sigma_combined = math.sqrt(k_sigma * k_sigma + sigma_exp * sigma_exp)
        delta_pcm = (k_calc - k_ref) * 1.0e5
        bound_pcm = max(150.0, 2.0 * sigma_combined * 1.0e5)
        sigma_ratio = abs(k_calc - k_ref) / sigma_combined if sigma_combined > 0 else 0.0
        status = "PASS" if abs(delta_pcm) <= bound_pcm else "FAIL"

        if status != old_status:
            changes.append((case, old_status, status, delta_pcm))

        out = {c: r.get(c, "") for c in OUT_COLUMNS}
        out.update(
            case=case,
            status=status,
            k_calc=f"{k_calc:.6f}",
            k_sigma=f"{k_sigma:.6f}",
            k_ref=f"{k_ref:.6f}",
            sigma_exp=f"{sigma_exp:.6f}",
            delta_pcm=f"{delta_pcm:.1f}",
            bound_pcm=f"{bound_pcm:.1f}",
            sigma_ratio=f"{sigma_ratio:.3f}",
            ref_source=ref_source,
        )
        out_rows.append(out)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    n_pass = sum(1 for r in out_rows if r.get("status") == "PASS")
    n_fail = sum(1 for r in out_rows if r.get("status") == "FAIL")
    print(f"re-graded {len(out_rows)} rows -> {out_csv}")
    print(f"  PASS={n_pass}  FAIL={n_fail}")
    print(f"  status changes ({len(changes)}):")
    for case, old, new, d in changes:
        print(f"    {case:32s} {old} -> {new}   (delta={d:+.0f} pcm)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_csv", type=Path, help="sweep CSV to re-grade")
    ap.add_argument("out_csv", type=Path, help="output corrected CSV")
    args = ap.parse_args()
    regrade(args.in_csv, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
