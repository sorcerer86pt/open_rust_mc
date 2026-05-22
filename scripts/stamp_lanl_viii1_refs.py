# SPDX-License-Identifier: MIT
"""Stamp scene JSONs with `benchmark.local_validation.viii1` references
sourced from Nobre et al. 2025/2026 Table LIX (LANL MCNP validation of
ENDF/B-VIII.1 ICSBEP benchmarks).

Why
---

For the 289 of our 375 scenes that LANL did validate, we have a published
k_eff calculated under VIII.1 we can use as a secondary grading target —
no OpenMC run needed. The other 86 ("orphans") are filled by
`openmc_orphans_viii1.py` with a local OpenMC run; this script handles
the LANL-covered subset.

This script is idempotent: re-running overwrites the LANL block with the
same numbers (no churn) and leaves any pre-existing `viii1.openmc_*` keys
alone — those mean a local OpenMC run already happened (e.g. HCI-003
case-1's legacy VII.1 OpenMC block, which lives at
`benchmark.local_validation.openmc_*`, is left intact since it's at a
different key path).

Source for the table data: `outputs/refs/endfb_viii1_table_lix.csv`,
parsed from arxiv:2511.03564 pages 201-221. Match-up between our scene
case_ids and TABLE LIX keys lives in
`outputs/refs/scene_to_viii1_ref.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF_CSV = REPO_ROOT / "outputs" / "refs" / "scene_to_viii1_ref.csv"


def stamp_one(scene_path: Path, row: dict, source: str) -> str:
    with open(scene_path, encoding="utf-8") as fh:
        scene = json.load(fh)
    bench = scene.setdefault("benchmark", {})
    lv = bench.setdefault("local_validation", {})
    new_block = {
        "_what_this_is": (
            "Published reference k_eff under ENDF/B-VIII.1 from LANL MCNP "
            "validation (Nobre et al. 2025, Nuclear Data Sheets — Table "
            "LIX, arxiv:2511.03564). Used as the secondary grading target "
            "so the engine isn't punished for library bias it inherits "
            "from VIII.1."
        ),
        "lanl_k_eff": float(row["k_viii1"]),
        "lanl_sigma": float(row["sigma_viii1"]),
        "lanl_ce_ratio": float(row["ce_viii1"]),
        "lanl_table_lix_key": row["matched_key"],
        "k_exp_table": float(row["k_exp_table"]),
        "sigma_exp_table": float(row["sigma_exp_table"]),
        "source": source,
    }
    existing = lv.get("viii1") or {}
    if existing == new_block:
        return "noop"
    lv["viii1"] = new_block
    tmp = scene_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, scene_path)
    return "stamped" if not existing else "updated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", default=str(DEFAULT_REF_CSV))
    ap.add_argument(
        "--source",
        default="Nobre et al. 2025/2026, Nuclear Data Sheets, Table LIX (LANL MCNP, arxiv:2511.03564)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.ref_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    counts = {"stamped": 0, "updated": 0, "noop": 0, "no_lanl_ref": 0, "missing_file": 0}
    for r in rows:
        if not r.get("matched_key"):
            counts["no_lanl_ref"] += 1
            continue
        sp = REPO_ROOT / "bench" / "icsbep" / r["scene_file"]
        if not sp.exists():
            counts["missing_file"] += 1
            continue
        if args.dry_run:
            counts["stamped"] += 1
            continue
        action = stamp_one(sp, r, args.source)
        counts[action] += 1

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
