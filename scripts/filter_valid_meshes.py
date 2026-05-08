#!/usr/bin/env python3
"""Filter rr_adjoint_sweep CSVs to keep only physically meaningful meshes.

For the slab benchmark (100 cm of water at 1 MeV, mfp ≈ 14.14 cm), a
voxel is meaningful only if its smallest edge is ≥ mfp/8 ≈ 1.77 cm.
Below that the geometric subdivision is finer than the underlying
physics supports — there is no signal at sub-mfp scale, only random-
ray noise — so any "compression failure" reflects experimental design,
not the algorithm.

Reads N input CSVs, concatenates their physically valid rows, sorts by
voxel count, deduplicates by (n_x, n_y, n_z), and writes one merged
output CSV per Frobenius tolerance.
"""

from __future__ import annotations

import csv
from pathlib import Path

SLAB_CM = 100.0
MFP_CM = 14.14  # water at 1 MeV
MIN_VOXEL_CM = MFP_CM / 8.0  # 1.7675 cm


def is_physical(row: dict) -> bool:
    nx = max(int(row["n_x"]), 1)
    ny = max(int(row["n_y"]), 1)
    nz = max(int(row["n_z"]), 1)
    min_edge = min(SLAB_CM / nx, SLAB_CM / ny, SLAB_CM / nz)
    return min_edge >= MIN_VOXEL_CM


def merge(inputs: list[Path], out: Path) -> None:
    rows: dict[tuple, dict] = {}
    header: list[str] | None = None
    for p in inputs:
        with p.open() as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = list(reader.fieldnames or [])
            for r in reader:
                if not r or not r.get("n_voxels"):
                    continue
                if not is_physical(r):
                    continue
                key = (int(r["n_x"]), int(r["n_y"]), int(r["n_z"]))
                # Keep the first occurrence; assume duplicates are equivalent.
                if key not in rows:
                    rows[key] = r
    sorted_rows = sorted(rows.values(), key=lambda r: int(r["n_voxels"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header or [])
        writer.writeheader()
        writer.writerows(sorted_rows)
    print(f"wrote {out}: {len(sorted_rows)} valid meshes")


def main() -> None:
    here = Path(__file__).resolve().parent.parent / "outputs"
    # 2.5% tolerance: merge the original sweeps + extras
    merge(
        inputs=[
            here / "adjoint_sweep_full_tol2.5pct.csv",
            here / "adjoint_sweep_extra_valid_tol2.5pct.csv",
        ],
        out=here / "adjoint_sweep_valid_tol2.5pct.csv",
    )
    # 1% tolerance: merge
    merge(
        inputs=[
            here / "adjoint_sweep_full_tol1pct.csv",
            here / "adjoint_sweep_extra_valid_tol1pct.csv",
        ],
        out=here / "adjoint_sweep_valid_tol1pct.csv",
    )


if __name__ == "__main__":
    main()
