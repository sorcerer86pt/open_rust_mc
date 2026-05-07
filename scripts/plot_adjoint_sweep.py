#!/usr/bin/env python3
"""Plot adjoint-flux SVD compression sweep across mesh sizes.

Reads two CSVs (one per Frobenius tolerance) produced by
`rr_adjoint_sweep` and emits a 3-panel figure:

  1. Compression ratio vs voxel count
  2. Frobenius reconstruction error vs voxel count
  3. Storage in bytes (dense vs picked) vs voxel count

Each panel handles ≥ 20 mesh sizes from 25 voxels to several million.
Annotations show the rank chosen by the picker (or "D" for dense).

Usage: python plot_adjoint_sweep.py [csv_tol_a csv_tol_b ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r and r.get("n_voxels")]
    rows.sort(key=lambda r: int(r["n_voxels"]))
    return rows


def main(out_dir: Path, csvs: List[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in csvs:
        rows = load_csv(p)
        if not rows:
            continue
        tol = float(rows[0]["frob_tol"])
        runs.append((tol, rows))
    runs.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax_ratio, ax_err, ax_bytes = axes

    for tol, rows in runs:
        n_vox = [int(r["n_voxels"]) for r in rows]
        ratio = [float(r["ratio"]) for r in rows]
        frob = [float(r["frob_err"]) for r in rows]
        kinds = [r["picker_kind"] for r in rows]
        ranks = [int(r["rank"]) for r in rows]
        comp_bytes = [int(r["comp_bytes"]) for r in rows]
        dense_bytes = [int(r["dense_bytes"]) for r in rows]

        label = f"frob_tol={tol*100:g}%"

        # 1: compression ratio
        ax_ratio.plot(n_vox, ratio, marker="o", label=label, linewidth=1.6, markersize=5)
        for x, y, k, rk in zip(n_vox, ratio, kinds, ranks):
            tag = "D" if k == "dense" else f"{rk}"
            ax_ratio.annotate(
                tag, (x, y), textcoords="offset points", xytext=(3, 4),
                fontsize=6, color="0.3"
            )

        # 2: Frob err — replace 0 (dense=exact) with eps for log scale
        eps = 1e-7
        frob_plot = [max(v, eps) for v in frob]
        ax_err.plot(n_vox, frob_plot, marker="o", label=label, linewidth=1.6, markersize=5)
        ax_err.axhline(tol, linestyle=":", alpha=0.5)

        # 3: bytes (dense vs picked)
        ax_bytes.plot(n_vox, comp_bytes, marker="o", label=f"{label} (picked)",
                      linewidth=1.6, markersize=5)

    # Dense baseline (single curve, drawn once from the last run since
    # dense bytes only depend on mesh size, not tolerance).
    if runs:
        last_rows = runs[-1][1]
        n_vox = [int(r["n_voxels"]) for r in last_rows]
        dense_bytes = [int(r["dense_bytes"]) for r in last_rows]
        ax_bytes.plot(n_vox, dense_bytes, color="black", linestyle="--",
                      label="dense baseline", alpha=0.6, linewidth=1.4)

    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("voxel count")
    ax_ratio.set_ylabel("compression ratio (× over dense)")
    ax_ratio.set_title("Compression ratio (numbers = SVD rank, D = dense)")
    ax_ratio.grid(True, which="both", alpha=0.3)
    ax_ratio.axhline(1.0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)
    ax_ratio.legend(loc="upper right")

    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("voxel count")
    ax_err.set_ylabel("Frobenius rel. recon error")
    ax_err.set_title("Reconstruction error (dotted = tolerance)")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(loc="lower right")

    ax_bytes.set_xscale("log")
    ax_bytes.set_yscale("log")
    ax_bytes.set_xlabel("voxel count")
    ax_bytes.set_ylabel("storage (bytes)")
    ax_bytes.set_title("On-disk storage (dense vs adaptive picker)")
    ax_bytes.grid(True, which="both", alpha=0.3)
    ax_bytes.legend(loc="upper left")

    fig.suptitle(
        "Phase 1 — adaptive SVD compression of the random-ray adjoint flux\n"
        "real slab, 1-group water at 1 MeV, 200x8000 mortal rays.  "
        "Voxels physically meaningful (smallest edge >= mfp/8 ~ 1.8 cm).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf = out_dir / "adjoint_compression.pdf"
    png = out_dir / "adjoint_compression.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    out_dir = here / "paper" / "figures"
    csvs = [
        here / "outputs" / "adjoint_sweep_valid_tol1pct.csv",
        here / "outputs" / "adjoint_sweep_valid_tol2.5pct.csv",
    ]
    if len(sys.argv) > 1:
        csvs = [Path(p) for p in sys.argv[1:]]
    main(out_dir, csvs)
