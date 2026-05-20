# SPDX-License-Identifier: MIT
"""Memory-vs-precision Pareto plot from rank-sweep CSVs.

Reads outputs/sweep_svd_wins_<geom>.csv (produced by sweep_svd_wins.py),
plots memory (MB) on x and |k - k_ref| (pcm) on y. SVD curve is
rank-labelled; Table and ACE+WMP are single points.

The reference k for each (geometry, scenario) panel is the ACE+WMP
k_inf at the highest measured rank (industry baseline).

Usage:
    python scripts/plot_memory_vs_precision.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def panel(ax, rows: list[dict], scenario: str, *, title: str) -> None:
    sub = [r for r in rows if r["scenario"] == scenario]
    if not sub:
        ax.set_visible(False)
        return

    # Reference: ACE+WMP at any rank (it's rank-independent for the table
    # baseline). Use the highest-rank row to match what production cites.
    wmp = [r for r in sub if r["provider"] == "ACE+WMP"]
    if not wmp:
        ax.set_visible(False)
        return
    k_ref = float(wmp[-1]["k_eff"])
    mem_wmp = float(wmp[-1]["mem_kb"]) / 1024.0

    # SVD curve
    svd = sorted([r for r in sub if r["provider"] == "SVD"],
                 key=lambda r: float(r["rank"]) if r["rank"] != "-" else 0)
    xs_svd = [float(r["mem_kb"]) / 1024.0 for r in svd]
    ys_svd = [abs(float(r["k_eff"]) - k_ref) * 1e5 for r in svd]
    ranks = [r["rank"] for r in svd]
    ax.plot(xs_svd, ys_svd, "o-", color="#1f77b4", label="SVD", markersize=8)
    for x, y, r in zip(xs_svd, ys_svd, ranks):
        ax.annotate(f"k={r}", (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color="#1f77b4")

    # Table point (one row, rank-independent)
    tbl = [r for r in sub if r["provider"] == "Table"]
    if tbl:
        x_t = float(tbl[-1]["mem_kb"]) / 1024.0
        y_t = abs(float(tbl[-1]["k_eff"]) - k_ref) * 1e5
        ax.plot([x_t], [y_t], "s", color="#d62728", markersize=10,
                label="Pointwise table")
        ax.annotate("Table", (x_t, y_t), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color="#d62728")

    # ACE+WMP reference point
    ax.plot([mem_wmp], [0], "^", color="#2ca02c", markersize=10,
            label="ACE+WMP (ref)")
    ax.annotate("ACE+WMP", (mem_wmp, 0), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color="#2ca02c")

    ax.set_xlabel("XS memory (MB)")
    ax.set_ylabel("|Δk| vs ACE+WMP (pcm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csvs = {
        "Godiva (3 nuclides, fast)":
            OUT / "sweep_svd_wins_godiva_fresh.csv",
        "PWR pin cell (9 nuclides, thermal)":
            OUT / "sweep_svd_wins_pwr.csv",
    }
    # Fall back to the older Godiva sweep if the fresh one isn't there yet.
    if not csvs["Godiva (3 nuclides, fast)"].exists():
        csvs["Godiva (3 nuclides, fast)"] = OUT / "sweep_svd_wins.csv"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=130)

    for col, (label, path) in enumerate(csvs.items()):
        if not path.exists():
            print(f"missing {path}; skipping")
            continue
        rows = load_csv(path)
        scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
        for row, scen in enumerate(scenarios[:2]):
            panel(axes[row, col], rows, scen,
                  title=f"{label}\n{scen}")

    fig.suptitle(
        "Memory vs precision Pareto — SVD curves, all-reactions, "
        "all-nuclide engine memory", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "memory_vs_precision.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
