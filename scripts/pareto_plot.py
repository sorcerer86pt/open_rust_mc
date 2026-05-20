# SPDX-License-Identifier: MIT
"""Pareto frontier: SVD reconstruction time vs cross-section RMSE.

Reads:
  outputs/pareto/xs_accuracy.csv  (rank -> per-reaction RMSE, ns/lookup)
  outputs/pareto/keff_sweep.csv   (rank -> Godiva k_eff and ns/particle)

Plots:
  - Reconstruction Pareto (X: ns/lookup, Y: RMSE log10)
  - k_eff vs rank with error bars
"""
from __future__ import annotations

import csv
from pathlib import Path
import math

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PARETO_DIR = ROOT / "outputs" / "pareto"
XS_CSV = PARETO_DIR / "xs_accuracy.csv"
KEFF_CSV = PARETO_DIR / "keff_sweep.csv"


def load_xs():
    rows = []
    with XS_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_keff():
    rows = []
    with KEFF_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate_by_rank(rows):
    """Per-rank mean of log10 RMSE and ns/lookup across the 8 reactions."""
    buckets: dict[str, dict[int, list[tuple[float, float, int]]]] = {}
    for r in rows:
        kind = r["kind"]
        rank = int(r["rank"])
        rmse = float(r["rmse_log10"])
        ns = float(r["ns_per_lookup"])
        mem = int(r["mem_bytes"])
        buckets.setdefault(kind, {}).setdefault(rank, []).append((rmse, ns, mem))

    agg = {}
    for kind, by_rank in buckets.items():
        for rank, items in by_rank.items():
            rmses = [x[0] for x in items]
            nss = [x[1] for x in items]
            mems = [x[2] for x in items]
            # Geometric mean of RMSE (handles orders-of-magnitude spread)
            nonzero = [x for x in rmses if x > 0]
            if nonzero:
                geom_rmse = math.exp(sum(math.log(x) for x in nonzero) / len(nonzero))
            else:
                geom_rmse = 0.0
            agg[(kind, rank)] = {
                "rmse_geom": geom_rmse,
                "rmse_mean": sum(rmses) / len(rmses),
                "rmse_max": max(rmses),
                "ns_mean": sum(nss) / len(nss),
                "mem_total": sum(mems),
                "n_reactions": len(items),
            }
    return agg


def plot_pareto(agg, keff_rows, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Pareto: ns/lookup vs RMSE (log10) ----
    svd_ranks = sorted(r for (k, r) in agg.keys() if k == "svd")

    svd_ns = [agg[("svd", r)]["ns_mean"] for r in svd_ranks]
    svd_rmse = [agg[("svd", r)]["rmse_geom"] for r in svd_ranks]
    # Floor rank-6 so it's visible on log axis
    svd_rmse_plot = [max(x, 1e-15) for x in svd_rmse]

    ax1.plot(svd_ns, svd_rmse_plot, "o-", color="C0", label="SVD (rank 2..6)", markersize=8)
    for r, x, y in zip(svd_ranks, svd_ns, svd_rmse_plot):
        ax1.annotate(f"k={r}", (x, y), xytext=(6, 6), textcoords="offset points")

    tbl = agg.get(("table", 0))
    if tbl:
        ax1.axhline(1e-14, color="C3", linestyle="--", alpha=0.3)
        ax1.plot([tbl["ns_mean"]], [1e-14], "s", color="C3", markersize=10,
                 label=f"Pointwise table ({tbl['ns_mean']:.1f} ns/lookup, exact)")

    ax1.set_xlabel("Reconstruction time (ns / lookup)")
    ax1.set_ylabel("Cross-section RMSE (log10 units, geometric mean over 8 reactions)")
    ax1.set_yscale("log")
    ax1.set_title("Pareto: SVD Reconstruction Speed vs Accuracy\n(closer to origin = better)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right")

    # ---- k_eff vs rank ----
    # Add seed-mean uncertainty: σ_SEM = σ_seed / sqrt(5)
    keff_mode = {r["mode"]: r for r in keff_rows}
    ranks_plot = [int(r["rank"]) for r in keff_rows if r["mode"] == "svd"]
    k_means = [float(r["k_eff"]) for r in keff_rows if r["mode"] == "svd"]
    k_sigs = [float(r["sigma_seed"]) / math.sqrt(5) for r in keff_rows if r["mode"] == "svd"]

    ax2.errorbar(ranks_plot, k_means, yerr=k_sigs, fmt="o-", color="C0",
                 label="SVD (error = σ_seed/√5)", markersize=8, capsize=4)

    tbl_keff = next((r for r in keff_rows if r["mode"] == "table"), None)
    if tbl_keff:
        kt = float(tbl_keff["k_eff"])
        st = float(tbl_keff["sigma_seed"]) / math.sqrt(5)
        ax2.axhline(kt, color="C3", linestyle="--", alpha=0.7,
                    label=f"Pointwise table: {kt:.5f} ± {st:.5f}")
        ax2.fill_between([min(ranks_plot) - 0.5, max(ranks_plot) + 0.5],
                         kt - st, kt + st, color="C3", alpha=0.15)

    ax2.set_xlabel("SVD rank")
    ax2.set_ylabel("k_eff (Godiva, 5 seeds × 10k particles × 50 batches)")
    ax2.set_xticks(ranks_plot)
    ax2.set_title("Physics Accuracy: Godiva k_eff vs Rank")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def print_summary(agg, keff_rows):
    print("\n=== XS Reconstruction (geom-mean over 8 reactions) ===")
    print(f"{'rank':>5} {'ns/lookup':>10} {'RMSE log10':>14} {'max abs':>12} {'mem MB':>10}")
    for rank in sorted(r for (k, r) in agg.keys() if k == "svd"):
        a = agg[("svd", rank)]
        print(f"{rank:>5} {a['ns_mean']:>10.2f} {a['rmse_geom']:>14.2e} "
              f"{a['rmse_max']:>12.2e} {a['mem_total']/1e6:>10.1f}")
    tbl = agg.get(("table", 0))
    if tbl:
        print(f"{'tbl':>5} {tbl['ns_mean']:>10.2f} {0.0:>14.2e} "
              f"{0.0:>12.2e} {tbl['mem_total']/1e6:>10.1f}")

    print("\n=== Godiva k_eff sweep (5 seeds) ===")
    print(f"{'mode':>6} {'rank':>5} {'k_eff':>10} {'sigma':>10} "
          f"{'ns/part':>10}  {'delta pcm':>20}")
    tbl = next((r for r in keff_rows if r["mode"] == "table"), None)
    k_table = float(tbl["k_eff"]) if tbl else None
    for r in keff_rows:
        k = float(r["k_eff"])
        s = float(r["sigma_seed"])
        delta = (k - k_table) * 1e5 if k_table else 0.0
        print(f"{r['mode']:>6} {r['rank']:>5} {k:>10.5f} {s:>10.5f} "
              f"{float(r['ns_per_particle']):>10.1f}  {delta:>+20.1f}")


def main():
    xs_rows = load_xs()
    keff_rows = load_keff()
    agg = aggregate_by_rank(xs_rows)
    print_summary(agg, keff_rows)
    plot_pareto(agg, keff_rows, PARETO_DIR / "pareto.png")


if __name__ == "__main__":
    main()
