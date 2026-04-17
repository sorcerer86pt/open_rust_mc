"""PWR Pareto: reconstruction speed vs cross-section accuracy, with 4 curves.

Reads:
  outputs/pareto/xs_accuracy.csv  (per-reaction RMSE + ns/lookup)
  outputs/pareto/keff_pwr.csv     (PWR k_inf for OpenMC / CPU table / CPU SVD / GPU pointwise / GPU SVD)

Output:
  outputs/pareto/pareto_pwr.png   Two panels: Pareto frontier + k_inf vs rank
  outputs/pareto/pareto_pwr.md    Summary markdown table
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PARETO_DIR = ROOT / "outputs" / "pareto"
XS_CSV = PARETO_DIR / "xs_accuracy.csv"
KEFF_CSV = PARETO_DIR / "keff_pwr.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def aggregate_xs(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, float]]:
    """Geom-mean RMSE and mean ns/lookup per (kind, rank) across 8 reactions."""
    buckets: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for r in rows:
        kind = r["kind"]
        rank = int(r["rank"])
        rmse = float(r["rmse_log10"])
        ns = float(r["ns_per_lookup"])
        buckets.setdefault((kind, rank), []).append((rmse, ns))

    out = {}
    for key, items in buckets.items():
        rmses = [x[0] for x in items]
        nss = [x[1] for x in items]
        nonzero = [x for x in rmses if x > 0]
        geom = (
            math.exp(sum(math.log(x) for x in nonzero) / len(nonzero))
            if nonzero else 0.0
        )
        out[key] = {
            "rmse_geom": geom,
            "rmse_max": max(rmses),
            "ns_mean": sum(nss) / len(nss),
            "n": len(items),
        }
    return out


def plot(xs_agg, keff_rows, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # -------- Panel 1: Pareto frontier (XS speed vs XS accuracy) --------
    svd_ranks = sorted(r for (k, r) in xs_agg if k == "svd")
    svd_ns = [xs_agg[("svd", r)]["ns_mean"] for r in svd_ranks]
    svd_rmse = [max(xs_agg[("svd", r)]["rmse_geom"], 1e-15) for r in svd_ranks]

    ax1.plot(svd_ns, svd_rmse, "o-", color="C0", label="CPU SVD (rank 2..6)", markersize=9)
    for r, x, y in zip(svd_ranks, svd_ns, svd_rmse):
        ax1.annotate(f"k={r}", (x, y), xytext=(8, 4), textcoords="offset points")

    tbl = xs_agg.get(("table", 0))
    if tbl:
        ax1.plot([tbl["ns_mean"]], [1e-14], "s", color="C3", markersize=11,
                 label=f"Pointwise table ({tbl['ns_mean']:.1f} ns/lookup, exact)")

    ax1.set_xlabel("Reconstruction time per XS lookup (ns, geom-mean over 8 U235/U238/U234 reactions)")
    ax1.set_ylabel("Cross-section RMSE (log10 units, geom-mean)")
    ax1.set_yscale("log")
    ax1.set_title("Pareto: XS Reconstruction Speed vs Accuracy\n(closer to origin = better)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper right")

    # -------- Panel 2: PWR k_inf vs rank -------- for 4 curves
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for r in keff_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    def to_rank_arrays(mode: str):
        rs = []
        ks = []
        us = []
        for row in by_mode.get(mode, []):
            if row["rank"] == "-" or row["rank"] == "0":
                continue
            rs.append(int(row["rank"]))
            ks.append(float(row["k_inf"]))
            us.append(float(row["sigma_seed"]) / math.sqrt(5))
        order = sorted(range(len(rs)), key=lambda i: rs[i])
        return [rs[i] for i in order], [ks[i] for i in order], [us[i] for i in order]

    cpu_r, cpu_k, cpu_u = to_rank_arrays("cpu_svd")
    gpu_r, gpu_k, gpu_u = to_rank_arrays("gpu_svd")
    ax2.errorbar(cpu_r, cpu_k, yerr=cpu_u, fmt="o-", color="C0",
                 label="CPU SVD (ours)", capsize=4, markersize=8)
    ax2.errorbar(gpu_r, gpu_k, yerr=gpu_u, fmt="s-", color="C1",
                 label="GPU SVD (ours, --force-svd)", capsize=4, markersize=8)

    # Horizontal refs: OpenMC, CPU Table, GPU Pointwise
    all_ranks = sorted(set(cpu_r + gpu_r)) or [2, 6]
    x_min, x_max = min(all_ranks) - 0.5, max(all_ranks) + 0.5

    def horiz(mode, label, color):
        rows = by_mode.get(mode, [])
        if not rows: return
        k = float(rows[0]["k_inf"])
        s = float(rows[0]["sigma_seed"]) / math.sqrt(5)
        ax2.axhline(k, color=color, linestyle="--", alpha=0.8,
                    label=f"{label}: {k:.5f} ± {s:.5f}")
        ax2.fill_between([x_min, x_max], k - s, k + s, color=color, alpha=0.12)

    horiz("openmc", "OpenMC", "C2")
    horiz("cpu_table", "CPU table (ours)", "C3")
    horiz("gpu_pointwise", "GPU pointwise (ours)", "C4")

    ax2.set_xlabel("SVD rank")
    ax2.set_ylabel("PWR pin-cell k_inf (5 seeds × 10k particles × 50 batches)")
    ax2.set_xticks(all_ranks)
    ax2.set_title("Physics Accuracy: PWR k_inf vs Rank")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def dump_markdown(xs_agg, keff_rows, out_path: Path):
    lines = []
    lines.append("# PWR Pareto Summary\n")
    lines.append("Conditions: 5 seeds × 50 batches × 10 000 particles.\n")
    lines.append("## 1. XS reconstruction (geom-mean over 8 U235/U238/U234 reactions)\n")
    lines.append("| rank | ns/lookup | RMSE log10 | max |err| log10 |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(r for (k, r) in xs_agg if k == "svd"):
        a = xs_agg[("svd", r)]
        lines.append(f"| {r} | {a['ns_mean']:.2f} | {a['rmse_geom']:.2e} | {a['rmse_max']:.2e} |")
    tbl = xs_agg.get(("table", 0))
    if tbl:
        lines.append(f"| table | {tbl['ns_mean']:.2f} | 0 (ref) | 0 (ref) |")

    lines.append("\n## 2. PWR k_inf (reference + all 4 variants)\n")
    lines.append("| mode | rank | k_inf | σ_seed | SEM(5) | Δ vs OpenMC (pcm) | ns/particle |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    openmc_k = next((float(r["k_inf"]) for r in keff_rows if r["mode"] == "openmc"), None)
    for r in keff_rows:
        k = float(r["k_inf"])
        s = float(r["sigma_seed"])
        sem = s / math.sqrt(5)
        dpc = (k - openmc_k) * 1e5 if openmc_k else 0.0
        ns = r["ns_per_particle"]
        lines.append(f"| {r['mode']} | {r['rank']} | {k:.5f} | {s:.5f} | {sem:.5f} | {dpc:+.1f} | {ns} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    xs_rows = read_csv(XS_CSV)
    keff_rows = read_csv(KEFF_CSV)
    xs_agg = aggregate_xs(xs_rows)
    plot(xs_agg, keff_rows, PARETO_DIR / "pareto_pwr.png")
    dump_markdown(xs_agg, keff_rows, PARETO_DIR / "pareto_pwr.md")


if __name__ == "__main__":
    main()
