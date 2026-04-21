"""
Generate memory and throughput comparison plots for the paper.

Outputs:
  outputs/pareto/memory_compare.png     — bar chart of representation
                                          byte counts (pointwise,
                                          full-SVD, hybrid current,
                                          hybrid smooth-only projection)
                                          for the 9-nuclide PWR set.
  outputs/pareto/throughput_pwr.png     — ns/particle for each provider
                                          on PWR pin cell, laptop and
                                          desktop.
  outputs/pareto/throughput_godiva.png  — ns/particle for each provider
                                          on Godiva, desktop rank sweep.
  outputs/pareto/per_lookup_cost.png    — per-lookup cost ns/lookup for
                                          SVD ranks 2..6 vs pointwise
                                          table (kernel benchmark).

Everything uses measured numbers from the final benchmark tables in
the paper.
"""

import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "pareto",
)
os.makedirs(OUT, exist_ok=True)

# ── Memory comparison, 9-nuclide PWR working set ──────────────────────

def plot_memory():
    labels = [
        "Pointwise table\n(4 rxn × 6T)",
        "Full SVD basis\n(all reactions)",
        "Hybrid current\n(SVD + WMP)",
        "Hybrid smooth-only\n(EFC restricted + WMP)",
    ]
    # From Table of hybrid_mem_engine in the paper (MB).
    values = [101.5, 517.9, 519.0, 487.6]
    colors = ["#3478c7", "#c75434", "#d69832", "#4fa86b"]

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("XS working-set memory (MB)")
    ax.set_title(
        "In-engine XS memory, 9-nuclide PWR pin cell, rank-5\n"
        "Measured from loaded kernels at run time")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    # Annotate with values
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.1f} MB",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.1)

    # Side note
    ax.text(0.02, 0.95,
            "The 132.9× representation-byte ratio in\n"
            "Table 4 compares the 1.37 MB WMP payload\n"
            "against a 177.7 MB pointwise table for the\n"
            "same 4 reactions × 6 T only.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.9,
                      edgecolor="gray", boxstyle="round,pad=0.4"))

    plt.tight_layout()
    out_path = os.path.join(OUT, "memory_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out_path)


# ── Per-lookup cost (kernel benchmark) ─────────────────────────────────

def plot_per_lookup():
    ranks = [2, 3, 4, 5, 6]
    ns_lookup = [35.9, 41.2, 43.3, 43.4, 44.3]
    rmse = [3.00e-2, 7.06e-3, 1.83e-3, 6.07e-4, 1.76e-15]
    table_ns = 90.6

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ranks, ns_lookup, "-o", color="#3478c7", lw=1.8, ms=7,
            markerfacecolor="white", markeredgewidth=1.5,
            label="SVD reconstruction")
    ax.axhline(table_ns, color="#c75434", ls="--", lw=1.4,
               label=f"Pointwise table ({table_ns:.1f} ns)")

    for k, n, r in zip(ranks, ns_lookup, rmse):
        if r < 1e-13:
            lbl = "machine ε"
        else:
            lbl = f"RMSE {r:.1e}"
        ax.annotate(lbl, xy=(k, n), xytext=(0, -18),
                    textcoords="offset points",
                    ha="center", fontsize=7.5, color="#555555")

    ax.set_xlabel("SVD rank")
    ax.set_ylabel("ns / lookup")
    ax.set_title(
        "Per-lookup reconstruction cost\n"
        "Geometric mean over 8 reactions, Ryzen 9800X3D single-core,\n"
        "cold-cache query sequence")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 100)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUT, "per_lookup_cost.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out_path)


# ── Throughput, PWR pin cell, laptop + desktop ────────────────────────

def plot_throughput_pwr():
    # (mode, laptop_ns, desktop_ns)
    # Laptop = pre-correction five-seed paper numbers (Table keff_pwr).
    # Desktop = post-correction ten-seed rerun (20042026_1836 log,
    # 150 batches / 20 inactive / 50k particles / 10 seeds).
    # Hybrid row is pre-correction (benchmark predates sampler fix);
    # displayed for completeness.
    rows = [
        ("CPU table",            54430, 21316),
        ("CPU SVD r=5",          28651, 15511),
        ("GPU pointwise",        38104, 5967),
        ("GPU SVD r=5",          50738, 7754),
        ("Hybrid SVD+WMP",       None,  33051),
    ]
    labels   = [r[0] for r in rows]
    laptop   = [r[1] for r in rows]
    desktop  = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    w = 0.36
    bars_l = ax.bar(x - w / 2, [v if v is not None else 0 for v in laptop],
                    w, color="#3478c7", edgecolor="black", lw=0.6,
                    label="Laptop (Ryzen 7 + RTX A1000)")
    bars_d = ax.bar(x + w / 2, desktop,
                    w, color="#c75434", edgecolor="black", lw=0.6,
                    label="Desktop (Ryzen 9800X3D + RTX 3080)")

    for b, v in zip(bars_l, laptop):
        if v is None:
            ax.text(b.get_x() + w / 2, 500, "n/a",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(b.get_x() + w / 2, v + 1500, f"{v}",
                    ha="center", fontsize=7.5)
    for b, v in zip(bars_d, desktop):
        ax.text(b.get_x() + w / 2, v + 1500, f"{v}",
                ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ns / particle")
    ax.set_title("PWR pin cell throughput (lower is better)\n"
                 "Five seeds laptop, ten seeds desktop (except hybrid = five)")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    laptop_vals = [v for v in laptop if v is not None]
    ax.set_ylim(0, max(max(laptop_vals, default=0), max(desktop)) * 1.15)
    plt.tight_layout()
    out_path = os.path.join(OUT, "throughput_pwr.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out_path)


# ── Throughput, Godiva, desktop rank sweep ────────────────────────────

def plot_throughput_godiva():
    # Post-correction four-provider desktop benchmark
    # (20042026_1836 log, 150/20/50k, 10 seeds).
    # ICSBEP HEU-MET-FAST-001 reference k_eff = 1.0000 +/- 0.0010.
    labels = ["CPU table", "CPU SVD r=5", "GPU pointwise", "GPU SVD r=5"]
    ns     = [1207, 846, 302, 378]
    kinf   = [1.00330, 1.00325, 1.00339, 1.00362]
    sigma  = [0.00048, 0.00055, 0.00065, 0.00053]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    colors = ["#3478c7", "#3478c7", "#c75434", "#c75434"]
    x = np.arange(len(labels))
    bars = ax1.bar(x, ns, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, ns):
        ax1.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:.0f}",
                 ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("ns / particle")
    ax1.set_title("Godiva throughput, desktop (post-correction)\n"
                  "(blue = CPU, red = GPU)")
    ax1.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)

    # k_eff
    ax2.errorbar(x, kinf, yerr=sigma, fmt="o", color="#2b2b2b",
                 ms=7, lw=1.5, capsize=3)
    ax2.axhspan(1.0000 - 0.0010, 1.0000 + 0.0010,
                color="#7fbf7f", alpha=0.25,
                label="ICSBEP experiment 1.0000 ± 0.0010")
    ax2.axhline(1.00330, color="#3478c7", ls=":", lw=1.0, alpha=0.8,
                label="CPU-table engine baseline 1.00330")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel(r"$k_\text{eff}$")
    ax2.set_title("Godiva $k_\\mathrm{eff}$, desktop post-correction "
                  "($\\sigma_\\mathrm{seed}$ bars)")
    ax2.grid(True, ls=":", lw=0.5, alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT, "throughput_godiva.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out_path)


if __name__ == "__main__":
    plot_memory()
    plot_per_lookup()
    plot_throughput_pwr()
    plot_throughput_godiva()
