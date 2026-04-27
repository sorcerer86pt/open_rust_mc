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
        "Pointwise table\n(single-T baseline)",
        "Full SVD basis\nrank-5 all reactions",
        "Hybrid SVD+WMP\nfull SVD grid",
        "Hybrid SVD+WMP\nsmooth-only rebuild",
    ]
    # Live in-engine measurement, single seed × 100 batches × 20k:
    # full SVD basis    = 517 870.5 KB
    # WMP payload       =   1 153.8 KB
    # TOTAL (full grid) = 519 024.3 KB → 519.0 MB
    # smooth-only rebuild drops 31 423.1 KB across MT=2/18/102 →
    # smooth SVD basis  = 486 447.4 KB
    # TOTAL (smooth)    = 487 601.2 KB → 487.6 MB  (1.06× reduction)
    # Pointwise table  baseline 103 064.1 KB ≈ 100.6 MB.
    # Source: outputs/full_test_run/{04,11,12}_pwr_*.txt
    values = [100.6, 517.9, 519.0, 487.6]
    colors = ["#3478c7", "#c75434", "#d69832", "#4fa86b"]

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("XS working-set memory (MB)")
    ax.set_title(
        "In-engine XS memory, 9-nuclide PWR pin cell, rank-5\n"
        "Live measurement on loaded kernels (smooth rebuild now realised)")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    # Annotate with values
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.1f} MB",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(values) * 1.1)

    # Side note
    ax.text(0.02, 0.95,
            "The 132.9× representation-byte ratio\n"
            "(Table 4) compares the 1.37 MB WMP\n"
            "payload to a 177.7 MB four-channel\n"
            "pointwise table — a different slice.",
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
    # Laptop  = small-L3 (i7-12800H) post-correction:
    #            CPU table / CPU SVD / GPU pointwise / GPU SVD = 5-seed × 50b × 10k
    #            Hybrid SVD+WMP / ACE+WMP = single-seed × 100b × 20k post-correction
    #            (high-stats hybrid redo is open work)
    # Desktop = large-L3 (Ryzen 9800X3D) post-correction 10-seed × 150b × 50k
    #            CPU table / CPU SVD / GPU pointwise / GPU SVD;
    #            Hybrid SVD+WMP = pre-correction 5-seed (legacy comparison row)
    rows = [
        ("CPU table",            54430, 21316),
        ("CPU SVD r=5",          28651, 15511),
        ("GPU pointwise",        38104, 5967),
        ("GPU SVD r=5",          50738, 7754),
        ("Hybrid SVD+WMP",       26662, 33051),
        ("ACE+WMP",              26317, None),
    ]
    labels   = [r[0] for r in rows]
    laptop   = [r[1] for r in rows]
    desktop  = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    w = 0.36
    bars_l = ax.bar(x - w / 2, [v if v is not None else 0 for v in laptop],
                    w, color="#3478c7", edgecolor="black", lw=0.6,
                    label="Small-L3 (i7-12800H + RTX A1000)")
    bars_d = ax.bar(x + w / 2, [v if v is not None else 0 for v in desktop],
                    w, color="#c75434", edgecolor="black", lw=0.6,
                    label="Large-L3 (Ryzen 9800X3D + RTX 3080)")

    for b, v in zip(bars_l, laptop):
        if v is None:
            ax.text(b.get_x() + w / 2, 500, "n/a",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(b.get_x() + w / 2, v + 1500, f"{v}",
                    ha="center", fontsize=7.5)
    for b, v in zip(bars_d, desktop):
        if v is None:
            ax.text(b.get_x() + w / 2, 500, "n/a",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(b.get_x() + w / 2, v + 1500, f"{v}",
                    ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ns / particle")
    ax.set_title("PWR pin cell throughput (lower is better)\n"
                 "Five seeds laptop, ten seeds desktop "
                 "(hybrid laptop = single-seed post-correction; "
                 "hybrid desktop = pre-correction)")
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    laptop_vals = [v for v in laptop if v is not None]
    desktop_vals = [v for v in desktop if v is not None]
    ax.set_ylim(0, max(max(laptop_vals, default=0), max(desktop_vals)) * 1.15)
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


# ── On-library four-way honesty test, single-seed post-correction ─────

def plot_pwr_four_way():
    """Bar chart of k_inf for Table / SVD / Hybrid SVD+WMP / ACE+WMP on
    PWR pin cell, single-seed post-correction × 100 batches × 20k.
    Reference line = OpenMC 0.15.3 on the same HDF5 (1.32770 ± 150 pcm
    at 5-seed × 50b × 10k).
    Source: outputs/full_test_run/{04,10,11}_pwr_*.txt, single seed."""
    labels = ["CPU table", "CPU SVD r=5", "Hybrid SVD+WMP", "ACE+WMP"]
    k = [1.32903, 1.32817, 1.32795, 1.32821]
    sigma_seed = [0.00093, 0.00100, 0.00103, 0.00098]
    # Gaps to ACE+WMP (1.32821) — consistent within σ_seed for all rows.
    gaps_pcm = [(ki - k[3]) * 1e5 for ki in k]
    openmc_k = 1.32770
    openmc_sig = 0.00150

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # k_inf bars with σ_seed error bars
    x = np.arange(len(labels))
    colors = ["#3478c7", "#c75434", "#d69832", "#4fa86b"]
    ax1.bar(x, k, color=colors, edgecolor="black", lw=0.6,
            yerr=sigma_seed, capsize=4)
    for xi, ki in zip(x, k):
        ax1.text(xi, ki + 0.0012, f"{ki:.5f}", ha="center", fontsize=8.5)

    ax1.axhspan(openmc_k - openmc_sig, openmc_k + openmc_sig,
                color="#7fbf7f", alpha=0.25,
                label=f"OpenMC 0.15.3 ({openmc_k:.5f} ± {openmc_sig:.5f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel(r"$k_\infty$")
    ax1.set_ylim(1.324, 1.331)
    ax1.set_title(
        "PWR pin cell on-library, four-way honesty test\n"
        "single seed × 100 batches × 20k particles, post-correction")
    ax1.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8)

    # Gap to ACE+WMP
    bars2 = ax2.bar(x, gaps_pcm, color=colors, edgecolor="black", lw=0.6)
    for b, g in zip(bars2, gaps_pcm):
        ax2.text(b.get_x() + b.get_width() / 2,
                 g + (4 if g >= 0 else -8),
                 f"{g:+.0f} pcm", ha="center", fontsize=9)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel(r"$\Delta k_\infty$ vs ACE+WMP (pcm)")
    ax2.set_title(
        "Gap to ACE+WMP industry baseline\n"
        r"$|\Delta| < 1\,\sigma_{\rm seed}$ for SVD and Hybrid; "
        r"Table at $\sim 0.8\,\sigma_{\rm seed}$")
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax2.set_ylim(min(gaps_pcm) - 30, max(gaps_pcm) + 40)

    plt.tight_layout()
    out_path = os.path.join(OUT, "pwr_four_way.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out_path)


if __name__ == "__main__":
    plot_memory()
    plot_per_lookup()
    plot_throughput_pwr()
    plot_throughput_godiva()
    plot_pwr_four_way()
