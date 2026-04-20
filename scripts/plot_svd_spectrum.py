"""
Render the SVD singular-value spectrum for representative reactions
across several PWR nuclides, plus a histogram showing how many of
U-235's 52 reactions are effectively rank-1.

Figures:
  outputs/pareto/svd_spectrum.png  — two-panel: (a) σ_k / σ_1 decay
                                     across reactions, (b) rank-1
                                     coverage histogram for U-235.

Input: data/endfb-vii.1-hdf5/neutron/*.h5 via openmc.data.
Requires the openmc conda env.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openmc.data

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEUTRON = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron"
OUT = "/mnt/c/Users/fog/madman_svd_experiment/outputs/pareto/svd_spectrum.png"

# (filename, mt, label) — reactions to show spectra for.
REACTIONS = [
    ("U235.h5", 18,  r"$^{235}$U fission"),
    ("U235.h5", 2,   r"$^{235}$U elastic"),
    ("U235.h5", 102, r"$^{235}$U capture"),
    ("U238.h5", 102, r"$^{238}$U capture"),
    ("U238.h5", 18,  r"$^{238}$U fission"),
    ("H1.h5",   2,   r"$^{1}$H elastic"),
    ("O16.h5",  2,   r"$^{16}$O elastic"),
    ("Zr90.h5", 102, r"$^{90}$Zr capture"),
]


def get_logA(nuc, mt):
    """Log-10 of cross section on union energy grid × T columns."""
    if mt not in nuc.reactions:
        return None
    rxn = nuc.reactions[mt]
    temps = sorted([t for t in nuc.energy if t.endswith("K") and
                    float(t.rstrip("K")) > 0],
                   key=lambda t: float(t.rstrip("K")))
    if len(temps) < 2:
        return None
    union_e = sorted(set(np.concatenate([nuc.energy[t] for t in temps])))
    cols = []
    for t in temps:
        if t not in rxn.xs:
            return None
        v = np.asarray(rxn.xs[t](union_e))
        cols.append(np.maximum(v, 1e-30))
    A = np.column_stack(cols)
    return np.log10(A)


def svd_spectrum(logA):
    return np.linalg.svd(logA, compute_uv=False)


def rank1_coverage(filename):
    """For each of U-235's reactions, return σ_2/σ_1 (0 if rank-1)."""
    nuc = openmc.data.IncidentNeutron.from_hdf5(
        os.path.join(NEUTRON, filename))
    ratios = {}
    for mt, rxn in nuc.reactions.items():
        if rxn.redundant:
            continue
        logA = get_logA(nuc, mt)
        if logA is None or logA.shape[1] < 2:
            continue
        s = svd_spectrum(logA)
        if s[0] <= 0:
            continue
        ratios[mt] = s[1] / s[0] if len(s) > 1 else 0.0
    return ratios


def main():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel (a): spectrum decay.
    cmap = plt.get_cmap("tab10")
    for i, (fn, mt, label) in enumerate(REACTIONS):
        path = os.path.join(NEUTRON, fn)
        try:
            nuc = openmc.data.IncidentNeutron.from_hdf5(path)
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        logA = get_logA(nuc, mt)
        if logA is None:
            print(f"  {fn} MT={mt} missing")
            continue
        s = svd_spectrum(logA)
        k = np.arange(1, len(s) + 1)
        ax1.semilogy(k, s / s[0], "-o", label=label, color=cmap(i), lw=1.4,
                     ms=4.5, markerfacecolor="white", markeredgewidth=1.0)
    ax1.set_xlabel("singular index $k$")
    ax1.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax1.set_title(
        "SVD singular spectrum, ENDF/B-VII.1,\n"
        r"$\log_{10}\sigma(E, T)$ over $N_T = 6$ library temperatures")
    ax1.axhline(1e-14, linestyle=":", color="gray", lw=0.8, alpha=0.7)
    ax1.text(6, 1.5e-14, "machine precision", color="gray", fontsize=8,
             va="bottom")
    ax1.legend(fontsize=8, loc="lower left", frameon=False)
    ax1.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax1.set_xlim(0.5, 6.5)
    ax1.set_ylim(1e-16, 2)

    # Panel (b): U-235 rank-1 histogram.
    ratios = rank1_coverage("U235.h5")
    vals = np.array(list(ratios.values()))
    n_total = len(vals)
    n_rank1 = int(np.sum(vals < 1e-13))  # σ_2/σ_1 below machine precision
    # Clip for display
    vals_clip = np.clip(vals, 1e-16, 1)
    bins = np.logspace(-16, 0, 17)
    ax2.hist(vals_clip, bins=bins, color="#3478c7", edgecolor="white")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\sigma_2 / \sigma_1$ per reaction")
    ax2.set_ylabel("count")
    ax2.set_title(
        rf"$^{{235}}$U non-redundant reactions ($n={n_total}$):"
        f"\n{n_rank1} rank-1 to machine precision")
    ax2.axvline(1e-13, color="gray", ls=":", lw=0.8)
    ax2.grid(True, ls=":", lw=0.5, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}  ({n_rank1}/{n_total} rank-1)")


if __name__ == "__main__":
    main()
