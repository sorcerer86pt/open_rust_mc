# SPDX-License-Identifier: MIT
"""
Phase 5.3 — Windowed SVD: independent SVD per energy region.

Hypothesis: the resonance region has different low-rank structure than
the thermal/fast regions. By applying SVD independently to each window,
the per-window rank can be lower than the global rank needed for the
same accuracy — analogous to Windowed Multipole but for SVD.

Windows:
  W1: E < 1 eV            (thermal — smooth 1/v behaviour)
  W2: 1 eV – 100 eV       (dominant resolved resonances)
  W3: 100 eV – 2.5 keV    (dense resonance forest)
  W4: 2.5 keV – 25 keV    (upper resolved / unresolved boundary)
  W5: E > 25 keV           (fast — smooth, no resonances)

For each window we run:
  • Full SVD + spectrum analysis (effective rank per window)
  • Leave-one-out cross-validation (interpolation error per window)
  • Compare: windowed vs. global SVD at same total storage budget

Usage:
    python scripts/phase5_3_windowed_svd.py [prefix]
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


WINDOWS = [
    ("Thermal",         0,       1.0),
    ("Low Resonance",   1.0,     100.0),
    ("Mid Resonance",   100.0,   2500.0),
    ("High Resonance",  2500.0,  25000.0),
    ("Fast",            25000.0, np.inf),
]


def load_data(output_dir: str, prefix: str = ""):
    A_raw = np.load(os.path.join(output_dir, f"{prefix}A_raw_u235_mt18.npy"))
    A_log = np.load(os.path.join(output_dir, f"{prefix}A_log_u235_mt18.npy"))
    energies = np.load(os.path.join(output_dir, f"{prefix}energies_u235.npy"))
    temperatures = np.load(os.path.join(output_dir, f"{prefix}temperatures_u235.npy"),
                           allow_pickle=True)
    return A_raw, A_log, energies, temperatures


def svd_per_window(A_log, energies):
    """Run SVD on each energy window independently."""
    results = {}
    for name, E_lo, E_hi in WINDOWS:
        mask = (energies >= E_lo) & (energies < E_hi)
        n_pts = mask.sum()
        if n_pts == 0:
            continue

        A_w = A_log[mask, :]
        U_w, S_w, Vt_w = svd(A_w, full_matrices=False)

        results[name] = {
            'mask': mask,
            'n_pts': n_pts,
            'U': U_w, 'S': S_w, 'Vt': Vt_w,
            'A_log': A_w,
            'E_range': (E_lo, E_hi),
        }

    return results


def print_spectrum_table(window_results):
    """Print comparative singular spectrum across all windows."""
    print("=" * 90)
    print("SINGULAR SPECTRUM PER WINDOW")
    print("=" * 90)

    # Header
    names = list(window_results.keys())
    max_rank = max(len(r['S']) for r in window_results.values())

    print(f"\n{'':>18}", end="")
    for name in names:
        print(f"  {name:>14}", end="")
    print()

    print(f"{'N_E':>18}", end="")
    for name in names:
        print(f"  {window_results[name]['n_pts']:>14,}", end="")
    print()

    print(f"{'E range':>18}", end="")
    for name in names:
        lo, hi = window_results[name]['E_range']
        hi_s = f"{hi:.0f}" if hi < 1e6 else "∞"
        print(f"  {lo:.0f}–{hi_s} eV".rjust(14), end="")
    print()

    print("-" * 90)

    # Singular values (normalized)
    print(f"\n{'σ_k/σ_1':>18}", end="")
    print()
    for k in range(max_rank):
        print(f"  k={k+1:>2}            ", end="")
        for name in names:
            S = window_results[name]['S']
            if k < len(S):
                print(f"  {S[k]/S[0]:>14.2e}", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()

    # Cumulative energy
    print(f"\n{'Energy captured':>18}")
    for k in range(max_rank):
        print(f"  k={k+1:>2}            ", end="")
        for name in names:
            S = window_results[name]['S']
            if k < len(S):
                cum = np.cumsum(S**2) / np.sum(S**2)
                print(f"  {cum[k]*100:>13.6f}%", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()


def loocv_per_window(window_results, A_raw, energies, temperatures):
    """Leave-one-out cross-validation per window at various k."""
    print("\n" + "=" * 90)
    print("LEAVE-ONE-OUT CROSS-VALIDATION PER WINDOW")
    print("=" * 90)

    N_T = len(temperatures)
    all_loocv = {}

    for wname, wr in window_results.items():
        mask = wr['mask']
        A_w_log = wr['A_log']
        A_w_raw = A_raw[mask, :]
        max_k = min(len(wr['S']), N_T - 1)

        loocv = {}
        for k in range(2, max_k + 1):
            errors = []
            for hold_idx in range(N_T):
                train_mask = np.ones(N_T, dtype=bool)
                train_mask[hold_idx] = False

                A_train = A_w_log[:, train_mask]
                U_tr, S_tr, Vt_tr = svd(A_train, full_matrices=False)
                k_use = min(k, len(S_tr))
                U_k = U_tr[:, :k_use]

                col_hold_log = A_w_log[:, hold_idx]
                coeffs, _, _, _ = np.linalg.lstsq(U_k, col_hold_log, rcond=None)
                col_recon = 10 ** (U_k @ coeffs)
                col_orig = A_w_raw[:, hold_idx]

                err_rel = np.abs(col_orig - col_recon) / (col_orig + 1e-30)
                errors.append({
                    'T': temperatures[hold_idx],
                    'max_err': err_rel.max(),
                    'mean_err': err_rel.mean(),
                    'p99': np.percentile(err_rel, 99),
                })
            loocv[k] = errors
        all_loocv[wname] = loocv

    return all_loocv


def print_loocv_summary(all_loocv):
    """Print compact summary: worst-case LOO error per window per k."""
    print("\nWorst-case LOO max error per window per k:")
    print(f"\n{'k':>4}", end="")
    for wname in all_loocv:
        print(f"  {wname:>14}", end="")
    print(f"  {'GLOBAL':>14}")
    print("-" * (4 + 16 * (len(all_loocv) + 1)))

    all_ks = sorted(set(k for loocv in all_loocv.values() for k in loocv))
    for k in all_ks:
        print(f"  {k:>2}", end="")
        window_maxes = []
        for wname, loocv in all_loocv.items():
            if k in loocv:
                worst = max(r['max_err'] for r in loocv[k])
                window_maxes.append(worst)
                print(f"  {worst:>14.2e}", end="")
            else:
                print(f"  {'—':>14}", end="")
        # Global = worst across all windows
        if window_maxes:
            print(f"  {max(window_maxes):>14.2e}", end="")
        print()

    # P99 version
    print("\nWorst-case LOO P99 error per window per k:")
    print(f"\n{'k':>4}", end="")
    for wname in all_loocv:
        print(f"  {wname:>14}", end="")
    print()
    print("-" * (4 + 16 * len(all_loocv)))

    for k in all_ks:
        print(f"  {k:>2}", end="")
        for wname, loocv in all_loocv.items():
            if k in loocv:
                worst_p99 = max(r['p99'] for r in loocv[k])
                print(f"  {worst_p99:>14.2e}", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()


def budget_comparison(window_results, all_loocv, A_raw, A_log, energies, temperatures):
    """Compare windowed vs. global SVD at the same storage budget."""
    print("\n" + "=" * 90)
    print("STORAGE BUDGET COMPARISON: WINDOWED vs. GLOBAL SVD")
    print("=" * 90)

    N_E, N_T = A_log.shape

    # Global SVD LOO at various k
    print("\nGlobal SVD (single SVD over full energy range):")
    global_results = {}
    for k in range(2, N_T):
        storage = k * (N_E + N_T + 1) * 8  # bytes
        errors = []
        for hold_idx in range(N_T):
            train_mask = np.ones(N_T, dtype=bool)
            train_mask[hold_idx] = False
            A_tr = A_log[:, train_mask]
            U_tr, S_tr, Vt_tr = svd(A_tr, full_matrices=False)
            k_use = min(k, len(S_tr))
            U_k = U_tr[:, :k_use]
            coeffs, _, _, _ = np.linalg.lstsq(U_k, A_log[:, hold_idx], rcond=None)
            col_recon = 10 ** (U_k @ coeffs)
            col_orig = A_raw[:, hold_idx]
            err_rel = np.abs(col_orig - col_recon) / (col_orig + 1e-30)
            errors.append(err_rel.max())
        worst = max(errors)
        global_results[k] = {'storage': storage, 'worst_err': worst}
        print(f"  k={k}: storage={storage/1024:.0f} KB, worst LOO max_err={worst:.2e}")

    # Windowed SVD: allocate k per window to minimize total error
    # Strategy: give each window enough k to reach ~same error level
    print("\nWindowed SVD (independent SVD per energy region):")

    # Try uniform k allocation first
    for k_per_window in range(2, N_T):
        total_storage = 0
        worst_err = 0
        details = []
        for wname, wr in window_results.items():
            n = wr['n_pts']
            kw = min(k_per_window, len(wr['S']))
            storage_w = kw * (n + N_T + 1) * 8
            total_storage += storage_w

            if k_per_window in all_loocv[wname]:
                w_err = max(r['max_err'] for r in all_loocv[wname][k_per_window])
            else:
                # Use highest available k
                avail = sorted(all_loocv[wname].keys())
                w_err = max(r['max_err'] for r in all_loocv[wname][avail[-1]])
            worst_err = max(worst_err, w_err)
            details.append(f"{wname}(k={kw})")

        # Find comparable global k (closest storage)
        closest_global_k = min(global_results.keys(),
                               key=lambda gk: abs(global_results[gk]['storage'] - total_storage))
        global_err = global_results[closest_global_k]['worst_err']

        improvement = global_err / worst_err if worst_err > 0 else float('inf')
        print(f"  k={k_per_window}/window: storage={total_storage/1024:.0f} KB, "
              f"worst_err={worst_err:.2e}  "
              f"(global k={closest_global_k} at {global_results[closest_global_k]['storage']/1024:.0f} KB "
              f"= {global_err:.2e}, "
              f"{'WINDOWED WINS' if worst_err < global_err else 'global wins'} "
              f"{improvement:.1f}×)")

    # Adaptive allocation: give resonance windows more k
    print("\n  Adaptive allocation (more k to resonance windows):")
    configs = [
        {"Thermal": 2, "Low Resonance": 5, "Mid Resonance": 5, "High Resonance": 4, "Fast": 2},
        {"Thermal": 2, "Low Resonance": 6, "Mid Resonance": 6, "High Resonance": 5, "Fast": 2},
        {"Thermal": 3, "Low Resonance": 7, "Mid Resonance": 7, "High Resonance": 6, "Fast": 3},
    ]

    for config in configs:
        total_storage = 0
        worst_err = 0
        for wname, wr in window_results.items():
            kw = min(config.get(wname, 3), len(wr['S']))
            storage_w = kw * (wr['n_pts'] + N_T + 1) * 8
            total_storage += storage_w

            avail_ks = sorted(all_loocv[wname].keys())
            k_use = min(kw, max(avail_ks))
            k_use = max(k_use, min(avail_ks))
            w_err = max(r['max_err'] for r in all_loocv[wname][k_use])
            worst_err = max(worst_err, w_err)

        closest_global_k = min(global_results.keys(),
                               key=lambda gk: abs(global_results[gk]['storage'] - total_storage))
        global_err = global_results[closest_global_k]['worst_err']

        alloc_str = ", ".join(f"{n[:3]}={k}" for n, k in config.items())
        print(f"  [{alloc_str}]: "
              f"storage={total_storage/1024:.0f} KB, worst_err={worst_err:.2e} "
              f"(vs global={global_err:.2e})")


def plot_windowed_results(window_results, all_loocv, output_dir):
    """Plot per-window singular spectra and LOO errors."""
    n_windows = len(window_results)
    fig, axes = plt.subplots(2, n_windows, figsize=(4 * n_windows, 10))
    if n_windows == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_windows))

    for i, (wname, wr) in enumerate(window_results.items()):
        S = wr['S']

        # Top row: singular spectrum
        ax = axes[0, i]
        ax.semilogy(range(1, len(S)+1), S/S[0], 'o-', color=colors[i],
                     markersize=6, linewidth=2)
        ax.axhline(y=1e-4, ls='--', color='red', alpha=0.4)
        ax.set_title(f'{wname}\n({wr["n_pts"]:,} pts)', fontsize=10, fontweight='bold')
        ax.set_xlabel('k')
        if i == 0:
            ax.set_ylabel('σ_k/σ_1')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, len(S)+1))

        # Bottom row: LOO worst error vs k
        ax2 = axes[1, i]
        loocv = all_loocv[wname]
        ks = sorted(loocv.keys())
        worst_errs = [max(r['max_err'] for r in loocv[k]) for k in ks]
        worst_p99 = [max(r['p99'] for r in loocv[k]) for k in ks]

        ax2.semilogy(ks, worst_errs, 'o-', color=colors[i], label='Max', linewidth=2)
        ax2.semilogy(ks, worst_p99, 's--', color=colors[i], label='P99', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=1e-2, ls='--', color='orange', alpha=0.4)
        ax2.axhline(y=1e-3, ls=':', color='red', alpha=0.4)
        ax2.set_xlabel('k')
        if i == 0:
            ax2.set_ylabel('LOO Error')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ks)

    axes[0, 0].set_ylabel('σ_k/σ_1', fontsize=11)
    axes[1, 0].set_ylabel('LOO Worst Error', fontsize=11)

    plt.suptitle('Windowed SVD: Per-Region Singular Spectra & LOO Errors',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "windowed_svd_analysis.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {path}")
    plt.close()


def main():
    output_dir = os.path.expanduser("~/madman_svd_experiment/outputs")
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""

    A_raw, A_log, energies, temperatures = load_data(output_dir, prefix)
    N_E, N_T = A_log.shape
    print(f"Dataset: {prefix or 'default'}")
    print(f"Matrix: {N_E} × {N_T}, temperatures: {list(temperatures)}")

    # 1. SVD per window
    print("\nWindows defined:")
    for name, lo, hi in WINDOWS:
        mask = (energies >= lo) & (energies < hi)
        print(f"  {name:>18}: {lo:>10.1f} – {'∞' if hi == np.inf else f'{hi:.1f}':>10} eV  "
              f"({mask.sum():>6,} points, {mask.sum()/N_E*100:.1f}%)")

    window_results = svd_per_window(A_log, energies)

    # 2. Spectrum comparison
    print_spectrum_table(window_results)

    # 3. LOO per window
    all_loocv = loocv_per_window(window_results, A_raw, energies, temperatures)
    print_loocv_summary(all_loocv)

    # 4. Budget comparison
    budget_comparison(window_results, all_loocv, A_raw, A_log, energies, temperatures)

    # 5. Plots
    plot_windowed_results(window_results, all_loocv, output_dir)

    # Verdict
    print("\n" + "=" * 90)
    print("WINDOWED SVD VERDICT")
    print("=" * 90)

    print("\n  Per-window effective rank (k for 99.99% energy):")
    for wname, wr in window_results.items():
        S = wr['S']
        cum = np.cumsum(S**2) / np.sum(S**2)
        k99_99 = int(np.argmax(cum >= 0.9999)) + 1
        ratio = S[1]/S[0] if len(S) > 1 else 0
        print(f"    {wname:>18}: k={k99_99}, σ_2/σ_1={ratio:.2e}, "
              f"N_E={wr['n_pts']:,}")


if __name__ == "__main__":
    main()
