"""
Phase 2 — SVD decomposition and singular spectrum analysis.

Loads the log-transformed matrix A from Phase 1, performs full SVD,
and produces diagnostic plots of the singular spectrum to determine
the effective rank and compression feasibility.

Usage:
    python scripts/phase2_svd_analysis.py [prefix]

    prefix: file prefix for Phase 1 outputs (default: "", use "jeff33_" for JEFF data)
"""

import os
import sys
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_phase1_data(output_dir: str, prefix: str = ""):
    """Load matrices saved by Phase 1."""
    A_log = np.load(os.path.join(output_dir, f"{prefix}A_log_u235_mt18.npy"))
    energies = np.load(os.path.join(output_dir, f"{prefix}energies_u235.npy"))
    temperatures = np.load(os.path.join(output_dir, f"{prefix}temperatures_u235.npy"),
                           allow_pickle=True)
    print(f"Loaded A_log: {A_log.shape}")
    print(f"Energy grid: {len(energies)} points")
    print(f"Temperatures: {temperatures}")
    return A_log, energies, temperatures


def perform_svd(A_log: np.ndarray):
    """Execute full SVD (economy mode)."""
    print(f"\nPerforming SVD on matrix {A_log.shape}...")
    U, S, Vt = svd(A_log, full_matrices=False)

    print(f"\nSingular values: {S}")
    print(f"Number of singular values: {len(S)}")
    print(f"\nRatios σ_k/σ_1:")
    for k, s in enumerate(S):
        print(f"  σ_{k+1}/σ_1 = {s/S[0]:.10e}")

    return U, S, Vt


def analyze_spectrum(S: np.ndarray, A_log: np.ndarray):
    """Analyze the singular spectrum and determine effective rank."""

    print("\n" + "=" * 60)
    print("SINGULAR SPECTRUM ANALYSIS")
    print("=" * 60)

    # Ratio analysis
    ratio_2_1 = S[1] / S[0]
    print(f"\nσ_2/σ_1 = {ratio_2_1:.6e}")

    if ratio_2_1 < 0.01:
        scenario = "A (EXCELLENT)"
        print(f"→ SCENARIO A: Effective rank is very low. SVD compression is highly viable.")
    elif ratio_2_1 < 0.1:
        scenario = "B (GOOD)"
        print(f"→ SCENARIO B: SVD viable with moderate k. May need optimizations.")
    else:
        scenario = "C (CHALLENGING)"
        print(f"→ SCENARIO C: SVD alone may be insufficient. Consider hybrid WMP+SVD.")

    # Energy captured
    S2 = S ** 2
    energy_cumulative = np.cumsum(S2) / np.sum(S2)
    print(f"\nCumulative energy captured:")
    for k, ec in enumerate(energy_cumulative):
        print(f"  k={k+1}: {ec * 100:.8f}%")

    # Reconstruction error for each truncation rank
    N_E, N_T = A_log.shape
    errors_frobenius = []
    norm_A = np.linalg.norm(A_log, 'fro')

    for k in range(1, len(S) + 1):
        # Efficient: error from discarded singular values
        err = np.sqrt(np.sum(S[k:] ** 2)) / norm_A
        errors_frobenius.append(err)

    print(f"\nReconstruction error (Frobenius, relative):")
    for k, err in enumerate(errors_frobenius):
        print(f"  k={k+1}: ε = {err:.2e}")

    # Find minimum k for each threshold
    print(f"\nMinimum k for error thresholds:")
    thresholds = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    results = {}
    for threshold in thresholds:
        k_min = next((i + 1 for i, e in enumerate(errors_frobenius)
                       if e < threshold), None)
        if k_min is not None:
            compression = 1 - (k_min * (N_E + N_T)) / (N_E * N_T)
            print(f"  ε < {threshold:.0e}: k={k_min}, "
                  f"compression={compression * 100:.1f}%")
            results[threshold] = (k_min, compression)
        else:
            print(f"  ε < {threshold:.0e}: NOT ACHIEVABLE with available rank")
            results[threshold] = (None, None)

    return scenario, energy_cumulative, errors_frobenius, results


def plot_spectrum(S, energy_cumulative, errors_frobenius, output_dir):
    """Generate the 3-panel diagnostic plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Singular values (absolute)
    axes[0].semilogy(range(1, len(S) + 1), S, 'o-', markersize=8,
                     linewidth=2, color='#1f77b4')
    axes[0].set_title('Singular Values σ_k', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Index k')
    axes[0].set_ylabel('σ_k')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(1, len(S) + 1))

    # Plot 2: Normalized decay
    axes[1].semilogy(range(1, len(S) + 1), S / S[0], 's-', color='#ff7f0e',
                     markersize=8, linewidth=2)
    axes[1].axhline(y=1e-4, linestyle='--', color='red', alpha=0.7, label='10⁻⁴')
    axes[1].axhline(y=1e-8, linestyle=':', color='darkred', alpha=0.7, label='10⁻⁸')
    axes[1].axhline(y=1e-12, linestyle='-.', color='purple', alpha=0.7, label='10⁻¹²')
    axes[1].set_title('Normalized Decay σ_k/σ_1', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Index k')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, len(S) + 1))

    # Plot 3: Cumulative energy captured
    axes[2].plot(range(1, len(energy_cumulative) + 1),
                 energy_cumulative * 100, 'D-', color='#2ca02c',
                 markersize=8, linewidth=2)
    axes[2].axhline(y=99.99, linestyle='--', color='red', alpha=0.7, label='99.99%')
    axes[2].axhline(y=99.9999, linestyle=':', color='darkred', alpha=0.7, label='99.9999%')
    axes[2].set_title('Cumulative Energy Captured (%)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Number of vectors k')
    axes[2].set_ylabel('%')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(1, len(energy_cumulative) + 1))
    axes[2].set_ylim(bottom=min(energy_cumulative) * 100 - 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "svd_spectrum_u235_mt18.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSpectrum plot saved to: {plot_path}")
    plt.close()

    # Additional plot: reconstruction error vs k
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(errors_frobenius) + 1), errors_frobenius,
                'o-', markersize=8, linewidth=2, color='#d62728')
    ax.axhline(y=1e-8, linestyle='--', color='green', alpha=0.7,
               label='Target: ε < 10⁻⁸')
    ax.set_title('SVD Reconstruction Error vs. Truncation Rank',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Truncation rank k')
    ax.set_ylabel('Relative Frobenius error')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(errors_frobenius) + 1))

    plot_path2 = os.path.join(output_dir, "svd_error_vs_rank_u235_mt18.png")
    plt.savefig(plot_path2, dpi=200, bbox_inches='tight')
    print(f"Error plot saved to: {plot_path2}")
    plt.close()


def main():
    output_dir = os.path.expanduser("~/madman_svd_experiment/outputs")
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""

    # Load Phase 1 data
    A_log, energies, temperatures = load_phase1_data(output_dir, prefix)

    # Perform SVD
    U, S, Vt = perform_svd(A_log)

    # Analyze spectrum
    scenario, energy_cumulative, errors_frobenius, results = analyze_spectrum(
        S, A_log
    )

    # Generate plots
    plot_spectrum(S, energy_cumulative, errors_frobenius, output_dir)

    # Save SVD factors
    np.save(os.path.join(output_dir, f"{prefix}svd_U_u235.npy"), U)
    np.save(os.path.join(output_dir, f"{prefix}svd_S_u235.npy"), S)
    np.save(os.path.join(output_dir, f"{prefix}svd_Vt_u235.npy"), Vt)
    print(f"\nSVD factors saved to {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"Scenario: {scenario}")
    print(f"σ_2/σ_1 = {S[1]/S[0]:.6e}")
    print(f"Singular values: {S}")

    target_k, target_comp = results.get(1e-8, (None, None))
    if target_k is not None:
        print(f"\nFor ε < 10⁻⁸: k={target_k}, compression={target_comp*100:.1f}%")
        if target_k <= 5 and target_comp > 0.9:
            print("→ VERDICT: SVD pure is sufficient. Architecture VALIDATED.")
        elif target_k <= 20 and target_comp > 0.8:
            print("→ VERDICT: SVD viable with optimizations.")
        else:
            print("→ VERDICT: SVD alone insufficient. Hybrid approach needed.")
    else:
        print("\nε < 10⁻⁸ not achievable — full rank needed.")
        print("→ VERDICT: Pure SVD not viable for this precision. "
              "Try windowed SVD or hybrid WMP+SVD.")

    return U, S, Vt


if __name__ == "__main__":
    main()
