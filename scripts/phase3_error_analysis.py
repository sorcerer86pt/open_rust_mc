"""
Phase 3 — Point-wise physical error analysis and Dense-and-Sparse hybrid.

Analyzes reconstruction error by energy region (thermal, resonance, fast),
identifies problematic resonance points, and tests the GEAR-inspired
Dense-and-Sparse approach for error compensation.

Usage:
    python scripts/phase3_error_analysis.py [k_chosen] [prefix]

    prefix: file prefix for Phase 1/2 outputs (default: "", use "jeff33_" for JEFF data)
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt


def load_data(output_dir: str, prefix: str = ""):
    """Load all Phase 1 and Phase 2 outputs."""
    A_raw = np.load(os.path.join(output_dir, f"{prefix}A_raw_u235_mt18.npy"))
    A_log = np.load(os.path.join(output_dir, f"{prefix}A_log_u235_mt18.npy"))
    energies = np.load(os.path.join(output_dir, f"{prefix}energies_u235.npy"))
    temperatures = np.load(os.path.join(output_dir, f"{prefix}temperatures_u235.npy"),
                           allow_pickle=True)
    U = np.load(os.path.join(output_dir, f"{prefix}svd_U_u235.npy"))
    S = np.load(os.path.join(output_dir, f"{prefix}svd_S_u235.npy"))
    Vt = np.load(os.path.join(output_dir, f"{prefix}svd_Vt_u235.npy"))
    return A_raw, A_log, energies, temperatures, U, S, Vt


def reconstruct(U, S, Vt, k: int):
    """Reconstruct A_log from rank-k truncation."""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def analyze_regional_error(A_raw, A_log, A_k_log, energies, temperatures):
    """Analyze reconstruction error by energy region."""
    A_k = 10 ** A_k_log  # undo log-transform

    # Define energy regions
    regions = {
        "Thermal (<1 eV)":       energies < 1.0,
        "Resonance (1 eV-25 keV)": (energies >= 1.0) & (energies < 25000),
        "Fast (>25 keV)":        energies >= 25000,
    }

    print("=" * 80)
    print("REGIONAL ERROR ANALYSIS (linear-space relative error)")
    print("=" * 80)

    all_errors = {}
    for i, T in enumerate(temperatures):
        sigma_orig = A_raw[:, i]
        sigma_recon = A_k[:, i]
        err_rel = np.abs(sigma_orig - sigma_recon) / (sigma_orig + 1e-30)

        all_errors[str(T)] = err_rel

        print(f"\n--- Temperature: {T} ---")
        for name, mask in regions.items():
            if np.sum(mask) == 0:
                print(f"  {name}: no points in this region")
                continue
            e = err_rel[mask]
            print(f"  {name}:")
            print(f"    max_err  = {e.max():.2e}")
            print(f"    mean_err = {e.mean():.2e}")
            print(f"    P95      = {np.percentile(e, 95):.2e}")
            print(f"    P99      = {np.percentile(e, 99):.2e}")
            print(f"    points   = {np.sum(mask)}")

    return all_errors, regions


def identify_worst_points(A_raw, A_k_log, energies, k_chosen, n_worst=50):
    """Find the worst reconstruction points in the resonance region."""
    A_k = 10 ** A_k_log
    mask_resonance = (energies >= 1.0) & (energies < 25000)

    print(f"\n{'='*80}")
    print(f"TOP {n_worst} WORST RECONSTRUCTION POINTS (Resonance Region)")
    print(f"{'='*80}")

    # Average relative error across all temperatures
    err_avg = np.zeros(A_raw.shape[0])
    for i in range(A_raw.shape[1]):
        err_avg += np.abs(A_raw[:, i] - A_k[:, i]) / (A_raw[:, i] + 1e-30)
    err_avg /= A_raw.shape[1]

    err_resonance = err_avg[mask_resonance]
    energies_resonance = energies[mask_resonance]

    worst_indices = np.argsort(err_resonance)[-n_worst:][::-1]
    worst_energies = energies_resonance[worst_indices]
    worst_errors = err_resonance[worst_indices]

    # Known dominant U-235 resonances
    known_resonances = [0.27, 1.14, 2.03, 3.61, 4.84, 6.21, 8.78,
                        11.66, 12.39, 19.30, 21.07]

    print(f"\n{'Energy (eV)':>15}  {'Avg Rel Error':>15}  {'Near Known Resonance?':>25}")
    print("-" * 60)
    for e, err in zip(worst_energies, worst_errors):
        # Check if near a known resonance
        near = ""
        for kr in known_resonances:
            if abs(e - kr) / kr < 0.1:  # within 10%
                near = f"~{kr} eV"
                break
        print(f"{e:>15.4f}  {err:>15.2e}  {near:>25}")

    return worst_energies, worst_errors


def test_dense_and_sparse(A_log, A_k_log, A_raw, energies,
                          sparse_percentiles=(90, 95, 97, 99)):
    """Test Dense-and-Sparse (GEAR) hybrid approach at various sparsity levels."""
    print(f"\n{'='*80}")
    print("DENSE-AND-SPARSE (GEAR) HYBRID ANALYSIS")
    print(f"{'='*80}")

    R = A_log - A_k_log  # residual matrix in log-space
    N_E, N_T = A_log.shape

    results = []
    for pct in sparse_percentiles:
        threshold = np.percentile(np.abs(R), pct)
        R_sparse = np.where(np.abs(R) > threshold, R, 0)

        A_hybrid_log = A_k_log + R_sparse
        A_hybrid = 10 ** A_hybrid_log

        # Error analysis
        err_hybrid = np.abs(A_raw - A_hybrid) / (A_raw + 1e-30)
        err_svd_only = np.abs(A_raw - 10 ** A_k_log) / (A_raw + 1e-30)

        # Size computation
        sparse_nnz = np.count_nonzero(R_sparse)
        k = A_k_log.shape[1] if A_k_log.ndim > 1 else 1
        # For SVD: store U[:,:k], S[:k], Vt[:k,:] + sparse indices and values
        # Approximate: k*(N_E + N_T + 1) + 3*sparse_nnz (row, col, value)
        size_svd = k * (N_E + N_T + 1)
        size_sparse = 3 * sparse_nnz  # (row_idx, col_idx, value)
        size_hybrid = size_svd + size_sparse
        size_original = N_E * N_T
        compression = 1 - size_hybrid / size_original

        print(f"\n--- Sparse threshold: top {100-pct:.0f}% of residuals ---")
        print(f"  Threshold value: {threshold:.2e}")
        print(f"  Non-zero residuals: {sparse_nnz} ({sparse_nnz/R.size*100:.1f}%)")
        print(f"  SVD-only  max error: {err_svd_only.max():.2e}")
        print(f"  Hybrid    max error: {err_hybrid.max():.2e}")
        print(f"  SVD-only  mean error: {err_svd_only.mean():.2e}")
        print(f"  Hybrid    mean error: {err_hybrid.mean():.2e}")
        print(f"  Compression: {compression*100:.1f}%")

        results.append({
            'percentile': pct,
            'sparse_nnz': sparse_nnz,
            'max_err_svd': err_svd_only.max(),
            'max_err_hybrid': err_hybrid.max(),
            'mean_err_svd': err_svd_only.mean(),
            'mean_err_hybrid': err_hybrid.mean(),
            'compression': compression,
        })

    return results


def plot_error_map(A_raw, A_k_log, energies, temperatures, output_dir, k_chosen):
    """Plot error heatmap across energy and temperature."""
    A_k = 10 ** A_k_log
    err_rel = np.abs(A_raw - A_k) / (A_raw + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Error vs energy for each temperature
    for i, T in enumerate(temperatures):
        axes[0].semilogy(energies, err_rel[:, i], alpha=0.6, linewidth=0.5,
                         label=str(T))
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Energy (eV)', fontsize=12)
    axes[0].set_ylabel('Relative Error', fontsize=12)
    axes[0].set_title(f'SVD Reconstruction Error (k={k_chosen})', fontsize=13,
                      fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1e-8, linestyle='--', color='red', alpha=0.5,
                    label='Target 10⁻⁸')

    # Mark energy regions
    for ax in axes[:1]:
        ax.axvline(x=1.0, linestyle=':', color='gray', alpha=0.5)
        ax.axvline(x=25000, linestyle=':', color='gray', alpha=0.5)
        ax.text(0.1, ax.get_ylim()[1] * 0.5, 'Thermal', fontsize=8,
                ha='center', color='gray')
        ax.text(150, ax.get_ylim()[1] * 0.5, 'Resonance', fontsize=8,
                ha='center', color='gray')
        ax.text(1e6, ax.get_ylim()[1] * 0.5, 'Fast', fontsize=8,
                ha='center', color='gray')

    # Cross-section comparison (first temperature)
    axes[1].loglog(energies, A_raw[:, 0], 'b-', linewidth=0.5, alpha=0.7,
                   label=f'Original ({temperatures[0]})')
    axes[1].loglog(energies, A_k[:, 0], 'r--', linewidth=0.5, alpha=0.7,
                   label=f'SVD k={k_chosen}')
    axes[1].set_xlabel('Energy (eV)', fontsize=12)
    axes[1].set_ylabel('Cross Section (barns)', fontsize=12)
    axes[1].set_title(f'Original vs Reconstructed ({temperatures[0]})',
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"error_analysis_k{k_chosen}_u235_mt18.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\nError analysis plot saved to: {path}")
    plt.close()


def main():
    output_dir = os.path.expanduser("~/madman_svd_experiment/outputs")
    prefix = ""

    # Parse args: [k_chosen] [prefix]
    args = sys.argv[1:]
    k_chosen = None
    for arg in args:
        if arg.endswith("_"):
            prefix = arg
        else:
            try:
                k_chosen = int(arg)
            except ValueError:
                prefix = arg

    if k_chosen is None:
        S = np.load(os.path.join(output_dir, f"{prefix}svd_S_u235.npy"))
        # Auto-select: smallest k that captures 99.99% of energy
        S2 = S ** 2
        energy_cum = np.cumsum(S2) / np.sum(S2)
        k_chosen = int(np.argmax(energy_cum >= 0.9999)) + 1
        k_chosen = max(k_chosen, 2)  # at least 2
        print(f"Auto-selected k={k_chosen} (captures {energy_cum[k_chosen-1]*100:.6f}% energy)")

    # Load data
    A_raw, A_log, energies, temperatures, U, S, Vt = load_data(output_dir, prefix)
    print(f"\nUsing truncation rank k={k_chosen}")

    # Reconstruct
    A_k_log = reconstruct(U, S, Vt, k_chosen)

    # Phase 3.1: Regional error analysis
    all_errors, regions = analyze_regional_error(
        A_raw, A_log, A_k_log, energies, temperatures
    )

    # Phase 3.2: Identify worst points
    worst_energies, worst_errors = identify_worst_points(
        A_raw, A_k_log, energies, k_chosen
    )

    # Phase 3.3: Dense-and-Sparse
    sparse_results = test_dense_and_sparse(A_log, A_k_log, A_raw, energies)

    # Generate plots
    plot_error_map(A_raw, A_k_log, energies, temperatures, output_dir, k_chosen)

    # Checkpoint 3 evaluation
    print(f"\n{'='*80}")
    print("CHECKPOINT 3 — EVALUATION")
    print(f"{'='*80}")

    best_hybrid = min(sparse_results, key=lambda r: r['max_err_hybrid'])
    print(f"\nBest hybrid configuration: top {100-best_hybrid['percentile']:.0f}% residuals")
    print(f"  Max error:    {best_hybrid['max_err_hybrid']:.2e} "
          f"(target: < 10⁻⁸)")
    print(f"  Compression:  {best_hybrid['compression']*100:.1f}% "
          f"(target: > 80%)")

    err_ok = best_hybrid['max_err_hybrid'] < 1e-8
    comp_ok = best_hybrid['compression'] > 0.8
    print(f"\n  Error < 10⁻⁸?       {'YES' if err_ok else 'NO'}")
    print(f"  Compression > 80%?  {'YES' if comp_ok else 'NO'}")

    if err_ok and comp_ok:
        print("\n→ CHECKPOINT 3 PASSED. Ready for k_eff benchmark (Phase 4).")
    elif comp_ok:
        print("\n→ Error too high. Try increasing k or adjusting sparse threshold.")
        print("  Consider windowed SVD (Phase 5.3) for better regional accuracy.")
    else:
        print("\n→ Compression insufficient. The data may have high effective rank.")
        print("  Consider WMP for resonance region + SVD for smooth regions.")

    return sparse_results


if __name__ == "__main__":
    main()
