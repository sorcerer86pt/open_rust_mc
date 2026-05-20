# SPDX-License-Identifier: MIT
"""
Phase 5.1 — SVD analysis for MT=2 (elastic) and MT=102 (capture),
plus MT=18 (fission) for comparison. All on ENDF/B-VII.1 U-235.
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc.data
from scipy.linalg import svd

H5_PATH = os.path.expanduser(
    "~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5"
)
OUTPUT_DIR = os.path.expanduser("~/madman_svd_experiment/outputs")


def analyze_reaction(u235, mt, label):
    """Full SVD analysis for one reaction."""
    reaction = u235.reactions[mt]
    temperatures = sorted(
        [t for t in u235.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )

    # Unionized energy grid
    all_e = [u235.energy[T] for T in temperatures]
    energies = np.unique(np.concatenate(all_e))
    n_e = len(energies)
    n_t = len(temperatures)

    # Build matrix
    cols = []
    for T in temperatures:
        sigma = reaction.xs[T](energies)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
    A = np.column_stack(cols)
    A_log = np.log10(A)

    # SVD
    U, S, Vt = svd(A_log, full_matrices=False)

    print(f"\n{'='*70}")
    print(f"MT={mt} — {label}")
    print(f"{'='*70}")
    print(f"  Matrix: {n_e} × {n_t}")
    print(f"  Singular values: {S}")
    print(f"  σ_2/σ_1 = {S[1]/S[0]:.4e}")

    # Cumulative energy
    S2 = S**2
    cum = np.cumsum(S2) / np.sum(S2)
    for k, c in enumerate(cum):
        print(f"    k={k+1}: {c*100:.8f}%")

    # Reconstruction error per region per k
    print("\n  Reconstruction accuracy (max relative error, linear scale):")
    print(f"  {'k':>4}  {'Thermal':>12}  {'Resonance':>12}  {'Fast':>12}  {'Overall':>12}")

    mask_th = energies < 1.0
    mask_res = (energies >= 1.0) & (energies < 25000)
    mask_fast = energies >= 25000

    for k in range(2, len(S) + 1):
        A_k_log = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        A_k = 10**A_k_log
        err = np.abs(A - A_k) / (A + 1e-30)

        e_th = err[mask_th].max() if mask_th.any() else 0
        e_res = err[mask_res].max() if mask_res.any() else 0
        e_fast = err[mask_fast].max() if mask_fast.any() else 0
        e_all = err.max()

        print(f"  {k:>4}  {e_th:>12.2e}  {e_res:>12.2e}  {e_fast:>12.2e}  {e_all:>12.2e}")

    return {
        'mt': mt, 'label': label, 'n_e': n_e, 'n_t': n_t,
        'S': S, 'ratio': S[1]/S[0],
        'U': U, 'Vt': Vt, 'energies': energies,
    }


def main():
    print("Loading U-235...")
    u235 = openmc.data.IncidentNeutron.from_hdf5(H5_PATH)

    reactions = [
        (18,  "(n,f) Fission"),
        (2,   "(n,n) Elastic"),
        (102, "(n,γ) Capture"),
    ]

    results = {}
    for mt, label in reactions:
        if mt in u235.reactions:
            results[mt] = analyze_reaction(u235, mt, label)
        else:
            print(f"\n  MT={mt} not available, skipping.")

    # Summary table
    print(f"\n{'='*70}")
    print("MULTI-REACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'MT':>4}  {'Reaction':>20}  {'N_E':>8}  {'σ_2/σ_1':>12}  {'k for 99.99%':>14}")
    for mt, r in results.items():
        cum = np.cumsum(r['S']**2) / np.sum(r['S']**2)
        k99 = int(np.argmax(cum >= 0.9999)) + 1
        print(f"  {mt:>4}  {r['label']:>20}  {r['n_e']:>8}  {r['ratio']:>12.4e}  {k99:>14}")


if __name__ == "__main__":
    main()
