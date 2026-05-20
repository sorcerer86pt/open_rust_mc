# SPDX-License-Identifier: MIT
"""
Sweep ALL reaction types in U-235 — singular spectrum analysis for each.
Produces a comprehensive table of which reactions are SVD-compressible.
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc.data
from scipy.linalg import svd

H5 = os.path.expanduser(
    "~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron/U235.h5"
)

# ENDF MT descriptions for common reactions
MT_NAMES = {
    1: "total", 2: "(n,n) elastic", 3: "nonelastic", 4: "(n,n') total",
    16: "(n,2n)", 17: "(n,3n)", 18: "(n,f) fission", 37: "(n,4n)",
    51: "(n,n'1)", 52: "(n,n'2)", 53: "(n,n'3)", 54: "(n,n'4)",
    55: "(n,n'5)", 56: "(n,n'6)", 57: "(n,n'7)", 58: "(n,n'8)",
    91: "(n,n'c)", 102: "(n,g) capture", 301: "heating", 444: "damage",
}


def main():
    u235 = openmc.data.IncidentNeutron.from_hdf5(H5)
    temps = sorted(
        [t for t in u235.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )

    all_e = [u235.energy[T] for T in temps]
    energies = np.unique(np.concatenate(all_e))
    n_t = len(temps)

    print(f"U-235: {len(u235.reactions)} reactions, {n_t} temperatures, "
          f"{len(energies)} union energy points\n")

    print(f"{'MT':>4}  {'Reaction':<20}  {'sigma2/sigma1':>14}  {'k(99.99%)':>10}  "
          f"{'Max err k=3':>12}  {'Max err k=5':>12}  {'Scenario':>10}")
    print("-" * 100)

    results = []
    for mt in sorted(u235.reactions.keys()):
        rxn = u235.reactions[mt]
        name = MT_NAMES.get(int(mt), f"MT={mt}")

        try:
            cols = []
            for T in temps:
                sigma = rxn.xs[T](energies)
                sigma = np.where(sigma > 0, sigma, 1e-30)
                cols.append(sigma)
            A = np.column_stack(cols)

            # Skip if all-zero
            if A.max() < 1e-20:
                continue

            A_log = np.log10(A)
            U, S, Vt = svd(A_log, full_matrices=False)

            ratio = S[1] / S[0] if len(S) > 1 else 0
            cum = np.cumsum(S**2) / np.sum(S**2)
            k99 = int(np.argmax(cum >= 0.9999)) + 1

            # Max error at k=3 and k=5
            errs = {}
            for k in [3, 5]:
                k_use = min(k, len(S))
                A_k = 10 ** (U[:, :k_use] @ np.diag(S[:k_use]) @ Vt[:k_use, :])
                err = np.abs(A - A_k) / (A + 1e-30)
                errs[k] = err.max()

            if ratio < 0.01:
                scenario = "A (easy)"
            elif ratio < 0.1:
                scenario = "B (good)"
            else:
                scenario = "C (hard)"

            print(f"{int(mt):>4}  {name:<20}  {ratio:>14.4e}  {k99:>10}  "
                  f"{errs[3]:>12.2e}  {errs[5]:>12.2e}  {scenario:>10}")

            results.append({
                'mt': int(mt), 'name': name, 'ratio': ratio,
                'k99': k99, 'err_k3': errs[3], 'err_k5': errs[5],
                'scenario': scenario, 'S': S,
            })
        except Exception as e:
            print(f"{int(mt):>4}  {name:<20}  FAILED: {e}")

    # Summary
    easy = sum(1 for r in results if r['ratio'] < 0.01)
    good = sum(1 for r in results if 0.01 <= r['ratio'] < 0.1)
    hard = sum(1 for r in results if r['ratio'] >= 0.1)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(results)} reactions analysed")
    print(f"  Scenario A (easy, sigma2/sigma1 < 0.01): {easy} reactions")
    print(f"  Scenario B (good, 0.01-0.1):             {good} reactions")
    print(f"  Scenario C (hard, > 0.1):                {hard} reactions")
    print(f"\n  All reactions are SVD-compressible to k<=5 with < 2% error: "
          f"{'YES' if all(r['err_k5'] < 0.02 for r in results) else 'NO'}")

    worst = max(results, key=lambda r: r['err_k5'])
    best = min(results, key=lambda r: r['ratio'])
    print(f"  Easiest: MT={best['mt']} {best['name']} (sigma2/sigma1 = {best['ratio']:.2e})")
    print(f"  Hardest: MT={worst['mt']} {worst['name']} (max err k=5 = {worst['err_k5']:.2e})")


if __name__ == "__main__":
    main()
