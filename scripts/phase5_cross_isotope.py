"""
Phase 5.5 — Cross-isotope basis sharing analysis.

Check if U-235, U-238, and Pu-239 share singular vectors for fission (MT=18).
If ||U_U235 - U_U238||_F is small → shared basis is possible → additional
compression of 30-50% (the "MiniCache" analogy from LLM KV-cache compression).
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc.data
from scipy.linalg import svd

DATA_DIR = os.path.expanduser(
    "~/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron"
)


def load_and_svd(h5_path, mt=18):
    """Load nuclide, build fission matrix, return SVD on common grid."""
    u = openmc.data.IncidentNeutron.from_hdf5(h5_path)
    name = os.path.basename(h5_path).replace('.h5', '')

    if mt not in u.reactions:
        return None

    temps = sorted(
        [t for t in u.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )
    rxn = u.reactions[mt]

    all_e = [u.energy[T] for T in temps]
    energies = np.unique(np.concatenate(all_e))

    cols = []
    for T in temps:
        sigma = rxn.xs[T](energies)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
    A = np.column_stack(cols)
    A_log = np.log10(A)

    U, S, Vt = svd(A_log, full_matrices=False)

    print(f"  {name}: N_E={len(energies)}, S={S[:4]}...")
    return {
        'name': name, 'energies': energies, 'A_log': A_log,
        'U': U, 'S': S, 'Vt': Vt, 'temps': temps,
    }


def compare_bases(r1, r2, max_k=5):
    """Compare singular vector subspaces between two nuclides."""
    name1, name2 = r1['name'], r2['name']

    # To compare U vectors, we need them on the same energy grid.
    # Find the common energy grid (intersection).
    e1 = set(r1['energies'])
    e2 = set(r2['energies'])
    common = sorted(e1 & e2)
    n_common = len(common)

    if n_common < 1000:
        print(f"\n  {name1} vs {name2}: only {n_common} common energy points, skipping.")
        return None

    # Map common indices
    common_arr = np.array(common)
    idx1 = np.searchsorted(r1['energies'], common_arr)
    idx2 = np.searchsorted(r2['energies'], common_arr)

    # Clamp indices
    idx1 = np.clip(idx1, 0, len(r1['energies']) - 1)
    idx2 = np.clip(idx2, 0, len(r2['energies']) - 1)

    print(f"\n  {name1} vs {name2} ({n_common} common energy points):")
    print(f"  {'k':>4}  {'Subspace angle (deg)':>22}  {'Frobenius ||U1-U2||':>22}  {'Cosine sim (col 1)':>20}")

    results = []
    for k in range(1, min(max_k + 1, min(r1['U'].shape[1], r2['U'].shape[1]) + 1)):
        U1_k = r1['U'][idx1, :k]
        U2_k = r2['U'][idx2, :k]

        # Normalize columns
        for j in range(k):
            U1_k[:, j] /= np.linalg.norm(U1_k[:, j])
            U2_k[:, j] /= np.linalg.norm(U2_k[:, j])

        # Frobenius distance
        frob = np.linalg.norm(U1_k - U2_k, 'fro')

        # Principal angle between subspaces (via SVD of U1^T @ U2)
        M = U1_k.T @ U2_k
        svals = np.linalg.svd(M, compute_uv=False)
        # Clamp to [-1, 1] for numerical safety
        svals = np.clip(svals, -1, 1)
        angles_rad = np.arccos(svals)
        max_angle_deg = np.degrees(angles_rad.max())

        # Cosine similarity of first column
        cos_sim = abs(np.dot(U1_k[:, 0], U2_k[:, 0]))

        print(f"  {k:>4}  {max_angle_deg:>22.2f}  {frob:>22.4f}  {cos_sim:>20.6f}")
        results.append({
            'k': k, 'angle_deg': max_angle_deg, 'frob': frob, 'cos_sim': cos_sim,
        })

    return results


def test_cross_reconstruction(r_source, r_target, k=4):
    """Use source's U basis to reconstruct target's cross-sections."""
    name_s, name_t = r_source['name'], r_target['name']

    # Common energy grid
    e_s = set(r_source['energies'])
    e_t = set(r_target['energies'])
    common = sorted(e_s & e_t)
    if len(common) < 1000:
        return

    common_arr = np.array(common)
    idx_s = np.clip(np.searchsorted(r_source['energies'], common_arr), 0, len(r_source['energies'])-1)
    idx_t = np.clip(np.searchsorted(r_target['energies'], common_arr), 0, len(r_target['energies'])-1)

    U_source = r_source['U'][idx_s, :k]
    A_target_log = r_target['A_log'][idx_t, :]

    # Project target onto source's basis: coeffs = U_source^T @ A_target_log
    coeffs = np.linalg.lstsq(U_source, A_target_log, rcond=None)[0]
    A_recon_log = U_source @ coeffs
    A_recon = 10**A_recon_log
    A_orig = 10**A_target_log

    err = np.abs(A_orig - A_recon) / (A_orig + 1e-30)

    print(f"\n  Cross-reconstruction: {name_s}'s basis (k={k}) → {name_t}'s data")
    print(f"    Max error:  {err.max():.2e}")
    print(f"    Mean error: {err.mean():.2e}")
    print(f"    P99 error:  {np.percentile(err, 99):.2e}")

    # Compare with target's own basis
    U_own = r_target['U'][idx_t, :k]
    S_own = r_target['S'][:k]
    Vt_own = r_target['Vt'][:k, :]
    A_own_recon_log = U_own @ np.diag(S_own) @ Vt_own
    A_own_recon = 10**A_own_recon_log
    err_own = np.abs(A_orig - A_own_recon) / (A_orig + 1e-30)
    print(f"    {name_t}'s own basis (k={k}): max={err_own.max():.2e}, mean={err_own.mean():.2e}")
    print(f"    Overhead of sharing: {err.max()/max(err_own.max(), 1e-30):.1f}x")


def main():
    nuclides = ['U235', 'U238', 'Pu239']
    h5_files = {n: os.path.join(DATA_DIR, f"{n}.h5") for n in nuclides}

    print("Loading nuclides and computing SVD (MT=18 fission)...\n")
    results = {}
    for name, path in h5_files.items():
        if os.path.exists(path):
            r = load_and_svd(path, mt=18)
            if r:
                results[name] = r
        else:
            print(f"  {path} not found, skipping.")

    if len(results) < 2:
        print("Need at least 2 nuclides for cross-isotope analysis.")
        return

    # Pairwise basis comparison
    print(f"\n{'='*70}")
    print("CROSS-ISOTOPE BASIS COMPARISON")
    print(f"{'='*70}")

    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            compare_bases(results[names[i]], results[names[j]])

    # Cross-reconstruction test
    print(f"\n{'='*70}")
    print("CROSS-ISOTOPE RECONSTRUCTION TEST")
    print(f"{'='*70}")

    for k in [3, 4, 5]:
        print(f"\n--- k={k} ---")
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    test_cross_reconstruction(results[names[i]], results[names[j]], k=k)


if __name__ == "__main__":
    main()
