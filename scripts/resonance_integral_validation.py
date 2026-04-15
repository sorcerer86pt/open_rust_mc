"""
Resonance Integral Validation — SVD vs pointwise reference.

Computes infinite-dilution resonance integrals (∫σ(E)dE/E) over standard
energy groups for SVD-reconstructed cross-sections vs the original data.
This is a more physically meaningful accuracy metric than pointwise max error.

Run in WSL with: conda activate openmc && python resonance_integral_validation.py
"""

import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import openmc.data
except ImportError:
    print("ERROR: openmc not available. Run in WSL with conda activate openmc")
    sys.exit(1)

from scipy.linalg import svd as scipy_svd

DATA_DIR = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron"

# Standard energy group boundaries (eV) — CASMO 70-group-like structure simplified
ENERGY_GROUPS = [
    ("Ultra-cold",      1e-5,    1e-3),
    ("Cold thermal",    1e-3,    0.0253),
    ("Thermal",         0.0253,  0.625),
    ("Epithermal low",  0.625,   4.0),
    ("Low resonance",   4.0,     100.0),
    ("Mid resonance",   100.0,   1000.0),
    ("High resonance",  1000.0,  10000.0),
    ("URR",             10000.0, 100000.0),
    ("Fast low",        100000.0, 1e6),
    ("Fast high",       1e6,     20e6),
]


def resonance_integral(energies, xs, e_lo, e_hi):
    """Compute ∫σ(E)dE/E over [e_lo, e_hi] using trapezoidal rule."""
    mask = (energies >= e_lo) & (energies <= e_hi)
    e = energies[mask]
    s = xs[mask]
    if len(e) < 2:
        return 0.0
    # Integrand: σ(E)/E
    integrand = s / e
    return np.trapz(integrand, e)


def svd_reconstruct(energies, xs_matrix_log, rank):
    """SVD reconstruction at given rank, returns linear-scale XS per temperature."""
    U, S, Vt = scipy_svd(xs_matrix_log, full_matrices=False)
    k = min(rank, len(S))
    recon_log = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return 10 ** recon_log


def analyze_nuclide(nuclide_name, mt, label):
    """Analyze resonance integrals for one nuclide/reaction."""
    h5_path = os.path.join(DATA_DIR, f"{nuclide_name}.h5")
    if not os.path.exists(h5_path):
        print(f"  WARNING: {h5_path} not found, skipping")
        return

    nuc = openmc.data.IncidentNeutron.from_hdf5(h5_path)

    if mt not in nuc.reactions:
        print(f"  WARNING: MT={mt} not in {nuclide_name}, skipping")
        return

    rxn = nuc.reactions[mt]
    temps = sorted(
        [t for t in nuc.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )

    # Use 294K (or first available)
    temp = '294K' if '294K' in temps else temps[0]
    temp_idx = temps.index(temp)

    # Build unionized grid and XS matrix
    all_e = [nuc.energy[T] for T in temps]
    energy_union = np.unique(np.concatenate(all_e))

    cols = []
    for T in temps:
        sigma = rxn.xs[T](energy_union)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
    A = np.column_stack(cols)

    # Reference: original XS at this temperature
    ref_xs = A[:, temp_idx]

    print(f"\n{'='*70}")
    print(f"  {nuclide_name} — {label} (MT={mt}), T={temp}")
    print(f"  Energy grid: {len(energy_union)} points, {len(temps)} temperatures")
    print(f"{'='*70}")

    header = f"  {'Group':<20} {'E range':<20} {'Ref RI':>12} {'k=3 RI':>12} {'k=3 err%':>10} {'k=5 RI':>12} {'k=5 err%':>10}"
    print(header)
    print(f"  {'-'*20} {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for rank in [3, 5]:
        A_log = np.log10(A)
        recon = svd_reconstruct(energy_union, A_log, rank)
        svd_xs = recon[:, temp_idx]

        if rank == 3:
            svd_xs_k3 = svd_xs
        else:
            svd_xs_k5 = svd_xs

    for name, e_lo, e_hi in ENERGY_GROUPS:
        ri_ref = resonance_integral(energy_union, ref_xs, e_lo, e_hi)
        ri_k3 = resonance_integral(energy_union, svd_xs_k3, e_lo, e_hi)
        ri_k5 = resonance_integral(energy_union, svd_xs_k5, e_lo, e_hi)

        err_k3 = abs(ri_k3 - ri_ref) / ri_ref * 100 if ri_ref > 0 else 0
        err_k5 = abs(ri_k5 - ri_ref) / ri_ref * 100 if ri_ref > 0 else 0

        e_range = f"{e_lo:.2g}–{e_hi:.2g} eV"
        print(f"  {name:<20} {e_range:<20} {ri_ref:>12.4f} {ri_k3:>12.4f} {err_k3:>9.3f}% {ri_k5:>12.4f} {err_k5:>9.3f}%")


def main():
    print("Resonance Integral Validation — SVD vs Pointwise Reference")
    print("Metric: ∫σ(E)dE/E over standard energy groups\n")

    nuclides = [
        ("U235", 18, "Fission"),
        ("U235", 2, "Elastic"),
        ("U235", 102, "Capture"),
        ("U238", 18, "Fission"),
        ("U238", 2, "Elastic"),
        ("U238", 102, "Capture"),
    ]

    for nuc_name, mt, label in nuclides:
        analyze_nuclide(nuc_name, mt, label)


if __name__ == "__main__":
    main()
