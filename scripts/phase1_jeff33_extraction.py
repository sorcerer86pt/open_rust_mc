# SPDX-License-Identifier: MIT
"""
Phase 1 (JEFF 3.3) — Extract U-235 fission cross-sections from both
JEFF 3.3 HDF5 libraries and merge into a single matrix with all
unique temperatures.

OpenMC-produced: 250K, 293.6K, 600K, 900K, 1200K, 2500K
NEA-converted:   293.6K, 600K, 900K, 1200K, 1500K, 1800K
Combined:        250K, 293.6K, 600K, 900K, 1200K, 1500K, 1800K, 2500K  (8 cols)

At overlapping temperatures we cross-validate that both libraries agree,
then keep one copy.

Usage:
    python scripts/phase1_jeff33_extraction.py <openmc_U235.h5> <nea_U235.h5>
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc.data


def load_nuclide(h5_path: str):
    """Load U-235 IncidentNeutron from HDF5."""
    print(f"Loading: {h5_path}")
    u235 = openmc.data.IncidentNeutron.from_hdf5(h5_path)
    temps = sorted(u235.temperatures, key=lambda t: float(t.rstrip('K')))
    print(f"  Temperatures: {temps}")
    print(f"  Reactions: {list(u235.reactions.keys())[:10]}...")
    return u235, temps


def build_union_grid(*nuclides_and_temps):
    """Build unionized energy grid across all nuclides and temperatures."""
    all_energies = []
    for u235, temps in nuclides_and_temps:
        for T in temps:
            all_energies.append(u235.energy[T])
    grid = np.unique(np.concatenate(all_energies))
    print(f"\nUnionized energy grid: {len(grid)} points")
    print(f"Energy range: {grid[0]:.4e} – {grid[-1]:.4e} eV")
    return grid


def cross_validate(u235_a, u235_b, overlap_temps, energies, mt=18):
    """Check that overlapping temperatures agree between both libraries."""
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION at overlapping temperatures (MT={mt})")
    print(f"{'='*60}")

    rxn_a = u235_a.reactions[mt]
    rxn_b = u235_b.reactions[mt]

    for T in overlap_temps:
        sigma_a = rxn_a.xs[T](energies)
        sigma_b = rxn_b.xs[T](energies)
        # Avoid division by zero
        mask = sigma_a > 1e-20
        rel_diff = np.abs(sigma_a[mask] - sigma_b[mask]) / sigma_a[mask]
        print(f"  T={T}: max_rel_diff = {rel_diff.max():.2e}, "
              f"mean_rel_diff = {rel_diff.mean():.2e}")
        if rel_diff.max() > 0.01:
            print(f"    ⚠ WARNING: >1% disagreement at {T}!")


def build_merged_matrix(u235_openmc, temps_openmc,
                        u235_nea, temps_nea,
                        energies, mt=18):
    """Build the merged σ(E,T) matrix from both libraries."""
    # Determine unique temperatures and which library provides each
    temp_source = {}  # temp_str → (nuclide, source_name)

    for T in temps_openmc:
        temp_source[T] = (u235_openmc, "OpenMC")
    for T in temps_nea:
        if T not in temp_source:
            temp_source[T] = (u235_nea, "NEA")
        # If overlapping, keep OpenMC version (arbitrary choice)

    # Sort by temperature value
    all_temps = sorted(temp_source.keys(), key=lambda t: float(t.rstrip('K')))

    print(f"\nMerged temperature grid ({len(all_temps)} columns):")
    for T in all_temps:
        src = temp_source[T][1]
        print(f"  {T:>8s}  ← {src}")

    # Build matrix
    cols = []
    for T in all_temps:
        u235, src = temp_source[T]
        rxn = u235.reactions[mt]
        sigma = rxn.xs[T](energies)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
        print(f"  T={T} ({src}): min={sigma.min():.2e}, max={sigma.max():.2e}")

    A = np.column_stack(cols)
    return A, all_temps


def main():
    output_dir = os.path.expanduser("~/madman_svd_experiment/outputs")
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) < 3:
        print("Usage: python phase1_jeff33_extraction.py <openmc_U235.h5> <nea_U235.h5>")
        sys.exit(1)

    openmc_h5 = sys.argv[1]
    nea_h5 = sys.argv[2]

    # Load both libraries
    u235_openmc, temps_openmc = load_nuclide(openmc_h5)
    u235_nea, temps_nea = load_nuclide(nea_h5)

    # Build unionized energy grid
    energies = build_union_grid(
        (u235_openmc, temps_openmc),
        (u235_nea, temps_nea)
    )

    # Cross-validate at overlapping temperatures
    overlap = sorted(set(temps_openmc) & set(temps_nea),
                     key=lambda t: float(t.rstrip('K')))
    print(f"\nOverlapping temperatures: {overlap}")
    if overlap:
        cross_validate(u235_openmc, u235_nea, overlap, energies, mt=18)

    # Build merged matrix
    A, all_temps = build_merged_matrix(
        u235_openmc, temps_openmc,
        u235_nea, temps_nea,
        energies, mt=18
    )

    print(f"\n{'='*60}")
    print(f"Matrix A constructed: {A.shape} (N_E={A.shape[0]}, N_T={A.shape[1]})")
    print(f"Memory: {A.nbytes / 1e6:.1f} MB")
    print(f"{'='*60}")

    # Validations
    N_E, N_T = A.shape
    assert N_E >= 10000, f"N_E={N_E} too small"
    assert N_T >= 7, f"N_T={N_T} — expected ≥ 7 from merging two libraries"
    assert not np.any(np.isnan(A)), "NaN detected"
    assert not np.any(np.isinf(A)), "Inf detected"
    assert np.all(A > 0), "Non-positive values detected"
    print("CHECKPOINT 1: All validations passed ✓")

    # Log-transform
    A_log = np.log10(A)
    print(f"Log-transformed range: [{A_log.min():.2f}, {A_log.max():.2f}]")

    # Save — use jeff33 suffix to distinguish from ENDF/B-VII.1 outputs
    prefix = "jeff33_"
    np.save(os.path.join(output_dir, f"{prefix}A_raw_u235_mt18.npy"), A)
    np.save(os.path.join(output_dir, f"{prefix}A_log_u235_mt18.npy"), A_log)
    np.save(os.path.join(output_dir, f"{prefix}energies_u235.npy"), energies)
    np.save(os.path.join(output_dir, f"{prefix}temperatures_u235.npy"),
            np.array(all_temps, dtype=object))

    print(f"\nSaved to {output_dir}:")
    for name in [f"{prefix}A_raw_u235_mt18.npy", f"{prefix}A_log_u235_mt18.npy",
                 f"{prefix}energies_u235.npy", f"{prefix}temperatures_u235.npy"]:
        fpath = os.path.join(output_dir, name)
        sz = os.path.getsize(fpath) / 1e6
        print(f"  {name}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
