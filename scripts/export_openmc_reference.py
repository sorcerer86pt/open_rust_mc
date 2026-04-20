"""
Export OpenMC reference cross-section values for validation against Rust.

Loads U235.h5 via openmc.data (the same API OpenMC uses internally),
evaluates σ(E,T) at the unionized energy grid, and saves as .npy files
that the Rust validator can load.
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc.data


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_openmc_reference.py <U235.h5> [output_dir]")
        sys.exit(1)

    h5_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/madman_svd_experiment/outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {h5_path} via openmc.data...")
    u235 = openmc.data.IncidentNeutron.from_hdf5(h5_path)

    temperatures = sorted(u235.temperatures, key=lambda t: float(t.rstrip('K')))
    # Filter out 0K
    temperatures = [t for t in temperatures if float(t.rstrip('K')) > 0]
    print(f"Temperatures: {temperatures}")

    reaction = u235.reactions[18]  # fission

    # Build unionized energy grid (same logic as the Rust code)
    all_energies = []
    for T in temperatures:
        all_energies.append(u235.energy[T])
    energy_union = np.unique(np.concatenate(all_energies))
    n_e = len(energy_union)
    n_t = len(temperatures)
    print(f"Unionized grid: {n_e} points")

    # Evaluate OpenMC cross-sections at each temperature on the union grid
    # This is what OpenMC does internally during transport
    ref_matrix = np.zeros((n_e, n_t), dtype=np.float64)
    for t_idx, T in enumerate(temperatures):
        xs_func = reaction.xs[T]
        ref_matrix[:, t_idx] = xs_func(energy_union)
        print(f"  {T}: min={ref_matrix[:, t_idx].min():.2e}, "
              f"max={ref_matrix[:, t_idx].max():.2e}")

    # Save reference data
    ref_path = os.path.join(output_dir, "openmc_ref_u235_mt18.npy")
    energy_path = os.path.join(output_dir, "openmc_ref_energies.npy")
    temp_path = os.path.join(output_dir, "openmc_ref_temperatures.npy")

    np.save(ref_path, ref_matrix)
    np.save(energy_path, energy_union)
    np.save(temp_path, np.array([float(t.rstrip('K')) for t in temperatures]))

    print("\nSaved reference data:")
    print(f"  {ref_path} ({ref_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  {energy_path}")
    print(f"  {temp_path}")
    print(f"  Shape: {ref_matrix.shape} (N_E={n_e}, N_T={n_t})")


if __name__ == "__main__":
    main()
