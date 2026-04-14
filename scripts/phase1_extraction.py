"""
Phase 1 — Extract U-235 cross-section data and build the SVD matrix.

Reads the HDF5 nuclear data file for U-235, extracts fission (MT=18)
cross-sections at multiple temperatures, and constructs the matrix
A ∈ R^(N_E × N_T) in log-space for SVD analysis.

Usage:
    python scripts/phase1_extraction.py [path_to_U235.h5]
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# Attempt openmc.data import; fall back to pure h5py if unavailable
try:
    import openmc.data
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False
    import h5py


def find_u235_h5(data_dir: str) -> str:
    """Search for the U-235 HDF5 file in the data directory tree."""
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f in ("U235.h5", "U_235.h5", "n-092_U_235.h5"):
                return os.path.join(root, f)
            # Also check inside nuclide-named dirs
            if "U235" in f and f.endswith(".h5"):
                return os.path.join(root, f)
    return ""


def extract_with_openmc(h5_path: str):
    """Extract cross-section data using openmc.data API."""
    print(f"Loading U-235 data from: {h5_path}")
    u235 = openmc.data.IncidentNeutron.from_hdf5(h5_path)

    print(f"Available temperatures: {u235.temperatures}")
    print(f"Available reactions: {list(u235.reactions.keys())}")

    # Get the fission reaction (MT=18)
    if 18 not in u235.reactions:
        raise ValueError("MT=18 (fission) not found in U-235 data!")
    reaction = u235.reactions[18]

    # Sort temperatures numerically (strip 'K', convert to float)
    temperatures = sorted(u235.temperatures, key=lambda t: float(t.rstrip('K')))
    print(f"\nUsing temperatures: {temperatures}")

    # Build unionized energy grid from all temperatures
    all_energies = []
    for T in temperatures:
        e = u235.energy[T]
        all_energies.append(e)
    energy_union = np.unique(np.concatenate(all_energies))
    print(f"Unionized energy grid: {len(energy_union)} points")
    print(f"Energy range: {energy_union[0]:.4e} eV to {energy_union[-1]:.4e} eV")

    # Build matrix A: each column is σ_f(E) at a given temperature
    cols = []
    for T in temperatures:
        xs = reaction.xs[T]
        sigma = xs(energy_union)  # interpolate to union grid
        # Replace zeros/negatives (interpolation artifacts)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
        print(f"  T={T}: min={sigma.min():.2e}, max={sigma.max():.2e}, "
              f"zeros_replaced={np.sum(sigma <= 1e-30)}")

    A = np.column_stack(cols)
    return energy_union, temperatures, A


def extract_with_h5py(h5_path: str):
    """Extract cross-section data using h5py directly (fallback)."""
    print(f"Loading U-235 data from: {h5_path} (h5py fallback)")

    with h5py.File(h5_path, 'r') as f:
        # List available groups
        print(f"Top-level keys: {list(f.keys())}")

        # Navigate the HDF5 structure to find temperatures and reactions
        # OpenMC HDF5 structure: /nuclide/reactions/reaction_XXX/...
        # with energy and xs datasets per temperature
        nuclide = f
        if 'U235' in f:
            nuclide = f['U235']

        # Discover structure
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(print_structure)

    raise NotImplementedError(
        "h5py extraction requires manual structure inspection. "
        "Please install openmc: pip install 'openmc @ git+https://github.com/openmc-dev/openmc.git'"
    )


def main():
    output_dir = os.path.expanduser("~/madman_svd_experiment/outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Find U-235 HDF5 file
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        data_dir = os.path.expanduser("~/madman_svd_experiment/data")
        h5_path = find_u235_h5(data_dir)
        if not h5_path:
            print("ERROR: Could not find U235.h5. Provide path as argument or "
                  "place it in data/ directory.")
            print(f"Searched in: {data_dir}")
            sys.exit(1)

    # Extract data
    if HAS_OPENMC:
        energies, temperatures, A = extract_with_openmc(h5_path)
    else:
        energies, temperatures, A = extract_with_h5py(h5_path)

    print(f"\n=== Matrix A constructed ===")
    print(f"Shape: {A.shape} (N_E={A.shape[0]}, N_T={A.shape[1]})")
    print(f"Memory: {A.nbytes / 1e6:.1f} MB")

    # Checkpoint 1 validation
    N_E, N_T = A.shape
    assert N_E >= 10000, f"N_E={N_E} is suspiciously small (expected >= 10,000)"
    assert N_T >= 3, f"N_T={N_T} is too small (need >= 3 temperatures)"
    assert not np.any(np.isnan(A)), "Matrix contains NaN values!"
    assert not np.any(np.isinf(A)), "Matrix contains Inf values!"
    assert np.all(A > 0), "Matrix contains non-positive values!"
    print("CHECKPOINT 1: All validations passed ✓")

    # Log-transform
    A_log = np.log10(A)
    print(f"\nLog-transformed matrix range: [{A_log.min():.2f}, {A_log.max():.2f}]")

    # Save outputs
    np.save(os.path.join(output_dir, "A_raw_u235_mt18.npy"), A)
    np.save(os.path.join(output_dir, "A_log_u235_mt18.npy"), A_log)
    np.save(os.path.join(output_dir, "energies_u235.npy"), energies)
    np.save(os.path.join(output_dir, "temperatures_u235.npy"),
            np.array(temperatures, dtype=object))

    print(f"\nSaved to {output_dir}:")
    print(f"  A_raw_u235_mt18.npy  ({A.nbytes / 1e6:.1f} MB)")
    print(f"  A_log_u235_mt18.npy  ({A_log.nbytes / 1e6:.1f} MB)")
    print(f"  energies_u235.npy    ({energies.nbytes / 1e6:.1f} MB)")
    print(f"  temperatures_u235.npy")

    return energies, temperatures, A, A_log


if __name__ == "__main__":
    main()
