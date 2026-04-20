"""
Phase 4 — Godiva k_eff benchmark: original vs SVD-reconstructed cross-sections.

ICSBEP HEU-MET-FAST-001 (Godiva): bare sphere of highly enriched uranium.
k_eff experimental = 1.0000 ± 0.0010

Procedure:
  1. Download ENDF/B-VIII.0 cross-section data (if not present)
  2. Run OpenMC with original U235 data → k_eff baseline
  3. For each SVD rank k: modify U235 fission xs, re-run → k_eff(k)
  4. Report Δk_eff in pcm for each rank

Run inside WSL with: conda activate openmc && python phase4_keff_benchmark.py
"""

import os
import sys
import numpy as np

import openmc
import openmc.data
from scipy.linalg import svd as scipy_svd


WORK_DIR = os.path.expanduser("~/openmc_godiva_benchmark")
DATA_DIR = "/mnt/c/Users/fog/madman_svd_experiment/data"

# Simulation parameters
# High statistics: 100k × 450 active = 45M histories → σ ≈ 2-3 pcm
BATCHES = 500
INACTIVE = 50
PARTICLES = 100000


def ensure_cross_sections():
    """Make sure cross-section data is available."""
    win_data = "/mnt/c/Users/fog/madman_svd_experiment/data"
    hdf5_dir = os.path.join(win_data, "endfb-vii.1-hdf5")
    xs_xml = os.path.join(hdf5_dir, "cross_sections_godiva.xml")

    if not os.path.exists(xs_xml):
        xs_xml = os.path.join(hdf5_dir, "cross_sections.xml")

    if os.path.exists(xs_xml):
        print(f"Using cross-sections from: {xs_xml}")
        os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml
        return win_data

    # Try conda-provided data
    conda_data = os.environ.get("OPENMC_CROSS_SECTIONS", "")
    if conda_data and os.path.exists(conda_data):
        print(f"Using conda cross-sections: {conda_data}")
        return os.path.dirname(conda_data)

    print("ERROR: No cross-section data found.")
    print("Set OPENMC_CROSS_SECTIONS or place data in expected path.")
    sys.exit(1)


def setup_godiva():
    """Create the Godiva benchmark model (ICSBEP HEU-MET-FAST-001)."""
    # Material: highly enriched uranium (93.5% U-235)
    fuel = openmc.Material(name='HEU')
    fuel.add_nuclide('U235', 0.93500)
    fuel.add_nuclide('U238', 0.05500)
    fuel.add_nuclide('U234', 0.01000)
    fuel.set_density('g/cm3', 18.74)
    materials = openmc.Materials([fuel])

    # Geometry: bare sphere, radius = 8.7407 cm (critical radius)
    sphere = openmc.Sphere(r=8.7407, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel, region=-sphere)
    universe = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(universe)

    # Settings
    settings = openmc.Settings()
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES
    settings.run_mode = 'eigenvalue'

    # Source: uniform in the sphere
    bounds = [-8.7407, -8.7407, -8.7407, 8.7407, 8.7407, 8.7407]
    uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
    settings.source = openmc.IndependentSource(space=uniform_dist)

    return materials, geometry, settings


def run_openmc(run_dir, materials, geometry, settings, label=""):
    """Run OpenMC in the given directory and return k_eff ± σ."""
    os.makedirs(run_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(run_dir)

    materials.export_to_xml()
    geometry.export_to_xml()
    settings.export_to_xml()

    print(f"\n  Running OpenMC [{label}] ({PARTICLES} particles × "
          f"{BATCHES - INACTIVE} active batches)...")
    openmc.run(output=False)

    # Extract k_eff from statepoint
    sp_file = f"statepoint.{BATCHES}.h5"
    with openmc.StatePoint(sp_file) as sp:
        keff = sp.keff
        k_val = float(keff.nominal_value)
        k_unc = float(keff.std_dev)

    os.chdir(orig_dir)
    print(f"  k_eff = {k_val:.5f} ± {k_unc:.5f}")
    return k_val, k_unc


def modify_u235_with_svd(original_h5, output_h5, k_rank, temperatures):
    """Load U235 HDF5, apply SVD truncation to MT=18, save modified copy."""
    u235 = openmc.data.IncidentNeutron.from_hdf5(original_h5)
    reaction = u235.reactions[18]

    # Build unionized energy grid
    temps = sorted([t for t in u235.temperatures if float(t.rstrip('K')) > 0],
                   key=lambda t: float(t.rstrip('K')))
    all_e = [u235.energy[T] for T in temps]
    energy_union = np.unique(np.concatenate(all_e))

    # Build cross-section matrix
    cols = []
    for T in temps:
        xs_func = reaction.xs[T]
        sigma = xs_func(energy_union)
        sigma = np.where(sigma > 0, sigma, 1e-30)
        cols.append(sigma)
    A = np.column_stack(cols)

    # SVD in log-space
    A_log = np.log10(A)
    U, S, Vt = scipy_svd(A_log, full_matrices=False)

    # Truncated reconstruction
    A_k_log = U[:, :k_rank] @ np.diag(S[:k_rank]) @ Vt[:k_rank, :]
    A_k = 10 ** A_k_log

    # Write modified cross-sections back.
    # Each temperature has its own energy grid in the HDF5 file;
    # interpolate the SVD reconstruction from the union grid back to it.
    for t_idx, T in enumerate(temps):
        xs_recon_union = A_k[:, t_idx]
        orig_energy = u235.energy[T]
        # Interpolate union→original grid
        xs_recon_orig = np.interp(orig_energy, energy_union, xs_recon_union)
        xs_recon_orig = np.where(xs_recon_orig > 0, xs_recon_orig, 1e-30)
        reaction.xs[T] = openmc.data.Tabulated1D(orig_energy, xs_recon_orig)

    if os.path.exists(output_h5):
        os.remove(output_h5)
    u235.export_to_hdf5(output_h5)
    print(f"  Wrote SVD k={k_rank} modified: {output_h5}")
    return A, S


def patch_cross_sections_xml(data_dir, modified_h5, run_dir):
    """Create a cross_sections.xml that points to our modified U235."""
    import xml.etree.ElementTree as ET

    src_xml = os.path.join(data_dir, "endfb-vii.1-hdf5", "cross_sections_godiva.xml")
    if not os.path.exists(src_xml):
        src_xml = os.path.join(data_dir, "endfb-vii.1-hdf5", "cross_sections.xml")
    tree = ET.parse(src_xml)
    root = tree.getroot()

    hdf5_dir = os.path.dirname(src_xml)
    for lib in root.findall(".//library"):
        mat = lib.get("materials", "")
        rel_path = lib.get("path", "")
        if "U235" in mat and "wmp" not in rel_path:
            lib.set("path", os.path.abspath(modified_h5))
        else:
            abs_path = os.path.join(hdf5_dir, rel_path)
            lib.set("path", os.path.abspath(abs_path))

    dst_xml = os.path.join(run_dir, "cross_sections_svd.xml")
    tree.write(dst_xml)
    return dst_xml


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 4 — Godiva k_eff Benchmark: Original vs SVD Reconstruction")
    print("=" * 70)

    data_dir = ensure_cross_sections()

    # Find U235.h5
    u235_path = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == "U235.h5":
                u235_path = os.path.join(root, f)
                break
    if not u235_path:
        print("ERROR: Cannot find U235.h5")
        sys.exit(1)
    print(f"U235 data: {u235_path}")

    # Setup Godiva
    materials, geometry, settings = setup_godiva()

    # ── Baseline run ─────────────────────────────────────────────────
    baseline_dir = os.path.join(WORK_DIR, "baseline")
    k_baseline, sigma_baseline = run_openmc(
        baseline_dir, materials, geometry, settings, label="BASELINE"
    )

    # ── SVD runs at various ranks ────────────────────────────────────
    results = [("baseline", None, k_baseline, sigma_baseline)]

    for k_rank in [2, 3, 4, 5]:
        svd_dir = os.path.join(WORK_DIR, f"svd_k{k_rank}")
        modified_h5 = os.path.join(WORK_DIR, f"U235_svd_k{k_rank}.h5")

        # Modify U235 cross-sections
        A, S = modify_u235_with_svd(u235_path, modified_h5, k_rank, None)

        # Patch cross_sections.xml
        os.makedirs(svd_dir, exist_ok=True)
        svd_xs_xml = patch_cross_sections_xml(data_dir, modified_h5, svd_dir)
        os.environ["OPENMC_CROSS_SECTIONS"] = svd_xs_xml

        # Run
        k_svd, sigma_svd = run_openmc(
            svd_dir, materials, geometry, settings, label=f"SVD k={k_rank}"
        )
        results.append((f"SVD k={k_rank}", k_rank, k_svd, sigma_svd))

        # Restore original cross-sections for next iteration
        os.environ["OPENMC_CROSS_SECTIONS"] = os.path.join(data_dir, "cross_sections.xml")

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  {'Method':<15} {'k_eff':>10} {'± σ':>10} {'Δk (pcm)':>10} {'Significant?':>14}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for label, k_rank, k_val, k_unc in results:
        if k_rank is None:
            delta_pcm = 0.0
            sig = ""
        else:
            delta_pcm = abs(k_val - k_baseline) / k_baseline * 1e5
            # Combined uncertainty
            combined_sigma = np.sqrt(k_unc**2 + sigma_baseline**2) / k_baseline * 1e5
            sig = "YES" if delta_pcm > 3 * combined_sigma else "no (< 3σ)"

        print(f"  {label:<15} {k_val:>10.5f} {k_unc:>10.5f} {delta_pcm:>10.1f} {sig:>14}")

    # ── Verdict ──────────────────────────────────────────────────────
    print("\n  TARGET: Δk < 10 pcm")
    for label, k_rank, k_val, k_unc in results:
        if k_rank is not None:
            delta = abs(k_val - k_baseline) / k_baseline * 1e5
            if delta < 10:
                print(f"  ✓ {label}: {delta:.1f} pcm — WITHIN TARGET")
            elif delta < 50:
                print(f"  ~ {label}: {delta:.1f} pcm — acceptable for many applications")
            else:
                print(f"  ✗ {label}: {delta:.1f} pcm — needs hybrid approach")

    print(f"\n  Baseline k_eff = {k_baseline:.5f} ± {sigma_baseline:.5f}")
    print(f"  MC uncertainty = {sigma_baseline / k_baseline * 1e5:.1f} pcm")


if __name__ == "__main__":
    main()
