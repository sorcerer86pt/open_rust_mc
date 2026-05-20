# SPDX-License-Identifier: MIT
"""
Phase 4 extension — PWR pin cell benchmark.

A single UO2 fuel pin in light water with Zircaloy cladding.
This is the standard benchmark geometry for reactor physics:
  - 3.1% enriched UO2 fuel (U-235, U-238, O-16)
  - Zircaloy-4 cladding (Zr-90, Zr-91, Zr-92, Zr-94, Zr-96, Sn, Fe, Cr)
  - Light water moderator (H-1, O-16) at ~600K

We modify all major U-235 and U-238 reactions with SVD and compare k_inf.
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import openmc
import openmc.data
from scipy.linalg import svd as scipy_svd
import xml.etree.ElementTree as ET

WORK_DIR = os.path.expanduser("~/openmc_pwr_pincell")
DATA_DIR = "/mnt/c/Users/fog/madman_svd_experiment/data"
HDF5_DIR = os.path.join(DATA_DIR, "endfb-vii.1-hdf5")

BATCHES = 500
INACTIVE = 50
PARTICLES = 100000

# Reactions to modify for U-235 and U-238
MODIFY_MTS = [2, 18, 102]  # elastic, fission, capture


def setup_pincell():
    """Create a standard PWR pin cell model."""
    # Materials
    fuel = openmc.Material(name='UO2 3.1%')
    fuel.add_nuclide('U235', 0.031)
    fuel.add_nuclide('U238', 0.969)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.4)
    fuel.temperature = 900  # K

    clad = openmc.Material(name='Zircaloy-4')
    clad.add_nuclide('Zr90', 0.5063)
    clad.add_nuclide('Zr91', 0.1103)
    clad.add_nuclide('Zr92', 0.1686)
    clad.add_nuclide('Zr94', 0.1709)
    clad.add_nuclide('Zr96', 0.0276)
    clad.set_density('g/cm3', 6.55)
    clad.temperature = 600  # K

    water = openmc.Material(name='Light Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 0.74)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = 600  # K

    materials = openmc.Materials([fuel, clad, water])

    # Geometry — standard PWR dimensions
    fuel_or = 0.4096  # cm (fuel outer radius)
    clad_ir = 0.4180  # cm (clad inner radius)
    clad_or = 0.4750  # cm (clad outer radius)
    pitch = 1.2600    # cm (pin pitch)

    fuel_cyl = openmc.ZCylinder(r=fuel_or)
    clad_inner = openmc.ZCylinder(r=clad_ir)
    clad_outer = openmc.ZCylinder(r=clad_or)

    # Pin cell box (reflective boundaries = infinite lattice)
    box = openmc.model.RectangularPrism(
        pitch, pitch, boundary_type='reflective'
    )

    fuel_cell = openmc.Cell(name='fuel', fill=fuel, region=-fuel_cyl)
    gap_cell = openmc.Cell(name='gap', region=+fuel_cyl & -clad_inner)  # void gap
    clad_cell = openmc.Cell(name='clad', fill=clad, region=+clad_inner & -clad_outer)
    water_cell = openmc.Cell(name='water', fill=water, region=+clad_outer & -box)

    universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
    geometry = openmc.Geometry(universe)

    # Settings
    settings = openmc.Settings()
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES
    settings.run_mode = 'eigenvalue'
    settings.temperature = {'method': 'interpolation'}

    # Source in the fuel
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-fuel_or, -fuel_or, -1],
                               [fuel_or, fuel_or, 1])
    )

    return materials, geometry, settings


def run_openmc(run_dir, materials, geometry, settings, label=""):
    os.makedirs(run_dir, exist_ok=True)
    orig = os.getcwd()
    os.chdir(run_dir)
    materials.export_to_xml()
    geometry.export_to_xml()
    settings.export_to_xml()
    print(f"\n  Running [{label}]...")
    openmc.run(output=False)
    with openmc.StatePoint(f"statepoint.{BATCHES}.h5") as sp:
        k_val = float(sp.keff.nominal_value)
        k_unc = float(sp.keff.std_dev)
    os.chdir(orig)
    print(f"  k_inf = {k_val:.5f} +/- {k_unc:.5f}")
    return k_val, k_unc


def modify_nuclide_svd(h5_path, output_path, k_rank, mts):
    """Apply SVD truncation to specified reactions."""
    nuc = openmc.data.IncidentNeutron.from_hdf5(h5_path)
    name = os.path.basename(h5_path).replace('.h5', '')

    temps = sorted(
        [t for t in nuc.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )
    all_e = [nuc.energy[T] for T in temps]
    energy_union = np.unique(np.concatenate(all_e))

    modified = []
    for mt in mts:
        if mt not in nuc.reactions:
            continue
        rxn = nuc.reactions[mt]

        cols = []
        for T in temps:
            sigma = rxn.xs[T](energy_union)
            sigma = np.where(sigma > 0, sigma, 1e-30)
            cols.append(sigma)
        A = np.column_stack(cols)
        A_log = np.log10(A)

        U, S, Vt = scipy_svd(A_log, full_matrices=False)
        k = min(k_rank, len(S))
        A_k = 10 ** (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :])

        for t_idx, T in enumerate(temps):
            orig_energy = nuc.energy[T]
            xs_recon = np.interp(orig_energy, energy_union, A_k[:, t_idx])
            xs_recon = np.where(xs_recon > 0, xs_recon, 1e-30)
            rxn.xs[T] = openmc.data.Tabulated1D(orig_energy, xs_recon)

        modified.append(mt)

    if os.path.exists(output_path):
        os.remove(output_path)
    nuc.export_to_hdf5(output_path)
    print(f"  {name}: modified MT={modified} at k={k_rank}")


def create_xs_xml(modified_nuclides, run_dir):
    """Create cross_sections.xml pointing to modified nuclides + originals."""
    src_xml = os.path.join(HDF5_DIR, "cross_sections.xml")
    tree = ET.parse(src_xml)
    root = tree.getroot()

    for lib in root.findall(".//library"):
        mat = lib.get("materials", "")
        rel_path = lib.get("path", "")
        lib_type = lib.get("type", "")

        # Check if this nuclide was modified
        if mat in modified_nuclides and lib_type == "neutron":
            lib.set("path", modified_nuclides[mat])
        else:
            # Absolute path to original
            abs_path = os.path.join(HDF5_DIR, rel_path)
            if os.path.exists(abs_path):
                lib.set("path", os.path.abspath(abs_path))

    dst = os.path.join(run_dir, "cross_sections.xml")
    tree.write(dst)
    return dst


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print("=" * 70)
    print("PWR Pin Cell Benchmark — SVD-Modified Cross-Sections")
    print("=" * 70)

    xs_xml = os.path.join(HDF5_DIR, "cross_sections.xml")
    if not os.path.exists(xs_xml):
        print(f"ERROR: {xs_xml} not found. Need full ENDF library.")
        sys.exit(1)
    os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml

    materials, geometry, settings = setup_pincell()

    # Baseline
    baseline_dir = os.path.join(WORK_DIR, "baseline")
    k_base, s_base = run_openmc(baseline_dir, materials, geometry, settings, "BASELINE")

    results = [("baseline", None, k_base, s_base)]

    # SVD runs — modify U-235 and U-238 simultaneously
    for k_rank in [3, 4, 5]:
        run_dir = os.path.join(WORK_DIR, f"svd_k{k_rank}")
        os.makedirs(run_dir, exist_ok=True)

        modified = {}
        for nuc_name in ['U235', 'U238']:
            h5_orig = os.path.join(HDF5_DIR, "neutron", f"{nuc_name}.h5")
            h5_mod = os.path.join(WORK_DIR, f"{nuc_name}_svd_k{k_rank}.h5")
            modify_nuclide_svd(h5_orig, h5_mod, k_rank, MODIFY_MTS)
            modified[nuc_name] = os.path.abspath(h5_mod)

        svd_xml = create_xs_xml(modified, run_dir)
        os.environ["OPENMC_CROSS_SECTIONS"] = svd_xml

        k_val, k_unc = run_openmc(run_dir, materials, geometry, settings,
                                   f"SVD k={k_rank}")
        results.append((f"SVD k={k_rank}", k_rank, k_val, k_unc))

        os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml

    # Report
    print(f"\n{'='*70}")
    print("RESULTS — PWR Pin Cell")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'k_inf':>10} {'+-sigma':>10} {'dk (pcm)':>10} {'Sig?':>12}")
    for label, k, kv, ku in results:
        delta = abs(kv - k_base) / k_base * 1e5 if k else 0
        combined = np.sqrt(ku**2 + s_base**2) / k_base * 1e5 if k else 0
        sig = "YES" if k and delta > 3 * combined else ("no" if k else "")
        print(f"  {label:<15} {kv:>10.5f} {ku:>10.5f} {delta:>10.1f} {sig:>12}")


if __name__ == "__main__":
    main()
