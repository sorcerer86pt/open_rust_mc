# SPDX-License-Identifier: MIT
"""
Phase 4 extension — Godiva k_eff with ALL reactions SVD-modified.

Modifies MT=2 (elastic), MT=18 (fission), and MT=102 (capture)
simultaneously at each rank k, then runs OpenMC.
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

WORK_DIR = os.path.expanduser("~/openmc_godiva_multi_rxn")
BATCHES = 500
INACTIVE = 50
PARTICLES = 100000

MTS = [2, 18, 102]  # elastic, fission, capture


def setup_godiva():
    fuel = openmc.Material(name='HEU')
    fuel.add_nuclide('U235', 0.93500)
    fuel.add_nuclide('U238', 0.05500)
    fuel.add_nuclide('U234', 0.01000)
    fuel.set_density('g/cm3', 18.74)
    materials = openmc.Materials([fuel])

    sphere = openmc.Sphere(r=8.7407, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel, region=-sphere)
    universe = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(universe)

    settings = openmc.Settings()
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES
    settings.run_mode = 'eigenvalue'
    bounds = [-8.7407]*3 + [8.7407]*3
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(bounds[:3], bounds[3:])
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
    print(f"  k_eff = {k_val:.5f} ± {k_unc:.5f}")
    return k_val, k_unc


def modify_all_reactions(u235_path, output_path, k_rank):
    """SVD-modify MT=2, 18, 102 at rank k."""
    u235 = openmc.data.IncidentNeutron.from_hdf5(u235_path)
    temps = sorted(
        [t for t in u235.temperatures if float(t.rstrip('K')) > 0],
        key=lambda t: float(t.rstrip('K'))
    )

    all_e = [u235.energy[T] for T in temps]
    energy_union = np.unique(np.concatenate(all_e))

    for mt in MTS:
        if mt not in u235.reactions:
            continue
        rxn = u235.reactions[mt]

        cols = []
        for T in temps:
            sigma = rxn.xs[T](energy_union)
            sigma = np.where(sigma > 0, sigma, 1e-30)
            cols.append(sigma)
        A = np.column_stack(cols)
        A_log = np.log10(A)

        U, S, Vt = scipy_svd(A_log, full_matrices=False)
        k = min(k_rank, len(S))
        A_k_log = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        A_k = 10**A_k_log

        for t_idx, T in enumerate(temps):
            orig_energy = u235.energy[T]
            xs_recon = np.interp(orig_energy, energy_union, A_k[:, t_idx])
            xs_recon = np.where(xs_recon > 0, xs_recon, 1e-30)
            rxn.xs[T] = openmc.data.Tabulated1D(orig_energy, xs_recon)

    if os.path.exists(output_path):
        os.remove(output_path)
    u235.export_to_hdf5(output_path)
    print(f"  Modified MT={MTS} at k={k_rank}: {output_path}")


def patch_xs_xml(data_dir, modified_h5, run_dir):
    hdf5_dir = os.path.join(data_dir, "endfb-vii.1-hdf5")
    src_xml = os.path.join(hdf5_dir, "cross_sections_godiva.xml")
    tree = ET.parse(src_xml)
    root = tree.getroot()

    for lib in root.findall(".//library"):
        mat = lib.get("materials", "")
        rel_path = lib.get("path", "")
        if "U235" in mat and "wmp" not in rel_path:
            lib.set("path", os.path.abspath(modified_h5))
        elif not os.path.isabs(rel_path):
            lib.set("path", os.path.abspath(os.path.join(hdf5_dir, rel_path)))

    dst = os.path.join(run_dir, "cross_sections_svd.xml")
    tree.write(dst)
    return dst


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    data_dir = "/mnt/c/Users/fog/madman_svd_experiment/data"
    hdf5_dir = os.path.join(data_dir, "endfb-vii.1-hdf5")
    xs_xml = os.path.join(hdf5_dir, "cross_sections_godiva.xml")
    os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml

    u235_path = os.path.join(hdf5_dir, "neutron", "U235.h5")

    print("=" * 70)
    print("Godiva k_eff — ALL REACTIONS MODIFIED (MT=2, 18, 102)")
    print("=" * 70)

    materials, geometry, settings = setup_godiva()

    # Baseline
    baseline_dir = os.path.join(WORK_DIR, "baseline")
    k_base, s_base = run_openmc(baseline_dir, materials, geometry, settings, "BASELINE")

    results = [("baseline", None, k_base, s_base)]

    for k_rank in [3, 4, 5]:
        run_dir = os.path.join(WORK_DIR, f"all_rxn_k{k_rank}")
        modified_h5 = os.path.join(WORK_DIR, f"U235_all_rxn_k{k_rank}.h5")

        modify_all_reactions(u235_path, modified_h5, k_rank)

        os.makedirs(run_dir, exist_ok=True)
        svd_xml = patch_xs_xml(data_dir, modified_h5, run_dir)
        os.environ["OPENMC_CROSS_SECTIONS"] = svd_xml

        k_val, k_unc = run_openmc(run_dir, materials, geometry, settings,
                                   f"ALL RXN k={k_rank}")
        results.append((f"all_rxn k={k_rank}", k_rank, k_val, k_unc))

        os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml

    # Report
    print(f"\n{'='*70}")
    print("RESULTS — ALL REACTIONS MODIFIED")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'k_eff':>10} {'±σ':>10} {'Δk (pcm)':>10} {'Sig?':>12}")
    for label, k, kv, ku in results:
        delta = abs(kv - k_base) / k_base * 1e5 if k else 0
        combined = np.sqrt(ku**2 + s_base**2) / k_base * 1e5 if k else 0
        sig = "YES" if k and delta > 3 * combined else ("no" if k else "")
        print(f"  {label:<20} {kv:>10.5f} {ku:>10.5f} {delta:>10.1f} {sig:>12}")


if __name__ == "__main__":
    main()
