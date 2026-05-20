# SPDX-License-Identifier: MIT
"""
Paper benchmark: OpenMC reference values for Godiva and PWR pin cell.
Runs multiple seeds and reports per-seed k_eff for Table in appendix.

Usage (from WSL):
    source ~/miniforge3/bin/activate openmc
    cd /mnt/c/Users/fog/madman_svd_experiment/scripts
    python paper_openmc_benchmark.py --seeds 10 --particles 50000 --batches 150
"""

import sys
import os
import json
import argparse
import time
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openmc

DATA_DIR = "/mnt/c/Users/fog/madman_svd_experiment/data"
HDF5_DIR = os.path.join(DATA_DIR, "endfb-vii.1-hdf5")

def setup_godiva(work_dir):
    """Godiva HEU-MET-FAST-001: bare U-235/U-238/U-234 sphere."""
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Materials
    fuel = openmc.Material(name='HEU')
    fuel.add_nuclide('U235', 0.9327, 'ao')
    fuel.add_nuclide('U238', 0.0524, 'ao')
    fuel.add_nuclide('U234', 0.0149, 'ao')
    fuel.set_density('g/cm3', 18.74)
    fuel.temperature = 294

    mats = openmc.Materials([fuel])
    mats.cross_sections = os.path.join(HDF5_DIR, "cross_sections.xml")
    mats.export_to_xml()

    # Geometry: bare sphere R = 8.7407 cm
    sphere = openmc.Sphere(r=8.7407, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel, region=-sphere)
    root = openmc.Universe(cells=[cell])
    geom = openmc.Geometry(root)
    geom.export_to_xml()

    return work_dir


def setup_pwr(work_dir):
    """Standard PWR pin cell."""
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    fuel = openmc.Material(name='UO2 3.1%')
    fuel.add_nuclide('U235', 0.031)
    fuel.add_nuclide('U238', 0.969)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.4)
    fuel.temperature = 900

    clad = openmc.Material(name='Zircaloy-4')
    clad.add_nuclide('Zr90', 0.5063)
    clad.add_nuclide('Zr91', 0.1103)
    clad.add_nuclide('Zr92', 0.1686)
    clad.add_nuclide('Zr94', 0.1709)
    clad.set_density('g/cm3', 6.55)
    clad.temperature = 600

    water = openmc.Material(name='Light Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 0.74)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = 600

    mats = openmc.Materials([fuel, clad, water])
    mats.cross_sections = os.path.join(HDF5_DIR, "cross_sections.xml")
    mats.export_to_xml()

    # Geometry
    fuel_or = openmc.ZCylinder(r=0.4096)
    clad_ir = openmc.ZCylinder(r=0.4180)
    clad_or = openmc.ZCylinder(r=0.4750)
    pitch = 1.2600
    box = openmc.rectangular_prism(pitch, pitch, boundary_type='reflective')
    z_lo = openmc.ZPlane(z0=-pitch/2, boundary_type='reflective')
    z_hi = openmc.ZPlane(z0=+pitch/2, boundary_type='reflective')
    z_region = +z_lo & -z_hi

    fuel_cell = openmc.Cell(fill=fuel, region=-fuel_or & z_region)
    gap_cell = openmc.Cell(region=+fuel_or & -clad_ir & z_region)
    clad_cell = openmc.Cell(fill=clad, region=+clad_ir & -clad_or & z_region)
    water_cell = openmc.Cell(fill=water, region=+clad_or & box & z_region)

    root = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell])
    geom = openmc.Geometry(root)
    geom.export_to_xml()

    return work_dir


def run_seeds(work_dir, batches, inactive, particles, n_seeds, label):
    """Run OpenMC multiple times with different seeds, collect k_eff."""
    os.chdir(work_dir)
    results = []

    print(f"\n{'='*60}")
    print(f"  OpenMC {label}: {n_seeds} seeds, {batches} batches, {particles} particles")
    print(f"{'='*60}")

    for seed in range(n_seeds):
        settings = openmc.Settings()
        settings.batches = batches
        settings.inactive = inactive
        settings.particles = particles
        settings.seed = seed + 1
        settings.temperature = {'method': 'interpolation'}

        if 'godiva' in label.lower():
            src = openmc.IndependentSource()
            src.space = openmc.stats.Point((0, 0, 0))
            src.energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
            settings.source = [src]
        else:
            src = openmc.IndependentSource()
            src.space = openmc.stats.Box((-0.4, -0.4, -0.6), (0.4, 0.4, 0.6))
            src.energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
            settings.source = [src]

        settings.export_to_xml()

        t0 = time.time()
        openmc.run(output=False)
        elapsed = time.time() - t0

        # Read results
        sp = openmc.StatePoint(f'statepoint.{batches}.h5')
        k = sp.keff
        k_mean = k.nominal_value
        k_std = k.std_dev

        results.append({
            'seed': seed,
            'k_mean': float(k_mean),
            'k_std': float(k_std),
            'time_s': elapsed,
        })

        print(f"  Seed {seed}: k={k_mean:.5f} +/- {k_std:.5f}  {elapsed:.1f}s")

        # Clean up
        for f in ['statepoint.{}.h5'.format(batches), 'tallies.out', 'summary.h5']:
            if os.path.exists(f):
                os.remove(f)

    # Aggregate
    k_vals = [r['k_mean'] for r in results]
    k_mean_all = np.mean(k_vals)
    k_std_all = np.std(k_vals, ddof=1) if len(k_vals) > 1 else results[0]['k_std']
    times = [r['time_s'] for r in results]

    print(f"\n  {label} SUMMARY ({n_seeds} seeds):")
    print(f"    k_eff = {k_mean_all:.5f} +/- {k_std_all:.5f}")
    print(f"    time  = {np.mean(times):.1f} +/- {np.std(times, ddof=1):.1f} s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--particles', type=int, default=50000)
    parser.add_argument('--batches', type=int, default=150)
    parser.add_argument('--inactive', type=int, default=20)
    args = parser.parse_args()

    all_results = {}

    # Godiva
    work_dir = setup_godiva(os.path.expanduser("~/openmc_paper_godiva"))
    godiva_results = run_seeds(work_dir, args.batches, args.inactive,
                                args.particles, args.seeds, "Godiva")
    all_results['godiva'] = godiva_results

    # PWR
    work_dir = setup_pwr(os.path.expanduser("~/openmc_paper_pwr"))
    pwr_results = run_seeds(work_dir, args.batches, args.inactive,
                             args.particles, args.seeds, "PWR pin cell")
    all_results['pwr'] = pwr_results

    # Save JSON
    output_file = "/mnt/c/Users/fog/madman_svd_experiment/openmc_paper_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
