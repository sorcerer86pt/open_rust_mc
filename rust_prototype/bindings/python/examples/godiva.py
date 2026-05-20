# SPDX-License-Identifier: MIT
"""Godiva (ICSBEP HEU-MET-FAST-001) k-eigenvalue benchmark via Python.

This is the Python-API counterpart to `src/bin/godiva.rs`. It builds
the same 8.7407 cm U-235 sphere, loads the same HDF5 files, and runs
the same engine path under the hood. Reference k_eff from the Rust
binary at rank=5, 50 batches × 5000 particles: ~1.0004.

Usage (after `maturin develop --release`):

    python bindings/python/examples/godiva.py path/to/endfb-vii.1-hdf5/neutron
"""
from __future__ import annotations

import argparse
import sys

from open_rust_mc import Material, Scene, Settings, Sphere, run_eigenvalue


def main() -> int:
    parser = argparse.ArgumentParser(description="Godiva via the Python API")
    parser.add_argument(
        "data_dir",
        help="Directory containing U234.h5, U235.h5, U238.h5 from the ENDF/B-VII.1 HDF5 release",
    )
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--inactive", type=int, default=10)
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # ── Material: HEU (93.5% U-235 by mass, matching the ICSBEP eval) ──
    heu = (
        Material("HEU", temperature=294.0, temp_idx=1)
        .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
        .add_nuclide("U235.h5", atom_density=0.04509, awr=233.025, nubar=2.43)
        .add_nuclide("U238.h5", atom_density=0.00265, awr=236.006, nubar=2.49)
    )

    # ── Geometry: single 8.7407 cm vacuum sphere ───────────────────────
    scene = (
        Scene(args.data_dir)
        .add_material("heu", heu)
        .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
        .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
        .add_cell("outside", region="+boundary")  # void outer cell
    )

    settings = Settings(
        batches=args.batches,
        inactive=args.inactive,
        particles=args.particles,
        seed=args.seed,
    )

    print(f"Running Godiva ({settings})")
    result = run_eigenvalue(scene, settings)
    print(f"\nk_eff = {result.k_eff:.5f} +/- {result.k_sigma:.5f}")
    print(f"active batches:  {result.active_batches}")
    print(f"total histories: {result.total_histories}")
    print(f"runtime:         {result.runtime_seconds:.2f} s")
    ns_per_particle = result.runtime_seconds * 1e9 / result.total_histories
    print(f"perf:            {ns_per_particle:.1f} ns/particle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
