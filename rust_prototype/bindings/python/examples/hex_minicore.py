# SPDX-License-Identifier: MIT
"""Hex-bounded PWR pin cell via the Python API.

Demonstrates ``Scene.add_hex_boundary`` + ``Scene.add_pin_cylinders``:
a single UO2/Zr/H2O pin inside a flat-top hex outer boundary with
reflective hex sides + reflective z planes. Counterpart to the Rust
``hex_minicore`` binary, scoped down because the Python scene builder
does not yet expose hex *lattices* — just the hex *boundary* helper.

The engine still walks the full hex geometry (6 reflective side
planes + 2 z planes) on every particle step, so this is a real
exercise of the helper.

Usage:
    python rust_prototype/bindings/python/examples/hex_minicore.py \\
        data/endfb-vii.1-hdf5/neutron
"""
from __future__ import annotations

import argparse
import sys

from open_rust_mc import (
    Scene,
    Settings,
    XsMode,
    ZCylinder,
    run_eigenvalue,
    uranium_oxide_material,
    water_material,
    zircaloy4_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hex-bounded PWR pin via Python")
    parser.add_argument("data_dir", help="Neutron HDF5 data directory")
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--inactive", type=int, default=15)
    parser.add_argument("--particles", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--pitch", type=float, default=1.260,
                        help="Pin pitch (cm). Inradius of the hex boundary "
                             "is 0.5 * pitch (single-element hex cell).")
    args = parser.parse_args()

    fuel_or, clad_ir, clad_or = 0.4096, 0.4180, 0.4750
    z_half = args.pitch / 2.0

    fuel = uranium_oxide_material(
        "UO2 3.1%", density_g_per_cm3=10.4, enrichment=0.031,
        temperature=900.0, temp_idx=3,
    )
    clad = zircaloy4_material(
        density_g_per_cm3=6.55, temperature=600.0, temp_idx=2,
    )
    water = water_material(
        density_g_per_cm3=0.74, temperature=600.0, temp_idx=2,
        thermal_file="c_H_in_H2O.h5",
    )

    scene = (
        Scene(args.data_dir)
        .set_xs_mode(XsMode.HybridSvdWmp)
        .set_svd_rank(args.rank)
        .add_material("fuel", fuel)
        .add_material("clad", clad)
        .add_material("water", water)
        .add_surface("fuel_or", ZCylinder(r=fuel_or))
        .add_surface("clad_ir", ZCylinder(r=clad_ir))
        .add_surface("clad_or", ZCylinder(r=clad_or))
    )

    # Hex outer boundary: rings=0 → inradius = 0.5*pitch (single-element
    # hex cell). All hex sides + z planes reflective so the cell
    # partition does not need an outside cell.
    scene, hex_inside = scene.add_hex_boundary(
        "outer", rings=0, pitch=args.pitch, orientation="flat",
        xy_bc="reflective", z_half=z_half, z_bc="reflective",
    )

    # Pin cells, intersected with the hex-inside region. Pyo3 region
    # parser is intersection-only, which composes naturally with the
    # `&`-joined hex-inside expression returned above.
    scene = (
        scene
        .add_cell("fuel", f"-fuel_or & {hex_inside}", fill="fuel", temperature=900.0)
        .add_cell("gap",  f"+fuel_or & -clad_ir & {hex_inside}", fill=None, temperature=600.0)
        .add_cell("clad", f"+clad_ir & -clad_or & {hex_inside}", fill="clad", temperature=600.0)
        .add_cell("water", f"+clad_or & {hex_inside}", fill="water", temperature=600.0)
        .set_svd_ranks({2: 1, 18: 1, 102: 1})
    )

    settings = Settings(
        batches=args.batches,
        inactive=args.inactive,
        particles=args.particles,
        seed=args.seed,
    )

    print(f"Hex pin cell ({args.pitch:.3f} cm pitch, hex reflective)")
    print(f"Settings: {settings.batches} batches "
          f"({settings.batches - settings.inactive} active) × "
          f"{settings.particles} particles, seed {settings.seed}")
    result = run_eigenvalue(scene, settings)
    st = result.stats()

    print()
    print(f"  k_inf            = {result.k_eff:.5f} ± {result.k_sigma:.5f}")
    print(f"  active batches   = {result.active_batches}")
    print(f"  total histories  = {result.total_histories:,}")
    print(f"  load time        = {st['load_time_seconds']:.2f} s")
    print(f"  sim time         = {st['sim_time_seconds']:.2f} s "
          f"({st['ns_per_history']:.1f} ns/history)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
