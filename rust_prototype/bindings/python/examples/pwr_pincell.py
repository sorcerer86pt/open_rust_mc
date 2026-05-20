# SPDX-License-Identifier: MIT
"""Neutron-only PWR pin cell k_inf via the Python API.

Standard 3.1 % UO2 / He gap / Zircaloy-4 / borated water pin at
600 K with a 1.26 cm reflective lattice — the canonical
LWR-benchmark cell. Uses S(α,β) thermal scattering for H in H2O
(the dominant reactivity effect at PWR temperatures).

Reports k_inf with statistical uncertainty and the per-cell
capture distribution. Roughly 30 s on a recent laptop CPU at the
default settings.

Usage:
    python rust_prototype/bindings/python/examples/pwr_pincell.py \\
        data/endfb-vii.1-hdf5/neutron
"""
from __future__ import annotations

import argparse
import sys

from open_rust_mc import (
    Scene,
    Settings,
    XPlane,
    YPlane,
    ZCylinder,
    ZPlane,
    XsMode,
    run_eigenvalue,
    uranium_oxide_material,
    water_material,
    zircaloy4_material,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Neutron-only PWR pin k_inf")
    parser.add_argument("data_dir", help="Neutron HDF5 data directory")
    parser.add_argument("--mode", default="hybrid_svd_wmp",
                        choices=["table", "svd", "hybrid_table_wmp", "hybrid_svd_wmp"])
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--inactive", type=int, default=30)
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    mode_lookup = {
        "table": XsMode.Table,
        "svd": XsMode.Svd,
        "hybrid_table_wmp": XsMode.HybridTableWmp,
        "hybrid_svd_wmp": XsMode.HybridSvdWmp,
    }
    xs_mode = mode_lookup[args.mode]

    # Materials — operating temperatures match the OpenMC reference deck.
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

    # Geometry — 3 concentric cylinders, square reflective lattice.
    FUEL_OR, CLAD_IR, CLAD_OR, PITCH = 0.4096, 0.4180, 0.4750, 1.2600
    half = PITCH / 2.0

    scene = (
        Scene(args.data_dir)
        .set_xs_mode(xs_mode)
        .set_svd_rank(args.rank)
        .add_material("fuel", fuel)
        .add_material("clad", clad)
        .add_material("water", water)
        .add_surface("fuel_or", ZCylinder(r=FUEL_OR))
        .add_surface("clad_ir", ZCylinder(r=CLAD_IR))
        .add_surface("clad_or", ZCylinder(r=CLAD_OR))
        .add_surface("xmin", XPlane(x0=-half, bc="reflective"))
        .add_surface("xmax", XPlane(x0=+half, bc="reflective"))
        .add_surface("ymin", YPlane(y0=-half, bc="reflective"))
        .add_surface("ymax", YPlane(y0=+half, bc="reflective"))
        .add_surface("zmin", ZPlane(z0=-half, bc="reflective"))
        .add_surface("zmax", ZPlane(z0=+half, bc="reflective"))
        .add_cell("fuel", "-fuel_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="fuel", temperature=900.0)
        .add_cell("gap", "+fuel_or -clad_ir +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill=None, temperature=600.0)
        .add_cell("clad", "+clad_ir -clad_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="clad", temperature=600.0)
        .add_cell("water", "+clad_or +xmin -xmax +ymin -ymax +zmin -zmax",
                  fill="water", temperature=600.0)
    )

    # For Hybrid SVD+WMP, smooth tails are well represented at rank 1
    # — WMP handles the resonance window exactly.
    if xs_mode == XsMode.HybridSvdWmp:
        scene = scene.set_svd_ranks({2: 1, 18: 1, 102: 1})

    settings = Settings(
        batches=args.batches,
        inactive=args.inactive,
        particles=args.particles,
        seed=args.seed,
    )

    print(f"Mode: {xs_mode!r} (rank {args.rank})")
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
    print(f"  XS memory        = {st['xs_memory_mib']:.1f} MiB "
          f"({st['wmp_covered_nuclides']} WMP-covered)")
    print()
    print("  Captures by cell (active total):")
    for name, count in result.captures_dict().items():
        print(f"    {name:<6s}  {count:>14.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
