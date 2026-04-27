"""
Smoke test: cycle through all four XsMode variants on the Godiva pin
and print the stats() dict for each. Demonstrates the builder toggle
and the new debug/stats path.

Usage:
    python xs_mode_demo.py [path/to/data/endfb-vii.1-hdf5/neutron]
"""
import json
import sys
from pathlib import Path

import open_rust_mc as orm

DEFAULT_DATA = Path(__file__).resolve().parents[4] / "data" / "endfb-vii.1-hdf5" / "neutron"


def godiva_scene(data_dir: Path, mode: orm.XsMode, rank: int) -> orm.Scene:
    s = orm.Scene(data_dir)
    s = s.set_xs_mode(mode)
    if rank > 0:
        s = s.set_svd_rank(rank)
    # 1 sphere, vacuum boundary
    s = s.add_surface("outer", orm.Sphere(0.0, 0.0, 0.0, 8.7407, "vacuum"))
    fuel = orm.Material("HEU", temperature=294.0, temp_idx=1)
    fuel = fuel.add_nuclide("U234.h5", 4.83e-4, 232.937, 2.428)
    fuel = fuel.add_nuclide("U235.h5", 4.4994e-2, 233.025, 2.428)
    fuel = fuel.add_nuclide("U238.h5", 2.4984e-3, 236.006, 2.428)
    s = s.add_material("HEU", fuel)
    s = s.add_cell("fuel", "-outer", "HEU", temperature=294.0)
    return s


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    if not data_dir.exists():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 1

    settings = orm.Settings(batches=40, inactive=10, particles=2000, seed=12345)

    modes = [
        (orm.XsMode.Table, 0),
        (orm.XsMode.Svd, 5),
        (orm.XsMode.HybridTableWmp, 0),
        (orm.XsMode.HybridSvdWmp, 5),
    ]

    print(f"# Godiva XS-mode round-trip on {data_dir}")
    print(f"# {settings.batches} batches × {settings.particles} particles, "
          f"{settings.batches - settings.inactive} active")
    print()
    print(f"{'mode':>20} {'k_eff':>9} {'sigma':>8} {'load_s':>7} "
          f"{'sim_s':>7} {'mem_MiB':>8} {'WMP':>4}")

    for mode, rank in modes:
        scene = godiva_scene(data_dir, mode, rank)
        result = orm.run_eigenvalue(scene, settings)
        st = result.stats()
        print(f"{st['mode']:>20} "
              f"{st['k_eff']:>9.5f} "
              f"{st['k_sigma']:>8.5f} "
              f"{st['load_time_seconds']:>7.2f} "
              f"{st['sim_time_seconds']:>7.2f} "
              f"{st['xs_memory_mib']:>8.1f} "
              f"{st['wmp_covered_nuclides']:>4d}")

    # Demonstrate the stats() dict for one mode
    print()
    print("# stats() for HybridSvdWmp:")
    scene = godiva_scene(data_dir, orm.XsMode.HybridSvdWmp, 5)
    result = orm.run_eigenvalue(scene, settings)
    print(json.dumps(result.stats(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
