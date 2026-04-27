"""Quick smoke test: one Godiva run per XsMode at very small N."""
import sys
from pathlib import Path
import open_rust_mc as orm

DATA = Path(r"C:\Users\fog\madman_svd_experiment\data\endfb-vii.1-hdf5\neutron")


def scene(mode: orm.XsMode, rank: int, ranks_per_mt: dict | None = None) -> orm.Scene:
    s = orm.Scene(DATA).set_xs_mode(mode)
    if rank > 0:
        s = s.set_svd_rank(rank)
    if ranks_per_mt:
        s = s.set_svd_ranks(ranks_per_mt)
    fuel = (orm.Material("HEU", temperature=294.0, temp_idx=1)
            .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
            .add_nuclide("U235.h5", atom_density=0.04509, awr=233.025, nubar=2.43)
            .add_nuclide("U238.h5", atom_density=0.00265, awr=236.006, nubar=2.49))
    s = (s.add_material("heu", fuel)
         .add_surface("boundary", orm.Sphere(r=8.7407, bc="vacuum"))
         .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
         .add_cell("outside", region="+boundary"))
    return s


def main() -> int:
    settings = orm.Settings(batches=15, inactive=5, particles=500, seed=42)
    print(f"# {settings.batches} batches × {settings.particles} particles, "
          f"{settings.batches - settings.inactive} active\n")
    print(f"{'configuration':>26} {'k_eff':>8} {'sigma':>8} {'load_s':>7} "
          f"{'sim_s':>7} {'mem_MiB':>8} {'WMP':>4}")

    cases = [
        ("Table",                        orm.XsMode.Table,          0,  None),
        ("SVD k=5",                      orm.XsMode.Svd,            5,  None),
        ("ACE+WMP",                      orm.XsMode.HybridTableWmp, 0,  None),
        ("Hybrid SVD+WMP k=5",           orm.XsMode.HybridSvdWmp,   5,  None),
        # Per-reaction adaptive: smooth reactions to rank 1 (WMP handles
        # the resonance window inside the hybrid; rank 1 captures the
        # smooth tails per the phase5 SVD-spectrum analysis).
        ("Hybrid SVD+WMP adaptive",      orm.XsMode.HybridSvdWmp,   5,
            {2: 1, 18: 1, 102: 1}),
        # Adaptive without WMP — pure SVD with rank 1 on smooth reactions.
        ("SVD adaptive (no WMP)",        orm.XsMode.Svd,            5,
            {2: 1, 18: 1, 102: 1}),
    ]

    for label, mode, rank, ranks_per_mt in cases:
        r = orm.run_eigenvalue(scene(mode, rank, ranks_per_mt), settings)
        st = r.stats()
        print(f"{label:>26} {st['k_eff']:>8.5f} {st['k_sigma']:>8.5f} "
              f"{st['load_time_seconds']:>7.2f} {st['sim_time_seconds']:>7.2f} "
              f"{st['xs_memory_mib']:>8.1f} {st['wmp_covered_nuclides']:>4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
