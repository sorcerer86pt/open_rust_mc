"""Multi-seed Godiva sweep — Student-t 95 % confidence interval on k_eff.

Runs the ICSBEP HEU-MET-FAST-001 benchmark N times with independent
seeds, then reports the across-seed mean, the within-run sigma, the
across-run sigma, and a 95 % CI from the Student-t distribution at
N-1 degrees of freedom. The pattern any benchmark-quality run wants
when reporting a number against an experimental band.

ICSBEP target: k = 1.0000 ± 100 pcm (experimental σ).

Usage:
    python rust_prototype/bindings/python/examples/seed_sweep.py \\
        data/endfb-vii.1-hdf5/neutron --seeds 5
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

from open_rust_mc import (
    Material,
    Scene,
    Settings,
    Sphere,
    XsMode,
    run_eigenvalue,
)


# Student-t two-sided 95 % critical values for nu = 1..15. ν = N-1.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131,
}


def godiva_scene(data_dir: str, mode: XsMode, rank: int) -> Scene:
    fuel = (
        Material("HEU", temperature=294.0, temp_idx=1)
        .add_nuclide("U234.h5", atom_density=0.000483, awr=232.937, nubar=2.428)
        .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.428)
        .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.428)
    )
    s = (
        Scene(data_dir)
        .set_xs_mode(mode)
        .set_svd_rank(rank)
        .add_material("heu", fuel)
        .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
        .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
        .add_cell("outside", region="+boundary")
    )
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed Godiva sweep with 95 % CI")
    parser.add_argument("data_dir", help="Neutron HDF5 data directory")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--inactive", type=int, default=20)
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--mode", default="svd",
                        choices=["table", "svd", "hybrid_table_wmp", "hybrid_svd_wmp"])
    parser.add_argument("--rank", type=int, default=5)
    args = parser.parse_args()

    if args.seeds < 2:
        print("--seeds must be ≥ 2 for a meaningful CI", file=sys.stderr)
        return 2

    mode_lookup = {
        "table": XsMode.Table,
        "svd": XsMode.Svd,
        "hybrid_table_wmp": XsMode.HybridTableWmp,
        "hybrid_svd_wmp": XsMode.HybridSvdWmp,
    }
    xs_mode = mode_lookup[args.mode]

    print(f"Mode: {xs_mode!r}  rank={args.rank}")
    print(f"Per-seed budget: {args.batches} batches × {args.particles} particles")
    print()
    print(f"  {'seed':>5}  {'k_eff':>8}  {'sigma':>8}  {'sim_s':>7}")

    k_per_seed: list[float] = []
    sigma_per_seed: list[float] = []
    for i in range(args.seeds):
        seed = 1 + i
        scene = godiva_scene(args.data_dir, xs_mode, args.rank)
        settings = Settings(
            batches=args.batches,
            inactive=args.inactive,
            particles=args.particles,
            seed=seed,
        )
        result = run_eigenvalue(scene, settings)
        k_per_seed.append(result.k_eff)
        sigma_per_seed.append(result.k_sigma)
        print(f"  {seed:>5}  {result.k_eff:>8.5f}  {result.k_sigma:>8.5f}  "
              f"{result.runtime_seconds:>7.2f}")

    n = len(k_per_seed)
    mean_k = statistics.fmean(k_per_seed)
    # Across-seed standard deviation (sample, not population):
    s_across = statistics.stdev(k_per_seed)
    se_mean = s_across / math.sqrt(n)
    # Within-seed sigma combined in quadrature:
    s_within = math.sqrt(sum(s * s for s in sigma_per_seed) / n) / math.sqrt(n)
    # Total 1-sigma uncertainty: across + within added in quadrature.
    s_total = math.sqrt(se_mean ** 2 + s_within ** 2)
    t = _T95.get(n - 1, 1.96)  # asymptotic z = 1.96 for ν > 15
    ci_half_width = t * s_total

    pcm = 1e5
    print()
    print(f"  Across-seed mean k = {mean_k:.5f}")
    print(f"  Across-seed σ      = {s_across:.5f}  ({s_across*pcm:>5.0f} pcm)")
    print(f"  Within-seed σ̄/√N   = {s_within:.5f}  ({s_within*pcm:>5.0f} pcm)")
    print(f"  Combined σ̄         = {s_total:.5f}  ({s_total*pcm:>5.0f} pcm)")
    print(f"  Student-t {100*0.95:.0f}% CI  = {mean_k:.5f} ± {ci_half_width:.5f}  "
          f"({ci_half_width*pcm:>5.0f} pcm half-width)")
    print()
    delta_icsbep_pcm = (mean_k - 1.0) * pcm
    print(f"  Δ vs ICSBEP target (k=1.0000 ± 100 pcm): "
          f"{delta_icsbep_pcm:+.0f} pcm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
