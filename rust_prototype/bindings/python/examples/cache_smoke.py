"""Cache smoke test — run 2 small cases from the same ICSBEP family
and dump `cache_stats()` between them. The first case populates the
L1 nuclide cache from L2 disk + fresh HDF5 parses; the second case
should HIT L1 on every shared nuclide, demonstrating the across-case
cache reuse that gates the wall time of multi-case sweeps.

Expected pattern (correct cache behaviour):

    Case 1 (cold): L1 misses + L2 hits/misses + puts
    Case 2 (warm): L1 hits = N (every nuclide), no L2 traffic, no puts

If case 2 shows zero L1 hits when both cases share nuclides, the
cache is broken or mis-keyed.

Usage:
    python rust_prototype/bindings/python/examples/cache_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import open_rust_mc as orm

# Walk up from this file to the repo root (parent of `rust_prototype`).
HERE = Path(__file__).resolve()
REPO = HERE
while REPO.parent != REPO and not (REPO / "bench" / "icsbep").is_dir():
    REPO = REPO.parent
if not (REPO / "bench" / "icsbep").is_dir():
    print("could not locate bench/icsbep from", HERE, file=sys.stderr)
    sys.exit(2)

DATA = REPO / "data" / "endfb-vii.1-hdf5" / "neutron"
BENCH = REPO / "bench" / "icsbep"
# Verified identical nuclide sets:
#   {Be9, C0, N14, N15, O16, O17, Ar36, Ar38, Ar40, Ni58, Ni60-62, Ni64,
#    U234, U235, U238}
# Case 1 populates L1 from L2-disk hits; case 3 should then show
# 17 fresh L1 hits because all keys overlap.
CASES = [
    "heu-met-fast-058_case-1.json",
    "heu-met-fast-058_case-3.json",
]


def _fmt_stats(s: dict) -> str:
    def rate(h: int, m: int) -> str:
        if h + m == 0:
            return "n/a"
        return f"{h / (h + m) * 100:5.1f}%"
    return (
        f"L1 {s['l1_hits']:5}/{s['l1_hits'] + s['l1_misses']:5} ({rate(s['l1_hits'], s['l1_misses'])}) | "
        f"L2 {s['l2_hits']:3}/{s['l2_hits'] + s['l2_misses']:3} ({rate(s['l2_hits'], s['l2_misses'])}) | "
        f"L3 {s['l3_hits']:3}/{s['l3_hits'] + s['l3_misses']:3} ({rate(s['l3_hits'], s['l3_misses'])}) | "
        f"puts={s['puts']}"
    )


def main() -> int:
    runner = orm.Runner.recommended()
    print(f"Runner: {runner.name()}\n")

    # Tiny settings so each case completes in seconds, not minutes.
    settings = orm.Settings()
    settings.batches = 30
    settings.inactive = 5
    settings.particles = 5000
    settings.seed = 42

    orm.cache_stats_reset()
    print(f"Initial: {_fmt_stats(orm.cache_stats())}\n")

    for i, case in enumerate(CASES):
        case_path = BENCH / case
        if not case_path.exists():
            print(f"SKIP {case}: not found at {case_path}")
            continue
        print(f"── Case {i + 1}: {case} ──")
        before = orm.cache_stats()
        res = orm.run_icsbep_case(
            case_json=case_path,
            data_dir=DATA,
            settings=settings,
            runner=runner,
            rank=5,
        )
        after = orm.cache_stats()
        delta = {k: after[k] - before.get(k, 0) for k in after if not k.endswith("_rate")}
        print(f"  k_eff       = {res.k_eff:.5f} ± {res.k_sigma:.5f}")
        print(f"  runtime     = {res.runtime_seconds:.2f} s")
        print(f"  Δ cache:    L1 +{delta['l1_hits']:4}h / +{delta['l1_misses']:4}m | "
              f"L2 +{delta['l2_hits']:3}h / +{delta['l2_misses']:3}m | "
              f"L3 +{delta['l3_hits']:3}h / +{delta['l3_misses']:3}m | "
              f"puts +{delta['puts']}")
        print(f"  Cumulative: {_fmt_stats(after)}\n")

    final = orm.cache_stats()
    print("── Summary ──")
    print(_fmt_stats(final))
    if "l1_hit_rate" in final:
        print(f"L1 hit rate: {final['l1_hit_rate'] * 100:.1f}%")

    # Quick sanity: at least *some* L1 hits expected if both cases ran.
    if final["l1_hits"] == 0 and final["puts"] > 0:
        print("\nWARNING: cache populated but never hit — investigate.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
