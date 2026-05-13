"""Run an ICSBEP regression case from Python.

The same JSON files `tests/cuda_runs.rs` consumes drive this harness.
Pick a case, choose the runner, and the engine grades `k_calc` against
the case's acceptance reference (handbook `k_eff_reference`, or the
`local_validation` block when the JSON ships one) using the
`|Δ| ≤ max(150 pcm, 2 σ_combined)` envelope.

Example
-------
    python icsbep_run.py heu-met-fast-001_case-1 cpu
    python icsbep_run.py pu-met-fast-006 gpu      # needs --features cuda

Build the Python extension first:
    cd bindings/python
    maturin develop --release                       # CPU only
    maturin develop --release --features cuda       # also enables Runner.GpuCuda
"""

from __future__ import annotations

import sys
from pathlib import Path

from open_rust_mc import Runner, Settings, run_icsbep_case


def main(case_stem: str, runner_label: str) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    case_json = repo_root / "bench" / "icsbep" / f"{case_stem}.json"
    data_dir = repo_root / "data" / "endfb-vii.1-hdf5" / "neutron"

    if not case_json.exists():
        print(f"case not found: {case_json}", file=sys.stderr)
        return 2
    if not data_dir.exists():
        print(f"data directory not found: {data_dir}", file=sys.stderr)
        return 2

    runner = {
        "cpu": Runner.Cpu,
        "gpu": Runner.GpuCuda,
        "gpu_cuda": Runner.GpuCuda,
    }.get(runner_label.lower())
    if runner is None:
        print(f"unknown runner {runner_label!r}; use 'cpu' or 'gpu'", file=sys.stderr)
        return 2

    settings = Settings(batches=80, inactive=20, particles=5000, seed=1)
    result = run_icsbep_case(
        case_json=case_json,
        data_dir=data_dir,
        settings=settings,
        runner=runner,
        rank=15,
    )

    # Avoid the unicode `sigma` glyph in the labels — Windows consoles
    # default to cp1252 and choke on it. Numbers are ASCII-clean.
    verdict = "PASS" if result.passed else "FAIL"
    print(f"  case          : {result.case}")
    print(f"  runner        : {runner.name()}")
    print(f"  reference     : {result.ref_source}")
    print(
        f"  handbook      : k = {result.handbook_k:.5f} +/- {result.handbook_sigma:.5f}"
    )
    print(
        f"  acceptance    : k = {result.k_ref:.5f} +/- {result.sigma_exp:.5f}   "
        f"sigma_combined = {result.sigma_combined * 1e5:.0f} pcm"
    )
    print(f"  k_calc        : {result.k_eff:.5f} +/- {result.k_sigma:.5f}")
    print(
        f"  delta         : {result.delta_pcm:+.0f} pcm   "
        f"{result.sigma_ratio:.2f}-sigma   bound = +/-{result.bound_pcm:.0f} pcm   [{verdict}]"
    )
    print(
        f"  timing        : load = {result.load_time_seconds:.2f} s, "
        f"sim = {result.sim_time_seconds:.2f} s, total = {result.runtime_seconds:.2f} s"
    )
    print(
        f"  active batches: {result.active_batches} "
        f"(total {result.total_histories:,} histories)"
    )
    print(
        f"  tallies       : coll = {result.total_collisions:,}, "
        f"fis = {result.total_fissions:,}, leak = {result.total_leakage:,}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "heu-met-fast-001_case-1"
    runner_label = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    sys.exit(main(case, runner_label))
