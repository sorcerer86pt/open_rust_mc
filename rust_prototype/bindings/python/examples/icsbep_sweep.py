# SPDX-License-Identifier: MIT
"""ICSBEP sweep launcher — thin Python wrapper around the
`benchmark_runner` Rust binary.

The in-process heterogeneous CPU/GPU pipeline lives in Rust
(`src/bin/benchmark_runner.rs`; see `docs/benchmark-pipeline-spec.md`).
This script exists so existing entry points (`run_benchmark.ps1`,
Jupyter, CI) can keep their `python icsbep_sweep.py ...` invocations
while delegating the actual work to the binary.

CLI flags map 1:1 to `benchmark_runner`. Unknown flags are forwarded
verbatim. Data-directory resolution is delegated to the binary, which
accepts a workspace root, a library root (`.../endfb-viii.1-hdf5`),
or a neutron directory directly.

Usage
-----
    python icsbep_sweep.py --runner gpu --csv outputs/sweep.csv
    python icsbep_sweep.py --runner cpu --filter heu-met-fast \\
        --batches 150 --inactive 30 --particles 20000 --seeds 5
    python icsbep_sweep.py --resume --csv outputs/sweep.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` until a directory containing both
    `bench/icsbep` and `rust_prototype/Cargo.toml` is found."""
    for p in [start, *start.parents]:
        if (p / "bench" / "icsbep").is_dir() and (p / "rust_prototype" / "Cargo.toml").is_file():
            return p
    raise SystemExit(
        f"could not locate workspace root (looking for bench/icsbep + "
        f"rust_prototype/Cargo.toml) starting from {start}"
    )


def find_binary(repo_root: Path) -> Path | None:
    """Locate the `benchmark_runner` binary in the standard cargo
    target directories. Prefer release over debug. Returns None if
    neither is present — caller falls back to `cargo run`."""
    exe = "benchmark_runner.exe" if os.name == "nt" else "benchmark_runner"
    for profile in ("release", "debug"):
        candidate = repo_root / "rust_prototype" / "target" / profile / exe
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the public sweep flags. Unknown args are returned in
    `extras` and forwarded verbatim to the binary, so any future
    `benchmark_runner` flag works without updating this script."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bench-dir", type=Path, default=None,
                   help="directory of *.json case files (default: <repo>/bench/icsbep)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="ENDF HDF5 directory. Accepts a workspace root, a library "
                        "root, or a neutron directory. Auto-discovered by the binary "
                        "if omitted.")
    p.add_argument("--filter", type=str, default=None,
                   help="substring filter applied to case file stem")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV output path (append, flushed per case)")
    p.add_argument("--telemetry", type=Path, default=None,
                   help="JSONL telemetry path (per-case timings + routing)")
    p.add_argument("--runner", choices=["cpu", "gpu", "auto"], default="auto",
                   help="execution backend (default: auto)")
    p.add_argument("--batches", type=int, default=None,
                   help="override active+inactive batches per case")
    p.add_argument("--inactive", type=int, default=None,
                   help="override inactive batches per case")
    p.add_argument("--particles", type=int, default=None,
                   help="override particles per batch")
    p.add_argument("--seeds", type=int, default=None,
                   help="seeds per case (multi-seed averaging)")
    p.add_argument("--base-seed", type=int, default=None,
                   help="first seed; subsequent seeds derived deterministically")
    p.add_argument("--rank", type=int, default=None, help="SVD rank")
    p.add_argument("--resume", action="store_true",
                   help="skip cases already present in --csv")
    p.add_argument("--stop-file", type=Path, default=None,
                   help="path; when created, the binary halts at the next case boundary")
    p.add_argument("--case-timeout-s", type=int, default=None,
                   help="per-case timeout (watchdog kick threshold)")
    p.add_argument("--n-sigma", type=float, default=None,
                   help="acceptance envelope multiplier")
    p.add_argument("--sequential", action="store_true",
                   help="single-thread driver (diagnostic / determinism)")
    # Survival biasing — DEFAULT ON. The analog GPU path's event loop
    # spins to `max_events_per_history = 1_000_000` whenever the
    # batch tail contains one persistent neutron (common on
    # ≥200k-particle runs and on water-moderated cases). Implicit
    # capture + Bernoulli-banked fission + Russian roulette
    # terminates such tails in O(log w) RR rolls without biasing
    # k_eff. Pass `--no-survival-bias` to opt back into pure
    # analog for A/B parity work.
    sb = p.add_mutually_exclusive_group()
    sb.add_argument("--survival-bias", dest="survival_bias",
                    action="store_true", default=True,
                    help="enable implicit capture + RR (default)")
    sb.add_argument("--no-survival-bias", dest="survival_bias",
                    action="store_false",
                    help="opt out — pure analog tracking")
    p.add_argument("--cargo-run", action="store_true",
                   help="invoke via `cargo run` instead of the prebuilt binary "
                        "(rebuilds if source has changed)")
    p.add_argument("--features", type=str, default=None,
                   help="cargo features (e.g. 'cuda'). Only used with --cargo-run; "
                        "if omitted, --runner=gpu/auto implies 'cuda'.")
    # Legacy flags from the previous in-process Python implementation —
    # accepted for back-compat with `run_benchmark.ps1` and existing
    # CI invocations, but currently not surfaced by the binary. A
    # warning is printed when one is non-default so the user knows the
    # flag had no effect.
    p.add_argument("--limit", type=int, default=None,
                   help="(legacy) cap number of cases after filtering — ignored; "
                        "the binary processes the full filtered set")
    p.add_argument("--fail-fast", action="store_true",
                   help="(legacy) ignored — the binary keeps going past failures")
    p.add_argument("--debug-metrics", type=Path, default=None,
                   help="(legacy) per-case GPU debug metrics — use --telemetry "
                        "instead (JSONL format)")
    p.add_argument("--gpu-refill-factor", type=float, default=None,
                   help="(legacy) GPU refill pool factor — not yet surfaced by the binary")
    p.add_argument("--gpu-auto-refill", action="store_true",
                   help="(legacy) GPU auto-refill — not yet surfaced by the binary")
    return p.parse_known_args()


def warn_legacy(args: argparse.Namespace) -> None:
    msgs: list[str] = []
    if args.limit is not None:
        msgs.append(f"--limit {args.limit} ignored (not yet surfaced by benchmark_runner)")
    if args.fail_fast:
        msgs.append("--fail-fast ignored (not yet surfaced by benchmark_runner)")
    if args.debug_metrics is not None:
        msgs.append(
            f"--debug-metrics {args.debug_metrics} ignored — use --telemetry instead"
        )
    if args.gpu_refill_factor is not None:
        msgs.append(
            f"--gpu-refill-factor {args.gpu_refill_factor} ignored "
            "(not yet surfaced by benchmark_runner)"
        )
    if args.gpu_auto_refill:
        msgs.append("--gpu-auto-refill ignored (not yet surfaced by benchmark_runner)")
    for m in msgs:
        print(f"[icsbep_sweep] warning: {m}", file=sys.stderr)


def build_runner_argv(args: argparse.Namespace) -> list[str]:
    """Translate parsed sweep args into the binary's flag surface."""
    argv: list[str] = []
    if args.bench_dir is not None:
        argv += ["--bench-dir", str(args.bench_dir)]
    if args.data_dir is not None:
        argv += ["--data-dir", str(args.data_dir)]
    if args.filter is not None:
        argv += ["--filter", args.filter]
    if args.csv is not None:
        argv += ["--csv", str(args.csv)]
    if args.telemetry is not None:
        argv += ["--telemetry", str(args.telemetry)]
    argv += ["--runner", args.runner]
    if args.batches is not None:
        argv += ["--batches", str(args.batches)]
    if args.inactive is not None:
        argv += ["--inactive-batches", str(args.inactive)]
    if args.particles is not None:
        argv += ["--particles-per-batch", str(args.particles)]
    if args.seeds is not None:
        argv += ["--seeds", str(args.seeds)]
    if args.base_seed is not None:
        argv += ["--base-seed", str(args.base_seed)]
    if args.rank is not None:
        argv += ["--rank", str(args.rank)]
    if args.resume:
        argv += ["--resume"]
    if args.stop_file is not None:
        argv += ["--stop-file", str(args.stop_file)]
    if args.case_timeout_s is not None:
        argv += ["--case-timeout-s", str(args.case_timeout_s)]
    if args.n_sigma is not None:
        argv += ["--n-sigma", str(args.n_sigma)]
    if args.sequential:
        argv += ["--sequential"]
    # `benchmark_runner` defaults --survival-bias to false; this
    # script's default is the opposite. Forward the flag only when on
    # to keep the binary's CLI surface as the source of truth on
    # default semantics (the binary's help text is what users read
    # when they `--help`).
    if args.survival_bias:
        argv += ["--survival-bias"]
    return argv


def main() -> int:
    args, extras = parse_args()
    warn_legacy(args)
    repo_root = find_repo_root(Path(__file__).resolve())

    # Make relative paths the user gave us resolve from the same place
    # the binary will resolve them from — the workspace root.
    os.chdir(repo_root)

    runner_argv = build_runner_argv(args)
    runner_argv += extras  # forward unknown flags verbatim

    if args.cargo_run:
        cargo = shutil.which("cargo")
        if cargo is None:
            print("error: --cargo-run was passed but `cargo` is not on PATH",
                  file=sys.stderr)
            return 127
        features = args.features
        if features is None and args.runner in ("gpu", "auto"):
            features = "cuda"
        cmd = [cargo, "run", "--release"]
        if features:
            cmd += ["--features", features]
        cmd += ["--bin", "benchmark_runner", "--", *runner_argv]
    else:
        bin_path = find_binary(repo_root)
        if bin_path is None:
            print(
                "error: benchmark_runner not found in rust_prototype/target/{release,debug}. "
                "Build it first:\n"
                "    cargo build --release --bin benchmark_runner            # CPU\n"
                "    cargo build --release --features cuda --bin benchmark_runner  # +GPU\n"
                "Or rerun this script with --cargo-run to build-and-launch in one step.",
                file=sys.stderr,
            )
            return 127
        cmd = [str(bin_path), *runner_argv]

    print(f"[icsbep_sweep] cwd: {repo_root}")
    print(f"[icsbep_sweep] exec: {' '.join(cmd)}")
    sys.stdout.flush()

    try:
        proc = subprocess.run(cmd)
    except KeyboardInterrupt:
        # The binary handles SIGINT internally (flushes the CSV); just
        # propagate the conventional 130 exit code.
        return 130
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
