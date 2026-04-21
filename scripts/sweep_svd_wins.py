"""Sweep Godiva across SVD ranks and temperature regimes to identify
where SVD wins on both memory and speed versus the ACE+WMP industry
baseline. Produces outputs/sweep_svd_wins.png plus a CSV with raw data.

Runs two scenarios (both with --discrete-rank 1, per PHYSOR reviewer):

  1. 294 K on-library  — single pointwise endpoint (best case for table)
  2. 450 K stochastic  — two endpoints with OpenMC pseudo-interpolation

For each scenario, scans SVD rank in {1, 2, 3, 5, 7}. At each grid point
we pull memory and ns/particle from the three providers (SVD, Table,
ACE+WMP) out of the godiva binary's STDOUT.

Usage:
    python scripts/sweep_svd_wins.py [--quick]

--quick runs 30 batches / 3000 particles / 2 seeds for fast iteration;
default is 80 / 5000 / 3 for paper-quality numbers.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BIN = REPO / "rust_prototype" / "target" / "release" / "godiva.exe"
DATA = REPO / "data" / "endfb-vii.1-hdf5" / "neutron"
OUT_DIR = REPO / "outputs"
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class Point:
    label: str              # "SVD", "Table", "ACE+WMP"
    scenario: str           # "on-library 294K" or "stochastic 450K"
    rank: int
    ns_per_p: float
    mem_kb: float
    k_eff: float


# ── Parsing ───────────────────────────────────────────────────────────

_RE_BLOCK = re.compile(
    r"^\s{2}(SVD[^:]*|Pointwise Table|ACE\+WMP):\s*$", re.MULTILINE
)
_RE_K = re.compile(r"k_eff\s*=\s*([0-9.]+)")
_RE_NS = re.compile(r"ns/particle\s*=\s*([0-9.]+)")
_RE_MEM = re.compile(r"XS memory\s*=\s*([0-9.]+)\s*KB")


def _parse_block(block: str) -> tuple[float, float, float]:
    """Return (k_eff, ns/p, mem_kb) from one provider's results block."""
    k = _RE_K.search(block)
    ns = _RE_NS.search(block)
    mem = _RE_MEM.search(block)
    if not (k and ns and mem):
        raise ValueError(f"unparsed block:\n{block[:400]}")
    return float(k.group(1)), float(ns.group(1)), float(mem.group(1))


def parse_output(text: str) -> dict[str, tuple[float, float, float]]:
    """Locate SVD / Table / ACE+WMP blocks and return per-provider tuples."""
    # Split on header lines and parse each provider block.
    heads = [(m.start(), m.group(1).strip()) for m in _RE_BLOCK.finditer(text)]
    results: dict[str, tuple[float, float, float]] = {}
    for idx, (start, label) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(text)
        block = text[start:end]
        key = (
            "SVD"
            if label.startswith("SVD")
            else "ACE+WMP"
            if label.startswith("ACE")
            else "Table"
        )
        results[key] = _parse_block(block)
    return results


# ── Driver ────────────────────────────────────────────────────────────

def run_one(rank: int, *, target_temp: float | None, batches: int,
            inactive: int, particles: int, seeds: int) -> dict[str, tuple[float, float, float]]:
    cmd = [
        str(BIN),
        str(DATA),
        "--mode", "all",
        "--rank", str(rank),
        "--batches", str(batches),
        "--inactive", str(inactive),
        "--particles", str(particles),
        "--seeds", str(seeds),
        "--discrete-rank", "1",
    ]
    if target_temp is not None:
        cmd += ["--target-temp", str(target_temp)]
    else:
        cmd += ["--temp-idx", "2"]  # 294K
    print(f"  > rank={rank}  target_temp={target_temp} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_output(proc.stdout)


def run_scan(*, scenarios, ranks, batches, inactive, particles, seeds) -> list[Point]:
    points: list[Point] = []
    for scen_label, target_t in scenarios:
        print(f"\n== {scen_label} ==")
        for r in ranks:
            results = run_one(r, target_temp=target_t, batches=batches,
                              inactive=inactive, particles=particles, seeds=seeds)
            for provider, (k, ns, mem) in results.items():
                points.append(Point(provider, scen_label, r, ns, mem, k))
    return points


# ── Plotting ──────────────────────────────────────────────────────────

def plot(points: list[Point], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({p.scenario for p in points})
    ranks = sorted({p.rank for p in points})

    fig, (ax_mem, ax_speed, ax_ratio) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=130)

    # Colour per scenario, marker per provider.
    scen_colour = {s: c for s, c in zip(scenarios, ["#1f77b4", "#d62728"])}
    prov_marker = {"SVD": "o", "Table": "s", "ACE+WMP": "^"}

    for scen in scenarios:
        for prov, marker in prov_marker.items():
            xs, ms, nps = [], [], []
            for r in ranks:
                pt = next((p for p in points if p.scenario == scen
                          and p.rank == r and p.label == prov), None)
                if pt is None:
                    continue
                xs.append(r)
                ms.append(pt.mem_kb / 1024.0)    # MB
                nps.append(pt.ns_per_p)
            if not xs:
                continue
            ax_mem.plot(xs, ms, marker=marker, linestyle="-",
                        color=scen_colour[scen],
                        label=f"{prov} · {scen}" if prov == "SVD" else None)
            ax_speed.plot(xs, nps, marker=marker, linestyle="-",
                          color=scen_colour[scen])

    # Legend with provider markers (scenario encoded in colour).
    from matplotlib.lines import Line2D
    prov_handles = [Line2D([0], [0], marker=m, color="black", linestyle="-",
                           markerfacecolor="white", label=p)
                    for p, m in prov_marker.items()]
    scen_handles = [Line2D([0], [0], color=scen_colour[s], linestyle="-",
                           label=s) for s in scenarios]
    ax_mem.legend(handles=prov_handles + scen_handles, fontsize=8,
                  loc="upper left", frameon=False)

    for ax in (ax_mem, ax_speed):
        ax.set_xlabel("SVD rank (discrete levels fixed at rank 1)")
        ax.grid(alpha=0.3)
        ax.set_xticks(ranks)

    ax_mem.set_ylabel("XS memory  (MB)")
    ax_mem.set_title("Memory vs rank — Godiva, 3 nuclides")
    ax_speed.set_ylabel("ns / particle")
    ax_speed.set_title("Transport speed vs rank")

    # Ratio panel: SVD relative to ACE+WMP. ">1 = SVD wins that axis."
    # mem_ratio = mem(WMP) / mem(SVD)   -> larger = SVD smaller
    # spd_ratio = ns(WMP)  / ns(SVD)    -> larger = SVD faster
    for scen in scenarios:
        xs, mem_r, spd_r = [], [], []
        for r in ranks:
            svd = next((p for p in points if p.scenario == scen
                       and p.rank == r and p.label == "SVD"), None)
            wmp = next((p for p in points if p.scenario == scen
                       and p.rank == r and p.label == "ACE+WMP"), None)
            if svd is None or wmp is None:
                continue
            xs.append(r)
            mem_r.append(wmp.mem_kb / svd.mem_kb)
            spd_r.append(wmp.ns_per_p / svd.ns_per_p)
        ax_ratio.plot(xs, mem_r, marker="o", linestyle="-",
                      color=scen_colour[scen], label=f"memory  ({scen})")
        ax_ratio.plot(xs, spd_r, marker="s", linestyle="--",
                      color=scen_colour[scen], label=f"speed   ({scen})")
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax_ratio.set_ylabel("ratio vs ACE+WMP  (>1 = SVD wins)")
    ax_ratio.set_title("Where does SVD beat the industry baseline?")
    ax_ratio.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Where does SVD win on both axes?  "
        "(blue = on-library 294 K, red = stochastic 450 K)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def write_csv(points: list[Point], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "provider", "rank", "k_eff", "ns_per_p", "mem_kb"])
        for p in points:
            w.writerow([p.scenario, p.label, p.rank, f"{p.k_eff:.5f}",
                        f"{p.ns_per_p:.2f}", f"{p.mem_kb:.1f}"])
    print(f"wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast dev settings (30b/3000p/2s)")
    args = ap.parse_args()

    if not BIN.exists():
        print(f"binary not found: {BIN}\nbuild with: cargo build --release --bin godiva",
              file=sys.stderr)
        return 1

    if args.quick:
        cfg = dict(batches=30, inactive=10, particles=3000, seeds=2)
    else:
        cfg = dict(batches=80, inactive=20, particles=5000, seeds=3)

    scenarios = [
        ("on-library 294 K", None),
        ("stochastic 450 K", 450.0),
    ]
    ranks = [1, 2, 3, 5, 7]

    points = run_scan(scenarios=scenarios, ranks=ranks, **cfg)

    write_csv(points, OUT_DIR / "sweep_svd_wins.csv")
    plot(points, OUT_DIR / "sweep_svd_wins.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
