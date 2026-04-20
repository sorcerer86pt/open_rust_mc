"""
Honesty Test — Three-way comparison: SVD vs Pointwise Table vs OpenMC.

Runs all three engines on the same Godiva benchmark (HEU-MET-FAST-001)
with matching parameters and reports CPU time, memory usage, and k_eff
fidelity for a fair head-to-head comparison.

Usage (from WSL with openmc conda env):
    conda activate openmc
    python honesty_test.py [--particles N] [--batches N] [--inactive N] [--rank K]

Default: 150 batches, 20 inactive, 20000 particles, rank 5
"""

import argparse
import json
import os
import resource
import subprocess
import time

import openmc
import openmc.data


# ── Configuration ──────────────────────────────────────────────────────

WIN_DATA = "/mnt/c/Users/fog/madman_svd_experiment/data"
WIN_PROJECT = "/mnt/c/Users/fog/madman_svd_experiment"
WORK_DIR = os.path.expanduser("~/openmc_honesty_test")


def parse_args():
    parser = argparse.ArgumentParser(description="Honesty test: SVD vs Table vs OpenMC")
    parser.add_argument("--particles", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=150)
    parser.add_argument("--inactive", type=int, default=20)
    parser.add_argument("--rank", type=int, default=5)
    return parser.parse_args()


# ── 1. OpenMC run ──────────────────────────────────────────────────────

def run_openmc_godiva(particles, batches, inactive):
    """Run OpenMC Godiva and return (k_eff, k_std, wall_time_s, peak_mem_kb)."""
    run_dir = os.path.join(WORK_DIR, "openmc")
    os.makedirs(run_dir, exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(run_dir)

    # Cross-sections
    hdf5_dir = os.path.join(WIN_DATA, "endfb-vii.1-hdf5")
    xs_xml = os.path.join(hdf5_dir, "cross_sections.xml")
    if not os.path.exists(xs_xml):
        # Try godiva-specific
        xs_xml = os.path.join(hdf5_dir, "cross_sections_godiva.xml")
    os.environ["OPENMC_CROSS_SECTIONS"] = xs_xml

    # Material
    fuel = openmc.Material(name="HEU")
    fuel.add_nuclide("U235", 0.93500)
    fuel.add_nuclide("U238", 0.05500)
    fuel.add_nuclide("U234", 0.01000)
    fuel.set_density("g/cm3", 18.74)
    materials = openmc.Materials([fuel])

    # Geometry
    sphere = openmc.Sphere(r=8.7407, boundary_type="vacuum")
    cell = openmc.Cell(fill=fuel, region=-sphere)
    universe = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(universe)

    # Settings
    settings = openmc.Settings()
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles
    settings.run_mode = "eigenvalue"
    bounds = [-8.7407, -8.7407, -8.7407, 8.7407, 8.7407, 8.7407]
    settings.source = openmc.IndependentSource(space=openmc.stats.Box(bounds[:3], bounds[3:]))

    materials.export_to_xml()
    geometry.export_to_xml()
    settings.export_to_xml()

    print(f"\n  Running OpenMC ({particles} particles x {batches} batches)...")

    # Measure time and memory
    t0 = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    openmc.run(output=False)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_time = time.perf_counter() - t0

    # Peak memory from child process (in KB on Linux)
    peak_mem_kb = usage_after.ru_maxrss  # Already in KB on Linux

    # Extract results
    sp_file = f"statepoint.{batches}.h5"
    with openmc.StatePoint(sp_file) as sp:
        keff = sp.keff
        k_val = float(keff.nominal_value)
        k_unc = float(keff.std_dev)

    os.chdir(orig_dir)
    return k_val, k_unc, wall_time, peak_mem_kb


# ── 2. Rust engine runs ───────────────────────────────────────────────

def run_rust_godiva(mode, particles, batches, inactive, rank):
    """Run the Rust Godiva binary and parse results.

    Returns (k_eff, k_std, load_ms, sim_ms, xs_memory_kb) or None on failure.
    """
    # The .exe needs Windows-style paths for its data_dir argument,
    # but subprocess needs the WSL /mnt/c/ path to find the binary.
    wsl_data_dir = os.path.join(WIN_DATA, "endfb-vii.1-hdf5", "neutron")
    win_data_dir = wsl_data_dir.replace("/mnt/c/", "C:/")

    wsl_exe = os.path.join(WIN_PROJECT, "rust_prototype", "target", "release", "godiva.exe")

    cmd = [
        wsl_exe, win_data_dir,
        "--mode", mode,
        "--rank", str(rank),
        "--batches", str(batches),
        "--inactive", str(inactive),
        "--particles", str(particles),
    ]

    print(f"\n  Running Rust engine (mode={mode}, rank={rank}, "
          f"{particles} particles x {batches} batches)...")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: Rust binary timed out after 3600s")
        return None
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR: Rust binary exited with code {result.returncode}")
        print(result.stderr[:500] if result.stderr else "")
        return None

    output = result.stdout
    # Parse k_eff, timing, memory from output
    k_mean = k_std = load_ms = sim_ms = xs_kb = None

    for line in output.split("\n"):
        line = line.strip()
        if "k_eff" in line and "=" in line and "+/-" in line:
            parts = line.split("=")[1].strip().split("+/-")
            try:
                k_mean = float(parts[0].strip())
                k_std = float(parts[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Load time" in line and "=" in line:
            try:
                load_ms = float(line.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Sim time" in line and "=" in line:
            try:
                sim_ms = float(line.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "XS memory" in line and "=" in line:
            try:
                xs_kb = float(line.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    if k_mean is None:
        print("  WARNING: Could not parse k_eff from output")
        print("  Last 20 lines of output:")
        for l in output.strip().split("\n")[-20:]:
            print(f"    {l}")
        return None

    return k_mean, k_std, load_ms or 0, sim_ms or 0, xs_kb or 0, wall_time


# ── 3. Comparison report ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("HONESTY TEST — Three-Way Comparison")
    print("  SVD-compressed vs Pointwise Table vs OpenMC")
    print("=" * 70)
    print("\n  Benchmark:  Godiva (HEU-MET-FAST-001)")
    print(f"  Particles:  {args.particles}")
    print(f"  Batches:    {args.batches} ({args.inactive} inactive)")
    print(f"  SVD rank:   {args.rank}")

    results = {}

    # ── OpenMC ──
    try:
        k, unc, wall_s, mem_kb = run_openmc_godiva(args.particles, args.batches, args.inactive)
        results["OpenMC"] = {
            "k_eff": k, "k_std": unc,
            "wall_time_s": wall_s,
            "peak_mem_kb": mem_kb,
        }
        print(f"  OpenMC: k={k:.5f} +/- {unc:.5f}, wall={wall_s:.1f}s, mem={mem_kb} KB")
    except Exception as e:
        print(f"  ERROR running OpenMC: {e}")

    # ── Rust SVD ──
    r = run_rust_godiva("svd", args.particles, args.batches, args.inactive, args.rank)
    if r:
        k, unc, load_ms, sim_ms, xs_kb, wall_s = r
        results["Rust SVD"] = {
            "k_eff": k, "k_std": unc,
            "load_ms": load_ms, "sim_ms": sim_ms,
            "xs_memory_kb": xs_kb,
            "wall_time_s": wall_s,
        }
        print(f"  Rust SVD:   k={k:.5f} +/- {unc:.5f}, sim={sim_ms:.0f}ms, xs_mem={xs_kb:.1f}KB")

    # ── Rust Table ──
    r = run_rust_godiva("table", args.particles, args.batches, args.inactive, args.rank)
    if r:
        k, unc, load_ms, sim_ms, xs_kb, wall_s = r
        results["Rust Table"] = {
            "k_eff": k, "k_std": unc,
            "load_ms": load_ms, "sim_ms": sim_ms,
            "xs_memory_kb": xs_kb,
            "wall_time_s": wall_s,
        }
        print(f"  Rust Table: k={k:.5f} +/- {unc:.5f}, sim={sim_ms:.0f}ms, xs_mem={xs_kb:.1f}KB")

    # ── Report ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    k_exp = 1.0000
    header = f"  {'Engine':<16} {'k_eff':>10} {'± σ':>8} {'Δ(exp) pcm':>11} {'Sim time':>10} {'XS Memory':>10}"
    print(header)
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*11} {'-'*10} {'-'*10}")

    for name, data in results.items():
        k = data["k_eff"]
        unc = data["k_std"]
        delta_pcm = abs(k - k_exp) / k_exp * 1e5

        if "sim_ms" in data:
            time_str = f"{data['sim_ms']:.0f} ms"
        else:
            time_str = f"{data['wall_time_s']:.1f} s"

        if "xs_memory_kb" in data:
            mem_str = f"{data['xs_memory_kb']:.1f} KB"
        elif "peak_mem_kb" in data:
            mem_str = f"{data['peak_mem_kb'] / 1024:.1f} MB"
        else:
            mem_str = "N/A"

        print(f"  {name:<16} {k:>10.5f} {unc:>8.5f} {delta_pcm:>11.0f} {time_str:>10} {mem_str:>10}")

    # ── Pairwise comparisons ──
    if "Rust SVD" in results and "Rust Table" in results:
        svd = results["Rust SVD"]
        tbl = results["Rust Table"]
        dk = abs(svd["k_eff"] - tbl["k_eff"]) / tbl["k_eff"] * 1e5
        mem_ratio = tbl["xs_memory_kb"] / svd["xs_memory_kb"] if svd["xs_memory_kb"] > 0 else 0
        speed_ratio = tbl["sim_ms"] / svd["sim_ms"] if svd["sim_ms"] > 0 else 0

        print("\n  SVD vs Table:")
        print(f"    k_eff gap          = {dk:.0f} pcm")
        print(f"    Memory compression = {mem_ratio:.1f}x")
        print(f"    Simulation speedup = {speed_ratio:.2f}x")

    if "Rust SVD" in results and "OpenMC" in results:
        svd = results["Rust SVD"]
        omc = results["OpenMC"]
        dk = abs(svd["k_eff"] - omc["k_eff"]) / omc["k_eff"] * 1e5
        print("\n  SVD vs OpenMC:")
        print(f"    k_eff gap = {dk:.0f} pcm")

    # Save results
    out_file = os.path.join(WORK_DIR, "honesty_test_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_file}")
    print(f"\n  Experimental k_eff = {k_exp:.5f}")


if __name__ == "__main__":
    main()
