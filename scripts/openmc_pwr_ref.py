"""Run OpenMC PWR pin cell with 5 seeds to produce a reference k_inf for the
Pareto plot. Matches the open_rust_mc pwr_pincell geometry exactly:
  - 3.1% enriched UO2 at 900 K (U235/U238/O16)
  - Zircaloy-4 clad at 600 K (Zr90/91/92/94 -- Zr96 zeroed for parity)
  - H2O at 600 K with c_H_in_H2O S(alpha,beta)

Intended for WSL (conda openmc env):
  source ~/miniforge3/bin/activate openmc
  python scripts/openmc_pwr_ref.py --seeds 5 --particles 10000 --batches 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import openmc

DATA = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5"


def build_model():
    fuel = openmc.Material(name="UO2 3.1%")
    fuel.add_nuclide("U235", 0.031)
    fuel.add_nuclide("U238", 0.969)
    fuel.add_nuclide("O16", 2.0)
    fuel.set_density("g/cm3", 10.4)
    fuel.temperature = 900

    clad = openmc.Material(name="Zr-4")
    clad.add_nuclide("Zr90", 0.5063)
    clad.add_nuclide("Zr91", 0.1103)
    clad.add_nuclide("Zr92", 0.1686)
    clad.add_nuclide("Zr94", 0.1709)
    # Zr96 omitted to mirror open_rust_mc (9 nuclides total across materials)
    clad.set_density("g/cm3", 6.55)
    clad.temperature = 600

    water = openmc.Material(name="H2O")
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    water.set_density("g/cm3", 0.74)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = 600

    mats = openmc.Materials([fuel, clad, water])
    mats.cross_sections = f"{DATA}/cross_sections.xml"

    fuel_or = 0.4096
    clad_ir = 0.4180
    clad_or = 0.4750
    pitch = 1.2600
    fc = openmc.ZCylinder(r=fuel_or)
    ci = openmc.ZCylinder(r=clad_ir)
    co = openmc.ZCylinder(r=clad_or)
    box = openmc.model.RectangularPrism(pitch, pitch, boundary_type="reflective")

    fuel_cell = openmc.Cell(fill=fuel, region=-fc)
    gap_cell = openmc.Cell(region=+fc & -ci)
    clad_cell = openmc.Cell(fill=clad, region=+ci & -co)
    water_cell = openmc.Cell(fill=water, region=+co & -box)

    geom = openmc.Geometry(openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell]))
    return mats, geom, fuel_or


def run_one(work_dir: Path, seed: int, batches: int, inactive: int, particles: int):
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    mats, geom, fuel_or = build_model()
    mats.export_to_xml()
    geom.export_to_xml()

    s = openmc.Settings()
    s.batches = batches
    s.inactive = inactive
    s.particles = particles
    s.run_mode = "eigenvalue"
    s.temperature = {"method": "interpolation"}
    s.seed = seed
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box([-fuel_or, -fuel_or, -1], [fuel_or, fuel_or, 1])
    )
    s.export_to_xml()

    t0 = time.time()
    openmc.run(output=False)
    dt = time.time() - t0
    with openmc.StatePoint(f"statepoint.{batches}.h5") as sp:
        k = float(sp.keff.nominal_value)
        u = float(sp.keff.std_dev)
    return k, u, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--inactive", type=int, default=10)
    p.add_argument("--particles", type=int, default=10000)
    p.add_argument("--workdir", default=os.path.expanduser("~/openmc_pwr_ref"))
    p.add_argument("--out", default="/mnt/c/Users/fog/madman_svd_experiment/outputs/pareto/openmc_pwr.json")
    args = p.parse_args()

    rows = []
    for i in range(args.seeds):
        wd = Path(args.workdir) / f"seed{i}"
        seed = 1 + i  # OpenMC requires seed > 0
        k, u, dt = run_one(wd, seed, args.batches, args.inactive, args.particles)
        print(f"seed {seed}: k={k:.5f} +/- {u:.5f} ({dt:.1f} s)")
        rows.append({"seed": seed, "k": k, "sigma_batch": u, "time_s": dt})

    ks = np.array([r["k"] for r in rows])
    mean = float(ks.mean())
    std = float(ks.std(ddof=1)) if len(ks) > 1 else float(rows[0]["sigma_batch"])
    total = sum(r["time_s"] for r in rows)
    print(f"\nk_inf mean = {mean:.5f} +/- {std:.5f} ({args.seeds} seeds)")
    print(f"total time = {total:.1f} s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "mean": mean, "sigma_seeds": std, "seeds": rows,
            "batches": args.batches, "inactive": args.inactive, "particles": args.particles,
        }, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
