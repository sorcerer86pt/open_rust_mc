"""OpenMC cross-code run for coupled n-gamma PWR pin cell gamma heating.

Mirrors bin/pwr_gamma_heating.rs geometry exactly (same materials,
dimensions, reflective lattice) and measures where gamma-ray source
energy born from (n,gamma), (n,f) and inelastic scatter ends up
depositing. Serves as the independent cross-check called out in
resume.md for the ~84/9/6 Rust split.

How photon-specific heating is isolated
---------------------------------------
OpenMC's `heating` score with ParticleFilter("photon") does not cleanly
return photon-transport deposition (verified empirically — scores near-
zero despite non-zero photon flux). The robust way to isolate photon
heating in OpenMC 0.15.3 is to run the problem twice:

  H_off(cell) = heating with photon_transport=False
              = neutron kerma only, photons are NOT produced as particles
  H_on(cell)  = heating with photon_transport=True
              = neutron kerma + photon deposition after transport

The per-cell photon deposition is then:

  E_photon(cell) = H_on(cell) - H_off(cell)

This is what the Rust binary's Phase-2 tally produces: the spatial
distribution of energy from gammas born at (n,gamma)/(n,f)/(n,n')
sites after transport. Intended for WSL (conda openmc env):

    source ~/miniforge3/bin/activate openmc
    python scripts/openmc_pwr_gamma_heating.py \\
        --batches 150 --inactive 50 --particles 20000

Caveats
-------
  - Gap cell is void (no material), so both codes land at 0 % there by
    construction. The Rust ~1.5 % gap is a CSDA electron-range
    artefact; this script is the evidence.
  - `electron_treatment="ttb"` (thick-target bremsstrahlung, no
    electron transport) matches the Rust binary's kerma-style
    treatment, making the comparison apples-to-apples.
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
OUT = "/mnt/c/Users/fog/madman_svd_experiment/outputs/openmc_pwr_gamma_heating.json"

FUEL_OR = 0.4096
CLAD_IR = 0.4180
CLAD_OR = 0.4750
PITCH = 1.2600

CELL_ORDER = ["fuel", "gap", "clad", "water"]

# Rust pwr_gamma_heating reference (resume.md, PR #3)
RUST_REF = {"fuel": 84.42, "gap": 1.46, "clad": 7.88, "water": 5.90}


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

    fc = openmc.ZCylinder(r=FUEL_OR)
    ci = openmc.ZCylinder(r=CLAD_IR)
    co = openmc.ZCylinder(r=CLAD_OR)
    box = openmc.model.RectangularPrism(PITCH, PITCH, boundary_type="reflective")

    fuel_cell = openmc.Cell(name="fuel", fill=fuel, region=-fc)
    gap_cell = openmc.Cell(name="gap", region=+fc & -ci)
    clad_cell = openmc.Cell(name="clad", fill=clad, region=+ci & -co)
    water_cell = openmc.Cell(name="water", fill=water, region=+co & -box)

    cells = {"fuel": fuel_cell, "gap": gap_cell, "clad": clad_cell, "water": water_cell}
    geom = openmc.Geometry(openmc.Universe(cells=list(cells.values())))
    return mats, geom, cells


def run_one(work_dir: Path, seed: int, batches: int, inactive: int,
            particles: int, photon_transport: bool):
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    mats, geom, cells = build_model()
    mats.export_to_xml()
    geom.export_to_xml()

    s = openmc.Settings()
    s.run_mode = "eigenvalue"
    s.batches = batches
    s.inactive = inactive
    s.particles = particles
    s.seed = seed
    s.temperature = {"method": "interpolation"}
    s.photon_transport = photon_transport
    if photon_transport:
        s.electron_treatment = "ttb"
        s.cutoff = {"energy_photon": 1.0e3}
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box([-FUEL_OR, -FUEL_OR, -1], [FUEL_OR, FUEL_OR, 1])
    )
    s.export_to_xml()

    cf = openmc.CellFilter([cells[n] for n in CELL_ORDER])
    heat = openmc.Tally(name="heating")
    heat.filters = [cf]
    heat.scores = ["heating"]
    kappa = openmc.Tally(name="kappa_fission")
    kappa.filters = [cf]
    kappa.scores = ["kappa-fission"]
    openmc.Tallies([heat, kappa]).export_to_xml()

    t0 = time.time()
    openmc.run(output=False)
    dt = time.time() - t0

    with openmc.StatePoint(work_dir / f"statepoint.{batches}.h5") as sp:
        k = float(sp.keff.nominal_value)
        k_sig = float(sp.keff.std_dev)
        h_mean = sp.get_tally(name="heating").mean.ravel()
        h_sig = sp.get_tally(name="heating").std_dev.ravel()
        kf_mean = sp.get_tally(name="kappa_fission").mean.ravel()

    return {
        "k": k,
        "k_sig": k_sig,
        "heating": h_mean,
        "heating_sig": h_sig,
        "kappa_fission": kf_mean,
        "runtime_s": dt,
    }


def print_report(off: dict, on: dict, photon: np.ndarray, photon_sig: np.ndarray,
                 frac: np.ndarray, frac_sig: np.ndarray):
    photon_total = float(photon.sum())
    print(f"k_eff (ptran=OFF) = {off['k']:.5f} +/- {off['k_sig']:.5f}")
    print(f"k_eff (ptran=ON)  = {on['k']:.5f} +/- {on['k_sig']:.5f}")
    print(f"runtime: off={off['runtime_s']:.1f} s, on={on['runtime_s']:.1f} s")
    print(f"\ntotal heating OFF = {off['heating'].sum():.3e} eV/src (neutron kerma only)")
    print(f"total heating ON  = {on['heating'].sum():.3e} eV/src (neutron + photon)")
    print(f"photon component  = {photon_total:.3e} eV/src")
    print()
    print(f"{'region':<6} {'H_off':>11} {'H_on':>11} {'photon':>11} "
          f"{'% of photon':>13} {'Rust':>8} {'Δ':>8}")
    for i, name in enumerate(CELL_ORDER):
        ref = RUST_REF.get(name, 0.0)
        pct = 100.0 * frac[i]
        print(
            f"{name:<6} {off['heating'][i]:>11.3e} {on['heating'][i]:>11.3e}"
            f" {photon[i]:>11.3e}"
            f" {pct:>11.3f} %"
            f" {ref:>6.2f} %"
            f" {pct - ref:>+7.2f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=150)
    p.add_argument("--inactive", type=int, default=50)
    p.add_argument("--particles", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workdir_off", default=os.path.expanduser("~/omc_pwr_gh_off"))
    p.add_argument("--workdir_on", default=os.path.expanduser("~/omc_pwr_gh_on"))
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    print(f"== photon_transport = OFF ==  ({args.workdir_off})")
    off = run_one(Path(args.workdir_off), args.seed, args.batches, args.inactive,
                  args.particles, photon_transport=False)
    print(f"  done: {off['runtime_s']:.1f} s")

    print(f"== photon_transport = ON ==  ({args.workdir_on})")
    on = run_one(Path(args.workdir_on), args.seed, args.batches, args.inactive,
                 args.particles, photon_transport=True)
    print(f"  done: {on['runtime_s']:.1f} s")

    # Photon heating = total heating(on) - total heating(off).
    # Per-cell uncertainty via quadrature (runs are independent).
    photon = on["heating"] - off["heating"]
    photon_sig = np.sqrt(on["heating_sig"] ** 2 + off["heating_sig"] ** 2)
    photon_total = float(photon.sum())
    frac = photon / photon_total if photon_total > 0 else np.zeros_like(photon)
    # Fraction uncertainty (first-order) assumes correlation with sum is small
    # (it is dominated by the fuel cell, so the other cells' fractions are
    # quasi-independent of the total; this is a loose upper bound).
    frac_sig = np.abs(photon_sig / photon_total) if photon_total > 0 else np.zeros_like(photon)

    print_report(off, on, photon, photon_sig, frac, frac_sig)

    rows = []
    for i, name in enumerate(CELL_ORDER):
        rows.append({
            "cell": name,
            "heating_off_ev_per_src": float(off["heating"][i]),
            "heating_off_sig": float(off["heating_sig"][i]),
            "heating_on_ev_per_src": float(on["heating"][i]),
            "heating_on_sig": float(on["heating_sig"][i]),
            "photon_heating_ev_per_src": float(photon[i]),
            "photon_heating_sig": float(photon_sig[i]),
            "photon_fraction": float(frac[i]),
            "photon_fraction_sig": float(frac_sig[i]),
            "rust_fraction": RUST_REF.get(name, 0.0) / 100.0,
        })

    payload = {
        "method": "heating(photon_transport=on) - heating(photon_transport=off)",
        "run": {
            "batches": args.batches,
            "inactive": args.inactive,
            "particles": args.particles,
            "seed": args.seed,
        },
        "k_eff_off": {"mean": off["k"], "sigma": off["k_sig"]},
        "k_eff_on": {"mean": on["k"], "sigma": on["k_sig"]},
        "runtime_off_s": off["runtime_s"],
        "runtime_on_s": on["runtime_s"],
        "photon_heating_total_ev_per_src": photon_total,
        "by_cell": rows,
        "rust_reference": RUST_REF,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
