"""Run OpenMC's depletion solver on the same PWR pin cell + pwr_actinides
chain that `deplete_pwr.rs` runs, so we can put the U-235 / Pu-239 /
Xe-135 / Sm-149 trajectories from both codes side-by-side.

This script is the OpenMC half of ICSBEP gate #4. The Rust half lives
in `rust_prototype/src/bin/deplete_pwr.rs`; both consume
`chains/pwr_actinides.json`. The chain XML required by OpenMC is
constructed in-process from that JSON — no external chain library
dependency.

Usage (from project root, in WSL with `openmc` conda env active):
    python scripts/openmc_pwr_depletion.py \\
        --chain chains/pwr_actinides.json \\
        --steps 16 --hours-per-step 48 --power-w-per-cm 200 \\
        --out outputs/openmc_depletion_actinides_32d.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openmc
from openmc.deplete import Chain, CoupledOperator, IndependentOperator, PredictorIntegrator

DATA = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5"


def zaid_to_openmc_name(zaid: int) -> str:
    z = zaid // 1000
    a = zaid % 1000
    elements = {
        1: "H", 5: "B", 6: "C", 8: "O",
        40: "Zr", 26: "Fe",
        53: "I", 54: "Xe", 55: "Cs", 61: "Pm", 62: "Sm",
        92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm",
    }
    el = elements.get(z, f"Z{z}")
    return f"{el}{a}"


def default_endf_target(parent_zaid: int, mt: int) -> int | None:
    """ENDF/B default daughter ZAID for a (parent, MT) pair when the
    chain JSON doesn't enumerate yields explicitly. Mirrors the
    rules baked into our Rust depletion module's
    `chain.rs::default_yield_for_reaction`."""
    z = parent_zaid // 1000
    a = parent_zaid % 1000
    return {
        102: 1000 * z + (a + 1),         # (n,γ): (Z, A+1)
        16: 1000 * z + (a - 1),          # (n,2n): (Z, A-1)
        17: 1000 * z + (a - 2),          # (n,3n): (Z, A-2)
        103: 1000 * (z - 1) + a,         # (n,p):  (Z-1, A)
        107: 1000 * (z - 2) + (a - 3),   # (n,α):  (Z-2, A-3)
    }.get(mt)


def build_chain_from_json(chain_json_path: str) -> Chain:
    """Build an in-memory OpenMC Chain that mirrors our pwr_actinides.json.

    The chain JSON has a tiny subset of what OpenMC normally consumes
    (decay constants, branching ratios, per-(parent, MT) one-group
    cross sections, fission yields). We translate to OpenMC's chain
    XML structure inline, then load via Chain.from_xml.
    """
    with open(chain_json_path) as f:
        spec = json.load(f)

    root = ET.Element("depletion_chain")
    for n in spec["nuclides"]:
        zaid = int(n["zaid"])
        name = zaid_to_openmc_name(zaid)
        nuc = ET.SubElement(root, "nuclide", {"name": name, "decay_modes": "1" if n["branches"] else "0"})
        # Half-life. OpenMC chain wants seconds; ours has λ in 1/s.
        if n["decay_constant"] > 0:
            half_life = float(np.log(2) / n["decay_constant"])
            nuc.set("half_life", f"{half_life:.6e}")
        else:
            # Stable — OpenMC treats half_life absent as stable.
            pass
        # Decay modes.
        for br in n.get("branches", []):
            mode = ET.SubElement(
                nuc,
                "decay",
                {
                    "type": "beta-",
                    "target": zaid_to_openmc_name(int(br["daughter"])),
                    "branching_ratio": f"{float(br['ratio']):.4f}",
                },
            )

    # Reactions. Group by parent.
    reactions_by_parent: dict[int, list[dict]] = {}
    for r in spec["reactions"]:
        reactions_by_parent.setdefault(int(r["parent"]), []).append(r)
    for parent_zaid, rxns in reactions_by_parent.items():
        # Find or create the parent nuclide element.
        parent_name = zaid_to_openmc_name(parent_zaid)
        nuc_el = root.find(f"./nuclide[@name='{parent_name}']")
        if nuc_el is None:
            nuc_el = ET.SubElement(
                root, "nuclide", {"name": parent_name, "decay_modes": "0"}
            )
        # OpenMC chain reactions: name, target, Q-value, branching, type.
        nuc_el.set("reactions", str(len(rxns)))
        for r in rxns:
            mt = int(r["mt"])
            type_name = {18: "fission", 102: "(n,gamma)", 16: "(n,2n)"}.get(
                mt, f"MT={mt}"
            )
            attrs = {"type": type_name, "Q": "0.0"}
            if mt != 18:
                # (n,gamma) etc. need a target nuclide. If the JSON
                # has explicit yields, use the dominant daughter; if
                # the yields dict is missing, fall back to the
                # standard ENDF rule for that MT (e.g. (n,γ) →
                # (Z, A+1)). Our Rust chain loader uses the same
                # default; the converter must match it for the
                # OpenMC chain to behave equivalently.
                yields = r.get("yields", None)
                target_zaid: int | None = None
                if yields:
                    target_zaid = int(
                        max(yields.items(), key=lambda kv: float(kv[1]))[0]
                    )
                elif yields is None:
                    target_zaid = default_endf_target(parent_zaid, mt)
                # yields == {} explicitly means pure removal — leave
                # target_zaid as None.
                if target_zaid is not None:
                    attrs["target"] = zaid_to_openmc_name(target_zaid)
                    attrs["branching_ratio"] = "1.0"
                else:
                    attrs["target"] = "Nothing"
                    attrs["branching_ratio"] = "1.0"
            ET.SubElement(nuc_el, "reaction", attrs)
        # Fission yields for fissile nuclides.
        fission = next((r for r in rxns if int(r["mt"]) == 18), None)
        if fission and fission.get("yields"):
            ny = ET.SubElement(nuc_el, "neutron_fission_yields")
            ny.set("energies", "0.0253")
            data = ET.SubElement(ny, "fission_yields", {"energy": "0.0253"})
            products = ET.SubElement(data, "products")
            data_el = ET.SubElement(data, "data")
            names, vals = [], []
            for daughter_zaid, y in fission["yields"].items():
                names.append(zaid_to_openmc_name(int(daughter_zaid)))
                vals.append(f"{float(y):.6e}")
            products.text = " ".join(names)
            data_el.text = " ".join(vals)

    # Write to a temp file then load via Chain.from_xml.
    tree = ET.ElementTree(root)
    chain_xml_path = Path("/tmp/pwr_actinides_chain.xml")
    tree.write(chain_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"  wrote {chain_xml_path}")

    try:
        return Chain.from_xml(str(chain_xml_path))
    except Exception as e:
        print("  Chain.from_xml failed — XML structure mismatch with OpenMC's expected format:")
        print(f"  {e}")
        raise


def build_pin_cell():
    fuel = openmc.Material(name="UO2 3.1%")
    fuel.add_nuclide("U235", 7.19e-4, "ao")
    fuel.add_nuclide("U238", 2.2482e-2, "ao")
    fuel.add_nuclide("O16", 4.6402e-2, "ao")
    fuel.set_density("g/cm3", 10.4)
    fuel.temperature = 900
    fuel.depletable = True
    fuel.volume = 0.4096 * 0.4096 * np.pi  # cm^3 per cm of axial length

    clad = openmc.Material(name="Zr-4")
    clad.add_nuclide("Zr90", 0.5063, "ao")
    clad.add_nuclide("Zr91", 0.1103, "ao")
    clad.add_nuclide("Zr92", 0.1686, "ao")
    clad.add_nuclide("Zr94", 0.1709, "ao")
    clad.set_density("g/cm3", 6.55)
    clad.temperature = 600

    water = openmc.Material(name="H2O")
    water.add_nuclide("H1", 2.0, "ao")
    water.add_nuclide("O16", 1.0, "ao")
    water.set_density("g/cm3", 0.74)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = 600

    mats = openmc.Materials([fuel, clad, water])
    mats.cross_sections = f"{DATA}/cross_sections.xml"

    fc = openmc.ZCylinder(r=0.4096)
    ci = openmc.ZCylinder(r=0.4180)
    co = openmc.ZCylinder(r=0.4750)
    box = openmc.model.RectangularPrism(1.26, 1.26, boundary_type="reflective")

    fuel_cell = openmc.Cell(fill=fuel, region=-fc)
    gap_cell = openmc.Cell(region=+fc & -ci)
    clad_cell = openmc.Cell(fill=clad, region=+ci & -co)
    water_cell = openmc.Cell(fill=water, region=+co & -box)

    geom = openmc.Geometry(openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, water_cell]))
    return mats, geom, fuel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chain", required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--hours-per-step", type=float, default=48.0)
    p.add_argument("--power-w-per-cm", type=float, default=200.0)
    p.add_argument("--particles", type=int, default=10_000)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--inactive", type=int, default=15)
    p.add_argument("--workdir", default=os.path.expanduser("~/openmc_pwr_dep"))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    chain_abs = os.path.abspath(args.chain)
    out_abs = os.path.abspath(args.out)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)

    print(f"Building chain from {chain_abs} ...")
    chain = build_chain_from_json(chain_abs)
    print(f"  Chain has {len(chain)} nuclides")

    mats, geom, fuel = build_pin_cell()
    mats.export_to_xml()
    geom.export_to_xml()

    s = openmc.Settings()
    s.batches = args.batches
    s.inactive = args.inactive
    s.particles = args.particles
    s.run_mode = "eigenvalue"
    s.temperature = {"method": "interpolation"}
    s.export_to_xml()

    operator = CoupledOperator(
        openmc.Model(geometry=geom, settings=s, materials=mats),
        chain_file="/tmp/pwr_actinides_chain.xml",
        normalization_mode="energy-deposition",
    )
    timesteps = [args.hours_per_step * 3600.0] * args.steps
    power = args.power_w_per_cm * 1.26  # W per pin
    integrator = PredictorIntegrator(
        operator, timesteps, power=power, timestep_units="s"
    )
    print(f"Running {args.steps} predictor-only depletion steps × {args.hours_per_step} h ...")
    integrator.integrate()

    # Pull results.
    results = openmc.deplete.Results.from_hdf5("depletion_results.h5")
    out = {
        "steps": args.steps,
        "hours_per_step": args.hours_per_step,
        "power_w_per_cm": args.power_w_per_cm,
        "trajectories": {},
    }
    for nuclide in ["U235", "U238", "Pu239", "Pu240", "Pu241", "Xe135", "Sm149", "Cs135"]:
        try:
            t, n = results.get_atoms(fuel, nuclide)
            out["trajectories"][nuclide] = {"t_s": t.tolist(), "atoms": n.tolist()}
            print(f"  {nuclide}: t[0]={t[0]:.0f}s n[0]={n[0]:.3e} → t[-1]={t[-1]:.0f}s n[-1]={n[-1]:.3e}")
        except Exception as e:
            print(f"  {nuclide}: not in results ({e})")

    with open(out_abs, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_abs}")


if __name__ == "__main__":
    main()
