# SPDX-License-Identifier: MIT
"""Build the OpenMC input for HEU-SOL-THERM-001 case-1 from the
identical NMC bundle that `tests/icsbep_runs.rs` runs, and execute
under the same ENDF/B-VII.1 HDF5 library. The goal is an A/B vs the
−1846 pcm bias our engine reports — if OpenMC also lands at
≈ −1800 pcm, the bias is in the NMC bundle / data-evaluation
combination. If OpenMC lands near k_ref, our engine has a real bug.

Runs inside the openmc/openmc:latest Docker image so the openmc
Python module is available without polluting the host.

Usage:
  python scripts/openmc_heu_sol_therm.py   # just generates files
  Then run from WSL:
    docker run --rm \\
      -v "/mnt/c/Users/fog/madman_svd_experiment:/work" \\
      -w /work/scripts/openmc_run \\
      -e OPENMC_CROSS_SECTIONS=/work/data/endfb-vii.1-hdf5/cross_sections.xml \\
      openmc/openmc:latest openmc -p 4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build_openmc():
    import openmc

    case = json.loads(Path("bench/icsbep/heu-sol-therm-001_case-1.json").read_text())
    bench = case["benchmark"]
    scene = case["scene"]

    # ── Materials ──────────────────────────────────────────────────────
    mats = []
    for mdef in scene["materials"]:
        m = openmc.Material(name=mdef["name"])
        m.temperature = mdef.get("temperature_K") or mdef.get("temperature") or 293.6
        for n in mdef.get("nuclides", []):
            zaid = n["zaid"]
            z, a = divmod(zaid, 1000)
            name = openmc.data.gnds_name(z, a)
            m.add_nuclide(name, n["atom_density"])
        if mdef.get("thermal_files"):
            for tf in mdef["thermal_files"]:
                stem = Path(tf).stem  # e.g. "c_H_in_H2O"
                m.add_s_alpha_beta(stem)
        m.set_density("sum")
        mats.append(m)
    materials = openmc.Materials(mats)

    # ── Geometry ───────────────────────────────────────────────────────
    surfs = []
    for sdef in scene["surfaces"]:
        bc = sdef.get("bc", "Transmission").lower()
        bc_map = {
            "transmission": "transmission",
            "vacuum": "vacuum",
            "reflective": "reflective",
        }
        t = sdef["type"]
        if t == "CylinderZ":
            s = openmc.ZCylinder(
                x0=sdef["center_x"], y0=sdef["center_y"], r=sdef["radius"],
                boundary_type=bc_map[bc],
            )
        elif t == "PlaneZ":
            s = openmc.ZPlane(z0=sdef["z0"], boundary_type=bc_map[bc])
        else:
            raise ValueError(f"unsupported surface type {t}")
        surfs.append(s)

    def region_from(node):
        op = node["op"]
        if op == "HalfSpace":
            s = surfs[node["surface_idx"]]
            return +s if node["positive"] else -s
        if op == "Intersection":
            return region_from(node["left"]) & region_from(node["right"])
        if op == "Union":
            return region_from(node["left"]) | region_from(node["right"])
        if op == "Complement":
            return ~region_from(node["inner"])
        raise ValueError(f"unsupported op {op}")

    cells = []
    for cdef in scene["cells"]:
        c = openmc.Cell(name=f"cell_{cdef['id']}", region=region_from(cdef["region"]))
        c.temperature = cdef.get("temperature", 293.6)
        fill = cdef["fill"]
        if fill["type"] == "Material":
            c.fill = mats[fill["material_idx"]]
        else:
            c.fill = None  # void
        cells.append(c)

    root = openmc.Universe(cells=cells)
    geom = openmc.Geometry(root)

    # ── Settings ───────────────────────────────────────────────────────
    settings = openmc.Settings()
    settings.batches = 80
    settings.inactive = 20
    settings.particles = 50_000
    settings.run_mode = "eigenvalue"
    # Source: cylindrical, inside the solution.
    settings.source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=openmc.stats.Uniform(0.0, 13.0),
            phi=openmc.stats.Uniform(0.0, 6.283185),
            z=openmc.stats.Uniform(1.0, 30.0),
        ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Watt(),
    )
    settings.temperature = {"method": "interpolation"}

    outdir = Path("scripts/openmc_run")
    outdir.mkdir(parents=True, exist_ok=True)
    model = openmc.Model(geom, materials, settings)
    model.export_to_model_xml(outdir / "model.xml")
    print(f"wrote {outdir / 'model.xml'}")
    print(f"benchmark k_ref = {bench['k_eff_reference']:.5f} "
          f"± {bench['k_eff_sigma']:.5f}")
    return outdir


if __name__ == "__main__":
    try:
        outdir = build_openmc()
    except ImportError:
        sys.exit(
            "openmc not importable on host. Run this inside the openmc docker "
            "image:\n  docker run --rm -v $PWD:/work -w /work "
            "openmc/openmc:latest python /work/scripts/openmc_heu_sol_therm.py"
        )
