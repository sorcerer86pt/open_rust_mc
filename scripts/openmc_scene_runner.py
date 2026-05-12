"""Run any of our `bench/icsbep/*.json` scenes through OpenMC at matched
statistics, for cross-code k_eff comparison.

Built to localise the HMF-008 −600 pcm gap reported by `metal_stats_diag`
when our engine ran the same scene JSON: agreement to ~60 pcm CPU↔GPU
ruled out a backend-specific bug, so the next axis is "do we agree with
OpenMC on this scene?".

Translates our scene-JSON DTOs (the same format `geometry::scene_io`
parses) into OpenMC objects:
* `surfaces[].type` →  openmc.Sphere / ZPlane / XCylinder / YCylinder
                       / ZCylinder, with `bc` mapped to
                       transmission / vacuum / reflective.
* `cells[].region`  →  recursive walk over Intersection / Union /
                       Complement / HalfSpace nodes.
* `cells[].fill`    →  Material / Void / Universe.
* `materials[]`     →  openmc.Material with atom-density nuclide entries
                       (atom densities sum to determine `set_density`).

Usage (WSL + docker, mirroring `openmc_godiva_tallies.py`):
    docker run --rm \
      -v /mnt/c/Users/fog/madman_svd_experiment:/mnt/c/Users/fog/madman_svd_experiment \
      -w /mnt/c/Users/fog/madman_svd_experiment \
      openmc/openmc:latest python scripts/openmc_scene_runner.py \
          bench/icsbep/heu-met-fast-008.json outputs/openmc_hmf008.json \
          --particles 20000 --batches 100 --inactive 20 --seeds 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import openmc

# ── ZAID → openmc nuclide-name mapping ────────────────────────────────
#
# OpenMC's `add_nuclide` takes ENDF-style names like "U235", "Fe56",
# "Cu63", etc. Our scene JSON ships either an `hdf5_file` ("U235.h5")
# or a `zaid` (92235). Both encode the same isotope; convert into the
# OpenMC name via the standard Z → element-symbol table.

ELEMENTS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 90: "Th", 91: "Pa",
    92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm",
}


def zaid_to_openmc_name(zaid: int, label: str | None) -> str:
    """Convert ZAID (1000·Z + A) to OpenMC nuclide name.

    Handles the natural-abundance convention (A = 0 → element name,
    e.g. ZAID=6000 → "C0" for natural carbon).
    """
    z = zaid // 1000
    a = zaid % 1000
    sym = ELEMENTS.get(z)
    if sym is None:
        raise ValueError(f"unknown Z={z} (zaid={zaid}, label={label!r})")
    if a == 0:
        # OpenMC names natural-abundance elements as e.g. "C0".
        return f"{sym}0"
    return f"{sym}{a}"


def hdf5_file_to_openmc_name(filename: str) -> str:
    """`U235.h5` → `U235`, `c_H_in_H2O.h5` → `c_H_in_H2O`."""
    return filename.rsplit("/", 1)[-1].removesuffix(".h5")


# ── Surface translation ───────────────────────────────────────────────


BC_MAP = {
    "Transmission": "transmission",
    "Vacuum": "vacuum",
    "Reflective": "reflective",
    "Periodic": "periodic",
    "White": "white",
}


def build_surfaces(dtos: list[dict]) -> list[openmc.Surface]:
    """Translate scene-JSON surfaces in their original order so cell
    `surface_idx` references resolve directly into the returned list."""
    out = []
    for i, s in enumerate(dtos):
        bc = BC_MAP.get(s.get("bc", "Transmission"), "transmission")
        t = s["type"]
        if t == "Sphere":
            c = s["center"]
            surf = openmc.Sphere(x0=c[0], y0=c[1], z0=c[2], r=s["radius"], boundary_type=bc)
        elif t == "PlaneX":
            surf = openmc.XPlane(x0=s["x0"], boundary_type=bc)
        elif t == "PlaneY":
            surf = openmc.YPlane(y0=s["y0"], boundary_type=bc)
        elif t == "PlaneZ":
            surf = openmc.ZPlane(z0=s["z0"], boundary_type=bc)
        elif t == "Plane":
            n = s["normal"]
            surf = openmc.Plane(
                a=n[0], b=n[1], c=n[2], d=s["offset"], boundary_type=bc
            )
        elif t == "CylinderX":
            surf = openmc.XCylinder(
                y0=s["center_y"], z0=s["center_z"], r=s["radius"], boundary_type=bc
            )
        elif t == "CylinderY":
            surf = openmc.YCylinder(
                x0=s["center_x"], z0=s["center_z"], r=s["radius"], boundary_type=bc
            )
        elif t == "CylinderZ":
            surf = openmc.ZCylinder(
                x0=s["center_x"], y0=s["center_y"], r=s["radius"], boundary_type=bc
            )
        else:
            raise NotImplementedError(f"surface[{i}] type {t!r} not translated")
        out.append(surf)
    return out


# ── Region tree translation ───────────────────────────────────────────


def build_region(node: dict, surfaces: list[openmc.Surface]) -> openmc.Region:
    """Recursive walk over our JSON region tree.

    Schema (mirrors `geometry::scene_io::RegionDto`):
    * `{op: "HalfSpace", surface_idx, positive}` →  ±surface
    * `{op: "Intersection"|"Union", children: […]}` → reduce(& | over children)
    * `{op: "Complement", child: …}` → ~child
    """
    op = node["op"]
    if op == "HalfSpace":
        surf = surfaces[node["surface_idx"]]
        return +surf if node["positive"] else -surf
    if op == "Intersection":
        kids = [build_region(c, surfaces) for c in _children(node)]
        if not kids:
            raise ValueError("empty Intersection region")
        out = kids[0]
        for k in kids[1:]:
            out = out & k
        return out
    if op == "Union":
        kids = [build_region(c, surfaces) for c in _children(node)]
        if not kids:
            raise ValueError("empty Union region")
        out = kids[0]
        for k in kids[1:]:
            out = out | k
        return out
    if op == "Complement":
        return ~build_region(node["child"], surfaces)
    raise NotImplementedError(f"region op {op!r} not translated")


def _children(node: dict) -> list[dict]:
    """Tolerate both `children: […]` and `left / right` shapes."""
    if "children" in node:
        return node["children"]
    if "left" in node and "right" in node:
        return [node["left"], node["right"]]
    if "left" in node:
        return [node["left"]]
    raise ValueError(f"region node has no children: {node!r}")


# ── Material translation ──────────────────────────────────────────────


def build_materials(dtos: list[dict]) -> list[openmc.Material]:
    """Build OpenMC Materials with atom densities matching our JSON.

    Our JSON stores per-nuclide `atom_density` in atoms/(barn·cm). OpenMC
    accepts a list of (name, fraction) pairs plus an overall density. We
    do this by summing atom densities for total `1/(barn·cm)` and giving
    each nuclide its atom fraction.
    """
    out = []
    for mi, m in enumerate(dtos):
        mat = openmc.Material(name=m["name"])
        total_ad = sum(n["atom_density"] for n in m["nuclides"])
        if total_ad <= 0:
            raise ValueError(f"material[{mi}] {m['name']} has zero atom density")
        for n in m["nuclides"]:
            if "hdf5_file" in n and n["hdf5_file"]:
                name = hdf5_file_to_openmc_name(n["hdf5_file"])
            elif "zaid" in n and n["zaid"]:
                name = zaid_to_openmc_name(int(n["zaid"]), n.get("label"))
            else:
                raise ValueError(f"nuclide in material[{mi}] has neither hdf5_file nor zaid")
            mat.add_nuclide(name, n["atom_density"] / total_ad, "ao")
        # Density in atoms/(barn·cm) → OpenMC's "atom/b-cm" unit.
        mat.set_density("atom/b-cm", total_ad)
        if "temperature" in m:
            mat.temperature = m["temperature"]
        # Per-nuclide thermal-file metadata isn't loaded here — every
        # ICSBEP case we're cross-checking is fast-spectrum metal where
        # S(α,β) is irrelevant. Extend if/when LCT or HEU-SOL is run
        # through this script.
        out.append(mat)
    return out


# ── Cell + universe assembly ──────────────────────────────────────────


def build_cells(
    dtos: list[dict],
    surfaces: list[openmc.Surface],
    materials: list[openmc.Material],
) -> list[openmc.Cell]:
    cells = []
    for ci, c in enumerate(dtos):
        region = build_region(c["region"], surfaces)
        fill = c["fill"]
        ft = fill["type"]
        if ft == "Material":
            cell_fill = materials[fill["material_idx"]]
        elif ft == "Void":
            cell_fill = None
        else:
            raise NotImplementedError(f"cell[{ci}] fill {ft!r} not yet translated")
        cell = openmc.Cell(cell_id=c["id"], region=region, fill=cell_fill)
        if "temperature" in c and cell_fill is not None:
            # OpenMC supports per-cell temperature override on
            # material-filled cells.
            cell.temperature = c["temperature"]
        cells.append(cell)
    return cells


# ── Top-level scene runner ───────────────────────────────────────────


def run_scene(
    scene_path: str,
    output_path: str,
    *,
    particles: int,
    batches: int,
    inactive: int,
    seeds: int,
    cross_sections: str,
):
    with open(scene_path) as fh:
        scene_json = json.load(fh)
    bench = scene_json["benchmark"]
    scene = scene_json["scene"]
    k_ref = bench["k_eff_reference"]
    sigma_exp = bench["k_eff_sigma"]
    case_id = bench["case_id"]

    print(f"\n=== {case_id} via OpenMC ===")
    print(f"  k_ref = {k_ref:.5f} ± {sigma_exp:.5f}  (handbook / openmc-validation)")
    print(
        f"  scene: {len(scene['surfaces'])} surfaces, "
        f"{len(scene['cells'])} cells, "
        f"{len(scene.get('universes', []))} universes, "
        f"{len(scene.get('materials', []))} materials"
    )

    work = f"/tmp/openmc_{case_id.lower().replace('-', '_')}"
    os.makedirs(work, exist_ok=True)
    os.chdir(work)

    surfaces = build_surfaces(scene["surfaces"])
    materials = build_materials(scene["materials"])
    cells = build_cells(scene["cells"], surfaces, materials)

    omc_mats = openmc.Materials(materials)
    omc_mats.cross_sections = cross_sections
    omc_mats.export_to_xml()

    root = openmc.Universe(cells=cells)
    geom = openmc.Geometry(root)
    geom.export_to_xml()

    # Per-cell + per-region tallies for downstream localisation. The
    # cell filter pegs the reaction-rate breakdown to every cell so we
    # can see exactly where in the reflector neutrons are lost.
    cell_filter = openmc.CellFilter([c for c in cells if c.fill is not None])
    tallies = openmc.Tallies()
    for score in ["fission", "absorption", "scatter", "elastic", "(n,gamma)"]:
        t = openmc.Tally(name=f"rate_{score}")
        t.filters = [cell_filter]
        t.scores = [score]
        tallies.append(t)
    tallies.export_to_xml()

    per_seed = []
    for seed in range(seeds):
        s = openmc.Settings()
        s.batches = batches
        s.inactive = inactive
        s.particles = particles
        s.seed = seed + 1
        src = openmc.IndependentSource()
        # Use a uniform point source at origin — simple and adequate;
        # benchmarks are critical with k well-converged in 80 batches.
        src.space = openmc.stats.Point((0, 0, 0))
        src.energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
        s.source = [src]
        s.export_to_xml()

        t0 = time.time()
        openmc.run(output=False)
        dt = time.time() - t0
        sp = openmc.StatePoint(f"statepoint.{batches}.h5")
        k = sp.keff
        # Per-tally seed-mean export. Only keep what we need —
        # rate_by_score arrays per-cell.
        rates = {}
        for tally in sp.tallies.values():
            mean = np.asarray(tally.mean).flatten()
            std = np.asarray(tally.std_dev).flatten()
            rates[tally.name] = {"mean": mean.tolist(), "std": std.tolist()}
        per_seed.append(
            {
                "seed": seed,
                "k": float(k.nominal_value),
                "sigma_k": float(k.std_dev),
                "time_s": dt,
                "rates": rates,
            }
        )
        print(
            f"  seed {seed}: k = {k.nominal_value:.5f} ± {k.std_dev:.5f}  ({dt:.1f}s)",
            flush=True,
        )
        sp.close()
        for f in (f"statepoint.{batches}.h5", "tallies.out", "summary.h5"):
            if os.path.exists(f):
                os.remove(f)

    ks = [r["k"] for r in per_seed]
    mean = float(np.mean(ks))
    sigma_seeds = float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0
    delta = mean - k_ref
    delta_pcm = delta * 1.0e5
    sigma_combined = (sigma_seeds**2 + sigma_exp**2) ** 0.5

    agg = {
        "case_id": case_id,
        "scene_json": os.path.basename(scene_path),
        "k_mean": mean,
        "sigma_seeds": sigma_seeds,
        "k_ref": k_ref,
        "sigma_exp": sigma_exp,
        "delta_pcm": delta_pcm,
        "n_sigma_combined": abs(delta) / sigma_combined if sigma_combined > 0 else 0.0,
        "settings": {
            "batches": batches,
            "inactive": inactive,
            "particles": particles,
            "seeds": seeds,
        },
        "per_seed": per_seed,
    }
    with open(output_path, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(
        f"\n  mean k = {mean:.5f} ± {sigma_seeds:.5f} (seed σ)   "
        f"Δ_ICSBEP = {delta_pcm:+.0f} pcm   {agg['n_sigma_combined']:.2f}σ_combined"
    )
    print(f"  wrote {output_path}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_json", help="bench/icsbep/*.json")
    ap.add_argument("output", help="output aggregate JSON (per-seed + mean)")
    ap.add_argument("--particles", type=int, default=20_000)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--inactive", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--cross-sections",
        default="/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5/cross_sections.xml",
        help="Path to OpenMC cross_sections.xml (defaults to our local ENDF/B-VII.1 library).",
    )
    args = ap.parse_args()

    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)
    run_scene(
        args.scene_json,
        args.output,
        particles=args.particles,
        batches=args.batches,
        inactive=args.inactive,
        seeds=args.seeds,
        cross_sections=args.cross_sections,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
