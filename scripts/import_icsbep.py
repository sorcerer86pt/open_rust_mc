#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bulk-import ICSBEP cases from open-source proxy repositories.

Walks the `mit-crpg/benchmarks` OpenMC XML tree (geometry/materials/settings)
and the `openmc-dev/validation` uncertainties.csv (k_ref + σ_exp per case),
emitting one JSON file per case in our `bench/icsbep/` directory.

Each emitted file follows the NMC scene-bundle schema (specs/nmc/) with two
blocks:

  - `benchmark`: matches manifest.json §3.1 — k_eff_reference, k_eff_sigma,
                 source, plus a `data_provenance` sub-block recording exactly
                 which open-source-proxy commit/SHA the geometry+materials
                 came from and which uncertainties.csv row supplied k_ref.

  - `scene`: matches scene.json — surfaces, cells, universes, materials,
             root_universe_id. Fed into `Geometry::from_json` once that
             deserializer lands. Until then `icsbep_bench` reports BLOCKED
             for these cases (no `runner` block invokes a CLI binary).

# Provenance and authoritativeness

The official source for ICSBEP benchmark specifications is the NEA/OECD
ICSBEP Handbook, which is gated behind a registration form at
https://www.oecd-nea.org/science/wpncs/icsbep/order.html. The handbook is
distributed by DVD and password-protected GitLab; its click-through
agreement explicitly prohibits redistribution. We cannot fetch it
automatically.

What this script uses instead:

  1. `mit-crpg/benchmarks` (Paul Romano et al., MIT-CRPG): hand-transcribed
     OpenMC inputs for ~116 ICSBEP cases, maintained by the same lab that
     develops OpenMC. Sourced from `/tmp/mit-crpg-benchmarks/icsbep/` (the
     user-provided shallow clone). Repo: github.com/mit-crpg/benchmarks.

  2. `openmc-dev/validation` (Paul Romano et al.): the canonical
     `uncertainties.csv` table mapping each ICSBEP `<case_id>, <sub-case>`
     to `<k_ref>, <σ_exp>` derived from the handbook. Sourced from
     `/tmp/openmc-validation/benchmarking/uncertainties.csv`. Repo:
     github.com/openmc-dev/validation.

Both repositories are industry-standard derivatives but are NOT the
canonical specifications. Every case file we emit carries a
`data_provenance` block citing both proxies so any downstream validation
report can re-verify against the registered handbook before publication.

# Usage

    python scripts/import_icsbep.py \\
        --mit-crpg  /tmp/mit-crpg-benchmarks/icsbep \\
        --openmc-validation /tmp/openmc-validation \\
        --output    bench/icsbep \\
        [--cases  pu-met-fast-001,heu-met-fast-001]   # filter

Outputs one `<case_id>[_<sub_case>].json` per case file. Skipped cases
(lattices, unsupported surfaces, unparseable region trees) are listed at
the end with the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

# Element symbol → atomic number Z, for ZAID = 1000·Z + A.
ELEMENT_Z: dict[str, int] = {
    "H": 1,   "He": 2,   "Li": 3,   "Be": 4,   "B": 5,   "C": 6,
    "N": 7,   "O": 8,    "F": 9,    "Ne": 10,  "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14,  "P": 15,   "S": 16,   "Cl": 17, "Ar": 18,
    "K": 19,  "Ca": 20,  "Sc": 21,  "Ti": 22,  "V": 23,  "Cr": 24,
    "Mn": 25, "Fe": 26,  "Co": 27,  "Ni": 28,  "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32,  "As": 33,  "Se": 34,  "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38,  "Y": 39,   "Zr": 40,  "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44,  "Rh": 45,  "Pd": 46,  "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50,  "Sb": 51,  "Te": 52,  "I": 53,  "Xe": 54,
    "Cs": 55, "Ba": 56,  "La": 57,  "Ce": 58,  "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62,  "Eu": 63,  "Gd": 64,  "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68,  "Tm": 69,  "Yb": 70,  "Lu": 71, "Hf": 72,
    "Ta": 73, "W": 74,   "Re": 75,  "Os": 76,  "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80,  "Tl": 81,  "Pb": 82,  "Bi": 83, "Po": 84,
    "At": 85, "Rn": 86,  "Fr": 87,  "Ra": 88,  "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,   "Np": 93,  "Pu": 94,  "Am": 95, "Cm": 96,
    "Bk": 97, "Cf": 98,  "Es": 99,  "Fm": 100,
}

CATEGORY_FROM_PREFIX: dict[str, str] = {
    "heu-met-fast":   "HEU-MET-FAST",
    "heu-met-inter":  "HEU-MET-INTER",
    "heu-comp-inter": "HEU-COMP-INTER",
    "heu-comp-therm": "HEU-COMP-THERM",
    "heu-sol-therm":  "HEU-SOL-THERM",
    "ieu-met-fast":   "IEU-MET-FAST",
    "ieu-comp-fast":  "IEU-COMP-FAST",
    "leu-comp-therm": "LEU-COMP-THERM",
    "leu-sol-therm":  "LEU-SOL-THERM",
    "pu-met-fast":    "PU-MET-FAST",
    "pu-met-inter":   "PU-MET-INTER",
    "pu-comp-inter":  "PU-COMP-INTER",
    "pu-sol-therm":   "PU-SOL-THERM",
    "mix-comp-fast":  "MIX-COMP-FAST",
    "mix-comp-therm": "MIX-COMP-THERM",
    "mix-met-fast":   "MIX-MET-FAST",
    "mix-met-inter":  "MIX-MET-INTER",
    "u233-met-fast":  "U233-MET-FAST",
    "u233-comp-therm": "U233-COMP-THERM",
    "u233-sol-inter": "U233-SOL-INTER",
    "u233-sol-therm": "U233-SOL-THERM",
    "spec-met-fast":  "SPEC-MET-FAST",
}


def canonical_case_id(slug: str) -> str:
    """`pu-met-fast-001` → `PU-MET-FAST-001` (ICSBEP canonical form)."""
    parts = slug.split("-")
    if len(parts) < 2:
        return slug.upper()
    return "-".join(p.upper() for p in parts)


def parse_nuclide_name(name: str) -> Optional[tuple[int, str]]:
    """`U235` → (92235, "U-235"); `Cd115_m1` → (48615, "Cd-115m"); None on parse failure."""
    m = re.match(r"^([A-Z][a-z]?)(\d+)(_m\d+)?$", name)
    if not m:
        return None
    sym, a_str, meta = m.groups()
    z = ELEMENT_Z.get(sym)
    if z is None:
        return None
    a = int(a_str)
    zaid = 1000 * z + a
    label = f"{sym}-{a}{meta.replace('_m', 'm') if meta else ''}"
    return zaid, label


@dataclass
class ParsedSurface:
    """Open_rust_mc surface in scene.json form."""
    id: int
    json: dict


@dataclass
class ParseStats:
    n_cases: int = 0
    n_emitted: int = 0
    n_skipped: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


def parse_surface(s: ET.Element) -> Optional[dict]:
    """OpenMC <surface> → open_rust_mc scene-schema dict, or None if unsupported.

    Supports both XML styles used in the mit-crpg corpus:
      - Attribute style: `<surface id="1" type="sphere" coeffs="0 0 0 5"
        boundary="vacuum"/>`
      - Child-element style: `<surface id="1"><type>sphere</type>
        <coeffs>0 0 0 5</coeffs><boundary>vacuum</boundary></surface>`
    """
    sid = int(s.attrib["id"])

    def field(name: str, default: str = "") -> str:
        if name in s.attrib:
            return s.attrib[name]
        child = s.find(name)
        if child is not None and child.text is not None:
            return child.text
        return default

    stype = field("type")
    if not stype:
        return None
    coeffs_raw = field("coeffs", "").split()
    coeffs = [float(x) for x in coeffs_raw]
    bc_raw = field("boundary", "transmission").lower()
    bc = {"vacuum": "Vacuum", "reflective": "Reflective", "transmission": "Transmission"}.get(
        bc_raw, "Transmission"
    )

    if stype == "sphere" and len(coeffs) == 4:
        return {
            "type": "Sphere",
            "center": coeffs[:3],
            "radius": coeffs[3],
            "bc": bc,
        }
    if stype == "z-cylinder" and len(coeffs) == 3:
        return {
            "type": "CylinderZ",
            "center_x": coeffs[0],
            "center_y": coeffs[1],
            "radius":   coeffs[2],
            "bc": bc,
        }
    if stype == "x-cylinder" and len(coeffs) == 3:
        return {
            "type": "CylinderX",
            "center_y": coeffs[0],
            "center_z": coeffs[1],
            "radius":   coeffs[2],
            "bc": bc,
        }
    if stype == "y-cylinder" and len(coeffs) == 3:
        return {
            "type": "CylinderY",
            "center_x": coeffs[0],
            "center_z": coeffs[1],
            "radius":   coeffs[2],
            "bc": bc,
        }
    if stype == "x-plane" and len(coeffs) == 1:
        return {"type": "PlaneX", "x0": coeffs[0], "bc": bc}
    if stype == "y-plane" and len(coeffs) == 1:
        return {"type": "PlaneY", "y0": coeffs[0], "bc": bc}
    if stype == "z-plane" and len(coeffs) == 1:
        return {"type": "PlaneZ", "z0": coeffs[0], "bc": bc}
    if stype == "plane" and len(coeffs) == 4:
        return {
            "type":   "Plane",
            "normal": coeffs[:3],
            "offset": coeffs[3],
            "bc":     bc,
        }
    return None


def parse_region(expr: str, sid_to_idx: dict[int, int]) -> Optional[dict]:
    """OpenMC region expression → scene-schema Region tree.

    Supports the subset used by all fast-metal ICSBEP cases:
      - Bare signed surface ID: `-3`  → HalfSpace(idx=sid_to_idx[3], positive=False)
                                `3`   → HalfSpace(..., positive=True)
      - Space-separated terms (intersection): `-1 2 -3`
      - Union `|` and parenthesised groups
      - Complement `~`

    Returns None if the expression contains unsupported tokens or maps to
    a surface ID we couldn't parse.
    """
    expr = expr.strip()
    if not expr:
        return None

    # Tokenise
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c in "()|~":
            tokens.append(c)
            i += 1
            continue
        # Signed surface ID
        m = re.match(r"-?\d+", expr[i:])
        if m:
            tokens.append(m.group(0))
            i += len(m.group(0))
            continue
        return None  # unknown token

    # Recursive-descent parser:
    #   expr     := union
    #   union    := intersect ('|' intersect)*
    #   intersect := unary (unary)*       (space-separated implicit AND)
    #   unary    := '~' unary | '(' union ')' | HALFSPACE
    pos = [0]

    def peek() -> Optional[str]:
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def consume() -> Optional[str]:
        t = peek()
        if t is not None:
            pos[0] += 1
        return t

    def parse_unary() -> Optional[dict]:
        t = peek()
        if t is None:
            return None
        if t == "~":
            consume()
            inner = parse_unary()
            if inner is None:
                return None
            return {"op": "Complement", "inner": inner}
        if t == "(":
            consume()
            inner = parse_union()
            if peek() != ")":
                return None
            consume()
            return inner
        # Must be a signed surface ID
        m = re.match(r"^(-?)(\d+)$", t)
        if not m:
            return None
        sign, sid_str = m.groups()
        sid = int(sid_str)
        if sid not in sid_to_idx:
            return None
        consume()
        # OpenMC convention: `-N` means inside surface N (negative
        # half-space). `N` means outside (positive half-space).
        return {
            "op": "HalfSpace",
            "surface_idx": sid_to_idx[sid],
            "positive": (sign == ""),
        }

    def parse_intersect() -> Optional[dict]:
        left = parse_unary()
        if left is None:
            return None
        while True:
            t = peek()
            if t is None or t in ("|", ")"):
                break
            right = parse_unary()
            if right is None:
                return None
            left = {"op": "Intersection", "left": left, "right": right}
        return left

    def parse_union() -> Optional[dict]:
        left = parse_intersect()
        if left is None:
            return None
        while peek() == "|":
            consume()
            right = parse_intersect()
            if right is None:
                return None
            left = {"op": "Union", "left": left, "right": right}
        return left

    region = parse_union()
    if region is None or pos[0] != len(tokens):
        return None
    return region


def parse_case(
    case_dir: Path,
    sub_label: str,
    full_case_id: str,
    category: str,
    k_ref: float,
    sigma_exp: float,
    mit_crpg_relpath: str,
    uncertainties_relpath: str,
    stats: ParseStats,
) -> Optional[dict]:
    """Build the case JSON for a single sub-case directory."""
    geometry_path = case_dir / "geometry.xml"
    materials_path = case_dir / "materials.xml"
    if not geometry_path.exists() or not materials_path.exists():
        stats.skipped.append((full_case_id, f"missing geometry.xml or materials.xml in {case_dir}"))
        return None

    try:
        geom_root = ET.parse(geometry_path).getroot()
        mat_root = ET.parse(materials_path).getroot()
    except ET.ParseError as e:
        stats.skipped.append((full_case_id, f"XML parse error: {e}"))
        return None

    # Lattices — collect first so cell-fill resolution can distinguish
    # `fill="<id>"` referring to a lattice vs a universe.
    rect_lattices: list[dict] = []
    lattice_id_to_idx: dict[int, int] = {}
    for lat in geom_root.findall("lattice"):
        lat_id = int(lat.attrib["id"])
        # `dimension="nx ny"` or `dimension="nx ny nz"`.
        dim_str = lat.attrib.get("dimension", "")
        dims = [int(x) for x in dim_str.split()]
        if len(dims) == 2:
            nx, ny = dims
            nz = 1
        elif len(dims) == 3:
            nx, ny, nz = dims
        else:
            stats.skipped.append((full_case_id, f"lattice {lat_id} bad dimension={dim_str!r}"))
            return None

        # `<lower_left>` and `<pitch>` are child elements with 2 or 3
        # space-separated floats. 2D lattices implicitly extend the
        # full z range — we map them to a large-z slab so the engine's
        # 3D-only RectLattice can store them.
        def parse_vec3_child(parent: ET.Element, tag: str) -> Optional[list[float]]:
            child = parent.find(tag)
            if child is None or child.text is None:
                return None
            vals = [float(x) for x in child.text.split()]
            return vals
        lower_left = parse_vec3_child(lat, "lower_left")
        pitch = parse_vec3_child(lat, "pitch")
        if lower_left is None or pitch is None:
            stats.skipped.append((full_case_id, f"lattice {lat_id} missing lower_left or pitch"))
            return None
        if len(lower_left) == 2:
            lower_left = lower_left + [-1.0e4]   # large slab → 2D effective
        if len(pitch) == 2:
            pitch = pitch + [2.0e4]

        universes_text = lat.find("universes")
        if universes_text is None or universes_text.text is None:
            stats.skipped.append((full_case_id, f"lattice {lat_id} missing <universes>"))
            return None
        universe_ids = [int(x) for x in universes_text.text.split()]
        expected_n = nx * ny * nz
        if len(universe_ids) != expected_n:
            stats.skipped.append((
                full_case_id,
                f"lattice {lat_id} universes count {len(universe_ids)} != nx·ny·nz {expected_n}",
            ))
            return None

        rect_lattices.append({
            "origin":             lower_left,
            "pitch":              pitch,
            "shape":              [nx, ny, nz],
            "universes":          universe_ids,
            "material_overrides": None,
        })
        lattice_id_to_idx[lat_id] = len(rect_lattices) - 1

    # Surfaces
    surfaces: list[dict] = []
    sid_to_idx: dict[int, int] = {}
    for s in geom_root.findall("surface"):
        sid = int(s.attrib["id"])
        json_surf = parse_surface(s)
        if json_surf is None:
            stats.skipped.append((full_case_id, f"unsupported surface type/coeffs: {ET.tostring(s).decode()}"))
            return None
        sid_to_idx[sid] = len(surfaces)
        surfaces.append(json_surf)

    # Materials
    materials: list[dict] = []
    mat_id_to_idx: dict[int, int] = {}
    for m in mat_root.findall("material"):
        mid = int(m.attrib["id"])
        name = m.attrib.get("name", f"mat_{mid}")
        nuclides_json: list[dict] = []
        sab_files: list[str] = []
        for n in m.findall("nuclide"):
            nuc_name = n.attrib["name"]
            ao = float(n.attrib.get("ao", "0"))
            parsed = parse_nuclide_name(nuc_name)
            if parsed is None:
                stats.skipped.append((full_case_id, f"unknown nuclide name: {nuc_name}"))
                return None
            zaid, label = parsed
            nuclides_json.append({
                "zaid":         zaid,
                "label":        label,
                "atom_density": ao,
            })
        for s in m.findall("sab"):
            sab_files.append(s.attrib["name"] + ".h5")
        materials.append({
            "name":     name,
            "temperature": 293.6,
            "nuclides": nuclides_json,
            "thermal_files": sab_files,
        })
        mat_id_to_idx[mid] = len(materials) - 1

    # Cells
    cells: list[dict] = []
    cell_id_to_idx: dict[int, int] = {}
    skip_universes = False
    universe_cells: dict[int, list[int]] = {}  # universe_id -> [cell idx]
    for c in geom_root.findall("cell"):
        cid = int(c.attrib["id"])
        region_expr = c.attrib.get("region", "")
        region = parse_region(region_expr, sid_to_idx) if region_expr else None
        # OpenMC cells without `region` are universe definitions; not
        # supported by this converter at this time.
        if region is None and region_expr:
            stats.skipped.append((full_case_id, f"unparseable region for cell {cid}: {region_expr!r}"))
            return None
        if region is None:
            stats.skipped.append((full_case_id, f"cell {cid} has no region (universe definition?)"))
            return None
        fill: dict
        if "material" in c.attrib:
            mat_attr = c.attrib["material"]
            if mat_attr in ("void", "0"):
                fill = {"type": "Void"}
            else:
                mid = int(mat_attr)
                if mid not in mat_id_to_idx:
                    stats.skipped.append((full_case_id, f"cell {cid} references unknown material {mid}"))
                    return None
                fill = {"type": "Material", "material_idx": mat_id_to_idx[mid]}
        elif "fill" in c.attrib:
            # OpenMC's `fill="<id>"` can refer to either a lattice or a
            # universe. Disambiguate against the lattice id map.
            fill_id = int(c.attrib["fill"])
            if fill_id in lattice_id_to_idx:
                fill = {"type": "Lattice", "lattice_idx": lattice_id_to_idx[fill_id]}
            else:
                fill = {"type": "Universe", "universe_id": fill_id}
        else:
            fill = {"type": "Void"}

        cells.append({
            "id":     cid,
            "region": region,
            "fill":   fill,
            "temperature": 293.6,
        })
        cell_id_to_idx[cid] = len(cells) - 1
        u_attr = c.attrib.get("universe", "0")
        try:
            u_id = int(u_attr)
        except ValueError:
            u_id = 0
        universe_cells.setdefault(u_id, []).append(len(cells) - 1)

    # Universes — must have at least one. OpenMC convention: universe 0
    # is the root unless explicitly overridden in settings.xml.
    #
    # The engine's `Geometry::new` validator treats UniverseId.0 as a
    # direct array index — `&self.universes[id.0 as usize]` — so the
    # IDs we emit must be dense `[0, n_universes)` after remapping.
    # ICSBEP lattice cases (LCT-008 etc.) use sparse IDs (1, 2, 99,
    # 111, 999, ...) which would trip the LatticeUniverseOutOfRange
    # check. Build a remap and rewrite every reference.
    if not universe_cells:
        stats.skipped.append((full_case_id, "no universes produced"))
        return None
    # Collect EVERY universe id referenced anywhere, not just universes
    # that have direct cells. Lattices may reference universes whose
    # only definition lives elsewhere in the geometry (e.g., a
    # "water-only" universe with cell `fill=<other_universe>` rather
    # than direct material cells). Missing such universes here would
    # break round-trip validation.
    all_uids: set[int] = set(universe_cells.keys())
    for lat in rect_lattices:
        all_uids.update(lat["universes"])
    for c in cells:
        if c["fill"].get("type") == "Universe":
            all_uids.add(c["fill"]["universe_id"])
    # Stable remap: universe 0 goes to dense idx 0 if present (OpenMC
    # root convention), otherwise the lowest sparse id becomes 0.
    sorted_uids = sorted(all_uids, key=lambda u: (u != 0, u))
    remap = {orig: dense for dense, orig in enumerate(sorted_uids)}
    # Apply remap.
    universes = [
        {"id": remap[uid], "cell_indices": sorted(universe_cells.get(uid, []))}
        for uid in sorted_uids
    ]
    for lat in rect_lattices:
        lat["universes"] = [remap[u] for u in lat["universes"]]
    for c in cells:
        if c["fill"].get("type") == "Universe":
            c["fill"]["universe_id"] = remap[c["fill"]["universe_id"]]
    root_orig = 0 if 0 in all_uids else min(all_uids)
    root_uid = remap[root_orig]

    case_json = {
        "benchmark": {
            "suite":           "ICSBEP",
            "case_id":         full_case_id,
            "case_name":       "",
            "category":        category,
            "k_eff_reference": k_ref,
            "k_eff_sigma":     sigma_exp,
            "source":          f"ICSBEP Handbook, {full_case_id} — NEA/NSC/DOC(95)03",
            "data_provenance": {
                "geometry_materials": {
                    "repo":         "github.com/mit-crpg/benchmarks",
                    "path":         mit_crpg_relpath,
                    "license":      "MIT",
                    "note":         "Open-source proxy. Hand-transcribed from the registered NEA/OECD ICSBEP Handbook by the OpenMC developers (MIT-CRPG). Re-verify against the registered handbook before publishing any validation report.",
                },
                "k_eff_reference": {
                    "repo":         "github.com/openmc-dev/validation",
                    "path":         uncertainties_relpath,
                    "row":          sub_label or "(top-level)",
                    "license":      "MIT",
                },
                "canonical_source": "NEA/OECD ICSBEP Handbook — registration required at https://www.oecd-nea.org/science/wpncs/icsbep/order.html. Open-source proxies are industry-standard derivatives but not the canonical specifications."
            },
        },
        "scene": {
            "surfaces":         surfaces,
            "cells":            cells,
            "universes":        universes,
            "rect_lattices":    rect_lattices,
            "materials":        materials,
            "root_universe_id": root_uid,
        },
    }
    return case_json


def load_uncertainties(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """`(case_slug, sub_case)` → `(k_ref, sigma_exp)`."""
    table: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.reader(f):
            if not row or len(row) < 4:
                continue
            case = row[0].strip()
            sub = row[1].strip()
            try:
                k_ref = float(row[2])
                sigma = float(row[3])
            except ValueError:
                continue
            table[(case, sub)] = (k_ref, sigma)
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mit-crpg", required=True, type=Path,
                   help="Path to mit-crpg-benchmarks/icsbep")
    p.add_argument("--openmc-validation", required=True, type=Path,
                   help="Path to openmc-validation repo root")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory for bench/icsbep/*.json")
    p.add_argument("--cases", default="",
                   help="Comma-separated list of case slugs to convert "
                        "(default: all). Example: pu-met-fast-001,heu-met-fast-001")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    uncert_path = args.openmc_validation / "benchmarking" / "uncertainties.csv"
    if not uncert_path.exists():
        print(f"error: {uncert_path} not found", file=sys.stderr)
        return 2

    uncertainties = load_uncertainties(uncert_path)
    print(f"loaded {len(uncertainties)} (case, sub) rows from {uncert_path}")

    filter_set = set(s.strip() for s in args.cases.split(",")) if args.cases else None

    stats = ParseStats()
    for case_dir in sorted(args.mit_crpg.iterdir()):
        if not case_dir.is_dir():
            continue
        slug = case_dir.name
        if filter_set and slug not in filter_set:
            continue
        prefix = "-".join(slug.split("-")[:3])
        category = CATEGORY_FROM_PREFIX.get(prefix, slug.upper())
        canonical = canonical_case_id(slug)
        openmc_dir = case_dir / "openmc"
        if not openmc_dir.is_dir():
            continue
        stats.n_cases += 1

        # Sub-cases: subdirectories like case-1, case-2, b-1, ...
        subdirs = sorted(
            d for d in openmc_dir.iterdir()
            if d.is_dir() and (d / "geometry.xml").exists()
        )
        if subdirs:
            for sub in subdirs:
                sub_label = sub.name
                k_data = uncertainties.get((slug, sub_label)) or uncertainties.get((slug, ""))
                if k_data is None:
                    stats.skipped.append((f"{canonical}.{sub_label}", "no row in uncertainties.csv"))
                    continue
                full_id = f"{canonical}.{sub_label}"
                case_json = parse_case(
                    case_dir=sub,
                    sub_label=sub_label,
                    full_case_id=full_id,
                    category=category,
                    k_ref=k_data[0],
                    sigma_exp=k_data[1],
                    mit_crpg_relpath=f"icsbep/{slug}/openmc/{sub_label}",
                    uncertainties_relpath="benchmarking/uncertainties.csv",
                    stats=stats,
                )
                if case_json is not None:
                    out_path = args.output / f"{slug}_{sub_label}.json"
                    out_path.write_text(json.dumps(case_json, indent=2))
                    stats.n_emitted += 1
        else:
            # Single top-level case (no sub-cases)
            k_data = uncertainties.get((slug, ""))
            if k_data is None:
                stats.skipped.append((canonical, "no row in uncertainties.csv"))
                continue
            case_json = parse_case(
                case_dir=openmc_dir,
                sub_label="",
                full_case_id=canonical,
                category=category,
                k_ref=k_data[0],
                sigma_exp=k_data[1],
                mit_crpg_relpath=f"icsbep/{slug}/openmc",
                uncertainties_relpath="benchmarking/uncertainties.csv",
                stats=stats,
            )
            if case_json is not None:
                out_path = args.output / f"{slug}.json"
                out_path.write_text(json.dumps(case_json, indent=2))
                stats.n_emitted += 1

    n_skipped = len(stats.skipped)
    print(f"\nProcessed: {stats.n_cases} case directories")
    print(f"  Emitted: {stats.n_emitted} JSON files into {args.output}")
    print(f"  Skipped: {n_skipped}")
    if stats.skipped:
        # Group skip reasons for readability
        reasons: dict[str, int] = {}
        for _, reason in stats.skipped:
            # Trim the reason for grouping
            key = reason.split(":")[0] if ":" in reason else reason
            reasons[key] = reasons.get(key, 0) + 1
        print("\nSkipped cases by reason:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:4}  {reason}")
        if n_skipped <= 30:
            print("\nDetails:")
            for case, reason in stats.skipped:
                print(f"  {case}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
