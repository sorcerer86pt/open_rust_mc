# SPDX-License-Identifier: MIT
"""Migrate natural-element ZAIDs in ICSBEP scene JSONs to isotopic entries.

Motivation
----------
ENDF/B-VIII.0 (Feb 2018) and ENDF/B-VIII.1 (Oct 2024) dropped the
natural-carbon evaluation (`C0.h5`) that ENDF/B-VII.1 shipped. The
isotopic evaluations `C12.h5` and `C13.h5` ship instead. Scene JSONs
that were generated against VII.1's cross_sections.xml carry literal
``"zaid": 6000`` (natural carbon) entries — they fail to load under
VIII.x.

This script walks every JSON in ``bench/icsbep/`` and rewrites each
``zaid % 1000 == 0`` nuclide entry into one entry per stable isotope,
scaling ``atom_density`` by the IUPAC 2021 natural abundance.

The migration is idempotent: once a JSON is rewritten, the
"zaid: 6000" entries are gone, so a subsequent run is a no-op. A
``"_split_from_zaid": 6000`` audit key is added to every new entry
so the provenance survives.

Currently only carbon needs migration (it's the only natural-Z that
appears in our corpus, ``grep`` confirms one ZAID). The script is
written generically so adding new abundances (e.g. if a future
library drops more naturals) is a one-line addition to
``NATURAL_ABUNDANCES``.

Usage
-----
    python scripts/migrate_natural_elements.py
        # rewrites bench/icsbep/*.json in place; prints a summary

    python scripts/migrate_natural_elements.py --dry-run
        # shows what would change, doesn't write anything

    python scripts/migrate_natural_elements.py --bench-dir <path>
        # override the default bench/icsbep location
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# IUPAC 2021 natural-isotopic-abundance table. Add elements here when a
# future library drops their natural evaluation; the migration script
# stays generic. Values are mole fractions and sum to 1.0 per element.
NATURAL_ABUNDANCES: dict[int, dict[int, float]] = {
    # Carbon — VIII.0+ ships only C12 + C13.
    # https://iupac.qmul.ac.uk/AtWt/  (C: 12 0.9893, 13 0.0107)
    6: {12: 0.9893, 13: 0.0107},
}

# Element symbols used to construct the rewritten ``label`` field
# (matches OpenMC's ``<Symbol><Mass>`` convention; e.g. ``C-12``).
ELEMENT_SYMBOL: dict[int, str] = {6: "C"}


def split_natural_zaid(zaid_natural: int, atom_density: float) -> list[dict]:
    """Return per-isotope entries summing to the original atom density."""
    z = zaid_natural // 1000
    table = NATURAL_ABUNDANCES.get(z)
    if table is None:
        raise KeyError(
            f"no natural abundance table for Z={z} (ZAID {zaid_natural}); "
            "add an entry to NATURAL_ABUNDANCES in this script"
        )
    symbol = ELEMENT_SYMBOL[z]
    out = []
    for a, fraction in table.items():
        out.append(
            {
                "zaid": z * 1000 + a,
                "label": f"{symbol}-{a}",
                "atom_density": atom_density * fraction,
                # Provenance trace so a future maintainer can see this
                # entry came from an automated migration, not a manual
                # edit. Survives idempotent re-runs.
                "_split_from_zaid": zaid_natural,
                "_split_abundance": fraction,
            }
        )
    return out


def migrate_material(material: dict) -> tuple[dict, int]:
    """Return ``(new_material, n_entries_split)``."""
    new_nuclides: list[dict] = []
    n_split = 0
    for entry in material.get("nuclides", []):
        zaid = entry.get("zaid")
        if isinstance(zaid, int) and zaid > 0 and zaid % 1000 == 0 and (zaid // 1000) in NATURAL_ABUNDANCES:
            new_nuclides.extend(split_natural_zaid(zaid, float(entry["atom_density"])))
            n_split += 1
        else:
            new_nuclides.append(entry)
    new_mat = copy.deepcopy(material)
    new_mat["nuclides"] = new_nuclides
    return new_mat, n_split


def migrate_scene_doc(doc: dict) -> tuple[dict, int]:
    """Return ``(new_doc, total_entries_split)``."""
    new = copy.deepcopy(doc)
    scene = new.get("scene", new)  # benches usually have a top-level "scene"
    materials = scene.get("materials", [])
    total = 0
    for i, mat in enumerate(materials):
        migrated, n_here = migrate_material(mat)
        if n_here:
            materials[i] = migrated
            total += n_here
    return new, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="Directory containing ICSBEP JSON files (default: <repo>/bench/icsbep)",
    )
    p.add_argument("--dry-run", action="store_true", help="don't write files; just print what would change")
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help="substring filter on case filenames (e.g. 'heu-comp-inter-003' "
        "to scope the migration to a single family)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.bench_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        bench_dir = repo_root / "bench" / "icsbep"
    else:
        bench_dir = args.bench_dir
    if not bench_dir.is_dir():
        print(f"bench dir not found: {bench_dir}", file=sys.stderr)
        return 2

    total_files_touched = 0
    total_entries_split = 0
    cases = sorted(bench_dir.glob("*.json"))
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]
    for case in cases:
        try:
            doc = json.loads(case.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[skip] {case.name}: invalid JSON ({e})", file=sys.stderr)
            continue
        new_doc, n_split = migrate_scene_doc(doc)
        if n_split == 0:
            continue
        total_files_touched += 1
        total_entries_split += n_split
        action = "DRY-RUN" if args.dry_run else "wrote "
        print(f"  {action} {case.name}: split {n_split} natural-Z entries")
        if not args.dry_run:
            # Match the prior file style (2-space indent + trailing newline).
            case.write_text(json.dumps(new_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    if total_files_touched == 0:
        print(f"No natural-element ZAIDs found in {len(cases)} case(s). Nothing to migrate.")
    else:
        verb = "would update" if args.dry_run else "updated"
        print(
            f"{verb} {total_files_touched} file(s); split {total_entries_split} "
            f"natural-Z entries across {len(cases)} case(s) scanned."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
