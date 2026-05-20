# SPDX-License-Identifier: MIT
"""
Diff the OpenMC and Rust-engine XS dumps to find where the shared
~120 pcm engine offset comes from.

For each (nuclide, energy) in the OpenMC reference that has a
matching Rust entry, compute:
  - dtotal      = rust.total   - openmc.total
  - delastic    = rust.elastic - openmc.elastic
  - dcapture    = rust.capture - (openmc.capture + openmc.np + openmc.nα
                                 + openmc.fis_19 + openmc.fis_20 + openmc.fis_21)
                  i.e. the Rust capture residue vs the sum of
                  OpenMC capture + charged-particle + first-chance fission
                  channels that Rust absorbs into capture.
  - dfission    = rust.fission - openmc.fission
  - dnu_bar     = rust.nu_bar  - openmc.nu_bar

Reports per-nuclide max |d|, and highlights energies where the
Rust total diverges from the OpenMC total by > 1%.

Inputs:
  outputs/xs_audit/openmc_ref.csv     (from xs_dump_openmc.py)
  outputs/xs_audit/rust_<mode>.csv    (from xs_dump binary)

Usage:
  python scripts/xs_audit_diff.py [svd|table|hybrid]
Defaults to svd.
"""

import os
import sys
import csv
import math

AUDIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "xs_audit",
)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(s):
    if s is None or s == "" or s == "None":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def key(row):
    # Match by (nuclide, target_K, rounded E)
    e = float(row["E_eV"])
    # log-scale bucketing at 6 significant digits
    if e <= 0:
        bucket = 0
    else:
        bucket = round(math.log10(e), 4)
    return (row["nuclide"], int(row["target_K"]), bucket)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "svd"
    openmc = read_csv(os.path.join(AUDIT_DIR, "openmc_ref.csv"))
    rust = read_csv(os.path.join(AUDIT_DIR, f"rust_{mode}.csv"))

    rust_idx = {key(r): r for r in rust}

    # Per-nuclide tracking
    per_nuc = {}  # name -> list of (e, dtot, del, dcap_effective, dfis, dnu)
    for r in openmc:
        k = key(r)
        if k not in rust_idx:
            continue
        rr = rust_idx[k]
        e = float(r["E_eV"])
        om_tot = to_float(r["total"])
        om_el  = to_float(r["elastic"])
        om_cap = to_float(r["capture"])
        om_np  = to_float(r["n_p"])
        om_na  = to_float(r["n_alpha"])
        om_f19 = to_float(r["fis_19"])
        om_f20 = to_float(r["fis_20"])
        om_f21 = to_float(r["fis_21"])
        om_fis = to_float(r["fission"])
        om_nu  = to_float(r["nu_bar"])

        ru_tot = to_float(rr["total"])
        ru_el  = to_float(rr["elastic"])
        ru_cap = to_float(rr["capture"])
        ru_fis = to_float(rr["fission"])
        ru_nu  = to_float(rr["nu_bar"])

        # Rust absorbs MT=103,107,19,20,21 into capture
        extra = 0.0
        for v in (om_np, om_na, om_f19, om_f20, om_f21):
            if not math.isnan(v):
                extra += v
        om_cap_effective = (om_cap if not math.isnan(om_cap) else 0.0) + extra

        def diff(r, o):
            if math.isnan(r) or math.isnan(o):
                return float("nan")
            return r - o

        name = r["nuclide"]
        per_nuc.setdefault(name, []).append({
            "E": e,
            "om_tot": om_tot, "ru_tot": ru_tot, "d_tot": diff(ru_tot, om_tot),
            "rel_tot": (diff(ru_tot, om_tot) / om_tot) if om_tot > 0 else float("nan"),
            "d_el": diff(ru_el, om_el),
            "d_cap_eff": diff(ru_cap, om_cap_effective),
            "d_fis": diff(ru_fis, om_fis),
            "d_nu":  diff(ru_nu, om_nu),
        })

    # Report per-nuclide max rel |dtotal| and which E it occurs at
    print(f"=== XS audit ({mode}) — max relative |dtotal| per nuclide ===")
    print(f"{'nuclide':<10} {'E_peak (eV)':>14} {'rel dtot':>12} "
          f"{'abs dtot (b)':>14} {'rust_tot':>12} {'omc_tot':>12}")
    worst = []
    for name, rows in per_nuc.items():
        # Exclude NaNs
        clean = [r for r in rows if not math.isnan(r["rel_tot"])]
        if not clean:
            continue
        r_max = max(clean, key=lambda r: abs(r["rel_tot"]))
        worst.append((name, r_max))
        print(f"{name:<10} {r_max['E']:>14.4e} {r_max['rel_tot']:>12.3%} "
              f"{r_max['d_tot']:>14.3e} {r_max['ru_tot']:>12.4e} {r_max['om_tot']:>12.4e}")

    # Report energies where rel|dtot|>1% across all nuclides
    print()
    print("=== Points where |dtotal/total| > 1% ===")
    print(f"{'nuclide':<10} {'E (eV)':>12} {'rel dtot':>10} "
          f"{'rel del':>10} {'dcap_eff':>12} {'dfis':>12} {'dnu':>10}")
    for name, rows in per_nuc.items():
        for r in rows:
            if math.isnan(r["rel_tot"]) or abs(r["rel_tot"]) <= 0.01:
                continue
            rel_el = (r["d_el"] / r["om_tot"]) if r["om_tot"] > 0 else float("nan")
            print(f"{name:<10} {r['E']:>12.4e} {r['rel_tot']:>10.2%} "
                  f"{rel_el:>10.2%} {r['d_cap_eff']:>12.3e} "
                  f"{r['d_fis']:>12.3e} {r['d_nu']:>10.4f}")

    # Summary: median relative |dtotal| in thermal (E<10 eV), RRR (10..25k),
    # fast (>25k)
    print()
    print("=== Median |rel dtot| by spectrum region, per nuclide ===")
    print(f"{'nuclide':<10} {'thermal':>10} {'RRR':>10} {'fast':>10}")
    for name, rows in per_nuc.items():
        buckets = {"thermal": [], "RRR": [], "fast": []}
        for r in rows:
            if math.isnan(r["rel_tot"]):
                continue
            e = r["E"]
            key_b = "thermal" if e < 10 else ("RRR" if e < 25_000 else "fast")
            buckets[key_b].append(abs(r["rel_tot"]))
        med = {k: (sorted(v)[len(v)//2] if v else float("nan")) for k, v in buckets.items()}
        print(f"{name:<10} {med['thermal']:>10.2%} {med['RRR']:>10.2%} {med['fast']:>10.2%}")


if __name__ == "__main__":
    main()
