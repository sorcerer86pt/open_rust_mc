"""
Per-nuclide cross-section reference dump from OpenMC's Python API.

For each of the nine PWR pin-cell nuclides, at the target temperature
used by open_rust_mc's pwr_pincell binary, evaluates:
  total (MT=1), elastic (MT=2), inelastic (MT=4 if present),
  (n,2n) (MT=16), (n,3n) (MT=17), fission (MT=18), capture (MT=102),
  summed leaf-absorption (MT=27), charged-particle channels where
  relevant (MT=103 (n,p), MT=107 (n,α)), and nu-bar.

Output: CSV with one row per (nuclide, energy, channel).
Intended to be diffed against the equivalent Rust dump to attribute
the ~120 pcm shared offset from OpenMC to a specific channel.

Usage (WSL + openmc conda env):
    python scripts/xs_dump_openmc.py
Writes outputs/xs_audit/openmc_ref.csv
"""
import os
import sys
import csv
import numpy as np
import openmc.data

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEUTRON_DIR = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron"
OUT_DIR = "/mnt/c/Users/fog/madman_svd_experiment/outputs/xs_audit"

# (filename, target_temp_K, short_name)
# Temperatures match NUCLIDE_SPECS in src/bin/pwr_pincell.rs.
# temp_idx 3 = 900 K (fuel), 2 = 600 K (clad + moderator).
NUCLIDES = [
    ("U235.h5", 900, "U235"),
    ("U238.h5", 900, "U238"),
    ("O16.h5",  900, "O16_fuel"),
    ("H1.h5",   600, "H1"),
    ("Zr90.h5", 600, "Zr90"),
    ("Zr91.h5", 600, "Zr91"),
    ("Zr92.h5", 600, "Zr92"),
    ("Zr94.h5", 600, "Zr94"),
    ("O16.h5",  600, "O16_mod"),
]

# 50 log-spaced energies from thermal to fast, plus specific resonance peaks.
ENERGIES = list(np.logspace(-2, 7, 50))  # 0.01 eV to 10 MeV
RESONANCES = [6.674, 20.9, 36.7, 66.0, 80.7, 102.5]  # U-238 RRR peaks
ENERGIES = sorted(set(ENERGIES + RESONANCES))


def temp_key(nuc, target_k):
    """Pick the library temperature closest to target_k."""
    keys = [k for k in nuc.energy if k.endswith("K")]
    if not keys:
        return None
    return min(keys, key=lambda k: abs(float(k.rstrip("K")) - target_k))


def eval_nuclide(path, target_k, short, writer):
    nuc = openmc.data.IncidentNeutron.from_hdf5(path)
    tkey = temp_key(nuc, target_k)
    if tkey is None:
        return
    lib_T = float(tkey.rstrip("K"))

    # Identify present reactions
    mts_present = list(nuc.reactions.keys())

    # Total XS: prefer MT=1 if exposed; else sum elastic + (MT=3 nonelastic)
    # openmc.data.IncidentNeutron usually has MT=1 via `total`, MT=2, MT=18, etc.
    # For comparability we fetch per-MT at the library temp.

    def get_xs(mt):
        rxn = nuc.reactions.get(mt)
        if rxn is None:
            return None
        if tkey in rxn.xs:
            return np.array(rxn.xs[tkey](ENERGIES))
        return None

    # OpenMC's HDF5 layout does not expose MT=1 directly (only ACE format
    # does). Compute total by summing all non-redundant reaction channels:
    # excludes MT=3, 4, 27, 101 (sum-MTs), KERMA MT=301, MT=444, MT=901, etc.
    # Includes MT=51..91 discrete levels, which together cover MT=4.
    total_xs = np.zeros(len(ENERGIES))
    for mt, rxn in nuc.reactions.items():
        if rxn.redundant:
            continue
        if tkey in rxn.xs:
            total_xs = total_xs + np.array(rxn.xs[tkey](ENERGIES))

    el  = get_xs(2)
    inel = get_xs(4)  # redundant in HDF5; reported here for diagnostics only
    n2n = get_xs(16)
    n3n = get_xs(17)
    fis = get_xs(18)
    cap = get_xs(102)
    np_abs = get_xs(103)
    na_abs = get_xs(107)
    # Fissionable first-chance variants (U-238 has these near threshold)
    f19 = get_xs(19)
    f20 = get_xs(20)
    f21 = get_xs(21)

    # nu-bar total = sum of ALL neutron products (prompt + delayed).
    # Earlier draft of this script fetched only the first neutron product,
    # which is prompt; that produced a spurious 0.035 delta for U-238
    # against the Rust engine, which correctly sums all neutron products.
    nu_bar_vals = np.full(len(ENERGIES), np.nan)
    if fis is not None and 18 in nuc.reactions:
        rxn18 = nuc.reactions[18]
        try:
            total = np.zeros(len(ENERGIES))
            any_found = False
            for prod in rxn18.products:
                if prod.particle != "neutron":
                    continue
                total = total + np.asarray(prod.yield_(ENERGIES))
                any_found = True
            if any_found:
                nu_bar_vals = total
        except (AttributeError, TypeError):
            pass

    for i, e in enumerate(ENERGIES):
        row = {
            "nuclide": short,
            "target_K": target_k,
            "lib_T_label": tkey,
            "lib_T_K": lib_T,
            "E_eV": e,
            "total":    float(total_xs[i]) if total_xs is not None else None,
            "elastic":  float(el[i])   if el   is not None else None,
            "inelast":  float(inel[i]) if inel is not None else None,
            "n2n":      float(n2n[i])  if n2n  is not None else None,
            "n3n":      float(n3n[i])  if n3n  is not None else None,
            "fission":  float(fis[i])  if fis  is not None else None,
            "capture":  float(cap[i])  if cap  is not None else None,
            "n_p":      float(np_abs[i]) if np_abs is not None else None,
            "n_alpha":  float(na_abs[i]) if na_abs is not None else None,
            "fis_19":   float(f19[i])  if f19  is not None else None,
            "fis_20":   float(f20[i])  if f20  is not None else None,
            "fis_21":   float(f21[i])  if f21  is not None else None,
            "nu_bar":   float(nu_bar_vals[i]),
            "mts":      ",".join(str(m) for m in sorted(mts_present)),
        }
        writer.writerow(row)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "openmc_ref.csv")
    fieldnames = [
        "nuclide", "target_K", "lib_T_label", "lib_T_K", "E_eV",
        "total", "elastic", "inelast", "n2n", "n3n", "fission",
        "capture", "n_p", "n_alpha", "fis_19", "fis_20", "fis_21",
        "nu_bar", "mts",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fn, tgt_k, short in NUCLIDES:
            path = os.path.join(NEUTRON_DIR, fn)
            print(f"  {short:<10}  T={tgt_k} K   {fn}")
            eval_nuclide(path, tgt_k, short, w)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
