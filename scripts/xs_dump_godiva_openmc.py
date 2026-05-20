# SPDX-License-Identifier: MIT
"""OpenMC reference dump for Godiva nuclides (U-234, U-235, U-238) at 294 K.
Mirrors the grid used by rust_prototype/src/bin/xs_dump_godiva.rs so that
the two CSVs can be diffed channel-by-channel to localise the fast-spectrum
offset.

Writes outputs/xs_audit/openmc_godiva_ref.csv
Usage (WSL + openmc conda env):
    python scripts/xs_dump_godiva_openmc.py
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

NUCLIDES = [
    ("U234.h5", 294, "U234"),
    ("U235.h5", 294, "U235"),
    ("U238.h5", 294, "U238"),
]

# Same 80-point grid as Rust xs_dump_godiva (log-spaced 0.1 eV to 100 MeV).
ENERGIES = np.array([10 ** (-1 + 9 * i / 79) for i in range(80)])


def nuclide_row(nd, e):
    """Return (total, elastic, inelastic_sum, n2n, n3n, fission, capture, nu_bar)."""
    # at 294 K
    T = "294K"

    def xs_at(mt):
        try:
            r = nd.reactions[mt]
        except KeyError:
            return 0.0
        xs = r.xs.get(T)
        if xs is None:
            return 0.0
        return float(xs(e))

    total    = xs_at(1)
    elastic  = xs_at(2)
    n2n      = xs_at(16)
    n3n      = xs_at(17)
    fission  = xs_at(18)
    capture  = xs_at(102)
    # Inelastic: sum 51..91 if present; if not, MT=4 total inelastic.
    inel = 0.0
    for mt in range(51, 92):
        inel += xs_at(mt)
    if inel == 0.0:
        inel = xs_at(4)

    # nu_bar: energy-dependent total (prompt + delayed)
    nu_bar = 0.0
    if fission > 0:
        try:
            # OpenMC IncidentNeutron.fission_energy isn't directly useful; use yield_product
            # sum of total neutron yield from fission products
            yields = []
            if 18 in nd.reactions:
                fis = nd.reactions[18]
                for p in fis.products:
                    if p.particle == "neutron":
                        # p.yield_ is Tabulated1D of neutron yield vs incident E
                        yields.append(float(p.yield_(e)))
                nu_bar = sum(yields)
        except Exception:
            nu_bar = 0.0

    return (total, elastic, inel, n2n, n3n, fission, capture, nu_bar)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "openmc_godiva_ref.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nuclide", "target_K", "E_eV", "total", "elastic", "inelastic",
                    "n2n", "n3n", "fission", "capture", "nu_bar"])
        for fname, T, short in NUCLIDES:
            path = os.path.join(NEUTRON_DIR, fname)
            nd = openmc.data.IncidentNeutron.from_hdf5(path)
            print(f"{short}: temperatures = {nd.temperatures}", flush=True)
            for e in ENERGIES:
                vals = nuclide_row(nd, e)
                w.writerow([short, T, f"{e:.6e}"] + [f"{v:.6e}" for v in vals])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
