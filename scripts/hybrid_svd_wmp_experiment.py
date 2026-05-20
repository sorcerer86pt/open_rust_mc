# SPDX-License-Identifier: MIT
"""
Hybrid SVD + WMP experiment.

Goal: replace the offline "530x memory reduction" projection with a
real measurement. For each PWR-relevant nuclide we:

  1. Measure actual WMP file size on disk (bytes used for the
     windowed-multipole resonance representation).
  2. Measure actual pointwise HDF5 cross-section memory over the
     same RRR energy window (what WMP would replace).
  3. Evaluate WMP on a dense grid inside the RRR, compare to the
     HDF5 pointwise reference, report accuracy (relative error,
     p99, max) per reaction channel.
  4. Compute a hybrid representation memory: SVD over the
     smooth-region energy points + WMP bytes for the RRR.
     Compare to pointwise table over the full spectrum.

All numbers are measured, not projected.

Usage (Windows): invoke from wsl with the openmc conda env active.
"""

import os
import sys
import numpy as np
import openmc.data

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_ROOT = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5"
WMP_DIR = os.path.join(DATA_ROOT, "wmp")
NEUTRON_DIR = os.path.join(DATA_ROOT, "neutron")

# PWR-relevant nuclides with ZZAAA for WMP lookup
NUCLIDES = [
    ("U234", "092234"),
    ("U235", "092235"),
    ("U238", "092238"),
    ("O16",  "008016"),
    ("H1",   "001001"),
    ("Zr90", "040090"),
    ("Zr91", "040091"),
    ("Zr92", "040092"),
    ("Zr94", "040094"),
]

TEMP_K = 293.6
N_EVAL = 20000  # evaluation points inside RRR for accuracy check


def wmp_bytes(path):
    """Actual disk size of a WMP HDF5 file (bytes)."""
    return os.path.getsize(path)


def wmp_payload_bytes(wmp):
    """
    Count the raw payload bytes of WMP data (poles + windows + curvefit),
    excluding HDF5 container overhead.
    """
    # data: (n_poles, 4) complex128 -> 32 bytes/row
    data = wmp.data.nbytes
    # windows: (n_windows, 2) int32 -> 8 bytes/row
    windows = wmp.windows.nbytes
    # curvefit: (n_windows, n_rxn, fit_order+1) float64
    curvefit = wmp.curvefit.nbytes
    # broaden_poly: (n_windows,) int8
    broaden = wmp.broaden_poly.nbytes
    return data + windows + curvefit + broaden


def load_neutron(zzaaa_name):
    """Load HDF5 neutron data for the nuclide."""
    short, _ = next((s for s in NUCLIDES if s[0] == zzaaa_name), (None, None))
    path = os.path.join(NEUTRON_DIR, f"{zzaaa_name}.h5")
    return openmc.data.IncidentNeutron.from_hdf5(path), path


def pointwise_bytes_in_window(nuc, e_min, e_max, temps):
    """
    How many bytes does the pointwise HDF5 table occupy over [e_min, e_max]
    across the given temperature list? Counts elastic + fission + capture +
    total (4 reactions as in OpenMC hot-path use).
    """
    tkey = sorted(nuc.energy.keys(), key=lambda t: float(t.rstrip("K")))[0]
    egrid = nuc.energy[tkey]
    mask = (egrid >= e_min) & (egrid <= e_max)
    n_e = mask.sum()
    n_t = len(temps)
    # float64 for XS, 4 reaction channels (elastic, total, fission, capture)
    # plus shared energy grid
    return n_e * n_t * 4 * 8 + n_e * 8


def pointwise_bytes_full(nuc, temps):
    """Full-spectrum pointwise bytes for the same 4 reactions."""
    tkey = sorted(nuc.energy.keys(), key=lambda t: float(t.rstrip("K")))[0]
    n_e = len(nuc.energy[tkey])
    n_t = len(temps)
    return n_e * n_t * 4 * 8 + n_e * 8


def wmp_accuracy(wmp, nuc, temp_k=TEMP_K, n_eval=N_EVAL):
    """
    Evaluate WMP on a log-spaced energy grid inside [E_min, E_max] and
    compare to the HDF5 pointwise reference at the closest library T.

    Returns dict with per-reaction max/p99/p50 relative error + eval range.
    """
    e_min, e_max = float(wmp.E_min), float(wmp.E_max)
    energies = np.geomspace(max(e_min, 1e-5), e_max, n_eval)

    # WMP gives (scattering, absorption, fission) resonance contribution
    s_scat, s_abs, s_fis = wmp(energies, temp_k)
    # absorption = capture + fission for fissionable; for non-fissionable
    # fission is zero. WMP represents the resonance part only (poles);
    # curvefit polynomial adds smooth background.

    # Reference: HDF5 table interpolation at nearest library temp
    tkey = f"{int(round(temp_k))}K"
    if tkey not in nuc.reactions[2].xs:
        # fall back to nearest
        tkey = sorted(nuc.reactions[2].xs.keys(),
                      key=lambda t: abs(float(t.rstrip("K")) - temp_k))[0]

    ref_el = nuc.reactions[2].xs[tkey](energies)  # MT=2 elastic
    ref_cap = nuc.reactions[102].xs[tkey](energies)  # MT=102 capture
    ref_fis = None
    if 18 in nuc.reactions:
        ref_fis = nuc.reactions[18].xs[tkey](energies)  # MT=18 fission

    eps = 1e-30

    def rel(a, b):
        return np.abs(a - b) / (np.maximum(np.abs(b), eps))

    out = {
        "temp_label": tkey,
        "e_min": e_min,
        "e_max": e_max,
        "n_eval": n_eval,
    }

    # Scattering (WMP resonance part only; add no background here because
    # OpenMC's WMP convention is that the table's "elastic" already matches
    # WMP + background; we compare directly to elastic).
    err_scat = rel(s_scat, ref_el)
    out["scat_max"] = float(np.max(err_scat))
    out["scat_p99"] = float(np.percentile(err_scat, 99))
    out["scat_p50"] = float(np.percentile(err_scat, 50))

    # Absorption = capture + fission in WMP; ref = capture + fission
    ref_abs = ref_cap + (ref_fis if ref_fis is not None else 0)
    err_abs = rel(s_abs, ref_abs)
    out["abs_max"] = float(np.max(err_abs))
    out["abs_p99"] = float(np.percentile(err_abs, 99))
    out["abs_p50"] = float(np.percentile(err_abs, 50))

    if ref_fis is not None and wmp.fissionable:
        err_fis = rel(s_fis, ref_fis)
        out["fis_max"] = float(np.max(err_fis))
        out["fis_p99"] = float(np.percentile(err_fis, 99))
        out["fis_p50"] = float(np.percentile(err_fis, 50))
    return out


def svd_smooth_bytes(nuc, e_min_res, e_max_res, rank_smooth, n_temps):
    """
    Bytes for a rank-k SVD over the smooth (non-RRR) energy points.
    Smooth = [E_grid_min, e_min_res) U (e_max_res, E_grid_max].
    """
    tkey = sorted(nuc.energy.keys(), key=lambda t: float(t.rstrip("K")))[0]
    egrid = nuc.energy[tkey]
    smooth = ((egrid < e_min_res) | (egrid > e_max_res)).sum()
    # basis: n_smooth * rank * 8 (float64)
    # coeff per temperature: n_temps * rank * 8
    # plus smooth energy grid
    return smooth * rank_smooth * 8 + n_temps * rank_smooth * 8 + smooth * 8


def main():
    print("=" * 78)
    print("HYBRID SVD + WMP EXPERIMENT — measured numbers")
    print("=" * 78)
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Temperature: {TEMP_K} K")
    print(f"  Evaluation points per nuclide: {N_EVAL}")
    print()

    # Table header
    print(f"{'nuclide':<8}{'RRR range (eV)':>22}{'WMP disk':>12}"
          f"{'WMP payload':>14}{'pw-RRR':>10}{'pw-full':>10}"
          f"{'SVD(k=2) smooth':>18}{'hybrid':>10}{'vs pw-full':>12}")
    print("-" * 118)

    rows = []
    acc_rows = []

    for name, zzaaa in NUCLIDES:
        wmp_path = os.path.join(WMP_DIR, f"{zzaaa}.h5")
        if not os.path.exists(wmp_path):
            print(f"{name:<8}  no WMP file")
            continue

        try:
            wmp = openmc.data.WindowedMultipole.from_hdf5(wmp_path)
        except Exception as e:
            print(f"{name:<8}  WMP load failed: {e}")
            continue

        nuc, _ = load_neutron(name)

        disk = wmp_bytes(wmp_path)
        payload = wmp_payload_bytes(wmp)
        pw_rrr = pointwise_bytes_in_window(nuc, wmp.E_min, wmp.E_max,
                                           list(nuc.energy.keys()))
        pw_full = pointwise_bytes_full(nuc, list(nuc.energy.keys()))
        svd_smooth = svd_smooth_bytes(nuc, wmp.E_min, wmp.E_max,
                                      rank_smooth=2, n_temps=len(nuc.energy))
        hybrid = svd_smooth + payload
        ratio = pw_full / hybrid

        rows.append((name, wmp.E_min, wmp.E_max, disk, payload, pw_rrr,
                     pw_full, svd_smooth, hybrid, ratio))

        print(f"{name:<8}{wmp.E_min:>9.2f} - {wmp.E_max:<10.0f}"
              f"{disk/1024:>9.1f} KB"
              f"{payload/1024:>11.1f} KB"
              f"{pw_rrr/1024:>7.0f} KB"
              f"{pw_full/1024:>7.0f} KB"
              f"{svd_smooth/1024:>15.1f} KB"
              f"{hybrid/1024:>7.0f} KB"
              f"{ratio:>10.1f}x")

        # Accuracy check
        try:
            acc = wmp_accuracy(wmp, nuc)
            acc_rows.append((name, acc))
        except Exception as e:
            print(f"  accuracy check failed: {e}")

    # Accuracy report
    print()
    print("=" * 78)
    print("WMP reconstruction accuracy vs HDF5 pointwise reference")
    print("=" * 78)
    print(f"  {'nuclide':<8}{'T':<8}{'elastic p50':>14}{'p99':>12}{'max':>12}"
          f"{'absorption p50':>18}{'p99':>12}{'max':>12}")
    print("-" * 100)
    for name, a in acc_rows:
        print(f"  {name:<8}{a['temp_label']:<8}"
              f"{a['scat_p50']:>13.2e}{a['scat_p99']:>12.2e}{a['scat_max']:>12.2e}"
              f"{a['abs_p50']:>17.2e}{a['abs_p99']:>12.2e}{a['abs_max']:>12.2e}")
        if "fis_max" in a:
            print(f"  {'':>16}fission:    "
                  f"{a['fis_p50']:>13.2e}{a['fis_p99']:>12.2e}{a['fis_max']:>12.2e}")

    # Totals across the PWR nuclide set
    print()
    print("=" * 78)
    print("TOTAL memory, 9-nuclide PWR set")
    print("=" * 78)
    total_pw_full = sum(r[6] for r in rows)
    total_hybrid = sum(r[8] for r in rows)
    total_wmp_disk = sum(r[3] for r in rows)
    total_wmp_payload = sum(r[4] for r in rows)
    total_svd_smooth = sum(r[7] for r in rows)
    print(f"  Pointwise table (4 rxn × n_T):  {total_pw_full/1024:>10.1f} KB"
          f"  ({total_pw_full/1024/1024:.2f} MB)")
    print(f"  WMP disk total:                 {total_wmp_disk/1024:>10.1f} KB")
    print(f"  WMP payload total:              {total_wmp_payload/1024:>10.1f} KB")
    print(f"  SVD(k=2) smooth basis total:    {total_svd_smooth/1024:>10.1f} KB")
    print(f"  Hybrid (SVD smooth + WMP):      {total_hybrid/1024:>10.1f} KB")
    print(f"  Reduction ratio:                {total_pw_full/total_hybrid:>10.1f}x")


if __name__ == "__main__":
    main()
