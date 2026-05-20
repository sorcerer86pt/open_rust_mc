# SPDX-License-Identifier: MIT
"""OpenMC cross-validation of water exposure buildup factor at 1 MeV.

Runs the same benchmark as tests/ansi_ans_buildup.rs using OpenMC.
If OpenMC gives similar ~20 % discrepancy vs Harima 1991 at deep
optical depths, the deep-depth slack in the Rust test is a
cross-code / literature uncertainty rather than a kernel bug.

Setup:
    - 1 MeV point isotropic photon source at origin
    - Water medium, vacuum boundary far beyond the deepest shell
    - Surface-crossing scalar flux tallies at mu_0 r = 1, 2, 4, 7, 10
    - Energy-weighted by E · mu_en(E)/rho (exposure convention)
    - 500 k histories

Usage (WSL):
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate openmc
    python scripts/openmc_water_buildup.py
"""
import json
import os

import numpy as np
import openmc

DATA = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5"
OUT = "/mnt/c/Users/fog/madman_svd_experiment/outputs/openmc_water_buildup.json"
WORK = "/tmp/openmc_water_buildup"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

SOURCE_E = 1.0e6  # eV
N_HIST = 500_000
BATCHES = 10

# Harima 1991 GP-fit reference for water, 1 MeV, point isotropic.
OPTICAL_DEPTHS = [1.0, 2.0, 4.0, 7.0, 10.0]
REFERENCE_BE = [2.09, 3.33, 6.58, 12.89, 20.31]

# NIST XCOM mu_en/rho table for water (cm^2/g), Hubbell & Seltzer 1995.
MU_EN_RHO = np.array([
    (1.0e3,   4.065e3),
    (1.5e3,   1.372e3),
    (2.0e3,   6.152e2),
    (3.0e3,   1.917e2),
    (4.0e3,   8.191e1),
    (5.0e3,   4.188e1),
    (6.0e3,   2.405e1),
    (8.0e3,   9.915e0),
    (1.0e4,   4.944e0),
    (1.5e4,   1.374e0),
    (2.0e4,   5.503e-1),
    (3.0e4,   1.557e-1),
    (4.0e4,   6.947e-2),
    (5.0e4,   4.223e-2),
    (6.0e4,   3.190e-2),
    (8.0e4,   2.597e-2),
    (1.0e5,   2.550e-2),
    (1.5e5,   2.764e-2),
    (2.0e5,   2.966e-2),
    (3.0e5,   3.192e-2),
    (4.0e5,   3.279e-2),
    (5.0e5,   3.299e-2),
    (6.0e5,   3.284e-2),
    (8.0e5,   3.206e-2),
    (1.0e6,   3.103e-2),
    (1.25e6,  2.965e-2),
    (1.5e6,   2.833e-2),
    (2.0e6,   2.608e-2),
    (3.0e6,   2.276e-2),
    (4.0e6,   2.075e-2),
    (5.0e6,   1.941e-2),
    (6.0e6,   1.846e-2),
    (8.0e6,   1.723e-2),
    (1.0e7,   1.647e-2),
])


def mu_en_rho_water(e_ev):
    e_ev_arr = np.atleast_1d(e_ev).astype(float)
    grid = MU_EN_RHO[:, 0]
    vals = MU_EN_RHO[:, 1]
    e_ev_arr = np.clip(e_ev_arr, grid[0], grid[-1])
    log_grid = np.log(grid)
    log_vals = np.log(vals)
    return np.exp(np.interp(np.log(e_ev_arr), log_grid, log_vals))


# === Material =============================================================
water = openmc.Material(name="water")
water.add_nuclide("H1", 2.0)
water.add_nuclide("O16", 1.0)
water.set_density("g/cm3", 1.0)
mats = openmc.Materials([water])
mats.cross_sections = f"{DATA}/cross_sections.xml"
mats.export_to_xml()

# === Geometry: thin shell cells at each tally radius ==================
# Water mu at 1 MeV = 0.07072 cm^-1 (NIST XCOM); mfp = 14.14 cm.
mu_0 = 0.07072
mfp = 1.0 / mu_0
print(f"mu_0 = {mu_0} cm^-1, mfp = {mfp:.3f} cm")

SHELL_HT_FRAC = 0.005
radii = [x * mfp for x in OPTICAL_DEPTHS]
half_ts = [r * SHELL_HT_FRAC for r in radii]

# Assemble all sphere radii (inner/outer of each shell + outer vacuum).
inner_outer = [(r - h, r + h) for r, h in zip(radii, half_ts)]
outer_r = radii[-1] * 1.5
sorted_radii = sorted({r for pair in inner_outer for r in pair} | {outer_r})
spheres = [openmc.Sphere(r=r) for r in sorted_radii]
spheres[-1].boundary_type = "vacuum"

# Build concentric cells.
bands = []  # (outer_radius, cell)
for i, s in enumerate(spheres):
    if i == 0:
        region = -s
    else:
        region = +spheres[i - 1] & -s
    cell = openmc.Cell(fill=water, region=region)
    bands.append((sorted_radii[i], cell))

# Identify the thin-shell tally cell for each optical depth and set
# its explicit volume so OpenMC produces flux × V (rather than just
# Σ L). We'll post-process using dr = 2·h per shell.
shell_cells = []
for (r_in, r_out), r_val, h_val in zip(inner_outer, radii, half_ts):
    for i, (rad_outer, cell) in enumerate(bands):
        if abs(rad_outer - r_out) < 1e-9 and i > 0 and abs(bands[i - 1][0] - r_in) < 1e-9:
            cell.volume = (4.0 / 3.0) * np.pi * (r_out ** 3 - r_in ** 3)
            shell_cells.append(cell)
            break
    else:
        raise RuntimeError(f"no cell for shell {r_in}-{r_out}")
assert len(shell_cells) == len(OPTICAL_DEPTHS)

cells = [c for _, c in bands]
geom = openmc.Geometry(openmc.Universe(cells=cells))
geom.export_to_xml()

# === Source ==============================================================
src = openmc.IndependentSource()
src.particle = "photon"
src.energy = openmc.stats.Discrete([SOURCE_E], [1.0])
src.space = openmc.stats.Point((0.0, 0.0, 0.0))
src.angle = openmc.stats.Isotropic()

# === Settings ============================================================
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = src
settings.batches = BATCHES
settings.particles = N_HIST // BATCHES
settings.photon_transport = True
settings.electron_treatment = "ttb"
settings.cutoff = {"energy_photon": 1.0e3}
settings.export_to_xml()

# === Tallies =============================================================
# flux score with SurfaceFilter gives scalar flux at the surface,
# computed as (1 / A_surface) × Σ w_i / |mu_i| per source particle.
# Exposure weighting by E · mu_en(E)/rho via EnergyFunctionFilter.

e_grid = MU_EN_RHO[:, 0].tolist()
weight_vals = (MU_EN_RHO[:, 0] * MU_EN_RHO[:, 1]).tolist()
e_func = openmc.EnergyFunctionFilter(e_grid, weight_vals)

tallies = openmc.Tallies()
# F1 surface-current tally on each tally sphere — specifically, the
# OUTER sphere of each thin shell (at radius r + h). OpenMC restricts
# SurfaceFilter to `current` score, so this is the net outward-minus-
# inward current per source particle.
#
# For a point isotropic source in infinite absorber:
#   uncollided net current through sphere of radius r' per source
#   = exp(-μ_0 r')
# We use r' = r (not r+h) by picking the outer sphere at r+h and
# accepting the ~0.5 % offset, since the thin-shell boundary is a
# negligible perturbation for the buildup ratio.
#
# Match surfaces from the geometry by ID so OpenMC can resolve them.
surface_by_radius = {round(r, 6): s for r, s in zip(sorted_radii, spheres)}
for i, (r, h) in enumerate(zip(radii, half_ts)):
    sph = surface_by_radius[round(r + h, 6)]
    sf = openmc.SurfaceFilter(sph)
    t = openmc.Tally(name=f"exposure_sphere_{i}")
    t.filters = [sf, e_func]
    t.scores = ["current"]
    tallies.append(t)
    t_unw = openmc.Tally(name=f"unweighted_sphere_{i}")
    t_unw.filters = [openmc.SurfaceFilter(sph)]
    t_unw.scores = ["current"]
    tallies.append(t_unw)

tallies.export_to_xml()

# === Run =================================================================
openmc.run(output=False)

# === Analysis ============================================================
sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")

# Uncollided reference:
#   exposure: E_0 · mu_en(E_0)/rho · exp(-mu_0 r) per source particle
#   (surface crossing of a point-isotropic source at r gives
#    probability exp(-mu_0 r) of reaching the sphere, weighted by
#    E_0 · mu_en(E_0)/rho for exposure)
source_weight = SOURCE_E * float(mu_en_rho_water(SOURCE_E)[0])

print(f"\nWater 1 MeV exposure buildup (OpenMC F1 net-current, "
      f"{N_HIST} histories):")
print(f"{'mu_r':>6} {'B_e_meas':>12} {'B_e_ref':>10} {'rel_err':>8}"
      f" {'curr_unw':>12} {'curr_unc':>12}")

results = []
for i, (mu_r, r) in enumerate(zip(OPTICAL_DEPTHS, radii)):
    t_weight = sp.get_tally(name=f"exposure_sphere_{i}")
    t_unw = sp.get_tally(name=f"unweighted_sphere_{i}")
    curr_w = float(t_weight.mean.ravel()[0])
    curr_unw = float(t_unw.mean.ravel()[0])
    curr_unw_uncoll = float(np.exp(-mu_r))
    curr_w_uncoll = source_weight * curr_unw_uncoll
    b_e = curr_w / curr_w_uncoll
    ref = REFERENCE_BE[i]
    rel_err = abs(b_e - ref) / ref
    print(
        f"{mu_r:>6.1f} {b_e:>12.3f} {ref:>10.3f} {rel_err*100:>7.1f}%"
        f" {curr_unw:>12.4f} {curr_unw_uncoll:>12.4f}"
    )
    results.append({
        "mu_r": mu_r,
        "radius_cm": r,
        "measured_be": b_e,
        "unweighted_current": curr_unw,
        "uncollided_current_analytic": curr_unw_uncoll,
        "reference_be": ref,
        "rel_err": rel_err,
    })

payload = {
    "source_energy_ev": SOURCE_E,
    "mu_0_per_cm": mu_0,
    "mfp_cm": mfp,
    "n_histories": N_HIST,
    "reference": "Harima 1991 GP-fit (ANSI/ANS-6.6.1 point-isotropic water 1 MeV)",
    "results": results,
}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nWrote {OUT}")
