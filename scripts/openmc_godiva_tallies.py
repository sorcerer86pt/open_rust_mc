# SPDX-License-Identifier: MIT
"""OpenMC Godiva with reaction-rate + leakage tallies for channel-level
cross-code comparison against the Rust engine. Goal: localize the
~430 pcm Godiva fast-spectrum offset.

Tallies (per source particle):
  - leakage current through outer sphere (r=8.7407)
  - per-nuclide reaction rates: elastic (MT=2), fission (MT=18),
    capture (MT=102), inelastic sum (MT=4), (n,2n) (MT=16),
    (n,3n) (MT=17)
  - nu-fission (production rate)
  - total collision rate (sum of all reactions)
Output: JSON with ratios to Rust's aggregates.

Usage (WSL):
    python scripts/openmc_godiva_tallies.py
"""
import os
import time
import json
import numpy as np
import openmc

DATA = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5"
OUT  = "/mnt/c/Users/fog/madman_svd_experiment/outputs/openmc_godiva_tallies.json"

WORK = "/tmp/openmc_godiva_tallies"
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

BATCHES, INACTIVE, PARTICLES, SEEDS = 100, 20, 20000, 3

# Materials
fuel = openmc.Material(name="HEU")
fuel.add_nuclide("U235", 0.9327, "ao")
fuel.add_nuclide("U238", 0.0524, "ao")
fuel.add_nuclide("U234", 0.0149, "ao")
fuel.set_density("g/cm3", 18.74)
fuel.temperature = 294
mats = openmc.Materials([fuel])
mats.cross_sections = f"{DATA}/cross_sections.xml"
mats.export_to_xml()

# Geometry
sph = openmc.Sphere(r=8.7407, boundary_type="vacuum")
cell = openmc.Cell(fill=fuel, region=-sph)
geom = openmc.Geometry(openmc.Universe(cells=[cell]))
geom.export_to_xml()

# Tallies
tallies = openmc.Tallies()

# 1. leakage current on outer sphere
leak_filter = openmc.SurfaceFilter(sph)
t_leak = openmc.Tally(name="leakage")
t_leak.filters = [leak_filter]
t_leak.scores = ["current"]
tallies.append(t_leak)

# 2. per-nuclide reaction rates in fuel cell
cell_filter = openmc.CellFilter(cell)
for score in ["elastic", "fission", "absorption", "(n,2n)", "(n,3n)",
              "(n,gamma)", "total", "nu-fission", "scatter"]:
    t = openmc.Tally(name=f"rate_{score}")
    t.filters = [cell_filter]
    t.nuclides = ["U235", "U238", "U234"]
    t.scores = [score]
    tallies.append(t)

# 3. energy-resolved total rate (coarse groups to see where collisions happen)
egroups = [0.0, 1e-1, 1e3, 1e5, 1e6, 2e6, 5e6, 2e7]
ebins = openmc.EnergyFilter(egroups)
t_e = openmc.Tally(name="rate_by_energy")
t_e.filters = [cell_filter, ebins]
t_e.scores = ["total", "fission", "absorption", "scatter"]
tallies.append(t_e)

# 3b. fine energy-resolved fission rate. The coarse 7-bin tally above
# gives ⟨E⟩ to ~1%, but its σ(E) computed from midpoints is wildly
# biased upward (the 0–1e-1 and 1e-1–1e3 bins contribute zero rate
# but huge bin-midpoint² values would, if rates weren't zero there;
# more importantly, real fission events that fall in the wide 1e6–2e6
# or 2e6–5e6 bin all get binned at the midpoint, smearing σ). The
# fine tally lets the Rust diagnostic compute σ(E_in fission) at
# 100-bin log resolution for a faithful CPU↔GPU↔OpenMC comparison.
fine_egroups = np.logspace(np.log10(1e3), np.log10(2e7), 101).tolist()
fine_egroups = [0.0] + fine_egroups
fine_ebins = openmc.EnergyFilter(fine_egroups)
t_fine_fis = openmc.Tally(name="fission_by_energy_fine")
t_fine_fis.filters = [cell_filter, fine_ebins]
t_fine_fis.scores = ["fission"]
tallies.append(t_fine_fis)

tallies.export_to_xml()

results_per_seed = []
for seed in range(SEEDS):
    s = openmc.Settings()
    s.batches = BATCHES
    s.inactive = INACTIVE
    s.particles = PARTICLES
    s.seed = seed + 1
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0, 0, 0))
    src.energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
    s.source = [src]
    s.export_to_xml()
    t0 = time.time()
    openmc.run(output=False)
    dt = time.time() - t0
    sp = openmc.StatePoint(f"statepoint.{BATCHES}.h5")
    k = sp.keff
    tally_results = {}
    for t in sp.tallies.values():
        # mean across active batches, per source particle (OpenMC normalises by default)
        mean = np.asarray(t.mean).flatten()
        std  = np.asarray(t.std_dev).flatten()
        nuclides = list(getattr(t, 'nuclides', [])) if t.nuclides else [None]
        tally_results[t.name] = {
            "mean": mean.tolist(),
            "std":  std.tolist(),
            "nuclides": [str(n) for n in nuclides] if nuclides != [None] else None,
        }
    results_per_seed.append({
        "seed": seed,
        "k": float(k.nominal_value),
        "sigma_k": float(k.std_dev),
        "time_s": dt,
        "tallies": tally_results,
    })
    print(f"seed {seed}: k={k.nominal_value:.5f} +/- {k.std_dev:.5f}  {dt:.1f}s", flush=True)
    sp.close()
    for f in [f"statepoint.{BATCHES}.h5", "tallies.out", "summary.h5"]:
        if os.path.exists(f):
            os.remove(f)

# Aggregate across seeds
ks = [r["k"] for r in results_per_seed]
agg = {
    "k_mean": float(np.mean(ks)),
    "sigma_seeds": float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0,
    "batches": BATCHES, "inactive": INACTIVE, "particles": PARTICLES, "seeds": SEEDS,
    "per_seed": results_per_seed,
}
# Per-tally mean across seeds (first seed's labels apply)
first = results_per_seed[0]["tallies"]
tally_agg = {}
for tname, tdata in first.items():
    stacked = np.stack([np.asarray(r["tallies"][tname]["mean"]) for r in results_per_seed])
    tally_agg[tname] = {
        "mean": stacked.mean(axis=0).tolist(),
        "std_seeds": stacked.std(axis=0, ddof=1).tolist() if len(results_per_seed) > 1 else np.zeros_like(stacked[0]).tolist(),
        "nuclides": tdata["nuclides"],
    }
agg["tallies_seed_mean"] = tally_agg
agg["energy_groups_MeV"] = [e/1e6 for e in egroups]
agg["fine_fission_groups_eV"] = fine_egroups
with open(OUT, "w") as fh:
    json.dump(agg, fh, indent=2)
print(f"\nmean k = {agg['k_mean']:.5f}  sigma_seeds = {agg['sigma_seeds']:.5f}")
print(f"wrote {OUT}")
