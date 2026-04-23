# Coupled neutron-photon transport + PWR γ-heating — 2026-04-23

## TL;DR

Built end-to-end photon transport on top of the existing neutron
engine, then coupled the two on a shared CSG geometry so the PWR pin
cell can drop a real γ-heating map. Landed across **three merged PRs
(#1, #2, #3)** on the `photon-transport` / `photon-spectra` /
`inelastic-gammas` branches. Final state:

- **Photon data layer** reads OpenMC per-element HDF5 files
  (`photon/*.h5`) with every sampling auxiliary a physics-correct
  kernel needs: `F(x,Z)`, `S(x,Z)`, Hartree-Fock Compton profiles,
  EADL subshells, Bremsstrahlung DCS.
- **Four photon kernels** — Compton (free KN + `S(x,Z)/Z` bound
  rejection + Doppler broadening), photoelectric with full EADL
  relaxation cascade, Bethe-Heitler pair production + in-flight
  annihilation, and Rayleigh (coherent).
- **Two transport drivers** — the original closure-based
  `transport_history` for Cs-137 / buildup validation, plus the new
  CSG-aware `transport_history_csg` that streams photons through the
  same `Surface`/`Cell`/`Region` geometry the neutron loop uses, with
  a per-cell `PhotonMaterial`.
- **Coupled (n,γ) source** sampled directly from the HDF5 per-nuclide
  cascade γ spectra (not a stub). The neutron loop tallies a
  `PhotonSourceEvent` at every capture (MT=102), fission (MT=18),
  (n,p) / (n,α) (MT=103/107, threshold-gated), and discrete-level
  inelastic (MT=4 via |Q_level|) collision. The photon phase
  consumes the event bank directly.
- **Simplified electron transport** — CSDA midrange deposit via
  Katz-Penfold ranges, with lattice reflection on the displaced
  positions so electrons born near a reflective BC stay inside the
  fundamental cell.
- **`pwr_gamma_heating` binary** ties it all together: short neutron
  eigenvalue → aggregate γ-source bank → photon phase → per-cell
  deposition report.

Converged run on the standard PWR pin (UO₂ 3.1 % / Zr-4 / H₂O,
1.26 cm pitch, reflective lattice, 150 batches × 50 k neutrons +
200 k γ, ~2.5 min on desktop CPU):

| region | fraction |
|--------|---------:|
| fuel   | **84.42 %** |
| gap    | 1.46 % (*midrange artefact — see below*) |
| clad   | 7.88 % |
| water  | 5.90 % |
| escape | 0.00 % (reflective lattice — sanity) |
| sum    | 99.66 % (missing 0.34 % = EADL valence-binding loss) |

The 1.46 % in the He gap is a known simplification artefact: a
fuel-surface-born electron with R_e/2 ≈ 0.02 cm lands inside the
gap instead of continuing through it into the clad. Full
track-integrated electron transport would re-attribute that slice
to clad, giving ~84 / 0 / 9 / 6.

## What the benchmark measures (layman-friendly)

A fresh UO₂ fuel rod in a reactor doesn't just release heat as
kinetic energy of fission fragments and neutrons — about **8-10 % of
its power comes out as high-energy γ-rays** (prompt fission γs
~6 MeV, capture γs ~7 MeV, inelastic γs ~1 MeV). Those γs are
volumetric; unlike the kinetic-energy pieces they don't all deposit
where they're born.

`pwr_gamma_heating` answers the question *"for every 100 units of γ
energy born inside the fuel pin, where does each unit end up actually
depositing?"* Reactor designers use this split to size cladding
thermal margins, predict moderator γ-heating, and compute fuel
temperature profiles.

The pipeline is two-phase, on a shared CSG:

1. **Neutron phase** — short k-eigenvalue on the pin. Every time a
   neutron captures, fissions, or inelastically scatters, sample a
   γ multiplicity from HDF5 yield tables and outgoing energies from
   HDF5 `distribution_0/energy`. Store `(cell, pos, E_γ, MT)`.
2. **Photon phase** — for each of 200 k photon histories, draw a
   random source event from the bank, transport through the same
   CSG with per-cell `PhotonMaterial` (cross sections + mass
   density), bin the per-collision deposits into the containing
   cell. Apply electron-range displacement on every Compton /
   photoelectric / pair deposit.

## Session progression

### PR #1 — CSG driver + first coupled pipeline (`photon-transport`)

Lifted the photon driver from "single homogeneous
material + closure-based `is_inside`" to a real CSG transport loop
mirroring the neutron one. Per-cell `PhotonMaterial`, full
Vacuum / Reflective / Transmission BC handling, void-cell streaming,
banked secondaries.

Added a per-cell `(n,γ)` capture tally to `BatchResult` so
`pwr_gamma_heating` can source photons from the real spatial
capture distribution rather than a uniform-in-fuel stub. Source
energy was a notional two-line spectrum (70 % × 1 MeV + 30 % ×
5 MeV) — good enough to prove the plumbing works.

### PR #2 — real HDF5 γ spectra (`photon-spectra`)

Replaced the stub source with the per-nuclide cascade spectra
stored under `reactions/reaction_{mt}/product_{N}` in the OpenMC
HDF5 layout. Reader, sampler, and XS-provider wiring mirror the
existing fission-neutron outgoing-energy path — zero new physics,
just re-aim the same ContinuousTabular reader at photon products.

Third pass (v3 of the same PR) loaded **all** photon products per
MT (O-16 has 4 products for MT=102, 6 for MT=107) and added
MT=103 `(n,p)` + MT=107 `(n,α)`. Verified isotropic-angular was
already exact for U-235/U-238 (ENDF stores photon angles as 2-point
linear μ ∈ [−1, 1] = uniform).

### PR #3 — inelastic γs + simplified electron transport (`inelastic-gammas`)

Final piece. Two changes:

1. **Inelastic γ emission.** ENDF/B-VII.1 doesn't tabulate
   `particle="photon"` products for discrete inelastic levels on
   U-235 / U-238 (the de-excitation γ is implicit in the level's
   Q-value). Added `CollisionOutcome::InelasticScatter { q_value_ev }`
   so the three collision sites in `transport::simulate` can bank
   a γ event with `energy = q_value_ev.abs()` and `mt = 4`.
2. **CSDA midrange electron transport.** Each Compton /
   photoelectric / pair recoil-electron deposit is displaced
   forward by `R_e(E) / 2` along the incoming photon direction,
   with `R_e` computed by Katz-Penfold `R[g/cm²] = 0.412 ·
   E^(1.265 - 0.0954 ln E)` divided by per-material mass density.
   `fold_into_lattice` in the binary reflects displaced positions
   back into the fundamental cell via a triangle-wave on each axis.

## Numerical results

| metric | PR #1 (stub) | PR #2 (HDF5 spectra) | PR #3 (+inelastic +e-range) |
|---|---:|---:|---:|
| photon source events | 50 k sampled | 27.9 M | 29.9 M |
| fuel fraction | 84.5 % | 84.8 % | **84.4 %** |
| gap fraction | 0 % | 0 % | 1.5 % *(e-range artefact)* |
| clad fraction | 9.7 % | 9.1 % | 7.9 % |
| water fraction | 5.6 % | 5.8 % | 5.9 % |
| escape | 0 | 0 | 0 |
| sum | 99.76 % | 99.65 % | 99.66 % |

Neutron side stayed at k_∞ = 1.327 (correct for fresh 3.1 % UO₂
infinite lattice) across all three runs — the γ-heating work didn't
touch the neutron physics.

The 27.9 M → 29.9 M jump in PR #3 is the +2.0 M inelastic-γ events
now flowing.

## Structural gap vs textbook (~93 / 3 / 2)

We land at ~84 / 9 / 6 rather than the commonly-quoted
~93 / 3 / 2 split. Possible contributors, ranked by suspected
impact:

1. **Benchmark geometry / composition mismatch.** Published numbers
   are usually VERA Problem 1 or similar; atom densities, gap
   thickness, moderator density, and enrichment differ by a few %
   from ours. An OpenMC cross-code comparison on the identical
   geometry would quantify this (half-day item, not done).
2. **Kerma-approximation delocalization.** Our CSDA midrange keeps
   electrons in fuel for 1 MeV-class γs (R_e/2 ≈ 0.02 cm in UO₂)
   but full condensed-history electron transport would recapture
   the ~1.5 % of fuel-surface electrons currently attributed to the
   gap.
3. **Inelastic γ yield approximation.** We emit one γ at
   `energy = |Q_level|` rather than sampling the actual cascade
   multiplicity and individual photon energies. OpenMC does the
   full cascade where the data supports it.

Whether the gap is real or a benchmark-mismatch is unresolved
pending the cross-code run.

## What's in `main` at this point

- `src/photon/{data,hdf5_reader,coherent,compton,photoelectric,pair,material,transport}.rs`
- `src/bin/{cs137_pulse_height,photon_dump,pwr_gamma_heating}.rs`
- `src/transport/simulate.rs::{PhotonSourceEvent, BatchResult::photon_events, sample_photon_products, ABSORPTION_PHOTON_MTS, FISSION_PHOTON_MTS}`
- `src/hdf5_reader.rs::{PhotonProduct, read_photon_products}`
- `src/physics/collision.rs::CollisionOutcome::InelasticScatter`
- Integration tests in `tests/`:
  - `cs137_pulse_height_validation.rs`
  - `hubbell_compton_differential.rs`
  - `ansi_ans_buildup.rs`

148 / 148 library tests green. `cargo fmt --check` + `clippy -D warnings`
clean on all platforms (ubuntu + windows) in CI.

## Next natural steps

- **OpenMC cross-code run** on the exact same UO₂ / Zr / H₂O / 1.26 cm
  pin (with `Settings.photon_transport=True`). Half-day item; the
  only way to know if our ~84 / 9 / 6 split is the right answer for
  this problem or if there's an actual physics fix to chase.
- **Full electron transport.** 2-3 weeks of work to add a
  condensed-history electron kernel with bremsstrahlung. Removes
  the kerma-approximation bias (~5 % in shielding, <1 % in PWR
  pin γ-heating because electron ranges are much smaller than the
  fuel pellet). Out of scope for this round.
- **Shielding benchmarks.** Kobayashi dog-leg and ANS-6.4.3 are
  now one binary away (just need a different geometry). Photon
  stack is already validated against Cs-137 pulse-height + Hubbell
  Compton + ANSI/ANS-6.6.1 buildup. No new kernels required.
- **SVD-compressed photon cross sections.** The novel angle: the
  incoherent / photoelectric / pair XS curves are smooth log-log,
  very likely rank ≤ 2. No published code compresses photon data
  this way. A paper's worth.

## Commit map

```
0341243  Merge PR #2  Coupled n-γ real HDF5 γ spectra
588b534  Merge PR #1  Photon transport data + kernels + CSG + coupled source
8739817  Merge PR #3  Inelastic γ + simplified electron transport
fd2f19e      CSDA midrange + lattice fold
43d9608      Inelastic γ at MT=51..91 sites
18ceb97      All photon products per MT + MT=103/107
02808e6      Real HDF5 γ spectra for capture + fission
5fb8bc8      Coupled neutron-photon v1: real (n,γ) capture tally
94d7439      pwr_gamma_heating binary (stub spectrum)
958e148      transport_history_csg CSG driver
```
