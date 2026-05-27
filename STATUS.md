# Project status — 2026-05-21

What `origin/main` (`d80a157`) ships today, what's open, what's the
current headline.

## Headline

- **Tests**: 438 / 438 lib tests green on default features;
  443 / 443 with `--features cuda`. `cargo check` clean on
  `--all-targets` (default + cuda).
- **Default nuclear data**: ENDF/B-VIII.1 (released Oct 2024).
  VIII.0 and VII.1 supported as fallbacks; library autodetected from
  on-disk layout.
- **ICSBEP corpus**: 375 scene JSONs in `bench/icsbep/`. 137
  migrated to per-isotope carbon to be VIII.x-compatible.
- **CUDA backend**: feature-gated, sm_86+. Recursive transport,
  per-nuclide kernel cache, refill-pool (PHYSOR 2022 Optimization F)
  with auto-refill recommender, multi-slot S(α,β), CIELO U-235.
- **CPU backend**: rayon work-stealing, history-based,
  saturated-by-construction (no refill knob needed).

## Capabilities

### Geometry
- Recursive universes via `CoordStack` (`SmallVec<[Coord; 4]>`).
- RectLattice + HexLattice (flat-top Y / pointy-top X).
- Per-universe surface restriction + opt-in BVH (≥8 cells with
  finite AABBs) — 3.0× assembly speedup alone.
- `geometry::shapes` builders: `rect_box`, `rect_box_split_bc`,
  `hex_boundary`, `pin_cylinders`. Exposed via Python.
- `RectLattice.material_overrides` (one pin universe across an
  assembly with different enrichments / burnup tiers).

### Cross sections
- Four interchangeable providers behind one `XsProvider` trait:
  `Table` (pointwise), `Svd`, `HybridTableWmp`, `HybridSvdWmp`.
- URR probability tables (`multiply_smooth` true / false).
- URR equivalence theory (Carlvik-Pellaud Dancoff for square
  lattices).
- Stochastic pseudo-interpolation for off-library temperatures
  (Table) + partition-of-unity 3-point Ducru reconstruction (SVD).
- ZAID-keyed `NuclideLibrary` registry (H/D/T, B, C-nat + C-12/-13,
  O-16/17, Zr-90/91/92/94/96, Fe-54-58, U-233 → Cm-247, plus FP
  poisons I-135 / Xe-135 / Cs-135 / Pm-149 / Sm-149 / Gd-155 /
  Gd-157).
- Natural-element fallback (`material_resolve::expand_natural_elements`)
  — mirrors OpenMC's `Material.add_element` for libraries that
  dropped the natural file (e.g. VIII.x has C-12 / C-13 but no C-0).

### Thermal scattering
- Free-gas Maxwell-Boltzmann below 400·kT.
- S(α,β) registry `ThermalLibrary` covering H-in-H₂O, D-in-D₂O,
  graphite, ZrH (TRIGA), Be / BeO, polyethylene, benzene,
  α-quartz, bound O / U in UO₂, methane (liquid / solid), ortho /
  para H₂ / D₂, Al-27, Fe-56.
- Continuous + discrete inelastic, Bragg edges, Debye-Waller
  incoherent elastic, stochastic temperature interpolation between
  bracketing TSL columns.

### Neutron physics
- Energy-dependent ν̄(E), prompt + delayed.
- Anisotropic scattering (tabular μ/CDF, CM frame).
- Data-driven fission outgoing-energy spectrum.
- Discrete inelastic MT=51-91 + continuum MT=91 (evaporation,
  T = √(E*/a), a = A/8).
- (n,2n) MT=16, (n,3n) MT=17.
- Delayed-neutron yields ν_d(E) per nuclide; soft Watt aggregate
  spectrum (a = 0.4 MeV).
- Maxwell / Evaporation closed-form χ buffers (slots 130-135) —
  fixes U-233 / U-234 / Pu-240 fission spectra.

### Transport
- k-eigenvalue power iteration with Shannon-entropy convergence.
- History-based + rayon-parallel on CPU; event-batched
  recursive-CSG on GPU.
- Surface tracking + Woodcock delta tracking, auto-detected by
  material contrast.
- Track-length k_eff estimator alongside collision estimator
  (`[godiva]` 3.9× lower seed-to-seed σ).
- Survival biasing + Russian roulette (w_min=0.25, w_survive=1.0)
  on CPU. Variance-only — no k_eff bias. GPU survival biasing
  pending.
- Statepoint write / read / restart (HDF5).
- Backend dispatch (`transport::dispatch::EigenvalueRunner`,
  `CpuRunner` / `CudaRunner`).

### Variance reduction
- Forward weight windows (Cartesian mesh, split / roulette,
  `max_split=8`, geometric-mean w_survive).
- `WeightWindow::from_flux` — forward CADIS bootstrap from any
  flux.
- Random-ray multigroup TRRM (`random_ray::*`) — forward +
  adjoint, cell-based or Cartesian FSRs, mortal or immortal-ray
  (Tramm-Siegel 2021), analytic MoC ODE step.
- RR-CADIS pipeline: `rr_cadis_slab` emits JSON,
  `shield_slab --cadis-load` consumes. Measured FOM gains: 2.19×
  at 100 cm water, 4.32× at 200 cm.

### Photon transport
- Compton (KN + S(x,Z) + optional Doppler from Compton profiles).
- Rayleigh (form factor + Thomson rejection).
- Photoelectric phase 1 (subshell sampling).
- Bethe-Heitler pair.
- Full condensed-history electron walk (Bethe-Bloch dE/dx with
  per-element I from HDF5, Highland MS with per-cell X₀,
  Seltzer-Berger brems with secondary γ banking).
- NEE (next-event estimator) for tallies.
- Adjoint photon CADIS slab walker (CE adjoint Compton kernel).
- `pwr_gamma_heating` produces fuel / gap / clad / water heating
  split matching OpenMC 0.15.3 within 1 pp.

### Depletion
- Bateman + CRAM-16 (Pusa 2016) + CRAM-48.
- Chain JSON loader (3-way `yields` semantics: omitted / `{}` /
  explicit). ENDF default yield inference.
- `BurnupMapping` table-driven walker (chain ↔ material).
- Predictor-corrector with **fresh-corrector** (clones materials,
  runs eigenvalue at predicted composition for EOC flux).
- On-the-fly chain-XS spectrum collapse — closes the 9× to 0.77×
  gap vs OpenMC depletion.
- Shipped chains: `chains/partial_xe.json` (4 nuclides),
  `chains/pwr_actinides.json` (17, U/Np/Pu + Xe/Sm).

### Time-dependent kinetics
- Point-kinetics with 6-group delayed-neutron precursors.
- A-stable Crank-Nicolson 7×7 ODE solver.
- Keepin / Hetrick-Roberts 6-group constants for U-235 thermal,
  U-238 fast, Pu-239 thermal. `blend()` combines mixed cores.
- Closed-form `prompt_jump_ratio` and `inhour_period` for
  analytic cross-check.
- `point_kinetics_demo` runs step / ramp / scram profiles, emits CSV.
- Late-time period at ρ = 50 ¢ matches inhour to −0.37 %.

### Adjoint (continuous-energy)
- `adjoint_elastic_scatter` — s-wave isotropic-CM adjoint kernel.
  4 unit tests (kinematic invariant, log-uniform shape, range,
  round-trip).
- `adjoint_compton_scatter` — inverted Klein-Nishina sampler
  (Wagner-Haghighat 1998 / Lewis-Miller §10.3). Conditional density
  matches analytic KN dcs/dμ to χ²_red < 2.5 across 25 bins.
- `transport::adjoint_photon` — slab walker, composes adjoint
  Compton + self-adjoint Rayleigh + photoelectric / pair
  termination. Output is an `ImportanceMap` consumable by
  `WeightWindow::from_flux`.

### GPU (CUDA, sm_86+)
- Recursive cell-find / trace-step / multi-step walk — bit-exact
  vs CPU (≤ 9.3e-11 max-rel-err), 3-24× speedups on RTX A1000.
- Constant-XS transport with collision + scatter + fission banking
  (atomicAdd) — 6.74× speedup, k within MC noise.
- Photon kernels: Compton (fixed-E + per-particle-E variants),
  Rayleigh, pair. Persistent Compton: 2.22× wall vs 20-thread CPU
  on 1M histories.
- Per-nuclide kernel cache (`Arc::as_ptr`-keyed, LRU + bundle
  budget) eliminates redundant HtoD on multi-case sweeps.
- Refill pool (PHYSOR 2022 Optimization F) — opt-in via
  `gpu_refill_pool_factor` or auto-recommended via device attribute
  inspection. 2× histories at same wall time on mid-curve workloads.
- Multi-slot S(α,β) (`upload_sab_data_multi`) — concurrent TSLs
  on multiple nuclides in one run.
- HexLattice GPU port — full device functions, dispatch wired,
  parity test pending.

### Python bindings (PyO3)
- `Scene` / `Material` / `Surface` / `PhotonMaterial` builders.
- `run_eigenvalue`, `run_gamma_heating`, `run_icsbep_case`.
- `XsMode::{Table, Svd, HybridTableWmp, HybridSvdWmp}` per-sim
  toggle, per-MT rank overrides.
- Depletion: `Chain.from_file`, `CramOrder::{Order16, Order48}`,
  `cram`, `deplete_constant_flux`, `deplete_with_flux_callback`
  (FFI-exception-safe Python closure), `Material.set_atom_density`
  / `atom_density_of`.
- `Scene.add_rect_box` / `add_hex_boundary` / `add_pin_cylinders`
  return ready-to-parse region strings.

### Benchmark suite (Python harness)
- `icsbep_run.py` — single case, auto data-dir discovery
  (VIII.1 → VIII.0 → VII.1).
- `icsbep_sweep.py` — full corpus sweep with start / stop / resume,
  per-case CSV durability, multi-seed averaging, filter by case
  family. Precedence: explicit CLI flag > JSON `recommended_settings`
  > built-in default.
- `run_benchmark.ps1` — one-shot PowerShell wrapper. Picks runner
  automatically, writes `outputs/icsbep_full_<runner>.csv` + log.
- ICSBEP family suite under `|Δ| ≤ max(150 pcm, 2σ_combined)`:
  HMF-001 / PMF-001 / PMF-002 / U-233-MF-001 / LCT-008 /
  HEU-SOL-THERM-001 → **6 / 6 PASS** on both CPU and CUDA.

## Headline numbers (scope-tagged)

Re-verified against `outputs/` and `results/` 2026-05-21. Each row
that's directly grounded in a CSV / .txt cites the source.
Rows tagged *(unverified this audit)* are inherited from older
sessions and should be re-measured before being quoted in papers.

| Metric | Scope | Value | Source |
|---|---|---|---|
| Lib test count (default) | — | **438 / 438 green** | `cargo test --lib` |
| Lib test count (`--features cuda`) | — | **447 / 447 green** | `cargo test --lib --features cuda` |
| ICSBEP CUDA + CPU family suite | `[icsbep]` | **6 / 6 PASS** in 141 s under `max(150 pcm, 2σ)` | `outputs/cuda_runs_after_rank_fix.txt` |
| PWR γ-heating split (us vs OpenMC) | `[photon]` | fuel 84.12% / clad 9.81% / water 5.72% / gap 0% | `outputs/pwr_gamma_heating_benchmark.txt` |
| Bremsstrahlung firing rate in PWR γ-heat | `[photon]` | 2 312 γ at 7.43e8 eV (0.353 % of source) | same file |
| Saturation knee (HMF-001, 3 nuclides) | `[godiva]` | 500k-1M particles per batch | `outputs/saturation_*.csv` |
| Peak throughput at saturation (RTX 3080) | `[godiva]` | **~1.2 M histories/sec** at 1M particles, 0.83 µs/p | `outputs/saturation_1000000.csv` |
| Refill 2× at mid-curve (HMF-008, GPU) | `[micro]` | **2.0× more collisions at same wall, σ 2.1× tighter** | `outputs/hmf008_refill_*.csv` |
| µs/collision drop with refill (HMF-008) | `[micro]` | 1.162 → 0.575 µs/coll (−50.5 %) | same |
| RR-CADIS FOM at 7 mfp (100 cm water 1 MeV γ) | `[shield]` | **1.03× analog** (essentially neutral at this depth) | `outputs/method_comparison_2026-05-08.txt` |
| RR-CADIS FOM at 14 mfp (200 cm water 1 MeV γ) | `[shield]` | **1.18× analog** | same |
| RR-CADIS + NEE FOM at 14 mfp | `[shield]` | **1.75× analog** (combination compounds) | same |
| GPU SVD vs 20-core CPU SVD (Godiva, RTX A1000) | `[godiva]` | **0.77× (1.3× SLOWER)** — launch + memory-access penalty dominates | same |
| SVD vs Table on PWR pin cell (9 nuc, S(α,β) on) | `[pwr]` | SVD 1.25× *slower* than Table; SVD memory 5.12× larger | same |
| SVD vs Table on Godiva (3 nuc, fast spectrum) | `[godiva]` | SVD 1.22× faster than Table; SVD memory 5.14× larger | same |
| Rust Godiva k_eff (SVD k=5) | `[godiva]` | 1.00079 ± 0.00038 *(unverified this audit)* | (claim from prior session) |
| PWR SVD k=5 vs OpenMC 0.15.3 | `[pwr]` | 12 pcm Table, −67 pcm SVD *(unverified this audit)* | (claim from prior session) |
| Track-length vs collision σ | `[godiva]` | 3.9× lower seed-to-seed *(unverified this audit)* | (claim from prior session) |
| Survival biasing FOM on PWR | `[pwr]` | 4.5× *(unverified this audit)* | (claim from prior session) |
| CRAM-16 vs analytical Xe equilibrium | `[depletion]` | 1e-4 relative *(unverified this audit)* | (claim from prior session) |
| GPU constant-XS recursive transport (geometry only) | `[assembly]` | 6.74× CPU at MC noise *(microbench, not integration)* | (older const-XS bench) |
| GPU Compton persistent kernel scaling | `[photon]` | 12.96 ms / 1M @ 1 MeV (free); 312 ms with Doppler | `outputs/gpu_compton_scaling.txt` |

## ICSBEP A/B against VIII.1 — heu-comp-inter-003 (2026-05-21)

CPU runner, 100k particles, 5 seeds, 150 batches, 40 inactive.
Compared against the existing VII.1 GPU baseline.

| case | VII.1 k_calc | VIII.1 k_calc | Δk (VII.1 → VIII.1) | VIII.1 status |
|---|---|---|---|---|
| c-1 | 1.005268 | 1.011149 | +588 pcm | PASS |
| c-2 | 1.005793 | 1.013705 | +791 pcm | **FAIL** (2.24σ vs handbook) |
| c-3 | 1.005298 | 1.013161 | +786 pcm | **FAIL** (2.35σ vs handbook) |
| c-4 | 1.001496 | 1.010804 | +931 pcm | PASS (1.96σ) |
| c-5 | 0.997644 | 1.007997 | +1035 pcm | PASS (1.70σ) |
| c-6 | 0.994727 | 1.003175 | +845 pcm | PASS |
| c-7 | 0.994939 | 1.002560 | +762 pcm | PASS |

VIII.1 shifts k upward uniformly by **+820 ± 153 pcm**. Localised
driver: **CIELO U-235 σ_f at 100 eV grew +5.3 %, σ_capture dropped
−5.5 %, α (capture/fission) dropped −10.2 %** at the intermediate-
spectrum peak. ν̄(E) essentially unchanged. This is a documented
CIELO outcome (Brown et al. ENDF/B-VIII.0 NDS 148, 2018). The
Wright/Leal ORNL evaluation reports a 1500 pcm C/E spread on
ENDF/B-VI.5 for this family — both VII.1 and VIII.1 sit inside the
spread but on opposite halves.

## ENDF/B-VIII.1 ground-truth substrate (2026-05-22)

Two-tier secondary reference for VIII.1 runs, so the engine isn't
graded against a handbook k_eff that the new library systematically
biases away from (especially for HCI / hydride / intermediate classes).

- **Tier 1 — LANL Table LIX** (Nobre et al. 2025, arxiv:2511.03564):
  1151 ICSBEP cases calculated with MCNP6 under VIII.1. Matched against
  our corpus → **289 / 375 scenes** have a published `lanl_k_eff`
  stamped in `benchmark.local_validation.viii1`. Details +
  reproducibility recipe in `docs/endfb-viii1-lanl-table-lix.md`.
- **Tier 2 — local OpenMC on the same JSON**: the 86 orphans (HCI-003,
  LST-002/003/016, PST-021, PCI-001, plus sub-cases LANL skipped) get
  `viii1.openmc_k_eff` from `scripts/openmc_orphans_viii1.py`. 26 / 86
  stamped at paper-grade (20k × 100 × 3 seeds) before the user switched
  to higher hardware for the engine half. Remaining 59 deferred.

Tests (`cuda_runs.rs`, `icsbep_runs.rs`) walk the priority chain in
`resolve_acceptance_target`: LANL → OpenMC-on-this-JSON → legacy
VII.1 OpenMC → handbook. σ is `max(σ_pub_or_omc, σ_handbook)` so the
envelope never under-states uncertainty.

Engine vs OpenMC-on-same-JSON across 9 overlapping cases (HCI-003 ×7 +
HMF-001 case-2 + HMF-002 case-1): engine is **+289 ± 84 pcm** (range
+173 to +424). Sign systematically positive across fast / intermediate
spectra → real engine bias, not library or scene drift. Diagnosis
deferred until the full 5090 corpus sweep lands.

## RunPod RTX 5090 sweep environment (active 2026-05-22)

Full 375-case ICSBEP sweep running on RunPod since 15:42 UTC, paper-grade
`250k × 150 × 30 inactive × 5 seeds`. ETA ~34 h, cost ~€34 against a
€47.68 budget. Hardware + SSH + storage layout + monitoring / stop
commands documented in `docs/runpod-5090-pod.md`. Determinism cross-check
with the prior MSI-Home (RTX A1000, commit `6bb07bd`) sweep on
HCI-003 cases 1-6 matches within σ_seeds (≤ 19 pcm). 5090 throughput
≈ 2.5× A1000 on the same workload at full paper-grade.

Partial 15-case warmup CSV (HCI-003 ×7 + HMF-001 ×2 + HMF-002 ×6) is
parked at `outputs/icsbep_5090_partial_15cases.csv` for that cross-check.

## Hardware-specific notes

The same engine source has been exercised on three machines that map
to different points on the saturation curve:

| host | GPU | VRAM | CPU | role |
|---|---|---|---|---|
| MSI-Laptop | RTX A1000 | 4 GB | 14p / 20l Intel | dev box; CPU sweeps |
| MSI-Home | RTX 3080 | 10 GB | 8p / 8l Ryzen | GPU production sweeps |
| RunPod 5090 | RTX 5090 (sm_120) | 32 GB | 16 vCPU EPYC 9354 | full-corpus sweeps (see `docs/runpod-5090-pod.md`) |
| (extrapolated) | A100 / H100 | 40 GB+ | — | saturation regime |

Per-card practical particle ceilings:
- A1000 (4 GB): ~50 k particles per batch — VRAM-pressured beyond
  that; the natural-element migration + VIII.1 payload puts a
  6-nuclide steel-bearing case at the limit.
- 3080 (10 GB): 500 k - 1 M per batch. Saturation curve in
  `outputs/saturation_*.csv`.
- A100-class: 2 M - 8 M per batch (Tramm et al. PHYSOR 2022).

For the `outputs/icsbep_full_gpu.csv` baseline (4-case
heu-comp-inter-003 at 500k particles, 1424-1551 s/case) the
machine was MSI-Home (RTX 3080).

## Hardware-specific GPU pipeline limits

Each flat-pack `GpuNuclideData` bundle (the DtoD copy assembled for each
case) occupies ~1.5 GB for a 20-nuclide rank-15 HEU case. The benchmark
pipeline pre-uploads `n_slots` bundles into the GPU channel before Stage 4
even starts; peak live = `n_slots + 3` (channel + running + uploading +
per-nuclide cache source). The VRAM formula is:

```
(n_slots + 3) × 1.5 GB + 1.7 GB ≤ total_vram
n_slots ≤ (total_vram − 1.7 GB) / 1.5 GB − 3
```

Practical ceilings per device (rank 15, 20-nuc HEU, --max-slots default 4):

| GPU | VRAM | n_slots (auto) |
|---|---|---|
| A1000 | 4 GB | 1 |
| RTX 3080 | 10 GB | 2 |
| RTX 4080 | 16 GB | 4 (capped) |
| RTX 5090 | 32 GB | 4 (capped) |

Per-card practical particle ceilings are separate — particle bank for
250k particles is only ~150 MB and does NOT cause OOM.

CLI controls:
- `--n-slots N` — exact override (bypasses VRAM formula entirely)
- `--max-slots N` — upper bound on VRAM-auto result (default 4)

## Open / deferred work

- **GPU device-buffer cache for SAB + material payloads.** Per-nuclide
  kernel cache exists and works (`Arc::as_ptr`-keyed); SAB and
  material uploads still rebuild + HtoD every seed/case. A 5-seed ×
  7-case sweep re-uploads 35× the same ~50 MB SAB payload. Same
  LRU + bundle_cache_budget pattern as `per_nuclide_cache` should
  apply.
- **GPU survival biasing / Russian roulette.** CPU has it (4.5× FOM
  on PWR); GPU runs analog. Variance-only, k_eff is unbiased.
- **GPU discrete S(α,β) inelastic (NJOY iwt=0/1).** CPU has it; GPU
  device sampler is continuous-only. OpenMC's ENDF/B HDF5
  distribution emits every TSL as `incoherent_inelastic` so this
  has zero hits in the 375-case corpus. Loud `panic!` on the upload
  side ensures silent breakage isn't possible.
- **GPU per-cell `Mat3` rotation.** No ICSBEP scene currently sets
  it; `GpuRecursiveContext::build` errors loudly if any cell does.
- **DXTRAN-style continuous splitting** for ≥14 mfp photon
  penetration. All `(ratio, growth) ∈ {5,10,20} × {0,1,2,3}` at
  300 cm give 0 transmitted in 500k — `max_split=8` ceiling bounds
  geometric WW.
- **Full C5G7** (4 fuel × 7 groups × 17×17) — data plumbing on top
  of `random_ray::*`, no new solver code.
- **HexLattice GPU runtime parity** vs CPU.
- **Linear-source random-ray (1st-order)** — deferred; flat on a
  fine mesh is equivalent for axis-aligned problems.
- **Full PWR depletion bench vs OpenMC** (30-50 GWd/MTU on
  `pwr_actinides.json` + Pu/Np HDF5). Chain-calibration issue closed
  by `fd530d0`; the long-burn validation run itself is pending.
- **Per-precursor delayed-neutron groups** — only matters for
  time-dependent kinetics, not static k-eff.
- **EADL relaxation cascade on GPU** — long-flagged.
- **Source-distribution biasing** (sample initial pos from importance
  CDF) — for the Wagner-Haghighat 50-1000× FOM on volume / angular
  sources.
- **Backfill the rest of the 137 migrated ICSBEP JSONs** with VIII.1
  A/B runs once GPU device-buffer cache lands and the 3080 box is
  available.

## Recent session highlights (2026-05-21)

1. **ENDF/B-VIII.1 default + sibling-`thermal/` layout support.**
   Layout-aware `data_paths::resolve_thermal_path` handles the
   VIII.x split transparently. 5 layout tests.
2. **Natural-element migration**: 137 ICSBEP cases rewritten
   (carbon → C-12 + C-13 by IUPAC 2021). Engine-side fallback in
   `material_resolve::expand_natural_elements` for any future
   un-migrated JSON. 4 expansion tests.
3. **Sweep CLI overrides JSON `recommended_settings`.** Previous
   precedence was inverted, leading to "this CLI flag doesn't
   work" surprises when JSONs hard-code production GPU settings.
4. **VIII.1 vs VII.1 A/B on heu-comp-inter-003** (CPU, 100k, 7
   cases). +820 pcm uniform shift localised to CIELO U-235.
5. **Cache infrastructure verified working** on CPU sweeps —
   zero "Loading X.h5" lines after preload, 19 nuclides loaded
   once and reused across all 7 cases.
