# Project status — 2026-06-11

What `origin/main` (`dc943e3`) ships today, what's open, what's the
current headline.

> **Orchestration-rollback note (merged to `main` via PR #10):**
> The in-process benchmark pipeline (`c05d678` Phase 1 scaffold,
> `8171705` Phases 2–5, `5c0844f` data-dir / wrapper, `2308edd`
> VRAM-aware n_slots) has been rolled back.
> Symptom that motivated the rollback: the `force_rebuild()`
> watchdog inside `stage4_gpu_executor` partially tore down GPU
> state while channel-held bundles still referenced it, corrupting
> the next case (NaN k_eff on case 1 of a 200k-particle 375-case
> sweep). Subprocess-per-case Python driver (`icsbep_sweep.py`)
> restored; per-case process death now isolates failures. The
> survival-biasing kernel (`db6b547`) and the MT=91 Law 61 mu
> coupling (`f1797d0`, etc.) are preserved.

## Headline

- **Tests**: 440 / 440 lib tests green on default features;
  447 / 447 with `--features cuda`. `cargo check` clean on
  `--all-targets` (default + cuda). The cherry-picked physics
  (MT=91 Law-61 + survival-biasing parity tests) is included in
  these counts.
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
| Lib test count (default) | — | **440 / 440 green** | `cargo test --lib` |
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

## Hardware-specific notes

The same engine source has been exercised on three machines that map
to different points on the saturation curve:

| host | GPU | VRAM | CPU | role |
|---|---|---|---|---|
| MSI-Laptop | RTX A1000 | 4 GB | 14p / 20l Intel | dev box; CPU sweeps |
| MSI-Home | RTX 3080 | 10 GB | 8p / 8l Ryzen | GPU production sweeps |
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

## Session 2026-06-11 — HMF-058 Be hot-bias: instrumentation + refill-normalization fix

Triggered by the partial B200 GPU sweep flagging beryllium-reflected
HEU-MET-FAST cases (058 / 066 / 009) ~+400–850 pcm hot vs LANL
MCNP-VIII.1 (Table LIX). Systematic isolation on `metal_stats_diag`
(`heu-met-fast-058_case-1`, A1000) eliminated four suspects and split
the bias into two independent problems.

**What shipped (built green, CUDA NVRTC-validated on A1000):**
1. **Explicit (n,2n)/(n,3n) tally, CPU + GPU.** `mt` tag on
   `CollisionOutcome::Multiplicity`; `n_n2n` / `n_n3n` / `n_nxn_out` +
   ⟨E_out⟩±σ through `ParticleResult` → `BatchResult` (CPU) and device
   atomics in `gr_multi_event` → `RecursiveTransportBatch` (GPU). The
   reconciliation residual `coll−(el+inel+fis+cap)` is **useless in
   S(α,β) systems** — it's swamped by thermal scatter (33/src), so the
   real (n,2n) rate needed explicit counters.
2. **Real GPU thermal-scatter counter** — `gr_elastic_event` counts
   S(α,β) into a new `cnt_thermal` (split out of `cnt_elastic` so the
   GPU/CPU elastic-vs-thermal buckets line up). Replaces the hardcoded
   `thermal_scatters: 0` in `dispatch.rs`.
3. **Refill rate-normalization fix.** `metal_stats_diag` divided
   per-source rates by nominal `N` (active_batches × particles), not
   the real `N + refilled` histories — inflating every rate by the
   refill factor (measured: n2n ratio 3.49 ≈ factor 3.48). Surfaced
   `total_histories` (`RecursiveTransportBatch` → `BatchResult
   .source_histories`); report() now divides by `Σ source_histories`.
   Verified: under refill=3.48, GPU n2n/src went from 0.257 → 0.0739
   (matches CPU 0.0737, Δ +0.19%). **k_eff was always immune** (its
   own denominator already used N+refilled).
4. **Sweep-faithful + per-case OpenMC** in `metal_stats_diag`:
   `refill=<f>` / `auto_refill` / `p=<N>` knobs and `openmc=<path>`
   (was hardcoded to the Godiva tally JSON — grading HMF-058 against
   Godiva's k). `openmc_scene_runner.py` now emits the
   `tallies_seed_mean` schema the diag reads (leakage on vacuum
   surfaces, MT4/MT91, fine fission, (n,2n)/(n,3n)).

**Findings — four suspects eliminated:**
- **(n,2n) rate**: CPU 0.07373 vs GPU 0.07387 /real-history — agree to
  0.19%. Not a rate bug. (GPU banks the secondary into the fission
  source, CPU transports in-generation, but the *rate* is identical.)
- **Under-convergence**: ruled out. CPU k *rises* with inactive batches
  (i=20 → +533, i=60 → +682 pcm) — drifts away from LANL (+138), not
  toward it. The CPU is converged and genuinely hot.
- **SVD rank truncation**: ruled out. CPU k is **bit-identical at rank
  15 and rank 60** (1.00533 ± 0.00105 both) — reconstruction saturates
  by rank 15 for these nuclides. Recorded in `CLAUDE.md`.
- **Refill mechanism**: not the lever. At fixed 20k particles,
  no-refill (+49) vs refill=3.48 (+68) differ ~19 pcm (within noise).

**Two independent problems remain:**
- **(A) Shared CPU+GPU Be bias** ~+400 pcm vs LANL — real, converged,
  not (n,2n)/convergence/SVD. Prime suspects: ⁹Be S(α,β) thermal
  sampling and/or ⁹Be(n,xn) secondary kinematics vs MCNP. Next step is
  the OpenMC-on-this-scene comparison (the fixed `openmc_scene_runner`
  produces a matching reference).
- **(B) GPU population/clustering instability** — GPU k drifts with
  raw particle count even with no refill (5k +222 → 20k +49) and spans
  +49 → +851 pcm (~10× statistics) across configs. Classic neutron
  clustering in a weakly-absorbing high-dominance-ratio Be reflector
  (cf. burn-up-instability literature). The B200 +851 is one draw on
  that unstable distribution, not a reproducible bias. A persistent
  ~8% GPU deficit in thermal/collision/fission tallies (n2n unaffected)
  points at long-thermal-history loss (event-cap / undersampling).

**Open follow-ups:** A/B-flag the (n,2n) bank-as-fission vs
transport-in-generation routing (deprioritized — CPU does it correctly
and is still hot, so it's not the shared bias); OpenMC-on-HMF-058 to
pin the shared Be physics; GPU population control / more inactive
batches for (B).

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
