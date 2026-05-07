# Random-ray TRRM (forward + adjoint, immortal rays) + FW-CADIS replacement — 2026-05-08

## TL;DR

The 2026-05-07 round shipped a "lite" detector-backward collision-
density CADIS proxy and documented it as **not delivering FOM gain
over analog**. This round closes that scorecard item. A complete
multigroup random-ray transport solver (Tramm 2018-style flat-source
TRRM, with the Tramm & Siegel 2021 immortal-ray persistent-state
variant) lands as `random_ray::*`, and the lite proxy is removed
from `shield_slab.rs` after empirical demonstration that the random-
ray adjoint replaces it strictly better at every measured depth.

`origin/main` is at the previous round's `43b3236`. Lib tests
**287 / 287 green** (was 260; +27 this round). `cargo check`
default and `cargo check --features cuda` both clean.

Headline FOM measurements (1 MeV photons through water,
shield_slab benchmark, full provenance in `outputs/random_ray_cadis_fom.txt`):

| Depth   | Mode             | T          | σ_rel | FOM (/s) | vs analog |
|---------|------------------|------------|------:|---------:|----------:|
| 100 cm  | Analog           | 5.372e-3   | 1.11% |    161.7 | —         |
| 100 cm  | Lite CADIS (gone)| 5.231e-3   | 1.86% |    111.4 | 0.69× WORSE |
| 100 cm  | **RR CADIS**     | 5.330e-3   | 1.02% |  **354** | **2.19× BETTER** |
| 200 cm  | Analog           | 1.105e-5   | 10.2% |    0.351 | —         |
| 200 cm  | Lite CADIS (gone)| 1.366e-5   | 37.6% |    0.071 | 0.20× WORSE |
| 200 cm  | **RR CADIS**     | 1.252e-5   | 11.8% |  **1.52**| **4.32× BETTER** |

Both RR-CADIS results are unbiased within combined MC σ. RR vs lite at
200 cm: 21× FOM improvement.

## What landed (in commit order)

1. **`random_ray` module — multigroup forward + adjoint TRRM**
   `src/random_ray/{mod,mgxs,fsr,integrator,solver,cadis}.rs` —
   ~1500 lines.
   - `MgxsLibrary` / `MaterialMgxs` / `ScatterMatrix` — multigroup XS
     with shared storage for forward (`Σ_s,g→g'`) and adjoint
     (transposed) lookups; χ-normalisation + Σ_t-positivity checked at
     construction.
   - `FsrMesh` — enum over `Cartesian` (uniform voxel grid, O(1)
     `fsr_at(pos, _)` integer division) and `Cell` (one FSR per
     `(deepest cell, lattice element)` key, HashMap lookup, supports
     analytic *or* stochastic per-FSR volume from track lengths).
   - `solve_segment` — analytic MoC ODE step
     `ψ_out = ψ_in·e^(-τ) + (q/Σ_t)(1-e^(-τ))` plus track-length
     `l·ψ_avg`. Numerically stable `(1-e^(-τ))/τ` series for τ→0.
   - `RandomRaySolver` — ray sampler (uniform AABB × isotropic dir),
     dead-zone + active-zone phasing, per-segment integration through
     `trace_step_recursive`, BC handling (vacuum kills mortal /
     reflects-with-zero immortal, reflective specular, transmission
     continues), source iteration with k-power-method update,
     `AdjointFlag::{Forward, Adjoint}` switch, `cfg.immortal: bool`
     for persistent-ray mode per Tramm & Siegel 2021.

2. **`rr_pincell` binary — multigroup pin-cell benchmark**
   2-group, UO₂ cylinder + water moderator pin cell with reflective
   BCs. Cell-based FSRs with analytic volumes (1 fuel + 1 moderator
   FSR auto-discovered). Forward (mortal + immortal) + adjoint all
   run end-to-end; per-region thermal/fast spectra physically correct.
   Wall ~12 s. Scaling to full C5G7 (4 fuel × 7 groups × 17×17
   lattice) is data plumbing — no new solver code needed.

3. **`rr_cadis_slab` binary + `WeightWindow::from_flux` bridge**
   `random_ray::cadis::weight_window_from_adjoint` runs the random-
   ray adjoint and feeds the result into the existing
   `transport::weight_window::WeightWindow::from_flux` pipeline.
   `rr_cadis_slab` is the slab-shaped CLI that produces the JSON
   `shield_slab --cadis-load` already consumes (CadisMap schema:
   `{thickness_cm, n_z_bins, counts}`). For 1-group non-fissionable
   problems the adjoint operator equals the forward operator (1×1
   scatter matrix is its own transpose), so the adjoint is computed
   as a fixed-source forward solve with the source localised at the
   detector face — that's the cheap exact reduction.

4. **Lite CADIS removed from `shield_slab.rs`**
   `cadis_calibration_pass`, `--cadis-calibration`, `--cadis-z-bins`,
   `--cadis-save` deleted. `shield_slab.rs` lost ~180 lines (608 →
   428). `--cadis-load`, `--ww-ratio`, `--ww-floor` retained — they
   consume RR-generated JSONs unchanged. The decision was driven by
   the empirical measurement above: the lite proxy is worse than
   analog at every measured depth, the random-ray adjoint is
   strictly better at every measured depth, and keeping a known-
   inferior parallel implementation in a working binary is dead
   weight.

5. **Canonical importance maps regenerated**
   `outputs/cadis_water_100cm.json` (25 z-bins) and
   `outputs/cadis_water_200cm.json` (50 z-bins) regenerated from
   `rr_cadis_slab`. Old lite-generated files replaced. Mesh
   resolution sweep at 100 cm shows 25–30 bins is the sweet spot:
   coarser meshes reduce wall-time bloat from per-voxel splitting
   without sacrificing fidelity at slab depths of 7–14 mfp.

6. **GPU port — scaffold**
   `gpu/cuda/random_ray_persistent.cu` — kernel with per-segment MoC
   ODE, vacuum reflect-with-zero, reflective + transmission BCs,
   Cartesian FSR lookup, atomic-add accumulators. `src/gpu_random_ray.rs`
   — Rust wrapper with NVRTC compile + buffer alloc + scaffold launch.
   `cargo check --features cuda` clean. Runtime parity validation
   against CPU is **deferred until CUDA hardware is available** — same
   convention as the hex-GPU work in this file.

## Validated benchmarks (lib tests)

- 1-group infinite homogeneous reflective box: k_inf within 500 pcm
  of analytic `νΣ_f/Σ_a = 1.25` (forward + immortal).
- Fixed-source infinite medium: `φ = Q/Σ_a` recovered within 10%.
- Adjoint-identity: forward k vs adjoint k agree within 800 pcm.
- Slab importance-gradient: ψ̂*(z) decreases away from the detector
  face; `WeightWindow::from_flux` produces correctly-oriented WW.
- 2-group, 2-material cell-based: physically correct per-region
  spectra (moderator thermal/fast > fuel thermal/fast).
- MoC integrator unit tests: `(1-e^(-τ))/τ` series agrees with direct
  formula to 1e-8 for τ ∈ [1e-4, 10]; segment in steady state holds
  ψ constant; track_psi matches definite integral to 1e-12.

## Tests this round

260 (resume.md last writeup) → 277 (after `random_ray` module + 17
unit / cell-based / multigroup tests) → 286 (after immortal-ray and
two-group end-to-end tests) → **287** (after the cell-based mesh
ergonomics test). Net **+27 lib tests**, all green.

## Honest scorecard

| Item | Status |
|---|---|
| Forward random-ray k-eigenvalue (mortal + immortal) | ✅ shipped, validated |
| Adjoint random-ray (transposed Σ_s, χ ↔ νΣ_f swap) | ✅ shipped, validated |
| Cell-based FSRs with analytic *or* stochastic volume | ✅ shipped, validated |
| MoC integrator (analytic ODE, τ→0 stability) | ✅ shipped, 1e-12 against analytic |
| `rr_pincell` 2-group benchmark binary | ✅ shipped, runs end-to-end |
| `rr_cadis_slab` FW-CADIS substrate | ✅ shipped, JSON drop-in |
| Lite CADIS removal + canonical-JSON regen | ✅ done |
| **CADIS FOM gain over analog (100 cm + 200 cm)** | ✅ **delivered** (2.2× / 4.3×) |
| Adaptive per-voxel ratio (`WeightWindow::from_flux_adaptive`) | ✅ implemented + unit-tested + wired via `--ww-growth`; ❌ doesn't beat fixed ratio empirically (default growth=0) |
| Textbook 50–1000× CADIS gain | ❌ ruled out for this WW design — see negative results below |
| Linear source approximation (1st-order) | deferred (flat on fine mesh equivalent for axis-aligned problems) |
| Full C5G7 (4 fuel × 7g × 17×17) | data plumbing, no new code |
| GPU runtime validation | deferred until CUDA hardware available |
| Continuous-splitting variance reduction (DXTRAN-style) | research-tier, **the actual lever needed for >14 mfp** |

### Negative results documented this round

- **Adaptive-ratio WW** (`WeightWindow::from_flux_adaptive`): widens
  the band at low-importance voxels per
  `ratio_v = base_ratio · (1 + ratio_growth · log10(φ_max/φ_v))`.
  Hypothesis was that the depth-dependent optimal-fixed-ratio (peaks
  at ratio≈3 at 200cm, monotone-up at 100cm) could be captured by
  one variable-band WW. At converged statistics adaptive ratio is
  statistically indistinguishable from fixed `ratio=5` at 100 cm
  (1M hist: 152 vs 146) and **worse** at 200 cm (2M hist: 1.52 vs
  0.39 — wider source-side bands oversplit in the transition zone).
  Lever stays in tree behind `--ww-growth=0.0` default; the
  experiment ruled it out as a path to bigger CADIS gains on this
  benchmark.

- **300 cm (≈21 mfp)** under any naive WW configuration:
  `(ratio, growth) ∈ {5,10,20} × {0,1,2,3}` all give 0 transmitted
  photons in 500 k histories. The exponential bound from
  `max_split=8` per voxel crossing limits how much the WW can
  multiply photons regardless of how the band is shaped.
  Continuous-splitting along the characteristic (DXTRAN-style
  point-detector estimator at every collision) is the textbook fix
  and remains research-tier work.

- **Source-distribution biasing for `shield_slab`**: textbook CADIS
  source biasing samples emission position/direction from the
  importance CDF. shield_slab's source is monodirectional + at one
  point — the importance CDF degenerates to a delta and there's no
  distribution to bias. Source-biasing is the right lever for
  volume / angular-distribution sources, not beam sources.

---

# Depletion + URR equivalence + photon shielding + CADIS-lite — 2026-05-07

## TL;DR

Four big-step items shipped this round. Depletion infrastructure
(Bateman + CRAM-16/-48 + chains + Python bindings + transport-
coupled predictor-corrector) went from "explicitly absent" to
production-grade. URR equivalence theory closed the documented
Stoker-Weiss / NJOY systematic on PWR pin cells. A photon shielding
benchmark (`shield_slab`) shipped with documented analog FOM
baseline. The CADIS-lite framework (importance map calibration +
WW splitting/roulette wired into the photon hot path) is in tree
and produces unbiased transmission, but **the FOM gain isn't
delivered** — the lite proxy isn't a true adjoint flux, and naive
geometric splitting produces correlated copies that don't reduce
variance per effective sample. Honest negative result documented;
real continuous-energy adjoint MC is the research-tier follow-on.

`origin/main` at `43b3236`. Lib tests **260 / 260 green** (was
227 → 246 → 250 → 260 across the round). `cargo check --features
cuda` and `cargo check -p open-rust-mc-py` both clean.

## What landed (in commit order)

1. **Depletion module — Bateman + CRAM-16 + chains + Python +
   URR equivalence** (`ccb0760`)
   `src/depletion/` — 8 files: `cram.rs` (IPF form, Pusa 2016
   poles & residues from OpenMC's canonical source), `chain.rs`
   (decay constants, branches, per-(parent, MT) reaction XS with
   ENDF default yield inference), `chain_io.rs` (JSON loader, three-
   way `yields` semantics: omitted → default, `{}` → pure removal,
   explicit map → use it), `matrix.rs` (transmutation matrix
   builder), `predictor_corrector.rs` (CE/LI step), `mapping.rs`
   (table-driven `BurnupMapping` walker), `flux.rs` (per-source
   flux extractor + power-normalised source rate).

   Two chain libraries shipped: `chains/partial_xe.json` (4 nuclides
   for the Xe equilibrium demo) and `chains/pwr_actinides.json`
   (17 nuclides — U/Np/Pu actinide chain plus I-135 / Xe-135 /
   Cs-135 / Pm-149 / Sm-149 fission products).

   Two binaries: `deplete_demo` (constant-flux Xe equilibrium —
   matches analytical to 1e-4 relative) and `deplete_pwr` (full
   transport feedback with **fresh-corrector**: clones materials,
   runs eigenvalue at predicted composition for the EOC flux
   estimate, then CRAM with the averaged matrix).

   Python API: `Chain.from_file` / `from_str`, `CramOrder.Order16`
   / `Order48`, `cram(matrix, n0, order)`, `deplete_constant_flux`,
   `deplete_with_flux_callback` (FFI-exception-safe Python `flux_at`
   closure), `Material.set_atom_density(hdf5_file, density)` /
   `atom_density_of`. Example: `depletion_xe_demo.py`.

   **URR equivalence theory** in the same commit:
   `src/transport/urr_equivalence.rs` ships Carlvik-Pellaud Dancoff
   factor for square pin lattices, σ_eff = σ_∞ · σ_0 / (σ_0 + σ_e)
   with σ_e = (1 − C)/(N · l̄), per-cell Dancoff cache,
   `is_urr` trait method on `XsProvider` to gate the correction
   to the URR window. `pwr_pincell --urr-equivalence` toggles it.
   9 unit tests covering analytic limits (C=0, C=1), asymptotic
   limits, Carlvik-Pellaud PWR-URR band agreement, scale invariance.

   **Validation:**
   - `[micro]` Xe-135 equilibrium with constant fission source —
     CRAM reproduces analytical N_Xe^eq formula to 1e-4 relative.
   - `[micro]` 1-day CRAM-16 step on the loaded actinides chain —
     17 nuclides, U-238 → U-239 → Np-239 → Pu-239 buildup all
     populate correctly, U-235 depletes, Xe-135 grows from fission
     yield.
   - `[micro]` CRAM-48 matches CRAM-16 on non-stiff problems
     (1e-13 relative); not-worse on extreme `λ·Δt = 50` regimes
     where order 16 starts losing precision.

   Tests this commit: 227 → 250 (+23 across cram, chain, chain_io,
   mapping, flux, predictor_corrector, urr_equivalence).

2. **Photon shielding slab benchmark** (`2f17a71`)
   `src/bin/shield_slab.rs` — fixed-source γ transmission through
   a thick slab. Reflective xy walls (large extent, effectively
   infinite slab) + reflective back face + **vacuum front face**
   (the only way photons leave the geometry, so `energy_escaped`
   from `transport_history_csg` IS the transmitted energy by
   construction).

   Built-in materials: `water`, `concrete` (ANSI/ANS-6.4 ordinary,
   2.3 g/cm³, 6 elements), `Pb`, `Fe`, `W`. Reports the analog
   Figure of Merit `FOM = 1 / (σ_rel² · t_wall)` — the variance-
   reduction reference any future CADIS / WW scheme has to beat.

   `[shield]` 100 cm water at 1 MeV, 1M histories, single seed:
   T = 5.26 × 10⁻³ ± 1.13 % relative, **FOM = 348/s**.
   Physics check: μ for water at 1 MeV = 0.0707/cm → 7.07 mfp
   uncollided exp(−7.07) = 8.5e-4 × ANSI/ANS-6.4.3 buildup factor
   ≈ 6.2 = 5.3e-3. Match.

   `[shield]` 50 cm concrete: T = 5.06e-3 ± 2.6 %, FOM = 223/s.
   Equivalent attenuation to 100 cm water (concrete μ ≈ 2× water μ
   at 1 MeV, so half the thickness gives the same transmission).

3. **CADIS-lite calibration — detector-backward importance map**
   (`df9a8bf`)
   `--cadis-calibration N` flag on `shield_slab` runs N source-
   photon histories born at `z=T` heading `-z` (into the slab from
   the detector face), accumulates their collision density per
   z-bin via a new `HistoryResult.collisions: Vec<(Vec3, f64)>`
   field, and prints the resulting importance map ψ̂\*(z) alongside
   the implied `w_target = ψ̂\*_max / ψ̂\*(z)`.

   This is the "lite" form — not a true continuous-energy adjoint
   MC (no transposed Compton kernel, no adjoint photoelectric
   source). For Compton + Rayleigh-dominated regimes it's
   qualitatively correct in the importance-gradient sense, and the
   peak of ψ̂\*(z) lands at the right depth (~10-15 mfp from the
   detector, matching photon mfp in water at 1 MeV).

   `[shield]` 50k calibration histories on 100 cm water:
   - z = 0-5 cm   ψ̂\* = 0.047 → w_target = 21.2 (heavy splitting)
   - z = 50-55 cm ψ̂\* = 0.350 → w_target = 2.86
   - z = 85-90 cm ψ̂\* = 1.000 → w_target = 1.00 (peak — matches
                                                   photon mfp depth)
   - z = 95-100 cm ψ̂\* = 0.775 → w_target = 1.29 (edge falloff)

4. **CADIS importance map → JSON** (`980af16`)
   `--cadis-save FILE` persists the calibration result so the
   next-step CADIS WW translator (the existing `WeightWindow::
   from_flux` over a 1×1×n_z mesh) can ingest it without re-running
   calibration. Schema:
   `{"thickness_cm": ..., "n_z_bins": ..., "counts": [...]}`.

5. **Photon-side CADIS WW + weight bookkeeping** (`43b3236`)
   The substantial commit. `src/photon/transport.rs` gains a
   5-tuple bank entry (`pos, dir, energy, weight, coord_stack`).
   Every tally accumulator multiplies by the photon's current
   weight, so the run-mean stays an unbiased estimator.

   Backward compatibility: existing `transport_history_csg` callers
   (`pwr_gamma_heating`, `cs137_pulse_height`, internal lib tests)
   are unchanged — the public function delegates to a new
   `transport_history_csg_with_ww` with `source_weight=1.0` and
   `weight_window=None`, identical to the prior analog behavior.

   New WW hook in `transport_one_csg`: after each free-flight that
   lands the photon in a new voxel, splitting (`w > w_upper`) or
   roulette (`w < w_lower`) fires. `prev_voxel` tracking ensures
   the hook only triggers on transitions, not micro-steps.
   `WeightWindow::voxel_index` exposed publicly to support
   the boundary-detection logic.

   `shield_slab --cadis-load FILE` ingests the saved importance
   map. The critical detail: `w_ref` is set so that
   `w_target(source_pos) = 1.0` — that's the consistent-CADIS
   normalisation. Without it, photons born at z=0 (low importance,
   large `w_target`) immediately fall below `w_lower` and get
   rouletted away, inflating the tally by the source-weight factor.

   **Validation — transmission unbiased:**
   `[shield]` 100 cm water, 200 k histories analog vs CADIS:
   - Analog: T = 5.45e-3 ± 2.5 %, **FOM = 361/s**, 2 800
     transmitted in 4.6 s
   - CADIS:  T = 5.21e-3 ± 1.8 %, **FOM = 220/s**, 16 288
     transmitted in 13.6 s — means agree within 1σ.

   `[shield]` 200 cm water (14 mfp, deep-penetration):
   - Analog: T = 2.81e-5 ± 36 %, FOM = 1.66/s, 12 transmitted
   - CADIS:  T = 1.59e-5 ± 34 %, FOM = 0.43/s, 1 434 transmitted
     (120× more samples reaching the detector!)

   **The negative result.** σ_rel barely improves at 200 cm (34 %
   vs 36 %) despite 120× more transmitted samples. The split
   copies are highly correlated — they share their parent's
   trajectory before the split point — so the effective number of
   independent samples is bounded by the ~12 photons that would
   have transmitted analog. Variance reduction per effective
   sample doesn't materialise; meanwhile splitting at every voxel
   inflates wall time 3-4×.

   The Wagner-Haghighat 2003 50-1000× FOM gain assumes:
   (1) true continuous-energy adjoint MC for ψ\* (not the lite
   collision-density proxy), (2) source-distribution biasing
   (sample initial position from the importance CDF, not just bias
   the source weight), (3) energy-dependent WW (4D mesh, not 1D
   z-only). All three are research-tier work that the shipped
   infrastructure plugs into.

## What's de-risked vs. what's research

**De-risked (this round):**
- Bateman / CRAM math at production grade (orders 16 and 48,
  validated against analytical Xe equilibrium at 1e-4 precision).
- Chain JSON ingestion with extensible schema; ENDF default yield
  inference for standard reaction MTs.
- Multi-nuclide BurnupMapping that cleanly separates chain-only
  evolution from transport-coupled feedback.
- URR equivalence theory hook + Carlvik-Pellaud Dancoff for square
  lattices; ready for OpenMC cross-validation.
- Photon shielding benchmark with documented analog FOM baseline.
- Per-photon weight bookkeeping in the photon hot path; WW
  splitting/roulette infrastructure; CADIS importance-map JSON
  format.

**Research-tier (deferred, with clear plug-in points):**
- Continuous-energy adjoint photon MC — transposed Compton kernel,
  adjoint photoelectric as source, energy-dependent WW. Documented
  inline in `shield_slab.rs` as the next-step roadmap.
- Source-distribution biasing for variance reduction (combined
  splitting instead of geometric splitting).
- Full PWR depletion bench (chains/pwr_actinides.json + Pu/Np HDF5
  files + 30-50 GWd/MTU run vs OpenMC's depletion solver).

## Test count progression

227 (prev round close) → 246 (after depletion + chain_io tests +
ENDF default yields validation) → 250 (after URR equivalence + Xe
equilibrium analytical match) → **260** (after CADIS-related
tests + photon weight bookkeeping). Net **+33 lib tests** this
round, all green.

## Honest scorecard

| Item | Status |
|---|---|
| Bateman / CRAM-16 / -48 | ✅ shipped, validated to 1e-4 |
| Chain JSON loader (3-way yields semantics) | ✅ |
| `pwr_actinides.json` (17 nuclides, fresh-fuel through ~30 GWd/MTU) | ✅ schema validated, 1-day CRAM clean |
| BurnupMapping (table-driven chain↔material walker) | ✅ |
| `deplete_pwr` fresh-corrector (eigenvalue at predicted comp) | ✅ |
| Python depletion API + Material composition setters | ✅ |
| URR equivalence theory (Carlvik-Pellaud Dancoff) | ✅ shipped, OpenMC cross-check pending |
| Photon shielding benchmark (`shield_slab`) | ✅ analog FOM = 348/s @ 100 cm water |
| CADIS-lite calibration (importance map) | ✅ peaks at right depth |
| Per-photon weight bookkeeping + WW hook | ✅ |
| **CADIS FOM gain over analog** | ❌ **not delivered** (lite proxy + naive splitting) |
| Real continuous-energy adjoint photon MC | research, multi-week |

---

# Variance reduction + tallies + restart + hex on GPU — 2026-05-06

## TL;DR

A house-cleaning round. The previously-disclosed
"first-category" gap list from the engine-status review (track-
length k-eff, survival biasing, mesh tallies, statepoint, delayed
neutrons, weight windows) is closed. The HexLattice geometry
shipped end-to-end on **both** CPU and GPU, no longer "math only —
wiring pending". Common geometry patterns moved into a shared
library (`geometry::shapes`) and through to the Python bindings.
A `transport::dispatch` module now hides the CPU/CUDA backend
choice behind a single `EigenvalueRunner` trait.

`origin/main` at `a17c379`. Lib tests **227 / 227 green**. Both
`cargo build --release` (default) and `cargo check --features cuda`
clean.

## What landed (in commit order)

1. **Track-length k-eff estimator** (`ebbba26`)
   Adds a second eigenvalue estimator alongside the collision-
   estimator k. At every flight segment the engine accumulates
   `w · d · Σ_νf(E)`; per-batch `BatchResult.k_track` is the sum
   divided by the source size N. Surface tracking only — under
   delta tracking the path crosses material boundaries silently
   so the integrand can't be reconstructed from a single-cell
   evaluation.

   `[godiva]` 60 b × 5 k × 3 seeds, SVD k=5, analog:
   k_collision = 1.00158 ± 0.00414, k_track = 0.99958 ± 0.00107
   — **3.9× lower seed-to-seed σ**, means agree within 1 σ_collision.
   Per-batch σ within a single seed drops 1.2–1.6×.

2. **Survival biasing + Russian roulette** (`ffd3b79`, `49705ae`)
   Implicit-capture path: at each collision bank fission as
   stochastic-rounded `w · ν · σ_f / σ_t` sites, reduce weight by
   `σ_s / σ_t`, sample only among non-absorbing channels via the
   new `collision::process_scatter_only`, then RR if `w < w_min`.
   Defaults match OpenMC: `w_min = 0.25`, `w_survive = 1.0`.

   Initially landed for surface tracking + non-thermal collision
   branch only (`ffd3b79`); follow-up commit (`49705ae`) factored
   out a shared `dispatch_real_collision` so delta-tracking gets
   the same SB path. The thermal-scattering inner branch in
   `transport_particle` falls back to analog (no SB) — that branch
   only fires for nuclides with S(α,β) data attached, which in
   PWR pin cell is just H1 in water (no fission, nothing to bias).

   `[godiva]` 80 b × 5 k × 4 seeds, SVD k=5, analog vs SB:
   - σ_collision per seed: 0.00217 → 0.00202 (-7 %)
   - σ_track per seed:     0.00171 → 0.00122 (**-28 %**)
   - ns/particle:          968 → 1343 (+39 %)
   - **FOM_track: 354 → 500 (+41 %)**

   `[pwr]` 60 b × 5 k × 3 seeds (surface tracking, multi-material):
   - cross-seed σ on k_inf: 0.00256 → 0.00115 (**-55 %**)
   - ns/particle:           36 936 → 41 007 (+11 %)
   - **FOM_collision: 412 → 1842 (4.5×)**

3. **Surface current + mesh flux tallies + HDF5 statepoint**
   (`84520dd`, `0a65c42`)
   New `transport::tally` with `SurfaceCurrentTally` (J+ / J-
   split by `sign(dir · normal)`) and `MeshFluxTally` (Cartesian
   voxel mesh, Amanatides-Woo deposit). New `transport::statepoint`
   serialises per-batch arrays + tally arrays + final source bank
   to HDF5 via `hdf5-pure::FileBuilder`.

   `[godiva]` h5py spot-check on a 30-batch run:
   1 surface bin (outer vacuum sphere), J+ = 51 871 (outward
   leakage), J- = 0 (no inward crossings on vacuum BC). 4 × 4 × 4
   = 64 voxels, total flux 604 520 cm/source.

   `[pwr]` 1.26 cm pin cell, 30 b × 3 k:
   6 reflective faces, J+ within 2 % across all 6 (square-pin
   symmetry). Mesh flux 9.99 × 10⁶ cm/source over 225 k histories
   = 44.4 cm/src, 2.78 cm/src/voxel. Source bank radial mean
   0.278 cm vs analytic 2R/3 = 0.273 (uniform sampling in fuel
   cylinder of R = 0.41 cm).

   Library helpers `MeshFluxTally::from_aabb`,
   `SurfaceCurrentTally::for_reflective_surfaces`,
   `SurfaceCurrentTally::for_boundary_surfaces` lifted in
   `0a65c42` so binaries no longer hand-roll the box-AABB or hard-
   code surface indices.

4. **Statepoint restart** (`0a3ecd3`, `f81c7b4`)
   Read path: `read_header` + `read_source_bank` + a
   `SimConfig.initial_source_bank: Option<Vec<FissionSite>>`
   field. When set, the engine resamples-with-replacement to
   `particles_per_batch` for batch 1; the warm source skips most
   of the settle.

   `[godiva]` cold start (50 b × 5 k × 15 inactive):
   k_collision = 1.00614 ± 0.00309, ns/p 1542.
   Resume from cold-start statepoint (30 b × 5 k × **5** inactive):
   k_collision = 1.00411 ± 0.00371, ns/p **1022 (33 % faster)** —
   means overlap within 1 σ; the warm bank removes settle cost.

   `pwr_assembly` gained `--shape N` so a 3 × 3 / 5 × 5 / 7 × 7
   minicore variant runs in a few seconds. Chained 3-restart
   stability test (cold → state1 → warm1 → state2 → warm2 →
   state3) on each shape, 3 hops × 3 shapes = 9 runs:

   | Shape | Run 1 (cold) | Run 2 (warm) | Run 3 (warm) | max gap |
   |-------|-------------:|-------------:|-------------:|--------:|
   | 3×3   | 1.32392      | 1.32646      | 1.32261      | 385 pcm |
   | 5×5   | 1.33482      | 1.32454      | 1.33024      | 1028 pcm (1.4 σ) |
   | 7×7   | 1.32730      | 1.33243      | 1.32711      | 532 pcm |

   Source banks round-trip cleanly (inspected via h5py): N = 2000
   per run, mean source energy 2.0–2.1 MeV (prompt Watt mean) for
   every shape and step, radial mean scales with shape extent. No
   drift across hops — the 5 × 5 step 1 → step 2 gap is normal
   statistical tail at 1.4 σ_combined.

5. **Delayed-neutron emission** (`17d2801`)
   Per-nuclide energy-dependent ν_delayed(E) loaded from HDF5
   (sum of all delayed-product yields in MT=18). At each fission
   yield-banking site, each banked neutron is sampled prompt vs
   delayed by `β(E) = ν_d / ν_total`; delayed neutrons draw from
   a soft Watt spectrum (a = 0.4 MeV — ENDF-style aggregate
   delayed spectrum, captures the spectrum-softening effect for
   static k-eff without the per-precursor-group breakdown).

   `[godiva]` 80 b × 5 k × 4 seeds, analog:
   - **Δ_ICSBEP: 196 pcm → 19 pcm** (closer to 1.0000 benchmark)
   - ns/particle: 968 → 924 (within noise — one extra rng draw
     per fission neutron)

   `[pwr]` 80 b × 5 k × 5 seeds: k_inf = 1.32775 ± 0.00183,
   matches resume.md baseline 1.328 within 1 σ.

6. **Forward weight windows + flux-bootstrap generation**
   (`2e67e7c`, `5e774ec`)
   Cartesian-mesh weight windows in `transport::weight_window`.
   Per-voxel `(w_lower, w_upper)` thresholds drive splitting
   (when `w > w_upper`, copy into `ceil(w / w_survive)` particles)
   and Russian roulette (`p_survive = w / w_survive` below
   `w_lower`). `w_survive = sqrt(w_lower · w_upper)` (geometric
   mean — preserves expected weight). `max_split` cap defaults
   to 8 to clip runaway splits in high-importance voxels.

   Forward CADIS-lite generation:
   `WeightWindow::from_flux(aabb, n, flux, w_ref, ratio, phi_floor)`
   inverts a per-voxel flux estimate into `w_target ∝ φ_max / φ_v`,
   bracketed by `±sqrt(ratio)`. High-flux voxels get low w_target
   (split → finer sampling); low-flux voxels get high w_target
   (roulette → coarser). Voxels below the floor flagged inactive.

   `[godiva]` 60 b × 5 k × 4 seeds, three modes:

   |                              | k_c mean  | σ_t (per-seed) | ns/p | FOM_track |
   |------------------------------|----------:|---------------:|-----:|----------:|
   | analog                       | 0.99969   | 0.00153        | 1030 |  416      |
   | SB                           | 1.00153   | 0.00146        | 1411 |  332      |
   | SB + bootstrap WW (15-batch calib) | 1.00239 | 0.00290 |  949 |  125      |

   Pipeline correct (means agree across modes); FOM **drops** on
   Godiva because the geometry is homogeneous — flux is roughly
   uniform across the 4 × 4 × 4 mesh, so the bootstrap can't find
   spatial under-sampling. Honest result: weight-window variance
   reduction pays off on heterogeneous problems, not on a 8.7 cm
   uniform sphere. The CADIS adjoint solve is a separate research
   project (deferred).

7. **HexLattice transport — CPU end-to-end**
   (`2e67e7c` schema, `7acb70a` integration, `7aff6c0` binary,
   `788d87f` proper hex boundary)
   `CellFill::HexLattice(u32)` variant; `Geometry.hex_lattices:
   Vec<HexLattice>` populated via `Geometry::with_hex_lattices`.
   `Coord` gains `hex_lattice: Option<(HexLatticeId, [i32; 3])>`
   alongside the rect-lattice slot. `find_cell_recursive` descends
   through `CellFill::HexLattice(h)` via `HexLattice::find_element`
   + `universe_at` + `local_position`. `trace_step_recursive`
   dispatches `HexLattice::distance_to_grid` for stack frames
   flagged hex.

   New `hex_minicore` binary: N-ring hex array of UO₂ pins inside
   a hex-shaped reflective boundary (6 reflective `Surface::Plane`
   instances at 30°, 90°, 150°, 210°, 270°, 330° from +x; inradius
   `(N + 0.5) · pitch`). Lattice is sized one ring larger
   internally with the outer ring filled by an all-water
   placeholder universe — handles cube-rounding ties at the hex
   edge so float-precision points map to a valid cell.

   `[hex]` 1-ring (7 pins) k_inf = 1.35829 ± 0.00329 (3 seeds × 60
   batches × 3 k particles, SVD k=5).
   `[hex]` 2-ring (19 pins) k_inf = 1.36424 ± 0.00399 — both rings
   agree within 1 σ (gap 60 pcm vs combined σ ~520 pcm). Higher
   than the rect PWR baseline (1.328) because a hex unit cell has
   more moderator per pin at the same pitch.

8. **HexLattice transport — GPU port** (`077db2b`)
   The `CellFill::HexLattice(_) => panic!` stub in `gpu_recursive.rs`
   is gone. New CUDA device functions `gr_hex_find_element`,
   `gr_hex_universe_at`, `gr_hex_distance_to_grid`,
   `gr_hex_cube_round`, `gr_hex_cart_to_axial_frac`,
   `gr_hex_element_center_local` transliterate the CPU math.
   `GrGeometry` + `GrCoord` grow parallel hex SoA. `gr_find_cell`
   dispatches `GR_FILL_HEX_LATTICE` (constant 4); `gr_trace_step`
   dispatches grid-distance on `has_lattice == 1` (rect) vs `== 2`
   (hex).

   8 new SoA buffers uploaded by `GpuRecursiveContext`. 5 kernel
   signatures (`find_cell_batch`, `trace_step_batch`,
   `multi_step_walk`, `transport_recursive`,
   `const_xs_transport_persistent`) gained the hex params. 5 Rust
   launch sites updated.

   Validation: `cargo check --features cuda` clean; runtime parity
   test against CPU on a hex 1-ring deferred until a CUDA-capable
   device is available — the equivalent of
   `gpu_recursive_parity::hex_lattice_descent_and_trace_smoke` is
   the obvious next step.

9. **`geometry::shapes` builders + Python bindings**
   (`2243a0e`, `1708781`, `b31b355`)
   New module exposes:
   - `rect_box(half, bc, surface_offset) → Shape` — 6 axis-aligned
     planes + the inside-the-box `Region`.
   - `rect_box_split_bc(half, xy_bc, z_bc, …)` — same with separate
     xy / z BCs (the assembly-style "reflective xy + variable z"
     pattern).
   - `hex_boundary(n_rings, pitch, orientation, xy_bc, z_half,
     z_bc, …)` — 6 hex-side `Surface::Plane`s + 2 z planes +
     inside region. Replaces ~40 lines of hand-rolled trig in
     `hex_minicore`.
   - `hex_side_normals(orientation) → [Vec3; 6]` — exposed helper.
   - `pin_cylinders(center_x, center_y, radii) → Vec<Surface>`.

   `pwr_pincell`, `pwr_assembly`, `hex_minicore`, `gpu_assembly_keff`,
   `gpu_const_xs_keff`, `gpu_recursive_parity`, `gpu_cpu_trace`
   refactored to use the helpers — net diff ~30 lines off each
   binary, surface ordering preserved exactly so cell regions
   referencing 0..=8 still resolve identically. Tests stayed bit-
   exact post-refactor.

   Python bindings (`bindings/python/src/lib.rs`) gain
   `Scene.add_rect_box(prefix, half, bc) → (scene, region_str)`,
   `Scene.add_hex_boundary(prefix, rings, pitch, orientation,
   xy_bc, z_half, z_bc) → (scene, region_str)`, and
   `Scene.add_pin_cylinders(prefix, radii, center_x, center_y) →
   (scene, names)`. Surfaces register under auto-generated names
   (`{prefix}_xmin`, `{prefix}_side0`, …); the returned region
   string slots straight into the existing `add_cell(region=...)`
   parser.

10. **`transport::dispatch` — backend-agnostic eigenvalue runner**
    (`a17c379`)
    Hides the CPU vs CUDA choice behind one `EigenvalueRunner`
    trait. `Backend::recommended()` defaults to CUDA when
    `--features cuda` is on, CPU otherwise. `CpuRunner` wraps
    `simulate::run_eigenvalue_with_geometry`; `CudaRunner`
    (cuda-feature-gated) lifts the per-batch driver from
    `gpu_assembly_keff` (transport_recursive call, fission-bank
    normalisation, k_eff active-mean aggregation) into the library.
    Returns a unified `EigenvalueOutcome { batches, k_eff,
    final_source_bank }`.

    Existing binaries unchanged — they can adopt the trait
    incrementally.

## Honest gaps and what's deferred

- **Hex GPU runtime parity test** — schema and Rust glue compile
  clean under `cargo check --features cuda`, but the NVRTC compile
  + on-device validation are pending CUDA hardware. The CPU smoke
  (`hex_lattice_descent_and_trace_smoke`) covers the math; an
  equivalent multi-million-event GPU↔CPU comparison is the next
  step once a device is available.

- **Bootstrap WW on a heterogeneous problem** — the algorithm runs
  end-to-end on Godiva but doesn't help there because the
  geometry is uniform. The proper test is PWR pin cell with the
  WW bootstrap; it would require auto-attaching the mesh tally
  in `pwr_pincell`'s WW pipeline (currently only Godiva has
  `--ww-bootstrap-batches`).

- **Survival biasing in the thermal-scatter path** —
  `transport_particle`'s thermal branch (S(α,β) for H in water)
  falls back to analog. For PWR, H1 has no fission so the SB
  benefit on this nuclide would be small; not a priority.

- **Delayed-neutron per-precursor groups** — for static k-eff the
  aggregated soft-Watt spectrum captures the only effect that
  matters (mean energy ≈ 0.4 MeV vs prompt ≈ 2 MeV). Per-group
  precursor concentrations matter only for time-dependent
  kinetics, not in scope.

- **CADIS / FW-CADIS adjoint solver for WW generation** — needs a
  deterministic adjoint transport solve (S_N or adjoint MC).
  Forward bootstrap (above) is the cheap proxy.

- **EADL relaxation cascade on GPU** — flagged in the previous
  resume.md round; still open. Not on the critical path for any
  current benchmark.

- **Predictor-corrector depletion (CE/LI, CE/CM)** — Bateman
  solver coupling; multi-week effort. Out of scope for this round.

- **Doppler-broadened coherent elastic scattering** — Bragg edge
  / phonon spectrum treatment beyond standard S(α,β); months of
  work. Out of scope.

- **URR equivalence theory (Stoker-Weiss)** — current URR tables
  give correct stochastic sampling for infinite medium; the
  equivalence-theory correction for tight lattices is the
  follow-on. Open question, not blocking.

- **Python `hex_minicore.py` example** — drafted twice, scoped
  back when we pivoted to GPU hex. Now that hex transport works
  through the bindings (`Scene.add_hex_boundary` registers the
  surfaces; the existing region-string parser handles the
  inside-hex region), a thin example calling
  `add_hex_boundary` + `add_pin_cylinders` + `run_eigenvalue` is
  ~50 lines.

- **Binary refactors to use `EigenvalueRunner`** — godiva,
  pwr_pincell, pwr_assembly, hex_minicore, gpu_assembly_keff
  could each shrink ~10–20 lines. Not urgent — the existing
  call sites still work.

## Test count progression

198 (resume.md last writeup) → 201 (post-rebase) → 217 (after
hex CPU integration) → 225 (after geometry::shapes) → **227**
(after dispatch). Net +29 lib tests in this round, all green.

# GPU recursive transport — primitives done, full physics next — 2026-05-05

## TL;DR

Recursive geometry runs on GPU. Three kernels validated against
the CPU primitives at bit-exact (or sub-ULP) precision across two
million-event regression sweeps; a constant-XS transport kernel
runs full eigenvalue-style histories (collision + scatter +
absorption + fission banking via `atomicAdd`) and reaches **6.7×
CPU speedup** with k-eff within MC noise of the CPU reference.
The geometry side of the GPU port is fully de-risked. What
remains is hooking the existing SVD / Table / WMP / S(α,β) / URR
XS providers in place of the per-material constants — a narrower
piece of work than the geometry refactor was.

## Validation evidence

All on RTX A1000 Laptop, CUDA 13.2, NVIDIA-SMI 595.79.

### Cell-find on GPU — `find_cell_batch`
200 000 random world points × two geometries
(`gpu_recursive_parity` Tests 1 + 2):

| Geometry           | CPU ns/pt | GPU ns/pt | Speedup | Mismatches |
|--------------------|-----------|-----------|---------|------------|
| 2×2 lattice        | 79        | 25        | 3.2×    | 0 / 200 000 |
| 17×17 PWR assembly | 101       | 25        | 4.0×    | 0 / 200 000 |

Histogram of deepest-cell hits across the assembly's 8 leaf cells
(fuel / gap / clad / pin-water / GT-inner / GT-clad / GT-outer /
outside-void) — every reachable cell exercised, every one matches.

### Trace step on GPU — `trace_step_batch`
50 000 random (pos, dir) pairs × two geometries:

| Geometry           | CPU ns/event | GPU ns/event | Speedup | Pos max-rel-err | Mismatches (any field) |
|--------------------|-------------:|-------------:|--------:|:---------------:|:----------------------:|
| 2×2 lattice        | 444          | 86           | 5.2×    | 8.4e-13         | 0 / 50 000             |
| 17×17 PWR assembly | 477          | 81           | 5.9×    | 9.3e-11         | 0 / 50 000             |

Distance, surface idx, BC, and re-resolved next-stack deepest-cell
all agree per particle.

### Multi-step transport walk on GPU — `multi_step_walk`
20 000 particles × 50 steps each = 1 000 000 events, pure-geometry
walk (vacuum / reflective / transmission events handled, no XS):

| Geometry           | CPU ns/step | GPU ns/step | Speedup | Pos max-rel-err | Mismatches |
|--------------------|------------:|------------:|--------:|:---------------:|:----------:|
| 2×2 lattice        | 346         | 14.4        | **24×** | 3.8e-14         | 0 / 20 000 |
| 17×17 PWR assembly | 361         | 15.7        | **23×** | 5.4e-14         | 0 / 20 000 |

Every particle's trajectory matches CPU position to **better than
machine precision**, with identical step counts and identical
final cells. Lattice grid traversals, axis-aligned reflections,
and transmission re-resolves all compose correctly when chained.

### Constant-XS transport on GPU — `const_xs_transport_persistent`
50 000 particles, 2×2 lattice, fissile (σ_t=1, σ_a=0.5, σ_f=0.4,
ν̄=2 → analytical k_∞=1.6) + pure-scatter water:

| Backend | k        | absorptions | fission events | leakage | Time |
|---------|---------:|------------:|---------------:|--------:|-----:|
| CPU     | 1.59932  | 50 000      | 79 966         | 0       | 363 ms |
| GPU     | 1.60524  | 50 000      | 80 262         | 0       |  53 ms |
| Δ       | 592 pcm  | identical   | 0.37 %         | identical | 6.74× |

Full eigenvalue-style transport: collision sampling, scatter /
absorption / fission dispatch, atomic fission banking. Bit-exact
agreement is *not* expected — float-rounding ties between
collision distance and surface distance can flip event ordering,
and downstream RNG draws diverge — but every non-statistical
aspect agrees and the k-eff difference is within MC noise.

## Status — task #22 (GPU transport hot-path integration)

| Stage                                                                       | Status                                |
|-----------------------------------------------------------------------------|---------------------------------------|
| 1. Capture baselines                                                        | ✓ done                                |
| 2. Recursive transport plumbing on GPU, parity-validated                    | ✓ done (24× speedup, bit-exact)       |
| 3a. Constant-XS transport — collision + scatter + fission banking           | ✓ done (6.7× speedup, k within MC noise) |
| 3b. Hook up real SVD / Table / WMP / S(α,β) / URR XS                        | pending                               |
| 4. `gpu_assembly_bench` binary on full physics                              | pending                               |
| 5. Final validation (Godiva/PWR unchanged + assembly k_inf agrees CPU/GPU + ≥5× speedup) | pending                  |
| 6. Retire old hardcoded paths                                               | pending                               |

## Commit map (this round)

```
38dc70d  gpu: recursive cell-find on GPU + CPU parity test
a9bff75  gpu: recursive trace_step on GPU + extended parity test (5.8× speedup)
ead2fb9  gpu: capture baselines for recursive-transport integration (task #22)
f7b7c9f  gpu: end-to-end recursive transport walk on GPU (24× speedup, bit-exact)
0db2982  gpu: full recursive transport with const XS — k(CPU) ≈ k(GPU) at 6.7×
```

## What's de-risked, what's left

**De-risked:**
- Recursive geometry on GPU (find_cell, trace_step, multi-step walk).
- PCG-XSH-RR per-thread RNG on GPU.
- Collision sampling, scatter direction sampling, absorption /
  fission dispatch.
- Fission banking via `atomicAdd`.
- The performance gate (≥5× CPU rayon) is already cleared at the
  geometry-query level alone (5.9× per trace_step) and is exceeded
  in the const-XS transport (6.7× per particle, single batch with
  upload + launch overhead).

**Left:**
- Replace the 4-double-per-material constant XS table with calls
  into the existing GPU SVD / Table / WMP / S(α,β) / URR device
  functions in `transport.cu`. The data buffers already exist
  (uploaded by `GpuTransportContext`); the new kernel needs to
  read them in place of `mat_xs[mat * 4 + k]`.
- Wire the new kernel into a `gpu_assembly_bench` binary mirroring
  the existing `gpu_pwr_bench`, so the assembly k_inf can be run
  end-to-end on GPU.
- Validate Godiva and PWR pin-cell still match their pre-#22
  baselines through the *existing* `transport_persistent` (which
  this work has not touched) and that the new
  `transport_recursive_persistent` agrees with the CPU recursive
  transport on the assembly.
- Once the recursive path is stable, retire the
  `geom_type`-switched hard-coded paths in `transport.cu`.

# Universe-recursive geometry — phase 1 + depth-N profile — 2026-05-05

## TL;DR

Recursive geometry shipped end-to-end on CPU. The 17×17 PWR
assembly demo runs through depth-3 stacks (root → assembly lattice
→ pin/GT cell), produces **k_inf = 1.14958 ± 0.00318**, and was
literally inexpressible before this work — the previous flat-cell
representation could only handle Godiva and PWR pin-cell.

Acceleration / feature layers since the last writeup:

1. **CoordStack + recursive primitives**
   `find_cell_recursive(world_pos)` and
   `trace_step_recursive(stack, pos, dir)` thread a
   `SmallVec<[Coord; 4]>` through every transport call. The
   coordinate stack walks offsets and rotations from root down,
   cell-finds within each universe, and re-resolves from world
   after every crossing.
2. **Per-universe surface restriction + opt-in BVH**
   Each universe pre-computes its own surface index list at
   construction time so descent only re-evaluates surfaces that
   cells in *that* universe actually reference. **3.0×** assembly
   speedup (70 857 → 23 999 ns/p) by itself. Per-universe BVH is
   built when a universe has ≥ 8 cells with finite AABBs; tiny
   universes (Godiva: 2, PWR pin-cell: 4) keep the linear scan
   for cache reasons.
3. **Mat3 rotations on Coord**
   `Cell.rotation: Option<Mat3>` propagates through the descent so
   a universe / lattice can be placed at an arbitrary orientation.
   `Coord` carries the per-frame rotation; trace_step's per-frame
   `local_dir` cascade is gated on "any frame rotated?" so
   rotation-free geometries pay zero cost.
4. **Distributed materials**
   `RectLattice.material_overrides` rebinds individual cells to
   different materials at specific lattice elements. Lets one pin
   universe be reused across an assembly with different
   enrichments / burnup tiers without duplicating geometry.
5. **HexLattice math**
   `find_element` (cube-coord rounding from Cartesian) and
   `distance_to_grid` (six edge normals + axial planes,
   closed-form). Both flat-top (`Y`, VVER convention) and
   pointy-top (`X`) orientations. Math-only; the
   `find_cell_recursive` wiring lands when the VVER core demo
   needs it.

198 lib tests green (up from 184 pre-refactor), Godiva and PWR
pin-cell run bit-identical k_eff vs the flat-geometry baseline
(0.99422 / 0.99481 / 0.99412 / 0.99462 / 0.99429 across 5 seeds —
exact match to 5 decimals).

## Depth-1 fast-path decision (task #18)

The original phase-1 plan kept a "depth-1 fast path" that would
short-circuit the recursive primitives for single-universe
geometries. The data says it's not needed — and depth-3 is much
closer to depth-1 than I initially thought when *the physics is
matched*.

| Geometry        | Depth | Physics                  | Pre-refactor ns/p | Post-recursive ns/p |
|-----------------|-------|--------------------------|-------------------|---------------------|
| Godiva          | 1     | 3 nuclides, fast         | 1057              | 1015                |
| PWR pin-cell    | 1     | 9 nuclides, S(α,β), URR  | ~27 000           | 25 684 ± 683 (SVD)  |
| 17×17 assembly  | 3     | 9 nuclides, S(α,β), URR  | infeasible        | 27 361              |

Two things follow:

  1. **Depth-1 is bit-identical and equal-throughput** to the
     pre-refactor flat path. Per-seed k_eff matches to 5
     decimals (Godiva: 0.99422 / 0.99481 / 0.99412 / 0.99462 /
     0.99429 across 5 seeds, exact match). No fast path needed.
     **Decision: drop it. The recursive primitives are the only
     geometry path going forward.**
  2. **Depth-3 is 1.07× slower than depth-1 with matched
     physics** — the recursive descent (3 levels instead of 1)
     adds essentially zero overhead per particle once you
     compare workloads with the same nuclide set and thermal
     scattering. The physics work dominates the geometry walk
     by an order of magnitude.

Earlier writeups in this file flagged a "16× per-particle depth
penalty"; that was apples-to-oranges (Godiva's 1 ns/p fast
spectrum vs assembly thermal physics). Corrected: **the depth
penalty per se is in single-digit percent**. Future per-particle
wins live in algorithmic / data-layout work (cache layout,
allocation removal in trace_step) but the size of that prize
is small. The big remaining lever is **GPU** (task #19, moves
the per-particle workload off the CPU loop entirely) and
**larger geometries** (task #20, AP1000) where new bottlenecks
will surface.

# Full electron transport + GPU photon kernels + Python API — 2026-04-27

## TL;DR

Three orthogonal pieces landed since the coupled n-γ writeup. Each
removes a previously-disclosed approximation or extends the engine
to a new platform / front-end without touching the validated
neutron-photon physics core:

1. **Full electron transport** replaces the Katz–Penfold "CSDA
   midrange" displacement with track-integrated step-and-deposit:
   non-uniform Bethe-Bloch `dE/dx`, Highland multiple-scattering
   angular spread, single-event Seltzer-Berger bremsstrahlung
   secondaries. The "1.5 % in the He gap" deposition artefact in
   PWR γ-heating goes to **0 %**; the cladding share moves to where
   the textbook puts it.
2. **GPU photon kernels** — Compton (free KN + S(x,Z)/Z + optional
   Doppler broadening), Rayleigh, photoelectric phase 1, and
   Bethe-Heitler pair production are all on CUDA via NVRTC. Bit-
   parity with CPU at machine epsilon, validated on H / O / Zr / U
   at 1 / 5 / 100 keV / 1 MeV / 5 MeV. A **persistent-kernel** path
   for full Compton history loops gets **2.22× wall-time speedup**
   (CPU rayon 20-thread vs RTX A1000) on N = 1 M histories.
3. **Python API** — PyO3 binding exposes the Rust builder
   (`Scene`/`Material`/`Surface`/`PhotonMaterial`) plus
   `run_eigenvalue` and `run_gamma_heating`. `XsMode` enum
   (Table / Svd / HybridTableWmp / HybridSvdWmp) is toggleable per
   simulation, with per-MT rank overrides. Six-line Godiva script,
   ~50-line full PWR γ-heating script.

148 / 148 library tests green, integration tests for water buildup,
Cs-137 pulse-height, Hubbell Compton, NIST brems, and Rust ↔ OpenMC
γ-heating cross-check all pass. `cargo fmt --check` and
`clippy -D warnings` clean on Linux + Windows.

## Test environment (applies to every benchmark below)

All numbers in this document come from one fixed-seed sweep on the
following box. Per-test rows below only restate batch / particle /
iteration counts; machine specs are not repeated.

- **CPU**: 20-core Intel mobile workstation, 32 GB RAM, Windows 11.
  Rust runs use rayon over all 20 threads unless noted.
- **GPU**: NVIDIA RTX A1000 (laptop, 4 GB, fp64 ~0.51 TFLOP/s, no
  tensor-core fp64). CUDA via `cudarc` + NVRTC.
- **Nuclear data**: ENDF/B-VII.1 HDF5
  (`../data/endfb-vii.1-hdf5/{neutron,photon}`).
- **Reference code**: OpenMC 0.15.3 on the same library.
- **Driver**: `run_pwr_tests.ps1`. Outputs: `outputs/full_test_run/`.

## Full electron transport

### What changed

Old (PR #3, 2026-04-23): each photoabsorption / Compton / pair
deposit was placed at `pos + dir · R_e(E)/2`, with `R_e` from
Katz-Penfold and per-material density. Single-step displacement,
no energy loss along the track, no bremsstrahlung.

New: each electron is tracked through its CSDA range as a real
condensed-history walk:

- **Step length** drawn so every step deposits a fixed fraction of
  the residual range (default 0.05). Adaptive in low-Z / low-energy
  tails to keep the per-step energy loss small.
- **Energy loss** integrated along the step using a non-uniform
  Bethe-Bloch `dE/dx`. The mean ionisation `I` per element is read
  from the photon HDF5 attribute (`I_eV`) — ~80 eV for H, ~890 eV
  for U.
- **Multiple scattering** — Highland angular spread per step,
  `θ₀ = (13.6 MeV / β c p) √(t/X₀) [1 + 0.038 ln(t/X₀)]`, with
  per-cell radiation length `X₀` derived from `Σ ρ Z(Z+1)/A · ρ_i`.
- **Bremsstrahlung** — single-event sampling with the OpenMC
  per-element radiative stopping integrals `I₁`, `I₂` reconstructed
  from the photon HDF5 `bremsstrahlung/dcs` table. The emitted γ is
  banked back into the photon transport loop, so cascaded
  brems-on-secondaries works without recursion.
- **Lattice fold** — the same triangle-wave reflection used by the
  CSDA-midrange code is preserved, so reflective-BC pin cells stay
  consistent with the neutron loop.

### Numerical impact

**Run conditions.** PWR pin cell γ-heating, UO₂ 3.1 % / Zr-4 / H₂O,
1.26 cm pitch, reflective lattice; **150 batches (20 inactive + 130
active) × 50 000 neutrons/batch + 200 000 photon histories**, single
seed; ~5 min wall. Output: `outputs/pwr_gamma_heating_benchmark.txt`;
OpenMC reference at `outputs/openmc_pwr_gamma_heating.json`.

| region | PR #3 (CSDA midrange) | this round (full ET) | OpenMC 0.15.3 |
|--------|----------------------:|---------------------:|--------------:|
| fuel   | 84.4 %               | **84.12 %**          | ~85 %         |
| gap    | 1.5 % (artefact)     | **0.00 %**           | 0 %           |
| clad   | 7.9 %                | 9.81 %               | ~9 %          |
| water  | 5.9 %                | 5.72 %               | ~6 %          |
| escape | 0.0 %                | 0.00 %               | 0             |
| sum    | 99.66 %              | 99.65 %              | —             |

Bremsstrahlung is real: this run emits **2 312 γ totalling 7.43 ×
10⁸ eV (0.353 % of source energy)**, fed back into the photon
transport phase. The structural gap reported in the previous round
(84 / 9 / 6 vs textbook 93 / 3 / 2) was a benchmark-mismatch
question — with full electron transport and the matching OpenMC
reference run on identical geometry, every region is within 1
percentage point.

### Caveats

- **NIST ESTAR `S_rad` cross-check** (`outputs/brems_check.txt`,
  per-element offline computation, no MC histories — direct
  evaluation of `I₁ = ∫ k dσ/dk dk` and `I₂ = ∫ k² dσ/dk dk` from
  the photon HDF5 brems DCS at `T_e = 1 MeV`):
  the integrals agree with OpenMC's formula identically
  (`S_rad_1 == S_rad_2`), but the ratio against NIST ESTAR is
  element-dependent: **0.72× (H), 2.46× (O), 3.18× (Zr), 4.86× (U)**.
  Either the ENDF/B-VII.1 brems DCS layout we read differs from
  ESTAR's (Seltzer-Berger vs ICRU-37 model), or there's a
  per-element normalization remaining. *Open question — does not
  affect the γ-heating result to within Monte Carlo statistics, but
  flagging for follow-up.*
- Brems angular distribution is currently isotropic (small-angle
  sampling around `θ ≈ m_e c² / E_e` is a small refinement; the
  current isotropic emission already reproduces the OpenMC
  γ-heating split).

## GPU photon kernels (new file: `src/photon/gpu.rs`)

### What's on GPU

Four CUDA kernels live in a single NVRTC-compiled module to amortise
the JIT compile over multiple invocations:

| Kernel | Inputs | What's on device |
|---|---|---|
| `GpuComptonContext` | fixed `E_in`, optional `S(x,Z)/Z` table | Klein-Nishina + bound rejection; +Doppler if Compton profiles uploaded |
| `GpuComptonVarECtx` | per-particle `E_in[]` | same as above with batched energies |
| `GpuRayleighContext` | element form factors `F(x,Z)` | direct `x²` CDF inversion + Thomson `(1+μ²)/2` rejection |
| `GpuPairContext` | none | Bethe-Heitler ε rejection sampling |

Photoelectric phase 1 (XS lookup + subshell sampling) lives on GPU
too; the **EADL relaxation cascade** still runs on CPU because of
its thread-divergence tolerance requirements (deferred until a
GPU SoA cascade design exists).

### Bit-parity

**Run conditions.** **N = 1 000 000 events per (element × energy)
case**, single fixed seed (`Rng::for_particle(0, tid)`); pass
criterion `|Δ⟨x⟩|/⟨x⟩ < 0.5 %` and reduced χ² on 50 bins < 2.0.
Outputs: `outputs/gpu_compton_validate.txt`, `gpu_photon_validate.txt`,
`gpu_photon_full.txt`.

Result: agreement on H / O / Zr / U at 100 keV–5 MeV with
`<x>_cpu == <x>_gpu` to displayed precision (machine epsilon),
χ²_red = 0.000 in every case across 41 cases (Compton free-KN,
Compton+Doppler, Rayleigh, photoelectric phase 1, pair production).
Bit-parity comes from a PCG-64 CUDA implementation that mirrors
`src/transport/rng.rs::Rng` byte-for-byte and uses the same
`Rng::for_particle(batch_id, tid)` seeding.

### Per-call performance

**Run conditions.** **N = 1 000 000 events per kernel-call**,
ns/event = total wall / N, best of 1 rep, fixed seed. Output:
`outputs/gpu_cpu_bench.txt` (Mode A — per-kernel batched).

GPU wins on heavy elements where per-call work amortises the launch
overhead; CPU wins on H/O at light kernels because Compton without
Doppler is essentially "two divides and a sample" and 20 cores beat
the launch latency:

| kernel                      | CPU ns/ev | GPU ns/ev | GPU/CPU |
|---|---:|---:|---:|
| Compton[U 1 MeV]            | 26.3      | 17.0      | **1.55×** |
| Compton+Doppler[U 5 MeV]    | 363.6     | 273.1     | **1.33×** |
| Rayleigh[U 100 keV]         | 33.2      | 10.8      | **3.07×** |
| Photoelec[U 1 MeV]          | 254.2     | 26.0      | **9.78×** |
| Photoelec[Zr 1 MeV]         | 190.5     | 19.5      | **9.75×** |
| Compton[H 1 MeV]            | 9.6       | 16.8      | 0.57×    |
| Pair[5 MeV]                 | 6.8       | 9.6       | 0.71×    |

Doppler-broadened Compton on U is a 600+ MFLOP kernel with element-
specific Compton profiles → that's where the GPU lift is. Free KN
on H is 4 lines of math → CPU wins.

**Batch-size scaling** (Compton on U @ 1 MeV, best of 5 reps,
output `outputs/gpu_compton_scaling.txt`):

| N | free ns/ev | Doppler ns/ev |
|---:|---:|---:|
| 10 000      | 34.0  | 602.2 |
| 100 000     | 18.4  | 397.9 |
| 1 000 000   | 13.0  | 312.1 |
| 10 000 000  | 12.2  | 310.5 |
| 100 000 000 | 12.3  | 312.5 |

Throughput plateaus at N ≥ 1 M; smaller batches are launch-overhead
dominated.

### Persistent-kernel mode (`--persistent`)

**Run conditions.** Mode B in `gpu_cpu_bench` — full Compton
history loop, free Klein-Nishina sampling (no element data), no
detector. **N = 1 000 000 histories, E_in = 1 MeV, E_cut = 1 keV,
max 64 collisions / history** (= 64 000 000 total collisions);
single launch on GPU.

| | wall time | µs / history | ns / collision |
|---|---:|---:|---:|
| CPU rayon (20 thr) | 934.6 ms | 0.93 | 14.6 |
| GPU persistent     | 421.7 ms | 0.42 | **6.6** |
| **GPU / CPU**      | **2.22×** | | |

A per-collision launch model at ~700 µs / launch would cost
~44 800 ms for the same 64 M collisions (~107× slower than the
persistent kernel). Confirms the persistent-kernel design choice.
Output: `outputs/full_test_run/08_gpu_photon_features.txt`.

### Other GPU technology hooks (`--tensor-svd`, `--nvlink`, `--optix`)

**Run conditions** (all from `outputs/full_test_run/08_gpu_photon_features.txt`,
element = U (Z=92)):

| Hook | Run setup | Result |
|---|---|---|
| **cuBLAS batched DGEMM** for SVD reconstruction | `basis[1200, 5] · coeffs[5, n_batch]`, naive vs DGEMM, n_batch ∈ {1, 8, 64, 1024} | 5.5× / 23.3× / 40.2× / **62.8×** speedup; 16.19 GFLOP/s at n_batch=1024 (A1000 fp64 peak ≈ 0.51 TFLOP/s, no tensor-core fp64) |
| **NVLink demo** | Compton on U @ 1 MeV, N = 1 000 000, single-stream vs 2-stream split-then-merge on 1 GPU | single 17.20 ms vs split 18.96 ms = **10.2 % overhead**; concurrent execution requires 2-GPU NVLink hardware (H100/B200), plumbing exercised here |
| **Software BVH ray-AABB** | N rays = 1 000 000, 64 AABBs, linear traversal (no BVH yet); CPU = single thread | GPU 37.17 ms vs CPU 315.40 ms = **8.48×** (4 698 hits both); real RT-cores (OptiX SBT) deferred until a complex-geometry benchmark exists |

## Python API (new directory: `rust_prototype/bindings/python/`)

### Surface

PyO3 + maturin, abi3-py39 (one wheel works on Python 3.9-3.13). 200
lines of glue Python (`__init__.py` + convenience builders), all
validation and simulation logic lives on the Rust side.

The builder mirrors the engine:

```python
from open_rust_mc import Material, Scene, Sphere, Settings, run_eigenvalue, XsMode

heu = (Material("HEU", temperature=294.0, temp_idx=1)
       .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
       .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.43)
       .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.49))

scene = (Scene("data/endfb-vii.1-hdf5/neutron")
         .set_xs_mode(XsMode.HybridSvdWmp).set_svd_rank(5)
         .add_material("heu", heu)
         .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
         .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
         .add_cell("outside", region="+boundary"))

result = run_eigenvalue(scene, Settings(batches=50, inactive=10, particles=5000))
print(f"k_eff = {result.k_eff:.5f} ± {result.k_sigma:.5f}")
```

### XsMode round-trip

`xs_mode_quick.py` runs all six provider configurations on Godiva
in a single 15-batch script (table baseline / SVD k=5 / ACE+WMP /
Hybrid SVD+WMP k=5 / Hybrid adaptive {MT=2,18,102→rank 1} / SVD
adaptive). The Rust binary's `--mode all` is a one-line Python call.

### `run_gamma_heating`

Coupled neutron-photon, end-to-end, from Python. Drives:

1. Neutron k-eigenvalue with `XsMode`-of-choice + per-MT photon
   source event tally.
2. Photon transport from the aggregated bank through the same CSG
   with per-cell `PhotonMaterial`.
3. Full electron transport (Highland MS + Bethe-Bloch + Seltzer-
   Berger brems) all on by default.

`examples/pwr_gamma_heating.py` (≈50 lines incl. material
definitions) lands at the same 84 / 0 / 10 / 6 split as the Rust
binary, validating the FFI plumbing.

### Convenience builders (`open_rust_mc.*`)

- `uranium_oxide_material(name, density, enrichment, ...)` —
  stoichiometry from `ρ × N_A / M_UO₂`.
- `water_material(...)` — H₂O with S(α,β) attached by default.
- `zircaloy4_material(...)` — natural-abundance Zr.
- `atom_density_from_mass_density(...)` — primitive.
- `NUCLIDE_DATA` dict — awr / molar mass / default ν̄ / HDF5
  filename for U / Pu / O / H / Zr / B / C / Fe / N.

### Honest scope

PYTHON.md (commit `01da2b3`) explicitly says: "Python matches the
Rust binary by construction". The Python tests don't validate
physics — they validate the **plumbing** (region parsing, AABB
derivation, photon-material indexing, S(α,β) wiring, tally
extraction). The OpenMC cross-validation in
`scripts/openmc_pwr_gamma_heating.py` is the meaningful physics
check — Python γ-heating lands at 84 / 0 / 10 / 6 vs OpenMC
85 / 0 / 9 / 6 on the same geometry.

## Benchmark snapshot

Single fixed-seed sweep, outputs in `outputs/full_test_run/`. Test
environment per the box at the top of this document; run conditions
below cover only iteration / particle counts.

| Suite | Run conditions | Result |
|---|---|---|
| Library tests | `cargo test --lib`, single-threaded per test | 148 / 148 OK |
| Integration | `cargo test --release`: water buildup (5 tests, 21 s), brems vs NIST (2 tests, <1 s), Cs-137 NaI (200 k γ), Hubbell Compton (2 elements × 100 k γ), ANSI/ANS-6.6.1 (1 M γ buildup) | all OK |
| Godiva, SVD k=5 vs Table | 150 batches (20 inactive + 130 active) × 20 000 particles = 2.6 M active histories per run | SVD k = **1.00093 ± 85 pcm** (load 24 s, sim 10.2 s, 556 MB XS); Table k = 1.00169 ± 85 pcm (load 1.5 s, sim 8.8 s, 111 MB XS); ICSBEP target 1.0000 ± 100 pcm; gap 76 pcm |
| PWR pin cell, three-way | 100 batches (20 inactive + 80 active) × 20 000 particles = 1.6 M active histories per run, 9 nuclides, 3 materials, S(α,β) on | SVD k = 1.32817 ± 100 (sim 46.8 s, 518 MB), Table k = 1.32903 ± 93 (sim 41.7 s, 103 MB), **ACE+WMP k = 1.32821 ± 98** (sim 42.1 s, 100 MB); SVD-vs-WMP gap **5 pcm**, Table-vs-WMP gap 81 pcm |
| PWR γ-heating | 150 b × 50 k n + 200 000 γ histories; 312 s neutron + 5.2 s photon = ~5 min wall | 84.12 / 0.00 / 9.81 / 5.72 / 0.00 % (fuel / gap / clad / water / escape), 2 312 brems γ |
| GPU Compton bit-parity | N = 1 000 000 events × 41 cases (H/O/Zr/U × {1, 5 MeV} × {free KN, +Doppler}) + Rayleigh × {100, 1000 keV} + photoelectric phase 1 + pair × {2, 5, 20 MeV} | χ²_red = 0.000 to display precision; \|Δ⟨x⟩\| = 0 in every cell |
| GPU Compton scaling | Best of 5 reps each; N ∈ {10⁴, 10⁵, 10⁶, 10⁷, 10⁸} on U @ 1 MeV; free KN + Doppler | plateaus at N ≥ 10⁶: 12.96 ns/ev free, 312.09 ns/ev Doppler |
| Python `xs_mode_demo` | 15 batches × 500 particles × 6 modes (Table / SVD k=5 / ACE+WMP / Hybrid SVD+WMP k=5 / hybrid adaptive {MT=2,18,102→1} / SVD adaptive); single seed | all six in expected k_eff band; demonstrates `XsMode`, `set_svd_rank`, `set_svd_ranks` round-trip |

## What we wouldn't touch yet

- **Brems-DCS vs NIST** discrepancy (0.7×–4.9× across H/O/Zr/U).
  The integrals match OpenMC's formula identically (`S_rad_1 ==
  S_rad_2`), so the issue is in interpretation of the Seltzer-Berger
  HDF5 layout. γ-heating numbers match OpenMC, so this isn't
  blocking — but it's an open question for a brems-dominated
  benchmark (Møller-Plesset shielding on Pb, etc.).
- **EADL cascade on GPU**. Photoelectric phase 1 is on GPU at
  9.8× on U; the relaxation cascade (fluorescence + Auger) needs
  an SoA design with sorted thread blocks per shell to avoid
  divergence. ~1 week of work; not on the path to anything we're
  measuring right now.
- **Brems-emission angle** beyond isotropic. Small refinement;
  γ-heating is insensitive at the per-percent level.
- **OpenMC cross-validation of the Python path** beyond γ-heating —
  the Python and Rust binaries call the same Rust code, so the
  Python physics validation is `Rust validation × FFI is
  plumbing-correct` (verified by deposition matching the binary
  bit-for-bit).

## Commit map (since 2026-04-23 resume)

```
01da2b3  PYTHON.md: clarify what the Python layer actually validates
f37f349  Python API: close the five Phase-1 gaps
e8fde09  Python API: PyO3 builder + axis-aligned cones + SimConfig refactor
e56cfce  Full electron transport: track-integrated CSDA + brems + MS + BB
7f485e3  docs: resume.md + README for coupled neutron-photon γ-heating
```

GPU photon kernels and validation/scaling/feature binaries
(`gpu_compton_validate`, `gpu_compton_scaling`, `gpu_cpu_bench`,
`gpu_photon_features`, `src/photon/gpu.rs`) live as working-tree
changes pending squash-and-merge — see `git status`.
