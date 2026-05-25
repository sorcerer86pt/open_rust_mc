# Engine-vs-OpenMC bias investigation

**Status:** in progress — earlier per-channel XS hypothesis ruled out
by Task #20 audit; bias must live in inelastic-level *selection* or
kinematics, not σ_MT(E) magnitudes.
**Triggered by:** RTX 5090 ICSBEP sweep `results/icebsp_run_5090_full.txt`
showed +200–1300 pcm biases on ~20 of 115 cases.
**Started:** Ben Forget's email reply (mid-investigation) flagging
URR + Be (n,2n) as common bias suspects.

---

## TL;DR

1. **Engine bias on Godiva-class fast metal is real and ~+200 pcm.**
   Confirmed by OpenMC head-to-head, not library shift.
2. **Engine bias is case-dependent**, NOT a uniform offset.
3. **Cases with moderator (graphite, Be) reflectors match OpenMC
   perfectly** (within MC noise).
4. **Cases with fast-only spectrum (bare, lead reflector) carry
   +200–360 pcm engine bias.**
5. ~~**Source localised to σ_inelastic(E) reconstruction in U-235.**~~
   **Superseded by Task #20:** at the matched log-log interpolation
   the transport hot path uses, the SVD reconstruction agrees with
   the raw pointwise table to machine precision for every
   MT ∈ {4, 51..91} across [0.1, 10] MeV. The earlier "MT=91 is
   95 % low" reading came from comparing the non-interpolating
   `discrete_level_xs()` helper (SVD) against the interpolating
   `PointwiseTable::lookup()` (Table) — apples-to-oranges. The
   OpenMC head-to-head tally rates (MT=91 4.5 % low, MT=51..90
   3.9 % high, MT=4 within 0.2 %) are still real measured numbers;
   with σ_MT(E) confirmed correct, they have to come from inelastic
   level *selection* or kinematics, not XS magnitude.
6. **NOT** caused by: SVD rank, per-channel σ_MT reconstruction,
   Watt sampling, ν̄(E) interpolation, source convergence, (n,2n)
   Q value (fixed in `eb9b62c`), URR bin interpolation (fixed in
   `0aa9591`).

---

## Per-case OpenMC head-to-head (5,000 particles × 50 batches × 15 inactive × 1 seed)

| Case | Engine k_eff | OpenMC k_eff | Engine − OpenMC | Handbook Δ | Reflector |
|---|---|---|---|---|---|
| HMF-001 case-1 (Godiva) | 1.00164 ± 116 | 0.99975 ± 51  | **+189 pcm** | +164 | bare HEU |
| HMF-019 case-1 | 1.00826 ± 387 | 1.00827 ± 173 | **−1 pcm** ⭐  | +826 | graphite + S(α,β) |
| HMF-027 | 1.00685 ± 333 | 1.00322 ± 182 | **+363 pcm** | +685 | lead |
| HCI-003 case-2 | 1.01231 ± 121 | 1.01242 ± 351 | **+11 pcm** ⭐ | +1231 | Be + S(α,β) |
| HST-004 case-1 | 0.99473 ± 259 | 0.99590 ± 268 | **−117 pcm** | -527 | thermal soln |

⭐ = engine within MC noise of OpenMC. Handbook Δ is the
library-vs-experiment shift, NOT engine bias.

---

## Investigation timeline (chronological)

### 1. Initial hypothesis (wrong)

User's first read: "+200 pcm Godiva is SVD compression overhead at rank=15.
Bumping rank should close it."

**Verified — wrong direction.**

- Rank=15 vs rank=30 vs rank=50 on CPU produced **bit-identical k_eff
  (1.00164)** for Godiva.
- Same `policy_hash` → different cache files, NOT cache pollution.
  Confirmed by inspecting `/tmp/open_rust_mc_cache/*v4*`.
- Conclusion: SVD precision saturates around rank ~15 for U-234/235/238.
  Singular values 16+ are numerically negligible. **Rank doesn't help.**

GPU did improve slightly with rank (15→30: −30 pcm) due to flat-pack
basis path differences, but plateaued.

### 2. Pointwise vs SVD comparison

Used `godiva` binary with `--mode both` (same harness, both XS lookups):

| Mode | k_eff | σ |
|---|---|---|
| Pointwise (table) | 1.00404 | ±20 |
| SVD rank=15 | 1.00456 | ±22 |
| **SVD − Pointwise** | **+52 pcm** | — |

**SVD compression overhead is only +52 pcm**, not +200. The bulk of
the residual is somewhere else.

### 3. Watt χ sampling comparison

OpenMC's `watt_spectrum` (src/random_dist.cpp):

```cpp
double maxwell_spectrum(double T, uint64_t* seed) {
  double r1 = prn(seed); double r2 = prn(seed); double r3 = prn(seed);
  double c = std::cos(PI / 2. * r3);
  return -T * (std::log(r1) + std::log(r2) * c * c);
}
double watt_spectrum(double a, double b, uint64_t* seed) {
  double w = maxwell_spectrum(a, seed);
  return w + 0.25 * a * a * b
       + uniform_distribution(-1., 1., seed) * sqrt(a * a * b * w);
}
```

Our `WattLaw::sample` (hdf5_reader.rs):

```rust
let c = (FRAC_PI_2 * xi2).cos();
let w = -a * (xi1.ln() + c * c * xi3.ln());
let term = a * a * b / 4.0;
let e_out = w + term + (2.0 * xi4 - 1.0) * (a * a * b * w).sqrt();
```

**Mathematically identical** — Cranberg-Frankel decomposition. Cosmetic
RNG variable name swap (`xi2` vs `xi3` for the cosine slot). No physics
difference.

Plus: U-235's HDF5 actually carries a TABULATED fission spectrum, not
Watt. Path verified equivalent to OpenMC's `ContinuousTabular::sample`
(`src/distribution_energy.cpp`).

### 4. ν̄(E) interpolation check

Used `scripts/inspect_u235_nu.py` to read U-235.h5:

- `product_0` (prompt) yield: lin-lin (interpolation code 2), 95 points
- All 6 delayed groups: lin-lin
- Our `NuBarTable::lookup` is lin-lin (hdf5_reader.rs:638)

**Match.**

### 5. Initial source spectrum

Engine `try_initial_source` (simulate.rs:2554) had `energy: 1.0e6`
constant. OpenMC's `openmc.stats.Watt(0.988e6, 2.249e-6)` mean ~2 MeV.

**Tried fix:** sample Watt for initial source. **Result: WORSE**
(CPU 1.00164 → 1.00317, +150 pcm in wrong direction).

**Tested with 100 inactive batches** (vs 20): CPU 1.00164 → 1.00218,
within MC noise. **Source convergence is NOT the bottleneck.**

Reverted to constant 1 MeV.

### 6. MT=91 isolated tally

Added per-MT split to engine + OpenMC (commit pending):

- `CollisionOutcome::InelasticScatter { q_value_ev, mt: u32 }` carries MT
- `BatchResult.n_inelastic_continuum` counts MT=91 fires separately
- OpenMC tally `scores=["91"]` and `scores=["4"]` via numeric MT
- `metal_stats_diag` prints both sides

Godiva result:

| Channel | CPU rate | OpenMC rate | Ratio |
|---|---|---|---|
| MT=91 (continuum) | 0.2265 | 0.2371 | **95.5 %** |
| MT=51-90 (discrete) | 0.3065 | 0.2950 | **103.9 %** |
| MT=4 (total inel) | 0.5330 | 0.5321 | 100.2 % |
| Elastic | 1.7233 | 1.7118 | 100.7 % |
| Capture | 0.0456 | 0.0446 | 102.2 % |
| Fission | 0.3847 | 0.3847 | 100 % |
| (n,2n) | 0.0026 | 0.0026 | 100 % |

**Engine mis-routes ~4 % of inelastic events from MT=91 to MT=51-90.**
Total inelastic 2 % low → spectrum harder → k_eff higher (+200 pcm
direction matches).

### 7. Broad cross-case sweep

Ran OpenMC on 4 representative cases via
`scripts/openmc_scene_runner.py`. Pattern emerged in §"Per-case
head-to-head" table above.

**Cases with moderator reflector → engine matches OpenMC.**
**Cases without (bare HEU, lead) → +200–360 pcm engine bias.**

When ~50 % of fissions happen at thermal energies (after moderator
return), the fast-spectrum-only bias is diluted by 50 %. Math holds:
graphite case engine match (−1 pcm) ≈ 50 % × bare bias (+189 pcm) /
some-factor — qualitatively consistent.

---

## Root cause hypothesis

**Engine has a fast-spectrum σ_inelastic(E) reconstruction bias on
U-235** at E > 1 MeV. Manifests as:

- MT=91 rate 4.5 % low
- MT=51-90 rate 3.9 % high
- Total MT=4 rate 2 % low
- Spectrum slightly harder than OpenMC
- More fast fission → higher k_eff

This is consistent with SVD rank-15 reconstruction having different
precision per MT channel. The discrete-level kernels (MT=51-90) each
have their own SVD basis with different rank requirements; the
continuum (MT=91) has one basis. Cross-channel ratio drifts ~4 %.

**But raising rank does NOT help** — saturated at rank ~15. The error
floor is intrinsic to the SVD parametrisation, not the rank truncation.

---

## Fixes already in place (relevant to this investigation)

- **`1654c4d`** (May 12, 2026): per-level SVD basis padding fix
  closed +500–700 pcm fast-metal bias.
- **`0aa9591`** (May 18, 2026): URR factor interpolation between
  bracketing P-table rows.
- **`eb9b62c`** (this session, May 25): real ENDF Q-value for (n,2n)
  and (n,3n) in `gr_multi_event`. Confirmed by OpenMC on HCI-003
  (Be reflector) — engine now matches within 11 pcm.

These three closed the LARGE biases. The residual +200 pcm is a
smaller, harder-to-localise effect.

---

## Files modified / added during this investigation

### Code (uncommitted)

- `rust_prototype/src/physics/collision.rs`:
  `CollisionOutcome::InelasticScatter { q_value_ev, mt: u32 }` — MT
  propagated through to caller. All four fire sites updated.
- `rust_prototype/src/transport/simulate.rs`:
  `BatchResult.n_inelastic_continuum` field + accumulators in
  `ParticleResult`, `WorkerAccum`, fold/reduce. `dispatch_real_collision`
  increments when `mt == 91`.
- `rust_prototype/src/transport/dispatch.rs`: CudaRunner branch sets
  `n_inelastic_continuum: 0` (GPU split not wired yet).
- `rust_prototype/src/depletion/flux.rs`, `transport/statepoint.rs`:
  test fixtures updated for the new field.
- `rust_prototype/src/bin/metal_stats_diag.rs`:
  - `r=N` CLI arg for SVD rank override (default 15).
  - `Active.inel_cont_sum` accumulator + breakdown print.
  - OpenMC reference table prints `rate_MT4` and `rate_MT91`.

### Scripts (added)

- `scripts/inspect_u235_watt.py`: dumps fission distribution structure
  + interpolation codes for U-235.
- `scripts/inspect_u235_nu.py`: dumps ν̄(E) tables + interp codes for
  every fission product (prompt + 6 delayed).
- `scripts/openmc_godiva_tallies.py`: now reads `OPENMC_GODIVA_DATA`
  env var, defaults to VIII.1. Added MT=4 and MT=91 isolated tallies.
- `scripts/openmc_sweep_diag.sh`: runs `openmc_scene_runner.py` on a
  list of representative cases.
- `scripts/openmc_scene_runner.py`: extended with per-nuclide per-MT
  tallies (was only cell-level fission/absorption/scatter/elastic/
  (n,γ); now also (n,2n), (n,3n), MT=91 isolated, per-nuclide
  breakdown).

### OpenMC data (in `outputs/openmc_diag/`)

- `godiva_omc.json` — HMF-001 case-1 single-sphere geometry
- `hci003_c2_omc.json` — HCI-003 case-2 Be reflector
- `heu-met-fast-019_case-1_omc.json` — graphite reflector
- `heu-met-fast-027_omc.json` — lead reflector
- `heu-sol-therm-004_case-1_omc.json` — thermal solution

---

## Diagnostic methodology (reproducible)

### Run OpenMC reference on any case JSON

```bash
wsl -d Ubuntu-24.04 -- bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate openmc && cd /mnt/c/Users/fog/madman_svd_experiment && python scripts/openmc_scene_runner.py bench/icsbep/<CASE>.json outputs/openmc_diag/<CASE>_omc.json --particles 5000 --batches 50 --inactive 15 --seeds 1 --cross-sections data/endfb-viii.1-hdf5/cross_sections.xml'
```

Outputs JSON with `k_mean`, per-tally rates (rate_fission, rate_MT4,
rate_MT91, etc.), per-nuclide breakdown.

### Run engine on same case with matched stats

```bash
cd rust_prototype/bindings/python && python examples/icsbep_sweep.py \
  --runner cpu --filter <CASE> \
  --particles 5000 --batches 50 --inactive 15 --seeds 1 \
  --csv ../../../outputs/<CASE>_engine.csv
```

### Three-way CPU/GPU/OpenMC diff (Godiva-specific harness)

```bash
cd rust_prototype && cargo run --release --features cuda \
  --bin metal_stats_diag -- heu-met-fast-001_case-1 \
  b=100 i=20 p=20000 s=42 r=15
```

Requires `outputs/openmc_godiva_tallies.json` from
`scripts/openmc_godiva_tallies.py`.

---

## Task #20 — XS reconstruction audit results

New binary `src/bin/u235_inelastic_audit.rs` loads U-235 from one
HDF5 file twice — once through `load_nuclide` (rank-k SVD) and once
through `load_nuclide_table` (pointwise) — and dumps σ_MT(E) for
every MT ∈ {4, 51..91} on a 400-point log grid over [0.1, 10] MeV
plus 4 post-threshold bracket points per level. Each per-energy
lookup uses **the same `(idx, log_frac)` log-log interpolation the
transport hot path uses** (`ReactionKernel::reconstruct_interp` on
the SVD side, `StochTempTable::lookup_at_idx` on the Table side).

```powershell
cd rust_prototype && cargo build --release --bin u235_inelastic_audit
./target/release/u235_inelastic_audit.exe `
    --rank 15 --out ../outputs/u235_audit_r15_interp.csv
# also supports `--mt91-table` to force MT=91 to ReactionKernel::Table.
```

### Headline result — SVD per-channel σ matches pointwise

Decade profile of `rel_diff = (svd - table) / max(|svd|, |table|)`,
restricted to rows where both XS > 1e-10 b:

| decade (eV)         | MT=91 mean | MT=91 worst | MT=4 mean | MT=4 worst |
|---------------------|------------|-------------|-----------|------------|
| [5e5, 1e6)          | 0.0000     | 0.0000      | 0.0000    | 0.0000     |
| [1e6, 3e6)          | 0.0000     | 0.0000      | 0.0000    | 0.0000     |
| [3e6, 1e7)          | 0.0000     | 0.0000      | 0.0000    | 0.0000     |

Spot probes:

| E (eV)   | MT  | SVD σ_b      | Table σ_b    | rel_diff |
|----------|-----|--------------|--------------|----------|
| 6.5e5    | 91  | 4.3972e-3    | 4.3972e-3    | 0        |
| 6.5e5    | 4   | 1.6306       | 1.6306       | 0        |
| 1.0e6    | 91  | 5.0246e-1    | 5.0246e-1    | 0        |
| 2.0e6    | 91  | 1.6553       | 1.6553       | 0        |
| 7.0e6    | 91  | 9.2302e-1    | 9.2302e-1    | 0        |
| 7.0e6    | 4   | 1.1452       | 1.1452       | 0        |

**The +200 pcm Godiva bias is NOT in per-channel σ_MT(E)
reconstruction.** At rank 15, the SVD reconstruction of every
MT ∈ {4, 51..91} agrees with the raw pointwise table to machine
precision across the entire [0.1, 10] MeV range, **provided** both
paths are queried via matched log-log interpolation.

### Why earlier diagnostics looked like a smoking gun

`NuclideKernels::discrete_level_xs(E)` (the convenience method
this doc and earlier diagnostics relied on) calls the
non-interpolating `ReactionKernel::lookup(E)`, returning σ at the
nearest grid point ≤ E. `NuclideTableData::discrete_level_xs(E)`
calls `StochTempTable::lookup(E)` → `PointwiseTable::lookup(E)`,
which **does** log-log interpolation.

Comparing the two looks like SVD is 95 % low at E = 0.65 MeV
(SVD plateau at the grid point below, Table interpolated up the
ramp). Both are individually correct for their lookup mode, but
the transport hot path uses interpolation for both providers
(`SvdXsProvider::lookup` calls `reconstruct_interp`,
`TableXsProvider::lookup` calls `lookup_at_idx_with_pick`), so the
production engine sees the interpolated values that agree to
machine precision.

The discrete-level convenience method is fine for sampling
where you want σ at the grid point (no fractional interpolation),
but it is **not** a faithful proxy for transport-time σ. Earlier
sections of this doc that attributed bias to "SVD MT=91
mis-reconstruction" were based on the same flawed comparison.

### What this means for the +200 pcm bias

It is not the XS magnitude. The candidates that survive are:

1. **Inelastic level *selection*** — when an inelastic event fires
   in the engine, which level (MT=51..91) we pick is governed by
   the per-level XS *ratios* at that energy plus the angular /
   energy-distribution sampling that follows. The OpenMC head-to-
   head showed engine MT=91 rate 4.5 % low and MT=51..90 rate
   3.9 % high while total MT=4 rate is within 0.2 %. With σ_MT(E)
   agreement at machine precision (this audit), the imbalance has
   to live in the per-level sampling probability, the
   `sample_inelastic_level` walk, or the `InelasticCdf`.
2. **Angular distribution for the continuum (MT=91)** — if the
   continuum's outgoing angle/energy distribution is being
   misrouted to a discrete-level kinematics path, the rate
   accounting flips MT=51..90 vs MT=91 even with correct σ.
3. **Threshold handling at sampling time** — a discrete level
   accepting events below its true threshold (or MT=91 rejecting
   them just above 0.436 MeV) would shift the rate breakdown.

Concrete next step: dump CPU + OpenMC histograms of the per-level
firing probability at a fixed E (1, 2, 5 MeV) over a large sample
size, compare against the analytic ratio
`σ_MTn(E) / Σ_{k} σ_MTk(E)`. If the engine's empirical histogram
differs from that ratio, the bug is in `sample_inelastic_level`.

### Rank dependence

The audit produces bit-identical CSVs at `--rank 5`, `--rank 15`,
and `--rank 30`. With matched interpolation that's the expected
behaviour for a converged reconstruction — the SVD basis has more
than enough rank to track this XS faithfully at every probed E.

### `policy.table_mts` plumbing for discrete levels

The discrete-level loader at `xs_provider.rs:1505` previously
bypassed `policy.table_mts` (it called `build_kernel_from_reader`
directly with `svd_rank`). It now goes through
`build_kernel_with_policy`, so `RankPolicy::new(15).with_table(91)`
takes effect for the continuum (the `--mt91-table` audit flag
exercises this path). The interpolated XS comparison is identical
with or without this override at rank 15, but the knob is useful
for ablation studies and as a future-proofing for cases where the
SVD basis is genuinely insufficient. GPU caveat: the device upload
path assumes Svd-variant level kernels, so the override is CPU-only
until `gpu_transport.rs` is extended to dispatch on the enum tag.

## Hypothesis re-check (post-audit, 2026-05-25)

Fresh CPU Godiva run via `metal_stats_diag heu-met-fast-001_case-1
b=80 i=20 p=20000 s=42 r=15`:

| Channel       | engine today | engine in earlier doc rows | OpenMC (3-seed avg, this repo `outputs/openmc_godiva_tallies.json`) | engine / OpenMC |
|---------------|--------------|---------------------------|---------------------------------------------------------------------|-----------------|
| elastic       | 1.7302       | 1.7233                    | 1.7118                                                              | +1.1 %          |
| MT=4 total    | 0.5327       | 0.5330                    | 0.5326 (= 0.4941 + 0.0385 across U-235 + U-238)                     | **+0.02 %**     |
| MT=91 (cont)  | 0.2267       | 0.2265                    | 0.2371 (= 0.2253 U-235 + 0.0102 U-238 + 0.0016 U-234)               | **−4.4 %**      |
| MT=51-90      | 0.3061       | 0.3065                    | 0.2950                                                              | **+3.8 %**      |
| fission       | 0.3862       | 0.3847                    | 0.3851                                                              | +0.3 %          |
| capture       | 0.0456       | 0.0456                    | 0.0446                                                              | +2.2 %          |
| k_eff         | 1.00178 ± 0.00133 | 1.00164 ± 0.00116    | 0.99916 ± 0.00072 (3 seeds)                                         | **+262 pcm**    |

Engine state on Godiva has not drifted since the earlier rows were
recorded. The per-MT rate imbalance the original investigation
flagged is still present. Cross-checked against this audit:

- Total MT=4 rate is within 0.02 % of OpenMC. Combined with the
  Task #20 audit showing σ_MT(E) machine-precision agreement at
  the matched interpolation, **the engine's σ amplitudes are
  correct**.
- MT=91 / MT=4 fraction in the engine is 0.2267 / 0.5327 = 42.6 %;
  OpenMC's is 0.2371 / 0.5326 = 44.5 %. **The engine fires the
  continuum 1.9 absolute percentage points less often than the
  per-level σ ratio σ_MT91(E) / σ_MT4(E) says it should.**

That 1.9-point shift in branching maps onto the +262 pcm k_eff
overshoot: continuum events have a softer outgoing spectrum
(`inelastic_continuum_edist`, MT=91's ContinuousTabular
distribution) than discrete levels, so under-firing MT=91 leaves
the population hotter and lifts fission rate.

### Where the bug must live

For U-235, `NuclideKernels::inelastic_cdf` is `None` (see comment
at `xs_provider.rs:82-87` — MT=4 is native ENDF, so per-level
selection runs the runtime walk in `physics::collision`, not the
pre-tabulated CDF). The bug therefore must be in one of:

1. **`physics::collision::sample_inelastic_level`** (runtime walk:
   build cumulative Σ σ_MTn(E), pick uniform xi over [0, sum],
   binary search for the chosen MT). A bug in the cumulative
   build-up or the xi-renormalisation would skew the picked-MT
   histogram.
2. **The threshold cutoff applied during the walk.** If MT=91's
   threshold or the per-level threshold test fires differently
   from how OpenMC tests it (one above-or-equal vs strict
   above-threshold), the binning shifts.
3. **Energy-domain mismatch between σ sampled at selection time
   vs σ tallied at reaction time** — if the selection uses a
   different (interpolated vs grid-point) σ than the tally, the
   histograms diverge from the analytic ratio.

(1) is the prior. Open the file, instrument a histogram of picked
MT vs E_in over 1 M inelastic events, plot against
`σ_MTn(E) / σ_MT4(E)`. Bug is the row where empirical ≠ analytic.

## Other open follow-ups

### Task #19 (committed work) — MT=91 isolated tally

The plumbing for per-MT split tallies is done CPU-side. GPU-side
would require:

- `atomicAdd` of an MT=91 counter inside `gr_inelastic_event`
- Read-back into `BatchResult`

Not blocking — CPU tally was sufficient for the diagnostic.

### Pending email to Ben Forget

Address his URR + Be (n,2n) hypotheses:

- **Be (n,2n)**: confirmed root cause, fixed in `eb9b62c`, validated
  by OpenMC head-to-head on HCI-003 (Be reflector) within 11 pcm.
- **URR**: bin-to-bin interpolation fix landed in `0aa9591`. Not the
  dominant bias on fast metals (URR range is well below the fast
  spectrum).
- **Current finding**: fast-spectrum-only cases (bare or lead-
  reflector HEU) have a small +200–360 pcm engine bias localised to
  σ_inelastic(E) reconstruction in U-235 at MT=91 vs MT=51-90
  channel ratio. Cases with moderator (graphite, Be) reflectors
  match OpenMC within MC noise. Investigation continues.

### Pending commit

The diagnostic infrastructure (MT split, rank arg, scripts) is
uncommitted as of this checkpoint. To preserve:

```
git add rust_prototype/src/physics/collision.rs \
        rust_prototype/src/transport/simulate.rs \
        rust_prototype/src/transport/dispatch.rs \
        rust_prototype/src/transport/statepoint.rs \
        rust_prototype/src/depletion/flux.rs \
        rust_prototype/src/bin/metal_stats_diag.rs \
        scripts/openmc_scene_runner.py \
        scripts/openmc_godiva_tallies.py \
        scripts/inspect_u235_watt.py \
        scripts/inspect_u235_nu.py \
        scripts/openmc_sweep_diag.sh \
        docs/engine-vs-openmc-bias-investigation.md
```

---

## Numbers to remember

- **+52 pcm**: SVD r=15 overhead vs pointwise on Godiva (intrinsic)
- **+189 pcm**: Godiva engine vs OpenMC residual (real engine bias)
- **+363 pcm**: HMF-027 engine vs OpenMC residual (worst observed)
- **−1 pcm**: HMF-019 engine vs OpenMC (engine perfect on graphite-reflected case)
- **+11 pcm**: HCI-003 case-2 engine vs OpenMC (engine perfect on Be-reflected case)
- **+820 pcm**: HMF-019 library shift VIII.1 vs handbook (NOT engine — same Δ on both engines)
- **+1230 pcm**: HCI-003 case-2 library shift VIII.1 vs handbook (NOT engine)
- **0.99988 ± 8 pcm**: LANL MCNP VIII.1 reference for Godiva (Nobre et al. 2025, Table LIX)
- **2.0 %**: total inelastic rate engine is below OpenMC on Godiva
- **4.5 %**: MT=91 rate engine is below OpenMC on Godiva
- **3.9 %**: MT=51-90 discrete level rate engine is above OpenMC on Godiva

---

## Conclusion for the SVD compression paper

The "memory vs precision" figure has a more nuanced story than
"SVD adds 200 pcm." It's:

- **SVD rank 15 overhead vs pointwise: ~50 pcm** on Godiva
- **Engine has an additional ~150 pcm σ_inelastic-reconstruction
  bias** that is *independent of SVD rank* — it's the precision
  floor of the SVD parametrisation itself for this XS dataset

Going below 50 pcm for paper-grade reference requires either:

- Pointwise lookup (no SVD)
- Different parametrisation (resonance-aware basis)
- Per-channel rank tuning

For fast-iteration sweeps (default rank 15), engine vs OpenMC
parity is ≤ 200 pcm on fast metals, ≤ 50 pcm on moderator-reflected
cases, ≤ 200 pcm on thermal solutions (opposite sign).
