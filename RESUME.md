# Resume: open_rust_mc

## First message

```
Read CLAUDE.md and RESUME.md. Continue working on open_rust_mc.
Working dir: C:\Users\fog\madman_svd_experiment
Rust:  rust_prototype/       |  CUDA:  gpu/cuda/transport.cu
Paper: paper/main.tex        |  Bib:   paper/references.bib
Data:  data/endfb-vii.1-hdf5/neutron/       WMP: .../wmp/
Git:   sorcerer86pt, GPG signed, new commits only
OpenMC: wsl -d Ubuntu-24.04, conda activate openmc
CUDA:   nvcc 12.9, cargo --features cuda
```

## RULE: No shortcuts

Every shortcut cost more time debugging than doing it right. Do not
approximate physics. Do not skip data uploads. Do not guess
parameters. Read real data from HDF5. Implement the exact CPU
algorithm on GPU. If it exists on CPU, port it correctly to GPU.

## Immediate next task

**Desktop re-run of GPU rows, item-2 hybrid on desktop 3080.**

GPU hybrid transport is wired end-to-end and runs without
crash (laptop A1000: $k_\infty = 1.32899 \pm 0.00094$ with 5
seeds, 20k particles, 50 active batches). GPU pointwise with
the corrected sampler (see below) moved from $-78$ to $-54$
pcm vs OpenMC on the laptop. Both are first-cut numbers on
the laptop; the desktop 10-seed paper benchmarks on the
3080 (150 batches, 50k particles, 100 inactive) are the
table numbers still to gather.

## Resolved this session

**Item 2 — GPU hybrid transport-loop integration (DONE).**

Wired the validated CUDA WMP evaluator
(`gpu/cuda/transport.cu::wmp_eval`) into the event-based
`transport_persistent` kernel. Completed steps:

1. `TransportParams` grew from 73 to 90 `u64` slots (`P_WMP_*` =
   73..89): `has`, per-nuclide scalars (`E_min`, `E_max`, `spacing`,
   `sqrt_awr`, `t_kelvin`, `fit_order`, `n_windows`, `fissionable`),
   and flat `poles` + `windows` + `broaden_poly` + `curvefit` with
   per-nuclide offsets. Rust side: new `GpuWmpData` struct and
   `upload_wmp_data` / `upload_wmp_data_empty` helpers in
   `gpu_transport.rs`.
2. `has_wmp[i]` bitmap wired through `P_WMP_HAS` and checked before
   the URR branch in `transport_persistent`.
3. Inside the per-nuclide XS-lookup: when `has_wmp[i]` and
   `E ∈ [E_min, E_max]`, elastic / fission / capture are replaced
   with `wmp_eval(E, T_K, ...)` and total is recomputed from
   partials. Inelastic, (n,2n), (n,3n) stay on the SVD path.
4. URR override is suppressed inside the WMP window.
5. CUDA recursion fix: `wmp_faddeeva` was folded to an iterative
   conjugate-flag form. The original recursive upper-half-plane
   fold-up (matching OpenMC's Python exactly) blows the per-thread
   stack when called under `__launch_bounds__(256, 2)` — the register
   pressure from the rest of `transport_persistent` leaves no room
   for a recursive frame and the launch faulted with
   `CUDA_ERROR_ILLEGAL_ADDRESS` on the first in-window energy. The
   rewritten evaluator is still bit-exact against the CPU
   reference at $\leq 5 \cdot 10^{-14}$ (re-validated via
   `gpu_wmp_validate.exe`).
6. `--mode {svd,hybrid}` flag on `gpu_pwr_bench.rs`. Hybrid loads
   the per-nuclide WMP HDF5 files from `data/wmp/` (U-235 + U-238
   succeed on ENDF/B-VII.1; O-16 / H-1 / Zr isotopes ship in a layout
   the current reader does not accept — they stay on the SVD path).
7. Paper update: `sections/gpu_wmp.tex` now has an
   "End-to-end GPU hybrid transport" paragraph covering the wiring,
   the recursion-to-iteration fix, and pointing to the PWR
   benchmark table for $k_\infty$.

**Item 3 — GPU sampler correction (DONE; laptop numbers below,
desktop rerun still pending).**

Ported the OpenMC stochastic-bin sampling to the GPU kernel,
mirroring the CPU fix from the previous session
(`src/hdf5_reader.rs::AngularDistribution::sample_mu` and
`EnergyDistribution::sample`):

- `gpu/cuda/transport.cu::sample_angular_dist` now uses
  `r = (E − E_lo)/(E_hi − E_lo)`, `pick_hi = (ξ_bin < r)`,
  sample μ from the chosen bin with a fresh ξ_μ. Old correlated
  single-ξ inversion + linear μ interpolation is gone.
- `sample_fission_energy` uses the same stochastic bin pick plus
  OpenMC's scaled kinematic remap (ContinuousTabular::sample,
  `distribution_energy.cpp`): remap `E_out` from the chosen bin's
  `[el1_lo, el1_hi]` to the interpolated `[E_1, E_K]`.
- New helpers `sample_mu_bin` and `sample_eout_bin` factor out
  the per-bin CDF inversion so both the edge-below / edge-above
  branches and the interior branch share the same code path.

**Item 4 — Shannon entropy auto-convergence (DONE).**

The previous session added `EntropyMesh` and printed `H` per
batch but never wired it into runtime control. Completed now:

- `transport/simulate.rs::EntropyConvergence` — new policy struct
  with `min_inactive` (20), `max_inactive` (200), sliding
  `window` (10), and `cv_tol` (5e-3, above the measured 2e-3
  noise floor at 10k particles / batch). `has_converged(history)`
  returns true when the trailing window's σ/μ drops below `cv_tol`
  and `history.len() >= min_inactive`.
- `SimConfig.auto_inactive: Option<EntropyConvergence>`. When
  `Some`, the run loop starts with `effective_inactive =
  config.batches` so no batch is active until the detector fires;
  on fire, `effective_inactive = batch` and batch+1 is the first
  active. Fallback: at `max_inactive`, force convergence.
- `BatchResult.active: bool` propagates the decision back to the
  binaries (`godiva.rs`, `pwr_pincell.rs`) so their post-hoc
  averaging filters on `r.active` instead of `r.batch > args.inactive`.
- CLI: `--auto-inactive` flag on both binaries. Verified on
  Godiva + 10k particles: plateau fires at batch 20 (min_inactive
  floor), batch 21 is first active, entropy CV ≈ 2e-3 once settled.

**Unit test expansion (DONE, 38 → 62).**

Targeted coverage for every piece touched this session, so the
CPU path guards the GPU port and the auto-inactive policy is
regression-testable without a full eigenvalue run:

- `EntropyConvergence` (7 tests): empty history is rejected;
  `min_inactive` gate holds; flat window fires; noisy window does
  not; only the tail matters; near-zero mean does not divide by
  zero; 0.5% vs 2% CV both check the threshold correctly.
- `EntropyMesh` (3 tests): empty bank → 0; concentrated bank
  (all sites in one bin) → 0; uniform bank → log₂(n³) saturation.
- `AngularDistribution::sample_mu` (6 tests): output always in
  [−1, +1]; isotropic ⟨μ⟩ ≈ 0; forward-peaked ⟨μ⟩ > 0.5;
  stochastic-bin r=0.25 with δ-like bins at ±1 predicts
  ⟨μ⟩ ≈ −0.5 (the actual statistical signature of the stochastic
  pick); below-grid and above-grid use the correct edge bin.
- `EnergyDistribution::sample` (3 tests): output > 1e-5 and
  bounded; the scaled kinematic remap at r=0.5 keeps samples in
  the interpolated `[E_1, E_K]` envelope; below-grid uses the
  first bin.
- `wmp::faddeeva` (5 extra tests): real-axis identity
  `Re(w(x)) ≈ exp(−x²)`; `w(0) = 1`; continuity across the
  `s = 5.5` region boundary (validates the iterative rewrite);
  OpenMC lower-half convention `w(x − iy) = (−Re, +Im)` of
  `w(x + iy)`; asymptotic magnitude `|w(z)| ≈ 1/(|z|√π)` in
  Region I. Plus `broaden_wmp_polynomials` coefficient sanity.

**Laptop smoke numbers (RTX A1000, 5 seeds, 20k particles,
150 batches, 100 inactive):**

| mode                              | k_inf   | σ_seed   | ns/p  |
|-----------------------------------|--------:|---------:|------:|
| GPU SVD (corrected sampler)       | 1.32716 | 0.00057  | 51 146|
| GPU Hybrid SVD+WMP (2/9 nuclides) | 1.32899 | 0.00094  | 77 447|

Desktop 3080 re-run remains to be done for the paper table.

## State as of this session

### PWR pin cell — desktop Ryzen 9800X3D + RTX 3080

All four in-engine providers verified against OpenMC 0.15.3. Final
ten-seed confirmation:

| mode                         | rank | k_inf     | σ_seed    | Δ OpenMC (pcm) | ns/p    |
|------------------------------|-----:|----------:|----------:|---------------:|--------:|
| OpenMC 0.15.3                | —    | 1.32770   | 0.00150   | +0             | —       |
| **CPU SVD (corrected)**      | **5**| **1.32769** | **0.00042** | **−1**   | **39 978** |
| CPU table                    | —    | 1.32649   | 0.00068   | −121           | 19 015  |
| GPU pointwise                | —    | 1.32692   | 0.00041   | −78            | 6 111   |
| GPU SVD (`--force-svd`)      | 5    | 1.32650   | 0.00042   | −120           | 7 946   |
| Hybrid SVD+WMP (5 seeds)     | 5    | 1.32557   | 0.00070   | −213           | 33 051  |

`SEM_10seed = σ_seed/√10 ≈ 13 pcm`. The previous ~120 pcm offset is
closed on the corrected CPU-SVD row. GPU rows were benchmarked on
the pre-correction sampler and still carry the ~100 pcm residual; a
re-run with the fixed sampler is the natural follow-up
(independent of item 2).

### Godiva — desktop rank sweep (10 seeds, 150 batches, 50k particles)

| mode              | rank | k_eff     | σ_seed    | ns/p     |
|-------------------|-----:|----------:|----------:|---------:|
| CPU table         | —    | 1.00579   | 0.00046   | 1 214    |
| CPU SVD           | 2    | 1.00477   | 0.00055   | 1 289    |
| CPU SVD           | 3    | 1.00547   | 0.00031   | 1 316    |
| CPU SVD           | 4    | 1.00523   | 0.00055   | 1 369    |
| CPU SVD           | 5    | 1.00507   | 0.00065   | 1 426    |
| CPU SVD           | 6    | 1.00524   | 0.00071   | 1 558    |
| GPU pointwise     | —    | 1.00590   | 0.00037   | 278.5    |
| GPU SVD           | 5    | 1.00536   | 0.00046   | 357.0    |

### XS reconstruction Pareto (desktop, single core)

| rank | ns/lookup | RMSE log10 |
|-----:|----------:|-----------:|
| 2                   | 35.9 | 3.00e-2 |
| 3                   | 41.2 | 7.06e-3 |
| 4                   | 43.3 | 1.83e-3 |
| 5                   | 43.4 | 6.07e-4 |
| 6                   | 44.3 | 1.76e-15 |
| pointwise table     | 90.6 | 0 (ref)  |

### Hybrid SVD+WMP — memory

Two separate measurements. Both real; they measure different things.

- **Representation-byte ratio (narrow)**: WMP + rank-2 SVD-smooth for
  four reactions × six temperatures = 1.37 MB for 9 nuclides, vs
  177.7 MB pointwise = **132.9×**. Script:
  `scripts/hybrid_svd_wmp_experiment.py`.
- **In-engine measured (broad)**: current hybrid (full SVD basis +
  WMP payload) = 519.0 MB; smooth-only projection = 487.6 MB;
  pointwise-table baseline = 101.5 MB. Hybrid is **4.8–5.1× larger**
  than pointwise in-engine because discrete-level SVD kernels dominate
  and are not WMP-covered. Reported at load time by
  `HybridSvdWmpXsProvider::memory_report`.

### GPU WMP evaluator (validated, not yet in transport loop)

- `gpu/cuda/transport.cu::wmp_faddeeva`, `wmp_broaden_poly`,
  `wmp_erf`, `wmp_eval`, `extern "C" __global__ wmp_test_eval`.
- Bit-exact vs CPU: max relative error `5.3e-14` (absorption),
  `2.0e-13` (fission) over 12 U-238 test energies.
- Throughput on RTX 3080: **9.5 ns/lookup** (saturated 1M-thread
  launch), vs 66.8 ns/lookup single-thread CPU — **7.0× speedup**.
- Validator binary: `src/bin/gpu_wmp_validate.rs`
  (`--gpu-n N --cpu-n N --reps N`).

## Bugs found and fixed this session

Tracing the original ~120 pcm OpenMC offset revealed four causes.
All four fixes landed.

**A. Capture-residue hypothesis — ruled out, not a bug.**
XS audit: total XS matches OpenMC to <0.25% across all nuclides, all
spectrum regions. No capture-residue drift. Scripts:
`scripts/xs_dump_openmc.py`, `src/bin/xs_dump.rs`,
`scripts/xs_audit_diff.py`.

**B. ν̄ delayed-neutron yield averaging bug.**
`src/hdf5_reader.rs::read_nu_bar_from_group` was collapsing tabulated
delayed-yield (E, ν_delayed(E)) pairs into a single per-product
constant. U-235 ν̄ was −0.002 low, U-238 ν̄ was −0.009 low across
their full energy ranges. Fix: interpolate each delayed yield on the
prompt grid and sum per-energy. ν̄(E) now bit-exact against OpenMC.

**C. Correlated-CDF → stochastic-bin sampling for angular and fission
energy.** OpenMC's `distribution_angle.cpp` and
`distribution_energy.cpp` use `r = (E − E_lo)/(E_hi − E_lo)`,
`pick_hi = (prn() < r)`, sample once inside chosen bin, plus for
energy distributions a scaled kinematic remap. We were using
single-ξ inversion at both bracketing bins followed by linear
interpolation of μ (or E_out). Switched to OpenMC's convention in
`AngularDistribution::sample_mu` and `EnergyDistribution::sample`.
**Shift: +51 pcm.**

**D. Source-convergence budget.** 20 inactive batches at 50k
particles was not enough. Raised to 100 inactive for the reported
benchmark. **Shift: +54 pcm.** Shannon-entropy monitor now
implemented (`src/transport/simulate.rs::EntropyMesh`, 8³ Cartesian
mesh, H emitted per batch alongside k_batch).

Combined with a +11 pcm noise reduction from 5 → 10 seeds, the
total shift is **+126 pcm**, bringing Δ(OpenMC) from −127 to −1.

## Paper state

**Paper is modular.** Entry point: `paper/main.tex` → `\input{sections/*.tex}` → `\bibliography{references}`.

```
paper/
├── main.tex                       ← build with pdflatex + bibtex
├── references.bib                 ← BibTeX
├── sections/
│   ├── abstract.tex  intro.tex  method.tex  implementation.tex
│   ├── spectrum.tex  godiva.tex  pwr.tex  gpu.tex
│   ├── hybrid.tex    gpu_wmp.tex  threats.tex  related.tex
│   ├── conclusion.tex  backmatter.tex
├── svd_cross_section_compression.tex   ← legacy monolithic file (still compiles)
├── svd_cross_section_compression.pre_honest.tex  ← backup (pre-honest-rewrite)
├── svd_cross_section_compression.pre_3amigos.tex ← backup (pre-3-amigos)
└── physics_bom.tex                 ← computational physics BoM (separate doc)
```

Current paper is **19 pages** with six figures:

- `outputs/pareto/svd_spectrum.png` — σ_k/σ_1 decay + rank-one histogram
- `outputs/pareto/per_lookup_cost.png` — kernel benchmark Pareto
- `outputs/pareto/pareto.png` — Godiva laptop Pareto
- `outputs/pareto/throughput_godiva.png` — Godiva desktop rank sweep
- `outputs/pareto/pareto_pwr.png` — PWR laptop Pareto
- `outputs/pareto/throughput_pwr.png` — PWR laptop+desktop throughput
- `outputs/pareto/memory_compare.png` — hybrid memory (honest in-engine)

The paper was rewritten once under a "3 amigos" discipline (physics,
statistics, engineering reviewers), then again to remove false
headline claims about the 132.9× memory reduction and present both
the representation-byte ratio and the in-engine measurement cleanly.

## File map (what's where)

```
rust_prototype/
  src/
    lib.rs
    wmp.rs                                — CPU WMP evaluator (Humlicek W4 + reader)
    transport/
      simulate.rs                         — + EntropyMesh, per-batch H emission
      xs_provider.rs                      — SvdXsProvider, TableXsProvider
      hybrid_xs.rs                        — HybridSvdWmpXsProvider + memory_report
    hdf5_reader.rs                        — ν̄ fix + stochastic-bin sampling
    gpu_transport.rs                      — Rust GPU orchestration
  src/bin/
    pwr_pincell.rs                        — --mode svd|table|both|hybrid
    godiva.rs                             — --mode svd|table|both
    gpu_pwr_bench.rs                      — GPU transport (no hybrid yet)
    wmp_validate.rs                       — CPU WMP validator vs OpenMC Python
    gpu_wmp_validate.rs                   — GPU WMP validator + throughput bench
    xs_dump.rs                            — engine-side XS dump for audit
    pareto_bench.rs                       — kernel-level Pareto
  gpu/cuda/transport.cu                   — + wmp_faddeeva, wmp_eval, wmp_test_eval

paper/                                    — see "Paper state" above
scripts/
  xs_dump_openmc.py                       — OpenMC reference XS dump
  xs_audit_diff.py                        — diff OpenMC vs engine
  hybrid_svd_wmp_experiment.py            — representation-byte memory analysis
  plot_svd_spectrum.py                    — SVD spectrum figure
  plot_memory_throughput.py               — memory + throughput figures

outputs/
  pareto/                                 — all figures
  xs_audit/                               — XS audit CSVs + PWR final 10-seed log
  hybrid_wmp/                             — hybrid memory + accuracy + 5-seed k_inf
```

## Known issues, not fixed

- **GPU rows in Table 4 use pre-correction sampler.** GPU
  pointwise and GPU SVD k_inf values (−78 and −120 pcm from OpenMC)
  were measured before the stochastic-bin + 100-inactive-batch fixes.
  Expected after re-run: both collapse to ~0 pcm like the corrected
  CPU row. Rerun is ~1h wall-time on desktop, but not essential
  for item 2.

- **Smooth-only SVD basis rebuild not implemented.** The shared
  energy grid across reactions complicates a per-reaction smooth-only
  restriction. The `memory_report()` projects 487.6 MB for the
  hypothetical rebuild; realising it in code is a separate
  engineering task (not required for item 2).

- **Full-core benchmark not attempted.** 9 nuclides, 1 geometry, 2
  hardware tiers is the scope. BEAVRS pin cell → assembly → full core
  would widen validity but is future work.

- **Only one S(α,β) library exercised.** H in H₂O only. D₂O,
  graphite, ZrH not tested.

## Quick commands

```bash
# Build everything
cd rust_prototype && cargo build --release --features cuda

# PWR pin cell, corrected sampling, 10-seed desktop benchmark
./target/release/pwr_pincell.exe ../data/endfb-vii.1-hdf5/neutron \
  --mode svd --rank 5 --batches 250 --inactive 100 \
  --particles 50000 --seeds 10

# Hybrid PWR (CPU)
./target/release/pwr_pincell.exe ../data/endfb-vii.1-hdf5/neutron \
  --mode hybrid --rank 5 --batches 100 --inactive 20 \
  --particles 20000 --seeds 5

# GPU WMP validation (--gpu-n / --cpu-n / --reps)
./target/release/gpu_wmp_validate.exe \
  ../data/endfb-vii.1-hdf5/wmp/092238.h5 \
  --gpu-n 1000000 --cpu-n 100000 --reps 5

# XS audit (OpenMC side needs WSL + openmc conda env)
wsl -d Ubuntu-24.04 -- bash -lic \
  'conda activate openmc; python /mnt/c/Users/fog/madman_svd_experiment/scripts/xs_dump_openmc.py'
./target/release/xs_dump.exe ../data/endfb-vii.1-hdf5/neutron \
  ../outputs/xs_audit/rust_svd.csv --mode svd --rank 5
python ../scripts/xs_audit_diff.py svd

# Build paper
cd ../paper && pdflatex -interaction=nonstopmode main.tex \
  && bibtex main \
  && pdflatex -interaction=nonstopmode main.tex \
  && pdflatex -interaction=nonstopmode main.tex
```

## Key numbers (after fixes)

| metric                                    | value                 |
|-------------------------------------------|-----------------------|
| Rank-1 U-235 non-redundant reactions      | 43 of 47              |
| Rank-6 ns/lookup vs table                 | 44 vs 91 (2.0×)       |
| CPU SVD speedup vs CPU table (PWR)        | 1.18×–1.90× hardware  |
| CPU SVD rank-5 vs OpenMC (corrected)      | **−1 pcm** (10 seeds) |
| Hybrid SVD+WMP vs CPU SVD (k_inf)         | −86 pcm (5 seeds)     |
| Hybrid throughput vs CPU SVD              | 2.06× slower          |
| GPU WMP lookup (RTX 3080, saturated)      | 9.5 ns                |
| GPU-vs-CPU WMP speedup                    | 7.0×                  |
| Representation byte ratio (Table 4)       | 132.9×                |
| In-engine hybrid vs pointwise (Fig ~mem~) | 5× **larger**         |
| OpenMC offset, pre-fix vs post-fix        | −127 → −1 pcm         |
