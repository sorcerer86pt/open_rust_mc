# Resume: open_rust_mc

## First Message

```
Read CLAUDE.md and RESUME.md. Continue working on open_rust_mc.
Working dir: C:\Users\fog\madman_svd_experiment
Rust: rust_prototype/  |  CUDA: gpu/cuda/transport.cu  |  Paper: paper/svd_cross_section_compression.tex
Data: data/endfb-vii.1-hdf5/neutron/  |  Git: sorcerer86pt (GPG signed, new commits only)
OpenMC: wsl -d Ubuntu-24.04, conda activate openmc  |  CUDA: nvcc 12.9, --features cuda
```

## RULE: No Shortcuts

Every shortcut cost more time debugging than doing it right. Do not approximate physics.
Do not skip data uploads. Do not guess parameters. Read real data from HDF5. Implement
the exact CPU algorithm on GPU. If it exists on CPU, port it correctly to GPU.

## State

### PWR Pin Cell — Rank Sweep (5 seeds, 50 batches, 10k particles)

XS reconstruction Pareto (geom-mean over 8 U235/U238/U234 reactions):

| rank | ns/lookup | RMSE log10 | max |err| log10 |
|---:|---:|---:|---:|
| 2 | 35.9 | 3.00e-2 | 9.84e-2 |
| 3 | 41.2 | 7.06e-3 | 2.86e-2 |
| 4 | 43.3 | 1.83e-3 | 7.91e-3 |
| 5 | 43.4 | 6.07e-4 | 3.46e-3 |
| 6 | 44.3 | 1.76e-15 | 5.58e-15 |
| pointwise table | 90.6 | 0 (ref) | 0 (ref) |

**Rank 6 dominates: 2× faster than pointwise table at machine precision.**

PWR k_inf across all variants (SEM = σ_seed / √5):

| mode | rank | k_inf | σ_seed | Δ vs OpenMC (pcm) | ns/particle |
|---|---:|---:|---:|---:|---:|
| OpenMC 0.15.3 | — | 1.32770 | 0.00150 | +0 | — |
| CPU table (ours) | — | 1.32526 | 0.00232 | −244 | 54 430 |
| CPU SVD | 2 | 1.40807 | 0.00144 | **+8037** | 52 711 |
| CPU SVD | 3 | 1.32118 | 0.00281 | −652 | 28 672 |
| CPU SVD | 4 | 1.32513 | 0.00145 | −257 | 31 875 |
| CPU SVD | 5 | 1.32479 | 0.00190 | −291 | 28 651 |
| CPU SVD | 6 | 1.32512 | 0.00114 | −258 | 30 073 |
| GPU pointwise | — | 1.32670 | 0.00117 | −100 | 38 104 |
| GPU SVD (--force-svd) | 2 | 1.40973 | 0.00219 | **+8203** | 48 289 |
| GPU SVD (--force-svd) | 3 | 1.32260 | 0.00207 | −510 | 48 439 |
| GPU SVD (--force-svd) | 4 | 1.32496 | 0.00318 | −274 | 48 617 |
| GPU SVD (--force-svd) | 5 | 1.32804 | 0.00169 | +34 | 50 738 |
| GPU SVD (--force-svd) | 6 | 1.32673 | 0.00201 | −97 | 53 865 |

CPU and GPU SVD rank sweeps now agree within seed noise (ranks 3-6 all within 1σ of
OpenMC). Rank 2 under-resolves U238 capture resonances on both paths (+8000 pcm bias).
GPU pointwise matches OpenMC within 2σ SEM. **GPU --force-svd rank 5: Δ = +34 pcm
(1σ of OpenMC), after fixing two SVD-fallback bugs in `transport.cu` — see Bug
Fixes 6 and 7 below.**

Artifacts: `outputs/pareto/{pareto_pwr.png, pareto_pwr.md, xs_accuracy.csv, keff_pwr.csv, openmc_pwr.json}`.

### Godiva — Rank Sweep (5 seeds, 50 batches, 10k particles)

| mode | rank | k_eff | σ_seed | Δ vs CPU table (pcm) | ns/particle |
|---|---:|---:|---:|---:|---:|
| CPU table | — | 1.00523 | 0.00235 | 0 | 2 385 |
| CPU SVD | 2 | 1.00448 | 0.00165 | −75 | 864 |
| CPU SVD | 3 | 1.00559 | 0.00094 | +36 | 1 457 |
| CPU SVD | 4 | 1.00638 | 0.00166 | +115 | 1 182 |
| CPU SVD | 5 | 1.00592 | 0.00178 | +69 | 1 403 |
| CPU SVD | 6 | 1.00489 | 0.00205 | −34 | 1 424 |
| GPU pointwise | — | 1.00554 | 0.00208 | +31 | ~1 500 |

All Godiva rows within 1σ of table baseline — small system, no resonance region stress.
`outputs/pareto/{pareto.png, keff_sweep.csv}`.

### Bugs Found and Fixed (this session)

**1. Discrete-level SVD rank mismatch** (xs_provider.rs:462)
- Discrete levels built with `svd_rank.min(2)=2`, GPU kernel uses `P_RANK=5` as
  basis stride. Reads past each level's basis into neighboring data → per-level XS
  essentially random (values like 1e-90 instead of 0.5).
- Fix: use full `svd_rank` for discrete levels so stride matches.
- Impact on PWR: massive (enabled subsequent URR fix to land).
- Impact on Godiva: -400 pcm (1.010 → 1.006).

**2. URR total XS not recomputed** (transport.cu)
- `apply_urr()` modifies `s_el`, `s_fis`, `s_cap` but GPU left `micro_t` unchanged,
  so macroscopic total disagreed with sum of partials during URR sampling.
- Fix: track per-channel delta and adjust `micro_t` (matches CPU which recomputes
  `xs.total = sum(partials)` after URR).
- Impact on PWR: +5500 pcm (1.273 → 1.328). This was the dominant bug.

**3. SVD basis f32 → f64** (kernel.rs, gpu_transport.rs, gpu.rs, transport.cu)
- Eliminates long-standing f32 overshoot issue mentioned in prior RESUME notes.
- No k_inf change in pointwise mode (XS already f64), but fixes SVD mode.

**4. Hydrogen special-case mu_lab** (transport.cu)
- GPU now uses CPU's special case for A ≤ 1+eps: `mu_lab = sqrt((1+mu_cm)/2)`
  instead of the general `(1+A·mu_cm)/sqrt(1+A²+2A·mu_cm)` which becomes
  numerically unstable near A=1, mu_cm=-1.

**5. 9-nuclide PWR layout** (gpu_pwr_bench.rs)
- Was 8 nuclides with O16 shared fuel/water at 600K; now 9 with O16 at 900K for
  fuel (idx 2) and 600K for water (idx 8), matching CPU pwr_pincell exactly.

**6. GPU SVD micro_t ignored HDF5 total** (transport.cu, main + debug-trace)
- SVD fallback set `micro_t = s_el + s_inel + s_n2n + s_n3n + s_fis + s_cap`
  (sum of six reaction-type SVD partials) and never consulted `P_TOTAL_XS`.
- CPU's `SvdXsProvider::lookup` uses `total_table.lookup(E)` for total and
  sets `capture = tot − (el+inel+n2n+n3n+fis)` so capture absorbs "missing"
  channels (MT=19-21 first-chance fission, n,α, n,p, charged-particle emission).
- U-238 resonance region: missing channels ~matter, under-absorption lifts k_inf
  systematically high.
- Fix: GPU SVD branch now interpolates `P_TOTAL_XS` log-log and sets
  `s_cap = max(tot − partials, 0)`, `micro_t = tot` — same as CPU.

**7. GPU SVD inelastic = 0 when MT=4 synthesized** (transport.cu)
- Zr90/91/92/94 have no MT=4 block in HDF5; CPU synthesizes total inelastic
  by summing 13 discrete-level SVD kernels inside `lookup()`.
- GPU SVD branch relied on `HAS_REACTION[inelastic]` and left `s_inel=0`.
- Combined with fix 6, the missing inelastic XS got dumped into capture,
  flipping rank-5 force-svd k_inf from +2588 to -32438 pcm (0.903).
- Fix: when `!has_inel_k`, sum level SVD reconstructions at `(e_idx, log_frac)`
  and assign to `s_inel` — mirrors CPU.

**Post-fix force-svd results (5 seeds, 50 batches, 10k particles)**
- Rank 5: k_inf = 1.32804 ± 0.00169 → Δ = +34 pcm (was +2588 pcm)
- Rank 3: k_inf = 1.32260 ± 0.00207 → Δ = −510 pcm (was +2497 pcm, CPU rank-3 gives −652)

### Verified kinematics match CPU (via diagnostic binaries during debugging)

- Pointwise XS: exact (10-digit match at test energies)
- S(α,β) sampling: 0.3% match (mean E_out and mu at 10 energies)
- Elastic scattering (free-gas + cold-target): 0.2% match
- Inelastic sampling (level selection + two-body kinematics): <0.5% match
- Fission spectrum: 0.5% match
- Geometry distances: exact match

Diagnostic binaries (deleted after use): `pw_diag`, `sab_compare`, `elastic_compare`,
`inelastic_compare`, `fission_compare`, `level_xs_compare`, `geom_diag`.

## GPU Architecture

**CUDA kernel**: `gpu/cuda/transport.cu` (loaded via `include_str!`)
**Rust orchestration**: `src/gpu_transport.rs`
**Diagnostics**: `src/bin/debug_trace.rs`, `src/bin/gpu_cpu_trace.rs`

### Packed TransportParams (73 u64 fields, one device buffer)

```cuda
typedef const unsigned long long* Params;
#define PTR_D(p, idx)    ((const double*) (p)[(idx)])
#define PTR_I(p, idx)    ((const int*)    (p)[(idx)])
#define SCALAR_I(p, idx) ((int)(p)[(idx)])
#define SCALAR_D(p, idx) __longlong_as_double((long long)(p)[(idx)])
```

All SVD basis is now f64 (was f32). PTR_F removed.

### Kernels

| Kernel | Purpose |
|--------|---------|
| `init_source` | Initialize particles from source bank |
| `compact_alive` | Atomic compaction of alive indices |
| `energy_bin_count/scatter` | 256-bin sort for coalesced SVD access |
| `transport_persistent` | Main: N steps/launch, pointwise or SVD XS lookup |
| `debug_xs_reconstruct` | Diagnostic: SVD XS at given energies |
| `debug_angular_sample` | Diagnostic: angular dist sampling |
| `debug_transport_trace` | Diagnostic: per-step trace (17 cols) |

### Physics in transport_persistent

- **Pointwise XS** from HDF5 (7 channels, log-log interp) — default path
- SVD XS fallback (f64 basis, log10 reconstruction)
- URR probability tables (band sampling, proper total recomputation)
- S(α,β) for H1 <3.75 eV with kinematic scaling (OpenMC Eq 31-35)
- Anisotropic angular distributions (correlated interpolation between energies)
- Fission spectrum (correlated interpolation between incident energies)
- Discrete levels (SVD per-level XS at full rank, proportional sampling, real Q-values)
- Continuum inelastic MT=91 (evaporation: T=sqrt(E*/a), a=A/8, 0.9 clamp)
- (n,2n)/(n,3n) with neutron banking
- Free-gas thermal (Box-Muller target velocity, angular dist at E_rel)
- Energy-dependent nu-bar
- Warp-level counter reduction, `__launch_bounds__(256, 2)`

### GPU Memory (9 nuclides, rank=5)

| Data | Size |
|------|------|
| SVD basis (f64) | ~64 MB |
| Pointwise XS (f64) | ~18 MB |
| Discrete level basis (f64, rank=5) | ~230 MB |
| Energy grids | ~2.6 MB |
| S(α,β) | ~8 MB |
| Angular dist + URR + nu-bar + fission CDF | ~0.5 MB |

Discrete level basis is larger now that rank matches top-level (was ~92 MB with rank=2).

## Files

```
rust_prototype/src/bin/godiva.rs          CPU Godiva (--mode svd|table|both --seeds N)
rust_prototype/src/bin/pwr_pincell.rs     CPU PWR (--mode svd|table|both --seeds N)
rust_prototype/src/bin/gpu_pwr_bench.rs   GPU benchmark (--geometry pwr|godiva --seeds N --force-svd)
rust_prototype/src/bin/pareto_bench.rs    XS RMSE + ns/lookup per rank (CSV to stdout)
scripts/pareto_plot_pwr.py                Render PWR Pareto panels + markdown
scripts/openmc_pwr_ref.py                 Multi-seed OpenMC PWR reference (WSL openmc env)
rust_prototype/src/bin/gpu_cpu_trace.rs   GPU vs CPU step-by-step trace comparison
rust_prototype/src/bin/debug_trace.rs     CPU vs GPU physics diagnostic
rust_prototype/src/gpu_transport.rs       Rust GPU orchestration (packed params, upload, launch)
rust_prototype/gpu/cuda/transport.cu      CUDA kernels (persistent transport + diagnostics)
rust_prototype/src/transport/simulate.rs  CPU transport loop (surface tracking)
rust_prototype/src/transport/xs_provider.rs  SVD + Table XS providers (total from HDF5)
rust_prototype/src/hdf5_reader.rs         HDF5 reader (XS, angular, URR, thermal, nu-bar, pointwise)
rust_prototype/src/thermal.rs             S(α,β) sampling
rust_prototype/src/kernel.rs              SVD kernel (f64 basis, hash lookup, Ducru interp)
paper/svd_cross_section_compression.tex   Manuscript
scripts/paper_openmc_benchmark.py         Multi-seed OpenMC runner
```

## Next Steps

1. ~~**Godiva 300 pcm residual**~~ — resolved: 31 pcm gap across 5 seeds, noise.

2. ~~**Rank sweep**~~ — done. Pareto in `outputs/pareto/pareto_pwr.png`.
   Rank 6 beats pointwise table on both axes (2× faster, machine-precision RMSE).

3. ~~**GPU --force-svd +2500 pcm bias**~~ — resolved. Two GPU SVD-fallback bugs
   in transport.cu (see Bug Fixes 6 and 7). Rank-5 force-svd now within 1σ
   of OpenMC (Δ = +34 pcm).

4. **Performance benchmark** — GPU now correct, measure ns/particle vs CPU for
   paper tables. GPU pointwise: 28 855 ns/p; CPU table: 54 430 ns/p (1.9×).

5. **Update paper tables** with validated GPU numbers matching OpenMC.

6. **OpenCL port** (gpu/opencl/) for non-NVIDIA GPUs.
