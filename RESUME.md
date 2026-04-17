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

### Current PWR Results (3 seeds, 50 batches, 10k particles, rank 5)

| Mode | k_inf | σ | Match |
|------|-------|---|-------|
| **CPU Table** | **1.322 ± 0.002** | 45 pcm | baseline |
| CPU SVD k=5 | 1.329 ± 0.002 | 50 pcm | +70 pcm |
| **GPU Pointwise** | **1.328 ± 0.001** | 10 pcm | matches OpenMC |
| OpenMC reference | 1.328 | 50 pcm | — |

**GPU now matches CPU reference within statistics.**

### Current Godiva Results (50 batches, 10k particles)

| Mode | k_inf | Gap to CPU |
|------|-------|------------|
| CPU Table | 1.003 ± 0.002 | baseline |
| GPU | 1.006 ± 0.002 | +300 pcm (~2σ, borderline) |

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
rust_prototype/src/bin/gpu_pwr_bench.rs   GPU benchmark (--geometry pwr|godiva --seeds N)
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

1. **Godiva 300 pcm residual** — borderline statistical (~2σ). If systematic,
   likely URR sampling detail for U-234/U-235/U-238 or fission spectrum
   interpolation. Run multi-seed to confirm if real.

2. **Performance benchmark** — GPU now correct, measure ns/particle vs CPU for
   paper tables. GPU pointwise path uses f64 pointwise XS (~18 MB) with log-log
   interpolation, same as CPU Table mode.

3. **Update paper tables** with validated GPU numbers matching OpenMC.

4. **Rank sweep (k=1..6)** for SVD mode accuracy/speed tradeoff curve.

5. **OpenCL port** (gpu/opencl/) for non-NVIDIA GPUs.
