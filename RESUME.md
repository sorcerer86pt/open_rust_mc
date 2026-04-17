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

### Current PWR Results (5-seed, 150 batches, 20k particles, rank 6)

| Mode | k_inf | σ | Gap to OpenMC |
|------|-------|---|---------------|
| **CPU Table** | **1.332 ± 0.001** | 35 pcm | **~180 pcm (stats)** |
| CPU SVD k=6 | 1.332 ± 0.001 | 94 pcm | ~250 pcm |
| GPU SVD k=6 | 1.345 ± 0.001 | 50 pcm | ~1700 pcm |
| GPU pointwise (WIP) | 1.274 | — | broken, temp bug |
| OpenMC | 1.328 ± 0.001 | 50 pcm | — |

**CPU Table matches OpenMC within statistics.** SVD-Table gap is ~37 pcm.
**GPU still has ~2000 pcm gap** from SVD reconstruction error (f32 basis overshoot).

### Root Cause of GPU-CPU Gap: SVD f32 Overshoot

Debug trace (gpu_cpu_trace binary) revealed:
- H1 SVD elastic at 0.02 eV: **60 barns** (SVD) vs **42 barns** (HDF5 at 600K) — **44% overshoot**
- U238 SVD sum at 1 MeV: **4.85 barns** vs **7.07 barns** (HDF5) — missing inelastic
- The Ducru temperature reconstruction amplifies error for nuclides with strong T-dependence

CPU fixes this by using `total_table` (exact HDF5 total) for collision distance.
GPU can't use total_table directly because SVD partials > HDF5 total at some energies,
making capture negative → proportional scaling distorts reaction ratios.

### Solution In Progress: GPU Pointwise XS Tables

Upload exact pointwise XS from HDF5 (7 channels × n_energy per nuclide, ~18 MB total).
GPU does binary search + log-log interpolation on uploaded tables — same as CPU Table mode.

**Status**: Infrastructure complete (`compute_pointwise_xs`, upload, GPU lookup).
Fuel nuclides read correct values. **H1 reads wrong temperature** (59.7 vs 41.7 at 600K).
Bug is in `compute_pointwise_xs(temp_idx=2)` — the interpolated value at 0.02 eV doesn't
match `h5py` direct read. Likely the `read_reaction → interpolate_to_grid` path picks up
wrong temperature data for H1.

**Next step**: Debug H1 pointwise temperature mismatch, then GPU should match CPU Table.

### Bugs Found and Fixed (this session — ~7650 pcm total correction)

10. **GPU S(α,β) kinematic energy scaling** (transport.cu sab_sample):
    Added OpenMC Eq 31/35 scaling, PDF-based CDF inversion (Eq 33/34),
    equiprobable angular bins with smearing. Correctness improvement.

11. **GPU cell-finding nudge** (transport.cu): 1e-10 → 1e-8 matching CPU.

12. **CPU void_crossings never reset** (simulate.rs): Counter killed thermal
    neutrons after 100 lifetime gap crossings. PWR neutrons cross the fuel-clad
    gap ~100+ times → 350 spurious leaks/batch (1.7%). Impact: **~2400 pcm**.

13. **Per-nuclide temperature indices** (pwr_pincell.rs, gpu_pwr_bench.rs):
    Was loading ALL nuclides at temp_idx=1 (294K room temperature). Now:
    U235/U238 at 900K (idx 3), O16/H1/Zr at 600K (idx 2).
    O16 split into fuel (900K) and water (600K) instances. Impact: **~2900 pcm**.

14. **Atom densities matched to OpenMC** (pwr_pincell.rs, gpu_pwr_bench.rs):
    Replaced hardcoded values with OpenMC-computed densities. Impact: **~450 pcm**.

15. **U-238 inelastic synthesis** (xs_provider.rs): MT=4 absent in HDF5 for U238.
    Now summed from discrete levels MT=51-91 at lookup time for both SVD and
    Table providers. Impact: **~1900 pcm**.

16. **Total XS from HDF5** (hdf5_reader.rs, xs_provider.rs): `compute_total_xs`
    reads all physics reactions (MT<200): uses MT=2+MT=3 when available, or sums
    leaf reactions (excluding sum-MTs 1,3,4). Captures missing channels (n,α, n,p,
    n,nα, etc.) that our 6-channel model omits. CPU Table now matches OpenMC.
    Impact: **~475 pcm** (Table mode).

17. **GPU missing channel correction** (transport.cu, gpu_transport.rs):
    Pre-computed `missing_xs = HDF5_total - sum(pointwise MT=2,4,16,17,18,102)`.
    Uploaded to GPU, added to capture. Partial fix (~400 pcm of ~2000 pcm gap).

### Previous Session Bugs (1-9)

1. CPU EnergyHashTable lookup bug (kernel.rs)
2. U-234 nu_bar fallback = 0 (gpu_pwr_bench.rs)
3. Inelastic two-body kinematics (transport.cu)
4. Log-log XS interpolation (xs_provider.rs + transport.cu)
5. Angular distribution interpolation (transport.cu)
6. Fission spectrum interpolation (transport.cu)
7. (n,2n)/(n,3n) neutron banking (transport.cu)
8. Reaction ordering (transport.cu)
9. Free-gas thermal scattering (transport.cu)

## GPU Architecture

**CUDA kernel**: `gpu/cuda/transport.cu` (loaded via `include_str!`)
**Rust orchestration**: `src/gpu_transport.rs`
**Diagnostics**: `src/bin/debug_trace.rs`, `src/bin/gpu_cpu_trace.rs`

### Packed TransportParams (73 u64 fields, one device buffer)

```cuda
typedef const unsigned long long* Params;
#define PTR_F(p, idx)    ((const float*)  (p)[(idx)])
#define PTR_D(p, idx)    ((const double*) (p)[(idx)])
#define PTR_I(p, idx)    ((const int*)    (p)[(idx)])
#define SCALAR_I(p, idx) ((int)(p)[(idx)])
#define SCALAR_D(p, idx) __longlong_as_double((long long)(p)[(idx)])
```

### Kernels

| Kernel | Purpose |
|--------|---------|
| `init_source` | Initialize particles from source bank |
| `compact_alive` | Atomic compaction of alive indices |
| `energy_bin_count/scatter` | 256-bin sort for coalesced SVD access |
| `transport_persistent` | Main: N steps/launch, pointwise or SVD XS lookup |
| `debug_angular_sample` | Diagnostic: angular dist CPU/GPU comparison |
| `debug_xs_reconstruct` | Diagnostic: XS value CPU/GPU comparison |
| `debug_transport_trace` | Diagnostic: per-step trace (17 cols) for GPU-CPU diff |

### Physics in transport_persistent

- **Pointwise XS** from HDF5 when available (7 channels, log-log interp) — WIP
- SVD XS fallback with log-log interpolation between grid points
- Anisotropic angular distributions (correlated interpolation between energies)
- Fission spectrum (correlated interpolation between incident energies)
- S(α,β) for H1 <3.75 eV with kinematic scaling (OpenMC Eq 31-35)
- URR probability tables (band sampling, multiply/absolute)
- Discrete levels (SVD per-level XS, proportional sampling, real Q-values)
- Continuum inelastic MT=91 (evaporation: T=sqrt(E*/a), a=A/8, 0.9 clamp)
- (n,2n)/(n,3n) with neutron banking
- Free-gas thermal (Box-Muller target velocity, angular dist at E_rel)
- Energy-dependent nu-bar
- Warp-level counter reduction, `__launch_bounds__(256, 2)`

### GPU Memory

| Data | Size |
|------|------|
| **Pointwise XS (f64)** | **~18 MB (9 PWR nuclides × 7 channels)** |
| SVD basis (f32) | ~39 MB (9 PWR nuclides) |
| Discrete level basis | ~93 MB |
| Energy grids | ~2.6 MB |
| S(alpha,beta) | ~8 MB/temp |
| Angular dist + URR + nu-bar + fission CDF | ~0.5 MB |

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
rust_prototype/src/thermal.rs             S(alpha,beta) sampling
rust_prototype/src/kernel.rs              SVD kernel (f32 basis, hash lookup, Ducru interp)
paper/svd_cross_section_compression.tex   Manuscript
scripts/paper_openmc_benchmark.py         Multi-seed OpenMC runner
```

## Next Steps

1. **Fix H1 pointwise temperature bug** — `compute_pointwise_xs(temp_idx=2)` gives
   59.7 b elastic at 0.02 eV, should be 41.7 b (600K). Check `read_reaction` →
   `interpolate_to_grid` temperature selection. Once fixed, GPU pointwise should
   match CPU Table (~1.33).

2. **5-seed GPU pointwise benchmark** — validate GPU matches CPU Table within stats.

3. **Rank sweep (k=1..6)** for paper accuracy/speed tradeoff curve.

4. **Update paper tables** with corrected numbers (CPU Table matches OpenMC).

5. **OpenCL port** (gpu/opencl/).
