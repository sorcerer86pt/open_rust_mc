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

### 10-Seed Results (150 batches, 50k particles, hash-fixed)

| Benchmark | Mode | k | sigma | ns/p | Δ(exp) |
|-----------|------|---|-------|------|--------|
| Godiva | Table | 0.99923 | 48 pcm | 1035 | 77 pcm |
| Godiva | SVD k=5 (interp) | 0.99845 | 42 pcm | 1604 | 155 pcm |
| Godiva | GPU SVD k=5 (interp) | 0.99700 | 47 pcm | 686 | 300 pcm |

**SVD vs Table gap**: 79 pcm (rank-5 compression error, not interpolation artifact)
**GPU vs CPU SVD gap**: 145 pcm (remaining transport loop differences)
**SVD rank does NOT reduce the gap** — tested k=5,8,12, all give ~79 pcm vs Table

### Key Finding: SVD Gap is NOT Rank-Limited

| Rank | SVD-Table gap | Notes |
|------|---------------|-------|
| 5 | 79 pcm | Same as rank 12 |
| 8 | 79 pcm | Gap is constant |
| 12 | 79 pcm | Irreducible at this architecture |

The 79 pcm is the residual from log-log interpolation on SVD-reconstructed values
vs the Table's native log-log interpolation. Both use the same energy grid and
interpolation scheme — the difference is SVD's f32 basis quantization.

### Bugs Found and Fixed (this session)

1. **CPU EnergyHashTable lookup bug** (kernel.rs): Hash started scan from
   bin UPPER edge instead of LOWER edge. Returned indices up to 239 positions
   off in the resonance region. Caused all previous SVD k_eff values to be wrong
   (old k=1.00019 was bogus). Fixed by using `bins[bin-1]` as starting index.

2. **U-234 nu_bar fallback = 0** (gpu_pwr_bench.rs): GPU Godiva had nu_bar=0.0
   for U-234. Since U-234 has no nu_bar table in HDF5, this meant zero fission
   neutrons from U-234. Fixed to 2.49 (matching CPU). Impact: ~640 pcm.

3. **Inelastic two-body kinematics** (transport.cu): GPU used simplified formula
   `E_out = E*A²/(A+1)² + Q*(A+1)/A` (no mu_cm coupling). Replaced with proper
   velocity-addition kinematics matching CPU's `inelastic_scatter()`.
   Also fixed MT=91 evaporation 0.9 clamp + Q-value path.

4. **Log-log XS interpolation** (xs_provider.rs + transport.cu): SVD mode was
   stair-stepping XS at grid points. Added log-log interpolation between
   adjacent SVD reconstructions, matching OpenMC/Table scheme. Closed 44 pcm
   of the 123 pcm stair-step artifact.

5. **Angular distribution interpolation** (transport.cu): Added correlated
   sampling between energy brackets (CPU had it, GPU didn't). Verified bit-exact
   match between CPU and GPU angular dist values via diagnostic.

6. **Fission spectrum interpolation** (transport.cu): Same correlated sampling
   pattern for fission outgoing energy between incident energy brackets.

7. **(n,2n)/(n,3n) neutron banking** (transport.cu): GPU now banks 1/2 extra
   neutrons and continues primary (matching CPU collision.rs). Was treating
   these as plain inelastic with no neutron production.

8. **Reaction ordering** (transport.cu): Reordered to match CPU:
   elastic → inelastic → n2n → n3n → fission → capture (remainder).

9. **Free-gas thermal scattering** (transport.cu): Removed `A < 10` guard
   (CPU has no mass cutoff). Added angular distribution at relative energy
   in free-gas path.

## GPU Architecture

**CUDA kernel**: `gpu/cuda/transport.cu` (loaded via `include_str!`)
**Rust orchestration**: `src/gpu_transport.rs`
**Diagnostics**: `src/bin/debug_trace.rs` (CPU vs GPU data validation)

### Packed TransportParams (66 u64 fields, one device buffer)

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
| `transport_persistent` | Main: N steps/launch, 104 regs, 0 spill |
| `debug_angular_sample` | Diagnostic: angular dist CPU/GPU comparison |
| `debug_xs_reconstruct` | Diagnostic: XS value CPU/GPU comparison |

### Physics in transport_persistent

- SVD XS with log-log interpolation between grid points (rank-k FMA × 2)
- Anisotropic angular distributions (correlated interpolation between energies)
- Fission spectrum (correlated interpolation between incident energies)
- S(alpha,beta) for H1 <3.75 eV (CDF: 106 E_in, 48k E_out, 771k mu)
- URR probability tables (band sampling, multiply/absolute)
- Discrete levels (SVD per-level XS, proportional sampling, real Q-values)
- Continuum inelastic MT=91 (evaporation: T=sqrt(E*/a), a=A/8, 0.9 clamp)
- (n,2n)/(n,3n) with neutron banking (1/2 evaporation neutrons + primary continues)
- Free-gas thermal (Box-Muller target velocity, angular dist at E_rel, no mass cutoff)
- Energy-dependent nu-bar (linear interpolation on per-nuclide tables)
- Warp-level counter reduction (`__shfl_down_sync`)
- `__launch_bounds__(256, 2)`

### GPU Memory

| Data | Size |
|------|------|
| SVD basis (f32) | ~32 MB (8 PWR nuclides) |
| Discrete level basis | ~100 MB |
| Energy grids | ~2.5 MB |
| S(alpha,beta) | ~8 MB/temp |
| Angular dist + URR + nu-bar + fission CDF | ~0.5 MB |

## Files

```
rust_prototype/src/bin/godiva.rs          CPU Godiva (--mode svd|table|both --seeds N)
rust_prototype/src/bin/pwr_pincell.rs     CPU PWR (--mode svd|table|both --seeds N)
rust_prototype/src/bin/gpu_pwr_bench.rs   GPU benchmark (--geometry pwr|godiva --seeds N)
rust_prototype/src/bin/debug_trace.rs     CPU vs GPU physics diagnostic
rust_prototype/src/gpu_transport.rs       Rust GPU orchestration (packed params, upload, launch)
rust_prototype/gpu/cuda/transport.cu      CUDA kernels (persistent transport + diagnostics)
rust_prototype/src/transport/simulate.rs  CPU transport loop (surface + delta tracking)
rust_prototype/src/transport/xs_provider.rs  SVD + Table XS providers (with log-log interp)
rust_prototype/src/hdf5_reader.rs         HDF5 reader (XS, angular, URR, thermal, nu-bar)
rust_prototype/src/thermal.rs             S(alpha,beta) sampling
rust_prototype/src/kernel.rs              SVD kernel (f32 basis, hash lookup, Ducru interp)
paper/svd_cross_section_compression.tex   Manuscript
scripts/paper_openmc_benchmark.py         Multi-seed OpenMC runner
run_paper_full.ps1                        Full benchmark script (CPU + GPU + scaling)
```

## Next Steps

1. Investigate remaining 145 pcm GPU-CPU gap (transport loop structure)
2. Rank sweep (k=1..6) for paper accuracy/speed tradeoff curve
3. Rerun PWR pin cell benchmarks with all fixes
4. Run OpenMC 10-seed reference via WSL (verify Table mode matches)
5. Update paper tables + appendix with corrected numbers
6. OpenCL port (gpu/opencl/)
