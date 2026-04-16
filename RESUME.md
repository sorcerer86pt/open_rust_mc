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

### 10-Seed CPU Results (150 batches, 50k particles)

| Benchmark | Mode | k | sigma | ns/p | sigma |
|-----------|------|---|-------|------|-------|
| Godiva | Table | 0.99923 | 48 pcm | 1009 | 59 |
| Godiva | SVD k=5 | 1.00019 | 35 pcm | 699 | 47 |
| PWR | Table | 1.35471 | 45 pcm | 22157 | 804 |
| PWR | SVD k=5 | 1.35675 | 42 pcm | 15417 | 1607 |

### GPU Results (RTX A1000 laptop, 4GB)

| Benchmark | k | sigma | ns/p | Gap vs CPU |
|-----------|---|-------|------|-----------|
| Godiva | 0.99160 | 60 pcm | 743 | 84 pcm |
| PWR | 1.37534 | 19 pcm | 16058 | ~200 pcm |

GPU scaling (PWR, 3 seeds): 5k→32k, 10k→21k, 20k→15k, 50k→11k, 100k→10k, 200k→9k ns/p

### Physics Gaps (GPU vs CPU)

- **Godiva 84 pcm**: continuum inelastic MT=91 evaporation model approximate
- **PWR 200 pcm**: same + URR `apply_urr` offset: `base = off*n_b + ie*n_b` should be `(off+ie)*n_b`
- Target: <10 pcm on both

## GPU Architecture

**CUDA kernel**: `gpu/cuda/transport.cu` (loaded via `include_str!`)
**Rust orchestration**: `src/gpu_transport.rs` (~780 lines)

### Packed TransportParams (66 u64 fields, one device buffer)

```cuda
typedef const unsigned long long* Params;
#define PTR_F(p, idx)    ((const float*)  (p)[(idx)])
#define PTR_D(p, idx)    ((const double*) (p)[(idx)])
#define PTR_I(p, idx)    ((const int*)    (p)[(idx)])
#define SCALAR_I(p, idx) ((int)(p)[(idx)])
#define SCALAR_D(p, idx) __longlong_as_double((long long)(p)[(idx)])
```

Indices: `P_BASIS=0` through `P_GEOM_TYPE=65`. Rust packs `Vec<u64>`, uploads once.
Device pointers extracted via `DevicePtr::device_ptr()`. Scalars cast. Doubles via `to_bits()`.

### Kernels

| Kernel | Purpose |
|--------|---------|
| `init_source` | Initialize particles from source bank |
| `compact_alive` | Atomic compaction of alive indices |
| `energy_bin_count/scatter` | 256-bin sort for coalesced SVD access |
| `transport_persistent` | Main: N steps/launch, registers, warp reductions |

### Physics in transport_persistent

- SVD XS reconstruct (rank-k FMA, `__ldg`, `__restrict__`)
- S(alpha,beta) for H1 <3.75 eV (CDF: 106 E_in, 48k E_out, 771k mu)
- URR probability tables (band sampling, multiply/absolute)
- Anisotropic angular distributions (CDF sampling from HDF5 tables)
- Discrete levels (SVD per-level XS, proportional sampling, real Q-values)
- Continuum inelastic MT=91 (evaporation: T=sqrt(E*/a), a=A/8)
- Free-gas thermal (Box-Muller target velocity, E<400kT, A<10)
- Energy-dependent nu-bar (linear interpolation on 79-point table)
- Fission spectrum (tabulated CDF from HDF5)
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

## HDF5 Data Format

ENDF/B-VII.1 from openmc.org. 444 nuclides, 5.8 GB.

```
U235.h5:
  /U235/energy/{temp}/           [N_E f64] sorted eV
  /U235/reactions/
    reaction_002/{temp}/xs       Elastic       [N_E f64]
    reaction_004/{temp}/xs       Inelastic     [N_E f64]
    reaction_016/{temp}/xs       (n,2n)        [N_E f64]
    reaction_018/{temp}/xs       Fission       [N_E f64]
    reaction_018/product_0/
      yield/{energy,yield}       Nu-bar table  [N_nu f64 each]
      energy_distribution/
        energy_out [5, N_out]    {E_out, PDF, CDF, mu_interp, mu_offsets}
          attr: offsets [N_inc]
        mu [3, N_mu]             {mu, PDF, CDF}
    reaction_051-091/{temp}/xs   Discrete levels (attr: Q_value f64)
    reaction_102/{temp}/xs       Capture       [N_E f64]
    reaction_002/product_0/angle/
      energy [N_ang f64]         Angular dist grid
      distribution/              Tabular mu/CDF (attr: center_of_mass)
  /U235/urr/{temp}/
    energies [N_urr f64]
    table [N_urr, 6, N_bands]    {cumprob, total, el, fis, cap, heat}
      attr: N_bands, multiply_smooth

c_H_in_H2O.h5:
  kTs [N_temp f64]
  inelastic/{temp}/
    energy_out [5, N_eout]       attr: offsets [N_inc]
    mu [3, N_mu]
  elastic/{temp}/                Optional (Bragg, Debye-Waller)
```

Sizes: U235=83k pts, U238=186k pts, H1=590 pts, c_H_in_H2O=9 temps x ~50k E_out x ~770k mu

## SVD Math

```
log10(sigma) = B[i,:] . c    where B = U_k * Sigma_k (f32), c = V_k^T[:,t] (f64)
sigma = exp2(log10(sigma) * log2(10))
```

GPU: sequential B row access (coalesced), c in shared mem, pure FMA, zero divergence.
Replaces O(log N) binary search on 186k-point table with O(k=5) sequential reads.

## Files

```
rust_prototype/src/bin/godiva.rs          CPU Godiva (--mode svd|table|both --seeds N)
rust_prototype/src/bin/pwr_pincell.rs     CPU PWR (--mode svd|table|both --seeds N)
rust_prototype/src/bin/gpu_pwr_bench.rs   GPU benchmark (--geometry pwr|godiva --seeds N)
rust_prototype/src/gpu_transport.rs       Rust GPU orchestration (packed params, upload, launch)
rust_prototype/gpu/cuda/transport.cu      CUDA kernels (persistent transport + utilities)
rust_prototype/src/transport/simulate.rs  CPU transport loop (surface + delta tracking)
rust_prototype/src/transport/xs_provider.rs  SVD + Table XS providers
rust_prototype/src/hdf5_reader.rs         HDF5 reader (XS, angular, URR, thermal, nu-bar)
rust_prototype/src/thermal.rs             S(alpha,beta) sampling
rust_prototype/src/kernel.rs              SVD kernel (f32 basis, hash lookup, Ducru interp)
paper/svd_cross_section_compression.tex   22-page manuscript
scripts/paper_openmc_benchmark.py         Multi-seed OpenMC runner
run_paper_full.ps1                        Full benchmark script (CPU + GPU + scaling)
```

## Next Steps

1. Fix GPU continuum inelastic + URR offset → close gap to <10 pcm
2. Rerun 10-seed GPU benchmarks (Godiva + PWR)
3. Run OpenMC 10-seed reference via WSL
4. Update paper tables + appendix with final numbers
5. OpenCL port (gpu/opencl/)
6. HPC benchmarking on dedicated cluster
