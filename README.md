# open_rust_mc

A pure-Rust Monte Carlo neutron transport engine with pointwise-table
and SVD cross-section providers on both CPU and GPU (CUDA). Reads
OpenMC HDF5 nuclear data directly (no C dependency), runs k-eigenvalue
simulations end-to-end, and is validated against OpenMC 0.15.3 on two
reference benchmarks.

## Headline: Four providers, two benchmarks, |Δ| ≤ 51 pcm vs OpenMC

The engine ships four cross-section backends behind the same interface:

| Provider | Where | What |
|----------|-------|------|
| CPU table | `table.rs` | OpenMC-style pointwise lookup (binary search) |
| CPU SVD | `kernel.rs` | FMA reconstruction from truncated basis |
| GPU pointwise | `gpu.rs` (CUDA) | Pointwise table on device |
| GPU SVD | `gpu.rs` (CUDA) | SVD reconstruction on device |

All four agree with OpenMC 0.15.3 to **|Δ| ≤ 51 pcm** on the PWR pin
cell, and cluster **325–362 pcm** above the ICSBEP Godiva value — which
is the known ENDF/B-VII.1 library bias, not an engine bias (OpenMC with
the same library is in the same place). Tracing the original 78–127 pcm
offset found three transport-level bugs; post-correction numbers are
below.

### PWR pin cell (9 nuclides, large-L3 system: Ryzen 9800X3D (96 MB L3) + RTX 3080)

10 seeds × 150 batches × 50k particles, 20 inactive:

| Provider | rank | k∞ | Δ vs OpenMC (pcm) | ns/particle |
|----------|:---:|------:|:---:|------:|
| CPU table | — | 1.32793 | +23 | 21 316 |
| **CPU SVD** | **5** | **1.32745** | **−25** | **15 511** |
| GPU pointwise | — | 1.32821 | +51 | 5 967 |
| GPU SVD | 5 | 1.32762 | −8 | 7 754 |

OpenMC 0.15.3 reference: k∞ = 1.32770 ± 0.00009.

### Godiva (HEU-MET-FAST-001, same large-L3 system, same config)

| Provider | rank | k_eff | Δ vs experiment (pcm) | ns/particle |
|----------|:---:|------:|:---:|------:|
| CPU table | — | 1.00330 | +330 | 1 207 |
| **CPU SVD** | **5** | **1.00325** | **+325** | **846** |
| GPU pointwise | — | 1.00339 | +339 | 302 |
| GPU SVD | 5 | 1.00362 | +362 | 378 |

The +325 to +362 pcm cluster matches the published ENDF/B-VII.1 Godiva
bias; all four providers reproduce the library-level bias and do not
add to it.

## CUDA kernel

`gpu.rs` implements both backends on device via `cudarc`:

- **GPU pointwise** is the fastest path overall (5 967 ns/p on PWR
  pin cell, 302 ns/p on Godiva — 3.6× and 4.0× the large-L3 CPU table).
- **GPU SVD** is 1.25–1.30× slower than GPU pointwise. The clad
  nuclides (Zr isotopes) lack an MT=4 block, so their total inelastic
  must be synthesised from 13 discrete-level SVD kernels per lookup —
  a CPU win that becomes a GPU draw.
- Bit-parity with CPU SVD at machine precision (force-SVD parity test
  passes across seeds).

See `cuda_bench/svd_gpu_bench.cu` for the standalone reconstruction
benchmark (~8.7× over CPU on RTX A1000; better on higher-end GPUs).

## Physics implemented

- Continuous-energy neutron transport, k-eigenvalue power iteration
- Energy-dependent ν̄ (prompt + delayed) from HDF5
- Anisotropic scattering (tabular μ/CDF, per-level angular on CPU and GPU)
- Data-driven fission outgoing-energy spectrum
- Discrete inelastic levels MT=51–91 with real Q-values
- Continuum inelastic MT=91 (evaporation spectrum)
- URR probability tables (multiply-smooth + absolute modes)
- (n,2n) MT=16, (n,3n) MT=17
- Free-gas thermal scattering (Maxwell–Boltzmann target sampling)
- S(α,β) thermal scattering for H in H₂O (continuous + discrete,
  Bragg edges, Debye–Waller)
- Rayon parallel transport, CSG geometry with BVH, auto-selected
  surface vs delta tracking

## The SVD companion: compression study and paper

The engine exists partly to host a head-to-head test of SVD-compressed
cross sections against a pointwise baseline *in the same transport
loop*. Findings are reported honestly — positive, mixed, and negative —
in `paper/main.pdf` (“SVD-Compressed Cross Sections in Monte Carlo
Neutron Transport: An implementation-led benchmark with positive,
mixed, and negative results”).

**Positive.** The singular spectrum of ENDF/B-VII.1 log σ(E,T) matrices
decays rapidly; 43 of 47 non-redundant U-235 channels are rank-one to
machine precision. Kernel-level, rank-6 SVD reconstructs at 44 ns vs
91 ns for a binary-searched pointwise table (2.0×). In-engine, CPU SVD
rank-5 runs 1.37×–1.90× faster than the CPU table on PWR pin cell and
1.43× faster on Godiva, while staying within 13–47 pcm of the table’s
own k∞.

**Mixed.** The CPU trade-off inverts on GPU for the PWR pin cell
(GPU pointwise 1.30× faster than GPU SVD). A WMP+SVD hybrid matches
k∞ to 86 pcm but runs 2.06× slower than CPU SVD — the Humlicek W4
Faddeeva evaluation is ~15× more expensive per lookup than the SVD FMA.

**Negative.** The WMP representation alone packs 9 nuclides into
1.37 MB against 177.7 MB for a hypothetical 4-channel × 6-temperature
pointwise table (132.9× representation ratio) — but the full engine
still has to carry SVD kernels for discrete inelastic levels, total
inelastic, (n,2n), (n,3n), and URR residues (none covered by the
public MIT WMP library). Measured engine-scale reduction of hybrid
over full-SVD is only 1.06×, and the hybrid engine actually carries
5× *more* memory than the pointwise baseline. Both numbers are real;
they measure different things.

## Repository layout

```
rust_prototype/
  src/
    physics/       Collision processing, scattering kinematics
    transport/     Particle tracking, k-eigenvalue solver, XS providers
    geometry/      CSG geometry, surfaces, BVH, lattices
    hdf5_reader.rs Pure-Rust HDF5 reader with single-pass caching
    thermal.rs     S(α,β) data structures + sampling
    kernel.rs      CPU SVD reconstruction hot path
    gpu.rs         CUDA backend (pointwise + SVD, feature-gated)
    table.rs       Pointwise table (OpenMC-style baseline)
  src/bin/
    godiva.rs      Godiva benchmark (--mode svd|table|both, --seeds)
    pwr_pincell.rs PWR pin cell benchmark (9 nuclides, 3 materials)
    gpu_bench.rs   GPU reconstruction microbenchmark
    bench_mem.rs   Memory comparison tool
cuda_bench/        Standalone CUDA SVD reconstruction kernel
scripts/           Python pipeline (HDF5 extraction, OpenMC cross-checks)
paper/             LaTeX manuscript + figures
```

## Quick start

```bash
cd rust_prototype
cargo build --release

# Nuclear data: ENDF/B-VII.1 HDF5 from https://openmc.org/data/
DATA=../data/endfb-vii.1-hdf5/neutron

# Smoke test (~10 s)
cargo run --release --bin godiva -- $DATA --mode both --rank 5 \
  --batches 15 --inactive 5 --particles 2000

# Godiva, 10 seeds, publication-grade (~40 min CPU)
cargo run --release --bin godiva -- $DATA --mode both --rank 5 \
  --batches 150 --inactive 20 --particles 1000000 --seeds 10

# PWR pin cell, 10 seeds
cargo run --release --bin pwr_pincell -- $DATA --mode both --rank 5 \
  --batches 150 --inactive 20 --particles 50000 --seeds 10

# GPU (requires CUDA toolkit)
cargo run --release --features cuda --bin gpu_bench -- $DATA \
  --rank 5 --particles 1000000

# Unit tests (36)
cargo test --lib
```

Detailed reproduction instructions and expected outputs:
[BENCHMARKS.md](BENCHMARKS.md).

## Hardware used for paper results

Two systems that differ in CPU L3 capacity and GPU class:

- **Large-L3**: Ryzen 9800X3D (8-core, 96 MB L3 3D V-cache) +
  RTX 3080 (68 SMs, 10 GB, 5 MB L2), 64 GB DDR5-6000
- **Small-L3** (Dell Precision 5570): Intel Core i7-12800H
  (12th-gen Alder Lake, 6 P + 8 E cores, 24 MB L3 Smart
  Cache) + RTX A1000 (16 SMs, 1.5 MB L2), 32 GB DDR4

The 4× L3 ratio is what drives the hardware-dependent
spread in CPU SVD speedup (1.37× on large-L3, 1.90× on
small-L3). Nuclear data: ENDF/B-VII.1 HDF5 (5.8 GB).
OpenMC reference: 0.15.3 (WSL Ubuntu 24.04).

## References

- Ducru et al., “Kernel reconstruction methods for Doppler broadening,”
  J. Comput. Phys. 335 (2017) 535–557
- Tramm et al., “Performance Portable MC Particle Transport on Intel,
  NVIDIA, and AMD GPUs,” EPJ Web Conf. 302 (2024) 04010
- Brown, “New Hash-Based Energy Lookup Algorithm for MC Codes,”
  Trans. ANS 111 (2014) 659–662
- Romano et al., “OpenMC: A state-of-the-art Monte Carlo code for
  research and development,” Ann. Nucl. Energy 82 (2015) 90–97

## Paper

`paper/main.pdf` — *SVD-Compressed Cross Sections in Monte Carlo
Neutron Transport: An implementation-led benchmark with positive,
mixed, and negative results.*

## License

MIT
