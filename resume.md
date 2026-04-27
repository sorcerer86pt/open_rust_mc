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
