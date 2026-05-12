# Technical Specification: Rust Nuclear Simulation Framework

This spec covers two work items that are not negotiable:

1. **Math migration** — every pure math / pure physics kernel moves out
   of `open_rust_mc` and into `Rust-MC-SimulationLib`. The transport
   orchestrator depends on the math lib, never the other way around.
2. **Big compute layer** — the runtime executes as a three-tier
   hybrid: MPI between nodes, Rayon between cores inside a node,
   CUDA inside the node when a device is present.

The remaining sections (physics, depletion, validation) describe the
existing engine and only matter inasmuch as they constrain those two
items.

---

## 1. Layered architecture

The ecosystem is split into two crates so the math kernels can be
reused (and re-validated) independently of the transport orchestrator.

### 1.1 `Rust-MC-SimulationLib` — math + physics foundation

The math/data engine. **Every kernel listed here owns its own tests,
benchmarks, and Python bindings inside the lib crate — `open_rust_mc`
imports them, never reimplements them.**

- **SVD / CP-PARAFAC compression**: cache-resident rank-k cross-
  section reconstruction (`kernel.rs`, `decompose.rs`,
  `cp_decompose.rs`).
- **Ducru reconstruction**: off-grid temperature interpolation with
  the partition-of-unity correction.
- **CRAM solver (modular)**: CRAM-16 / CRAM-48 engines for the
  Bateman equations, including the complex linear algebra and
  partial-pivot LU. Currently being lifted out of
  `open_rust_mc::depletion::cram` and `matrix.rs`.
- **Adjoint kernels**: exact samplers for adjoint Compton and
  adjoint elastic neutrons (used by CADIS).
- **CDF sampling primitives**: log-decimated CDF sampling for level
  and channel selection (shared between CPU and CUDA paths).

### 1.2 `open_rust_mc` — transport + geometry orchestrator

The continuous-energy neutron–photon–electron transport engine.

- **CSG geometry**: quadratic-surface navigation with BVH
  acceleration.
- **Recursive lattices**: hexagonal and rectangular lattices on
  both CPU and GPU.
- **Hybrid backend**: parallel CPU execution via `rayon`, GPU
  execution via `cuda` (NVRTC-compiled NVIDIA kernels).

---

## 2. Big compute layer (MPI + Rayon + CUDA)

To break the memory and throughput ceilings on clusters, the engine
runs as a three-tier hybrid. This is the headline target of the
current work; everything else (physics, depletion) already exists
and is being adapted to fit it.

### 2.1 Inter-node tier — MPI

- **Global orchestration**: power-iteration (k-eigenvalue) cycles
  synchronised across cluster nodes.
- **Global fission bank**: end-of-cycle aggregation and
  redistribution of the neutron source via `MPI_Allreduce` (and
  `MPI_Scatterv` when the global bank is gathered for rebalancing).
- **Distributed tallies**: global reduction of mesh fluxes and
  reaction rates to minimise per-node memory footprint.

### 2.2 Intra-node tier — Rayon + CUDA

- **Thread parallelism**: local transport is scaled across every
  available core via `rayon`.
- **Hardware acceleration**: transport and physics kernels (Watt χ
  spectrum, EADL cascades, photon kernels, recursive geometry walk)
  are dispatched to CUDA when a device is present.
- **SVD efficiency**: leverages compression to fit full nuclide
  libraries on RAM-limited nodes (up to 1.35× more efficient at
  off-library temperatures).

### 2.3 Required cross-tier invariants

- Bit-parity (or documented sub-pcm bias) between CPU and CUDA
  results on every kernel that exists in both backends.
- A single `EigenvalueRunner` trait above the MPI / Rayon / CUDA
  split so binaries don't branch on the backend.
- Deterministic seeding: per-rank, per-thread, per-particle seeds
  derived from a single root seed, reproducible across reruns and
  across MPI sizes for fixed input.

---

## 3. Physics and variance reduction

- **Coupled N–P–E transport**: full neutron–photon–electron
  pipeline with energy deposition validated against γ-heating.
- **Random-ray (TRRM)**: multigroup forward/adjoint solver with
  the "immortal-ray" mode for stable importance-map generation.
- **FW-CADIS**: adjoint-flux–driven weight windows for variance
  reduction on deep-penetration problems.
- **Next-Event Estimator (NEE)**: deterministic estimator corrected
  for electron binding via S(x,Z) and azimuthal corrections.
- **URR equivalence**: Stoker-Weiss / NJOY equivalence theory with
  Carlvik-Pellaud Dancoff factors for square lattices.

---

## 4. Depletion and isotopic evolution

- **Predictor-corrector**: CE/LI scheme with EOC-flux estimation.
- **On-the-fly spectrum collapse**: chain cross sections collapsed
  at run time using the converged spectrum, reaching 0.77× the
  OpenMC reference performance.
- **Python FFI**: CRAM solvers and atomic-density management
  exposed via PyO3 for coupling with external codes.

---

## 5. Validation and regression (ICSBEP)

- **ICSBEP-on-CUDA harness**: automation infrastructure to run the
  nuclear-safety benchmark suite directly on GPU.
- **Physics fidelity**: validated against Godiva (HMF-001) and PWR
  pin cells (agreement within 51 pcm).
- **Bit parity**: strict CPU/GPU consistency for SVD reconstruction
  and the physics samplers; cross-rank MPI parity for the
  power-iteration loop.
