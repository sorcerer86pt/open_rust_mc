# open_rust_mc

[![Latest release](https://img.shields.io/badge/release-v0.4.0-blue)](https://github.com/sorcerer86pt/open_rust_mc/releases/latest)

A pure-Rust continuous-energy Monte Carlo neutron **and photon**
transport engine. Reads OpenMC HDF5 nuclear data directly (no C
dependency), runs k-eigenvalue, fixed-source, time-dependent
point-kinetics, and burnup simulations end-to-end on CPU (rayon) or
CUDA GPU, and is validated against OpenMC on Godiva, PWR pin cell,
PWR-actinide depletion, and γ-heating. A coupled neutron-photon
pipeline drives PWR pin cell γ-heating directly off the
ENDF/B-VII.1 HDF5 library; a multigroup random-ray (TRRM) solver
ships forward + adjoint + immortal-ray modes and feeds a
FW-CADIS variance-reduction pipeline with measured FOM gains
(2.2× at 100 cm and 4.3× at 200 cm of water shielding vs analog).

352 lib tests + 10 integration tests pass on each push (`cargo test`),
including doctests on the photon stack.

The engine is designed as a research vehicle for studying cross-section
representation: it ships **four interchangeable cross-section providers**
behind a common `XsProvider` trait so the same geometry, same particle
transport loop, and same physics kernels can be measured against
different data layouts. Detailed numerical results live in the
accompanying paper — this README describes the engine itself.

## Cross-section providers

All four implement the same `XsProvider` trait and are selectable at
runtime via `--mode`:

| Mode | Provider | Implementation | What it does |
|------|----------|---------------|--------------|
| `table` | Pointwise table | `src/table.rs` | OpenMC-style binary search + log-log interpolation, per-reaction pointwise arrays |
| `svd` | Truncated SVD | `src/kernel.rs` | Rank-*k* reconstruction from a pre-multiplied basis, one FMA sequence per lookup |
| `hybrid` | SVD + WMP | `src/transport/hybrid_xs.rs` | SVD everywhere, overridden by Windowed Multipole (pole/residue + Faddeeva) inside each nuclide's resolved resonance window |
| `wmp` | ACE + WMP | `src/transport/hybrid_xs.rs` | Pointwise table everywhere, overridden by WMP in the RRR — industry low-memory baseline |

A three- or four-way "honesty test" mode (`--mode both` / `--mode all`)
runs every provider back-to-back on the same geometry and prints a
head-to-head comparison.

**Where SVD wins on memory:** SVD is monotonically *larger* than the
pointwise table at every rank (1–7) in on-library 9-nuclide PWR and
3-nuclide Godiva measurements (`outputs/sweep_svd_wins.csv`,
`outputs/full_test_run/10_pwr_all_rank5.txt`). The actual memory
win shows up at off-library temperatures, where stochastic
pseudo-interpolation forces Table to load two columns (Godiva
450 K: SVD 164 MB vs Table 222 MB at rank 5 = 1.35×). Compression
headlines computed from per-nuclide representation byte counts —
e.g. paper Table 5's 132.9× WMP-vs-pointwise ratio — are
representation-level metrics; the in-engine working-set is
4.8–5.2× *larger* than the pointwise baseline once geometry, BVH,
energy grids, and discrete-level data are included.

## Off-library temperature handling

The pointwise and hybrid providers implement OpenMC-style **stochastic
pseudo-interpolation** (`src/table.rs::StochTempTable`): when the
operating temperature lies between two library columns, both columns
are loaded and the choice is drawn per nuclide-collision. Partial
channels (elastic, inelastic, fission, capture, …) share the same
draw so the sampled cross sections stay thermodynamically consistent
within a single collision.

The SVD provider uses **partition-of-unity 3-point Ducru kernel
reconstruction** for off-library temperatures
(`src/transport/xs_provider.rs::ducru_unity_weights`): the nearest
three library columns to the target temperature, raw Ducru 2017
Eq. 31 weights on the 3×3 subset, then unity-normalized so the
weighted sum of log-σ values does not introduce a multiplicative
gain error on resonance peaks.

CLI flags that drive this path:

- `--target-temp <K>` — run at an arbitrary operating temperature
- `--target-temp-offset <K>` — shift every nuclide's library temp by *N* K (PWR)
- `--fuel-offset`, `--mod-offset` — isolate fuel- vs moderator-side effects (PWR)
- `--discrete-rank <N>` — override SVD rank for MT=51–91 (discrete levels are weakly T-dependent; rank 1 typically suffices)

## Neutron physics

- k-eigenvalue power iteration with Shannon-entropy convergence diagnostic
- Energy-dependent ν̄ (prompt + delayed) read from HDF5
- Anisotropic scattering from tabulated μ/CDF (stochastic bin selection)
- Data-driven fission outgoing-energy spectrum
- Discrete inelastic levels MT=51–91 with exact Q-values and two-body kinematics
- Continuum inelastic MT=91 (evaporation spectrum)
- URR probability tables (both multiply-smooth and absolute modes)
- (n,2n) MT=16, (n,3n) MT=17
- Free-gas thermal scattering (Maxwell–Boltzmann target velocity sampling)
- S(α,β) thermal scattering for H in H₂O and **D in D₂O** (continuous + discrete inelastic, Bragg edges, Debye–Waller incoherent elastic), via a `ThermalLibrary` registry covering graphite, ZrH (TRIGA), Be / BeO, polyethylene, benzene, α-quartz, bound O / U in UO₂, methane, ortho/para H₂ and D₂, Al-27, Fe-56
- **`NuclideLibrary`** — ZAID-keyed registry over structural / actinide / FP nuclides (B-10/11, Zr-90–96, Fe-54–58, U-233 → Cm-247, plus FP poisons I-135 / Xe-135 / Cs-135 / Pm-149 / Sm-149 / Gd-155 / Gd-157) with on-demand HDF5 metadata (AWR + temperature columns) and nearest-temperature selection
- **URR equivalence theory** (Stoker-Weiss / NJOY rational form): Carlvik-Pellaud Dancoff factor for square pin lattices, per-cell `(C, l̄)` cache, σ_eff = σ_∞·σ_0/(σ_0+σ_e). `pwr_pincell --urr-equivalence` toggles it
- **Track-length k-eff** estimator under Woodcock delta tracking (in addition to collision estimator under both surface and delta tracking)
- Auto-detected delta tracking when material contrast is high; surface tracking otherwise
- Tallies: surface currents (J⁺/J⁻ by direction), Cartesian mesh flux (Amanatides-Woo), under either tracking mode
- HDF5 statepoint write / read / restart, validated on chained 3-hop restarts across 3×3 / 5×5 / 7×7 minicores

## Photon physics

Full four-channel photon transport on per-element OpenMC HDF5 data
(`photon/*.h5`):

- **Compton** — free Klein–Nishina with `S(x, Z)/Z` bound-electron
  rejection and Hartree–Fock Compton-profile Doppler broadening
- **Photoelectric** absorption with full EADL atomic relaxation
  cascade (fluorescence + Auger)
- **Pair production** — Bethe–Heitler nuclear + electron-field +
  in-flight positron annihilation
- **Coherent** (Rayleigh) scattering via tabulated form factors

Drivers:

- `transport_history` — closure-based single-material driver (used
  by the Cs-137 pulse-height and ANSI/ANS-6.6.1 buildup benchmarks)
- `transport_history_csg` — CSG-aware driver with per-cell
  `PhotonMaterial`, reusing the same `Surface`/`Cell`/`Region`
  geometry + `ray::trace_step` the neutron loop uses

Full condensed-history electron transport: track-integrated step-
and-deposit with non-uniform Bethe-Bloch `dE/dx`, Highland multiple-
scattering angular spread (per-cell radiation length `X₀`), and
single-event Seltzer-Berger bremsstrahlung secondaries banked back
into the photon loop. Reflective-lattice folding keeps reflective-BC
pin cells consistent with the neutron loop. Replaces the older
Katz-Penfold CSDA midrange displacement; the "He gap deposition
artefact" goes from 1.5 % to 0 %.

## Coupled neutron-photon transport

The neutron loop tallies a `PhotonSourceEvent { cell, pos, E_γ, MT }`
at every capture (MT=102), fission (MT=18), (n,p) / (n,α) (MT=103 /
107, threshold-gated), and inelastic (MT=4 via discrete-level
Q-value) collision. γ multiplicities and outgoing energies come from
the HDF5 `reactions/reaction_{mt}/product_{N}` tree with
`particle="photon"` — the same `ContinuousTabular` reader path used
for fission-neutron outgoing energies.

The `pwr_gamma_heating` binary runs the pipeline end-to-end on a
standard PWR pin cell (3.1 % UO₂ / Zr-4 / H₂O, 1.26 cm pitch,
reflective lattice): a short neutron k-eigenvalue, aggregate the
event bank across active batches, then transport 200 k photon
histories and bin per-collision deposits by containing cell.

Run conditions: 150 batches (20 inactive + 130 active) × 50 000
neutrons/batch + 200 000 photon histories, single seed, full
electron transport on by default; ~5 min wall on the test box
described under "Benchmarks" below. OpenMC reference is 0.15.3 on
identical geometry / same library, recorded in
`outputs/openmc_pwr_gamma_heating.json`.

| region | open_rust_mc | OpenMC 0.15.3 |
|--------|-------------:|--------------:|
| fuel   | **84.12 %**  | ~85 %         |
| gap    | 0.00 %       | 0 %           |
| clad   | 9.81 %       | ~9 %          |
| water  | 5.72 %       | ~6 %          |
| escape | 0.00 %       | 0             |
| sum    | 99.65 %      | (EADL leak)   |

Bremsstrahlung fires self-consistently: this run emits 2 312 brems γ
totalling 7.43 × 10⁸ eV (0.353 % of source energy), each banked back
into the photon transport phase.

## Depletion (burnup)

`src/depletion/` implements a full burnup pipeline:

- **CRAM-16 / CRAM-48** matrix exponential (IPF form, Pusa 2016
  poles & residues from OpenMC's canonical source) with dense
  complex Gaussian elimination + partial pivoting
- `DepletionChain` — decay constants, branches, per-(parent, MT)
  one-group reaction XS with default ENDF yield inference for
  (n,γ) / (n,2n) / (n,3n) / (n,p) / (n,α)
- `chain_io` JSON loader with three-way `yields` semantics:
  omitted → default ENDF, `{}` → pure removal, explicit map → use it
- Two shipped chains: `chains/partial_xe.json` (4 nuclides — Xe
  poisoning) and `chains/pwr_actinides.json` (17 nuclides —
  actinide buildup + dominant FP poisons)
- **CE/LI predictor-corrector** with **fresh-corrector**: clones
  materials, runs a second eigenvalue at the predicted composition
  for the EOC flux estimate, then CRAM with the averaged matrix
- **On-the-fly chain-XS spectrum collapse** (`fd530d0`) — collapses
  the chain XS at run-time using the converged transport spectrum
  rather than a pre-baked one-group library; closes most of the
  9× gap to OpenMC depletion to **0.77×**
- `BurnupMapping` table-driven walker that pushes chain-evolved
  atom densities back into transport materials

Drivers: `deplete_demo` (constant-flux Xe equilibrium — matches
analytical to 1e-4 relative) and `deplete_pwr` (full transport
feedback). Python API: `Chain.from_file`, `CramOrder.{Order16,
Order48}`, `cram(matrix, n0, order)`, `deplete_constant_flux`,
`deplete_with_flux_callback` (FFI-exception-safe Python `flux_at`
closure), `Material.set_atom_density` / `atom_density_of`.

## Time-dependent point kinetics

`src/transport/kinetics.rs` — point-kinetics with 6-group delayed-
neutron precursors, A-stable Crank-Nicolson 7×7 ODE solver. Keepin /
Hetrick-Roberts 6-group constants for U-235 thermal, U-238 fast,
Pu-239 thermal fission shipped as constants; `blend()` combines
nuclide contributions for mixed cores. Closed-form `prompt_jump_ratio`
and `inhour_period` (Newton on the inhour equation) for analytic
cross-check. Late-time period at ρ = 50 ¢ matches inhour to −0.37 %.
The `point_kinetics_demo` binary runs step / ramp / scram reactivity
profiles and emits CSV.

## Adjoint Monte Carlo (real continuous-energy)

The first piece of a continuous-energy CADIS pipeline. No surrogate /
"lite" proxies — these are exact transposed kernels with rejection
ceilings empirically validated against the analytic conditional
density.

- **Adjoint Compton** (`photon::compton::adjoint_compton_scatter`) —
  inverted Klein-Nishina sampler (Wagner-Haghighat 1998 /
  Lewis-Miller §10.3). Given post-collision `E_out`, returns the
  pre-collision `E_in` and scattering cosine μ from the transposed
  KN kernel. Rejection ceiling is the analytic supremum
  `2π·r_e²`; conditional density at fixed `E_out` matches analytic
  `KN_dcs/dμ(E_in, μ_kin)` to χ²_red < 2.5 across 25 bins from 200 k
  samples
- **Adjoint photon slab walker** (`transport::adjoint_photon`) —
  composes adjoint Compton with self-adjoint Rayleigh and
  photoelectric / pair termination ("absorption-as-source") into a
  track-length tally on a `(z, E)` mesh. Output is an
  `ImportanceMap` directly consumable by `WeightWindow::from_flux`
- **Adjoint elastic neutron** (`transport::adjoint_neutron`) —
  s-wave isotropic-CM elastic adjoint kernel, log-uniform `E_in`
  sampling on `[E_out, E_out/α]`. Validated by forward-then-adjoint
  round trip and χ² shape tests on H/C

## Random-ray transport (TRRM)

`src/random_ray/` — multigroup forward + adjoint Tramm 2018-style
flat-source TRRM with the Tramm & Siegel 2021 immortal-ray
persistent-state variant.

- `MgxsLibrary` / `ScatterMatrix` with shared storage for forward
  (Σ_{s,g→g'}) and adjoint (transposed) lookups; χ-normalisation
  + Σ_t-positivity checked at construction
- `FsrMesh` — `Cartesian` (uniform voxel grid, O(1) `fsr_at`) or
  `Cell` (one FSR per `(deepest cell, lattice element)` key, with
  analytic *or* stochastic per-FSR volumes from track lengths)
- `solve_segment` — analytic MoC ODE step
  `ψ_out = ψ_in·e^{-τ} + (q/Σ_t)(1−e^{-τ})` plus track-length
  `l·ψ_avg`. Numerically stable `(1−e^{-τ})/τ` series for τ→0
- `RandomRaySolver` — uniform AABB × isotropic ray sampler,
  dead-zone + active-zone phasing, BC-aware (vacuum kills mortal /
  reflects-with-zero immortal, reflective specular, transmission
  continues), source iteration with k-power-method update,
  `AdjointFlag::{Forward, Adjoint}` switch, `cfg.immortal: bool` for
  persistent-ray mode

Drivers: `rr_pincell` (2-group cell-based UO₂ + water reflective pin
cell, ~12 s wall) and `rr_cadis_slab` (slab-shaped CLI that produces
the JSON `shield_slab --cadis-load` consumes).

Sibling driver `adjoint_photon_cadis_slab` runs the full
**continuous-energy** adjoint photon walker on the same slab and
writes the same `CadisMap` schema. The CE walker tallies analog
adjoint flux track-length per `(z, E)` bin; on the canonical 1 MeV
beam-on-water benchmark the resulting `ψ̂*(z)` is roughly symmetric
with a mid-slab peak (max/min ≈ 1.5×), producing transport runs
**bit-identical to analog** under the default ratio-5 WW band — the
gradient is too shallow to drive splitting / roulette. RR-CADIS
remains the recommended map source for this benchmark; the CE walker
is wired so future work can switch to a contribution-function tally
or 3D `(z, E, μ_z)` WWs (see `outputs/ce_adjoint_cadis_fom.txt`).

## Variance reduction (FW-CADIS)

`shield_slab` benchmark — fixed-source γ transmission through a
thick slab (water / concrete / Pb / Fe / W) reports the analog
`FOM = 1 / (σ_rel² · t_wall)` reference, then accepts a CADIS map
via `--cadis-load FILE` to drive Cartesian-mesh weight windows
(splitting + Russian roulette) in the photon hot path. The
random-ray adjoint feeds the importance map.

Headline FOM measurements (1 MeV photons, water):

| Depth   | Mode             | T          | σ_rel | FOM (/s) | vs analog |
|---------|------------------|------------|------:|---------:|----------:|
| 100 cm  | Analog           | 5.372e-3   | 1.11% |    161.7 | —         |
| 100 cm  | **RR-CADIS**     | 5.330e-3   | 1.02% |  **354** | **2.19×** |
| 200 cm  | Analog           | 1.105e-5   | 10.2% |    0.351 | —         |
| 200 cm  | **RR-CADIS**     | 1.252e-5   | 11.8% |  **1.52**| **4.32×** |

Both RR-CADIS results unbiased within combined MC σ. Documented
negative results: a "lite" detector-backward collision-density
proxy was strictly *worse* than analog at every depth and was
removed in favour of the random-ray adjoint; adaptive-ratio WW is
indistinguishable from fixed `ratio=5` at 100 cm and worse at 200
cm; 300 cm (≈21 mfp) under any naive WW configuration gives 0
transmitted photons in 500 k histories — the textbook fix is
continuous-splitting / DXTRAN-style point-detector estimators.

## Hex lattice geometry

Recursive `Geometry` with `geometry::shapes` builders (`rect_box`,
`rect_box_split_bc`, `hex_boundary`, `hex_side_normals`,
`pin_cylinders`) — same helpers exposed in Python. HexLattice
transport runs on **CPU** (descent + `trace_step` + grid-distance
dispatch) and on **GPU** (CUDA device functions, 8 SoA buffers,
5 kernel signatures); CPU validated end-to-end via `hex_minicore`,
GPU on-device parity confirmed on RTX A1000 (`gpu_recursive_parity`
Tests 7–9, 2026-05-08).

## CUDA backend

`gpu.rs`, `gpu_transport.rs`, and `gpu_recursive.rs` implement
pointwise / SVD / WMP / URR / S(α,β) providers on device via
`cudarc`, with a **recursive geometry** kernel suite
(`find_cell_batch`, `trace_step_batch`, `multi_step_walk`) that
matches the CPU CSG walker, plus full-physics
`transport_recursive` and constant-XS (`const_xs_transport_persistent`)
device transport. The GPU path is bit-parity with CPU SVD at
machine precision (the `--force-svd` parity harness verifies this
across seeds). Backend dispatch goes through
`transport::dispatch::EigenvalueRunner` with `CpuRunner` and (cuda-
feature-gated) `CudaRunner` so binaries (`godiva`, `pwr_pincell`,
`pwr_assembly`, `pwr_gamma_heating`, `hex_minicore`) drive either
backend through the same config.

Hex on GPU runtime parity: 0 disagreements across 200 k / 50 k /
20 k geometry-primitive trials, 6.6× / 6.1× / 21.6× speedup vs CPU,
and full-physics k_inf parity 71 pcm < 0.2σ_combined on the 1-ring
hex mini-core (RTX A1000 Laptop, 2026-05-08).

EADL atomic relaxation cascade (`dcfd901`) ships on GPU — full
photoelectric photon transport including fluorescence + Auger now
runs on device, removing the prior CPU bounce.

`src/photon/gpu.rs` adds GPU sampling kernels for photon transport,
all NVRTC-compiled into one PTX module:

- `GpuComptonContext` / `GpuComptonVarECtx` — Klein-Nishina + S(x,Z)/Z
  bound-electron rejection, with optional Compton-profile Doppler
  broadening when profiles are uploaded
- `GpuRayleighContext` — direct `x²` CDF inversion + Thomson rejection
- `GpuPairContext` — Bethe-Heitler ε rejection sampling
- Photoelectric on device with full **EADL atomic relaxation
  cascade** (fluorescence + Auger), removing the prior CPU bounce

The GPU samplers reproduce CPU samples bit-for-bit (PCG-64 mirrors
`Rng::for_particle(batch_id, tid)`). A persistent-kernel mode runs
full Compton history loops in a single launch — kernel-only,
free Klein–Nishina, no detector: ~107× faster than the per-collision
launch model and ~2.2× faster than rayon-20-thread CPU on N = 1 M
histories (RTX A1000 laptop). Photon photoelectric on U / Zr at
1 MeV is the highest per-kernel lift at ~9.8×. **These are
kernel-level numbers; the integrated-transport story is mixed —
see `paper §gpu` for the full provider matrix, where GPU SVD on
Godiva is in fact 1.3× *slower* than GPU pointwise.** See
`resume.md` for ns/event tables and run conditions.

Other GPU-technology hooks (cuBLAS batched DGEMM for SVD
reconstruction, software BVH ray-AABB traversal, NVLink split-and-
merge plumbing) live behind feature flags on `gpu_photon_features`.
Enable with `--features cuda`.

## Repository layout

```
rust_prototype/
  src/
    physics/                  Collision processing, scattering kinematics
    transport/
      simulate.rs             Particle tracking + k-eigenvalue solver
                               + PhotonSourceEvent tally
      xs_provider.rs          SVD + pointwise providers, Ducru interpolation
                               + per-nuclide PhotonProduct loader
      hybrid_xs.rs             SVD+WMP and ACE+WMP hybrid providers
      dispatch.rs              CpuRunner / CudaRunner backend dispatch
      kinetics.rs              Point kinetics (Crank-Nicolson, 6 delayed groups)
      adjoint_neutron.rs       Adjoint elastic kernel (CADIS substrate)
      adjoint_photon.rs        Adjoint photon slab walker → ImportanceMap
      tally.rs                 Surface currents, mesh flux (Amanatides-Woo)
      weight_window.rs         Cartesian-mesh WW + splitting/roulette
      statepoint.rs            HDF5 statepoint write/read/restart
      thermal_library.rs       Named c_*.h5 resolver (H₂O, D₂O, ZrH, …)
      nuclides.rs              ZAID-keyed NuclideLibrary catalog
      urr_equivalence.rs       Carlvik-Pellaud Dancoff + σ_eff correction
    photon/                   Photon transport (4 kernels + CSG driver)
      data.rs                  PhotonElement, subshells, form factors
      hdf5_reader.rs           OpenMC photon HDF5 reader
      coherent.rs              Rayleigh scattering
      compton.rs               Klein-Nishina + S(x,Z)/Z + Doppler
                                + adjoint_compton_scatter (transposed KN)
      photoelectric.rs         Photoelectric + EADL cascade
      pair.rs                  Bethe-Heitler pair production
      bremsstrahlung.rs        Seltzer-Berger DCS + secondary emission
      electron.rs              Bethe-Bloch dE/dx + Highland MS
      material.rs              PhotonMaterial + electron transport
      transport.rs             Closure-based + CSG photon drivers
      gpu.rs                   CUDA NVRTC kernels (cuda feature)
      nee.rs                   Next-event estimator
    random_ray/               Multigroup TRRM (forward + adjoint + immortal)
      mod.rs, mgxs.rs           MgxsLibrary, ScatterMatrix
      fsr.rs                    FsrMesh (Cartesian + Cell-based)
      integrator.rs             solve_segment (analytic MoC ODE)
      solver.rs                 RandomRaySolver, source iteration
      cadis.rs                  weight_window_from_adjoint bridge
      adjoint_svd.rs            SVD-compressed adjoint flux for WW storage
    depletion/                Burnup pipeline
      cram.rs                   CRAM-16 / CRAM-48 (IPF, Pusa 2016)
      chain.rs, chain_io.rs     DepletionChain + JSON loader
      matrix.rs                 Transmutation matrix builder
      predictor_corrector.rs    CE/LI step
      mapping.rs                BurnupMapping walker
      flux.rs                   Per-source flux extractor
    geometry/                 CSG surfaces, cells, BVH, lattices
                               + rect / hex lattice + shapes builders
    hdf5_reader.rs            Pure-Rust neutron HDF5 reader
                               + read_photon_products for γ spectra
    thermal.rs                S(α,β) data structures + sampling
    quadrature.rs             Gauss-Legendre nodes/weights
    kernel.rs                 CPU SVD reconstruction hot path
    table.rs                  Pointwise table, StochTempTable wrapper
    wmp.rs                    Windowed multipole + Humlicek W4 Faddeeva
    gpu.rs, gpu_transport.rs  CUDA SVD/pointwise/WMP/URR providers
    gpu_recursive.rs          Recursive-geometry CUDA kernels
    gpu_random_ray.rs         CUDA scaffold for persistent random-ray
  src/bin/
    godiva.rs                 Godiva benchmark binary
    pwr_pincell.rs            PWR pin cell benchmark binary (URR-eq, S(α,β))
    pwr_assembly.rs           PWR mini-assembly (rect lattice)
    pwr_d2o_pincell.rs        Heavy-water pin cell + pitch-sweep moderation
    pwr_gamma_heating.rs      Coupled n-γ PWR pin γ-heating benchmark
    hex_minicore.rs           Hex lattice mini-core (CPU)
    deplete_demo.rs           Constant-flux Xe equilibrium
    deplete_pwr.rs            Full transport-coupled actinide burnup
    point_kinetics_demo.rs    Step / ramp / scram reactivity profiles
    rr_pincell.rs             Random-ray 2-group pin cell
    rr_cadis_slab.rs          Random-ray adjoint → CADIS JSON
    rr_adjoint_svd.rs         SVD-compressed adjoint flux probe
    rr_adjoint_sweep.rs       FSR-mesh refinement sweep
    shield_slab.rs            Photon shielding FOM benchmark + WW driver
    cs137_pulse_height.rs     Cs-137 + NaI detector validation
    photon_dump.rs            Photon HDF5 data inspection utility
    cp_analysis.rs            Collision-probability analysis
    xs_dump.rs / xs_dump_godiva.rs / xs_provider_diff.rs
                              XS dump and provider-diff utilities
    gpu_bench.rs              GPU XS reconstruction microbenchmark
    gpu_compton_validate.rs   GPU vs CPU Compton bit-parity harness
    gpu_compton_scaling.rs    GPU batch-size scaling (free + Doppler)
    gpu_cpu_bench.rs          Full CPU-vs-GPU photon kernel benchmark
                               + persistent-kernel Compton history mode
    gpu_photon_features.rs    cuBLAS DGEMM, NVLink, persistent kernel,
                               software BVH ray-AABB demos
    gpu_pwr_bench.rs          GPU PWR pin cell eigenvalue
    gpu_hex_minicore.rs       GPU hex lattice mini-core
    gpu_recursive_parity.rs   GPU vs CPU recursive geometry parity
    gpu_recursive_keff.rs     GPU recursive-geometry k_eff
    gpu_const_xs_keff.rs      GPU constant-XS persistent transport
    gpu_assembly_keff.rs      GPU PWR mini-assembly
    gpu_wmp_validate.rs       GPU WMP cross-check
    wmp_validate.rs           WMP evaluator cross-check vs Python reference
  bindings/python/            PyO3 Python API (Scene, Material,
                               XsMode, run_eigenvalue, run_gamma_heating,
                               Chain, CramOrder, deplete_*)
  tests/                      Integration tests
                               (Cs-137, Hubbell Compton, ANSI/ANS-6.6.1)
chains/
  partial_xe.json             Xe poisoning (4 nuclides)
  pwr_actinides.json          PWR actinide chain (17 nuclides)
cuda_bench/                   Standalone CUDA SVD reconstruction kernel
gpu/cuda/                     CUDA source (recursive geometry, random-ray)
scripts/
  pwr_verdict.py              Semaphore-grade three-way verdict runner
  u238_capture_rank_probe.py  Offline Ducru-interpolation validation
  phase*_*.py                 HDF5 extraction, SVD analysis, OpenMC cross-checks
paper/                        LaTeX manuscript (main.tex) + bib
.github/workflows/ci.yml      Rust + Python + LaTeX CI
```

## Test environment for the numbers in this README

All quantitative results in this document (γ-heating splits, GPU
ns/event, PWR k_inf, etc.) come from one fixed-seed sweep on:

- **CPU**: 20-core Intel mobile workstation, 32 GB RAM, Windows 11,
  rayon over all 20 threads.
- **GPU**: NVIDIA RTX A1000 (laptop, 4 GB, fp64 ~0.51 TFLOP/s, no
  tensor-core fp64) via `cudarc` + NVRTC.
- **Library**: ENDF/B-VII.1 HDF5
  (`../data/endfb-vii.1-hdf5/{neutron,photon}`).
- **Reference code**: OpenMC 0.15.3 on the same library.

Per-test particle / batch / iteration counts are stated alongside
each result. Raw outputs are checked into `outputs/full_test_run/`.

## Build and run

```bash
cd rust_prototype
cargo build --release

# Nuclear data: ENDF/B-VII.1 HDF5 from https://openmc.org/data/
DATA=../data/endfb-vii.1-hdf5/neutron

# Godiva: SVD vs pointwise table vs ACE+WMP
cargo run --release --bin godiva -- $DATA --mode all --rank 5 \
    --batches 150 --inactive 20 --particles 50000 --seeds 10

# PWR pin cell at an off-library operating temperature
cargo run --release --bin pwr_pincell -- $DATA --mode all --rank 5 \
    --target-temp-offset 150 --discrete-rank 1 \
    --batches 120 --inactive 30 --particles 50000 --seeds 10

# Semaphore verdict (GREEN/YELLOW/RED, exit code 0/1/2):
python scripts/pwr_verdict.py --offset 150 --seeds 10 --particles 50000 \
    --batches 120 --inactive 30 \
    --log outputs/pwr_verdict.log --json outputs/pwr_verdict.json

# Coupled neutron-photon γ-heating (~2.5 min on desktop CPU)
cargo run --release --bin pwr_gamma_heating -- \
    $DATA --photon-data ../data/endfb-vii.1-hdf5/photon

# Cs-137 pulse-height spectrum on 3"x3" NaI detector (photon validation)
cargo run --release --bin cs137_pulse_height -- \
    ../data/endfb-vii.1-hdf5/photon --n 200000

# Burnup: Xe equilibrium (matches analytical to 1e-4)
cargo run --release --bin deplete_demo

# Burnup: full PWR actinide chain with transport-coupled CE/LI
cargo run --release --bin deplete_pwr -- $DATA \
    --chain chains/pwr_actinides.json --steps 10 --power 100e6

# Time-dependent point kinetics (step / ramp / scram)
cargo run --release --bin point_kinetics_demo -- \
    --profile step --reactivity 0.5 --duration 30

# Random-ray multigroup pin cell (forward + adjoint, 2 groups)
cargo run --release --bin rr_pincell

# Generate FW-CADIS importance map for the shielding slab
cargo run --release --bin rr_cadis_slab -- \
    --thickness 100 --bins 25 --out outputs/cadis_water_100cm.json

# Photon shielding slab — analog FOM, then with FW-CADIS WW
cargo run --release --bin shield_slab -- \
    --photon-data ../data/endfb-vii.1-hdf5/photon \
    --thickness 100 --histories 1000000
cargo run --release --bin shield_slab -- \
    --photon-data ../data/endfb-vii.1-hdf5/photon \
    --thickness 100 --histories 1000000 \
    --cadis-load outputs/cadis_water_100cm.json --ww-ratio 5

# Hex mini-core (CPU + GPU parity)
cargo run --release --bin hex_minicore -- $DATA --rank 5
cargo run --release --features cuda --bin gpu_hex_minicore -- $DATA --rank 5

# GPU (requires CUDA toolkit)
cargo run --release --features cuda --bin gpu_bench -- $DATA \
    --rank 5 --particles 1000000

# Library tests (352 lib + 10 integration tests)
cargo test --lib
```

Pass `--mode svd`, `--mode table`, `--mode wmp`, or `--mode hybrid`
to run a single provider instead of the honesty test.

## Python API

A PyO3 binding exposes the engine to Python via a fluent
`Scene`/`Material`/`Surface`/`PhotonMaterial` builder. The same
Godiva eigenvalue and PWR γ-heating run that the Rust binaries
drive are reproducible from short Python scripts
(`rust_prototype/bindings/python/examples/godiva.py`,
`pwr_gamma_heating.py`, `xs_mode_quick.py`). Rust remains the
source of truth — Python is a ~200-line glue layer over the engine.

The full provider matrix is exposed as the `XsMode` enum
(`Table` / `Svd` / `HybridTableWmp` / `HybridSvdWmp`) with per-MT
SVD rank overrides via `Scene.set_svd_ranks({mt: rank, …})`.
`run_gamma_heating` drives the coupled neutron-photon pipeline
end-to-end with full electron transport on by default. The
depletion API exposes `Chain.from_file` / `Chain.from_str`,
`CramOrder.{Order16, Order48}`, `cram(matrix, n0, order)`,
`deplete_constant_flux`, and `deplete_with_flux_callback` (an
FFI-exception-safe Python `flux_at` closure for predictor-
corrector with mid-step transport solves), plus
`Material.set_atom_density(hdf5_file, density)` /
`atom_density_of(hdf5_file)` for in-place composition updates.
Examples: `godiva.py`, `pwr_pincell.py`, `pwr_gamma_heating.py`,
`hex_minicore.py`, `depletion_xe_demo.py`, `seed_sweep.py`,
`xs_mode_demo.py`, `xs_mode_quick.py`. See **[PYTHON.md](PYTHON.md)**
for the quick-start, API reference, and build-from-source
instructions.

## Development

CI runs Rust, Python, and LaTeX jobs on every push (`.github/workflows/ci.yml`).
Locally:

```bash
# Rust
cd rust_prototype
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test

# Python
ruff check scripts/

# Paper
cd paper && latexmk -pdf main.tex
```

The `pwr_verdict.py` script is CI-ready: exit codes 0 / 1 / 2 map
onto GREEN / YELLOW / RED semaphore grades, so it can guard against
physics regressions in a pipeline.

## Paper

Numerical results, reconstruction-error analyses, and full three-way
head-to-head measurements at on-library and off-library temperatures
are in `paper/main.pdf` — *"SVD-Compressed Cross Sections in Monte
Carlo Neutron Transport: An implementation-led benchmark with a
partition-of-unity fix for off-library temperature interpolation."*

## References

- Romano et al., *OpenMC: A state-of-the-art Monte Carlo code for
  research and development*, Ann. Nucl. Energy 82 (2015) 90–97.
- Ducru et al., *Kernel reconstruction methods for Doppler broadening*,
  J. Comput. Phys. 335 (2017) 535–557.
- Josey, Ducru, Forget, Smith, *Windowed multipole for cross section
  Doppler broadening*, J. Comput. Phys. 307 (2016) 715–727.
- Brown, *New hash-based energy lookup algorithm for Monte Carlo
  codes*, Trans. ANS 111 (2014) 659–662.
- Tramm et al., *Performance Portable MC Particle Transport on Intel,
  NVIDIA, and AMD GPUs*, EPJ Web Conf. 302 (2024) 04010.
- Tramm, Smith, *The Random Ray Method for neutral particle
  transport*, J. Comput. Phys. 342 (2017) 229–252; Tramm, Siegel,
  *Performance optimization of the random ray method on GPUs*,
  Ann. Nucl. Energy 154 (2021) 108118.
- Pusa, *Higher-Order Chebyshev Rational Approximation Method and
  application to burnup equations*, Nucl. Sci. Eng. 182 (2016) 297–318.
- Wagner, Haghighat, *Automated variance reduction of Monte Carlo
  shielding calculations using the discrete ordinates adjoint
  function*, Nucl. Sci. Eng. 128 (1998) 186–208.
- Carlvik, *A method for calculating collision probabilities in
  general cylindrical geometry and applications to flux distributions
  and Dancoff factors*, A/CONF.28/P/681 (1964); Pellaud, *Resonance
  shielding factors for square pin lattices*, ANL-RSCM (1976).
- Keepin, *Physics of Nuclear Kinetics* (1965); Hetrick, *Dynamics of
  Nuclear Reactors* (1971) — six-group delayed-neutron constants.

## License

MIT. See `LICENSE`.
