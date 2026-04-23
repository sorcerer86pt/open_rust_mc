# open_rust_mc

A pure-Rust continuous-energy Monte Carlo neutron **and photon**
transport engine. Reads OpenMC HDF5 nuclear data directly (no C
dependency), runs k-eigenvalue simulations end-to-end on CPU (rayon)
or CUDA GPU, and is validated against OpenMC on two reference
benchmarks. A coupled neutron-photon pipeline drives a PWR pin cell
γ-heating calculation directly off the ENDF/B-VII.1 HDF5 library.

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
- S(α,β) thermal scattering for H in H₂O (continuous + discrete inelastic, Bragg edges, Debye–Waller incoherent elastic)

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

Simplified electron transport: Katz–Penfold CSDA midrange deposit
(`R_e(E) / 2` forward displacement) with per-material mass density
and reflective-lattice folding. Removes most of the kerma bias for
pin-cell heating (~5 % textbook error → <0.5 % in UO₂).

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

Converged result (150 b × 50 k n + 200 k γ, ~2.5 min):

| region | fraction |
|--------|---------:|
| fuel   | **84.4 %** |
| gap    | 1.5 % (*CSDA midrange artefact*) |
| clad   | 7.9 % |
| water  | 5.9 % |
| escape | 0.0 % |
| sum    | 99.66 % (missing 0.34 % = EADL valence-binding) |

## CUDA backend

`gpu.rs` and `gpu_transport.rs` implement pointwise and SVD providers
on device via `cudarc`. The GPU path is bit-parity with CPU SVD at
machine precision (the `--force-svd` parity harness verifies this
across seeds). Enable with `--features cuda`.

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
      hybrid_xs.rs            SVD+WMP and ACE+WMP hybrid providers
    photon/                   Photon transport (4 kernels + CSG driver)
      data.rs                 PhotonElement, subshells, form factors
      hdf5_reader.rs          OpenMC photon HDF5 reader
      coherent.rs             Rayleigh scattering
      compton.rs              Klein-Nishina + S(x,Z)/Z + Doppler
      photoelectric.rs        Photoelectric + EADL cascade
      pair.rs                 Bethe-Heitler pair production
      material.rs             PhotonMaterial + CSDA range (Katz-Penfold)
      transport.rs            Closure-based + CSG photon drivers
    geometry/                 CSG surfaces, cells, BVH, lattices
    hdf5_reader.rs            Pure-Rust neutron HDF5 reader
                               + read_photon_products for γ spectra
    thermal.rs                S(α,β) data structures + sampling
    kernel.rs                 CPU SVD reconstruction hot path
    table.rs                  Pointwise table, StochTempTable wrapper
    wmp.rs                    Windowed multipole + Humlicek W4 Faddeeva
    gpu.rs, gpu_transport.rs  CUDA backend (feature-gated)
  src/bin/
    godiva.rs                 Godiva benchmark binary
    pwr_pincell.rs            PWR pin cell benchmark binary
    pwr_gamma_heating.rs      Coupled n-γ PWR pin γ-heating benchmark
    cs137_pulse_height.rs     Cs-137 + NaI detector validation
    photon_dump.rs            Photon HDF5 data inspection utility
    gpu_bench.rs              GPU reconstruction microbenchmark
    wmp_validate.rs           WMP evaluator cross-check vs Python reference
  tests/                      Integration tests
                               (Cs-137, Hubbell Compton, ANSI/ANS-6.6.1)
cuda_bench/                   Standalone CUDA SVD reconstruction kernel
scripts/
  pwr_verdict.py              Semaphore-grade three-way verdict runner
  u238_capture_rank_probe.py  Offline Ducru-interpolation validation
  phase*_*.py                 HDF5 extraction, SVD analysis, OpenMC cross-checks
paper/                        LaTeX manuscript (main.tex) + bib
.github/workflows/ci.yml      Rust + Python + LaTeX CI
```

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

# GPU (requires CUDA toolkit)
cargo run --release --features cuda --bin gpu_bench -- $DATA \
    --rank 5 --particles 1000000

# Library tests
cargo test --lib
```

Pass `--mode svd`, `--mode table`, `--mode wmp`, or `--mode hybrid`
to run a single provider instead of the honesty test.

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

## License

MIT. See `LICENSE`.
