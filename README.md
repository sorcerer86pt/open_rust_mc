# open_rust_mc

[![Latest release](https://img.shields.io/badge/release-v0.4.0-blue)](https://github.com/sorcerer86pt/open_rust_mc/releases/latest)

A pure-Rust continuous-energy Monte Carlo neutron **and photon**
transport engine. Reads OpenMC HDF5 nuclear data directly (no C
dependency), runs k-eigenvalue, fixed-source, time-dependent
point-kinetics, and burnup simulations end-to-end on CPU (rayon) or
CUDA GPU. Validated against OpenMC on Godiva, PWR pin cell,
PWR-actinide depletion, and γ-heating; against ICSBEP handbook
values on a 375-case scene corpus.

Started as an SVD cross-section compression study (`paper/main.tex`);
grew into a research vehicle for studying cross-section
representation, recursive geometry on GPU, and CIELO-era nuclear-data
evaluation effects across ENDF/B-VII.1 / VIII.0 / VIII.1.

438 / 438 lib tests + 5 integration tests pass on every push
(`cargo test`); 443 / 443 with `--features cuda`.

## Highlights

- **Four interchangeable cross-section providers** behind one
  `XsProvider` trait — pointwise Table, SVD, Hybrid SVD+WMP, and
  Hybrid Table+WMP — selectable at runtime via `--mode`. A
  three-way honesty-test mode runs all providers back-to-back on
  the same geometry.
- **Recursive universe geometry** (`CoordStack`-based) with
  rectangular + hexagonal lattices, per-cell `Mat3` rotations, and
  per-universe BVH acceleration. Full 17×17 PWR assembly + N-ring
  hex mini-cores ship as built-in benchmarks.
- **CUDA backend** for transport, photon kernels, random-ray, SVD
  reconstruction, and constant-XS event-batched eigenvalue. Includes
  PHYSOR-2022-style refill-pool with auto-recommendation from device
  attributes.
- **Coupled neutron-photon** pipeline. PWR γ-heating runs end-to-end
  in ~5 min with full electron condensed-history transport
  (Bethe-Bloch + Highland MS + Seltzer-Berger brems). Result agrees
  with OpenMC 0.15.3 fuel/gap/clad/water split within 1 pp.
- **Depletion** via CRAM-16 / CRAM-48 with on-the-fly chain-XS
  spectrum collapse and fresh-corrector predictor. Xe equilibrium
  matches analytical to 1e-4 relative.
- **Variance reduction**: forward weight windows + flux bootstrap +
  full multigroup random-ray (Tramm 2018, immortal-ray
  Tramm-Siegel 2021) feeding a measured FW-CADIS pipeline. 4.32×
  FOM gain on 200 cm water at 1 MeV.
- **ENDF/B-VIII.1 default**, with VIII.0 and VII.1 supported as
  fallbacks. Engine handles the sibling-`thermal/` layout split
  transparently and ships an idempotent natural-element migration
  script for benchmark JSONs.
- **Python (PyO3) front-end** — `Scene`, `Material`, `Surface`,
  `run_eigenvalue`, `run_icsbep_case`, full XsMode / runner /
  depletion / refill-pool plumbing.

## Quick start

### Prerequisites

- Rust stable (1.79+), `cargo`.
- (Optional) CUDA toolkit 12.x + NVRTC for the GPU backend.
- (Optional) Python 3.9+ + [`maturin`](https://www.maturin.rs/) for
  the Python wheel.

### Build

```powershell
git clone https://github.com/sorcerer86pt/open_rust_mc
cd open_rust_mc/rust_prototype
cargo build --release                       # CPU only
cargo build --release --features cuda       # + CUDA backend
cargo test --lib                            # 438 / 438 default
cargo test --lib --features cuda            # 443 / 443
```

### Download nuclear data (ENDF/B-VIII.1, ~6.5 GB)

```powershell
.\scripts\setup_nuclear_data.ps1            # VIII.1 (default)
.\scripts\setup_nuclear_data.ps1 -All       # all four supported libs
.\scripts\setup_nuclear_data.ps1 -Vii1      # legacy VII.1 only
```

The script auto-flattens the wrapper directory the VIII.1 tarball
uses, and the verification block reports key-file presence per
library.

### Run a benchmark

Godiva HEU sphere, end-to-end, SVD provider, k=5:

```powershell
cargo run --release --bin godiva -- data\endfb-viii.1-hdf5\neutron `
  --rank 5 --batches 80 --inactive 15 --particles 10000
```

A single ICSBEP case via Python:

```powershell
cd rust_prototype/bindings/python
maturin develop --release --features cuda
cd ../../..
python rust_prototype/bindings/python/examples/icsbep_run.py heu-met-fast-001_case-1 gpu
```

Full corpus sweep (auto-detects GPU vs CPU):

```powershell
.\rust_prototype\bindings\python\examples\run_benchmark.ps1
```

A scoped sweep with explicit CLI overrides (explicit flag wins over
the JSON's `recommended_settings`):

```powershell
python rust_prototype/bindings/python/examples/icsbep_sweep.py `
    --runner cpu --filter heu-comp-inter-003 `
    --particles 100000 --batches 150 --inactive 40 --seeds 5 `
    --csv outputs/heu_comp_inter_003_v81.csv `
    --stop-file outputs/STOP
```

## Headline benchmarks

All k_eff numbers carry a scope tag — `[godiva]` means end-to-end
3-nuclide Godiva sphere; `[pwr]` means 9-nuclide PWR pin cell;
`[assembly]` is depth-3 recursive 17×17. See `STATUS.md` for the
full table.

Only numbers re-verified against `outputs/` and `results/` this
audit are shown here. See [`STATUS.md`](STATUS.md) for the full
audited table including older claims still awaiting re-measurement.

| Metric | Scope | Value | Source |
|---|---|---|---|
| Lib tests (default) | — | **438 / 438 green** | `cargo test --lib` |
| Lib tests (`--features cuda`) | — | **447 / 447 green** | `cargo test --lib --features cuda` |
| ICSBEP family suite | `[icsbep]` | **6 / 6 PASS** (141 s, `max(150 pcm, 2σ)`) | `outputs/cuda_runs_after_rank_fix.txt` |
| PWR γ-heating fuel/clad/water | `[photon]` | 84.12 % / 9.81 % / 5.72 % (gap 0 %) | `outputs/pwr_gamma_heating_benchmark.txt` |
| Saturation knee (HMF-001, RTX 3080) | `[godiva]` | 500k–1M particles per batch | `outputs/saturation_*.csv` |
| Peak throughput at saturation | `[godiva]` | ~1.2 M histories/sec at 1M particles | `outputs/saturation_1000000.csv` |
| Refill 2× at mid-curve (HMF-008) | `[micro]` | 2.0× more collisions at same wall; σ 2.1× tighter; µs/coll −50 % | `outputs/hmf008_refill_*.csv` |
| RR-CADIS shielding gain at 14 mfp (200 cm water) | `[shield]` | **1.18×** analog (RR-CADIS alone); **1.75×** with NEE | `outputs/method_comparison_2026-05-08.txt` |
| SVD on Godiva vs Table | `[godiva]` | 1.22× faster; 5.14× memory | same |
| SVD on PWR (9 nuc + S(α,β)) vs Table | `[pwr]` | **1.25× *slower***; 5.12× memory | same |
| GPU SVD vs 20-core CPU SVD on Godiva | `[godiva]` | **0.77×** (1.3× *slower* — launch + memory penalty dominates) | same |

The headline result of the SVD compression study (the engine's
original purpose) is in `paper/main.pdf`: SVD beats the pointwise
table on small fast-spectrum problems by ~22 %, loses by ~25 % on
realistic thermal-spectrum PWR pin cells, and costs ~5× memory in
both regimes. The honest reading is in the paper — this engine is
also a *vehicle* for measuring SVD against realistic baselines,
not a claim that SVD is universally faster.

## Architecture

The engine is structured around one common particle-transport loop
that runs against any geometry, any cross-section provider, and any
backend:

```
┌──────────────────────────────────────────────────────────────┐
│  Scene JSON                                                  │
│  └─> material_resolve  ─>  XsProvider                        │
│      (natural-element       ┌── Table   (pointwise)          │
│       expansion, thermal    ├── Svd     (rank-k FMA)         │
│       binding, kernel       ├── HybridTableWmp               │
│       dedup)                └── HybridSvdWmp                 │
│                                                              │
│  └─> Geometry  ─>  Surface / Cell / Region / Universe        │
│                    BVH (per universe, ≥8 cells)              │
│                                                              │
│  └─> EigenvalueRunner  ─>  CpuRunner   (rayon, history)      │
│                            CudaRunner  (sm_86+, event-batched)│
└──────────────────────────────────────────────────────────────┘
```

Key modules under `rust_prototype/src/`:

- `data_paths.rs` — layout-aware ENDF/B HDF5 path helpers
  (sibling-`thermal/` resolver, multi-library discovery).
- `kernel.rs`, `decompose.rs`, `cp_decompose.rs` — SVD
  reconstruction + decomposition.
- `hdf5_reader.rs`, `thermal.rs` — pure-Rust HDF5 + S(α,β).
- `table.rs`, `wmp.rs` — pointwise + Windowed Multipole providers.
- `geometry/` — recursive universes, BVH, hex/rect lattices.
- `physics/` — collision, scatter, kinematics.
- `transport/` — `simulate`, `dispatch`, `material_resolve`,
  `xs_provider`, `hybrid_xs`, `urr_equivalence`, `weight_window`,
  `tally`, `statepoint`, `kinetics`, `adjoint_neutron`,
  `adjoint_photon`.
- `photon/` — Compton, Rayleigh, photoelectric, pair, brems,
  electron, transport.
- `random_ray/` — multigroup TRRM (forward + adjoint + immortal),
  CADIS, adjoint-SVD.
- `depletion/` — CRAM, chain, predictor-corrector, mapping, flux.

## Cross-section providers

All four implement `XsProvider`; selectable at runtime via `--mode`.

| Mode | Provider | Implementation | What it does |
|------|----------|---------------|--------------|
| `table` | Pointwise table | `src/table.rs` | OpenMC-style binary search + log-log interpolation per reaction |
| `svd` | Truncated SVD | `src/kernel.rs` | Rank-*k* reconstruction, one FMA sequence per lookup |
| `hybrid_svd_wmp` | SVD + WMP | `src/transport/hybrid_xs.rs` | SVD everywhere, WMP override (pole/residue + Faddeeva) inside the RRR |
| `hybrid_table_wmp` | Table + WMP | `src/transport/hybrid_xs.rs` | Pointwise everywhere, WMP in the RRR — industry low-memory baseline |

Off-library temperatures use stochastic pseudo-interpolation
(Table / Hybrid) or partition-of-unity 3-point Ducru reconstruction
(SVD). Detailed numerical results live in `paper/main.tex`.

## Photon physics

Full four-channel transport on per-element OpenMC HDF5 photon
libraries:

- **Compton** — Klein-Nishina + `S(x,Z)/Z` bound-electron rejection
  + optional Hartree-Fock Doppler broadening.
- **Photoelectric** — subshell absorption + EADL atomic-relaxation
  cascade (fluorescence + Auger).
- **Pair production** — Bethe-Heitler nuclear + electron-field +
  in-flight positron annihilation.
- **Rayleigh** — form-factor + Thomson rejection.

Plus condensed-history electron transport (Bethe-Bloch dE/dx with
per-element I, Highland multiple-scattering with per-cell X₀,
Seltzer-Berger bremsstrahlung secondaries banked back into the
photon loop). The He-gap deposition artefact present in the
older Katz-Penfold CSDA path goes from 1.5 % to 0 %.

Coupled neutron-photon: the neutron loop tallies a
`PhotonSourceEvent { cell, pos, E_γ, MT }` at every capture
(MT=102), fission (MT=18), (n,p) / (n,α) (MT=103 / 107,
threshold-gated), and inelastic (MT=4 via discrete-level Q-value)
collision. γ multiplicities and outgoing energies come from
the HDF5 `reactions/reaction_{mt}/product_{N}` tree with
`particle="photon"`.

## Depletion (burnup)

- **CRAM-16 / CRAM-48** matrix-exponential evaluator (IPF form,
  Pusa 2016 poles + residues from OpenMC's canonical source) with
  dense complex Gaussian elimination + partial pivoting.
- `DepletionChain` — decay constants, branches, per-(parent, MT)
  one-group reaction XS with default ENDF yield inference for
  (n,γ) / (n,2n) / (n,3n) / (n,p) / (n,α).
- `chain_io` JSON loader with three-way `yields` semantics:
  omitted → default ENDF, `{}` → pure removal, explicit map → use it.
- **CE/LI predictor-corrector** with **fresh-corrector**: clones
  materials, runs a second eigenvalue at the predicted composition
  for the EOC flux, then CRAM with the averaged matrix.
- **On-the-fly chain-XS spectrum collapse** — closes the
  9× to 0.77× gap to OpenMC depletion.
- Shipped chains: `chains/partial_xe.json` (4-nuclide Xe poisoning)
  and `chains/pwr_actinides.json` (17-nuclide actinide buildup +
  dominant FP poisons).

## CUDA backend

`--features cuda` enables an Ampere-class (sm_86+) backend that
shares the same physics as CPU:

- Recursive cell-find / trace-step / multi-step walk —
  bit-exact vs CPU (≤ 9.3e-11 max-rel-err), 3-24× speedups on
  RTX A1000.
- Constant-XS transport with atomic-add fission banking (6.74×).
- Multi-slot S(α,β) — concurrent TSLs on multiple nuclides in one
  run. Stochastic temperature interpolation between bracketing
  slots based on per-cell kT.
- Per-nuclide kernel cache (`Arc::as_ptr`-keyed, LRU + bundle
  budget) eliminates redundant HtoD on multi-case sweeps.
- Refill pool (PHYSOR 2022 Optimization F) — opt-in via
  `gpu_refill_pool_factor` or auto-recommended via device-attribute
  inspection. 2× histories at same wall time on mid-curve
  workloads.
- HexLattice device functions ported; runtime parity test pending.

GPU bench drivers: `gpu_bench`, `gpu_cpu_bench`,
`gpu_recursive_keff`, `gpu_const_xs_keff`, `gpu_assembly_keff`,
`gpu_pwr_bench`, `gpu_hex_minicore`, `gpu_compton_validate`,
`gpu_compton_scaling`, `gpu_photon_features`, `gpu_wmp_validate`.

## Python bindings

```python
from open_rust_mc import (
    Scene, Material, Sphere, Settings, Runner,
    run_eigenvalue, run_icsbep_case,
)

fuel = (Material("HEU", temperature=294.0)
    .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
    .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.43)
    .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.49))

scene = (Scene("data/endfb-viii.1-hdf5/neutron")
    .add_material("heu", fuel)
    .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
    .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
    .add_cell("outside", region="+boundary"))

result = run_eigenvalue(scene, Settings(batches=50, inactive=10, particles=5000))
print(result.k_eff, result.k_sigma)
```

Full Scene builder + Material API documented in `PYTHON.md`. ICSBEP
harness lives at `rust_prototype/bindings/python/examples/`.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — engineer-facing project memory: hard
  invariants, file layout, build commands, recent session changes.
- [`STATUS.md`](STATUS.md) — current state: capabilities, headline
  numbers, open work, A/B against handbook references.
- [`PYTHON.md`](PYTHON.md) — Python API reference.
- [`BENCHMARKS.md`](BENCHMARKS.md) — bench-suite layout + scene-JSON
  schema.
- [`ICSBEP.md`](ICSBEP.md) — phased ICSBEP rollout plan.
- [`paper/main.tex`](paper/main.tex) — SVD cross-section
  compression paper (compiled to `paper/main.pdf`).

## Recent (May 2026)

- ENDF/B-VIII.1 became the default library; sibling-`thermal/`
  layout handled transparently.
- Natural-element migration on the ICSBEP corpus — 137 cases
  rewritten to per-isotope carbon. Engine-side fallback covers
  un-migrated future JSONs.
- Sweep precedence inverted: explicit CLI flag > JSON
  `recommended_settings` > built-in default.
- VII.1 → VIII.1 A/B on heu-comp-inter-003: VIII.1 shifts k upward
  by +820 pcm uniformly; localised driver is CIELO U-235
  (σ_f +5.3 %, α −10.2 % at 100 eV).

## License

MIT. See `LICENSE` (every source file carries
`SPDX-License-Identifier: MIT`).
