# Phase 3 handoff

State as of `128888b`. Stages A + B (data model + LFU + preload-weights
warm-start) are landed end-to-end. Stage C (kernel ABI change for
per-nuclide GPU indirection) is held.

---

## What landed this session

23 commits total since the 376-case GPU sweep that surfaced the
original HMF-014 stall. The 5 newest are the Phase 3 A/B work:

| commit | scope |
|---|---|
| `a2fe506` | `nuclide_cache::eviction` — byte-budgeted LFU-with-recency policy. Shared by both caches. |
| `99a980c` | `hardware_profile` — real RAM / CPU cache / GPU detection via `hardware-query` 0.2.1. Replaces 16 GiB stand-in. Shared `KIB/MIB/GIB` constants. |
| `ac80b0c` | L1MemoryStore (CPU host) on LFU + `set_preload_weights` API. Pending preload weights stash. |
| `8af5dc2` | GPU bundle cache on the same LFU policy via `BundleCacheAdapter`. Startup banner + hardware self-test. |
| `128888b` | `preload_nuclide_cache_weights` PyO3 binding. `icsbep_sweep.py` walks every case JSON pre-loop, counts `(zaid, temperature)` appearances, hands the histogram to the engine. |

Earlier commits this session covered Phase 1 (cache leak fixes), Phase
1c (host-side cache bound), Phase 2 (TransportBuffers per-batch pool),
Phase 8 (lib warning cleanup), and the partial lib-wide comment trim.

## Validation status

- **Lib tests**: 428 / 428 pass under `cargo test --features cuda --lib`.
- **CPU+GPU 5-case subset** (HMF-001 / HMF-014 / PMF-001 / U233-MF-001
  / HST-001) PASS on both backends on the A1000. HMF-014 (the hang
  that started this) holds at 2.6 GB peak VRAM.
- **Pre-scan smoke test** (4-case HEU subset): `10/10` nuclide weights
  resolved in ~15 s. Top entries: U-234 / U-235 / U-238 each at
  weight 10, N-14 / O-16 at weight 1.
- **Banner**: hardware-query produces the right summary on the dev
  box — `RAM 31.7 GB, 14p/20l cores, L1/L2/L3 = 288/256/8192 KB,
  GPU 4 GB sm_86`. AVX2+FMA detected via `std::is_x86_feature_detected!`
  (hardware-query under-reports FMA on this Windows box).

## Not pushed yet

`git push` fails — local credential is `fabio-andre-rodrigues`, remote
is `sorcerer86pt/open_rust_mc`. All 23 commits are local on `main`:

```
$ git log --oneline origin/main..HEAD | head
128888b nuclide_cache: sweep-manifest pre-scan for warm-start
8af5dc2 gpu_transport: LFU-with-recency on bundle cache; banner + self-test
ac80b0c nuclide_cache/l1_memory: LFU-with-recency + set_preload_weights
99a980c hardware_profile: real RAM / CPU cache / GPU detection via hardware-query
a2fe506 nuclide_cache/eviction: byte-budgeted LFU-with-recency policy
d8459e0 comments: trim lib.rs + physics/scatter
eedebf2 comments: trim photon mod + compton
8952b3e comments: trim hybrid_xs + l2_disk
ea0dd7e comments: trim geometry/bvh + geometry/shapes
248d115 comments: trim material_resolve, weight_window, tally
```

Run `gh auth login` (or `git config user.email sorcerer86pt@...`) to
fix the credential before `git push`.

---

## Stage C — kernel ABI change (held)

### What it is

Today's `GpuNuclideData` is a flat-packed bundle: every per-nuclide
field (basis, coeffs, energy grids, discrete-level kernels, ...) is
concatenated across all nuclides into one big `CudaSlice` per kind.
Kernels read via `__ldg(&PTR_D(p, P_BASIS)[offset + e*rank + r])`
with `offset = basis_offsets[nuc * n_rxn + rxn]`.

Stage C splits `GpuNuclideData` into:

```rust
struct PerNuclideGpu {
    basis: [Option<CudaSlice<f64>>; 6],
    coeffs: [Option<CudaSlice<f64>>; 6],
    grid: CudaSlice<f64>,
    discrete_level_basis: CudaSlice<f64>,
    discrete_level_coeffs: CudaSlice<f64>,
    ...
    awr: f64,
    nu_bar_const: f64,
}

struct GpuBundle {
    nuclides: Vec<Arc<PerNuclideGpu>>,
    basis_ptrs: CudaSlice<u64>,   // device array of CUdeviceptr
    coeffs_ptrs: CudaSlice<u64>,
    grid_ptrs: CudaSlice<u64>,
    n_energy_per_nuc: CudaSlice<i32>,
    ...
}
```

Kernel access changes from:

```cuda
double val = basis[basis_offsets[nuc * n_rxn + rxn] + e * rank + r];
```

to:

```cuda
const double* basis_for_nuc_rxn = (const double*)basis_ptrs[nuc * n_rxn + rxn];
double val = basis_for_nuc_rxn[e * rank + r];
```

### Wins

Cross-case sharing on a 376-case ICSBEP sweep:

| metric | today (`128888b`) | Stage C |
|---|---|---|
| Sweep load wall time | ~750 s | ~75 s |
| Total H→D bytes uploaded | ~530 GB | ~7 GB |
| Sweep peak VRAM (12 GB 3080) | ~8 GB | ~5 GB steady |

Numbers from `docs/gpu-cache-redesign.md` — predictions, need
re-measure after implementation.

### Risks

Kernel ABI changes have historically introduced silent physics
regressions. The per-level SVD rank-padding bug
(commit `1654c4d`, +500-700 pcm hot bias on fast metals) was exactly
this class — a single missed access site silently read adjacent
levels' bytes.

### Scope

- **Data restructure** in `gpu_transport.rs`:
  `upload_nuclide_data_uncached` splits into `upload_one_nuclide()`
  + `assemble_bundle()`. ~400 LOC.
- **Kernel access sites** in `gpu/cuda/transport.cu`,
  `transport_recursive.cu`, `transport_recursive_const.cu`:
  every `basis_offsets[...]` / `coeffs_offsets[...]` / equivalent
  becomes a pointer-array load. Estimated 40-60 sites across the
  three files. The photon stack uses a separate upload path and is
  not affected.
- **Validation campaign**:
  1. `cargo test --features cuda --lib gpu_recursive` (CPU↔GPU
     event-level parity).
  2. `cargo run --release --features cuda --bin level_xs_compare`
     (per-discrete-level XS A/B).
  3. `cargo run --release --features cuda --bin nu_lookup_compare`
     (ν̄(E) bit-identical CPU↔GPU).
  4. `cargo run --release --features cuda --bin chi_compare`
     (fission χ).
  5. `cargo run --release --features cuda --bin metal_stats_diag`
     (3-way CPU / GPU / OpenMC on Godiva, PMF-001, HMF-001 —
     catches +500-pcm-class regressions that unit tests miss).
  6. `cargo test --features cuda --test cuda_runs` (ICSBEP regression
     suite, ~10 min).

### Sequencing for Stage C

1. **3080 sweep validation** of `128888b` first — confirm Phase 1+2+B
   alone close the original HMF-014 stall, and quantify whether
   Stage C is genuinely needed for the 376-case workload or just a
   nice-to-have.
2. **Branch** for Stage C (`feat/per-nuclide-gpu-cache`). Don't land
   on `main` until parity is reproven.
3. **Data model first** (no kernel changes): introduce
   `PerNuclideGpu` + `GpuBundle`, keep the existing flat-pack
   `GpuNuclideData` as a *view* assembled from per-nuclide pointers
   via `cuMemcpyDtoD`. This stage is a no-op for the kernel; tests
   should still pass bit-identically.
4. **Kernel ABI** as a separate commit. After every access site is
   converted, run the full validation campaign before merge.
5. **LFU + preload weights** already plug in transparently — the
   policy module doesn't know about per-nuclide vs per-bundle, only
   about `(Key, EvictionStats, bytes)`. Bumping the cache from
   per-bundle to per-nuclide is a key-type swap.

---

## Open follow-ups from earlier phases

| item | size | status |
|---|---|---|
| Push 23 local commits to `origin/main` | credential fix | blocked on user |
| 3080 sweep against `128888b` (376 ICSBEP cases) | hours | user-side |
| Remaining lib comment cleanup (~50 files) | ~2 sessions | optional; engine-notes.md already anchors load-bearing context |
| Per-nuclide GPU cache (Stage C above) | multi-session | held |

---

## Quick-start commands

```powershell
# Validate Phase 1+2+B end-to-end on a small subset:
.\venv\Scripts\python.exe rust_prototype\bindings\python\examples\icsbep_sweep.py `
    --runner gpu --batches 20 --inactive 5 --particles 2000 `
    --filter "heu-met-fast-001_case-1|heu-met-fast-014|pu-met-fast-001"

# Full paper-quality 376-case GPU sweep (~12-15 h):
.\rust_prototype\bindings\python\examples\run_benchmark.ps1

# Hardware self-test (writes detected config to stderr):
cd rust_prototype
cargo test --features cuda --lib hardware_profile::tests::banner_self_test -- --nocapture

# Lib unit tests:
cargo test --features cuda --lib

# Run a single case with the banner:
.\venv\Scripts\python.exe rust_prototype\bindings\python\examples\icsbep_run.py heu-met-fast-014 gpu
```

## Environment knobs

| variable | default | purpose |
|---|---|---|
| `OPEN_RUST_MC_QUIET=1` | (off) | Suppress the startup banner. |
| `OPEN_RUST_MC_NUCLIDE_CACHE_BYTES=N` | (auto) | Explicit host L1 budget in bytes. |
| `OPEN_RUST_MC_NUCLIDE_CACHE_FRACTION=F` | 0.75 | Host L1 as fraction of detected RAM. |
| `OPEN_RUST_MC_GPU_BUNDLE_CACHE_BYTES=N` | (auto) | Explicit GPU bundle budget. |
| `OPEN_RUST_MC_GPU_BUNDLE_CACHE_FRACTION=F` | 0.75 | GPU bundle as fraction of total VRAM. |
| `OPEN_RUST_MC_CACHE_DIR=PATH` | tempdir | L2 disk cache root. `off` to disable. |
