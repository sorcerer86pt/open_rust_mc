# Engine notes

Extracted load-bearing comments. Source-level comments stripped during the
2026-05-15 comment-cleanup pass; the historical context, ENDF references,
commit hashes, and regression history live here instead.

Cross-reference: when adding code, check the relevant section first to
avoid re-introducing a closed bug class.

---

## `transport/simulate.rs` — eigenvalue power iteration

### `MAX_NUCLIDES` (alias for `MAX_NUCLIDES_PER_MATERIAL`)

Stack-allocated XS buffer dim for the collision hot path. Single source of
truth at `lib.rs::MAX_NUCLIDES_PER_MATERIAL` — the alias here is for
readability. Bumping requires re-checking every fixed-size `[…; MAX_NUCLIDES]`
in this module and the GPU's `MAX_NUC_PER_MAT` NVRTC define
(`gpu_recursive.rs::assemble_kernel_source`).

Sample population sizes: Godiva (3), PWR pin cell (8), full PWR-actinide
fuel material (18: U-235/238 + O-16 + Xe-135 + 14 chain nuclides for
actinide buildup + Sm/Pm/I/Cs poisoning).

### `SimConfig::verbose` — Windows stdout deadlock

`verbose=false` (default for PyO3 callers) avoids a Windows-specific
deadlock: locking stdout from a host process that also uses stdout
(e.g. Python) can hang. CLI binaries set `verbose=true` and accept the
risk.

### `SimConfig::parallel` — rayon vs Windows loader lock

`parallel=false` falls back to sequential iteration. The rayon first-use
thread-pool init can deadlock against Python's loader lock when called
from a PyO3 extension on Windows. Sequential is slower but safe.

### `SimConfig::survival_biasing`

Surface-tracking + non-thermal collision branch only. Thermal-scattering
and delta-tracking paths fall back to analog absorption regardless of
this setting.

### `SimConfig::disable_delayed_neutrons`

Ablation knob — production path always samples ~0.65 % of fission
neutrons from the soft-Watt delayed spectrum. Setting `true` ignores
ν_d(E) entirely. Closes ~196 → 19 pcm Δ_ICSBEP improvement on Godiva
when enabled; disabling is for studies, not production.

### `SurvivalBiasing` defaults

`w_min = 0.25`, `w_survive = 1.0` — OpenMC-style defaults. Particles
below `w_min` survive with probability `w / w_survive` at the higher
weight. Expectation preserved (unbiased).

---

## `gpu_transport.rs` — GPU XS upload + bundle LRU

### `GpuUploadKey` (pointer key)

Works only because `nuclide_cache::TieredStore::L1MemoryStore` returns
the same `Arc<NuclideKernels>` for the same `(file_hash, policy_hash,
temp_idx)` tuple. Callers that bypass the upstream cache get fresh Arcs
every time → cache miss; bundle LRU bounds the damage.

### `BUNDLE_CACHE_DEFAULT_FRACTION = 0.75`

Empirically fits the assembly XS upload + per-batch SoA + recursive
context on A1000 (4 GB → 3 GB budget) and 3080 (12 GB → 9 GB).
Env overrides: `OPEN_RUST_MC_GPU_BUNDLE_CACHE_FRACTION` (clamped
[0.05, 0.95]) or `_BYTES` (explicit byte count, wins).

### Eviction order — eager (pre-upload), not lazy (post-upload)

Lazy (`insert → pop_front`) doubles peak VRAM because the new upload
allocates while the old bundle is still cached. OOMs a 4 GB A1000.
Eager evicts → trims pool → uploads → inserts. Sequential single-case
sweeps reach the `Arc` strong-count-0 path; multi-threaded callers
that hold onto the previous bundle's `Arc` keep it alive across
eviction (correct, but the cache itself frees its reference).

### `trim_async_mempool` is mandatory after eviction

CUDA 11.2+ async allocator: `cuMemFreeAsync` parks freed bytes in the
stream pool. Without `cuMemPoolTrimTo(0)` after eviction,
`nvidia-smi memory.used` keeps showing the high-water mark and the
next allocation grows VRAM rather than reusing pool bytes. Pre-CUDA-11.2
devices use the sync path; `has_async_alloc=false` → no-op trim.

### `build_transport_params_vec` slot layout

`N_PARAMS = 130`. Layout is co-authored with `gpu/cuda/transport.cu`,
`gpu/cuda/transport_recursive.cu`, `gpu/cuda/transport_recursive_const.cu`.
Slot blocks (load-bearing for every kernel):

| slots | purpose | added in / refs |
|------:|---------|------------------|
|     0 | rank scalar | original |
|  1-42 | base SVD/material/SAB | original |
|  43-55 | SAB flat data | multi-TSL extension |
|  56-89 | level + ang dists | per-level fix `1654c4d` |
|  90-103 | inel CDF / pointwise / WMP | original |
| 104-109 | Watt closed-form χ (Law 11) | `chi_compare` campaign |
| 110-113 | delayed ν̄(E) — drives β(E) | delayed-neutron extension |
|     114 | fission χ PDF (OpenMC quad lin-lin) | +500-700 pcm fix |
| 115-122 | MT=91 continuum inelastic | +400 keV ⟨E_out⟩ fix |
| 123-129 | multi-slot SAB indirection | H₂O+D₂O+graphite together |
| 130-135 | Maxwell/Evaporation closed-form χ | U-233/U-234/Pu-240 fix |

### Per-level SVD rank-padding invariant (commit `1654c4d`)

Each discrete-inelastic level's SVD may have `level_rank < global P_RANK`
on sparse HDF5 grids. The device kernel reads
`basis[e_idx × P_RANK + j]` for `j ∈ [0, P_RANK)`. Pad each level's
basis to `[n_e × P_RANK]` with zero columns, and pad coeffs to
length `P_RANK` with zeros. Skipping this reads adjacent levels' bytes
→ silent +500-700 pcm hot bias on fast-metal benchmarks.

---

## `nuclide_cache::l1_memory` — host RAM LRU

### Budget knobs

`OPEN_RUST_MC_NUCLIDE_CACHE_BYTES` (explicit) or `_FRACTION` × 16 GiB
stand-in. Default 4 GiB. The 16 GiB stand-in is hard-coded because
adding a `sysinfo` dep was out of scope for the byte-budget patch
(commit `1b91b78`). Override explicitly on bigger boxes.

### Dropped semantics: `Vec<Arc<NuclideKernels>>` per key

The pre-`1b91b78` DashMap stored a `Vec` "for future bulk-dump APIs".
Nothing ever populated index 1. The refactor stores a single Arc per
key — matches the only semantic actually used.

---

## `gpu_recursive.rs` — recursive geometry + transport

### `TransportBuffers` lifetime

Tied to one `(n, fis_cap, n_materials, n_lattices, params_len)` tuple.
Cross-case reuse requires identical sizes; the `CudaRunner` builds
one per case lifetime. Drop releases every `CudaSlice`.

### `transport_recursive` vs `transport_recursive_with_buffers`

The unpooled `transport_recursive` is the backwards-compat path for
bin diagnostics; it allocates a throw-away `TransportBuffers` each
call. Production callers (`CudaRunner`) use the pooled entry point.

### Lattice-override dummy fillers

The recursive demo path doesn't use distributed materials; the
override scratch is `[-1; n_lattices+1]` / `[0; n_lattices+1]`
filler. When the recursive path gains real `RectLattice.material_overrides`
support, swap the `htod_i` calls in
`transport_recursive_with_buffers` for the real upload.

### GPU recursive kernel pinned to `sm_86`

NVRTC arch hardcoded for Ampere (RTX A1000 / 3080). Needed for
`atomicAdd(double*, double)`. Bumping to support older / newer
architectures requires regenerating the kernel and re-validating
parity.

### CudaRunner buffer pool — `RefCell<Option<TransportBuffers>>`

`EigenvalueRunner::run` takes `&self`; the runner is `!Sync` via
`Box<dyn Fn>`, so the `RefCell` runtime borrow check is sound. Build
on first batch, reuse for all subsequent batches in the case.

---

## Geometry invariants (kept inline; cross-link only)

The hard contracts live in `CLAUDE.md § Invariants`. Key ones:

- `RectLattice::local_position` is element-CENTRE-relative
  (OpenMC convention), not corner-relative.
- `MAX_NUCLIDES_PER_MATERIAL = 32` — single source of truth at
  `lib.rs`.
- `SimLimits` separates engine policy from per-run user intent.
- Initial-source sampler is material-aware, not cell-order-aware
  (`try_initial_source_in_materials`).
- `N_PARAMS = 130` on `transport.cu` / `gpu_transport.rs`.
- GPU recursive kernel pinned to `sm_86`.
- CPU transport uses `TransportCtx` + rayon `fold().reduce()`.
