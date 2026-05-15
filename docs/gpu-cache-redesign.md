# GPU XS cache redesign — design doc

Status: **design**. No code yet. Reviewed → implementation begins.

Phase 1 (`4591ba4`) + 1.5 (`097c282`) already shipped: byte-budgeted
bundle LRU + `cuMemPoolTrimTo`. This doc plans Phase 2 (per-batch
buffer reuse) and Phase 3 (per-nuclide cache with usage-weighted
eviction).

## What the current state buys us

After `097c282`:

- `nuclide_buffer_cache: VecDeque<(GpuUploadKey, Arc<GpuNuclideData>, usize)>`,
  keyed on `(rank, Vec<Arc::as_ptr>)`, byte-budgeted at 0.75 × total
  VRAM (configurable).
- Each entry is a **whole-case bundle** — every nuclide for one case
  packed into one set of flat `CudaSlice`s.
- Eviction is LRU on the *bundle* axis.
- Cross-case win only when cases re-use the same exact nuclide
  composition.

This is the floor: VRAM bounded, no leak. The ceiling is much higher
because:

- A 376-case ICSBEP sweep has ~50 unique nuclides total but ~376
  unique bundles (each case combines a different subset).
- Bundle-level cache hits are rare. Per-nuclide hit rate would be
  enormous — U-235, O-16, Fe-56, U-238 appear in dozens of cases.
- Today's cache uploads U-235 every case. At ~12 MB pointwise + SVD
  basis per cached temp column, that's ~12 MB × 376 = 4.5 GB of
  redundant device traffic per sweep. Same story for every shared
  nuclide.

## Phase 3 — per-nuclide GPU cache

### Data model

Today's flat-pack `GpuNuclideData`:

```
struct GpuNuclideData {
    all_basis: CudaSlice<f64>,   // [n_nuc × n_rxn × n_e × rank] concatenated
    basis_offsets: CudaSlice<i32>,  // [n_nuc × n_rxn]
    // ~80 more fields, all concatenated by nuclide
}
```

Proposed per-nuclide layout:

```
struct PerNuclideGpu {
    // Each field is sized for this single nuclide.
    basis: [Option<CudaSlice<f64>>; 6],  // per reaction
    coeffs: [Option<CudaSlice<f64>>; 6],
    grid: CudaSlice<f64>,
    n_energy: usize,
    awr: f64,
    nu_bar_const: f64,
    discrete_level_basis: CudaSlice<f64>,    // per-nuclide concatenated levels
    discrete_level_coeffs: CudaSlice<f64>,
    discrete_level_offsets: CudaSlice<i32>,
    // ... every per-nuclide field
}

struct GpuBundle {
    nuclides: Vec<Arc<PerNuclideGpu>>,
    // Per-bundle assembly metadata: device arrays of POINTERS into
    // the per-nuclide CudaSlices. Cheap to build (16 bytes × n_nuc
    // × n_rxn = ~5 KB per bundle).
    basis_ptrs: CudaSlice<u64>,         // CUdeviceptr for each nuc×rxn
    coeffs_ptrs: CudaSlice<u64>,
    grid_ptrs: CudaSlice<u64>,
    n_energy_per_nuc: CudaSlice<i32>,
    // Per-nuclide AWR table stays bundle-local (cheap and the kernel
    // already reads it via material index → nuclide index).
}
```

### Kernel ABI

Today:

```cuda
double val = basis[basis_offsets[nuc * n_rxn + rxn] + e * rank + r];
```

After:

```cuda
const double* p = (const double*)basis_ptrs[nuc * n_rxn + rxn];
double val = p[e * rank + r];
```

One extra device load (the pointer) but one fewer add (no offset
add). Hot path register pressure changes — needs profiling. The
extra `__ldg` is on read-only data and benefits from constant cache.

### Cache structure

```
struct PerNuclideCache {
    // Keyed on the upstream NuclideKey (file_hash, policy_hash,
    // temp_idx, format_version) — already what nuclide_cache::
    // TieredStore uses on the CPU side, so cross-context Arc identity
    // composes naturally with content identity.
    entries: HashMap<NuclideKey, Arc<PerNuclideGpu>>,
    // Per-key usage histogram for eviction priority. Updated
    // atomically on every upload + lookup.
    usage: HashMap<NuclideKey, AtomicU64>,
    // Bytes currently held (sum of every entry's device_bytes()).
    total_bytes: AtomicUsize,
    // Soft budget; same fraction-of-total knob as today.
    budget_bytes: usize,
}
```

### Eviction policy — LFU-with-recency (2Q variant)

Pure LFU is sticky: a once-hot entry that goes cold stays cached
forever. Pure LRU ignores popularity. The hybrid: **eviction score =
hits × recency_weight**, where `recency_weight = exp(-age / τ)` for
some half-life τ (e.g. τ = 100 inserts).

For warm-start cases (user knows the sweep manifest up front), the
caller can **pre-populate** the usage histogram by scanning all case
JSONs and counting nuclide appearances. Then the *first* case starts
with U-235 weighted heavily, Au-197 weighted lightly — the cache
keeps the right things even before any uploads have happened.

```
fn preload_weights(case_jsons: &[Path]) -> HashMap<NuclideKey, u64> {
    // One pass over every JSON: parse scene.materials.nuclides,
    // resolve each to a NuclideKey, bump the count. Output is a
    // histogram the harness can hand to the cache.
}

GpuTransportContext::set_preload_weights(weights);
// First call to upload_per_nuclide() seeds usage[key] = weights[key].
```

### Eviction algorithm

Pre-upload (predict bytes needed for the new bundle's *unique*
nuclides, not the whole bundle):

```
needed = sum(per_nuclide_bytes(n) for n in bundle if n not in cache)
while total_bytes + needed > budget:
    let victim = entries.iter()
        .min_by_key(|(k, _)| score(usage[k], age[k]))
        .key();
    drop(entries.remove(victim));
trim_async_mempool();
```

Where `score(hits, age) = hits / (1 + age * decay)`. Lowest score
evicts first.

### Implementation cost

- **Data restructure** in `gpu_transport.rs`: `upload_nuclide_data_uncached`
  splits into `upload_one_nuclide()` + `assemble_bundle()`. ~400 LOC
  refactor in this file alone.
- **Kernel ABI change** in `gpu/cuda/transport.cu`, `transport_recursive.cu`,
  `transport_recursive_const.cu`: every `basis_offsets[...]` /
  `coeffs_offsets[...]` access becomes a pointer-array load. Estimated
  ~40-60 access sites across the three files. Photon kernels not
  affected (they use a different upload path).
- **Validation**:
  - `gpu_recursive_parity` — per-event CPU↔GPU bit-parity.
  - `level_xs_compare` — per-level discrete-inelastic XS A/B.
  - `nu_lookup_compare`, `chi_compare` — per-reaction kinematic
    parity.
  - `metal_stats_diag` — three-way CPU/GPU/OpenMC on Godiva, PMF-001,
    HMF-001. Catches +500-pcm-class regressions that unit tests miss.
  - Re-run ICSBEP `tests/cuda_runs.rs` (6 cases, ~10 min) for the
    handbook envelope.
- **Risk**: kernel ABI changes have historically introduced silent
  physics regressions (`1654c4d` per-level SVD rank-padding bug).
  Need to land in a dedicated branch with `metal_stats_diag` as
  pre-merge gate.

### Estimated wins

For a 376-case ICSBEP sweep:

| metric                         | today (097c282)   | after Phase 3       |
|--------------------------------|-------------------|---------------------|
| Cache hits / case (cold)       | 0                 | ~80% of nuclides    |
| Per-case load wall time        | ~2 s              | ~0.2 s              |
| Sweep load wall time           | ~750 s            | ~75 s               |
| Sweep peak VRAM (12 GB 3080)   | ~8 GB             | ~5 GB (steady)      |
| Sweep peak VRAM (4 GB A1000)   | ~2.6 GB           | ~1.5 GB             |
| Total H→D bytes uploaded       | ~530 GB           | ~7 GB               |

Numbers are predictions; need re-measure post-implementation.

## Phase 2 — per-batch SoA buffer reuse

Independent of Phase 3. The current `transport_recursive` allocates
~25 device buffers per batch (`stream.clone_htod(&xs)`, etc.) and
they free at end-of-call. Per NVIDIA Best Practices §9.2, persistent
allocation + reuse is preferred.

### Design

Caller-supplied buffer pool (chosen earlier):

```
struct TransportBuffers {
    n: usize,
    fis_cap: usize,
    d_xs: CudaSlice<f64>,         // [n]
    d_ys: CudaSlice<f64>,
    d_zs: CudaSlice<f64>,
    d_dxs: CudaSlice<f64>,
    d_dys: CudaSlice<f64>,
    d_dzs: CudaSlice<f64>,
    d_e: CudaSlice<f64>,
    d_alive: CudaSlice<i32>,
    d_rng_state: CudaSlice<u64>,
    d_rng_inc: CudaSlice<u64>,
    d_fis_x: CudaSlice<f64>,      // [fis_cap]
    d_fis_y: CudaSlice<f64>,
    d_fis_z: CudaSlice<f64>,
    d_fis_e: CudaSlice<f64>,
    d_fis_w: CudaSlice<f64>,
    d_fis_count: CudaSlice<i32>,  // [1]
    d_cnt_*: CudaSlice<i32>,      // [1] each, 7 of them
    d_e_*: CudaSlice<f64>,        // [1] each, 8 of them
}

impl TransportBuffers {
    fn new(stream: &Arc<CudaStream>, n: usize, fis_cap: usize) -> Result<...>;
    fn reset_for_batch(
        &mut self,
        stream: &CudaStream,
        positions: &[(f64, f64, f64, f64)],
        directions: &[(f64, f64, f64)],
        rng_seeds: &[(u64, u64)],
    ) -> Result<()>;
}
```

`reset_for_batch` does `memcpy_htod` (in-place) instead of
`clone_htod` (allocate-then-copy), and `memset_zeros` on the
counters. Same data, no allocation.

Caller (`CudaRunner` in `dispatch.rs`) owns the buffers, passes
`&mut TransportBuffers` to `transport_recursive_with_buffers`.

### Where the pool lives

Per the earlier scope discussion: on `CudaRunner`, dropped at end of
case. Survives across batches within one case. Independent of the
nuclide cache lifecycle.

### Expected win

~30 device allocations × ~5 μs each = ~150 μs / batch saved. At 150
batches × 5 seeds = 750 batches/case, that's ~110 ms / case saved.
Small but free.

## Phase 8 — warning cleanup

| file                                      | warning                                   | fix                       |
|-------------------------------------------|-------------------------------------------|---------------------------|
| `src/photon/gpu.rs:1273-1274`             | deprecated `memcpy_stod` → `clone_htod`   | rename                    |
| `src/bin/preview_scene.rs:799`            | dead `auto_color`                         | delete or `#[allow]`      |
| `src/bin/debug_lct.rs:8`                  | unused `CellFill` import                  | delete                    |
| `src/bin/elastic_kinematics_diag.rs:29,69,169` | dead `PI`, `gpu_free_gas_elastic`, `run_free_gas_ab` | `#[allow(dead_code)]` (these are kept for future probe) |
| `src/bin/metal_stats_diag.rs:30,187,191,199` | dead `K_B_EV_PER_K`, `diff_pcm`, `diff_pct`, `report_delta` | same |
| `src/gpu_recursive.rs:97`                 | dead `eval_scratch`                       | check usage; likely keep   |

Low-risk; not interleaved with the cache work.

## Sequencing

1. **Run 3080 sweep against `097c282`** — confirms Phase 1 is
   sufficient or quantifies the residual gap. **No new code until
   this is done.**
2. **Phase 2 (buffer reuse)** — independent of the cache work, small
   surface, easy to validate.
3. **Phase 8 (warning cleanup)** — at any point; touches no logic.
4. **Phase 3 design review** — read this doc, agree on the
   per-nuclide schema + LFU eviction policy + preload API.
5. **Phase 3 implementation** — dedicated branch. Land per-nuclide
   data model first (no behaviour change, just internal restructure).
   Then kernel ABI change. Then LFU eviction. Each stage gated on
   `metal_stats_diag` 3-way passing.

## Open questions

1. **Pre-load weights API** — should the harness compute them
   (Python-side, parse every JSON) or should the engine expose a
   helper? Engine-side helper is cleaner but adds a JSON-parsing
   dependency to the Rust crate. Python-side gives the harness full
   control. Lean toward Python-side.
2. **Decay half-life τ** — needs empirical tuning. 100 inserts feels
   right for a sweep but a long-running interactive session might
   want longer.
3. **Per-nuclide vs per-(nuclide, rank) keying** — same nuclide at
   different ranks needs separate cache entries because the basis
   bytes differ. NuclideKey already has policy_hash which captures
   rank, so this falls out for free.
4. **Multi-temp same nuclide** — already separate keys via
   `temp_idx`. No change.
