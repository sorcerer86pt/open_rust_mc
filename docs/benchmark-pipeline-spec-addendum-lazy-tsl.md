# Addendum — Lazy TSL upload (bracket-pair filter)

**Companion to:** `docs/benchmark-pipeline-spec.md` (v2)
**Status:** design delta, no implementation yet. Lands inside Phase 1
of the v2 migration plan without expanding its surface area.
**Motivation:** the 2026-05-25 RunPod 10 GB GPU sweep showed the
host uploading all 11 Be-9 temperatures (~1 GB of VRAM) for cases
where the material kT bracket selects only 2 of them. Same waste
hits any TSL with a wide temperature grid (H-in-H₂O, D-in-D₂O,
graphite). On the 5090 it merely costs VRAM that's available
anyway; on a 10 GB device it pushes Be-S(α,β) cases into the
defensive-clear band before any kernel runs.

This addendum specifies the minimal change to `upload_sab_data_multi_cached`
and its planning helpers so Stage 3 uploads **only the bracket pair
each material actually selects from**.

---

## 1. Physics correctness — why "lazy" is safe here

`thermal::ThermalScatteringData::select_temperature(T, ξ)` at
`thermal.rs:147` performs **stochastic interpolation between two
bracketing temperatures**:

```rust
let f = (kt - kts[i]) / (kts[i + 1] - kts[i]);
if xi < f { i + 1 } else { i }
```

where `i` is the largest index with `kts[i] ≤ kt`. The returned
index is therefore always in `{i, i+1}` when `kt` falls inside the
grid, or clamped to an endpoint when it falls outside.

**Implication.** For a given material temperature `T_mat`, the
kernel reads slots `i` and `i+1` only. Slots outside that pair are
never selected and uploading them is pure VRAM waste. Stochastic
interpolation makes lazy upload safe **iff both bracket slots are
present**; loading only one would silently bias the temperature.

Edge cases (same code path):

| `T_mat` position | `select_temperature` outcome | Slots needed |
|---|---|---|
| Below `kts[0]` | `f < 0`, always returns `0` | `{0}` (but upload `{0, 1}` for symmetry) |
| Above `kts[last]` | early-return `last` | `{last}` |
| Single-T grid | `kts.len() == 1` early-return `0` | `{0}` |
| Strictly between `kts[i]` and `kts[i+1]` | stochastic `{i, i+1}` | `{i, i+1}` |

**Multi-material union.** When two materials share the same TSL but
sit at different temperatures (e.g. cell A at 294 K, cell B at
600 K with Be-9 in both), the required slot set is the **union of
bracket pairs**: `{2, 3} ∪ {4, 5} = {2, 3, 4, 5}`. Stage 3 computes
this union at material-resolve time.

---

## 2. Four deltas, all fitting inside Phase 1

The v2 spec §4.4 and §5.3.2 already enumerate the surgery that
Phase 1 performs on the SAB upload path:

- new `estimate_sab_device_bytes` helper,
- `&CudaStream` parameter on every `upload_*`,
- `SabCacheKey` extension implied by the cache class.

This addendum extends each item with one extra parameter or field.
No new surface; each modified entity already had to be touched.

### 2.1 `estimate_sab_device_bytes` — add `temp_filter`

v2 spec signature:

```rust
pub fn estimate_sab_device_bytes(
    slots: &[(Arc<ThermalScatteringData>, usize)],
    n_nuc:  usize,
) -> usize
```

Addendum:

```rust
pub fn estimate_sab_device_bytes(
    slots:       &[(Arc<ThermalScatteringData>, usize)],
    temp_filter: &[Vec<usize>],   // per-slot kts indices to count
    n_nuc:       usize,
) -> usize
```

Loop body changes from `for t in 0..tsl.kts.len()` to
`for &t in &temp_filter[slot_i]`. Header overhead (per-slot fixed
cost) stays the same. Bragg-edge and inelastic-table contributions
are gated by which `t` index is included.

**Backwards-compat shim** for callers that haven't been migrated:

```rust
pub fn estimate_sab_device_bytes_all_temps(
    slots: &[(Arc<ThermalScatteringData>, usize)],
    n_nuc: usize,
) -> usize {
    let filter: Vec<Vec<usize>> = slots.iter()
        .map(|(tsl, _)| (0..tsl.kts.len()).collect())
        .collect();
    estimate_sab_device_bytes(slots, &filter, n_nuc)
}
```

Phase 1 lands the new signature + the shim, with all current call
sites using the shim → bitwise-identical behaviour. Phase 2/3
swaps Stage 3 over to the real filter.

### 2.2 `upload_sab_data_multi_cached` — add `temp_filter`

v2 spec already extends this with `&CudaStream` (Phase 1 §4.4).
Addendum adds one more parameter alongside:

```rust
pub fn upload_sab_data_multi_cached(
    &self,
    slots:       &[(Arc<ThermalScatteringData>, usize)],
    temp_filter: &[Vec<usize>],
    n_nuc:       usize,
    stream:      &CudaStream,
) -> Result<Arc<GpuSabData>, Box<dyn Error>>
```

Inside the function:

- The per-slot accumulator loop iterates `&temp_filter[slot_i]`
  instead of `0..tsl.kts.len()`.
- `slot_count_per_nuc[nuc_idx]` becomes the **filter length**, not
  `tsl.kts.len()`.
- A new on-device array `kts_filtered_flat: Vec<f64>` ships the
  actual `kT` values that correspond to the uploaded slots, in
  upload order. Kernel reads from this array for the
  `select_temperature` lookup; it must not assume the full grid.
- `slot_kts_off_per_nuc[nuc_idx]` is a new int slot pointing at
  the start of that nuclide's filtered `kT` window inside
  `kts_filtered_flat`. Mirrors how `slot_per_nuc` indexes the
  payload arrays.

**Kernel change** (gpu/cuda/transport.cu, SAB path): the device-side
analog of `select_temperature` currently reads `mat_kT[mat]`
against the slot's full `kts` array. Switch it to:

```cuda
int n   = slot_count_per_nuc[nuc];
int off = slot_kts_off_per_nuc[nuc];
// bracket against kts_filtered[off..off+n] exactly as the CPU does
```

This is a one-array swap, not a logic rewrite. The bracket math is
identical.

**Backwards-compat shim** mirrors §2.1.

### 2.3 `SabCacheKey` — include filter in the hash

Current key (Phase 1 keeps as-is):

```rust
pub struct SabCacheKey {
    slots: Vec<(usize /*arc_ptr*/, usize /*nuc_idx*/)>,
    n_nuc: usize,
}
```

Addendum:

```rust
pub struct SabCacheKey {
    slots: Vec<(
        usize,        // arc_ptr (Arc::as_ptr of the TSL)
        usize,        // nuc_idx
        Vec<usize>,   // temp_filter for this slot — order-sensitive
    )>,
    n_nuc: usize,
}
```

Consequences:

- Two cases at the **same** `T_mat` hit the same cache entry (good
  — that's the warm path for a corpus where most materials are
  room-temperature).
- Two cases at **different** `T_mat` for the same TSL get separate
  cache entries (correct — the device payloads are genuinely
  different).
- Memory cost of the key itself is negligible (each `Vec<usize>` is
  typically 2 elements).

Phase 1 keeps the old key during the shim period. Phase 2 swaps in
the new key when callers start passing real filters.

### 2.4 New helper — `compute_tsl_temp_filter`

Lives in the `benchmark::case_bundle` module (v2 §4.3 already
introduces the module). Pure function, ~20 lines:

```rust
/// For each nuclide that carries thermal scattering data, compute
/// the union of bracket pairs across every material that uses it.
///
/// Returned shape mirrors the `slots` argument to
/// `upload_sab_data_multi_cached`: outer index is the slot
/// (`(tsl, nuc_idx)`), inner is the sorted list of kts indices to
/// upload.
pub fn compute_tsl_temp_filter(
    resolved: &ResolvedMaterials,
) -> Vec<Vec<usize>> {
    // For each (tsl, nuc_idx) pair in the resolved provider:
    //   collect every material.temperature that references it;
    //   for each T, compute bracket (i, i+1) using the same logic
    //     as ThermalScatteringData::select_temperature, minus ξ;
    //   union the bracket indices;
    //   sort + dedup.
    //
    // Edge cases (mirror select_temperature):
    //   - kts.len() == 1     → [0]
    //   - T below kts[0]     → [0, 1]   (1 not strictly needed but
    //                                    cheap insurance against
    //                                    floating-point edge fuzz)
    //   - T above kts[last]  → [last]
    todo!()
}
```

**Bracket helper** (factored out of `select_temperature` for reuse
without an RNG draw):

```rust
impl ThermalScatteringData {
    /// Deterministic bracket — returns the `(lo, hi)` pair such
    /// that `select_temperature(T, ξ)` returns either `lo` or `hi`.
    /// When T is outside the grid, `hi == lo`.
    pub fn bracket(&self, temperature_k: f64) -> (usize, usize) {
        // Same loop body as select_temperature but returns the pair
        // instead of sampling.
        // ...
    }
}
```

Land `bracket` in Phase 1 (purely additive on `ThermalScatteringData`).
`select_temperature` can then be expressed in terms of it without
behaviour change.

---

## 3. Where this plugs into Stage 3

§5.3.2 of the main spec lists the per-case load pipeline. The
addendum inserts one step between steps 1 and 2:

> 1. Resolve materials. (unchanged)
>
> **1.5. Compute TSL temperature filter.**
>     `let temp_filter = compute_tsl_temp_filter(&resolved);`
>     Pure function over already-resolved data; no I/O, no GPU.
>
> 2. Pre-flight VRAM check. The `predicted` calculation now reads:
>     ```
>     predicted = Σ nuc.approx_device_bytes()
>               + gpu_t.estimate_sab_device_bytes(
>                     &sab_slots, &temp_filter, n_nuc)
>               + flat_pack_overhead(resolved.nuclides.len())
>               + particle_bank_bytes(sim_config.particles_per_batch)
>     ```
>
> 3. H→D upload — adds `&temp_filter` to the SAB call:
>     ```rust
>     gpu_t.upload_sab_data_multi_cached(
>         &sab_slots, &temp_filter, n_nuc, &stream_transfer)?
>     ```
>
> 4-5. (unchanged)

The `temp_filter` rides inside the `CaseBundle` so that the
GpuExecutor doesn't need to recompute it.

---

## 4. Expected impact

For Be-9 (11 tabulated temperatures) at a room-temperature material:

- Today: 11 slots, ~1 GB device payload (continuous-inelastic
  E_out / mu tables dominate).
- With filter: 2 slots (294 K + 296 K bracket for T_mat = 293.6 K),
  ~180 MB device payload.
- **Savings: ~82% of TSL VRAM** for that nuclide.

For H-in-H₂O / D-in-D₂O (smaller per-T payload, also wide T grid)
the relative savings are similar; the absolute savings are smaller
because the payloads themselves are smaller.

For a 10 GB GPU running the ICSBEP thermal corpus, this is the
difference between "Be-bearing cases compete with actinide bundles
for VRAM and trigger defensive clears" and "Be-bearing cases sit
comfortably under the 0.85 × free budget."

Out of scope but worth noting: depletion / kinetics with time-varying
material temperature will need to either (a) recompute the filter
per timestep and re-upload, or (b) pre-load a widened bracket. The
v2 spec's watchdog rebuild path handles the worst-case device
state already; the per-timestep refresh is a small follow-on.

---

## 5. Phase mapping

| Phase | Lazy-TSL work in this phase |
|---|---|
| Phase 1 — entities + stream plumbing | Add `bracket()` to `ThermalScatteringData`. Add `temp_filter` parameter to both `estimate_sab_device_bytes` and `upload_sab_data_multi_cached`. Provide all-temps shim wrappers. Switch in-engine `SabCacheKey` to the 3-tuple form, keyed at `[(0..kts.len()).collect()]` from the shim. Tests stay green because every caller goes through the shim. |
| Phase 2 — sequential pipeline | `parse_case_json` → `compute_tsl_temp_filter` stub returning the all-temps filter. Still bit-identical to Phase 1. |
| Phase 3 — real pipeline parallelism | `compute_tsl_temp_filter` returns the actual bracket-union. Stage 3 passes it through. VRAM gate sees the smaller footprint. Smoke target: heu-comp-inter-003 case-1, Be-9 SAB payload < 250 MB measured via `gpu_debug_metrics()`. |
| Phase 4 — production polish | Add per-case telemetry field `sab_slots_uploaded`, `sab_slots_skipped` so the JSONL records prove the filter is working in the wild. |

---

## 6. Acceptance criteria (extends §9)

In addition to the v2 acceptance criteria:

6. On the 10 GB RTX 3080 / A1000-class device, `heu-comp-inter-003`
   case-1 (Be-9 reflector) completes Stage 3 upload with
   `mem_get_info().free` staying above 5 GB throughout. Today this
   case drops free VRAM to ~3.5 GB during SAB upload.

7. JSONL telemetry shows `sab_slots_uploaded[Be9] ≤ 3` for any
   room-temperature ICSBEP case. (3 covers the bracket + one
   safety neighbour for boundary materials.)

8. CPU/GPU k_eff parity envelope unchanged for the 7 Be-bearing
   cases in the ICSBEP corpus (none exceed the existing
   `|Δk| ≤ max(150 pcm, 2σ_combined)` envelope).

---

**Document status:** addendum draft. Ready to be folded into Phase 1
of the v2 migration plan. Does not require a separate Phase or
review cycle — every change rides on a Phase-1 line item that was
already going to be touched.
