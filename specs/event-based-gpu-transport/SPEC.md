# Spec: Event-Based GPU Transport

**Created:** 2026-05-18  
**Branch:** feat/gpu-perf-and-per-nuclide-tally  
**Motivation:** ncu profile of `transport_recursive_persistent` on PWR-17×17 shows
active threads/warp = **6.2 / 32** (82% idle) and 75% of cycles stalled on MIO
short-scoreboard BRX branches — the reaction-type dispatch (elastic / inelastic /
fission / n2n / capture) causes every warp to diverge.

Event-based transport (Tramm 2024): sort particles by reaction type before executing
reactions → all threads in a reaction kernel execute the same code path → zero
divergence inside each kernel.

---

## Architecture

### Key decision

Full replacement of the persistent history-based kernel. Default particles per batch
bumped 5k → 50k to amortise per-kernel-launch overhead.

### New global particle state (added to `TransportBuffers`)

**Coordinate stack** — replaces `GrCoord stack[GR_MAX_DEPTH]` per-thread local memory.
`GrCoord` = 7 ints + 3 doubles. Stored as 10 SoA arrays, each of size `n × GR_MAX_DEPTH`:

```
d_stack_universe    i32[n × 4]
d_stack_cell_idx    i32[n × 4]
d_stack_has_lattice i32[n × 4]
d_stack_lattice_id  i32[n × 4]
d_stack_lat_ix      i32[n × 4]
d_stack_lat_iy      i32[n × 4]
d_stack_lat_iz      i32[n × 4]
d_stack_offx        f64[n × 4]
d_stack_offy        f64[n × 4]
d_stack_offz        f64[n × 4]
d_depth             i32[n]
```

Memory: ~10.4 MB at n=50k.

**Per-event data** — output of geometry kernel, input of reaction kernels:

```
d_event_type     i32[n]   -1=none/dead, 0=elastic, 1=inelastic,
                           2=fission, 3=n2n, 4=n3n (capture handled inline)
d_event_hit_nuc  i32[n]   nuclide index
d_event_mat      i32[n]   material index
d_event_kT       f64[n]   cell kT at collision
d_event_hit_Ni   f64[n]   atom density of hit nuclide
d_event_urr_xi   f64[n]   URR random draw (reused in reaction kernels)
```

**Partition buffers:**

```
d_type_count    i32[5]   atomic count per type (zeroed per step)
d_type_offsets  i32[6]   exclusive prefix sums (computed on host, uploaded)
d_type_scatter  i32[5]   atomic scatter position per type (zeroed per step)
d_sorted_idx    i32[n]   particle indices sorted by event_type
```

---

## New CUDA kernels (added to `gpu/cuda/transport_recursive.cu`)

All kernels share existing device functions via `assemble_kernel_source()`.

### Kernel 1: `gr_init_stacks`

Called once per batch before the event loop. For each alive particle: call
`gr_find_cell` on initial position, write result to `d_stack_*` / `d_depth`. Particles
outside geometry: `alive[i]=0`, `cnt_leak++`.

### Kernel 2: `gr_trace_and_sample`

The core geometry + reaction-selection kernel. One thread per particle:

1. Load coord_stack from global arrays
2. `tr_effective_material` → mat
3. **Void path**: `gr_trace_step` → handle BC (vacuum/reflect/transmit) inline, update stack
4. **Material path**:
   - eval macro XS via `eval_nuclide_macro_xs` loop → `sum_t`, `nuc_t[]`
   - sample `d_coll = -log(ξ) / sum_t`
   - `gr_trace_step` → `d_s`
   - if `d_s < d_coll`: surface crossing → handle BC inline, continue
   - else collision: advance position, sample `hit_nuc`, sample reaction type (cumulative XS)
   - write `event_type / hit_nuc / mat / kT / Ni_hit / urr_xi`
   - `atomicAdd(&d_type_count[t], 1)`
5. **Capture handled inline**: `alive[i]=0` (no reaction kernel launched)
6. Write pos/dir, coord_stack, rng back to global
7. Flush per-thread tally accumulators via `atomicAdd`

Surface crossings where particle survives set `event_type=-1` (no reaction needed).

### Kernel 3: `gr_partition`

```c
int t = event_type[tid];
if (t < 0 || !alive[tid]) return;
int pos = atomicAdd(&d_type_scatter[t], 1);
d_sorted_idx[d_type_offsets[t] + pos] = tid;
```

### Kernel 4: `gr_elastic_event` (count[ELASTIC] threads)

Loads (E, rng, hit_nuc, mat, kT, urr_xi) via `sorted_idx`. Executes the elastic block
from transport_recursive.cu lines 312–395 verbatim:
SAB check → `sab_sample` or free-gas thermal → hard-sphere kinematics.
Updates E, direction. **Zero warp divergence** (all threads elastic).

### Kernel 5: `gr_inelastic_event` (count[INELASTIC] threads)

Executes the `do_inelastic` block (lines 489–642 verbatim):
level selection via CDF or SVD-XS → MT=91 continuum outgoing energy → CM→lab
kinematics via `sample_level_angular`. Updates E, direction.

### Kernel 6: `gr_fission_event` (count[FISSION] threads)

Sample ν via `nu_bar_lookup`, loop to bank ν secondaries via `atomicAdd` to fission
bank, set `alive[i]=0`. Extracted from fission block lines 461–481.

### Kernel 7: `gr_multi_event` (count[N2N] + count[N3N] threads)

Both N2N and N3N participate. Per-thread secondary count from `event_type[i]`
(1 or 2 — one small non-divergent comparison). Bank secondaries, primary continues
with sampled outgoing energy/direction. Extracted from lines 400–460.

---

## Rust driver loop (`gpu_recursive.rs`)

```rust
// Once per batch:
gr_init_stacks_kernel(...)

for step in 0..max_events_per_history {
    // zero d_type_count[5], d_type_scatter[5]
    gr_trace_and_sample_kernel(n_particles threads, ...)

    // download d_type_count[5] (24 bytes DtoH)
    let counts = stream.clone_dtoh(&d_type_count)?;
    if counts.iter().sum::<i32>() == 0 { break; }  // all dead

    // host: compute prefix sums → type_offsets[6]
    // upload type_offsets (24 bytes HtoD)

    gr_partition_kernel(n_particles threads, ...)

    if counts[ELASTIC]    > 0 { gr_elastic_event(counts[ELASTIC] threads)   }
    if counts[INELASTIC]  > 0 { gr_inelastic_event(counts[INELASTIC] threads)}
    if counts[FISSION]    > 0 { gr_fission_event(counts[FISSION] threads)   }
    if counts[N2N] + counts[N3N] > 0 { gr_multi_event(combined threads)     }
}

// download fission bank + tallies → RecursiveTransportBatch (unchanged API)
```

DtoH per step: **1 × 24 bytes** (type counts) vs current 20 DtoH per batch.

---

## Files to modify

| File | Change |
|------|--------|
| `rust_prototype/gpu/cuda/transport_recursive.cu` | Add 7 kernels after existing `transport_recursive_persistent`; keep old kernel commented for reference during validation |
| `rust_prototype/src/gpu_recursive.rs` | Extend `TransportBuffers` with new fields; add `transport_event_based()` inner function; call from `transport_recursive_with_buffers` |
| `rust_prototype/src/transport/dispatch.rs` | Change default particles-per-batch 5k → 50k in `CudaRunner` |

---

## Device functions reused (no new math needed)

| Function | Source file | Used by |
|----------|-------------|---------|
| `eval_nuclide_macro_xs` | transport.cu | gr_trace_and_sample |
| `gr_find_cell` | geom_recursive.cu | gr_init_stacks |
| `gr_trace_step` | geom_recursive.cu | gr_trace_and_sample |
| `tr_effective_material` | transport_recursive.cu | gr_trace_and_sample |
| `sab_sample`, `sab_select_slot` | transport.cu | gr_elastic_event |
| `sample_angular_dist`, `sample_level_angular` | transport.cu | gr_inelastic_event |
| `sample_fission_emit_energy` | transport.cu | gr_fission_event |
| `rotate_direction`, `gr_reflect_direction` | transport.cu / geom_recursive.cu | multiple |
| `nu_bar_lookup` | transport.cu | gr_fission_event |
| `pcg_uniform` | transport.cu | all kernels |

---

## Implementation order

1. `TransportBuffers` additions (Rust) — add all new fields
2. `gr_init_stacks` kernel — simple, verifiable immediately
3. `gr_trace_and_sample` kernel — largest piece (~350 lines, extract from existing loop)
4. `gr_partition_kernel` — ~20 lines
5. Four reaction kernels — ~400 lines total, mostly verbatim extraction
6. Rust driver loop — ~100 lines
7. Smoke test → ICSBEP 6-case sweep → ncu re-profile

---

## Acceptance criteria

| AC | Check |
|----|-------|
| 6/6 ICSBEP regression cases PASS (same envelope as before) | `icsbep_run.py <case> gpu` × 6 |
| ncu active_threads/warp > 20 (vs 6.2 baseline) on elastic kernel | `ncu --section WarpStateStats gr_elastic_event` |
| ns/particle improves vs 30 400 ns/p baseline on PWR-17×17 50k particles | `gpu_assembly_keff --particles 50000 --seeds 3` |
| `cargo test --lib --features cuda` still 384/384 green | `cargo test --lib --features cuda` |
