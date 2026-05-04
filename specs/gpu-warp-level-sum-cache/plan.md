# Implementation Plan: GPU Per-Warp Level-Sum Cache

**Created**: 2026-05-04
**Source**: README.md, resume.md, paper §gpu (`paper/sections/gpu.tex`),
paper §threats (`paper/sections/threats.tex`), paper §conclusion
**Status**: Ready for review

---

## 1. Executive summary

### Goal

Close the GPU SVD vs GPU pointwise gap on PWR pin cell by caching the
per-discrete-level SVD reconstructions and the level-sum in shared memory,
keyed on `(nuclide_idx, energy_idx)` — the exact engineering item flagged
in `paper/sections/gpu.tex:19` and `paper/sections/threats.tex:55-57`,
where the paper explicitly says:

> "A per-warp cache keyed on $(n, E_\text{idx})$ after the energy sort
> would close the gap, and we note this as engineering work that the
> present paper does not cover."

Currently GPU SVD = $7\,754$ ns/p vs GPU pointwise = $5\,967$ ns/p
(`paper/sections/pwr.tex` Table~\ref{tab:keff_pwr_desktop}, ratio
$1.30\times$). The cause is mechanical: four clad nuclides
(Zr-90, 91, 92, 94) lack the MT=4 total-inelastic block in
ENDF/B-VII.1 HDF5, so total inelastic is *synthesised at lookup time*
by summing 13 discrete-level SVD kernels. The current CUDA hot path
(`rust_prototype/gpu/cuda/transport.cu:1297-1340`) executes
`svd_reconstruct` **26 times per inelastic collision** (sum-pass +
re-select-pass on up to 64 levels), with the in-source comment
explicitly admitting "SVD cost doubles vs the old cached array".

After the existing energy sort (`gpu_transport.rs:1432-1475`),
threads in a warp are highly likely to share the same
`(nuc_idx, e_idx)` bucket; a small shared-memory cache eliminates the
redundant reconstructions.

### Scope

**In scope:**
- Per-warp shared-memory cache for level-sum and per-level XS in
  `transport.cu` inelastic branch.
- Cache invalidation key on `(nuc_idx, e_idx)` warp-broadcast after
  the energy sort.
- New benchmark binary that measures GPU SVD vs GPU pointwise on
  PWR with the cache on/off (compare-mode flag).
- Bit-parity validation against the existing CPU SVD path
  (`--force-svd` harness already exists).

**Out of scope:**
- CPU-side level-sum caching (CPU is not the bottleneck; rayon
  amortises the redundant work — paper §gpu confirms "On CPU the
  scalar pipeline amortises this").
- Anything in the smooth (non-discrete-level) reaction path.
- Multi-nuclide cache (each warp typically has one hit_nuc per
  inelastic event; nuclide is the bucket key, not a cache axis).
- GPU hybrid / GPU WMP — paper §threats §gpu_hybrid: not validated
  in transport loop, separate workstream.
- Photon transport caches.
- New physics, new benchmarks beyond PWR pin cell.

### Affected components

| Component | Path | Impact |
|-----------|----------------------------------------------------------|--------|
| CUDA src  | `rust_prototype/gpu/cuda/transport.cu`                   | High   |
| Rust GPU  | `rust_prototype/src/gpu_transport.rs`                    | Medium |
| Bench bin | `rust_prototype/src/bin/gpu_pwr_bench.rs` (existing)     | Low    |
| New bench | `rust_prototype/src/bin/gpu_warp_cache_bench.rs` (new)   | New    |
| Paper     | `paper/sections/gpu.tex`, `paper/sections/threats.tex`   | Update |

---

## 2. Acceptance criteria

| AC  | Description                                                      | Verified by                                             | Tasks |
|-----|------------------------------------------------------------------|---------------------------------------------------------|-------|
| AC1 | Cache enabled GPU SVD throughput is within 5% of GPU pointwise   | `gpu_warp_cache_bench` ns/p ratio                       | 3.1, 3.2, 4.3 |
| AC2 | k_inf with cache matches no-cache GPU SVD within combined SEM    | Side-by-side run, `\|Δk\| ≤ 3 SEM_combined`             | 4.1, 4.2 |
| AC3 | Cache miss path is bit-parity with current GPU SVD               | `gpu_compton_validate`-style check on inelastic samples | 2.4, 4.1 |
| AC4 | Compiles with `--features cuda` and without (CPU build untouched)| `cargo build --release` and `--features cuda`           | 1.4, 5.2 |
| AC5 | All `cargo test --lib` library tests still green                 | `cargo test --lib`                                      | 5.1   |
| AC6 | clippy clean and fmt clean                                       | `cargo clippy -D warnings`, `cargo fmt --check`         | 5.3   |
| AC7 | Documented in paper §gpu and §threats — gap-claim updated        | Diff to `paper/sections/gpu.tex`                        | 6.1   |

---

## 3. Codebase context

### Hot path today

The redundant work lives in `transport.cu:1297-1340`. Two passes over
up to 64 discrete levels per inelastic collision:

```cpp
// Pass 1: sum
for (int l = 0; l < lev_cap; l++) {
    if (E >= thr[gl] && has_k[gl]) {
        lxs_sum += svd_reconstruct(basis, coeffs, e_idx, rank);
    }
}
// Pass 2: re-compute and select on the fly
for (int l = 0; l < lev_cap; l++) {
    double lxs = svd_reconstruct(basis, coeffs, e_idx, rank);
    run += lxs;
    if (xi_l < run) { selected = l; break; }
}
```

In-source comment confirms the cost doubling is deliberate
(register pressure trade-off): *"SVD cost doubles vs the old cached
array, but registers stay tight and 256-byte stack load disappears"*.

The fix is not to reintroduce a per-thread stack array (which causes
DRAM spills per NVIDIA BPG §10.2) but to use **shared memory** keyed
on `(nuc_idx, e_idx)` — broadcast across the warp after the energy
sort puts neighbours in the same bucket.

### Energy sort already in place

`gpu_transport.rs:1432-1475`: bin-count → CPU prefix-sum → scatter
into `d_compact_idx_sorted`, then swap. After this swap, lane $i$ in
warp $w$ has energy in bin $b_i$; bin granularity is `n_bins = 256`
(`BLOCK_SIZE` constant near the top of `gpu_transport.rs`). Within a
warp of 32 threads, energy span is much smaller than one bin → most
lanes share the same `e_idx` after `energy_index()`. This is the
sort that the paper said the cache would key on.

### SVD reconstruction cost

`transport.cu:324-356`: `svd_reconstruct` is a length-`rank` FMA
(rank=5 production). Per call: ~5 FMAs + 1 `exp2`. The cost surfaces
in CUDA profiles as `__expf`/`exp2` sequences; cutting the count by
half by deduplicating is roughly the speedup the paper expects.

### Reference implementations to follow

| File | Pattern used | Why relevant |
|------|--------------|--------------|
| `transport.cu:1297-1340` | Two-pass discrete-level sampler (current) | The exact site to replace |
| `transport.cu:1002-1021` | `svd_reconstruct_interp` (smooth-XS hot path) | Same kernel, cached differently — energy interp model |
| `gpu_transport.rs:1432-1475` | Energy bin-count + scatter sort | Already establishes the warp-coherence the cache exploits |
| `src/photon/gpu.rs` (per the README) | `__shared__` cache patterns from photon kernels | Existing in-tree shared-memory style |
| `cuda_bench/svd_gpu_bench.cu` | Microbenchmark harness | Pattern for `gpu_warp_cache_bench` |

### Files to modify

- `rust_prototype/gpu/cuda/transport.cu` — add shared-memory cache,
  rewrite the two-pass loop into one pass against the cache,
  fall through to `svd_reconstruct` on miss.
- `rust_prototype/src/gpu_transport.rs` — bump
  `shared_mem_bytes` on the persistent-kernel `LaunchConfig`,
  expose a `--no-warp-cache` flag for A/B testing,
  thread it into the kernel as a `Params` int.
- `rust_prototype/src/bin/pwr_pincell.rs` (or a new
  `gpu_warp_cache_bench.rs`) — three-way GPU comparison
  (pointwise / SVD-no-cache / SVD-with-cache).

### Conventions found

- CUDA source is `include_str!`'d into Rust at compile time
  (`gpu_transport.rs:25`); no separate build step.
- `Params` struct `P_*` index constants must match CUDA `N_PARAMS`
  (`gpu_transport.rs:16`).
- `__ldg(...)` is used everywhere for read-only loads — keep it.
- All cache misses must reproduce CPU bit-parity (the
  `--force-svd` harness will catch any drift).

---

## 4. Multi-perspective analysis

### Architect

- **Boundary respected**: cache lives entirely inside the persistent
  kernel and the params block; no new HDF5 reader work, no new
  on-device tables. Falls back to current path on miss.
- **Failure mode**: if the cache key collides incorrectly, k_inf
  drifts. AC3 (bit-parity) and AC2 (k_inf within SEM) gate this.
- **Scaling**: one warp = 32 lanes × 2 doubles (level-sum + selected
  Q) per cache slot, plus 64 doubles (per-level XS) for the
  selection re-walk = $(32 \cdot 8) + (64 \cdot 8) = 768$ bytes per
  warp; at 8 warps/block = $6\,144$ bytes shared, well under the
  48 KB Ampere limit.

### Developer

- **Atomic decomposition**: cache-add → cache-use in pass-1 →
  cache-use in pass-2 → flag → bench → paper update. Each step
  reversible.
- **Testing strategy**: bit-parity harness exists
  (`gpu_compton_validate.rs` pattern), reuse for inelastic.
- **Risk**: shared-memory race between warps. Mitigate by *per-warp*
  slots indexed by `threadIdx.x >> 5` so each warp owns its slice.
- **Alternative considered**: per-thread register cache. Rejected
  because the existing comment in `transport.cu:1306` documents why
  this regresses (256-byte stack spill). Shared memory is the
  documented correct path.

### QA

- AC1 measurable: ns/p in `gpu_warp_cache_bench` against the no-cache
  baseline — same particles, same seed.
- AC2 measurable: $|\Delta k_\infty|$ with combined SEM at 5–10 seeds.
- AC3 measurable: same energies, same seeds, in/out of cache,
  per-collision XS sum recomputed offline matches.
- Regression risk: any change in `Params` layout breaks every CUDA
  binary. Mitigated by adding new fields at the end and bumping
  `N_PARAMS` in lockstep.

---

## 5. Implementation phases

### Phase 1 — Cache scaffolding

**Objective:** wire shared memory and the cache key broadcast without
changing physics yet. The kernel still calls `svd_reconstruct` for
every level, but now writes to and reads from a cache that is
guaranteed-cold (every access is a miss). Output is bit-identical.

**Prerequisite:** none.

- [ ] **Task 1.1**: Add `P_WARP_CACHE_ENABLE` int param at end of `Params`
  - **Files**: `rust_prototype/gpu/cuda/transport.cu` (top), `rust_prototype/src/gpu_transport.rs` (`P_*` consts)
  - **Pattern**: Follow existing `P_*` index pattern at top of `transport.cu` and the matching `P_*` constants in `gpu_transport.rs`
  - **Acceptance**: `N_PARAMS` bumps by 1 on both sides; default value `1` (cache on)
  - **Verify**: `cargo build --release --features cuda` passes
  - **Depends**: None
  - **Complexity**: S

- [ ] **Task 1.2**: Define warp-local cache slot in shared memory
  - **Files**: `rust_prototype/gpu/cuda/transport.cu` (kernel `transport_persistent`)
  - **Pattern**: Allocate at kernel entry as `extern __shared__ double smem[]`; partition: `[block_dim/32]` warp slots × `(1 + 64)` doubles (sum + per-level)
  - **Acceptance**: `__shared__` declared, indexed by `threadIdx.x >> 5`, no out-of-bounds
  - **Verify**: NVRTC compile succeeds (`cargo build --release --features cuda`); ptx dump shows `.shared` allocation of expected size
  - **Depends**: 1.1
  - **Complexity**: M

- [ ] **Task 1.3**: Add cache-key state per warp slot (`int last_nuc`, `int last_eidx`, init to `-1`)
  - **Files**: `rust_prototype/gpu/cuda/transport.cu`
  - **Pattern**: One lane (lane 0 via `__shfl_sync`) writes the key; all lanes test
  - **Acceptance**: Init at kernel entry, never read before written, `__syncwarp()` after init
  - **Verify**: Same as 1.2; correctness check is a no-op since cache is unused yet
  - **Depends**: 1.2
  - **Complexity**: S

- [ ] **Task 1.4**: Bump `shared_mem_bytes` in persistent-kernel `LaunchConfig`
  - **Files**: `rust_prototype/src/gpu_transport.rs:1480-1483`
  - **Pattern**: Compute exactly `(BLOCK_SIZE / 32) * (1 + 64) * sizeof::<f64>()`
  - **Acceptance**: Launch succeeds; CUDA reports no resource error
  - **Verify**: `cargo run --release --features cuda --bin pwr_pincell -- $DATA --mode svd --batches 5 --inactive 1 --particles 1000` runs without launch failure
  - **Depends**: 1.2, 1.3
  - **Complexity**: S

**Phase 1 checkpoint:**
- [ ] `cargo build --release --features cuda` clean
- [ ] `cargo run --release --features cuda --bin pwr_pincell -- $DATA --mode all --batches 5 --inactive 1 --particles 1000` produces same k_inf as before (cache unused)
- [ ] No new clippy or fmt warnings

---

### Phase 2 — Cache write + read on the inelastic path

**Objective:** activate the cache. Pass 1 fills it, pass 2 reads it
when the key matches. Keep the fall-through path so a warp with
mixed `(nuc, e_idx)` still produces correct output.

**Prerequisite:** Phase 1 done.

- [ ] **Task 2.1**: In Pass 1 (sum), populate `smem[slot].lxs[l]` per level and accumulate into `smem[slot].sum`
  - **Files**: `rust_prototype/gpu/cuda/transport.cu:1297-1322`
  - **Pattern**: Follow existing pass-1 structure; add `smem[slot].lxs[l] = lxs_term; smem[slot].sum += lxs_term` after each `svd_reconstruct`
  - **Acceptance**: Cache is *populated by lane 0 only* via warp shuffle; other lanes see same value via `__syncwarp() + __shfl_sync`
  - **Verify**: `cargo run --release --features cuda --bin pwr_pincell -- $DATA --mode svd --batches 5 --inactive 1 --particles 1000`; k_inf identical to Phase 1
  - **Depends**: 1.4
  - **Complexity**: M

- [ ] **Task 2.2**: Replace Pass 2 (re-select) with cache read + on-the-fly running sum
  - **Files**: `rust_prototype/gpu/cuda/transport.cu:1327-1339`
  - **Pattern**: Read `smem[slot].lxs[l]` instead of calling `svd_reconstruct`; use `smem[slot].sum` for the rolling-pivot termination
  - **Acceptance**: Pass 2 calls `svd_reconstruct` zero times when key matches; identical Q-value selected as before for a fixed seed
  - **Verify**: Re-run PWR mini-run; k_inf identical to Phase 1 within fp determinism
  - **Depends**: 2.1
  - **Complexity**: M

- [ ] **Task 2.3**: Add cache-key broadcast — first lane in warp computes `(nuc_idx, e_idx)`, broadcasts via `__shfl_sync`, lanes whose key differs take fall-through path
  - **Files**: `rust_prototype/gpu/cuda/transport.cu`
  - **Pattern**: Use `__shfl_sync(0xffffffff, value, 0)` for warp-leader broadcast; matching mask `__ballot_sync` to gate cache use
  - **Acceptance**: Mixed-key warp still produces correct XS sum (verified against fall-through)
  - **Verify**: Targeted unit on a forced-mixed-energy population; PWR k_inf unchanged within fp determinism
  - **Depends**: 2.2
  - **Complexity**: L

- [ ] **Task 2.4**: Wire `P_WARP_CACHE_ENABLE` so the kernel can be A/B-toggled at runtime
  - **Files**: `rust_prototype/gpu/cuda/transport.cu`, `rust_prototype/src/gpu_transport.rs` CLI plumbing
  - **Pattern**: `if (cache_on) { ... } else { /* original code path */ }`
  - **Acceptance**: Setting param to `0` reproduces pre-cache behaviour byte-for-byte
  - **Verify**: PWR mini-run with cache on vs off, both vs current `main` — should be three identical k_inf within seed determinism
  - **Depends**: 2.3
  - **Complexity**: S

**Phase 2 checkpoint:**
- [ ] PWR pin cell SVD k_inf with cache ≡ k_inf without cache within combined SEM (10 seeds)
- [ ] AC3 holds: cache-on output matches CPU SVD reference where it
      already matched (no new bit-drift)
- [ ] Profiler (`nvprof` / `nsys`) shows reduction in
      `__nv_exp` / `__nv_log` invocations on the inelastic path

---

### Phase 3 — Benchmark binary

**Objective:** quantify the win on the same PWR pin cell config the
paper uses. Auto-runnable for the paper update.

**Prerequisite:** Phase 2 done and bit-parity proven.

- [ ] **Task 3.1**: New bin `gpu_warp_cache_bench.rs`
  - **Files**: `rust_prototype/src/bin/gpu_warp_cache_bench.rs` (new)
  - **Pattern**: Follow `rust_prototype/src/bin/gpu_pwr_bench.rs` for harness structure; PWR pin cell, 9 nuclides, 100 b × 20 k particles × 3 seeds (matches `outputs/full_test_run/10_pwr_all_rank5.txt` config)
  - **Acceptance**: Three rows printed: `gpu_pointwise`, `gpu_svd_no_cache`, `gpu_svd_cache`; each with k_inf ± SEM, ns/p, MB
  - **Verify**: `cargo run --release --features cuda --bin gpu_warp_cache_bench -- $DATA --rank 5`
  - **Depends**: 2.4
  - **Complexity**: M

- [ ] **Task 3.2**: Capture an output baseline at `outputs/gpu_warp_cache_bench.txt`
  - **Files**: `outputs/gpu_warp_cache_bench.txt` (new)
  - **Pattern**: Run the new bin with `--seeds 5` on the small-L3 box first; redirect stdout
  - **Acceptance**: Numbers committed; ns/p ratio recorded
  - **Verify**: File present, parseable by paper-update script
  - **Depends**: 3.1
  - **Complexity**: S

- [ ] **Task 3.3**: Add `--profile` flag that emits `nvprof`/`nsys`
      timing for the persistent kernel only
  - **Files**: `rust_prototype/src/bin/gpu_warp_cache_bench.rs`
  - **Pattern**: Use `cudaEventRecord` around the persistent-kernel
        launch (cudarc `Stream::record_event`)
  - **Acceptance**: Per-launch ms printed
  - **Verify**: Run with `--profile`, observe per-launch ms log
  - **Depends**: 3.1
  - **Complexity**: S

**Phase 3 checkpoint:**
- [ ] AC1 holds: cache-on ns/p within 5% of `gpu_pointwise`
- [ ] Profile shows persistent-kernel time reduced by the cache hit
      ratio expected from sort coherence

---

### Phase 4 — Validation

**Objective:** prove physics is unchanged across the full PWR
ten-seed sweep, including off-library temps.

**Prerequisite:** Phase 3 done.

- [ ] **Task 4.1**: Bit-parity sweep — cache on vs cache off on
      `--force-svd`, 10 seeds, 5 b × 1 k particles (small)
  - **Files**: `rust_prototype/src/bin/pwr_pincell.rs` (existing
        `--force-svd` flag) plus `--no-warp-cache` from 2.4
  - **Pattern**: Follow `gpu_compton_validate.rs:*` per-seed loop
  - **Acceptance**: Per-seed k_inf identical to fp determinism
  - **Verify**: `for s in 1..=10; do diff <(... --no-warp-cache --seed $s) <(... --seed $s); done`
  - **Depends**: 3.1
  - **Complexity**: M

- [ ] **Task 4.2**: Full ten-seed PWR run — cache on vs cache off,
      paper-grade statistics (`120 b × 50 k × 10 seeds`)
  - **Files**: command in `outputs/gpu_warp_cache_bench.txt`
  - **Pattern**: `cargo run --release --features cuda --bin pwr_pincell -- $DATA --mode svd --batches 120 --inactive 30 --particles 50000 --seeds 10`, twice
  - **Acceptance**: $|\Delta k_\infty| \leq 3 \cdot \mathrm{SEM}_{\text{combined}}$ (AC2)
  - **Verify**: Output diff; SEM-combined check arithmetic
  - **Depends**: 4.1
  - **Complexity**: M

- [ ] **Task 4.3**: Off-library +150 K sweep
  - **Files**: same bin, `--target-temp-offset 150`
  - **Pattern**: `scripts/pwr_verdict.py` already drives this — re-run with cache on
  - **Acceptance**: Verdict still GREEN (exit 0)
  - **Verify**: `python scripts/pwr_verdict.py --offset 150 --seeds 10 --particles 50000 --batches 120 --inactive 30`
  - **Depends**: 4.2
  - **Complexity**: S

**Phase 4 checkpoint:**
- [ ] Both AC1 (within 5% of GPU pointwise) and AC2 (k_inf within
      SEM) confirmed on paper-grade statistics
- [ ] Off-library verdict still GREEN

---

### Phase 5 — Hygiene

- [ ] **Task 5.1**: `cargo test --lib` and `--features cuda` test pass
  - **Verify**: `cargo test --lib && cargo test --lib --features cuda`
  - **Depends**: 4.3
  - **Complexity**: S

- [ ] **Task 5.2**: CPU-only build still clean
  - **Verify**: `cargo build --release` (no `--features`)
  - **Depends**: 4.3
  - **Complexity**: S

- [ ] **Task 5.3**: clippy + fmt
  - **Verify**: `cargo clippy --all-targets --features cuda -- -D warnings && cargo fmt --check`
  - **Depends**: 4.3
  - **Complexity**: S

---

### Phase 6 — Paper update

- [ ] **Task 6.1**: Update `paper/sections/gpu.tex` and
      `paper/sections/threats.tex` to reflect the cache landed and the
      measured GPU SVD vs GPU pointwise ratio
  - **Files**: `paper/sections/gpu.tex:10-23`, `paper/sections/threats.tex:54-57`,
        `paper/sections/conclusion.tex:67-74`
  - **Pattern**: Replace "would close the gap... not implemented" with the measured ratio from `outputs/gpu_warp_cache_bench.txt`
  - **Acceptance**: Numbers cited in three places agree; bibliography untouched
  - **Verify**: `cd paper && latexmk -pdf main.tex`
  - **Depends**: 4.3
  - **Complexity**: S

**Phase 6 checkpoint:**
- [ ] LaTeX builds clean
- [ ] Paper §gpu, §threats, §conclusion all agree on the new ratio

---

## 6. Test strategy

| Layer       | What | Where |
|-------------|------|-------|
| Unit        | `Params` round-trip with new `P_WARP_CACHE_ENABLE` | `rust_prototype/src/gpu_transport.rs` (existing param tests) |
| Integration | Cache on vs off vs CPU SVD bit-parity              | `gpu_warp_cache_bench --bit-parity` |
| Physics     | k_inf ten-seed PWR on-library and +150 K off-library | `pwr_pincell --seeds 10` and `pwr_verdict.py` |
| Throughput  | ns/p comparison vs GPU pointwise                   | `gpu_warp_cache_bench` |
| Regression  | Full library test suite                            | `cargo test --lib --features cuda` |

---

## 7. Open questions

1. **Cache slot size**: 64 doubles per warp is comfortably under the
   48 KB shared-memory budget on Ampere, but on smaller SMs (RTX
   A1000, 4 GB, 1.5 MB L2) we may want to fall back to a 32-level
   cap. The current code uses `lev_cap = min(n_lev, 64)`. Confirm at
   bench time: does any nuclide actually hit the 64-cap on the PWR
   set? (Answer expected: no — 13 levels for Zr, ≤ 41 for U.)
2. **Hit ratio baseline**: do we want a CUDA hardware counter on
   cache-hit vs cache-miss, or is profiling-derived ns/p enough? The
   plan above assumes ns/p is sufficient. If the reviewers want it,
   add a `__device__` `unsigned long long` counter.
3. **Reservation for multi-nuclide warps**: if a single warp ever
   has lanes with different `hit_nuc`, the cache key broadcast in
   2.3 forces those lanes to the fall-through path. This is
   correct but possibly wasteful. Quantify the multi-nuclide
   fraction with the profiler; if non-trivial, a follow-up could add
   a small associative cache (e.g. 4 slots per warp).

---

## 8. References

- Source paper: `paper/main.pdf` and `paper/sections/{gpu,threats,conclusion,pwr}.tex`
- README: `README.md` (esp. "GPU SVD beats GPU pointwise" disclaimer)
- Resume: `resume.md` (GPU benchmarking environment, persistent-kernel design)
- Hot path source: `rust_prototype/gpu/cuda/transport.cu:1297-1340`
- Energy sort: `rust_prototype/src/gpu_transport.rs:1432-1475`
- Existing GPU bench harness pattern: `rust_prototype/src/bin/gpu_pwr_bench.rs`
- Existing photon shared-mem patterns: `rust_prototype/src/photon/gpu.rs` (per README)
- Cache feasibility motivation: `scripts/cache_feasibility_analysis.py`
- ICSBEP HMF-001 baseline: `outputs/full_test_run/10_pwr_all_rank5.txt`
- Tramm et al. event-based GPU MC reference: cited in `paper/sections/intro.tex`
