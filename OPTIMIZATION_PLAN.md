# Optimization Plan — Based on Honesty Test Results

## The Problem

The 1M-particle honesty test revealed three surprises:

| Metric | OpenMC | Rust Table | Rust SVD (k=5) |
|--------|--------|------------|-----------------|
| k_eff | 0.99844 | 0.99909 | 1.00022 |
| Sim time | 141 s | 177 s | 256 s |
| XS memory | 817 MB | 216 MB | 361 MB |

**SVD is 1.45x slower and 1.67x larger than the table it's supposed to replace.**

Root causes:
1. **Cache pressure**: 361 MB SVD data >> L3 cache (16-32 MB)
2. **Energy grid duplication**: 111 MB wasted on duplicate grids (29%)
3. **Expensive transcendentals**: `10^x` (powf) called per lookup
4. **Redundant binary search**: 6 binary searches per nuclide per collision
5. **Per-collision heap allocation**: `Vec<MicroXs>` allocated every collision

---

## Phase 1 — Low-Hanging Fruit (expected: 2-3x faster, 65% less memory)

### 1.1 Share energy grids across reactions (free 111 MB)

Currently every `SvdKernel` and `PointwiseTable` stores its own copy of
the unionized energy grid. For U-238 with 46 reactions, that's 46 copies
of 186K points.

**Change**: Store energy grids in `Arc<Vec<f64>>` shared per nuclide.
Each kernel/table holds an `Arc` reference instead of an owned `Vec`.

- Memory saved: **111 MB (29% of total SVD memory)**
- Impact on speed: slightly better cache utilization
- Complexity: low — change `Vec<f64>` to `Arc<[f64]>` in `SvdKernel`/`PointwiseTable`

### 1.2 f32 basis vectors (halve basis memory)

The SVD basis stores pre-multiplied `U*S` values. At rank=5, the basis
is 5 f64 values per energy point per reaction = 265 MB total. The SVD
truncation error at rank=5 is already ~1e-3 to 1e-6, so f32 precision
(~7 decimal digits) loses nothing meaningful.

**Change**: Store `basis: Vec<f32>`, convert to f64 only for the dot product.

- Memory saved: **133 MB (35% of total)**
- Combined with grid sharing: **361 MB → 135 MB (2.7x smaller)**
- Complexity: low-medium — new `SvdKernel` variant or generic over float type

### 1.3 Eliminate per-collision Vec allocation

`transport_particle()` allocates `Vec<MicroXs>` and `Vec<f64>` at every
collision. With 1M particles × ~20 collisions each = 20M mallocs.

**Change**: Pre-allocate once per particle lifetime and reuse.
For Godiva (3 nuclides), use a fixed-size array `[MicroXs; 8]`.

```rust
let mut micro_xs_buf = [MicroXs::default(); 8];
// In collision loop:
for (i, nuc) in material.nuclides.iter().enumerate() {
    micro_xs_buf[i] = xs_provider.lookup(nuc.xs_kernel_idx, particle.energy);
}
```

- Speedup: ~10-15%
- Complexity: low

### 1.4 Replace `10^x` with `exp2(x * LOG2_10)`

The `10.0_f64.powf(log_val)` call in every SVD lookup and table lookup
is a general-purpose powf (~25-40 cycles). Since the base is always 10,
use the identity: `10^x = 2^(x * log2(10))`.

`exp2()` has hardware support on x86 and is 3-5x faster than `powf()`.

**Change**: Replace `10.0_f64.powf(log_val)` with `f64::exp2(log_val * std::f64::consts::LOG2_10)`.

- Speedup: ~10-15% (both SVD and table paths)
- Complexity: trivial — one-line change

### 1.5 Single binary search per nuclide per collision

Currently: 6 separate binary searches (elastic, inelastic, n2n, n3n,
fission, capture) on the **same** energy grid per nuclide. Each lookup
in `ReactionKernel::lookup()` does its own binary search.

**Change**: Factor out the binary search into `NuclideKernels::lookup_all()`:
```rust
fn lookup_all(&self, energy: f64) -> MicroXs {
    let idx = binary_search(&self.shared_energy_grid, energy);
    // Reuse idx for all 6 reactions
    let elastic = self.elastic.reconstruct_at(idx);
    let fission = self.fission.reconstruct_at(idx);
    // ...
}
```

- Speedup: ~15-20% (saves 5 of 6 binary searches per nuclide)
- Complexity: medium — requires shared grid + refactored lookup
- Prerequisite: 1.1 (shared grids)

**Phase 1 combined estimate**:
- Memory: 361 MB → **~135 MB** (2.7x reduction, now smaller than table)
- Speed: ~50-60% faster (256s → ~105-130s)

---

## Phase 2 — Architectural (expected: another 2-3x)

### 2.1 Event-based transport with energy sorting

**Problem**: History-based transport processes one particle to completion.
At 1M particles, each collision accesses XS data at a random energy →
cache thrashing across the 135 MB (post-Phase 1) basis.

**Solution**: Switch to event-based transport:
1. Advance all particles to their next collision point
2. Sort particles by energy
3. Batch XS lookups (sequential energy access → cache-friendly)
4. Process collisions
5. Repeat

**References**:
- Tramm et al., "Performance Analysis of Monte Carlo Neutron Transport
  Codes" (ANS 2018) — measured 2-4x speedup from event-based on GPU
- Siegel et al., "Analysis of Performance Bottlenecks in MC Transport"
  (2014) — identifies memory bandwidth as primary bottleneck
- OpenMC's event-based mode (merged 2020)

- Speedup: 2-4x for memory-bound workloads
- Complexity: high — major refactor of `simulate.rs`

### 2.2 Hash-based energy grid lookup (O(1) instead of O(log N))

**Problem**: Binary search on 186K-point grid = 17 comparisons × cache
misses. Even after the Phase 1 "single search" optimization, this is
~17 random memory accesses per nuclide per collision.

**Solution**: Precompute a hash table that maps energy → grid index.

**References**:
- Brown, "New Hash-Based Energy Lookup Algorithm for Monte Carlo Codes"
  (LANL, LA-UR-14-24530, 2014)
- Walsh et al., "Optimizations of the Energy Grid Search Algorithm"
  (Ann. Nucl. Energy, 2015)
- OpenMC uses a "nuclide grid method" with per-nuclide direct lookup

Implementation: divide the log-energy range into N uniform bins. For each
bin, store the grid index. Lookup: `idx = (ln(E) - ln(E_min)) / delta`
then linear scan ≤2 entries.

- Speedup: ~20-30% (eliminates binary search cache misses)
- Complexity: medium — additional data structure at load time

### 2.3 Majorant cross-section / delta tracking

**Problem**: Every collision requires looking up all 6 partial XS for all
nuclides. Most of this data is only needed to determine IF a real
collision occurs and WHICH nuclide/reaction is sampled.

**Solution**: Pre-compute a majorant (maximum) total macroscopic XS over
all energies. Use it to sample collision distances. At each potential
collision point, look up the true total XS and accept/reject:
- Accept probability = Σ_t(E) / Σ_maj
- If rejected: "virtual collision", particle continues unchanged

For Godiva (single homogeneous material), the majorant is just the
maximum of Σ_t(E). This avoids full multi-reaction lookups for the
~30-50% of virtual collisions.

Only when a real collision occurs do we need the partial XS breakdown.

- Speedup: ~30-50% fewer full XS lookups
- Complexity: medium — need pre-computed majorant table
- Caveat: less effective for heterogeneous geometries

### 2.4 Adaptive windowed rank

**Problem**: rank=5 everywhere wastes memory in smooth energy regions.

**Solution**: Split the energy range into windows:
- Thermal (< 1 eV): rank=1 (smooth)
- Resonance (1 eV – 100 keV): rank=3-5 (structured)
- Fast (> 100 keV): rank=1 (smooth)

U-235 has 47/52 reactions that are rank-1. Even fission only needs rank=3
for <10 pcm error.

**References**:
- Our own phase5_3_windowed_svd.py analysis
- Singular spectrum analysis shows 5 orders of magnitude decay

- Memory: further 30-50% reduction in basis size
- Complexity: medium — window boundary handling in lookup
- k_eff impact: <10 pcm if resonance region keeps rank≥3

**Phase 2 combined estimate**:
- Speed: another 2-3x → total ~4-8x faster than current
- Target: **~30-60s** for 150M histories (competitive with OpenMC's 141s)

---

## Phase 3 — GPU (expected: 5-20x on top)

### 3.1 CUDA/wgpu event-based transport

The SVD reconstruction is ideal for GPU execution:
- Sequential basis streaming (coalesced memory access)
- Fixed-rank dot product (no branch divergence)
- Event-based batching provides natural parallelism

After Phase 2 (event-based + energy sorting), the transport loop maps
cleanly to GPU warps:
- Sort 1M particles by energy → adjacent threads access adjacent basis rows
- Rank-5 dot product → 5 FMAs per thread, fully utilized

**References**:
- Our cuda_bench/ shows 2.6-2.8x on laptop RTX A1000
- Expected 10-20x on RTX 3080/A100 with event-based

- Speedup: 5-20x over CPU (post-Phase 2)
- Complexity: very high — full GPU transport implementation

---

## Priority Implementation Order

| # | Optimization | Speedup | Memory | Complexity | Prereqs |
|---|-------------|---------|--------|------------|---------|
| 1 | `exp2` instead of `powf` | +10-15% | — | Trivial | — |
| 2 | Eliminate Vec alloc | +10-15% | — | Low | — |
| 3 | Share energy grids | — | -29% | Low | — |
| 4 | f32 basis | — | -35% | Low-Med | — |
| 5 | Single binary search | +15-20% | — | Medium | #3 |
| 6 | Hash-based grid lookup | +20-30% | — | Medium | #3 |
| 7 | Delta tracking | +30-50% | — | Medium | — |
| 8 | Event-based transport | +100-200% | — | High | #6 |
| 9 | Adaptive windowed rank | — | -30-50% | Medium | — |
| 10| GPU transport | +500-2000% | — | Very High | #8 |

Items 1-4 can be done in a day. Items 5-7 in a week. Item 8 is a
multi-week refactor. Item 10 is a separate project.

---

## Success Criteria

After Phase 1 (items 1-5):
- [ ] SVD memory < table memory (135 MB < 216 MB)
- [ ] SVD sim time < table sim time
- [ ] k_eff unchanged (< 5 pcm impact)

After Phase 2 (items 6-9):
- [ ] Total throughput > 2M histories/s (competitive with OpenMC)
- [ ] Memory < 100 MB for 3-nuclide Godiva
- [ ] k_eff < 30 pcm from experiment
