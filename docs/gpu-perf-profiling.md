# GPU performance profiling — transport_recursive_persistent

**Date:** 2026-05-18  
**Tool:** Nsight Systems 2025.1.3 + Nsight Compute 2025.2.0  
**Device:** RTX A1000 laptop (sm_86, 4 GB, 1.5 MB L2)  
**Kernel:** `transport_recursive_persistent` in `gpu/cuda/transport.cu`  

---

## 1. Time budget (nsys — ICSBEP cases)

| Case | Batches | Kernel time | Memcpy DtoH† | Memcpy HtoD | Kernel % |
|------|---------|-------------|-------------|-------------|---------|
| Godiva HMF-001 (3 nuc) | 80 | 933 ms | 5.5 ms | 146 ms | **100%** |
| PWR-17×17 (9 nuc + SAB) | 80 | 11 557 ms | 5.6 ms | 146 ms | **98%** |

†`cuMemcpyDtoHAsync` API time is inflated by stream synchronisation on the kernel;
actual data transferred is only ~3.5 µs/call.  

**The kernel is the bottleneck.** Memcpy and host overhead are negligible.  
Per-batch time scales ~12× from Godiva → PWR assembly, proportional to geometry
complexity (bare sphere vs. depth-3 recursive 17×17 lattice).

---

## 2. Cache sweep (ncu — 3 cases profiled, 2 OOM)

3 kernel launches each, steady-state. Metrics collected with
`ncu --kernel-name transport_recursive_persistent --metrics ...`.

| Case | n\_nuc | L1 hit | L2 hit | DRAM % | L1 bw % | L2 bw % | SM % | Warps active |
|------|--------|--------|--------|--------|---------|---------|------|-------------|
| Godiva (fast, bare) | 3 | 61.9 | 90.7 | 2.1 | 3.6 | 3.1 | 62.8 | 17.7 |
| PMF-001 (Pu+Ga fast) | 5 | 61.4 | 88.3 | 1.7 | 2.8 | 2.5 | 57.7 | 17.0 |
| PWR-17×17 (SAB+geom) | 9 | 51.0 | 87.1 | 4.3 | 6.9 | 5.5 | 63.2 | 18.8 |
| MMF-007 (c\_Be, 6 nuc) | 6 | — | — | — | — | — | — | — |
| HST-001 (38 nuc, sol) | 38 | — | — | — | — | — | — | — |

MMF-007 and HST-001 caused ncu OOM (`driver resource unavailable`) — their extra
GPU allocations (Be coherent-elastic + 38-nuclide XS buffers) left insufficient
VRAM for ncu's performance counter buffers.

**Observations:**
- L2 hit rate stable at 87–91% across all tested nuclide counts. XS lookup data
  fits in L2 regardless of nuclide count (at least up to 9 nuclides with SAB).
- L1 drops 62% → 51% for PWR vs. fast-metal. Depth-3 geometry traversal puts
  extra pressure on L1 (geometry buffers compete with XS data).
- DRAM doubles for PWR (4.3% vs 2%) but remains very low in absolute terms.
  S(α,β) table + deep geometry walk causes the extra DRAM traffic.
- **SM throughput ~58–63% and active warps ~17–19% are nearly identical across
  cases.** Cache and nuclide count are not the bottleneck — the warp-internal
  divergence is.

---

## 3. Warp-level analysis (ncu WarpStateStats — PWR-17×17)

```
Warp Cycles Per Issued Instruction   42.0 cycles
Avg. Active Threads Per Warp          6.2 / 32   (81% idle per warp)
Avg. Not-Predicated-Off Threads       5.7 / 32

Est. Local Speedup (MIO stall)       75.3%
Est. Local Speedup (divergence)      82.3%
```

### Primary bottleneck — MIO short-scoreboard stall (75.3% of cycles)

31.6 of 42 cycles per instruction are spent waiting on MIO operations.
ncu diagnosis: "frequent execution of special math instructions (e.g. MUFU)
or dynamic branching (e.g. BRX, JMX)".

The MUFU/MIO cost comes from `svd_reconstruct` in `transport.cu`:

```c
// rank-5 FMA chain + exp2 per level (lines 324–356):
double val = 0.0;
for (int j = 0; j < rank; j++)
    val += basis[e_idx*rank + j] * coeffs[j];
return exp2(val);   // ← MUFU instruction, ~20 cycle latency, MIO scoreboard
```

`exp2` is a MUFU instruction on Ampere. It goes through the MIO pipeline (not
the main FP64 SIMD pipe), has ~20 cycle latency, and creates a short-scoreboard
stall on every SVD reconstruction. Each inelastic collision calls
`svd_reconstruct` up to 2×41 times (legacy two-pass path) or 1× (CDF fast
path). Elastic collisions call it once per reaction per nuclide per collision.

Additional MIO pressure from the reaction-type dispatch chains (`do_inelastic`,
`do_elastic`, `do_sab`, `do_fission` — all BRX/JMX dynamic branches in the
compiled PTX).

### Secondary bottleneck — warp divergence (82.3% potential speedup)

Only **6.2 / 32 threads active per warp**. Each warp's 32 threads are mid-flight
through their independent histories. At any given collision step they are at
different reaction types (elastic / inelastic / fission / capture / SAB), so each
thread follows a different branch. History-based GPU MC always diverges here;
the symptom is the 6.2 active-thread measurement.

**Compute Workload Analysis:**

```
SM Busy                 79.7%    (SM has warps scheduled)
Issue Slots Busy         5.2%    (SM actually issuing instructions)
Executed IPC Active      0.21    (very low; peak for sm_86 FP64 is ~2.0)
```

SM is busy (warps exist) but issuing only 5% of the time.
The gap between SM Busy and Issue Slots Busy is the MIO stall.

---

## 4. Bottleneck summary

| Bottleneck | Impact | Root cause |
|-----------|--------|-----------|
| MIO stall (MUFU `exp2`) | **75% of cycles** | `svd_reconstruct` exp2 on every XS lookup |
| Warp divergence | **82% potential speedup** | History-based MC: 6.2/32 threads same path |
| L1 pressure (geometry) | Minor (L1 62→51%) | Depth-3 recursive lattice traversal |
| DRAM bandwidth | Negligible (2–4%) | Data fits in L2; not a bottleneck |
| Host↔device memcpy | Negligible | Already async; <1% of wall time |

The two bottlenecks are partially independent: fixing warp divergence (event-based
MC) does not reduce the per-call MUFU cost; fixing MUFU (replace exp2 or switch
to f32) does not reduce divergence. Both are needed to close the gap to theoretical
GPU throughput.

---

## 5. Optimization candidates

### O1 — ~~Replace `exp2` with fast-math intrinsic~~ **TESTED, NO GAIN — closed**

**Tested 2026-05-18.** Three approaches tried:

1. `use_fast_math: Some(true)` in NVRTC → **PWR assembly: +1132 pcm FAIL** (f64
   divisions / reciprocals throughout URR/SAB/inelastic paths degraded).
2. Surgical `(double)exp2f((float)val)` in `svd_reconstruct` / `svd_reconstruct_interp` →
   **6/6 ICSBEP PASS** (precision OK: ~1 ppm per XS eval, negligible in k_eff).
   **Speedup: +0.35%** (within noise). `gpu_assembly_keff` 3-seed: 30290 vs 30397 ns/p.
3. Reverted. exp2 latency is hidden by 176 concurrent active warps; it is not the
   rate-limiting step.

**Root cause correction:** the ncu MIO short-scoreboard stall is dominated by
**BRX/JMX dynamic branch instructions** (the reaction-type dispatch chains:
`do_inelastic`, `do_elastic`, `do_fission`, `do_sab`) — not MUFU transcendentals.
These branch targets feed into the MIO scoreboard the same way MUFU does.
Reducing exp2 cost does not reduce branch stalls.

### O2 — Increase particle count per batch *(trivial, immediate)*

Active warps = 17–19% of peak (target ≥ 50% for good hiding of stalls). The RTX
A1000 has 16 SMs × 64 warps = 1024 warps. At 5000 particles/batch with one
warp per particle we're at 5000/1024 ≈ 5 warps/SM. Need ~50k particles/batch
to saturate. Larger batches hide the MIO latency behind more in-flight warps.

**Expected gain:** at 50k particles/batch, stall hiding improves; effective IPC
rises even if per-instruction latency is unchanged.

### O3 — Event-based transport *(high effort, high reward)*

Sort particles by collision type before processing. Threads in a warp would then
all execute the same branch (all elastic, all inelastic, etc.), pushing active
threads/warp from 6.2 → 32. This is the Tramm 2024 approach flagged in
`STATUS.md`. The 82.3% divergence potential speedup is the theoretical ceiling.

**Expected gain:** ~5× on the divergence-bound portion; combined with O1 could
approach the paper's CPU–GPU parity target.

### O4 — Persist XS data in registers / L1 for hot nuclides  *(medium effort)*

For Godiva (3 nuclides) L1 is 62% — there's room to improve. The basis/coeffs
arrays are read per collision via `__ldg` (read-only cache). For cases where all
nuclides fit, binding them to `__constant__` memory (64 KB on Ampere) would
give L1-equivalent latency with broadcast semantics across the warp. For Godiva's
3 nuclides (rank-5 × ~83k energy pts) this exceeds constant memory; but the
coeffs (rank-5 per nuclide × n_reactions) might fit.

### O5 — Pre-tabulate SVD reconstruction as pointwise table at XS load *(medium effort)*

For nuclides where we already upload pointwise XS, the SVD lookup is bypassed.
The CDF path for inelastic (commit `5e9e5e8`) already demonstrates this for the
level-fraction CDF. A similar pre-tabulation of the full reaction XS at the
loaded temperature would remove all `svd_reconstruct` calls from the GPU hot
path entirely, at the cost of memory (same as the existing pointwise tables).
This is only beneficial where pointwise tables aren't already uploaded.

---

## 6. Next steps

Priority order given the measurements:

1. **O2** (particles/batch) — measure immediately; free gain, just change the default.
2. **O1 closed** — exp2f gives no gain; BRX branching is the actual MIO cause.
3. **Profile BRX stalls** — run ncu `SourceCounters` section to find the top stall
   PCs and confirm which branches dominate. May surface opportunities to flatten
   the dispatch (branch-free predicated code for common reactions).
4. **O3** (event-based) — plan separately; multi-week effort; the only lever that
   directly fixes the BRX divergence.

Revised understanding of the bottleneck:

```
MIO stall 75% of cycles
├── BRX/JMX dynamic branches  ← dominant (reaction-type dispatch chains)
└── MUFU exp2(double)          ← secondary, already latency-hidden
```

Fixing the BRX dominance requires either:
- Predicated execution to eliminate actual branches (hard for complex MC dispatch)
- Event-based sort to make warps homogeneous (all threads take the same branch)

Artifacts:
- `outputs/nsight/icsbep_godiva_admin.nsys-rep`
- `outputs/nsight/icsbep_pwr_assembly.nsys-rep`
- `outputs/nsight/ncu_assembly.ncu-rep` (WarpStateStats + Compute)
- `outputs/nsight/cache_sweep.csv`
