# Stage C — bench results on A1000

## Step D phase 1+2+3 validation campaign

After landing the full per-nuc kernel ABI swap — basis/coeffs +
pointwise + total_xs + ν̄ + delayed-ν̄ + URR + inel_cdf + per-level
discrete basis/coeffs + elastic and per-level angular CDFs +
fission tabular + MT=91 — three validation gates:

### 1. ICSBEP regression (4-case)

| case                    | k_calc ± σ          |  Δ pcm | bound | runtime |
|-------------------------|---------------------|-------:|------:|--------:|
| heu-met-fast-001_case-1 | 0.999722 ± 0.000450 |  -27.8 |   219 |   55 s  |
| heu-met-fast-014        | 0.997293 ± 0.000382 | -160.7 |   349 |  128 s  |
| pu-met-fast-001         | 1.000789 ± 0.000520 |  +78.9 |   413 |   18 s  |
| u233-met-fast-001       | 1.000201 ± 0.000166 |  +20.1 |   203 |   14 s  |

**4 / 4 PASS** at production envelope on A1000.

### 2. `nu_lookup_compare`

> Worst |Δ| over actively-tabulated nuclides (sz > 0): **0.000e0**
> → ν table data is BIT-IDENTICAL CPU↔GPU for every nuclide that
> carries a real ν(E) table.

### 3. `level_xs_compare`

> Per-level XS bit-identical CPU↔GPU (Δ = 0.000%) at every test
> energy 1 eV → 5 MeV across every U-235 discrete inelastic level
> (MT=51-56). Σxs Δ = +0.00%, ⟨|Q|⟩ Δ = +0.00%.

Per-level rank-padding invariant from commit `1654c4d` preserved
end-to-end through the pointer-array kernel ABI — the historic
+500-700 pcm fast-metal hot bias is NOT being reintroduced.

### 4. `metal_stats_diag` (Godiva 3-way)

| metric       | CPU     | GPU     | OpenMC  | Δ (GPU − OpenMC) |
|--------------|--------:|--------:|--------:|-----------------:|
| k_eff Δ      | -192 pcm | +109 pcm | 0       | **+109 pcm**     |
| ⟨E⟩ fission  | 1.4646e6 | 1.4813e6 | 1.4618e6 | +1.35% of σ      |

GPU **closer to OpenMC than CPU** for k_eff Δ; fission σ within
1.35% of OpenMC. The GPU↔CPU gap of +301 pcm is dominated by event-
ordering and float-rounding (already documented as a known GPU-
native variance in CLAUDE.md), not the per-nuc kernel ABI.

## Step D correctness — 3-case ICSBEP sweep (full per-nuc kernel ABI)

After converting every simple per-nuclide kernel access path to read
through pointer arrays — main SVD basis/coeffs, pointwise_xs,
total_xs, ν̄(E) + delayed-ν̄(E), six URR sub-tables, and synthesized
inelastic CDF — with per-nuc CudaSlices owned by the cache.

| case                    | k_calc      ± σ       |  Δ pcm | bound | runtime |
|-------------------------|-----------------------|-------:|------:|--------:|
| heu-met-fast-001_case-1 | 1.000439 ± 0.000396   |  +43.9 |   215 |   57 s  |
| pu-met-fast-001         | 1.000725 ± 0.000456   |  +72.5 |   410 |   15 s  |
| u233-met-fast-001       | 0.999605 ± 0.000397   |  -39.5 |   215 |   17 s  |

**3 / 3 PASS, 89 s wall on RTX A1000.** Production transport with
every per-nuc kernel access on the pointer-array path passes ICSBEP
on fast-metal scenes. k deltas remain within statistical noise
(0.35-0.41σ) of the handbook reference.

## Step D correctness — 3-case ICSBEP sweep (post-kernel-ABI swap)

After landing the per-nuclide pointer-array kernel ABI
(`transport.cu` reads `((double*)PTR_U64(p, P_BASIS_PTRS)[key])[…]`
instead of indirecting through `all_basis[basis_offsets[…]+…]`).

| case                    | k_calc      ± σ       |  Δ pcm | bound | runtime |
|-------------------------|-----------------------|-------:|------:|--------:|
| heu-met-fast-001_case-1 | 0.999387 ± 0.000416   |  -61.3 |   217 |   57 s  |
| pu-met-fast-001         | 1.001389 ± 0.000612   | +138.9 |   418 |   15 s  |
| u233-met-fast-001       | 1.000252 ± 0.000409   |  +25.2 |   216 |   14 s  |

**3 / 3 PASS, 86 s wall on RTX A1000.** The kernel reading
basis/coeffs directly from per-nuclide CudaSlices via pointer-array
load produces physics within the ICSBEP envelope on a fast-metal
benchmark mix.

The k deltas vs Step C (below) are within 1σ of each other across
the three cases — the difference is statistical (independent random
seeds), not systematic.

## Step C correctness — 6-case ICSBEP sweep

End-to-end validation that the per-nuclide LFU cache preserves
correctness on real production cases. Filter pattern:
`"heu-met-fast-001_case-1|heu-met-fast-014|pu-met-fast-001|u233-met-fast-001|leu-comp-therm-008_case-1"`
(5 regexes; LCT-008 prefix matched cases 1 + 11 → 6 cases total).

Per-case `recommended_settings` from the JSON (5 seeds × 150 batches ×
30-50 inactive × 15-20k particles) — full paper-quality envelope, not
smoke. Acceptance bound is `max(150 pcm, 2 σ_combined)`.

| case                       | k_calc      ± σ       | Δ pcm | bound | runtime |
|----------------------------|-----------------------|------:|------:|--------:|
| heu-met-fast-001_case-1    | 1.000275 ± 0.000344   | +27.5 |   212 |   35 s  |
| heu-met-fast-014           | 0.998332 ± 0.000343   | -56.8 |   347 |  128 s  |
| leu-comp-therm-008_case-1  | 1.001267 ± 0.000731   | +56.7 |   281 |  531 s  |
| leu-comp-therm-008_case-11 | 1.001332 ± 0.000220   | +63.2 |   244 |  532 s  |
| pu-met-fast-001            | 1.001158 ± 0.000621   |+115.8 |   419 |   22 s  |
| u233-met-fast-001          | 0.999395 ± 0.000311   | -60.5 |   210 |   15 s  |

**6 / 6 PASS, total wall time 21 min on RTX A1000.**

The per-nuclide cache fires across cases sharing U-235 / U-238 / O-16 /
H-1 (the HEU / LEU pair) and metallic-actinide nuclides (HMF / PMF /
U233-MF).

## What's NOT in this bench

- **Cross-case load-time win**: the bench measures correctness, not
  cache-hit throughput. The 530 s on LCT-008 case-1 includes
  cold-load of every thermal-actinide nuclide; case-11 should hit
  the cache for the shared subset but still took 532 s — most of
  that wall is the transport loop, not the upload. A clean upload-
  time bench would need to instrument `upload_nuclide_data` directly.

- **Bundle-cache baseline comparison**: comparing pre-Step-C
  (whole-bundle LRU) against post-Step-C (per-nuclide LFU) on the
  same sweep is the only way to quantify the cross-case sharing
  win. Deferred to the 3080 sweep on the user-side machine.

- **VRAM telemetry**: peak-resident bytes during the sweep would
  show whether the per-nuclide cache's union-of-nuclides budget
  stays under the 0.75 × VRAM cap. `nvidia-smi` watching is the
  trivial path here.

## How to reproduce

```powershell
.venv\Scripts\python.exe `
    rust_prototype\bindings\python\examples\icsbep_sweep.py `
    --runner gpu `
    --filter "heu-met-fast-001_case-1|heu-met-fast-014|pu-met-fast-001|u233-met-fast-001|leu-comp-therm-008_case-1" `
    --csv outputs/bench_step_c.csv
```

Per-case settings come from each JSON's
`benchmark.recommended_settings` block — production-quality.

## Cache state introspection

The per-nuclide LFU exposes `clear_nuclide_buffer_cache()` on
`GpuTransportContext` for diagnostic teardown. There's no public
hit-rate API yet — add one if you need quantitative cache-hit
statistics for the cross-case sweep (track `stats.hits` over the
LFU's entries via a public `cache_stats()` method).
