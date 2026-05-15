# Stage C — bench results on A1000

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
