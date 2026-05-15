# Stage C completion summary

Full per-nuclide GPU XS architecture landed across two PRs:

| PR | Branch | Scope |
|----|--------|-------|
| #4 (merged) | `feat/per-nuclide-gpu-cache` | Stage A (per-nuclide extraction) + Stage B (`assemble_*_cat` DtoD bundle assembly) |
| pending     | `feat/stage-c-finish` | Step C (per-nuclide LFU cache) + Step D (full kernel ABI swap to per-nuclide pointer arrays) + validation campaign |

## What's running in production (post `feat/stage-c-finish`)

### Step C — per-nuclide LFU cache
Bundle-level LRU replaced with per-nuclide LFU keyed on
`(Arc::as_ptr(nuc), rank)`. Cross-case sharing fires through
upstream Arc-dedup in `nuclide_cache::TieredStore::L1MemoryStore`.
`OPEN_RUST_MC_GPU_BUNDLE_CACHE_BYTES` / `_FRACTION` still control
budget; semantics shift from "one bundle's bytes" to "union of
cached per-nuclide bytes."

### Step D — per-nuclide kernel ABI
Every kernel access on the per-nuc hot path reads through pointer
arrays instead of indirecting through the concatenated bundle
slabs. Per-nuc base pointers + un-shifted within-nuc offset arrays
replace the legacy global-shifted offset arrays.

Categories converted:
1. Per-(nuc × reaction) SVD basis/coeffs — `P_BASIS_PTRS` / `P_COEFFS_PTRS`
2. Pointwise XS — `P_PW_XS_PTRS`
3. Total XS — `P_TOTAL_XS_PTRS`
4. ν̄(E) + delayed-ν̄(E) — `P_NB_E_PTRS`, `P_NB_V_PTRS`, `P_DNB_E_PTRS`, `P_DNB_V_PTRS`
5. URR (6 sub-tables) — `P_URR_E_PTRS` + 5 factor table pointer arrays
6. Synthesized inelastic CDF — `P_INEL_CDF_PTRS`
7. **Discrete inelastic level basis/coeffs** — `P_LEVEL_BASIS_PTRS`,
   `P_LEVEL_COEFFS_PTRS` + within-nuc offset arrays
   `P_LEVEL_BLOCAL_OFF`, `P_LEVEL_CLOCAL_OFF`. Rank-padding
   invariant from `1654c4d` preserved per-nuclide.
8. Elastic + per-level angular CDFs — 6 pointer arrays + 3
   un-shifted offset arrays. `sample_level_angular` takes new
   `hit_nuc` parameter (both call sites have it in scope).
9. Fission tabular outgoing-energy distribution — 4 pointer
   arrays + un-shifted offset array.
10. MT=91 continuum — 4 pointer arrays + un-shifted offset array.

`N_PARAMS` 136 → 174 on transport.cu (+38 slots).

## Validation campaign — A1000

| gate | result |
|------|--------|
| lib tests `cargo test --features cuda --lib` | 442 / 442 pass |
| ICSBEP 4-case sweep at production envelope | **4 / 4 PASS** (HMF-001 / HMF-014 / PMF-001 / U233-MF-001) |
| `nu_lookup_compare` | ν table data BIT-IDENTICAL CPU↔GPU |
| `level_xs_compare` | per-level XS bit-identical (Δ = 0.000%) for U-235 MT=51-56 across 1 eV → 5 MeV |
| `metal_stats_diag` (Godiva 3-way) | GPU +109 pcm vs OpenMC; **closer to OpenMC than CPU** (-192 pcm) |

The historic +500-700 pcm fast-metal hot bias from the per-level
rank-padding bug class is NOT reintroduced.

## Outstanding cleanup (Phase 4 — code hygiene only, no behaviour change)

The kernel no longer reads these `GpuNuclideData` fields. They're
still uploaded each call (waste of D→D bandwidth + VRAM):

- **Bundle slabs** — `all_basis`, `all_coeffs`, `total_xs`,
  `pointwise_xs`, `nu_bar_energies`/`values`, `delayed_nu_bar_energies`/
  `values`, `level_basis`, `level_coeffs`, `ang_energies`/`mu`/`cdf`,
  `lev_ang_energies`/`mu`/`cdf`, `fis_inc_energies`/`e_out`/`cdf`/
  `pdf`, `inel91_inc_energies`/`e_out`/`cdf`/`pdf`, `urr_*` (6 tables),
  `inel_cdf_data`.
- **Legacy global-shifted offsets** — `basis_offsets`, `coeffs_offsets`,
  `total_xs_offsets`, `pw_offsets`, `nu_bar_offsets`, `delayed_nu_bar_offsets`,
  `level_basis_offsets`, `level_coeffs_offsets`, `ang_dist_offsets`,
  `lev_ang_dist_off`, `lev_ang_lev_off` (we use the new `_LOCAL_OFF`
  variants), `fis_dist_offsets`, `inel91_dist_offsets`, `urr_offsets`.

Cleanup steps:
1. Drop the fields from `GpuNuclideData`.
2. Drop the corresponding `assemble_*_cat` outputs (or stop calling them).
3. Remove the corresponding `dptr!` lines from `build_transport_params_vec` — but the kernel slot IDs would shift, so simplest is to leave the slots and skip the build. A future "compact slot layout" commit can renumber.
4. Update the byte-equality regression tests (`gpu_per_nuclide::tests`) — they compare per-category against `upload_nuclide_data_uncached_legacy`, which still emits the legacy slabs. Either delete the byte-equality tests (the production transport tests + diagnostic-bin parity now serve as the validation gate) or keep them as a reference for the legacy path.
5. Drop the legacy paths (`upload_nuclide_data_uncached`, `upload_nuclide_data_uncached_legacy`, the entire bundle-assembly stage) once the byte-equality tests are gone.

Estimated cleanup commit: ~200 LOC removal, ~50 LOC test updates.
Defer-justification: production transport is already on the new
ABI; the legacy paths exist only as cache-disabled references.
Cleanup buys ~5-15% VRAM (the unused bundle slabs are small
relative to per-nuclide data on a typical case) and one DtoD pass
saved per `assemble_b_cat` call. Doesn't move correctness or
end-user wall time materially.

## Future work beyond Stage C

- 3080 sweep (376-case) — quantify cross-case sharing wins
  against the bundle-cache baseline.
- WMP runtime temperature interpolation — orthogonal to per-nuc
  cache; for off-library (non-294K/600K) workloads.
- Per-nuc cache hit-rate API — `cache_stats()` method on
  `GpuTransportContext` so sweep harness can report hits/misses.
