# ICSBEP regression suite — current state (2026-05-11)

The International Criticality Safety Benchmark Evaluation Project
(ICSBEP) is the canonical published validation set for Monte Carlo
criticality codes. This file documents what's wired into the engine,
how to run it, where the canonical specifications come from, and what's
still gated on follow-up work.

## TL;DR

- **`Geometry::from_json` ships** (`src/geometry/scene_io.rs`).
  Deserializes the full open_rust_mc geometry schema (every surface
  variant, every CSG operator, rect and hex lattices). Validated
  against the **367 ICSBEP cases** imported from `mit-crpg/benchmarks`
  — all round-trip cleanly.
- **`material_resolve::resolve_materials`** bridges the schema's
  HDF5-path / ZAID material entries to engine `Material` + an
  `SvdXsProvider` with deduplicated kernels.
- **`cargo test --test icsbep_runs --release -- --ignored`** is the
  canonical regression entry point. Each ICSBEP case is a
  `#[test]` function; pass criterion is `|Δk| < 3σ_combined`.
  - **PASS today (3 cases):** HEU-MET-FAST-001 (Godiva, Δ ≈ −445 pcm,
    1.73σ); PU-MET-FAST-001 (Jezebel, Δ ≈ +131 pcm, 0.41σ);
    PU-MET-FAST-002 (Pu-240-enriched).
  - **KNOWN FAIL (1 case):** U233-MET-FAST-001 (Δ ≈ −2876 pcm) —
    real engine bug exposed by the harness (0.68 b missing-channels
    gap in U-233 SVD decomposition). Tracked as a follow-up.
- **`bench/icsbep/`** holds 367 case JSON files (NMC bundle format,
  `benchmark` + `scene` blocks). Authored by
  `scripts/import_icsbep.py` from the open-source proxy
  `mit-crpg/benchmarks` + `openmc-dev/validation`. Every case carries
  a `data_provenance` chain.
- The legacy `icsbep_bench` CLI orchestrator (subprocess-based) is
  retained for the small handful of cases that wrap a separate
  pre-existing binary (`godiva.rs`, `pwr_pincell.rs`). It will be
  deprecated as those binaries are subsumed into scene-driven cases.

## What runs today

```bash
# Full ICSBEP cargo-test regression (release-mode required, slow):
cargo test --test icsbep_runs --release -- --ignored

# Single case:
cargo test --test icsbep_runs --release -- --ignored pu_met_fast_001_jezebel

# Override the nuclear-data path:
ICSBEP_DATA_DIR=/path/to/endfb-vii.1-hdf5/neutron \
  cargo test --test icsbep_runs --release -- --ignored

# Legacy CLI orchestrator (for hand-authored runner cases):
.\target\release\icsbep_bench.exe ../bench/icsbep \
  --data-dir ../data/endfb-vii.1-hdf5/neutron \
  --output ../outputs/icsbep_bench.csv
```

Add a new case to the cargo-test harness by adding a `#[test]
#[ignore]` function in `tests/icsbep_runs.rs` that calls
`run_case_e2e(...)` against the corresponding `bench/icsbep/*.json`.
The deserializer + material resolver handle everything else.

## Authoritative source vs open-source proxies

The **official** source for ICSBEP benchmark specifications is the
NEA/OECD ICSBEP Handbook, distributed through the registered request
form:

  https://www.oecd-nea.org/science/wpncs/icsbep/order.html

Access is restricted to authorised requesters from OECD member
countries and contributing institutions. Distribution is by DVD and
password-protected GitLab. The click-through agreement explicitly
prohibits redistribution. **We cannot fetch it automatically.**

What we use instead, with full provenance baked into every case file:

| Layer | Source | License | Note |
|---|---|---|---|
| Geometry + materials | `github.com/mit-crpg/benchmarks` (Paul Romano et al., MIT-CRPG) | MIT | Hand-transcribed OpenMC XMLs of the registered handbook by the OpenMC development team. Industry-standard derivative. |
| k_ref + σ_exp | `github.com/openmc-dev/validation` (`benchmarking/uncertainties.csv`, 487 rows) | MIT | Per-case k_ref ± σ_exp table sourced from the handbook by the same team. |
| Canonical source | NEA/OECD ICSBEP Handbook (NEA/NSC/DOC(95)03, 2022–2024 editions) | restricted | The registered, peer-reviewed evaluations. |

**Handbook-fidelity policy.** Any case shipped into our regression
suite that will be cited in a validation report MUST be re-verified
against the registered handbook by an authorised requester before
publication. The open-source proxies are correct in practice but are
not the canonical specifications. Every `bench/icsbep/*.json` file
emitted by `scripts/import_icsbep.py` carries this caveat verbatim in
its `benchmark.data_provenance` block.

## Import pipeline

```text
mit-crpg/benchmarks/icsbep/<case>/openmc/{geometry,materials,settings}.xml
                            │
                            ▼
                scripts/import_icsbep.py
                            │ + uncertainties.csv
                            ▼
            bench/icsbep/<case>.json
              (NMC scene-bundle format, benchmark + scene blocks)
                            │
                            ▼
       Geometry::from_json (src/geometry/scene_io.rs)
                            │
                            ▼
       resolve_materials (src/transport/material_resolve.rs)
                            │ + NuclideLibrary
                            ▼
                  CpuRunner.run(SimConfig)
                            │
                            ▼
                    k_calc ± σ_calc
                            │
                            ▼
        |k_calc − k_ref| ≤ n_σ · √(σ_calc² + σ_exp²)
```

## Coverage today

| Suite | Imported | Runnable today | Notes |
|---|---:|---:|---|
| HEU-MET-FAST | ~80 | 1 (HMF-001) | Godiva variants share materials/geometry; expanding the cargo-test functions is one-line additions. |
| PU-MET-FAST  | ~40 | 2 (PMF-001 / PMF-002) | Jezebel + Pu-240-enriched bare sphere both pass. |
| IEU-MET-FAST | 10  | 0 | Big-Ten, Imf-006/007. One-line additions; intermediate-enrichment U works through the same path. |
| U233-MET-FAST | 6  | 0 of 1 attempted | KNOWN FAIL on U233-MF-001 — see follow-up tracker. |
| HEU-COMP-INTER | 7 | 0 | Imported, deserializer accepts them. Needs cargo-test functions added. |
| HEU-SOL-THERM | 9 | 0 | Solution cases. Imported with `data_provenance`. S(α,β) integration through `MaterialDto.thermal_file` is a follow-up. |
| LEU-COMP-THERM | 1 | 0 | LCT-008 imported (non-lattice case-3/4 only). Most LCT cases use lattices — `Geometry::from_json` supports them, deserializer round-trips, but the import script currently skips them as `uses <lattice>`. |
| LEU-SOL-THERM | 6 | 0 | Same status as HEU-SOL-THERM. |
| MIX-COMP-FAST/THERM | 5 | 0 | MOX cases. Pu fissile chain available. |
| MIX-MET-FAST | 4 | 0 | Pu-U metal mixes. |
| PU-COMP-INTER | 1 | 0 | |
| PU-MET-INTER | 1 | 0 | |
| PU-SOL-THERM | 14 | 0 | |
| SPEC-MET-FAST | 1 | 0 | |
| U233-COMP-THERM / U233-SOL-* | 7 | 0 | |

**Totals: 367 ICSBEP cases imported, 3 wired into the cargo-test harness, 1 known fail.**

Adding more cases is mechanical: one `#[test] #[ignore] fn …` per
case in `tests/icsbep_runs.rs`. The remaining work is investigating
the U-233 missing-channels gap (real engine bug), wiring S(α,β)
through `material_resolve`, and re-enabling the 15 lattice cases that
the import script currently skips.

## Skip and block reasons

The import script's skip log:

```
  15  uses <lattice>                    — converter scope; deserializer supports them
  11  no row in uncertainties.csv        — handbook cases the OpenMC validation team
                                           hasn't keyed in yet
```

The lattice-skip is a script limitation, not an engine limitation —
`Geometry::from_json` happily round-trips `RectLatticeDto` and
`HexLatticeDto` (tested in `geometry::scene_io::tests`). Extending
the Python import script to walk OpenMC's `<lattice>` element is a
1-day follow-up. The 11 missing uncertainties are inherent to the
upstream table; resolution requires either the handbook or an updated
upstream CSV.

## Follow-up backlog

1. **U-233 missing-channels investigation** (tracker task) — find the
   missing reaction channel in `xs_provider`'s MT decomposition for
   U-233. Likely a thermal-region (n,γ) or low-MT channel that's
   covered for U-235/Pu-239 but not U-233.
2. **Extend cargo-test coverage** — add `#[test] #[ignore]` functions
   for the next 5–10 cases in each suite. Targeting full corpus
   coverage is mechanical.
3. **S(α,β) through `material_resolve`** — surface the schema's
   `nuclide.thermal_file` field into the resolved `SvdXsProvider`.
   Unblocks all *-COMP-THERM and *-SOL-THERM cases.
4. **Lattice import in `scripts/import_icsbep.py`** — walk OpenMC's
   `<lattice>` element into `rect_lattices` / `hex_lattices`.
   Unblocks the 15 currently-skipped cases (LCT-008, MIX-COMP-THERM,
   HEU-MET-FAST-026.c-11).
5. **Retire `icsbep_bench` CLI** — the cargo-test harness fully
   replaces it for scene-bearing cases. The hand-authored runner
   cases (3 files) can be reimplemented as scene cases once the
   engine supports the corresponding compositions.
6. **Handbook-fidelity verification pass** — once a validation report
   is being prepared, an authorised requester must re-verify each
   case against the registered NEA handbook and update the
   `data_provenance` block to reflect the verification.

## Files of interest

```
specs/nmc/                                — bundle spec + JSON schema (canonical)
  NMC_SPEC.md                             — §3.1 documents the `benchmark` block
  open_rust_mc_geometry.schema.json       — exact field-by-field schema

scripts/import_icsbep.py                  — bulk converter from mit-crpg/openmc-dev → bench/

bench/icsbep/                             — 370 case files (367 imported + 3 hand-authored runner)
  pu-met-fast-001.json                    — PMF-001 Jezebel reference scene
  heu-met-fast-001_case-1.json            — HMF-001 Godiva case 1
  hmf-001_godiva.json                     — hand-authored runner case (legacy)
  …

rust_prototype/src/
  geometry/scene_io.rs                    — Geometry::from_json (DTOs + converter)
  transport/material_resolve.rs           — MaterialDto → engine Material
  transport/nuclides.rs                   — NuclideLibrary (125+ entries)
  transport/material.rs                   — from_mass_fractions* helpers

rust_prototype/tests/
  icsbep_runs.rs                          — cargo-test ICSBEP harness
  scene_io_corpus.rs                      — corpus deserialization smoke test
```
