# ICSBEP benchmark suite — deferred spec

A parking-lot specification for running the **International Criticality
Safety Benchmark Evaluation Project** (ICSBEP) handbook against this
engine. Captured here so the audit and phased delivery plan don't get
lost; **execution is deferred** until the harder, research-grade items
ahead of it land first.

This is a **forward-looking spec**, not a status report. For current
state see `STATUS.md`; for round-by-round narrative see `resume.md`.

## Position in the priority queue

ICSBEP is engineering-heavy but research-light. None of it requires new
physics; all of it requires wiring existing data into existing
infrastructure. That's a 3-4 month focused arc. The argument for
deferring it is that the items below all require **new physics** that
ICSBEP-driven validation would help characterize, not the other way
around — running ICSBEP first would lock validation against an engine
that's still missing core capabilities, and we'd have to redo the
runs once the missing pieces land.

**Items that go first** (cross-references to `STATUS.md`):

| Priority | Item | Effort | Why it goes before ICSBEP |
|---|---|---|---|
| 1 | **Time-dependent (point) kinetics** with 6-group delayed-neutron precursors | ~3-4 weeks | Unlocks reactivity-induced excursions, prompt-jump analysis, RBMK / SPERT-style benchmarks. Without it we can't run kinetics-based ICSBEP entries (subcritical neutron-source evaluations) in the BFS / Bigten sub-prompt regime. |
| 2 | **Real continuous-energy adjoint photon MC** (CADIS proper) | ~4-6 weeks | The variance-reduction infrastructure is shipped (`shield_slab` + WW); the FOM gain is gated on adjoint kernels. Running ICSBEP shielding sub-suites (concrete / lead / water reflectors) without working CADIS is wall-time prohibitive. |
| 3 | **CADIS for neutrons** | ~3-4 weeks | Same machinery, transposed for neutron scatter / fission / capture. Shielding ICSBEP entries depend on this for tractable runtimes. |
| 4 | **Full PWR depletion bench** validated against OpenMC's depletion solver | ~3-4 weeks | The depletion module is shipped (CRAM-16/-48, chains, BurnupMapping, fresh-corrector). Validation against OpenMC at long burnup is the missing piece. ICSBEP burnup-credit benchmarks (LWBR, Saxton MOX) live downstream of this. |
| 5 | **Thermal-hydraulic coupling** (or external coupling to RELAP / TRACE) | months / out of scope | Not all ICSBEP cases need this, but the operating-power-feedback ones do (e.g. SPERT-3, BORAX). Out of scope for the engine itself; flagged here only to note that *some* ICSBEP cases are gated on it. |

After items 1-4 land, ICSBEP becomes the natural validation programme
that anchors the codebase to a published, peer-reviewed reference set.
The point of doing it then rather than now is that "X out of 600
ICSBEP cases match within σ_exp" is the single number a reactor-
physics code is judged by — and we want it to mean something across
the *full* engine, not just the static-eigenvalue subset.

## Engine capability matrix vs ICSBEP requirements

### What's wired today

| Capability | Status |
|---|---|
| Bare / reflected sphere geometry | ✅ |
| Cylinder, slab, cubic-array geometries | ✅ |
| Square + hex lattices (CPU + GPU) | ✅ |
| k-eigenvalue with collision + track-length estimators | ✅ |
| Doppler broadening (URR + WMP + Ducru) | ✅ |
| URR equivalence theory for tight lattices | ✅ |
| S(α,β) thermal scattering | ✅ infrastructure ready, only `c_H_in_H2O.h5` wired into a binary |
| Continuous-energy fixed-source shielding | ⚠️ framework in tree (`shield_slab`), FOM-gain pending real adjoint MC |
| Time-dependent kinetics | ❌ static k-eigenvalue only |

### Nuclear data — available vs wired into a binary

```
Available in data/endfb-vii.1-hdf5/neutron/   → 444 nuclide files (full ENDF/B-VII.1)
   + 21 thermal-scattering kernels: c_Graphite, c_Be, c_Be_in_BeO,
     c_H_in_CH2 (polyethylene), c_H_in_H2O, c_D_in_D2O, c_H_in_ZrH (TRIGA),
     c_O_in_UO2, c_U_in_UO2, c_C6H6 (benzene), c_ortho_H, c_para_H, …

Wired into a running binary (NUCLIDE_SPECS in any src/bin/*.rs)  → 10 unique nuclides:
   U-234, U-235, U-238, O-16, H-1, Zr-90/91/92/94, Xe-135
   + WMP: 092234, 092235, 092238
   + S(α,β): c_H_in_H2O only
```

The full ICSBEP-relevant set (Pu-238/239/240/241/242, U-233, Np-237/239,
Am-241, Be-9, C-0, Fe-54/56/57/58, Cr-50/52/53/54, Ni-58/60/61/62/64,
Pb-204/206/207/208, Si-28, Al-27, B-10/11) is **all present as data,
none wired**.

## ICSBEP categorization, by class

ICSBEP entries are tagged `<fissile>-<form>-<spectrum>-NNN`. The full
suite is ~600 evaluated benchmarks across these axes. Per-class status:

| Class | Examples | Engine status |
|---|---|---|
| **HEU-MET-FAST** (bare/reflected HEU metal, fast spectrum) | Godiva (Hmf-001), Topsy (Hmf-005), Flattop-25 | ✅ **Godiva validated within σ_exp**. Topsy/Flattop need depleted-U / water reflector — engineering only, ~1 day each. |
| **IEU-MET-FAST** (intermediate enrichment metal) | Big-Ten (Imf-007) | ⚠️ Big-Ten uses U-235 + U-238 only → **doable today** with a parameter-edit on `godiva.rs`. |
| **PU-MET-FAST** | Jezebel (Pmf-001), Flattop-Pu | ❌ blocked: no Pu-239 in any binary's NUCLIDE_SPECS. ~2 hours to wire Pu-239 + Pu-240 + Pu-241 |
| **PU-MET-INTER** | Pmi-001 through 003 (graphite-reflected Pu) | ❌ blocked on Pu + graphite (C-0 + `c_Graphite.h5`) |
| **LEU-COMP-THERM** (PWR pellet lattices) | Lct-001 to ~100 | ⚠️ `pwr_pincell` is morally a Lct-007-style problem. Specific LCT cases vary in pitch / enrichment / boron / temperature — each is a few hours of binary parameter editing |
| **HEU-COMP-THERM** | Hct-001 (HEU oxide / nitrate) | partial — solution cases need N-14 + acid; oxide cases doable |
| **MIX-COMP-THERM** (MOX) | Mct-001 (Saxton MOX), Tank Cell | ❌ blocked on Pu-239/240/241 wiring |
| **U233-\*** (any) | All U-233 benchmarks | ❌ blocked: U-233 not wired |
| **HEU-SOL-THERM**, **LEU-SOL-THERM**, **PU-SOL-THERM** (nitrate solutions) | Hst-001 (PNL-2), Lst-001, Pst-018 | ❌ blocked: no nitrate species; needs N-14, acid model |
| **\*-MET-INTER** (graphite / beryllium reflected) | Imi-001, Hmi-006 | ❌ blocked: no Be-9 / C-0 wired, no graphite or Be S(α,β) |

## Block breakdown by class of work

The blocks are **entirely engineering**, not physics:

| Block class | What's missing | Effort | Unlocks |
|---|---|---|---|
| Pu benchmarks (~80 cases) | Pu-238/239/240/241 wired into a NUCLIDE_SPECS table; new binary `jezebel.rs`, `flattop_pu.rs` | 1 week to ship Jezebel binary + validate within σ_exp | All Pu-MET-FAST + foundation for MOX |
| U-233 benchmarks (~40 cases) | U-233 wired; new binary | 1 day per case | Jezebel-23 + U-233 thorium-cycle benchmarks |
| Solution benchmarks (~150 cases) | N-14, density-dependent acid model, fissile-in-solution material builder | 1-2 weeks | All `*-SOL-*` (largest single ICSBEP sub-class) |
| Graphite/Be reflected (~50 cases) | C-0 + `c_Graphite.h5` thermal scattering wired; same for Be-9 | 2-3 days each | Pu-MET-INTER, HEU-MET-INTER, RBMK-style problems |
| MOX / mixed-fuel (~40 cases) | Pu wiring + multi-isotope material composition | 1 week after Pu | Saxton, Tank Cell, MOX-fueled fast benchmarks |
| Stainless steel reflected (~30 cases) | Fe / Cr / Ni isotope set wired | 1 week for full alloy | Reactor-vessel-relevant benchmarks |

## The bigger blocker — no benchmark runner

Even if every nuclide were wired, **we can't iterate over the ICSBEP
suite automatically.** Each ICSBEP entry is a hand-crafted MCNP input
deck (~300 lines of `cell` / `surface` / `material` cards). To
run them in our engine we'd need one of:

1. **An MCNP input parser** mapping surfaces / cells / materials onto
   our `geometry` / `Material` types. The MCNP language has hundreds
   of cards but the ICSBEP subset is bounded — geometry is mostly
   spheres, cylinders, simple lattices, planes. Estimated: 2-4 weeks
   for "good enough" coverage.
2. **A YAML/JSON case-spec format** plus a small library of converters
   from ICSBEP MCNP decks. Estimated: 1-2 weeks for the format +
   handful of conversions, plus 1-2 weeks per benchmark cluster.
3. **OpenMC's machinery** — they ship `openmc.examples.icsbep_*` for a
   curated subset. We could call OpenMC from Python, dump its
   geometry, then re-translate. Defeats the purpose; we're a Rust
   engine, not an OpenMC orchestrator.

Recommended path: **option 2** (JSON case-spec format) — bounded scope,
human-readable, version-controllable, and the conversion from MCNP
decks to JSON is a one-shot script that doesn't need to be perfect
because each case can be hand-touched once.

## Phased delivery plan

Total ~13 weeks of focused work. Each phase is independently shippable
and produces a measurable validation number.

### Phase 1 — Fast-spectrum metal suite (~3 weeks)

Cluster of ICSBEP fast-metal benchmarks that share simple geometry
(bare or simply reflected spheres / cylinders).

- Wire Pu-238/239/240/241 into a `pu_metal` binary template.
- Ship binaries for **Jezebel** (Pmf-001), **Flattop-25**
  (Hmf-001-reflected), **Flattop-Pu** (Pmf-006), **Big-Ten**
  (Imf-007), **Topsy** (Hmf-005). Godiva is already validated.
- Validate all 6 against ICSBEP within σ_exp.
- Coverage: ~30 ICSBEP fast-spectrum metal cases (the 6 above + minor
  variants).
- Headline number: "6/6 fast-metal benchmarks within σ_exp."

### Phase 2 — Thermal LWR lattice suite (~2 weeks)

- Generalise `pwr_pincell` into a parameterised binary (pitch /
  enrichment / pin radius / boron concentration / temperature as CLI
  flags).
- Run a sweep across **Lct-001 to Lct-050**.
- Coverage: ~50 ICSBEP LCT cases with one binary.
- Headline number: "X/50 LCT benchmarks within σ_exp."

### Phase 3 — Graphite + Be reflected, U-233 (~3 weeks)

- Wire **C-0** + **Be-9** + **U-233** + thermal-scattering kernels for
  graphite and Be (`c_Graphite.h5`, `c_Be.h5`).
- Three new binary templates: `pu_graphite_reflected.rs`,
  `heu_be_reflected.rs`, `u233_metal.rs`.
- Coverage: ~80 ICSBEP cases (Pmi-001..3, Hmi-* series, U233-MET-FAST,
  U233-MET-THERM).

### Phase 4 — Solution benchmarks (~3 weeks)

- Wire **N-14** + density-dependent acid model + fissile-in-solution
  material builder.
- Solution-tank geometries (cylinders with annular reflectors).
- Coverage: ~150 ICSBEP `*-SOL-*` cases — the single largest
  sub-class.

### Phase 5 — Bench runner + JSON case-spec format (~2 weeks)

- Define `bench/<case>.json` schema with geometry + materials + tally
  + reference k_eff + σ_exp.
- One-shot conversion script for the ICSBEP MCNP decks we want to
  cover (manual touch-ups expected).
- A `bench_runner` binary that ingests a directory of specs, runs each
  through the right engine binary, and emits a CSV of `(k_calc -
  k_ref) / σ_exp` for the whole suite.
- This is what gives us a single regression number ("X% of N
  benchmarks within σ_exp") for ongoing code-health tracking.

## Success criteria

By end of Phase 5:

- ≥ 90 % of run cases land within σ_exp (i.e. `|k_calc - k_ref| <
  σ_exp`).
- ≥ 95 % within 2σ_exp.
- Outliers categorised: nuclear-data issue (compare to OpenMC on the
  same library) vs engine bug vs benchmark-spec interpretation
  ambiguity (these exist; cross-checked with the published
  evaluations).
- The full suite runs unattended (single command, single CSV output).

## Things that explicitly do NOT block ICSBEP

The following items are flagged as research / hard-engineering in
`STATUS.md`, but ICSBEP doesn't depend on them — they can land in
parallel or after:

- Doppler-broadened coherent elastic / Bragg-edge phonon treatment
  (only matters for crystalline-moderated reactors; ICSBEP doesn't
  exercise those at the per-percent level).
- EADL fluorescence / Auger cascade on GPU (photon-side; ICSBEP is
  neutron-eigenvalue dominated).
- Event-based GPU transport (~6× perf gain per Tramm 2024; nice to
  have, doesn't change correctness).
- Photon depletion / activation transport (separate problem class).

## Cross-references

- Current status: [`STATUS.md`](STATUS.md)
- Round-by-round narrative: [`resume.md`](resume.md)
- Engine architecture overview: [`CLAUDE.md`](CLAUDE.md)
