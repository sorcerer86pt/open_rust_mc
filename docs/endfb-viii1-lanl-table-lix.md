# ENDF/B-VIII.1 ground-truth substrate — LANL Table LIX

Companion reference dataset for VIII.1 validation. Used by `tests/cuda_runs.rs`
and `tests/icsbep_runs.rs` (via `resolve_acceptance_target`) so the engine
isn't graded against an experimental k_eff that the new library biases away
from — without losing the canonical handbook as the primary target.

## Source

- **Paper**: Nobre G.P.A. *et al.*, "ENDF/B-VIII.1: Updated Nuclear Reaction
  Data Library for Science and Applications", *Nuclear Data Sheets* (2026)
  — preprint arxiv:2511.03564 (v1: Nov 2025, v2: Apr 2026).
- **Specific table**: TABLE LIX (pages 201-221 of v1) — "Complete list of
  validation results of ENDF/B-VIII.1 in criticality benchmarks performed
  by Los Alamos National Laboratory" with MCNP6 against the ICSBEP
  Handbook.
- **NNDC mirror**: <https://www.nndc.bnl.gov/endf-library/B-VIII.1/summary/>
  (release notes; the table itself is in the paper).
- **Local archived PDFs**: `outputs/refs/endfb_viii1_nobre2025.pdf` (v1,
  9 GB) and `outputs/refs/endfb_viii1_nobre2026v2.pdf` (v2). Not committed
  (in `outputs/.gitignore` scope).

## Columns

```
benchmark        — ICSBEP key, e.g. "HEU-MET-FAST-001-001"
k_exp            — experimental handbook k_eff
sigma_exp        — handbook σ_exp
k_viii1          — LANL MCNP calculated k under ENDF/B-VIII.1
sigma_viii1      — LANL σ_calc (~5-15 pcm typical at 22M histories)
ce_viii1         — C/E ratio (k_viii1 / k_exp)
k_viii0          — same under ENDF/B-VIII.0 (for cross-library shift)
sigma_viii0
ce_viii0
```

## Files in this repo

| File | Rows | What |
|---|---:|---|
| `outputs/refs/endfb_viii1_table_lix.csv` | 1151 | Full TABLE LIX dump |
| `outputs/refs/scene_to_viii1_ref.csv` | 375 | Our scene case_id → TABLE LIX key, with k_handbook left-joined |
| `outputs/refs/orphans_to_run.csv` | 86 | Scenes NOT matched (run OpenMC ourselves) |

## Coverage on our corpus

| Match path | n |
|---|---:|
| ✅ Direct match (`{base}-{NNN}` or `{base}-{N}-{N}` sub-numbering) | 289 / 375 |
| ❌ Evaluation entirely absent from LIX (HCI-003, PCI-001, LST-002/003/016, PST-021, B03AT3V17 mock) | 38 |
| ❌ Sub-case not in LIX (LANL only validated some sub-cases of certain evals) | 48 |

The 86 unmatched cases get a `benchmark.local_validation.viii1` block
populated by `scripts/openmc_orphans_viii1.py` instead — local OpenMC
run on the exact same JSON the engine consumes, under VIII.1.

## How tests consume it

`tests/cuda_runs.rs::resolve_acceptance_target` and
`tests/icsbep_runs.rs::resolve_acceptance_target` walk priority:

```text
1. benchmark.local_validation.viii1.lanl_k_eff   (Nobre 2025 LIX)
2. benchmark.local_validation.viii1.openmc_k_eff (our OpenMC on this JSON)
3. benchmark.local_validation.openmc_k_eff       (legacy VII.1 OpenMC)
4. benchmark.k_eff_reference                     (handbook experimental)
```

`σ_pub` for tiers 1-2 is `max(σ_pub, σ_handbook)` so the envelope never
under-states uncertainty when LANL's `σ_calc` is artificially tight.

## Regenerating the CSV

```bash
# 1. Pull the paper PDF (v1, 9 GB)
mkdir -p outputs/refs
curl -sL https://arxiv.org/pdf/2511.03564v1 \
    -o outputs/refs/endfb_viii1_nobre2025.pdf

# 2. Parse pages 201-221 → endfb_viii1_table_lix.csv
# (parser is the one-shot Python snippet in commit a29e723 message;
#  paste it into a scratch script if regeneration is needed)

# 3. Match against our scenes → scene_to_viii1_ref.csv
python scripts/stamp_lanl_viii1_refs.py --dry-run    # sanity
python scripts/stamp_lanl_viii1_refs.py              # writes blocks
```

## Caveats and known gaps

- TABLE LIX **omits HEU-COMP-INTER entirely** (no Linenberger UH3 cases).
  LANL evidently chose not to validate this class. For HCI-003 (where
  our engine sits +1000-1300 pcm above experiment) the only secondary
  reference is the OpenMC run on the same JSON — included in tier 2.
- The CSV uses ICSBEP-handbook benchmark naming. ICSBEP 2024 / 2025
  editions added new cases (HEU-MET-INTER-013, HEU-MET-FAST-106, etc.)
  but did NOT rename existing evaluations — our matcher is forward-stable.
- LANL's σ_calc is ~5-15 pcm. Our engine's σ_calc at the sweep settings
  (250k × 150 × 5 seeds) is ~50-100 pcm. Use `max(σ_lanl, σ_handbook)`
  to keep the envelope honest.
- The LANL k values are at one specific temperature (293.6 K typically);
  some scenes specify other temperatures and the small TSL drift is
  already absorbed in the published σ band, but flag it for unusually
  cold / hot cases (e.g. KAERI HEU-MET-INTER-006 at 4.5 K).

## Sister table — LLNL Mercury (GNDS-2.0)

Page 218+ of the same paper has **TABLE LX** — 137 entries calculated
by LLNL Mercury using the GNDS-2.0 / FUDGE pipeline. Smaller, focused on
fast critical assemblies. Useful as a *cross-code* check on a subset
(MCNP6 vs Mercury agree to ~10 pcm on shared cases), but our matcher
currently only consumes TABLE LIX since LIX is a strict superset on the
cases we care about. Extending to LX is a one-script refactor if the
secondary reference is ever needed.
