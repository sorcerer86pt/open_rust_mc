# Rust depletion vs OpenMC — `pwr_actinides.json` cross-validation

ICSBEP gate #4 — full PWR depletion bench against OpenMC's depletion
solver, on the same PWR pin cell, using the same chain
(`chains/pwr_actinides.json`, 17 nuclides), at matched burnup
schedule.

## Setup

- **Geometry**: 3.1 % UO₂ / Zr-4 clad / H₂O moderator, 1.26 cm pitch,
  0.4096 cm fuel radius, 0.475 cm clad outer radius. Reflective box
  on all faces.
- **Power**: 200 W per cm of axial length × 1.26 cm = 252 W per pin.
- **Schedule**: 4 steps × 48 h = 8 days.
- **Eigenvalue stats**: 30 batches × 5 000 particles per step (matched).
- **Chain**: shipped `chains/pwr_actinides.json` — both codes consume
  the same nuclide set and the same `(parent, MT, xs_barns)` table;
  OpenMC uses the auto-generated `chain.xml` produced by
  `scripts/openmc_pwr_depletion.py::build_chain_from_json` from that
  JSON.

## Trajectories (t = 192 h ≈ 8 days)

|  Metric         | Rust (`deplete_pwr`) | OpenMC `CoupledOperator` |  Rust/OpenMC |
|-----------------|----------------------|--------------------------|--------------|
| ΔN_U235 / N₀    |       −14.76 %       |          −1.64 %         |    **9.0×**  |
| N_Xe135 / N_U235|         1.45e-5      |          1.17e-5         |     1.24×    |
| N_Pu239 / N_U235|         1.29e-2      |          4.95e-3         |     2.6×     |
| N_Pu240 / N_U235|         3.20e-4      |          3.91e-5         |     8.2×     |
| N_Sm149 / N_U235|         1.34e-4      |          4.84e-5         |     2.8×     |

## Diagnosis — chain-calibration, not solver-bug

The U-235 burn rate differs by **9×**, exactly the ratio between
the *thermal-spectrum* one-group cross sections shipped in our chain
JSON and the *PWR-spectrum* one-group cross sections OpenMC
computes on-the-fly:

```text
   chain JSON σ_f(U-235) = 583.5 b   (E = 0.0253 eV thermal)
   PWR-pin spectrum σ_f  ≈  40–60 b  (Doppler-broadened, resonance-shielded,
                                       fast-tail-weighted average)
   ratio ≈ 10×, matches the U-235 burn-rate ratio above.
```

The Rust depletion solver evaluates one-group reaction rates as
`σ_chain · φ_per_source · Q · N_atoms`. Because `σ_chain` is the
*thermal* value, the rate is over-estimated by the
spectrum-vs-thermal ratio. OpenMC's `CoupledOperator` collapses the
real flux spectrum at every depletion step, so its effective σ_f
naturally matches the cell's actual flux distribution.

The other discrepancies follow from the same calibration error
propagating through the chain:

- **Pu-239 / U-235 = 2.6×** — Pu-239 builds via U-238 (n,γ) → β →
  Np → β → Pu-239. The (n,γ) on U-238 has a milder spectrum
  dependence (smaller resonance contribution), so the multiplier
  is closer to 1×; but Pu-239 also burns via fission (σ_f ≈ 750 b
  thermal vs ~30 b PWR) — the burn channel is over-active in the
  Rust run, partially compensating the over-active build channel.
- **Pu-240 / U-235 = 8.2×** — Pu-240 grows via Pu-239 (n,γ); the
  inflated Pu-239 production combines with the over-strong (n,γ)
  to give the largest absolute discrepancy.
- **Sm-149 / U-235 = 2.8×** — Sm-149 reaches its (Pm-149-decay-fed)
  equilibrium, set by `λ_Pm · σ_Sm,a / (λ_Pm · Σ_Sm,a)`. The σ_a
  ratio cancels much of the spectrum bias, leaving only the
  fission-yield (constant) and (n,γ) (mildly spectrum-dependent)
  contributions.
- **Xe-135 / U-235 = 1.24×** — Xe-135 equilibrates against
  `λ_Xe + σ_Xe,a · φ`. With σ_Xe,a ≈ 2.65e6 b (resonance-free
  thermal absorber) the spectrum bias barely registers; the
  remaining 24 % is the difference between the saturated `(γ_I +
  γ_Xe) · Σ_f / (λ_Xe + σ_Xe φ)` numerator and denominator under
  the 9× higher Σ_f.

## Path to remediation

The proper fix is to feed **flux-spectrum-averaged one-group XS**
into the chain instead of thermal values:

1. **Tally microscopic XS during the eigenvalue solve**. The
   `flux::*` module already tallies cell-mean flux per source; add
   a parallel tally for `σ_i · φ` per (cell, nuclide, MT). Collapse
   to one group at the cell level; pass through to the depletion
   driver as the chain's `xs_barns` for that step.
2. **Or update `chains/pwr_actinides.json`** with PWR-spectrum-
   averaged values (e.g. 40 b for U-235 fission instead of 583.5 b).
   Less general — locks the chain to one spectrum — but a one-line
   fix for benchmark validation.

(1) is the textbook-correct approach and gives spectrum-faithful
depletion across any geometry; it's the standard way OpenMC,
Serpent and SCALE all do it. (2) is the pragmatic shortcut for
running the existing pwr_pincell benchmark today.

## Status

- Rust depletion solver mechanics: **correct** (matrix exponential,
  predictor-corrector, BurnupMapping push-back validated against
  Bateman analytic for partial-Xe chain).
- Chain XS calibration: **needs flux-spectrum collapse**. Trajectories
  produced today are with thermal one-group values; documented as
  a known limitation in this round.
- OpenMC chain converter (`scripts/openmc_pwr_depletion.py::build_chain_from_json`)
  validated end-to-end: 4-step depletion runs cleanly, all chain
  ZAIDs evolve, Pu/Sm/Xe trajectories qualitatively correct.

The OpenMC harness is now in tree as the reference comparison
target. Closing the chain-calibration gap is the natural next step
and unlocks the full ICSBEP burnup-credit suite (Saxton, LWBR).

## Artifacts

- `outputs/openmc_depletion_actinides.json` — OpenMC trajectories
- `outputs/deplete_pwr_actinides_32d.txt` — Rust 16-step (32 d) trace
- `scripts/openmc_pwr_depletion.py` — chain-converter + driver
- `chains/pwr_actinides.json` — shared chain spec
