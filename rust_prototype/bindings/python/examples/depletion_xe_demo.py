"""Xe-135 equilibrium poisoning via the Python depletion API.

Constant-flux Bateman / CRAM demo — same physics as the Rust
`deplete_demo` binary, end-to-end from Python:

  1. Load the partial Xe chain from a JSON file (or paste a JSON
     string in-line; both paths are supported).
  2. Build a starting composition with U-235 only.
  3. Step the composition forward at constant PWR thermal flux
     using `deplete_constant_flux` (CE/LI predictor-corrector with
     CRAM-16 by default; pass `CramOrder.Order48` for stiff cases).
  4. Compare the converged Xe-135 inventory to the textbook
     equilibrium formula.

Usage:
    python rust_prototype/bindings/python/examples/depletion_xe_demo.py \\
        chains/partial_xe.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from open_rust_mc import Chain, CramOrder, deplete_constant_flux


def main() -> int:
    parser = argparse.ArgumentParser(description="Xe-135 equilibrium via Python depletion API")
    parser.add_argument("chain", type=Path,
                        help="Path to a chain JSON file (e.g. chains/partial_xe.json)")
    parser.add_argument("--flux", type=float, default=3.0e14,
                        help="Thermal flux n/(cm² s) — PWR core average ≈ 3e14")
    parser.add_argument("--total-hours", type=float, default=80.0,
                        help="Total burn time. ~80 h reaches Xe-135 equilibrium.")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--n-u235-0", type=float, default=7.19e-4,
                        help="Initial U-235 atom density (atoms/(b·cm)).")
    parser.add_argument("--cram-order", choices=["16", "48"], default="16",
                        help="CRAM order: 16 (default) or 48.")
    args = parser.parse_args()

    chain = Chain.from_file(str(args.chain))
    print(f"Loaded chain: {chain!r}")
    print(f"  description: {chain.description}")

    # Build the initial composition vector indexed by chain order.
    composition = [0.0] * chain.n_nuclides
    u235_idx = chain.index_of_zaid(92235)
    if u235_idx is None:
        print("ERROR: chain has no U-235 (ZAID 92235) — cannot run Xe demo.")
        return 2
    composition[u235_idx] = args.n_u235_0

    order = CramOrder.Order48 if args.cram_order == "48" else CramOrder.Order16
    dt = args.total_hours * 3600.0 / args.steps

    print(f"\n  step    t [h]      N_U235        N_I-135       N_Xe-135      N_Cs-135")
    print(f"  ----  --------  ------------  ------------  ------------  ------------")

    nuclide_list = chain.nuclide_list()
    name_for_zaid = {z: name for z, name in nuclide_list}
    i135_idx = chain.index_of_zaid(53135)
    xe135_idx = chain.index_of_zaid(54135)
    cs135_idx = chain.index_of_zaid(55135)

    for step in range(args.steps):
        composition = deplete_constant_flux(chain, composition, args.flux, dt, order)
        t_hr = (step + 1) * args.total_hours / args.steps
        print(
            f"  {step+1:>4}  {t_hr:>8.2f}  "
            f"{composition[u235_idx]:>12.4e}  "
            f"{composition[i135_idx]:>12.4e}  "
            f"{composition[xe135_idx]:>12.4e}  "
            f"{composition[cs135_idx]:>12.4e}"
        )

    # Analytical equilibrium check.
    # γ_I = 0.06309, γ_Xe = 0.00256 (from u235_thermal_iodine_xenon_yields)
    # λ_I = 2.9264e-5, λ_Xe = 2.10653e-5, σ_a^Xe = 2.65e6 b
    BARN = 1.0e-24
    sigma_f_thermal = 583.5  # b — same as in the chain
    gamma_i = 0.06309
    gamma_xe = 0.00256
    lambda_i = 2.9264e-5
    lambda_xe = 2.10653e-5
    sigma_a_xe = 2.65e6

    sigma_f_macro = composition[u235_idx] * sigma_f_thermal
    n_i_eq = gamma_i * sigma_f_macro * args.flux * BARN / lambda_i
    n_xe_eq = (
        (gamma_i + gamma_xe) * sigma_f_macro * args.flux * BARN
        / (lambda_xe + sigma_a_xe * BARN * args.flux)
    )
    n_i_cram = composition[i135_idx]
    n_xe_cram = composition[xe135_idx]

    print()
    print("  ── Equilibrium comparison ──────────────────────────────────")
    print(f"  N_I-135   analytical = {n_i_eq:>12.4e}    CRAM-{args.cram_order} = {n_i_cram:>12.4e}    Δ = {100*(n_i_cram - n_i_eq)/n_i_eq:>+5.2f}%")
    print(f"  N_Xe-135  analytical = {n_xe_eq:>12.4e}    CRAM-{args.cram_order} = {n_xe_cram:>12.4e}    Δ = {100*(n_xe_cram - n_xe_eq)/n_xe_eq:>+5.2f}%")
    print()
    print(f"  N_Xe / N_U235 (atomic ratio) = {n_xe_cram / max(composition[u235_idx], 1e-30):.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
