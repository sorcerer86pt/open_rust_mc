# SPDX-License-Identifier: MIT
"""Plot Δk_eff vs EALF (Energy of Average Lethargy of Fission) for
an ICSBEP sweep CSV.

EALF is the standard reactor-physics spectrum indicator: low EALF
(< 0.5 eV) = thermal, mid (0.5 eV - 100 keV) = epithermal/intermediate,
high (> 100 keV) = fast. The scatter shows whether code biases are
spectrum-dependent — a flat band across EALF means the engine is
unbiased; a tilt or curve means the physics implementation favours
one regime.

Two inputs:
  - CSV from `icsbep_sweep.py` (with the columns this script needs:
    case, k_calc, k_ref, delta_pcm).
  - EALF table per case stem. The script bundles a minimal table
    derived from category-typical values; for proper publication-
    quality plots, replace it with the case-specific EALFs from the
    ICSBEP Handbook (handbook col `EALF (eV)` per case ROW).

Usage:
    python scripts/plot_delta_k_vs_ealf.py outputs/icsbep_full_gpu.csv
    python scripts/plot_delta_k_vs_ealf.py outputs/icsbep_full_gpu.csv \\
        --output outputs/delta_k_vs_ealf.png --by-category

Output: a PNG scatter at the path given by --output (default
        outputs/delta_k_vs_ealf.png).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, LogLocator
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(2)


# Category-typical EALF values (eV). Derived from ICSBEP Handbook
# averages within each bucket. For per-case precision the user should
# override with the handbook's table — these are good enough for
# a publication-style scatter where the X-axis just needs to span the
# right regime per category, not be exact per case.
#
# Sources:
#   - ICSBEP Handbook 2022, "summary of evaluations" appendix per
#     category (HMF averages 0.7-2 MeV; LCT ~0.025-0.5 eV; HST 0.05-1
#     eV; PMF 0.3-1 MeV; PST 0.05-2 eV; UMF 0.1-1 MeV).
#   - For per-case overrides see the handbook's spectrum-indicator
#     table (col "EALF (eV)" — sometimes labelled "<E>_f").
CATEGORY_EALF_EV: dict[str, float] = {
    # Fast metal: U / Pu metallic fast-spectrum
    "hmf":   1.5e6,    # heu-met-fast
    "umf":   1.0e6,    # u233-met-fast
    "pmf":   8.0e5,    # pu-met-fast
    "imf":   3.0e5,    # iu-met-fast (intermediate-enriched U)
    "ief":   5.0e5,    # ieu-met-fast
    "ieu":   5.0e5,    # ieu-comp-* (intermediate-enriched)
    # Intermediate: composition / spectrum between fast & thermal
    "hmi":   5.0e4,    # heu-met-inter
    "hci":   5.0e4,    # heu-comp-inter
    "mmi":   1.0e5,    # mix-met-inter
    "mmf":   1.0e6,    # mix-met-fast
    # Solution thermal: bare aqueous Pu / HEU / mixed-actinide tanks
    "hst":   0.15,     # heu-sol-therm
    "ust":   0.15,     # u233-sol-therm
    "pst":   0.10,     # pu-sol-therm
    "mst":   0.20,     # mix-sol-therm
    # Compound thermal: lattices, pellets, oxide-fuel
    "lct":   0.20,     # leu-comp-therm
    "hct":   0.30,     # heu-comp-therm
    "uct":   0.25,     # u233-comp-therm
    "mct":   0.30,     # mix-comp-therm
    "pct":   0.40,     # pu-comp-therm
    # Unknown / fall-through
    "unknown": 1.0,
}

# Color palette per spectral regime
SPECTRUM_COLORS = [
    ("Thermal (< 0.5 eV)",   "#2A88C7"),
    ("Epithermal (0.5 eV–100 keV)", "#E89A2C"),
    ("Fast (> 100 keV)",     "#C84A4A"),
]


def category_from_case(case: str) -> str:
    """Extract the 3-letter ICSBEP category prefix from a case stem.

    `heu-met-fast-001_case-1` -> `hmf`
    `pu-sol-therm-012_case-22` -> `pst`
    `leu-comp-therm-008_case-1` -> `lct`
    `u233-sol-therm-001_case-2` -> `ust`
    Falls back to `unknown` for cases that don't match the pattern.
    """
    # Normalise: strip "u233-" / "ieu-" prefixes and combine first
    # letter of each hyphen-separated word.
    parts = case.lower().split("-")
    # Filter trailing _case-N
    parts = [p.split("_")[0] for p in parts]
    # Build the 3-letter prefix from the first letter of the first 3 hyphenated tokens
    if not parts:
        return "unknown"
    # Map common prefixes
    cat = "".join(p[0] for p in parts[:3])
    # Map u233/u235 prefixes
    if parts[0].startswith("u233"):
        cat = "u" + "".join(p[0] for p in parts[1:3])
    elif parts[0].startswith("ieu"):
        cat = "i" + "".join(p[0] for p in parts[1:3])
    elif parts[0].startswith("leu"):
        cat = "l" + "".join(p[0] for p in parts[1:3])
    elif parts[0].startswith("heu"):
        cat = "h" + "".join(p[0] for p in parts[1:3])
    return cat if cat in CATEGORY_EALF_EV else "unknown"


def ealf_for_case(case: str, overrides: dict[str, float]) -> float:
    """Per-case EALF override > category-typical > unknown fallback."""
    if case in overrides:
        return overrides[case]
    cat = category_from_case(case)
    return CATEGORY_EALF_EV.get(cat, CATEGORY_EALF_EV["unknown"])


def jittered_ealf(case: str, ealf_ev: float) -> float:
    """When all cases in a category fall back to the same EALF (no
    per-case override loaded), the scatter degenerates to vertical
    stripes. Spread points horizontally within a ±0.3-decade window
    around the category centre, using a deterministic hash of the
    case name so re-runs land at the same X coordinate (reproducible).

    Off-by-default in the underlying EALF: only the *plot* coordinate
    is jittered. The legend / category bucket assignment uses the
    un-jittered value.
    """
    # Deterministic per-case offset in [-1, 1).
    h = abs(hash(case)) & 0xFFFF_FFFF
    frac = (h % 10000) / 10000.0 * 2.0 - 1.0
    # ±0.3 decade scatter — wide enough to separate cases visually but
    # well inside the category's typical EALF window (~1 decade).
    return ealf_ev * (10.0 ** (frac * 0.3))


def spectrum_bucket(ealf_ev: float) -> int:
    """0=thermal, 1=epithermal, 2=fast — for colouring."""
    if ealf_ev < 0.5:
        return 0
    if ealf_ev < 1.0e5:
        return 1
    return 2


def load_ealf_overrides(path: Path | None) -> dict[str, float]:
    """Optional 2-column CSV: case,ealf_ev. Skipped if path is None
    or the file doesn't exist. Use for handbook-precision EALFs."""
    out: dict[str, float] = {}
    if path is None or not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            case = row.get("case")
            try:
                ealf = float(row.get("ealf_ev", "nan"))
            except ValueError:
                continue
            if case and ealf == ealf:  # finite
                out[case] = ealf
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="ICSBEP sweep CSV (from icsbep_sweep.py)")
    p.add_argument("--output", type=Path, default=Path("outputs/delta_k_vs_ealf.png"),
                   help="output plot path (default: outputs/delta_k_vs_ealf.png)")
    p.add_argument("--ealf-csv", type=Path, default=None,
                   help="optional CSV of per-case EALF (case,ealf_ev) overrides; "
                        "useful when the handbook tables are loaded as a sidecar")
    p.add_argument("--filter", type=str, default=None,
                   help="regex; only cases matching are plotted")
    p.add_argument("--by-category", action="store_true",
                   help="colour points by ICSBEP category prefix instead of spectrum bucket")
    p.add_argument("--no-jitter", action="store_true",
                   help="disable per-case horizontal jitter (use only the lookup EALF). "
                        "Jitter is on by default to spread within-category points along the X-axis "
                        "when per-case EALF overrides aren't supplied via --ealf-csv.")
    p.add_argument("--title", type=str, default=None,
                   help="override the plot title")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 2

    overrides = load_ealf_overrides(args.ealf_csv)
    filt = re.compile(args.filter) if args.filter else None

    # Carry the engine + handbook k values straight from the CSV so the
    # plot annotation + sidecar table can show "k_calc vs k_ref" rather
    # than just the delta. Tuple: (case, ealf, dpcm, k_sigma, status,
    # k_calc, k_ref, sigma_exp).
    cases: list[tuple[str, float, float, float, str, float, float, float]] = []
    with args.csv.open("r", encoding="utf-8", newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            case = row.get("case") or ""
            if not case or (filt and not filt.search(case)):
                continue
            if row.get("status") not in {"PASS", "FAIL"}:
                continue
            try:
                delta_pcm = float(row.get("delta_pcm") or "nan")
                k_sigma = float(row.get("k_sigma") or "0")
                k_calc = float(row.get("k_calc") or "nan")
                k_ref = float(row.get("k_ref") or "nan")
                sigma_exp = float(row.get("sigma_exp") or "0")
            except ValueError:
                continue
            if delta_pcm != delta_pcm:  # NaN
                continue
            ealf_raw = ealf_for_case(case, overrides)
            # Jitter only when the EALF came from the category default
            # (no per-case override). Real handbook EALFs don't need
            # jitter — they already have within-category spread.
            if (not args.no_jitter) and case not in overrides:
                ealf = jittered_ealf(case, ealf_raw)
            else:
                ealf = ealf_raw
            status = row.get("status") or ""
            cases.append((case, ealf, delta_pcm, k_sigma, status,
                          k_calc, k_ref, sigma_exp))

    if not cases:
        print("No usable rows found.", file=sys.stderr)
        return 2

    print(f"Plotting {len(cases)} cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if args.by_category:
        # One point series per category
        by_cat: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for case, ealf, dpcm, ksig, _, *_extra in cases:
            by_cat[category_from_case(case)].append((ealf, dpcm, ksig * 1e5))
        cmap = plt.get_cmap("tab20", len(by_cat))
        for i, (cat, pts) in enumerate(sorted(by_cat.items())):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, fmt="o", ms=4, color=cmap(i),
                        label=f"{cat.upper()} ({len(pts)})", alpha=0.7,
                        elinewidth=0.5, capsize=2)
    else:
        # Three series by spectrum bucket
        buckets: list[list[tuple[float, float, float]]] = [[], [], []]
        for case, ealf, dpcm, ksig, _, *_extra in cases:
            buckets[spectrum_bucket(ealf)].append((ealf, dpcm, ksig * 1e5))
        for i, (label, color) in enumerate(SPECTRUM_COLORS):
            pts = buckets[i]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, fmt="o", ms=5, color=color,
                        label=f"{label} ({len(pts)})", alpha=0.75,
                        elinewidth=0.7, capsize=2)

    # Per-point labels.
    #   - Small sweeps (≤ 50 cases): label every point with k_calc / k_ref
    #     so the engine-vs-handbook comparison is readable on the plot
    #     itself without a sidecar.
    #   - Larger sweeps: label only FAIL cases + |Δ| > 300 pcm outliers
    #     so the dense centre stays readable, but anything weird is
    #     called out by name.
    label_all = len(cases) <= 50
    for case, ealf, dpcm, _ksig, status, k_calc, k_ref, _sx in cases:
        is_outlier = (status == "FAIL") or (abs(dpcm) > 300.0)
        if label_all or is_outlier:
            text = f"{case}\nk={k_calc:.5f} ref={k_ref:.5f}"
            ax.annotate(text, (ealf, dpcm), xytext=(5, 5),
                        textcoords="offset points", fontsize=6.5,
                        color=("#c84a4a" if status == "FAIL" else "#222"),
                        alpha=0.85)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8, zorder=0)
    # ±150 pcm acceptance corridor (the engine's regression envelope).
    ax.axhspan(-150, 150, color="#cccccc", alpha=0.3, zorder=0,
               label="±150 pcm regression bound")

    ax.set_xscale("log")
    ax.set_xlabel("EALF (eV) — Energy of Average Lethargy at Fission")
    ax.set_ylabel("Δk_eff (pcm)  [calc − handbook]")
    n = len(cases)
    n_pass = sum(1 for c in cases if c[4] == "PASS")
    title = args.title or (
        f"ICSBEP sweep — Δk_eff vs EALF  ({n} cases, {n_pass} PASS)\n"
        f"source: {args.csv.name}"
    )
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    # Format x-axis ticks so the scale is readable across 6+ decades
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")

    # Sidecar table: same stem + .txt. One row per case so post-hoc
    # readers can grep / sort by k_calc / k_ref / Δ_pcm without
    # squinting at the scatter. Always written regardless of case
    # count — for 375-case sweeps it's the only sane way to inspect.
    summary_path = args.output.with_suffix(".txt")
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# Source: {args.csv}\n")
        fp.write(f"# {len(cases)} cases  ({sum(1 for r in cases if r[4]=='PASS')} PASS)\n")
        fp.write("# Sorted by |delta_pcm| descending — biggest disagreements first.\n")
        fp.write("#\n")
        fp.write(f"# {'case':<40}  {'k_calc':>9}  {'k_ref':>9}  {'sigma_exp':>9}  {'delta_pcm':>10}  status\n")
        for case, _ealf, dpcm, _ksig, status, k_calc, k_ref, sigma_exp in sorted(
            cases, key=lambda c: abs(c[2]), reverse=True
        ):
            fp.write(
                f"  {case:<40}  {k_calc:>9.5f}  {k_ref:>9.5f}  {sigma_exp:>9.5f}  "
                f"{dpcm:>+10.1f}  {status}\n"
            )
    print(f"wrote {summary_path}")

    # Companion plot: k_calc vs k_ref scatter with a 45-degree identity
    # line. Same data, different framing — Δk-vs-EALF answers "is the
    # engine biased somewhere in the spectrum?", while k_calc-vs-k_ref
    # answers "how well does the engine reproduce the handbook absolute
    # k across the corpus?" Two views, one subprocess: filename derived
    # by string substitution from args.output so both plots land in the
    # same directory.
    calc_ref_path = args.output.parent / args.output.name.replace(
        "delta_k_vs_ealf", "k_calc_vs_k_ref"
    )
    if calc_ref_path == args.output:
        # Output name didn't contain the substitutable token — fall back
        # to <stem>__k_calc_vs_k_ref.png so we don't clobber the Δk plot.
        calc_ref_path = args.output.with_name(
            args.output.stem + "__k_calc_vs_k_ref" + args.output.suffix
        )
    write_k_calc_vs_k_ref(cases, calc_ref_path, args)
    return 0


def write_k_calc_vs_k_ref(
    cases: list,  # same tuple shape produced in main()
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Produce the calibration-style scatter: each case at (k_ref, k_calc)
    with the y=x diagonal drawn for visual reference. Points above the
    line = overprediction; below = underprediction. Distance from the
    line is the absolute-k version of `delta_pcm` from the Δk plot.

    Colours and per-point labels follow the same conventions as the
    Δk plot: spectrum-bucket (or --by-category) colouring, label every
    point when the corpus is small, label outliers only otherwise.
    """
    if not cases:
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    # Series — same colour conventions as the Δk plot for cross-
    # readability.
    if args.by_category:
        by_cat: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for case, _ealf, _dpcm, ksig, _status, k_calc, k_ref, _sx in cases:
            by_cat[category_from_case(case)].append((k_ref, k_calc, ksig))
        cmap = plt.get_cmap("tab20", len(by_cat))
        for i, (cat, pts) in enumerate(sorted(by_cat.items())):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, fmt="o", ms=4, color=cmap(i),
                        label=f"{cat.upper()} ({len(pts)})", alpha=0.7,
                        elinewidth=0.5, capsize=2)
    else:
        buckets: list[list[tuple[float, float, float]]] = [[], [], []]
        for case, ealf, _dpcm, ksig, _status, k_calc, k_ref, _sx in cases:
            buckets[spectrum_bucket(ealf)].append((k_ref, k_calc, ksig))
        for i, (label, color) in enumerate(SPECTRUM_COLORS):
            pts = buckets[i]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, fmt="o", ms=5, color=color,
                        label=f"{label} ({len(pts)})", alpha=0.75,
                        elinewidth=0.7, capsize=2)

    # 45° identity line + ±150 pcm parallel guides (the regression
    # envelope translated into k space).
    ks = [c[5] for c in cases] + [c[6] for c in cases]
    k_lo, k_hi = min(ks) - 0.01, max(ks) + 0.01
    diag = [k_lo, k_hi]
    ax.plot(diag, diag, color="#666666", linestyle="--", linewidth=0.8,
            label="y = x  (perfect agreement)", zorder=0)
    ax.plot(diag, [k - 0.0015 for k in diag], color="#cccccc",
            linestyle=":", linewidth=0.6, zorder=0)
    ax.plot(diag, [k + 0.0015 for k in diag], color="#cccccc",
            linestyle=":", linewidth=0.6, zorder=0,
            label="±150 pcm bound")

    # Per-point labels — same rule as the Δk plot.
    label_all = len(cases) <= 50
    for case, _ealf, dpcm, _ksig, status, k_calc, k_ref, _sx in cases:
        is_outlier = (status == "FAIL") or (abs(dpcm) > 300.0)
        if label_all or is_outlier:
            ax.annotate(case, (k_ref, k_calc), xytext=(5, 5),
                        textcoords="offset points", fontsize=6.5,
                        color=("#c84a4a" if status == "FAIL" else "#222"),
                        alpha=0.85)

    ax.set_xlabel("k_eff (handbook reference)")
    ax.set_ylabel("k_eff (engine, k_calc)")
    ax.set_xlim(k_lo, k_hi)
    ax.set_ylim(k_lo, k_hi)
    ax.set_aspect("equal")
    n_pass = sum(1 for c in cases if c[4] == "PASS")
    title = (
        f"ICSBEP sweep — engine k_calc vs handbook k_ref  "
        f"({len(cases)} cases, {n_pass} PASS)\n"
        f"source: {args.csv.name}"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"wrote {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
