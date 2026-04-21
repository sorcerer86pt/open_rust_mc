"""Morning experiment: does U-238 capture (MT=102) stay low-rank under
log-E SVD, and does low-rank reconstruction survive off-library T?

This answers whether the Faddeeva-kernel route is worth pursuing. We
hold out the 900 K library column, rebuild the SVD from the other five
({250, 294, 600, 1200, 2500} K), reconstruct σ(E, 900 K) via √T-linear
interpolation of V^T, and compare against the true 900 K evaluation.

If rank 1-2 is enough for a sub-% reconstruction, the T-structure is
compressible even in plain log-E — the paper's SVD basis just needs
a partition-of-unity interpolation that hugs Doppler shape. If rank
scales badly, log-E is the wrong coordinate and we need ψ/χ or a
denser library before the Faddeeva kernel even makes sense.

Output: outputs/u238_capture_rank_probe.{png,txt}
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
H5 = REPO / "data" / "endfb-vii.1-hdf5" / "neutron" / "U238.h5"
OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)

NUCLIDE = "U238"
MT = 102
TRAIN_TEMPS = ["250K", "294K", "600K", "1200K", "2500K"]
HOLDOUT = "900K"
HOLDOUT_K = 900.0
BRACKET = ("600K", "1200K")  # the two library temps we'd interpolate between

CLAMP = 1e-30  # log floor for zero XS


def read_xs(h5path: Path, mt: int, temp: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5path, "r") as f:
        base = f[f"{NUCLIDE}"]
        e = base[f"energy/{temp}"][:]
        rxn = base[f"reactions/reaction_{mt:03d}"][temp]
        xs = rxn["xs"][:]
        # OpenMC HDF5 convention: xs is offset from the end of the grid
        # (leading zeros implicit). Pad to match.
        if len(xs) < len(e):
            pad = np.zeros(len(e) - len(xs))
            xs = np.concatenate([pad, xs])
    return e, xs


def interp_log(e_src: np.ndarray, xs_src: np.ndarray,
               e_dst: np.ndarray) -> np.ndarray:
    """Log-log linear interpolation onto e_dst."""
    safe = np.maximum(xs_src, CLAMP)
    return np.exp(
        np.interp(np.log(e_dst), np.log(e_src), np.log(safe))
    )


def sqrt_T_alpha(t_target: float, t_lo: float, t_hi: float) -> float:
    return (np.sqrt(t_target) - np.sqrt(t_lo)) / (np.sqrt(t_hi) - np.sqrt(t_lo))


def rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Relative L2 error, optionally over a masked energy range."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b ** 2))
    return num / den


def main() -> None:
    print(f"Loading {NUCLIDE} MT={MT}")

    # 1. Load every temperature on its native grid; unionize to densest.
    all_temps = TRAIN_TEMPS + [HOLDOUT]
    native = {t: read_xs(H5, MT, t) for t in all_temps}
    # Densest = 250K (163 k pts).
    e_ref = native["250K"][0]
    print(f"  reference grid: {len(e_ref)} pts, {e_ref[0]:.2e} to {e_ref[-1]:.2e} eV")

    xs_on_ref = {t: interp_log(e, xs, e_ref) for t, (e, xs) in native.items()}

    # 2. Build training matrix (log₁₀), N_E × N_T_train.
    train_cols = [xs_on_ref[t] for t in TRAIN_TEMPS]
    A_train = np.log10(np.maximum(np.column_stack(train_cols), CLAMP))
    print(f"  training matrix: {A_train.shape} (log10 space)")

    # 3. SVD of the training matrix.
    U, S, Vt = np.linalg.svd(A_train, full_matrices=False)
    print(f"  singular values: {S}")
    print(f"  ratios s_k/s_0:  {S / S[0]}")

    # 4. Reconstruction experiment for the holdout 900 K.
    # √T-linear alpha between the two bracketing training temps.
    bracket_ks = np.array([float(t.rstrip("K")) for t in BRACKET])
    alpha = sqrt_T_alpha(HOLDOUT_K, bracket_ks[0], bracket_ks[1])
    # Bracketing training-column indices.
    i_lo = TRAIN_TEMPS.index(BRACKET[0])
    i_hi = TRAIN_TEMPS.index(BRACKET[1])

    # Region masks for reporting: resonance region (6-300 eV) vs smooth elsewhere.
    res_mask = (e_ref >= 6.0) & (e_ref <= 300.0)    # U-238 RRR tight band
    rrr_mask = (e_ref >= 1.0) & (e_ref <= 1.0e4)    # broader "resolved" band

    true_900 = xs_on_ref[HOLDOUT]

    lines: list[str] = []
    lines.append(f"U-238 capture MT={MT}: off-library 900 K held out")
    lines.append(
        f"  bracket = ({BRACKET[0]}, {BRACKET[1]}); "
        f"sqrt(T)-linear alpha = {alpha:.4f}"
    )
    lines.append(
        f"  singular-value ratios s_k/s_0 = "
        f"[{', '.join(f'{x:.2e}' for x in S / S[0])}]"
    )
    lines.append("")
    lines.append(
        f"  {'rank':>4}  "
        f"{'global L2':>10}  "
        f"{'RRR (1e0–1e4 eV)':>18}  "
        f"{'6.67-eV (6–300 eV)':>20}"
    )

    for k in range(1, len(S) + 1):
        # Truncated basis: first k columns of U and singular values.
        Uk = U[:, :k]
        Sk = S[:k]
        # V^T rows for the two bracketing training temps.
        vt_lo = Vt[:k, i_lo]
        vt_hi = Vt[:k, i_hi]
        coeffs = (1.0 - alpha) * vt_lo + alpha * vt_hi   # (k,)
        log_hat = Uk @ (Sk * coeffs)                      # (N_E,)
        hat = np.power(10.0, log_hat)

        e_global = rel_l2(hat, true_900)
        e_rrr = rel_l2(hat, true_900, rrr_mask)
        e_667 = rel_l2(hat, true_900, res_mask)

        lines.append(
            f"  {k:>4}  {e_global:>10.3e}  {e_rrr:>18.3e}  {e_667:>20.3e}"
        )

    # Baseline: raw linear interp of σ(E,600K) and σ(E,1200K), no SVD.
    raw_hat = ((1.0 - alpha) * xs_on_ref[BRACKET[0]]
               + alpha * xs_on_ref[BRACKET[1]])
    lines.append("")
    lines.append("  raw sqrt(T)-linear of two library columns (no SVD, for reference):")
    lines.append(
        f"    global={rel_l2(raw_hat, true_900):.3e}  "
        f"RRR={rel_l2(raw_hat, true_900, rrr_mask):.3e}  "
        f"6.67={rel_l2(raw_hat, true_900, res_mask):.3e}"
    )

    report = "\n".join(lines)
    print("\n" + report)
    (OUT / "u238_capture_rank_probe.txt").write_text(report)

    # 5. Plot: singular spectrum + reconstruction of the 6.67 eV region
    # for rank 1, 3, 5 vs truth.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); skipping plot")
        return

    fig, (axs, axres) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)

    axs.semilogy(np.arange(1, len(S) + 1), S / S[0], marker="o")
    axs.set_xlabel("rank index")
    axs.set_ylabel(r"$s_k / s_1$")
    axs.set_title(f"Singular spectrum — {NUCLIDE} capture ({len(TRAIN_TEMPS)} train temps)")
    axs.grid(alpha=0.3)

    # Resonance-region reconstruction panel.
    zoom = (e_ref >= 5.5) & (e_ref <= 8.0)   # bracket the 6.67 eV resonance
    axres.loglog(e_ref[zoom], true_900[zoom], "k-", linewidth=1.4, label="truth 900 K")
    for k, colour in [(1, "#ff7f0e"), (3, "#2ca02c"), (5, "#9467bd")]:
        Uk = U[:, :k]
        Sk = S[:k]
        vt_lo = Vt[:k, i_lo]
        vt_hi = Vt[:k, i_hi]
        coeffs = (1.0 - alpha) * vt_lo + alpha * vt_hi
        log_hat = Uk @ (Sk * coeffs)
        hat = np.power(10.0, log_hat)
        axres.loglog(e_ref[zoom], hat[zoom], "--",
                     color=colour, label=f"rank {k}", linewidth=1.0)
    axres.loglog(e_ref[zoom], raw_hat[zoom], ":", color="red",
                 label="sqrt(T)-linear, no SVD", linewidth=1.0)
    axres.set_xlabel("E (eV)")
    axres.set_ylabel("σ_capture (barns)")
    axres.set_title("Reconstruction at 900 K, 6.67 eV region")
    axres.legend(fontsize=8, frameon=False)
    axres.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Does U-238 capture stay low-rank off-library?  "
        f"(train = {TRAIN_TEMPS}, hold out {HOLDOUT})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = OUT / "u238_capture_rank_probe.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    print(f"wrote {OUT / 'u238_capture_rank_probe.txt'}")


if __name__ == "__main__":
    main()
