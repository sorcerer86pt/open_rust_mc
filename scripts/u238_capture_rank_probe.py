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


def ducru_weights_2temp(t_target: float, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Raw 2-point Ducru (2017) weights — L2-optimal in the free-Doppler
    kernel approximation but NOT a partition of unity in general.
    Matches the formula in rust_prototype/src/kernel.rs::ducru_weights.
    """
    if abs(t_target - t_lo) < 1e-6:
        return (1.0, 0.0)
    if abs(t_target - t_hi) < 1e-6:
        return (0.0, 1.0)
    t = t_target
    # w_j = sqrt(T_j T)/(T_j+T) * (T - T_i)/(T + T_i) * (T_j + T_i)/(T_j - T_i)
    w_lo = (
        np.sqrt(t_lo * t) / (t_lo + t)
        * (t - t_hi) / (t + t_hi)
        * (t_lo + t_hi) / (t_lo - t_hi)
    )
    w_hi = (
        np.sqrt(t_hi * t) / (t_hi + t)
        * (t - t_lo) / (t + t_lo)
        * (t_hi + t_lo) / (t_hi - t_lo)
    )
    return (w_lo, w_hi)


def ducru_unity_weights(t_target: float, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Partition-of-unity normalization of the 2-point Ducru weights.
    Preserves the Faddeeva-derived ratio w_lo/w_hi (shape tracking) while
    enforcing w_lo + w_hi = 1 (no log-space gain error on peaks).
    """
    w_lo, w_hi = ducru_weights_2temp(t_target, t_lo, t_hi)
    s = w_lo + w_hi
    if abs(s) < 1e-12:
        return (0.5, 0.5)
    return (w_lo / s, w_hi / s)


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
    bracket_ks = np.array([float(t.rstrip("K")) for t in BRACKET])
    alpha = sqrt_T_alpha(HOLDOUT_K, bracket_ks[0], bracket_ks[1])
    w_lo_du, w_hi_du = ducru_unity_weights(HOLDOUT_K, bracket_ks[0], bracket_ks[1])
    w_lo_raw, w_hi_raw = ducru_weights_2temp(HOLDOUT_K, bracket_ks[0], bracket_ks[1])
    # Bracketing training-column indices.
    i_lo = TRAIN_TEMPS.index(BRACKET[0])
    i_hi = TRAIN_TEMPS.index(BRACKET[1])
    print(
        f"  weights: sqrt(T) alpha = {alpha:.4f}  "
        f"ducru_raw = ({w_lo_raw:.4f}, {w_hi_raw:.4f}) sum={w_lo_raw+w_hi_raw:.4f}  "
        f"ducru_unity = ({w_lo_du:.4f}, {w_hi_du:.4f})"
    )

    # Region masks for reporting: resonance region (6-300 eV) vs smooth elsewhere.
    res_mask = (e_ref >= 6.0) & (e_ref <= 300.0)    # U-238 RRR tight band
    rrr_mask = (e_ref >= 1.0) & (e_ref <= 1.0e4)    # broader "resolved" band

    true_900 = xs_on_ref[HOLDOUT]

    lines: list[str] = []
    lines.append(f"U-238 capture MT={MT}: off-library 900 K held out")
    lines.append(
        f"  bracket = ({BRACKET[0]}, {BRACKET[1]}); "
        f"sqrt(T) alpha = {alpha:.4f}; "
        f"ducru_unity = ({w_lo_du:.4f}, {w_hi_du:.4f})"
    )
    lines.append(
        f"  singular-value ratios s_k/s_0 = "
        f"[{', '.join(f'{x:.2e}' for x in S / S[0])}]"
    )
    lines.append("")
    lines.append(
        f"  {'scheme':>24}  "
        f"{'rank':>4}  "
        f"{'global L2':>10}  "
        f"{'RRR':>10}  "
        f"{'6.67-band':>10}  "
        f"{'peak ratio':>10}"
    )

    # Peak-height check: ratio hat(E_peak)/truth(E_peak) at the 6.67 eV peak.
    # Find the argmax of truth within the resonance zoom.
    peak_idx = int(np.argmax(np.where(res_mask, true_900, -1.0)))
    peak_E = e_ref[peak_idx]
    peak_truth = true_900[peak_idx]

    def record(label: str, rank: int | str, hat: np.ndarray) -> None:
        lines.append(
            f"  {label:>24}  "
            f"{str(rank):>4}  "
            f"{rel_l2(hat, true_900):>10.3e}  "
            f"{rel_l2(hat, true_900, rrr_mask):>10.3e}  "
            f"{rel_l2(hat, true_900, res_mask):>10.3e}  "
            f"{hat[peak_idx] / peak_truth:>10.4f}"
        )

    # Raw two-column interpolations (no SVD).
    raw_sqrt = ((1.0 - alpha) * xs_on_ref[BRACKET[0]]
                + alpha * xs_on_ref[BRACKET[1]])
    record("raw sqrt(T)-linear", "-", raw_sqrt)
    raw_ducru = (w_lo_du * xs_on_ref[BRACKET[0]]
                 + w_hi_du * xs_on_ref[BRACKET[1]])
    record("raw ducru-unity", "-", raw_ducru)

    # SVD-based reconstructions across ranks, both weighting schemes.
    for k in range(1, len(S) + 1):
        Uk = U[:, :k]
        Sk = S[:k]
        vt_lo = Vt[:k, i_lo]
        vt_hi = Vt[:k, i_hi]

        # sqrt(T)-linear on V^T (existing baseline).
        coeffs_sqrt = (1.0 - alpha) * vt_lo + alpha * vt_hi
        hat_sqrt = np.power(10.0, Uk @ (Sk * coeffs_sqrt))
        record("SVD + sqrt(T)-linear", k, hat_sqrt)

        # Normalized Ducru on V^T (new).
        coeffs_du = w_lo_du * vt_lo + w_hi_du * vt_hi
        hat_du = np.power(10.0, Uk @ (Sk * coeffs_du))
        record("SVD + ducru-unity", k, hat_du)

    lines.append(f"\n  peak-ratio column measured at E = {peak_E:.3f} eV "
                 f"(truth = {peak_truth:.1f} barns)")

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

    # Relative error in the 6.67 eV band — makes the Ducru vs sqrt(T)
    # difference visible (in linear-scale absolute-XS the four curves
    # overlap at plot resolution).
    zoom = (e_ref >= 5.5) & (e_ref <= 8.0)
    Uk = U[:, :3]
    Sk = S[:3]
    vt_lo = Vt[:3, i_lo]
    vt_hi = Vt[:3, i_hi]
    hat_sqrt = np.power(10.0, Uk @ (Sk * ((1.0 - alpha) * vt_lo + alpha * vt_hi)))
    hat_du = np.power(10.0, Uk @ (Sk * (w_lo_du * vt_lo + w_hi_du * vt_hi)))
    err_raw_sqrt = (raw_sqrt - true_900) / true_900
    err_raw_du = (raw_ducru - true_900) / true_900
    err_svd_sqrt = (hat_sqrt - true_900) / true_900
    err_svd_du = (hat_du - true_900) / true_900
    axres.axhline(0.0, color="black", linewidth=0.6)
    axres.plot(e_ref[zoom], err_raw_sqrt[zoom] * 100, ":",
               color="#d62728", linewidth=0.9, label="raw sqrt(T)-linear")
    axres.plot(e_ref[zoom], err_raw_du[zoom] * 100, ":",
               color="#2ca02c", linewidth=0.9, label="raw ducru-unity")
    axres.plot(e_ref[zoom], err_svd_sqrt[zoom] * 100, "--",
               color="#d62728", linewidth=1.2, label="rank 3, sqrt(T)-linear")
    axres.plot(e_ref[zoom], err_svd_du[zoom] * 100, "--",
               color="#2ca02c", linewidth=1.2, label="rank 3, ducru-unity")
    axres.axvline(peak_E, color="black", linestyle=":", alpha=0.3)
    axres.set_xscale("log")
    axres.set_ylim(-6, 6)
    axres.set_xlabel("E (eV)")
    axres.set_ylabel("relative error  (hat / truth - 1)  [%]")
    axres.set_title("Reconstruction error at 900 K, 6.67 eV region")
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
