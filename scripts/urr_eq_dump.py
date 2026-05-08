"""Diagnostic dump of the URR-equivalence math at a PWR pin cell point.
Uses OpenMC's HDF5 directly to get U-238 ground-truth XS in the URR
window, then walks through what the Rust apply_equivalence_correction
would do step by step. The goal is to attribute the +764 pcm Rust
shift either to (a) numerics in Rust, (b) the formula's domain of
applicability, or (c) the parameter choices."""
import sys
import h5py
import numpy as np
from math import pi, exp

DATA = "/mnt/c/Users/fog/madman_svd_experiment/data/endfb-vii.1-hdf5/neutron"


def get_xs(file, mt, t_idx_label):
    with h5py.File(f"{DATA}/{file}", "r") as f:
        nuc = list(f.keys())[0]
        E = f[nuc]["energy"][t_idx_label][:]
        rxn_grp = f[nuc][f"reactions/reaction_{mt:03d}"][t_idx_label]
        thr = int(rxn_grp.attrs.get("threshold_idx", 0))
        xs = rxn_grp["xs"][:]
        full = np.zeros_like(E)
        if thr <= 1:
            full[: len(xs)] = xs
        else:
            full[thr - 1 : thr - 1 + len(xs)] = xs
        return E, full


def get_urr(file, t_idx_label):
    with h5py.File(f"{DATA}/{file}", "r") as f:
        nuc = list(f.keys())[0]
        u = f[nuc]["urr"][t_idx_label]
        E = u["energy"][:]
        tab = u["table"][:]
        return E, tab


# Geometry
PITCH = 1.26
FUEL_OR = 0.475
FUEL_R = 0.4096
SIGMA_M = 1.5

# Carlvik-Pellaud Dancoff
v_m = PITCH ** 2 - pi * FUEL_OR ** 2
s_pin = 2 * pi * FUEL_OR
l_bar_m = 4 * v_m / s_pin
alpha = 4.58
C = exp(-SIGMA_M * l_bar_m / alpha)
print(f"Geometry: pitch={PITCH} fuel_or={FUEL_OR} fuel_r={FUEL_R}")
print(f"  v_m = {v_m:.4f} cm^2/cm")
print(f"  s_pin = {s_pin:.4f} cm")
print(f"  l_bar_m (mod chord) = {l_bar_m:.4f} cm")
print(f"  Carlvik-Pellaud C(sigma_m={SIGMA_M}) = {C:.4f}")
print()

# Mean chord through fuel
L_BAR_FUEL = 2 * FUEL_R
print(f"Fuel mean chord l_bar = 2R = {L_BAR_FUEL:.4f} cm")
N_U238 = 0.022482
print(f"N_U238 = {N_U238:.5e} atoms/(b*cm)")
sigma_e = (1.0 - C) / (N_U238 * L_BAR_FUEL)
print(f"sigma_e = (1-C)/(N*l) = {sigma_e:.3f} b")
print()

# Background sigma_0 at URR window (use real U-235 + O-16 averages)
print("U-235 + O-16 XS at URR window (from HDF5, 900K):")
E_u235, s_t_u235_grid = get_xs("U235.h5", 1, "900K") if False else (None, None)
# Use total XS via summation: capture + elastic for low-E approximation.
E_u235, s_c_u235 = get_xs("U235.h5", 102, "900K")
_, s_el_u235 = get_xs("U235.h5", 2, "900K")
mask_u235 = (E_u235 >= 2e4) & (E_u235 <= 1.5e5)
sigma_t_u235 = np.trapezoid(
    (s_c_u235 + s_el_u235)[mask_u235], E_u235[mask_u235]
) / np.trapezoid(np.ones(mask_u235.sum()), E_u235[mask_u235])
print(f"  <sigma_t U-235> over 20-150 keV: {sigma_t_u235:.3f} b (capture+elastic)")

E_o16, s_c_o16 = get_xs("O16.h5", 102, "900K")
_, s_el_o16 = get_xs("O16.h5", 2, "900K")
mask_o16 = (E_o16 >= 2e4) & (E_o16 <= 1.5e5)
sigma_t_o16 = np.trapezoid(
    (s_c_o16 + s_el_o16)[mask_o16], E_o16[mask_o16]
) / np.trapezoid(np.ones(mask_o16.sum()), E_o16[mask_o16])
print(f"  <sigma_t O-16>  over 20-150 keV: {sigma_t_o16:.3f} b (capture+elastic)")

N_U235 = 7.19e-4
N_O16 = 4.6402e-2
sigma_0 = (N_U235 * sigma_t_u235 + N_O16 * sigma_t_o16) / N_U238
print(f"\nBackground sigma_0 (per U-238 atom) at ~50 keV:")
print(f"  N_U235 * sigma_t_U235 / N_U238 = {N_U235 * sigma_t_u235 / N_U238:.3f} b")
print(f"  N_O16  * sigma_t_O16  / N_U238 = {N_O16 * sigma_t_o16 / N_U238:.3f} b")
print(f"  sigma_0 = {sigma_0:.3f} b")
print()

factor = sigma_0 / (sigma_0 + sigma_e)
print(f"Rust correction factor sigma_0/(sigma_0+sigma_e) = {factor:.4f}")
print(f"  applied uniformly to elastic, fission, capture of U-238 URR sample.")
print(f"  -> URR XS of U-238 reduced to {100 * factor:.1f}% of dilute value")
print(f"  -> {100 * (1 - factor):.1f}% effective reduction")
print()

print("U-238 ground truth at URR window (from HDF5, 900K):")
E_xs, sigma_c = get_xs("U238.h5", 102, "900K")
_, sigma_el = get_xs("U238.h5", 2, "900K")
mask = (E_xs >= 2e4) & (E_xs <= 1.5e5)
sigma_c_avg = np.trapezoid(sigma_c[mask], E_xs[mask]) / np.trapezoid(np.ones(mask.sum()), E_xs[mask])
sigma_el_avg = np.trapezoid(sigma_el[mask], E_xs[mask]) / np.trapezoid(np.ones(mask.sum()), E_xs[mask])
print(f"  <sigma_capture> over URR window (E-averaged): {sigma_c_avg:.4f} b")
print(f"  <sigma_elastic> over URR window: {sigma_el_avg:.4f} b")
print()

print("U-238 URR PT (HDF5):")
try:
    E_urr, urr_tab = get_urr("U238.h5", "900K")
    print(f"  URR energy range: {E_urr.min():.3e} - {E_urr.max():.3e} eV")
    print(f"  URR table shape: {urr_tab.shape}  (n_E, 6 channels, n_bands)")
except Exception as e:
    print(f"  (URR PT not loadable): {e}")
print()

print(f"After Rust eq correction (factor {factor:.4f}):")
print(f"  capture: {sigma_c_avg:.4f} -> {sigma_c_avg * factor:.4f} b ({100 * (factor - 1):+.1f}%)")
print(f"  elastic: {sigma_el_avg:.4f} -> {sigma_el_avg * factor:.4f} b ({100 * (factor - 1):+.1f}%)")
print()

print("=== Reference: textbook self-shielding magnitude ===")
print("Bondarenko-like shielding factor for U-238 in standard PWR fuel")
print("at URR (50 keV): in the 0.85-0.98 range (Sanchez 1981 Table II,")
print("Stamm'ler 1983 §6.4, NJOY PURR module benchmarks).")
print(f"Rust's effective factor: {factor:.4f}")
print(f"Rust's effective reduction: {100 * (1 - factor):.1f}% (textbook: 2-15%)")
print()

print("=== Root cause hypothesis ===")
print("The rational approximation sigma_eff = sigma_inf * sigma_0/(sigma_0+sigma_e)")
print("is the Bondarenko shielding factor for the **resonance integral**, i.e.")
print("the resonance-fluctuation part of the absorption XS above the smooth")
print("baseline. In the URR, only the resonance ladder contributes to the")
print("sample's deviation from smooth — potential elastic and smooth capture")
print("should NOT be shielded.")
print()
print("Rust applies the factor to the FULL elastic, fission, and capture URR")
print("samples — including the smooth baseline. Since the URR window of U-238")
print("has elastic XS dominated by potential scattering (~9 b smooth) and")
print("capture dominated by smooth s-wave (~0.3-0.5 b smooth), reducing them")
print(f"by {100 * (1 - factor):.0f}% over-shields by an order of magnitude.")
print()
print("Fix sketch: apply the factor only to (sigma_URR - sigma_smooth) with")
print("sigma_smooth being the off-resonance interpolant. NJOY PURR uses the")
print("Hwang superposition method; equivalent to keeping the dilute-limit")
print("baseline and shielding only the resonance contribution.")
print()
print("Quick patch alternative: gate the correction on capture only (the")
print("only channel where URR fluctuations dominate over smooth baseline)")
print("— would cut the over-correction by ~10x and likely get within 50-200 pcm.")
