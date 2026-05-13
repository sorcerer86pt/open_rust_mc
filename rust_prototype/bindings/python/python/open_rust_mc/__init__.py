"""open-rust-mc: Python bindings for a pure-Rust Monte Carlo neutron transport engine.

This is a thin convenience wrapper around the native `_core` extension
compiled from `bindings/python/src/lib.rs`. All validation and
simulation logic lives on the Rust side — Python is only for scripting.

Example
-------
    >>> from open_rust_mc import Scene, Material, Sphere, Settings, run_eigenvalue
    >>> fuel = (Material("HEU", temperature=294.0)
    ...     .add_nuclide("U234.h5", atom_density=0.000483, awr=232.029, nubar=2.49)
    ...     .add_nuclide("U235.h5", atom_density=0.04509,  awr=233.025, nubar=2.43)
    ...     .add_nuclide("U238.h5", atom_density=0.00265,  awr=236.006, nubar=2.49))
    >>> scene = (Scene("data/endfb-vii.1-hdf5/neutron")
    ...     .add_material("heu", fuel)
    ...     .add_surface("boundary", Sphere(r=8.7407, bc="vacuum"))
    ...     .add_cell("fuel", region="-boundary", fill="heu", temperature=294.0)
    ...     .add_cell("outside", region="+boundary"))
    >>> result = run_eigenvalue(scene, Settings(batches=50, inactive=10, particles=5000))
    >>> print(result.k_eff, result.k_sigma)
"""

from ._core import (
    # Surfaces
    Sphere,
    ZCylinder,
    XCylinder,
    YCylinder,
    XPlane,
    YPlane,
    ZPlane,
    # Entities
    Material,
    PhotonMaterial,
    Settings,
    Scene,
    XsMode,
    Runner,
    # Results
    EigenvalueResult,
    GammaHeatingResult,
    IcsbepResult,
    # Top-level functions
    run_eigenvalue,
    run_gamma_heating,
    run_icsbep_case,
    # Depletion
    Chain,
    CramOrder,
    cram,
    deplete_constant_flux,
    deplete_with_flux_callback,
)

# Avogadro's number, exact SI-2019 value.
_N_A = 6.02214076e23

# Atomic weight ratios and atomic masses (g/mol) for common nuclides.
# `awr` is the mass relative to the neutron mass; `a` is the molar mass
# in g/mol. Both are drawn from the ENDF/B-VII.1 HDF5 attributes so
# they match what the Rust side reads at simulation time.
NUCLIDE_DATA: dict[str, dict[str, float]] = {
    # Isotope           awr      a (g/mol)      default ν̄   HDF5 filename
    "U234": {"awr": 232.0302, "a": 234.0410, "nubar": 0.0,  "file": "U234.h5"},
    "U235": {"awr": 233.0250, "a": 235.0439, "nubar": 2.43, "file": "U235.h5"},
    "U238": {"awr": 236.0058, "a": 238.0508, "nubar": 2.49, "file": "U238.h5"},
    "Pu239": {"awr": 236.9986, "a": 239.0522, "nubar": 2.88, "file": "Pu239.h5"},
    "Pu240": {"awr": 237.9916, "a": 240.0538, "nubar": 2.80, "file": "Pu240.h5"},
    "Pu241": {"awr": 238.9851, "a": 241.0568, "nubar": 2.93, "file": "Pu241.h5"},
    "O16":  {"awr": 15.8575,   "a": 15.9949,  "nubar": 0.0,  "file": "O16.h5"},
    "H1":   {"awr": 0.9991673, "a": 1.00783,  "nubar": 0.0,  "file": "H1.h5"},
    "H2":   {"awr": 1.9968,    "a": 2.01410,  "nubar": 0.0,  "file": "H2.h5"},
    "Zr90": {"awr": 89.1322,   "a": 89.9047,  "nubar": 0.0,  "file": "Zr90.h5"},
    "Zr91": {"awr": 90.1297,   "a": 90.9056,  "nubar": 0.0,  "file": "Zr91.h5"},
    "Zr92": {"awr": 91.1256,   "a": 91.9050,  "nubar": 0.0,  "file": "Zr92.h5"},
    "Zr94": {"awr": 93.1197,   "a": 93.9063,  "nubar": 0.0,  "file": "Zr94.h5"},
    "B10":  {"awr": 9.9269,    "a": 10.0129,  "nubar": 0.0,  "file": "B10.h5"},
    "B11":  {"awr": 10.9147,   "a": 11.0093,  "nubar": 0.0,  "file": "B11.h5"},
    "C0":   {"awr": 11.8966,   "a": 12.011,   "nubar": 0.0,  "file": "C0.h5"},
    "Fe54": {"awr": 53.4761,   "a": 53.9396,  "nubar": 0.0,  "file": "Fe54.h5"},
    "Fe56": {"awr": 55.4547,   "a": 55.9349,  "nubar": 0.0,  "file": "Fe56.h5"},
    "N14":  {"awr": 13.8827,   "a": 14.0031,  "nubar": 0.0,  "file": "N14.h5"},
}


def _lookup(nuc: str) -> dict[str, float]:
    if nuc not in NUCLIDE_DATA:
        raise KeyError(
            f"no atomic data for {nuc!r}. Add it to NUCLIDE_DATA "
            f"(awr, a, nubar, file) or use Material.add_nuclide with "
            f"explicit awr and atom_density."
        )
    return NUCLIDE_DATA[nuc]


def atom_density_from_mass_density(
    mass_density_g_per_cm3: float,
    molar_mass_g_per_mol: float,
) -> float:
    """Convert a mass density + molar mass to atom density [atoms/(b·cm)].

    Formula: n = ρ · N_A / M, then × 1e-24 to convert cm⁻³ → /(b·cm).
    """
    return mass_density_g_per_cm3 * _N_A / molar_mass_g_per_mol * 1.0e-24


def uranium_oxide_material(
    name: str,
    density_g_per_cm3: float,
    enrichment: float,
    temperature: float = 900.0,
    temp_idx: int = 3,
) -> Material:
    """Build a UO₂ material at the given U-235 enrichment (atom fraction).

    Example
    -------
        >>> fuel = uranium_oxide_material("UO2 3.1%", 10.4, 0.031)
        >>> # → U234, U235, U238, O16 with correct atom densities

    Parameters
    ----------
    name : str
        Display name for the material (passed to the engine).
    density_g_per_cm3 : float
        Macro mass density (e.g. 10.4 for fresh PWR fuel pellets).
    enrichment : float
        U-235 atom fraction within the uranium (0.031 = 3.1 % enriched).
        U-234 is set to 1 % of the U-235 value — a reasonable default for
        commercial enriched uranium; override with `add_nuclide` for
        exact HEU compositions.
    temperature : float
        Bulk cell temperature in K.
    temp_idx : int
        HDF5 library-temperature index. For ENDF/B-VII.1, idx=3 is 900 K.
    """
    u235 = _lookup("U235")
    u238 = _lookup("U238")
    u234 = _lookup("U234")
    o16 = _lookup("O16")

    # U-234 fraction: ~1% of U-235 is a common trace-level approximation
    # for low-enriched uranium. Set to 0 for higher precision needs.
    u234_frac = 0.01 * enrichment
    u235_frac = enrichment
    u238_frac = 1.0 - enrichment - u234_frac

    # Molar mass of the UO₂ molecule:
    m_u = (
        u234_frac * u234["a"]
        + u235_frac * u235["a"]
        + u238_frac * u238["a"]
    )
    m_uo2 = m_u + 2.0 * o16["a"]

    # Molecule atom density in atoms/(b·cm).
    n_molec = atom_density_from_mass_density(density_g_per_cm3, m_uo2)

    mat = Material(name, temperature, temp_idx)
    mat.add_nuclide(u234["file"], u234_frac * n_molec, u234["awr"], u234["nubar"])
    mat.add_nuclide(u235["file"], u235_frac * n_molec, u235["awr"], u235["nubar"])
    mat.add_nuclide(u238["file"], u238_frac * n_molec, u238["awr"], u238["nubar"])
    mat.add_nuclide(o16["file"], 2.0 * n_molec, o16["awr"], o16["nubar"])
    return mat


def water_material(
    name: str = "H2O",
    density_g_per_cm3: float = 0.74,
    temperature: float = 600.0,
    temp_idx: int = 2,
    thermal_file: str | None = "c_H_in_H2O.h5",
) -> Material:
    """Build a H₂O moderator material.

    By default attaches the S(α,β) thermal scattering library for
    hydrogen bound in water; pass `thermal_file=None` to use free-gas
    kinematics only. The default density 0.74 g/cm³ is the PWR hot-leg
    value; 1.00 g/cm³ is room temperature.
    """
    h1 = _lookup("H1")
    o16 = _lookup("O16")

    m_h2o = 2.0 * h1["a"] + o16["a"]
    n_molec = atom_density_from_mass_density(density_g_per_cm3, m_h2o)

    mat = Material(name, temperature, temp_idx)
    mat.add_nuclide(
        h1["file"], 2.0 * n_molec, h1["awr"], h1["nubar"], thermal_file=thermal_file
    )
    mat.add_nuclide(o16["file"], n_molec, o16["awr"], o16["nubar"])
    return mat


def zircaloy4_material(
    name: str = "Zircaloy-4",
    density_g_per_cm3: float = 6.55,
    temperature: float = 600.0,
    temp_idx: int = 2,
) -> Material:
    """Build a Zircaloy-4 cladding material with the four major Zr isotopes.

    Zr-96 is zeroed to match the Rust PWR pin cell configuration. The
    isotope ratios are natural abundances.
    """
    zr_isotopes = [("Zr90", 0.5063), ("Zr91", 0.1103), ("Zr92", 0.1686), ("Zr94", 0.1709)]
    m_zr = sum(frac * _lookup(iso)["a"] for iso, frac in zr_isotopes) / sum(
        frac for _, frac in zr_isotopes
    )
    n_zr = atom_density_from_mass_density(density_g_per_cm3, m_zr)

    mat = Material(name, temperature, temp_idx)
    for iso, frac in zr_isotopes:
        d = _lookup(iso)
        mat.add_nuclide(d["file"], frac * n_zr, d["awr"], d["nubar"])
    return mat


__all__ = [
    "Sphere",
    "ZCylinder",
    "XCylinder",
    "YCylinder",
    "XPlane",
    "YPlane",
    "ZPlane",
    "Material",
    "PhotonMaterial",
    "Settings",
    "Scene",
    "XsMode",
    "Runner",
    "EigenvalueResult",
    "GammaHeatingResult",
    "IcsbepResult",
    "run_eigenvalue",
    "run_gamma_heating",
    "run_icsbep_case",
    # Depletion
    "Chain",
    "CramOrder",
    "cram",
    "deplete_constant_flux",
    "deplete_with_flux_callback",
    # Helpers
    "NUCLIDE_DATA",
    "atom_density_from_mass_density",
    "uranium_oxide_material",
    "water_material",
    "zircaloy4_material",
]

__version__ = "0.1.0"
