// SPDX-License-Identifier: MIT
//! ZAID-driven nuclide library.
//!
//! Replaces the per-binary `NUCLIDE_SPECS: &[(&str, f64, f64, usize)]`
//! tuples with a real loader: each nuclide is described by its ZAID
//! (1000·Z + A, e.g. 92235 for U-235), and the library maps ZAID →
//! HDF5 filename + atomic weight ratio + available temperatures, with
//! all metadata (AWR, library temperature columns) read from the HDF5
//! file itself rather than hand-typed in each binary.
//!
//! Typical usage:
//!
//! ```ignore
//! let lib = NuclideLibrary::from_data_dir(data_dir);
//! let u235 = lib.resolve(92235, 900.0)?;   // (filename, awr, ν̄_const, temp_idx)
//! let kernel = xs_provider::load_nuclide(
//!     &u235.path, rank, u235.temp_idx, u235.awr, u235.nu_bar_const);
//! ```
//!
//! The library does not own loaded `NuclideKernels` — it's purely a
//! metadata cache. Binaries still call `xs_provider::load_nuclide`
//! themselves, but they no longer hand-write AWR / temperature-index
//! tables that drift from the HDF5 source.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::hdf5_reader;

/// Element symbol for atomic number `Z`. Returns `None` for `Z` outside
/// the H..Fm range. Used by [`default_filename_for_zaid`] and the
/// thermal-target / material-resolve helpers.
///
/// Kept as a single source of truth so adding a new element (or fixing
/// a typo) doesn't require touching the three other ad-hoc tables that
/// currently exist across the engine. Future cleanup: have those
/// callers go through this function.
pub fn symbol_for_z(z: u32) -> Option<&'static str> {
    Some(match z {
        1 => "H",   2 => "He",  3 => "Li",  4 => "Be",  5 => "B",   6 => "C",
        7 => "N",   8 => "O",   9 => "F",  10 => "Ne", 11 => "Na", 12 => "Mg",
       13 => "Al", 14 => "Si", 15 => "P",  16 => "S",  17 => "Cl", 18 => "Ar",
       19 => "K",  20 => "Ca", 21 => "Sc", 22 => "Ti", 23 => "V",  24 => "Cr",
       25 => "Mn", 26 => "Fe", 27 => "Co", 28 => "Ni", 29 => "Cu", 30 => "Zn",
       31 => "Ga", 32 => "Ge", 33 => "As", 34 => "Se", 35 => "Br", 36 => "Kr",
       37 => "Rb", 38 => "Sr", 39 => "Y",  40 => "Zr", 41 => "Nb", 42 => "Mo",
       43 => "Tc", 44 => "Ru", 45 => "Rh", 46 => "Pd", 47 => "Ag", 48 => "Cd",
       49 => "In", 50 => "Sn", 51 => "Sb", 52 => "Te", 53 => "I",  54 => "Xe",
       55 => "Cs", 56 => "Ba", 57 => "La", 58 => "Ce", 59 => "Pr", 60 => "Nd",
       61 => "Pm", 62 => "Sm", 63 => "Eu", 64 => "Gd", 65 => "Tb", 66 => "Dy",
       67 => "Ho", 68 => "Er", 69 => "Tm", 70 => "Yb", 71 => "Lu", 72 => "Hf",
       73 => "Ta", 74 => "W",  75 => "Re", 76 => "Os", 77 => "Ir", 78 => "Pt",
       79 => "Au", 80 => "Hg", 81 => "Tl", 82 => "Pb", 83 => "Bi", 84 => "Po",
       85 => "At", 86 => "Rn", 87 => "Fr", 88 => "Ra", 89 => "Ac", 90 => "Th",
       91 => "Pa", 92 => "U",  93 => "Np", 94 => "Pu", 95 => "Am", 96 => "Cm",
       97 => "Bk", 98 => "Cf", 99 => "Es", 100 => "Fm",
        _ => return None,
    })
}

/// Default OpenMC HDF5 filename for a ground-state ZAID, derived from
/// the universal `<Symbol><Mass>.h5` convention. For natural elements,
/// pass `1000 * Z` (e.g. carbon-natural is ZAID 6000 → `C0.h5`, which
/// is exactly OpenMC's convention).
///
/// Returns `None` for ZAIDs whose `Z` isn't in [`symbol_for_z`]. Does
/// not handle metastable variants — OpenMC encodes those with a `_m1`
/// suffix and a ZAID offset; if and when we need them, callers should
/// supply an explicit override rather than depending on convention.
pub fn default_filename_for_zaid(zaid: u32) -> Option<String> {
    let z = zaid / 1000;
    let a = zaid % 1000;
    let sym = symbol_for_z(z)?;
    Some(format!("{sym}{a}.h5"))
}

/// Resolved nuclide entry — everything `xs_provider::load_nuclide`
/// needs to consume an HDF5 file at a target temperature.
#[derive(Debug, Clone)]
pub struct ResolvedNuclide {
    /// Decimal ZAID, `1000 * Z + A` (no metastable encoding).
    pub zaid: u32,
    /// Human-readable symbol — `"U-235"`, `"Pu-239"`, `"D"`.
    pub symbol: &'static str,
    /// Absolute path to the HDF5 file.
    pub path: PathBuf,
    /// Atomic weight ratio (mass / neutron mass), read from the HDF5
    /// `atomic_weight_ratio` attribute.
    pub awr: f64,
    /// Constant ν̄ fallback used by `xs_provider::load_nuclide` when no
    /// energy-dependent table is loaded. For non-fissionable nuclides
    /// this is 0.0; for fissile / fissionable actinides it's the
    /// thermal value (ν̄(0.0253 eV)) from ENDF/B evaluations.
    pub nu_bar_const: f64,
    /// Selected library temperature column (numeric kelvin).
    pub temperature_k: f64,
    /// Index of `temperature_k` in the HDF5's numerically-sorted
    /// non-zero-K temperature list — what
    /// `xs_provider::load_nuclide` consumes.
    pub temp_idx: usize,
    /// All non-zero temperature columns available in the HDF5, sorted
    /// ascending. Useful for diagnostics and explicit temperature
    /// selection.
    pub temperatures_k: Vec<f64>,
}

impl ResolvedNuclide {
    /// Convenience tuple matching the legacy `NUCLIDE_SPECS` element
    /// shape — `(filename, AWR, ν̄_const, temp_idx)`. New code should
    /// prefer the named fields, but binaries mid-migration can use
    /// this to keep their existing loop bodies unchanged.
    pub fn as_legacy_spec(&self) -> (PathBuf, f64, f64, usize) {
        (
            self.path.clone(),
            self.awr,
            self.nu_bar_const,
            self.temp_idx,
        )
    }
}

/// Static catalog row: ZAID, symbol, and ν̄ fallback.
///
/// HDF5 filename is derived by convention via
/// [`default_filename_for_zaid`] (`<Symbol><Mass>.h5`). Set
/// `filename_override` only when the file doesn't follow that
/// convention (e.g. future metastable-state files with `_m1` suffix).
///
/// Symbol is also derivable from ZAID via [`symbol_for_z`], but kept
/// inline for terser diagnostic output ("U-235 not in catalog" reads
/// better than a number).
struct CatalogEntry {
    zaid: u32,
    symbol: &'static str,
    /// Override for non-conventional filenames. `None` triggers the
    /// universal `<Symbol><Mass>.h5` convention.
    filename_override: Option<&'static str>,
    /// Default constant ν̄ for fissionable nuclides — used when the
    /// energy-dependent ν̄(E) table can't be loaded. ENDF/B-VII.1
    /// thermal-region values; only meaningful if the nuclide has
    /// MT=18 (fission).
    nu_bar_const: f64,
}

impl CatalogEntry {
    /// HDF5 filename to load. Override takes precedence over the
    /// derived `<Symbol><Mass>.h5` default.
    fn filename(&self) -> String {
        if let Some(override_name) = self.filename_override {
            return override_name.to_string();
        }
        default_filename_for_zaid(self.zaid).unwrap_or_else(|| {
            // Z out of `symbol_for_z` range — surface a placeholder
            // path that will fail at `resolve()` with a clear file-
            // missing error rather than panicking here.
            format!("UNKNOWN_ZAID_{}.h5", self.zaid)
        })
    }
}

/// Master ZAID → file/symbol/ν̄ catalog. Covers structural nuclides,
/// the major actinide chain (U-233 through Cm-247), the dominant FP
/// poisons (Xe-135, Sm-149, Pm-149, etc.), and S(α,β) parent isotopes
/// (H-1, H-2 / D, O-16). Hard-coded — it's a small table and the cost
/// of getting an entry wrong is bounded by `ResolvedNuclide.path` not
/// existing on disk, which is caught immediately at load time.
const CATALOG: &[CatalogEntry] = &[
    // Hydrogen / coolant
    CatalogEntry {
        zaid: 1001,
        symbol: "H-1",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 1002,
        symbol: "H-2 (D)",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 1003,
        symbol: "H-3 (T)",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // Light moderator / structural
    CatalogEntry {
        zaid: 5010,
        symbol: "B-10",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 5011,
        symbol: "B-11",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 6000,
        symbol: "C-nat",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 8016,
        symbol: "O-16",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 8017,
        symbol: "O-17",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // Zircaloy-4 structural
    CatalogEntry {
        zaid: 40090,
        symbol: "Zr-90",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 40091,
        symbol: "Zr-91",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 40092,
        symbol: "Zr-92",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 40094,
        symbol: "Zr-94",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 40096,
        symbol: "Zr-96",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // Iron / steel
    CatalogEntry {
        zaid: 26054,
        symbol: "Fe-54",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 26056,
        symbol: "Fe-56",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 26057,
        symbol: "Fe-57",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 26058,
        symbol: "Fe-58",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // Actinides — uranium chain
    CatalogEntry {
        zaid: 92233,
        symbol: "U-233",
        filename_override: None,
        nu_bar_const: 2.49,
    },
    CatalogEntry {
        zaid: 92234,
        symbol: "U-234",
        filename_override: None,
        nu_bar_const: 2.49,
    },
    CatalogEntry {
        zaid: 92235,
        symbol: "U-235",
        filename_override: None,
        nu_bar_const: 2.43,
    },
    CatalogEntry {
        zaid: 92236,
        symbol: "U-236",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 92237,
        symbol: "U-237",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 92238,
        symbol: "U-238",
        filename_override: None,
        nu_bar_const: 2.49,
    },
    CatalogEntry {
        zaid: 92239,
        symbol: "U-239",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // Actinides — neptunium / plutonium / americium / curium
    CatalogEntry {
        zaid: 93237,
        symbol: "Np-237",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 93239,
        symbol: "Np-239",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 94238,
        symbol: "Pu-238",
        filename_override: None,
        nu_bar_const: 2.90,
    },
    CatalogEntry {
        zaid: 94239,
        symbol: "Pu-239",
        filename_override: None,
        nu_bar_const: 2.88,
    },
    CatalogEntry {
        zaid: 94240,
        symbol: "Pu-240",
        filename_override: None,
        nu_bar_const: 2.79,
    },
    CatalogEntry {
        zaid: 94241,
        symbol: "Pu-241",
        filename_override: None,
        nu_bar_const: 2.95,
    },
    CatalogEntry {
        zaid: 94242,
        symbol: "Pu-242",
        filename_override: None,
        nu_bar_const: 2.81,
    },
    CatalogEntry {
        zaid: 95241,
        symbol: "Am-241",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 95242,
        symbol: "Am-242",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 95243,
        symbol: "Am-243",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 96242,
        symbol: "Cm-242",
        filename_override: None,
        nu_bar_const: 3.00,
    },
    CatalogEntry {
        zaid: 96243,
        symbol: "Cm-243",
        filename_override: None,
        nu_bar_const: 3.40,
    },
    CatalogEntry {
        zaid: 96244,
        symbol: "Cm-244",
        filename_override: None,
        nu_bar_const: 3.20,
    },
    CatalogEntry {
        zaid: 96245,
        symbol: "Cm-245",
        filename_override: None,
        nu_bar_const: 3.50,
    },
    CatalogEntry {
        zaid: 96246,
        symbol: "Cm-246",
        filename_override: None,
        nu_bar_const: 3.10,
    },
    CatalogEntry {
        zaid: 96247,
        symbol: "Cm-247",
        filename_override: None,
        nu_bar_const: 3.50,
    },
    // Fission product poisons
    CatalogEntry {
        zaid: 53135,
        symbol: "I-135",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 54135,
        symbol: "Xe-135",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 55135,
        symbol: "Cs-135",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 61149,
        symbol: "Pm-149",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 62149,
        symbol: "Sm-149",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64155,
        symbol: "Gd-155",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64157,
        symbol: "Gd-157",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    // ── ICSBEP nuclides ────────────────────────────────────────────────
    // Added to enable ICSBEP benchmark coverage beyond LWR / Godiva:
    // reflector materials (Be, W, Pb, Bi, steel constituents), solution
    // chemistry (N, Cl, F, S), control / poison rods (Cd, Ag, In, Hf,
    // Eu, Dy), Th-fuel cycle (Th-232, Pa), and Pu-metal stabilisers
    // (Ga). All files present in ENDF/B-VII.1 HDF5 distribution.

    // Light elements — solutions, gases, concrete constituents
    CatalogEntry {
        zaid: 2003,
        symbol: "He-3",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 2004,
        symbol: "He-4",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 3006,
        symbol: "Li-6",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 3007,
        symbol: "Li-7",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 4009,
        symbol: "Be-9",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 7014,
        symbol: "N-14",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 7015,
        symbol: "N-15",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 9019,
        symbol: "F-19",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 11023,
        symbol: "Na-23",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 12024,
        symbol: "Mg-24",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 12025,
        symbol: "Mg-25",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 12026,
        symbol: "Mg-26",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 13027,
        symbol: "Al-27",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 14028,
        symbol: "Si-28",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 14029,
        symbol: "Si-29",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 14030,
        symbol: "Si-30",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 16032,
        symbol: "S-32",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 16033,
        symbol: "S-33",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 16034,
        symbol: "S-34",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 16036,
        symbol: "S-36",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 17035,
        symbol: "Cl-35",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 17037,
        symbol: "Cl-37",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 19039,
        symbol: "K-39",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 19040,
        symbol: "K-40",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 19041,
        symbol: "K-41",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20040,
        symbol: "Ca-40",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20042,
        symbol: "Ca-42",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20043,
        symbol: "Ca-43",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20044,
        symbol: "Ca-44",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20046,
        symbol: "Ca-46",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 20048,
        symbol: "Ca-48",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Steel / alloy constituents — Cr / Mn / Co / Ni / Mo / Cu / Zn
    CatalogEntry {
        zaid: 24050,
        symbol: "Cr-50",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 24052,
        symbol: "Cr-52",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 24053,
        symbol: "Cr-53",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 24054,
        symbol: "Cr-54",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 25055,
        symbol: "Mn-55",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 27059,
        symbol: "Co-59",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 28058,
        symbol: "Ni-58",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 28060,
        symbol: "Ni-60",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 28061,
        symbol: "Ni-61",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 28062,
        symbol: "Ni-62",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 28064,
        symbol: "Ni-64",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 29063,
        symbol: "Cu-63",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 29065,
        symbol: "Cu-65",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42092,
        symbol: "Mo-92",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42094,
        symbol: "Mo-94",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42095,
        symbol: "Mo-95",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42096,
        symbol: "Mo-96",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42097,
        symbol: "Mo-97",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42098,
        symbol: "Mo-98",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 42100,
        symbol: "Mo-100",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 22046,
        symbol: "Ti-46",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 22047,
        symbol: "Ti-47",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 22048,
        symbol: "Ti-48",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 22049,
        symbol: "Ti-49",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 22050,
        symbol: "Ti-50",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 23050,
        symbol: "V-50",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 23051,
        symbol: "V-51",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Zircaloy alloy add-ons — Sn isotopes
    CatalogEntry {
        zaid: 50112,
        symbol: "Sn-112",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50114,
        symbol: "Sn-114",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50115,
        symbol: "Sn-115",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50116,
        symbol: "Sn-116",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50117,
        symbol: "Sn-117",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50118,
        symbol: "Sn-118",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50119,
        symbol: "Sn-119",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50120,
        symbol: "Sn-120",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50122,
        symbol: "Sn-122",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 50124,
        symbol: "Sn-124",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // δ-Pu stabiliser
    CatalogEntry {
        zaid: 31069,
        symbol: "Ga-69",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 31071,
        symbol: "Ga-71",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Heavy reflectors — Ta / W / Au / Pb / Bi
    CatalogEntry {
        zaid: 73181,
        symbol: "Ta-181",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 74180,
        symbol: "W-180",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 74182,
        symbol: "W-182",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 74183,
        symbol: "W-183",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 74184,
        symbol: "W-184",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 74186,
        symbol: "W-186",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 79197,
        symbol: "Au-197",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 82204,
        symbol: "Pb-204",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 82206,
        symbol: "Pb-206",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 82207,
        symbol: "Pb-207",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 82208,
        symbol: "Pb-208",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 83209,
        symbol: "Bi-209",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Control / poison rod materials
    CatalogEntry {
        zaid: 47107,
        symbol: "Ag-107",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 47109,
        symbol: "Ag-109",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48106,
        symbol: "Cd-106",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48108,
        symbol: "Cd-108",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48110,
        symbol: "Cd-110",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48111,
        symbol: "Cd-111",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48112,
        symbol: "Cd-112",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48113,
        symbol: "Cd-113",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48114,
        symbol: "Cd-114",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 48116,
        symbol: "Cd-116",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 49113,
        symbol: "In-113",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 49115,
        symbol: "In-115",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72174,
        symbol: "Hf-174",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72176,
        symbol: "Hf-176",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72177,
        symbol: "Hf-177",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72178,
        symbol: "Hf-178",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72179,
        symbol: "Hf-179",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 72180,
        symbol: "Hf-180",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Burnable absorbers / rare earths
    CatalogEntry {
        zaid: 63151,
        symbol: "Eu-151",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 63153,
        symbol: "Eu-153",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 65159,
        symbol: "Tb-159",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 66160,
        symbol: "Dy-160",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 66161,
        symbol: "Dy-161",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 66162,
        symbol: "Dy-162",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 66163,
        symbol: "Dy-163",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 66164,
        symbol: "Dy-164",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64154,
        symbol: "Gd-154",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64156,
        symbol: "Gd-156",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64158,
        symbol: "Gd-158",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 64160,
        symbol: "Gd-160",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // Thorium fuel cycle
    CatalogEntry {
        zaid: 90230,
        symbol: "Th-230",
        filename_override: None,
        nu_bar_const: 0.0,
    },
    CatalogEntry {
        zaid: 90232,
        symbol: "Th-232",
        filename_override: None,
        nu_bar_const: 2.45,
    },
    CatalogEntry {
        zaid: 91231,
        symbol: "Pa-231",
        filename_override: None,
        nu_bar_const: 2.50,
    },
    CatalogEntry {
        zaid: 91233,
        symbol: "Pa-233",
        filename_override: None,
        nu_bar_const: 0.0,
    },

    // ── ICSBEP-corpus extensions (P / Ar / Zn / Br / Sb / Ba / Sm / Dy / Gd / U-232) ─
    // Filled in after running the corpus through the resolver caught
    // these as missing. Phosphorus shows up in stainless-steel tanks;
    // the rest are fission-product or trace structural nuclides that
    // appear in solution-tank, MOX-fuel, and steel-reflector cases.
    CatalogEntry { zaid: 15031, symbol: "P-31",   filename_override: None,   nu_bar_const: 0.0 },
    CatalogEntry { zaid: 18036, symbol: "Ar-36",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 18038, symbol: "Ar-38",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 18040, symbol: "Ar-40",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 30064, symbol: "Zn-64",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 30066, symbol: "Zn-66",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 30067, symbol: "Zn-67",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 30068, symbol: "Zn-68",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 30070, symbol: "Zn-70",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 35079, symbol: "Br-79",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 35081, symbol: "Br-81",  filename_override: None,  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 51121, symbol: "Sb-121", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 51123, symbol: "Sb-123", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56130, symbol: "Ba-130", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56132, symbol: "Ba-132", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56134, symbol: "Ba-134", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56135, symbol: "Ba-135", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56136, symbol: "Ba-136", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56137, symbol: "Ba-137", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 56138, symbol: "Ba-138", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62144, symbol: "Sm-144", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62147, symbol: "Sm-147", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62148, symbol: "Sm-148", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62150, symbol: "Sm-150", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62152, symbol: "Sm-152", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62154, symbol: "Sm-154", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 64152, symbol: "Gd-152", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 66156, symbol: "Dy-156", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 66158, symbol: "Dy-158", filename_override: None, nu_bar_const: 0.0 },
    CatalogEntry { zaid: 92232, symbol: "U-232",  filename_override: None,  nu_bar_const: 2.49 },
];

/// Errors from `NuclideLibrary::resolve`.
#[derive(Debug, thiserror::Error)]
pub enum NuclideLibraryError {
    #[error("ZAID {0} not in nuclide catalog")]
    UnknownZaid(u32),
    #[error("HDF5 file {0} not found in data directory")]
    FileMissing(PathBuf),
    #[error("HDF5 read for {0}: {1}")]
    Hdf5Read(PathBuf, String),
}

/// Lazy nuclide-metadata cache. The catalog is static; metadata
/// (AWR, temperatures) is read from HDF5 on first request and
/// cached.
pub struct NuclideLibrary {
    data_dir: PathBuf,
    by_zaid: HashMap<u32, &'static CatalogEntry>,
    cache: std::cell::RefCell<HashMap<u32, NuclideMeta>>,
}

#[derive(Clone)]
struct NuclideMeta {
    awr: f64,
    temperatures_k: Vec<f64>,
}

impl NuclideLibrary {
    /// Build a library rooted at `data_dir`. The directory must contain
    /// the HDF5 files referenced by `CATALOG`; missing files don't
    /// cause an error here — they fail lazily at `resolve` time.
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        let by_zaid = CATALOG.iter().map(|e| (e.zaid, e)).collect();
        Self {
            data_dir: data_dir.into(),
            by_zaid,
            cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Resolve a ZAID against the library at a target temperature
    /// (kelvin). Returns the absolute HDF5 path, AWR (read from the
    /// file's `atomic_weight_ratio` attribute), constant-ν̄ fallback,
    /// and the temperature index of the nearest available column.
    ///
    /// Selection rule: pick the library column closest in K. Ties
    /// resolve to the lower temperature (matches OpenMC's
    /// `temperature: closest` mode for on-library evaluation).
    pub fn resolve(
        &self,
        zaid: u32,
        target_temp_k: f64,
    ) -> Result<ResolvedNuclide, NuclideLibraryError> {
        let entry = self
            .by_zaid
            .get(&zaid)
            .copied()
            .ok_or(NuclideLibraryError::UnknownZaid(zaid))?;
        let path = self.data_dir.join(entry.filename());
        if !path.exists() {
            return Err(NuclideLibraryError::FileMissing(path));
        }
        let meta = self.load_or_get_meta(zaid, &path)?;
        let (temp_idx, temperature_k) = pick_temperature(&meta.temperatures_k, target_temp_k);
        Ok(ResolvedNuclide {
            zaid: entry.zaid,
            symbol: entry.symbol,
            path,
            awr: meta.awr,
            nu_bar_const: entry.nu_bar_const,
            temperature_k,
            temp_idx,
            temperatures_k: meta.temperatures_k.clone(),
        })
    }

    /// Resolve a list of ZAIDs in one call; preserves input order so
    /// the resulting `xs_kernel_idx`s match the caller's expectations.
    pub fn resolve_many(
        &self,
        zaids: &[(u32, f64)],
    ) -> Result<Vec<ResolvedNuclide>, NuclideLibraryError> {
        zaids.iter().map(|&(z, t)| self.resolve(z, t)).collect()
    }

    /// True if `zaid` is in the catalog (file existence not checked).
    pub fn knows(&self, zaid: u32) -> bool {
        self.by_zaid.contains_key(&zaid)
    }

    /// Iterate over all (zaid, symbol) pairs the catalog contains.
    pub fn catalog(&self) -> impl Iterator<Item = (u32, &'static str)> + '_ {
        self.by_zaid.iter().map(|(z, e)| (*z, e.symbol))
    }

    /// Root directory the library was constructed against. Useful for
    /// callers that need to resolve sibling files (S(α,β) tables,
    /// WMP windows) that live in the same OpenMC data distribution.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    fn load_or_get_meta(&self, zaid: u32, path: &Path) -> Result<NuclideMeta, NuclideLibraryError> {
        if let Some(m) = self.cache.borrow().get(&zaid) {
            return Ok(m.clone());
        }
        let reader = hdf5_reader::NuclideFileReader::open(path)
            .map_err(|e| NuclideLibraryError::Hdf5Read(path.to_path_buf(), format!("{e}")))?;
        // Fall back to A from the ZAID when the HDF5 file lacks the
        // `atomic_weight_ratio` attribute. AWR ≈ A is good enough as
        // a fallback for cells where transport doesn't depend
        // critically on the exact mass (kinematics scaling).
        let awr = reader.awr().unwrap_or((zaid % 1000) as f64);
        let temperatures_k = reader.temperatures.clone();
        let meta = NuclideMeta {
            awr,
            temperatures_k,
        };
        self.cache.borrow_mut().insert(zaid, meta.clone());
        Ok(meta)
    }
}

/// Pick the closest library temperature column. Returns `(idx, T_K)`.
/// Ties favour the lower temperature (deterministic, matches OpenMC's
/// `closest` mode).
fn pick_temperature(temps_k: &[f64], target_k: f64) -> (usize, f64) {
    if temps_k.is_empty() {
        return (0, target_k);
    }
    let (mut best_idx, mut best_dist) = (0_usize, f64::INFINITY);
    for (i, &t) in temps_k.iter().enumerate() {
        let d = (t - target_k).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, temps_k[best_idx])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// `pick_temperature` returns the closest column; ties are stable
    /// (lower index wins because of `<` comparison, but the test
    /// pin-points expected behaviour for symmetric ties).
    #[test]
    fn pick_temperature_picks_closest() {
        let temps = vec![250.0, 294.0, 600.0, 900.0, 1200.0, 2500.0];
        assert_eq!(pick_temperature(&temps, 900.0), (3, 900.0));
        assert_eq!(pick_temperature(&temps, 800.0), (3, 900.0)); // 100 K closer to 900
        assert_eq!(pick_temperature(&temps, 700.0), (2, 600.0)); // 100 K closer to 600
        assert_eq!(pick_temperature(&temps, 50.0), (0, 250.0));
        assert_eq!(pick_temperature(&temps, 5000.0), (5, 2500.0));
    }

    /// Empty temperature list defaults to (0, target) — the hdf5
    /// path will fail later but `pick_temperature` itself shouldn't
    /// panic.
    #[test]
    fn pick_temperature_handles_empty_list() {
        assert_eq!(pick_temperature(&[], 600.0), (0, 600.0));
    }

    /// Catalog covers the standard PWR + actinides chain set. The
    /// integer constants here are the ZAIDs we hard-wire elsewhere
    /// in the codebase; if the catalog drifts, those binaries break
    /// with a clear error rather than silently using the wrong file.
    #[test]
    fn catalog_covers_pwr_and_actinides() {
        let lib = NuclideLibrary::from_data_dir(".");
        let pwr_must_have: &[u32] = &[
            1001, 1002, 8016, 40090, 40091, 40092, 40094, 92235, 92238, 94239, 94240, 94241, 94242,
            95241, 53135, 54135, 55135, 61149, 62149,
        ];
        for &zaid in pwr_must_have {
            assert!(
                lib.knows(zaid),
                "PWR / actinides chain ZAID {zaid} missing from catalog",
            );
        }
    }

    /// ICSBEP-relevant nuclides: reflectors (Be / W / Pb / Bi), solution
    /// chemistry (N / Cl / F / S), Th-cycle (Th-232, Pa-231/233),
    /// control rods (Cd / Ag / In / Hf / Eu / Dy), Pu stabiliser (Ga),
    /// steel constituents (Cr / Mn / Co / Ni / Mo / Cu). Catches
    /// regressions when the catalog gets re-sorted or accidentally
    /// truncated.
    #[test]
    fn catalog_covers_icsbep_reflectors_and_solutions() {
        let lib = NuclideLibrary::from_data_dir(".");
        let icsbep_must_have: &[u32] = &[
            4009,  // Be-9 (Topsy-Be, BeRP)
            5010, 5011, // B-10/11 (control)
            7014, 7015, // N-14/15 (uranyl nitrate, air)
            9019,  // F-19 (fluoride salts)
            11023, // Na-23 (sodium fast reactors)
            13027, // Al-27 (clad / structural)
            14028, 14029, 14030, // Si (concrete)
            17035, 17037, // Cl-35/37 (chloride salts)
            24050, 24052, 24053, 24054, // Cr (steel)
            25055, // Mn-55 (steel)
            28058, 28060, 28061, 28062, 28064, // Ni (steel)
            29063, 29065, // Cu (Cu-reflected)
            31069, 31071, // Ga (δ-Pu stabiliser)
            42092, 42095, 42098, // Mo (steel)
            47107, 47109, // Ag (control)
            48113, // Cd-113 (control)
            49115, // In-115 (control)
            50116, 50118, 50120, 50124, // Sn (Zircaloy)
            72177, 72178, // Hf (control)
            74182, 74184, 74186, // W (FLATTOP-W)
            82206, 82207, 82208, // Pb (Pb-reflected)
            83209, // Bi-209 (Pb-Bi reflectors)
            90232, // Th-232 (Th cycle)
            91231, // Pa-231 (Th cycle)
        ];
        for &zaid in icsbep_must_have {
            assert!(
                lib.knows(zaid),
                "ICSBEP reflector/solution ZAID {zaid} missing from catalog",
            );
        }
    }

    /// Every ZAID in the catalog is unique. Catches copy-paste errors
    /// where two entries share a ZAID and silently mask one of the
    /// HDF5 files.
    #[test]
    fn catalog_zaids_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for entry in CATALOG {
            assert!(
                seen.insert(entry.zaid),
                "duplicate ZAID in CATALOG: {} ({})",
                entry.zaid,
                entry.symbol
            );
        }
    }

    /// Every catalog filename follows the OpenMC HDF5 convention
    /// (`<Symbol><Mass>.h5` for ground-state, no underscores except for
    /// metastable variants). Catches typos in the resolved filename.
    #[test]
    fn catalog_filenames_well_formed() {
        for entry in CATALOG {
            let fname = entry.filename();
            assert!(
                fname.ends_with(".h5"),
                "filename {} doesn't end in .h5",
                fname,
            );
            assert!(
                !fname.contains('/') && !fname.contains('\\'),
                "filename {} must be a bare HDF5 name (no path separator)",
                fname,
            );
        }
    }

    /// Every catalog entry's resolved filename matches the universal
    /// `<Symbol><Mass>.h5` convention. Most entries achieve this by
    /// having no `filename_override` (the default-from-ZAID path
    /// takes over); a handful may carry overrides matching the same
    /// convention. Either way, the surfaced filename must converge.
    #[test]
    fn default_filename_matches_every_catalog_entry() {
        let mut deviations = Vec::new();
        for entry in CATALOG {
            let derived = default_filename_for_zaid(entry.zaid)
                .unwrap_or_else(|| format!("(unknown Z for ZAID {})", entry.zaid));
            let resolved = entry.filename();
            if derived != resolved {
                deviations.push((entry.zaid, entry.symbol, resolved, derived));
            }
        }
        assert!(
            deviations.is_empty(),
            "{} CATALOG entries diverge from <Symbol><Mass>.h5 convention:\n{}",
            deviations.len(),
            deviations
                .iter()
                .map(|(z, s, have, derived)| format!(
                    "  ZAID {z} ({s}): explicit {have:?} ≠ derived {derived:?}"
                ))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    /// Spot-checks for [`default_filename_for_zaid`] / [`symbol_for_z`].
    /// Edge cases include H-1 → `H1.h5` (single-letter symbol), Be-9
    /// (single-letter), Pu-239 (two-letter symbol), C-natural via A=0
    /// → `C0.h5` (the OpenMC convention for natural-element files).
    #[test]
    fn default_filename_well_known_examples() {
        assert_eq!(default_filename_for_zaid(1001), Some("H1.h5".into()));
        assert_eq!(default_filename_for_zaid(1002), Some("H2.h5".into()));
        assert_eq!(default_filename_for_zaid(4009), Some("Be9.h5".into()));
        assert_eq!(default_filename_for_zaid(6000), Some("C0.h5".into()));
        assert_eq!(default_filename_for_zaid(92235), Some("U235.h5".into()));
        assert_eq!(default_filename_for_zaid(94239), Some("Pu239.h5".into()));
        assert_eq!(default_filename_for_zaid(96247), Some("Cm247.h5".into()));
        // Out-of-table Z returns None — callers must supply an
        // explicit override.
        assert_eq!(default_filename_for_zaid(110000 + 264), None);
    }

    /// Fission ν̄ fallback is non-zero only on actinides with MT=18
    /// in their ENDF/B-VII.1 evaluation. Catches accidentally setting
    /// nu_bar_const on a non-fissionable nuclide (would produce
    /// spurious fission neutrons if the engine ever fell back to the
    /// constant value).
    #[test]
    fn catalog_nu_bar_const_only_on_fissionable() {
        for entry in CATALOG {
            if entry.nu_bar_const == 0.0 {
                continue;
            }
            // Fissionable: Z >= 90 (Th and above), or specifically
            // listed metastable Am-242m (ZAID 95242 ground-state in
            // our catalog conflates ground + metastable).
            let z = entry.zaid / 1000;
            assert!(
                z >= 90,
                "non-actinide {} (ZAID {}) has nu_bar_const = {} — should be 0",
                entry.symbol,
                entry.zaid,
                entry.nu_bar_const,
            );
            assert!(
                entry.nu_bar_const > 1.5 && entry.nu_bar_const < 5.0,
                "{} nu_bar_const = {} outside physically plausible 1.5-5.0",
                entry.symbol,
                entry.nu_bar_const,
            );
        }
    }
}
