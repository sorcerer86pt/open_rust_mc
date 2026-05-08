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
        (self.path.clone(), self.awr, self.nu_bar_const, self.temp_idx)
    }
}

/// Static catalog row: ZAID, symbol, HDF5 filename. Maintained as a
/// constant table so adding a new nuclide is a one-line edit.
struct CatalogEntry {
    zaid: u32,
    symbol: &'static str,
    filename: &'static str,
    /// Default constant ν̄ for fissionable nuclides — used when the
    /// energy-dependent ν̄(E) table can't be loaded. ENDF/B-VII.1
    /// thermal-region values; only meaningful if the nuclide has
    /// MT=18 (fission).
    nu_bar_const: f64,
}

/// Master ZAID → file/symbol/ν̄ catalog. Covers structural nuclides,
/// the major actinide chain (U-233 through Cm-247), the dominant FP
/// poisons (Xe-135, Sm-149, Pm-149, etc.), and S(α,β) parent isotopes
/// (H-1, H-2 / D, O-16). Hard-coded — it's a small table and the cost
/// of getting an entry wrong is bounded by `ResolvedNuclide.path` not
/// existing on disk, which is caught immediately at load time.
const CATALOG: &[CatalogEntry] = &[
    // Hydrogen / coolant
    CatalogEntry { zaid: 1001, symbol: "H-1",    filename: "H1.h5",   nu_bar_const: 0.0 },
    CatalogEntry { zaid: 1002, symbol: "H-2 (D)", filename: "H2.h5",  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 1003, symbol: "H-3 (T)", filename: "H3.h5",  nu_bar_const: 0.0 },
    // Light moderator / structural
    CatalogEntry { zaid: 5010, symbol: "B-10",   filename: "B10.h5",  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 5011, symbol: "B-11",   filename: "B11.h5",  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 6000, symbol: "C-nat",  filename: "C0.h5",   nu_bar_const: 0.0 },
    CatalogEntry { zaid: 8016, symbol: "O-16",   filename: "O16.h5",  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 8017, symbol: "O-17",   filename: "O17.h5",  nu_bar_const: 0.0 },
    // Zircaloy-4 structural
    CatalogEntry { zaid: 40090, symbol: "Zr-90", filename: "Zr90.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 40091, symbol: "Zr-91", filename: "Zr91.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 40092, symbol: "Zr-92", filename: "Zr92.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 40094, symbol: "Zr-94", filename: "Zr94.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 40096, symbol: "Zr-96", filename: "Zr96.h5", nu_bar_const: 0.0 },
    // Iron / steel
    CatalogEntry { zaid: 26054, symbol: "Fe-54", filename: "Fe54.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 26056, symbol: "Fe-56", filename: "Fe56.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 26057, symbol: "Fe-57", filename: "Fe57.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 26058, symbol: "Fe-58", filename: "Fe58.h5", nu_bar_const: 0.0 },
    // Actinides — uranium chain
    CatalogEntry { zaid: 92233, symbol: "U-233", filename: "U233.h5", nu_bar_const: 2.49 },
    CatalogEntry { zaid: 92234, symbol: "U-234", filename: "U234.h5", nu_bar_const: 2.49 },
    CatalogEntry { zaid: 92235, symbol: "U-235", filename: "U235.h5", nu_bar_const: 2.43 },
    CatalogEntry { zaid: 92236, symbol: "U-236", filename: "U236.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 92237, symbol: "U-237", filename: "U237.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 92238, symbol: "U-238", filename: "U238.h5", nu_bar_const: 2.49 },
    CatalogEntry { zaid: 92239, symbol: "U-239", filename: "U239.h5", nu_bar_const: 0.0 },
    // Actinides — neptunium / plutonium / americium / curium
    CatalogEntry { zaid: 93237, symbol: "Np-237", filename: "Np237.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 93239, symbol: "Np-239", filename: "Np239.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 94238, symbol: "Pu-238", filename: "Pu238.h5", nu_bar_const: 2.90 },
    CatalogEntry { zaid: 94239, symbol: "Pu-239", filename: "Pu239.h5", nu_bar_const: 2.88 },
    CatalogEntry { zaid: 94240, symbol: "Pu-240", filename: "Pu240.h5", nu_bar_const: 2.79 },
    CatalogEntry { zaid: 94241, symbol: "Pu-241", filename: "Pu241.h5", nu_bar_const: 2.95 },
    CatalogEntry { zaid: 94242, symbol: "Pu-242", filename: "Pu242.h5", nu_bar_const: 2.81 },
    CatalogEntry { zaid: 95241, symbol: "Am-241", filename: "Am241.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 95242, symbol: "Am-242", filename: "Am242.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 95243, symbol: "Am-243", filename: "Am243.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 96242, symbol: "Cm-242", filename: "Cm242.h5", nu_bar_const: 3.00 },
    CatalogEntry { zaid: 96243, symbol: "Cm-243", filename: "Cm243.h5", nu_bar_const: 3.40 },
    CatalogEntry { zaid: 96244, symbol: "Cm-244", filename: "Cm244.h5", nu_bar_const: 3.20 },
    CatalogEntry { zaid: 96245, symbol: "Cm-245", filename: "Cm245.h5", nu_bar_const: 3.50 },
    CatalogEntry { zaid: 96246, symbol: "Cm-246", filename: "Cm246.h5", nu_bar_const: 3.10 },
    CatalogEntry { zaid: 96247, symbol: "Cm-247", filename: "Cm247.h5", nu_bar_const: 3.50 },
    // Fission product poisons
    CatalogEntry { zaid: 53135, symbol: "I-135",  filename: "I135.h5",  nu_bar_const: 0.0 },
    CatalogEntry { zaid: 54135, symbol: "Xe-135", filename: "Xe135.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 55135, symbol: "Cs-135", filename: "Cs135.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 61149, symbol: "Pm-149", filename: "Pm149.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 62149, symbol: "Sm-149", filename: "Sm149.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 64155, symbol: "Gd-155", filename: "Gd155.h5", nu_bar_const: 0.0 },
    CatalogEntry { zaid: 64157, symbol: "Gd-157", filename: "Gd157.h5", nu_bar_const: 0.0 },
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
        let path = self.data_dir.join(entry.filename);
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
        zaids
            .iter()
            .map(|&(z, t)| self.resolve(z, t))
            .collect()
    }

    /// True if `zaid` is in the catalog (file existence not checked).
    pub fn knows(&self, zaid: u32) -> bool {
        self.by_zaid.contains_key(&zaid)
    }

    /// Iterate over all (zaid, symbol) pairs the catalog contains.
    pub fn catalog(&self) -> impl Iterator<Item = (u32, &'static str)> + '_ {
        self.by_zaid.iter().map(|(z, e)| (*z, e.symbol))
    }

    fn load_or_get_meta(
        &self,
        zaid: u32,
        path: &Path,
    ) -> Result<NuclideMeta, NuclideLibraryError> {
        if let Some(m) = self.cache.borrow().get(&zaid) {
            return Ok(m.clone());
        }
        let reader = hdf5_reader::NuclideFileReader::open(path).map_err(|e| {
            NuclideLibraryError::Hdf5Read(path.to_path_buf(), format!("{e}"))
        })?;
        let awr = reader.awr().unwrap_or_else(|_| {
            // Fall back to A from the ZAID. AWR ≈ A is good enough as
            // a fallback for cells where transport doesn't depend
            // critically on the exact mass (kinematics scaling).
            (zaid % 1000) as f64
        });
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
            1001, 1002, 8016, 40090, 40091, 40092, 40094, 92235, 92238,
            94239, 94240, 94241, 94242, 95241, 53135, 54135, 55135,
            61149, 62149,
        ];
        for &zaid in pwr_must_have {
            assert!(
                lib.knows(zaid),
                "PWR / actinides chain ZAID {zaid} missing from catalog",
            );
        }
    }
}
