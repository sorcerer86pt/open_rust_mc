//! Resolve `scene_io::MaterialDto` entries into engine
//! [`Material`]s + an [`SvdXsProvider`].
//!
//! Bridge between the schema-level material spec (HDF5 paths + atom
//! densities) and the engine-internal representation (atom density +
//! `xs_kernel_idx` into a global nuclide-kernels array).
//!
//! Two pieces of work happen here:
//!
//! 1. **ZAID resolution.** Each `NuclideEntryDto` either carries a
//!    `zaid` (preferred, set by the import script) or an `hdf5_file`
//!    path (per schema). Both forms are mapped through
//!    [`NuclideLibrary`] to an absolute HDF5 path + AWR + temperature
//!    index.
//!
//! 2. **Deduplication.** When multiple materials reference the same
//!    nuclide at the same library temperature, the kernel is loaded
//!    once and shared (a HashMap keys on `(zaid, temp_idx)`). Different
//!    temperatures produce distinct kernels.
//!
//! Scope of this round: SVD provider only. The matching Table / Hybrid
//! / WMP providers can layer in the same way once we need them for
//! validation cases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::geometry::scene_io::{MaterialDto, NuclideEntryDto};
use crate::hdf5_reader;
use crate::thermal::ThermalScatteringData;
use crate::transport::material::Material;
use crate::transport::nuclides::{NuclideLibrary, NuclideLibraryError, ResolvedNuclide};
use crate::transport::xs_provider::{self, NuclideKernels, SvdXsProvider};

/// Result of resolving a `Vec<MaterialDto>` into runnable engine
/// material + XS provider.
pub struct ResolvedMaterials {
    pub provider: SvdXsProvider,
    pub materials: Vec<Material>,
}

impl ResolvedMaterials {
    /// Per-material fissionability flags indexed by material idx.
    /// A material is fissionable iff at least one of its nuclides has
    /// a positive constant ν̄ in the loaded XS data (the loader keeps
    /// `nu_bar_const = 0.0` for non-fissionable nuclides). Mirrors
    /// Serpent's `is_fissile` predicate (`src/findfismat.c`).
    ///
    /// Used by `simulate::try_initial_source` and the Python /
    /// ICSBEP harness to decide which cells can host the first-batch
    /// source — replaces the historical "first Material cell" /
    /// "smallest-volume material" heuristics, which broke on
    /// reflected-metal, multi-shell, BWR-cruciform, PWR-burnable-
    /// poison, and HFIR-plate geometries.
    pub fn fissionable_materials(&self) -> Vec<bool> {
        self.materials
            .iter()
            .map(|m| {
                m.nuclides.iter().any(|n| {
                    self.provider
                        .nuclides
                        .get(n.xs_kernel_idx)
                        .map(|nk| nk.nu_bar_const > 0.0)
                        .unwrap_or(false)
                })
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("material {material} nuclide {nuclide_idx}: neither `zaid` nor `hdf5_file` is set")]
    MissingNuclideIdentifier {
        material: String,
        nuclide_idx: usize,
    },
    #[error("material {material} nuclide {nuclide_idx}: cannot parse ZAID from hdf5_file {path:?}")]
    UnparseableHdf5Path {
        material: String,
        nuclide_idx: usize,
        path: String,
    },
    #[error("material {material} nuclide {nuclide_idx}: {source}")]
    NuclideLibrary {
        material: String,
        nuclide_idx: usize,
        #[source]
        source: NuclideLibraryError,
    },
    #[error("material {material}: S(α,β) file {path:?} — failed to load: {reason}")]
    ThermalLoad {
        material: String,
        path: String,
        reason: String,
    },
    #[error("material {material}: S(α,β) file {path:?} has unrecognised name (expected `c_<Symbol>_…` or `c_<Symbol><Mass>_…`)")]
    UnparseableThermalName {
        material: String,
        path: String,
    },
    #[error(
        "material {material}: S(α,β) file {path:?} targets element {target_symbol} (Z={target_z}) \
         but the material has no matching nuclide"
    )]
    ThermalTargetNotFound {
        material: String,
        path: String,
        target_symbol: String,
        target_z: u32,
    },
}

/// Resolve a list of [`MaterialDto`] entries into engine
/// [`Material`]s + a shared [`SvdXsProvider`].
///
/// `lib` must have file existence for every referenced HDF5 file —
/// missing files surface as `NuclideLibraryError::FileMissing` in the
/// returned error. `svd_rank` is applied uniformly across reactions
/// (use [`xs_provider::load_nuclide_with_policy`] directly for
/// per-MT-tuned rank). `data_dir` is the root directory containing
/// the OpenMC HDF5 distribution — used to resolve thermal-scattering
/// files referenced by [`MaterialDto::thermal_files`] or
/// [`NuclideEntryDto::thermal_file`] when those carry bare filenames
/// like `"c_H_in_H2O.h5"` rather than absolute paths.
pub fn resolve_materials(
    materials: &[MaterialDto],
    lib: &NuclideLibrary,
    svd_rank: usize,
) -> Result<ResolvedMaterials, ResolveError> {
    resolve_materials_with_data_dir(materials, lib, svd_rank, lib.data_dir())
}

/// Like [`resolve_materials`] but lets the caller separate the
/// nuclide-library root (where `U235.h5` etc. live) from the thermal-
/// scattering root (where `c_*.h5` live). Both default to the same
/// directory in `resolve_materials`; the OpenMC `data` distribution
/// keeps them together too.
pub fn resolve_materials_with_data_dir(
    materials: &[MaterialDto],
    lib: &NuclideLibrary,
    svd_rank: usize,
    thermal_dir: &Path,
) -> Result<ResolvedMaterials, ResolveError> {
    // (zaid, temp_idx) → kernel_idx — share kernels across materials
    // when temperature columns coincide.
    let mut kernel_idx: HashMap<(u32, usize), usize> = HashMap::new();
    let mut kernels: Vec<NuclideKernels> = Vec::new();
    let mut engine_mats: Vec<Material> = Vec::with_capacity(materials.len());

    // Track which kernel each (material_idx, nuclide_dto_idx) ended up
    // at — we need the index again when attaching per-nuclide
    // thermal_file entries.
    let mut per_material_kernel_idx: Vec<Vec<usize>> = Vec::with_capacity(materials.len());
    // (kernel_idx, thermal_filename) entries to load after the
    // kernel-loading pass. Deduplicated by canonical absolute path.
    let mut thermal_requests: Vec<(usize, PathBuf)> = Vec::new();

    // Per-nuclide policy pre-pass — identifies which (zaid, temp_idx)
    // keys will have S(α,β) thermal data attached so the kernel
    // loader can switch MT=2 to a pointwise Table for those keys.
    // The free-atom elastic XS is replaced at runtime by the thermal
    // total below `energy_max`, so the SVD basis × coeffs is dead
    // weight for moderator nuclides — Table saves both the
    // reconstruction work and the basis memory.
    let mut keys_with_thermal: std::collections::HashSet<(u32, usize)> =
        std::collections::HashSet::new();
    for mat in materials {
        // Per-nuclide thermal_file: marks just that nuclide.
        for (n_idx, nuc) in mat.nuclides.iter().enumerate() {
            if nuc.thermal_file.is_some() {
                let zaid = resolve_zaid(nuc, &mat.name, n_idx)?;
                let resolved = lib.resolve(zaid, mat.temperature).map_err(|e| {
                    ResolveError::NuclideLibrary {
                        material: mat.name.clone(),
                        nuclide_idx: n_idx,
                        source: e,
                    }
                })?;
                keys_with_thermal.insert((zaid, resolved.temp_idx));
            }
        }
        // Material-level thermal_files: match by (Z, optional A).
        for thermal_name in &mat.thermal_files {
            let (target_z, target_a) = match parse_thermal_target(thermal_name) {
                Some(t) => t,
                None => continue, // emits error later, after kernels loaded
            };
            for (n_idx, nuc) in mat.nuclides.iter().enumerate() {
                let zaid = resolve_zaid(nuc, &mat.name, n_idx)?;
                let z = zaid / 1000;
                let a = zaid % 1000;
                if z == target_z && target_a.map(|ta| a == ta).unwrap_or(true) {
                    let resolved = lib.resolve(zaid, mat.temperature).map_err(|e| {
                        ResolveError::NuclideLibrary {
                            material: mat.name.clone(),
                            nuclide_idx: n_idx,
                            source: e,
                        }
                    })?;
                    keys_with_thermal.insert((zaid, resolved.temp_idx));
                }
            }
        }
    }

    for mat in materials {
        let mut engine_mat = Material::new(&mat.name, mat.temperature);
        let mut kernel_idxs: Vec<usize> = Vec::with_capacity(mat.nuclides.len());
        for (n_idx, nuc) in mat.nuclides.iter().enumerate() {
            let zaid = resolve_zaid(nuc, &mat.name, n_idx)?;
            let resolved =
                lib.resolve(zaid, mat.temperature)
                    .map_err(|e| ResolveError::NuclideLibrary {
                        material: mat.name.clone(),
                        nuclide_idx: n_idx,
                        source: e,
                    })?;
            let key = (zaid, resolved.temp_idx);
            let idx = if let Some(&i) = kernel_idx.get(&key) {
                i
            } else {
                let kernel = load_kernel_for(
                    &resolved,
                    svd_rank,
                    keys_with_thermal.contains(&key),
                );
                let i = kernels.len();
                kernels.push(kernel);
                kernel_idx.insert(key, i);
                i
            };
            engine_mat.add_nuclide(nuc.atom_density, idx);
            kernel_idxs.push(idx);

            // Per-nuclide thermal_file — schema's canonical form.
            if let Some(thermal_name) = &nuc.thermal_file {
                let path = resolve_thermal_path(thermal_dir, thermal_name);
                thermal_requests.push((idx, path));
            }
        }
        per_material_kernel_idx.push(kernel_idxs);
        engine_mats.push(engine_mat);
    }

    // Material-level `thermal_files` — the legacy form emitted by the
    // current import script (mit-crpg `<sab name="c_H_in_H2O"/>` lives
    // at material scope). For each entry, find the nuclide in the
    // material whose element matches the thermal file's target, and
    // bind there. Multi-isotope targets (e.g. `c_Zr_in_ZrH`) bind to
    // every matching nuclide.
    for (m_idx, mat) in materials.iter().enumerate() {
        for thermal_name in &mat.thermal_files {
            let path = resolve_thermal_path(thermal_dir, thermal_name);
            let (target_z, target_a) = parse_thermal_target(thermal_name).ok_or_else(|| {
                ResolveError::UnparseableThermalName {
                    material: mat.name.clone(),
                    path: thermal_name.clone(),
                }
            })?;
            let mut matched_any = false;
            for (n_idx, nuc) in mat.nuclides.iter().enumerate() {
                let zaid = resolve_zaid(nuc, &mat.name, n_idx)?;
                let z = zaid / 1000;
                let a = zaid % 1000;
                let matches_z = z == target_z;
                let matches_a = target_a.map(|ta| a == ta).unwrap_or(true);
                if matches_z && matches_a {
                    thermal_requests.push((per_material_kernel_idx[m_idx][n_idx], path.clone()));
                    matched_any = true;
                }
            }
            if !matched_any {
                return Err(ResolveError::ThermalTargetNotFound {
                    material: mat.name.clone(),
                    path: thermal_name.clone(),
                    target_symbol: symbol_for_z(target_z).unwrap_or("?").to_string(),
                    target_z,
                });
            }
        }
    }

    // Deduplicate path loads — the same `c_H_in_H2O.h5` shared across
    // multiple materials should only hit disk once.
    let mut loaded: HashMap<PathBuf, Arc<ThermalScatteringData>> = HashMap::new();
    let mut thermal: Vec<Option<Arc<ThermalScatteringData>>> = vec![None; kernels.len()];
    for (k_idx, path) in thermal_requests {
        let arc = if let Some(arc) = loaded.get(&path) {
            Arc::clone(arc)
        } else {
            let tsl = hdf5_reader::load_thermal_scattering(&path).map_err(|e| {
                ResolveError::ThermalLoad {
                    material: String::new(),
                    path: path.display().to_string(),
                    reason: format!("{e}"),
                }
            })?;
            let arc = Arc::new(tsl);
            loaded.insert(path.clone(), Arc::clone(&arc));
            arc
        };
        thermal[k_idx] = Some(arc);
    }

    Ok(ResolvedMaterials {
        provider: SvdXsProvider {
            nuclides: kernels,
            thermal,
        },
        materials: engine_mats,
    })
}

/// Resolve a thermal-file string to an absolute path. If `name` is
/// already absolute or contains a path separator, use it as-is.
/// Otherwise join with `thermal_dir`.
fn resolve_thermal_path(thermal_dir: &Path, name: &str) -> PathBuf {
    let as_path = Path::new(name);
    if as_path.is_absolute() || name.contains('/') || name.contains('\\') {
        as_path.to_path_buf()
    } else {
        thermal_dir.join(name)
    }
}

/// Parse a thermal-scattering filename into the `(Z, optional A)` of
/// the target nuclide.
///
/// OpenMC thermal files use the convention `c_<Symbol>[_in_<compound>].h5`
/// for the natural-element form, or `c_<Symbol><Mass>.h5` for isotope-
/// specific tables. Compounds without a parseable target element
/// (`c_Graphite`, `c_C6H6`, `c_SiO2_alpha`) are mapped by hand.
///
/// Returns `(Z, Some(A))` when the file targets a specific isotope,
/// `(Z, None)` when it targets a natural element. `None` for files
/// whose target can't be determined.
pub fn parse_thermal_target(name: &str) -> Option<(u32, Option<u32>)> {
    let stem = Path::new(name).file_stem()?.to_str()?;
    let body = stem.strip_prefix("c_")?;

    // Hard-coded compound mappings — Graphite → C, BeO/UO2 inferred
    // from `c_X_in_…` form, but `c_Graphite` and `c_C6H6` lack the
    // `_in_` separator. Same for `c_SiO2_alpha` which targets both Si
    // and O — convention is to attach S(α,β) to the dominant nuclide
    // and let the caller pre-split if both need binding.
    //
    // Natural-element targets return `(Z, None)` so the attach phase
    // can bind to whichever isotope of that element is actually in
    // the material (typically H-1 / C-nat / Si — natural-abundance
    // ENDF/B handles the isotopic split). Only D and T return a
    // specific mass because their single-letter symbols denote H-2
    // and H-3 by convention.
    match body {
        "Graphite" => return Some((6, None)),         // C (natural)
        "C6H6" => return Some((1, None)),             // natural H in benzene
        "SiO2_alpha" => return Some((14, None)),      // Si (α-quartz)
        "ortho_H" | "para_H" => return Some((1, None)),
        "ortho_D" | "para_D" => return Some((1, Some(2))),
        _ => {}
    }

    // `c_<Symbol>_in_<compound>` — strip everything after `_in_`.
    let target = body.split("_in_").next().unwrap_or(body);

    // Now either `<Symbol>` (natural element) or `<Symbol><Mass>`
    // (specific isotope).
    let mut sym_end = 0;
    for (i, c) in target.char_indices() {
        if c.is_ascii_alphabetic() {
            sym_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if sym_end == 0 {
        return None;
    }
    let (sym, rest) = target.split_at(sym_end);
    // `D` and `T` are isotope glyphs for hydrogen-2 and hydrogen-3
    // — they're not in `symbol_to_z` (whose codomain is element
    // symbols). Recognise them here before falling through to the
    // generic element lookup.
    if rest.is_empty() {
        if sym == "D" {
            return Some((1, Some(2)));
        }
        if sym == "T" {
            return Some((1, Some(3)));
        }
    }
    let z = symbol_to_z(sym)?;
    if rest.is_empty() {
        Some((z, None))
    } else {
        let a: u32 = rest.parse().ok()?;
        Some((z, Some(a)))
    }
}

fn symbol_for_z(z: u32) -> Option<&'static str> {
    const TABLE: &[(u32, &str)] = &[
        (1, "H"), (2, "He"), (3, "Li"), (4, "Be"), (5, "B"), (6, "C"),
        (7, "N"), (8, "O"), (9, "F"), (11, "Na"), (12, "Mg"), (13, "Al"),
        (14, "Si"), (16, "S"), (17, "Cl"), (24, "Cr"), (26, "Fe"), (40, "Zr"),
        (47, "Ag"), (48, "Cd"), (74, "W"), (82, "Pb"), (90, "Th"), (92, "U"),
        (94, "Pu"),
    ];
    TABLE.iter().find(|(zz, _)| *zz == z).map(|(_, s)| *s)
}

fn resolve_zaid(
    nuc: &NuclideEntryDto,
    material_name: &str,
    nuc_idx: usize,
) -> Result<u32, ResolveError> {
    if let Some(z) = nuc.zaid {
        return Ok(z);
    }
    if let Some(path) = &nuc.hdf5_file {
        return zaid_from_hdf5_filename(path).ok_or_else(|| {
            ResolveError::UnparseableHdf5Path {
                material: material_name.to_string(),
                nuclide_idx: nuc_idx,
                path: path.clone(),
            }
        });
    }
    Err(ResolveError::MissingNuclideIdentifier {
        material: material_name.to_string(),
        nuclide_idx: nuc_idx,
    })
}

/// Map an OpenMC-style HDF5 filename → ZAID. `U235.h5` → 92235,
/// `Pu239.h5` → 94239, `Cd113.h5` → 48113.
///
/// Returns `None` for unrecognised filenames (thermal-scattering
/// `c_*.h5`, metastable variants `*_m1.h5`, etc.). Caller is expected
/// to provide a ZAID directly for those.
pub fn zaid_from_hdf5_filename(path: &str) -> Option<u32> {
    let stem = Path::new(path).file_stem()?.to_str()?;
    // OpenMC filenames: <Symbol><Mass>.h5
    let mut sym_end = 0;
    for (i, c) in stem.char_indices() {
        if c.is_ascii_alphabetic() {
            sym_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if sym_end == 0 || sym_end == stem.len() {
        return None;
    }
    let (sym, a_str) = stem.split_at(sym_end);
    let a: u32 = a_str.parse().ok()?;
    let z = symbol_to_z(sym)?;
    Some(1000 * z + a)
}

fn symbol_to_z(sym: &str) -> Option<u32> {
    // Compact PT lookup — covers everything in our nuclide catalog.
    Some(match sym {
        "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, "C" => 6,
        "N" => 7, "O" => 8, "F" => 9, "Ne" => 10, "Na" => 11, "Mg" => 12,
        "Al" => 13, "Si" => 14, "P" => 15, "S" => 16, "Cl" => 17, "Ar" => 18,
        "K" => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22, "V" => 23, "Cr" => 24,
        "Mn" => 25, "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30,
        "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35, "Kr" => 36,
        "Rb" => 37, "Sr" => 38, "Y" => 39, "Zr" => 40, "Nb" => 41, "Mo" => 42,
        "Tc" => 43, "Ru" => 44, "Rh" => 45, "Pd" => 46, "Ag" => 47, "Cd" => 48,
        "In" => 49, "Sn" => 50, "Sb" => 51, "Te" => 52, "I" => 53, "Xe" => 54,
        "Cs" => 55, "Ba" => 56, "La" => 57, "Ce" => 58, "Pr" => 59, "Nd" => 60,
        "Pm" => 61, "Sm" => 62, "Eu" => 63, "Gd" => 64, "Tb" => 65, "Dy" => 66,
        "Ho" => 67, "Er" => 68, "Tm" => 69, "Yb" => 70, "Lu" => 71, "Hf" => 72,
        "Ta" => 73, "W" => 74, "Re" => 75, "Os" => 76, "Ir" => 77, "Pt" => 78,
        "Au" => 79, "Hg" => 80, "Tl" => 81, "Pb" => 82, "Bi" => 83,
        "Th" => 90, "Pa" => 91, "U" => 92, "Np" => 93, "Pu" => 94,
        "Am" => 95, "Cm" => 96, "Bk" => 97, "Cf" => 98, "Es" => 99, "Fm" => 100,
        _ => return None,
    })
}

fn load_kernel_for(
    resolved: &ResolvedNuclide,
    svd_rank: usize,
    has_thermal: bool,
) -> NuclideKernels {
    let mut policy = xs_provider::RankPolicy::new(svd_rank);
    if has_thermal {
        // MT=2 free-atom elastic is replaced at runtime by the
        // S(α,β) thermal total below `energy_max`; the SVD basis
        // for MT=2 is unused below the cutoff and only consulted
        // for the small high-energy tail above it. Pointwise Table
        // costs ~`n_energy × 8 B` instead of the rank×grid SVD
        // basis (~5 MB for a thermal-attached nuclide at rank 15),
        // and the lookup collapses to one array index. Other
        // channels (capture, inelastic if any, ...) stay on the
        // configured SVD rank.
        policy = policy.with_table(2);
    }
    xs_provider::load_nuclide_with_policy(
        &resolved.path,
        &policy,
        resolved.temp_idx,
        resolved.awr,
        resolved.nu_bar_const,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn zaid_from_filename_handles_canonical_isotopes() {
        assert_eq!(zaid_from_hdf5_filename("U235.h5"), Some(92235));
        assert_eq!(zaid_from_hdf5_filename("Pu239.h5"), Some(94239));
        assert_eq!(zaid_from_hdf5_filename("H1.h5"), Some(1001));
        assert_eq!(zaid_from_hdf5_filename("Be9.h5"), Some(4009));
        assert_eq!(zaid_from_hdf5_filename("Pb208.h5"), Some(82208));
        assert_eq!(zaid_from_hdf5_filename("Th232.h5"), Some(90232));
        assert_eq!(zaid_from_hdf5_filename("Cd113.h5"), Some(48113));
    }

    #[test]
    fn zaid_from_filename_with_directory_prefix() {
        assert_eq!(
            zaid_from_hdf5_filename("data/endfb-vii.1-hdf5/U235.h5"),
            Some(92235),
        );
        assert_eq!(
            zaid_from_hdf5_filename("/some/abs/path/Pu239.h5"),
            Some(94239),
        );
    }

    #[test]
    fn zaid_from_filename_rejects_thermal_and_metastable() {
        // Thermal scattering files have a `c_` prefix and a compound
        // name — not parseable as `<Symbol><Mass>`. Resolver should
        // return None and let the caller supply ZAID explicitly.
        assert_eq!(zaid_from_hdf5_filename("c_H_in_H2O.h5"), None);
        // Metastable files have `_m1` suffix that breaks the simple
        // `<Symbol><Mass>` split. None — caller supplies ZAID.
        assert_eq!(zaid_from_hdf5_filename("Cd115_m1.h5"), None);
        assert_eq!(zaid_from_hdf5_filename("nonsense"), None);
    }

    /// ZAID-form material entry resolves through the dispatcher
    /// without needing `hdf5_file`. Exercises the canonical path used
    /// by every case the import script produces.
    #[test]
    fn resolve_zaid_form_entry() {
        let nuc = NuclideEntryDto {
            hdf5_file: None,
            zaid: Some(94239),
            label: Some("Pu-239".into()),
            atom_density: 0.037047,
            thermal_file: None,
        };
        assert_eq!(resolve_zaid(&nuc, "delta_Pu", 0).unwrap(), 94239);
    }

    /// Schema-form material entry (with `hdf5_file`) resolves through
    /// the filename parser.
    #[test]
    fn resolve_hdf5_path_entry() {
        let nuc = NuclideEntryDto {
            hdf5_file: Some("data/endfb-vii.1-hdf5/U235.h5".into()),
            zaid: None,
            label: None,
            atom_density: 0.045,
            thermal_file: None,
        };
        assert_eq!(resolve_zaid(&nuc, "HEU", 0).unwrap(), 92235);
    }

    /// Entry with neither `zaid` nor `hdf5_file` is a clear error
    /// (not a silent fallthrough).
    #[test]
    fn resolve_missing_both_fields_is_error() {
        let nuc = NuclideEntryDto {
            hdf5_file: None,
            zaid: None,
            label: None,
            atom_density: 0.04,
            thermal_file: None,
        };
        match resolve_zaid(&nuc, "broken", 7) {
            Err(ResolveError::MissingNuclideIdentifier {
                material,
                nuclide_idx,
            }) => {
                assert_eq!(material, "broken");
                assert_eq!(nuclide_idx, 7);
            }
            other => panic!("expected MissingNuclideIdentifier, got {other:?}"),
        }
    }

    /// Unparseable HDF5 path (thermal-scattering file with no plain
    /// `<Symbol><Mass>` form) bubbles up as `UnparseableHdf5Path`.
    #[test]
    fn resolve_unparseable_hdf5_path_is_error() {
        let nuc = NuclideEntryDto {
            hdf5_file: Some("c_H_in_H2O.h5".into()),
            zaid: None,
            label: None,
            atom_density: 0.067,
            thermal_file: None,
        };
        assert!(matches!(
            resolve_zaid(&nuc, "water", 0),
            Err(ResolveError::UnparseableHdf5Path { .. })
        ));
    }

    /// `c_<Symbol>_in_<compound>` → target element. Most LWR / CANDU /
    /// VHTR thermal cases use this form. Returns `(Z, None)` for
    /// natural elements — the attach phase binds to whichever isotope
    /// of that element is in the material. Only `D` and `T` (Z=1
    /// hydrogen) carry a specific mass because their single-letter
    /// symbols mean H-2 and H-3 by convention.
    #[test]
    fn thermal_target_x_in_compound() {
        assert_eq!(parse_thermal_target("c_H_in_H2O.h5"),       Some((1, None)));
        assert_eq!(parse_thermal_target("c_D_in_D2O.h5"),       Some((1, Some(2))));
        assert_eq!(parse_thermal_target("c_H_in_ZrH.h5"),       Some((1, None)));
        assert_eq!(parse_thermal_target("c_Zr_in_ZrH.h5"),      Some((40, None)));
        assert_eq!(parse_thermal_target("c_Be_in_BeO.h5"),      Some((4, None)));
        assert_eq!(parse_thermal_target("c_O_in_BeO.h5"),       Some((8, None)));
        assert_eq!(parse_thermal_target("c_O_in_UO2.h5"),       Some((8, None)));
        assert_eq!(parse_thermal_target("c_U_in_UO2.h5"),       Some((92, None)));
        assert_eq!(parse_thermal_target("c_H_in_CH2.h5"),       Some((1, None)));
        assert_eq!(parse_thermal_target("c_H_in_CH4_liquid.h5"), Some((1, None)));
    }

    /// `c_<Symbol><Mass>` — isotope-specific files (metallic Al-27,
    /// metallic Fe-56).
    #[test]
    fn thermal_target_isotope_specific() {
        assert_eq!(parse_thermal_target("c_Al27.h5"), Some((13, Some(27))));
        assert_eq!(parse_thermal_target("c_Fe56.h5"), Some((26, Some(56))));
    }

    /// `c_<Compound>` (no `_in_` separator) — graphite, benzene,
    /// silica. Mapped by the hand-coded table. Natural-element form
    /// for everything except where the filename encodes a specific
    /// isotope glyph (D / T).
    #[test]
    fn thermal_target_compound_only() {
        assert_eq!(parse_thermal_target("c_Graphite.h5"),   Some((6, None)));
        assert_eq!(parse_thermal_target("c_C6H6.h5"),       Some((1, None)));
        assert_eq!(parse_thermal_target("c_SiO2_alpha.h5"), Some((14, None)));
    }

    /// Cold-neutron-source variants. Ortho/para H → natural H
    /// (binds to whichever H isotope is in the material). Ortho/para
    /// D → specifically H-2 because the `D` glyph encodes it.
    #[test]
    fn thermal_target_cold_sources() {
        assert_eq!(parse_thermal_target("c_ortho_H.h5"), Some((1, None)));
        assert_eq!(parse_thermal_target("c_para_H.h5"),  Some((1, None)));
        assert_eq!(parse_thermal_target("c_ortho_D.h5"), Some((1, Some(2))));
        assert_eq!(parse_thermal_target("c_para_D.h5"),  Some((1, Some(2))));
    }

    /// Non-thermal filenames (regular nuclide files, malformed names)
    /// return None — caller surfaces a clear error to the user.
    #[test]
    fn thermal_target_rejects_non_thermal_names() {
        assert_eq!(parse_thermal_target("U235.h5"), None);
        assert_eq!(parse_thermal_target("garbage"), None);
        // Files without the `c_` prefix don't match — the convention
        // requires it, and falling through to a generic parser would
        // silently mis-attach data.
        assert_eq!(parse_thermal_target("H_in_H2O.h5"), None);
    }
}
