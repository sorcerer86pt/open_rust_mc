// SPDX-License-Identifier: MIT
//! Layout-aware lookup helpers for the OpenMC HDF5 data
//! distributions.
//!
//! Background — the on-disk layout changed between ENDF/B releases:
//!
//! ```text
//! ENDF/B-VII.1                 ENDF/B-VIII.0 + VIII.1
//! ── data_root/                 ── data_root/
//!    └─ neutron/                   ├─ neutron/
//!       ├─ U235.h5                 │   └─ U235.h5 (incident-neutron only)
//!       └─ c_H_in_H2O.h5           ├─ thermal/
//!                                  │   └─ c_H_in_H2O.h5 (S(α,β) moved out)
//!                                  └─ photon/
//! ```
//!
//! Callers historically passed `data_dir = data_root/neutron` and
//! resolved a thermal file as `data_dir.join("c_H_in_H2O.h5")`. That
//! still works for VII.1 but misses on VIII.0 / VIII.1, where the
//! file lives at `data_root/thermal/`. [`resolve_thermal_path`]
//! probes both and returns whichever exists.
//!
//! [`discover_neutron_dir`] / [`discover_photon_dir`] pick the best-
//! available library when probing a workspace tree — used by tests
//! and the workspace-walking diagnostics so a developer with only
//! VIII.1 installed (the current default) doesn't get skipped tests.

use std::path::{Path, PathBuf};

/// Library probe order: highest-priority first. Mirrors the
/// `setup_nuclear_data.ps1` default. VIII.1 wins, with VIII.0 and
/// VII.1 as fallbacks so partial installs keep working.
const LIBRARY_PRIORITY: &[&str] = &["endfb-viii.1-hdf5", "endfb-viii.0-hdf5", "endfb-vii.1-hdf5"];

/// Resolve a bare thermal-scattering filename (e.g. `"c_H_in_H2O.h5"`)
/// to an absolute path. Path semantics, in priority order:
///
/// 1. Absolute paths or names containing a separator are returned
///    verbatim — caller already specified the layout.
/// 2. `neutron_dir/name` (VII.1 layout — TSL files mixed into the
///    neutron directory) is returned if it exists.
/// 3. `neutron_dir/../thermal/name` (VIII.x layout) is returned if it
///    exists.
/// 4. Otherwise fall through to `neutron_dir/name` so the caller
///    sees a consistent "missing file at this path" error message.
pub fn resolve_thermal_path(neutron_dir: &Path, name: &str) -> PathBuf {
    let as_path = Path::new(name);
    if as_path.is_absolute() || name.contains('/') || name.contains('\\') {
        return as_path.to_path_buf();
    }

    let same_dir = neutron_dir.join(name);
    if same_dir.exists() {
        return same_dir;
    }

    if let Some(parent) = neutron_dir.parent() {
        let sibling = parent.join("thermal").join(name);
        if sibling.exists() {
            return sibling;
        }
    }

    same_dir
}

/// Walk up from `start` until a `data/<lib>/neutron` directory is
/// found, where `<lib>` is the highest-priority match from
/// [`LIBRARY_PRIORITY`]. Returns `None` if none are present.
///
/// Used by the test suite's `data_dir()` helpers — they previously
/// hardcoded `endfb-vii.1-hdf5` and would skip silently when only
/// VIII.1 is installed.
pub fn discover_neutron_dir(start: &Path) -> Option<PathBuf> {
    discover_subdir(start, "neutron")
}

/// Same as [`discover_neutron_dir`] but for the per-element photon
/// libraries (`data/<lib>/photon`).
pub fn discover_photon_dir(start: &Path) -> Option<PathBuf> {
    discover_subdir(start, "photon")
}

fn discover_subdir(start: &Path, subdir: &str) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        for lib in LIBRARY_PRIORITY {
            let candidate = cur.join("data").join(lib).join(subdir);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
        if !cur.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn absolute_or_separator_paths_pass_through() {
        let neutron = Path::new("/data/x/neutron");
        assert_eq!(
            resolve_thermal_path(neutron, "/abs/c_H_in_H2O.h5"),
            PathBuf::from("/abs/c_H_in_H2O.h5")
        );
        let with_sep = resolve_thermal_path(neutron, "sub/c_H_in_H2O.h5");
        assert!(with_sep.ends_with("sub/c_H_in_H2O.h5"));
    }

    #[test]
    fn missing_file_falls_back_to_neutron_join() {
        let tmp = std::env::temp_dir().join("orm_data_paths_missing");
        let neutron = tmp.join("neutron");
        fs::create_dir_all(&neutron).unwrap();
        let got = resolve_thermal_path(&neutron, "c_ghost.h5");
        assert_eq!(got, neutron.join("c_ghost.h5"));
    }

    #[test]
    fn vii1_layout_uses_neutron_dir() {
        // c_*.h5 sits next to the neutron .h5 files (VII.1 convention).
        let tmp = std::env::temp_dir().join("orm_data_paths_vii1");
        let _ = fs::remove_dir_all(&tmp);
        let neutron = tmp.join("neutron");
        fs::create_dir_all(&neutron).unwrap();
        let here = neutron.join("c_H_in_H2O.h5");
        fs::write(&here, b"").unwrap();
        let got = resolve_thermal_path(&neutron, "c_H_in_H2O.h5");
        assert_eq!(got, here);
    }

    #[test]
    fn viiix_layout_picks_sibling_thermal() {
        // c_*.h5 lives in a sibling thermal/ directory (VIII.0/VIII.1).
        let tmp = std::env::temp_dir().join("orm_data_paths_viiix");
        let _ = fs::remove_dir_all(&tmp);
        let neutron = tmp.join("neutron");
        let thermal = tmp.join("thermal");
        fs::create_dir_all(&neutron).unwrap();
        fs::create_dir_all(&thermal).unwrap();
        let in_thermal = thermal.join("c_H_in_H2O.h5");
        fs::write(&in_thermal, b"").unwrap();
        let got = resolve_thermal_path(&neutron, "c_H_in_H2O.h5");
        assert_eq!(got, in_thermal);
    }

    #[test]
    fn vii1_wins_when_both_locations_have_the_file() {
        // If the operator put it in both places, prefer the legacy
        // sibling location — same dir as the file the caller named.
        // Keeps semantics deterministic.
        let tmp = std::env::temp_dir().join("orm_data_paths_both");
        let _ = fs::remove_dir_all(&tmp);
        let neutron = tmp.join("neutron");
        let thermal = tmp.join("thermal");
        fs::create_dir_all(&neutron).unwrap();
        fs::create_dir_all(&thermal).unwrap();
        let in_neutron = neutron.join("c_H_in_H2O.h5");
        let in_thermal = thermal.join("c_H_in_H2O.h5");
        fs::write(&in_neutron, b"").unwrap();
        fs::write(&in_thermal, b"").unwrap();
        let got = resolve_thermal_path(&neutron, "c_H_in_H2O.h5");
        assert_eq!(got, in_neutron);
    }
}
