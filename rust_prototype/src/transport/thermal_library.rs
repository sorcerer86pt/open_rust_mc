//! S(α,β) thermal-scattering library.
//!
//! Mirrors `NuclideLibrary` for the thermal-scattering side: instead of
//! per-binary `data_dir.join("c_H_in_H2O.h5")` calls, ask the
//! `ThermalLibrary` for a named binding (e.g. `H_IN_H2O`,
//! `D_IN_D2O`) and get back the absolute path. Loading the actual
//! `ThermalScatteringData` is still done by
//! `hdf5_reader::load_thermal_scattering`; this is purely a name →
//! path table.
//!
//! # Available bindings
//!
//! Every entry maps to one ENDF/B `c_<name>.h5` file shipped under
//! `data/endfb-vii.1-hdf5/neutron`. The catalog covers the common
//! reactor moderators / structural compounds:
//!
//! - light water (`H_IN_H2O`)
//! - heavy water (`D_IN_D2O`) — for CANDU / research reactors
//! - graphite (`GRAPHITE`) — for VHTR / pebble-bed
//! - zirconium hydride (`H_IN_ZRH`, `ZR_IN_ZRH`) — TRIGA
//! - beryllium / beryllium oxide (`BE`, `BE_IN_BEO`, `O_IN_BEO`)
//! - polyethylene (`H_IN_CH2`)
//! - benzene (`C6H6`) — calibration source
//! - silica (`SIO2_ALPHA`) — borated glass
//! - bound O / U in fuel (`O_IN_UO2`, `U_IN_UO2`)
//! - liquid / solid methane (`H_IN_CH4_LIQUID`, `H_IN_CH4_SOLID`)
//! - ortho / para H₂ and D₂ (`ORTHO_H`, `PARA_H`, `ORTHO_D`,
//!   `PARA_D`) — cold neutron sources
//! - aluminium / iron metals (`AL27`, `FE56`)

use std::path::{Path, PathBuf};

use crate::hdf5_reader;
use crate::thermal::ThermalScatteringData;

/// Named entry in the static `c_*.h5` catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalBinding {
    HInH2O,
    DInD2O,
    Graphite,
    HInZrH,
    ZrInZrH,
    Be,
    BeInBeO,
    OInBeO,
    HInCH2,
    HInCH4Liquid,
    HInCH4Solid,
    C6H6,
    SiO2Alpha,
    OInUO2,
    UInUO2,
    OrthoH,
    ParaH,
    OrthoD,
    ParaD,
    Al27,
    Fe56,
}

impl ThermalBinding {
    /// Filename component under `data_dir/neutron/`.
    pub const fn filename(self) -> &'static str {
        match self {
            ThermalBinding::HInH2O => "c_H_in_H2O.h5",
            ThermalBinding::DInD2O => "c_D_in_D2O.h5",
            ThermalBinding::Graphite => "c_Graphite.h5",
            ThermalBinding::HInZrH => "c_H_in_ZrH.h5",
            ThermalBinding::ZrInZrH => "c_Zr_in_ZrH.h5",
            ThermalBinding::Be => "c_Be.h5",
            ThermalBinding::BeInBeO => "c_Be_in_BeO.h5",
            ThermalBinding::OInBeO => "c_O_in_BeO.h5",
            ThermalBinding::HInCH2 => "c_H_in_CH2.h5",
            ThermalBinding::HInCH4Liquid => "c_H_in_CH4_liquid.h5",
            ThermalBinding::HInCH4Solid => "c_H_in_CH4_solid.h5",
            ThermalBinding::C6H6 => "c_C6H6.h5",
            ThermalBinding::SiO2Alpha => "c_SiO2_alpha.h5",
            ThermalBinding::OInUO2 => "c_O_in_UO2.h5",
            ThermalBinding::UInUO2 => "c_U_in_UO2.h5",
            ThermalBinding::OrthoH => "c_ortho_H.h5",
            ThermalBinding::ParaH => "c_para_H.h5",
            ThermalBinding::OrthoD => "c_ortho_D.h5",
            ThermalBinding::ParaD => "c_para_D.h5",
            ThermalBinding::Al27 => "c_Al27.h5",
            ThermalBinding::Fe56 => "c_Fe56.h5",
        }
    }

    /// Diagnostic short name — what the binding represents physically.
    pub const fn description(self) -> &'static str {
        match self {
            ThermalBinding::HInH2O => "H in light water",
            ThermalBinding::DInD2O => "D in heavy water",
            ThermalBinding::Graphite => "graphite",
            ThermalBinding::HInZrH => "H in ZrH (TRIGA)",
            ThermalBinding::ZrInZrH => "Zr in ZrH (TRIGA)",
            ThermalBinding::Be => "metallic beryllium",
            ThermalBinding::BeInBeO => "Be in BeO",
            ThermalBinding::OInBeO => "O in BeO",
            ThermalBinding::HInCH2 => "H in polyethylene",
            ThermalBinding::HInCH4Liquid => "H in liquid methane",
            ThermalBinding::HInCH4Solid => "H in solid methane",
            ThermalBinding::C6H6 => "benzene",
            ThermalBinding::SiO2Alpha => "α-quartz",
            ThermalBinding::OInUO2 => "O in UO₂",
            ThermalBinding::UInUO2 => "U in UO₂",
            ThermalBinding::OrthoH => "ortho-H₂",
            ThermalBinding::ParaH => "para-H₂",
            ThermalBinding::OrthoD => "ortho-D₂",
            ThermalBinding::ParaD => "para-D₂",
            ThermalBinding::Al27 => "metallic Al-27",
            ThermalBinding::Fe56 => "metallic Fe-56",
        }
    }
}

/// S(α,β) data-file resolver.
pub struct ThermalLibrary {
    data_dir: PathBuf,
}

impl ThermalLibrary {
    pub fn from_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    /// Path to the HDF5 file for `binding`. The file may not exist —
    /// caller checks with `Path::exists` before passing to
    /// `load_thermal_scattering`. Use `try_load` to do both in one
    /// call.
    pub fn path(&self, binding: ThermalBinding) -> PathBuf {
        self.data_dir.join(binding.filename())
    }

    /// Path-exists check — no I/O beyond stat.
    pub fn has(&self, binding: ThermalBinding) -> bool {
        self.path(binding).exists()
    }

    /// Resolve and load `binding`. Returns `Ok(None)` when the file
    /// doesn't exist on disk (callers can fall back to free-atom
    /// elastic), `Err(_)` when the file exists but fails to parse.
    pub fn try_load(
        &self,
        binding: ThermalBinding,
    ) -> Result<Option<ThermalScatteringData>, String> {
        let path = self.path(binding);
        if !path.exists() {
            return Ok(None);
        }
        match hdf5_reader::load_thermal_scattering(&path) {
            Ok(tsl) => Ok(Some(tsl)),
            Err(e) => Err(format!(
                "{} ({}): {e}",
                binding.description(),
                path.display()
            )),
        }
    }

    /// Load `binding` or die with a panic. Used by tests / demos
    /// that own the geometry and know which thermal data must exist.
    pub fn load_required(&self, binding: ThermalBinding) -> ThermalScatteringData {
        let path = self.path(binding);
        hdf5_reader::load_thermal_scattering(&path).unwrap_or_else(|e| {
            panic!(
                "required thermal binding {} failed: {e}",
                binding.description()
            )
        })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// `filename` is non-empty and ends in `.h5` for every binding.
    /// Catches typos in the const table at compile-test time.
    #[test]
    fn every_binding_has_h5_filename() {
        let bindings: &[ThermalBinding] = &[
            ThermalBinding::HInH2O,
            ThermalBinding::DInD2O,
            ThermalBinding::Graphite,
            ThermalBinding::HInZrH,
            ThermalBinding::ZrInZrH,
            ThermalBinding::Be,
            ThermalBinding::BeInBeO,
            ThermalBinding::OInBeO,
            ThermalBinding::HInCH2,
            ThermalBinding::HInCH4Liquid,
            ThermalBinding::HInCH4Solid,
            ThermalBinding::C6H6,
            ThermalBinding::SiO2Alpha,
            ThermalBinding::OInUO2,
            ThermalBinding::UInUO2,
            ThermalBinding::OrthoH,
            ThermalBinding::ParaH,
            ThermalBinding::OrthoD,
            ThermalBinding::ParaD,
            ThermalBinding::Al27,
            ThermalBinding::Fe56,
        ];
        for &b in bindings {
            let f = b.filename();
            assert!(f.starts_with("c_"), "filename {f} doesn't start with c_");
            assert!(f.ends_with(".h5"), "filename {f} doesn't end with .h5");
            assert!(!b.description().is_empty());
        }
    }

    /// `path` joins data_dir with the binding filename. No I/O.
    #[test]
    fn path_joins_data_dir_with_filename() {
        let lib = ThermalLibrary::from_data_dir("/data/foo");
        let p = lib.path(ThermalBinding::DInD2O);
        assert!(p.ends_with("c_D_in_D2O.h5"));
    }
}
