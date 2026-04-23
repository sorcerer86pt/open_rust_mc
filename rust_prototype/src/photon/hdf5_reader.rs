//! HDF5 reader for OpenMC photon-interaction files (`photon/<Sym>.h5`).
//!
//! Layout (verified against ENDF/B-VII.1 photon library, `filetype =
//! "data_photon"`, `version = [3, 0]`):
//!
//! ```text
//! /                                              attrs: filetype, version
//! /<Sym>                                         attrs: Z
//! /<Sym>/energy                                  shared grid (eV), shape [N_E]
//! /<Sym>/coherent/xs                             shape [N_E]
//! /<Sym>/coherent/scattering_factor              shape [2, N_ff]  (x, F(x,Z))
//! /<Sym>/coherent/integrated_scattering_factor   shape [2, N_ff]  (x², ∫₀ˣ² F² dx'²)
//! /<Sym>/coherent/anomalous_real                 shape [2, N_r]   (E, f'(E))
//! /<Sym>/coherent/anomalous_imag                 shape [2, N_i]   (E, f''(E))
//! /<Sym>/incoherent/xs                           shape [N_E]
//! /<Sym>/incoherent/scattering_factor            shape [2, N_sf]  (x, S(x,Z))
//! /<Sym>/compton_profiles/binding_energy         shape [N_cp]
//! /<Sym>/compton_profiles/num_electrons          shape [N_cp]
//! /<Sym>/compton_profiles/pz                     shape [N_pz]     (|p_z| a.u., ≥ 0)
//! /<Sym>/compton_profiles/J                      shape [N_cp, N_pz]  (J_i(|p_z|))
//! /<Sym>/photoelectric/xs                        shape [N_E]       (sum of subshells)
//! /<Sym>/pair_production_nuclear/xs              shape [N_E]
//! /<Sym>/pair_production_electron/xs             shape [N_E]
//! /<Sym>/subshells                               attrs: designators (required)
//! /<Sym>/subshells/<Shell>                       attrs: binding_energy, num_electrons
//! /<Sym>/subshells/<Shell>/xs                    shape [≤ N_E] (tail-aligned)
//! /<Sym>/subshells/<Shell>/transitions           shape [N_t, 4] (may be empty for outer shells)
//! /<Sym>/bremsstrahlung                          attrs: I (mean excitation eV)
//! /<Sym>/bremsstrahlung/electron_energy          shape [200]
//! /<Sym>/bremsstrahlung/photon_energy            shape [30]
//! /<Sym>/bremsstrahlung/dcs                      shape [200, 30]
//! /<Sym>/bremsstrahlung/ionization_energy        shape [N_osc]
//! /<Sym>/bremsstrahlung/num_electrons            shape [N_osc]
//! ```
//!
//! Conventions confirmed against Carbon (`N_E = 1206`, `N_cp = 3`) and
//! Uranium (`N_E = 3361`, `N_cp = 27`, `N_osc = 26`).

use std::path::Path;

use crate::error::{Result, SvdError};
use crate::photon::data::{
    AnomalousFactors, Bremsstrahlung, ComptonProfiles, PhotonElement, ScatteringFactor, Subshell,
    TabulatedFactor,
};

impl PhotonElement {
    /// Load one element's photon-interaction data from an OpenMC HDF5 file.
    pub fn from_hdf5(path: &Path) -> Result<Self> {
        let hdf5_err = |detail: String| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail,
        };

        let file = hdf5_pure::File::open(path).map_err(|e| hdf5_err(format!("open: {e}")))?;
        let root = file.root();

        let symbol = root
            .groups()
            .map_err(|e| hdf5_err(format!("cannot list root groups: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| hdf5_err("no element group at root".into()))?;

        let element = root
            .group(&symbol)
            .map_err(|e| hdf5_err(format!("cannot open /{symbol}: {e}")))?;

        let z = match element
            .attrs()
            .map_err(|e| hdf5_err(format!("cannot read /{symbol} attrs: {e}")))?
            .get("Z")
        {
            Some(hdf5_pure::AttrValue::I64(z)) => *z as u32,
            Some(_) | None => {
                return Err(hdf5_err(format!("/{symbol} missing Z attribute")));
            }
        };

        let energy = element
            .dataset("energy")
            .map_err(|e| hdf5_err(format!("cannot open /{symbol}/energy: {e}")))?
            .read_f64()
            .map_err(|e| hdf5_err(format!("cannot read /{symbol}/energy: {e}")))?;
        let n_e = energy.len();

        let coherent_xs = read_xs(&element, "coherent", n_e, path)?;
        let incoherent_xs = read_xs(&element, "incoherent", n_e, path)?;
        let photoelectric_xs = read_xs(&element, "photoelectric", n_e, path)?;
        let pair_production_nuclear_xs = read_xs(&element, "pair_production_nuclear", n_e, path)?;
        let pair_production_electron_xs = read_xs(&element, "pair_production_electron", n_e, path)?;

        let coherent_form_factor =
            read_scattering_factor(&element, "coherent", "scattering_factor", path)?;
        let coherent_integrated_form_factor =
            read_scattering_factor(&element, "coherent", "integrated_scattering_factor", path)?;
        let coherent_anomalous = read_anomalous_factors(&element, path)?;

        let incoherent_scattering_factor =
            read_scattering_factor(&element, "incoherent", "scattering_factor", path)?;
        let compton_profiles = read_compton_profiles(&element, path)?;

        let subshells = read_subshells(&element, path)?;

        let bremsstrahlung = read_bremsstrahlung(&element, path)?;

        Ok(PhotonElement {
            z,
            symbol,
            energy,
            coherent_xs,
            incoherent_xs,
            photoelectric_xs,
            pair_production_nuclear_xs,
            pair_production_electron_xs,
            coherent_form_factor,
            coherent_integrated_form_factor,
            coherent_anomalous,
            incoherent_scattering_factor,
            compton_profiles,
            subshells,
            bremsstrahlung,
        })
    }
}

fn read_xs(
    element: &hdf5_pure::Group,
    channel: &str,
    expected_len: usize,
    path: &Path,
) -> Result<Vec<f64>> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };

    let group = element
        .group(channel)
        .map_err(|e| hdf5_err(format!("cannot open /{channel}: {e}")))?;
    let xs = group
        .dataset("xs")
        .map_err(|e| hdf5_err(format!("cannot open /{channel}/xs: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read /{channel}/xs: {e}")))?;
    if xs.len() != expected_len {
        return Err(hdf5_err(format!(
            "/{channel}/xs length {} != energy grid length {}",
            xs.len(),
            expected_len
        )));
    }
    Ok(xs)
}

fn read_scattering_factor(
    element: &hdf5_pure::Group,
    channel: &str,
    dataset: &str,
    path: &Path,
) -> Result<ScatteringFactor> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };

    let group = element
        .group(channel)
        .map_err(|e| hdf5_err(format!("cannot open /{channel}: {e}")))?;
    let ds = group
        .dataset(dataset)
        .map_err(|e| hdf5_err(format!("cannot open /{channel}/{dataset}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| hdf5_err(format!("cannot read /{channel}/{dataset} shape: {e}")))?;
    let flat = ds
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read /{channel}/{dataset}: {e}")))?;

    if shape.len() != 2 || shape[0] != 2 {
        return Err(hdf5_err(format!(
            "/{channel}/{dataset} shape {:?} is not [2, N]",
            shape
        )));
    }
    let n = shape[1] as usize;
    if flat.len() != 2 * n {
        return Err(hdf5_err(format!(
            "/{channel}/{dataset} flat length {} != 2*{}",
            flat.len(),
            n
        )));
    }
    // Row-major: row 0 = x, row 1 = value.
    let x = flat[..n].to_vec();
    let value = flat[n..].to_vec();
    Ok(ScatteringFactor { x, value })
}

fn read_subshells(element: &hdf5_pure::Group, path: &Path) -> Result<Vec<Subshell>> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };

    let subshells_group = element
        .group("subshells")
        .map_err(|e| hdf5_err(format!("cannot open /subshells: {e}")))?;

    // The `designators` attribute is required — it defines the physical
    // K → L → M → ... order in which subshells appear. Falling back to
    // `groups()` returns HDF5-internal order (often alphabetical), which
    // breaks any convention that assumes outer shells are indexed later.
    //
    // HDF5 can store a single-entry attribute as either `String` or a
    // one-element `StringArray` depending on the writer; we accept both.
    let designators: Vec<String> = match subshells_group
        .attrs()
        .map_err(|e| hdf5_err(format!("cannot read subshells attrs: {e}")))?
        .get("designators")
    {
        Some(hdf5_pure::AttrValue::StringArray(arr)) => arr.clone(),
        Some(hdf5_pure::AttrValue::String(s)) => vec![s.clone()],
        _ => {
            return Err(hdf5_err(
                "/subshells is missing required `designators` attribute".into(),
            ));
        }
    };

    let mut out = Vec::with_capacity(designators.len());
    for designator in designators {
        let shell = subshells_group
            .group(&designator)
            .map_err(|e| hdf5_err(format!("cannot open /subshells/{designator}: {e}")))?;
        let shell_attrs = shell
            .attrs()
            .map_err(|e| hdf5_err(format!("cannot read subshell {designator} attrs: {e}")))?;
        let binding_energy = match shell_attrs.get("binding_energy") {
            Some(hdf5_pure::AttrValue::F64(v)) => *v,
            _ => {
                return Err(hdf5_err(format!(
                    "/subshells/{designator} missing binding_energy attribute"
                )));
            }
        };
        let num_electrons = match shell_attrs.get("num_electrons") {
            Some(hdf5_pure::AttrValue::F64(v)) => *v,
            _ => {
                return Err(hdf5_err(format!(
                    "/subshells/{designator} missing num_electrons attribute"
                )));
            }
        };
        let xs = shell
            .dataset("xs")
            .map_err(|e| hdf5_err(format!("cannot open /subshells/{designator}/xs: {e}")))?
            .read_f64()
            .map_err(|e| hdf5_err(format!("cannot read /subshells/{designator}/xs: {e}")))?;

        // Transitions are optional — outer shells carry none.
        let transitions = read_transitions(&shell);

        out.push(Subshell {
            designator,
            binding_energy,
            num_electrons,
            xs,
            transitions,
        });
    }
    Ok(out)
}

fn read_tabulated_2xn(
    group: &hdf5_pure::Group,
    dataset: &str,
    context: &str,
    path: &Path,
) -> Result<TabulatedFactor> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };
    let ds = group
        .dataset(dataset)
        .map_err(|e| hdf5_err(format!("cannot open {context}/{dataset}: {e}")))?;
    let shape = ds
        .shape()
        .map_err(|e| hdf5_err(format!("cannot read {context}/{dataset} shape: {e}")))?;
    let flat = ds
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read {context}/{dataset}: {e}")))?;
    if shape.len() != 2 || shape[0] != 2 {
        return Err(hdf5_err(format!(
            "{context}/{dataset} shape {:?} is not [2, N]",
            shape
        )));
    }
    let n = shape[1] as usize;
    if flat.len() != 2 * n {
        return Err(hdf5_err(format!(
            "{context}/{dataset} flat length {} != 2*{}",
            flat.len(),
            n
        )));
    }
    Ok(TabulatedFactor {
        grid: flat[..n].to_vec(),
        value: flat[n..].to_vec(),
    })
}

fn read_anomalous_factors(element: &hdf5_pure::Group, path: &Path) -> Result<AnomalousFactors> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };
    let coherent = element
        .group("coherent")
        .map_err(|e| hdf5_err(format!("cannot open /coherent: {e}")))?;
    let real = read_tabulated_2xn(&coherent, "anomalous_real", "/coherent", path)?;
    let imag = read_tabulated_2xn(&coherent, "anomalous_imag", "/coherent", path)?;
    Ok(AnomalousFactors { real, imag })
}

fn read_compton_profiles(element: &hdf5_pure::Group, path: &Path) -> Result<ComptonProfiles> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };
    let cp = element
        .group("compton_profiles")
        .map_err(|e| hdf5_err(format!("cannot open /compton_profiles: {e}")))?;

    let binding_energy = cp
        .dataset("binding_energy")
        .map_err(|e| hdf5_err(format!("cannot open binding_energy: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read binding_energy: {e}")))?;
    let num_electrons = cp
        .dataset("num_electrons")
        .map_err(|e| hdf5_err(format!("cannot open num_electrons: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read num_electrons: {e}")))?;
    let pz = cp
        .dataset("pz")
        .map_err(|e| hdf5_err(format!("cannot open pz: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read pz: {e}")))?;

    let n_shells = binding_energy.len();
    if num_electrons.len() != n_shells {
        return Err(hdf5_err(format!(
            "compton_profiles: num_electrons len {} != binding_energy len {}",
            num_electrons.len(),
            n_shells
        )));
    }
    let n_pz = pz.len();

    let j_ds = cp
        .dataset("J")
        .map_err(|e| hdf5_err(format!("cannot open J: {e}")))?;
    let j_shape = j_ds
        .shape()
        .map_err(|e| hdf5_err(format!("cannot read J shape: {e}")))?;
    let j_flat = j_ds
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read J: {e}")))?;
    if j_shape.len() != 2 || j_shape[0] as usize != n_shells || j_shape[1] as usize != n_pz {
        return Err(hdf5_err(format!(
            "compton_profiles/J shape {:?} != [{}, {}]",
            j_shape, n_shells, n_pz
        )));
    }
    let mut j = Vec::with_capacity(n_shells);
    for i in 0..n_shells {
        j.push(j_flat[i * n_pz..(i + 1) * n_pz].to_vec());
    }

    Ok(ComptonProfiles {
        binding_energy,
        num_electrons,
        pz,
        j,
    })
}

fn read_bremsstrahlung(element: &hdf5_pure::Group, path: &Path) -> Result<Bremsstrahlung> {
    let hdf5_err = |detail: String| SvdError::Hdf5 {
        path: path.display().to_string(),
        detail,
    };
    let br = element
        .group("bremsstrahlung")
        .map_err(|e| hdf5_err(format!("cannot open /bremsstrahlung: {e}")))?;

    let mean_excitation_energy = match br
        .attrs()
        .map_err(|e| hdf5_err(format!("cannot read bremsstrahlung attrs: {e}")))?
        .get("I")
    {
        Some(hdf5_pure::AttrValue::F64(v)) => *v,
        _ => 0.0,
    };

    let electron_energy = br
        .dataset("electron_energy")
        .map_err(|e| hdf5_err(format!("cannot open electron_energy: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read electron_energy: {e}")))?;
    let photon_energy = br
        .dataset("photon_energy")
        .map_err(|e| hdf5_err(format!("cannot open photon_energy: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read photon_energy: {e}")))?;
    let ionization_energy = br
        .dataset("ionization_energy")
        .map_err(|e| hdf5_err(format!("cannot open ionization_energy: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read ionization_energy: {e}")))?;
    let num_electrons = br
        .dataset("num_electrons")
        .map_err(|e| hdf5_err(format!("cannot open bremsstrahlung num_electrons: {e}")))?
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read bremsstrahlung num_electrons: {e}")))?;

    let n_e = electron_energy.len();
    let n_k = photon_energy.len();
    let dcs_ds = br
        .dataset("dcs")
        .map_err(|e| hdf5_err(format!("cannot open dcs: {e}")))?;
    let dcs_shape = dcs_ds
        .shape()
        .map_err(|e| hdf5_err(format!("cannot read dcs shape: {e}")))?;
    let dcs_flat = dcs_ds
        .read_f64()
        .map_err(|e| hdf5_err(format!("cannot read dcs: {e}")))?;
    if dcs_shape.len() != 2 || dcs_shape[0] as usize != n_e || dcs_shape[1] as usize != n_k {
        return Err(hdf5_err(format!(
            "bremsstrahlung/dcs shape {:?} != [{}, {}]",
            dcs_shape, n_e, n_k
        )));
    }
    let mut dcs = Vec::with_capacity(n_e);
    for i in 0..n_e {
        dcs.push(dcs_flat[i * n_k..(i + 1) * n_k].to_vec());
    }

    if ionization_energy.len() != num_electrons.len() {
        return Err(hdf5_err(format!(
            "bremsstrahlung: ionization_energy len {} != num_electrons len {}",
            ionization_energy.len(),
            num_electrons.len()
        )));
    }

    Ok(Bremsstrahlung {
        mean_excitation_energy,
        electron_energy,
        photon_energy,
        dcs,
        ionization_energy,
        num_electrons,
    })
}

fn read_transitions(shell: &hdf5_pure::Group) -> Vec<[f64; 4]> {
    let ds = match shell.dataset("transitions") {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let shape = ds.shape().unwrap_or_default();
    if shape.len() != 2 || shape[1] != 4 {
        return Vec::new();
    }
    let flat = match ds.read_f64() {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let n = shape[0] as usize;
    if flat.len() != 4 * n {
        return Vec::new();
    }
    (0..n)
        .map(|i| {
            [
                flat[4 * i],
                flat[4 * i + 1],
                flat[4 * i + 2],
                flat[4 * i + 3],
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn photon_path(filename: &str) -> Option<PathBuf> {
        // Walk up from CARGO_MANIFEST_DIR (rust_prototype/) to project root.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon")
            .join(filename);
        if path.exists() { Some(path) } else { None }
    }

    /// Element-agnostic structural and physical-consistency checks.
    ///
    /// `expected_z` is cross-checked against the HDF5 `Z` attribute.
    /// Consistency checks verified here are properties every OpenMC
    /// photon file must satisfy; they should pass for any element.
    fn check_photon_element(elem: &PhotonElement, expected_z: u32, expected_symbol: &str) {
        // --- Identity ------------------------------------------------------
        assert_eq!(elem.z, expected_z);
        assert_eq!(elem.symbol, expected_symbol);

        // --- Grid shape and monotonicity ----------------------------------
        let n = elem.n_energy();
        assert!(n > 0);
        assert_eq!(elem.coherent_xs.len(), n);
        assert_eq!(elem.incoherent_xs.len(), n);
        assert_eq!(elem.photoelectric_xs.len(), n);
        assert_eq!(elem.pair_production_nuclear_xs.len(), n);
        assert_eq!(elem.pair_production_electron_xs.len(), n);
        for w in elem.energy.windows(2) {
            assert!(w[1] > w[0], "energy grid not strictly increasing");
        }

        // --- Cross-section non-negativity ---------------------------------
        for xs in &elem.coherent_xs {
            assert!(*xs >= 0.0);
        }
        for xs in &elem.incoherent_xs {
            assert!(*xs >= 0.0);
        }
        for xs in &elem.photoelectric_xs {
            assert!(*xs >= 0.0);
        }
        for xs in &elem.pair_production_nuclear_xs {
            assert!(*xs >= 0.0);
        }
        for xs in &elem.pair_production_electron_xs {
            assert!(*xs >= 0.0);
        }

        // --- Pair-production thresholds ----------------------------------
        // Nuclear: 2 m_e c² ≈ 1.022 MeV. Triplet: 4 m_e c² ≈ 2.044 MeV.
        const M_E_C2_EV: f64 = 510_998.95;
        for (i, &e) in elem.energy.iter().enumerate() {
            if e < 2.0 * M_E_C2_EV {
                assert_eq!(
                    elem.pair_production_nuclear_xs[i], 0.0,
                    "nuclear pair production nonzero below threshold at E={e}"
                );
            }
            if e < 4.0 * M_E_C2_EV {
                assert_eq!(
                    elem.pair_production_electron_xs[i], 0.0,
                    "triplet pair production nonzero below threshold at E={e}"
                );
            }
        }

        // --- Scattering factor physical limits ---------------------------
        // F(x = 0, Z) = Z (forward elastic scattering is coherent over Z electrons).
        let f0 = elem.coherent_form_factor.value[0];
        assert!(
            (f0 - expected_z as f64).abs() < 1e-6,
            "F(0, Z) = {f0}, expected Z = {expected_z}"
        );

        // S(x → ∞, Z) → Z (all electrons behave as free at large momentum transfer).
        let s_tail = *elem.incoherent_scattering_factor.value.last().unwrap();
        assert!(
            (s_tail - expected_z as f64).abs() < 0.05 * expected_z as f64,
            "S(x_max, Z) = {s_tail}, expected ≈ Z = {expected_z}"
        );

        // F and S monotone in their respective directions.
        let ff = &elem.coherent_form_factor;
        assert_eq!(ff.x.len(), ff.value.len());
        for w in ff.x.windows(2) {
            assert!(w[1] > w[0]);
        }
        // F(x, Z) decreases from Z at x=0 toward 0 at x → ∞.
        assert!(
            ff.value[0] >= *ff.value.last().unwrap(),
            "form factor should decrease from x=0 to x=∞"
        );
        let sf = &elem.incoherent_scattering_factor;
        for w in sf.x.windows(2) {
            assert!(w[1] > w[0]);
        }

        // --- Integrated form factor is non-decreasing --------------------
        let iff = &elem.coherent_integrated_form_factor;
        assert_eq!(iff.x.len(), iff.value.len());
        for w in iff.value.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "integrated form factor not monotone: {} -> {}",
                w[0],
                w[1]
            );
        }

        // --- Anomalous factors sized consistently ------------------------
        let af = &elem.coherent_anomalous;
        assert_eq!(af.real.grid.len(), af.real.value.len());
        assert_eq!(af.imag.grid.len(), af.imag.value.len());
        assert!(!af.real.grid.is_empty());
        assert!(!af.imag.grid.is_empty());

        // --- Compton profiles: electron conservation ---------------------
        let cp = &elem.compton_profiles;
        assert!(cp.n_shells() > 0);
        assert_eq!(cp.n_pz(), 31);
        assert_eq!(cp.j.len(), cp.n_shells());
        for row in &cp.j {
            assert_eq!(row.len(), cp.n_pz());
            for v in row {
                assert!(*v >= 0.0);
            }
        }
        assert!(cp.pz[0].abs() < 1e-12, "pz grid should start at 0");
        for w in cp.pz.windows(2) {
            assert!(w[1] > w[0]);
        }
        let total_cp_occ: f64 = cp.num_electrons.iter().sum();
        assert!(
            (total_cp_occ - expected_z as f64).abs() < 1e-6,
            "Compton profile total occupancy = {total_cp_occ}, expected Z = {expected_z}"
        );

        // --- Photoelectric subshells: electron conservation --------------
        let total_pe_occ: f64 = elem.subshells.iter().map(|s| s.num_electrons).sum();
        assert!(
            (total_pe_occ - expected_z as f64).abs() < 1e-6,
            "photoelectric subshell total occupancy = {total_pe_occ}, expected Z = {expected_z}"
        );

        // --- Cross-consistency: sum of subshell partial XS ≈ total PE XS ---
        // The total photoelectric cross section is taken from the ENDF/B
        // evaluation while the per-subshell cross sections come from the
        // LLNL EADL/EPICS photoatomic library. The two evaluations agree
        // well on the integrated XS but can disagree by up to ~1 % at
        // individual energy points, especially near absorption edges
        // where shell contributions switch on discontinuously. A 2 %
        // relative tolerance (OR 0.1 barn absolute floor for very small
        // total XS) validates that the tail-alignment convention and
        // per-subshell reads are structurally correct without rejecting
        // legitimate evaluation noise.
        let probe_indices = [n / 4, n / 2, n - n / 4, n.saturating_sub(1)];
        for &i in &probe_indices {
            let from_subshells: f64 = elem.subshells.iter().map(|s| s.xs_at(n, i)).sum();
            let total = elem.photoelectric_xs[i];
            let tol = 0.02 * total.abs().max(0.1);
            assert!(
                (from_subshells - total).abs() <= tol,
                "photoelectric consistency failed at energy[{i}] = {} eV: \
                 Σ subshells = {from_subshells}, total = {total}, tol = {tol}",
                elem.energy[i]
            );
        }

        // --- Bremsstrahlung shape --------------------------------------
        let br = &elem.bremsstrahlung;
        assert_eq!(br.electron_energy.len(), 200);
        assert_eq!(br.photon_energy.len(), 30);
        assert_eq!(br.dcs.len(), 200);
        for row in &br.dcs {
            assert_eq!(row.len(), 30);
        }
        assert!(br.mean_excitation_energy > 0.0);
        assert_eq!(br.ionization_energy.len(), br.num_electrons.len());
        // Note: `num_electrons` in the Sternheimer-Berger block is not
        // a pure electron count but a fit parameter for the
        // density-effect correction; individual values can be negative
        // and the sum need not equal Z (for U the sum is 86 vs Z = 92).
        // We therefore only assert the data is present and sized
        // consistently; semantic validation is the TTB kernel's job.
        assert!(!br.num_electrons.is_empty());

        // --- Transitions: probabilities within each shell sum to ≤ 1 ----
        for s in &elem.subshells {
            let total_prob: f64 = s.transitions.iter().map(|t| t[3]).sum();
            assert!(
                total_prob <= 1.0 + 1e-6,
                "subshell {} transition probabilities sum to {total_prob} > 1",
                s.designator
            );
        }
    }

    #[test]
    fn load_carbon_photon_data() {
        let Some(path) = photon_path("C.h5") else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/C.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load carbon photon data");
        check_photon_element(&elem, 6, "C");

        // Carbon-specific spot checks.
        assert_eq!(elem.n_energy(), 1206);
        assert_eq!(elem.compton_profiles.n_shells(), 3);

        // Carbon K, L1, L2, L3 in order.
        let designators: Vec<&str> = elem
            .subshells
            .iter()
            .map(|s| s.designator.as_str())
            .collect();
        assert_eq!(designators, vec!["K", "L1", "L2", "L3"]);

        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        assert!(
            (k.binding_energy - 291.01).abs() < 0.01,
            "C K-shell binding = {} eV, expected 291.01",
            k.binding_energy
        );
        assert!((k.num_electrons - 2.0).abs() < 1e-9);

        // ICRU-37 mean excitation energy for carbon.
        assert!((elem.bremsstrahlung.mean_excitation_energy - 81.0).abs() < 1.0);

        // Tail-alignment spot check on K-shell.
        let n = elem.n_energy();
        let offset = n - k.xs.len();
        assert_eq!(k.xs_at(n, offset), k.xs[0]);
        assert_eq!(k.xs_at(n, n - 1), *k.xs.last().unwrap());
        assert_eq!(k.xs_at(n, offset.saturating_sub(1)), 0.0);
    }

    #[test]
    fn load_uranium_photon_data() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load uranium photon data");
        check_photon_element(&elem, 92, "U");

        // Uranium-specific spot checks.
        assert_eq!(elem.n_energy(), 3361);
        assert_eq!(elem.compton_profiles.n_shells(), 27);
        assert_eq!(elem.subshells.len(), 29);

        // K-shell binding from ENDF/B-VII.1 evaluation.
        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        assert!(
            (k.binding_energy - 116_110.0).abs() < 10.0,
            "U K-shell binding = {} eV, expected 116110",
            k.binding_energy
        );
        assert!((k.num_electrons - 2.0).abs() < 1e-9);

        // U has rich K-shell relaxation.
        assert!(
            k.transitions.len() > 100,
            "U K-shell should have many transitions, got {}",
            k.transitions.len()
        );
        // At least one transition is radiative (secondary = 0).
        assert!(k.transitions.iter().any(|t| t[1] == 0.0));
        // At least one is Auger (secondary != 0).
        assert!(k.transitions.iter().any(|t| t[1] != 0.0));

        // ICRU-37 mean excitation energy for uranium.
        assert!((elem.bremsstrahlung.mean_excitation_energy - 890.0).abs() < 5.0);
    }

    // -------------------------------------------------------------------
    // Broad-Z element coverage
    //
    // Each test runs the full element-agnostic consistency suite from
    // `check_photon_element`. Picked to span the periodic table:
    //   - H  (Z=1): minimal atom, single K shell
    //   - Fe (Z=26): mid-Z, common structural nuclide
    //   - Pb (Z=82): heavy shielding material
    // -------------------------------------------------------------------

    #[test]
    fn load_hydrogen_photon_data() {
        let Some(path) = photon_path("H.h5") else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/H.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load hydrogen photon data");
        check_photon_element(&elem, 1, "H");

        // H has exactly one subshell: K (1s, 1 electron).
        assert_eq!(elem.subshells.len(), 1);
        let k = &elem.subshells[0];
        assert_eq!(k.designator, "K");
        assert!((k.num_electrons - 1.0).abs() < 1e-9);
        // H K-shell binding is 13.6 eV (Rydberg). Evaluated libraries
        // allow small adjustments for chemical environment.
        assert!(
            (k.binding_energy - 13.6).abs() < 0.5,
            "H K-shell binding = {} eV, expected ~13.6",
            k.binding_energy
        );

        // A 1-electron atom has no relaxation pathway.
        assert!(
            k.transitions.is_empty(),
            "H K-shell should have no transitions, got {}",
            k.transitions.len()
        );

        // Compton profiles: 1 shell for 1 electron.
        assert_eq!(elem.compton_profiles.n_shells(), 1);
        assert!((elem.compton_profiles.num_electrons[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn load_iron_photon_data() {
        let Some(path) = photon_path("Fe.h5") else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/Fe.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load iron photon data");
        check_photon_element(&elem, 26, "Fe");

        // Iron has K, L1-3, M1-5 filled (9 subshells populated + M4-M5 last).
        assert!(elem.subshells.len() >= 9);
        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        // ENDF/B-VII.1 evaluated Fe K-shell binding ≈ 7083 eV (the NIST
        // X-ray absorption edge value of 7112 eV is for the solid-state
        // Fe K-edge and differs by several tens of eV from the isolated
        // neutral-atom binding the evaluated file stores).
        assert!(
            (k.binding_energy - 7083.0).abs() < 30.0,
            "Fe K-shell binding = {} eV, expected ~7083 (ENDF/B-VII.1)",
            k.binding_energy
        );
    }

    #[test]
    fn load_lead_photon_data() {
        let Some(path) = photon_path("Pb.h5") else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/Pb.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load lead photon data");
        check_photon_element(&elem, 82, "Pb");

        // ENDF/B-VII.1 evaluated Pb K-shell binding ≈ 88290 eV (the NIST
        // X-ray K-edge of 88005 eV is the solid-state value; evaluated
        // photoatomic libraries store the isolated neutral-atom binding
        // which differs by a few hundred eV).
        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        assert!(
            (k.binding_energy - 88_290.0).abs() < 500.0,
            "Pb K-shell binding = {} eV, expected ~88290 (ENDF/B-VII.1)",
            k.binding_energy
        );

        // Pb has a rich M and N shell structure.
        let designators: Vec<String> = elem
            .subshells
            .iter()
            .map(|s| s.designator.clone())
            .collect();
        assert!(designators.iter().any(|d| d.starts_with('M')));
        assert!(designators.iter().any(|d| d.starts_with('N')));
    }

    // -------------------------------------------------------------------
    // Pure-function unit tests — no HDF5 dependency
    // -------------------------------------------------------------------

    mod subshell_xs_at {
        use super::super::*;

        fn make_subshell(xs: Vec<f64>) -> Subshell {
            Subshell {
                designator: "K".into(),
                binding_energy: 100.0,
                num_electrons: 2.0,
                xs,
                transitions: Vec::new(),
            }
        }

        #[test]
        fn at_offset_returns_first_point() {
            let s = make_subshell(vec![10.0, 20.0, 30.0]);
            // master grid length 5, xs length 3 => offset = 2.
            assert_eq!(s.xs_at(5, 2), 10.0);
        }

        #[test]
        fn at_last_master_index_returns_last_xs() {
            let s = make_subshell(vec![10.0, 20.0, 30.0]);
            assert_eq!(s.xs_at(5, 4), 30.0);
        }

        #[test]
        fn below_offset_returns_zero() {
            let s = make_subshell(vec![10.0, 20.0, 30.0]);
            assert_eq!(s.xs_at(5, 0), 0.0);
            assert_eq!(s.xs_at(5, 1), 0.0);
        }

        #[test]
        fn empty_xs_returns_zero_everywhere() {
            let s = make_subshell(vec![]);
            // offset = master_len, every valid index is below offset.
            for i in 0..5 {
                assert_eq!(s.xs_at(5, i), 0.0, "expected 0 at i={i}");
            }
        }

        #[test]
        fn xs_fully_covers_master_grid() {
            // When xs.len() == master_len the whole grid is tabulated.
            let s = make_subshell(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(s.xs_at(5, 0), 1.0);
            assert_eq!(s.xs_at(5, 4), 5.0);
        }

        #[test]
        #[should_panic(expected = "i_master")]
        #[cfg(debug_assertions)]
        fn oob_master_index_panics_in_debug() {
            let s = make_subshell(vec![1.0, 2.0, 3.0]);
            let _ = s.xs_at(5, 5);
        }

        #[test]
        #[should_panic(expected = "master grid")]
        #[cfg(debug_assertions)]
        fn xs_longer_than_master_panics_in_debug() {
            let s = make_subshell(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let _ = s.xs_at(5, 0);
        }
    }

    mod photon_element_helpers {
        use super::super::*;
        use crate::photon::data::{
            AnomalousFactors, Bremsstrahlung, ComptonProfiles, ScatteringFactor, TabulatedFactor,
        };

        fn empty_sf() -> ScatteringFactor {
            ScatteringFactor {
                x: vec![],
                value: vec![],
            }
        }

        fn empty_tab() -> TabulatedFactor {
            TabulatedFactor {
                grid: vec![],
                value: vec![],
            }
        }

        fn fixture(xs: Vec<[f64; 5]>) -> PhotonElement {
            let n = xs.len();
            PhotonElement {
                z: 1,
                symbol: "X".into(),
                energy: (0..n).map(|i| i as f64).collect(),
                coherent_xs: xs.iter().map(|r| r[0]).collect(),
                incoherent_xs: xs.iter().map(|r| r[1]).collect(),
                photoelectric_xs: xs.iter().map(|r| r[2]).collect(),
                pair_production_nuclear_xs: xs.iter().map(|r| r[3]).collect(),
                pair_production_electron_xs: xs.iter().map(|r| r[4]).collect(),
                coherent_form_factor: empty_sf(),
                coherent_integrated_form_factor: empty_sf(),
                coherent_anomalous: AnomalousFactors {
                    real: empty_tab(),
                    imag: empty_tab(),
                },
                incoherent_scattering_factor: empty_sf(),
                compton_profiles: ComptonProfiles {
                    binding_energy: vec![],
                    num_electrons: vec![],
                    pz: vec![],
                    j: vec![],
                },
                subshells: vec![],
                bremsstrahlung: Bremsstrahlung {
                    mean_excitation_energy: 0.0,
                    electron_energy: vec![],
                    photon_energy: vec![],
                    dcs: vec![],
                    ionization_energy: vec![],
                    num_electrons: vec![],
                },
            }
        }

        #[test]
        fn n_energy_matches_grid_length() {
            let elem = fixture(vec![[0.0; 5]; 7]);
            assert_eq!(elem.n_energy(), 7);
        }

        #[test]
        fn total_xs_is_sum_of_five_channels() {
            let elem = fixture(vec![[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]]);
            assert_eq!(elem.total_xs_at(0), 15.0);
            assert_eq!(elem.total_xs_at(1), 2.5);
        }

        #[test]
        fn total_xs_handles_zero_channels() {
            let elem = fixture(vec![[0.0, 5.0, 0.0, 0.0, 0.0]]);
            assert_eq!(elem.total_xs_at(0), 5.0);
        }
    }

    mod compton_profiles {
        use crate::photon::data::ComptonProfiles;

        #[test]
        fn n_shells_matches_binding_energy_len() {
            let cp = ComptonProfiles {
                binding_energy: vec![1.0, 2.0, 3.0],
                num_electrons: vec![2.0, 2.0, 2.0],
                pz: vec![0.0, 1.0],
                j: vec![vec![1.0, 0.5]; 3],
            };
            assert_eq!(cp.n_shells(), 3);
            assert_eq!(cp.n_pz(), 2);
        }

        #[test]
        fn empty_profiles_report_zero() {
            let cp = ComptonProfiles {
                binding_energy: vec![],
                num_electrons: vec![],
                pz: vec![],
                j: vec![],
            };
            assert_eq!(cp.n_shells(), 0);
            assert_eq!(cp.n_pz(), 0);
        }
    }

    // -------------------------------------------------------------------
    // Error-path tests — no data file required
    // -------------------------------------------------------------------

    mod error_paths {
        use super::super::*;
        use std::io::Write;
        use std::path::PathBuf;

        #[test]
        fn missing_file_returns_err() {
            let path = PathBuf::from("data/photon/this-definitely-does-not-exist-1234.h5");
            let result = PhotonElement::from_hdf5(&path);
            assert!(
                result.is_err(),
                "loading a non-existent path should return Err, got Ok"
            );
        }

        #[test]
        fn non_hdf5_file_returns_err() {
            // Write a plain text file and try to parse it as HDF5.
            let dir = std::env::temp_dir();
            let path = dir.join(format!("photon_bogus_{}.h5", std::process::id()));
            {
                let mut f = std::fs::File::create(&path).expect("create scratch file");
                f.write_all(b"this is not HDF5, it's plain text\n").unwrap();
            }
            let result = PhotonElement::from_hdf5(&path);
            let _ = std::fs::remove_file(&path);
            assert!(
                result.is_err(),
                "loading a non-HDF5 file should return Err, got Ok"
            );
        }
    }

    // -------------------------------------------------------------------
    // Cross-data physics consistency — checks that require real files
    // -------------------------------------------------------------------

    /// Validate that every transition in every subshell has well-formed
    /// data: primary designator positive, secondary either 0 (radiative)
    /// or positive, energy and probability non-negative.
    ///
    /// EADL uses a 1-based designator scheme running from 1 (K) up to
    /// ~39 (outermost shells), independent of whether each shell appears
    /// in a given element's photoelectric subshell list. The relaxation
    /// cascade can therefore reference shells that have no tabulated
    /// photoelectric cross section (e.g. valence shells receive holes
    /// but the partial PE XS is absorbed into the total). We bound by
    /// 50 rather than `subshells.len()` for this reason.
    #[test]
    fn uranium_transitions_well_formed() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load uranium");
        const EADL_MAX_DESIGNATOR: i64 = 50; // safely above any physical shell

        for s in &elem.subshells {
            for (i, t) in s.transitions.iter().enumerate() {
                let primary = t[0];
                let secondary = t[1];

                let primary_idx = primary.round() as i64;
                assert!(
                    (1..=EADL_MAX_DESIGNATOR).contains(&primary_idx),
                    "shell {} transition {i}: primary designator {primary} outside [1, {EADL_MAX_DESIGNATOR}]",
                    s.designator
                );

                if secondary != 0.0 {
                    let secondary_idx = secondary.round() as i64;
                    assert!(
                        (1..=EADL_MAX_DESIGNATOR).contains(&secondary_idx),
                        "shell {} transition {i}: secondary designator {secondary} outside [1, {EADL_MAX_DESIGNATOR}]",
                        s.designator
                    );
                }

                assert!(
                    t[2] >= 0.0,
                    "shell {} transition {i}: energy {} < 0",
                    s.designator,
                    t[2]
                );
                assert!(
                    t[3] >= 0.0,
                    "shell {} transition {i}: probability {} < 0",
                    s.designator,
                    t[3]
                );
            }
        }
    }

    /// Physical-regime check: for lead at 10 keV, photoelectric must be
    /// the dominant channel (by a large margin). At 10 MeV, pair
    /// production becomes competitive with Compton. This catches gross
    /// channel-misidentification bugs (e.g., swapping coherent/incoherent).
    #[test]
    fn lead_channel_dominance_by_energy() {
        let Some(path) = photon_path("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load Pb");

        // Find the index closest to 10 keV.
        let idx_10kev = elem
            .energy
            .iter()
            .position(|e| *e >= 1.0e4)
            .expect("10 keV in grid");
        let pe_10k = elem.photoelectric_xs[idx_10kev];
        let total_10k = elem.total_xs_at(idx_10kev);
        assert!(
            pe_10k / total_10k > 0.9,
            "Pb at 10 keV: photoelectric should dominate, fraction = {} (total = {total_10k})",
            pe_10k / total_10k
        );

        // At 1 MeV, Compton (incoherent) should dominate for Pb.
        let idx_1mev = elem
            .energy
            .iter()
            .position(|e| *e >= 1.0e6)
            .expect("1 MeV in grid");
        let inc_1m = elem.incoherent_xs[idx_1mev];
        let total_1m = elem.total_xs_at(idx_1mev);
        assert!(
            inc_1m / total_1m > 0.5,
            "Pb at 1 MeV: Compton fraction = {} (expected > 0.5)",
            inc_1m / total_1m
        );

        // At 100 MeV, pair production should dominate over Compton.
        let idx_100mev = elem
            .energy
            .iter()
            .position(|e| *e >= 1.0e8)
            .expect("100 MeV in grid");
        let pp_100m = elem.pair_production_nuclear_xs[idx_100mev]
            + elem.pair_production_electron_xs[idx_100mev];
        let inc_100m = elem.incoherent_xs[idx_100mev];
        assert!(
            pp_100m > inc_100m,
            "Pb at 100 MeV: pair production ({pp_100m} b) should exceed Compton ({inc_100m} b)"
        );
    }

    /// Compton profile physics: for a closed-shell Hartree-Fock ground
    /// state J is an even, non-increasing function of |p_z|, so the
    /// maximum is at p_z = 0. Ties at adjacent low-p_z points are
    /// expected for deeply-bound inner shells where the profile is
    /// broad (e.g. U K-shell electrons have ~100 keV binding energy
    /// and the profile is flat-topped over the first few tabulated
    /// p_z points). We therefore check `J(p_z) ≤ J(0)` with a small
    /// tolerance rather than strict peaking.
    #[test]
    fn uranium_compton_profile_maximal_at_zero_pz() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load U");
        let cp = &elem.compton_profiles;
        for (shell, prof) in cp.j.iter().enumerate() {
            let j0 = prof[0];
            for (k, &j_val) in prof.iter().enumerate() {
                assert!(
                    j_val <= j0 + 1e-12,
                    "U shell {shell}: J[pz={}] = {j_val} exceeds J[0] = {j0}",
                    cp.pz[k]
                );
            }
        }
    }

    /// EADL designator → subshell lookup. For Uranium, the 29-entry
    /// PE subshell list must expose K (1) through Q1 (29), and any
    /// designator ≥ 30 must return None (those shells aren't in the
    /// tabulated PE list even though EADL transitions can reference
    /// them).
    #[test]
    fn uranium_subshell_by_eadl_designator() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load U");
        assert_eq!(elem.subshells.len(), 29);

        // Zero → None (invalid EADL designator).
        assert!(elem.subshell_by_eadl_designator(0).is_none());

        // K=1, L1=2, L2=3, L3=4 are the first four.
        assert_eq!(elem.subshell_by_eadl_designator(1).unwrap().designator, "K");
        assert_eq!(
            elem.subshell_by_eadl_designator(2).unwrap().designator,
            "L1"
        );
        assert_eq!(
            elem.subshell_by_eadl_designator(3).unwrap().designator,
            "L2"
        );
        assert_eq!(
            elem.subshell_by_eadl_designator(4).unwrap().designator,
            "L3"
        );

        // Q1 is the 29th (last) subshell in U.
        assert_eq!(
            elem.subshell_by_eadl_designator(29).unwrap().designator,
            "Q1"
        );

        // Beyond 29: outside U's PE tabulation, returns None.
        assert!(elem.subshell_by_eadl_designator(30).is_none());
        assert!(elem.subshell_by_eadl_designator(37).is_none());
        assert!(elem.subshell_by_eadl_designator(u32::MAX).is_none());
    }

    /// Every EADL transition in Uranium whose `primary` designator is
    /// within the PE subshell list (≤ 29) must resolve to a Subshell.
    /// Primaries that exceed 29 must not (and the cascade kernel will
    /// handle them by local deposition).
    #[test]
    fn uranium_transition_primaries_resolve_or_are_out_of_list() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load U");
        let n = elem.subshells.len() as u32;

        for s in &elem.subshells {
            for (i, t) in s.transitions.iter().enumerate() {
                let primary = t[0].round() as u32;
                let lookup = elem.subshell_by_eadl_designator(primary);
                if primary > 0 && primary <= n {
                    assert!(
                        lookup.is_some(),
                        "shell {} transition {i}: primary {primary} within [1, {n}] but lookup failed",
                        s.designator
                    );
                } else {
                    assert!(
                        lookup.is_none(),
                        "shell {} transition {i}: primary {primary} outside [1, {n}] but lookup succeeded",
                        s.designator
                    );
                }
            }
        }
    }

    /// For a closed-shell Hartree-Fock ground state J(p_z) is non-
    /// increasing in |p_z|. Verify on all U shells (monotonic test
    /// within the tabulated non-negative half).
    #[test]
    fn uranium_compton_profile_non_increasing() {
        let Some(path) = photon_path("U.h5") else {
            eprintln!("skipping: U.h5 not present");
            return;
        };
        let elem = PhotonElement::from_hdf5(&path).expect("load U");
        let cp = &elem.compton_profiles;
        for (shell, prof) in cp.j.iter().enumerate() {
            for w in prof.windows(2) {
                assert!(
                    w[1] <= w[0] + 1e-12,
                    "U shell {shell}: J increased from {} to {}",
                    w[0],
                    w[1]
                );
            }
        }
    }
}
