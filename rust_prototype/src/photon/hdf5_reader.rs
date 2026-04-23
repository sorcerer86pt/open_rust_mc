//! HDF5 reader for OpenMC photon-interaction files (`photon/<Sym>.h5`).
//!
//! Layout (verified against ENDF/B-VII.1 photon library, `filetype =
//! "data_photon"`, `version = [3, 0]`):
//!
//! ```text
//! /                                       attrs: filetype, version
//! /<Sym>                                  attrs: Z
//! /<Sym>/energy                           shared grid (eV), shape [N_E]
//! /<Sym>/coherent/xs                      shape [N_E]
//! /<Sym>/coherent/scattering_factor       shape [2, N_ff]  (x, F(x,Z))
//! /<Sym>/incoherent/xs                    shape [N_E]
//! /<Sym>/incoherent/scattering_factor     shape [2, N_sf]  (x, S(x,Z))
//! /<Sym>/photoelectric/xs                 shape [N_E]
//! /<Sym>/pair_production_nuclear/xs       shape [N_E]
//! /<Sym>/pair_production_electron/xs      shape [N_E]
//! /<Sym>/subshells                        attrs: designators
//! /<Sym>/subshells/<Shell>                attrs: binding_energy, num_electrons
//! /<Sym>/subshells/<Shell>/xs             shape [<= N_E] (aligned to tail)
//! /<Sym>/subshells/<Shell>/transitions    shape [N_t, 4]   (optional)
//! ```
//!
//! Coherent anomalous scattering factors, Compton profiles, and
//! bremsstrahlung data are present in the file but not loaded in Phase 1;
//! they are additive features for later physics work.

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
        let pair_production_electron_xs =
            read_xs(&element, "pair_production_electron", n_e, path)?;

        let coherent_form_factor =
            read_scattering_factor(&element, "coherent", "scattering_factor", path)?;
        let coherent_integrated_form_factor = read_scattering_factor(
            &element,
            "coherent",
            "integrated_scattering_factor",
            path,
        )?;
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

    let subshells_group = match element.group("subshells") {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };

    // Prefer the `designators` attribute for ordering; fall back to groups().
    let designators: Vec<String> = match subshells_group
        .attrs()
        .map_err(|e| hdf5_err(format!("cannot read subshells attrs: {e}")))?
        .get("designators")
    {
        Some(hdf5_pure::AttrValue::StringArray(arr)) => arr.clone(),
        _ => subshells_group
            .groups()
            .map_err(|e| hdf5_err(format!("cannot list subshells: {e}")))?,
    };

    let mut out = Vec::with_capacity(designators.len());
    for designator in designators {
        let shell = match subshells_group.group(&designator) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let shell_attrs = shell
            .attrs()
            .map_err(|e| hdf5_err(format!("cannot read subshell {designator} attrs: {e}")))?;
        let binding_energy = match shell_attrs.get("binding_energy") {
            Some(hdf5_pure::AttrValue::F64(v)) => *v,
            _ => 0.0,
        };
        let num_electrons = match shell_attrs.get("num_electrons") {
            Some(hdf5_pure::AttrValue::F64(v)) => *v,
            _ => 0.0,
        };
        let xs = shell
            .dataset("xs")
            .and_then(|ds| ds.read_f64())
            .unwrap_or_default();
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

fn read_anomalous_factors(
    element: &hdf5_pure::Group,
    path: &Path,
) -> Result<AnomalousFactors> {
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

fn read_compton_profiles(
    element: &hdf5_pure::Group,
    path: &Path,
) -> Result<ComptonProfiles> {
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

fn read_bremsstrahlung(
    element: &hdf5_pure::Group,
    path: &Path,
) -> Result<Bremsstrahlung> {
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

    fn carbon_path() -> Option<PathBuf> {
        // Walk up from CARGO_MANIFEST_DIR (rust_prototype/) to project root.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon/C.h5");
        if path.exists() { Some(path) } else { None }
    }

    #[test]
    fn load_carbon_photon_data() {
        let Some(path) = carbon_path() else {
            eprintln!("skipping: data/endfb-vii.1-hdf5/photon/C.h5 not present");
            return;
        };

        let elem = PhotonElement::from_hdf5(&path).expect("load carbon photon data");
        assert_eq!(elem.z, 6);
        assert_eq!(elem.symbol, "C");
        assert!(!elem.energy.is_empty());

        let n = elem.n_energy();
        assert_eq!(elem.coherent_xs.len(), n);
        assert_eq!(elem.incoherent_xs.len(), n);
        assert_eq!(elem.photoelectric_xs.len(), n);
        assert_eq!(elem.pair_production_nuclear_xs.len(), n);
        assert_eq!(elem.pair_production_electron_xs.len(), n);

        // Energy grid is strictly increasing.
        for w in elem.energy.windows(2) {
            assert!(w[1] > w[0], "energy grid not strictly increasing");
        }

        // Cross sections are non-negative.
        for xs in &elem.coherent_xs {
            assert!(*xs >= 0.0);
        }
        for xs in &elem.photoelectric_xs {
            assert!(*xs >= 0.0);
        }

        // Pair production has a threshold at 2 m_e c² ≈ 1.022 MeV.
        let below = elem
            .energy
            .iter()
            .position(|e| *e > 1.0e6)
            .map(|i| elem.pair_production_nuclear_xs[i.saturating_sub(1)]);
        if let Some(xs_below_threshold) = below {
            assert_eq!(xs_below_threshold, 0.0, "pair production below 1 MeV");
        }

        // Carbon has K, L1, L2, L3 shells.
        assert!(elem.subshells.len() >= 4);
        let designators: Vec<&str> =
            elem.subshells.iter().map(|s| s.designator.as_str()).collect();
        assert!(designators.contains(&"K"));

        // K-shell binding energy of carbon is 291 eV.
        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        assert!(
            (k.binding_energy - 291.0).abs() < 1.0,
            "K-shell binding energy = {} eV, expected ~291",
            k.binding_energy
        );
        assert!((k.num_electrons - 2.0).abs() < 1e-6);

        // Scattering factors monotonic in x.
        for w in elem.coherent_form_factor.x.windows(2) {
            assert!(w[1] > w[0]);
        }
        for w in elem.incoherent_scattering_factor.x.windows(2) {
            assert!(w[1] > w[0]);
        }

        // Integrated form factor is non-decreasing (it is a cumulative of F²).
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

        // Anomalous factors present and sized consistently.
        let af = &elem.coherent_anomalous;
        assert_eq!(af.real.grid.len(), af.real.value.len());
        assert_eq!(af.imag.grid.len(), af.imag.value.len());
        assert!(!af.real.grid.is_empty());

        // Compton profiles present; Carbon has 3 shells in the Compton
        // tabulation (per inspection: shape (3, 31)).
        let cp = &elem.compton_profiles;
        assert_eq!(cp.n_shells(), 3);
        assert_eq!(cp.n_pz(), 31);
        assert_eq!(cp.j.len(), 3);
        for row in &cp.j {
            assert_eq!(row.len(), 31);
        }
        // pz grid ascends from 0.
        assert!(cp.pz[0] == 0.0 || cp.pz[0].abs() < 1e-12);
        for w in cp.pz.windows(2) {
            assert!(w[1] > w[0]);
        }
        // Each profile J_i(p_z) is non-negative.
        for row in &cp.j {
            for v in row {
                assert!(*v >= 0.0);
            }
        }

        // Bremsstrahlung: 200 electron-energy × 30 photon-energy grid.
        let br = &elem.bremsstrahlung;
        assert_eq!(br.electron_energy.len(), 200);
        assert_eq!(br.photon_energy.len(), 30);
        assert_eq!(br.dcs.len(), 200);
        for row in &br.dcs {
            assert_eq!(row.len(), 30);
        }
        assert!(br.mean_excitation_energy > 0.0); // Carbon: I ≈ 81 eV
        assert!((br.mean_excitation_energy - 81.0).abs() < 1.0);
        assert_eq!(br.ionization_energy.len(), br.num_electrons.len());

        // Subshell tail-alignment convention.
        let k = elem.subshells.iter().find(|s| s.designator == "K").unwrap();
        let n = elem.n_energy();
        let offset = n - k.xs.len();
        assert_eq!(
            k.xs_at(n, offset),
            k.xs[0],
            "tail alignment off at offset"
        );
        assert_eq!(
            k.xs_at(n, n - 1),
            *k.xs.last().unwrap(),
            "tail alignment off at last point"
        );
        assert_eq!(k.xs_at(n, offset.saturating_sub(1)), 0.0);
    }
}
