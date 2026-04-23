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
use crate::photon::data::{PhotonElement, ScatteringFactor, Subshell};

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
        let incoherent_scattering_factor =
            read_scattering_factor(&element, "incoherent", "scattering_factor", path)?;

        let subshells = read_subshells(&element, path)?;

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
            incoherent_scattering_factor,
            subshells,
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
    }
}
