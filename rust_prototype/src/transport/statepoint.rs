//! HDF5 statepoint = analysis + restart snapshot. Datasets:
//! `/header` scalars, `/k_eff`, `/k_track`,
//! `/source_bank/{positions,energies,weights}`,
//! `/tallies/{surface_current_pos|neg, mesh_flux}`.
//!
//! Standard HDF5; readable by h5py / OpenMC tools. Round-trip via
//! `hdf5_pure::File::from_bytes`.

use std::path::Path;

use hdf5_pure::{AttrValue, FileBuilder};

use crate::geometry::Vec3;
use crate::transport::particle::FissionSite;
use crate::transport::simulate::BatchResult;

pub struct StatepointInputs<'a> {
    pub batches: &'a [BatchResult],
    pub source_bank: &'a [FissionSite],
    pub n_active: u32,
    pub particles_per_batch: u32,
    pub seed: u64,
    pub k_eff_mean: f64,
    /// 0 = disabled.
    pub n_surface_bins: usize,
    /// 0 = disabled.
    pub n_mesh_voxels: usize,
}

/// Overwrites existing file.
pub fn write_statepoint(path: &Path, sp: &StatepointInputs<'_>) -> std::io::Result<()> {
    let bytes = build_statepoint(sp)
        .map_err(|e| std::io::Error::other(format!("hdf5 build failed: {e:?}")))?;
    std::fs::write(path, bytes)
}

#[derive(Debug, Clone)]
pub struct StatepointHeader {
    pub n_batches: u64,
    pub n_active: u64,
    pub particles_per_batch: u64,
    pub seed: u64,
    pub k_eff_mean: f64,
    pub n_surface_bins: u64,
    pub n_mesh_voxels: u64,
}

/// Restart-payload-only reader; cheaper than `read_statepoint`.
pub fn read_source_bank(path: &Path) -> std::io::Result<Vec<FissionSite>> {
    let file = open_statepoint(path)?;
    let pos = read_dataset_f64(&file, "source_bank_positions")?;
    let energy = read_dataset_f64(&file, "source_bank_energies")?;
    let weight = read_dataset_f64(&file, "source_bank_weights")?;
    if pos.len() != 3 * energy.len() || energy.len() != weight.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "source_bank shape mismatch: pos={}, energy={}, weight={}",
                pos.len(),
                energy.len(),
                weight.len()
            ),
        ));
    }
    let mut bank = Vec::with_capacity(energy.len());
    for i in 0..energy.len() {
        bank.push(FissionSite {
            pos: Vec3::new(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]),
            energy: energy[i],
            weight: weight[i],
        });
    }
    Ok(bank)
}

/// Scalar metadata only.
pub fn read_header(path: &Path) -> std::io::Result<StatepointHeader> {
    let file = open_statepoint(path)?;
    let attrs = file.root().attrs().map_err(io_err)?;
    let u64_attr = |name: &str| -> std::io::Result<u64> {
        match attrs.get(name) {
            Some(AttrValue::U64(v)) => Ok(*v),
            Some(AttrValue::I64(v)) => Ok(*v as u64),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("attr `{name}` missing or wrong type: {other:?}"),
            )),
        }
    };
    let f64_attr = |name: &str| -> std::io::Result<f64> {
        match attrs.get(name) {
            Some(AttrValue::F64(v)) => Ok(*v),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("attr `{name}` missing or wrong type: {other:?}"),
            )),
        }
    };
    Ok(StatepointHeader {
        n_batches: u64_attr("n_batches")?,
        n_active: u64_attr("n_active")?,
        particles_per_batch: u64_attr("particles_per_batch")?,
        seed: u64_attr("seed")?,
        k_eff_mean: f64_attr("k_eff_mean")?,
        n_surface_bins: u64_attr("n_surface_bins")?,
        n_mesh_voxels: u64_attr("n_mesh_voxels")?,
    })
}

fn open_statepoint(path: &Path) -> std::io::Result<hdf5_pure::File> {
    let bytes = std::fs::read(path)?;
    hdf5_pure::File::from_bytes(bytes).map_err(io_err)
}

fn read_dataset_f64(file: &hdf5_pure::File, name: &str) -> std::io::Result<Vec<f64>> {
    file.dataset(name)
        .map_err(io_err)?
        .read_f64()
        .map_err(io_err)
}

fn io_err<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("hdf5: {e:?}"))
}

/// In-memory variant of `write_statepoint`; used by tests + FFI.
pub fn build_statepoint(sp: &StatepointInputs<'_>) -> Result<Vec<u8>, hdf5_pure::Error> {
    let mut fb = FileBuilder::new();

    fb.set_attr("n_batches", AttrValue::U64(sp.batches.len() as u64));
    fb.set_attr("n_active", AttrValue::U64(sp.n_active as u64));
    fb.set_attr(
        "particles_per_batch",
        AttrValue::U64(sp.particles_per_batch as u64),
    );
    fb.set_attr("seed", AttrValue::U64(sp.seed));
    fb.set_attr("k_eff_mean", AttrValue::F64(sp.k_eff_mean));
    fb.set_attr("n_surface_bins", AttrValue::U64(sp.n_surface_bins as u64));
    fb.set_attr("n_mesh_voxels", AttrValue::U64(sp.n_mesh_voxels as u64));

    let n_b = sp.batches.len();

    let k_collision: Vec<f64> = sp.batches.iter().map(|b| b.k_eff).collect();
    let k_track: Vec<f64> = sp.batches.iter().map(|b| b.k_track).collect();
    let entropy: Vec<f64> = sp.batches.iter().map(|b| b.shannon_entropy).collect();
    let active_flag: Vec<i64> = sp
        .batches
        .iter()
        .map(|b| if b.active { 1 } else { 0 })
        .collect();

    fb.create_dataset("k_collision_per_batch")
        .with_f64_data(&k_collision)
        .with_shape(&[n_b as u64]);
    fb.create_dataset("k_track_per_batch")
        .with_f64_data(&k_track)
        .with_shape(&[n_b as u64]);
    fb.create_dataset("shannon_entropy_per_batch")
        .with_f64_data(&entropy)
        .with_shape(&[n_b as u64]);
    fb.create_dataset("active_per_batch")
        .with_i64_data(&active_flag)
        .with_shape(&[n_b as u64]);

    let n_src = sp.source_bank.len();
    let mut positions: Vec<f64> = Vec::with_capacity(3 * n_src);
    let mut energies: Vec<f64> = Vec::with_capacity(n_src);
    let mut weights: Vec<f64> = Vec::with_capacity(n_src);
    for site in sp.source_bank {
        positions.push(site.pos.x);
        positions.push(site.pos.y);
        positions.push(site.pos.z);
        energies.push(site.energy);
        weights.push(site.weight);
    }
    fb.create_dataset("source_bank_positions")
        .with_f64_data(&positions)
        .with_shape(&[n_src as u64, 3]);
    fb.create_dataset("source_bank_energies")
        .with_f64_data(&energies)
        .with_shape(&[n_src as u64]);
    fb.create_dataset("source_bank_weights")
        .with_f64_data(&weights)
        .with_shape(&[n_src as u64]);

    if sp.n_surface_bins > 0 {
        let total = n_b * sp.n_surface_bins;
        let mut pos_flat = Vec::with_capacity(total);
        let mut neg_flat = Vec::with_capacity(total);
        // Pad-zero on short rows (defensive; well-formed runs match).
        for b in sp.batches {
            for i in 0..sp.n_surface_bins {
                pos_flat.push(*b.tallies.surface_current_pos.get(i).unwrap_or(&0.0));
                neg_flat.push(*b.tallies.surface_current_neg.get(i).unwrap_or(&0.0));
            }
        }
        fb.create_dataset("surface_current_pos")
            .with_f64_data(&pos_flat)
            .with_shape(&[n_b as u64, sp.n_surface_bins as u64]);
        fb.create_dataset("surface_current_neg")
            .with_f64_data(&neg_flat)
            .with_shape(&[n_b as u64, sp.n_surface_bins as u64]);
    }

    // Mesh flux tally (only if enabled)
    if sp.n_mesh_voxels > 0 {
        let total = n_b * sp.n_mesh_voxels;
        let mut flux_flat = Vec::with_capacity(total);
        for b in sp.batches {
            for i in 0..sp.n_mesh_voxels {
                flux_flat.push(*b.tallies.mesh_flux.get(i).unwrap_or(&0.0));
            }
        }
        fb.create_dataset("mesh_flux")
            .with_f64_data(&flux_flat)
            .with_shape(&[n_b as u64, sp.n_mesh_voxels as u64]);
    }

    fb.finish()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default
)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::transport::simulate::PhotonSourceEvent;

    fn dummy_batch(batch: u32, k: f64, n_surf: usize, n_vox: usize) -> BatchResult {
        BatchResult {
            batch,
            k_eff: k,
            leakage: 0,
            absorptions: 0,
            fissions: 0,
            collisions: 0,
            thermal_scatters: 0,
            surface_crossings: 0,
            shannon_entropy: 7.95,
            active: true,
            captures_by_cell: vec![],
            photon_events: Vec::<PhotonSourceEvent>::new(),
            k_track: k - 0.001,
            tallies: crate::transport::tally::BatchTallies {
                surface_current_pos: vec![0.5; n_surf],
                surface_current_neg: vec![0.3; n_surf],
                mesh_flux: vec![1.25; n_vox],
                rr_flux: vec![],
                rr_rate: vec![],
            },
            n_elastic: 0,
            n_inelastic: 0,
            n_capture: 0,
            e_fis_in_sum: 0.0,
            e_el_in_sum: 0.0,
            e_inel_in_sum: 0.0,
            e_inel_out_sum: 0.0,
            e_fis_in_sq_sum: 0.0,
            e_el_in_sq_sum: 0.0,
            e_inel_in_sq_sum: 0.0,
            q_inel_sum: 0.0,
        }
    }

    #[test]
    fn statepoint_roundtrip_minimal() {
        let batches = vec![
            dummy_batch(1, 0.999, 0, 0),
            dummy_batch(2, 1.001, 0, 0),
            dummy_batch(3, 1.000, 0, 0),
        ];
        let bank = vec![
            FissionSite {
                pos: Vec3::new(0.1, 0.2, 0.3),
                energy: 1e6,
                weight: 1.0,
            },
            FissionSite {
                pos: Vec3::new(-0.1, 0.0, 0.5),
                energy: 2e6,
                weight: 1.0,
            },
        ];
        let sp = StatepointInputs {
            batches: &batches,
            source_bank: &bank,
            n_active: 2,
            particles_per_batch: 5000,
            seed: 42,
            k_eff_mean: 1.000,
            n_surface_bins: 0,
            n_mesh_voxels: 0,
        };
        let bytes = build_statepoint(&sp).expect("build");
        let f = hdf5_pure::File::from_bytes(bytes).expect("read");

        let k = f
            .dataset("k_collision_per_batch")
            .unwrap()
            .read_f64()
            .unwrap();
        assert_eq!(k.len(), 3);
        assert!((k[0] - 0.999).abs() < 1e-12);
        assert!((k[2] - 1.000).abs() < 1e-12);

        let kt = f.dataset("k_track_per_batch").unwrap().read_f64().unwrap();
        assert!((kt[0] - 0.998).abs() < 1e-12);

        let pos = f
            .dataset("source_bank_positions")
            .unwrap()
            .read_f64()
            .unwrap();
        assert_eq!(pos.len(), 6);
        assert!((pos[0] - 0.1).abs() < 1e-12);
        assert!((pos[5] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn read_header_and_source_bank_via_disk_roundtrip() {
        let batches = vec![dummy_batch(1, 1.0, 0, 0)];
        let bank = vec![
            FissionSite {
                pos: Vec3::new(1.0, 2.0, 3.0),
                energy: 1e6,
                weight: 1.0,
            },
            FissionSite {
                pos: Vec3::new(-1.5, 0.0, 2.5),
                energy: 5e5,
                weight: 0.5,
            },
        ];
        let sp = StatepointInputs {
            batches: &batches,
            source_bank: &bank,
            n_active: 1,
            particles_per_batch: 100,
            seed: 7,
            k_eff_mean: 0.99,
            n_surface_bins: 0,
            n_mesh_voxels: 0,
        };
        let path = std::env::temp_dir().join("orm_statepoint_roundtrip.h5");
        write_statepoint(&path, &sp).expect("write");

        let header = read_header(&path).expect("read header");
        assert_eq!(header.n_batches, 1);
        assert_eq!(header.particles_per_batch, 100);
        assert_eq!(header.seed, 7);
        assert!((header.k_eff_mean - 0.99).abs() < 1e-12);

        let loaded = read_source_bank(&path).expect("read bank");
        assert_eq!(loaded.len(), 2);
        assert!((loaded[0].pos.x - 1.0).abs() < 1e-12);
        assert!((loaded[0].energy - 1e6).abs() < 1e-12);
        assert!((loaded[1].pos.z - 2.5).abs() < 1e-12);
        assert!((loaded[1].weight - 0.5).abs() < 1e-12);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn statepoint_roundtrip_with_tallies() {
        let n_surf = 2;
        let n_vox = 4;
        let batches = vec![
            dummy_batch(1, 0.99, n_surf, n_vox),
            dummy_batch(2, 1.01, n_surf, n_vox),
        ];
        let sp = StatepointInputs {
            batches: &batches,
            source_bank: &[],
            n_active: 2,
            particles_per_batch: 1000,
            seed: 1,
            k_eff_mean: 1.0,
            n_surface_bins: n_surf,
            n_mesh_voxels: n_vox,
        };
        let bytes = build_statepoint(&sp).expect("build");
        let f = hdf5_pure::File::from_bytes(bytes).expect("read");

        let pos = f
            .dataset("surface_current_pos")
            .unwrap()
            .read_f64()
            .unwrap();
        assert_eq!(pos.len(), 2 * n_surf);
        // Every dummy batch wrote 0.5 into every bin.
        for v in &pos {
            assert!((v - 0.5).abs() < 1e-12);
        }

        let mesh = f.dataset("mesh_flux").unwrap().read_f64().unwrap();
        assert_eq!(mesh.len(), 2 * n_vox);
        for v in &mesh {
            assert!((v - 1.25).abs() < 1e-12);
        }
    }
}
