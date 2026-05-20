// SPDX-License-Identifier: MIT
//! Flux extractor — bridges `transport::tally::MeshFluxTally`
//! results into a one-group flux scalar that the depletion matrix
//! consumes.
//!
//! Two-step convention used here, matching standard depletion
//! pipelines (OpenMC, Serpent):
//!
//!  1. Run an eigenvalue solve with a `MeshFluxTally` covering the
//!     burnable region. Per active batch, the engine accumulates
//!     `Σ d_voxel · w` (track length × weight) per voxel.
//!  2. After the run, sum over active batches and divide by
//!     (`active_batches × particles_per_batch × voxel_volume`) to
//!     get **flux per source neutron** in `cm⁻²`.
//!  3. Scale by the source rate `Q` (neutrons emitted per second by
//!     the system) to get physical flux in `n / (cm² · s)`.
//!
//! Source rate `Q` follows from a target power `P` and the average
//! energy per fission `E_f`:
//!   `Q = P / (E_f · F)`,
//! where `F` is the active-batch-mean fission count per source
//! neutron (≈ k_eff for a converged source). `power_normalized_source`
//! computes `Q` directly from the run's batch results.
//!
//! For the simple Xe-poisoning demo this is overkill — caller can
//! also just specify a target physical flux and skip the power
//! normalization.

use crate::transport::simulate::BatchResult;
use crate::transport::tally::MeshFluxTally;

/// Mean energy released per fission (J). 200 MeV is the textbook
/// value; total recoverable energy is closer to 207 MeV when
/// gamma + neutrino contributions are included, but 200 MeV is
/// standard for one-group depletion bookkeeping.
pub const E_PER_FISSION_J: f64 = 200.0e6 * 1.602_176_634e-19;

/// Sum the mesh-flux tally over active batches, divide by total
/// active source particles, and divide by per-voxel volume. Returns
/// per-voxel flux **per source neutron** in units of `cm⁻²` —
/// dimensionless on the source side, so the caller scales by the
/// source rate `Q [n/s]` to get physical flux `[n/(cm²·s)]`.
pub fn voxel_flux_per_source(batches: &[BatchResult], mesh: &MeshFluxTally) -> Vec<f64> {
    let n_vox = mesh.n_voxels();
    let mut flux = vec![0.0_f64; n_vox];
    let mut n_active = 0_u64;
    let mut n_per_batch = 0_u64;
    for r in batches {
        if !r.active {
            continue;
        }
        n_active += 1;
        // Track lengths per voxel summed by the engine across all
        // particles in this batch — dimension `cm × weight`.
        for (out, &v) in flux.iter_mut().zip(r.tallies.mesh_flux.iter()) {
            *out += v;
        }
        // Source-particle count per batch. The engine writes
        // `r.fissions` etc. as totals, but we need `particles_per_batch`
        // to normalize. Recover it as the maximum over `n_active`-th
        // batches (every batch transports the same number of
        // particles in the eigenvalue loop).
        let est = (r.fissions as u64) + (r.absorptions as u64) + (r.leakage as u64);
        if est > n_per_batch {
            n_per_batch = est;
        }
    }
    if n_active == 0 || n_per_batch == 0 {
        return flux;
    }
    let v_voxel = mesh.spacing[0] * mesh.spacing[1] * mesh.spacing[2];
    let denom = (n_active * n_per_batch) as f64 * v_voxel;
    for f in flux.iter_mut() {
        *f /= denom;
    }
    flux
}

/// Mean flux per source neutron over the whole tally region (sum of
/// per-voxel `flux × volume` divided by total volume). Suitable for
/// a homogeneous burnable cell occupying the entire mesh.
pub fn mean_flux_per_source(batches: &[BatchResult], mesh: &MeshFluxTally) -> f64 {
    let voxels = voxel_flux_per_source(batches, mesh);
    if voxels.is_empty() {
        return 0.0;
    }
    let v_voxel = mesh.spacing[0] * mesh.spacing[1] * mesh.spacing[2];
    let total: f64 = voxels.iter().map(|&phi| phi * v_voxel).sum();
    total / (voxels.len() as f64 * v_voxel)
}

/// Power-normalized source rate `Q` in `n/s` for a system that
/// produces `target_power_w` watts of fission heat. `fission_volume_cm3`
/// is the volume in which the fissions occur (the tally region for
/// power accounting); `mean_fission_per_source` is the active-batch
/// mean of `r.fissions / particles_per_batch` (≈ k_eff for converged
/// source).
pub fn power_normalized_source(
    target_power_w: f64,
    mean_fission_per_source: f64,
    e_per_fission_j: f64,
) -> f64 {
    if mean_fission_per_source <= 0.0 {
        return 0.0;
    }
    target_power_w / (mean_fission_per_source * e_per_fission_j)
}

/// Active-batch mean of `r.fissions / particles_per_batch`. This is
/// the effective `<ν Σ_f φ V> / <Σ_t φ V>`-style ratio expressed per
/// source neutron. Used by `power_normalized_source` to convert
/// power → source rate.
pub fn mean_fissions_per_source(batches: &[BatchResult]) -> f64 {
    let mut n_active = 0_u64;
    let mut sum_fissions = 0_u64;
    let mut n_per_batch = 0_u64;
    for r in batches {
        if !r.active {
            continue;
        }
        n_active += 1;
        sum_fissions += r.fissions as u64;
        let est = (r.fissions as u64) + (r.absorptions as u64) + (r.leakage as u64);
        if est > n_per_batch {
            n_per_batch = est;
        }
    }
    if n_active == 0 || n_per_batch == 0 {
        return 0.0;
    }
    (sum_fissions as f64) / ((n_active * n_per_batch) as f64)
}

/// Collapse the per-(cell, xs_idx, MT) reaction-rate tally over
/// active batches into a one-group `<σ>` (barns) per nuclide-MT for
/// a single target cell. Used by `deplete_pwr` to override the
/// thermal-spectrum chain XS with the actual cell-flux-spectrum-
/// averaged value at every depletion step.
///
/// Returns a `Vec<((xs_idx, mt), <σ_barns>)>` for every
/// `(xs_idx, mt)` slot whose tally has non-zero flux.
pub fn collapsed_reaction_xs(
    batches: &[crate::transport::simulate::BatchResult],
    rr_template: &crate::transport::tally::ReactionRateTally,
    target_cell: usize,
) -> Vec<((usize, u32), f64)> {
    let n_xs_idx = rr_template.n_xs_idx;
    let n_mts = rr_template.n_mts;
    let n_cells = rr_template.n_cells;
    let stride = n_xs_idx * n_mts;
    if target_cell >= n_cells {
        return Vec::new();
    }

    let mut flux_sum = 0.0_f64;
    let mut rate_sum = vec![0.0_f64; stride];
    let mut n_active = 0usize;
    for b in batches {
        if !b.active {
            continue;
        }
        if b.tallies.rr_flux.len() != n_cells
            || b.tallies.rr_rate.len() != n_cells * stride
        {
            continue;
        }
        n_active += 1;
        flux_sum += b.tallies.rr_flux[target_cell];
        let base = target_cell * stride;
        for (acc, &v) in rate_sum.iter_mut().zip(&b.tallies.rr_rate[base..base + stride]) {
            *acc += v;
        }
    }
    if n_active == 0 || flux_sum <= 0.0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(stride);
    for xs_idx in 0..n_xs_idx {
        for (m, &mt) in rr_template.mts.iter().enumerate() {
            let r = rate_sum[xs_idx * n_mts + m];
            if r <= 0.0 {
                continue;
            }
            out.push(((xs_idx, mt), r / flux_sum));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::tally::MeshFluxTally;

    fn synth_batch(active: bool, fissions: u32, mesh_flux: Vec<f64>) -> BatchResult {
        BatchResult {
            batch: 1,
            k_eff: 1.0,
            leakage: 100,
            absorptions: 50,
            fissions,
            collisions: 1000,
            thermal_scatters: 0,
            surface_crossings: 0,
            shannon_entropy: 0.0,
            active,
            captures_by_cell: vec![],
            photon_events: vec![],
            k_track: 0.0,
            tallies: crate::transport::tally::BatchTallies {
                surface_current_pos: vec![],
                surface_current_neg: vec![],
                mesh_flux,
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
    fn mean_flux_per_source_uniform_box() {
        let mesh = MeshFluxTally::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2, 2, 2]);
        // Each batch tallies 8 voxels × 1.5 cm track length per voxel.
        let per_voxel: Vec<f64> = vec![1.5_f64; 8];
        let batches = vec![
            synth_batch(false, 800, per_voxel.clone()), // inactive — ignored
            synth_batch(true, 800, per_voxel.clone()),
            synth_batch(true, 800, per_voxel.clone()),
        ];
        // n_per_batch deduced from fissions+absorptions+leakage = 800+50+100 = 950.
        // Per-voxel flux per source = sum_active / (n_active × n_per_batch × V_voxel)
        //   = (1.5 × 2) / (2 × 950 × 1) = 1.5 / 950
        // mean_flux_per_source averages per-voxel flux over voxels (uniform → same value).
        let phi = mean_flux_per_source(&batches, &mesh);
        let expected = 1.5 / 950.0;
        assert!(
            (phi - expected).abs() / expected < 1e-12,
            "got {phi}, expected {expected}",
        );
    }

    #[test]
    fn power_normalization_scales_correctly() {
        // Target 1 watt, with one fission per source neutron and
        // 200 MeV per fission. Q must equal 1 / (1 × 200 MeV in J).
        let q = power_normalized_source(1.0, 1.0, E_PER_FISSION_J);
        let expected = 1.0 / E_PER_FISSION_J;
        assert!((q - expected).abs() / expected < 1e-15);
    }

    #[test]
    fn mean_fissions_per_source_recovers_input_value() {
        // 800 fissions per batch with 950 particles per batch
        // (deduced from totals) → mean = 800/950.
        let batches = vec![
            synth_batch(true, 800, vec![]),
            synth_batch(true, 800, vec![]),
        ];
        let f = mean_fissions_per_source(&batches);
        let expected = 800.0 / 950.0;
        assert!(
            (f - expected).abs() / expected < 1e-12,
            "got {f}, expected {expected}",
        );
    }
}
