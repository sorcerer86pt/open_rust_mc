//! Event-based GPU neutron transport.
//!
//! CUDA kernel source in `gpu/cuda/transport.cu`, loaded via `include_str!`.
//! Uses packed `TransportParams` struct — all read-only data in one device buffer.
//! Persistent kernel with warp-level reductions and energy-sorted compaction.
//! Full physics parity with CPU: SVD XS, S(α,β), discrete levels, URR, angular dist.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc;

/// Number of u64 fields in the packed TransportParams buffer.
/// Must match N_PARAMS in transport.cu.
const N_PARAMS: usize = 66;

// ── CUDA kernel source ────────────────────────────────────────────────

/// All CUDA kernels for event-based transport.
///
/// PWR pin cell geometry is hardcoded (9 surfaces, 4 cells, 3 materials).
/// SVD basis data is passed via global memory, coefficients via shared memory.
const TRANSPORT_KERNELS: &str = include_str!("../gpu/cuda/transport.cu");


// ── Rust-side GPU transport context ──────────────────────────────

/// Compiled CUDA kernels for event-based transport.
pub struct GpuTransportContext {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    k_init_source: CudaFunction,
    k_count_alive: CudaFunction,
    k_compact_alive: CudaFunction,
    k_energy_bin_count: CudaFunction,
    k_energy_bin_scatter: CudaFunction,
    k_transport_persistent: CudaFunction,
}

/// SVD data + physics tables uploaded to GPU for all nuclides.
pub struct GpuNuclideData {
    // SVD basis data
    pub all_basis: CudaSlice<f32>,
    pub all_coeffs: CudaSlice<f64>,
    pub all_energy_grids: CudaSlice<f64>,
    pub basis_offsets: CudaSlice<i32>,
    pub grid_offsets: CudaSlice<i32>,
    pub n_energies: CudaSlice<i32>,
    pub has_reaction: CudaSlice<i32>,
    pub coeffs_offsets: CudaSlice<i32>,
    pub rank: i32,
    // Energy-dependent nu-bar tables
    pub nu_bar_energies: CudaSlice<f64>,
    pub nu_bar_values: CudaSlice<f64>,
    pub nu_bar_offsets: CudaSlice<i32>,
    pub nu_bar_sizes: CudaSlice<i32>,
    // Discrete inelastic levels (Q-values + SVD basis for XS-proportional sampling)
    pub level_q_values: CudaSlice<f64>,      // flat: all Q-values concatenated
    pub level_thresholds: CudaSlice<f64>,    // flat: all thresholds concatenated
    pub level_offsets: CudaSlice<i32>,        // per-nuclide offset into level arrays
    pub level_counts: CudaSlice<i32>,         // per-nuclide number of levels
    pub level_basis: CudaSlice<f32>,          // flat: SVD basis for each level's XS
    pub level_coeffs: CudaSlice<f64>,         // flat: SVD coefficients for each level
    pub level_basis_offsets: CudaSlice<i32>,  // per-level offset into level_basis
    pub level_coeffs_offsets: CudaSlice<i32>, // per-level offset into level_coeffs
    pub level_has_kernel: CudaSlice<i32>,     // per-level: 1 if kernel exists, 0 if not
    pub level_mt: CudaSlice<i32>,             // per-level: MT number (51-91)
    // Anisotropic elastic scattering angular distributions
    pub ang_energies: CudaSlice<f64>,         // flat: energy grids for angular dist
    pub ang_mu: CudaSlice<f64>,               // flat: cosine values
    pub ang_cdf: CudaSlice<f64>,              // flat: CDF values
    pub ang_dist_offsets: CudaSlice<i32>,      // per (nuc, energy) → offset into mu/cdf
    pub ang_dist_sizes: CudaSlice<i32>,        // per (nuc, energy) → n_mu
    pub ang_nuc_offsets: CudaSlice<i32>,       // per-nuclide → offset into ang_energies
    pub ang_nuc_n_energies: CudaSlice<i32>,    // per-nuclide → number of angular energies
    pub ang_is_cm: CudaSlice<i32>,             // per-nuclide → 1 if CM frame
    // Fission energy distributions (tabulated CDF)
    pub fis_inc_energies: CudaSlice<f64>,
    pub fis_dist_offsets: CudaSlice<i32>,
    pub fis_dist_sizes: CudaSlice<i32>,
    pub fis_e_out: CudaSlice<f64>,
    pub fis_cdf: CudaSlice<f64>,
    pub fis_nuc_offsets: CudaSlice<i32>,
    pub fis_nuc_n_inc: CudaSlice<i32>,
    // URR probability tables
    pub urr_energies: CudaSlice<f64>,
    pub urr_cum_prob: CudaSlice<f64>,
    pub urr_total_f: CudaSlice<f64>,
    pub urr_elastic_f: CudaSlice<f64>,
    pub urr_fission_f: CudaSlice<f64>,
    pub urr_capture_f: CudaSlice<f64>,
    pub urr_offsets: CudaSlice<i32>,
    pub urr_n_energies: CudaSlice<i32>,
    pub urr_n_bands: CudaSlice<i32>,
    pub urr_multiply_smooth: CudaSlice<i32>,
}

/// S(α,β) thermal scattering data on GPU (for one temperature).
pub struct GpuSabData {
    pub inc_energies: CudaSlice<f64>,
    pub n_inc: i32,
    pub eout_offsets: CudaSlice<i32>,
    pub eout_sizes: CudaSlice<i32>,
    pub e_out: CudaSlice<f64>,
    pub cdf_e: CudaSlice<f64>,
    pub mu_offsets: CudaSlice<i32>,
    pub mu_sizes: CudaSlice<i32>,
    pub mu: CudaSlice<f64>,
    pub cdf_mu: CudaSlice<f64>,
    pub xs: CudaSlice<f64>,
    pub energy_max: f64,
}

/// Material composition data on GPU.
pub struct GpuMaterialData {
    pub mat_n_nuclides: CudaSlice<i32>,
    pub mat_nuclide_idx: CudaSlice<i32>,
    pub mat_atom_density: CudaSlice<f64>,
    pub awr_table: CudaSlice<f64>,
    pub nu_bar_const: CudaSlice<f64>,
}

/// Result of one batch on GPU.
pub struct GpuBatchResult {
    pub k_eff: f64,
    pub collisions: u32,
    pub fissions: u32,
    pub leakage: u32,
    pub surface_crossings: u32,
    /// Fission sites for next generation.
    pub fission_bank: Vec<(f64, f64, f64, f64)>, // (x, y, z, energy)
}

const BLOCK_SIZE: u32 = 256;

impl GpuTransportContext {
    /// Compile all CUDA kernels and initialize GPU context.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let ptx = nvrtc::compile_ptx(TRANSPORT_KERNELS)?;
        let module = ctx.load_module(ptx)?;

        let k_init_source = module.load_function("init_source")?;
        let k_count_alive = module.load_function("count_alive")?;
        let k_compact_alive = module.load_function("compact_alive")?;
        let k_energy_bin_count = module.load_function("energy_bin_count")?;
        let k_energy_bin_scatter = module.load_function("energy_bin_scatter")?;
        let k_transport_persistent = module.load_function("transport_persistent")?;
        let stream = ctx.default_stream();

        println!("  GPU transport kernels compiled (8 kernels)");

        Ok(Self {
            _ctx: ctx, stream,
            k_init_source, k_count_alive, k_compact_alive,
            k_energy_bin_count, k_energy_bin_scatter,
            k_transport_persistent,
        })
    }

    /// Debug: sample angular distributions at given (energy, xi) pairs.
    /// Returns (stairstep_mu, interpolated_mu) for comparison with CPU.
    pub fn debug_angular_sample(
        &self,
        energies: &[f64],
        xis: &[f64],
        nuc_idx: i32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        geom_type: i32,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let n = energies.len();
        assert_eq!(n, xis.len());

        let d_energies = self.stream.clone_htod(energies)?;
        let d_xis = self.stream.clone_htod(xis)?;
        let mut d_out_ss: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_out_interp: CudaSlice<f64> = self.stream.alloc_zeros(n)?;

        // Build params buffer (same as run_batch)
        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr
            }};
        }
        let params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis), dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids), dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets), dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction), dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides), dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density), dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies), dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets), dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies), dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes), dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf), dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values), dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets), dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis), dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets), dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel), dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies), dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf), dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes), dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies), dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies), sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets), dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out), dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets), dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu), dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs), sab_data.energy_max.to_bits(),
            dptr!(&nuc_data.urr_energies), dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f), dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f), dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets), dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands), dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
        ];
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        // Load debug kernel
        let ptx = nvrtc::compile_ptx(TRANSPORT_KERNELS)?;
        let module = self._ctx.load_module(ptx)?;
        let k_debug = module.load_function("debug_angular_sample")?;

        let n_i32 = n as i32;
        let grid = ((n as u32 + 255) / 256, 1, 1);
        let block = (256u32, 1, 1);
        let cfg = cudarc::driver::LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };

        unsafe {
            self.stream.launch_builder(&k_debug)
                .arg(&d_params)
                .arg(&d_energies)
                .arg(&d_xis)
                .arg(&n_i32)
                .arg(&nuc_idx)
                .arg(&mut d_out_ss)
                .arg(&mut d_out_interp)
                .launch(cfg)?;
        }

        let ss = self.stream.clone_dtoh(&d_out_ss)?;
        let interp = self.stream.clone_dtoh(&d_out_interp)?;
        Ok((ss, interp))
    }

    /// Debug: reconstruct XS at given energies for a nuclide on GPU.
    /// Returns [n * 6] flat array: elastic, inelastic, n2n, n3n, fission, capture per energy.
    pub fn debug_xs_reconstruct(
        &self,
        energies: &[f64],
        nuc_idx: i32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        geom_type: i32,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = energies.len();
        let d_energies = self.stream.clone_htod(energies)?;
        let mut d_out: CudaSlice<f64> = self.stream.alloc_zeros(n * 6)?;

        macro_rules! dptr {
            ($slice:expr) => {{ let (ptr, _guard) = $slice.device_ptr(&self.stream); ptr }};
        }
        let params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis), dptr!(&nuc_data.all_coeffs),
            dptr!(&nuc_data.all_energy_grids), dptr!(&nuc_data.basis_offsets),
            dptr!(&nuc_data.grid_offsets), dptr!(&nuc_data.n_energies),
            dptr!(&nuc_data.has_reaction), dptr!(&nuc_data.coeffs_offsets),
            nuc_data.rank as u64,
            dptr!(&mat_data.mat_n_nuclides), dptr!(&mat_data.mat_nuclide_idx),
            dptr!(&mat_data.mat_atom_density), dptr!(&mat_data.awr_table),
            dptr!(&mat_data.nu_bar_const),
            dptr!(&nuc_data.nu_bar_energies), dptr!(&nuc_data.nu_bar_values),
            dptr!(&nuc_data.nu_bar_offsets), dptr!(&nuc_data.nu_bar_sizes),
            dptr!(&nuc_data.fis_inc_energies), dptr!(&nuc_data.fis_dist_offsets),
            dptr!(&nuc_data.fis_dist_sizes), dptr!(&nuc_data.fis_e_out),
            dptr!(&nuc_data.fis_cdf), dptr!(&nuc_data.fis_nuc_offsets),
            dptr!(&nuc_data.fis_nuc_n_inc),
            dptr!(&nuc_data.level_q_values), dptr!(&nuc_data.level_thresholds),
            dptr!(&nuc_data.level_offsets), dptr!(&nuc_data.level_counts),
            dptr!(&nuc_data.level_basis), dptr!(&nuc_data.level_coeffs),
            dptr!(&nuc_data.level_basis_offsets), dptr!(&nuc_data.level_coeffs_offsets),
            dptr!(&nuc_data.level_has_kernel), dptr!(&nuc_data.level_mt),
            dptr!(&nuc_data.ang_energies), dptr!(&nuc_data.ang_mu),
            dptr!(&nuc_data.ang_cdf), dptr!(&nuc_data.ang_dist_offsets),
            dptr!(&nuc_data.ang_dist_sizes), dptr!(&nuc_data.ang_nuc_offsets),
            dptr!(&nuc_data.ang_nuc_n_energies), dptr!(&nuc_data.ang_is_cm),
            dptr!(&sab_data.inc_energies), sab_data.n_inc as u64,
            dptr!(&sab_data.eout_offsets), dptr!(&sab_data.eout_sizes),
            dptr!(&sab_data.e_out), dptr!(&sab_data.cdf_e),
            dptr!(&sab_data.mu_offsets), dptr!(&sab_data.mu_sizes),
            dptr!(&sab_data.mu), dptr!(&sab_data.cdf_mu),
            dptr!(&sab_data.xs), sab_data.energy_max.to_bits(),
            dptr!(&nuc_data.urr_energies), dptr!(&nuc_data.urr_cum_prob),
            dptr!(&nuc_data.urr_total_f), dptr!(&nuc_data.urr_elastic_f),
            dptr!(&nuc_data.urr_fission_f), dptr!(&nuc_data.urr_capture_f),
            dptr!(&nuc_data.urr_offsets), dptr!(&nuc_data.urr_n_energies),
            dptr!(&nuc_data.urr_n_bands), dptr!(&nuc_data.urr_multiply_smooth),
            geom_type as u64,
        ];
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        let ptx = nvrtc::compile_ptx(TRANSPORT_KERNELS)?;
        let module = self._ctx.load_module(ptx)?;
        let k_debug = module.load_function("debug_xs_reconstruct")?;

        let n_i32 = n as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (((n as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&k_debug)
                .arg(&d_params)
                .arg(&d_energies)
                .arg(&n_i32)
                .arg(&nuc_idx)
                .arg(&mut d_out)
                .launch(cfg)?;
        }
        Ok(self.stream.clone_dtoh(&d_out)?)
    }

    /// Expose the CUDA stream for diagnostic buffer downloads.
    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }

    /// Upload SVD nuclide data to GPU.
    pub fn upload_nuclide_data(
        &self,
        nuclides: &[crate::transport::xs_provider::NuclideKernels],
        rank: usize,
    ) -> Result<GpuNuclideData, Box<dyn std::error::Error>> {
        let n_nuc = nuclides.len();
        let n_rxn = 6; // elastic, inelastic, n2n, n3n, fission, capture

        // Concatenate all basis, coefficients, and energy grids
        let mut all_basis_vec: Vec<f32> = Vec::new();
        let mut all_coeffs_vec: Vec<f64> = Vec::new();
        let mut all_grids_vec: Vec<f64> = Vec::new();
        let mut basis_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut coeffs_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut grid_offsets_vec = vec![0_i32; n_nuc];
        let mut n_energies_vec = vec![0_i32; n_nuc];
        let mut has_reaction_vec = vec![0_i32; n_nuc * n_rxn];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            // Energy grid (shared across reactions for this nuclide)
            let grid_offset = all_grids_vec.len();
            grid_offsets_vec[nuc_idx] = grid_offset as i32;

            // Get energy grid from any available reaction
            let any_kernel = nuc.elastic.as_ref()
                .or(nuc.fission.as_ref())
                .or(nuc.capture.as_ref())
                .or(nuc.inelastic.as_ref())
                .or(nuc.n2n.as_ref())
                .or(nuc.n3n.as_ref());

            if let Some(rk) = any_kernel {
                all_grids_vec.extend_from_slice(&rk.kernel.energies);
                n_energies_vec[nuc_idx] = rk.kernel.n_energy() as i32;
            }

            // Each reaction
            let reactions: [Option<&crate::transport::xs_provider::ReactionKernel>; 6] = [
                nuc.elastic.as_ref(),
                nuc.inelastic.as_ref(),
                nuc.n2n.as_ref(),
                nuc.n3n.as_ref(),
                nuc.fission.as_ref(),
                nuc.capture.as_ref(),
            ];

            for (rxn_idx, rxn_opt) in reactions.iter().enumerate() {
                let key = nuc_idx * n_rxn + rxn_idx;
                if let Some(rk) = rxn_opt {
                    has_reaction_vec[key] = 1;
                    basis_offsets_vec[key] = all_basis_vec.len() as i32;
                    all_basis_vec.extend_from_slice(rk.kernel.basis_f32());
                    coeffs_offsets_vec[key] = all_coeffs_vec.len() as i32;
                    all_coeffs_vec.extend_from_slice(&rk.coeffs);
                } else {
                    basis_offsets_vec[key] = 0;
                    coeffs_offsets_vec[key] = 0;
                }
            }
        }

        // Ensure we have data
        if all_basis_vec.is_empty() { all_basis_vec.push(0.0); }
        if all_coeffs_vec.is_empty() { all_coeffs_vec.push(0.0); }
        if all_grids_vec.is_empty() { all_grids_vec.push(0.0); }

        // ── Pack discrete inelastic levels (Q-values + SVD basis) ──
        let mut lev_q_vec: Vec<f64> = Vec::new();
        let mut lev_thr_vec: Vec<f64> = Vec::new();
        let mut lev_off_vec = vec![0_i32; n_nuc];
        let mut lev_cnt_vec = vec![0_i32; n_nuc];
        let mut lev_basis_vec: Vec<f32> = Vec::new();
        let mut lev_coeffs_vec: Vec<f64> = Vec::new();
        let mut lev_basis_off_vec: Vec<i32> = Vec::new();
        let mut lev_coeffs_off_vec: Vec<i32> = Vec::new();
        let mut lev_has_kernel_vec: Vec<i32> = Vec::new();
        let mut lev_mt_vec: Vec<i32> = Vec::new();

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            lev_off_vec[nuc_idx] = lev_q_vec.len() as i32;
            lev_cnt_vec[nuc_idx] = nuc.discrete_levels.len() as i32;
            for lev in &nuc.discrete_levels {
                lev_q_vec.push(lev.info.q_value);
                lev_thr_vec.push(lev.info.threshold);
                lev_mt_vec.push(lev.info.mt as i32);
                if let Some(ref rk) = lev.kernel {
                    lev_has_kernel_vec.push(1);
                    lev_basis_off_vec.push(lev_basis_vec.len() as i32);
                    lev_basis_vec.extend_from_slice(rk.kernel.basis_f32());
                    lev_coeffs_off_vec.push(lev_coeffs_vec.len() as i32);
                    lev_coeffs_vec.extend_from_slice(&rk.coeffs);
                } else {
                    lev_has_kernel_vec.push(0);
                    lev_basis_off_vec.push(0);
                    lev_coeffs_off_vec.push(0);
                }
            }
        }
        if lev_q_vec.is_empty() {
            lev_q_vec.push(0.0); lev_thr_vec.push(0.0); lev_mt_vec.push(0);
            lev_has_kernel_vec.push(0); lev_basis_off_vec.push(0); lev_coeffs_off_vec.push(0);
        }
        if lev_basis_vec.is_empty() { lev_basis_vec.push(0.0); }
        if lev_coeffs_vec.is_empty() { lev_coeffs_vec.push(0.0); }

        let n_total_levels: usize = lev_cnt_vec.iter().map(|&c| c as usize).sum();
        println!("  GPU: {} discrete levels, {:.1} MB level basis",
                 n_total_levels, lev_basis_vec.len() as f64 * 4.0 / 1e6);

        // ── Pack angular distributions ──
        let mut ang_e_vec: Vec<f64> = Vec::new();
        let mut ang_mu_vec: Vec<f64> = Vec::new();
        let mut ang_cdf_vec: Vec<f64> = Vec::new();
        let mut ang_doff_vec: Vec<i32> = Vec::new();
        let mut ang_dsz_vec: Vec<i32> = Vec::new();
        let mut ang_noff_vec = vec![0_i32; n_nuc];
        let mut ang_nne_vec = vec![0_i32; n_nuc];
        let mut ang_cm_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref ad) = nuc.elastic_angle {
                ang_noff_vec[nuc_idx] = ang_e_vec.len() as i32;
                ang_nne_vec[nuc_idx] = ad.energies.len() as i32;
                ang_cm_vec[nuc_idx] = if ad.center_of_mass { 1 } else { 0 };
                for (i, e) in ad.energies.iter().enumerate() {
                    ang_e_vec.push(*e);
                    let dist = &ad.distributions[i];
                    ang_doff_vec.push(ang_mu_vec.len() as i32);
                    ang_dsz_vec.push(dist.mu.len() as i32);
                    ang_mu_vec.extend_from_slice(&dist.mu);
                    ang_cdf_vec.extend_from_slice(&dist.cdf);
                }
            }
        }
        if ang_e_vec.is_empty() { ang_e_vec.push(0.0); }
        if ang_mu_vec.is_empty() { ang_mu_vec.push(0.0); ang_cdf_vec.push(0.0); }
        if ang_doff_vec.is_empty() { ang_doff_vec.push(0); ang_dsz_vec.push(0); }

        // ── Pack nu-bar tables (flat with offsets) ──
        let mut nb_energies_vec: Vec<f64> = Vec::new();
        let mut nb_values_vec: Vec<f64> = Vec::new();
        let mut nb_offsets_vec = vec![0_i32; n_nuc];
        let mut nb_sizes_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref t) = nuc.nu_bar_table {
                if !t.energies.is_empty() {
                    nb_offsets_vec[nuc_idx] = nb_energies_vec.len() as i32;
                    nb_sizes_vec[nuc_idx] = t.energies.len() as i32;
                    nb_energies_vec.extend_from_slice(&t.energies);
                    nb_values_vec.extend_from_slice(&t.values);
                }
            }
        }
        if nb_energies_vec.is_empty() { nb_energies_vec.push(0.0); nb_values_vec.push(0.0); }

        // ── Pack fission energy distributions (flat CDFs with offsets) ──
        let mut fis_inc_e_vec: Vec<f64> = Vec::new();
        let mut fis_dist_off_vec: Vec<i32> = Vec::new();
        let mut fis_dist_sz_vec: Vec<i32> = Vec::new();
        let mut fis_eout_vec: Vec<f64> = Vec::new();
        let mut fis_cdf_vec: Vec<f64> = Vec::new();
        let mut fis_nuc_off_vec = vec![0_i32; n_nuc];
        let mut fis_nuc_ninc_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref edist) = nuc.fission_energy_dist {
                fis_nuc_off_vec[nuc_idx] = fis_inc_e_vec.len() as i32;
                fis_nuc_ninc_vec[nuc_idx] = edist.energies.len() as i32;
                for (i, e_inc) in edist.energies.iter().enumerate() {
                    fis_inc_e_vec.push(*e_inc);
                    let dist = &edist.distributions[i];
                    fis_dist_off_vec.push(fis_eout_vec.len() as i32);
                    fis_dist_sz_vec.push(dist.e_out.len() as i32);
                    fis_eout_vec.extend_from_slice(&dist.e_out);
                    fis_cdf_vec.extend_from_slice(&dist.cdf);
                }
            }
        }
        if fis_inc_e_vec.is_empty() { fis_inc_e_vec.push(0.0); }
        if fis_eout_vec.is_empty() { fis_eout_vec.push(0.0); fis_cdf_vec.push(0.0); }
        if fis_dist_off_vec.is_empty() { fis_dist_off_vec.push(0); fis_dist_sz_vec.push(0); }

        // ── Pack URR probability tables ──
        let mut urr_e_vec: Vec<f64> = Vec::new();
        let mut urr_cp_vec: Vec<f64> = Vec::new();
        let mut urr_tf_vec: Vec<f64> = Vec::new();
        let mut urr_ef_vec: Vec<f64> = Vec::new();
        let mut urr_ff_vec: Vec<f64> = Vec::new();
        let mut urr_cf_vec: Vec<f64> = Vec::new();
        let mut urr_off_vec = vec![0_i32; n_nuc];
        let mut urr_ne_vec = vec![0_i32; n_nuc];
        let mut urr_nb_vec = vec![0_i32; n_nuc];
        let mut urr_ms_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref urr) = nuc.urr_tables {
                urr_off_vec[nuc_idx] = urr_e_vec.len() as i32;
                urr_ne_vec[nuc_idx] = urr.energies.len() as i32;
                urr_nb_vec[nuc_idx] = urr.n_bands as i32;
                urr_ms_vec[nuc_idx] = if urr.multiply_smooth { 1 } else { 0 };
                urr_e_vec.extend_from_slice(&urr.energies);
                for row in &urr.cum_prob { urr_cp_vec.extend_from_slice(row); }
                for row in &urr.total_factor { urr_tf_vec.extend_from_slice(row); }
                for row in &urr.elastic_factor { urr_ef_vec.extend_from_slice(row); }
                for row in &urr.fission_factor { urr_ff_vec.extend_from_slice(row); }
                for row in &urr.capture_factor { urr_cf_vec.extend_from_slice(row); }
            }
        }
        // Always have at least one element so device pointers are never null
        if urr_e_vec.is_empty() { urr_e_vec.push(0.0); }
        if urr_cp_vec.is_empty() { urr_cp_vec.push(0.0); }
        if urr_tf_vec.is_empty() { urr_tf_vec.push(0.0); }
        if urr_ef_vec.is_empty() { urr_ef_vec.push(0.0); }
        if urr_ff_vec.is_empty() { urr_ff_vec.push(0.0); }
        if urr_cf_vec.is_empty() { urr_cf_vec.push(0.0); }

        println!("  GPU: basis={:.1} MB, grids={:.1} MB, nu-bar={} pts, fis_spec={} pts",
                 all_basis_vec.len() as f64 * 4.0 / 1e6,
                 all_grids_vec.len() as f64 * 8.0 / 1e6,
                 nb_energies_vec.len(),
                 fis_eout_vec.len());

        Ok(GpuNuclideData {
            all_basis: self.stream.clone_htod(&all_basis_vec)?,
            all_coeffs: self.stream.clone_htod(&all_coeffs_vec)?,
            all_energy_grids: self.stream.clone_htod(&all_grids_vec)?,
            basis_offsets: self.stream.clone_htod(&basis_offsets_vec)?,
            grid_offsets: self.stream.clone_htod(&grid_offsets_vec)?,
            n_energies: self.stream.clone_htod(&n_energies_vec)?,
            has_reaction: self.stream.clone_htod(&has_reaction_vec)?,
            coeffs_offsets: self.stream.clone_htod(&coeffs_offsets_vec)?,
            rank: rank as i32,
            level_q_values: self.stream.clone_htod(&lev_q_vec)?,
            level_thresholds: self.stream.clone_htod(&lev_thr_vec)?,
            level_offsets: self.stream.clone_htod(&lev_off_vec)?,
            level_counts: self.stream.clone_htod(&lev_cnt_vec)?,
            level_basis: self.stream.clone_htod(&lev_basis_vec)?,
            level_coeffs: self.stream.clone_htod(&lev_coeffs_vec)?,
            level_basis_offsets: self.stream.clone_htod(&lev_basis_off_vec)?,
            level_coeffs_offsets: self.stream.clone_htod(&lev_coeffs_off_vec)?,
            level_has_kernel: self.stream.clone_htod(&lev_has_kernel_vec)?,
            level_mt: self.stream.clone_htod(&lev_mt_vec)?,
            ang_energies: self.stream.clone_htod(&ang_e_vec)?,
            ang_mu: self.stream.clone_htod(&ang_mu_vec)?,
            ang_cdf: self.stream.clone_htod(&ang_cdf_vec)?,
            ang_dist_offsets: self.stream.clone_htod(&ang_doff_vec)?,
            ang_dist_sizes: self.stream.clone_htod(&ang_dsz_vec)?,
            ang_nuc_offsets: self.stream.clone_htod(&ang_noff_vec)?,
            ang_nuc_n_energies: self.stream.clone_htod(&ang_nne_vec)?,
            ang_is_cm: self.stream.clone_htod(&ang_cm_vec)?,
            nu_bar_energies: self.stream.clone_htod(&nb_energies_vec)?,
            nu_bar_values: self.stream.clone_htod(&nb_values_vec)?,
            nu_bar_offsets: self.stream.clone_htod(&nb_offsets_vec)?,
            nu_bar_sizes: self.stream.clone_htod(&nb_sizes_vec)?,
            fis_inc_energies: self.stream.clone_htod(&fis_inc_e_vec)?,
            fis_dist_offsets: self.stream.clone_htod(&fis_dist_off_vec)?,
            fis_dist_sizes: self.stream.clone_htod(&fis_dist_sz_vec)?,
            fis_e_out: self.stream.clone_htod(&fis_eout_vec)?,
            fis_cdf: self.stream.clone_htod(&fis_cdf_vec)?,
            fis_nuc_offsets: self.stream.clone_htod(&fis_nuc_off_vec)?,
            fis_nuc_n_inc: self.stream.clone_htod(&fis_nuc_ninc_vec)?,
            urr_energies: self.stream.clone_htod(&urr_e_vec)?,
            urr_cum_prob: self.stream.clone_htod(&urr_cp_vec)?,
            urr_total_f: self.stream.clone_htod(&urr_tf_vec)?,
            urr_elastic_f: self.stream.clone_htod(&urr_ef_vec)?,
            urr_fission_f: self.stream.clone_htod(&urr_ff_vec)?,
            urr_capture_f: self.stream.clone_htod(&urr_cf_vec)?,
            urr_offsets: self.stream.clone_htod(&urr_off_vec)?,
            urr_n_energies: self.stream.clone_htod(&urr_ne_vec)?,
            urr_n_bands: self.stream.clone_htod(&urr_nb_vec)?,
            urr_multiply_smooth: self.stream.clone_htod(&urr_ms_vec)?,
        })
    }

    /// Upload material composition data to GPU.
    pub fn upload_material_data(
        &self,
        materials: &[crate::transport::material::Material],
        nuclide_awrs: &[f64],
        nuclide_nu_bars: &[f64],
    ) -> Result<GpuMaterialData, Box<dyn std::error::Error>> {
        let max_nuc = 4; // max nuclides per material
        let n_mat = materials.len();

        let mut n_nuclides = vec![0_i32; n_mat];
        let mut nuc_idx = vec![0_i32; n_mat * max_nuc];
        let mut atom_dens = vec![0.0_f64; n_mat * max_nuc];

        for (m, mat) in materials.iter().enumerate() {
            n_nuclides[m] = mat.nuclides.len() as i32;
            for (i, nuc) in mat.nuclides.iter().enumerate() {
                nuc_idx[m * max_nuc + i] = nuc.xs_kernel_idx as i32;
                atom_dens[m * max_nuc + i] = nuc.atom_density;
            }
        }

        Ok(GpuMaterialData {
            mat_n_nuclides: self.stream.clone_htod(&n_nuclides)?,
            mat_nuclide_idx: self.stream.clone_htod(&nuc_idx)?,
            mat_atom_density: self.stream.clone_htod(&atom_dens)?,
            awr_table: self.stream.clone_htod(nuclide_awrs)?,
            nu_bar_const: self.stream.clone_htod(nuclide_nu_bars)?,
        })
    }

    /// Upload S(α,β) thermal scattering data for one temperature.
    pub fn upload_sab_data(
        &self,
        tsl: &crate::thermal::ThermalScatteringData,
        temp_idx: usize,
    ) -> Result<GpuSabData, Box<dyn std::error::Error>> {

        let inel = &tsl.inelastic[temp_idx];
        match &inel.dist {
            crate::thermal::InelasticDist::Continuous(c) => {
                // Pack incident energy grid and XS
                let inc_e: Vec<f64> = inel.energy.clone();
                let xs: Vec<f64> = inel.xs.clone();
                let n_inc = inc_e.len() as i32;

                // Pack outgoing energy CDFs with offsets
                let mut eout_offsets = Vec::with_capacity(c.n_inc);
                let mut eout_sizes = Vec::with_capacity(c.n_inc);
                for i in 0..c.n_inc {
                    let start = c.offsets[i];
                    let end = if i + 1 < c.offsets.len() { c.offsets[i + 1] } else { c.e_out.len() };
                    eout_offsets.push(start as i32);
                    eout_sizes.push((end - start) as i32);
                }

                // Pack mu offsets/sizes (one per outgoing energy bin)
                let mut mu_offs = Vec::with_capacity(c.mu_offsets.len());
                let mut mu_szs = Vec::with_capacity(c.mu_offsets.len());
                for i in 0..c.mu_offsets.len() {
                    let start = c.mu_offsets[i];
                    let end = if i + 1 < c.mu_offsets.len() { c.mu_offsets[i + 1] } else { c.mu.len() };
                    mu_offs.push(start as i32);
                    mu_szs.push((end - start) as i32);
                }

                // Ensure non-empty
                if mu_offs.is_empty() { mu_offs.push(0); mu_szs.push(0); }

                println!("  GPU S(a,b): {} inc energies, {} E_out pts, {} mu pts",
                         n_inc, c.e_out.len(), c.mu.len());

                Ok(GpuSabData {
                    inc_energies: self.stream.clone_htod(&inc_e)?,
                    n_inc,
                    eout_offsets: self.stream.clone_htod(&eout_offsets)?,
                    eout_sizes: self.stream.clone_htod(&eout_sizes)?,
                    e_out: self.stream.clone_htod(&c.e_out)?,
                    cdf_e: self.stream.clone_htod(&c.cdf_e)?,
                    mu_offsets: self.stream.clone_htod(&mu_offs)?,
                    mu_sizes: self.stream.clone_htod(&mu_szs)?,
                    mu: self.stream.clone_htod(&c.mu)?,
                    cdf_mu: self.stream.clone_htod(&c.cdf_mu)?,
                    xs: self.stream.clone_htod(&xs)?,
                    energy_max: tsl.energy_max,
                })
            }
            crate::thermal::InelasticDist::Discrete(_) => {
                // Discrete mode — create empty placeholder (not yet supported on GPU)
                println!("  GPU S(a,b): discrete mode, using empty placeholder");
                Ok(GpuSabData {
                    inc_energies: self.stream.clone_htod(&[0.0_f64])?,
                    n_inc: 0,
                    eout_offsets: self.stream.clone_htod(&[0_i32])?,
                    eout_sizes: self.stream.clone_htod(&[0_i32])?,
                    e_out: self.stream.clone_htod(&[0.0_f64])?,
                    cdf_e: self.stream.clone_htod(&[0.0_f64])?,
                    mu_offsets: self.stream.clone_htod(&[0_i32])?,
                    mu_sizes: self.stream.clone_htod(&[0_i32])?,
                    mu: self.stream.clone_htod(&[0.0_f64])?,
                    cdf_mu: self.stream.clone_htod(&[0.0_f64])?,
                    xs: self.stream.clone_htod(&[0.0_f64])?,
                    energy_max: 0.0,
                })
            }
        }
    }

    /// Create an empty S(α,β) placeholder (no thermal scattering data).
    pub fn upload_sab_data_empty(&self) -> Result<GpuSabData, Box<dyn std::error::Error>> {
        Ok(GpuSabData {
            inc_energies: self.stream.clone_htod(&[0.0_f64])?,
            n_inc: 0,
            eout_offsets: self.stream.clone_htod(&[0_i32])?,
            eout_sizes: self.stream.clone_htod(&[0_i32])?,
            e_out: self.stream.clone_htod(&[0.0_f64])?,
            cdf_e: self.stream.clone_htod(&[0.0_f64])?,
            mu_offsets: self.stream.clone_htod(&[0_i32])?,
            mu_sizes: self.stream.clone_htod(&[0_i32])?,
            mu: self.stream.clone_htod(&[0.0_f64])?,
            cdf_mu: self.stream.clone_htod(&[0.0_f64])?,
            xs: self.stream.clone_htod(&[0.0_f64])?,
            energy_max: 0.0,
        })
    }

    /// Run one batch of transport on GPU.
    ///
    /// geom_type: 0=PWR pin cell, 1=Godiva bare sphere.
    pub fn run_batch(
        &self,
        source_bank: &[(f64, f64, f64, f64)],
        batch: u32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        max_steps: u32,
        geom_type: i32,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error>> {
        let n = source_bank.len();
        let n_i32 = n as i32;
        let grid_full = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg_full = LaunchConfig {
            grid_dim: (grid_full, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // Unpack source bank into SoA
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        let mut sz = Vec::with_capacity(n);
        let mut se = Vec::with_capacity(n);
        for &(x, y, z, e) in source_bank {
            sx.push(x); sy.push(y); sz.push(z); se.push(e);
        }

        let d_src_x = self.stream.clone_htod(&sx)?;
        let d_src_y = self.stream.clone_htod(&sy)?;
        let d_src_z = self.stream.clone_htod(&sz)?;
        let d_src_e = self.stream.clone_htod(&se)?;

        // Pre-allocate all particle state arrays (reused across steps)
        let mut d_pos_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_energy: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_alive: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_rng_state: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let mut d_rng_inc: CudaSlice<u64> = self.stream.alloc_zeros(n)?;

        // Compaction + sort buffers
        let mut d_compact_idx: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_compact_idx_sorted: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let n_bins = 256;
        // Fission bank
        let max_fission = (n * 3) as i32;
        let mut d_fis_x: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_y: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_z: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_e: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_w: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Counters
        let mut d_cnt_coll: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_fis: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_leak: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_surf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Initialize source
        let batch_seed = batch as u64 * 1_000_000;
        unsafe {
            self.stream.launch_builder(&self.k_init_source)
                .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                .arg(&d_src_x).arg(&d_src_y).arg(&d_src_z).arg(&d_src_e)
                .arg(&n_i32)
                .arg(&batch_seed)
                .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                .arg(&geom_type)
                .launch(cfg_full)?;
        }

        // Build packed TransportParams buffer (N_PARAMS u64 values)
        // Extract raw device pointers from each CudaSlice
        macro_rules! dptr {
            ($slice:expr) => {{
                let (ptr, _guard) = $slice.device_ptr(&self.stream);
                ptr  // CUdeviceptr = u64
            }};
        }
        let params_vec: Vec<u64> = vec![
            dptr!(&nuc_data.all_basis),            //  0 P_BASIS
            dptr!(&nuc_data.all_coeffs),           //  1 P_COEFFS
            dptr!(&nuc_data.all_energy_grids),     //  2 P_ENERGY_GRIDS
            dptr!(&nuc_data.basis_offsets),         //  3 P_BASIS_OFFSETS
            dptr!(&nuc_data.grid_offsets),          //  4 P_GRID_OFFSETS
            dptr!(&nuc_data.n_energies),            //  5 P_N_ENERGIES
            dptr!(&nuc_data.has_reaction),          //  6 P_HAS_REACTION
            dptr!(&nuc_data.coeffs_offsets),        //  7 P_COEFFS_OFFSETS
            nuc_data.rank as u64,                   //  8 P_RANK
            dptr!(&mat_data.mat_n_nuclides),        //  9 P_MAT_N_NUC
            dptr!(&mat_data.mat_nuclide_idx),       // 10 P_MAT_NUC_IDX
            dptr!(&mat_data.mat_atom_density),      // 11 P_MAT_ATOM_DENS
            dptr!(&mat_data.awr_table),             // 12 P_AWR_TABLE
            dptr!(&mat_data.nu_bar_const),          // 13 P_NU_BAR_CONST
            dptr!(&nuc_data.nu_bar_energies),       // 14 P_NB_ENERGIES
            dptr!(&nuc_data.nu_bar_values),         // 15 P_NB_VALUES
            dptr!(&nuc_data.nu_bar_offsets),         // 16 P_NB_OFFSETS
            dptr!(&nuc_data.nu_bar_sizes),           // 17 P_NB_SIZES
            dptr!(&nuc_data.fis_inc_energies),       // 18 P_FIS_INC_E
            dptr!(&nuc_data.fis_dist_offsets),       // 19 P_FIS_DIST_OFF
            dptr!(&nuc_data.fis_dist_sizes),         // 20 P_FIS_DIST_SZ
            dptr!(&nuc_data.fis_e_out),              // 21 P_FIS_E_OUT
            dptr!(&nuc_data.fis_cdf),                // 22 P_FIS_CDF
            dptr!(&nuc_data.fis_nuc_offsets),        // 23 P_FIS_NUC_OFF
            dptr!(&nuc_data.fis_nuc_n_inc),          // 24 P_FIS_NUC_NINC
            dptr!(&nuc_data.level_q_values),         // 25 P_LEVEL_Q
            dptr!(&nuc_data.level_thresholds),       // 26 P_LEVEL_THR
            dptr!(&nuc_data.level_offsets),           // 27 P_LEVEL_OFFSETS
            dptr!(&nuc_data.level_counts),            // 28 P_LEVEL_COUNTS
            dptr!(&nuc_data.level_basis),             // 29 P_LEVEL_BASIS
            dptr!(&nuc_data.level_coeffs),            // 30 P_LEVEL_COEFFS
            dptr!(&nuc_data.level_basis_offsets),     // 31 P_LEVEL_BOFF
            dptr!(&nuc_data.level_coeffs_offsets),    // 32 P_LEVEL_COFF
            dptr!(&nuc_data.level_has_kernel),        // 33 P_LEVEL_HAS_K
            dptr!(&nuc_data.level_mt),                // 34 P_LEVEL_MT
            dptr!(&nuc_data.ang_energies),            // 35 P_ANG_ENERGIES
            dptr!(&nuc_data.ang_mu),                  // 36 P_ANG_MU
            dptr!(&nuc_data.ang_cdf),                 // 37 P_ANG_CDF
            dptr!(&nuc_data.ang_dist_offsets),        // 38 P_ANG_DIST_OFF
            dptr!(&nuc_data.ang_dist_sizes),          // 39 P_ANG_DIST_SZ
            dptr!(&nuc_data.ang_nuc_offsets),         // 40 P_ANG_NUC_OFF
            dptr!(&nuc_data.ang_nuc_n_energies),      // 41 P_ANG_NUC_NE
            dptr!(&nuc_data.ang_is_cm),               // 42 P_ANG_IS_CM
            dptr!(&sab_data.inc_energies),            // 43 P_SAB_INC_E
            sab_data.n_inc as u64,                    // 44 P_SAB_N_INC
            dptr!(&sab_data.eout_offsets),            // 45 P_SAB_EOUT_OFF
            dptr!(&sab_data.eout_sizes),              // 46 P_SAB_EOUT_SZ
            dptr!(&sab_data.e_out),                   // 47 P_SAB_E_OUT
            dptr!(&sab_data.cdf_e),                   // 48 P_SAB_CDF_E
            dptr!(&sab_data.mu_offsets),              // 49 P_SAB_MU_OFF
            dptr!(&sab_data.mu_sizes),                // 50 P_SAB_MU_SZ
            dptr!(&sab_data.mu),                      // 51 P_SAB_MU
            dptr!(&sab_data.cdf_mu),                  // 52 P_SAB_CDF_MU
            dptr!(&sab_data.xs),                      // 53 P_SAB_XS
            sab_data.energy_max.to_bits(),            // 54 P_SAB_EMAX (f64 as bits)
            dptr!(&nuc_data.urr_energies),            // 55 P_URR_ENERGIES
            dptr!(&nuc_data.urr_cum_prob),            // 56 P_URR_CUM_PROB
            dptr!(&nuc_data.urr_total_f),             // 57 P_URR_TOTAL_F
            dptr!(&nuc_data.urr_elastic_f),           // 58 P_URR_ELASTIC_F
            dptr!(&nuc_data.urr_fission_f),           // 59 P_URR_FISSION_F
            dptr!(&nuc_data.urr_capture_f),           // 60 P_URR_CAPTURE_F
            dptr!(&nuc_data.urr_offsets),              // 61 P_URR_OFFSETS
            dptr!(&nuc_data.urr_n_energies),           // 62 P_URR_N_ENERGIES
            dptr!(&nuc_data.urr_n_bands),              // 63 P_URR_N_BANDS
            dptr!(&nuc_data.urr_multiply_smooth),      // 64 P_URR_MULT_SM
            geom_type as u64,                         // 65 P_GEOM_TYPE
        ];
        assert_eq!(params_vec.len(), N_PARAMS);
        let d_params = self.stream.clone_htod(&params_vec)?;

        let mut n_alive = n as i32;
        let compact_interval = 10; // Re-compact every N steps

        let mut step = 0_u32;
        while step < max_steps && n_alive > 0 {
                // 1. Compact: build dense list of alive particle indices
                let mut d_compact_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
                let compact_grid = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let compact_cfg = LaunchConfig {
                    grid_dim: (compact_grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.stream.launch_builder(&self.k_compact_alive)
                        .arg(&d_alive).arg(&n_i32)
                        .arg(&mut d_compact_idx).arg(&mut d_compact_count)
                        .launch(compact_cfg)?;
                }
                let count = self.stream.clone_dtoh(&d_compact_count)?;
                n_alive = count[0];
                if n_alive <= 0 { break; }

                // 2. Energy sort: bin count → prefix sum → scatter
                let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let alive_cfg = LaunchConfig {
                    grid_dim: (alive_grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };

                // 2a. Count particles per energy bin
                let mut d_bin_counts: CudaSlice<i32> = self.stream.alloc_zeros(n_bins)?;
                unsafe {
                    self.stream.launch_builder(&self.k_energy_bin_count)
                        .arg(&d_energy).arg(&d_compact_idx).arg(&n_alive)
                        .arg(&mut d_bin_counts)
                        .launch(alive_cfg)?;
                }

                // 2b. Prefix sum on CPU (256 ints — trivial)
                let counts = self.stream.clone_dtoh(&d_bin_counts)?;
                let mut offsets = vec![0_i32; n_bins];
                let mut running = 0_i32;
                for i in 0..n_bins {
                    offsets[i] = running;
                    running += counts[i];
                }
                let d_bin_offsets = self.stream.clone_htod(&offsets)?;

                // 2c. Scatter compact indices into energy-sorted order
                unsafe {
                    self.stream.launch_builder(&self.k_energy_bin_scatter)
                        .arg(&d_energy).arg(&d_compact_idx).arg(&n_alive)
                        .arg(&mut d_compact_idx_sorted).arg(&d_bin_offsets)
                        .launch(alive_cfg)?;
                }

                // Swap: sorted becomes the active compact index
                std::mem::swap(&mut d_compact_idx, &mut d_compact_idx_sorted);

            // Launch persistent kernel: N steps in one kernel call
            let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let alive_cfg = LaunchConfig {
                grid_dim: (alive_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            let steps_this_launch = compact_interval as i32;
            unsafe {
                self.stream.launch_builder(&self.k_transport_persistent)
                    .arg(&d_params)
                    .arg(&d_compact_idx)
                    .arg(&n_alive)
                    .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                    .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                    .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                    .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                    .arg(&mut d_fis_x).arg(&mut d_fis_y).arg(&mut d_fis_z)
                    .arg(&mut d_fis_e).arg(&mut d_fis_w)
                    .arg(&mut d_fis_count).arg(&max_fission)
                    .arg(&mut d_cnt_coll).arg(&mut d_cnt_fis)
                    .arg(&mut d_cnt_leak).arg(&mut d_cnt_surf)
                    .arg(&steps_this_launch)
                    .launch(alive_cfg)?;
            }

            step += compact_interval; // persistent kernel did N steps
        }

        // Download results
        let fis_count = self.stream.clone_dtoh(&d_fis_count)?[0] as usize;
        let cnt_coll = self.stream.clone_dtoh(&d_cnt_coll)?[0] as u32;
        let cnt_fis = self.stream.clone_dtoh(&d_cnt_fis)?[0] as u32;
        let cnt_leak = self.stream.clone_dtoh(&d_cnt_leak)?[0] as u32;
        let cnt_surf = self.stream.clone_dtoh(&d_cnt_surf)?[0] as u32;

        let fis_count_clamped = fis_count.min(max_fission as usize);
        let fission_bank = if fis_count_clamped > 0 {
            let fx = self.stream.clone_dtoh(&d_fis_x)?;
            let fy = self.stream.clone_dtoh(&d_fis_y)?;
            let fz = self.stream.clone_dtoh(&d_fis_z)?;
            let fe = self.stream.clone_dtoh(&d_fis_e)?;
            (0..fis_count_clamped)
                .map(|i| (fx[i], fy[i], fz[i], fe[i]))
                .collect()
        } else {
            vec![]
        };

        let k_eff = fission_bank.len() as f64 / n as f64;

        Ok(GpuBatchResult {
            k_eff,
            collisions: cnt_coll,
            fissions: cnt_fis,
            leakage: cnt_leak,
            surface_crossings: cnt_surf,
            fission_bank,
        })
    }
}
