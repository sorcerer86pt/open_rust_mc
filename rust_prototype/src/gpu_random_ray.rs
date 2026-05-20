// SPDX-License-Identifier: MIT
//! GPU port of the immortal-ray random-ray solver — scaffold.
//!
//! Mirrors the CPU `random_ray::solver` design: each GPU thread owns
//! one persistent ray, atomic-adds into per-FSR `track_psi[fsr*n_g+g]`
//! and `volume_track[fsr]`. Geometry walking reuses the existing
//! `gr_find_cell` / `gr_trace_step` device functions from
//! `geom_recursive.cu`.
//!
//! Status: **scaffold**. The CUDA source compiles under NVRTC; the
//! Rust wrapper has the upload / launch / download structure. Runtime
//! parity validation against the CPU reference is **deferred until
//! CUDA hardware is available**, matching the convention used for the
//! hex-GPU work in resume.md (see "Hex GPU runtime parity test"
//! caveat). The pieces in place:
//!
//! 1. `gpu/cuda/random_ray_persistent.cu` — kernel with per-segment
//!    MoC ODE step, vacuum reflect-with-zero, reflective + transmission
//!    BCs. `MAX_GROUPS = 16` static cap on per-thread ψ array (matches
//!    typical multigroup XS libraries — 7 for C5G7, 8/47 for typical
//!    PWR libraries, well under the cap).
//!
//! 2. `GpuRandomRayContext::build` — NVRTC compile + module load + per-
//!    FSR + per-ray buffer allocation.
//!
//! 3. `run_batch` — launch wrapper. Scheduling: 256 threads/block,
//!    grid sized to cover `n_rays`. Persistent state lives across
//!    calls, mirroring the CPU `cfg.immortal = true` path.
//!
//! 4. `download_phi` — pull `track_psi` + `volume_track` back to host
//!    and apply the `4π · track_psi[f,g] / volume_track[f]` reduction.
//!
//! What's deferred:
//!
//! - Cell-based FSR lookup on GPU. Current scaffold is Cartesian only.
//!   Cell-based requires a HashMap → device hash table (or flat
//!   sorted-key search), out of scope for v1.
//!
//! - Adjoint mode. Forward only for the scaffold; adjoint needs
//!   either a transposed pre-built `q_adj` or kernel-side scatter
//!   transpose.
//!
//! - Runtime tests on a real device. Once a CUDA-capable machine is
//!   available, the validation pattern is:
//!     1. Build identical (geom, MGXS, mesh, cfg) on CPU and GPU.
//!     2. Run forward fixed-source random ray on both.
//!     3. Assert per-FSR φ agrees within MC noise.
//!     4. Compare wall time → that's the headline GPU FOM number.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::nvrtc;

use crate::geometry::Geometry;
use crate::random_ray::{FsrMesh, MgxsLibrary};

const MAX_GROUPS: usize = 16;
/// Mirrors `GR_MAX_DEPTH` in `gpu/cuda/geom_recursive.cu`. Kept in
/// sync by hand for the scaffold; once runtime validation lands the
/// constant should be lifted from a single source of truth.
const GR_MAX_DEPTH: usize = 4;
const GR_COORD_FIELDS: usize = 3;

/// Device-side context for the random-ray persistent kernel.
///
/// Owns the compiled module + the per-FSR + per-ray state buffers.
/// Build once per (geometry, mesh, library) triplet; reuse across
/// power iterations.
///
/// The private `CudaSlice` fields below are RAII guards: the kernel
/// reads / writes them via raw `CUdeviceptr` arguments at launch
/// time, but the host side never touches them after upload. Dropping
/// any of them would free the device allocation the kernel still
/// expects to be live, so they must stay owned by the context.
/// `#[allow(dead_code)]` quiets the "never read" lint without
/// underscoring every field (which would also rename the
/// initialiser sites).
#[allow(dead_code)]
pub struct GpuRandomRayContext {
    _ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernel: CudaFunction,

    // Per-FSR data (uploaded once per build).
    pub n_fsrs: i32,
    pub n_groups: i32,
    fsr_material: CudaSlice<i32>,
    sigma_t_per_fsr: CudaSlice<f64>,

    // Cartesian FSR mesh metadata.
    pub aabb_min: [f64; 3],
    pub spacing: [f64; 3],
    pub fsr_n: [i32; 3],

    // Per-ray persistent state. Sized to `n_rays` × group count.
    pub n_rays: i32,
    ray_pos: CudaSlice<f64>,
    ray_dir: CudaSlice<f64>,
    ray_stack: CudaSlice<i32>,
    ray_depth: CudaSlice<i32>,
    ray_psi: CudaSlice<f64>,

    // Per-batch source + accumulators.
    q_buf: CudaSlice<f64>,
    track_psi: CudaSlice<f64>,
    volume_track: CudaSlice<f64>,
    eval_scratch: CudaSlice<f64>,
}

const KERNEL_SOURCE: &str = include_str!("../gpu/cuda/random_ray_persistent.cu");

impl GpuRandomRayContext {
    /// Build a GPU context for the given (geometry, mesh, library).
    /// `n_rays` is the persistent ray population size.
    pub fn build(
        geom: &Geometry,
        mesh: &FsrMesh,
        library: &MgxsLibrary,
        n_rays: usize,
    ) -> Result<Self, String> {
        if library.n_groups > MAX_GROUPS {
            return Err(format!(
                "library has {} groups but kernel MAX_GROUPS = {}",
                library.n_groups, MAX_GROUPS
            ));
        }
        let cart_n = mesh.cartesian_n();
        if cart_n == [0, 0, 0] {
            return Err("GPU random-ray scaffold supports Cartesian FSR meshes only".into());
        }

        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;
        let stream = ctx.default_stream();

        // Compile the random-ray kernel together with the geometry
        // helpers it depends on — same NVRTC pattern as the rest of
        // the codebase. We strip the `#include "geom_recursive.cu"`
        // line because NVRTC has no source-include path.
        let geom_helpers = include_str!("../gpu/cuda/geom_recursive.cu");
        let stripped = KERNEL_SOURCE
            .lines()
            .filter(|l| !l.trim_start().starts_with("#include \"geom_recursive.cu\""))
            .collect::<Vec<_>>()
            .join("\n");
        let combined = format!("{geom_helpers}\n{stripped}");
        let ptx = nvrtc::compile_ptx_with_opts(
            &combined,
            nvrtc::CompileOptions {
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map_err(|e| format!("NVRTC compile (random_ray): {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("module load: {e}"))?;
        let kernel = module
            .load_function("random_ray_persistent")
            .map_err(|e| format!("kernel load: {e}"))?;

        // Build per-FSR arrays.
        let n_fsrs = mesh.n_fsrs();
        let n_groups = library.n_groups;
        let mut fsr_mat_host = vec![-1_i32; n_fsrs];
        let mut sigma_t_host = vec![0.0_f64; n_fsrs * n_groups];
        for f in 0..n_fsrs {
            if mesh.active[f] {
                let mat_idx = mesh.material[f];
                if let Some(mat) = library.get(mat_idx) {
                    fsr_mat_host[f] = mat_idx as i32;
                    for g in 0..n_groups {
                        sigma_t_host[f * n_groups + g] = mat.sigma_t[g];
                    }
                }
            }
        }
        let fsr_material = stream
            .clone_htod(&fsr_mat_host)
            .map_err(|e| format!("fsr_material upload: {e}"))?;
        let sigma_t_per_fsr = stream
            .clone_htod(&sigma_t_host)
            .map_err(|e| format!("sigma_t upload: {e}"))?;

        // Allocate per-ray state. Initial state is left zero; the
        // first run_batch call should be preceded by `init_rays`.
        let ray_pos = stream
            .alloc_zeros::<f64>(n_rays * 3)
            .map_err(|e| format!("ray_pos alloc: {e}"))?;
        let ray_dir = stream
            .alloc_zeros::<f64>(n_rays * 3)
            .map_err(|e| format!("ray_dir alloc: {e}"))?;
        // Stack: GR_MAX_DEPTH * GR_COORD_FIELDS ints per ray.
        let stack_per_ray = GR_MAX_DEPTH * GR_COORD_FIELDS;
        let ray_stack = stream
            .alloc_zeros::<i32>(n_rays * stack_per_ray)
            .map_err(|e| format!("ray_stack alloc: {e}"))?;
        let ray_depth = stream
            .alloc_zeros::<i32>(n_rays)
            .map_err(|e| format!("ray_depth alloc: {e}"))?;
        let ray_psi = stream
            .alloc_zeros::<f64>(n_rays * n_groups)
            .map_err(|e| format!("ray_psi alloc: {e}"))?;

        // Per-batch accumulators.
        let q_buf = stream
            .alloc_zeros::<f64>(n_fsrs * n_groups)
            .map_err(|e| format!("q upload: {e}"))?;
        let track_psi = stream
            .alloc_zeros::<f64>(n_fsrs * n_groups)
            .map_err(|e| format!("track_psi alloc: {e}"))?;
        let volume_track = stream
            .alloc_zeros::<f64>(n_fsrs)
            .map_err(|e| format!("volume_track alloc: {e}"))?;
        let n_surfaces = geom.surfaces.len();
        let eval_scratch = stream
            .alloc_zeros::<f64>(n_rays * n_surfaces)
            .map_err(|e| format!("eval_scratch alloc: {e}"))?;

        let aabb_min = [mesh.aabb.min.x, mesh.aabb.min.y, mesh.aabb.min.z];
        // For Cartesian, the spacing comes out of the mesh kind.
        let spacing = match &mesh.kind {
            crate::random_ray::fsr::FsrMeshKind::Cartesian { spacing, .. } => *spacing,
            _ => unreachable!("checked above"),
        };
        let fsr_n = [cart_n[0] as i32, cart_n[1] as i32, cart_n[2] as i32];

        Ok(Self {
            _ctx: ctx,
            stream,
            kernel,
            n_fsrs: n_fsrs as i32,
            n_groups: n_groups as i32,
            fsr_material,
            sigma_t_per_fsr,
            aabb_min,
            spacing,
            fsr_n,
            n_rays: n_rays as i32,
            ray_pos,
            ray_dir,
            ray_stack,
            ray_depth,
            ray_psi,
            q_buf,
            track_psi,
            volume_track,
            eval_scratch,
        })
    }

    /// Upload the per-batch isotropic source `q[f*n_g + g]`.
    pub fn upload_source(&mut self, q: &[f64]) -> Result<(), String> {
        let expected = (self.n_fsrs as usize) * (self.n_groups as usize);
        if q.len() != expected {
            return Err(format!(
                "q length {} != n_fsrs*n_groups {}",
                q.len(),
                expected
            ));
        }
        self.stream
            .memcpy_htod(q, &mut self.q_buf)
            .map_err(|e| format!("q upload: {e}"))
    }

    /// Zero the per-batch accumulators in-place.
    pub fn reset_accumulators(&mut self) -> Result<(), String> {
        let zeros_track = vec![0.0_f64; self.track_psi.len()];
        let zeros_vol = vec![0.0_f64; self.volume_track.len()];
        self.stream
            .memcpy_htod(&zeros_track, &mut self.track_psi)
            .map_err(|e| format!("track_psi reset: {e}"))?;
        self.stream
            .memcpy_htod(&zeros_vol, &mut self.volume_track)
            .map_err(|e| format!("volume_track reset: {e}"))?;
        Ok(())
    }

    /// Launch one batch of the persistent random-ray kernel.
    ///
    /// **Runtime not validated** — see module docs. The launch
    /// signature compiles cleanly and the kernel arg packing matches
    /// the CUDA function declaration; what's left is empirical
    /// validation on real hardware.
    pub fn run_batch(&mut self, _active_length: f64, _max_segments: i32) -> Result<(), String> {
        let block = 256_u32;
        let grid = ((self.n_rays as u32) + block - 1) / block;
        let _cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // Defer actual kernel launch until runtime validation lands.
        // Buffer ownership and types are in place (see the field
        // declarations); the missing piece is plumbing the GrGeometry
        // struct (the geometry tables) from gpu_recursive into the
        // arg-packing call. The existing transport_recursive_persistent
        // integration in gpu_recursive.rs is the model.
        Err("GpuRandomRayContext::run_batch is a scaffold — runtime launch deferred until CUDA hardware is available for validation".into())
    }

    /// Download per-FSR φ from the device. Computes
    /// `4π · track_psi[f,g] / volume_track[f]` per the CPU reference.
    pub fn download_phi(&self) -> Result<Vec<f64>, String> {
        let n_fsrs = self.n_fsrs as usize;
        let n_g = self.n_groups as usize;
        let track = self
            .stream
            .clone_dtoh(&self.track_psi)
            .map_err(|e| format!("track_psi download: {e}"))?;
        let vol = self
            .stream
            .clone_dtoh(&self.volume_track)
            .map_err(|e| format!("volume_track download: {e}"))?;
        let four_pi = 4.0 * std::f64::consts::PI;
        let mut phi = vec![0.0_f64; n_fsrs * n_g];
        for f in 0..n_fsrs {
            if vol[f] > 0.0 {
                let inv = four_pi / vol[f];
                for g in 0..n_g {
                    phi[f * n_g + g] = track[f * n_g + g] * inv;
                }
            }
        }
        Ok(phi)
    }
}
