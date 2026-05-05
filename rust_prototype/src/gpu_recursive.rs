//! GPU port of the recursive cell-find primitives.
//!
//! Converts a `Geometry` to SoA arrays consumed by the device
//! functions in `gpu/cuda/geom_recursive.cu`, uploads them, and
//! exposes a parity-test entry point that runs `find_cell_recursive`
//! on both CPU and GPU at N random points and checks for bit-exact
//! agreement on the deepest cell index.
//!
//! Scope: this is the proof-of-life half of task #19. Full transport
//! integration (replacing the geom_type-switched hot path) lives in
//! a follow-up — too risky to land in one go.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc;

use crate::geometry::cell::{CellFill, Region};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{Geometry, Vec3};

// ── Tag constants — must match `gpu/cuda/geom_recursive.cu` ─────────

const SURF_PLANE_X: i32 = 0;
const SURF_PLANE_Y: i32 = 1;
const SURF_PLANE_Z: i32 = 2;
const SURF_SPHERE: i32 = 3;
const SURF_CYL_Z: i32 = 4;
const SURF_CYL_X: i32 = 5;
const SURF_CYL_Y: i32 = 6;
const SURF_PLANE_GENERAL: i32 = 7;

const BC_TRANSMISSION: i32 = 0;
const BC_VACUUM: i32 = 1;
const BC_REFLECTIVE: i32 = 2;

const REGION_HALFSPACE_POS: i32 = 0;
const REGION_HALFSPACE_NEG: i32 = 1;
const REGION_INTERSECTION: i32 = 2;
const REGION_UNION: i32 = 3;
const REGION_COMPLEMENT: i32 = 4;

const FILL_MATERIAL: i32 = 0;
const FILL_VOID: i32 = 1;
const FILL_UNIVERSE: i32 = 2;
const FILL_LATTICE: i32 = 3;

// ── Host-side SoA tables before upload ──────────────────────────────

#[derive(Default)]
struct HostTables {
    surf_type: Vec<i32>,
    surf_params: Vec<f64>, // 8 doubles per surface
    surf_bc: Vec<i32>,

    cell_region_off: Vec<i32>,
    cell_region_len: Vec<i32>,
    cell_fill_type: Vec<i32>,
    cell_fill_data: Vec<i32>,
    cell_aabb_min: Vec<f64>, // 3 doubles per cell
    cell_aabb_max: Vec<f64>,

    region_op: Vec<i32>,
    region_arg: Vec<i32>,

    univ_cells_off: Vec<i32>,
    univ_cells_len: Vec<i32>,
    univ_surfaces_off: Vec<i32>,
    univ_surfaces_len: Vec<i32>,
    univ_cell_indices: Vec<i32>,
    univ_surface_indices: Vec<i32>,

    lat_origin: Vec<f64>,
    lat_pitch: Vec<f64>,
    lat_shape: Vec<i32>,
    lat_universes_off: Vec<i32>,
    lat_universes: Vec<i32>,
}

fn pack_surface(s: &Surface, params_out: &mut Vec<f64>, bc_out: &mut Vec<i32>) -> i32 {
    fn push8(v: &mut Vec<f64>, slots: [f64; 8]) {
        v.extend_from_slice(&slots);
    }
    let bc_int = |bc: BoundaryCondition| match bc {
        BoundaryCondition::Transmission => BC_TRANSMISSION,
        BoundaryCondition::Vacuum => BC_VACUUM,
        BoundaryCondition::Reflective => BC_REFLECTIVE,
    };
    match *s {
        Surface::PlaneX { x0, bc } => {
            push8(params_out, [x0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            bc_out.push(bc_int(bc));
            SURF_PLANE_X
        }
        Surface::PlaneY { y0, bc } => {
            push8(params_out, [y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            bc_out.push(bc_int(bc));
            SURF_PLANE_Y
        }
        Surface::PlaneZ { z0, bc } => {
            push8(params_out, [z0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            bc_out.push(bc_int(bc));
            SURF_PLANE_Z
        }
        Surface::Sphere { center, radius, bc } => {
            push8(
                params_out,
                [center.x, center.y, center.z, radius, 0.0, 0.0, 0.0, 0.0],
            );
            bc_out.push(bc_int(bc));
            SURF_SPHERE
        }
        Surface::CylinderZ {
            center_x,
            center_y,
            radius,
            bc,
        } => {
            push8(
                params_out,
                [center_x, center_y, radius, 0.0, 0.0, 0.0, 0.0, 0.0],
            );
            bc_out.push(bc_int(bc));
            SURF_CYL_Z
        }
        Surface::CylinderX {
            center_y,
            center_z,
            radius,
            bc,
        } => {
            push8(
                params_out,
                [center_y, center_z, radius, 0.0, 0.0, 0.0, 0.0, 0.0],
            );
            bc_out.push(bc_int(bc));
            SURF_CYL_X
        }
        Surface::CylinderY {
            center_x,
            center_z,
            radius,
            bc,
        } => {
            push8(
                params_out,
                [center_x, center_z, radius, 0.0, 0.0, 0.0, 0.0, 0.0],
            );
            bc_out.push(bc_int(bc));
            SURF_CYL_Y
        }
        Surface::Plane {
            normal,
            offset,
            bc,
        } => {
            push8(
                params_out,
                [normal.x, normal.y, normal.z, offset, 0.0, 0.0, 0.0, 0.0],
            );
            bc_out.push(bc_int(bc));
            SURF_PLANE_GENERAL
        }
        // Cones not yet supported on the recursive-GPU path; surface
        // types from the assembly demo are PlaneX/Y/Z + CylinderZ +
        // PlaneZ, so this is acceptable for v1. The cell-find helper
        // returns a sentinel huge eval for unsupported surfaces if
        // any leak through; the parity test catches that as a
        // mismatch.
        _ => {
            push8(params_out, [0.0; 8]);
            bc_out.push(BC_VACUUM);
            -1
        }
    }
}

/// Walk a CSG region tree and emit postfix opcodes into `op` / `arg`.
fn flatten_region(region: &Region, op: &mut Vec<i32>, arg: &mut Vec<i32>) {
    match region {
        Region::HalfSpace {
            surface_idx,
            positive,
        } => {
            op.push(if *positive {
                REGION_HALFSPACE_POS
            } else {
                REGION_HALFSPACE_NEG
            });
            arg.push(*surface_idx as i32);
        }
        Region::Intersection(a, b) => {
            flatten_region(a, op, arg);
            flatten_region(b, op, arg);
            op.push(REGION_INTERSECTION);
            arg.push(0);
        }
        Region::Union(a, b) => {
            flatten_region(a, op, arg);
            flatten_region(b, op, arg);
            op.push(REGION_UNION);
            arg.push(0);
        }
        Region::Complement(a) => {
            flatten_region(a, op, arg);
            op.push(REGION_COMPLEMENT);
            arg.push(0);
        }
    }
}

fn finite_aabb(lo: Vec3, hi: Vec3) -> ([f64; 3], [f64; 3]) {
    let clamp = |v: f64| {
        if v.is_finite() {
            v
        } else if v > 0.0 {
            1e20
        } else {
            -1e20
        }
    };
    (
        [clamp(lo.x), clamp(lo.y), clamp(lo.z)],
        [clamp(hi.x), clamp(hi.y), clamp(hi.z)],
    )
}

fn build_host_tables(geom: &Geometry) -> HostTables {
    let mut t = HostTables::default();

    // Surfaces.
    for s in &geom.surfaces {
        let tag = pack_surface(s, &mut t.surf_params, &mut t.surf_bc);
        t.surf_type.push(tag);
    }

    // Cells: region trees flattened, fill packed.
    for c in &geom.cells {
        let off = t.region_op.len() as i32;
        flatten_region(&c.region, &mut t.region_op, &mut t.region_arg);
        let len = t.region_op.len() as i32 - off;
        t.cell_region_off.push(off);
        t.cell_region_len.push(len);
        let (ft, fd) = match c.fill {
            CellFill::Material(m) => (FILL_MATERIAL, m as i32),
            CellFill::Void => (FILL_VOID, 0),
            CellFill::Universe(u) => (FILL_UNIVERSE, u as i32),
            CellFill::Lattice(l) => (FILL_LATTICE, l as i32),
        };
        t.cell_fill_type.push(ft);
        t.cell_fill_data.push(fd);
        let (lo, hi) = finite_aabb(c.aabb.min, c.aabb.max);
        t.cell_aabb_min.extend_from_slice(&lo);
        t.cell_aabb_max.extend_from_slice(&hi);
    }

    // Universes: flatten cell + surface index lists.
    for (u_idx, u) in geom.universes.iter().enumerate() {
        let c_off = t.univ_cell_indices.len() as i32;
        for &ci in &u.cell_indices {
            t.univ_cell_indices.push(ci as i32);
        }
        t.univ_cells_off.push(c_off);
        t.univ_cells_len.push(u.cell_indices.len() as i32);

        let s_off = t.univ_surface_indices.len() as i32;
        for &si in &geom.universe_surfaces[u_idx] {
            t.univ_surface_indices.push(si as i32);
        }
        t.univ_surfaces_off.push(s_off);
        t.univ_surfaces_len
            .push(geom.universe_surfaces[u_idx].len() as i32);
    }

    // Lattices: flatten universe arrays.
    for lat in &geom.lattices {
        t.lat_origin
            .extend_from_slice(&[lat.origin.x, lat.origin.y, lat.origin.z]);
        t.lat_pitch
            .extend_from_slice(&[lat.pitch.x, lat.pitch.y, lat.pitch.z]);
        t.lat_shape.extend_from_slice(&[
            lat.shape[0] as i32,
            lat.shape[1] as i32,
            lat.shape[2] as i32,
        ]);
        let off = t.lat_universes.len() as i32;
        for u in &lat.universes {
            t.lat_universes.push(u.0 as i32);
        }
        t.lat_universes_off.push(off);
    }

    t
}

// ── Device-side handles ─────────────────────────────────────────────

pub struct GpuRecursiveContext {
    _ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub k_find_cell_batch: CudaFunction,
    pub k_trace_step_batch: CudaFunction,
    pub k_multi_step_walk: CudaFunction,
    pub k_const_xs_transport: CudaFunction,
    pub k_transport_recursive: CudaFunction,
    // Geometry tables on device.
    surf_type: CudaSlice<i32>,
    surf_params: CudaSlice<f64>,
    surf_bc: CudaSlice<i32>,
    cell_region_off: CudaSlice<i32>,
    cell_region_len: CudaSlice<i32>,
    cell_fill_type: CudaSlice<i32>,
    cell_fill_data: CudaSlice<i32>,
    cell_aabb_min: CudaSlice<f64>,
    cell_aabb_max: CudaSlice<f64>,
    region_op: CudaSlice<i32>,
    region_arg: CudaSlice<i32>,
    univ_cells_off: CudaSlice<i32>,
    univ_cells_len: CudaSlice<i32>,
    univ_surfaces_off: CudaSlice<i32>,
    univ_surfaces_len: CudaSlice<i32>,
    univ_cell_indices: CudaSlice<i32>,
    univ_surface_indices: CudaSlice<i32>,
    lat_origin: CudaSlice<f64>,
    lat_pitch: CudaSlice<f64>,
    lat_shape: CudaSlice<i32>,
    lat_universes_off: CudaSlice<i32>,
    lat_universes: CudaSlice<i32>,
    // Per-thread scratch evals — sized as `n_surfaces * max_threads`.
    evals_scratch: CudaSlice<f64>,
    // Scalars retained for kernel arg packing.
    n_surfaces: i32,
    root_universe: i32,
    n_threads_max: usize,
}

const RECURSIVE_DEVICE: &str = include_str!("../gpu/cuda/geom_recursive.cu");
const RECURSIVE_KERNELS: &str = include_str!("../gpu/cuda/geom_recursive_kernels.cu");
const CONST_XS_KERNEL: &str = include_str!("../gpu/cuda/transport_recursive_const.cu");
const TRANSPORT_KERNELS: &str = include_str!("../gpu/cuda/transport.cu");
const TRANSPORT_RECURSIVE: &str = include_str!("../gpu/cuda/transport_recursive.cu");

fn assemble_kernel_source() -> String {
    // NVRTC has no concept of source-include paths — concatenate the
    // device helpers and the kernel entries into a single string and
    // strip every `#include "..."` line (they'd otherwise fail to
    // resolve at compile time).
    let strip = |src: &str| -> String {
        src.lines()
            .filter(|line| {
                let t = line.trim_start();
                !(t.starts_with("#include \"geom_recursive.cu\"")
                    || t.starts_with("#include \"transport.cu\""))
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    // Order matters: device helpers and Params layout from transport.cu
    // come first, then the recursive geometry primitives, then the
    // kernels that consume both.
    format!(
        "{}\n{RECURSIVE_DEVICE}\n{}\n{}\n{}",
        strip(TRANSPORT_KERNELS),
        strip(RECURSIVE_KERNELS),
        strip(CONST_XS_KERNEL),
        strip(TRANSPORT_RECURSIVE),
    )
}

impl GpuRecursiveContext {
    /// Build a context for `geom`. The kernel is compiled with NVRTC
    /// and the geometry tables are uploaded once.
    pub fn build(geom: &Geometry, n_threads_max: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;
        let stream = ctx.default_stream();

        // Compile recursive kernels.
        let source = assemble_kernel_source();
        let ptx = nvrtc::compile_ptx_with_opts(
            &source,
            nvrtc::CompileOptions {
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map_err(|e| format!("NVRTC compile: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("module load: {e}"))?;
        let k_find_cell_batch = module
            .load_function("find_cell_batch")
            .map_err(|e| format!("kernel load: {e}"))?;
        let k_trace_step_batch = module
            .load_function("trace_step_batch")
            .map_err(|e| format!("kernel load (trace): {e}"))?;
        let k_multi_step_walk = module
            .load_function("multi_step_walk")
            .map_err(|e| format!("kernel load (walk): {e}"))?;
        let k_const_xs_transport = module
            .load_function("const_xs_transport_persistent")
            .map_err(|e| format!("kernel load (const_xs): {e}"))?;
        let k_transport_recursive = module
            .load_function("transport_recursive_persistent")
            .map_err(|e| format!("kernel load (transport_recursive): {e}"))?;

        // Build host SoA + upload.
        let t = build_host_tables(geom);
        let n_surfaces = t.surf_type.len() as i32;
        let root_universe = geom.root_universe.0 as i32;

        let surf_type = stream.clone_htod(&t.surf_type).map_err(|e| e.to_string())?;
        let surf_params = stream.clone_htod(&t.surf_params).map_err(|e| e.to_string())?;
        let surf_bc = stream.clone_htod(&t.surf_bc).map_err(|e| e.to_string())?;
        let cell_region_off = stream
            .clone_htod(&t.cell_region_off)
            .map_err(|e| e.to_string())?;
        let cell_region_len = stream
            .clone_htod(&t.cell_region_len)
            .map_err(|e| e.to_string())?;
        let cell_fill_type = stream
            .clone_htod(&t.cell_fill_type)
            .map_err(|e| e.to_string())?;
        let cell_fill_data = stream
            .clone_htod(&t.cell_fill_data)
            .map_err(|e| e.to_string())?;
        let cell_aabb_min = stream
            .clone_htod(&t.cell_aabb_min)
            .map_err(|e| e.to_string())?;
        let cell_aabb_max = stream
            .clone_htod(&t.cell_aabb_max)
            .map_err(|e| e.to_string())?;
        let region_op = stream.clone_htod(&t.region_op).map_err(|e| e.to_string())?;
        let region_arg = stream.clone_htod(&t.region_arg).map_err(|e| e.to_string())?;
        let univ_cells_off = stream
            .clone_htod(&t.univ_cells_off)
            .map_err(|e| e.to_string())?;
        let univ_cells_len = stream
            .clone_htod(&t.univ_cells_len)
            .map_err(|e| e.to_string())?;
        let univ_surfaces_off = stream
            .clone_htod(&t.univ_surfaces_off)
            .map_err(|e| e.to_string())?;
        let univ_surfaces_len = stream
            .clone_htod(&t.univ_surfaces_len)
            .map_err(|e| e.to_string())?;
        let univ_cell_indices = stream
            .clone_htod(&t.univ_cell_indices)
            .map_err(|e| e.to_string())?;
        let univ_surface_indices = stream
            .clone_htod(&t.univ_surface_indices)
            .map_err(|e| e.to_string())?;
        let lat_origin = if t.lat_origin.is_empty() {
            stream.alloc_zeros::<f64>(1).map_err(|e| e.to_string())?
        } else {
            stream.clone_htod(&t.lat_origin).map_err(|e| e.to_string())?
        };
        let lat_pitch = if t.lat_pitch.is_empty() {
            stream.alloc_zeros::<f64>(1).map_err(|e| e.to_string())?
        } else {
            stream.clone_htod(&t.lat_pitch).map_err(|e| e.to_string())?
        };
        let lat_shape = if t.lat_shape.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream.clone_htod(&t.lat_shape).map_err(|e| e.to_string())?
        };
        let lat_universes_off = if t.lat_universes_off.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream
                .clone_htod(&t.lat_universes_off)
                .map_err(|e| e.to_string())?
        };
        let lat_universes = if t.lat_universes.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream
                .clone_htod(&t.lat_universes)
                .map_err(|e| e.to_string())?
        };

        // Per-thread evals scratch.
        let evals_scratch = stream
            .alloc_zeros::<f64>(n_surfaces as usize * n_threads_max)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            _ctx: ctx,
            stream,
            k_find_cell_batch,
            k_trace_step_batch,
            k_multi_step_walk,
            k_const_xs_transport,
            k_transport_recursive,
            surf_type,
            surf_params,
            surf_bc,
            cell_region_off,
            cell_region_len,
            cell_fill_type,
            cell_fill_data,
            cell_aabb_min,
            cell_aabb_max,
            region_op,
            region_arg,
            univ_cells_off,
            univ_cells_len,
            univ_surfaces_off,
            univ_surfaces_len,
            univ_cell_indices,
            univ_surface_indices,
            lat_origin,
            lat_pitch,
            lat_shape,
            lat_universes_off,
            lat_universes,
            evals_scratch,
            n_surfaces,
            root_universe,
            n_threads_max,
        })
    }

    /// Run `find_cell_recursive` on `points` (SoA xs/ys/zs) and return
    /// the deepest cell index for each point (-1 on leakage).
    pub fn find_cell_batch(&self, points: &[(f64, f64, f64)]) -> Result<Vec<i32>, String> {
        let n = points.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.n_threads_max {
            return Err(format!(
                "batch size {n} exceeds n_threads_max {}",
                self.n_threads_max
            ));
        }
        let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = points.iter().map(|p| p.2).collect();
        let xs = self.stream.clone_htod(&xs).map_err(|e| e.to_string())?;
        let ys = self.stream.clone_htod(&ys).map_err(|e| e.to_string())?;
        let zs = self.stream.clone_htod(&zs).map_err(|e| e.to_string())?;
        let mut out = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| e.to_string())?;

        let block = 128_u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;

        let stream = &self.stream;
        let mut launch = stream.launch_builder(&self.k_find_cell_batch);
        launch
            .arg(&xs)
            .arg(&ys)
            .arg(&zs)
            .arg(&n_i32)
            // surfaces
            .arg(&self.surf_type)
            .arg(&self.surf_params)
            .arg(&self.surf_bc)
            .arg(&n_surf)
            // cells
            .arg(&self.cell_region_off)
            .arg(&self.cell_region_len)
            .arg(&self.cell_fill_type)
            .arg(&self.cell_fill_data)
            .arg(&self.cell_aabb_min)
            .arg(&self.cell_aabb_max)
            // region tree
            .arg(&self.region_op)
            .arg(&self.region_arg)
            // universes
            .arg(&self.univ_cells_off)
            .arg(&self.univ_cells_len)
            .arg(&self.univ_surfaces_off)
            .arg(&self.univ_surfaces_len)
            .arg(&self.univ_cell_indices)
            .arg(&self.univ_surface_indices)
            .arg(&root)
            // lattices
            .arg(&self.lat_origin)
            .arg(&self.lat_pitch)
            .arg(&self.lat_shape)
            .arg(&self.lat_universes_off)
            .arg(&self.lat_universes)
            // scratch + output
            .arg(&self.evals_scratch)
            .arg(&mut out);
        // SAFETY: kernel signature matches the argument list above.
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }

        let host_out = self.stream.clone_dtoh(&out).map_err(|e| e.to_string())?;
        let _ = (xs, ys, zs);
        Ok(host_out)
    }
}

/// One trace_step result: distance, surface idx (-1 = grid line),
/// boundary condition (matches the device-side enum), deepest cell
/// idx of the next stack (-1 = leakage).
#[derive(Debug, Clone, Copy)]
pub struct GpuTraceResult {
    pub distance: f64,
    pub surface_idx: i32,
    pub bc: i32,
    pub next_deepest_cell: i32,
}

impl GpuRecursiveContext {
    /// Run `find_cell + trace_step_recursive` on a batch of (pos, dir)
    /// pairs and return the per-particle event distance, surface
    /// index, BC, and the deepest cell of the re-resolved next stack.
    pub fn trace_step_batch(
        &self,
        positions: &[(f64, f64, f64)],
        directions: &[(f64, f64, f64)],
    ) -> Result<Vec<GpuTraceResult>, String> {
        let n = positions.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n != directions.len() {
            return Err("position / direction lengths differ".into());
        }
        if n > self.n_threads_max {
            return Err(format!(
                "batch size {n} exceeds n_threads_max {}",
                self.n_threads_max
            ));
        }

        let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = positions.iter().map(|p| p.2).collect();
        let dxs: Vec<f64> = directions.iter().map(|d| d.0).collect();
        let dys: Vec<f64> = directions.iter().map(|d| d.1).collect();
        let dzs: Vec<f64> = directions.iter().map(|d| d.2).collect();
        let xs = self.stream.clone_htod(&xs).map_err(|e| e.to_string())?;
        let ys = self.stream.clone_htod(&ys).map_err(|e| e.to_string())?;
        let zs = self.stream.clone_htod(&zs).map_err(|e| e.to_string())?;
        let dxs = self.stream.clone_htod(&dxs).map_err(|e| e.to_string())?;
        let dys = self.stream.clone_htod(&dys).map_err(|e| e.to_string())?;
        let dzs = self.stream.clone_htod(&dzs).map_err(|e| e.to_string())?;
        let mut out_dist = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| e.to_string())?;
        let mut out_surf = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| e.to_string())?;
        let mut out_bc = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| e.to_string())?;
        let mut out_next = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| e.to_string())?;

        let block = 128_u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;

        let mut launch = self.stream.launch_builder(&self.k_trace_step_batch);
        launch
            .arg(&xs)
            .arg(&ys)
            .arg(&zs)
            .arg(&dxs)
            .arg(&dys)
            .arg(&dzs)
            .arg(&n_i32)
            .arg(&self.surf_type)
            .arg(&self.surf_params)
            .arg(&self.surf_bc)
            .arg(&n_surf)
            .arg(&self.cell_region_off)
            .arg(&self.cell_region_len)
            .arg(&self.cell_fill_type)
            .arg(&self.cell_fill_data)
            .arg(&self.cell_aabb_min)
            .arg(&self.cell_aabb_max)
            .arg(&self.region_op)
            .arg(&self.region_arg)
            .arg(&self.univ_cells_off)
            .arg(&self.univ_cells_len)
            .arg(&self.univ_surfaces_off)
            .arg(&self.univ_surfaces_len)
            .arg(&self.univ_cell_indices)
            .arg(&self.univ_surface_indices)
            .arg(&root)
            .arg(&self.lat_origin)
            .arg(&self.lat_pitch)
            .arg(&self.lat_shape)
            .arg(&self.lat_universes_off)
            .arg(&self.lat_universes)
            .arg(&self.evals_scratch)
            .arg(&mut out_dist)
            .arg(&mut out_surf)
            .arg(&mut out_bc)
            .arg(&mut out_next);
        // SAFETY: kernel signature matches argument list.
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }

        let dist_h = self.stream.clone_dtoh(&out_dist).map_err(|e| e.to_string())?;
        let surf_h = self.stream.clone_dtoh(&out_surf).map_err(|e| e.to_string())?;
        let bc_h = self.stream.clone_dtoh(&out_bc).map_err(|e| e.to_string())?;
        let next_h = self.stream.clone_dtoh(&out_next).map_err(|e| e.to_string())?;
        let _ = (xs, ys, zs, dxs, dys, dzs);

        let out: Vec<GpuTraceResult> = (0..n)
            .map(|i| GpuTraceResult {
                distance: dist_h[i],
                surface_idx: surf_h[i],
                bc: bc_h[i],
                next_deepest_cell: next_h[i],
            })
            .collect();
        Ok(out)
    }
}

/// Output of one K-step walk per particle.
#[derive(Debug, Clone, Copy)]
pub struct GpuWalkResult {
    pub final_pos: (f64, f64, f64),
    pub n_steps: i32,
    pub final_cell: i32,
}

impl GpuRecursiveContext {
    /// Run a deterministic K-step pure-geometry walk per particle.
    /// At every step the kernel takes the next event distance from
    /// `gr_trace_step`, advances, and on a reflective surface flips
    /// the corresponding axis-aligned direction component. No
    /// physics — just geometry traversal. Used to validate the
    /// recursive-geometry-in-transport-context plumbing end-to-end.
    pub fn multi_step_walk(
        &self,
        positions: &[(f64, f64, f64)],
        directions: &[(f64, f64, f64)],
        max_steps: i32,
    ) -> Result<Vec<GpuWalkResult>, String> {
        let n = positions.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n != directions.len() {
            return Err("position / direction lengths differ".into());
        }
        if n > self.n_threads_max {
            return Err(format!(
                "batch size {n} exceeds n_threads_max {}",
                self.n_threads_max
            ));
        }
        let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = positions.iter().map(|p| p.2).collect();
        let dxs: Vec<f64> = directions.iter().map(|d| d.0).collect();
        let dys: Vec<f64> = directions.iter().map(|d| d.1).collect();
        let dzs: Vec<f64> = directions.iter().map(|d| d.2).collect();
        let xs = self.stream.clone_htod(&xs).map_err(|e| e.to_string())?;
        let ys = self.stream.clone_htod(&ys).map_err(|e| e.to_string())?;
        let zs = self.stream.clone_htod(&zs).map_err(|e| e.to_string())?;
        let dxs = self.stream.clone_htod(&dxs).map_err(|e| e.to_string())?;
        let dys = self.stream.clone_htod(&dys).map_err(|e| e.to_string())?;
        let dzs = self.stream.clone_htod(&dzs).map_err(|e| e.to_string())?;
        let mut out_x = self.stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
        let mut out_y = self.stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
        let mut out_z = self.stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
        let mut out_steps = self.stream.alloc_zeros::<i32>(n).map_err(|e| e.to_string())?;
        let mut out_cell = self.stream.alloc_zeros::<i32>(n).map_err(|e| e.to_string())?;

        let block = 128_u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;

        let mut launch = self.stream.launch_builder(&self.k_multi_step_walk);
        launch
            .arg(&xs)
            .arg(&ys)
            .arg(&zs)
            .arg(&dxs)
            .arg(&dys)
            .arg(&dzs)
            .arg(&n_i32)
            .arg(&max_steps)
            .arg(&self.surf_type)
            .arg(&self.surf_params)
            .arg(&self.surf_bc)
            .arg(&n_surf)
            .arg(&self.cell_region_off)
            .arg(&self.cell_region_len)
            .arg(&self.cell_fill_type)
            .arg(&self.cell_fill_data)
            .arg(&self.cell_aabb_min)
            .arg(&self.cell_aabb_max)
            .arg(&self.region_op)
            .arg(&self.region_arg)
            .arg(&self.univ_cells_off)
            .arg(&self.univ_cells_len)
            .arg(&self.univ_surfaces_off)
            .arg(&self.univ_surfaces_len)
            .arg(&self.univ_cell_indices)
            .arg(&self.univ_surface_indices)
            .arg(&root)
            .arg(&self.lat_origin)
            .arg(&self.lat_pitch)
            .arg(&self.lat_shape)
            .arg(&self.lat_universes_off)
            .arg(&self.lat_universes)
            .arg(&self.evals_scratch)
            .arg(&mut out_x)
            .arg(&mut out_y)
            .arg(&mut out_z)
            .arg(&mut out_steps)
            .arg(&mut out_cell);
        // SAFETY: kernel signature matches argument list.
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }
        let xh = self.stream.clone_dtoh(&out_x).map_err(|e| e.to_string())?;
        let yh = self.stream.clone_dtoh(&out_y).map_err(|e| e.to_string())?;
        let zh = self.stream.clone_dtoh(&out_z).map_err(|e| e.to_string())?;
        let sh = self.stream.clone_dtoh(&out_steps).map_err(|e| e.to_string())?;
        let ch = self.stream.clone_dtoh(&out_cell).map_err(|e| e.to_string())?;
        let _ = (xs, ys, zs, dxs, dys, dzs);
        Ok((0..n)
            .map(|i| GpuWalkResult {
                final_pos: (xh[i], yh[i], zh[i]),
                n_steps: sh[i],
                final_cell: ch[i],
            })
            .collect())
    }
}

/// Constant cross sections per material: σ_t, σ_a, σ_f, ν̄.
#[derive(Debug, Clone, Copy)]
pub struct ConstXs {
    pub sigma_t: f64,
    pub sigma_a: f64,
    pub sigma_f: f64,
    pub nu_bar: f64,
}

/// Result of one batch through `const_xs_transport`.
#[derive(Debug, Clone)]
pub struct ConstXsBatch {
    pub fission_sites: Vec<(f64, f64, f64)>,
    pub n_collisions: u64,
    pub n_absorptions: u64,
    pub n_fissions: u64,
    pub n_leakage: u64,
    pub n_surf_xings: u64,
}

impl GpuRecursiveContext {
    /// Run one batch of constant-XS transport on GPU. Each particle
    /// is transported to absorption / leakage / step cap. Fission
    /// sites are appended to a shared bank via `atomicAdd`.
    ///
    /// Inputs and outputs:
    ///   * `positions`, `directions`, `rng_seeds` — per-particle.
    ///     RNG seeds are independently advanced; the same seed on
    ///     CPU and GPU does NOT guarantee bit-identical histories
    ///     because the order of (collision-distance, surface-distance)
    ///     decisions can flip on float-rounding ties. Aggregate counts
    ///     should agree within MC noise.
    ///   * `materials` — `ConstXs` per material id.
    ///   * `max_events_per_history` — safety cap.
    ///
    /// Returns the fission-site bank built up by all the surviving
    /// histories, plus aggregate event counts.
    #[allow(clippy::too_many_arguments)]
    pub fn const_xs_transport(
        &self,
        positions: &[(f64, f64, f64)],
        directions: &[(f64, f64, f64)],
        rng_seeds: &[(u64, u64)],
        materials: &[ConstXs],
        max_events_per_history: i32,
        fis_capacity: usize,
    ) -> Result<ConstXsBatch, String> {
        let n = positions.len();
        if n == 0 {
            return Ok(ConstXsBatch {
                fission_sites: Vec::new(),
                n_collisions: 0,
                n_absorptions: 0,
                n_fissions: 0,
                n_leakage: 0,
                n_surf_xings: 0,
            });
        }
        if directions.len() != n || rng_seeds.len() != n {
            return Err("position / direction / rng_seeds length mismatch".into());
        }

        // Particle SoA.
        let xs: Vec<f64> = positions.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = positions.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = positions.iter().map(|p| p.2).collect();
        let dxs: Vec<f64> = directions.iter().map(|d| d.0).collect();
        let dys: Vec<f64> = directions.iter().map(|d| d.1).collect();
        let dzs: Vec<f64> = directions.iter().map(|d| d.2).collect();
        let alive: Vec<i32> = vec![1; n];
        let rng_state: Vec<u64> = rng_seeds.iter().map(|s| s.0).collect();
        let rng_inc: Vec<u64> = rng_seeds.iter().map(|s| s.1).collect();
        let mut d_xs = self.stream.clone_htod(&xs).map_err(|e| e.to_string())?;
        let mut d_ys = self.stream.clone_htod(&ys).map_err(|e| e.to_string())?;
        let mut d_zs = self.stream.clone_htod(&zs).map_err(|e| e.to_string())?;
        let mut d_dxs = self.stream.clone_htod(&dxs).map_err(|e| e.to_string())?;
        let mut d_dys = self.stream.clone_htod(&dys).map_err(|e| e.to_string())?;
        let mut d_dzs = self.stream.clone_htod(&dzs).map_err(|e| e.to_string())?;
        let mut d_alive = self.stream.clone_htod(&alive).map_err(|e| e.to_string())?;
        let mut d_rng_state = self
            .stream
            .clone_htod(&rng_state)
            .map_err(|e| e.to_string())?;
        let mut d_rng_inc = self.stream.clone_htod(&rng_inc).map_err(|e| e.to_string())?;

        // Materials table: [σ_t, σ_a, σ_f, ν̄] per material.
        let mut mat_flat: Vec<f64> = Vec::with_capacity(materials.len() * 4);
        for m in materials {
            mat_flat.extend_from_slice(&[m.sigma_t, m.sigma_a, m.sigma_f, m.nu_bar]);
        }
        let d_mat = self.stream.clone_htod(&mat_flat).map_err(|e| e.to_string())?;
        let n_materials = materials.len() as i32;

        // Material override tables — placeholder empties (no overrides
        // wired through this kernel yet; the assembly demo doesn't use
        // them, and #16's CPU lookup is the source of truth elsewhere).
        let dummy_off: Vec<i32> = vec![-1; self.lat_origin.len() / 3 + 1];
        let dummy_count: Vec<i32> = vec![0; self.lat_origin.len() / 3 + 1];
        let d_lat_override_off = self.stream.clone_htod(&dummy_off).map_err(|e| e.to_string())?;
        let d_lat_override_count = self.stream.clone_htod(&dummy_count).map_err(|e| e.to_string())?;
        let d_override_lat_idx = self.stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let d_override_cell_idx = self.stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let d_override_mat = self.stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;

        // Fission bank.
        let mut d_fis_x = self
            .stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_y = self
            .stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_z = self
            .stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_count = self.stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;

        // Counter slots.
        let mut d_cnt_coll = self
            .stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| e.to_string())?;
        let mut d_cnt_abs = self.stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_fis = self.stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_leak = self.stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_surf = self.stream.alloc_zeros::<u64>(1).map_err(|e| e.to_string())?;

        let block = 128_u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;
        let n_lat = (self.lat_origin.len() / 3) as i32;
        let fis_cap_i = fis_capacity as i32;

        let mut launch = self.stream.launch_builder(&self.k_const_xs_transport);
        launch
            .arg(&mut d_xs)
            .arg(&mut d_ys)
            .arg(&mut d_zs)
            .arg(&mut d_dxs)
            .arg(&mut d_dys)
            .arg(&mut d_dzs)
            .arg(&mut d_alive)
            .arg(&mut d_rng_state)
            .arg(&mut d_rng_inc)
            .arg(&n_i32)
            .arg(&max_events_per_history)
            .arg(&d_mat)
            .arg(&n_materials)
            .arg(&self.surf_type)
            .arg(&self.surf_params)
            .arg(&self.surf_bc)
            .arg(&n_surf)
            .arg(&self.cell_region_off)
            .arg(&self.cell_region_len)
            .arg(&self.cell_fill_type)
            .arg(&self.cell_fill_data)
            .arg(&self.cell_aabb_min)
            .arg(&self.cell_aabb_max)
            .arg(&self.region_op)
            .arg(&self.region_arg)
            .arg(&self.univ_cells_off)
            .arg(&self.univ_cells_len)
            .arg(&self.univ_surfaces_off)
            .arg(&self.univ_surfaces_len)
            .arg(&self.univ_cell_indices)
            .arg(&self.univ_surface_indices)
            .arg(&root)
            .arg(&self.lat_origin)
            .arg(&self.lat_pitch)
            .arg(&self.lat_shape)
            .arg(&self.lat_universes_off)
            .arg(&self.lat_universes)
            .arg(&n_lat)
            .arg(&d_lat_override_off)
            .arg(&d_lat_override_count)
            .arg(&d_override_lat_idx)
            .arg(&d_override_cell_idx)
            .arg(&d_override_mat)
            .arg(&self.evals_scratch)
            .arg(&mut d_fis_x)
            .arg(&mut d_fis_y)
            .arg(&mut d_fis_z)
            .arg(&mut d_fis_count)
            .arg(&fis_cap_i)
            .arg(&mut d_cnt_coll)
            .arg(&mut d_cnt_abs)
            .arg(&mut d_cnt_fis)
            .arg(&mut d_cnt_leak)
            .arg(&mut d_cnt_surf);
        // SAFETY: kernel signature matches the argument list.
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }

        let fis_count_h = self
            .stream
            .clone_dtoh(&d_fis_count)
            .map_err(|e| e.to_string())?[0]
            .max(0) as usize;
        let n_banked = fis_count_h.min(fis_capacity);
        let fis_x_h = self.stream.clone_dtoh(&d_fis_x).map_err(|e| e.to_string())?;
        let fis_y_h = self.stream.clone_dtoh(&d_fis_y).map_err(|e| e.to_string())?;
        let fis_z_h = self.stream.clone_dtoh(&d_fis_z).map_err(|e| e.to_string())?;
        let fission_sites: Vec<(f64, f64, f64)> = (0..n_banked)
            .map(|i| (fis_x_h[i], fis_y_h[i], fis_z_h[i]))
            .collect();
        let cnt_coll = self
            .stream
            .clone_dtoh(&d_cnt_coll)
            .map_err(|e| e.to_string())?[0];
        let cnt_abs = self
            .stream
            .clone_dtoh(&d_cnt_abs)
            .map_err(|e| e.to_string())?[0];
        let cnt_fis = self
            .stream
            .clone_dtoh(&d_cnt_fis)
            .map_err(|e| e.to_string())?[0];
        let cnt_leak = self
            .stream
            .clone_dtoh(&d_cnt_leak)
            .map_err(|e| e.to_string())?[0];
        let cnt_surf = self
            .stream
            .clone_dtoh(&d_cnt_surf)
            .map_err(|e| e.to_string())?[0];
        let _ = (
            d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs, d_alive, d_rng_state, d_rng_inc, d_mat,
            d_lat_override_off, d_lat_override_count, d_override_lat_idx, d_override_cell_idx,
            d_override_mat,
        );

        Ok(ConstXsBatch {
            fission_sites,
            n_collisions: cnt_coll,
            n_absorptions: cnt_abs,
            n_fissions: cnt_fis,
            n_leakage: cnt_leak,
            n_surf_xings: cnt_surf,
        })
    }
}

/// Result of one batch through `transport_recursive_persistent`.
#[derive(Debug, Clone)]
pub struct RecursiveTransportBatch {
    /// (x, y, z, energy) for each banked fission / (n,2n) / (n,3n) site.
    pub fission_bank: Vec<(f64, f64, f64, f64)>,
    pub n_collisions: u64,
    pub n_fissions: u64,
    pub n_leakage: u64,
    pub n_surf_xings: u64,
    pub k_eff: f64,
}

impl GpuRecursiveContext {
    /// Run one batch of full-physics neutron transport on the recursive
    /// geometry. Cross-section data is supplied via the
    /// `GpuTransportContext` upload-* paths (SVD / Pointwise / WMP / URR
    /// / discrete levels / S(α,β)). Each particle is transported to
    /// absorption / leakage / max-events.
    ///
    /// Inputs:
    /// * `source_bank` — per-particle (x, y, z, energy) at birth.
    /// * `rng_seeds` — per-particle PCG-64 (state, inc) pair. Same seed
    ///   on CPU and GPU does **not** guarantee bit-identical histories
    ///   because float-rounding ties between collision distance and
    ///   surface distance can flip event ordering between the two
    ///   implementations; aggregate counts agree within MC noise.
    /// * `mat_kT` — per-material temperature in eV (kT). Used for the
    ///   free-gas thermal threshold.
    /// * `sab_nuc_idx` — index of the nuclide carrying S(α,β) data,
    ///   or `-1` if none applies.
    /// * `gpu_t` — used only to build the `TransportParams` buffer
    ///   (centralised slot layout); the kernel itself runs on this
    ///   recursive context.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn transport_recursive(
        &self,
        gpu_t: &crate::gpu_transport::GpuTransportContext,
        nuc_data: &crate::gpu_transport::GpuNuclideData,
        mat_data: &crate::gpu_transport::GpuMaterialData,
        sab_data: &crate::gpu_transport::GpuSabData,
        wmp_data: &crate::gpu_transport::GpuWmpData,
        source_bank: &[(f64, f64, f64, f64)],
        rng_seeds: &[(u64, u64)],
        mat_kT: &[f64],
        sab_nuc_idx: i32,
        max_events_per_history: i32,
        fis_capacity: usize,
    ) -> Result<RecursiveTransportBatch, String> {
        let n = source_bank.len();
        if n == 0 {
            return Ok(RecursiveTransportBatch {
                fission_bank: Vec::new(),
                n_collisions: 0,
                n_fissions: 0,
                n_leakage: 0,
                n_surf_xings: 0,
                k_eff: 0.0,
            });
        }
        if rng_seeds.len() != n {
            return Err("source_bank / rng_seeds length mismatch".into());
        }

        // Particle SoA.
        let xs: Vec<f64> = source_bank.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = source_bank.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = source_bank.iter().map(|p| p.2).collect();
        let es: Vec<f64> = source_bank.iter().map(|p| p.3).collect();
        // Initial directions: isotropic, same convention as init_source.
        // We reuse the seed's first two draws to derive a direction,
        // matching the const-XS path's CPU/GPU comparison style.
        let mut dxs = Vec::with_capacity(n);
        let mut dys = Vec::with_capacity(n);
        let mut dzs = Vec::with_capacity(n);
        for &(s, inc) in rng_seeds {
            // Draw mu, phi from a private RNG so we don't perturb the
            // seed the kernel actually uses.
            let mut rng = rust_mc_sim::Pcg64::from_state(s ^ 0xA5A5_A5A5_A5A5_A5A5, inc | 1);
            let mu = 2.0 * rng.uniform() - 1.0;
            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
            let s_th = (1.0 - mu * mu).max(0.0).sqrt();
            dxs.push(s_th * phi.cos());
            dys.push(s_th * phi.sin());
            dzs.push(mu);
        }
        let alive: Vec<i32> = vec![1; n];
        let rng_state: Vec<u64> = rng_seeds.iter().map(|s| s.0).collect();
        let rng_inc: Vec<u64> = rng_seeds.iter().map(|s| s.1).collect();

        let stream = &self.stream;
        let mut d_xs = stream.clone_htod(&xs).map_err(|e| e.to_string())?;
        let mut d_ys = stream.clone_htod(&ys).map_err(|e| e.to_string())?;
        let mut d_zs = stream.clone_htod(&zs).map_err(|e| e.to_string())?;
        let mut d_dxs = stream.clone_htod(&dxs).map_err(|e| e.to_string())?;
        let mut d_dys = stream.clone_htod(&dys).map_err(|e| e.to_string())?;
        let mut d_dzs = stream.clone_htod(&dzs).map_err(|e| e.to_string())?;
        let mut d_e = stream.clone_htod(&es).map_err(|e| e.to_string())?;
        let mut d_alive = stream.clone_htod(&alive).map_err(|e| e.to_string())?;
        let mut d_rng_state = stream.clone_htod(&rng_state).map_err(|e| e.to_string())?;
        let mut d_rng_inc = stream.clone_htod(&rng_inc).map_err(|e| e.to_string())?;

        // Material-temperature table.
        let d_mat_kt = stream.clone_htod(mat_kT).map_err(|e| e.to_string())?;
        let n_materials = mat_kT.len() as i32;

        // Empty material-override tables (recursive demo doesn't use
        // distributed materials yet — the const-XS kernel uses the
        // same fallback).
        let dummy_off: Vec<i32> = vec![-1; self.lat_origin.len() / 3 + 1];
        let dummy_count: Vec<i32> = vec![0; self.lat_origin.len() / 3 + 1];
        let d_lat_override_off = stream.clone_htod(&dummy_off).map_err(|e| e.to_string())?;
        let d_lat_override_count = stream
            .clone_htod(&dummy_count)
            .map_err(|e| e.to_string())?;
        let d_override_lat_idx = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let d_override_cell_idx = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let d_override_mat = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;

        // Fission bank.
        let mut d_fis_x = stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_y = stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_z = stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_e = stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_w = stream
            .alloc_zeros::<f64>(fis_capacity.max(1))
            .map_err(|e| e.to_string())?;
        let mut d_fis_count = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;

        // Counter slots (i32 to match transport.cu's atomic style).
        let mut d_cnt_coll = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_fis = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_leak = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;
        let mut d_cnt_surf = stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?;

        // Build the packed TransportParams buffer using GpuTransportContext's
        // helper. P_GEOM_TYPE is irrelevant here (the recursive kernel
        // bypasses the find_cell switch) — we pass 0 for compactness.
        let params_vec =
            gpu_t.build_transport_params_vec(nuc_data, mat_data, sab_data, wmp_data, 0);
        let d_params = stream.clone_htod(&params_vec).map_err(|e| e.to_string())?;

        let block = 128_u32;
        let grid = (n as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;
        let n_lat = (self.lat_origin.len() / 3) as i32;
        let max_fis_i = fis_capacity as i32;

        let mut launch = stream.launch_builder(&self.k_transport_recursive);
        launch
            .arg(&d_params)
            // particle SoA
            .arg(&mut d_xs)
            .arg(&mut d_ys)
            .arg(&mut d_zs)
            .arg(&mut d_dxs)
            .arg(&mut d_dys)
            .arg(&mut d_dzs)
            .arg(&mut d_e)
            .arg(&mut d_alive)
            .arg(&mut d_rng_state)
            .arg(&mut d_rng_inc)
            .arg(&n_i32)
            .arg(&max_events_per_history)
            // recursive geometry tables
            .arg(&self.surf_type)
            .arg(&self.surf_params)
            .arg(&self.surf_bc)
            .arg(&n_surf)
            .arg(&self.cell_region_off)
            .arg(&self.cell_region_len)
            .arg(&self.cell_fill_type)
            .arg(&self.cell_fill_data)
            .arg(&self.cell_aabb_min)
            .arg(&self.cell_aabb_max)
            .arg(&self.region_op)
            .arg(&self.region_arg)
            .arg(&self.univ_cells_off)
            .arg(&self.univ_cells_len)
            .arg(&self.univ_surfaces_off)
            .arg(&self.univ_surfaces_len)
            .arg(&self.univ_cell_indices)
            .arg(&self.univ_surface_indices)
            .arg(&root)
            .arg(&self.lat_origin)
            .arg(&self.lat_pitch)
            .arg(&self.lat_shape)
            .arg(&self.lat_universes_off)
            .arg(&self.lat_universes)
            .arg(&n_lat)
            .arg(&d_lat_override_off)
            .arg(&d_lat_override_count)
            .arg(&d_override_lat_idx)
            .arg(&d_override_cell_idx)
            .arg(&d_override_mat)
            .arg(&d_mat_kt)
            .arg(&n_materials)
            .arg(&sab_nuc_idx)
            .arg(&self.evals_scratch)
            // fission bank
            .arg(&mut d_fis_x)
            .arg(&mut d_fis_y)
            .arg(&mut d_fis_z)
            .arg(&mut d_fis_e)
            .arg(&mut d_fis_w)
            .arg(&mut d_fis_count)
            .arg(&max_fis_i)
            // counters
            .arg(&mut d_cnt_coll)
            .arg(&mut d_cnt_fis)
            .arg(&mut d_cnt_leak)
            .arg(&mut d_cnt_surf);
        // SAFETY: kernel signature matches the argument list above
        // (transport_recursive_persistent in transport_recursive.cu).
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }

        let fis_count = stream
            .clone_dtoh(&d_fis_count)
            .map_err(|e| e.to_string())?[0]
            .max(0) as usize;
        let n_banked = fis_count.min(fis_capacity);
        let fx = stream.clone_dtoh(&d_fis_x).map_err(|e| e.to_string())?;
        let fy = stream.clone_dtoh(&d_fis_y).map_err(|e| e.to_string())?;
        let fz = stream.clone_dtoh(&d_fis_z).map_err(|e| e.to_string())?;
        let fe = stream.clone_dtoh(&d_fis_e).map_err(|e| e.to_string())?;
        let fission_bank: Vec<(f64, f64, f64, f64)> = (0..n_banked)
            .map(|i| (fx[i], fy[i], fz[i], fe[i]))
            .collect();

        let cnt_coll = stream.clone_dtoh(&d_cnt_coll).map_err(|e| e.to_string())?[0] as u64;
        let cnt_fis = stream.clone_dtoh(&d_cnt_fis).map_err(|e| e.to_string())?[0] as u64;
        let cnt_leak = stream.clone_dtoh(&d_cnt_leak).map_err(|e| e.to_string())?[0] as u64;
        let cnt_surf = stream.clone_dtoh(&d_cnt_surf).map_err(|e| e.to_string())?[0] as u64;

        // Suppress unused-warning on retained guards.
        let _ = (
            d_xs,
            d_ys,
            d_zs,
            d_dxs,
            d_dys,
            d_dzs,
            d_e,
            d_alive,
            d_rng_state,
            d_rng_inc,
            d_mat_kt,
            d_lat_override_off,
            d_lat_override_count,
            d_override_lat_idx,
            d_override_cell_idx,
            d_override_mat,
            d_params,
            d_fis_w,
        );

        let k_eff = fission_bank.len() as f64 / n as f64;
        Ok(RecursiveTransportBatch {
            fission_bank,
            n_collisions: cnt_coll,
            n_fissions: cnt_fis,
            n_leakage: cnt_leak,
            n_surf_xings: cnt_surf,
            k_eff,
        })
    }
}

/// Suppress unused-field warnings on borrowed-only device buffers.
impl GpuRecursiveContext {
    #[allow(dead_code)]
    fn keep_alive(&self) -> usize {
        std::mem::size_of_val(&self.surf_type)
            + std::mem::size_of_val(&self.surf_params)
            + std::mem::size_of_val(&self.surf_bc)
    }
}
