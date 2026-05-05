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
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
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

fn assemble_kernel_source() -> String {
    // NVRTC has no concept of source-include paths — concatenate the
    // device helpers and the kernel entry into a single string and
    // strip the `#include` line from the kernels file (it would
    // otherwise fail to resolve at compile time).
    let kernels_no_include: String = RECURSIVE_KERNELS
        .lines()
        .filter(|line| !line.trim_start().starts_with("#include \"geom_recursive.cu\""))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{RECURSIVE_DEVICE}\n{kernels_no_include}")
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

        // Build host SoA + upload.
        let t = build_host_tables(geom);
        let n_surfaces = t.surf_type.len() as i32;
        let root_universe = geom.root_universe.0 as i32;

        let surf_type = stream.memcpy_stod(&t.surf_type).map_err(|e| e.to_string())?;
        let surf_params = stream.memcpy_stod(&t.surf_params).map_err(|e| e.to_string())?;
        let surf_bc = stream.memcpy_stod(&t.surf_bc).map_err(|e| e.to_string())?;
        let cell_region_off = stream
            .memcpy_stod(&t.cell_region_off)
            .map_err(|e| e.to_string())?;
        let cell_region_len = stream
            .memcpy_stod(&t.cell_region_len)
            .map_err(|e| e.to_string())?;
        let cell_fill_type = stream
            .memcpy_stod(&t.cell_fill_type)
            .map_err(|e| e.to_string())?;
        let cell_fill_data = stream
            .memcpy_stod(&t.cell_fill_data)
            .map_err(|e| e.to_string())?;
        let cell_aabb_min = stream
            .memcpy_stod(&t.cell_aabb_min)
            .map_err(|e| e.to_string())?;
        let cell_aabb_max = stream
            .memcpy_stod(&t.cell_aabb_max)
            .map_err(|e| e.to_string())?;
        let region_op = stream.memcpy_stod(&t.region_op).map_err(|e| e.to_string())?;
        let region_arg = stream.memcpy_stod(&t.region_arg).map_err(|e| e.to_string())?;
        let univ_cells_off = stream
            .memcpy_stod(&t.univ_cells_off)
            .map_err(|e| e.to_string())?;
        let univ_cells_len = stream
            .memcpy_stod(&t.univ_cells_len)
            .map_err(|e| e.to_string())?;
        let univ_surfaces_off = stream
            .memcpy_stod(&t.univ_surfaces_off)
            .map_err(|e| e.to_string())?;
        let univ_surfaces_len = stream
            .memcpy_stod(&t.univ_surfaces_len)
            .map_err(|e| e.to_string())?;
        let univ_cell_indices = stream
            .memcpy_stod(&t.univ_cell_indices)
            .map_err(|e| e.to_string())?;
        let univ_surface_indices = stream
            .memcpy_stod(&t.univ_surface_indices)
            .map_err(|e| e.to_string())?;
        let lat_origin = if t.lat_origin.is_empty() {
            stream.alloc_zeros::<f64>(1).map_err(|e| e.to_string())?
        } else {
            stream.memcpy_stod(&t.lat_origin).map_err(|e| e.to_string())?
        };
        let lat_pitch = if t.lat_pitch.is_empty() {
            stream.alloc_zeros::<f64>(1).map_err(|e| e.to_string())?
        } else {
            stream.memcpy_stod(&t.lat_pitch).map_err(|e| e.to_string())?
        };
        let lat_shape = if t.lat_shape.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream.memcpy_stod(&t.lat_shape).map_err(|e| e.to_string())?
        };
        let lat_universes_off = if t.lat_universes_off.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream
                .memcpy_stod(&t.lat_universes_off)
                .map_err(|e| e.to_string())?
        };
        let lat_universes = if t.lat_universes.is_empty() {
            stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string())?
        } else {
            stream
                .memcpy_stod(&t.lat_universes)
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
        let xs = self.stream.memcpy_stod(&xs).map_err(|e| e.to_string())?;
        let ys = self.stream.memcpy_stod(&ys).map_err(|e| e.to_string())?;
        let zs = self.stream.memcpy_stod(&zs).map_err(|e| e.to_string())?;
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

        let host_out = self.stream.memcpy_dtov(&out).map_err(|e| e.to_string())?;
        let _ = (xs, ys, zs);
        Ok(host_out)
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
