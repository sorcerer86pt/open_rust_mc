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
const FILL_HEX_LATTICE: i32 = 4;

// Hex orientation discriminants — match CUDA `GR_HEX_ORIENT_*`.
const HEX_ORIENT_Y: i32 = 0;
const HEX_ORIENT_X: i32 = 1;

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
    // Hex lattice SoA — parallel to the rect arrays. The `n_*`
    // counters match the layout in `geom_recursive.cu::GrGeometry`.
    hex_center: Vec<f64>,
    hex_pitch_xy: Vec<f64>,
    hex_pitch_z: Vec<f64>,
    hex_n_rings: Vec<i32>,
    hex_n_axial: Vec<i32>,
    hex_orientation: Vec<i32>,
    hex_universes_off: Vec<i32>,
    hex_universes: Vec<i32>,
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
        Surface::Plane { normal, offset, bc } => {
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
            CellFill::HexLattice(h) => (FILL_HEX_LATTICE, h as i32),
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

    // Hex lattices: flatten parallel SoA. The CUDA `gr_hex_*` device
    // functions consume the same per-element data layout as the CPU
    // `HexLattice` struct.
    use crate::geometry::lattice::HexOrientation;
    for hex in &geom.hex_lattices {
        t.hex_center
            .extend_from_slice(&[hex.center.x, hex.center.y, hex.center.z]);
        t.hex_pitch_xy.push(hex.pitch_xy);
        t.hex_pitch_z.push(hex.pitch_z);
        t.hex_n_rings.push(hex.n_rings as i32);
        t.hex_n_axial.push(hex.n_axial as i32);
        t.hex_orientation.push(match hex.orientation {
            HexOrientation::Y => HEX_ORIENT_Y,
            HexOrientation::X => HEX_ORIENT_X,
        });
        let off = t.hex_universes.len() as i32;
        for u in &hex.universes {
            t.hex_universes.push(u.0 as i32);
        }
        t.hex_universes_off.push(off);
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
    // Event-based pipeline kernels (Tramm et al., PHYSOR 2022 —
    // "Toward Portable GPU Acceleration of the OpenMC Monte Carlo
    // Particle Transport Code"; original formulation: Brown & Martin,
    // Prog. Nucl. Energy 14(3), 1984). Replace the single
    // persistent history kernel with a 7-stage pipeline that sorts
    // particles by reaction type between geom steps so each reaction
    // kernel sees a single code path (no warp divergence).
    pub k_eb_init_stacks: CudaFunction,
    /// PHYSOR 2022 Optimization F (continuous particle refill, opt-in).
    /// Refills dead slots from a pending pool between event steps to
    /// keep the kernel grid full through the batch tail. Always
    /// loaded; only launched when the caller passes a `RefillBuffers`
    /// to `transport_recursive_with_buffers`.
    pub k_eb_refill_dead: CudaFunction,
    pub k_eb_trace_and_sample: CudaFunction,
    pub k_eb_scan_offsets: CudaFunction,
    pub k_eb_partition: CudaFunction,
    pub k_eb_elastic: CudaFunction,
    pub k_eb_inelastic: CudaFunction,
    pub k_eb_fission: CudaFunction,
    pub k_eb_multi: CudaFunction,
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
    hex_center: CudaSlice<f64>,
    hex_pitch_xy: CudaSlice<f64>,
    hex_pitch_z: CudaSlice<f64>,
    hex_n_rings: CudaSlice<i32>,
    hex_n_axial: CudaSlice<i32>,
    hex_orientation: CudaSlice<i32>,
    hex_universes_off: CudaSlice<i32>,
    hex_universes: CudaSlice<i32>,
    n_hex_lattices: i32,
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
const TRANSPORT_EVENT_BASED: &str = include_str!("../gpu/cuda/transport_event_based.cu");

/// Event-based reaction class count — mirrors `EV_TYPE_COUNT` in
/// `transport_event_based.cu` (elastic, inelastic, fission, n2n,
/// n3n). Used by `TransportBuffers` to size the partition arrays.
pub const EV_TYPE_COUNT: usize = 5;
/// Energy-bin count in the 3-D partition key. Mirrors `EB_N_EBINS`
/// in `transport_event_based.cu`.
pub const EB_N_EBINS: usize = 16;

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
        "{}\n{RECURSIVE_DEVICE}\n{}\n{}\n{}\n{}",
        strip(TRANSPORT_KERNELS),
        strip(RECURSIVE_KERNELS),
        strip(CONST_XS_KERNEL),
        strip(TRANSPORT_RECURSIVE),
        strip(TRANSPORT_EVENT_BASED),
    )
}

impl GpuRecursiveContext {
    /// Build a context for `geom`. The kernel is compiled with NVRTC
    /// and the geometry tables are uploaded once.
    pub fn build(geom: &Geometry, n_threads_max: usize) -> Result<Self, String> {
        // CPU geometry supports per-cell Mat3 rotations propagating
        // through `CoordStack` descent (`geometry/coord.rs:75`,
        // `geometry/ray.rs:174-202`). The GPU `gr_find_cell` /
        // `gr_trace_step` kernels descend axis-aligned only — they
        // never apply a rotation matrix to the parent-frame position
        // / direction during cell entry. No ICSBEP / KARMA scene
        // currently sets `rotation` on any cell, but a custom scene
        // (VVER hex pin rotations, rotated assemblies) would silently
        // mis-find cells on the GPU. Fail loud here so the gap is
        // visible the moment a rotated scene is loaded.
        if let Some((cell_idx, _)) = geom
            .cells
            .iter()
            .enumerate()
            .find(|(_, c)| c.rotation.is_some())
        {
            return Err(format!(
                "GPU recursive geometry does not support per-cell Mat3 \
                 rotations yet (cell index {cell_idx} has `rotation = \
                 Some(...)`). CPU recursive descent applies rotations in \
                 `CoordStack::local_pos / local_dir`; the GPU descent \
                 path has no equivalent. Use Runner.Cpu for this scene \
                 or remove the cell rotation."
            ));
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init: {e}"))?;
        let stream = ctx.default_stream();

        // Compile recursive kernels.
        let source = assemble_kernel_source();
        // Pin sm_86 (Ampere / RTX A1000) so the kernel can use
        // `atomicAdd(double*, double)` — required for the spectrum-
        // hardening diagnostic tallies (e_fis_in_sum etc.). The default
        // NVRTC arch is sm_52, which lacks double-add atomics.
        let ptx = nvrtc::compile_ptx_with_opts(
            &source,
            nvrtc::CompileOptions {
                use_fast_math: Some(false),
                arch: Some("sm_86"),
                // Single source of truth for the per-material nuclide
                // cap — see `crate::MAX_NUCLIDES_PER_MATERIAL`. The
                // recursive transport kernel inherits `transport.cu`
                // via concatenation, and that header `#error`s out if
                // `MAX_NUC_PER_MAT` isn't supplied here.
                options: vec![format!(
                    "-DMAX_NUC_PER_MAT={}",
                    crate::MAX_NUCLIDES_PER_MATERIAL
                )],
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
        // Event-based pipeline.
        let k_eb_init_stacks = module
            .load_function("gr_init_stacks")
            .map_err(|e| format!("kernel load (gr_init_stacks): {e}"))?;
        let k_eb_trace_and_sample = module
            .load_function("gr_trace_and_sample")
            .map_err(|e| format!("kernel load (gr_trace_and_sample): {e}"))?;
        let k_eb_scan_offsets = module
            .load_function("gr_scan_offsets")
            .map_err(|e| format!("kernel load (gr_scan_offsets): {e}"))?;
        let k_eb_partition = module
            .load_function("gr_partition")
            .map_err(|e| format!("kernel load (gr_partition): {e}"))?;
        let k_eb_elastic = module
            .load_function("gr_elastic_event")
            .map_err(|e| format!("kernel load (gr_elastic_event): {e}"))?;
        let k_eb_inelastic = module
            .load_function("gr_inelastic_event")
            .map_err(|e| format!("kernel load (gr_inelastic_event): {e}"))?;
        let k_eb_fission = module
            .load_function("gr_fission_event")
            .map_err(|e| format!("kernel load (gr_fission_event): {e}"))?;
        let k_eb_multi = module
            .load_function("gr_multi_event")
            .map_err(|e| format!("kernel load (gr_multi_event): {e}"))?;
        let k_eb_refill_dead = module
            .load_function("gr_refill_dead")
            .map_err(|e| format!("kernel load (gr_refill_dead): {e}"))?;

        // Build host SoA + upload.
        let t = build_host_tables(geom);
        let n_surfaces = t.surf_type.len() as i32;
        let root_universe = geom.root_universe.0 as i32;

        let surf_type = stream.clone_htod(&t.surf_type).map_err(|e| e.to_string())?;
        let surf_params = stream
            .clone_htod(&t.surf_params)
            .map_err(|e| e.to_string())?;
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
        let region_arg = stream
            .clone_htod(&t.region_arg)
            .map_err(|e| e.to_string())?;
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
            stream
                .clone_htod(&t.lat_origin)
                .map_err(|e| e.to_string())?
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

        // Hex lattice buffers — same empty-fallback pattern so kernels
        // always receive a non-null device pointer.
        let alloc1_f = || stream.alloc_zeros::<f64>(1).map_err(|e| e.to_string());
        let alloc1_i = || stream.alloc_zeros::<i32>(1).map_err(|e| e.to_string());
        let hex_center = if t.hex_center.is_empty() {
            alloc1_f()?
        } else {
            stream
                .clone_htod(&t.hex_center)
                .map_err(|e| e.to_string())?
        };
        let hex_pitch_xy = if t.hex_pitch_xy.is_empty() {
            alloc1_f()?
        } else {
            stream
                .clone_htod(&t.hex_pitch_xy)
                .map_err(|e| e.to_string())?
        };
        let hex_pitch_z = if t.hex_pitch_z.is_empty() {
            alloc1_f()?
        } else {
            stream
                .clone_htod(&t.hex_pitch_z)
                .map_err(|e| e.to_string())?
        };
        let hex_n_rings = if t.hex_n_rings.is_empty() {
            alloc1_i()?
        } else {
            stream
                .clone_htod(&t.hex_n_rings)
                .map_err(|e| e.to_string())?
        };
        let hex_n_axial = if t.hex_n_axial.is_empty() {
            alloc1_i()?
        } else {
            stream
                .clone_htod(&t.hex_n_axial)
                .map_err(|e| e.to_string())?
        };
        let hex_orientation = if t.hex_orientation.is_empty() {
            alloc1_i()?
        } else {
            stream
                .clone_htod(&t.hex_orientation)
                .map_err(|e| e.to_string())?
        };
        let hex_universes_off = if t.hex_universes_off.is_empty() {
            alloc1_i()?
        } else {
            stream
                .clone_htod(&t.hex_universes_off)
                .map_err(|e| e.to_string())?
        };
        let hex_universes = if t.hex_universes.is_empty() {
            alloc1_i()?
        } else {
            stream
                .clone_htod(&t.hex_universes)
                .map_err(|e| e.to_string())?
        };
        let n_hex_lattices = geom.hex_lattices.len() as i32;

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
            k_eb_init_stacks,
            k_eb_trace_and_sample,
            k_eb_scan_offsets,
            k_eb_partition,
            k_eb_elastic,
            k_eb_inelastic,
            k_eb_fission,
            k_eb_multi,
            k_eb_refill_dead,
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
            hex_center,
            hex_pitch_xy,
            hex_pitch_z,
            hex_n_rings,
            hex_n_axial,
            hex_orientation,
            hex_universes_off,
            hex_universes,
            n_hex_lattices,
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
            // hex lattices
            .arg(&self.hex_center)
            .arg(&self.hex_pitch_xy)
            .arg(&self.hex_pitch_z)
            .arg(&self.hex_n_rings)
            .arg(&self.hex_n_axial)
            .arg(&self.hex_orientation)
            .arg(&self.hex_universes_off)
            .arg(&self.hex_universes)
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
            .arg(&self.hex_center)
            .arg(&self.hex_pitch_xy)
            .arg(&self.hex_pitch_z)
            .arg(&self.hex_n_rings)
            .arg(&self.hex_n_axial)
            .arg(&self.hex_orientation)
            .arg(&self.hex_universes_off)
            .arg(&self.hex_universes)
            .arg(&self.evals_scratch)
            .arg(&mut out_dist)
            .arg(&mut out_surf)
            .arg(&mut out_bc)
            .arg(&mut out_next);
        // SAFETY: kernel signature matches argument list.
        unsafe {
            launch.launch(cfg).map_err(|e| e.to_string())?;
        }

        let dist_h = self
            .stream
            .clone_dtoh(&out_dist)
            .map_err(|e| e.to_string())?;
        let surf_h = self
            .stream
            .clone_dtoh(&out_surf)
            .map_err(|e| e.to_string())?;
        let bc_h = self.stream.clone_dtoh(&out_bc).map_err(|e| e.to_string())?;
        let next_h = self
            .stream
            .clone_dtoh(&out_next)
            .map_err(|e| e.to_string())?;
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
        let mut out_x = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| e.to_string())?;
        let mut out_y = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| e.to_string())?;
        let mut out_z = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| e.to_string())?;
        let mut out_steps = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| e.to_string())?;
        let mut out_cell = self
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
            .arg(&self.hex_center)
            .arg(&self.hex_pitch_xy)
            .arg(&self.hex_pitch_z)
            .arg(&self.hex_n_rings)
            .arg(&self.hex_n_axial)
            .arg(&self.hex_orientation)
            .arg(&self.hex_universes_off)
            .arg(&self.hex_universes)
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
        let sh = self
            .stream
            .clone_dtoh(&out_steps)
            .map_err(|e| e.to_string())?;
        let ch = self
            .stream
            .clone_dtoh(&out_cell)
            .map_err(|e| e.to_string())?;
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
        let mut d_rng_inc = self
            .stream
            .clone_htod(&rng_inc)
            .map_err(|e| e.to_string())?;

        // Materials table: [σ_t, σ_a, σ_f, ν̄] per material.
        let mut mat_flat: Vec<f64> = Vec::with_capacity(materials.len() * 4);
        for m in materials {
            mat_flat.extend_from_slice(&[m.sigma_t, m.sigma_a, m.sigma_f, m.nu_bar]);
        }
        let d_mat = self
            .stream
            .clone_htod(&mat_flat)
            .map_err(|e| e.to_string())?;
        let n_materials = materials.len() as i32;

        // Material override tables — placeholder empties (no overrides
        // wired through this kernel yet; the assembly demo doesn't use
        // them, and #16's CPU lookup is the source of truth elsewhere).
        let dummy_off: Vec<i32> = vec![-1; self.lat_origin.len() / 3 + 1];
        let dummy_count: Vec<i32> = vec![0; self.lat_origin.len() / 3 + 1];
        let d_lat_override_off = self
            .stream
            .clone_htod(&dummy_off)
            .map_err(|e| e.to_string())?;
        let d_lat_override_count = self
            .stream
            .clone_htod(&dummy_count)
            .map_err(|e| e.to_string())?;
        let d_override_lat_idx = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| e.to_string())?;
        let d_override_cell_idx = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| e.to_string())?;
        let d_override_mat = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| e.to_string())?;

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
        let mut d_fis_count = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| e.to_string())?;

        // Counter slots.
        let mut d_cnt_coll = self
            .stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| e.to_string())?;
        let mut d_cnt_abs = self
            .stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| e.to_string())?;
        let mut d_cnt_fis = self
            .stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| e.to_string())?;
        let mut d_cnt_leak = self
            .stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| e.to_string())?;
        let mut d_cnt_surf = self
            .stream
            .alloc_zeros::<u64>(1)
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
            .arg(&self.hex_center)
            .arg(&self.hex_pitch_xy)
            .arg(&self.hex_pitch_z)
            .arg(&self.hex_n_rings)
            .arg(&self.hex_n_axial)
            .arg(&self.hex_orientation)
            .arg(&self.hex_universes_off)
            .arg(&self.hex_universes)
            .arg(&self.n_hex_lattices)
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
        let fis_x_h = self
            .stream
            .clone_dtoh(&d_fis_x)
            .map_err(|e| e.to_string())?;
        let fis_y_h = self
            .stream
            .clone_dtoh(&d_fis_y)
            .map_err(|e| e.to_string())?;
        let fis_z_h = self
            .stream
            .clone_dtoh(&d_fis_z)
            .map_err(|e| e.to_string())?;
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
            d_xs,
            d_ys,
            d_zs,
            d_dxs,
            d_dys,
            d_dzs,
            d_alive,
            d_rng_state,
            d_rng_inc,
            d_mat,
            d_lat_override_off,
            d_lat_override_count,
            d_override_lat_idx,
            d_override_cell_idx,
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
    // Per-reaction tallies (added for spectrum-hardening diagnosis;
    // see `bin/metal_stats_diag`). Mean E at each reaction is
    // `e_*_sum / n_*`. Inelastic energy loss is
    // `(e_inel_in_sum − e_inel_out_sum) / n_inelastic`.
    pub n_elastic: u64,
    pub n_inelastic: u64,
    pub n_capture: u64,
    pub e_fis_in_sum: f64,
    pub e_el_in_sum: f64,
    pub e_inel_in_sum: f64,
    pub e_inel_out_sum: f64,
    /// Squared-energy accumulators for σ(E_at_reaction). σ at fission
    /// is the diagnostic that distinguishes a tail-driven ⟨ν⟩ bias
    /// from a mean-driven one once ν(E) parity is confirmed.
    pub e_fis_in_sq_sum: f64,
    pub e_el_in_sq_sum: f64,
    pub e_inel_in_sq_sum: f64,
    /// Σ |Q| over inelastic events. ⟨|Q|⟩ = q_inel_sum / n_inelastic
    /// is the CM-frame energy lost per event. CPU and GPU should
    /// converge here if level-XS-proportional sampling is unbiased;
    /// a gap localises the metal hot bias to the level selection
    /// path.
    pub q_inel_sum: f64,
}

/// Persistent device-side buffer pool for one (n_particles, fis_cap,
/// n_materials, n_lattices) tuple. Per NVIDIA Best Practices §9.2:
/// allocate once, reuse across batches. Replaces ~25 `clone_htod` per
/// batch with in-place `memcpy_htod` + `memset_zeros`.
#[allow(non_snake_case)]
pub struct TransportBuffers {
    n: usize,
    fis_cap: usize,
    pub d_xs: CudaSlice<f64>,
    pub d_ys: CudaSlice<f64>,
    pub d_zs: CudaSlice<f64>,
    pub d_dxs: CudaSlice<f64>,
    pub d_dys: CudaSlice<f64>,
    pub d_dzs: CudaSlice<f64>,
    pub d_e: CudaSlice<f64>,
    pub d_alive: CudaSlice<i32>,
    pub d_rng_state: CudaSlice<u64>,
    pub d_rng_inc: CudaSlice<u64>,
    pub d_mat_kt: CudaSlice<f64>,
    pub d_lat_override_off: CudaSlice<i32>,
    pub d_lat_override_count: CudaSlice<i32>,
    pub d_override_lat_idx: CudaSlice<i32>,
    pub d_override_cell_idx: CudaSlice<i32>,
    pub d_override_mat: CudaSlice<i32>,
    pub d_fis_x: CudaSlice<f64>,
    pub d_fis_y: CudaSlice<f64>,
    pub d_fis_z: CudaSlice<f64>,
    pub d_fis_e: CudaSlice<f64>,
    pub d_fis_w: CudaSlice<f64>,
    pub d_fis_count: CudaSlice<i32>,
    pub d_cnt_coll: CudaSlice<i32>,
    pub d_cnt_fis: CudaSlice<i32>,
    pub d_cnt_leak: CudaSlice<i32>,
    pub d_cnt_surf: CudaSlice<i32>,
    pub d_cnt_el: CudaSlice<i32>,
    pub d_cnt_inel: CudaSlice<i32>,
    pub d_cnt_cap: CudaSlice<i32>,
    pub d_e_fis_in: CudaSlice<f64>,
    pub d_e_el_in: CudaSlice<f64>,
    pub d_e_inel_in: CudaSlice<f64>,
    pub d_e_inel_out: CudaSlice<f64>,
    pub d_e_fis_in_sq: CudaSlice<f64>,
    pub d_e_el_in_sq: CudaSlice<f64>,
    pub d_e_inel_in_sq: CudaSlice<f64>,
    pub d_q_inel: CudaSlice<f64>,
    /// `TransportParams` packed buffer. Host repacks per batch
    /// (pointers depend on the current nuc/mat/sab/wmp uploads).
    pub d_params: CudaSlice<u64>,
    /// `vec![1_i32; n]` lifted out of the per-batch path.
    alive_host_ones: Vec<i32>,

    // ── Event-based pipeline buffers ─────────────────────────────────
    /// SoA coord stack ─ [n × GR_MAX_DEPTH=4] per field. Replaces the
    /// per-thread `GrCoord stack[GR_MAX_DEPTH]` register/local-memory
    /// allocation in the persistent history kernel so the geom kernel
    /// can hand off state between launches.
    pub d_stack_universe: CudaSlice<i32>,
    pub d_stack_cell_idx: CudaSlice<i32>,
    pub d_stack_has_lattice: CudaSlice<i32>,
    pub d_stack_lattice_id: CudaSlice<i32>,
    pub d_stack_lat_ix: CudaSlice<i32>,
    pub d_stack_lat_iy: CudaSlice<i32>,
    pub d_stack_lat_iz: CudaSlice<i32>,
    pub d_stack_offx: CudaSlice<f64>,
    pub d_stack_offy: CudaSlice<f64>,
    pub d_stack_offz: CudaSlice<f64>,
    pub d_depth: CudaSlice<i32>,
    /// Per-event metadata emitted by `gr_trace_and_sample`, consumed
    /// by the four reaction kernels.
    pub d_event_type: CudaSlice<i32>,
    pub d_event_hit_nuc: CudaSlice<i32>,
    pub d_event_mat: CudaSlice<i32>,
    pub d_event_kT: CudaSlice<f64>,
    pub d_event_hit_Ni: CudaSlice<f64>,
    pub d_event_urr_xi: CudaSlice<f64>,
    /// Packed sub-slot (`hit_nuc_local * EB_N_EBINS + ebin`) emitted
    /// per event by `gr_trace_and_sample`, read by `gr_partition` to
    /// compute the 3-D (class, nuc_local, energy_bin) write slot.
    /// Extends PHYSOR 2022 Optimization G — particles in the same
    /// (class, nuc_local, ebin) end up adjacent in d_sorted_idx so
    /// reaction kernels see nuclide-AND-energy-clustered workloads
    /// (warps hit the same SVD basis + same energy region).
    pub d_event_ebin: CudaSlice<i32>,
    /// Partitioning. 3-D layout:
    /// `d_type_count[(class * MAX_NUC_PER_MAT + nuc_local) * 16 + bin]`
    /// over EB_N_PART_BINS = 5 × 128 × 16 = 10 240 slots (40 KB).
    /// `d_type_offsets[10241]` is the exclusive prefix sum. Per-class
    /// totals (for reaction-kernel launch sizing) and per-class
    /// starting offsets live in `d_type_class_total[5]` /
    /// `d_type_class_offsets[5]`. Scatter cursor mirrors the count
    /// layout.
    pub d_type_count: CudaSlice<i32>,
    pub d_type_offsets: CudaSlice<i32>,
    pub d_type_class_total: CudaSlice<i32>,
    pub d_type_class_offsets: CudaSlice<i32>,
    pub d_type_scatter: CudaSlice<i32>,
    pub d_sorted_idx: CudaSlice<i32>,
    /// Single-int device-side total of all event-type counts. Written
    /// by `gr_scan_offsets`, used by the host every K=EB_SYNC_EVERY
    /// outer steps to detect "all particles dead" without forcing a
    /// PCIe round-trip every step.
    pub d_type_total: CudaSlice<i32>,
}

impl TransportBuffers {
    /// `params_len` is queried from `GpuTransportContext` so this
    /// struct doesn't have to track `N_PARAMS`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: &Arc<CudaStream>,
        n: usize,
        fis_cap: usize,
        n_materials: usize,
        n_lattices: usize,
        params_len: usize,
    ) -> Result<Self, String> {
        let mk_d = |len: usize| stream.alloc_zeros::<f64>(len).map_err(|e| e.to_string());
        let mk_i = |len: usize| stream.alloc_zeros::<i32>(len).map_err(|e| e.to_string());
        let mk_u = |len: usize| stream.alloc_zeros::<u64>(len).map_err(|e| e.to_string());
        let fis = fis_cap.max(1);
        Ok(Self {
            n,
            fis_cap: fis,
            d_xs: mk_d(n)?,
            d_ys: mk_d(n)?,
            d_zs: mk_d(n)?,
            d_dxs: mk_d(n)?,
            d_dys: mk_d(n)?,
            d_dzs: mk_d(n)?,
            d_e: mk_d(n)?,
            d_alive: mk_i(n)?,
            d_rng_state: mk_u(n)?,
            d_rng_inc: mk_u(n)?,
            d_mat_kt: mk_d(n_materials.max(1))?,
            d_lat_override_off: mk_i(n_lattices + 1)?,
            d_lat_override_count: mk_i(n_lattices + 1)?,
            d_override_lat_idx: mk_i(1)?,
            d_override_cell_idx: mk_i(1)?,
            d_override_mat: mk_i(1)?,
            d_fis_x: mk_d(fis)?,
            d_fis_y: mk_d(fis)?,
            d_fis_z: mk_d(fis)?,
            d_fis_e: mk_d(fis)?,
            d_fis_w: mk_d(fis)?,
            d_fis_count: mk_i(1)?,
            d_cnt_coll: mk_i(1)?,
            d_cnt_fis: mk_i(1)?,
            d_cnt_leak: mk_i(1)?,
            d_cnt_surf: mk_i(1)?,
            d_cnt_el: mk_i(1)?,
            d_cnt_inel: mk_i(1)?,
            d_cnt_cap: mk_i(1)?,
            d_e_fis_in: mk_d(1)?,
            d_e_el_in: mk_d(1)?,
            d_e_inel_in: mk_d(1)?,
            d_e_inel_out: mk_d(1)?,
            d_e_fis_in_sq: mk_d(1)?,
            d_e_el_in_sq: mk_d(1)?,
            d_e_inel_in_sq: mk_d(1)?,
            d_q_inel: mk_d(1)?,
            d_params: mk_u(params_len)?,
            alive_host_ones: vec![1_i32; n],
            // Event-based pipeline. GR_MAX_DEPTH = 4 (matches the CUDA
            // header). Memory ~208 bytes per particle for the stack
            // arrays — 10.4 MB at n=50k.
            d_stack_universe: mk_i(n * 4)?,
            d_stack_cell_idx: mk_i(n * 4)?,
            d_stack_has_lattice: mk_i(n * 4)?,
            d_stack_lattice_id: mk_i(n * 4)?,
            d_stack_lat_ix: mk_i(n * 4)?,
            d_stack_lat_iy: mk_i(n * 4)?,
            d_stack_lat_iz: mk_i(n * 4)?,
            d_stack_offx: mk_d(n * 4)?,
            d_stack_offy: mk_d(n * 4)?,
            d_stack_offz: mk_d(n * 4)?,
            d_depth: mk_i(n)?,
            d_event_type: mk_i(n)?,
            d_event_hit_nuc: mk_i(n)?,
            d_event_mat: mk_i(n)?,
            d_event_kT: mk_d(n)?,
            d_event_hit_Ni: mk_d(n)?,
            d_event_urr_xi: mk_d(n)?,
            // 3-D partition: 5 classes × MAX_NUCLIDES_PER_MATERIAL
            // × 16 energy bins = 10 240 slots (40 KB). Sub-slot
            // (nuc_local * 16 + ebin) is packed into d_event_ebin so
            // the kernel signatures don't change.
            d_event_ebin: mk_i(n)?,
            d_type_count: mk_i(
                EV_TYPE_COUNT * crate::MAX_NUCLIDES_PER_MATERIAL * EB_N_EBINS,
            )?,
            d_type_offsets: mk_i(
                EV_TYPE_COUNT * crate::MAX_NUCLIDES_PER_MATERIAL * EB_N_EBINS + 1,
            )?,
            d_type_class_total: mk_i(EV_TYPE_COUNT)?,
            d_type_class_offsets: mk_i(EV_TYPE_COUNT)?,
            d_type_scatter: mk_i(
                EV_TYPE_COUNT * crate::MAX_NUCLIDES_PER_MATERIAL * EB_N_EBINS,
            )?,
            d_sorted_idx: mk_i(n)?,
            d_type_total: mk_i(1)?,
        })
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn fis_capacity(&self) -> usize {
        self.fis_cap
    }
}

/// Optional refill pool for PHYSOR 2022 Optimization F (continuous
/// particle refill). When passed to `transport_recursive_with_buffers`,
/// the outer event loop will keep dead slots topped up from this bank
/// between geometry steps so the kernel grid stays full through the
/// batch tail. Disabled by default — the standard transport call uses
/// `None` and is identical to the historical path.
///
/// Sizing: `capacity` is the upper bound on the total number of
/// histories this refill bank can spawn into the active slots. When
/// the device-side atomic cursor reaches `capacity` the kernel stops
/// refilling and the batch winds down naturally as the remaining
/// active particles die.
///
/// Statistical model: no per-particle weights. Each refilled particle
/// is a full unit-weight history. The batch's `total_histories` count
/// becomes `initial_in_flight + refilled_count` (both readable by the
/// host after the loop completes). k_eff = nu_bar * fissions /
/// total_histories — same expression as today, just with the
/// denominator inflated by the refill count.
pub struct RefillBuffers {
    pub d_refill_pos_x: CudaSlice<f64>,
    pub d_refill_pos_y: CudaSlice<f64>,
    pub d_refill_pos_z: CudaSlice<f64>,
    pub d_refill_energy: CudaSlice<f64>,
    pub d_refill_rng_state: CudaSlice<u64>,
    pub d_refill_rng_inc: CudaSlice<u64>,
    /// Atomic cursor — incremented per refilled slot. After the batch,
    /// host reads `min(value, capacity)` to know how many additional
    /// histories the bank fed.
    pub d_next_refill_idx: CudaSlice<i32>,
    /// Diagnostic counter — atomic ↑ on each successful refill. Equal
    /// to `min(d_next_refill_idx, capacity)` minus refills whose
    /// initial cell-find returned outside-geometry (leakage on spawn).
    /// Tracked separately because cnt_leak already absorbs that case.
    pub d_refilled_count: CudaSlice<i32>,
    /// Upper bound on slots the bank can fill — host-side mirror of
    /// the size of the refill arrays. Passed to the kernel as the
    /// `refill_bank_size` arg.
    pub capacity: usize,
}

impl RefillBuffers {
    pub fn new(stream: &Arc<CudaStream>, capacity: usize) -> Result<Self, String> {
        let mk_d = |len: usize| stream.alloc_zeros::<f64>(len).map_err(|e| e.to_string());
        let mk_u = |len: usize| stream.alloc_zeros::<u64>(len).map_err(|e| e.to_string());
        let mk_i = |len: usize| stream.alloc_zeros::<i32>(len).map_err(|e| e.to_string());
        let cap = capacity.max(1);
        Ok(Self {
            d_refill_pos_x: mk_d(cap)?,
            d_refill_pos_y: mk_d(cap)?,
            d_refill_pos_z: mk_d(cap)?,
            d_refill_energy: mk_d(cap)?,
            d_refill_rng_state: mk_u(cap)?,
            d_refill_rng_inc: mk_u(cap)?,
            d_next_refill_idx: mk_i(1)?,
            d_refilled_count: mk_i(1)?,
            capacity: cap,
        })
    }

    /// Reset the device cursors. Called by the host before each batch
    /// that opts into refill. Does NOT re-upload the source bank —
    /// the caller is responsible for filling the refill arrays before
    /// calling `transport_recursive_with_buffers` with this struct.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        stream
            .memset_zeros(&mut self.d_next_refill_idx)
            .map_err(|e| e.to_string())?;
        stream
            .memset_zeros(&mut self.d_refilled_count)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Read the device-side refilled counter back to the host. Caller
    /// uses this to bump `total_histories` for the batch.
    pub fn read_refilled_count(&self, stream: &Arc<CudaStream>) -> Result<usize, String> {
        let h = stream
            .clone_dtoh(&self.d_refilled_count)
            .map_err(|e| e.to_string())?;
        Ok(h[0].max(0) as usize)
    }
}

impl GpuRecursiveContext {
    /// Run one batch of full-physics neutron transport on the recursive
    /// geometry. Cross-section data is supplied via the
    /// `GpuTransportContext` upload-* paths (SVD / Pointwise / WMP / URR
    /// / discrete levels / S(α,β)). Each particle is transported to
    /// absorption / leakage / max-events.
    ///
    /// Convenience wrapper that builds a one-shot `TransportBuffers`
    /// and delegates to [`transport_recursive_with_buffers`]. Pays the
    /// driver-allocation cost on every call — fine for one-off
    /// diagnostics; production callers (e.g. `CudaRunner`) should
    /// pool their own buffers and call the pooled entry point
    /// directly.
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
            return Ok(empty_recursive_batch());
        }
        if rng_seeds.len() != n {
            return Err("source_bank / rng_seeds length mismatch".into());
        }
        // Throw-away buffers — backwards-compatible path. Pool callers
        // use `transport_recursive_with_buffers` instead.
        let params_len = gpu_t
            .build_transport_params_vec(nuc_data, mat_data, sab_data, wmp_data, 0)
            .len();
        let mut buffers = TransportBuffers::new(
            &self.stream,
            n,
            fis_capacity,
            mat_kT.len(),
            self.lat_origin.len() / 3,
            params_len,
        )?;
        self.transport_recursive_with_buffers(
            &mut buffers,
            gpu_t,
            nuc_data,
            mat_data,
            sab_data,
            wmp_data,
            source_bank,
            rng_seeds,
            mat_kT,
            sab_nuc_idx,
            max_events_per_history,
            fis_capacity,
            None,
        )
    }

    /// Pooled-buffer entry point. `buffers` must be sized for the same
    /// `n = source_bank.len()` and `fis_capacity` as previous calls;
    /// build a new pool if either changes mid-case.
    ///
    /// `refill` enables PHYSOR 2022 Optimization F (continuous particle
    /// refill). Pass `None` for the historical behaviour. When `Some`,
    /// the caller must have already uploaded the refill source bank
    /// into the `RefillBuffers` arrays and called `reset()` to zero the
    /// device counters. The returned `RecursiveTransportBatch.cnt_coll`
    /// and friends are summed across both initial-population and
    /// refilled histories. Read the additional history count via
    /// `RefillBuffers::read_refilled_count()` afterwards if you need to
    /// bump `total_histories` for k_eff bookkeeping.
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn transport_recursive_with_buffers(
        &self,
        buffers: &mut TransportBuffers,
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
        refill: Option<&mut RefillBuffers>,
    ) -> Result<RecursiveTransportBatch, String> {
        let n = source_bank.len();
        if n == 0 {
            return Ok(empty_recursive_batch());
        }
        if rng_seeds.len() != n {
            return Err("source_bank / rng_seeds length mismatch".into());
        }
        if buffers.n != n {
            return Err(format!(
                "TransportBuffers sized for n={} but batch has n={}",
                buffers.n, n
            ));
        }
        if buffers.fis_cap < fis_capacity {
            return Err(format!(
                "TransportBuffers fis_cap={} < requested {}",
                buffers.fis_cap, fis_capacity
            ));
        }

        let xs: Vec<f64> = source_bank.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = source_bank.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = source_bank.iter().map(|p| p.2).collect();
        let es: Vec<f64> = source_bank.iter().map(|p| p.3).collect();
        let mut dxs = Vec::with_capacity(n);
        let mut dys = Vec::with_capacity(n);
        let mut dzs = Vec::with_capacity(n);
        for &(s, inc) in rng_seeds {
            let mut rng = rust_mc_sim::Pcg64::from_state(s ^ 0xA5A5_A5A5_A5A5_A5A5, inc | 1);
            let mu = 2.0 * rng.uniform() - 1.0;
            let phi = 2.0 * std::f64::consts::PI * rng.uniform();
            let s_th = (1.0 - mu * mu).max(0.0).sqrt();
            dxs.push(s_th * phi.cos());
            dys.push(s_th * phi.sin());
            dzs.push(mu);
        }
        let rng_state: Vec<u64> = rng_seeds.iter().map(|s| s.0).collect();
        let rng_inc: Vec<u64> = rng_seeds.iter().map(|s| s.1).collect();

        let stream = &self.stream;
        let htod_d = |src: &[f64], dst: &mut CudaSlice<f64>| -> Result<(), String> {
            stream.memcpy_htod(src, dst).map_err(|e| e.to_string())
        };
        let htod_i = |src: &[i32], dst: &mut CudaSlice<i32>| -> Result<(), String> {
            stream.memcpy_htod(src, dst).map_err(|e| e.to_string())
        };
        let htod_u = |src: &[u64], dst: &mut CudaSlice<u64>| -> Result<(), String> {
            stream.memcpy_htod(src, dst).map_err(|e| e.to_string())
        };
        htod_d(&xs, &mut buffers.d_xs)?;
        htod_d(&ys, &mut buffers.d_ys)?;
        htod_d(&zs, &mut buffers.d_zs)?;
        htod_d(&dxs, &mut buffers.d_dxs)?;
        htod_d(&dys, &mut buffers.d_dys)?;
        htod_d(&dzs, &mut buffers.d_dzs)?;
        htod_d(&es, &mut buffers.d_e)?;
        htod_i(&buffers.alive_host_ones.clone(), &mut buffers.d_alive)?;
        htod_u(&rng_state, &mut buffers.d_rng_state)?;
        htod_u(&rng_inc, &mut buffers.d_rng_inc)?;
        htod_d(mat_kT, &mut buffers.d_mat_kt)?;
        let n_materials = mat_kT.len() as i32;

        // Dummy lattice-override fillers (recursive demo path).
        let n_lat_owned = self.lat_origin.len() / 3;
        let dummy_off: Vec<i32> = vec![-1; n_lat_owned + 1];
        let dummy_count: Vec<i32> = vec![0; n_lat_owned + 1];
        htod_i(&dummy_off, &mut buffers.d_lat_override_off)?;
        htod_i(&dummy_count, &mut buffers.d_lat_override_count)?;

        let zero_i = |dst: &mut CudaSlice<i32>| -> Result<(), String> {
            stream.memset_zeros(dst).map_err(|e| e.to_string())
        };
        let zero_f = |dst: &mut CudaSlice<f64>| -> Result<(), String> {
            stream.memset_zeros(dst).map_err(|e| e.to_string())
        };
        zero_i(&mut buffers.d_fis_count)?;
        zero_i(&mut buffers.d_cnt_coll)?;
        zero_i(&mut buffers.d_cnt_fis)?;
        zero_i(&mut buffers.d_cnt_leak)?;
        zero_i(&mut buffers.d_cnt_surf)?;
        zero_i(&mut buffers.d_cnt_el)?;
        zero_i(&mut buffers.d_cnt_inel)?;
        zero_i(&mut buffers.d_cnt_cap)?;
        zero_f(&mut buffers.d_e_fis_in)?;
        zero_f(&mut buffers.d_e_el_in)?;
        zero_f(&mut buffers.d_e_inel_in)?;
        zero_f(&mut buffers.d_e_inel_out)?;
        zero_f(&mut buffers.d_e_fis_in_sq)?;
        zero_f(&mut buffers.d_e_el_in_sq)?;
        zero_f(&mut buffers.d_e_inel_in_sq)?;
        zero_f(&mut buffers.d_q_inel)?;

        // Reborrow `refill` so the event loop can keep using it across
        // iterations without consuming the original Option. The `mut`
        // is needed to let `as_deref_mut()` reborrow per iteration.
        let mut refill = refill;
        if let Some(r) = refill.as_deref_mut() {
            r.reset(stream)?;
        }

        // Packed params hold device pointers that change every time
        // nuc/mat/sab/wmp uploads change; can't cache between batches.
        let params_vec =
            gpu_t.build_transport_params_vec(nuc_data, mat_data, sab_data, wmp_data, 0);
        stream
            .memcpy_htod(&params_vec, &mut buffers.d_params)
            .map_err(|e| e.to_string())?;

        let block = 128_u32;
        let cfg_n = |n_threads: u32| LaunchConfig {
            grid_dim: (n_threads.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let n_surf = self.n_surfaces;
        let root = self.root_universe;
        let n_lat = n_lat_owned as i32;
        let max_fis_i = fis_capacity as i32;

        // Event-based pipeline (Tramm et al., PHYSOR 2022). Replaces the single
        // persistent history kernel — the ncu profile showed
        // active_threads_per_warp = 6.2/32 in the history kernel
        // because reaction-type dispatch diverged every warp. Sorting
        // by reaction type between geom steps gives each reaction
        // kernel a single code path. Affects every scene whose
        // history kernel saw warp divergence, not just PWR-17×17.
        //
        // Step 1: gr_init_stacks (once per batch). Locates each
        // particle's deepest cell and seeds the SoA stack arrays.
        {
            let mut launch = stream.launch_builder(&self.k_eb_init_stacks);
            launch
                .arg(&mut buffers.d_xs)
                .arg(&mut buffers.d_ys)
                .arg(&mut buffers.d_zs)
                .arg(&mut buffers.d_alive)
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
                .arg(&n_lat)
                .arg(&self.hex_center)
                .arg(&self.hex_pitch_xy)
                .arg(&self.hex_pitch_z)
                .arg(&self.hex_n_rings)
                .arg(&self.hex_n_axial)
                .arg(&self.hex_orientation)
                .arg(&self.hex_universes_off)
                .arg(&self.hex_universes)
                .arg(&self.n_hex_lattices)
                .arg(&self.evals_scratch)
                .arg(&mut buffers.d_stack_universe)
                .arg(&mut buffers.d_stack_cell_idx)
                .arg(&mut buffers.d_stack_has_lattice)
                .arg(&mut buffers.d_stack_lattice_id)
                .arg(&mut buffers.d_stack_lat_ix)
                .arg(&mut buffers.d_stack_lat_iy)
                .arg(&mut buffers.d_stack_lat_iz)
                .arg(&mut buffers.d_stack_offx)
                .arg(&mut buffers.d_stack_offy)
                .arg(&mut buffers.d_stack_offz)
                .arg(&mut buffers.d_depth)
                .arg(&mut buffers.d_cnt_leak);
            // SAFETY: kernel signature matches the argument list
            // (gr_init_stacks in transport_event_based.cu).
            unsafe { launch.launch(cfg_n(n as u32)).map_err(|e| e.to_string())?; }
        }

        // Event loop. One iteration = one geom step + one reaction
        // dispatch round.
        //
        // ── PCIe sync trimmed to 1 round-trip per step ───────────────
        // The original event-based design did two syncs per step (DtoH
        // the 5 per-type counts, host prefix-sum, HtoD the 6 offsets).
        // We move the prefix sum onto the device (gr_scan_offsets) so
        // the HtoD vanishes; what remains is one DtoH of the 5 counts
        // which the host still needs for variable-size reaction-kernel
        // launches. Empirically (RTX A1000 sm_86, Godiva @ 20k), going
        // to batch-size launches with K-step periodic sync was 5x
        // SLOWER than variable-size launches on this hardware: per-
        // kernel block-scheduling overhead for 78 mostly-empty blocks
        // dominates the saved PCIe round-trip. Variable launches keep
        // empty warps off the SMs entirely.
        for _step in 0..max_events_per_history {
            // d_type_count is zeroed in-place by gr_scan_offsets after
            // it reads each slot — same for d_type_scatter. Buffer is
            // also alloc_zeros'd so the very first step starts clean.
            // One fewer host-side launch per step compared to the
            // previous memset_zeros + scan_offsets sequence.

            // Step 2: gr_trace_and_sample.
            {
                let mut launch = stream.launch_builder(&self.k_eb_trace_and_sample);
                launch
                    .arg(&buffers.d_params)
                    .arg(&mut buffers.d_xs)
                    .arg(&mut buffers.d_ys)
                    .arg(&mut buffers.d_zs)
                    .arg(&mut buffers.d_dxs)
                    .arg(&mut buffers.d_dys)
                    .arg(&mut buffers.d_dzs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_alive)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
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
                    .arg(&n_lat)
                    .arg(&self.hex_center)
                    .arg(&self.hex_pitch_xy)
                    .arg(&self.hex_pitch_z)
                    .arg(&self.hex_n_rings)
                    .arg(&self.hex_n_axial)
                    .arg(&self.hex_orientation)
                    .arg(&self.hex_universes_off)
                    .arg(&self.hex_universes)
                    .arg(&self.n_hex_lattices)
                    .arg(&buffers.d_lat_override_off)
                    .arg(&buffers.d_lat_override_count)
                    .arg(&buffers.d_override_lat_idx)
                    .arg(&buffers.d_override_cell_idx)
                    .arg(&buffers.d_override_mat)
                    .arg(&buffers.d_mat_kt)
                    .arg(&n_materials)
                    .arg(&sab_nuc_idx)
                    .arg(&self.evals_scratch)
                    .arg(&mut buffers.d_stack_universe)
                    .arg(&mut buffers.d_stack_cell_idx)
                    .arg(&mut buffers.d_stack_has_lattice)
                    .arg(&mut buffers.d_stack_lattice_id)
                    .arg(&mut buffers.d_stack_lat_ix)
                    .arg(&mut buffers.d_stack_lat_iy)
                    .arg(&mut buffers.d_stack_lat_iz)
                    .arg(&mut buffers.d_stack_offx)
                    .arg(&mut buffers.d_stack_offy)
                    .arg(&mut buffers.d_stack_offz)
                    .arg(&mut buffers.d_depth)
                    .arg(&mut buffers.d_event_type)
                    .arg(&mut buffers.d_event_hit_nuc)
                    .arg(&mut buffers.d_event_mat)
                    .arg(&mut buffers.d_event_kT)
                    .arg(&mut buffers.d_event_hit_Ni)
                    .arg(&mut buffers.d_event_urr_xi)
                    .arg(&mut buffers.d_event_ebin)
                    .arg(&mut buffers.d_type_count)
                    .arg(&mut buffers.d_cnt_coll)
                    .arg(&mut buffers.d_cnt_leak)
                    .arg(&mut buffers.d_cnt_surf)
                    .arg(&mut buffers.d_cnt_cap)
                    .arg(&mut buffers.d_e_el_in)
                    .arg(&mut buffers.d_e_el_in_sq);
                // SAFETY: kernel signature matches the argument list
                // (gr_trace_and_sample in transport_event_based.cu).
                unsafe { launch.launch(cfg_n(n as u32)).map_err(|e| e.to_string())?; }
            }

            // Step 3: device-side prefix sum over the 80 (class,bin)
            // slots. Also emits per-class totals + per-class start
            // offsets so reaction kernels can launch from a single
            // contiguous range per class. Zeroes d_type_scatter for
            // partition.
            {
                let mut launch = stream.launch_builder(&self.k_eb_scan_offsets);
                launch
                    .arg(&buffers.d_type_count)
                    .arg(&mut buffers.d_type_offsets)
                    .arg(&mut buffers.d_type_class_total)
                    .arg(&mut buffers.d_type_class_offsets)
                    .arg(&mut buffers.d_type_total)
                    .arg(&mut buffers.d_type_scatter);
                let one_thread = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { launch.launch(one_thread).map_err(|e| e.to_string())?; }
            }

            // Step 4: DtoH the 5 per-class totals (20 bytes) for
            // reaction-kernel launch sizing. The 80-slot per-(class,
            // bin) counts stay on device.
            let class_totals: Vec<i32> = stream
                .clone_dtoh(&buffers.d_type_class_total)
                .map_err(|e| e.to_string())?;
            let total: i32 = class_totals.iter().sum();
            if total == 0 { break; }

            // Step 5: gr_partition — scatter alive indices into
            // d_sorted_idx by (class, energy_bin), reading
            // d_event_ebin emitted by trace_and_sample.
            {
                let mut launch = stream.launch_builder(&self.k_eb_partition);
                launch
                    .arg(&buffers.d_event_type)
                    .arg(&buffers.d_event_ebin)
                    .arg(&buffers.d_alive)
                    .arg(&n_i32)
                    .arg(&buffers.d_type_offsets)
                    .arg(&mut buffers.d_type_scatter)
                    .arg(&mut buffers.d_sorted_idx);
                unsafe { launch.launch(cfg_n(n as u32)).map_err(|e| e.to_string())?; }
            }

            let c_el  = class_totals[0];
            let c_in  = class_totals[1];
            let c_fis = class_totals[2];
            let c_n2n = class_totals[3];
            let c_n3n = class_totals[4];
            let c_multi = c_n2n + c_n3n;

            // Reaction kernels read d_type_class_total[CLASS] and
            // d_type_class_offsets[CLASS] from device. Particles
            // within the class are already bin-sorted by energy in
            // d_sorted_idx (PHYSOR 2022 Optimization G), so adjacent
            // threads access nearby XS-table regions.
            if c_el > 0 {
                let mut launch = stream.launch_builder(&self.k_eb_elastic);
                launch
                    .arg(&buffers.d_params)
                    .arg(&buffers.d_type_class_total)
                    .arg(&buffers.d_type_class_offsets)
                    .arg(&buffers.d_sorted_idx)
                    .arg(&mut buffers.d_dxs)
                    .arg(&mut buffers.d_dys)
                    .arg(&mut buffers.d_dzs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
                    .arg(&buffers.d_event_hit_nuc)
                    .arg(&buffers.d_event_kT)
                    .arg(&buffers.d_event_urr_xi)
                    .arg(&sab_nuc_idx)
                    .arg(&mut buffers.d_cnt_el);
                unsafe { launch.launch(cfg_n(c_el as u32)).map_err(|e| e.to_string())?; }
            }
            if c_in > 0 {
                let mut launch = stream.launch_builder(&self.k_eb_inelastic);
                launch
                    .arg(&buffers.d_params)
                    .arg(&buffers.d_type_class_total)
                    .arg(&buffers.d_type_class_offsets)
                    .arg(&buffers.d_sorted_idx)
                    .arg(&mut buffers.d_dxs)
                    .arg(&mut buffers.d_dys)
                    .arg(&mut buffers.d_dzs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
                    .arg(&buffers.d_event_hit_nuc)
                    .arg(&mut buffers.d_cnt_inel)
                    .arg(&mut buffers.d_e_inel_in)
                    .arg(&mut buffers.d_e_inel_in_sq)
                    .arg(&mut buffers.d_e_inel_out)
                    .arg(&mut buffers.d_q_inel);
                unsafe { launch.launch(cfg_n(c_in as u32)).map_err(|e| e.to_string())?; }
            }
            if c_fis > 0 {
                let mut launch = stream.launch_builder(&self.k_eb_fission);
                launch
                    .arg(&buffers.d_params)
                    .arg(&buffers.d_type_class_total)
                    .arg(&buffers.d_type_class_offsets)
                    .arg(&buffers.d_sorted_idx)
                    .arg(&buffers.d_xs)
                    .arg(&buffers.d_ys)
                    .arg(&buffers.d_zs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_alive)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
                    .arg(&buffers.d_event_hit_nuc)
                    .arg(&mut buffers.d_fis_x)
                    .arg(&mut buffers.d_fis_y)
                    .arg(&mut buffers.d_fis_z)
                    .arg(&mut buffers.d_fis_e)
                    .arg(&mut buffers.d_fis_w)
                    .arg(&mut buffers.d_fis_count)
                    .arg(&max_fis_i)
                    .arg(&mut buffers.d_cnt_fis)
                    .arg(&mut buffers.d_e_fis_in)
                    .arg(&mut buffers.d_e_fis_in_sq);
                unsafe { launch.launch(cfg_n(c_fis as u32)).map_err(|e| e.to_string())?; }
            }
            if c_multi > 0 {
                let mut launch = stream.launch_builder(&self.k_eb_multi);
                launch
                    .arg(&buffers.d_params)
                    .arg(&buffers.d_type_class_total)
                    .arg(&buffers.d_type_class_offsets)
                    .arg(&buffers.d_sorted_idx)
                    .arg(&buffers.d_xs)
                    .arg(&buffers.d_ys)
                    .arg(&buffers.d_zs)
                    .arg(&mut buffers.d_dxs)
                    .arg(&mut buffers.d_dys)
                    .arg(&mut buffers.d_dzs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
                    .arg(&buffers.d_event_type)
                    .arg(&buffers.d_event_hit_nuc)
                    .arg(&mut buffers.d_fis_x)
                    .arg(&mut buffers.d_fis_y)
                    .arg(&mut buffers.d_fis_z)
                    .arg(&mut buffers.d_fis_e)
                    .arg(&mut buffers.d_fis_w)
                    .arg(&mut buffers.d_fis_count)
                    .arg(&max_fis_i);
                unsafe { launch.launch(cfg_n(c_multi as u32)).map_err(|e| e.to_string())?; }
            }

            // PHYSOR 2022 Optimization F (opt-in via the `refill` arg).
            // Must launch AFTER all event kernels for this step — the
            // event kernels (fission/multi) read d_event_type[tid] for
            // particles that the partition picked up earlier. Refilling
            // a slot mid-step would let gr_multi_event overwrite the
            // fresh particle's energy/dir with stale n2n/n3n kinematics
            // and inject a -7000 pcm bias on Godiva (measured here on
            // the A1000 before the launch was relocated). Refill self-
            // gates on alive[tid] == 0 per particle.
            if let Some(refill) = refill.as_deref_mut() {
                let refill_cap_i = refill.capacity as i32;
                let mut launch = stream.launch_builder(&self.k_eb_refill_dead);
                launch
                    .arg(&n_i32)
                    .arg(&mut refill.d_next_refill_idx)
                    .arg(&refill_cap_i)
                    .arg(&refill.d_refill_pos_x)
                    .arg(&refill.d_refill_pos_y)
                    .arg(&refill.d_refill_pos_z)
                    .arg(&refill.d_refill_energy)
                    .arg(&refill.d_refill_rng_state)
                    .arg(&refill.d_refill_rng_inc)
                    .arg(&mut buffers.d_xs)
                    .arg(&mut buffers.d_ys)
                    .arg(&mut buffers.d_zs)
                    .arg(&mut buffers.d_dxs)
                    .arg(&mut buffers.d_dys)
                    .arg(&mut buffers.d_dzs)
                    .arg(&mut buffers.d_e)
                    .arg(&mut buffers.d_alive)
                    .arg(&mut buffers.d_rng_state)
                    .arg(&mut buffers.d_rng_inc)
                    .arg(&mut buffers.d_depth)
                    .arg(&mut buffers.d_event_type)
                    .arg(&mut buffers.d_event_ebin)
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
                    .arg(&self.hex_center)
                    .arg(&self.hex_pitch_xy)
                    .arg(&self.hex_pitch_z)
                    .arg(&self.hex_n_rings)
                    .arg(&self.hex_n_axial)
                    .arg(&self.hex_orientation)
                    .arg(&self.hex_universes_off)
                    .arg(&self.hex_universes)
                    .arg(&self.n_hex_lattices)
                    .arg(&self.evals_scratch)
                    .arg(&mut buffers.d_stack_universe)
                    .arg(&mut buffers.d_stack_cell_idx)
                    .arg(&mut buffers.d_stack_has_lattice)
                    .arg(&mut buffers.d_stack_lattice_id)
                    .arg(&mut buffers.d_stack_lat_ix)
                    .arg(&mut buffers.d_stack_lat_iy)
                    .arg(&mut buffers.d_stack_lat_iz)
                    .arg(&mut buffers.d_stack_offx)
                    .arg(&mut buffers.d_stack_offy)
                    .arg(&mut buffers.d_stack_offz)
                    .arg(&mut buffers.d_cnt_leak)
                    .arg(&mut refill.d_refilled_count);
                unsafe {
                    launch.launch(cfg_n(n as u32)).map_err(|e| e.to_string())?;
                }
            }
        }

        let fis_count =
            stream.clone_dtoh(&buffers.d_fis_count).map_err(|e| e.to_string())?[0].max(0) as usize;
        let n_banked = fis_count.min(fis_capacity);
        let fx = stream.clone_dtoh(&buffers.d_fis_x).map_err(|e| e.to_string())?;
        let fy = stream.clone_dtoh(&buffers.d_fis_y).map_err(|e| e.to_string())?;
        let fz = stream.clone_dtoh(&buffers.d_fis_z).map_err(|e| e.to_string())?;
        let fe = stream.clone_dtoh(&buffers.d_fis_e).map_err(|e| e.to_string())?;
        let fission_bank: Vec<(f64, f64, f64, f64)> = (0..n_banked)
            .map(|i| (fx[i], fy[i], fz[i], fe[i]))
            .collect();

        let cnt_coll = stream.clone_dtoh(&buffers.d_cnt_coll).map_err(|e| e.to_string())?[0] as u64;
        let cnt_fis = stream.clone_dtoh(&buffers.d_cnt_fis).map_err(|e| e.to_string())?[0] as u64;
        let cnt_leak = stream.clone_dtoh(&buffers.d_cnt_leak).map_err(|e| e.to_string())?[0] as u64;
        let cnt_surf = stream.clone_dtoh(&buffers.d_cnt_surf).map_err(|e| e.to_string())?[0] as u64;
        let cnt_el = stream.clone_dtoh(&buffers.d_cnt_el).map_err(|e| e.to_string())?[0] as u64;
        let cnt_inel = stream.clone_dtoh(&buffers.d_cnt_inel).map_err(|e| e.to_string())?[0] as u64;
        let cnt_cap = stream.clone_dtoh(&buffers.d_cnt_cap).map_err(|e| e.to_string())?[0] as u64;
        let e_fis_in = stream.clone_dtoh(&buffers.d_e_fis_in).map_err(|e| e.to_string())?[0];
        let e_el_in = stream.clone_dtoh(&buffers.d_e_el_in).map_err(|e| e.to_string())?[0];
        let e_inel_in = stream.clone_dtoh(&buffers.d_e_inel_in).map_err(|e| e.to_string())?[0];
        let e_inel_out = stream.clone_dtoh(&buffers.d_e_inel_out).map_err(|e| e.to_string())?[0];
        let e_fis_in_sq = stream.clone_dtoh(&buffers.d_e_fis_in_sq).map_err(|e| e.to_string())?[0];
        let e_el_in_sq = stream.clone_dtoh(&buffers.d_e_el_in_sq).map_err(|e| e.to_string())?[0];
        let e_inel_in_sq = stream.clone_dtoh(&buffers.d_e_inel_in_sq).map_err(|e| e.to_string())?[0];
        let q_inel = stream.clone_dtoh(&buffers.d_q_inel).map_err(|e| e.to_string())?[0];

        // k_eff = fission_bank.len() / total_histories. When refill is
        // active, total_histories = n + refilled_count (host reads the
        // device-side counter after the event loop). When refill is
        // off, refilled = 0 and the denominator falls back to n.
        let refilled = if let Some(r) = refill.as_deref() {
            r.read_refilled_count(stream)?
        } else {
            0
        };
        let total_histories = n + refilled;
        let k_eff = fission_bank.len() as f64 / total_histories as f64;
        Ok(RecursiveTransportBatch {
            fission_bank,
            n_collisions: cnt_coll,
            n_fissions: cnt_fis,
            n_leakage: cnt_leak,
            n_surf_xings: cnt_surf,
            k_eff,
            n_elastic: cnt_el,
            n_inelastic: cnt_inel,
            n_capture: cnt_cap,
            e_fis_in_sum: e_fis_in,
            e_el_in_sum: e_el_in,
            e_inel_in_sum: e_inel_in,
            e_inel_out_sum: e_inel_out,
            e_fis_in_sq_sum: e_fis_in_sq,
            e_el_in_sq_sum: e_el_in_sq,
            e_inel_in_sq_sum: e_inel_in_sq,
            q_inel_sum: q_inel,
        })
    }
}

/// Zero-batch sentinel — kept in one place so both entry points return
/// the same shape on empty input.
fn empty_recursive_batch() -> RecursiveTransportBatch {
    RecursiveTransportBatch {
        fission_bank: Vec::new(),
        n_collisions: 0,
        n_fissions: 0,
        n_leakage: 0,
        n_surf_xings: 0,
        k_eff: 0.0,
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

impl GpuRecursiveContext {
    /// Number of rect lattices in this geometry. Sized by `lat_origin /
    /// 3` (each lattice contributes a 3-vector origin). Used by
    /// `TransportBuffers::new` to size the override scratch.
    pub fn n_lattices(&self) -> usize {
        self.lat_origin.len() / 3
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
