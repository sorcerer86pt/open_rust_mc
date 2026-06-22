// SPDX-License-Identifier: MIT
//! Backend-agnostic flattening of a recursive [`Geometry`] into
//! struct-of-arrays (SoA) tables.
//!
//! A `Geometry` is a tree of universes, cells, CSG region trees, and
//! (rect / hex) lattices connected by indices. GPU kernels can't chase
//! `Vec<Vec<…>>` pointers, so this module linearises the whole thing
//! into flat `Vec`s with explicit `off` / `len` index pairs — the
//! layout every device backend consumes.
//!
//! This used to live inside the `cuda`-gated `gpu_recursive` module.
//! It was lifted out (no behaviour change) so the CubeCL renderer and
//! the legacy CUDA path share **one** definition of the on-device
//! layout: the tag constants below are the single source of truth, and
//! both `gpu/cuda/geom_recursive.cu` (`GR_*`) and the CubeCL kernel
//! must agree with them.
//!
//! Pure: no GPU, no `unsafe`, no feature gate. `build_host_tables` is a
//! deterministic function of the `Geometry` alone.

use crate::geometry::cell::{CellFill, Region};
use crate::geometry::surface::{BoundaryCondition, Surface};
use crate::geometry::{Geometry, Vec3};

// ── Tag constants — must match `gpu/cuda/geom_recursive.cu` (GR_*) and
//    the CubeCL kernel. Single source of truth. ──────────────────────

pub const SURF_PLANE_X: i32 = 0;
pub const SURF_PLANE_Y: i32 = 1;
pub const SURF_PLANE_Z: i32 = 2;
pub const SURF_SPHERE: i32 = 3;
pub const SURF_CYL_Z: i32 = 4;
pub const SURF_CYL_X: i32 = 5;
pub const SURF_CYL_Y: i32 = 6;
pub const SURF_PLANE_GENERAL: i32 = 7;

pub const BC_TRANSMISSION: i32 = 0;
pub const BC_VACUUM: i32 = 1;
pub const BC_REFLECTIVE: i32 = 2;

pub const REGION_HALFSPACE_POS: i32 = 0;
pub const REGION_HALFSPACE_NEG: i32 = 1;
pub const REGION_INTERSECTION: i32 = 2;
pub const REGION_UNION: i32 = 3;
pub const REGION_COMPLEMENT: i32 = 4;

pub const FILL_MATERIAL: i32 = 0;
pub const FILL_VOID: i32 = 1;
pub const FILL_UNIVERSE: i32 = 2;
pub const FILL_LATTICE: i32 = 3;
pub const FILL_HEX_LATTICE: i32 = 4;

// Hex orientation discriminants — match CUDA `GR_HEX_ORIENT_*`.
pub const HEX_ORIENT_Y: i32 = 0;
pub const HEX_ORIENT_X: i32 = 1;

/// Doubles packed per surface in [`HostTables::surf_params`]. Fixed
/// stride so a device kernel indexes `surf_params[s_idx * 8 + k]`.
pub const SURF_PARAM_STRIDE: usize = 8;

// ── Host-side SoA tables ────────────────────────────────────────────

/// Flat SoA view of a `Geometry`, ready to upload to any GPU backend.
///
/// Index conventions (mirrored in every device walk):
/// - `surf_params` holds [`SURF_PARAM_STRIDE`] f64 per surface.
/// - `cell_aabb_min` / `cell_aabb_max` hold 3 f64 per cell.
/// - per-cell region opcodes live at `region_op[cell_region_off[c] ..
///   + cell_region_len[c]]` (postfix CSG stack machine).
/// - per-universe cell / surface index lists use the matching
///   `univ_*_off` / `univ_*_len` pairs into `univ_cell_indices` /
///   `univ_surface_indices`.
/// - lattice / hex universe arrays are flattened with `*_universes_off`.
#[derive(Default)]
pub struct HostTables {
    pub surf_type: Vec<i32>,
    pub surf_params: Vec<f64>, // SURF_PARAM_STRIDE doubles per surface
    pub surf_bc: Vec<i32>,

    pub cell_region_off: Vec<i32>,
    pub cell_region_len: Vec<i32>,
    pub cell_fill_type: Vec<i32>,
    pub cell_fill_data: Vec<i32>,
    pub cell_aabb_min: Vec<f64>, // 3 doubles per cell
    pub cell_aabb_max: Vec<f64>,

    pub region_op: Vec<i32>,
    pub region_arg: Vec<i32>,

    pub univ_cells_off: Vec<i32>,
    pub univ_cells_len: Vec<i32>,
    pub univ_surfaces_off: Vec<i32>,
    pub univ_surfaces_len: Vec<i32>,
    pub univ_cell_indices: Vec<i32>,
    pub univ_surface_indices: Vec<i32>,

    pub lat_origin: Vec<f64>,
    pub lat_pitch: Vec<f64>,
    pub lat_shape: Vec<i32>,
    pub lat_universes_off: Vec<i32>,
    pub lat_universes: Vec<i32>,
    // Hex lattice SoA — parallel to the rect arrays. The `n_*`
    // counters match the layout in `geom_recursive.cu::GrGeometry`.
    pub hex_center: Vec<f64>,
    pub hex_pitch_xy: Vec<f64>,
    pub hex_pitch_z: Vec<f64>,
    pub hex_n_rings: Vec<i32>,
    pub hex_n_axial: Vec<i32>,
    pub hex_orientation: Vec<i32>,
    pub hex_universes_off: Vec<i32>,
    pub hex_universes: Vec<i32>,
}

/// Pack one surface's parameters into `params_out` (stride-8) and its
/// boundary condition into `bc_out`; return the surface type tag.
///
/// Cones are not yet supported on the recursive-GPU path (the ICSBEP
/// scenes that exercise it use PlaneX/Y/Z + CylinderZ + Sphere); a cone
/// packs as a sentinel `-1` type so the device walk yields a huge eval
/// and the parity test flags it as a mismatch rather than silently
/// rendering garbage.
pub fn pack_surface(s: &Surface, params_out: &mut Vec<f64>, bc_out: &mut Vec<i32>) -> i32 {
    fn push8(v: &mut Vec<f64>, slots: [f64; SURF_PARAM_STRIDE]) {
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
        // Cones not yet supported on the recursive-GPU path — sentinel.
        _ => {
            push8(params_out, [0.0; SURF_PARAM_STRIDE]);
            bc_out.push(BC_VACUUM);
            -1
        }
    }
}

/// Walk a CSG region tree and emit postfix opcodes into `op` / `arg`.
pub fn flatten_region(region: &Region, op: &mut Vec<i32>, arg: &mut Vec<i32>) {
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

/// Clamp infinite AABB bounds to a large finite box so device slab
/// tests stay numerically sane (an unbounded outer cell has ±∞ extent).
pub fn finite_aabb(lo: Vec3, hi: Vec3) -> ([f64; 3], [f64; 3]) {
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

/// Flatten a whole `Geometry` into [`HostTables`]. Deterministic and
/// side-effect-free — the shared entry point for every GPU backend.
pub fn build_host_tables(geom: &Geometry) -> HostTables {
    use crate::geometry::lattice::HexOrientation;

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
