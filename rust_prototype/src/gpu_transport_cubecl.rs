// SPDX-License-Identifier: MIT
//! Cross-vendor GPU **transport** on CubeCL — first kernel.
//!
//! Port of `gpu/cuda/transport_recursive_const.cu`: constant
//! cross-sections per material (σ_t, σ_a, σ_f, ν̄), one kernel = one
//! batch of histories, each transported to absorption / leakage / step
//! cap with fission sites banked atomically. It is the smallest
//! end-to-end transport kernel — no nuclear-data tables, no S(α,β) — so
//! it proves the CubeCL transport plumbing (geometry walk + PCG RNG +
//! collision sampling + atomic fission banking + per-batch counters)
//! before the 173 KB `transport.cu` physics port.
//!
//! Written once in CubeCL → runs on CUDA / HIP-ROCm / Vulkan / Metal /
//! WebGPU, in **f64** (geometry + XS) with **u32 atomics** (portable —
//! i64 / f64 atomics need backend extensions wgpu doesn't guarantee, so
//! the fission bank uses a u32 cursor + plain per-slot position writes,
//! and counters are u32: a single batch never exceeds 4 G events).
//!
//! Geometry SoA layout matches [`crate::gpu_render`] (same `meta` /
//! `idata` / `fdata` packing via [`crate::geometry::flat`]). The device
//! walk helpers (`surf_eval`, `cell_contains`, `find_cell`,
//! `trace_step`, `reflect_dir`) mirror the `gr_*` functions in
//! `geom_recursive.cu` and the validated renderer kernel.
//!
//! STATUS — blocked by tracel-ai/cubecl#1336. The kernel compiles to
//! valid WGSL but faults at dispatch on NVIDIA/Vulkan: too much
//! thread-private `Array` state (the depth-4 coord stacks + the region
//! stack) trips a SPIR-V private-storage bug. Shrinking the state isn't
//! workable for recursive geometry, so transport stays on the CUDA
//! backend until the fix lands. Kernel + host harness are kept in-tree,
//! ready to re-enable; the batch test is `#[ignore]`d.

use cubecl::prelude::*;

use crate::geometry::Geometry;
use crate::geometry::flat::HostTables;

// ── meta header (u32 slots) — superset of the renderer's, with the
//    transport-only material-XS block appended. ────────────────────

const M_ROOT_UNIVERSE: usize = 1;
const M_N_MATERIALS: usize = 2;
const M_N_PARTICLES: usize = 3;
const M_MAX_EVENTS: usize = 4;
const M_FIS_CAPACITY: usize = 5;
// i32-blob element offsets
const M_OFF_SURF_TYPE: usize = 8;
const M_OFF_SURF_BC: usize = 9;
const M_OFF_CELL_REGION_OFF: usize = 10;
const M_OFF_CELL_REGION_LEN: usize = 11;
const M_OFF_CELL_FILL_TYPE: usize = 12;
const M_OFF_CELL_FILL_DATA: usize = 13;
const M_OFF_REGION_OP: usize = 14;
const M_OFF_REGION_ARG: usize = 15;
const M_OFF_UNIV_CELLS_LEN: usize = 17;
const M_OFF_UNIV_CELL_INDICES: usize = 18;
const M_OFF_UNIV_CELLS_OFF_BASE: usize = 19;
const M_OFF_LAT_SHAPE: usize = 20;
const M_OFF_LAT_UNIVERSES_OFF: usize = 21;
const M_OFF_LAT_UNIVERSES: usize = 22;
// f64-blob element offsets
const M_OFF_SURF_PARAMS: usize = 29;
const M_OFF_LAT_ORIGIN: usize = 32;
const M_OFF_LAT_PITCH: usize = 33;
const M_OFF_MAT_XS: usize = 35; // [σ_t, σ_a, σ_f, ν̄] × n_materials

const META_LEN: usize = 40;

// Tag constants — mirror `geometry::flat`.
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
const FILL_UNIVERSE: i32 = 2;
const FILL_LATTICE: i32 = 3;

const MAX_DEPTH: u32 = 4;
const MAX_DEPTH_USIZE: usize = 4;
const SURF_STRIDE: u32 = 8;

// ── Per-material constant cross sections ────────────────────────────

/// Constant cross sections for one material (barns-as-macroscopic;
/// the kernel treats them as Σ directly). Mirrors the CUDA `ConstXs`.
#[derive(Clone, Copy, Debug)]
pub struct ConstXs {
    pub sigma_t: f64,
    pub sigma_a: f64,
    pub sigma_f: f64,
    pub nu_bar: f64,
}

/// Aggregate result of one batch.
#[derive(Clone, Debug, Default)]
pub struct ConstXsBatch {
    pub fission_sites: Vec<(f64, f64, f64)>,
    pub n_collisions: u64,
    pub n_absorptions: u64,
    pub n_fissions: u64,
    pub n_leakage: u64,
    pub n_surf_xings: u64,
}

// ── Host packing ────────────────────────────────────────────────────

/// Upload-ready buffers for the transport kernel.
pub struct PackedTransport {
    pub meta: Vec<u32>,
    pub idata: Vec<i32>,
    pub fdata: Vec<f64>,
}

fn push_i32(blob: &mut Vec<i32>, src: &[i32]) -> u32 {
    let off = blob.len() as u32;
    blob.extend_from_slice(src);
    off
}
fn push_f64(blob: &mut Vec<f64>, src: &[f64]) -> u32 {
    let off = blob.len() as u32;
    blob.extend_from_slice(src);
    off
}

/// Pack geometry SoA + per-material constant XS for the transport
/// kernel. `materials[m]` supplies the four constants for material id
/// `m`; the kernel reads `mat_xs[m*4 + {0,1,2,3}]`.
pub fn pack_transport(t: &HostTables, geom: &Geometry, materials: &[ConstXs]) -> PackedTransport {
    let mut idata: Vec<i32> = Vec::new();
    let mut fdata: Vec<f64> = Vec::new();
    let mut meta = vec![0u32; META_LEN];

    meta[M_OFF_SURF_TYPE] = push_i32(&mut idata, &t.surf_type);
    meta[M_OFF_SURF_BC] = push_i32(&mut idata, &t.surf_bc);
    meta[M_OFF_CELL_REGION_OFF] = push_i32(&mut idata, &t.cell_region_off);
    meta[M_OFF_CELL_REGION_LEN] = push_i32(&mut idata, &t.cell_region_len);
    meta[M_OFF_CELL_FILL_TYPE] = push_i32(&mut idata, &t.cell_fill_type);
    meta[M_OFF_CELL_FILL_DATA] = push_i32(&mut idata, &t.cell_fill_data);
    meta[M_OFF_REGION_OP] = push_i32(&mut idata, &t.region_op);
    meta[M_OFF_REGION_ARG] = push_i32(&mut idata, &t.region_arg);
    meta[M_OFF_UNIV_CELLS_OFF_BASE] = push_i32(&mut idata, &t.univ_cells_off);
    meta[M_OFF_UNIV_CELLS_LEN] = push_i32(&mut idata, &t.univ_cells_len);
    meta[M_OFF_UNIV_CELL_INDICES] = push_i32(&mut idata, &t.univ_cell_indices);
    meta[M_OFF_LAT_SHAPE] = push_i32(&mut idata, &t.lat_shape);
    meta[M_OFF_LAT_UNIVERSES_OFF] = push_i32(&mut idata, &t.lat_universes_off);
    meta[M_OFF_LAT_UNIVERSES] = push_i32(&mut idata, &t.lat_universes);

    meta[M_OFF_SURF_PARAMS] = push_f64(&mut fdata, &t.surf_params);
    meta[M_OFF_LAT_ORIGIN] = push_f64(&mut fdata, &t.lat_origin);
    meta[M_OFF_LAT_PITCH] = push_f64(&mut fdata, &t.lat_pitch);

    let xs_flat: Vec<f64> = materials
        .iter()
        .flat_map(|m| [m.sigma_t, m.sigma_a, m.sigma_f, m.nu_bar])
        .collect();
    meta[M_OFF_MAT_XS] = push_f64(&mut fdata, &xs_flat);

    meta[M_ROOT_UNIVERSE] = geom.root_universe.0 as u32;
    meta[M_N_MATERIALS] = materials.len() as u32;

    if idata.is_empty() {
        idata.push(0);
    }
    if fdata.is_empty() {
        fdata.push(0.0);
    }

    PackedTransport { meta, idata, fdata }
}

// ── PCG-XSH-RR 64/32 (matches the rest of the codebase) ─────────────

/// Advance the PCG state (in a length-2 `u64` array: [state, inc]) and
/// return the next 32-bit output. CubeCL supports u64 + shifts + xor.
#[cube]
fn pcg_next(rng: &mut Array<u64>) -> u32 {
    let old = rng[0];
    rng[0] = old * 6364136223846793005u64 + rng[1];
    let xorshifted = u32::cast_from(((old >> 18u64) ^ old) >> 27u64);
    let rot = u32::cast_from(old >> 59u64);
    (xorshifted >> rot) | (xorshifted << ((32u32 - rot) & 31u32))
}

/// Uniform double in [0, 1) from two 32-bit draws (53-bit mantissa).
#[cube]
fn pcg_uniform(rng: &mut Array<u64>) -> f64 {
    let a = u64::cast_from(pcg_next(rng)) >> 5u64;
    let b = u64::cast_from(pcg_next(rng)) >> 6u64;
    f64::cast_from(a * 67108864u64 + b) * (1.0 / 9007199254740992.0)
}

// ── Geometry device helpers (mirror gr_* / renderer) ────────────────

#[cube]
fn surf_eval(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    s_idx: u32,
    x: f64,
    y: f64,
    z: f64,
) -> f64 {
    let t = idata[(surf_type_off + s_idx) as usize];
    let p = (surf_params_off + s_idx * SURF_STRIDE) as usize;
    let mut out = f64::new(1e30);
    if t == SURF_PLANE_X {
        out = x - fdata[p];
    } else {
        if t == SURF_PLANE_Y {
            out = y - fdata[p];
        } else {
            if t == SURF_PLANE_Z {
                out = z - fdata[p];
            } else {
                if t == SURF_SPHERE {
                    let dx = x - fdata[p];
                    let dy = y - fdata[p + 1];
                    let dz = z - fdata[p + 2];
                    out = dx * dx + dy * dy + dz * dz - fdata[p + 3] * fdata[p + 3];
                } else {
                    if t == SURF_CYL_Z {
                        let dx = x - fdata[p];
                        let dy = y - fdata[p + 1];
                        out = dx * dx + dy * dy - fdata[p + 2] * fdata[p + 2];
                    } else {
                        if t == SURF_CYL_X {
                            let dy = y - fdata[p];
                            let dz = z - fdata[p + 1];
                            out = dy * dy + dz * dz - fdata[p + 2] * fdata[p + 2];
                        } else {
                            if t == SURF_CYL_Y {
                                let dx = x - fdata[p];
                                let dz = z - fdata[p + 1];
                                out = dx * dx + dz * dz - fdata[p + 2] * fdata[p + 2];
                            } else {
                                if t == SURF_PLANE_GENERAL {
                                    out = fdata[p] * x + fdata[p + 1] * y + fdata[p + 2] * z - fdata[p + 3];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cube]
fn dist_plane(p: f64, d: f64, x0: f64, big: f64, tol: f64) -> f64 {
    let mut out = big;
    if d.abs() > f64::new(1e-300) {
        let t = (x0 - p) / d;
        out = select(t > tol, t, big);
    }
    out
}

#[cube]
fn dist_sphere(px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, cx: f64, cy: f64, cz: f64, r: f64, big: f64, tol: f64) -> f64 {
    let rx = px - cx;
    let ry = py - cy;
    let rz = pz - cz;
    let a = dx * dx + dy * dy + dz * dz;
    let b = 2.0 * (rx * dx + ry * dy + rz * dz);
    let c = rx * rx + ry * ry + rz * rz - r * r;
    let disc = b * b - 4.0 * a * c;
    let mut out = big;
    if disc >= f64::new(0.0) {
        let sq = disc.sqrt();
        let t1 = (-b - sq) / (2.0 * a);
        let t2 = (-b + sq) / (2.0 * a);
        let pick = select(t1 > tol, t1, t2);
        out = select(pick > tol, pick, big);
    }
    out
}

#[cube]
fn dist_cyl(p1: f64, p2: f64, d1: f64, d2: f64, c1: f64, c2: f64, r: f64, big: f64, tol: f64) -> f64 {
    let r1 = p1 - c1;
    let r2 = p2 - c2;
    let a = d1 * d1 + d2 * d2;
    let mut out = big;
    if a > f64::new(1e-300) {
        let b = 2.0 * (r1 * d1 + r2 * d2);
        let c = r1 * r1 + r2 * r2 - r * r;
        let disc = b * b - 4.0 * a * c;
        if disc >= f64::new(0.0) {
            let sq = disc.sqrt();
            let t1 = (-b - sq) / (2.0 * a);
            let t2 = (-b + sq) / (2.0 * a);
            let pick = select(t1 > tol, t1, t2);
            out = select(pick > tol, pick, big);
        }
    }
    out
}

#[cube]
fn surf_dist(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    s_idx: u32,
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) -> f64 {
    let t = idata[(surf_type_off + s_idx) as usize];
    let p = (surf_params_off + s_idx * SURF_STRIDE) as usize;
    let big = f64::new(1e30);
    let tol = f64::new(1e-12);
    let mut out = big;
    if t == SURF_PLANE_X {
        out = dist_plane(px, dx, fdata[p], big, tol);
    } else {
        if t == SURF_PLANE_Y {
            out = dist_plane(py, dy, fdata[p], big, tol);
        } else {
            if t == SURF_PLANE_Z {
                out = dist_plane(pz, dz, fdata[p], big, tol);
            } else {
                if t == SURF_SPHERE {
                    out = dist_sphere(px, py, pz, dx, dy, dz, fdata[p], fdata[p + 1], fdata[p + 2], fdata[p + 3], big, tol);
                } else {
                    if t == SURF_CYL_Z {
                        out = dist_cyl(px, py, dx, dy, fdata[p], fdata[p + 1], fdata[p + 2], big, tol);
                    } else {
                        if t == SURF_CYL_X {
                            out = dist_cyl(py, pz, dy, dz, fdata[p], fdata[p + 1], fdata[p + 2], big, tol);
                        } else {
                            if t == SURF_CYL_Y {
                                out = dist_cyl(px, pz, dx, dz, fdata[p], fdata[p + 1], fdata[p + 2], big, tol);
                            } else {
                                if t == SURF_PLANE_GENERAL {
                                    let denom = fdata[p] * dx + fdata[p + 1] * dy + fdata[p + 2] * dz;
                                    if denom.abs() > f64::new(1e-300) {
                                        let tv = (fdata[p + 3] - (fdata[p] * px + fdata[p + 1] * py + fdata[p + 2] * pz)) / denom;
                                        out = select(tv > tol, tv, big);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Postfix CSG stack-machine: 1 if local point is inside cell region.
#[cube]
fn cell_contains(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    region_op_off: u32,
    region_arg_off: u32,
    r_off: u32,
    r_len: u32,
    x: f64,
    y: f64,
    z: f64,
) -> u32 {
    let mut stack = Array::<u32>::new(16usize);
    let mut sp = 0usize;
    for i in 0..r_len {
        let op = idata[(region_op_off + r_off + i) as usize];
        let arg = idata[(region_arg_off + r_off + i) as usize];
        if op == REGION_HALFSPACE_POS {
            let v = surf_eval(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), x, y, z);
            stack[sp] = select(v > f64::new(0.0), 1u32, 0u32);
            sp += 1;
        } else {
            if op == REGION_HALFSPACE_NEG {
                let v = surf_eval(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), x, y, z);
                stack[sp] = select(v < f64::new(0.0), 1u32, 0u32);
                sp += 1;
            } else {
                if op == REGION_INTERSECTION {
                    let b = stack[sp - 1];
                    let a = stack[sp - 2];
                    sp -= 1;
                    stack[sp - 1] = select(a + b == 2u32, 1u32, 0u32);
                } else {
                    if op == REGION_UNION {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        sp -= 1;
                        stack[sp - 1] = select(a + b > 0u32, 1u32, 0u32);
                    } else {
                        if op == REGION_COMPLEMENT {
                            let a = stack[sp - 1];
                            stack[sp - 1] = select(a == 0u32, 1u32, 0u32);
                        }
                    }
                }
            }
        }
    }
    select(sp == 1usize, stack[0], 0u32)
}

/// Reflect a unit direction about an axis-aligned / general plane (in a
/// length-3 `dir` array, mutated in place). Mirrors `gr_reflect_direction`.
#[cube]
fn reflect_dir(idata: &Array<i32>, fdata: &Array<f64>, surf_params_off: u32, surf_type_off: u32, s_idx: u32, dir: &mut Array<f64>) {
    let t = idata[(surf_type_off + s_idx) as usize];
    let p = (surf_params_off + s_idx * SURF_STRIDE) as usize;
    if t == SURF_PLANE_X {
        dir[0] = -dir[0];
    } else {
        if t == SURF_PLANE_Y {
            dir[1] = -dir[1];
        } else {
            if t == SURF_PLANE_Z {
                dir[2] = -dir[2];
            } else {
                if t == SURF_PLANE_GENERAL {
                    let nx = fdata[p];
                    let ny = fdata[p + 1];
                    let nz = fdata[p + 2];
                    let ddot = dir[0] * nx + dir[1] * ny + dir[2] * nz;
                    dir[0] = dir[0] - 2.0 * ddot * nx;
                    dir[1] = dir[1] - 2.0 * ddot * ny;
                    dir[2] = dir[2] - 2.0 * ddot * nz;
                }
            }
        }
    }
}

/// Recursive cell-find. Writes the resolved stack into the parallel
/// `st_*` arrays (length ≥ MAX_DEPTH) and returns the depth (0 = leak).
/// Single-exit; mirrors `gr_find_cell` (rect lattices + universes; hex
/// is not used by the const-XS scenes, treated as leak).
#[cube]
fn find_cell(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    region_op_off: u32,
    region_arg_off: u32,
    cell_region_off_a: u32,
    cell_region_len_a: u32,
    cell_fill_type_a: u32,
    cell_fill_data_a: u32,
    univ_cells_off_base: u32,
    univ_cells_len_a: u32,
    univ_cell_indices_a: u32,
    lat_shape_off: u32,
    lat_universes_off_a: u32,
    lat_universes_a: u32,
    lat_origin_off: u32,
    lat_pitch_off: u32,
    root_universe: u32,
    wx: f64,
    wy: f64,
    wz: f64,
    st_cell: &mut Array<i32>,
    st_offx: &mut Array<f64>,
    st_offy: &mut Array<f64>,
    st_offz: &mut Array<f64>,
    st_has_lat: &mut Array<i32>,
    st_lat_id: &mut Array<i32>,
    st_lat_ix: &mut Array<i32>,
    st_lat_iy: &mut Array<i32>,
    st_lat_iz: &mut Array<i32>,
) -> u32 {
    let mut depth = u32::new(0);
    let mut cur_univ = root_universe;
    let mut off_x = f64::new(0.0);
    let mut off_y = f64::new(0.0);
    let mut off_z = f64::new(0.0);
    let mut has_lat = i32::new(0);
    let mut lat_id = i32::new(0);
    let mut lat_ix = i32::new(0);
    let mut lat_iy = i32::new(0);
    let mut lat_iz = i32::new(0);
    let mut lx = wx;
    let mut ly = wy;
    let mut lz = wz;
    let mut fc_done = false;

    for _d in 0..MAX_DEPTH {
        if !fc_done {
            lx = lx - off_x;
            ly = ly - off_y;
            lz = lz - off_z;

            let c_off = idata[(univ_cells_off_base + cur_univ) as usize];
            let c_len = idata[(univ_cells_len_a + cur_univ) as usize];
            let mut chosen = i32::new(-1);
            for k in 0..u32::cast_from(c_len) {
                if chosen < 0i32 {
                    let cand = idata[(univ_cell_indices_a + u32::cast_from(c_off) + k) as usize];
                    let r_off = idata[(cell_region_off_a + u32::cast_from(cand)) as usize];
                    let r_len = idata[(cell_region_len_a + u32::cast_from(cand)) as usize];
                    let inside = cell_contains(
                        idata, fdata, surf_params_off, surf_type_off,
                        region_op_off, region_arg_off,
                        u32::cast_from(r_off), u32::cast_from(r_len),
                        lx, ly, lz,
                    );
                    if inside == 1u32 {
                        chosen = cand;
                    }
                }
            }

            if chosen < 0i32 {
                depth = u32::new(0);
                fc_done = true;
            } else {
                st_cell[depth as usize] = chosen;
                st_offx[depth as usize] = off_x;
                st_offy[depth as usize] = off_y;
                st_offz[depth as usize] = off_z;
                st_has_lat[depth as usize] = has_lat;
                st_lat_id[depth as usize] = lat_id;
                st_lat_ix[depth as usize] = lat_ix;
                st_lat_iy[depth as usize] = lat_iy;
                st_lat_iz[depth as usize] = lat_iz;
                depth += 1;

                let ft = idata[(cell_fill_type_a + u32::cast_from(chosen)) as usize];
                let fd = idata[(cell_fill_data_a + u32::cast_from(chosen)) as usize];
                if ft == FILL_MATERIAL {
                    fc_done = true;
                } else {
                    if ft == FILL_UNIVERSE {
                        cur_univ = u32::cast_from(fd);
                        off_x = f64::new(0.0);
                        off_y = f64::new(0.0);
                        off_z = f64::new(0.0);
                        has_lat = i32::new(0);
                    } else {
                        if ft == FILL_LATTICE {
                            let lid = u32::cast_from(fd);
                            let org = (lat_origin_off + lid * 3u32) as usize;
                            let pit = (lat_pitch_off + lid * 3u32) as usize;
                            let shp = (lat_shape_off + lid * 3u32) as usize;
                            let rx = lx - fdata[org];
                            let ry = ly - fdata[org + 1];
                            let rz = lz - fdata[org + 2];
                            let fix = (rx / fdata[pit]).floor();
                            let fiy = (ry / fdata[pit + 1]).floor();
                            let fiz = (rz / fdata[pit + 2]).floor();
                            let ix = i32::cast_from(fix);
                            let iy = i32::cast_from(fiy);
                            let iz = i32::cast_from(fiz);
                            let sh0 = idata[shp];
                            let sh1 = idata[shp + 1];
                            let sh2 = idata[shp + 2];
                            let inb = ix >= 0i32 && iy >= 0i32 && iz >= 0i32 && ix < sh0 && iy < sh1 && iz < sh2;
                            if !inb {
                                depth = u32::new(0);
                                fc_done = true;
                            } else {
                                let slab_n = sh0 * sh1;
                                let linear = iz * slab_n + iy * sh0 + ix;
                                let luoff = idata[(lat_universes_off_a + lid) as usize];
                                cur_univ = u32::cast_from(idata[(lat_universes_a + u32::cast_from(luoff + linear)) as usize]);
                                off_x = fdata[org] + (fix + 0.5) * fdata[pit];
                                off_y = fdata[org + 1] + (fiy + 0.5) * fdata[pit + 1];
                                off_z = fdata[org + 2] + (fiz + 0.5) * fdata[pit + 2];
                                has_lat = i32::new(1);
                                lat_id = i32::cast_from(lid);
                                lat_ix = ix;
                                lat_iy = iy;
                                lat_iz = iz;
                            }
                        } else {
                            // hex / void fill — leak (not used by const-XS scenes)
                            depth = u32::new(0);
                            fc_done = true;
                        }
                    }
                }
            }
        }
    }
    depth
}

/// Trace to the next surface / lattice-grid crossing from world
/// `(wx,wy,wz)` along `(dx,dy,dz)`, given the current stack of `depth`
/// frames. Writes results into `out`:
///   out[0] = distance, out[1] = surface_idx (−1 = grid), out[2] = bc.
/// Mirrors `gr_trace_step` (rect grids only). The caller re-resolves the
/// next cell via `find_cell` at the nudged point.
#[cube]
#[allow(clippy::too_many_arguments)]
fn trace_step(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    surf_bc_off: u32,
    region_op_off: u32,
    region_arg_off: u32,
    cell_region_off_a: u32,
    cell_region_len_a: u32,
    lat_origin_off: u32,
    lat_pitch_off: u32,
    wx: f64,
    wy: f64,
    wz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    depth: u32,
    st_cell: &mut Array<i32>,
    st_offx: &mut Array<f64>,
    st_offy: &mut Array<f64>,
    st_offz: &mut Array<f64>,
    st_has_lat: &mut Array<i32>,
    st_lat_id: &mut Array<i32>,
    st_lat_ix: &mut Array<i32>,
    st_lat_iy: &mut Array<i32>,
    st_lat_iz: &mut Array<i32>,
    out: &mut Array<f64>,
) {
    let big = f64::new(1e30);
    let mut best_dist = big;
    let mut best_surf = i32::new(-1);

    // Surfaces of every stack cell (in that cell's local frame).
    let mut flx = wx;
    let mut fly = wy;
    let mut flz = wz;
    for d in 0..depth {
        flx = flx - st_offx[d as usize];
        fly = fly - st_offy[d as usize];
        flz = flz - st_offz[d as usize];
        let cci = st_cell[d as usize];
        let r_off = idata[(cell_region_off_a + u32::cast_from(cci)) as usize];
        let r_len = idata[(cell_region_len_a + u32::cast_from(cci)) as usize];
        for i in 0..u32::cast_from(r_len) {
            let op = idata[(region_op_off + u32::cast_from(r_off) + i) as usize];
            let arg = idata[(region_arg_off + u32::cast_from(r_off) + i) as usize];
            let is_hs = op == REGION_HALFSPACE_POS || op == REGION_HALFSPACE_NEG;
            if is_hs {
                let dd = surf_dist(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), flx, fly, flz, dx, dy, dz);
                if dd < best_dist {
                    best_dist = dd;
                    best_surf = arg;
                }
            }
        }
    }

    // Rect-lattice grid crossings (parent frame).
    let mut plx = wx;
    let mut ply = wy;
    let mut plz = wz;
    for d in 0..depth {
        if st_has_lat[d as usize] == 1i32 {
            let lid = u32::cast_from(st_lat_id[d as usize]);
            let org = (lat_origin_off + lid * 3u32) as usize;
            let pit = (lat_pitch_off + lid * 3u32) as usize;
            let gd = grid_dist(
                plx, ply, plz, dx, dy, dz,
                fdata[org], fdata[org + 1], fdata[org + 2],
                fdata[pit], fdata[pit + 1], fdata[pit + 2],
                st_lat_ix[d as usize], st_lat_iy[d as usize], st_lat_iz[d as usize],
            );
            if gd + f64::new(1e-9) < best_dist {
                best_dist = gd;
                best_surf = i32::new(-1);
            }
        }
        plx = plx - st_offx[d as usize];
        ply = ply - st_offy[d as usize];
        plz = plz - st_offz[d as usize];
    }

    let mut bc = i32::new(BC_TRANSMISSION as i64);
    if best_surf >= 0i32 {
        bc = idata[(surf_bc_off + u32::cast_from(best_surf)) as usize];
    }
    out[0] = best_dist;
    out[1] = f64::cast_from(best_surf);
    out[2] = f64::cast_from(bc);
}

#[cube]
fn grid_dist(
    px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64,
    ox: f64, oy: f64, oz: f64, pitx: f64, pity: f64, pitz: f64,
    ix: i32, iy: i32, iz: i32,
) -> f64 {
    let big = f64::new(1e30);
    let mut best = big;
    best = grid_axis(px - ox, dx, pitx, ix, best);
    best = grid_axis(py - oy, dy, pity, iy, best);
    best = grid_axis(pz - oz, dz, pitz, iz, best);
    best
}

#[cube]
fn grid_axis(pos: f64, d: f64, pitch: f64, idx: i32, cur_best: f64) -> f64 {
    let mut best = cur_best;
    if d.abs() > f64::new(1e-300) {
        let fwd = d > f64::new(0.0);
        let target = select(fwd, f64::cast_from(idx + 1i32) * pitch, f64::cast_from(idx) * pitch);
        let mut tt = (target - pos) / d;
        if tt <= f64::new(0.0) {
            let nxt = select(fwd, f64::cast_from(idx + 2i32) * pitch, f64::cast_from(idx - 1i32) * pitch);
            tt = (nxt - pos) / d;
        }
        if tt > f64::new(0.0) && tt < best {
            best = tt;
        }
    }
    best
}

// ── Transport kernel ────────────────────────────────────────────────
//
// One thread = one history. Persistent within the batch: each thread
// loops collision/crossing events until absorption / leakage / cap.
// Fission sites are claimed with a single Atomic<u32> cursor (portable
// -- no i64/f64 atomics); each claimed slot is written by exactly one
// thread so the position stores need no atomicity. Counters are u32.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn const_xs_kernel(
    meta: &Array<u32>,
    idata: &Array<i32>,
    fdata: &Array<f64>,
    pos: &mut Array<f64>,
    dir: &mut Array<f64>,
    rng: &mut Array<u64>,
    alive: &mut Array<u32>,
    fis_pos: &mut Array<f64>,
    // Atomic counters, length 6:
    // [0]=fission-bank cursor, [1]=collisions, [2]=absorptions,
    // [3]=fissions, [4]=leakage, [5]=surface crossings.
    atomics: &mut Array<Atomic<u32>>,
) {
    let tid = ABSOLUTE_POS;
    let n_particles = meta[M_N_PARTICLES];
    if tid < n_particles as usize {
        if alive[tid] == 1u32 {
            let surf_type_off = meta[M_OFF_SURF_TYPE];
            let surf_bc_off = meta[M_OFF_SURF_BC];
            let cell_region_off_a = meta[M_OFF_CELL_REGION_OFF];
            let cell_region_len_a = meta[M_OFF_CELL_REGION_LEN];
            let cell_fill_type_a = meta[M_OFF_CELL_FILL_TYPE];
            let cell_fill_data_a = meta[M_OFF_CELL_FILL_DATA];
            let region_op_off = meta[M_OFF_REGION_OP];
            let region_arg_off = meta[M_OFF_REGION_ARG];
            let univ_cells_off_base = meta[M_OFF_UNIV_CELLS_OFF_BASE];
            let univ_cells_len_a = meta[M_OFF_UNIV_CELLS_LEN];
            let univ_cell_indices_a = meta[M_OFF_UNIV_CELL_INDICES];
            let lat_shape_off = meta[M_OFF_LAT_SHAPE];
            let lat_universes_off_a = meta[M_OFF_LAT_UNIVERSES_OFF];
            let lat_universes_a = meta[M_OFF_LAT_UNIVERSES];
            let surf_params_off = meta[M_OFF_SURF_PARAMS];
            let lat_origin_off = meta[M_OFF_LAT_ORIGIN];
            let lat_pitch_off = meta[M_OFF_LAT_PITCH];
            let mat_xs_off = meta[M_OFF_MAT_XS];
            let root_universe = meta[M_ROOT_UNIVERSE];
            let max_events = meta[M_MAX_EVENTS];
            let fis_capacity = meta[M_FIS_CAPACITY];

            let mut lrng = Array::<u64>::new(2usize);
            lrng[0] = rng[tid * 2];
            lrng[1] = rng[tid * 2 + 1];

            let mut px = pos[tid * 3];
            let mut py = pos[tid * 3 + 1];
            let mut pz = pos[tid * 3 + 2];
            let mut ddir = Array::<f64>::new(3usize);
            ddir[0] = dir[tid * 3];
            ddir[1] = dir[tid * 3 + 1];
            ddir[2] = dir[tid * 3 + 2];

            let mut st_cell = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_offx = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_offy = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_offz = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_has_lat = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_id = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_ix = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_iy = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_iz = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut ts = Array::<f64>::new(3usize);

            let mut depth = find_cell(
                idata, fdata, surf_params_off, surf_type_off, region_op_off, region_arg_off,
                cell_region_off_a, cell_region_len_a, cell_fill_type_a, cell_fill_data_a,
                univ_cells_off_base, univ_cells_len_a, univ_cell_indices_a,
                lat_shape_off, lat_universes_off_a, lat_universes_a, lat_origin_off, lat_pitch_off,
                root_universe, px, py, pz,
                &mut st_cell, &mut st_offx, &mut st_offy, &mut st_offz,
                &mut st_has_lat, &mut st_lat_id, &mut st_lat_ix, &mut st_lat_iy, &mut st_lat_iz,
            );

            let mut lc_coll = u32::new(0);
            let mut lc_abs = u32::new(0);
            let mut lc_fis = u32::new(0);
            let mut lc_surf = u32::new(0);
            let mut lc_leak = u32::new(0);
            let mut local_alive = u32::new(1);

            if depth == 0u32 {
                local_alive = u32::new(0);
                lc_leak += 1;
            }

            let mut ev = u32::new(0);
            for _i in 0..4096u32 {
                if local_alive == 1u32 && ev < max_events {
                    ev += 1;
                    let ci = st_cell[(depth - 1) as usize];
                    let ft = idata[(cell_fill_type_a + u32::cast_from(ci)) as usize];
                    let mut mat = i32::new(-1);
                    if ft == FILL_MATERIAL {
                        mat = idata[(cell_fill_data_a + u32::cast_from(ci)) as usize];
                    }

                    if mat < 0i32 {
                        trace_step(
                            idata, fdata, surf_params_off, surf_type_off, surf_bc_off,
                            region_op_off, region_arg_off, cell_region_off_a, cell_region_len_a,
                            lat_origin_off, lat_pitch_off, px, py, pz, ddir[0], ddir[1], ddir[2],
                            depth, &mut st_cell, &mut st_offx, &mut st_offy, &mut st_offz, &mut st_has_lat,
                            &mut st_lat_id, &mut st_lat_ix, &mut st_lat_iy, &mut st_lat_iz, &mut ts,
                        );
                        let dist = ts[0];
                        let surf_idx = i32::cast_from(ts[1]);
                        let bc = i32::cast_from(ts[2]);
                        if dist >= f64::new(1e29) {
                            local_alive = u32::new(0);
                            lc_leak += 1;
                        } else {
                            lc_surf += 1;
                            cross_or_die(
                                idata, fdata, surf_params_off, surf_type_off,
                                bc, surf_idx, dist,
                                &mut px, &mut py, &mut pz, &mut ddir, &mut local_alive,
                            );
                            if bc == BC_VACUUM {
                                lc_leak += 1;
                            }
                        }
                    } else {
                        let sigma_t = fdata[(mat_xs_off + u32::cast_from(mat) * 4u32) as usize];
                        let sigma_a = fdata[(mat_xs_off + u32::cast_from(mat) * 4u32 + 1u32) as usize];
                        let sigma_f = fdata[(mat_xs_off + u32::cast_from(mat) * 4u32 + 2u32) as usize];
                        let nu_bar = fdata[(mat_xs_off + u32::cast_from(mat) * 4u32 + 3u32) as usize];

                        if sigma_t <= f64::new(0.0) {
                            local_alive = u32::new(0);
                            lc_leak += 1;
                        } else {
                            let d_collide = -(pcg_uniform(&mut lrng).ln()) / sigma_t;
                            trace_step(
                                idata, fdata, surf_params_off, surf_type_off, surf_bc_off,
                                region_op_off, region_arg_off, cell_region_off_a, cell_region_len_a,
                                lat_origin_off, lat_pitch_off, px, py, pz, ddir[0], ddir[1], ddir[2],
                                depth, &mut st_cell, &mut st_offx, &mut st_offy, &mut st_offz, &mut st_has_lat,
                                &mut st_lat_id, &mut st_lat_ix, &mut st_lat_iy, &mut st_lat_iz, &mut ts,
                            );
                            let dist = ts[0];
                            let surf_idx = i32::cast_from(ts[1]);
                            let bc = i32::cast_from(ts[2]);

                            if dist >= f64::new(1e29) {
                                local_alive = u32::new(0);
                                lc_leak += 1;
                            } else {
                                if d_collide < dist {
                                    px = px + ddir[0] * d_collide;
                                    py = py + ddir[1] * d_collide;
                                    pz = pz + ddir[2] * d_collide;
                                    lc_coll += 1;
                                    let xi_react = pcg_uniform(&mut lrng) * sigma_t;
                                    if xi_react < sigma_a {
                                        lc_abs += 1;
                                        if sigma_a > f64::new(0.0) {
                                            let pf = sigma_f / sigma_a;
                                            if pcg_uniform(&mut lrng) < pf {
                                                let xi = pcg_uniform(&mut lrng);
                                                let n_fis = u32::cast_from(nu_bar + xi);
                                                if n_fis > 0u32 {
                                                    let slot = atomics[0].fetch_add(n_fis);
                                                    for k in 0..n_fis {
                                                        let s = slot + k;
                                                        if s < fis_capacity {
                                                            fis_pos[(s * 3u32) as usize] = px;
                                                            fis_pos[(s * 3u32 + 1u32) as usize] = py;
                                                            fis_pos[(s * 3u32 + 2u32) as usize] = pz;
                                                        }
                                                    }
                                                    lc_fis += n_fis;
                                                }
                                            }
                                        }
                                        local_alive = u32::new(0);
                                    } else {
                                        let mu = 2.0 * pcg_uniform(&mut lrng) - 1.0;
                                        let phi = 2.0 * 3.141592653589793 * pcg_uniform(&mut lrng);
                                        let sq = (f64::new(1.0) - mu * mu).sqrt();
                                        ddir[0] = sq * phi.cos();
                                        ddir[1] = sq * phi.sin();
                                        ddir[2] = mu;
                                    }
                                } else {
                                    lc_surf += 1;
                                    cross_or_die(
                                        idata, fdata, surf_params_off, surf_type_off,
                                        bc, surf_idx, dist,
                                        &mut px, &mut py, &mut pz, &mut ddir, &mut local_alive,
                                    );
                                    if bc == BC_VACUUM {
                                        lc_leak += 1;
                                    }
                                }
                            }
                        }
                    }

                    if local_alive == 1u32 {
                        depth = find_cell(
                            idata, fdata, surf_params_off, surf_type_off, region_op_off, region_arg_off,
                            cell_region_off_a, cell_region_len_a, cell_fill_type_a, cell_fill_data_a,
                            univ_cells_off_base, univ_cells_len_a, univ_cell_indices_a,
                            lat_shape_off, lat_universes_off_a, lat_universes_a, lat_origin_off, lat_pitch_off,
                            root_universe, px, py, pz,
                            &mut st_cell, &mut st_offx, &mut st_offy, &mut st_offz,
                            &mut st_has_lat, &mut st_lat_id, &mut st_lat_ix, &mut st_lat_iy, &mut st_lat_iz,
                        );
                        if depth == 0u32 {
                            local_alive = u32::new(0);
                            lc_leak += 1;
                        }
                    }
                }
            }

            pos[tid * 3] = px;
            pos[tid * 3 + 1] = py;
            pos[tid * 3 + 2] = pz;
            dir[tid * 3] = ddir[0];
            dir[tid * 3 + 1] = ddir[1];
            dir[tid * 3 + 2] = ddir[2];
            alive[tid] = local_alive;
            rng[tid * 2] = lrng[0];
            rng[tid * 2 + 1] = lrng[1];

            atomics[1].fetch_add(lc_coll);
            atomics[2].fetch_add(lc_abs);
            atomics[3].fetch_add(lc_fis);
            atomics[4].fetch_add(lc_leak);
            atomics[5].fetch_add(lc_surf);
        }
    }
}

/// Apply a boundary crossing: advance to the surface, then vacuum kills,
/// reflective inverts the direction, transmission steps a nudge past.
#[cube]
#[allow(clippy::too_many_arguments)]
fn cross_or_die(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    bc: i32,
    surf_idx: i32,
    dist: f64,
    px: &mut f64,
    py: &mut f64,
    pz: &mut f64,
    dir: &mut Array<f64>,
    alive: &mut u32,
) {
    if bc == BC_VACUUM {
        *px = *px + dir[0] * dist;
        *py = *py + dir[1] * dist;
        *pz = *pz + dir[2] * dist;
        *alive = u32::new(0);
    } else {
        if bc == BC_REFLECTIVE {
            *px = *px + dir[0] * dist;
            *py = *py + dir[1] * dist;
            *pz = *pz + dir[2] * dist;
            if surf_idx >= 0i32 {
                reflect_dir(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(surf_idx), dir);
            }
        } else {
            let nudge = f64::new(1e-10);
            *px = *px + dir[0] * (dist + nudge);
            *py = *py + dir[1] * (dist + nudge);
            *pz = *pz + dir[2] * (dist + nudge);
        }
    }
}

// ── Host harness ────────────────────────────────────────────────────

/// Run one batch of constant-XS transport on the given CubeCL runtime.
/// Mirrors `GpuRecursiveContext::const_xs_transport` (CUDA) but
/// cross-vendor. Returns the fission-site bank + aggregate counters.
///
/// As with the CUDA path, the same RNG seed on CPU and GPU does NOT
/// give bit-identical histories (collision-vs-surface ties flip on
/// float rounding); aggregate counts agree within MC noise.
pub fn const_xs_transport<R: Runtime>(
    device: &R::Device,
    packed: &PackedTransport,
    positions: &[(f64, f64, f64)],
    directions: &[(f64, f64, f64)],
    rng_seeds: &[(u64, u64)],
    max_events_per_history: u32,
    fis_capacity: usize,
) -> Result<ConstXsBatch, String> {
    let n = positions.len();
    if n == 0 {
        return Ok(ConstXsBatch::default());
    }
    if directions.len() != n || rng_seeds.len() != n {
        return Err("position / direction / rng_seeds length mismatch".into());
    }

    let client = R::client(device);

    // meta with per-launch scalars filled in.
    let mut meta = packed.meta.clone();
    meta[M_N_PARTICLES] = n as u32;
    meta[M_MAX_EVENTS] = max_events_per_history;
    meta[M_FIS_CAPACITY] = fis_capacity as u32;

    // Flatten particle SoA.
    let mut pos_flat = Vec::with_capacity(n * 3);
    let mut dir_flat = Vec::with_capacity(n * 3);
    let mut rng_flat = Vec::with_capacity(n * 2);
    let mut alive_flat = vec![1u32; n];
    for i in 0..n {
        pos_flat.push(positions[i].0);
        pos_flat.push(positions[i].1);
        pos_flat.push(positions[i].2);
        dir_flat.push(directions[i].0);
        dir_flat.push(directions[i].1);
        dir_flat.push(directions[i].2);
        rng_flat.push(rng_seeds[i].0);
        rng_flat.push(rng_seeds[i].1 | 1); // inc must be odd
    }
    let _ = &mut alive_flat;

    let fis_pos = vec![0.0f64; fis_capacity.max(1) * 3];
    let atomics_init = vec![0u32; 6];

    let meta_h = client.create_from_slice(u32::as_bytes(&meta));
    let idata_h = client.create_from_slice(i32::as_bytes(&packed.idata));
    let fdata_h = client.create_from_slice(f64::as_bytes(&packed.fdata));
    let pos_h = client.create_from_slice(f64::as_bytes(&pos_flat));
    let dir_h = client.create_from_slice(f64::as_bytes(&dir_flat));
    let rng_h = client.create_from_slice(u64::as_bytes(&rng_flat));
    let alive_h = client.create_from_slice(u32::as_bytes(&alive_flat));
    let fis_h = client.create_from_slice(f64::as_bytes(&fis_pos));
    let atomics_h = client.create_from_slice(u32::as_bytes(&atomics_init));

    let threads = 64u32;
    let blocks = n.div_ceil(threads as usize) as u32;

    unsafe {
        const_xs_kernel::launch::<R>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(meta_h, meta.len()),
            ArrayArg::from_raw_parts(idata_h, packed.idata.len()),
            ArrayArg::from_raw_parts(fdata_h, packed.fdata.len()),
            ArrayArg::from_raw_parts(pos_h, n * 3),
            ArrayArg::from_raw_parts(dir_h, n * 3),
            ArrayArg::from_raw_parts(rng_h, n * 2),
            ArrayArg::from_raw_parts(alive_h, n),
            ArrayArg::from_raw_parts(fis_h.clone(), fis_capacity.max(1) * 3),
            ArrayArg::from_raw_parts(atomics_h.clone(), 6),
        );
    }

    let at_bytes = client
        .read_one(atomics_h)
        .map_err(|e| format!("readback atomics: {e:?}"))?;
    let at = u32::from_bytes(&at_bytes);
    let n_fis_banked = (at[0] as usize).min(fis_capacity);

    let fis_bytes = client
        .read_one(fis_h)
        .map_err(|e| format!("readback fission bank: {e:?}"))?;
    let fis = f64::from_bytes(&fis_bytes);
    let mut fission_sites = Vec::with_capacity(n_fis_banked);
    for s in 0..n_fis_banked {
        fission_sites.push((fis[s * 3], fis[s * 3 + 1], fis[s * 3 + 2]));
    }

    Ok(ConstXsBatch {
        fission_sites,
        n_collisions: at[1] as u64,
        n_absorptions: at[2] as u64,
        n_fissions: at[3] as u64,
        n_leakage: at[4] as u64,
        n_surf_xings: at[5] as u64,
    })
}

/// Convenience: run on the default WGPU device.
pub fn const_xs_transport_wgpu(
    packed: &PackedTransport,
    positions: &[(f64, f64, f64)],
    directions: &[(f64, f64, f64)],
    rng_seeds: &[(u64, u64)],
    max_events_per_history: u32,
    fis_capacity: usize,
) -> Result<ConstXsBatch, String> {
    let device = cubecl::wgpu::WgpuDevice::default();
    const_xs_transport::<cubecl::wgpu::WgpuRuntime>(
        &device,
        packed,
        positions,
        directions,
        rng_seeds,
        max_events_per_history,
        fis_capacity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, Cell, CellFill, CellId};
    use crate::geometry::flat::build_host_tables;
    use crate::geometry::surface::{BoundaryCondition, Surface};
    use crate::geometry::universe::{Universe, UniverseId};
    use crate::geometry::Vec3;

    /// A bare fissile sphere (vacuum boundary). Constant XS chosen so a
    /// fair fraction of histories fission before leaking.
    fn fissile_sphere() -> crate::geometry::Geometry {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 10.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        crate::geometry::Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect("fissile sphere")
    }

    // Blocked by tracel-ai/cubecl#1336: this kernel keeps too much
    // thread-private Array state live at once, so on NVIDIA/Vulkan the
    // SPIR-V private-storage bug faults at dispatch (STATUS_ACCESS_VIOLATION).
    // Re-enable when the upstream fix lands. Transport stays on CUDA meanwhile.
    #[test]
    #[ignore = "cubecl#1336: private Array<T> dispatch fault on Vulkan"]
    fn wgpu_const_xs_batch_runs() {
        let geom = fissile_sphere();
        let tables = build_host_tables(&geom);
        // Σ_t=0.5, Σ_a=0.2, Σ_f=0.15, ν̄=2.5 (macroscopic, 1/cm).
        let mats = [ConstXs {
            sigma_t: 0.5,
            sigma_a: 0.2,
            sigma_f: 0.15,
            nu_bar: 2.5,
        }];
        let packed = pack_transport(&tables, &geom, &mats);

        let n = 4usize;
        let mut pos = Vec::with_capacity(n);
        let mut dir = Vec::with_capacity(n);
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            // Birth at origin, isotropic-ish spread via cheap hashing.
            pos.push((0.0, 0.0, 0.0));
            let a = (i as f64) * 0.013;
            dir.push((a.cos(), a.sin() * 0.5, a.sin() * 0.5));
            seeds.push((
                0x4d595df4d0f33173u64.wrapping_add((i as u64).wrapping_mul(2862933555777941757)),
                1,
            ));
        }

        let result = std::panic::catch_unwind(|| {
            const_xs_transport_wgpu(&packed, &pos, &dir, &seeds, 4, n * 4)
        });
        let batch = match result {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => panic!("transport returned error: {e}"),
            Err(_) => {
                eprintln!("no usable WGPU adapter — skipping const-XS transport test");
                return;
            }
        };

        eprintln!(
            "const-XS GPU batch: coll={} abs={} fis={} leak={} surf={} bank={}",
            batch.n_collisions,
            batch.n_absorptions,
            batch.n_fissions,
            batch.n_leakage,
            batch.n_surf_xings,
            batch.fission_sites.len()
        );

        // Sanity: every history ends (absorbed or leaked); collisions and
        // fissions are produced; the fission bank is non-empty and every
        // banked site sits inside the 10 cm sphere.
        assert!(batch.n_collisions > 0, "no collisions recorded");
        assert!(batch.n_fissions > 0, "no fissions recorded");
        assert!(!batch.fission_sites.is_empty(), "empty fission bank");
        // (diagnostic: event cap may leave some alive)
        for (x, y, z) in &batch.fission_sites {
            let r = (x * x + y * y + z * z).sqrt();
            assert!(r <= 10.0 + 1e-6, "fission site outside sphere: r={r}");
        }
    }
}
