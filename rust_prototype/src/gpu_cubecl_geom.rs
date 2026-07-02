// SPDX-License-Identifier: MIT
//! Shared CubeCL device-side geometry walk.
//!
//! `gpu_transport_cubecl` (const-XS transport), `gpu_ce_cubecl` (fused
//! CE transport), and `gpu_render` (preview raycaster) each need the
//! same recursive universe/rect-lattice descent over the flattened
//! geometry SoA. This module is the single copy: the `#[cube]`
//! primitives (surface eval/distance, CSG containment, boundary
//! crossing) plus the `find_cell` / `trace_step` walk, written once so
//! a fix applies to every kernel instead of needing separate patches
//! to N hand-copied files.
//!
//! The SoA tag constants are re-exported from
//! [`crate::geometry::flat`], the single source of truth also
//! consumed by the legacy CUDA (`.cu`) backend — this module does not
//! redefine them.
//!
//! `find_cell` resolves `CellFill::Void` cells as ordinary terminal
//! cells (matching `gr_find_cell` in `geom_recursive.cu` and the CPU
//! `EffectiveFill::Void` free-stream) rather than treating them as an
//! immediate leak; callers detect "no material" via `cell_fill_type !=
//! FILL_MATERIAL` and free-stream through with `trace_step`, exactly
//! as a vacuum/air region should. Hex-lattice fills are not
//! implemented by this walk (only the legacy CUDA path has hex
//! descent); callers must reject `CellFill::HexLattice` scenes before
//! upload via [`crate::geometry::flat::check_gpu_supported`] — the
//! hex branch here is unreachable defensive fallback, not a supported
//! path.

use cubecl::prelude::*;

pub(crate) use crate::geometry::flat::{
    BC_REFLECTIVE, BC_TRANSMISSION, BC_VACUUM, FILL_LATTICE, FILL_MATERIAL, FILL_UNIVERSE,
    FILL_VOID, REGION_COMPLEMENT, REGION_HALFSPACE_NEG, REGION_HALFSPACE_POS, REGION_INTERSECTION,
    REGION_UNION, SURF_CYL_X, SURF_CYL_Y, SURF_CYL_Z, SURF_PLANE_GENERAL, SURF_PLANE_X,
    SURF_PLANE_Y, SURF_PLANE_Z, SURF_SPHERE,
};

/// Max recursion depth of the universe/lattice descent (root universe
/// + up to 3 nested lattices/universes covers every ICSBEP scene).
pub(crate) const MAX_DEPTH: u32 = 4;
pub(crate) const MAX_DEPTH_USIZE: usize = 4;
/// f64 slots per surface in `fdata[surf_params_off..]`.
pub(crate) const SURF_STRIDE: u32 = crate::geometry::flat::SURF_PARAM_STRIDE as u32;

/// Evaluate surface `s_idx` at local point — sign tells which halfspace.
/// Mirrors `gr_surf_eval`.
#[cube]
pub(crate) fn surf_eval(
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
pub(crate) fn dist_plane(p: f64, d: f64, x0: f64, big: f64, tol: f64) -> f64 {
    let mut out = big;
    if d.abs() > f64::new(1e-300) {
        let t = (x0 - p) / d;
        out = select(t > tol, t, big);
    }
    out
}

#[cube]
pub(crate) fn dist_sphere(px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64, cx: f64, cy: f64, cz: f64, r: f64, big: f64, tol: f64) -> f64 {
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
pub(crate) fn dist_cyl(p1: f64, p2: f64, d1: f64, d2: f64, c1: f64, c2: f64, r: f64, big: f64, tol: f64) -> f64 {
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

/// Distance from `(px,py,pz)` along unit `(dx,dy,dz)` to surface
/// `s_idx`; `1e30` for no forward hit. Mirrors `gr_surf_dist`.
#[cube]
pub(crate) fn surf_dist(
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
/// Mirrors `gr_cell_contains`.
#[cube]
pub(crate) fn cell_contains(
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
pub(crate) fn reflect_dir(idata: &Array<i32>, fdata: &Array<f64>, surf_params_off: u32, surf_type_off: u32, s_idx: u32, dir: &mut Array<f64>) {
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

/// Apply a boundary crossing: advance to the surface, then vacuum kills,
/// reflective inverts the direction, transmission steps a nudge past.
/// Mirrors `gr_apply_boundary`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cross_or_die(
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

/// Recursive cell-find. Writes the resolved stack into the parallel
/// `st_*` arrays (length ≥ MAX_DEPTH) and returns the depth (0 = leak).
/// Single-exit; mirrors `gr_find_cell` (rect lattices + universes;
/// hex lattices are rejected before upload by
/// [`crate::geometry::flat::check_gpu_supported`], so that branch is
/// unreachable defensive fallback here, not a supported path).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_cell(
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
                    if ft == FILL_VOID {
                        // Void is a valid terminal cell — the caller
                        // free-streams through it via `trace_step`
                        // (mat < 0 branch), exactly like an air/vacuum
                        // gap on the CPU/CUDA paths. Not a leak.
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
                                // Hex lattice — unsupported on this walk.
                                // `check_gpu_supported` rejects any scene
                                // with a HexLattice cell before a kernel
                                // using this function is ever launched;
                                // this is unreachable defensive fallback.
                                depth = u32::new(0);
                                fc_done = true;
                            }
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
pub(crate) fn trace_step(
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

/// Rect-lattice distance to next grid crossing along the ray. Mirrors
/// `gr_lattice_distance_to_grid`.
#[cube]
pub(crate) fn grid_dist(
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
pub(crate) fn grid_axis(pos: f64, d: f64, pitch: f64, idx: i32, cur_best: f64) -> f64 {
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
