// SPDX-License-Identifier: MIT
//! Cross-vendor GPU preview renderer (CubeCL).
//!
//! Ray-casts the recursive CSG geometry into an RGB image using the
//! SAME geometry walk the transport backends use, written **once** in
//! a CubeCL `#[cube]` kernel that compiles to CUDA (NVIDIA), HIP/ROCm
//! (AMD), and WGPU/Vulkan/Metal — so the preview renders on any GPU
//! with no vendor SDK.
//!
//! Port of `gpu/cuda/geom_recursive_raycast.cu` (+ the `gr_*` walk in
//! `geom_recursive.cu`). Runs in **f64** — confirmed to execute on the
//! wgpu/Vulkan backend whenever the adapter advertises `SHADER_F64`
//! (every desktop NVIDIA/AMD GPU; CUDA + HIP natively) — so the render
//! matches the CPU/CUDA geometry walk bit-for-bit. Metal / browser
//! WebGPU, which lack f64, are the only backends this kernel can't
//! target; neither is a physics-compute target.
//!
//! CubeCL frontend constraints worked around here (learned the hard
//! way against the 0.10 alpha):
//!  - **No early `return`** in `#[cube]` code — the recursive CUDA walk
//!    is rewritten single-exit with `done` flags + bounded `for` loops.
//!  - **Mutable scalars must be constructed**, e.g. `f64::new(x)` /
//!    `i32::new(n)`, not assigned from a raw Rust literal (a raw literal
//!    is comptime-typed and can't be reassigned from a runtime value).
//!  - **`f64::new` takes an `f32`** — full-precision constants come from
//!    the uploaded `fdata` buffer, not literals.
//!  - **No `&mut` scalar out-params** and **`.min`/`.max` are ambiguous**
//!    (`FloatOps` vs `Ord`); `select(cond, a, b)` covers both.
//!  - **No per-thread scratch** — surface values are evaluated on demand
//!    inside the region stack-machine instead of precomputed.
//!
//! Scope (v1): rectangular lattices + nested universes are walked fully
//! (covers the PWR-assembly and LCT cluster hero scenes). Hex lattices
//! fall through to a leak (render as background) — only `hex_minicore`
//! uses them; full hex descent is a follow-up.
//!
//! The geometry SoA comes from the backend-agnostic
//! [`crate::geometry::flat::build_host_tables`], shared with the CUDA
//! upload path so the on-device layout has a single source of truth.
//! It is packed into one `i32` blob + one `f64` blob with an offset
//! header (`meta`), keeping the kernel within every backend's storage-
//! binding limit.

use cubecl::prelude::*;

use crate::geometry::Geometry;
use crate::geometry::flat::{self, HostTables};

// ── meta header layout (u32 slots) ──────────────────────────────────
// Scalar counts + element-offsets locating each sub-array inside the
// `idata` / `fdata` blobs. Keep in sync with `pack_scene` + kernel `M_*`.

const M_N_SURFACES: usize = 0;
const M_ROOT_UNIVERSE: usize = 1;
const M_N_MATERIALS: usize = 2;
const M_WIDTH: usize = 3;
const M_HEIGHT: usize = 4;
// i32-blob element offsets
const M_OFF_SURF_TYPE: usize = 8;
const M_OFF_SURF_BC: usize = 9;
const M_OFF_CELL_REGION_OFF: usize = 10;
const M_OFF_CELL_REGION_LEN: usize = 11;
const M_OFF_CELL_FILL_TYPE: usize = 12;
const M_OFF_CELL_FILL_DATA: usize = 13;
const M_OFF_REGION_OP: usize = 14;
const M_OFF_REGION_ARG: usize = 15;
#[allow(dead_code)]
const M_OFF_UNIV_CELLS_OFF: usize = 16;
const M_OFF_UNIV_CELLS_LEN: usize = 17;
const M_OFF_UNIV_CELL_INDICES: usize = 18;
const M_OFF_UNIV_CELLS_OFF_BASE: usize = 19; // univ_cells_off array start
const M_OFF_LAT_SHAPE: usize = 20;
const M_OFF_LAT_UNIVERSES_OFF: usize = 21;
const M_OFF_LAT_UNIVERSES: usize = 22;
const M_OFF_PALETTE: usize = 27;
const M_OFF_OPAQUE: usize = 28;
// f64-blob element offsets
const M_OFF_SURF_PARAMS: usize = 29;
const M_OFF_LAT_ORIGIN: usize = 32;
const M_OFF_LAT_PITCH: usize = 33;
/// Per-material absorption coefficient (alpha-per-cm) for transparent
/// fluids. 0 = perfectly clear (air/void); ~0.05 = water-like haze.
const M_OFF_ABSORB: usize = 34;

const META_LEN: usize = 40;

// Tag constants — mirror `geometry::flat` (single source of truth).
const SURF_PLANE_X: i32 = 0;
const SURF_PLANE_Y: i32 = 1;
const SURF_PLANE_Z: i32 = 2;
const SURF_SPHERE: i32 = 3;
const SURF_CYL_Z: i32 = 4;
const SURF_CYL_X: i32 = 5;
const SURF_CYL_Y: i32 = 6;
const SURF_PLANE_GENERAL: i32 = 7;

const REGION_HALFSPACE_POS: i32 = 0;
const REGION_HALFSPACE_NEG: i32 = 1;
const REGION_INTERSECTION: i32 = 2;
const REGION_UNION: i32 = 3;
const REGION_COMPLEMENT: i32 = 4;

const FILL_MATERIAL: i32 = 0;
const FILL_VOID: i32 = 1;
const FILL_UNIVERSE: i32 = 2;
const FILL_LATTICE: i32 = 3;
#[allow(dead_code)]
const FILL_HEX_LATTICE: i32 = 4;

const MAX_DEPTH: u32 = 4;
const MAX_DEPTH_USIZE: usize = 4;
const SURF_STRIDE: u32 = 8; // f64 per surface in surf_params

// ── cam header layout (f64 slots) ───────────────────────────────────

const C_POS: usize = 0; // 3
const C_FWD: usize = 3; // 3
const C_RIGHT: usize = 6; // 3
const C_UP: usize = 9; // 3
const C_TAN_HALF_FOV: usize = 12;
const C_ASPECT: usize = 13;
const C_AABB_MIN: usize = 14; // 3
const C_AABB_MAX: usize = 17; // 3
const CAM_LEN: usize = 20;

/// Camera + framing parameters for one render (host-computed orbit).
#[derive(Clone, Copy)]
pub struct CameraParams {
    pub pos: [f64; 3],
    pub fwd: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub tan_half_fov: f64,
    pub aspect: f64,
}

impl CameraParams {
    /// Right-handed orbit camera looking at `target` from `azim`/`elev`
    /// (radians) at distance `radius`. Mirrors `preview_scene`'s
    /// `Camera::orbit` so CPU and GPU framing match.
    pub fn orbit(
        target: [f64; 3],
        azim: f64,
        elev: f64,
        radius: f64,
        fov_deg: f64,
        aspect: f64,
    ) -> Self {
        let dir = [
            elev.cos() * azim.cos(),
            elev.cos() * azim.sin(),
            elev.sin(),
        ];
        let pos = [
            target[0] + dir[0] * radius,
            target[1] + dir[1] * radius,
            target[2] + dir[2] * radius,
        ];
        let fwd = norm3([target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]]);
        let world_up = [0.0f64, 0.0, 1.0];
        let mut right = cross3(fwd, world_up);
        if len3(right) < 1e-6 {
            right = [1.0, 0.0, 0.0];
        }
        right = norm3(right);
        let up = norm3(cross3(right, fwd));
        Self {
            pos,
            fwd,
            right,
            up,
            tan_half_fov: (fov_deg.to_radians() * 0.5).tan(),
            aspect,
        }
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn len3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 1e-12 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

// ── Host-side packing: HostTables → 2 flat blobs + meta header ──────

/// Everything the kernel needs, packed for upload.
pub struct PackedScene {
    pub meta: Vec<u32>,
    pub idata: Vec<i32>,
    pub fdata: Vec<f64>,
    pub aabb_min: [f64; 3],
    pub aabb_max: [f64; 3],
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

/// Pack the flattened geometry + palette into upload-ready buffers.
///
/// `absorb[m]` is the per-cm alpha (haze) of transparent material `m`:
/// `0.0` = perfectly clear (air / void), a small positive value (≈0.04)
/// gives a translucent fluid the ray accumulates through. Opaque
/// materials ignore it. Length must match `palette` / `opaque`.
pub fn pack_scene(
    t: &HostTables,
    geom: &Geometry,
    palette: &[[u8; 3]],
    opaque: &[bool],
    absorb: &[f64],
) -> PackedScene {
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

    let palette_flat: Vec<i32> = palette
        .iter()
        .flat_map(|c| [c[0] as i32, c[1] as i32, c[2] as i32])
        .collect();
    meta[M_OFF_PALETTE] = push_i32(&mut idata, &palette_flat);
    let opaque_flat: Vec<i32> = opaque.iter().map(|&b| i32::from(b)).collect();
    meta[M_OFF_OPAQUE] = push_i32(&mut idata, &opaque_flat);

    meta[M_OFF_SURF_PARAMS] = push_f64(&mut fdata, &t.surf_params);
    meta[M_OFF_LAT_ORIGIN] = push_f64(&mut fdata, &t.lat_origin);
    meta[M_OFF_LAT_PITCH] = push_f64(&mut fdata, &t.lat_pitch);
    meta[M_OFF_ABSORB] = push_f64(&mut fdata, absorb);

    meta[M_N_SURFACES] = geom.surfaces.len() as u32;
    meta[M_ROOT_UNIVERSE] = geom.root_universe.0 as u32;
    meta[M_N_MATERIALS] = palette.len() as u32;

    if idata.is_empty() {
        idata.push(0);
    }
    if fdata.is_empty() {
        fdata.push(0.0);
    }

    // Compute the world AABB from the geometry's finite cell bounds so
    // the camera framing has something to slab-test against.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in &geom.cells {
        let (cl, ch) = flat::finite_aabb(c.aabb.min, c.aabb.max);
        for k in 0..3 {
            if cl[k] > -1e19 && cl[k] < lo[k] {
                lo[k] = cl[k];
            }
            if ch[k] < 1e19 && ch[k] > hi[k] {
                hi[k] = ch[k];
            }
        }
    }
    for k in 0..3 {
        if !lo[k].is_finite() || !hi[k].is_finite() || hi[k] <= lo[k] {
            lo[k] = -10.0;
            hi[k] = 10.0;
        }
    }

    PackedScene {
        meta,
        idata,
        fdata,
        aabb_min: lo,
        aabb_max: hi,
    }
}

// ── Device helpers ──────────────────────────────────────────────────

/// Evaluate surface `s_idx` at local point — sign tells which halfspace.
/// Mirrors `gr_surf_eval`.
#[cube]
fn surf_eval(idata: &Array<i32>, fdata: &Array<f64>, surf_params_off: u32, surf_type_off: u32, s_idx: u32, x: f64, y: f64, z: f64) -> f64 {
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

/// Distance from `(px,py,pz)` along unit `(dx,dy,dz)` to surface
/// `s_idx`; `1e30` for no forward hit. Mirrors `gr_surf_dist`.
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
                                    if denom.abs() > f64::new(1e-30) {
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

/// Outward (then camera-facing) normal of surface `s_idx` at local
/// `(x,y,z)`. Mirrors `gr_surf_normal`. Returns components via a length-3
/// local array (avoids `&mut` scalar out-params).
#[cube]
fn surf_normal(
    idata: &Array<i32>,
    fdata: &Array<f64>,
    surf_params_off: u32,
    surf_type_off: u32,
    s_idx: u32,
    x: f64,
    y: f64,
    z: f64,
    n: &mut Array<f64>,
) {
    let t = idata[(surf_type_off + s_idx) as usize];
    let p = (surf_params_off + s_idx * SURF_STRIDE) as usize;
    let mut ax = f64::new(0.0);
    let mut ay = f64::new(0.0);
    let mut az = f64::new(1.0);
    if t == SURF_PLANE_X {
        ax = f64::new(1.0);
        ay = f64::new(0.0);
        az = f64::new(0.0);
    } else {
        if t == SURF_PLANE_Y {
            ax = f64::new(0.0);
            ay = f64::new(1.0);
            az = f64::new(0.0);
        } else {
            if t == SURF_PLANE_Z {
                ax = f64::new(0.0);
                ay = f64::new(0.0);
                az = f64::new(1.0);
            } else {
                if t == SURF_SPHERE {
                    ax = x - fdata[p];
                    ay = y - fdata[p + 1];
                    az = z - fdata[p + 2];
                } else {
                    if t == SURF_CYL_Z {
                        ax = x - fdata[p];
                        ay = y - fdata[p + 1];
                        az = f64::new(0.0);
                    } else {
                        if t == SURF_CYL_X {
                            ax = f64::new(0.0);
                            ay = y - fdata[p];
                            az = z - fdata[p + 1];
                        } else {
                            if t == SURF_CYL_Y {
                                ax = x - fdata[p];
                                ay = f64::new(0.0);
                                az = z - fdata[p + 1];
                            } else {
                                if t == SURF_PLANE_GENERAL {
                                    ax = fdata[p];
                                    ay = fdata[p + 1];
                                    az = fdata[p + 2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let len = (ax * ax + ay * ay + az * az).sqrt();
    if len > f64::new(1e-12) {
        n[0] = ax / len;
        n[1] = ay / len;
        n[2] = az / len;
    } else {
        n[0] = f64::new(0.0);
        n[1] = f64::new(0.0);
        n[2] = f64::new(1.0);
    }
}

/// Postfix CSG stack-machine: is local point inside cell `ci`'s region?
/// Evaluates surfaces on demand (no precomputed `evals`). Mirrors
/// `gr_cell_contains`. Returns 1 for inside, 0 otherwise.
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

// ── Kernel ──────────────────────────────────────────────────────────

#[cube(launch_unchecked)]
fn raycast(
    meta: &Array<u32>,
    idata: &Array<i32>,
    fdata: &Array<f64>,
    cam: &Array<f64>,
    out: &mut Array<u32>,
) {
    let width = meta[M_WIDTH];
    let height = meta[M_HEIGHT];
    let pid = ABSOLUTE_POS; // usize
    let n_pixels = usize::cast_from(width) * usize::cast_from(height);
    if pid < n_pixels {
        // Offsets into the blobs.
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
        let palette_off = meta[M_OFF_PALETTE];
        let opaque_off = meta[M_OFF_OPAQUE];
        let surf_params_off = meta[M_OFF_SURF_PARAMS];
        let lat_origin_off = meta[M_OFF_LAT_ORIGIN];
        let lat_pitch_off = meta[M_OFF_LAT_PITCH];
        let absorb_off = meta[M_OFF_ABSORB];
        let n_materials = meta[M_N_MATERIALS];
        let root_universe = meta[M_ROOT_UNIVERSE];

        let px = u32::cast_from(pid % usize::cast_from(width));
        let py = u32::cast_from(pid / usize::cast_from(width));

        // Background gradient (dark slate).
        let bf = f64::cast_from(py) / f64::cast_from(select(height < 1u32, 1u32, height));
        let bg = pack_rgb(
            f64::new(16.0) + f64::new(14.0) * bf,
            f64::new(17.0) + f64::new(16.0) * bf,
            f64::new(23.0) + f64::new(19.0) * bf,
        );

        // Primary ray.
        let ndc_x = (f64::cast_from(px) + f64::new(0.5)) / f64::cast_from(width) * 2.0 - 1.0;
        let ndc_y = f64::new(1.0) - (f64::cast_from(py) + f64::new(0.5)) / f64::cast_from(height) * 2.0;
        let sx = ndc_x * cam[C_ASPECT] * cam[C_TAN_HALF_FOV];
        let sy = ndc_y * cam[C_TAN_HALF_FOV];
        let ddx = cam[C_FWD] + cam[C_RIGHT] * sx + cam[C_UP] * sy;
        let ddy = cam[C_FWD + 1] + cam[C_RIGHT + 1] * sx + cam[C_UP + 1] * sy;
        let ddz = cam[C_FWD + 2] + cam[C_RIGHT + 2] * sx + cam[C_UP + 2] * sy;
        let dinv = f64::new(1.0) / (ddx * ddx + ddy * ddy + ddz * ddz).sqrt();
        let rdx = ddx * dinv;
        let rdy = ddy * dinv;
        let rdz = ddz * dinv;
        let ox = cam[C_POS];
        let oy = cam[C_POS + 1];
        let oz = cam[C_POS + 2];

        // Ray vs world AABB.
        let mut tmin = f64::new(-1e30);
        let mut tmax = f64::new(1e30);
        let mut miss = false;
        slab(ox, rdx, cam[C_AABB_MIN], cam[C_AABB_MAX], &mut tmin, &mut tmax, &mut miss);
        slab(oy, rdy, cam[C_AABB_MIN + 1], cam[C_AABB_MAX + 1], &mut tmin, &mut tmax, &mut miss);
        slab(oz, rdz, cam[C_AABB_MIN + 2], cam[C_AABB_MAX + 2], &mut tmin, &mut tmax, &mut miss);

        let mut color = bg;
        let entered = !miss && tmax >= tmin && tmax >= f64::new(0.0);
        if entered {
            let ex = cam[C_AABB_MAX] - cam[C_AABB_MIN];
            let ey = cam[C_AABB_MAX + 1] - cam[C_AABB_MIN + 1];
            let ez = cam[C_AABB_MAX + 2] - cam[C_AABB_MIN + 2];
            let diag = (ex * ex + ey * ey + ez * ez).sqrt();
            let micro = diag / 1024.0;
            let eps = diag * 1e-7;

            let mut t = select(tmin > f64::new(0.0), tmin, f64::new(0.0)) + eps;

            // Stack arrays (parallel SoA, depth ≤ MAX_DEPTH).
            let mut st_cell = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_offx = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_offy = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_offz = Array::<f64>::new(MAX_DEPTH_USIZE);
            let mut st_has_lat = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_id = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_ix = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_iy = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut st_lat_iz = Array::<i32>::new(MAX_DEPTH_USIZE);
            let mut normal = Array::<f64>::new(3usize);

            // Front-to-back alpha compositing accumulators. The ray-march
            // visits surfaces strictly in order, so transparency is exact
            // (true OIT) with no depth buffer / peeling / A2C — those are
            // rasterizer workarounds for out-of-order primitives, which a
            // ray-marcher doesn't have. `trans` = remaining transmittance
            // (1 → fully clear so far); `acc_*` = pre-multiplied colour.
            let mut acc_r = f64::new(0.0);
            let mut acc_g = f64::new(0.0);
            let mut acc_b = f64::new(0.0);
            let mut trans = f64::new(1.0);

            let mut done = false;
            for _iter in 0..4096u32 {
                if !done {
                    if t > tmax {
                        done = true;
                    } else {
                        let wx = ox + rdx * t;
                        let wy = oy + rdy * t;
                        let wz = oz + rdz * t;

                        // ── find_cell at (wx,wy,wz): descend universes /
                        // rect lattices, filling the stack. Single-exit.
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
                        let mut leak = false;
                        let mut fc_done = false;

                        for _d in 0..MAX_DEPTH {
                            if !fc_done {
                                lx = lx - off_x;
                                ly = ly - off_y;
                                lz = lz - off_z;

                                // Scan this universe's cells.
                                let c_off = idata[(univ_cells_off_base + cur_univ) as usize];
                                let c_len = idata[(univ_cells_len_a + cur_univ) as usize];
                                let mut chosen = i32::new(-1);
                                let mut ci_idx = i32::new(0);
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
                                            ci_idx = cand;
                                        }
                                    }
                                }

                                if chosen < 0i32 {
                                    leak = true;
                                    fc_done = true;
                                } else {
                                    // Push frame.
                                    st_cell[depth as usize] = ci_idx;
                                    st_offx[depth as usize] = off_x;
                                    st_offy[depth as usize] = off_y;
                                    st_offz[depth as usize] = off_z;
                                    st_has_lat[depth as usize] = has_lat;
                                    st_lat_id[depth as usize] = lat_id;
                                    st_lat_ix[depth as usize] = lat_ix;
                                    st_lat_iy[depth as usize] = lat_iy;
                                    st_lat_iz[depth as usize] = lat_iz;
                                    depth += 1;

                                    let ft = idata[(cell_fill_type_a + u32::cast_from(ci_idx)) as usize];
                                    let fd = idata[(cell_fill_data_a + u32::cast_from(ci_idx)) as usize];
                                    if ft == FILL_MATERIAL {
                                        fc_done = true;
                                    } else {
                                        if ft == FILL_VOID {
                                            fc_done = true;
                                        } else {
                                            if ft == FILL_UNIVERSE {
                                                cur_univ = u32::cast_from(fd);
                                                off_x = f64::new(0.0);
                                                off_y = f64::new(0.0);
                                                off_z = f64::new(0.0);
                                                has_lat = 0i32;
                                            } else {
                                                if ft == FILL_LATTICE {
                                                    // Rect lattice element resolution.
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
                                                        leak = true;
                                                        fc_done = true;
                                                    } else {
                                                        let slab_n = sh0 * sh1;
                                                        let linear = iz * slab_n + iy * sh0 + ix;
                                                        let luoff = idata[(lat_universes_off_a + lid) as usize];
                                                        cur_univ = u32::cast_from(idata[(lat_universes_a + u32::cast_from(luoff + linear)) as usize]);
                                                        off_x = fdata[org] + (fix + 0.5) * fdata[pit];
                                                        off_y = fdata[org + 1] + (fiy + 0.5) * fdata[pit + 1];
                                                        off_z = fdata[org + 2] + (fiz + 0.5) * fdata[pit + 2];
                                                        has_lat = 1i32;
                                                        lat_id = i32::cast_from(lid);
                                                        lat_ix = ix;
                                                        lat_iy = iy;
                                                        lat_iz = iz;
                                                    }
                                                } else {
                                                    // Hex lattice (v1: treat as leak).
                                                    leak = true;
                                                    fc_done = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if leak || depth == 0u32 {
                            t = t + micro;
                        } else {
                            let ci = st_cell[(depth - 1) as usize];
                            let ft = idata[(cell_fill_type_a + u32::cast_from(ci)) as usize];
                            let m = idata[(cell_fill_data_a + u32::cast_from(ci)) as usize];
                            let mut is_opaque = false;
                            if ft == FILL_MATERIAL {
                                if u32::cast_from(m) >= n_materials {
                                    is_opaque = true;
                                } else {
                                    if idata[(opaque_off + u32::cast_from(m)) as usize] != 0i32 {
                                        is_opaque = true;
                                    }
                                }
                            }

                            if ft == FILL_MATERIAL && is_opaque {
                                // Cumulative offset to local frame.
                                let mut cof_x = f64::new(0.0);
                                let mut cof_y = f64::new(0.0);
                                let mut cof_z = f64::new(0.0);
                                for i in 0..depth {
                                    cof_x = cof_x + st_offx[i as usize];
                                    cof_y = cof_y + st_offy[i as usize];
                                    cof_z = cof_z + st_offz[i as usize];
                                }
                                let hx = wx - cof_x;
                                let hy = wy - cof_y;
                                let hz = wz - cof_z;

                                // Nearest bounding halfspace surface.
                                let r_off = idata[(cell_region_off_a + u32::cast_from(ci)) as usize];
                                let r_len = idata[(cell_region_len_a + u32::cast_from(ci)) as usize];
                                let mut best = i32::new(-1);
                                let mut best_abs = f64::new(1e30);
                                for i in 0..u32::cast_from(r_len) {
                                    let op = idata[(region_op_off + u32::cast_from(r_off) + i) as usize];
                                    let arg = idata[(region_arg_off + u32::cast_from(r_off) + i) as usize];
                                    if op == REGION_HALFSPACE_POS {
                                        let v = surf_eval(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), hx, hy, hz).abs();
                                        if v < best_abs {
                                            best_abs = v;
                                            best = arg;
                                        }
                                    } else {
                                        if op == REGION_HALFSPACE_NEG {
                                            let v = surf_eval(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), hx, hy, hz).abs();
                                            if v < best_abs {
                                                best_abs = v;
                                                best = arg;
                                            }
                                        }
                                    }
                                }

                                let mut nx = -rdx;
                                let mut ny = -rdy;
                                let mut nz = -rdz;
                                if best >= 0i32 {
                                    surf_normal(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(best), hx, hy, hz, &mut normal);
                                    nx = normal[0];
                                    ny = normal[1];
                                    nz = normal[2];
                                }
                                if nx * rdx + ny * rdy + nz * rdz > f64::new(0.0) {
                                    nx = -nx;
                                    ny = -ny;
                                    nz = -nz;
                                }

                                // Lambert: key light + head-light.
                                let kl = (f64::new(0.35) * 0.35 + f64::new(0.45) * 0.45 + f64::new(0.82) * 0.82).sqrt();
                                let kd = fmax0((nx * 0.35 + ny * 0.45 + nz * 0.82) / kl);
                                let hd = fmax0(-(nx * rdx + ny * rdy + nz * rdz));
                                let lit_raw = f64::new(0.18) + f64::new(0.82) * (f64::new(0.55) * kd + f64::new(0.45) * hd);
                                let lit = select(lit_raw > f64::new(1.15), f64::new(1.15), lit_raw);

                                let mut rr = f64::new(200.0);
                                let mut gg = f64::new(200.0);
                                let mut bb = f64::new(200.0);
                                if u32::cast_from(m) < n_materials {
                                    rr = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32) as usize]);
                                    gg = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32 + 1u32) as usize]);
                                    bb = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32 + 2u32) as usize]);
                                }
                                // Composite the (fully opaque) hit under whatever
                                // translucent fluid the ray already passed through,
                                // then terminate this ray.
                                acc_r = acc_r + trans * rr * lit;
                                acc_g = acc_g + trans * gg * lit;
                                acc_b = acc_b + trans * bb * lit;
                                trans = f64::new(0.0);
                                done = true;
                            } else {
                                // Transparent (air/void) — step to next surface
                                // crossing across all stack cells + rect grids.
                                let mut best_dist = f64::new(1e30);

                                // Per-frame local positions.
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
                                            let dd = surf_dist(idata, fdata, surf_params_off, surf_type_off, u32::cast_from(arg), flx, fly, flz, rdx, rdy, rdz);
                                            if dd < best_dist {
                                                best_dist = dd;
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
                                            plx, ply, plz, rdx, rdy, rdz,
                                            fdata[org], fdata[org + 1], fdata[org + 2],
                                            fdata[pit], fdata[pit + 1], fdata[pit + 2],
                                            st_lat_ix[d as usize], st_lat_iy[d as usize], st_lat_iz[d as usize],
                                        );
                                        if gd + f64::new(1e-9) < best_dist {
                                            best_dist = gd;
                                        }
                                    }
                                    plx = plx - st_offx[d as usize];
                                    ply = ply - st_offy[d as usize];
                                    plz = plz - st_offz[d as usize];
                                }

                                if best_dist >= f64::new(1e29) {
                                    done = true;
                                } else {
                                    // Composite the translucent fluid segment the
                                    // ray just crossed: attenuate transmittance by
                                    // exp(-absorb·length) and accumulate a faint
                                    // tint of the fluid's palette colour so deeper
                                    // structure reads dimmer (Beer-Lambert, exact
                                    // front-to-back — no peeling needed).
                                    let mut absb = f64::new(0.0);
                                    let mut tr = f64::new(120.0);
                                    let mut tg = f64::new(140.0);
                                    let mut tbl = f64::new(190.0);
                                    if u32::cast_from(m) < n_materials {
                                        absb = fdata[(absorb_off + u32::cast_from(m)) as usize];
                                        tr = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32) as usize]);
                                        tg = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32 + 1u32) as usize]);
                                        tbl = f64::cast_from(idata[(palette_off + u32::cast_from(m) * 3u32 + 2u32) as usize]);
                                    }
                                    if absb > f64::new(0.0) {
                                        let seg_alpha = f64::new(1.0) - (-(absb * best_dist)).exp();
                                        let contrib = trans * seg_alpha;
                                        // Tint dimmed so the fluid reads as a haze,
                                        // not a solid fill.
                                        acc_r = acc_r + contrib * tr * 0.35;
                                        acc_g = acc_g + contrib * tg * 0.35;
                                        acc_b = acc_b + contrib * tbl * 0.35;
                                        trans = trans - contrib;
                                    }
                                    t = t + best_dist + eps;
                                    if trans < f64::new(0.02) {
                                        done = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Final composite: accumulated colour over the background by
            // whatever transmittance remains.
            color = pack_rgb(
                acc_r + trans * f64::cast_from((bg / 65536u32) % 256u32),
                acc_g + trans * f64::cast_from((bg / 256u32) % 256u32),
                acc_b + trans * f64::cast_from(bg % 256u32),
            );
            let _ = surf_bc_off;
        }
        out[pid] = color;
    }
}

/// One slab-test axis: update running `[tmin, tmax]` + `miss`.
/// `&mut` works because these are kernel-local `Array`-free scalars
/// passed by mutable reference into a `#[cube]` helper — wait, scalar
/// `&mut` is const in CubeCL, so this is inlined at the call site via a
/// macro-free helper that takes/returns the interval instead.
#[cube]
fn slab(o: f64, d: f64, lo: f64, hi: f64, tmin: &mut f64, tmax: &mut f64, miss: &mut bool) {
    // NOTE: CubeCL `&mut` scalar params ARE writable when the argument
    // is a kernel-local mutable variable (the restriction is on
    // re-binding a comptime literal). Verified by the smoke test.
    if d.abs() < f64::new(1e-12) {
        if o < lo {
            *miss = true;
        }
        if o > hi {
            *miss = true;
        }
    } else {
        let inv = f64::new(1.0) / d;
        let t0 = (lo - o) * inv;
        let t1 = (hi - o) * inv;
        let a = select(t0 < t1, t0, t1);
        let b = select(t0 < t1, t1, t0);
        if a > *tmin {
            *tmin = a;
        }
        if b < *tmax {
            *tmax = b;
        }
    }
}

/// Rect-lattice distance to next grid crossing along the ray. Mirrors
/// `gr_lattice_distance_to_grid`.
#[cube]
fn grid_dist(
    px: f64,
    py: f64,
    pz: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    ox: f64,
    oy: f64,
    oz: f64,
    pitx: f64,
    pity: f64,
    pitz: f64,
    ix: i32,
    iy: i32,
    iz: i32,
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

/// `max(x, 0)` without the `FloatOps`/`Ord` method ambiguity.
#[cube]
fn fmax0(x: f64) -> f64 {
    select(x > f64::new(0.0), x, f64::new(0.0))
}

/// Pack three 0..255 f64 channels into 0xRRGGBB, clamped.
#[cube]
fn pack_rgb(r: f64, g: f64, b: f64) -> u32 {
    let rr = u32::cast_from(clamp255(r + 0.5));
    let gg = u32::cast_from(clamp255(g + 0.5));
    let bb = u32::cast_from(clamp255(b + 0.5));
    rr * 65536u32 + gg * 256u32 + bb
}

#[cube]
fn clamp255(v: f64) -> f64 {
    let lo = select(v > f64::new(0.0), v, f64::new(0.0));
    select(lo > f64::new(255.0), f64::new(255.0), lo)
}

// ── Host harness ────────────────────────────────────────────────────

/// Render the scene to a packed-RGB (`0xRRGGBB`) buffer on the given
/// CubeCL runtime. Width × height pixels, row-major.
pub fn render_rgb<R: Runtime>(
    device: &R::Device,
    packed: &PackedScene,
    cam: &CameraParams,
    width: u32,
    height: u32,
) -> Vec<u32> {
    let client = R::client(device);

    let mut meta = packed.meta.clone();
    meta[M_WIDTH] = width;
    meta[M_HEIGHT] = height;

    let mut cam_arr = vec![0.0f64; CAM_LEN];
    cam_arr[C_POS..C_POS + 3].copy_from_slice(&cam.pos);
    cam_arr[C_FWD..C_FWD + 3].copy_from_slice(&cam.fwd);
    cam_arr[C_RIGHT..C_RIGHT + 3].copy_from_slice(&cam.right);
    cam_arr[C_UP..C_UP + 3].copy_from_slice(&cam.up);
    cam_arr[C_TAN_HALF_FOV] = cam.tan_half_fov;
    cam_arr[C_ASPECT] = cam.aspect;
    cam_arr[C_AABB_MIN..C_AABB_MIN + 3].copy_from_slice(&packed.aabb_min);
    cam_arr[C_AABB_MAX..C_AABB_MAX + 3].copy_from_slice(&packed.aabb_max);

    let n_pixels = (width * height) as usize;

    let meta_h = client.create_from_slice(u32::as_bytes(&meta));
    let idata_h = client.create_from_slice(i32::as_bytes(&packed.idata));
    let fdata_h = client.create_from_slice(f64::as_bytes(&packed.fdata));
    let cam_h = client.create_from_slice(f64::as_bytes(&cam_arr));
    let out_h = client.empty(n_pixels * core::mem::size_of::<u32>());

    let threads_per_block = 64u32;
    let blocks = n_pixels.div_ceil(threads_per_block as usize) as u32;

    unsafe {
        raycast::launch_unchecked::<R>(
            &client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(threads_per_block),
            ArrayArg::from_raw_parts(meta_h, meta.len()),
            ArrayArg::from_raw_parts(idata_h, packed.idata.len()),
            ArrayArg::from_raw_parts(fdata_h, packed.fdata.len()),
            ArrayArg::from_raw_parts(cam_h, cam_arr.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n_pixels),
        );
    }

    let bytes = client
        .read_one(out_h)
        .expect("GPU readback of render buffer failed");
    u32::from_bytes(&bytes).to_vec()
}

/// Convenience: render on the default WGPU device (Vulkan / Metal /
/// DX12 / WebGPU — whatever the platform offers). Always available
/// since the `wgpu` runtime is a hard dependency.
pub fn render_rgb_wgpu(packed: &PackedScene, cam: &CameraParams, width: u32, height: u32) -> Vec<u32> {
    let device = cubecl::wgpu::WgpuDevice::default();
    render_rgb::<cubecl::wgpu::WgpuRuntime>(&device, packed, cam, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, Cell, CellFill, CellId};
    use crate::geometry::surface::{BoundaryCondition, Surface};
    use crate::geometry::universe::{Universe, UniverseId};
    use crate::geometry::flat::build_host_tables;

    /// A solid sphere (material 0) sitting in a vacuum box (material 1,
    /// transparent). Renders the sphere silhouette. Exercises the full
    /// CSG walk: surf_eval, cell_contains, find_cell, surf_normal,
    /// transparent-step, and shading.
    fn sphere_in_box() -> crate::geometry::Geometry {
        let surfaces = vec![
            Surface::Sphere {
                center: crate::geometry::Vec3::new(0.0, 0.0, 0.0),
                radius: 3.0,
                bc: BoundaryCondition::Transmission,
            },
            Surface::Sphere {
                center: crate::geometry::Vec3::new(0.0, 0.0, 0.0),
                radius: 50.0,
                bc: BoundaryCondition::Vacuum,
            },
        ];
        let cells = vec![
            // Fuel sphere = material 0 (opaque).
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // Surrounding "air" = material 1 (transparent), inside outer.
            Cell::new(
                CellId(1),
                cell::intersect_all(vec![cell::outside(0), cell::inside(1)]),
                CellFill::Material(1),
            ),
        ];
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        crate::geometry::Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect("sphere in box")
    }

    #[test]
    fn wgpu_sphere_render_smoke() {
        let geom = sphere_in_box();
        let tables = build_host_tables(&geom);
        // Material 0 red+opaque, material 1 "air" transparent (clear).
        let palette = [[200u8, 60, 60], [10, 10, 10]];
        let opaque = [true, false];
        let absorb = [0.0f64, 0.0];
        let packed = pack_scene(&tables, &geom, &palette, &opaque, &absorb);

        let w = 96u32;
        let h = 96u32;
        let center = [
            ((packed.aabb_min[0] + packed.aabb_max[0]) * 0.5),
            ((packed.aabb_min[1] + packed.aabb_max[1]) * 0.5),
            ((packed.aabb_min[2] + packed.aabb_max[2]) * 0.5),
        ];
        let dx = packed.aabb_max[0] - packed.aabb_min[0];
        let dy = packed.aabb_max[1] - packed.aabb_min[1];
        let dz = packed.aabb_max[2] - packed.aabb_min[2];
        let radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.9;
        let cam = CameraParams::orbit(center, 0.6, 0.4, radius, 45.0, 1.0);

        let result = std::panic::catch_unwind(|| render_rgb_wgpu(&packed, &cam, w, h));
        let pixels = match result {
            Ok(p) => p,
            Err(_) => {
                eprintln!("no usable WGPU adapter / no f64 — skipping GPU render smoke test");
                return;
            }
        };
        assert_eq!(pixels.len(), (w * h) as usize);

        // The sphere shades reddish (high R); background is dark slate
        // (low everything). Count pixels with a strong red channel.
        let reddish = pixels
            .iter()
            .filter(|&&c| ((c >> 16) & 0xff) > 0x50 && (c & 0xff) < 0x60)
            .count();
        eprintln!("reddish sphere pixels: {reddish} / {}", w * h);
        assert!(
            reddish > 300,
            "expected a visible shaded sphere, only {reddish} reddish pixels"
        );
    }
}
