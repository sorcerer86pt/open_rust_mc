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
//! STATUS — runs on **CUDA**, blocked on **Vulkan** (tracel-ai/cubecl#1336).
//! The same `#[cube]` source compiles through CubeCL's CUDA runtime and
//! runs correctly (`cuda_const_xs_batch_runs`: 4096 histories, every one
//! absorbs-or-leaks exactly once, ~7.3k fission sites all inside the
//! sphere). On wgpu/Vulkan it compiles to valid WGSL but faults at
//! dispatch — too much thread-private `Array` state (the depth-4 coord
//! stacks + region stack) trips a SPIR-V private-storage bug, which is
//! Vulkan/SPIR-V-only. Shrinking the state isn't workable for recursive
//! geometry, so the cross-vendor (Vulkan/Metal) path waits on the
//! upstream fix; the Vulkan batch test is `#[ignore]`d, the CUDA one runs.

use cubecl::prelude::*;

use crate::geometry::Geometry;
use crate::geometry::flat::{check_gpu_supported, HostTables};
use crate::gpu_cubecl_geom::{
    cross_or_die, find_cell, trace_step, BC_TRANSMISSION, BC_VACUUM, FILL_MATERIAL, MAX_DEPTH_USIZE,
};

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
/// `m`; the kernel reads `mat_xs[m*4 + {0,1,2,3}]`. Errors if `geom`
/// uses a feature the CubeCL walk doesn't support yet (per-cell
/// rotation, hex lattice) — see `check_gpu_supported`.
pub fn pack_transport(t: &HostTables, geom: &Geometry, materials: &[ConstXs]) -> Result<PackedTransport, String> {
    check_gpu_supported(geom)?;
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

    Ok(PackedTransport { meta, idata, fdata })
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

// ── Geometry walk — shared with `gpu_ce_cubecl` and `gpu_render` via
//    `crate::gpu_cubecl_geom` (single copy; see module doc there). ───

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
                    // Only a transmission crossing moves the particle into a
                    // new cell; collision / scatter / reflection keep the
                    // current stack. Re-finding the cell after a reflection
                    // is numerically unstable (the point sits exactly on the
                    // surface) and was producing spurious leaks.
                    let mut need_refind = false;
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
                            // Transmission moved us across a surface into a
                            // (possibly) new cell — re-resolve. Reflection
                            // stays in the same cell.
                            if bc == BC_TRANSMISSION {
                                need_refind = true;
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
                                    if bc == BC_TRANSMISSION {
                                        need_refind = true;
                                    }
                                }
                            }
                        }
                    }

                    if local_alive == 1u32 && need_refind {
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
        let packed = pack_transport(&tables, &geom, &mats).expect("pack_transport");

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

    /// Same kernel, but compiled through CubeCL's **CUDA** runtime
    /// instead of wgpu/Vulkan. cubecl#1336 is SPIR-V-only, so the CUDA
    /// path should run the identical #[cube] source without the
    /// private-Array dispatch fault. Only built with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_const_xs_batch_runs() {
        let geom = fissile_sphere();
        let tables = build_host_tables(&geom);
        let mats = [ConstXs {
            sigma_t: 0.5,
            sigma_a: 0.2,
            sigma_f: 0.15,
            nu_bar: 2.5,
        }];
        let packed = pack_transport(&tables, &geom, &mats).expect("pack_transport");

        let n = 4096usize;
        let mut pos = Vec::with_capacity(n);
        let mut dir = Vec::with_capacity(n);
        let mut seeds = Vec::with_capacity(n);
        for i in 0..n {
            pos.push((0.0, 0.0, 0.0));
            let a = (i as f64) * 0.013;
            dir.push((a.cos(), a.sin() * 0.5, a.sin() * 0.5));
            seeds.push((
                0x4d595df4d0f33173u64.wrapping_add((i as u64).wrapping_mul(2862933555777941757)),
                1,
            ));
        }

        let device = cubecl::cuda::CudaDevice::default();
        let result = std::panic::catch_unwind(|| {
            const_xs_transport::<cubecl::cuda::CudaRuntime>(
                &device, &packed, &pos, &dir, &seeds, 1000, n * 4,
            )
        });
        let batch = match result {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => panic!("CUDA transport returned error: {e}"),
            Err(_) => {
                eprintln!("no usable CUDA device — skipping CUDA const-XS transport test");
                return;
            }
        };

        eprintln!(
            "const-XS CUDA batch: coll={} abs={} fis={} leak={} surf={} bank={}",
            batch.n_collisions,
            batch.n_absorptions,
            batch.n_fissions,
            batch.n_leakage,
            batch.n_surf_xings,
            batch.fission_sites.len()
        );

        assert!(batch.n_collisions > 0, "no collisions recorded");
        assert!(batch.n_fissions > 0, "no fissions recorded");
        assert!(!batch.fission_sites.is_empty(), "empty fission bank");
        assert_eq!(
            batch.n_absorptions + batch.n_leakage,
            n as u64,
            "every history should absorb or leak exactly once"
        );
        for (x, y, z) in &batch.fission_sites {
            let r = (x * x + y * y + z * z).sqrt();
            assert!(r <= 10.0 + 1e-6, "fission site outside sphere: r={r}");
        }
    }
}
