// ═══════════════════════════════════════════════════════════════════════
// Event-based GPU transport. Canonical recent GPU reference is Tramm
// et al., "Toward Portable GPU Acceleration of the OpenMC Monte Carlo
// Particle Transport Code", PHYSOR 2022. Original event-based
// formulation: Brown & Martin, "Monte Carlo methods for radiation
// transport analysis on vector computers", Prog. Nucl. Energy 14(3),
// 1984. The PHYSOR 2022 paper uses 8 coarser-grained events and a
// per-event-type queue (atomic enqueue) instead of a sorted partition
// — our implementation is the simpler "sort by reaction class then
// dispatch" variant. See `specs/event-based-gpu-transport/SPEC.md`
// for the architectural choice.
//
// Sorts particles by reaction type between geometry steps so each
// reaction kernel sees a single code path. Eliminates the warp
// divergence that the persistent history-based kernel suffers on
// PWR-17×17 (active threads/warp = 6.2/32).
//
// Pipeline (one driver step per outer iteration):
//   1. gr_init_stacks         (once per batch, before the loop)
//   2. gr_trace_and_sample    (per step: geom + reaction selection)
//   3. host: prefix-sum type counts → type_offsets
//   4. gr_partition           (per step: scatter alive indices)
//   5. gr_elastic_event       (one kernel per reaction class)
//      gr_inelastic_event
//      gr_fission_event
//      gr_multi_event
//
// All device helpers are reused from transport.cu / geom_recursive.cu
// / transport_recursive.cu (concatenated by
// gpu_recursive::assemble_kernel_source).
// ═══════════════════════════════════════════════════════════════════════

#ifndef TRANSPORT_EVENT_BASED_CU
#define TRANSPORT_EVENT_BASED_CU

// Event-type encoding (matches `EV_*` constants on the host).
#define EV_NONE      -1
#define EV_ELASTIC    0
#define EV_INELASTIC  1
#define EV_FISSION    2
#define EV_N2N        3
#define EV_N3N        4
#define EV_TYPE_COUNT 5

// ── SoA stack pack / unpack helpers ───────────────────────────────────

__device__ __forceinline__ void eb_load_stack(
    int tid, int depth,
    const int* s_univ, const int* s_cell, const int* s_has_lat,
    const int* s_lat_id, const int* s_lat_ix, const int* s_lat_iy, const int* s_lat_iz,
    const double* s_offx, const double* s_offy, const double* s_offz,
    int n, GrCoord* out)
{
    // Stride layout: [n × GR_MAX_DEPTH], particle-major.
    int base = tid * GR_MAX_DEPTH;
    for (int d = 0; d < depth; ++d) {
        out[d].universe     = s_univ[base + d];
        out[d].cell_idx     = s_cell[base + d];
        out[d].has_lattice  = s_has_lat[base + d];
        out[d].lattice_id   = s_lat_id[base + d];
        out[d].lat_ix       = s_lat_ix[base + d];
        out[d].lat_iy       = s_lat_iy[base + d];
        out[d].lat_iz       = s_lat_iz[base + d];
        out[d].offx         = s_offx[base + d];
        out[d].offy         = s_offy[base + d];
        out[d].offz         = s_offz[base + d];
    }
}

__device__ __forceinline__ void eb_store_stack(
    int tid, int depth,
    int* s_univ, int* s_cell, int* s_has_lat,
    int* s_lat_id, int* s_lat_ix, int* s_lat_iy, int* s_lat_iz,
    double* s_offx, double* s_offy, double* s_offz,
    int n, const GrCoord* in)
{
    int base = tid * GR_MAX_DEPTH;
    for (int d = 0; d < depth; ++d) {
        s_univ[base + d]    = in[d].universe;
        s_cell[base + d]    = in[d].cell_idx;
        s_has_lat[base + d] = in[d].has_lattice;
        s_lat_id[base + d]  = in[d].lattice_id;
        s_lat_ix[base + d]  = in[d].lat_ix;
        s_lat_iy[base + d]  = in[d].lat_iy;
        s_lat_iz[base + d]  = in[d].lat_iz;
        s_offx[base + d]    = in[d].offx;
        s_offy[base + d]    = in[d].offy;
        s_offz[base + d]    = in[d].offz;
    }
}

// Build GrGeometry from a long argument list. Centralised so every
// kernel uses the same canonical pack. Mirrors the inline assembly in
// transport_recursive_persistent.
__device__ __forceinline__ GrGeometry eb_make_geom(
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    const int* region_op, const int* region_arg,
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    int n_lattices,
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    int n_hex_lattices,
    double* evals_per_thread)
{
    GrGeometry g;
    g.surf_type = surf_type; g.surf_params = surf_params; g.surf_bc = surf_bc;
    g.n_surfaces = n_surfaces;
    g.cell_region_off = cell_region_off; g.cell_region_len = cell_region_len;
    g.cell_fill_type = cell_fill_type; g.cell_fill_data = cell_fill_data;
    g.cell_aabb_min = cell_aabb_min; g.cell_aabb_max = cell_aabb_max;
    g.n_cells = 0;
    g.region_op = region_op; g.region_arg = region_arg;
    g.univ_cells_off = univ_cells_off; g.univ_cells_len = univ_cells_len;
    g.univ_surfaces_off = univ_surfaces_off; g.univ_surfaces_len = univ_surfaces_len;
    g.univ_cell_indices = univ_cell_indices; g.univ_surface_indices = univ_surface_indices;
    g.n_universes = 0; g.root_universe = root_universe;
    g.lat_origin = lat_origin; g.lat_pitch = lat_pitch; g.lat_shape = lat_shape;
    g.lat_universes_off = lat_universes_off; g.lat_universes = lat_universes;
    g.n_lattices = n_lattices;
    g.hex_center = hex_center; g.hex_pitch_xy = hex_pitch_xy;
    g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings; g.hex_n_axial = hex_n_axial;
    g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off; g.hex_universes = hex_universes;
    g.n_hex_lattices = n_hex_lattices;
    g.evals = evals_per_thread;
    return g;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 1: gr_init_stacks
//
// Called once per batch. Locates each alive particle's deepest cell and
// writes the coord stack to the SoA arrays. Particles outside any cell
// are killed and counted as leakage.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_init_stacks(
    const double* pos_x, const double* pos_y, const double* pos_z,
    int* alive,
    int n_particles,
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    const int* region_op, const int* region_arg,
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    int n_lattices,
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    int n_hex_lattices,
    double* evals_scratch,
    // SoA stack output
    int* d_stack_universe, int* d_stack_cell_idx, int* d_stack_has_lattice,
    int* d_stack_lattice_id, int* d_stack_lat_ix, int* d_stack_lat_iy, int* d_stack_lat_iz,
    double* d_stack_offx, double* d_stack_offy, double* d_stack_offz,
    int* d_depth,
    int* cnt_leak)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (!alive[tid]) { d_depth[tid] = 0; return; }

    GrGeometry g = eb_make_geom(
        surf_type, surf_params, surf_bc, n_surfaces,
        cell_region_off, cell_region_len, cell_fill_type, cell_fill_data,
        cell_aabb_min, cell_aabb_max, region_op, region_arg,
        univ_cells_off, univ_cells_len,
        univ_surfaces_off, univ_surfaces_len,
        univ_cell_indices, univ_surface_indices,
        root_universe,
        lat_origin, lat_pitch, lat_shape,
        lat_universes_off, lat_universes, n_lattices,
        hex_center, hex_pitch_xy, hex_pitch_z,
        hex_n_rings, hex_n_axial, hex_orientation,
        hex_universes_off, hex_universes, n_hex_lattices,
        evals_scratch + tid * n_surfaces);

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, pos_x[tid], pos_y[tid], pos_z[tid], stack);
    if (depth == 0) {
        alive[tid] = 0;
        d_depth[tid] = 0;
        atomicAdd(cnt_leak, 1);
        return;
    }
    eb_store_stack(tid, depth,
        d_stack_universe, d_stack_cell_idx, d_stack_has_lattice,
        d_stack_lattice_id, d_stack_lat_ix, d_stack_lat_iy, d_stack_lat_iz,
        d_stack_offx, d_stack_offy, d_stack_offz,
        n_particles, stack);
    d_depth[tid] = depth;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 2: gr_trace_and_sample
//
// Per step. For each alive particle:
//   * Effective material lookup
//   * If void: trace to surface, handle BC, continue (event_type = -1)
//   * Else: evaluate XS, sample collision distance vs surface distance.
//       - Surface crossing: handle BC inline (event_type = -1)
//       - Collision: sample nuclide + reaction; emit (event_type,
//         hit_nuc, mat, kT, Ni_hit, urr_xi). Capture handled inline
//         (alive[i]=0, event_type=-1). Other reactions: atomic-incr
//         d_type_count[t] and write event metadata.
//
// Surviving but no-reaction particles set event_type = -1.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(256, 2)
gr_trace_and_sample(
    Params p,
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    int* alive,
    unsigned long long* rng_state_arr,
    unsigned long long* rng_inc_arr,
    int n_particles,
    // Geometry
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    const int* region_op, const int* region_arg,
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    int n_lattices,
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    int n_hex_lattices,
    const int* lat_override_off, const int* lat_override_count,
    const int* override_lat_idx, const int* override_cell_idx,
    const int* override_mat,
    const double* mat_kT,
    int n_materials,
    int sab_nuc_idx,
    double* evals_scratch,
    // SoA stack (in/out)
    int* d_stack_universe, int* d_stack_cell_idx, int* d_stack_has_lattice,
    int* d_stack_lattice_id, int* d_stack_lat_ix, int* d_stack_lat_iy, int* d_stack_lat_iz,
    double* d_stack_offx, double* d_stack_offy, double* d_stack_offz,
    int* d_depth,
    // Per-event output
    int* d_event_type, int* d_event_hit_nuc, int* d_event_mat,
    double* d_event_kT, double* d_event_hit_Ni, double* d_event_urr_xi,
    // Atomic per-step counters
    int* d_type_count,
    // Tallies / counters
    int* cnt_coll, int* cnt_leak, int* cnt_surf, int* cnt_capture,
    double* e_el_in_sum, double* e_el_in_sq_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (!alive[tid]) { d_event_type[tid] = EV_NONE; return; }

    int rank = SCALAR_I(p, P_RANK);
    GrGeometry g = eb_make_geom(
        surf_type, surf_params, surf_bc, n_surfaces,
        cell_region_off, cell_region_len, cell_fill_type, cell_fill_data,
        cell_aabb_min, cell_aabb_max, region_op, region_arg,
        univ_cells_off, univ_cells_len,
        univ_surfaces_off, univ_surfaces_len,
        univ_cell_indices, univ_surface_indices,
        root_universe,
        lat_origin, lat_pitch, lat_shape,
        lat_universes_off, lat_universes, n_lattices,
        hex_center, hex_pitch_xy, hex_pitch_z,
        hex_n_rings, hex_n_axial, hex_orientation,
        hex_universes_off, hex_universes, n_hex_lattices,
        evals_scratch + tid * n_surfaces);

    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    double E = energy[tid];
    PcgState rng; rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];

    int depth = d_depth[tid];
    GrCoord stack[GR_MAX_DEPTH];
    eb_load_stack(tid, depth,
        d_stack_universe, d_stack_cell_idx, d_stack_has_lattice,
        d_stack_lattice_id, d_stack_lat_ix, d_stack_lat_iy, d_stack_lat_iz,
        d_stack_offx, d_stack_offy, d_stack_offz,
        n_particles, stack);

    int ev = EV_NONE;
    int hit_nuc_out = -1;
    int mat_out = -1;
    double kT_out = -1.0;
    double Ni_out = 0.0;
    double urr_xi_out = 0.0;

    // Single-step semantics: at most one tracing operation per kernel
    // invocation. A particle may free-stream through any number of
    // vacuum-filled cells / surface transmissions in one call, but
    // resolves AT MOST one collision before returning. This matches the
    // structure of the persistent history-based loop in
    // transport_recursive.cu but with the outer while replaced by the
    // outer driver loop in gpu_recursive.rs.
    int local_leak = 0, local_surf = 0, local_coll = 0, local_cap = 0;
    double local_e_el_in = 0.0, local_e_el_in_sq = 0.0;

    while (true) {
        int mat = tr_effective_material(
            &g, stack, depth,
            lat_override_off, lat_override_count,
            override_lat_idx, override_cell_idx, override_mat,
            n_lattices);

        if (mat < 0) {
            double dist; int surf_idx; int bc; int next_depth;
            GrCoord next_stack[GR_MAX_DEPTH];
            if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                               &dist, &surf_idx, &bc, next_stack, &next_depth)) {
                alive[tid] = 0; local_leak++; break;
            }
            local_surf++;
            if (bc == GR_BC_VACUUM) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                alive[tid] = 0; local_leak++; break;
            }
            if (bc == GR_BC_REFLECTIVE) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
                const double* sp = (surf_idx >= 0) ? surf_params + surf_idx * 8 : nullptr;
                gr_reflect_direction(t, sp, &dx, &dy, &dz);
                continue;
            }
            const double NUDGE = 1e-10;
            px += dx * (dist + NUDGE); py += dy * (dist + NUDGE); pz += dz * (dist + NUDGE);
            if (next_depth == 0) { alive[tid] = 0; local_leak++; break; }
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
            continue;
        }

        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t = 0.0;
        double nuc_t[MAX_NUC_PER_MAT] = {};
        double urr_xi = pcg_uniform(&rng);
        double xs_cell_kT = (mat >= 0 && mat < n_materials) ? mat_kT[mat] : -1.0;

        for (int i = 0; i < n_nuc; i++) {
            int ni    = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + i]);
            NuclideMacroXs xs = eval_nuclide_macro_xs(ni, Ni, E, urr_xi,
                                                     sab_nuc_idx, rank, p, xs_cell_kT);
            nuc_t[i] = xs.s_t;
            sum_t   += xs.s_t;
        }
        if (sum_t <= 0.0) { alive[tid] = 0; break; }

        double d_coll = -log(pcg_uniform(&rng)) / sum_t;
        double d_s; int surf_idx; int bc; int next_depth;
        GrCoord next_stack[GR_MAX_DEPTH];
        if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                           &d_s, &surf_idx, &bc, next_stack, &next_depth)) {
            alive[tid] = 0; local_leak++; break;
        }

        if (d_s < d_coll) {
            local_surf++;
            if (bc == GR_BC_REFLECTIVE) {
                px += dx * d_s; py += dy * d_s; pz += dz * d_s;
                int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
                const double* sp = (surf_idx >= 0) ? surf_params + surf_idx * 8 : nullptr;
                gr_reflect_direction(t, sp, &dx, &dy, &dz);
                continue;
            }
            if (bc == GR_BC_VACUUM) {
                px += dx * d_s; py += dy * d_s; pz += dz * d_s;
                alive[tid] = 0; local_leak++; break;
            }
            const double NUDGE = 1e-10;
            px += dx * (d_s + NUDGE); py += dy * (d_s + NUDGE); pz += dz * (d_s + NUDGE);
            if (next_depth == 0) { alive[tid] = 0; local_leak++; break; }
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
            continue;
        }

        // Collision.
        local_coll++;
        px += dx * d_coll; py += dy * d_coll; pz += dz * d_coll;

        // Sample nuclide.
        double xi_nuc = pcg_uniform(&rng) * sum_t;
        double cum = 0.0; int hit_l = 0;
        for (int i = 0; i < n_nuc; i++) {
            cum += nuc_t[i]; if (xi_nuc < cum) { hit_l = i; break; } hit_l = i;
        }
        int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + hit_l]);
        double Ni_hit = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + hit_l]);

        NuclideMacroXs hit_xs = eval_nuclide_macro_xs(
            hit_nuc, Ni_hit, E, urr_xi, sab_nuc_idx, rank, p, xs_cell_kT);

        double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
        double cum_rxn = 0.0;
        int rxn = -1;
        cum_rxn += hit_xs.s_el;     if (xi_rxn < cum_rxn) { rxn = EV_ELASTIC; }
        else { cum_rxn += hit_xs.s_inel; if (xi_rxn < cum_rxn) { rxn = EV_INELASTIC; }
        else { cum_rxn += hit_xs.s_n2n;  if (xi_rxn < cum_rxn) { rxn = EV_N2N; }
        else { cum_rxn += hit_xs.s_n3n;  if (xi_rxn < cum_rxn) { rxn = EV_N3N; }
        else { cum_rxn += hit_xs.s_fis;  if (xi_rxn < cum_rxn) { rxn = EV_FISSION; }
        else { rxn = -2; /* capture */ } } } } }

        if (rxn == -2) {
            // Capture handled inline; no reaction kernel needed.
            local_cap++;
            alive[tid] = 0;
            break;
        }

        // Elastic E-in tally happens here so it doesn't drift between
        // the geom kernel and the elastic kernel.
        if (rxn == EV_ELASTIC) {
            local_e_el_in += E;
            local_e_el_in_sq += E * E;
        }

        ev = rxn;
        hit_nuc_out = hit_nuc;
        mat_out = mat;
        kT_out = xs_cell_kT;
        Ni_out = Ni_hit;
        urr_xi_out = urr_xi;
        atomicAdd(&d_type_count[rxn], 1);
        break;
    }

    // Persist state.
    pos_x[tid] = px; pos_y[tid] = py; pos_z[tid] = pz;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    energy[tid] = E;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;
    d_depth[tid] = depth;
    eb_store_stack(tid, depth,
        d_stack_universe, d_stack_cell_idx, d_stack_has_lattice,
        d_stack_lattice_id, d_stack_lat_ix, d_stack_lat_iy, d_stack_lat_iz,
        d_stack_offx, d_stack_offy, d_stack_offz,
        n_particles, stack);

    d_event_type[tid] = ev;
    d_event_hit_nuc[tid] = hit_nuc_out;
    d_event_mat[tid] = mat_out;
    d_event_kT[tid] = kT_out;
    d_event_hit_Ni[tid] = Ni_out;
    d_event_urr_xi[tid] = urr_xi_out;

    if (local_coll > 0) atomicAdd(cnt_coll, local_coll);
    if (local_leak > 0) atomicAdd(cnt_leak, local_leak);
    if (local_surf > 0) atomicAdd(cnt_surf, local_surf);
    if (local_cap  > 0) atomicAdd(cnt_capture, local_cap);
    if (local_e_el_in    != 0.0) atomicAdd(e_el_in_sum,    local_e_el_in);
    if (local_e_el_in_sq != 0.0) atomicAdd(e_el_in_sq_sum, local_e_el_in_sq);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 2b: gr_scan_offsets
//
// Tiny scan over the 5 per-type counts. Single thread is fine — N is
// fixed at 5 and the alternative (a warp shuffle) saves microseconds
// at most. Side effects:
//   * d_type_offsets[6] ← exclusive prefix sum (offsets[0]=0)
//   * d_type_total[1]   ← total event count (used by host driver every
//                         K=EB_SYNC_EVERY steps to detect "all dead")
//   * d_type_scatter[5] ← zeroed in preparation for partition kernel
//
// Moving this onto the device eliminates the per-step DtoH→host-prefix-
// sum→HtoD round-trip the previous driver paid: the prefix sum is 5
// integer adds, smaller than the PCIe sync latency that gated the
// host-side computation.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_scan_offsets(
    const int* d_type_count,
    int* d_type_offsets,
    int* d_type_total,
    int* d_type_scatter)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int acc = 0;
    d_type_offsets[0] = 0;
    #pragma unroll
    for (int i = 0; i < EV_TYPE_COUNT; ++i) {
        d_type_scatter[i] = 0;
        acc += d_type_count[i];
        d_type_offsets[i + 1] = acc;
    }
    d_type_total[0] = acc;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 3: gr_partition
//
// Scatter alive particle indices into d_sorted_idx grouped by event
// type. Uses precomputed host-side prefix sums in d_type_offsets and a
// per-type atomic write cursor d_type_scatter.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_partition(
    const int* d_event_type,
    const int* alive,
    int n_particles,
    const int* d_type_offsets,
    int* d_type_scatter,
    int* d_sorted_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    int t = d_event_type[tid];
    if (t < 0 || !alive[tid]) return;
    int pos = atomicAdd(&d_type_scatter[t], 1);
    d_sorted_idx[d_type_offsets[t] + pos] = tid;
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 4: gr_elastic_event
//
// Thread i in [0, count_elastic). Resolves sorted_idx[type_offsets[0]+i]
// → particle tid, executes the elastic reaction body.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_elastic_event(
    Params p,
    const int* d_type_count, const int* d_type_offsets,
    const int* d_sorted_idx,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    unsigned long long* rng_state_arr,
    unsigned long long* rng_inc_arr,
    const int* d_event_hit_nuc, const double* d_event_kT, const double* d_event_urr_xi,
    int sab_nuc_idx,
    int* cnt_elastic)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = d_type_count[EV_ELASTIC];
    if (idx >= count) return;
    int tid = d_sorted_idx[d_type_offsets[EV_ELASTIC] + idx];

    int rank = SCALAR_I(p, P_RANK);
    int hit_nuc = d_event_hit_nuc[tid];
    double xs_cell_kT = d_event_kT[tid];
    double E = energy[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    PcgState rng; rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];
    double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);
    (void)rank;
    (void)d_event_urr_xi;

    // S(α,β) per-nuclide slot lookup.
    if (SCALAR_I(p, P_SAB_N_SLOTS) > 0) {
        int sab_slot = sab_select_slot(hit_nuc, xs_cell_kT, &rng, p);
        if (sab_slot >= 0 && E < PTR_D(p, P_SAB_SLOT_EMAX)[sab_slot]) {
            double E_sab, mu_sab;
            sab_sample(E, &rng, sab_slot, p, &E_sab, &mu_sab);
            E = fmax(E_sab, 1e-11);
            double phi = 2.0 * PI * pcg_uniform(&rng);
            rotate_direction(&dx, &dy, &dz, mu_sab, phi);
            goto end_elastic;
        }
    }
    {
        double cell_kT = xs_cell_kT > 0.0 ? xs_cell_kT : 600.0 * 8.617333262e-5;
        if (E < 400.0 * cell_kT) {
            double sigma = sqrt(cell_kT / A), v_n = sqrt(2.0 * E);
            double u1, u2, r_bm, th;
            u1 = pcg_uniform(&rng); u2 = pcg_uniform(&rng);
            r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30))); th = 2.0 * PI * u2;
            double vtx = r_bm * cos(th), vty = r_bm * sin(th);
            u1 = pcg_uniform(&rng); u2 = pcg_uniform(&rng);
            r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30))); th = 2.0 * PI * u2;
            double vtz = r_bm * cos(th);
            double vnx = dx * v_n, vny = dy * v_n, vnz = dz * v_n;
            double vrx = vnx - vtx, vry = vny - vty, vrz = vnz - vtz;
            double vr = sqrt(vrx * vrx + vry * vry + vrz * vrz);
            if (vr < 1e-20) vr = 1e-20;
            double ia1 = 1.0 / (1.0 + A);
            double vcx = (vnx + A * vtx) * ia1;
            double vcy = (vny + A * vty) * ia1;
            double vcz = (vnz + A * vtz) * ia1;
            double vcn = vr * A * ia1;
            double e_rel = 0.5 * (A / (A + 1.0)) * vr * vr;
            double mu_cm = sample_angular_dist(e_rel, &rng, p, hit_nuc);
            double phi = 2.0 * PI * pcg_uniform(&rng);
            double st = sqrt(fmax(0.0, 1.0 - mu_cm * mu_cm));
            double vrh_x = vrx / vr, vrh_y = vry / vr, vrh_z = vrz / vr;
            double px2, py2, pz2;
            if (fabs(vrh_z) < 0.999) {
                double ip = 1.0 / sqrt(1.0 - vrh_z * vrh_z);
                px2 = -vrh_y * ip; py2 = vrh_x * ip; pz2 = 0.0;
            } else {
                double ip = 1.0 / sqrt(1.0 - vrh_x * vrh_x);
                px2 = 0.0; py2 = -vrh_z * ip; pz2 = vrh_y * ip;
            }
            double qx = vrh_y * pz2 - vrh_z * py2;
            double qy = vrh_z * px2 - vrh_x * pz2;
            double qz = vrh_x * py2 - vrh_y * px2;
            double s_phi, c_phi; sincos(phi, &s_phi, &c_phi);
            double sx2 = mu_cm * vrh_x + st * (c_phi * px2 + s_phi * qx);
            double sy2 = mu_cm * vrh_y + st * (c_phi * py2 + s_phi * qy);
            double sz2 = mu_cm * vrh_z + st * (c_phi * pz2 + s_phi * qz);
            double vox = vcx + vcn * sx2;
            double voy = vcy + vcn * sy2;
            double voz = vcz + vcn * sz2;
            double vo = sqrt(vox * vox + voy * voy + voz * voz);
            E = 0.5 * vo * vo; if (E < 1e-11) E = 1e-11;
            if (vo > 1e-20) { dx = vox / vo; dy = voy / vo; dz = voz / vo; }
        } else {
            double mu_cm = sample_angular_dist(E, &rng, p, hit_nuc);
            double alpha = ((A - 1.0) / (A + 1.0)) * ((A - 1.0) / (A + 1.0));
            E = E * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;
            if (E < 1e-11) E = 1e-11;
            double mu_lab = (A > 1.0 + 1e-10)
                ? (1.0 + A * mu_cm) / sqrt(1.0 + A * A + 2.0 * A * mu_cm)
                : sqrt(fmax(0.0, (1.0 + mu_cm) * 0.5));
            double phi = 2.0 * PI * pcg_uniform(&rng);
            rotate_direction(&dx, &dy, &dz, mu_lab, phi);
        }
    }
end_elastic:
    energy[tid] = E;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;
    atomicAdd(cnt_elastic, 1);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 5: gr_inelastic_event
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_inelastic_event(
    Params p,
    const int* d_type_count, const int* d_type_offsets,
    const int* d_sorted_idx,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    unsigned long long* rng_state_arr,
    unsigned long long* rng_inc_arr,
    const int* d_event_hit_nuc,
    int* cnt_inelastic,
    double* e_inel_in_sum, double* e_inel_in_sq_sum,
    double* e_inel_out_sum, double* q_inel_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = d_type_count[EV_INELASTIC];
    if (idx >= count) return;
    int tid = d_sorted_idx[d_type_offsets[EV_INELASTIC] + idx];

    int rank = SCALAR_I(p, P_RANK);
    int hit_nuc = d_event_hit_nuc[tid];
    double E = energy[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    PcgState rng; rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];
    double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

    double l_inel_in = E;
    double l_inel_in_sq = E * E;

    int lv_off = __ldg(&PTR_I(p, P_LEVEL_OFFSETS)[hit_nuc]);
    int n_lev  = __ldg(&PTR_I(p, P_LEVEL_COUNTS)[hit_nuc]);
    double Q = -0.5e6;
    int selected = 0;
    int cdf_off = __ldg(&PTR_I(p, P_INEL_CDF_OFF)[hit_nuc]);
    if (cdf_off >= 0 && n_lev > 0) {
        int cdf_n_e   = __ldg(&PTR_I(p, P_INEL_CDF_N_E)[hit_nuc]);
        int cdf_n_lev = __ldg(&PTR_I(p, P_INEL_CDF_N_LEV)[hit_nuc]);
        double log_e_min = __ldg(&PTR_D(p, P_INEL_CDF_LOG_EMIN)[hit_nuc]);
        double log_e_max = __ldg(&PTR_D(p, P_INEL_CDF_LOG_EMAX)[hit_nuc]);
        double log_e = log10(fmax(E, 1e-12));
        double f = (log_e - log_e_min) / (log_e_max - log_e_min);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        double f_idx = f * (double)(cdf_n_e - 1);
        int idx_e = (int)f_idx;
        if (idx_e >= cdf_n_e - 1) idx_e = cdf_n_e - 2;
        if (idx_e < 0) idx_e = 0;
        double alpha = f_idx - (double)idx_e;
        const double* cdf_base = &PTR_D(p, P_INEL_CDF_DATA)[cdf_off];
        double xi_l = pcg_uniform(&rng);
        int sampled = cdf_n_lev - 1;
        int row_lo = idx_e       * cdf_n_lev;
        int row_hi = (idx_e + 1) * cdf_n_lev;
        #pragma unroll 1
        for (int l = 0; l < cdf_n_lev - 1; l++) {
            double F = cdf_base[row_lo + l]
                     + alpha * (cdf_base[row_hi + l] - cdf_base[row_lo + l]);
            if (xi_l <= F) { sampled = l; break; }
        }
        selected = sampled;
        Q = __ldg(&PTR_D(p, P_LEVEL_Q)[lv_off + selected]);
    } else if (n_lev > 0) {
        double lxs_sum = 0.0;
        int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[hit_nuc]);
        int n_e   = __ldg(&PTR_I(p, P_N_ENERGIES)[hit_nuc]);
        int e_idx = energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off], n_e, E);
        int lev_cap = n_lev < LEGACY_LEV_CAP ? n_lev : LEGACY_LEV_CAP;
        const double* nuc_lvl_basis =
            (const double*) __ldg(&PTR_U64(p, P_LEVEL_BASIS_PTRS)[hit_nuc]);
        const double* nuc_lvl_coeffs =
            (const double*) __ldg(&PTR_U64(p, P_LEVEL_COEFFS_PTRS)[hit_nuc]);
        #pragma unroll 1
        for (int l = 0; l < lev_cap; l++) {
            int gl = lv_off + l;
            if (E >= __ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) {
                lxs_sum += svd_reconstruct(
                    &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                    &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                    e_idx, rank);
            }
        }
        if (lxs_sum > 0.0) {
            double xi_l = pcg_uniform(&rng) * lxs_sum;
            double run = 0.0;
            selected = lev_cap - 1;
            #pragma unroll 1
            for (int l = 0; l < lev_cap; l++) {
                int gl = lv_off + l;
                double lxs = 0.0;
                if (E >= __ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                    && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) {
                    lxs = svd_reconstruct(
                        &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                        &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                        e_idx, rank);
                }
                run += lxs;
                if (xi_l < run) { selected = l; break; }
            }
            Q = __ldg(&PTR_D(p, P_LEVEL_Q)[lv_off + selected]);
        }
    }
    int sel_mt = (n_lev > 0) ? __ldg(&PTR_I(p, P_LEVEL_MT)[lv_off + selected]) : 0;
    if (sel_mt == 91) {
        double ecm_mev = E * A / ((A + 1.0) * 1e6);
        int n_inc91 = __ldg(&PTR_I(p, P_INEL91_NUC_NINC)[hit_nuc]);
        double eo_mev;
        if (n_inc91 > 0) {
            double eo_ev = sample_inel91_energy(E, &rng, p, hit_nuc);
            eo_mev = eo_ev / 1.0e6;
        } else {
            double a_p = A / 8.0;
            double eex = fmax(ecm_mev, 0.1);
            double T = sqrt(eex / a_p);
            double x1 = fmax(pcg_uniform(&rng), 1e-30);
            double x2 = fmax(pcg_uniform(&rng), 1e-30);
            eo_mev = -T * log(x1 * x2);
        }
        eo_mev = fmin(eo_mev, ecm_mev * 0.9);
        Q = -(ecm_mev - eo_mev) * 1e6;
    }
    double l_q_inel = fabs(Q);
    double e_cm = E * A / (A + 1.0);
    double e_cm_out = e_cm + Q;
    if (e_cm_out <= 0.0) {
        double mu_fb = 2.0 * pcg_uniform(&rng) - 1.0;
        double alpha = ((A - 1.0) / (A + 1.0)) * ((A - 1.0) / (A + 1.0));
        E = E * (1.0 + alpha + (1.0 - alpha) * mu_fb) / 2.0;
        if (E < 1e-11) E = 1e-11;
        double mu_lab = (A > 1.0 + 1e-10)
            ? (1.0 + A * mu_fb) / sqrt(1.0 + A * A + 2.0 * A * mu_fb)
            : sqrt(fmax(0.0, (1.0 + mu_fb) * 0.5));
        double phi = 2.0 * PI * pcg_uniform(&rng);
        rotate_direction(&dx, &dy, &dz, mu_lab, phi);
    } else {
        double mu_cm;
        if (n_lev > 0 && sel_mt != 91) {
            mu_cm = sample_level_angular(E, &rng, p, lv_off + selected, hit_nuc);
        } else {
            mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
        }
        double ap1 = A + 1.0;
        double e_n_cm = e_cm_out * A / ap1;
        double v_n_i = sqrt(2.0 * e_n_cm);
        double v_cm_s = sqrt(2.0 * E / (ap1 * ap1));
        double v2sum = v_n_i * v_n_i + v_cm_s * v_cm_s + 2.0 * v_n_i * v_cm_s * mu_cm;
        E = fmax(0.5 * v2sum, 1e-5);
        double denom = sqrt(fmax(v2sum, 1e-40));
        double mu_lab;
        if (v_n_i + v_cm_s > 1e-20) {
            mu_lab = (v_cm_s + v_n_i * mu_cm) / denom;
            mu_lab = fmax(-1.0, fmin(1.0, mu_lab));
        } else {
            mu_lab = 2.0 * pcg_uniform(&rng) - 1.0;
        }
        double phi = 2.0 * PI * pcg_uniform(&rng);
        rotate_direction(&dx, &dy, &dz, mu_lab, phi);
    }

    double l_inel_out = E;
    energy[tid] = E;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;
    atomicAdd(cnt_inelastic, 1);
    atomicAdd(e_inel_in_sum, l_inel_in);
    atomicAdd(e_inel_in_sq_sum, l_inel_in_sq);
    atomicAdd(e_inel_out_sum, l_inel_out);
    atomicAdd(q_inel_sum, l_q_inel);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 6: gr_fission_event
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_fission_event(
    Params p,
    const int* d_type_count, const int* d_type_offsets,
    const int* d_sorted_idx,
    const double* pos_x, const double* pos_y, const double* pos_z,
    double* energy,
    int* alive,
    unsigned long long* rng_state_arr,
    unsigned long long* rng_inc_arr,
    const int* d_event_hit_nuc,
    double* fis_x, double* fis_y, double* fis_z,
    double* fis_e, double* fis_w, int* fis_count, int max_fis,
    int* cnt_fis,
    double* e_fis_in_sum, double* e_fis_in_sq_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = d_type_count[EV_FISSION];
    if (idx >= count) return;
    int tid = d_sorted_idx[d_type_offsets[EV_FISSION] + idx];

    int hit_nuc = d_event_hit_nuc[tid];
    double E = energy[tid];
    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    PcgState rng; rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];

    int nb_off = __ldg(&PTR_I(p, P_NB_OFFSETS)[hit_nuc]);
    int nb_sz  = __ldg(&PTR_I(p, P_NB_SIZES)[hit_nuc]);
    double nu = (nb_sz > 0)
        ? nu_bar_lookup(E, PTR_D(p, P_NB_ENERGIES), PTR_D(p, P_NB_VALUES), nb_off, nb_sz)
        : __ldg(&PTR_D(p, P_NU_BAR_CONST)[hit_nuc]);
    int ns = (int)nu;
    if (pcg_uniform(&rng) < (nu - (double)ns)) ns++;
    for (int s = 0; s < ns; s++) {
        int fidx = atomicAdd(fis_count, 1);
        if (fidx < max_fis) {
            fis_x[fidx] = px; fis_y[fidx] = py; fis_z[fidx] = pz;
            fis_e[fidx] = sample_fission_emit_energy(E, nu, &rng, p, hit_nuc);
            fis_w[fidx] = 1.0;
        }
    }
    alive[tid] = 0;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;
    atomicAdd(cnt_fis, 1);
    atomicAdd(e_fis_in_sum, E);
    atomicAdd(e_fis_in_sq_sum, E * E);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 7: gr_multi_event  (handles (n,2n) and (n,3n))
//
// One thread per particle. Reads event_type to decide how many extra
// neutrons to bank (1 for n2n, 2 for n3n) — the only divergence is a
// small comparison, not a different code path.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void gr_multi_event(
    Params p,
    const int* d_type_count, const int* d_type_offsets,
    const int* d_sorted_idx,
    const double* pos_x, const double* pos_y, const double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    unsigned long long* rng_state_arr,
    unsigned long long* rng_inc_arr,
    const int* d_event_type, const int* d_event_hit_nuc,
    double* fis_x, double* fis_y, double* fis_z,
    double* fis_e, double* fis_w, int* fis_count, int max_fis)
{
    // Sweeps both N2N and N3N classes — their slots are adjacent in
    // d_sorted_idx (offsets[3]..offsets[5]). Each thread reads its
    // own event_type to pick the right secondary count and Q value.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = d_type_count[EV_N2N] + d_type_count[EV_N3N];
    if (idx >= count) return;
    int tid = d_sorted_idx[d_type_offsets[EV_N2N] + idx];

    int ev = d_event_type[tid];
    int hit_nuc = d_event_hit_nuc[tid];
    double E = energy[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    PcgState rng; rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];
    double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

    int n_extra = (ev == EV_N2N) ? 1 : 2;
    double Q_mult = (ev == EV_N2N) ? -E * 0.1 : -E * 0.2;

    for (int s = 0; s < n_extra; s++) {
        double temp = E / 10.0;
        double x1 = fmax(pcg_uniform(&rng), 1e-30);
        double x2 = fmax(pcg_uniform(&rng), 1e-30);
        double e_sec = fmax(fmin(-temp * log(x1 * x2), E), 1e-5);
        int fidx = atomicAdd(fis_count, 1);
        if (fidx < max_fis) {
            fis_x[fidx] = px; fis_y[fidx] = py; fis_z[fidx] = pz;
            fis_e[fidx] = e_sec; fis_w[fidx] = 1.0;
        }
    }
    {
        double e_cm = E * A / (A + 1.0);
        double e_cm_out = e_cm + Q_mult;
        if (e_cm_out <= 0.0) e_cm_out = E * 0.01;
        double mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
        double ap1 = A + 1.0;
        double e_n = e_cm_out * A / ap1;
        double vni = sqrt(2.0 * e_n);
        double vcs = sqrt(2.0 * E / (ap1 * ap1));
        double v2 = vni * vni + vcs * vcs + 2.0 * vni * vcs * mu_cm;
        E = fmax(0.5 * v2, 1e-5);
        double den = sqrt(fmax(v2, 1e-40));
        double ml = (vni + vcs > 1e-20)
            ? fmax(-1.0, fmin(1.0, (vcs + vni * mu_cm) / den))
            : 2.0 * pcg_uniform(&rng) - 1.0;
        double phi = 2.0 * PI * pcg_uniform(&rng);
        rotate_direction(&dx, &dy, &dz, ml, phi);
    }
    energy[tid] = E;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;
    (void)px; (void)py; (void)pz;
}

#endif // TRANSPORT_EVENT_BASED_CU
