// Stage 3a of task #22: simplest end-to-end recursive transport on GPU.
//
// Constant cross-sections per material (sigma_t, sigma_a, sigma_f, nu).
// One kernel = one batch of histories. Per-particle:
//   - find_cell at birth.
//   - Loop until absorption / leakage / step cap:
//       mat = effective_material(stack)
//       d_coll = exp(sigma_t[mat])
//       hit = gr_trace_step(stack, pos, dir)
//       if d_coll < hit.distance:
//           collide → sample reaction; absorb or scatter
//           on fission, atomicAdd a fission site
//       else:
//           cross → reflective inverts axis-aligned dir; transmission
//                   adopts next_stack; vacuum kills
//
// Validates that recursive geometry composes correctly with collision
// sampling and fission banking on GPU. Full SVD/Table/WMP physics
// integration is a follow-up; this stage proves the plumbing.

#include "geom_recursive.cu"

// Material-constant XS layout (host uploads `n_materials * 4` doubles):
//   [sigma_t_0, sigma_a_0, sigma_f_0, nu_0,
//    sigma_t_1, sigma_a_1, sigma_f_1, nu_1, ...]

// PCG-XSH-RR 64/32 — same as the rest of the codebase.
struct ConstPcg { unsigned long long state; unsigned long long inc; };

__device__ unsigned int pcg_next_const(ConstPcg* rng) {
    unsigned long long old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

__device__ double pcg_uniform_const(ConstPcg* rng) {
    unsigned long long a = ((unsigned long long)pcg_next_const(rng)) >> 5;
    unsigned long long b = ((unsigned long long)pcg_next_const(rng)) >> 6;
    return (double)(a * 67108864ULL + b) * (1.0 / 9007199254740992.0);
}

__device__ double pcg_exp_const(ConstPcg* rng, double rate) {
    return -log(pcg_uniform_const(rng)) / rate;
}

__device__ void pcg_isotropic_const(ConstPcg* rng,
    double* out_dx, double* out_dy, double* out_dz)
{
    double mu = 2.0 * pcg_uniform_const(rng) - 1.0;
    double phi = 2.0 * 3.141592653589793 * pcg_uniform_const(rng);
    double s = sqrt(1.0 - mu * mu);
    *out_dx = s * cos(phi);
    *out_dy = s * sin(phi);
    *out_dz = mu;
}

// effective_material — applies a lattice override if present, else
// reads cell.fill. Returns -1 for void.
__device__ int gr_effective_material(
    const GrGeometry* g, const GrCoord* stack, int depth,
    // material-override tables (mirrors RectLattice.material_overrides
    // on the CPU side; flat layout described in the host code):
    const int* lat_override_off,         // [n_lattices]: -1 = no overrides
    const int* lat_override_count,       // [n_lattices]
    const int* override_lat_idx,         // [total]: linear element idx
    const int* override_cell_idx,        // [total]: global cell idx
    const int* override_mat,             // [total]: overriding material
    int n_lattices)
{
    if (depth <= 0) return -1;
    const GrCoord* d = &stack[depth - 1];
    int cell = d->cell_idx;

    if (d->has_lattice && d->lattice_id < n_lattices) {
        int lid = d->lattice_id;
        int off = lat_override_off[lid];
        if (off >= 0) {
            int cnt = lat_override_count[lid];
            const int* sh = g->lat_shape + lid * 3;
            int lin = d->lat_iz * sh[0] * sh[1] + d->lat_iy * sh[0] + d->lat_ix;
            for (int i = 0; i < cnt; ++i) {
                if (override_lat_idx[off + i] == lin
                    && override_cell_idx[off + i] == cell) {
                    return override_mat[off + i];
                }
            }
        }
    }

    int ft = g->cell_fill_type[cell];
    int fd = g->cell_fill_data[cell];
    if (ft == GR_FILL_MATERIAL) return fd;
    return -1;  // void or non-leaf (shouldn't appear)
}

// Per-particle state lives in registers. The kernel reads/writes SoA
// arrays at entry/exit. Each thread runs many transport events in
// the kernel body (persistent within one batch).

extern "C" __global__ void const_xs_transport_persistent(
    // Particle state (SoA, mutable)
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    double* __restrict__ dir_x,
    double* __restrict__ dir_y,
    double* __restrict__ dir_z,
    int*    __restrict__ alive,
    unsigned long long* __restrict__ rng_state,
    unsigned long long* __restrict__ rng_inc,
    int n_particles,
    int max_events_per_history,
    // Per-material constant XS: layout [sigma_t, sigma_a, sigma_f, nu]
    const double* mat_xs,
    int n_materials,
    // Geometry tables (same layout as find_cell_batch / trace_step_batch)
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
    // Hex lattices
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    int n_hex_lattices,
    // Material-override tables (sparse, optional). When all -1, no overrides.
    const int* lat_override_off,
    const int* lat_override_count,
    const int* override_lat_idx,
    const int* override_cell_idx,
    const int* override_mat,
    // Per-thread evals scratch (n_surfaces * n_particles doubles)
    double* evals_scratch,
    // Fission bank — preallocated, atomically appended
    double* __restrict__ fis_x,
    double* __restrict__ fis_y,
    double* __restrict__ fis_z,
    int*    __restrict__ fis_count,
    int     fis_capacity,
    // Per-batch counters (one slot, atomic)
    unsigned long long* __restrict__ cnt_collisions,
    unsigned long long* __restrict__ cnt_absorptions,
    unsigned long long* __restrict__ cnt_fissions,
    unsigned long long* __restrict__ cnt_leakage,
    unsigned long long* __restrict__ cnt_surf_xings)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (!alive[tid]) return;

    GrGeometry g;
    g.surf_type = surf_type; g.surf_params = surf_params; g.surf_bc = surf_bc; g.n_surfaces = n_surfaces;
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
    g.evals = evals_scratch + tid * n_surfaces;

    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    ConstPcg rng; rng.state = rng_state[tid]; rng.inc = rng_inc[tid];

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, px, py, pz, stack);
    if (depth == 0) {
        alive[tid] = 0;
        atomicAdd(cnt_leakage, 1ULL);
        return;
    }

    unsigned long long lc_coll = 0, lc_abs = 0, lc_fis = 0, lc_surf = 0;
    int events = 0;
    int local_alive = 1;

    while (local_alive && events < max_events_per_history) {
        events++;
        int mat = gr_effective_material(
            &g, stack, depth,
            lat_override_off, lat_override_count,
            override_lat_idx, override_cell_idx, override_mat,
            n_lattices);

        if (mat < 0) {
            // void — free-stream to next surface
            double dist; int surf_idx; int bc; int next_depth;
            GrCoord next_stack[GR_MAX_DEPTH];
            if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                               &dist, &surf_idx, &bc, next_stack, &next_depth)) {
                local_alive = 0;
                atomicAdd(cnt_leakage, 1ULL);
                break;
            }
            lc_surf++;
            if (bc == GR_BC_VACUUM) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                local_alive = 0;
                atomicAdd(cnt_leakage, 1ULL);
                break;
            }
            if (bc == GR_BC_REFLECTIVE) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
                if (t == GR_SURF_PLANE_X) dx = -dx;
                else if (t == GR_SURF_PLANE_Y) dy = -dy;
                else if (t == GR_SURF_PLANE_Z) dz = -dz;
                continue;
            }
            const double NUDGE = 1e-10;
            px += dx * (dist + NUDGE); py += dy * (dist + NUDGE); pz += dz * (dist + NUDGE);
            if (next_depth == 0) {
                local_alive = 0;
                atomicAdd(cnt_leakage, 1ULL);
                break;
            }
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
            continue;
        }

        const double sigma_t = mat_xs[mat * 4 + 0];
        const double sigma_a = mat_xs[mat * 4 + 1];
        const double sigma_f = mat_xs[mat * 4 + 2];
        const double nu_bar  = mat_xs[mat * 4 + 3];

        if (sigma_t <= 0.0) {
            local_alive = 0;
            atomicAdd(cnt_leakage, 1ULL);
            break;
        }
        double d_collide = pcg_exp_const(&rng, sigma_t);

        double dist; int surf_idx; int bc; int next_depth;
        GrCoord next_stack[GR_MAX_DEPTH];
        if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                           &dist, &surf_idx, &bc, next_stack, &next_depth)) {
            local_alive = 0;
            atomicAdd(cnt_leakage, 1ULL);
            break;
        }

        if (d_collide < dist) {
            // collision
            px += dx * d_collide; py += dy * d_collide; pz += dz * d_collide;
            lc_coll++;
            double xi_react = pcg_uniform_const(&rng) * sigma_t;
            if (xi_react < sigma_a) {
                // absorption
                lc_abs++;
                if (sigma_a > 0.0) {
                    double pf = sigma_f / sigma_a;
                    if (pcg_uniform_const(&rng) < pf) {
                        // fission — sample integer multiplicity
                        double xi = pcg_uniform_const(&rng);
                        int n_fission = (int)(nu_bar + xi);
                        if (n_fission > 0) {
                            int slot = atomicAdd(fis_count, n_fission);
                            for (int k = 0; k < n_fission; ++k) {
                                int s = slot + k;
                                if (s < fis_capacity) {
                                    fis_x[s] = px;
                                    fis_y[s] = py;
                                    fis_z[s] = pz;
                                }
                            }
                            lc_fis += (unsigned long long)n_fission;
                        }
                    }
                }
                local_alive = 0;
                break;
            } else {
                // scatter
                pcg_isotropic_const(&rng, &dx, &dy, &dz);
                continue;
            }
        }

        // crossing
        lc_surf++;
        if (bc == GR_BC_VACUUM) {
            px += dx * dist; py += dy * dist; pz += dz * dist;
            local_alive = 0;
            atomicAdd(cnt_leakage, 1ULL);
            break;
        }
        if (bc == GR_BC_REFLECTIVE) {
            px += dx * dist; py += dy * dist; pz += dz * dist;
            int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
            if (t == GR_SURF_PLANE_X) dx = -dx;
            else if (t == GR_SURF_PLANE_Y) dy = -dy;
            else if (t == GR_SURF_PLANE_Z) dz = -dz;
            continue;
        }
        const double NUDGE = 1e-10;
        px += dx * (dist + NUDGE); py += dy * (dist + NUDGE); pz += dz * (dist + NUDGE);
        if (next_depth == 0) {
            local_alive = 0;
            atomicAdd(cnt_leakage, 1ULL);
            break;
        }
        for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
        depth = next_depth;
    }

    pos_x[tid] = px; pos_y[tid] = py; pos_z[tid] = pz;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    alive[tid] = local_alive;
    rng_state[tid] = rng.state; rng_inc[tid] = rng.inc;

    atomicAdd(cnt_collisions, lc_coll);
    atomicAdd(cnt_absorptions, lc_abs);
    atomicAdd(cnt_fissions, lc_fis);
    atomicAdd(cnt_surf_xings, lc_surf);
}
