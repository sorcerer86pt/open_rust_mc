// SPDX-License-Identifier: MIT
// Launchable kernel that runs `gr_find_cell` over a batch of points.
// Used by the GPU<->CPU parity test for the recursive geometry port.

#include "geom_recursive.cu"

extern "C" __global__ void find_cell_batch(
    const double* xs, const double* ys, const double* zs, int n_points,
    // surfaces
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    // cells
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    // region tree
    const int* region_op, const int* region_arg,
    // universes
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    // lattices
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    // hex lattices
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    // scratch + output
    double* evals_scratch,
    int* out_deepest_cell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

    GrGeometry g;
    g.surf_type = surf_type;
    g.surf_params = surf_params;
    g.surf_bc = surf_bc;
    g.n_surfaces = n_surfaces;

    g.cell_region_off = cell_region_off;
    g.cell_region_len = cell_region_len;
    g.cell_fill_type = cell_fill_type;
    g.cell_fill_data = cell_fill_data;
    g.cell_aabb_min = cell_aabb_min;
    g.cell_aabb_max = cell_aabb_max;
    g.n_cells = 0;  // unused on the device — we walk by index, not range

    g.region_op = region_op;
    g.region_arg = region_arg;

    g.univ_cells_off = univ_cells_off;
    g.univ_cells_len = univ_cells_len;
    g.univ_surfaces_off = univ_surfaces_off;
    g.univ_surfaces_len = univ_surfaces_len;
    g.univ_cell_indices = univ_cell_indices;
    g.univ_surface_indices = univ_surface_indices;
    g.n_universes = 0;
    g.root_universe = root_universe;

    g.lat_origin = lat_origin;
    g.lat_pitch = lat_pitch;
    g.lat_shape = lat_shape;
    g.lat_universes_off = lat_universes_off;
    g.lat_universes = lat_universes;
    g.n_lattices = 0;
    g.hex_center = hex_center;
    g.hex_pitch_xy = hex_pitch_xy;
    g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings;
    g.hex_n_axial = hex_n_axial;
    g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off;
    g.hex_universes = hex_universes;
    g.n_hex_lattices = 0;

    // Per-thread evals scratch.
    g.evals = evals_scratch + tid * n_surfaces;

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, xs[tid], ys[tid], zs[tid], stack);
    out_deepest_cell[tid] = (depth > 0) ? stack[depth - 1].cell_idx : -1;
}

// trace_step_batch — runs `find_cell + trace_step_recursive` per thread
// and reports the event distance, surface index, BC, and the deepest
// cell index of the next stack. Used by the parity test for the GPU
// trace step.
extern "C" __global__ void trace_step_batch(
    const double* xs, const double* ys, const double* zs,
    const double* dxs, const double* dys, const double* dzs,
    int n_points,
    // surfaces
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    // cells
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    // region tree
    const int* region_op, const int* region_arg,
    // universes
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    // lattices
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    // hex lattices
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    // scratch + outputs
    double* evals_scratch,
    double* out_distance, int* out_surface_idx, int* out_bc,
    int* out_next_deepest_cell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

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
    g.n_lattices = 0;
    g.hex_center = hex_center; g.hex_pitch_xy = hex_pitch_xy;
    g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings; g.hex_n_axial = hex_n_axial;
    g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off; g.hex_universes = hex_universes;
    g.n_hex_lattices = 0;
    g.evals = evals_scratch + tid * n_surfaces;

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, xs[tid], ys[tid], zs[tid], stack);
    if (depth == 0) {
        out_distance[tid] = 1e300;
        out_surface_idx[tid] = -1;
        out_bc[tid] = GR_BC_VACUUM;
        out_next_deepest_cell[tid] = -1;
        return;
    }

    double dist; int surf; int bc; int next_depth;
    GrCoord next_stack[GR_MAX_DEPTH];
    gr_trace_step(
        &g, stack, depth,
        xs[tid], ys[tid], zs[tid],
        dxs[tid], dys[tid], dzs[tid],
        &dist, &surf, &bc,
        next_stack, &next_depth);

    out_distance[tid] = dist;
    out_surface_idx[tid] = surf;
    out_bc[tid] = bc;
    out_next_deepest_cell[tid] = (next_depth > 0)
        ? next_stack[next_depth - 1].cell_idx : -1;
}

// multi_step_walk — runs a deterministic K-step walk per particle,
// purely a geometry traversal (no collision, no fission, no XS).
// On every step the kernel computes the next event distance via
// gr_trace_step, advances by (dist + nudge), and on a Reflective
// boundary inverts the corresponding direction component (axis-
// aligned planes only — same simplification as the parity test).
// On Vacuum / leakage the particle stops, freezing the step count.
// Final position + step count is emitted per particle for direct
// CPU/GPU comparison.
extern "C" __global__ void multi_step_walk(
    const double* xs0, const double* ys0, const double* zs0,
    const double* dxs0, const double* dys0, const double* dzs0,
    int n_points, int max_steps,
    // surfaces
    const int* surf_type, const double* surf_params, const int* surf_bc,
    int n_surfaces,
    // cells
    const int* cell_region_off, const int* cell_region_len,
    const int* cell_fill_type, const int* cell_fill_data,
    const double* cell_aabb_min, const double* cell_aabb_max,
    // region tree
    const int* region_op, const int* region_arg,
    // universes
    const int* univ_cells_off, const int* univ_cells_len,
    const int* univ_surfaces_off, const int* univ_surfaces_len,
    const int* univ_cell_indices, const int* univ_surface_indices,
    int root_universe,
    // lattices
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    // hex lattices
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    // scratch + outputs
    double* evals_scratch,
    double* out_x, double* out_y, double* out_z,
    int* out_steps, int* out_final_cell)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

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
    g.n_lattices = 0;
    g.hex_center = hex_center; g.hex_pitch_xy = hex_pitch_xy;
    g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings; g.hex_n_axial = hex_n_axial;
    g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off; g.hex_universes = hex_universes;
    g.n_hex_lattices = 0;
    g.evals = evals_scratch + tid * n_surfaces;

    double px = xs0[tid], py = ys0[tid], pz = zs0[tid];
    double dx = dxs0[tid], dy = dys0[tid], dz = dzs0[tid];

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, px, py, pz, stack);
    int steps = 0;
    int final_cell = (depth > 0) ? stack[depth - 1].cell_idx : -1;

    if (depth == 0) {
        out_x[tid] = px; out_y[tid] = py; out_z[tid] = pz;
        out_steps[tid] = 0;
        out_final_cell[tid] = -1;
        return;
    }

    for (int s = 0; s < max_steps; ++s) {
        double dist; int surf; int bc; int next_depth;
        GrCoord next_stack[GR_MAX_DEPTH];
        if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                           &dist, &surf, &bc, next_stack, &next_depth)) {
            break;
        }
        if (bc == GR_BC_VACUUM) {
            // leak — advance to surface, stop
            px += dx * dist; py += dy * dist; pz += dz * dist;
            steps++;
            final_cell = -1;
            break;
        }
        if (bc == GR_BC_REFLECTIVE) {
            // Reflect about the surface normal. Axis-aligned planes flip
            // a single direction component; arbitrary-orientation
            // `GR_SURF_PLANE_GENERAL` (e.g. hex-lattice outer faces) use
            // d' = d - 2 (d·n) n with n stored in the surface params.
            px += dx * dist; py += dy * dist; pz += dz * dist;
            int t = (surf >= 0) ? surf_type[surf] : -1;
            const double* sp = (surf >= 0) ? surf_params + surf * 8 : nullptr;
            gr_reflect_direction(t, sp, &dx, &dy, &dz);
            steps++;
            // stack unchanged on reflection
            continue;
        }
        // Transmission — advance with nudge and adopt the new stack.
        const double NUDGE = 1e-10;
        px += dx * (dist + NUDGE); py += dy * (dist + NUDGE); pz += dz * (dist + NUDGE);
        if (next_depth == 0) {
            steps++;
            final_cell = -1;
            break;
        }
        for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
        depth = next_depth;
        final_cell = stack[depth - 1].cell_idx;
        steps++;
    }

    out_x[tid] = px; out_y[tid] = py; out_z[tid] = pz;
    out_steps[tid] = steps;
    out_final_cell[tid] = final_cell;
}
