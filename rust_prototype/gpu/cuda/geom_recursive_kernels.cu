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
