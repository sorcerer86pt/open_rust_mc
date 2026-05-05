// Recursive-geometry device functions for the GPU port (task #19).
//
// Mirrors the CPU primitives in `geometry/ray.rs`:
//   * find_cell_recursive  — descend root → universe / lattice → leaf
//   * trace_step_recursive — distance to the next event (surface or
//                            lattice grid line) along the particle ray
//
// Geometry is uploaded as a fixed set of SoA tables (one buffer per
// "field"), described by the `RecGeomTables` struct further down. The
// region tree of every cell is encoded as a postfix opcode array so
// the device just runs a tiny stack machine — no recursion, no
// pointers, GPU-friendly.
//
// All math is f64 to match the CPU side bit-for-bit on the depth-1
// fast-path (no rotation, no nested universe). Once nested-universe
// pushes appear, sub-ULP drift between CPU and GPU is inevitable
// because the order-of-operations in the rotation cascade isn't
// guaranteed identical, but the parity test only checks the deepest
// cell index, which is order-invariant.

#ifndef GEOM_RECURSIVE_CU
#define GEOM_RECURSIVE_CU

// NVRTC ships device intrinsics (sqrt, floor, etc.) without needing
// <math.h>; including the host header makes NVRTC fail to resolve.

// ── Surface, region, fill type tags (must match Rust side) ──────────

#define GR_SURF_PLANE_X 0
#define GR_SURF_PLANE_Y 1
#define GR_SURF_PLANE_Z 2
#define GR_SURF_SPHERE  3
#define GR_SURF_CYL_Z   4
#define GR_SURF_CYL_X   5
#define GR_SURF_CYL_Y   6
#define GR_SURF_PLANE_GENERAL 7

#define GR_BC_TRANSMISSION 0
#define GR_BC_VACUUM       1
#define GR_BC_REFLECTIVE   2

#define GR_REGION_HALFSPACE_POS 0  // operand: surface index
#define GR_REGION_HALFSPACE_NEG 1
#define GR_REGION_INTERSECTION  2  // pop 2, push (a && b)
#define GR_REGION_UNION         3  // pop 2, push (a || b)
#define GR_REGION_COMPLEMENT    4  // pop 1, push (!a)

#define GR_FILL_MATERIAL 0
#define GR_FILL_VOID     1
#define GR_FILL_UNIVERSE 2
#define GR_FILL_LATTICE  3

#define GR_MAX_DEPTH 4

// ── SoA geometry tables ─────────────────────────────────────────────
//
// Surfaces:      surf_type[ns], surf_params[ns*8], surf_bc[ns]
// Cells:         cell_region_off[nc], cell_region_len[nc],
//                cell_fill_type[nc], cell_fill_data[nc],
//                cell_aabb_min[nc*3], cell_aabb_max[nc*3]
// Region tree:   region_op[total], region_arg[total]
// Universes:     univ_cells_off[nu], univ_cells_len[nu],
//                univ_surfaces_off[nu], univ_surfaces_len[nu]
// Cell-index list (one slab per universe):
//                univ_cell_indices[total]
// Surface-index list (one slab per universe):
//                univ_surface_indices[total]
// Lattices:      lat_origin[nl*3], lat_pitch[nl*3], lat_shape[nl*3],
//                lat_universes_off[nl], lat_universes[total]
//
// All SoA pointers are passed via the shared u64 params table the
// rest of the kernel already uses, with new P_GR_* slot names. The
// Rust side fills them in `gpu_recursive::upload`.

struct GrGeometry {
    // Surfaces
    const int*    surf_type;      // [ns]
    const double* surf_params;    // [ns*8]
    const int*    surf_bc;        // [ns]
    int           n_surfaces;
    // Cells
    const int*    cell_region_off;
    const int*    cell_region_len;
    const int*    cell_fill_type;
    const int*    cell_fill_data;
    const double* cell_aabb_min;  // [nc*3]
    const double* cell_aabb_max;  // [nc*3]
    int           n_cells;
    // Region opcodes
    const int*    region_op;
    const int*    region_arg;
    // Universes
    const int*    univ_cells_off;
    const int*    univ_cells_len;
    const int*    univ_surfaces_off;
    const int*    univ_surfaces_len;
    int           n_universes;
    int           root_universe;
    // Per-universe cell index list (flattened across all universes)
    const int*    univ_cell_indices;
    // Per-universe surface index list (flattened)
    const int*    univ_surface_indices;
    // Lattices
    const double* lat_origin;     // [nl*3]
    const double* lat_pitch;      // [nl*3]
    const int*    lat_shape;      // [nl*3]
    const int*    lat_universes_off;
    const int*    lat_universes;
    int           n_lattices;
    // Eval scratchpad: [n_surfaces] doubles per thread (the caller
    // owns this buffer and zeroes it as needed).
    double*       evals;
};

struct GrCoord {
    int  universe;
    int  cell_idx;
    int  has_lattice;
    int  lattice_id;
    int  lat_ix;
    int  lat_iy;
    int  lat_iz;
    double offx, offy, offz;
};

// ── Surface evaluation ──────────────────────────────────────────────

__device__ __forceinline__ double gr_surf_eval(
    const GrGeometry* g, int s_idx,
    double x, double y, double z)
{
    int t = g->surf_type[s_idx];
    const double* p = g->surf_params + s_idx * 8;
    switch (t) {
        case GR_SURF_PLANE_X:        return x - p[0];
        case GR_SURF_PLANE_Y:        return y - p[0];
        case GR_SURF_PLANE_Z:        return z - p[0];
        case GR_SURF_SPHERE: {
            double dx = x - p[0], dy = y - p[1], dz = z - p[2];
            return dx*dx + dy*dy + dz*dz - p[3]*p[3];
        }
        case GR_SURF_CYL_Z: {
            double dx = x - p[0], dy = y - p[1];
            return dx*dx + dy*dy - p[2]*p[2];
        }
        case GR_SURF_CYL_X: {
            double dy = y - p[0], dz = z - p[1];
            return dy*dy + dz*dz - p[2]*p[2];
        }
        case GR_SURF_CYL_Y: {
            double dx = x - p[0], dz = z - p[1];
            return dx*dx + dz*dz - p[2]*p[2];
        }
        case GR_SURF_PLANE_GENERAL:
            return p[0]*x + p[1]*y + p[2]*z - p[3];
        default: return 1e300;
    }
}

// ── Region eval — postfix stack machine ─────────────────────────────

__device__ bool gr_cell_contains(
    const GrGeometry* g, int cell_idx, const double* evals)
{
    int off = g->cell_region_off[cell_idx];
    int len = g->cell_region_len[cell_idx];
    bool stack[16];  // ample for any sane region tree
    int sp = 0;
    for (int i = 0; i < len; ++i) {
        int op  = g->region_op[off + i];
        int arg = g->region_arg[off + i];
        switch (op) {
            case GR_REGION_HALFSPACE_POS: stack[sp++] = evals[arg] > 0.0; break;
            case GR_REGION_HALFSPACE_NEG: stack[sp++] = evals[arg] < 0.0; break;
            case GR_REGION_INTERSECTION: {
                bool b = stack[--sp];
                bool a = stack[--sp];
                stack[sp++] = a && b;
                break;
            }
            case GR_REGION_UNION: {
                bool b = stack[--sp];
                bool a = stack[--sp];
                stack[sp++] = a || b;
                break;
            }
            case GR_REGION_COMPLEMENT: {
                bool a = stack[--sp];
                stack[sp++] = !a;
                break;
            }
        }
    }
    return sp == 1 ? stack[0] : false;
}

__device__ __forceinline__ bool gr_cell_aabb_contains(
    const GrGeometry* g, int cell_idx,
    double x, double y, double z)
{
    const double* lo = g->cell_aabb_min + cell_idx * 3;
    const double* hi = g->cell_aabb_max + cell_idx * 3;
    return x >= lo[0] && x <= hi[0]
        && y >= lo[1] && y <= hi[1]
        && z >= lo[2] && z <= hi[2];
}

// ── Lattice element resolution ──────────────────────────────────────

__device__ __forceinline__ bool gr_lattice_find_element(
    const GrGeometry* g, int lat_id,
    double x, double y, double z,
    int* out_ix, int* out_iy, int* out_iz)
{
    const double* org = g->lat_origin + lat_id * 3;
    const double* pit = g->lat_pitch  + lat_id * 3;
    const int*    sh  = g->lat_shape  + lat_id * 3;
    double rx = x - org[0], ry = y - org[1], rz = z - org[2];
    int ix = (int)floor(rx / pit[0]);
    int iy = (int)floor(ry / pit[1]);
    int iz = (int)floor(rz / pit[2]);
    if (ix < 0 || iy < 0 || iz < 0) return false;
    if (ix >= sh[0] || iy >= sh[1] || iz >= sh[2]) return false;
    *out_ix = ix; *out_iy = iy; *out_iz = iz;
    return true;
}

__device__ __forceinline__ int gr_lattice_universe_at(
    const GrGeometry* g, int lat_id, int ix, int iy, int iz)
{
    const int* sh = g->lat_shape + lat_id * 3;
    int slab = sh[0] * sh[1];
    int row  = sh[0];
    int linear = iz * slab + iy * row + ix;
    int off = g->lat_universes_off[lat_id];
    return g->lat_universes[off + linear];
}

// ── Recursive cell-find ─────────────────────────────────────────────
//
// Returns the depth of the resolved stack (1..GR_MAX_DEPTH) on success
// and 0 on leakage. `out_stack` must hold GR_MAX_DEPTH GrCoord entries.

__device__ int gr_find_cell(
    const GrGeometry* g,
    double world_x, double world_y, double world_z,
    GrCoord* out_stack)
{
    int    depth = 0;
    int    current_universe = g->root_universe;
    double next_off_x = 0.0, next_off_y = 0.0, next_off_z = 0.0;
    int    next_has_lat = 0, next_lat_id = 0,
           next_lat_ix = 0, next_lat_iy = 0, next_lat_iz = 0;
    double local_x = world_x, local_y = world_y, local_z = world_z;
    double* evals = g->evals;

    while (depth < GR_MAX_DEPTH) {
        local_x -= next_off_x;
        local_y -= next_off_y;
        local_z -= next_off_z;

        // Refresh only the universe-relevant surface evaluations.
        int s_off = g->univ_surfaces_off[current_universe];
        int s_len = g->univ_surfaces_len[current_universe];
        for (int i = 0; i < s_len; ++i) {
            int s_idx = g->univ_surface_indices[s_off + i];
            evals[s_idx] = gr_surf_eval(g, s_idx, local_x, local_y, local_z);
        }

        // Linear scan over this universe's cells.
        int c_off = g->univ_cells_off[current_universe];
        int c_len = g->univ_cells_len[current_universe];
        int chosen = -1;
        for (int i = 0; i < c_len; ++i) {
            int c_idx = g->univ_cell_indices[c_off + i];
            if (!gr_cell_aabb_contains(g, c_idx, local_x, local_y, local_z)) continue;
            if (gr_cell_contains(g, c_idx, evals)) { chosen = c_idx; break; }
        }
        if (chosen < 0) return 0;  // leakage

        // Push this frame.
        GrCoord* fr = out_stack + depth;
        fr->universe   = current_universe;
        fr->cell_idx   = chosen;
        fr->has_lattice = next_has_lat;
        fr->lattice_id = next_lat_id;
        fr->lat_ix = next_lat_ix; fr->lat_iy = next_lat_iy; fr->lat_iz = next_lat_iz;
        fr->offx = next_off_x; fr->offy = next_off_y; fr->offz = next_off_z;
        depth++;

        int ft = g->cell_fill_type[chosen];
        int fd = g->cell_fill_data[chosen];
        if (ft == GR_FILL_MATERIAL || ft == GR_FILL_VOID) {
            return depth;
        }
        if (ft == GR_FILL_UNIVERSE) {
            current_universe = fd;
            next_off_x = 0.0; next_off_y = 0.0; next_off_z = 0.0;
            next_has_lat = 0;
            continue;
        }
        if (ft == GR_FILL_LATTICE) {
            int lat_id = fd;
            int ix, iy, iz;
            if (!gr_lattice_find_element(g, lat_id, local_x, local_y, local_z,
                                         &ix, &iy, &iz)) {
                return 0;  // off the lattice
            }
            current_universe = gr_lattice_universe_at(g, lat_id, ix, iy, iz);
            const double* org = g->lat_origin + lat_id * 3;
            const double* pit = g->lat_pitch  + lat_id * 3;
            // local_in_element = local_pos − origin − idx*pitch
            // element_offset   = local_pos − local_in_element
            //                  = origin + idx*pitch
            next_off_x = org[0] + ix * pit[0];
            next_off_y = org[1] + iy * pit[1];
            next_off_z = org[2] + iz * pit[2];
            next_has_lat = 1; next_lat_id = lat_id;
            next_lat_ix = ix; next_lat_iy = iy; next_lat_iz = iz;
            continue;
        }
        // unknown fill type
        return 0;
    }
    return 0;
}

#endif // GEOM_RECURSIVE_CU
