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

#define GR_FILL_MATERIAL    0
#define GR_FILL_VOID        1
#define GR_FILL_UNIVERSE    2
#define GR_FILL_LATTICE     3
#define GR_FILL_HEX_LATTICE 4

// Hex orientation flags — match Rust `HexOrientation` discriminants.
#define GR_HEX_ORIENT_Y 0  // flat-top (side midpoints at 30° from +x)
#define GR_HEX_ORIENT_X 1  // pointy-top (side midpoints at 0° from +x)

// GrCoord.has_lattice values: 0 = none, 1 = rect, 2 = hex.
// In the hex case, (lat_ix, lat_iy, lat_iz) encode (q, r, z_layer).

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
    // Hex lattices (parallel SoA — `CellFill::HexLattice(h)` indexes
    // these arrays; rect lattices live in lat_*).
    const double* hex_center;     // [nh*3]
    const double* hex_pitch_xy;   // [nh]
    const double* hex_pitch_z;    // [nh]
    const int*    hex_n_rings;    // [nh]
    const int*    hex_n_axial;    // [nh]
    const int*    hex_orientation; // [nh]
    const int*    hex_universes_off; // [nh]
    const int*    hex_universes;  // flattened (2*N+1)² × n_axial per lattice
    int           n_hex_lattices;
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

// ── Hex lattice math (cube-coord round) ─────────────────────────────
//
// Mirrors `HexLattice::find_element` / `local_position` / `universe_at`
// / `distance_to_grid` from `src/geometry/lattice.rs`. The math is
// closed-form arithmetic — no iteration — so the device port is a
// straight transliteration of the CPU code.

__device__ __forceinline__ double gr_hex_side_length(double pitch_xy) {
    // For both orientations, centre-to-centre = sqrt(3) * side.
    return pitch_xy * 0.5773502691896258;  // 1/sqrt(3)
}

// Cartesian → axial fractional coords.
__device__ __forceinline__ void gr_hex_cart_to_axial_frac(
    double x, double y, double s, int orientation,
    double* out_qf, double* out_rf)
{
    if (orientation == GR_HEX_ORIENT_Y) {
        // Flat-top inverse:
        //   q = (2/3 * x) / s
        //   r = (-x/3 + sqrt(3)/3 * y) / s
        *out_qf = (2.0 / 3.0 * x) / s;
        *out_rf = (-x / 3.0 + (1.7320508075688772 / 3.0) * y) / s;
    } else {
        // Pointy-top inverse:
        //   q = (sqrt(3)/3 * x - y/3) / s
        //   r = (2/3 * y) / s
        *out_qf = ((1.7320508075688772 / 3.0) * x - y / 3.0) / s;
        *out_rf = (2.0 / 3.0 * y) / s;
    }
}

// Cube-coord rounding: pick the integer triple (x, y, z) with x+y+z=0
// minimising |frac - rounded| sum.
__device__ __forceinline__ void gr_hex_cube_round(
    double qf, double rf,
    int* out_q, int* out_r)
{
    double xf = qf;
    double zf = rf;
    double yf = -xf - zf;
    double x = round(xf);
    double y = round(yf);
    double z = round(zf);
    double xd = fabs(x - xf);
    double yd = fabs(y - yf);
    double zd = fabs(z - zf);
    if (xd > yd && xd > zd) {
        x = -y - z;
    } else if (yd > zd) {
        // y = -x - z;  // not needed — q and r come from x and z
    } else {
        z = -x - y;
    }
    *out_q = (int)x;
    *out_r = (int)z;
}

__device__ __forceinline__ bool gr_hex_find_element(
    const GrGeometry* g, int hex_id,
    double x, double y, double z,
    int* out_q, int* out_r, int* out_zi)
{
    const double* ctr = g->hex_center + hex_id * 3;
    double pitch_xy = g->hex_pitch_xy[hex_id];
    double pitch_z = g->hex_pitch_z[hex_id];
    int n_rings = g->hex_n_rings[hex_id];
    int n_axial = g->hex_n_axial[hex_id];
    int orient = g->hex_orientation[hex_id];

    double px = x - ctr[0];
    double py = y - ctr[1];
    double pz = z - ctr[2];
    double s = gr_hex_side_length(pitch_xy);

    double qf, rf;
    gr_hex_cart_to_axial_frac(px, py, s, orient, &qf, &rf);

    int q, r;
    gr_hex_cube_round(qf, rf, &q, &r);

    // Bounds check via cube ring radius.
    int cube_s = -q - r;
    int aq = q < 0 ? -q : q;
    int ar = r < 0 ? -r : r;
    int as_ = cube_s < 0 ? -cube_s : cube_s;
    int ring = aq;
    if (ar > ring) ring = ar;
    if (as_ > ring) ring = as_;
    if (ring > n_rings) return false;

    // z layer: lattice z range is [-n_axial/2, +n_axial/2) * pitch_z.
    double dz = pz / pitch_z + ((double)n_axial) * 0.5;
    int zi = (int)floor(dz);
    if (zi < 0 || zi >= n_axial) return false;

    *out_q = q;
    *out_r = r;
    *out_zi = zi;
    return true;
}

// Cartesian centre of element (q, r) in lattice frame (relative to
// hex_center).
__device__ __forceinline__ void gr_hex_element_center_local(
    double pitch_xy, int orientation, int q, int r,
    double* out_x, double* out_y)
{
    double s = gr_hex_side_length(pitch_xy);
    double qf = (double)q;
    double rf = (double)r;
    if (orientation == GR_HEX_ORIENT_Y) {
        // x = 1.5 * s * q
        // y = sqrt(3) * s * (r + q/2)
        *out_x = 1.5 * s * qf;
        *out_y = 1.7320508075688772 * s * (rf + qf * 0.5);
    } else {
        // x = sqrt(3) * s * (q + r/2)
        // y = 1.5 * s * r
        *out_x = 1.7320508075688772 * s * (qf + rf * 0.5);
        *out_y = 1.5 * s * rf;
    }
}

__device__ __forceinline__ int gr_hex_universe_at(
    const GrGeometry* g, int hex_id, int q, int r, int zi)
{
    int n_rings = g->hex_n_rings[hex_id];
    int stride = 2 * n_rings + 1;
    int qi = q + n_rings;
    int ri = r + n_rings;
    int slab = stride * stride;
    int linear = zi * slab + ri * stride + qi;
    int off = g->hex_universes_off[hex_id];
    return g->hex_universes[off + linear];
}

// Distance to next hex grid crossing along (dx, dy, dz) from
// (px, py, pz) currently inside element (q, r, zi). Six in-plane
// edges plus 2 axial planes; min positive is reported.
__device__ double gr_hex_distance_to_grid(
    const GrGeometry* g, int hex_id,
    double px, double py, double pz,
    double dx, double dy, double dz,
    int q, int r, int zi)
{
    const double* ctr = g->hex_center + hex_id * 3;
    double pitch_xy = g->hex_pitch_xy[hex_id];
    double pitch_z = g->hex_pitch_z[hex_id];
    int n_axial = g->hex_n_axial[hex_id];
    int orient = g->hex_orientation[hex_id];

    // Position relative to lattice frame, then to element centre.
    double rel_x = (px - ctr[0]);
    double rel_y = (py - ctr[1]);
    double rel_z = (pz - ctr[2]);
    double cx, cy;
    gr_hex_element_center_local(pitch_xy, orient, q, r, &cx, &cy);
    double pos_x = rel_x - cx;
    double pos_y = rel_y - cy;
    double d_perp = pitch_xy * 0.5;  // half centre-to-centre

    double best = 1e300;

    // Six in-plane edge normals. For HexOrientation::Y the edge
    // midpoints are at 30°, 90°, 150°, 210°, 270°, 330°. For X they're
    // at 0°, 60°, 120°, 180°, 240°, 300°.
    double n_x[6], n_y[6];
    if (orient == GR_HEX_ORIENT_Y) {
        const double s30 = 0.5;
        const double c30 = 0.8660254037844387;  // sqrt(3)/2
        n_x[0] =  c30; n_y[0] =  s30;
        n_x[1] =  0.0; n_y[1] =  1.0;
        n_x[2] = -c30; n_y[2] =  s30;
        n_x[3] = -c30; n_y[3] = -s30;
        n_x[4] =  0.0; n_y[4] = -1.0;
        n_x[5] =  c30; n_y[5] = -s30;
    } else {
        const double s30 = 0.5;
        const double c30 = 0.8660254037844387;
        n_x[0] =  1.0; n_y[0] =  0.0;
        n_x[1] =  s30; n_y[1] =  c30;
        n_x[2] = -s30; n_y[2] =  c30;
        n_x[3] = -1.0; n_y[3] =  0.0;
        n_x[4] = -s30; n_y[4] = -c30;
        n_x[5] =  s30; n_y[5] = -c30;
    }
    for (int i = 0; i < 6; ++i) {
        double pn = pos_x * n_x[i] + pos_y * n_y[i];
        double dn = dx * n_x[i] + dy * n_y[i];
        if (fabs(dn) < 1e-300) continue;
        // Edge plane: pn + t*dn = d_perp
        double t = (d_perp - pn) / dn;
        if (t > 1e-12 && t < best) best = t;
    }

    // Axial planes: z layer boundaries.
    if (fabs(dz) > 1e-300) {
        double layer_low = ((double)zi - (double)n_axial * 0.5) * pitch_z;
        double layer_hi = layer_low + pitch_z;
        double tlo = (layer_low - rel_z) / dz;
        double thi = (layer_hi - rel_z) / dz;
        if (tlo > 1e-12 && tlo < best) best = tlo;
        if (thi > 1e-12 && thi < best) best = thi;
    }

    return best;
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
        if (ft == GR_FILL_HEX_LATTICE) {
            int hex_id = fd;
            int q, r, zi;
            if (!gr_hex_find_element(g, hex_id, local_x, local_y, local_z,
                                     &q, &r, &zi)) {
                return 0;  // off the hex lattice
            }
            current_universe = gr_hex_universe_at(g, hex_id, q, r, zi);
            // Element offset in lattice frame: hex_center + element_centre.
            const double* ctr = g->hex_center + hex_id * 3;
            double pitch_xy = g->hex_pitch_xy[hex_id];
            double pitch_z = g->hex_pitch_z[hex_id];
            int orient = g->hex_orientation[hex_id];
            int n_axial = g->hex_n_axial[hex_id];
            double cx, cy;
            gr_hex_element_center_local(pitch_xy, orient, q, r, &cx, &cy);
            double cz = ((double)zi - (double)n_axial * 0.5 + 0.5) * pitch_z;
            next_off_x = ctr[0] + cx;
            next_off_y = ctr[1] + cy;
            next_off_z = ctr[2] + cz;
            // has_lattice = 2 distinguishes hex from rect (=1).
            next_has_lat = 2;
            next_lat_id = hex_id;
            next_lat_ix = q; next_lat_iy = r; next_lat_iz = zi;
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

// ── Surface distance along a ray ────────────────────────────────────

__device__ __forceinline__ double gr_dist_plane(double p, double d, double x0) {
    if (fabs(d) < 1e-300) return 1e300;
    double t = (x0 - p) / d;
    return (t > 1e-12) ? t : 1e300;
}

__device__ __forceinline__ double gr_dist_sphere(
    double px, double py, double pz, double dx, double dy, double dz,
    double cx, double cy, double cz, double r)
{
    double rx = px - cx, ry = py - cy, rz = pz - cz;
    double a = dx*dx + dy*dy + dz*dz;
    double b = 2.0 * (rx*dx + ry*dy + rz*dz);
    double c = rx*rx + ry*ry + rz*rz - r*r;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return 1e300;
    double sq = sqrt(disc);
    double t1 = (-b - sq) / (2.0 * a);
    double t2 = (-b + sq) / (2.0 * a);
    if (t1 > 1e-12) return t1;
    if (t2 > 1e-12) return t2;
    return 1e300;
}

__device__ __forceinline__ double gr_dist_cyl(
    double p1, double p2, double d1, double d2, double c1, double c2, double r)
{
    double r1 = p1 - c1, r2 = p2 - c2;
    double a = d1*d1 + d2*d2;
    if (a < 1e-300) return 1e300;
    double b = 2.0 * (r1*d1 + r2*d2);
    double c = r1*r1 + r2*r2 - r*r;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return 1e300;
    double sq = sqrt(disc);
    double t1 = (-b - sq) / (2.0 * a);
    double t2 = (-b + sq) / (2.0 * a);
    if (t1 > 1e-12) return t1;
    if (t2 > 1e-12) return t2;
    return 1e300;
}

__device__ __forceinline__ double gr_surf_dist(
    const GrGeometry* g, int s_idx,
    double px, double py, double pz,
    double dx, double dy, double dz)
{
    int t = g->surf_type[s_idx];
    const double* p = g->surf_params + s_idx * 8;
    switch (t) {
        case GR_SURF_PLANE_X:        return gr_dist_plane(px, dx, p[0]);
        case GR_SURF_PLANE_Y:        return gr_dist_plane(py, dy, p[0]);
        case GR_SURF_PLANE_Z:        return gr_dist_plane(pz, dz, p[0]);
        case GR_SURF_SPHERE:         return gr_dist_sphere(px, py, pz, dx, dy, dz, p[0], p[1], p[2], p[3]);
        case GR_SURF_CYL_Z:          return gr_dist_cyl(px, py, dx, dy, p[0], p[1], p[2]);
        case GR_SURF_CYL_X:          return gr_dist_cyl(py, pz, dy, dz, p[0], p[1], p[2]);
        case GR_SURF_CYL_Y:          return gr_dist_cyl(px, pz, dx, dz, p[0], p[1], p[2]);
        case GR_SURF_PLANE_GENERAL: {
            double denom = p[0]*dx + p[1]*dy + p[2]*dz;
            if (fabs(denom) < 1e-300) return 1e300;
            double t_val = (p[3] - (p[0]*px + p[1]*py + p[2]*pz)) / denom;
            return (t_val > 1e-12) ? t_val : 1e300;
        }
        default: return 1e300;
    }
}

// ── Lattice grid distance ───────────────────────────────────────────

__device__ double gr_lattice_distance_to_grid(
    const GrGeometry* g, int lat_id,
    double px, double py, double pz,
    double dx, double dy, double dz,
    int ix, int iy, int iz)
{
    const double* org = g->lat_origin + lat_id * 3;
    const double* pit = g->lat_pitch  + lat_id * 3;
    int idx[3] = {ix, iy, iz};
    double pos[3] = {px - org[0], py - org[1], pz - org[2]};
    double dir[3] = {dx, dy, dz};
    double best = 1e300;
    for (int axis = 0; axis < 3; ++axis) {
        double d = dir[axis];
        if (d == 0.0) continue;
        double pitch = pit[axis];
        double target = (d > 0.0)
            ? ((double)(idx[axis] + 1)) * pitch
            : ((double)idx[axis]) * pitch;
        double t = (target - pos[axis]) / d;
        if (t <= 0.0) {
            double next_target = (d > 0.0)
                ? ((double)(idx[axis] + 2)) * pitch
                : ((double)(idx[axis] - 1)) * pitch;
            t = (next_target - pos[axis]) / d;
        }
        if (t > 0.0 && t < best) best = t;
    }
    return best;
}

// ── Recursive trace step ────────────────────────────────────────────
//
// Output:
//   *out_distance      — distance to next event along world_dir
//   *out_surface_idx   — global surface index hit, -1 if grid crossing
//   *out_bc            — boundary condition (matches GR_BC_*)
//   *out_next_stack    — re-resolved stack at the new world_pos
//   *out_next_depth    — depth of that stack (0 = leakage)
//
// Returns 1 on success (got an event), 0 if no event found (1e300
// distance — also leakage).

__device__ int gr_trace_step(
    const GrGeometry* g,
    const GrCoord* stack, int depth,
    double world_x, double world_y, double world_z,
    double world_dx, double world_dy, double world_dz,
    double* out_distance, int* out_surface_idx, int* out_bc,
    GrCoord* out_next_stack, int* out_next_depth)
{
    // Per-frame local positions (rotation is identity in v1 — the GPU
    // port doesn't yet implement Mat3 cascades).
    double locals_x[GR_MAX_DEPTH], locals_y[GR_MAX_DEPTH], locals_z[GR_MAX_DEPTH];
    {
        double lx = world_x, ly = world_y, lz = world_z;
        for (int i = 0; i < depth; ++i) {
            lx -= stack[i].offx;
            ly -= stack[i].offy;
            lz -= stack[i].offz;
            locals_x[i] = lx; locals_y[i] = ly; locals_z[i] = lz;
        }
    }

    double best_dist = 1e300;
    int best_surface = -1;

    // Source 1+2: every cell on the stack contributes its surfaces.
    for (int d = 0; d < depth; ++d) {
        int cell_idx = stack[d].cell_idx;
        int r_off = g->cell_region_off[cell_idx];
        int r_len = g->cell_region_len[cell_idx];
        // Walk region opcodes; for HALFSPACE_* the arg is a global
        // surface index — try its distance.
        double lx = locals_x[d], ly = locals_y[d], lz = locals_z[d];
        for (int i = 0; i < r_len; ++i) {
            int op  = g->region_op[r_off + i];
            int arg = g->region_arg[r_off + i];
            if (op == GR_REGION_HALFSPACE_POS || op == GR_REGION_HALFSPACE_NEG) {
                double t = gr_surf_dist(g, arg, lx, ly, lz, world_dx, world_dy, world_dz);
                if (t < best_dist) {
                    best_dist = t;
                    best_surface = arg;
                }
            }
        }
    }

    // Source 3: lattice grid lines, evaluated in the parent frame.
    // has_lattice == 1 → rect (Amanatides-Woo on axis-aligned grid).
    // has_lattice == 2 → hex (6 edge normals + 2 axial planes).
    const double COINCIDENCE_TOL_GRID = 1e-9;
    for (int d = 0; d < depth; ++d) {
        if (stack[d].has_lattice == 0) continue;
        int lat_id = stack[d].lattice_id;
        double px, py, pz;
        if (d == 0) {
            px = world_x; py = world_y; pz = world_z;
        } else {
            px = locals_x[d - 1]; py = locals_y[d - 1]; pz = locals_z[d - 1];
        }
        double t;
        if (stack[d].has_lattice == 1) {
            t = gr_lattice_distance_to_grid(
                g, lat_id, px, py, pz,
                world_dx, world_dy, world_dz,
                stack[d].lat_ix, stack[d].lat_iy, stack[d].lat_iz);
        } else {
            // has_lattice == 2 — hex.
            t = gr_hex_distance_to_grid(
                g, lat_id, px, py, pz,
                world_dx, world_dy, world_dz,
                stack[d].lat_ix, stack[d].lat_iy, stack[d].lat_iz);
        }
        if (t + COINCIDENCE_TOL_GRID < best_dist) {
            best_dist = t;
            best_surface = -1;
        }
    }

    if (best_dist >= 1e299) {
        *out_distance = 1e300;
        *out_surface_idx = -1;
        *out_bc = GR_BC_VACUUM;
        *out_next_depth = 0;
        return 0;
    }

    *out_distance = best_dist;
    *out_surface_idx = best_surface;
    *out_bc = (best_surface >= 0) ? g->surf_bc[best_surface] : GR_BC_TRANSMISSION;

    // Re-resolve the stack at the new world position (offset by a
    // small nudge along world_dir).
    const double NUDGE = 1e-10;
    double nx = world_x + world_dx * (best_dist + NUDGE);
    double ny = world_y + world_dy * (best_dist + NUDGE);
    double nz = world_z + world_dz * (best_dist + NUDGE);
    int next_depth = gr_find_cell(g, nx, ny, nz, out_next_stack);
    *out_next_depth = next_depth;
    return 1;
}

#endif // GEOM_RECURSIVE_CU
