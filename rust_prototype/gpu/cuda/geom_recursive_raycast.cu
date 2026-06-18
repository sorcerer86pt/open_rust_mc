// SPDX-License-Identifier: MIT
// GPU ray-cast preview kernel. Renders a perspective 3D image of the
// recursive CSG geometry by casting one camera ray per pixel through
// the SAME device geometry walk the transport kernels use
// (`gr_find_cell` / `gr_trace_step` from geom_recursive.cu) — no
// transport state (RNG / cross-sections / tallies) is touched.
//
// Mirror of the CPU `render3d` path in src/bin/preview_scene.rs:
//   - march to the first OPAQUE material (air/void skipped),
//   - shade Lambert (key light + head-light) tinted by the material's
//     palette colour,
//   - analytic surface normal from the hit cell's bounding surface,
//   - dark-slate background gradient on a miss / leak.
//
// Concatenated after geom_recursive.cu by `assemble_kernel_source`, so
// all gr_* helpers and GR_* constants are already in scope.

// Outward (un-normalised then normalised) surface normal at a local-
// frame point. Translation-only descent (the GPU path rejects per-cell
// rotations) means the local normal direction equals the world normal.
__device__ __forceinline__ void gr_surf_normal(
    const GrGeometry* g, int s_idx, double x, double y, double z,
    double* nx, double* ny, double* nz)
{
    int t = g->surf_type[s_idx];
    const double* p = g->surf_params + s_idx * 8;
    double ax = 0.0, ay = 0.0, az = 0.0;
    switch (t) {
        case GR_SURF_PLANE_X:       ax = 1.0; break;
        case GR_SURF_PLANE_Y:       ay = 1.0; break;
        case GR_SURF_PLANE_Z:       az = 1.0; break;
        case GR_SURF_SPHERE:        ax = x - p[0]; ay = y - p[1]; az = z - p[2]; break;
        case GR_SURF_CYL_Z:         ax = x - p[0]; ay = y - p[1]; az = 0.0; break;
        case GR_SURF_CYL_X:         ax = 0.0; ay = y - p[0]; az = z - p[1]; break;
        case GR_SURF_CYL_Y:         ax = x - p[0]; ay = 0.0; az = z - p[1]; break;
        case GR_SURF_PLANE_GENERAL: ax = p[0]; ay = p[1]; az = p[2]; break;
        default:                    az = 1.0; break;
    }
    double len = sqrt(ax * ax + ay * ay + az * az);
    if (len > 1e-12) { *nx = ax / len; *ny = ay / len; *nz = az / len; }
    else { *nx = 0.0; *ny = 0.0; *nz = 1.0; }
}

// Ray vs axis-aligned box (slab method). Returns the [t0, t1] overlap;
// false on a miss. t1 >= t0 and t1 >= 0 on success.
__device__ bool gr_ray_aabb(
    double ox, double oy, double oz, double dx, double dy, double dz,
    double minx, double miny, double minz,
    double maxx, double maxy, double maxz,
    double* t0, double* t1)
{
    double tmin = -1e300, tmax = 1e300;
    // x slab
    if (fabs(dx) < 1e-12) { if (ox < minx || ox > maxx) return false; }
    else { double inv = 1.0 / dx; double a = (minx - ox) * inv, b = (maxx - ox) * inv;
           if (a > b) { double t = a; a = b; b = t; } if (a > tmin) tmin = a; if (b < tmax) tmax = b; }
    // y slab
    if (fabs(dy) < 1e-12) { if (oy < miny || oy > maxy) return false; }
    else { double inv = 1.0 / dy; double a = (miny - oy) * inv, b = (maxy - oy) * inv;
           if (a > b) { double t = a; a = b; b = t; } if (a > tmin) tmin = a; if (b < tmax) tmax = b; }
    // z slab
    if (fabs(dz) < 1e-12) { if (oz < minz || oz > maxz) return false; }
    else { double inv = 1.0 / dz; double a = (minz - oz) * inv, b = (maxz - oz) * inv;
           if (a > b) { double t = a; a = b; b = t; } if (a > tmin) tmin = a; if (b < tmax) tmax = b; }
    if (tmax >= tmin && tmax >= 0.0) { *t0 = tmin; *t1 = tmax; return true; }
    return false;
}

extern "C" __global__ void raycast_preview(
    // camera basis + projection
    double cam_px, double cam_py, double cam_pz,
    double fwd_x, double fwd_y, double fwd_z,
    double right_x, double right_y, double right_z,
    double up_x, double up_y, double up_z,
    double tan_half_fov, double aspect,
    int width, int height,
    // scene bounds (for ray entry + micro-step / eps scaling)
    double aabb_min_x, double aabb_min_y, double aabb_min_z,
    double aabb_max_x, double aabb_max_y, double aabb_max_z,
    // shading: palette [n_materials*3] (0..255), opaque flags [n_materials]
    const int* palette, const int* opaque_mask, int n_materials,
    // geometry (same SoA arg order as find_cell_batch)
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
    const double* lat_origin, const double* lat_pitch, const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    double* evals_scratch,
    unsigned int* out_rgb)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pixels = width * height;
    if (tid >= n_pixels) return;

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
    g.hex_center = hex_center; g.hex_pitch_xy = hex_pitch_xy; g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings; g.hex_n_axial = hex_n_axial; g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off; g.hex_universes = hex_universes;
    g.n_hex_lattices = 0;
    g.evals = evals_scratch + tid * n_surfaces;

    // Background gradient (matches the CPU path's dark slate).
    int py_i = tid / width;
    int px_i = tid % width;
    double bf = (double) py_i / (double) (height > 1 ? height : 1);
    unsigned int bg =
        (((unsigned int) (int) (16.0 + 14.0 * bf)) << 16) |
        (((unsigned int) (int) (17.0 + 16.0 * bf)) << 8) |
        ((unsigned int) (int) (23.0 + 19.0 * bf));

    // Primary ray for this pixel.
    double ndc_x = ((px_i + 0.5) / (double) width) * 2.0 - 1.0;
    double ndc_y = 1.0 - ((py_i + 0.5) / (double) height) * 2.0;
    double dx = fwd_x + right_x * (ndc_x * aspect * tan_half_fov) + up_x * (ndc_y * tan_half_fov);
    double dy = fwd_y + right_y * (ndc_x * aspect * tan_half_fov) + up_y * (ndc_y * tan_half_fov);
    double dz = fwd_z + right_z * (ndc_x * aspect * tan_half_fov) + up_z * (ndc_y * tan_half_fov);
    {
        double dl = sqrt(dx * dx + dy * dy + dz * dz);
        if (dl > 1e-30) { dx /= dl; dy /= dl; dz /= dl; }
    }

    double t_enter, t_exit;
    if (!gr_ray_aabb(cam_px, cam_py, cam_pz, dx, dy, dz,
                     aabb_min_x, aabb_min_y, aabb_min_z,
                     aabb_max_x, aabb_max_y, aabb_max_z,
                     &t_enter, &t_exit)) {
        out_rgb[tid] = bg;
        return;
    }

    double ex = aabb_max_x - aabb_min_x;
    double ey = aabb_max_y - aabb_min_y;
    double ez = aabb_max_z - aabb_min_z;
    double diag = sqrt(ex * ex + ey * ey + ez * ez);
    double micro = fmax(diag / 1024.0, 1e-6);
    double eps = fmax(diag * 1e-7, 1e-9);

    double t = fmax(t_enter, 0.0) + eps;
    double px = cam_px + dx * t, py = cam_py + dy * t, pz = cam_pz + dz * t;

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, px, py, pz, stack);

    for (int iter = 0; iter < 8192; ++iter) {
        if (t > t_exit) { out_rgb[tid] = bg; return; }
        if (depth > 0) {
            int ci = stack[depth - 1].cell_idx;
            int ft = g.cell_fill_type[ci];
            int m = g.cell_fill_data[ci];
            bool is_opaque = (ft == GR_FILL_MATERIAL) &&
                             (m >= n_materials || opaque_mask[m] != 0);
            if (ft == GR_FILL_MATERIAL && is_opaque) {
                // Analytic normal from the hit cell's nearest bounding
                // surface (local frame == world direction; no rotation).
                double offx = 0.0, offy = 0.0, offz = 0.0;
                for (int i = 0; i < depth; ++i) {
                    offx += stack[i].offx; offy += stack[i].offy; offz += stack[i].offz;
                }
                double lx = px - offx, ly = py - offy, lz = pz - offz;
                int roff = g.cell_region_off[ci];
                int rlen = g.cell_region_len[ci];
                int best = -1; double best_abs = 1e300;
                for (int i = 0; i < rlen; ++i) {
                    int op = g.region_op[roff + i];
                    int arg = g.region_arg[roff + i];
                    if (op == GR_REGION_HALFSPACE_POS || op == GR_REGION_HALFSPACE_NEG) {
                        double v = fabs(gr_surf_eval(&g, arg, lx, ly, lz));
                        if (v < best_abs) { best_abs = v; best = arg; }
                    }
                }
                double nx, ny, nz;
                if (best >= 0) { gr_surf_normal(&g, best, lx, ly, lz, &nx, &ny, &nz); }
                else { nx = -dx; ny = -dy; nz = -dz; }
                if (nx * dx + ny * dy + nz * dz > 0.0) { nx = -nx; ny = -ny; nz = -nz; }

                // Lambert: fixed key light + head-light.
                double kx = 0.35, ky = 0.45, kz = 0.82;
                double kl = sqrt(kx * kx + ky * ky + kz * kz);
                kx /= kl; ky /= kl; kz /= kl;
                double kd = fmax(0.0, nx * kx + ny * ky + nz * kz);
                double hd = fmax(0.0, -(nx * dx + ny * dy + nz * dz));
                double lit = fmin(0.18 + 0.82 * (0.55 * kd + 0.45 * hd), 1.15);

                int r = 200, gg = 200, b = 200;
                if (m >= 0 && m < n_materials) {
                    r = palette[m * 3 + 0]; gg = palette[m * 3 + 1]; b = palette[m * 3 + 2];
                }
                int rr = (int) fmin(255.0, fmax(0.0, r * lit + 0.5));
                int gr2 = (int) fmin(255.0, fmax(0.0, gg * lit + 0.5));
                int bb = (int) fmin(255.0, fmax(0.0, b * lit + 0.5));
                out_rgb[tid] = ((unsigned int) rr << 16) | ((unsigned int) gr2 << 8) | (unsigned int) bb;
                return;
            }
            // Transparent leaf (air material or VOID): step to next surface.
            double dist; int surf; int bc; int next_depth;
            GrCoord next_stack[GR_MAX_DEPTH];
            if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                               &dist, &surf, &bc, next_stack, &next_depth)
                || !(dist < 1e299)) {
                out_rgb[tid] = bg; return;
            }
            t += dist + eps;
            px = cam_px + dx * t; py = cam_py + dy * t; pz = cam_pz + dz * t;
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
        } else {
            // Vacuum / leak gap: creep forward until we re-enter a cell.
            t += micro;
            px = cam_px + dx * t; py = cam_py + dy * t; pz = cam_pz + dz * t;
            depth = gr_find_cell(&g, px, py, pz, stack);
        }
    }
    out_rgb[tid] = bg;
}
