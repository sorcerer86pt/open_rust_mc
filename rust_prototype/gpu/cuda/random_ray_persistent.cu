// Persistent immortal-ray random-ray kernel.
//
// Each CUDA thread owns one persistent ray (pos, dir, psi[g], stack).
// The kernel runs one batch of the random-ray power iteration: walks
// each ray for `active_length` cm, accumulating per-FSR per-group
// track-length contributions via atomicAdd.
//
// Inputs (device buffers):
//   geom_*           : geometry tables (same layout as transport_recursive_const.cu)
//   ray_pos / dir    : Vec3 per thread (persistent across batches)
//   ray_stack_*      : flattened CoordStack per thread
//   ray_psi          : per-group ψ per thread (n_threads × n_groups)
//   q                : per-FSR per-group source Q[fsr*n_g + g]
//   sigma_t_per_fsr  : per-FSR per-group Σ_t[fsr*n_g + g]
//   fsr_aabb_min/max : Cartesian FSR mesh bounds (Vec3 each)
//   fsr_n            : Cartesian dims [n_x, n_y, n_z]
//   fsr_spacing      : voxel spacing [s_x, s_y, s_z]
//
// Outputs (device buffers, atomicAdd):
//   track_psi        : per-FSR per-group track-length numerator
//                      [n_fsrs × n_groups]
//   volume_track     : per-FSR ray-track-length sum [n_fsrs]
//
// Configuration:
//   active_length    : cm to advance per ray per batch
//   n_groups         : number of energy groups (≤ MAX_GROUPS)
//   immortal         : if 1, ray state is preserved; if 0, dead-zone
//                      run-up before active accumulation.
//
// Runtime parity validation against CPU is deferred — see
// `random_ray::solver::tests::immortal_*` for the CPU-side tests this
// kernel is expected to match within MC noise.

#include "geom_recursive.cu"

#define MAX_GROUPS 16
#define FOUR_PI 12.566370614359172

// Stable (1 - exp(-tau))/tau evaluation. Mirrors the Rust
// `integrator::exp_m1_over` series.
__device__ double rr_exp_m1_over(double tau) {
    double a = tau < 0.0 ? -tau : tau;
    if (a < 1.0e-4) {
        return 1.0 - tau * (0.5 - tau * ((1.0 / 6.0) - tau * (1.0 / 24.0)));
    }
    return (1.0 - exp(-tau)) / tau;
}

// Cartesian FSR lookup for the random-ray kernel.
__device__ int rr_fsr_at_cartesian(
    double px, double py, double pz,
    double aabb_min_x, double aabb_min_y, double aabb_min_z,
    double sx, double sy, double sz,
    int nx, int ny, int nz)
{
    int ix = (int)floor((px - aabb_min_x) / sx);
    int iy = (int)floor((py - aabb_min_y) / sy);
    int iz = (int)floor((pz - aabb_min_z) / sz);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
        return -1;
    }
    return (ix * ny + iy) * nz + iz;
}

// Outward-pointing normal at a position on a surface. Limited to
// axis-aligned planes + spheres + cylinder-Z; matches the CPU
// random-ray reflective-BC handler.
__device__ void rr_surface_normal(
    int surf_idx,
    const int* surf_type,
    const double* surf_params,
    double px, double py, double pz,
    double* nx, double* ny, double* nz)
{
    int t = surf_type[surf_idx];
    int p_off = surf_idx * 8; // surf_params layout: 8 doubles per surface
    if (t == 0) { // generic Plane: normal in params[0..2]
        *nx = surf_params[p_off + 0];
        *ny = surf_params[p_off + 1];
        *nz = surf_params[p_off + 2];
    } else if (t == 1) { // PlaneX
        *nx = 1.0; *ny = 0.0; *nz = 0.0;
    } else if (t == 2) { // PlaneY
        *nx = 0.0; *ny = 1.0; *nz = 0.0;
    } else if (t == 3) { // PlaneZ
        *nx = 0.0; *ny = 0.0; *nz = 1.0;
    } else if (t == 4) { // Sphere
        double cx = surf_params[p_off + 0];
        double cy = surf_params[p_off + 1];
        double cz = surf_params[p_off + 2];
        double dx = px - cx;
        double dy = py - cy;
        double dz = pz - cz;
        double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
        *nx = dx * inv;
        *ny = dy * inv;
        *nz = dz * inv;
    } else if (t == 5) { // CylinderZ
        double cx = surf_params[p_off + 0];
        double cy = surf_params[p_off + 1];
        double dx = px - cx;
        double dy = py - cy;
        double inv = 1.0 / sqrt(dx * dx + dy * dy);
        *nx = dx * inv;
        *ny = dy * inv;
        *nz = 0.0;
    } else {
        *nx = 0.0; *ny = 0.0; *nz = 1.0;
    }
}

// Material id for the deepest cell in the ray's stack.
__device__ int rr_effective_material(
    GrGeometry* g,
    int* stack,
    int depth)
{
    int last_frame = (depth - 1) * GR_COORD_FIELDS;
    int cell_idx = stack[last_frame + 0];
    int fill_t = g->cell_fill_type[cell_idx];
    int fill_d = g->cell_fill_data[cell_idx];
    if (fill_t == 1 /* Material */) return fill_d;
    if (fill_t == 4 /* Void */) return -1;
    // Universe / Lattice should not be deepest — defensive return.
    return -1;
}

// Persistent random-ray kernel.
//
// One thread per ray. Caller chooses grid/block such that
// gridDim.x * blockDim.x == n_rays.
extern "C" __global__ void random_ray_persistent(
    GrGeometry g,
    // Per-ray persistent state (n_rays):
    double* ray_pos,        // 3 × n_rays
    double* ray_dir,        // 3 × n_rays
    int* ray_stack,         // GR_MAX_DEPTH * GR_COORD_FIELDS × n_rays
    int* ray_depth,         // n_rays
    double* ray_psi,        // n_groups × n_rays
    int n_rays,
    int n_groups,
    // Source + XS (read-only):
    const double* q,                  // n_fsrs × n_groups
    const double* sigma_t_per_fsr,    // n_fsrs × n_groups
    const int* fsr_material,          // n_fsrs (-1 = inactive)
    // Cartesian FSR mesh:
    double aabb_min_x, double aabb_min_y, double aabb_min_z,
    double sx, double sy, double sz,
    int n_fsr_x, int n_fsr_y, int n_fsr_z,
    int n_fsrs,
    // Accumulators (atomicAdd):
    double* track_psi,    // n_fsrs × n_groups
    double* volume_track, // n_fsrs
    // Per-segment scratch (n_rays × n_surfaces, supplied by host):
    double* eval_scratch,
    int n_surfaces,
    // Configuration:
    double active_length,
    int max_segments)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rays) return;

    // Load persistent state for this thread's ray.
    double px = ray_pos[3 * tid + 0];
    double py = ray_pos[3 * tid + 1];
    double pz = ray_pos[3 * tid + 2];
    double dx = ray_dir[3 * tid + 0];
    double dy = ray_dir[3 * tid + 1];
    double dz = ray_dir[3 * tid + 2];
    int* stack = ray_stack + tid * GR_MAX_DEPTH * GR_COORD_FIELDS;
    int depth = ray_depth[tid];
    double psi[MAX_GROUPS];
    for (int gg = 0; gg < n_groups; gg++) {
        psi[gg] = ray_psi[tid * n_groups + gg];
    }

    if (depth == 0) {
        // Re-resolve coord stack for this ray.
        depth = gr_find_cell(&g, px, py, pz, stack);
        if (depth == 0) {
            // Off-mesh — skip this ray.
            ray_depth[tid] = 0;
            return;
        }
    }

    double traveled = 0.0;
    double* my_evals = eval_scratch + (size_t)tid * (size_t)n_surfaces;

    for (int step = 0; step < max_segments; step++) {
        // Find nearest surface crossing.
        double dist; int surf_idx; int bc; int next_depth;
        bool ok = gr_trace_step(&g, stack, depth,
            px, py, pz, dx, dy, dz,
            &dist, &surf_idx, &bc, &next_depth, my_evals);
        if (!ok) break;

        double dist_active = active_length - traveled;
        double seg_len = dist < dist_active ? dist : dist_active;
        if (!isfinite(seg_len) || seg_len <= 0.0) break;

        // Per-segment MoC ODE step + accumulate.
        int fsr = rr_fsr_at_cartesian(
            px, py, pz,
            aabb_min_x, aabb_min_y, aabb_min_z,
            sx, sy, sz, n_fsr_x, n_fsr_y, n_fsr_z);
        if (fsr >= 0 && fsr < n_fsrs && fsr_material[fsr] >= 0) {
            atomicAdd(&volume_track[fsr], seg_len);
            for (int gg = 0; gg < n_groups; gg++) {
                double sigma_t = sigma_t_per_fsr[fsr * n_groups + gg];
                double q_per_sr = q[fsr * n_groups + gg] / FOUR_PI;
                double tau = sigma_t * seg_len;
                double f = rr_exp_m1_over(tau);
                double q_over_t = q_per_sr / sigma_t;
                double psi_avg = psi[gg] * f + q_over_t * (1.0 - f);
                double exp_neg_tau;
                if (tau < 1.0e-4 && tau > -1.0e-4) {
                    exp_neg_tau = 1.0 - tau * (1.0 - tau * (0.5 - tau * (1.0 / 6.0)));
                } else {
                    exp_neg_tau = exp(-tau);
                }
                double psi_out = psi[gg] * exp_neg_tau + q_over_t * (1.0 - exp_neg_tau);
                psi[gg] = psi_out;
                atomicAdd(&track_psi[fsr * n_groups + gg], seg_len * psi_avg);
            }
        }

        traveled += seg_len;
        if (traveled >= active_length - 1.0e-12) {
            // Advance final position so we resume here next batch.
            px += dx * seg_len;
            py += dy * seg_len;
            pz += dz * seg_len;
            break;
        }

        if (dist <= dist_active) {
            // Geometry crossing — handle BC.
            // bc: 0 = Transmission, 1 = Reflective, 2 = Vacuum
            if (bc == 2) {
                // Vacuum: reflect direction, zero ψ.
                px += dx * dist;
                py += dy * dist;
                pz += dz * dist;
                double nx, ny, nz;
                rr_surface_normal(surf_idx,
                    g.surf_type, g.surf_params, px, py, pz, &nx, &ny, &nz);
                double two_dn = 2.0 * (dx * nx + dy * ny + dz * nz);
                dx -= nx * two_dn;
                dy -= ny * two_dn;
                dz -= nz * two_dn;
                for (int gg = 0; gg < n_groups; gg++) psi[gg] = 0.0;
                px += dx * 1.0e-10;
                py += dy * 1.0e-10;
                pz += dz * 1.0e-10;
                depth = gr_find_cell(&g, px, py, pz, stack);
                if (depth == 0) break;
            } else if (bc == 1) {
                // Reflective: reflect direction, preserve ψ.
                px += dx * dist;
                py += dy * dist;
                pz += dz * dist;
                double nx, ny, nz;
                rr_surface_normal(surf_idx,
                    g.surf_type, g.surf_params, px, py, pz, &nx, &ny, &nz);
                double two_dn = 2.0 * (dx * nx + dy * ny + dz * nz);
                dx -= nx * two_dn;
                dy -= ny * two_dn;
                dz -= nz * two_dn;
                px += dx * 1.0e-10;
                py += dy * 1.0e-10;
                pz += dz * 1.0e-10;
                depth = gr_find_cell(&g, px, py, pz, stack);
                if (depth == 0) break;
            } else {
                // Transmission: continue across the surface; the
                // updated stack came back from gr_trace_step.
                px += dx * (dist + 1.0e-10);
                py += dy * (dist + 1.0e-10);
                pz += dz * (dist + 1.0e-10);
                depth = next_depth > 0 ? next_depth : gr_find_cell(&g, px, py, pz, stack);
                if (depth == 0) break;
            }
        } else {
            // Phase budget hit — advance position only.
            px += dx * seg_len;
            py += dy * seg_len;
            pz += dz * seg_len;
        }
    }

    // Write back persistent state.
    ray_pos[3 * tid + 0] = px;
    ray_pos[3 * tid + 1] = py;
    ray_pos[3 * tid + 2] = pz;
    ray_dir[3 * tid + 0] = dx;
    ray_dir[3 * tid + 1] = dy;
    ray_dir[3 * tid + 2] = dz;
    ray_depth[tid] = depth;
    for (int gg = 0; gg < n_groups; gg++) {
        ray_psi[tid * n_groups + gg] = psi[gg];
    }
}
