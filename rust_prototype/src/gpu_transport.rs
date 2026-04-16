//! Event-based GPU neutron transport for PWR pin cell.
//!
//! Implements the Tramm et al. (2024) event-based approach:
//!   Step 1: Batch SVD cross-section reconstruction (all particles)
//!   Step 2: Sample collision distance + ray-trace to nearest surface
//!   Step 3: Process event (advance, surface crossing or collision)
//!
//! Each step is a separate CUDA kernel launch — no warp divergence
//! within a step. Particles stay on GPU for the entire batch.
//!
//! Simplified physics for benchmarking (no thermal scattering, URR,
//! discrete levels). Enough for k_eff comparison with CPU.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc;

// ── CUDA kernel source ────────────────────────────────────────────────

/// All CUDA kernels for event-based transport.
///
/// PWR pin cell geometry is hardcoded (9 surfaces, 4 cells, 3 materials).
/// SVD basis data is passed via global memory, coefficients via shared memory.
const TRANSPORT_KERNELS: &str = r#"

// ════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════

#define COINCIDENCE_TOL 1e-12
#define MAX_RANK 8
#define N_REACTIONS 6   // elastic, inelastic, n2n, n3n, fission, capture
#define N_NUCLIDES 8
#define PI 3.14159265358979323846

// Reaction indices
#define RXN_ELASTIC   0
#define RXN_INELASTIC 1
#define RXN_N2N       2
#define RXN_N3N       3
#define RXN_FISSION   4
#define RXN_CAPTURE   5

// ════════════════════════════════════════════════════════════════════
// PCG-64 RNG (matches CPU implementation)
// ════════════════════════════════════════════════════════════════════

struct PcgState {
    unsigned long long state;
    unsigned long long inc;
};

__device__ unsigned int pcg_next(PcgState* rng) {
    unsigned long long old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = (unsigned int)(((old >> 18u) ^ old) >> 27u);
    unsigned int rot = (unsigned int)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ double pcg_uniform(PcgState* rng) {
    unsigned long long a = (unsigned long long)(pcg_next(rng) >> 5);
    unsigned long long b = (unsigned long long)(pcg_next(rng) >> 6);
    return (double)(a * 67108864ULL + b) * (1.0 / 9007199254740992.0);
}

__device__ void pcg_init(PcgState* rng, unsigned long long seed, unsigned long long stream) {
    rng->inc = (stream << 1u) | 1u;
    rng->state = 0;
    pcg_next(rng);
    rng->state += seed;
    pcg_next(rng);
}

// ════════════════════════════════════════════════════════════════════
// PWR Pin Cell Geometry (hardcoded)
// ════════════════════════════════════════════════════════════════════
//
// Surfaces:
//   0: CylinderZ  R=0.4096  (fuel outer)      Transmission
//   1: CylinderZ  R=0.4180  (clad inner)      Transmission
//   2: CylinderZ  R=0.4750  (clad outer)      Transmission
//   3: PlaneX    x=-0.63                       Reflective
//   4: PlaneX    x=+0.63                       Reflective
//   5: PlaneY    y=-0.63                       Reflective
//   6: PlaneY    y=+0.63                       Reflective
//   7: PlaneZ    z=-0.63                       Reflective
//   8: PlaneZ    z=+0.63                       Reflective
//
// Cells:
//   0: Fuel  (inside surf 0, between Z planes)          Material 0
//   1: Gap   (outside 0, inside 1, between Z planes)    Void
//   2: Clad  (outside 1, inside 2, between Z planes)    Material 1
//   3: Water (outside 2, inside box)                     Material 2

#define FUEL_OR   0.4096
#define CLAD_IR   0.4180
#define CLAD_OR   0.4750
#define HALF_PITCH 0.63

// Boundary condition types
#define BC_TRANSMISSION 0
#define BC_REFLECTIVE   1
#define BC_VACUUM       2

// Cell/material mapping
#define CELL_FUEL  0
#define CELL_GAP   1
#define CELL_CLAD  2
#define CELL_WATER 3

#define MAT_FUEL  0
#define MAT_CLAD  1
#define MAT_WATER 2
#define MAT_VOID  -1

// Material compositions (atom densities in atoms/barn-cm)
// Fuel (UO2): U235(idx=0), U238(idx=1), O16(idx=2)
// Clad (Zircaloy): Zr90(idx=4), Zr91(idx=5), Zr92(idx=6), Zr94(idx=7)
// Water (H2O): H1(idx=3), O16(idx=2)

// Material nuclide indices and densities are passed as kernel arguments.

// ── Geometry helpers ──────────────────────────────────────────────

// Evaluate distance to CylinderZ surface centered at origin
__device__ double dist_cylinder_z(double px, double py, double dx, double dy, double R) {
    double a = dx*dx + dy*dy;
    if (a < COINCIDENCE_TOL) return -1.0;
    double b = 2.0 * (px*dx + py*dy);
    double c = px*px + py*py - R*R;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return -1.0;
    double sq = sqrt(disc);
    double t1 = (-b - sq) / (2.0*a);
    double t2 = (-b + sq) / (2.0*a);
    if (t1 > COINCIDENCE_TOL) return t1;
    if (t2 > COINCIDENCE_TOL) return t2;
    return -1.0;
}

// Evaluate distance to PlaneX/Y/Z
__device__ double dist_plane(double p, double d, double x0) {
    if (fabs(d) < COINCIDENCE_TOL) return -1.0;
    double t = (x0 - p) / d;
    return (t > COINCIDENCE_TOL) ? t : -1.0;
}

// Find which cell contains a point
__device__ int find_cell(double x, double y, double z) {
    double r2 = x*x + y*y;
    bool in_z = (z > -HALF_PITCH) && (z < HALF_PITCH);
    if (!in_z) return -1;

    if (r2 < FUEL_OR * FUEL_OR) return CELL_FUEL;
    if (r2 < CLAD_IR * CLAD_IR) return CELL_GAP;
    if (r2 < CLAD_OR * CLAD_OR) return CELL_CLAD;

    bool in_box = (x > -HALF_PITCH) && (x < HALF_PITCH) &&
                  (y > -HALF_PITCH) && (y < HALF_PITCH);
    if (in_box) return CELL_WATER;
    return -1;
}

// Get material index for a cell (-1 = void)
__device__ int cell_material(int cell) {
    if (cell == CELL_FUEL)  return MAT_FUEL;
    if (cell == CELL_CLAD)  return MAT_CLAD;
    if (cell == CELL_WATER) return MAT_WATER;
    return MAT_VOID;  // gap
}

// Trace to nearest surface from current cell.
// Returns: distance, surface_id, boundary_condition, next_cell
__device__ void trace_surface(
    double px, double py, double pz,
    double dx, double dy, double dz,
    int cell,
    double* out_dist, int* out_bc, int* out_next_cell)
{
    double best_t = 1e20;
    int best_bc = BC_VACUUM;
    int best_next = -1;

    // Lambda-like helper: test a surface and update best
    #define TEST_SURF(t_val, bc_val) do { \
        double _t = (t_val); \
        if (_t > COINCIDENCE_TOL && _t < best_t) { \
            best_t = _t; \
            best_bc = (bc_val); \
            /* next_cell determined after advance */ \
        } \
    } while(0)

    // Test surfaces based on current cell
    // All cells have Z planes
    TEST_SURF(dist_plane(pz, dz, -HALF_PITCH), BC_REFLECTIVE);
    TEST_SURF(dist_plane(pz, dz,  HALF_PITCH), BC_REFLECTIVE);

    if (cell == CELL_FUEL) {
        // Fuel: bounded by cylinder 0
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, FUEL_OR), BC_TRANSMISSION);
    } else if (cell == CELL_GAP) {
        // Gap: between cylinder 0 (inner) and cylinder 1 (outer)
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, FUEL_OR), BC_TRANSMISSION);
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, CLAD_IR), BC_TRANSMISSION);
    } else if (cell == CELL_CLAD) {
        // Clad: between cylinder 1 and cylinder 2
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, CLAD_IR), BC_TRANSMISSION);
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, CLAD_OR), BC_TRANSMISSION);
    } else if (cell == CELL_WATER) {
        // Water: outside cylinder 2, inside reflective box
        TEST_SURF(dist_cylinder_z(px, py, dx, dy, CLAD_OR), BC_TRANSMISSION);
        TEST_SURF(dist_plane(px, dx, -HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(px, dx,  HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(py, dy, -HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(py, dy,  HALF_PITCH), BC_REFLECTIVE);
    }

    #undef TEST_SURF

    // Determine next cell for transmission crossings
    if (best_bc == BC_TRANSMISSION && best_t < 1e19) {
        double nx = px + dx * (best_t + 1e-10);
        double ny = py + dy * (best_t + 1e-10);
        double nz = pz + dz * (best_t + 1e-10);
        best_next = find_cell(nx, ny, nz);
    }

    *out_dist = best_t;
    *out_bc = best_bc;
    *out_next_cell = best_next;
}

// ════════════════════════════════════════════════════════════════════
// SVD cross-section reconstruction helpers
// ════════════════════════════════════════════════════════════════════

// Binary search on energy grid to find index
__device__ int energy_index(const double* grid, int n_e, double energy) {
    if (energy <= grid[0]) return 0;
    if (energy >= grid[n_e - 1]) return n_e - 1;
    int lo = 0, hi = n_e - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (grid[mid] <= energy) lo = mid;
        else hi = mid;
    }
    return lo;
}

// Reconstruct one cross-section from SVD basis + coefficients
__device__ double svd_reconstruct(
    const float* basis,    // [n_e * rank]
    const double* coeffs,  // [rank]
    int e_idx, int rank)
{
    const float* row = &basis[e_idx * rank];
    double acc = 0.0;
    for (int j = 0; j < rank; j++) {
        acc = fma((double)row[j], coeffs[j], acc);
    }
    return exp2(acc * 3.321928094887362);  // log10->linear
}

// ════════════════════════════════════════════════════════════════════
// KERNEL 1: Cross-section lookup for all alive particles
// ════════════════════════════════════════════════════════════════════
//
// For each alive particle, reconstruct micro XS for all reactions
// of all nuclides in the particle's material.
//
// Output: xs_total[N], xs_elastic[N], xs_fission[N], xs_capture[N],
//         xs_inelastic[N], xs_n2n[N], xs_n3n[N]
//         (macroscopic, already weighted by atom density and summed)
//         Also: micro_fission[N], awr_collision[N], nu_bar[N]
//         for collision processing

extern "C" __global__ void xs_lookup(
    // Particle state (SoA)
    const double* energy,      // [N]
    const int* cell_idx,       // [N]
    const int* alive,          // [N]
    int n_particles,
    // SVD data: flat arrays with offsets
    // For each nuclide, for each reaction: basis, coeffs, energy_grid, n_e
    // Packed as: nuclide_offsets[nuc][rxn] -> {basis_ptr, coeffs_ptr, grid_ptr, n_e}
    const float* all_basis,          // concatenated basis data
    const double* all_coeffs,        // concatenated coefficients
    const double* all_energy_grids,  // concatenated energy grids
    const int* basis_offsets,        // [N_NUCLIDES * N_REACTIONS] offset into all_basis
    const int* grid_offsets,         // [N_NUCLIDES] offset into all_energy_grids
    const int* n_energies,           // [N_NUCLIDES] energy grid sizes
    const int* has_reaction,         // [N_NUCLIDES * N_REACTIONS] 0/1
    const int* coeffs_offsets,       // [N_NUCLIDES * N_REACTIONS] offset into all_coeffs
    int rank,
    // Material data
    const int* mat_n_nuclides,       // [3] number of nuclides per material
    const int* mat_nuclide_idx,      // [3 * 4] nuclide indices per material (padded)
    const double* mat_atom_density,  // [3 * 4] atom densities per material (padded)
    const double* awr_table,         // [N_NUCLIDES] atomic weight ratios
    const double* nu_bar_const,      // [N_NUCLIDES] fallback nu-bar
    // Output: macroscopic XS per particle
    double* out_macro_total,         // [N]
    double* out_macro_elastic,       // [N]
    double* out_macro_fission,       // [N]
    double* out_macro_capture,       // [N]
    // Per-nuclide micro XS for collision sampling (flattened)
    double* out_micro_total,         // [N * 4] (max 4 nuclides/material)
    double* out_awr,                 // [N] AWR of sampled collision nuclide
    double* out_nu_bar)              // [N] nu-bar at this energy
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles || !alive[tid]) return;

    double E = energy[tid];
    int cell = cell_idx[tid];
    int mat = cell_material(cell);
    if (mat < 0) {
        // Void cell — zero XS
        out_macro_total[tid] = 0.0;
        return;
    }

    int n_nuc = mat_n_nuclides[mat];
    double sum_total = 0.0, sum_elastic = 0.0, sum_fission = 0.0, sum_capture = 0.0;
    double weighted_awr = 0.0;
    double weighted_nu = 0.0;

    for (int i = 0; i < n_nuc; i++) {
        int nuc_idx = mat_nuclide_idx[mat * 4 + i];
        double N_i = mat_atom_density[mat * 4 + i];

        // Find energy index for this nuclide
        int g_off = grid_offsets[nuc_idx];
        int n_e = n_energies[nuc_idx];
        int e_idx = energy_index(&all_energy_grids[g_off], n_e, E);

        // Reconstruct each reaction
        double sigma[N_REACTIONS];
        for (int r = 0; r < N_REACTIONS; r++) {
            int key = nuc_idx * N_REACTIONS + r;
            if (has_reaction[key]) {
                int b_off = basis_offsets[key];
                int c_off = coeffs_offsets[key];
                sigma[r] = svd_reconstruct(&all_basis[b_off], &all_coeffs[c_off], e_idx, rank);
            } else {
                sigma[r] = 0.0;
            }
        }

        double micro_total = sigma[RXN_ELASTIC] + sigma[RXN_INELASTIC] +
                             sigma[RXN_N2N] + sigma[RXN_N3N] +
                             sigma[RXN_FISSION] + sigma[RXN_CAPTURE];

        sum_total   += N_i * micro_total;
        sum_elastic += N_i * sigma[RXN_ELASTIC];
        sum_fission += N_i * sigma[RXN_FISSION];
        sum_capture += N_i * sigma[RXN_CAPTURE];

        // Store micro total for nuclide sampling during collision
        out_micro_total[tid * 4 + i] = N_i * micro_total;

        // Weight AWR and nu-bar by macroscopic fission XS contribution
        if (sigma[RXN_FISSION] > 0.0) {
            weighted_awr += N_i * sigma[RXN_FISSION] * awr_table[nuc_idx];
            weighted_nu  += N_i * sigma[RXN_FISSION] * nu_bar_const[nuc_idx];
        }
    }

    out_macro_total[tid] = sum_total;
    out_macro_elastic[tid] = sum_elastic;
    out_macro_fission[tid] = sum_fission;
    out_macro_capture[tid] = sum_capture;
    out_awr[tid] = (sum_fission > 0.0) ? weighted_awr / sum_fission : awr_table[0];
    out_nu_bar[tid] = (sum_fission > 0.0) ? weighted_nu / sum_fission : nu_bar_const[0];
}

// ════════════════════════════════════════════════════════════════════
// KERNEL 2: Sample collision distance + trace to nearest surface
// ════════════════════════════════════════════════════════════════════

extern "C" __global__ void sample_and_trace(
    // Particle state
    const double* pos_x, const double* pos_y, const double* pos_z,
    const double* dir_x, const double* dir_y, const double* dir_z,
    const int* cell_idx,
    const int* alive,
    int n_particles,
    // XS data from kernel 1
    const double* macro_total,
    // RNG state
    unsigned long long* rng_state,
    unsigned long long* rng_inc,
    // Output
    double* out_dist_collision,  // [N]
    double* out_dist_surface,    // [N]
    int* out_surf_bc,            // [N]
    int* out_next_cell)          // [N]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles || !alive[tid]) return;

    PcgState rng;
    rng.state = rng_state[tid];
    rng.inc = rng_inc[tid];

    double sigma_t = macro_total[tid];
    double d_coll;
    if (sigma_t > 0.0) {
        double xi = pcg_uniform(&rng);
        d_coll = -log(xi) / sigma_t;
    } else {
        d_coll = 1e20;  // infinite mean free path in void
    }

    double d_surf;
    int bc, next;
    trace_surface(pos_x[tid], pos_y[tid], pos_z[tid],
                  dir_x[tid], dir_y[tid], dir_z[tid],
                  cell_idx[tid], &d_surf, &bc, &next);

    out_dist_collision[tid] = d_coll;
    out_dist_surface[tid] = d_surf;
    out_surf_bc[tid] = bc;
    out_next_cell[tid] = next;

    // Save RNG state back
    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;
}

// ════════════════════════════════════════════════════════════════════
// KERNEL 3: Process event (advance + surface/collision)
// ════════════════════════════════════════════════════════════════════

extern "C" __global__ void process_event(
    // Particle state (read/write)
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    int* cell_idx,
    int* alive,
    int n_particles,
    // Event data from kernel 2
    const double* dist_collision,
    const double* dist_surface,
    const int* surf_bc,
    const int* next_cell,
    // XS data from kernel 1
    const double* macro_total,
    const double* macro_elastic,
    const double* macro_fission,
    const double* macro_capture,
    const double* awr,
    const double* nu_bar_arr,
    // RNG
    unsigned long long* rng_state,
    unsigned long long* rng_inc,
    // Fission bank (atomic append)
    double* fission_x, double* fission_y, double* fission_z,
    double* fission_e, double* fission_w,
    int* fission_count,    // atomic counter
    int max_fission_bank,  // capacity
    // Counters (atomics)
    int* cnt_collisions,
    int* cnt_fissions,
    int* cnt_leakage,
    int* cnt_surface_crossings)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles || !alive[tid]) return;

    PcgState rng;
    rng.state = rng_state[tid];
    rng.inc = rng_inc[tid];

    double d_coll = dist_collision[tid];
    double d_surf = dist_surface[tid];
    int cell = cell_idx[tid];

    // ── Void cell: free-stream to surface ──
    if (cell_material(cell) < 0) {
        if (d_surf > 1e19) {
            alive[tid] = 0;
            atomicAdd(cnt_leakage, 1);
            goto done;
        }
        int bc = surf_bc[tid];
        if (bc == BC_REFLECTIVE) {
            // Advance exactly to surface, reflect
            pos_x[tid] += dir_x[tid] * d_surf;
            pos_y[tid] += dir_y[tid] * d_surf;
            pos_z[tid] += dir_z[tid] * d_surf;
            // Determine which surface was hit from position
            double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
            // Reflect off the appropriate plane
            if (fabs(pz - HALF_PITCH) < 1e-6 || fabs(pz + HALF_PITCH) < 1e-6) {
                dir_z[tid] = -dir_z[tid];
            } else if (fabs(px - HALF_PITCH) < 1e-6 || fabs(px + HALF_PITCH) < 1e-6) {
                dir_x[tid] = -dir_x[tid];
            } else if (fabs(py - HALF_PITCH) < 1e-6 || fabs(py + HALF_PITCH) < 1e-6) {
                dir_y[tid] = -dir_y[tid];
            }
            atomicAdd(cnt_surface_crossings, 1);
        } else if (bc == BC_TRANSMISSION) {
            double nudge = fmax(d_surf * 1e-8, 1e-8);
            pos_x[tid] += dir_x[tid] * (d_surf + nudge);
            pos_y[tid] += dir_y[tid] * (d_surf + nudge);
            pos_z[tid] += dir_z[tid] * (d_surf + nudge);
            int nc = next_cell[tid];
            if (nc >= 0) {
                cell_idx[tid] = nc;
            } else {
                alive[tid] = 0;
                atomicAdd(cnt_leakage, 1);
            }
            atomicAdd(cnt_surface_crossings, 1);
        } else {
            alive[tid] = 0;
            atomicAdd(cnt_leakage, 1);
        }
        goto done;
    }

    // ── Surface crossing (surface is closer than collision) ──
    if (d_surf < d_coll) {
        int bc = surf_bc[tid];
        atomicAdd(cnt_surface_crossings, 1);

        if (bc == BC_REFLECTIVE) {
            // Advance exactly to surface, then reflect
            pos_x[tid] += dir_x[tid] * d_surf;
            pos_y[tid] += dir_y[tid] * d_surf;
            pos_z[tid] += dir_z[tid] * d_surf;
            double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
            // Determine reflection normal from position
            if (fabs(pz - HALF_PITCH) < 1e-6 || fabs(pz + HALF_PITCH) < 1e-6) {
                dir_z[tid] = -dir_z[tid];
            } else if (fabs(px - HALF_PITCH) < 1e-6 || fabs(px + HALF_PITCH) < 1e-6) {
                dir_x[tid] = -dir_x[tid];
            } else if (fabs(py - HALF_PITCH) < 1e-6 || fabs(py + HALF_PITCH) < 1e-6) {
                dir_y[tid] = -dir_y[tid];
            }
            // Cylinder reflections don't happen (all transmission)
        } else if (bc == BC_TRANSMISSION) {
            double nudge = fmax(d_surf * 1e-8, 1e-8);
            pos_x[tid] += dir_x[tid] * (d_surf + nudge);
            pos_y[tid] += dir_y[tid] * (d_surf + nudge);
            pos_z[tid] += dir_z[tid] * (d_surf + nudge);
            int nc = next_cell[tid];
            if (nc >= 0) {
                cell_idx[tid] = nc;
            } else {
                alive[tid] = 0;
                atomicAdd(cnt_leakage, 1);
            }
        } else {
            // Vacuum
            alive[tid] = 0;
            atomicAdd(cnt_leakage, 1);
        }
        goto done;
    }

    // ── Collision ──
    {
        atomicAdd(cnt_collisions, 1);

        // Advance to collision point
        pos_x[tid] += dir_x[tid] * d_coll;
        pos_y[tid] += dir_y[tid] * d_coll;
        pos_z[tid] += dir_z[tid] * d_coll;

        // Re-find cell (safety)
        cell_idx[tid] = find_cell(pos_x[tid], pos_y[tid], pos_z[tid]);
        if (cell_idx[tid] < 0) {
            alive[tid] = 0;
            atomicAdd(cnt_leakage, 1);
            goto done;
        }

        double sigma_t = macro_total[tid];
        if (sigma_t <= 0.0) { alive[tid] = 0; goto done; }

        double xi = pcg_uniform(&rng) * sigma_t;

        // Sample reaction type
        double sig_el = macro_elastic[tid];
        double sig_fis = macro_fission[tid];
        double sig_cap = macro_capture[tid];

        if (xi < sig_el) {
            // ── Elastic scattering (isotropic in CM) ──
            double A = awr[tid];
            double mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;  // isotropic
            double E_old = energy[tid];

            // CM to Lab energy transfer
            double alpha = ((A - 1.0) / (A + 1.0)) * ((A - 1.0) / (A + 1.0));
            double E_new = E_old * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;
            if (E_new < 1e-11) E_new = 1e-11;
            energy[tid] = E_new;

            // Lab scattering cosine
            double mu_lab = (1.0 + A * mu_cm) / sqrt(1.0 + A*A + 2.0*A*mu_cm);
            // Isotropic azimuthal angle
            double phi = 2.0 * PI * pcg_uniform(&rng);
            double sin_mu = sqrt(fmax(0.0, 1.0 - mu_lab * mu_lab));

            double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
            double w2 = dz * dz;
            if (w2 < 0.999) {
                double inv_sq = 1.0 / sqrt(1.0 - w2);
                dir_x[tid] = mu_lab * dx + sin_mu * (dx*dz*cos(phi) - dy*sin(phi)) * inv_sq;
                dir_y[tid] = mu_lab * dy + sin_mu * (dy*dz*cos(phi) + dx*sin(phi)) * inv_sq;
                dir_z[tid] = mu_lab * dz - sin_mu * sqrt(1.0 - w2) * cos(phi);
            } else {
                double sign = (dz > 0.0) ? 1.0 : -1.0;
                dir_x[tid] = sin_mu * cos(phi);
                dir_y[tid] = sin_mu * sin(phi) * sign;
                dir_z[tid] = mu_lab * sign;
            }
        } else if (xi < sig_el + sig_fis) {
            // ── Fission ──
            atomicAdd(cnt_fissions, 1);
            double nu = nu_bar_arr[tid];
            int n_sites = (int)nu;
            if (pcg_uniform(&rng) < (nu - (double)n_sites)) n_sites++;

            for (int s = 0; s < n_sites; s++) {
                int idx = atomicAdd(fission_count, 1);
                if (idx < max_fission_bank) {
                    fission_x[idx] = pos_x[tid];
                    fission_y[idx] = pos_y[tid];
                    fission_z[idx] = pos_z[tid];
                    // Watt fission spectrum: E = -a*ln(xi1) - a*ln(xi2)*cos²(pi*xi3/2)
                    // Simplified: just sample from Maxwell-like
                    double a = 0.988; // U-235 Watt parameter
                    double b = 2.249;
                    double E_fiss;
                    // Rejection sampling for Watt spectrum
                    for (int attempt = 0; attempt < 100; attempt++) {
                        double x1 = -log(pcg_uniform(&rng));
                        double x2 = -log(pcg_uniform(&rng));
                        double cosarg = cos(PI/2.0 * pcg_uniform(&rng));
                        E_fiss = a * (x1 + x2 * cosarg * cosarg);
                        if (E_fiss > 0.0) break;
                    }
                    fission_e[idx] = E_fiss * 1e6;  // Convert to eV
                    fission_w[idx] = 1.0;
                }
            }
            alive[tid] = 0;  // particle absorbed in fission
        } else {
            // ── Capture (absorption) ──
            alive[tid] = 0;
        }
    }

done:
    // Save RNG state
    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;
}

// ════════════════════════════════════════════════════════════════════
// KERNEL: Initialize source particles in fuel region
// ════════════════════════════════════════════════════════════════════

extern "C" __global__ void init_source(
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy_arr,
    int* cell_idx,
    int* alive,
    // Source bank
    const double* src_x, const double* src_y, const double* src_z,
    const double* src_e,
    int n_particles,
    // RNG seed
    unsigned long long batch_seed,
    unsigned long long* rng_state,
    unsigned long long* rng_inc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    pos_x[tid] = src_x[tid];
    pos_y[tid] = src_y[tid];
    pos_z[tid] = src_z[tid];
    energy_arr[tid] = src_e[tid];
    cell_idx[tid] = find_cell(src_x[tid], src_y[tid], src_z[tid]);
    alive[tid] = 1;

    // Initialize RNG for this particle
    PcgState rng;
    pcg_init(&rng, batch_seed + (unsigned long long)tid * 100000ULL, (unsigned long long)tid);
    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;

    // Sample isotropic direction
    double mu = 2.0 * pcg_uniform(&rng) - 1.0;
    double phi = 2.0 * PI * pcg_uniform(&rng);
    double sin_theta = sqrt(1.0 - mu * mu);
    dir_x[tid] = sin_theta * cos(phi);
    dir_y[tid] = sin_theta * sin(phi);
    dir_z[tid] = mu;

    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;
}

// ════════════════════════════════════════════════════════════════════
// KERNEL: Count alive particles (reduction)
// ════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════
// KERNEL: Energy bin counting (for sort)
// ════════════════════════════════════════════════════════════════════
// Log-scale binning: 256 bins from 1e-5 eV to 20 MeV.
// Particles with similar energies end up in the same bin, so after
// scatter they access adjacent rows of the SVD basis → coalesced reads.

#define N_ENERGY_BINS 256
// log2(1e-5) ≈ -16.6, log2(20e6) ≈ 24.3, range ≈ 41
#define LOG_E_MIN (-16.6096)
#define LOG_E_RANGE 40.9193
#define INV_LOG_STEP (N_ENERGY_BINS / LOG_E_RANGE)

__device__ int energy_to_bin(double E) {
    double log_e = log2(fmax(E, 1e-11));
    int bin = (int)((log_e - LOG_E_MIN) * INV_LOG_STEP);
    return max(0, min(N_ENERGY_BINS - 1, bin));
}

extern "C" __global__ void energy_bin_count(
    const double* energy, const int* compact_idx, int n_alive,
    int* bin_counts)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_alive) return;
    int tid = compact_idx[lane];
    atomicAdd(&bin_counts[energy_to_bin(energy[tid])], 1);
}

extern "C" __global__ void energy_bin_scatter(
    const double* energy, const int* compact_idx_in, int n_alive,
    int* compact_idx_out, int* bin_offsets)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_alive) return;
    int tid = compact_idx_in[lane];
    int pos = atomicAdd(&bin_offsets[energy_to_bin(energy[tid])], 1);
    compact_idx_out[pos] = tid;
}

// ════════════════════════════════════════════════════════════════════

extern "C" __global__ void count_alive(
    const int* alive, int n_particles, int* count)
{
    // Simple atomic count — not optimal but correct
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (alive[tid]) atomicAdd(count, 1);
}

// ════════════════════════════════════════════════════════════════════
// KERNEL: Stream compaction — build dense index of alive particles
// ════════════════════════════════════════════════════════════════════

extern "C" __global__ void compact_alive(
    const int* alive, int n_particles,
    int* compact_idx,     // output: dense list of alive particle indices
    int* compact_count)   // output: number of alive particles (atomic)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles || !alive[tid]) return;
    int pos = atomicAdd(compact_count, 1);
    compact_idx[pos] = tid;
}

// ════════════════════════════════════════════════════════════════════
// KERNEL: Fused transport step (XS lookup + trace + process)
// ════════════════════════════════════════════════════════════════════
//
// Single kernel does all three steps per particle. Intermediate data
// (macro XS, collision distance, surface distance) stays in registers
// — no global memory round-trip. This eliminates 2/3 of kernel launch
// overhead and the associated global memory traffic.
//
// Uses compact_idx for indirect indexing: only alive particles run.

extern "C" __global__ void transport_step_fused(
    // Compact index (only alive particles)
    const int* compact_idx,
    int n_alive,
    // Particle state (SoA, indexed by compact_idx)
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy,
    int* cell_idx,
    int* alive,
    // SVD data
    const float* all_basis,
    const double* all_coeffs,
    const double* all_energy_grids,
    const int* basis_offsets,
    const int* grid_offsets,
    const int* n_energies,
    const int* has_reaction,
    const int* coeffs_offsets,
    int rank,
    // Material data
    const int* mat_n_nuclides,
    const int* mat_nuclide_idx,
    const double* mat_atom_density,
    const double* awr_table,
    const double* nu_bar_const,
    // RNG
    unsigned long long* rng_state,
    unsigned long long* rng_inc,
    // Fission bank
    double* fission_x, double* fission_y, double* fission_z,
    double* fission_e, double* fission_w,
    int* fission_count,
    int max_fission_bank,
    // Counters
    int* cnt_collisions,
    int* cnt_fissions,
    int* cnt_leakage,
    int* cnt_surface_crossings)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_alive) return;
    int tid = compact_idx[lane];  // actual particle index

    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    double E = energy[tid];
    int cell = cell_idx[tid];

    PcgState rng;
    rng.state = rng_state[tid];
    rng.inc = rng_inc[tid];

    // ── Handle void cells ──
    int mat = cell_material(cell);
    if (mat < 0) {
        double d_surf; int bc, next;
        trace_surface(px, py, pz, dx, dy, dz, cell, &d_surf, &bc, &next);
        if (d_surf > 1e19) {
            alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_rng;
        }
        atomicAdd(cnt_surface_crossings, 1);
        if (bc == BC_REFLECTIVE) {
            px += dx * d_surf; py += dy * d_surf; pz += dz * d_surf;
            if (fabs(pz - HALF_PITCH) < 1e-6 || fabs(pz + HALF_PITCH) < 1e-6) dz = -dz;
            else if (fabs(px - HALF_PITCH) < 1e-6 || fabs(px + HALF_PITCH) < 1e-6) dx = -dx;
            else if (fabs(py - HALF_PITCH) < 1e-6 || fabs(py + HALF_PITCH) < 1e-6) dy = -dy;
        } else if (bc == BC_TRANSMISSION) {
            double nudge = fmax(d_surf * 1e-8, 1e-8);
            px += dx*(d_surf+nudge); py += dy*(d_surf+nudge); pz += dz*(d_surf+nudge);
            if (next >= 0) cell = next;
            else { alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_state; }
        } else { alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_rng; }
        goto save_state;
    }

    // ── Step 1: XS lookup (registers only) ──
    {
        int n_nuc = mat_n_nuclides[mat];
        double sum_total = 0.0, sum_elastic = 0.0, sum_fission = 0.0, sum_capture = 0.0;
        double best_awr = awr_table[0], best_nu = nu_bar_const[0];
        double nuc_macro[4]; // per-nuclide macroscopic total (max 4 nuclides/mat)

        for (int i = 0; i < n_nuc; i++) {
            int nuc_idx = mat_nuclide_idx[mat * 4 + i];
            double N_i = mat_atom_density[mat * 4 + i];
            int g_off = grid_offsets[nuc_idx];
            int n_e = n_energies[nuc_idx];
            int e_idx = energy_index(&all_energy_grids[g_off], n_e, E);

            double sigma[N_REACTIONS];
            for (int r = 0; r < N_REACTIONS; r++) {
                int key = nuc_idx * N_REACTIONS + r;
                if (has_reaction[key]) {
                    sigma[r] = svd_reconstruct(&all_basis[basis_offsets[key]],
                                                &all_coeffs[coeffs_offsets[key]], e_idx, rank);
                } else { sigma[r] = 0.0; }
            }
            double micro_t = sigma[0]+sigma[1]+sigma[2]+sigma[3]+sigma[4]+sigma[5];
            double macro_t = N_i * micro_t;
            nuc_macro[i] = macro_t;
            sum_total += macro_t;
            sum_elastic += N_i * sigma[RXN_ELASTIC];
            sum_fission += N_i * sigma[RXN_FISSION];
            sum_capture += N_i * sigma[RXN_CAPTURE];
            if (sigma[RXN_FISSION] > 0.0) {
                best_awr = awr_table[nuc_idx];
                best_nu = nu_bar_const[nuc_idx];
            }
        }

        if (sum_total <= 0.0) { alive[tid] = 0; goto save_rng; }

        // ── Step 2: Sample collision distance + trace ──
        double d_coll = -log(pcg_uniform(&rng)) / sum_total;
        double d_surf; int bc, next;
        trace_surface(px, py, pz, dx, dy, dz, cell, &d_surf, &bc, &next);

        // ── Step 3: Process event ──
        if (d_surf < d_coll) {
            // Surface crossing
            atomicAdd(cnt_surface_crossings, 1);
            if (bc == BC_REFLECTIVE) {
                px += dx*d_surf; py += dy*d_surf; pz += dz*d_surf;
                if (fabs(pz - HALF_PITCH) < 1e-6 || fabs(pz + HALF_PITCH) < 1e-6) dz = -dz;
                else if (fabs(px - HALF_PITCH) < 1e-6 || fabs(px + HALF_PITCH) < 1e-6) dx = -dx;
                else if (fabs(py - HALF_PITCH) < 1e-6 || fabs(py + HALF_PITCH) < 1e-6) dy = -dy;
            } else if (bc == BC_TRANSMISSION) {
                double nudge = fmax(d_surf * 1e-8, 1e-8);
                px += dx*(d_surf+nudge); py += dy*(d_surf+nudge); pz += dz*(d_surf+nudge);
                if (next >= 0) cell = next;
                else { alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_state; }
            } else { alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_state; }
        } else {
            // Collision
            atomicAdd(cnt_collisions, 1);
            px += dx*d_coll; py += dy*d_coll; pz += dz*d_coll;
            cell = find_cell(px, py, pz);
            if (cell < 0) { alive[tid] = 0; atomicAdd(cnt_leakage, 1); goto save_state; }

            double xi = pcg_uniform(&rng) * sum_total;
            if (xi < sum_elastic) {
                // Elastic scatter
                double A = best_awr;
                double mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
                double alpha = ((A-1.0)/(A+1.0))*((A-1.0)/(A+1.0));
                E = E * (1.0 + alpha + (1.0-alpha)*mu_cm) / 2.0;
                if (E < 1e-11) E = 1e-11;

                double mu_lab = (1.0 + A*mu_cm) / sqrt(1.0 + A*A + 2.0*A*mu_cm);
                double phi = 2.0 * PI * pcg_uniform(&rng);
                double sin_mu = sqrt(fmax(0.0, 1.0 - mu_lab*mu_lab));
                double w2 = dz*dz;
                if (w2 < 0.999) {
                    double inv_sq = 1.0 / sqrt(1.0 - w2);
                    double dx2 = mu_lab*dx + sin_mu*(dx*dz*cos(phi) - dy*sin(phi))*inv_sq;
                    double dy2 = mu_lab*dy + sin_mu*(dy*dz*cos(phi) + dx*sin(phi))*inv_sq;
                    double dz2 = mu_lab*dz - sin_mu*sqrt(1.0-w2)*cos(phi);
                    dx=dx2; dy=dy2; dz=dz2;
                } else {
                    double sign = (dz > 0.0) ? 1.0 : -1.0;
                    dx = sin_mu*cos(phi); dy = sin_mu*sin(phi)*sign; dz = mu_lab*sign;
                }
            } else if (xi < sum_elastic + sum_fission) {
                // Fission
                atomicAdd(cnt_fissions, 1);
                double nu = best_nu;
                int n_sites = (int)nu;
                if (pcg_uniform(&rng) < (nu - (double)n_sites)) n_sites++;
                for (int s = 0; s < n_sites; s++) {
                    int idx = atomicAdd(fission_count, 1);
                    if (idx < max_fission_bank) {
                        fission_x[idx] = px; fission_y[idx] = py; fission_z[idx] = pz;
                        double a = 0.988, E_fiss;
                        for (int att = 0; att < 100; att++) {
                            double x1 = -log(pcg_uniform(&rng));
                            double x2 = -log(pcg_uniform(&rng));
                            double c2 = cos(PI/2.0*pcg_uniform(&rng));
                            E_fiss = a * (x1 + x2*c2*c2);
                            if (E_fiss > 0.0) break;
                        }
                        fission_e[idx] = E_fiss * 1e6;
                        fission_w[idx] = 1.0;
                    }
                }
                alive[tid] = 0; goto save_state;
            } else {
                // Capture
                alive[tid] = 0; goto save_state;
            }
        }
    }

save_state:
    pos_x[tid] = px; pos_y[tid] = py; pos_z[tid] = pz;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    energy[tid] = E;
    cell_idx[tid] = cell;
save_rng:
    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;
}

"#;

// ── Rust-side GPU transport context ──────────────────────────────

/// Compiled CUDA kernels for event-based transport.
pub struct GpuTransportContext {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // Original 3-kernel approach
    k_xs_lookup: CudaFunction,
    k_sample_trace: CudaFunction,
    k_process_event: CudaFunction,
    // Shared
    k_init_source: CudaFunction,
    k_count_alive: CudaFunction,
    // Optimized: fused kernel + compaction + energy sort
    k_transport_fused: CudaFunction,
    k_compact_alive: CudaFunction,
    k_energy_bin_count: CudaFunction,
    k_energy_bin_scatter: CudaFunction,
}

/// SVD data uploaded to GPU for all nuclides.
pub struct GpuNuclideData {
    pub all_basis: CudaSlice<f32>,
    pub all_coeffs: CudaSlice<f64>,
    pub all_energy_grids: CudaSlice<f64>,
    pub basis_offsets: CudaSlice<i32>,
    pub grid_offsets: CudaSlice<i32>,
    pub n_energies: CudaSlice<i32>,
    pub has_reaction: CudaSlice<i32>,
    pub coeffs_offsets: CudaSlice<i32>,
    pub rank: i32,
}

/// Material composition data on GPU.
pub struct GpuMaterialData {
    pub mat_n_nuclides: CudaSlice<i32>,
    pub mat_nuclide_idx: CudaSlice<i32>,
    pub mat_atom_density: CudaSlice<f64>,
    pub awr_table: CudaSlice<f64>,
    pub nu_bar_const: CudaSlice<f64>,
}

/// Result of one batch on GPU.
pub struct GpuBatchResult {
    pub k_eff: f64,
    pub collisions: u32,
    pub fissions: u32,
    pub leakage: u32,
    pub surface_crossings: u32,
    /// Fission sites for next generation.
    pub fission_bank: Vec<(f64, f64, f64, f64)>, // (x, y, z, energy)
}

const BLOCK_SIZE: u32 = 256;

impl GpuTransportContext {
    /// Compile all CUDA kernels and initialize GPU context.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let ptx = nvrtc::compile_ptx(TRANSPORT_KERNELS)?;
        let module = ctx.load_module(ptx)?;

        let k_xs_lookup = module.load_function("xs_lookup")?;
        let k_sample_trace = module.load_function("sample_and_trace")?;
        let k_process_event = module.load_function("process_event")?;
        let k_init_source = module.load_function("init_source")?;
        let k_count_alive = module.load_function("count_alive")?;
        let k_transport_fused = module.load_function("transport_step_fused")?;
        let k_compact_alive = module.load_function("compact_alive")?;
        let k_energy_bin_count = module.load_function("energy_bin_count")?;
        let k_energy_bin_scatter = module.load_function("energy_bin_scatter")?;
        let stream = ctx.default_stream();

        println!("  GPU transport kernels compiled (9 kernels)");

        Ok(Self {
            _ctx: ctx, stream,
            k_xs_lookup, k_sample_trace, k_process_event,
            k_init_source, k_count_alive,
            k_transport_fused, k_compact_alive,
            k_energy_bin_count, k_energy_bin_scatter,
        })
    }

    /// Upload SVD nuclide data to GPU.
    pub fn upload_nuclide_data(
        &self,
        nuclides: &[crate::transport::xs_provider::NuclideKernels],
        rank: usize,
    ) -> Result<GpuNuclideData, Box<dyn std::error::Error>> {
        let n_nuc = nuclides.len();
        let n_rxn = 6; // elastic, inelastic, n2n, n3n, fission, capture

        // Concatenate all basis, coefficients, and energy grids
        let mut all_basis_vec: Vec<f32> = Vec::new();
        let mut all_coeffs_vec: Vec<f64> = Vec::new();
        let mut all_grids_vec: Vec<f64> = Vec::new();
        let mut basis_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut coeffs_offsets_vec = vec![0_i32; n_nuc * n_rxn];
        let mut grid_offsets_vec = vec![0_i32; n_nuc];
        let mut n_energies_vec = vec![0_i32; n_nuc];
        let mut has_reaction_vec = vec![0_i32; n_nuc * n_rxn];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            // Energy grid (shared across reactions for this nuclide)
            let grid_offset = all_grids_vec.len();
            grid_offsets_vec[nuc_idx] = grid_offset as i32;

            // Get energy grid from any available reaction
            let any_kernel = nuc.elastic.as_ref()
                .or(nuc.fission.as_ref())
                .or(nuc.capture.as_ref())
                .or(nuc.inelastic.as_ref())
                .or(nuc.n2n.as_ref())
                .or(nuc.n3n.as_ref());

            if let Some(rk) = any_kernel {
                all_grids_vec.extend_from_slice(&rk.kernel.energies);
                n_energies_vec[nuc_idx] = rk.kernel.n_energy() as i32;
            }

            // Each reaction
            let reactions: [Option<&crate::transport::xs_provider::ReactionKernel>; 6] = [
                nuc.elastic.as_ref(),
                nuc.inelastic.as_ref(),
                nuc.n2n.as_ref(),
                nuc.n3n.as_ref(),
                nuc.fission.as_ref(),
                nuc.capture.as_ref(),
            ];

            for (rxn_idx, rxn_opt) in reactions.iter().enumerate() {
                let key = nuc_idx * n_rxn + rxn_idx;
                if let Some(rk) = rxn_opt {
                    has_reaction_vec[key] = 1;
                    basis_offsets_vec[key] = all_basis_vec.len() as i32;
                    all_basis_vec.extend_from_slice(rk.kernel.basis_f32());
                    coeffs_offsets_vec[key] = all_coeffs_vec.len() as i32;
                    all_coeffs_vec.extend_from_slice(&rk.coeffs);
                } else {
                    basis_offsets_vec[key] = 0;
                    coeffs_offsets_vec[key] = 0;
                }
            }
        }

        // Ensure we have data
        if all_basis_vec.is_empty() { all_basis_vec.push(0.0); }
        if all_coeffs_vec.is_empty() { all_coeffs_vec.push(0.0); }
        if all_grids_vec.is_empty() { all_grids_vec.push(0.0); }

        println!("  GPU: basis={:.1} MB, grids={:.1} MB, coeffs={:.1} KB",
                 all_basis_vec.len() as f64 * 4.0 / 1e6,
                 all_grids_vec.len() as f64 * 8.0 / 1e6,
                 all_coeffs_vec.len() as f64 * 8.0 / 1e3);

        Ok(GpuNuclideData {
            all_basis: self.stream.clone_htod(&all_basis_vec)?,
            all_coeffs: self.stream.clone_htod(&all_coeffs_vec)?,
            all_energy_grids: self.stream.clone_htod(&all_grids_vec)?,
            basis_offsets: self.stream.clone_htod(&basis_offsets_vec)?,
            grid_offsets: self.stream.clone_htod(&grid_offsets_vec)?,
            n_energies: self.stream.clone_htod(&n_energies_vec)?,
            has_reaction: self.stream.clone_htod(&has_reaction_vec)?,
            coeffs_offsets: self.stream.clone_htod(&coeffs_offsets_vec)?,
            rank: rank as i32,
        })
    }

    /// Upload material composition data to GPU.
    pub fn upload_material_data(
        &self,
        materials: &[crate::transport::material::Material],
        nuclide_awrs: &[f64],
        nuclide_nu_bars: &[f64],
    ) -> Result<GpuMaterialData, Box<dyn std::error::Error>> {
        let max_nuc = 4; // max nuclides per material
        let n_mat = materials.len();

        let mut n_nuclides = vec![0_i32; n_mat];
        let mut nuc_idx = vec![0_i32; n_mat * max_nuc];
        let mut atom_dens = vec![0.0_f64; n_mat * max_nuc];

        for (m, mat) in materials.iter().enumerate() {
            n_nuclides[m] = mat.nuclides.len() as i32;
            for (i, nuc) in mat.nuclides.iter().enumerate() {
                nuc_idx[m * max_nuc + i] = nuc.xs_kernel_idx as i32;
                atom_dens[m * max_nuc + i] = nuc.atom_density;
            }
        }

        Ok(GpuMaterialData {
            mat_n_nuclides: self.stream.clone_htod(&n_nuclides)?,
            mat_nuclide_idx: self.stream.clone_htod(&nuc_idx)?,
            mat_atom_density: self.stream.clone_htod(&atom_dens)?,
            awr_table: self.stream.clone_htod(nuclide_awrs)?,
            nu_bar_const: self.stream.clone_htod(nuclide_nu_bars)?,
        })
    }

    /// Run one batch of transport on GPU.
    ///
    /// Returns batch result with k_eff and fission bank.
    pub fn run_batch(
        &self,
        source_bank: &[(f64, f64, f64, f64)], // (x, y, z, energy)
        batch: u32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        max_events_per_particle: u32,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error>> {
        let n = source_bank.len();
        let n_i32 = n as i32;
        let grid = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // Unpack source bank into SoA
        let (sx, sy, sz, se): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = source_bank.iter()
            .map(|&(x, y, z, e)| (x, y, z, e))
            .fold((vec![], vec![], vec![], vec![]), |mut acc, (x, y, z, e)| {
                acc.0.push(x); acc.1.push(y); acc.2.push(z); acc.3.push(e); acc
            });

        // Upload source bank
        let d_src_x = self.stream.clone_htod(&sx)?;
        let d_src_y = self.stream.clone_htod(&sy)?;
        let d_src_z = self.stream.clone_htod(&sz)?;
        let d_src_e = self.stream.clone_htod(&se)?;

        // Allocate particle state arrays
        let mut d_pos_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_energy: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_alive: CudaSlice<i32> = self.stream.alloc_zeros(n)?;

        // RNG state arrays
        let mut d_rng_state: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let mut d_rng_inc: CudaSlice<u64> = self.stream.alloc_zeros(n)?;

        // XS output arrays
        let mut d_macro_total: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_macro_elastic: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_macro_fission: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_macro_capture: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_micro_total: CudaSlice<f64> = self.stream.alloc_zeros(n * 4)?;
        let mut d_awr: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_nu_bar: CudaSlice<f64> = self.stream.alloc_zeros(n)?;

        // Trace output arrays
        let mut d_dist_coll: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dist_surf: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_surf_bc: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_next_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;

        // Fission bank (pre-allocated, 3x source size should be enough)
        let max_fission = (n * 3) as i32;
        let mut d_fis_x: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_y: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_z: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_e: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_w: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Counters
        let mut d_cnt_coll: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_fis: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_leak: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_surf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // ── Initialize source ──
        let batch_seed = batch as u64 * 1_000_000;
        unsafe {
            self.stream.launch_builder(&self.k_init_source)
                .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                .arg(&d_src_x).arg(&d_src_y).arg(&d_src_z).arg(&d_src_e)
                .arg(&n_i32)
                .arg(&batch_seed)
                .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                .launch(cfg)?;
        }

        // ── Event loop ──
        let check_interval = 50; // Only check alive count every N steps
        for step in 0..max_events_per_particle {
            // Only poll alive count periodically to avoid GPU→CPU sync overhead
            if step % check_interval == 0 && step > 0 {
                let mut d_alive_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
                unsafe {
                    self.stream.launch_builder(&self.k_count_alive)
                        .arg(&d_alive).arg(&n_i32).arg(&mut d_alive_count)
                        .launch(cfg)?;
                }
                let alive_count = self.stream.clone_dtoh(&d_alive_count)?;
                if alive_count[0] == 0 { break; }
            }

            // Step 1: XS lookup
            unsafe {
                self.stream.launch_builder(&self.k_xs_lookup)
                    .arg(&d_energy).arg(&d_cell).arg(&d_alive).arg(&n_i32)
                    .arg(&nuc_data.all_basis)
                    .arg(&nuc_data.all_coeffs)
                    .arg(&nuc_data.all_energy_grids)
                    .arg(&nuc_data.basis_offsets)
                    .arg(&nuc_data.grid_offsets)
                    .arg(&nuc_data.n_energies)
                    .arg(&nuc_data.has_reaction)
                    .arg(&nuc_data.coeffs_offsets)
                    .arg(&nuc_data.rank)
                    .arg(&mat_data.mat_n_nuclides)
                    .arg(&mat_data.mat_nuclide_idx)
                    .arg(&mat_data.mat_atom_density)
                    .arg(&mat_data.awr_table)
                    .arg(&mat_data.nu_bar_const)
                    .arg(&mut d_macro_total)
                    .arg(&mut d_macro_elastic)
                    .arg(&mut d_macro_fission)
                    .arg(&mut d_macro_capture)
                    .arg(&mut d_micro_total)
                    .arg(&mut d_awr)
                    .arg(&mut d_nu_bar)
                    .launch(cfg)?;
            }

            // Step 2: Sample distance + trace
            unsafe {
                self.stream.launch_builder(&self.k_sample_trace)
                    .arg(&d_pos_x).arg(&d_pos_y).arg(&d_pos_z)
                    .arg(&d_dir_x).arg(&d_dir_y).arg(&d_dir_z)
                    .arg(&d_cell).arg(&d_alive).arg(&n_i32)
                    .arg(&d_macro_total)
                    .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                    .arg(&mut d_dist_coll).arg(&mut d_dist_surf)
                    .arg(&mut d_surf_bc).arg(&mut d_next_cell)
                    .launch(cfg)?;
            }

            // Step 3: Process event
            unsafe {
                self.stream.launch_builder(&self.k_process_event)
                    .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                    .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                    .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                    .arg(&n_i32)
                    .arg(&d_dist_coll).arg(&d_dist_surf)
                    .arg(&d_surf_bc).arg(&d_next_cell)
                    .arg(&d_macro_total).arg(&d_macro_elastic)
                    .arg(&d_macro_fission).arg(&d_macro_capture)
                    .arg(&d_awr).arg(&d_nu_bar)
                    .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                    .arg(&mut d_fis_x).arg(&mut d_fis_y).arg(&mut d_fis_z)
                    .arg(&mut d_fis_e).arg(&mut d_fis_w)
                    .arg(&mut d_fis_count).arg(&max_fission)
                    .arg(&mut d_cnt_coll).arg(&mut d_cnt_fis)
                    .arg(&mut d_cnt_leak).arg(&mut d_cnt_surf)
                    .launch(cfg)?;
            }
        }

        // ── Download results ──
        let fis_count = self.stream.clone_dtoh(&d_fis_count)?[0] as usize;
        let cnt_coll = self.stream.clone_dtoh(&d_cnt_coll)?[0] as u32;
        let cnt_fis = self.stream.clone_dtoh(&d_cnt_fis)?[0] as u32;
        let cnt_leak = self.stream.clone_dtoh(&d_cnt_leak)?[0] as u32;
        let cnt_surf = self.stream.clone_dtoh(&d_cnt_surf)?[0] as u32;

        let fis_count_clamped = fis_count.min(max_fission as usize);
        let fission_bank = if fis_count_clamped > 0 {
            let fx = self.stream.clone_dtoh(&d_fis_x)?;
            let fy = self.stream.clone_dtoh(&d_fis_y)?;
            let fz = self.stream.clone_dtoh(&d_fis_z)?;
            let fe = self.stream.clone_dtoh(&d_fis_e)?;
            (0..fis_count_clamped)
                .map(|i| (fx[i], fy[i], fz[i], fe[i]))
                .collect()
        } else {
            vec![]
        };

        let k_eff = fission_bank.len() as f64 / n as f64;

        Ok(GpuBatchResult {
            k_eff,
            collisions: cnt_coll,
            fissions: cnt_fis,
            leakage: cnt_leak,
            surface_crossings: cnt_surf,
            fission_bank,
        })
    }

    /// Optimized batch: fused kernel + stream compaction.
    ///
    /// Single kernel per step (XS + trace + process in registers).
    /// Compaction removes dead particles between steps so only alive
    /// particles consume GPU threads.
    pub fn run_batch_fused(
        &self,
        source_bank: &[(f64, f64, f64, f64)],
        batch: u32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        max_steps: u32,
    ) -> Result<GpuBatchResult, Box<dyn std::error::Error>> {
        let n = source_bank.len();
        let n_i32 = n as i32;
        let grid_full = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg_full = LaunchConfig {
            grid_dim: (grid_full, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // Unpack source bank into SoA
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        let mut sz = Vec::with_capacity(n);
        let mut se = Vec::with_capacity(n);
        for &(x, y, z, e) in source_bank {
            sx.push(x); sy.push(y); sz.push(z); se.push(e);
        }

        let d_src_x = self.stream.clone_htod(&sx)?;
        let d_src_y = self.stream.clone_htod(&sy)?;
        let d_src_z = self.stream.clone_htod(&sz)?;
        let d_src_e = self.stream.clone_htod(&se)?;

        // Pre-allocate all particle state arrays (reused across steps)
        let mut d_pos_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_pos_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_x: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_y: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_dir_z: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_energy: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let mut d_cell: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_alive: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_rng_state: CudaSlice<u64> = self.stream.alloc_zeros(n)?;
        let mut d_rng_inc: CudaSlice<u64> = self.stream.alloc_zeros(n)?;

        // Compaction + sort buffers
        let mut d_compact_idx: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let mut d_compact_idx_sorted: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
        let n_bins = 256;
        // Fission bank
        let max_fission = (n * 3) as i32;
        let mut d_fis_x: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_y: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_z: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_e: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_w: CudaSlice<f64> = self.stream.alloc_zeros(max_fission as usize)?;
        let mut d_fis_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Counters
        let mut d_cnt_coll: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_fis: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_leak: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
        let mut d_cnt_surf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;

        // Initialize source
        let batch_seed = batch as u64 * 1_000_000;
        unsafe {
            self.stream.launch_builder(&self.k_init_source)
                .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                .arg(&d_src_x).arg(&d_src_y).arg(&d_src_z).arg(&d_src_e)
                .arg(&n_i32)
                .arg(&batch_seed)
                .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                .launch(cfg_full)?;
        }

        let mut n_alive = n as i32;
        let compact_interval = 10; // Re-compact every N steps

        for step in 0..max_steps {
            if n_alive <= 0 { break; }

            // Compact + sort alive particles periodically
            if step % compact_interval == 0 {
                // 1. Compact: build dense list of alive particle indices
                let mut d_compact_count: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
                let compact_grid = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let compact_cfg = LaunchConfig {
                    grid_dim: (compact_grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.stream.launch_builder(&self.k_compact_alive)
                        .arg(&d_alive).arg(&n_i32)
                        .arg(&mut d_compact_idx).arg(&mut d_compact_count)
                        .launch(compact_cfg)?;
                }
                let count = self.stream.clone_dtoh(&d_compact_count)?;
                n_alive = count[0];
                if n_alive <= 0 { break; }

                // 2. Energy sort: bin count → prefix sum → scatter
                let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let alive_cfg = LaunchConfig {
                    grid_dim: (alive_grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };

                // 2a. Count particles per energy bin
                let mut d_bin_counts: CudaSlice<i32> = self.stream.alloc_zeros(n_bins)?;
                unsafe {
                    self.stream.launch_builder(&self.k_energy_bin_count)
                        .arg(&d_energy).arg(&d_compact_idx).arg(&n_alive)
                        .arg(&mut d_bin_counts)
                        .launch(alive_cfg)?;
                }

                // 2b. Prefix sum on CPU (256 ints — trivial)
                let counts = self.stream.clone_dtoh(&d_bin_counts)?;
                let mut offsets = vec![0_i32; n_bins];
                let mut running = 0_i32;
                for i in 0..n_bins {
                    offsets[i] = running;
                    running += counts[i];
                }
                let d_bin_offsets = self.stream.clone_htod(&offsets)?;

                // 2c. Scatter compact indices into energy-sorted order
                unsafe {
                    self.stream.launch_builder(&self.k_energy_bin_scatter)
                        .arg(&d_energy).arg(&d_compact_idx).arg(&n_alive)
                        .arg(&mut d_compact_idx_sorted).arg(&d_bin_offsets)
                        .launch(alive_cfg)?;
                }

                // Swap: sorted becomes the active compact index
                std::mem::swap(&mut d_compact_idx, &mut d_compact_idx_sorted);
            }

            // Launch fused kernel with (possibly sorted) compact index
            let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let alive_cfg = LaunchConfig {
                grid_dim: (alive_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&self.k_transport_fused)
                    .arg(&d_compact_idx)
                    .arg(&n_alive)
                    .arg(&mut d_pos_x).arg(&mut d_pos_y).arg(&mut d_pos_z)
                    .arg(&mut d_dir_x).arg(&mut d_dir_y).arg(&mut d_dir_z)
                    .arg(&mut d_energy).arg(&mut d_cell).arg(&mut d_alive)
                    .arg(&nuc_data.all_basis)
                    .arg(&nuc_data.all_coeffs)
                    .arg(&nuc_data.all_energy_grids)
                    .arg(&nuc_data.basis_offsets)
                    .arg(&nuc_data.grid_offsets)
                    .arg(&nuc_data.n_energies)
                    .arg(&nuc_data.has_reaction)
                    .arg(&nuc_data.coeffs_offsets)
                    .arg(&nuc_data.rank)
                    .arg(&mat_data.mat_n_nuclides)
                    .arg(&mat_data.mat_nuclide_idx)
                    .arg(&mat_data.mat_atom_density)
                    .arg(&mat_data.awr_table)
                    .arg(&mat_data.nu_bar_const)
                    .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                    .arg(&mut d_fis_x).arg(&mut d_fis_y).arg(&mut d_fis_z)
                    .arg(&mut d_fis_e).arg(&mut d_fis_w)
                    .arg(&mut d_fis_count).arg(&max_fission)
                    .arg(&mut d_cnt_coll).arg(&mut d_cnt_fis)
                    .arg(&mut d_cnt_leak).arg(&mut d_cnt_surf)
                    .launch(alive_cfg)?;
            }
        }

        // Download results
        let fis_count = self.stream.clone_dtoh(&d_fis_count)?[0] as usize;
        let cnt_coll = self.stream.clone_dtoh(&d_cnt_coll)?[0] as u32;
        let cnt_fis = self.stream.clone_dtoh(&d_cnt_fis)?[0] as u32;
        let cnt_leak = self.stream.clone_dtoh(&d_cnt_leak)?[0] as u32;
        let cnt_surf = self.stream.clone_dtoh(&d_cnt_surf)?[0] as u32;

        let fis_count_clamped = fis_count.min(max_fission as usize);
        let fission_bank = if fis_count_clamped > 0 {
            let fx = self.stream.clone_dtoh(&d_fis_x)?;
            let fy = self.stream.clone_dtoh(&d_fis_y)?;
            let fz = self.stream.clone_dtoh(&d_fis_z)?;
            let fe = self.stream.clone_dtoh(&d_fis_e)?;
            (0..fis_count_clamped)
                .map(|i| (fx[i], fy[i], fz[i], fe[i]))
                .collect()
        } else {
            vec![]
        };

        let k_eff = fission_bank.len() as f64 / n as f64;

        Ok(GpuBatchResult {
            k_eff,
            collisions: cnt_coll,
            fissions: cnt_fis,
            leakage: cnt_leak,
            surface_crossings: cnt_surf,
            fission_bank,
        })
    }
}
