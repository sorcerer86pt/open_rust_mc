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

// Geometry types
#define GEOM_PWR    0
#define GEOM_GODIVA 1

// Godiva: bare HEU sphere
#define GODIVA_RADIUS 8.7407

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
__device__ int find_cell(double x, double y, double z, int geom_type) {
    if (geom_type == GEOM_GODIVA) {
        double r2 = x*x + y*y + z*z;
        return (r2 < GODIVA_RADIUS * GODIVA_RADIUS) ? 0 : -1;
    }
    // PWR pin cell
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
__device__ int cell_material(int cell, int geom_type) {
    if (geom_type == GEOM_GODIVA) return (cell == 0) ? 0 : -1;
    // PWR
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
    int cell, int geom_type,
    double* out_dist, int* out_bc, int* out_next_cell)
{
    if (geom_type == GEOM_GODIVA) {
        // Sphere intersection: |p + t*d|^2 = R^2
        double a = dx*dx + dy*dy + dz*dz;
        double b = 2.0*(px*dx + py*dy + pz*dz);
        double c = px*px + py*py + pz*pz - GODIVA_RADIUS*GODIVA_RADIUS;
        double disc = b*b - 4.0*a*c;
        if (disc < 0.0) { *out_dist = 1e20; *out_bc = BC_VACUUM; *out_next_cell = -1; return; }
        double sq = sqrt(disc);
        double t1 = (-b - sq) / (2.0*a);
        double t2 = (-b + sq) / (2.0*a);
        double t = (t1 > COINCIDENCE_TOL) ? t1 : ((t2 > COINCIDENCE_TOL) ? t2 : 1e20);
        *out_dist = t;
        *out_bc = BC_VACUUM;
        *out_next_cell = -1;
        return;
    }

    // PWR pin cell geometry
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
        best_next = find_cell(nx, ny, nz, geom_type);
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

// Reconstruct one cross-section from SVD basis + coefficients.
// Uses __ldg() for read-only cache (texture path) on basis data.
__device__ double svd_reconstruct(
    const float* __restrict__ basis,    // [n_e * rank]
    const double* __restrict__ coeffs,  // [rank]
    int e_idx, int rank)
{
    const float* row = &basis[e_idx * rank];
    double acc = 0.0;
    for (int j = 0; j < rank; j++) {
        acc = fma((double)__ldg(&row[j]), __ldg(&coeffs[j]), acc);
    }
    return exp2(acc * 3.321928094887362);  // log10->linear
}

// ════════════════════════════════════════════════════════════════════
// Physics helper functions (full parity with CPU)
// ════════════════════════════════════════════════════════════════════

// ── Energy-dependent nu-bar (linear interpolation on table) ──
__device__ double nu_bar_lookup(
    double E,
    const double* __restrict__ energies,
    const double* __restrict__ values,
    int offset, int n_pts)
{
    if (n_pts <= 0) return 0.0;
    const double* e = &energies[offset];
    const double* v = &values[offset];
    if (E <= e[0]) return v[0];
    if (E >= e[n_pts-1]) return v[n_pts-1];
    // Binary search
    int lo = 0, hi = n_pts - 1;
    while (hi - lo > 1) { int mid=(lo+hi)/2; if (e[mid]<=E) lo=mid; else hi=mid; }
    double f = (E - e[lo]) / (e[hi] - e[lo]);
    return v[lo] + f * (v[hi] - v[lo]);
}

// ── Fission spectrum: sample outgoing energy from tabulated CDF ──
__device__ double sample_fission_energy(
    double E_inc, PcgState* rng,
    const double* __restrict__ fis_inc_energies,  // incident energy grid
    const int* __restrict__ fis_dist_offsets,      // per inc energy → offset into e_out/cdf
    const int* __restrict__ fis_dist_sizes,        // per inc energy → n_eout
    const double* __restrict__ fis_e_out,          // outgoing energies (flat)
    const double* __restrict__ fis_cdf,            // CDF values (flat)
    int nuc_fis_offset,  // offset into fis_inc_energies for this nuclide
    int n_inc)           // number of incident energies for this nuclide
{
    if (n_inc <= 0) {
        // Fallback: Watt spectrum
        double a = 0.988;
        double x1 = -log(fmax(pcg_uniform(rng), 1e-30));
        double x2 = -log(fmax(pcg_uniform(rng), 1e-30));
        double c = cos(PI/2.0 * pcg_uniform(rng));
        return a * (x1 + x2*c*c) * 1e6;
    }

    const double* inc_e = &fis_inc_energies[nuc_fis_offset];

    // Find bracketing incident energy
    int ie = 0;
    if (E_inc >= inc_e[n_inc-1]) ie = n_inc-1;
    else {
        for (int i = 0; i < n_inc-1; i++) {
            if (E_inc >= inc_e[i] && E_inc < inc_e[i+1]) { ie = i; break; }
        }
    }

    // Sample from CDF at this incident energy
    int off = fis_dist_offsets[nuc_fis_offset + ie];
    int sz = fis_dist_sizes[nuc_fis_offset + ie];
    if (sz <= 1) return E_inc * 0.5;

    double xi = pcg_uniform(rng);
    const double* eo = &fis_e_out[off];
    const double* cd = &fis_cdf[off];

    // Binary search on CDF
    int lo = 0, hi = sz - 1;
    while (hi - lo > 1) { int mid=(lo+hi)/2; if (cd[mid]<=xi) lo=mid; else hi=mid; }
    double f = (xi - cd[lo]) / fmax(cd[hi] - cd[lo], 1e-30);
    return eo[lo] + f * (eo[hi] - eo[lo]);
}

// ── Anisotropic scattering: sample mu from tabulated CDF ──
__device__ double sample_angular_dist(
    double E, PcgState* rng,
    const double* __restrict__ ang_energies,
    const int* __restrict__ ang_dist_offsets,
    const int* __restrict__ ang_dist_sizes,
    const double* __restrict__ ang_mu,
    const double* __restrict__ ang_cdf,
    int nuc_ang_offset,
    int n_ang_e,
    int is_cm)  // 1=center-of-mass, 0=lab
{
    if (n_ang_e <= 0) return 2.0*pcg_uniform(rng) - 1.0;  // isotropic fallback

    const double* ae = &ang_energies[nuc_ang_offset];

    // Find bracketing energy
    int ie = 0;
    if (E <= ae[0]) ie = 0;
    else if (E >= ae[n_ang_e-1]) ie = n_ang_e-1;
    else {
        int lo=0, hi=n_ang_e-1;
        while (hi-lo>1) { int mid=(lo+hi)/2; if (ae[mid]<=E) lo=mid; else hi=mid; }
        ie = lo;
    }

    int off = ang_dist_offsets[nuc_ang_offset + ie];
    int sz = ang_dist_sizes[nuc_ang_offset + ie];
    if (sz <= 1) return 2.0*pcg_uniform(rng) - 1.0;

    double xi = pcg_uniform(rng);
    const double* mu_arr = &ang_mu[off];
    const double* cd = &ang_cdf[off];

    int lo=0, hi=sz-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (cd[mid]<=xi) lo=mid; else hi=mid; }
    double f = (xi - cd[lo]) / fmax(cd[hi] - cd[lo], 1e-30);
    double mu = mu_arr[lo] + f * (mu_arr[hi] - mu_arr[lo]);
    return fmax(-1.0, fmin(1.0, mu));
}

// ── URR probability tables: sample band and modify XS ──
__device__ void apply_urr_gpu(
    double* total, double* elastic, double* fission, double* capture,
    double E, double xi,
    const double* __restrict__ urr_energies,
    const double* __restrict__ urr_cum_prob,
    const double* __restrict__ urr_total_f,
    const double* __restrict__ urr_elastic_f,
    const double* __restrict__ urr_fission_f,
    const double* __restrict__ urr_capture_f,
    int urr_offset, int n_urr_e, int n_bands, int multiply_smooth)
{
    if (n_urr_e <= 0) return;
    const double* ue = &urr_energies[urr_offset];
    if (E < ue[0] || E > ue[n_urr_e-1]) return;

    // Find energy index
    int ie = 0;
    int lo=0, hi=n_urr_e-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (ue[mid]<=E) lo=mid; else hi=mid; }
    ie = lo;

    // Sample band from cumulative probability
    int base = urr_offset * n_bands + ie * n_bands;  // simplified offset
    // Actually need proper flattened offset: urr_offset_bands + ie * n_bands
    const double* cp = &urr_cum_prob[base];
    int band = 0;
    for (int b = 0; b < n_bands; b++) {
        if (xi < cp[b]) { band = b; break; }
        band = b;
    }

    double ft = urr_total_f[base + band];
    double fe = urr_elastic_f[base + band];
    double ff = urr_fission_f[base + band];
    double fc = urr_capture_f[base + band];

    if (multiply_smooth) {
        *total *= ft; *elastic *= fe; *fission *= ff; *capture *= fc;
    } else {
        *total = ft; *elastic = fe; *fission = ff; *capture = fc;
    }
}

// ── S(α,β) thermal scattering ──
// Continuous inelastic: sample (E_out, mu) from CDF tables
__device__ void sab_sample(
    double E_in, PcgState* rng,
    double* E_out, double* mu_out,
    const double* __restrict__ sab_inc_energies,  // incident energy grid
    int n_sab_inc,
    const int* __restrict__ sab_eout_offsets,     // per inc energy → offset into e_out arrays
    const int* __restrict__ sab_eout_sizes,       // per inc energy → n_eout
    const double* __restrict__ sab_e_out,         // outgoing energies (flat)
    const double* __restrict__ sab_cdf_e,         // energy CDF (flat)
    const int* __restrict__ sab_mu_offsets,       // per (inc_e, eout) → offset into mu arrays
    const int* __restrict__ sab_mu_sizes,         // per (inc_e, eout) → n_mu
    const double* __restrict__ sab_mu,            // discrete cosines (flat)
    const double* __restrict__ sab_cdf_mu)        // cosine CDF (flat)
{
    if (n_sab_inc <= 0) { *E_out = E_in; *mu_out = 2.0*pcg_uniform(rng)-1.0; return; }

    // Find bracketing incident energy
    int ie = 0;
    if (E_in <= sab_inc_energies[0]) ie = 0;
    else if (E_in >= sab_inc_energies[n_sab_inc-1]) ie = n_sab_inc-1;
    else {
        int lo=0, hi=n_sab_inc-1;
        while (hi-lo>1) { int mid=(lo+hi)/2; if (sab_inc_energies[mid]<=E_in) lo=mid; else hi=mid; }
        // Stochastic interpolation between lo and hi
        double f = (E_in - sab_inc_energies[lo]) / fmax(sab_inc_energies[hi]-sab_inc_energies[lo], 1e-30);
        ie = (pcg_uniform(rng) < f) ? hi : lo;
    }

    // Sample outgoing energy from CDF
    int eo_off = sab_eout_offsets[ie];
    int eo_sz = sab_eout_sizes[ie];
    if (eo_sz <= 1) { *E_out = E_in; *mu_out = 2.0*pcg_uniform(rng)-1.0; return; }

    double xi_e = pcg_uniform(rng);
    const double* eo = &sab_e_out[eo_off];
    const double* cdf_e = &sab_cdf_e[eo_off];

    int lo=0, hi=eo_sz-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (cdf_e[mid]<=xi_e) lo=mid; else hi=mid; }
    double f_e = (xi_e - cdf_e[lo]) / fmax(cdf_e[hi]-cdf_e[lo], 1e-30);
    *E_out = eo[lo] + f_e * (eo[hi] - eo[lo]);
    if (*E_out < 1e-11) *E_out = 1e-11;

    int eout_bin = lo;  // which outgoing energy bin we sampled

    // Sample cosine from mu CDF at this (inc_energy, eout_bin)
    int mu_key = eo_off + eout_bin;  // linearized index for mu lookup
    int mu_off = sab_mu_offsets[mu_key];
    int mu_sz = sab_mu_sizes[mu_key];
    if (mu_sz <= 1) { *mu_out = 2.0*pcg_uniform(rng)-1.0; return; }

    double xi_mu = pcg_uniform(rng);
    const double* mu_arr = &sab_mu[mu_off];
    const double* cdf_mu = &sab_cdf_mu[mu_off];

    lo=0; hi=mu_sz-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (cdf_mu[mid]<=xi_mu) lo=mid; else hi=mid; }
    double f_mu = (xi_mu - cdf_mu[lo]) / fmax(cdf_mu[hi]-cdf_mu[lo], 1e-30);
    *mu_out = mu_arr[lo] + f_mu * (mu_arr[hi] - mu_arr[lo]);
    *mu_out = fmax(-1.0, fmin(1.0, *mu_out));
}

// S(α,β) total XS at a given energy (sum over outgoing energy PDF)
__device__ double sab_total_xs(
    double E_in,
    const double* __restrict__ sab_inc_energies,
    const double* __restrict__ sab_xs,  // total XS at each incident energy
    int n_sab_inc)
{
    if (n_sab_inc <= 0) return 0.0;
    if (E_in <= sab_inc_energies[0]) return sab_xs[0];
    if (E_in >= sab_inc_energies[n_sab_inc-1]) return sab_xs[n_sab_inc-1];
    int lo=0, hi=n_sab_inc-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (sab_inc_energies[mid]<=E_in) lo=mid; else hi=mid; }
    double f = (E_in - sab_inc_energies[lo]) / fmax(sab_inc_energies[hi]-sab_inc_energies[lo], 1e-30);
    return sab_xs[lo] + f * (sab_xs[hi] - sab_xs[lo]);
}

// ════════════════════════════════════════════════════════════════════
// Utility kernels
// ════════════════════════════════════════════════════════════════════

// Energy binning for sorted compaction (256 log-scale bins)
#define N_ENERGY_BINS 256
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

extern "C" __global__ void count_alive(
    const int* alive, int n_particles, int* count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (alive[tid]) atomicAdd(count, 1);
}

extern "C" __global__ void compact_alive(
    const int* alive, int n_particles,
    int* compact_idx, int* compact_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles || !alive[tid]) return;
    int pos = atomicAdd(compact_count, 1);
    compact_idx[pos] = tid;
}

extern "C" __global__ void init_source(
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy_arr,
    int* cell_idx,
    int* alive,
    const double* src_x, const double* src_y, const double* src_z,
    const double* src_e,
    int n_particles,
    unsigned long long batch_seed,
    unsigned long long* rng_state,
    unsigned long long* rng_inc,
    int geom_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    pos_x[tid] = src_x[tid];
    pos_y[tid] = src_y[tid];
    pos_z[tid] = src_z[tid];
    energy_arr[tid] = src_e[tid];
    cell_idx[tid] = find_cell(src_x[tid], src_y[tid], src_z[tid], geom_type);
    alive[tid] = 1;

    PcgState rng;
    pcg_init(&rng, batch_seed + (unsigned long long)tid * 100000ULL, (unsigned long long)tid);
    rng_state[tid] = rng.state;
    rng_inc[tid] = rng.inc;

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
// KERNEL: Persistent transport (multiple steps per launch)
// ════════════════════════════════════════════════════════════════════
//
// Combines all optimizations:
//   - Particle state in registers across N steps (no global memory per step)
//   - Warp-level reduction for counters (__shfl_down_sync)
//   - __ldg() for read-only SVD data
//   - Local counter accumulation (one warp atomic per N steps)
//   - __restrict__ hints for pointer aliasing

extern "C" __global__ void __launch_bounds__(256, 2)
transport_persistent(
    const int* __restrict__ compact_idx, int n_alive,
    double* __restrict__ pos_x, double* __restrict__ pos_y, double* __restrict__ pos_z,
    double* __restrict__ dir_x, double* __restrict__ dir_y, double* __restrict__ dir_z,
    double* __restrict__ energy,
    int* __restrict__ cell_idx,
    int* __restrict__ alive,
    const float* __restrict__ all_basis,
    const double* __restrict__ all_coeffs,
    const double* __restrict__ all_energy_grids,
    const int* __restrict__ basis_offsets,
    const int* __restrict__ grid_offsets,
    const int* __restrict__ n_energies,
    const int* __restrict__ has_reaction,
    const int* __restrict__ coeffs_offsets,
    int rank,
    const int* __restrict__ mat_n_nuclides,
    const int* __restrict__ mat_nuclide_idx,
    const double* __restrict__ mat_atom_density,
    const double* __restrict__ awr_table,
    const double* __restrict__ nu_bar_const,
    // Energy-dependent nu-bar tables (flat with offsets)
    const double* __restrict__ nu_bar_energies,
    const double* __restrict__ nu_bar_values,
    const int* __restrict__ nu_bar_offsets,
    const int* __restrict__ nu_bar_sizes,
    // Fission energy distribution (flat CDFs)
    const double* __restrict__ fis_inc_energies,
    const int* __restrict__ fis_dist_offsets,
    const int* __restrict__ fis_dist_sizes,
    const double* __restrict__ fis_e_out,
    const double* __restrict__ fis_cdf,
    const int* __restrict__ fis_nuc_offsets,
    const int* __restrict__ fis_nuc_n_inc,
    // S(α,β) thermal scattering (for H1, nuclide idx 3)
    const double* __restrict__ sab_inc_energies,
    int n_sab_inc,
    const int* __restrict__ sab_eout_offsets,
    const int* __restrict__ sab_eout_sizes,
    const double* __restrict__ sab_e_out_arr,
    const double* __restrict__ sab_cdf_e_arr,
    const int* __restrict__ sab_mu_offsets_arr,
    const int* __restrict__ sab_mu_sizes_arr,
    const double* __restrict__ sab_mu_arr,
    const double* __restrict__ sab_cdf_mu_arr,
    const double* __restrict__ sab_xs_arr,
    double sab_energy_max,
    // Discrete inelastic levels
    const double* __restrict__ level_q_values,
    const double* __restrict__ level_thresholds,
    const int* __restrict__ level_offsets,
    const int* __restrict__ level_counts,
    const float* __restrict__ level_basis,
    const double* __restrict__ level_coeffs,
    const int* __restrict__ level_basis_offsets,
    const int* __restrict__ level_coeffs_offsets,
    const int* __restrict__ level_has_kernel,
    // RNG
    unsigned long long* __restrict__ rng_state_arr,
    unsigned long long* __restrict__ rng_inc_arr,
    double* __restrict__ fission_x, double* __restrict__ fission_y,
    double* __restrict__ fission_z, double* __restrict__ fission_e,
    double* __restrict__ fission_w,
    int* fission_count, int max_fission_bank,
    int* cnt_collisions, int* cnt_fissions, int* cnt_leakage, int* cnt_surface_crossings,
    int steps_per_launch,
    int geom_type)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_alive) return;
    int tid = compact_idx[lane];

    // Load particle state into registers
    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    double E = energy[tid];
    int cell = cell_idx[tid];
    int is_alive = alive[tid];

    PcgState rng;
    rng.state = rng_state_arr[tid];
    rng.inc = rng_inc_arr[tid];

    // Local counters — accumulated across steps, reduced at end
    int lcnt_coll = 0, lcnt_fis = 0, lcnt_leak = 0, lcnt_surf = 0;

    for (int step = 0; step < steps_per_launch && is_alive; step++) {
        int mat = cell_material(cell, geom_type);

        // ── Void: stream to surface ──
        if (mat < 0) {
            double d_surf; int bc, next;
            trace_surface(px, py, pz, dx, dy, dz, cell, geom_type, &d_surf, &bc, &next);
            if (d_surf > 1e19) { is_alive = 0; lcnt_leak++; break; }
            lcnt_surf++;
            if (bc == BC_REFLECTIVE) {
                px += dx*d_surf; py += dy*d_surf; pz += dz*d_surf;
                if (fabs(pz-HALF_PITCH)<1e-6 || fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if (fabs(px-HALF_PITCH)<1e-6 || fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if (fabs(py-HALF_PITCH)<1e-6 || fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc == BC_TRANSMISSION) {
                double nudge = fmax(d_surf*1e-8, 1e-8);
                px+=dx*(d_surf+nudge); py+=dy*(d_surf+nudge); pz+=dz*(d_surf+nudge);
                if (next >= 0) cell = next;
                else { is_alive = 0; lcnt_leak++; break; }
            } else { is_alive = 0; lcnt_leak++; break; }
            continue;
        }

        // ── XS lookup (all in registers) ──
        // Store per-nuclide macro XS for nuclide sampling
        int n_nuc = __ldg(&mat_n_nuclides[mat]);
        double sum_total=0, sum_elastic=0, sum_fission=0, sum_capture=0;
        double nuc_macro_t[4] = {0,0,0,0}; // per-nuclide macroscopic total
        double nuc_sig_el[4] = {0,0,0,0};  // per-nuclide macro elastic
        double nuc_sig_fis[4] = {0,0,0,0}; // per-nuclide macro fission
        double nuc_sig_cap[4] = {0,0,0,0}; // per-nuclide macro capture

        for (int i = 0; i < n_nuc; i++) {
            int nuc_idx = __ldg(&mat_nuclide_idx[mat*4+i]);
            double N_i = __ldg(&mat_atom_density[mat*4+i]);
            int g_off = __ldg(&grid_offsets[nuc_idx]);
            int n_e = __ldg(&n_energies[nuc_idx]);
            int e_idx = energy_index(&all_energy_grids[g_off], n_e, E);

            double sig_el=0, sig_fis=0, sig_cap=0, sig_rest=0;
            for (int r = 0; r < N_REACTIONS; r++) {
                int key = nuc_idx * N_REACTIONS + r;
                if (__ldg(&has_reaction[key])) {
                    double s = svd_reconstruct(
                        &all_basis[__ldg(&basis_offsets[key])],
                        &all_coeffs[__ldg(&coeffs_offsets[key])], e_idx, rank);
                    if (r == RXN_ELASTIC)  sig_el = s;
                    else if (r == RXN_FISSION) sig_fis = s;
                    else if (r == RXN_CAPTURE) sig_cap = s;
                    else sig_rest += s;
                }
            }
            // S(α,β) thermal scattering: replace elastic for H1 below energy_max
            double sab_xs_val = 0.0;
            if (nuc_idx == 3 && E < sab_energy_max && E > 0.0 && n_sab_inc > 0) {
                sab_xs_val = sab_total_xs(E, sab_inc_energies, sab_xs_arr, n_sab_inc);
                if (sab_xs_val > 0.0) {
                    sig_el = sab_xs_val;  // replace free-gas elastic with S(α,β)
                }
            }

            double micro_t = sig_el + sig_fis + sig_cap + sig_rest;
            nuc_macro_t[i] = N_i * micro_t;
            nuc_sig_el[i]  = N_i * sig_el;
            nuc_sig_fis[i] = N_i * sig_fis;
            nuc_sig_cap[i] = N_i * sig_cap;
            sum_total   += N_i * micro_t;
            sum_elastic += N_i * sig_el;
            sum_fission += N_i * sig_fis;
            sum_capture += N_i * sig_cap;
        }

        if (sum_total <= 0.0) { is_alive = 0; break; }

        // ── Sample + trace ──
        double d_coll = -log(pcg_uniform(&rng)) / sum_total;
        double d_surf; int bc, next;
        trace_surface(px, py, pz, dx, dy, dz, cell, geom_type, &d_surf, &bc, &next);

        // ── Process event ──
        if (d_surf < d_coll) {
            lcnt_surf++;
            if (bc == BC_REFLECTIVE) {
                px+=dx*d_surf; py+=dy*d_surf; pz+=dz*d_surf;
                if (fabs(pz-HALF_PITCH)<1e-6||fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if (fabs(px-HALF_PITCH)<1e-6||fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if (fabs(py-HALF_PITCH)<1e-6||fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc == BC_TRANSMISSION) {
                double nudge = fmax(d_surf*1e-8, 1e-8);
                px+=dx*(d_surf+nudge); py+=dy*(d_surf+nudge); pz+=dz*(d_surf+nudge);
                if (next >= 0) cell = next;
                else { is_alive = 0; lcnt_leak++; break; }
            } else { is_alive = 0; lcnt_leak++; break; }
        } else {
            lcnt_coll++;
            px+=dx*d_coll; py+=dy*d_coll; pz+=dz*d_coll;
            cell = find_cell(px, py, pz, geom_type);
            if (cell < 0) { is_alive = 0; lcnt_leak++; break; }

            // ── Sample which NUCLIDE was hit (proportional to macro XS) ──
            double xi_nuc = pcg_uniform(&rng) * sum_total;
            double cumul = 0.0;
            int hit_local = 0;  // local index within material
            for (int i = 0; i < n_nuc; i++) {
                cumul += nuc_macro_t[i];
                if (xi_nuc < cumul) { hit_local = i; break; }
            }
            int hit_nuc_idx = __ldg(&mat_nuclide_idx[mat*4+hit_local]);
            double A = __ldg(&awr_table[hit_nuc_idx]);

            // ── Sample reaction type for this nuclide ──
            double xi_rxn = pcg_uniform(&rng) * nuc_macro_t[hit_local];
            if (xi_rxn < nuc_sig_el[hit_local]) {
                // ══ Elastic/thermal scatter with correct AWR ══

                // S(α,β) thermal scattering for H1 below energy_max
                if (hit_nuc_idx == 3 && E < sab_energy_max && n_sab_inc > 0) {
                    double E_sab, mu_sab;
                    sab_sample(E, &rng, &E_sab, &mu_sab,
                               sab_inc_energies, n_sab_inc,
                               sab_eout_offsets, sab_eout_sizes,
                               sab_e_out_arr, sab_cdf_e_arr,
                               sab_mu_offsets_arr, sab_mu_sizes_arr,
                               sab_mu_arr, sab_cdf_mu_arr);
                    E = fmax(E_sab, 1e-11);
                    // Apply scattering angle
                    double phi = 2.0*PI*pcg_uniform(&rng);
                    double sin_mu = sqrt(fmax(0.0, 1.0-mu_sab*mu_sab));
                    double w2 = dz*dz;
                    if (w2 < 0.999) {
                        double inv_sq = 1.0/sqrt(1.0-w2);
                        double dx2=mu_sab*dx+sin_mu*(dx*dz*cos(phi)-dy*sin(phi))*inv_sq;
                        double dy2=mu_sab*dy+sin_mu*(dy*dz*cos(phi)+dx*sin(phi))*inv_sq;
                        double dz2=mu_sab*dz-sin_mu*sqrt(1.0-w2)*cos(phi);
                        dx=dx2; dy=dy2; dz=dz2;
                    } else {
                        double sign=(dz>0.0)?1.0:-1.0;
                        dx=sin_mu*cos(phi); dy=sin_mu*sin(phi)*sign; dz=mu_sab*sign;
                    }
                    goto end_collision;
                }

                // Cell temperatures: fuel=900K, clad=600K, water=600K
                double cell_kT;
                if (cell == CELL_FUEL) cell_kT = 900.0 * 8.617333262e-5;
                else cell_kT = 600.0 * 8.617333262e-5; // clad & water

                // Free-gas thermal scattering for E < 400*kT
                // (target nucleus has thermal motion — critical for H-1)
                if (E < 400.0 * cell_kT && A < 10.0) {
                    // Box-Muller target velocity sampling
                    double sigma = sqrt(cell_kT / A);
                    double v_n = sqrt(2.0 * E);

                    // Target velocity components (Box-Muller)
                    double u1 = pcg_uniform(&rng), u2 = pcg_uniform(&rng);
                    double r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30)));
                    double theta_bm = 2.0 * PI * u2;
                    double vt_x = r_bm * cos(theta_bm);
                    double vt_y = r_bm * sin(theta_bm);
                    u1 = pcg_uniform(&rng); u2 = pcg_uniform(&rng);
                    r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30)));
                    theta_bm = 2.0 * PI * u2;
                    double vt_z = r_bm * cos(theta_bm);

                    // Neutron velocity in lab
                    double vn_x = dx * v_n, vn_y = dy * v_n, vn_z = dz * v_n;

                    // Relative velocity
                    double vr_x = vn_x - vt_x, vr_y = vn_y - vt_y, vr_z = vn_z - vt_z;
                    double v_rel = sqrt(vr_x*vr_x + vr_y*vr_y + vr_z*vr_z);
                    if (v_rel < 1e-20) v_rel = 1e-20;

                    // CM velocity
                    double inv_ap1 = 1.0 / (1.0 + A);
                    double vcm_x = (vn_x + A*vt_x) * inv_ap1;
                    double vcm_y = (vn_y + A*vt_y) * inv_ap1;
                    double vcm_z = (vn_z + A*vt_z) * inv_ap1;

                    // Neutron speed in CM = v_rel * A/(A+1)
                    double v_cm_n = v_rel * A * inv_ap1;

                    // Isotropic scatter in CM
                    double mu_cm = 2.0*pcg_uniform(&rng) - 1.0;
                    double phi = 2.0*PI*pcg_uniform(&rng);
                    double sin_t = sqrt(fmax(0.0, 1.0-mu_cm*mu_cm));

                    // Scattered direction in CM (relative to v_rel direction)
                    double vr_hat_x = vr_x/v_rel, vr_hat_y = vr_y/v_rel, vr_hat_z = vr_z/v_rel;
                    // Build orthonormal basis from vr_hat
                    double abs_z = fabs(vr_hat_z);
                    double perp_x, perp_y, perp_z;
                    if (abs_z < 0.999) {
                        double inv_p = 1.0/sqrt(1.0-vr_hat_z*vr_hat_z);
                        perp_x = -vr_hat_y*inv_p; perp_y = vr_hat_x*inv_p; perp_z = 0.0;
                    } else {
                        double inv_p = 1.0/sqrt(1.0-vr_hat_x*vr_hat_x);
                        perp_x = 0.0; perp_y = -vr_hat_z*inv_p; perp_z = vr_hat_y*inv_p;
                    }
                    // cross product for third basis vector
                    double perp2_x = vr_hat_y*perp_z - vr_hat_z*perp_y;
                    double perp2_y = vr_hat_z*perp_x - vr_hat_x*perp_z;
                    double perp2_z = vr_hat_x*perp_y - vr_hat_y*perp_x;

                    double scat_x = mu_cm*vr_hat_x + sin_t*(cos(phi)*perp_x + sin(phi)*perp2_x);
                    double scat_y = mu_cm*vr_hat_y + sin_t*(cos(phi)*perp_y + sin(phi)*perp2_y);
                    double scat_z = mu_cm*vr_hat_z + sin_t*(cos(phi)*perp_z + sin(phi)*perp2_z);

                    // Lab velocity = CM velocity + scattered CM neutron velocity
                    double vout_x = vcm_x + v_cm_n * scat_x;
                    double vout_y = vcm_y + v_cm_n * scat_y;
                    double vout_z = vcm_z + v_cm_n * scat_z;

                    double v_out = sqrt(vout_x*vout_x + vout_y*vout_y + vout_z*vout_z);
                    E = 0.5 * v_out * v_out;
                    if (E < 1e-11) E = 1e-11;
                    if (v_out > 1e-20) {
                        dx = vout_x/v_out; dy = vout_y/v_out; dz = vout_z/v_out;
                    }
                } else {
                    // Standard two-body elastic (stationary target)
                    double mu_cm = 2.0*pcg_uniform(&rng) - 1.0;
                    double alpha = ((A-1.0)/(A+1.0))*((A-1.0)/(A+1.0));
                    E = E * (1.0+alpha+(1.0-alpha)*mu_cm) / 2.0;
                    if (E < 1e-11) E = 1e-11;
                    double mu_lab = (1.0+A*mu_cm)/sqrt(1.0+A*A+2.0*A*mu_cm);
                    double phi = 2.0*PI*pcg_uniform(&rng);
                    double sin_mu = sqrt(fmax(0.0, 1.0-mu_lab*mu_lab));
                    double w2 = dz*dz;
                    if (w2 < 0.999) {
                        double inv_sq = 1.0/sqrt(1.0-w2);
                        double dx2=mu_lab*dx+sin_mu*(dx*dz*cos(phi)-dy*sin(phi))*inv_sq;
                        double dy2=mu_lab*dy+sin_mu*(dy*dz*cos(phi)+dx*sin(phi))*inv_sq;
                        double dz2=mu_lab*dz-sin_mu*sqrt(1.0-w2)*cos(phi);
                        dx=dx2; dy=dy2; dz=dz2;
                    } else {
                        double sign = (dz>0.0)?1.0:-1.0;
                        dx=sin_mu*cos(phi); dy=sin_mu*sin(phi)*sign; dz=mu_lab*sign;
                    }
                }
            } else if (xi_rxn < nuc_sig_el[hit_local]+nuc_sig_fis[hit_local]) {
                // Fission — energy-dependent nu-bar from table
                lcnt_fis++;
                int nb_off = __ldg(&nu_bar_offsets[hit_nuc_idx]);
                int nb_sz = __ldg(&nu_bar_sizes[hit_nuc_idx]);
                double nu;
                if (nb_sz > 0) {
                    nu = nu_bar_lookup(E, nu_bar_energies, nu_bar_values, nb_off, nb_sz);
                } else {
                    nu = __ldg(&nu_bar_const[hit_nuc_idx]);
                }
                int n_sites = (int)nu;
                if (pcg_uniform(&rng) < (nu-(double)n_sites)) n_sites++;
                for (int s = 0; s < n_sites; s++) {
                    int idx = atomicAdd(fission_count, 1);
                    if (idx < max_fission_bank) {
                        fission_x[idx]=px; fission_y[idx]=py; fission_z[idx]=pz;
                        // Data-driven fission spectrum from tabulated CDF
                        int fi_off = __ldg(&fis_nuc_offsets[hit_nuc_idx]);
                        int fi_ninc = __ldg(&fis_nuc_n_inc[hit_nuc_idx]);
                        double E_fiss = sample_fission_energy(
                            E, &rng, fis_inc_energies, fis_dist_offsets,
                            fis_dist_sizes, fis_e_out, fis_cdf, fi_off, fi_ninc);
                        fission_e[idx] = E_fiss;
                        fission_w[idx] = 1.0;
                    }
                }
                is_alive = 0;
            } else if (xi_rxn < nuc_sig_el[hit_local]+nuc_sig_fis[hit_local]+nuc_sig_cap[hit_local]) {
                // Capture — absorbed
                is_alive = 0;
            } else {
                // Inelastic / (n,2n) / (n,3n) — proper discrete level sampling
                // Reconstruct per-level XS, sample proportional to cross-section
                int lv_off = __ldg(&level_offsets[hit_nuc_idx]);
                int n_levels = __ldg(&level_counts[hit_nuc_idx]);

                double Q = -0.5e6; // fallback if no levels
                if (n_levels > 0) {
                    // Reconstruct level XS and build CDF
                    double level_xs_sum = 0.0;
                    double level_xs_cum[64]; // max 64 levels
                    int n_active = 0;
                    for (int l = 0; l < n_levels && l < 64; l++) {
                        int gl = lv_off + l; // global level index
                        double lxs = 0.0;
                        if (E >= __ldg(&level_thresholds[gl]) && __ldg(&level_has_kernel[gl])) {
                            int bo = __ldg(&level_basis_offsets[gl]);
                            int co = __ldg(&level_coeffs_offsets[gl]);
                            // Use parent nuclide energy index (already computed)
                            int g_off_nuc = __ldg(&grid_offsets[hit_nuc_idx]);
                            int n_e_nuc = __ldg(&n_energies[hit_nuc_idx]);
                            int e_idx_l = energy_index(&all_energy_grids[g_off_nuc], n_e_nuc, E);
                            lxs = svd_reconstruct(&level_basis[bo], &level_coeffs[co], e_idx_l, rank);
                        }
                        level_xs_sum += lxs;
                        level_xs_cum[l] = level_xs_sum;
                        n_active++;
                    }

                    // Sample level from CDF
                    if (level_xs_sum > 0.0) {
                        double xi_lev = pcg_uniform(&rng) * level_xs_sum;
                        int selected = 0;
                        for (int l = 0; l < n_active; l++) {
                            if (xi_lev < level_xs_cum[l]) { selected = l; break; }
                            selected = l;
                        }
                        Q = __ldg(&level_q_values[lv_off + selected]);
                    }
                }

                // Two-body kinematics with selected Q-value
                double A_ratio = A / (A + 1.0);
                double E_out = E * A_ratio * A_ratio + Q * (A + 1.0) / A;
                if (E_out <= 0.0) E_out = E * 0.01; // below threshold fallback
                E = fmax(E_out, 1e-11);

                // Isotropic scattering in CM frame
                double mu_cm = 2.0*pcg_uniform(&rng) - 1.0;
                double phi = 2.0*PI*pcg_uniform(&rng);
                double sin_t = sqrt(fmax(0.0, 1.0 - mu_cm*mu_cm));
                dx = sin_t*cos(phi); dy = sin_t*sin(phi); dz = mu_cm;
            }
            end_collision: ;
        }
    } // end step loop

    // Write back state
    pos_x[tid]=px; pos_y[tid]=py; pos_z[tid]=pz;
    dir_x[tid]=dx; dir_y[tid]=dy; dir_z[tid]=dz;
    energy[tid]=E; cell_idx[tid]=cell; alive[tid]=is_alive;
    rng_state_arr[tid]=rng.state; rng_inc_arr[tid]=rng.inc;

    // Warp-level reduction: sum counters across warp, one atomic per warp
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2) {
        lcnt_coll += __shfl_down_sync(mask, lcnt_coll, offset);
        lcnt_fis  += __shfl_down_sync(mask, lcnt_fis,  offset);
        lcnt_leak += __shfl_down_sync(mask, lcnt_leak, offset);
        lcnt_surf += __shfl_down_sync(mask, lcnt_surf, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        if (lcnt_coll > 0) atomicAdd(cnt_collisions, lcnt_coll);
        if (lcnt_fis > 0)  atomicAdd(cnt_fissions, lcnt_fis);
        if (lcnt_leak > 0) atomicAdd(cnt_leakage, lcnt_leak);
        if (lcnt_surf > 0) atomicAdd(cnt_surface_crossings, lcnt_surf);
    }
}

"#;

// ── Rust-side GPU transport context ──────────────────────────────

/// Compiled CUDA kernels for event-based transport.
pub struct GpuTransportContext {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    k_init_source: CudaFunction,
    k_count_alive: CudaFunction,
    k_compact_alive: CudaFunction,
    k_energy_bin_count: CudaFunction,
    k_energy_bin_scatter: CudaFunction,
    k_transport_persistent: CudaFunction,
}

/// SVD data + physics tables uploaded to GPU for all nuclides.
pub struct GpuNuclideData {
    // SVD basis data
    pub all_basis: CudaSlice<f32>,
    pub all_coeffs: CudaSlice<f64>,
    pub all_energy_grids: CudaSlice<f64>,
    pub basis_offsets: CudaSlice<i32>,
    pub grid_offsets: CudaSlice<i32>,
    pub n_energies: CudaSlice<i32>,
    pub has_reaction: CudaSlice<i32>,
    pub coeffs_offsets: CudaSlice<i32>,
    pub rank: i32,
    // Energy-dependent nu-bar tables
    pub nu_bar_energies: CudaSlice<f64>,
    pub nu_bar_values: CudaSlice<f64>,
    pub nu_bar_offsets: CudaSlice<i32>,
    pub nu_bar_sizes: CudaSlice<i32>,
    // Discrete inelastic levels (Q-values + SVD basis for XS-proportional sampling)
    pub level_q_values: CudaSlice<f64>,      // flat: all Q-values concatenated
    pub level_thresholds: CudaSlice<f64>,    // flat: all thresholds concatenated
    pub level_offsets: CudaSlice<i32>,        // per-nuclide offset into level arrays
    pub level_counts: CudaSlice<i32>,         // per-nuclide number of levels
    pub level_basis: CudaSlice<f32>,          // flat: SVD basis for each level's XS
    pub level_coeffs: CudaSlice<f64>,         // flat: SVD coefficients for each level
    pub level_basis_offsets: CudaSlice<i32>,  // per-level offset into level_basis
    pub level_coeffs_offsets: CudaSlice<i32>, // per-level offset into level_coeffs
    pub level_has_kernel: CudaSlice<i32>,     // per-level: 1 if kernel exists, 0 if not
    // Fission energy distributions (tabulated CDF)
    pub fis_inc_energies: CudaSlice<f64>,
    pub fis_dist_offsets: CudaSlice<i32>,
    pub fis_dist_sizes: CudaSlice<i32>,
    pub fis_e_out: CudaSlice<f64>,
    pub fis_cdf: CudaSlice<f64>,
    pub fis_nuc_offsets: CudaSlice<i32>,
    pub fis_nuc_n_inc: CudaSlice<i32>,
}

/// S(α,β) thermal scattering data on GPU (for one temperature).
pub struct GpuSabData {
    pub inc_energies: CudaSlice<f64>,
    pub n_inc: i32,
    pub eout_offsets: CudaSlice<i32>,
    pub eout_sizes: CudaSlice<i32>,
    pub e_out: CudaSlice<f64>,
    pub cdf_e: CudaSlice<f64>,
    pub mu_offsets: CudaSlice<i32>,
    pub mu_sizes: CudaSlice<i32>,
    pub mu: CudaSlice<f64>,
    pub cdf_mu: CudaSlice<f64>,
    pub xs: CudaSlice<f64>,
    pub energy_max: f64,
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

        let k_init_source = module.load_function("init_source")?;
        let k_count_alive = module.load_function("count_alive")?;
        let k_compact_alive = module.load_function("compact_alive")?;
        let k_energy_bin_count = module.load_function("energy_bin_count")?;
        let k_energy_bin_scatter = module.load_function("energy_bin_scatter")?;
        let k_transport_persistent = module.load_function("transport_persistent")?;
        let stream = ctx.default_stream();

        println!("  GPU transport kernels compiled (6 kernels)");

        Ok(Self {
            _ctx: ctx, stream,
            k_init_source, k_count_alive, k_compact_alive,
            k_energy_bin_count, k_energy_bin_scatter,
            k_transport_persistent,
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

        // ── Pack discrete inelastic levels (Q-values + SVD basis) ──
        let mut lev_q_vec: Vec<f64> = Vec::new();
        let mut lev_thr_vec: Vec<f64> = Vec::new();
        let mut lev_off_vec = vec![0_i32; n_nuc];
        let mut lev_cnt_vec = vec![0_i32; n_nuc];
        let mut lev_basis_vec: Vec<f32> = Vec::new();
        let mut lev_coeffs_vec: Vec<f64> = Vec::new();
        let mut lev_basis_off_vec: Vec<i32> = Vec::new();
        let mut lev_coeffs_off_vec: Vec<i32> = Vec::new();
        let mut lev_has_kernel_vec: Vec<i32> = Vec::new();

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            lev_off_vec[nuc_idx] = lev_q_vec.len() as i32;
            lev_cnt_vec[nuc_idx] = nuc.discrete_levels.len() as i32;
            for lev in &nuc.discrete_levels {
                lev_q_vec.push(lev.info.q_value);
                lev_thr_vec.push(lev.info.threshold);
                if let Some(ref rk) = lev.kernel {
                    lev_has_kernel_vec.push(1);
                    lev_basis_off_vec.push(lev_basis_vec.len() as i32);
                    lev_basis_vec.extend_from_slice(rk.kernel.basis_f32());
                    lev_coeffs_off_vec.push(lev_coeffs_vec.len() as i32);
                    lev_coeffs_vec.extend_from_slice(&rk.coeffs);
                } else {
                    lev_has_kernel_vec.push(0);
                    lev_basis_off_vec.push(0);
                    lev_coeffs_off_vec.push(0);
                }
            }
        }
        if lev_q_vec.is_empty() {
            lev_q_vec.push(0.0); lev_thr_vec.push(0.0);
            lev_has_kernel_vec.push(0); lev_basis_off_vec.push(0); lev_coeffs_off_vec.push(0);
        }
        if lev_basis_vec.is_empty() { lev_basis_vec.push(0.0); }
        if lev_coeffs_vec.is_empty() { lev_coeffs_vec.push(0.0); }

        let n_total_levels: usize = lev_cnt_vec.iter().map(|&c| c as usize).sum();
        println!("  GPU: {} discrete levels, {:.1} MB level basis",
                 n_total_levels, lev_basis_vec.len() as f64 * 4.0 / 1e6);

        // ── Pack nu-bar tables (flat with offsets) ──
        let mut nb_energies_vec: Vec<f64> = Vec::new();
        let mut nb_values_vec: Vec<f64> = Vec::new();
        let mut nb_offsets_vec = vec![0_i32; n_nuc];
        let mut nb_sizes_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref t) = nuc.nu_bar_table {
                if !t.energies.is_empty() {
                    nb_offsets_vec[nuc_idx] = nb_energies_vec.len() as i32;
                    nb_sizes_vec[nuc_idx] = t.energies.len() as i32;
                    nb_energies_vec.extend_from_slice(&t.energies);
                    nb_values_vec.extend_from_slice(&t.values);
                }
            }
        }
        if nb_energies_vec.is_empty() { nb_energies_vec.push(0.0); nb_values_vec.push(0.0); }

        // ── Pack fission energy distributions (flat CDFs with offsets) ──
        let mut fis_inc_e_vec: Vec<f64> = Vec::new();
        let mut fis_dist_off_vec: Vec<i32> = Vec::new();
        let mut fis_dist_sz_vec: Vec<i32> = Vec::new();
        let mut fis_eout_vec: Vec<f64> = Vec::new();
        let mut fis_cdf_vec: Vec<f64> = Vec::new();
        let mut fis_nuc_off_vec = vec![0_i32; n_nuc];
        let mut fis_nuc_ninc_vec = vec![0_i32; n_nuc];

        for (nuc_idx, nuc) in nuclides.iter().enumerate() {
            if let Some(ref edist) = nuc.fission_energy_dist {
                fis_nuc_off_vec[nuc_idx] = fis_inc_e_vec.len() as i32;
                fis_nuc_ninc_vec[nuc_idx] = edist.energies.len() as i32;
                for (i, e_inc) in edist.energies.iter().enumerate() {
                    fis_inc_e_vec.push(*e_inc);
                    let dist = &edist.distributions[i];
                    fis_dist_off_vec.push(fis_eout_vec.len() as i32);
                    fis_dist_sz_vec.push(dist.e_out.len() as i32);
                    fis_eout_vec.extend_from_slice(&dist.e_out);
                    fis_cdf_vec.extend_from_slice(&dist.cdf);
                }
            }
        }
        if fis_inc_e_vec.is_empty() { fis_inc_e_vec.push(0.0); }
        if fis_eout_vec.is_empty() { fis_eout_vec.push(0.0); fis_cdf_vec.push(0.0); }
        if fis_dist_off_vec.is_empty() { fis_dist_off_vec.push(0); fis_dist_sz_vec.push(0); }

        println!("  GPU: basis={:.1} MB, grids={:.1} MB, nu-bar={} pts, fis_spec={} pts",
                 all_basis_vec.len() as f64 * 4.0 / 1e6,
                 all_grids_vec.len() as f64 * 8.0 / 1e6,
                 nb_energies_vec.len(),
                 fis_eout_vec.len());

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
            level_q_values: self.stream.clone_htod(&lev_q_vec)?,
            level_thresholds: self.stream.clone_htod(&lev_thr_vec)?,
            level_offsets: self.stream.clone_htod(&lev_off_vec)?,
            level_counts: self.stream.clone_htod(&lev_cnt_vec)?,
            level_basis: self.stream.clone_htod(&lev_basis_vec)?,
            level_coeffs: self.stream.clone_htod(&lev_coeffs_vec)?,
            level_basis_offsets: self.stream.clone_htod(&lev_basis_off_vec)?,
            level_coeffs_offsets: self.stream.clone_htod(&lev_coeffs_off_vec)?,
            level_has_kernel: self.stream.clone_htod(&lev_has_kernel_vec)?,
            nu_bar_energies: self.stream.clone_htod(&nb_energies_vec)?,
            nu_bar_values: self.stream.clone_htod(&nb_values_vec)?,
            nu_bar_offsets: self.stream.clone_htod(&nb_offsets_vec)?,
            nu_bar_sizes: self.stream.clone_htod(&nb_sizes_vec)?,
            fis_inc_energies: self.stream.clone_htod(&fis_inc_e_vec)?,
            fis_dist_offsets: self.stream.clone_htod(&fis_dist_off_vec)?,
            fis_dist_sizes: self.stream.clone_htod(&fis_dist_sz_vec)?,
            fis_e_out: self.stream.clone_htod(&fis_eout_vec)?,
            fis_cdf: self.stream.clone_htod(&fis_cdf_vec)?,
            fis_nuc_offsets: self.stream.clone_htod(&fis_nuc_off_vec)?,
            fis_nuc_n_inc: self.stream.clone_htod(&fis_nuc_ninc_vec)?,
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

    /// Upload S(α,β) thermal scattering data for one temperature.
    pub fn upload_sab_data(
        &self,
        tsl: &crate::thermal::ThermalScatteringData,
        temp_idx: usize,
    ) -> Result<GpuSabData, Box<dyn std::error::Error>> {

        let inel = &tsl.inelastic[temp_idx];
        match &inel.dist {
            crate::thermal::InelasticDist::Continuous(c) => {
                // Pack incident energy grid and XS
                let inc_e: Vec<f64> = inel.energy.clone();
                let xs: Vec<f64> = inel.xs.clone();
                let n_inc = inc_e.len() as i32;

                // Pack outgoing energy CDFs with offsets
                let mut eout_offsets = Vec::with_capacity(c.n_inc);
                let mut eout_sizes = Vec::with_capacity(c.n_inc);
                for i in 0..c.n_inc {
                    let start = c.offsets[i];
                    let end = if i + 1 < c.offsets.len() { c.offsets[i + 1] } else { c.e_out.len() };
                    eout_offsets.push(start as i32);
                    eout_sizes.push((end - start) as i32);
                }

                // Pack mu offsets/sizes (one per outgoing energy bin)
                let mut mu_offs = Vec::with_capacity(c.mu_offsets.len());
                let mut mu_szs = Vec::with_capacity(c.mu_offsets.len());
                for i in 0..c.mu_offsets.len() {
                    let start = c.mu_offsets[i];
                    let end = if i + 1 < c.mu_offsets.len() { c.mu_offsets[i + 1] } else { c.mu.len() };
                    mu_offs.push(start as i32);
                    mu_szs.push((end - start) as i32);
                }

                // Ensure non-empty
                if mu_offs.is_empty() { mu_offs.push(0); mu_szs.push(0); }

                println!("  GPU S(a,b): {} inc energies, {} E_out pts, {} mu pts",
                         n_inc, c.e_out.len(), c.mu.len());

                Ok(GpuSabData {
                    inc_energies: self.stream.clone_htod(&inc_e)?,
                    n_inc,
                    eout_offsets: self.stream.clone_htod(&eout_offsets)?,
                    eout_sizes: self.stream.clone_htod(&eout_sizes)?,
                    e_out: self.stream.clone_htod(&c.e_out)?,
                    cdf_e: self.stream.clone_htod(&c.cdf_e)?,
                    mu_offsets: self.stream.clone_htod(&mu_offs)?,
                    mu_sizes: self.stream.clone_htod(&mu_szs)?,
                    mu: self.stream.clone_htod(&c.mu)?,
                    cdf_mu: self.stream.clone_htod(&c.cdf_mu)?,
                    xs: self.stream.clone_htod(&xs)?,
                    energy_max: tsl.energy_max,
                })
            }
            crate::thermal::InelasticDist::Discrete(_) => {
                // Discrete mode — create empty placeholder (not yet supported on GPU)
                println!("  GPU S(a,b): discrete mode, using empty placeholder");
                Ok(GpuSabData {
                    inc_energies: self.stream.clone_htod(&[0.0_f64])?,
                    n_inc: 0,
                    eout_offsets: self.stream.clone_htod(&[0_i32])?,
                    eout_sizes: self.stream.clone_htod(&[0_i32])?,
                    e_out: self.stream.clone_htod(&[0.0_f64])?,
                    cdf_e: self.stream.clone_htod(&[0.0_f64])?,
                    mu_offsets: self.stream.clone_htod(&[0_i32])?,
                    mu_sizes: self.stream.clone_htod(&[0_i32])?,
                    mu: self.stream.clone_htod(&[0.0_f64])?,
                    cdf_mu: self.stream.clone_htod(&[0.0_f64])?,
                    xs: self.stream.clone_htod(&[0.0_f64])?,
                    energy_max: 0.0,
                })
            }
        }
    }

    /// Create an empty S(α,β) placeholder (no thermal scattering data).
    pub fn upload_sab_data_empty(&self) -> Result<GpuSabData, Box<dyn std::error::Error>> {
        Ok(GpuSabData {
            inc_energies: self.stream.clone_htod(&[0.0_f64])?,
            n_inc: 0,
            eout_offsets: self.stream.clone_htod(&[0_i32])?,
            eout_sizes: self.stream.clone_htod(&[0_i32])?,
            e_out: self.stream.clone_htod(&[0.0_f64])?,
            cdf_e: self.stream.clone_htod(&[0.0_f64])?,
            mu_offsets: self.stream.clone_htod(&[0_i32])?,
            mu_sizes: self.stream.clone_htod(&[0_i32])?,
            mu: self.stream.clone_htod(&[0.0_f64])?,
            cdf_mu: self.stream.clone_htod(&[0.0_f64])?,
            xs: self.stream.clone_htod(&[0.0_f64])?,
            energy_max: 0.0,
        })
    }

    /// Run one batch of transport on GPU.
    ///
    /// geom_type: 0=PWR pin cell, 1=Godiva bare sphere.
    pub fn run_batch(
        &self,
        source_bank: &[(f64, f64, f64, f64)],
        batch: u32,
        nuc_data: &GpuNuclideData,
        mat_data: &GpuMaterialData,
        sab_data: &GpuSabData,
        max_steps: u32,
        geom_type: i32,
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
                .arg(&geom_type)
                .launch(cfg_full)?;
        }

        let mut n_alive = n as i32;
        let compact_interval = 10; // Re-compact every N steps

        let mut step = 0_u32;
        while step < max_steps && n_alive > 0 {
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

            // Launch persistent kernel: N steps in one kernel call
            let alive_grid = (n_alive as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            let alive_cfg = LaunchConfig {
                grid_dim: (alive_grid, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            let steps_this_launch = compact_interval as i32;
            unsafe {
                self.stream.launch_builder(&self.k_transport_persistent)
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
                    // Nu-bar tables
                    .arg(&nuc_data.nu_bar_energies)
                    .arg(&nuc_data.nu_bar_values)
                    .arg(&nuc_data.nu_bar_offsets)
                    .arg(&nuc_data.nu_bar_sizes)
                    // Fission spectrum CDFs
                    .arg(&nuc_data.fis_inc_energies)
                    .arg(&nuc_data.fis_dist_offsets)
                    .arg(&nuc_data.fis_dist_sizes)
                    .arg(&nuc_data.fis_e_out)
                    .arg(&nuc_data.fis_cdf)
                    .arg(&nuc_data.fis_nuc_offsets)
                    .arg(&nuc_data.fis_nuc_n_inc)
                    // S(α,β) thermal scattering
                    .arg(&sab_data.inc_energies)
                    .arg(&sab_data.n_inc)
                    .arg(&sab_data.eout_offsets)
                    .arg(&sab_data.eout_sizes)
                    .arg(&sab_data.e_out)
                    .arg(&sab_data.cdf_e)
                    .arg(&sab_data.mu_offsets)
                    .arg(&sab_data.mu_sizes)
                    .arg(&sab_data.mu)
                    .arg(&sab_data.cdf_mu)
                    .arg(&sab_data.xs)
                    .arg(&sab_data.energy_max)
                    // Discrete levels
                    .arg(&nuc_data.level_q_values)
                    .arg(&nuc_data.level_thresholds)
                    .arg(&nuc_data.level_offsets)
                    .arg(&nuc_data.level_counts)
                    .arg(&nuc_data.level_basis)
                    .arg(&nuc_data.level_coeffs)
                    .arg(&nuc_data.level_basis_offsets)
                    .arg(&nuc_data.level_coeffs_offsets)
                    .arg(&nuc_data.level_has_kernel)
                    // RNG
                    .arg(&mut d_rng_state).arg(&mut d_rng_inc)
                    .arg(&mut d_fis_x).arg(&mut d_fis_y).arg(&mut d_fis_z)
                    .arg(&mut d_fis_e).arg(&mut d_fis_w)
                    .arg(&mut d_fis_count).arg(&max_fission)
                    .arg(&mut d_cnt_coll).arg(&mut d_cnt_fis)
                    .arg(&mut d_cnt_leak).arg(&mut d_cnt_surf)
                    .arg(&steps_this_launch)
                    .arg(&geom_type)
                    .launch(alive_cfg)?;
            }

            step += compact_interval; // persistent kernel did N steps
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
