// ═══════════════════════════════════════════════════════════════════════
// open_rust_mc — Event-based GPU neutron transport kernel
//
// Single packed parameter struct for all read-only data.
// Persistent kernel with warp-level reductions and energy-sorted
// compaction. Supports PWR pin cell and Godiva geometries.
// ═══════════════════════════════════════════════════════════════════════

#define COINCIDENCE_TOL 1e-12
#define PI 3.14159265358979323846
#define N_REACTIONS 7
#define RXN_ELASTIC 0
#define RXN_TOTAL   6
#define RXN_INELASTIC 1
#define RXN_N2N 2
#define RXN_N3N 3
#define RXN_FISSION 4
#define RXN_CAPTURE 5

// Geometry
#define GEOM_PWR 0
#define GEOM_GODIVA 1
#define FUEL_OR 0.4096
#define CLAD_IR 0.4180
#define CLAD_OR 0.4750
#define HALF_PITCH 0.63
#define GODIVA_RADIUS 8.7407
#define BC_TRANSMISSION 0
#define BC_REFLECTIVE 1
#define BC_VACUUM 2
#define CELL_FUEL 0
#define CELL_GAP 1
#define CELL_CLAD 2
#define CELL_WATER 3

// Energy binning
#define N_ENERGY_BINS 256
#define LOG_E_MIN (-16.6096)
#define LOG_E_RANGE 40.9193
#define INV_LOG_STEP (N_ENERGY_BINS / LOG_E_RANGE)

// ═══════════════════════════════════════════════════════════════════════
// Packed parameter struct — ALL read-only data in one device buffer.
// All fields are unsigned long long (8 bytes) for zero-padding layout.
// Pointers stored as device addresses. Scalars cast to/from ull.
// Doubles stored via __longlong_as_double / __double_as_longlong.
//
// The Rust host packs a Vec<u64> with the same field order, uploads it,
// and the kernel casts the buffer pointer to TransportParams*.
// ═══════════════════════════════════════════════════════════════════════

// Field indices — keep in sync with Rust PackedParams builder
#define P_BASIS           0
#define P_COEFFS          1
#define P_ENERGY_GRIDS    2
#define P_BASIS_OFFSETS   3
#define P_GRID_OFFSETS    4
#define P_N_ENERGIES      5
#define P_HAS_REACTION    6
#define P_COEFFS_OFFSETS  7
#define P_RANK            8
#define P_MAT_N_NUC       9
#define P_MAT_NUC_IDX    10
#define P_MAT_ATOM_DENS  11
#define P_AWR_TABLE      12
#define P_NU_BAR_CONST   13
#define P_NB_ENERGIES    14
#define P_NB_VALUES      15
#define P_NB_OFFSETS     16
#define P_NB_SIZES       17
#define P_FIS_INC_E      18
#define P_FIS_DIST_OFF   19
#define P_FIS_DIST_SZ    20
#define P_FIS_E_OUT      21
#define P_FIS_CDF        22
#define P_FIS_NUC_OFF    23
#define P_FIS_NUC_NINC   24
#define P_LEVEL_Q        25
#define P_LEVEL_THR      26
#define P_LEVEL_OFFSETS  27
#define P_LEVEL_COUNTS   28
#define P_LEVEL_BASIS    29
#define P_LEVEL_COEFFS   30
#define P_LEVEL_BOFF     31
#define P_LEVEL_COFF     32
#define P_LEVEL_HAS_K    33
#define P_LEVEL_MT       34
#define P_ANG_ENERGIES   35
#define P_ANG_MU         36
#define P_ANG_CDF        37
#define P_ANG_DIST_OFF   38
#define P_ANG_DIST_SZ    39
#define P_ANG_NUC_OFF    40
#define P_ANG_NUC_NE     41
#define P_ANG_IS_CM      42
#define P_SAB_INC_E      43
#define P_SAB_N_INC      44
#define P_SAB_EOUT_OFF   45
#define P_SAB_EOUT_SZ    46
#define P_SAB_E_OUT      47
#define P_SAB_CDF_E      48
#define P_SAB_MU_OFF     49
#define P_SAB_MU_SZ      50
#define P_SAB_MU         51
#define P_SAB_CDF_MU     52
#define P_SAB_XS         53
#define P_SAB_EMAX       54
#define P_SAB_PDF_E      55
#define P_URR_ENERGIES   56
#define P_URR_CUM_PROB   57
#define P_URR_TOTAL_F    58
#define P_URR_ELASTIC_F  59
#define P_URR_FISSION_F  60
#define P_URR_CAPTURE_F  61
#define P_URR_OFFSETS    62
#define P_URR_N_ENERGIES 63
#define P_URR_N_BANDS    64
#define P_URR_MULT_SM    65
#define P_GEOM_TYPE      66
#define P_TOTAL_XS       67
#define P_TOTAL_XS_OFF   68
#define P_HAS_TOTAL_XS   69
#define P_PW_XS          70
#define P_PW_OFF         71
#define P_HAS_PW         72
// ── Windowed-Multipole (hybrid) — per-nuclide WMP data ─────────────────
#define P_WMP_HAS         73
#define P_WMP_E_MIN       74
#define P_WMP_E_MAX       75
#define P_WMP_SPACING     76
#define P_WMP_SQRT_AWR    77
#define P_WMP_T_KELVIN    78
#define P_WMP_FIT_ORDER   79
#define P_WMP_N_WINDOWS   80
#define P_WMP_FISSIONABLE 81
#define P_WMP_POLES       82
#define P_WMP_POLE_OFF    83
#define P_WMP_WINDOWS     84
#define P_WMP_WIN_OFF     85
#define P_WMP_BROADEN     86
#define P_WMP_BROADEN_OFF 87
#define P_WMP_CURVEFIT    88
#define P_WMP_CF_OFF      89

// Per-discrete-level CM-frame angular distributions (MT=51-91). Indexed
// by the same global level index as P_LEVEL_Q / P_LEVEL_MT. A zero
// P_LEV_ANG_LEV_NE entry means "no tabulated angular dist" — the kernel
// falls back to isotropic in the CM frame, matching the old behaviour.
#define P_LEV_ANG_ENERGIES 90
#define P_LEV_ANG_MU       91
#define P_LEV_ANG_CDF      92
#define P_LEV_ANG_DIST_OFF 93
#define P_LEV_ANG_DIST_SZ  94
#define P_LEV_ANG_LEV_OFF  95
#define P_LEV_ANG_LEV_NE   96
// Per-level CDF for inelastic-level sampling (replaces the 13-level
// walk in do_inelastic with a binary search). Per-nuclide metadata.
// inel_cdf_off[ni] = -1 means "no CDF, fall through to legacy walk".
//
// NOTE: a per-warp shared-memory cache keyed on (n, E_idx) was
// previously sat at index 97 (P_WARP_CACHE_ENABLE) — implemented and
// then removed once synthesis + CDF made it dead weight. The
// falsification of the "per-warp cache closes the gap" hypothesis
// is preserved in paper §threats; the kernel surface is back to its
// pre-cache shape modulo the synth+CDF additions.
#define P_INEL_CDF_DATA      97
#define P_INEL_CDF_OFF       98
#define P_INEL_CDF_N_E       99
#define P_INEL_CDF_N_T      100
#define P_INEL_CDF_N_LEV    101
#define P_INEL_CDF_LOG_EMIN 102
#define P_INEL_CDF_LOG_EMAX 103
#define N_PARAMS            104

// Access helpers — read from the flat u64 params buffer
// PTR_F removed — all basis data is now f64 (PTR_D)
#define PTR_D(p, idx)   ((const double*) (p)[(idx)])
#define PTR_I(p, idx)   ((const int*)    (p)[(idx)])
#define PTR_B(p, idx)   ((const signed char*) (p)[(idx)])
#define PTR_D2(p, idx)  ((const double2*) (p)[(idx)])
#define SCALAR_I(p, idx) ((int)(p)[(idx)])
#define SCALAR_D(p, idx) __longlong_as_double((long long)(p)[(idx)])

// Convenience: typed param buffer is just unsigned long long*
typedef const unsigned long long* Params;

// ═══════════════════════════════════════════════════════════════════════
// PCG-64 RNG
// ═══════════════════════════════════════════════════════════════════════

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

// NOTE: a `warp_atomic_alloc` helper (prefix-scan via `__shfl_up_sync`
// + one `atomicAdd` per warp) was prototyped here and discarded before
// committing: `__shfl_up_sync(mask, val, off)` is undefined when source
// lane `(lane_id - off)` is not in `mask`, and fission / (n,2n) / (n,3n)
// branches reach this atomic with sparse active masks. In smoke runs
// Godiva k_eff collapsed from ~1.00 to ~0.63 because neutrons were
// banked at wrong offsets. A correct implementation must use
// cooperative_groups::coalesced_threads() or run every lane in the
// warp with `my_count = 0` fallback under a full-warp mask (0xffffffff).
// Until that lands, plain `atomicAdd` is the ground truth.

// ═══════════════════════════════════════════════════════════════════════
// Geometry
// ═══════════════════════════════════════════════════════════════════════

__device__ double dist_cylinder_z(double px, double py, double dx, double dy, double R) {
    double a = dx*dx + dy*dy;
    if (a < COINCIDENCE_TOL) return -1.0;
    double b = 2.0*(px*dx + py*dy);
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

__device__ double dist_plane(double p, double d, double x0) {
    if (fabs(d) < COINCIDENCE_TOL) return -1.0;
    double t = (x0 - p) / d;
    return (t > COINCIDENCE_TOL) ? t : -1.0;
}

__device__ int find_cell(double x, double y, double z, int geom_type) {
    if (geom_type == GEOM_GODIVA) {
        return (x*x + y*y + z*z < GODIVA_RADIUS*GODIVA_RADIUS) ? 0 : -1;
    }
    double r2 = x*x + y*y;
    if (z <= -HALF_PITCH || z >= HALF_PITCH) return -1;
    if (r2 < FUEL_OR*FUEL_OR) return CELL_FUEL;
    if (r2 < CLAD_IR*CLAD_IR) return CELL_GAP;
    if (r2 < CLAD_OR*CLAD_OR) return CELL_CLAD;
    if (x > -HALF_PITCH && x < HALF_PITCH && y > -HALF_PITCH && y < HALF_PITCH) return CELL_WATER;
    return -1;
}

__device__ int cell_material(int cell, int geom_type) {
    if (geom_type == GEOM_GODIVA) return (cell == 0) ? 0 : -1;
    if (cell == CELL_FUEL) return 0;
    if (cell == CELL_CLAD) return 1;
    if (cell == CELL_WATER) return 2;
    return -1;
}

__device__ void trace_surface(
    double px, double py, double pz,
    double dx, double dy, double dz,
    int cell, int geom_type,
    double* out_dist, int* out_bc, int* out_next_cell)
{
    if (geom_type == GEOM_GODIVA) {
        double a = dx*dx+dy*dy+dz*dz;
        double b = 2.0*(px*dx+py*dy+pz*dz);
        double c = px*px+py*py+pz*pz - GODIVA_RADIUS*GODIVA_RADIUS;
        double disc = b*b-4.0*a*c;
        if (disc < 0.0) { *out_dist=1e20; *out_bc=BC_VACUUM; *out_next_cell=-1; return; }
        double sq = sqrt(disc);
        double t1 = (-b-sq)/(2.0*a), t2 = (-b+sq)/(2.0*a);
        double t = (t1 > COINCIDENCE_TOL) ? t1 : ((t2 > COINCIDENCE_TOL) ? t2 : 1e20);
        *out_dist=t; *out_bc=BC_VACUUM; *out_next_cell=-1; return;
    }

    double best_t = 1e20;
    int best_bc = BC_VACUUM, best_next = -1;

    #define TEST_SURF(t_val, bc_val) do { \
        double _t = (t_val); \
        if (_t > COINCIDENCE_TOL && _t < best_t) { best_t = _t; best_bc = (bc_val); } \
    } while(0)

    TEST_SURF(dist_plane(pz, dz, -HALF_PITCH), BC_REFLECTIVE);
    TEST_SURF(dist_plane(pz, dz,  HALF_PITCH), BC_REFLECTIVE);

    if (cell == CELL_FUEL) {
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,FUEL_OR), BC_TRANSMISSION);
    } else if (cell == CELL_GAP) {
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,FUEL_OR), BC_TRANSMISSION);
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,CLAD_IR), BC_TRANSMISSION);
    } else if (cell == CELL_CLAD) {
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,CLAD_IR), BC_TRANSMISSION);
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,CLAD_OR), BC_TRANSMISSION);
    } else if (cell == CELL_WATER) {
        TEST_SURF(dist_cylinder_z(px,py,dx,dy,CLAD_OR), BC_TRANSMISSION);
        TEST_SURF(dist_plane(px,dx,-HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(px,dx, HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(py,dy,-HALF_PITCH), BC_REFLECTIVE);
        TEST_SURF(dist_plane(py,dy, HALF_PITCH), BC_REFLECTIVE);
    }
    #undef TEST_SURF

    if (best_bc == BC_TRANSMISSION && best_t < 1e19) {
        double nx=px+dx*(best_t+1e-8), ny=py+dy*(best_t+1e-8), nz=pz+dz*(best_t+1e-8);
        best_next = find_cell(nx, ny, nz, geom_type);
    }
    *out_dist=best_t; *out_bc=best_bc; *out_next_cell=best_next;
}

// ═══════════════════════════════════════════════════════════════════════
// Physics helpers
// ═══════════════════════════════════════════════════════════════════════

__device__ int energy_index(const double* grid, int n_e, double energy) {
    if (energy <= grid[0]) return 0;
    if (energy >= grid[n_e-1]) return n_e-1;
    int lo=0, hi=n_e-1;
    while (hi-lo > 1) { int mid=(lo+hi) >> 1; if (grid[mid]<=energy) lo=mid; else hi=mid; }
    return lo;
}

// SVD reconstruct in LOG10 space
__device__ double svd_reconstruct_log(
    const double* __restrict__ basis,
    const double* __restrict__ coeffs,
    int e_idx, int rank)
{
    const double* row = &basis[e_idx * rank];
    double acc = 0.0;
    for (int j = 0; j < rank; j++)
        acc = fma(__ldg(&row[j]), __ldg(&coeffs[j]), acc);
    return acc;
}

// SVD reconstruct with log-log interpolation between grid points (OpenMC scheme)
__device__ double svd_reconstruct_interp(
    const double* __restrict__ basis,
    const double* __restrict__ coeffs,
    int e_idx, int n_e, int rank, double log_frac)
{
    double log_lo = svd_reconstruct_log(basis, coeffs, e_idx, rank);
    if (e_idx + 1 >= n_e || log_frac <= 0.0)
        return exp2(log_lo * 3.321928094887362);
    double log_hi = svd_reconstruct_log(basis, coeffs, e_idx + 1, rank);
    double log_interp = log_lo + log_frac * (log_hi - log_lo);
    return exp2(log_interp * 3.321928094887362);
}

// Legacy: single-point reconstruct (used by discrete levels)
__device__ double svd_reconstruct(
    const double* __restrict__ basis,
    const double* __restrict__ coeffs,
    int e_idx, int rank)
{
    return exp2(svd_reconstruct_log(basis, coeffs, e_idx, rank) * 3.321928094887362);
}

__device__ double nu_bar_lookup(
    double E, const double* energies, const double* values, int offset, int n)
{
    if (n <= 0) return 0.0;
    const double* e = &energies[offset];
    const double* v = &values[offset];
    if (E <= e[0]) return v[0];
    if (E >= e[n-1]) return v[n-1];
    int lo=0, hi=n-1;
    while (hi-lo>1) { int mid=(lo+hi) >> 1; if (e[mid]<=E) lo=mid; else hi=mid; }
    double f = (E-e[lo]) / (e[hi]-e[lo]);
    return v[lo] + f*(v[hi]-v[lo]);
}

// CDF-invert E_out within one bin using a pre-drawn xi.
__device__ __forceinline__ double sample_eout_bin(
    double xi, const double* eo, const double* cd, int sz)
{
    if (sz <= 1) return eo[0];
    int lo=0, hi=sz-1;
    while (hi-lo > 1) { int mid=(lo+hi) >> 1; if (cd[mid] <= xi) lo=mid; else hi=mid; }
    double f = (xi - cd[lo]) / fmax(cd[hi]-cd[lo], 1e-30);
    return eo[lo] + f * (eo[hi] - eo[lo]);
}

__device__ double sample_fission_energy(
    double E_inc, PcgState* rng, Params p, int hit_nuc)
{
    int fi_off = __ldg(&PTR_I(p, P_FIS_NUC_OFF)[hit_nuc]);
    int fi_n   = __ldg(&PTR_I(p, P_FIS_NUC_NINC)[hit_nuc]);
    if (fi_n <= 0) {
        // Watt fallback when no tabulated spectrum available.
        double a=0.988, x1=-log(fmax(pcg_uniform(rng),1e-30));
        double x2=-log(fmax(pcg_uniform(rng),1e-30));
        double c=cos(PI/2.0*pcg_uniform(rng));
        return a*(x1+x2*c*c)*1e6;
    }
    const double* inc_e = &PTR_D(p, P_FIS_INC_E)[fi_off];

    // Edge: below grid — sample directly from first bin.
    if (E_inc <= inc_e[0]) {
        int off=PTR_I(p, P_FIS_DIST_OFF)[fi_off], sz=PTR_I(p, P_FIS_DIST_SZ)[fi_off];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &PTR_D(p, P_FIS_E_OUT)[off],
                                    &PTR_D(p, P_FIS_CDF)[off], sz), 1e-5);
    }
    // Edge: above grid — sample from last bin.
    if (E_inc >= inc_e[fi_n-1]) {
        int off=PTR_I(p, P_FIS_DIST_OFF)[fi_off+fi_n-1], sz=PTR_I(p, P_FIS_DIST_SZ)[fi_off+fi_n-1];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &PTR_D(p, P_FIS_E_OUT)[off],
                                    &PTR_D(p, P_FIS_CDF)[off], sz), 1e-5);
    }

    // Binary search for bracket
    int ie; { int lo=0, hi=fi_n-1;
        while(hi-lo>1){int mid=(lo+hi) >> 1; if(inc_e[mid]<=E_inc) lo=mid; else hi=mid;} ie=lo; }

    // OpenMC stochastic-bin sampling + scaled kinematic remap
    // (distribution_energy.cpp ContinuousTabular::sample).
    double r = (E_inc - inc_e[ie]) / fmax(inc_e[ie+1] - inc_e[ie], 1e-30);
    bool pick_hi = pcg_uniform(rng) < r;
    int chosen_lo = fi_off + ie;
    int chosen_hi = fi_off + ie + 1;
    int chosen = pick_hi ? chosen_hi : chosen_lo;
    int off_l = PTR_I(p, P_FIS_DIST_OFF)[chosen];
    int sz_l  = PTR_I(p, P_FIS_DIST_SZ)[chosen];
    const double* eo_l = &PTR_D(p, P_FIS_E_OUT)[off_l];
    const double* cd_l = &PTR_D(p, P_FIS_CDF)[off_l];
    double e_out = sample_eout_bin(pcg_uniform(rng), eo_l, cd_l, sz_l);

    // Scaled kinematic adjustment: remap e_out from chosen bin's
    // [el1_lo, el1_hi] to the interpolated [e1, eK] between both bins.
    int off_a = PTR_I(p, P_FIS_DIST_OFF)[chosen_lo];
    int sz_a  = PTR_I(p, P_FIS_DIST_SZ)[chosen_lo];
    int off_b = PTR_I(p, P_FIS_DIST_OFF)[chosen_hi];
    int sz_b  = PTR_I(p, P_FIS_DIST_SZ)[chosen_hi];
    const double* eo_a = &PTR_D(p, P_FIS_E_OUT)[off_a];
    const double* eo_b = &PTR_D(p, P_FIS_E_OUT)[off_b];
    double el1_lo = eo_l[0];
    double el1_hi = (sz_l > 0) ? eo_l[sz_l-1] : el1_lo;
    double ea_lo  = eo_a[0];
    double ea_hi  = (sz_a > 0) ? eo_a[sz_a-1] : ea_lo;
    double eb_lo  = eo_b[0];
    double eb_hi  = (sz_b > 0) ? eo_b[sz_b-1] : eb_lo;
    double e1 = (1.0 - r) * ea_lo + r * eb_lo;
    double eK = (1.0 - r) * ea_hi + r * eb_hi;
    double span_l = el1_hi - el1_lo;
    double adjusted = (fabs(span_l) < 1e-30) ? e_out
                    : e1 + (e_out - el1_lo) * (eK - e1) / span_l;
    return fmax(adjusted, 1e-5);
}

// CDF-invert mu within one bin using a pre-drawn xi.
__device__ __forceinline__ double sample_mu_bin(
    double xi, const double* mu, const double* cd, int sz)
{
    if (sz <= 1) return 2.0*xi - 1.0;
    int lo=0, hi=sz-1;
    while (hi-lo > 1) { int mid=(lo+hi) >> 1; if (cd[mid] <= xi) lo=mid; else hi=mid; }
    double f = (xi - cd[lo]) / fmax(cd[hi]-cd[lo], 1e-30);
    double m = mu[lo] + f * (mu[hi] - mu[lo]);
    return fmax(-1.0, fmin(1.0, m));
}

__device__ double sample_angular_dist(
    double E, PcgState* rng, Params p, int hit_nuc)
{
    int a_off = __ldg(&PTR_I(p, P_ANG_NUC_OFF)[hit_nuc]);
    int a_ne = __ldg(&PTR_I(p, P_ANG_NUC_NE)[hit_nuc]);
    if (a_ne <= 0) return 2.0*pcg_uniform(rng)-1.0;
    const double* ae = &PTR_D(p, P_ANG_ENERGIES)[a_off];

    // Edge: below grid
    if (E <= ae[0]) {
        int off=PTR_I(p, P_ANG_DIST_OFF)[a_off], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off];
        return sample_mu_bin(pcg_uniform(rng),
                             &PTR_D(p, P_ANG_MU)[off],
                             &PTR_D(p, P_ANG_CDF)[off], sz);
    }
    // Edge: above grid
    if (E >= ae[a_ne-1]) {
        int off=PTR_I(p, P_ANG_DIST_OFF)[a_off+a_ne-1], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off+a_ne-1];
        return sample_mu_bin(pcg_uniform(rng),
                             &PTR_D(p, P_ANG_MU)[off],
                             &PTR_D(p, P_ANG_CDF)[off], sz);
    }

    // Binary search for energy bracket
    int ie; { int lo=0, hi=a_ne-1;
        while(hi-lo>1){int mid=(lo+hi) >> 1; if(ae[mid]<=E) lo=mid; else hi=mid;} ie=lo; }

    // OpenMC stochastic-bin sampling (distribution_angle.cpp):
    //   r = (E - E_lo)/(E_hi - E_lo); pick_hi = (ξ_bin < r); then sample
    //   μ from the chosen bin with a fresh ξ_μ. Two draws total.
    double r = (E - ae[ie]) / fmax(ae[ie+1] - ae[ie], 1e-30);
    bool pick_hi = pcg_uniform(rng) < r;
    int chosen = pick_hi ? (a_off + ie + 1) : (a_off + ie);
    int off = PTR_I(p, P_ANG_DIST_OFF)[chosen];
    int sz  = PTR_I(p, P_ANG_DIST_SZ)[chosen];
    return sample_mu_bin(pcg_uniform(rng),
                         &PTR_D(p, P_ANG_MU)[off],
                         &PTR_D(p, P_ANG_CDF)[off], sz);
}

// ═══════════════════════════════════════════════════════════════════════
// Per-discrete-level angular distribution sampler (MT=51-91).
// Mirrors CPU AngularDistribution::sample_mu via stochastic-bin selection
// on the level's own energy grid. `global_lev_idx` is the index into the
// flat per-nuclide level array (same space as P_LEVEL_MT / P_LEVEL_Q).
// Returns 2*ξ−1 when the level has no tabulated angular distribution —
// matches the CPU isotropic fallback.
// ═══════════════════════════════════════════════════════════════════════
__device__ double sample_level_angular(
    double E, PcgState* rng, Params p, int global_lev_idx)
{
    int n_e = __ldg(&PTR_I(p, P_LEV_ANG_LEV_NE)[global_lev_idx]);
    if (n_e <= 0) return 2.0 * pcg_uniform(rng) - 1.0;

    int e_off = __ldg(&PTR_I(p, P_LEV_ANG_LEV_OFF)[global_lev_idx]);
    const double* ae = &PTR_D(p, P_LEV_ANG_ENERGIES)[e_off];

    // Below / above grid: pick the edge distribution directly.
    if (E <= ae[0]) {
        int off = PTR_I(p, P_LEV_ANG_DIST_OFF)[e_off];
        int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[e_off];
        return sample_mu_bin(pcg_uniform(rng),
                             &PTR_D(p, P_LEV_ANG_MU)[off],
                             &PTR_D(p, P_LEV_ANG_CDF)[off], sz);
    }
    if (E >= ae[n_e - 1]) {
        int off = PTR_I(p, P_LEV_ANG_DIST_OFF)[e_off + n_e - 1];
        int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[e_off + n_e - 1];
        return sample_mu_bin(pcg_uniform(rng),
                             &PTR_D(p, P_LEV_ANG_MU)[off],
                             &PTR_D(p, P_LEV_ANG_CDF)[off], sz);
    }

    int ie; { int lo = 0, hi = n_e - 1;
        while (hi - lo > 1) { int mid = (lo + hi) / 2;
            if (ae[mid] <= E) lo = mid; else hi = mid; } ie = lo; }

    double r = (E - ae[ie]) / fmax(ae[ie+1] - ae[ie], 1e-30);
    bool pick_hi = pcg_uniform(rng) < r;
    int chosen = e_off + (pick_hi ? ie + 1 : ie);
    int off = PTR_I(p, P_LEV_ANG_DIST_OFF)[chosen];
    int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[chosen];
    return sample_mu_bin(pcg_uniform(rng),
                         &PTR_D(p, P_LEV_ANG_MU)[off],
                         &PTR_D(p, P_LEV_ANG_CDF)[off], sz);
}

__device__ double sab_total_xs(double E, Params p) {
    if (SCALAR_I(p, P_SAB_N_INC) <= 0) return 0.0;
    const double* e = PTR_D(p, P_SAB_INC_E);
    const double* xs = PTR_D(p, P_SAB_XS);
    int n = SCALAR_I(p, P_SAB_N_INC);
    if (E <= e[0]) return xs[0];
    if (E >= e[n-1]) return xs[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(lo+hi) >> 1;if(e[mid]<=E)lo=mid;else hi=mid;}
    double f=(E-e[lo])/fmax(e[hi]-e[lo],1e-30);
    return xs[lo]+f*(xs[hi]-xs[lo]);
}

__device__ void sab_sample(
    double E_in, PcgState* rng, Params p,
    double* E_out, double* mu_out)
{
    int n = SCALAR_I(p, P_SAB_N_INC);
    if (n <= 0) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }

    const double* inc_e = PTR_D(p, P_SAB_INC_E);

    // Step 1: Find bounding incident energies i_lo, i_hi
    int i_hi = 1;
    if (E_in <= inc_e[0]) { i_hi = 1; }
    else if (E_in >= inc_e[n-1]) { i_hi = n-1; }
    else {
        int lo=0, hi=n-1;
        while(hi-lo>1){int mid=(lo+hi) >> 1; if(inc_e[mid]<=E_in) lo=mid; else hi=mid;}
        i_hi = hi;
    }
    int i_lo = i_hi - 1;
    double denom = inc_e[i_hi] - inc_e[i_lo];
    double f = (denom > 1e-30) ? (E_in - inc_e[i_lo]) / denom : 0.0;

    // Step 2: Stochastic table selection
    int ell = (pcg_uniform(rng) > f) ? i_lo : i_hi;

    int eo_off = PTR_I(p, P_SAB_EOUT_OFF)[ell];
    int eo_sz  = PTR_I(p, P_SAB_EOUT_SZ)[ell];
    if (eo_sz <= 1) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }

    const double* eo    = &PTR_D(p, P_SAB_E_OUT)[eo_off];
    const double* cdf_e = &PTR_D(p, P_SAB_CDF_E)[eo_off];
    const double* pdf_e = &PTR_D(p, P_SAB_PDF_E)[eo_off];

    // Step 3: Sample outgoing energy bin from CDF
    double xi_e = pcg_uniform(rng);
    int j = 1;
    { int lo=0, hi=eo_sz-1;
      while(hi-lo>1){int mid=(lo+hi) >> 1; if(cdf_e[mid]<xi_e) lo=mid; else hi=mid;}
      j = (lo == 0 && cdf_e[0] >= xi_e) ? 0 : lo;
    }
    if (j >= eo_sz-1) j = eo_sz-2;

    // Step 4: PDF-based within-bin interpolation (OpenMC Eq 33/34)
    double e_hat;
    double dp = pdf_e[j+1] - pdf_e[j];
    if (fabs(dp) < 1e-30) {
        // Histogram bin
        e_hat = (fabs(pdf_e[j]) < 1e-30) ? eo[j]
              : eo[j] + (xi_e - cdf_e[j]) / pdf_e[j];
    } else {
        // Linear-linear interpolation (Eq 34)
        double m = dp / fmax(eo[j+1] - eo[j], 1e-30);
        double disc = pdf_e[j]*pdf_e[j] + 2.0*m*(xi_e - cdf_e[j]);
        e_hat = (disc < 0.0) ? eo[j] : eo[j] + (sqrt(fmax(disc,0.0)) - pdf_e[j]) / m;
    }

    // Step 5: Kinematic energy scaling (OpenMC Eq 31/35)
    int off_lo = PTR_I(p, P_SAB_EOUT_OFF)[i_lo];
    int sz_lo  = PTR_I(p, P_SAB_EOUT_SZ)[i_lo];
    int off_hi = PTR_I(p, P_SAB_EOUT_OFF)[i_hi];
    int sz_hi  = PTR_I(p, P_SAB_EOUT_SZ)[i_hi];
    const double* eo_all = PTR_D(p, P_SAB_E_OUT);
    double e_min = eo_all[off_lo] + f * (eo_all[off_hi] - eo_all[off_lo]);
    double e_max = eo_all[off_lo + sz_lo - 1] + f * (eo_all[off_hi + sz_hi - 1] - eo_all[off_lo + sz_lo - 1]);
    double e_ell_min = eo[0];
    double e_ell_max = eo[eo_sz - 1];
    double e_range = e_ell_max - e_ell_min;
    double e_out_final = (e_range > 1e-30)
        ? e_min + (e_hat - e_ell_min) / e_range * (e_max - e_min)
        : e_hat;
    *E_out = fmax(e_out_final, 1e-11);

    // Step 6: Angular distribution — equiprobable discrete bins with smearing
    int mu_key = eo_off + j;
    int mu_off = PTR_I(p, P_SAB_MU_OFF)[mu_key];
    int mu_sz  = PTR_I(p, P_SAB_MU_SZ)[mu_key];
    if (mu_sz <= 1) { *mu_out = 2.0*pcg_uniform(rng) - 1.0; return; }

    const double* mu_arr = &PTR_D(p, P_SAB_MU)[mu_off];
    int k = (int)(pcg_uniform(rng) * mu_sz);
    if (k >= mu_sz) k = mu_sz - 1;
    double mu_k = mu_arr[k];
    double left  = (k > 0)        ? (mu_k - mu_arr[k-1]) : (mu_k + 1.0);
    double right = (k+1 < mu_sz)  ? (mu_arr[k+1] - mu_k) : (1.0 - mu_k);
    double hw = fmin(left, right);
    *mu_out = fmax(-1.0, fmin(1.0, mu_k + hw * (pcg_uniform(rng) - 0.5)));
}

__device__ void apply_urr(
    Params p, int nuc_idx,
    double* sig_el, double* sig_fis, double* sig_cap, double E, double xi)
{
    int n_e = __ldg(&PTR_I(p, P_URR_N_ENERGIES)[nuc_idx]);
    if (n_e <= 0) return;
    int off = __ldg(&PTR_I(p, P_URR_OFFSETS)[nuc_idx]);
    int n_b = __ldg(&PTR_I(p, P_URR_N_BANDS)[nuc_idx]);
    const double* ue = &PTR_D(p, P_URR_ENERGIES)[off];
    if (E < ue[0] || E > ue[n_e-1]) return;
    // Find energy index
    int ie=0; { int lo=0,hi=n_e-1;
        while(hi-lo>1){int mid=(lo+hi) >> 1;if(ue[mid]<=E)lo=mid;else hi=mid;} ie=lo; }
    // Sample band
    int base = off*n_b + ie*n_b;
    const double* cp = &PTR_D(p, P_URR_CUM_PROB)[base];
    int band=0;
    for (int b=0; b<n_b; b++) { if (xi < cp[b]) { band=b; break; } band=b; }
    // ft (total factor) not used — reaction-specific factors applied directly
    (void)PTR_D(p, P_URR_TOTAL_F);
    double fe=PTR_D(p, P_URR_ELASTIC_F)[base+band];
    double ff=PTR_D(p, P_URR_FISSION_F)[base+band];
    double fc=PTR_D(p, P_URR_CAPTURE_F)[base+band];
    int ms = __ldg(&PTR_I(p, P_URR_MULT_SM)[nuc_idx]);
    if (ms) { *sig_el*=fe; *sig_fis*=ff; *sig_cap*=fc; }
    else { *sig_el=fe; *sig_fis=ff; *sig_cap=fc; }
}

__device__ int energy_to_bin(double E) {
    double log_e = log2(fmax(E, 1e-11));
    int bin = (int)((log_e - LOG_E_MIN) * INV_LOG_STEP);
    return max(0, min(N_ENERGY_BINS-1, bin));
}

// ═══════════════════════════════════════════════════════════════════════
// Direction rotation helper
// ═══════════════════════════════════════════════════════════════════════

__device__ void rotate_direction(
    double* dx, double* dy, double* dz, double mu, double phi)
{
    double sin_mu = sqrt(fmax(0.0, 1.0-mu*mu));
    double w2 = (*dz)*(*dz);
    // Paired trig via a single SFU dispatch (NVIDIA BPG §12.1.1).
    double s_phi, c_phi;
    sincos(phi, &s_phi, &c_phi);
    if (w2 < 0.999) {
        double inv_sq = 1.0/sqrt(1.0-w2);
        double dx2=mu*(*dx)+sin_mu*((*dx)*(*dz)*c_phi-(*dy)*s_phi)*inv_sq;
        double dy2=mu*(*dy)+sin_mu*((*dy)*(*dz)*c_phi+(*dx)*s_phi)*inv_sq;
        double dz2=mu*(*dz)-sin_mu*sqrt(1.0-w2)*c_phi;
        *dx=dx2; *dy=dy2; *dz=dz2;
    } else {
        double sign = (*dz > 0.0) ? 1.0 : -1.0;
        *dx=sin_mu*c_phi; *dy=sin_mu*s_phi*sign; *dz=mu*sign;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Utility kernels
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void count_alive(const int* alive, int n, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && alive[tid]) atomicAdd(count, 1);
}

extern "C" __global__ void compact_alive(
    const int* alive, int n, int* compact_idx, int* compact_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && alive[tid]) { int pos = atomicAdd(compact_count, 1); compact_idx[pos] = tid; }
}

extern "C" __global__ void energy_bin_count(
    const double* energy, const int* compact_idx, int n_alive, int* bin_counts) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < n_alive) atomicAdd(&bin_counts[energy_to_bin(energy[compact_idx[lane]])], 1);
}

extern "C" __global__ void energy_bin_scatter(
    const double* energy, const int* in_idx, int n_alive, int* out_idx, int* offsets) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < n_alive) {
        int tid = in_idx[lane];
        int pos = atomicAdd(&offsets[energy_to_bin(energy[tid])], 1);
        out_idx[pos] = tid;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Diagnostic kernel: sample angular distribution at given (energy, xi) pairs
// Writes mu values to output buffer for comparison with CPU.
// Also tests stair-step vs interpolated in the same kernel.
// ═══════════════════════════════════════════════════════════════════════

// Stair-step version (for A/B comparison within same kernel)
__device__ double sample_angular_dist_stairstep(
    double E, double xi, Params p, int hit_nuc)
{
    int a_off = PTR_I(p, P_ANG_NUC_OFF)[hit_nuc];
    int a_ne = PTR_I(p, P_ANG_NUC_NE)[hit_nuc];
    if (a_ne <= 0) return 2.0*xi-1.0;
    const double* ae = &PTR_D(p, P_ANG_ENERGIES)[a_off];
    int ie=0;
    if (E <= ae[0]) ie=0;
    else if (E >= ae[a_ne-1]) ie=a_ne-1;
    else { int lo=0,hi=a_ne-1; while(hi-lo>1){int mid=(lo+hi) >> 1;if(ae[mid]<=E)lo=mid;else hi=mid;} ie=lo; }
    int off = PTR_I(p, P_ANG_DIST_OFF)[a_off+ie];
    int sz = PTR_I(p, P_ANG_DIST_SZ)[a_off+ie];
    if (sz <= 1) return 2.0*xi-1.0;
    const double* mu = &PTR_D(p, P_ANG_MU)[off];
    const double* cd = &PTR_D(p, P_ANG_CDF)[off];
    int lo=0, hi=sz-1;
    while(hi-lo>1){int mid=(lo+hi) >> 1;if(cd[mid]<=xi)lo=mid;else hi=mid;}
    double f = (xi-cd[lo])/fmax(cd[hi]-cd[lo],1e-30);
    return fmax(-1.0, fmin(1.0, mu[lo]+f*(mu[hi]-mu[lo])));
}

// Interpolated version (for A/B comparison)
__device__ double sample_angular_dist_interp(
    double E, double xi, Params p, int hit_nuc)
{
    int a_off = PTR_I(p, P_ANG_NUC_OFF)[hit_nuc];
    int a_ne = PTR_I(p, P_ANG_NUC_NE)[hit_nuc];
    if (a_ne <= 0) return 2.0*xi-1.0;
    const double* ae = &PTR_D(p, P_ANG_ENERGIES)[a_off];
    if (E <= ae[0]) {
        int off=PTR_I(p, P_ANG_DIST_OFF)[a_off], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off];
        if (sz<=1) return 2.0*xi-1.0;
        const double* mu=&PTR_D(p, P_ANG_MU)[off]; const double* cd=&PTR_D(p, P_ANG_CDF)[off];
        int lo=0,hi=sz-1; while(hi-lo>1){int mid=(lo+hi) >> 1;if(cd[mid]<=xi)lo=mid;else hi=mid;}
        double f=(xi-cd[lo])/fmax(cd[hi]-cd[lo],1e-30);
        return fmax(-1.0,fmin(1.0,mu[lo]+f*(mu[hi]-mu[lo])));
    }
    if (E >= ae[a_ne-1]) {
        int off=PTR_I(p, P_ANG_DIST_OFF)[a_off+a_ne-1], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off+a_ne-1];
        if (sz<=1) return 2.0*xi-1.0;
        const double* mu=&PTR_D(p, P_ANG_MU)[off]; const double* cd=&PTR_D(p, P_ANG_CDF)[off];
        int lo=0,hi=sz-1; while(hi-lo>1){int mid=(lo+hi) >> 1;if(cd[mid]<=xi)lo=mid;else hi=mid;}
        double f=(xi-cd[lo])/fmax(cd[hi]-cd[lo],1e-30);
        return fmax(-1.0,fmin(1.0,mu[lo]+f*(mu[hi]-mu[lo])));
    }
    int ie; { int lo=0,hi=a_ne-1;
        while(hi-lo>1){int mid=(lo+hi) >> 1;if(ae[mid]<=E)lo=mid;else hi=mid;} ie=lo; }
    double mu0, mu1;
    { int off=PTR_I(p, P_ANG_DIST_OFF)[a_off+ie], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off+ie];
      if(sz<=1){ mu0=2.0*xi-1.0; }
      else { const double* ma=&PTR_D(p, P_ANG_MU)[off]; const double* ca=&PTR_D(p, P_ANG_CDF)[off];
        int lo=0,hi=sz-1; while(hi-lo>1){int mid=(lo+hi) >> 1;if(ca[mid]<=xi)lo=mid;else hi=mid;}
        double f=(xi-ca[lo])/fmax(ca[hi]-ca[lo],1e-30); mu0=ma[lo]+f*(ma[hi]-ma[lo]); }
    }
    { int off=PTR_I(p, P_ANG_DIST_OFF)[a_off+ie+1], sz=PTR_I(p, P_ANG_DIST_SZ)[a_off+ie+1];
      if(sz<=1){ mu1=2.0*xi-1.0; }
      else { const double* mb=&PTR_D(p, P_ANG_MU)[off]; const double* cb=&PTR_D(p, P_ANG_CDF)[off];
        int lo=0,hi=sz-1; while(hi-lo>1){int mid=(lo+hi) >> 1;if(cb[mid]<=xi)lo=mid;else hi=mid;}
        double f=(xi-cb[lo])/fmax(cb[hi]-cb[lo],1e-30); mu1=mb[lo]+f*(mb[hi]-mb[lo]); }
    }
    double frac=(E-ae[ie])/(ae[ie+1]-ae[ie]);
    return fmax(-1.0,fmin(1.0,(1.0-frac)*mu0+frac*mu1));
}

// Diagnostic: reconstruct XS at given energies for a nuclide
// Output: 6 doubles per sample (elastic, inelastic, n2n, n3n, fission, capture)
extern "C" __global__ void debug_xs_reconstruct(
    Params p,
    const double* energies, int n_samples, int nuc_idx,
    double* out_xs) // [n_samples * N_REACTIONS]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;
    double E = energies[tid];
    int rank = SCALAR_I(p, P_RANK);
    int g_off = PTR_I(p, P_GRID_OFFSETS)[nuc_idx];
    int n_e = PTR_I(p, P_N_ENERGIES)[nuc_idx];
    int e_idx = energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off], n_e, E);
    for (int r = 0; r < N_REACTIONS; r++) {
        int key = nuc_idx * N_REACTIONS + r;
        double xs = 0.0;
        if (PTR_I(p, P_HAS_REACTION)[key]) {
            xs = svd_reconstruct(
                &PTR_D(p, P_BASIS)[PTR_I(p, P_BASIS_OFFSETS)[key]],
                &PTR_D(p, P_COEFFS)[PTR_I(p, P_COEFFS_OFFSETS)[key]],
                e_idx, rank);
        }
        out_xs[tid * N_REACTIONS + r] = xs;
    }
}


extern "C" __global__ void debug_angular_sample(
    Params p,
    const double* energies, const double* xis, int n_samples,
    int nuc_idx,
    double* out_stairstep, double* out_interp)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;
    double E = energies[tid];
    double xi = xis[tid];
    out_stairstep[tid] = sample_angular_dist_stairstep(E, xi, p, nuc_idx);
    out_interp[tid] = sample_angular_dist_interp(E, xi, p, nuc_idx);
}

extern "C" __global__ void init_source(
    double* px, double* py, double* pz,
    double* ddx, double* ddy, double* ddz,
    double* energy, int* cell_idx, int* alive,
    const double* sx, const double* sy, const double* sz, const double* se,
    int n, unsigned long long batch_seed,
    unsigned long long* rng_state, unsigned long long* rng_inc,
    int geom_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    px[tid]=sx[tid]; py[tid]=sy[tid]; pz[tid]=sz[tid];
    energy[tid]=se[tid];
    cell_idx[tid]=find_cell(sx[tid],sy[tid],sz[tid],geom_type);
    alive[tid]=1;
    PcgState rng;
    pcg_init(&rng, batch_seed+(unsigned long long)tid*100000ULL, (unsigned long long)tid);
    double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
    double st=sqrt(1.0-mu*mu);
    double s_phi, c_phi; sincos(phi, &s_phi, &c_phi);
    ddx[tid]=st*c_phi; ddy[tid]=st*s_phi; ddz[tid]=mu;
    rng_state[tid]=rng.state; rng_inc[tid]=rng.inc;
}

// ═══════════════════════════════════════════════════════════════════════
// PERSISTENT TRANSPORT KERNEL
// ═══════════════════════════════════════════════════════════════════════

// Forward declaration — defined after transport kernels, used by
// the hybrid XS path in transport_persistent.
__device__ void wmp_eval(
    double e, double t_kelvin,
    double e_min, double e_max, double spacing, double sqrt_awr,
    int n_windows, int fit_order, int fissionable,
    const double2* poles,
    const int* windows,
    const signed char* broaden_poly,
    const double* curvefit,
    double* out_s, double* out_a, double* out_f);

// Cap on discrete inelastic levels we walk on the legacy fallback
// path (used when no CDF is built — i.e. nuclides whose ENDF/B-VII.1
// evaluation provides MT=4 natively, like U-235). Keep at 64 to match
// historical behaviour.
#define LEGACY_LEV_CAP 64

extern "C" __global__ void __launch_bounds__(256, 2)
transport_persistent(
    Params p,
    const int* __restrict__ compact_idx, int n_alive,
    // Mutable particle state (SoA). `__restrict__` lets ptxas assume no
    // aliasing between these SoA arrays, freeing registers that would
    // otherwise have to re-reload invariants across the step loop
    // (NVIDIA BPG §10.2 — reduces stack spills under launch_bounds 256×2).
    double* __restrict__ pos_x, double* __restrict__ pos_y, double* __restrict__ pos_z,
    double* __restrict__ dir_x, double* __restrict__ dir_y, double* __restrict__ dir_z,
    double* __restrict__ energy, int* __restrict__ cell_idx, int* __restrict__ alive,
    unsigned long long* __restrict__ rng_state_arr, unsigned long long* __restrict__ rng_inc_arr,
    // Fission bank
    double* __restrict__ fis_x, double* __restrict__ fis_y, double* __restrict__ fis_z,
    double* __restrict__ fis_e, double* __restrict__ fis_w,
    int* __restrict__ fis_count, int max_fis,
    // Counters
    int* __restrict__ cnt_coll, int* __restrict__ cnt_fis,
    int* __restrict__ cnt_leak, int* __restrict__ cnt_surf,
    int steps_per_launch)
{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n_alive) return;
    int tid = compact_idx[lane];
    int gt = SCALAR_I(p, P_GEOM_TYPE);
    int rank = SCALAR_I(p, P_RANK);

    double px=pos_x[tid], py=pos_y[tid], pz=pos_z[tid];
    double dx=dir_x[tid], dy=dir_y[tid], dz=dir_z[tid];
    double E=energy[tid];
    int cell=cell_idx[tid], is_alive=alive[tid];

    PcgState rng;
    rng.state=rng_state_arr[tid]; rng.inc=rng_inc_arr[tid];
    int lcnt_coll=0, lcnt_fis=0, lcnt_leak=0, lcnt_surf=0;

    for (int step=0; step < steps_per_launch && is_alive; step++) {
        int mat = cell_material(cell, gt);

        // Void: stream to surface
        if (mat < 0) {
            double d_s; int bc, nc;
            trace_surface(px,py,pz,dx,dy,dz,cell,gt,&d_s,&bc,&nc);
            if (d_s > 1e19) { is_alive=0; lcnt_leak++; break; }
            lcnt_surf++;
            if (bc==BC_REFLECTIVE) {
                px+=dx*d_s; py+=dy*d_s; pz+=dz*d_s;
                if(fabs(pz-HALF_PITCH)<1e-6||fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if(fabs(px-HALF_PITCH)<1e-6||fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if(fabs(py-HALF_PITCH)<1e-6||fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc==BC_TRANSMISSION) {
                double nudge=fmax(d_s*1e-8,1e-8);
                px+=dx*(d_s+nudge); py+=dy*(d_s+nudge); pz+=dz*(d_s+nudge);
                if (nc>=0) cell=nc; else { is_alive=0; lcnt_leak++; break; }
            } else { is_alive=0; lcnt_leak++; break; }
            continue;
        }

        // XS lookup — track all 6 reactions separately
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t=0;
        double nuc_t[4]={}, nuc_el[4]={}, nuc_inel[4]={}, nuc_n2n[4]={};
        double nuc_n3n[4]={}, nuc_fis[4]={}, nuc_cap[4]={};
        double urr_xi = pcg_uniform(&rng);

        for (int i=0; i<n_nuc; i++) {
            int ni = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat*4+i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+i]);
            int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[ni]);
            int n_e = __ldg(&PTR_I(p, P_N_ENERGIES)[ni]);
            const double* grid = &PTR_D(p, P_ENERGY_GRIDS)[g_off];
            int e_idx = energy_index(grid, n_e, E);
            // Log-log interpolation fraction (OpenMC scheme)
            double log_frac = 0.0;
            if (e_idx + 1 < n_e && grid[e_idx] > 0.0) {
                double log_e = log(E);
                double log_lo = log(grid[e_idx]);
                double log_hi = log(grid[e_idx+1]);
                if (log_hi > log_lo) log_frac = (log_e - log_lo) / (log_hi - log_lo);
                if (log_frac < 0.0) log_frac = 0.0;
                if (log_frac > 1.0) log_frac = 1.0;
            }

            double s_el=0, s_inel=0, s_n2n=0, s_n3n=0, s_fis=0, s_cap=0, micro_t=0;

            if (__ldg(&PTR_I(p, P_HAS_PW)[ni])) {
                // Pointwise table lookup — exact HDF5 values, log-log interpolation
                int pw_off = __ldg(&PTR_I(p, P_PW_OFF)[ni]);
                const double* pw0 = &PTR_D(p, P_PW_XS)[pw_off + e_idx * 7];
                const double* pw1 = (e_idx+1 < n_e) ? &PTR_D(p, P_PW_XS)[pw_off + (e_idx+1) * 7] : pw0;
                double xs7[7];
                for (int ch=0; ch<7; ch++) {
                    double lo = pw0[ch], hi = pw1[ch];
                    xs7[ch] = (lo > 1e-30 && hi > 1e-30 && log_frac > 0.0)
                        ? exp(log(lo) + log_frac * (log(hi) - log(lo))) : lo;
                }
                s_el=xs7[0]; s_inel=xs7[1]; s_n2n=xs7[2]; s_n3n=xs7[3];
                s_fis=xs7[4]; s_cap=xs7[5]; micro_t=xs7[6];
                // Ensure capture absorbs any remainder
                double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
                s_cap = fmax(micro_t - partials, 0.0);
            } else {
                // SVD fallback
                bool has_inel_k = false;
                for (int r=0; r<6; r++) {
                    int key = ni*N_REACTIONS+r;
                    if (__ldg(&PTR_I(p, P_HAS_REACTION)[key])) {
                        double s = svd_reconstruct_interp(
                            &PTR_D(p, P_BASIS)[__ldg(&PTR_I(p, P_BASIS_OFFSETS)[key])],
                            &PTR_D(p, P_COEFFS)[__ldg(&PTR_I(p, P_COEFFS_OFFSETS)[key])],
                            e_idx, n_e, rank, log_frac);
                        if(r==0) s_el=s; else if(r==1) { s_inel=s; has_inel_k=true; }
                        else if(r==2) s_n2n=s; else if(r==3) s_n3n=s;
                        else if(r==4) s_fis=s; else if(r==5) s_cap=s;
                    }
                }
                // Match CPU: when MT=4 kernel is absent, synthesize inelastic by
                // summing discrete-level SVD reconstructions at this energy.
                if (!has_inel_k) {
                    int lv_off = __ldg(&PTR_I(p, P_LEVEL_OFFSETS)[ni]);
                    int n_lev  = __ldg(&PTR_I(p, P_LEVEL_COUNTS)[ni]);
                    double lsum = 0.0;
                    for (int l=0; l<n_lev; l++) {
                        int gl = lv_off + l;
                        if (!__ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) continue;
                        if (E < __ldg(&PTR_D(p, P_LEVEL_THR)[gl])) continue;
                        double lxs = svd_reconstruct_interp(
                            &PTR_D(p, P_LEVEL_BASIS)[__ldg(&PTR_I(p, P_LEVEL_BOFF)[gl])],
                            &PTR_D(p, P_LEVEL_COEFFS)[__ldg(&PTR_I(p, P_LEVEL_COFF)[gl])],
                            e_idx, n_e, rank, log_frac);
                        if (lxs > 0.0) lsum += lxs;
                    }
                    s_inel = lsum;
                }
                // Match CPU: if HDF5 total is available, set micro_t to the HDF5
                // total and reabsorb the delta into capture. This captures the
                // "missing" absorption channels (n,a / n,p / MT=19-21 etc.) that
                // the 6-channel SVD basis does not represent. Without this step,
                // U-238 resonance-region absorption is underestimated and k_inf
                // comes in ~+2500 pcm high.
                if (__ldg(&PTR_I(p, P_HAS_TOTAL_XS)[ni])) {
                    int t_off = __ldg(&PTR_I(p, P_TOTAL_XS_OFF)[ni]);
                    const double* tot_grid = &PTR_D(p, P_TOTAL_XS)[t_off];
                    double tot_lo = tot_grid[e_idx];
                    double tot_hi = (e_idx+1 < n_e) ? tot_grid[e_idx+1] : tot_lo;
                    double tot = (tot_lo > 1e-30 && tot_hi > 1e-30 && log_frac > 0.0)
                        ? exp(log(tot_lo) + log_frac*(log(tot_hi)-log(tot_lo)))
                        : tot_lo;
                    double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
                    s_cap = fmax(tot - partials, 0.0);
                    micro_t = tot;
                } else {
                    micro_t = s_el + s_inel + s_n2n + s_n3n + s_fis + s_cap;
                }
            }

            // Hybrid SVD+WMP: if this nuclide has WMP coverage and E is inside
            // the resolved-resonance window, replace elastic/fission/capture with
            // the windowed-multipole evaluation and suppress URR (resonances are
            // already explicit via poles). Outside the window the SVD/PW path
            // stands and URR applies as usual.
            bool in_wmp = false;
            if (__ldg(&PTR_I(p, P_WMP_HAS)[ni])) {
                double e_lo = __ldg(&PTR_D(p, P_WMP_E_MIN)[ni]);
                double e_hi = __ldg(&PTR_D(p, P_WMP_E_MAX)[ni]);
                if (E >= e_lo && E <= e_hi) in_wmp = true;
            }

            if (!in_wmp) {
                // URR — modifies s_el, s_fis, s_cap. Recompute micro_t to match CPU behavior.
                double prev_el = s_el, prev_fis = s_fis, prev_cap = s_cap;
                apply_urr(p, ni, &s_el, &s_fis, &s_cap, E, urr_xi);
                micro_t += (s_el - prev_el) + (s_fis - prev_fis) + (s_cap - prev_cap);
            } else {
                int pole_off = __ldg(&PTR_I(p, P_WMP_POLE_OFF)[ni]);
                int win_off  = __ldg(&PTR_I(p, P_WMP_WIN_OFF)[ni]);
                int bro_off  = __ldg(&PTR_I(p, P_WMP_BROADEN_OFF)[ni]);
                int cf_off   = __ldg(&PTR_I(p, P_WMP_CF_OFF)[ni]);
                double w_emin = __ldg(&PTR_D(p, P_WMP_E_MIN)[ni]);
                double w_emax = __ldg(&PTR_D(p, P_WMP_E_MAX)[ni]);
                double w_spc  = __ldg(&PTR_D(p, P_WMP_SPACING)[ni]);
                double w_sqra = __ldg(&PTR_D(p, P_WMP_SQRT_AWR)[ni]);
                double w_tk   = __ldg(&PTR_D(p, P_WMP_T_KELVIN)[ni]);
                int w_nw      = __ldg(&PTR_I(p, P_WMP_N_WINDOWS)[ni]);
                int w_fo      = __ldg(&PTR_I(p, P_WMP_FIT_ORDER)[ni]);
                int w_fiss    = __ldg(&PTR_I(p, P_WMP_FISSIONABLE)[ni]);

                const double2* w_poles    = PTR_D2(p, P_WMP_POLES) + pole_off;
                const int* w_windows      = PTR_I(p, P_WMP_WINDOWS) + win_off;
                const signed char* w_bro  = PTR_B(p, P_WMP_BROADEN) + bro_off;
                const double* w_curvefit  = PTR_D(p, P_WMP_CURVEFIT) + cf_off;

                double w_s = 0.0, w_a = 0.0, w_f = 0.0;
                wmp_eval(E, w_tk, w_emin, w_emax, w_spc, w_sqra,
                         w_nw, w_fo, w_fiss,
                         w_poles, w_windows, w_bro, w_curvefit,
                         &w_s, &w_a, &w_f);

                double new_el  = fmax(w_s, 0.0);
                double new_fis = fmax(w_f, 0.0);
                double new_cap = fmax(w_a - w_f, 0.0);
                micro_t = new_el + s_inel + s_n2n + s_n3n + new_fis + new_cap;
                s_el = new_el; s_fis = new_fis; s_cap = new_cap;
            }

            // S(alpha,beta) for H1 (nuclide idx 3 in PWR)
            if (ni==3 && E < SCALAR_D(p, P_SAB_EMAX) && E > 0.0 && SCALAR_I(p, P_SAB_N_INC) > 0) {
                double sab_xs_val = sab_total_xs(E, p);
                if (sab_xs_val > 0.0) {
                    double delta = sab_xs_val - s_el;
                    micro_t += delta;
                    s_el = sab_xs_val;
                }
            }
            // debug: uncomment to trace per-nuclide XS on first step
            // if (lane==0 && step==0) {
            //     printf("  nuc=%d Ni=%.6f el=%.4f inel=%.4f fis=%.4f cap=%.4f tot=%.4f E=%.2f pw=%d\n",
            //         ni, Ni, s_el, s_inel, s_fis, s_cap, micro_t, E, __ldg(&PTR_I(p, P_HAS_PW)[ni]));
            // }
            nuc_t[i]=Ni*micro_t; nuc_el[i]=Ni*s_el; nuc_inel[i]=Ni*s_inel;
            nuc_n2n[i]=Ni*s_n2n; nuc_n3n[i]=Ni*s_n3n;
            nuc_fis[i]=Ni*s_fis; nuc_cap[i]=Ni*s_cap;
            sum_t+=Ni*micro_t;
        }

        if (sum_t <= 0.0) { is_alive=0; break; }

        double d_coll = -log(pcg_uniform(&rng)) / sum_t;
        double d_s; int bc, nc;
        trace_surface(px,py,pz,dx,dy,dz,cell,gt,&d_s,&bc,&nc);

        if (d_s < d_coll) {
            // Surface crossing
            lcnt_surf++;
            if (bc==BC_REFLECTIVE) {
                px+=dx*d_s; py+=dy*d_s; pz+=dz*d_s;
                if(fabs(pz-HALF_PITCH)<1e-6||fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if(fabs(px-HALF_PITCH)<1e-6||fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if(fabs(py-HALF_PITCH)<1e-6||fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc==BC_TRANSMISSION) {
                double nudge=fmax(d_s*1e-8,1e-8);
                px+=dx*(d_s+nudge); py+=dy*(d_s+nudge); pz+=dz*(d_s+nudge);
                if (nc>=0) cell=nc; else { is_alive=0; lcnt_leak++; break; }
            } else { is_alive=0; lcnt_leak++; break; }
        } else {
            // Collision
            lcnt_coll++;
            px+=dx*d_coll; py+=dy*d_coll; pz+=dz*d_coll;
            cell = find_cell(px,py,pz,gt);
            if (cell<0) { is_alive=0; lcnt_leak++; break; }

            // Sample nuclide
            double xi_nuc = pcg_uniform(&rng) * sum_t;
            double cum=0.0; int hit_l=0;
            for (int i=0; i<n_nuc; i++) { cum+=nuc_t[i]; if(xi_nuc<cum){hit_l=i;break;} hit_l=i; }
            int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat*4+hit_l]);
            double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

            // Sample reaction — order matches CPU: el, inel, n2n, n3n, fis, cap
            double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
            double cum_rxn = 0.0;

            cum_rxn += nuc_el[hit_l];
            if (xi_rxn < cum_rxn) {
                // ═══ Elastic scattering ═══

                // S(alpha,beta) for H1
                if (hit_nuc==3 && E < SCALAR_D(p, P_SAB_EMAX) && SCALAR_I(p, P_SAB_N_INC) > 0) {
                    double E_sab, mu_sab;
                    sab_sample(E, &rng, p, &E_sab, &mu_sab);
                    E = fmax(E_sab, 1e-11);
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_sab,phi);
                    goto end_coll;
                }

                // Free-gas thermal for light nuclides
                double cell_kT = (cell==0 && gt==GEOM_PWR) ? 900.0*8.617333262e-5 : 600.0*8.617333262e-5;
                if (gt==GEOM_GODIVA) cell_kT = 294.0*8.617333262e-5;

                if (E < 400.0*cell_kT) {
                    double sigma=sqrt(cell_kT/A), v_n=sqrt(2.0*E);
                    double u1,u2,r_bm,th;
                    u1=pcg_uniform(&rng); u2=pcg_uniform(&rng);
                    r_bm=sigma*sqrt(-2.0*log(fmax(u1,1e-30))); th=2.0*PI*u2;
                    double vtx=r_bm*cos(th), vty=r_bm*sin(th);
                    u1=pcg_uniform(&rng); u2=pcg_uniform(&rng);
                    r_bm=sigma*sqrt(-2.0*log(fmax(u1,1e-30))); th=2.0*PI*u2;
                    double vtz=r_bm*cos(th);
                    double vnx=dx*v_n, vny=dy*v_n, vnz=dz*v_n;
                    double vrx=vnx-vtx, vry=vny-vty, vrz=vnz-vtz;
                    double vr=sqrt(vrx*vrx+vry*vry+vrz*vrz);
                    if(vr<1e-20) vr=1e-20;
                    double ia1=1.0/(1.0+A);
                    double vcx=(vnx+A*vtx)*ia1, vcy=(vny+A*vty)*ia1, vcz=(vnz+A*vtz)*ia1;
                    double vcn=vr*A*ia1;
                    // Angular dist at relative energy (matches CPU free_gas_scatter)
                    double e_rel=0.5*(A/(A+1.0))*vr*vr;
                    double mu_cm=sample_angular_dist(e_rel,&rng,p,hit_nuc);
                    double phi=2.0*PI*pcg_uniform(&rng);
                    double st=sqrt(fmax(0.0,1.0-mu_cm*mu_cm));
                    double vrh_x=vrx/vr, vrh_y=vry/vr, vrh_z=vrz/vr;
                    double px2,py2,pz2;
                    if(fabs(vrh_z)<0.999){
                        double ip=1.0/sqrt(1.0-vrh_z*vrh_z);
                        px2=-vrh_y*ip; py2=vrh_x*ip; pz2=0.0;
                    } else {
                        double ip=1.0/sqrt(1.0-vrh_x*vrh_x);
                        px2=0.0; py2=-vrh_z*ip; pz2=vrh_y*ip;
                    }
                    double qx=vrh_y*pz2-vrh_z*py2, qy=vrh_z*px2-vrh_x*pz2, qz=vrh_x*py2-vrh_y*px2;
                    double s_phi, c_phi; sincos(phi, &s_phi, &c_phi);
                    double sx2=mu_cm*vrh_x+st*(c_phi*px2+s_phi*qx);
                    double sy2=mu_cm*vrh_y+st*(c_phi*py2+s_phi*qy);
                    double sz2=mu_cm*vrh_z+st*(c_phi*pz2+s_phi*qz);
                    double vox=vcx+vcn*sx2, voy=vcy+vcn*sy2, voz=vcz+vcn*sz2;
                    double vo=sqrt(vox*vox+voy*voy+voz*voz);
                    E=0.5*vo*vo; if(E<1e-11)E=1e-11;
                    if(vo>1e-20){dx=vox/vo;dy=voy/vo;dz=voz/vo;}
                } else {
                    // Anisotropic two-body elastic
                    double mu_cm = sample_angular_dist(E, &rng, p, hit_nuc);
                    double alpha=((A-1.0)/(A+1.0))*((A-1.0)/(A+1.0));
                    E=E*(1.0+alpha+(1.0-alpha)*mu_cm)/2.0;
                    if(E<1e-11) E=1e-11;
                    // CPU-matching mu_lab: hydrogen (A<=1+eps) uses special case
                    double mu_lab = (A > 1.0 + 1e-10)
                        ? (1.0+A*mu_cm)/sqrt(1.0+A*A+2.0*A*mu_cm)
                        : sqrt(fmax(0.0, (1.0+mu_cm)*0.5));
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_lab,phi);
                }

            } else if ((cum_rxn+=nuc_inel[hit_l]), xi_rxn < cum_rxn) {
                // ═══ Inelastic — proper discrete level sampling ═══
                goto do_inelastic;

            } else if ((cum_rxn+=nuc_n2n[hit_l]), xi_rxn < cum_rxn) {
                // ═══ (n,2n) — bank 1 extra neutron, primary continues ═══
                { double temp=E/10.0;
                  double x1=fmax(pcg_uniform(&rng),1e-30), x2=fmax(pcg_uniform(&rng),1e-30);
                  double e_sec=fmax(fmin(-temp*log(x1*x2),E),1e-5);
                  int idx2=atomicAdd(fis_count,1);
                  if(idx2<max_fis){ fis_x[idx2]=px;fis_y[idx2]=py;fis_z[idx2]=pz;fis_e[idx2]=e_sec;fis_w[idx2]=1.0; }
                }
                { double Q_n2n=-E*0.1, e_cm=E*A/(A+1.0), e_cm_out=e_cm+Q_n2n;
                  if(e_cm_out<=0.0) e_cm_out=E*0.01;
                  double mu_cm=2.0*pcg_uniform(&rng)-1.0, ap1=A+1.0;
                  double e_n=e_cm_out*A/ap1, vni=sqrt(2.0*e_n), vcs=sqrt(2.0*E/(ap1*ap1));
                  double v2=vni*vni+vcs*vcs+2.0*vni*vcs*mu_cm;
                  E=fmax(0.5*v2,1e-5);
                  double den=sqrt(fmax(v2,1e-40));
                  double ml=(vni+vcs>1e-20)?fmax(-1.0,fmin(1.0,(vcs+vni*mu_cm)/den)):2.0*pcg_uniform(&rng)-1.0;
                  double phi=2.0*PI*pcg_uniform(&rng);
                  rotate_direction(&dx,&dy,&dz,ml,phi);
                }

            } else if ((cum_rxn+=nuc_n3n[hit_l]), xi_rxn < cum_rxn) {
                // ═══ (n,3n) — bank 2 extra neutrons, primary continues ═══
                for(int ns3=0;ns3<2;ns3++){
                  double temp=E/10.0;
                  double x1=fmax(pcg_uniform(&rng),1e-30), x2=fmax(pcg_uniform(&rng),1e-30);
                  double e_sec=fmax(fmin(-temp*log(x1*x2),E),1e-5);
                  int idx2=atomicAdd(fis_count,1);
                  if(idx2<max_fis){ fis_x[idx2]=px;fis_y[idx2]=py;fis_z[idx2]=pz;fis_e[idx2]=e_sec;fis_w[idx2]=1.0; }
                }
                { double Q_n3n=-E*0.2, e_cm=E*A/(A+1.0), e_cm_out=e_cm+Q_n3n;
                  if(e_cm_out<=0.0) e_cm_out=E*0.01;
                  double mu_cm=2.0*pcg_uniform(&rng)-1.0, ap1=A+1.0;
                  double e_n=e_cm_out*A/ap1, vni=sqrt(2.0*e_n), vcs=sqrt(2.0*E/(ap1*ap1));
                  double v2=vni*vni+vcs*vcs+2.0*vni*vcs*mu_cm;
                  E=fmax(0.5*v2,1e-5);
                  double den=sqrt(fmax(v2,1e-40));
                  double ml=(vni+vcs>1e-20)?fmax(-1.0,fmin(1.0,(vcs+vni*mu_cm)/den)):2.0*pcg_uniform(&rng)-1.0;
                  double phi=2.0*PI*pcg_uniform(&rng);
                  rotate_direction(&dx,&dy,&dz,ml,phi);
                }

            } else if ((cum_rxn+=nuc_fis[hit_l]), xi_rxn < cum_rxn) {
                // ═══ Fission ═══
                lcnt_fis++;
                int nb_off=__ldg(&PTR_I(p, P_NB_OFFSETS)[hit_nuc]);
                int nb_sz=__ldg(&PTR_I(p, P_NB_SIZES)[hit_nuc]);
                double nu = (nb_sz>0) ?
                    nu_bar_lookup(E,PTR_D(p, P_NB_ENERGIES),PTR_D(p, P_NB_VALUES),nb_off,nb_sz) :
                    __ldg(&PTR_D(p, P_NU_BAR_CONST)[hit_nuc]);
                int ns=(int)nu; if(pcg_uniform(&rng)<(nu-(double)ns)) ns++;
                for(int s=0;s<ns;s++){
                    int idx=atomicAdd(fis_count,1);
                    if(idx<max_fis){
                        fis_x[idx]=px; fis_y[idx]=py; fis_z[idx]=pz;
                        fis_e[idx]=sample_fission_energy(E,&rng,p,hit_nuc);
                        fis_w[idx]=1.0;
                    }
                }
                is_alive=0;

            } else {
                // ═══ Capture (remainder) ═══
                is_alive=0;
                goto end_coll;
            }
            if(0) { do_inelastic:
                // ═══ Inelastic — proper discrete level sampling ═══
                int lv_off=__ldg(&PTR_I(p, P_LEVEL_OFFSETS)[hit_nuc]);
                int n_lev=__ldg(&PTR_I(p, P_LEVEL_COUNTS)[hit_nuc]);
                double Q=-0.5e6; int selected=0;
                // ── Fast path: pre-tabulated per-level CDF ──
                // Replaces the 13-level walk (Pass 1 sum + Pass 2 select)
                // with a single linear scan over a log-decimated CDF.
                // Active when `inel_cdf_off >= 0` (set by the loader for
                // nuclides with synthesized MT=4 — Zr-90/91/92/94, U-238).
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
                    int idx = (int)f_idx;
                    if (idx >= cdf_n_e - 1) idx = cdf_n_e - 2;
                    if (idx < 0) idx = 0;
                    double alpha = f_idx - (double)idx;
                    const double* cdf_base = &PTR_D(p, P_INEL_CDF_DATA)[cdf_off];
                    double xi_l = pcg_uniform(&rng);
                    int sampled = cdf_n_lev - 1;
                    int row_lo = idx       * cdf_n_lev;
                    int row_hi = (idx + 1) * cdf_n_lev;
                    #pragma unroll 1
                    for (int l = 0; l < cdf_n_lev - 1; l++) {
                        double F = cdf_base[row_lo + l]
                                 + alpha * (cdf_base[row_hi + l] - cdf_base[row_lo + l]);
                        if (xi_l <= F) { sampled = l; break; }
                    }
                    selected = sampled;
                    Q = __ldg(&PTR_D(p, P_LEVEL_Q)[lv_off + selected]);
                } else if (n_lev > 0) {
                    // Legacy fallback: per-level walk (Pass 1 sum + Pass 2
                    // select) for nuclides whose ENDF/B-VII.1 evaluation
                    // provides MT=4 natively (e.g. U-235), where the
                    // synthesis path doesn't fire and no CDF is built.
                    //
                    // Two-pass single-scan keeps registers tight (NVIDIA
                    // BPG §10.2: runtime-indexed local arrays spill to
                    // DRAM, so we recompute the SVD in Pass 2 instead).
                    // The doubled SVD cost was previously addressed with
                    // a per-warp shared-memory cache; it has been removed
                    // because synthesis + CDF eliminate this code path on
                    // the geometries where it dominated (Zr clad / U-238).
                    // The cache falsification result is preserved in
                    // paper §threats.
                    double lxs_sum=0.0;
                    int g_off=__ldg(&PTR_I(p, P_GRID_OFFSETS)[hit_nuc]);
                    int n_e=__ldg(&PTR_I(p, P_N_ENERGIES)[hit_nuc]);
                    int e_idx=energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off],n_e,E);
                    int lev_cap = n_lev < LEGACY_LEV_CAP ? n_lev : LEGACY_LEV_CAP;
                    #pragma unroll 1
                    for(int l=0;l<lev_cap;l++){
                        int gl=lv_off+l;
                        if(E>=__ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                           && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])){
                            lxs_sum += svd_reconstruct(
                                &PTR_D(p, P_LEVEL_BASIS)[__ldg(&PTR_I(p, P_LEVEL_BOFF)[gl])],
                                &PTR_D(p, P_LEVEL_COEFFS)[__ldg(&PTR_I(p, P_LEVEL_COFF)[gl])],
                                e_idx, rank);
                        }
                    }
                    if(lxs_sum>0.0){
                        double xi_l=pcg_uniform(&rng)*lxs_sum;
                        double run=0.0;
                        selected = lev_cap - 1;
                        #pragma unroll 1
                        for(int l=0;l<lev_cap;l++){
                            int gl=lv_off+l; double lxs=0.0;
                            if(E>=__ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                               && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])){
                                lxs=svd_reconstruct(
                                    &PTR_D(p, P_LEVEL_BASIS)[__ldg(&PTR_I(p, P_LEVEL_BOFF)[gl])],
                                    &PTR_D(p, P_LEVEL_COEFFS)[__ldg(&PTR_I(p, P_LEVEL_COFF)[gl])],
                                    e_idx, rank);
                            }
                            run += lxs;
                            if (xi_l < run) { selected = l; break; }
                        }
                        Q=__ldg(&PTR_D(p, P_LEVEL_Q)[lv_off+selected]);
                    }
                }
                int sel_mt=(n_lev>0)?__ldg(&PTR_I(p, P_LEVEL_MT)[lv_off+selected]):0;
                // Continuum (MT=91): compute effective Q from evaporation model
                if(sel_mt==91){
                    double a_p=A/8.0;
                    double ecm_mev=E*A/((A+1.0)*1e6);
                    double eex=fmax(ecm_mev,0.1);
                    double T=sqrt(eex/a_p);
                    double x1=fmax(pcg_uniform(&rng),1e-30), x2=fmax(pcg_uniform(&rng),1e-30);
                    double eo=-T*log(x1*x2);
                    eo=fmin(eo,ecm_mev*0.9);
                    Q = -(ecm_mev - eo)*1e6;
                }
                // Two-body inelastic kinematics (matches CPU inelastic_scatter)
                double e_cm = E * A / (A + 1.0);
                double e_cm_out = e_cm + Q;
                if(e_cm_out <= 0.0) {
                    // Below threshold — elastic fallback
                    double mu_fb = 2.0*pcg_uniform(&rng)-1.0;
                    double alpha=((A-1.0)/(A+1.0))*((A-1.0)/(A+1.0));
                    E=E*(1.0+alpha+(1.0-alpha)*mu_fb)/2.0;
                    if(E<1e-11) E=1e-11;
                    // CPU-matching mu_lab: hydrogen (A<=1+eps) uses special case
                    double mu_lab = (A > 1.0 + 1e-10)
                        ? (1.0+A*mu_fb)/sqrt(1.0+A*A+2.0*A*mu_fb)
                        : sqrt(fmax(0.0, (1.0+mu_fb)*0.5));
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_lab,phi);
                } else {
                    // Prefer per-level ENDF angular distribution (MT=51-91)
                    // when the evaluation stored one. Continuum MT=91 and
                    // "no tabulated data" paths fall back to isotropic CM.
                    double mu_cm;
                    if (n_lev > 0 && sel_mt != 91) {
                        mu_cm = sample_level_angular(E, &rng, p, lv_off + selected);
                    } else {
                        mu_cm = 2.0*pcg_uniform(&rng) - 1.0;
                    }
                    double ap1 = A + 1.0;
                    double e_n_cm = e_cm_out * A / ap1;
                    double v_n_i = sqrt(2.0 * e_n_cm);
                    double v_cm_s = sqrt(2.0 * E / (ap1 * ap1));
                    double v2sum = v_n_i*v_n_i + v_cm_s*v_cm_s + 2.0*v_n_i*v_cm_s*mu_cm;
                    E = fmax(0.5 * v2sum, 1e-5);
                    double denom = sqrt(fmax(v2sum, 1e-40));
                    double mu_lab;
                    if(v_n_i + v_cm_s > 1e-20) {
                        mu_lab = (v_cm_s + v_n_i*mu_cm) / denom;
                        mu_lab = fmax(-1.0, fmin(1.0, mu_lab));
                    } else {
                        mu_lab = 2.0*pcg_uniform(&rng)-1.0;
                    }
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_lab,phi);
                }
            } // end do_inelastic
            end_coll: ;
        }
    }

    pos_x[tid]=px; pos_y[tid]=py; pos_z[tid]=pz;
    dir_x[tid]=dx; dir_y[tid]=dy; dir_z[tid]=dz;
    energy[tid]=E; cell_idx[tid]=cell; alive[tid]=is_alive;
    rng_state_arr[tid]=rng.state; rng_inc_arr[tid]=rng.inc;

    // Warp-level reduction
    unsigned mask = __activemask();
    for(int off=16;off>0;off/=2){
        lcnt_coll+=__shfl_down_sync(mask,lcnt_coll,off);
        lcnt_fis+=__shfl_down_sync(mask,lcnt_fis,off);
        lcnt_leak+=__shfl_down_sync(mask,lcnt_leak,off);
        lcnt_surf+=__shfl_down_sync(mask,lcnt_surf,off);
    }
    if((threadIdx.x&31)==0){
        if(lcnt_coll>0)atomicAdd(cnt_coll,lcnt_coll);
        if(lcnt_fis>0)atomicAdd(cnt_fis,lcnt_fis);
        if(lcnt_leak>0)atomicAdd(cnt_leak,lcnt_leak);
        if(lcnt_surf>0)atomicAdd(cnt_surf,lcnt_surf);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DEBUG TRACE KERNEL — logs every transport step for GPU-CPU comparison
//
// Each particle writes max_steps rows. Per row:
//   [energy, pos_x, pos_y, pos_z, cell, material,
//    macro_total, d_coll, d_surf, event_type,
//    hit_nuc, micro_el, micro_inel, micro_fis, micro_cap,
//    outgoing_energy, rng_uniform_1]
//
// event_type: 0=elastic, 1=inelastic, 2=n2n, 3=n3n, 4=fission,
//             5=capture, 6=reflective, 7=transmission, 8=leak, 9=void_stream
// ═══════════════════════════════════════════════════════════════════════
#define TRACE_COLS 17

extern "C" __global__ void debug_transport_trace(
    Params p,
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy, int* cell_idx, int* alive,
    unsigned long long* rng_state_arr, unsigned long long* rng_inc_arr,
    double* fis_x, double* fis_y, double* fis_z,
    double* fis_e, double* fis_w,
    int* fis_count, int max_fis,
    // Trace output
    double* trace,          // [n_particles * max_steps * TRACE_COLS]
    int* step_counts,       // [n_particles]: actual steps taken
    int n_particles, int max_steps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    int gt = SCALAR_I(p, P_GEOM_TYPE);
    int rank = SCALAR_I(p, P_RANK);

    double px=pos_x[tid], py=pos_y[tid], pz=pos_z[tid];
    double dx=dir_x[tid], dy=dir_y[tid], dz=dir_z[tid];
    double E=energy[tid];
    int cell=cell_idx[tid], is_alive=alive[tid];

    PcgState rng;
    rng.state=rng_state_arr[tid]; rng.inc=rng_inc_arr[tid];

    int row_base = tid * max_steps * TRACE_COLS;
    int actual_steps = 0;

    for (int step=0; step < max_steps && is_alive; step++) {
        int row = row_base + step * TRACE_COLS;
        int mat = cell_material(cell, gt);

        // Record pre-step state
        trace[row+0] = E;
        trace[row+1] = px; trace[row+2] = py; trace[row+3] = pz;
        trace[row+4] = (double)cell;
        trace[row+5] = (double)mat;

        if (mat < 0) {
            // Void streaming
            double d_s; int bc, nc;
            trace_surface(px,py,pz,dx,dy,dz,cell,gt,&d_s,&bc,&nc);
            trace[row+6] = 0.0;  // no macro_total in void
            trace[row+7] = 0.0;
            trace[row+8] = d_s;
            trace[row+9] = 9.0;  // void_stream
            if (d_s > 1e19) { trace[row+9]=8.0; is_alive=0; break; }
            if (bc==BC_REFLECTIVE) {
                px+=dx*d_s; py+=dy*d_s; pz+=dz*d_s;
                if(fabs(pz-HALF_PITCH)<1e-6||fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if(fabs(px-HALF_PITCH)<1e-6||fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if(fabs(py-HALF_PITCH)<1e-6||fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc==BC_TRANSMISSION) {
                double nudge=fmax(d_s*1e-8,1e-8);
                px+=dx*(d_s+nudge); py+=dy*(d_s+nudge); pz+=dz*(d_s+nudge);
                if (nc>=0) cell=nc; else { trace[row+9]=8.0; is_alive=0; break; }
            } else { trace[row+9]=8.0; is_alive=0; break; }
            trace[row+15] = E; // outgoing energy unchanged
            actual_steps++;
            continue;
        }

        // XS lookup — same as transport_persistent
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t=0;
        double nuc_t[4]={}, nuc_el[4]={}, nuc_inel[4]={}, nuc_n2n[4]={};
        double nuc_n3n[4]={}, nuc_fis[4]={}, nuc_cap[4]={};
        double urr_xi = pcg_uniform(&rng);

        for (int i=0; i<n_nuc; i++) {
            int ni = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat*4+i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+i]);
            int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[ni]);
            int n_e = __ldg(&PTR_I(p, P_N_ENERGIES)[ni]);
            const double* grid = &PTR_D(p, P_ENERGY_GRIDS)[g_off];
            int e_idx = energy_index(grid, n_e, E);
            double log_frac = 0.0;
            if (e_idx + 1 < n_e && grid[e_idx] > 0.0) {
                double log_e = log(E), log_lo = log(grid[e_idx]), log_hi = log(grid[e_idx+1]);
                if (log_hi > log_lo) log_frac = (log_e - log_lo) / (log_hi - log_lo);
                if (log_frac < 0.0) log_frac = 0.0;
                if (log_frac > 1.0) log_frac = 1.0;
            }
            double s_el=0, s_inel=0, s_n2n=0, s_n3n=0, s_fis=0, s_cap=0, micro_t=0;
            if (__ldg(&PTR_I(p, P_HAS_PW)[ni])) {
                int pw_off = __ldg(&PTR_I(p, P_PW_OFF)[ni]);
                const double* pw0 = &PTR_D(p, P_PW_XS)[pw_off + e_idx * 7];
                const double* pw1 = (e_idx+1 < n_e) ? &PTR_D(p, P_PW_XS)[pw_off + (e_idx+1) * 7] : pw0;
                double xs7[7];
                for (int ch=0; ch<7; ch++) {
                    double lo = pw0[ch], hi = pw1[ch];
                    xs7[ch] = (lo > 1e-30 && hi > 1e-30 && log_frac > 0.0)
                        ? exp(log(lo) + log_frac * (log(hi) - log(lo))) : lo;
                }
                s_el=xs7[0]; s_inel=xs7[1]; s_n2n=xs7[2]; s_n3n=xs7[3];
                s_fis=xs7[4]; s_cap=xs7[5]; micro_t=xs7[6];
                double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
                s_cap = fmax(micro_t - partials, 0.0);
            } else {
                for (int r=0; r<6; r++) {
                    int key = ni*N_REACTIONS+r;
                    if (__ldg(&PTR_I(p, P_HAS_REACTION)[key])) {
                        double s = svd_reconstruct_interp(
                            &PTR_D(p, P_BASIS)[__ldg(&PTR_I(p, P_BASIS_OFFSETS)[key])],
                            &PTR_D(p, P_COEFFS)[__ldg(&PTR_I(p, P_COEFFS_OFFSETS)[key])],
                            e_idx, n_e, rank, log_frac);
                        if(r==0) s_el=s; else if(r==1) s_inel=s;
                        else if(r==2) s_n2n=s; else if(r==3) s_n3n=s;
                        else if(r==4) s_fis=s; else if(r==5) s_cap=s;
                    }
                }
                // Same CPU-parity fix as main transport: set micro_t to HDF5 total
                // and reabsorb delta into capture.
                if (__ldg(&PTR_I(p, P_HAS_TOTAL_XS)[ni])) {
                    int t_off = __ldg(&PTR_I(p, P_TOTAL_XS_OFF)[ni]);
                    const double* tot_grid = &PTR_D(p, P_TOTAL_XS)[t_off];
                    double tot_lo = tot_grid[e_idx];
                    double tot_hi = (e_idx+1 < n_e) ? tot_grid[e_idx+1] : tot_lo;
                    double tot = (tot_lo > 1e-30 && tot_hi > 1e-30 && log_frac > 0.0)
                        ? exp(log(tot_lo) + log_frac*(log(tot_hi)-log(tot_lo)))
                        : tot_lo;
                    double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
                    s_cap = fmax(tot - partials, 0.0);
                    micro_t = tot;
                } else {
                    micro_t = s_el + s_inel + s_n2n + s_n3n + s_fis + s_cap;
                }
            }
            // URR — recompute micro_t via delta
            {
                double prev_el = s_el, prev_fis = s_fis, prev_cap = s_cap;
                apply_urr(p, ni, &s_el, &s_fis, &s_cap, E, urr_xi);
                micro_t += (s_el - prev_el) + (s_fis - prev_fis) + (s_cap - prev_cap);
            }
            if (ni==3 && E < SCALAR_D(p, P_SAB_EMAX) && E > 0.0 && SCALAR_I(p, P_SAB_N_INC) > 0) {
                double sab_xs_val = sab_total_xs(E, p);
                if (sab_xs_val > 0.0) {
                    double delta = sab_xs_val - s_el;
                    micro_t += delta;
                    s_el = sab_xs_val;
                }
            }
            nuc_t[i]=Ni*micro_t; nuc_el[i]=Ni*s_el; nuc_inel[i]=Ni*s_inel;
            nuc_n2n[i]=Ni*s_n2n; nuc_n3n[i]=Ni*s_n3n;
            nuc_fis[i]=Ni*s_fis; nuc_cap[i]=Ni*s_cap;
            sum_t+=Ni*micro_t;
        }
        if (sum_t <= 0.0) { trace[row+9]=8.0; is_alive=0; actual_steps++; break; }

        trace[row+6] = sum_t; // macro_total

        double xi_coll = pcg_uniform(&rng);
        double d_coll = -log(xi_coll) / sum_t;
        trace[row+16] = xi_coll; // save the RNG value used for collision distance

        double d_s; int bc, nc;
        trace_surface(px,py,pz,dx,dy,dz,cell,gt,&d_s,&bc,&nc);
        trace[row+7] = d_coll;
        trace[row+8] = d_s;

        if (d_s < d_coll) {
            // Surface crossing
            if (bc==BC_REFLECTIVE) {
                trace[row+9] = 6.0;
                px+=dx*d_s; py+=dy*d_s; pz+=dz*d_s;
                if(fabs(pz-HALF_PITCH)<1e-6||fabs(pz+HALF_PITCH)<1e-6) dz=-dz;
                else if(fabs(px-HALF_PITCH)<1e-6||fabs(px+HALF_PITCH)<1e-6) dx=-dx;
                else if(fabs(py-HALF_PITCH)<1e-6||fabs(py+HALF_PITCH)<1e-6) dy=-dy;
            } else if (bc==BC_TRANSMISSION) {
                trace[row+9] = 7.0;
                double nudge=fmax(d_s*1e-8,1e-8);
                px+=dx*(d_s+nudge); py+=dy*(d_s+nudge); pz+=dz*(d_s+nudge);
                if (nc>=0) cell=nc; else { trace[row+9]=8.0; is_alive=0; }
            } else { trace[row+9]=8.0; is_alive=0; }
            trace[row+15] = E;
        } else {
            // Collision
            px+=dx*d_coll; py+=dy*d_coll; pz+=dz*d_coll;
            cell = find_cell(px,py,pz,gt);
            if (cell<0) { trace[row+9]=8.0; is_alive=0; actual_steps++; break; }

            double xi_nuc = pcg_uniform(&rng) * sum_t;
            double cum=0.0; int hit_l=0;
            for (int i=0; i<n_nuc; i++) { cum+=nuc_t[i]; if(xi_nuc<cum){hit_l=i;break;} hit_l=i; }
            int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat*4+hit_l]);
            double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

            trace[row+10] = (double)hit_nuc;
            trace[row+11] = nuc_el[hit_l] / __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+hit_l]);
            trace[row+12] = nuc_inel[hit_l] / __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+hit_l]);
            trace[row+13] = nuc_fis[hit_l] / __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+hit_l]);
            trace[row+14] = nuc_cap[hit_l] / __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+hit_l]);

            double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
            double cum_rxn = 0.0;

            cum_rxn += nuc_el[hit_l];
            if (xi_rxn < cum_rxn) {
                trace[row+9] = 0.0; // elastic
                if (hit_nuc==3 && E < SCALAR_D(p, P_SAB_EMAX) && SCALAR_I(p, P_SAB_N_INC) > 0) {
                    double E_sab, mu_sab;
                    sab_sample(E, &rng, p, &E_sab, &mu_sab);
                    E = fmax(E_sab, 1e-11);
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_sab,phi);
                } else {
                    double cell_kT = (cell==0 && gt==GEOM_PWR) ? 900.0*8.617333262e-5 : 600.0*8.617333262e-5;
                    if (gt==GEOM_GODIVA) cell_kT = 294.0*8.617333262e-5;
                    if (E < 400.0*cell_kT) {
                        double sigma=sqrt(cell_kT/A), v_n=sqrt(2.0*E);
                        double u1,u2,r_bm,th;
                        u1=pcg_uniform(&rng); u2=pcg_uniform(&rng);
                        r_bm=sigma*sqrt(-2.0*log(fmax(u1,1e-30))); th=2.0*PI*u2;
                        double vtx=r_bm*cos(th), vty=r_bm*sin(th);
                        u1=pcg_uniform(&rng); u2=pcg_uniform(&rng);
                        r_bm=sigma*sqrt(-2.0*log(fmax(u1,1e-30))); th=2.0*PI*u2;
                        double vtz=r_bm*cos(th);
                        double vnx=dx*v_n, vny=dy*v_n, vnz=dz*v_n;
                        double vrx=vnx-vtx, vry=vny-vty, vrz=vnz-vtz;
                        double vr=sqrt(vrx*vrx+vry*vry+vrz*vrz);
                        if(vr<1e-20) vr=1e-20;
                        double ia1=1.0/(1.0+A);
                        double vcx=(vnx+A*vtx)*ia1, vcy=(vny+A*vty)*ia1, vcz=(vnz+A*vtz)*ia1;
                        double vcn=vr*A*ia1;
                        double e_rel=0.5*(A/(A+1.0))*vr*vr;
                        double mu_cm=sample_angular_dist(e_rel,&rng,p,hit_nuc);
                        double phi=2.0*PI*pcg_uniform(&rng);
                        double st=sqrt(fmax(0.0,1.0-mu_cm*mu_cm));
                        double vrh_x=vrx/vr, vrh_y=vry/vr, vrh_z=vrz/vr;
                        double px2,py2,pz2;
                        if(fabs(vrh_z)<0.999){
                            double ip=1.0/sqrt(1.0-vrh_z*vrh_z);
                            px2=-vrh_y*ip; py2=vrh_x*ip; pz2=0.0;
                        } else {
                            double ip=1.0/sqrt(1.0-vrh_x*vrh_x);
                            px2=0.0; py2=-vrh_z*ip; pz2=vrh_y*ip;
                        }
                        double qx=vrh_y*pz2-vrh_z*py2, qy=vrh_z*px2-vrh_x*pz2, qz=vrh_x*py2-vrh_y*px2;
                        double s_phi, c_phi; sincos(phi, &s_phi, &c_phi);
                        double sx2=mu_cm*vrh_x+st*(c_phi*px2+s_phi*qx);
                        double sy2=mu_cm*vrh_y+st*(c_phi*py2+s_phi*qy);
                        double sz2=mu_cm*vrh_z+st*(c_phi*pz2+s_phi*qz);
                        double vox=vcx+vcn*sx2, voy=vcy+vcn*sy2, voz=vcz+vcn*sz2;
                        double vo=sqrt(vox*vox+voy*voy+voz*voz);
                        E=0.5*vo*vo; if(E<1e-11)E=1e-11;
                        if(vo>1e-20){dx=vox/vo;dy=voy/vo;dz=voz/vo;}
                    } else {
                        double mu_cm = sample_angular_dist(E, &rng, p, hit_nuc);
                        double alpha=((A-1.0)/(A+1.0))*((A-1.0)/(A+1.0));
                        E=E*(1.0+alpha+(1.0-alpha)*mu_cm)/2.0;
                        if(E<1e-11) E=1e-11;
                        // CPU-matching mu_lab: hydrogen (A<=1+eps) uses special case
                        double mu_lab = (A > 1.0 + 1e-10)
                            ? (1.0+A*mu_cm)/sqrt(1.0+A*A+2.0*A*mu_cm)
                            : sqrt(fmax(0.0, (1.0+mu_cm)*0.5));
                        double phi=2.0*PI*pcg_uniform(&rng);
                        rotate_direction(&dx,&dy,&dz,mu_lab,phi);
                    }
                }
            } else if ((cum_rxn+=nuc_inel[hit_l]), xi_rxn < cum_rxn) {
                trace[row+9] = 1.0; // inelastic
                E = fmax(E * 0.5, 1e-5); // simplified for trace
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=nuc_n2n[hit_l]), xi_rxn < cum_rxn) {
                trace[row+9] = 2.0;
                E = fmax(E * 0.3, 1e-5);
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=nuc_n3n[hit_l]), xi_rxn < cum_rxn) {
                trace[row+9] = 3.0;
                E = fmax(E * 0.2, 1e-5);
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=nuc_fis[hit_l]), xi_rxn < cum_rxn) {
                trace[row+9] = 4.0; // fission
                is_alive = 0;
            } else {
                trace[row+9] = 5.0; // capture
                is_alive = 0;
            }
            trace[row+15] = E;
        }
        actual_steps++;
    }

    pos_x[tid]=px; pos_y[tid]=py; pos_z[tid]=pz;
    dir_x[tid]=dx; dir_y[tid]=dy; dir_z[tid]=dz;
    energy[tid]=E; cell_idx[tid]=cell; alive[tid]=is_alive;
    rng_state_arr[tid]=rng.state; rng_inc_arr[tid]=rng.inc;
    step_counts[tid] = actual_steps;
}

// ═══════════════════════════════════════════════════════════════════════
// Windowed Multipole (WMP) evaluator for hybrid SVD+WMP mode.
// Mirrors src/wmp.rs (Humlicek W4 Faddeeva + broadened curvefit + pole sum).
// ═══════════════════════════════════════════════════════════════════════

#define WMP_K_BOLTZMANN 8.6173285e-5
#define WMP_INV_SQRT_PI 0.5641895835477563

__device__ __forceinline__ double2 c2mul(double2 a, double2 b) {
    return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ double2 c2add(double2 a, double2 b) { return make_double2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ double2 c2sub(double2 a, double2 b) { return make_double2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ double2 c2scale(double2 a, double s) { return make_double2(a.x*s, a.y*s); }
__device__ __forceinline__ double2 c2div(double2 a, double2 b) {
    double d = b.x*b.x + b.y*b.y;
    return make_double2((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d);
}

__device__ double2 wmp_horner(double2 z, const double* c, int n) {
    double2 acc = make_double2(c[n-1], 0.0);
    for (int i = n-2; i >= 0; --i) {
        acc = c2mul(acc, z);
        acc.x += c[i];
    }
    return acc;
}

__device__ double2 wmp_faddeeva(double2 z) {
    // Iterative form — avoid the recursive conjugate fold-up. CUDA supports
    // recursion but the extra stack frame blows local memory when this is
    // called from transport_persistent under __launch_bounds__(256, 2),
    // which has high register pressure. Instead: flip to upper half plane,
    // compute there, and un-flip at the end.
    bool conj = (z.y < 0.0);
    if (conj) z.y = -z.y;

    double x = z.x, y = z.y;
    double s = fabs(x) + y;
    double2 t = make_double2(y, -x);
    double2 result;

    if (s >= 15.0) {
        double2 u = c2mul(t, t);
        double2 num = c2scale(t, WMP_INV_SQRT_PI);
        double2 den = make_double2(u.x + 0.5, u.y);
        result = c2div(num, den);
    } else if (s >= 5.5) {
        double2 u = c2mul(t, t);
        double2 uu = c2mul(u, u);
        double2 num = c2mul(t, make_double2(1.410474 + u.x*WMP_INV_SQRT_PI, u.y*WMP_INV_SQRT_PI));
        double2 den = make_double2(0.75 + 3.0*u.x + uu.x, 3.0*u.y + uu.y);
        result = c2div(num, den);
    } else if (y >= 0.195 * fabs(x) - 0.176) {
        const double p_c[5] = {16.4955, 20.20933, 11.96482, 3.778987, 0.5642236};
        const double q_c[6] = {16.4955, 38.82363, 39.27121, 21.69274, 6.699398, 1.0};
        double2 num = wmp_horner(t, p_c, 5);
        double2 den = wmp_horner(t, q_c, 6);
        result = c2div(num, den);
    } else {
        // Region IV
        double2 u = c2mul(t, t);
        const double p_c[7] = {36183.31, -3321.9905, 1540.787, -219.0313, 35.76683, -1.320522, 0.56419};
        const double q_c[8] = {32066.6, -24322.84, 9022.228, -2186.181, 364.2191, -61.57037, 1.841439, -1.0};
        double2 p = wmp_horner(u, p_c, 7);
        double2 q = wmp_horner(u, q_c, 8);
        double e_abs = exp(u.x);
        double ss, cc;
        sincos(u.y, &ss, &cc);
        double2 exp_u = make_double2(e_abs*cc, e_abs*ss);
        double2 corr = c2mul(t, c2div(p, q));
        result = c2sub(exp_u, corr);
    }

    if (conj) {
        // OpenMC convention: for Im(z) < 0, w(z) = -conj(w(z*)) where z* = conj(z)
        result.x = -result.x;
        // result.y stays positive (antisymmetric under conjugation)
    }
    return result;
}

__device__ double wmp_erf(double x) {
    double sgn = (x < 0.0) ? -1.0 : 1.0;
    double ax = fabs(x);
    double tt = 1.0 / (1.0 + 0.3275911 * ax);
    double y = 1.0 - (((((1.061405429*tt - 1.453152027)*tt + 1.421413741)*tt
                        - 0.284496736)*tt + 0.254829592)*tt) * exp(-ax*ax);
    return sgn * y;
}

__device__ void wmp_broaden_poly(double e, double dopp, int n, double* out) {
    double sqrt_e = sqrt(e);
    double beta = sqrt_e * dopp;
    double half_inv_d2 = 0.5 / (dopp*dopp);
    double quarter_inv_d4 = half_inv_d2 * half_inv_d2;
    double erf_b, exp_mb2;
    if (beta > 6.0) { erf_b = 1.0; exp_mb2 = 0.0; }
    else { erf_b = wmp_erf(beta); exp_mb2 = exp(-beta*beta); }
    out[0] = erf_b / e;
    if (n > 1) out[1] = 1.0 / sqrt_e;
    if (n > 2) out[2] = out[0] * (half_inv_d2 + e) + exp_mb2 / (beta * sqrt(PI));
    for (int i = 1; i < n - 2; ++i) {
        double di = (double)i;
        if (i != 1) {
            out[i+2] = -out[i-2] * (di - 1.0) * di * quarter_inv_d4
                     + out[i] * (e + (1.0 + 2.0*di) * half_inv_d2);
        } else {
            out[i+2] = out[i] * (e + (1.0 + 2.0*di) * half_inv_d2);
        }
    }
}

__device__ void wmp_eval(
    double e, double t_kelvin,
    double e_min, double e_max, double spacing, double sqrt_awr,
    int n_windows, int fit_order, int fissionable,
    const double2* poles,
    const int* windows,
    const signed char* broaden_poly,
    const double* curvefit,
    double* out_s, double* out_a, double* out_f)
{
    *out_s = 0.0; *out_a = 0.0; *out_f = 0.0;
    if (e < e_min || e > e_max) return;

    double sqrt_kt = sqrt(WMP_K_BOLTZMANN * t_kelvin);
    double sqrt_e = sqrt(e);
    double inv_e = 1.0 / e;
    double sqrt_e_min = sqrt(e_min);
    int iw = (int)floor((sqrt_e - sqrt_e_min) / spacing);
    if (iw < 0) iw = 0;
    if (iw > n_windows - 1) iw = n_windows - 1;

    int startw = windows[2*iw];
    int endw   = windows[2*iw + 1];

    int order1 = fit_order + 1;
    const double* cf_base = curvefit + (size_t)iw * order1 * 3;

    if (sqrt_kt != 0.0 && broaden_poly[iw]) {
        double dopp = sqrt_awr / sqrt_kt;
        double factors[8];
        wmp_broaden_poly(e, dopp, order1, factors);
        for (int ip = 0; ip < order1; ++ip) {
            *out_s += cf_base[ip*3 + 0] * factors[ip];
            *out_a += cf_base[ip*3 + 1] * factors[ip];
            if (fissionable) *out_f += cf_base[ip*3 + 2] * factors[ip];
        }
    } else {
        double temp = inv_e;
        for (int ip = 0; ip < order1; ++ip) {
            *out_s += cf_base[ip*3 + 0] * temp;
            *out_a += cf_base[ip*3 + 1] * temp;
            if (fissionable) *out_f += cf_base[ip*3 + 2] * temp;
            temp *= sqrt_e;
        }
    }

    if (startw >= 0 && endw > startw) {
        double sqrt_pi_v = sqrt(PI);
        double dopp = sqrt_awr / sqrt_kt;
        for (int ip = startw; ip < endw; ++ip) {
            double2 ea = poles[ip*4 + 0];
            double2 rs = poles[ip*4 + 1];
            double2 ra = poles[ip*4 + 2];
            double2 rf = poles[ip*4 + 3];
            double2 zc = make_double2((sqrt_e - ea.x)*dopp, -ea.y*dopp);
            double2 w = wmp_faddeeva(zc);
            double scale = dopp * inv_e * sqrt_pi_v;
            double2 wv = c2scale(w, scale);
            *out_s += c2mul(rs, wv).x;
            *out_a += c2mul(ra, wv).x;
            if (fissionable) *out_f += c2mul(rf, wv).x;
        }
    }
}

extern "C" __global__ void wmp_test_eval(
    int n_e,
    const double* energies,
    double t_kelvin,
    double e_min, double e_max, double spacing, double sqrt_awr,
    int n_windows, int fit_order, int fissionable,
    const double2* poles,
    const int* windows,
    const signed char* broaden_poly,
    const double* curvefit,
    double* out_s, double* out_a, double* out_f)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_e) return;
    double s, a, f;
    wmp_eval(energies[tid], t_kelvin,
             e_min, e_max, spacing, sqrt_awr,
             n_windows, fit_order, fissionable,
             poles, windows, broaden_poly, curvefit,
             &s, &a, &f);
    out_s[tid] = s;
    out_a[tid] = a;
    out_f[tid] = f;
}
