// ═══════════════════════════════════════════════════════════════════════
// open_rust_mc — Event-based GPU neutron transport kernel
//
// Single packed parameter struct for all read-only data.
// Persistent kernel with warp-level reductions and energy-sorted
// compaction. Supports PWR pin cell and Godiva geometries.
// ═══════════════════════════════════════════════════════════════════════

#define COINCIDENCE_TOL 1e-12
#define PI 3.14159265358979323846
#define N_REACTIONS 6
#define RXN_ELASTIC 0
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
#define P_URR_ENERGIES   55
#define P_URR_CUM_PROB   56
#define P_URR_TOTAL_F    57
#define P_URR_ELASTIC_F  58
#define P_URR_FISSION_F  59
#define P_URR_CAPTURE_F  60
#define P_URR_OFFSETS    61
#define P_URR_N_ENERGIES 62
#define P_URR_N_BANDS    63
#define P_URR_MULT_SM    64
#define P_GEOM_TYPE      65
#define N_PARAMS          66

// Access helpers — read from the flat u64 params buffer
#define PTR_F(p, idx)   ((const float*)  (p)[(idx)])
#define PTR_D(p, idx)   ((const double*) (p)[(idx)])
#define PTR_I(p, idx)   ((const int*)    (p)[(idx)])
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
        double nx=px+dx*(best_t+1e-10), ny=py+dy*(best_t+1e-10), nz=pz+dz*(best_t+1e-10);
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
    while (hi-lo > 1) { int mid=(lo+hi)/2; if (grid[mid]<=energy) lo=mid; else hi=mid; }
    return lo;
}

__device__ double svd_reconstruct(
    const float* __restrict__ basis,
    const double* __restrict__ coeffs,
    int e_idx, int rank)
{
    const float* row = &basis[e_idx * rank];
    double acc = 0.0;
    for (int j = 0; j < rank; j++)
        acc = fma((double)__ldg(&row[j]), __ldg(&coeffs[j]), acc);
    return exp2(acc * 3.321928094887362);
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
    while (hi-lo>1) { int mid=(lo+hi)/2; if (e[mid]<=E) lo=mid; else hi=mid; }
    double f = (E-e[lo]) / (e[hi]-e[lo]);
    return v[lo] + f*(v[hi]-v[lo]);
}

__device__ double sample_fission_energy(
    double E_inc, PcgState* rng, Params p, int hit_nuc)
{
    int fi_off = __ldg(&PTR_I(p, P_FIS_NUC_OFF)[hit_nuc]);
    int fi_n = __ldg(&PTR_I(p, P_FIS_NUC_NINC)[hit_nuc]);
    if (fi_n <= 0) {
        double a=0.988, x1=-log(fmax(pcg_uniform(rng),1e-30));
        double x2=-log(fmax(pcg_uniform(rng),1e-30));
        double c=cos(PI/2.0*pcg_uniform(rng));
        return a*(x1+x2*c*c)*1e6;
    }
    const double* inc_e = &PTR_D(p, P_FIS_INC_E)[fi_off];
    int ie = 0;
    for (int i=0; i<fi_n-1; i++) { if (E_inc >= inc_e[i] && E_inc < inc_e[i+1]) { ie=i; break; } }
    if (E_inc >= inc_e[fi_n-1]) ie = fi_n-1;
    int off = PTR_I(p, P_FIS_DIST_OFF)[fi_off+ie];
    int sz = PTR_I(p, P_FIS_DIST_SZ)[fi_off+ie];
    if (sz <= 1) return E_inc*0.5;
    double xi = pcg_uniform(rng);
    const double* eo = &PTR_D(p, P_FIS_E_OUT)[off];
    const double* cd = &PTR_D(p, P_FIS_CDF)[off];
    int lo=0, hi=sz-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (cd[mid]<=xi) lo=mid; else hi=mid; }
    double f = (xi-cd[lo]) / fmax(cd[hi]-cd[lo],1e-30);
    return eo[lo] + f*(eo[hi]-eo[lo]);
}

__device__ double sample_angular_dist(
    double E, PcgState* rng, Params p, int hit_nuc)
{
    int a_off = __ldg(&PTR_I(p, P_ANG_NUC_OFF)[hit_nuc]);
    int a_ne = __ldg(&PTR_I(p, P_ANG_NUC_NE)[hit_nuc]);
    if (a_ne <= 0) return 2.0*pcg_uniform(rng)-1.0;
    const double* ae = &PTR_D(p, P_ANG_ENERGIES)[a_off];
    int ie=0;
    if (E <= ae[0]) ie=0;
    else if (E >= ae[a_ne-1]) ie=a_ne-1;
    else { int lo=0,hi=a_ne-1; while(hi-lo>1){int mid=(lo+hi)/2;if(ae[mid]<=E)lo=mid;else hi=mid;} ie=lo; }
    int off = PTR_I(p, P_ANG_DIST_OFF)[a_off+ie];
    int sz = PTR_I(p, P_ANG_DIST_SZ)[a_off+ie];
    if (sz <= 1) return 2.0*pcg_uniform(rng)-1.0;
    double xi = pcg_uniform(rng);
    const double* mu = &PTR_D(p, P_ANG_MU)[off];
    const double* cd = &PTR_D(p, P_ANG_CDF)[off];
    int lo=0, hi=sz-1;
    while(hi-lo>1){int mid=(lo+hi)/2;if(cd[mid]<=xi)lo=mid;else hi=mid;}
    double f = (xi-cd[lo])/fmax(cd[hi]-cd[lo],1e-30);
    return fmax(-1.0, fmin(1.0, mu[lo]+f*(mu[hi]-mu[lo])));
}

__device__ double sab_total_xs(double E, Params p) {
    if (SCALAR_I(p, P_SAB_N_INC) <= 0) return 0.0;
    const double* e = PTR_D(p, P_SAB_INC_E);
    const double* xs = PTR_D(p, P_SAB_XS);
    int n = SCALAR_I(p, P_SAB_N_INC);
    if (E <= e[0]) return xs[0];
    if (E >= e[n-1]) return xs[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(lo+hi)/2;if(e[mid]<=E)lo=mid;else hi=mid;}
    double f=(E-e[lo])/fmax(e[hi]-e[lo],1e-30);
    return xs[lo]+f*(xs[hi]-xs[lo]);
}

__device__ void sab_sample(
    double E_in, PcgState* rng, Params p,
    double* E_out, double* mu_out)
{
    int n = SCALAR_I(p, P_SAB_N_INC);
    if (n <= 0) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }
    int ie=0;
    if (E_in<=PTR_D(p, P_SAB_INC_E)[0]) ie=0;
    else if (E_in>=PTR_D(p, P_SAB_INC_E)[n-1]) ie=n-1;
    else {
        int lo=0,hi=n-1;
        while(hi-lo>1){int mid=(lo+hi)/2;if(PTR_D(p, P_SAB_INC_E)[mid]<=E_in)lo=mid;else hi=mid;}
        double f=(E_in-PTR_D(p, P_SAB_INC_E)[lo])/fmax(PTR_D(p, P_SAB_INC_E)[hi]-PTR_D(p, P_SAB_INC_E)[lo],1e-30);
        ie = (pcg_uniform(rng)<f) ? hi : lo;
    }
    int eo_off=PTR_I(p, P_SAB_EOUT_OFF)[ie], eo_sz=PTR_I(p, P_SAB_EOUT_SZ)[ie];
    if (eo_sz<=1) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }
    double xi_e=pcg_uniform(rng);
    const double* eo=&PTR_D(p, P_SAB_E_OUT)[eo_off];
    const double* cdf_e=&PTR_D(p, P_SAB_CDF_E)[eo_off];
    int lo=0,hi=eo_sz-1;
    while(hi-lo>1){int mid=(lo+hi)/2;if(cdf_e[mid]<=xi_e)lo=mid;else hi=mid;}
    double f_e=(xi_e-cdf_e[lo])/fmax(cdf_e[hi]-cdf_e[lo],1e-30);
    *E_out=fmax(eo[lo]+f_e*(eo[hi]-eo[lo]),1e-11);
    int eout_bin=lo;
    int mu_key=eo_off+eout_bin;
    int mu_off=PTR_I(p, P_SAB_MU_OFF)[mu_key], mu_sz=PTR_I(p, P_SAB_MU_SZ)[mu_key];
    if (mu_sz<=1) { *mu_out=2.0*pcg_uniform(rng)-1.0; return; }
    double xi_mu=pcg_uniform(rng);
    const double* mu_arr=&PTR_D(p, P_SAB_MU)[mu_off];
    const double* cdf_mu=&PTR_D(p, P_SAB_CDF_MU)[mu_off];
    lo=0; hi=mu_sz-1;
    while(hi-lo>1){int mid=(lo+hi)/2;if(cdf_mu[mid]<=xi_mu)lo=mid;else hi=mid;}
    double f_mu=(xi_mu-cdf_mu[lo])/fmax(cdf_mu[hi]-cdf_mu[lo],1e-30);
    *mu_out=fmax(-1.0,fmin(1.0,mu_arr[lo]+f_mu*(mu_arr[hi]-mu_arr[lo])));
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
        while(hi-lo>1){int mid=(lo+hi)/2;if(ue[mid]<=E)lo=mid;else hi=mid;} ie=lo; }
    // Sample band
    int base = off*n_b + ie*n_b;
    const double* cp = &PTR_D(p, P_URR_CUM_PROB)[base];
    int band=0;
    for (int b=0; b<n_b; b++) { if (xi < cp[b]) { band=b; break; } band=b; }
    double ft=PTR_D(p, P_URR_TOTAL_F)[base+band];
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
    if (w2 < 0.999) {
        double inv_sq = 1.0/sqrt(1.0-w2);
        double dx2=mu*(*dx)+sin_mu*((*dx)*(*dz)*cos(phi)-(*dy)*sin(phi))*inv_sq;
        double dy2=mu*(*dy)+sin_mu*((*dy)*(*dz)*cos(phi)+(*dx)*sin(phi))*inv_sq;
        double dz2=mu*(*dz)-sin_mu*sqrt(1.0-w2)*cos(phi);
        *dx=dx2; *dy=dy2; *dz=dz2;
    } else {
        double sign = (*dz > 0.0) ? 1.0 : -1.0;
        *dx=sin_mu*cos(phi); *dy=sin_mu*sin(phi)*sign; *dz=mu*sign;
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
    ddx[tid]=st*cos(phi); ddy[tid]=st*sin(phi); ddz[tid]=mu;
    rng_state[tid]=rng.state; rng_inc[tid]=rng.inc;
}

// ═══════════════════════════════════════════════════════════════════════
// PERSISTENT TRANSPORT KERNEL
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void __launch_bounds__(256, 2)
transport_persistent(
    Params p,
    const int* __restrict__ compact_idx, int n_alive,
    // Mutable particle state (SoA)
    double* pos_x, double* pos_y, double* pos_z,
    double* dir_x, double* dir_y, double* dir_z,
    double* energy, int* cell_idx, int* alive,
    unsigned long long* rng_state_arr, unsigned long long* rng_inc_arr,
    // Fission bank
    double* fis_x, double* fis_y, double* fis_z,
    double* fis_e, double* fis_w,
    int* fis_count, int max_fis,
    // Counters
    int* cnt_coll, int* cnt_fis, int* cnt_leak, int* cnt_surf,
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

        // XS lookup
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t=0, sum_el=0, sum_fis=0, sum_cap=0;
        double nuc_t[4]={}, nuc_el[4]={}, nuc_fis[4]={}, nuc_cap[4]={};
        double urr_xi = pcg_uniform(&rng);

        for (int i=0; i<n_nuc; i++) {
            int ni = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat*4+i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat*4+i]);
            int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[ni]);
            int n_e = __ldg(&PTR_I(p, P_N_ENERGIES)[ni]);
            int e_idx = energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off], n_e, E);

            double s_el=0, s_fis=0, s_cap=0, s_rest=0;
            for (int r=0; r<N_REACTIONS; r++) {
                int key = ni*N_REACTIONS+r;
                if (__ldg(&PTR_I(p, P_HAS_REACTION)[key])) {
                    double s = svd_reconstruct(
                        &PTR_F(p, P_BASIS)[__ldg(&PTR_I(p, P_BASIS_OFFSETS)[key])],
                        &PTR_D(p, P_COEFFS)[__ldg(&PTR_I(p, P_COEFFS_OFFSETS)[key])], e_idx, rank);
                    if(r==RXN_ELASTIC) s_el=s;
                    else if(r==RXN_FISSION) s_fis=s;
                    else if(r==RXN_CAPTURE) s_cap=s;
                    else s_rest+=s;
                }
            }

            // URR
            apply_urr(p, ni, &s_el, &s_fis, &s_cap, E, urr_xi);

            // S(alpha,beta) for H1 (nuclide idx 3 in PWR)
            double sab_xs_val = 0.0;
            if (ni==3 && E < SCALAR_D(p, P_SAB_EMAX) && E > 0.0 && SCALAR_I(p, P_SAB_N_INC) > 0) {
                sab_xs_val = sab_total_xs(E, p);
                if (sab_xs_val > 0.0) s_el = sab_xs_val;
            }

            double micro_t = s_el + s_fis + s_cap + s_rest;
            nuc_t[i]=Ni*micro_t; nuc_el[i]=Ni*s_el; nuc_fis[i]=Ni*s_fis; nuc_cap[i]=Ni*s_cap;
            sum_t+=Ni*micro_t; sum_el+=Ni*s_el; sum_fis+=Ni*s_fis; sum_cap+=Ni*s_cap;
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

            // Sample reaction
            double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];

            if (xi_rxn < nuc_el[hit_l]) {
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

                if (E < 400.0*cell_kT && A < 10.0) {
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
                    double mu_cm=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
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
                    double sx2=mu_cm*vrh_x+st*(cos(phi)*px2+sin(phi)*qx);
                    double sy2=mu_cm*vrh_y+st*(cos(phi)*py2+sin(phi)*qy);
                    double sz2=mu_cm*vrh_z+st*(cos(phi)*pz2+sin(phi)*qz);
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
                    double mu_lab=(1.0+A*mu_cm)/sqrt(1.0+A*A+2.0*A*mu_cm);
                    double phi=2.0*PI*pcg_uniform(&rng);
                    rotate_direction(&dx,&dy,&dz,mu_lab,phi);
                }

            } else if (xi_rxn < nuc_el[hit_l]+nuc_fis[hit_l]) {
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

            } else if (xi_rxn < nuc_el[hit_l]+nuc_fis[hit_l]+nuc_cap[hit_l]) {
                // ═══ Capture ═══
                is_alive=0;

            } else {
                // ═══ Inelastic — proper discrete level sampling ═══
                int lv_off=__ldg(&PTR_I(p, P_LEVEL_OFFSETS)[hit_nuc]);
                int n_lev=__ldg(&PTR_I(p, P_LEVEL_COUNTS)[hit_nuc]);
                double Q=-0.5e6; int selected=0;
                if (n_lev>0) {
                    double lxs_sum=0.0, lxs_cum[64]; int na=0;
                    int g_off=__ldg(&PTR_I(p, P_GRID_OFFSETS)[hit_nuc]);
                    int n_e=__ldg(&PTR_I(p, P_N_ENERGIES)[hit_nuc]);
                    int e_idx=energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off],n_e,E);
                    for(int l=0;l<n_lev&&l<64;l++){
                        int gl=lv_off+l; double lxs=0.0;
                        if(E>=__ldg(&PTR_D(p, P_LEVEL_THR)[gl])&&__ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])){
                            lxs=svd_reconstruct(
                                &PTR_F(p, P_LEVEL_BASIS)[__ldg(&PTR_I(p, P_LEVEL_BOFF)[gl])],
                                &PTR_D(p, P_LEVEL_COEFFS)[__ldg(&PTR_I(p, P_LEVEL_COFF)[gl])],e_idx,rank);
                        }
                        lxs_sum+=lxs; lxs_cum[l]=lxs_sum; na++;
                    }
                    if(lxs_sum>0.0){
                        double xi_l=pcg_uniform(&rng)*lxs_sum;
                        for(int l=0;l<na;l++){if(xi_l<lxs_cum[l]){selected=l;break;}selected=l;}
                        Q=__ldg(&PTR_D(p, P_LEVEL_Q)[lv_off+selected]);
                    }
                }
                int sel_mt=(n_lev>0)?__ldg(&PTR_I(p, P_LEVEL_MT)[lv_off+selected]):0;
                double E_out;
                if(sel_mt==91){
                    double a_p=A/8.0, ecm=E*A/((A+1.0)*1e6), eex=fmax(ecm,0.1);
                    double T=sqrt(eex/a_p);
                    double x1=fmax(pcg_uniform(&rng),1e-30), x2=fmax(pcg_uniform(&rng),1e-30);
                    double eo=-T*log(x1*x2); eo=fmin(eo,ecm);
                    E_out=fmax(eo*1e6,1e-5);
                } else {
                    double ar=A/(A+1.0);
                    E_out=E*ar*ar+Q*(A+1.0)/A;
                    if(E_out<=0.0) E_out=E*0.01;
                }
                E=fmax(E_out,1e-11);
                double mu_cm=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                double st=sqrt(fmax(0.0,1.0-mu_cm*mu_cm));
                dx=st*cos(phi); dy=st*sin(phi); dz=mu_cm;
            }
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
