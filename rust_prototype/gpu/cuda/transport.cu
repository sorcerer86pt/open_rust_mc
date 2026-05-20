// SPDX-License-Identifier: MIT
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
// Closed-form Watt (Law 11) χ — see GpuNuclideData::watt_*.
// `P_WATT_NUC_N[i] > 0` flags nuclide `i` as Watt-parameterised; the
// device looks up a(E_in), b(E_in) by lin-lin interpolation on the
// shared incident-energy grid `P_WATT_INC_E[off..off+n]`.
#define P_WATT_INC_E        104
#define P_WATT_A            105
#define P_WATT_B            106
#define P_WATT_U            107
#define P_WATT_NUC_OFF      108
#define P_WATT_NUC_N        109
// Delayed-only ν̄(E) per nuclide — mirrors P_NB_* (which carries
// ν_total = ν_prompt + Σ ν_delayed). β(E) = ν_delayed / ν_total is
// computed at the fission emission site; with probability β each
// banked neutron is drawn from the soft-Watt delayed spectrum
// (sample_delayed_energy) instead of the prompt χ. Nuclides without
// delayed-product data leave `P_DNB_SIZES[i] = 0`, in which case the
// kernel falls through to the prompt-only path.
#define P_DNB_ENERGIES      110
#define P_DNB_VALUES        111
#define P_DNB_OFFSETS       112
#define P_DNB_SIZES         113
// PDF samples aligned 1:1 with P_FIS_E_OUT / P_FIS_CDF. Enables the
// quadratic lin-lin CDF inversion in `sample_eout_bin`, matching the
// OpenMC `Tabular::sample` algorithm used on CPU. Pre-fix the GPU's
// linear-CDF fallback was biasing χ outgoing spectra hard → less
// leakage → +500-700 pcm hot on Godiva / PMF metal benchmarks.
#define P_FIS_PDF           114

// ── MT=91 continuum inelastic outgoing-energy distribution ───────
//
// Mirror of P_FIS_* but for the ENDF MT=91 (continuum inelastic)
// secondary-energy law. Added to close a +400 keV ⟨E_out⟩ gap vs the
// CPU: the kernel previously used a Weisskopf evaporation
// approximation (`E_out = -T·log(ξ1·ξ2)` with `T = √(E_exc / (A/8))`)
// for every nuclide, but the CPU samples directly from the tabulated
// MT=91 distribution when one is present (see
// `physics/collision.rs::sample_inelastic_level`). For Godiva the
// evaporation approximation gave ⟨E_out inelastic⟩ ≈ 1.25 MeV on GPU
// vs 0.85 MeV on CPU — a 48 % hardening that flowed straight through
// to a +500–700 pcm `k_eff` bias on fast-metal benchmarks.
//
// When `P_INEL91_NUC_NINC[nuc] == 0` the kernel falls back to the
// evaporation formula (light isotopes / older evaluations that ship
// no MT=91 distribution).
#define P_INEL91_INC_E      115
#define P_INEL91_DIST_OFF   116
#define P_INEL91_DIST_SZ    117
#define P_INEL91_E_OUT      118
#define P_INEL91_CDF        119
#define P_INEL91_PDF        120
#define P_INEL91_NUC_OFF    121
#define P_INEL91_NUC_NINC   122

// ── Multi-slot S(α,β) lookup tables. Slots 43–55 are the flat data
// arrays (single concatenated stream across all nuclides that carry a
// TSL); the slots below index that stream per-slot, and
// `P_SAB_SLOT_PER_NUC[nuc_idx]` returns either the slot index or -1.
// Callers must dispatch on the per-nuclide lookup rather than a
// scalar `sab_nuc_idx` for any new code path that wants more than one
// SAB-bearing nuclide in the same run.
#define P_SAB_N_SLOTS           123
#define P_SAB_SLOT_PER_NUC      124
#define P_SAB_SLOT_INC_E_OFF    125
#define P_SAB_SLOT_N_INC        126
#define P_SAB_SLOT_EOUT_TABLE_OFF 127
#define P_SAB_SLOT_MU_TABLE_OFF 128
#define P_SAB_SLOT_EMAX         129

// ── Maxwell (ENDF Law 7) / Evaporation (ENDF Law 9) closed-form χ.
// Per-nuclide θ(E_in) table — single 1D, shared by both laws.
// `P_MAXEVAP_LAW[i]` is 7 for Maxwell, 9 for Evaporation, 0 for neither.
// Dispatched from sample_fission_energy when tabular χ (P_FIS_*) is
// absent AND no Watt parameters (P_WATT_*) were uploaded — replaces
// the prior fall-through to the U-235 Cranberg Watt parameters that
// was biasing every U-233 (Maxwell), U-234 (Maxwell), and several
// Pu-240 / Pu-241 (Evaporation) GPU benchmark on fast-spectrum
// scenes by ~100-400 pcm.
#define P_MAXEVAP_INC_E         130
#define P_MAXEVAP_THETA         131
#define P_MAXEVAP_U             132
#define P_MAXEVAP_LAW           133
#define P_MAXEVAP_NUC_OFF       134
#define P_MAXEVAP_NUC_N         135

// Stage C step D — per-nuclide pointer arrays. Each slot stores a
// flat `u64` table sized `[n_nuc × N_REACTIONS]` (P_BASIS_PTRS /
// P_COEFFS_PTRS) or `[n_nuc]` (the rest) containing the
// `CUdeviceptr` of the corresponding per-nuclide CudaSlice.
// Accessed via `(const T*) PTR_U64(p, P_*)[key]`. Absent slots
// store `0`; the kernel gates on has_* sentinels so it never loads
// through a null pointer.
#define P_BASIS_PTRS            136
#define P_COEFFS_PTRS           137
#define P_PW_XS_PTRS            138
#define P_TOTAL_XS_PTRS         139
#define P_NB_E_PTRS             140
#define P_NB_V_PTRS             141
#define P_DNB_E_PTRS            142
#define P_DNB_V_PTRS            143
#define P_URR_E_PTRS            144
#define P_URR_CP_PTRS           145
#define P_URR_TF_PTRS           146
#define P_URR_EF_PTRS           147
#define P_URR_FF_PTRS           148
#define P_URR_CF_PTRS           149
#define P_INEL_CDF_PTRS         150
// Per-nuc base pointers into the per-nuclide LevelSlicesGpu.basis /
// .coeffs CudaSlices. `[n_nuc]` u64 entries; the kernel uses
// `hit_nuc` (already in scope at every discrete-level access site)
// to pick the right per-nuc base, then indexes by the per-level
// within-nuc offset arrays below.
#define P_LEVEL_BASIS_PTRS      151
#define P_LEVEL_COEFFS_PTRS     152
// Per-(global level) within-nuc byte offsets into the per-nuc
// basis / coeffs buffers. `[total_levels]` i32 entries — same
// indexing as the legacy `P_LEVEL_BOFF` / `P_LEVEL_COFF` slabs,
// but un-shifted (no global running-offset added). Preserves the
// `1654c4d` rank-padding invariant: every level's basis is padded
// to `[n_e × global_rank]` per LevelSlicesGpu::build_level_slices.
#define P_LEVEL_BLOCAL_OFF      153
#define P_LEVEL_CLOCAL_OFF      154
// Per-nuc base pointers for elastic + per-level angular CDFs.
// Slots 155-160 are u64 [n_nuc]; slots 161-162 are i32
// arrays of within-nuc offsets (sized [total_ang_e] /
// [total_lev_ang_dist] respectively).
#define P_ANG_E_PTRS            155
#define P_ANG_MU_PTRS           156
#define P_ANG_CDF_PTRS          157
#define P_LEV_ANG_E_PTRS        158
#define P_LEV_ANG_MU_PTRS       159
#define P_LEV_ANG_CDF_PTRS      160
#define P_ANG_DIST_LOCAL_OFF    161
#define P_LEV_ANG_LEV_LOCAL_OFF 162
#define P_LEV_ANG_DIST_LOCAL_OFF 163
// Fission tabular + MT=91 per-nuc base pointers and un-shifted
// per-(global inc_e) offset arrays. 8 ptr slots + 2 int arrays.
#define P_FIS_INC_E_PTRS        164
#define P_FIS_E_OUT_PTRS        165
#define P_FIS_CDF_PTRS          166
#define P_FIS_PDF_PTRS          167
#define P_FIS_DIST_LOCAL_OFF    168
#define P_INEL91_INC_E_PTRS     169
#define P_INEL91_E_OUT_PTRS     170
#define P_INEL91_CDF_PTRS       171
#define P_INEL91_PDF_PTRS       172
#define P_INEL91_DIST_LOCAL_OFF 173

// ── SAB elastic channel — slots 174-180 ─────────────────────────────
//
// Adds the coherent / incoherent elastic S(α,β) data the GPU was
// missing. `P_SAB_SLOT_ELASTIC_MODE[slot]` selects per-slot variant:
//   0 = none, 1 = Coherent (Bragg), 2 = Incoherent (Debye-Waller).
// Coherent uses the flat `P_SAB_COH_BRAGG_EDGES` / `P_SAB_COH_FACTORS`
// arrays plus per-slot offset / count (`P_SAB_SLOT_COH_OFF` /
// `P_SAB_SLOT_COH_N`). Incoherent reads per-slot scalars from
// `P_SAB_SLOT_INC_BOUND_XS` / `P_SAB_SLOT_INC_DEBYE_WALLER`.
//
// Restores the reflector-return mechanism in thick-Be / -graphite /
// -polyethylene geometries (HEU-MET-FAST-058 case-1 was -2500 pcm
// cold against handbook before this).
#define P_SAB_SLOT_ELASTIC_MODE     174
#define P_SAB_COH_BRAGG_EDGES       175
#define P_SAB_COH_FACTORS           176
#define P_SAB_SLOT_COH_OFF          177
#define P_SAB_SLOT_COH_N            178
#define P_SAB_SLOT_INC_BOUND_XS     179
#define P_SAB_SLOT_INC_DEBYE_WALLER 180

// ── Angular-distribution PDF per-nuc base pointers — slots 181-182 ──
//
// Mirrors P_ANG_MU_PTRS / P_LEV_ANG_MU_PTRS layout exactly: `[n_nuc]`
// u64 entries, each one the device address of the corresponding
// per-nuclide `AngularSlicesGpu::pdf` / `LevelSlicesGpu::ang_pdf`
// buffer. Pairs with the existing P_ANG_DIST_LOCAL_OFF / P_LEV_ANG_
// DIST_LOCAL_OFF tables — same offsets are used for mu / cdf / pdf
// since the three arrays are length-matched on upload.
//
// Drives the quadratic lin-lin CDF inversion in `sample_mu_bin` for
// ENDF/B-VII.1 angular distributions that aren't histogram (every
// nuclide with `histogram=false` — most of them). Without these the
// kernel falls back to a linear-CDF approximation that biased
// forward-peaked scatter for Al-27, Mg-24..26, Mn-55, Cr-50..54, and
// W-180..186, surfacing as the +500-700 pcm CPU↔GPU gap on multi-
// nuclide fast-spectrum benchmarks (ieu-met-fast-001, heu-met-fast-
// 011) — analog of the fis_pdf / inel91_pdf fixes that closed the
// same gap on the χ outgoing spectrum.
#define P_ANG_PDF_PTRS              181
#define P_LEV_ANG_PDF_PTRS          182

// ── Stochastic temperature interpolation across SAB kT columns ─────
// `P_SAB_SLOT_COUNT_PER_NUC[nuc]` gives the count of consecutive
// slots reserved for that nuclide (= TSL's kT-grid length); the
// existing `P_SAB_SLOT_PER_NUC[nuc]` is now the FIRST slot. The
// kernel reads `P_SAB_SLOT_KT[slot]` for each slot's kT and selects
// stochastically between the two bracketing entries via cell_kT —
// mirrors CPU's `tsl.select_temperature(cell.T, ξ)` (4 call sites
// in simulate.rs). Closes the GPU's static-temperature mis-sampling
// for any cell whose temperature lies strictly between SAB columns.
#define P_SAB_SLOT_COUNT_PER_NUC    183
#define P_SAB_SLOT_KT               184

// ── URR interpolation code per nuclide (slot 185) ─────────────────
// `[n_nuc]`: 2 = lin-lin (default), 5 = log-log; `0` when no URR is
// bound for this nuclide. Drives the bin-to-bin interpolation in
// `apply_urr`. Without this, the kernel just used the lower-bin URR
// factor at E < grid[i_hi] — systematically biased the spectrum on
// multi-nuclide structural-reflector cases.
#define P_URR_INTERP                185

#define N_PARAMS            186

// ───────────────────────────────────────────────────────────────────────
// Per-material nuclide stride. Single source of truth is the Rust
// constant `MAX_NUCLIDES_PER_MATERIAL` (crate root in `lib.rs`); the
// NVRTC host injects it as `-DMAX_NUC_PER_MAT=N` on every compile
// (gpu_transport.rs and gpu_recursive.rs::assemble_kernel_source).
// Falling back to a literal here would let host / device disagree if
// the Rust constant ever changes without a clean rebuild.
#ifndef MAX_NUC_PER_MAT
#  error "MAX_NUC_PER_MAT not defined — NVRTC must pass -DMAX_NUC_PER_MAT=N from gpu_transport.rs / gpu_recursive.rs"
#endif

// Access helpers — read from the flat u64 params buffer
// PTR_F removed — all basis data is now f64 (PTR_D)
#define PTR_D(p, idx)   ((const double*) (p)[(idx)])
#define PTR_I(p, idx)   ((const int*)    (p)[(idx)])
#define PTR_B(p, idx)   ((const signed char*) (p)[(idx)])
#define PTR_D2(p, idx)  ((const double2*) (p)[(idx)])
#define PTR_U64(p, idx) ((const unsigned long long*) (p)[(idx)])
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
// Note: exp2f(float) was tested here (commit context) — gives 0% speedup because
// the MIO stall is dominated by BRX dynamic branching, not MUFU transcendentals.
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

// Quadratic lin-lin CDF inversion (OpenMC `Tabular::sample`).
// Mirrors `TabularEnergyDist::sample_with_xi` in `hdf5_reader.rs`.
//
// ─── Empirical note (this session) ────────────────────────────────
// The original GPU `sample_eout_bin` used a linear-CDF / histogram-
// PDF fallback. The hypothesis was that this biased fission χ hard
// on Godiva and was responsible for the +500-700 pcm GPU↔CPU gap on
// fast-metal benchmarks. We ported the proper quadratic formula
// (matching the CPU bit-for-bit, verified by `bin/chi_compare`:
// max per-sample Δ < 6 ppm of E_out for U-235 χ across all 20
// incident-energy bins, ⟨E⟩ agrees to 5 significant figures).
//
// Result: **NO statistically significant shift** in Godiva k_eff
// across 5-seed batches once 5-seed σ_mean ≈ 110 pcm is accounted
// for. The historical "+330 pcm" worsening looked real on one batch
// of 5 seeds but evaporated to <1.5σ over the next batch — pure
// noise from GPU atomicAdd ordering + small-N seed sampling.
//
// Therefore: the quadratic formula is now correct and parity-locked
// with the CPU, but the +500-700 pcm fast-metal CPU↔GPU gap lives
// elsewhere — angular kinematics CM→lab, URR equivalence, or
// initial-source seeding are the remaining candidates. χ sampling
// is ruled out by direct numerical comparison.
// ──────────────────────────────────────────────────────────────────
__device__ __forceinline__ double sample_eout_bin(
    double xi, const double* eo, const double* cd, const double* pd, int sz)
{
    if (sz <= 1) return eo[0];
    int lo = 0, hi = sz - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cd[mid] <= xi) lo = mid; else hi = mid;
    }
    const double e_lo = eo[lo];
    const double e_hi = eo[hi];
    const double cdf_lo = cd[lo];
    const double cdf_hi = cd[hi];
    const double de = e_hi - e_lo;
    if (fabs(cdf_hi - cdf_lo) < 1e-15) return e_lo;
    if (pd != nullptr && de > 0.0) {
        const double p_lo = pd[lo];
        const double p_hi = pd[hi];
        const double dc = xi - cdf_lo;
        if (p_lo > 0.0 || p_hi > 0.0) {
            const double m = (p_hi - p_lo) / de;
            if (fabs(m) < 1e-30) {
                if (p_lo > 0.0) return e_lo + dc / p_lo;
            } else {
                const double disc = p_lo * p_lo + 2.0 * m * dc;
                if (disc >= 0.0) return e_lo + (sqrt(disc) - p_lo) / m;
            }
        }
    }
    const double f = (xi - cdf_lo) / fmax(cdf_hi - cdf_lo, 1e-30);
    return e_lo + f * de;
}

/// Mathematically correct Watt sampler matching the analytic moments
/// 3a/2 + a²b/4. Derivation in the CPU `WattLaw::sample` (see
/// `hdf5_reader.rs`): W ~ Maxwell(a) via Coveyou–Macpherson
///   W = -a · (ln ξ₁ + cos²(π ξ₂ / 2) · ln ξ₃)
/// then the Watt step
///   E_out = W + a²b/4 + (2ξ₄ − 1) · √(a²b·W)
/// with a in eV and b in /eV. The prior GPU fallback dropped the
/// `a²b/4` shift and the jitter term entirely, giving a pure Maxwell
/// with mean 3a/2 — 27% LOW on the outgoing fission spectrum for
/// Cranberg U-235 parameters.
__device__ __forceinline__ double sample_watt_ab(
    double a_eV, double b_inv_eV, PcgState* rng)
{
    double xi1 = fmax(pcg_uniform(rng), 1e-30);
    double xi2 = pcg_uniform(rng);
    double xi3 = fmax(pcg_uniform(rng), 1e-30);
    double xi4 = pcg_uniform(rng);
    double c   = cos(PI * 0.5 * xi2);
    double W   = -a_eV * (log(xi1) + c * c * log(xi3));
    double term = 0.25 * a_eV * a_eV * b_inv_eV;
    double e_out = W + term + (2.0 * xi4 - 1.0) * sqrt(a_eV * a_eV * b_inv_eV * W);
    return fmax(e_out, 1e-5);
}

/// Lin-lin interpolation of θ(E_in) on the per-nuclide grid stored
/// at P_MAXEVAP_INC_E / P_MAXEVAP_THETA. Mirrors
/// `MaxwellLaw::theta_at` in `hdf5_reader.rs`.
__device__ __forceinline__ double maxevap_theta_at(
    double E_inc, const double* e_grid, const double* theta, int n)
{
    if (n <= 0) return 1.0;
    if (E_inc <= e_grid[0]) return theta[0];
    if (E_inc >= e_grid[n - 1]) return theta[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (e_grid[mid] <= E_inc) lo = mid; else hi = mid;
    }
    double frac = (E_inc - e_grid[lo]) / fmax(e_grid[lo + 1] - e_grid[lo], 1e-30);
    return theta[lo] + frac * (theta[lo + 1] - theta[lo]);
}

/// Maxwell fission spectrum χ(E) ∝ √E · exp(−E/θ).
/// Coveyou–Macpherson rejection sampler — mirrors
/// `MaxwellLaw::sample_maxwell` in `hdf5_reader.rs`:
///     E = −θ · (ln ξ₁ + cos²(π ξ₂ / 2) · ln ξ₃)
/// with rejection if E > E_in − u (the closed-form ENDF restriction).
/// Bounded retry — same 100-try safety valve as the CPU sampler.
__device__ __forceinline__ double sample_maxwell_theta(
    double E_inc, double theta, double u, PcgState* rng)
{
    double max_e = fmax(E_inc - u, 1e-5);
    for (int t = 0; t < 100; ++t) {
        double xi1 = fmax(pcg_uniform(rng), 1e-30);
        double xi2 = pcg_uniform(rng);
        double xi3 = fmax(pcg_uniform(rng), 1e-30);
        double c   = cos(PI * 0.5 * xi2);
        double e   = -theta * (log(xi1) + c * c * log(xi3));
        if (e > 0.0 && e <= max_e) return fmax(e, 1e-5);
    }
    return fmax(fmin(theta, max_e), 1e-5);
}

/// Evaporation fission spectrum χ(E) ∝ E · exp(−E/θ).
/// Direct inversion via the product of two uniforms — mirrors
/// `MaxwellLaw::sample_evaporation` in `hdf5_reader.rs`:
///     E = −θ · ln(ξ₁ · ξ₂)
/// with rejection if E > E_in − u.
__device__ __forceinline__ double sample_evaporation_theta(
    double E_inc, double theta, double u, PcgState* rng)
{
    double max_e = fmax(E_inc - u, 1e-5);
    for (int t = 0; t < 100; ++t) {
        double xi1 = fmax(pcg_uniform(rng), 1e-30);
        double xi2 = fmax(pcg_uniform(rng), 1e-30);
        double e   = -theta * log(xi1 * xi2);
        if (e > 0.0 && e <= max_e) return fmax(e, 1e-5);
    }
    return fmax(fmin(theta, max_e), 1e-5);
}

__device__ double sample_fission_energy(
    double E_inc, PcgState* rng, Params p, int hit_nuc)
{
    int fi_off = __ldg(&PTR_I(p, P_FIS_NUC_OFF)[hit_nuc]);
    int fi_n   = __ldg(&PTR_I(p, P_FIS_NUC_NINC)[hit_nuc]);
    if (fi_n <= 0) {
        // Tabular χ not stored for this nuclide — check for a Watt
        // closed-form (ENDF Law 11) upload before falling back to the
        // U-235 Cranberg parameters. Watt is the dominant
        // closed-form path for actinides whose MT=18 ships a(E),
        // b(E) (e.g. U-233 / U-234 multi-chance contributions).
        int w_n   = __ldg(&PTR_I(p, P_WATT_NUC_N)[hit_nuc]);
        if (w_n > 0) {
            int w_off = __ldg(&PTR_I(p, P_WATT_NUC_OFF)[hit_nuc]);
            const double* w_e = &PTR_D(p, P_WATT_INC_E)[w_off];
            const double* w_a = &PTR_D(p, P_WATT_A)[w_off];
            const double* w_b = &PTR_D(p, P_WATT_B)[w_off];
            double a_eV, b_inv_eV;
            if (E_inc <= w_e[0]) {
                a_eV = w_a[0]; b_inv_eV = w_b[0];
            } else if (E_inc >= w_e[w_n - 1]) {
                a_eV = w_a[w_n - 1]; b_inv_eV = w_b[w_n - 1];
            } else {
                int lo = 0, hi = w_n - 1;
                while (hi - lo > 1) {
                    int mid = (lo + hi) >> 1;
                    if (w_e[mid] <= E_inc) lo = mid; else hi = mid;
                }
                double t = (E_inc - w_e[lo]) /
                           fmax(w_e[lo + 1] - w_e[lo], 1e-30);
                a_eV     = w_a[lo] + t * (w_a[lo + 1] - w_a[lo]);
                b_inv_eV = w_b[lo] + t * (w_b[lo + 1] - w_b[lo]);
            }
            double u_cut = __ldg(&PTR_D(p, P_WATT_U)[hit_nuc]);
            double max_e = fmax(E_inc - u_cut, 1e-5);
            // Resample until E_out ≤ E_in − u; bound at 100 tries
            // for the same safety reason as the CPU sampler.
            for (int t = 0; t < 100; ++t) {
                double e_out = sample_watt_ab(a_eV, b_inv_eV, rng);
                if (e_out <= max_e) return e_out;
            }
            return fmin(sample_watt_ab(a_eV, b_inv_eV, rng), max_e);
        }
        // Maxwell (Law 7) / Evaporation (Law 9) — single θ(E_in)
        // table per nuclide, sampler chosen by `P_MAXEVAP_LAW`. The
        // host packs this slot for every nuclide whose ENDF
        // evaluation carries one of those laws (U-233 / U-234 are
        // Maxwell; Pu-240 / Pu-241 are Evaporation in several
        // evaluations). Closes the wrong-spectrum GPU bias that the
        // Cranberg fallback below was producing on those nuclides.
        int me_law = __ldg(&PTR_I(p, P_MAXEVAP_LAW)[hit_nuc]);
        int me_n   = __ldg(&PTR_I(p, P_MAXEVAP_NUC_N)[hit_nuc]);
        if (me_n > 0 && (me_law == 7 || me_law == 9)) {
            int me_off = __ldg(&PTR_I(p, P_MAXEVAP_NUC_OFF)[hit_nuc]);
            const double* me_e  = &PTR_D(p, P_MAXEVAP_INC_E)[me_off];
            const double* me_th = &PTR_D(p, P_MAXEVAP_THETA)[me_off];
            double theta  = maxevap_theta_at(E_inc, me_e, me_th, me_n);
            double u_cut  = __ldg(&PTR_D(p, P_MAXEVAP_U)[hit_nuc]);
            if (me_law == 7) {
                return sample_maxwell_theta(E_inc, theta, u_cut, rng);
            } else {
                return sample_evaporation_theta(E_inc, theta, u_cut, rng);
            }
        }
        // No Watt / Maxwell / Evaporation parameters — fall back to
        // U-235 Cranberg. This path now fires only for nuclides whose
        // evaluation carries a law the engine still doesn't handle
        // (Madland-Nix, ...). Safe to keep as a numerical floor; the
        // upload no longer logs a warning for Maxwell / Evaporation
        // since those are handled above.
        return sample_watt_ab(0.988e6, 2.249e-6, rng);
    }
    // Stage C step D — per-nuc base pointers for the four tabular
    // fission buffers. The dist_off lookups use the global-indexed
    // `P_FIS_DIST_LOCAL_OFF` (un-shifted values).
    const double* inc_e =
        (const double*) __ldg(&PTR_U64(p, P_FIS_INC_E_PTRS)[hit_nuc]);
    const double* nuc_eo =
        (const double*) __ldg(&PTR_U64(p, P_FIS_E_OUT_PTRS)[hit_nuc]);
    const double* nuc_cdf =
        (const double*) __ldg(&PTR_U64(p, P_FIS_CDF_PTRS)[hit_nuc]);
    const double* nuc_pdf =
        (const double*) __ldg(&PTR_U64(p, P_FIS_PDF_PTRS)[hit_nuc]);

    // Edge: below grid — sample directly from first bin.
    if (E_inc <= inc_e[0]) {
        int off = PTR_I(p, P_FIS_DIST_LOCAL_OFF)[fi_off];
        int sz  = PTR_I(p, P_FIS_DIST_SZ)[fi_off];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &nuc_eo[off], &nuc_cdf[off], &nuc_pdf[off], sz), 1e-5);
    }
    // Edge: above grid — sample from last bin.
    if (E_inc >= inc_e[fi_n-1]) {
        int off = PTR_I(p, P_FIS_DIST_LOCAL_OFF)[fi_off + fi_n - 1];
        int sz  = PTR_I(p, P_FIS_DIST_SZ)[fi_off + fi_n - 1];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &nuc_eo[off], &nuc_cdf[off], &nuc_pdf[off], sz), 1e-5);
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
    int off_l = PTR_I(p, P_FIS_DIST_LOCAL_OFF)[chosen];
    int sz_l  = PTR_I(p, P_FIS_DIST_SZ)[chosen];
    const double* eo_l = &nuc_eo[off_l];
    const double* cd_l = &nuc_cdf[off_l];
    const double* pd_l = &nuc_pdf[off_l];
    double e_out = sample_eout_bin(pcg_uniform(rng), eo_l, cd_l, pd_l, sz_l);

    // Scaled kinematic adjustment: remap e_out from chosen bin's
    // [el1_lo, el1_hi] to the interpolated [e1, eK] between both bins.
    int off_a = PTR_I(p, P_FIS_DIST_LOCAL_OFF)[chosen_lo];
    int sz_a  = PTR_I(p, P_FIS_DIST_SZ)[chosen_lo];
    int off_b = PTR_I(p, P_FIS_DIST_LOCAL_OFF)[chosen_hi];
    int sz_b  = PTR_I(p, P_FIS_DIST_SZ)[chosen_hi];
    const double* eo_a = &nuc_eo[off_a];
    const double* eo_b = &nuc_eo[off_b];
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

// ───────────────────────────────────────────────────────────────────────
// MT=91 continuum-inelastic outgoing energy sampler — mirrors
// `sample_fission_energy` above but reads from the P_INEL91_* slots
// and has no Watt / Cranberg fallback (MT=91 evaluations only ship a
// tabular law). Callers must check `P_INEL91_NUC_NINC[hit_nuc] > 0`
// before calling; the evaporation fallback in the kernel handles the
// `n_inc == 0` case (light isotopes / older ENDF evaluations).
//
// Same OpenMC `ContinuousTabular::sample` algorithm as the fission
// sampler: stochastic bin pick proportional to incident-E interpolation
// fraction, then `sample_eout_bin` on the chosen bin, then scaled
// kinematic remap onto the interpolated outgoing-energy support.
// ───────────────────────────────────────────────────────────────────────
__device__ double sample_inel91_energy(
    double E_inc, PcgState* rng, Params p, int hit_nuc)
{
    int fi_off = __ldg(&PTR_I(p, P_INEL91_NUC_OFF)[hit_nuc]);
    int fi_n   = __ldg(&PTR_I(p, P_INEL91_NUC_NINC)[hit_nuc]);
    if (fi_n <= 0) return -1.0;  // caller must guard with NUC_NINC > 0
    // Stage C step D — per-nuc base pointers.
    const double* inc_e =
        (const double*) __ldg(&PTR_U64(p, P_INEL91_INC_E_PTRS)[hit_nuc]);
    const double* nuc_eo =
        (const double*) __ldg(&PTR_U64(p, P_INEL91_E_OUT_PTRS)[hit_nuc]);
    const double* nuc_cdf =
        (const double*) __ldg(&PTR_U64(p, P_INEL91_CDF_PTRS)[hit_nuc]);
    const double* nuc_pdf =
        (const double*) __ldg(&PTR_U64(p, P_INEL91_PDF_PTRS)[hit_nuc]);

    if (E_inc <= inc_e[0]) {
        int off = PTR_I(p, P_INEL91_DIST_LOCAL_OFF)[fi_off];
        int sz  = PTR_I(p, P_INEL91_DIST_SZ)[fi_off];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &nuc_eo[off], &nuc_cdf[off], &nuc_pdf[off], sz), 1e-5);
    }
    if (E_inc >= inc_e[fi_n - 1]) {
        int off = PTR_I(p, P_INEL91_DIST_LOCAL_OFF)[fi_off + fi_n - 1];
        int sz  = PTR_I(p, P_INEL91_DIST_SZ)[fi_off + fi_n - 1];
        return fmax(sample_eout_bin(pcg_uniform(rng),
                                    &nuc_eo[off], &nuc_cdf[off], &nuc_pdf[off], sz), 1e-5);
    }

    int ie;
    {
        int lo = 0, hi = fi_n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (inc_e[mid] <= E_inc) lo = mid; else hi = mid;
        }
        ie = lo;
    }

    double r = (E_inc - inc_e[ie]) / fmax(inc_e[ie + 1] - inc_e[ie], 1e-30);
    bool pick_hi = pcg_uniform(rng) < r;
    int chosen_lo = fi_off + ie;
    int chosen_hi = fi_off + ie + 1;
    int chosen = pick_hi ? chosen_hi : chosen_lo;
    int off_l = PTR_I(p, P_INEL91_DIST_LOCAL_OFF)[chosen];
    int sz_l  = PTR_I(p, P_INEL91_DIST_SZ)[chosen];
    const double* eo_l = &nuc_eo[off_l];
    const double* cd_l = &nuc_cdf[off_l];
    const double* pd_l = &nuc_pdf[off_l];
    double e_out = sample_eout_bin(pcg_uniform(rng), eo_l, cd_l, pd_l, sz_l);

    int off_a = PTR_I(p, P_INEL91_DIST_LOCAL_OFF)[chosen_lo];
    int sz_a  = PTR_I(p, P_INEL91_DIST_SZ)[chosen_lo];
    int off_b = PTR_I(p, P_INEL91_DIST_LOCAL_OFF)[chosen_hi];
    int sz_b  = PTR_I(p, P_INEL91_DIST_SZ)[chosen_hi];
    const double* eo_a = &nuc_eo[off_a];
    const double* eo_b = &nuc_eo[off_b];
    double el1_lo = eo_l[0];
    double el1_hi = (sz_l > 0) ? eo_l[sz_l - 1] : el1_lo;
    double ea_lo  = eo_a[0];
    double ea_hi  = (sz_a > 0) ? eo_a[sz_a - 1] : ea_lo;
    double eb_lo  = eo_b[0];
    double eb_hi  = (sz_b > 0) ? eo_b[sz_b - 1] : eb_lo;
    double e1 = (1.0 - r) * ea_lo + r * eb_lo;
    double eK = (1.0 - r) * ea_hi + r * eb_hi;
    double span_l = el1_hi - el1_lo;
    double adjusted = (fabs(span_l) < 1e-30) ? e_out
                    : e1 + (e_out - el1_lo) * (eK - e1) / span_l;
    return fmax(adjusted, 1e-5);
}

// ───────────────────────────────────────────────────────────────────────
// Delayed-neutron outgoing energy — soft Watt spectrum mirroring the
// CPU `physics/collision.rs::sample_delayed_energy`. Delayed neutrons
// from precursor β-decay have a mean ~0.4 MeV (much softer than the
// prompt ~2 MeV); the single combined spectrum is sufficient for
// static k-eigenvalue, per-precursor breakdown is only needed for
// time-dependent kinetics. ENDF-style parameters: a = 0.4 MeV,
// b = 2.249 /MeV.
// ───────────────────────────────────────────────────────────────────────
__device__ __forceinline__ double sample_delayed_energy(PcgState* rng) {
    const double a = 4.0e5;     // 0.4 MeV in eV
    const double b = 2.249e-6;  // 2.249 /MeV in /eV
    const double term = 0.25 * a * a * b;
    // Bounded retry — matches the unbounded CPU loop with a 100-try cap
    // for the same safety reason as the Watt sampler above.
    for (int t = 0; t < 100; ++t) {
        double xi1 = fmax(pcg_uniform(rng), 1e-30);
        double e_prime = -a * log(xi1);
        double xi2 = pcg_uniform(rng);
        double e = e_prime + term
                 + (2.0 * xi2 - 1.0) * sqrt(a * a * b * e_prime) * 0.5;
        if (e > 0.0) return e;
    }
    return 1e-5;
}

// Fission emission with prompt/delayed split. β(E) = ν_delayed / ν_total
// is sampled per banked neutron; with probability β the outgoing energy
// is drawn from `sample_delayed_energy`, otherwise from the prompt χ
// (`sample_fission_energy`). Mirrors the CPU dispatch in
// `transport/simulate.rs:1294` and `physics/collision.rs:387`.
//
// `nu_total` is the value already looked up by the caller for the
// integer-neutron count; passing it in avoids a redundant lin-lin
// interpolation per emitted neutron.
__device__ __forceinline__ double sample_fission_emit_energy(
    double E_inc, double nu_total, PcgState* rng, Params p, int hit_nuc)
{
    int dnb_sz = __ldg(&PTR_I(p, P_DNB_SIZES)[hit_nuc]);
    if (dnb_sz > 0 && nu_total > 0.0) {
        // Stage C step D — per-nuclide pointer load. Offset becomes
        // 0 since the per-nuc slice starts at its own base.
        const double* dnb_e =
            (const double*) __ldg(&PTR_U64(p, P_DNB_E_PTRS)[hit_nuc]);
        const double* dnb_v =
            (const double*) __ldg(&PTR_U64(p, P_DNB_V_PTRS)[hit_nuc]);
        double nu_d = nu_bar_lookup(E_inc, dnb_e, dnb_v, 0, dnb_sz);
        double beta = nu_d / nu_total;
        if (beta > 1.0) beta = 1.0;
        if (beta > 0.0 && pcg_uniform(rng) < beta) {
            return sample_delayed_energy(rng);
        }
    }
    return sample_fission_energy(E_inc, rng, p, hit_nuc);
}

// CDF-invert mu within one bin using a pre-drawn xi.
// Inverse-CDF μ sampler. Quadratic lin-lin inversion when a PDF array
// is supplied (the OpenMC `Tabular::sample` formula — mirrors
// `TabularMuDist::sample_with_xi` in hdf5_reader.rs and the
// `sample_eout_bin` quadratic in this file). Falls back to histogram-
// PDF / linear-CDF interpolation when `pd == nullptr` or the PDF is
// zero — that's the legacy path, kept for backward compatibility but
// no longer the default for ENDF/B-VII.1 angular data.
//
// Without the quadratic path every forward-peaked elastic scatter on
// Al-27 / Mg / Cr / Mn / W-180..186 was biased — surfaced as
// ~+500-700 pcm CPU↔GPU gap on ieu-met-fast-001 / heu-met-fast-011.
__device__ __forceinline__ double sample_mu_bin(
    double xi, const double* mu, const double* cd, const double* pd, int sz)
{
    if (sz <= 1) return 2.0*xi - 1.0;
    int lo=0, hi=sz-1;
    while (hi-lo > 1) { int mid=(lo+hi) >> 1; if (cd[mid] <= xi) lo=mid; else hi=mid; }
    const double mu_lo = mu[lo];
    const double mu_hi = mu[hi];
    const double cdf_lo = cd[lo];
    const double cdf_hi = cd[hi];
    const double dmu = mu_hi - mu_lo;
    if (fabs(cdf_hi - cdf_lo) < 1e-15) return fmax(-1.0, fmin(1.0, mu_lo));
    if (pd != nullptr && dmu > 0.0) {
        const double p_lo = pd[lo];
        const double p_hi = pd[hi];
        const double dc = xi - cdf_lo;
        if (p_lo > 0.0 || p_hi > 0.0) {
            const double slope = (p_hi - p_lo) / dmu;
            if (fabs(slope) < 1e-30) {
                if (p_lo > 0.0) {
                    double m = mu_lo + dc / p_lo;
                    return fmax(-1.0, fmin(1.0, m));
                }
            } else {
                const double disc = p_lo * p_lo + 2.0 * slope * dc;
                if (disc >= 0.0) {
                    double m = mu_lo + (sqrt(disc) - p_lo) / slope;
                    return fmax(-1.0, fmin(1.0, m));
                }
            }
        }
    }
    // Histogram fallback.
    double f = (xi - cdf_lo) / fmax(cdf_hi - cdf_lo, 1e-30);
    return fmax(-1.0, fmin(1.0, mu_lo + f * dmu));
}

__device__ double sample_angular_dist(
    double E, PcgState* rng, Params p, int hit_nuc)
{
    int a_off = __ldg(&PTR_I(p, P_ANG_NUC_OFF)[hit_nuc]);
    int a_ne = __ldg(&PTR_I(p, P_ANG_NUC_NE)[hit_nuc]);
    if (a_ne <= 0) return 2.0*pcg_uniform(rng)-1.0;
    // Stage C step D — per-nuc base pointers. The kernel still
    // uses the GLOBAL P_ANG_DIST_LOCAL_OFF / P_ANG_DIST_SZ arrays
    // (indexed by global ang_energy idx `chosen`), but the VALUES
    // are within-nuc ang_mu offsets so the per-nuc `nuc_mu` /
    // `nuc_cdf` base pointers can be used directly.
    const double* ae =
        (const double*) __ldg(&PTR_U64(p, P_ANG_E_PTRS)[hit_nuc]);
    const double* nuc_mu =
        (const double*) __ldg(&PTR_U64(p, P_ANG_MU_PTRS)[hit_nuc]);
    const double* nuc_cdf =
        (const double*) __ldg(&PTR_U64(p, P_ANG_CDF_PTRS)[hit_nuc]);
    const double* nuc_pdf =
        (const double*) __ldg(&PTR_U64(p, P_ANG_PDF_PTRS)[hit_nuc]);

    // Edge: below grid
    if (E <= ae[0]) {
        int off = PTR_I(p, P_ANG_DIST_LOCAL_OFF)[a_off];
        int sz  = PTR_I(p, P_ANG_DIST_SZ)[a_off];
        return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
    }
    // Edge: above grid
    if (E >= ae[a_ne-1]) {
        int off = PTR_I(p, P_ANG_DIST_LOCAL_OFF)[a_off + a_ne - 1];
        int sz  = PTR_I(p, P_ANG_DIST_SZ)[a_off + a_ne - 1];
        return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
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
    int off = PTR_I(p, P_ANG_DIST_LOCAL_OFF)[chosen];
    int sz  = PTR_I(p, P_ANG_DIST_SZ)[chosen];
    return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
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
    double E, PcgState* rng, Params p, int global_lev_idx, int hit_nuc)
{
    int n_e = __ldg(&PTR_I(p, P_LEV_ANG_LEV_NE)[global_lev_idx]);
    if (n_e <= 0) return 2.0 * pcg_uniform(rng) - 1.0;

    // Stage C step D — per-nuc bases for energies/mu/cdf. The
    // within-nuc ang_energy offset for this level is supplied by
    // P_LEV_ANG_LEV_LOCAL_OFF (global concat of per-nuc values).
    // P_LEV_ANG_LEV_OFF still exists as the GLOBAL ang_energy
    // offset and is used to index into P_LEV_ANG_DIST_LOCAL_OFF
    // (which is also globally indexed but holds within-nuc ang_mu
    // offsets so the per-nuc mu/cdf bases can be applied).
    int e_off_local = __ldg(&PTR_I(p, P_LEV_ANG_LEV_LOCAL_OFF)[global_lev_idx]);
    int e_off_global = __ldg(&PTR_I(p, P_LEV_ANG_LEV_OFF)[global_lev_idx]);
    const double* nuc_ae =
        (const double*) __ldg(&PTR_U64(p, P_LEV_ANG_E_PTRS)[hit_nuc]);
    const double* nuc_mu =
        (const double*) __ldg(&PTR_U64(p, P_LEV_ANG_MU_PTRS)[hit_nuc]);
    const double* nuc_cdf =
        (const double*) __ldg(&PTR_U64(p, P_LEV_ANG_CDF_PTRS)[hit_nuc]);
    const double* nuc_pdf =
        (const double*) __ldg(&PTR_U64(p, P_LEV_ANG_PDF_PTRS)[hit_nuc]);
    const double* ae = &nuc_ae[e_off_local];

    // Below / above grid: pick the edge distribution directly.
    if (E <= ae[0]) {
        int off = PTR_I(p, P_LEV_ANG_DIST_LOCAL_OFF)[e_off_global];
        int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[e_off_global];
        return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
    }
    if (E >= ae[n_e - 1]) {
        int off = PTR_I(p, P_LEV_ANG_DIST_LOCAL_OFF)[e_off_global + n_e - 1];
        int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[e_off_global + n_e - 1];
        return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
    }

    int ie; { int lo = 0, hi = n_e - 1;
        while (hi - lo > 1) { int mid = (lo + hi) / 2;
            if (ae[mid] <= E) lo = mid; else hi = mid; } ie = lo; }

    double r = (E - ae[ie]) / fmax(ae[ie+1] - ae[ie], 1e-30);
    bool pick_hi = pcg_uniform(rng) < r;
    int chosen = e_off_global + (pick_hi ? ie + 1 : ie);
    int off = PTR_I(p, P_LEV_ANG_DIST_LOCAL_OFF)[chosen];
    int sz  = PTR_I(p, P_LEV_ANG_DIST_SZ)[chosen];
    return sample_mu_bin(pcg_uniform(rng), &nuc_mu[off], &nuc_cdf[off], &nuc_pdf[off], sz);
}

// Per-slot accessor helpers. `slot` is the index returned by
// `P_SAB_SLOT_PER_NUC[nuc_idx]`. The flat data arrays at slots 43-55
// hold all packs concatenated; per-slot tables (slots 125-129) point
// into them. When `n_slots == 0` the caller short-circuits before
// these are reached.
__device__ __forceinline__ int sab_slot_inc_e_off(int slot, Params p) {
    return PTR_I(p, P_SAB_SLOT_INC_E_OFF)[slot];
}
__device__ __forceinline__ int sab_slot_n_inc(int slot, Params p) {
    return PTR_I(p, P_SAB_SLOT_N_INC)[slot];
}
__device__ __forceinline__ int sab_slot_eout_table_off(int slot, Params p) {
    return PTR_I(p, P_SAB_SLOT_EOUT_TABLE_OFF)[slot];
}
__device__ __forceinline__ int sab_slot_mu_table_off(int slot, Params p) {
    return PTR_I(p, P_SAB_SLOT_MU_TABLE_OFF)[slot];
}
__device__ __forceinline__ double sab_slot_emax(int slot, Params p) {
    return PTR_D(p, P_SAB_SLOT_EMAX)[slot];
}

// ── Stochastic temperature interpolation across SAB kT columns ─────
//
// Each SAB-bearing nuclide owns `P_SAB_SLOT_COUNT_PER_NUC[nuc]`
// consecutive slots starting at `P_SAB_SLOT_PER_NUC[nuc]`. Each slot
// carries data for one kT column of the TSL, with kT recorded in
// `P_SAB_SLOT_KT[slot]`. Returns the chosen slot index for the
// requested `cell_kT`, drawn stochastically between the two
// bracketing columns via an inverse-CDF on the kT axis (matches CPU
// `ThermalScatteringData::select_temperature`).
//
// Returns -1 when the nuclide has no SAB bound (slot_count == 0).
__device__ __forceinline__ int sab_select_slot(
    int nuc_idx, double cell_kT, PcgState* rng, Params p)
{
    int first = PTR_I(p, P_SAB_SLOT_PER_NUC)[nuc_idx];
    if (first < 0) return -1;
    int count = PTR_I(p, P_SAB_SLOT_COUNT_PER_NUC)[nuc_idx];
    if (count <= 0) return -1;
    if (count == 1) return first;
    const double* kts = &PTR_D(p, P_SAB_SLOT_KT)[first];
    if (cell_kT <= kts[0]) return first;
    if (cell_kT >= kts[count - 1]) return first + count - 1;
    int lo = 0, hi = count - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (kts[mid] <= cell_kT) lo = mid; else hi = mid;
    }
    double f = (cell_kT - kts[lo]) / fmax(kts[hi] - kts[lo], 1e-30);
    return first + (pcg_uniform(rng) < f ? hi : lo);
}

// Variant for sites that lack an rng (e.g. XS evaluator on the macro
// path) — returns the LOWER bracketing slot deterministically. Used
// only when the caller can't draw a uniform; the bias is bounded by
// (cell_kT - kt_lo) / (kt_hi - kt_lo) × Δσ, which is small for the
// typical 50-100 K column spacing.
__device__ __forceinline__ int sab_select_slot_det(
    int nuc_idx, double cell_kT, Params p)
{
    int first = PTR_I(p, P_SAB_SLOT_PER_NUC)[nuc_idx];
    if (first < 0) return -1;
    int count = PTR_I(p, P_SAB_SLOT_COUNT_PER_NUC)[nuc_idx];
    if (count <= 0) return -1;
    if (count == 1) return first;
    const double* kts = &PTR_D(p, P_SAB_SLOT_KT)[first];
    if (cell_kT <= kts[0]) return first;
    if (cell_kT >= kts[count - 1]) return first + count - 1;
    int lo = 0, hi = count - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (kts[mid] <= cell_kT) lo = mid; else hi = mid;
    }
    return first + lo;
}

// ── SAB elastic channel ────────────────────────────────────────────
//
// Per-slot dispatch on `P_SAB_SLOT_ELASTIC_MODE[slot]`:
//   0 = none, returns σ = 0 / no-op angular sample.
//   1 = Coherent (Bragg) — σ(E) = (1/E) Σ_{E_i < E} s_i ; angular
//       sample picks an edge by cumulative factor and computes
//       μ = 1 − 2 E_i / E (OpenMC docs Eq. 79 + Eq. 82).
//   2 = Incoherent (Debye-Waller) — σ(E) = σ_b/(2) · (1 − e^{-4EW})
//       / (2 E W) ; angular sample uses the inverse-CDF closed form
//       μ = 1 + ln(1 − ξ (1 − e^{-4EW})) / (2EW) (Mac MacFarlane
//       NJOY THERMR notes / OpenMC sample_inelastic in CE).
__device__ __forceinline__ int sab_slot_elastic_mode(int slot, Params p) {
    return PTR_I(p, P_SAB_SLOT_ELASTIC_MODE)[slot];
}

__device__ double sab_coherent_xs(double E, int slot, Params p) {
    int off = PTR_I(p, P_SAB_SLOT_COH_OFF)[slot];
    int n   = PTR_I(p, P_SAB_SLOT_COH_N)[slot];
    if (off < 0 || n <= 0 || E <= 0.0) return 0.0;
    const double* edges   = &PTR_D(p, P_SAB_COH_BRAGG_EDGES)[off];
    const double* factors = &PTR_D(p, P_SAB_COH_FACTORS)[off];
    // Find idx = first edge index whose energy is >= E. Then the sum
    // of structure factors for edges below E is `factors[idx-1]`
    // because the input is cumulative.
    if (E <= edges[0]) return 0.0;
    if (E >= edges[n-1]) return factors[n-1] / E;
    int lo = 0, hi = n;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (edges[mid] < E) lo = mid; else hi = mid;
    }
    // edges[lo] < E <= edges[hi], so factors[lo] is the cumulative
    // sum below E.
    return factors[lo] / E;
}

__device__ double sab_incoherent_elastic_xs(double E, int slot, Params p) {
    double bxs = PTR_D(p, P_SAB_SLOT_INC_BOUND_XS)[slot];
    double w   = PTR_D(p, P_SAB_SLOT_INC_DEBYE_WALLER)[slot];
    if (bxs <= 0.0 || w <= 0.0 || E <= 0.0) return 0.0;
    double x = 4.0 * E * w;
    if (x < 1e-10) return bxs;  // limit (1−e^{-x})/x → 1
    return bxs * 0.5 * (1.0 - exp(-x)) / (2.0 * E * w);
}

__device__ double sab_elastic_xs(double E, int slot, Params p) {
    if (slot < 0) return 0.0;
    int mode = sab_slot_elastic_mode(slot, p);
    if (mode == 1) return sab_coherent_xs(E, slot, p);
    if (mode == 2) return sab_incoherent_elastic_xs(E, slot, p);
    return 0.0;
}

// Sample an elastic-scatter angle for the slot's mode. Energy is
// preserved by elastic kinematics; caller keeps E_in.
__device__ double sab_elastic_sample_mu(double E, int slot, Params p,
                                        PcgState* rng) {
    if (slot < 0) return 2.0 * pcg_uniform(rng) - 1.0;
    int mode = sab_slot_elastic_mode(slot, p);
    if (mode == 1) {
        // Coherent: sample a Bragg edge weighted by factor s_i, then
        // μ = 1 − 2 E_i / E. (OpenMC docs Eq. 81–82.)
        int off = PTR_I(p, P_SAB_SLOT_COH_OFF)[slot];
        int n   = PTR_I(p, P_SAB_SLOT_COH_N)[slot];
        if (off < 0 || n <= 0 || E <= 0.0) {
            return 2.0 * pcg_uniform(rng) - 1.0;
        }
        const double* edges   = &PTR_D(p, P_SAB_COH_BRAGG_EDGES)[off];
        const double* factors = &PTR_D(p, P_SAB_COH_FACTORS)[off];
        // Number of edges below E:
        int n_eff;
        if (E <= edges[0]) return 2.0 * pcg_uniform(rng) - 1.0;
        if (E >= edges[n-1]) n_eff = n;
        else {
            int lo = 0, hi = n;
            while (hi - lo > 1) {
                int mid = (lo + hi) >> 1;
                if (edges[mid] < E) lo = mid; else hi = mid;
            }
            n_eff = lo + 1;  // edges 0..lo inclusive are below E
        }
        double total_s = factors[n_eff - 1];
        double xi = pcg_uniform(rng) * total_s;
        // First edge whose cumulative factor >= xi.
        int lo = 0, hi = n_eff;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (factors[mid] < xi) lo = mid; else hi = mid;
        }
        int edge_idx = (factors[lo] < xi) ? hi : lo;
        if (edge_idx >= n_eff) edge_idx = n_eff - 1;
        double mu = 1.0 - 2.0 * edges[edge_idx] / E;
        return fmax(-1.0, fmin(1.0, mu));
    }
    if (mode == 2) {
        // Incoherent Debye-Waller: inverse CDF
        //   F(μ) = (1 − e^{-2EW(1−μ)}) / (1 − e^{-4EW})
        //   μ = 1 + ln(1 − ξ(1 − e^{-4EW})) / (2EW)
        double w = PTR_D(p, P_SAB_SLOT_INC_DEBYE_WALLER)[slot];
        if (w <= 0.0 || E <= 0.0) return 2.0 * pcg_uniform(rng) - 1.0;
        double four_ew = 4.0 * E * w;
        double xi = pcg_uniform(rng);
        double denom = 1.0 - exp(-four_ew);
        if (denom <= 1e-20) return 2.0 * pcg_uniform(rng) - 1.0;
        double mu = 1.0 + log(fmax(1.0 - xi * denom, 1e-300)) / (2.0 * E * w);
        return fmax(-1.0, fmin(1.0, mu));
    }
    return 2.0 * pcg_uniform(rng) - 1.0;
}

__device__ double sab_total_xs(double E, int slot, Params p) {
    if (slot < 0) return 0.0;
    int n = sab_slot_n_inc(slot, p);
    if (n <= 0) return 0.0;
    int e_off = sab_slot_inc_e_off(slot, p);
    const double* e = &PTR_D(p, P_SAB_INC_E)[e_off];
    const double* xs = &PTR_D(p, P_SAB_XS)[e_off];
    if (E <= e[0]) return xs[0];
    if (E >= e[n-1]) return xs[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(lo+hi) >> 1;if(e[mid]<=E)lo=mid;else hi=mid;}
    double f=(E-e[lo])/fmax(e[hi]-e[lo],1e-30);
    return xs[lo]+f*(xs[hi]-xs[lo]);
}

__device__ void sab_sample(
    double E_in, PcgState* rng, int slot, Params p,
    double* E_out, double* mu_out)
{
    if (slot < 0) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }

    // Roll elastic vs inelastic by their cross-section ratio at E_in.
    // If we land on the elastic branch, energy is preserved (Bragg /
    // Debye-Waller is purely angular); the inelastic branch samples
    // E_out / μ from the continuous SAB tables below.
    double inel = sab_total_xs(E_in, slot, p);
    double el   = sab_elastic_xs(E_in, slot, p);
    double tot  = inel + el;
    if (tot > 0.0 && pcg_uniform(rng) * tot < el) {
        *E_out  = E_in;
        *mu_out = sab_elastic_sample_mu(E_in, slot, p, rng);
        return;
    }
    int n = sab_slot_n_inc(slot, p);
    if (n <= 0) { *E_out=E_in; *mu_out=2.0*pcg_uniform(rng)-1.0; return; }

    int e_off = sab_slot_inc_e_off(slot, p);
    int eo_tbl_off = sab_slot_eout_table_off(slot, p);
    const double* inc_e = &PTR_D(p, P_SAB_INC_E)[e_off];
    const int* eout_off_arr = &PTR_I(p, P_SAB_EOUT_OFF)[eo_tbl_off];
    const int* eout_sz_arr  = &PTR_I(p, P_SAB_EOUT_SZ)[eo_tbl_off];
    // mu_offsets_flat / mu_sizes_flat parallel e_out_flat globally,
    // so `eo_off + j` already indexes them correctly without a
    // per-slot table offset.
    (void)sab_slot_mu_table_off(slot, p);

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

    int eo_off = eout_off_arr[ell];
    int eo_sz  = eout_sz_arr[ell];
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
    int off_lo = eout_off_arr[i_lo];
    int sz_lo  = eout_sz_arr[i_lo];
    int off_hi = eout_off_arr[i_hi];
    int sz_hi  = eout_sz_arr[i_hi];
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

// Pick the band index at energy-grid row `base` from the cumulative
// probability table `cp` using one ξ. Mirrors the `upper_bound` /
// CPU `pick` closure (xs_provider.rs:518 → hdf5_reader.rs:1975).
__device__ __forceinline__ int urr_pick_band(
    const double* cp, int base, int n_b, double xi)
{
    int band = n_b - 1;
    for (int b = 0; b < n_b; b++) {
        if (xi < cp[base + b]) { band = b; break; }
    }
    return band;
}

// URR factor lookup. Mirrors CPU `UrrProbabilityTables::sample`:
//   1. Find bracketing energy indices i_lo, i_hi = i_lo + 1.
//   2. Sample band INDEPENDENTLY at both rows using the SAME ξ (the
//      "correlated CDF interpolation" OpenMC documents in
//      `calculate_urr_xs`).
//   3. Look up the per-band factor at both rows.
//   4. Interpolate between them — lin-lin (interp == 2) or log-log
//      (interp == 5).
// Edge case: at the boundary (i_lo == n_e-1, or E == ue[0]) use the
// single-row lookup, matching CPU.
//
// Pre-fix the GPU just used the i_lo row's factor — no interpolation
// at all. That biased the URR factor toward the lower-energy band
// at every intermediate energy → systematic spectrum-hardening on
// multi-nuclide structural-reflector cases (Fe / Cr / Mn / Ni / Cu
// URR ranges in IEU + bare-HEU benchmarks).
__device__ void apply_urr(
    Params p, int nuc_idx,
    double* sig_el, double* sig_fis, double* sig_cap, double E, double xi)
{
    int n_e = __ldg(&PTR_I(p, P_URR_N_ENERGIES)[nuc_idx]);
    if (n_e <= 0) return;
    int n_b = __ldg(&PTR_I(p, P_URR_N_BANDS)[nuc_idx]);
    const double* ue =
        (const double*) __ldg(&PTR_U64(p, P_URR_E_PTRS)[nuc_idx]);
    if (E < ue[0] || E > ue[n_e-1]) return;
    int ie = 0; {
        int lo = 0, hi = n_e - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (ue[mid] <= E) lo = mid; else hi = mid;
        }
        ie = lo;
    }
    const double* cp =
        (const double*) __ldg(&PTR_U64(p, P_URR_CP_PTRS)[nuc_idx]);
    // ft (total factor) not used — reaction-specific factors applied directly
    (void) PTR_U64(p, P_URR_TF_PTRS);
    const double* fe_arr =
        (const double*) __ldg(&PTR_U64(p, P_URR_EF_PTRS)[nuc_idx]);
    const double* ff_arr =
        (const double*) __ldg(&PTR_U64(p, P_URR_FF_PTRS)[nuc_idx]);
    const double* fc_arr =
        (const double*) __ldg(&PTR_U64(p, P_URR_CF_PTRS)[nuc_idx]);

    int base_lo = ie * n_b;
    int band_lo = urr_pick_band(cp, base_lo, n_b, xi);
    double fe = fe_arr[base_lo + band_lo];
    double ff = ff_arr[base_lo + band_lo];
    double fc = fc_arr[base_lo + band_lo];

    // Edge case: at the upper bin or exactly on the lower energy
    // — single-row lookup, matching CPU's
    // `if (n_e == 1 || i_lo+1 >= n_e || energy <= self.energies[0])`.
    if (ie + 1 < n_e && E > ue[0]) {
        int base_hi = (ie + 1) * n_b;
        int band_hi = urr_pick_band(cp, base_hi, n_b, xi);
        double fe_hi = fe_arr[base_hi + band_hi];
        double ff_hi = ff_arr[base_hi + band_hi];
        double fc_hi = fc_arr[base_hi + band_hi];

        double e_lo = ue[ie], e_hi = ue[ie + 1];
        int interp = __ldg(&PTR_I(p, P_URR_INTERP)[nuc_idx]);
        double f;
        if (interp == 5 && e_lo > 0.0 && e_hi > 0.0) {
            // Log-log
            double den = log(e_hi / e_lo);
            f = (fabs(den) > 1e-30) ? log(E / e_lo) / den : 0.0;
        } else {
            // Lin-lin (interp == 2, default)
            f = (E - e_lo) / fmax(e_hi - e_lo, 1e-30);
        }
        fe = (1.0 - f) * fe + f * fe_hi;
        ff = (1.0 - f) * ff + f * ff_hi;
        fc = (1.0 - f) * fc + f * fc_hi;
    }

    int ms = __ldg(&PTR_I(p, P_URR_MULT_SM)[nuc_idx]);
    if (ms) { *sig_el *= fe; *sig_fis *= ff; *sig_cap *= fc; }
    else    { *sig_el  = fe; *sig_fis  = ff; *sig_cap  = fc; }
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
            // Stage C step D — direct per-nuclide pointer load.
            // basis_ptrs[key] is the CUdeviceptr of
            // PerNuclideGpu::basis[slot]; the kernel no longer
            // indirects through the all_basis concatenation.
            xs = svd_reconstruct(
                (const double*) PTR_U64(p, P_BASIS_PTRS)[key],
                (const double*) PTR_U64(p, P_COEFFS_PTRS)[key],
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

// ═══════════════════════════════════════════════════════════════════════
// Per-nuclide macroscopic XS evaluation.
//
// Returns Σ_x = N · σ_x(E) for the six tracked reactions plus the
// macroscopic total. Encapsulates the SVD / Pointwise / WMP / URR /
// S(α,β) lattice so the calling kernels can store a single
// `nuc_t[MAX_NUC_PER_MAT]` array for nuclide selection and recompute
// only the hit nuclide's reaction breakdown at sampling time. This is
// what lets `MAX_NUC_PER_MAT = 32` fit in registers: the streaming
// pattern replaces 7 × 32-wide per-thread arrays (1.8 KB → spill) with
// one 32-wide array (256 B, stays in registers / L1).
//
// Deterministic given `urr_xi`. Callers pre-draw `urr_xi` once at the
// top of the collision step and pass the same value to both the
// nuclide-selection pass and the reaction-selection pass, so the two
// calls produce identical XS.
// ═══════════════════════════════════════════════════════════════════════
struct NuclideMacroXs {
    double s_t;
    double s_el, s_inel, s_n2n, s_n3n, s_fis, s_cap;
};

// `cell_kT` (eV) drives the SAB stochastic-temperature lookup; pass
// `mat_kT[mat]` from the caller. -1.0 disables SAB temp interp and
// falls back to the legacy single-temp behaviour (used by debug
// kernels that don't carry a meaningful temperature).
__device__ NuclideMacroXs eval_nuclide_macro_xs(
    int ni, double Ni, double E, double urr_xi,
    int sab_nuc_idx, int rank, Params p, double cell_kT)
{
    int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[ni]);
    int n_e   = __ldg(&PTR_I(p, P_N_ENERGIES)[ni]);
    const double* grid = &PTR_D(p, P_ENERGY_GRIDS)[g_off];
    int e_idx = energy_index(grid, n_e, E);
    double log_frac = 0.0;
    if (e_idx + 1 < n_e && grid[e_idx] > 0.0) {
        double log_e = log(E);
        double log_lo = log(grid[e_idx]);
        double log_hi = log(grid[e_idx + 1]);
        if (log_hi > log_lo) log_frac = (log_e - log_lo) / (log_hi - log_lo);
        if (log_frac < 0.0) log_frac = 0.0;
        if (log_frac > 1.0) log_frac = 1.0;
    }

    double s_el = 0, s_inel = 0, s_n2n = 0, s_n3n = 0, s_fis = 0, s_cap = 0, micro_t = 0;

    if (__ldg(&PTR_I(p, P_HAS_PW)[ni])) {
        // Stage C step D — direct per-nuclide pointer load. The
        // bundle-level P_PW_XS / P_PW_OFF indirection is no longer
        // needed; pw_ptrs[ni] is the device address of
        // PerNuclideGpu::pointwise_xs.
        const double* pw_base = (const double*) __ldg(&PTR_U64(p, P_PW_XS_PTRS)[ni]);
        const double* pw0 = &pw_base[e_idx * 7];
        const double* pw1 = (e_idx + 1 < n_e) ? &pw_base[(e_idx + 1) * 7] : pw0;
        double xs7[7];
        for (int ch = 0; ch < 7; ch++) {
            double lo = pw0[ch], hi = pw1[ch];
            xs7[ch] = (lo > 1e-30 && hi > 1e-30 && log_frac > 0.0)
                ? exp(log(lo) + log_frac * (log(hi) - log(lo))) : lo;
        }
        s_el = xs7[0]; s_inel = xs7[1]; s_n2n = xs7[2]; s_n3n = xs7[3];
        s_fis = xs7[4]; s_cap = xs7[5]; micro_t = xs7[6];
        double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
        s_cap = fmax(micro_t - partials, 0.0);
    } else {
        bool has_inel_k = false;
        for (int r = 0; r < 6; r++) {
            int key = ni * N_REACTIONS + r;
            if (__ldg(&PTR_I(p, P_HAS_REACTION)[key])) {
                // Stage C step D — direct per-nuclide pointer load.
                double s = svd_reconstruct_interp(
                    (const double*) __ldg(&PTR_U64(p, P_BASIS_PTRS)[key]),
                    (const double*) __ldg(&PTR_U64(p, P_COEFFS_PTRS)[key]),
                    e_idx, n_e, rank, log_frac);
                if (r == 0)      s_el = s;
                else if (r == 1) { s_inel = s; has_inel_k = true; }
                else if (r == 2) s_n2n = s;
                else if (r == 3) s_n3n = s;
                else if (r == 4) s_fis = s;
                else if (r == 5) s_cap = s;
            }
        }
        if (!has_inel_k) {
            int lv_off = __ldg(&PTR_I(p, P_LEVEL_OFFSETS)[ni]);
            int n_lev  = __ldg(&PTR_I(p, P_LEVEL_COUNTS)[ni]);
            double lsum = 0.0;
            // Stage C step D — per-nuclide level pointer base. Hoisted
            // out of the per-level loop; each loop iter loads a u64
            // (P_LEVEL_BLOCAL_OFF[gl]) instead of two i32s + one f64*
            // indirection.
            const double* nuc_lvl_basis =
                (const double*) __ldg(&PTR_U64(p, P_LEVEL_BASIS_PTRS)[ni]);
            const double* nuc_lvl_coeffs =
                (const double*) __ldg(&PTR_U64(p, P_LEVEL_COEFFS_PTRS)[ni]);
            for (int l = 0; l < n_lev; l++) {
                int gl = lv_off + l;
                if (!__ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) continue;
                if (E < __ldg(&PTR_D(p, P_LEVEL_THR)[gl])) continue;
                double lxs = svd_reconstruct_interp(
                    &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                    &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                    e_idx, n_e, rank, log_frac);
                if (lxs > 0.0) lsum += lxs;
            }
            s_inel = lsum;
        }
        if (__ldg(&PTR_I(p, P_HAS_TOTAL_XS)[ni])) {
            // Stage C step D — per-nuclide pointer load.
            const double* tot_grid =
                (const double*) __ldg(&PTR_U64(p, P_TOTAL_XS_PTRS)[ni]);
            double tot_lo = tot_grid[e_idx];
            double tot_hi = (e_idx + 1 < n_e) ? tot_grid[e_idx + 1] : tot_lo;
            double tot = (tot_lo > 1e-30 && tot_hi > 1e-30 && log_frac > 0.0)
                ? exp(log(tot_lo) + log_frac * (log(tot_hi) - log(tot_lo)))
                : tot_lo;
            double partials = s_el + s_inel + s_n2n + s_n3n + s_fis;
            s_cap = fmax(tot - partials, 0.0);
            micro_t = tot;
        } else {
            micro_t = s_el + s_inel + s_n2n + s_n3n + s_fis + s_cap;
        }
    }

    bool in_wmp = false;
    if (__ldg(&PTR_I(p, P_WMP_HAS)[ni])) {
        double e_lo = __ldg(&PTR_D(p, P_WMP_E_MIN)[ni]);
        double e_hi = __ldg(&PTR_D(p, P_WMP_E_MAX)[ni]);
        if (E >= e_lo && E <= e_hi) in_wmp = true;
    }
    if (!in_wmp) {
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
        int w_nw   = __ldg(&PTR_I(p, P_WMP_N_WINDOWS)[ni]);
        int w_fo   = __ldg(&PTR_I(p, P_WMP_FIT_ORDER)[ni]);
        int w_fiss = __ldg(&PTR_I(p, P_WMP_FISSIONABLE)[ni]);
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

    // S(α,β) override for nuclides that have a TSL slot bound. The
    // per-nuclide lookup `P_SAB_SLOT_PER_NUC[ni]` is -1 when no TSL is
    // attached, so this branch is a no-op for the bare-uranium /
    // metal-fast geometries. `sab_nuc_idx` is retained as a legacy
    // function parameter for ABI stability but is no longer consulted
    // — the per-nuclide table is authoritative.
    (void)sab_nuc_idx;
    if (SCALAR_I(p, P_SAB_N_SLOTS) > 0 && E > 0.0) {
        // Deterministic select on the XS path (no rng in scope) — the
        // sampling path below uses the stochastic variant. Bias from
        // picking the lower-bracket column on the XS is bounded by
        // the per-bin σ difference, much smaller than ignoring temp
        // interp entirely.
        int sab_slot = sab_select_slot_det(ni, cell_kT, p);
        if (sab_slot >= 0 && E < PTR_D(p, P_SAB_SLOT_EMAX)[sab_slot]) {
            // Replace the free-atom elastic with SAB inelastic + SAB
            // elastic (coh / inc) — the elastic block was previously
            // unimplemented on the GPU, biasing thick-moderator
            // problems cold (HEU-MET-FAST-058 case-1 was -2500 pcm).
            double sab_inel = sab_total_xs(E, sab_slot, p);
            double sab_el   = sab_elastic_xs(E, sab_slot, p);
            double sab_tot  = sab_inel + sab_el;
            if (sab_tot > 0.0) {
                double delta = sab_tot - s_el;
                micro_t += delta;
                s_el = sab_tot;
            }
        }
    }

    NuclideMacroXs out;
    out.s_t    = Ni * micro_t;
    out.s_el   = Ni * s_el;
    out.s_inel = Ni * s_inel;
    out.s_n2n  = Ni * s_n2n;
    out.s_n3n  = Ni * s_n3n;
    out.s_fis  = Ni * s_fis;
    out.s_cap  = Ni * s_cap;
    return out;
}

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

        // XS lookup — Pass 1 stores only Σ_t per nuclide; per-reaction
        // breakdown is recomputed for the chosen nuclide at sampling
        // time. This keeps per-thread footprint flat as
        // MAX_NUC_PER_MAT grows (see eval_nuclide_macro_xs comment).
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t = 0;
        double nuc_t[MAX_NUC_PER_MAT] = {};
        double urr_xi = pcg_uniform(&rng);

        for (int i = 0; i < n_nuc; i++) {
            int ni    = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + i]);
            // Legacy `transport_persistent` is PWR-pin-cell-only; nuclide
            // index 3 is H-1 in that hardcoded geometry. Pass sab_nuc_idx
            // = 3 so the helper reproduces the prior `ni == 3` gate.
            NuclideMacroXs xs = eval_nuclide_macro_xs(ni, Ni, E, urr_xi,
                                                     /*sab_nuc_idx*/ 3, rank, p,
                                                     /*cell_kT*/ -1.0);
            nuc_t[i] = xs.s_t;
            sum_t   += xs.s_t;
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
            int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + hit_l]);
            double Ni_hit = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + hit_l]);
            double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

            // Reaction-channel breakdown for the chosen nuclide. Same
            // urr_xi as Pass 1 — deterministic, so Σ_x sum back to Σ_t
            // and reaction weights stay normalised.
            NuclideMacroXs hit_xs = eval_nuclide_macro_xs(
                hit_nuc, Ni_hit, E, urr_xi, /*sab_nuc_idx*/ 3, rank, p,
                /*cell_kT*/ -1.0);

            // Sample reaction — order matches CPU: el, inel, n2n, n3n, fis, cap
            double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
            double cum_rxn = 0.0;

            cum_rxn += hit_xs.s_el;
            if (xi_rxn < cum_rxn) {
                // ═══ Elastic scattering ═══

                // Legacy hardcoded cell_kT — kept here only because the
                // pre-recursive `transport_persistent` kernel has no
                // material table to read mat_kT[mat] from. Used by
                // both the SAB slot select and the free-gas branch.
                double cell_kT = (cell==0 && gt==GEOM_PWR) ? 900.0*8.617333262e-5 : 600.0*8.617333262e-5;
                if (gt==GEOM_GODIVA) cell_kT = 294.0*8.617333262e-5;

                // S(α,β) via the per-nuclide slot lookup. The
                // hardcoded `hit_nuc==3` legacy assumption is gone;
                // any nuclide can now carry a TSL.
                if (SCALAR_I(p, P_SAB_N_SLOTS) > 0) {
                    int sab_slot = sab_select_slot(hit_nuc, cell_kT, &rng, p);
                    if (sab_slot >= 0 && E < PTR_D(p, P_SAB_SLOT_EMAX)[sab_slot]) {
                        double E_sab, mu_sab;
                        sab_sample(E, &rng, sab_slot, p, &E_sab, &mu_sab);
                        E = fmax(E_sab, 1e-11);
                        double phi=2.0*PI*pcg_uniform(&rng);
                        rotate_direction(&dx,&dy,&dz,mu_sab,phi);
                        goto end_coll;
                    }
                }

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

            } else if ((cum_rxn+=hit_xs.s_inel), xi_rxn < cum_rxn) {
                // ═══ Inelastic — proper discrete level sampling ═══
                goto do_inelastic;

            } else if ((cum_rxn+=hit_xs.s_n2n), xi_rxn < cum_rxn) {
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

            } else if ((cum_rxn+=hit_xs.s_n3n), xi_rxn < cum_rxn) {
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

            } else if ((cum_rxn+=hit_xs.s_fis), xi_rxn < cum_rxn) {
                // ═══ Fission ═══
                lcnt_fis++;
                // Stage C step D — per-nuclide pointer load.
                int nb_sz=__ldg(&PTR_I(p, P_NB_SIZES)[hit_nuc]);
                double nu;
                if (nb_sz > 0) {
                    const double* nb_e =
                        (const double*) __ldg(&PTR_U64(p, P_NB_E_PTRS)[hit_nuc]);
                    const double* nb_v =
                        (const double*) __ldg(&PTR_U64(p, P_NB_V_PTRS)[hit_nuc]);
                    nu = nu_bar_lookup(E, nb_e, nb_v, 0, nb_sz);
                } else {
                    nu = __ldg(&PTR_D(p, P_NU_BAR_CONST)[hit_nuc]);
                }
                int ns=(int)nu; if(pcg_uniform(&rng)<(nu-(double)ns)) ns++;
                for(int s=0;s<ns;s++){
                    int idx=atomicAdd(fis_count,1);
                    if(idx<max_fis){
                        fis_x[idx]=px; fis_y[idx]=py; fis_z[idx]=pz;
                        fis_e[idx]=sample_fission_emit_energy(E,nu,&rng,p,hit_nuc);
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
                    // Stage C step D — per-nuclide pointer load.
                    const double* cdf_base =
                        (const double*) __ldg(&PTR_U64(p, P_INEL_CDF_PTRS)[hit_nuc]);
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
                    // Stage C step D — per-nuc base pointers hoisted.
                    const double* nuc_lvl_basis =
                        (const double*) __ldg(&PTR_U64(p, P_LEVEL_BASIS_PTRS)[hit_nuc]);
                    const double* nuc_lvl_coeffs =
                        (const double*) __ldg(&PTR_U64(p, P_LEVEL_COEFFS_PTRS)[hit_nuc]);
                    #pragma unroll 1
                    for(int l=0;l<lev_cap;l++){
                        int gl=lv_off+l;
                        if(E>=__ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                           && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])){
                            lxs_sum += svd_reconstruct(
                                &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                                &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
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
                                    &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                                    &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                                    e_idx, rank);
                            }
                            run += lxs;
                            if (xi_l < run) { selected = l; break; }
                        }
                        Q=__ldg(&PTR_D(p, P_LEVEL_Q)[lv_off+selected]);
                    }
                }
                int sel_mt=(n_lev>0)?__ldg(&PTR_I(p, P_LEVEL_MT)[lv_off+selected]):0;
                // Prefer the ENDF MT=91 tabulated outgoing distribution;
                // fall back to Weisskopf evaporation when no table.
                // Matches `sample_inelastic_level` in
                // `physics/collision.rs`. Closes a small portion (~5 %)
                // of inelastic-MT=91 events on Godiva — see
                // transport_recursive.cu for the full investigation note.
                if(sel_mt==91){
                    double ecm_mev=E*A/((A+1.0)*1e6);
                    int n_inc91 = __ldg(&PTR_I(p, P_INEL91_NUC_NINC)[hit_nuc]);
                    double eo_mev;
                    if (n_inc91 > 0) {
                        double eo_ev = sample_inel91_energy(E, &rng, p, hit_nuc);
                        eo_mev = eo_ev / 1.0e6;
                    } else {
                        double a_p=A/8.0;
                        double eex=fmax(ecm_mev,0.1);
                        double T=sqrt(eex/a_p);
                        double x1=fmax(pcg_uniform(&rng),1e-30), x2=fmax(pcg_uniform(&rng),1e-30);
                        eo_mev=-T*log(x1*x2);
                    }
                    eo_mev=fmin(eo_mev, ecm_mev*0.9);
                    Q = -(ecm_mev - eo_mev)*1e6;
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
                        mu_cm = sample_level_angular(E, &rng, p, lv_off + selected, hit_nuc);
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

    // Per-thread atomicAdd. The earlier warp-reduction-then-lane-0
    // path was correct for full warps, but with the persistent kernel's
    // multi-launch compaction the trailing warp is partially populated
    // every launch — the `__activemask`-driven reduction returns the
    // surviving lanes' partial sums to lane 0, but in practice we saw
    // ~26× under-reporting on Godiva (4207 collisions for 50 000
    // particles vs the ~110 000 the recursive kernel sees on the same
    // workload). Direct per-thread atomics dodge the corner case for
    // the cost of a few extra atomics per launch — the counter
    // contention is trivial compared to the XS evaluation work.
    if (lcnt_coll > 0) atomicAdd(cnt_coll, lcnt_coll);
    if (lcnt_fis  > 0) atomicAdd(cnt_fis,  lcnt_fis);
    if (lcnt_leak > 0) atomicAdd(cnt_leak, lcnt_leak);
    if (lcnt_surf > 0) atomicAdd(cnt_surf, lcnt_surf);
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

        // XS lookup — same algebra as transport_persistent, via the
        // streaming helper. Diagnostics that previously read per-nuclide
        // reaction breakdown (trace rows 11-14) re-call the helper for
        // the chosen nuclide below.
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t = 0;
        double nuc_t[MAX_NUC_PER_MAT] = {};
        double urr_xi = pcg_uniform(&rng);

        for (int i = 0; i < n_nuc; i++) {
            int ni    = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + i]);
            NuclideMacroXs xs = eval_nuclide_macro_xs(ni, Ni, E, urr_xi,
                                                     /*sab_nuc_idx*/ 3, rank, p,
                                                     /*cell_kT*/ -1.0);
            nuc_t[i] = xs.s_t;
            sum_t   += xs.s_t;
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
            int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + hit_l]);
            double Ni_hit = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + hit_l]);
            double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

            // Reaction breakdown for the chosen nuclide. Same urr_xi as
            // Pass 1 → Σ_x sum back to nuc_t[hit_l]; reaction weights
            // stay normalised.
            NuclideMacroXs hit_xs = eval_nuclide_macro_xs(
                hit_nuc, Ni_hit, E, urr_xi, /*sab_nuc_idx*/ 3, rank, p,
                /*cell_kT*/ -1.0);

            trace[row+10] = (double)hit_nuc;
            trace[row+11] = hit_xs.s_el  / Ni_hit;
            trace[row+12] = hit_xs.s_inel / Ni_hit;
            trace[row+13] = hit_xs.s_fis / Ni_hit;
            trace[row+14] = hit_xs.s_cap / Ni_hit;

            double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
            double cum_rxn = 0.0;

            cum_rxn += hit_xs.s_el;
            if (xi_rxn < cum_rxn) {
                trace[row+9] = 0.0; // elastic
                int dbg_sab_slot = (SCALAR_I(p, P_SAB_N_SLOTS) > 0)
                    ? PTR_I(p, P_SAB_SLOT_PER_NUC)[hit_nuc] : -1;
                if (dbg_sab_slot >= 0
                    && E < PTR_D(p, P_SAB_SLOT_EMAX)[dbg_sab_slot]) {
                    double E_sab, mu_sab;
                    sab_sample(E, &rng, dbg_sab_slot, p, &E_sab, &mu_sab);
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
            } else if ((cum_rxn+=hit_xs.s_inel), xi_rxn < cum_rxn) {
                trace[row+9] = 1.0; // inelastic
                E = fmax(E * 0.5, 1e-5); // simplified for trace
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=hit_xs.s_n2n), xi_rxn < cum_rxn) {
                trace[row+9] = 2.0;
                E = fmax(E * 0.3, 1e-5);
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=hit_xs.s_n3n), xi_rxn < cum_rxn) {
                trace[row+9] = 3.0;
                E = fmax(E * 0.2, 1e-5);
                double mu=2.0*pcg_uniform(&rng)-1.0, phi=2.0*PI*pcg_uniform(&rng);
                rotate_direction(&dx,&dy,&dz,mu,phi);
            } else if ((cum_rxn+=hit_xs.s_fis), xi_rxn < cum_rxn) {
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
