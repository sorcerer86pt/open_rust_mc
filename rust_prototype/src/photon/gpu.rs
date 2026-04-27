//! GPU-accelerated photon sampling kernels (feature `cuda`).
//!
//! Mirrors the CPU samplers in `src/photon/{compton,coherent,pair}.rs`:
//!
//!  * `GpuComptonContext` — Klein-Nishina + S(x,Z)/Z bound rejection,
//!    fixed `E_in` per batch.
//!  * `GpuComptonVarECtx` — same, but per-particle `E_in[]`.
//!  * `GpuRayleighContext` — coherent scattering via direct `x²` CDF
//!    inversion + Thomson `(1+μ²)/2` rejection.
//!  * `GpuPairContext` — Bethe-Heitler ε rejection sampling, no
//!    element data needed.
//!
//! All kernels share a PCG-64 implementation that mirrors
//! `src/transport/rng.rs::Rng` byte-for-byte; particle `tid` is seeded
//! via `Rng::for_particle(batch_id, tid)` exactly as on the CPU.
//!
//! Doppler broadening, photoelectric absorption with EADL cascade, and
//! bremsstrahlung secondary emission are *not* yet on GPU — those need
//! larger data-marshalling and (for photoelectric) a thread-divergence
//! tolerant cascade design.

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc;

    use crate::photon::PhotonElement;

    /// All four CUDA kernels share a header (PCG-64, interp helpers).
    /// Keeping them in one PTX module amortises NVRTC compile time.
    const KERNELS_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u32 rotr32(u32 x, u32 r) {
    u32 rm = r & 31u;
    return (x >> rm) | (x << ((32u - rm) & 31u));
}

struct PCG { u64 state; u64 inc; };

__device__ __forceinline__ u32 pcg_next_u32(PCG* r) {
    u64 old_state = r->state;
    r->state = old_state * 6364136223846793005ULL + r->inc;
    u32 xorshifted = (u32)(((old_state >> 18) ^ old_state) >> 27);
    u32 rot = (u32)(old_state >> 59);
    return rotr32(xorshifted, rot);
}

__device__ __forceinline__ double pcg_uniform(PCG* r) {
    u64 a = (u64)(pcg_next_u32(r) >> 5);
    u64 b = (u64)(pcg_next_u32(r) >> 6);
    return (double)(a * 67108864ULL + b) * (1.0 / 9007199254740992.0);
}

__device__ void pcg_for_particle(PCG* r, u64 batch, u64 pid) {
    u64 seed = batch * 6364136223846793005ULL + pid;
    r->inc = (pid << 1) | 1ULL;
    r->state = 0ULL;
    (void)pcg_next_u32(r);
    r->state = r->state + seed;
    (void)pcg_next_u32(r);
}

__device__ double interp_clamp(const double* xg, const double* yg, int n, double xq) {
    if (n == 0) return 0.0;
    if (xq <= xg[0]) return yg[0];
    if (xq >= xg[n - 1]) return yg[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xg[mid] < xq) lo = mid; else hi = mid;
    }
    double t = (xq - xg[lo]) / (xg[hi] - xg[lo]);
    return yg[lo] + t * (yg[hi] - yg[lo]);
}

// Inverse-CDF: yg is monotonic non-decreasing, xg matched-length grid;
// solve for x such that yg(x) = y_target by binary-search + linear interp.
__device__ double invert_cdf(const double* xg, const double* yg, int n, double y) {
    if (n == 0) return 0.0;
    if (y <= yg[0]) return xg[0];
    if (y >= yg[n - 1]) return xg[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (yg[mid] < y) lo = mid; else hi = mid;
    }
    double denom = yg[hi] - yg[lo];
    if (denom <= 0.0) return xg[lo];
    double t = (y - yg[lo]) / denom;
    return xg[lo] + t * (xg[hi] - xg[lo]);
}

// ---- Compton (fixed E_in) -------------------------------------------

extern "C" __global__ void compton_kn_bound_batch(
    const double* __restrict__ sx_x,
    const double* __restrict__ sx_y,
    int sx_n,
    int z,
    double energy_in,
    u64 batch_id,
    double* __restrict__ k_out,
    double* __restrict__ mu_out,
    int*    __restrict__ iters_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double M_E_C2 = 510998.95;
    const double HC = 12398.419843320025;

    double alpha = energy_in / M_E_C2;
    double kappa = 1.0 + 2.0 * alpha;
    double kappa_inv = 1.0 / kappa;
    double kappa_inv_sq = kappa_inv * kappa_inv;
    double a1 = log(kappa);
    double a2 = 0.5 * (1.0 - kappa_inv_sq);
    double p1 = a1 / (a1 + a2);
    double zf = (double)z;
    double hc_inv = energy_in / HC;

    double k = 0.0, mu = 0.0;
    int iter = 0;
    for (; iter < 256; ++iter) {
        double xi_b = pcg_uniform(&rng);
        double xi_s = pcg_uniform(&rng);
        double xi_r = pcg_uniform(&rng);
        if (xi_b < p1) k = exp(-xi_s * a1);
        else           k = sqrt(kappa_inv_sq + xi_s * (1.0 - kappa_inv_sq));
        mu = 1.0 - (1.0 - k) / (alpha * k);
        double x = hc_inv * sqrt(0.5 * (1.0 - mu));
        double s = interp_clamp(sx_x, sx_y, sx_n, x);
        double kn_acc = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
        if (xi_r < kn_acc * (s / zf)) break;
    }
    k_out[tid] = k;
    mu_out[tid] = mu;
    iters_out[tid] = iter + 1;
}

// ---- Compton (per-particle E_in) ------------------------------------

extern "C" __global__ void compton_kn_bound_var_e(
    const double* __restrict__ sx_x,
    const double* __restrict__ sx_y,
    int sx_n,
    int z,
    const double* __restrict__ energy_in_arr,
    u64 batch_id,
    double* __restrict__ k_out,
    double* __restrict__ mu_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double M_E_C2 = 510998.95;
    const double HC = 12398.419843320025;

    double energy_in = energy_in_arr[tid];
    double alpha = energy_in / M_E_C2;
    double kappa = 1.0 + 2.0 * alpha;
    double kappa_inv = 1.0 / kappa;
    double kappa_inv_sq = kappa_inv * kappa_inv;
    double a1 = log(kappa);
    double a2 = 0.5 * (1.0 - kappa_inv_sq);
    double p1 = a1 / (a1 + a2);
    double zf = (double)z;
    double hc_inv = energy_in / HC;

    double k = 0.0, mu = 0.0;
    for (int iter = 0; iter < 256; ++iter) {
        double xi_b = pcg_uniform(&rng);
        double xi_s = pcg_uniform(&rng);
        double xi_r = pcg_uniform(&rng);
        if (xi_b < p1) k = exp(-xi_s * a1);
        else           k = sqrt(kappa_inv_sq + xi_s * (1.0 - kappa_inv_sq));
        mu = 1.0 - (1.0 - k) / (alpha * k);
        double x = hc_inv * sqrt(0.5 * (1.0 - mu));
        double s = interp_clamp(sx_x, sx_y, sx_n, x);
        double kn_acc = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
        if (xi_r < kn_acc * (s / zf)) break;
    }
    k_out[tid] = k;
    mu_out[tid] = mu;
}

// ---- Coherent (Rayleigh) --------------------------------------------
//
// iff_x is the x² grid, iff_y is the cumulative ∫₀ˣ² F²(x',Z) dx'².
// At E_in, x²_max = (E/hc)². CDF_max = interp(iff at x²_max). Each
// rejection iteration draws a uniform on [0, CDF_max], inverts the CDF,
// converts to μ, and accepts with Thomson probability (1+μ²)/2.

extern "C" __global__ void rayleigh_batch(
    const double* __restrict__ iff_x,
    const double* __restrict__ iff_y,
    int iff_n,
    double energy_in,
    u64 batch_id,
    double* __restrict__ mu_out,
    int*    __restrict__ iters_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double HC = 12398.419843320025;
    double lambda = HC / energy_in;
    double x_max = energy_in / HC;
    double x_max_sq = x_max * x_max;
    double cdf_max = interp_clamp(iff_x, iff_y, iff_n, x_max_sq);

    double mu = 0.0;
    int iter = 0;
    for (; iter < 256; ++iter) {
        double xi = pcg_uniform(&rng);
        double target = xi * cdf_max;
        double x_sq = invert_cdf(iff_x, iff_y, iff_n, target);
        if (x_sq > x_max_sq) x_sq = x_max_sq;
        mu = 1.0 - 2.0 * x_sq * lambda * lambda;
        if (mu < -1.0) mu = -1.0;
        if (mu >  1.0) mu =  1.0;
        double accept = 0.5 * (1.0 + mu * mu);
        double xi2 = pcg_uniform(&rng);
        if (xi2 < accept) break;
    }
    mu_out[tid] = mu;
    iters_out[tid] = iter + 1;
}

// ---- Compton with Doppler broadening --------------------------------
//
// Full kernel: free-KN + S(x,Z)/Z bound rejection (as compton_kn_bound_batch),
// then Doppler broadening via per-shell Hartree-Fock Compton profile sampling
// (PENELOPE §2.3.5 / Ribberfors 1975).
//
// Layout:
//   pz[n_pz]                shared momentum grid in atomic units
//   j_flat[n_shells*n_pz]   row-major Jᵢ(pz[k])
//   binding_ev[n_shells]    binding energy in eV
//   n_e_per_shell[n_shells] electron occupancy (drops out of relative weights;
//                           kept for completeness / future weighting changes)
//
// Outputs: E_out (eV), μ.

// Trapezoidal integral ∫₀^p_max J(p) dp on the shared pz grid for shell `s`.
__device__ double cum_profile(
    const double* pz, const double* j_row, int n_pz, double p_max)
{
    if (n_pz == 0 || p_max <= pz[0]) return 0.0;
    double acc = 0.0;
    for (int k = 1; k < n_pz; ++k) {
        if (pz[k] <= p_max) {
            acc += 0.5 * (j_row[k - 1] + j_row[k]) * (pz[k] - pz[k - 1]);
        } else {
            double frac = (p_max - pz[k - 1]) / (pz[k] - pz[k - 1]);
            double j_at = j_row[k - 1] + frac * (j_row[k] - j_row[k - 1]);
            acc += 0.5 * (j_row[k - 1] + j_at) * (p_max - pz[k - 1]);
            break;
        }
    }
    return acc;
}

// Sample |p_z| from J(p) truncated at p_max via inverse-CDF.
__device__ double sample_profile_gpu(
    const double* pz, const double* j_row, int n_pz, double p_max,
    PCG* rng)
{
    double cum_max = cum_profile(pz, j_row, n_pz, p_max);
    if (cum_max <= 0.0) return 0.0;
    double target = pcg_uniform(rng) * cum_max;
    double acc = 0.0;
    for (int k = 1; k < n_pz; ++k) {
        double pk = (pz[k] <= p_max) ? pz[k] : p_max;
        double jk;
        if (pz[k] <= p_max) {
            jk = j_row[k];
        } else {
            double frac = (p_max - pz[k - 1]) / (pz[k] - pz[k - 1]);
            jk = j_row[k - 1] + frac * (j_row[k] - j_row[k - 1]);
        }
        double bin = 0.5 * (j_row[k - 1] + jk) * (pk - pz[k - 1]);
        if (target <= acc + bin) {
            double leftover = target - acc;
            double dp = pk - pz[k - 1];
            double j_lo = j_row[k - 1];
            double m = (jk - j_lo) / fmax(dp, 1e-30);
            if (fabs(m) < 1e-12) {
                return pz[k - 1] + leftover / fmax(j_lo, 1e-30);
            }
            double disc = j_lo * j_lo + 2.0 * m * leftover;
            if (disc < 0.0) return pz[k - 1] + 0.5 * dp;
            double t = (-j_lo + sqrt(disc)) / m;
            if (t < 0.0) t = 0.0;
            if (t > dp)  t = dp;
            return pz[k - 1] + t;
        }
        acc += bin;
        if (pz[k] >= p_max) break;
    }
    double last = pz[n_pz - 1];
    return (p_max < last) ? p_max : last;
}

extern "C" __global__ void compton_doppler_batch(
    const double* __restrict__ sx_x,
    const double* __restrict__ sx_y,
    int sx_n,
    int z,
    const double* __restrict__ pz,
    const double* __restrict__ j_flat,
    const double* __restrict__ binding_ev,
    int n_pz,
    int n_shells,
    double energy_in,
    u64 batch_id,
    double* __restrict__ e_out,
    double* __restrict__ mu_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double M_E_C2 = 510998.95;
    const double HC = 12398.419843320025;
    const double FINE_ALPHA     = 7.297352569300e-3;
    const double INV_FINE_ALPHA = 1.0 / FINE_ALPHA;

    double alpha = energy_in / M_E_C2;
    double kappa = 1.0 + 2.0 * alpha;
    double kappa_inv = 1.0 / kappa;
    double kappa_inv_sq = kappa_inv * kappa_inv;
    double a1 = log(kappa);
    double a2 = 0.5 * (1.0 - kappa_inv_sq);
    double p1 = a1 / (a1 + a2);
    double zf = (double)z;
    double hc_inv = energy_in / HC;

    // ---- KN + bound rejection ----
    double k = 0.0, mu = 0.0;
    for (int it = 0; it < 256; ++it) {
        double xi_b = pcg_uniform(&rng);
        double xi_s = pcg_uniform(&rng);
        double xi_r = pcg_uniform(&rng);
        if (xi_b < p1) k = exp(-xi_s * a1);
        else           k = sqrt(kappa_inv_sq + xi_s * (1.0 - kappa_inv_sq));
        mu = 1.0 - (1.0 - k) / (alpha * k);
        double x = hc_inv * sqrt(0.5 * (1.0 - mu));
        double s = interp_clamp(sx_x, sx_y, sx_n, x);
        double kn_acc = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
        if (xi_r < kn_acc * (s / zf)) break;
    }

    // ---- Doppler ----
    double k_free = k;
    double e_free = energy_in * k_free;
    double e_kin_free = energy_in - e_free;

    if (n_shells == 0) {
        e_out[tid]  = e_free;
        mu_out[tid] = mu;
        return;
    }

    double pz_last = pz[n_pz - 1];

    // Pass 1: total weight over kinematically-accessible shells.
    double total_w = 0.0;
    for (int s = 0; s < n_shells; ++s) {
        double b_ev = binding_ev[s];
        if (b_ev >= e_kin_free) continue;
        double binding_alpha = b_ev / M_E_C2;
        double ap = alpha - binding_alpha;
        double denom_sq = alpha * alpha + ap * ap - 2.0 * alpha * ap * mu;
        if (denom_sq <= 0.0) continue;
        double pmax_mec = (alpha * ap * (1.0 - mu) - binding_alpha) / sqrt(denom_sq);
        double pmax_au = pmax_mec * INV_FINE_ALPHA;
        if (pmax_au < 0.0) pmax_au = 0.0;
        if (pmax_au > pz_last) pmax_au = pz_last;
        double cum_j = cum_profile(pz, &j_flat[s * n_pz], n_pz, pmax_au);
        total_w += cum_j;
    }

    if (total_w <= 0.0) {
        e_out[tid]  = e_free;
        mu_out[tid] = mu;
        return;
    }

    // Up to 32 doppler-rejection redraws.
    bool got = false;
    double e_out_ev = e_free;
    for (int redraw = 0; redraw < 32 && !got; ++redraw) {
        // Pass 2: select shell by weight cumulative.
        double xi = pcg_uniform(&rng) * total_w;
        double cum = 0.0;
        int chosen = -1;
        double pmax_au_chosen = 0.0;
        for (int s = 0; s < n_shells; ++s) {
            double b_ev = binding_ev[s];
            if (b_ev >= e_kin_free) continue;
            double binding_alpha = b_ev / M_E_C2;
            double ap = alpha - binding_alpha;
            double denom_sq = alpha * alpha + ap * ap - 2.0 * alpha * ap * mu;
            if (denom_sq <= 0.0) continue;
            double pmax_mec = (alpha * ap * (1.0 - mu) - binding_alpha) / sqrt(denom_sq);
            double pmax_au = pmax_mec * INV_FINE_ALPHA;
            if (pmax_au < 0.0) pmax_au = 0.0;
            if (pmax_au > pz_last) pmax_au = pz_last;
            double cum_j = cum_profile(pz, &j_flat[s * n_pz], n_pz, pmax_au);
            cum += cum_j;
            if (xi < cum) {
                chosen = s;
                pmax_au_chosen = pmax_au;
                break;
            }
        }
        if (chosen < 0) continue;

        double pz_au = sample_profile_gpu(pz, &j_flat[chosen * n_pz], n_pz, pmax_au_chosen, &rng);
        double sign = (pcg_uniform(&rng) < 0.5) ? -1.0 : 1.0;
        double pz_signed_au = sign * pz_au;
        double t = pz_signed_au * FINE_ALPHA;

        double eps = 1.0 + alpha * (1.0 - mu);
        double alpha_free_root = alpha / eps;
        double a_coef = t * t - eps * eps;
        double b_coef = 2.0 * alpha * (eps - t * t * mu);
        double c_coef = alpha * alpha * (t * t - 1.0);
        double disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
        if (disc < 0.0 || a_coef == 0.0) continue;
        double sqd = sqrt(disc);
        double two_a = 2.0 * a_coef;
        double rp = (-b_coef + sqd) / two_a;
        double rm = (-b_coef - sqd) / two_a;
        double alpha_out;
        bool rp_pos = rp > 0.0;
        bool rm_pos = rm > 0.0;
        if (rp_pos && rm_pos) {
            alpha_out = (fabs(rp - alpha_free_root) <= fabs(rm - alpha_free_root)) ? rp : rm;
        } else if (rp_pos) {
            alpha_out = rp;
        } else if (rm_pos) {
            alpha_out = rm;
        } else {
            continue;
        }
        double e_try = alpha_out * M_E_C2;
        if (e_try <= 0.0 || e_try >= energy_in) continue;
        e_out_ev = e_try;
        got = true;
    }

    e_out[tid]  = e_out_ev;
    mu_out[tid] = mu;
}

// ---- Photoelectric (primary photoelectron only) ---------------------
//
// Phase 1: samples struck subshell from per-shell partial XS at the
// incident photon energy; outputs photoelectron KE = E - B_i. The full
// EADL relaxation cascade (recursive hole stack with variable-length
// fluorescence-photon emission) is *not* on GPU yet — that requires:
//
//   (a) a per-thread fixed-size hole stack and fluorescence buffer
//       (warp-friendly bounded-depth iteration);
//   (b) compaction of variable-length fluorescence outputs across the
//       block before D2H copy;
//   (c) packing the EADL transition table (jagged per-shell) into a
//       flat array with offsets, mirroring the photoelectric XS layout.
//
// Phase 1 already requires the data marshalling work for (c); when the
// cascade kernel ships it slots in next to this one with shared data.
//
// XS layout: per-element padded matrix `pe_xs_flat[n_shells * n_master]`,
// row s gives subshell s's σ_pe,s on the master energy grid (zero below
// its tail-alignment offset). Total size n_shells × n_master × 8 bytes.
// For uranium (≈30 shells × 1200 master pts) ≈ 288 KB — fits in global
// memory comfortably.

extern "C" __global__ void photoelectric_phase1_batch(
    const double* __restrict__ master_e,
    const double* __restrict__ pe_total,
    const double* __restrict__ pe_xs_flat,
    const double* __restrict__ binding_ev,
    int n_master,
    int n_shells,
    double energy_in,
    u64 batch_id,
    double* __restrict__ t_out_ev,
    int*    __restrict__ struck_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);

    // log-log interp of total PE XS at energy_in, on the master grid.
    // (Inline rather than reusing interp_clamp to do log-log.)
    double total_pe;
    if (energy_in <= master_e[0]) {
        total_pe = pe_total[0];
    } else if (energy_in >= master_e[n_master - 1]) {
        total_pe = pe_total[n_master - 1];
    } else {
        int lo = 0, hi = n_master - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (master_e[mid] < energy_in) lo = mid; else hi = mid;
        }
        double e_lo = master_e[lo], e_hi = master_e[hi];
        double y_lo = pe_total[lo], y_hi = pe_total[hi];
        if (y_lo > 0.0 && y_hi > 0.0) {
            double t = (log(energy_in) - log(e_lo)) / (log(e_hi) - log(e_lo));
            total_pe = exp(log(y_lo) + t * (log(y_hi) - log(y_lo)));
        } else {
            double t = (energy_in - e_lo) / (e_hi - e_lo);
            total_pe = y_lo + t * (y_hi - y_lo);
        }
    }
    if (total_pe <= 0.0) {
        t_out_ev[tid] = 0.0;
        struck_out[tid] = -1;
        return;
    }

    // Sample subshell by cumulative partial XS at energy_in.
    double xi = pcg_uniform(&rng) * total_pe;
    int chosen = n_shells - 1;
    double running = 0.0;
    // Find the master-grid bin once outside the shell loop.
    int i_hi;
    if (energy_in <= master_e[0]) {
        i_hi = 1;
    } else if (energy_in >= master_e[n_master - 1]) {
        i_hi = n_master - 1;
    } else {
        int lo = 0, hi = n_master - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (master_e[mid] < energy_in) lo = mid; else hi = mid;
        }
        i_hi = hi;
    }
    int i_lo = i_hi - 1;
    double e_lo_g = master_e[i_lo], e_hi_g = master_e[i_hi];
    double log_e = log(energy_in);
    double log_e_lo = log(e_lo_g);
    double log_e_hi = log(e_hi_g);
    double t_master = (log_e - log_e_lo) / (log_e_hi - log_e_lo);

    for (int s = 0; s < n_shells; ++s) {
        double y_lo = pe_xs_flat[s * n_master + i_lo];
        double y_hi = pe_xs_flat[s * n_master + i_hi];
        double sigma_s;
        if (y_lo <= 0.0 || y_hi <= 0.0) {
            // Fall back to linear when log undefined — typically near
            // the shell edge where padded zeros end.
            sigma_s = y_lo + t_master * (y_hi - y_lo);
            if (sigma_s < 0.0) sigma_s = 0.0;
        } else {
            sigma_s = exp(log(y_lo) + t_master * (log(y_hi) - log(y_lo)));
        }
        running += sigma_s;
        if (running >= xi) {
            chosen = s;
            break;
        }
    }

    double te = energy_in - binding_ev[chosen];
    if (te < 0.0) te = 0.0;
    t_out_ev[tid] = te;
    struck_out[tid] = chosen + 1;  // EADL designator = idx+1
}

// ---- Pair production (Bethe-Heitler) --------------------------------
//
// Threshold 2 m_e c² = 1.022 MeV. ε ∈ [0,1] sampled by rejection from
// f(ε) = ε² + (1−ε)² + (2/3)ε(1−ε) (env. ≤ 1, accept-rate 5/6).
// Outputs T_-, T_+ in eV.

extern "C" __global__ void pair_bh_batch(
    double energy_in,
    u64 batch_id,
    double* __restrict__ te_minus_out,
    double* __restrict__ te_plus_out,
    int*    __restrict__ iters_out,
    int n_particles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    PCG rng; pcg_for_particle(&rng, batch_id, (u64)tid);
    const double M_E_C2 = 510998.95;
    const double THRESH = 2.0 * M_E_C2;

    if (energy_in < THRESH) {
        te_minus_out[tid] = 0.0;
        te_plus_out[tid]  = 0.0;
        iters_out[tid]    = 0;
        return;
    }
    double t_total = energy_in - THRESH;

    double eps = 0.0;
    int iter = 0;
    for (; iter < 256; ++iter) {
        double e = pcg_uniform(&rng);
        double xi = pcg_uniform(&rng);
        double f = e*e + (1.0 - e)*(1.0 - e) + (2.0/3.0) * e * (1.0 - e);
        if (xi < f) { eps = e; break; }
    }
    te_minus_out[tid] = eps * t_total;
    te_plus_out[tid]  = (1.0 - eps) * t_total;
    iters_out[tid]    = iter + 1;
}
"#;

    fn build_module(
        ctx: &Arc<CudaContext>,
    ) -> Result<Arc<cudarc::driver::CudaModule>, Box<dyn std::error::Error>> {
        let ptx = nvrtc::compile_ptx(KERNELS_SRC)?;
        Ok(ctx.load_module(ptx)?)
    }

    // ---- Compton (fixed E) -------------------------------------------

    pub struct GpuComptonBatch {
        pub k: Vec<f64>,
        pub mu: Vec<f64>,
        pub iters: Vec<i32>,
    }

    pub struct GpuComptonContext {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
        d_sx_x: CudaSlice<f64>,
        d_sx_y: CudaSlice<f64>,
        sx_n: i32,
        z: i32,
    }

    impl GpuComptonContext {
        pub fn new(elem: &PhotonElement) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("compton_kn_bound_batch")?;
            let stream = ctx.default_stream();
            let sf = &elem.incoherent_scattering_factor;
            let d_sx_x = stream.clone_htod(&sf.x)?;
            let d_sx_y = stream.clone_htod(&sf.value)?;
            Ok(Self {
                _ctx: ctx,
                stream,
                func,
                d_sx_x,
                d_sx_y,
                sx_n: sf.x.len() as i32,
                z: elem.z as i32,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: f64,
            batch_id: u64,
            n: usize,
        ) -> Result<GpuComptonBatch, Box<dyn std::error::Error>> {
            let mut d_k: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_mu: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_iters: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&self.d_sx_x)
                    .arg(&self.d_sx_y)
                    .arg(&self.sx_n)
                    .arg(&self.z)
                    .arg(&energy_in)
                    .arg(&batch_id)
                    .arg(&mut d_k)
                    .arg(&mut d_mu)
                    .arg(&mut d_iters)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok(GpuComptonBatch {
                k: self.stream.clone_dtoh(&d_k)?,
                mu: self.stream.clone_dtoh(&d_mu)?,
                iters: self.stream.clone_dtoh(&d_iters)?,
            })
        }
    }

    // ---- Compton (variable E) ----------------------------------------

    pub struct GpuComptonVarECtx {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
        d_sx_x: CudaSlice<f64>,
        d_sx_y: CudaSlice<f64>,
        sx_n: i32,
        z: i32,
    }

    impl GpuComptonVarECtx {
        pub fn new(elem: &PhotonElement) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("compton_kn_bound_var_e")?;
            let stream = ctx.default_stream();
            let sf = &elem.incoherent_scattering_factor;
            let d_sx_x = stream.clone_htod(&sf.x)?;
            let d_sx_y = stream.clone_htod(&sf.value)?;
            Ok(Self {
                _ctx: ctx,
                stream,
                func,
                d_sx_x,
                d_sx_y,
                sx_n: sf.x.len() as i32,
                z: elem.z as i32,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: &[f64],
            batch_id: u64,
        ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
            let n = energy_in.len();
            let d_e = self.stream.clone_htod(energy_in)?;
            let mut d_k: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_mu: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&self.d_sx_x)
                    .arg(&self.d_sx_y)
                    .arg(&self.sx_n)
                    .arg(&self.z)
                    .arg(&d_e)
                    .arg(&batch_id)
                    .arg(&mut d_k)
                    .arg(&mut d_mu)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok((
                self.stream.clone_dtoh(&d_k)?,
                self.stream.clone_dtoh(&d_mu)?,
            ))
        }
    }

    // ---- Rayleigh ----------------------------------------------------

    pub struct GpuRayleighBatch {
        pub mu: Vec<f64>,
        pub iters: Vec<i32>,
    }

    pub struct GpuRayleighContext {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
        d_iff_x: CudaSlice<f64>,
        d_iff_y: CudaSlice<f64>,
        iff_n: i32,
    }

    impl GpuRayleighContext {
        pub fn new(elem: &PhotonElement) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("rayleigh_batch")?;
            let stream = ctx.default_stream();
            let iff = &elem.coherent_integrated_form_factor;
            let d_iff_x = stream.clone_htod(&iff.x)?;
            let d_iff_y = stream.clone_htod(&iff.value)?;
            Ok(Self {
                _ctx: ctx,
                stream,
                func,
                d_iff_x,
                d_iff_y,
                iff_n: iff.x.len() as i32,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: f64,
            batch_id: u64,
            n: usize,
        ) -> Result<GpuRayleighBatch, Box<dyn std::error::Error>> {
            let mut d_mu: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_iters: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&self.d_iff_x)
                    .arg(&self.d_iff_y)
                    .arg(&self.iff_n)
                    .arg(&energy_in)
                    .arg(&batch_id)
                    .arg(&mut d_mu)
                    .arg(&mut d_iters)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok(GpuRayleighBatch {
                mu: self.stream.clone_dtoh(&d_mu)?,
                iters: self.stream.clone_dtoh(&d_iters)?,
            })
        }
    }

    // ---- Compton with Doppler ----------------------------------------

    pub struct GpuComptonDopplerBatch {
        pub energy_out: Vec<f64>,
        pub mu: Vec<f64>,
    }

    pub struct GpuComptonDopplerCtx {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
        d_sx_x: CudaSlice<f64>,
        d_sx_y: CudaSlice<f64>,
        d_pz: CudaSlice<f64>,
        d_j_flat: CudaSlice<f64>,
        d_binding: CudaSlice<f64>,
        sx_n: i32,
        n_pz: i32,
        n_shells: i32,
        z: i32,
    }

    impl GpuComptonDopplerCtx {
        pub fn new(elem: &PhotonElement) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("compton_doppler_batch")?;
            let stream = ctx.default_stream();
            let sf = &elem.incoherent_scattering_factor;
            let d_sx_x = stream.clone_htod(&sf.x)?;
            let d_sx_y = stream.clone_htod(&sf.value)?;

            let cp = &elem.compton_profiles;
            let n_pz = cp.pz.len();
            let n_shells = cp.binding_energy.len();
            let mut j_flat = Vec::with_capacity(n_shells * n_pz);
            for s in 0..n_shells {
                let row = &cp.j[s];
                if row.len() != n_pz {
                    return Err(format!(
                        "compton profile row {} has {} pts, expected {}",
                        s,
                        row.len(),
                        n_pz
                    )
                    .into());
                }
                j_flat.extend_from_slice(row);
            }
            let d_pz = stream.clone_htod(&cp.pz)?;
            let d_j_flat = stream.clone_htod(&j_flat)?;
            let d_binding = stream.clone_htod(&cp.binding_energy)?;
            Ok(Self {
                _ctx: ctx,
                stream,
                func,
                d_sx_x,
                d_sx_y,
                d_pz,
                d_j_flat,
                d_binding,
                sx_n: sf.x.len() as i32,
                n_pz: n_pz as i32,
                n_shells: n_shells as i32,
                z: elem.z as i32,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: f64,
            batch_id: u64,
            n: usize,
        ) -> Result<GpuComptonDopplerBatch, Box<dyn std::error::Error>> {
            let mut d_e: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_mu: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&self.d_sx_x)
                    .arg(&self.d_sx_y)
                    .arg(&self.sx_n)
                    .arg(&self.z)
                    .arg(&self.d_pz)
                    .arg(&self.d_j_flat)
                    .arg(&self.d_binding)
                    .arg(&self.n_pz)
                    .arg(&self.n_shells)
                    .arg(&energy_in)
                    .arg(&batch_id)
                    .arg(&mut d_e)
                    .arg(&mut d_mu)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok(GpuComptonDopplerBatch {
                energy_out: self.stream.clone_dtoh(&d_e)?,
                mu: self.stream.clone_dtoh(&d_mu)?,
            })
        }
    }

    // ---- Photoelectric (Phase 1: primary only, no cascade) -----------

    pub struct GpuPhotoelectricBatch {
        /// Photoelectron kinetic energy in eV per particle.
        pub t_e: Vec<f64>,
        /// EADL designator (1-based) of the struck subshell.
        pub struck: Vec<i32>,
    }

    pub struct GpuPhotoelectricCtx {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
        d_master_e: CudaSlice<f64>,
        d_pe_total: CudaSlice<f64>,
        d_pe_xs_flat: CudaSlice<f64>,
        d_binding: CudaSlice<f64>,
        n_master: i32,
        n_shells: i32,
    }

    impl GpuPhotoelectricCtx {
        pub fn new(elem: &PhotonElement) -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("photoelectric_phase1_batch")?;
            let stream = ctx.default_stream();

            let n_master = elem.energy.len();
            let n_shells = elem.subshells.len();
            // Pad each subshell's tail-aligned σ_pe,s into a full master-
            // length row so the kernel can index by master grid bin.
            let mut pe_xs_flat = vec![0.0f64; n_master * n_shells];
            let mut binding = Vec::with_capacity(n_shells);
            for (s, shell) in elem.subshells.iter().enumerate() {
                let len = shell.xs.len();
                let offset = n_master - len;
                let dst = &mut pe_xs_flat[s * n_master + offset..(s + 1) * n_master];
                dst.copy_from_slice(&shell.xs);
                binding.push(shell.binding_energy);
            }

            let d_master_e = stream.clone_htod(&elem.energy)?;
            let d_pe_total = stream.clone_htod(&elem.photoelectric_xs)?;
            let d_pe_xs_flat = stream.clone_htod(&pe_xs_flat)?;
            let d_binding = stream.clone_htod(&binding)?;

            Ok(Self {
                _ctx: ctx,
                stream,
                func,
                d_master_e,
                d_pe_total,
                d_pe_xs_flat,
                d_binding,
                n_master: n_master as i32,
                n_shells: n_shells as i32,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: f64,
            batch_id: u64,
            n: usize,
        ) -> Result<GpuPhotoelectricBatch, Box<dyn std::error::Error>> {
            let mut d_t: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_struck: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&self.d_master_e)
                    .arg(&self.d_pe_total)
                    .arg(&self.d_pe_xs_flat)
                    .arg(&self.d_binding)
                    .arg(&self.n_master)
                    .arg(&self.n_shells)
                    .arg(&energy_in)
                    .arg(&batch_id)
                    .arg(&mut d_t)
                    .arg(&mut d_struck)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok(GpuPhotoelectricBatch {
                t_e: self.stream.clone_dtoh(&d_t)?,
                struck: self.stream.clone_dtoh(&d_struck)?,
            })
        }
    }

    // ---- Pair production ---------------------------------------------

    pub struct GpuPairBatch {
        pub te_minus: Vec<f64>,
        pub te_plus: Vec<f64>,
        pub iters: Vec<i32>,
    }

    pub struct GpuPairContext {
        _ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        func: CudaFunction,
    }

    impl GpuPairContext {
        pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let ctx = CudaContext::new(0)?;
            let module = build_module(&ctx)?;
            let func = module.load_function("pair_bh_batch")?;
            let stream = ctx.default_stream();
            Ok(Self {
                _ctx: ctx,
                stream,
                func,
            })
        }

        pub fn sample_batch(
            &self,
            energy_in: f64,
            batch_id: u64,
            n: usize,
        ) -> Result<GpuPairBatch, Box<dyn std::error::Error>> {
            let mut d_em: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_ep: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
            let mut d_iters: CudaSlice<i32> = self.stream.alloc_zeros(n)?;
            let block: u32 = 256;
            let grid = (n as u32 + block - 1) / block;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&energy_in)
                    .arg(&batch_id)
                    .arg(&mut d_em)
                    .arg(&mut d_ep)
                    .arg(&mut d_iters)
                    .arg(&n_i32)
                    .launch(cfg)?;
            }
            Ok(GpuPairBatch {
                te_minus: self.stream.clone_dtoh(&d_em)?,
                te_plus: self.stream.clone_dtoh(&d_ep)?,
                iters: self.stream.clone_dtoh(&d_iters)?,
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{
    GpuComptonBatch, GpuComptonContext, GpuComptonDopplerBatch, GpuComptonDopplerCtx,
    GpuComptonVarECtx, GpuPairBatch, GpuPairContext, GpuPhotoelectricBatch, GpuPhotoelectricCtx,
    GpuRayleighBatch, GpuRayleighContext,
};
