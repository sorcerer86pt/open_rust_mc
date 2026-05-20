// SPDX-License-Identifier: MIT
// ═══════════════════════════════════════════════════════════════════════
// Stage 3b of task #22: full-physics GPU transport on recursive geometry.
//
// Combines:
//   * Recursive geometry primitives (`gr_find_cell`, `gr_trace_step`) from
//     `geom_recursive.cu`.
//   * Cross-section evaluation (SVD / Pointwise / WMP / S(α,β) / URR) and
//     reaction sampling (elastic, inelastic with discrete-level CDF, n2n,
//     n3n, fission, capture) from `transport.cu` (Params buffer + every
//     `__device__` helper it ships).
//
// Both source files are pre-pended into the same NVRTC translation unit
// by `gpu_recursive::assemble_kernel_source`. The body of the main loop
// mirrors `transport.cu::transport_persistent` event-for-event; only the
// geometry hooks (cell-find, surface trace, material-of-cell, free-gas
// kT) are replaced. RNG, XS evaluation, and collision/scatter sampling
// reuse the same byte-for-byte device code.
// ═══════════════════════════════════════════════════════════════════════

#ifndef TRANSPORT_RECURSIVE_CU
#define TRANSPORT_RECURSIVE_CU

// Effective material at the deepest stack frame, applying RectLattice
// material overrides if present. Renamed to avoid colliding with
// `gr_effective_material` in transport_recursive_const.cu when both are
// concatenated into the same translation unit.
__device__ int tr_effective_material(
    const GrGeometry* g, const GrCoord* stack, int depth,
    const int* lat_override_off, const int* lat_override_count,
    const int* override_lat_idx, const int* override_cell_idx,
    const int* override_mat, int n_lattices)
{
    if (depth <= 0) return -1;
    const GrCoord* d = &stack[depth - 1];
    int cell = d->cell_idx;

    if (d->has_lattice && d->lattice_id < n_lattices) {
        int lid = d->lattice_id;
        int off = lat_override_off[lid];
        if (off >= 0) {
            int cnt = lat_override_count[lid];
            const int* sh = g->lat_shape + lid * 3;
            int lin = d->lat_iz * sh[0] * sh[1] + d->lat_iy * sh[0] + d->lat_ix;
            for (int i = 0; i < cnt; ++i) {
                if (override_lat_idx[off + i] == lin
                    && override_cell_idx[off + i] == cell) {
                    return override_mat[off + i];
                }
            }
        }
    }

    int ft = g->cell_fill_type[cell];
    int fd = g->cell_fill_data[cell];
    if (ft == GR_FILL_MATERIAL) return fd;
    return -1;
}

// One step of full-physics transport on recursive geometry. Each thread
// handles one particle for up to `max_events_per_history` events.
extern "C" __global__ void __launch_bounds__(256, 2)
transport_recursive_persistent(
    Params p,
    // ── Mutable particle state (SoA) ─────────────────────────────────
    double* __restrict__ pos_x, double* __restrict__ pos_y, double* __restrict__ pos_z,
    double* __restrict__ dir_x, double* __restrict__ dir_y, double* __restrict__ dir_z,
    double* __restrict__ energy,
    int*    __restrict__ alive,
    unsigned long long* __restrict__ rng_state_arr,
    unsigned long long* __restrict__ rng_inc_arr,
    int n_particles,
    int max_events_per_history,
    // ── Recursive geometry tables (same layout as const_xs_transport) ─
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
    const double* lat_origin, const double* lat_pitch,
    const int* lat_shape,
    const int* lat_universes_off, const int* lat_universes,
    int n_lattices,
    const double* hex_center, const double* hex_pitch_xy,
    const double* hex_pitch_z,
    const int* hex_n_rings, const int* hex_n_axial,
    const int* hex_orientation,
    const int* hex_universes_off, const int* hex_universes,
    int n_hex_lattices,
    const int* lat_override_off, const int* lat_override_count,
    const int* override_lat_idx, const int* override_cell_idx,
    const int* override_mat,
    // Per-material kT (in eV) — drives the free-gas thermal threshold.
    // Replaces the cell-keyed hard-coded rule in transport.cu.
    const double* mat_kT,
    int n_materials,
    // Index of the nuclide carrying S(α,β) data (e.g. H1 in PWR pin
    // cell), or −1 if no thermal scattering applies.
    int sab_nuc_idx,
    // Per-thread surface-eval scratch (`n_surfaces` doubles per particle).
    double* evals_scratch,
    // ── Fission bank ────────────────────────────────────────────────
    double* __restrict__ fis_x, double* __restrict__ fis_y, double* __restrict__ fis_z,
    double* __restrict__ fis_e, double* __restrict__ fis_w,
    int* __restrict__ fis_count, int max_fis,
    // ── Counters (single-slot atomics) ──────────────────────────────
    int* __restrict__ cnt_coll, int* __restrict__ cnt_fis,
    int* __restrict__ cnt_leak, int* __restrict__ cnt_surf,
    // ── Per-reaction tallies (added for spectrum-hardening diagnosis,
    //    bin/metal_stats_diag). Per-reaction event counts plus
    //    E-in / E-out accumulators so the host can compute
    //    ⟨E_in at fission⟩, ⟨E_in elastic⟩, and the inelastic
    //    energy-loss moment. Mean is taken on the host as
    //    `e_*_sum / cnt_*`. Atomics are double-add (compute capability
    //    ≥ 6.0 = Ampere/RTX A1000 OK).
    int* __restrict__ cnt_elastic,
    int* __restrict__ cnt_inelastic,
    int* __restrict__ cnt_capture,
    double* __restrict__ e_fis_in_sum,
    double* __restrict__ e_el_in_sum,
    double* __restrict__ e_inel_in_sum,
    double* __restrict__ e_inel_out_sum,
    // Squared-energy accumulators (added to localise the metal hot
    // bias to higher moments of the E-at-reaction distribution after
    // ν-table parity was confirmed by `bin/nu_lookup_compare`).
    // Host computes σ(E_at_reaction) = sqrt(⟨E²⟩ − ⟨E⟩²).
    double* __restrict__ e_fis_in_sq_sum,
    double* __restrict__ e_el_in_sq_sum,
    double* __restrict__ e_inel_in_sq_sum,
    // Σ |Q| over inelastic events — `bin/metal_stats_diag` reports
    // ⟨|Q|⟩_inel as the CM-frame energy lost per inelastic event.
    // Used to localise the +400 keV ⟨E_out⟩ gap to per-level XS-weighted
    // sampling (CPU and GPU should converge here if level selection is
    // unbiased).
    double* __restrict__ q_inel_sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    if (!alive[tid]) return;

    int rank = SCALAR_I(p, P_RANK);

    GrGeometry g;
    g.surf_type = surf_type; g.surf_params = surf_params; g.surf_bc = surf_bc;
    g.n_surfaces = n_surfaces;
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
    g.n_lattices = n_lattices;
    g.hex_center = hex_center; g.hex_pitch_xy = hex_pitch_xy;
    g.hex_pitch_z = hex_pitch_z;
    g.hex_n_rings = hex_n_rings; g.hex_n_axial = hex_n_axial;
    g.hex_orientation = hex_orientation;
    g.hex_universes_off = hex_universes_off; g.hex_universes = hex_universes;
    g.n_hex_lattices = n_hex_lattices;
    g.evals = evals_scratch + tid * n_surfaces;

    double px = pos_x[tid], py = pos_y[tid], pz = pos_z[tid];
    double dx = dir_x[tid], dy = dir_y[tid], dz = dir_z[tid];
    double E = energy[tid];
    PcgState rng;
    rng.state = rng_state_arr[tid]; rng.inc = rng_inc_arr[tid];

    GrCoord stack[GR_MAX_DEPTH];
    int depth = gr_find_cell(&g, px, py, pz, stack);
    if (depth == 0) {
        alive[tid] = 0;
        atomicAdd(cnt_leak, 1);
        return;
    }

    int lcnt_coll = 0, lcnt_fis = 0, lcnt_leak = 0, lcnt_surf = 0;
    int lcnt_el = 0, lcnt_inel = 0, lcnt_cap = 0;
    double l_e_fis_in_sum = 0.0;
    double l_e_el_in_sum = 0.0;
    double l_e_inel_in_sum = 0.0;
    double l_e_inel_out_sum = 0.0;
    double l_e_fis_in_sq_sum = 0.0;
    double l_e_el_in_sq_sum = 0.0;
    double l_e_inel_in_sq_sum = 0.0;
    double l_q_inel_sum = 0.0;
    int events = 0;
    int is_alive = 1;

    while (is_alive && events < max_events_per_history) {
        events++;
        int mat = tr_effective_material(
            &g, stack, depth,
            lat_override_off, lat_override_count,
            override_lat_idx, override_cell_idx, override_mat,
            n_lattices);

        // ── Void: free-stream to next surface ────────────────────────
        if (mat < 0) {
            double dist; int surf_idx; int bc; int next_depth;
            GrCoord next_stack[GR_MAX_DEPTH];
            if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                               &dist, &surf_idx, &bc, next_stack, &next_depth)) {
                is_alive = 0; lcnt_leak++; break;
            }
            lcnt_surf++;
            if (bc == GR_BC_VACUUM) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                is_alive = 0; lcnt_leak++; break;
            }
            if (bc == GR_BC_REFLECTIVE) {
                px += dx * dist; py += dy * dist; pz += dz * dist;
                int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
                const double* sp = (surf_idx >= 0) ? surf_params + surf_idx * 8 : nullptr;
                gr_reflect_direction(t, sp, &dx, &dy, &dz);
                continue;
            }
            // transmission
            const double NUDGE = 1e-10;
            px += dx * (dist + NUDGE); py += dy * (dist + NUDGE); pz += dz * (dist + NUDGE);
            if (next_depth == 0) { is_alive = 0; lcnt_leak++; break; }
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
            continue;
        }

        // ── XS evaluation (geometry-agnostic; mirrors transport.cu) ──
        int n_nuc = __ldg(&PTR_I(p, P_MAT_N_NUC)[mat]);
        double sum_t = 0.0;
        double nuc_t[MAX_NUC_PER_MAT] = {};
        double urr_xi = pcg_uniform(&rng);
        // Cell kT (eV) for SAB stochastic-T interpolation in
        // eval_nuclide_macro_xs.
        double xs_cell_kT = (mat >= 0 && mat < n_materials) ? mat_kT[mat] : -1.0;

        for (int i = 0; i < n_nuc; i++) {
            int ni    = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + i]);
            double Ni = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + i]);
            NuclideMacroXs xs = eval_nuclide_macro_xs(ni, Ni, E, urr_xi,
                                                     sab_nuc_idx, rank, p, xs_cell_kT);
            nuc_t[i] = xs.s_t;
            sum_t   += xs.s_t;
        }

        if (sum_t <= 0.0) { is_alive = 0; break; }

        double d_coll = -log(pcg_uniform(&rng)) / sum_t;
        double d_s; int surf_idx; int bc; int next_depth;
        GrCoord next_stack[GR_MAX_DEPTH];
        if (!gr_trace_step(&g, stack, depth, px, py, pz, dx, dy, dz,
                           &d_s, &surf_idx, &bc, next_stack, &next_depth)) {
            is_alive = 0; lcnt_leak++; break;
        }

        if (d_s < d_coll) {
            // ─── Surface crossing ───
            lcnt_surf++;
            if (bc == GR_BC_REFLECTIVE) {
                px += dx * d_s; py += dy * d_s; pz += dz * d_s;
                int t = (surf_idx >= 0) ? surf_type[surf_idx] : -1;
                const double* sp = (surf_idx >= 0) ? surf_params + surf_idx * 8 : nullptr;
                gr_reflect_direction(t, sp, &dx, &dy, &dz);
                continue;
            }
            if (bc == GR_BC_VACUUM) {
                px += dx * d_s; py += dy * d_s; pz += dz * d_s;
                is_alive = 0; lcnt_leak++; break;
            }
            const double NUDGE = 1e-10;
            px += dx * (d_s + NUDGE); py += dy * (d_s + NUDGE); pz += dz * (d_s + NUDGE);
            if (next_depth == 0) { is_alive = 0; lcnt_leak++; break; }
            for (int i = 0; i < next_depth; ++i) stack[i] = next_stack[i];
            depth = next_depth;
            continue;
        }

        // ─── Collision ───
        lcnt_coll++;
        px += dx * d_coll; py += dy * d_coll; pz += dz * d_coll;
        // d_coll < d_s by construction, so the collision is strictly
        // inside the current cell — the stack remains valid. Mirrors
        // the CPU recursive path (no re-resolve after collision) and
        // the const-XS GPU kernel.

        // Sample nuclide
        double xi_nuc = pcg_uniform(&rng) * sum_t;
        double cum = 0.0; int hit_l = 0;
        for (int i = 0; i < n_nuc; i++) {
            cum += nuc_t[i]; if (xi_nuc < cum) { hit_l = i; break; } hit_l = i;
        }
        int hit_nuc = __ldg(&PTR_I(p, P_MAT_NUC_IDX)[mat * MAX_NUC_PER_MAT + hit_l]);
        double Ni_hit = __ldg(&PTR_D(p, P_MAT_ATOM_DENS)[mat * MAX_NUC_PER_MAT + hit_l]);
        double A = __ldg(&PTR_D(p, P_AWR_TABLE)[hit_nuc]);

        // Per-reaction breakdown for the chosen nuclide. Re-uses the
        // urr_xi drawn above so the resampled Σ_x sum back to nuc_t[hit_l]
        // and reaction selection stays unbiased.
        NuclideMacroXs hit_xs = eval_nuclide_macro_xs(
            hit_nuc, Ni_hit, E, urr_xi, sab_nuc_idx, rank, p, xs_cell_kT);

        // Sample reaction — el, inel, n2n, n3n, fis, cap (matches CPU)
        double xi_rxn = pcg_uniform(&rng) * nuc_t[hit_l];
        double cum_rxn = 0.0;
        cum_rxn += hit_xs.s_el;
        if (xi_rxn < cum_rxn) {
            // ═══ Elastic ═══
            lcnt_el++;
            l_e_el_in_sum += E;
            l_e_el_in_sq_sum += E * E;
            // S(α,β) via per-nuclide slot lookup. Multiple TSL-bearing
            // nuclides (e.g. H-in-H₂O + D-in-D₂O) each route to the
            // correct slot — `sab_nuc_idx` is retained as a kernel
            // parameter for ABI stability but no longer consulted.
            if (SCALAR_I(p, P_SAB_N_SLOTS) > 0) {
                // Stochastic-T variant — independent rng draw from
                // the XS-path deterministic select. Mirrors CPU
                // simulate.rs:1080 (separate select_temperature
                // call from line 868). Two independent uniform draws
                // on the kT axis are valid samples from the same
                // interpolation distribution.
                int sab_slot = sab_select_slot(hit_nuc, xs_cell_kT, &rng, p);
                if (sab_slot >= 0 && E < PTR_D(p, P_SAB_SLOT_EMAX)[sab_slot]) {
                    double E_sab, mu_sab;
                    sab_sample(E, &rng, sab_slot, p, &E_sab, &mu_sab);
                    E = fmax(E_sab, 1e-11);
                    double phi = 2.0 * PI * pcg_uniform(&rng);
                    rotate_direction(&dx, &dy, &dz, mu_sab, phi);
                    goto end_coll;
                }
            }
            double cell_kT = (mat >= 0 && mat < n_materials)
                ? mat_kT[mat]
                : 600.0 * 8.617333262e-5;
            if (E < 400.0 * cell_kT) {
                double sigma = sqrt(cell_kT / A), v_n = sqrt(2.0 * E);
                double u1, u2, r_bm, th;
                u1 = pcg_uniform(&rng); u2 = pcg_uniform(&rng);
                r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30))); th = 2.0 * PI * u2;
                double vtx = r_bm * cos(th), vty = r_bm * sin(th);
                u1 = pcg_uniform(&rng); u2 = pcg_uniform(&rng);
                r_bm = sigma * sqrt(-2.0 * log(fmax(u1, 1e-30))); th = 2.0 * PI * u2;
                double vtz = r_bm * cos(th);
                double vnx = dx * v_n, vny = dy * v_n, vnz = dz * v_n;
                double vrx = vnx - vtx, vry = vny - vty, vrz = vnz - vtz;
                double vr = sqrt(vrx * vrx + vry * vry + vrz * vrz);
                if (vr < 1e-20) vr = 1e-20;
                double ia1 = 1.0 / (1.0 + A);
                double vcx = (vnx + A * vtx) * ia1;
                double vcy = (vny + A * vty) * ia1;
                double vcz = (vnz + A * vtz) * ia1;
                double vcn = vr * A * ia1;
                double e_rel = 0.5 * (A / (A + 1.0)) * vr * vr;
                double mu_cm = sample_angular_dist(e_rel, &rng, p, hit_nuc);
                double phi = 2.0 * PI * pcg_uniform(&rng);
                double st = sqrt(fmax(0.0, 1.0 - mu_cm * mu_cm));
                double vrh_x = vrx / vr, vrh_y = vry / vr, vrh_z = vrz / vr;
                double px2, py2, pz2;
                if (fabs(vrh_z) < 0.999) {
                    double ip = 1.0 / sqrt(1.0 - vrh_z * vrh_z);
                    px2 = -vrh_y * ip; py2 = vrh_x * ip; pz2 = 0.0;
                } else {
                    double ip = 1.0 / sqrt(1.0 - vrh_x * vrh_x);
                    px2 = 0.0; py2 = -vrh_z * ip; pz2 = vrh_y * ip;
                }
                double qx = vrh_y * pz2 - vrh_z * py2;
                double qy = vrh_z * px2 - vrh_x * pz2;
                double qz = vrh_x * py2 - vrh_y * px2;
                double s_phi, c_phi; sincos(phi, &s_phi, &c_phi);
                double sx2 = mu_cm * vrh_x + st * (c_phi * px2 + s_phi * qx);
                double sy2 = mu_cm * vrh_y + st * (c_phi * py2 + s_phi * qy);
                double sz2 = mu_cm * vrh_z + st * (c_phi * pz2 + s_phi * qz);
                double vox = vcx + vcn * sx2;
                double voy = vcy + vcn * sy2;
                double voz = vcz + vcn * sz2;
                double vo = sqrt(vox * vox + voy * voy + voz * voz);
                E = 0.5 * vo * vo; if (E < 1e-11) E = 1e-11;
                if (vo > 1e-20) { dx = vox / vo; dy = voy / vo; dz = voz / vo; }
            } else {
                double mu_cm = sample_angular_dist(E, &rng, p, hit_nuc);
                double alpha = ((A - 1.0) / (A + 1.0)) * ((A - 1.0) / (A + 1.0));
                E = E * (1.0 + alpha + (1.0 - alpha) * mu_cm) / 2.0;
                if (E < 1e-11) E = 1e-11;
                double mu_lab = (A > 1.0 + 1e-10)
                    ? (1.0 + A * mu_cm) / sqrt(1.0 + A * A + 2.0 * A * mu_cm)
                    : sqrt(fmax(0.0, (1.0 + mu_cm) * 0.5));
                double phi = 2.0 * PI * pcg_uniform(&rng);
                rotate_direction(&dx, &dy, &dz, mu_lab, phi);
            }
        } else if ((cum_rxn += hit_xs.s_inel), xi_rxn < cum_rxn) {
            // ═══ Inelastic ═══ — handled below at do_inelastic
            goto do_inelastic;

        } else if ((cum_rxn += hit_xs.s_n2n), xi_rxn < cum_rxn) {
            // ═══ (n,2n) — bank 1 extra neutron, primary continues ═══
            { double temp = E / 10.0;
              double x1 = fmax(pcg_uniform(&rng), 1e-30);
              double x2 = fmax(pcg_uniform(&rng), 1e-30);
              double e_sec = fmax(fmin(-temp * log(x1 * x2), E), 1e-5);
              int idx2 = atomicAdd(fis_count, 1);
              if (idx2 < max_fis) {
                  fis_x[idx2] = px; fis_y[idx2] = py; fis_z[idx2] = pz;
                  fis_e[idx2] = e_sec; fis_w[idx2] = 1.0;
              }
            }
            { double Q_n2n = -E * 0.1;
              double e_cm = E * A / (A + 1.0);
              double e_cm_out = e_cm + Q_n2n;
              if (e_cm_out <= 0.0) e_cm_out = E * 0.01;
              double mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
              double ap1 = A + 1.0;
              double e_n = e_cm_out * A / ap1;
              double vni = sqrt(2.0 * e_n);
              double vcs = sqrt(2.0 * E / (ap1 * ap1));
              double v2 = vni * vni + vcs * vcs + 2.0 * vni * vcs * mu_cm;
              E = fmax(0.5 * v2, 1e-5);
              double den = sqrt(fmax(v2, 1e-40));
              double ml = (vni + vcs > 1e-20)
                  ? fmax(-1.0, fmin(1.0, (vcs + vni * mu_cm) / den))
                  : 2.0 * pcg_uniform(&rng) - 1.0;
              double phi = 2.0 * PI * pcg_uniform(&rng);
              rotate_direction(&dx, &dy, &dz, ml, phi);
            }
        } else if ((cum_rxn += hit_xs.s_n3n), xi_rxn < cum_rxn) {
            // ═══ (n,3n) — bank 2 extra neutrons ═══
            for (int ns3 = 0; ns3 < 2; ns3++) {
                double temp = E / 10.0;
                double x1 = fmax(pcg_uniform(&rng), 1e-30);
                double x2 = fmax(pcg_uniform(&rng), 1e-30);
                double e_sec = fmax(fmin(-temp * log(x1 * x2), E), 1e-5);
                int idx2 = atomicAdd(fis_count, 1);
                if (idx2 < max_fis) {
                    fis_x[idx2] = px; fis_y[idx2] = py; fis_z[idx2] = pz;
                    fis_e[idx2] = e_sec; fis_w[idx2] = 1.0;
                }
            }
            { double Q_n3n = -E * 0.2;
              double e_cm = E * A / (A + 1.0);
              double e_cm_out = e_cm + Q_n3n;
              if (e_cm_out <= 0.0) e_cm_out = E * 0.01;
              double mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
              double ap1 = A + 1.0;
              double e_n = e_cm_out * A / ap1;
              double vni = sqrt(2.0 * e_n);
              double vcs = sqrt(2.0 * E / (ap1 * ap1));
              double v2 = vni * vni + vcs * vcs + 2.0 * vni * vcs * mu_cm;
              E = fmax(0.5 * v2, 1e-5);
              double den = sqrt(fmax(v2, 1e-40));
              double ml = (vni + vcs > 1e-20)
                  ? fmax(-1.0, fmin(1.0, (vcs + vni * mu_cm) / den))
                  : 2.0 * pcg_uniform(&rng) - 1.0;
              double phi = 2.0 * PI * pcg_uniform(&rng);
              rotate_direction(&dx, &dy, &dz, ml, phi);
            }
        } else if ((cum_rxn += hit_xs.s_fis), xi_rxn < cum_rxn) {
            // ═══ Fission ═══
            lcnt_fis++;
            l_e_fis_in_sum += E;
            l_e_fis_in_sq_sum += E * E;
            int nb_off = __ldg(&PTR_I(p, P_NB_OFFSETS)[hit_nuc]);
            int nb_sz  = __ldg(&PTR_I(p, P_NB_SIZES)[hit_nuc]);
            double nu = (nb_sz > 0)
                ? nu_bar_lookup(E, PTR_D(p, P_NB_ENERGIES), PTR_D(p, P_NB_VALUES), nb_off, nb_sz)
                : __ldg(&PTR_D(p, P_NU_BAR_CONST)[hit_nuc]);
            int ns = (int)nu;
            if (pcg_uniform(&rng) < (nu - (double)ns)) ns++;
            for (int s = 0; s < ns; s++) {
                int idx = atomicAdd(fis_count, 1);
                if (idx < max_fis) {
                    fis_x[idx] = px; fis_y[idx] = py; fis_z[idx] = pz;
                    fis_e[idx] = sample_fission_emit_energy(E, nu, &rng, p, hit_nuc);
                    fis_w[idx] = 1.0;
                }
            }
            is_alive = 0;
        } else {
            // ═══ Capture (remainder) ═══
            lcnt_cap++;
            is_alive = 0;
            goto end_coll;
        }

        if (0) { do_inelastic:
            // Discrete-level inelastic — same algorithm as transport.cu.
            lcnt_inel++;
            l_e_inel_in_sum += E;
            l_e_inel_in_sq_sum += E * E;
            const double e_inel_pre = E;
            int lv_off = __ldg(&PTR_I(p, P_LEVEL_OFFSETS)[hit_nuc]);
            int n_lev  = __ldg(&PTR_I(p, P_LEVEL_COUNTS)[hit_nuc]);
            double Q = -0.5e6;
            int selected = 0;
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
                double lxs_sum = 0.0;
                int g_off = __ldg(&PTR_I(p, P_GRID_OFFSETS)[hit_nuc]);
                int n_e   = __ldg(&PTR_I(p, P_N_ENERGIES)[hit_nuc]);
                int e_idx = energy_index(&PTR_D(p, P_ENERGY_GRIDS)[g_off], n_e, E);
                int lev_cap = n_lev < LEGACY_LEV_CAP ? n_lev : LEGACY_LEV_CAP;
                // Stage C step D — per-nuc base pointers hoisted.
                const double* nuc_lvl_basis =
                    (const double*) __ldg(&PTR_U64(p, P_LEVEL_BASIS_PTRS)[hit_nuc]);
                const double* nuc_lvl_coeffs =
                    (const double*) __ldg(&PTR_U64(p, P_LEVEL_COEFFS_PTRS)[hit_nuc]);
                #pragma unroll 1
                for (int l = 0; l < lev_cap; l++) {
                    int gl = lv_off + l;
                    if (E >= __ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                        && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) {
                        lxs_sum += svd_reconstruct(
                            &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                            &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                            e_idx, rank);
                    }
                }
                if (lxs_sum > 0.0) {
                    double xi_l = pcg_uniform(&rng) * lxs_sum;
                    double run = 0.0;
                    selected = lev_cap - 1;
                    #pragma unroll 1
                    for (int l = 0; l < lev_cap; l++) {
                        int gl = lv_off + l;
                        double lxs = 0.0;
                        if (E >= __ldg(&PTR_D(p, P_LEVEL_THR)[gl])
                            && __ldg(&PTR_I(p, P_LEVEL_HAS_K)[gl])) {
                            lxs = svd_reconstruct(
                                &nuc_lvl_basis[__ldg(&PTR_I(p, P_LEVEL_BLOCAL_OFF)[gl])],
                                &nuc_lvl_coeffs[__ldg(&PTR_I(p, P_LEVEL_CLOCAL_OFF)[gl])],
                                e_idx, rank);
                        }
                        run += lxs;
                        if (xi_l < run) { selected = l; break; }
                    }
                    Q = __ldg(&PTR_D(p, P_LEVEL_Q)[lv_off + selected]);
                }
            }
            int sel_mt = (n_lev > 0) ? __ldg(&PTR_I(p, P_LEVEL_MT)[lv_off + selected]) : 0;
            if (sel_mt == 91) {
                // Prefer the ENDF MT=91 tabulated outgoing distribution
                // when uploaded. This matches the CPU's
                // `sample_inelastic_level` continuum path. NOTE:
                // experimentally MT=91 only fires for ~5% of inelastic
                // events on Godiva, so this alone does not close the
                // +400 keV ⟨E_out inel⟩ CPU↔GPU gap — but it's the
                // correct algorithm and harmless when the table is
                // present. The remaining gap lives in the discrete-
                // level path; investigation ongoing.
                double ecm_mev = E * A / ((A + 1.0) * 1e6);
                int n_inc91 = __ldg(&PTR_I(p, P_INEL91_NUC_NINC)[hit_nuc]);
                double eo_mev;
                if (n_inc91 > 0) {
                    double eo_ev = sample_inel91_energy(E, &rng, p, hit_nuc);
                    eo_mev = eo_ev / 1.0e6;
                } else {
                    double a_p = A / 8.0;
                    double eex = fmax(ecm_mev, 0.1);
                    double T = sqrt(eex / a_p);
                    double x1 = fmax(pcg_uniform(&rng), 1e-30);
                    double x2 = fmax(pcg_uniform(&rng), 1e-30);
                    eo_mev = -T * log(x1 * x2);
                }
                eo_mev = fmin(eo_mev, ecm_mev * 0.9);
                Q = -(ecm_mev - eo_mev) * 1e6;
            }
            // Accumulate |Q| (CM-frame excitation energy per inelastic
            // event). ⟨|Q|⟩_inel ≈ ⟨ΔE⟩_inel for heavy nuclei (CM ≈ lab).
            l_q_inel_sum += fabs(Q);
            double e_cm = E * A / (A + 1.0);
            double e_cm_out = e_cm + Q;
            if (e_cm_out <= 0.0) {
                double mu_fb = 2.0 * pcg_uniform(&rng) - 1.0;
                double alpha = ((A - 1.0) / (A + 1.0)) * ((A - 1.0) / (A + 1.0));
                E = E * (1.0 + alpha + (1.0 - alpha) * mu_fb) / 2.0;
                if (E < 1e-11) E = 1e-11;
                double mu_lab = (A > 1.0 + 1e-10)
                    ? (1.0 + A * mu_fb) / sqrt(1.0 + A * A + 2.0 * A * mu_fb)
                    : sqrt(fmax(0.0, (1.0 + mu_fb) * 0.5));
                double phi = 2.0 * PI * pcg_uniform(&rng);
                rotate_direction(&dx, &dy, &dz, mu_lab, phi);
            } else {
                double mu_cm;
                if (n_lev > 0 && sel_mt != 91) {
                    mu_cm = sample_level_angular(E, &rng, p, lv_off + selected, hit_nuc);
                } else {
                    mu_cm = 2.0 * pcg_uniform(&rng) - 1.0;
                }
                double ap1 = A + 1.0;
                double e_n_cm = e_cm_out * A / ap1;
                double v_n_i = sqrt(2.0 * e_n_cm);
                double v_cm_s = sqrt(2.0 * E / (ap1 * ap1));
                double v2sum = v_n_i * v_n_i + v_cm_s * v_cm_s + 2.0 * v_n_i * v_cm_s * mu_cm;
                E = fmax(0.5 * v2sum, 1e-5);
                double denom = sqrt(fmax(v2sum, 1e-40));
                double mu_lab;
                if (v_n_i + v_cm_s > 1e-20) {
                    mu_lab = (v_cm_s + v_n_i * mu_cm) / denom;
                    mu_lab = fmax(-1.0, fmin(1.0, mu_lab));
                } else {
                    mu_lab = 2.0 * pcg_uniform(&rng) - 1.0;
                }
                double phi = 2.0 * PI * pcg_uniform(&rng);
                rotate_direction(&dx, &dy, &dz, mu_lab, phi);
            }
            // Tally outgoing E for the inelastic energy-loss moment.
            // `e_inel_pre` snapshots E before the kinematics block; the
            // host computes ⟨ΔE⟩ = (e_inel_in_sum − e_inel_out_sum) / n_inel.
            l_e_inel_out_sum += E;
            (void)e_inel_pre;
        }
        end_coll: ;
    } // while

    pos_x[tid] = px; pos_y[tid] = py; pos_z[tid] = pz;
    dir_x[tid] = dx; dir_y[tid] = dy; dir_z[tid] = dz;
    energy[tid] = E; alive[tid] = is_alive;
    rng_state_arr[tid] = rng.state; rng_inc_arr[tid] = rng.inc;

    // Per-thread atomicAdd. Mirrors the pattern in transport.cu's
    // `transport_persistent`. The earlier warp-reduction-then-lane-0
    // path was correct for full warps but corrupted by partial warps
    // exiting on `alive[tid] = 0` early returns — see the explanatory
    // comment in `gpu_transport.rs:1545`. Per-thread atomicAdd is
    // slightly more contended but exactly correct under any
    // execution mask.
    if (lcnt_coll > 0) atomicAdd(cnt_coll, lcnt_coll);
    if (lcnt_fis  > 0) atomicAdd(cnt_fis,  lcnt_fis);
    if (lcnt_leak > 0) atomicAdd(cnt_leak, lcnt_leak);
    if (lcnt_surf > 0) atomicAdd(cnt_surf, lcnt_surf);
    if (lcnt_el   > 0) atomicAdd(cnt_elastic,   lcnt_el);
    if (lcnt_inel > 0) atomicAdd(cnt_inelastic, lcnt_inel);
    if (lcnt_cap  > 0) atomicAdd(cnt_capture,   lcnt_cap);
    if (l_e_fis_in_sum   != 0.0) atomicAdd(e_fis_in_sum,   l_e_fis_in_sum);
    if (l_e_el_in_sum    != 0.0) atomicAdd(e_el_in_sum,    l_e_el_in_sum);
    if (l_e_inel_in_sum  != 0.0) atomicAdd(e_inel_in_sum,  l_e_inel_in_sum);
    if (l_e_inel_out_sum != 0.0) atomicAdd(e_inel_out_sum, l_e_inel_out_sum);
    if (l_e_fis_in_sq_sum  != 0.0) atomicAdd(e_fis_in_sq_sum,  l_e_fis_in_sq_sum);
    if (l_e_el_in_sq_sum   != 0.0) atomicAdd(e_el_in_sq_sum,   l_e_el_in_sq_sum);
    if (l_e_inel_in_sq_sum != 0.0) atomicAdd(e_inel_in_sq_sum, l_e_inel_in_sq_sum);
    if (l_q_inel_sum       != 0.0) atomicAdd(q_inel_sum,       l_q_inel_sum);
}

#endif // TRANSPORT_RECURSIVE_CU
