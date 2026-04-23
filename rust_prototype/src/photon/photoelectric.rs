//! Photoelectric absorption kernel with full EADL relaxation cascade.
//!
//! Photon destroyed, electron ejected with `T_e = E − B_i` where `B_i`
//! is the binding energy of the struck subshell. The resulting hole is
//! filled by an electron from a less-tightly-bound shell, emitting a
//! fluorescence photon (radiative) or an Auger electron (non-radiative)
//! and moving the hole to that shell. The cascade recurses until all
//! holes reach shells with no transition data.
//!
//! # Algorithm
//! 1. Sample subshell `i` with probability `σ_pe,i(E)/σ_pe(E)` using
//!    the element's per-subshell partial cross sections with tail-
//!    alignment on the master energy grid.
//! 2. Photoelectron kinetic energy `T_e = E - B_i`. Deposit locally
//!    under the kerma approximation (no electron transport).
//! 3. Push EADL designator of subshell `i` onto a hole stack.
//! 4. Repeat while holes remain:
//!    - Pop a hole designator; resolve to a `Subshell` via
//!      `PhotonElement::subshell_by_eadl_designator`.
//!    - If unresolved (outside the tabulated PE list), terminate.
//!      Binding energy of valence shells that fall off the list is
//!      <100 eV, small compared to any photon energy of transport
//!      interest; the small accounting error is documented.
//!    - If the shell has no transitions, deposit `B_hole` locally and
//!      continue.
//!    - Sample a transition by its probability column. If the sum
//!      of probabilities is < 1 the hole persists with the residual
//!      probability; treat as local deposit.
//!    - Radiative (`secondary == 0`): emit photon of energy `t[2]`
//!      if above `photon_cutoff`, else deposit locally. Move hole to
//!      `primary`.
//!    - Non-radiative (`secondary != 0`): deposit Auger energy
//!      `t[2]` locally (no electron transport), push holes onto both
//!      `primary` and `secondary`.
//!
//! # References
//! - OpenMC `src/photon.cpp::atomic_relaxation`
//! - PENELOPE-2018 §2.2 (Salvat)
//! - Perkins et al., LLNL EADL (UCRL-50400 vol. 30)

use crate::photon::data::{PhotonElement, Subshell};
use crate::transport::rng::Rng;

/// Photon cutoff below which fluorescence photons are killed and
/// their energy deposited locally. Matches OpenMC's default 1 keV.
pub const DEFAULT_PHOTON_CUTOFF_EV: f64 = 1_000.0;

/// Outcome of a single photoelectric-absorption event, including the
/// full relaxation cascade.
#[derive(Debug, Clone)]
pub struct PhotoelectricOutcome {
    /// Fluorescence photon energies (eV) whose emitted photons should
    /// be added to the photon bank for continued transport. All
    /// entries are above `photon_cutoff`.
    pub fluorescence_photons: Vec<f64>,
    /// Total kinetic energy deposited locally (eV): photoelectron KE +
    /// Auger electron energies + below-cutoff fluorescence + residual
    /// hole binding energies for shells without transition data.
    pub local_deposition: f64,
    /// EADL designator of the initially-struck subshell (K=1, L1=2, ...).
    /// Useful for diagnostics and conditional tallies.
    pub struck_subshell_designator: u32,
}

/// Sample a photoelectric absorption event at incoming photon energy
/// `energy_in` (eV) on the element `elem`. Walks the full EADL
/// relaxation cascade from the struck subshell.
///
/// `photon_cutoff` (eV) is the energy threshold below which emitted
/// fluorescence photons are killed and their energy deposited locally.
/// Use `DEFAULT_PHOTON_CUTOFF_EV` (1 keV) unless a specific deck
/// requires different behaviour.
pub fn photoelectric_absorb(
    elem: &PhotonElement,
    energy_in: f64,
    photon_cutoff: f64,
    rng: &mut Rng,
) -> PhotoelectricOutcome {
    let n_master = elem.n_energy();
    let total_pe = interpolate_log_log(&elem.energy, &elem.photoelectric_xs, energy_in);
    // Sample struck subshell by partial XS.
    let struck_idx = sample_struck_subshell(elem, energy_in, total_pe, n_master, rng);
    let struck = &elem.subshells[struck_idx];
    let struck_designator = (struck_idx as u32) + 1;

    // Photoelectron kinetic energy (kerma — deposit locally).
    let mut local = (energy_in - struck.binding_energy).max(0.0);

    // Relaxation cascade.
    let mut fluorescence = Vec::new();
    let mut hole_stack: Vec<u32> = vec![struck_designator];

    while let Some(hole_designator) = hole_stack.pop() {
        let Some(hole_shell) = elem.subshell_by_eadl_designator(hole_designator) else {
            // Outside tabulated PE list: terminate cascade. Valence
            // binding energies <100 eV introduce a small accounting
            // error that is negligible at photon-transport energies.
            continue;
        };

        if hole_shell.transitions.is_empty() {
            // No decay pathway — deposit the hole's binding energy
            // locally.
            local += hole_shell.binding_energy;
            continue;
        }

        // Sample a transition by its probability column. The sum may
        // be < 1; the deficit means the hole persists indefinitely
        // (deposit B_hole) in a fair interpretation of the data.
        let xi = rng.uniform();
        let mut cdf = 0.0;
        let mut chosen: Option<&[f64; 4]> = None;
        for t in &hole_shell.transitions {
            cdf += t[3];
            if xi < cdf {
                chosen = Some(t);
                break;
            }
        }
        let Some(t) = chosen else {
            // Fell through — hole persists.
            local += hole_shell.binding_energy;
            continue;
        };

        let primary = t[0].round() as u32;
        let secondary = t[1].round() as u32;
        let transition_energy = t[2];

        if secondary == 0 {
            // Radiative fluorescence.
            if transition_energy >= photon_cutoff {
                fluorescence.push(transition_energy);
            } else {
                local += transition_energy;
            }
            hole_stack.push(primary);
        } else {
            // Auger / Coster-Kronig: deposit the Auger electron KE
            // (no electron transport).
            local += transition_energy;
            hole_stack.push(primary);
            hole_stack.push(secondary);
        }
    }

    PhotoelectricOutcome {
        fluorescence_photons: fluorescence,
        local_deposition: local,
        struck_subshell_designator: struck_designator,
    }
}

// --- Subshell sampling ----------------------------------------------------

fn sample_struck_subshell(
    elem: &PhotonElement,
    energy_in: f64,
    total_pe: f64,
    n_master: usize,
    rng: &mut Rng,
) -> usize {
    // Build a "partial cross section at E" for each subshell by
    // log-log interpolation on the master grid with tail alignment.
    // Could precompute per-element cumulative arrays for speed; the
    // per-event cost is O(n_subshells · log n_E) which is small
    // (≤ 30 shells × ~11 bin-search steps) and we avoid a separate
    // lifetime-managed cache.
    let xi = rng.uniform();
    let target = xi * total_pe;

    let mut running = 0.0;
    for (idx, shell) in elem.subshells.iter().enumerate() {
        let sigma_i = interpolate_subshell_xs(&elem.energy, shell, n_master, energy_in);
        running += sigma_i;
        if running >= target {
            return idx;
        }
    }
    // Fall-through (numerical tie at the boundary): last shell.
    elem.subshells.len().saturating_sub(1)
}

/// Log-log linear interpolation of `sigma_pe,i(E)` using the tail-
/// aligned subshell XS array. Returns 0 below the shell's binding
/// tabulation, matching `Subshell::xs_at`.
fn interpolate_subshell_xs(energy: &[f64], shell: &Subshell, n_master: usize, e_query: f64) -> f64 {
    if shell.xs.is_empty() {
        return 0.0;
    }
    let offset = n_master - shell.xs.len();
    // E below the first-tabulated point: return 0.
    if e_query < energy[offset] {
        return 0.0;
    }
    if e_query >= *energy.last().unwrap() {
        return *shell.xs.last().unwrap();
    }
    // Binary search on master grid.
    let i_hi = energy.partition_point(|e| *e < e_query);
    debug_assert!(i_hi >= 1 && i_hi < n_master);
    let i_lo = i_hi - 1;
    // Below the tail-aligned window? Use first tabulated value as floor.
    if i_lo < offset {
        return shell.xs[0];
    }
    let j_lo = i_lo - offset;
    let j_hi = i_hi - offset;
    let e_lo = energy[i_lo];
    let e_hi = energy[i_hi];
    let y_lo = shell.xs[j_lo];
    let y_hi = shell.xs[j_hi];
    if y_lo <= 0.0 || y_hi <= 0.0 {
        // Fallback to linear when a log is undefined (rare edge
        // case near the shell edge where tabulation starts at 0).
        let t = (e_query - e_lo) / (e_hi - e_lo);
        return y_lo + t * (y_hi - y_lo);
    }
    let log_e = e_query.ln();
    let log_e_lo = e_lo.ln();
    let log_e_hi = e_hi.ln();
    let log_y_lo = y_lo.ln();
    let log_y_hi = y_hi.ln();
    let t = (log_e - log_e_lo) / (log_e_hi - log_e_lo);
    (log_y_lo + t * (log_y_hi - log_y_lo)).exp()
}

/// Log-log linear interpolation on a strictly monotonic energy grid.
/// Clamps to endpoints outside the grid.
pub fn interpolate_log_log(grid: &[f64], values: &[f64], query: f64) -> f64 {
    if grid.is_empty() {
        return 0.0;
    }
    if query <= grid[0] {
        return values[0];
    }
    let last = grid.len() - 1;
    if query >= grid[last] {
        return values[last];
    }
    let idx = grid.partition_point(|v| *v < query);
    let e_lo = grid[idx - 1];
    let e_hi = grid[idx];
    let y_lo = values[idx - 1];
    let y_hi = values[idx];
    if y_lo <= 0.0 || y_hi <= 0.0 {
        // Linear fallback when log undefined.
        let t = (query - e_lo) / (e_hi - e_lo);
        return y_lo + t * (y_hi - y_lo);
    }
    let t = (query.ln() - e_lo.ln()) / (e_hi.ln() - e_lo.ln());
    (y_lo.ln() + t * (y_hi.ln() - y_lo.ln())).exp()
}

// --- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn photon_path(name: &str) -> Option<PathBuf> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let p = manifest
            .parent()?
            .join("data/endfb-vii.1-hdf5/photon")
            .join(name);
        if p.exists() { Some(p) } else { None }
    }

    fn load(name: &str) -> Option<PhotonElement> {
        Some(PhotonElement::from_hdf5(&photon_path(name)?).unwrap())
    }

    /// Energy conservation per event: `E_in = Σ fluorescence + local`.
    #[test]
    fn energy_conserved_within_cascade_accuracy() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(0xFEED, 1);
        let energy = 1.0e5; // 100 keV — well above K edge

        for _ in 0..2_000 {
            let out = photoelectric_absorb(&pb, energy, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            let total: f64 = out.fluorescence_photons.iter().sum::<f64>() + out.local_deposition;
            // Accounting accuracy: valence binding energies not on the
            // PE subshell list can be lost to "unresolved" holes. For
            // Pb this is <100 eV vs 100 keV (~0.1 %). Tolerance 1 %.
            let rel_err = (total - energy).abs() / energy;
            assert!(
                rel_err < 1.0e-2,
                "energy violation: in={energy}, out={total}, rel={rel_err}"
            );
        }
    }

    /// All emitted fluorescence photons must be ≥ `photon_cutoff`.
    #[test]
    fn fluorescence_above_cutoff() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(7, 1);
        for _ in 0..1_000 {
            let out = photoelectric_absorb(&pb, 1.0e5, 1_000.0, &mut rng);
            for &e in &out.fluorescence_photons {
                assert!(e >= 1_000.0, "photon {e} below cutoff");
            }
        }
    }

    /// All energies (local and fluorescence) non-negative.
    #[test]
    fn outputs_non_negative() {
        let Some(c) = load("C.h5") else {
            eprintln!("skipping: C.h5 not present");
            return;
        };
        let mut rng = Rng::new(1, 2);
        for _ in 0..2_000 {
            let out = photoelectric_absorb(&c, 1.0e4, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            assert!(out.local_deposition >= 0.0);
            for &e in &out.fluorescence_photons {
                assert!(e >= 0.0);
            }
        }
    }

    /// Above the K-edge on a heavy element the K shell dominates the
    /// photoelectric cross section (~80 % for Pb at 100 keV).
    #[test]
    fn k_shell_dominates_above_k_edge_on_pb() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(13, 3);
        let energy = 1.0e5; // Pb K-edge ≈ 88 keV

        let mut k_count = 0;
        let n = 10_000;
        for _ in 0..n {
            let out = photoelectric_absorb(&pb, energy, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            if out.struck_subshell_designator == 1 {
                k_count += 1;
            }
        }
        let k_frac = k_count as f64 / n as f64;
        assert!(
            k_frac > 0.7,
            "K-shell fraction at 100 keV above K-edge on Pb = {k_frac}, expected > 0.7"
        );
    }

    /// Below the K-edge on Pb (70 keV < 88 keV) the K shell must NOT
    /// be struck — the tail-alignment convention requires `xs = 0` at
    /// energies below the shell's first tabulated point.
    #[test]
    fn k_shell_suppressed_below_k_edge_on_pb() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(17, 5);
        let energy = 7.0e4; // Below Pb K-edge ≈ 88 keV

        let mut k_count = 0;
        let n = 10_000;
        for _ in 0..n {
            let out = photoelectric_absorb(&pb, energy, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            if out.struck_subshell_designator == 1 {
                k_count += 1;
            }
        }
        assert_eq!(
            k_count, 0,
            "K-shell struck below K-edge on Pb ({k_count}/{n} events)"
        );
    }

    /// On a single-subshell atom (H), every event must strike the
    /// K shell and emit no fluorescence (H has no relaxation pathway).
    #[test]
    fn hydrogen_always_strikes_k_with_no_cascade() {
        let Some(h) = load("H.h5") else {
            eprintln!("skipping: H.h5 not present");
            return;
        };
        let mut rng = Rng::new(42, 1);
        for _ in 0..1_000 {
            let out = photoelectric_absorb(&h, 1.0e4, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            assert_eq!(out.struck_subshell_designator, 1);
            assert!(out.fluorescence_photons.is_empty());
            // Local deposition ≈ E - B_K
            // (plus the small B_K addition from the un-decaying K hole
            // with no transitions).
            let expected_min = 1.0e4 - 14.0;
            assert!(out.local_deposition >= expected_min);
        }
    }

    /// Fluorescence yield is significant on heavy elements above the
    /// K-edge (>20 % of K holes produce a K-α or K-β fluorescence
    /// photon for Z > 70). Verify Pb produces fluorescence ≥ 20 %
    /// of the time at 100 keV.
    #[test]
    fn heavy_element_above_k_edge_emits_fluorescence() {
        let Some(pb) = load("Pb.h5") else {
            eprintln!("skipping: Pb.h5 not present");
            return;
        };
        let mut rng = Rng::new(99, 1);
        let mut with_fluor = 0;
        let n = 2_000;
        for _ in 0..n {
            let out = photoelectric_absorb(&pb, 1.0e5, DEFAULT_PHOTON_CUTOFF_EV, &mut rng);
            if !out.fluorescence_photons.is_empty() {
                with_fluor += 1;
            }
        }
        let frac = with_fluor as f64 / n as f64;
        assert!(
            frac > 0.2,
            "Pb fluorescence fraction at 100 keV = {frac}, expected > 0.2"
        );
    }

    mod interp {
        use super::super::*;

        #[test]
        fn log_log_midpoint() {
            // y = x² at x = 1, 4 → values 1, 16. log(y) is linear in
            // log(x), so y at geometric mean (x=2) is exp(log(4)) = 4.
            let grid = vec![1.0, 4.0];
            let vals = vec![1.0, 16.0];
            let y = interpolate_log_log(&grid, &vals, 2.0);
            assert!((y - 4.0).abs() < 1e-10);
        }

        #[test]
        fn log_log_clamps_below() {
            let g = vec![1.0, 2.0];
            let v = vec![10.0, 20.0];
            assert_eq!(interpolate_log_log(&g, &v, 0.5), 10.0);
        }

        #[test]
        fn log_log_clamps_above() {
            let g = vec![1.0, 2.0];
            let v = vec![10.0, 20.0];
            assert_eq!(interpolate_log_log(&g, &v, 100.0), 20.0);
        }

        #[test]
        fn log_log_falls_back_to_linear_for_zero() {
            // If an endpoint is zero, log is undefined; linear fallback.
            let g = vec![0.0, 2.0];
            let v = vec![0.0, 10.0];
            assert_eq!(interpolate_log_log(&g, &v, 1.0), 5.0);
        }
    }
}
