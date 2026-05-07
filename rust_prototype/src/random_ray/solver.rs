//! Random Ray solver — forward + adjoint, multigroup, flat source.
//!
//! Top-level loop:
//!
//! ```text
//! while not converged:
//!   reset per-batch accumulators
//!   for ray in 0..n_rays:
//!     sample (pos, dir) uniform in (mesh AABB, sphere)
//!     ψ_g[..] ← initial guess (e.g. q_g/Σ_t,g of the FSR at birth)
//!     dead-zone: trace Z cm without depositing into accumulators
//!     active-zone: trace D cm, depositing track_psi[f,g] += l·ψ_avg,
//!                                 volume_track[f] += l
//!   φ_f,g = 4π · track_psi[f,g] / volume_track[f]
//!   if k-eigenvalue: update k from fission-source ratio
//!   Q_f,g ← (Σ_g' Σ_s,g'→g · φ_f,g'  +  (1/k) χ_g · Σ_g' νΣ_f,g' · φ_f,g')
//!         (adjoint: Σ_s transposed, χ ↔ νΣ_f swapped)
//!   inactive batches discarded; active batches accumulate run-mean φ.
//! ```
//!
//! v1 deliberate simplifications:
//! - Cartesian voxel FSRs (not cell-based).
//! - Mortal rays (sampled fresh each batch). Immortal rays follow.
//! - Single-thread; rayon parallel is straightforward over rays but
//!   needs split atomic accumulators — left for the GPU/perf pass.

use crate::geometry::coord::CoordStack;
use crate::geometry::ray::{find_cell_recursive, trace_step_recursive};
use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{Geometry, Vec3};
use crate::transport::rng::Rng;

use super::fsr::FsrMesh;
use super::integrator::solve_segment;
use super::mgxs::MgxsLibrary;
use crate::physics_constants::FOUR_PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMode {
    /// k-eigenvalue power iteration.
    Eigenvalue,
    /// Fixed-source: external source `Q_ext` provided by caller; no
    /// k update. Used for adjoint-driven response or simple shielding.
    FixedSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjointFlag {
    Forward,
    Adjoint,
}

#[derive(Debug, Clone)]
pub struct RaySolverConfig {
    /// Total number of rays sampled per batch.
    pub rays_per_batch: usize,
    /// Dead-zone (read-only) length per ray, cm.
    /// Mortal rays only — immortal rays carry ψ across iterations
    /// so dead-zone amortisation is unnecessary.
    pub dead_zone: f64,
    /// Active length per ray, cm.
    pub active_length: f64,
    /// Total batches, including inactive.
    pub batches: usize,
    /// Inactive batches. Active batches accumulate the run-mean.
    pub inactive: usize,
    pub mode: SolverMode,
    pub adjoint: AdjointFlag,
    pub seed: u64,
    /// Immortal rays per Tramm & Siegel 2021. When `true`, the ray
    /// population is initialised once at the start of `run()` and
    /// each ray's `(pos, dir, ψ_g)` is persisted across power
    /// iterations. Vacuum boundaries reflect with ψ zeroed; reflective
    /// boundaries reflect ψ as-is. Skips the dead-zone (rays warm up
    /// during the inactive batches via the source iteration itself).
    pub immortal: bool,
}

impl Default for RaySolverConfig {
    fn default() -> Self {
        Self {
            rays_per_batch: 1000,
            dead_zone: 5.0,
            active_length: 50.0,
            batches: 100,
            inactive: 30,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 1,
            immortal: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolverResult {
    /// FSR-major scalar flux, `phi[fsr * n_groups + g]`.
    pub phi: Vec<f64>,
    pub n_groups: usize,
    pub n_fsrs: usize,
    /// k-eigenvalue (1.0 in fixed-source mode).
    pub k_eff: f64,
    /// Per-batch k history (length = batches), ones in fixed-source mode.
    pub k_history: Vec<f64>,
    pub n_active_batches: usize,
}

impl SolverResult {
    /// Per-FSR scalar flux at group `g`, indexed by the same flat
    /// voxel index the FsrMesh uses (and that `WeightWindow::from_flux`
    /// expects).
    pub fn flux_group(&self, g: usize) -> Vec<f64> {
        debug_assert!(g < self.n_groups);
        let mut out = vec![0.0; self.n_fsrs];
        for f in 0..self.n_fsrs {
            out[f] = self.phi[f * self.n_groups + g];
        }
        out
    }

    /// Group-summed scalar flux per FSR. Useful for FW-CADIS WW
    /// generation when no detector response is being used.
    pub fn flux_total(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_fsrs];
        for f in 0..self.n_fsrs {
            let mut acc = 0.0;
            for g in 0..self.n_groups {
                acc += self.phi[f * self.n_groups + g];
            }
            out[f] = acc;
        }
        out
    }
}

/// Persistent state for a single immortal ray.
///
/// One per ray slot; the population is created once at the start of
/// `run()` (when `cfg.immortal` is true) and the same vector is mutated
/// in place across power iterations. After each iteration, every ray's
/// `(pos, dir, psi)` is the seed for the next iteration's active sweep.
///
/// `stack` is the `find_cell_recursive` coord stack at `pos` — cached
/// to avoid re-resolving the geometry at every batch boundary.
#[derive(Debug, Clone)]
pub struct RayState {
    pub pos: Vec3,
    pub dir: Vec3,
    pub stack: CoordStack,
    /// Per-group angular flux. Length must equal the library's
    /// `n_groups`.
    pub psi: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FsrKind {
    /// FSR backed by a real material in the library — full sweep.
    Material,
    /// FSR resolved as void; rays attenuate trivially (Σ_t → 0
    /// fall-back) but no scalar flux is computed.
    Void,
}

pub struct RandomRaySolver<'g> {
    geom: &'g Geometry,
    mesh: FsrMesh,
    library: MgxsLibrary,
    n_groups: usize,
    /// Per-FSR material id, or u32::MAX for void.
    fsr_material: Vec<u32>,
    fsr_kind: Vec<FsrKind>,
    /// AABB used for ray birth sampling — taken from the FSR mesh.
    sample_aabb: crate::geometry::Aabb,
    /// External source for fixed-source mode, FSR-major: `q_ext[f*n_g + g]`.
    /// Empty in eigenvalue mode.
    q_ext: Vec<f64>,
}

impl<'g> RandomRaySolver<'g> {
    pub fn new(geom: &'g Geometry, mesh: FsrMesh, library: MgxsLibrary) -> Self {
        let n_groups = library.n_groups;
        let n_fsrs = mesh.n_fsrs();
        let mut fsr_material = vec![FsrMesh::VOID; n_fsrs];
        let mut fsr_kind = vec![FsrKind::Void; n_fsrs];
        for f in 0..n_fsrs {
            if mesh.active[f] {
                let mat = mesh.material[f];
                if library.get(mat).is_some() {
                    fsr_material[f] = mat;
                    fsr_kind[f] = FsrKind::Material;
                }
            }
        }
        let sample_aabb = mesh.aabb;
        Self {
            geom,
            mesh,
            library,
            n_groups,
            fsr_material,
            fsr_kind,
            sample_aabb,
            q_ext: Vec::new(),
        }
    }

    /// Set an external isotropic source (fixed-source mode). Length
    /// must equal `n_fsrs * n_groups`.
    pub fn with_external_source(mut self, q_ext: Vec<f64>) -> Self {
        assert_eq!(
            q_ext.len(),
            self.mesh.n_fsrs() * self.n_groups,
            "external source size must match n_fsrs * n_groups"
        );
        self.q_ext = q_ext;
        self
    }

    pub fn n_fsrs(&self) -> usize {
        self.mesh.n_fsrs()
    }

    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    pub fn mesh(&self) -> &FsrMesh {
        &self.mesh
    }

    /// Run the solver.
    pub fn run(&self, cfg: &RaySolverConfig) -> SolverResult {
        let n_fsrs = self.mesh.n_fsrs();
        let n_g = self.n_groups;

        // Initial scalar flux: 1.0 in every active group/FSR. (k power
        // iteration is insensitive to overall scale; flat scattering
        // source seeds the first sweep with something positive.)
        let mut phi = vec![0.0; n_fsrs * n_g];
        for f in 0..n_fsrs {
            if self.fsr_kind[f] == FsrKind::Material {
                for g in 0..n_g {
                    phi[f * n_g + g] = 1.0;
                }
            }
        }

        // q[f*n_g + g] is the per-FSR isotropic source rebuilt each
        // outer iteration from φ.
        let mut q = vec![0.0; n_fsrs * n_g];

        let mut k_eff = 1.0;
        let mut k_history = Vec::with_capacity(cfg.batches);

        // Run-mean accumulator over active batches.
        let mut phi_sum = vec![0.0; n_fsrs * n_g];
        let mut n_active = 0usize;

        // Per-FSR volume estimate. Seeded from the mesh's analytic
        // volume when present; updated each batch from the stochastic
        // track-length sum for FSRs without analytic volume. Used by
        // `fission_integral` to weight contributions by FSR volume.
        let mut volume_estimate = vec![1.0_f64; n_fsrs];
        for f in 0..n_fsrs {
            let v = self.mesh.fsr_volume(f);
            if v > 0.0 {
                volume_estimate[f] = v;
            }
        }

        // Immortal-ray population — initialised lazily on first
        // batch when cfg.immortal is true.
        let mut immortal_states: Vec<RayState> = Vec::new();
        if cfg.immortal {
            immortal_states = self.init_immortal_rays(cfg, &q);
        }

        for batch in 0..cfg.batches {
            // Build q from the current φ (and current k for eigenvalue).
            self.build_source(&phi, k_eff, cfg.adjoint, &mut q);

            // Sweep: produce a fresh per-batch φ. Mortal sweep spawns
            // a fresh ray population every call; immortal sweep mutates
            // the persisted state in place.
            let (phi_batch, volume_track) = if cfg.immortal {
                self.sweep_one_immortal_batch(&q, cfg, batch, &mut immortal_states)
            } else {
                self.sweep_one_batch(&q, cfg, batch)
            };

            // Update volume estimate for FSRs without analytic volume.
            for f in 0..n_fsrs {
                if self.mesh.fsr_volume(f) <= 0.0 && volume_track[f] > 0.0 {
                    volume_estimate[f] = volume_track[f];
                }
            }

            // k update (eigenvalue mode only): integrate fission source
            // before and after this sweep.
            if cfg.mode == SolverMode::Eigenvalue {
                let f_old = self.fission_integral(&phi, &volume_estimate);
                let f_new = self.fission_integral(&phi_batch, &volume_estimate);
                if f_old > 0.0 {
                    k_eff *= f_new / f_old;
                }
            }

            phi = phi_batch;
            k_history.push(k_eff);

            if batch >= cfg.inactive {
                for i in 0..phi.len() {
                    phi_sum[i] += phi[i];
                }
                n_active += 1;
            }
        }

        // Run-mean flux over active batches.
        if n_active > 0 {
            let inv_n = 1.0 / n_active as f64;
            for v in phi_sum.iter_mut() {
                *v *= inv_n;
            }
        } else {
            phi_sum = phi;
        }

        SolverResult {
            phi: phi_sum,
            n_groups: n_g,
            n_fsrs,
            k_eff,
            k_history,
            n_active_batches: n_active,
        }
    }

    /// Build the FSR isotropic source `Q_f,g` from the current scalar
    /// flux φ. Forward and adjoint differ by the scattering matrix
    /// transpose and the χ ↔ νΣ_f swap in the fission term.
    fn build_source(&self, phi: &[f64], k: f64, adjoint: AdjointFlag, q: &mut [f64]) {
        let n_g = self.n_groups;
        let n_fsrs = self.mesh.n_fsrs();
        let inv_k = if k > 0.0 { 1.0 / k } else { 0.0 };
        for f in 0..n_fsrs {
            if self.fsr_kind[f] != FsrKind::Material {
                for g in 0..n_g {
                    q[f * n_g + g] = 0.0;
                }
                continue;
            }
            let mat = self
                .library
                .get(self.fsr_material[f])
                .expect("material exists for material FSR");
            for g in 0..n_g {
                let mut scatter_src = 0.0;
                for gp in 0..n_g {
                    let s = match adjoint {
                        AdjointFlag::Forward => mat.scatter.forward(gp, g),
                        AdjointFlag::Adjoint => mat.scatter.adjoint(gp, g),
                    };
                    scatter_src += s * phi[f * n_g + gp];
                }
                let fission_src = match adjoint {
                    AdjointFlag::Forward => {
                        // χ_g · Σ_g' νΣ_f,g' · φ_g'
                        let mut nfp = 0.0;
                        for gp in 0..n_g {
                            nfp += mat.nu_sigma_f[gp] * phi[f * n_g + gp];
                        }
                        mat.chi[g] * nfp
                    }
                    AdjointFlag::Adjoint => {
                        // νΣ_f,g · Σ_g' χ_g' · ψ*_g'
                        let mut chi_phi = 0.0;
                        for gp in 0..n_g {
                            chi_phi += mat.chi[gp] * phi[f * n_g + gp];
                        }
                        mat.nu_sigma_f[g] * chi_phi
                    }
                };
                let mut total = scatter_src + inv_k * fission_src;
                if !self.q_ext.is_empty() {
                    total += self.q_ext[f * n_g + g];
                }
                q[f * n_g + g] = total;
            }
        }
    }

    /// Volume-weighted fission integral `Σ_f V_f · Σ_g νΣ_f,g · φ_f,g`.
    /// `volume_estimate[f]` is the per-FSR volume (analytic if known,
    /// otherwise the running stochastic estimate).
    fn fission_integral(&self, phi: &[f64], volume_estimate: &[f64]) -> f64 {
        let n_g = self.n_groups;
        let mut acc = 0.0;
        for f in 0..self.mesh.n_fsrs() {
            if self.fsr_kind[f] != FsrKind::Material {
                continue;
            }
            let mat = self.library.get(self.fsr_material[f]).expect("material");
            let mut local = 0.0;
            for g in 0..n_g {
                local += mat.nu_sigma_f[g] * phi[f * n_g + g];
            }
            acc += volume_estimate[f] * local;
        }
        acc
    }

    /// One sweep — sample N rays, integrate each, return
    /// `(phi, volume_track)`. `volume_track[f]` is the sum of all ray
    /// segment lengths in FSR `f` for this batch — the solver uses it
    /// both for flux normalisation (already done here) and to update
    /// the running volume estimate used by `fission_integral`.
    fn sweep_one_batch(
        &self,
        q: &[f64],
        cfg: &RaySolverConfig,
        batch: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_fsrs = self.mesh.n_fsrs();
        let n_g = self.n_groups;
        let mut track_psi = vec![0.0_f64; n_fsrs * n_g];
        let mut volume_track = vec![0.0_f64; n_fsrs];

        for r in 0..cfg.rays_per_batch {
            let mut rng = Rng::new(cfg.seed.wrapping_add(batch as u64), r as u64);
            self.trace_one_ray(q, cfg, &mut rng, &mut track_psi, &mut volume_track);
        }

        // φ_f,g = 4π · track_psi[f,g] / volume_track[f]
        let mut phi = vec![0.0_f64; n_fsrs * n_g];
        for f in 0..n_fsrs {
            if self.fsr_kind[f] != FsrKind::Material {
                continue;
            }
            let v_track = volume_track[f];
            if v_track <= 0.0 {
                // No ray visited this FSR this batch — keep last
                // estimate by leaving phi[f,*] at 0; the run-mean will
                // smooth this out across batches. (For very small N
                // the user should bump rays_per_batch.)
                continue;
            }
            let inv_v = FOUR_PI / v_track;
            for g in 0..n_g {
                phi[f * n_g + g] = track_psi[f * n_g + g] * inv_v;
            }
        }
        (phi, volume_track)
    }

    fn trace_one_ray(
        &self,
        q: &[f64],
        cfg: &RaySolverConfig,
        rng: &mut Rng,
        track_psi: &mut [f64],
        volume_track: &mut [f64],
    ) {
        // Sample ray birth: uniform in mesh AABB, uniform on the sphere.
        let pos = sample_in_aabb(rng, &self.sample_aabb);
        let dir = sample_isotropic(rng);

        let mut stack = match find_cell_recursive(pos, self.geom) {
            Some(s) => s,
            None => return, // birth outside the geometry — skip.
        };

        // Initial ψ guess: per-group source/Σ_t at the birth FSR. Falls
        // back to 0 if the FSR is void.
        let mut psi = vec![0.0_f64; self.n_groups];
        let f0 = match self.mesh.fsr_at(pos, &stack) {
            Some(idx) if self.fsr_kind[idx] == FsrKind::Material => idx,
            _ => usize::MAX,
        };
        if f0 != usize::MAX {
            let mat = self.library.get(self.fsr_material[f0]).expect("material");
            for g in 0..self.n_groups {
                let q_per_sr = q[f0 * self.n_groups + g] / FOUR_PI;
                psi[g] = q_per_sr / mat.sigma_t[g];
            }
        }

        let mut world_pos = pos;
        let mut world_dir = dir;
        let mut active = false;
        let mut traveled_dead = 0.0_f64;
        let mut traveled_active = 0.0_f64;

        // Safety cap on segment count to bound any pathological loop.
        const MAX_SEGMENTS: usize = 100_000;
        for _step in 0..MAX_SEGMENTS {
            // Distance budget for this segment: nearest crossing OR
            // remainder of the dead-zone OR remainder of the active-
            // zone, whichever is smaller.
            let hit = trace_step_recursive(&stack, world_pos, world_dir, self.geom);
            let dist_geo = hit.as_ref().map(|h| h.distance).unwrap_or(f64::INFINITY);

            let dist_dead = if !active {
                cfg.dead_zone - traveled_dead
            } else {
                f64::INFINITY
            };
            let dist_active = if active {
                cfg.active_length - traveled_active
            } else {
                f64::INFINITY
            };
            let dist_phase = dist_dead.min(dist_active);

            let segment_len = dist_geo.min(dist_phase);
            if !segment_len.is_finite() || segment_len <= 0.0 {
                return;
            }

            // Resolve current FSR (deepest cell's voxel).
            let f_idx = match self.mesh.fsr_at(world_pos, &stack) {
                Some(idx) if self.fsr_kind[idx] == FsrKind::Material => Some(idx),
                _ => None,
            };

            // Integrate per-group along the segment.
            if let Some(f) = f_idx {
                let mat = self.library.get(self.fsr_material[f]).expect("material");
                if active {
                    volume_track[f] += segment_len;
                }
                for g in 0..self.n_groups {
                    let q_per_sr = q[f * self.n_groups + g] / FOUR_PI;
                    let r = solve_segment(mat.sigma_t[g], q_per_sr, segment_len, psi[g]);
                    psi[g] = r.psi_out;
                    if active {
                        track_psi[f * self.n_groups + g] += r.track_psi;
                    }
                }
            } else {
                // Void FSR or off-mesh: pure streaming, no source.
                // (Σ_t = 0 effectively; ψ is unchanged.)
            }

            // Advance phase counters; check phase transition.
            if !active {
                traveled_dead += segment_len;
                if traveled_dead >= cfg.dead_zone - 1e-12 {
                    active = true;
                }
            } else {
                traveled_active += segment_len;
                if traveled_active >= cfg.active_length - 1e-12 {
                    return;
                }
            }

            // Was the segment terminated by geometry? If so, advance
            // the world position + handle BC. If not, just advance
            // along the direction.
            if dist_geo <= dist_phase {
                let h = match hit {
                    Some(h) => h,
                    None => return,
                };
                match h.bc {
                    BoundaryCondition::Vacuum => {
                        // Ray escapes — angular flux is zeroed and ray
                        // terminates. (For the immortal-ray variant
                        // we'd reflect with ψ=0; mortal rays just stop.)
                        return;
                    }
                    BoundaryCondition::Reflective => {
                        // Reflect direction about the surface normal.
                        if let Some(s_idx) = h.surface_idx {
                            world_pos = world_pos + world_dir * h.distance;
                            let n = surface_normal_at(&self.geom.surfaces[s_idx], world_pos);
                            let two_dn = 2.0 * world_dir.dot(n);
                            world_dir = world_dir - n * two_dn;
                            // Re-resolve coord stack at the post-reflection
                            // position, nudged a bit along the new direction.
                            world_pos = world_pos + world_dir * 1e-10;
                            match find_cell_recursive(world_pos, self.geom) {
                                Some(s) => stack = s,
                                None => return,
                            }
                        } else {
                            return;
                        }
                    }
                    BoundaryCondition::Transmission => {
                        // Transmission: continue across the surface
                        // into the next cell; trace_step_recursive
                        // already nudged past it.
                        world_pos = world_pos + world_dir * (h.distance + 1e-10);
                        match h.next_stack {
                            Some(s) => stack = s,
                            None => match find_cell_recursive(world_pos, self.geom) {
                                Some(s) => stack = s,
                                None => return,
                            },
                        }
                    }
                }
            } else {
                // Phase budget hit — advance position only.
                world_pos = world_pos + world_dir * segment_len;
            }
        }
    }

    /// Initialise the persistent ray population for an immortal-ray
    /// run. Spawns `cfg.rays_per_batch` rays, each with `(pos, dir)`
    /// uniform in the mesh AABB × sphere, `psi[g]` set to the local
    /// q_g/Σ_t,g of the birth FSR (steady-state ansatz). Rays whose
    /// birth point falls outside the geometry or in a void FSR get a
    /// dummy zero-ψ state at the AABB centroid; the first batch will
    /// see them but they contribute nothing.
    fn init_immortal_rays(&self, cfg: &RaySolverConfig, q: &[f64]) -> Vec<RayState> {
        let mut states = Vec::with_capacity(cfg.rays_per_batch);
        let n_g = self.n_groups;
        for r in 0..cfg.rays_per_batch {
            // Use a distinct sub-stream so init RNG draws don't alias
            // batch-0 sweep RNG draws.
            let mut rng = Rng::new(cfg.seed.wrapping_add(0xA5A5_5A5A), r as u64);
            let pos = sample_in_aabb(&mut rng, &self.sample_aabb);
            let dir = sample_isotropic(&mut rng);
            let mut psi = vec![0.0_f64; n_g];
            let stack = match find_cell_recursive(pos, self.geom) {
                Some(s) => s,
                None => {
                    states.push(RayState {
                        pos,
                        dir,
                        stack: smallvec::smallvec![],
                        psi,
                    });
                    continue;
                }
            };
            if let Some(idx) = self.mesh.fsr_at(pos, &stack) {
                if self.fsr_kind[idx] == FsrKind::Material {
                    let mat = self.library.get(self.fsr_material[idx]).expect("material");
                    for g in 0..n_g {
                        let q_per_sr = q[idx * n_g + g] / FOUR_PI;
                        psi[g] = q_per_sr / mat.sigma_t[g];
                    }
                }
            }
            states.push(RayState {
                pos,
                dir,
                stack,
                psi,
            });
        }
        states
    }

    /// Sweep one batch of immortal rays — same accumulator math as
    /// `sweep_one_batch` but the ray state is mutated in place rather
    /// than re-spawned.
    fn sweep_one_immortal_batch(
        &self,
        q: &[f64],
        cfg: &RaySolverConfig,
        _batch: usize,
        states: &mut [RayState],
    ) -> (Vec<f64>, Vec<f64>) {
        let n_fsrs = self.mesh.n_fsrs();
        let n_g = self.n_groups;
        let mut track_psi = vec![0.0_f64; n_fsrs * n_g];
        let mut volume_track = vec![0.0_f64; n_fsrs];

        for state in states.iter_mut() {
            self.trace_immortal_ray(q, cfg, state, &mut track_psi, &mut volume_track);
        }

        // Same closure as sweep_one_batch.
        let mut phi = vec![0.0_f64; n_fsrs * n_g];
        for f in 0..n_fsrs {
            if self.fsr_kind[f] != FsrKind::Material {
                continue;
            }
            let v_track = volume_track[f];
            if v_track <= 0.0 {
                continue;
            }
            let inv_v = FOUR_PI / v_track;
            for g in 0..n_g {
                phi[f * n_g + g] = track_psi[f * n_g + g] * inv_v;
            }
        }
        (phi, volume_track)
    }

    /// Trace one immortal ray for `cfg.active_length` cm. Identical
    /// segment-stepping logic to the mortal path except:
    ///   - No dead zone — immortal rays carry warm ψ across iterations.
    ///   - Vacuum boundary: ψ is zeroed and direction reflects, ray
    ///     keeps going (Tramm 2021 reflect-with-zero).
    ///   - Final `(pos, dir, psi, stack)` is written back to `state` so
    ///     the next batch starts where this one ended.
    fn trace_immortal_ray(
        &self,
        q: &[f64],
        cfg: &RaySolverConfig,
        state: &mut RayState,
        track_psi: &mut [f64],
        volume_track: &mut [f64],
    ) {
        if state.stack.is_empty() {
            // Ray was orphaned at init — try to recover by reseeding
            // its position to AABB centroid.
            let centroid = Vec3::new(
                0.5 * (self.sample_aabb.min.x + self.sample_aabb.max.x),
                0.5 * (self.sample_aabb.min.y + self.sample_aabb.max.y),
                0.5 * (self.sample_aabb.min.z + self.sample_aabb.max.z),
            );
            match find_cell_recursive(centroid, self.geom) {
                Some(s) => {
                    state.stack = s;
                    state.pos = centroid;
                }
                None => return,
            }
        }

        let mut world_pos = state.pos;
        let mut world_dir = state.dir;
        let mut stack = state.stack.clone();
        let psi = &mut state.psi;
        let mut traveled_active = 0.0_f64;

        const MAX_SEGMENTS: usize = 100_000;
        for _step in 0..MAX_SEGMENTS {
            let hit = trace_step_recursive(&stack, world_pos, world_dir, self.geom);
            let dist_geo = hit.as_ref().map(|h| h.distance).unwrap_or(f64::INFINITY);
            let dist_active = cfg.active_length - traveled_active;

            let segment_len = dist_geo.min(dist_active);
            if !segment_len.is_finite() || segment_len <= 0.0 {
                break;
            }

            let f_idx = match self.mesh.fsr_at(world_pos, &stack) {
                Some(idx) if self.fsr_kind[idx] == FsrKind::Material => Some(idx),
                _ => None,
            };

            if let Some(f) = f_idx {
                let mat = self.library.get(self.fsr_material[f]).expect("material");
                volume_track[f] += segment_len;
                for g in 0..self.n_groups {
                    let q_per_sr = q[f * self.n_groups + g] / FOUR_PI;
                    let r = solve_segment(mat.sigma_t[g], q_per_sr, segment_len, psi[g]);
                    psi[g] = r.psi_out;
                    track_psi[f * self.n_groups + g] += r.track_psi;
                }
            }

            traveled_active += segment_len;
            if traveled_active >= cfg.active_length - 1e-12 {
                world_pos = world_pos + world_dir * segment_len;
                break;
            }

            if dist_geo <= dist_active {
                let h = match hit {
                    Some(h) => h,
                    None => break,
                };
                match h.bc {
                    BoundaryCondition::Vacuum => {
                        // Reflect-with-zero per Tramm 2021. Direction
                        // flips off the surface normal; ψ is zeroed in
                        // every group (no incoming flux from outside).
                        if let Some(s_idx) = h.surface_idx {
                            world_pos = world_pos + world_dir * h.distance;
                            let n = surface_normal_at(&self.geom.surfaces[s_idx], world_pos);
                            let two_dn = 2.0 * world_dir.dot(n);
                            world_dir = world_dir - n * two_dn;
                            for v in psi.iter_mut() {
                                *v = 0.0;
                            }
                            world_pos = world_pos + world_dir * 1e-10;
                            match find_cell_recursive(world_pos, self.geom) {
                                Some(s) => stack = s,
                                None => break,
                            }
                        } else {
                            break;
                        }
                    }
                    BoundaryCondition::Reflective => {
                        if let Some(s_idx) = h.surface_idx {
                            world_pos = world_pos + world_dir * h.distance;
                            let n = surface_normal_at(&self.geom.surfaces[s_idx], world_pos);
                            let two_dn = 2.0 * world_dir.dot(n);
                            world_dir = world_dir - n * two_dn;
                            world_pos = world_pos + world_dir * 1e-10;
                            match find_cell_recursive(world_pos, self.geom) {
                                Some(s) => stack = s,
                                None => break,
                            }
                        } else {
                            break;
                        }
                    }
                    BoundaryCondition::Transmission => {
                        world_pos = world_pos + world_dir * (h.distance + 1e-10);
                        match h.next_stack {
                            Some(s) => stack = s,
                            None => match find_cell_recursive(world_pos, self.geom) {
                                Some(s) => stack = s,
                                None => break,
                            },
                        }
                    }
                }
            } else {
                world_pos = world_pos + world_dir * segment_len;
            }
        }

        state.pos = world_pos;
        state.dir = world_dir;
        state.stack = stack;
        // psi already mutated in place via &mut.
    }
}

#[inline]
fn sample_in_aabb(rng: &mut Rng, aabb: &crate::geometry::Aabb) -> Vec3 {
    let x = aabb.min.x + rng.uniform() * (aabb.max.x - aabb.min.x);
    let y = aabb.min.y + rng.uniform() * (aabb.max.y - aabb.min.y);
    let z = aabb.min.z + rng.uniform() * (aabb.max.z - aabb.min.z);
    Vec3::new(x, y, z)
}

#[inline]
fn sample_isotropic(rng: &mut Rng) -> Vec3 {
    let mu = 2.0 * rng.uniform() - 1.0;
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let s = (1.0 - mu * mu).max(0.0).sqrt();
    Vec3::new(s * phi.cos(), s * phi.sin(), mu)
}

/// Outward-pointing surface normal at world position `p`. Used by the
/// reflective-BC branch.
fn surface_normal_at(surface: &crate::geometry::Surface, p: Vec3) -> Vec3 {
    use crate::geometry::Surface as S;
    match surface {
        S::Plane { normal, .. } => *normal,
        S::PlaneX { .. } => Vec3::new(1.0, 0.0, 0.0),
        S::PlaneY { .. } => Vec3::new(0.0, 1.0, 0.0),
        S::PlaneZ { .. } => Vec3::new(0.0, 0.0, 1.0),
        S::Sphere { center, .. } => (p - *center).normalized(),
        S::CylinderZ {
            center_x, center_y, ..
        } => {
            let dx = p.x - center_x;
            let dy = p.y - center_y;
            let inv = 1.0 / (dx * dx + dy * dy).sqrt();
            Vec3::new(dx * inv, dy * inv, 0.0)
        }
        // Other variants fall back to +z; reflective on cones / arbitrary
        // surfaces is out of scope for v1.
        _ => Vec3::new(0.0, 0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, CellFill, CellId, Region};
    use crate::geometry::surface::BoundaryCondition;
    use crate::geometry::{Aabb, Cell, Surface};
    use crate::random_ray::mgxs::MaterialMgxs;

    fn reflective_unit_cube_geom() -> Geometry {
        let half = 5.0;
        let surfaces = vec![
            Surface::PlaneX {
                x0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: half,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let inside = cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]);
        let outside = Region::Complement(Box::new(cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ])));
        let cells = vec![
            Cell::new(CellId(0), inside, CellFill::Material(0)),
            Cell::new(CellId(1), outside, CellFill::Void),
        ];
        Geometry::flat(surfaces, cells).expect("flat geometry")
    }

    fn one_group_fissionable() -> MaterialMgxs {
        // Σ_t = 1.0, Σ_a = 0.4, Σ_s = 0.6, νΣ_f = 0.5, χ = 1.
        // k_inf = νΣ_f / Σ_a = 0.5 / 0.4 = 1.25.
        MaterialMgxs::fissionable(vec![1.0], vec![0.4], vec![0.5], vec![1.0], vec![0.6])
            .expect("valid 1g material")
    }

    #[test]
    fn one_group_infinite_medium_kinf_matches_analytic() {
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let mesh = FsrMesh::from_geometry(aabb, [4, 4, 4], &geom);
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");
        let solver = RandomRaySolver::new(&geom, mesh, library);

        let cfg = RaySolverConfig {
            rays_per_batch: 800,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 80,
            inactive: 30,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 42,
            immortal: false,
        };
        let r = solver.run(&cfg);
        // Analytic k_inf = 1.25.
        let k_target = 1.25;
        let dk_pcm = (r.k_eff - k_target) / k_target * 1e5;
        assert!(
            dk_pcm.abs() < 500.0,
            "k_eff = {}, target = {}, Δ = {} pcm",
            r.k_eff,
            k_target,
            dk_pcm
        );
    }

    #[test]
    fn fixed_source_infinite_medium_recovers_q_over_sigma_a() {
        // Pure absorber + scatter, no fission, fixed external source.
        // Steady-state φ = Q / Σ_a.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let mesh = FsrMesh::from_geometry(aabb, [2, 2, 2], &geom);
        // Σ_t = 1.0, Σ_a = 0.3, Σ_s = 0.7.
        let mat = MaterialMgxs::nonfissionable(vec![1.0], vec![0.3], vec![0.7]).expect("nonfiss");
        let library = MgxsLibrary::new(vec![mat]).expect("lib");
        let n_fsrs = mesh.n_fsrs();
        let solver =
            RandomRaySolver::new(&geom, mesh, library).with_external_source(vec![1.0; n_fsrs]); // Q = 1 per FSR

        let cfg = RaySolverConfig {
            rays_per_batch: 1000,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 50,
            inactive: 15,
            mode: SolverMode::FixedSource,
            adjoint: AdjointFlag::Forward,
            seed: 7,
            immortal: false,
        };
        let r = solver.run(&cfg);

        // Expected φ = 1 / 0.3 ≈ 3.333.
        let expected = 1.0 / 0.3;
        let phi_avg: f64 = r.phi.iter().sum::<f64>() / r.phi.len() as f64;
        let rel = ((phi_avg - expected) / expected).abs();
        assert!(
            rel < 0.10,
            "φ_avg = {phi_avg}, expected = {expected}, rel = {rel}"
        );
    }

    #[test]
    fn adjoint_kinf_matches_forward_kinf_one_group() {
        // For k-eigenvalue, forward and adjoint share the dominant
        // eigenvalue. Run both and check agreement.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");

        let make_solver = || {
            let mesh = FsrMesh::from_geometry(aabb, [2, 2, 2], &geom);
            RandomRaySolver::new(&geom, mesh, library.clone())
        };

        let mut cfg = RaySolverConfig {
            rays_per_batch: 600,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 60,
            inactive: 20,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 11,
            immortal: false,
        };
        let s_fwd = make_solver();
        let r_fwd = s_fwd.run(&cfg);

        cfg.adjoint = AdjointFlag::Adjoint;
        cfg.seed = 13;
        let s_adj = make_solver();
        let r_adj = s_adj.run(&cfg);

        let dk = (r_fwd.k_eff - r_adj.k_eff).abs() / r_fwd.k_eff * 1e5;
        assert!(
            dk < 800.0,
            "forward k {} vs adjoint k {} differ by {} pcm",
            r_fwd.k_eff,
            r_adj.k_eff,
            dk
        );
    }

    #[test]
    fn immortal_kinf_matches_analytic_one_group() {
        // Same problem as `one_group_infinite_medium_kinf_matches_analytic`
        // but with cfg.immortal = true. Tests:
        //   1. Persistent ray state survives the power iteration loop.
        //   2. Reflective-BC ψ propagation works across batches.
        //   3. Final k_eff lands at analytic 1.25 within tolerance.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let mesh = FsrMesh::from_geometry(aabb, [4, 4, 4], &geom);
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");
        let solver = RandomRaySolver::new(&geom, mesh, library);

        let cfg = RaySolverConfig {
            rays_per_batch: 800,
            dead_zone: 0.0, // immortal → unused
            active_length: 30.0,
            batches: 80,
            inactive: 30,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 17,
            immortal: true,
        };
        let r = solver.run(&cfg);
        let dk_pcm = (r.k_eff - 1.25) / 1.25 * 1e5;
        assert!(
            dk_pcm.abs() < 800.0,
            "immortal k_eff = {}, target = 1.25, Δ = {} pcm",
            r.k_eff,
            dk_pcm
        );
    }

    #[test]
    fn immortal_vacuum_zeroes_psi_on_reflect() {
        // Slab with a vacuum face: in immortal mode, the ψ that crosses
        // the vacuum boundary must be zeroed before the reflected ray
        // re-enters the geometry. We check this indirectly by running
        // a fixed-source slab with a strong source at the back face
        // and a vacuum at the front: the front-side flux should be
        // strictly less than the back-side flux (no spurious recycling
        // of leakage flux back into the problem).
        let thickness = 20.0;
        let half_yz = 50.0;
        let surfaces = vec![
            Surface::PlaneX {
                x0: 0.0,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneX {
                x0: thickness,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: -half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: half_yz,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let inside = cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]);
        let outside = Region::Complement(Box::new(cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ])));
        let cells = vec![
            Cell::new(CellId(0), inside, CellFill::Material(0)),
            Cell::new(CellId(1), outside, CellFill::Void),
        ];
        let geom = Geometry::flat(surfaces, cells).expect("slab");

        let aabb = Aabb::new(
            Vec3::new(0.0, -half_yz, -half_yz),
            Vec3::new(thickness, half_yz, half_yz),
        );
        let n = [10_usize, 1, 1];
        let mesh = FsrMesh::from_geometry(aabb, n, &geom);
        // Σ_t = 0.1, Σ_a = 0.04. Strong source at back voxel only.
        let mat =
            MaterialMgxs::nonfissionable(vec![0.1], vec![0.04], vec![0.06]).expect("water-like");
        let library = MgxsLibrary::new(vec![mat]).expect("lib");
        let mut q_ext = vec![0.0_f64; n[0]];
        q_ext[n[0] - 1] = 10.0;

        let cfg = RaySolverConfig {
            rays_per_batch: 1500,
            dead_zone: 0.0,
            active_length: 60.0,
            batches: 60,
            inactive: 25,
            mode: SolverMode::FixedSource,
            adjoint: AdjointFlag::Forward,
            seed: 23,
            immortal: true,
        };
        let solver = RandomRaySolver::new(&geom, mesh, library).with_external_source(q_ext);
        let r = solver.run(&cfg);
        let phi = r.flux_group(0);
        // Vacuum-side voxel must have less flux than the source-side
        // voxel. If vacuum reflection failed to zero ψ, leakage would
        // recycle back and the gradient could collapse.
        assert!(
            phi[0] < phi[n[0] - 1],
            "vacuum-side phi[0] = {} should be < source-side phi[last] = {}",
            phi[0],
            phi[n[0] - 1]
        );
        // Also verify no NaN / negative flux from runaway state.
        for (i, v) in phi.iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "phi[{i}] = {v}");
        }
    }

    #[test]
    fn cell_based_fsr_kinf_matches_analytic_with_analytic_volume() {
        // Same 1g infinite-medium problem but with a single cell-based
        // FSR. Provides analytic volume so the fission integral has a
        // sane weight from batch 1.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let mut analytic = std::collections::HashMap::new();
        analytic.insert((0_u32, None), 1000.0); // 10×10×10 cube
        let mesh = FsrMesh::cell_based(aabb, &geom, [4, 4, 4], Some(&analytic));
        assert_eq!(mesh.n_fsrs(), 1);
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");
        let solver = RandomRaySolver::new(&geom, mesh, library);

        let cfg = RaySolverConfig {
            rays_per_batch: 800,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 70,
            inactive: 25,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 41,
            immortal: false,
        };
        let r = solver.run(&cfg);
        let dk_pcm = (r.k_eff - 1.25) / 1.25 * 1e5;
        assert!(
            dk_pcm.abs() < 800.0,
            "cell-based k_eff = {}, target = 1.25, Δ = {} pcm",
            r.k_eff,
            dk_pcm
        );
    }

    #[test]
    fn two_group_two_material_cell_based_runs_end_to_end() {
        // Mini pin-cell-style problem: 2 groups, 2 cell-based FSRs
        // (fuel + moderator), reflective box. Doesn't validate against
        // a published reference k_inf — just locks in that the
        // multigroup × multimaterial × cell-based combination produces
        // a physical answer (moderator thermal/fast > fuel thermal/fast,
        // k_eff finite and positive).
        let half = 0.5_f64;
        let surfaces = vec![
            Surface::PlaneX {
                x0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: -half,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: half,
                bc: BoundaryCondition::Reflective,
            },
            // Splits the box into +x (fuel) and -x (moderator) halves.
            Surface::PlaneX {
                x0: 0.0,
                bc: BoundaryCondition::Transmission,
            },
        ];
        let inside_box = cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]);
        let fuel = Region::Intersection(Box::new(inside_box.clone()), Box::new(cell::outside(6)));
        let moderator =
            Region::Intersection(Box::new(inside_box.clone()), Box::new(cell::inside(6)));
        let outside = Region::Complement(Box::new(inside_box));
        let cells = vec![
            Cell::new(CellId(0), fuel, CellFill::Material(0)),
            Cell::new(CellId(1), moderator, CellFill::Material(1)),
            Cell::new(CellId(2), outside, CellFill::Void),
        ];
        let geom = Geometry::flat(surfaces, cells).expect("split-box");

        let aabb = Aabb::new(Vec3::new(-half, -half, -half), Vec3::new(half, half, half));
        let mesh = FsrMesh::cell_based(aabb, &geom, [4, 4, 4], None);
        assert_eq!(mesh.n_fsrs(), 2);

        let fuel_mat = MaterialMgxs::fissionable(
            vec![0.6, 1.0],
            vec![0.05, 0.4],
            vec![0.025, 0.7],
            vec![1.0, 0.0],
            vec![0.5, 0.05, 0.001, 0.55],
        )
        .expect("fuel");
        let mod_mat = MaterialMgxs::nonfissionable(
            vec![1.05, 1.6],
            vec![0.005, 0.01],
            vec![0.6, 0.4, 0.001, 1.5],
        )
        .expect("mod");
        let library = MgxsLibrary::new(vec![fuel_mat, mod_mat]).expect("lib");
        let solver = RandomRaySolver::new(&geom, mesh, library);

        let cfg = RaySolverConfig {
            rays_per_batch: 600,
            dead_zone: 0.5,
            active_length: 8.0,
            batches: 60,
            inactive: 25,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 51,
            immortal: false,
        };
        let r = solver.run(&cfg);
        assert!(
            r.k_eff.is_finite() && r.k_eff > 0.0 && r.k_eff < 5.0,
            "k_eff = {} should be a sane finite positive value",
            r.k_eff
        );
        // Two FSRs, two groups. Find which is fuel (material id 0) and
        // which is moderator (id 1) by checking the underlying mesh.
        // Both per-FSR thermal/fast ratios should be positive and the
        // moderator's should be > the fuel's.
        let phi = &r.phi;
        let r0 = phi[1] / phi[0].max(1e-30);
        let r1 = phi[3] / phi[2].max(1e-30);
        // Whichever FSR has the higher thermal/fast ratio is the
        // moderator side. That's the only direction-agnostic check.
        let max_r = r0.max(r1);
        let min_r = r0.min(r1);
        assert!(
            max_r > 1.05 * min_r,
            "moderator should have >5% higher thermal/fast ratio than fuel: r0={r0}, r1={r1}"
        );
    }

    #[test]
    fn cell_based_fsr_kinf_matches_with_stochastic_volume() {
        // No analytic volume provided — solver must use the stochastic
        // track-length estimate. k_eff should still land near analytic.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let mesh = FsrMesh::cell_based(aabb, &geom, [4, 4, 4], None);
        assert_eq!(mesh.n_fsrs(), 1);
        assert_eq!(mesh.fsr_volume(0), 0.0); // stochastic-only
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");
        let solver = RandomRaySolver::new(&geom, mesh, library);

        let cfg = RaySolverConfig {
            rays_per_batch: 800,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 70,
            inactive: 25,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 43,
            immortal: false,
        };
        let r = solver.run(&cfg);
        let dk_pcm = (r.k_eff - 1.25) / 1.25 * 1e5;
        assert!(
            dk_pcm.abs() < 800.0,
            "cell-based stochastic k_eff = {}, target = 1.25, Δ = {} pcm",
            r.k_eff,
            dk_pcm
        );
    }

    #[test]
    fn immortal_and_mortal_kinf_agree_within_combined_sigma() {
        // Same physics, same statistical bandwidth, both modes: results
        // should agree within combined Monte Carlo σ. This is the
        // weakest sanity check that both pipelines are computing the
        // same underlying eigenvalue.
        let geom = reflective_unit_cube_geom();
        let aabb = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let library = MgxsLibrary::new(vec![one_group_fissionable()]).expect("lib");
        let make_solver = || {
            let mesh = FsrMesh::from_geometry(aabb, [4, 4, 4], &geom);
            RandomRaySolver::new(&geom, mesh, library.clone())
        };

        let cfg_mortal = RaySolverConfig {
            rays_per_batch: 600,
            dead_zone: 3.0,
            active_length: 30.0,
            batches: 70,
            inactive: 25,
            mode: SolverMode::Eigenvalue,
            adjoint: AdjointFlag::Forward,
            seed: 31,
            immortal: false,
        };
        let s_mortal = make_solver();
        let r_mortal = s_mortal.run(&cfg_mortal);

        let mut cfg_immortal = cfg_mortal.clone();
        cfg_immortal.immortal = true;
        cfg_immortal.dead_zone = 0.0;
        cfg_immortal.seed = 37;
        let s_immortal = make_solver();
        let r_immortal = s_immortal.run(&cfg_immortal);

        let dk_pcm = (r_mortal.k_eff - r_immortal.k_eff).abs() / r_mortal.k_eff * 1e5;
        assert!(
            dk_pcm < 1000.0,
            "mortal k = {}, immortal k = {}, Δ = {} pcm",
            r_mortal.k_eff,
            r_immortal.k_eff,
            dk_pcm
        );
    }
}
