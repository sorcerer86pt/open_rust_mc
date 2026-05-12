//! Backend dispatch — one entry point that runs the eigenvalue loop
//! on CPU or CUDA without the binary having to branch on the
//! `cuda` feature flag.
//!
//! The two backends solve the same problem (k-eigenvalue power
//! iteration on a `Geometry` with reflective / vacuum boundaries
//! and a fission-bank source), but they consume different
//! cross-section state:
//!
//!   - CPU: a Rust `XsProvider` trait object (e.g. `SvdXsProvider`,
//!     `TableXsProvider`, `HybridSvdWmpXsProvider`).
//!   - CUDA: a bundle of pre-uploaded device buffers
//!     (`GpuRecursiveContext`, `GpuTransportContext`, `GpuNuclideData`,
//!     `GpuMaterialData`, `GpuSabData`, `GpuWmpData`, `mat_kT`,
//!     `sab_nuc_idx`).
//!
//! The dispatch hides this difference behind the `EigenvalueRunner`
//! trait. Each backend constructs its own runner; binaries call
//! `runner.run(&config)` and consume the result the same way.
//!
//! Geometry construction (via `geometry::shapes::*` helpers, the
//! `Geometry` builder, hex / rect lattices) is backend-agnostic —
//! the same `Geometry` instance feeds both runners.

use crate::transport::particle::FissionBank;
use crate::transport::simulate::{BatchResult, SimConfig};

/// Compile-time/runtime tag for the active backend. Callers can pick
/// `Backend::recommended()` to default to the build's optimal choice
/// (`Cuda` when the `cuda` feature is enabled, `Cpu` otherwise) and
/// override per-binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Backend {
    /// The default for the current build. Caller may still construct
    /// a different `Backend` explicitly (e.g. force CPU even with
    /// `cuda` enabled, for parity tests).
    pub const fn recommended() -> Self {
        #[cfg(feature = "cuda")]
        {
            Backend::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            Backend::Cpu
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Backend::Cuda => "CUDA",
        }
    }
}

/// Outcome of a full eigenvalue run — same shape regardless of
/// backend so binaries can post-process identically.
pub struct EigenvalueOutcome {
    pub batches: Vec<BatchResult>,
    pub k_eff: f64,
    /// Optional final source bank (post-normalize). CPU populates
    /// this only when the SimConfig requested a statepoint write
    /// (otherwise `None`); CUDA can populate it from the last
    /// batch's fission bank.
    pub final_source_bank: Option<FissionBank>,
}

/// A backend-specific eigenvalue driver. Binaries hold one of these
/// and call `run(&config)` to drive the per-batch loop.
pub trait EigenvalueRunner {
    fn backend(&self) -> Backend;

    /// Run the full eigenvalue loop with the given simulation config.
    /// The CPU and CUDA implementations share the public observable —
    /// per-batch results + active-batch mean k_eff. Variance reduction
    /// (survival biasing, weight windows) and tallies (mesh flux,
    /// surface currents) are honoured by each backend that supports
    /// them; backends that don't gracefully ignore them.
    fn run(&self, config: &SimConfig) -> EigenvalueOutcome;
}

/// CPU runner — wraps a geometry + materials + XS provider and
/// drives `simulate::run_eigenvalue_with_geometry` under the hood.
pub struct CpuRunner<'a, XS: crate::transport::simulate::XsProvider> {
    pub geometry: &'a crate::geometry::Geometry,
    pub materials: &'a [crate::transport::material::Material],
    pub xs_provider: &'a XS,
}

impl<'a, XS: crate::transport::simulate::XsProvider> EigenvalueRunner for CpuRunner<'a, XS> {
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn run(&self, config: &SimConfig) -> EigenvalueOutcome {
        let (batches, k_eff) = crate::transport::simulate::run_eigenvalue_with_geometry(
            config,
            self.geometry,
            self.materials,
            self.xs_provider,
        );
        EigenvalueOutcome {
            batches,
            k_eff,
            final_source_bank: None,
        }
    }
}

/// CUDA runner — same observable shape as `CpuRunner`. Driven by
/// `GpuRecursiveContext::transport_recursive` per batch.
///
/// Note: this runner is only available with `--features cuda`. The
/// per-batch source-bank management, k_eff active-mean aggregation,
/// and statepoint hook all live here so binaries don't reproduce
/// that loop themselves.
#[cfg(feature = "cuda")]
pub struct CudaRunner<'a> {
    pub recursive: &'a crate::gpu_recursive::GpuRecursiveContext,
    pub transport: &'a crate::gpu_transport::GpuTransportContext,
    pub nuc_data: &'a crate::gpu_transport::GpuNuclideData,
    pub mat_data: &'a crate::gpu_transport::GpuMaterialData,
    pub sab_data: &'a crate::gpu_transport::GpuSabData,
    pub wmp_data: &'a crate::gpu_transport::GpuWmpData,
    pub mat_k_t: &'a [f64],
    pub sab_nuc_idx: i32,
    pub max_events_per_history: i32,
    pub fis_capacity: usize,
    /// Closure that builds the batch-1 source bank when no restart
    /// state is provided. Default: rejection-sample uniform inside
    /// the geometry's fissile cells (caller supplies this via
    /// `crate::transport::simulate::initial_source` or equivalent).
    pub initial_source: Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)> + 'a>,
}

#[cfg(feature = "cuda")]
impl<'a> EigenvalueRunner for CudaRunner<'a> {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn run(&self, config: &SimConfig) -> EigenvalueOutcome {
        use crate::transport::particle::FissionSite;
        use rust_mc_sim::Pcg64;

        let n = config.particles_per_batch as usize;
        let mut source = (self.initial_source)(n, config.seed);
        let mut batches: Vec<BatchResult> = Vec::with_capacity(config.batches as usize);
        let mut k_sum = 0.0_f64;
        let mut k_count = 0_u32;

        for batch in 1..=config.batches {
            let batch_seed = config.seed * 100_000 + batch as u64 * 1_000;
            let rng_seeds: Vec<(u64, u64)> = (0..n)
                .map(|i| {
                    let p = Pcg64::for_particle(batch_seed, i as u64);
                    (p.state(), p.stream())
                })
                .collect();

            let result = self
                .recursive
                .transport_recursive(
                    self.transport,
                    self.nuc_data,
                    self.mat_data,
                    self.sab_data,
                    self.wmp_data,
                    &source,
                    &rng_seeds,
                    self.mat_k_t,
                    self.sab_nuc_idx,
                    self.max_events_per_history,
                    self.fis_capacity,
                )
                .expect("transport_recursive failed");

            let active = batch > config.inactive;
            if active {
                k_sum += result.k_eff;
                k_count += 1;
            }
            batches.push(BatchResult {
                batch,
                k_eff: result.k_eff,
                leakage: result.n_leakage as u32,
                // Codebase convention: BatchResult.absorptions = capture
                // events only (matches CPU semantics in
                // simulate.rs::dispatch_real_collision; the flux estimator
                // in depletion/flux.rs:62 also assumes captures and
                // fissions are counted separately). The GPU emits a
                // capture counter as `n_capture`; wire that through.
                // The "OpenMC-style absorption" = captures + fissions
                // is reconstructed in bin/metal_stats_diag from this plus
                // BatchResult.fissions.
                absorptions: result.n_capture as u32,
                fissions: result.n_fissions as u32,
                collisions: result.n_collisions as u32,
                thermal_scatters: 0,
                // GPU's transport_recursive_persistent does tally `cnt_surf` —
                // wire it through so diagnostics (e.g. `bin/metal_stats_diag`)
                // can compare against the CPU surface_crossings count.
                surface_crossings: result.n_surf_xings as u32,
                shannon_entropy: 0.0,
                active,
                captures_by_cell: vec![],
                photon_events: vec![],
                k_track: 0.0,
                tallies: crate::transport::tally::BatchTallies::default(),
                // Spectrum-hardening diagnostic — GPU populates these
                // so `bin/metal_stats_diag` can compute ⟨E_in at
                // fission⟩, ⟨E_in elastic⟩, and the inelastic energy-
                // loss moment for the CPU↔GPU↔OpenMC 3-way.
                n_elastic: result.n_elastic,
                n_inelastic: result.n_inelastic,
                n_capture: result.n_capture,
                e_fis_in_sum: result.e_fis_in_sum,
                e_el_in_sum: result.e_el_in_sum,
                e_inel_in_sum: result.e_inel_in_sum,
                e_inel_out_sum: result.e_inel_out_sum,
                e_fis_in_sq_sum: result.e_fis_in_sq_sum,
                e_el_in_sq_sum: result.e_el_in_sq_sum,
                e_inel_in_sq_sum: result.e_inel_in_sq_sum,
                q_inel_sum: result.q_inel_sum,
            });

            // Normalize fission bank → next-batch source.
            source = normalize_gpu_bank(&result.fission_bank, n, batch_seed);
        }

        let k_eff = if k_count > 0 {
            k_sum / k_count as f64
        } else {
            0.0
        };

        // Final fission bank for restart support.
        let final_bank: Vec<FissionSite> = source
            .iter()
            .map(|&(x, y, z, e)| FissionSite {
                pos: crate::geometry::Vec3::new(x, y, z),
                energy: e,
                weight: 1.0,
            })
            .collect();
        let mut fb = FissionBank::new();
        fb.sites = final_bank;

        EigenvalueOutcome {
            batches,
            k_eff,
            final_source_bank: Some(fb),
        }
    }
}

/// Resample-with-replacement from a fission bank to N particles for
/// the next batch's source. Mirrors the CPU
/// `simulate::normalize_fission_bank` but takes the GPU's
/// `(x, y, z, energy)` tuple format.
#[cfg(feature = "cuda")]
fn normalize_gpu_bank(
    bank: &[(f64, f64, f64, f64)],
    n: usize,
    batch_seed: u64,
) -> Vec<(f64, f64, f64, f64)> {
    if bank.is_empty() {
        return vec![(0.0, 0.0, 0.0, 1.0e6); n];
    }
    let mut rng = crate::transport::rng::Rng::new(batch_seed, 0);
    (0..n)
        .map(|_| {
            let idx = (rng.uniform() * bank.len() as f64) as usize;
            bank[idx.min(bank.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_matches_feature_flag() {
        let r = Backend::recommended();
        #[cfg(feature = "cuda")]
        assert_eq!(r, Backend::Cuda);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(r, Backend::Cpu);
    }

    #[test]
    fn label_strings() {
        assert_eq!(Backend::Cpu.label(), "CPU");
        #[cfg(feature = "cuda")]
        assert_eq!(Backend::Cuda.label(), "CUDA");
    }
}
