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
///
/// **Recommended `particles_per_batch` for the event-based pipeline.**
/// Tramm et al., "Toward Portable GPU Acceleration of the OpenMC
/// Monte Carlo Particle Transport Code" (PHYSOR 2022), report that
/// on an A100 the event-based mode continues to gain performance up
/// to **8 million particles in-flight** before exhausting device
/// memory — i.e. saturation is two orders of magnitude beyond what
/// is conventional for CPU MC runs. A reasonable target on a 3080
/// with 10 GB VRAM is **100k–1M particles per batch**; on the 4 GB
/// RTX A1000 laptop, 50 k is the practical ceiling. Smaller batches
/// (≤5 k) leave most of the per-kernel-launch and per-step PCIe
/// overhead unamortised. ncu profiling of the previous persistent
/// kernel showed active_threads_per_warp = 6.2/32 on PWR-17×17 and
/// every other scene where reaction-type dispatch diverged warps.
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
    pub initial_source: Box<dyn Fn(usize, u64) -> Vec<(f64, f64, f64, f64)> + 'a>,
    /// Lazily built on first batch, reused across batches. `RefCell`
    /// for `&self`-on-run; CudaRunner is !Sync anyway.
    pub buffers: std::cell::RefCell<Option<crate::gpu_recursive::TransportBuffers>>,
    /// PHYSOR 2022 Optimization F refill pool, allocated lazily when
    /// `config.gpu_refill_pool_factor` is `Some(_)` on the first
    /// batch. Sized for the *overflow* beyond the in-flight slots —
    /// i.e. `n * (factor - 1)` particles, capped at the per-batch
    /// (factor * n) total bank size.
    pub refill: std::cell::RefCell<Option<crate::gpu_recursive::RefillBuffers>>,
}

#[cfg(feature = "cuda")]
impl<'a> EigenvalueRunner for CudaRunner<'a> {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn run(&self, config: &SimConfig) -> EigenvalueOutcome {
        use crate::transport::particle::FissionSite;
        use crate::transport::rng::Rng;
        use rust_mc_sim::Pcg64;

        let n = config.particles_per_batch as usize;
        // PHYSOR 2022 Optimization F refill. When `gpu_refill_pool_factor`
        // is `Some(f)`, the source bank is sized to `n * f` and the
        // overflow goes into the device-side RefillBuffers. The kernel
        // refills dead slots between event steps until the bank drains.
        // `f = 1.0` is equivalent to disabled (no overflow). `n_overflow`
        // is the per-batch refill capacity.
        let refill_factor = config.gpu_refill_pool_factor.unwrap_or(1.0).max(1.0);
        let total_bank = ((n as f64) * refill_factor).round() as usize;
        let n_overflow = total_bank.saturating_sub(n);
        let use_refill = n_overflow > 0;

        // Build the full bank (size = n + n_overflow). First `n`
        // entries seed the active slots, the rest are uploaded to the
        // RefillBuffers below.
        let full_bank: Vec<(f64, f64, f64, f64)> = match config.initial_source_bank.as_ref() {
            Some(bank) if !bank.is_empty() => {
                let mut rng = Rng::new(config.seed * 100_000, 1);
                (0..total_bank)
                    .map(|_| {
                        let idx = (rng.uniform() * bank.len() as f64) as usize;
                        let s = &bank[idx.min(bank.len() - 1)];
                        (s.pos.x, s.pos.y, s.pos.z, s.energy)
                    })
                    .collect()
            }
            _ => (self.initial_source)(total_bank, config.seed),
        };
        let mut source: Vec<(f64, f64, f64, f64)> = full_bank[..n].to_vec();
        let mut batches: Vec<BatchResult> = Vec::with_capacity(config.batches as usize);
        let mut k_sum = 0.0_f64;
        let mut k_count = 0_u32;

        // When refill is active the fission bank has to fit
        // `total_bank * nu_max` daughters per batch, not just `n * nu`.
        // Scale fis_capacity by the refill factor (ceil for safety
        // margin); without this the bank silently caps at fis_capacity
        // and k_eff is reported as `capped_count / total_histories`,
        // which is biased low (-7000 pcm on Godiva with factor=2.0).
        let effective_fis_capacity = if use_refill {
            ((self.fis_capacity as f64) * refill_factor).ceil() as usize
        } else {
            self.fis_capacity
        };

        for batch in 1..=config.batches {
            let batch_seed = config.seed * 100_000 + batch as u64 * 1_000;
            let rng_seeds: Vec<(u64, u64)> = (0..n)
                .map(|i| {
                    let p = Pcg64::for_particle(batch_seed, i as u64);
                    (p.state(), p.stream())
                })
                .collect();

            let mut buffers_guard = self.buffers.borrow_mut();
            if buffers_guard.is_none() {
                let params_len = self
                    .transport
                    .build_transport_params_vec(
                        self.nuc_data,
                        self.mat_data,
                        self.sab_data,
                        self.wmp_data,
                        0,
                    )
                    .len();
                let pool = crate::gpu_recursive::TransportBuffers::new(
                    &self.recursive.stream,
                    n,
                    effective_fis_capacity,
                    self.mat_k_t.len(),
                    self.recursive.n_lattices(),
                    params_len,
                )
                .expect("TransportBuffers::new failed");
                *buffers_guard = Some(pool);
            }
            let buffers = buffers_guard.as_mut().expect("buffers init above");

            // Lazy-allocate the refill pool on the first batch that
            // requests it. Reuse across subsequent batches so the
            // device allocations don't churn — the capacity is fixed
            // by the first observed `n_overflow` and the loop calls
            // RefillBuffers::reset() before each transport call.
            let mut refill_guard = self.refill.borrow_mut();
            if use_refill && refill_guard.is_none() {
                let refill = crate::gpu_recursive::RefillBuffers::new(
                    &self.recursive.stream,
                    n_overflow,
                )
                .expect("RefillBuffers::new failed");
                *refill_guard = Some(refill);
            }

            // Build the per-batch overflow bank from the SAME
            // distribution as `source` (the active slots). For batch 1
            // this is the initial source; for batch 2+ it must come
            // from the just-converged fission bank, not the stale
            // initial-source `full_bank` we built before the loop. If
            // we use full_bank[n..] for batch 2+, the refilled
            // particles inherit the initial source's spatial
            // distribution (lower importance for converged scenes) and
            // bias k_eff down ~700 pcm on Godiva (measured).
            let overflow_bank: Vec<(f64, f64, f64, f64)> = if !use_refill {
                Vec::new()
            } else if batch == 1 {
                full_bank[n..].to_vec()
            } else {
                // Resample n_overflow particles from the same fission
                // bank that produced `source` for this batch. Uses a
                // distinct RNG stream so the overflow draws don't
                // collide with the active-slot resampling done in
                // normalize_gpu_bank.
                let mut rng = Rng::new(
                    batch_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    1,
                );
                (0..n_overflow)
                    .map(|_| {
                        let bank = &source;
                        let idx = (rng.uniform() * bank.len() as f64) as usize;
                        bank[idx.min(bank.len() - 1)]
                    })
                    .collect()
            };

            // Upload the overflow source into the refill bank if
            // active. Per-particle PCG seeds derive from the batch
            // seed mirroring the same scheme as the active slots, but
            // with offset `n..total_bank` so they don't collide.
            let result = if let (true, Some(refill)) = (use_refill, refill_guard.as_mut()) {
                let stream = &self.recursive.stream;
                let mut rx = Vec::with_capacity(n_overflow);
                let mut ry = Vec::with_capacity(n_overflow);
                let mut rz = Vec::with_capacity(n_overflow);
                let mut re = Vec::with_capacity(n_overflow);
                let mut rrs = Vec::with_capacity(n_overflow);
                let mut rri = Vec::with_capacity(n_overflow);
                for i in 0..n_overflow {
                    let s = &overflow_bank[i];
                    rx.push(s.0);
                    ry.push(s.1);
                    rz.push(s.2);
                    re.push(s.3);
                    let p = Pcg64::for_particle(batch_seed, (n + i) as u64);
                    rrs.push(p.state());
                    rri.push(p.stream());
                }
                stream.memcpy_htod(&rx, &mut refill.d_refill_pos_x).expect("htod refill x");
                stream.memcpy_htod(&ry, &mut refill.d_refill_pos_y).expect("htod refill y");
                stream.memcpy_htod(&rz, &mut refill.d_refill_pos_z).expect("htod refill z");
                stream.memcpy_htod(&re, &mut refill.d_refill_energy).expect("htod refill e");
                stream.memcpy_htod(&rrs, &mut refill.d_refill_rng_state).expect("htod refill rs");
                stream.memcpy_htod(&rri, &mut refill.d_refill_rng_inc).expect("htod refill ri");

                let r = self
                    .recursive
                    .transport_recursive_with_buffers(
                        buffers,
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
                        effective_fis_capacity,
                        Some(refill),
                    )
                    .expect("transport_recursive_with_buffers (refill) failed");
                r
            } else {
                self.recursive
                    .transport_recursive_with_buffers(
                        buffers,
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
                        effective_fis_capacity,
                        None,
                    )
                    .expect("transport_recursive_with_buffers failed")
            };

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
