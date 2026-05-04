//! Parallel-safe pseudo-random number generator.
//!
//! Re-exports the PCG-64 implementation from
//! [`rust_mc_sim::Pcg64`](rust_mc_sim::Pcg64) under the engine's
//! historical name `Rng`. Same byte-for-byte behaviour as the
//! pre-migration local copy: per-particle seeding via
//! `Rng::for_particle(batch, particle_id)` is deterministic and
//! reproducible across runs and platforms.

pub use rust_mc_sim::Pcg64 as Rng;
