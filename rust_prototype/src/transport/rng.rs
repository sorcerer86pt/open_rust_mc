// SPDX-License-Identifier: MIT
//! PCG-64 re-exported as the historical `Rng` name.
//! `Rng::for_particle(batch, particle_id)` is byte-for-byte
//! reproducible across runs / platforms.

pub use rust_mc_sim::Pcg64 as Rng;
