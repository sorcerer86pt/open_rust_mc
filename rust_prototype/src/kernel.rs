//! SVD reconstruction kernel — re-exports from
//! [`rust_mc_sim`](rust_mc_sim).
//!
//! `SvdKernel` and `ducru_weights` come from
//! [`rust_mc_sim::svd`](rust_mc_sim::svd) and
//! [`rust_mc_sim::ducru`](rust_mc_sim::ducru). The historical
//! `EnergyHashTable` is the same type as
//! [`rust_mc_sim::svd::LogHashIndex`] — re-exported under both names
//! so existing call sites keep compiling.

pub use rust_mc_sim::ducru::ducru_weights;
pub use rust_mc_sim::svd::{reconstruct_log_faer, LogHashIndex as EnergyHashTable, SvdKernel};
