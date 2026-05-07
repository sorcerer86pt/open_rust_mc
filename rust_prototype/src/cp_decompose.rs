//! CP / PARAFAC decomposition of a 3-tensor — re-exports the
//! implementation from
//! [`rust_mc_sim::cp`](rust_mc_sim::cp) under the engine's historical
//! module path. Same algorithm (greedy rank-1 deflation), same public
//! types (`CpDecomposition`, `cp_greedy_rank1`,
//! `relative_l2_error`, `max_abs_error`).

pub use rust_mc_sim::cp::{CpDecomposition, cp_greedy_rank1, max_abs_error, relative_l2_error};
