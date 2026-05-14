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
pub use rust_mc_sim::svd::{LogHashIndex as EnergyHashTable, SvdKernel, reconstruct_log_faer};

/// Bin count for the `LogHashIndex` built by [`rehydrate_for_runtime`].
/// Matches the value `SvdKernel::from_data` uses internally — keep them
/// in sync so live-load and cache-decode produce identical hash
/// granularity.
pub const HASH_BIN_COUNT: usize = 8192;

/// Threshold below which the hash isn't worth building (small kernels
/// are dominated by the hash overhead rather than the lookup
/// speedup). Mirrors `SvdKernel::from_data`'s `if n_rows > 100` rule.
pub const HASH_MIN_ROWS: usize = 100;

/// **The** policy for getting a freshly-decoded or freshly-built
/// [`SvdKernel`] into the same shape the transport hot path expects.
/// Single source of truth so live-load (HDF5 → `SvdKernel::from_data`)
/// and cache-decode (`nuclide_cache::binary_format::decode_svd_kernel`)
/// stay byte-for-byte semantically equivalent.
///
/// What this guards against
/// ========================
///
/// `LogHashIndex::lookup` returns the **lower** bracket
/// (`largest idx with axis[idx] ≤ value`).
/// `SvdKernel::row_index_binary` (the fallback when `hash` is `None`)
/// returns the **upper** bracket (`smallest idx with axis[idx] ≥
/// value`). They differ by 1 on every off-grid query.
///
/// The CPU transport hot path was written against the lower-bracket
/// convention: `XsProvider::lookup` does one shared binary search,
/// reads `grid[idx]` as the lower bound, and interpolates with
/// `log_frac = (log(E) - log(grid[idx])) / (log(grid[idx+1]) -
/// log(grid[idx]))`. Without the hash, `row_index` returns the
/// **upper** bracket → `grid[idx]` is the upper edge → `log_frac`
/// clamps to 0 → the SVD reconstruction returns the upper-edge value
/// at every off-grid energy. On U-235 thermal capture this dragged
/// CPU SVD k_inf by ~19 000 pcm; on heu-comp-inter-003 thermal cases
/// it dragged k by ~6 000 pcm before this helper was extracted.
///
/// Why this is `n_rows > HASH_MIN_ROWS`
/// =====================================
///
/// On axes with fewer than 100 rows the hash overhead (~32 KB for
/// `n_bins = 8192`) exceeds the lookup-speedup payoff. `from_data`
/// applies the same threshold; matching it here keeps the live-load
/// and cache-decode memory footprints identical.
///
/// Why this can't be a `From` impl on `SvdKernel`
/// ===============================================
///
/// `SvdKernel` lives in `rust_mc_sim`; we don't own the type. The
/// policy ("when to build the hash") is consumer-specific — different
/// engines might decide differently. Keeping the policy in *our*
/// crate, applied via a named free function, is the right ownership
/// shape. If `rust_mc_sim` ever adds another runtime-only internal
/// (e.g., a per-temperature interpolator cache), extend this function
/// once and every consumer benefits.
pub fn rehydrate_for_runtime(kernel: &mut SvdKernel) {
    if kernel.n_rows() > HASH_MIN_ROWS {
        kernel.build_hash(HASH_BIN_COUNT);
    }
}
