// SPDX-License-Identifier: MIT
//! Stage 6 — end-of-sweep summary + final artefacts.
//!
//! Runs after every `ExecutionResult` has been processed. Writes the
//! final scatter / EALF correlation plots, prints the human-readable
//! pass/fail summary table, evicts the L1 nuclide cache (L2 stays so
//! the next run benefits), and exits with the right code:
//!
//!   * 0 — all PASS
//!   * 1 — any FAIL but no ERROR
//!   * 2 — any ERROR
//!
//! Phase 1: scaffold only.
