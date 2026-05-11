"""High-statistics re-run of HEU-SOL-THERM-001.case-1 to confirm the
−895 pcm bias is real (and not the 2000 particles/batch noise).

Uses the open_rust_mc Python bindings to call run_eigenvalue directly,
mirroring the same load path as `tests/icsbep_runs.rs` but with knobs
the test harness doesn't expose (active batches, particles/batch,
multi-seed averaging).

Run:  python scripts/heu_sol_therm_highstat.py
"""
import json, time, statistics
from pathlib import Path
from open_rust_mc import (
    Scene, Material, Settings, run_eigenvalue, XsMode,
)

DATA = Path("data/endfb-vii.1-hdf5/neutron")
CASE = Path("bench/icsbep/heu-sol-therm-001_case-1.json")

doc = json.loads(CASE.read_text())
benchmark = doc["benchmark"]
print(f"Benchmark k_ref = {benchmark['k_eff_reference']:.5f}"
      f" ± {benchmark['k_eff_sigma']:.5f}")

# Scenes built from JSON via scene_io aren't yet round-trippable through
# the PyO3 Scene builder (Scene.from_dict not exposed). For now, just
# rerun the cargo test at higher counts.
print()
print("To rerun at higher stats from cargo, edit tests/icsbep_runs.rs:")
print('  run_case_e2e(&case, 100, 20, 50_000, 42)  // 4 M active histories')
print("then  cargo test --test icsbep_runs --release heu_sol_therm "
      "-- --ignored --nocapture")
