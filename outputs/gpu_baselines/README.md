# GPU baselines captured before task #22 (recursive transport integration)

These runs lock in the bit-identical references that the recursive
transport integration must preserve through every step of the
refactor. Reproduced on:

  RTX A1000 Laptop, CUDA 13.2, NVIDIA-SMI 595.79
  Windows 11 Enterprise 26200, OUTSYSTEMS+fog
  HEAD = a9bff75 (parity primitives + extended trace_step parity)

## `gpu_pwr_pre22.txt` — PWR pin-cell, GPU SVD k=5
Single seed, 50 batches × 10 000 particles, 40 active batches.

  k_inf       = 1.32568
  ns/particle = 31 522
  total sim   = 12.6 s
  load time   = 4.3 s CPU + 2.4 s GPU

This is the depth-1, hardcoded-pin-cell path through
`transport_persistent`. Any change to `transport.cu` must reproduce
these numbers byte-for-byte when the `geom_type` stays at `GEOM_PWR`.

## `gpu_godiva_xs_pre22.txt` — Godiva XS reconstruction, single nuclide
1 000 000 particles, U-235 fission XS reconstruction.

  CPU      = 15.0 ns/particle
  GPU      =  3.3 ns/particle
  speedup  = 4.6×
  CPU-vs-GPU max relative error = 0.0 (first 10 reconstructions)

Locks in the SVD-on-GPU bit-identical agreement. Any change to the
SVD reconstruction kernel must keep `err = 0.0`.

## What the recursive integration must NOT change

  * Per-seed Godiva k_eff on `gpu_pwr_bench --mode svd` for the
    PWR pin-cell geom: must reproduce 1.32568 byte-for-byte when
    geom_type = GEOM_PWR.
  * U-235 SVD reconstruction max relative error: must stay 0.0.

## What the recursive integration adds (target)

  * 17×17 PWR assembly k_inf via GPU agrees with CPU within combined
    MC noise (CPU recorded at k_inf = 1.14958 ± 0.00318 across 175 000
    histories — see outputs/recursive_geometry_regression/).
  * 17×17 assembly per-particle GPU throughput at least 5× CPU rayon.
    CPU rayon record: 27 361 ns/p.
    Target on GPU:     ≤ 5 472 ns/p.

## Phased plan (task #22)

  1. ✓ Capture baselines (this dir).
  2. Add P_GR_* parameter slots + Rust-side upload alongside the
     existing transport context. No kernel changes — confirms the
     data path doesn't disturb existing kernels.
  3. Add a `transport_recursive_persistent` kernel — clone of
     `transport_persistent` with `find_cell` / `trace_surface` /
     `cell_material` calls dispatched to `gr_*` variants. Existing
     `transport_persistent` stays untouched.
  4. Build `gpu_assembly_bench` binary that drives the new kernel.
  5. Validate Godiva and PWR pin-cell unchanged through the existing
     entry point. Validate assembly k_inf agrees with CPU within MC
     noise. Validate ≥5× speedup on assembly.
  6. Once stable, the existing `transport_persistent` + hardcoded
     `geom_type` paths can be retired in a separate cleanup task.
