# Stage C data model — `PerNuclideGpu` + `GpuBundle`

Status: **design**. Implementation lands incrementally on
`feat/per-nuclide-gpu-cache`. Each commit converts one category and
keeps `cargo test --features cuda --lib` green.

Reference for follow-on sessions. Cross-link from `gpu_transport.rs`
when each category lands.

## Source we're splitting

`gpu_transport.rs::upload_nuclide_data_uncached` — 808 lines
(1225-2033). Returns a single flat-pack `GpuNuclideData` with ~80
fields. Every per-nuclide datum is concatenated into one big
`CudaSlice` per field; access is offset-indirect from `*_offsets`
arrays.

## Target architecture (no kernel changes in stage 3)

```
upload_one_nuclide(stream, &NuclideKernels, rank)
    -> Arc<PerNuclideGpu>              // owns single-nuclide CudaSlices

assemble_bundle(stream, &[Arc<PerNuclideGpu>], rank)
    -> GpuNuclideData                  // flat-pack view, kernel ABI unchanged
        - allocates flat destination CudaSlices sized = sum of per-nuclide bytes
        - cuMemcpyDtoD per category from per-nuclide → bundle
        - rebuilds *_offsets on host with running global offsets, clone_htod
```

Kernel sees byte-identical `GpuNuclideData`. Cache hit path = per-nuclide
upload skipped, assembly stage still runs (cheap; D→D bandwidth ≈ 200
GB/s on A1000, a 4 GB bundle assembles in ~20 ms).

Stage 4 (separate commit, gated on full validation campaign) drops
the assembly view and changes the kernel ABI to read pointer arrays
directly.

## Field inventory — categories

### A. Pure per-nuclide, no inter-nuclide stride

Each entry is a single nuclide's CudaSlice. Bundle assembly concats by
nuclide and emits `*_offsets[nuc_idx]` arrays.

| `GpuNuclideData` field(s)                                | source on `NuclideKernels`                  |
|----------------------------------------------------------|---------------------------------------------|
| `all_energy_grids` + `grid_offsets` + `n_energies`       | `kernel.energies()`, `kernel.n_energy()`    |
| `total_xs` + `total_xs_offsets` + `has_total_xs`         | `nuc.total_xs_raw`                          |
| `pointwise_xs` + `pw_offsets` + `has_pw`                 | `nuc.pointwise_xs`                          |
| `nu_bar_*` (4 fields)                                    | `nuc.nu_bar_table`                          |
| `delayed_nu_bar_*` (4 fields)                            | `nuc.delayed_nu_bar_table`                  |
| `inel_cdf_*` (7 fields)                                  | `nuc.inelastic_cdf`                         |
| `urr_*` (10 fields)                                      | `nuc.urr_tables`                            |
| `inel91_*` (8 fields)                                    | `nuc.inelastic_continuum_edist`             |
| `watt_*` (6 fields)                                      | `nuc.fission_energy_dist` (Watt branch)     |
| `maxevap_*` (6 fields)                                   | `nuc.fission_energy_dist` (Maxwell/Evap)    |
| `fis_*` (8 fields, tabular branch)                       | `nuc.fission_energy_dist` (tabular)         |
| `ang_*` (8 fields)                                       | `nuc.elastic_angle`                         |

### B. Per-(nuclide × reaction) — `n_nuc × n_rxn` stride

`n_rxn = 7` (elastic, inelastic, n2n, n3n, fission, capture, total).
Per-nuclide slot holds up to 7 SVD or pointwise-as-SVD basis/coeffs
pairs.

| `GpuNuclideData` field         | rule                                              |
|--------------------------------|---------------------------------------------------|
| `all_basis` + `basis_offsets`  | concat `basis_f64()` (rank-padded if Table)       |
| `all_coeffs` + `coeffs_offsets`| concat `coeffs` (length = rank)                   |
| `has_reaction`                 | 0/1 per slot                                      |

### C. Discrete inelastic levels — per-(nuclide × level) stride

Per-nuclide level count varies (Au-197 has 1, U-238 has 13, Pu-239
has ~20). Bundle indexes by *global level index*; per-nuclide first
global level is `level_offsets[nuc_idx]`.

| `GpuNuclideData` field                            | per-level rule                              |
|---------------------------------------------------|---------------------------------------------|
| `level_q_values`, `level_thresholds`, `level_mt`  | scalar per global level                     |
| `level_has_kernel`                                | scalar per global level                     |
| `level_offsets[nuc]`, `level_counts[nuc]`         | bundle metadata                             |
| `level_basis`, `level_basis_offsets`              | rank-padded basis per global level          |
| `level_coeffs`, `level_coeffs_offsets`            | rank-padded coeffs per global level         |
| `lev_ang_lev_off[lev]`, `lev_ang_lev_ne[lev]`     | per-level CM angular dist locator           |
| `lev_ang_energies`, `lev_ang_dist_off/sz`         | per-(level, inc_energy) flatten             |
| `lev_ang_mu`, `lev_ang_cdf`                       | per-(level, inc_energy, mu_idx) flatten     |

**Per-level SVD rank-padding invariant** (`1654c4d`) — must be
preserved by `upload_one_nuclide`. Each level's basis is padded from
`[n_e × level_rank]` to `[n_e × global_rank]` with zero columns;
coeffs padded from `level_rank` to `global_rank` with zeros. The
device kernel reads `basis[e_idx * P_RANK + j]` for the full j range,
so omitting the padding silently reads adjacent levels' bytes →
+500-700 pcm fast-metal hot bias.

**Rank-dependence** of per-nuclide cache key: per-level basis bytes
depend on the run's global rank. `NuclideKey::policy_hash` captures
rank → same nuclide at rank=5 and rank=7 produces two cache entries.

### D. Bundle-level scalar

- `rank: i32` (run-global, not per-nuclide). Lives on `GpuNuclideData`
  directly; per-nuclide cache entries record the rank they were padded
  for so the assembly stage can verify.

## `PerNuclideGpu` schema (proposed)

```rust
pub struct PerNuclideGpu {
    pub rank: i32,                                 // padding rank, must match bundle rank
    pub n_energy: i32,

    // A.1 energy grid (always present; sentinel `[0.0]` if empty)
    pub energy_grid: CudaSlice<f64>,

    // B per-reaction (slot=7; index by rxn_idx, 0..7)
    pub has_reaction: [i32; 7],
    pub basis: [Option<CudaSlice<f64>>; 7],        // [n_e × rank] per slot
    pub coeffs: [Option<CudaSlice<f64>>; 7],       // [rank] per slot

    // A.2 pointwise tables
    pub total_xs: Option<CudaSlice<f64>>,
    pub pointwise_xs: Option<CudaSlice<f64>>,      // [n_e × 7] (7 channels concatenated)

    // A.3 ν̄ tables (prompt + delayed)
    pub nu_bar: Option<NuBarSlicesGpu>,            // (energies, values, size)
    pub delayed_nu_bar: Option<NuBarSlicesGpu>,

    // C discrete levels — already concatenated per-nuclide
    pub levels: LevelSlicesGpu,                    // see below

    // A.4 elastic angular dist
    pub elastic_angle: Option<AngularSlicesGpu>,

    // A.5 fission energy distribution
    pub fission_edist: FissionEdistGpu,            // Tabular | Watt | MaxEvap | None

    // A.6 MT=91 continuum
    pub inel91: Option<TabularEdistSlicesGpu>,

    // A.7 URR
    pub urr: Option<UrrSlicesGpu>,

    // A.8 synthesized inelastic CDF (Zr-90/91/92/94, U-238)
    pub inel_cdf: Option<InelCdfSlicesGpu>,
}

pub struct LevelSlicesGpu {
    pub n_levels: i32,
    pub q_values: CudaSlice<f64>,
    pub thresholds: CudaSlice<f64>,
    pub mt: CudaSlice<i32>,
    pub has_kernel: CudaSlice<i32>,
    pub basis: CudaSlice<f64>,                     // [Σ n_e_l × rank], pre-padded
    pub coeffs: CudaSlice<f64>,                    // [n_levels × rank], pre-padded
    pub basis_local_offsets: Vec<i32>,             // host-side, [n_levels]; shifted at assembly
    pub coeffs_local_offsets: Vec<i32>,            // host-side, [n_levels]
    pub ang_energies: CudaSlice<f64>,              // [Σ n_e_inc per level]
    pub ang_mu: CudaSlice<f64>,
    pub ang_cdf: CudaSlice<f64>,
    pub ang_lev_local_off: Vec<i32>,               // host-side, [n_levels] (shifted)
    pub ang_lev_ne: Vec<i32>,                      // host-side, [n_levels]
    pub ang_dist_local_off: Vec<i32>,              // host-side, per (level, e_idx)
    pub ang_dist_sz: Vec<i32>,                     // host-side
}
```

**Host-side offset arrays**: per-nuclide local offsets stay on the
host. The bundle assembly stage shifts them by running global offsets
and `clone_htod` the final `*_offsets` arrays. Per-nuclide host
offsets cost ~32 bytes per nuclide per category — negligible vs the
device payload.

## Bundle assembly algorithm

```
assemble_bundle(per_nucs, rank) -> GpuNuclideData:
  // Stage 1: pre-compute totals + per-nuclide global offsets (host)
  for each category C:
      compute total_bytes_C = Σ per_nucs[i].C.bytes()
      compute global_off_C[i] = Σ_{j<i} per_nucs[j].C.bytes() / elem_size

  // Stage 2: allocate destination CudaSlices (one per flat field)
  bundle.all_basis = stream.alloc_zeros(total_basis_count)
  bundle.level_basis = stream.alloc_zeros(total_level_basis_count)
  // ... etc

  // Stage 3: D→D copy per per-nuclide payload
  for nuc_idx, p in per_nucs.iter().enumerate():
      for r in 0..7:
          if let Some(b) = &p.basis[r]:
              dst = bundle.all_basis.slice_mut(global_off ..)
              memcpy_dtod(dst, b)
              update host-side basis_offsets_vec[nuc_idx*7 + r]
      // ... per category

  // Stage 4: clone_htod the offset arrays
  bundle.basis_offsets = stream.clone_htod(&basis_offsets_vec)
  // ... etc
```

## Implementation order (commits on branch)

Each commit lands one category, with `cargo check --features cuda` +
`cargo test --features cuda --lib` green before commit.

1. **Skeleton** — `PerNuclideGpu` struct, `upload_one_nuclide` returns
   a stub with only `energy_grid` + `n_energy`. `assemble_bundle`
   builds the rest by falling through to the old packing path so the
   refactor stays bisectable.
2. **Pointwise tables** — `total_xs`, `pointwise_xs`. Simplest
   per-nuclide concat.
3. **ν̄ tables** — `nu_bar_*`, `delayed_nu_bar_*`. Same shape.
4. **Elastic angular** — `ang_*` (per-nuclide, per-energy).
5. **Per-reaction SVD** — `all_basis`, `all_coeffs`, `has_reaction`
   (Cat B). Touches the hottest path for k-eff; validates SVD parity.
6. **Discrete levels** — `level_*`, `lev_ang_*` (Cat C). Largest
   payload; per-level rank-padding invariant lives here.
7. **Fission edist** — `fis_*`, `watt_*`, `maxevap_*`. Tagged union.
8. **MT=91 continuum** — `inel91_*`.
9. **URR** — `urr_*`.
10. **Synth inelastic CDF** — `inel_cdf_*`.
11. **Cleanup** — drop the fallthrough path; `upload_nuclide_data_uncached`
    is now just `nuclides.iter().map(upload_one_nuclide).collect()`
    followed by `assemble_bundle`.

Each commit ~150 LOC.

## Validation gates

After every commit:
- `cargo check --features cuda` clean
- `cargo test --features cuda --lib` 428 / 428 pass

After commit 11 (full split landed):
- `cargo run --release --features cuda --bin level_xs_compare` —
  per-discrete-level XS A/B CPU↔GPU, no regressions
- `cargo run --release --features cuda --bin metal_stats_diag` —
  3-way Godiva/PMF-001/HMF-001 stays at post-`1654c4d` values
- `cargo test --features cuda --test cuda_runs` — 6 / 6 ICSBEP pass
