//! Real-HDF5 `NuclideKernels` ↔ binary-format roundtrip.
//!
//! The 8 in-lib `nuclide_cache::binary_format::tests` cover the empty
//! `NuclideKernels` and synthetic sub-types — every Option-None branch
//! and every primitive encoder. Those tests catch protocol bugs but
//! say nothing about whether real ENDF/B-VII.1 evaluations (with
//! their multi-MB SVD bases, populated angular distributions, URR
//! tables, photon products, and discrete-level walks) survive the
//! round-trip.
//!
//! This integration test loads four representative actinide / light
//! nuclides from disk via `load_nuclide`, encodes via
//! `encode_nuclide_kernels`, decodes back, **re-encodes**, and asserts
//! the two byte streams are identical. Bytemuck-cast `Vec<f64>` blocks
//! mean a successful re-encode is byte-exact bijection — any silent
//! drift (mis-ordered field, wrong section length, off-by-one
//! discriminant) shows up as a mismatch in the first differing byte.
//!
//! Spot-checks on AWR, `nu_bar_const`, fission-spectrum law variant,
//! URR-table presence, and discrete-level count cover the
//! semantically important fields a byte-equal check would technically
//! tolerate (encoded zero ≠ encoded zero would still be byte-equal —
//! we want to confirm the right values are coming back).
//!
//! Data dir: `ICSBEP_DATA_DIR` env var → walks up to find
//! `data/endfb-vii.1-hdf5/neutron`. Skipped (`return`) when not found,
//! same convention as `tests/cuda_runs.rs` and `tests/icsbep_runs.rs`.

use std::path::PathBuf;

use open_rust_mc::hdf5_reader::FissionEnergyLaw;
use open_rust_mc::transport::nuclide_cache::binary_format::{
    decode_nuclide_kernels, encode_nuclide_kernels,
};
use open_rust_mc::transport::xs_provider::load_nuclide;

fn data_dir() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("ICSBEP_DATA_DIR") {
        return Some(PathBuf::from(v));
    }
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("data/endfb-vii.1-hdf5/neutron").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    let neutron = p.join("data/endfb-vii.1-hdf5/neutron");
    neutron.is_dir().then_some(neutron)
}

fn byte_exact_roundtrip(file: &str, awr_fb: f64, nu_bar_fb: f64) {
    let Some(dir) = data_dir() else {
        eprintln!("[skip] ENDF data dir not found — set ICSBEP_DATA_DIR");
        return;
    };
    let path = dir.join(file);
    if !path.is_file() {
        eprintln!("[skip] {} not on disk", path.display());
        return;
    }

    let original = load_nuclide(&path, 5, 0, awr_fb, nu_bar_fb);

    let bytes1 = encode_nuclide_kernels(&original).expect("first encode must succeed");
    let decoded = decode_nuclide_kernels(&bytes1).expect("decode must succeed");
    let bytes2 = encode_nuclide_kernels(&decoded).expect("re-encode must succeed");

    // Byte-for-byte equality is the strong contract — `bytemuck::cast_slice`
    // means any drift in field ordering, section length, or
    // discriminant value shows up as a mismatch in the first
    // differing byte.
    assert_eq!(
        bytes1.len(),
        bytes2.len(),
        "encoded length mismatch on {file}: {} vs {}",
        bytes1.len(),
        bytes2.len(),
    );
    assert_eq!(bytes1, bytes2, "encoded bytes differ on {file}");

    // Spot-check semantic equality on a few fields that the
    // byte-equality test technically wouldn't catch on its own
    // (if we accidentally read everything as zero, byte-equality
    // would still pass).
    assert_eq!(decoded.awr, original.awr, "AWR mismatch on {file}");
    assert_eq!(
        decoded.nu_bar_const, original.nu_bar_const,
        "nu_bar_const mismatch on {file}"
    );
    assert_eq!(
        decoded.has_continuum_inelastic, original.has_continuum_inelastic,
        "has_continuum_inelastic mismatch on {file}"
    );
    assert_eq!(
        decoded.discrete_levels.len(),
        original.discrete_levels.len(),
        "discrete_levels.len mismatch on {file}"
    );
    assert_eq!(
        decoded.urr_tables.is_some(),
        original.urr_tables.is_some(),
        "urr_tables presence mismatch on {file}"
    );
    assert_eq!(
        decoded.fission_energy_dist.is_some(),
        original.fission_energy_dist.is_some(),
        "fission_energy_dist presence mismatch on {file}"
    );
    // Fission spectrum law variant — fissile nuclides only. Confirms
    // the FissionEnergyLaw discriminant byte (Watt=0, Maxwell=1,
    // Evaporation=2) round-trips.
    if let (Some(a), Some(b)) = (&original.fission_energy_dist, &decoded.fission_energy_dist)
        && let (Some(a_law), Some(b_law)) = (&a.closed_form, &b.closed_form)
    {
        let same_variant = matches!(
            (a_law, b_law),
            (FissionEnergyLaw::Watt(_), FissionEnergyLaw::Watt(_))
                | (FissionEnergyLaw::Maxwell(_), FissionEnergyLaw::Maxwell(_))
                | (FissionEnergyLaw::Evaporation(_), FissionEnergyLaw::Evaporation(_))
        );
        assert!(
            same_variant,
            "fission spectrum law variant mismatch on {file}"
        );
    }

    eprintln!(
        "  {file}: round-trip OK ({} MB, {} discrete levels, urr={})",
        bytes1.len() / 1024 / 1024,
        decoded.discrete_levels.len(),
        decoded.urr_tables.is_some(),
    );
}

/// **U-235** — Watt fission spectrum (Cranberg-ish a=0.988 MeV / b=2.249
/// /MeV). Heavy actinide: ~10-15 MB of SVD basis at rank 5, populated
/// URR tables, ~30 discrete inelastic levels, photon products on
/// MT=18 / MT=102. The canonical "if any nuclide round-trips, this one
/// does" sanity check.
#[test]
fn u235_real_hdf5_roundtrip() {
    byte_exact_roundtrip("U235.h5", 233.025, 2.43);
}

/// **U-233** — Maxwell fission spectrum. Exercises the
/// `FissionEnergyLaw::Maxwell(MaxwellLaw)` encode/decode pair on real
/// HDF5 data, complementing the synthetic-data test in
/// `binary_format::tests::fission_energy_law_three_variants_roundtrip`.
/// Also the nuclide our GPU Maxwell/Evap fix targets.
#[test]
fn u233_real_hdf5_roundtrip() {
    byte_exact_roundtrip("U233.h5", 231.038, 2.49);
}

/// **Pu-239** — Watt χ; the dominant fissile in every Pu-mixed ICSBEP
/// case. Larger discrete-level table than U-235, lots of photon
/// products. Stresses the photon-products `Vec<(u32, PhotonProduct)>`
/// encoder + the nested `NuBarTable + EnergyDistribution` it carries.
#[test]
fn pu239_real_hdf5_roundtrip() {
    byte_exact_roundtrip("Pu239.h5", 236.999, 2.88);
}

/// **Zr-90** — non-fissile, no URR, no fission spectrum, no photon
/// products. The opposite end of the type tree from U-235: most
/// fields are `None`, the encoders walk a lot of `write_option(...,
/// None)` branches. Validates the cheap-case path.
#[test]
fn zr90_real_hdf5_roundtrip() {
    byte_exact_roundtrip("Zr90.h5", 89.132, 0.0);
}

/// **Lookup-equivalence regression.** Byte-exact roundtrip is a
/// necessary condition but not sufficient: a decoded `NuclideKernels`
/// can serialize to identical bytes while behaving differently at
/// transport time. Specifically, the L2 disk-cache miss path used to
/// reconstruct `SvdKernel` via `from_factors`, which leaves
/// `kernel.hash = None`, and `row_index_binary` (the fallback) returns
/// the **upper** bracket — but every other lookup site in the codebase
/// uses the **lower** bracket convention `LogHashIndex::lookup`
/// returns. The off-by-one index produced wrong XS lookups → ~6000
/// pcm k_eff drift on `heu-comp-inter-003` thermal cases on a freshly
/// populated L2 disk cache.
///
/// This test asserts the cure: after `decode_nuclide_kernels`, every
/// SVD kernel goes through `kernel::rehydrate_for_runtime` (just like
/// the live HDF5 load path) and produces **bit-identical XS lookups**
/// at off-grid energies as a kernel loaded directly from HDF5. If the
/// decoder ever drifts again, this test catches it without needing a
/// full ICSBEP sweep to surface the regression.
#[test]
fn decoded_kernel_lookup_matches_live_load_at_off_grid_energies() {
    use open_rust_mc::transport::simulate::XsProvider;
    use open_rust_mc::transport::xs_provider::SvdXsProvider;
    use std::sync::Arc;

    let Some(dir) = data_dir() else {
        eprintln!("[skip] ENDF data dir not found — set ICSBEP_DATA_DIR");
        return;
    };
    let path = dir.join("U235.h5");
    if !path.is_file() {
        return;
    }

    let live = load_nuclide(&path, 5, 0, 233.025, 2.43);
    let bytes = encode_nuclide_kernels(&live).unwrap();
    let decoded = decode_nuclide_kernels(&bytes).unwrap();

    // Build two providers, one with the live kernel, one with the
    // decoded kernel. Compare XS lookups at off-grid energies — these
    // are the exact queries the upper/lower-bracket bug corrupted.
    let live_p = SvdXsProvider {
        nuclides: vec![Arc::new(live)],
        thermal: vec![None],
    };
    let dec_p = SvdXsProvider {
        nuclides: vec![Arc::new(decoded)],
        thermal: vec![None],
    };

    // Energies deliberately placed BETWEEN grid points so the
    // bracket-convention bug manifests. Span the thermal → fast range
    // since heu-comp-inter-003 spectra extend across all four orders
    // of magnitude.
    let test_energies = [
        0.025_1, 0.099, 1.001, 9.99, 99.9, 1_000.3, 9_999.7, 1.5e5, 7.7e6,
    ];
    for &e in &test_energies {
        let a = live_p.lookup(0, e);
        let b = dec_p.lookup(0, e);
        // Bit-exact equality is achievable because both providers
        // walk the same hashed `LogHashIndex::lookup` once the
        // decoder calls `rehydrate_for_runtime`. Any difference at
        // this scale is the bracket-convention bug regressing.
        assert_eq!(
            a.total, b.total,
            "total XS mismatch at E = {e:.3e}: live = {} dec = {}",
            a.total, b.total,
        );
        assert_eq!(
            a.fission, b.fission,
            "fission XS mismatch at E = {e:.3e}",
        );
        assert_eq!(
            a.capture, b.capture,
            "capture XS mismatch at E = {e:.3e}",
        );
        assert_eq!(
            a.elastic, b.elastic,
            "elastic XS mismatch at E = {e:.3e}",
        );
    }
}
