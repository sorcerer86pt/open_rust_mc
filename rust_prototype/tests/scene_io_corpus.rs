// SPDX-License-Identifier: MIT
//! Smoke test: load every `bench/icsbep/*.json` case file through
//! `geometry::scene_io::load_scene_from_json` and report what fraction
//! the deserializer accepts.
//!
//! This is the binary `cargo test` equivalent of running the full
//! ICSBEP regression — except all it does is exercise the JSON
//! deserializer, not the transport engine. Cases are tagged
//! `#[ignore]` so the default `cargo test` run stays fast; full corpus
//! sweep with `cargo test --release --ignored scene_io_corpus`.

use std::path::Path;

use open_rust_mc::geometry::scene_io;

/// Walk `bench/icsbep/*.json`, parse each, count successes vs failures.
/// Currently expects ≥ 350 of the 367 imported cases to load — the
/// stragglers are tests that intentionally have only the `benchmark`
/// block (our hand-authored `runner`-based cases, no `scene`) and will
/// fail to parse as a full SceneDto.
#[test]
#[ignore = "corpus sweep — opt in via --ignored"]
fn full_icsbep_corpus_round_trips() {
    let bench_dir = workspace_root().join("bench").join("icsbep");
    let entries: Vec<_> = std::fs::read_dir(&bench_dir)
        .expect("bench/icsbep dir present")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    assert!(
        !entries.is_empty(),
        "no *.json case files found under {}",
        bench_dir.display()
    );

    let mut n_with_scene = 0;
    let mut n_loaded = 0;
    let mut n_no_scene = 0;
    let mut failures: Vec<(String, String)> = Vec::new();
    for path in &entries {
        let text = std::fs::read_to_string(path).expect("read case file");
        // Skip hand-authored cases that have only a benchmark+runner
        // block (no `scene`) — they're not in scope for this test.
        let probe: serde_json::Value =
            serde_json::from_str(&text).expect("each case file is valid JSON");
        if probe.get("scene").is_none() {
            n_no_scene += 1;
            continue;
        }
        n_with_scene += 1;
        let scene_str = probe.get("scene").unwrap().to_string();
        match scene_io::load_scene_from_json(&scene_str) {
            Ok(_) => n_loaded += 1,
            Err(e) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                failures.push((name, format!("{e}")));
            }
        }
    }

    println!(
        "scene-deserialization corpus: {} total files, {} with scene block, \
         {} without (hand-authored runner cases), {} loaded successfully, \
         {} failures",
        entries.len(),
        n_with_scene,
        n_no_scene,
        n_loaded,
        failures.len(),
    );
    for (name, reason) in failures.iter().take(20) {
        println!("  FAIL {name}: {reason}");
    }
    if failures.len() > 20 {
        println!("  ... and {} more failures", failures.len() - 20);
    }
    // Target: every scene-bearing case loads. If the deserializer is
    // stable this assertion holds; if mit-crpg ships a new surface
    // type or CSG operator, that's a known regression and worth
    // flagging here.
    assert!(
        failures.is_empty(),
        "{} of {} scene-bearing cases failed to deserialize",
        failures.len(),
        n_with_scene,
    );
}

/// Walk up from `CARGO_MANIFEST_DIR` until we find the repo root
/// (the directory containing `bench/`). Lets the test run regardless
/// of whether `cargo test` is invoked from the rust_prototype subdir
/// or the repo root.
fn workspace_root() -> std::path::PathBuf {
    let mut p: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    loop {
        if p.join("bench").join("icsbep").exists() {
            return p;
        }
        if !p.pop() {
            panic!(
                "could not locate repo root with bench/icsbep starting from {:?}",
                Path::new(env!("CARGO_MANIFEST_DIR"))
            );
        }
    }
}
