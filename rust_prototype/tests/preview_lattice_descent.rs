// SPDX-License-Identifier: MIT
//! Regression test for the historical "stretched pin" preview bug.
//!
//! Symptom (now fixed): `find_cell_recursive` on a JSON-loaded scene
//! with a `RectLattice` always returned the same coord stack for every
//! probe inside the lattice — so the renderer drew one pin universe
//! stretched across the full lattice extent instead of the 17×17 grid.
//!
//! Two ways to detect the regression by probing `find_cell_recursive`
//! directly:
//!
//! 1. Probes in **different lattice elements** must report different
//!    `(lattice_id, ix, iy, iz)` tuples in their coord stack. If the
//!    descent always returns element (0,0,0) regardless of world
//!    position, this fails first.
//!
//! 2. Probes near **the pin's centre vs the pin's edge** must report
//!    **different leaf cell indices** (centre → fuel; edge → water).
//!    If `local_pos` isn't being properly translated per-element, every
//!    probe lands at the universe origin → always fuel.
//!
//! Test fixture: `bench/icsbep/pwr_assembly_17x17.json` (Westinghouse
//! 17×17, 264 fuel pins + 24 guide tubes + 1 instrument tube). Pitch
//! 1.26 cm. Element (0,0,0) centre = (-10.08, -10.08); element (16,16,0)
//! centre = (+10.08, +10.08).

use std::path::PathBuf;

use open_rust_mc::geometry::Vec3;
use open_rust_mc::geometry::cell::CellFill;
use open_rust_mc::geometry::ray::find_cell_recursive;
use open_rust_mc::geometry::scene_io::load_scene_from_json;

fn pwr_assembly_path() -> PathBuf {
    // Walk up from CARGO_MANIFEST_DIR until we find bench/icsbep.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    p.join("bench/icsbep/pwr_assembly_17x17.json")
}

#[test]
fn lattice_descent_resolves_distinct_elements() {
    let path = pwr_assembly_path();
    let text = std::fs::read_to_string(&path).expect("read scene JSON");
    // ICSBEP case JSONs wrap the SceneDto under a `scene` key alongside
    // benchmark metadata. scene_io::load_scene_from_json expects the
    // SceneDto at the top level — extract the inner `scene` first.
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("scene JSON parse");
    let scene = value.get("scene").expect("case JSON has no `scene` block");
    let loaded = load_scene_from_json(&scene.to_string()).expect("scene_io");
    let geom = &loaded.geometry;

    // Pitch = 1.26 cm, 17×17, origin -10.71 (= -17*0.5*1.26). Element
    // (i, j, 0) centre = origin + (i+0.5)*pitch = -10.71 + (i+0.5)*1.26.
    let pitch = 1.26;
    let origin = -17.0 * 0.5 * pitch;
    let elem_center = |i: i32, j: i32| -> Vec3 {
        Vec3::new(
            origin + (i as f64 + 0.5) * pitch,
            origin + (j as f64 + 0.5) * pitch,
            0.0,
        )
    };

    // Probe the centre of element (0,0), (16,0), and (8,8). Different
    // elements; the lattice indices in the coord stack must differ.
    let s00 = find_cell_recursive(elem_center(0, 0), geom).expect("stack at (0,0)");
    let s_far = find_cell_recursive(elem_center(16, 0), geom).expect("stack at (16,0)");
    let s_mid = find_cell_recursive(elem_center(8, 8), geom).expect("stack at (8,8)");

    let lat_idx = |stack: &open_rust_mc::geometry::CoordStack| {
        stack.iter().find_map(|c| c.lattice.map(|(_, ijk)| ijk))
    };

    let l00 = lat_idx(&s00).expect("stack at (0,0) must report a lattice element");
    let l_far = lat_idx(&s_far).expect("stack at (16,0) must report a lattice element");
    let l_mid = lat_idx(&s_mid).expect("stack at (8,8) must report a lattice element");

    // The diagnostic that would fail under the "stretched pin" bug.
    assert_ne!(
        l00, l_far,
        "lattice descent collapsed: element (0,0) and (16,0) both \
         reported {:?} — every pixel in the lattice would render as \
         the same pin universe.",
        l00,
    );
    assert_ne!(
        l00, l_mid,
        "lattice descent collapsed between (0,0) and (8,8): both {:?}",
        l00,
    );
    assert_eq!(l00, [0, 0, 0], "element (0,0,0) centre should resolve to lattice index [0,0,0]");
    assert_eq!(l_far, [16, 0, 0], "element (16,0,0) centre should resolve to [16,0,0]");
    assert_eq!(l_mid, [8, 8, 0], "element (8,8,0) centre should resolve to [8,8,0]");
}

#[test]
fn lattice_descent_resolves_pin_internals() {
    let path = pwr_assembly_path();
    let text = std::fs::read_to_string(&path).expect("read scene JSON");
    // ICSBEP case JSONs wrap the SceneDto under a `scene` key alongside
    // benchmark metadata. scene_io::load_scene_from_json expects the
    // SceneDto at the top level — extract the inner `scene` first.
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("scene JSON parse");
    let scene = value.get("scene").expect("case JSON has no `scene` block");
    let loaded = load_scene_from_json(&scene.to_string()).expect("scene_io");
    let geom = &loaded.geometry;

    let pitch = 1.26;
    let origin = -17.0 * 0.5 * pitch;
    let elem_center = |i: i32, j: i32| -> Vec3 {
        Vec3::new(
            origin + (i as f64 + 0.5) * pitch,
            origin + (j as f64 + 0.5) * pitch,
            0.0,
        )
    };

    // Probe at the centre of a fuel pin element (deep inside the
    // fuel cylinder, radius ~0.41 cm) vs a point just inside the
    // same element but well outside the cylinder (offset 0.6 cm from
    // centre — beyond the 0.475 cm clad outer radius). These must
    // resolve to different leaf cells.
    let center_pos = elem_center(5, 5);
    let mut edge_pos = center_pos;
    edge_pos.x += 0.6; // still inside element extent of 1.26 cm

    let s_center = find_cell_recursive(center_pos, geom).expect("stack at element centre");
    let s_edge = find_cell_recursive(edge_pos, geom).expect("stack at element edge");

    let leaf_cell = |s: &open_rust_mc::geometry::CoordStack| {
        s.last().map(|c| c.cell_idx).unwrap_or(u32::MAX)
    };
    let c_center = leaf_cell(&s_center);
    let c_edge = leaf_cell(&s_edge);

    assert_ne!(
        c_center, c_edge,
        "pin-universe descent collapsed: probes at element centre and edge \
         both resolved to cell {} — find_cell_recursive isn't translating \
         local_pos per-element. The renderer would show one stretched pin \
         across the entire lattice element.",
        c_center,
    );

    // Centre and edge must resolve to **different materials**. Material
    // indices vary between JSON exports (fuel is sometimes 0, sometimes
    // 3, etc.), so the assertion is "distinct materials", not "specific
    // material IDs". The stretched-pin bug would produce identical
    // materials here because every probe lands at the pin-universe
    // origin.
    let center_fill = &geom.cells[c_center as usize].fill;
    let edge_fill = &geom.cells[c_edge as usize].fill;
    match (center_fill, edge_fill) {
        (CellFill::Material(a), CellFill::Material(b)) if a != b => {}
        other => panic!(
            "pin-universe descent produced identical materials at \
             element centre vs edge: {:?}. Local-pos translation per \
             element is broken — the renderer would show a stretched \
             pin across the entire element.",
            other,
        ),
    }
}
