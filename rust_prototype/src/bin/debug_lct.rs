//! Diagnostic: walk find_cell_recursive on LCT-008 at a handful of
//! world positions and dump the stack + leaf cell.fill so we can see
//! why initial_source can't resolve to a Material leaf.

use std::path::PathBuf;

use open_rust_mc::geometry::{Vec3, scene_io};
use open_rust_mc::geometry::cell::CellFill;
use open_rust_mc::geometry::ray::find_cell_recursive;

fn main() {
    let mut p: PathBuf = env!("CARGO_MANIFEST_DIR").into();
    while p.parent().is_some() && !p.join("bench/icsbep").is_dir() {
        p = p.parent().unwrap().to_path_buf();
    }
    let case_file = p.join("bench/icsbep/leu-comp-therm-008_case-1.json");
    let text = std::fs::read_to_string(&case_file).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let loaded = scene_io::load_scene_from_json(&value["scene"].to_string()).unwrap();
    let geom = &loaded.geometry;

    println!("Geometry: {} cells, {} universes, {} rect lattices, root_universe = {:?}",
             geom.cells.len(), geom.universes.len(), geom.lattices.len(), geom.root_universe);
    println!("Surfaces:");
    for (i, s) in geom.surfaces.iter().enumerate() {
        println!("  s{i}: {s:?}");
    }
    println!("Per-universe surfaces:");
    for (u_idx, surfs) in geom.universe_surfaces.iter().enumerate() {
        let cells: Vec<usize> = geom.universes[u_idx].cell_indices.iter().copied().collect();
        let has_bvh = geom.universe_bvhs[u_idx].is_some();
        println!("  univ {u_idx}: cells {cells:?}, surfaces {surfs:?}, has_bvh = {has_bvh}");
    }
    println!("Cells:");
    for (i, c) in geom.cells.iter().enumerate() {
        println!("  cell {i}: aabb = {:?}, fill = {:?}", c.aabb, c.fill);
    }

    // Verify root cell's region predicates fire at origin.
    let pos = Vec3::new(0.0, 0.0, 0.0);
    let mut evals = vec![0.0_f64; geom.surfaces.len()];
    for (i, s) in geom.surfaces.iter().enumerate() {
        evals[i] = s.evaluate(pos);
    }
    println!("Surface evals at origin (world):");
    for (i, e) in evals.iter().enumerate() {
        println!("  s{i}: eval = {e:+.4}");
    }
    let root_cell = &geom.cells[28];
    println!("Cell 28 (root): region = {:?}", root_cell.region);
    println!("Cell 28 contains origin? {}", root_cell.contains(&evals));

    let test_points = [
        Vec3::new(0.0, 0.0, 0.0),         // origin — central pin
        Vec3::new(5.0, 5.0, 0.0),         // peripheral
        Vec3::new(0.4, 0.0, 0.0),         // inside one pin radius
        Vec3::new(2.0, 0.0, 0.0),         // in water between pins
        Vec3::new(40.0, 0.0, 0.0),        // far peripheral
        Vec3::new(80.0, 0.0, 0.0),        // near cylinder boundary
        Vec3::new(0.0, 0.0, 80.0),        // top
        Vec3::new(0.0, 0.0, -80.0),       // bottom
    ];

    for pos in test_points {
        let result = find_cell_recursive(pos, geom);
        print!("pos = ({:+.2}, {:+.2}, {:+.2}) -> ", pos.x, pos.y, pos.z);
        match result {
            Some(stack) => {
                print!("stack[{}]: ", stack.len());
                for frame in &stack {
                    print!("[univ {:?} cell {}] ", frame.universe, frame.cell_idx);
                }
                let deepest = stack.last().unwrap().cell_idx as usize;
                if let Some(c) = geom.cells.get(deepest) {
                    println!("leaf fill = {:?}", c.fill);
                } else {
                    println!("DEEPEST {deepest} OUT OF RANGE (n_cells={})", geom.cells.len());
                }
            }
            None => println!("None (no cell)"),
        }
    }
}
