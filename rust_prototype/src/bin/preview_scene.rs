//! Interactive preview window for any scene JSON in `bench/icsbep/`
//! (or any compatible scene spec). Mirrors `pwr_assembly --preview`
//! but pulls the geometry from `scene_io::load_scene_from_json`
//! instead of a hand-built `Geometry`, so the same machinery covers
//! every ICSBEP case, the engine-internal PWR / 17x17 / Godiva
//! scenes, and anything else the Python `run_icsbep_case` path can
//! consume.
//!
//! Requires the `preview` feature (gates the `rust_mc_sim::preview`
//! module):
//!
//!     cargo run --release --features preview --bin preview_scene -- \
//!         bench/icsbep/pwr_assembly_17x17.json \
//!         data/endfb-vii.1-hdf5/neutron
//!
//! The window walks `find_cell_recursive` per pixel and colours each
//! material from a name-derived palette. Pan / zoom / close via the
//! same controls `pwr_assembly --preview` uses.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use clap::Parser;

/// Resolve `name` to an existing file. Tries (in order):
///   1. The path as given.
///   2. `<repo>/bench/icsbep/<name>` and `…/<name>.json` — where
///      `<repo>` is the first ancestor of CWD (or the executable's
///      parent) that contains a `bench/icsbep` directory. Matches
///      the Python sweep script's `find_repo_root` logic so users
///      don't have to think about cwd.
fn resolve_case_path(name: &Path) -> PathBuf {
    if name.exists() {
        return name.to_path_buf();
    }
    let candidates_relative_to = [
        std::env::current_dir().ok(),
        std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)),
    ];
    for start in candidates_relative_to.into_iter().flatten() {
        let mut cur: Option<&Path> = Some(&start);
        while let Some(p) = cur {
            let bench = p.join("bench").join("icsbep");
            if bench.is_dir() {
                let direct = bench.join(name);
                if direct.exists() {
                    return direct;
                }
                let with_ext = bench.join(format!(
                    "{}.json",
                    name.file_name().and_then(|s| s.to_str()).unwrap_or("?")
                ));
                if with_ext.exists() {
                    return with_ext;
                }
            }
            cur = p.parent();
        }
    }
    // Fall through with the original path; the caller's `read_to_string`
    // will surface a clear error.
    name.to_path_buf()
}

/// Same trick for the HDF5 data directory.
fn resolve_data_dir(name: &Path) -> PathBuf {
    if name.is_dir() {
        return name.to_path_buf();
    }
    let candidates_relative_to = [
        std::env::current_dir().ok(),
        std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)),
    ];
    for start in candidates_relative_to.into_iter().flatten() {
        let mut cur: Option<&Path> = Some(&start);
        while let Some(p) = cur {
            let candidate = p.join(name);
            if candidate.is_dir() {
                return candidate;
            }
            cur = p.parent();
        }
    }
    name.to_path_buf()
}

#[derive(Parser, Debug)]
#[command(
    name = "preview_scene",
    about = "Interactive XY-cross-section preview of an ICSBEP / internal scene JSON",
)]
struct Args {
    /// Path to the scene JSON (e.g. bench/icsbep/pwr_assembly_17x17.json).
    case_json: PathBuf,

    /// Directory holding the ENDF HDF5 neutron files (used only to
    /// resolve materials — the preview itself does no transport).
    data_dir: PathBuf,

    /// SVD rank for nuclide loading. Has no effect on the preview
    /// image itself; lower values save load time when previewing many
    /// scenes back-to-back.
    #[arg(long, default_value_t = 1)]
    rank: usize,

    /// Override the default initial viewport half-size (cm). When
    /// unset, the binary picks ~1.05 × the geometry's enclosing
    /// surface AABB.
    #[arg(long)]
    half_size: Option<f64>,

    /// z-slice the cross-section samples at (cm). Default 0.
    #[arg(long, default_value_t = 0.0)]
    z: f64,

    /// Render resolution (square, in pixels).
    #[arg(long, default_value_t = 900)]
    resolution: u32,
}

#[cfg(feature = "preview")]
fn run_preview(args: &Args) {
    use open_rust_mc::geometry::cell::CellFill;
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use open_rust_mc::geometry::scene_io;
    use open_rust_mc::geometry::Vec3;
    use open_rust_mc::transport::material::Material;
    use open_rust_mc::transport::material_resolve;
    use open_rust_mc::transport::nuclides::NuclideLibrary;
    use rust_mc_sim::preview::{
        auto_color_from_name, show_window, LegendEntry, MaterialPalette, Viewport,
    };

    let case_path = resolve_case_path(&args.case_json);
    let data_dir = resolve_data_dir(&args.data_dir);
    let text = std::fs::read_to_string(&case_path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\n  (resolved from {})",
            case_path.display(),
            args.case_json.display()
        )
    });
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("scene JSON parse failed");
    let scene = value
        .get("scene")
        .expect("case JSON has no `scene` block — this is a CLI-runner manifest");

    let loaded = scene_io::load_scene_from_json(&scene.to_string())
        .expect("scene_io::load_scene_from_json failed");
    let lib = NuclideLibrary::from_data_dir(&data_dir);
    let resolved: material_resolve::ResolvedMaterials =
        material_resolve::resolve_materials(&loaded.materials, &lib, args.rank)
            .expect("material_resolve failed");
    let materials: &[Material] = &resolved.materials;
    let geometry = loaded.geometry;

    // ── Material palette (name-derived) + legend ────────────────────
    let fallback = MaterialPalette::default();
    let palette = MaterialPalette {
        colors: materials
            .iter()
            .enumerate()
            .map(|(i, m)| {
                auto_color_from_name(&m.name)
                    .unwrap_or_else(|| fallback.colors.get(i).copied().unwrap_or(fallback.void))
            })
            .collect(),
        void: fallback.void,
    };
    let legend: Vec<LegendEntry> = materials
        .iter()
        .enumerate()
        .map(|(i, m)| {
            LegendEntry::new(
                m.name.clone(),
                palette.colors.get(i).copied().unwrap_or(palette.void),
            )
        })
        .collect();

    // ── Default viewport: union of all surface AABBs (axis-aligned). ──
    // Spheres / cylinders constrain at least two axes; ignore the
    // unconstrained ones for the initial framing and fall back to
    // ±10 cm if nothing is finite.
    let half = args.half_size.unwrap_or_else(|| {
        let mut bound = 0.0_f64;
        for s in &geometry.surfaces {
            let aabb = s.aabb();
            for v in [
                aabb.min.x.abs(),
                aabb.max.x.abs(),
                aabb.min.y.abs(),
                aabb.max.y.abs(),
            ] {
                if v.is_finite() && v > bound {
                    bound = v;
                }
            }
        }
        if bound <= 0.0 {
            10.0
        } else {
            1.05 * bound
        }
    });
    let initial = Viewport::square_centered(half, args.z, args.resolution);

    // ── Per-pixel render closure (matches pwr_assembly's structure) ──
    let render = move |vp: &Viewport| -> Vec<u32> {
        let w = vp.width as usize;
        let h = vp.height as usize;
        let dx = (vp.x_max - vp.x_min) / vp.width as f64;
        let dy = (vp.y_max - vp.y_min) / vp.height as f64;
        let mut buf = vec![0u32; w * h];
        for py in 0..vp.height {
            let world_y = vp.y_max - (py as f64 + 0.5) * dy;
            for px in 0..vp.width {
                let world_x = vp.x_min + (px as f64 + 0.5) * dx;
                let pos = Vec3::new(world_x, world_y, vp.z_slice);
                let color = match find_cell_recursive(pos, &geometry) {
                    Some(stack) => {
                        let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                        match geometry.cells[deepest].fill {
                            CellFill::Material(m) => palette
                                .colors
                                .get(m as usize)
                                .copied()
                                .unwrap_or(palette.void),
                            _ => palette.void,
                        }
                    }
                    None => palette.void,
                };
                let [r, g, b] = color;
                buf[(py as usize) * w + (px as usize)] =
                    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        }
        buf
    };

    let title = format!(
        "preview_scene — {}",
        case_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?"),
    );
    show_window(initial, &title, legend, render);
}

#[cfg(not(feature = "preview"))]
fn run_preview(_args: &Args) {
    eprintln!(
        "preview_scene requires the `preview` feature. Re-run with:\n\
         cargo run --release --features preview --bin preview_scene -- \
         <scene.json> <data_dir>"
    );
    std::process::exit(2);
}

fn main() {
    let args = Args::parse();
    run_preview(&args);
}
