// SPDX-License-Identifier: MIT
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

#![allow(dead_code)]

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
    long_about = "\
Render a top-down XY cross-section of an ICSBEP scene JSON.

Three output paths:

  - Interactive window (default, requires --features preview):
      Scroll wheel = multiplicative zoom around viewport centre
      Window drag  = pan
      Window resize = zoom (world bounds constant)
      R = reset to initial viewport
      L = toggle legend
      Escape = quit

  - Headless PNG  (--png-out <PATH>):
      Single static render. Use --resolution to control pixel size
      (4000×4000 gives ample detail for zoom-in via any image
      viewer; PNG compresses solid-colour regions to ~1-5 MB).
      Works without the `preview` feature.

  - Headless PPM (--ppm-out <PATH>):
      Same as PNG but raw RGB. Use if downstream tools require it.

Auto-viewport: walks geometry surfaces + lattices, centres on the
midpoint, samples outward to tighten when the explicit bounds are
loose. --zoom <factor> scales the result for scenes where the
fixture is small inside a large containment (PST-012)."
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

    /// z-slice the cross-section samples at (cm). When unset, the
    /// binary picks the midpoint of the geometry's z-extent — that
    /// fixes the "everything renders as void" symptom on stacked-can
    /// experiments (heu-met-fast-069, pu-sol-therm-012, ...) where
    /// the geometry sits entirely above or below z=0.
    #[arg(long)]
    z: Option<f64>,

    /// Render resolution (square, in pixels).
    #[arg(long, default_value_t = 900)]
    resolution: u32,

    /// Render the cross-section to a PPM file and exit (no window).
    /// Useful for headless debugging — the file can be opened in any
    /// image viewer / converted with `magick`. Bypasses the
    /// `preview` feature gate entirely so this works on a default
    /// `cargo run --bin preview_scene -- ... --ppm-out out.ppm`.
    #[arg(long)]
    ppm_out: Option<PathBuf>,

    /// Render the cross-section to a PNG file and exit (no window).
    /// PNG is preferred over PPM for geometry diagrams — solid-
    /// colour regions compress aggressively (a 4000×4000 PPM is
    /// 48 MB raw; the same content as PNG lands at ~2-5 MB). Open
    /// in any browser, image viewer, or converted IDE preview.
    #[arg(long)]
    png_out: Option<PathBuf>,

    /// Multiplier applied to the auto-computed half-size. Default
    /// `1.0` = the binary's best guess. Use values < 1 to zoom in
    /// when the auto-zoom shows the experimental fixture surrounded
    /// by a large pool / containment / building (e.g. PST-012 has
    /// concrete walls at ±655 cm but the actual solution is at
    /// ±64 cm — `--zoom 0.1` gives a ±65.5 cm view that shows the
    /// fixture details). Ignored when `--half-size` is explicit.
    #[arg(long, default_value_t = 1.0)]
    zoom: f64,

    /// Convenience companion to `--ppm-out`: emit ALSO a
    /// `<stem>_zoom<N>.ppm` per stage at the listed zoom factors.
    /// Useful for "show me the geometry at every interesting scale"
    /// without re-invoking the binary per zoom. Example:
    /// `--zoom-stages 0.5,0.1,0.02`. Each stage uses the same
    /// auto-centred bounds, scaled by the factor.
    #[arg(long, value_delimiter = ',')]
    zoom_stages: Vec<f64>,

    /// Print, for a 3×3 grid of sample positions across the viewport,
    /// what `find_cell_recursive` returns: the full CoordStack path
    /// (universe / cell_idx / lattice indices at each level) plus the
    /// deepest cell's fill. Read alongside the PPM render to confirm
    /// whether the lattice descent is producing distinct
    /// element-local coordinates per pixel (or returning the same
    /// pin for the whole lattice — the "stretched pin" symptom).
    #[arg(long)]
    debug_samples: bool,
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
        auto_color_from_name, LegendEntry, MaterialPalette, Viewport,
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
                // Prefer the upstream semantic colour when it's
                // RECOGNIZABLE (water → blue, concrete → tan, ...);
                // fall back to an index-cycled bright HSV palette
                // for everything else. This guarantees that two
                // unrelated materials in the same scene never collide
                // visually — without it, "Air" + "Stainless steel" +
                // "Steel (pool wall)" all mapped within ~30 RGB
                // distance and PST-012 rendered as undifferentiated
                // grey.
                semantic_or_index_color(&m.name, i)
                    .unwrap_or_else(|| fallback.colors.get(i).copied().unwrap_or(fallback.void))
            })
            .collect(),
        // Bright magenta for void — wholly outside any reasonable
        // material colour the engine could produce, so void pixels
        // are unambiguous and never confused with dark materials.
        void: [255, 0, 220],
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

    // ── Default viewport: explicit-bound box → outward-probe tightened.
    // Same algorithm as `render_ppm` (see comments there); shared
    // helpers so interactive and PPM modes produce pixel-identical
    // viewports by default.
    let bounds = world_bounds_xy(&geometry);
    let z_slice = args.z.unwrap_or_else(|| bounds.as_ref().map(|b| b.default_z()).unwrap_or(0.0));
    let initial = match (args.half_size, bounds.as_ref()) {
        (Some(h), _) => Viewport::square_centered(h, z_slice, args.resolution),
        (None, Some(b)) => {
            let cx = b.cx();
            let cy = b.cy();
            let rough_half = 0.5 * b.xy_extent();
            let origin = open_rust_mc::geometry::Vec3::new(cx, cy, 0.0);
            let probe_x_pos = tighten_along_axis(&geometry, origin,
                open_rust_mc::geometry::Vec3::new( 1.0,  0.0, 0.0), z_slice, rough_half);
            let probe_x_neg = tighten_along_axis(&geometry, origin,
                open_rust_mc::geometry::Vec3::new(-1.0,  0.0, 0.0), z_slice, rough_half);
            let probe_y_pos = tighten_along_axis(&geometry, origin,
                open_rust_mc::geometry::Vec3::new( 0.0,  1.0, 0.0), z_slice, rough_half);
            let probe_y_neg = tighten_along_axis(&geometry, origin,
                open_rust_mc::geometry::Vec3::new( 0.0, -1.0, 0.0), z_slice, rough_half);
            let tight = [probe_x_pos, probe_x_neg, probe_y_pos, probe_y_neg]
                .iter().fold(0.0_f64, |a, &b| a.max(b));
            let half_raw = if tight > 0.0 { tight * 1.05 } else { rough_half * 1.05 };
            let half = half_raw * args.zoom;
            Viewport {
                x_min: cx - half,
                x_max: cx + half,
                y_min: cy - half,
                y_max: cy + half,
                z_slice,
                width: args.resolution,
                height: args.resolution,
            }
        }
        (None, None) => Viewport::square_centered(10.0 * args.zoom, z_slice, args.resolution),
    };

    // ── Per-pixel render closure (parallelised across rows) ──
    //
    // Every scroll-wheel tick + window resize re-invokes this; on a
    // 900×900 viewport over a deep PWR lattice the serial walk was
    // ~600-800 ms per redraw, making zoom feel sluggish. Rayon over
    // rows takes that to ~100-150 ms on an 8-core CPU. Each row owns
    // its own intermediate Vec then we flatten — keeps the writers
    // independent (no shared mutable buffer).
    let render = move |vp: &Viewport| -> Vec<u32> {
        use rayon::prelude::*;
        let w = vp.width as usize;
        let h = vp.height as usize;
        let dx = (vp.x_max - vp.x_min) / vp.width as f64;
        let dy = (vp.y_max - vp.y_min) / vp.height as f64;
        let rows: Vec<Vec<u32>> = (0..vp.height).into_par_iter().map(|py| {
            let world_y = vp.y_max - (py as f64 + 0.5) * dy;
            (0..vp.width).map(|px| {
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
                ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
            }).collect()
        }).collect();
        let mut buf = Vec::with_capacity(w * h);
        for row in &rows {
            buf.extend_from_slice(row);
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
    show_window_cursor_zoom(initial, &title, legend, render);
}

/// Custom event loop with **cursor-centred zoom** (the upstream
/// `rust_mc_sim::preview::show_window` zooms around the viewport
/// midpoint, which makes inspecting off-centre features awkward —
/// you have to drag-pan to the feature, then zoom, then re-pan).
///
/// Differences from upstream:
///
/// - **Scroll wheel** zooms around the cursor position. The world
///   point under the cursor stays under the cursor after zoom,
///   matching every modern map app (Google Maps, OSM, Figma, ...).
/// - **Right-click + drag** pans the viewport. Upstream has no pan;
///   the only way to recenter was to resize the window
///   asymmetrically.
/// - Same `R` / `L` / `Escape` keybinds, same render closure
///   contract.
///
/// Geometry is borrowed via the closure, so this only runs under
/// the `preview` feature (the closure itself uses
/// `find_cell_recursive` which is always available).
#[cfg(feature = "preview")]
fn show_window_cursor_zoom<F>(
    initial: rust_mc_sim::preview::Viewport,
    title: &str,
    _legend: Vec<rust_mc_sim::preview::LegendEntry>,
    mut render: F,
)
where
    F: FnMut(&rust_mc_sim::preview::Viewport) -> Vec<u32>,
{
    use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
    use rust_mc_sim::preview::Viewport;

    let mut viewport = initial;
    let mut window = Window::new(
        title,
        viewport.width as usize,
        viewport.height as usize,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("preview_scene window: {e}"));
    window.set_target_fps(60);

    // Initial render.
    let mut buf = render(&viewport);
    window
        .update_with_buffer(&buf, viewport.width as usize, viewport.height as usize)
        .ok();

    let mut last_size = (viewport.width as usize, viewport.height as usize);
    let mut prev_r = false;
    let mut prev_mouse: Option<(f32, f32)> = None;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let cur_size = window.get_size();
        let mut needs_render = false;

        // Resize: keep px/cm constant so dragging the window edge
        // gives more world area at the same scale (upstream behaviour).
        if cur_size != last_size && cur_size.0 > 0 && cur_size.1 > 0 {
            let cx = (viewport.x_min + viewport.x_max) * 0.5;
            let cy = (viewport.y_min + viewport.y_max) * 0.5;
            let px_per_cm = (viewport.width as f64 / (viewport.x_max - viewport.x_min)).abs();
            let new_w_world = cur_size.0 as f64 / px_per_cm;
            let new_h_world = cur_size.1 as f64 / px_per_cm;
            viewport.x_min = cx - new_w_world * 0.5;
            viewport.x_max = cx + new_w_world * 0.5;
            viewport.y_min = cy - new_h_world * 0.5;
            viewport.y_max = cy + new_h_world * 0.5;
            viewport.width = cur_size.0 as u32;
            viewport.height = cur_size.1 as u32;
            last_size = cur_size;
            needs_render = true;
        }

        // Cursor-centred scroll-zoom. The world coordinate under the
        // cursor before and after the zoom is held constant — the
        // viewport bounds slide so the point under the mouse stays
        // pinned.
        //
        // Scroll-delta clamping: minifb passes the raw OS scroll
        // value through `get_scroll_wheel`. Precision mice / trackpads
        // on Windows emit many small-magnitude events per perceptual
        // notch (sy ≈ 0.1 each); raw `|sy| > 0` would zoom 50+× per
        // gesture and the viewport collapses into sub-pixel void.
        // Clamping `|sy|` ≤ 1 and using `0.85^|sy|` keeps perceptual
        // zoom roughly one notch per event, regardless of the OS
        // event-coalescing policy.
        if let Some((_, sy)) = window.get_scroll_wheel() {
            if sy.abs() > 0.05 {
                let mag = (sy.abs() as f64).min(1.0);
                let step = 0.85_f64.powf(mag);
                let factor = if sy > 0.0 { step } else { 1.0 / step };
                let (mx, my) = window
                    .get_mouse_pos(MouseMode::Discard)
                    .unwrap_or((viewport.width as f32 / 2.0, viewport.height as f32 / 2.0));
                let fx = (mx as f64 / viewport.width as f64).clamp(0.0, 1.0);
                let fy = (my as f64 / viewport.height as f64).clamp(0.0, 1.0);
                let world_w = viewport.x_max - viewport.x_min;
                let world_h = viewport.y_max - viewport.y_min;
                // World point currently under the cursor. y is flipped:
                // py=0 = top of screen = world y_max.
                let wx = viewport.x_min + fx * world_w;
                let wy = viewport.y_max - fy * world_h;
                let new_w = world_w * factor;
                let new_h = world_h * factor;
                // Floors to keep us out of the sub-pixel void where
                // floating-point precision degrades and find_cell
                // queries return nonsense. 1e-3 cm = 10 micrometres
                // is plenty for any realistic scene.
                const MIN_HALF_CM: f64 = 1.0e-3;
                if new_w > MIN_HALF_CM && new_h > MIN_HALF_CM {
                    // Recenter so (wx, wy) maps to the same (mx, my)
                    // screen position after the zoom.
                    viewport.x_min = wx - fx * new_w;
                    viewport.x_max = viewport.x_min + new_w;
                    viewport.y_max = wy + fy * new_h;
                    viewport.y_min = viewport.y_max - new_h;
                    needs_render = true;
                }
            }
        }

        // Right-click drag = pan. Holding the button and moving the
        // cursor slides the viewport by the corresponding world
        // delta. Left button intentionally reserved for selection /
        // future click-to-probe.
        let mouse_now = window.get_mouse_pos(MouseMode::Discard);
        if window.get_mouse_down(MouseButton::Right) {
            if let (Some((mx, my)), Some((pmx, pmy))) = (mouse_now, prev_mouse) {
                let dx_px = (mx - pmx) as f64;
                let dy_px = (my - pmy) as f64;
                let world_per_px_x = (viewport.x_max - viewport.x_min) / viewport.width as f64;
                let world_per_px_y = (viewport.y_max - viewport.y_min) / viewport.height as f64;
                let dx_world = dx_px * world_per_px_x;
                let dy_world = dy_px * world_per_px_y;
                viewport.x_min -= dx_world;
                viewport.x_max -= dx_world;
                viewport.y_min += dy_world; // y flipped
                viewport.y_max += dy_world;
                needs_render = true;
            }
            prev_mouse = mouse_now;
        } else {
            prev_mouse = None;
        }

        // R = reset.
        let r_now = window.is_key_down(Key::R);
        if r_now && !prev_r {
            viewport = initial;
            last_size = (initial.width as usize, initial.height as usize);
            needs_render = true;
        }
        prev_r = r_now;

        if needs_render {
            buf = render(&viewport);
        }
        window
            .update_with_buffer(&buf, viewport.width as usize, viewport.height as usize)
            .ok();
    }
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

// ── Feature-free debug renderer ─────────────────────────────────────
//
// Walks `find_cell_recursive` per pixel like the interactive preview,
// but writes the framebuffer to a PPM file instead of opening a
// window. Bypasses the `rust_mc_sim::preview` dependency entirely so
// this path runs on a default `cargo run` without the `preview`
// feature. Built for the lattice-expansion bug investigation: the
// PPM render is the ground truth we compare against the interactive
// preview, and `--debug-samples` prints `find_cell_recursive`'s
// CoordStack at known positions so we can tell whether the lattice
// descent is producing the right per-element local coords.

/// True (min_x, max_x, min_y, max_y) bounding box of the geometry,
/// walking finite surface AABBs + every lattice extent. Returns
/// `None` when nothing finite is in scope.
///
/// Why this isn't `max |bound|`: an earlier version of the auto-
/// viewport collapsed to `max(|min|, |max|)` and assumed geometry
/// centered at origin. That worked for symmetric scenes (PWR
/// assembly, Godiva) but broke wholesale on off-centre experiments
/// (LCT-008 LEU rod arrays, PST-012 Pu solutions, HMF-069 oralloy
/// cylinders) — those have their geometry offset from origin, the
/// `max |bound|` viewport stayed centred at (0, 0), and the entire
/// scene rendered as void.
/// Tighten an axis bound by sampling outward from a starting point.
///
/// Returns the last r along `dir` (in cm) at which `find_cell_recursive`
/// returns `Some`. Stops after [`AXIS_PROBE_MAX_MISS_RUN`] consecutive
/// misses — that's "you saw a big empty, the previous hit is the
/// real boundary". The motivation: explicit bounds (PlaneZ z0 values,
/// outer boundary surfaces) can be set hundreds of cm beyond the
/// actual fissile / structural material — e.g. LCT-008 has reflective
/// boundaries at ±280 cm but the actual rod array is ~30 cm across.
/// Auto-zooming to ±280 cm renders the rods as a single pixel.
///
/// Sampled at `AXIS_PROBE_STEPS` points spanning `0..max_extent` so
/// the resolution adapts to the rough explicit bound. With 200
/// steps a 30 cm geometry at ±280 cm explicit bound is sampled
/// every 1.4 cm — plenty of resolution to find the actual edge.
fn tighten_along_axis(
    geom: &open_rust_mc::geometry::Geometry,
    origin: open_rust_mc::geometry::Vec3,
    dir: open_rust_mc::geometry::Vec3,
    z_slice: f64,
    max_extent: f64,
) -> f64 {
    use open_rust_mc::geometry::ray::find_cell_recursive;
    let mut last_hit = 0.0_f64;
    let mut consecutive_misses = 0_usize;
    for i in 1..=AXIS_PROBE_STEPS {
        let r = max_extent * i as f64 / AXIS_PROBE_STEPS as f64;
        let probe = open_rust_mc::geometry::Vec3::new(
            origin.x + dir.x * r,
            origin.y + dir.y * r,
            z_slice,
        );
        if find_cell_recursive(probe, geom).is_some() {
            last_hit = r;
            consecutive_misses = 0;
        } else {
            consecutive_misses += 1;
            if consecutive_misses >= AXIS_PROBE_MAX_MISS_RUN {
                break;
            }
        }
    }
    last_hit
}

const AXIS_PROBE_STEPS: usize = 200;
const AXIS_PROBE_MAX_MISS_RUN: usize = 6;

struct GeomBounds {
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    /// Sorted distinct PlaneZ `z0` values. Used by `default_z` to
    /// pick a sensible slice when the explicit `(z_min, z_max)`
    /// midpoint lands in a void / air region (PST-012 has floor
    /// planes at z=-100 and ceiling planes at z=950, midpoint =
    /// 441 cm is in mid-air; median of the 13 distinct planes is
    /// 70 cm which IS inside the solution tank).
    z_plane_positions: Vec<f64>,
}

impl GeomBounds {
    fn xy_extent(&self) -> f64 {
        (self.x_max - self.x_min).max(self.y_max - self.y_min)
    }
    fn cx(&self) -> f64 { 0.5 * (self.x_min + self.x_max) }
    fn cy(&self) -> f64 { 0.5 * (self.y_min + self.y_max) }
    /// Sensible z-slice when the user didn't pass `--z`. Prefers the
    /// **midpoint of the two median consecutive PlaneZ positions**
    /// over the (z_min, z_max) midpoint:
    ///
    ///   - (z_min, z_max) midpoint can land in air when the geometry
    ///     has a far-floor + far-ceiling around the fixture (PST-012:
    ///     -103, ..., 100, 937, 987 → midrange 441 cm is room-height
    ///     air, not the solution).
    ///   - Picking one plane value exactly lands the slice ON a
    ///     boundary surface. The HalfSpace::evaluate function returns
    ///     0.0 on the surface; in our cell-find logic that's in
    ///     **neither** half-space, so `find_cell_recursive` returns
    ///     None and the pixel renders as void (PST-012 again: median
    ///     plane 69.57 cm is one of the actual z0 values, so the
    ///     entire centre rendered as void / magenta).
    ///
    /// Midpoint of two adjacent planes is always strictly between
    /// surfaces, so it sits cleanly inside some cell.
    ///
    /// Falls back to (z_min, z_max) midpoint when ≤ 1 plane exists,
    /// then to 0.
    fn default_z(&self) -> f64 {
        let zs = &self.z_plane_positions;
        if zs.len() >= 2 {
            // Find the SMALLEST gap between consecutive planes —
            // that's the densest region, almost always where the
            // experimental fixture lives. PST-012 plane gaps:
            //
            //   -103 -63 -51 -50.5 -0.5 0  69.6  80  81  99  100  937  987
            //       40  12  0.8    50  0.5 69.6  10  1   18  1    837  50
            //
            // Smallest gap is 0.5 between -0.5 and 0.0 — but those
            // are floor / pool-bottom layers. Picking the smallest-
            // gap midpoint there would render the floor instead of
            // the solution. So: among the smallest-K gaps, take the
            // one whose midpoint is closest to the geometry's
            // explicit (z_min + z_max) / 2 — biases toward the
            // CENTRE while still avoiding cells that lie on a
            // surface boundary.
            let mut gaps: Vec<(usize, f64)> = (0..zs.len() - 1)
                .map(|i| (i, zs[i + 1] - zs[i]))
                .collect();
            // Sort by gap size ascending; keep the smaller half.
            gaps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (gaps.len() / 2).max(1);
            let candidates = &gaps[..keep];
            let geom_centre_z = if self.z_min.is_finite() && self.z_max.is_finite() {
                0.5 * (self.z_min + self.z_max)
            } else {
                0.0
            };
            // Pick the candidate whose midpoint is closest to the
            // geometry centre.
            let best = candidates.iter()
                .map(|&(i, _)| (i, 0.5 * (zs[i] + zs[i + 1])))
                .min_by(|a, b| {
                    (a.1 - geom_centre_z).abs()
                        .partial_cmp(&(b.1 - geom_centre_z).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(_, mid)| mid)
                .unwrap_or(geom_centre_z);
            return best;
        }
        if let Some(&z) = zs.first() {
            return z;
        }
        if self.z_min.is_finite() && self.z_max.is_finite() {
            0.5 * (self.z_min + self.z_max)
        } else if self.z_min.is_finite() {
            self.z_min
        } else if self.z_max.is_finite() {
            self.z_max
        } else {
            0.0
        }
    }
}

fn world_bounds_xy(
    geom: &open_rust_mc::geometry::Geometry,
) -> Option<GeomBounds> {
    let mut b = GeomBounds {
        x_min: f64::INFINITY, x_max: f64::NEG_INFINITY,
        y_min: f64::INFINITY, y_max: f64::NEG_INFINITY,
        z_min: f64::INFINITY, z_max: f64::NEG_INFINITY,
        z_plane_positions: Vec::new(),
    };
    let mut touched = false;
    let update_x = |b: &mut GeomBounds, lo: f64, hi: f64| {
        if lo.is_finite() && lo < b.x_min { b.x_min = lo; }
        if hi.is_finite() && hi > b.x_max { b.x_max = hi; }
    };
    let update_y = |b: &mut GeomBounds, lo: f64, hi: f64| {
        if lo.is_finite() && lo < b.y_min { b.y_min = lo; }
        if hi.is_finite() && hi > b.y_max { b.y_max = hi; }
    };
    let update_z = |b: &mut GeomBounds, lo: f64, hi: f64| {
        if lo.is_finite() && lo < b.z_min { b.z_min = lo; }
        if hi.is_finite() && hi > b.z_max { b.z_max = hi; }
    };
    use open_rust_mc::geometry::surface::Surface;
    for s in &geom.surfaces {
        let aabb = s.aabb();
        if aabb.min.x.is_finite() || aabb.max.x.is_finite() {
            update_x(&mut b, aabb.min.x, aabb.max.x);
            touched = true;
        }
        if aabb.min.y.is_finite() || aabb.max.y.is_finite() {
            update_y(&mut b, aabb.min.y, aabb.max.y);
            touched = true;
        }
        if aabb.min.z.is_finite() || aabb.max.z.is_finite() {
            update_z(&mut b, aabb.min.z, aabb.max.z);
            touched = true;
        }
        // `PlaneZ::aabb` returns `Aabb::INFINITE` (the plane extends
        // infinitely in x,y) and therefore contributes NOTHING via
        // the aabb() path, even though its `z0` field is a finite
        // cutting plane. On axially stacked experiments
        // (heu-met-fast-069, pu-sol-therm-012) PlaneZ is the only
        // source of z info; without this extra pull, auto-z fell
        // through to 0.0 and the preview rendered the gap BELOW the
        // geometry as solid void. Same logic should apply to PlaneX
        // / PlaneY when those are the only finite axis source — leaf
        // for later, no scene ships that pattern today.
        match s {
            Surface::PlaneZ { z0, .. } => {
                update_z(&mut b, *z0, *z0);
                b.z_plane_positions.push(*z0);
                touched = true;
            }
            Surface::PlaneX { x0, .. } => {
                update_x(&mut b, *x0, *x0);
                touched = true;
            }
            Surface::PlaneY { y0, .. } => {
                update_y(&mut b, *y0, *y0);
                touched = true;
            }
            _ => {}
        }
    }
    // Dedup + sort for stable median lookup.
    b.z_plane_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    b.z_plane_positions.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    for lat in &geom.lattices {
        let x_hi = lat.origin.x + lat.shape[0] as f64 * lat.pitch.x;
        let y_hi = lat.origin.y + lat.shape[1] as f64 * lat.pitch.y;
        let z_hi = lat.origin.z + lat.shape[2] as f64 * lat.pitch.z;
        update_x(&mut b, lat.origin.x, x_hi);
        update_y(&mut b, lat.origin.y, y_hi);
        update_z(&mut b, lat.origin.z, z_hi);
        touched = true;
    }
    for hex in &geom.hex_lattices {
        let r = hex.n_rings as f64 * hex.pitch_xy;
        let z_hi = hex.center.z + hex.n_axial as f64 * hex.pitch_z;
        update_x(&mut b, hex.center.x - r, hex.center.x + r);
        update_y(&mut b, hex.center.y - r, hex.center.y + r);
        update_z(&mut b, hex.center.z, z_hi);
        touched = true;
    }
    touched.then_some(b)
}

/// Simple grayscale-ish palette built from a hash of the material
/// name. Independent of `rust_mc_sim::preview` so this works without
/// the `preview` feature.
fn auto_color(name: &str) -> [u8; 3] {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let r = ((h >> 24) & 0xff) as u8;
    let g = ((h >> 12) & 0xff) as u8;
    let b = (h & 0xff) as u8;
    // Avoid pure black (reserved for void) by floor-clamping.
    [r.max(32), g.max(32), b.max(32)]
}

/// 16-hue bright palette for material-index fallback. Hand-picked
/// for maximum perceptual separation across red / orange / yellow /
/// green / cyan / blue / purple / pink. Every entry has high
/// saturation + ≥ 60% lightness so cell boundaries are always
/// legible against neighbours and against the magenta void.
const INDEX_PALETTE: [[u8; 3]; 16] = [
    [220,  70,  70],  // red
    [240, 140,  50],  // orange
    [240, 200,  60],  // yellow
    [150, 210,  60],  // lime
    [ 60, 200, 100],  // green
    [ 60, 200, 180],  // teal
    [ 60, 160, 220],  // sky blue
    [ 80, 110, 220],  // blue
    [140,  80, 220],  // violet
    [200,  80, 200],  // pink
    [240, 160, 180],  // rose
    [180, 180, 100],  // olive
    [110, 180, 110],  // mint
    [180, 140,  80],  // sand
    [200, 200, 220],  // pearl
    [120, 140, 180],  // slate
];

/// Pick a material colour by SEMANTIC mapping first (water → blue,
/// concrete → tan, ...) then fall back to a fixed index-cycled
/// palette. Returns `None` only if the index palette is empty
/// (it never is) — the `Option` is kept so the call site can
/// distinguish "matched semantic" from "fell back to index" if it
/// ever needs to log palette decisions for debugging.
///
/// Bypasses `rust_mc_sim::preview::auto_color_from_name`'s default
/// of grey for Air / Steel / Stainless / Iron — those names all
/// map to ~`[110, 110, 120]` upstream which makes PST-012 (with
/// Air + Stainless + Steel + Concrete in adjacent cells) render
/// as a single grey blob.
fn semantic_or_index_color(name: &str, index: usize) -> Option<[u8; 3]> {
    let n = name.to_lowercase();
    // Strong semantic anchors only — colours that mean something
    // physically. Air gets an obviously-different colour from any
    // structural metal.
    if n.contains("water") && !n.contains("heavy") {
        return Some([ 80, 150, 230]);  // blue
    }
    if n.contains("heavy water") || n.contains("d2o") {
        return Some([ 40,  80, 180]);  // deep blue
    }
    if n.contains("concrete") {
        return Some([180, 160, 120]);  // tan
    }
    if n.contains("air") || n.contains("void") || n.contains("vacuum") {
        return Some([220, 230, 240]);  // pale blue-white — visibly
                                       // NOT a structural material
    }
    if n.contains("plutonium") || n.contains("mox") {
        return Some([240, 140,  50]);  // orange
    }
    if n.contains("uranium") || n.contains("uo2") || n.contains("fuel") {
        return Some([200,  80,  60]);  // red
    }
    // Everything else: cycle through the index palette so that
    // (e.g.) "Stainless steel" and "Steel (pool wall)" get DIFFERENT
    // colours instead of both ending up grey [110, 110, 120].
    Some(INDEX_PALETTE[index % INDEX_PALETTE.len()])
}

fn render_ppm(args: &Args, ppm_path: &Path) {
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use open_rust_mc::geometry::scene_io;
    use open_rust_mc::geometry::Vec3;
    use open_rust_mc::transport::material_resolve;
    use open_rust_mc::transport::nuclides::NuclideLibrary;

    let case_path = resolve_case_path(&args.case_json);
    let data_dir = resolve_data_dir(&args.data_dir);
    let text = std::fs::read_to_string(&case_path).unwrap_or_else(|e| {
        panic!("read {}: {e}", case_path.display())
    });
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("scene JSON parse failed");
    let scene = value
        .get("scene")
        .expect("case JSON has no `scene` block");
    let loaded = scene_io::load_scene_from_json(&scene.to_string())
        .expect("scene_io::load_scene_from_json failed");
    let lib = NuclideLibrary::from_data_dir(&data_dir);
    let resolved = material_resolve::resolve_materials(&loaded.materials, &lib, args.rank)
        .expect("material_resolve failed");
    let materials = &resolved.materials;
    let geometry = loaded.geometry;

    // Distinct-colour palette (same logic as the interactive path):
    // semantic colour when the name is recognized, otherwise a
    // cycle through 16 bright high-saturation hues by index. No two
    // materials map to the same colour even when they all hash to
    // similar greys (PST-012's Air / Stainless / Steel collision).
    let palette: Vec<[u8; 3]> = materials
        .iter()
        .enumerate()
        .map(|(i, m)| semantic_or_index_color(&m.name, i).unwrap_or_else(|| {
            // Final fallback: cycle through INDEX_PALETTE by index
            // mod len. semantic_or_index_color already does this when
            // its semantic-map misses, so this branch is unreachable
            // in practice — kept for type-safety.
            INDEX_PALETTE[i % INDEX_PALETTE.len()]
        }))
        .collect();
    // Bright magenta — outside any plausible material colour, so
    // void is unambiguous in the rendered image.
    let void = [255_u8, 0, 220];

    // Auto-viewport + auto-z + outward-probe tightening.
    //
    // Step 1: explicit bounds via `world_bounds_xy` (lattices,
    // surface AABBs, PlaneX/Y/Z z0). Step 2: pick z_slice = midpoint
    // of z-extent so stacked-can experiments render where geometry
    // exists. Step 3: sample outward from the bound centre on
    // +x/-x/+y/-y axes until we see a long enough miss run — that
    // tightens the box for cases where explicit bounds are wider
    // than the actual fissile / structural material (LCT-008 has
    // reflective boundaries at ±280 cm but the rod array is ~30 cm).
    let bounds = world_bounds_xy(&geometry);
    let z_slice = args.z.unwrap_or_else(|| bounds.as_ref().map(|b| b.default_z()).unwrap_or(0.0));
    let (x_min, x_max, y_min, y_max) = match (args.half_size, bounds.as_ref()) {
        (Some(h), _) => (-h, h, -h, h),
        (None, Some(b)) => {
            let cx = b.cx();
            let cy = b.cy();
            // Rough explicit half-extent (one-sided from centre).
            let rough_half = 0.5 * b.xy_extent();
            // Probe outward from (cx, cy) on each axis. The probe
            // returns the last r at which the geometry contains
            // something, so the actual edge along that direction.
            let origin = open_rust_mc::geometry::Vec3::new(cx, cy, 0.0);
            let dx_pos = open_rust_mc::geometry::Vec3::new(1.0, 0.0, 0.0);
            let dx_neg = open_rust_mc::geometry::Vec3::new(-1.0, 0.0, 0.0);
            let dy_pos = open_rust_mc::geometry::Vec3::new(0.0, 1.0, 0.0);
            let dy_neg = open_rust_mc::geometry::Vec3::new(0.0, -1.0, 0.0);
            let probe_x_pos = tighten_along_axis(&geometry, origin, dx_pos, z_slice, rough_half);
            let probe_x_neg = tighten_along_axis(&geometry, origin, dx_neg, z_slice, rough_half);
            let probe_y_pos = tighten_along_axis(&geometry, origin, dy_pos, z_slice, rough_half);
            let probe_y_neg = tighten_along_axis(&geometry, origin, dy_neg, z_slice, rough_half);
            // Take max of all 4 + 5% padding. If all probes returned
            // 0 (centre is void), fall back to the rough half — the
            // user can always pass --half-size explicitly.
            let tight = [probe_x_pos, probe_x_neg, probe_y_pos, probe_y_neg]
                .iter().fold(0.0_f64, |a, &b| a.max(b));
            let half_raw = if tight > 0.0 { tight * 1.05 } else { rough_half * 1.05 };
            let half = half_raw * args.zoom;
            (cx - half, cx + half, cy - half, cy + half)
        }
        (None, None) => (-10.0 * args.zoom, 10.0 * args.zoom,
                         -10.0 * args.zoom, 10.0 * args.zoom),
    };
    let res = args.resolution;
    let buf = render_frame(
        &geometry, &palette, void, x_min, x_max, y_min, y_max, z_slice, res,
    );

    // Optional sample-grid debug print BEFORE the file write so the
    // operator sees it on stderr even if the PPM write fails.
    if args.debug_samples {
        eprintln!("\n── find_cell_recursive samples (3×3 grid across viewport) ──");
        for j in 0..3 {
            for i in 0..3 {
                let sx = x_min + (i as f64 + 0.5) * (x_max - x_min) / 3.0;
                let sy = y_max - (j as f64 + 0.5) * (y_max - y_min) / 3.0;
                let pos = Vec3::new(sx, sy, z_slice);
                match find_cell_recursive(pos, &geometry) {
                    Some(stack) => {
                        let deepest = stack.last().map(|c| c.cell_idx).unwrap_or(0);
                        let fill = &geometry.cells[deepest as usize].fill;
                        // Format the CoordStack: each level's
                        // (universe, cell_idx, lattice_index?) tuple.
                        let path: Vec<String> = stack
                            .iter()
                            .map(|c| match c.lattice {
                                Some((lid, [ix, iy, iz])) => format!(
                                    "u{}/c{}/L{}[{},{},{}]",
                                    c.universe.0, c.cell_idx, lid.0, ix, iy, iz
                                ),
                                None => format!("u{}/c{}", c.universe.0, c.cell_idx),
                            })
                            .collect();
                        eprintln!(
                            "  ({sx:+8.3}, {sy:+8.3}) → depth {} : {}  fill={:?}",
                            stack.len(),
                            path.join(" → "),
                            fill,
                        );
                    }
                    None => eprintln!("  ({sx:+8.3}, {sy:+8.3}) → leak"),
                }
            }
        }
        eprintln!();
    }

    // Write the frame to disk in the requested format.
    write_image(ppm_path, &buf, res);
    eprintln!("wrote {} ({}×{})  half=±{:.2} cm  z={:.2}",
        ppm_path.display(), res, res, 0.5 * (x_max - x_min), z_slice);

    // Optional multi-stage emit. Each stage scales the auto-half by
    // a user-supplied factor and writes <stem>_zoom<factor>.ppm.
    // Skipped when `--half-size` was explicit (overriding the auto
    // half makes the stage semantics ambiguous).
    if args.half_size.is_some() || args.zoom_stages.is_empty() {
        return;
    }
    let (cx, cy) = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5);
    let base_half = 0.5 * (x_max - x_min) / args.zoom;
    for &factor in &args.zoom_stages {
        let stage_half = base_half * factor;
        let s_xmin = cx - stage_half;
        let s_xmax = cx + stage_half;
        let s_ymin = cy - stage_half;
        let s_ymax = cy + stage_half;
        let s_buf = render_frame(
            &geometry, &palette, void, s_xmin, s_xmax, s_ymin, s_ymax, z_slice, res,
        );
        let stem = ppm_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("preview");
        let parent = ppm_path.parent().unwrap_or_else(|| Path::new("."));
        let ext = ppm_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("ppm");
        let stage_path = parent.join(format!("{stem}_zoom{factor}.{ext}"));
        write_image(&stage_path, &s_buf, res);
        eprintln!("wrote {} ({}×{})  half=±{:.2} cm  z={:.2}",
            stage_path.display(), res, res, stage_half, z_slice);
    }
}

/// Parallel per-pixel render. Each row independently walks
/// `find_cell_recursive` over its pixels — no shared mutable state
/// makes this trivially `rayon::par_iter()`-able. On an 8-core
/// machine this gives ~5-7× wall-clock speedup over the serial loop,
/// which translates directly into snappier scroll-wheel zoom in the
/// interactive `--features preview` window (every zoom tick
/// re-renders, so the bottleneck is the per-frame compute).
///
/// Geometry borrowing: `Geometry` is `Sync` because all its fields
/// are read-only after construction; rayon happily fans it out
/// across worker threads. The palette / void color are tiny
/// `Copy`-able structures, also `Sync` trivially.
fn render_frame(
    geom: &open_rust_mc::geometry::Geometry,
    palette: &[[u8; 3]],
    void: [u8; 3],
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_slice: f64,
    res: u32,
) -> Vec<[u8; 3]> {
    use open_rust_mc::geometry::cell::CellFill;
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use open_rust_mc::geometry::Vec3;
    use rayon::prelude::*;

    let dx = (x_max - x_min) / res as f64;
    let dy = (y_max - y_min) / res as f64;
    let res_us = res as usize;

    // Render each row in parallel — collect Vec<Vec<[u8;3]>> then
    // flatten. The two-level Vec avoids needing to declare the full
    // framebuffer up front and lets each worker thread write into
    // its own allocation (cache-friendly).
    let rows: Vec<Vec<[u8; 3]>> = (0..res).into_par_iter().map(|py| {
        let world_y = y_max - (py as f64 + 0.5) * dy;
        (0..res).map(|px| {
            let world_x = x_min + (px as f64 + 0.5) * dx;
            let pos = Vec3::new(world_x, world_y, z_slice);
            match find_cell_recursive(pos, geom) {
                Some(stack) => {
                    let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                    match geom.cells[deepest].fill {
                        CellFill::Material(m) => palette
                            .get(m as usize)
                            .copied()
                            .unwrap_or(void),
                        _ => void,
                    }
                }
                None => void,
            }
        }).collect()
    }).collect();

    // Flatten Vec<Vec<…>> to Vec<…> in scan order. Pre-allocate so
    // we know the exact final capacity; `extend_from_slice` is a
    // single memcpy per row.
    let mut buf: Vec<[u8; 3]> = Vec::with_capacity(res_us * res_us);
    for row in &rows {
        buf.extend_from_slice(row);
    }
    buf
}

/// Write an RGB framebuffer to disk. Picks PPM (binary P6) or PNG
/// from the path's extension; PNG is encoded by the `png` crate
/// with default compression (level 6), which is plenty for the
/// solid-colour geometry diagrams the preview emits. PPM stays as
/// a fallback for the dep-free case (no `png` crate available
/// historically — kept for forward / backward compatibility and
/// for downstream tools that prefer raw RGB).
fn write_image(path: &Path, buf: &[[u8; 3]], res: u32) {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("ppm");
    if ext.eq_ignore_ascii_case("png") {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), res, res);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        // Flatten [[u8; 3]] → [u8] (already contiguous in memory).
        let flat: &[u8] = bytemuck::cast_slice(buf);
        writer.write_image_data(flat).expect("png data");
    } else {
        use std::io::Write as _;
        let mut out = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        write!(out, "P6\n{} {}\n255\n", res, res).unwrap();
        for px in buf {
            out.write_all(px).unwrap();
        }
    }
}

fn main() {
    let args = Args::parse();
    // Headless render: `--ppm-out` and `--png-out` both route through
    // `render_ppm` (despite the legacy name) since the format is
    // picked up from the file extension by `write_image`.
    if let Some(out) = args.png_out.as_deref().or(args.ppm_out.as_deref()) {
        render_ppm(&args, out);
        return;
    }
    run_preview(&args);
}
