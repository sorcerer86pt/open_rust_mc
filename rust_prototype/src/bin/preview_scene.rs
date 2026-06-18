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

    /// Render the cross-section in the *terminal* instead of a Win32
    /// window (requires `--features tui`). Half-block truecolor cells
    /// give 2 vertical pixels per character row; pan / zoom / z-slice
    /// with the keyboard or mouse. Works over SSH / on headless GPU
    /// boxes where the minifb window can't open. By default this
    /// launches in a NEW console window so the current shell stays
    /// free — pass `--inline` to take over the current terminal.
    #[arg(long)]
    tui: bool,

    /// With `--tui`, run the terminal viewer in the CURRENT terminal
    /// rather than spawning a new console window. Useful inside tmux /
    /// an IDE terminal, or when the new-window spawn isn't wanted.
    #[arg(long)]
    inline: bool,

    /// Interactive 3D view of the geometry (requires `--features
    /// preview`). Ray-casts the actual CSG with the engine's geometry
    /// walk, shading each material by colour + lighting. Orbit with
    /// left-drag, zoom with the scroll wheel, pan with right-drag,
    /// `r` resets, Escape quits.
    #[arg(long = "3d")]
    three_d: bool,

    /// Headless 3D render to a PNG (no window). Ray-casts from the
    /// camera angle given by `--cam-azim` / `--cam-elev` at
    /// `--resolution`. Works without the `preview` feature — usable
    /// over SSH / on headless boxes.
    #[arg(long = "render3d-out")]
    render3d_out: Option<PathBuf>,

    /// Camera azimuth (degrees around +z) for the 3D view's initial /
    /// headless angle.
    #[arg(long, default_value_t = 35.0)]
    cam_azim: f64,

    /// Camera elevation (degrees above the xy-plane) for the 3D view's
    /// initial / headless angle.
    #[arg(long, default_value_t = 25.0)]
    cam_elev: f64,
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

/// Everything `preview_scene` needs to draw a cross-section, loaded
/// once: the resolved geometry, an index→colour palette, the void
/// colour, and the per-material names (for the TUI legend). Shared by
/// the PPM/PNG renderer and the `--tui` terminal viewer so both paths
/// produce identical colours from identical inputs.
struct LoadedPreview {
    geometry: open_rust_mc::geometry::Geometry,
    palette: Vec<[u8; 3]>,
    void: [u8; 3],
    names: Vec<String>,
}

/// Load a scene JSON + ENDF materials and build the distinct-colour
/// palette. Factored out of `render_ppm` so the terminal viewer
/// reuses the exact same colour logic (semantic anchor → 16-hue
/// index cycle) without duplicating the load/resolve/palette code.
fn load_preview(args: &Args) -> LoadedPreview {
    use open_rust_mc::geometry::scene_io;
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
    let names: Vec<String> = materials.iter().map(|m| m.name.clone()).collect();

    LoadedPreview {
        geometry: loaded.geometry,
        palette,
        // Bright magenta — outside any plausible material colour, so
        // void is unambiguous in the rendered image.
        void: [255, 0, 220],
        names,
    }
}

/// Auto-viewport + auto-z + outward-probe tightening, returning
/// `(x_min, x_max, y_min, y_max, z_slice)`.
///
/// Step 1: explicit bounds via `world_bounds_xy` (lattices, surface
/// AABBs, PlaneX/Y/Z z0). Step 2: pick z_slice = densest-plane-gap
/// midpoint so stacked-can experiments render where geometry exists.
/// Step 3: sample outward from the bound centre on +x/-x/+y/-y axes
/// until we see a long enough miss run — that tightens the box for
/// cases where explicit bounds are wider than the actual fissile /
/// structural material (LCT-008 has reflective boundaries at ±280 cm
/// but the rod array is ~30 cm).
///
/// `--half-size` overrides everything; `--zoom` scales the auto half.
/// Shared by `render_ppm` and the terminal viewer's initial fit.
fn auto_bounds(
    geometry: &open_rust_mc::geometry::Geometry,
    args: &Args,
) -> (f64, f64, f64, f64, f64) {
    let bounds = world_bounds_xy(geometry);
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
            let probe_x_pos = tighten_along_axis(geometry, origin, dx_pos, z_slice, rough_half);
            let probe_x_neg = tighten_along_axis(geometry, origin, dx_neg, z_slice, rough_half);
            let probe_y_pos = tighten_along_axis(geometry, origin, dy_pos, z_slice, rough_half);
            let probe_y_neg = tighten_along_axis(geometry, origin, dy_neg, z_slice, rough_half);
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
    (x_min, x_max, y_min, y_max, z_slice)
}

fn render_ppm(args: &Args, ppm_path: &Path) {
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use open_rust_mc::geometry::Vec3;

    let LoadedPreview { geometry, palette, void, names: _ } = load_preview(args);
    let (x_min, x_max, y_min, y_max, z_slice) = auto_bounds(&geometry, args);
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

/// Env guard that marks the process as the already-spawned TUI child.
/// Set on the child when `preview_scene --tui` re-execs itself into a
/// new console window; the child sees it and runs the viewer inline
/// instead of spawning yet another window (infinite-respawn guard).
const TUI_CHILD_ENV: &str = "PREVIEW_SCENE_TUI_CHILD";

/// Re-exec this binary with the same arguments inside a brand-new
/// console window, then return `true` so the caller (the parent) can
/// exit. The child carries [`TUI_CHILD_ENV`] so it runs the viewer
/// inline rather than spawning again.
///
/// Windows: `CREATE_NEW_CONSOLE` (0x10) gives the child its own
/// console window with fresh stdin/stdout — exactly what a TUI needs.
/// Other platforms have no portable "new terminal window" primitive,
/// so we report that and let the caller fall back to running inline.
#[cfg(feature = "tui")]
fn spawn_in_new_window() -> bool {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("preview_scene: cannot find own exe to relaunch ({e}); running inline");
            return false;
        }
    };
    let forwarded: Vec<String> = std::env::args().skip(1).collect();

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NEW_CONSOLE — child gets its own console window.
        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
        match std::process::Command::new(&exe)
            .args(&forwarded)
            .env(TUI_CHILD_ENV, "1")
            .creation_flags(CREATE_NEW_CONSOLE)
            .spawn()
        {
            Ok(_) => {
                println!("preview_scene: launched terminal viewer in a new window.");
                true
            }
            Err(e) => {
                eprintln!("preview_scene: new-window spawn failed ({e}); running inline");
                false
            }
        }
    }

    #[cfg(not(windows))]
    {
        // No portable cross-terminal launcher; try a couple of common
        // emulators, else fall back to inline. Honour $TERMINAL first.
        let exe_str = exe.to_string_lossy().to_string();
        let mut cmd = String::from(&exe_str);
        for a in &forwarded {
            cmd.push(' ');
            cmd.push_str(a);
        }
        let terminals: Vec<(String, Vec<String>)> = {
            let mut v = Vec::new();
            if let Ok(t) = std::env::var("TERMINAL") {
                v.push((t, vec!["-e".into(), "sh".into(), "-c".into(), cmd.clone()]));
            }
            v.push(("x-terminal-emulator".into(),
                    vec!["-e".into(), "sh".into(), "-c".into(), cmd.clone()]));
            v
        };
        for (term, targs) in terminals {
            if std::process::Command::new(&term)
                .args(&targs)
                .env(TUI_CHILD_ENV, "1")
                .spawn()
                .is_ok()
            {
                println!("preview_scene: launched terminal viewer via {term}.");
                return true;
            }
        }
        eprintln!("preview_scene: no terminal emulator found; running inline");
        false
    }
}

fn main() {
    let args = Args::parse();

    // Headless 3D ray-cast render to PNG (works without `preview`).
    if let Some(out) = args.render3d_out.as_deref() {
        render3d::render_to_png(&args, out);
        return;
    }

    // Headless 2D render: `--ppm-out` and `--png-out` both route through
    // `render_ppm` (despite the legacy name) since the format is
    // picked up from the file extension by `write_image`.
    if let Some(out) = args.png_out.as_deref().or(args.ppm_out.as_deref()) {
        render_ppm(&args, out);
        return;
    }

    // Interactive 3D orbit view (needs the graphical window).
    if args.three_d {
        #[cfg(feature = "preview")]
        {
            render3d::run_window(&args);
            return;
        }
        #[cfg(not(feature = "preview"))]
        {
            eprintln!(
                "preview_scene --3d needs the `preview` feature for the window. \
                 Either re-run with `--features preview`, or use \
                 `--render3d-out <file.png>` for a headless 3D render."
            );
            std::process::exit(2);
        }
    }

    // Terminal viewer path.
    if args.tui {
        #[cfg(feature = "tui")]
        {
            let already_child = std::env::var_os(TUI_CHILD_ENV).is_some();
            // Default to a new console window; `--inline` keeps it in
            // the current terminal. Never re-spawn from the child.
            if !args.inline && !already_child && spawn_in_new_window() {
                return;
            }
            if let Err(e) = tui_preview::run(&args) {
                eprintln!("preview_scene --tui: {e}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "tui"))]
        {
            eprintln!(
                "preview_scene --tui requires the `tui` feature. Re-run with:\n\
                 cargo run --release --features tui --bin preview_scene -- \
                 --tui <scene.json> <data_dir>"
            );
            std::process::exit(2);
        }
    }

    run_preview(&args);
}

// ── Terminal (ratatui) cross-section viewer ─────────────────────────
//
// A half-block truecolor renderer for the XY cross-section, drawn into
// the terminal instead of a Win32 window. Each character cell carries
// TWO vertical pixels (`▀` U+2580: foreground paints the top half,
// background the bottom half), so a 120×40 terminal renders a
// 120×80 image. Reuses the engine's `find_cell_recursive` walk and the
// shared `load_preview` / `auto_bounds` machinery, so colours and the
// default viewport match the PNG/PPM path exactly.
//
// Controls: arrows or hjkl pan · +/- zoom · scroll-wheel zoom-at-cursor
// · left-drag pan · [ / ] step the z-slice · L toggle legend · r reset
// · q / Esc quit. The status bar probes the material under the cursor.
#[cfg(feature = "tui")]
mod tui_preview {
    use super::{
        auto_bounds, load_preview, resolve_case_path, world_bounds_xy, Args, LoadedPreview,
    };
    use open_rust_mc::geometry::cell::CellFill;
    use open_rust_mc::geometry::ray::find_cell_recursive;
    use open_rust_mc::geometry::{Geometry, Vec3};
    use ratatui::crossterm::event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton,
        MouseEventKind,
    };
    use ratatui::crossterm::execute;
    use ratatui::layout::{Constraint, Layout, Rect};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Paragraph};
    use std::time::Duration;

    /// Half-block glyph: fg = upper pixel, bg = lower pixel.
    const HALF_BLOCK: &str = "▀";

    #[inline]
    fn col(c: [u8; 3]) -> Color {
        Color::Rgb(c[0], c[1], c[2])
    }

    /// Live viewport: world centre `(cx, cy)`, `scale` in cm per pixel
    /// (one terminal sub-pixel), and the current z-slice. `home_*` hold
    /// the auto-fit target so `r` can recompute the initial framing for
    /// whatever the terminal size is now.
    struct ViewState {
        cx: f64,
        cy: f64,
        scale: f64,
        z: f64,
        home_cx: f64,
        home_cy: f64,
        home_w: f64,
        home_h: f64,
        z_min: f64,
        z_max: f64,
        show_legend: bool,
        needs_fit: bool,
        cursor: Option<(u16, u16)>,
        drag_prev: Option<(u16, u16)>,
        last_inner: Rect,
    }

    impl ViewState {
        /// Re-fit `scale`/centre to frame the whole geometry inside the
        /// current canvas (called on startup and on `r`).
        fn fit(&mut self, inner: Rect) {
            let pw = inner.width.max(1) as f64;
            let ph2 = (inner.height.max(1) as f64) * 2.0;
            self.cx = self.home_cx;
            self.cy = self.home_cy;
            // Fit both axes; the looser axis sets the scale so nothing
            // is clipped. Guard against a zero-extent (single-plane)
            // scene with a 1 cm floor.
            let sx = (self.home_w / pw).max(1e-9);
            let sy = (self.home_h / ph2).max(1e-9);
            self.scale = sx.max(sy).max(1e-6);
            self.needs_fit = false;
        }

        /// World point at the centre of terminal cell `(fx, fy)` (frame
        /// coords). Uses the midpoint of the cell's two sub-pixels.
        fn world_at(&self, fx: u16, fy: u16) -> (f64, f64) {
            let inner = self.last_inner;
            let pw = inner.width.max(1) as f64;
            let ph2 = (inner.height.max(1) as f64) * 2.0;
            let sx = (fx.saturating_sub(inner.x)) as f64 + 0.5;
            let sy = ((fy.saturating_sub(inner.y)) as f64) * 2.0 + 1.0;
            let wx = self.cx + (sx - pw * 0.5) * self.scale;
            let wy = self.cy - (sy - ph2 * 0.5) * self.scale;
            (wx, wy)
        }

        /// Multiply `scale` by `factor`, holding the world point under
        /// the given cursor cell fixed (zoom-at-cursor). `factor < 1`
        /// zooms in.
        fn zoom_at(&mut self, factor: f64, anchor: Option<(u16, u16)>) {
            let inner = self.last_inner;
            let pw = inner.width.max(1) as f64;
            let ph2 = (inner.height.max(1) as f64) * 2.0;
            // Anchor sub-pixel: cursor cell centre, else viewport centre.
            let (sx, sy) = match anchor {
                Some((fx, fy)) => (
                    (fx.saturating_sub(inner.x)) as f64 + 0.5,
                    ((fy.saturating_sub(inner.y)) as f64) * 2.0 + 1.0,
                ),
                None => (pw * 0.5, ph2 * 0.5),
            };
            let wx = self.cx + (sx - pw * 0.5) * self.scale;
            let wy = self.cy - (sy - ph2 * 0.5) * self.scale;
            self.scale = (self.scale * factor).max(1e-6);
            self.cx = wx - (sx - pw * 0.5) * self.scale;
            self.cy = wy + (sy - ph2 * 0.5) * self.scale;
        }
    }

    /// Describe the cell under a world point: material name + colour, or
    /// void / leak. Drives the status-bar probe and legend highlighting.
    fn probe(
        geom: &Geometry,
        palette: &[[u8; 3]],
        names: &[String],
        void: [u8; 3],
        wx: f64,
        wy: f64,
        z: f64,
    ) -> (String, Color) {
        match find_cell_recursive(Vec3::new(wx, wy, z), geom) {
            Some(stack) => {
                let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                match geom.cells[deepest].fill {
                    CellFill::Material(m) => {
                        let idx = m as usize;
                        let name = names.get(idx).map(String::as_str).unwrap_or("material");
                        let c = palette.get(idx).copied().unwrap_or(void);
                        (format!("{name} (#{idx})"), col(c))
                    }
                    _ => ("void (non-material fill)".to_string(), col(void)),
                }
            }
            None => ("leak — outside geometry".to_string(), col(void)),
        }
    }

    pub fn run(args: &Args) -> std::io::Result<()> {
        let LoadedPreview {
            geometry,
            palette,
            void,
            names,
        } = load_preview(args);
        let (x_min, x_max, y_min, y_max, z0) = auto_bounds(&geometry, args);
        // z extent for slice stepping / clamping.
        let (z_min, z_max) = world_bounds_xy(&geometry)
            .map(|b| (b.z_min, b.z_max))
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));

        let title = resolve_case_path(&args.case_json)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("scene")
            .to_string();

        let mut state = ViewState {
            cx: 0.5 * (x_min + x_max),
            cy: 0.5 * (y_min + y_max),
            scale: 0.0,
            z: z0,
            home_cx: 0.5 * (x_min + x_max),
            home_cy: 0.5 * (y_min + y_max),
            home_w: x_max - x_min,
            home_h: y_max - y_min,
            z_min,
            z_max,
            show_legend: true,
            needs_fit: true,
            cursor: None,
            drag_prev: None,
            last_inner: Rect::ZERO,
        };

        let mut terminal = ratatui::init();
        // ratatui::init() handles raw mode + alternate screen + a
        // panic hook that restores them. Mouse capture is on top of
        // that; disable it explicitly before restore so the host
        // terminal isn't left swallowing clicks.
        let _ = execute!(std::io::stdout(), EnableMouseCapture);

        let result = event_loop(&mut terminal, &mut state, &geometry, &palette, &void, &names, &title);

        let _ = execute!(std::io::stdout(), DisableMouseCapture);
        ratatui::restore();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn event_loop(
        terminal: &mut ratatui::DefaultTerminal,
        state: &mut ViewState,
        geometry: &Geometry,
        palette: &[[u8; 3]],
        void: &[u8; 3],
        names: &[String],
        title: &str,
    ) -> std::io::Result<()> {
        let mut dirty = true;
        loop {
            if dirty {
                terminal.draw(|frame| {
                    ui(frame, state, geometry, palette, *void, names, title);
                })?;
                dirty = false;
            }
            // Block up to 250 ms for input; redraw only when something
            // actually changed so an idle viewer doesn't burn CPU
            // re-walking the geometry.
            if !event::poll(Duration::from_millis(250))? {
                continue;
            }
            match event::read()? {
                Event::Resize(_, _) => dirty = true,
                Event::Key(k) if k.kind != KeyEventKind::Release => {
                    if handle_key(k.code, state) {
                        return Ok(());
                    }
                    dirty = true;
                }
                Event::Mouse(m) => {
                    handle_mouse(m, state);
                    dirty = true;
                }
                _ => {}
            }
        }
    }

    /// Returns `true` when the key requests quit.
    fn handle_key(code: KeyCode, state: &mut ViewState) -> bool {
        // Pan one-tenth of the visible span per press.
        let step_x = 0.1 * state.last_inner.width.max(1) as f64 * state.scale;
        let step_y = 0.1 * (state.last_inner.height.max(1) as f64 * 2.0) * state.scale;
        // z step: 2% of the axial extent, floored at 1 cm.
        let dz = if state.z_min.is_finite() && state.z_max.is_finite() {
            (1.0_f64).max((state.z_max - state.z_min).abs() * 0.02)
        } else {
            1.0
        };
        match code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => return true,
            KeyCode::Left | KeyCode::Char('h') => state.cx -= step_x,
            KeyCode::Right | KeyCode::Char('l') => state.cx += step_x,
            KeyCode::Up | KeyCode::Char('k') => state.cy += step_y,
            KeyCode::Down | KeyCode::Char('j') => state.cy -= step_y,
            KeyCode::Char('+') | KeyCode::Char('=') => state.zoom_at(0.8, state.cursor),
            KeyCode::Char('-') | KeyCode::Char('_') => state.zoom_at(1.25, state.cursor),
            KeyCode::Char('[') => {
                state.z -= dz;
                if state.z_min.is_finite() {
                    state.z = state.z.max(state.z_min);
                }
            }
            KeyCode::Char(']') => {
                state.z += dz;
                if state.z_max.is_finite() {
                    state.z = state.z.min(state.z_max);
                }
            }
            KeyCode::Char('L') => state.show_legend = !state.show_legend,
            KeyCode::Char('r') | KeyCode::Char('R') => state.needs_fit = true,
            _ => {}
        }
        false
    }

    fn handle_mouse(m: event::MouseEvent, state: &mut ViewState) {
        state.cursor = Some((m.column, m.row));
        match m.kind {
            MouseEventKind::ScrollUp => state.zoom_at(0.8, Some((m.column, m.row))),
            MouseEventKind::ScrollDown => state.zoom_at(1.25, Some((m.column, m.row))),
            MouseEventKind::Down(MouseButton::Left) => state.drag_prev = Some((m.column, m.row)),
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((px, py)) = state.drag_prev {
                    let dx = (m.column as f64 - px as f64) * state.scale;
                    // Two world-pixels per row vertically.
                    let dy = (m.row as f64 - py as f64) * 2.0 * state.scale;
                    state.cx -= dx;
                    state.cy += dy;
                }
                state.drag_prev = Some((m.column, m.row));
            }
            MouseEventKind::Up(MouseButton::Left) => state.drag_prev = None,
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        frame: &mut ratatui::Frame,
        state: &mut ViewState,
        geometry: &Geometry,
        palette: &[[u8; 3]],
        void: [u8; 3],
        names: &[String],
        title: &str,
    ) {
        let area = frame.area();
        // body (canvas + optional legend) over a 3-line status strip.
        let [body, status_area] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(3)]).areas(area);
        let legend_w: u16 = if state.show_legend {
            // Widest material name + swatch + padding, clamped.
            let longest = names.iter().map(String::len).max().unwrap_or(4) as u16;
            (longest + 8).clamp(16, 36)
        } else {
            0
        };
        let [canvas_area, legend_area] =
            Layout::horizontal([Constraint::Min(1), Constraint::Length(legend_w)]).areas(body);

        // ── Canvas ──────────────────────────────────────────────────
        let canvas_block = Block::bordered().title(Line::from(vec![
            Span::raw(" "),
            Span::styled(title, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" "),
        ]));
        let inner = canvas_block.inner(canvas_area);
        frame.render_widget(canvas_block, canvas_area);

        if state.needs_fit {
            state.fit(inner);
        }
        state.last_inner = inner;
        paint_canvas(frame, inner, state, geometry, palette, void);

        // ── Legend ──────────────────────────────────────────────────
        if legend_w > 0 {
            let mut lines: Vec<Line> = Vec::with_capacity(names.len() + 1);
            for (i, name) in names.iter().enumerate() {
                let c = palette.get(i).copied().unwrap_or(void);
                lines.push(Line::from(vec![
                    Span::styled("██ ", Style::default().fg(col(c))),
                    Span::raw(name.clone()),
                ]));
            }
            lines.push(Line::from(vec![
                Span::styled("██ ", Style::default().fg(col(void))),
                Span::raw("void / leak"),
            ]));
            let legend = Paragraph::new(lines).block(Block::bordered().title(" Materials "));
            frame.render_widget(legend, legend_area);
        }

        // ── Status ──────────────────────────────────────────────────
        let pw = inner.width.max(1) as f64;
        let ph2 = inner.height.max(1) as f64 * 2.0;
        let view_w = pw * state.scale;
        let view_h = ph2 * state.scale;
        let line1 = Line::from(format!(
            " z={:.2} cm   scale={:.4} cm/px   view {:.1}×{:.1} cm   centre ({:.2}, {:.2}) ",
            state.z, state.scale, view_w, view_h, state.cx, state.cy
        ));

        let line2 = match state.cursor {
            Some((fx, fy)) if rect_contains(inner, fx, fy) => {
                let (wx, wy) = state.world_at(fx, fy);
                let (label, c) = probe(geometry, palette, names, void, wx, wy, state.z);
                Line::from(vec![
                    Span::raw(format!(" ({wx:+.2}, {wy:+.2}) → ")),
                    Span::styled("██ ", Style::default().fg(c)),
                    Span::styled(label, Style::default().add_modifier(Modifier::BOLD)),
                ])
            }
            _ => Line::from(
                " arrows/hjkl pan · +/- zoom · scroll zoom@cursor · [ ] z-slice · L legend · r reset · q quit",
            )
            .style(Style::default().dim()),
        };
        let status = Paragraph::new(vec![line1, line2]).block(Block::new().borders(
            ratatui::widgets::Borders::TOP,
        ));
        frame.render_widget(status, status_area);
    }

    fn rect_contains(r: Rect, x: u16, y: u16) -> bool {
        x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height
    }

    /// Sample the geometry over `inner` at 2× vertical resolution and
    /// write half-block cells straight into the frame buffer. The
    /// per-pixel `find_cell_recursive` walk is parallelised across
    /// sub-rows with rayon (the same trick the PNG renderer uses) so
    /// pan/zoom stays responsive on deep recursive lattices.
    fn paint_canvas(
        frame: &mut ratatui::Frame,
        inner: Rect,
        state: &ViewState,
        geometry: &Geometry,
        palette: &[[u8; 3]],
        void: [u8; 3],
    ) {
        use rayon::prelude::*;
        if inner.width == 0 || inner.height == 0 {
            return;
        }
        let pw = inner.width as usize;
        let ph2 = inner.height as usize * 2;
        let (cx, cy, scale, z) = (state.cx, state.cy, state.scale, state.z);
        let pw_f = pw as f64;
        let ph2_f = ph2 as f64;

        // Color grid in scan order, sub-row major. rayon's flat_map
        // preserves ordering on collect.
        let grid: Vec<[u8; 3]> = (0..ph2)
            .into_par_iter()
            .flat_map(|sr| {
                (0..pw)
                    .map(|px| {
                        let wx = cx + (px as f64 + 0.5 - pw_f * 0.5) * scale;
                        let wy = cy - (sr as f64 + 0.5 - ph2_f * 0.5) * scale;
                        match find_cell_recursive(Vec3::new(wx, wy, z), geometry) {
                            Some(stack) => {
                                let d = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                                match geometry.cells[d].fill {
                                    CellFill::Material(m) => {
                                        palette.get(m as usize).copied().unwrap_or(void)
                                    }
                                    _ => void,
                                }
                            }
                            None => void,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let buf = frame.buffer_mut();
        for r in 0..inner.height {
            for c in 0..inner.width {
                let top = grid[(2 * r as usize) * pw + c as usize];
                let bot = grid[(2 * r as usize + 1) * pw + c as usize];
                if let Some(cell) = buf.cell_mut((inner.x + c, inner.y + r)) {
                    cell.set_symbol(HALF_BLOCK);
                    cell.set_fg(col(top));
                    cell.set_bg(col(bot));
                }
            }
        }

        // Crosshair marker at the cursor cell (keeps the lower-pixel
        // colour as the background so the probe stays legible).
        if let Some((fx, fy)) = state.cursor {
            if rect_contains(inner, fx, fy) {
                if let Some(cell) = buf.cell_mut((fx, fy)) {
                    cell.set_symbol("+");
                    cell.set_fg(Color::Yellow);
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn test_state() -> ViewState {
            ViewState {
                cx: 5.0,
                cy: -3.0,
                scale: 0.0,
                z: 0.0,
                home_cx: 5.0,
                home_cy: -3.0,
                home_w: 40.0,
                home_h: 20.0,
                z_min: -10.0,
                z_max: 10.0,
                show_legend: true,
                needs_fit: true,
                cursor: None,
                drag_prev: None,
                last_inner: Rect::new(1, 1, 80, 24),
            }
        }

        #[test]
        fn fit_frames_the_whole_geometry() {
            let mut s = test_state();
            let inner = Rect::new(0, 0, 80, 25); // 80 cols × 50 sub-rows
            s.fit(inner);
            // Looser axis sets the scale: width 40/80 = 0.5, height
            // 20/50 = 0.4 → 0.5 cm/px so nothing is clipped.
            assert!((s.scale - 0.5).abs() < 1e-9, "scale = {}", s.scale);
            assert!(!s.needs_fit);
            // Whole geometry must fit inside the rendered span.
            assert!(80.0 * s.scale >= s.home_w - 1e-9);
            assert!((25.0 * 2.0) * s.scale >= s.home_h - 1e-9);
        }

        #[test]
        fn world_at_centre_returns_view_centre() {
            let mut s = test_state();
            s.scale = 0.5;
            s.needs_fit = false;
            let inner = s.last_inner;
            // Centre cell of an 80×24 inner area.
            let (wx, wy) = s.world_at(inner.x + inner.width / 2, inner.y + inner.height / 2);
            assert!((wx - s.cx).abs() < s.scale, "wx={wx} cx={}", s.cx);
            assert!((wy - s.cy).abs() < 2.0 * s.scale, "wy={wy} cy={}", s.cy);
        }

        #[test]
        fn zoom_at_cursor_pins_the_anchor_world_point() {
            let mut s = test_state();
            s.scale = 0.5;
            s.needs_fit = false;
            let anchor = (20u16, 8u16); // some off-centre cell
            let before = s.world_at(anchor.0, anchor.1);
            s.zoom_at(0.8, Some(anchor));
            let after = s.world_at(anchor.0, anchor.1);
            // The world point under the cursor must stay put across zoom.
            assert!((before.0 - after.0).abs() < 1e-6, "x drift {before:?} {after:?}");
            assert!((before.1 - after.1).abs() < 1e-6, "y drift {before:?} {after:?}");
            assert!(s.scale < 0.5, "zoom-in must shrink scale");
        }

        #[test]
        fn paint_canvas_writes_half_blocks_without_a_real_terminal() {
            use open_rust_mc::geometry::Geometry;
            use ratatui::backend::TestBackend;
            use ratatui::Terminal;

            // Empty geometry → every sample leaks → void fill. We're
            // exercising the buffer-write path + half-block math, not
            // the physics: the whole canvas must be painted with the
            // half-block glyph (no untouched cells).
            let geom = Geometry::flat(Vec::new(), Vec::new()).expect("empty geometry");
            let palette: Vec<[u8; 3]> = vec![[10, 20, 30]];
            let void = [255, 0, 220];
            let names = vec!["only-mat".to_string()];

            let mut s = test_state();
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
            terminal
                .draw(|f| ui(f, &mut s, &geom, &palette, void, &names, "unit-test"))
                .expect("draw");

            let buf = terminal.backend().buffer();
            // The canvas interior (inside the border) must be half-blocks.
            let inner = s.last_inner;
            assert!(inner.width > 0 && inner.height > 0);
            let mid = buf[(inner.x + inner.width / 2, inner.y + inner.height / 2)]
                .symbol()
                .to_string();
            assert_eq!(mid, HALF_BLOCK, "canvas interior should be half-blocks");
        }
    }
}

// ── 3D ray-cast geometry view ───────────────────────────────────────
//
// A real perspective-3D render of the CSG, built on the very same
// geometry walk the Monte Carlo transport uses: cast one camera ray
// per pixel, step through cells with `trace_step_recursive` until the
// first *opaque* material (air / void / vacuum are skipped so you see
// the solids inside them), estimate a surface normal from the gradient
// of the occupancy field, and shade with Lambert lighting tinted by
// the material's palette colour.
//
// No GPU, no wgpu, no 3D engine — the scenes are small (a handful of
// surfaces) so a rayon-parallel CPU cast renders an orbit frame in a
// fraction of a second. The shading path is a pure function of
// `(geometry, camera)`, so a CUDA ray-cast kernel reusing the device
// CSG in `geom_recursive.cu` could later drop in behind this same API.
mod render3d {
    use super::{load_preview, world_bounds_xy, Args, LoadedPreview};
    use open_rust_mc::geometry::cell::CellFill;
    use open_rust_mc::geometry::ray::{find_cell_recursive, trace_step_recursive};
    use open_rust_mc::geometry::{Geometry, Vec3};
    use std::path::Path;

    /// Orbit camera: spherical position about `target`, +z up.
    struct Camera {
        pos: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        tan_half_fov: f64,
    }

    impl Camera {
        fn orbit(target: Vec3, azim_rad: f64, elev_rad: f64, radius: f64, fov_deg: f64) -> Self {
            let ce = elev_rad.cos();
            let dir = Vec3::new(ce * azim_rad.cos(), ce * azim_rad.sin(), elev_rad.sin());
            let pos = target + dir * radius;
            let forward = (target - pos).normalized();
            let world_up = Vec3::new(0.0, 0.0, 1.0);
            // Guard against gimbal lock when looking straight down/up.
            let right = {
                let r = forward.cross(world_up);
                if r.length() < 1e-6 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    r.normalized()
                }
            };
            let up = right.cross(forward).normalized();
            Self {
                pos,
                forward,
                right,
                up,
                tan_half_fov: (fov_deg.to_radians() * 0.5).tan(),
            }
        }

        /// Primary ray direction through pixel `(px, py)` of a `w × h`
        /// image (pixel centres, y flipped so row 0 is the top).
        fn ray_dir(&self, px: usize, py: usize, w: usize, h: usize) -> Vec3 {
            let aspect = w as f64 / h as f64;
            let ndc_x = ((px as f64 + 0.5) / w as f64) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((py as f64 + 0.5) / h as f64) * 2.0;
            let d = self.forward
                + self.right * (ndc_x * aspect * self.tan_half_fov)
                + self.up * (ndc_y * self.tan_half_fov);
            d.normalized()
        }
    }

    /// Axis-aligned bounds of the geometry (xy from surfaces/lattices,
    /// z from PlaneZ / lattice extents), padded 6%. Falls back to a
    /// cube around the z-slice when the axial extent is unbounded.
    fn scene_aabb(geom: &Geometry) -> (Vec3, Vec3) {
        match world_bounds_xy(geom) {
            Some(b) => {
                let (zmin, zmax) = if b.z_min.is_finite() && b.z_max.is_finite() && b.z_max > b.z_min
                {
                    (b.z_min, b.z_max)
                } else {
                    let c = b.default_z();
                    let e = 0.5 * b.xy_extent().max(1.0);
                    (c - e, c + e)
                };
                let min = Vec3::new(b.x_min, b.y_min, zmin);
                let max = Vec3::new(b.x_max, b.y_max, zmax);
                let pad = (max - min) * 0.06;
                (min - pad, max + pad)
            }
            None => (Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0)),
        }
    }

    /// Per-axis slab clip. Returns the `(t_near, t_far)` overlap of the
    /// ray with `[lo, hi]`; an empty interval encodes a miss.
    #[inline]
    fn slab(o: f64, d: f64, lo: f64, hi: f64) -> (f64, f64) {
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                (f64::INFINITY, f64::NEG_INFINITY)
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            }
        } else {
            let inv = 1.0 / d;
            let t1 = (lo - o) * inv;
            let t2 = (hi - o) * inv;
            if t1 <= t2 { (t1, t2) } else { (t2, t1) }
        }
    }

    /// Ray vs AABB. `Some((t_enter, t_exit))` with `t_exit >= t_enter`
    /// and `t_exit >= 0`, else `None`.
    fn ray_aabb(o: Vec3, d: Vec3, min: Vec3, max: Vec3) -> Option<(f64, f64)> {
        let (ax0, ax1) = slab(o.x, d.x, min.x, max.x);
        let (ay0, ay1) = slab(o.y, d.y, min.y, max.y);
        let (az0, az1) = slab(o.z, d.z, min.z, max.z);
        let t_enter = ax0.max(ay0).max(az0);
        let t_exit = ax1.min(ay1).min(az1);
        if t_exit >= t_enter && t_exit >= 0.0 {
            Some((t_enter, t_exit))
        } else {
            None
        }
    }

    /// `true` for materials that should be drawn solid. Air / void /
    /// vacuum (and the no-material `Void` fill) are transparent so the
    /// camera sees the objects suspended inside them.
    fn opaque_mask(names: &[String]) -> Vec<bool> {
        names
            .iter()
            .map(|n| {
                let n = n.to_lowercase();
                !(n.contains("air") || n.contains("void") || n.contains("vacuum"))
            })
            .collect()
    }

    /// Occupancy field: 1.0 inside an opaque material, 0.0 elsewhere.
    #[inline]
    fn occ(p: Vec3, geom: &Geometry, opaque: &[bool]) -> f64 {
        match find_cell_recursive(p, geom) {
            Some(stack) => {
                let ci = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                if let CellFill::Material(m) = geom.cells[ci].fill {
                    if opaque.get(m as usize).copied().unwrap_or(true) {
                        return 1.0;
                    }
                }
                0.0
            }
            None => 0.0,
        }
    }

    struct Hit {
        mat: usize,
        normal: Vec3,
    }

    /// March a camera ray to the first opaque material. Exact DDA via
    /// `trace_step_recursive` while inside a cell; small fixed micro-
    /// steps only to cross genuine void/leak gaps (where the geometry
    /// walk has no cell to step from). Returns the material index + an
    /// outward surface normal estimated from the occupancy gradient.
    fn cast_ray(o: Vec3, d: Vec3, geom: &Geometry, opaque: &[bool], aabb: (Vec3, Vec3)) -> Option<Hit> {
        let (t_enter, t_exit) = ray_aabb(o, d, aabb.0, aabb.1)?;
        let diag = (aabb.1 - aabb.0).length();
        let micro = (diag / 1024.0).max(1e-6);
        let eps = (diag * 1e-7).max(1e-9);

        let mut t = t_enter.max(0.0) + eps;
        let mut pos = o + d * t;
        let mut stack = find_cell_recursive(pos, geom);
        // Bound the walk so a degenerate geometry can't spin forever.
        for _ in 0..8192 {
            if t > t_exit {
                return None;
            }
            match stack {
                Some(ref s) => {
                    let ci = s.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                    if let CellFill::Material(m) = geom.cells[ci].fill {
                        if opaque.get(m as usize).copied().unwrap_or(true) {
                            return Some(Hit {
                                mat: m as usize,
                                normal: surface_normal(pos, ci, d, geom, opaque, micro),
                            });
                        }
                    }
                    // Transparent cell — jump to the next surface exactly.
                    match trace_step_recursive(s, pos, d, geom) {
                        Some(h) if h.distance.is_finite() => {
                            t += h.distance + eps;
                            pos = o + d * t;
                            stack = h.next_stack;
                        }
                        _ => return None,
                    }
                }
                None => {
                    // In a vacuum/leak gap: creep forward until we enter
                    // a cell again or exit the box.
                    t += micro;
                    pos = o + d * t;
                    stack = find_cell_recursive(pos, geom);
                }
            }
        }
        None
    }

    /// Camera-facing surface normal at a hit point. Prefers the
    /// analytic normal of whichever of the hit cell's bounding surfaces
    /// the point lies on (crisp on planes/quadrics, no occupancy-
    /// quantisation banding); falls back to the occupancy gradient when
    /// the cell has no surface that `p` sits on (e.g. a lattice element
    /// in a frame this flat-evaluation doesn't capture). The result is
    /// flipped to face the incoming ray so visible surfaces are lit.
    fn surface_normal(
        p: Vec3,
        cell_idx: usize,
        view_dir: Vec3,
        geom: &Geometry,
        opaque: &[bool],
        micro: f64,
    ) -> Vec3 {
        let mut idxs = Vec::new();
        geom.cells[cell_idx].region.surface_indices(&mut idxs);
        idxs.sort_unstable();
        idxs.dedup();

        // Surface the point lies on = smallest |evaluate|. The hit
        // point sits on a cell boundary by construction (we just
        // crossed into this cell, or micro-stepped to its edge), so the
        // nearest bounding surface is the visible one. `evaluate` is not
        // a metric distance — for a cylinder it's x²+y²−r² (cm²) — so we
        // compare magnitudes only to *rank* surfaces, never threshold
        // against a length.
        let mut best: Option<usize> = None;
        let mut best_abs = f64::INFINITY;
        for &si in &idxs {
            let v = geom.surfaces[si].evaluate(p).abs();
            if v < best_abs {
                best_abs = v;
                best = Some(si);
            }
        }
        if let Some(si) = best {
            let mut n = geom.surfaces[si].normal_at(p);
            let len = n.length();
            if len > 1e-12 {
                n = n * (1.0 / len);
                // Face the camera: flip if it points along the ray.
                if n.dot(view_dir) > 0.0 {
                    n = -n;
                }
                return n;
            }
        }
        occ_normal(p, view_dir, geom, opaque, micro)
    }

    /// Outward normal from the gradient of the occupancy field. The
    /// gradient points into the solid, so the outward normal is its
    /// negation; degenerate (zero-gradient) samples fall back to facing
    /// the camera.
    fn occ_normal(p: Vec3, view_dir: Vec3, geom: &Geometry, opaque: &[bool], micro: f64) -> Vec3 {
        let h = micro * 2.0;
        let dx = occ(p + Vec3::new(h, 0.0, 0.0), geom, opaque)
            - occ(p - Vec3::new(h, 0.0, 0.0), geom, opaque);
        let dy = occ(p + Vec3::new(0.0, h, 0.0), geom, opaque)
            - occ(p - Vec3::new(0.0, h, 0.0), geom, opaque);
        let dz = occ(p + Vec3::new(0.0, 0.0, h), geom, opaque)
            - occ(p - Vec3::new(0.0, 0.0, h), geom, opaque);
        let g = Vec3::new(dx, dy, dz);
        let len = g.length();
        if len > 1e-9 {
            g * (-1.0 / len)
        } else {
            -view_dir
        }
    }

    /// Lambert shade: a fixed key light plus a softer head-light so
    /// camera-facing back surfaces aren't pure black. Returns sRGB-ish
    /// 8-bit colour tinted by the material.
    fn shade(base: [u8; 3], n: Vec3, view_dir: Vec3) -> [u8; 3] {
        let key = Vec3::new(0.35, 0.45, 0.82).normalized();
        let kd = n.dot(key).max(0.0);
        let hd = n.dot(-view_dir).max(0.0);
        let diff = 0.55 * kd + 0.45 * hd;
        let lit = (0.18 + 0.82 * diff).min(1.15);
        [
            ((base[0] as f64) * lit).round().clamp(0.0, 255.0) as u8,
            ((base[1] as f64) * lit).round().clamp(0.0, 255.0) as u8,
            ((base[2] as f64) * lit).round().clamp(0.0, 255.0) as u8,
        ]
    }

    /// Background gradient: dark slate, slightly lighter toward the
    /// bottom so the silhouette reads against it.
    #[inline]
    fn background(py: usize, h: usize) -> [u8; 3] {
        let f = py as f64 / (h.max(1) as f64);
        let top = [16.0, 17.0, 23.0];
        let bot = [30.0, 33.0, 42.0];
        [
            (top[0] + (bot[0] - top[0]) * f) as u8,
            (top[1] + (bot[1] - top[1]) * f) as u8,
            (top[2] + (bot[2] - top[2]) * f) as u8,
        ]
    }

    /// Render one `w × h` RGB frame. Parallel across rows; each ray is
    /// independent so this is embarrassingly parallel.
    fn render_frame(
        geom: &Geometry,
        palette: &[[u8; 3]],
        opaque: &[bool],
        aabb: (Vec3, Vec3),
        cam: &Camera,
        w: usize,
        h: usize,
    ) -> Vec<[u8; 3]> {
        use rayon::prelude::*;
        let rows: Vec<Vec<[u8; 3]>> = (0..h)
            .into_par_iter()
            .map(|py| {
                (0..w)
                    .map(|px| {
                        let dir = cam.ray_dir(px, py, w, h);
                        match cast_ray(cam.pos, dir, geom, opaque, aabb) {
                            Some(hit) => {
                                let base = palette.get(hit.mat).copied().unwrap_or([200, 200, 200]);
                                shade(base, hit.normal, dir)
                            }
                            None => background(py, h),
                        }
                    })
                    .collect()
            })
            .collect();
        let mut buf = Vec::with_capacity(w * h);
        for row in &rows {
            buf.extend_from_slice(row);
        }
        buf
    }

    /// Headless: ray-cast a single frame from the CLI camera angle and
    /// write it to PNG.
    pub fn render_to_png(args: &Args, out: &Path) {
        let LoadedPreview {
            geometry,
            palette,
            names,
            ..
        } = load_preview(args);
        let opaque = opaque_mask(&names);
        let aabb = scene_aabb(&geometry);
        let target = (aabb.0 + aabb.1) * 0.5;
        let radius = (aabb.1 - aabb.0).length() * 0.9;
        let cam = Camera::orbit(
            target,
            args.cam_azim.to_radians(),
            args.cam_elev.to_radians(),
            radius,
            45.0,
        );
        let w = args.resolution as usize;
        let h = args.resolution as usize;
        let buf = render_frame(&geometry, &palette, &opaque, aabb, &cam, w, h);
        write_png_wh(out, &buf, w as u32, h as u32);
        eprintln!(
            "wrote {} ({}×{})  azim={:.1}° elev={:.1}°",
            out.display(),
            w,
            h,
            args.cam_azim,
            args.cam_elev
        );
    }

    /// PNG writer for an arbitrary `w × h` RGB buffer (the shared
    /// `write_image` helper only handles square images).
    fn write_png_wh(path: &Path, buf: &[[u8; 3]], w: u32, h: u32) {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        let flat: &[u8] = bytemuck::cast_slice(buf);
        writer.write_image_data(flat).expect("png data");
    }

    /// Interactive orbit-camera window (minifb). Left-drag rotates,
    /// scroll zooms, right-drag pans the target, `r` resets, Escape
    /// quits. Re-renders only when the camera actually moves.
    #[cfg(feature = "preview")]
    pub fn run_window(args: &Args) {
        use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};

        let LoadedPreview {
            geometry,
            palette,
            names,
            ..
        } = load_preview(args);
        let opaque = opaque_mask(&names);
        let aabb = scene_aabb(&geometry);
        let target0 = (aabb.0 + aabb.1) * 0.5;
        let radius0 = (aabb.1 - aabb.0).length() * 0.9;

        // Print the legend once — minifb can't draw text.
        eprintln!("── materials ──");
        for (i, n) in names.iter().enumerate() {
            let c = palette.get(i).copied().unwrap_or([200, 200, 200]);
            let solid = if opaque.get(i).copied().unwrap_or(true) {
                "solid"
            } else {
                "transparent"
            };
            eprintln!("  [{i}] {n}  rgb({},{},{})  {solid}", c[0], c[1], c[2]);
        }
        eprintln!(
            "controls: left-drag orbit · scroll zoom · right-drag pan · r reset · Esc quit"
        );

        let (mut w, mut h) = (820usize, 600usize);
        let mut azim = args.cam_azim.to_radians();
        let mut elev = args.cam_elev.to_radians();
        let mut radius = radius0;
        let mut target = target0;

        let mut window = Window::new(
            &format!(
                "preview_scene 3D — {}",
                super::resolve_case_path(&args.case_json)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("scene")
            ),
            w,
            h,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("preview_scene 3D window: {e}"));
        window.set_target_fps(60);

        let frame_u32 = |w: usize,
                         h: usize,
                         azim: f64,
                         elev: f64,
                         radius: f64,
                         target: Vec3|
         -> Vec<u32> {
            let cam = Camera::orbit(target, azim, elev, radius, 45.0);
            let rgb = render_frame(&geometry, &palette, &opaque, aabb, &cam, w, h);
            rgb.iter()
                .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32))
                .collect()
        };

        let mut buf = frame_u32(w, h, azim, elev, radius, target);
        window.update_with_buffer(&buf, w, h).ok();

        let mut last_size = (w, h);
        let mut prev_left: Option<(f32, f32)> = None;
        let mut prev_right: Option<(f32, f32)> = None;
        let mut prev_r = false;

        while window.is_open() && !window.is_key_down(Key::Escape) {
            let mut dirty = false;

            let cur = window.get_size();
            if cur != last_size && cur.0 > 0 && cur.1 > 0 {
                w = cur.0;
                h = cur.1;
                last_size = cur;
                dirty = true;
            }

            let mouse = window.get_mouse_pos(MouseMode::Discard);

            // Left-drag = orbit.
            if window.get_mouse_down(MouseButton::Left) {
                if let (Some((mx, my)), Some((px, py))) = (mouse, prev_left) {
                    azim += (mx - px) as f64 * 0.01;
                    elev = (elev - (my - py) as f64 * 0.01).clamp(-1.5, 1.5);
                    dirty = true;
                }
                prev_left = mouse;
            } else {
                prev_left = None;
            }

            // Right-drag = pan the target across the camera plane.
            if window.get_mouse_down(MouseButton::Right) {
                if let (Some((mx, my)), Some((px, py))) = (mouse, prev_right) {
                    let cam = Camera::orbit(target, azim, elev, radius, 45.0);
                    let scale = radius * 0.0015;
                    target = target + cam.right * (-(mx - px) as f64 * scale)
                        + cam.up * ((my - py) as f64 * scale);
                    dirty = true;
                }
                prev_right = mouse;
            } else {
                prev_right = None;
            }

            // Scroll = zoom (dolly).
            if let Some((_, sy)) = window.get_scroll_wheel() {
                if sy.abs() > 0.05 {
                    let mag = (sy.abs() as f64).min(1.0);
                    let step = 0.85_f64.powf(mag);
                    radius = (radius * if sy > 0.0 { step } else { 1.0 / step })
                        .clamp(1e-3, 1e7);
                    dirty = true;
                }
            }

            let r_now = window.is_key_down(Key::R);
            if r_now && !prev_r {
                azim = args.cam_azim.to_radians();
                elev = args.cam_elev.to_radians();
                radius = radius0;
                target = target0;
                dirty = true;
            }
            prev_r = r_now;

            if dirty {
                buf = frame_u32(w, h, azim, elev, radius, target);
            }
            window.update_with_buffer(&buf, w, h).ok();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ray_aabb_hits_and_misses() {
            let min = Vec3::new(-1.0, -1.0, -1.0);
            let max = Vec3::new(1.0, 1.0, 1.0);
            // Ray straight down +x through the box.
            let hit = ray_aabb(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), min, max);
            let (t0, t1) = hit.expect("should hit");
            assert!((t0 - 4.0).abs() < 1e-9 && (t1 - 6.0).abs() < 1e-9, "{t0} {t1}");
            // Parallel ray that misses entirely.
            let miss = ray_aabb(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0), min, max);
            assert!(miss.is_none());
        }

        #[test]
        fn camera_ray_centre_points_at_target() {
            let cam = Camera::orbit(Vec3::new(0.0, 0.0, 0.0), 0.7, 0.4, 10.0, 45.0);
            // The centre pixel ray should be (within fov) the forward
            // axis, i.e. roughly antiparallel to the camera position.
            let dir = cam.ray_dir(50, 50, 100, 100);
            let to_target = (Vec3::new(0.0, 0.0, 0.0) - cam.pos).normalized();
            assert!(dir.dot(to_target) > 0.999, "centre ray should aim at target");
        }

        #[test]
        fn opaque_mask_marks_air_transparent() {
            let names = vec![
                "Air".to_string(),
                "Depleted Uranium".to_string(),
                "vacuum gap".to_string(),
            ];
            let m = opaque_mask(&names);
            assert_eq!(m, vec![false, true, false]);
        }

        #[test]
        fn renders_a_concentric_sphere_scene() {
            use open_rust_mc::geometry::cell::{inside, intersect_all, outside, Cell};
            use open_rust_mc::geometry::surface::{BoundaryCondition, Surface};
            use open_rust_mc::geometry::CellId;

            // Inner solid sphere (material 0, surface 0) inside an outer
            // shell (material 1, between surfaces 0 and 1), both centred
            // at origin. Built with the same CSG region helpers the
            // scene loader uses (surfaces referenced by vec index).
            let s_inner = Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 2.0,
                bc: BoundaryCondition::Transmission,
            };
            let s_outer = Surface::Sphere {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 4.0,
                bc: BoundaryCondition::Vacuum,
            };
            let inner = Cell::new(CellId(0), inside(0), CellFill::Material(0));
            let shell = Cell::new(
                CellId(1),
                intersect_all(vec![inside(1), outside(0)]),
                CellFill::Material(1),
            );
            let geom = Geometry::flat(vec![s_inner, s_outer], vec![inner, shell])
                .expect("two-sphere geometry");

            let palette = vec![[220u8, 80, 60], [90, 160, 220]];
            let opaque = vec![true, true];
            let aabb = scene_aabb(&geom);
            let target = (aabb.0 + aabb.1) * 0.5;
            let radius = (aabb.1 - aabb.0).length() * 0.9;
            let cam = Camera::orbit(target, 0.6, 0.3, radius, 45.0);
            let buf = render_frame(&geom, &palette, &opaque, aabb, &cam, 64, 64);

            // The centre pixel must land on the sphere (a shaded
            // material colour), not the dark background.
            let centre = buf[32 * 64 + 32];
            let bg = background(32, 64);
            assert_ne!(centre, bg, "centre of frame should hit the sphere");
            // And at least a quarter of the frame should be covered by
            // the object (sanity that it isn't a single stray pixel).
            let hits = buf.iter().filter(|&&p| p != background(0, 64) ).count();
            assert!(hits > 64 * 64 / 8, "object coverage too small: {hits}");
        }
    }
}
