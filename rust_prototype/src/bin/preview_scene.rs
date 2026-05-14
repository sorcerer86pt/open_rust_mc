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
}

impl GeomBounds {
    fn xy_extent(&self) -> f64 {
        (self.x_max - self.x_min).max(self.y_max - self.y_min)
    }
    fn cx(&self) -> f64 { 0.5 * (self.x_min + self.x_max) }
    fn cy(&self) -> f64 { 0.5 * (self.y_min + self.y_max) }
    /// Sensible z-slice when the user didn't pass `--z`. Midpoint
    /// of the geometry's z-bounds — picks z = 0 for origin-centred
    /// geometries and z > 0 for stacked-can experiments that live
    /// entirely above z = 0.
    fn default_z(&self) -> f64 {
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

    let palette: Vec<[u8; 3]> = materials.iter().map(|m| auto_color(&m.name)).collect();
    let void = [0_u8, 0, 0];

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
