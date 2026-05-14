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

    /// Render the cross-section to a PPM file and exit (no window).
    /// Useful for headless debugging — the file can be opened in any
    /// image viewer / converted with `magick`. Bypasses the
    /// `preview` feature gate entirely so this works on a default
    /// `cargo run --bin preview_scene -- ... --ppm-out out.ppm`.
    #[arg(long)]
    ppm_out: Option<PathBuf>,

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

    // ── Default viewport: union of finite surface AABBs + every
    // lattice extent. See `render_ppm` for the bug story (without
    // the lattice walk, pin-cylinder surfaces alone collapse the
    // viewport to a single pin's radius).
    let half = args.half_size.unwrap_or_else(|| {
        let mut bound = 0.0_f64;
        for s in &geometry.surfaces {
            let aabb = s.aabb();
            for v in [
                aabb.min.x.abs(), aabb.max.x.abs(),
                aabb.min.y.abs(), aabb.max.y.abs(),
            ] {
                if v.is_finite() && v > bound {
                    bound = v;
                }
            }
        }
        for lat in &geometry.lattices {
            let upper_x = lat.origin.x + lat.shape[0] as f64 * lat.pitch.x;
            let upper_y = lat.origin.y + lat.shape[1] as f64 * lat.pitch.y;
            for v in [lat.origin.x.abs(), lat.origin.y.abs(), upper_x.abs(), upper_y.abs()] {
                if v.is_finite() && v > bound {
                    bound = v;
                }
            }
        }
        for hex in &geometry.hex_lattices {
            let r = hex.n_rings as f64 * hex.pitch_xy;
            for v in [
                (hex.center.x - r).abs(), (hex.center.x + r).abs(),
                (hex.center.y - r).abs(), (hex.center.y + r).abs(),
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
    use open_rust_mc::geometry::cell::CellFill;
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

    // Viewport: union of every finite surface AABB, plus every
    // lattice extent. The lattice walk is what made this bug visible
    // — pin-cylinder surfaces alone bound the viewport to a single
    // pin (~±0.4 cm) and the entire 17×17 grid collapsed visually
    // into the centre element. Walking `geometry.lattices` /
    // `geometry.hex_lattices` and including their world-space
    // extents fixes the auto-zoom to actually show the lattice.
    let half = args.half_size.unwrap_or_else(|| {
        let mut bound = 0.0_f64;
        for s in &geometry.surfaces {
            let aabb = s.aabb();
            for v in [aabb.min.x.abs(), aabb.max.x.abs(), aabb.min.y.abs(), aabb.max.y.abs()] {
                if v.is_finite() && v > bound { bound = v; }
            }
        }
        // RectLattice: origin (lower-left) + shape × pitch (upper-right).
        for lat in &geometry.lattices {
            let upper = open_rust_mc::geometry::Vec3::new(
                lat.origin.x + lat.shape[0] as f64 * lat.pitch.x,
                lat.origin.y + lat.shape[1] as f64 * lat.pitch.y,
                lat.origin.z + lat.shape[2] as f64 * lat.pitch.z,
            );
            for v in [
                lat.origin.x.abs(), lat.origin.y.abs(),
                upper.x.abs(), upper.y.abs(),
            ] {
                if v.is_finite() && v > bound { bound = v; }
            }
        }
        // HexLattice: centre ± n_rings × pitch_xy.
        for hex in &geometry.hex_lattices {
            let r = hex.n_rings as f64 * hex.pitch_xy;
            for v in [
                (hex.center.x - r).abs(), (hex.center.x + r).abs(),
                (hex.center.y - r).abs(), (hex.center.y + r).abs(),
            ] {
                if v.is_finite() && v > bound { bound = v; }
            }
        }
        if bound <= 0.0 { 10.0 } else { 1.05 * bound }
    });
    let res = args.resolution;
    let x_min = -half;
    let x_max =  half;
    let y_min = -half;
    let y_max =  half;
    let dx = (x_max - x_min) / res as f64;
    let dy = (y_max - y_min) / res as f64;

    // Per-pixel walk.
    let mut buf = vec![[0u8, 0, 0]; (res * res) as usize];
    for py in 0..res {
        let world_y = y_max - (py as f64 + 0.5) * dy;
        for px in 0..res {
            let world_x = x_min + (px as f64 + 0.5) * dx;
            let pos = Vec3::new(world_x, world_y, args.z);
            let color = match find_cell_recursive(pos, &geometry) {
                Some(stack) => {
                    let deepest = stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
                    match geometry.cells[deepest].fill {
                        CellFill::Material(m) => {
                            palette.get(m as usize).copied().unwrap_or(void)
                        }
                        _ => void,
                    }
                }
                None => void,
            };
            buf[(py * res + px) as usize] = color;
        }
    }

    // Optional sample-grid debug print BEFORE the file write so the
    // operator sees it on stderr even if the PPM write fails.
    if args.debug_samples {
        eprintln!("\n── find_cell_recursive samples (3×3 grid across viewport) ──");
        for j in 0..3 {
            for i in 0..3 {
                let sx = x_min + (i as f64 + 0.5) * (x_max - x_min) / 3.0;
                let sy = y_max - (j as f64 + 0.5) * (y_max - y_min) / 3.0;
                let pos = Vec3::new(sx, sy, args.z);
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

    // Write PPM P6 (binary RGB).
    use std::io::Write;
    let mut out = std::fs::File::create(ppm_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", ppm_path.display()));
    write!(out, "P6\n{} {}\n255\n", res, res).unwrap();
    for px in &buf {
        out.write_all(px).unwrap();
    }
    eprintln!("wrote {} ({}×{})", ppm_path.display(), res, res);
}

fn main() {
    let args = Args::parse();
    if let Some(ppm) = args.ppm_out.as_deref() {
        render_ppm(&args, ppm);
        return;
    }
    run_preview(&args);
}
