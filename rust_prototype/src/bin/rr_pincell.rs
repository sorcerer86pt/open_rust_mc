//! Random-ray PWR pin-cell benchmark — 2-group multigroup transport.
//!
//! Validates the multigroup random-ray pipeline (forward + adjoint,
//! mortal + immortal, cell-based FSRs) on a single PWR-style pin cell.
//! UO₂ fuel cylinder centred in a 1.26-cm-pitch square cell of water,
//! reflective BC on all four sides (infinite lattice approximation).
//!
//! 2-group XS are simplified PWR-thermal-spectrum values:
//!   - Fast (1) and thermal (2) energy groups.
//!   - Fuel: fissile, with downscatter and a realistic χ spectrum.
//!   - Moderator: pure absorber/scatter with thermalising upscatter.
//!
//! Expected k_inf is ~1.30 (typical UO₂ pin cell result). The value is
//! benchmark-illustrative, not verified against a published reference —
//! the goal is to demonstrate the multigroup pipeline runs end-to-end.
//!
//! Scaling this binary to full C5G7 (4 fuel materials × 7 groups) is a
//! data plumbing exercise (drop the published XS into `mgxs::fuel_xs`
//! / `mgxs::moderator_xs`, expand the geometry to a 17×17 lattice). No
//! new code in the solver is needed.

use std::collections::HashMap;
use std::time::Instant;

use open_rust_mc::geometry::cell::{self, Cell, CellFill, CellId, Region};
use open_rust_mc::geometry::surface::BoundaryCondition;
use open_rust_mc::geometry::{Aabb, Geometry, Surface, Vec3};
use open_rust_mc::random_ray::{
    AdjointFlag, FsrMesh, MaterialMgxs, MgxsLibrary, RandomRaySolver, RaySolverConfig, SolverMode,
};

const PITCH: f64 = 1.26;
const FUEL_RADIUS: f64 = 0.4096; // ~standard PWR fuel rod outer radius
const HEIGHT_HALF: f64 = 5.0; // 10-cm-tall slab — reflective top/bottom

/// 2-group MGXS for UO₂ fuel. Loosely PWR-thermal style.
fn fuel_mgxs() -> MaterialMgxs {
    // Fast group is mostly down-scatter; thermal has self-scatter.
    // Σ_s[g_in * n + g_out]:
    //   row 0 (from fast): 0 -> 0 = 0.5, 0 -> 1 = 0.05
    //   row 1 (from therm): 1 -> 0 = 0.001, 1 -> 1 = 0.55
    let scatter = vec![
        0.5, 0.05, // from fast
        0.001, 0.55, // from thermal
    ];
    let sigma_t = vec![0.6, 1.0];
    let sigma_a = vec![0.05, 0.4];
    let nu_sigma_f = vec![0.025, 0.7];
    let chi = vec![1.0, 0.0]; // all fission neutrons born fast
    MaterialMgxs::fissionable(sigma_t, sigma_a, nu_sigma_f, chi, scatter).expect("fuel mgxs")
}

/// 2-group MGXS for water moderator. No fission. Strong thermalisation
/// via 0→1 down-scatter and 1→1 self-scatter.
fn moderator_mgxs() -> MaterialMgxs {
    let scatter = vec![
        0.6, 0.4, // from fast: 0->0=0.6, 0->1=0.4 (heavy down-scatter)
        0.001, 1.5, // from thermal: 1->0=0.001, 1->1=1.5
    ];
    let sigma_t = vec![1.05, 1.6];
    let sigma_a = vec![0.005, 0.01];
    MaterialMgxs::nonfissionable(sigma_t, sigma_a, scatter).expect("mod mgxs")
}

/// Build the pin-cell geometry: fuel cylinder (CylinderZ) inside a
/// 1.26 × 1.26 × 10 cm box with reflective xy faces and reflective z.
fn pincell_geometry() -> Geometry {
    let half = 0.5 * PITCH;
    let surfaces = vec![
        // 0: outer box -x face (reflective)
        Surface::PlaneX {
            x0: -half,
            bc: BoundaryCondition::Reflective,
        },
        // 1: outer box +x face
        Surface::PlaneX {
            x0: half,
            bc: BoundaryCondition::Reflective,
        },
        // 2: outer box -y face
        Surface::PlaneY {
            y0: -half,
            bc: BoundaryCondition::Reflective,
        },
        // 3: outer box +y face
        Surface::PlaneY {
            y0: half,
            bc: BoundaryCondition::Reflective,
        },
        // 4/5: top/bottom z (reflective — 2D effective)
        Surface::PlaneZ {
            z0: -HEIGHT_HALF,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneZ {
            z0: HEIGHT_HALF,
            bc: BoundaryCondition::Reflective,
        },
        // 6: fuel cylinder (axis = Z, transmission BC — internal)
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: FUEL_RADIUS,
            bc: BoundaryCondition::Transmission,
        },
    ];
    let inside_box = cell::intersect_all(vec![
        cell::outside(0),
        cell::inside(1),
        cell::outside(2),
        cell::inside(3),
        cell::outside(4),
        cell::inside(5),
    ]);
    // Fuel cell: inside box AND inside cylinder
    let fuel_region = Region::Intersection(
        Box::new(inside_box.clone()),
        Box::new(cell::inside(6)), // -S6: inside cylinder
    );
    // Moderator cell: inside box AND outside cylinder
    let mod_region = Region::Intersection(
        Box::new(inside_box.clone()),
        Box::new(cell::outside(6)), // +S6: outside cylinder
    );
    // Outside-everything (void, for off-mesh rays).
    let outside_region = Region::Complement(Box::new(inside_box));
    let cells = vec![
        Cell::new(CellId(0), fuel_region, CellFill::Material(0)),
        Cell::new(CellId(1), mod_region, CellFill::Material(1)),
        Cell::new(CellId(2), outside_region, CellFill::Void),
    ];
    Geometry::flat(surfaces, cells).expect("pin cell geometry")
}

fn main() {
    let geom = pincell_geometry();
    let library = MgxsLibrary::new(vec![fuel_mgxs(), moderator_mgxs()]).expect("lib");

    let half = 0.5 * PITCH;
    let aabb = Aabb::new(
        Vec3::new(-half, -half, -HEIGHT_HALF),
        Vec3::new(half, half, HEIGHT_HALF),
    );

    // Cell-based FSRs with analytic volumes — 1 FSR for fuel, 1 for
    // moderator. Volumes: fuel = π·r²·H_total; mod = pitch² · H_total - fuel.
    let h_total = 2.0 * HEIGHT_HALF;
    let fuel_vol = std::f64::consts::PI * FUEL_RADIUS * FUEL_RADIUS * h_total;
    let mod_vol = PITCH * PITCH * h_total - fuel_vol;
    let mut analytic = HashMap::new();
    // Cell-based keys are (cell_idx, lattice_element). No lattice here →
    // (cell_idx, None).
    analytic.insert((0_u32, None), fuel_vol);
    analytic.insert((1_u32, None), mod_vol);
    let mesh = FsrMesh::cell_based(aabb, &geom, [16, 16, 4], Some(&analytic));
    println!(
        "Cell-based FSRs discovered: {} (expected 2 — fuel + moderator)",
        mesh.n_fsrs()
    );
    for f in 0..mesh.n_fsrs() {
        println!(
            "  FSR {}: material = {}, analytic V = {:.4} cm³",
            f,
            mesh.material[f],
            mesh.fsr_volume(f)
        );
    }

    // Run forward eigenvalue — both mortal and immortal modes for
    // comparison. Then forward + adjoint to validate adjoint identity.
    println!("\n=== Forward k-eigenvalue (mortal) ===");
    let cfg_mortal = RaySolverConfig {
        rays_per_batch: 4000,
        dead_zone: 1.0,
        active_length: 20.0,
        batches: 200,
        inactive: 60,
        mode: SolverMode::Eigenvalue,
        adjoint: AdjointFlag::Forward,
        seed: 1,
        immortal: false,
    };
    let solver_mortal = RandomRaySolver::new(&geom, mesh.clone(), library.clone());
    let t0 = Instant::now();
    let r_mortal = solver_mortal.run(&cfg_mortal);
    let dt_mortal = t0.elapsed().as_secs_f64();
    println!(
        "k_eff (mortal)   = {:.5}    (active batches = {}, wall = {:.2}s)",
        r_mortal.k_eff, r_mortal.n_active_batches, dt_mortal
    );
    print_per_fsr_flux(&r_mortal);

    println!("\n=== Forward k-eigenvalue (immortal) ===");
    let cfg_immortal = RaySolverConfig {
        immortal: true,
        dead_zone: 0.0,
        seed: 2,
        ..cfg_mortal.clone()
    };
    let solver_immortal = RandomRaySolver::new(&geom, mesh.clone(), library.clone());
    let t0 = Instant::now();
    let r_immortal = solver_immortal.run(&cfg_immortal);
    let dt_immortal = t0.elapsed().as_secs_f64();
    println!(
        "k_eff (immortal) = {:.5}    (active batches = {}, wall = {:.2}s)",
        r_immortal.k_eff, r_immortal.n_active_batches, dt_immortal
    );
    print_per_fsr_flux(&r_immortal);

    println!("\n=== Adjoint identity check ===");
    let cfg_adj = RaySolverConfig {
        adjoint: AdjointFlag::Adjoint,
        seed: 3,
        ..cfg_mortal.clone()
    };
    let solver_adj = RandomRaySolver::new(&geom, mesh.clone(), library.clone());
    let r_adj = solver_adj.run(&cfg_adj);
    let dk_pcm = (r_mortal.k_eff - r_adj.k_eff).abs() / r_mortal.k_eff * 1e5;
    println!("k_eff (forward)  = {:.5}", r_mortal.k_eff);
    println!("k_eff (adjoint)  = {:.5}", r_adj.k_eff);
    println!("Δ                = {:.1} pcm", dk_pcm);

    println!("\n=== Summary ===");
    println!(
        "Forward mortal vs immortal Δ = {:.1} pcm",
        ((r_mortal.k_eff - r_immortal.k_eff).abs() / r_mortal.k_eff) * 1e5
    );
    println!("Forward vs adjoint Δ         = {:.1} pcm", dk_pcm);
    println!(
        "Wall: mortal {:.2}s vs immortal {:.2}s ({:.2}× ratio)",
        dt_mortal,
        dt_immortal,
        dt_immortal / dt_mortal.max(1e-9)
    );
}

fn print_per_fsr_flux(r: &open_rust_mc::random_ray::SolverResult) {
    let n_g = r.n_groups;
    println!("Per-FSR scalar flux:");
    for f in 0..r.n_fsrs {
        let g0 = r.phi[f * n_g];
        let g1 = r.phi[f * n_g + 1];
        let ratio = if g0 > 0.0 { g1 / g0 } else { 0.0 };
        println!(
            "  FSR {}: φ_fast = {:.4e}, φ_thermal = {:.4e}, ratio = {:.3}",
            f, g0, g1, ratio
        );
    }
}
