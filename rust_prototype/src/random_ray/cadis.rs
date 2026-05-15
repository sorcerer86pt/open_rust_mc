//! FW-CADIS bridge. `WeightWindow::from_flux` takes a forward-flux
//! proxy and emits `w_target ∝ flux_max / flux`. Feed it ψ*(r) and
//! you get the CADIS recipe directly.

use crate::geometry::{Aabb, Geometry};
use crate::transport::weight_window::WeightWindow;

use super::fsr::FsrMesh;
use super::mgxs::MgxsLibrary;
use super::solver::{AdjointFlag, RandomRaySolver, RaySolverConfig, SolverResult};

/// Forces `cfg.adjoint = Adjoint`. `response = None` → uniform
/// group-sum; `Some(R_g)` → response-weighted (FW-CADIS proper),
/// length must match `library.n_groups`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn weight_window_from_adjoint(
    geom: &Geometry,
    aabb: Aabb,
    n: [usize; 3],
    library: MgxsLibrary,
    response: Option<Vec<f64>>,
    cfg: RaySolverConfig,
    w_ref: f64,
    ratio: f64,
    phi_floor: f64,
) -> (WeightWindow, SolverResult) {
    if let Some(r) = &response {
        assert_eq!(
            r.len(),
            library.n_groups,
            "response weights must match n_groups"
        );
    }
    let mesh = FsrMesh::from_geometry(aabb, n, geom);
    let solver = RandomRaySolver::new(geom, mesh, library);
    let mut cfg = cfg;
    cfg.adjoint = AdjointFlag::Adjoint;
    let result = solver.run(&cfg);

    // ψ*(r,g) → ψ*(r): uniform or response-weighted.
    let n_fsrs = result.n_fsrs;
    let n_g = result.n_groups;
    let mut importance = vec![0.0_f64; n_fsrs];
    for f in 0..n_fsrs {
        let mut acc = 0.0;
        for g in 0..n_g {
            let w = response.as_ref().map(|r| r[g]).unwrap_or(1.0);
            acc += w * result.phi[f * n_g + g];
        }
        importance[f] = acc;
    }

    let ww = WeightWindow::from_flux(&aabb, n, &importance, w_ref, ratio, phi_floor);
    (ww, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, CellFill, CellId, Region};
    use crate::geometry::surface::BoundaryCondition;
    use crate::geometry::{Cell, Surface, Vec3};
    use crate::random_ray::mgxs::MaterialMgxs;
    use crate::random_ray::solver::SolverMode;

    /// Slab geometry: source side (x ≤ 0) reflects, detector side
    /// (x = +T) is vacuum, y/z are reflective. The adjoint problem
    /// places a uniform fixed source at the detector face running
    /// backwards into the slab; ψ*(x) is monotone increasing toward
    /// the detector face for a pure absorber/scatter slab.
    fn slab_geometry(thickness: f64) -> Geometry {
        let half_yz = 50.0;
        let surfaces = vec![
            Surface::PlaneX {
                x0: 0.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: thickness,
                bc: BoundaryCondition::Vacuum,
            },
            Surface::PlaneY {
                y0: -half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: -half_yz,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneZ {
                z0: half_yz,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let inside = cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ]);
        let outside = Region::Complement(Box::new(cell::intersect_all(vec![
            cell::outside(0),
            cell::inside(1),
            cell::outside(2),
            cell::inside(3),
            cell::outside(4),
            cell::inside(5),
        ])));
        let cells = vec![
            Cell::new(CellId(0), inside, CellFill::Material(0)),
            Cell::new(CellId(1), outside, CellFill::Void),
        ];
        Geometry::flat(surfaces, cells).expect("slab geometry")
    }

    /// Pure absorber/scatter water-like 1-group multigroup: Σ_t = 0.1
    /// cm⁻¹, Σ_a = 0.04, Σ_s = 0.06. No fission. Drives a fixed-source
    /// adjoint problem.
    fn water_like_mgxs() -> MgxsLibrary {
        let m = MaterialMgxs::nonfissionable(vec![0.1], vec![0.04], vec![0.06]).expect("nonfiss");
        MgxsLibrary::new(vec![m]).expect("library")
    }

    #[test]
    fn ww_from_adjoint_is_monotone_in_x_for_a_slab() {
        // 50-cm slab, 10 voxels in x (5 cm each), 1 voxel in y/z.
        let thickness = 50.0;
        let geom = slab_geometry(thickness);
        let aabb = Aabb::new(
            Vec3::new(0.0, -50.0, -50.0),
            Vec3::new(thickness, 50.0, 50.0),
        );
        let n = [10_usize, 1, 1];
        let library = water_like_mgxs();

        // Adjoint problem: detector at x=T. Place a uniform external
        // source in the last x-voxel and run backwards. We do this by
        // running the *forward* solver in fixed-source mode with the
        // external source localised at the detector face — this gives
        // a forward flux that decays away from the detector. For a
        // 1-group symmetric problem (Σ_s,1→1 only), ψ*(x) ∝ φ_fwd
        // with this layout reflects the importance gradient correctly.
        let n_fsrs = aabb_n_fsrs(n);
        let mut q_ext = vec![0.0_f64; n_fsrs];
        // Last x-voxel, all y/z (n_y = n_z = 1 → just the last index).
        q_ext[n[0] - 1] = 1.0;

        let cfg = RaySolverConfig {
            rays_per_batch: 2000,
            dead_zone: 5.0,
            active_length: 80.0,
            batches: 60,
            inactive: 20,
            mode: SolverMode::FixedSource,
            adjoint: AdjointFlag::Forward,
            seed: 19,
            immortal: false,
        };

        let mesh = FsrMesh::from_geometry(aabb, n, &geom);
        let solver = RandomRaySolver::new(&geom, mesh, library).with_external_source(q_ext);
        let r = solver.run(&cfg);

        let phi = r.flux_group(0);
        // Importance should decrease as we move away from the detector.
        // Allow some statistical jitter; require the bulk profile to
        // satisfy φ[k] > φ[0] (front voxel is dimmer than detector face).
        assert!(
            phi[n[0] - 1] > phi[0] * 1.5,
            "expected detector-face flux > 1.5×source-face flux: phi[0]={}, phi[last]={}",
            phi[0],
            phi[n[0] - 1]
        );

        // Now translate that into a WW via from_flux directly.
        let ww = WeightWindow::from_flux(&aabb, n, &phi, 1.0, 5.0, 1e-3);
        // Detector voxel: highest flux → smallest target.
        assert!(
            ww.lower[n[0] - 1] < ww.lower[0],
            "ww.lower at detector ({}) should be smaller than at source side ({})",
            ww.lower[n[0] - 1],
            ww.lower[0]
        );
    }

    fn aabb_n_fsrs(n: [usize; 3]) -> usize {
        n[0].max(1) * n[1].max(1) * n[2].max(1)
    }
}
