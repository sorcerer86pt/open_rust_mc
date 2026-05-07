//! `Geometry` — the canonical owner of surfaces, cells, universes and
//! lattices for a problem.
//!
//! Transport hot paths still take `&[Surface]` / `&[Cell]` slices for
//! cache-friendliness; `Geometry` is the construction-time owner that
//! validates the structure and hands those slices out via accessors.

use super::bvh::Bvh;
use super::cell::CellFill;
use super::{Cell, LatticeId, RectLattice, Surface, Universe, UniverseId};

/// Errors that can be returned from `Geometry::new`.
#[derive(Debug, thiserror::Error)]
pub enum GeometryError {
    #[error("root universe id {0} is out of range (have {1} universes)")]
    RootUniverseOutOfRange(u32, usize),
    #[error("universe {universe} references cell index {cell} but only {n_cells} cells exist")]
    CellIndexOutOfRange {
        universe: u32,
        cell: usize,
        n_cells: usize,
    },
    #[error("cell {cell} fills with universe {universe} but only {n_universes} universes exist")]
    UniverseFillOutOfRange {
        cell: usize,
        universe: u32,
        n_universes: usize,
    },
    #[error("cell {cell} fills with lattice {lattice} but only {n_lattices} lattices exist")]
    LatticeFillOutOfRange {
        cell: usize,
        lattice: u32,
        n_lattices: usize,
    },
    #[error(
        "lattice {lattice} references universe id {universe} but only {n_universes} universes exist"
    )]
    LatticeUniverseOutOfRange {
        lattice: u32,
        universe: u32,
        n_universes: usize,
    },
    #[error(
        "region in cell {cell} references surface index {surface} but only {n_surfaces} surfaces exist"
    )]
    SurfaceIndexOutOfRange {
        cell: usize,
        surface: usize,
        n_surfaces: usize,
    },
}

/// A complete Monte Carlo geometry: surfaces, cells, universes, lattices.
///
/// `surfaces` and `cells` are flat global arrays. `Cell.fill` may point
/// at a material, void, another universe (`CellFill::Universe`), or a
/// lattice (`CellFill::Lattice`). `universes` partition cells into
/// reusable groups; `lattices` are regular tilings of universes.
///
/// Two pre-computed acceleration structures are derived from the rest
/// at construction time:
///   * `universe_surfaces[u]` — sorted, deduped list of every surface
///     index referenced by any cell in universe `u`. Lets
///     `find_cell_recursive` evaluate only relevant surfaces per
///     descent level instead of every global surface.
///   * `universe_bvhs[u]` — `Some(Bvh)` over the universe's cells if
///     every referenced cell has a finite AABB; `None` otherwise.
///     Falls back to linear scan when `None`.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub surfaces: Vec<Surface>,
    pub cells: Vec<Cell>,
    pub universes: Vec<Universe>,
    pub lattices: Vec<RectLattice>,
    /// Hex-grid lattices, filled in only by `with_hex_lattices`.
    /// `CellFill::HexLattice(idx)` indexes into this vec. Default
    /// empty so no existing geometry construction breaks.
    pub hex_lattices: Vec<crate::geometry::lattice::HexLattice>,
    pub root_universe: UniverseId,
    pub universe_surfaces: Vec<Vec<usize>>,
    pub universe_bvhs: Vec<Option<Bvh>>,
}

impl Geometry {
    /// Construct a `Geometry` after validating internal references.
    ///
    /// Validates:
    ///   - Every cell index referenced by a universe is in range.
    ///   - Every `CellFill::Universe` / `CellFill::Lattice` index is in
    ///     range.
    ///   - Every lattice element's universe id is in range.
    ///   - Every surface index referenced by a cell's region is in
    ///     range.
    ///   - `root_universe` is a valid universe index.
    ///
    /// Does *not* check space-partition completeness (no gaps / no
    /// overlaps in a universe's cells); that's a non-trivial CSG
    /// problem and is left to be caught empirically by transport tests.
    pub fn new(
        surfaces: Vec<Surface>,
        cells: Vec<Cell>,
        universes: Vec<Universe>,
        lattices: Vec<RectLattice>,
        root_universe: UniverseId,
    ) -> Result<Self, GeometryError> {
        let n_surfaces = surfaces.len();
        let n_cells = cells.len();
        let n_universes = universes.len();
        let n_lattices = lattices.len();

        if root_universe.0 as usize >= n_universes {
            return Err(GeometryError::RootUniverseOutOfRange(
                root_universe.0,
                n_universes,
            ));
        }

        for (u_idx, universe) in universes.iter().enumerate() {
            for &cell_idx in &universe.cell_indices {
                if cell_idx >= n_cells {
                    return Err(GeometryError::CellIndexOutOfRange {
                        universe: u_idx as u32,
                        cell: cell_idx,
                        n_cells,
                    });
                }
            }
        }

        for (cell_idx, cell) in cells.iter().enumerate() {
            match cell.fill {
                CellFill::Universe(u) => {
                    if (u as usize) >= n_universes {
                        return Err(GeometryError::UniverseFillOutOfRange {
                            cell: cell_idx,
                            universe: u,
                            n_universes,
                        });
                    }
                }
                CellFill::Lattice(l) => {
                    if (l as usize) >= n_lattices {
                        return Err(GeometryError::LatticeFillOutOfRange {
                            cell: cell_idx,
                            lattice: l,
                            n_lattices,
                        });
                    }
                }
                CellFill::HexLattice(_) | CellFill::Material(_) | CellFill::Void => {
                    // HexLattice indices are validated separately in
                    // `with_hex_lattices` since hex_lattices defaults
                    // to empty in the bare `new` constructor.
                }
            }

            let mut surface_indices = Vec::new();
            cell.region.surface_indices(&mut surface_indices);
            for s in surface_indices {
                if s >= n_surfaces {
                    return Err(GeometryError::SurfaceIndexOutOfRange {
                        cell: cell_idx,
                        surface: s,
                        n_surfaces,
                    });
                }
            }
        }

        for (l_idx, lattice) in lattices.iter().enumerate() {
            for u_id in &lattice.universes {
                if (u_id.0 as usize) >= n_universes {
                    return Err(GeometryError::LatticeUniverseOutOfRange {
                        lattice: l_idx as u32,
                        universe: u_id.0,
                        n_universes,
                    });
                }
            }
        }

        // Per-universe surface index lists — restricts
        // find_cell_recursive's surface evaluation loop to surfaces
        // actually referenced by cells in the current universe. Big
        // win for nested geometries where each universe references a
        // small subset of the global surface list.
        let mut universe_surfaces: Vec<Vec<usize>> = Vec::with_capacity(universes.len());
        for universe in &universes {
            let mut tmp: Vec<usize> = Vec::new();
            for &cell_idx in &universe.cell_indices {
                if let Some(cell) = cells.get(cell_idx) {
                    cell.region.surface_indices(&mut tmp);
                }
            }
            tmp.sort_unstable();
            tmp.dedup();
            universe_surfaces.push(tmp);
        }

        // Per-universe BVH — built only when every cell in the
        // universe has a finite AABB AND the universe is big enough
        // to amortise the BVH-traversal overhead. For small universes
        // (≤ MIN_CELLS_FOR_BVH cells) a flat linear scan over a
        // contiguous Vec wins on cache behaviour; the BVH only pays
        // off once the cell count grows past O(10).
        const MIN_CELLS_FOR_BVH: usize = 8;
        let mut universe_bvhs: Vec<Option<Bvh>> = Vec::with_capacity(universes.len());
        for universe in &universes {
            let all_finite = !universe.cell_indices.is_empty()
                && universe.cell_indices.iter().all(|&i| {
                    cells
                        .get(i)
                        .map(|c| c.aabb.surface_area().is_finite())
                        .unwrap_or(false)
                });
            if all_finite && universe.cell_indices.len() >= MIN_CELLS_FOR_BVH {
                universe_bvhs.push(Some(Bvh::build_subset(&cells, &universe.cell_indices)));
            } else {
                universe_bvhs.push(None);
            }
        }

        Ok(Self {
            surfaces,
            cells,
            universes,
            lattices,
            hex_lattices: Vec::new(),
            root_universe,
            universe_surfaces,
            universe_bvhs,
        })
    }

    /// Attach hex-grid lattices to a geometry built via `new` /
    /// `flat` / `from_slices`. Cells using `CellFill::HexLattice(idx)`
    /// must reference an index in range; the function does not
    /// re-validate the existing rect-lattice references.
    pub fn with_hex_lattices(
        mut self,
        hex_lattices: Vec<crate::geometry::lattice::HexLattice>,
    ) -> Result<Self, GeometryError> {
        let n_hex = hex_lattices.len();
        for (cell_idx, cell) in self.cells.iter().enumerate() {
            if let CellFill::HexLattice(h) = cell.fill
                && (h as usize) >= n_hex
            {
                return Err(GeometryError::LatticeFillOutOfRange {
                    cell: cell_idx,
                    lattice: h,
                    n_lattices: n_hex,
                });
            }
        }
        self.hex_lattices = hex_lattices;
        Ok(self)
    }

    /// Construct a "flat" geometry with a single root universe owning
    /// all the cells. Convenience for existing single-level call sites.
    pub fn flat(surfaces: Vec<Surface>, cells: Vec<Cell>) -> Result<Self, GeometryError> {
        let cell_indices: Vec<usize> = (0..cells.len()).collect();
        let universes = vec![Universe::new(UniverseId(0), cell_indices)];
        Self::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
    }

    /// Same as [`Self::flat`] but takes slices and clones — useful at
    /// transport entry points that have `&[Surface]` / `&[Cell]` from
    /// existing APIs.
    pub fn from_slices(surfaces: &[Surface], cells: &[Cell]) -> Result<Self, GeometryError> {
        Self::flat(surfaces.to_vec(), cells.to_vec())
    }

    #[inline]
    pub fn universe(&self, id: UniverseId) -> &Universe {
        &self.universes[id.0 as usize]
    }

    #[inline]
    pub fn lattice(&self, id: LatticeId) -> &RectLattice {
        &self.lattices[id.0 as usize]
    }

    #[inline]
    pub fn hex_lattice(
        &self,
        id: crate::geometry::HexLatticeId,
    ) -> &crate::geometry::lattice::HexLattice {
        &self.hex_lattices[id.0 as usize]
    }

    #[inline]
    pub fn root(&self) -> &Universe {
        self.universe(self.root_universe)
    }

    /// Resolve the effective material index for a particle's deepest
    /// cell, applying any per-lattice-element override.
    ///
    /// Returns `Some(material_idx)` when the deepest cell is a
    /// `Material` cell (or a cell whose static fill the user
    /// rebound via `RectLattice::material_overrides`). Returns
    /// `None` for `Void` cells. Falls through to the static fill
    /// when the deepest stack frame isn't inside a lattice or when
    /// the lattice has no override for this cell.
    pub fn effective_material_idx(&self, stack: &super::coord::CoordStack) -> EffectiveFill {
        let Some(deepest) = stack.last() else {
            return EffectiveFill::Void;
        };
        let cell_idx = deepest.cell_idx as usize;

        if let Some((lattice_id, element)) = deepest.lattice {
            let lattice = self.lattice(lattice_id);
            if let Some(mat) = lattice.material_override(element, cell_idx) {
                return EffectiveFill::Material(mat);
            }
        }

        match self.cells.get(cell_idx).map(|c| c.fill) {
            Some(CellFill::Material(m)) => EffectiveFill::Material(m),
            Some(CellFill::Void) => EffectiveFill::Void,
            // Universe / Lattice should never be the deepest cell —
            // find_cell_recursive descends through them. Treat as
            // void leakage if it does happen.
            _ => EffectiveFill::Void,
        }
    }
}

/// What `Geometry::effective_material_idx` resolves to: a material
/// index (with any lattice override applied) or void.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveFill {
    Material(u32),
    Void,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::geometry::cell::{self, CellId};
    use crate::geometry::surface::BoundaryCondition;

    fn godiva_surfaces_and_cells() -> (Vec<Surface>, Vec<Cell>) {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 8.7407,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        (surfaces, cells)
    }

    #[test]
    fn flat_geometry_constructs_root_universe() {
        let (surfaces, cells) = godiva_surfaces_and_cells();
        let geom = Geometry::flat(surfaces, cells).expect("flat construction");
        assert_eq!(geom.root().cell_indices, vec![0, 1]);
        assert_eq!(geom.universes.len(), 1);
        assert_eq!(geom.lattices.len(), 0);
    }

    #[test]
    fn rejects_out_of_range_root() {
        let (surfaces, cells) = godiva_surfaces_and_cells();
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        let err = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(7))
            .expect_err("should reject");
        matches!(err, GeometryError::RootUniverseOutOfRange(7, 1));
    }

    #[test]
    fn rejects_universe_referring_to_missing_cell() {
        let (surfaces, cells) = godiva_surfaces_and_cells();
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1, 99])];
        let err = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect_err("should reject");
        matches!(err, GeometryError::CellIndexOutOfRange { cell: 99, .. });
    }

    #[test]
    fn rejects_cell_filled_with_missing_universe() {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Universe(5)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        let err = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect_err("should reject");
        matches!(
            err,
            GeometryError::UniverseFillOutOfRange {
                cell: 0,
                universe: 5,
                ..
            }
        );
    }

    #[test]
    fn rejects_cell_filled_with_missing_lattice() {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![
            Cell::new(CellId(0), cell::inside(0), CellFill::Lattice(2)),
            Cell::new(CellId(1), cell::outside(0), CellFill::Void),
        ];
        let universes = vec![Universe::new(UniverseId(0), vec![0, 1])];
        let err = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect_err("should reject");
        matches!(
            err,
            GeometryError::LatticeFillOutOfRange {
                cell: 0,
                lattice: 2,
                ..
            }
        );
    }

    #[test]
    fn material_override_resolves_per_lattice_element() {
        use crate::geometry::coord::CoordStack;
        use crate::geometry::lattice::MaterialOverrideMap;
        use std::collections::HashMap;

        // Geometry: 2×2 lattice of a single pin universe. Cell 0 of
        // the pin universe is fuel (static fill = Material(0)).
        // Element (0,0) and (1,1) override cell 0 to Material(2);
        // the other two elements use the static fill.
        //
        // Without the override, a particle whose deepest stack frame
        // is "lattice element X, cell 0" should resolve to material 0.
        // With the override applied, elements (0,0) and (1,1) should
        // resolve to material 2; (1,0) and (0,1) stay at material 0.
        let surfaces = vec![
            // 0: pin cylinder at element-local (0.5, 0.5) R=0.3
            Surface::CylinderZ {
                center_x: 0.5,
                center_y: 0.5,
                radius: 0.3,
                bc: BoundaryCondition::Transmission,
            },
            // 1..4: outer box at +/- 1
            Surface::PlaneX {
                x0: -1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneX {
                x0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: -1.0,
                bc: BoundaryCondition::Reflective,
            },
            Surface::PlaneY {
                y0: 1.0,
                bc: BoundaryCondition::Reflective,
            },
        ];
        let cells = vec![
            // 0: fuel (static fill = Material 0)
            Cell::new(CellId(0), cell::inside(0), CellFill::Material(0)),
            // 1: water (static fill = Material 1)
            Cell::new(CellId(1), cell::outside(0), CellFill::Material(1)),
            // 2: root cell — lattice fill
            Cell::new(
                CellId(2),
                cell::intersect_all(vec![
                    cell::outside(1),
                    cell::inside(2),
                    cell::outside(3),
                    cell::inside(4),
                ]),
                CellFill::Lattice(0),
            ),
        ];
        let universes = vec![
            Universe::new(UniverseId(0), vec![2]),    // root
            Universe::new(UniverseId(1), vec![0, 1]), // pin
        ];

        // Per-element override maps: element (0,0) and (1,1) override
        // cell 0 → material 2.
        let mut e00 = MaterialOverrideMap::new();
        e00.insert(0, 2);
        let mut e10 = MaterialOverrideMap::new();
        let mut e01 = MaterialOverrideMap::new();
        let mut e11 = MaterialOverrideMap::new();
        e11.insert(0, 2);
        // Empty maps for elements without overrides — explicit so the
        // Vec aligns with the row-major layout.
        let _ = (&mut e10, &mut e01);

        let lattice = RectLattice {
            origin: Vec3::new(-1.0, -1.0, -1e6),
            pitch: Vec3::new(1.0, 1.0, 2e6),
            shape: [2, 2, 1],
            universes: vec![UniverseId(1); 4],
            material_overrides: Some(vec![e00, HashMap::new(), HashMap::new(), e11]),
        };

        let geom = Geometry::new(surfaces, cells, universes, vec![lattice], UniverseId(0))
            .expect("geometry");

        let lookup_at = |element: [i32; 3], cell_idx: u32| -> EffectiveFill {
            use crate::geometry::Coord;
            let stack: CoordStack = smallvec::smallvec![
                Coord::root(UniverseId(0), 2),
                Coord {
                    universe: UniverseId(1),
                    cell_idx,
                    lattice: Some((LatticeId(0), element)),
                    hex_lattice: None,
                    offset: Vec3::new(0.0, 0.0, 0.0),
                    rotation: None,
                },
            ];
            geom.effective_material_idx(&stack)
        };

        // Cell 0 (fuel) lookup at every element.
        assert_eq!(
            lookup_at([0, 0, 0], 0),
            EffectiveFill::Material(2),
            "(0,0): override cell 0 → mat 2"
        );
        assert_eq!(
            lookup_at([1, 0, 0], 0),
            EffectiveFill::Material(0),
            "(1,0): no override → static fill mat 0"
        );
        assert_eq!(
            lookup_at([0, 1, 0], 0),
            EffectiveFill::Material(0),
            "(0,1): no override → static fill mat 0"
        );
        assert_eq!(
            lookup_at([1, 1, 0], 0),
            EffectiveFill::Material(2),
            "(1,1): override cell 0 → mat 2"
        );

        // Cell 1 (water) — never overridden, always material 1.
        assert_eq!(lookup_at([0, 0, 0], 1), EffectiveFill::Material(1));
        assert_eq!(lookup_at([1, 1, 0], 1), EffectiveFill::Material(1));
    }

    #[test]
    fn rejects_lattice_referring_to_missing_universe() {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![Cell::new(CellId(0), cell::inside(0), CellFill::Material(0))];
        let universes = vec![Universe::new(UniverseId(0), vec![0])];
        let lattices = vec![RectLattice {
            origin: Vec3::new(0.0, 0.0, 0.0),
            pitch: Vec3::new(1.0, 1.0, 1.0),
            shape: [2, 2, 1],
            universes: vec![UniverseId(0), UniverseId(0), UniverseId(0), UniverseId(99)],
            material_overrides: None,
        }];
        let err = Geometry::new(surfaces, cells, universes, lattices, UniverseId(0))
            .expect_err("should reject");
        matches!(
            err,
            GeometryError::LatticeUniverseOutOfRange {
                lattice: 0,
                universe: 99,
                ..
            }
        );
    }

    #[test]
    fn rejects_region_referring_to_missing_surface() {
        let surfaces = vec![Surface::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            bc: BoundaryCondition::Vacuum,
        }];
        let cells = vec![Cell::new(
            CellId(0),
            cell::inside(99),
            CellFill::Material(0),
        )];
        let universes = vec![Universe::new(UniverseId(0), vec![0])];
        let err = Geometry::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
            .expect_err("should reject");
        matches!(
            err,
            GeometryError::SurfaceIndexOutOfRange {
                cell: 0,
                surface: 99,
                ..
            }
        );
    }
}
