//! `Geometry` — the canonical owner of surfaces, cells, universes and
//! lattices for a problem.
//!
//! Transport hot paths still take `&[Surface]` / `&[Cell]` slices for
//! cache-friendliness; `Geometry` is the construction-time owner that
//! validates the structure and hands those slices out via accessors.

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
    #[error("region in cell {cell} references surface index {surface} but only {n_surfaces} surfaces exist")]
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
#[derive(Debug, Clone)]
pub struct Geometry {
    pub surfaces: Vec<Surface>,
    pub cells: Vec<Cell>,
    pub universes: Vec<Universe>,
    pub lattices: Vec<RectLattice>,
    pub root_universe: UniverseId,
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
                CellFill::Material(_) | CellFill::Void => {}
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

        Ok(Self {
            surfaces,
            cells,
            universes,
            lattices,
            root_universe,
        })
    }

    /// Construct a "flat" geometry with a single root universe owning
    /// all the cells. Convenience for existing single-level call sites.
    pub fn flat(surfaces: Vec<Surface>, cells: Vec<Cell>) -> Result<Self, GeometryError> {
        let cell_indices: Vec<usize> = (0..cells.len()).collect();
        let universes = vec![Universe::new(UniverseId(0), cell_indices)];
        Self::new(surfaces, cells, universes, Vec::new(), UniverseId(0))
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
    pub fn root(&self) -> &Universe {
        self.universe(self.root_universe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::cell::{self, CellId};
    use crate::geometry::surface::BoundaryCondition;
    use crate::geometry::Vec3;

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
        matches!(
            err,
            GeometryError::CellIndexOutOfRange {
                cell: 99,
                ..
            }
        );
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
            universes: vec![
                UniverseId(0),
                UniverseId(0),
                UniverseId(0),
                UniverseId(99),
            ],
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
