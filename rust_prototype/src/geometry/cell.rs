//! CSG cells: half-space ∧/∨/¬ combinations.

use super::{Aabb, Mat3};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub u32);

#[derive(Debug, Clone, Copy)]
pub enum CellFill {
    Material(u32),
    Universe(u32),
    Lattice(u32),
    /// For VVER / FBR hex tessellations.
    HexLattice(u32),
    Void,
}

#[derive(Debug, Clone)]
pub enum Region {
    HalfSpace { surface_idx: usize, positive: bool },
    Intersection(Box<Region>, Box<Region>),
    Union(Box<Region>, Box<Region>),
    Complement(Box<Region>),
}

impl Region {
    /// `surface_evals` is pre-computed once per point, reused across
    /// cells that share surfaces.
    #[inline]
    pub fn contains(&self, surface_evals: &[f64]) -> bool {
        match self {
            Self::HalfSpace {
                surface_idx,
                positive,
            } => {
                if *positive {
                    surface_evals[*surface_idx] > 0.0
                } else {
                    surface_evals[*surface_idx] < 0.0
                }
            }
            Self::Intersection(a, b) => a.contains(surface_evals) && b.contains(surface_evals),
            Self::Union(a, b) => a.contains(surface_evals) || b.contains(surface_evals),
            Self::Complement(a) => !a.contains(surface_evals),
        }
    }

    pub fn surface_indices(&self, out: &mut Vec<usize>) {
        match self {
            Self::HalfSpace { surface_idx, .. } => out.push(*surface_idx),
            Self::Intersection(a, b) | Self::Union(a, b) => {
                a.surface_indices(out);
                b.surface_indices(out);
            }
            Self::Complement(a) => a.surface_indices(out),
        }
    }

    /// Conservative world-AABB via Roth 1982 / OpenCSG recursion.
    /// `Complement(_) → INF`. Plane HalfSpaces produce a slab on the
    /// gated axis. Used by `simulate::try_initial_source` to size
    /// per-cell rejection-sampling boxes (JSON loader sets
    /// `cell.aabb = Aabb::INFINITE`).
    pub fn world_aabb(&self, surfaces: &[crate::geometry::Surface]) -> crate::geometry::Aabb {
        use crate::geometry::{Aabb, Surface, Vec3};
        const INF: Aabb = Aabb {
            min: Vec3 {
                x: f64::NEG_INFINITY,
                y: f64::NEG_INFINITY,
                z: f64::NEG_INFINITY,
            },
            max: Vec3 {
                x: f64::INFINITY,
                y: f64::INFINITY,
                z: f64::INFINITY,
            },
        };
        match self {
            Self::HalfSpace {
                surface_idx,
                positive,
            } => {
                let s = match surfaces.get(*surface_idx) {
                    Some(s) => s,
                    None => return INF,
                };
                if !*positive {
                    // Inside a closed surface bounds the cell to the
                    // surface's own AABB. Planes return INFINITE here
                    // — the half-slab bound is handled below.
                    match s {
                        Surface::PlaneX { x0, .. } => Aabb::new(
                            Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                            Vec3::new(*x0, f64::INFINITY, f64::INFINITY),
                        ),
                        Surface::PlaneY { y0, .. } => Aabb::new(
                            Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                            Vec3::new(f64::INFINITY, *y0, f64::INFINITY),
                        ),
                        Surface::PlaneZ { z0, .. } => Aabb::new(
                            Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
                            Vec3::new(f64::INFINITY, f64::INFINITY, *z0),
                        ),
                        _ => s.aabb(),
                    }
                } else {
                    // Positive half-space: outside of a closed surface
                    // is unbounded; for a plane it's the opposite slab.
                    match s {
                        Surface::PlaneX { x0, .. } => Aabb::new(
                            Vec3::new(*x0, f64::NEG_INFINITY, f64::NEG_INFINITY),
                            Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                        ),
                        Surface::PlaneY { y0, .. } => Aabb::new(
                            Vec3::new(f64::NEG_INFINITY, *y0, f64::NEG_INFINITY),
                            Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                        ),
                        Surface::PlaneZ { z0, .. } => Aabb::new(
                            Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, *z0),
                            Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
                        ),
                        _ => INF,
                    }
                }
            }
            Self::Intersection(a, b) => {
                let aa = a.world_aabb(surfaces);
                let bb = b.world_aabb(surfaces);
                // Saturating per-axis intersection. `max(a, b)` on
                // f64 handles `-INF` correctly; `min` handles `+INF`.
                // Result is an empty AABB (min > max) if the regions
                // can't overlap on some axis — caller treats that as
                // an unbounded fallback.
                Aabb::new(
                    Vec3::new(aa.min.x.max(bb.min.x), aa.min.y.max(bb.min.y), aa.min.z.max(bb.min.z)),
                    Vec3::new(aa.max.x.min(bb.max.x), aa.max.y.min(bb.max.y), aa.max.z.min(bb.max.z)),
                )
            }
            Self::Union(a, b) => {
                let aa = a.world_aabb(surfaces);
                let bb = b.world_aabb(surfaces);
                Aabb::new(
                    Vec3::new(aa.min.x.min(bb.min.x), aa.min.y.min(bb.min.y), aa.min.z.min(bb.min.z)),
                    Vec3::new(aa.max.x.max(bb.max.x), aa.max.y.max(bb.max.y), aa.max.z.max(bb.max.z)),
                )
            }
            // The complement of a bounded region is unbounded; let
            // the caller fall back to the parent universe's AABB.
            Self::Complement(_) => INF,
        }
    }
}

/// A cell in the geometry.
#[derive(Debug, Clone)]
pub struct Cell {
    pub id: CellId,
    pub region: Region,
    pub fill: CellFill,
    /// Temperature of the cell (K). Used for cross-section lookup.
    pub temperature: f64,
    /// Pre-computed bounding box.
    pub aabb: Aabb,
    /// Optional rotation applied when descending into this cell's
    /// fill (`CellFill::Universe` or `CellFill::Lattice`). The
    /// rotation acts on the parent-frame coordinates *after* the
    /// translation step. `None` is equivalent to identity. Has no
    /// effect on `CellFill::Material` / `CellFill::Void` cells.
    pub rotation: Option<Mat3>,
}

impl Cell {
    /// Create a new cell.
    pub fn new(id: CellId, region: Region, fill: CellFill) -> Self {
        Self {
            id,
            region,
            fill,
            temperature: 293.6,   // default room temperature
            aabb: Aabb::INFINITE, // will be computed later
            rotation: None,
        }
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the bounding box.
    pub fn with_aabb(mut self, aabb: Aabb) -> Self {
        self.aabb = aabb;
        self
    }

    /// Set the rotation applied when descending into this cell's
    /// fill universe / lattice. `None` (the default) means identity.
    pub fn with_rotation(mut self, rotation: Mat3) -> Self {
        self.rotation = Some(rotation);
        self
    }

    /// Test if a point is inside this cell.
    #[inline]
    pub fn contains(&self, surface_evals: &[f64]) -> bool {
        self.region.contains(surface_evals)
    }
}

// ── Builder helpers for common cell definitions ────────────────────────────

/// Intersection of two half-spaces: -S1 ∩ -S2 (inside both surfaces).
pub fn inside_both(s1: usize, s2: usize) -> Region {
    Region::Intersection(
        Box::new(Region::HalfSpace {
            surface_idx: s1,
            positive: false,
        }),
        Box::new(Region::HalfSpace {
            surface_idx: s2,
            positive: false,
        }),
    )
}

/// Inside s1 but outside s2: -S1 ∩ +S2.
pub fn between(inner: usize, outer: usize) -> Region {
    Region::Intersection(
        Box::new(Region::HalfSpace {
            surface_idx: inner,
            positive: true,
        }),
        Box::new(Region::HalfSpace {
            surface_idx: outer,
            positive: false,
        }),
    )
}

/// Inside a single surface (negative half-space).
pub fn inside(s: usize) -> Region {
    Region::HalfSpace {
        surface_idx: s,
        positive: false,
    }
}

/// Outside a single surface (positive half-space).
pub fn outside(s: usize) -> Region {
    Region::HalfSpace {
        surface_idx: s,
        positive: true,
    }
}

/// Intersection of multiple half-spaces.
pub fn intersect_all(regions: Vec<Region>) -> Region {
    regions
        .into_iter()
        .reduce(|a, b| Region::Intersection(Box::new(a), Box::new(b)))
        .expect("need at least one region")
}
