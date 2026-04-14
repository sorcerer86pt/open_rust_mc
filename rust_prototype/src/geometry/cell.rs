//! Cells — regions of space defined by boolean combinations of surface half-spaces.
//!
//! A cell region is represented as a boolean expression:
//!   - `HalfSpace(surface_id, positive)` — one side of a surface
//!   - `Intersection(a, b)` — a AND b
//!   - `Union(a, b)` — a OR b
//!   - `Complement(a)` — NOT a
//!
//! This is stored as an algebraic data type (enum), not a class hierarchy.

use super::Aabb;

/// Unique identifier for a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub u32);

/// Material filling a cell — either a material index or another universe.
#[derive(Debug, Clone, Copy)]
pub enum CellFill {
    /// Index into the materials array.
    Material(u32),
    /// Index into the universes array (for nested geometry).
    Universe(u32),
    /// Void (no material).
    Void,
}

/// Boolean region expression — CSG in algebraic data type form.
#[derive(Debug, Clone)]
pub enum Region {
    /// One side of a surface: (surface_index, true = positive half-space).
    HalfSpace { surface_idx: usize, positive: bool },
    /// Intersection: both regions must contain the point.
    Intersection(Box<Region>, Box<Region>),
    /// Union: either region must contain the point.
    Union(Box<Region>, Box<Region>),
    /// Complement: point must NOT be in the region.
    Complement(Box<Region>),
}

impl Region {
    /// Test if a point is inside this region.
    ///
    /// `surface_evals` is a pre-computed array of surface evaluations at the point.
    /// This avoids redundant surface evaluations when multiple cells share surfaces.
    #[inline]
    pub fn contains(&self, surface_evals: &[f64]) -> bool {
        match self {
            Self::HalfSpace { surface_idx, positive } => {
                if *positive {
                    surface_evals[*surface_idx] > 0.0
                } else {
                    surface_evals[*surface_idx] < 0.0
                }
            }
            Self::Intersection(a, b) => {
                a.contains(surface_evals) && b.contains(surface_evals)
            }
            Self::Union(a, b) => {
                a.contains(surface_evals) || b.contains(surface_evals)
            }
            Self::Complement(a) => !a.contains(surface_evals),
        }
    }

    /// Collect all surface indices referenced by this region.
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
}

impl Cell {
    /// Create a new cell.
    pub fn new(id: CellId, region: Region, fill: CellFill) -> Self {
        Self {
            id,
            region,
            fill,
            temperature: 293.6, // default room temperature
            aabb: Aabb::INFINITE, // will be computed later
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
        Box::new(Region::HalfSpace { surface_idx: s1, positive: false }),
        Box::new(Region::HalfSpace { surface_idx: s2, positive: false }),
    )
}

/// Inside s1 but outside s2: -S1 ∩ +S2.
pub fn between(inner: usize, outer: usize) -> Region {
    Region::Intersection(
        Box::new(Region::HalfSpace { surface_idx: inner, positive: true }),
        Box::new(Region::HalfSpace { surface_idx: outer, positive: false }),
    )
}

/// Inside a single surface (negative half-space).
pub fn inside(s: usize) -> Region {
    Region::HalfSpace { surface_idx: s, positive: false }
}

/// Outside a single surface (positive half-space).
pub fn outside(s: usize) -> Region {
    Region::HalfSpace { surface_idx: s, positive: true }
}

/// Intersection of multiple half-spaces.
pub fn intersect_all(regions: Vec<Region>) -> Region {
    regions
        .into_iter()
        .reduce(|a, b| Region::Intersection(Box::new(a), Box::new(b)))
        .expect("need at least one region")
}
