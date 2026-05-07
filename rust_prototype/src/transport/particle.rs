//! Particle state and bank management.
//!
//! Individual particle struct for history tracking + fission bank.
//! A future SoA bank for batch processing is planned.

use crate::geometry::coord::{Coord, CoordStack};
use crate::geometry::{UniverseId, Vec3};
use smallvec::smallvec;

/// Status of a particle during transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleStatus {
    /// Actively being transported.
    Alive,
    /// Absorbed or leaked — history over.
    Dead,
}

/// A single neutron being transported.
#[derive(Debug, Clone)]
pub struct Particle {
    /// Position in cm.
    pub pos: Vec3,
    /// Direction (unit vector).
    pub dir: Vec3,
    /// Energy in eV.
    pub energy: f64,
    /// Statistical weight.
    pub weight: f64,
    /// Index of the cell the particle is in (always equal to
    /// `coord_stack.deepest_cell_idx()`; kept as a separate field for
    /// fast access in the hot loop).
    pub cell_idx: usize,
    /// Coordinate stack through the recursive geometry. For flat
    /// (single-universe) geometries this is always depth 1.
    pub coord_stack: CoordStack,
    /// Status.
    pub status: ParticleStatus,
    /// Number of collisions so far.
    pub n_collisions: u32,
}

impl Particle {
    /// Create a new particle with a single root-universe coordinate
    /// frame. Used by callers that don't (yet) construct a full
    /// recursive `CoordStack` — assumes depth-1 geometry.
    pub fn new(pos: Vec3, dir: Vec3, energy: f64, cell_idx: usize) -> Self {
        let coord_stack: CoordStack = smallvec![Coord::root(UniverseId(0), cell_idx as u32)];
        Self {
            pos,
            dir,
            energy,
            weight: 1.0,
            cell_idx,
            coord_stack,
            status: ParticleStatus::Alive,
            n_collisions: 0,
        }
    }

    /// Create a new particle with an explicit coordinate stack. Used
    /// when the geometry is nested (Universe/Lattice fills) and the
    /// caller has already resolved the stack via
    /// `find_cell_recursive`.
    pub fn with_stack(pos: Vec3, dir: Vec3, energy: f64, coord_stack: CoordStack) -> Self {
        let cell_idx = coord_stack.last().map(|c| c.cell_idx as usize).unwrap_or(0);
        Self {
            pos,
            dir,
            energy,
            weight: 1.0,
            cell_idx,
            coord_stack,
            status: ParticleStatus::Alive,
            n_collisions: 0,
        }
    }

    /// Move the particle a distance along its current direction.
    #[inline]
    pub fn advance(&mut self, distance: f64) {
        self.pos = self.pos + self.dir * distance;
    }

    /// Kill the particle.
    #[inline]
    pub fn kill(&mut self) {
        self.status = ParticleStatus::Dead;
    }

    /// Is the particle still alive?
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.status == ParticleStatus::Alive
    }
}

/// A fission site — records where a fission happened for the next generation.
#[derive(Debug, Clone)]
pub struct FissionSite {
    pub pos: Vec3,
    pub energy: f64,
    pub weight: f64,
}

/// Fission bank — collects fission sites during a generation.
#[derive(Debug, Default)]
pub struct FissionBank {
    pub sites: Vec<FissionSite>,
}

impl FissionBank {
    pub fn new() -> Self {
        Self { sites: Vec::new() }
    }

    pub fn push(&mut self, site: FissionSite) {
        self.sites.push(site);
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub fn clear(&mut self) {
        self.sites.clear();
    }
}
