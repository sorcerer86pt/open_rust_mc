//! Particle state and fission bank.

use crate::geometry::coord::{Coord, CoordStack};
use crate::geometry::{UniverseId, Vec3};
use smallvec::smallvec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleStatus {
    Alive,
    Dead,
}

#[derive(Debug, Clone)]
pub struct Particle {
    /// cm.
    pub pos: Vec3,
    /// Unit vector.
    pub dir: Vec3,
    /// eV.
    pub energy: f64,
    pub weight: f64,
    /// Always equal to `coord_stack.deepest_cell_idx()`; cached for
    /// hot-loop speed.
    pub cell_idx: usize,
    pub coord_stack: CoordStack,
    pub status: ParticleStatus,
    pub n_collisions: u32,
}

impl Particle {
    /// Flat-geometry constructor; depth-1 CoordStack at universe 0.
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

    /// Caller has already resolved the stack via `find_cell_recursive`.
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

    #[inline]
    pub fn advance(&mut self, distance: f64) {
        self.pos = self.pos + self.dir * distance;
    }

    #[inline]
    pub fn kill(&mut self) {
        self.status = ParticleStatus::Dead;
    }

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.status == ParticleStatus::Alive
    }
}

#[derive(Debug, Clone)]
pub struct FissionSite {
    pub pos: Vec3,
    pub energy: f64,
    pub weight: f64,
}

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
