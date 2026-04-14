//! Particle state and bank management.
//!
//! Individual particle struct for history tracking + fission bank.
//! A future SoA bank for batch processing is planned.

use crate::geometry::Vec3;

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
    /// Index of the cell the particle is in.
    pub cell_idx: usize,
    /// Status.
    pub status: ParticleStatus,
    /// Number of collisions so far.
    pub n_collisions: u32,
}

impl Particle {
    /// Create a new particle.
    pub fn new(pos: Vec3, dir: Vec3, energy: f64, cell_idx: usize) -> Self {
        Self {
            pos,
            dir,
            energy,
            weight: 1.0,
            cell_idx,
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
