//! Universes — collections of cells that tile a region of space.
//!
//! A universe contains a set of cells. One universe is the "root" of the
//! geometry. Cells can be filled with another universe for nested geometry
//! (e.g., a fuel pin universe repeated inside a lattice).


/// Unique identifier for a universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniverseId(pub u32);

/// A universe: a set of cell indices that tile space.
#[derive(Debug, Clone)]
pub struct Universe {
    pub id: UniverseId,
    /// Indices into the global cells array.
    pub cell_indices: Vec<usize>,
}

impl Universe {
    pub fn new(id: UniverseId, cell_indices: Vec<usize>) -> Self {
        Self { id, cell_indices }
    }
}
