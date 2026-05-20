// SPDX-License-Identifier: MIT
//! Universes tile space; cells may recursively be filled with
//! another universe (pin universe repeated inside a lattice).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniverseId(pub u32);

#[derive(Debug, Clone)]
pub struct Universe {
    pub id: UniverseId,
    /// Indices into `Geometry::cells`.
    pub cell_indices: Vec<usize>,
}

impl Universe {
    pub fn new(id: UniverseId, cell_indices: Vec<usize>) -> Self {
        Self { id, cell_indices }
    }
}
