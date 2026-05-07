//! Bounding Volume Hierarchy — O(log n) cell lookup.
//!
//! Builds a binary tree of AABBs over cells. Traversal skips entire
//! subtrees when the ray doesn't intersect the bounding box.
//! Construction uses the Surface Area Heuristic (SAH) for optimal splits.

use super::{Aabb, Cell, Surface, Vec3};

/// BVH node — either a leaf (single cell) or an internal node (two children).
#[derive(Debug, Clone)]
enum BvhNode {
    Leaf {
        cell_idx: usize,
        aabb: Aabb,
    },
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

/// The BVH acceleration structure.
#[derive(Debug, Clone)]
pub struct Bvh {
    root: Option<BvhNode>,
}

impl Bvh {
    /// Build a BVH from a set of cells.
    pub fn build(cells: &[Cell]) -> Self {
        if cells.is_empty() {
            return Self { root: None };
        }

        // Collect (cell_index, aabb, centroid) for cells with finite AABBs
        let mut entries: Vec<(usize, Aabb, Vec3)> = cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.aabb.surface_area().is_finite())
            .map(|(i, c)| (i, c.aabb, c.aabb.center()))
            .collect();

        if entries.is_empty() {
            return Self { root: None };
        }

        let root = build_recursive(&mut entries);
        Self { root: Some(root) }
    }

    /// Find which cell contains a point, using BVH acceleration.
    pub fn find_cell(&self, pos: Vec3, surfaces: &[Surface], cells: &[Cell]) -> Option<usize> {
        let root = self.root.as_ref()?;
        let evals: Vec<f64> = surfaces.iter().map(|s| s.evaluate(pos)).collect();
        find_cell_recursive(root, pos, &evals, cells)
    }

    /// Build a BVH over only the cells whose indices appear in
    /// `cell_indices`. Leaves carry the absolute (into `cells`)
    /// index, so a hit can be looked up directly in the global cells
    /// array. Returns an empty BVH if every referenced cell has a
    /// non-finite AABB (the caller must fall back to a linear scan in
    /// that case).
    pub fn build_subset(cells: &[Cell], cell_indices: &[usize]) -> Self {
        let mut entries: Vec<(usize, Aabb, Vec3)> = cell_indices
            .iter()
            .copied()
            .filter(|&i| {
                cells
                    .get(i)
                    .map(|c| c.aabb.surface_area().is_finite())
                    .unwrap_or(false)
            })
            .map(|i| (i, cells[i].aabb, cells[i].aabb.center()))
            .collect();
        if entries.is_empty() {
            return Self { root: None };
        }
        let root = build_recursive(&mut entries);
        Self { root: Some(root) }
    }

    /// Like `find_cell` but takes pre-computed surface evaluations,
    /// indexed by absolute surface idx. Cheap re-use for
    /// `find_cell_recursive`, which already keeps an `evals` buffer
    /// for the cell-region tests.
    #[inline]
    pub fn find_cell_with_evals(&self, pos: Vec3, evals: &[f64], cells: &[Cell]) -> Option<usize> {
        let root = self.root.as_ref()?;
        find_cell_recursive(root, pos, evals, cells)
    }

    /// True if the BVH has at least one leaf — useful for the
    /// "fallback to linear scan" decision in `find_cell_recursive`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
}

fn find_cell_recursive(
    node: &BvhNode,
    pos: Vec3,
    surface_evals: &[f64],
    cells: &[Cell],
) -> Option<usize> {
    match node {
        BvhNode::Leaf { cell_idx, aabb } => {
            if aabb.contains(pos) && cells[*cell_idx].contains(surface_evals) {
                Some(*cell_idx)
            } else {
                None
            }
        }
        BvhNode::Internal { aabb, left, right } => {
            if !aabb.contains(pos) {
                return None;
            }
            find_cell_recursive(left, pos, surface_evals, cells)
                .or_else(|| find_cell_recursive(right, pos, surface_evals, cells))
        }
    }
}

/// Recursively build the BVH using midpoint splitting.
fn build_recursive(entries: &mut [(usize, Aabb, Vec3)]) -> BvhNode {
    if entries.len() == 1 {
        return BvhNode::Leaf {
            cell_idx: entries[0].0,
            aabb: entries[0].1,
        };
    }

    // Compute overall AABB
    let overall_aabb = entries
        .iter()
        .map(|(_, aabb, _)| *aabb)
        .reduce(Aabb::union)
        .expect("non-empty");

    if entries.len() == 2 {
        return BvhNode::Internal {
            aabb: overall_aabb,
            left: Box::new(BvhNode::Leaf {
                cell_idx: entries[0].0,
                aabb: entries[0].1,
            }),
            right: Box::new(BvhNode::Leaf {
                cell_idx: entries[1].0,
                aabb: entries[1].1,
            }),
        };
    }

    // Find the axis with the largest extent
    let extent = overall_aabb.max - overall_aabb.min;
    let split_axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Sort by centroid along the split axis
    entries.sort_by(|a, b| {
        let ca = match split_axis {
            0 => a.2.x,
            1 => a.2.y,
            _ => a.2.z,
        };
        let cb = match split_axis {
            0 => b.2.x,
            1 => b.2.y,
            _ => b.2.z,
        };
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Split at the midpoint
    let mid = entries.len() / 2;
    let (left_entries, right_entries) = entries.split_at_mut(mid);

    let left = build_recursive(left_entries);
    let right = build_recursive(right_entries);

    BvhNode::Internal {
        aabb: overall_aabb,
        left: Box::new(left),
        right: Box::new(right),
    }
}
