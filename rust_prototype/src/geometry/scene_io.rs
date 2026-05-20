// SPDX-License-Identifier: MIT
//! `scene.json` deserializer — loads NMC bundle geometry into a
//! runnable [`Geometry`].
//!
//! Maps the JSON schema documented in
//! `specs/nmc/open_rust_mc_geometry.schema.json` to the engine's
//! geometry types. The deserialization is staged through a parallel
//! set of "DTO" (data transfer object) types so the engine types stay
//! free of serde derives, AABBs, BVHs, and other internal accelerators
//! that the schema doesn't expose.
//!
//! Scope of this round:
//!   * Surfaces, cells, regions, universes — fully supported.
//!   * Rectangular lattices, hex lattices — fully supported.
//!   * Materials — schema is parsed but conversion to engine
//!     [`crate::transport::material::Material`] is deferred. Materials
//!     reference HDF5 cross-section files which must be resolved
//!     through `NuclideLibrary` + `xs_provider::load_nuclide` at a
//!     higher layer (see `transport::xs_provider`). This module returns
//!     the raw [`MaterialDto`] list so callers can resolve them.
//!   * Cell AABBs default to `Aabb::INFINITE`. Per-universe BVH falls
//!     back to linear scan when AABBs are infinite, so correctness is
//!     preserved at a small performance cost. Pre-computing AABBs from
//!     the region tree is a separate optimization.
//!
//! # Quick reference
//!
//! ```ignore
//! let scene = std::fs::read_to_string("godiva.scene.json")?;
//! let loaded = scene_io::load_scene_from_json(&scene)?;
//! let geom: Geometry = loaded.geometry;
//! let mats: Vec<MaterialDto> = loaded.materials;
//! ```

use std::collections::HashMap;

use serde::Deserialize;

use super::cell::{Cell, CellFill, CellId, Region};
use super::lattice::{HexLattice, MaterialOverrideMap, RectLattice};
use super::scene::{Geometry, GeometryError};
use super::surface::{BoundaryCondition, Surface, SurfaceId};
use super::universe::{Universe, UniverseId};
use super::{Aabb, Mat3, Vec3};

// ── DTO types — match the schema field-for-field ──────────────────────

/// Boundary-condition discriminant in scene.json.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum BoundaryConditionDto {
    Transmission,
    Reflective,
    Vacuum,
}

impl From<BoundaryConditionDto> for BoundaryCondition {
    fn from(b: BoundaryConditionDto) -> Self {
        match b {
            BoundaryConditionDto::Transmission => BoundaryCondition::Transmission,
            BoundaryConditionDto::Reflective => BoundaryCondition::Reflective,
            BoundaryConditionDto::Vacuum => BoundaryCondition::Vacuum,
        }
    }
}

/// Optional human-readable metadata that may appear on any scene
/// primitive (surface, cell, material, universe, lattice). Survives
/// the load into `SceneNames` so engine diagnostics, panic messages,
/// and IDE tooling can resolve numeric indices to user-supplied names
/// without forcing every engine struct to carry a String.
///
/// JSON convention:
/// * `name` — short identifier (1-3 words), e.g. `"Outer reflector"`.
/// * `comment` — longer free-form description.
/// * Both accept the underscored aliases `_name` / `_comment` so users
///   who prefer the JSON "non-semantic metadata" convention can pick
///   either spelling.
///
/// Both fields are `Option<String>` and default to `None`, so every
/// existing scene.json (which omits them) still parses unchanged.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct PrimitiveMetadata {
    #[serde(default, alias = "_name")]
    pub name: Option<String>,
    #[serde(default, alias = "_comment")]
    pub comment: Option<String>,
}

/// Surface entry with optional metadata. `meta` and `surface` are both
/// `#[serde(flatten)]` so the JSON looks identical to a bare
/// `SurfaceDto` plus optional `name` / `comment` fields at the same
/// level — no nesting, no schema break for existing scene.json files.
#[derive(Debug, Clone, Deserialize)]
pub struct NamedSurface {
    #[serde(flatten)]
    pub meta: PrimitiveMetadata,
    #[serde(flatten)]
    pub surface: SurfaceDto,
}

/// Tagged-enum surface variant. Field names mirror
/// `open_rust_mc_geometry.schema.json` §$defs/Surface.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum SurfaceDto {
    Plane {
        normal: [f64; 3],
        offset: f64,
        bc: BoundaryConditionDto,
    },
    PlaneX {
        x0: f64,
        bc: BoundaryConditionDto,
    },
    PlaneY {
        y0: f64,
        bc: BoundaryConditionDto,
    },
    PlaneZ {
        z0: f64,
        bc: BoundaryConditionDto,
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
        bc: BoundaryConditionDto,
    },
    CylinderZ {
        center_x: f64,
        center_y: f64,
        radius: f64,
        bc: BoundaryConditionDto,
    },
    CylinderX {
        center_y: f64,
        center_z: f64,
        radius: f64,
        bc: BoundaryConditionDto,
    },
    CylinderY {
        center_x: f64,
        center_z: f64,
        radius: f64,
        bc: BoundaryConditionDto,
    },
    ConeZ {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryConditionDto,
    },
    ConeX {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryConditionDto,
    },
    ConeY {
        x0: f64,
        y0: f64,
        z0: f64,
        r_sq: f64,
        bc: BoundaryConditionDto,
    },
}

impl From<SurfaceDto> for Surface {
    fn from(s: SurfaceDto) -> Surface {
        match s {
            SurfaceDto::Plane { normal, offset, bc } => Surface::Plane {
                normal: Vec3::new(normal[0], normal[1], normal[2]),
                offset,
                bc: bc.into(),
            },
            SurfaceDto::PlaneX { x0, bc } => Surface::PlaneX { x0, bc: bc.into() },
            SurfaceDto::PlaneY { y0, bc } => Surface::PlaneY { y0, bc: bc.into() },
            SurfaceDto::PlaneZ { z0, bc } => Surface::PlaneZ { z0, bc: bc.into() },
            SurfaceDto::Sphere { center, radius, bc } => Surface::Sphere {
                center: Vec3::new(center[0], center[1], center[2]),
                radius,
                bc: bc.into(),
            },
            SurfaceDto::CylinderZ {
                center_x,
                center_y,
                radius,
                bc,
            } => Surface::CylinderZ {
                center_x,
                center_y,
                radius,
                bc: bc.into(),
            },
            SurfaceDto::CylinderX {
                center_y,
                center_z,
                radius,
                bc,
            } => Surface::CylinderX {
                center_y,
                center_z,
                radius,
                bc: bc.into(),
            },
            SurfaceDto::CylinderY {
                center_x,
                center_z,
                radius,
                bc,
            } => Surface::CylinderY {
                center_x,
                center_z,
                radius,
                bc: bc.into(),
            },
            SurfaceDto::ConeZ {
                x0,
                y0,
                z0,
                r_sq,
                bc,
            } => Surface::ConeZ {
                x0,
                y0,
                z0,
                r_sq,
                bc: bc.into(),
            },
            SurfaceDto::ConeX {
                x0,
                y0,
                z0,
                r_sq,
                bc,
            } => Surface::ConeX {
                x0,
                y0,
                z0,
                r_sq,
                bc: bc.into(),
            },
            SurfaceDto::ConeY {
                x0,
                y0,
                z0,
                r_sq,
                bc,
            } => Surface::ConeY {
                x0,
                y0,
                z0,
                r_sq,
                bc: bc.into(),
            },
        }
    }
}

/// CSG region tree. Tagged-enum variants match schema §$defs/Region.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op")]
pub enum RegionDto {
    HalfSpace {
        surface_idx: usize,
        positive: bool,
    },
    Intersection {
        left: Box<RegionDto>,
        right: Box<RegionDto>,
    },
    Union {
        left: Box<RegionDto>,
        right: Box<RegionDto>,
    },
    Complement {
        inner: Box<RegionDto>,
    },
}

impl From<RegionDto> for Region {
    fn from(r: RegionDto) -> Region {
        match r {
            RegionDto::HalfSpace {
                surface_idx,
                positive,
            } => Region::HalfSpace {
                surface_idx,
                positive,
            },
            RegionDto::Intersection { left, right } => {
                Region::Intersection(Box::new((*left).into()), Box::new((*right).into()))
            }
            RegionDto::Union { left, right } => {
                Region::Union(Box::new((*left).into()), Box::new((*right).into()))
            }
            RegionDto::Complement { inner } => Region::Complement(Box::new((*inner).into())),
        }
    }
}

/// Cell-fill discriminant. Schema §$defs/CellFill.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type")]
pub enum CellFillDto {
    Material { material_idx: u32 },
    Universe { universe_id: u32 },
    Lattice { lattice_idx: u32 },
    HexLattice { hex_lattice_idx: u32 },
    Void,
}

impl From<CellFillDto> for CellFill {
    fn from(c: CellFillDto) -> CellFill {
        match c {
            CellFillDto::Material { material_idx } => CellFill::Material(material_idx),
            CellFillDto::Universe { universe_id } => CellFill::Universe(universe_id),
            CellFillDto::Lattice { lattice_idx } => CellFill::Lattice(lattice_idx),
            CellFillDto::HexLattice { hex_lattice_idx } => CellFill::HexLattice(hex_lattice_idx),
            CellFillDto::Void => CellFill::Void,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CellDto {
    pub id: u32,
    pub region: RegionDto,
    pub fill: CellFillDto,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub rotation: Option<[[f64; 3]; 3]>,
}

/// Cell with optional metadata. See [`PrimitiveMetadata`].
#[derive(Debug, Clone, Deserialize)]
pub struct NamedCell {
    #[serde(flatten)]
    pub meta: PrimitiveMetadata,
    #[serde(flatten)]
    pub cell: CellDto,
}

fn default_temperature() -> f64 {
    293.6
}

impl CellDto {
    fn into_cell(self) -> Cell {
        let rotation = self.rotation.map(|m| Mat3 {
            rows: [
                Vec3::new(m[0][0], m[0][1], m[0][2]),
                Vec3::new(m[1][0], m[1][1], m[1][2]),
                Vec3::new(m[2][0], m[2][1], m[2][2]),
            ],
        });
        let mut cell = Cell::new(CellId(self.id), self.region.into(), self.fill.into())
            .with_temperature(self.temperature);
        // AABB defaults to infinite — keeps construction simple, drops
        // BVH optimization until a later pass.
        cell.aabb = Aabb::INFINITE;
        cell.rotation = rotation;
        cell
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct UniverseDto {
    pub id: u32,
    pub cell_indices: Vec<usize>,
}

/// Universe with optional metadata. See [`PrimitiveMetadata`].
#[derive(Debug, Clone, Deserialize)]
pub struct NamedUniverse {
    #[serde(flatten)]
    pub meta: PrimitiveMetadata,
    #[serde(flatten)]
    pub universe: UniverseDto,
}

impl From<UniverseDto> for Universe {
    fn from(u: UniverseDto) -> Universe {
        Universe::new(UniverseId(u.id), u.cell_indices)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RectLatticeDto {
    pub origin: [f64; 3],
    pub pitch: [f64; 3],
    pub shape: [usize; 3],
    pub universes: Vec<u32>,
    #[serde(default)]
    pub material_overrides: Option<Vec<HashMap<String, u32>>>,
}

/// Rect lattice with optional metadata. See [`PrimitiveMetadata`].
#[derive(Debug, Clone, Deserialize)]
pub struct NamedRectLattice {
    #[serde(flatten)]
    pub meta: PrimitiveMetadata,
    #[serde(flatten)]
    pub lattice: RectLatticeDto,
}

impl From<RectLatticeDto> for RectLattice {
    fn from(l: RectLatticeDto) -> RectLattice {
        let universes = l.universes.into_iter().map(UniverseId).collect();
        // String keys → usize for the engine's MaterialOverrideMap.
        let material_overrides = l.material_overrides.map(|maps| {
            maps.into_iter()
                .map(|m| {
                    m.into_iter()
                        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|kk| (kk, v)))
                        .collect::<MaterialOverrideMap>()
                })
                .collect::<Vec<_>>()
        });
        RectLattice {
            origin: Vec3::new(l.origin[0], l.origin[1], l.origin[2]),
            pitch: Vec3::new(l.pitch[0], l.pitch[1], l.pitch[2]),
            shape: l.shape,
            universes,
            material_overrides,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HexLatticeDto {
    pub center: [f64; 3],
    pub pitch_xy: f64,
    pub pitch_z: f64,
    pub n_rings: usize,
    pub n_axial: usize,
    pub orientation: String,
    pub universes: Vec<u32>,
    #[serde(default)]
    pub material_overrides: Option<Vec<HashMap<String, u32>>>,
}

/// Hex lattice with optional metadata. See [`PrimitiveMetadata`].
#[derive(Debug, Clone, Deserialize)]
pub struct NamedHexLattice {
    #[serde(flatten)]
    pub meta: PrimitiveMetadata,
    #[serde(flatten)]
    pub lattice: HexLatticeDto,
}

impl HexLatticeDto {
    fn into_hex_lattice(self) -> Result<HexLattice, SceneLoadError> {
        let orientation = match self.orientation.as_str() {
            "Y" => super::lattice::HexOrientation::Y,
            "X" => super::lattice::HexOrientation::X,
            other => return Err(SceneLoadError::UnknownHexOrientation(other.to_string())),
        };
        let universes = self.universes.into_iter().map(UniverseId).collect();
        let material_overrides = self.material_overrides.map(|maps| {
            maps.into_iter()
                .map(|m| {
                    m.into_iter()
                        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|kk| (kk, v)))
                        .collect::<MaterialOverrideMap>()
                })
                .collect::<Vec<_>>()
        });
        Ok(HexLattice {
            center: Vec3::new(self.center[0], self.center[1], self.center[2]),
            pitch_xy: self.pitch_xy,
            pitch_z: self.pitch_z,
            n_rings: self.n_rings,
            n_axial: self.n_axial,
            orientation,
            universes,
            material_overrides,
        })
    }
}

/// One nuclide entry inside a material. Carries the HDF5 file path
/// rather than a resolved kernel index, since kernel loading happens
/// at a higher layer (`transport::xs_provider`).
#[derive(Debug, Clone, Deserialize)]
pub struct NuclideEntryDto {
    /// Path to the OpenMC-format HDF5 file. Schema requires it; some
    /// imported cases use a `zaid` + `label` form instead.
    #[serde(default)]
    pub hdf5_file: Option<String>,
    /// ZAID (1000·Z + A). Optional in the schema; present in
    /// import-script outputs.
    #[serde(default)]
    pub zaid: Option<u32>,
    /// Human-readable label, e.g. `"U-235"`.
    #[serde(default)]
    pub label: Option<String>,
    /// Atom density in atoms / (barn · cm).
    pub atom_density: f64,
    /// Path to an S(α,β) HDF5 file (e.g. `c_H_in_H2O.h5`). `None` for
    /// free-gas treatment.
    #[serde(default)]
    pub thermal_file: Option<String>,
}

/// Material as it appears in scene.json. Engine
/// [`crate::transport::material::Material`] is built separately by a
/// higher layer that resolves HDF5 files to xs_kernel indices.
#[derive(Debug, Clone, Deserialize)]
pub struct MaterialDto {
    pub name: String,
    /// Optional free-form description, surfaced through `SceneNames`
    /// for diagnostics / IDE tooling. Accepts `_comment` as an alias
    /// matching the user-side metadata convention.
    #[serde(default, alias = "_comment")]
    pub comment: Option<String>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    pub nuclides: Vec<NuclideEntryDto>,
    /// Schema's per-nuclide `thermal_file` is the canonical form, but
    /// some imported cases collect S(α,β) into a material-level list.
    /// Either form is accepted on the way in.
    #[serde(default)]
    pub thermal_files: Vec<String>,
}

/// Top-level scene.json structure.
///
/// Every primitive list (`surfaces`, `cells`, `universes`,
/// `rect_lattices`, `hex_lattices`) accepts optional `name` and
/// `comment` fields on each entry via the `Named*` wrappers. Existing
/// scene.json files without those fields still parse — the wrapper is
/// transparent under `#[serde(flatten)]`.
#[derive(Debug, Clone, Deserialize)]
pub struct SceneDto {
    pub surfaces: Vec<NamedSurface>,
    pub cells: Vec<NamedCell>,
    pub universes: Vec<NamedUniverse>,
    #[serde(default)]
    pub rect_lattices: Vec<NamedRectLattice>,
    #[serde(default)]
    pub hex_lattices: Vec<NamedHexLattice>,
    #[serde(default)]
    pub materials: Vec<MaterialDto>,
    pub root_universe_id: u32,
}

// ── Errors ────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SceneLoadError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("geometry validation: {0}")]
    Geometry(#[from] GeometryError),
    #[error("unknown hex orientation: {0} (expected 'X' or 'Y')")]
    UnknownHexOrientation(String),
}

// ── Public API ────────────────────────────────────────────────────────

/// Parallel-array lookup of human-readable names for every primitive
/// in the scene. Index space matches `Geometry.surfaces /
/// cells / universes / rect_lattices / hex_lattices` and
/// `LoadedScene.materials`. `None` for any entry that omitted the
/// field. Engine diagnostics, panic messages, and IDE tooling can
/// consult this without forcing every engine struct to carry a String.
#[derive(Debug, Default, Clone)]
pub struct SceneNames {
    pub surfaces: Vec<PrimitiveMetadata>,
    pub cells: Vec<PrimitiveMetadata>,
    pub universes: Vec<PrimitiveMetadata>,
    pub rect_lattices: Vec<PrimitiveMetadata>,
    pub hex_lattices: Vec<PrimitiveMetadata>,
    pub materials: Vec<PrimitiveMetadata>,
}

impl SceneNames {
    /// Look up a surface's name by index, falling back to `surface[i]`
    /// when no name was supplied. Convenient for error / log messages.
    pub fn surface_label(&self, idx: usize) -> String {
        Self::label("surface", &self.surfaces, idx)
    }
    pub fn cell_label(&self, idx: usize) -> String {
        Self::label("cell", &self.cells, idx)
    }
    pub fn universe_label(&self, idx: usize) -> String {
        Self::label("universe", &self.universes, idx)
    }
    pub fn rect_lattice_label(&self, idx: usize) -> String {
        Self::label("rect_lattice", &self.rect_lattices, idx)
    }
    pub fn hex_lattice_label(&self, idx: usize) -> String {
        Self::label("hex_lattice", &self.hex_lattices, idx)
    }
    pub fn material_label(&self, idx: usize) -> String {
        Self::label("material", &self.materials, idx)
    }

    fn label(kind: &str, table: &[PrimitiveMetadata], idx: usize) -> String {
        table
            .get(idx)
            .and_then(|m| m.name.as_deref())
            .map(|n| format!("{kind}[{idx}] \"{n}\""))
            .unwrap_or_else(|| format!("{kind}[{idx}]"))
    }
}

/// Result of loading a scene.json file. `geometry` is fully built and
/// validated; `materials` carries the raw schema entries so a caller
/// can resolve their HDF5 files into engine `Material`s when needed;
/// `names` carries the optional `name` / `comment` metadata for every
/// primitive in index-aligned form (see [`SceneNames`]).
#[derive(Debug)]
pub struct LoadedScene {
    pub geometry: Geometry,
    pub materials: Vec<MaterialDto>,
    pub names: SceneNames,
}

/// Parse `scene.json` (or the `scene` block of an NMC bundle) into a
/// runnable [`Geometry`] plus the raw material list. Validates
/// surface/cell/universe/lattice index references at construction.
pub fn load_scene_from_json(json: &str) -> Result<LoadedScene, SceneLoadError> {
    let dto: SceneDto = serde_json::from_str(json)?;
    load_scene_from_dto(dto)
}

/// Build a [`Geometry`] + materials from an already-parsed
/// [`SceneDto`]. Useful when the JSON is embedded in a larger
/// document (e.g. an `.nmc` bundle's `scene` field).
pub fn load_scene_from_dto(dto: SceneDto) -> Result<LoadedScene, SceneLoadError> {
    // Split each Named<Primitive> into its metadata and the inner DTO.
    // The order in each Vec must stay in lock-step with the engine
    // index space (Geometry uses positional indices to reference
    // surfaces, cells, etc.) so that `SceneNames::*_label(idx)`
    // resolves correctly.
    let (surface_names, surfaces): (Vec<_>, Vec<Surface>) = dto
        .surfaces
        .into_iter()
        .map(|ns| (ns.meta, Surface::from(ns.surface)))
        .unzip();
    let (cell_names, cells): (Vec<_>, Vec<Cell>) = dto
        .cells
        .into_iter()
        .map(|nc| (nc.meta, nc.cell.into_cell()))
        .unzip();
    let (universe_names, universes): (Vec<_>, Vec<Universe>) = dto
        .universes
        .into_iter()
        .map(|nu| (nu.meta, Universe::from(nu.universe)))
        .unzip();
    let (rect_lat_names, rect_lattices): (Vec<_>, Vec<RectLattice>) = dto
        .rect_lattices
        .into_iter()
        .map(|nl| (nl.meta, RectLattice::from(nl.lattice)))
        .unzip();
    let mut hex_lat_names: Vec<PrimitiveMetadata> = Vec::with_capacity(dto.hex_lattices.len());
    let mut hex_lattices: Vec<HexLattice> = Vec::with_capacity(dto.hex_lattices.len());
    for nl in dto.hex_lattices {
        hex_lat_names.push(nl.meta);
        hex_lattices.push(nl.lattice.into_hex_lattice()?);
    }
    let material_names: Vec<PrimitiveMetadata> = dto
        .materials
        .iter()
        .map(|m| PrimitiveMetadata {
            name: Some(m.name.clone()),
            comment: m.comment.clone(),
        })
        .collect();

    let geometry = Geometry::new(
        surfaces,
        cells,
        universes,
        rect_lattices,
        UniverseId(dto.root_universe_id),
    )?
    .with_hex_lattices(hex_lattices)?;
    Ok(LoadedScene {
        geometry,
        materials: dto.materials,
        names: SceneNames {
            surfaces: surface_names,
            cells: cell_names,
            universes: universe_names,
            rect_lattices: rect_lat_names,
            hex_lattices: hex_lat_names,
            materials: material_names,
        },
    })
}

/// Convenience overload — preserved for old callers that already
/// stripped the materials list. Returns only the [`Geometry`].
pub fn geometry_from_json(json: &str) -> Result<Geometry, SceneLoadError> {
    Ok(load_scene_from_json(json)?.geometry)
}

/// Errors from [`load_scene_from_path`].
#[derive(Debug, thiserror::Error)]
pub enum ScenePathError {
    #[error("read {0}: {1}")]
    Io(std::path::PathBuf, std::io::Error),
    #[error(transparent)]
    Load(#[from] SceneLoadError),
}

/// Engine-side convenience: parse scene.json from a file path.
pub fn load_scene_from_path(path: &std::path::Path) -> Result<LoadedScene, ScenePathError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ScenePathError::Io(path.to_path_buf(), e))?;
    Ok(load_scene_from_json(&text)?)
}

// ── Surface-id alias (matches schema's first-position convention) ─────

impl SurfaceDto {
    /// Surface IDs in the schema are implicit — index into the
    /// `surfaces` array. This helper exists so external callers
    /// constructing DTOs by hand can tag a surface for forward-compat.
    pub fn _stable_id(idx: usize) -> SurfaceId {
        SurfaceId(idx as u32)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Optional `name` / `comment` metadata on surfaces, cells, and
    /// materials round-trips through the loader into `SceneNames` with
    /// the expected index alignment. Also verifies that the underscore
    /// alias (`_name` / `_comment`) works alongside the plain spelling.
    /// Scenes that omit the metadata fields (every existing scene.json)
    /// still parse — covered implicitly by the unchanged
    /// `load_godiva_bare_sphere` test below.
    #[test]
    fn primitive_metadata_round_trips() {
        let json = r#"{
            "surfaces": [
                {
                    "type": "Sphere",
                    "center": [0, 0, 0],
                    "radius": 8.7407,
                    "bc": "Vacuum",
                    "name": "Outer reflector boundary",
                    "_comment": "ICSBEP HMF-001 case-1 outer radius"
                }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": false },
                    "fill":   { "type": "Material", "material_idx": 0 },
                    "_name": "HEU core"
                },
                {
                    "id": 1,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": true },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [
                { "id": 0, "cell_indices": [0, 1], "name": "root" }
            ],
            "materials": [
                {
                    "name": "HEU",
                    "comment": "93.7 % U-235 / 5.3 % U-238 / 1.0 % U-234",
                    "temperature": 293.6,
                    "nuclides": [
                        { "hdf5_file": "U235.h5", "atom_density": 0.045 }
                    ]
                }
            ],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        // Indices line up with Geometry's positional ordering.
        assert_eq!(loaded.names.surfaces.len(), 1);
        assert_eq!(loaded.names.cells.len(), 2);
        assert_eq!(loaded.names.universes.len(), 1);
        assert_eq!(loaded.names.materials.len(), 1);
        // `name` parses both ways: plain (surfaces[0]) and underscored
        // alias (cells[0] uses `_name`).
        assert_eq!(
            loaded.names.surfaces[0].name.as_deref(),
            Some("Outer reflector boundary")
        );
        assert_eq!(loaded.names.cells[0].name.as_deref(), Some("HEU core"));
        // Cell[1] omits metadata — `None` survives the round-trip.
        assert_eq!(loaded.names.cells[1].name, None);
        assert_eq!(loaded.names.cells[1].comment, None);
        // `_comment` alias works on the surface.
        assert_eq!(
            loaded.names.surfaces[0].comment.as_deref(),
            Some("ICSBEP HMF-001 case-1 outer radius")
        );
        // Universe name parses.
        assert_eq!(loaded.names.universes[0].name.as_deref(), Some("root"));
        // Material name + comment surface through SceneNames.
        assert_eq!(loaded.names.materials[0].name.as_deref(), Some("HEU"));
        assert!(
            loaded.names.materials[0]
                .comment
                .as_deref()
                .unwrap()
                .starts_with("93.7")
        );
        // Helpers render the labelled form when a name exists, fall
        // back to the bare index otherwise.
        assert_eq!(
            loaded.names.surface_label(0),
            "surface[0] \"Outer reflector boundary\""
        );
        assert_eq!(loaded.names.cell_label(1), "cell[1]");
    }

    /// Bare Godiva sphere — single sphere, single cell, single
    /// universe. Smallest possible scene. Verifies every surface
    /// variant tag works.
    #[test]
    fn load_godiva_bare_sphere() {
        let json = r#"{
            "surfaces": [
                { "type": "Sphere", "center": [0, 0, 0], "radius": 8.7407, "bc": "Vacuum" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": false },
                    "fill":   { "type": "Material", "material_idx": 0 },
                    "temperature": 293.6
                },
                {
                    "id": 1,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": true },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [
                { "id": 0, "cell_indices": [0, 1] }
            ],
            "materials": [
                {
                    "name": "HEU",
                    "temperature": 293.6,
                    "nuclides": [
                        { "hdf5_file": "U235.h5", "atom_density": 0.045 },
                        { "hdf5_file": "U238.h5", "atom_density": 0.0025 }
                    ]
                }
            ],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        assert_eq!(loaded.geometry.surfaces.len(), 1);
        assert_eq!(loaded.geometry.cells.len(), 2);
        assert_eq!(loaded.geometry.universes.len(), 1);
        assert!(matches!(
            loaded.geometry.surfaces[0],
            Surface::Sphere { radius, .. } if (radius - 8.7407).abs() < 1e-9
        ));
        assert_eq!(loaded.materials.len(), 1);
        assert_eq!(loaded.materials[0].nuclides.len(), 2);
    }

    /// PMF-001 (Jezebel) — single Pu sphere, validates ZAID-form
    /// materials (no hdf5_file, just zaid + label as produced by the
    /// import script).
    #[test]
    fn load_jezebel_with_zaid_form_materials() {
        let json = r#"{
            "surfaces": [
                { "type": "Sphere", "center": [0,0,0], "radius": 6.3849, "bc": "Vacuum" }
            ],
            "cells": [
                {
                    "id": 1,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": false },
                    "fill":   { "type": "Material", "material_idx": 0 },
                    "temperature": 293.6
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "materials": [
                {
                    "name": "delta_Pu",
                    "temperature": 293.6,
                    "nuclides": [
                        { "zaid": 94239, "label": "Pu-239", "atom_density": 0.037047 },
                        { "zaid": 94240, "label": "Pu-240", "atom_density": 0.0017512 },
                        { "zaid": 94241, "label": "Pu-241", "atom_density": 0.00011674 },
                        { "zaid": 31069, "label": "Ga-69",  "atom_density": 8.266e-4 },
                        { "zaid": 31071, "label": "Ga-71",  "atom_density": 5.486e-4 }
                    ]
                }
            ],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        assert_eq!(loaded.geometry.surfaces.len(), 1);
        let mat = &loaded.materials[0];
        assert_eq!(mat.nuclides.len(), 5);
        assert_eq!(mat.nuclides[0].zaid, Some(94239));
    }

    /// Verifies every Surface variant tag — `Plane`, `PlaneX/Y/Z`,
    /// `Sphere`, `CylinderX/Y/Z`, `ConeX/Y/Z`. Catches discriminator
    /// drift if any variant name diverges between the engine and the
    /// schema.
    #[test]
    fn every_surface_variant_round_trips() {
        let json = r#"{
            "surfaces": [
                { "type": "Plane",     "normal": [1,0,0], "offset": 1.0, "bc": "Vacuum" },
                { "type": "PlaneX",    "x0": -1.0,                          "bc": "Reflective" },
                { "type": "PlaneY",    "y0": -1.0,                          "bc": "Reflective" },
                { "type": "PlaneZ",    "z0": -1.0,                          "bc": "Reflective" },
                { "type": "Sphere",    "center": [0,0,0], "radius": 5.0,    "bc": "Vacuum" },
                { "type": "CylinderZ", "center_x": 0, "center_y": 0, "radius": 0.5, "bc": "Transmission" },
                { "type": "CylinderX", "center_y": 0, "center_z": 0, "radius": 0.5, "bc": "Transmission" },
                { "type": "CylinderY", "center_x": 0, "center_z": 0, "radius": 0.5, "bc": "Transmission" },
                { "type": "ConeZ",     "x0": 0, "y0": 0, "z0": 0, "r_sq": 0.25, "bc": "Vacuum" },
                { "type": "ConeX",     "x0": 0, "y0": 0, "z0": 0, "r_sq": 0.25, "bc": "Vacuum" },
                { "type": "ConeY",     "x0": 0, "y0": 0, "z0": 0, "r_sq": 0.25, "bc": "Vacuum" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 4, "positive": false },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "materials": [],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        let s = &loaded.geometry.surfaces;
        assert_eq!(s.len(), 11);
        assert!(matches!(s[0], Surface::Plane { .. }));
        assert!(matches!(s[1], Surface::PlaneX { .. }));
        assert!(matches!(s[2], Surface::PlaneY { .. }));
        assert!(matches!(s[3], Surface::PlaneZ { .. }));
        assert!(matches!(s[4], Surface::Sphere { .. }));
        assert!(matches!(s[5], Surface::CylinderZ { .. }));
        assert!(matches!(s[6], Surface::CylinderX { .. }));
        assert!(matches!(s[7], Surface::CylinderY { .. }));
        assert!(matches!(s[8], Surface::ConeZ { .. }));
        assert!(matches!(s[9], Surface::ConeX { .. }));
        assert!(matches!(s[10], Surface::ConeY { .. }));
    }

    /// Region tree with all four ops (HalfSpace, Intersection, Union,
    /// Complement). Catches mis-tagged or recursive-deserialization
    /// drift.
    #[test]
    fn nested_region_tree_round_trips() {
        let json = r#"{
            "surfaces": [
                { "type": "PlaneX", "x0": -1, "bc": "Reflective" },
                { "type": "PlaneX", "x0":  1, "bc": "Reflective" },
                { "type": "PlaneY", "y0": -1, "bc": "Reflective" },
                { "type": "PlaneY", "y0":  1, "bc": "Reflective" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": {
                        "op": "Intersection",
                        "left":  {
                            "op": "Union",
                            "left":  { "op": "HalfSpace", "surface_idx": 0, "positive": true },
                            "right": { "op": "HalfSpace", "surface_idx": 1, "positive": false }
                        },
                        "right": {
                            "op": "Complement",
                            "inner": { "op": "HalfSpace", "surface_idx": 2, "positive": false }
                        }
                    },
                    "fill": { "type": "Void" }
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "materials": [],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        let region = &loaded.geometry.cells[0].region;
        assert!(matches!(region, Region::Intersection(..)));
    }

    /// Out-of-range surface index in a region triggers the geometry
    /// validator (not just a serde error).
    #[test]
    fn out_of_range_surface_idx_caught_by_validator() {
        let json = r#"{
            "surfaces": [
                { "type": "Sphere", "center": [0,0,0], "radius": 1.0, "bc": "Vacuum" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 5, "positive": true },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "materials": [],
            "root_universe_id": 0
        }"#;
        let err = load_scene_from_json(json).unwrap_err();
        assert!(
            matches!(err, SceneLoadError::Geometry(GeometryError::SurfaceIndexOutOfRange { .. })),
            "expected SurfaceIndexOutOfRange, got: {err:?}",
        );
    }

    /// Out-of-range root universe is reported with a clear error.
    #[test]
    fn out_of_range_root_universe_caught() {
        let json = r#"{
            "surfaces": [
                { "type": "Sphere", "center": [0,0,0], "radius": 1.0, "bc": "Vacuum" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": false },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "materials": [],
            "root_universe_id": 7
        }"#;
        let err = load_scene_from_json(json).unwrap_err();
        assert!(
            matches!(err, SceneLoadError::Geometry(GeometryError::RootUniverseOutOfRange(..))),
            "expected RootUniverseOutOfRange, got: {err:?}",
        );
    }

    /// Rect lattice with a 1×1×1 grid round-trips. Validates the
    /// `material_overrides: HashMap<String, u32>` → engine
    /// `HashMap<usize, u32>` conversion.
    #[test]
    fn rect_lattice_with_material_overrides() {
        let json = r#"{
            "surfaces": [
                { "type": "PlaneX", "x0": -1, "bc": "Reflective" },
                { "type": "PlaneX", "x0":  1, "bc": "Reflective" }
            ],
            "cells": [
                {
                    "id": 0,
                    "region": { "op": "HalfSpace", "surface_idx": 0, "positive": true },
                    "fill":   { "type": "Void" }
                }
            ],
            "universes": [{ "id": 0, "cell_indices": [0] }],
            "rect_lattices": [
                {
                    "origin":     [-1, -1, -1],
                    "pitch":      [ 2,  2,  2],
                    "shape":      [ 1,  1,  1],
                    "universes":  [0],
                    "material_overrides": [{ "0": 3 }]
                }
            ],
            "materials": [],
            "root_universe_id": 0
        }"#;
        let loaded = load_scene_from_json(json).unwrap();
        let lat = &loaded.geometry.lattices[0];
        let overrides = lat.material_overrides.as_ref().expect("overrides present");
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].get(&0_usize), Some(&3_u32));
    }
}
