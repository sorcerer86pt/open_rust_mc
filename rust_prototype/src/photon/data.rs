//! Photon-interaction data structures, element-indexed.
//!
//! An OpenMC photon HDF5 file (`photon/<Sym>.h5`) stores one element's
//! photoatomic cross sections and sampling auxiliaries on a shared energy
//! grid of ~10³ points spanning ~1 eV to ~100 GeV. The reactions are:
//!
//!   - Coherent (Rayleigh) scattering — elastic, no energy transfer
//!   - Incoherent (Compton) scattering — energy loss, electron ejection
//!   - Photoelectric absorption — photon destroyed, electron ejected with
//!     possible subsequent fluorescence / Auger relaxation
//!   - Pair production in the field of a nucleus (Z² scaling)
//!   - Pair production in the field of an atomic electron (triplet, Z scaling)
//!
//! Sampling auxiliaries stored alongside:
//!   - Coherent: atomic form factor F(x, Z) for the angular distribution
//!   - Incoherent: incoherent scattering function S(x, Z) for the Compton
//!     angular distribution (plus optional Compton profiles for bound-electron
//!     Doppler broadening; not loaded in Phase 1)
//!   - Photoelectric: per-subshell cross sections, binding energies, and
//!     relaxation (transitions) data for fluorescence emission

/// All photon-interaction data for one element, indexed by Z.
///
/// Energy grid is shared across all cross-section arrays.
pub struct PhotonElement {
    /// Atomic number.
    pub z: u32,
    /// Element symbol (e.g., "C").
    pub symbol: String,
    /// Shared photon energy grid in eV, ascending. Typical length ~1200 points.
    pub energy: Vec<f64>,

    /// Coherent (Rayleigh) cross section in barns/atom, same length as `energy`.
    pub coherent_xs: Vec<f64>,
    /// Incoherent (Compton) cross section in barns/atom.
    pub incoherent_xs: Vec<f64>,
    /// Total photoelectric cross section in barns/atom (sum of subshells).
    pub photoelectric_xs: Vec<f64>,
    /// Pair production in nuclear field in barns/atom.
    pub pair_production_nuclear_xs: Vec<f64>,
    /// Pair production in electron field (triplet) in barns/atom.
    pub pair_production_electron_xs: Vec<f64>,

    /// Coherent form factor F(x, Z) used for Rayleigh angular sampling.
    pub coherent_form_factor: ScatteringFactor,
    /// Incoherent scattering function S(x, Z) used for Compton angular
    /// sampling (deviation from free Klein-Nishina).
    pub incoherent_scattering_factor: ScatteringFactor,

    /// Per-subshell photoelectric data (binding energies, partial cross
    /// sections) and atomic relaxation transitions.
    pub subshells: Vec<Subshell>,
}

/// A tabulated scattering factor on a shared momentum-transfer grid.
///
/// Both coherent F(x, Z) and incoherent S(x, Z) are stored as shape
/// `(2, N)` in HDF5: row 0 is `x` (the momentum-transfer variable,
/// inverse Ångström), row 1 is the factor value. We keep them split
/// into two equal-length Vecs for clarity.
pub struct ScatteringFactor {
    /// Momentum-transfer variable x in inverse Ångström, ascending.
    pub x: Vec<f64>,
    /// Factor value at each `x`.
    pub value: Vec<f64>,
}

/// One atomic subshell: binding energy, occupancy, partial photoelectric
/// cross section, and relaxation transitions.
///
/// Subshell designator examples: "K", "L1", "L2", "L3", "M1", …
pub struct Subshell {
    /// Designator string from HDF5 (e.g., "K", "L3").
    pub designator: String,
    /// Binding energy in eV.
    pub binding_energy: f64,
    /// Number of electrons in this subshell (may be fractional for
    /// partially-filled shells).
    pub num_electrons: f64,
    /// Partial photoelectric cross section in barns/atom on the element's
    /// energy grid. Length may be less than `PhotonElement::energy.len()`
    /// if the cross section is stored only above the binding energy;
    /// alignment is against the tail of the shared grid.
    pub xs: Vec<f64>,
    /// Atomic relaxation transitions. Each row is
    /// `[primary_subshell, secondary_subshell, transition_energy_eV,
    /// transition_probability]`. The secondary subshell is 0 for
    /// radiative (fluorescence) transitions and non-zero for
    /// non-radiative (Auger / Coster-Kronig) transitions.
    pub transitions: Vec<[f64; 4]>,
}

impl PhotonElement {
    /// Number of energy grid points.
    pub fn n_energy(&self) -> usize {
        self.energy.len()
    }

    /// Total photon cross section in barns/atom at grid point `i`.
    ///
    /// Sum of coherent, incoherent, photoelectric, and both pair-production
    /// channels. Callers that discretise the reaction-selection step should
    /// use the per-channel arrays directly; this helper is for diagnostics.
    pub fn total_xs_at(&self, i: usize) -> f64 {
        self.coherent_xs[i]
            + self.incoherent_xs[i]
            + self.photoelectric_xs[i]
            + self.pair_production_nuclear_xs[i]
            + self.pair_production_electron_xs[i]
    }
}
