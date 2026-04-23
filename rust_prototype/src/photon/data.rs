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
//!   - Coherent: atomic form factor `F(x, Z)`, cumulative form-factor-squared
//!     `∫₀^{x²} F² dx'²` for direct sampling, and anomalous scattering factors
//!     `f'(E) + i f''(E)` for low-energy amplitude corrections.
//!   - Incoherent: incoherent scattering function `S(x, Z)` for bound-electron
//!     rejection, plus shell-resolved Hartree-Fock Compton profiles
//!     `Jᵢ(p_z)` for Doppler broadening of the outgoing photon energy.
//!   - Photoelectric: per-subshell cross sections, binding energies, and
//!     EADL-based atomic relaxation transitions.
//!   - Bremsstrahlung: Seltzer-Berger differential cross section `dσ/dk` on
//!     a (T_e, k) grid, used for thick-target bremsstrahlung from
//!     Compton/photoelectron/pair-production secondary electrons.

/// All photon-interaction data for one element, indexed by Z.
///
/// Energy grid is shared across all per-channel cross sections.
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

    /// Coherent form factor `F(x, Z)` used for Rayleigh angular sampling.
    pub coherent_form_factor: ScatteringFactor,
    /// Cumulative integral `∫₀^{x²} F²(x', Z) dx'²` used for direct `x²`
    /// sampling in Rayleigh scattering (PENELOPE §2.1.2 method).
    pub coherent_integrated_form_factor: ScatteringFactor,
    /// Anomalous scattering factors `f'(E)` and `f''(E)` applied to the
    /// coherent amplitude at low energies. Each on an independent energy
    /// grid in eV.
    pub coherent_anomalous: AnomalousFactors,

    /// Incoherent scattering function `S(x, Z)` for bound-electron Compton
    /// rejection.
    pub incoherent_scattering_factor: ScatteringFactor,
    /// Shell-resolved Hartree-Fock Compton profiles for Doppler broadening.
    pub compton_profiles: ComptonProfiles,

    /// Per-subshell photoelectric data: binding energies, partial cross
    /// sections, and EADL atomic relaxation transitions.
    pub subshells: Vec<Subshell>,

    /// Seltzer-Berger bremsstrahlung data and Sternheimer-Berger oscillator
    /// parameters for secondary-electron TTB / stopping-power calculations.
    pub bremsstrahlung: Bremsstrahlung,
}

/// A tabulated factor `y(x)` on a shared independent-variable grid.
///
/// Used for both `F(x, Z)` and `S(x, Z)`, which are stored as shape
/// `(2, N)` in HDF5 with row 0 the independent variable and row 1 the
/// factor value. We keep them split into two equal-length vectors.
pub struct ScatteringFactor {
    /// Momentum-transfer variable `x` in inverse Ångström, ascending.
    pub x: Vec<f64>,
    /// Factor value at each `x`.
    pub value: Vec<f64>,
}

/// Generic 2×N HDF5 dataset unpacked into two vectors of equal length.
///
/// Used for anomalous scattering and any other (grid, value) pair we load
/// without a physically-meaningful x-axis alias.
pub struct TabulatedFactor {
    /// Independent variable grid, ascending.
    pub grid: Vec<f64>,
    /// Dependent variable at each grid point.
    pub value: Vec<f64>,
}

/// Real and imaginary parts of the anomalous coherent scattering factor.
///
/// Each part is independently tabulated in energy; their grids are
/// generally different sizes. At photon energy `E`, the coherent
/// differential cross section is modulated by
/// `|F(x, Z) + f'(E) + i f''(E)|²`. For photon energies above ~100 keV
/// the anomalous correction is negligible; we load it for completeness
/// and for x-ray-regime users.
pub struct AnomalousFactors {
    /// `f'(E)` in electrons per atom on its own energy grid in eV.
    pub real: TabulatedFactor,
    /// `f''(E)` in electrons per atom on its own energy grid in eV.
    pub imag: TabulatedFactor,
}

/// Shell-resolved Hartree-Fock Compton profiles.
///
/// The outgoing photon energy in Compton scattering from a bound electron
/// deviates from the free-electron Klein-Nishina value by
/// `p_z / (m_e c)`, the projection of the electron's pre-collision
/// momentum on the scattering axis. `Jᵢ(p_z)` is the probability density
/// of that projection for an electron in shell `i`, tabulated on the
/// shared `pz` grid in atomic units (1 a.u. = α m_e c ≈ 1/137 m_e c).
///
/// At sampling time we select a "Compton shell" (whose subshell
/// partitioning may not coincide with the photoelectric subshells) from
/// `(binding_energy, num_electrons)`, then sample `p_z` from
/// `Jᵢ(p_z) · 𝟙[|p_z| < p_z_max(i, E, θ)]`.
pub struct ComptonProfiles {
    /// Binding energy of each Compton shell in eV.
    pub binding_energy: Vec<f64>,
    /// Number of electrons in each Compton shell (may be fractional).
    pub num_electrons: Vec<f64>,
    /// Shared momentum grid `p_z` in atomic units, ascending from 0.
    pub pz: Vec<f64>,
    /// Profiles: `j[i][k]` is `Jᵢ(pz[k])` in inverse atomic units.
    pub j: Vec<Vec<f64>>,
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
    /// Partial photoelectric cross section in barns/atom. Length may be
    /// less than `PhotonElement::energy.len()` because the shell cannot
    /// absorb below its binding energy; alignment is against the *tail*
    /// of the master grid: `xs[j]` corresponds to
    /// `PhotonElement::energy[N_E - xs.len() + j]` (OpenMC convention).
    pub xs: Vec<f64>,
    /// EADL atomic relaxation transitions. Each row is
    /// `[primary_subshell, secondary_subshell, transition_energy_eV,
    /// transition_probability]`. `secondary = 0` flags a radiative
    /// (fluorescence) transition; non-zero flags Auger / Coster-Kronig.
    pub transitions: Vec<[f64; 4]>,
}

/// Seltzer-Berger bremsstrahlung differential cross section and
/// Sternheimer-Berger mean-excitation-energy oscillator parameters.
///
/// The DCS `dσ/dk` is tabulated on an (electron kinetic energy,
/// scaled photon energy) grid. `dcs[i_e * n_k + i_k]` is the scaled
/// differential at `(electron_energy[i_e], photon_energy[i_k])`;
/// conventions and units match the OpenMC/ENDF representation
/// (scaled by `k / Z²` into `mbarn · MeV / (MeV · electron)`).
///
/// The oscillator strengths (`ionization_energy`, `num_electrons`) define
/// the Sternheimer density-effect correction to the electron
/// collisional stopping power via Berger-Seltzer parametrisation.
/// `mean_excitation_energy` is the atomic I-value in eV.
pub struct Bremsstrahlung {
    /// Mean excitation energy (I-value) in eV from the HDF5 attribute.
    pub mean_excitation_energy: f64,
    /// Electron kinetic energy grid in eV, ascending.
    pub electron_energy: Vec<f64>,
    /// Outgoing photon scaled energy grid `k = E_γ / T_e`, ascending in [0, 1].
    pub photon_energy: Vec<f64>,
    /// Scaled DCS `k · dσ/dk / Z²` stored row-major: `dcs[i_e][i_k]`.
    pub dcs: Vec<Vec<f64>>,
    /// Binding energies of Sternheimer oscillators in eV.
    pub ionization_energy: Vec<f64>,
    /// Number of electrons per Sternheimer oscillator.
    pub num_electrons: Vec<f64>,
}

impl PhotonElement {
    /// Number of energy grid points.
    pub fn n_energy(&self) -> usize {
        self.energy.len()
    }

    /// Total photon cross section in barns/atom at grid point `i`.
    ///
    /// Sum of coherent, incoherent, photoelectric, and both pair-production
    /// channels. For runtime reaction sampling use the per-channel arrays
    /// directly; this helper is for diagnostics.
    pub fn total_xs_at(&self, i: usize) -> f64 {
        self.coherent_xs[i]
            + self.incoherent_xs[i]
            + self.photoelectric_xs[i]
            + self.pair_production_nuclear_xs[i]
            + self.pair_production_electron_xs[i]
    }
}

impl Subshell {
    /// Given the master energy grid and an index `i_master` into it,
    /// return the subshell partial cross section at that energy, or 0
    /// if below the subshell's tabulation window (i.e. below binding).
    ///
    /// Uses the OpenMC tail-alignment convention
    /// `xs[j] = sigma_at(energy[N_E - xs.len() + j])`.
    pub fn xs_at(&self, n_energy_master: usize, i_master: usize) -> f64 {
        let offset = n_energy_master - self.xs.len();
        if i_master < offset {
            0.0
        } else {
            self.xs[i_master - offset]
        }
    }
}

impl ComptonProfiles {
    /// Number of Compton shells.
    pub fn n_shells(&self) -> usize {
        self.binding_energy.len()
    }

    /// Momentum grid length (typically 31 for OpenMC files).
    pub fn n_pz(&self) -> usize {
        self.pz.len()
    }
}
