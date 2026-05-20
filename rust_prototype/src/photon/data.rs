// SPDX-License-Identifier: MIT
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

/// A tabulated factor `y(x)` on a shared momentum-transfer grid.
///
/// Used for `F(x, Z)` (coherent form factor), its cumulative
/// `∫₀^{x²} F² dx'²`, and `S(x, Z)` (incoherent scattering function).
/// All three are stored as shape `(2, N)` in HDF5 with row 0 the
/// independent variable and row 1 the dependent value. We keep them
/// split into two equal-length vectors for clarity.
///
/// The `x` grid follows OpenMC's tabulation convention
/// (Hubbell et al. 1975): `x = sin(θ/2) / λ` in inverse Ångström
/// where `λ = hc / E`. OpenMC extends the tabulated grid past any
/// physically reachable `x` (up to ~10⁹ Å⁻¹) with zero-valued factors
/// so that interpolation at any kinematic point is always in-range;
/// the physical cutoff at a given photon energy is considerably lower
/// (`x_max ≈ E [eV] / 12398.4 Å⁻¹`).
pub struct ScatteringFactor {
    /// Momentum-transfer variable `x` in inverse Ångström, ascending
    /// from 0.
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
/// The outgoing photon energy in Compton scattering from a bound
/// electron deviates from the free-electron Klein-Nishina value by
/// `p_z / (m_e c)`, the projection of the electron's pre-collision
/// momentum along the scattering axis. `Jᵢ(p_z)` is the probability
/// density of that projection for an electron in shell `i`. Tabulated
/// on the shared `pz` grid in atomic units of momentum
/// (1 a.u. = α m_e c ≈ m_e c / 137).
///
/// **Symmetric storage.** OpenMC stores only non-negative `p_z` values
/// because `Jᵢ` is even (closed-shell Hartree-Fock ground state): the
/// full density is `Jᵢ(p_z) = Jᵢ(|p_z|)`. Sampling a signed `p_z` at
/// runtime is a reflect-by-coin-flip after drawing from the stored
/// non-negative half.
///
/// **Shell partitioning ≠ photoelectric subshells.** The Compton shell
/// list generally has fewer entries than `PhotonElement::subshells`
/// (27 vs 29 for Uranium, 3 vs 4 for Carbon) because the Hartree-Fock
/// tabulation merges some outer shells. The Compton sampler selects a
/// shell from `(binding_energy, num_electrons)` here, independent of
/// the photoelectric subshell list.
///
/// **Sampling sketch** (to be implemented in the Compton kernel):
/// 1. Select shell `i` by occupancy, weighted by whether the kinematic
///    limit `p_z_max(i, E, θ)` is positive.
/// 2. Sample `|p_z|` from `Jᵢ(|p_z|) · 𝟙[|p_z| < p_z_max]`.
/// 3. Sign of `p_z` by fair coin.
/// 4. Solve the Doppler-shifted `E'(p_z, θ)` quadratic.
pub struct ComptonProfiles {
    /// Binding energy of each Compton shell in eV.
    pub binding_energy: Vec<f64>,
    /// Number of electrons in each Compton shell (may be fractional).
    pub num_electrons: Vec<f64>,
    /// Non-negative momentum grid `|p_z|` in atomic units, ascending
    /// from 0.
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
    /// `[primary, secondary, energy_eV, probability]`:
    ///
    /// - `primary` (f64 cast of a subshell designator index):
    ///   the shell that donated the electron filling the hole.
    /// - `secondary`: 0 for radiative (fluorescence) transitions,
    ///   non-zero designator index for non-radiative
    ///   (Auger / Coster-Kronig) transitions — the shell from which
    ///   the Auger electron is ejected.
    /// - `energy_eV`: the directly-emitted particle's kinetic or
    ///   photon energy, taken from the EADL evaluation (not derived
    ///   from binding-energy differences at runtime).
    /// - `probability`: branching probability for this transition.
    ///
    /// Transition probabilities within one subshell sum to ≤ 1; the
    /// deficit is the probability the hole persists (no decay). Outer
    /// shells typically carry no transitions.
    pub transitions: Vec<[f64; 4]>,
}

/// Seltzer-Berger bremsstrahlung differential cross section and
/// Sternheimer-Berger mean-excitation-energy oscillator parameters.
///
/// The DCS is tabulated on a two-dimensional (electron kinetic energy,
/// scaled photon energy) grid. Shape: `[N_electron, N_photon]`. Exact
/// scaling (whether the stored values are `dσ/dk`, `k·dσ/dk/Z²`, or
/// some other SB convention) is the TTB kernel's responsibility to
/// verify against Seltzer-Berger 1986 and OpenMC's
/// `thick_target_bremsstrahlung` routine — the data-layer contract is
/// only that `dcs[i_e][i_k]` maps to
/// `(electron_energy[i_e], photon_energy[i_k])`.
///
/// The Sternheimer oscillators (`ionization_energy`, `num_electrons`)
/// define the atomic density-effect correction to the electron
/// collisional stopping power via the Berger-Seltzer parametrisation.
/// `mean_excitation_energy` is the atomic `I`-value in eV matching
/// ICRU-37 / Seltzer-Berger tabulations (e.g. C: 81 eV, U: 890 eV).
pub struct Bremsstrahlung {
    /// Mean excitation energy (`I`-value) in eV from the HDF5 attribute.
    pub mean_excitation_energy: f64,
    /// Electron kinetic energy grid in eV, ascending.
    pub electron_energy: Vec<f64>,
    /// Outgoing photon scaled energy grid `k = E_γ / T_e`, ascending in
    /// `[0, 1]`.
    pub photon_energy: Vec<f64>,
    /// DCS table, row-major: `dcs[i_e][i_k]` at
    /// `(electron_energy[i_e], photon_energy[i_k])`. Scaling convention
    /// is Seltzer-Berger 1986; exact factors deferred to the TTB
    /// kernel.
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

    /// Look up a photoelectric subshell by its EADL 1-based designator
    /// (K=1, L1=2, L2=3, L3=4, M1=5, ..., M5=9, N1=10, ..., Q1=29).
    ///
    /// Returns `None` if the designator is 0 or exceeds the element's
    /// tabulated subshell list. The latter happens for outer valence
    /// shells that can donate electrons during relaxation but have no
    /// tabulated partial photoelectric cross section — e.g. an EADL
    /// transition with `primary = 37` in uranium references a shell
    /// beyond the 29-entry Q1 terminus of `self.subshells`, and the
    /// photoelectric cascade should treat the resulting hole as
    /// locally-deposited energy.
    ///
    /// Relies on OpenMC's photon HDF5 storing subshells in K → L1 →
    /// L2 → L3 → M1 → ... order, so the EADL designator equals the
    /// 1-based position in `self.subshells`. The reader enforces this
    /// order via the required `designators` attribute.
    pub fn subshell_by_eadl_designator(&self, designator: u32) -> Option<&Subshell> {
        if designator == 0 {
            return None;
        }
        self.subshells.get(designator as usize - 1)
    }
}

impl Subshell {
    /// Given the master energy grid of length `n_energy_master` and an
    /// index `i_master` into it, return the subshell partial cross
    /// section at that energy. Returns 0 if the master-grid index sits
    /// below the subshell's tabulation window (i.e. below the shell's
    /// binding energy).
    ///
    /// Uses the OpenMC tail-alignment convention
    /// `xs[j] = sigma_at(energy[N_E - xs.len() + j])`.
    ///
    /// # Panics
    /// Debug-only: panics if `i_master >= n_energy_master` or if
    /// `self.xs.len() > n_energy_master`. These are caller-side
    /// programming errors (out-of-range master index, or calling on a
    /// subshell that was loaded against a different master grid).
    pub fn xs_at(&self, n_energy_master: usize, i_master: usize) -> f64 {
        debug_assert!(
            i_master < n_energy_master,
            "i_master {i_master} >= n_energy_master {n_energy_master}"
        );
        debug_assert!(
            self.xs.len() <= n_energy_master,
            "subshell xs.len() {} > master grid len {n_energy_master}",
            self.xs.len()
        );
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
