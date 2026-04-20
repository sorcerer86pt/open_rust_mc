//! S(α,β) thermal scattering data and sampling algorithms.
//!
//! Implements thermal neutron scattering for bound atoms in molecules and
//! crystals (e.g., H in H₂O, C in graphite). Below `energy_max` (~4 eV),
//! the free-gas elastic scattering model is replaced by data from S(α,β)
//! tables that account for chemical binding and collective effects.
//!
//! References:
//!   - OpenMC docs, Section 5.11: S(α,β,T) Tables
//!   - OpenMC docs, Section 3.3: Thermal Neutron Scattering Data (HDF5 format)
//!   - Williams, "The Slowing Down and Thermalization of Neutrons" (1966)
//!   - Squires, "Introduction to the Theory of Thermal Neutron Scattering" (1978)

use crate::transport::rng::Rng;

// ── Data Structures ────────────────────────────────────────────────────

/// Complete thermal scattering data for one material (e.g., H in H₂O).
///
/// Loaded from an OpenMC HDF5 thermal scattering file (e.g., `c_H_in_H2O.h5`).
/// Contains per-temperature inelastic (always) and elastic (optional) data.
pub struct ThermalScatteringData {
    /// Name (e.g., "c_H_in_H2O").
    pub name: String,
    /// Nuclide names this data applies to (e.g., \["H1"\]).
    pub nuclides: Vec<String>,
    /// Maximum energy in eV — above this, use free-gas model.
    pub energy_max: f64,
    /// Atomic weight ratio of the scatterer.
    pub awr: f64,
    /// kT values in eV for each temperature, sorted ascending.
    pub kts: Vec<f64>,
    /// Temperature labels (e.g., \["294K", "600K"\]).
    pub temp_labels: Vec<String>,
    /// Inelastic scattering data per temperature (always present).
    pub inelastic: Vec<InelasticThermal>,
    /// Elastic scattering data per temperature (optional — absent for H in H₂O).
    pub elastic: Option<Vec<ElasticThermal>>,
}

/// Inelastic thermal scattering for one temperature.
///
/// Cross section + correlated angle-energy distribution.
pub struct InelasticThermal {
    /// Incident energy grid in eV (n_inc points).
    pub energy: Vec<f64>,
    /// Cross section in barns at each incident energy.
    pub xs: Vec<f64>,
    /// Distribution type.
    pub dist: InelasticDist,
}

/// Inelastic scattering distribution — two variants.
pub enum InelasticDist {
    /// Continuous tabular (iwt=2 in NJOY): PDF/CDF for outgoing energy,
    /// discrete equiprobable cosines for each (E_in, E_out) pair.
    Continuous(ContinuousInelastic),
    /// Discrete equiprobable outgoing energies (iwt=0 or 1 in NJOY).
    Discrete(DiscreteInelastic),
}

/// Continuous inelastic distribution (most common in modern libraries).
///
/// Layout follows OpenMC's "correlated" / "incoherent_inelastic" format:
///   - `energy_out_data[5][N_total]`: packed outgoing energy distributions
///   - `mu_data[3][M_total]`: packed angular distributions
pub struct ContinuousInelastic {
    /// Incoming energy grid (same as parent `InelasticThermal.energy`).
    pub n_inc: usize,
    /// Offset into `energy_out_*` arrays for each incoming energy.
    pub offsets: Vec<usize>,
    /// Interpolation code per incoming energy distribution.
    pub interp: Vec<u32>,
    /// Outgoing energies (flattened across all incoming energies).
    pub e_out: Vec<f64>,
    /// PDF for outgoing energies.
    pub pdf_e: Vec<f64>,
    /// CDF for outgoing energies.
    pub cdf_e: Vec<f64>,
    /// Interpolation codes for angular distributions (one per outgoing energy point).
    pub mu_interp: Vec<u32>,
    /// Offsets into `mu_*` arrays for angular distributions (one per outgoing energy point).
    pub mu_offsets: Vec<usize>,
    /// Cosine values (flattened across all angular distributions).
    pub mu: Vec<f64>,
    /// PDF for cosines.
    pub pdf_mu: Vec<f64>,
    /// CDF for cosines.
    pub cdf_mu: Vec<f64>,
}

/// Discrete inelastic distribution (equiprobable or skewed outgoing energies).
pub struct DiscreteInelastic {
    /// Outgoing energies: `[n_inc][n_out]` row-major.
    pub energy_out: Vec<f64>,
    /// Number of outgoing energy bins.
    pub n_out: usize,
    /// Cosines: `[n_inc][n_out][n_mu]` row-major.
    pub mu_out: Vec<f64>,
    /// Number of discrete cosines per (E_in, E_out) pair.
    pub n_mu: usize,
    /// Whether distribution is skewed (first/last=1, second/second-to-last=4, rest=10).
    pub skewed: bool,
}

/// Elastic thermal scattering for one temperature.
pub enum ElasticThermal {
    /// Coherent elastic (crystalline materials: graphite, Be).
    /// σ(E) = (1/E) Σ_{E_i < E} s_i
    Coherent {
        /// Bragg edge energies in eV, sorted ascending.
        bragg_edges: Vec<f64>,
        /// Cumulative structure factors (partial sums of s_i).
        factors: Vec<f64>,
    },
    /// Incoherent elastic (hydrogenous solids: polyethylene, ZrH).
    /// σ(E) = (σ_b/2) · (1 - e^{-4EW'}) / (2EW')
    Incoherent {
        /// Characteristic bound cross section in barns.
        bound_xs: f64,
        /// Debye-Waller integral divided by atomic mass, in eV⁻¹.
        debye_waller: f64,
    },
    /// Incoherent elastic with discrete equiprobable cosines (from ACE data).
    IncoherentDiscrete {
        /// Characteristic bound cross section in barns.
        bound_xs: f64,
        /// Debye-Waller integral in eV⁻¹.
        debye_waller: f64,
        /// Energy grid for angular distributions.
        energy: Vec<f64>,
        /// Discrete cosines: `[n_energy][n_mu]` row-major.
        mu_out: Vec<f64>,
        /// Number of discrete cosines per energy.
        n_mu: usize,
    },
}

// ── Cross Section Evaluation ───────────────────────────────────────────

impl ThermalScatteringData {
    /// Select temperature index using stochastic interpolation (OpenMC method).
    ///
    /// Given actual temperature T (in K), finds bounding temperatures and
    /// randomly selects one: P(T_{i+1}) = (T - T_i) / (T_{i+1} - T_i).
    pub fn select_temperature(&self, temperature_k: f64, xi: f64) -> usize {
        let k_boltzmann = 8.617_333_262e-5; // eV/K
        let kt = temperature_k * k_boltzmann;

        if self.kts.len() == 1 {
            return 0;
        }

        // Find bounding temperatures
        let mut i = 0;
        while i + 1 < self.kts.len() && self.kts[i + 1] < kt {
            i += 1;
        }
        if i + 1 >= self.kts.len() {
            return self.kts.len() - 1;
        }

        let f = (kt - self.kts[i]) / (self.kts[i + 1] - self.kts[i]);
        if xi < f { i + 1 } else { i }
    }

    /// Get total thermal scattering cross section at given energy and temperature index.
    pub fn total_xs(&self, energy: f64, temp_idx: usize) -> f64 {
        if energy > self.energy_max {
            return 0.0;
        }
        let mut sigma = self.inelastic_xs(energy, temp_idx);
        if let Some(ref elastic) = self.elastic {
            sigma += elastic[temp_idx].xs(energy);
        }
        sigma
    }

    /// Get inelastic cross section by linear interpolation on the energy grid.
    pub fn inelastic_xs(&self, energy: f64, temp_idx: usize) -> f64 {
        let inel = &self.inelastic[temp_idx];
        interp_lin(&inel.energy, &inel.xs, energy)
    }
}

impl ElasticThermal {
    /// Evaluate elastic thermal scattering cross section.
    pub fn xs(&self, energy: f64) -> f64 {
        match self {
            Self::Coherent {
                bragg_edges,
                factors,
            } => {
                // σ(E) = (1/E) Σ_{E_i < E} s_i  (Eq. 79 in OpenMC docs)
                let idx = bragg_edges.partition_point(|&e| e < energy);
                if idx == 0 {
                    0.0
                } else {
                    factors[idx - 1] / energy
                }
            }
            Self::Incoherent {
                bound_xs,
                debye_waller,
            }
            | Self::IncoherentDiscrete {
                bound_xs,
                debye_waller,
                ..
            } => {
                // σ(E) = (σ_b/2) · (1 - e^{-4EW'}) / (2EW')  (Eq. 80)
                let w = *debye_waller;
                let x = 4.0 * energy * w;
                if x < 1e-10 {
                    *bound_xs
                } else {
                    bound_xs / 2.0 * (1.0 - (-x).exp()) / (2.0 * energy * w)
                }
            }
        }
    }
}

// ── Sampling Algorithms ────────────────────────────────────────────────

impl ThermalScatteringData {
    /// Sample outgoing energy and angle from thermal scattering.
    ///
    /// Returns `(E_out, mu)` where `E_out` is the post-collision energy in eV
    /// and `mu` is the cosine of the scattering angle in the lab frame.
    ///
    /// The energy does NOT change for elastic scattering — only the angle changes.
    pub fn sample(&self, energy: f64, temp_idx: usize, rng: &mut Rng) -> (f64, f64) {
        let sigma_inel = self.inelastic_xs(energy, temp_idx);
        let sigma_el = self
            .elastic
            .as_ref()
            .map(|el| el[temp_idx].xs(energy))
            .unwrap_or(0.0);
        let sigma_total = sigma_inel + sigma_el;

        if sigma_total <= 0.0 {
            // Fallback: isotropic with no energy change
            return (energy, 2.0 * rng.uniform() - 1.0);
        }

        // Sample elastic vs inelastic
        if rng.uniform() < sigma_el / sigma_total {
            // Elastic scattering — energy unchanged, sample angle
            let mu = self.sample_elastic_angle(energy, temp_idx, rng);
            (energy, mu)
        } else {
            // Inelastic scattering — sample outgoing energy and angle
            self.sample_inelastic(energy, temp_idx, rng)
        }
    }

    /// Sample angle for elastic thermal scattering.
    fn sample_elastic_angle(&self, energy: f64, temp_idx: usize, rng: &mut Rng) -> f64 {
        let elastic = self.elastic.as_ref().expect("elastic data required");
        match &elastic[temp_idx] {
            ElasticThermal::Coherent {
                bragg_edges,
                factors,
            } => {
                // Sample Bragg edge with probability s_i / Σs_j (Eq. 81)
                let n = bragg_edges.partition_point(|&e| e < energy);
                if n == 0 {
                    return 2.0 * rng.uniform() - 1.0;
                }
                let total_s = factors[n - 1];
                let xi = rng.uniform() * total_s;
                let edge_idx = factors[..n].partition_point(|&f| f < xi);
                let edge_idx = edge_idx.min(n - 1);
                // μ = 1 - 2E_i/E (Eq. 82)
                let mu = 1.0 - 2.0 * bragg_edges[edge_idx] / energy;
                mu.clamp(-1.0, 1.0)
            }
            ElasticThermal::Incoherent { debye_waller, .. } => {
                // μ = (1/c) · ln(1 + ξ·(e^{2c} - 1)) - 1 (Eq. 83)
                let c = 2.0 * energy * debye_waller;
                if c < 1e-10 {
                    return 2.0 * rng.uniform() - 1.0;
                }
                let xi = rng.uniform();
                let mu = (1.0 / c) * (1.0 + xi * ((2.0 * c).exp() - 1.0)).ln() - 1.0;
                mu.clamp(-1.0, 1.0)
            }
            ElasticThermal::IncoherentDiscrete {
                energy: e_grid,
                mu_out,
                n_mu,
                debye_waller,
                ..
            } => {
                // Discrete equiprobable cosines with interpolation + smearing (Eq. 84-87)
                sample_discrete_elastic_angle(energy, e_grid, mu_out, *n_mu, *debye_waller, rng)
            }
        }
    }

    /// Sample outgoing energy and angle for inelastic thermal scattering.
    fn sample_inelastic(&self, energy: f64, temp_idx: usize, rng: &mut Rng) -> (f64, f64) {
        let inel = &self.inelastic[temp_idx];
        match &inel.dist {
            InelasticDist::Continuous(c) => {
                sample_continuous_inelastic(energy, &inel.energy, c, rng)
            }
            InelasticDist::Discrete(d) => sample_discrete_inelastic(energy, &inel.energy, d, rng),
        }
    }
}

// ── Continuous Inelastic Sampling (iwt=2) ──────────────────────────────

/// Sample outgoing energy and angle from continuous tabular inelastic distribution.
///
/// Algorithm (OpenMC Section 5.11.4.3 / 5.8.2.2):
///   1. Find bounding incoming energies, compute interpolation factor f
///   2. Statistical interpolation: choose table ℓ (stochastic)
///   3. Sample outgoing energy bin from CDF
///   4. Linear-linear interpolation for outgoing energy
///   5. Scale to kinematic bounds
///   6. Sample discrete equiprobable cosine with smearing
fn sample_continuous_inelastic(
    energy: f64,
    inc_energy: &[f64],
    c: &ContinuousInelastic,
    rng: &mut Rng,
) -> (f64, f64) {
    let n_inc = inc_energy.len();
    if n_inc == 0 {
        return (energy, 2.0 * rng.uniform() - 1.0);
    }

    // Step 1: Find bounding incoming energies
    let mut i = inc_energy.partition_point(|&e| e < energy);
    if i == 0 {
        i = 1;
    }
    if i >= n_inc {
        i = n_inc - 1;
    }
    let i_lo = i - 1;
    let i_hi = i;

    let f = if (inc_energy[i_hi] - inc_energy[i_lo]).abs() < 1e-30 {
        0.0
    } else {
        (energy - inc_energy[i_lo]) / (inc_energy[i_hi] - inc_energy[i_lo])
    };

    // Step 2: Statistical interpolation — choose table ℓ
    let xi1 = rng.uniform();
    let ell = if xi1 > f { i_lo } else { i_hi };

    // Get the energy_out distribution for table ℓ
    let start = c.offsets[ell];
    let end = if ell + 1 < c.offsets.len() {
        c.offsets[ell + 1]
    } else {
        c.e_out.len()
    };
    let n_out = end - start;
    if n_out < 2 {
        return (energy, 2.0 * rng.uniform() - 1.0);
    }

    let e_out = &c.e_out[start..end];
    let pdf = &c.pdf_e[start..end];
    let cdf = &c.cdf_e[start..end];

    // Step 3: Sample outgoing energy bin from CDF
    let xi2 = rng.uniform();
    let mut j = cdf.partition_point(|&c| c < xi2);
    if j == 0 {
        j = 1;
    }
    if j >= n_out {
        j = n_out - 1;
    }
    let j = j - 1; // cdf[j] < xi2 <= cdf[j+1]

    // Step 4: Linear-linear interpolation for outgoing energy (Eq. 34)
    let e_hat = if (pdf[j + 1] - pdf[j]).abs() < 1e-30 {
        // Histogram interpolation (Eq. 33)
        if pdf[j].abs() < 1e-30 {
            e_out[j]
        } else {
            e_out[j] + (xi2 - cdf[j]) / pdf[j]
        }
    } else {
        // Linear-linear (Eq. 34)
        let m = (pdf[j + 1] - pdf[j]) / (e_out[j + 1] - e_out[j]);
        let discriminant = pdf[j] * pdf[j] + 2.0 * m * (xi2 - cdf[j]);
        if discriminant < 0.0 {
            e_out[j]
        } else {
            e_out[j] + (discriminant.sqrt() - pdf[j]) / m
        }
    };

    // Step 5: Remap to the incident energy. OpenMC
    // (src/secondary_thermal.cpp, IncoherentInelasticAE::sample_params)
    // uses a two-branch piecewise form keyed off the chosen table's
    // incident energy E_l = inc_energy[ell]:
    //     if E_out < 0.5 * E_l: E_out *= 2*E_in/E_l - 1
    //     else:                 E_out += E_in - E_l
    // This matches NJOY/ACE-file thermal-scattering conventions and
    // diverges from the canonical Eqs 31/35 linear-bounds remap
    // described in the OpenMC manual. The piecewise form is what
    // actually runs in the reference implementation, so we mirror it
    // to keep thermal spectra consistent (important for PWR k_inf).
    let e_l = inc_energy[ell];
    let e_out_final = if e_l <= 0.0 {
        e_hat
    } else if e_hat < 0.5 * e_l {
        e_hat * (2.0 * energy / e_l - 1.0)
    } else {
        e_hat + energy - e_l
    };
    let e_out_final = e_out_final.max(0.0);

    // Step 6: Sample angular distribution
    // The mu offset for this (ℓ, j) pair is in mu_offsets[start + j]
    let mu_off_idx = start + j;
    let mu = if mu_off_idx < c.mu_offsets.len() {
        let mu_start = c.mu_offsets[mu_off_idx];
        let mu_end = if mu_off_idx + 1 < c.mu_offsets.len() {
            c.mu_offsets[mu_off_idx + 1]
        } else {
            c.mu.len()
        };
        if mu_end > mu_start && mu_start < c.mu.len() {
            // Discrete equiprobable cosines — sample uniformly + smear
            let n_mu = mu_end - mu_start;
            let mu_vals = &c.mu[mu_start..mu_end];

            // For continuous inelastic, use the same smearing as OpenMC (Eq. 87-89)
            // But first: simple equiprobable sampling with linear interpolation
            // between tables for incoming energy ℓ
            let k = (rng.uniform() * n_mu as f64) as usize;
            let k = k.min(n_mu - 1);
            let mu_k = mu_vals[k];

            // Smear: μ = μ_k + min(μ_k - μ_{k-1}, μ_{k+1} - μ_k) · (ξ - 0.5)
            let left = if k > 0 {
                mu_k - mu_vals[k - 1]
            } else {
                mu_k + 1.0
            };
            let right = if k + 1 < n_mu {
                mu_vals[k + 1] - mu_k
            } else {
                1.0 - mu_k
            };
            let half_width = left.min(right);
            mu_k + half_width * (rng.uniform() - 0.5)
        } else {
            2.0 * rng.uniform() - 1.0
        }
    } else {
        2.0 * rng.uniform() - 1.0
    };

    (e_out_final, mu.clamp(-1.0, 1.0))
}

// ── Discrete Inelastic Sampling (iwt=0,1) ──────────────────────────────

/// Sample outgoing energy and angle from discrete inelastic distribution.
///
/// Algorithm (OpenMC Section 5.11.4.1 / 5.11.4.2):
///   1. Find bounding incoming energies, compute interpolation factor f
///   2. Sample outgoing energy bin (uniform or skewed probability)
///   3. Interpolate outgoing energy between tables
///   4. Sample cosine from discrete equiprobable bins + interpolation + smearing
fn sample_discrete_inelastic(
    energy: f64,
    inc_energy: &[f64],
    d: &DiscreteInelastic,
    rng: &mut Rng,
) -> (f64, f64) {
    let n_inc = inc_energy.len();
    if n_inc == 0 {
        return (energy, 2.0 * rng.uniform() - 1.0);
    }

    // Find bounding incoming energies
    let mut i = inc_energy.partition_point(|&e| e < energy);
    if i == 0 {
        i = 1;
    }
    if i >= n_inc {
        i = n_inc - 1;
    }
    let i_lo = i - 1;

    let f = if (inc_energy[i] - inc_energy[i_lo]).abs() < 1e-30 {
        0.0
    } else {
        (energy - inc_energy[i_lo]) / (inc_energy[i] - inc_energy[i_lo])
    };

    let n_out = d.n_out;
    let n_mu = d.n_mu;

    // Sample outgoing energy bin
    let j = if d.skewed {
        // Skewed: first/last=1, second/second-to-last=4, all others=10
        sample_skewed_bin(n_out, rng)
    } else {
        // Equiprobable
        let j = (rng.uniform() * n_out as f64) as usize;
        j.min(n_out - 1)
    };

    // Interpolate outgoing energy (Eq. 88)
    let e_lo = d.energy_out[i_lo * n_out + j];
    let e_hi = d.energy_out[i * n_out + j];
    let e_out = e_lo + f * (e_hi - e_lo);

    // Sample discrete equiprobable cosine bin
    let k = (rng.uniform() * n_mu as f64) as usize;
    let k = k.min(n_mu - 1);

    // Interpolate cosine (Eq. 89)
    let mu_lo = d.mu_out[(i_lo * n_out + j) * n_mu + k];
    let mu_hi = d.mu_out[(i * n_out + j) * n_mu + k];
    let mu_prime = mu_lo + f * (mu_hi - mu_lo);

    // Smearing (Eq. 86-87)
    let mu_left = if k > 0 {
        let ml = d.mu_out[(i_lo * n_out + j) * n_mu + k - 1];
        let mh = d.mu_out[(i * n_out + j) * n_mu + k - 1];
        ml + f * (mh - ml)
    } else {
        -1.0
    };
    let mu_right = if k + 1 < n_mu {
        let ml = d.mu_out[(i_lo * n_out + j) * n_mu + k + 1];
        let mh = d.mu_out[(i * n_out + j) * n_mu + k + 1];
        ml + f * (mh - ml)
    } else {
        1.0
    };
    let half_width = (mu_prime - mu_left).min(mu_right - mu_prime);
    let mu = mu_prime + half_width * (rng.uniform() - 0.5);

    (e_out.max(0.0), mu.clamp(-1.0, 1.0))
}

// ── Discrete Elastic Angle Sampling ────────────────────────────────────

/// Sample angle for incoherent elastic with discrete cosines.
fn sample_discrete_elastic_angle(
    energy: f64,
    e_grid: &[f64],
    mu_out: &[f64],
    n_mu: usize,
    _debye_waller: f64,
    rng: &mut Rng,
) -> f64 {
    let n_e = e_grid.len();
    if n_e == 0 || n_mu == 0 {
        return 2.0 * rng.uniform() - 1.0;
    }

    // Find bounding energies
    let mut i = e_grid.partition_point(|&e| e < energy);
    if i == 0 {
        i = 1;
    }
    if i >= n_e {
        i = n_e - 1;
    }
    let i_lo = i - 1;

    let f = if (e_grid[i] - e_grid[i_lo]).abs() < 1e-30 {
        0.0
    } else {
        (energy - e_grid[i_lo]) / (e_grid[i] - e_grid[i_lo])
    };

    // Sample equiprobable cosine bin
    let j = (rng.uniform() * n_mu as f64) as usize;
    let j = j.min(n_mu - 1);

    // Interpolate (Eq. 84)
    let mu_lo = mu_out[i_lo * n_mu + j];
    let mu_hi = mu_out[i * n_mu + j];
    let mu_prime = mu_lo + f * (mu_hi - mu_lo);

    // Smearing (Eq. 86-87)
    let mu_left = if j > 0 {
        let ml = mu_out[i_lo * n_mu + j - 1];
        let mh = mu_out[i * n_mu + j - 1];
        ml + f * (mh - ml)
    } else {
        -1.0
    };
    let mu_right = if j + 1 < n_mu {
        let ml = mu_out[i_lo * n_mu + j + 1];
        let mh = mu_out[i * n_mu + j + 1];
        ml + f * (mh - ml)
    } else {
        1.0
    };
    let half_width = (mu_prime - mu_left).min(mu_right - mu_prime);
    let mu = mu_prime + half_width * (rng.uniform() - 0.5);
    mu.clamp(-1.0, 1.0)
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Sample from skewed discrete distribution (NJOY iwt=0).
/// First/last bins have weight 1, second/second-to-last have weight 4,
/// all others have weight 10.
fn sample_skewed_bin(n: usize, rng: &mut Rng) -> usize {
    if n <= 2 {
        return (rng.uniform() * n as f64) as usize;
    }

    let total: f64 = if n <= 4 {
        // All have explicit weights
        (0..n).map(|i| skewed_weight(i, n)).sum()
    } else {
        2.0 * 1.0 + 2.0 * 4.0 + (n - 4) as f64 * 10.0
    };

    let xi = rng.uniform() * total;
    let mut cum = 0.0;
    for i in 0..n {
        cum += skewed_weight(i, n);
        if xi < cum {
            return i;
        }
    }
    n - 1
}

fn skewed_weight(i: usize, n: usize) -> f64 {
    if i == 0 || i == n - 1 {
        1.0
    } else if i == 1 || i == n - 2 {
        4.0
    } else {
        10.0
    }
}

/// Linear interpolation on a sorted grid.
fn interp_lin(x: &[f64], y: &[f64], xq: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    if xq <= x[0] {
        return y[0];
    }
    if xq >= x[x.len() - 1] {
        return y[x.len() - 1];
    }

    let mut i = x.partition_point(|&v| v < xq);
    if i == 0 {
        i = 1;
    }
    if i >= x.len() {
        return y[x.len() - 1];
    }

    let f = (xq - x[i - 1]) / (x[i] - x[i - 1]);
    y[i - 1] + f * (y[i] - y[i - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_elastic_xs() {
        let el = ElasticThermal::Coherent {
            bragg_edges: vec![0.001, 0.002, 0.003],
            factors: vec![1.0, 3.0, 6.0],
        };
        // Below first edge: xs = 0
        assert_eq!(el.xs(0.0005), 0.0);
        // Between first and second: xs = s_0 / E = 1.0 / 0.0015
        let xs = el.xs(0.0015);
        assert!((xs - 1.0 / 0.0015).abs() < 1e-10);
        // After all edges: xs = s_total / E = 6.0 / 0.01
        let xs = el.xs(0.01);
        assert!((xs - 6.0 / 0.01).abs() < 1e-10);
    }

    #[test]
    fn incoherent_elastic_xs() {
        let el = ElasticThermal::Incoherent {
            bound_xs: 20.0,
            debye_waller: 1.0,
        };
        // At E=0.01: σ = (20/2) * (1 - e^{-0.04}) / (0.02)
        let xs = el.xs(0.01);
        let expected = 10.0 * (1.0 - (-0.04_f64).exp()) / 0.02;
        assert!((xs - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn interp_lin_basic() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 10.0, 20.0];
        assert!((interp_lin(&x, &y, 0.5) - 5.0).abs() < 1e-10);
        assert!((interp_lin(&x, &y, 1.5) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn skewed_bin_weights() {
        // n=5: weights = [1, 4, 10, 4, 1], total = 20
        assert_eq!(skewed_weight(0, 5), 1.0);
        assert_eq!(skewed_weight(1, 5), 4.0);
        assert_eq!(skewed_weight(2, 5), 10.0);
        assert_eq!(skewed_weight(3, 5), 4.0);
        assert_eq!(skewed_weight(4, 5), 1.0);
    }

    // ── Continuous inelastic iwt=2 piecewise remap (OpenMC form) ──
    //
    // Regression for the NJOY/ACE-convention remap ported from
    // OpenMC's IncoherentInelasticAE::sample_params. We drive the
    // sampler with a minimal, deterministic ContinuousInelastic table
    // and assert energy bounds and the two-branch behaviour.

    fn build_minimal_continuous_inelastic() -> (Vec<f64>, ContinuousInelastic) {
        // Two incident-energy tables at E = 1.0 and 2.0 eV. Each table
        // has 3 outgoing-energy points; a linear CDF 0.0, 0.5, 1.0 so a
        // binary-search inversion hits the mid bin for xi≈0.75.
        let inc_energy = vec![1.0_f64, 2.0_f64];
        let offsets = vec![0_usize, 3_usize];
        let e_out = vec![0.1_f64, 0.5_f64, 0.9_f64, 0.2_f64, 1.0_f64, 1.8_f64];
        let pdf_e = vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
        let cdf_e = vec![0.0_f64, 0.5_f64, 1.0_f64, 0.0_f64, 0.5_f64, 1.0_f64];
        // One μ bin per (ℓ, j): μ = 0.0. No smearing exercised.
        let mu_offsets = vec![0_usize, 1, 2, 3, 4, 5];
        let mu = vec![0.0_f64; 6];

        let n_inc = inc_energy.len();
        let interp = vec![2_u32; n_inc]; // lin-lin
        let mu_interp = vec![2_u32; mu_offsets.len()];
        let pdf_mu = vec![0.5_f64; mu.len()];
        let cdf_mu = vec![1.0_f64; mu.len()];
        let c = ContinuousInelastic {
            n_inc,
            offsets,
            interp,
            e_out,
            pdf_e,
            cdf_e,
            mu_interp,
            mu_offsets,
            mu,
            pdf_mu,
            cdf_mu,
        };
        (inc_energy, c)
    }

    #[test]
    fn thermal_sample_continuous_stays_positive_and_bounded() {
        let (inc_energy, c) = build_minimal_continuous_inelastic();
        let mut rng = Rng::new(2026, 42);
        for _ in 0..500 {
            let e_in = 1.0 + rng.uniform();
            let (e_out, mu) = sample_continuous_inelastic(e_in, &inc_energy, &c, &mut rng);
            assert!(e_out >= 0.0, "E_out must be non-negative, got {e_out}");
            // Hard upper bound: the larger of the two branches' outputs,
            // roughly E_in + (largest tabulated E_hat) ≈ 2 + 1.8 = 3.8.
            assert!(e_out < 5.0, "E_out runaway: {e_out} at E_in={e_in}");
            assert!((-1.0..=1.0).contains(&mu));
        }
    }

    #[test]
    fn thermal_sample_at_table_energy_is_near_tabulated_eout() {
        // When E_in exactly equals a table's incident energy, both
        // branches of the piecewise remap collapse to identity:
        // low branch: e_hat * (2*E_in/E_l - 1) = e_hat * 1 = e_hat
        // high branch: e_hat + E_in - E_l = e_hat
        // → returned E_out must equal sampled e_hat.
        let (inc_energy, c) = build_minimal_continuous_inelastic();
        let mut rng = Rng::new(2027, 1);
        let trials = 2000;
        let mut sum_in_table_range = 0;
        for _ in 0..trials {
            let (e_out, _) = sample_continuous_inelastic(1.0, &inc_energy, &c, &mut rng);
            // Table-1 e_out range is [0.1, 0.9]; allow for either table pick.
            if (0.1..=1.8).contains(&e_out) {
                sum_in_table_range += 1;
            }
        }
        // At E_in = E_lo = 1.0 (f=0 so statistical pick always chooses lo
        // once xi > 0 ≈ always), e_out should fall inside tabulated
        // range. Allow a few outliers from boundary numerics.
        assert!(
            sum_in_table_range > (trials * 95) / 100,
            "only {sum_in_table_range}/{trials} within tabulated range"
        );
    }
}
