//! Material definitions — nuclide compositions and macroscopic cross-sections.

/// A nuclide within a material: atom density + cross-section kernel index.
#[derive(Debug, Clone)]
pub struct NuclideEntry {
    /// Atom density (atoms/barn-cm).
    pub atom_density: f64,
    /// Index into the global cross-section kernels array.
    pub xs_kernel_idx: usize,
}

/// A material: a mixture of nuclides at given atom densities.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub nuclides: Vec<NuclideEntry>,
    /// Temperature in Kelvin (used for cross-section lookup).
    pub temperature: f64,
}

impl Material {
    pub fn new(name: &str, temperature: f64) -> Self {
        Self {
            name: name.to_string(),
            nuclides: Vec::new(),
            temperature,
        }
    }

    pub fn add_nuclide(&mut self, atom_density: f64, xs_kernel_idx: usize) {
        self.nuclides.push(NuclideEntry {
            atom_density,
            xs_kernel_idx,
        });
    }

    /// Compute macroscopic total cross-section at a given energy.
    ///
    /// Σ_t(E) = Σ_i N_i · σ_t,i(E)
    ///
    /// `micro_totals` is a slice of microscopic total cross-sections (barns)
    /// for each nuclide in this material, evaluated at the particle's energy.
    #[inline]
    pub fn macro_total(&self, micro_totals: &[f64]) -> f64 {
        self.nuclides
            .iter()
            .zip(micro_totals.iter())
            .map(|(nuc, &sigma)| nuc.atom_density * sigma)
            .sum()
    }

    /// Sample which nuclide a collision occurs with.
    ///
    /// Returns the index into `self.nuclides`.
    /// Probability proportional to N_i · σ_t,i(E).
    #[inline]
    pub fn sample_nuclide(&self, micro_totals: &[f64], macro_total: f64, xi: f64) -> usize {
        let threshold = xi * macro_total;
        let mut cumulative = 0.0;
        for (i, (nuc, &sigma)) in self.nuclides.iter().zip(micro_totals.iter()).enumerate() {
            cumulative += nuc.atom_density * sigma;
            if threshold < cumulative {
                return i;
            }
        }
        self.nuclides.len() - 1
    }
}
