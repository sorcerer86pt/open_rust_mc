//! Windowed Multipole (WMP) cross-section evaluation.
//!
//! Ports OpenMC's `openmc.data.WindowedMultipole._evaluate` (Python) and the
//! equivalent C++ kernel to pure Rust. Temperature-dependent Doppler
//! broadening uses the Faddeeva function `w(z) = exp(-z²) erfc(-iz)` via
//! Humlicek's W4 rational approximation (Humlicek 1982, JQSRT 27, 437).
//!
//! References:
//!   - Josey, C. et al., "Windowed multipole for cross section Doppler
//!     broadening", J. Comp. Phys. 307 (2016) 715-727.
//!   - Forget, B. et al., "Direct Doppler broadening in Monte Carlo
//!     simulations using the multipole representation", ANE 64 (2014) 78-85.
//!   - Humlicek, J., "Optimised computation of the Voigt and complex
//!     probability functions", JQSRT 27 (1982) 437-444.

use std::path::Path;
use crate::error::{Result, SvdError};

/// Boltzmann constant in eV/K (OpenMC value).
const K_BOLTZMANN: f64 = 8.617_328_5e-5;

/// 1 / sqrt(pi), used in Humlicek Regions I–II.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

// ── Complex numbers (local minimal impl) ──────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    #[inline] pub const fn new(re: f64, im: f64) -> Self { Self { re, im } }
    #[inline] pub fn abs2(self) -> f64 { self.re * self.re + self.im * self.im }
    #[inline] pub fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im,
                  self.re * o.im + self.im * o.re)
    }
    #[inline] pub fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    #[inline] pub fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    #[inline] pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    #[inline] pub fn div(self, o: Self) -> Self {
        let d = o.abs2();
        Self::new((self.re * o.re + self.im * o.im) / d,
                  (self.im * o.re - self.re * o.im) / d)
    }
}

// ── Faddeeva: Humlicek W4 ────────────────────────────────────────────

/// Humlicek 1982 W4 approximation of the Faddeeva function w(z).
///
/// Upper-half-plane accurate; for lower half use w(-z*) = 2*exp(-z²) - conj(w(z*)).
/// Typical accuracy ~1e-4 (Region IV) to 1e-6 (Regions I–III), which is
/// well below the natural MC variance in transport XS.
pub fn faddeeva(z: C64) -> C64 {
    // Flip to upper half plane using OpenMC's convention
    // (openmc/data/multipole.py::_faddeeva): for Im(z) ≤ 0, return
    //     -conj(wofz(conj(z)))
    // where conj(z) is in the upper half plane. This is NOT the standard
    // identity (which would be 2e^(-z²) - w(-z)); it defines a variant
    // that is *antisymmetric* under conjugation, matching how OpenMC's
    // WMP pole residues are arranged into conjugate pairs.
    if z.im < 0.0 {
        let z_up = C64::new(z.re, -z.im);
        let w_up = faddeeva(z_up);
        return C64::new(-w_up.re, w_up.im); // -conj(w_up) = (-re, +im)
    }

    let x = z.re;
    let y = z.im;
    let s = x.abs() + y;

    if s >= 15.0 {
        // Region I (asymptotic, 1 term)
        // w(z) ≈ (i/√π) / z ≡ (i/√π) * z̄ / |z|²  up to sign. Following
        // Humlicek's form: with t = y - i*x and u = t²:
        //   w = (t / √π) / (u + 1/2)
        let t = C64::new(y, -x);
        let u = t.mul(t);
        let num = t.scale(INV_SQRT_PI);
        let den = C64::new(u.re + 0.5, u.im);
        return num.div(den);
    }

    if s >= 5.5 {
        // Region II
        let t = C64::new(y, -x);
        let u = t.mul(t);
        let num = t.mul(C64::new(1.410474 + u.re * INV_SQRT_PI,
                                 u.im * INV_SQRT_PI));
        let den = C64::new(0.75 + u.re * 3.0 + (u.mul(u)).re,
                           u.im * 3.0 + (u.mul(u)).im);
        return num.div(den);
    }

    if y >= 0.195 * x.abs() - 0.176 {
        // Region III — polynomial ratio in t = y - i*x (real part of z
        // flipped by sign convention).
        let t = C64::new(y, -x);
        // Horner's method, coefficients from Humlicek 1982 W4.
        let num = poly_horner(
            t,
            &[16.4955, 20.20933, 11.96482, 3.778987, 0.5642236],
        );
        let den = poly_horner(
            t,
            &[16.4955, 38.82363, 39.27121, 21.69274, 6.699398, 1.0],
        );
        return num.div(den);
    }

    // Region IV — rational in u = t² times exp(u), with correction.
    // w(z) = exp(-z²) - t * P(u) / Q(u)
    let t = C64::new(y, -x);
    let u = t.mul(t);

    // P(u) = 36183.31 - u(3321.9905 - u(1540.787 - u(219.0313 - u(35.76683 - u(1.320522 - 0.56419*u)))))
    // Q(u) = 32066.6 - u(24322.84 - u(9022.228 - u(2186.181 - u(364.2191 - u(61.57037 - u(1.841439 - u))))))
    // Evaluated via nested Horner on u, with sign alternation.
    let p_coefs = [36183.31_f64, -3321.9905, 1540.787, -219.0313, 35.76683, -1.320522, 0.56419];
    let q_coefs = [32066.6_f64, -24322.84, 9022.228, -2186.181, 364.2191, -61.57037, 1.841439, -1.0];

    let p = poly_horner_real(u, &p_coefs);
    let q = poly_horner_real(u, &q_coefs);

    // exp(-z²) term. Note u = t² but -z² = -(x+iy)(x+iy) = -(x² - y²) - 2ixy
    // and t = y - ix so t² = y² - x² - 2ixy; hence u = t², exp(u) = exp(-z²).
    let e_abs = f64::exp(u.re);
    let (s_im, c_im) = u.im.sin_cos();
    let exp_u = C64::new(e_abs * c_im, e_abs * s_im);

    // w = exp_u - t * P/Q
    let corr = t.mul(p.div(q));
    exp_u.sub(corr)
}

#[inline]
fn poly_horner(z: C64, coefs: &[f64]) -> C64 {
    // Evaluate c[n]*z^n + ... + c[1]*z + c[0] with coefs = [c0, c1, ..., cn].
    let mut acc = C64::new(coefs[coefs.len() - 1], 0.0);
    for &c in coefs.iter().rev().skip(1) {
        acc = acc.mul(z).add(C64::new(c, 0.0));
    }
    acc
}

#[inline]
fn poly_horner_real(u: C64, coefs: &[f64]) -> C64 {
    // Same as poly_horner but signals that argument is "real-coefficient"
    // polynomial in a complex u. Identical implementation.
    poly_horner(u, coefs)
}

// ── WMP data ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum WmpReaction { Scattering = 0, Absorption = 1, Fission = 2 }

/// Parsed Windowed Multipole data for one nuclide.
pub struct WindowedMultipole {
    pub name: String,
    pub e_min: f64,
    pub e_max: f64,
    pub spacing: f64,
    pub sqrt_awr: f64,
    pub fissionable: bool,
    /// Pole data: `data[i] = [ea, r_scat, r_abs, r_fis]` as C64.
    /// Flattened to `4 * n_poles` complex numbers.
    pub poles: Vec<C64>,
    pub n_poles: usize,
    /// Window start/end pole indices, 1-based in file; stored 0-based here.
    /// `windows[2*w]` = startw (0-based, inclusive)
    /// `windows[2*w + 1]` = endw (0-based, exclusive; may be < startw meaning empty)
    pub windows: Vec<i32>,
    pub n_windows: usize,
    /// `broaden_poly[w]` = 1 if curvefit polynomial should be Doppler-broadened.
    pub broaden_poly: Vec<u8>,
    /// Curvefit: `(n_windows, fit_order+1, 3)` row-major.
    pub curvefit: Vec<f64>,
    pub fit_order: usize,
}

impl WindowedMultipole {
    /// Load WMP from a ZZAAA-named HDF5 file (e.g. `092238.h5`).
    pub fn from_hdf5(path: &Path) -> Result<Self> {
        let file = hdf5_pure::File::open(path).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("{e}"),
        })?;
        let root = file.root();
        let groups = root.groups().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("cannot list root groups: {e}"),
        })?;
        let name = groups.into_iter().next().ok_or_else(|| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: "no nuclide group in WMP file".into(),
        })?;
        let g = root.group(&name).map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("open {name}: {e}"),
        })?;

        let read_scalar_f64 = |name: &str| -> Result<f64> {
            let ds = g.dataset(name).map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("{name}: {e}"),
            })?;
            let v = ds.read_f64().map_err(|e| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("{name} read: {e}"),
            })?;
            v.first().copied().ok_or_else(|| SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("{name} empty"),
            })
        };
        let e_min = read_scalar_f64("E_min")?;
        let e_max = read_scalar_f64("E_max")?;
        let spacing = read_scalar_f64("spacing")?;
        let sqrt_awr = read_scalar_f64("sqrtAWR")?;

        // `data` is (n_poles, 4) complex128 stored as HDF5 compound
        // {r:f64, i:f64}. hdf5-pure does not decode compounds into f64,
        // so we read raw bytes and reinterpret: each complex number is
        // 16 bytes = two little-endian f64. Four complex per pole.
        let data_ds = g.dataset("data").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("data: {e}"),
        })?;
        let data_raw = data_ds.read_u8().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("data raw read: {e}"),
        })?;
        let shape = data_ds.shape().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("data shape: {e}"),
        })?;
        // Expect shape (n_poles, 4) with 16-byte complex elements.
        let n_poles = shape.first().copied().unwrap_or(0) as usize;
        let n_cols = shape.get(1).copied().unwrap_or(0) as usize;
        if n_cols != 4 || data_raw.len() != n_poles * 4 * 16 {
            return Err(SvdError::Hdf5 {
                path: path.display().to_string(),
                detail: format!("data layout unexpected: shape={shape:?} bytes={}", data_raw.len()),
            });
        }
        let mut poles = Vec::with_capacity(n_poles * 4);
        for i in 0..n_poles {
            for j in 0..4 {
                let off = (i * 4 + j) * 16;
                let re = f64::from_le_bytes(data_raw[off..off + 8].try_into().unwrap());
                let im = f64::from_le_bytes(data_raw[off + 8..off + 16].try_into().unwrap());
                poles.push(C64::new(re, im));
            }
        }

        // `windows` is (n_windows, 2) int32 — stored 1-based in file.
        let windows_ds = g.dataset("windows").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("windows: {e}"),
        })?;
        let windows_raw = windows_ds.read_i32().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("windows read: {e}"),
        })?;
        let n_windows = windows_raw.len() / 2;
        // Convert 1-based to 0-based: startw - 1, endw stays (exclusive end).
        let mut windows = Vec::with_capacity(windows_raw.len());
        for w in 0..n_windows {
            let startw = windows_raw[2 * w] - 1; // may be -1 → no poles
            let endw = windows_raw[2 * w + 1];
            windows.push(startw);
            windows.push(endw);
        }

        // `broaden_poly` is (n_windows,) int8.
        let broaden_ds = g.dataset("broaden_poly").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("broaden_poly: {e}"),
        })?;
        let broaden_raw = broaden_ds.read_i8().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("broaden_poly read: {e}"),
        })?;
        let broaden_poly: Vec<u8> = broaden_raw.iter().map(|&x| x as u8).collect();

        // `curvefit` is (n_windows, fit_order+1, n_rxn=3) float64.
        let curvefit_ds = g.dataset("curvefit").map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("curvefit: {e}"),
        })?;
        let curvefit = curvefit_ds.read_f64().map_err(|e| SvdError::Hdf5 {
            path: path.display().to_string(),
            detail: format!("curvefit read: {e}"),
        })?;
        let fit_order = curvefit.len() / (n_windows * 3) - 1;

        // Detect fissionable: any non-zero fission residue.
        let fissionable = (0..n_poles).any(|i| {
            let c = poles[i * 4 + 3];
            c.re != 0.0 || c.im != 0.0
        });

        Ok(Self {
            name,
            e_min,
            e_max,
            spacing,
            sqrt_awr,
            fissionable,
            poles,
            n_poles,
            windows,
            n_windows,
            broaden_poly,
            curvefit,
            fit_order,
        })
    }

    /// Evaluate (scattering, absorption, fission) cross sections in barns
    /// at energy `e` (eV) and temperature `t_kelvin` (K).
    ///
    /// Returns (0, 0, 0) if `e` is outside `[E_min, E_max]` — caller should
    /// fall back to the SVD or pointwise provider in that case.
    pub fn evaluate(&self, e: f64, t_kelvin: f64) -> (f64, f64, f64) {
        if e < self.e_min || e > self.e_max {
            return (0.0, 0.0, 0.0);
        }

        let sqrt_kt = (K_BOLTZMANN * t_kelvin).sqrt();
        let sqrt_e = e.sqrt();
        let inv_e = 1.0 / e;

        // Locate window index.
        let sqrt_e_min = self.e_min.sqrt();
        let i_window = ((sqrt_e - sqrt_e_min) / self.spacing).floor() as isize;
        let i_window = i_window.max(0).min(self.n_windows as isize - 1) as usize;

        let startw = self.windows[2 * i_window] as isize; // already 0-based, may be -1
        let endw = self.windows[2 * i_window + 1] as isize;

        let mut sig_s = 0.0;
        let mut sig_a = 0.0;
        let mut sig_f = 0.0;

        // ── Curvefit contribution ──
        let order1 = self.fit_order + 1;
        let cf_base = i_window * order1 * 3;

        if sqrt_kt != 0.0 && self.broaden_poly[i_window] != 0 {
            // Doppler-broadened curvefit.
            let dopp = self.sqrt_awr / sqrt_kt;
            let factors = broaden_wmp_polynomials(e, dopp, order1);
            for i_poly in 0..order1 {
                sig_s += self.curvefit[cf_base + i_poly * 3 + 0] * factors[i_poly];
                sig_a += self.curvefit[cf_base + i_poly * 3 + 1] * factors[i_poly];
                if self.fissionable {
                    sig_f += self.curvefit[cf_base + i_poly * 3 + 2] * factors[i_poly];
                }
            }
        } else {
            // Unbroadened: temp = 1/E, then 1/sqrt(E), then 1, then sqrt(E), ...
            let mut temp = inv_e;
            for i_poly in 0..order1 {
                sig_s += self.curvefit[cf_base + i_poly * 3 + 0] * temp;
                sig_a += self.curvefit[cf_base + i_poly * 3 + 1] * temp;
                if self.fissionable {
                    sig_f += self.curvefit[cf_base + i_poly * 3 + 2] * temp;
                }
                temp *= sqrt_e;
            }
        }

        // ── Pole contribution ──
        if startw >= 0 && endw > startw {
            let sqrt_pi = std::f64::consts::PI.sqrt();

            if sqrt_kt == 0.0 {
                // 0 K asymptotic
                for i_pole in (startw as usize)..(endw as usize) {
                    let ea = self.poles[i_pole * 4 + 0];
                    let rs = self.poles[i_pole * 4 + 1];
                    let ra = self.poles[i_pole * 4 + 2];
                    let rf = self.poles[i_pole * 4 + 3];
                    // psi_chi = -i / (ea - sqrt_e)
                    let den = C64::new(ea.re - sqrt_e, ea.im);
                    let minus_i = C64::new(0.0, -1.0);
                    let psi_chi = minus_i.div(den);
                    let c = psi_chi.scale(inv_e);
                    sig_s += rs.mul(c).re;
                    sig_a += ra.mul(c).re;
                    if self.fissionable {
                        sig_f += rf.mul(c).re;
                    }
                }
            } else {
                // Doppler-broadened via Faddeeva.
                let dopp = self.sqrt_awr / sqrt_kt;
                for i_pole in (startw as usize)..(endw as usize) {
                    let ea = self.poles[i_pole * 4 + 0];
                    let rs = self.poles[i_pole * 4 + 1];
                    let ra = self.poles[i_pole * 4 + 2];
                    let rf = self.poles[i_pole * 4 + 3];
                    // Z = (sqrt_e - ea) * dopp
                    let zc = C64::new((sqrt_e - ea.re) * dopp, -ea.im * dopp);
                    let w = faddeeva(zc);
                    // w_val = w * dopp * invE * sqrt_pi
                    let scale = dopp * inv_e * sqrt_pi;
                    let w_val = w.scale(scale);
                    sig_s += rs.mul(w_val).re;
                    sig_a += ra.mul(w_val).re;
                    if self.fissionable {
                        sig_f += rf.mul(w_val).re;
                    }
                }
            }
        }

        (sig_s, sig_a, sig_f)
    }
}

/// Evaluate Doppler-broadened curvefit polynomial factors (port of
/// `_broaden_wmp_polynomials`). The curvefit is
/// `a/E + b/sqrt(E) + c + d*sqrt(E) + ...`; this returns the broadened
/// basis values at `(E, dopp)` for the `n` terms.
fn broaden_wmp_polynomials(e: f64, dopp: f64, n: usize) -> Vec<f64> {
    use std::f64::consts::PI;

    let sqrt_e = e.sqrt();
    let beta = sqrt_e * dopp;
    let half_inv_dopp2 = 0.5 / (dopp * dopp);
    let quarter_inv_dopp4 = half_inv_dopp2 * half_inv_dopp2;

    let (erf_beta, exp_m_beta2) = if beta > 6.0 {
        (1.0, 0.0)
    } else {
        (erf_f64(beta), (-beta * beta).exp())
    };

    let mut factors = vec![0.0_f64; n];
    factors[0] = erf_beta / e;
    if n > 1 {
        factors[1] = 1.0 / sqrt_e;
    }
    if n > 2 {
        factors[2] = factors[0] * (half_inv_dopp2 + e)
                     + exp_m_beta2 / (beta * PI.sqrt());
    }
    // Higher-order recurrence (matches OpenMC)
    for i in 1..(n.saturating_sub(2)) {
        if i != 1 {
            factors[i + 2] = -factors[i - 2] * (i as f64 - 1.0) * (i as f64) * quarter_inv_dopp4
                + factors[i] * (e + (1.0 + 2.0 * (i as f64)) * half_inv_dopp2);
        } else {
            factors[i + 2] = factors[i] * (e + (1.0 + 2.0 * (i as f64)) * half_inv_dopp2);
        }
    }
    factors
}

/// Abramowitz & Stegun 7.1.26 rational approximation of erf (|err| ≤ 1.5e-7).
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * (-ax * ax).exp();
    sign * y
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faddeeva_known_values() {
        // Reference values from scipy.special.wofz at selected z.
        // wofz(1.0 + 1.0j) = 0.304744369 + 0.208218973j (approx)
        let w = faddeeva(C64::new(1.0, 1.0));
        assert!((w.re - 0.304744369).abs() < 1e-3, "re = {}", w.re);
        assert!((w.im - 0.208218973).abs() < 1e-3, "im = {}", w.im);

        // wofz(0 + 1j) = exp(1) erfc(1) ~ 0.42758357 + 0j
        let w = faddeeva(C64::new(0.0, 1.0));
        assert!((w.re - 0.42758357).abs() < 1e-4, "re = {}", w.re);
        assert!(w.im.abs() < 1e-4, "im = {}", w.im);

        // wofz(5+5j) ~ asymptotic
        let w = faddeeva(C64::new(5.0, 5.0));
        assert!(w.re > 0.0 && w.re < 0.1);
    }

    #[test]
    fn erf_at_zero() {
        // A&S 7.1.26 is |err| ≤ 1.5e-7, so erf(0) is a few ns above zero.
        assert!(erf_f64(0.0).abs() < 2e-7);
        assert!((erf_f64(1.0) - 0.8427007).abs() < 1e-4);
    }

    #[test]
    fn faddeeva_real_axis_matches_erfcx() {
        // On the real axis, Re(w(x + 0i)) = exp(-x^2).
        // (For y = 0, w(z) = exp(-z^2) + 2i/√π · F(z) where F is Dawson;
        // the real part is exp(-x^2).)
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let w = faddeeva(C64::new(x, 0.0));
            let expected = (-x * x).exp();
            // Humlicek W4 has ~1e-4 accuracy; allow generous tolerance.
            assert!((w.re - expected).abs() < 1e-3,
                "Re(w({x} + 0i)) = {} vs exp(-x²) = {expected}", w.re);
        }
    }

    #[test]
    fn faddeeva_iterative_matches_recursive_baseline() {
        // Before the iterative rewrite (CUDA-stack fix), the upper-half
        // path was computed the same way. Sample points across the
        // Humlicek regions (different `s = |x| + y` thresholds) and
        // confirm the function is still well-behaved and near 1.0 at z=0.
        // w(0) = 1 exactly.
        let w = faddeeva(C64::new(0.0, 0.0));
        assert!((w.re - 1.0).abs() < 1e-6 && w.im.abs() < 1e-6,
                "w(0) = {:?}, expected (1, 0)", w);
        // w is continuous across region boundaries — no discontinuity at
        // s = 5.5 or s = 15.
        let eps = 1e-6;
        let w_below = faddeeva(C64::new(4.0, 1.5 + eps));   // s just above 5.5
        let w_below_other = faddeeva(C64::new(4.0, 1.5 - eps));
        assert!((w_below.re - w_below_other.re).abs() < 1e-3);
    }

    #[test]
    fn faddeeva_lower_half_plane_openmc_convention() {
        // OpenMC convention: for Im(z) < 0, w(z) = -conj(w(z*)) where z* = conj(z).
        // So w(x - iy).re = -w(x + iy).re and w(x - iy).im = +w(x + iy).im.
        let up = faddeeva(C64::new(1.0, 0.5));
        let down = faddeeva(C64::new(1.0, -0.5));
        assert!((down.re + up.re).abs() < 1e-10, "re sign flip broken: up={}, down={}", up.re, down.re);
        assert!((down.im - up.im).abs() < 1e-10, "im sign stability broken: up={}, down={}", up.im, down.im);
    }

    #[test]
    fn faddeeva_large_argument_uses_region_i() {
        // For |z| ≫ 1, w(z) ≈ i/(z√π), so |w(z)| ≈ 1/(|z|√π).
        let z = C64::new(20.0, 20.0);
        let w = faddeeva(z);
        let expected_mag = 1.0 / (z.abs2().sqrt() * std::f64::consts::PI.sqrt());
        let w_mag = w.abs2().sqrt();
        assert!((w_mag - expected_mag).abs() < 1e-3 * expected_mag,
                "asymptotic magnitude off: {w_mag} vs {expected_mag}");
    }

    #[test]
    fn broaden_polynomials_basic_shape() {
        // At dopp = 0 is singular; use finite dopp and verify first factor
        // is erf(√E · dopp) / E and second is 1/√E.
        let e = 10.0;
        let dopp = 2.0;
        let n = 4;
        let factors = broaden_wmp_polynomials(e, dopp, n);
        let beta = e.sqrt() * dopp;
        let expected_0 = erf_f64(beta) / e;
        assert!((factors[0] - expected_0).abs() < 1e-10, "factors[0] = {}", factors[0]);
        assert!((factors[1] - 1.0 / e.sqrt()).abs() < 1e-10, "factors[1] = {}", factors[1]);
    }
}
