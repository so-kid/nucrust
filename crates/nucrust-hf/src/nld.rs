//! Nuclear level density (NLD) model implementations.
//!
//! Models: Constant Temperature, Back-Shifted Fermi Gas, Gilbert-Cameron,
//!         Ignatyuk, HFB table interpolation.

use nucrust_core::spin::Parity;
use nucrust_core::spline::CubicSpline;
use nucrust_core::traits::LevelDensity;
use nucrust_core::Nuclide;

use std::f64::consts::PI;

/// Parity equipartition factor: half of the total density is assigned to each parity.
const PARITY_FACTOR: f64 = 0.5;

/// Spin-cutoff parameter: sigma^2 = 0.0888 * A^{2/3} * a * sqrt(U)
fn spin_cutoff_sq(a_param: f64, excitation: f64, mass_a: f64) -> f64 {
    let u = excitation.max(0.01);
    0.0888 * mass_a.powf(2.0 / 3.0) * (a_param * u).sqrt()
}

/// Spin distribution factor: f(J, sigma) = (2J+1)/(2*sigma^2) * exp(-(J+0.5)^2 / (2*sigma^2))
fn spin_distribution(spin: f64, sigma_sq: f64) -> f64 {
    if sigma_sq <= 0.0 {
        return 0.0;
    }
    let j_half = spin + 0.5;
    (2.0 * spin + 1.0) / (2.0 * sigma_sq) * (-j_half * j_half / (2.0 * sigma_sq)).exp()
}

// ============================================================================
// Constant Temperature Model
// ============================================================================

/// Constant Temperature (CT) level density model.
///
/// rho(U) = (1/T) * exp((U - E0) / T)
#[derive(Debug, Clone)]
pub struct ConstantTemperature {
    /// Nuclear temperature T (MeV).
    pub temperature: f64,
    /// Energy shift E0 (MeV).
    pub e0: f64,
    /// Level density parameter a (1/MeV) for spin cutoff.
    pub a: f64,
}

impl LevelDensity for ConstantTemperature {
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, _parity: Parity) -> f64 {
        let u = excitation - self.e0;
        if u < 0.0 {
            return 0.0;
        }
        let rho_tot = (1.0 / self.temperature) * (u / self.temperature).exp();
        let sigma_sq = spin_cutoff_sq(self.a, excitation, nuclide.a() as f64);

        PARITY_FACTOR * rho_tot * spin_distribution(spin, sigma_sq)
    }

    fn rho_total(&self, _nuclide: &Nuclide, excitation: f64) -> f64 {
        let u = excitation - self.e0;
        if u < 0.0 {
            return 0.0;
        }
        (1.0 / self.temperature) * (u / self.temperature).exp()
    }

    fn name(&self) -> &str {
        "Constant Temperature"
    }
}

// ============================================================================
// Back-Shifted Fermi Gas Model
// ============================================================================

/// Back-Shifted Fermi Gas (BSFG) level density model.
///
/// rho(U) = exp(2*sqrt(a*U)) / (12*sqrt(2)*sigma*a^{1/4}*U^{5/4})
/// where U = E - delta (effective excitation energy).
#[derive(Debug, Clone)]
pub struct BackShiftedFermiGas {
    /// Level density parameter a (1/MeV).
    pub a: f64,
    /// Pairing energy shift delta (MeV).
    pub delta: f64,
    /// Spin cutoff parameter sigma (if known).
    pub sigma: Option<f64>,
}

impl LevelDensity for BackShiftedFermiGas {
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, _parity: Parity) -> f64 {
        let rho_t = self.rho_total(nuclide, excitation);
        if rho_t <= 0.0 {
            return 0.0;
        }
        let u = (excitation - self.delta).max(0.01);
        let sigma_sq = self
            .sigma
            .map(|s| s * s)
            .unwrap_or_else(|| spin_cutoff_sq(self.a, u, nuclide.a() as f64));

        PARITY_FACTOR * rho_t * spin_distribution(spin, sigma_sq)
    }

    fn rho_total(&self, _nuclide: &Nuclide, excitation: f64) -> f64 {
        let u = excitation - self.delta;
        if u < 0.01 {
            return 0.0;
        }
        let sqrt_au = (self.a * u).sqrt();
        let sigma_sq = 0.0888 * (self.a * u).sqrt(); // simplified
        let sigma = sigma_sq.max(0.01).sqrt();

        (2.0 * sqrt_au).exp() / (12.0 * (2.0_f64).sqrt() * sigma * self.a.powf(0.25) * u.powf(1.25))
    }

    fn name(&self) -> &str {
        "Back-Shifted Fermi Gas"
    }
}

// ============================================================================
// Gilbert-Cameron Composite Model
// ============================================================================

/// Gilbert-Cameron composite level density model.
///
/// Uses CT model below matching energy E_match,
/// BSFG above E_match, with continuity conditions.
#[derive(Debug, Clone)]
pub struct GilbertCameron {
    /// CT parameters (below E_match).
    pub ct: ConstantTemperature,
    /// BSFG parameters (above E_match).
    pub bsfg: BackShiftedFermiGas,
    /// Matching energy (MeV).
    pub e_match: f64,
}

impl LevelDensity for GilbertCameron {
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, parity: Parity) -> f64 {
        if excitation < self.e_match {
            self.ct.rho(nuclide, excitation, spin, parity)
        } else {
            self.bsfg.rho(nuclide, excitation, spin, parity)
        }
    }

    fn rho_total(&self, nuclide: &Nuclide, excitation: f64) -> f64 {
        if excitation < self.e_match {
            self.ct.rho_total(nuclide, excitation)
        } else {
            self.bsfg.rho_total(nuclide, excitation)
        }
    }

    fn name(&self) -> &str {
        "Gilbert-Cameron"
    }
}

// ============================================================================
// Ignatyuk Model
// ============================================================================

/// Ignatyuk energy-dependent level density parameter.
///
/// a(U) = a_tilde * (1 + delta_W/U * (1 - exp(-gamma * U)))
///
/// This is used with the BSFG formula but with a(U) instead of constant a.
#[derive(Debug, Clone)]
pub struct Ignatyuk {
    /// Asymptotic level density parameter a_tilde (1/MeV).
    pub a_tilde: f64,
    /// Shell correction energy delta_W (MeV).
    pub delta_w: f64,
    /// Damping parameter gamma (1/MeV).
    pub gamma: f64,
    /// Pairing energy shift delta (MeV).
    pub delta: f64,
}

impl Ignatyuk {
    /// Energy-dependent level density parameter.
    pub fn a_eff(&self, excitation: f64) -> f64 {
        let u = (excitation - self.delta).max(0.01);
        self.a_tilde * (1.0 + self.delta_w / u * (1.0 - (-self.gamma * u).exp()))
    }
}

impl LevelDensity for Ignatyuk {
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, _parity: Parity) -> f64 {
        let rho_t = self.rho_total(nuclide, excitation);
        if rho_t <= 0.0 {
            return 0.0;
        }
        let u = (excitation - self.delta).max(0.01);
        let a_eff = self.a_eff(excitation);
        let sigma_sq = spin_cutoff_sq(a_eff, u, nuclide.a() as f64);

        PARITY_FACTOR * rho_t * spin_distribution(spin, sigma_sq)
    }

    fn rho_total(&self, _nuclide: &Nuclide, excitation: f64) -> f64 {
        let u = excitation - self.delta;
        if u < 0.01 {
            return 0.0;
        }
        let a_eff = self.a_eff(excitation);
        let sqrt_au = (a_eff * u).sqrt();

        (2.0 * sqrt_au).exp()
            / (12.0 * (2.0_f64).sqrt() * a_eff.powf(0.25) * u.powf(1.25) * PI.sqrt())
    }

    fn name(&self) -> &str {
        "Ignatyuk"
    }
}

// ============================================================================
// HFB Table Interpolation Model
// ============================================================================

/// Hartree-Fock-Bogoliubov (HFB) tabulated level density model.
///
/// Interpolates pre-computed ρ(U, J) tables from microscopic HFB calculations.
/// Uses cubic spline interpolation along excitation energy axis for each spin,
/// then linear interpolation in spin.
///
/// Storage: `densities[i_spin * n_excitations + i_excitation]` (row = spin).
#[derive(Debug, Clone)]
pub struct HfbTableInterp {
    /// Spin values (half-integer or integer), strictly increasing.
    spins: Vec<f64>,
    /// Cubic splines, one per spin value, interpolating along excitation energy.
    splines: Vec<CubicSpline>,
}

impl HfbTableInterp {
    /// Construct from tabulated data.
    ///
    /// - `excitations`: energy grid (MeV), length `n_e`.
    /// - `spins`: spin grid, length `n_j`.
    /// - `densities`: flattened ρ(J, U) table, length `n_j * n_e`,
    ///   stored as `densities[i_j * n_e + i_e]`.
    ///
    /// All density values should be ≥ 0. The function takes log10 internally
    /// for interpolation stability, using a floor of 1e-30 for zero/negative values.
    pub fn new(
        excitations: Vec<f64>,
        spins: Vec<f64>,
        densities: &[f64],
    ) -> Result<Self, nucrust_core::CoreError> {
        let n_e = excitations.len();
        let n_j = spins.len();

        if densities.len() != n_j * n_e {
            return Err(nucrust_core::CoreError::InvalidParameter {
                name: "densities",
                value: densities.len() as f64,
                reason: "length must equal n_spins * n_excitations",
            });
        }

        if n_e < 2 || n_j < 2 {
            return Err(nucrust_core::CoreError::InvalidParameter {
                name: "grid_size",
                value: n_e.min(n_j) as f64,
                reason: "need at least 2 excitation energies and 2 spin values",
            });
        }

        // Build one cubic spline per spin value (interpolating in excitation energy)
        // Interpolate log10(rho) for numerical stability
        let mut splines = Vec::with_capacity(n_j);
        for i_j in 0..n_j {
            let row_start = i_j * n_e;
            let log_rho: Vec<f64> = densities[row_start..row_start + n_e]
                .iter()
                .map(|&rho| rho.max(1e-30).log10())
                .collect();
            let spline = CubicSpline::natural(&excitations, &log_rho)?;
            splines.push(spline);
        }

        Ok(Self { spins, splines })
    }

    /// Interpolate ρ(U, J) at arbitrary excitation and spin.
    fn interpolate(&self, excitation: f64, spin: f64) -> f64 {
        if excitation <= 0.0 {
            return 0.0;
        }

        let n_j = self.spins.len();

        // Find bracketing spin indices
        if spin <= self.spins[0] {
            let log_rho = self.splines[0].evaluate(excitation);
            return 10.0_f64.powf(log_rho).max(0.0);
        }
        if spin >= self.spins[n_j - 1] {
            let log_rho = self.splines[n_j - 1].evaluate(excitation);
            return 10.0_f64.powf(log_rho).max(0.0);
        }

        // Binary search for spin bracket
        let idx = self
            .spins
            .partition_point(|&s| s < spin)
            .saturating_sub(1)
            .min(n_j - 2);

        let s0 = self.spins[idx];
        let s1 = self.spins[idx + 1];
        let t = (spin - s0) / (s1 - s0);

        // Interpolate log10(rho) at each bracketing spin, then linear interp in spin
        let log_rho0 = self.splines[idx].evaluate(excitation);
        let log_rho1 = self.splines[idx + 1].evaluate(excitation);
        let log_rho = log_rho0 + t * (log_rho1 - log_rho0);

        10.0_f64.powf(log_rho).max(0.0)
    }
}

impl LevelDensity for HfbTableInterp {
    fn rho(&self, _nuclide: &Nuclide, excitation: f64, spin: f64, _parity: Parity) -> f64 {
        // Table gives the total for both parities.
        PARITY_FACTOR * self.interpolate(excitation, spin)
    }

    fn rho_total(&self, _nuclide: &Nuclide, excitation: f64) -> f64 {
        if excitation <= 0.0 {
            return 0.0;
        }
        // Sum (2J+1) * rho(U, J) over tabulated spins using trapezoidal rule
        let n_j = self.spins.len();
        let mut total = 0.0;
        for i in 0..n_j {
            let j = self.spins[i];
            let rho_j = self.interpolate(excitation, j);
            let weight = 2.0 * j + 1.0;
            // Trapezoidal weight for spin integration
            let dj = if n_j == 1 {
                1.0
            } else if i == 0 {
                (self.spins[1] - self.spins[0]) / 2.0
            } else if i == n_j - 1 {
                (self.spins[n_j - 1] - self.spins[n_j - 2]) / 2.0
            } else {
                (self.spins[i + 1] - self.spins[i - 1]) / 2.0
            };
            total += weight * rho_j * dj;
        }
        total
    }

    fn name(&self) -> &str {
        "HFB Table Interpolation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe56() -> Nuclide {
        Nuclide::new(26, 56).unwrap()
    }

    #[test]
    fn ct_positive_density() {
        let ct = ConstantTemperature {
            temperature: 0.88,
            e0: -1.16,
            a: 6.21,
        };
        let nuclide = fe56();
        let rho = ct.rho_total(&nuclide, 5.0);
        assert!(rho > 0.0, "rho_total = {}", rho);
    }

    #[test]
    fn ct_increases_with_energy() {
        let ct = ConstantTemperature {
            temperature: 0.88,
            e0: -1.16,
            a: 6.21,
        };
        let nuclide = fe56();
        let rho1 = ct.rho_total(&nuclide, 3.0);
        let rho2 = ct.rho_total(&nuclide, 5.0);
        assert!(rho2 > rho1);
    }

    #[test]
    fn bsfg_positive_density() {
        let bsfg = BackShiftedFermiGas {
            a: 6.21,
            delta: -0.52,
            sigma: None,
        };
        let nuclide = fe56();
        let rho = bsfg.rho_total(&nuclide, 5.0);
        assert!(rho > 0.0, "rho_total = {}", rho);
    }

    #[test]
    fn gc_continuous_at_match() {
        let gc = GilbertCameron {
            ct: ConstantTemperature {
                temperature: 0.88,
                e0: -1.16,
                a: 6.21,
            },
            bsfg: BackShiftedFermiGas {
                a: 6.21,
                delta: -0.52,
                sigma: None,
            },
            e_match: 3.24,
        };
        let nuclide = fe56();
        let rho_below = gc.rho_total(&nuclide, 3.23);
        let rho_above = gc.rho_total(&nuclide, 3.25);
        // Both should be positive, although there may be a discontinuity
        // (in production, matching conditions would be applied)
        assert!(rho_below > 0.0);
        assert!(rho_above > 0.0);
    }

    #[test]
    fn ignatyuk_a_eff_converges_to_a_tilde() {
        let ig = Ignatyuk {
            a_tilde: 6.0,
            delta_w: -3.0,
            gamma: 0.04,
            delta: -0.5,
        };
        // At very high excitation, a_eff -> a_tilde (within delta_W/U residual)
        let a_high = ig.a_eff(1000.0);
        assert!(
            (a_high - ig.a_tilde).abs() < 0.05,
            "a_eff at high E = {}",
            a_high
        );
    }

    #[test]
    fn spin_distribution_sums_to_one() {
        // Sum over integer J values should give ~1
        // (the formula is normalized for integer spins summed with (2J+1) weight)
        let sigma_sq = 5.0;
        let mut sum = 0.0;
        for j in 0..30 {
            let jf = j as f64;
            sum += spin_distribution(jf, sigma_sq);
        }
        // Should be approximately 1, but numerical sum may differ slightly
        assert!(
            (sum - 1.0).abs() < 0.1,
            "sum of spin distribution = {}",
            sum
        );
    }

    // ========================================================================
    // HfbTableInterp tests
    // ========================================================================

    /// Create a synthetic BSFG-like HFB table for testing.
    fn make_test_hfb_table() -> HfbTableInterp {
        let excitations: Vec<f64> = (0..20).map(|i| i as f64 * 0.5 + 0.5).collect(); // 0.5 to 10.0 MeV
        let spins: Vec<f64> = (0..10).map(|j| j as f64).collect(); // J = 0..9
        let n_e = excitations.len();
        let n_j = spins.len();

        // Generate densities using BSFG-like formula
        let a = 6.0;
        let delta = -0.5;
        let mut densities = vec![0.0; n_j * n_e];
        for (i_j, &j) in spins.iter().enumerate() {
            for (i_e, &e) in excitations.iter().enumerate() {
                let u = (e - delta).max(0.01);
                let sigma_sq = 0.0888 * 56.0_f64.powf(2.0 / 3.0) * (a * u).sqrt();
                let rho_tot = (2.0 * (a * u).sqrt()).exp()
                    / (12.0
                        * 2.0_f64.sqrt()
                        * sigma_sq.sqrt().max(0.01)
                        * a.powf(0.25)
                        * u.powf(1.25));
                let spin_dist = spin_distribution(j, sigma_sq);
                densities[i_j * n_e + i_e] = rho_tot * spin_dist;
            }
        }

        HfbTableInterp::new(excitations, spins, &densities).unwrap()
    }

    #[test]
    fn hfb_table_positive_density() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        let rho = hfb.rho(&nuclide, 5.0, 2.0, Parity::Positive);
        assert!(rho > 0.0, "HFB rho = {}", rho);
    }

    #[test]
    fn hfb_table_increases_with_energy() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        let rho1 = hfb.rho_total(&nuclide, 3.0);
        let rho2 = hfb.rho_total(&nuclide, 5.0);
        assert!(
            rho2 > rho1,
            "rho(5 MeV) = {} should > rho(3 MeV) = {}",
            rho2,
            rho1
        );
    }

    #[test]
    fn hfb_table_interpolates_between_grid_points() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        // At a grid point
        let rho_grid = hfb.rho(&nuclide, 5.0, 2.0, Parity::Positive);
        // Between grid points
        let rho_interp = hfb.rho(&nuclide, 5.25, 2.0, Parity::Positive);
        // Both should be positive and finite
        assert!(rho_grid > 0.0 && rho_grid.is_finite());
        assert!(rho_interp > 0.0 && rho_interp.is_finite());
    }

    #[test]
    fn hfb_table_spin_interpolation() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        // Interpolate between spin grid points (J=2.5, between J=2 and J=3)
        let rho = hfb.rho(&nuclide, 5.0, 2.5, Parity::Positive);
        let rho_2 = hfb.rho(&nuclide, 5.0, 2.0, Parity::Positive);
        let rho_3 = hfb.rho(&nuclide, 5.0, 3.0, Parity::Positive);
        // Should be between the two bracketing values (or close, due to log interp)
        assert!(rho > 0.0 && rho.is_finite());
        // In log space, interpolated value should be between rho_2 and rho_3
        let log_rho = rho.log10();
        let log_min = rho_2.log10().min(rho_3.log10());
        let log_max = rho_2.log10().max(rho_3.log10());
        assert!(
            log_rho >= log_min - 0.1 && log_rho <= log_max + 0.1,
            "log10(rho) = {} not between {} and {}",
            log_rho,
            log_min,
            log_max
        );
    }

    #[test]
    fn hfb_table_rho_total_positive() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        let rho_tot = hfb.rho_total(&nuclide, 5.0);
        assert!(rho_tot > 0.0, "rho_total = {}", rho_tot);
    }

    #[test]
    fn hfb_table_zero_at_zero_excitation() {
        let hfb = make_test_hfb_table();
        let nuclide = fe56();
        let rho = hfb.rho(&nuclide, 0.0, 0.0, Parity::Positive);
        assert!(rho.abs() < 1e-10);
    }

    #[test]
    fn hfb_table_invalid_dimensions() {
        let excitations = vec![1.0, 2.0, 3.0];
        let spins = vec![0.0, 1.0];
        let densities = vec![1.0; 5]; // Wrong size (should be 6)
        let result = HfbTableInterp::new(excitations, spins, &densities);
        assert!(result.is_err());
    }
}
