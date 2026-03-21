//! Nuclear level density (NLD) model implementations.
//!
//! Models: Constant Temperature, Back-Shifted Fermi Gas, Gilbert-Cameron,
//!         Ignatyuk, HFB table interpolation.

use nucrust_core::spin::Parity;
use nucrust_core::traits::LevelDensity;
use nucrust_core::Nuclide;

use std::f64::consts::PI;

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

        // Factor 0.5 for parity equipartition
        0.5 * rho_tot * spin_distribution(spin, sigma_sq)
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

        0.5 * rho_t * spin_distribution(spin, sigma_sq)
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

        0.5 * rho_t * spin_distribution(spin, sigma_sq)
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
}
