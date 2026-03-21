//! Gamma-ray strength function (GSF) model implementations.
//!
//! Models: Standard Lorentzian (SLO), Enhanced Generalized Lorentzian (EGLO).

use nucrust_core::traits::{GammaStrength, Multipole};
use nucrust_core::Nuclide;

use std::f64::consts::PI;

// ============================================================================
// Standard Lorentzian (SLO)
// ============================================================================

/// Standard Lorentzian (Brink-Axel) gamma-ray strength function.
///
/// f_{E1}(E_gamma) = (1 / 3*pi^2) * sigma_GDR * Gamma_GDR * E_gamma * Gamma_GDR
///                    / ((E_gamma^2 - E_GDR^2)^2 + E_gamma^2 * Gamma_GDR^2)
#[derive(Debug, Clone)]
pub struct StandardLorentzian {
    /// GDR peak energy (MeV).
    pub e_gdr: f64,
    /// GDR width (MeV).
    pub gamma_gdr: f64,
    /// GDR peak cross section (mb).
    pub sigma_gdr: f64,
    /// M1 scissors mode parameters (optional).
    pub m1_params: Option<M1ScissorsParams>,
}

/// M1 scissors mode parameters.
#[derive(Debug, Clone)]
pub struct M1ScissorsParams {
    pub e_sc: f64,
    pub gamma_sc: f64,
    pub sigma_sc: f64,
}

impl StandardLorentzian {
    /// Evaluate Lorentzian: sigma * Gamma * E_gamma * Gamma / ((E^2 - E0^2)^2 + E^2 * Gamma^2)
    fn lorentzian(e_gamma: f64, e0: f64, gamma: f64, sigma: f64) -> f64 {
        let e2 = e_gamma * e_gamma;
        let e02 = e0 * e0;
        let g2 = gamma * gamma;
        let denom = (e2 - e02) * (e2 - e02) + e2 * g2;
        if denom == 0.0 {
            return 0.0;
        }
        sigma * gamma * e_gamma * gamma / denom / (3.0 * PI * PI)
    }
}

impl GammaStrength for StandardLorentzian {
    fn strength(&self, _nuclide: &Nuclide, e_gamma: f64, multipole: Multipole) -> f64 {
        if e_gamma <= 0.0 {
            return 0.0;
        }
        match multipole {
            Multipole::E1 => Self::lorentzian(e_gamma, self.e_gdr, self.gamma_gdr, self.sigma_gdr),
            Multipole::M1 => {
                if let Some(ref m1) = self.m1_params {
                    Self::lorentzian(e_gamma, m1.e_sc, m1.gamma_sc, m1.sigma_sc)
                } else {
                    // Default M1: simple single-particle estimate
                    // f_M1 ≈ 1.58e-9 * A^{0.47} (constant approximation)
                    1.0e-9
                }
            }
            Multipole::E2 => {
                // E2: typically small, use single-particle estimate
                // f_E2 ≈ 5.2e-8 * A^{2/3} / E_GDR^2
                5.2e-8 / (self.e_gdr * self.e_gdr)
            }
            _ => 0.0,
        }
    }

    fn name(&self) -> &str {
        "Standard Lorentzian (SLO)"
    }
}

// ============================================================================
// Enhanced Generalized Lorentzian (EGLO)
// ============================================================================

/// Enhanced Generalized Lorentzian (Kopecky-Uhl) GSF.
///
/// Temperature-dependent width: Gamma_k(E_gamma, T) = Gamma_GDR / E_GDR^2 * (E_gamma^2 + 4*pi^2*T^2)
///
/// f_{E1}(E_gamma) = kappa * [E_gamma * Gamma_k / ((E^2-E0^2)^2 + E^2*Gamma_k^2) + 0.7*Gamma_k(0,T)/(E0^3)]
#[derive(Debug, Clone)]
pub struct EnhancedGeneralizedLorentzian {
    /// GDR peak energy (MeV).
    pub e_gdr: f64,
    /// GDR width (MeV).
    pub gamma_gdr: f64,
    /// GDR peak cross section (mb).
    pub sigma_gdr: f64,
    /// Nuclear temperature (MeV).
    pub temperature: f64,
}

impl EnhancedGeneralizedLorentzian {
    /// Temperature-dependent width.
    fn gamma_k(&self, e_gamma: f64) -> f64 {
        let e02 = self.e_gdr * self.e_gdr;
        self.gamma_gdr / e02
            * (e_gamma * e_gamma + 4.0 * PI * PI * self.temperature * self.temperature)
    }
}

impl GammaStrength for EnhancedGeneralizedLorentzian {
    fn strength(&self, _nuclide: &Nuclide, e_gamma: f64, multipole: Multipole) -> f64 {
        if e_gamma <= 0.0 {
            return 0.0;
        }
        match multipole {
            Multipole::E1 => {
                let kappa = 1.0 / (3.0 * PI * PI);
                let gamma_k = self.gamma_k(e_gamma);
                let gamma_k0 = self.gamma_k(0.0);
                let e02 = self.e_gdr * self.e_gdr;
                let e2 = e_gamma * e_gamma;

                let lorentz = self.sigma_gdr * self.gamma_gdr * e_gamma * gamma_k
                    / ((e2 - e02) * (e2 - e02) + e2 * gamma_k * gamma_k);

                let zero_limit =
                    0.7 * self.sigma_gdr * self.gamma_gdr * gamma_k0 / (self.e_gdr * e02);

                kappa * (lorentz + zero_limit)
            }
            Multipole::M1 => 1.0e-9, // Default single-particle estimate
            Multipole::E2 => 5.2e-8 / (self.e_gdr * self.e_gdr),
            _ => 0.0,
        }
    }

    fn name(&self) -> &str {
        "Enhanced Generalized Lorentzian (EGLO)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe56() -> Nuclide {
        Nuclide::new(26, 56).unwrap()
    }

    #[test]
    fn slo_e1_positive() {
        let slo = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let f = slo.strength(&fe56(), 8.0, Multipole::E1);
        assert!(f > 0.0, "f_E1 = {}", f);
    }

    #[test]
    fn slo_peaks_near_gdr() {
        let slo = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let nuclide = fe56();
        let f_peak = slo.strength(&nuclide, 16.36, Multipole::E1);
        let f_low = slo.strength(&nuclide, 5.0, Multipole::E1);
        let f_high = slo.strength(&nuclide, 25.0, Multipole::E1);
        assert!(f_peak > f_low);
        assert!(f_peak > f_high);
    }

    #[test]
    fn eglo_positive() {
        let eglo = EnhancedGeneralizedLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            temperature: 0.5,
        };
        let f = eglo.strength(&fe56(), 8.0, Multipole::E1);
        assert!(f > 0.0, "f_E1 EGLO = {}", f);
    }

    #[test]
    fn eglo_nonzero_at_zero_energy_limit() {
        // EGLO should have non-zero strength near zero energy (unlike SLO)
        let eglo = EnhancedGeneralizedLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            temperature: 0.5,
        };
        let f = eglo.strength(&fe56(), 0.1, Multipole::E1);
        assert!(f > 0.0);
    }
}
