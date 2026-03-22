//! Gamma-ray strength function (GSF) model implementations.
//!
//! Models: Standard Lorentzian (SLO), Enhanced Generalized Lorentzian (EGLO).

use nucrust_core::spline::CubicSpline;
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
    /// Scissors mode resonance energy (MeV).
    pub e_sc: f64,
    /// Scissors mode width (MeV).
    pub gamma_sc: f64,
    /// Scissors mode peak cross section (mb).
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

// ============================================================================
// QRPA Table Interpolation
// ============================================================================

/// Quasiparticle Random Phase Approximation (QRPA) tabulated GSF.
///
/// Interpolates pre-computed f_{E1}(E_γ) and f_{M1}(E_γ) tables from
/// microscopic QRPA calculations using cubic spline interpolation.
#[derive(Debug, Clone)]
pub struct QrpaTableInterp {
    /// Cubic spline for E1 strength function.
    spline_e1: CubicSpline,
    /// Cubic spline for M1 strength function.
    spline_m1: CubicSpline,
    /// E2 single-particle estimate (constant).
    e2_estimate: f64,
}

impl QrpaTableInterp {
    /// Construct from tabulated E1 and M1 strength data.
    ///
    /// - `energies`: photon energy grid (MeV), strictly increasing.
    /// - `strengths_e1`: f_{E1}(E_γ) at each grid point (MeV⁻³).
    /// - `strengths_m1`: f_{M1}(E_γ) at each grid point (MeV⁻³).
    ///
    /// For E2, a constant single-particle estimate is used.
    pub fn new(
        energies: &[f64],
        strengths_e1: &[f64],
        strengths_m1: &[f64],
    ) -> Result<Self, nucrust_core::CoreError> {
        if energies.len() != strengths_e1.len() || energies.len() != strengths_m1.len() {
            return Err(nucrust_core::CoreError::InvalidParameter {
                name: "strengths",
                value: energies.len() as f64,
                reason: "energies, strengths_e1, and strengths_m1 must have equal length",
            });
        }

        let spline_e1 = CubicSpline::natural(energies, strengths_e1)?;
        let spline_m1 = CubicSpline::natural(energies, strengths_m1)?;

        // Default E2 estimate: f_E2 ≈ 5.2e-8 / E_peak^2, use midpoint as rough peak
        let e_mid = energies[energies.len() / 2];
        let e2_estimate = 5.2e-8 / (e_mid * e_mid);

        Ok(Self {
            spline_e1,
            spline_m1,
            e2_estimate,
        })
    }
}

impl GammaStrength for QrpaTableInterp {
    fn strength(&self, _nuclide: &Nuclide, e_gamma: f64, multipole: Multipole) -> f64 {
        if e_gamma <= 0.0 {
            return 0.0;
        }
        match multipole {
            Multipole::E1 => self.spline_e1.evaluate(e_gamma).max(0.0),
            Multipole::M1 => self.spline_m1.evaluate(e_gamma).max(0.0),
            Multipole::E2 => self.e2_estimate,
            _ => 0.0,
        }
    }

    fn name(&self) -> &str {
        "QRPA Table Interpolation"
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

    // ========================================================================
    // QrpaTableInterp tests
    // ========================================================================

    /// Create a synthetic SLO-like QRPA table for testing.
    fn make_test_qrpa_table() -> QrpaTableInterp {
        let n = 50;
        let energies: Vec<f64> = (1..=n).map(|i| i as f64 * 0.5).collect(); // 0.5 to 25.0 MeV
        let e_gdr = 16.36;
        let gamma_gdr = 4.58;
        let sigma_gdr = 136.0;

        let strengths_e1: Vec<f64> = energies
            .iter()
            .map(|&e| {
                let e2 = e * e;
                let e02 = e_gdr * e_gdr;
                let g2 = gamma_gdr * gamma_gdr;
                let denom = (e2 - e02) * (e2 - e02) + e2 * g2;
                sigma_gdr * gamma_gdr * e * gamma_gdr / denom / (3.0 * PI * PI)
            })
            .collect();

        let strengths_m1: Vec<f64> = energies.iter().map(|_| 1.0e-9).collect();

        QrpaTableInterp::new(&energies, &strengths_e1, &strengths_m1).unwrap()
    }

    #[test]
    fn qrpa_e1_positive() {
        let qrpa = make_test_qrpa_table();
        let f = qrpa.strength(&fe56(), 8.0, Multipole::E1);
        assert!(f > 0.0, "QRPA f_E1 = {}", f);
    }

    #[test]
    fn qrpa_peaks_near_gdr() {
        let qrpa = make_test_qrpa_table();
        let nuclide = fe56();
        let f_peak = qrpa.strength(&nuclide, 16.5, Multipole::E1);
        let f_low = qrpa.strength(&nuclide, 5.0, Multipole::E1);
        let f_high = qrpa.strength(&nuclide, 24.0, Multipole::E1);
        assert!(f_peak > f_low, "peak {} > low {}", f_peak, f_low);
        assert!(f_peak > f_high, "peak {} > high {}", f_peak, f_high);
    }

    #[test]
    fn qrpa_m1_interpolation() {
        let qrpa = make_test_qrpa_table();
        let f = qrpa.strength(&fe56(), 8.0, Multipole::M1);
        assert!(
            (f - 1.0e-9).abs() < 1.0e-10,
            "M1 should be ~1e-9, got {}",
            f
        );
    }

    #[test]
    fn qrpa_e2_fallback() {
        let qrpa = make_test_qrpa_table();
        let f = qrpa.strength(&fe56(), 8.0, Multipole::E2);
        assert!(f > 0.0, "E2 fallback = {}", f);
    }

    #[test]
    fn qrpa_zero_at_zero_energy() {
        let qrpa = make_test_qrpa_table();
        let f = qrpa.strength(&fe56(), 0.0, Multipole::E1);
        assert!(f.abs() < 1e-15);
    }

    #[test]
    fn qrpa_matches_slo() {
        // QRPA table built from SLO formula should closely match SLO evaluation
        let qrpa = make_test_qrpa_table();
        let slo = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let nuclide = fe56();

        // Test at grid points
        for e in [5.0, 10.0, 15.0, 20.0] {
            let f_qrpa = qrpa.strength(&nuclide, e, Multipole::E1);
            let f_slo = slo.strength(&nuclide, e, Multipole::E1);
            let rel_err = (f_qrpa - f_slo).abs() / f_slo;
            assert!(
                rel_err < 1e-6,
                "At E={} MeV: QRPA={:.6e}, SLO={:.6e}, rel_err={:.2e}",
                e,
                f_qrpa,
                f_slo,
                rel_err
            );
        }
    }

    #[test]
    fn qrpa_invalid_lengths() {
        let energies = vec![1.0, 2.0, 3.0];
        let strengths_e1 = vec![1.0, 2.0]; // Wrong length
        let strengths_m1 = vec![1.0, 2.0, 3.0];
        let result = QrpaTableInterp::new(&energies, &strengths_e1, &strengths_m1);
        assert!(result.is_err());
    }
}
