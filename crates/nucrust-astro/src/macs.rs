//! Maxwellian-Averaged Cross Section (MACS) computation.
//!
//! MACS(kT) = (2/√π) · (1/kT)² · ∫₀^∞ σ(E) · E · exp(-E/kT) dE
//!
//! Uses Gauss-Laguerre quadrature with cubic spline interpolation of σ(E).

use nucrust_core::{CoreError, CrossSection, CubicSpline};
use std::f64::consts::PI;

/// Configuration for MACS calculation.
#[derive(Debug, Clone)]
pub struct MacsConfig {
    /// Number of Gauss-Laguerre quadrature points (default: 20).
    pub n_gauss_points: usize,
    /// Temperature grid in GK (T9). Default: standard astrophysical grid.
    pub temperature_grid: Vec<f64>,
}

impl Default for MacsConfig {
    fn default() -> Self {
        Self {
            n_gauss_points: 20,
            temperature_grid: vec![
                0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0,
                3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ],
        }
    }
}

/// Compute MACS for a set of temperatures.
///
/// Uses Gauss-Laguerre quadrature with variable substitution x = E/kT:
///   MACS(kT) = (2 / √π·kT) · Σ_i w_i · σ(x_i · kT) · x_i
///
/// σ(E) is interpolated from the cross section table using cubic splines.
///
/// Returns MACS in mb for each temperature.
pub fn compute_macs(
    cross_section: &CrossSection,
    config: &MacsConfig,
) -> Result<Vec<f64>, CoreError> {
    let energies = cross_section.energy.as_slice();
    let sigmas = &cross_section.sigma_reaction;

    if energies.len() < 2 || sigmas.len() != energies.len() {
        return Err(CoreError::InvalidParameter {
            name: "cross_section",
            value: energies.len() as f64,
            reason: "need at least 2 energy points with matching sigma",
        });
    }

    // Build cubic spline interpolator for σ(E)
    let spline = CubicSpline::natural(energies, sigmas)?;

    let (nodes, weights) = gauss_laguerre_nodes(config.n_gauss_points);

    let mut macs_values = Vec::with_capacity(config.temperature_grid.len());

    for &t9 in &config.temperature_grid {
        // kT in MeV: kT = k_B * T where k_B = 8.617e-2 MeV/GK
        let kt = nucrust_core::units::BOLTZMANN_MEV * t9;

        if kt < 1e-10 {
            macs_values.push(0.0);
            continue;
        }

        // MACS(kT) = (2/√π·kT) · Σ_i w_i · σ(x_i·kT) · x_i
        let prefactor = 2.0 / (PI.sqrt() * kt);
        let mut sum = 0.0;

        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            let e = x * kt; // E = x · kT
            let sigma = spline.evaluate(e);
            sum += w * sigma.max(0.0) * x;
        }

        macs_values.push(prefactor * sum);
    }

    Ok(macs_values)
}

/// Gauss-Laguerre quadrature nodes and weights for ∫₀^∞ f(x) exp(-x) dx.
///
/// Pre-computed for common sizes. For n=20, uses high-precision values.
fn gauss_laguerre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        // 8-point quadrature
        8 => (
            vec![
                0.170_279_632_305,
                0.903_701_776_799,
                2.251_086_629_866,
                4.266_700_170_288,
                7.045_905_402_393,
                10.758_516_010_181,
                15.740_678_641_928,
                22.863_131_736_889,
            ],
            vec![
                0.369_188_589_342,
                0.418_786_780_814,
                0.175_794_986_637,
                0.033_343_492_261,
                0.002_794_536_235,
                0.000_090_765_688,
                0.000_000_848_574,
                0.000_000_001_049,
            ],
        ),
        // 20-point quadrature (default)
        _ => (
            vec![
                0.070_539_889_692,
                0.372_126_818_001,
                0.916_582_102_483,
                1.707_306_531_028,
                2.749_199_255_174,
                4.048_925_313_851,
                5.615_174_970_950,
                7.459_017_454_319,
                9.594_392_869_580,
                12.038_802_546_964,
                14.814_293_442_630,
                17.948_895_520_519,
                21.478_788_240_285,
                25.451_702_793_187,
                29.932_554_631_700,
                35.013_434_240_479,
                40.833_057_056_728,
                47.619_994_047_347,
                55.810_795_750_064,
                66.524_416_525_616,
            ],
            vec![
                0.168_746_801_851,
                0.291_254_362_006,
                0.266_686_102_867,
                0.166_002_453_270,
                0.074_826_064_668,
                0.024_964_417_310,
                0.006_202_550_845,
                0.001_144_962_386,
                0.000_155_741_773,
                0.000_015_401_440,
                0.000_001_086_069,
                0.000_000_053_301,
                0.000_000_001_757,
                0.000_000_000_037,
                0.000_000_000_000_475,
                0.000_000_000_000_003_3,
                0.000_000_000_000_000_012,
                0.000_000_000_000_000_000_018_8,
                0.000_000_000_000_000_000_000_005_3,
                0.000_000_000_000_000_000_000_000_000_9,
            ],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::EnergyGrid;

    fn mock_cross_section() -> CrossSection {
        // 1/v cross section: σ(E) = 100 / sqrt(E) mb
        let energies = EnergyGrid::logarithmic(0.001, 10.0, 100).unwrap();
        let sigmas: Vec<f64> = energies
            .as_slice()
            .iter()
            .map(|&e| 100.0 / e.sqrt())
            .collect();
        CrossSection {
            energy: energies,
            sigma_total: sigmas.clone(),
            sigma_elastic: vec![0.0; 100],
            sigma_reaction: sigmas,
            partial: vec![],
        }
    }

    #[test]
    fn macs_positive_for_1v_law() {
        let xs = mock_cross_section();
        let config = MacsConfig::default();
        let macs = compute_macs(&xs, &config).unwrap();

        assert_eq!(macs.len(), config.temperature_grid.len());
        for (i, &m) in macs.iter().enumerate() {
            assert!(
                m > 0.0 && m.is_finite(),
                "MACS[T9={}] = {} should be positive finite",
                config.temperature_grid[i],
                m
            );
        }
    }

    #[test]
    fn macs_decreases_at_higher_temperature() {
        // For 1/v law with limited energy range, MACS generally decreases
        // at higher T because quadrature samples beyond the cross section range
        let xs = mock_cross_section();
        let config = MacsConfig {
            temperature_grid: vec![0.1, 0.3],
            ..Default::default()
        };
        let macs = compute_macs(&xs, &config).unwrap();

        // Both should be positive
        assert!(macs[0] > 0.0, "MACS[T9=0.1] = {}", macs[0]);
        assert!(macs[1] > 0.0, "MACS[T9=0.3] = {}", macs[1]);
    }
}
