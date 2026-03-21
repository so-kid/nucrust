//! R-matrix types and data structures.

use nucrust_core::{Nuclide, Projectile, SpinParity};
use std::path::PathBuf;

/// R-matrix parameter set defining the full problem.
#[derive(Debug, Clone)]
pub struct RMatrixParams {
    /// Reaction channels.
    pub channels: Vec<RMatrixChannel>,
    /// R-matrix levels (poles).
    pub levels: Vec<RMatrixLevel>,
    /// Boundary condition specification.
    pub boundary_condition: BoundaryCondition,
}

/// A single R-matrix channel.
#[derive(Debug, Clone)]
pub struct RMatrixChannel {
    /// Particle pair for this channel.
    pub pair: ParticlePair,
    /// Orbital angular momentum quantum number.
    pub l: u32,
    /// Channel spin.
    pub s: f64,
    /// Total angular momentum J^pi.
    pub j: SpinParity,
    /// Channel radius a (fm).
    pub radius: f64,
}

/// Particle pair (projectile + target).
#[derive(Debug, Clone)]
pub struct ParticlePair {
    /// Light particle (projectile / ejectile).
    pub light: Projectile,
    /// Heavy particle (target / residual).
    pub heavy: Nuclide,
    /// Q-value (MeV).
    pub q_value: f64,
    /// Separation energy (MeV).
    pub separation_energy: f64,
}

/// A single R-matrix level (pole).
#[derive(Debug, Clone)]
pub struct RMatrixLevel {
    /// Pole energy E_lambda (MeV).
    pub energy: f64,
    /// Reduced width amplitudes gamma_{lambda,c} (MeV^{1/2}), one per channel.
    pub reduced_widths: Vec<f64>,
}

/// Boundary condition specification.
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// Standard R-matrix with explicit boundary condition values B_c.
    Standard {
        /// Boundary condition values, one per channel.
        b: Vec<f64>,
    },
    /// Brune alternative parameterization (B_c = S_c(E_lambda)).
    Brune,
}

/// Result of an R-matrix cross section calculation.
#[derive(Debug, Clone)]
pub struct RMatrixResult {
    /// Collision matrix U_{cc'}(E).
    pub collision_matrix: nucrust_core::CollisionMatrix,
    /// Cross sections computed from the collision matrix.
    pub cross_sections: nucrust_core::CrossSection,
}

/// Experimental data for fitting.
#[derive(Debug, Clone)]
pub struct ExperimentalData {
    /// Center-of-mass energies (MeV).
    pub energies: Vec<f64>,
    /// Measured cross sections (mb).
    pub cross_sections: Vec<f64>,
    /// Measurement uncertainties (mb).
    pub errors: Vec<f64>,
}

/// Index identifying a fit parameter within `RMatrixParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamIndex {
    /// Level energy: levels[level_idx].energy
    LevelEnergy { level_idx: usize },
    /// Reduced width amplitude: levels[level_idx].reduced_widths[channel_idx]
    ReducedWidth {
        level_idx: usize,
        channel_idx: usize,
    },
    /// Channel radius: channels[channel_idx].radius
    ChannelRadius { channel_idx: usize },
}

/// Result of a Levenberg-Marquardt fit.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Optimized chi-squared value.
    pub chi_squared: f64,
    /// Reduced chi-squared (chi^2 / dof).
    pub reduced_chi_squared: f64,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the fit converged.
    pub converged: bool,
    /// Covariance matrix (n_params x n_params, row-major).
    pub covariance: Vec<f64>,
    /// Final parameter values in the same order as `free_params`.
    pub param_values: Vec<f64>,
}

/// Configuration for Levenberg-Marquardt fitting.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Initial damping factor lambda.
    pub lambda_init: f64,
    /// Factor for increasing/decreasing lambda.
    pub lambda_factor: f64,
    /// Convergence threshold on chi-squared change.
    pub convergence_chi2: f64,
    /// Convergence threshold on parameter change.
    pub convergence_params: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            lambda_init: 1e-3,
            lambda_factor: 10.0,
            convergence_chi2: 1e-8,
            convergence_params: 1e-8,
        }
    }
}

/// Result of an MCMC sampling run.
#[derive(Debug, Clone)]
pub struct McmcResult {
    /// In-memory chains \[walker\]\[sample\]\[param\] (None if streamed to file).
    pub chains: Option<Vec<Vec<Vec<f64>>>>,
    /// Log-likelihood values \[walker\]\[sample\] (None if streamed).
    pub log_likelihood: Option<Vec<Vec<f64>>>,
    /// Acceptance rate.
    pub acceptance_rate: f64,
    /// Output file path (if streamed).
    pub output_path: Option<PathBuf>,
    /// Mean of each parameter.
    pub param_means: Vec<f64>,
    /// Standard deviation of each parameter.
    pub param_stds: Vec<f64>,
}

/// Configuration for MCMC sampling.
#[derive(Debug, Clone)]
pub struct McmcConfig {
    /// Number of walkers.
    pub n_walkers: u32,
    /// Number of samples per walker.
    pub n_samples: u32,
    /// Burn-in period (samples to discard).
    pub burn_in: u32,
    /// Stretch move scale parameter.
    pub stretch_scale: f64,
    /// Optional streaming output path.
    pub stream_output: Option<PathBuf>,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            n_walkers: 32,
            n_samples: 10_000,
            burn_in: 1_000,
            stretch_scale: 2.0,
            stream_output: None,
        }
    }
}

impl RMatrixParams {
    /// Get the value of a parameter by index.
    pub fn get_param(&self, idx: &ParamIndex) -> f64 {
        match idx {
            ParamIndex::LevelEnergy { level_idx } => self.levels[*level_idx].energy,
            ParamIndex::ReducedWidth {
                level_idx,
                channel_idx,
            } => self.levels[*level_idx].reduced_widths[*channel_idx],
            ParamIndex::ChannelRadius { channel_idx } => self.channels[*channel_idx].radius,
        }
    }

    /// Set the value of a parameter by index.
    pub fn set_param(&mut self, idx: &ParamIndex, value: f64) {
        match idx {
            ParamIndex::LevelEnergy { level_idx } => self.levels[*level_idx].energy = value,
            ParamIndex::ReducedWidth {
                level_idx,
                channel_idx,
            } => self.levels[*level_idx].reduced_widths[*channel_idx] = value,
            ParamIndex::ChannelRadius { channel_idx } => self.channels[*channel_idx].radius = value,
        }
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

impl ParticlePair {
    /// Reduced mass in amu.
    pub fn reduced_mass_amu(&self) -> f64 {
        let m_light = self.light.mass_amu();
        let m_heavy = self.heavy.a() as f64; // approximate: A in amu
        nucrust_core::units::reduced_mass(m_light, m_heavy)
    }

    /// Charge product Z1 * Z2.
    pub fn charge_product(&self) -> f64 {
        self.light.z() as f64 * self.heavy.z() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::{Nuclide, Parity, Projectile, SpinParity};

    fn make_simple_params() -> RMatrixParams {
        let pair = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: 0.0,
            separation_energy: 4.946,
        };
        let channel = RMatrixChannel {
            pair,
            l: 0,
            s: 0.5,
            j: SpinParity::new(1, Parity::Positive).unwrap(),
            radius: 5.0,
        };
        RMatrixParams {
            channels: vec![channel],
            levels: vec![RMatrixLevel {
                energy: 1.0,
                reduced_widths: vec![0.5],
            }],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0] },
        }
    }

    #[test]
    fn param_get_set() {
        let mut params = make_simple_params();
        let idx = ParamIndex::LevelEnergy { level_idx: 0 };
        assert!((params.get_param(&idx) - 1.0).abs() < 1e-15);
        params.set_param(&idx, 2.0);
        assert!((params.get_param(&idx) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn reduced_mass() {
        let pair = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: 0.0,
            separation_energy: 4.946,
        };
        let mu = pair.reduced_mass_amu();
        // mu = m_n * 12 / (m_n + 12) ~ 0.926
        assert!(mu > 0.9 && mu < 1.0);
    }

    #[test]
    fn charge_product_neutron() {
        let pair = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: 0.0,
            separation_energy: 4.946,
        };
        assert!((pair.charge_product()).abs() < 1e-15);
    }
}
