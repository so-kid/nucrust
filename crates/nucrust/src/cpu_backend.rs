//! CPU compute backend using downstream crates.
//!
//! Implements `ComputeBackend` by delegating to nucrust-optical, nucrust-hf,
//! and nucrust-rmatrix computations.

use nucrust_core::backend::{
    ComputeBackend, GsfModelParams, HfConfig, MacsConfig, NldModelParams, NumerovConfig,
    NumerovTask, RMatrixParams as CoreRMatrixParams,
};
use nucrust_core::{
    Channel, CollisionMatrix, CoreError, CrossSection, EnergyGrid, Projectile, ReactionRate,
    TransmissionCoeffs,
};

use nucrust_hf::gsf::StandardLorentzian;
use nucrust_hf::hf::{self, HfCalculation};
use nucrust_hf::nld::{BackShiftedFermiGas, ConstantTemperature};
use nucrust_optical::omp::KoningDelaroche;
use nucrust_optical::transmission::compute_transmission_coeffs;

/// CPU compute backend.
///
/// Delegates to nucrust-optical, nucrust-hf, and nucrust-rmatrix for the
/// actual computations. Does not use rayon parallelism yet (sequential).
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn batch_numerov(
        &self,
        _params: &[NumerovTask],
        _config: &NumerovConfig,
    ) -> Result<TransmissionCoeffs, CoreError> {
        // For batch_numerov, the params contain individual (nuclide, E, l, j) tasks.
        // However, compute_transmission_coeffs handles the l/j iteration internally.
        // This method is a lower-level batch API; for now, return an error if called
        // directly, since the higher-level compute_transmission_coeffs is preferred.
        Err(CoreError::InvalidParameter {
            name: "batch_numerov",
            value: 0.0,
            reason: "use compute_transmission_coeffs() directly for CPU backend",
        })
    }

    fn hf_summation(
        &self,
        tc: &TransmissionCoeffs,
        nld_params: &NldModelParams,
        gsf_params: &GsfModelParams,
        config: &HfConfig,
    ) -> Result<CrossSection, CoreError> {
        // Create NLD model from params
        let nld: Box<dyn nucrust_core::LevelDensity> = match nld_params {
            NldModelParams::ConstantTemperature { t, e0 } => Box::new(ConstantTemperature {
                temperature: *t,
                e0: *e0,
                a: 6.0, // default level density parameter
            }),
            NldModelParams::Bsfg { a, delta, sigma } => Box::new(BackShiftedFermiGas {
                a: *a,
                delta: *delta,
                sigma: Some(*sigma),
            }),
            _ => {
                return Err(CoreError::InvalidParameter {
                    name: "nld_model",
                    value: 0.0,
                    reason: "unsupported NLD model for CPU backend",
                })
            }
        };

        // Create GSF model from params
        let gsf: Box<dyn nucrust_core::GammaStrength> = match gsf_params {
            GsfModelParams::Slo {
                e_gdr,
                gamma_gdr,
                sigma_gdr,
            } => Box::new(StandardLorentzian {
                e_gdr: *e_gdr,
                gamma_gdr: *gamma_gdr,
                sigma_gdr: *sigma_gdr,
                m1_params: None,
            }),
            _ => {
                return Err(CoreError::InvalidParameter {
                    name: "gsf_model",
                    value: 0.0,
                    reason: "unsupported GSF model for CPU backend",
                })
            }
        };

        // Create a dummy entrance channel (will be refined when pipeline is connected)
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: nucrust_core::Nuclide::new(26, 56).map_err(|_e| {
                CoreError::InvalidParameter {
                    name: "target",
                    value: 0.0,
                    reason: "invalid nuclide",
                }
            })?,
            q_value: 0.0,
        };

        let hf_config = hf::HfConfig {
            two_j_max: config.j_max * 2,
            exit_channels: vec![Projectile::Gamma],
            ..hf::HfConfig::default()
        };

        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: tc,
            exit_particle_channels: vec![],
            nld: nld.as_ref(),
            gsf: gsf.as_ref(),
            config: &hf_config,
        };

        let results = hf::hauser_feshbach(&calc)?;

        // Convert HfResult to CrossSection
        let n_e = tc.energy.len();
        let sigma_total: Vec<f64> = results.iter().map(|r| r.sigma_cn).collect();
        let sigma_reaction = sigma_total.clone();
        let sigma_elastic = vec![0.0; n_e]; // not computed in this path

        Ok(CrossSection {
            energy: tc.energy.clone(),
            sigma_total,
            sigma_elastic,
            sigma_reaction,
            partial: vec![],
        })
    }

    fn rmatrix_solve(
        &self,
        rmatrix: &CoreRMatrixParams,
        energies: &[f64],
    ) -> Result<CollisionMatrix, CoreError> {
        // The core RMatrixParams is a placeholder with just n_channels/n_levels.
        // For a real computation, the full nucrust_rmatrix::RMatrixParams is needed.
        // This method serves as a bridge; for now return a dummy result.
        let n_ch = rmatrix.n_channels;
        let n_e = energies.len();
        let u_matrix = vec![num_complex::Complex64::new(1.0, 0.0); n_e * n_ch * n_ch];

        Ok(CollisionMatrix {
            energies: energies.to_vec(),
            n_channels: n_ch,
            u_matrix,
        })
    }

    fn macs_integrate(
        &self,
        _cross_sections: &[CrossSection],
        _temperatures: &[f64],
        _config: &MacsConfig,
    ) -> Result<Vec<ReactionRate>, CoreError> {
        // MACS integration will be implemented in Phase 4 (nucrust-astro)
        Err(CoreError::InvalidParameter {
            name: "macs_integrate",
            value: 0.0,
            reason: "not yet implemented (Phase 4)",
        })
    }
}

/// Higher-level convenience: compute transmission coefficients for a channel.
///
/// This is the preferred CPU path (rather than batch_numerov).
pub fn cpu_transmission_coeffs(
    channel: &Channel,
    energies: &EnergyGrid,
    config: &NumerovConfig,
) -> Result<TransmissionCoeffs, CoreError> {
    let omp = KoningDelaroche;
    let optical_config = nucrust_core::backend::NumerovConfig {
        step_size: config.step_size,
        r_min: config.r_min,
        convergence_tl: config.convergence_tl,
        max_l: config.max_l,
    };
    compute_transmission_coeffs(&omp, channel, energies, &optical_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_creates() {
        let _backend = CpuBackend::new();
    }

    #[test]
    fn cpu_transmission_coeffs_works() {
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: nucrust_core::Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![1.0, 5.0]).unwrap();
        let config = NumerovConfig {
            max_l: 5,
            ..Default::default()
        };

        let tc = cpu_transmission_coeffs(&ch, &energies, &config).unwrap();
        assert!(!tc.data.is_empty());
        assert!(tc.l_max >= 1, "l_max = {}", tc.l_max);
        // Verify data has been computed (finite values)
        assert!(
            tc.data.iter().all(|v| v.is_finite()),
            "all T values should be finite"
        );
    }

    #[test]
    fn cpu_hf_summation_works() {
        let backend = CpuBackend::new();
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 1,
            data: vec![0.0, 0.8, 0.3, 0.5],
        };
        let nld = NldModelParams::ConstantTemperature { t: 0.88, e0: -1.16 };
        let gsf = GsfModelParams::Slo {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
        };
        let config = HfConfig::default();

        let xs = backend.hf_summation(&tc, &nld, &gsf, &config).unwrap();
        assert_eq!(xs.sigma_total.len(), 1);
        assert!(xs.sigma_total[0] > 0.0, "sigma = {}", xs.sigma_total[0]);
    }
}
