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

use nucrust_hf::gsf::{EnhancedGeneralizedLorentzian, StandardLorentzian};
use nucrust_hf::hf::{self, HfCalculation};
use nucrust_hf::nld::{BackShiftedFermiGas, ConstantTemperature, GilbertCameron, Ignatyuk};
use nucrust_optical::omp::KoningDelaroche;
use nucrust_optical::transmission::compute_transmission_coeffs;
use nucrust_rmatrix::types::{
    BoundaryCondition, ParticlePair, RMatrixChannel, RMatrixLevel, RMatrixParams,
};

/// CPU compute backend.
///
/// Delegates to nucrust-optical, nucrust-hf, and nucrust-rmatrix for the
/// actual computations. Does not use rayon parallelism yet (sequential).
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend.
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
        let nld = build_nld(nld_params)?;
        let gsf = build_gsf(gsf_params)?;

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
            wfc_model: config.wfc_model,
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
            discrete_levels: None,
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
        // Convert CoreRMatrixParams to nucrust_rmatrix::RMatrixParams
        let channels: Vec<RMatrixChannel> = rmatrix
            .channels
            .iter()
            .map(|ch| RMatrixChannel {
                pair: ParticlePair {
                    light: ch.projectile,
                    heavy: ch.target,
                    q_value: ch.q_value,
                    separation_energy: ch.separation_energy,
                },
                l: ch.l,
                s: ch.s,
                j: ch.j,
                radius: ch.radius,
            })
            .collect();

        let levels: Vec<RMatrixLevel> = rmatrix
            .levels
            .iter()
            .map(|lv| RMatrixLevel {
                energy: lv.energy,
                reduced_widths: lv.reduced_widths.clone(),
            })
            .collect();

        let boundary_condition = match &rmatrix.boundary_b {
            Some(b) => BoundaryCondition::Standard { b: b.clone() },
            None => BoundaryCondition::Brune,
        };

        let params = RMatrixParams {
            channels,
            levels,
            boundary_condition,
        };

        let result = nucrust_rmatrix::rmatrix_cross_section(&params, energies)?;
        Ok(result.collision_matrix)
    }

    fn macs_integrate(
        &self,
        cross_sections: &[CrossSection],
        temperatures: &[f64],
        _config: &MacsConfig,
    ) -> Result<Vec<ReactionRate>, CoreError> {
        let astro_config = nucrust_astro::MacsConfig {
            n_gauss_points: _config.n_quadrature_points,
            temperature_grid: temperatures.to_vec(),
        };

        let mut rates = Vec::with_capacity(cross_sections.len());
        for xs in cross_sections {
            let macs = nucrust_astro::compute_macs(xs, &astro_config)?;
            // Use a default reduced mass (1 amu for simplicity);
            // in production this would come from the channel data.
            let rate = nucrust_astro::compute_reaction_rate(&macs, temperatures, 1.0)?;
            rates.push(rate);
        }
        Ok(rates)
    }
}

/// Instantiate a level density model from backend parameters.
fn build_nld(
    nld_params: &NldModelParams,
) -> Result<Box<dyn nucrust_core::LevelDensity>, CoreError> {
    Ok(match nld_params {
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
        NldModelParams::GilbertCameron {
            a,
            delta,
            t,
            e0,
            e_match,
            sigma,
        } => Box::new(GilbertCameron {
            ct: ConstantTemperature {
                temperature: *t,
                e0: *e0,
                a: *a,
            },
            bsfg: BackShiftedFermiGas {
                a: *a,
                delta: *delta,
                sigma: Some(*sigma),
            },
            e_match: *e_match,
        }),
        NldModelParams::Ignatyuk {
            a_tilde,
            delta_w,
            gamma,
            delta,
        } => Box::new(Ignatyuk {
            a_tilde: *a_tilde,
            delta_w: *delta_w,
            gamma: *gamma,
            delta: *delta,
        }),
        NldModelParams::HfbTable {
            excitations,
            spins,
            densities,
        } => Box::new(
            nucrust_hf::HfbTableInterp::new(excitations.clone(), spins.clone(), densities)
                .map_err(|_| CoreError::InvalidParameter {
                    name: "hfb_table",
                    value: 0.0,
                    reason: "invalid HFB table data",
                })?,
        ),
    })
}

/// Instantiate a gamma strength function model from backend parameters.
fn build_gsf(
    gsf_params: &GsfModelParams,
) -> Result<Box<dyn nucrust_core::GammaStrength>, CoreError> {
    Ok(match gsf_params {
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
        GsfModelParams::Eglo {
            e_gdr,
            gamma_gdr,
            sigma_gdr,
            temperature,
        } => Box::new(EnhancedGeneralizedLorentzian {
            e_gdr: *e_gdr,
            gamma_gdr: *gamma_gdr,
            sigma_gdr: *sigma_gdr,
            temperature: *temperature,
        }),
        GsfModelParams::QrpaTable {
            energies,
            strengths_e1,
            strengths_m1,
        } => Box::new(
            nucrust_hf::QrpaTableInterp::new(energies, strengths_e1, strengths_m1).map_err(
                |_| CoreError::InvalidParameter {
                    name: "qrpa_table",
                    value: 0.0,
                    reason: "invalid QRPA table data",
                },
            )?,
        ),
    })
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

    #[test]
    fn cpu_hf_all_nld_gsf_variants() {
        let backend = CpuBackend::new();
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 1,
            data: vec![0.0, 0.8, 0.3, 0.5],
        };
        let config = HfConfig::default();

        // GilbertCameron NLD
        let nld_gc = NldModelParams::GilbertCameron {
            a: 6.21,
            delta: -0.52,
            t: 0.88,
            e0: -1.16,
            e_match: 3.24,
            sigma: 3.5,
        };
        let gsf_slo = GsfModelParams::Slo {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
        };
        let xs = backend
            .hf_summation(&tc, &nld_gc, &gsf_slo, &config)
            .unwrap();
        assert!(xs.sigma_total[0] > 0.0);

        // Ignatyuk NLD
        let nld_ig = NldModelParams::Ignatyuk {
            a_tilde: 6.0,
            delta_w: -3.0,
            gamma: 0.04,
            delta: -0.5,
        };
        let xs = backend
            .hf_summation(&tc, &nld_ig, &gsf_slo, &config)
            .unwrap();
        assert!(xs.sigma_total[0] > 0.0);

        // EGLO GSF
        let gsf_eglo = GsfModelParams::Eglo {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            temperature: 0.5,
        };
        let nld_ct = NldModelParams::ConstantTemperature { t: 0.88, e0: -1.16 };
        let xs = backend
            .hf_summation(&tc, &nld_ct, &gsf_eglo, &config)
            .unwrap();
        assert!(xs.sigma_total[0] > 0.0);
    }

    #[test]
    fn cpu_rmatrix_solve_works() {
        use nucrust_core::backend::{RMatrixChannelData, RMatrixLevelData};
        use nucrust_core::SpinParity;

        let backend = CpuBackend::new();
        let params = CoreRMatrixParams {
            n_channels: 1,
            n_levels: 1,
            channels: vec![RMatrixChannelData {
                projectile: Projectile::Proton,
                target: nucrust_core::Nuclide::new(4, 7).unwrap(), // Be-7
                q_value: 0.0,
                separation_energy: 0.1375,
                l: 0,
                s: 1.0,
                j: SpinParity {
                    two_j: 2,
                    parity: nucrust_core::Parity::Positive,
                },
                radius: 3.75,
            }],
            levels: vec![RMatrixLevelData {
                energy: 0.6,
                reduced_widths: vec![0.5],
            }],
            boundary_b: Some(vec![0.0]),
        };

        let energies = vec![0.3, 0.5, 0.6, 0.7, 1.0];
        let result = backend.rmatrix_solve(&params, &energies).unwrap();
        assert_eq!(result.energies.len(), 5);
        assert_eq!(result.n_channels, 1);
        // U-matrix should have non-trivial values near resonance
        assert!(result.u_matrix.iter().all(|u| u.norm() <= 1.01));
    }

    #[test]
    fn cpu_macs_integrate_works() {
        let backend = CpuBackend::new();
        // Create a simple cross section
        let energies = EnergyGrid::from_values(vec![0.01, 0.1, 0.5, 1.0, 5.0, 10.0]).unwrap();
        let sigmas: Vec<f64> = energies
            .as_slice()
            .iter()
            .map(|&e| 100.0 / e.sqrt())
            .collect();
        let xs = CrossSection {
            energy: energies,
            sigma_total: sigmas.clone(),
            sigma_elastic: vec![0.0; 6],
            sigma_reaction: sigmas,
            partial: vec![],
        };

        let temperatures = vec![0.3, 1.0, 3.0];
        let config = MacsConfig {
            n_quadrature_points: 20,
        };
        let rates = backend
            .macs_integrate(&[xs], &temperatures, &config)
            .unwrap();
        assert_eq!(rates.len(), 1);
        assert_eq!(rates[0].temperatures.len(), 3);
        assert!(rates[0]
            .na_sigma_v
            .iter()
            .all(|&r| r > 0.0 && r.is_finite()));
    }
}
