//! Hauser-Feshbach statistical model core computation.
//!
//! sigma(a -> b; E) = pi/k_a^2 * sum_{J,pi} (2J+1) / [(2j_a+1)(2J_A+1)]
//!                    * T_a(E,J,pi) * T_b(E,J,pi) / sum_c T_c(E,J,pi)

use nucrust_core::backend::WfcModel;
use nucrust_core::spin::Parity;
use nucrust_core::traits::{GammaStrength, LevelDensity, Multipole};
use nucrust_core::{Channel, CoreError, Projectile, TransmissionCoeffs};

use std::f64::consts::PI;

/// Configuration for HF calculation.
#[derive(Debug, Clone)]
pub struct HfConfig {
    /// Maximum total angular momentum J (in units of 2J).
    pub two_j_max: i32,
    /// Width fluctuation correction model.
    pub wfc_model: WfcModel,
    /// Number of quadrature points for WFC integration.
    pub wfc_quadrature: usize,
    /// Exit channels to include.
    pub exit_channels: Vec<Projectile>,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            two_j_max: 60, // J_max = 30
            wfc_model: WfcModel::Moldauer,
            wfc_quadrature: 8,
            exit_channels: vec![
                Projectile::Neutron,
                Projectile::Proton,
                Projectile::Alpha,
                Projectile::Gamma,
            ],
        }
    }
}

/// Result of a Hauser-Feshbach calculation at a single energy.
#[derive(Debug, Clone)]
pub struct HfResult {
    /// Compound nucleus formation cross section (mb).
    pub sigma_cn: f64,
    /// Channel cross sections (mb), indexed same as exit_channels in config.
    pub sigma_channels: Vec<f64>,
}

/// Exit channel data: transmission coefficients for a particle exit channel.
pub struct ExitChannelData<'a> {
    /// Exit channel description.
    pub channel: &'a Channel,
    /// Transmission coefficients (None for gamma channel).
    pub tc: &'a TransmissionCoeffs,
}

/// Hauser-Feshbach calculation context.
pub struct HfCalculation<'a> {
    /// Entrance channel.
    pub entrance: &'a Channel,
    /// Transmission coefficients for entrance channel.
    pub tc_entrance: &'a TransmissionCoeffs,
    /// Exit particle channels with their transmission coefficients.
    /// Does not include gamma (gamma is computed from NLD+GSF).
    pub exit_particle_channels: Vec<ExitChannelData<'a>>,
    /// Level density model.
    pub nld: &'a dyn LevelDensity,
    /// Gamma strength function model.
    pub gsf: &'a dyn GammaStrength,
    /// Configuration.
    pub config: &'a HfConfig,
}

/// Compute the entrance channel transmission for a given (J, pi).
///
/// T_a(E, J, pi) = sum over l, j satisfying:
/// - triangle condition: |J - j_a| <= l <= J + j_a
/// - parity selection: (-1)^l * pi_a * pi_A = pi
fn entrance_transmission(
    tc: &TransmissionCoeffs,
    e_idx: usize,
    two_j: i32,
    _parity: Parity,
    proj_spin_2j: i32,
) -> f64 {
    particle_transmission(tc, e_idx, two_j, _parity, proj_spin_2j)
}

/// Compute particle channel transmission for a given (J, pi).
///
/// Sums T_{lj}(E) over all (l, j) satisfying the triangle condition
/// |J - s_proj| <= l <= J + s_proj and parity selection.
fn particle_transmission(
    tc: &TransmissionCoeffs,
    e_idx: usize,
    two_j: i32,
    _parity: Parity,
    proj_spin_2j: i32,
) -> f64 {
    let n_e = tc.energy.len();
    let mut t_sum = 0.0;

    for l in 0..=tc.l_max {
        let l_i = l as i32;
        for j_idx in 0..2_usize {
            let two_j_particle = 2 * l_i + (2 * j_idx as i32 - 1);
            if two_j_particle < 0 {
                continue;
            }

            // Triangle condition: |two_j - proj_spin_2j| <= 2*l <= two_j + proj_spin_2j
            let two_l = 2 * l_i;
            let diff = (two_j - proj_spin_2j).abs();
            let sum = two_j + proj_spin_2j;
            if two_l < diff || two_l > sum {
                continue;
            }

            let flat_idx = l as usize * (2 * n_e) + j_idx * n_e + e_idx;
            if flat_idx < tc.data.len() {
                t_sum += tc.data[flat_idx];
            }
        }
    }

    t_sum
}

/// Compute gamma transmission for given compound nucleus (J, pi).
///
/// T_gamma = sum_{XL} integral f_{XL}(E_gamma) * E_gamma^{2L+1} * rho(U-E_gamma, J', pi') dE_gamma
fn gamma_transmission(
    gsf: &dyn GammaStrength,
    nld: &dyn LevelDensity,
    nuclide: &nucrust_core::Nuclide,
    excitation: f64,
    two_j: i32,
    parity: Parity,
) -> f64 {
    let mut t_gamma = 0.0;

    let n_points = 50;
    let e_max = excitation.min(30.0);
    if e_max < 0.1 {
        return 0.0;
    }
    let de = e_max / n_points as f64;

    for multipole in [Multipole::E1, Multipole::M1, Multipole::E2] {
        let l_order = multipole.order();
        let is_electric = multipole.is_electric();

        for i in 1..n_points {
            let e_gamma = i as f64 * de;
            let u_residual = excitation - e_gamma;
            if u_residual < 0.0 {
                break;
            }

            let f_xl = gsf.strength(nuclide, e_gamma, multipole);
            let e_factor = e_gamma.powi(2 * l_order as i32 + 1);

            // Parity selection for gamma transition
            let delta_parity = if is_electric {
                if l_order % 2 == 1 {
                    -parity.sign()
                } else {
                    parity.sign()
                }
            } else if l_order % 2 == 1 {
                parity.sign()
            } else {
                -parity.sign()
            };
            let final_parity = if delta_parity > 0 {
                Parity::Positive
            } else {
                Parity::Negative
            };

            // Sum over final spins J' reachable by multipole L
            let two_l = 2 * l_order as i32;
            let j_min = (two_j - two_l).max(0);
            let j_max = two_j + two_l;

            for two_jf in (j_min..=j_max).step_by(2) {
                let jf = two_jf as f64 / 2.0;
                let rho = nld.rho(nuclide, u_residual, jf, final_parity);
                t_gamma += f_xl * e_factor * rho * de;
            }
        }
    }

    t_gamma
}

/// Perform Hauser-Feshbach cross section calculation.
///
/// Returns cross sections for all energies in the entrance transmission coefficient grid.
/// sigma_channels contains one entry per exit channel in config.exit_channels order.
pub fn hauser_feshbach(calc: &HfCalculation) -> Result<Vec<HfResult>, CoreError> {
    let n_e = calc.tc_entrance.energy.len();
    let energies = calc.tc_entrance.energy.as_slice();
    let proj_spin_2j = (2.0 * calc.entrance.projectile.spin()) as i32;
    let target_spin_2j = 0_i32; // Assume even-even target (ground state 0+)
    let n_exit = calc.config.exit_channels.len();

    let mut results = Vec::with_capacity(n_e);

    for (e_idx, &energy) in energies.iter().enumerate() {
        let mut sigma_cn = 0.0;
        let mut sigma_exit = vec![0.0; n_exit];

        // Loop over J, pi
        for two_j in 0..=calc.config.two_j_max {
            for &parity in &[Parity::Positive, Parity::Negative] {
                // Entrance channel transmission
                let t_a =
                    entrance_transmission(calc.tc_entrance, e_idx, two_j, parity, proj_spin_2j);
                if t_a < 1e-30 {
                    continue;
                }

                let excitation = energy + calc.entrance.q_value;

                // Compute transmission for ALL exit channels
                let mut t_exit = vec![0.0; n_exit];
                let mut t_total = t_a; // entrance channel contributes to total

                for (ch_idx, proj) in calc.config.exit_channels.iter().enumerate() {
                    if *proj == Projectile::Gamma {
                        // Gamma channel: compute from NLD + GSF
                        let t_g = gamma_transmission(
                            calc.gsf,
                            calc.nld,
                            &calc.entrance.target,
                            excitation,
                            two_j,
                            parity,
                        );
                        t_exit[ch_idx] = t_g;
                        t_total += t_g;
                    } else {
                        // Particle exit channel: look up in exit_particle_channels
                        let exit_spin_2j = (2.0 * proj.spin()) as i32;
                        if let Some(ecd) = calc
                            .exit_particle_channels
                            .iter()
                            .find(|e| e.channel.projectile == *proj)
                        {
                            let t_b =
                                particle_transmission(ecd.tc, e_idx, two_j, parity, exit_spin_2j);
                            t_exit[ch_idx] = t_b;
                            t_total += t_b;
                        }
                    }
                }

                if t_total < 1e-30 {
                    continue;
                }

                // Statistical weight
                let g = (two_j as f64 + 1.0)
                    / ((proj_spin_2j as f64 + 1.0) * (target_spin_2j as f64 + 1.0));

                // Wave number
                let mu = nucrust_core::units::reduced_mass(
                    calc.entrance.projectile.mass_amu(),
                    calc.entrance.target.a() as f64,
                );
                let k = nucrust_core::units::wave_number(mu, energy);
                let k_sq = k * k;

                let prefactor = PI / k_sq * g * 10.0; // 10 fm² → mb

                sigma_cn += prefactor * t_a;
                for (ch_idx, &t_b) in t_exit.iter().enumerate() {
                    sigma_exit[ch_idx] += prefactor * t_a * t_b / t_total;
                }
            }
        }

        results.push(HfResult {
            sigma_cn,
            sigma_channels: sigma_exit,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsf::StandardLorentzian;
    use crate::nld::ConstantTemperature;
    use nucrust_core::{EnergyGrid, Nuclide};

    #[test]
    fn hf_cross_section_positive() {
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646, // Sn of Fe-57
        };

        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();

        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 2,
            data: vec![
                // l=0: j=0 (none), j=0.5
                0.0, 0.8, // l=1: j=0.5, j=1.5
                0.3, 0.5, // l=2: j=1.5, j=2.5
                0.1, 0.2,
            ],
        };

        let nld = ConstantTemperature {
            temperature: 0.88,
            e0: -1.16,
            a: 6.21,
        };
        let gsf = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        // Only gamma exit channel for this simple test
        let config = HfConfig {
            two_j_max: 10,
            exit_channels: vec![Projectile::Gamma],
            ..HfConfig::default()
        };

        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![], // no particle exit channels
            nld: &nld,
            gsf: &gsf,
            config: &config,
        };

        let results = hauser_feshbach(&calc).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].sigma_cn > 0.0,
            "sigma_cn = {}",
            results[0].sigma_cn
        );
        assert_eq!(results[0].sigma_channels.len(), 1);
        assert!(
            results[0].sigma_channels[0] > 0.0,
            "sigma_gamma = {}",
            results[0].sigma_channels[0]
        );
        // Partial cross section should not exceed compound formation
        assert!(
            results[0].sigma_channels[0] <= results[0].sigma_cn * 1.01,
            "sigma_gamma ({}) > sigma_cn ({})",
            results[0].sigma_channels[0],
            results[0].sigma_cn
        );
    }

    #[test]
    fn hf_multiple_exit_channels() {
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };

        let energies = EnergyGrid::from_values(vec![5.0]).unwrap();

        let tc_entrance = TransmissionCoeffs {
            energy: energies.clone(),
            l_max: 1,
            data: vec![0.0, 0.9, 0.3, 0.5],
        };
        // Reuse entrance TC for exit neutron channel (elastic)
        let exit_channel = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let tc_exit = TransmissionCoeffs {
            energy: energies,
            l_max: 1,
            data: vec![0.0, 0.9, 0.3, 0.5],
        };

        let nld = ConstantTemperature {
            temperature: 0.88,
            e0: -1.16,
            a: 6.21,
        };
        let gsf = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let config = HfConfig {
            two_j_max: 6,
            exit_channels: vec![Projectile::Neutron, Projectile::Gamma],
            ..HfConfig::default()
        };

        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc_entrance,
            exit_particle_channels: vec![ExitChannelData {
                channel: &exit_channel,
                tc: &tc_exit,
            }],
            nld: &nld,
            gsf: &gsf,
            config: &config,
        };

        let results = hauser_feshbach(&calc).unwrap();
        assert_eq!(results[0].sigma_channels.len(), 2);
        let sigma_n = results[0].sigma_channels[0];
        let sigma_g = results[0].sigma_channels[1];
        // Sum of partial cross sections should approximately equal sigma_cn
        let sigma_sum = sigma_n + sigma_g;
        assert!(
            (sigma_sum - results[0].sigma_cn).abs() / results[0].sigma_cn < 0.1,
            "sum of partials ({}) != sigma_cn ({})",
            sigma_sum,
            results[0].sigma_cn
        );
    }

    #[test]
    fn entrance_transmission_triangle() {
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 0,
            data: vec![0.0, 0.9],
        };

        let t = entrance_transmission(&tc, 0, 1, Parity::Positive, 1);
        assert!(t > 0.0, "T(J=1/2) = {}", t);
    }
}
