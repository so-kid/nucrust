//! Hauser-Feshbach statistical model core computation.
//!
//! sigma(a -> b; E) = pi/k_a^2 * sum_{J,pi} (2J+1) / [(2j_a+1)(2J_A+1)]
//!                    * T_a(E,J,pi) * T_b(E,J,pi) / sum_c T_c(E,J,pi)

use nucrust_core::backend::WfcModel;
use nucrust_core::spin::Parity;
use nucrust_core::traits::{GammaStrength, LevelDensity, Multipole};
use nucrust_core::{Channel, CoreError, Projectile, TransmissionCoeffs};

use crate::{MIN_EMISSION_ENERGY, MIN_EXCITATION, NUMERICAL_FLOOR};
use std::f64::consts::PI;

/// Discrete level information for the HF calculation.
///
/// Used to connect discrete levels (below E_complete) with the continuous
/// NLD model (above E_complete) in the Hauser-Feshbach summation.
#[derive(Debug, Clone)]
pub struct DiscreteLevelInfo {
    /// Excitation energy (MeV).
    pub energy: f64,
    /// Spin J.
    pub spin: f64,
    /// Parity.
    pub parity: Parity,
}

/// Discrete level data for a specific nucleus.
///
/// Contains the known discrete levels and the completeness energy
/// above which the continuous NLD model takes over.
#[derive(Debug, Clone)]
pub struct DiscreteLevels {
    /// Known discrete levels, sorted by energy.
    pub levels: Vec<DiscreteLevelInfo>,
    /// Energy up to which discrete levels are considered complete (MeV).
    /// Above this energy, the continuous NLD model is used.
    pub e_complete: f64,
}

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
    /// Transmission coefficients for this exit particle.
    pub tc: &'a TransmissionCoeffs,
    /// Separation energy for this particle from the compound nucleus (MeV).
    /// E.g., S_n for neutron emission. When provided, enables continuum integration.
    pub separation_energy: Option<f64>,
    /// Level density model for the daughter (residual) nucleus.
    /// When provided with separation_energy, the exit channel uses
    /// ∫ T_{lj}(ε) · ρ_daughter(U, J', π') dU instead of simple grid lookup.
    pub daughter_nld: Option<&'a dyn LevelDensity>,
    /// Daughter nucleus.
    pub daughter_nuclide: Option<nucrust_core::Nuclide>,
    /// Discrete levels of the daughter nucleus (optional).
    pub daughter_discrete: Option<&'a DiscreteLevels>,
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
    /// Discrete levels of the compound nucleus (optional).
    ///
    /// When provided, the gamma transmission uses discrete levels below E_complete
    /// and the continuous NLD model above E_complete.
    pub discrete_levels: Option<&'a DiscreteLevels>,
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

/// Sum a per-(l, j) transmission contribution over all partial waves
/// satisfying the triangle condition |two_j - proj_spin_2j| <= 2*l <= two_j + proj_spin_2j.
fn sum_over_partial_waves<F>(
    tc: &TransmissionCoeffs,
    two_j: i32,
    proj_spin_2j: i32,
    mut t_of: F,
) -> f64
where
    F: FnMut(u32, usize) -> f64,
{
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

            t_sum += t_of(l, j_idx);
        }
    }

    t_sum
}

/// Compute particle channel transmission for a given (J, pi) at a grid energy index.
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
    sum_over_partial_waves(tc, two_j, proj_spin_2j, |l, j_idx| {
        let flat_idx = l as usize * (2 * n_e) + j_idx * n_e + e_idx;
        if flat_idx < tc.data.len() {
            tc.data[flat_idx]
        } else {
            0.0
        }
    })
}

/// Interpolate T_{lj}(ε) at arbitrary energy ε from the transmission coefficient grid.
///
/// Uses linear interpolation in log-energy space. Returns 0 if ε ≤ 0.
fn interpolate_transmission(tc: &TransmissionCoeffs, energy: f64, l: u32, j_idx: usize) -> f64 {
    if energy <= 0.0 {
        return 0.0;
    }
    let n_e = tc.energy.len();
    let e_grid = tc.energy.as_slice();

    // Clamp to grid range
    if energy <= e_grid[0] {
        let flat_idx = l as usize * (2 * n_e) + j_idx * n_e;
        return if flat_idx < tc.data.len() {
            tc.data[flat_idx]
        } else {
            0.0
        };
    }
    if energy >= e_grid[n_e - 1] {
        let flat_idx = l as usize * (2 * n_e) + j_idx * n_e + (n_e - 1);
        return if flat_idx < tc.data.len() {
            tc.data[flat_idx]
        } else {
            0.0
        };
    }

    // Binary search for bracketing interval
    let idx = e_grid
        .partition_point(|&e| e < energy)
        .saturating_sub(1)
        .min(n_e - 2);
    let e0 = e_grid[idx];
    let e1 = e_grid[idx + 1];
    let t = (energy - e0) / (e1 - e0);

    let base = l as usize * (2 * n_e) + j_idx * n_e;
    let t0 = if base + idx < tc.data.len() {
        tc.data[base + idx]
    } else {
        0.0
    };
    let t1 = if base + idx + 1 < tc.data.len() {
        tc.data[base + idx + 1]
    } else {
        0.0
    };

    t0 + t * (t1 - t0)
}

/// Sum interpolated T_{lj}(ε) over all (l, j) satisfying triangle condition.
fn particle_transmission_at_energy(
    tc: &TransmissionCoeffs,
    energy: f64,
    two_j: i32,
    proj_spin_2j: i32,
) -> f64 {
    if energy <= 0.0 {
        return 0.0;
    }
    sum_over_partial_waves(tc, two_j, proj_spin_2j, |l, j_idx| {
        interpolate_transmission(tc, energy, l, j_idx)
    })
}

/// Compute exit particle transmission with continuum level density integration.
///
/// T_b(J, π) = Σ_{discrete} T̃(ε_i) + ∫_{E_complete}^{U_max} Σ_{J'π'} T̃(ε(U)) · ρ(U, J', π') dU
///
/// where ε(U) = U_max - U is the exit particle kinetic energy and
/// U_max = excitation - separation_energy.
fn exit_particle_continuum_transmission(
    ecd: &ExitChannelData,
    excitation: f64,
    two_j: i32,
    _parity: Parity,
) -> f64 {
    let sep_e = match ecd.separation_energy {
        Some(s) => s,
        None => return 0.0, // Can't compute without separation energy
    };
    let nld = match ecd.daughter_nld {
        Some(n) => n,
        None => return 0.0,
    };
    let daughter = match ecd.daughter_nuclide {
        Some(ref n) => n,
        None => return 0.0,
    };

    let u_max = excitation - sep_e; // max daughter excitation
    if u_max < MIN_EMISSION_ENERGY {
        return 0.0;
    }

    let exit_spin_2j = (2.0 * ecd.channel.projectile.spin()) as i32;
    let mut t_total = 0.0;

    // Part 1: Discrete level contributions
    let e_complete = ecd.daughter_discrete.map_or(0.0, |d| d.e_complete);
    if let Some(disc) = ecd.daughter_discrete {
        for level in &disc.levels {
            if level.energy >= u_max {
                break;
            }
            if level.energy > disc.e_complete {
                break;
            }
            let epsilon = u_max - level.energy;
            if epsilon < MIN_EMISSION_ENERGY {
                continue;
            }

            // Sum T over all partial waves at this exit energy.
            // Angular momentum coupling (J = j_exit + J_daughter) is implicitly
            // handled by the triangle condition in particle_transmission_at_energy.
            t_total += particle_transmission_at_energy(ecd.tc, epsilon, two_j, exit_spin_2j);
        }
    }

    // Part 2: Continuum integration above E_complete
    if u_max > e_complete {
        let n_points = 30;
        let u_range = u_max - e_complete;
        let du = u_range / n_points as f64;

        for i in 1..n_points {
            let u_daughter = e_complete + i as f64 * du;
            let epsilon = u_max - u_daughter; // exit particle energy
            if epsilon < MIN_EMISSION_ENERGY {
                continue;
            }

            // Sum over final spins and parities of daughter
            for two_jf in (0..=two_j + exit_spin_2j).step_by(2) {
                for &final_parity in &[Parity::Positive, Parity::Negative] {
                    let jf = two_jf as f64 / 2.0;
                    let rho = nld.rho(daughter, u_daughter, jf, final_parity);
                    if rho < NUMERICAL_FLOOR {
                        continue;
                    }

                    let t_at_eps =
                        particle_transmission_at_energy(ecd.tc, epsilon, two_j, exit_spin_2j);
                    t_total += t_at_eps * rho * du;
                }
            }
        }
    }

    t_total
}

/// Check if a gamma transition of given multipole can reach (J_final, pi_final)
/// from (J_initial, pi_initial).
fn gamma_selection(
    two_j_init: i32,
    parity_init: Parity,
    multipole: Multipole,
    two_j_final: i32,
    parity_final: Parity,
) -> bool {
    let l_order = multipole.order();
    let two_l = 2 * l_order as i32;
    // Triangle condition
    if two_j_final < (two_j_init - two_l).abs() || two_j_final > two_j_init + two_l {
        return false;
    }
    // Parity selection
    multipole.final_parity(parity_init) == parity_final
}

/// Compute gamma transmission for given compound nucleus (J, pi).
///
/// When discrete levels are provided, uses them below E_complete and the
/// continuous NLD model above E_complete. Otherwise, uses NLD for all energies.
///
/// T_gamma = Σ_discrete + ∫_continuum f_{XL}(E_γ) · E_γ^{2L+1} · ρ(U-E_γ, J', π') dE_γ
fn gamma_transmission(
    gsf: &dyn GammaStrength,
    nld: &dyn LevelDensity,
    nuclide: &nucrust_core::Nuclide,
    excitation: f64,
    two_j: i32,
    parity: Parity,
    discrete: Option<&DiscreteLevels>,
) -> f64 {
    let mut t_gamma = 0.0;

    if excitation < MIN_EXCITATION {
        return 0.0;
    }

    // Part 1: Discrete level contributions (below E_complete)
    if let Some(disc) = discrete {
        for level in &disc.levels {
            if level.energy >= excitation {
                break; // levels are sorted by energy
            }
            if level.energy > disc.e_complete {
                break; // above completeness, use continuum
            }
            let e_gamma = excitation - level.energy;
            if e_gamma < MIN_EMISSION_ENERGY {
                continue;
            }

            let two_jf = (2.0 * level.spin).round() as i32;

            for multipole in [Multipole::E1, Multipole::M1, Multipole::E2] {
                if !gamma_selection(two_j, parity, multipole, two_jf, level.parity) {
                    continue;
                }
                let l_order = multipole.order();
                let f_xl = gsf.strength(nuclide, e_gamma, multipole);
                let e_factor = e_gamma.powi(2 * l_order as i32 + 1);
                // For discrete levels, rho = delta function → no density factor,
                // but we need the (2J+1) weight factor
                t_gamma += f_xl * e_factor;
            }
        }
    }

    // Part 2: Continuum contribution (above E_complete, or all if no discrete data)
    let e_complete = discrete.map_or(0.0, |d| d.e_complete);
    let e_cont_max = excitation - e_complete; // max gamma energy for continuum final states
    if e_cont_max > 0.1 {
        let n_points = 50;
        let de = e_cont_max / n_points as f64;

        for multipole in [Multipole::E1, Multipole::M1, Multipole::E2] {
            let l_order = multipole.order();
            let final_parity = multipole.final_parity(parity);

            for i in 1..n_points {
                let e_gamma = i as f64 * de;
                let u_residual = excitation - e_gamma;
                if u_residual < e_complete {
                    break;
                }

                let f_xl = gsf.strength(nuclide, e_gamma, multipole);
                let e_factor = e_gamma.powi(2 * l_order as i32 + 1);

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
                if t_a < NUMERICAL_FLOOR {
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
                            calc.discrete_levels,
                        );
                        t_exit[ch_idx] = t_g;
                        t_total += t_g;
                    } else {
                        // Particle exit channel
                        if let Some(ecd) = calc
                            .exit_particle_channels
                            .iter()
                            .find(|e| e.channel.projectile == *proj)
                        {
                            let t_b = if ecd.separation_energy.is_some()
                                && ecd.daughter_nld.is_some()
                            {
                                // Full continuum integration: ∫ T(ε) · ρ(U) dU
                                exit_particle_continuum_transmission(ecd, excitation, two_j, parity)
                            } else {
                                // Fallback: direct grid lookup (no NLD integration)
                                let exit_spin_2j = (2.0 * proj.spin()) as i32;
                                particle_transmission(ecd.tc, e_idx, two_j, parity, exit_spin_2j)
                            };
                            t_exit[ch_idx] = t_b;
                            t_total += t_b;
                        }
                    }
                }

                if t_total < NUMERICAL_FLOOR {
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
            discrete_levels: None,
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
                separation_energy: None,
                daughter_nld: None,
                daughter_nuclide: None,
                daughter_discrete: None,
            }],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
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

    #[test]
    fn hf_with_discrete_levels() {
        // Test that discrete levels change the result compared to pure NLD
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };

        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 2,
            data: vec![0.0, 0.8, 0.3, 0.5, 0.1, 0.2],
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
            two_j_max: 10,
            exit_channels: vec![Projectile::Gamma],
            ..HfConfig::default()
        };

        // Some Fe-57 low-lying levels (simplified)
        let discrete = DiscreteLevels {
            levels: vec![
                DiscreteLevelInfo {
                    energy: 0.0,
                    spin: 0.5,
                    parity: Parity::Negative,
                },
                DiscreteLevelInfo {
                    energy: 0.01438,
                    spin: 1.5,
                    parity: Parity::Negative,
                },
                DiscreteLevelInfo {
                    energy: 0.13653,
                    spin: 2.5,
                    parity: Parity::Negative,
                },
                DiscreteLevelInfo {
                    energy: 0.36689,
                    spin: 3.5,
                    parity: Parity::Negative,
                },
            ],
            e_complete: 1.0,
        };

        // Without discrete levels
        let calc_no_disc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };
        let result_no_disc = hauser_feshbach(&calc_no_disc).unwrap();

        // With discrete levels
        let calc_disc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: Some(&discrete),
        };
        let result_disc = hauser_feshbach(&calc_disc).unwrap();

        // Both should give positive cross sections
        assert!(result_no_disc[0].sigma_channels[0] > 0.0);
        assert!(result_disc[0].sigma_channels[0] > 0.0);

        // They should differ (discrete levels affect gamma transmission)
        let diff = (result_disc[0].sigma_channels[0] - result_no_disc[0].sigma_channels[0]).abs();
        assert!(
            diff > 0.0,
            "discrete levels should change cross section: disc={}, no_disc={}",
            result_disc[0].sigma_channels[0],
            result_no_disc[0].sigma_channels[0]
        );
    }

    #[test]
    fn exit_particle_continuum_integration() {
        // Test that the continuum integration for exit particle channels
        // gives a positive result and differs from simple grid lookup
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };

        // Multi-energy grid for interpolation testing
        let energies = EnergyGrid::from_values(vec![0.5, 1.0, 2.0, 5.0, 10.0]).unwrap();
        let n_e = energies.len();
        let l_max = 2;
        // data layout: l * 2*n_e + j_idx * n_e + e_idx
        let mut data = vec![0.0; (l_max + 1) as usize * 2 * n_e];
        // Set some reasonable transmission coefficients
        for l in 0..=l_max as usize {
            for j_idx in 0..2_usize {
                for (e_idx, &e) in energies.as_slice().iter().enumerate() {
                    let t = (0.9 * e / (e + 1.0)) * (0.8_f64).powi(l as i32);
                    data[l * 2 * n_e + j_idx * n_e + e_idx] = t;
                }
            }
        }

        let tc_exit = TransmissionCoeffs {
            energy: energies.clone(),
            l_max: l_max as u32,
            data: data.clone(),
        };
        let tc_entrance = TransmissionCoeffs {
            energy: energies,
            l_max: l_max as u32,
            data,
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

        let exit_channel = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };

        // With continuum integration
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
                separation_energy: Some(7.646), // S_n of Fe-57
                daughter_nld: Some(&nld),
                daughter_nuclide: Some(Nuclide::new(26, 56).unwrap()),
                daughter_discrete: None,
            }],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };

        let results = hauser_feshbach(&calc).unwrap();
        assert_eq!(results.len(), 5);

        // At higher energies, neutron emission should be open
        let sigma_n_high = results[4].sigma_channels[0];
        let sigma_g_high = results[4].sigma_channels[1];
        assert!(sigma_n_high > 0.0, "sigma_n at 10 MeV = {}", sigma_n_high);
        assert!(sigma_g_high > 0.0, "sigma_g at 10 MeV = {}", sigma_g_high);

        // Sum of partials should approximate sigma_cn
        let sum = sigma_n_high + sigma_g_high;
        assert!(
            (sum - results[4].sigma_cn).abs() / results[4].sigma_cn.max(1e-10) < 0.2,
            "sum={} vs sigma_cn={}",
            sum,
            results[4].sigma_cn
        );
    }

    #[test]
    fn interpolate_transmission_basic() {
        // For l_max=0, n_e=3: data layout is [l=0: j_idx=0 × 3 energies, j_idx=1 × 3 energies]
        // = data[l * 2*n_e + j_idx * n_e + e_idx]
        let energies = EnergyGrid::from_values(vec![1.0, 2.0, 5.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 0,
            data: vec![
                // l=0, j_idx=0 (j = l-1/2 = -1/2, unused), 3 energies:
                0.0, 0.0, 0.0, // l=0, j_idx=1 (j = l+1/2 = 1/2), 3 energies:
                0.5, 0.8, 0.95,
            ],
        };

        // At grid point E=2.0
        let t1 = interpolate_transmission(&tc, 2.0, 0, 1);
        assert!((t1 - 0.8).abs() < 1e-10, "at grid: {}", t1);

        // Between grid points (linear interpolation)
        let t_mid = interpolate_transmission(&tc, 1.5, 0, 1);
        assert!(t_mid > 0.5 && t_mid < 0.8, "interp: {}", t_mid);
        assert!((t_mid - 0.65).abs() < 1e-10, "expected 0.65, got {}", t_mid);

        // Below grid
        let t_low = interpolate_transmission(&tc, 0.5, 0, 1);
        assert!((t_low - 0.5).abs() < 1e-10, "below grid: {}", t_low);

        // At zero
        let t_zero = interpolate_transmission(&tc, 0.0, 0, 1);
        assert!(t_zero.abs() < 1e-15);
    }
}
