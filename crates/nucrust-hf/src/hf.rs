//! Hauser-Feshbach statistical model core computation.
//!
//! sigma(a -> b; E) = pi/k_a^2 * sum_{J,pi} (2J+1) / [(2j_a+1)(2J_A+1)]
//!                    * T_a(E,J,pi) * T_b(E,J,pi) / sum_c T_c(E,J,pi)

use nucrust_core::backend::WfcModel;
use nucrust_core::spin::Parity;
use nucrust_core::traits::{GammaStrength, LevelDensity, Multipole};
use nucrust_core::{Channel, CoreError, Projectile, TransmissionCoeffs};

use crate::wfc::{moldauer_w_row, MoldauerQuadrature};
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
    /// Width fluctuation correction model. `Moldauer` (default) and `None` are
    /// supported; `Goe` is rejected by [`hauser_feshbach`].
    pub wfc_model: WfcModel,
    /// Number of quadrature points for the Moldauer WFC integral
    /// (see [`MoldauerQuadrature`]).
    pub wfc_quadrature: usize,
    /// Exit channels to include.
    pub exit_channels: Vec<Projectile>,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            two_j_max: 60, // J_max = 30
            wfc_model: WfcModel::Moldauer,
            wfc_quadrature: 40,
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

/// Spin and parity of the target ground state.
///
/// Target spins are not yet carried by `Nuclide`; an even-even 0+ ground state
/// is assumed (correct for the Fe-56 benchmark and all even-even targets).
const TARGET_TWO_SPIN: i32 = 0;
const TARGET_PARITY: Parity = Parity::Positive;

/// Parity (-1)^l of an orbital angular momentum l.
fn orbital_parity(l: u32) -> Parity {
    if l % 2 == 0 {
        Parity::Positive
    } else {
        Parity::Negative
    }
}

/// Compute the entrance channel transmission for a given (J, pi).
///
/// T_a(E, J, pi) = sum of T_{lj}(E) over all (l, j) with
/// - j-I coupling: |j - I| <= J <= j + I (I = target spin)
/// - parity selection: (-1)^l * pi_A = pi (nucleons and alphas have positive intrinsic parity)
#[cfg(test)]
fn entrance_transmission(
    tc: &TransmissionCoeffs,
    e_idx: usize,
    two_j: i32,
    parity: Parity,
    proj_spin_2j: i32,
) -> f64 {
    let mut channels = Vec::new();
    push_grid_channels(
        tc,
        e_idx,
        two_j,
        parity,
        proj_spin_2j,
        TARGET_TWO_SPIN,
        TARGET_PARITY,
        &mut channels,
    );
    channels.iter().sum()
}

/// Visit every partial wave (l, j) of a particle that couples to the compound state
/// (J, pi) together with a residual/target state (I, pi_I).
///
/// For a spin-1/2 particle the stored j = l -+ 1/2 values are used directly and
/// each (l, j) is visited once if |j - I| <= J <= j + I and
/// (-1)^l pi_I = pi. For a spin-0 particle j = l and the j_index = 1 slot holds T_l.
/// Summed over J with weight (2J+1) this reproduces
/// sigma_CN = pi/k^2 * sum_{lj} (2j+1)/(2s+1) T_{lj}.
fn for_each_partial_wave<F>(
    l_max: u32,
    two_j: i32,
    parity: Parity,
    proj_spin_2j: i32,
    two_i: i32,
    parity_i: Parity,
    mut visit: F,
) where
    F: FnMut(u32, usize),
{
    for l in 0..=l_max {
        if orbital_parity(l) * parity_i != parity {
            continue;
        }
        let l_i = l as i32;
        for j_idx in 0..TransmissionCoeffs::J_SLOTS {
            let two_j_particle = if proj_spin_2j == 0 {
                if j_idx == 0 {
                    continue; // spin-0: single j = l, stored in slot 1
                }
                2 * l_i
            } else {
                2 * l_i + (2 * j_idx as i32 - 1)
            };
            if two_j_particle < 0 {
                continue;
            }

            // Angular momentum coupling J = j + I.
            if (two_j_particle + two_i + two_j) % 2 != 0
                || two_j < (two_j_particle - two_i).abs()
                || two_j > two_j_particle + two_i
            {
                continue;
            }

            visit(l, j_idx);
        }
    }
}

/// T_{lj} at grid index `e_idx` (0 outside the stored range).
fn grid_transmission(tc: &TransmissionCoeffs, e_idx: usize, l: u32, j_idx: usize) -> f64 {
    let n_e = tc.energy.len();
    let flat_idx = l as usize * (2 * n_e) + j_idx * n_e + e_idx;
    tc.data.get(flat_idx).copied().unwrap_or(0.0)
}

/// Push the individual T_{lj}(E_grid[e_idx]) of every partial wave coupling (J, pi) to
/// the residual state (I, pi_I) onto `out` (zero entries are skipped).
#[allow(clippy::too_many_arguments)]
fn push_grid_channels(
    tc: &TransmissionCoeffs,
    e_idx: usize,
    two_j: i32,
    parity: Parity,
    proj_spin_2j: i32,
    two_i: i32,
    parity_i: Parity,
    out: &mut Vec<f64>,
) {
    for_each_partial_wave(
        tc.l_max,
        two_j,
        parity,
        proj_spin_2j,
        two_i,
        parity_i,
        |l, j_idx| {
            let t = grid_transmission(tc, e_idx, l, j_idx);
            if t > 0.0 {
                out.push(t);
            }
        },
    );
}

/// Interpolate T_{lj}(ε) at arbitrary energy ε from the transmission coefficient grid.
///
/// Linear interpolation inside the grid; constant above it. Below the first grid
/// energy e_0 the low-energy behavior T_l ∝ ε^(l+1/2) of neutral particles is used,
/// T(ε) = T(e_0) (ε/e_0)^(l+1/2) (for charged particles this is an upper bound of the
/// Coulomb-suppressed value). Returns 0 if ε ≤ 0.
fn interpolate_transmission(tc: &TransmissionCoeffs, energy: f64, l: u32, j_idx: usize) -> f64 {
    if energy <= 0.0 {
        return 0.0;
    }
    let n_e = tc.energy.len();
    let e_grid = tc.energy.as_slice();

    if energy <= e_grid[0] {
        let t0 = grid_transmission(tc, 0, l, j_idx);
        return t0 * (energy / e_grid[0]).powf(l as f64 + 0.5);
    }
    if energy >= e_grid[n_e - 1] {
        return grid_transmission(tc, n_e - 1, l, j_idx);
    }

    // Binary search for bracketing interval
    let idx = e_grid
        .partition_point(|&e| e < energy)
        .saturating_sub(1)
        .min(n_e - 2);
    let e0 = e_grid[idx];
    let e1 = e_grid[idx + 1];
    let t = (energy - e0) / (e1 - e0);

    let t0 = grid_transmission(tc, idx, l, j_idx);
    let t1 = grid_transmission(tc, idx + 1, l, j_idx);
    t0 + t * (t1 - t0)
}

/// Sum interpolated T_{lj}(ε) over all (l, j) coupling (J, pi) to a residual state (I, pi_I).
fn particle_transmission_at_energy(
    tc: &TransmissionCoeffs,
    energy: f64,
    two_j: i32,
    parity: Parity,
    proj_spin_2j: i32,
    two_i: i32,
    parity_i: Parity,
) -> f64 {
    if energy <= 0.0 {
        return 0.0;
    }
    let mut t_sum = 0.0;
    for_each_partial_wave(
        tc.l_max,
        two_j,
        parity,
        proj_spin_2j,
        two_i,
        parity_i,
        |l, j_idx| t_sum += interpolate_transmission(tc, energy, l, j_idx),
    );
    t_sum
}

/// Transmission of one exit channel for a given compound state (J, pi), split for the
/// width fluctuation correction into individually resolved channels (a single (l, j)
/// to a discrete residual state) and a lumped sum of many weak channels (continuum bins,
/// gamma rays).
#[derive(Debug, Clone, Default)]
struct ChannelGroup {
    resolved: Vec<f64>,
    lumped: f64,
}

impl ChannelGroup {
    fn total(&self) -> f64 {
        self.resolved.iter().sum::<f64>() + self.lumped
    }
}

/// Compute exit particle transmission with continuum level density integration.
///
/// T_b(J, π) = Σ_{discrete i} Σ_{lj} T_{lj}(ε_i) + ∫_{E_complete}^{U_max} Σ_{J'π'} T̃(ε(U)) · ρ(U, J', π') dU
///
/// where ε(U) = U_max - U is the exit particle kinetic energy and
/// U_max = excitation - separation_energy. Every open final state is included down to
/// ε → 0 (no emission-energy cutoff): at low incident energy the decay back to the
/// target ground state (compound elastic) is the dominant channel.
///
/// Discrete levels give one resolved channel per (l, j); the continuum is lumped.
/// With `skip_ground_state`, a discrete level at zero excitation is left out (the caller
/// supplies the compound-elastic channels from the entrance transmissions instead).
fn exit_particle_continuum_transmission(
    ecd: &ExitChannelData,
    excitation: f64,
    two_j: i32,
    parity: Parity,
    skip_ground_state: bool,
    out: &mut ChannelGroup,
) {
    let (Some(sep_e), Some(nld), Some(daughter)) = (
        ecd.separation_energy,
        ecd.daughter_nld,
        ecd.daughter_nuclide.as_ref(),
    ) else {
        return; // Can't compute without separation energy, NLD and daughter
    };

    let u_max = excitation - sep_e; // max daughter excitation
    if u_max <= 0.0 {
        return;
    }

    let exit_spin_2j = (2.0 * ecd.channel.projectile.spin()) as i32;

    // Part 1: Discrete level contributions, one resolved channel per (l, j).
    let e_complete = ecd.daughter_discrete.map_or(0.0, |d| d.e_complete);
    if let Some(disc) = ecd.daughter_discrete {
        for level in &disc.levels {
            if level.energy >= u_max || level.energy > disc.e_complete {
                break;
            }
            if skip_ground_state && level.energy <= GROUND_STATE_TOLERANCE {
                continue;
            }
            let epsilon = u_max - level.energy;
            let two_jf = (2.0 * level.spin).round() as i32;
            for_each_partial_wave(
                ecd.tc.l_max,
                two_j,
                parity,
                exit_spin_2j,
                two_jf,
                level.parity,
                |l, j_idx| {
                    let t = interpolate_transmission(ecd.tc, epsilon, l, j_idx);
                    if t > 0.0 {
                        out.resolved.push(t);
                    }
                },
            );
        }
    }

    // Part 2: Continuum integration above E_complete (trapezoidal rule; the U = U_max
    // end point has ε = 0 and T = 0).
    if u_max > e_complete {
        let n_points = 30;
        let du = (u_max - e_complete) / n_points as f64;

        for i in 0..n_points {
            let u_daughter = e_complete + i as f64 * du;
            let epsilon = u_max - u_daughter; // exit particle energy, > 0
            let weight = if i == 0 { 0.5 * du } else { du };

            // Sum over final spins and parities of daughter: integer spins for
            // even-A, half-integer for odd-A daughters, up to J + j_max.
            let two_jf_min = i32::from(daughter.a() % 2);
            let two_jf_max = two_j + 2 * ecd.tc.l_max as i32 + exit_spin_2j;
            for two_jf in (two_jf_min..=two_jf_max).step_by(2) {
                for &final_parity in &[Parity::Positive, Parity::Negative] {
                    let jf = two_jf as f64 / 2.0;
                    let rho = nld.rho(daughter, u_daughter, jf, final_parity);
                    if rho < NUMERICAL_FLOOR {
                        continue;
                    }

                    let t_at_eps = particle_transmission_at_energy(
                        ecd.tc,
                        epsilon,
                        two_j,
                        parity,
                        exit_spin_2j,
                        two_jf,
                        final_parity,
                    );
                    out.lumped += t_at_eps * rho * weight;
                }
            }
        }
    }
}

/// Discrete levels at or below this excitation (MeV) are treated as the ground state.
const GROUND_STATE_TOLERANCE: f64 = 1e-6;

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
/// T_gamma = 2π [Σ_discrete f_{XL} E_γ^{2L+1}
///              + ∫_continuum f_{XL}(E_γ) · E_γ^{2L+1} · ρ(U-E_γ, J', π') dE_γ]
///
/// with f_{XL} in MeV^{-(2L+1)} and ρ in MeV⁻¹ (dimensionless T_gamma).
/// `nuclide` is the emitting (compound) nucleus.
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
                // |J - L|, not max(J - L, 0): keeps half-integer J' for half-integer J.
                let j_min = (two_j - two_l).abs();
                let j_max = two_j + two_l;

                for two_jf in (j_min..=j_max).step_by(2) {
                    let jf = two_jf as f64 / 2.0;
                    let rho = nld.rho(nuclide, u_residual, jf, final_parity);
                    t_gamma += f_xl * e_factor * rho * de;
                }
            }
        }
    }

    2.0 * PI * t_gamma
}

/// Compute the Hauser-Feshbach cross section at a single entrance energy.
///
/// Sums the (J, pi) contributions for the grid energy at `e_idx`:
///
/// sigma_b = pi/k^2 sum_{J,pi} g_J sum_{a in entrance} sum_{c in b} T_a T_c W_ac / T_total
///
/// where W_ac are the Moldauer width fluctuation factors (1 without WFC).
fn hauser_feshbach_at_energy(
    calc: &HfCalculation,
    quad: Option<&MoldauerQuadrature>,
    e_idx: usize,
    energy: f64,
) -> HfResult {
    let proj_spin_2j = (2.0 * calc.entrance.projectile.spin()) as i32;
    let target_spin_2j = TARGET_TWO_SPIN;
    let n_exit = calc.config.exit_channels.len();

    // Gamma rays are emitted by the compound nucleus (target + projectile).
    let target = calc.entrance.target;
    let projectile = calc.entrance.projectile;
    let compound =
        nucrust_core::Nuclide::new(target.z() + projectile.z(), target.a() + projectile.a())
            .unwrap_or(target);

    // The exit particle channel (if any) that contains the compound-elastic decay back to
    // the target ground state, and whether that state is resolved there: either the
    // daughter's discrete level scheme includes the ground state at emission energy
    // E (continuum treatment), or the direct grid lookup to the target ground state is used.
    // The elastic channels then take the entrance T_lj, so that elastic decay and
    // formation are described by the same (l, j) channels (needed for W_aa).
    let elastic_idx = calc
        .config
        .exit_channels
        .iter()
        .position(|p| *p == projectile);
    let elastic_ecd = elastic_idx.and_then(|_| {
        calc.exit_particle_channels
            .iter()
            .find(|e| e.channel.projectile == projectile)
    });
    let entrance_in_exits = elastic_ecd.is_some();
    let elastic_resolved = elastic_ecd.is_some_and(|ecd| {
        if ecd.separation_energy.is_some() && ecd.daughter_nld.is_some() {
            ecd.daughter_nuclide == Some(target)
                && ecd
                    .separation_energy
                    .is_some_and(|s| (s - calc.entrance.q_value).abs() < GROUND_STATE_TOLERANCE)
                && ecd.daughter_discrete.is_some_and(|d| {
                    d.levels
                        .first()
                        .is_some_and(|l| l.energy <= GROUND_STATE_TOLERANCE)
                })
        } else {
            ecd.channel.target == target
        }
    });

    // Wave number and statistical prefactor 10 fm² → mb.
    let mu = nucrust_core::units::reduced_mass(
        calc.entrance.projectile.mass_amu(),
        calc.entrance.target.a() as f64,
    );
    let k = nucrust_core::units::wave_number(mu, energy);
    let pi_over_k_sq = PI / (k * k) * 10.0;
    let excitation = energy + calc.entrance.q_value;

    let mut sigma_cn = 0.0;
    let mut sigma_exit = vec![0.0; n_exit];
    let mut entrance = Vec::new();
    let mut groups = vec![ChannelGroup::default(); n_exit];

    // Loop over J, pi
    for two_j in 0..=calc.config.two_j_max {
        for &parity in &[Parity::Positive, Parity::Negative] {
            // Entrance channel transmissions T_a, one per (l, j)
            entrance.clear();
            push_grid_channels(
                calc.tc_entrance,
                e_idx,
                two_j,
                parity,
                proj_spin_2j,
                TARGET_TWO_SPIN,
                TARGET_PARITY,
                &mut entrance,
            );
            let t_a: f64 = entrance.iter().sum();
            if t_a < NUMERICAL_FLOOR {
                continue;
            }

            // Transmissions of ALL exit channels
            for (ch_idx, proj) in calc.config.exit_channels.iter().enumerate() {
                let group = &mut groups[ch_idx];
                group.resolved.clear();
                group.lumped = 0.0;
                if *proj == Projectile::Gamma {
                    // Gamma channel: many weak channels, computed from NLD + GSF
                    group.lumped = gamma_transmission(
                        calc.gsf,
                        calc.nld,
                        &compound,
                        excitation,
                        two_j,
                        parity,
                        calc.discrete_levels,
                    );
                    continue;
                }
                let Some(ecd) = calc
                    .exit_particle_channels
                    .iter()
                    .find(|e| e.channel.projectile == *proj)
                else {
                    continue;
                };
                let is_elastic = Some(ch_idx) == elastic_idx && elastic_resolved;
                if is_elastic {
                    group.resolved.extend_from_slice(&entrance);
                }
                if ecd.separation_energy.is_some() && ecd.daughter_nld.is_some() {
                    // Full continuum integration: ∫ T(ε) · ρ(U) dU
                    exit_particle_continuum_transmission(
                        ecd, excitation, two_j, parity, is_elastic, group,
                    );
                } else if !is_elastic {
                    // Fallback: direct grid lookup to the residual ground state
                    // (assumed 0+ like the target), no NLD integration.
                    let exit_spin_2j = (2.0 * proj.spin()) as i32;
                    push_grid_channels(
                        ecd.tc,
                        e_idx,
                        two_j,
                        parity,
                        exit_spin_2j,
                        TARGET_TWO_SPIN,
                        TARGET_PARITY,
                        &mut group.resolved,
                    );
                }
            }

            // The entrance channel competes in the denominator unless an exit particle
            // channel of the same type already accounts for it.
            let t_exit_sum: f64 = groups.iter().map(ChannelGroup::total).sum();
            let t_total = if entrance_in_exits {
                t_exit_sum
            } else {
                t_exit_sum + t_a
            };
            if t_total < NUMERICAL_FLOOR {
                continue;
            }

            // Statistical weight
            let g = (two_j as f64 + 1.0)
                / ((proj_spin_2j as f64 + 1.0) * (target_spin_2j as f64 + 1.0));
            let prefactor = pi_over_k_sq * g;
            sigma_cn += prefactor * t_a;

            // Width fluctuations need the entrance channels among the resolved ones:
            // as the compound-elastic exit channels, or as an extra competing channel
            // when no particle exit channel of the entrance type is included.
            let wfc_quad = quad.filter(|_| elastic_resolved || !entrance_in_exits);
            let Some(wfc_quad) = wfc_quad else {
                for (sigma, group) in sigma_exit.iter_mut().zip(&groups) {
                    *sigma += prefactor * t_a * group.total() / t_total;
                }
                continue;
            };

            // Flatten the resolved channels: entrance channels first.
            let mut resolved = Vec::new();
            let mut lumped = 0.0;
            if !elastic_resolved {
                resolved.extend_from_slice(&entrance);
            }
            let mut ranges = Vec::with_capacity(n_exit);
            for group in &groups {
                let start = resolved.len();
                resolved.extend_from_slice(&group.resolved);
                ranges.push(start..resolved.len());
                lumped += group.lumped;
            }
            // Position of the entrance channels in `resolved`.
            let entrance_start = match elastic_idx {
                Some(idx) if elastic_resolved => ranges[idx].start,
                _ => 0,
            };

            for (i, &t_ent) in entrance.iter().enumerate() {
                let (w_row, w_lumped) =
                    moldauer_w_row(&resolved, lumped, entrance_start + i, wfc_quad);
                for ((sigma, group), range) in sigma_exit.iter_mut().zip(&groups).zip(&ranges) {
                    let t_w: f64 = resolved[range.clone()]
                        .iter()
                        .zip(&w_row[range.clone()])
                        .map(|(t, w)| t * w)
                        .sum::<f64>()
                        + group.lumped * w_lumped;
                    *sigma += prefactor * t_ent * t_w / t_total;
                }
            }
        }
    }

    HfResult {
        sigma_cn,
        sigma_channels: sigma_exit,
    }
}

/// Perform Hauser-Feshbach cross section calculation.
///
/// Returns cross sections for all energies in the entrance transmission coefficient grid.
/// sigma_channels contains one entry per exit channel in config.exit_channels order.
///
/// Width fluctuations follow `config.wfc_model`: `Moldauer` applies the Moldauer
/// correction (see [`moldauer_w_row`](crate::wfc::moldauer_w_row)), with the gamma
/// channel and particle continua treated as lumped weak channels. It needs the entrance
/// channels resolved: when a particle exit channel of the entrance type is included, its
/// daughter discrete levels must contain the target ground state (or the direct grid
/// lookup must be used); otherwise the correction is skipped (W = 1). `Goe` returns an
/// error (the GOE triple integral is not validated).
///
/// With the `parallel` feature, energies are distributed across rayon worker
/// threads (each energy point is independent); results are identical to the
/// sequential path.
pub fn hauser_feshbach(calc: &HfCalculation) -> Result<Vec<HfResult>, CoreError> {
    let energies = calc.tc_entrance.energy.as_slice();
    let quad = match calc.config.wfc_model {
        WfcModel::None => None,
        WfcModel::Moldauer => Some(MoldauerQuadrature::new(calc.config.wfc_quadrature)),
        WfcModel::Goe => {
            return Err(CoreError::InvalidParameter {
                name: "wfc_model",
                value: f64::NAN,
                reason: "GOE width fluctuations are not supported by hauser_feshbach; \
                         use Moldauer or None",
            })
        }
    };
    let quad = quad.as_ref();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Ok(energies
            .par_iter()
            .enumerate()
            .map(|(e_idx, &energy)| hauser_feshbach_at_energy(calc, quad, e_idx, energy))
            .collect())
    }

    #[cfg(not(feature = "parallel"))]
    {
        Ok(energies
            .iter()
            .enumerate()
            .map(|(e_idx, &energy)| hauser_feshbach_at_energy(calc, quad, e_idx, energy))
            .collect())
    }
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
    fn entrance_transmission_selects_single_lj_for_spin_zero_target() {
        // n + 0+ target: (J, pi) is reached by exactly one (l, j = J), l fixed by parity.
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 2,
            // l=0: j=1/2 -> 0.9; l=1: j=1/2 -> 0.3, j=3/2 -> 0.5; l=2: j=3/2 -> 0.1, j=5/2 -> 0.2
            data: vec![0.0, 0.9, 0.3, 0.5, 0.1, 0.2],
        };
        let t = |two_j, parity| entrance_transmission(&tc, 0, two_j, parity, 1);
        assert_eq!(t(1, Parity::Positive), 0.9); // s1/2
        assert_eq!(t(1, Parity::Negative), 0.3); // p1/2
        assert_eq!(t(3, Parity::Negative), 0.5); // p3/2
        assert_eq!(t(3, Parity::Positive), 0.1); // d3/2
        assert_eq!(t(5, Parity::Positive), 0.2); // d5/2
        assert_eq!(t(5, Parity::Negative), 0.0); // f5/2 not tabulated
        assert_eq!(t(2, Parity::Positive), 0.0); // integer J unreachable
    }

    #[test]
    fn sigma_cn_equals_optical_reaction_cross_section() {
        // Summing the HF entrance weights over (J, pi) must reproduce
        // sigma_R = pi/k^2 * sum_{lj} (2j+1)/2 * T_{lj}  (n + 0+ target), in mb.
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let data = vec![0.0, 0.93, 0.21, 0.20, 0.35, 0.38, 0.004, 0.003];
        let tc = TransmissionCoeffs {
            energy: energies,
            l_max: 3,
            data: data.clone(),
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
            two_j_max: 20,
            exit_channels: vec![Projectile::Gamma],
            ..HfConfig::default()
        };
        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };
        let r = hauser_feshbach(&calc).unwrap();

        let mu = nucrust_core::units::reduced_mass(Projectile::Neutron.mass_amu(), 56.0);
        let k = nucrust_core::units::wave_number(mu, 1.0);
        // k for 1 MeV (CM) neutrons on Fe-56 is ~0.218 fm^-1 -> pi/k^2 ~ 663 mb.
        assert!((k - 0.2177).abs() < 1e-3, "k = {k}");
        let mut weighted = 0.0;
        for l in 0..=3_usize {
            for j_idx in 0..2_usize {
                let two_j = 2 * l as i32 + 2 * j_idx as i32 - 1;
                if two_j < 0 {
                    continue;
                }
                weighted += (two_j as f64 + 1.0) / 2.0 * data[l * 2 + j_idx];
            }
        }
        let sigma_r = PI / (k * k) * weighted * 10.0;
        assert!(
            (r[0].sigma_cn / sigma_r - 1.0).abs() < 1e-12,
            "sigma_cn = {} mb, sigma_R = {} mb",
            r[0].sigma_cn,
            sigma_r
        );
        // ~2.3 b for these Fe-56-like T_lj at 1 MeV.
        assert!(r[0].sigma_cn > 2000.0 && r[0].sigma_cn < 2600.0);
        // Capture is a small fraction of compound formation (T_gamma << T_n).
        assert!(r[0].sigma_channels[0] < 0.1 * r[0].sigma_cn);
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

        // Below grid: T_l ∝ ε^(l+1/2) (s wave: sqrt)
        let t_low = interpolate_transmission(&tc, 0.5, 0, 1);
        assert!(
            (t_low - 0.5 * 0.5f64.sqrt()).abs() < 1e-12,
            "below grid: {}",
            t_low
        );

        // At zero
        let t_zero = interpolate_transmission(&tc, 0.0, 0, 1);
        assert!(t_zero.abs() < 1e-15);
    }

    /// Fe-56 + n with Fe-56-like neutron T_lj (T_l ∝ E^(l+1/2) at low energy), gamma +
    /// neutron exit channels, neutron emission to the 0+ ground state, the 2+ level at
    /// 0.847 MeV and a CT continuum above 1.0 MeV.
    fn fe56_like_results(energies: Vec<f64>, wfc_model: WfcModel) -> Vec<HfResult> {
        fn neutron_tc(energies: Vec<f64>) -> TransmissionCoeffs {
            let grid = EnergyGrid::from_values(energies).unwrap();
            let n_e = grid.len();
            let l_max = 4_u32;
            let mut data = vec![0.0; (l_max as usize + 1) * 2 * n_e];
            for l in 0..=l_max as usize {
                for j_idx in 0..2 {
                    if l == 0 && j_idx == 0 {
                        continue;
                    }
                    for (e_idx, &e) in grid.as_slice().iter().enumerate() {
                        let x = (e / 1.5_f64).powf(l as f64 + 0.5);
                        // s wave: T_0 ~ 0.03 at 1 keV.
                        data[l * 2 * n_e + j_idx * n_e + e_idx] = 0.95 * x / (1.0 + x);
                    }
                }
            }
            TransmissionCoeffs {
                energy: grid,
                l_max,
                data,
            }
        }
        let q = 7.646;
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: q,
        };
        let exit_channel = Channel {
            q_value: 0.0,
            ..entrance.clone()
        };
        let tc = neutron_tc(energies);
        let exit_grid: Vec<f64> = (0..60).map(|i| 1e-4 * 1.2_f64.powi(i)).collect();
        let tc_exit = neutron_tc(exit_grid);
        let nld_cn = ConstantTemperature {
            temperature: 0.88,
            e0: 1.94,
            a: 6.21,
        };
        let nld_fe56 = ConstantTemperature {
            temperature: 1.0,
            e0: 1.0 - 2f64.ln(),
            a: 6.0,
        };
        let fe56_levels = DiscreteLevels {
            levels: vec![
                DiscreteLevelInfo {
                    energy: 0.0,
                    spin: 0.0,
                    parity: Parity::Positive,
                },
                DiscreteLevelInfo {
                    energy: 0.846778,
                    spin: 2.0,
                    parity: Parity::Positive,
                },
            ],
            e_complete: 1.0,
        };
        let gsf = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let config = HfConfig {
            two_j_max: 21,
            wfc_model,
            exit_channels: vec![Projectile::Gamma, Projectile::Neutron],
            ..HfConfig::default()
        };
        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![ExitChannelData {
                channel: &exit_channel,
                tc: &tc_exit,
                separation_energy: Some(q),
                daughter_nld: Some(&nld_fe56),
                daughter_nuclide: Some(exit_channel.target),
                daughter_discrete: Some(&fe56_levels),
            }],
            nld: &nld_cn,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };
        hauser_feshbach(&calc).unwrap()
    }

    #[test]
    fn compound_elastic_kept_below_10_kev() {
        // Regression: emission energies below 10 keV were dropped, which removed compound
        // elastic for E < 10 keV and gave sigma_gamma = sigma_CN.
        let energies = vec![0.001, 0.005, 0.02];
        for wfc in [WfcModel::None, WfcModel::Moldauer] {
            let r = fe56_like_results(energies.clone(), wfc);
            for (res, e) in r.iter().zip(&energies) {
                let (sigma_g, sigma_n) = (res.sigma_channels[0], res.sigma_channels[1]);
                assert!(
                    sigma_g < 0.05 * res.sigma_cn,
                    "{wfc:?} E={e}: sigma_g = {sigma_g} mb, sigma_CN = {} mb",
                    res.sigma_cn
                );
                assert!(
                    sigma_n > 0.95 * res.sigma_cn,
                    "{wfc:?} E={e}: sigma_n = {sigma_n}"
                );
            }
            // 1/v-like capture: sigma_g * sqrt(E) roughly constant at keV energies.
            let ratio = r[0].sigma_channels[0] * 0.001f64.sqrt()
                / (r[2].sigma_channels[0] * 0.02f64.sqrt());
            assert!((0.5..2.0).contains(&ratio), "{wfc:?}: 1/v ratio {ratio}");
        }
    }

    #[test]
    fn exit_channels_conserve_flux() {
        // sigma_gamma + sigma_n = sigma_CN, with and without width fluctuations, from the
        // elastic-only region through inelastic and continuum emission.
        let energies = vec![0.001, 0.1, 0.9, 2.0, 5.0];
        for wfc in [WfcModel::None, WfcModel::Moldauer] {
            for (res, e) in fe56_like_results(energies.clone(), wfc)
                .iter()
                .zip(&energies)
            {
                let sum: f64 = res.sigma_channels.iter().sum();
                assert!(
                    (sum / res.sigma_cn - 1.0).abs() < 1e-6,
                    "{wfc:?} E={e}: sum of partials {sum} vs sigma_CN {}",
                    res.sigma_cn
                );
            }
        }
    }

    #[test]
    fn moldauer_wfc_enhances_compound_elastic() {
        let energies = vec![0.001, 0.1, 0.5];
        let plain = fe56_like_results(energies.clone(), WfcModel::None);
        let wfc = fe56_like_results(energies.clone(), WfcModel::Moldauer);
        for ((p, w), e) in plain.iter().zip(&wfc).zip(&energies) {
            assert_eq!(p.sigma_cn, w.sigma_cn);
            // Elastic enhanced, competing gamma channel depleted.
            assert!(
                w.sigma_channels[1] > p.sigma_channels[1],
                "E={e}: sigma_n {} (WFC) <= {} (none)",
                w.sigma_channels[1],
                p.sigma_channels[1]
            );
            let depletion = w.sigma_channels[0] / p.sigma_channels[0];
            assert!(
                depletion < 0.99 && depletion > 0.3,
                "E={e}: sigma_g WFC/none = {depletion}"
            );
        }
    }

    #[test]
    fn moldauer_wfc_with_entrance_only_in_denominator() {
        // Gamma-only exit list: the entrance channel competes as a resolved channel;
        // W_aa enhancement of the (unlisted) elastic channel lowers sigma_gamma.
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };
        let tc = TransmissionCoeffs {
            energy: EnergyGrid::from_values(vec![0.01]).unwrap(),
            l_max: 1,
            data: vec![0.0, 0.1, 0.002, 0.004],
        };
        let nld = ConstantTemperature {
            temperature: 0.88,
            e0: 1.94,
            a: 6.21,
        };
        let gsf = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let run = |wfc_model| {
            let config = HfConfig {
                two_j_max: 7,
                wfc_model,
                exit_channels: vec![Projectile::Gamma],
                ..HfConfig::default()
            };
            let calc = HfCalculation {
                entrance: &entrance,
                tc_entrance: &tc,
                exit_particle_channels: vec![],
                nld: &nld,
                gsf: &gsf,
                config: &config,
                discrete_levels: None,
            };
            hauser_feshbach(&calc).unwrap()[0].sigma_channels[0]
        };
        let (plain, wfc) = (run(WfcModel::None), run(WfcModel::Moldauer));
        assert!(
            wfc < plain && wfc > 0.3 * plain,
            "WFC {wfc} vs none {plain}"
        );
    }

    #[test]
    fn goe_wfc_model_is_rejected() {
        let entrance = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 7.646,
        };
        let tc = TransmissionCoeffs {
            energy: EnergyGrid::from_values(vec![1.0]).unwrap(),
            l_max: 0,
            data: vec![0.0, 0.5],
        };
        let nld = ConstantTemperature {
            temperature: 0.88,
            e0: 1.94,
            a: 6.21,
        };
        let gsf = StandardLorentzian {
            e_gdr: 16.36,
            gamma_gdr: 4.58,
            sigma_gdr: 136.0,
            m1_params: None,
        };
        let config = HfConfig {
            wfc_model: WfcModel::Goe,
            exit_channels: vec![Projectile::Gamma],
            ..HfConfig::default()
        };
        let calc = HfCalculation {
            entrance: &entrance,
            tc_entrance: &tc,
            exit_particle_channels: vec![],
            nld: &nld,
            gsf: &gsf,
            config: &config,
            discrete_levels: None,
        };
        assert!(hauser_feshbach(&calc).is_err());
    }

    #[test]
    fn discrete_emission_below_10_kev_included() {
        // Emission to discrete levels at ε = 1 keV (ground state) and ε = 3 keV (2+ level)
        // must be counted: no low-energy emission cutoff.
        let grid = EnergyGrid::from_values(vec![1e-4, 1e-3, 1e-2, 1.0]).unwrap();
        let n_e = grid.len();
        let mut data = vec![0.0; 3 * 2 * n_e];
        for (l, t_l) in [
            (0_usize, [0.003, 0.03, 0.1, 0.9]),
            (2, [1e-9, 1e-6, 1e-4, 0.3]),
        ] {
            for j_idx in 0..2 {
                if l > 0 || j_idx == 1 {
                    data[l * 2 * n_e + j_idx * n_e..][..n_e].copy_from_slice(&t_l);
                }
            }
        }
        let tc = TransmissionCoeffs {
            energy: grid,
            l_max: 2,
            data,
        };
        let channel = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let nld = ConstantTemperature {
            temperature: 1.0,
            e0: -0.5,
            a: 6.0,
        };
        let levels = DiscreteLevels {
            levels: vec![
                DiscreteLevelInfo {
                    energy: 0.0,
                    spin: 0.0,
                    parity: Parity::Positive,
                },
                DiscreteLevelInfo {
                    energy: 0.0002,
                    spin: 2.0,
                    parity: Parity::Positive,
                },
            ],
            e_complete: 0.0002,
        };
        let ecd = ExitChannelData {
            channel: &channel,
            tc: &tc,
            separation_energy: Some(7.646),
            daughter_nld: Some(&nld),
            daughter_nuclide: Some(channel.target),
            daughter_discrete: Some(&levels),
        };
        // J = 1/2+: s1/2 to the 0+ ground state (ε = 1 keV); d3/2 and d5/2 to the 2+ level.
        let mut group = ChannelGroup::default();
        exit_particle_continuum_transmission(
            &ecd,
            7.646 + 0.001,
            1,
            Parity::Positive,
            false,
            &mut group,
        );
        assert_eq!(group.resolved.len(), 3, "{group:?}");
        assert!((group.resolved[0] - 0.03).abs() < 1e-12, "{group:?}");
        assert!(
            group.resolved[1] > 0.0 && group.resolved[1] < 1e-6,
            "{group:?}"
        );
        assert!(group.lumped > 0.0, "continuum above 0.2 keV: {group:?}");
        // skip_ground_state leaves only the 2+ level.
        let mut group = ChannelGroup::default();
        exit_particle_continuum_transmission(
            &ecd,
            7.646 + 0.001,
            1,
            Parity::Positive,
            true,
            &mut group,
        );
        assert_eq!(group.resolved.len(), 2, "{group:?}");
    }
}
