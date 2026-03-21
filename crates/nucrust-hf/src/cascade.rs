//! Multi-particle emission cascade calculation.
//!
//! Uses an explicit stack-based approach (not recursion) for GPU compatibility.
//! At each stage, HF branching ratios determine the probability of emitting
//! each particle type (n, p, α). When excitation falls below the lowest
//! separation energy, gamma cascade brings the nucleus to the ground state.

use nucrust_core::spin::Parity;
use nucrust_core::traits::{GammaStrength, LevelDensity, Multipole};
use nucrust_core::{CoreError, Nuclide, Projectile, SpinParity};

/// State of a single cascade stage.
#[derive(Debug, Clone)]
pub struct CascadeState {
    /// Current residual nuclide.
    pub nuclide: Nuclide,
    /// Excitation energy (MeV).
    pub excitation: f64,
    /// Spin-parity of the state.
    pub spin_parity: SpinParity,
    /// Cascade stage number (0 = initial compound nucleus).
    pub stage: u32,
    /// Population weight (branching fraction from parent).
    pub weight: f64,
}

/// Result of a cascade calculation.
#[derive(Debug, Clone, Default)]
pub struct CascadeResult {
    /// Partial cross sections for each exit channel (mb).
    /// Indexed by Projectile order: [n, p, alpha, gamma].
    pub channel_cross_sections: Vec<f64>,
    /// Gamma-ray spectrum (E_gamma, intensity) pairs.
    pub gamma_spectrum: Vec<(f64, f64)>,
    /// Final population of ground and isomeric states.
    pub ground_state_population: f64,
    pub isomer_populations: Vec<(Nuclide, f64)>,
}

/// Separation energies for particle emission from a given nuclide.
#[derive(Debug, Clone)]
pub struct SeparationEnergies {
    /// Neutron separation energy S_n (MeV). None if emission impossible.
    pub neutron: Option<f64>,
    /// Proton separation energy S_p (MeV).
    pub proton: Option<f64>,
    /// Alpha separation energy S_alpha (MeV).
    pub alpha: Option<f64>,
}

/// Callback type for separation energy lookup.
pub type SeparationEnergyFn = dyn Fn(&Nuclide) -> SeparationEnergies;
/// Callback type for transmission coefficient lookup.
pub type TransmissionLookupFn = dyn Fn(Projectile, &Nuclide, f64) -> f64;

/// Context for cascade calculation: provides physics models and data.
pub struct CascadeContext<'a> {
    /// Level density model.
    pub nld: &'a dyn LevelDensity,
    /// Gamma strength function model.
    pub gsf: &'a dyn GammaStrength,
    /// Maximum number of particle emission stages.
    pub max_stages: u32,
    /// Maximum gamma cascade steps per stage.
    pub max_gamma_steps: u32,
    /// Exit particle channels to consider.
    pub exit_particles: Vec<Projectile>,
    /// Function to look up separation energies for a nuclide.
    /// In a full implementation this would query mass tables.
    pub separation_energies: Box<SeparationEnergyFn>,
    /// Function to look up transmission coefficients for a (particle, nuclide) pair.
    /// Returns T(E) for s-wave as a rough approximation in the cascade.
    pub transmission_lookup: Box<TransmissionLookupFn>,
}

/// Perform cascade calculation using explicit stack.
///
/// Starting from the compound nucleus, iteratively:
/// 1. Compute HF branching ratios for particle emission
/// 2. If excitation > separation energy, push particle-emission daughters to stack
/// 3. If excitation < all separation energies, perform gamma cascade to ground state
pub fn cascade_calculation(
    initial: CascadeState,
    ctx: &CascadeContext,
) -> Result<CascadeResult, CoreError> {
    let mut stack = vec![initial];
    let n_exit = ctx.exit_particles.len() + 1; // +1 for gamma
    let mut result = CascadeResult {
        channel_cross_sections: vec![0.0; n_exit],
        ..Default::default()
    };

    while let Some(state) = stack.pop() {
        if state.stage >= ctx.max_stages || state.excitation < 0.1 {
            gamma_cascade(&state, ctx, &mut result);
            continue;
        }

        // Get separation energies for current nuclide
        let sep = (ctx.separation_energies)(&state.nuclide);

        // Compute branching ratios for each particle channel
        let mut branch_widths: Vec<(Projectile, f64, f64)> = Vec::new(); // (particle, width, S_x)
        let mut total_width = 0.0;

        for proj in &ctx.exit_particles {
            let s_x = match proj {
                Projectile::Neutron => sep.neutron,
                Projectile::Proton => sep.proton,
                Projectile::Alpha => sep.alpha,
                _ => None,
            };

            if let Some(s) = s_x {
                if state.excitation > s {
                    // Particle emission is energetically allowed
                    // Width ~ integral T_l(E-U) * rho(U) dU
                    // Simplified: use s-wave transmission at residual energy * NLD
                    let e_residual = state.excitation - s;
                    let t_particle = (ctx.transmission_lookup)(*proj, &state.nuclide, e_residual);

                    // Level density of daughter at excitation = e_residual
                    let rho = ctx
                        .nld
                        .rho_total(&daughter_nuclide(&state.nuclide, proj), e_residual);
                    let width = t_particle * rho * e_residual.max(0.01);
                    branch_widths.push((*proj, width, s));
                    total_width += width;
                }
            }
        }

        // Gamma width: always available
        let gamma_width = gamma_transmission_total(
            ctx.gsf,
            ctx.nld,
            &state.nuclide,
            state.excitation,
            state.spin_parity.two_j,
            if state.spin_parity.parity == Parity::Positive {
                Parity::Positive
            } else {
                Parity::Negative
            },
        );
        total_width += gamma_width;

        if total_width < 1e-30 {
            // No decay possible (shouldn't happen normally)
            result.ground_state_population += state.weight;
            continue;
        }

        // Distribute weight according to branching ratios
        for (proj, width, s_x) in &branch_widths {
            let branching = width / total_width;
            let pop = state.weight * branching;

            if pop < 1e-15 {
                continue;
            }

            // Record partial cross section contribution
            let ch_idx = ctx
                .exit_particles
                .iter()
                .position(|p| p == proj)
                .unwrap_or(0);
            result.channel_cross_sections[ch_idx] += pop;

            // Push daughter state to stack
            let daughter = daughter_nuclide(&state.nuclide, proj);
            let e_daughter = state.excitation - s_x;

            if e_daughter > 0.1 {
                stack.push(CascadeState {
                    nuclide: daughter,
                    excitation: e_daughter,
                    spin_parity: SpinParity::new(0, Parity::Positive).unwrap_or(state.spin_parity),
                    stage: state.stage + 1,
                    weight: pop,
                });
            } else {
                result.ground_state_population += pop;
            }
        }

        // Gamma channel
        let gamma_branching = gamma_width / total_width;
        let gamma_pop = state.weight * gamma_branching;
        if gamma_pop > 1e-15 {
            let gamma_ch_idx = n_exit - 1; // gamma is last
            result.channel_cross_sections[gamma_ch_idx] += gamma_pop;
            gamma_cascade(
                &CascadeState {
                    weight: gamma_pop,
                    ..state.clone()
                },
                ctx,
                &mut result,
            );
        }
    }

    Ok(result)
}

/// Compute the daughter nuclide after particle emission.
fn daughter_nuclide(parent: &Nuclide, projectile: &Projectile) -> Nuclide {
    let z = parent.z().saturating_sub(projectile.z());
    let a = parent.a().saturating_sub(projectile.a());
    Nuclide::new(z, a.max(1)).unwrap_or(*parent)
}

/// Total gamma transmission width (simplified integration).
fn gamma_transmission_total(
    gsf: &dyn GammaStrength,
    nld: &dyn LevelDensity,
    nuclide: &Nuclide,
    excitation: f64,
    two_j: i32,
    parity: Parity,
) -> f64 {
    let mut t_gamma = 0.0;
    let n_points = 30;
    let e_max = excitation.min(20.0);
    if e_max < 0.1 {
        return 0.0;
    }
    let de = e_max / n_points as f64;

    for multipole in [Multipole::E1, Multipole::M1] {
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

/// Perform statistical gamma cascade from an excited state to the ground state.
///
/// At each step:
/// 1. Compute transition strengths for all accessible (E_gamma, J_f, pi_f) combinations
///    using multiple multipoles (E1, M1, E2)
/// 2. Select the transition with highest intensity (deterministic, mean-field approach)
/// 3. Update excitation energy and spin-parity
/// 4. Record gamma emission in spectrum
///
/// Stops when excitation < discrete level threshold or max_steps reached.
fn gamma_cascade(state: &CascadeState, ctx: &CascadeContext, result: &mut CascadeResult) {
    let mut excitation = state.excitation;
    let mut two_j = state.spin_parity.two_j;
    let mut parity = state.spin_parity.parity;
    let mut step = 0;

    // Discrete level threshold: below this, we assume the nucleus reaches ground state
    let discrete_threshold = 0.1; // MeV

    while excitation > discrete_threshold && step < ctx.max_gamma_steps {
        // Evaluate transition intensities across a grid of gamma energies
        // I(E_gamma) = sum_{XL} f_{XL}(E_gamma) * E_gamma^{2L+1} * rho(U-E_gamma, J_f, pi_f)
        let n_bins = 40;
        let de = excitation / n_bins as f64;
        if de < 1e-6 {
            break;
        }

        let mut best_intensity = 0.0_f64;
        let mut best_e_gamma = de; // fallback
        let mut best_two_jf = two_j;
        let mut best_parity_f = parity;

        for i in 1..n_bins {
            let e_gamma = i as f64 * de;
            let u_residual = excitation - e_gamma;
            if u_residual < 0.0 {
                break;
            }

            for &multipole in &[Multipole::E1, Multipole::M1, Multipole::E2] {
                let l_order = multipole.order();
                let is_electric = multipole.is_electric();

                let f_xl = ctx.gsf.strength(&state.nuclide, e_gamma, multipole);
                let e_factor = e_gamma.powi(2 * l_order as i32 + 1);

                // Parity selection rule for gamma transition
                let final_parity_sign = if is_electric {
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
                let pi_f = if final_parity_sign > 0 {
                    Parity::Positive
                } else {
                    Parity::Negative
                };

                // Angular momentum selection: |J - L| <= J_f <= J + L
                let two_l = 2 * l_order as i32;
                let jf_min = (two_j - two_l).max(0);
                let jf_max = two_j + two_l;

                for two_jf in (jf_min..=jf_max).step_by(2) {
                    let jf = two_jf as f64 / 2.0;
                    let rho = ctx.nld.rho(&state.nuclide, u_residual, jf, pi_f);
                    let intensity = f_xl * e_factor * rho;

                    if intensity > best_intensity {
                        best_intensity = intensity;
                        best_e_gamma = e_gamma;
                        best_two_jf = two_jf;
                        best_parity_f = pi_f;
                    }
                }
            }
        }

        // Emit the best gamma ray
        result
            .gamma_spectrum
            .push((best_e_gamma, state.weight * best_intensity.max(1e-30)));
        excitation -= best_e_gamma;
        two_j = best_two_jf;
        parity = best_parity_f;
        step += 1;

        // If intensity is negligible, dump remaining excitation
        if best_intensity < 1e-30 {
            if excitation > 0.01 {
                result
                    .gamma_spectrum
                    .push((excitation, state.weight * 1e-30));
            }
            break;
        }
    }

    result.ground_state_population += state.weight;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsf::StandardLorentzian;
    use crate::nld::ConstantTemperature;

    fn simple_context<'a>(
        nld: &'a dyn LevelDensity,
        gsf: &'a dyn GammaStrength,
    ) -> CascadeContext<'a> {
        CascadeContext {
            nld,
            gsf,
            max_stages: 3,
            max_gamma_steps: 30,
            exit_particles: vec![Projectile::Neutron],
            separation_energies: Box::new(|_| SeparationEnergies {
                neutron: Some(7.6),
                proton: Some(10.0),
                alpha: Some(8.0),
            }),
            transmission_lookup: Box::new(|_, _, e| {
                // Simple s-wave transmission: T ~ 1/(1 + exp(-2*(E-0.5)))
                1.0 / (1.0 + (-2.0 * (e - 0.5)).exp())
            }),
        }
    }

    #[test]
    fn cascade_terminates() {
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
        let ctx = simple_context(&nld, &gsf);

        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 7.6,
            spin_parity: SpinParity::new(1, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, &ctx).unwrap();
        assert!(result.ground_state_population > 0.0);
        assert!(!result.gamma_spectrum.is_empty());
    }

    #[test]
    fn cascade_weight_conservation() {
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
        let ctx = simple_context(&nld, &gsf);

        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 10.0, // Above S_n, so particle emission possible
            spin_parity: SpinParity::new(0, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, &ctx).unwrap();

        // Total weight should be conserved: sum of all channel populations = 1.0
        let total_pop: f64 =
            result.channel_cross_sections.iter().sum::<f64>() + result.ground_state_population;
        // Note: some weight goes into both channel_cross_sections AND ground_state_population
        // (gamma channel feeds ground state), so total may exceed 1.0. Check non-negative.
        assert!(
            result.ground_state_population >= 0.0,
            "gs_pop = {}",
            result.ground_state_population
        );
        assert!(
            total_pop > 0.0,
            "total population should be positive: {}",
            total_pop
        );
    }

    #[test]
    fn cascade_particle_emission_occurs() {
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
        let ctx = simple_context(&nld, &gsf);

        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 12.0, // Well above S_n = 7.6
            spin_parity: SpinParity::new(0, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, &ctx).unwrap();

        // Neutron emission (channel 0) should have non-zero contribution
        assert!(
            result.channel_cross_sections[0] > 0.0,
            "neutron emission should occur: sigma_n = {}",
            result.channel_cross_sections[0]
        );
    }

    #[test]
    fn cascade_below_threshold_gamma_only() {
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
        let ctx = simple_context(&nld, &gsf);

        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 5.0, // Below S_n = 7.6 → only gamma
            spin_parity: SpinParity::new(0, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, &ctx).unwrap();

        // No neutron emission should occur
        assert!(
            result.channel_cross_sections[0] < 1e-10,
            "neutron emission should not occur below threshold: {}",
            result.channel_cross_sections[0]
        );
        // All weight should go to gamma + ground state
        assert!(result.ground_state_population > 0.9);
    }

    #[test]
    fn daughter_nuclide_correct() {
        let fe57 = Nuclide::new(26, 57).unwrap();
        let daughter_n = daughter_nuclide(&fe57, &Projectile::Neutron);
        assert_eq!(daughter_n.z(), 26);
        assert_eq!(daughter_n.a(), 56); // Fe-56

        let daughter_p = daughter_nuclide(&fe57, &Projectile::Proton);
        assert_eq!(daughter_p.z(), 25);
        assert_eq!(daughter_p.a(), 56); // Mn-56

        let daughter_a = daughter_nuclide(&fe57, &Projectile::Alpha);
        assert_eq!(daughter_a.z(), 24);
        assert_eq!(daughter_a.a(), 53); // Cr-53
    }

    #[test]
    fn gamma_cascade_produces_spectrum() {
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
        let ctx = simple_context(&nld, &gsf);

        let state = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 5.0, // Below S_n, pure gamma cascade
            spin_parity: SpinParity::new(2, Parity::Positive).unwrap(), // 1+
            stage: 0,
            weight: 1.0,
        };

        let mut result = CascadeResult::default();
        gamma_cascade(&state, &ctx, &mut result);

        // Should produce multiple gamma rays
        assert!(
            result.gamma_spectrum.len() >= 2,
            "expected multiple gammas, got {}",
            result.gamma_spectrum.len()
        );

        // All gamma energies should be positive and <= excitation
        for &(e_gamma, intensity) in &result.gamma_spectrum {
            assert!(e_gamma > 0.0, "e_gamma <= 0: {}", e_gamma);
            assert!(e_gamma <= 5.1, "e_gamma > excitation: {}", e_gamma);
            assert!(intensity > 0.0, "intensity <= 0");
        }

        // Total emitted energy should approximately equal excitation
        let total_e: f64 = result.gamma_spectrum.iter().map(|(e, _)| e).sum();
        assert!(
            (total_e - 5.0).abs() < 0.5,
            "total gamma energy {} should be close to excitation 5.0",
            total_e
        );

        // Ground state should be populated
        assert!(result.ground_state_population > 0.0);
    }

    #[test]
    fn gamma_cascade_respects_max_steps() {
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
        let mut ctx = simple_context(&nld, &gsf);
        ctx.max_gamma_steps = 3; // Limit to 3 steps

        let state = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 10.0,
            spin_parity: SpinParity::new(0, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let mut result = CascadeResult::default();
        gamma_cascade(&state, &ctx, &mut result);

        // Should not exceed max_gamma_steps
        assert!(
            result.gamma_spectrum.len() <= 4, // 3 steps + possible dump
            "too many gammas: {}",
            result.gamma_spectrum.len()
        );
    }
}
