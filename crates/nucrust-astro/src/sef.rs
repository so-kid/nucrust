//! Stellar Enhancement Factor (SEF) computation.
//!
//! SEF(T) = σ*(T) / σ_gs(T)
//!
//! where σ*(T) is the thermally averaged cross section including
//! contributions from excited target states, and σ_gs(T) is the
//! ground-state cross section.

use nucrust_core::{CoreError, CrossSection};

/// Compute the stellar enhancement factor.
///
/// SEF(T) = [Σ_i (2J_i+1) σ_i(T) exp(-E_i/kT)] / [(2J_0+1) σ_0(T) · G(T)]
///
/// where G(T) = Σ_i (2J_i+1) exp(-E_i/kT) is the partition function.
///
/// Simplified version: when only the ground state cross section is available,
/// SEF ≈ 1 + corrections from excited states.
///
/// # Arguments
/// * `gs_cross_section` - Ground state (n,γ) cross section
/// * `excited_contributions` - (excitation_energy_MeV, spin_2j, cross_section) for each excited state
/// * `temperatures` - T9 grid (GK)
/// * `gs_spin_2j` - Ground state spin (2J representation)
pub fn compute_sef(
    gs_cross_section: &CrossSection,
    excited_contributions: &[(f64, i32, CrossSection)],
    temperatures: &[f64],
    gs_spin_2j: i32,
) -> Result<Vec<f64>, CoreError> {
    let kt_factor = nucrust_core::units::BOLTZMANN_MEV; // MeV/GK

    let gs_sigmas = &gs_cross_section.sigma_reaction;
    if gs_sigmas.is_empty() {
        return Err(CoreError::InvalidParameter {
            name: "gs_cross_section",
            value: 0.0,
            reason: "ground state cross section is empty",
        });
    }

    // Use the first energy point's cross section as representative
    // (full implementation would integrate over energy for each T)
    let sigma_gs_representative = gs_sigmas.iter().sum::<f64>() / gs_sigmas.len() as f64;

    let gs_weight = (gs_spin_2j + 1) as f64;

    let sef_values: Vec<f64> = temperatures
        .iter()
        .map(|&t9| {
            let kt = kt_factor * t9;
            if kt < 1e-15 {
                return 1.0;
            }

            // Partition function: G(T) = Σ (2J+1) exp(-E/kT)
            let mut partition = gs_weight; // ground state contributes (2J_0+1) × exp(0) = (2J_0+1)
            let mut weighted_sigma = gs_weight * sigma_gs_representative;

            for (e_exc, spin_2j, xs) in excited_contributions {
                let boltzmann = (-e_exc / kt).exp();
                let state_weight = (*spin_2j + 1) as f64;

                let sigma_exc =
                    xs.sigma_reaction.iter().sum::<f64>() / xs.sigma_reaction.len().max(1) as f64;

                partition += state_weight * boltzmann;
                weighted_sigma += state_weight * sigma_exc * boltzmann;
            }

            // SEF = σ*(T) / σ_gs = (weighted_sigma / partition) / (sigma_gs / gs_weight * gs_weight)
            //     = weighted_sigma / (partition * sigma_gs_representative / gs_weight * gs_weight)
            //     = weighted_sigma * gs_weight / (partition * sigma_gs_representative * gs_weight)
            // Simplify: SEF = (weighted_sigma / partition) / sigma_gs_representative
            if sigma_gs_representative.abs() < 1e-30 {
                return 1.0;
            }
            (weighted_sigma / partition) / sigma_gs_representative
        })
        .collect();

    Ok(sef_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::EnergyGrid;

    fn simple_xs(sigma: f64) -> CrossSection {
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        CrossSection {
            energy: energies,
            sigma_total: vec![sigma],
            sigma_elastic: vec![0.0],
            sigma_reaction: vec![sigma],
            partial: vec![],
        }
    }

    #[test]
    fn sef_unity_without_excited_states() {
        let gs = simple_xs(100.0);
        let sef = compute_sef(&gs, &[], &[0.3, 1.0, 3.0], 0).unwrap();
        for &s in &sef {
            assert!(
                (s - 1.0).abs() < 1e-10,
                "SEF should be 1.0 without excited states: {}",
                s
            );
        }
    }

    #[test]
    fn sef_greater_than_one_with_excited_state() {
        let gs = simple_xs(100.0);
        // Excited state at 0.1 MeV with larger cross section
        let exc = simple_xs(200.0);
        let excited = vec![(0.1, 2, exc)]; // 0.1 MeV, J=1 (2J=2)

        let sef = compute_sef(&gs, &excited, &[1.0, 3.0], 0).unwrap();
        // With a larger excited-state cross section, SEF > 1
        for &s in &sef {
            assert!(
                s > 1.0,
                "SEF should be > 1 with larger excited state xs: {}",
                s
            );
        }
    }

    #[test]
    fn sef_approaches_unity_at_low_temperature() {
        let gs = simple_xs(100.0);
        let exc = simple_xs(200.0);
        let excited = vec![(1.0, 2, exc)]; // High excitation: 1 MeV

        let sef = compute_sef(&gs, &excited, &[0.01], 0).unwrap();
        // At very low T, Boltzmann factor exp(-1/kT) ~ 0, so SEF ~ 1
        assert!(
            (sef[0] - 1.0).abs() < 0.01,
            "SEF at low T should be ~1: {}",
            sef[0]
        );
    }
}
