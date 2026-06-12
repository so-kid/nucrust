//! Transmission coefficient computation.
//!
//! Computes T_{l,j}(E) for all partial waves up to l_max convergence.

use nucrust_core::backend::NumerovConfig;
use nucrust_core::traits::OpticalPotential;
use nucrust_core::{Channel, CoreError, EnergyGrid, TransmissionCoeffs};

use crate::numerov::numerov_integrate;

/// Compute T_{l,j} for both spin-orbit couplings (j = l ∓ 1/2) at one energy.
///
/// Returns `[T_{l-1/2}, T_{l+1/2}]`; the j < 0 coupling (l = 0) stays 0.
fn transmission_at_energy(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energy: f64,
    l: u32,
    config: &NumerovConfig,
) -> Result<[f64; 2], CoreError> {
    let mut t_lj = [0.0_f64; 2];
    for (j_idx, &dj) in [-(0.5_f64), 0.5].iter().enumerate() {
        let j = l as f64 + dj;
        if j < 0.0 {
            continue;
        }
        let s_lj = numerov_integrate(potential, channel, energy, l, j, config)?;
        t_lj[j_idx] = (1.0 - s_lj.norm_sqr()).clamp(0.0, 1.0);
    }
    Ok(t_lj)
}

/// Compute T_{l,j} for all energies of one partial wave.
///
/// With the `parallel` feature, energies are distributed across rayon worker
/// threads (each Numerov integration is independent); results are identical
/// to the sequential path.
fn transmission_for_partial_wave(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energies: &EnergyGrid,
    l: u32,
    config: &NumerovConfig,
) -> Result<Vec<[f64; 2]>, CoreError> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        energies
            .as_slice()
            .par_iter()
            .map(|&e| transmission_at_energy(potential, channel, e, l, config))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        energies
            .as_slice()
            .iter()
            .map(|&e| transmission_at_energy(potential, channel, e, l, config))
            .collect()
    }
}

/// Compute transmission coefficients for all partial waves and energies.
///
/// For each partial wave l, computes j = l-1/2 and j = l+1/2 (spin-orbit splitting).
/// Convergence: stops when max T_l over all energies falls below `config.convergence_tl`.
pub fn compute_transmission_coeffs(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energies: &EnergyGrid,
    config: &NumerovConfig,
) -> Result<TransmissionCoeffs, CoreError> {
    let n_e = energies.len();
    let mut l_max_actual = 0_u32;
    let mut all_data: Vec<Vec<[f64; 2]>> = Vec::new(); // [l][e_index] -> [T_{l-1/2}, T_{l+1/2}]

    for l in 0..=config.max_l {
        let tl_data = transmission_for_partial_wave(potential, channel, energies, l, config)?;
        let max_tl = tl_data.iter().flatten().fold(0.0_f64, |acc, &t| acc.max(t));

        all_data.push(tl_data);
        l_max_actual = l;

        // Convergence check: if T_l is below threshold for all energies, stop
        if max_tl < config.convergence_tl && l > 0 {
            break;
        }
    }

    // Pack into TransmissionCoeffs SoA layout:
    // data[l][j_index][e_index] flattened as:
    // [l0_j0_e0, l0_j0_e1, ..., l0_j1_e0, ..., l1_j0_e0, ...]
    let total_size = (l_max_actual as usize + 1) * 2 * n_e;
    let mut data = vec![0.0; total_size];

    #[allow(clippy::needless_range_loop)]
    for (l, tl_data) in all_data.iter().enumerate() {
        for j_idx in 0..2 {
            for e_idx in 0..n_e {
                let flat_idx = l * (2 * n_e) + j_idx * n_e + e_idx;
                data[flat_idx] = tl_data[e_idx][j_idx];
            }
        }
    }

    Ok(TransmissionCoeffs {
        energy: energies.clone(),
        l_max: l_max_actual,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omp::CustomOmp;
    use nucrust_core::{Nuclide, Projectile};

    fn fe56_n() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn transmission_coeffs_in_range() {
        let omp = CustomOmp::default();
        let ch = fe56_n();
        let energies = EnergyGrid::logarithmic(1.0, 10.0, 5).unwrap();
        let config = NumerovConfig {
            max_l: 10,
            ..NumerovConfig::default()
        };

        let tc = compute_transmission_coeffs(&omp, &ch, &energies, &config).unwrap();

        // All T values should be in [0, 1]
        for &t in &tc.data {
            assert!((0.0..=1.0).contains(&t), "T = {} out of range", t);
        }
    }

    #[test]
    fn transmission_computation_succeeds() {
        let omp = CustomOmp::default();
        let ch = fe56_n();
        let energies = EnergyGrid::from_values(vec![5.0]).unwrap();
        let config = NumerovConfig {
            max_l: 5,
            ..NumerovConfig::default()
        };

        let tc = compute_transmission_coeffs(&omp, &ch, &energies, &config);
        assert!(tc.is_ok(), "Transmission computation should succeed");
        let tc = tc.unwrap();
        assert!(tc.l_max >= 1, "Should compute at least l=0 and l=1");
    }
}
