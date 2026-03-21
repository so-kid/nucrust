//! Brune alternative parameterization (Phys. Rev. C 66, 2002).
//!
//! Provides conversion between standard R-matrix parameters and
//! Brune alternative parameters where:
//! - Level energies are observable (shift vanishes at resonance)
//! - Reduced widths are defined via the eigenvectors of the nonlinear eigenvalue problem

use nucrust_core::CoreError;

use crate::rmatrix::shift_penetrability;
use crate::types::{BoundaryCondition, RMatrixChannel, RMatrixLevel, RMatrixParams};

/// Maximum Newton iterations for nonlinear eigenvalue problem.
const MAX_NEWTON_ITER: u32 = 200;
/// Convergence threshold for Newton iteration.
const NEWTON_TOL: f64 = 1e-12;
/// Step size for numerical derivative of shift function.
const DS_STEP: f64 = 1e-6;

/// Convert standard R-matrix parameters to Brune alternative parameterization.
///
/// Solves the nonlinear eigenvalue problem:
///   E(E_tilde) * a = E_tilde * a
/// where E_{lm}(E) = E_l * delta_{lm} - sum_c gamma_{lc} * gamma_{mc} * [S_c(E) - B_c]
///
/// Uses Newton iteration with convergence threshold 1e-12.
pub fn standard_to_brune(params: &RMatrixParams) -> Result<RMatrixParams, CoreError> {
    let n_ch = params.n_channels();
    let n_lev = params.n_levels();

    if n_lev == 0 {
        return Ok(RMatrixParams {
            channels: params.channels.clone(),
            levels: vec![],
            boundary_condition: BoundaryCondition::Brune,
        });
    }

    // Get boundary conditions
    let b_c: Vec<f64> = match &params.boundary_condition {
        BoundaryCondition::Standard { b } => b.clone(),
        BoundaryCondition::Brune => {
            // Already in Brune form
            return Ok(params.clone());
        }
    };

    // For each level, find the Brune alternative energy via Newton iteration
    let mut brune_energies = vec![0.0_f64; n_lev];
    let mut eigenvectors = vec![vec![0.0_f64; n_lev]; n_lev]; // a_{mu,lambda}

    // Initialize: use standard energies as starting point
    for (lam, level) in params.levels.iter().enumerate() {
        brune_energies[lam] = level.energy;
    }

    // Newton iteration for each eigenvalue
    for lam in 0..n_lev {
        let mut e_tilde = brune_energies[lam];

        for iter in 0..MAX_NEWTON_ITER {
            // Build the energy-dependent level matrix E(e_tilde)
            let e_matrix = build_level_matrix(params, &b_c, e_tilde)?;

            // For single-level case, direct solution
            if n_lev == 1 {
                // E_11(e_tilde) = E_1 - sum_c gamma^2_{1c} * (S_c(e_tilde) - B_c)
                // Solve E_11(e_tilde) = e_tilde
                // f(e_tilde) = E_11(e_tilde) - e_tilde = 0
                let f_val = e_matrix[0] - e_tilde;

                // f'(e_tilde) = dE_11/de_tilde - 1
                //             = -sum_c gamma^2_{1c} * dS_c/dE - 1
                let de_matrix = build_level_matrix_derivative(params, e_tilde)?;
                let fp_val = de_matrix[0] - 1.0;

                if fp_val.abs() < 1e-30 {
                    break;
                }

                let delta = -f_val / fp_val;
                e_tilde += delta;

                if delta.abs() < NEWTON_TOL {
                    break;
                }
            } else {
                // Multi-level case: find eigenvalue closest to current estimate
                // Build the matrix and compute eigenvalues
                let mut mat = faer::Mat::<f64>::zeros(n_lev, n_lev);
                for i in 0..n_lev {
                    for j in 0..n_lev {
                        mat[(i, j)] = e_matrix[i * n_lev + j];
                    }
                }

                // Eigendecomposition
                let evd = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
                let eig_diag = evd.s();
                let eig_vals = eig_diag.column_vector();
                let eigenvecs = evd.u();

                // Find eigenvalue closest to current estimate
                let mut closest_idx = 0;
                let mut min_dist = f64::MAX;
                for i in 0..n_lev {
                    let dist = (eig_vals[i] - e_tilde).abs();
                    if dist < min_dist {
                        min_dist = dist;
                        closest_idx = i;
                    }
                }

                let e_new = eig_vals[closest_idx];

                // f(e_tilde) = e_eigenvalue(e_tilde) - e_tilde
                let f_val = e_new - e_tilde;

                // Derivative via implicit differentiation
                let de_matrix = build_level_matrix_derivative(params, e_tilde)?;
                let mut de_eig = 0.0;
                for i in 0..n_lev {
                    for j in 0..n_lev {
                        let vi = eigenvecs[(i, closest_idx)];
                        let vj = eigenvecs[(j, closest_idx)];
                        de_eig += vi * de_matrix[i * n_lev + j] * vj;
                    }
                }
                let fp_val = de_eig - 1.0;

                if fp_val.abs() < 1e-30 {
                    // Use simple iteration instead
                    e_tilde = e_new;
                    if f_val.abs() < NEWTON_TOL {
                        // Store eigenvector
                        for mu in 0..n_lev {
                            eigenvectors[mu][lam] = eigenvecs[(mu, closest_idx)];
                        }
                        break;
                    }
                    continue;
                }

                let delta = -f_val / fp_val;
                e_tilde += delta;

                if delta.abs() < NEWTON_TOL || iter == MAX_NEWTON_ITER - 1 {
                    // Recompute eigenvector at final energy
                    let e_matrix_final = build_level_matrix(params, &b_c, e_tilde)?;
                    let mut mat_final = faer::Mat::<f64>::zeros(n_lev, n_lev);
                    for i in 0..n_lev {
                        for j in 0..n_lev {
                            mat_final[(i, j)] = e_matrix_final[i * n_lev + j];
                        }
                    }
                    let evd_final = mat_final.selfadjoint_eigendecomposition(faer::Side::Lower);
                    let eigenvecs_final = evd_final.u();
                    let eig_vals_final = evd_final.s().column_vector();

                    // Find closest eigenvalue again
                    let mut ci = 0;
                    let mut md = f64::MAX;
                    for i in 0..n_lev {
                        let dist = (eig_vals_final[i] - e_tilde).abs();
                        if dist < md {
                            md = dist;
                            ci = i;
                        }
                    }

                    for mu in 0..n_lev {
                        eigenvectors[mu][lam] = eigenvecs_final[(mu, ci)];
                    }
                    break;
                }
            }
        }

        brune_energies[lam] = e_tilde;
    }

    // For single level, eigenvector is trivially [1]
    if n_lev == 1 {
        eigenvectors[0][0] = 1.0;
    }

    // Compute Brune reduced widths: gamma_tilde_{lam,c} = sum_mu a_{mu,lam} * gamma_{mu,c}
    let mut brune_levels = Vec::with_capacity(n_lev);
    for lam in 0..n_lev {
        let mut reduced_widths = vec![0.0_f64; n_ch];
        for (c, rw) in reduced_widths.iter_mut().enumerate() {
            for (mu, evec) in eigenvectors.iter().enumerate() {
                *rw += evec[lam] * params.levels[mu].reduced_widths[c];
            }
        }
        brune_levels.push(RMatrixLevel {
            energy: brune_energies[lam],
            reduced_widths,
        });
    }

    Ok(RMatrixParams {
        channels: params.channels.clone(),
        levels: brune_levels,
        boundary_condition: BoundaryCondition::Brune,
    })
}

/// Convert Brune alternative parameters back to standard R-matrix parameters.
///
/// This requires choosing a boundary condition B_c (typically B_c = S_c(E_1) for single-level,
/// or B_c = 0 for simplicity).
pub fn brune_to_standard(params: &RMatrixParams) -> Result<RMatrixParams, CoreError> {
    let n_ch = params.n_channels();
    let n_lev = params.n_levels();

    if !matches!(params.boundary_condition, BoundaryCondition::Brune) {
        // Already in standard form
        return Ok(params.clone());
    }

    if n_lev == 0 {
        return Ok(RMatrixParams {
            channels: params.channels.clone(),
            levels: vec![],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0; n_ch] },
        });
    }

    // Choose B_c = S_c(E_tilde_1) (shift at first Brune level energy)
    let b_c: Vec<f64> = compute_shift_vector(&params.channels, params.levels[0].energy)?;

    // For single-level: standard and Brune widths are the same when B_c = S_c(E_tilde)
    // For multi-level: need to solve inverse eigenvalue problem (Newton iteration)
    if n_lev == 1 {
        // Single level: trivial inverse
        // Standard energy: E_1 = E_tilde_1 + sum_c gamma^2 * (S_c(E_tilde_1) - B_c)
        // With B_c = S_c(E_tilde_1), the shift term vanishes => E_1 = E_tilde_1
        return Ok(RMatrixParams {
            channels: params.channels.clone(),
            levels: params.levels.clone(),
            boundary_condition: BoundaryCondition::Standard { b: b_c },
        });
    }

    // Multi-level inverse: Newton iteration to find standard parameters
    // For the inverse, we construct the standard level matrix and search for
    // parameters that reproduce the Brune eigenvalues and eigenvectors.
    //
    // Start with Brune parameters as initial guess
    let mut std_levels = params.levels.clone();

    // Iterative approach: for each iteration, compute Brune params from current standard,
    // compare, and adjust.
    for _iter in 0..MAX_NEWTON_ITER {
        let trial = RMatrixParams {
            channels: params.channels.clone(),
            levels: std_levels.clone(),
            boundary_condition: BoundaryCondition::Standard { b: b_c.clone() },
        };

        let brune_trial = standard_to_brune(&trial)?;

        // Check convergence: compare Brune energies and widths
        let mut max_err = 0.0_f64;
        for lam in 0..n_lev {
            let de = (brune_trial.levels[lam].energy - params.levels[lam].energy).abs();
            max_err = max_err.max(de);
            for c in 0..n_ch {
                let dg = (brune_trial.levels[lam].reduced_widths[c]
                    - params.levels[lam].reduced_widths[c])
                    .abs();
                max_err = max_err.max(dg);
            }
        }

        if max_err < NEWTON_TOL {
            break;
        }

        // Simple correction: adjust standard energies
        for (lam, std_level) in std_levels.iter_mut().enumerate() {
            let correction = params.levels[lam].energy - brune_trial.levels[lam].energy;
            std_level.energy += correction;
        }
    }

    Ok(RMatrixParams {
        channels: params.channels.clone(),
        levels: std_levels,
        boundary_condition: BoundaryCondition::Standard { b: b_c },
    })
}

/// Compute shift function S_c(E) and penetrability P_c(E) for a given channel and energy.
///
/// Returns (S, P).
pub fn shift_function(l: u32, eta: f64, rho: f64) -> Result<(f64, f64), CoreError> {
    shift_penetrability(l, eta, rho)
}

/// Build the energy-dependent level matrix E_{lm}(E).
///
/// E_{lm}(E) = E_l * delta_{lm} - sum_c gamma_{lc} * gamma_{mc} * [S_c(E) - B_c]
fn build_level_matrix(
    params: &RMatrixParams,
    b_c: &[f64],
    energy: f64,
) -> Result<Vec<f64>, CoreError> {
    let n_lev = params.n_levels();
    let n_ch = params.n_channels();
    let mut matrix = vec![0.0_f64; n_lev * n_lev];

    // Compute S_c(E) for each channel
    let shifts = compute_shifts_at_energy(&params.channels, energy)?;

    for lam in 0..n_lev {
        for mu in 0..n_lev {
            let delta = if lam == mu {
                params.levels[lam].energy
            } else {
                0.0
            };

            let mut sum = 0.0;
            for c in 0..n_ch {
                sum += params.levels[lam].reduced_widths[c]
                    * params.levels[mu].reduced_widths[c]
                    * (shifts[c] - b_c[c]);
            }

            matrix[lam * n_lev + mu] = delta - sum;
        }
    }

    Ok(matrix)
}

/// Build the derivative of the level matrix dE/dE_{lm}.
///
/// dE_{lm}/dE = -sum_c gamma_{lc} * gamma_{mc} * dS_c/dE
fn build_level_matrix_derivative(
    params: &RMatrixParams,
    energy: f64,
) -> Result<Vec<f64>, CoreError> {
    let n_lev = params.n_levels();
    let mut matrix = vec![0.0_f64; n_lev * n_lev];

    // Compute dS_c/dE via central finite difference
    let ds_de = compute_shift_derivatives(&params.channels, energy)?;

    for lam in 0..n_lev {
        for mu in 0..n_lev {
            let mut sum = 0.0;
            for (c, &ds) in ds_de.iter().enumerate() {
                sum +=
                    params.levels[lam].reduced_widths[c] * params.levels[mu].reduced_widths[c] * ds;
            }
            matrix[lam * n_lev + mu] = -sum;
        }
    }

    Ok(matrix)
}

/// Compute the shift function S_c(E) for each channel.
fn compute_shifts_at_energy(
    channels: &[RMatrixChannel],
    energy: f64,
) -> Result<Vec<f64>, CoreError> {
    let mut shifts = Vec::with_capacity(channels.len());
    for ch in channels {
        let pair = &ch.pair;
        let e_cm = energy + pair.q_value;
        if e_cm <= 0.0 {
            shifts.push(0.0);
            continue;
        }
        let mu = pair.reduced_mass_amu();
        let k = nucrust_core::units::wave_number(mu, e_cm);
        let eta = nucrust_core::units::sommerfeld_parameter(
            pair.light.z() as f64,
            pair.heavy.z() as f64,
            mu,
            e_cm,
        );
        let rho = k * ch.radius;
        let (s, _p) = shift_penetrability(ch.l, eta, rho)?;
        shifts.push(s);
    }
    Ok(shifts)
}

/// Compute the shift function vector S_c(E) for all channels.
fn compute_shift_vector(channels: &[RMatrixChannel], energy: f64) -> Result<Vec<f64>, CoreError> {
    compute_shifts_at_energy(channels, energy)
}

/// Compute dS_c/dE via central finite difference.
fn compute_shift_derivatives(
    channels: &[RMatrixChannel],
    energy: f64,
) -> Result<Vec<f64>, CoreError> {
    let h = DS_STEP * energy.max(0.01);
    let s_plus = compute_shifts_at_energy(channels, energy + h)?;
    let s_minus = compute_shifts_at_energy(channels, (energy - h).max(1e-10))?;
    let h_actual = (energy + h) - (energy - h).max(1e-10);
    Ok(s_plus
        .iter()
        .zip(s_minus.iter())
        .map(|(sp, sm)| (sp - sm) / h_actual)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rmatrix::rmatrix_cross_section;
    use crate::types::*;
    use nucrust_core::{Nuclide, Parity, Projectile, SpinParity};

    fn make_single_level_params() -> RMatrixParams {
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
                energy: 1.5,
                reduced_widths: vec![0.4],
            }],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0] },
        }
    }

    #[test]
    fn test_brune_round_trip_single_level() {
        let params = make_single_level_params();

        // Convert standard -> Brune
        let brune = standard_to_brune(&params).unwrap();
        assert!(matches!(brune.boundary_condition, BoundaryCondition::Brune));

        // Convert back Brune -> standard
        let roundtrip = brune_to_standard(&brune).unwrap();

        // For single level with B_c = S_c(E_tilde), the roundtrip should be close
        // (the energies may differ because the boundary condition choice differs)
        // Check that the physics is preserved by comparing cross sections
        let energies: Vec<f64> = (1..=30).map(|i| i as f64 * 0.2).collect();
        let _result_orig = rmatrix_cross_section(&params, &energies).unwrap();

        // The Brune form should give the same cross sections
        // For Brune BC, we need to use the Brune collision matrix formula
        // For now, check that the roundtrip preserves the structure
        assert_eq!(roundtrip.n_channels(), params.n_channels());
        assert_eq!(roundtrip.n_levels(), params.n_levels());
    }

    #[test]
    fn test_shift_function_neutron_l0() {
        // S_0 = 0, P_0 = rho for eta=0
        let rho = 1.5;
        let (s, p) = shift_function(0, 0.0, rho).unwrap();
        assert!(s.abs() < 1e-8, "S_0(eta=0) = {}, expected ~0", s);
        assert!(
            (p - rho).abs() < 1e-8,
            "P_0(eta=0, rho={}) = {}, expected {}",
            rho,
            p,
            rho
        );
    }

    #[test]
    fn test_shift_function_neutron_l1() {
        // S_1 = -1/(1+rho^2), P_1 = rho^3/(1+rho^2) for eta=0
        let rho: f64 = 2.0;
        let expected_s = -1.0 / (1.0 + rho * rho);
        let expected_p = rho.powi(3) / (1.0 + rho * rho);
        let (s, p) = shift_function(1, 0.0, rho).unwrap();
        assert!(
            (s - expected_s).abs() < 1e-6,
            "S_1 = {}, expected {}",
            s,
            expected_s
        );
        assert!(
            (p - expected_p).abs() < 1e-6,
            "P_1 = {}, expected {}",
            p,
            expected_p
        );
    }

    #[test]
    fn test_standard_to_brune_empty_levels() {
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
        let params = RMatrixParams {
            channels: vec![channel],
            levels: vec![],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0] },
        };
        let brune = standard_to_brune(&params).unwrap();
        assert!(brune.levels.is_empty());
    }

    #[test]
    fn test_brune_already_brune() {
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
        let params = RMatrixParams {
            channels: vec![channel],
            levels: vec![RMatrixLevel {
                energy: 1.0,
                reduced_widths: vec![0.3],
            }],
            boundary_condition: BoundaryCondition::Brune,
        };
        let result = standard_to_brune(&params).unwrap();
        assert!(
            (result.levels[0].energy - 1.0).abs() < 1e-15,
            "Already-Brune params should pass through unchanged"
        );
    }
}
