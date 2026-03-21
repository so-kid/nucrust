//! R-matrix parameter fitting: Levenberg-Marquardt and MCMC.
//!
//! - Levenberg-Marquardt: gradient-based chi-squared minimization with
//!   central finite-difference Jacobian
//! - MCMC: Goodman-Weare affine-invariant ensemble sampler

use nucrust_core::CoreError;

use crate::rmatrix::rmatrix_cross_section;
use crate::types::{
    ExperimentalData, FitResult, LmConfig, McmcConfig, McmcResult, ParamIndex, RMatrixParams,
};

/// Step size for finite-difference Jacobian (relative).
const FD_STEP_REL: f64 = 1e-5;
/// Minimum absolute step size for finite-difference Jacobian.
const FD_STEP_MIN: f64 = 1e-10;

/// Compute chi-squared for given parameters and experimental data.
fn chi_squared(params: &RMatrixParams, exp_data: &ExperimentalData) -> Result<f64, CoreError> {
    let result = rmatrix_cross_section(params, &exp_data.energies)?;
    let sigma_calc = &result.cross_sections.sigma_total;

    let chi2: f64 = sigma_calc
        .iter()
        .zip(exp_data.cross_sections.iter())
        .zip(exp_data.errors.iter())
        .map(|((&sc, &se), &err)| {
            let residual = (sc - se) / err;
            residual * residual
        })
        .sum();
    Ok(chi2)
}

/// Compute the residual vector r_i = (sigma_calc(E_i) - sigma_exp(E_i)) / delta_sigma_i.
fn residuals(params: &RMatrixParams, exp_data: &ExperimentalData) -> Result<Vec<f64>, CoreError> {
    let result = rmatrix_cross_section(params, &exp_data.energies)?;
    let sigma_calc = &result.cross_sections.sigma_total;

    Ok(exp_data
        .energies
        .iter()
        .enumerate()
        .map(|(i, _)| (sigma_calc[i] - exp_data.cross_sections[i]) / exp_data.errors[i])
        .collect())
}

/// Compute the Jacobian matrix J_{ij} = d(r_i)/d(p_j) via central finite differences.
fn compute_jacobian(
    params: &RMatrixParams,
    free_params: &[ParamIndex],
    exp_data: &ExperimentalData,
) -> Result<Vec<Vec<f64>>, CoreError> {
    let n_data = exp_data.energies.len();
    let n_params = free_params.len();
    let mut jacobian = vec![vec![0.0_f64; n_params]; n_data];

    for (j, idx) in free_params.iter().enumerate() {
        let p0 = params.get_param(idx);
        let h = (p0.abs() * FD_STEP_REL).max(FD_STEP_MIN);

        // Forward step
        let mut params_plus = params.clone();
        params_plus.set_param(idx, p0 + h);
        let r_plus = residuals(&params_plus, exp_data)?;

        // Backward step
        let mut params_minus = params.clone();
        params_minus.set_param(idx, p0 - h);
        let r_minus = residuals(&params_minus, exp_data)?;

        // Central difference
        for i in 0..n_data {
            jacobian[i][j] = (r_plus[i] - r_minus[i]) / (2.0 * h);
        }
    }

    Ok(jacobian)
}

/// Levenberg-Marquardt algorithm for R-matrix parameter optimization.
///
/// Minimizes chi^2 = sum_i [(sigma_calc(E_i) - sigma_exp(E_i)) / delta_sigma_i]^2
///
/// Uses central finite-difference Jacobian and standard LM damping factor adaptation.
pub fn levenberg_marquardt(
    params: &mut RMatrixParams,
    free_params: &[ParamIndex],
    exp_data: &ExperimentalData,
    config: &LmConfig,
) -> Result<FitResult, CoreError> {
    let n_data = exp_data.energies.len();
    let n_params = free_params.len();

    if n_data <= n_params {
        return Err(CoreError::InvalidParameter {
            name: "n_data",
            value: n_data as f64,
            reason: "need more data points than free parameters",
        });
    }

    let mut lambda = config.lambda_init;
    let mut chi2 = chi_squared(params, exp_data)?;
    let mut converged = false;
    let mut iteration = 0_u32;

    for iter in 0..config.max_iterations {
        iteration = iter + 1;

        // Compute Jacobian
        let jac = compute_jacobian(params, free_params, exp_data)?;

        // Compute residuals
        let r = residuals(params, exp_data)?;

        // Build J^T * J (Hessian approximation) and J^T * r (gradient)
        let mut jtj = vec![0.0_f64; n_params * n_params];
        let mut jtr = vec![0.0_f64; n_params];

        for j in 0..n_params {
            for k in 0..n_params {
                let sum: f64 = jac.iter().map(|row| row[j] * row[k]).sum();
                jtj[j * n_params + k] = sum;
            }
            let sum: f64 = jac.iter().zip(r.iter()).map(|(row, &ri)| row[j] * ri).sum();
            jtr[j] = sum;
        }

        // Apply damping: (J^T*J + lambda*diag(J^T*J)) * delta_p = -J^T*r
        let mut augmented = jtj.clone();
        for j in 0..n_params {
            augmented[j * n_params + j] *= 1.0 + lambda;
        }

        // Solve the linear system using faer
        let delta_p = solve_linear_system(n_params, &augmented, &jtr)?;

        // Try the step
        let mut params_trial = params.clone();
        for (j, idx) in free_params.iter().enumerate() {
            let p0 = params.get_param(idx);
            params_trial.set_param(idx, p0 - delta_p[j]);
        }

        let chi2_trial = chi_squared(&params_trial, exp_data)?;

        if chi2_trial < chi2 {
            // Accept step, reduce lambda
            let chi2_change = (chi2 - chi2_trial) / chi2.max(1e-30);
            chi2 = chi2_trial;
            *params = params_trial;
            lambda /= config.lambda_factor;

            // Check convergence
            let max_dp = delta_p
                .iter()
                .zip(free_params.iter())
                .map(|(dp, idx)| {
                    let p = params.get_param(idx);
                    dp.abs() / p.abs().max(1e-30)
                })
                .fold(0.0_f64, f64::max);

            if chi2_change < config.convergence_chi2 && max_dp < config.convergence_params {
                converged = true;
                break;
            }
        } else {
            // Reject step, increase lambda
            lambda *= config.lambda_factor;
        }
    }

    // Compute covariance matrix: (J^T * J)^{-1}
    let jac = compute_jacobian(params, free_params, exp_data)?;
    let mut jtj = vec![0.0_f64; n_params * n_params];
    for j in 0..n_params {
        for k in 0..n_params {
            let sum: f64 = jac.iter().map(|row| row[j] * row[k]).sum();
            jtj[j * n_params + k] = sum;
        }
    }

    // Invert J^T*J for covariance
    let covariance =
        invert_matrix(n_params, &jtj).unwrap_or_else(|_| vec![0.0; n_params * n_params]);

    let dof = (n_data - n_params) as f64;
    let param_values: Vec<f64> = free_params
        .iter()
        .map(|idx| params.get_param(idx))
        .collect();

    Ok(FitResult {
        chi_squared: chi2,
        reduced_chi_squared: chi2 / dof,
        iterations: iteration,
        converged,
        covariance,
        param_values,
    })
}

/// Solve a linear system A*x = b using faer LU decomposition.
fn solve_linear_system(n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, CoreError> {
    use faer::prelude::*;

    let mut mat_a = faer::Mat::<f64>::zeros(n, n);
    let mut mat_b = faer::Mat::<f64>::zeros(n, 1);

    for i in 0..n {
        for j in 0..n {
            mat_a[(i, j)] = a[i * n + j];
        }
        mat_b[(i, 0)] = b[i];
    }

    let lu = mat_a.partial_piv_lu();
    let x = lu.solve(&mat_b);

    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = x[(i, 0)];
    }

    Ok(result)
}

/// Invert a matrix using faer LU decomposition.
fn invert_matrix(n: usize, a: &[f64]) -> Result<Vec<f64>, CoreError> {
    use faer::prelude::*;

    let mut mat_a = faer::Mat::<f64>::zeros(n, n);
    let mut identity = faer::Mat::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            mat_a[(i, j)] = a[i * n + j];
        }
        identity[(i, i)] = 1.0;
    }

    let lu = mat_a.partial_piv_lu();
    let inv = lu.solve(&identity);

    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = inv[(i, j)];
        }
    }

    Ok(result)
}

/// Goodman-Weare affine-invariant ensemble MCMC sampler.
///
/// Implements the stretch move from Goodman & Weare (2010).
/// Each step:
/// 1. For each walker, propose a new position via stretch move
/// 2. Compute log-likelihood at proposed position
/// 3. Accept/reject via Metropolis-Hastings criterion
pub fn mcmc_sample(
    params: &RMatrixParams,
    free_params: &[ParamIndex],
    exp_data: &ExperimentalData,
    config: &McmcConfig,
) -> Result<McmcResult, CoreError> {
    let n_params = free_params.len();
    let n_walkers = config.n_walkers as usize;
    let n_samples = config.n_samples as usize;

    if n_walkers < 2 * n_params {
        return Err(CoreError::InvalidParameter {
            name: "n_walkers",
            value: n_walkers as f64,
            reason: "need at least 2 * n_params walkers",
        });
    }

    // Initialize walkers around the current parameter values
    let mut walker_positions = vec![vec![0.0_f64; n_params]; n_walkers];
    let mut walker_log_like = vec![0.0_f64; n_walkers];

    // Simple seeded RNG (xorshift64)
    let mut rng_state = 42u64;
    let mut rand_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };

    // Initialize walkers with small perturbations
    for w in 0..n_walkers {
        for (j, idx) in free_params.iter().enumerate() {
            let p0 = params.get_param(idx);
            let perturbation = p0 * 0.01 * (2.0 * rand_f64() - 1.0);
            walker_positions[w][j] = p0 + perturbation;
        }

        // Compute initial log-likelihood
        let mut trial_params = params.clone();
        for (j, idx) in free_params.iter().enumerate() {
            trial_params.set_param(idx, walker_positions[w][j]);
        }
        let chi2 = chi_squared(&trial_params, exp_data).unwrap_or(f64::MAX);
        walker_log_like[w] = -0.5 * chi2;
    }

    // Storage for chains
    let mut chains = vec![vec![Vec::new(); n_samples]; n_walkers];
    let mut log_likes = vec![vec![0.0_f64; n_samples]; n_walkers];
    let mut n_accepted = 0_u64;
    let mut n_total = 0_u64;

    // MCMC loop
    for sample in 0..n_samples {
        for w in 0..n_walkers {
            // Pick a random complementary walker
            let mut w_comp = w;
            while w_comp == w {
                w_comp = (rand_f64() * n_walkers as f64) as usize;
                w_comp = w_comp.min(n_walkers - 1);
            }

            // Generate stretch factor z from g(z) ~ 1/sqrt(z) on [1/a, a]
            let a = config.stretch_scale;
            let u = rand_f64();
            let z = ((a - 1.0) * u + 1.0).powi(2) / a;

            // Propose new position: x_new = x_comp + z * (x_w - x_comp)
            let mut proposed = vec![0.0_f64; n_params];
            for j in 0..n_params {
                proposed[j] = walker_positions[w_comp][j]
                    + z * (walker_positions[w][j] - walker_positions[w_comp][j]);
            }

            // Compute log-likelihood at proposed position
            let mut trial_params = params.clone();
            for (j, idx) in free_params.iter().enumerate() {
                trial_params.set_param(idx, proposed[j]);
            }
            let chi2_proposed = chi_squared(&trial_params, exp_data).unwrap_or(f64::MAX);
            let log_like_proposed = -0.5 * chi2_proposed;

            // Acceptance criterion
            let log_alpha =
                (n_params as f64 - 1.0) * z.ln() + log_like_proposed - walker_log_like[w];

            n_total += 1;
            if log_alpha.is_finite() && rand_f64().ln() < log_alpha {
                walker_positions[w] = proposed;
                walker_log_like[w] = log_like_proposed;
                n_accepted += 1;
            }

            // Store sample
            chains[w][sample] = walker_positions[w].clone();
            log_likes[w][sample] = walker_log_like[w];
        }
    }

    // Compute summary statistics (after burn-in)
    let burn_in = config.burn_in as usize;
    let mut param_sums = vec![0.0_f64; n_params];
    let mut param_sq_sums = vec![0.0_f64; n_params];
    let mut count = 0_u64;

    for walker_chain in chains.iter() {
        for sample in walker_chain.iter().skip(burn_in) {
            for (j, &val) in sample.iter().enumerate() {
                param_sums[j] += val;
                param_sq_sums[j] += val * val;
            }
            count += 1;
        }
    }

    let param_means: Vec<f64> = param_sums.iter().map(|&s| s / count as f64).collect();
    let param_stds: Vec<f64> = param_sq_sums
        .iter()
        .zip(param_means.iter())
        .map(|(&sq, &m)| (sq / count as f64 - m * m).max(0.0).sqrt())
        .collect();

    let acceptance_rate = if n_total > 0 {
        n_accepted as f64 / n_total as f64
    } else {
        0.0
    };

    Ok(McmcResult {
        chains: Some(chains),
        log_likelihood: Some(log_likes),
        acceptance_rate,
        output_path: None,
        param_means,
        param_stds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use nucrust_core::{Nuclide, Parity, Projectile, SpinParity};

    fn make_test_params() -> RMatrixParams {
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
                reduced_widths: vec![0.3],
            }],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0] },
        }
    }

    fn generate_synthetic_data(params: &RMatrixParams) -> ExperimentalData {
        let energies: Vec<f64> = (1..=30).map(|i| i as f64 * 0.2).collect();
        let result = rmatrix_cross_section(params, &energies).unwrap();
        let cross_sections = result.cross_sections.sigma_total.clone();
        let errors: Vec<f64> = cross_sections
            .iter()
            .map(|&s| (s * 0.05).max(0.01))
            .collect();
        ExperimentalData {
            energies,
            cross_sections,
            errors,
        }
    }

    #[test]
    fn test_lm_convergence_synthetic() {
        let true_params = make_test_params();
        let exp_data = generate_synthetic_data(&true_params);

        // First verify: at exact parameters, chi^2 should be ~0
        let chi2_exact = {
            let result = rmatrix_cross_section(&true_params, &exp_data.energies).unwrap();
            let sigma = &result.cross_sections.sigma_total;
            let mut c2 = 0.0;
            for (i, (&sc, &se)) in sigma.iter().zip(exp_data.cross_sections.iter()).enumerate() {
                c2 += ((sc - se) / exp_data.errors[i]).powi(2);
            }
            c2
        };
        assert!(
            chi2_exact < 1e-10,
            "Chi^2 at true params should be ~0, got {}",
            chi2_exact
        );

        // Start with a small perturbation in the reduced width (smoother landscape)
        let mut fit_params = true_params.clone();
        fit_params.levels[0].reduced_widths[0] = 0.31; // small perturbation from 0.3

        let free_params = vec![ParamIndex::ReducedWidth {
            level_idx: 0,
            channel_idx: 0,
        }];

        let config = LmConfig {
            max_iterations: 200,
            lambda_init: 1e-3,
            lambda_factor: 10.0,
            convergence_chi2: 1e-8,
            convergence_params: 1e-8,
        };

        let result =
            levenberg_marquardt(&mut fit_params, &free_params, &exp_data, &config).unwrap();

        // The fit should converge
        assert!(
            result.converged || result.chi_squared < 1.0,
            "LM should converge: chi^2={}, iterations={}, converged={}",
            result.chi_squared,
            result.iterations,
            result.converged,
        );

        // Width should be close to true value
        let fitted_width = fit_params.levels[0].reduced_widths[0];
        assert!(
            (fitted_width - 0.3).abs() < 0.05,
            "Fitted width {} should be close to true value 0.3",
            fitted_width
        );
    }

    #[test]
    fn test_lm_too_few_data() {
        let mut params = make_test_params();
        let exp_data = ExperimentalData {
            energies: vec![1.0],
            cross_sections: vec![100.0],
            errors: vec![10.0],
        };

        let free_params = vec![
            ParamIndex::LevelEnergy { level_idx: 0 },
            ParamIndex::ReducedWidth {
                level_idx: 0,
                channel_idx: 0,
            },
        ];

        let result =
            levenberg_marquardt(&mut params, &free_params, &exp_data, &LmConfig::default());
        assert!(result.is_err(), "Should error with too few data points");
    }

    #[test]
    fn test_mcmc_basic() {
        let params = make_test_params();
        let exp_data = generate_synthetic_data(&params);

        let free_params = vec![ParamIndex::LevelEnergy { level_idx: 0 }];

        let config = McmcConfig {
            n_walkers: 8,
            n_samples: 100,
            burn_in: 20,
            stretch_scale: 2.0,
            stream_output: None,
        };

        let result = mcmc_sample(&params, &free_params, &exp_data, &config).unwrap();

        // Basic structural checks
        assert_eq!(result.param_means.len(), 1);
        assert_eq!(result.param_stds.len(), 1);
        assert!(result.param_means[0].is_finite(), "Mean should be finite");
        assert!(result.param_stds[0].is_finite(), "Std should be finite");
        assert!(result.param_stds[0] >= 0.0, "Std should be non-negative");

        // Check that chains were stored
        let chains = result.chains.as_ref().unwrap();
        assert_eq!(chains.len(), 8); // n_walkers
        assert_eq!(chains[0].len(), 100); // n_samples
    }

    #[test]
    fn test_mcmc_too_few_walkers() {
        let params = make_test_params();
        let exp_data = generate_synthetic_data(&params);
        let free_params = vec![
            ParamIndex::LevelEnergy { level_idx: 0 },
            ParamIndex::ReducedWidth {
                level_idx: 0,
                channel_idx: 0,
            },
        ];

        let config = McmcConfig {
            n_walkers: 2, // Need at least 2*2=4
            n_samples: 10,
            burn_in: 0,
            stretch_scale: 2.0,
            stream_output: None,
        };

        let result = mcmc_sample(&params, &free_params, &exp_data, &config);
        assert!(result.is_err(), "Should error with too few walkers");
    }
}
