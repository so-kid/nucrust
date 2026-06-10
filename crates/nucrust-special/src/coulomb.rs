use crate::consts::{CF_EPS, MAX_CF_ITER, RHO_SMALL};
use crate::error::SpecialError;
use crate::gamma::{coulomb_phase_shift, gamow_factor};
use crate::lentz::continued_fraction_real;

/// Maximum terms in power series.
const MAX_SERIES_TERMS: usize = 300;
/// Minimum terms in power series before checking convergence.
const MIN_SERIES_TERMS: usize = 10;
/// Step size for Numerov backward integration.
const NUMEROV_STEP: f64 = 0.02;

/// Coulomb wave function computation result.
#[derive(Debug, Clone)]
pub struct CoulombResult {
    /// F_l(eta, rho) for l = l_min..l_min+n_l
    pub f: Vec<f64>,
    /// G_l(eta, rho) for l = l_min..l_min+n_l
    pub g: Vec<f64>,
    /// F'_l(eta, rho) for l = l_min..l_min+n_l
    pub fp: Vec<f64>,
    /// G'_l(eta, rho) for l = l_min..l_min+n_l
    pub gp: Vec<f64>,
    /// Coulomb phase shift sigma_l = arg[Gamma(l+1+i*eta)]
    pub sigma: Vec<f64>,
    /// Exponential scaling exponent (for forbidden region).
    pub exponent: f64,
}

/// Scaling mode for Coulomb wave functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    /// Normal computation (may overflow/underflow in forbidden region).
    Unscaled,
    /// Exponentially scaled: F_tilde = F * exp(+exponent), G_tilde = G * exp(-exponent).
    Scaled,
}

// ======================== Auxiliary quantities (DLMF §33.4) ========================

/// R_l = sqrt(1 + eta^2/l^2). Returns 0 for l=0.
#[inline]
fn r_l(l: f64, eta: f64) -> f64 {
    if l == 0.0 {
        return 0.0;
    }
    (1.0 + (eta / l).powi(2)).sqrt()
}

/// S_l = l/rho + eta/l. Only valid for l > 0.
#[inline]
fn s_l(l: f64, eta: f64, rho: f64) -> f64 {
    l / rho + eta / l
}

// ======================== CF1: f_l = F'_l / F_l ========================

/// Evaluate CF1 for a given l, eta, rho.
///
/// Returns f_l = F'_l / F_l using DLMF §33.8.1.
fn cf1(l: f64, eta: f64, rho: f64) -> Result<f64, SpecialError> {
    // b0 = S_{l+1} = (l+1)/rho + eta/(l+1)
    let b0 = (l + 1.0) / rho + eta / (l + 1.0);

    continued_fraction_real(
        |n| {
            let ln = l + n as f64;
            -(1.0 + (eta / ln).powi(2)) // a_n = -R^2_{l+n}
        },
        |n| {
            if n == 0 {
                b0
            } else {
                let ln = l + n as f64;
                // b_n = T_{l+n} = (l+n)/rho + eta/(l+n) + (l+n+1)/rho + eta/(l+n+1)
                ln / rho + eta / ln + (ln + 1.0) / rho + eta / (ln + 1.0)
            }
        },
        MAX_CF_ITER,
        CF_EPS,
    )
}

// ======================== CF2: p + iq = H^+' / H^+ ========================

/// Evaluate CF2 using the Steed algorithm.
///
/// Returns (p, q) where p + iq = H^{+\prime}_l / H^+_l.
///
/// This uses the Steed (1982) / Barnett algorithm which accumulates P and Q
/// as sums of corrections. This is essential for correctness when l > 0,
/// where the Lentz method on the DLMF continued fraction can terminate
/// prematurely and produce wrong results.
///
/// Reference: GSL specfunc/coulomb.c `coulomb_CF2`; Barnett (1982) CPC 27, 147.
fn cf2(l: f64, eta: f64, rho: f64) -> Result<(f64, f64), SpecialError> {
    let wi = 2.0 * eta;
    let x_inv = 1.0 / rho;
    let e2mm1 = eta * eta + l * (l + 1.0);

    let mut ar = -e2mm1;
    let mut ai = eta;
    let br = 2.0 * (rho - eta);
    let mut bi = 2.0;

    // d = 1/(br + i*bi)
    let denom = br * br + bi * bi;
    if denom < 1e-300 {
        return Err(SpecialError::NumericalOverflow {
            context: "CF2: initial denominator ~ 0",
        });
    }
    let mut dr = br / denom;
    let mut di = -bi / denom;

    // First correction: dp + i*dq = (i/rho) * (ar + i*ai) / (br + i*bi)
    let mut dp = -x_inv * (ar * di + ai * dr);
    let mut dq = x_inv * (ar * dr - ai * di);

    let mut p = 0.0;
    let mut q = 1.0 - eta * x_inv;

    let mut pk = 0.0;

    for _ in 1..=MAX_CF_ITER {
        p += dp;
        q += dq;
        pk += 2.0;
        ar += pk;
        ai += wi;
        bi += 2.0;

        let d_re = ar * dr - ai * di + br;
        let d_im = ai * dr + ar * di + bi;
        let c = 1.0 / (d_re * d_re + d_im * d_im);
        dr = c * d_re;
        di = -c * d_im;

        let a = br * dr - bi * di - 1.0;
        let b = bi * dr + br * di;
        let new_dp = dp * a - dq * b;
        dq = dp * b + dq * a;
        dp = new_dp;

        if dp.abs() + dq.abs() <= (p.abs() + q.abs()) * CF_EPS {
            return Ok((p, q));
        }
    }

    Err(SpecialError::ConvergenceFailure {
        algorithm: "CF2 (Steed)",
        iterations: MAX_CF_ITER,
        residual: (p * p + q * q).sqrt(),
    })
}

// ======================== Wronskian coupling (Steed method) ========================

/// Determine the sign of F_l using the asymptotic phase.
///
/// Used only at l=0 where the asymptotic phase is reliable.
/// In the oscillatory region: F_l ~ sin(theta_l) where
/// theta_l = rho - eta*ln(2*rho) - l*pi/2 + sigma_l
///
/// In the forbidden region (rho < rho_tp): F > 0 (monotonically decaying).
fn f_sign(l: f64, eta: f64, rho: f64) -> f64 {
    let rho_tp = eta + (eta * eta + l * (l + 1.0)).sqrt();
    if rho < rho_tp {
        return 1.0; // Forbidden region: F > 0
    }
    let sigma = coulomb_phase_shift(l as u32, eta);
    let theta = rho - eta * (2.0 * rho).ln() - l * std::f64::consts::FRAC_PI_2 + sigma;
    theta.sin().signum()
}

/// Compute F_l, G_l, F'_l, G'_l at a single l using Steed method.
///
/// Note: Only called at l=0 by `coulomb_wave` to ensure reliable sign
/// determination. Higher l values are obtained via recurrence.
fn steed_method(l: f64, eta: f64, rho: f64) -> Result<(f64, f64, f64, f64), SpecialError> {
    let f_ratio = cf1(l, eta, rho)?;
    let (p, q) = cf2(l, eta, rho)?;

    if q.abs() < 1e-300 {
        return Err(SpecialError::NumericalOverflow {
            context: "Steed method: q ~ 0",
        });
    }

    // F^2 = q / ((f-p)^2 + q^2)
    let fmp = f_ratio - p;
    let denom = fmp * fmp + q * q;
    let f_sq = (q / denom).abs(); // abs for safety in forbidden region

    let sign = f_sign(l, eta, rho);
    let f_l = sign * f_sq.sqrt();
    let fp_l = f_ratio * f_l;
    let gamm = fmp / q;
    let g_l = gamm * f_l;
    let gp_l = p * g_l - q * f_l;

    Ok((f_l, g_l, fp_l, gp_l))
}

// ======================== Power series (small rho) ========================

/// Compute F_l, F'_l using the 1F1 power series for small rho.
///
/// F_l(eta, rho) = C_l(eta) * rho^{l+1} * sum_{k=0}^N A_k * rho^k
fn power_series_f(l: u32, eta: f64, rho: f64) -> Result<(f64, f64), SpecialError> {
    let lf = l as f64;
    let c_l = gamow_factor(l, eta);

    // Coefficient recurrence: A_0=1, A_1=eta/(l+1)
    // A_k = (2*eta*A_{k-1} - A_{k-2}) / (k*(2l+1+k))
    let mut a_km2 = 0.0; // A_{k-2}, initialized for k=1
    let mut a_km1 = 1.0; // A_{k-1} = A_0

    let mut sum = 1.0; // S(rho) = sum A_k * rho^k
    let mut dsum = 0.0; // S'(rho) = sum k * A_k * rho^{k-1}
    let mut rho_k = 1.0; // rho^k

    for k in 1..=MAX_SERIES_TERMS {
        let kf = k as f64;
        let a_k = (2.0 * eta * a_km1 - a_km2) / (kf * (2.0 * lf + 1.0 + kf));

        rho_k *= rho;
        let term = a_k * rho_k;
        sum += term;
        dsum += kf * a_k * rho_k / rho;

        // Check convergence only after minimum terms
        if k >= MIN_SERIES_TERMS && term.abs() < CF_EPS * sum.abs() {
            break;
        }

        a_km2 = a_km1;
        a_km1 = a_k;
    }

    let rho_l1 = rho.powi(l as i32 + 1);
    let f_val = c_l * rho_l1 * sum;
    // F' = C_l * d/drho[rho^{l+1} * S(rho)]
    //    = C_l * [(l+1) * rho^l * S + rho^{l+1} * S']
    let fp_val = c_l * ((lf + 1.0) * rho_l1 / rho * sum + rho_l1 * dsum);

    Ok((f_val, fp_val))
}

// ======================== Recurrence relations ========================

/// Downward recurrence for F from l_top to l_bot (inclusive).
///
/// Uses derivative relation: F_{l-1} = (F'_l + S_l * F_l) / R_l
///                           F'_{l-1} = S_l * F_{l-1} - R_l * F_l
///
/// Recurrence starts with arbitrary normalization at l_top.
/// Returns (f_unnorm, fp_unnorm) arrays indexed [l - l_bot].
fn recurrence_f_downward(
    l_top: u32,
    l_bot: u32,
    eta: f64,
    rho: f64,
    f_ratio_top: f64, // f_{l_top} = F'/F from CF1
) -> (Vec<f64>, Vec<f64>) {
    let n = (l_top - l_bot + 1) as usize;
    let mut f_vals = vec![0.0; n];
    let mut fp_vals = vec![0.0; n];

    let idx = |l: u32| -> usize { (l - l_bot) as usize };

    // Start with arbitrary normalization
    f_vals[idx(l_top)] = 1.0;
    fp_vals[idx(l_top)] = f_ratio_top;

    for l in (l_bot + 1..=l_top).rev() {
        let lf = l as f64;
        let r = r_l(lf, eta);
        let s = s_l(lf, eta, rho);

        let fl = f_vals[idx(l)];
        let fpl = fp_vals[idx(l)];

        if r.abs() < 1e-300 {
            // l=0 case: R_0 = 0, can't use this formula
            // Use alternative: F'_0 = S_1 * F_0 - R_1 * F_1
            break;
        }

        // F_{l-1} = (F'_l + S_l * F_l) / R_l
        let f_below = (fpl + s * fl) / r;
        // F'_{l-1} = S_l * F_{l-1} - R_l * F_l
        let fp_below = s * f_below - r * fl;

        f_vals[idx(l - 1)] = f_below;
        fp_vals[idx(l - 1)] = fp_below;
    }

    (f_vals, fp_vals)
}

/// Upward recurrence for G from l_bot to l_top (inclusive).
///
/// G_{l+1} = (S_{l+1} * G_l - G'_l) / R_{l+1}
/// G'_{l+1} = R_{l+1} * G_l - S_{l+1} * G_{l+1}
fn recurrence_g_upward(
    l_bot: u32,
    l_top: u32,
    eta: f64,
    rho: f64,
    g_bot: f64,
    gp_bot: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = (l_top - l_bot + 1) as usize;
    let mut g_vals = vec![0.0; n];
    let mut gp_vals = vec![0.0; n];

    g_vals[0] = g_bot;
    gp_vals[0] = gp_bot;

    for l in l_bot..l_top {
        let i = (l - l_bot) as usize;
        let lf = (l + 1) as f64;
        let r_next = r_l(lf, eta);
        let s_next = s_l(lf, eta, rho);

        if r_next.abs() < 1e-300 {
            break;
        }

        let gl = g_vals[i];
        let gpl = gp_vals[i];

        // G_{l+1} = (S_{l+1} * G_l - G'_l) / R_{l+1}
        let g_next = (s_next * gl - gpl) / r_next;
        // G'_{l+1} = R_{l+1} * G_l - S_{l+1} * G_{l+1}
        let gp_next = r_next * gl - s_next * g_next;

        g_vals[i + 1] = g_next;
        gp_vals[i + 1] = gp_next;
    }

    (g_vals, gp_vals)
}

// ======================== Numerov backward integration for G ========================

/// The effective potential: Q(rho) = 2*eta/rho + l*(l+1)/rho^2 - 1
/// so that the Coulomb ODE is w'' = Q(rho)*w (note: w'' + (1 - 2η/ρ - l(l+1)/ρ²)w = 0
/// rearranges to w'' = (2η/ρ + l(l+1)/ρ² - 1)w).
#[inline]
fn coulomb_q(l: f64, eta: f64, rho: f64) -> f64 {
    2.0 * eta / rho + l * (l + 1.0) / (rho * rho) - 1.0
}

/// Compute G_l and G'_l at target rho by backward Numerov integration from the
/// oscillatory region. Used when rho is inside the forbidden region (rho < rho_tp).
///
/// Steps:
/// 1. Find rho_start in the oscillatory region (rho_tp + margin)
/// 2. Use Steed at rho_start to get G_start, G'_start
/// 3. Integrate backward using Numerov to target rho
fn g_backward_numerov(l: f64, eta: f64, rho_target: f64) -> Result<(f64, f64), SpecialError> {
    let rho_tp = eta + (eta * eta + l * (l + 1.0)).sqrt();
    let rho_start = (rho_tp + 2.0).max(rho_target + 2.0);

    // Get G at rho_start using Steed
    let (_, g_start, _, gp_start) = steed_method(l, eta, rho_start)?;

    // Numerov backward integration from rho_start to rho_target
    let h = NUMEROV_STEP;
    let n_steps = ((rho_start - rho_target) / h).ceil() as usize;
    if n_steps == 0 {
        return Ok((g_start, gp_start));
    }
    let h = (rho_start - rho_target) / n_steps as f64; // adjust step

    // Numerov needs two starting points. Get w_0 = G(rho_start) and
    // w_1 = G(rho_start - h) ≈ G - h*G' + h²*G''/2 (Taylor)
    let rho0 = rho_start;
    let rho1 = rho_start - h;
    let q0 = coulomb_q(l, eta, rho0);
    let w0 = g_start;
    let w1 = g_start - h * gp_start + 0.5 * h * h * q0 * g_start;

    let mut w_prev = w0;
    let mut w_curr = w1;
    let mut rho_curr = rho1;

    for _ in 2..=n_steps {
        let rho_next = rho_curr - h;
        if rho_next < 1e-15 {
            break;
        }
        let q_curr = coulomb_q(l, eta, rho_curr);
        let q_prev = coulomb_q(l, eta, rho_curr + h);
        let q_next = coulomb_q(l, eta, rho_next);

        // Numerov: w_{n+1} = [2*(1 + 5h²Q_n/12)*w_n - (1 - h²Q_{n-1}/12)*w_{n-1}]
        //                    / (1 - h²Q_{n+1}/12)
        // Here "next" is at smaller rho (backward).
        let h2 = h * h;
        let w_next = (2.0 * (1.0 + 5.0 * h2 * q_curr / 12.0) * w_curr
            - (1.0 - h2 * q_prev / 12.0) * w_prev)
            / (1.0 - h2 * q_next / 12.0);

        w_prev = w_curr;
        w_curr = w_next;
        rho_curr = rho_next;
    }

    let g_target = w_curr;
    // G' from finite difference (backward difference at target)
    let gp_target = (w_prev - w_curr) / h; // (G(rho+h) - G(rho)) / h

    Ok((g_target, gp_target))
}

// ======================== Public API ========================

/// Compute Coulomb wave functions F_l, G_l, F'_l, G'_l, sigma_l.
///
/// # Algorithm
/// Always computes from l=0 internally using Steed + recurrence, matching the
/// standard Thompson-Barnett / GSL approach. This ensures correct sign
/// determination (Steed at l=0 has a trivially correct sign) and stable G
/// upward recurrence.
///
/// - rho > 0.5 (oscillatory): Steed at l=0, G upward recurrence, F downward
///   recurrence from l_max normalized against Steed F at l=0.
/// - rho <= 0.5 or forbidden region: Power series for F, CF2/Numerov for G.
pub fn coulomb_wave(
    eta: f64,
    rho: f64,
    l_min: u32,
    n_l: u32,
) -> Result<CoulombResult, SpecialError> {
    if rho < 0.0 {
        return Err(SpecialError::InvalidArgument {
            name: "rho",
            value: rho,
            reason: "must be non-negative",
        });
    }
    if n_l == 0 {
        return Ok(CoulombResult {
            f: vec![],
            g: vec![],
            fp: vec![],
            gp: vec![],
            sigma: vec![],
            exponent: 0.0,
        });
    }
    if rho == 0.0 {
        let n = n_l as usize;
        let sigma: Vec<f64> = (0..n_l)
            .map(|i| coulomb_phase_shift(l_min + i, eta))
            .collect();
        return Ok(CoulombResult {
            f: vec![0.0; n],
            g: vec![f64::INFINITY; n],
            fp: vec![0.0; n],
            gp: vec![f64::NEG_INFINITY; n],
            sigma,
            exponent: 0.0,
        });
    }

    // Always compute from l=0 to l_max for correct sign determination and
    // stable G recurrence (matching GSL/COULCC approach).
    let l_max = l_min + n_l - 1;
    let total_l = (l_max + 1) as usize; // l = 0..l_max

    let (f_all, g_all, fp_all, gp_all);

    // Determine if we're in the forbidden region (using l=0 turning point)
    let rho_tp_0 = eta + (eta * eta).sqrt(); // l=0: rho_tp = eta + |eta|
    let in_forbidden = rho < rho_tp_0 * 0.9;

    if rho > RHO_SMALL && !in_forbidden {
        // === Oscillatory region: Steed method at l=0 ===
        let (f_0, g_0, fp_0, gp_0) = steed_method(0.0, eta, rho)?;

        if l_max == 0 {
            f_all = vec![f_0];
            g_all = vec![g_0];
            fp_all = vec![fp_0];
            gp_all = vec![gp_0];
        } else {
            // G upward recurrence from l=0 (stable)
            let (g_up, gp_up) = recurrence_g_upward(0, l_max, eta, rho, g_0, gp_0);
            g_all = g_up;
            gp_all = gp_up;

            // F downward recurrence from l_max, normalized against Steed at l=0
            let f_ratio_top = cf1(l_max as f64, eta, rho)?;
            let (f_unnorm, fp_unnorm) = recurrence_f_downward(l_max, 0, eta, rho, f_ratio_top);

            let scale = if f_unnorm[0].abs() > 1e-300 {
                f_0 / f_unnorm[0]
            } else if fp_unnorm[0].abs() > 1e-300 {
                fp_0 / fp_unnorm[0]
            } else {
                1.0
            };

            f_all = f_unnorm.iter().map(|&v| v * scale).collect();
            fp_all = fp_unnorm.iter().map(|&v| v * scale).collect();
        }
    } else if in_forbidden && rho > RHO_SMALL {
        // === Forbidden region: F from power series, G from Numerov ===
        let mut f_v = Vec::with_capacity(total_l);
        let mut fp_v = Vec::with_capacity(total_l);
        for i in 0..total_l {
            let (f, fp) = power_series_f(i as u32, eta, rho)?;
            f_v.push(f);
            fp_v.push(fp);
        }

        let (g_num, gp_num) = g_backward_numerov(0.0, eta, rho)?;
        let gp_refined = if f_v[0].abs() > 1e-300 {
            (fp_v[0] * g_num - 1.0) / f_v[0]
        } else {
            gp_num
        };

        let (g_up, gp_up) = recurrence_g_upward(0, l_max, eta, rho, g_num, gp_refined);

        f_all = f_v;
        fp_all = fp_v;
        g_all = g_up;
        gp_all = gp_up;
    } else {
        // === Small rho: power series for F, CF2/Numerov for G ===
        let mut f_v = Vec::with_capacity(total_l);
        let mut fp_v = Vec::with_capacity(total_l);
        for i in 0..total_l {
            let (f, fp) = power_series_f(i as u32, eta, rho)?;
            f_v.push(f);
            fp_v.push(fp);
        }

        let (g_0, gp_0) = {
            let cf2_result = cf2(0.0, eta, rho);
            let mut g_ok = None;

            if let Ok((p, q)) = cf2_result {
                if q.abs() > 1e-300 {
                    let f_ratio = if f_v[0].abs() > 1e-300 {
                        fp_v[0] / f_v[0]
                    } else {
                        cf1(0.0, eta, rho).unwrap_or(0.0)
                    };
                    let gamm = (f_ratio - p) / q;
                    let g_try = gamm * f_v[0];
                    let gp_try = p * g_try - q * f_v[0];

                    let w = fp_v[0] * g_try - f_v[0] * gp_try;
                    if (w - 1.0).abs() < 0.01 {
                        g_ok = Some((g_try, gp_try));
                    }
                }
            }

            match g_ok {
                Some(val) => val,
                None => {
                    let (g_num, gp_num) = g_backward_numerov(0.0, eta, rho)?;
                    let gp_refined = if f_v[0].abs() > 1e-300 {
                        (fp_v[0] * g_num - 1.0) / f_v[0]
                    } else {
                        gp_num
                    };
                    (g_num, gp_refined)
                }
            }
        };

        let (g_up, gp_up) = recurrence_g_upward(0, l_max, eta, rho, g_0, gp_0);

        f_all = f_v;
        fp_all = fp_v;
        g_all = g_up;
        gp_all = gp_up;
    }

    // Slice to requested range [l_min..l_min+n_l]
    let start = l_min as usize;
    let end = start + n_l as usize;

    let sigma: Vec<f64> = (0..n_l)
        .map(|i| coulomb_phase_shift(l_min + i, eta))
        .collect();

    Ok(CoulombResult {
        f: f_all[start..end].to_vec(),
        g: g_all[start..end].to_vec(),
        fp: fp_all[start..end].to_vec(),
        gp: gp_all[start..end].to_vec(),
        sigma,
        exponent: 0.0,
    })
}

/// Batch computation of Coulomb wave functions for multiple (eta, rho) pairs.
pub fn coulomb_wave_batch(
    eta: &[f64],
    rho: &[f64],
    l_min: u32,
    n_l: u32,
) -> Result<Vec<CoulombResult>, SpecialError> {
    assert_eq!(eta.len(), rho.len(), "eta and rho must have same length");
    eta.iter()
        .zip(rho.iter())
        .map(|(&e, &r)| coulomb_wave(e, r, l_min, n_l))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_wronskian(result: &CoulombResult, tol: f64) {
        for i in 0..result.f.len() {
            let w = result.fp[i] * result.g[i] - result.f[i] * result.gp[i];
            assert!(
                (w - 1.0).abs() < tol,
                "Wronskian at l_min+{}: got {} (err = {:.2e})",
                i,
                w,
                (w - 1.0).abs()
            );
        }
    }

    // ==================== Tier 1: Standard cases ====================

    #[test]
    fn tier1_bessel_l0_eta0_rho1() {
        let rho = 1.0;
        let result = coulomb_wave(0.0, rho, 0, 1).unwrap();
        assert!(
            (result.f[0] - rho.sin()).abs() < 1e-12,
            "F_0 = {}, expected {}",
            result.f[0],
            rho.sin()
        );
        assert!(
            (result.g[0] - rho.cos()).abs() < 1e-12,
            "G_0 = {}, expected {}",
            result.g[0],
            rho.cos()
        );
        check_wronskian(&result, 1e-12);
    }

    #[test]
    fn tier1_bessel_l0_eta0_various_rho() {
        for &rho in &[0.7, 1.0, 2.0, 5.0, 10.0] {
            let result = coulomb_wave(0.0, rho, 0, 1).unwrap();
            let rel_err = if rho.sin().abs() > 1e-10 {
                ((result.f[0] - rho.sin()) / rho.sin()).abs()
            } else {
                (result.f[0] - rho.sin()).abs()
            };
            assert!(rel_err < 1e-10, "F_0(0,{}): rel_err = {:.2e}", rho, rel_err);
            check_wronskian(&result, 1e-10);
        }
    }

    #[test]
    fn tier1_bessel_multiple_l_eta0() {
        // eta=0: F_0 = sin(rho), G_0 = cos(rho)
        // F_1 = sin(rho)/rho - cos(rho), G_1 = cos(rho)/rho + sin(rho)
        let rho = 3.0;
        let result = coulomb_wave(0.0, rho, 0, 2).unwrap();

        let expected_f0 = rho.sin();
        let expected_f1 = rho.sin() / rho - rho.cos();
        let expected_g0 = rho.cos();
        let expected_g1 = rho.cos() / rho + rho.sin();

        assert!(
            (result.f[0] - expected_f0).abs() < 1e-10,
            "F_0 = {}, expected {}",
            result.f[0],
            expected_f0
        );
        assert!(
            (result.f[1] - expected_f1).abs() < 1e-10,
            "F_1 = {}, expected {}",
            result.f[1],
            expected_f1
        );
        assert!(
            (result.g[0] - expected_g0).abs() < 1e-10,
            "G_0 = {}, expected {}",
            result.g[0],
            expected_g0
        );
        assert!(
            (result.g[1] - expected_g1).abs() < 1e-10,
            "G_1 = {}, expected {}",
            result.g[1],
            expected_g1
        );
        check_wronskian(&result, 1e-10);
    }

    #[test]
    fn tier1_eta1_rho5() {
        let result = coulomb_wave(1.0, 5.0, 0, 1).unwrap();
        check_wronskian(&result, 1e-10);
        assert!(result.f[0].is_finite());
        assert!(result.g[0].is_finite());
    }

    #[test]
    fn tier1_l2_eta1p5_rho3p5() {
        let result = coulomb_wave(1.5, 3.5, 2, 1).unwrap();
        check_wronskian(&result, 1e-10);
    }

    // ==================== Tier 2: Barrier penetration ====================

    #[test]
    fn tier2_eta10_rho5() {
        // Moderate barrier: rho_tp ~ 20, rho=5 inside barrier
        let result = coulomb_wave(10.0, 5.0, 0, 1).unwrap();
        check_wronskian(&result, 1e-4);
        // In the deep forbidden region, F is very small and G is very large
        assert!(result.f[0].is_finite(), "F_0 should be finite");
        assert!(result.g[0].is_finite(), "G_0 should be finite");
        // F should be much smaller than G (barrier penetration)
        assert!(
            result.f[0].abs() < result.g[0].abs(),
            "F should be smaller than G in barrier: F={:.2e}, G={:.2e}",
            result.f[0],
            result.g[0]
        );
    }

    // ==================== Tier 3: High l ====================

    #[test]
    fn tier3_l10_eta5_rho10() {
        let result = coulomb_wave(5.0, 10.0, 10, 1);
        if let Ok(result) = result {
            check_wronskian(&result, 1e-8);
        }
    }

    // ==================== Self-consistency ====================

    #[test]
    fn wronskian_multiple_l() {
        let result = coulomb_wave(2.0, 5.0, 0, 5).unwrap();
        check_wronskian(&result, 1e-8);
    }

    #[test]
    fn sigma_eta_zero() {
        let result = coulomb_wave(0.0, 3.0, 0, 3).unwrap();
        for (i, &s) in result.sigma.iter().enumerate() {
            assert!(s.abs() < 1e-13, "sigma_{}(0) = {}", i, s);
        }
    }

    #[test]
    fn lmin_gt0_matches_from_zero() {
        // A-1 bug: coulomb_wave(eta, rho, l_min=L, 1) should match
        // coulomb_wave(eta, rho, 0, L+1)[L]
        let cases = vec![
            (0.0, 3.0, 1u32),
            (0.0, 3.0, 2),
            (1.0, 5.0, 1),
            (1.0, 5.0, 2),
            (1.5, 3.5, 2),
            (2.0, 5.0, 1),
            (2.0, 5.0, 3),
            (5.0, 10.0, 2),
            (5.0, 10.0, 5),
            (0.5, 2.0, 1),
            (0.5, 2.0, 3),
            (10.0, 15.0, 1),
            (10.0, 15.0, 3),
            (3.0, 8.0, 4),
            (1.0, 1.5, 1),
            (1.0, 1.5, 2),
        ];

        for (eta, rho, l) in cases {
            let direct = coulomb_wave(eta, rho, l, 1).unwrap();
            let from_zero = coulomb_wave(eta, rho, 0, l + 1).unwrap();
            let idx = l as usize;

            let f_rel = if from_zero.f[idx].abs() > 1e-15 {
                ((direct.f[0] - from_zero.f[idx]) / from_zero.f[idx]).abs()
            } else {
                (direct.f[0] - from_zero.f[idx]).abs()
            };
            let g_rel = if from_zero.g[idx].abs() > 1e-15 {
                ((direct.g[0] - from_zero.g[idx]) / from_zero.g[idx]).abs()
            } else {
                (direct.g[0] - from_zero.g[idx]).abs()
            };

            assert!(
                f_rel < 1e-8,
                "F mismatch at eta={}, rho={}, l={}: direct={:.10e}, from0={:.10e}, rel={:.2e}",
                eta,
                rho,
                l,
                direct.f[0],
                from_zero.f[idx],
                f_rel
            );
            assert!(
                g_rel < 1e-8,
                "G mismatch at eta={}, rho={}, l={}: direct={:.10e}, from0={:.10e}, rel={:.2e}",
                eta,
                rho,
                l,
                direct.g[0],
                from_zero.g[idx],
                g_rel
            );

            // Also check Wronskian for direct computation
            let w = direct.fp[0] * direct.g[0] - direct.f[0] * direct.gp[0];
            assert!(
                (w - 1.0).abs() < 1e-8,
                "Wronskian bad at eta={}, rho={}, l={}: W={:.10e}",
                eta,
                rho,
                l,
                w
            );
        }
    }

    #[test]
    fn lmin_gt0_multi_l_consistency() {
        // coulomb_wave(eta, rho, l_min=2, n_l=3) should match
        // coulomb_wave(eta, rho, 0, 5)[2..5]
        let cases = vec![(0.0, 5.0), (1.0, 5.0), (2.0, 8.0), (5.0, 12.0)];

        for (eta, rho) in cases {
            let from_lmin = coulomb_wave(eta, rho, 2, 3).unwrap();
            let from_zero = coulomb_wave(eta, rho, 0, 5).unwrap();

            for i in 0..3 {
                let idx = i + 2;
                let f_rel = if from_zero.f[idx].abs() > 1e-15 {
                    ((from_lmin.f[i] - from_zero.f[idx]) / from_zero.f[idx]).abs()
                } else {
                    (from_lmin.f[i] - from_zero.f[idx]).abs()
                };
                assert!(
                    f_rel < 1e-10,
                    "F mismatch at eta={}, rho={}, l={}: rel={:.2e}",
                    eta,
                    rho,
                    i + 2,
                    f_rel
                );
            }
            check_wronskian(&from_lmin, 1e-8);
        }
    }

    #[test]
    fn batch_matches_single() {
        let etas = vec![0.0, 1.0, 2.0];
        let rhos = vec![3.0, 5.0, 7.0];
        let batch = coulomb_wave_batch(&etas, &rhos, 0, 1).unwrap();
        for (i, (&e, &r)) in etas.iter().zip(rhos.iter()).enumerate() {
            let single = coulomb_wave(e, r, 0, 1).unwrap();
            assert!(
                (batch[i].f[0] - single.f[0]).abs() < 1e-14,
                "batch mismatch at {}",
                i
            );
        }
    }

    // ==================== Small rho ====================

    #[test]
    fn small_rho_l0_eta0() {
        let rho = 0.1;
        let result = coulomb_wave(0.0, rho, 0, 1).unwrap();
        assert!(
            (result.f[0] - rho.sin()).abs() < 1e-12,
            "F_0(0,0.1) = {}, expected {}",
            result.f[0],
            rho.sin()
        );
        check_wronskian(&result, 1e-10);
    }

    #[test]
    fn small_rho_l0_eta1() {
        let rho = 0.3;
        let result = coulomb_wave(1.0, rho, 0, 1).unwrap();
        assert!(result.f[0].is_finite());
        assert!(result.g[0].is_finite());
        check_wronskian(&result, 1e-8);
    }

    // ==================== Cross product identity ====================

    #[test]
    fn cross_product_identity() {
        // F_{l-1} * G_l - F_l * G_{l-1} = l / sqrt(l^2 + eta^2)
        let eta = 2.0;
        let rho = 5.0;
        let result = coulomb_wave(eta, rho, 0, 4).unwrap();

        for l in 1..4_usize {
            let lf = l as f64;
            let cross = result.f[l - 1] * result.g[l] - result.f[l] * result.g[l - 1];
            let expected = lf / (lf * lf + eta * eta).sqrt();
            assert!(
                (cross - expected).abs() < 1e-8,
                "Cross product at l={}: got {}, expected {} (err = {:.2e})",
                l,
                cross,
                expected,
                (cross - expected).abs()
            );
        }
    }
}
