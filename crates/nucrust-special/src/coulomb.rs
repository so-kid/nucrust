use crate::consts::{CF_EPS, CF_ZERO_GUARD, MAX_CF_ITER, RHO_SMALL};
use crate::error::SpecialError;
use crate::gamma::{coulomb_phase_shift, gamow_factor};

/// Target relative accuracy of the l = 0 normalization; a method whose
/// estimated rounding error exceeds this is passed over for a better one.
const NORM_TOL: f64 = 1e-14;
/// Maximum terms in power series.
const MAX_SERIES_TERMS: usize = 1000;
/// Minimum terms in power series before checking convergence.
const MIN_SERIES_TERMS: usize = 10;

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

/// Evaluate CF1 for a given l, eta, rho (DLMF §33.8.1) with the modified Lentz method.
///
/// Returns `(f_l, sign)` where f_l = F'_l / F_l and `sign` is the sign of F_l.
///
/// The sign comes for free from the Lentz iteration (Barnett's COULFG / GSL
/// `fcl_sign`): the denominators D_n are ratios B_{n-1}/B_n of the continuants
/// B_n, which obey the same three-term recurrence as the Coulomb functions with
/// B_{-1} = 0, B_0 = 1. Hence B_n ∝ F_l G_{l+n+1} - G_l F_{l+n+1} with a positive
/// factor, and once the fraction has converged G dominates and
/// sign(B_N) = sign(F_l) = Π sign(D_n). This is exact, unlike the asymptotic
/// phase, which is unreliable near the turning point.
fn cf1(l: f64, eta: f64, rho: f64) -> Result<(f64, f64), SpecialError> {
    let a = |n: u32| {
        let ln = l + n as f64;
        -(1.0 + (eta / ln).powi(2)) // a_n = -R^2_{l+n}
    };
    let b = |n: u32| {
        let ln = l + n as f64;
        // b_n = T_{l+n} = S_{l+n} + S_{l+n+1}
        ln / rho + eta / ln + (ln + 1.0) / rho + eta / (ln + 1.0)
    };

    // b_0 = S_{l+1}
    let mut h = (l + 1.0) / rho + eta / (l + 1.0);
    if h.abs() < CF_ZERO_GUARD {
        h = CF_ZERO_GUARD;
    }
    let mut c = h;
    let mut d = 0.0_f64;
    let mut sign = 1.0_f64;

    for n in 1..=MAX_CF_ITER {
        let (an, bn) = (a(n), b(n));
        d = bn + an * d;
        if d.abs() < CF_ZERO_GUARD {
            d = CF_ZERO_GUARD;
        }
        c = bn + an / c;
        if c.abs() < CF_ZERO_GUARD {
            c = CF_ZERO_GUARD;
        }
        d = 1.0 / d;
        if d < 0.0 {
            sign = -sign;
        }
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < CF_EPS {
            return Ok((h, sign));
        }
    }

    Err(SpecialError::ConvergenceFailure {
        algorithm: "CF1 (Lentz)",
        iterations: MAX_CF_ITER,
        residual: h.abs(),
    })
}

// ======================== CF2: p + iq = H^+' / H^+ ========================

/// Result of CF2: p + iq = H^{+\prime}_l / H^+_l.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cf2 {
    pub(crate) p: f64,
    pub(crate) q: f64,
    /// Estimated absolute rounding error of `q`.
    ///
    /// In the classically forbidden region q = 1/(F^2 + G^2) is tiny and is
    /// obtained as a sum whose partial sums are much larger (q starts at
    /// 1 - eta/rho), so its relative accuracy degrades as
    /// eps * max|partial sum| / q. This is the reason Steed's method fails there.
    pub(crate) q_err: f64,
}

/// Evaluate CF2 using the Steed algorithm.
///
/// This uses the Steed (1982) / Barnett algorithm which accumulates P and Q
/// as sums of corrections. This is essential for correctness when l > 0,
/// where the Lentz method on the DLMF continued fraction can terminate
/// prematurely and produce wrong results.
///
/// Reference: GSL specfunc/coulomb.c `coulomb_CF2`; Barnett (1982) CPC 27, 147.
fn cf2(l: f64, eta: f64, rho: f64) -> Result<Cf2, SpecialError> {
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
    let mut q_scale = q.abs().max(1.0);

    let mut pk = 0.0;

    for _ in 1..=MAX_CF_ITER {
        p += dp;
        q += dq;
        q_scale = q_scale.max(q.abs()).max(dq.abs());
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
            return Ok(Cf2 {
                p,
                q,
                q_err: f64::EPSILON * q_scale + dq.abs(),
            });
        }
    }

    Err(SpecialError::ConvergenceFailure {
        algorithm: "CF2 (Steed)",
        iterations: MAX_CF_ITER,
        residual: (p * p + q * q).sqrt(),
    })
}

// ======================== Power series (small rho) ========================

/// Result of the 1F1 power series for F_l.
#[derive(Debug, Clone, Copy)]
struct SeriesF {
    f: f64,
    fp: f64,
    /// Estimated relative rounding error from cancellation between terms.
    rel_err: f64,
}

/// Compute F_l, F'_l using the 1F1 power series (DLMF §33.6.1).
///
/// F_l(eta, rho) = C_l(eta) * rho^{l+1} * sum_{k=0}^N A_k * rho^k
///
/// Returns `None` if the series does not converge within `MAX_SERIES_TERMS`.
fn power_series_f(l: u32, eta: f64, rho: f64) -> Option<SeriesF> {
    let lf = l as f64;
    let c_l = gamow_factor(l, eta);

    // Coefficient recurrence: A_0=1, A_1=eta/(l+1)
    // A_k = (2*eta*A_{k-1} - A_{k-2}) / (k*(2l+1+k)),
    // carried out on the terms t_k = A_k * rho^k directly so that neither
    // A_k nor rho^k overflows on its own at large rho:
    // t_k = (2*eta*rho*t_{k-1} - rho^2*t_{k-2}) / (k*(2l+1+k))
    let mut t_km2 = 0.0; // t_{k-2}, initialized for k=1
    let mut t_km1 = 1.0; // t_{k-1} = t_0

    let mut sum = 1.0; // S(rho) = sum t_k
    let mut dsum = 0.0; // rho * S'(rho) = sum k * t_k
    let mut max_term = 1.0_f64;
    let mut converged = false;

    for k in 1..=MAX_SERIES_TERMS {
        let kf = k as f64;
        let term = (2.0 * eta * rho * t_km1 - rho * rho * t_km2) / (kf * (2.0 * lf + 1.0 + kf));

        sum += term;
        dsum += kf * term;
        max_term = max_term.max((kf * term).abs());
        if !sum.is_finite() {
            return None;
        }

        // Check convergence only after minimum terms. Two consecutive small
        // terms guard against an accidental near-zero A_k.
        if k >= MIN_SERIES_TERMS
            && term.abs() < CF_EPS * sum.abs()
            && t_km1.abs() < CF_EPS * sum.abs()
        {
            converged = true;
            break;
        }

        t_km2 = t_km1;
        t_km1 = term;
    }
    if !converged {
        return None;
    }

    let rho_l1 = rho.powi(l as i32 + 1);
    let f = c_l * rho_l1 * sum;
    // F' = C_l * d/drho[rho^{l+1} * S(rho)]
    //    = C_l * [(l+1) * rho^l * S + rho^{l+1} * S']
    let fp = c_l * rho_l1 / rho * ((lf + 1.0) * sum + dsum);
    let scale = sum.abs().min((dsum + (lf + 1.0) * sum).abs());
    let rel_err = if scale > 0.0 {
        f64::EPSILON * max_term / scale
    } else {
        f64::INFINITY
    };

    Some(SeriesF { f, fp, rel_err })
}

// ======================== Recurrence relations ========================

/// Downward recurrence for F from l_top to 0 (DLMF §33.4).
///
/// Uses derivative relation: F_{l-1} = (F'_l + S_l * F_l) / R_l
///                           F'_{l-1} = S_l * F_{l-1} - R_l * F_l
///
/// Starts from F_{l_top} = `sign` (the true sign of F_{l_top}) and
/// F'_{l_top} = sign * f_top, so the result equals the true F, F' times an
/// unknown *positive* factor. Rescales on the way down to avoid overflow.
/// Returns (f_unnorm, fp_unnorm) arrays indexed by l.
fn recurrence_f_downward(
    l_top: u32,
    eta: f64,
    rho: f64,
    sign: f64,
    f_top: f64, // f_{l_top} = F'/F from CF1
) -> (Vec<f64>, Vec<f64>) {
    const BIG: f64 = 1e250;
    let n = l_top as usize + 1;
    let mut f_vals = vec![0.0; n];
    let mut fp_vals = vec![0.0; n];

    f_vals[n - 1] = sign;
    fp_vals[n - 1] = sign * f_top;

    for l in (1..=l_top as usize).rev() {
        let lf = l as f64;
        let r = r_l(lf, eta);
        let s = s_l(lf, eta, rho);

        // F_{l-1} = (F'_l + S_l * F_l) / R_l
        let f_below = (fp_vals[l] + s * f_vals[l]) / r;
        // F'_{l-1} = S_l * F_{l-1} - R_l * F_l
        let fp_below = s * f_below - r * f_vals[l];

        f_vals[l - 1] = f_below;
        fp_vals[l - 1] = fp_below;

        if f_below.abs().max(fp_below.abs()) > BIG {
            for v in f_vals[l - 1..]
                .iter_mut()
                .chain(fp_vals[l - 1..].iter_mut())
            {
                *v /= BIG;
            }
        }
    }

    (f_vals, fp_vals)
}

/// Upward recurrence for G from 0 to l_top (inclusive).
///
/// G_{l+1} = (S_{l+1} * G_l - G'_l) / R_{l+1}
/// G'_{l+1} = R_{l+1} * G_l - S_{l+1} * G_{l+1}
fn recurrence_g_upward(l_top: u32, eta: f64, rho: f64, g0: f64, gp0: f64) -> (Vec<f64>, Vec<f64>) {
    let n = l_top as usize + 1;
    let mut g_vals = vec![0.0; n];
    let mut gp_vals = vec![0.0; n];

    g_vals[0] = g0;
    gp_vals[0] = gp0;

    for l in 0..l_top as usize {
        let lf = (l + 1) as f64;
        let r_next = r_l(lf, eta);
        let s_next = s_l(lf, eta, rho);

        // G_{l+1} = (S_{l+1} * G_l - G'_l) / R_{l+1}
        let g_next = (s_next * g_vals[l] - gp_vals[l]) / r_next;
        // G'_{l+1} = R_{l+1} * G_l - S_{l+1} * G_{l+1}
        gp_vals[l + 1] = r_next * g_vals[l] - s_next * g_next;
        g_vals[l + 1] = g_next;
    }

    (g_vals, gp_vals)
}

// ======================== Taylor-series ODE stepping ========================

/// Integrate the l = 0 Coulomb equation w'' = (2*eta/rho - 1) w from `rho_from`
/// to `rho_to` with high-order Taylor steps, starting from (w, w').
///
/// About rho_0 the equation rho w'' = (2 eta - rho) w gives the exact
/// coefficient recurrence (x = rho - rho_0, w = Σ c_n x^n)
///   rho_0 (m+2)(m+1) c_{m+2} = (2 eta - rho_0) c_m - c_{m-1} - (m+1) m c_{m+1},
/// so each step is accurate to rounding as long as |h| stays well inside the
/// radius of convergence rho_0. Used to carry G from the turning point into the
/// forbidden region, where G is dominant and inward integration is stable.
fn taylor_integrate_l0(
    eta: f64,
    rho_from: f64,
    rho_to: f64,
    mut w: f64,
    mut wp: f64,
) -> Option<(f64, f64)> {
    const MAX_STEP: f64 = 2.0;
    const MAX_TERMS: usize = 200;
    let mut rho0 = rho_from;
    while rho0 != rho_to {
        let dist = rho_to - rho0;
        let h = if dist.abs() <= (0.5 * rho0).min(MAX_STEP) {
            dist
        } else {
            dist.signum() * (0.5 * rho0).min(MAX_STEP)
        };
        // Scaled coefficients d_n = c_n h^n.
        let (mut d_m1, mut d0, mut d1) = (0.0, w, wp * h);
        let (mut sum, mut dsum) = (d0 + d1, d1);
        let mut converged = false;
        for m in 0..MAX_TERMS {
            let mf = m as f64;
            let d2 =
                ((2.0 * eta - rho0) * h * h * d0 - h * h * h * d_m1 - (mf + 1.0) * mf * h * d1)
                    / (rho0 * (mf + 2.0) * (mf + 1.0));
            sum += d2;
            dsum += (mf + 2.0) * d2;
            if m >= 4 && d2.abs() + d1.abs() <= f64::EPSILON * 0.1 * sum.abs() {
                converged = true;
                break;
            }
            (d_m1, d0, d1) = (d0, d1, d2);
        }
        if !converged || !sum.is_finite() {
            return None;
        }
        w = sum;
        wp = dsum / h;
        rho0 = if h == dist { rho_to } else { rho0 + h };
    }
    Some((w, wp))
}

// ======================== Normalization at l = 0 ========================

/// Compute F_0, F'_0, G_0, G'_0 given the unnormalized (but correctly signed)
/// F_0, F'_0 from the downward recurrence.
///
/// Three absolute normalizations are available (Thompson & Barnett 1985):
///
/// * **Steed's method** (COULCC case 3): with p + iq = H^+'/H^+ from CF2,
///   the Wronskian F'G - FG' = 1 and G' = pG - qF fix the scale
///   s^2 = (F'_u - p F_u)^2 / q + q F_u^2. Its accuracy is that of q, which
///   is poor in the classically forbidden region (see [`Cf2::q_err`]).
/// * **1F1 power series** (COULCC case 1) for F, with G from the exact identity
///   G = (1 - q F^2) / (F' - p F) (Wronskian + G' = pG - qF). There q enters
///   only through q F^2 << 1 and p ≈ G'/G is accurate, so G inherits the
///   accuracy of the series. The series is itself ill-conditioned close to the
///   turning point at large eta.
/// * **rho shift** (RQ-01): Steed at the turning point rho' = 2 eta, where it is
///   accurate, then G is carried inward to rho by Taylor steps (stable, G is
///   dominant) and F follows from the Wronskian. Used only when neither of the
///   above reaches [`NORM_TOL`].
///
/// Steed is used when accurate, else the more accurate of the other two.
fn normalize_l0(
    eta: f64,
    rho: f64,
    fu: f64,
    fpu: f64,
    cf: Result<Cf2, SpecialError>,
) -> Result<(f64, f64, f64, f64), SpecialError> {
    let (cf, cf_err) = match cf {
        Ok(c) => (Some(c), None),
        Err(e) => (None, Some(e)),
    };
    let steed_err = match cf {
        Some(c) if c.q > 0.0 => c.q_err / c.q,
        _ => f64::INFINITY,
    };
    if steed_err <= NORM_TOL && rho > RHO_SMALL {
        if let Some(c) = cf {
            return Ok(steed_normalize(fu, fpu, c));
        }
    }

    let series = power_series_f(0, eta, rho).filter(|s| s.rel_err < steed_err);
    if let (Some(sr), Some(c)) = (series, cf) {
        if sr.rel_err <= NORM_TOL || eta <= 0.0 || rho >= 2.0 * eta {
            let g = (1.0 - c.q * sr.f * sr.f) / (sr.fp - c.p * sr.f);
            let gp = c.p * g - c.q * sr.f;
            return Ok((sr.f, sr.fp, g, gp));
        }
    }

    // rho shift: only reachable for eta > 0 inside the l = 0 turning point.
    if eta > 0.0 && rho < 2.0 * eta {
        let rho_tp = 2.0 * eta;
        if let Ok(c_tp) = cf2(0.0, eta, rho_tp) {
            let (f_tp, sign_tp) = cf1(0.0, eta, rho_tp)?;
            let (_, _, g_tp, gp_tp) = steed_normalize(sign_tp, sign_tp * f_tp, c_tp);
            if let Some((g, gp)) = taylor_integrate_l0(eta, rho_tp, rho, g_tp, gp_tp) {
                // Wronskian: (F'_u G - F_u G') / s = 1
                let s = fpu * g - fu * gp;
                return Ok((fu / s, fpu / s, g, gp));
            }
        }
    }

    // Fall back to the best available estimate.
    match (series, cf) {
        (Some(sr), Some(c)) => {
            let g = (1.0 - c.q * sr.f * sr.f) / (sr.fp - c.p * sr.f);
            let gp = c.p * g - c.q * sr.f;
            Ok((sr.f, sr.fp, g, gp))
        }
        (None, Some(c)) if c.q > 0.0 => Ok(steed_normalize(fu, fpu, c)),
        _ => Err(cf_err.unwrap_or(SpecialError::NumericalOverflow {
            context: "Steed method: q <= 0",
        })),
    }
}

/// Steed's normalization: F, F', G, G' from F_u, F'_u (true values times an
/// unknown positive factor) and CF2.
fn steed_normalize(fu: f64, fpu: f64, c: Cf2) -> (f64, f64, f64, f64) {
    let w = fpu - c.p * fu;
    let s = (w * w / c.q + c.q * fu * fu).sqrt();
    let (f, fp) = (fu / s, fpu / s);
    let g = w / (c.q * s);
    let gp = c.p * g - c.q * f;
    (f, fp, g, gp)
}

// ======================== Public API ========================

/// Compute Coulomb wave functions F_l, G_l, F'_l, G'_l, sigma_l.
///
/// # Algorithm
/// Thompson & Barnett (1985) COULCC / Barnett COULFG for real eta and rho > 0:
///
/// 1. CF1 at l_max gives F'/F and the exact sign of F_{l_max}; downward
///    recurrence (stable for F) gives F_l, F'_l for l = 0..=l_max up to a
///    positive factor.
/// 2. The factor and G_0, G'_0 are fixed at l = 0 by Steed's method (CF2) or,
///    where that loses accuracy (classically forbidden region, small rho), by
///    the 1F1 power series for F_0 together with CF2 for G_0, or by a rho
///    shift to the turning point with Taylor integration of G
///    (see `normalize_l0`).
/// 3. Upward recurrence (stable for G) gives G_l, G'_l.
///
/// l = 0 is always the normalization point, regardless of `l_min`.
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
    let sigma: Vec<f64> = (0..n_l)
        .map(|i| coulomb_phase_shift(l_min + i, eta))
        .collect();
    if rho == 0.0 {
        let n = n_l as usize;
        return Ok(CoulombResult {
            f: vec![0.0; n],
            g: vec![f64::INFINITY; n],
            fp: vec![0.0; n],
            gp: vec![f64::NEG_INFINITY; n],
            sigma,
            exponent: 0.0,
        });
    }

    let l_max = l_min + n_l - 1;
    let cf1_top = cf1(l_max as f64, eta, rho)?;
    let cf2_0 = cf2(0.0, eta, rho);
    assemble(eta, rho, l_min, n_l, cf1_top, cf2_0, sigma)
}

/// Steps 1-3 of [`coulomb_wave`] given CF1 at l_max (ratio, sign) and CF2 at
/// l = 0. Shared with the SIMD batch path, which evaluates the continued
/// fractions four lanes at a time. Requires rho > 0 and n_l > 0.
pub(crate) fn assemble(
    eta: f64,
    rho: f64,
    l_min: u32,
    n_l: u32,
    (f_top, sign_top): (f64, f64),
    cf2_0: Result<Cf2, SpecialError>,
    sigma: Vec<f64>,
) -> Result<CoulombResult, SpecialError> {
    let l_max = l_min + n_l - 1;

    // Step 1: F_l up to a positive factor, with correct signs.
    let (mut f_all, mut fp_all) = recurrence_f_downward(l_max, eta, rho, sign_top, f_top);

    // Step 2: absolute normalization and G at l = 0.
    let (f0, fp0, g0, gp0) = normalize_l0(eta, rho, f_all[0], fp_all[0], cf2_0)?;
    let scale = if f_all[0].abs() >= fp_all[0].abs() {
        f0 / f_all[0]
    } else {
        fp0 / fp_all[0]
    };
    for v in f_all.iter_mut().chain(fp_all.iter_mut()) {
        *v *= scale;
    }
    f_all[0] = f0;
    fp_all[0] = fp0;

    // Step 3: G_l by upward recurrence.
    let (g_all, gp_all) = recurrence_g_upward(l_max, eta, rho, g0, gp0);

    // Slice to requested range [l_min..l_min+n_l]
    let start = l_min as usize;

    Ok(CoulombResult {
        f: f_all.split_off(start),
        g: g_all[start..].to_vec(),
        fp: fp_all.split_off(start),
        gp: gp_all[start..].to_vec(),
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

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        eta.par_iter()
            .zip(rho.par_iter())
            .map(|(&e, &r)| coulomb_wave(e, r, l_min, n_l))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        eta.iter()
            .zip(rho.iter())
            .map(|(&e, &r)| coulomb_wave(e, r, l_min, n_l))
            .collect()
    }
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
        check_wronskian(&result, 1e-14);
        assert!(result.f[0].is_finite());
        assert!(result.g[0].is_finite());
    }

    #[test]
    fn tier1_l2_eta1p5_rho3p5() {
        let result = coulomb_wave(1.5, 3.5, 2, 1).unwrap();
        check_wronskian(&result, 1e-14);
    }

    // ==================== Tier 2: Barrier penetration ====================

    #[test]
    fn tier2_eta10_rho5() {
        // Moderate barrier: rho_tp ~ 20, rho=5 inside barrier
        let result = coulomb_wave(10.0, 5.0, 0, 1).unwrap();
        check_wronskian(&result, 1e-14);
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
        let result = coulomb_wave(5.0, 10.0, 10, 1).unwrap();
        check_wronskian(&result, 1e-14);
    }

    // ==================== Self-consistency ====================

    #[test]
    fn wronskian_multiple_l() {
        let result = coulomb_wave(2.0, 5.0, 0, 5).unwrap();
        check_wronskian(&result, 1e-14);
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
                f_rel < 1e-13,
                "F mismatch at eta={}, rho={}, l={}: direct={:.10e}, from0={:.10e}, rel={:.2e}",
                eta,
                rho,
                l,
                direct.f[0],
                from_zero.f[idx],
                f_rel
            );
            assert!(
                g_rel < 1e-13,
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
                (w - 1.0).abs() < 1e-14,
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
                    f_rel < 1e-13,
                    "F mismatch at eta={}, rho={}, l={}: rel={:.2e}",
                    eta,
                    rho,
                    i + 2,
                    f_rel
                );
            }
            check_wronskian(&from_lmin, 1e-14);
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
        check_wronskian(&result, 1e-14);
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
                (cross - expected).abs() < 1e-14,
                "Cross product at l={}: got {}, expected {} (err = {:.2e})",
                l,
                cross,
                expected,
                (cross - expected).abs()
            );
        }
    }

    // ==================== Regression: sign and forbidden-region G ====================

    #[test]
    fn signs_near_turning_point() {
        // mpmath: F_0 and G_0 are both positive at these points just below the
        // l=0 turning point rho = 2 eta; the former asymptotic-phase sign rule
        // returned negative values for both.
        for &(eta, rho) in &[(5.0, 10.0), (10.0, 20.0), (20.0, 40.0)] {
            let r = coulomb_wave(eta, rho, 0, 1).unwrap();
            assert!(
                r.f[0] > 0.0 && r.g[0] > 0.0,
                "eta={eta}, rho={rho}: F={}, G={}",
                r.f[0],
                r.g[0]
            );
        }
    }

    #[test]
    fn forbidden_region_g_positive_for_all_l() {
        // Inside the barrier F and G are both positive and G grows with l.
        let r = coulomb_wave(10.0, 3.0, 0, 6).unwrap();
        for l in 0..6 {
            assert!(
                r.f[l] > 0.0 && r.g[l] > 0.0,
                "l={l}: F={}, G={}",
                r.f[l],
                r.g[l]
            );
            if l > 0 {
                assert!(r.g[l] > r.g[l - 1] && r.f[l] < r.f[l - 1]);
            }
        }
        check_wronskian(&r, 1e-14);
    }

    #[test]
    fn negative_rho_rejected_and_rho_zero() {
        assert!(coulomb_wave(1.0, -1.0, 0, 1).is_err());
        let r = coulomb_wave(1.0, 0.0, 0, 2).unwrap();
        assert_eq!(r.f, vec![0.0, 0.0]);
        assert!(r.g.iter().all(|g| g.is_infinite()));
    }
}
