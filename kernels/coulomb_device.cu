/**
 * On-the-fly Coulomb wave function computation as __device__ functions.
 *
 * Device port of the CPU algorithm in crates/nucrust-special/src/coulomb.rs
 * (Thompson & Barnett COULCC / Barnett COULFG for real eta and rho > 0):
 *
 *   1. CF1 at the requested l gives F'/F and the exact sign of F_l; downward
 *      recurrence (stable for F) carries F_l, F'_l to l = 0 up to a positive
 *      factor.
 *   2. The factor and G_0, G'_0 are fixed at l = 0 by Steed's method (CF2) or,
 *      where that loses accuracy (classically forbidden region, small rho), by
 *      the 1F1 power series for F_0 together with CF2 for G_0, or by a rho
 *      shift to the turning point with Taylor integration of G.
 *   3. Upward recurrence (stable for G) gives G_l, G'_l.
 *
 * Each thread computes F_l, G_l, F'_l, G'_l at arbitrary (eta, rho, l)
 * without a pre-computed lookup table and without per-l storage.
 *
 * hipify-clang compatible (plain C, only standard device math functions).
 */

#ifndef COULOMB_DEVICE_CU
#define COULOMB_DEVICE_CU

// Keep in sync with crates/nucrust-special/src/consts.rs and coulomb.rs.
#define COUL_MAX_CF_ITER 20000
#define COUL_CF_EPS 1e-15
#define COUL_CF_ZERO_GUARD 1e-50
#define COUL_RHO_SMALL 0.5
#define COUL_NORM_TOL 1e-14
#define COUL_MAX_SERIES_TERMS 1000
#define COUL_MIN_SERIES_TERMS 10
#define COUL_DBL_EPSILON 2.220446049250313e-16
#define COUL_DBL_MAX 1.7976931348623157e308
#define COUL_PI 3.141592653589793

/// Status codes returned by coulomb_wave_device.
#define COUL_OK 0
#define COUL_ERR_CF1 -1
#define COUL_ERR_CF2 -2

/// True for finite x (false for NaN and +-inf).
__device__ int coul_isfinite(double x) {
    return fabs(x) <= COUL_DBL_MAX;
}

// ======================== CF1: f_l = F'_l / F_l ========================

/// CF1 (DLMF 33.8.1) with the modified Lentz method.
///
/// Writes f_l = F'_l / F_l and the sign of F_l. The sign is exact: the Lentz
/// denominators D_n are ratios of continuants that obey the Coulomb
/// recurrence, so sign(F_l) = prod sign(D_n) once the fraction has converged
/// (Barnett COULFG / GSL fcl_sign). Returns 0 on success.
__device__ int coul_cf1(double l, double eta, double rho, double* f_out, double* sign_out) {
    double h = (l + 1.0) / rho + eta / (l + 1.0);  // b_0 = S_{l+1}
    if (fabs(h) < COUL_CF_ZERO_GUARD) h = COUL_CF_ZERO_GUARD;
    double c = h;
    double d = 0.0;
    double sign = 1.0;

    for (int n = 1; n <= COUL_MAX_CF_ITER; n++) {
        double ln = l + (double)n;
        double an = -(1.0 + (eta / ln) * (eta / ln));  // a_n = -R^2_{l+n}
        // b_n = T_{l+n} = S_{l+n} + S_{l+n+1}
        double bn = ln / rho + eta / ln + (ln + 1.0) / rho + eta / (ln + 1.0);

        d = bn + an * d;
        if (fabs(d) < COUL_CF_ZERO_GUARD) d = COUL_CF_ZERO_GUARD;
        c = bn + an / c;
        if (fabs(c) < COUL_CF_ZERO_GUARD) c = COUL_CF_ZERO_GUARD;
        d = 1.0 / d;
        if (d < 0.0) sign = -sign;
        double delta = c * d;
        h *= delta;
        if (fabs(delta - 1.0) < COUL_CF_EPS) {
            *f_out = h;
            *sign_out = sign;
            return 0;
        }
    }
    return -1;
}

// ======================== CF2: p + iq = H+' / H+ ========================

/// CF2 at angular momentum l with Steed's summation (GSL coulomb_CF2,
/// Barnett 1982). Writes p, q and q_err, the estimated absolute rounding
/// error of q (q = 1/(F^2+G^2) is tiny in the forbidden region and is formed
/// from much larger partial sums). Returns 0 on success.
__device__ int coul_cf2(double l, double eta, double rho,
                        double* p_out, double* q_out, double* q_err_out) {
    double wi = 2.0 * eta;
    double x_inv = 1.0 / rho;
    double e2mm1 = eta * eta + l * (l + 1.0);

    double ar = -e2mm1;
    double ai = eta;
    double br = 2.0 * (rho - eta);
    double bi = 2.0;

    double denom = br * br + bi * bi;
    if (denom < 1e-300) return -1;
    double dr = br / denom;
    double di = -bi / denom;

    double dp = -x_inv * (ar * di + ai * dr);
    double dq = x_inv * (ar * dr - ai * di);

    double p = 0.0;
    double q = 1.0 - eta * x_inv;
    double q_scale = fmax(fabs(q), 1.0);
    double pk = 0.0;

    for (int n = 1; n <= COUL_MAX_CF_ITER; n++) {
        p += dp;
        q += dq;
        q_scale = fmax(q_scale, fmax(fabs(q), fabs(dq)));
        pk += 2.0;
        ar += pk;
        ai += wi;
        bi += 2.0;

        double d_re = ar * dr - ai * di + br;
        double d_im = ai * dr + ar * di + bi;
        double c = 1.0 / (d_re * d_re + d_im * d_im);
        dr = c * d_re;
        di = -c * d_im;

        double a = br * dr - bi * di - 1.0;
        double b = bi * dr + br * di;
        double new_dp = dp * a - dq * b;
        dq = dp * b + dq * a;
        dp = new_dp;

        if (fabs(dp) + fabs(dq) <= (fabs(p) + fabs(q)) * COUL_CF_EPS) {
            *p_out = p;
            *q_out = q;
            *q_err_out = COUL_DBL_EPSILON * q_scale + fabs(dq);
            return 0;
        }
    }
    return -1;
}

// ======================== Power series (small rho) ========================

/// Gamow factor C_l(eta) = 2^l exp(-pi eta/2) |Gamma(l+1+i eta)| / (2l+1)!
/// in a form that does not overflow for large |eta|.
__device__ double coul_gamow_factor(int l, double eta) {
    double two_pi_eta = 2.0 * COUL_PI * eta;
    double c;
    if (eta > 0.0) {
        c = sqrt(two_pi_eta / -expm1(-two_pi_eta)) * exp(-COUL_PI * eta);
    } else if (eta < 0.0) {
        c = sqrt(two_pi_eta / expm1(two_pi_eta));
    } else {
        c = 1.0;
    }
    for (int k = 1; k <= l; k++) {
        double kf = (double)k;
        c *= sqrt(kf * kf + eta * eta) / (kf * (2.0 * kf + 1.0));
    }
    return c;
}

/// F_0, F'_0 from the 1F1 power series (DLMF 33.6.1). Writes rel_err, the
/// estimated relative rounding error from cancellation between terms.
/// Returns 0 on success, -1 if the series does not converge.
__device__ int coul_power_series_f0(double eta, double rho,
                                    double* f_out, double* fp_out, double* rel_err_out) {
    const double lf = 0.0;
    // Recurrence on t_k = A_k rho^k so neither A_k nor rho^k overflows:
    // t_k = (2 eta rho t_{k-1} - rho^2 t_{k-2}) / (k (2l+1+k))
    double t_km2 = 0.0;
    double t_km1 = 1.0;
    double sum = 1.0;
    double dsum = 0.0;  // rho * S'(rho)
    double max_term = 1.0;
    int converged = 0;

    for (int k = 1; k <= COUL_MAX_SERIES_TERMS; k++) {
        double kf = (double)k;
        double term = (2.0 * eta * rho * t_km1 - rho * rho * t_km2) / (kf * (2.0 * lf + 1.0 + kf));
        sum += term;
        dsum += kf * term;
        max_term = fmax(max_term, fabs(kf * term));
        if (!coul_isfinite(sum)) return -1;

        if (k >= COUL_MIN_SERIES_TERMS
            && fabs(term) < COUL_CF_EPS * fabs(sum)
            && fabs(t_km1) < COUL_CF_EPS * fabs(sum)) {
            converged = 1;
            break;
        }
        t_km2 = t_km1;
        t_km1 = term;
    }
    if (!converged) return -1;

    double c0 = coul_gamow_factor(0, eta);
    *f_out = c0 * rho * sum;
    *fp_out = c0 * ((lf + 1.0) * sum + dsum);
    double scale = fmin(fabs(sum), fabs(dsum + (lf + 1.0) * sum));
    *rel_err_out = scale > 0.0 ? COUL_DBL_EPSILON * max_term / scale : COUL_DBL_MAX;
    return 0;
}

// ======================== Taylor-series ODE stepping ========================

/// Integrate the l = 0 Coulomb equation w'' = (2 eta/rho - 1) w from rho_from
/// to rho_to with high-order Taylor steps, starting from (w, w'). Used to carry
/// G from the turning point into the forbidden region (stable: G dominant).
/// Returns 0 on success.
__device__ int coul_taylor_integrate_l0(double eta, double rho_from, double rho_to,
                                        double* w_io, double* wp_io) {
    const double MAX_STEP = 2.0;
    const int MAX_TERMS = 200;
    double w = *w_io;
    double wp = *wp_io;
    double rho0 = rho_from;
    while (rho0 != rho_to) {
        double dist = rho_to - rho0;
        double step = fmin(0.5 * rho0, MAX_STEP);
        double h = fabs(dist) <= step ? dist : copysign(step, dist);
        // Scaled coefficients d_n = c_n h^n.
        double d_m1 = 0.0, d0 = w, d1 = wp * h;
        double sum = d0 + d1, dsum = d1;
        int converged = 0;
        for (int m = 0; m < MAX_TERMS; m++) {
            double mf = (double)m;
            double d2 = ((2.0 * eta - rho0) * h * h * d0 - h * h * h * d_m1
                         - (mf + 1.0) * mf * h * d1)
                        / (rho0 * (mf + 2.0) * (mf + 1.0));
            sum += d2;
            dsum += (mf + 2.0) * d2;
            if (m >= 4 && fabs(d2) + fabs(d1) <= COUL_DBL_EPSILON * 0.1 * fabs(sum)) {
                converged = 1;
                break;
            }
            d_m1 = d0;
            d0 = d1;
            d1 = d2;
        }
        if (!converged || !coul_isfinite(sum)) return -1;
        w = sum;
        wp = dsum / h;
        rho0 = (h == dist) ? rho_to : rho0 + h;
    }
    *w_io = w;
    *wp_io = wp;
    return 0;
}

// ======================== Normalization at l = 0 ========================

/// Steed's normalization: F, F', G, G' from F_u, F'_u (true values times an
/// unknown positive factor) and CF2 at l = 0.
__device__ void coul_steed_normalize(double fu, double fpu, double p, double q,
                                     double* f, double* fp, double* g, double* gp) {
    double w = fpu - p * fu;
    double s = sqrt(w * w / q + q * fu * fu);
    *f = fu / s;
    *fp = fpu / s;
    *g = w / (q * s);
    *gp = p * (*g) - q * (*f);
}

/// F_0, F'_0, G_0, G'_0 from the unnormalized, correctly signed F_0, F'_0.
///
/// Steed's method when accurate; otherwise the 1F1 series for F with
/// G = (1 - q F^2) / (F' - p F), or a rho shift to the turning point
/// rho' = 2 eta with Taylor integration of G inward. See normalize_l0 in
/// crates/nucrust-special/src/coulomb.rs. Returns 0 on success.
__device__ int coul_normalize_l0(double eta, double rho, double fu, double fpu,
                                 double* f0, double* fp0, double* g0, double* gp0) {
    double p = 0.0, q = 0.0, q_err = 0.0;
    int cf2_ok = coul_cf2(0.0, eta, rho, &p, &q, &q_err) == 0;

    double steed_err = (cf2_ok && q > 0.0) ? q_err / q : COUL_DBL_MAX;
    if (steed_err <= COUL_NORM_TOL && rho > COUL_RHO_SMALL) {
        coul_steed_normalize(fu, fpu, p, q, f0, fp0, g0, gp0);
        return 0;
    }

    double sf = 0.0, sfp = 0.0, s_err = 0.0;
    int series_ok = coul_power_series_f0(eta, rho, &sf, &sfp, &s_err) == 0
                    && s_err < steed_err;
    if (series_ok && cf2_ok
        && (s_err <= COUL_NORM_TOL || eta <= 0.0 || rho >= 2.0 * eta)) {
        *f0 = sf;
        *fp0 = sfp;
        *g0 = (1.0 - q * sf * sf) / (sfp - p * sf);
        *gp0 = p * (*g0) - q * sf;
        return 0;
    }

    // rho shift: only reachable for eta > 0 inside the l = 0 turning point.
    if (eta > 0.0 && rho < 2.0 * eta) {
        double rho_tp = 2.0 * eta;
        double p_tp, q_tp, q_err_tp, f_tp, sign_tp;
        if (coul_cf2(0.0, eta, rho_tp, &p_tp, &q_tp, &q_err_tp) == 0) {
            if (coul_cf1(0.0, eta, rho_tp, &f_tp, &sign_tp) != 0) return COUL_ERR_CF1;
            double ft, fpt, g, gp;
            coul_steed_normalize(sign_tp, sign_tp * f_tp, p_tp, q_tp, &ft, &fpt, &g, &gp);
            if (coul_taylor_integrate_l0(eta, rho_tp, rho, &g, &gp) == 0) {
                // Wronskian: (F'_u G - F_u G') / s = 1
                double s = fpu * g - fu * gp;
                *f0 = fu / s;
                *fp0 = fpu / s;
                *g0 = g;
                *gp0 = gp;
                return 0;
            }
        }
    }

    // Fall back to the best available estimate.
    if (series_ok && cf2_ok) {
        *f0 = sf;
        *fp0 = sfp;
        *g0 = (1.0 - q * sf * sf) / (sfp - p * sf);
        *gp0 = p * (*g0) - q * sf;
        return 0;
    }
    if (cf2_ok && q > 0.0) {
        coul_steed_normalize(fu, fpu, p, q, f0, fp0, g0, gp0);
        return 0;
    }
    return COUL_ERR_CF2;
}

// ======================== Public device function ========================

/// Compute Coulomb wave functions F_l, G_l, F'_l, G'_l at (eta, rho, l),
/// with their true signs (the convention of DLMF chapter 33 and mpmath).
///
/// Returns COUL_OK (0) on success, COUL_ERR_CF1 / COUL_ERR_CF2 (< 0) if a
/// continued fraction failed to converge; the outputs are then NaN.
/// rho <= 0 returns F = F' = 0, G = +inf, G' = -inf (the rho -> 0+ limit).
__device__ int coulomb_wave_device(
    double eta, double rho, int l,
    double* F, double* G, double* Fp, double* Gp
) {
    if (rho <= 0.0) {
        *F = 0.0; *G = COUL_DBL_MAX * 2.0; *Fp = 0.0; *Gp = -COUL_DBL_MAX * 2.0;
        return COUL_OK;
    }
    double qnan = nan("");
    *F = qnan; *G = qnan; *Fp = qnan; *Gp = qnan;

    // Step 1: F_l, F'_l up to a positive factor, with the correct sign.
    double f_top, sign_top;
    if (coul_cf1((double)l, eta, rho, &f_top, &sign_top) != 0) return COUL_ERR_CF1;

    const double BIG = 1e250;
    double fl = sign_top;           // F_l (unnormalized, rescaled with fu)
    double fpl = sign_top * f_top;  // F'_l
    double fu = fl, fpu = fpl;      // running F_k, F'_k, k = l..0
    for (int k = l; k >= 1; k--) {
        double kf = (double)k;
        double r = sqrt(1.0 + (eta / kf) * (eta / kf));  // R_k
        double s = kf / rho + eta / kf;                   // S_k
        double f_below = (fpu + s * fu) / r;              // F_{k-1}
        double fp_below = s * f_below - r * fu;           // F'_{k-1}
        fu = f_below;
        fpu = fp_below;
        if (fmax(fabs(fu), fabs(fpu)) > BIG) {
            fu /= BIG; fpu /= BIG;
            fl /= BIG; fpl /= BIG;
        }
    }

    // Step 2: absolute normalization and G at l = 0.
    double f0, fp0, g, gp;
    int st = coul_normalize_l0(eta, rho, fu, fpu, &f0, &fp0, &g, &gp);
    if (st != 0) return st;

    if (l == 0) {
        *F = f0; *Fp = fp0;
    } else {
        double scale = fabs(fu) >= fabs(fpu) ? f0 / fu : fp0 / fpu;
        *F = fl * scale;
        *Fp = fpl * scale;
    }

    // Step 3: G_l by upward recurrence.
    for (int k = 1; k <= l; k++) {
        double kf = (double)k;
        double r = sqrt(1.0 + (eta / kf) * (eta / kf));  // R_k
        double s = kf / rho + eta / kf;                   // S_k
        double g_next = (s * g - gp) / r;                 // G_k
        gp = r * g - s * g_next;                          // G'_k
        g = g_next;
    }
    *G = g;
    *Gp = gp;
    return COUL_OK;
}

/// Batch kernel: Coulomb functions for a batch of (eta, rho, l) points.
/// fp_out, gp_out and status_out may be null.
extern "C" __global__ void coulomb_batch_device(
    const double* __restrict__ eta_arr,
    const double* __restrict__ rho_arr,
    const int*    __restrict__ l_arr,
    double* __restrict__ f_out,
    double* __restrict__ g_out,
    double* __restrict__ fp_out,
    double* __restrict__ gp_out,
    int*    __restrict__ status_out,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double F, G, Fp, Gp;
    int st = coulomb_wave_device(eta_arr[tid], rho_arr[tid], l_arr[tid], &F, &G, &Fp, &Gp);
    f_out[tid] = F;
    g_out[tid] = G;
    if (fp_out) fp_out[tid] = Fp;
    if (gp_out) gp_out[tid] = Gp;
    if (status_out) status_out[tid] = st;
}

#endif // COULOMB_DEVICE_CU
