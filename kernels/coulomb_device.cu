/**
 * On-the-fly Coulomb wave function computation as __device__ functions.
 *
 * Implements Thompson-Barnett CF1 + CF2 + Steed method directly on GPU.
 * Each thread can compute F_l, G_l at arbitrary (eta, rho, l) without
 * requiring a pre-computed lookup table.
 *
 * For use by the Numerov kernel's S-matrix extraction step.
 *
 * hipify-clang compatible.
 */

#ifndef COULOMB_DEVICE_CU
#define COULOMB_DEVICE_CU

struct cdouble {
    double re, im;
};

__device__ cdouble make_cd(double re, double im) {
    cdouble z; z.re = re; z.im = im; return z;
}
__device__ cdouble cd_add(cdouble a, cdouble b) {
    return make_cd(a.re + b.re, a.im + b.im);
}
__device__ cdouble cd_sub(cdouble a, cdouble b) {
    return make_cd(a.re - b.re, a.im - b.im);
}
__device__ cdouble cd_mul(cdouble a, cdouble b) {
    return make_cd(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
}
__device__ cdouble cd_div(cdouble a, cdouble b) {
    double d = b.re*b.re + b.im*b.im;
    if (d < 1e-300) return make_cd(0.0, 0.0);
    return make_cd((a.re*b.re + a.im*b.im)/d, (a.im*b.re - a.re*b.im)/d);
}
__device__ double cd_norm(cdouble z) {
    return sqrt(z.re*z.re + z.im*z.im);
}

/// Modified Lentz method for real continued fractions.
/// f = b0 + a1/(b1 + a2/(b2 + ...))
/// Returns the continued fraction value.
__device__ double cf_lentz_real(
    int l, double eta, double rho,
    int is_cf1,  // 1 = CF1 (F'/F), 0 = CF2 (not used here)
    int max_iter
) {
    const double SMALL = 1e-50;
    const double EPS = 1e-12;

    // CF1: b0 = S_{l+1} = (l+1)/rho + eta/(l+1)
    double lf = (double)l;
    double b0 = (lf + 1.0) / rho + eta / (lf + 1.0);

    double h = b0;
    if (fabs(h) < SMALL) h = SMALL;
    double d = 0.0;
    double c = h;

    for (int n = 1; n <= max_iter; n++) {
        double ln = lf + (double)n;
        // a_n = -(1 + eta^2/ln^2)
        double an = -(1.0 + (eta/ln)*(eta/ln));
        // b_n = T_{l+n} = ln/rho + eta/ln + (ln+1)/rho + eta/(ln+1)
        double bn = ln/rho + eta/ln + (ln+1.0)/rho + eta/(ln+1.0);

        d = bn + an * d;
        if (fabs(d) < SMALL) d = SMALL;
        c = bn + an / c;
        if (fabs(c) < SMALL) c = SMALL;
        d = 1.0 / d;
        double delta = c * d;
        h *= delta;

        if (fabs(delta - 1.0) < EPS) break;
    }

    return h;
}

/// CF2: complex continued fraction for H+'/H+ = p + iq
__device__ void cf2_complex(
    int l, double eta, double rho,
    double* p_out, double* q_out,
    int max_iter
) {
    const double SMALL = 1e-50;
    const double EPS = 1e-12;
    double lf = (double)l;

    // b0 = i*(1 - eta/rho)
    cdouble h = make_cd(0.0, 1.0 - eta/rho);
    if (cd_norm(h) < SMALL) h = make_cd(SMALL, 0.0);
    cdouble d_val = make_cd(0.0, 0.0);
    cdouble c_val = h;

    for (int n = 1; n <= max_iter; n++) {
        double nf = (double)n;
        // a_n = (n+l+i*eta)*(n-1-l+i*eta)
        cdouble an = cd_mul(make_cd(nf+lf, eta), make_cd(nf-1.0-lf, eta));
        // b_n = 2*(rho - eta + n*i)
        cdouble bn = make_cd(2.0*(rho - eta), 2.0*nf);

        d_val = cd_add(bn, cd_mul(an, d_val));
        if (cd_norm(d_val) < SMALL) d_val = make_cd(SMALL, 0.0);
        c_val = cd_add(bn, cd_div(an, c_val));
        if (cd_norm(c_val) < SMALL) c_val = make_cd(SMALL, 0.0);
        d_val = cd_div(make_cd(1.0, 0.0), d_val);
        cdouble delta = cd_mul(c_val, d_val);
        h = cd_mul(h, delta);

        cdouble diff = cd_sub(delta, make_cd(1.0, 0.0));
        if (cd_norm(diff) < EPS) break;
    }

    *p_out = h.re;
    *q_out = h.im;
}

/// Compute Coulomb wave functions F_l, G_l, F'_l, G'_l at (eta, rho, l).
/// Uses Steed method (CF1 + CF2 + Wronskian coupling).
///
/// Returns 0 on success, -1 on failure.
__device__ int coulomb_wave_device(
    double eta, double rho, int l,
    double* F, double* G, double* Fp, double* Gp
) {
    if (rho <= 0.0) {
        *F = 0.0; *G = 1e30; *Fp = 0.0; *Gp = -1e30;
        return 0;
    }

    const int MAX_ITER = 5000; // Reduced for GPU (vs 20000 on CPU)

    // CF1: f = F'/F
    double f_ratio = cf_lentz_real(l, eta, rho, 1, MAX_ITER);

    // CF2: p + iq = H+'/H+
    double p, q;
    cf2_complex(l, eta, rho, &p, &q, MAX_ITER);

    if (fabs(q) < 1e-300) {
        // CF2 failed (forbidden region) - return approximate values
        *F = 0.0; *G = 1e10; *Fp = 0.0; *Gp = 0.0;
        return -1;
    }

    // Steed: F^2 = q / ((f-p)^2 + q^2)
    double fmp = f_ratio - p;
    double denom = fmp * fmp + q * q;
    double f_sq = fabs(q / denom);

    double f_val = sqrt(f_sq);
    // Sign from asymptotic phase (simplified)
    double lf = (double)l;
    double theta = rho - eta * log(2.0 * rho) - lf * 1.5707963267948966 /* pi/2 */;
    if (sin(theta) < 0.0) f_val = -f_val;

    double fp_val = f_ratio * f_val;
    double gamm = fmp / q;
    double g_val = gamm * f_val;
    double gp_val = p * g_val - q * f_val;

    *F = f_val;
    *G = g_val;
    *Fp = fp_val;
    *Gp = gp_val;
    return 0;
}

/// Test kernel: compute Coulomb functions for a batch of (eta, rho, l) points.
extern "C" __global__ void coulomb_batch_device(
    const double* __restrict__ eta_arr,
    const double* __restrict__ rho_arr,
    const int*    __restrict__ l_arr,
    double* __restrict__ f_out,
    double* __restrict__ g_out,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double F, G, Fp, Gp;
    coulomb_wave_device(eta_arr[tid], rho_arr[tid], l_arr[tid], &F, &G, &Fp, &Gp);
    f_out[tid] = F;
    g_out[tid] = G;
}

#endif // COULOMB_DEVICE_CU
