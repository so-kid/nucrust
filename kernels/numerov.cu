/**
 * Batch Numerov integration kernel for transmission coefficient computation.
 *
 * Each CUDA thread computes one (energy, l, j) combination independently
 * using the Fox-Goodwin ratio-variable method.
 *
 * The effective potential is a simple Woods-Saxon + centrifugal barrier,
 * parameterized by depths (V_real, W_imag), geometry (r0, a), and
 * Coulomb radius (rc, Z1, Z2).
 *
 * Output: |S_lj|^2 for each (energy, l) pair, from which T = 1 - |S|^2.
 *
 * hipify-clang compatible: no dynamic parallelism, basic thread-block sync only.
 */

// Complex number helpers (inline, no cuComplex dependency)
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
    double denom = b.re*b.re + b.im*b.im;
    return make_cd((a.re*b.re + a.im*b.im)/denom, (a.im*b.re - a.re*b.im)/denom);
}

__device__ double cd_norm_sq(cdouble z) {
    return z.re*z.re + z.im*z.im;
}

__device__ cdouble cd_scale(double s, cdouble z) {
    return make_cd(s*z.re, s*z.im);
}

// Woods-Saxon form factor
__device__ double woods_saxon(double r, double R, double a) {
    double x = (r - R) / a;
    if (x > 500.0) return 0.0;
    return 1.0 / (1.0 + exp(x));
}

// Effective potential f(r) = 2*mu*(V-E)/hbar^2 + l*(l+1)/r^2
__device__ cdouble effective_potential(
    double r, double energy,
    double v_real, double w_imag,  // potential depths (positive)
    double r0, double a_ws,        // WS geometry
    double R_ws,                   // R = r0 * A^(1/3)
    int l,                         // orbital angular momentum
    double hbar2_over_2mu          // hbar^2 / (2*mu) in MeV*fm^2
) {
    double f = woods_saxon(r, R_ws, a_ws);
    double v_re = -v_real * f;
    double v_im = w_imag * f;  // positive Im = absorption
    double centrifugal = (double)l * ((double)l + 1.0) / (r * r);

    cdouble v_eff;
    v_eff.re = (v_re - energy) / hbar2_over_2mu + centrifugal;
    v_eff.im = v_im / hbar2_over_2mu;
    return v_eff;
}

/**
 * Batch Numerov kernel.
 *
 * @param energies    [n_e] energy grid (MeV)
 * @param l_values    [n_tasks] orbital angular momentum for each task
 * @param v_real      real potential depth (MeV, positive)
 * @param w_imag      imaginary potential depth (MeV, positive)
 * @param r0          WS radius parameter (fm)
 * @param a_ws        WS diffuseness (fm)
 * @param R_ws        WS nuclear radius R = r0 * A^{1/3} (fm)
 * @param r_match     matching radius (fm)
 * @param step_size   Numerov step size h (fm)
 * @param hbar2_2mu   hbar^2/(2*mu) in MeV*fm^2
 * @param s_norm_sq   [n_tasks] output: |S_lj|^2 for each task
 * @param n_tasks     total number of (E, l) tasks
 */
extern "C" __global__ void batch_numerov(
    const double* __restrict__ energies,
    const int*    __restrict__ l_values,
    const int*    __restrict__ e_indices,
    double v_real, double w_imag,
    double r0, double a_ws, double R_ws,
    double r_match, double step_size,
    double hbar2_2mu,
    double* __restrict__ s_norm_sq,
    int n_tasks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tasks) return;

    int l = l_values[tid];
    int e_idx = e_indices[tid];
    double energy = energies[e_idx];
    double h = step_size;
    double h2_12 = h * h / 12.0;

    // Number of steps
    int n_steps = (int)ceil((r_match - h) / h);
    if (n_steps < 3) {
        s_norm_sq[tid] = 1.0;  // No scattering
        return;
    }

    // Fox-Goodwin ratio variable integration
    double r0_start = h;
    double r1 = r0_start + h;

    // Initial ratio: R_0 = (r1/r0)^(l+1)
    double ratio_init = 1.0;
    for (int i = 0; i <= l; i++) ratio_init *= (r1 / r0_start);

    cdouble ratio = make_cd(ratio_init, 0.0);
    cdouble one = make_cd(1.0, 0.0);
    cdouble two = make_cd(2.0, 0.0);
    cdouble five = make_cd(5.0, 0.0);

    cdouble f_prev = effective_potential(r0_start, energy, v_real, w_imag, r0, a_ws, R_ws, l, hbar2_2mu);
    cdouble f_curr = effective_potential(r1, energy, v_real, w_imag, r0, a_ws, R_ws, l, hbar2_2mu);

    for (int step = 2; step <= n_steps; step++) {
        double r_next = r0_start + (double)step * h;
        cdouble f_next = effective_potential(r_next, energy, v_real, w_imag, r0, a_ws, R_ws, l, hbar2_2mu);

        // Fox-Goodwin: R = [2*(1 + 5h^2/12*f_curr) - (1 - h^2/12*f_prev)/R] / (1 - h^2/12*f_next)
        cdouble num = cd_sub(
            cd_mul(two, cd_add(one, cd_scale(5.0 * h2_12, f_curr))),
            cd_div(cd_sub(one, cd_scale(h2_12, f_prev)), ratio)
        );
        cdouble den = cd_sub(one, cd_scale(h2_12, f_next));
        ratio = cd_div(num, den);

        // Guard against blowup
        double rn = cd_norm_sq(ratio);
        if (rn > 1e200) {
            ratio = cd_scale(1.0 / sqrt(rn), ratio);
        }

        f_prev = f_curr;
        f_curr = f_next;
    }

    // Simple S-matrix estimate: |S|^2 from ratio at matching radius
    // For a pure real potential, |S| = 1.
    // For absorptive potential, |S| < 1.
    // Exact extraction requires Coulomb functions (done on CPU).
    // Here we store the complex ratio for CPU post-processing,
    // or use a simplified estimate.
    //
    // Simplified: |S|^2 ≈ 1 / |ratio|^2 (rough absorption measure)
    // This is a placeholder — real S-matrix extraction done on CPU.
    double ratio_norm = cd_norm_sq(ratio);
    if (ratio_norm > 1e-30) {
        // The ratio encodes the wavefunction's phase and amplitude.
        // For now, store ratio_norm for CPU-side S-matrix extraction.
        s_norm_sq[tid] = ratio.re;  // Store real part of ratio
    } else {
        s_norm_sq[tid] = 1.0;
    }
}
