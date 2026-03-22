/**
 * R-matrix batched LU solve kernel.
 *
 * Each CUDA thread computes the R-matrix cross section for one energy point:
 * 1. Build R-matrix: R_{cc'} = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
 * 2. Solve (1 - R*L0)^{-1} via in-thread LU decomposition (small matrices, N <= 10)
 * 3. Construct collision matrix U_{cc'}
 * 4. Extract cross section: sigma = pi/k^2 * |1 - U_00|^2
 *
 * hipify-clang compatible.
 */

// Maximum number of channels supported per thread
#define MAX_CH 10

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
    return make_cd((a.re*b.re + a.im*b.im)/d, (a.im*b.re - a.re*b.im)/d);
}
__device__ double cd_abs2(cdouble z) {
    return z.re*z.re + z.im*z.im;
}

// In-thread LU decomposition and solve for small matrices (N <= MAX_CH)
// Solves A*x = b for each column of b (identity matrix → A^{-1})
__device__ void invert_small(cdouble* A, cdouble* Ainv, int n) {
    // Initialize Ainv to identity
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Ainv[i*n+j] = make_cd(i == j ? 1.0 : 0.0, 0.0);

    // Gauss-Jordan elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int pivot = col;
        double max_val = cd_abs2(A[col*n+col]);
        for (int row = col+1; row < n; row++) {
            double val = cd_abs2(A[row*n+col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }

        // Swap rows
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                cdouble tmp = A[col*n+j]; A[col*n+j] = A[pivot*n+j]; A[pivot*n+j] = tmp;
                tmp = Ainv[col*n+j]; Ainv[col*n+j] = Ainv[pivot*n+j]; Ainv[pivot*n+j] = tmp;
            }
        }

        // Scale pivot row
        cdouble pivot_val = A[col*n+col];
        if (cd_abs2(pivot_val) < 1e-30) continue; // Singular
        for (int j = 0; j < n; j++) {
            A[col*n+j] = cd_div(A[col*n+j], pivot_val);
            Ainv[col*n+j] = cd_div(Ainv[col*n+j], pivot_val);
        }

        // Eliminate column
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            cdouble factor = A[row*n+col];
            for (int j = 0; j < n; j++) {
                A[row*n+j] = cd_sub(A[row*n+j], cd_mul(factor, A[col*n+j]));
                Ainv[row*n+j] = cd_sub(Ainv[row*n+j], cd_mul(factor, Ainv[col*n+j]));
            }
        }
    }
}

/**
 * Batch R-matrix cross section computation.
 *
 * @param energies       [n_e] energy grid (MeV)
 * @param gamma_widths   [n_levels * n_ch] reduced width amplitudes gamma_{lc}
 * @param level_energies [n_levels] pole energies E_lambda (MeV)
 * @param penetrabilities [n_ch] penetrability P_c at representative energy
 * @param shift_funcs    [n_ch] shift function S_c
 * @param n_ch           number of channels
 * @param n_levels       number of R-matrix levels
 * @param k_sq           k^2 for the entrance channel (fm^{-2})
 * @param sigma_out      [n_e] output: reaction cross section (mb)
 * @param n_e            number of energy points
 */
extern "C" __global__ void rmatrix_solve(
    const double* __restrict__ energies,
    const double* __restrict__ gamma_widths,
    const double* __restrict__ level_energies,
    const double* __restrict__ penetrabilities,
    const double* __restrict__ shift_funcs,
    int n_ch, int n_levels,
    double k_sq,
    double* __restrict__ sigma_out,
    int n_e
) {
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (e_idx >= n_e) return;
    if (n_ch > MAX_CH || n_ch < 1) {
        sigma_out[e_idx] = 0.0;
        return;
    }

    double energy = energies[e_idx];

    // Build R-matrix: R_{cc'} = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
    cdouble R[MAX_CH * MAX_CH];
    for (int i = 0; i < n_ch*n_ch; i++) R[i] = make_cd(0.0, 0.0);

    for (int lam = 0; lam < n_levels; lam++) {
        double denom = level_energies[lam] - energy;
        if (fabs(denom) < 1e-30) denom = 1e-30; // Avoid division by zero near pole
        for (int c = 0; c < n_ch; c++) {
            double g_c = gamma_widths[lam * n_ch + c];
            for (int cp = 0; cp < n_ch; cp++) {
                double g_cp = gamma_widths[lam * n_ch + cp];
                R[c*n_ch+cp].re += g_c * g_cp / denom;
            }
        }
    }

    // Build (1 - R*L0) where L0 = S + iP (shift + i*penetrability)
    // M = I - R * diag(L0)
    cdouble M[MAX_CH * MAX_CH];
    for (int c = 0; c < n_ch; c++) {
        cdouble L0_c = make_cd(shift_funcs[c], penetrabilities[c]);
        for (int cp = 0; cp < n_ch; cp++) {
            cdouble rl = cd_mul(R[c*n_ch+cp], L0_c);
            M[c*n_ch+cp] = (c == cp) ?
                cd_sub(make_cd(1.0, 0.0), rl) :
                cd_sub(make_cd(0.0, 0.0), rl);
        }
    }

    // Invert M → M^{-1}
    cdouble Minv[MAX_CH * MAX_CH];
    invert_small(M, Minv, n_ch);

    // A = M^{-1} * R
    cdouble A[MAX_CH * MAX_CH];
    for (int c = 0; c < n_ch; c++) {
        for (int cp = 0; cp < n_ch; cp++) {
            A[c*n_ch+cp] = make_cd(0.0, 0.0);
            for (int k = 0; k < n_ch; k++) {
                A[c*n_ch+cp] = cd_add(A[c*n_ch+cp], cd_mul(Minv[c*n_ch+k], R[k*n_ch+cp]));
            }
        }
    }

    // U_{cc'} = delta_{cc'} + 2i * sqrt(P_c) * A_{cc'} * sqrt(P_c')
    // (simplified: omitting Coulomb phase factors for now)
    // Reaction cross section: sigma_r = pi/k^2 * (1 - |U_00|^2)
    double p0 = penetrabilities[0];
    double sqrt_p0 = sqrt(fabs(p0));

    cdouble u00 = make_cd(1.0, 0.0);
    u00 = cd_add(u00, cd_mul(make_cd(0.0, 2.0 * sqrt_p0 * sqrt_p0), A[0]));

    double s_norm_sq = cd_abs2(u00);
    double sigma_r = 3.14159265358979323846 / k_sq * (1.0 - s_norm_sq);
    sigma_r *= 10.0; // fm^2 → mb
    if (sigma_r < 0.0) sigma_r = 0.0;

    sigma_out[e_idx] = sigma_r;
}
