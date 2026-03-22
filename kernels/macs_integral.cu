/**
 * Maxwellian-Averaged Cross Section (MACS) integration kernel.
 *
 * Each thread computes MACS for one temperature point:
 *   MACS(kT) = (2 / sqrt(pi) * kT) * sum_i w_i * sigma(x_i * kT) * x_i
 *
 * Uses pre-computed Gauss-Laguerre quadrature nodes and weights.
 * Cross section sigma(E) is interpolated from a tabulated grid using
 * linear interpolation (cubic spline would be better but more complex).
 *
 * hipify-clang compatible.
 */

// Linear interpolation of cross section from energy grid
__device__ double interp_sigma(
    const double* energies, const double* sigmas, int n_e, double e
) {
    if (e <= energies[0]) return sigmas[0];
    if (e >= energies[n_e - 1]) return sigmas[n_e - 1];

    // Binary search for interval
    int lo = 0, hi = n_e - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (energies[mid] <= e) lo = mid;
        else hi = mid;
    }

    double t = (e - energies[lo]) / (energies[hi] - energies[lo]);
    return sigmas[lo] + t * (sigmas[hi] - sigmas[lo]);
}

/**
 * Compute MACS for a batch of temperatures.
 *
 * @param temperatures   [n_t] temperature grid (GK, T9)
 * @param energies       [n_e] cross section energy grid (MeV)
 * @param sigmas         [n_e] reaction cross section (mb)
 * @param n_e            number of energy points
 * @param gl_nodes       [n_quad] Gauss-Laguerre nodes
 * @param gl_weights     [n_quad] Gauss-Laguerre weights
 * @param n_quad         number of quadrature points
 * @param k_boltzmann    Boltzmann constant (MeV/GK)
 * @param macs_out       [n_t] output: MACS values (mb)
 * @param n_t            number of temperature points
 */
extern "C" __global__ void macs_integral(
    const double* __restrict__ temperatures,
    const double* __restrict__ energies,
    const double* __restrict__ sigmas,
    int n_e,
    const double* __restrict__ gl_nodes,
    const double* __restrict__ gl_weights,
    int n_quad,
    double k_boltzmann,
    double* __restrict__ macs_out,
    int n_t
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_t) return;

    double t9 = temperatures[tid];
    double kt = k_boltzmann * t9;  // kT in MeV

    if (kt < 1e-15) {
        macs_out[tid] = 0.0;
        return;
    }

    // MACS(kT) = (2 / sqrt(pi) * kT) * sum_i w_i * sigma(x_i * kT) * x_i
    double prefactor = 2.0 / (sqrt(3.14159265358979323846) * kt);
    double sum = 0.0;

    for (int i = 0; i < n_quad; i++) {
        double x = gl_nodes[i];
        double e = x * kt;
        double sigma = interp_sigma(energies, sigmas, n_e, e);
        if (sigma < 0.0) sigma = 0.0;
        sum += gl_weights[i] * sigma * x;
    }

    macs_out[tid] = prefactor * sum;
}
