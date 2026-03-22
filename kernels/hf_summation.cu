/**
 * Hauser-Feshbach J-pi summation kernel.
 *
 * Each thread computes the HF cross section for one energy point:
 *   sigma(E) = pi/k^2 * sum_{J,pi} g_J * T_a(J,pi) * T_gamma(J,pi) / T_total(J,pi)
 *
 * The transmission coefficients T_{lj}(E) are pre-computed (by GPU Numerov or CPU).
 * Gamma transmission is approximated from parameters (simplified NLD + GSF).
 *
 * hipify-clang compatible.
 */

/**
 * Compute HF cross section for a batch of energies.
 *
 * @param energies       [n_e] energy grid (MeV)
 * @param tc_data        [n_l * 2 * n_e] transmission coefficients T_{l,j}(E)
 * @param n_e            number of energy points
 * @param n_l            number of partial waves (l = 0..n_l-1)
 * @param two_j_max      maximum 2*J for summation
 * @param q_value        Q-value for (n,gamma) reaction (MeV)
 * @param nld_a          NLD level density parameter a (1/MeV)
 * @param nld_t          NLD temperature parameter T (MeV)
 * @param gsf_e_gdr      GSF GDR energy (MeV)
 * @param gsf_gamma_gdr  GSF GDR width (MeV)
 * @param gsf_sigma_gdr  GSF GDR peak cross section (mb)
 * @param hbar2_2mu      hbar^2/(2*mu) in MeV*fm^2
 * @param sigma_out      [n_e] output: compound formation cross section (mb)
 * @param sigma_gamma    [n_e] output: (n,gamma) cross section (mb)
 */
extern "C" __global__ void hf_summation(
    const double* __restrict__ energies,
    const double* __restrict__ tc_data,
    int n_e, int n_l, int two_j_max,
    double q_value,
    double nld_a, double nld_t,
    double gsf_e_gdr, double gsf_gamma_gdr, double gsf_sigma_gdr,
    double hbar2_2mu,
    double* __restrict__ sigma_out,
    double* __restrict__ sigma_gamma_out
) {
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (e_idx >= n_e) return;

    double energy = energies[e_idx];
    if (energy <= 0.0) {
        sigma_out[e_idx] = 0.0;
        sigma_gamma_out[e_idx] = 0.0;
        return;
    }

    // Wave number: k^2 = 2*mu*E / hbar^2
    double k_sq = energy / hbar2_2mu;
    double pi_over_k2 = 3.14159265358979323846 / k_sq;
    // Convert fm^2 to mb: 1 fm^2 = 10 mb
    pi_over_k2 *= 10.0;

    double excitation = energy + q_value;
    double sig_cn = 0.0;
    double sig_gam = 0.0;

    // J-pi loop
    for (int two_j = 0; two_j <= two_j_max; two_j++) {
        // Statistical weight: g = (2J+1) / ((2*s_proj+1) * (2*J_target+1))
        // For neutron (s=1/2) on even-even target (J=0): g = (2J+1) / 2
        double g_j = (double)(two_j + 1) / 2.0;

        // Entrance transmission: sum T_{lj}(E) over l satisfying triangle condition
        // |J - s| <= l <= J + s, where s = 1/2 (proj_spin_2j = 1)
        double t_entrance = 0.0;
        for (int l = 0; l < n_l; l++) {
            int two_l = 2 * l;
            // Triangle: |two_j - 1| <= 2*l <= two_j + 1
            int diff = (two_j - 1 < 0) ? (1 - two_j) : (two_j - 1);
            int sum_jl = two_j + 1;
            if (two_l < diff || two_l > sum_jl) continue;

            // j = l + 1/2: j_idx = 1, flat index = l * 2 * n_e + 1 * n_e + e_idx
            int idx_j1 = l * 2 * n_e + 1 * n_e + e_idx;
            if (idx_j1 < n_l * 2 * n_e) {
                t_entrance += tc_data[idx_j1];
            }

            // j = l - 1/2: j_idx = 0 (only if l > 0)
            if (l > 0) {
                int idx_j0 = l * 2 * n_e + 0 * n_e + e_idx;
                if (idx_j0 < n_l * 2 * n_e) {
                    t_entrance += tc_data[idx_j0];
                }
            }
        }

        if (t_entrance < 1e-30) continue;

        // Gamma transmission (simplified: E1 Lorentzian * NLD)
        // T_gamma ~ integral f_E1(E_g) * E_g^3 * rho(U - E_g) dE_g
        double t_gamma = 0.0;
        int n_gamma_pts = 20;
        double de = fmin(excitation, 20.0) / (double)n_gamma_pts;
        if (de > 0.01) {
            for (int ig = 1; ig < n_gamma_pts; ig++) {
                double e_gamma = (double)ig * de;
                double u_res = excitation - e_gamma;
                if (u_res < 0.0) break;

                // Standard Lorentzian GSF
                double denom = (e_gamma * e_gamma - gsf_e_gdr * gsf_e_gdr);
                double f_e1 = gsf_sigma_gdr * gsf_gamma_gdr * e_gamma * gsf_gamma_gdr
                    / ((denom * denom + e_gamma * e_gamma * gsf_gamma_gdr * gsf_gamma_gdr)
                       * 3.14159265358979323846);

                // E^{2L+1} for E1: E^3
                double e_factor = e_gamma * e_gamma * e_gamma;

                // Constant temperature NLD: rho ~ exp(U/T) / T
                double rho = 0.0;
                if (u_res > 0.0 && nld_t > 0.0) {
                    rho = exp(u_res / nld_t) / nld_t;
                }

                t_gamma += f_e1 * e_factor * rho * de;
            }
        }

        // Total transmission
        double t_total = t_entrance + t_gamma;
        if (t_total < 1e-30) continue;

        // HF formula
        sig_cn += pi_over_k2 * g_j * t_entrance;
        sig_gam += pi_over_k2 * g_j * t_entrance * t_gamma / t_total;
    }

    sigma_out[e_idx] = sig_cn;
    sigma_gamma_out[e_idx] = sig_gam;
}
