/**
 * Hauser-Feshbach J-pi summation kernel.
 *
 * Each thread computes the HF cross sections for one energy point:
 *   sigma_CN(E)    = pi/k^2 * sum_{J,pi} g_J * T_a(J,pi)
 *   sigma_gamma(E) = pi/k^2 * sum_{J,pi} g_J * T_a(J,pi) * T_gamma(J,pi) / T_total(J,pi)
 * with g_J = (2J+1) / ((2s+1)(2I+1)) and T_total = T_a + T_gamma
 * (compound elastic + capture).
 *
 * Mirrors nucrust-hf `hauser_feshbach` with exit_channels = [gamma], no exit
 * particle channels, no discrete levels, a constant-temperature level density
 * and the Standard Lorentzian GSF without scissors mode. The host function
 * `hf_reference::hf_summation_host` replicates this arithmetic line by line.
 *
 * Transmission coefficient layout ([l][j_idx][e_idx], 2 j-slots for every l):
 *   tc_data[l * 2 * n_e + j_idx * n_e + e_idx]
 *   spin-1/2 projectile: j_idx = 0 -> j = l - 1/2, j_idx = 1 -> j = l + 1/2
 *                        (slot 0 of l = 0 is unused)
 *   spin-0 projectile:   j = l stored in j_idx = 1
 *
 * hipify-clang compatible: extern "C" __global__, no dynamic parallelism,
 * no cuComplex, no textures.
 */

#define HF_PI 3.14159265358979323846
/* 0.1 fm^2/mb / (3 pi^2 (hbar c)^2) in mb^-1 MeV^-2 (gsf.rs GSF_E1_CONST). */
#define HF_GSF_E1_CONST 8.673733205921499e-8
/* SLO single-particle M1 / E2 strengths (gsf.rs StandardLorentzian). */
#define HF_GSF_M1_DEFAULT 1.0e-9
#define HF_GSF_E2_COEFF 5.2e-8
/* nucrust-hf lib.rs NUMERICAL_FLOOR / MIN_EXCITATION. */
#define HF_NUMERICAL_FLOOR 1e-30
#define HF_MIN_EXCITATION 0.1
/* Number of gamma-energy panels in the continuum integral (hf.rs). */
#define HF_N_GAMMA_POINTS 50

/**
 * Sum T_{lj}(E) over all (l, j) that couple the compound state (2J, parity)
 * with a target state (2I, parity_i): (-1)^l * parity_i == parity and
 * |j - I| <= J <= j + I.  Parities are +1 / -1.
 */
__device__ double hf_entrance_transmission(
    const double* __restrict__ tc_data, int n_e, int n_l, int e_idx,
    int two_j, int parity, int proj_two_s, int two_i, int parity_i
) {
    double t_sum = 0.0;
    for (int l = 0; l < n_l; l++) {
        int orbital_parity = (l % 2 == 0) ? 1 : -1;
        if (orbital_parity * parity_i != parity) continue;
        for (int j_idx = 0; j_idx < 2; j_idx++) {
            int two_j_particle;
            if (proj_two_s == 0) {
                if (j_idx == 0) continue; /* spin-0: single j = l in slot 1 */
                two_j_particle = 2 * l;
            } else {
                two_j_particle = 2 * l + (2 * j_idx - 1);
            }
            if (two_j_particle < 0) continue;

            /* Angular momentum coupling J = j + I. */
            int diff = two_j_particle - two_i;
            if (diff < 0) diff = -diff;
            if ((two_j_particle + two_i + two_j) % 2 != 0
                || two_j < diff
                || two_j > two_j_particle + two_i) {
                continue;
            }
            t_sum += tc_data[l * 2 * n_e + j_idx * n_e + e_idx];
        }
    }
    return t_sum;
}

/**
 * Constant-temperature level density rho(U, J, pi) (nld.rs ConstantTemperature):
 *   0.5 * exp((U - E0)/T)/T * (2J+1)/(2 s^2) * exp(-(J+1/2)^2 / (2 s^2)),
 *   s^2 = 0.0888 * A^{2/3} * sqrt(a * max(U, 0.01)).
 */
__device__ double hf_rho_ct(
    double excitation, int two_jf, double nld_t, double nld_e0, double nld_a, double mass_a
) {
    double u = excitation - nld_e0;
    if (u < 0.0) return 0.0;
    double rho_tot = (1.0 / nld_t) * exp(u / nld_t);
    double u_cut = fmax(excitation, 0.01);
    double sigma_sq = 0.0888 * pow(mass_a, 2.0 / 3.0) * sqrt(nld_a * u_cut);
    if (sigma_sq <= 0.0) return 0.0;
    double spin = (double)two_jf / 2.0;
    double j_half = spin + 0.5;
    double f_spin = (2.0 * spin + 1.0) / (2.0 * sigma_sq)
        * exp(-j_half * j_half / (2.0 * sigma_sq));
    return 0.5 * rho_tot * f_spin;
}

/**
 * Standard Lorentzian strength f_XL(E_gamma) in MeV^-(2L+1).
 * multipole: 0 = E1, 1 = M1, 2 = E2.
 */
__device__ double hf_gsf_slo(
    double e_gamma, int multipole, double e_gdr, double gamma_gdr, double sigma_gdr
) {
    if (e_gamma <= 0.0) return 0.0;
    if (multipole == 0) {
        double e2 = e_gamma * e_gamma;
        double e02 = e_gdr * e_gdr;
        double g2 = gamma_gdr * gamma_gdr;
        double denom = (e2 - e02) * (e2 - e02) + e2 * g2;
        if (denom == 0.0) return 0.0;
        return HF_GSF_E1_CONST * sigma_gdr * gamma_gdr * e_gamma * gamma_gdr / denom;
    }
    if (multipole == 1) return HF_GSF_M1_DEFAULT;
    return HF_GSF_E2_COEFF / (e_gdr * e_gdr);
}

/**
 * Gamma transmission of the compound state (2J, pi) at excitation U:
 *   T_gamma = 2 pi * sum_{XL} int_0^U f_XL(E_g) E_g^{2L+1}
 *             sum_{J'=|J-L|}^{J+L} rho(U - E_g, J', pi') dE_g
 * (E1, M1, E2; rectangle rule with HF_N_GAMMA_POINTS panels).
 */
__device__ double hf_gamma_transmission(
    double excitation, int two_j,
    double nld_t, double nld_e0, double nld_a, double mass_a,
    double e_gdr, double gamma_gdr, double sigma_gdr
) {
    if (excitation < HF_MIN_EXCITATION) return 0.0;
    double t_gamma = 0.0;
    double e_cont_max = excitation; /* no discrete levels: E_complete = 0 */
    if (e_cont_max > 0.1) {
        double de = e_cont_max / (double)HF_N_GAMMA_POINTS;
        for (int multipole = 0; multipole < 3; multipole++) {
            /* E1, M1: L = 1; E2: L = 2. The final parity (-pi for E1, pi otherwise)
             * does not enter: the CT density carries a flat 1/2 parity factor. */
            int l_order = (multipole == 2) ? 2 : 1;
            for (int i = 1; i < HF_N_GAMMA_POINTS; i++) {
                double e_gamma = (double)i * de;
                double u_residual = excitation - e_gamma;
                if (u_residual < 0.0) break;

                double f_xl = hf_gsf_slo(e_gamma, multipole, e_gdr, gamma_gdr, sigma_gdr);
                double e_factor = e_gamma * e_gamma * e_gamma;
                if (l_order == 2) e_factor *= e_gamma * e_gamma;

                int two_l = 2 * l_order;
                int j_min = two_j - two_l;
                if (j_min < 0) j_min = -j_min;
                int j_max = two_j + two_l;
                for (int two_jf = j_min; two_jf <= j_max; two_jf += 2) {
                    double rho = hf_rho_ct(u_residual, two_jf, nld_t, nld_e0, nld_a, mass_a);
                    t_gamma += f_xl * e_factor * rho * de;
                }
            }
        }
    }
    return 2.0 * HF_PI * t_gamma;
}

/**
 * Compute HF cross sections for a batch of energies.
 *
 * @param energies        [n_e] energy grid (MeV)
 * @param tc_data         [n_l * 2 * n_e] transmission coefficients T_{l,j}(E)
 * @param n_e             number of energy points
 * @param n_l             number of partial waves (l = 0..n_l-1)
 * @param two_j_max       maximum 2*J for summation
 * @param proj_two_s      2 * projectile spin
 * @param target_two_i    2 * target ground-state spin
 * @param target_parity   target ground-state parity (+1 / -1)
 * @param q_value         Q-value of capture: U = E + Q (MeV)
 * @param compound_a      mass number of the compound nucleus (spin cutoff)
 * @param nld_a           level density parameter a (1/MeV), spin cutoff only
 * @param nld_t           CT temperature T (MeV)
 * @param nld_e0          CT energy shift E0 (MeV)
 * @param gsf_e_gdr       GDR energy (MeV)
 * @param gsf_gamma_gdr   GDR width (MeV)
 * @param gsf_sigma_gdr   GDR peak cross section (mb)
 * @param hbar2_2mu       hbar^2/(2*mu) in MeV*fm^2 (k^2 = E / hbar2_2mu)
 * @param sigma_out       [n_e] output: compound formation cross section (mb)
 * @param sigma_gamma_out [n_e] output: capture cross section (mb)
 */
extern "C" __global__ void hf_summation(
    const double* __restrict__ energies,
    const double* __restrict__ tc_data,
    int n_e, int n_l, int two_j_max,
    int proj_two_s, int target_two_i, int target_parity,
    double q_value, double compound_a,
    double nld_a, double nld_t, double nld_e0,
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

    /* pi/k^2 in mb (1 fm^2 = 10 mb). */
    double k_sq = energy / hbar2_2mu;
    double pi_over_k2 = HF_PI / k_sq * 10.0;

    double excitation = energy + q_value;
    double spin_weight = (double)(proj_two_s + 1) * (double)(target_two_i + 1);
    double sig_cn = 0.0;
    double sig_gam = 0.0;

    for (int two_j = 0; two_j <= two_j_max; two_j++) {
        for (int ip = 0; ip < 2; ip++) {
            int parity = (ip == 0) ? 1 : -1;

            double t_a = hf_entrance_transmission(
                tc_data, n_e, n_l, e_idx, two_j, parity,
                proj_two_s, target_two_i, target_parity);
            if (t_a < HF_NUMERICAL_FLOOR) continue;

            double t_gamma = hf_gamma_transmission(
                excitation, two_j, nld_t, nld_e0, nld_a, compound_a,
                gsf_e_gdr, gsf_gamma_gdr, gsf_sigma_gdr);

            double t_total = t_a + t_gamma;
            if (t_total < HF_NUMERICAL_FLOOR) continue;

            double g_j = (double)(two_j + 1) / spin_weight;
            double prefactor = pi_over_k2 * g_j;
            sig_cn += prefactor * t_a;
            sig_gam += prefactor * t_a * t_gamma / t_total;
        }
    }

    sigma_out[e_idx] = sig_cn;
    sigma_gamma_out[e_idx] = sig_gam;
}
