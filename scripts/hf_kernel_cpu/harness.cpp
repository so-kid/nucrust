/* Runs the hf_summation kernel source on the CPU for n + Fe-56 and compares
 * with the values produced by nucrust-hf's CPU Hauser-Feshbach for the same
 * inputs (test hf_reference::tests::kernel_replica_matches_cpu_hauser_feshbach). */
#include <cstdio>
#include <vector>

int main() {
    const int n_e = 3, n_l = 4;
    double energies[n_e] = {0.1, 1.0, 3.0};
    const double base[n_l][2] = {{0.0, 0.93}, {0.21, 0.20}, {0.35, 0.38}, {0.004, 0.003}};
    std::vector<double> tc(n_l * 2 * n_e, 0.0);
    for (int l = 0; l < n_l; l++)
        for (int j = 0; j < 2; j++)
            for (int e = 0; e < n_e; e++) {
                double t = base[l][j] * (0.5 + 0.25 * e);
                tc[l * 2 * n_e + j * n_e + e] = t < 1.0 ? t : 1.0;
            }
    const double mu = 1.008664916 * 56.0 / (1.008664916 + 56.0);
    const double hbar_c = 197.3269804, amu = 931.49410372;
    const double hbar2_2mu = hbar_c * hbar_c / (2.0 * mu * amu);

    double sigma_cn[n_e], sigma_gamma[n_e];
    blockDim.x = 1;
    threadIdx.x = 0;
    for (int e = 0; e < n_e; e++) {
        blockIdx.x = e;
        hf_summation(energies, tc.data(), n_e, n_l, 20, 1, 0, 1, 7.646, 57.0,
                     6.21, 0.88, -1.16, 16.36, 4.58, 136.0, hbar2_2mu,
                     sigma_cn, sigma_gamma);
    }

    /* nucrust-hf hauser_feshbach (CPU) for the same case, in mb. */
    const double cpu_cn[n_e] = {1.12791972347277988e4, 1.69187958520916936e3, 7.51946482315186358e2};
    const double cpu_gamma[n_e] = {2.44445543941670024e2, 6.22078144879991513e1, 1.37978304112162846e2};
    int fail = 0;
    for (int e = 0; e < n_e; e++) {
        double r_cn = fabs(sigma_cn[e] / cpu_cn[e] - 1.0);
        double r_g = fabs(sigma_gamma[e] / cpu_gamma[e] - 1.0);
        printf("E = %4.1f MeV: sigma_CN = %.12e mb (rel %.1e), sigma_gamma = %.12e mb (rel %.1e)\n",
               energies[e], sigma_cn[e], r_cn, sigma_gamma[e], r_g);
        if (r_cn > 1e-12 || r_g > 1e-12) fail = 1;
    }
    puts(fail ? "FAIL" : "OK");
    return fail;
}
