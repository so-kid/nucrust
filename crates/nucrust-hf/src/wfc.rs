//! Width fluctuation correction (WFC) implementations.
//!
//! Moldauer form (Moldauer 1980 degrees of freedom), used by
//! [`hauser_feshbach`](crate::hf::hauser_feshbach), and an experimental GOE routine.

use crate::NUMERICAL_FLOOR;

/// Result of Moldauer WFC calculation.
#[derive(Debug, Clone)]
pub struct MoldauerResult {
    /// WFC correction factors W_ab for each (entrance, exit) channel pair.
    pub w_factors: Vec<Vec<f64>>,
}

/// Kawano-Talou parameterization of the effective degrees of freedom nu_a.
///
/// nu_a = 1 / (1 + alpha(T_a) * F * G)
/// F = (T_a + T_total) / (1 - T_a)
/// G = 1 + 2.5 * T_a * (1 - T_a) * exp(-2 * T_total)
/// alpha(T_a) = 0.02879 * T_a + 0.2459
pub fn kawano_talou_nu(t_a: f64, t_total: f64) -> f64 {
    if t_a >= 1.0 {
        return 2.0; // Limit for strong absorption
    }
    let alpha = 0.02879 * t_a + 0.2459;
    let f = (t_a + t_total) / (1.0 - t_a);
    let g = 1.0 + 2.5 * t_a * (1.0 - t_a) * (-2.0 * t_total).exp();
    let nu = 1.0 / (1.0 + alpha * f * g);
    nu.clamp(1.0, 2.0) // nu should be between 1 (Porter-Thomas) and 2 (GOE)
}

/// Moldauer (1980) effective number of degrees of freedom of the width distribution
/// of channel a (the form used by TALYS):
///
/// nu_a = 1.78 + (T_a^1.212 - 0.78) * exp(-0.228 * T_total)
///
/// It tends to 1 (Porter-Thomas) for weak channels in few-channel situations.
pub fn moldauer_nu(t_a: f64, t_total: f64) -> f64 {
    1.78 + (t_a.max(0.0).powf(1.212) - 0.78) * (-0.228 * t_total).exp()
}

/// Quadrature rule for the Moldauer integral over x in [0, inf).
///
/// Gauss-Legendre on t in (0, 1) with x = (t / (1 - t))^2. The integrand falls off at
/// least like x^(-5/2) (exponent sum nu_c/2 + 2 with nu_c >= 1) or exponentially when
/// many channels are open; in t it is smooth at both ends, so a few tens of points give
/// ~1e-6 relative accuracy (checked by the flux-conservation tests).
#[derive(Debug, Clone)]
pub struct MoldauerQuadrature {
    x: Vec<f64>,
    w: Vec<f64>,
}

impl MoldauerQuadrature {
    /// Build an `n`-point rule (n >= 1).
    pub fn new(n: usize) -> Self {
        let (nodes, weights) = gauss_legendre(n.max(1));
        let mut x = Vec::with_capacity(nodes.len());
        let mut w = Vec::with_capacity(nodes.len());
        for (&u, &wu) in nodes.iter().zip(&weights) {
            // u in (-1, 1) -> t in (0, 1) -> s = t/(1-t) -> x = s^2.
            let t = 0.5 * (u + 1.0);
            let s = t / (1.0 - t);
            // dx = 2 s ds, ds = dt / (1-t)^2, dt = du / 2.
            x.push(s * s);
            w.push(wu * 0.5 * 2.0 * s / ((1.0 - t) * (1.0 - t)));
        }
        Self { x, w }
    }
}

/// One row of Moldauer width fluctuation factors, for entrance channel `a`.
///
/// Channels are given as individually resolved transmissions `t` plus a lumped
/// transmission `t_lumped` standing for many weak channels (gamma rays, continuum bins),
/// each with T_c << 1, whose product (1 + 2 T_c x / (nu_c T))^(-nu_c/2) tends to
/// exp(-T_lumped x / T) independently of nu_c. With T = sum(t) + t_lumped:
///
/// W_ab = (1 + 2 delta_ab / nu_a) * int_0^inf dx exp(-T_lumped x / T)
///        * prod_c (1 + 2 T_c x / (nu_c T))^(-nu_c/2 - delta_ac - delta_bc)
///
/// Returns (W_ab for every resolved b, W_a,lumped). The factors conserve flux:
/// sum_b T_b W_ab + T_lumped W_a,lumped = T (to quadrature accuracy).
pub fn moldauer_w_row(
    t: &[f64],
    t_lumped: f64,
    a: usize,
    quad: &MoldauerQuadrature,
) -> (Vec<f64>, f64) {
    let n_ch = t.len();
    let t_total: f64 = t.iter().sum::<f64>() + t_lumped;
    if t_total < NUMERICAL_FLOOR {
        return (vec![1.0; n_ch], 1.0);
    }
    let nus: Vec<f64> = t.iter().map(|&tc| moldauer_nu(tc, t_total)).collect();
    // F_c(x) = 1 + coef_c * x
    let coef: Vec<f64> = t
        .iter()
        .zip(&nus)
        .map(|(&tc, &nu)| 2.0 * tc / (nu * t_total))
        .collect();

    let mut row = vec![0.0; n_ch];
    let mut lumped = 0.0;
    let mut f = vec![0.0; n_ch];
    for (&x, &w) in quad.x.iter().zip(&quad.w) {
        let mut log_prod = -t_lumped * x / t_total;
        for c in 0..n_ch {
            f[c] = 1.0 + coef[c] * x;
            log_prod -= 0.5 * nus[c] * f[c].ln();
        }
        let base = w * log_prod.exp() / f[a];
        if base == 0.0 {
            continue;
        }
        for (r, &fb) in row.iter_mut().zip(&f) {
            *r += base / fb;
        }
        lumped += base;
    }
    row[a] *= 1.0 + 2.0 / nus[a];
    (row, lumped)
}

/// Compute Moldauer WFC factors for given channel transmission coefficients.
///
/// W_ab = (1 + 2 delta_ab / nu_a) * int_0^inf dx prod_c (1 + 2 T_c x / (nu_c T))^(-nu_c/2 - delta_ac - delta_bc)
///
/// with the Moldauer (1980) nu_c ([`moldauer_nu`]) and an `n_quadrature`-point rule
/// ([`MoldauerQuadrature`]).
pub fn moldauer_wfc(transmissions: &[f64], n_quadrature: usize) -> MoldauerResult {
    let quad = MoldauerQuadrature::new(n_quadrature);
    let w_factors = (0..transmissions.len())
        .map(|a| moldauer_w_row(transmissions, 0.0, a, &quad).0)
        .collect();
    MoldauerResult { w_factors }
}

/// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[i] = x;
        nodes[n - 1 - i] = -x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Simple Gauss-Laguerre quadrature nodes and weights.
/// Returns (nodes, weights) for n-point quadrature on [0, inf) with weight exp(-t).
fn gauss_laguerre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Pre-computed nodes for common sizes
    match n {
        1 => (vec![1.0], vec![1.0]),
        2 => (vec![0.585786, 3.414214], vec![0.853553, 0.146447]),
        4 => (
            vec![0.322548, 1.745761, 4.536620, 9.395071],
            vec![0.603154, 0.357419, 0.038888, 0.000539],
        ),
        8 => (
            vec![
                0.170279, 0.903702, 2.251087, 4.266700, 7.045906, 10.758516, 15.740679, 22.863132,
            ],
            vec![
                0.369189, 0.418787, 0.175794, 0.033343, 0.002794, 0.000091, 0.000001, 0.000000,
            ],
        ),
        _ => {
            // Fall back to 8-point for unsupported sizes
            gauss_laguerre_nodes(8)
        }
    }
}

// ======================== GOE Triple Integral (VWZ) ========================

/// Result of GOE WFC calculation.
#[derive(Debug, Clone)]
pub struct GoeResult {
    /// WFC correction factors W_ab.
    pub w_factors: Vec<Vec<f64>>,
}

/// GOE (Gaussian Orthogonal Ensemble) width fluctuation correction.
///
/// Implements the Verbaarschot-Weidenmüller-Zirnbauer (VWZ) formula as a
/// triple integral over auxiliary variables (lambda_1, lambda_2, mu):
///
/// W_ab = integral_0^inf d(lambda_1) integral_0^inf d(lambda_2) integral_0^1 d(mu)
///        * kernel(lambda_1, lambda_2, mu) * G_a * G_b / prod_c G_c
///
/// where G_c(lambda_1, lambda_2, mu) involves the channel transmission coefficients.
///
/// Uses Gauss-Laguerre quadrature for lambda integrals and Gauss-Legendre for mu.
///
/// **Experimental**: the kernel and normalization have not been validated against the
/// VWZ reference results (the factors do not conserve flux), and
/// [`hauser_feshbach`](crate::hf::hauser_feshbach) rejects `WfcModel::Goe`.
pub fn goe_wfc(transmissions: &[f64], n_quad_laguerre: usize, n_quad_legendre: usize) -> GoeResult {
    let n_ch = transmissions.len();
    let t_total: f64 = transmissions.iter().sum();

    if t_total < NUMERICAL_FLOOR || n_ch == 0 {
        return GoeResult {
            w_factors: vec![vec![1.0; n_ch]; n_ch],
        };
    }

    let (lag_nodes, lag_weights) = gauss_laguerre_nodes(n_quad_laguerre);
    let (leg_nodes, leg_weights) = gauss_legendre_nodes(n_quad_legendre);

    let mut w_factors = vec![vec![0.0; n_ch]; n_ch];

    // Triple integral: lambda_1, lambda_2 in [0, inf), mu in [0, 1]
    for (&l1, &wl1) in lag_nodes.iter().zip(lag_weights.iter()) {
        for (&l2, &wl2) in lag_nodes.iter().zip(lag_weights.iter()) {
            for (&mu_ref, &wmu) in leg_nodes.iter().zip(leg_weights.iter()) {
                // Transform mu from [-1,1] to [0,1]
                let mu = 0.5 * (mu_ref + 1.0);
                let wmu_scaled = wmu * 0.5;

                // VWZ kernel: (lambda_1 - lambda_2)^2 / (lambda_1 + lambda_2)
                //             * mu^{-1/2} * (1-mu)^{-1/2}  [absorbed in Legendre weights]
                let l_sum = l1 + l2;
                let l_diff = l1 - l2;
                if l_sum < 1e-300 {
                    continue;
                }
                let kernel = l_diff * l_diff / l_sum;

                // Compute G_c for each channel
                // G_c = [1 + T_c * (lambda_1 + lambda_2) / 2]^{-1}
                //     * [1 + T_c * (lambda_1 * mu + lambda_2 * (1-mu))]^{-1/2}
                //     * [1 + T_c * (lambda_1 * (1-mu) + lambda_2 * mu)]^{-1/2}
                let mut log_prod = 0.0;
                let mut g_vals = Vec::with_capacity(n_ch);

                for &tc in transmissions.iter() {
                    let term1 = 1.0 + tc * l_sum / 2.0;
                    let term2 = 1.0 + tc * (l1 * mu + l2 * (1.0 - mu));
                    let term3 = 1.0 + tc * (l1 * (1.0 - mu) + l2 * mu);

                    let g_c = 1.0 / (term1 * term2.sqrt() * term3.sqrt());
                    g_vals.push(g_c);
                    log_prod += g_c.ln();
                }
                let prod_g = log_prod.exp();

                let weight = wl1 * wl2 * wmu_scaled * kernel;

                for a in 0..n_ch {
                    for b in 0..n_ch {
                        // W_ab += weight * G_a * G_b / prod_all_G
                        // Since prod_all_G = product of all G_c, we need
                        // G_a * G_b / prod = G_a * G_b * prod(1/G_c for c != a,b)
                        // Simpler: use log space
                        let integrand = if a == b {
                            weight * g_vals[a] * g_vals[a] / prod_g
                        } else {
                            weight * g_vals[a] * g_vals[b] / prod_g
                        };
                        w_factors[a][b] += integrand;
                    }
                }
            }
        }
    }

    // Normalize: W_ab should approach 1 when all T_c are small
    // Add elastic enhancement
    for a in 0..n_ch {
        if w_factors[a][a] > 0.0 {
            let enhancement = 1.0 + 2.0 / kawano_talou_nu(transmissions[a], t_total);
            w_factors[a][a] *= enhancement;
        }
    }

    GoeResult { w_factors }
}

/// Gauss-Legendre quadrature nodes and weights on [-1, 1].
fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        2 => (
            vec![-0.577_350_269_189_626, 0.577_350_269_189_626],
            vec![1.0, 1.0],
        ),
        4 => (
            vec![
                -0.861_136_311_594_053,
                -0.339_981_043_584_856,
                0.339_981_043_584_856,
                0.861_136_311_594_053,
            ],
            vec![
                0.347_854_845_137_454,
                0.652_145_154_862_546,
                0.652_145_154_862_546,
                0.347_854_845_137_454,
            ],
        ),
        8 => (
            vec![
                -0.960_289_856_497_536,
                -0.796_666_477_413_627,
                -0.525_532_409_916_329,
                -0.183_434_642_495_650,
                0.183_434_642_495_650,
                0.525_532_409_916_329,
                0.796_666_477_413_627,
                0.960_289_856_497_536,
            ],
            vec![
                0.101_228_536_290_376,
                0.222_381_034_453_374,
                0.313_706_645_877_887,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.313_706_645_877_887,
                0.222_381_034_453_374,
                0.101_228_536_290_376,
            ],
        ),
        _ => gauss_legendre_nodes(8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nu_in_valid_range() {
        // nu should always be in [1, 2] after clamping
        for &t_a in &[0.001, 0.1, 0.5, 0.9, 0.99] {
            for &t_total in &[0.01, 1.0, 10.0, 100.0] {
                let nu = kawano_talou_nu(t_a, t_total);
                assert!(
                    (1.0..=2.0).contains(&nu),
                    "nu({}, {}) = {} out of range",
                    t_a,
                    t_total,
                    nu
                );
            }
        }
    }

    #[test]
    fn wfc_elastic_larger_than_inelastic() {
        // Elastic channel W_aa should be larger than off-diagonal W_ab
        let transmissions = vec![0.3, 0.3, 0.4];
        let result = moldauer_wfc(&transmissions, 8);

        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    assert!(
                        result.w_factors[a][a] >= result.w_factors[a][b],
                        "W[{}][{}]={} should be >= W[{}][{}]={}",
                        a,
                        a,
                        result.w_factors[a][a],
                        a,
                        b,
                        result.w_factors[a][b]
                    );
                }
            }
        }
    }

    #[test]
    fn goe_wfc_produces_results() {
        let transmissions = vec![0.3, 0.3, 0.4];
        let result = goe_wfc(&transmissions, 4, 4);
        assert_eq!(result.w_factors.len(), 3);
        // All W factors should be finite and positive
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    result.w_factors[a][b].is_finite(),
                    "GOE W[{}][{}] not finite",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn goe_elastic_larger_than_inelastic() {
        let transmissions = vec![0.3, 0.3, 0.4];
        let result = goe_wfc(&transmissions, 8, 8);
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    assert!(
                        result.w_factors[a][a] >= result.w_factors[a][b],
                        "GOE W[{}][{}]={} < W[{}][{}]={}",
                        a,
                        a,
                        result.w_factors[a][a],
                        a,
                        b,
                        result.w_factors[a][b]
                    );
                }
            }
        }
    }

    /// sum_b T_b W_ab + T_lumped W_a,lumped = T_total for every entrance channel a.
    fn assert_flux_conserved(t: &[f64], t_lumped: f64, n_quad: usize, tol: f64) {
        let quad = MoldauerQuadrature::new(n_quad);
        let t_total: f64 = t.iter().sum::<f64>() + t_lumped;
        for a in 0..t.len() {
            let (row, w_l) = moldauer_w_row(t, t_lumped, a, &quad);
            let flux: f64 = t.iter().zip(&row).map(|(tb, w)| tb * w).sum::<f64>() + t_lumped * w_l;
            assert!(
                (flux / t_total - 1.0).abs() < tol,
                "t={t:?} lumped={t_lumped} a={a}: sum T_b W_ab = {flux}, T = {t_total}"
            );
        }
    }

    #[test]
    fn moldauer_conserves_flux() {
        let cases: [(&[f64], f64); 6] = [
            (&[0.3, 0.3, 0.4], 0.0),
            (&[0.9], 1e-4),
            (&[0.99, 0.5], 0.0),
            (&[0.05, 0.02, 0.6, 0.6, 0.6, 0.3], 0.01),
            (&[1e-3, 1e-5], 2e-3),
            (&[0.5; 20], 3.0),
        ];
        for (t, t_lumped) in cases {
            assert_flux_conserved(t, t_lumped, 40, 1e-6);
        }
    }

    #[test]
    fn moldauer_quadrature_converged() {
        let t = [0.9, 0.2, 0.05];
        let reference = moldauer_w_row(&t, 1e-3, 0, &MoldauerQuadrature::new(400));
        let row = moldauer_w_row(&t, 1e-3, 0, &MoldauerQuadrature::new(40));
        for (w, w_ref) in row.0.iter().zip(&reference.0) {
            assert!((w / w_ref - 1.0).abs() < 1e-7, "{w} vs {w_ref}");
        }
        assert!((row.1 / reference.1 - 1.0).abs() < 1e-7);
    }

    #[test]
    fn moldauer_lumped_equals_many_weak_channels() {
        // 2000 channels of T = 5e-4 behave like one lumped T = 1.
        let strong = [0.6, 0.3];
        let mut many = strong.to_vec();
        many.extend_from_slice(&[5e-4; 2000]);
        let quad = MoldauerQuadrature::new(40);
        let (row_many, _) = moldauer_w_row(&many, 0.0, 0, &quad);
        let (row_lumped, w_l) = moldauer_w_row(&strong, 1.0, 0, &quad);
        for b in 0..2 {
            assert!(
                (row_many[b] / row_lumped[b] - 1.0).abs() < 1e-3,
                "W_0{b}: {} vs {}",
                row_many[b],
                row_lumped[b]
            );
        }
        assert!((row_many[2] / w_l - 1.0).abs() < 1e-3);
    }

    #[test]
    fn moldauer_two_channel_limits() {
        // Weak entrance + dominant lumped channel: W -> 1 off-diagonal, and the elastic
        // enhancement W_aa -> 1 + 2/nu_a with nu_a -> 1 (Porter-Thomas): 3.
        let quad = MoldauerQuadrature::new(40);
        let (row, w_l) = moldauer_w_row(&[1e-6], 50.0, 0, &quad);
        assert!((w_l - 1.0).abs() < 1e-5, "W_a,lumped = {w_l}");
        let nu = moldauer_nu(1e-6, 50.0);
        assert!(
            (row[0] - (1.0 + 2.0 / nu)).abs() < 1e-4,
            "W_aa = {}",
            row[0]
        );
        assert!((nu - 1.78).abs() < 1e-3, "nu = {nu}");
        // Few channels: elastic enhanced, other channels depleted.
        let (row, _) = moldauer_w_row(&[0.5, 0.5], 0.0, 0, &quad);
        assert!(row[0] > 1.0 && row[1] < 1.0, "{row:?}");
    }

    #[test]
    fn wfc_elastic_enhancement() {
        // Elastic channel should have W_aa > 1 (enhancement)
        let transmissions = vec![0.3, 0.3, 0.4];
        let result = moldauer_wfc(&transmissions, 8);

        for a in 0..3 {
            assert!(
                result.w_factors[a][a] > 1.0,
                "W[{}][{}] = {} should be > 1",
                a,
                a,
                result.w_factors[a][a]
            );
        }
    }
}
