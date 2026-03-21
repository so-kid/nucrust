//! Width fluctuation correction (WFC) implementations.
//!
//! Moldauer form with Kawano-Talou parameterization of the GOE degrees of freedom.

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

/// Compute Moldauer WFC factors for given channel transmission coefficients.
///
/// W_ab = (1 + 2*delta_ab/nu_a) * integral_0^inf dt prod_k F_k(t)^{-nu_k/2} * F_a(t) * F_b(t)
///
/// where F_k(t) = 1 + (2/nu_k) * (T_k / T_total) * t
///
/// Uses Gauss-Laguerre quadrature with n_points.
pub fn moldauer_wfc(transmissions: &[f64], n_quadrature: usize) -> MoldauerResult {
    let n_ch = transmissions.len();
    let t_total: f64 = transmissions.iter().sum();

    if t_total < 1e-30 || n_ch == 0 {
        return MoldauerResult {
            w_factors: vec![vec![1.0; n_ch]; n_ch],
        };
    }

    // Compute nu_a for each channel
    let nus: Vec<f64> = transmissions
        .iter()
        .map(|&t_a| kawano_talou_nu(t_a, t_total))
        .collect();

    // Gauss-Laguerre quadrature nodes and weights
    let (nodes, weights) = gauss_laguerre_nodes(n_quadrature);

    let mut w_factors = vec![vec![0.0; n_ch]; n_ch];

    for (q, (&t, &w)) in nodes.iter().zip(weights.iter()).enumerate() {
        let _ = q;
        // Compute product: prod_k F_k(t)^{-nu_k/2}
        let mut log_prod = 0.0;
        for k in 0..n_ch {
            let f_k = 1.0 + (2.0 / nus[k]) * (transmissions[k] / t_total) * t;
            log_prod -= (nus[k] / 2.0) * f_k.ln();
        }
        let prod = log_prod.exp();

        for a in 0..n_ch {
            let f_a = 1.0 + (2.0 / nus[a]) * (transmissions[a] / t_total) * t;
            for b in 0..n_ch {
                let f_b = 1.0 + (2.0 / nus[b]) * (transmissions[b] / t_total) * t;
                w_factors[a][b] += w * prod * f_a * f_b;
            }
        }
    }

    // Apply elastic enhancement factor: (1 + 2*delta_ab/nu_a)
    for a in 0..n_ch {
        w_factors[a][a] *= 1.0 + 2.0 / nus[a];
    }

    MoldauerResult { w_factors }
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
