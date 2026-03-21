//! Johnson log-derivative method for coupled-channel optical model.
//!
//! Propagates the N×N log-derivative matrix Z(R) = Y'(R) Y(R)⁻¹
//! using the piecewise-constant potential approximation:
//!
//!   Z_{n+1} = Q_n - Q_n² (Q_n + Z_n)⁻¹
//!
//! where Q_n = W_n^{1/2} cot(h · W_n^{1/2}).
//!
//! Reference: Johnson, J. Comput. Phys. 13 (1973) 445.
//! See also: Thompson & Nunes, "Nuclear Reactions for Astrophysics" (2009), §3.4.

use faer::complex_native::c64;
use faer::prelude::*;
use nucrust_core::CoreError;
use num_complex::Complex64;

/// Configuration for the Johnson log-derivative propagation.
#[derive(Debug, Clone)]
pub struct JohnsonConfig {
    /// Step size in fm.
    pub step_size: f64,
    /// Starting radius in fm.
    pub r_min: f64,
    /// Matching radius in fm.
    pub r_max: f64,
}

impl Default for JohnsonConfig {
    fn default() -> Self {
        Self {
            step_size: 0.05,
            r_min: 0.1,
            r_max: 15.0,
        }
    }
}

/// Result of Johnson log-derivative propagation.
pub struct LogDerivResult {
    /// Log-derivative matrix Z(R_match) = Y'(R) Y(R)⁻¹ at matching radius.
    /// Stored as flat Vec in row-major order (N×N).
    pub z_matrix: Vec<Complex64>,
    /// Number of channels.
    pub n_channels: usize,
    /// Matching radius used.
    pub r_match: f64,
}

/// Propagate the log-derivative matrix from r_min to r_max.
///
/// `w_matrix_fn` returns the N×N W-matrix (coupling + centrifugal + potential)
/// at radius r, as a flat Vec<Complex64> in row-major order.
///
/// `l_values`: orbital angular momentum for each channel (for initial conditions).
/// If None, uses l=0 for all channels.
pub fn johnson_propagate<F>(
    n_channels: usize,
    config: &JohnsonConfig,
    w_matrix_fn: F,
) -> Result<LogDerivResult, CoreError>
where
    F: Fn(f64) -> Vec<Complex64>,
{
    johnson_propagate_with_l(n_channels, config, w_matrix_fn, None)
}

/// Propagate with explicit per-channel orbital angular momenta.
pub fn johnson_propagate_with_l<F>(
    n_channels: usize,
    config: &JohnsonConfig,
    w_matrix_fn: F,
    l_values: Option<&[u32]>,
) -> Result<LogDerivResult, CoreError>
where
    F: Fn(f64) -> Vec<Complex64>,
{
    let h = config.step_size;
    let n = n_channels;
    let n_steps = ((config.r_max - config.r_min) / h).ceil() as usize;

    if n_steps < 2 {
        return Err(CoreError::InvalidParameter {
            name: "r_max - r_min",
            value: config.r_max - config.r_min,
            reason: "too few steps for Johnson propagation",
        });
    }

    // Initialize Z at r_min from the boundary condition u_c(r) ~ r^{l_c+1}.
    // The log-derivative of r^{l+1} is (l+1)/r, so Z_0 = diag((l_c+1)/r_min).
    let mut z = faer::Mat::<c64>::zeros(n, n);
    for i in 0..n {
        let l_c = l_values.map_or(0, |lv| lv[i]);
        z[(i, i)] = c64::new((l_c as f64 + 1.0) / config.r_min, 0.0);
    }

    // Propagate outward
    for step in 0..n_steps {
        let r = config.r_min + (step as f64 + 0.5) * h; // Midpoint
        let w_flat = w_matrix_fn(r);

        // Convert W to faer matrix
        let w = flat_to_faer(&w_flat, n);

        // Compute Q_n = W^{1/2} cot(h W^{1/2})
        let q = compute_q_matrix(&w, h, n)?;

        // Z_{n+1} = Q - Q² (Q + Z)⁻¹
        //         = Q - Q (Q + Z)⁻¹ Q
        // Which is equivalent to the Johnson step
        let qpz = &q + &z; // Q + Z

        // Solve (Q + Z) X = Q  =>  X = (Q + Z)⁻¹ Q
        let lu = qpz.partial_piv_lu();
        let x = lu.solve(&q);

        // Z_{n+1} = Q - Q X = Q (I - X)
        let identity = faer::Mat::<c64>::identity(n, n);
        let imx = &identity - &x;
        z = &q * &imx;
    }

    // Convert back to Vec<Complex64>
    let z_flat = faer_to_flat(&z, n);

    Ok(LogDerivResult {
        z_matrix: z_flat,
        n_channels: n,
        r_match: config.r_max,
    })
}

/// Compute Q = W^{1/2} cot(h W^{1/2}) via eigendecomposition.
///
/// For W = U Λ U⁻¹:
///   Q = U diag(√λ_i cot(h √λ_i)) U⁻¹
fn compute_q_matrix(w: &faer::Mat<c64>, h: f64, n: usize) -> Result<faer::Mat<c64>, CoreError> {
    // Eigendecomposition of W
    let (eigenvalues, eigvecs) = complex_eigendecomposition(w, n)?;

    // Apply the scalar function f(λ) = √λ cot(h√λ) to each eigenvalue
    let mut diag_vals = vec![c64::new(0.0, 0.0); n];
    for i in 0..n {
        let lambda = eigenvalues[i];
        diag_vals[i] = scalar_q_function(lambda, h);
    }

    // Reconstruct: Q = V diag(f(λ_i)) V⁻¹
    reconstruct_from_eigen(&eigvecs, &diag_vals, n)
}

/// Scalar function: Q(λ) = √λ cot(h√λ).
///
/// When λ has a small imaginary part relative to its real part (typical for optical
/// model potentials where Im(V)/Re(V) ~ 0.5%), the direct formula suffers from
/// catastrophic cancellation. In that regime we use first-order perturbation theory:
///
///   Q(λ_R + iλ_I) ≈ Q(λ_R) + i·λ_I · dQ/dλ|_{λ=λ_R}
///
/// For real negative λ_R (interior of nuclear well):
///   Q(λ_R) = √|λ_R| · coth(h·√|λ_R|)             (real)
///   dQ/dλ  = [-coth(α) + α/sinh²(α)] / (2√|λ_R|)  where α = h√|λ_R|
fn scalar_q_function(lambda: c64, h: f64) -> c64 {
    let lam_re = lambda.re;
    let lam_im = lambda.im;

    // Check if perturbation approach is needed: |Im(λ)| << |Re(λ)|
    let use_perturbation = lam_re.abs() > 1e-10 && (lam_im / lam_re).abs() < 0.1;

    if use_perturbation && lam_re < 0.0 {
        // λ ≈ negative real: √λ = i√|λ|, cot(ih√|λ|) = -i coth(h√|λ|)
        let abs_lam = lam_re.abs();
        let sqrt_abs = abs_lam.sqrt();
        let alpha = h * sqrt_abs;

        // Q_real = √|λ| · coth(α)
        let coth_a = if alpha > 500.0 {
            1.0
        } else if alpha < 1e-10 {
            return c64::new(1.0 / h, lam_im * h / 3.0);
        } else {
            alpha.cosh() / alpha.sinh()
        };
        let q_real = sqrt_abs * coth_a;

        // dQ/dλ at real negative λ:
        // = [-coth(α) + α/sinh²(α)] / (2√|λ|)
        let sinh_a = alpha.sinh();
        let dq_dlam = (-coth_a + alpha / (sinh_a * sinh_a)) / (2.0 * sqrt_abs);

        // Q ≈ Q_real + i · λ_I · dQ/dλ
        return c64::new(q_real, lam_im * dq_dlam);
    }

    if use_perturbation && lam_re > 0.0 {
        // λ ≈ positive real: √λ = √λ_R, cot(h√λ_R) is real
        let sqrt_lam = lam_re.sqrt();
        let alpha = h * sqrt_lam;
        let (sin_a, cos_a) = alpha.sin_cos();

        if sin_a.abs() < 1e-30 {
            return c64::new(1.0 / h, 0.0);
        }

        let cot_a = cos_a / sin_a;
        let q_real = sqrt_lam * cot_a;

        // dQ/dλ = [cot(α) - α/sin²(α)] / (2√λ_R)
        let dq_dlam = (cot_a - alpha / (sin_a * sin_a)) / (2.0 * sqrt_lam);

        return c64::new(q_real, lam_im * dq_dlam);
    }

    // General complex case: direct computation
    let sqrt_lam = complex_sqrt(lambda);
    let arg = c64::new(h, 0.0) * sqrt_lam;

    let sin_a = complex_sin(arg);
    let cos_a = complex_cos(arg);

    let sin_norm = sin_a.re * sin_a.re + sin_a.im * sin_a.im;
    if sin_norm < 1e-30 {
        return c64::new(1.0 / h, 0.0);
    }

    sqrt_lam * cos_a / sin_a
}

/// Complex eigendecomposition using faer.
///
/// Returns (eigenvalues, eigenvector_matrix).
fn complex_eigendecomposition(
    m: &faer::Mat<c64>,
    n: usize,
) -> Result<(Vec<c64>, faer::Mat<c64>), CoreError> {
    if n == 1 {
        // Trivial case: single channel
        let lambda = m[(0, 0)];
        let mut v = faer::Mat::<c64>::zeros(1, 1);
        v[(0, 0)] = c64::new(1.0, 0.0);
        return Ok((vec![lambda], v));
    }

    if n == 2 {
        // Analytical 2×2 eigendecomposition for efficiency
        return eigen_2x2(m);
    }

    // General case: use faer's complex eigendecomposition
    let evd = m.complex_eigendecomposition();
    let s_diag = evd.s();
    let s_col = s_diag.column_vector();
    let u = evd.u();

    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(s_col[i]);
    }

    // Clone eigenvector matrix
    let eigvecs = u.to_owned();

    Ok((eigenvalues, eigvecs))
}

/// Analytical 2×2 eigendecomposition.
fn eigen_2x2(m: &faer::Mat<c64>) -> Result<(Vec<c64>, faer::Mat<c64>), CoreError> {
    let a = m[(0, 0)];
    let b = m[(0, 1)];
    let cc = m[(1, 0)];
    let d = m[(1, 1)];

    let trace = a + d;
    let det = a * d - b * cc;
    let disc = trace * trace - c64::new(4.0, 0.0) * det;
    let sqrt_disc = complex_sqrt(disc);

    let half = c64::new(0.5, 0.0);
    let lam1 = half * (trace + sqrt_disc);
    let lam2 = half * (trace - sqrt_disc);

    let mut v = faer::Mat::<c64>::zeros(2, 2);

    // Eigenvectors
    let b_norm = b.re * b.re + b.im * b.im;
    if b_norm > 1e-30 {
        v[(0, 0)] = b;
        v[(1, 0)] = lam1 - a;
        v[(0, 1)] = b;
        v[(1, 1)] = lam2 - a;
    } else {
        // b ≈ 0: use cc instead, or diagonal matrix
        let cc_norm = cc.re * cc.re + cc.im * cc.im;
        if cc_norm > 1e-30 {
            v[(0, 0)] = lam1 - d;
            v[(1, 0)] = cc;
            v[(0, 1)] = lam2 - d;
            v[(1, 1)] = cc;
        } else {
            // Diagonal matrix
            v[(0, 0)] = c64::new(1.0, 0.0);
            v[(1, 1)] = c64::new(1.0, 0.0);
        }
    }

    // Normalize columns
    for col in 0..2 {
        let norm = (v[(0, col)].re.powi(2)
            + v[(0, col)].im.powi(2)
            + v[(1, col)].re.powi(2)
            + v[(1, col)].im.powi(2))
        .sqrt();
        if norm > 1e-30 {
            let inv_norm = c64::new(1.0 / norm, 0.0);
            v[(0, col)] *= inv_norm;
            v[(1, col)] *= inv_norm;
        }
    }

    Ok((vec![lam1, lam2], v))
}

/// Reconstruct matrix from eigendecomposition: M = V diag(d) V⁻¹.
fn reconstruct_from_eigen(
    v: &faer::Mat<c64>,
    d: &[c64],
    n: usize,
) -> Result<faer::Mat<c64>, CoreError> {
    // Build diagonal matrix
    let mut diag = faer::Mat::<c64>::zeros(n, n);
    for i in 0..n {
        diag[(i, i)] = d[i];
    }

    // V * diag * V⁻¹
    let vd = v * &diag;
    let lu = v.partial_piv_lu();
    // Solve V * X = I to get V⁻¹
    let identity = faer::Mat::<c64>::identity(n, n);
    let v_inv = lu.solve(&identity);

    Ok(&vd * &v_inv)
}

// ============================================================================
// Complex arithmetic helpers using faer's c64
// ============================================================================

fn complex_sqrt(z: c64) -> c64 {
    let c = Complex64::new(z.re, z.im).sqrt();
    c64::new(c.re, c.im)
}

fn complex_sin(z: c64) -> c64 {
    // sin(a+ib) = sin(a)cosh(b) + i cos(a)sinh(b)
    let (sa, ca) = (z.re.sin(), z.re.cos());
    let (shb, chb) = (z.im.sinh(), z.im.cosh());
    c64::new(sa * chb, ca * shb)
}

fn complex_cos(z: c64) -> c64 {
    // cos(a+ib) = cos(a)cosh(b) - i sin(a)sinh(b)
    let (sa, ca) = (z.re.sin(), z.re.cos());
    let (shb, chb) = (z.im.sinh(), z.im.cosh());
    c64::new(ca * chb, -sa * shb)
}

// ============================================================================
// Conversion helpers between flat Vec<Complex64> and faer::Mat<c64>
// ============================================================================

fn flat_to_faer(flat: &[Complex64], n: usize) -> faer::Mat<c64> {
    let mut m = faer::Mat::<c64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = flat[i * n + j];
            m[(i, j)] = c64::new(v.re, v.im);
        }
    }
    m
}

fn faer_to_flat(m: &faer::Mat<c64>, n: usize) -> Vec<Complex64> {
    let mut flat = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let v = m[(i, j)];
            flat[i * n + j] = Complex64::new(v.re, v.im);
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_c64(a: c64, b: c64, tol: f64, msg: &str) {
        let diff = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
        assert!(
            diff < tol,
            "{msg}: expected ({}, {}i), got ({}, {}i), diff = {diff}",
            b.re,
            b.im,
            a.re,
            a.im
        );
    }

    #[test]
    fn scalar_q_small_lambda() {
        // For small h²λ: cot(x)/x ≈ 1/x - x/3 - ...
        // Q = √λ cot(h√λ) ≈ 1/h for h→0
        let lambda = c64::new(1.0, 0.0);
        let h = 0.001;
        let q = scalar_q_function(lambda, h);
        // √1 cot(0.001) ≈ 1/0.001 = 1000
        assert!(
            (q.re - 1000.0).abs() < 1.0,
            "Q ≈ 1/h for small h, got {}",
            q.re
        );
    }

    #[test]
    fn scalar_q_negative_lambda() {
        // For negative λ (evanescent): √(-|λ|) = i√|λ|
        // cot(ih√|λ|) = -i coth(h√|λ|) → gives real positive result
        let lambda = c64::new(-10.0, 0.0);
        let h = 0.1;
        let q = scalar_q_function(lambda, h);
        // Should be real (imaginary part ≈ 0)
        assert!(
            q.im.abs() < 1e-10,
            "Q for negative λ should be real, got im = {}",
            q.im
        );
    }

    #[test]
    fn eigen_2x2_diagonal() {
        let mut m = faer::Mat::<c64>::zeros(2, 2);
        m[(0, 0)] = c64::new(2.0, 0.0);
        m[(1, 1)] = c64::new(5.0, 0.0);

        let (evals, _evecs) = eigen_2x2(&m).unwrap();
        let mut evals_sorted: Vec<f64> = evals.iter().map(|e| e.re).collect();
        evals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((evals_sorted[0] - 2.0).abs() < 1e-12);
        assert!((evals_sorted[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn eigen_2x2_symmetric() {
        let mut m = faer::Mat::<c64>::zeros(2, 2);
        m[(0, 0)] = c64::new(3.0, 0.0);
        m[(0, 1)] = c64::new(1.0, 0.0);
        m[(1, 0)] = c64::new(1.0, 0.0);
        m[(1, 1)] = c64::new(3.0, 0.0);

        let (evals, _evecs) = eigen_2x2(&m).unwrap();
        let mut evals_sorted: Vec<f64> = evals.iter().map(|e| e.re).collect();
        evals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Eigenvalues of [[3,1],[1,3]] are 2 and 4
        assert!((evals_sorted[0] - 2.0).abs() < 1e-12);
        assert!((evals_sorted[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn reconstruct_identity() {
        let n = 3;
        let identity = faer::Mat::<c64>::identity(n, n);
        let d = vec![c64::new(1.0, 0.0); n];
        let result = reconstruct_from_eigen(&identity, &d, n).unwrap();

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close_c64(
                    result[(i, j)],
                    c64::new(expected, 0.0),
                    1e-12,
                    &format!("identity[{i},{j}]"),
                );
            }
        }
    }

    #[test]
    fn johnson_single_channel_free_particle() {
        // Single channel, free particle (W = -k²)
        // Log-derivative of spherical Bessel: j_0(kr)/j_0(kr) → -k tan(kr)/(kr) at surface
        let k = 1.0; // fm⁻¹
        let n = 1;
        let config = JohnsonConfig {
            step_size: 0.01,
            r_min: 0.01,
            r_max: 5.0,
        };

        let result = johnson_propagate(n, &config, |_r| vec![Complex64::new(-k * k, 0.0)]).unwrap();

        // Log-derivative should be finite and real for free particle
        let z = result.z_matrix[0];
        assert!(z.re.is_finite(), "Z should be finite, got {}", z.re);
        assert!(
            z.im.abs() < 0.1,
            "Z should be nearly real for free particle, im = {}",
            z.im
        );
    }

    #[test]
    fn johnson_two_channel_completes() {
        // Two uncoupled channels
        let n = 2;
        let config = JohnsonConfig {
            step_size: 0.05,
            r_min: 0.1,
            r_max: 10.0,
        };

        let result = johnson_propagate(n, &config, |_r| {
            let mut w = vec![Complex64::new(0.0, 0.0); 4];
            w[0] = Complex64::new(-1.0, 0.0); // Open channel
            w[3] = Complex64::new(10.0, 0.0); // Closed channel
            w
        })
        .unwrap();

        assert_eq!(result.n_channels, 2);
        // Diagonal elements should be finite
        assert!(result.z_matrix[0].re.is_finite());
        assert!(result.z_matrix[3].re.is_finite());
    }

    #[test]
    fn conversion_roundtrip() {
        let n = 3;
        let flat: Vec<Complex64> = (0..9)
            .map(|i| Complex64::new(i as f64, -(i as f64)))
            .collect();
        let m = flat_to_faer(&flat, n);
        let flat2 = faer_to_flat(&m, n);
        for i in 0..9 {
            assert!((flat[i].re - flat2[i].re).abs() < 1e-15);
            assert!((flat[i].im - flat2[i].im).abs() < 1e-15);
        }
    }
}
