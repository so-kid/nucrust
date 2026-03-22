use crate::CoreError;

/// Natural cubic spline interpolator.
///
/// Boundary condition: second derivative = 0 at both endpoints.
/// Required for MACS calculation where Gauss-Laguerre quadrature nodes
/// do not generally coincide with user-defined energy grid points.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Coefficients [a, b, c, d] for each interval:
    /// S_i(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Vec<[f64; 4]>,
}

impl CubicSpline {
    /// Construct a natural cubic spline from data points.
    ///
    /// Requires at least 2 points. x must be strictly increasing.
    pub fn natural(x: &[f64], y: &[f64]) -> Result<Self, CoreError> {
        let n = x.len();
        if n < 2 || n != y.len() {
            return Err(CoreError::InvalidParameter {
                name: "spline_data",
                value: n as f64,
                reason: "need at least 2 points with matching x and y lengths",
            });
        }

        // Check strict monotonicity
        for i in 1..n {
            if x[i] <= x[i - 1] {
                return Err(CoreError::InvalidParameter {
                    name: "x",
                    value: x[i],
                    reason: "x values must be strictly increasing",
                });
            }
        }

        let m = n - 1; // number of intervals
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

        // Solve tridiagonal system for second derivatives (c_i)
        // Natural boundary: c[0] = c[n-1] = 0
        let mut c = vec![0.0; n];

        if n > 2 {
            // Set up tridiagonal system for interior points
            let inner = n - 2;
            let mut diag = vec![0.0; inner];
            let mut upper = vec![0.0; inner];
            let mut lower = vec![0.0; inner];
            let mut rhs = vec![0.0; inner];

            for i in 0..inner {
                let ii = i + 1; // index in original arrays
                diag[i] = 2.0 * (h[ii - 1] + h[ii]);
                rhs[i] = 3.0 * ((y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]);
                if i > 0 {
                    lower[i] = h[ii - 1];
                }
                if i < inner - 1 {
                    upper[i] = h[ii];
                }
            }

            // Thomas algorithm (tridiagonal solver)
            for i in 1..inner {
                let w = lower[i] / diag[i - 1];
                diag[i] -= w * upper[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }

            c[inner] = rhs[inner - 1] / diag[inner - 1];
            for i in (0..inner - 1).rev() {
                c[i + 1] = (rhs[i] - upper[i] * c[i + 2]) / diag[i];
            }
        }

        // Build coefficients
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let a = y[i];
            let b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            let d = (c[i + 1] - c[i]) / (3.0 * h[i]);
            coeffs.push([a, b, c[i], d]);
        }

        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            coeffs,
        })
    }

    /// Evaluate the spline at a single point.
    ///
    /// Extrapolation uses the nearest boundary interval.
    pub fn evaluate(&self, xv: f64) -> f64 {
        let i = self.find_interval(xv);
        let dx = xv - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    /// Evaluate the spline at multiple points.
    ///
    /// For best performance, input should be sorted (uses sequential search).
    pub fn evaluate_batch(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&xv| self.evaluate(xv)).collect()
    }

    /// Return the original data points.
    pub fn data_points(&self) -> (&[f64], &[f64]) {
        (&self.x, &self.y)
    }

    /// Find the interval index for a given x value.
    fn find_interval(&self, xv: f64) -> usize {
        let n = self.x.len();
        if xv <= self.x[0] {
            return 0;
        }
        if xv >= self.x[n - 1] {
            return n - 2; // last interval
        }
        // Binary search
        match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&xv).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i.min(n - 2),
            Err(i) => i - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_linear_data_exactly() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let s = CubicSpline::natural(&x, &y).unwrap();

        for &xv in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            assert!((s.evaluate(xv) - xv).abs() < 1e-12, "at x={xv}");
        }
    }

    #[test]
    fn passes_through_data_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let s = CubicSpline::natural(&x, &y).unwrap();

        for i in 0..x.len() {
            assert!(
                (s.evaluate(x[i]) - y[i]).abs() < 1e-12,
                "at data point {}",
                i
            );
        }
    }

    #[test]
    fn batch_matches_single() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 2.0, 5.0];
        let s = CubicSpline::natural(&x, &y).unwrap();

        let xs = vec![0.5, 1.5, 2.5];
        let batch = s.evaluate_batch(&xs);
        for (i, &xv) in xs.iter().enumerate() {
            assert!((batch[i] - s.evaluate(xv)).abs() < 1e-15);
        }
    }

    #[test]
    fn rejects_insufficient_data() {
        assert!(CubicSpline::natural(&[1.0], &[1.0]).is_err());
    }

    #[test]
    fn rejects_non_monotonic() {
        assert!(CubicSpline::natural(&[1.0, 0.5, 2.0], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn two_points() {
        let s = CubicSpline::natural(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert!((s.evaluate(0.5) - 0.5).abs() < 1e-12);
    }
}
