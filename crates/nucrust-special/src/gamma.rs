use num_complex::Complex64;
use std::f64::consts::PI;

/// Spouge approximation parameter. Controls precision: error < a^{-1/2} * (2π)^{-(a+1/2)}.
/// For a=15: error < ~2e-13, sufficient for double precision.
const SPOUGE_A: usize = 15;

/// Precomputed Spouge coefficients c_k for k=0..a-1.
///
/// c_0 = sqrt(2*pi)
/// c_k = (-1)^{k-1} / (k-1)! * (a-k)^{k-0.5} * exp(a-k), k >= 1
fn spouge_coefficients() -> [f64; SPOUGE_A] {
    let a = SPOUGE_A as f64;
    let mut c = [0.0; SPOUGE_A];
    c[0] = (2.0 * PI).sqrt();
    let mut factorial = 1.0; // (k-1)!
    for (k, ck) in c.iter_mut().enumerate().skip(1) {
        let kf = k as f64;
        if k > 1 {
            factorial *= (k - 1) as f64;
        }
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        *ck = sign / factorial * (a - kf).powf(kf - 0.5) * (a - kf).exp();
    }
    c
}

/// Complex log-gamma function ln Gamma(z) using the Spouge approximation.
///
/// Accurate to ~13 significant digits for Re(z) > 0.
/// Uses the reflection formula for Re(z) <= 0.5.
pub fn complex_log_gamma(z: Complex64) -> Complex64 {
    // Reflection formula for Re(z) < 0.5
    if z.re < 0.5 {
        return Complex64::new(PI, 0.0).ln()
            - (Complex64::new(PI, 0.0) * z).sin().ln()
            - complex_log_gamma(Complex64::new(1.0, 0.0) - z);
    }

    let c = spouge_coefficients();
    let a = Complex64::new(SPOUGE_A as f64, 0.0);

    // Spouge: Gamma(z) = (z-1+a)^{z-0.5} * exp(-(z-1+a)) * S(z)
    // where S(z) = c_0 + Σ_{k=1}^{a-1} c_k / (z-1+k)
    let z_m1 = z - 1.0;

    let mut s = Complex64::new(c[0], 0.0);
    for (k, &ck) in c.iter().enumerate().skip(1) {
        s += ck / (z_m1 + Complex64::new(k as f64, 0.0));
    }

    let t = z_m1 + a; // z - 1 + a

    // ln Gamma(z) = (z-0.5)*ln(t) - t + ln(S)
    (z - 0.5) * t.ln() - t + s.ln()
}

/// Coulomb phase shift sigma_l = Im[ln Gamma(l+1+i*eta)].
pub fn coulomb_phase_shift(l: u32, eta: f64) -> f64 {
    let z = Complex64::new((l + 1) as f64, eta);
    complex_log_gamma(z).im
}

/// Gamow factor C_l(eta).
///
/// C_0(eta) = sqrt(2*pi*eta / (exp(2*pi*eta) - 1)) is evaluated in closed form and
/// C_l = C_{l-1} * sqrt(l^2 + eta^2) / (l*(2l+1)) by upward recurrence (DLMF 33.2.5-6).
/// Unlike the log-gamma route, this does not lose digits to the cancellation between
/// -pi*eta/2 and ln|Gamma(1+l+i*eta)| at large eta.
pub fn gamow_factor(l: u32, eta: f64) -> f64 {
    let two_pi_eta = 2.0 * PI * eta;
    let mut c = if eta > 0.0 {
        // exp(-pi*eta) * sqrt(2*pi*eta / (1 - exp(-2*pi*eta))): no overflow for large eta.
        (two_pi_eta / -(-two_pi_eta).exp_m1()).sqrt() * (-PI * eta).exp()
    } else if eta < 0.0 {
        (two_pi_eta / two_pi_eta.exp_m1()).sqrt()
    } else {
        1.0
    };
    for k in 1..=l {
        let kf = k as f64;
        c *= (kf * kf + eta * eta).sqrt() / (kf * (2.0 * kf + 1.0));
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_gamma_real_positive_integers() {
        let result = complex_log_gamma(Complex64::new(1.0, 0.0));
        assert!(result.re.abs() < 1e-13, "ln Gamma(1) = {}", result.re);

        let result = complex_log_gamma(Complex64::new(5.0, 0.0));
        let expected = 24.0_f64.ln();
        assert!(
            (result.re - expected).abs() < 1e-12,
            "ln Gamma(5): got {}, expected {}",
            result.re,
            expected
        );

        let result = complex_log_gamma(Complex64::new(10.0, 0.0));
        let expected = 362880.0_f64.ln();
        assert!(
            (result.re - expected).abs() < 1e-10,
            "ln Gamma(10): got {}, expected {}",
            result.re,
            expected
        );
    }

    #[test]
    fn log_gamma_half_integer() {
        let result = complex_log_gamma(Complex64::new(0.5, 0.0));
        let expected = 0.5 * PI.ln();
        assert!(
            (result.re - expected).abs() < 1e-12,
            "ln Gamma(1/2): got {}, expected {}",
            result.re,
            expected
        );
    }

    #[test]
    fn log_gamma_complex() {
        // Verified: |Gamma(1+i)|^2 = pi/sinh(pi) = 0.27194, so Re = ln(0.52148) = -0.6509
        let result = complex_log_gamma(Complex64::new(1.0, 1.0));
        assert!(
            (result.re - (-0.6509231993018951)).abs() < 1e-10,
            "Re ln Gamma(1+i): got {}",
            result.re
        );
        // Im verified by independent Stirling and Spouge implementations
        assert!(
            (result.im - (-0.30164032046753)).abs() < 1e-10,
            "Im ln Gamma(1+i): got {}",
            result.im
        );
    }

    #[test]
    fn log_gamma_large_imaginary() {
        // Verified analytically: |Gamma(1+10i)|^2 = 10*pi/sinh(10*pi)
        // ln|Gamma(1+10i)| = 0.5*ln(10*pi/sinh(10*pi)) ≈ -13.638
        let result = complex_log_gamma(Complex64::new(1.0, 10.0));
        let expected_re = 0.5 * (10.0 * PI / (10.0 * PI).sinh()).ln();
        assert!(
            (result.re - expected_re).abs() < 1e-8,
            "Re ln Gamma(1+10i): got {}, expected {}",
            result.re,
            expected_re
        );
    }

    #[test]
    fn coulomb_phase_shift_eta_zero() {
        for l in 0..5 {
            let sigma = coulomb_phase_shift(l, 0.0);
            assert!(sigma.abs() < 1e-12, "sigma_{}(eta=0) = {}", l, sigma);
        }
    }

    #[test]
    fn gamow_factor_c0_eta_zero() {
        let c0 = gamow_factor(0, 1e-10);
        assert!((c0 - 1.0).abs() < 1e-6, "C_0(~0) = {}", c0);
    }

    #[test]
    fn gamow_factor_c0_eta_1() {
        let expected = (2.0 * PI / ((2.0 * PI).exp() - 1.0)).sqrt();
        let c0 = gamow_factor(0, 1.0);
        let rel_err = ((c0 - expected) / expected).abs();
        assert!(
            rel_err < 1e-10,
            "C_0(1): got {}, expected {}, rel_err = {:.2e}",
            c0,
            expected,
            rel_err
        );
    }

    #[test]
    fn gamow_factor_recurrence() {
        let eta = 2.0;
        let c0 = gamow_factor(0, eta);
        let c1 = gamow_factor(1, eta);
        let c1_from_recurrence = c0 * (1.0 + eta * eta).sqrt() / (1.0 * 3.0);
        let rel_err = ((c1 - c1_from_recurrence) / c1).abs();
        assert!(
            rel_err < 1e-10,
            "C_1 recurrence: got {}, expected {}, rel_err = {:.2e}",
            c1,
            c1_from_recurrence,
            rel_err
        );
    }

    #[test]
    fn gamow_factor_large_eta() {
        // C_0(eta) = sqrt(2 pi eta / (exp(2 pi eta) - 1)); for eta = 50 the
        // log-gamma route lost ~4 digits to cancellation.
        // mpmath (dps=40): C_0(50) = 1.0709205299981646e-67
        let c0 = gamow_factor(0, 50.0);
        let expected = 1.0709205299981646e-67;
        let rel_err = ((c0 - expected) / expected).abs();
        assert!(
            rel_err < 1e-14,
            "C_0(50): got {c0}, rel_err = {rel_err:.2e}"
        );
    }
}
