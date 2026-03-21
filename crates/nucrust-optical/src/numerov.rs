//! Fox-Goodwin ratio-variable Numerov integration.
//!
//! Integrates the radial Schrödinger equation outward from r_min to r_match,
//! then extracts the S-matrix element by matching to Coulomb wave functions.

use nucrust_core::backend::NumerovConfig;
use nucrust_core::traits::OpticalPotential;
use nucrust_core::units::{self, HBAR_C};
use nucrust_core::{Channel, CoreError};
use nucrust_special::coulomb_wave;
use num_complex::Complex64;

/// Perform Numerov integration for a single (E, l, j) and extract S-matrix element.
///
/// Uses the Fox-Goodwin ratio-variable method:
///   R_n = u_{n+1} / u_n
///   R_n = [2(1 + 5h²/12·f_n) - (1 - h²/12·f_{n-1})/R_{n-1}] / (1 - h²/12·f_{n+1})
///
/// Returns the S-matrix element S_{lj}(E).
pub fn numerov_integrate(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energy: f64,
    l: u32,
    j: f64,
    config: &NumerovConfig,
) -> Result<Complex64, CoreError> {
    let h = config.step_size;
    let r_match = potential.matching_radius(channel);

    // Reduced mass and wave number
    let mu = reduced_mass_channel(channel);
    let k = units::wave_number(mu, energy);

    // Number of steps
    let n_steps = ((r_match - config.r_min) / h).ceil() as usize;
    if n_steps < 3 {
        return Err(CoreError::InvalidParameter {
            name: "r_match",
            value: r_match,
            reason: "matching radius too small for Numerov integration",
        });
    }

    // f(r) = 2μ/ℏ² [V(r) - E] + l(l+1)/r²
    // In our units: V in MeV, r in fm, ℏ²/(2μ) = ℏc² / (2 * μ * c²)
    let hbar2_over_2mu = HBAR_C * HBAR_C / (2.0 * mu * units::AMU_MEV);
    let l_f = l as f64;
    let centrifugal = l_f * (l_f + 1.0);

    let f_at = |r: f64| -> Complex64 {
        if r < 1e-10 {
            return Complex64::new(1e30, 0.0); // Large repulsive barrier at origin
        }
        let v = potential.potential(r, energy, l, j, channel);
        let v_eff = v - Complex64::new(energy, 0.0);
        v_eff / hbar2_over_2mu + Complex64::new(centrifugal / (r * r), 0.0)
    };

    // Fox-Goodwin ratio variable integration
    // Initial value: R_0 = 2^{l+1} (from power-series behavior near origin)
    let h2_12 = h * h / 12.0;
    let one = Complex64::new(1.0, 0.0);
    let two = Complex64::new(2.0, 0.0);
    let five = Complex64::new(5.0, 0.0);

    let r0 = config.r_min;
    let r1 = r0 + h;

    let f0 = f_at(r0);
    let f1 = f_at(r1);

    // Initial ratio from power series: u ~ r^{l+1}
    let mut ratio = Complex64::new((r1 / r0).powi(l as i32 + 1), 0.0);

    let mut f_prev = f0;
    let mut f_curr = f1;

    for step in 2..=n_steps {
        let r_next = config.r_min + step as f64 * h;
        let f_next = f_at(r_next);

        // Fox-Goodwin: R_n = numerator / denominator
        let numer = two * (one + five * h2_12 * f_curr) - (one - h2_12 * f_prev) / ratio;
        let denom = one - h2_12 * f_next;

        ratio = numer / denom;

        // Guard against blowup
        if ratio.norm() > 1e100 {
            ratio = ratio / ratio.norm();
        }

        f_prev = f_curr;
        f_curr = f_next;
    }

    // Extract S-matrix at matching radius
    // Numerical log-derivative: L = (R_N - 1) / h + correction
    // More precisely: L_numerical = k * r_match * (ratio - 1) / h / u
    // But with ratio variable, the log-derivative is:
    //   u'(r)/u(r) ≈ (R_n - 1) / h  at the matching point
    let log_deriv = (ratio - one) / h;
    let rho = k * r_match;

    // Get Coulomb functions at matching radius
    let eta = sommerfeld_eta(channel, energy);
    let coulomb = coulomb_wave(eta, rho, l, 1)?;

    let f_l = coulomb.f[0];
    let g_l = coulomb.g[0];
    let fp_l = coulomb.fp[0];
    let gp_l = coulomb.gp[0];
    let sigma_l = coulomb.sigma[0];

    // Match: u = alpha * F + beta * G
    // u'/u = L  =>  alpha = L * G - k * G'  ,  beta = k * F' - L * F
    // (using Wronskian F*G' - F'*G = 1)
    // But L is complex (from optical potential), so:
    let l_times_r = log_deriv * r_match;
    let alpha = Complex64::new(g_l, 0.0) * l_times_r - Complex64::new(k * r_match * gp_l, 0.0);
    let beta = Complex64::new(k * r_match * fp_l, 0.0) - Complex64::new(f_l, 0.0) * l_times_r;

    // S_l = exp(2i*sigma_l) * (alpha - i*beta) / (alpha + i*beta)
    let i = Complex64::new(0.0, 1.0);
    let phase = Complex64::from_polar(1.0, 2.0 * sigma_l);
    let s_lj = phase * (alpha - i * beta) / (alpha + i * beta);

    Ok(s_lj)
}

/// Compute reduced mass for a channel (in amu).
fn reduced_mass_channel(channel: &Channel) -> f64 {
    let m1 = channel.projectile.mass_amu();
    let m2 = channel.target.a() as f64; // approximate mass in amu
    units::reduced_mass(m1, m2)
}

/// Compute Sommerfeld parameter eta for a channel.
fn sommerfeld_eta(channel: &Channel, energy: f64) -> f64 {
    let z1 = channel.projectile.z() as f64;
    let z2 = channel.target.z() as f64;
    if z1 == 0.0 || z2 == 0.0 {
        return 0.0;
    }
    let mu = reduced_mass_channel(channel);
    units::sommerfeld_parameter(z1, z2, mu, energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omp::CustomOmp;
    use nucrust_core::{Nuclide, Projectile};

    fn fe56_n_channel() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn numerov_s_matrix_finite() {
        // S-matrix element should be finite
        let omp = CustomOmp::default();
        let ch = fe56_n_channel();
        let config = NumerovConfig::default();

        let s = numerov_integrate(&omp, &ch, 5.0, 0, 0.5, &config).unwrap();
        assert!(s.norm().is_finite(), "|S| = {} should be finite", s.norm());
    }

    #[test]
    fn numerov_completes_without_error() {
        // Numerov integration should complete successfully
        let omp = CustomOmp::default();
        let ch = fe56_n_channel();
        let config = NumerovConfig::default();

        let result = numerov_integrate(&omp, &ch, 5.0, 2, 2.5, &config);
        assert!(result.is_ok(), "Numerov should succeed: {:?}", result.err());
    }

    #[test]
    fn numerov_multiple_l_values() {
        let omp = CustomOmp::default();
        let ch = fe56_n_channel();
        let config = NumerovConfig::default();

        // Should be able to compute for l=0, 1, 2, 3
        for l in 0..4 {
            let j = l as f64 + 0.5;
            let result = numerov_integrate(&omp, &ch, 5.0, l, j, &config);
            assert!(
                result.is_ok(),
                "Numerov failed for l={}: {:?}",
                l,
                result.err()
            );
        }
    }
}
