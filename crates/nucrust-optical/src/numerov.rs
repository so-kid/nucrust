//! Fox-Goodwin ratio-variable Numerov integration.
//!
//! Integrates the radial Schrödinger equation outward from r_min to r_match,
//! then extracts the S-matrix element by matching to Coulomb wave functions.

use nucrust_core::backend::{Kinematics, NumerovConfig, OmpEnergy};
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
    let kin = RadialKinematics::new(channel, energy, config);

    // Number of steps
    let n_steps = ((r_match - config.r_min) / h).ceil() as usize;
    if n_steps < 3 {
        return Err(CoreError::InvalidParameter {
            name: "r_match",
            value: r_match,
            reason: "matching radius too small for Numerov integration",
        });
    }

    // u'' = f(r) u with f(r) = 2μ/ℏ² V(r) - k² + l(l+1)/r²
    // (non-relativistically k² = 2μE/ℏ²; V in MeV, r in fm).
    let l_f = l as f64;
    let centrifugal = l_f * (l_f + 1.0);
    let k2 = kin.k * kin.k;

    let f_at = |r: f64| -> Complex64 {
        if r < 1e-10 {
            return Complex64::new(1e30, 0.0); // Large repulsive barrier at origin
        }
        let v = potential.potential(r, kin.omp_energy, l, j, channel);
        v / kin.hbar2_over_2mu + Complex64::new(centrifugal / (r * r) - k2, 0.0)
    };

    // Fox-Goodwin ratio variable integration
    let h2_12 = h * h / 12.0;
    let one = Complex64::new(1.0, 0.0);
    let two = Complex64::new(2.0, 0.0);
    let five = Complex64::new(5.0, 0.0);

    let r0 = config.r_min;
    let r1 = r0 + h;

    let f0 = f_at(r0);
    let f1 = f_at(r1);

    // Initial ratio from the regular power series u = r^{l+1} (1 + c r^2 + ...), with
    // c = q / (2 (2l + 3)) and q = 2 mu (V - E) / hbar^2 near the origin. Without the r^2
    // term the start admixes the irregular solution at O(k^2 h^2), which for l = 0
    // (irregular ~ cos kr) does not decay and limits T_0 to ~1e-5 relative accuracy.
    let q0 = f0 - Complex64::new(centrifugal / (r0 * r0), 0.0);
    let c = q0 / (2.0 * (2.0 * l_f + 3.0));
    let mut ratio = Complex64::new((r1 / r0).powi(l as i32 + 1), 0.0) * (one + c * r1 * r1)
        / (one + c * r0 * r0);

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

    // Two-point matching: `ratio` = u(r_N) / u(r_{N-1}) on the last two grid points,
    // matched to u = alpha F_l(k r) + beta G_l(k r). Unlike the one-sided log-derivative
    // (R_N - 1)/h, which is only O(h) accurate, this keeps the O(h^4) Numerov accuracy.
    let r_n = config.r_min + n_steps as f64 * h;
    s_matrix_two_point(ratio, kin.k, r_n - h, r_n, kin.eta, l)
}

/// Wave number, potential coupling and OMP evaluation energy for one channel energy.
struct RadialKinematics {
    /// Wave number k (1/fm).
    k: f64,
    /// ℏ²/(2μ) in MeV·fm² (μ: reduced mass, or reduced total energy relativistically).
    hbar2_over_2mu: f64,
    /// Sommerfeld parameter η = Z₁Z₂e²/(ℏ v) = Z₁Z₂e² μ/(ℏ² k).
    eta: f64,
    /// Energy passed to the optical potential (CM or LAB, see [`OmpEnergy`]).
    omp_energy: f64,
}

impl RadialKinematics {
    /// Kinematics for the relative-motion energy `energy` (MeV) per `config`.
    fn new(channel: &Channel, energy: f64, config: &NumerovConfig) -> Self {
        let m1 = channel.projectile.mass_amu();
        let m2 = config.target_mass_amu.unwrap_or(channel.target.a() as f64);
        let e_lab = energy * (m1 + m2) / m2;

        let (k, hbar2_over_2mu) = match config.kinematics {
            Kinematics::NonRelativistic => {
                let mu = units::reduced_mass(m1, m2);
                (
                    units::wave_number(mu, energy),
                    HBAR_C * HBAR_C / (2.0 * mu * units::AMU_MEV),
                )
            }
            Kinematics::Relativistic => {
                // Invariant mass s = (m1 + m2)^2 + 2 m2 E_lab; CM momentum p and total CM
                // energies E_1, E_2 of the two particles (all in MeV, c = 1).
                let (m1c2, m2c2) = (m1 * units::AMU_MEV, m2 * units::AMU_MEV);
                let s = (m1c2 + m2c2).powi(2) + 2.0 * m2c2 * e_lab;
                let p2 = m2c2 * m2c2 * e_lab * (e_lab + 2.0 * m1c2) / s;
                let e1 = (m1c2 * m1c2 + p2).sqrt();
                let e2 = (m2c2 * m2c2 + p2).sqrt();
                let mu_e = e1 * e2 / (e1 + e2);
                (p2.sqrt() / HBAR_C, HBAR_C * HBAR_C / (2.0 * mu_e))
            }
        };

        let z1z2 = (channel.projectile.z() as f64) * (channel.target.z() as f64);
        let eta = if z1z2 == 0.0 {
            0.0
        } else {
            z1z2 * units::FINE_STRUCTURE * HBAR_C / (2.0 * hbar2_over_2mu * k)
        };

        let omp_energy = match config.omp_energy {
            OmpEnergy::CenterOfMass => energy,
            OmpEnergy::Laboratory => e_lab,
        };

        Self {
            k,
            hbar2_over_2mu,
            eta,
            omp_energy,
        }
    }
}

/// S-matrix element from the ratio `ratio = u(r2) / u(r1)` of the interior solution at two
/// radii in the potential-free region, by matching to `u = alpha F_l + beta G_l`.
///
/// Returns `exp(2 i sigma_l) (alpha - i beta) / (alpha + i beta)`, the S-matrix element in
/// the sign convention of this crate's potentials (absorption = positive `Im V`).
fn s_matrix_two_point(
    ratio: Complex64,
    k: f64,
    r1: f64,
    r2: f64,
    eta: f64,
    l: u32,
) -> Result<Complex64, CoreError> {
    let c1 = coulomb_wave(eta, k * r1, l, 1)?;
    let c2 = coulomb_wave(eta, k * r2, l, 1)?;
    let (f1, g1) = (c1.f[0], c1.g[0]);
    let (f2, g2) = (c2.f[0], c2.g[0]);

    // alpha F2 + beta G2 = ratio (alpha F1 + beta G1)
    let alpha = ratio * g1 - g2;
    let beta = Complex64::new(f2, 0.0) - ratio * f1;

    let i = Complex64::new(0.0, 1.0);
    let phase = Complex64::from_polar(1.0, 2.0 * c1.sigma[0]);
    Ok(phase * (alpha - i * beta) / (alpha + i * beta))
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

    fn zero_potential() -> CustomOmp {
        CustomOmp {
            v_real: 0.0,
            w_vol: 0.0,
            w_surf: 0.0,
            v_so: 0.0,
            w_so: 0.0,
            ..CustomOmp::default()
        }
    }

    #[test]
    fn numerov_free_particle_gives_unit_s_matrix() {
        // With V = 0 the regular solution is F_l itself, so S must be exactly 1. The old
        // one-sided log-derivative (R_N - 1)/h matching left an O(h) phase error here.
        let omp = zero_potential();
        let ch = fe56_n_channel();
        let config = NumerovConfig::default();
        for &e in &[0.001, 0.1, 1.0, 10.0] {
            for l in [0u32, 1, 3] {
                let s = numerov_integrate(&omp, &ch, e, l, l as f64 + 0.5, &config).unwrap();
                // O(h^4) phase error at h = 0.05 fm stays ~5e-6 up to 10 MeV; the
                // log-derivative matching was off by ~k h / 2 (~2e-2 at 10 MeV).
                assert!(
                    (s - Complex64::new(1.0, 0.0)).norm() < 2e-5,
                    "E={e} l={l}: S = {s}"
                );
            }
        }
    }

    #[test]
    fn numerov_converges_at_fourth_order_in_step_size() {
        // KD s-wave at 1 keV: the log-derivative matching was ~1.5% off at h = 0.05 fm.
        let kd = crate::omp::KoningDelaroche;
        let ch = fe56_n_channel();
        let t = |h: f64| {
            let config = NumerovConfig {
                step_size: h,
                ..NumerovConfig::default()
            };
            1.0 - numerov_integrate(&kd, &ch, 0.001, 0, 0.5, &config)
                .unwrap()
                .norm_sqr()
        };
        // Fourth order: halving h divides the error by ~16. Both the O(h) log-derivative
        // matching and an O(h^2) start (u ~ r^{l+1} without the r^2 term) break this.
        let reference = t(0.003125);
        let (e1, e2) = (t(0.05) - reference, t(0.025) - reference);
        let order = (e1 / e2).log2();
        assert!(
            (3.5..4.5).contains(&order),
            "convergence order {order} (errors {e1:e} at h=0.05, {e2:e} at h=0.025)"
        );
        assert!(
            (e1 / reference).abs() < 2e-6,
            "T_0 rel. error {} at h = 0.05 fm",
            e1 / reference
        );
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
