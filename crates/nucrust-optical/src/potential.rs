//! Woods-Saxon potential shape factors and helper functions.

use nucrust_core::Channel;

/// Woods-Saxon shape factor: f(r) = 1 / (1 + exp((r - R) / a))
/// where R = r0 * A_target^{1/3}
#[inline]
pub fn woods_saxon(r: f64, r0: f64, a: f64, channel: &Channel) -> f64 {
    let big_r = r0 * (channel.target.a() as f64).cbrt();
    let x = (r - big_r) / a;
    // Guard against overflow in exp
    if x > 500.0 {
        return 0.0;
    }
    1.0 / (1.0 + x.exp())
}

/// Surface-peaked Woods-Saxon derivative (normalized):
/// -4a * df/dr = 4 * exp((r-R)/a) / (1 + exp((r-R)/a))^2
///
/// This is the derivative form used for surface imaginary potential.
#[inline]
pub fn woods_saxon_deriv(r: f64, r0: f64, a: f64, channel: &Channel) -> f64 {
    let big_r = r0 * (channel.target.a() as f64).cbrt();
    let x = (r - big_r) / a;
    if x.abs() > 500.0 {
        return 0.0;
    }
    let ex = x.exp();
    4.0 * ex / ((1.0 + ex) * (1.0 + ex))
}

/// Spin-orbit coupling factor: <l·s> = [j(j+1) - l(l+1) - s(s+1)] / 2
/// For nucleons, s = 1/2.
#[inline]
pub fn spin_orbit_factor(l: u32, j: f64) -> f64 {
    let l_f = l as f64;
    let s = 0.5; // nucleon spin
    (j * (j + 1.0) - l_f * (l_f + 1.0) - s * (s + 1.0)) / 2.0
}

/// Coulomb potential for a uniformly charged sphere.
/// V_C(r) = Z1*Z2*e^2 / (2*R_C) * (3 - (r/R_C)^2)  for r < R_C
/// V_C(r) = Z1*Z2*e^2 / r                              for r >= R_C
///
/// Uses: e^2 = alpha * hbar*c ≈ 1.4399764 MeV·fm
#[inline]
pub fn coulomb_potential(r: f64, r_c: f64, channel: &Channel) -> f64 {
    let z1 = channel.projectile.z() as f64;
    let z2 = channel.target.z() as f64;
    if z1 == 0.0 || z2 == 0.0 {
        return 0.0; // neutrons have no Coulomb
    }

    let e2 = nucrust_core::units::FINE_STRUCTURE * nucrust_core::units::HBAR_C; // ~1.44 MeV·fm
    let r_coul = r_c * (channel.target.a() as f64).cbrt();

    if r >= r_coul {
        z1 * z2 * e2 / r
    } else {
        let ratio = r / r_coul;
        z1 * z2 * e2 / (2.0 * r_coul) * (3.0 - ratio * ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::{Nuclide, Projectile};

    fn fe56_neutron_channel() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    fn fe56_proton_channel() -> Channel {
        Channel {
            projectile: Projectile::Proton,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn ws_center_is_half() {
        let ch = fe56_neutron_channel();
        let r0 = 1.25;
        let a = 0.65;
        let big_r = r0 * 56.0_f64.cbrt();
        let f = woods_saxon(big_r, r0, a, &ch);
        assert!((f - 0.5).abs() < 1e-14);
    }

    #[test]
    fn ws_monotonically_decreasing() {
        let ch = fe56_neutron_channel();
        let r0 = 1.25;
        let a = 0.65;
        let f1 = woods_saxon(1.0, r0, a, &ch);
        let f2 = woods_saxon(5.0, r0, a, &ch);
        let f3 = woods_saxon(10.0, r0, a, &ch);
        assert!(f1 > f2);
        assert!(f2 > f3);
    }

    #[test]
    fn ws_deriv_peaks_at_surface() {
        let ch = fe56_neutron_channel();
        let r0 = 1.25;
        let a = 0.65;
        let big_r = r0 * 56.0_f64.cbrt();
        // Derivative peaks at r = R
        let d_surface = woods_saxon_deriv(big_r, r0, a, &ch);
        let d_inside = woods_saxon_deriv(big_r - 2.0, r0, a, &ch);
        let d_outside = woods_saxon_deriv(big_r + 2.0, r0, a, &ch);
        assert!(d_surface > d_inside);
        assert!(d_surface > d_outside);
        assert!((d_surface - 1.0).abs() < 1e-14); // Peak value is exactly 1.0
    }

    #[test]
    fn spin_orbit_j_plus_half() {
        // j = l + 1/2: <l·s> = l/2
        assert!((spin_orbit_factor(2, 2.5) - 1.0).abs() < 1e-14);
        assert!((spin_orbit_factor(0, 0.5) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn spin_orbit_j_minus_half() {
        // j = l - 1/2: <l·s> = -(l+1)/2
        assert!((spin_orbit_factor(2, 1.5) - (-1.5)).abs() < 1e-14);
        assert!((spin_orbit_factor(1, 0.5) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn coulomb_zero_for_neutron() {
        let ch = fe56_neutron_channel();
        assert_eq!(coulomb_potential(5.0, 1.25, &ch), 0.0);
    }

    #[test]
    fn coulomb_positive_for_proton() {
        let ch = fe56_proton_channel();
        let v = coulomb_potential(10.0, 1.25, &ch);
        assert!(v > 0.0);
    }

    #[test]
    fn coulomb_continuous_at_radius() {
        let ch = fe56_proton_channel();
        let r_c = 1.25;
        let r_coul = r_c * 56.0_f64.cbrt();
        let eps = 1e-8;
        let v_inside = coulomb_potential(r_coul - eps, r_c, &ch);
        let v_outside = coulomb_potential(r_coul + eps, r_c, &ch);
        assert!((v_inside - v_outside).abs() < 1e-3);
    }
}
