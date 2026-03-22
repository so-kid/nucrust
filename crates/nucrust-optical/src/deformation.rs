//! Coupled-channel optical model for deformed nuclei.
//!
//! Implements:
//! - Deformed Woods-Saxon potential with rotational coupling
//! - Coupling potential matrix V^{J_T}_{cc'}(R) construction
//! - Deformed OMP wrapping the spherical Koning-Delaroche potential

use nucrust_core::coupled_channel::{DeformationParams, RotationalBand};
use nucrust_core::traits::OpticalPotential;
use nucrust_core::wigner;
use nucrust_core::{Channel, SpinParity};
use num_complex::Complex64;

/// Coupled-channel information for a given total J^π.
///
/// For a rotational band (0⁺, 2⁺, 4⁺, ...) coupled with a neutron of orbital
/// angular momentum l, the coupled channels are labeled by (l, I, J_total)
/// where I is the target spin and l, I couple to J_total.
#[derive(Debug, Clone)]
pub struct CoupledChannelSystem {
    /// Deformation parameters of the target.
    pub deformation: DeformationParams,
    /// Rotational band states.
    pub band: RotationalBand,
    /// Channel list: (l, j, state_index, excitation_energy).
    pub channels: Vec<CcChannel>,
    /// Total J (in 2J representation).
    pub two_j_total: i32,
}

/// A single channel in the CC system.
#[derive(Debug, Clone)]
pub struct CcChannel {
    /// Orbital angular momentum.
    pub l: u32,
    /// Total angular momentum j = l ± 1/2.
    pub j: f64,
    /// Index into the rotational band.
    pub state_index: usize,
    /// Excitation energy of this state (MeV).
    pub excitation_energy: f64,
    /// Target spin (2I representation).
    pub two_i: i32,
}

impl CoupledChannelSystem {
    /// Build coupled channels for a given J_total (in 2J representation).
    ///
    /// For each state I in the rotational band, find all (l, j) that
    /// can couple to J_total via the triangle condition |I - j| ≤ J_total ≤ I + j.
    pub fn build(
        deformation: DeformationParams,
        band: &RotationalBand,
        two_j_total: i32,
        l_max: u32,
    ) -> Self {
        let mut channels = Vec::new();

        for state in &band.states {
            let two_i = state.spin_parity.two_j;

            for l in 0..=l_max {
                // j = l ± 1/2
                let j_values: Vec<f64> = if l == 0 {
                    vec![0.5]
                } else {
                    vec![l as f64 - 0.5, l as f64 + 0.5]
                };

                for j in j_values {
                    let two_j = (2.0 * j) as i32;

                    // Triangle condition: |I - j| ≤ J_total ≤ I + j
                    if SpinParity::triangle_condition(two_i, two_j, two_j_total) {
                        // Parity conservation: π_target × (-1)^l = π_total
                        // For even-even ground-band: π_target = +1 for all states
                        // So we need (-1)^l to match the desired total parity
                        channels.push(CcChannel {
                            l,
                            j,
                            state_index: state.index,
                            excitation_energy: state.excitation_energy,
                            two_i,
                        });
                    }
                }
            }
        }

        Self {
            deformation,
            band: band.clone(),
            channels,
            two_j_total,
        }
    }

    /// Number of coupled channels.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Compute the coupling potential matrix element V_{cc'}(r) for the deformed
    /// Woods-Saxon potential with nucleon (spin-1/2) scattering.
    ///
    /// Uses the jI coupling scheme (Thompson & Nunes, 2009):
    ///
    /// V^J_{(lj,I),(l'j',I')} = Σ_λ f_λ(r) × Z_λ(c,c',J)
    ///
    /// where the geometric coupling factor Z_λ is:
    ///
    /// Z_λ = (-1)^{j'+I+J} √((2j+1)(2j'+1))  {j I J; I' j' λ}
    ///     × (-1)^{l+1/2-j} √((2j+1)(2l+1)(2l'+1)(2λ+1)/(4π))
    ///     × (l 0 λ 0 | l' 0)  {l 1/2 j; j' λ l'}
    ///     × ⟨I ‖ T_λ ‖ I'⟩
    ///
    /// First 6j handles j-I recoupling, second handles l-s recoupling.
    /// For even-even rotor: ⟨I ‖ T_λ ‖ I'⟩ = CG(I' 0; λ 0 | I 0) × √(2I+1)
    pub fn coupling_matrix_element(
        &self,
        c: usize,
        c_prime: usize,
        _r: f64,
        r0: f64,
        dv_dr: f64,
    ) -> f64 {
        if c == c_prime {
            return 0.0; // Diagonal is handled by the spherical potential
        }

        let ch_c = &self.channels[c];
        let ch_cp = &self.channels[c_prime];

        let two_l = 2 * ch_c.l as i32;
        let two_lp = 2 * ch_cp.l as i32;
        let two_j = (2.0 * ch_c.j) as i32;
        let two_jp = (2.0 * ch_cp.j) as i32;
        let two_i = ch_c.two_i;
        let two_ip = ch_cp.two_i;
        let two_jt = self.two_j_total;
        let two_s = 1; // nucleon spin = 1/2

        let mut v_coupling = 0.0;

        for &(lambda, beta) in self.deformation.multipoles().iter() {
            let two_lam = 2 * lambda as i32;

            // Selection rules on orbital angular momenta
            if !SpinParity::triangle_condition(two_l, two_lam, two_lp) {
                continue;
            }
            // Selection rules on target spins
            if !SpinParity::triangle_condition(two_i, two_lam, two_ip) {
                continue;
            }
            // Parity: (-1)^(l + lambda + l') must be even
            if (ch_c.l + lambda + ch_cp.l) % 2 != 0 {
                continue;
            }

            // Form factor: f_λ(r) = -β_λ R₀ dV/dr
            let f_lambda = -beta * r0 * dv_dr;

            // 6j #1: {j I J; I' j' λ} — j-I recoupling
            let w6j_1 = wigner::wigner_6j(two_j, two_i, two_jt, two_ip, two_jp, two_lam);
            if w6j_1.abs() < 1e-30 {
                continue;
            }

            // 6j #2: {l 1/2 j; j' λ l'} — l-s recoupling
            let w6j_2 = wigner::wigner_6j(two_l, two_s, two_j, two_jp, two_lam, two_lp);
            if w6j_2.abs() < 1e-30 {
                continue;
            }

            // CG coefficient: (l 0 λ 0 | l' 0) for the Y_λ coupling
            let cg_orbital = wigner::clebsch_gordan(two_l, 0, two_lam, 0, two_lp, 0);
            if cg_orbital.abs() < 1e-30 {
                continue;
            }

            // Reduced matrix element of target operator (even-even rotor):
            // ⟨I ‖ T_λ ‖ I'⟩ = CG(I' 0; λ 0 | I 0) × √(2I+1)
            let rme_t = wigner::clebsch_gordan(two_ip, 0, two_lam, 0, two_i, 0)
                * ((two_i + 1) as f64).sqrt();

            // Phase: (-1)^{j'+I+J} × (-1)^{l+1/2-j}
            // In 2j notation: (-1)^{(two_jp+two_i+two_jt)/2} × (-1)^{(two_l+1-two_j)/2}
            let phase_exp_1 = (two_jp + two_i + two_jt) / 2;
            let phase_exp_2 = (two_l + two_s - two_j) / 2;
            let total_phase_exp = phase_exp_1 + phase_exp_2;
            let phase = if total_phase_exp % 2 == 0 { 1.0 } else { -1.0 };

            // Geometric prefactor: √((2j+1)(2j'+1)) × √((2j+1)(2l+1)(2l'+1)(2λ+1)/(4π))
            let geom = ((two_j + 1) as f64 * (two_jp + 1) as f64).sqrt()
                * ((two_j + 1) as f64
                    * (two_l + 1) as f64
                    * (two_lp + 1) as f64
                    * (two_lam + 1) as f64
                    / (4.0 * std::f64::consts::PI))
                    .sqrt();

            v_coupling += f_lambda * phase * geom * w6j_1 * w6j_2 * cg_orbital * rme_t;
        }

        v_coupling
    }
}

/// Deformed nucleon optical model potential wrapping Koning-Delaroche.
///
/// Adds rotational coupling to the spherical KD potential.
#[derive(Debug, Clone)]
pub struct DeformedKoningDelaroche {
    /// Deformation parameters.
    pub deformation: DeformationParams,
    /// Rotational band of the target.
    pub band: RotationalBand,
}

impl DeformedKoningDelaroche {
    /// Creates a new deformed Koning-Delaroche potential with the given deformation and rotational band.
    pub fn new(deformation: DeformationParams, band: RotationalBand) -> Self {
        Self { deformation, band }
    }

    /// Compute the derivative of the real volume Woods-Saxon potential dV/dr.
    ///
    /// dV/dr = V_depth × d/dr[1/(1+exp((r-R)/a))]
    ///       = -V_depth/(a) × exp((r-R)/a) / (1+exp((r-R)/a))²
    pub fn ws_derivative(&self, r: f64, r0: f64, a: f64, a_target: f64) -> f64 {
        let big_r = r0 * a_target.cbrt();
        let x = (r - big_r) / a;
        if x.abs() > 500.0 {
            return 0.0;
        }
        let ex = x.exp();
        -ex / (a * (1.0 + ex) * (1.0 + ex))
    }
}

/// Compute the radial derivative of the real volume potential (Woods-Saxon only,
/// no spin-orbit or Coulomb) for coupling form factor calculation.
///
/// Returns V_depth × df_WS/dr where f_WS is the volume Woods-Saxon shape.
pub fn volume_ws_derivative(r: f64, v_depth: f64, r0: f64, a: f64, a_target: f64) -> f64 {
    let big_r = r0 * a_target.cbrt();
    let x = (r - big_r) / a;
    if x.abs() > 500.0 {
        return 0.0;
    }
    let ex = x.exp();
    // d/dr f_WS = -1/a × exp(x) / (1+exp(x))²
    -v_depth * ex / (a * (1.0 + ex) * (1.0 + ex))
}

/// Build the W-matrix (coupling + centrifugal + diagonal potential) at radius r.
///
/// W_{cc'}(r) = (2μ/ℏ²) [V^{J_T}_{cc'}(r) + δ_{cc'}(l_c(l_c+1)/r² + V_sph(r) - E_c)]
///
/// where E_c = E - ε_c (kinetic energy in channel c, accounting for excitation energy).
///
/// Returns an N×N matrix as a flat `Vec<Complex64>` in row-major order.
pub fn build_w_matrix(
    cc_system: &CoupledChannelSystem,
    spherical_potential: &dyn OpticalPotential,
    channel: &Channel,
    energy: f64,
    r: f64,
    hbar2_over_2mu: f64,
) -> Vec<Complex64> {
    let n = cc_system.n_channels();
    let mut w = vec![Complex64::new(0.0, 0.0); n * n];

    // KD volume parameters for the coupling derivative.
    // Use typical KD geometry for the derivative.
    let a_target = channel.target.a() as f64;
    let a13 = a_target.cbrt();
    let rv = 1.3039 - 0.4054 / a13;
    let av = 0.6778 - 1.487e-4 * a_target;
    let r0_fm = rv * a13;

    // Approximate real-volume depth for coupling form factor.
    // For KD: V ≈ 50 MeV (typical depth at relevant energies).
    // Use the WS derivative of the real volume term only.
    let dv_dr = volume_ws_derivative(r, 50.0, rv, av, a_target);

    for c in 0..n {
        let ch_c = &cc_system.channels[c];
        let l_c = ch_c.l as f64;
        let e_c = energy - ch_c.excitation_energy; // Available kinetic energy

        // Diagonal: spherical potential + centrifugal - kinetic energy
        let v_sph = spherical_potential.potential(r, energy, ch_c.l, ch_c.j, channel);
        let centrifugal = if r > 1e-10 {
            l_c * (l_c + 1.0) / (r * r)
        } else {
            1e30
        };

        w[c * n + c] =
            (v_sph - Complex64::new(e_c, 0.0)) / hbar2_over_2mu + Complex64::new(centrifugal, 0.0);

        // Off-diagonal: coupling potential
        for c_prime in 0..n {
            if c_prime == c {
                continue;
            }
            let v_coupling = cc_system.coupling_matrix_element(c, c_prime, r, r0_fm, dv_dr);
            w[c * n + c_prime] = Complex64::new(v_coupling / hbar2_over_2mu, 0.0);
        }
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::coupled_channel::{DeformationParams, RotationalBand};
    use nucrust_core::{Nuclide, Projectile};

    fn u238_channel() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(92, 238).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn cc_system_build() {
        let deformation = DeformationParams::quadrupole(0.22);
        // ²³⁸U: 0⁺(0.0), 2⁺(0.04491), 4⁺(0.14848)
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);

        // J_total = 1/2 (2J=1), l_max = 4
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 4);

        // Should have channels: for each state I, find l,j with triangle (I, j, 1/2)
        // I=0: j=1/2 → l=0 (j=1/2). Also l=1 (j=1/2). etc.
        assert!(
            cc.n_channels() > 0,
            "Should have at least one channel, got {}",
            cc.n_channels()
        );
    }

    #[test]
    fn cc_system_channels_correct() {
        let deformation = DeformationParams::quadrupole(0.22);
        // Simple: just 0⁺ and 2⁺
        let band = RotationalBand::even_even(2, &[0.04491]);

        // J_total = 1/2 (2J=1), l_max = 3
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 3);

        // I=0, j=1/2: l can be 0 or 1 (j=1/2 for both, triangle with J=1/2)
        // I=2, j can be 3/2, 5/2 to couple to J=1/2: need |I-j| ≤ 1/2 ≤ I+j
        // I=2(two_i=4), j=3/2(two_j=3): |4-3|=1 ≤ 1 ≤ 7 ✓ and 4+3+1=8 even ✓
        // I=2(two_i=4), j=5/2(two_j=5): |4-5|=1 ≤ 1 ≤ 9 ✓ and 4+5+1=10 even ✓
        assert!(
            cc.n_channels() >= 4,
            "Expected at least 4 channels, got {}",
            cc.n_channels()
        );
    }

    #[test]
    fn coupling_diagonal_is_zero() {
        let deformation = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(2, &[0.04491]);
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 2);

        // Diagonal coupling should be zero
        for c in 0..cc.n_channels() {
            let v = cc.coupling_matrix_element(c, c, 5.0, 5.0, -10.0);
            assert!(
                v.abs() < 1e-15,
                "Diagonal coupling should be zero, got {}",
                v
            );
        }
    }

    #[test]
    fn coupling_nonzero_for_deformed() {
        let deformation = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(2, &[0.04491]);
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 3);

        if cc.n_channels() >= 2 {
            // At least some off-diagonal elements should be nonzero
            let mut found_nonzero = false;
            for c in 0..cc.n_channels() {
                for cp in 0..cc.n_channels() {
                    if c != cp {
                        let v = cc.coupling_matrix_element(c, cp, 5.0, 5.0, -10.0);
                        if v.abs() > 1e-10 {
                            found_nonzero = true;
                        }
                    }
                }
            }
            assert!(found_nonzero, "Should have nonzero coupling for β₂=0.22");
        }
    }

    #[test]
    fn coupling_zero_for_spherical() {
        let deformation = DeformationParams::quadrupole(0.0); // No deformation
        let band = RotationalBand::even_even(2, &[0.04491]);
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 3);

        // All off-diagonal coupling should be zero
        for c in 0..cc.n_channels() {
            for cp in 0..cc.n_channels() {
                if c != cp {
                    let v = cc.coupling_matrix_element(c, cp, 5.0, 5.0, -10.0);
                    assert!(
                        v.abs() < 1e-15,
                        "Coupling should be zero for spherical nucleus, got {}",
                        v
                    );
                }
            }
        }
    }

    #[test]
    fn w_matrix_has_correct_size() {
        let deformation = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(2, &[0.04491]);
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 2);
        let ch = u238_channel();
        let omp = crate::omp::KoningDelaroche;
        let hbar2_2mu = 20.736; // approximate ℏ²/(2μ) in MeV·fm²

        let w = build_w_matrix(&cc, &omp, &ch, 1.0, 5.0, hbar2_2mu);
        let n = cc.n_channels();
        assert_eq!(w.len(), n * n);
    }

    #[test]
    fn w_matrix_diagonal_finite() {
        let deformation = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(2, &[0.04491]);
        let cc = CoupledChannelSystem::build(deformation, &band, 1, 2);
        let ch = u238_channel();
        let omp = crate::omp::KoningDelaroche;
        let hbar2_2mu = 20.736;

        let w = build_w_matrix(&cc, &omp, &ch, 1.0, 5.0, hbar2_2mu);
        let n = cc.n_channels();

        for c in 0..n {
            let diag = w[c * n + c];
            assert!(
                diag.re.is_finite() && diag.im.is_finite(),
                "W[{c},{c}] should be finite, got {diag}"
            );
        }
    }
}
