use crate::{CollisionMatrix, SpinParity, TransmissionCoeffs};

/// Deformation parameters for axially-symmetric nuclei.
///
/// Nuclear surface: R(θ) = R₀[1 + Σ_λ β_λ Y_{λ0}(θ)]
#[derive(Debug, Clone, Copy)]
pub struct DeformationParams {
    /// Quadrupole deformation β₂.
    pub beta2: f64,
    /// Octupole deformation β₃ (optional, usually zero for ground-state rotational bands).
    pub beta3: f64,
    /// Hexadecapole deformation β₄.
    pub beta4: f64,
}

impl DeformationParams {
    /// Create with quadrupole deformation only.
    pub fn quadrupole(beta2: f64) -> Self {
        Self {
            beta2,
            beta3: 0.0,
            beta4: 0.0,
        }
    }

    /// Create with β₂ and β₄.
    pub fn quadrupole_hexadecapole(beta2: f64, beta4: f64) -> Self {
        Self {
            beta2,
            beta3: 0.0,
            beta4,
        }
    }

    /// Whether this nucleus has significant deformation.
    pub fn is_deformed(&self, threshold: f64) -> bool {
        self.beta2.abs() > threshold
    }

    /// Non-zero deformation multipolarities and their β values.
    pub fn multipoles(&self) -> Vec<(u32, f64)> {
        let mut v = Vec::new();
        if self.beta2.abs() > 1e-10 {
            v.push((2, self.beta2));
        }
        if self.beta3.abs() > 1e-10 {
            v.push((3, self.beta3));
        }
        if self.beta4.abs() > 1e-10 {
            v.push((4, self.beta4));
        }
        v
    }
}

/// A coupled channel in the CC optical model.
///
/// Describes one state in the rotational (or vibrational) band
/// that is coupled in the CC equations.
#[derive(Debug, Clone)]
pub struct CoupledState {
    /// Index within the coupled-channel set (0 = ground state).
    pub index: usize,
    /// Spin-parity of this state (e.g., 0+, 2+, 4+ for even-even rotors).
    pub spin_parity: SpinParity,
    /// Excitation energy relative to ground state (MeV).
    pub excitation_energy: f64,
}

/// Rotational band for an even-even nucleus.
///
/// Ground-state rotational band: 0⁺, 2⁺, 4⁺, 6⁺, ...
/// with E(I) = (ℏ²/2𝒥) I(I+1).
#[derive(Debug, Clone)]
pub struct RotationalBand {
    /// Coupled states in this band.
    pub states: Vec<CoupledState>,
}

impl RotationalBand {
    /// Create a ground-state rotational band for an even-even nucleus.
    ///
    /// `max_spin` is the maximum spin to include (e.g., 4 for 0⁺,2⁺,4⁺).
    /// `energies` provides the experimental excitation energies for each state.
    /// If not enough energies are provided, uses rigid-rotor formula E(I) = E(2⁺) * I(I+1)/6.
    pub fn even_even(max_spin: u32, energies: &[f64]) -> Self {
        assert!(
            max_spin % 2 == 0,
            "max_spin must be even for even-even nuclei"
        );
        let mut states = Vec::new();

        // Ground state
        states.push(CoupledState {
            index: 0,
            spin_parity: SpinParity {
                two_j: 0,
                parity: crate::Parity::Positive,
            },
            excitation_energy: 0.0,
        });

        let e2plus = energies.first().copied().unwrap_or(0.0);

        for i in (2..=max_spin).step_by(2) {
            let idx = (i / 2) as usize;
            let energy = if idx <= energies.len() {
                energies.get(idx - 1).copied().unwrap_or_else(|| {
                    // Rigid-rotor estimate: E(I) = E(2+) * I(I+1) / 6
                    e2plus * (i * (i + 1)) as f64 / 6.0
                })
            } else {
                e2plus * (i * (i + 1)) as f64 / 6.0
            };
            states.push(CoupledState {
                index: idx,
                spin_parity: SpinParity {
                    two_j: (2 * i) as i32,
                    parity: crate::Parity::Positive,
                },
                excitation_energy: energy,
            });
        }

        Self { states }
    }

    /// Number of coupled states.
    pub fn n_states(&self) -> usize {
        self.states.len()
    }
}

/// Output from optical model calculation: either spherical or coupled-channel.
#[derive(Debug, Clone)]
pub enum TransmissionOutput {
    /// Spherical (uncoupled) transmission coefficients.
    Spherical(TransmissionCoeffs),
    /// Coupled-channel collision matrix.
    Coupled(CoupledTransmission),
}

/// Coupled-channel transmission result.
///
/// Contains the full collision matrix plus derived diagonal transmission coefficients.
#[derive(Debug, Clone)]
pub struct CoupledTransmission {
    /// Full collision matrix U_{cc'}(E) for all coupled channels.
    pub collision_matrix: CollisionMatrix,
    /// Diagonal transmission coefficients T_c(E) = 1 - |U_{cc}|² for each channel,
    /// extracted from the collision matrix for use in HF calculations.
    pub diagonal_transmission: TransmissionCoeffs,
    /// Description of coupled states.
    pub band: RotationalBand,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deformation_quadrupole() {
        let d = DeformationParams::quadrupole(0.22);
        assert!((d.beta2 - 0.22).abs() < 1e-15);
        assert!((d.beta3).abs() < 1e-15);
        assert!((d.beta4).abs() < 1e-15);
        assert!(d.is_deformed(0.05));
        assert!(!d.is_deformed(0.3));
    }

    #[test]
    fn deformation_multipoles() {
        let d = DeformationParams::quadrupole_hexadecapole(0.22, -0.07);
        let m = d.multipoles();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].0, 2);
        assert_eq!(m[1].0, 4);
    }

    #[test]
    fn rotational_band_even_even() {
        // ²³⁸U: E(2+) = 0.04491 MeV, E(4+) = 0.14848 MeV
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);
        assert_eq!(band.n_states(), 3); // 0+, 2+, 4+
        assert_eq!(band.states[0].spin_parity.two_j, 0);
        assert!((band.states[0].excitation_energy).abs() < 1e-15);
        assert_eq!(band.states[1].spin_parity.two_j, 4); // 2J = 4 for I=2
        assert!((band.states[1].excitation_energy - 0.04491).abs() < 1e-10);
        assert_eq!(band.states[2].spin_parity.two_j, 8); // 2J = 8 for I=4
        assert!((band.states[2].excitation_energy - 0.14848).abs() < 1e-10);
    }

    #[test]
    fn rotational_band_rigid_rotor_fallback() {
        // Only provide E(2+), E(4+) should be estimated
        let band = RotationalBand::even_even(4, &[0.04491]);
        assert_eq!(band.n_states(), 3);
        // E(4+) ~ E(2+) * 4*5/6 = 0.04491 * 20/6 ≈ 0.1497
        let e4_est = 0.04491 * 20.0 / 6.0;
        assert!((band.states[2].excitation_energy - e4_est).abs() < 1e-10);
    }
}
