use crate::{Nuclide, Parity};
use num_complex::Complex64;

use crate::Channel;

/// Electromagnetic multipole type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Multipole {
    /// Electric dipole.
    E1,
    /// Magnetic dipole.
    M1,
    /// Electric quadrupole.
    E2,
    /// Magnetic quadrupole.
    M2,
    /// Electric octupole.
    E3,
}

impl Multipole {
    /// Multipolarity order L.
    #[inline]
    pub fn order(&self) -> u32 {
        match self {
            Self::E1 | Self::M1 => 1,
            Self::E2 | Self::M2 => 2,
            Self::E3 => 3,
        }
    }

    /// Whether this is an electric (E) or magnetic (M) transition.
    #[inline]
    pub fn is_electric(&self) -> bool {
        matches!(self, Self::E1 | Self::E2 | Self::E3)
    }
}

/// Nuclear level density model.
pub trait LevelDensity: Send + Sync {
    /// Level density rho(U, J, pi) at excitation energy U, spin J, parity pi.
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, parity: Parity) -> f64;

    /// Total level density rho_tot(U) = sum_{J,pi} (2J+1) rho(U, J, pi).
    fn rho_total(&self, nuclide: &Nuclide, excitation: f64) -> f64;

    /// Model name (for logging/output).
    fn name(&self) -> &str;
}

/// Gamma-ray strength function model.
pub trait GammaStrength: Send + Sync {
    /// f_{XL}(E_gamma) for multipole XL.
    fn strength(&self, nuclide: &Nuclide, e_gamma: f64, multipole: Multipole) -> f64;

    /// Model name.
    fn name(&self) -> &str;
}

/// Optical model potential.
pub trait OpticalPotential: Send + Sync {
    /// Complex potential V(r, E_cm, l, j) at radius r.
    ///
    /// Returns (real part, imaginary part) in MeV.
    fn potential(&self, r: f64, e_cm: f64, l: u32, j: f64, channel: &Channel) -> Complex64;

    /// Coulomb radius R_C (fm).
    fn coulomb_radius(&self, channel: &Channel) -> f64;

    /// Matching radius R_match (fm) beyond which nuclear force is negligible.
    fn matching_radius(&self, channel: &Channel) -> f64;

    /// Model name.
    fn name(&self) -> &str;
}
