use serde::{Deserialize, Serialize};

/// Projectile / ejectile particle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Projectile {
    /// Neutron (n).
    Neutron,
    /// Proton (p).
    Proton,
    /// Deuteron (d, 2H).
    Deuteron,
    /// Triton (t, 3H).
    Triton,
    /// Helion (3He).
    Helion,
    /// Alpha particle (4He).
    Alpha,
    /// Gamma ray (photon).
    Gamma,
}

impl Projectile {
    /// Charge number Z of the projectile.
    #[inline]
    pub fn z(&self) -> u16 {
        match self {
            Self::Neutron | Self::Gamma => 0,
            Self::Proton | Self::Deuteron | Self::Triton => 1,
            Self::Helion => 2,
            Self::Alpha => 2,
        }
    }

    /// Mass number A of the projectile.
    #[inline]
    pub fn a(&self) -> u16 {
        match self {
            Self::Gamma => 0,
            Self::Neutron | Self::Proton => 1,
            Self::Deuteron => 2,
            Self::Triton => 3,
            Self::Helion => 3,
            Self::Alpha => 4,
        }
    }

    /// Mass in atomic mass units (amu).
    ///
    /// Values from AME2020 atomic mass evaluation.
    #[inline]
    pub fn mass_amu(&self) -> f64 {
        match self {
            Self::Gamma => 0.0,
            Self::Neutron => 1.008_664_916,
            Self::Proton => 1.007_276_467,
            Self::Deuteron => 2.013_553_213,
            Self::Triton => 3.015_500_716,
            Self::Helion => 3.014_932_247,
            Self::Alpha => 4.001_506_179,
        }
    }

    /// Intrinsic spin of the projectile.
    #[inline]
    pub fn spin(&self) -> f64 {
        match self {
            Self::Neutron | Self::Proton | Self::Triton | Self::Helion => 0.5,
            Self::Deuteron | Self::Gamma => 1.0,
            Self::Alpha => 0.0,
        }
    }

    /// Two times the intrinsic spin (integer representation).
    #[inline]
    pub fn two_spin(&self) -> i32 {
        match self {
            Self::Neutron | Self::Proton | Self::Triton | Self::Helion => 1,
            Self::Deuteron | Self::Gamma => 2,
            Self::Alpha => 0,
        }
    }

    /// Whether the projectile carries electric charge.
    #[inline]
    pub fn is_charged(&self) -> bool {
        self.z() > 0
    }
}

impl std::fmt::Display for Projectile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Neutron => "n",
            Self::Proton => "p",
            Self::Deuteron => "d",
            Self::Triton => "t",
            Self::Helion => "3He",
            Self::Alpha => "a",
            Self::Gamma => "g",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutron_properties() {
        let n = Projectile::Neutron;
        assert_eq!(n.z(), 0);
        assert_eq!(n.a(), 1);
        assert!(!n.is_charged());
        assert_eq!(n.spin(), 0.5);
        assert_eq!(n.two_spin(), 1);
    }

    #[test]
    fn alpha_properties() {
        let a = Projectile::Alpha;
        assert_eq!(a.z(), 2);
        assert_eq!(a.a(), 4);
        assert!(a.is_charged());
        assert_eq!(a.spin(), 0.0);
        assert_eq!(a.two_spin(), 0);
    }

    #[test]
    fn gamma_properties() {
        let g = Projectile::Gamma;
        assert_eq!(g.z(), 0);
        assert_eq!(g.a(), 0);
        assert!(!g.is_charged());
        assert_eq!(g.spin(), 1.0);
    }
}
