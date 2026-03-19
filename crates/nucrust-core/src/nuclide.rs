use crate::CoreError;
use serde::{Deserialize, Serialize};

/// Nuclide identifier (immutable, Copy, hashable).
///
/// Represents a nucleus with atomic number Z and mass number A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Nuclide {
    z: u16,
    a: u16,
}

impl Nuclide {
    /// Create a new nuclide with validation.
    ///
    /// # Constraints
    /// - Z: 0..=118
    /// - A: 1..=350
    /// - A >= Z (neutron number N = A - Z >= 0)
    pub fn new(z: u16, a: u16) -> Result<Self, CoreError> {
        if z > 118 || a == 0 || a > 350 || a < z {
            return Err(CoreError::InvalidNuclide { z, a });
        }
        Ok(Self { z, a })
    }

    /// Atomic number Z.
    #[inline]
    pub fn z(&self) -> u16 {
        self.z
    }

    /// Mass number A.
    #[inline]
    pub fn a(&self) -> u16 {
        self.a
    }

    /// Neutron number N = A - Z.
    #[inline]
    pub fn n(&self) -> u16 {
        self.a - self.z
    }
}

impl std::fmt::Display for Nuclide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.z, self.a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_nuclides() {
        let fe56 = Nuclide::new(26, 56).unwrap();
        assert_eq!(fe56.z(), 26);
        assert_eq!(fe56.a(), 56);
        assert_eq!(fe56.n(), 30);
    }

    #[test]
    fn neutron() {
        let n = Nuclide::new(0, 1).unwrap();
        assert_eq!(n.z(), 0);
        assert_eq!(n.n(), 1);
    }

    #[test]
    fn invalid_z_too_large() {
        assert!(Nuclide::new(119, 200).is_err());
    }

    #[test]
    fn invalid_a_zero() {
        assert!(Nuclide::new(0, 0).is_err());
    }

    #[test]
    fn invalid_a_less_than_z() {
        assert!(Nuclide::new(10, 5).is_err());
    }

    #[test]
    fn invalid_a_too_large() {
        assert!(Nuclide::new(26, 351).is_err());
    }

    #[test]
    fn copy_and_hash() {
        let a = Nuclide::new(26, 56).unwrap();
        let b = a; // Copy
        assert_eq!(a, b);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }
}
