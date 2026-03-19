use crate::CoreError;
use serde::{Deserialize, Serialize};

/// Parity quantum number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Parity {
    Positive,
    Negative,
}

impl Parity {
    /// Create from sign (+1 or -1).
    pub fn from_sign(sign: i8) -> Self {
        if sign >= 0 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    /// Return +1 or -1.
    #[inline]
    pub fn sign(&self) -> i8 {
        match self {
            Self::Positive => 1,
            Self::Negative => -1,
        }
    }
}

impl std::ops::Mul for Parity {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self == rhs {
            Self::Positive
        } else {
            Self::Negative
        }
    }
}

/// Spin-parity quantum number J^pi.
///
/// Uses `two_j` (2*J) integer representation to avoid floating-point for half-integer spins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpinParity {
    /// 2*J (e.g., J=5/2 -> two_j=5). i32 to support subtraction in triangle conditions.
    pub two_j: i32,
    pub parity: Parity,
}

impl SpinParity {
    pub fn new(two_j: i32, parity: Parity) -> Result<Self, CoreError> {
        if two_j < 0 {
            return Err(CoreError::InvalidParameter {
                name: "two_j",
                value: two_j as f64,
                reason: "must be non-negative",
            });
        }
        Ok(Self { two_j, parity })
    }

    /// Triangle condition: |j1 - j2| <= j3 <= j1 + j2 (all in 2J representation).
    ///
    /// Also checks that j1 + j2 + j3 is even (angular momentum coupling selection rule).
    #[inline]
    pub fn triangle_condition(two_j1: i32, two_j2: i32, two_j3: i32) -> bool {
        let diff = (two_j1 - two_j2).abs();
        let sum = two_j1 + two_j2;
        two_j3 >= diff && two_j3 <= sum && (two_j1 + two_j2 + two_j3) % 2 == 0
    }
}

impl std::fmt::Display for SpinParity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = match self.parity {
            Parity::Positive => "+",
            Parity::Negative => "-",
        };
        if self.two_j % 2 == 0 {
            write!(f, "{}{}", self.two_j / 2, p)
        } else {
            write!(f, "{}/2{}", self.two_j, p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_multiplication() {
        assert_eq!(Parity::Positive * Parity::Positive, Parity::Positive);
        assert_eq!(Parity::Positive * Parity::Negative, Parity::Negative);
        assert_eq!(Parity::Negative * Parity::Negative, Parity::Positive);
    }

    #[test]
    fn spin_parity_display() {
        let sp = SpinParity::new(5, Parity::Positive).unwrap();
        assert_eq!(sp.to_string(), "5/2+");

        let sp = SpinParity::new(4, Parity::Negative).unwrap();
        assert_eq!(sp.to_string(), "2-");

        let sp = SpinParity::new(0, Parity::Positive).unwrap();
        assert_eq!(sp.to_string(), "0+");
    }

    #[test]
    fn triangle_condition() {
        // |1 - 1| = 0 <= 2 <= 2 = 1 + 1, and 1+1+2=4 even
        assert!(SpinParity::triangle_condition(1, 1, 2));
        assert!(SpinParity::triangle_condition(1, 1, 0));
        // 1+1+1=3 odd -> false
        assert!(!SpinParity::triangle_condition(1, 1, 1));
        // |2 - 4| = 2 <= 3? 3 < 2 is false... wait 3 >= 2. 3 <= 6. 2+4+3=9 odd -> false
        assert!(!SpinParity::triangle_condition(2, 4, 3));
        // 2+4+2=8 even, 2 >= 2, 2 <= 6 -> true
        assert!(SpinParity::triangle_condition(2, 4, 2));
    }

    #[test]
    fn invalid_negative_two_j() {
        assert!(SpinParity::new(-1, Parity::Positive).is_err());
    }
}
