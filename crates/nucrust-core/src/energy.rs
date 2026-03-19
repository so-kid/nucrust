use crate::CoreError;

/// Energy grid (sorted, positive values guaranteed).
///
/// Internal unit: MeV.
#[derive(Debug, Clone)]
pub struct EnergyGrid {
    values: Vec<f64>,
}

impl EnergyGrid {
    /// Create a logarithmically spaced energy grid.
    pub fn logarithmic(e_min: f64, e_max: f64, n_points: usize) -> Result<Self, CoreError> {
        Self::validate_range(e_min, e_max, n_points)?;
        let log_min = e_min.ln();
        let log_max = e_max.ln();
        let values = if n_points == 1 {
            vec![e_min]
        } else {
            (0..n_points)
                .map(|i| {
                    let t = i as f64 / (n_points - 1) as f64;
                    (log_min + t * (log_max - log_min)).exp()
                })
                .collect()
        };
        Ok(Self { values })
    }

    /// Create a linearly spaced energy grid.
    pub fn linear(e_min: f64, e_max: f64, n_points: usize) -> Result<Self, CoreError> {
        Self::validate_range(e_min, e_max, n_points)?;
        let values = if n_points == 1 {
            vec![e_min]
        } else {
            (0..n_points)
                .map(|i| {
                    let t = i as f64 / (n_points - 1) as f64;
                    e_min + t * (e_max - e_min)
                })
                .collect()
        };
        Ok(Self { values })
    }

    /// Create from arbitrary values (will be sorted and validated).
    pub fn from_values(mut values: Vec<f64>) -> Result<Self, CoreError> {
        if values.is_empty() {
            return Err(CoreError::InvalidParameter {
                name: "n_points",
                value: 0.0,
                reason: "energy grid must have at least 1 point",
            });
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if values[0] <= 0.0 || !values[0].is_finite() {
            return Err(CoreError::EnergyOutOfRange {
                value: values[0],
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(Self { values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn validate_range(e_min: f64, e_max: f64, n_points: usize) -> Result<(), CoreError> {
        if n_points == 0 {
            return Err(CoreError::InvalidParameter {
                name: "n_points",
                value: 0.0,
                reason: "energy grid must have at least 1 point",
            });
        }
        if e_min <= 0.0 {
            return Err(CoreError::EnergyOutOfRange {
                value: e_min,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if e_max < e_min {
            return Err(CoreError::InvalidParameter {
                name: "e_max",
                value: e_max,
                reason: "e_max must be >= e_min",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logarithmic_grid() {
        let g = EnergyGrid::logarithmic(0.001, 10.0, 5).unwrap();
        assert_eq!(g.len(), 5);
        let s = g.as_slice();
        assert!((s[0] - 0.001).abs() < 1e-15);
        assert!((s[4] - 10.0).abs() < 1e-12);
        // Check monotonicity
        for i in 1..s.len() {
            assert!(s[i] > s[i - 1]);
        }
    }

    #[test]
    fn linear_grid() {
        let g = EnergyGrid::linear(1.0, 5.0, 5).unwrap();
        let s = g.as_slice();
        assert_eq!(s.len(), 5);
        assert!((s[0] - 1.0).abs() < 1e-15);
        assert!((s[4] - 5.0).abs() < 1e-15);
        assert!((s[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn from_values_sorts() {
        let g = EnergyGrid::from_values(vec![3.0, 1.0, 2.0]).unwrap();
        assert_eq!(g.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_negative_energy() {
        assert!(EnergyGrid::from_values(vec![-1.0, 1.0]).is_err());
        assert!(EnergyGrid::logarithmic(-1.0, 10.0, 5).is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(EnergyGrid::from_values(vec![]).is_err());
        assert!(EnergyGrid::linear(1.0, 2.0, 0).is_err());
    }

    #[test]
    fn single_point() {
        let g = EnergyGrid::logarithmic(1.0, 1.0, 1).unwrap();
        assert_eq!(g.len(), 1);
        assert!((g.as_slice()[0] - 1.0).abs() < 1e-15);
    }
}
