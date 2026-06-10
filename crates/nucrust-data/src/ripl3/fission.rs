//! RIPL-3 fission barrier parser (`fission/fission-barriers-*.dat`).

use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;

/// Fission barrier parameters for a single nuclide.
#[derive(Debug, Clone)]
pub struct FissionBarrier {
    /// Target nuclide.
    pub nuclide: Nuclide,
    /// Height of the first (inner) barrier (MeV).
    pub barrier_a: f64,
    /// Curvature of the first barrier (MeV).
    pub hw_a: f64,
    /// Height of the second (outer) barrier (MeV).
    pub barrier_b: Option<f64>,
    /// Curvature of the second barrier (MeV).
    pub hw_b: Option<f64>,
    /// Height of the third barrier (MeV), if present (e.g., actinides).
    pub barrier_c: Option<f64>,
    /// Curvature of the third barrier (MeV).
    pub hw_c: Option<f64>,
}

/// Parse RIPL-3 fission barrier file.
///
/// Format: comment lines (#), then Z(i5) A(i5) Ba(f10) hwa(f10) Bb(f10) hwb(f10) [Bc(f10) hwc(f10)]
pub fn parse_fission_barriers(input: &str) -> Result<Vec<FissionBarrier>, CoreError> {
    let mut entries = Vec::new();

    for line in input.lines() {
        if is_skippable_line(line) {
            continue;
        }

        let z = match fixed_field_u16(line, 0, 5) {
            Some(z) => z,
            None => continue,
        };
        let a = match fixed_field_u16(line, 5, 10) {
            Some(a) => a,
            None => continue,
        };

        let nuclide = match Nuclide::new(z, a) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let barrier_a = fixed_field_f64(line, 10, 20).unwrap_or(0.0);
        let hw_a = fixed_field_f64(line, 20, 30).unwrap_or(1.0);
        let barrier_b = fixed_field_f64_opt(line, 30, 40);
        let hw_b = fixed_field_f64_opt(line, 40, 50);
        let barrier_c = fixed_field_f64_opt(line, 50, 60);
        let hw_c = fixed_field_f64_opt(line, 60, 70);

        entries.push(FissionBarrier {
            nuclide,
            barrier_a,
            hw_a,
            barrier_b,
            hw_b,
            barrier_c,
            hw_c,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# Z    A   Barrier_A  hw_A     Barrier_B  hw_B
   92  236     5.670     1.040     5.150     0.600
   94  240     6.050     1.060     5.450     0.580     3.100     0.500
";

    #[test]
    fn parse_double_humped() {
        let result = parse_fission_barriers(SAMPLE).unwrap();
        assert_eq!(result.len(), 2);

        let u236 = &result[0];
        assert_eq!(u236.nuclide.z(), 92);
        assert_eq!(u236.nuclide.a(), 236);
        assert!((u236.barrier_a - 5.67).abs() < 0.01);
        assert!(u236.barrier_b.is_some());
        assert!(u236.barrier_c.is_none());
    }

    #[test]
    fn parse_triple_humped() {
        let result = parse_fission_barriers(SAMPLE).unwrap();
        let pu240 = &result[1];
        assert!(pu240.barrier_c.is_some());
        assert!((pu240.barrier_c.unwrap() - 3.1).abs() < 0.01);
    }
}
