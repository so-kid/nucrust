//! RIPL-3 resonance parameter parser (`resonances/resonances?.dat`).

use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;

/// Resonance parameter entry for a single nuclide.
#[derive(Debug, Clone)]
pub struct ResonanceParams {
    /// Target nuclide.
    pub nuclide: Nuclide,
    /// Average s-wave resonance spacing D0 (eV).
    pub d0: Option<f64>,
    /// Average s-wave neutron strength function S0 (x10^-4).
    pub s0: Option<f64>,
    /// Average s-wave radiative width Gamma_gamma0 (meV).
    pub gamma_gamma0: Option<f64>,
    /// Average p-wave resonance spacing D1 (eV).
    pub d1: Option<f64>,
    /// Average p-wave neutron strength function S1 (x10^-4).
    pub s1: Option<f64>,
}

/// Parse RIPL-3 resonance parameter file.
///
/// Format: comment lines (#), then Z(i5) A(i5) D0(e12) S0(e12) Gg0(e12) D1(e12) S1(e12)
pub fn parse_resonances(input: &str) -> Result<Vec<ResonanceParams>, CoreError> {
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

        // Remaining fields after Z(i5) A(i5): whitespace-separated
        let rest = line.get(10..).unwrap_or("");
        let fields: Vec<&str> = rest.split_whitespace().collect();

        fn parse_opt(fields: &[&str], idx: usize) -> Option<f64> {
            fields.get(idx).and_then(|s| {
                let v: f64 = s.parse().ok()?;
                if v == -1.0 {
                    None
                } else {
                    Some(v)
                }
            })
        }

        let d0 = parse_opt(&fields, 0);
        let s0 = parse_opt(&fields, 1);
        let gamma_gamma0 = parse_opt(&fields, 2);
        let d1 = parse_opt(&fields, 3);
        let s1 = parse_opt(&fields, 4);

        entries.push(ResonanceParams {
            nuclide,
            d0,
            s0,
            gamma_gamma0,
            d1,
            s1,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# Z    A         D0          S0       Gg0          D1          S1
   26   57 2.500E+04   5.600E-05   1.200E+03
   28   59 1.200E+04   3.200E-05   8.500E+02   4.800E+03   1.100E-04
";

    #[test]
    fn parse_resonance_entries() {
        let result = parse_resonances(SAMPLE).unwrap();
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].nuclide.z(), 26);
        assert_eq!(result[0].nuclide.a(), 57);
        assert!(result[0].d0.is_some());
        assert!(result[0].d1.is_none()); // not provided

        assert_eq!(result[1].nuclide.z(), 28);
        assert!(result[1].d1.is_some());
    }
}
