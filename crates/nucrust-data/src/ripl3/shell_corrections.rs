//! RIPL-3 shell correction parser (`shellcorrections/shell-corrections.dat`).

use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;

/// Shell correction entry for a single nuclide.
#[derive(Debug, Clone)]
pub struct ShellCorrection {
    /// Target nuclide.
    pub nuclide: Nuclide,
    /// Shell correction energy (MeV).
    pub shell_correction: f64,
    /// Pairing correction energy (MeV).
    pub pairing_correction: Option<f64>,
    /// Deformation parameter beta2.
    pub beta2: Option<f64>,
    /// Deformation parameter beta4.
    pub beta4: Option<f64>,
}

/// Parse RIPL-3 shell corrections file.
///
/// Format: comment lines (#), then Z(i5) A(i5) shell(f10) pairing(f10) beta2(f10) beta4(f10)
pub fn parse_shell_corrections(input: &str) -> Result<Vec<ShellCorrection>, CoreError> {
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

        let shell_correction = fixed_field_f64(line, 10, 20).unwrap_or(0.0);
        let pairing_correction = fixed_field_f64_opt(line, 20, 30);
        let beta2 = fixed_field_f64_opt(line, 30, 40);
        let beta4 = fixed_field_f64_opt(line, 40, 50);

        entries.push(ShellCorrection {
            nuclide,
            shell_correction,
            pairing_correction,
            beta2,
            beta4,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# Z    A    Shell     Pairing   Beta2     Beta4
   26   56    -3.470     0.000     0.000     0.000
   82  208   -11.430    -0.860     0.000     0.000
   92  238    -1.240    -0.530     0.274     0.093
";

    #[test]
    fn parse_shell_entries() {
        let result = parse_shell_corrections(SAMPLE).unwrap();
        assert_eq!(result.len(), 3);

        let fe56 = &result[0];
        assert_eq!(fe56.nuclide.z(), 26);
        assert_eq!(fe56.nuclide.a(), 56);
        assert!((fe56.shell_correction - (-3.47)).abs() < 0.01);

        let pb208 = &result[1];
        assert!((pb208.shell_correction - (-11.43)).abs() < 0.01);

        let u238 = &result[2];
        assert!(u238.beta2.is_some());
        assert!((u238.beta2.unwrap() - 0.274).abs() < 0.001);
    }
}
