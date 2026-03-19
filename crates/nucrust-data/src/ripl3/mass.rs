use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;
use crate::ripl3::types::MassEntry;

/// Parse RIPL-3 mass table (masses/mass-frdm.dat or similar).
///
/// Expected format: comment lines starting with '#', then data lines with
/// Z, A, element symbol, mass excess, etc. in fixed-width columns.
/// Common format: Z(i5) A(i5) symbol(a3) mass_excess(f12.6) ...
pub fn parse_mass_table(input: &str) -> Result<Vec<MassEntry>, CoreError> {
    let mut entries = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
            continue;
        }

        // Try to parse as data line: Z(1-5) N(6-10) A(11-15) ... mass_excess(~)
        // The exact format varies between mass files. We support a common layout:
        // Z(i5) A(i5) element(a4) mass_excess(f12.6) binding_energy_per_a(f12.6) ...
        let z = match fixed_field_u16(line, 0, 5) {
            Some(z) => z,
            None => continue,
        };
        let a = match fixed_field_u16(line, 5, 10) {
            Some(a) => a,
            None => {
                // Try alternate layout: Z N A
                let n = fixed_field_u16(line, 5, 10);
                let a_alt = fixed_field_u16(line, 10, 15);
                match (n, a_alt) {
                    (Some(_), Some(a)) => a,
                    _ => continue,
                }
            }
        };

        let nuclide = match Nuclide::new(z, a) {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Mass excess typically starts around column 18-30
        let mass_excess = match fixed_field_f64(line, 18, 30) {
            Some(v) => v,
            None => {
                // Try wider range
                fixed_field_f64(line, 15, 30).unwrap_or(0.0)
            }
        };

        let binding_energy_per_a = fixed_field_f64(line, 30, 42);
        let beta_decay_energy = fixed_field_f64(line, 42, 54);
        let atomic_mass_micro_u = fixed_field_f64(line, 54, 68);

        entries.push(MassEntry {
            nuclide,
            mass_excess,
            binding_energy_per_a,
            beta_decay_energy,
            atomic_mass_micro_u,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MASS: &str = "\
# Z    A   EL    Mass Excess   BE/A       Beta       AtomicMass
   26   56  Fe      -60.601     8.790     -4.566    55921677.432
   26   57  Fe      -60.176     8.770     -0.836    56935393.320
   28   58  Ni      -60.223     8.732      2.820    57935342.640
";

    #[test]
    fn parse_mass_entries() {
        let result = parse_mass_table(SAMPLE_MASS).unwrap();
        assert_eq!(result.len(), 3);

        assert_eq!(result[0].nuclide.z(), 26);
        assert_eq!(result[0].nuclide.a(), 56);
        assert!((result[0].mass_excess - (-60.601)).abs() < 0.01);

        assert_eq!(result[2].nuclide.z(), 28);
        assert_eq!(result[2].nuclide.a(), 58);
    }

    #[test]
    fn skip_comments_and_empty() {
        let input = "# comment\n\n! another comment\n";
        let result = parse_mass_table(input).unwrap();
        assert!(result.is_empty());
    }
}
