//! RIPL-3 optical model parameter database parser (`om-parameter-u.dat`).
//!
//! Los Alamos convention: each OMP set = 1 header line + 7 parameter lines.
//! Each parameter line has potential depths V0..V4 (cols 12-66, 11 chars each)
//! and geometry parameters r0, r1, C, a0, a1.
//! Set numbers: 1-3999 = neutron, 4000-5999 = proton, 6000+ = composite.

use nucrust_core::CoreError;

use crate::fixed_field::*;

/// Optical model parameter set from RIPL-3.
#[derive(Debug, Clone)]
pub struct OmpParameterSet {
    /// Set number (1-3999: neutron, 4000-5999: proton, 6000+: composite).
    pub set_number: u32,
    /// Reference string.
    pub reference: String,
    /// Minimum Z for applicability.
    pub z_min: u16,
    /// Maximum Z for applicability.
    pub z_max: u16,
    /// Minimum A for applicability.
    pub a_min: u16,
    /// Maximum A for applicability.
    pub a_max: u16,
    /// Minimum energy (MeV).
    pub e_min: f64,
    /// Maximum energy (MeV).
    pub e_max: f64,
    /// Potential type (1 = volume, 2 = surface, etc.).
    pub potential_type: u32,
    /// 7 rows of potential parameters.
    /// Each row: [V0, V1, V2, V3, V4, r0, r1, C, a0, a1]
    /// Rows: 0=real volume, 1=imag volume, 2=real surface,
    ///        3=imag surface, 4=real spin-orbit, 5=imag spin-orbit, 6=Coulomb
    pub rows: [[f64; 10]; 7],
}

impl OmpParameterSet {
    /// Whether this set applies to neutrons.
    pub fn is_neutron(&self) -> bool {
        self.set_number < 4000
    }

    /// Whether this set applies to protons.
    pub fn is_proton(&self) -> bool {
        self.set_number >= 4000 && self.set_number < 6000
    }

    /// Whether this set applies to composite particles.
    pub fn is_composite(&self) -> bool {
        self.set_number >= 6000
    }

    /// Check if this parameter set is applicable for a given (Z, A, E).
    pub fn is_applicable(&self, z: u16, a: u16, e_mev: f64) -> bool {
        z >= self.z_min
            && z <= self.z_max
            && a >= self.a_min
            && a <= self.a_max
            && e_mev >= self.e_min
            && e_mev <= self.e_max
    }
}

/// Parse RIPL-3 optical model parameter database (`om-parameter-u.dat`).
///
/// Format: Each set consists of 1 header line + 7 parameter lines.
/// Header: set_number(i5) + type(i3) + Z_min(i4) + Z_max(i4) + A_min(i4) + A_max(i4)
///         + E_min(f8.1) + E_max(f8.1) + reference(rest)
/// Parameter lines: label(a12) + V0(f11) + V1(f11) + V2(f11) + V3(f11) + V4(f11)
///                  + r0(f7) + r1(f7) + C(f7) + a0(f7) + a1(f7)
pub fn parse_omp_database(input: &str) -> Result<Vec<OmpParameterSet>, CoreError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut sets = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip empty and comment lines
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
            i += 1;
            continue;
        }

        // Try to parse header line
        let set_number = match fixed_field_u32(line, 0, 5) {
            Some(n) if n >= 1 => n,
            _ => {
                i += 1;
                continue;
            }
        };

        let potential_type = fixed_field_u32(line, 5, 8).unwrap_or(1);
        let z_min = fixed_field_u16(line, 8, 12).unwrap_or(0);
        let z_max = fixed_field_u16(line, 12, 16).unwrap_or(118);
        let a_min = fixed_field_u16(line, 16, 20).unwrap_or(1);
        let a_max = fixed_field_u16(line, 20, 24).unwrap_or(350);
        let e_min = fixed_field_f64(line, 24, 32).unwrap_or(0.0);
        let e_max = fixed_field_f64(line, 32, 40).unwrap_or(200.0);
        let reference = fixed_field_str(line, 40, line.len().min(120)).to_string();

        // Read 7 parameter lines
        if i + 7 >= lines.len() {
            break;
        }

        let mut rows = [[0.0_f64; 10]; 7];
        for row in &mut rows {
            i += 1;
            let pline = lines[i];
            // Potential depths: V0..V4 at columns 12-66, each 11 chars
            for (k, cell) in row.iter_mut().enumerate().take(5) {
                let start = 12 + k * 11;
                let end = start + 11;
                *cell = fixed_field_f64(pline, start, end).unwrap_or(0.0);
            }
            // Geometry parameters: r0, r1, C, a0, a1 at columns 67+, each 7 chars
            for k in 0..5 {
                let start = 67 + k * 7;
                let end = start + 7;
                row[5 + k] = fixed_field_f64(pline, start, end).unwrap_or(0.0);
            }
        }

        sets.push(OmpParameterSet {
            set_number,
            reference,
            z_min,
            z_max,
            a_min,
            a_max,
            e_min,
            e_max,
            potential_type,
            rows,
        });

        i += 1;
    }

    Ok(sets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_omp_data() -> String {
        // Simplified OMP data: 1 header + 7 parameter lines
        let mut s = String::new();
        // Header: set=1, type=1, Z=20-82, A=40-210, E=0.001-200.0
        s.push_str("    1  1  20  82  40 210   0.001 200.000 Koning-Delaroche 2003\n");
        // 7 rows: label(12) + V0..V4(5x11) + r0..a1(5x7)
        for row_idx in 0..7 {
            s.push_str(&format!(
                "Row{:<9}{:>11.4}{:>11.4}{:>11.4}{:>11.4}{:>11.4}{:>7.3}{:>7.3}{:>7.3}{:>7.3}{:>7.3}\n",
                row_idx, 52.9 - row_idx as f64, 0.7, 0.0, 0.0, 0.0,
                1.245, 0.0, 0.0, 0.66, 0.0
            ));
        }
        s
    }

    #[test]
    fn parse_single_set() {
        let data = sample_omp_data();
        let sets = parse_omp_database(&data).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].set_number, 1);
        assert!(sets[0].is_neutron());
        assert!(!sets[0].is_proton());
        assert_eq!(sets[0].z_min, 20);
        assert_eq!(sets[0].z_max, 82);
        assert!(sets[0].rows[0][0] > 50.0); // V0 of real volume
    }

    #[test]
    fn applicability_check() {
        let data = sample_omp_data();
        let sets = parse_omp_database(&data).unwrap();
        let s = &sets[0];
        assert!(s.is_applicable(26, 56, 10.0)); // Fe-56 at 10 MeV
        assert!(!s.is_applicable(10, 20, 10.0)); // Z=10 < z_min=20
    }

    #[test]
    fn empty_input() {
        let sets = parse_omp_database("# comment only\n").unwrap();
        assert!(sets.is_empty());
    }
}
