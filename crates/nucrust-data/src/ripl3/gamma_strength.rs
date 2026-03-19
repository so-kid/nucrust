//! RIPL-3 gamma-ray strength function parsers.
//!
//! Handles:
//! - `gamma/gdr-parameters.dat`: Giant Dipole Resonance (GDR) Lorentzian parameters
//! - `gamma/gamma-strength/z???.dat`: Tabulated GSF from QRPA calculations

use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;

/// GDR (Giant Dipole Resonance) parameters for the standard Lorentzian model.
#[derive(Debug, Clone)]
pub struct GdrParams {
    pub nuclide: Nuclide,
    /// GDR peak energy E_GDR (MeV) — component 1.
    pub e_gdr1: f64,
    /// GDR width Gamma_GDR (MeV) — component 1.
    pub gamma_gdr1: f64,
    /// GDR peak cross section sigma_GDR (mb) — component 1.
    pub sigma_gdr1: f64,
    /// GDR peak energy E_GDR (MeV) — component 2 (for deformed nuclei).
    pub e_gdr2: Option<f64>,
    /// GDR width Gamma_GDR (MeV) — component 2.
    pub gamma_gdr2: Option<f64>,
    /// GDR peak cross section sigma_GDR (mb) — component 2.
    pub sigma_gdr2: Option<f64>,
}

/// Parse RIPL-3 GDR parameter file (`gamma/gdr-parameters.dat`).
///
/// Format: comment lines starting with #, then fixed-width data lines.
/// Z(i5) A(i5) E1(f10) Gamma1(f10) Sigma1(f10) E2(f10) Gamma2(f10) Sigma2(f10)
pub fn parse_gdr_params(input: &str) -> Result<Vec<GdrParams>, CoreError> {
    let mut entries = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
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

        let e_gdr1 = fixed_field_f64(line, 10, 20).unwrap_or(0.0);
        let gamma_gdr1 = fixed_field_f64(line, 20, 30).unwrap_or(0.0);
        let sigma_gdr1 = fixed_field_f64(line, 30, 40).unwrap_or(0.0);
        let e_gdr2 = fixed_field_f64_opt(line, 40, 50);
        let gamma_gdr2 = fixed_field_f64_opt(line, 50, 60);
        let sigma_gdr2 = fixed_field_f64_opt(line, 60, 70);

        entries.push(GdrParams {
            nuclide,
            e_gdr1,
            gamma_gdr1,
            sigma_gdr1,
            e_gdr2,
            gamma_gdr2,
            sigma_gdr2,
        });
    }

    Ok(entries)
}

/// A single entry in the tabulated gamma-ray strength function.
#[derive(Debug, Clone)]
pub struct GsfTableEntry {
    /// Gamma-ray energy (MeV).
    pub e_gamma: f64,
    /// E1 strength function f(E1) (MeV^{-3}).
    pub f_e1: f64,
    /// M1 strength function f(M1) (MeV^{-3}).
    pub f_m1: f64,
    /// E2 strength function f(E2) (MeV^{-5}), if available.
    pub f_e2: Option<f64>,
}

/// Tabulated GSF for a single nuclide.
#[derive(Debug, Clone)]
pub struct GsfTable {
    pub nuclide: Nuclide,
    pub entries: Vec<GsfTableEntry>,
}

/// Parse RIPL-3 tabulated GSF file (`gamma/gamma-strength/z???.dat`).
///
/// Format: blocks per nuclide, each with a header Z(i5) A(i5) N_entries(i5)
/// followed by data rows: E_gamma(f10) f_E1(e12) f_M1(e12) [f_E2(e12)]
pub fn parse_gsf_table(input: &str, z: u16) -> Result<Vec<GsfTable>, CoreError> {
    let mut tables = Vec::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('!') {
            i += 1;
            continue;
        }

        let line_z = match fixed_field_u16(line, 0, 5) {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };
        let line_a = match fixed_field_u16(line, 5, 10) {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };

        if line_z != z {
            i += 1;
            continue;
        }

        let nuclide = match Nuclide::new(line_z, line_a) {
            Ok(n) => n,
            Err(_) => {
                i += 1;
                continue;
            }
        };

        let n_entries = fixed_field_u32(line, 10, 15).unwrap_or(0) as usize;
        let mut entries = Vec::with_capacity(n_entries);

        for _ in 0..n_entries {
            i += 1;
            if i >= lines.len() {
                break;
            }
            let dline = lines[i];
            let fields: Vec<&str> = dline.split_whitespace().collect();
            if fields.len() < 3 {
                continue;
            }

            let e_gamma: f64 = match fields[0].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let f_e1: f64 = fields[1].parse().unwrap_or(0.0);
            let f_m1: f64 = fields[2].parse().unwrap_or(0.0);
            let f_e2: Option<f64> = fields.get(3).and_then(|s| s.parse().ok());

            entries.push(GsfTableEntry {
                e_gamma,
                f_e1,
                f_m1,
                f_e2,
            });
        }

        tables.push(GsfTable { nuclide, entries });
        i += 1;
    }

    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GDR: &str = "\
# Z    A      E_GDR1  Gamma1  Sigma1    E_GDR2  Gamma2  Sigma2
   26   56    16.360     4.580   136.000
   26   57    16.200     5.200   130.000    18.500     3.100    50.000
";

    #[test]
    fn parse_gdr_single_component() {
        let result = parse_gdr_params(SAMPLE_GDR).unwrap();
        assert_eq!(result.len(), 2);

        let fe56 = &result[0];
        assert_eq!(fe56.nuclide.z(), 26);
        assert_eq!(fe56.nuclide.a(), 56);
        assert!((fe56.e_gdr1 - 16.36).abs() < 0.01);
        assert!((fe56.sigma_gdr1 - 136.0).abs() < 0.1);
        assert!(fe56.e_gdr2.is_none());
    }

    #[test]
    fn parse_gdr_double_component() {
        let result = parse_gdr_params(SAMPLE_GDR).unwrap();
        let fe57 = &result[1];
        assert!(fe57.e_gdr2.is_some());
        assert!((fe57.e_gdr2.unwrap() - 18.5).abs() < 0.01);
    }

    const SAMPLE_GSF: &str =
        "   26   56    2\n    1.000  2.500E-08  1.200E-09\n    2.000  5.100E-08  2.400E-09\n";

    #[test]
    fn parse_gsf_table_entries() {
        let tables = parse_gsf_table(SAMPLE_GSF, 26).unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].entries.len(), 2);
        assert!((tables[0].entries[0].e_gamma - 1.0).abs() < 1e-10);
        assert!(tables[0].entries[0].f_e1 > 0.0);
    }
}
