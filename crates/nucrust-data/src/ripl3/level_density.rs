//! RIPL-3 level density parameter parsers.
//!
//! Handles:
//! - `densities/total/level-densities-*.dat`: phenomenological NLD parameters
//! - `densities/microscopic/z???.dat`: HFB microscopic density tables

use nucrust_core::{CoreError, Nuclide};

use crate::fixed_field::*;

/// Phenomenological level density parameters for a single nuclide.
#[derive(Debug, Clone)]
pub struct LevelDensityParams {
    pub nuclide: Nuclide,
    /// Level density parameter a (1/MeV).
    pub a: f64,
    /// Pairing energy correction delta (MeV).
    pub delta: f64,
    /// Constant temperature T (MeV) for CT model.
    pub temperature: Option<f64>,
    /// Energy shift E0 (MeV) for CT model.
    pub e0: Option<f64>,
    /// Matching energy E_match (MeV) for Gilbert-Cameron composite model.
    pub e_match: Option<f64>,
    /// Spin cutoff parameter sigma.
    pub sigma: Option<f64>,
    /// Number of discrete levels used in fitting (Nmax).
    pub n_discrete: Option<u32>,
    /// Average s-wave resonance spacing D0 (eV).
    pub d0: Option<f64>,
}

/// Parse RIPL-3 phenomenological level density parameters.
///
/// Expected format: header lines (starting with # or !), then data lines with
/// Z(i5) A(i5) ... a(f10) delta(f10) ... in fixed-width columns.
pub fn parse_level_density_params(input: &str) -> Result<Vec<LevelDensityParams>, CoreError> {
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

        // Common layout: Z(5) A(5) Ndisc(5) D0(12) a(10) delta(10) T(10) E0(10) Ematch(10) sigma(10)
        let n_discrete = fixed_field_u32(line, 10, 15);
        let d0 = fixed_field_f64_opt(line, 15, 27);
        let a_param = fixed_field_f64(line, 27, 37).unwrap_or(0.0);
        let delta = fixed_field_f64(line, 37, 47).unwrap_or(0.0);
        let temperature = fixed_field_f64_opt(line, 47, 57);
        let e0 = fixed_field_f64_opt(line, 57, 67);
        let e_match = fixed_field_f64_opt(line, 67, 77);
        let sigma = fixed_field_f64_opt(line, 77, 87);

        entries.push(LevelDensityParams {
            nuclide,
            a: a_param,
            delta,
            temperature,
            e0,
            e_match,
            sigma,
            n_discrete,
            d0,
        });
    }

    Ok(entries)
}

/// A single entry in the HFB microscopic level density table.
#[derive(Debug, Clone)]
pub struct HfbDensityEntry {
    /// Excitation energy (MeV).
    pub excitation: f64,
    /// Spin J.
    pub spin: f64,
    /// Parity: +1 or -1.
    pub parity: i8,
    /// Level density at (U, J, pi) (1/MeV).
    pub density: f64,
}

/// HFB microscopic level density table for a single nuclide.
#[derive(Debug, Clone)]
pub struct HfbDensityTable {
    pub nuclide: Nuclide,
    pub entries: Vec<HfbDensityEntry>,
}

/// Parse RIPL-3 HFB microscopic level density table (densities/microscopic/z???.dat).
///
/// Format: header lines, then blocks per nuclide.
/// Each block: header with Z, A, then rows of U(f10) J(f5) pi(i3) rho(e12).
pub fn parse_hfb_density_table(input: &str, z: u16) -> Result<Vec<HfbDensityTable>, CoreError> {
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

        // Look for nuclide header: Z(i5) A(i5) n_entries(i5) ...
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

        // Verify Z matches expected
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
            if fields.len() < 4 {
                continue;
            }

            let excitation: f64 = match fields[0].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let spin: f64 = fields[1].parse().unwrap_or(0.0);
            let parity: i8 = fields[2].parse().unwrap_or(1);
            let density: f64 = fields[3].parse().unwrap_or(0.0);

            entries.push(HfbDensityEntry {
                excitation,
                spin,
                parity,
                density,
            });
        }

        tables.push(HfbDensityTable { nuclide, entries });
        i += 1;
    }

    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_NLD: &str = "\
# Z    A  Ndisc       D0        a     delta         T        E0    Ematch     sigma
   26   56   20  2.500E+04    6.210    -0.520     0.880    -1.160     3.240     3.510
   26   57   15  1.200E+03    6.350    -1.230     0.920    -1.780     2.560     3.620
";

    #[test]
    fn parse_nld_params() {
        let result = parse_level_density_params(SAMPLE_NLD).unwrap();
        assert_eq!(result.len(), 2);

        let fe56 = &result[0];
        assert_eq!(fe56.nuclide.z(), 26);
        assert_eq!(fe56.nuclide.a(), 56);
        assert!((fe56.a - 6.210).abs() < 0.001);
        assert!((fe56.delta - (-0.520)).abs() < 0.001);
        assert!(fe56.temperature.is_some());
        assert!((fe56.temperature.unwrap() - 0.880).abs() < 0.001);
        assert_eq!(fe56.n_discrete, Some(20));
    }

    const SAMPLE_HFB: &str = "   26   56    3\n    0.000  0.0  1  1.000E-05\n    0.500  0.0  1  3.200E-03\n    1.000  1.0 -1  1.500E-01\n";

    #[test]
    fn parse_hfb_table() {
        let tables = parse_hfb_density_table(SAMPLE_HFB, 26).unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].nuclide.a(), 56);
        assert_eq!(tables[0].entries.len(), 3);
        assert!((tables[0].entries[2].excitation - 1.0).abs() < 1e-10);
        assert_eq!(tables[0].entries[2].parity, -1);
    }
}
