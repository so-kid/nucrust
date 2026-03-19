use nucrust_core::{CoreError, Nuclide, Parity};

use crate::fixed_field::*;
use crate::ripl3::types::*;

/// Parse RIPL-3 discrete levels file (levels/z???.dat).
///
/// Each isotope block consists of:
/// - 1 header line: FORMAT (a5, 6i5, 2f12.6)
/// - N level lines: FORMAT (i3, 1x, f10.6, 1x, f5.1, i3, ...)
/// - For each level with Ng > 0: Ng gamma lines: FORMAT (39x, i4, 1x, f10.4, 3(1x, e10.3))
pub fn parse_discrete_levels(input: &str) -> Result<Vec<IsotopeData>, CoreError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        // Skip empty lines
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Try to parse as header line
        let header = match parse_header(line, i + 1) {
            Some(h) => h,
            None => {
                i += 1;
                continue;
            }
        };
        i += 1;

        // Parse levels
        let mut levels = Vec::with_capacity(header.n_levels);
        for _ in 0..header.n_levels {
            if i >= lines.len() {
                break;
            }
            let level_line = lines[i];
            i += 1;

            let level_result = parse_level_line(level_line);
            let mut level = match level_result {
                Some(l) => l,
                None => continue,
            };

            // Parse gamma transitions for this level
            for _ in 0..level.n_gammas {
                if i >= lines.len() {
                    break;
                }
                let gamma_line = lines[i];
                i += 1;
                if let Some(gamma) = parse_gamma_line(gamma_line) {
                    level.gammas.push(gamma);
                }
            }

            levels.push(level);
        }

        let nuclide = Nuclide::new(header.z, header.a)?;
        result.push(IsotopeData {
            nuclide,
            n_levels: header.n_levels,
            n_max: header.n_max,
            n_unique: header.n_unique,
            sn: header.sn,
            sp: header.sp,
            levels,
        });
    }

    Ok(result)
}

struct HeaderData {
    a: u16,
    z: u16,
    n_levels: usize,
    n_max: usize,
    n_unique: usize,
    sn: Option<f64>,
    sp: Option<f64>,
}

/// Parse header line: FORMAT (a5, 6i5, 2f12.6)
/// Columns: SYMB(1-5) A(6-10) Z(11-15) Nol(16-20) Nog(21-25) Nmax(26-30) Nc(31-35) Sn(36-47) Sp(48-59)
fn parse_header(line: &str, _line_num: usize) -> Option<HeaderData> {
    let a = fixed_field_u16(line, 5, 10)?;
    let z = fixed_field_u16(line, 10, 15)?;
    let n_levels = fixed_field_u32(line, 15, 20).unwrap_or(0) as usize;
    let _n_gammas = fixed_field_u32(line, 20, 25).unwrap_or(0);
    let n_max = fixed_field_u32(line, 25, 30).unwrap_or(0) as usize;
    let n_unique = fixed_field_u32(line, 30, 35).unwrap_or(0) as usize;
    let sn = fixed_field_f64_opt(line, 35, 47);
    let sp = fixed_field_f64_opt(line, 47, 59);

    Some(HeaderData {
        a,
        z,
        n_levels,
        n_max,
        n_unique,
        sn,
        sp,
    })
}

/// Parse level line: FORMAT (i3, 1x, f10.6, 1x, f5.1, i3, 1x, e10.3, i3, ...)
/// Columns: Nl(1-3) Elv(5-14) spin(16-20) parity(21-23) T1/2(25-34) Ng(35-37)
fn parse_level_line(line: &str) -> Option<DiscreteLevel> {
    let index = fixed_field_u32(line, 0, 3)?;
    let energy = fixed_field_f64(line, 4, 14)?;
    let spin = fixed_field_f64_opt(line, 15, 20);
    let parity_raw = fixed_field_i32(line, 20, 23);
    let parity = parity_raw.and_then(|p| match p {
        1 => Some(Parity::Positive),
        -1 => Some(Parity::Negative),
        _ => None, // 0 or other = unknown
    });

    // Half-life: e10.3 format. -1.0 means stable/unknown.
    let half_life = fixed_field_f64_opt(line, 24, 34);

    let n_gammas = fixed_field_u32(line, 34, 37).unwrap_or(0);

    Some(DiscreteLevel {
        index,
        energy,
        spin,
        parity,
        half_life,
        n_gammas,
        gammas: Vec::with_capacity(n_gammas as usize),
    })
}

/// Parse gamma transition line: FORMAT (39x, i4, 1x, f10.4, 3(1x, e10.3))
/// Starting at column 39: Nf(40-43) Eg(45-54) Pg(56-65) Pe(67-76) ICC(78-87)
fn parse_gamma_line(line: &str) -> Option<GammaTransition> {
    let final_level = fixed_field_u32(line, 39, 43)?;
    let energy = fixed_field_f64(line, 44, 54)?;
    let branching_ratio = fixed_field_f64(line, 55, 65).unwrap_or(0.0);
    let total_transition_prob = fixed_field_f64(line, 66, 76);
    let icc = fixed_field_f64(line, 77, 87);

    Some(GammaTransition {
        final_level,
        energy,
        branching_ratio,
        total_transition_prob,
        icc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal RIPL-3 discrete levels test data (Fe-56 style).
    /// Header: FORMAT (a5, 6i5, 2f12.6)
    ///   SYMB(1-5) A(6-10) Z(11-15) Nol(16-20) Nog(21-25) Nmax(26-30) Nc(31-35) Sn(36-47) Sp(48-59)
    /// Level: FORMAT (i3, 1x, f10.6, 1x, f5.1, i3, 1x, e10.3, i3, ...)
    ///   Nl(1-3) _Elv(5-14) _spin(16-20) parity(21-23) _T1/2(25-34) Ng(35-37)
    /// Gamma: FORMAT (39x, i4, 1x, f10.4, 3(1x, e10.3))
    ///   39 spaces then: Nf(40-43) _Eg(45-54) _Pg(56-65) _Pe(67-76) _ICC(78-87)
    fn sample_fe56() -> String {
        // Header: a5="56Fe " a=56 z=26 nol=3 nog=2 nmax=3 nc=2 sn=11.1974 sp=8.8841
        let header = format!(
            "{:<5}{:5}{:5}{:5}{:5}{:5}{:5}{:12.6}{:12.6}",
            "56Fe", 56, 26, 3, 2, 3, 2, 11.197400, 8.884100
        );
        // Level 1: ground state 0+, stable, 1 gamma
        let lev1 = format!(
            "{:3} {:10.6} {:5.1}{:3} {:10.3E}{:3}",
            1, 0.0, 0.0, 1, -1.0_f64, 1
        );
        // Gamma from level 1 -> level 2
        let gam1 = format!(
            "{:39}{:4} {:10.4} {:10.3E} {:10.3E} {:10.3E}",
            "", 2, 0.8468, 1.0_f64, 1.0_f64, 0.0_f64
        );
        // Level 2: 2+ at 0.8468 MeV, 1 gamma
        let lev2 = format!(
            "{:3} {:10.6} {:5.1}{:3} {:10.3E}{:3}",
            2, 0.846800, 2.0, 1, 8.3e-12_f64, 1
        );
        // Gamma from level 2 -> level 1
        let gam2 = format!(
            "{:39}{:4} {:10.4} {:10.3E} {:10.3E} {:10.3E}",
            "", 1, 0.8468, 1.0_f64, 1.0_f64, 0.0_f64
        );
        // Level 3: 4+ at 2.085 MeV, no gammas
        let lev3 = format!(
            "{:3} {:10.6} {:5.1}{:3} {:10.3E}{:3}",
            3, 2.085100, 4.0, 1, 5.3e-13_f64, 0
        );
        format!("{header}\n{lev1}\n{gam1}\n{lev2}\n{gam2}\n{lev3}\n")
    }

    #[test]
    fn parse_fe56_header() {
        let data = sample_fe56();
        let result = parse_discrete_levels(&data).unwrap();
        assert_eq!(result.len(), 1);
        let iso = &result[0];
        assert_eq!(iso.nuclide.z(), 26);
        assert_eq!(iso.nuclide.a(), 56);
        assert_eq!(iso.n_levels, 3);
        assert_eq!(iso.n_max, 3);
        assert_eq!(iso.n_unique, 2);
        assert!((iso.sn.unwrap() - 11.1974).abs() < 1e-3);
        assert!((iso.sp.unwrap() - 8.8841).abs() < 1e-3);
    }

    #[test]
    fn parse_fe56_levels() {
        let data = sample_fe56();
        let result = parse_discrete_levels(&data).unwrap();
        let levels = &result[0].levels;
        assert_eq!(levels.len(), 3, "levels: {:?}", levels);

        // Ground state
        assert_eq!(levels[0].index, 1);
        assert!((levels[0].energy - 0.0).abs() < 1e-6);
        assert_eq!(levels[0].spin, Some(0.0));
        assert_eq!(levels[0].parity, Some(Parity::Positive));
        assert_eq!(levels[0].half_life, None); // -1.0 → None (stable)

        // First excited state: 2+
        assert_eq!(levels[1].index, 2);
        assert!((levels[1].energy - 0.8468).abs() < 1e-4);
        assert_eq!(levels[1].spin, Some(2.0));
        assert_eq!(levels[1].parity, Some(Parity::Positive));
        assert!(levels[1].half_life.is_some());

        // Second excited: 4+, no gammas
        assert_eq!(levels[2].index, 3);
        assert_eq!(levels[2].spin, Some(4.0));
        assert_eq!(levels[2].n_gammas, 0);
        assert_eq!(levels[2].gammas.len(), 0);
    }

    #[test]
    fn parse_gamma_transitions() {
        let data = sample_fe56();
        let result = parse_discrete_levels(&data).unwrap();
        let levels = &result[0].levels;

        // Ground state has 1 gamma
        assert_eq!(levels[0].gammas.len(), 1);
        let g = &levels[0].gammas[0];
        assert_eq!(g.final_level, 2);
        assert!((g.energy - 0.8468).abs() < 1e-3);

        // First excited has 1 gamma
        assert_eq!(levels[1].gammas.len(), 1);
        assert_eq!(levels[1].gammas[0].final_level, 1);
    }

    #[test]
    fn parse_empty_input() {
        let result = parse_discrete_levels("").unwrap();
        assert!(result.is_empty());
    }
}
