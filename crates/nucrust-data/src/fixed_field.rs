//! Fortran fixed-width field parser utilities.
//!
//! RIPL-3 data files use Fortran FORMAT strings that define exact column positions.
//! These utilities parse fields by byte-range slicing + trim + type conversion.

/// Parse a fixed-width integer field. Returns None for empty/unparseable fields.
pub fn fixed_field_i32(line: &str, start: usize, end: usize) -> Option<i32> {
    line.get(start..end)?.trim().parse().ok()
}

/// Parse a fixed-width u16 field.
pub fn fixed_field_u16(line: &str, start: usize, end: usize) -> Option<u16> {
    line.get(start..end)?.trim().parse().ok()
}

/// Parse a fixed-width u32 field.
pub fn fixed_field_u32(line: &str, start: usize, end: usize) -> Option<u32> {
    line.get(start..end)?.trim().parse().ok()
}

/// Parse a fixed-width f64 field. Returns None for empty, "-1.0", or unparseable fields.
pub fn fixed_field_f64(line: &str, start: usize, end: usize) -> Option<f64> {
    let s = line.get(start..end)?.trim();
    if s.is_empty() {
        return None;
    }
    s.parse().ok()
}

/// Parse a fixed-width f64 field, treating -1.0 as missing.
pub fn fixed_field_f64_opt(line: &str, start: usize, end: usize) -> Option<f64> {
    let v = fixed_field_f64(line, start, end)?;
    if v == -1.0 {
        None
    } else {
        Some(v)
    }
}

/// Extract a trimmed string slice from a fixed-width field.
pub fn fixed_field_str(line: &str, start: usize, end: usize) -> &str {
    line.get(start..end).unwrap_or("").trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_integer_fields() {
        //           01234567890123
        let line = "   26   56   30";
        assert_eq!(fixed_field_i32(line, 0, 5), Some(26));
        assert_eq!(fixed_field_i32(line, 5, 10), Some(56));
        assert_eq!(fixed_field_i32(line, 10, 15), Some(30));
    }

    #[test]
    fn parse_float_fields() {
        let line = "    8.071323   11.197000";
        assert_eq!(fixed_field_f64(line, 0, 12), Some(8.071323));
        assert_eq!(fixed_field_f64(line, 12, 24), Some(11.197));
    }

    #[test]
    fn parse_missing_values() {
        assert_eq!(fixed_field_f64("     ", 0, 5), None);
        assert_eq!(fixed_field_f64_opt("  -1.0    ", 0, 10), None);
        assert_eq!(fixed_field_i32("     ", 0, 5), None);
    }

    #[test]
    fn parse_scientific_notation() {
        let line = " 1.234E+03";
        assert_eq!(fixed_field_f64(line, 0, 10), Some(1234.0));

        let line = " 3.500E-02";
        let v = fixed_field_f64(line, 0, 10).unwrap();
        assert!((v - 0.035).abs() < 1e-15);
    }

    #[test]
    fn extract_string_field() {
        let line = " 56Fe   ";
        assert_eq!(fixed_field_str(line, 0, 5), "56Fe");
        assert_eq!(fixed_field_str(line, 5, 8), "");
    }

    #[test]
    fn short_line_returns_none() {
        let line = "abc";
        assert_eq!(fixed_field_i32(line, 5, 10), None);
        assert_eq!(fixed_field_f64(line, 5, 10), None);
    }
}
