use faer::prelude::*;
use nucrust_core::CoreError;

/// REACLIB rate type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateType {
    NonResonant,
    Resonant,
    Weak,
    Other,
}

/// A single REACLIB entry (3-line record in R1 format).
#[derive(Debug, Clone)]
pub struct ReaclibEntry {
    /// Chapter number (1-11): reaction topology.
    pub chapter: u8,
    /// Reactant nuclide names (up to 3).
    pub reactants: Vec<String>,
    /// Product nuclide names (up to 4).
    pub products: Vec<String>,
    /// Set label (4 characters).
    pub label: String,
    /// Rate type.
    pub rate_type: RateType,
    /// Whether this is a reverse rate from detailed balance.
    pub is_reverse: bool,
    /// Q-value (MeV).
    pub q_value: f64,
    /// 7 REACLIB coefficients a0..a6.
    pub coefficients: [f64; 7],
}

impl ReaclibEntry {
    /// Evaluate the REACLIB 7-parameter rate formula at temperature T9 (in GK).
    ///
    /// lambda = exp(a0 + a1/T9 + a2/T9^{1/3} + a3*T9^{1/3} + a4*T9 + a5*T9^{5/3} + a6*ln(T9))
    pub fn evaluate(&self, t9: f64) -> f64 {
        if t9 <= 0.0 {
            return 0.0;
        }
        let a = &self.coefficients;
        let t9_13 = t9.cbrt();
        let t9_53 = t9_13.powi(5);
        let exponent = a[0]
            + a[1] / t9
            + a[2] / t9_13
            + a[3] * t9_13
            + a[4] * t9
            + a[5] * t9_53
            + a[6] * t9.ln();
        exponent.exp()
    }
}

/// Parse REACLIB R1 format file.
///
/// Each entry consists of 3 lines:
/// - Line 1: chapter(1) + 4x + 6 nuclides(5 each) + 8x + label(4) + flag(1) + reverse(1) + 3x + Q(12)
/// - Line 2: a0..a3 (4 x E13.6)
/// - Line 3: a4..a6 (3 x E13.6)
pub fn parse_reaclib(input: &str) -> Result<Vec<ReaclibEntry>, CoreError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut entries = Vec::new();
    let mut i = 0;

    while i + 2 < lines.len() {
        let line1 = lines[i];
        let line2 = lines[i + 1];
        let line3 = lines[i + 2];

        // Skip empty lines or comment-like lines
        if line1.trim().is_empty() || line1.len() < 54 {
            i += 1;
            continue;
        }

        let entry = match parse_reaclib_entry(line1, line2, line3) {
            Some(e) => e,
            None => {
                i += 1;
                continue;
            }
        };

        entries.push(entry);
        i += 3;
    }

    Ok(entries)
}

fn parse_reaclib_entry(line1: &str, line2: &str, line3: &str) -> Option<ReaclibEntry> {
    // Line 1: chapter(col 0, 1 char) + 4x + 6 nuclides(col 5-34, each 5 chars)
    //         + 8x + label(col 43-46, 4 chars) + flag(col 47) + reverse(col 48)
    //         + 3x + Q-value(col 52-63, 12 chars)
    let chapter: u8 = line1.get(0..1)?.trim().parse().ok()?;

    let mut nuclides = Vec::new();
    for k in 0..6 {
        let start = 5 + k * 5;
        let name = line1.get(start..start + 5)?.trim();
        if !name.is_empty() {
            nuclides.push(name.to_string());
        }
    }

    let label = line1.get(43..47).unwrap_or("    ").trim().to_string();
    let flag_char = line1.get(47..48).unwrap_or(" ").trim();
    let reverse_char = line1.get(48..49).unwrap_or(" ").trim();

    let rate_type = match flag_char {
        "n" => RateType::NonResonant,
        "r" => RateType::Resonant,
        "w" => RateType::Weak,
        _ => RateType::Other,
    };
    let is_reverse = reverse_char == "v";

    let q_value: f64 = line1
        .get(52..64)
        .unwrap_or("0.0")
        .trim()
        .parse()
        .unwrap_or(0.0);

    // Determine reactants/products based on chapter
    let (n_reactants, n_products) = chapter_topology(chapter);
    let reactants: Vec<String> = nuclides.iter().take(n_reactants).cloned().collect();
    let products: Vec<String> = nuclides
        .iter()
        .skip(n_reactants)
        .take(n_products)
        .cloned()
        .collect();

    // Line 2: 4 coefficients in E13.6 format
    let mut coefficients = [0.0; 7];
    for (k, coeff) in coefficients.iter_mut().enumerate().take(4) {
        let start = k * 13;
        *coeff = line2
            .get(start..start + 13)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
    }
    // Line 3: 3 coefficients
    for k in 0..3 {
        let start = k * 13;
        coefficients[4 + k] = line3
            .get(start..start + 13)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
    }

    Some(ReaclibEntry {
        chapter,
        reactants,
        products,
        label,
        rate_type,
        is_reverse,
        q_value,
        coefficients,
    })
}

/// Map chapter number to (n_reactants, n_products).
fn chapter_topology(chapter: u8) -> (usize, usize) {
    match chapter {
        1 => (1, 1),  // e.g., decay
        2 => (1, 2),  // e.g., photodisintegration
        3 => (1, 3),  // e.g., 3-body decay
        4 => (2, 1),  // e.g., capture
        5 => (2, 2),  // e.g., (n,p)
        6 => (2, 3),  // e.g., (n,2p)
        7 => (2, 4),  // e.g., (n,alpha+p)
        8 => (3, 1),  // 3-body capture
        9 => (3, 2),  //
        10 => (1, 4), //
        11 => (4, 2), //
        _ => (2, 2),  // default
    }
}

/// Fit REACLIB 7-parameter coefficients to reaction rate data.
///
/// Performs a linear least-squares fit in log-space:
///   minimize sum_i [ln(rate_i) - sum_k a_k * phi_k(T9_i)]^2
///
/// Basis functions phi_k(T9) = {1, 1/T9, T9^{-1/3}, T9^{1/3}, T9, T9^{5/3}, ln(T9)}
///
/// Uses faer QR decomposition for numerical stability.
pub fn fit_reaclib_params(temperatures: &[f64], rates: &[f64]) -> Result<[f64; 7], CoreError> {
    let n = temperatures.len();
    if n < 7 {
        return Err(CoreError::InvalidParameter {
            name: "temperatures",
            value: n as f64,
            reason: "need at least 7 data points for 7-parameter fit",
        });
    }
    if n != rates.len() {
        return Err(CoreError::InvalidParameter {
            name: "rates",
            value: rates.len() as f64,
            reason: "temperatures and rates must have the same length",
        });
    }

    // Build design matrix A (n x 7) and RHS vector b (n x 1)
    let mut a_mat = faer::Mat::<f64>::zeros(n, 7);
    let mut b_vec = faer::Mat::<f64>::zeros(n, 1);

    for i in 0..n {
        let t9 = temperatures[i];
        if t9 <= 0.0 || rates[i] <= 0.0 {
            return Err(CoreError::InvalidParameter {
                name: "temperature/rate",
                value: t9,
                reason: "temperatures and rates must be positive for log-space fit",
            });
        }

        let t9_13 = t9.cbrt();
        let t9_53 = t9_13.powi(5);

        // phi_k(T9) = {1, 1/T9, T9^{-1/3}, T9^{1/3}, T9, T9^{5/3}, ln(T9)}
        a_mat[(i, 0)] = 1.0;
        a_mat[(i, 1)] = 1.0 / t9;
        a_mat[(i, 2)] = 1.0 / t9_13;
        a_mat[(i, 3)] = t9_13;
        a_mat[(i, 4)] = t9;
        a_mat[(i, 5)] = t9_53;
        a_mat[(i, 6)] = t9.ln();

        b_vec[(i, 0)] = rates[i].ln();
    }

    // Solve least-squares via QR decomposition: min ||A*x - b||
    let qr = a_mat.col_piv_qr();
    let x = qr.solve_lstsq(&b_vec);

    let mut coefficients = [0.0; 7];
    for k in 0..7 {
        coefficients[k] = x[(k, 0)];
    }

    Ok(coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_reaclib() -> String {
        // Chapter 5 (2 reactants, 2 products): n + fe56 -> fe57 + gamma
        // Line 1: ch(1) + 4x + nuclides(6x5) + 8x + label(4) + flag(1) + rev(1) + 3x + Q(12)
        let line1 = format!(
            "{}    {:<5}{:<5}{:<5}{:<5}{:<5}{:<5}        {:<4}{}{}{:>15.5E}",
            5, "n", "fe56", "fe57", "g", "", "", "ec  ", "n", " ", 7.64600e+00
        );
        let line2 = " 1.81200e+01 0.00000e+00-2.34400e+01-1.22200e+00";
        let line3 = " 1.47600e-01-1.11600e-02-6.67000e-01";
        format!("{line1}\n{line2}\n{line3}\n")
    }

    #[test]
    fn parse_single_entry() {
        let data = sample_reaclib();
        let entries = parse_reaclib(&data).unwrap();
        assert_eq!(entries.len(), 1);

        let e = &entries[0];
        assert_eq!(e.chapter, 5);
        assert_eq!(e.reactants.len(), 2);
        assert_eq!(e.reactants[0], "n");
        assert_eq!(e.reactants[1], "fe56");
        assert!(e.products.len() >= 1);
        assert_eq!(e.products[0], "fe57");
        assert!(!e.is_reverse);
    }

    #[test]
    fn evaluate_rate() {
        let entry = ReaclibEntry {
            chapter: 4,
            reactants: vec!["n".into(), "fe56".into()],
            products: vec!["fe57".into()],
            label: "test".into(),
            rate_type: RateType::NonResonant,
            is_reverse: false,
            q_value: 7.646,
            coefficients: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        // a0=1, rest 0: lambda = exp(1) = e
        let rate = entry.evaluate(1.0);
        assert!((rate - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn evaluate_rate_temperature_dependence() {
        let entry = ReaclibEntry {
            chapter: 4,
            reactants: vec!["n".into()],
            products: vec!["p".into()],
            label: "test".into(),
            rate_type: RateType::Weak,
            is_reverse: false,
            q_value: 0.0,
            coefficients: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
        };
        // a6=1.5: lambda = exp(1.5 * ln(T9)) = T9^1.5
        let rate = entry.evaluate(4.0);
        let expected = 4.0_f64.powf(1.5);
        assert!(
            (rate - expected).abs() < 1e-10,
            "rate = {}, expected {}",
            rate,
            expected
        );
    }

    #[test]
    fn evaluate_zero_temperature() {
        let entry = ReaclibEntry {
            chapter: 1,
            reactants: vec![],
            products: vec![],
            label: String::new(),
            rate_type: RateType::Other,
            is_reverse: false,
            q_value: 0.0,
            coefficients: [1.0; 7],
        };
        assert_eq!(entry.evaluate(0.0), 0.0);
        assert_eq!(entry.evaluate(-1.0), 0.0);
    }

    #[test]
    fn fit_known_coefficients() {
        // Generate synthetic data from known coefficients
        let known = [18.12, 0.0, -23.44, -1.222, 0.1476, -0.01116, -0.667];
        let temperatures: Vec<f64> = (1..=20).map(|i| 0.1 * i as f64).collect();
        let rates: Vec<f64> = temperatures
            .iter()
            .map(|&t9| {
                let t9_13 = t9.cbrt();
                let exponent = known[0]
                    + known[1] / t9
                    + known[2] / t9_13
                    + known[3] * t9_13
                    + known[4] * t9
                    + known[5] * t9_13.powi(5)
                    + known[6] * t9.ln();
                exponent.exp()
            })
            .collect();

        let fitted = fit_reaclib_params(&temperatures, &rates).unwrap();

        for k in 0..7 {
            assert!(
                (fitted[k] - known[k]).abs() < 1e-6,
                "coefficient a[{}]: fitted={}, known={}",
                k,
                fitted[k],
                known[k]
            );
        }
    }

    #[test]
    fn fit_too_few_points() {
        let temps = vec![1.0, 2.0];
        let rates = vec![1.0, 2.0];
        assert!(fit_reaclib_params(&temps, &rates).is_err());
    }
}
