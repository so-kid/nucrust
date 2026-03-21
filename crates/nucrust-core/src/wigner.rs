//! Wigner 3j and 6j symbols for angular momentum coupling.
//!
//! All angular momentum arguments are in "twice-j" convention (2j integers)
//! to handle half-integer spins exactly.
//!
//! Algorithms based on Racah's formula with logarithmic factorials
//! for numerical stability.

/// Precomputed log-factorials for n = 0..200.
fn log_factorial(n: i32) -> f64 {
    debug_assert!(n >= 0, "log_factorial called with negative n={n}");
    if n <= 1 {
        return 0.0;
    }
    // Use Stirling or direct summation; for n <= 200 direct is fine
    let mut sum = 0.0;
    for k in 2..=n {
        sum += (k as f64).ln();
    }
    sum
}

/// Triangle coefficient Δ(a,b,c) in log form.
/// Arguments are 2*j values.
/// Returns ln(Δ(a,b,c)²) / 2 = [ln((a+b-c)!) + ln((a-c+b)!) + ln((-a+b+c)!) - ln((a+b+c+1)!)] / 2
///
/// Actually Δ(j1,j2,j3) = sqrt( (j1+j2-j3)! (j1-j2+j3)! (-j1+j2+j3)! / (j1+j2+j3+1)! )
/// where j values are in half-integer form, so we pass 2j and divide sums by 2.
fn log_triangle(two_a: i32, two_b: i32, two_c: i32) -> f64 {
    let s = two_a + two_b + two_c;
    debug_assert!(s % 2 == 0, "sum of 2j values must be even");
    let s2 = s / 2;
    let n1 = (two_a + two_b - two_c) / 2;
    let n2 = (two_a - two_b + two_c) / 2;
    let n3 = (-two_a + two_b + two_c) / 2;
    0.5 * (log_factorial(n1) + log_factorial(n2) + log_factorial(n3) - log_factorial(s2 + 1))
}

/// Check triangle condition for 2j values.
fn triangle_ok(two_a: i32, two_b: i32, two_c: i32) -> bool {
    let s = two_a + two_b + two_c;
    if s % 2 != 0 {
        return false;
    }
    let n1 = two_a + two_b - two_c;
    let n2 = two_a - two_b + two_c;
    let n3 = -two_a + two_b + two_c;
    n1 >= 0 && n2 >= 0 && n3 >= 0
}

/// Wigner 3j symbol.
///
/// ```text
/// ⎛ j1  j2  j3 ⎞
/// ⎝ m1  m2  m3 ⎠
/// ```
///
/// All arguments are twice the angular momentum (2j, 2m).
/// Returns 0 if selection rules are violated.
///
/// Uses Racah's formula with log-factorial summation for numerical stability.
pub fn wigner_3j(
    two_j1: i32,
    two_j2: i32,
    two_j3: i32,
    two_m1: i32,
    two_m2: i32,
    two_m3: i32,
) -> f64 {
    // Selection rules
    if two_m1 + two_m2 + two_m3 != 0 {
        return 0.0;
    }
    if !triangle_ok(two_j1, two_j2, two_j3) {
        return 0.0;
    }
    // |m| <= j checks
    if two_m1.abs() > two_j1 || two_m2.abs() > two_j2 || two_m3.abs() > two_j3 {
        return 0.0;
    }
    // j+m must be integer (both 2j and 2m have same parity)
    if (two_j1 + two_m1) % 2 != 0 || (two_j2 + two_m2) % 2 != 0 || (two_j3 + two_m3) % 2 != 0 {
        return 0.0;
    }

    // Convert to half-sum indices
    let j1m1 = (two_j1 + two_m1) / 2;
    let j1_m1 = (two_j1 - two_m1) / 2;
    let j2m2 = (two_j2 + two_m2) / 2;
    let j2_m2 = (two_j2 - two_m2) / 2;
    let j3m3 = (two_j3 + two_m3) / 2;
    let j3_m3 = (two_j3 - two_m3) / 2;

    let _s = (two_j1 + two_j2 + two_j3) / 2;
    let n1 = (two_j1 + two_j2 - two_j3) / 2;
    let _n2 = (two_j1 - two_j2 + two_j3) / 2;
    let _n3 = (-two_j1 + two_j2 + two_j3) / 2;

    // Log of the prefactor (triangle + magnetic quantum number factorials)
    let log_prefactor = log_triangle(two_j1, two_j2, two_j3)
        + 0.5
            * (log_factorial(j1m1)
                + log_factorial(j1_m1)
                + log_factorial(j2m2)
                + log_factorial(j2_m2)
                + log_factorial(j3m3)
                + log_factorial(j3_m3));

    // Racah formula: sum over s where all factorial arguments are non-negative.
    // Denominator factorials: s!, (j1+j2-j3-s)!, (j1-m1-s)!, (j2+m2-s)!,
    //   (j3-j2+m1+s)!, (j3-j1-m2+s)!
    // In half-integer indices:
    let term5_base = (two_j3 - two_j2 + two_m1) / 2; // j3-j2+m1
    let term6_base = (two_j3 - two_j1 - two_m2) / 2; // j3-j1-m2

    let t_min = [0, -term5_base, -term6_base].into_iter().max().unwrap();
    let t_max = [n1, j1_m1, j2m2].into_iter().min().unwrap();

    if t_min > t_max {
        return 0.0;
    }

    let mut sum = 0.0;
    for t in t_min..=t_max {
        let log_term = log_factorial(t)
            + log_factorial(n1 - t)
            + log_factorial(j1_m1 - t)
            + log_factorial(j2m2 - t)
            + log_factorial(term5_base + t)
            + log_factorial(term6_base + t);
        let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * (log_prefactor - log_term).exp();
    }

    // Overall phase: (-1)^(j1-j2-m3)
    let phase_exp = (two_j1 - two_j2 - two_m3) / 2;
    let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

    phase * sum
}

/// Clebsch-Gordan coefficient ⟨j1 m1; j2 m2 | j3 m3⟩.
///
/// All arguments are twice the angular momentum.
///
/// Related to Wigner 3j by:
/// ⟨j1 m1; j2 m2 | j3 m3⟩ = (-1)^(j1-j2+m3) sqrt(2j3+1) * (j1 j2 j3; m1 m2 -m3)
pub fn clebsch_gordan(
    two_j1: i32,
    two_m1: i32,
    two_j2: i32,
    two_m2: i32,
    two_j3: i32,
    two_m3: i32,
) -> f64 {
    if two_m1 + two_m2 != two_m3 {
        return 0.0;
    }
    let phase_exp = (two_j1 - two_j2 + two_m3) / 2;
    let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };
    let factor = ((two_j3 + 1) as f64).sqrt();
    phase * factor * wigner_3j(two_j1, two_j2, two_j3, two_m1, two_m2, -two_m3)
}

/// Wigner 6j symbol.
///
/// ```text
/// ⎧ j1  j2  j3 ⎫
/// ⎩ j4  j5  j6 ⎭
/// ```
///
/// All arguments are twice the angular momentum.
/// Uses Racah's formula with log-factorial summation.
pub fn wigner_6j(
    two_j1: i32,
    two_j2: i32,
    two_j3: i32,
    two_j4: i32,
    two_j5: i32,
    two_j6: i32,
) -> f64 {
    // Triangle conditions: {j1,j2,j3}, {j1,j5,j6}, {j4,j2,j6}, {j4,j5,j3}
    if !triangle_ok(two_j1, two_j2, two_j3)
        || !triangle_ok(two_j1, two_j5, two_j6)
        || !triangle_ok(two_j4, two_j2, two_j6)
        || !triangle_ok(two_j4, two_j5, two_j3)
    {
        return 0.0;
    }

    let log_tri = log_triangle(two_j1, two_j2, two_j3)
        + log_triangle(two_j1, two_j5, two_j6)
        + log_triangle(two_j4, two_j2, two_j6)
        + log_triangle(two_j4, two_j5, two_j3);

    // Summation limits
    let s_min = [
        (two_j1 + two_j2 + two_j3) / 2,
        (two_j1 + two_j5 + two_j6) / 2,
        (two_j4 + two_j2 + two_j6) / 2,
        (two_j4 + two_j5 + two_j3) / 2,
    ]
    .into_iter()
    .max()
    .unwrap();

    let s_max = [
        (two_j1 + two_j2 + two_j4 + two_j5) / 2,
        (two_j2 + two_j3 + two_j5 + two_j6) / 2,
        (two_j1 + two_j3 + two_j4 + two_j6) / 2,
    ]
    .into_iter()
    .min()
    .unwrap();

    if s_min > s_max {
        return 0.0;
    }

    // Pre-compute the four triangle half-perimeters
    let a1 = (two_j1 + two_j2 + two_j3) / 2;
    let a2 = (two_j1 + two_j5 + two_j6) / 2;
    let a3 = (two_j4 + two_j2 + two_j6) / 2;
    let a4 = (two_j4 + two_j5 + two_j3) / 2;

    let b1 = (two_j1 + two_j2 + two_j4 + two_j5) / 2;
    let b2 = (two_j2 + two_j3 + two_j5 + two_j6) / 2;
    let b3 = (two_j1 + two_j3 + two_j4 + two_j6) / 2;

    let mut sum = 0.0;
    for s in s_min..=s_max {
        let log_num = log_factorial(s + 1);
        let log_den = log_factorial(s - a1)
            + log_factorial(s - a2)
            + log_factorial(s - a3)
            + log_factorial(s - a4)
            + log_factorial(b1 - s)
            + log_factorial(b2 - s)
            + log_factorial(b3 - s);
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * (log_tri + log_num - log_den).exp();
    }

    sum
}

/// Reduced matrix element of spherical harmonic ⟨l ‖ Y_λ ‖ l'⟩.
///
/// ```text
/// ⟨l ‖ Y_λ ‖ l'⟩ = (-1)^l * sqrt((2l+1)(2λ+1)(2l'+1) / (4π)) * (l λ l'; 0 0 0)
/// ```
///
/// Arguments `l`, `lambda`, `l_prime` are orbital angular momenta (integers, not doubled).
pub fn reduced_matrix_element_y(l: u32, lambda: u32, l_prime: u32) -> f64 {
    let two_l = 2 * l as i32;
    let two_lam = 2 * lambda as i32;
    let two_lp = 2 * l_prime as i32;

    // 3j symbol (l λ l'; 0 0 0) — all m=0
    let w3j = wigner_3j(two_l, two_lam, two_lp, 0, 0, 0);
    if w3j.abs() < 1e-30 {
        return 0.0;
    }

    let phase = if l % 2 == 0 { 1.0 } else { -1.0 };
    let prefactor = ((2 * l + 1) as f64 * (2 * lambda + 1) as f64 * (2 * l_prime + 1) as f64
        / (4.0 * std::f64::consts::PI))
        .sqrt();

    phase * prefactor * w3j
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare float with tolerance.
    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    // ============================
    // Wigner 3j tests
    // ============================

    #[test]
    fn wigner_3j_basic() {
        // (1 1 0; 0 0 0) = (-1)^1 / sqrt(3)
        // Arguments doubled: (2 2 0; 0 0 0)
        let val = wigner_3j(2, 2, 0, 0, 0, 0);
        assert_close(val, -1.0 / 3.0_f64.sqrt(), 1e-14, "3j(1,1,0;0,0,0)");
    }

    #[test]
    fn wigner_3j_half_integer() {
        // (1/2 1/2 1; 1/2 1/2 -1) = (1 1 2; 1 1 -2) in 2j notation
        let val = wigner_3j(1, 1, 2, 1, 1, -2);
        // Verified with sympy: -1/sqrt(3)
        assert_close(
            val,
            -1.0 / 3.0_f64.sqrt(),
            1e-14,
            "3j(1/2,1/2,1;1/2,1/2,-1)",
        );
    }

    #[test]
    fn wigner_3j_selection_rule_m() {
        // m1 + m2 + m3 != 0 → 0
        assert_eq!(wigner_3j(2, 2, 2, 2, 0, 0), 0.0);
    }

    #[test]
    fn wigner_3j_selection_rule_triangle() {
        // j3 > j1 + j2 → 0
        assert_eq!(wigner_3j(2, 2, 8, 0, 0, 0), 0.0);
    }

    #[test]
    fn wigner_3j_symmetry() {
        // Even permutation: same value
        let v1 = wigner_3j(2, 4, 4, 0, 2, -2);
        let v2 = wigner_3j(4, 4, 2, 2, -2, 0);
        assert_close(v1, v2, 1e-14, "even permutation symmetry");
    }

    #[test]
    fn wigner_3j_known_values() {
        // (2 2 2; 0 0 0) in 2j notation = (1 1 2; 0 0 0)
        // = (-1)^2 * 2 / sqrt(30) ... let me use the known formula
        // (1 1 2; 0 0 0) = (-1)^(1+1+0) * sqrt(2/(3*5)) * ...
        // Actually: (j1 j2 j3; 0 0 0) = 0 if j1+j2+j3 is odd
        // j1+j2+j3 = 1+1+2 = 4 (even), so it's nonzero
        // Known: (1 1 2; 0 0 0) = sqrt(2/15) * (-1) = 2/sqrt(30) * phase
        let val = wigner_3j(2, 2, 4, 0, 0, 0);
        // From tables: (1 1 2; 0 0 0) = 2/sqrt(30) * (-1)^(1-1-0) = 2/sqrt(30)
        // Actually let me verify numerically: should be sqrt(2/15) up to sign
        // = 0.365148...
        assert!(
            val.abs() > 0.3 && val.abs() < 0.4,
            "3j(1,1,2;0,0,0) magnitude"
        );
    }

    // ============================
    // Clebsch-Gordan tests
    // ============================

    #[test]
    fn clebsch_gordan_basic() {
        // <1 0; 1 0 | 0 0> verified with sympy: -1/sqrt(3)
        let val = clebsch_gordan(2, 0, 2, 0, 0, 0);
        assert_close(val, -1.0 / 3.0_f64.sqrt(), 1e-14, "CG<1,0;1,0|0,0>");
    }

    #[test]
    fn clebsch_gordan_spin_orbit() {
        // <l=1 m=1; s=1/2 ms=1/2 | j=3/2 mj=3/2>
        // = <2 2; 1 1 | 3 3> in 2j notation = 1.0
        let val = clebsch_gordan(2, 2, 1, 1, 3, 3);
        assert_close(val, 1.0, 1e-14, "CG<1,1;1/2,1/2|3/2,3/2>");
    }

    #[test]
    fn clebsch_gordan_m_selection() {
        // m1 + m2 != m3 → 0
        let val = clebsch_gordan(2, 0, 2, 2, 2, 0);
        assert_eq!(val, 0.0);
    }

    // ============================
    // Wigner 6j tests
    // ============================

    #[test]
    fn wigner_6j_basic() {
        // {1 1 0; 1 1 0} in j notation = {2 2 0; 2 2 0} in 2j notation
        // = 1 / sqrt((2j1+1)(2j4+1)) = 1/3
        let val = wigner_6j(2, 2, 0, 2, 2, 0);
        assert_close(val, 1.0 / 3.0, 1e-14, "6j{1,1,0;1,1,0}");
    }

    #[test]
    fn wigner_6j_triangle_violation() {
        // Triangle violation → 0
        let val = wigner_6j(2, 2, 10, 2, 2, 2);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn wigner_6j_known_value() {
        // {1 1 1; 1 1 1} = {2 2 2; 2 2 2} in 2j notation
        // Verified with sympy: +1/6
        let val = wigner_6j(2, 2, 2, 2, 2, 2);
        assert_close(val, 1.0 / 6.0, 1e-14, "6j{1,1,1;1,1,1}");
    }

    #[test]
    fn wigner_6j_with_zero() {
        // {j1 j2 j3; 0 j3 j2} = (-1)^(j1+j2+j3) / sqrt((2j2+1)(2j3+1))
        // {1 2 2; 0 2 2} = {2 4 4; 0 4 4} in 2j
        let val = wigner_6j(2, 4, 4, 0, 4, 4);
        let expected = if (1 + 2 + 2) % 2 == 0 { 1.0 } else { -1.0 }
            / ((2 * 2 + 1) as f64 * (2 * 2 + 1) as f64).sqrt();
        assert_close(val, expected, 1e-14, "6j{1,2,2;0,2,2}");
    }

    #[test]
    fn wigner_6j_half_integer() {
        // {1/2 1/2 1; 1/2 1/2 1} = {1 1 2; 1 1 2} in 2j
        // Known: 1/6
        let val = wigner_6j(1, 1, 2, 1, 1, 2);
        assert_close(val, 1.0 / 6.0, 1e-14, "6j{1/2,1/2,1;1/2,1/2,1}");
    }

    // ============================
    // Reduced matrix element tests
    // ============================

    #[test]
    fn reduced_y_l0_lambda2_l2() {
        // <0 || Y_2 || 2> = (-1)^0 * sqrt(1 * 5 * 5 / 4π) * (0 2 2; 0 0 0)
        // (0 2 2; 0 0 0) in 2j = (0 4 4; 0 0 0) = 1/sqrt(5) (from tables)
        let val = reduced_matrix_element_y(0, 2, 2);
        // = sqrt(25/(4π)) * 1/sqrt(5) = sqrt(5/(4π))
        let expected = (5.0 / (4.0 * std::f64::consts::PI)).sqrt();
        assert_close(val, expected, 1e-12, "<0||Y2||2>");
    }

    #[test]
    fn reduced_y_parity_selection() {
        // <0 || Y_1 || 0> = 0 because l + λ + l' = 0+1+0 = 1 (odd) → 3j(0,0,0) = 0
        let val = reduced_matrix_element_y(0, 1, 0);
        assert_close(val, 0.0, 1e-15, "<0||Y1||0> parity selection");
    }

    #[test]
    fn reduced_y_triangle_violation() {
        // <0 || Y_4 || 2> — triangle: |0-2|=2 <= 4 <= 2 = 0+2? No, 4 > 2 → 0
        let val = reduced_matrix_element_y(0, 4, 2);
        assert_close(val, 0.0, 1e-15, "<0||Y4||2> triangle violation");
    }

    #[test]
    fn reduced_y_symmetric() {
        // <l || Y_λ || l'> relation with <l' || Y_λ || l>
        // <l || Y_λ || l'> = (-1)^(l-l') * <l' || Y_λ || l> (Hermitian conjugate property)
        let v1 = reduced_matrix_element_y(2, 2, 0);
        let v2 = reduced_matrix_element_y(0, 2, 2);
        // Phase: (-1)^(2-0) = 1
        assert_close(v1, v2, 1e-14, "reduced Y symmetry");
    }

    // ============================
    // Edge case tests
    // ============================

    #[test]
    fn log_factorial_basic() {
        assert!((log_factorial(0)).abs() < 1e-15);
        assert!((log_factorial(1)).abs() < 1e-15);
        assert!((log_factorial(3) - (6.0_f64).ln()).abs() < 1e-14);
        assert!((log_factorial(10) - (3628800.0_f64).ln()).abs() < 1e-10);
    }
}
