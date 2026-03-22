/// Crate-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Invalid nuclide specification (Z or A out of range).
    #[error("Invalid nuclide: Z={z}, A={a}")]
    InvalidNuclide {
        /// Atomic number.
        z: u16,
        /// Mass number.
        a: u16,
    },

    /// Energy value outside the valid range.
    #[error("Energy out of range: {value} MeV (expected {min}..{max})")]
    EnergyOutOfRange {
        /// The out-of-range energy (MeV).
        value: f64,
        /// Lower bound of the valid range (MeV).
        min: f64,
        /// Upper bound of the valid range (MeV).
        max: f64,
    },

    /// A physics or configuration parameter has an invalid value.
    #[error("Invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// The invalid value.
        value: f64,
        /// Explanation of why the value is invalid.
        reason: &'static str,
    },

    /// An iterative algorithm failed to converge.
    #[error("Convergence failure in {algorithm} after {iterations} iterations (residual: {residual:.2e})")]
    ConvergenceFailure {
        /// Name of the algorithm that failed.
        algorithm: &'static str,
        /// Number of iterations performed.
        iterations: u32,
        /// Final residual norm.
        residual: f64,
    },

    /// Numerical overflow or underflow in a calculation.
    #[error("Numerical overflow/underflow in {context}")]
    NumericalOverflow {
        /// Description of the computation context.
        context: &'static str,
    },

    /// Required nuclear data was not found.
    #[error("Data not found: {description}")]
    DataNotFound {
        /// Description of the missing data.
        description: String,
    },

    /// Error parsing a nuclear data file.
    #[error("Parse error in {file} at line {line}: {message}")]
    ParseError {
        /// File path or name.
        file: String,
        /// Line number where the error occurred.
        line: usize,
        /// Description of the parse error.
        message: String,
    },

    /// I/O error from the standard library.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
