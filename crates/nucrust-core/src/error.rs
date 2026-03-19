/// Crate-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Invalid nuclide: Z={z}, A={a}")]
    InvalidNuclide { z: u16, a: u16 },

    #[error("Energy out of range: {value} MeV (expected {min}..{max})")]
    EnergyOutOfRange { value: f64, min: f64, max: f64 },

    #[error("Invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        name: &'static str,
        value: f64,
        reason: &'static str,
    },

    #[error("Convergence failure in {algorithm} after {iterations} iterations (residual: {residual:.2e})")]
    ConvergenceFailure {
        algorithm: &'static str,
        iterations: u32,
        residual: f64,
    },

    #[error("Numerical overflow/underflow in {context}")]
    NumericalOverflow { context: &'static str },

    #[error("Data not found: {description}")]
    DataNotFound { description: String },

    #[error("Parse error in {file} at line {line}: {message}")]
    ParseError {
        file: String,
        line: usize,
        message: String,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
