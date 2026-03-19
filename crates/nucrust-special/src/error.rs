use nucrust_core::CoreError;

/// Errors specific to special function computations.
#[derive(Debug, thiserror::Error)]
pub enum SpecialError {
    #[error("Convergence failure in {algorithm} after {iterations} iterations (residual: {residual:.2e})")]
    ConvergenceFailure {
        algorithm: &'static str,
        iterations: u32,
        residual: f64,
    },

    #[error("Numerical overflow/underflow in {context}")]
    NumericalOverflow { context: &'static str },

    #[error("Invalid argument: {name} = {value} ({reason})")]
    InvalidArgument {
        name: &'static str,
        value: f64,
        reason: &'static str,
    },
}

impl From<SpecialError> for CoreError {
    fn from(e: SpecialError) -> Self {
        match e {
            SpecialError::ConvergenceFailure {
                algorithm,
                iterations,
                residual,
            } => CoreError::ConvergenceFailure {
                algorithm,
                iterations,
                residual,
            },
            SpecialError::NumericalOverflow { context } => CoreError::NumericalOverflow { context },
            SpecialError::InvalidArgument {
                name,
                value,
                reason,
            } => CoreError::InvalidParameter {
                name,
                value,
                reason,
            },
        }
    }
}
