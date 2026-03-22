use nucrust_core::CoreError;

/// Errors specific to special function computations.
#[derive(Debug, thiserror::Error)]
pub enum SpecialError {
    /// An iterative algorithm failed to converge within the allowed iterations.
    #[error("Convergence failure in {algorithm} after {iterations} iterations (residual: {residual:.2e})")]
    ConvergenceFailure {
        /// Name of the algorithm that failed.
        algorithm: &'static str,
        /// Number of iterations performed before giving up.
        iterations: u32,
        /// Residual magnitude at termination.
        residual: f64,
    },

    /// A numerical overflow or underflow occurred during computation.
    #[error("Numerical overflow/underflow in {context}")]
    NumericalOverflow {
        /// Description of the computation context where overflow occurred.
        context: &'static str,
    },

    /// An input argument was outside its valid domain.
    #[error("Invalid argument: {name} = {value} ({reason})")]
    InvalidArgument {
        /// Name of the invalid parameter.
        name: &'static str,
        /// The invalid value that was provided.
        value: f64,
        /// Explanation of why the value is invalid.
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
