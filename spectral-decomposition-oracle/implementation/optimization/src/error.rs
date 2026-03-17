//! Optimization-specific error types.
//!
//! Provides [`OptError`] for all optimization operations, wrapping
//! the spectral-types [`OptimizationError`] and adding optimization-specific
//! variants for infeasibility, unboundedness, numerical issues, and convergence
//! failures.

use std::fmt;
use thiserror::Error;

/// Primary error type for the optimization crate.
#[derive(Debug, Error)]
pub enum OptError {
    /// The optimization problem has no feasible solution.
    #[error("Infeasible: {reason}")]
    Infeasible { reason: String },

    /// The optimization problem is unbounded (objective can go to -inf or +inf).
    #[error("Unbounded: {reason}")]
    Unbounded { reason: String },

    /// A numerical error occurred during computation.
    #[error("Numerical error: {context}")]
    NumericalError { context: String },

    /// The solver failed to converge within the iteration limit.
    #[error("Convergence failure after {iterations} iterations: {message}")]
    ConvergenceFailure { iterations: usize, message: String },

    /// The solver exceeded its time limit.
    #[error("Time limit exceeded: {elapsed:.2}s > {limit:.2}s")]
    TimeLimitExceeded { elapsed: f64, limit: f64 },

    /// The problem formulation is invalid.
    #[error("Invalid problem: {reason}")]
    InvalidProblem { reason: String },

    /// A generic solver error.
    #[error("Solver error: {message}")]
    SolverError { message: String },

    /// No problem has been loaded into the solver.
    #[error("No problem loaded")]
    NoProblem,

    /// Solver has not been run yet.
    #[error("Solver not run — call solve_lp() first")]
    SolverNotRun,

    /// Wrapped error from spectral-types.
    #[error("Spectral: {0}")]
    Spectral(#[from] spectral_types::error::OptimizationError),
}

/// Convenience result type for optimization operations.
pub type OptResult<T> = std::result::Result<T, OptError>;

impl OptError {
    pub fn infeasible(reason: impl Into<String>) -> Self {
        OptError::Infeasible {
            reason: reason.into(),
        }
    }

    pub fn unbounded(reason: impl Into<String>) -> Self {
        OptError::Unbounded {
            reason: reason.into(),
        }
    }

    pub fn numerical(context: impl Into<String>) -> Self {
        OptError::NumericalError {
            context: context.into(),
        }
    }

    pub fn convergence(iterations: usize, message: impl Into<String>) -> Self {
        OptError::ConvergenceFailure {
            iterations,
            message: message.into(),
        }
    }

    pub fn time_limit(elapsed: f64, limit: f64) -> Self {
        OptError::TimeLimitExceeded { elapsed, limit }
    }

    pub fn invalid_problem(reason: impl Into<String>) -> Self {
        OptError::InvalidProblem {
            reason: reason.into(),
        }
    }

    pub fn solver(message: impl Into<String>) -> Self {
        OptError::SolverError {
            message: message.into(),
        }
    }

    pub fn is_infeasible(&self) -> bool {
        matches!(self, OptError::Infeasible { .. })
    }

    pub fn is_unbounded(&self) -> bool {
        matches!(self, OptError::Unbounded { .. })
    }

    pub fn is_numerical(&self) -> bool {
        matches!(self, OptError::NumericalError { .. })
    }

    pub fn is_convergence_failure(&self) -> bool {
        matches!(self, OptError::ConvergenceFailure { .. })
    }

    pub fn is_time_limit(&self) -> bool {
        matches!(self, OptError::TimeLimitExceeded { .. })
    }

    pub fn severity(&self) -> ErrorSeverity {
        match self {
            OptError::Infeasible { .. } | OptError::Unbounded { .. } => ErrorSeverity::Expected,
            OptError::TimeLimitExceeded { .. } | OptError::ConvergenceFailure { .. } => {
                ErrorSeverity::Warning
            }
            OptError::NumericalError { .. } => ErrorSeverity::Serious,
            OptError::InvalidProblem { .. } | OptError::NoProblem | OptError::SolverNotRun => {
                ErrorSeverity::UserError
            }
            OptError::SolverError { .. } | OptError::Spectral(_) => ErrorSeverity::Internal,
        }
    }
}

/// Classification of error severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    Expected,
    Warning,
    Serious,
    UserError,
    Internal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Expected => write!(f, "EXPECTED"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Serious => write!(f, "SERIOUS"),
            ErrorSeverity::UserError => write!(f, "USER_ERROR"),
            ErrorSeverity::Internal => write!(f, "INTERNAL"),
        }
    }
}

/// Validate a condition, returning an error if not met.
pub fn ensure(condition: bool, error: impl FnOnce() -> OptError) -> OptResult<()> {
    if condition {
        Ok(())
    } else {
        Err(error())
    }
}

/// Ensure a value is finite.
pub fn ensure_finite(value: f64, name: &str) -> OptResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OptError::numerical(format!(
            "{} is not finite: {}",
            name, value
        )))
    }
}

/// Ensure a dimension is positive.
pub fn ensure_positive(value: usize, name: &str) -> OptResult<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(OptError::invalid_problem(format!(
            "{} must be positive, got {}",
            name, value
        )))
    }
}

/// Ensure two lengths match.
pub fn ensure_lengths_match(
    name_a: &str,
    len_a: usize,
    name_b: &str,
    len_b: usize,
) -> OptResult<()> {
    if len_a == len_b {
        Ok(())
    } else {
        Err(OptError::invalid_problem(format!(
            "Length mismatch: {} has {} elements, {} has {} elements",
            name_a, len_a, name_b, len_b
        )))
    }
}

/// Ensure a value is within bounds.
pub fn ensure_in_range(value: f64, lo: f64, hi: f64, name: &str) -> OptResult<()> {
    if value >= lo && value <= hi {
        Ok(())
    } else {
        Err(OptError::invalid_problem(format!(
            "{} = {} is out of range [{}, {}]",
            name, value, lo, hi
        )))
    }
}

/// Track multiple errors during validation.
#[derive(Debug, Default)]
pub struct ErrorAccumulator {
    errors: Vec<OptError>,
    warnings: Vec<String>,
}

impl ErrorAccumulator {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn push_error(&mut self, error: OptError) {
        self.errors.push(error);
    }

    pub fn push_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    pub fn into_result(mut self) -> OptResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.remove(0))
        }
    }

    pub fn into_errors(self) -> Vec<OptError> {
        self.errors
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn summary(&self) -> String {
        format!(
            "{} error(s), {} warning(s)",
            self.errors.len(),
            self.warnings.len()
        )
    }
}

impl fmt::Display for ErrorAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, w) in self.warnings.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "  [WARN] {}", w)?;
        }
        for (i, e) in self.errors.iter().enumerate() {
            if i > 0 || !self.warnings.is_empty() {
                writeln!(f)?;
            }
            write!(f, "  [ERR]  {}", e)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infeasible_error() {
        let err = OptError::infeasible("no solution exists");
        assert!(err.is_infeasible());
        assert!(!err.is_unbounded());
        assert!(err.to_string().contains("no solution exists"));
    }

    #[test]
    fn test_unbounded_error() {
        let err = OptError::unbounded("variable x1 unbounded");
        assert!(err.is_unbounded());
        assert_eq!(err.severity(), ErrorSeverity::Expected);
    }

    #[test]
    fn test_numerical_error() {
        let err = OptError::numerical("singular basis matrix");
        assert!(err.is_numerical());
        assert_eq!(err.severity(), ErrorSeverity::Serious);
    }

    #[test]
    fn test_convergence_failure() {
        let err = OptError::convergence(1000, "gap did not close");
        assert!(err.is_convergence_failure());
        assert!(err.to_string().contains("1000"));
    }

    #[test]
    fn test_time_limit() {
        let err = OptError::time_limit(120.5, 60.0);
        assert!(err.is_time_limit());
        assert!(err.to_string().contains("120.50"));
    }

    #[test]
    fn test_invalid_problem() {
        let err = OptError::invalid_problem("empty constraint matrix");
        assert!(err.to_string().contains("empty constraint matrix"));
        assert_eq!(err.severity(), ErrorSeverity::UserError);
    }

    #[test]
    fn test_solver_error_severity() {
        let err = OptError::solver("internal failure");
        assert_eq!(err.severity(), ErrorSeverity::Internal);
    }

    #[test]
    fn test_ensure_true() {
        assert!(ensure(true, || OptError::infeasible("test")).is_ok());
    }

    #[test]
    fn test_ensure_false() {
        assert!(ensure(false, || OptError::infeasible("test")).is_err());
    }

    #[test]
    fn test_ensure_finite_ok() {
        assert!(ensure_finite(1.0, "x").is_ok());
    }

    #[test]
    fn test_ensure_finite_nan() {
        assert!(ensure_finite(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_ensure_positive() {
        assert!(ensure_positive(5, "n").is_ok());
        assert!(ensure_positive(0, "n").is_err());
    }

    #[test]
    fn test_ensure_lengths_match() {
        assert!(ensure_lengths_match("a", 3, "b", 3).is_ok());
        assert!(ensure_lengths_match("a", 3, "b", 4).is_err());
    }

    #[test]
    fn test_ensure_in_range() {
        assert!(ensure_in_range(0.5, 0.0, 1.0, "x").is_ok());
        assert!(ensure_in_range(1.5, 0.0, 1.0, "x").is_err());
    }

    #[test]
    fn test_error_accumulator_empty() {
        let acc = ErrorAccumulator::new();
        assert!(!acc.has_errors());
        assert!(acc.into_result().is_ok());
    }

    #[test]
    fn test_error_accumulator_with_errors() {
        let mut acc = ErrorAccumulator::new();
        acc.push_error(OptError::infeasible("test"));
        acc.push_warning("minor issue");
        assert!(acc.has_errors());
        assert_eq!(acc.error_count(), 1);
        assert_eq!(acc.warning_count(), 1);
        assert!(acc.into_result().is_err());
    }

    #[test]
    fn test_error_accumulator_summary() {
        let mut acc = ErrorAccumulator::new();
        acc.push_error(OptError::infeasible("a"));
        acc.push_error(OptError::unbounded("b"));
        acc.push_warning("w");
        assert_eq!(acc.summary(), "2 error(s), 1 warning(s)");
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(ErrorSeverity::Expected.to_string(), "EXPECTED");
        assert_eq!(ErrorSeverity::Serious.to_string(), "SERIOUS");
    }
}
