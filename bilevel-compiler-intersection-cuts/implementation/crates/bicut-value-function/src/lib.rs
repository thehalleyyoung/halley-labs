//! Value function computation for the BiCut bilevel optimization compiler.
//!
//! This crate computes the lower-level value function
//! φ(x) = min { c^T y : Ay ≤ b + Bx, y ≥ 0 }
//! for bilevel optimization problems.
//!
//! # Modules
//!
//! - [`oracle`]: Value function oracle trait and implementations
//! - [`parametric`]: Parametric LP solver with basis tracking
//! - [`critical_region`]: Critical region computation and enumeration
//! - [`piecewise_linear`]: Piecewise linear value function representation
//! - [`sampling`]: Sampling-based value function approximation
//! - [`lifting`]: Value function lifting for intersection cuts
//! - [`dual_analysis`]: Dual analysis and shadow prices
//! - [`bounds`]: Value function bounds computation

pub mod bounds;
pub mod critical_region;
pub mod dual_analysis;
pub mod lifting;
pub mod oracle;
pub mod parametric;
pub mod piecewise_linear;
pub mod sampling;

// Re-exports of primary types and traits.
pub use bounds::{BoundTighteningResult, BoundsComputer, ValueFunctionBounds};
pub use critical_region::{CriticalRegion, CriticalRegionEnumerator, RegionAdjacency};
pub use dual_analysis::{DualAnalyzer, DualStabilityInfo, ShadowPriceInfo};
pub use lifting::{LiftingCoefficients, LiftingComputer, SubadditiveApprox};
pub use oracle::{CachedOracle, ExactLpOracle, OracleStatistics, ValueFunctionOracle};
pub use parametric::{BasisInfo, ParametricSolver, SensitivityRange};
pub use piecewise_linear::{AffinePiece, PiecewiseLinearVF, SubdifferentialInfo};
pub use sampling::{AdaptiveSampler, SamplingApproximation, SamplingConfig};

use thiserror::Error;

/// Errors that can occur in value function computation.
#[derive(Error, Debug)]
pub enum VFError {
    #[error("LP solver error: {0}")]
    LpError(String),

    #[error("Infeasible lower-level problem for given x")]
    InfeasibleLowerLevel,

    #[error("Unbounded lower-level problem")]
    UnboundedLowerLevel,

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("No critical regions found")]
    NoCriticalRegions,

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Sampling error: {0}")]
    SamplingError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for value function operations.
pub type VFResult<T> = Result<T, VFError>;

/// Default numerical tolerance.
pub const TOLERANCE: f64 = 1e-8;

/// Large constant used as infinity proxy.
pub const BIG_M: f64 = 1e10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = VFError::InfeasibleLowerLevel;
        assert!(format!("{}", e).contains("Infeasible"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let e = VFError::DimensionMismatch {
            expected: 3,
            got: 5,
        };
        assert!(format!("{}", e).contains("3"));
        assert!(format!("{}", e).contains("5"));
    }

    #[test]
    fn test_tolerance_constant() {
        assert!(TOLERANCE > 0.0);
        assert!(TOLERANCE < 1e-6);
    }
}
