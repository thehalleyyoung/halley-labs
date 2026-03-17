//! # regsynth-pareto
//!
//! Pareto synthesis engine for RegSynth — multi-objective optimization,
//! trajectory planning, and incremental frontier maintenance for
//! regulatory compliance.

pub mod cost_model;
pub mod dominance;
pub mod frontier;
pub mod incremental_maintenance;
pub mod iterative_maxsmt;
pub mod metrics;
pub mod scalarization;
pub mod strategy_repr;
pub mod trajectory;

pub use cost_model::{CostModel, ObligationCostEstimate};
pub use dominance::{
    dominates, epsilon_dominates, fast_non_dominated_sort, filter_dominated, pareto_compare,
    ParetoOrdering,
};
pub use frontier::ParetoFrontier;
pub use incremental_maintenance::{ConstraintDiff, IncrementalMaintainer};
pub use iterative_maxsmt::{IterativeMaxSmtConfig, ParetoEnumerator};
pub use metrics::ParetoMetrics;
pub use scalarization::{
    ChebyshevScalarizer, EpsilonConstraintScalarizer, PascolettiSerafiniScalarizer,
    ScalarizedObjective, WeightedSumScalarizer,
};
pub use strategy_repr::{ComplianceStrategy, ObligationEntry, StrategyBitVec};
pub use trajectory::{ComplianceTrajectory, ParetoTrajectory, TrajectoryOptimizer};

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// CostVector — the fundamental multi-objective cost representation
// ---------------------------------------------------------------------------

/// Standard dimension indices for regulatory compliance cost vectors.
pub mod dim {
    pub const FINANCIAL_COST: usize = 0;
    pub const TIME_TO_COMPLIANCE: usize = 1;
    pub const REGULATORY_RISK: usize = 2;
    pub const IMPLEMENTATION_COMPLEXITY: usize = 3;
    pub const DEFAULT_DIM: usize = 4;
}

/// A multi-dimensional cost vector for Pareto analysis.
///
/// All objectives are assumed to be *minimization* targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostVector {
    pub values: Vec<f64>,
}

impl CostVector {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Create a zero vector of the given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Standard 4-D regulatory compliance cost vector.
    pub fn regulatory(
        financial_cost: f64,
        time_to_compliance: f64,
        regulatory_risk: f64,
        implementation_complexity: f64,
    ) -> Self {
        Self {
            values: vec![
                financial_cost,
                time_to_compliance,
                regulatory_risk,
                implementation_complexity,
            ],
        }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, idx: usize) -> f64 {
        self.values[idx]
    }

    /// Component-wise minimum of two cost vectors.
    pub fn component_min(&self, other: &CostVector) -> CostVector {
        assert_eq!(self.dim(), other.dim());
        CostVector {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a.min(*b))
                .collect(),
        }
    }

    /// Component-wise maximum of two cost vectors.
    pub fn component_max(&self, other: &CostVector) -> CostVector {
        assert_eq!(self.dim(), other.dim());
        CostVector {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a.max(*b))
                .collect(),
        }
    }

    /// Euclidean distance to another cost vector.
    pub fn euclidean_distance(&self, other: &CostVector) -> f64 {
        assert_eq!(self.dim(), other.dim());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Chebyshev (L∞) distance.
    pub fn chebyshev_distance(&self, other: &CostVector) -> f64 {
        assert_eq!(self.dim(), other.dim());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Weighted sum: ∑ wᵢ · vᵢ.
    pub fn weighted_sum(&self, weights: &[f64]) -> f64 {
        assert_eq!(self.dim(), weights.len());
        self.values
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    /// Dot product with another vector.
    pub fn dot(&self, other: &CostVector) -> f64 {
        assert_eq!(self.dim(), other.dim());
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Normalize to unit length; returns zero vector if norm is zero.
    pub fn normalize(&self) -> CostVector {
        let n = self.norm();
        if n < f64::EPSILON {
            CostVector::zeros(self.dim())
        } else {
            CostVector {
                values: self.values.iter().map(|v| v / n).collect(),
            }
        }
    }

    /// Add two cost vectors component-wise.
    pub fn add(&self, other: &CostVector) -> CostVector {
        assert_eq!(self.dim(), other.dim());
        CostVector {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Subtract other from self component-wise.
    pub fn sub(&self, other: &CostVector) -> CostVector {
        assert_eq!(self.dim(), other.dim());
        CostVector {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Scale all components by a scalar.
    pub fn scale(&self, s: f64) -> CostVector {
        CostVector {
            values: self.values.iter().map(|v| v * s).collect(),
        }
    }
}

impl fmt::Display for CostVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", v)?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ParetoError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Empty frontier")]
    EmptyFrontier,

    #[error("Solver error: {0}")]
    SolverError(String),

    #[error("Infeasible problem: {0}")]
    Infeasible(String),

    #[error("Timeout after {0} iterations")]
    Timeout(usize),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
}

pub type ParetoResult<T> = Result<T, ParetoError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_vector_regulatory() {
        let cv = CostVector::regulatory(1000.0, 6.0, 0.1, 50.0);
        assert_eq!(cv.dim(), 4);
        assert_eq!(cv.get(dim::FINANCIAL_COST), 1000.0);
        assert_eq!(cv.get(dim::TIME_TO_COMPLIANCE), 6.0);
    }

    #[test]
    fn test_cost_vector_distance() {
        let a = CostVector::new(vec![0.0, 0.0]);
        let b = CostVector::new(vec![3.0, 4.0]);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-10);
        assert!((a.chebyshev_distance(&b) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_vector_weighted_sum() {
        let cv = CostVector::new(vec![2.0, 3.0, 4.0]);
        let w = vec![1.0, 2.0, 0.5];
        assert!((cv.weighted_sum(&w) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_vector_arithmetic() {
        let a = CostVector::new(vec![1.0, 2.0]);
        let b = CostVector::new(vec![3.0, 1.0]);
        assert_eq!(a.add(&b).values, vec![4.0, 3.0]);
        assert_eq!(a.sub(&b).values, vec![-2.0, 1.0]);
        assert_eq!(a.scale(2.0).values, vec![2.0, 4.0]);
    }

    #[test]
    fn test_cost_vector_component_min_max() {
        let a = CostVector::new(vec![1.0, 5.0, 3.0]);
        let b = CostVector::new(vec![2.0, 2.0, 4.0]);
        assert_eq!(a.component_min(&b).values, vec![1.0, 2.0, 3.0]);
        assert_eq!(a.component_max(&b).values, vec![2.0, 5.0, 4.0]);
    }

    #[test]
    fn test_cost_vector_normalize() {
        let v = CostVector::new(vec![3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_vector_zeros_normalize() {
        let z = CostVector::zeros(3);
        let n = z.normalize();
        assert_eq!(n.values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_display() {
        let cv = CostVector::new(vec![1.0, 2.5]);
        assert_eq!(format!("{}", cv), "[1.0000, 2.5000]");
    }
}
