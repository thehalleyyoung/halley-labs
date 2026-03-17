//! Bundle methods for nonsmooth optimization and Lagrangian relaxation.
//!
//! This module provides:
//! - [`BundleMethod`] – proximal bundle method for minimising a convex nonsmooth function.
//! - [`SubgradientSolver`] – classical subgradient methods with several step-size rules.
//! - [`LagrangianRelaxation`] – Lagrangian relaxation framework that decomposes an LP.

pub mod lagrangian;
pub mod proximal;
pub mod subgradient;

pub use lagrangian::{LagrangianConfig, LagrangianMethod, LagrangianProblem, LagrangianRelaxation};
pub use proximal::{BundleMethod, CuttingPlane};
pub use subgradient::{SubgradientConfig, SubgradientSolver, SubgradientStepRule};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared configuration
// ---------------------------------------------------------------------------

/// Configuration shared across bundle-type solvers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleConfig {
    pub max_iterations: usize,
    pub gap_tolerance: f64,
    pub time_limit: f64,
    pub bundle_capacity: usize,
    pub initial_mu: f64,
    pub mu_increase_factor: f64,
    pub mu_decrease_factor: f64,
    pub serious_step_threshold: f64,
    pub min_mu: f64,
    pub max_mu: f64,
    pub verbose: bool,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            gap_tolerance: 1e-6,
            time_limit: 3600.0,
            bundle_capacity: 50,
            initial_mu: 1.0,
            mu_increase_factor: 2.0,
            mu_decrease_factor: 0.5,
            serious_step_threshold: 0.1,
            min_mu: 1e-8,
            max_mu: 1e8,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Oracle output
// ---------------------------------------------------------------------------

/// Information returned by a subgradient oracle evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgradientInfo {
    /// Point at which the oracle was evaluated.
    pub point: Vec<f64>,
    /// Function value at that point.
    pub value: f64,
    /// A subgradient of the function at that point.
    pub subgradient: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Solver output
// ---------------------------------------------------------------------------

/// Result of a bundle / subgradient solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleResult {
    pub optimal_point: Vec<f64>,
    pub optimal_value: f64,
    pub iterations: usize,
    pub gap: f64,
    pub converged: bool,
    pub history: Vec<IterationInfo>,
}

/// Per-iteration bookkeeping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationInfo {
    pub iteration: usize,
    pub objective: f64,
    pub best_bound: f64,
    pub gap: f64,
    pub step_type: StepType,
    pub mu: f64,
}

/// Whether the step was a serious (descent) step or a null step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepType {
    Serious,
    Null,
}

impl std::fmt::Display for StepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepType::Serious => write!(f, "Serious"),
            StepType::Null => write!(f, "Null"),
        }
    }
}

// ---------------------------------------------------------------------------
// Lagrangian-specific result
// ---------------------------------------------------------------------------

/// Result of Lagrangian relaxation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianResult {
    pub dual_bound: f64,
    pub multipliers: Vec<f64>,
    pub primal_estimate: Vec<f64>,
    pub subproblem_solutions: Vec<Vec<f64>>,
    pub iterations: usize,
    pub gap: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_config_default() {
        let cfg = BundleConfig::default();
        assert_eq!(cfg.max_iterations, 500);
        assert!((cfg.gap_tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.time_limit - 3600.0).abs() < 1e-9);
        assert_eq!(cfg.bundle_capacity, 50);
        assert!((cfg.initial_mu - 1.0).abs() < 1e-15);
        assert!(!cfg.verbose);
    }

    #[test]
    fn test_subgradient_info_clone() {
        let info = SubgradientInfo {
            point: vec![1.0, 2.0],
            value: 3.0,
            subgradient: vec![0.5, -0.5],
        };
        let info2 = info.clone();
        assert_eq!(info.point, info2.point);
        assert!((info.value - info2.value).abs() < 1e-15);
    }

    #[test]
    fn test_bundle_result_fields() {
        let res = BundleResult {
            optimal_point: vec![1.0],
            optimal_value: -5.0,
            iterations: 42,
            gap: 1e-8,
            converged: true,
            history: vec![],
        };
        assert!(res.converged);
        assert_eq!(res.iterations, 42);
    }

    #[test]
    fn test_step_type_display() {
        assert_eq!(format!("{}", StepType::Serious), "Serious");
        assert_eq!(format!("{}", StepType::Null), "Null");
    }

    #[test]
    fn test_iteration_info() {
        let info = IterationInfo {
            iteration: 0,
            objective: 10.0,
            best_bound: 8.0,
            gap: 0.2,
            step_type: StepType::Serious,
            mu: 1.0,
        };
        assert_eq!(info.step_type, StepType::Serious);
    }

    #[test]
    fn test_lagrangian_result() {
        let res = LagrangianResult {
            dual_bound: -10.0,
            multipliers: vec![0.5, 0.3],
            primal_estimate: vec![1.0, 2.0, 3.0],
            subproblem_solutions: vec![vec![1.0], vec![2.0, 3.0]],
            iterations: 100,
            gap: 0.01,
        };
        assert_eq!(res.multipliers.len(), 2);
        assert_eq!(res.subproblem_solutions.len(), 2);
    }

    #[test]
    fn test_bundle_config_custom() {
        let cfg = BundleConfig {
            max_iterations: 1000,
            gap_tolerance: 1e-8,
            time_limit: 60.0,
            bundle_capacity: 100,
            initial_mu: 0.5,
            mu_increase_factor: 3.0,
            mu_decrease_factor: 0.3,
            serious_step_threshold: 0.2,
            min_mu: 1e-10,
            max_mu: 1e10,
            verbose: true,
        };
        assert_eq!(cfg.max_iterations, 1000);
        assert!(cfg.verbose);
    }

    #[test]
    fn test_step_type_eq() {
        assert_eq!(StepType::Serious, StepType::Serious);
        assert_ne!(StepType::Serious, StepType::Null);
    }

    #[test]
    fn test_bundle_config_serde_roundtrip() {
        let cfg = BundleConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: BundleConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.max_iterations, cfg2.max_iterations);
        assert!((cfg.gap_tolerance - cfg2.gap_tolerance).abs() < 1e-15);
    }

    #[test]
    fn test_subgradient_info_serde_roundtrip() {
        let info = SubgradientInfo {
            point: vec![1.0, 2.0, 3.0],
            value: 42.0,
            subgradient: vec![-1.0, 0.0, 1.0],
        };
        let json = serde_json::to_string(&info).unwrap();
        let info2: SubgradientInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.point, info2.point);
        assert!((info.value - info2.value).abs() < 1e-15);
    }
}
