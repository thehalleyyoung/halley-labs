//! Solver trait definitions and configuration types.

use std::fmt;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use bilevel_types::{BilevelError, BilevelResult, Bounds, ConstraintSense, ObjectiveSense, VariableType, VarIdx, ConIdx};
use crate::model::SolverModel;
use crate::solution::Solution;
use crate::warmstart::WarmStartInfo;

/// Status of a solver after optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalError,
    NotSolved,
    Feasible,
    InfeasibleOrUnbounded,
}

impl fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Optimal => write!(f, "Optimal"),
            Self::Infeasible => write!(f, "Infeasible"),
            Self::Unbounded => write!(f, "Unbounded"),
            Self::IterationLimit => write!(f, "Iteration limit"),
            Self::TimeLimit => write!(f, "Time limit"),
            Self::NumericalError => write!(f, "Numerical error"),
            Self::NotSolved => write!(f, "Not solved"),
            Self::Feasible => write!(f, "Feasible (not proven optimal)"),
            Self::InfeasibleOrUnbounded => write!(f, "Infeasible or unbounded"),
        }
    }
}

/// Pricing strategy for the simplex method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingStrategy {
    Dantzig,
    SteepestEdge,
    Bland,
    Devex,
}

impl Default for PricingStrategy {
    fn default() -> Self { PricingStrategy::SteepestEdge }
}

/// Branching strategy for MIP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchingStrategy {
    MostFractional,
    StrongBranching,
    PseudoCost,
    FirstFractional,
    ReliabilityBranching,
}

impl Default for BranchingStrategy {
    fn default() -> Self { BranchingStrategy::MostFractional }
}

/// Node selection strategy for branch-and-bound
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeSelection {
    BestFirst,
    DepthFirst,
    BestEstimate,
    Hybrid,
}

impl Default for NodeSelection {
    fn default() -> Self { NodeSelection::BestFirst }
}

/// Presolve level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresolveLevel {
    Off,
    Light,
    Aggressive,
}

impl Default for PresolveLevel {
    fn default() -> Self { PresolveLevel::Light }
}

/// Solver configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub max_iterations: usize,
    pub time_limit: Option<Duration>,
    pub feasibility_tol: f64,
    pub optimality_tol: f64,
    pub integrality_tol: f64,
    pub pivot_tol: f64,
    pub zero_tol: f64,
    pub pricing: PricingStrategy,
    pub branching: BranchingStrategy,
    pub node_selection: NodeSelection,
    pub presolve: PresolveLevel,
    pub mip_gap: f64,
    pub mip_gap_abs: f64,
    pub max_nodes: usize,
    pub strong_branching_candidates: usize,
    pub warm_start: bool,
    pub verbosity: u32,
    pub scaling: bool,
    pub perturbation: f64,
    pub refactorization_interval: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            max_iterations: 1_000_000,
            time_limit: None,
            feasibility_tol: 1e-8,
            optimality_tol: 1e-8,
            integrality_tol: 1e-6,
            pivot_tol: 1e-10,
            zero_tol: 1e-12,
            pricing: PricingStrategy::default(),
            branching: BranchingStrategy::default(),
            node_selection: NodeSelection::default(),
            presolve: PresolveLevel::default(),
            mip_gap: 1e-4,
            mip_gap_abs: 1e-6,
            max_nodes: 1_000_000,
            strong_branching_candidates: 10,
            warm_start: true,
            verbosity: 0,
            scaling: true,
            perturbation: 1e-6,
            refactorization_interval: 100,
        }
    }
}

impl SolverConfig {
    pub fn fast() -> Self {
        SolverConfig {
            max_iterations: 10_000,
            time_limit: Some(Duration::from_secs(10)),
            feasibility_tol: 1e-6,
            optimality_tol: 1e-6,
            presolve: PresolveLevel::Off,
            ..Default::default()
        }
    }

    pub fn precise() -> Self {
        SolverConfig {
            feasibility_tol: 1e-10,
            optimality_tol: 1e-10,
            integrality_tol: 1e-8,
            pivot_tol: 1e-12,
            ..Default::default()
        }
    }

    pub fn for_obbt() -> Self {
        SolverConfig {
            warm_start: true,
            presolve: PresolveLevel::Off,
            pricing: PricingStrategy::Dantzig,
            max_iterations: 100_000,
            ..Default::default()
        }
    }
}

/// Statistics from a solve
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStatistics {
    pub iterations: usize,
    pub phase1_iterations: usize,
    pub phase2_iterations: usize,
    pub nodes_explored: usize,
    pub nodes_remaining: usize,
    pub solve_time: f64,
    pub refactorizations: usize,
    pub degenerate_pivots: usize,
    pub bound_flips: usize,
    pub best_bound: f64,
    pub best_incumbent: f64,
    pub mip_gap: f64,
    pub presolve_reductions: usize,
}

impl fmt::Display for SolverStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Solver Statistics:")?;
        writeln!(f, "  Iterations: {} (Phase I: {}, Phase II: {})",
            self.iterations, self.phase1_iterations, self.phase2_iterations)?;
        writeln!(f, "  Solve time: {:.3}s", self.solve_time)?;
        writeln!(f, "  Refactorizations: {}", self.refactorizations)?;
        writeln!(f, "  Degenerate pivots: {}", self.degenerate_pivots)?;
        if self.nodes_explored > 0 {
            writeln!(f, "  B&B Nodes: {} explored, {} remaining",
                self.nodes_explored, self.nodes_remaining)?;
            writeln!(f, "  MIP gap: {:.6}", self.mip_gap)?;
        }
        Ok(())
    }
}

/// Action returned by a solver callback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAction {
    Continue,
    Stop,
}

/// Context information passed to callbacks
#[derive(Debug, Clone)]
pub struct CallbackContext {
    pub status: SolverStatus,
    pub iteration: usize,
    pub objective: Option<f64>,
    pub best_bound: Option<f64>,
    pub mip_gap: Option<f64>,
    pub nodes_explored: usize,
    pub primal_solution: Option<Vec<f64>>,
}

/// Solver callback type
pub type SolverCallback = Box<dyn FnMut(&CallbackContext) -> CallbackAction + Send>;

/// Solver-specific error types
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Infeasible: {0}")]
    Infeasible(String),
    #[error("Unbounded: {0}")]
    Unbounded(String),
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    #[error("Iteration limit reached after {0} iterations")]
    IterationLimit(usize),
    #[error("Time limit reached after {0:.2}s")]
    TimeLimit(f64),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    #[error("Basis error: {0}")]
    BasisError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<SolverError> for BilevelError {
    fn from(e: SolverError) -> Self {
        BilevelError::SolverError { message: e.to_string() }
    }
}

/// Result from a solve operation
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub status: SolverStatus,
    pub objective_value: f64,
    pub solution: Option<Solution>,
    pub statistics: SolverStatistics,
}

impl SolveResult {
    pub fn not_solved() -> Self {
        SolveResult {
            status: SolverStatus::NotSolved,
            objective_value: f64::NAN,
            solution: None,
            statistics: SolverStatistics::default(),
        }
    }

    pub fn is_optimal(&self) -> bool {
        self.status == SolverStatus::Optimal
    }

    pub fn has_solution(&self) -> bool {
        matches!(self.status, SolverStatus::Optimal | SolverStatus::Feasible)
    }

    pub fn primal_values(&self) -> &[f64] {
        self.solution.as_ref().expect("No solution available").primal_values()
    }

    pub fn dual_values(&self) -> &[f64] {
        self.solution.as_ref().expect("No solution available").dual_values()
    }
}

/// Trait for solver backends
pub trait SolverBackend: Send {
    fn name(&self) -> &str;
    fn solve(&mut self, model: &SolverModel) -> BilevelResult<SolveResult>;
    fn solve_warm(&mut self, model: &SolverModel, warm_start: &WarmStartInfo) -> BilevelResult<SolveResult>;
    fn config(&self) -> &SolverConfig;
    fn set_config(&mut self, config: SolverConfig);
    fn set_callback(&mut self, callback: SolverCallback);
    fn reset(&mut self);
}

/// Trait for incremental solver operations (modify and re-solve)
pub trait IncrementalSolver: SolverBackend {
    fn modify_rhs_and_resolve(&mut self, model: &mut SolverModel, constraint: ConIdx, new_rhs: f64) -> BilevelResult<SolveResult>;
    fn modify_bounds_and_resolve(&mut self, model: &mut SolverModel, variable: VarIdx, new_bounds: Bounds) -> BilevelResult<SolveResult>;
    fn add_constraint_and_resolve(&mut self, model: &mut SolverModel, coeffs: Vec<(VarIdx, f64)>, sense: ConstraintSense, rhs: f64, name: &str) -> BilevelResult<SolveResult>;
    fn remove_constraint_and_resolve(&mut self, model: &mut SolverModel, constraint: ConIdx) -> BilevelResult<SolveResult>;
    fn fix_variable_and_resolve(&mut self, model: &mut SolverModel, variable: VarIdx, value: f64) -> BilevelResult<SolveResult>;
    fn get_warm_start(&self) -> Option<WarmStartInfo>;
}

/// Available built-in solver types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinSolver {
    PrimalSimplex,
    DualSimplex,
    Auto,
}

impl Default for BuiltinSolver {
    fn default() -> Self { BuiltinSolver::Auto }
}

/// Create a solver backend
pub fn create_solver(solver_type: BuiltinSolver) -> Box<dyn SolverBackend> {
    match solver_type {
        BuiltinSolver::PrimalSimplex => Box::new(crate::simplex::SimplexSolver::new()),
        BuiltinSolver::DualSimplex => Box::new(crate::dual_simplex::DualSimplexSolver::new()),
        BuiltinSolver::Auto => Box::new(crate::dual_simplex::DualSimplexSolver::new()),
    }
}

/// Create a solver backend with custom configuration
pub fn create_solver_with_config(solver_type: BuiltinSolver, config: SolverConfig) -> Box<dyn SolverBackend> {
    let mut solver = create_solver(solver_type);
    solver.set_config(config);
    solver
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_status_display() {
        assert_eq!(format!("{}", SolverStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SolverStatus::Infeasible), "Infeasible");
    }

    #[test]
    fn test_default_config() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 1_000_000);
        assert!(config.warm_start);
    }

    #[test]
    fn test_fast_config() {
        let config = SolverConfig::fast();
        assert_eq!(config.max_iterations, 10_000);
        assert!(config.time_limit.is_some());
    }

    #[test]
    fn test_obbt_config() {
        let config = SolverConfig::for_obbt();
        assert!(config.warm_start);
        assert_eq!(config.presolve, PresolveLevel::Off);
    }

    #[test]
    fn test_solve_result() {
        let result = SolveResult::not_solved();
        assert!(!result.is_optimal());
        assert!(!result.has_solution());
    }

    #[test]
    fn test_create_solver() {
        let solver = create_solver(BuiltinSolver::PrimalSimplex);
        assert_eq!(solver.name(), "PrimalSimplex");
    }

    #[test]
    fn test_solver_error_conversion() {
        let err = SolverError::Infeasible("test".to_string());
        let bilevel_err: BilevelError = err.into();
        assert!(matches!(bilevel_err, BilevelError::SolverError { .. }));
    }
}
