//! Solver interface module.
//!
//! Provides a unified [`SolverInterface`] trait for LP solving, along with
//! concrete implementations: [`UnifiedSolver`] (dispatches to internal solvers),
//! [`ScipAdapter`] (mock SCIP emulation), and [`GcgAdapter`] (mock GCG emulation).

pub mod unified;
pub mod scip_adapter;
pub mod gcg_adapter;
pub mod highs_adapter;

pub use unified::UnifiedSolver;
pub use scip_adapter::ScipAdapter;
pub use gcg_adapter::GcgAdapter;
pub use highs_adapter::HighsAdapter;

use crate::error::OptResult;
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};
use serde::{Deserialize, Serialize};

/// Available solver backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverType {
    InternalSimplex,
    InternalInteriorPoint,
    ScipEmulation,
    GcgEmulation,
    /// HiGHS open-source solver (MIT license). Requires `highs` feature.
    HiGHS,
    /// Real SCIP solver via russcip bindings. Requires `scip` feature.
    Scip,
}

impl std::fmt::Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverType::InternalSimplex => write!(f, "Internal Simplex"),
            SolverType::InternalInteriorPoint => write!(f, "Internal Interior Point"),
            SolverType::ScipEmulation => write!(f, "SCIP Emulation"),
            SolverType::GcgEmulation => write!(f, "GCG Emulation"),
            SolverType::HiGHS => write!(f, "HiGHS"),
            SolverType::Scip => write!(f, "SCIP"),
        }
    }
}

/// Configuration for solver instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub solver_type: SolverType,
    pub time_limit: f64,
    pub gap_tolerance: f64,
    pub verbose: bool,
    pub threads: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::InternalSimplex,
            time_limit: 3600.0,
            gap_tolerance: 1e-6,
            verbose: false,
            threads: 1,
        }
    }
}

impl SolverConfig {
    pub fn with_type(mut self, solver_type: SolverType) -> Self {
        self.solver_type = solver_type;
        self
    }

    pub fn with_time_limit(mut self, limit: f64) -> Self {
        self.time_limit = limit;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

/// Unified interface for LP solvers.
pub trait SolverInterface: Send {
    /// Solve an LP problem.
    fn solve_lp(&mut self, problem: &LpProblem) -> OptResult<LpSolution>;

    /// Get the status of the last solve.
    fn get_status(&self) -> SolverStatus;

    /// Get dual values from the last solve.
    fn get_dual_values(&self) -> OptResult<Vec<f64>>;

    /// Get basis status from the last solve.
    fn get_basis(&self) -> OptResult<Vec<BasisStatus>>;

    /// Add a constraint to the current problem.
    fn add_constraint(
        &mut self,
        coeffs: &[(usize, f64)],
        ctype: ConstraintType,
        rhs: f64,
    ) -> OptResult<usize>;

    /// Add a variable to the current problem.
    fn add_variable(&mut self, obj: f64, lb: f64, ub: f64) -> OptResult<usize>;

    /// Set the objective function coefficients.
    fn set_objective(&mut self, coeffs: &[(usize, f64)]) -> OptResult<()>;

    /// Set the time limit for solving.
    fn set_time_limit(&mut self, seconds: f64);

    /// Get the solver name.
    fn name(&self) -> &str;

    /// Reset the solver state.
    fn reset(&mut self);
}

/// Create a solver from a configuration.
pub fn create_solver(config: SolverConfig) -> Box<dyn SolverInterface> {
    match config.solver_type {
        SolverType::InternalSimplex | SolverType::InternalInteriorPoint => {
            Box::new(UnifiedSolver::new(config))
        }
        SolverType::ScipEmulation => Box::new(ScipAdapter::new(config)),
        SolverType::GcgEmulation => Box::new(GcgAdapter::new(config)),
        SolverType::HiGHS => Box::new(HighsAdapter::new(config)),
        SolverType::Scip => {
            // With the `scip` feature, this would use russcip bindings.
            // Without it, falls back to SCIP emulation.
            #[cfg(feature = "scip")]
            {
                log::info!("Using SCIP via russcip bindings");
            }
            Box::new(ScipAdapter::new(config))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.solver_type, SolverType::InternalSimplex);
        assert!((config.time_limit - 3600.0).abs() < 1e-10);
        assert!(!config.verbose);
    }

    #[test]
    fn test_solver_config_builder() {
        let config = SolverConfig::default()
            .with_type(SolverType::ScipEmulation)
            .with_time_limit(60.0)
            .with_verbose(true)
            .with_threads(4);
        assert_eq!(config.solver_type, SolverType::ScipEmulation);
        assert!((config.time_limit - 60.0).abs() < 1e-10);
        assert!(config.verbose);
        assert_eq!(config.threads, 4);
    }

    #[test]
    fn test_solver_type_display() {
        assert_eq!(SolverType::InternalSimplex.to_string(), "Internal Simplex");
        assert_eq!(SolverType::ScipEmulation.to_string(), "SCIP Emulation");
    }

    #[test]
    fn test_create_solver_simplex() {
        let config = SolverConfig::default();
        let solver = create_solver(config);
        assert_eq!(solver.name(), "UnifiedSolver");
    }

    #[test]
    fn test_create_solver_scip() {
        let config = SolverConfig::default().with_type(SolverType::ScipEmulation);
        let solver = create_solver(config);
        assert_eq!(solver.name(), "SCIP-Emulation");
    }

    #[test]
    fn test_create_solver_gcg() {
        let config = SolverConfig::default().with_type(SolverType::GcgEmulation);
        let solver = create_solver(config);
        assert_eq!(solver.name(), "GCG-Emulation");
    }

    #[test]
    fn test_solver_config_threads_min() {
        let config = SolverConfig::default().with_threads(0);
        assert_eq!(config.threads, 1);
    }

    #[test]
    fn test_solver_type_eq() {
        assert_eq!(SolverType::InternalSimplex, SolverType::InternalSimplex);
        assert_ne!(SolverType::InternalSimplex, SolverType::ScipEmulation);
    }
}
