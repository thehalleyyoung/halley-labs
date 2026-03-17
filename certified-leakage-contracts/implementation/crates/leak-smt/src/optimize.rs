//! Optimisation-based SMT solving.
//!
//! [`OptimizeSolver`] extends the basic solver interface with objective
//! functions, enabling the framework to find minimal leakage bounds or
//! optimal contract parameters.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::expr::{Expr, ExprId, ExprPool, Sort, Value};
use crate::smtlib::Script;
use crate::solver::{Model, SmtSolver, SolverBackend, SolverConfig, SolverResult};

// ---------------------------------------------------------------------------
// Objective direction
// ---------------------------------------------------------------------------

/// Direction for an optimisation objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

impl fmt::Display for ObjectiveDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveDirection::Minimize => write!(f, "minimize"),
            ObjectiveDirection::Maximize => write!(f, "maximize"),
        }
    }
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

/// A named optimisation objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    /// Human-readable label.
    pub label: String,
    /// The expression to optimise.
    pub expr: ExprId,
    /// Minimise or maximise.
    pub direction: ObjectiveDirection,
}

// ---------------------------------------------------------------------------
// OptimizeSolver
// ---------------------------------------------------------------------------

/// Solver wrapper that supports objective-based optimisation queries.
///
/// When backed by Z3, this uses the `(minimize ...)` / `(maximize ...)`
/// extensions.  For solvers without native support, it falls back to
/// iterative binary-search refinement.
#[derive(Debug)]
pub struct OptimizeSolver {
    /// The underlying solver config.
    config: SolverConfig,
    /// Registered objectives.
    objectives: Vec<Objective>,
}

impl OptimizeSolver {
    /// Create a new optimisation solver with the given config.
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            objectives: Vec::new(),
        }
    }

    /// Create an optimisation solver using Z3 defaults.
    pub fn z3_default() -> Self {
        Self::new(SolverConfig::z3_default())
    }

    /// Register an objective to minimise.
    pub fn minimize(&mut self, label: impl Into<String>, expr: ExprId) {
        self.objectives.push(Objective {
            label: label.into(),
            expr,
            direction: ObjectiveDirection::Minimize,
        });
    }

    /// Register an objective to maximise.
    pub fn maximize(&mut self, label: impl Into<String>, expr: ExprId) {
        self.objectives.push(Objective {
            label: label.into(),
            expr,
            direction: ObjectiveDirection::Maximize,
        });
    }

    /// Clear all registered objectives.
    pub fn clear_objectives(&mut self) {
        self.objectives.clear();
    }

    /// Number of registered objectives.
    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }

    /// Run the optimisation query and return the solver result.
    ///
    /// The returned model (if `Sat`) includes the optimal values for the
    /// registered objectives.
    pub fn optimize(&mut self, script: &Script, pool: &ExprPool) -> SolverResult {
        // TODO: implement Z3-opt integration or binary-search fallback
        log::warn!("OptimizeSolver::optimize is a stub");
        SolverResult::Unknown("optimize stub".to_string())
    }

    /// Access the solver configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Access registered objectives.
    pub fn objectives(&self) -> &[Objective] {
        &self.objectives
    }
}
