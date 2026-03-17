//! SMT solver abstraction layer.
//!
//! Defines a trait [`SmtSolver`] and concrete backends for communicating with
//! external SMT solvers (Z3, CVC5) as well as a lightweight internal solver
//! for trivial satisfiability checks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::expr::{ExprId, ExprPool, Value};
use crate::smtlib::{Script, SmtResponse};

// ---------------------------------------------------------------------------
// SolverBackend
// ---------------------------------------------------------------------------

/// Which external solver binary to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverBackend {
    Z3,
    CVC5,
    /// A custom solver invoked by its binary path.
    Custom(String),
}

impl fmt::Display for SolverBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverBackend::Z3 => write!(f, "z3"),
            SolverBackend::CVC5 => write!(f, "cvc5"),
            SolverBackend::Custom(path) => write!(f, "custom({})", path),
        }
    }
}

// ---------------------------------------------------------------------------
// SolverConfig
// ---------------------------------------------------------------------------

/// Configuration for an SMT solver instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// The solver backend to use.
    pub backend: SolverBackend,
    /// Timeout in seconds (0 = no timeout).
    pub timeout_secs: u64,
    /// Produce models on sat results.
    pub produce_models: bool,
    /// Additional CLI flags forwarded to the solver process.
    pub extra_args: Vec<String>,
    /// SMT-LIB2 logic string (e.g. "QF_ABV").
    pub logic: Option<String>,
}

impl SolverConfig {
    /// Default configuration for Z3 with a 60-second timeout.
    pub fn z3_default() -> Self {
        Self {
            backend: SolverBackend::Z3,
            timeout_secs: 60,
            produce_models: true,
            extra_args: Vec::new(),
            logic: None,
        }
    }

    /// Default configuration for CVC5 with a 60-second timeout.
    pub fn cvc5_default() -> Self {
        Self {
            backend: SolverBackend::CVC5,
            timeout_secs: 60,
            produce_models: true,
            extra_args: Vec::new(),
            logic: None,
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::z3_default()
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A satisfying assignment returned by the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Variable name → concrete value assignments.
    pub assignments: HashMap<String, Value>,
    /// Raw textual model as returned by the solver (if available).
    pub raw: Option<String>,
}

impl Model {
    /// Create an empty model.
    pub fn empty() -> Self {
        Self {
            assignments: HashMap::new(),
            raw: None,
        }
    }

    /// Look up a variable in the model.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.assignments.get(name)
    }

    /// Number of assignments in this model.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SolverResult
// ---------------------------------------------------------------------------

/// Result of a solver invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverResult {
    /// The formula is satisfiable, with an optional model.
    Sat(Option<Model>),
    /// The formula is unsatisfiable.
    Unsat,
    /// The solver could not determine satisfiability.
    Unknown(String),
    /// The solver encountered an error.
    Error(String),
    /// The query timed out.
    Timeout,
}

impl SolverResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SolverResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SolverResult::Unsat)
    }

    /// Extract the model if the result is `Sat` with a model.
    pub fn model(&self) -> Option<&Model> {
        match self {
            SolverResult::Sat(Some(m)) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SmtSolver trait
// ---------------------------------------------------------------------------

/// Common interface for all SMT solver implementations.
pub trait SmtSolver: fmt::Debug + Send {
    /// Check satisfiability of the given script.
    fn check_sat(&mut self, script: &Script, pool: &ExprPool) -> SolverResult;

    /// Check satisfiability and return a model if sat.
    fn check_sat_with_model(&mut self, script: &Script, pool: &ExprPool) -> SolverResult;

    /// Reset the solver state.
    fn reset(&mut self);

    /// Return the backend identifier.
    fn backend(&self) -> SolverBackend;
}

// ---------------------------------------------------------------------------
// ExternalSolver
// ---------------------------------------------------------------------------

/// Solver that shells out to an external SMT solver process (Z3 / CVC5).
#[derive(Debug)]
pub struct ExternalSolver {
    config: SolverConfig,
}

impl ExternalSolver {
    /// Create a new external solver with the given configuration.
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Return the path or name of the solver binary.
    pub fn binary_name(&self) -> &str {
        match &self.config.backend {
            SolverBackend::Z3 => "z3",
            SolverBackend::CVC5 => "cvc5",
            SolverBackend::Custom(p) => p.as_str(),
        }
    }
}

impl SmtSolver for ExternalSolver {
    fn check_sat(&mut self, _script: &Script, _pool: &ExprPool) -> SolverResult {
        // TODO: spawn solver process, write script, parse output
        log::warn!("ExternalSolver::check_sat is a stub");
        SolverResult::Unknown("stub implementation".to_string())
    }

    fn check_sat_with_model(&mut self, _script: &Script, _pool: &ExprPool) -> SolverResult {
        log::warn!("ExternalSolver::check_sat_with_model is a stub");
        SolverResult::Unknown("stub implementation".to_string())
    }

    fn reset(&mut self) {
        // No persistent state to clear for one-shot process invocation.
    }

    fn backend(&self) -> SolverBackend {
        self.config.backend.clone()
    }
}

// ---------------------------------------------------------------------------
// InternalSolver
// ---------------------------------------------------------------------------

/// Lightweight built-in solver for trivial or structurally simple queries.
///
/// This solver handles common constant-propagation and simple Boolean
/// satisfiability checks without spawning an external process.
#[derive(Debug)]
pub struct InternalSolver {
    config: SolverConfig,
}

impl InternalSolver {
    /// Create a new internal solver.
    pub fn new() -> Self {
        Self {
            config: SolverConfig {
                backend: SolverBackend::Z3, // label only
                timeout_secs: 0,
                produce_models: false,
                extra_args: Vec::new(),
                logic: None,
            },
        }
    }
}

impl Default for InternalSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtSolver for InternalSolver {
    fn check_sat(&mut self, _script: &Script, _pool: &ExprPool) -> SolverResult {
        log::warn!("InternalSolver::check_sat is a stub");
        SolverResult::Unknown("stub implementation".to_string())
    }

    fn check_sat_with_model(&mut self, _script: &Script, _pool: &ExprPool) -> SolverResult {
        log::warn!("InternalSolver::check_sat_with_model is a stub");
        SolverResult::Unknown("stub implementation".to_string())
    }

    fn reset(&mut self) {}

    fn backend(&self) -> SolverBackend {
        self.config.backend.clone()
    }
}
