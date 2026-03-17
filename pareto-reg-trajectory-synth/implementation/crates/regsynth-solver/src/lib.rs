// regsynth-solver: Solver backend for RegSynth
//
// Native solver implementations for multi-jurisdictional regulatory compliance:
// - CDCL SAT solver with watched literals, VSIDS, and 1UIP conflict analysis
// - DPLL(T) SMT solver with linear arithmetic and equality theory solvers
// - Weighted partial MaxSMT solver (Fu-Malik style)
// - ILP solver with simplex LP relaxation and branch-and-bound
// - Pareto frontier enumeration via iterative MaxSMT
// - MUS (Minimal Unsatisfiable Subset) extraction (deletion-based + MARCO)
// - Incremental solving with push/pop scope management

pub mod result;
pub mod solver_config;
pub mod sat_solver;
pub mod smt_solver;
pub mod maxsmt_solver;
pub mod ilp_solver;
pub mod pareto_enumerator;
pub mod conflict_extractor;
pub mod incremental;

// Re-export key types for convenient use
pub use result::{
    Assignment, Clause, IlpResult, IlpSolution, Literal, MaxSmtResult, MaxSmtStatus,
    MinimalUnsatisfiableSubset, Model, ParetoFrontier, ParetoPoint, ParetoResult, SatResult,
    SmtResult, SolverResult, SolverStatistics, Variable,
};
pub use solver_config::{SolverBackend, SolverConfig};

pub use sat_solver::{DpllSolver, solve_cnf, solve_cnf_with_config};
pub use smt_solver::SmtSolver;
pub use maxsmt_solver::{MaxSmtSolver, SoftClause};
pub use ilp_solver::{IlpSolver, SimplexSolver};
pub use pareto_enumerator::{LinearObjective, ParetoEnumerator};
pub use conflict_extractor::MusExtractor;
pub use incremental::IncrementalSolver;

// ─── Domain-level types (preserved from original) ───────────────────────────

use serde::{Deserialize, Serialize};
use regsynth_types::Id;

/// High-level result of a regulatory compliance solver invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceResult {
    Feasible(Solution),
    Infeasible(ConflictCore),
    Timeout,
    Unknown,
}

/// A feasible compliance solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub objective_value: f64,
    pub variable_assignments: Vec<(String, f64)>,
    pub satisfied_obligations: Vec<Id>,
    pub waived_obligations: Vec<Id>,
}

/// A conflict core: minimal set of mutually infeasible obligations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictCore {
    pub obligation_ids: Vec<Id>,
    pub explanation: String,
    pub conflict_type: ConflictType,
}

impl ConflictCore {
    pub fn new(ids: Vec<Id>, explanation: impl Into<String>, conflict_type: ConflictType) -> Self {
        Self {
            obligation_ids: ids,
            explanation: explanation.into(),
            conflict_type,
        }
    }

    pub fn size(&self) -> usize {
        self.obligation_ids.len()
    }
}

/// Type of conflict detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    ResourceConflict,
    TemporalConflict,
    LogicalContradiction,
    BudgetExceeded,
    StaffShortage,
}
