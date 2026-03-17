// safestep-solver: SAT/SMT solver crate for bounded model checking of deployment plans.
//
// Provides a self-contained DPLL/CDCL SAT solver with clause learning, VSIDS,
// theory propagation (SMT), incremental solving, MaxSAT optimization, and proof traces.

pub mod cdcl;
pub mod clause;
pub mod config;
pub mod incremental;
pub mod optimization;
pub mod proof;
pub mod propagation;
pub mod smt;
pub mod theory;
pub mod variable;

// Re-export primary types for ergonomic use.
pub use cdcl::{CdclSolver, SatResult, SolverStats, UnsatCore};
pub use clause::{Clause, ClauseDatabase, ClauseId, ClauseStatus};
pub use config::SolverConfig;
pub use incremental::IncrementalSolver;
pub use optimization::{MaxSatSolver, OptimizationResult};
pub use proof::{ProofChecker, ProofTrace, UnsatCertificate};
pub use propagation::PropagationEngine;
pub use smt::{FormulaBuilder, Model, SmtFormula, SmtResult, SmtSolver};
pub use theory::{CombinedTheory, EqualityTheory, LinearArithmeticTheory, TheoryPropagator};
pub use variable::{Assignment, Literal, Variable, VariableManager};
