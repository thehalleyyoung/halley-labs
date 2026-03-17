//! # smt-solver
//!
//! SMT solver abstraction layer for the MutSpec project. Provides a
//! Rust-native AST for SMT-LIB2, context management, incremental solving
//! support, model extraction, theory-specific utilities, and an s-expression
//! parser for interpreting solver responses.
//!
//! The primary workflow is:
//!
//! 1. Build formulas using [`ast::SmtExpr`] builders or convert from
//!    [`shared_types::Formula`] via [`context::SmtContext`].
//! 2. Assert them into an [`context::SmtContext`].
//! 3. Feed the context to a [`solver::SmtSolver`] (e.g., [`solver::ProcessSolver`]).
//! 4. Inspect [`solver::SolverResult`], extract [`model::SmtModel`] on SAT,
//!    or [`solver::SmtSolver::get_unsat_core`] on UNSAT.

pub mod ast;
pub mod context;
pub mod incremental;
pub mod model;
pub mod sexp_parser;
pub mod solver;
pub mod theories;

// Re-export primary types for convenient access.
pub use ast::{SmtCommand, SmtExpr, SmtScript, SmtSort};
pub use context::SmtContext;
pub use incremental::IncrementalSolver;
pub use model::{ModelValue, SmtModel};
pub use sexp_parser::SExp;
pub use solver::{ProcessSolver, SmtSolver, SolverConfig, SolverResult};
pub use theories::{arrays, core, qf_lia};
