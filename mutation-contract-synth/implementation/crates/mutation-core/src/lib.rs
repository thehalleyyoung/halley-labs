//! # mutation-core
//!
//! Core mutation testing engine for MutSpec. Implements mutation operators,
//! mutant generation, execution, kill-matrix analysis, replay, and statistics
//! for loop-free QF-LIA programs.
//!
//! ## Modules
//!
//! - [`operators`] – Mutation operator trait, registry, and concrete operators
//!   (AOR, ROR, LCR, UOI).
//! - [`mutant`] – Mutant representation, sets, filtering, diff, and equivalence.
//! - [`kill_matrix`] – 2-D kill matrix with sparse storage, statistics, and set-cover.
//! - [`execution`] – Interpreter-based mutant execution engine with parallel support.
//! - [`replay`] – Source-level mutation replay from bytecode descriptors.
//! - [`statistics`] – Aggregated mutation-testing statistics and reports.

pub mod execution;
pub mod kill_matrix;
pub mod mutant;
pub mod operators;
pub mod replay;
pub mod statistics;

// Re-exports for convenient access.
pub use execution::{ExecutionEngine, ExecutionResult, MutantExecutor, TestResult};
pub use kill_matrix::KillMatrix;
pub use mutant::{Mutant, MutantDiff, MutantFilter, MutantSet};
pub use operators::{create_standard_operators, MutationOperatorTrait, OperatorRegistry};
pub use replay::MutationReplay;
pub use statistics::MutationStatistics;
