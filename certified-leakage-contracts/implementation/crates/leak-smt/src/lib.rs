//! SMT-based contract verification engine for certified leakage contracts.
//!
//! This crate provides an SMT (Satisfiability Modulo Theories) backend for
//! verifying information leakage contracts on x86-64 cryptographic binaries.
//! It generates SMT-LIB2 formulas and can interface with Z3/CVC5 solvers
//! via command-line, or provide self-contained verification for simpler cases.

pub mod expr;
pub mod smtlib;
pub mod solver;
pub mod theories;
pub mod encoding;
pub mod verification;
pub mod optimize;

pub use expr::{Expr, ExprId, ExprPool, Sort, Value};
pub use smtlib::{SmtLib2Writer, SmtCommand, Script, SmtResponse};
pub use solver::{
    SmtSolver, SolverConfig, SolverBackend, SolverResult, Model,
    ExternalSolver, InternalSolver,
};
pub use theories::CacheTheory;
pub use encoding::LeakageEncoder;
pub use verification::{ContractVerifier, VerificationResult, VerificationReport};
pub use optimize::OptimizeSolver;
