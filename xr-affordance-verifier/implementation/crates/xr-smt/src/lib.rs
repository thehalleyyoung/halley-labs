//! SMT encoding for linearized kinematics verification (QF_LRA).
//!
//! This crate implements the SMT (Satisfiability Modulo Theories) encoding
//! layer for the XR Affordance Verifier. It converts accessibility predicates
//! into quantifier-free linear real arithmetic (QF_LRA) formulas and solves
//! them to verify reachability of interactable elements across body parameter
//! populations.
//!
//! # Architecture
//!
//! The verification pipeline proceeds as:
//! 1. **Linearization** (`linearization`): Taylor-approximate FK at reference config
//! 2. **Encoding** (`encoder`): Convert accessibility predicates to SMT expressions
//! 3. **Constraint management** (`constraints`): Normalize and manage constraint sets
//! 4. **QF_LRA theory** (`qf_lra`): Linear arithmetic feasibility checking
//! 5. **Solving** (`solver`): DPLL(T)-style SAT+Theory integration
//! 6. **Verification** (`verification`): Orchestrate region-based verification
//! 7. **Proof** (`proof`): Generate verification certificates
//! 8. **Optimization** (`optimization`): Find boundary cases and allocate budgets

pub mod expr;
pub mod encoder;
pub mod linearization;
pub mod solver;
pub mod constraints;
pub mod qf_lra;
pub mod verification;
pub mod proof;
pub mod optimization;

pub use expr::{SmtExpr, SmtSort, SmtDecl};
pub use encoder::AccessibilityEncoder;
pub use linearization::{LinearizationEngine, LinearizedModel};
pub use solver::{SmtSolver, SolverResult, InternalSolver, ExternalSolverInterface};
pub use constraints::{
    ConstraintSet, BoundedVariable, ConstraintNormalizer,
    BoundPropagator, PropagationResult, VariableScope,
};
pub use qf_lra::{LinearConstraint, FeasibilityChecker};
pub use verification::{SmtVerifier, RegionVerdict};
pub use proof::{SmtProof, ProofStep, ProofCertificate};
pub use optimization::{SmtOptimizer, BudgetAllocator};
