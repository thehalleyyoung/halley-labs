//! Temporal logic module for CABER — QCTL_F specification and model checking.
//!
//! This module provides the quantitative coalgebraic temporal logic (QCTL_F)
//! used for specifying behavioral properties of LLMs:
//!
//! - `syntax`: AST, parsing, pretty-printing, simplification of QCTL_F formulas
//! - `templates`: Human-readable specification templates compiled to QCTL_F
//! - `semantics`: Denotational semantics on finite probabilistic coalgebras
//! - `predicates`: Predicate liftings for the LLM behavioral functor

pub mod syntax;
pub mod templates;
pub mod semantics;
pub mod predicates;

pub use syntax::{
    Formula, StateFormula, PathFormula, QuantFormula,
    BoolOp, TemporalOp, PathQuantifier, ComparisonOp,
    FormulaParser, FormulaPrinter, FormulaSimplifier,
    FormulaInfo,
};
pub use templates::{
    SpecTemplate, TemplateKind, TemplateParam,
    RefusalPersistence, ParaphraseInvariance, VersionStability,
    SycophancyResistance, InstructionHierarchy, JailbreakResistance,
    TemplateComposer, CompositionOp,
};
pub use semantics::{
    SatisfactionDegree, SemanticEvaluator, FixedPointComputer,
    KripkeStructure, KripkeState, Transition,
};
pub use predicates::{
    Predicate, AtomicPredicate, PredicateKind,
    PredicateEvaluator, PredicateAbstraction,
    CompoundPredicate, CompoundOp,
};
