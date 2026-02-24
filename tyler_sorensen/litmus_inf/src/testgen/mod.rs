//! Test generation for LITMUS∞.
//!
//! Provides systematic, template-based, constraint-based, random,
//! coverage-guided, and mutation-based litmus test generation,
//! plus a catalog of standard litmus test patterns.

pub mod generator;
pub mod mutation;
pub mod coverage;
pub mod catalog;

pub use generator::{
    TestGenerator, GeneratorConfig, GenerationStrategy,
    TemplateSpec, ConstraintSpec, TestFamily,
};
pub use mutation::{
    MutationOperator, MutationEngine, MutationConfig, MutationResult,
};
pub use coverage::{
    CoverageMetric, CoverageTracker, CoverageConfig, CoverageReport,
    AxiomCoverage, RelationCoverage, PatternCoverage,
};
pub use catalog::{
    TestCatalog, PatternKind, CatalogEntry,
};

pub mod equivalence;
