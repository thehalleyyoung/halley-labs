//! Corpus-based test input generator for metamorphic testing.
//!
//! Manages seed corpora, coverage-guided selection, and constraint-based
//! test case generation targeting specific transformation coverage goals.

pub mod coverage;
pub mod seed_corpus;
pub mod selector;
pub mod generator;

pub use coverage::{CoverageGoal, CoverageTracker, TransformationCoverage};
pub use seed_corpus::{SeedCorpus, SeedSentence, AnnotatedSeed, CorpusStats};
pub use selector::{SeedSelector, SelectionStrategy, SelectionResult};
pub use generator::{TestGenerator, GeneratorConfig, GeneratedTestCase};
