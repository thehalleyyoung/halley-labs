//! guardpharma-guideline-parser — parse clinical practice guidelines into
//! formal automata representations for polypharmacy verification.
//!
//! This crate provides:
//! - Structured guideline document formats ([`format`])
//! - JSON / YAML parsing ([`parser`])
//! - Priced Timed Automata construction ([`pta_builder`])
//! - Guard expression compilation & optimisation ([`guard_compiler`])
//! - Ready-made clinical guideline templates ([`template`])
//! - Multi-guideline composition & conflict detection ([`composition`])
//! - Static validation of guideline documents ([`validation`])

pub mod format;
pub mod parser;
pub mod pta_builder;
pub mod guard_compiler;
pub mod template;
pub mod composition;
pub mod validation;

// Re-export key entry points for convenience.
pub use format::GuidelineDocument;
pub use parser::GuidelineParser;
pub use pta_builder::PtaBuilder;
pub use guard_compiler::GuardCompiler;
pub use template::GuidelineTemplate;
pub use composition::GuidelineComposer;
pub use validation::GuidelineValidator;
