//! Code generation backends for the Choreo XR Interaction Compiler.
//!
//! Translates compiled [`choreo_automata::automaton::SpatialEventAutomaton`]
//! into executable source code for various target platforms:
//!
//! - **Rust** (`rust_backend`) – standalone `no_std`-compatible state machine
//! - **C#** (`csharp_backend`) – Unity / MRTK `MonoBehaviour`
//! - **TypeScript** (`typescript_backend`) – WebXR session handler
//! - **DOT** (`dot_backend`) – Graphviz visualisation
//! - **JSON** (`json_backend`) – machine-readable import / export
//! - **Template** (`template`) – lightweight template engine used by backends

pub mod csharp_backend;
pub mod dot_backend;
pub mod json_backend;
pub mod rust_backend;
pub mod template;
pub mod typescript_backend;

use choreo_automata::AutomataError;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Shared error & trait
// ---------------------------------------------------------------------------

/// Errors that can occur during code generation.
#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("template error: {0}")]
    Template(String),
    #[error("automaton error: {0}")]
    Automaton(#[from] AutomataError),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid configuration: {0}")]
    Config(String),
}

pub type CodegenResult<T> = Result<T, CodegenError>;

/// Common interface implemented by every code generation backend.
pub trait CodeGenerator {
    /// Generate source text from the given automaton.
    fn generate(
        &self,
        automaton: &choreo_automata::automaton::SpatialEventAutomaton,
    ) -> CodegenResult<String>;

    /// Human-readable backend name (e.g. `"Rust"`, `"C#"`).
    fn name(&self) -> &str;

    /// File extension for the generated output (without leading dot).
    fn file_extension(&self) -> &str;
}
