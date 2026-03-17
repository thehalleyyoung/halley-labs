//! # negsyn-slicer
//!
//! Protocol-aware program slicer for negotiation code extraction.
//!
//! Implements **ALG1: PROTOSLICE** — the core slicing algorithm that extracts
//! negotiation-relevant code from TLS/SSH protocol library source (LLVM IR).
//!
//! ## Architecture
//!
//! The slicer pipeline operates in phases:
//!
//! 1. **IR Parsing** (`ir`): Load LLVM IR into an in-memory representation
//! 2. **Call Graph** (`callgraph`): Build inter-procedural call graph with indirect call resolution
//! 3. **CFG** (`cfg`): Construct control flow graphs, dominator trees, control dependence
//! 4. **Points-To** (`points_to`): Andersen/Steensgaard pointer analysis for vtable resolution
//! 5. **Taint** (`taint`): Forward/backward taint propagation from negotiation fields
//! 6. **Dependency** (`dependency`): Program dependence graph (data + control)
//! 7. **VTable** (`vtable`): SSL_METHOD vtable and callback chain analysis
//! 8. **Slice** (`slice`): Core PROTOSLICE algorithm combining all analyses
//! 9. **Validation** (`validation`): Slice completeness and CVE reachability checking
//!
//! ## Usage
//!
//! ```ignore
//! use negsyn_slicer::{ir, slice, taint, callgraph};
//!
//! let module = ir::Module::from_bitcode("path/to/libssl.bc")?;
//! let cg = callgraph::CallGraphBuilder::new(&module).build();
//! let slicer = slice::ProtocolAwareSlicer::new(&module, &cg);
//! let criterion = slice::SliceCriterion::negotiation_outcome("SSL_do_handshake");
//! let program_slice = slicer.slice(&criterion)?;
//! ```

pub mod ir;
pub mod taint;
pub mod points_to;
pub mod slice;
pub mod callgraph;
pub mod cfg;
pub mod dependency;
pub mod vtable;
pub mod validation;

// Re-exports for convenience
pub use ir::{Module, Function, BasicBlock, Instruction, Value, Type as IrType};
pub use taint::{TaintAnalysis, TaintSource, TaintTag, TaintState};
pub use points_to::{PointsToSet, AbstractLocation, PointsToGraph, AndersonAnalysis};
pub use slice::{SliceCriterion, ProgramSlice, ProtocolAwareSlicer};
pub use callgraph::{CallGraph, CallSite, CallGraphBuilder};
pub use cfg::{CFG, DominatorTree, PostDominatorTree, ControlDependence};
pub use dependency::{DependencyGraph, ProgramDependenceGraph};
pub use vtable::{VTableLayout, VTableResolver, CallbackAnalysis};
pub use validation::{SliceValidator, CoverageChecker, ValidationReport};

use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Errors specific to the slicer crate.
#[derive(Debug, Error)]
pub enum SlicerError {
    #[error("IR parse error: {0}")]
    IrParseError(String),
    #[error("Analysis error in {phase}: {message}")]
    AnalysisError { phase: String, message: String },
    #[error("Slice criterion not satisfiable: {0}")]
    UnsatisfiableCriterion(String),
    #[error("Points-to analysis diverged after {iterations} iterations")]
    PointsToTimeout { iterations: usize },
    #[error("Taint analysis failed: {0}")]
    TaintError(String),
    #[error("Call graph construction error: {0}")]
    CallGraphError(String),
    #[error("CFG error: {0}")]
    CfgError(String),
    #[error("Validation failed: {reasons:?}")]
    ValidationFailed { reasons: Vec<String> },
    #[error("VTable resolution error: {0}")]
    VTableError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for slicer operations.
pub type SlicerResult<T> = Result<T, SlicerError>;

/// Identifier for an instruction within a module.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstructionId {
    pub function: String,
    pub block: String,
    pub index: usize,
}

impl InstructionId {
    pub fn new(function: impl Into<String>, block: impl Into<String>, index: usize) -> Self {
        Self {
            function: function.into(),
            block: block.into(),
            index,
        }
    }
}

impl std::fmt::Display for InstructionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}[{}]", self.function, self.block, self.index)
    }
}

/// Protocol negotiation phase markers used throughout the slicer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NegotiationPhase {
    ClientHello,
    ServerHello,
    CipherSuiteSelection,
    VersionNegotiation,
    ExtensionProcessing,
    CertificateHandling,
    KeyExchange,
    Finished,
}

impl NegotiationPhase {
    /// Return the canonical function prefixes associated with this phase.
    pub fn function_prefixes(&self) -> &[&str] {
        match self {
            Self::ClientHello => &["ssl_client_hello", "tls_construct_client_hello", "SSH_MSG_KEXINIT"],
            Self::ServerHello => &["ssl_server_hello", "tls_process_server_hello"],
            Self::CipherSuiteSelection => &["ssl_cipher", "tls_choose_cipher", "ssl3_choose_cipher"],
            Self::VersionNegotiation => &["ssl_version", "tls_version", "ssl_set_version"],
            Self::ExtensionProcessing => &["tls_parse_ext", "tls_construct_ext", "ssl_parse_ext"],
            Self::CertificateHandling => &["ssl_cert", "tls_process_cert"],
            Self::KeyExchange => &["tls_process_key_exchange", "ssl_key_exchange", "kex_"],
            Self::Finished => &["tls_process_finished", "ssl_finished"],
        }
    }

    /// Check whether a function name is relevant to this phase.
    pub fn matches_function(&self, func_name: &str) -> bool {
        self.function_prefixes().iter().any(|prefix| func_name.contains(prefix))
    }
}

/// Configuration for the slicer pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicerConfig {
    /// Maximum iterations for fixed-point analyses.
    pub max_iterations: usize,
    /// Whether to use context-sensitive points-to analysis.
    pub context_sensitive: bool,
    /// Maximum call depth for interprocedural analysis.
    pub max_call_depth: usize,
    /// Whether to track taint through memory (heap/globals).
    pub taint_through_memory: bool,
    /// Negotiation phases to include in slicing criterion.
    pub target_phases: Vec<NegotiationPhase>,
    /// Known protocol function patterns for relevance checking.
    pub protocol_patterns: Vec<String>,
    /// Whether to apply slice minimization.
    pub minimize_slice: bool,
    /// Timeout in seconds for the whole pipeline.
    pub timeout_secs: u64,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            context_sensitive: true,
            max_call_depth: 10,
            taint_through_memory: true,
            target_phases: vec![
                NegotiationPhase::CipherSuiteSelection,
                NegotiationPhase::VersionNegotiation,
                NegotiationPhase::ExtensionProcessing,
            ],
            protocol_patterns: vec![
                "SSL_*".into(), "ssl_*".into(), "tls_*".into(), "TLS_*".into(),
                "kex_*".into(), "SSH_*".into(),
            ],
            minimize_slice: true,
            timeout_secs: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_id() {
        let id = InstructionId::new("SSL_do_handshake", "entry", 3);
        assert_eq!(id.function, "SSL_do_handshake");
        assert_eq!(id.block, "entry");
        assert_eq!(id.index, 3);
        assert_eq!(id.to_string(), "SSL_do_handshake::entry[3]");
    }

    #[test]
    fn test_negotiation_phase_matching() {
        assert!(NegotiationPhase::CipherSuiteSelection.matches_function("ssl3_choose_cipher_v2"));
        assert!(NegotiationPhase::VersionNegotiation.matches_function("ssl_set_version_bound"));
        assert!(!NegotiationPhase::ClientHello.matches_function("ssl_read_bytes"));
    }

    #[test]
    fn test_slicer_config_default() {
        let cfg = SlicerConfig::default();
        assert_eq!(cfg.max_iterations, 1000);
        assert!(cfg.context_sensitive);
        assert_eq!(cfg.target_phases.len(), 3);
    }

    #[test]
    fn test_slicer_error_display() {
        let err = SlicerError::AnalysisError {
            phase: "taint".into(),
            message: "diverged".into(),
        };
        assert!(err.to_string().contains("taint"));
        assert!(err.to_string().contains("diverged"));
    }

    #[test]
    fn test_negotiation_phase_prefixes() {
        let phase = NegotiationPhase::KeyExchange;
        let prefixes = phase.function_prefixes();
        assert!(prefixes.contains(&"kex_"));
    }
}
