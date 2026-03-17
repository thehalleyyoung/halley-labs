//! `negsyn-types` — Core types, traits, and error handling for protocol
//! downgrade synthesis.
//!
//! This is the foundational crate for the NegSynth pipeline. It defines
//! the type vocabulary shared across slicing, merging, extraction, encoding,
//! and concretization phases.

pub mod adversary;
pub mod certificate;
pub mod config;
pub mod display;
pub mod error;
pub mod graph;
pub mod metrics;
pub mod protocol;
pub mod smt;
pub mod symbolic;

pub use error::{NegSynthError, NegSynthResult, MergeError};

/// Type alias for merge results.
pub type MergeResult<T> = Result<T, MergeError>;

pub use protocol::{
    AuthAlgorithm, CipherSuite, CipherSuiteRegistry, EncryptionAlgorithm, Extension,
    HandshakePhase, KeyExchange, MacAlgorithm, NegotiationLTS, NegotiationOutcome,
    NegotiationState, ProtocolFamily, ProtocolVersion, SecurityLevel, TransitionLabel,
};

/// Type alias: `BulkEncryption` is the same as `EncryptionAlgorithm`.
pub type BulkEncryption = EncryptionAlgorithm;

pub use adversary::{
    AdversaryAction, AdversaryBudget, AdversaryTrace, BoundedDYAdversary, DowngradeInfo,
    KnowledgeSet, MessageTerm,
};

pub use symbolic::{
    BinOp, ConcreteValue, ExecutionTree, MemoryRegion, MemoryPermissions, MergeableState,
    PathConstraint, SymSort, SymbolicMemory, SymbolicState, SymbolicValue, UnOp,
};

/// Type alias for symbolic state identifiers.
pub type SymbolicId = u64;

pub use smt::{SmtExpr, SmtFormula, SmtModel, SmtProof, SmtResult, SmtSort, SmtValue};

pub use certificate::{
    AnalysisResult, AttackTrace, BoundsSpec, Certificate, CertificateChain, CertificateValidity,
    LibraryIdentifier,
};

pub use graph::{
    BisimulationRelation, QuotientGraph, StateData, StateGraph, StateId, TransitionId,
};

pub use config::{
    AnalysisConfig, ConcretizerConfig, EncodingConfig, ExtractionConfig, MergeConfig,
    MergeStrategy, ProtocolConfig, SlicerConfig,
};

pub use metrics::{
    AnalysisMetrics, CoverageMetrics, MergeStatistics, MetricReport, PerformanceMetrics,
    PhaseTimer,
};

/// Alias: `MergeMetrics` maps to `MergeStatistics`.
pub type MergeMetrics = MergeStatistics;

pub use display::Summary;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_exports_accessible() {
        let _phase = HandshakePhase::Init;
        let _level = SecurityLevel::High;
        let _version = ProtocolVersion::tls13();
        let _budget = AdversaryBudget::standard();
        let _expr = SmtExpr::BoolConst(true);
        let _val = SymbolicValue::bool_const(true);
        let _sort = SmtSort::Bool;
    }

    #[test]
    fn test_error_result_alias() {
        fn example_fn() -> NegSynthResult<u32> {
            Ok(42)
        }
        assert_eq!(example_fn().unwrap(), 42);
    }

    #[test]
    fn test_cross_module_integration() {
        let init = NegotiationState::initial();
        let mut lts = NegotiationLTS::new(init);

        let mut hello = NegotiationState::initial();
        hello.phase = HandshakePhase::ClientHelloSent;
        hello.offered_ciphers = vec![
            CipherSuiteRegistry::lookup(0x1301).unwrap(),
            CipherSuiteRegistry::lookup(0x0005).unwrap(),
        ];
        let s1 = lts.add_state(hello);

        let label = TransitionLabel::ClientAction(protocol::ClientActionKind::SendClientHello {
            ciphers: vec![0x1301, 0x0005],
            version: 0x0304,
        });
        lts.add_transition(0, label, s1);

        assert_eq!(lts.size(), (2, 1));

        let strong = CipherSuiteRegistry::lookup(0x1301).unwrap();
        let weak = CipherSuiteRegistry::lookup(0x0005).unwrap();
        let downgrade = adversary::DowngradeChecker::check_cipher_downgrade(
            &[strong.clone(), weak.clone()],
            &weak,
        );
        assert!(downgrade.is_some());
    }
}
