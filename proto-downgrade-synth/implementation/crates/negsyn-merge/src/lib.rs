//! # negsyn-merge: Protocol-Aware Merge Operator
//!
//! Implements ALG2: PROTOMERGE — the protocol-aware merge operator for
//! symbolic state merging in the NegSynth protocol downgrade synthesis tool.
//!
//! The merge operator exploits algebraic properties of cipher-suite negotiation:
//! - **A1**: Finite outcomes (|C| ≤ 350, |V| ≤ 6, |E| ≤ 30)
//! - **A2**: Lattice preferences (partial order on cipher suites)
//! - **A3**: Monotonic progression (handshake phases form acyclic DAG)
//! - **A4**: Deterministic selection (given fixed offered sets, selection is unique)

pub mod algebraic;
pub mod cache;
pub mod cost;
pub mod fallback;
pub mod lattice;
pub mod operator;
pub mod region;
pub mod symbolic_merge;

pub use algebraic::{
    DeterminismChecker, FiniteOutcomeChecker, LatticeChecker, MonotonicityChecker,
    PropertyChecker, PropertyViolation,
};
pub use cache::{CacheKey, MergeCache, StateSignature};
pub use cost::{AdaptiveMergePolicy, CostBenefitAnalysis, CostEstimator, MergeCost};
pub use fallback::{FallbackDecider, FallbackStatistics, FallbackStrategy, RegionDecomposer};
pub use lattice::{
    AuthStrength, EncryptionStrength, KeyExchangeStrength, MacStrength, PreferenceLattice,
    SecurityLattice, SecurityLevel, SelectionFunction,
};
pub use operator::{MergeContext, MergeOperator, MergeOutput, MergeabilityPredicate, ProtocolMerge};
pub use region::{
    MergeRegion, RegionAnalysis, RegionBoundary, RegionClassifier, RegionClassification,
};
pub use symbolic_merge::{
    ConstraintMerge, MemoryMerge, PhiNodeInsertion, SymbolicMerger, ValueMerge,
};
