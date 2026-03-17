//! Shared types for the CollusionProof algorithmic collusion certification system.
//!
//! This crate provides all shared types, error handling, and utilities used
//! across the entire workspace. Every other crate depends on this one.

pub mod config;
pub mod errors;
pub mod evidence;
pub mod identifiers;
pub mod interval;
pub mod rational;
pub mod serialization;
pub mod statistics;
pub mod types;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use types::{
    AlgorithmConfig, AlgorithmType, CollusionIndex, Cost, DemandSystem, EvaluationMode,
    GameConfig, HoldoutSegment, MarketOutcome, MarketType, OracleAccessLevel, PlayerAction,
    PlayerId, Price, PriceGrid, PriceTrajectory, Profit, Quantity, RoundNumber, SimulationConfig,
    TestingSegment, TrainingSegment, TrajectorySegment, ValidationSegment,
};

pub use errors::{CollusionError, CollusionResult, ErrorContext};

pub use interval::{Interval, IntervalComparison, IntervalF64};

pub use rational::{DualPath, DualPathVerifier, RationalNum};

pub use identifiers::{BundleId, CertificateId, ScenarioId, SimulationId, TestId};

pub use config::{
    BootstrapConfig, ConfidenceLevel, GlobalConfig, MonteCarloConfig, SignificanceLevel,
    TieredNullConfig,
};

pub use statistics::{
    AlphaBudget, BootstrapResult, ConfidenceInterval, EffectSize, FWERControl,
    HypothesisTestResult, PValue, PermutationResult, TestBattery, TestStatistic,
};

pub use evidence::{EvidenceBundle, EvidenceItem, EvidenceStrength, MerkleNode, MerkleTree};
