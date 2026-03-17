//! Quantitative Information Flow (QIF) analysis for cache side-channel leakage.
//!
//! This crate extends traditional information flow control with precise bit-level
//! measurements of information leakage through cache side channels. It provides:
//!
//! - **Entropy computations**: Shannon, min-entropy, guessing entropy, max-leakage
//! - **Channel models**: Discrete memoryless channels for cache observations
//! - **Counting abstractions**: Counting distinguishable cache states
//! - **Probability distributions**: Discrete distributions with standard operations
//! - **Leakage models**: Cache timing, access pattern, and speculative models
//! - **Leakage bounds**: Per-function and compositional bound computation
//! - **Metrics**: Bits leaked, vulnerability score, guessing advantage

pub mod distribution;
pub mod entropy;
pub mod channel;
pub mod counting;
pub mod leakage_model;
pub mod bounds;
pub mod metrics;

pub use distribution::{
    Distribution, UniformDistribution, PointDistribution,
    ConditionalDistribution, JointDistribution,
};
pub use entropy::{
    ShannonEntropy, MinEntropy, GuessingEntropy, MaxLeakage,
    ConditionalEntropy, MutualInformation, EntropyBound,
};
pub use channel::{
    Channel, CacheChannel, ChannelCapacity, ChannelMatrix,
    ObservationSet, CacheObservation,
};
pub use counting::{
    CountingDomain, TaintRestrictedCounting, SetCounting,
    DistinguishableStates, CountBound,
};
pub use leakage_model::{
    LeakageModel, CacheTimingModel, AccessPatternModel,
    PowerModel, ComposedModel, SpeculativeLeakageModel,
};
pub use bounds::{
    LeakageBound, BoundComputation, PerFunctionBound,
    CompositionalBound, WholeLibraryBound,
};
pub use metrics::{
    LeakageMetric, BitsLeaked, MultiplicativeLeakage,
    VulnerabilityScore, GuessingAdvantage,
};

use thiserror::Error;

/// Errors that can occur during quantitative leakage analysis.
#[derive(Debug, Error)]
pub enum QuantifyError {
    #[error("invalid probability distribution: {0}")]
    InvalidDistribution(String),

    #[error("channel dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("convergence failure after {iterations} iterations (residual={residual:.2e})")]
    ConvergenceFailure { iterations: usize, residual: f64 },

    #[error("numerical instability: {0}")]
    NumericalInstability(String),

    #[error("empty support set")]
    EmptySupport,

    #[error("invalid cache configuration: {0}")]
    InvalidCacheConfig(String),

    #[error("bound computation failed: {0}")]
    BoundComputationFailed(String),

    #[error("leakage model error: {0}")]
    ModelError(String),

    #[error("metric conversion error: {0}")]
    MetricConversion(String),
}

pub type QuantifyResult<T> = Result<T, QuantifyError>;
