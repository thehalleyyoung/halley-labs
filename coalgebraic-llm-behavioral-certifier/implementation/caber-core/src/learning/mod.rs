//! Learning module — active automata learning for coalgebraic behavioral models.
//!
//! This module implements the Probabilistic Coalgebraic L* (PCL*) algorithm
//! and its supporting infrastructure for learning behavioral automata from
//! black-box LLM interactions.
//!
//! # Architecture
//!
//! - [`observation_table`]: Angluin-style observation table generalized to coalgebras
//! - [`pcl_star`]: Main PCL* learning algorithm
//! - [`query_oracle`]: Query oracle abstractions and implementations
//! - [`hypothesis`]: Hypothesis automaton construction and refinement
//! - [`convergence`]: Convergence analysis and PAC bounds
//! - [`active_learning`]: Generic active learning framework
//! - [`counterexample`]: Counter-example processing algorithms

pub mod observation_table;
pub mod pcl_star;
pub mod query_oracle;
pub mod hypothesis;
pub mod convergence;
pub mod active_learning;
pub mod counterexample;

pub use observation_table::{ObservationTable, TableEntry, ClosednessResult, ConsistencyResult};
pub use pcl_star::{PCLStar, PCLStarConfig, LearningResult, LearningStats};
pub use query_oracle::{
    QueryOracle, StatisticalMembershipOracle, ApproximateEquivalenceOracle,
    CachedOracle, BatchOracle, MockOracle, StochasticOracle,
    MembershipQuery, MembershipResult, EquivalenceQuery, EquivalenceResult,
};
pub use hypothesis::{HypothesisAutomaton, HypothesisState, HypothesisTransition};
pub use convergence::{
    ConvergenceAnalyzer, ConvergenceStatus, PACBounds, SampleComplexity,
    DriftDetector, ConfidenceInterval,
};
pub use active_learning::{
    ActiveLearner, TeacherLearnerProtocol, QuerySelector, LearningCurve,
    IncrementalLearner,
};
pub use counterexample::{
    CounterExample, CounterExampleProcessor, DecompositionMethod,
    CounterExampleCache,
};
