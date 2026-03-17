//! Core detection engine for the CollusionProof algorithmic collusion certification system.
//!
//! This crate integrates market simulation with algorithm interfaces, providing:
//! - Black-box algorithm interface (`PricingAlgorithm` trait)
//! - Q-learning, DQN, grim trigger, tit-for-tat, and bandit implementations
//! - Collusion detection pipeline (Layer 0/1/2)
//! - Evaluation scenario definitions
//! - End-to-end certification pipeline

pub mod algorithm;
pub mod asymmetric;
pub mod bandit;
pub mod detector;
pub mod dqn;
pub mod grim_trigger;
pub mod pipeline;
pub mod q_learning;
pub mod scenario;
pub mod tit_for_tat;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use algorithm::{
    AlgorithmFactory, AlgorithmMetrics, AlgorithmSandbox, AlgorithmState, BatchOracle,
    CheckpointOracle, OracleInterface, PassiveOracle, PricingAlgorithm, RewindOracle,
    SandboxConfig, SandboxedExecution,
};

pub use q_learning::{DecaySchedule, QLearningAgent, QLearningConfig};

pub use grim_trigger::{ForgivingGrimTrigger, GradualTrigger, GrimTriggerAgent, GrimTriggerConfig};

pub use tit_for_tat::{
    BoundedMemoryTFT, GenerousTitForTat, SuspiciousTitForTat, TitForTatAgent, TitForTwoTats,
};

pub use dqn::{ActivationFn, DQNAgent, DQNConfig, SimpleNeuralNetwork};

pub use bandit::{
    EXP3Bandit, EpsilonGreedyBandit, ThompsonSamplingBandit, UCB1Bandit,
};

pub use detector::{
    CollusionDetector, CollusionReport, DetectionConfig, DetectionPipeline, DetectionResult,
    Verdict,
};

pub use scenario::{Scenario, ScenarioGenerator, ScenarioLibrary};

pub use pipeline::{
    CertificationPipeline, CertificationResult, PipelineConfig, PipelineStage,
};

pub use asymmetric::{
    AsymmetricAgent, AsymmetricCollusionCertificate, AsymmetricGame, AsymmetricGameType,
    AsymmetricNashEquilibrium, AsymmetricPayoffMatrix, BenchmarkResult,
    asymmetric_auction_example, asymmetric_bertrand_example, asymmetric_cournot_example,
    certify_asymmetric_coalition, find_nash_equilibrium, run_asymmetric_benchmarks,
    print_benchmark_results,
};
