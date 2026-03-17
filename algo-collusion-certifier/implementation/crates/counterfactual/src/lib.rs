//! Counterfactual deviation analysis for the CollusionProof system.
//!
//! This crate implements the counterfactual analysis pipeline:
//! - **M2**: Black-box deviation oracle (checkpoint and full-rewind)
//! - **M3**: Punishment detection via controlled perturbation
//! - Deviation strategy enumeration and optimization
//! - Counterfactual re-simulation with variance reduction
//! - Importance sampling for efficient estimation
//! - Sensitivity analysis across parameter space

pub(crate) mod market_helper;

pub mod deviation;
pub mod oracle;
pub mod punishment;
pub mod resimulation;
pub mod counterfactual_analysis;
pub mod importance_sampling;
pub mod sensitivity;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use deviation::{
    DeviationStrategy, DeviationEnumerator, DeviationEnumeratorConfig,
    OptimalDeviation, DeviationBound, DeviationResult, MultiPeriodDeviation,
    DeviationProfile, ProfitableDeviation, DeviationCatalog,
};

pub use oracle::{
    DeviationOracle, Layer1Oracle, Layer2Oracle, AdaptiveRefinement,
    CheckpointSchedule, OracleQueryBudget, CertifiedBound, QueryEfficiency,
};

pub use punishment::{
    PunishmentDetector, PunishmentDetectorConfig, ControlledPerturbation,
    PunishmentTest, PunishmentMetrics, PunishmentClassification,
    InjectionSchedule, PunishmentEvidence,
};

pub use resimulation::{
    ResimulationEngine, CounterfactualScenario, ResimulationResult,
    ParallelResimulation, TruncatedHorizon, VarianceReduction,
    ResimulationBudget, ResimulationCheckpoint,
};

pub use counterfactual_analysis::{
    CounterfactualAnalyzer, CounterfactualAnalyzerConfig,
    CompareFactualCounterfactual, SelfEnforcingCheck,
    IncentiveCompatibility, CounterfactualReport,
};

pub use importance_sampling::{
    ImportanceSampler, ProposalDistribution, ImportanceWeight,
    EffectiveSampleSize, SelfNormalizedEstimator,
    StratifiedImportanceSampling, ImportanceSamplingCI,
};

pub use sensitivity::{
    SensitivityAnalyzer, ParameterSweep, SweepResult,
    LatinHypercubeSampling, SobolIndices,
    SensitivityReport, RobustnessScore,
};
