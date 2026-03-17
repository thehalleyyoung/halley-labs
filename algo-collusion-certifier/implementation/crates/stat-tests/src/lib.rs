//! Statistical hypothesis testing for the CollusionProof system.
//!
//! This crate implements:
//! - **M1**: Composite hypothesis testing against tiered null hierarchies
//! - **M7**: Directed closed testing with FWER control
//! - Individual price-based, correlation-based, and permutation tests
//! - Bootstrap methods for confidence intervals and p-values
//! - Multiple testing corrections (Bonferroni, Holm, BH-FDR)
//! - Effect size measures and power analysis
//! - Null distribution computation (analytical, simulated, asymptotic)

pub mod effect_size;
pub mod multiple_testing;
pub mod bootstrap;
pub mod null_distribution;
pub mod permutation_tests;
pub mod correlation_tests;
pub mod price_tests;
pub mod closed_testing;
pub mod composite_test;

use shared_types::*;
use serde::{Deserialize, Serialize};

// ── Legacy types (backward-compatible) ──────────────────────────────────────

/// The type of statistical test being performed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TestType {
    PriceCorrelation,
    VarianceRatio,
    MeanReversion,
    DistributionFit,
    Stationarity,
    GrangerCausality,
    StructuralBreak,
    Bootstrap,
    Permutation,
    Custom(String),
}

impl std::fmt::Display for TestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestType::PriceCorrelation => write!(f, "PriceCorrelation"),
            TestType::VarianceRatio => write!(f, "VarianceRatio"),
            TestType::MeanReversion => write!(f, "MeanReversion"),
            TestType::DistributionFit => write!(f, "DistributionFit"),
            TestType::Stationarity => write!(f, "Stationarity"),
            TestType::GrangerCausality => write!(f, "GrangerCausality"),
            TestType::StructuralBreak => write!(f, "StructuralBreak"),
            TestType::Bootstrap => write!(f, "Bootstrap"),
            TestType::Permutation => write!(f, "Permutation"),
            TestType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// Result of a single statistical test in the collusion detection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_type: TestType,
    pub test_name: String,
    pub statistic: f64,
    pub p_value: PValue,
    pub reject_null: bool,
    pub effect_size: Option<f64>,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub sample_size: usize,
    pub alpha_spent: f64,
}

impl TestResult {
    pub fn new(test_type: TestType, test_name: &str, statistic: f64, p_value: f64, alpha: f64) -> Self {
        let pv = PValue::new_unchecked(p_value);
        Self {
            test_type,
            test_name: test_name.to_string(),
            statistic,
            p_value: pv,
            reject_null: pv.is_significant(alpha),
            effect_size: None,
            confidence_interval: None,
            sample_size: 0,
            alpha_spent: alpha,
        }
    }
}

/// Composite result from a battery of tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositTestResult {
    pub results: Vec<TestResult>,
    pub combined_p_value: Option<PValue>,
    pub fwer_control: Option<FWERControl>,
    pub overall_reject: bool,
}

impl CompositTestResult {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            combined_p_value: None,
            fwer_control: None,
            overall_reject: false,
        }
    }
}

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use composite_test::{
    CompositeTest, CompositeTestStatistic, TieredNull, NullHypothesis,
    CompositeDecision, SubTest, SubTestResult,
};
pub use price_tests::{
    SupraCompetitivePriceTest, PriceParallelismTest, PricePersistenceTest,
    ConvergencePatternTest, PriceDispersionTest, EdgeworthCycleTest,
};
pub use correlation_tests::{
    CrossFirmCorrelation, PartialCorrelation, DynamicConditionalCorrelation,
    GrangerCausalityTest, ExcessCorrelationStatistic, CorrelationBootstrap,
};
pub use permutation_tests::{
    PermutationTest, PunishmentPermutationTest, PriceShuffleTest,
    CrossFirmPermutationTest, MonteCarloPermutation, StratifiedPermutation,
};
pub use bootstrap::{
    BootstrapEngine, NonparametricBootstrap, BlockBootstrap,
    ParametricBootstrap, BootstrapCI, CollusionPremiumBootstrap,
    BootstrapPValue, DoubleBootstrap,
};
pub use closed_testing::{
    ClosedTestingProcedure, DirectedClosedTesting, HolmBonferroni,
    Hochberg, HommelProcedure, AlphaBudgetTracker,
    FWERGuarantee, TestRejectionSequence, PowerImprovement,
};
pub use null_distribution::{
    NullDistribution, AnalyticalNull, SimulatedNull, AsymptoticNull,
    BerryEsseenCorrection, CompetitiveSimulator, NullCalibration,
    ConservativeNull,
};
pub use multiple_testing::{
    MultipleTestCorrection, BonferroniCorrection, HolmBonferroniCorrection,
    BenjaminiHochbergFDR, SidakCorrection, FWERControlGuarantee,
    FDRControl, DependenceStructure, EffectiveNumberOfTests,
};
pub use effect_size::{
    CohenD, HedgesG, GlassDelta, PointBiserialR, OddsRatio,
    CollusionEffectSize, PowerAnalysis, EffectSizeCI,
    EffectInterpretation,
};
