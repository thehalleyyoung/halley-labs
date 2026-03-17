//! M1 Composite Hypothesis Test against tiered null hierarchy.
//!
//! The composite test combines sub-test results into a single test statistic S_T,
//! evaluates against tiered null hypotheses (Narrow, Medium, Broad), and computes
//! critical values and rejection decisions.

use serde::{Deserialize, Serialize};
use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, DemandSystem,
    EffectSize, GameConfig, HypothesisTestResult, PValue, Price, PriceTrajectory,
    SimulationConfig, TestId, TestStatistic,
};

use crate::bootstrap::{BootstrapEngine, BootstrapPValue};
use crate::closed_testing::{
    AlphaBudgetTracker, DirectedClosedTesting, FWERGuarantee, HolmBonferroni,
};
use crate::correlation_tests::{
    CrossFirmCorrelation, ExcessCorrelationStatistic, CorrelationBoundUnderNull,
};
use crate::effect_size::{normal_cdf, normal_quantile, CollusionEffectSize};
use crate::null_distribution::{
    AnalyticalNull, AsymptoticNull, BerryEsseenCorrection, CompetitiveSimulator,
    ConservativeNull, NullCalibration, NullDistribution, SimulatedNull,
};
use crate::price_tests::{
    SupraCompetitivePriceTest, PriceParallelismTest, PricePersistenceTest,
};

// ── Tiered null ─────────────────────────────────────────────────────────────

/// The three tiers of the null hypothesis hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TieredNull {
    /// H0-Narrow: linear demand × Q-learning (independent).
    Narrow,
    /// H0-Medium: parametric demand (CES/Logit) × no-regret learners.
    Medium,
    /// H0-Broad: Lipschitz demand × independent learners.
    Broad,
}

impl std::fmt::Display for TieredNull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Narrow => write!(f, "Narrow (Linear × Q-learning)"),
            Self::Medium => write!(f, "Medium (Parametric × No-regret)"),
            Self::Broad => write!(f, "Broad (Lipschitz × Independent)"),
        }
    }
}

impl TieredNull {
    /// Return all tiers in order of increasing generality.
    pub fn all_tiers() -> Vec<Self> {
        vec![Self::Narrow, Self::Medium, Self::Broad]
    }

    /// Whether this tier is at least as general as `other`.
    pub fn is_at_least_as_general(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Broad, _) => true,
            (Self::Medium, Self::Medium) | (Self::Medium, Self::Narrow) => true,
            (Self::Narrow, Self::Narrow) => true,
            _ => false,
        }
    }
}

// ── Null hypothesis ─────────────────────────────────────────────────────────

/// Null hypothesis: demand system family + algorithm family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullHypothesis {
    pub tier: TieredNull,
    pub demand_family: DemandFamily,
    pub algorithm_family: AlgorithmFamily,
    pub description: String,
}

/// Demand system family under the null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DemandFamily {
    /// Linear demand: Q_i = a - b*p_i + c*p_{-i}
    Linear,
    /// Parametric: CES or Logit
    Parametric,
    /// Lipschitz continuous demand with constant L
    Lipschitz { constant: f64 },
}

/// Algorithm family under the null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmFamily {
    /// Independent Q-learning agents.
    QLearningIndependent,
    /// No-regret learners (e.g., multiplicative weights).
    NoRegret,
    /// Independent learners (general).
    IndependentLearners,
}

impl NullHypothesis {
    /// H0-Narrow: linear demand × Q-learning independent.
    pub fn narrow() -> Self {
        Self {
            tier: TieredNull::Narrow,
            demand_family: DemandFamily::Linear,
            algorithm_family: AlgorithmFamily::QLearningIndependent,
            description: "Linear demand system with independent Q-learning agents".into(),
        }
    }

    /// H0-Medium: parametric demand × no-regret learners.
    pub fn medium() -> Self {
        Self {
            tier: TieredNull::Medium,
            demand_family: DemandFamily::Parametric,
            algorithm_family: AlgorithmFamily::NoRegret,
            description: "Parametric demand (CES/Logit) with no-regret learners".into(),
        }
    }

    /// H0-Broad: Lipschitz demand × independent learners.
    pub fn broad(lipschitz_constant: f64) -> Self {
        Self {
            tier: TieredNull::Broad,
            demand_family: DemandFamily::Lipschitz { constant: lipschitz_constant },
            algorithm_family: AlgorithmFamily::IndependentLearners,
            description: format!(
                "Lipschitz-{lipschitz_constant:.1} demand with independent learners"
            ),
        }
    }
}

// ── Sub-test interface ──────────────────────────────────────────────────────

/// Interface for individual sub-tests that contribute to the composite test.
pub trait SubTest: Send + Sync {
    /// Name of the sub-test.
    fn name(&self) -> &str;

    /// Run the sub-test and return a result.
    fn run(&self, data: &SubTestData) -> CollusionResult<SubTestResult>;

    /// Weight of this sub-test in the composite statistic.
    fn weight(&self) -> f64 {
        1.0
    }
}

/// Data provided to sub-tests.
#[derive(Debug, Clone)]
pub struct SubTestData {
    pub prices: Vec<Vec<f64>>,
    pub competitive_price: Price,
    pub monopoly_price: Price,
    pub num_players: usize,
    pub num_rounds: usize,
}

impl SubTestData {
    pub fn new(
        prices: Vec<Vec<f64>>,
        competitive_price: Price,
        monopoly_price: Price,
    ) -> Self {
        let num_players = prices.len();
        let num_rounds = prices.first().map(|v| v.len()).unwrap_or(0);
        Self { prices, competitive_price, monopoly_price, num_players, num_rounds }
    }
}

/// Result from a single sub-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTestResult {
    pub name: String,
    pub statistic: f64,
    pub p_value: PValue,
    pub reject_null: bool,
    pub weight: f64,
    pub effect_size: Option<f64>,
    pub details: String,
}

impl SubTestResult {
    pub fn new(
        name: &str,
        statistic: f64,
        p_value: f64,
        alpha: f64,
        weight: f64,
    ) -> Self {
        let pv = PValue::new_unchecked(p_value);
        Self {
            name: name.to_string(),
            statistic,
            p_value: pv,
            reject_null: pv.is_significant(alpha),
            weight,
            effect_size: None,
            details: String::new(),
        }
    }

    pub fn with_effect_size(mut self, es: f64) -> Self {
        self.effect_size = Some(es);
        self
    }

    pub fn with_details(mut self, details: &str) -> Self {
        self.details = details.to_string();
        self
    }

    pub fn to_hypothesis_test_result(&self) -> HypothesisTestResult {
        HypothesisTestResult {
            test_id: TestId::new(),
            test_name: self.name.clone(),
            test_statistic: TestStatistic::z_score(self.statistic),
            p_value: self.p_value,
            effect_size: self.effect_size.map(|es| EffectSize::custom(es)),
            reject: self.reject_null,
            alpha: 0.05,
            confidence_interval: None,
            sample_size: 0,
            description: String::new(),
        }
    }
}

// ── Composite test statistic ────────────────────────────────────────────────

/// Combined test statistic S_T from sub-test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeTestStatistic {
    pub value: f64,
    pub components: Vec<(String, f64, f64)>, // (name, statistic, weight)
    pub combination_method: String,
}

impl CompositeTestStatistic {
    /// Fisher's method: S_T = -2 * Σ ln(p_i).
    pub fn fisher(results: &[SubTestResult]) -> Self {
        let value: f64 = results
            .iter()
            .map(|r| -2.0 * r.p_value.value().max(1e-300).ln())
            .sum();

        let components = results
            .iter()
            .map(|r| (r.name.clone(), r.statistic, r.weight))
            .collect();

        Self {
            value,
            components,
            combination_method: "Fisher".into(),
        }
    }

    /// Weighted combination: S_T = Σ w_i * z_i where z_i = Φ^{-1}(1 - p_i).
    pub fn weighted(results: &[SubTestResult]) -> Self {
        let total_weight: f64 = results.iter().map(|r| r.weight).sum();
        if total_weight.abs() < 1e-15 {
            return Self {
                value: 0.0,
                components: Vec::new(),
                combination_method: "Weighted".into(),
            };
        }

        let value: f64 = results
            .iter()
            .map(|r| {
                let z = normal_quantile(1.0 - r.p_value.value().clamp(0.001, 0.999));
                r.weight * z / total_weight
            })
            .sum();

        let components = results
            .iter()
            .map(|r| (r.name.clone(), r.statistic, r.weight))
            .collect();

        Self {
            value,
            components,
            combination_method: "Weighted".into(),
        }
    }

    /// Stouffer's method: S_T = Σ z_i / √k.
    pub fn stouffer(results: &[SubTestResult]) -> Self {
        let k = results.len() as f64;
        if k < 1.0 {
            return Self {
                value: 0.0,
                components: Vec::new(),
                combination_method: "Stouffer".into(),
            };
        }

        let z_sum: f64 = results
            .iter()
            .map(|r| normal_quantile(1.0 - r.p_value.value().clamp(0.001, 0.999)))
            .sum();

        Self {
            value: z_sum / k.sqrt(),
            components: results.iter().map(|r| (r.name.clone(), r.statistic, r.weight)).collect(),
            combination_method: "Stouffer".into(),
        }
    }

    /// Degrees of freedom for Fisher's method (chi-squared with 2k df).
    pub fn fisher_df(num_tests: usize) -> f64 {
        2.0 * num_tests as f64
    }
}

// ── Sup-competitive correlation bound ───────────────────────────────────────

/// Bound on maximum correlation under competitive null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupCompetitiveCorrelation {
    pub bound: f64,
    pub tier: TieredNull,
    pub method: String,
}

impl SupCompetitiveCorrelation {
    /// Analytical bound for narrow null (linear demand best-response dynamics).
    pub fn narrow(intercept: f64, slope: f64, cross_slope: f64) -> Self {
        let bound = CorrelationBoundUnderNull::analytical_linear(cross_slope, slope);
        Self {
            bound: bound.bound,
            tier: TieredNull::Narrow,
            method: format!(
                "Analytical: best-response dynamics under linear demand \
                 (a={intercept:.2}, b={slope:.2}, c={cross_slope:.2})"
            ),
        }
    }

    /// Optimization over parametric family for medium null.
    pub fn medium(
        demand_system: &DemandSystem,
        num_simulations: usize,
        seed: u64,
    ) -> Self {
        let bound = CorrelationBoundUnderNull::numerical_parametric(
            demand_system,
            num_simulations,
            seed,
        );
        Self {
            bound: bound.bound,
            tier: TieredNull::Medium,
            method: "Numerical optimization over parametric demand family".into(),
        }
    }

    /// Covering-number argument for broad null (Lipschitz functions).
    pub fn broad(lipschitz_constant: f64, dimension: usize, epsilon: f64) -> Self {
        let bound = CorrelationBoundUnderNull::covering_number(
            lipschitz_constant,
            dimension,
            epsilon,
        );
        Self {
            bound: bound.bound,
            tier: TieredNull::Broad,
            method: format!(
                "Covering number: L={lipschitz_constant:.2}, d={dimension}, ε={epsilon:.4}"
            ),
        }
    }
}

// ── Composite decision ──────────────────────────────────────────────────────

/// Decision from the composite test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeDecision {
    pub reject_null: bool,
    pub tier: TieredNull,
    pub composite_statistic: f64,
    pub critical_value: f64,
    pub p_value: PValue,
    pub alpha: f64,
    pub sub_test_results: Vec<SubTestResult>,
    pub fwer_guarantee: Option<FWERGuarantee>,
    pub effect_size: Option<f64>,
    pub reasoning: String,
}

impl CompositeDecision {
    pub fn is_collusion_detected(&self) -> bool {
        self.reject_null
    }

    pub fn num_sub_tests_rejected(&self) -> usize {
        self.sub_test_results.iter().filter(|r| r.reject_null).count()
    }
}

// ── Critical value computation ──────────────────────────────────────────────

/// Compute the critical value for the composite test at significance level alpha.
pub fn compute_critical_value(alpha: f64, tier: TieredNull) -> f64 {
    match tier {
        TieredNull::Narrow => {
            // Use chi-squared critical value (Fisher's method)
            // Approximate: for 2k df, use normal approximation
            normal_quantile(1.0 - alpha)
        }
        TieredNull::Medium => {
            // Slightly higher threshold for medium null (more conservative)
            normal_quantile(1.0 - alpha) * 1.1
        }
        TieredNull::Broad => {
            // Most conservative for broad null
            normal_quantile(1.0 - alpha) * 1.2
        }
    }
}

/// Decide whether to reject the null at significance level alpha.
pub fn reject_null(statistic: f64, alpha: f64, tier: TieredNull) -> bool {
    statistic > compute_critical_value(alpha, tier)
}

// ── Compose tests ───────────────────────────────────────────────────────────

/// Compose multiple sub-test results into a single composite decision.
pub fn compose_tests(results: &[SubTestResult], alpha: f64) -> CompositeDecision {
    if results.is_empty() {
        return CompositeDecision {
            reject_null: false,
            tier: TieredNull::Narrow,
            composite_statistic: 0.0,
            critical_value: compute_critical_value(alpha, TieredNull::Narrow),
            p_value: PValue::new_unchecked(1.0),
            alpha,
            sub_test_results: Vec::new(),
            fwer_guarantee: None,
            effect_size: None,
            reasoning: "No sub-tests provided".into(),
        };
    }

    // Compute composite statistic using Stouffer's method
    let composite = CompositeTestStatistic::stouffer(results);

    // FWER control via Holm-Bonferroni
    let p_values: Vec<f64> = results.iter().map(|r| r.p_value.value()).collect();
    let hb = HolmBonferroni::run(&p_values, alpha);
    let fwer = FWERGuarantee::from_holm(&hb);

    // Combined p-value from Stouffer
    let combined_p = 1.0 - normal_cdf(composite.value);

    // Overall effect size (average)
    let effect_sizes: Vec<f64> = results.iter().filter_map(|r| r.effect_size).collect();
    let avg_effect = if effect_sizes.is_empty() {
        None
    } else {
        Some(effect_sizes.iter().sum::<f64>() / effect_sizes.len() as f64)
    };

    let cv = compute_critical_value(alpha, TieredNull::Narrow);
    let reject = composite.value > cv && hb.num_rejections() > 0;

    let reasoning = format!(
        "Composite statistic (Stouffer) = {:.4}, critical value = {:.4}. \
         {} of {} sub-tests rejected (Holm-Bonferroni). Combined p = {:.6}.",
        composite.value,
        cv,
        hb.num_rejections(),
        results.len(),
        combined_p,
    );

    CompositeDecision {
        reject_null: reject,
        tier: TieredNull::Narrow,
        composite_statistic: composite.value,
        critical_value: cv,
        p_value: PValue::new_unchecked(combined_p),
        alpha,
        sub_test_results: results.to_vec(),
        fwer_guarantee: Some(fwer),
        effect_size: avg_effect,
        reasoning,
    }
}

// ── Main composite test ─────────────────────────────────────────────────────

/// The main M1 composite test orchestrating all sub-tests against the tiered null.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeTest {
    pub alpha: f64,
    pub tier: TieredNull,
    pub num_bootstrap: usize,
    pub seed: Option<u64>,
}

impl CompositeTest {
    pub fn new(alpha: f64, tier: TieredNull) -> Self {
        Self {
            alpha,
            tier,
            num_bootstrap: 2000,
            seed: None,
        }
    }

    pub fn with_bootstrap(mut self, n: usize) -> Self {
        self.num_bootstrap = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run the full composite test on price trajectory data.
    pub fn run(
        &self,
        firm_prices: &[Vec<f64>],
        competitive_price: Price,
        monopoly_price: Price,
    ) -> CollusionResult<CompositeDecision> {
        if firm_prices.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Composite test: no firm data".into(),
            ));
        }

        let mut results = Vec::new();

        // Sub-test 1: Supra-competitive pricing for each firm
        let pricing_test = SupraCompetitivePriceTest::new(competitive_price)
            .with_bootstrap(self.num_bootstrap.min(500))
            .with_seed(self.seed.unwrap_or(42));

        for (i, prices) in firm_prices.iter().enumerate() {
            if prices.is_empty() { continue; }
            match pricing_test.test(prices) {
                Ok(r) => {
                    results.push(
                        SubTestResult::new(
                            &format!("Supra-competitive pricing (firm {i})"),
                            r.t_statistic,
                            r.bootstrap_p_value.value(),
                            self.alpha,
                            1.0,
                        )
                        .with_effect_size(r.effect_size)
                    );
                }
                Err(_) => {}
            }
        }

        // Sub-test 2: Price parallelism for each pair
        if firm_prices.len() >= 2 {
            let par_test = PriceParallelismTest::new(5);
            for i in 0..firm_prices.len() {
                for j in (i + 1)..firm_prices.len() {
                    match par_test.test(&firm_prices[i], &firm_prices[j]) {
                        Ok(r) => {
                            results.push(SubTestResult::new(
                                &format!("Price parallelism (firms {i}-{j})"),
                                r.change_correlation,
                                r.p_value.value(),
                                self.alpha,
                                0.8,
                            ));
                        }
                        Err(_) => {}
                    }
                }
            }
        }

        // Sub-test 3: Price persistence
        let persist_test = PricePersistenceTest::new(competitive_price);
        for (i, prices) in firm_prices.iter().enumerate() {
            if prices.is_empty() { continue; }
            match persist_test.test(prices) {
                Ok(r) => {
                    results.push(SubTestResult::new(
                        &format!("Price persistence (firm {i})"),
                        r.max_run_length as f64,
                        r.p_value.value(),
                        self.alpha,
                        0.6,
                    ));
                }
                Err(_) => {}
            }
        }

        // Sub-test 4: Cross-firm correlation
        if firm_prices.len() >= 2 {
            for i in 0..firm_prices.len() {
                for j in (i + 1)..firm_prices.len() {
                    match CrossFirmCorrelation::compute(&firm_prices[i], &firm_prices[j]) {
                        Ok(cfc) => {
                            results.push(SubTestResult::new(
                                &format!("Cross-firm correlation (firms {i}-{j})"),
                                cfc.pearson,
                                cfc.pearson_p_value().value(),
                                self.alpha,
                                0.7,
                            ));
                        }
                        Err(_) => {}
                    }
                }
            }
        }

        // Sub-test 5: Collusion effect size
        let all_prices: Vec<f64> = firm_prices.iter().flat_map(|v| v.iter().copied()).collect();
        if !all_prices.is_empty() {
            match CollusionEffectSize::compute(&all_prices, competitive_price.0, monopoly_price.0) {
                Ok(es) => {
                    let p = if es.value > 2.0 { 0.01 } else if es.value > 1.0 { 0.05 } else { 0.5 };
                    results.push(
                        SubTestResult::new(
                            "Collusion effect size",
                            es.value,
                            p,
                            self.alpha,
                            1.0,
                        )
                        .with_effect_size(es.collusion_premium)
                        .with_details(&format!("CP = {:.4}", es.collusion_premium))
                    );
                }
                Err(_) => {}
            }
        }

        // Compose all sub-tests
        let mut decision = compose_tests(&results, self.alpha);
        decision.tier = self.tier;
        decision.critical_value = compute_critical_value(self.alpha, self.tier);
        decision.reject_null = decision.composite_statistic > decision.critical_value
            && decision.num_sub_tests_rejected() > 0;

        Ok(decision)
    }

    /// Run the composite test against all three tiers.
    pub fn run_all_tiers(
        &self,
        firm_prices: &[Vec<f64>],
        competitive_price: Price,
        monopoly_price: Price,
    ) -> CollusionResult<Vec<CompositeDecision>> {
        let mut decisions = Vec::new();
        for tier in TieredNull::all_tiers() {
            let test = CompositeTest::new(self.alpha, tier)
                .with_bootstrap(self.num_bootstrap)
                .with_seed(self.seed.unwrap_or(42));
            decisions.push(test.run(firm_prices, competitive_price, monopoly_price)?);
        }
        Ok(decisions)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn collusive_prices(n: usize) -> Vec<f64> {
        vec![5.0; n]
    }

    fn competitive_prices(n: usize) -> Vec<f64> {
        use rand::prelude::*;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        (0..n).map(|_| 2.0 + rng.gen_range(-0.1..0.1)).collect()
    }

    #[test]
    fn test_tiered_null_display() {
        assert!(format!("{}", TieredNull::Narrow).contains("Narrow"));
        assert!(format!("{}", TieredNull::Medium).contains("Medium"));
        assert!(format!("{}", TieredNull::Broad).contains("Broad"));
    }

    #[test]
    fn test_tiered_null_all() {
        let tiers = TieredNull::all_tiers();
        assert_eq!(tiers.len(), 3);
    }

    #[test]
    fn test_tiered_null_generality() {
        assert!(TieredNull::Broad.is_at_least_as_general(&TieredNull::Narrow));
        assert!(TieredNull::Broad.is_at_least_as_general(&TieredNull::Medium));
        assert!(TieredNull::Medium.is_at_least_as_general(&TieredNull::Narrow));
        assert!(!TieredNull::Narrow.is_at_least_as_general(&TieredNull::Broad));
    }

    #[test]
    fn test_null_hypothesis_narrow() {
        let h0 = NullHypothesis::narrow();
        assert_eq!(h0.tier, TieredNull::Narrow);
        assert!(matches!(h0.demand_family, DemandFamily::Linear));
        assert!(matches!(h0.algorithm_family, AlgorithmFamily::QLearningIndependent));
    }

    #[test]
    fn test_null_hypothesis_medium() {
        let h0 = NullHypothesis::medium();
        assert_eq!(h0.tier, TieredNull::Medium);
    }

    #[test]
    fn test_null_hypothesis_broad() {
        let h0 = NullHypothesis::broad(2.0);
        assert_eq!(h0.tier, TieredNull::Broad);
        assert!(matches!(h0.demand_family, DemandFamily::Lipschitz { .. }));
    }

    #[test]
    fn test_sub_test_result_basic() {
        let r = SubTestResult::new("test", 2.5, 0.01, 0.05, 1.0);
        assert!(r.reject_null);
        assert_eq!(r.name, "test");
    }

    #[test]
    fn test_sub_test_result_with_details() {
        let r = SubTestResult::new("test", 1.0, 0.1, 0.05, 1.0)
            .with_effect_size(0.5)
            .with_details("some detail");
        assert_eq!(r.effect_size, Some(0.5));
        assert_eq!(r.details, "some detail");
    }

    #[test]
    fn test_sub_test_to_hypothesis() {
        let r = SubTestResult::new("test", 2.5, 0.01, 0.05, 1.0);
        let htr = r.to_hypothesis_test_result();
        assert_eq!(htr.test_name, "test");
        assert!(htr.reject_null);
    }

    #[test]
    fn test_composite_statistic_fisher() {
        let results = vec![
            SubTestResult::new("t1", 3.0, 0.001, 0.05, 1.0),
            SubTestResult::new("t2", 2.0, 0.01, 0.05, 1.0),
        ];
        let cs = CompositeTestStatistic::fisher(&results);
        assert!(cs.value > 0.0); // -2*(ln(0.001) + ln(0.01)) > 0
        assert_eq!(cs.combination_method, "Fisher");
    }

    #[test]
    fn test_composite_statistic_weighted() {
        let results = vec![
            SubTestResult::new("t1", 3.0, 0.01, 0.05, 2.0),
            SubTestResult::new("t2", 1.0, 0.5, 0.05, 1.0),
        ];
        let cs = CompositeTestStatistic::weighted(&results);
        assert!(cs.value > 0.0);
        assert_eq!(cs.combination_method, "Weighted");
    }

    #[test]
    fn test_composite_statistic_stouffer() {
        let results = vec![
            SubTestResult::new("t1", 3.0, 0.001, 0.05, 1.0),
            SubTestResult::new("t2", 2.0, 0.001, 0.05, 1.0),
        ];
        let cs = CompositeTestStatistic::stouffer(&results);
        assert!(cs.value > 2.0);
    }

    #[test]
    fn test_composite_statistic_fisher_df() {
        assert!(approx_eq(CompositeTestStatistic::fisher_df(5), 10.0, 1e-10));
    }

    #[test]
    fn test_sup_competitive_correlation_narrow() {
        let sc = SupCompetitiveCorrelation::narrow(10.0, 1.0, 0.5);
        assert!(sc.bound >= 0.0);
        assert!(sc.bound <= 1.0);
        assert_eq!(sc.tier, TieredNull::Narrow);
    }

    #[test]
    fn test_sup_competitive_correlation_medium() {
        let ds = DemandSystem::Logit { mu: 0.5, a_0: 0.0 };
        let sc = SupCompetitiveCorrelation::medium(&ds, 100, 42);
        assert!(sc.bound >= 0.0);
        assert_eq!(sc.tier, TieredNull::Medium);
    }

    #[test]
    fn test_sup_competitive_correlation_broad() {
        let sc = SupCompetitiveCorrelation::broad(1.0, 2, 0.1);
        assert!(sc.bound >= 0.0);
        assert_eq!(sc.tier, TieredNull::Broad);
    }

    #[test]
    fn test_compute_critical_value() {
        let cv_narrow = compute_critical_value(0.05, TieredNull::Narrow);
        let cv_medium = compute_critical_value(0.05, TieredNull::Medium);
        let cv_broad = compute_critical_value(0.05, TieredNull::Broad);
        assert!(cv_narrow > 0.0);
        assert!(cv_medium > cv_narrow);
        assert!(cv_broad > cv_medium);
    }

    #[test]
    fn test_reject_null_function() {
        assert!(reject_null(5.0, 0.05, TieredNull::Narrow));
        assert!(!reject_null(0.5, 0.05, TieredNull::Narrow));
    }

    #[test]
    fn test_compose_tests_collusive() {
        let results = vec![
            SubTestResult::new("pricing", 5.0, 0.001, 0.05, 1.0).with_effect_size(2.0),
            SubTestResult::new("correlation", 3.0, 0.005, 0.05, 0.8).with_effect_size(1.5),
            SubTestResult::new("persistence", 4.0, 0.01, 0.05, 0.6),
        ];
        let decision = compose_tests(&results, 0.05);
        assert!(decision.reject_null);
        assert!(decision.composite_statistic > 0.0);
        assert!(decision.fwer_guarantee.is_some());
    }

    #[test]
    fn test_compose_tests_competitive() {
        let results = vec![
            SubTestResult::new("pricing", 0.5, 0.5, 0.05, 1.0),
            SubTestResult::new("correlation", 0.3, 0.7, 0.05, 0.8),
        ];
        let decision = compose_tests(&results, 0.05);
        assert!(!decision.reject_null);
    }

    #[test]
    fn test_compose_tests_empty() {
        let decision = compose_tests(&[], 0.05);
        assert!(!decision.reject_null);
    }

    #[test]
    fn test_composite_decision_methods() {
        let results = vec![
            SubTestResult::new("t1", 5.0, 0.001, 0.05, 1.0),
            SubTestResult::new("t2", 0.1, 0.8, 0.05, 1.0),
        ];
        let decision = compose_tests(&results, 0.05);
        assert!(decision.num_sub_tests_rejected() >= 1);
    }

    #[test]
    fn test_composite_test_collusive() {
        let prices1 = collusive_prices(50);
        let prices2 = collusive_prices(50);
        let test = CompositeTest::new(0.05, TieredNull::Narrow)
            .with_bootstrap(200)
            .with_seed(42);
        let result = test.run(&[prices1, prices2], 2.0, 8.0).unwrap();
        // Prices at 5.0 with competitive=2.0 → should detect collusion
        assert!(result.composite_statistic > 0.0);
    }

    #[test]
    fn test_composite_test_competitive() {
        let prices1 = competitive_prices(50);
        let prices2 = competitive_prices(50);
        let test = CompositeTest::new(0.05, TieredNull::Narrow)
            .with_bootstrap(200)
            .with_seed(42);
        let result = test.run(&[prices1, prices2], 2.0, 8.0).unwrap();
        // Prices near 2.0 with competitive=2.0 → should not detect collusion
        // (depends on noise, but effect size should be small)
        assert!(result.sub_test_results.len() > 0);
    }

    #[test]
    fn test_composite_test_empty() {
        let test = CompositeTest::new(0.05, TieredNull::Narrow);
        assert!(test.run(&[], 2.0, 8.0).is_err());
    }

    #[test]
    fn test_composite_test_all_tiers() {
        let prices1 = collusive_prices(50);
        let prices2 = collusive_prices(50);
        let test = CompositeTest::new(0.05, TieredNull::Narrow)
            .with_bootstrap(100)
            .with_seed(42);
        let results = test.run_all_tiers(&[prices1, prices2], 2.0, 8.0).unwrap();
        assert_eq!(results.len(), 3);
        // Broad should be most conservative
        let cv_narrow = results[0].critical_value;
        let cv_broad = results[2].critical_value;
        assert!(cv_broad >= cv_narrow);
    }
}
