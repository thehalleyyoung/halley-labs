//! Statistical primitive types for the CollusionProof system.
//!
//! Contains test statistics, p-values, confidence intervals, hypothesis test
//! results, test batteries, alpha budget tracking, FWER correction methods,
//! and bootstrap/permutation result types.

use crate::identifiers::TestId;
use serde::{Deserialize, Serialize};
use std::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Core statistical types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A test statistic with metadata about its distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestStatistic {
    pub value: f64,
    pub distribution: DistributionInfo,
    pub name: String,
}

impl TestStatistic {
    pub fn new(value: f64, distribution: DistributionInfo, name: impl Into<String>) -> Self {
        TestStatistic { value, distribution, name: name.into() }
    }

    pub fn z_score(value: f64) -> Self {
        TestStatistic::new(value, DistributionInfo::Normal { mean: 0.0, std: 1.0 }, "z")
    }

    pub fn t_score(value: f64, df: f64) -> Self {
        TestStatistic::new(value, DistributionInfo::StudentT { df }, "t")
    }

    pub fn chi_squared(value: f64, df: f64) -> Self {
        TestStatistic::new(value, DistributionInfo::ChiSquared { df }, "χ²")
    }

    pub fn f_stat(value: f64, df1: f64, df2: f64) -> Self {
        TestStatistic::new(value, DistributionInfo::FDist { df1, df2 }, "F")
    }
}

impl fmt::Display for TestStatistic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.4} ({})", self.name, self.value, self.distribution)
    }
}

/// Information about a probability distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionInfo {
    Normal { mean: f64, std: f64 },
    StudentT { df: f64 },
    ChiSquared { df: f64 },
    FDist { df1: f64, df2: f64 },
    Permutation { num_permutations: usize },
    Bootstrap { num_samples: usize },
    Empirical,
    Unknown,
}

impl fmt::Display for DistributionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionInfo::Normal { mean, std } => write!(f, "N({:.2},{:.2})", mean, std),
            DistributionInfo::StudentT { df } => write!(f, "t(df={:.1})", df),
            DistributionInfo::ChiSquared { df } => write!(f, "χ²(df={:.1})", df),
            DistributionInfo::FDist { df1, df2 } => write!(f, "F({:.0},{:.0})", df1, df2),
            DistributionInfo::Permutation { num_permutations } => write!(f, "Perm(n={})", num_permutations),
            DistributionInfo::Bootstrap { num_samples } => write!(f, "Boot(n={})", num_samples),
            DistributionInfo::Empirical => write!(f, "Empirical"),
            DistributionInfo::Unknown => write!(f, "Unknown"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// P-value
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A validated p-value in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PValue(f64);

impl PValue {
    /// Create a new p-value. Returns `None` if outside [0, 1].
    pub fn new(value: f64) -> Option<Self> {
        if value >= 0.0 && value <= 1.0 {
            Some(PValue(value))
        } else {
            None
        }
    }

    /// Create without validation.
    pub fn new_unchecked(value: f64) -> Self {
        PValue(value.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f64 { self.0 }

    /// Whether this p-value is significant at the given alpha level.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.0 < alpha
    }

    /// Convert to a star-rating string (common in papers).
    pub fn stars(&self) -> &'static str {
        if self.0 < 0.001 { "***" }
        else if self.0 < 0.01 { "**" }
        else if self.0 < 0.05 { "*" }
        else if self.0 < 0.1 { "." }
        else { "" }
    }

    /// Combine p-values using Fisher's method: -2 Σ ln(pᵢ) ~ χ²(2k).
    pub fn fisher_combine(pvalues: &[PValue]) -> f64 {
        -2.0 * pvalues.iter().map(|p| p.0.max(1e-300).ln()).sum::<f64>()
    }

    /// Bonferroni-corrected p-value: min(p * n, 1).
    pub fn bonferroni_correct(&self, num_tests: usize) -> PValue {
        PValue::new_unchecked(self.0 * num_tests as f64)
    }
}

impl Default for PValue {
    fn default() -> Self { PValue(1.0) }
}

impl fmt::Display for PValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 0.001 {
            write!(f, "p<0.001{}", self.stars())
        } else {
            write!(f, "p={:.4}{}", self.0, self.stars())
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Confidence interval
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A confidence interval [lower, upper] at a given confidence level.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
    pub point_estimate: f64,
}

impl ConfidenceInterval {
    pub fn new(lower: f64, upper: f64, level: f64, point_estimate: f64) -> Self {
        ConfidenceInterval { lower, upper, level, point_estimate }
    }

    /// Symmetric CI from point estimate and margin of error.
    pub fn symmetric(estimate: f64, margin: f64, level: f64) -> Self {
        ConfidenceInterval {
            lower: estimate - margin,
            upper: estimate + margin,
            level,
            point_estimate: estimate,
        }
    }

    pub fn width(&self) -> f64 { self.upper - self.lower }

    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Whether the CI excludes zero (indicating significance).
    pub fn excludes_zero(&self) -> bool {
        self.lower > 0.0 || self.upper < 0.0
    }

    pub fn midpoint(&self) -> f64 { (self.lower + self.upper) / 2.0 }

    pub fn margin_of_error(&self) -> f64 { self.width() / 2.0 }
}

impl fmt::Display for ConfidenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}] ({:.0}% CI, est={:.4})",
            self.lower, self.upper, self.level * 100.0, self.point_estimate)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Effect size
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// An effect size measure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectSize {
    pub value: f64,
    pub measure: EffectSizeMeasure,
}

/// Type of effect size measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectSizeMeasure {
    CohenD,
    HedgesG,
    GlassDelta,
    Eta2,
    PartialEta2,
    R2,
    CramersV,
    Custom,
}

impl EffectSize {
    pub fn cohen_d(value: f64) -> Self {
        EffectSize { value, measure: EffectSizeMeasure::CohenD }
    }

    pub fn hedges_g(value: f64) -> Self {
        EffectSize { value, measure: EffectSizeMeasure::HedgesG }
    }

    pub fn r_squared(value: f64) -> Self {
        EffectSize { value, measure: EffectSizeMeasure::R2 }
    }

    pub fn custom(value: f64) -> Self {
        EffectSize { value, measure: EffectSizeMeasure::Custom }
    }

    /// Qualitative interpretation following Cohen's conventions.
    pub fn interpretation(&self) -> &'static str {
        let v = self.value.abs();
        match self.measure {
            EffectSizeMeasure::CohenD | EffectSizeMeasure::HedgesG | EffectSizeMeasure::GlassDelta => {
                if v < 0.2 { "negligible" }
                else if v < 0.5 { "small" }
                else if v < 0.8 { "medium" }
                else { "large" }
            }
            EffectSizeMeasure::R2 | EffectSizeMeasure::Eta2 | EffectSizeMeasure::PartialEta2 => {
                if v < 0.01 { "negligible" }
                else if v < 0.06 { "small" }
                else if v < 0.14 { "medium" }
                else { "large" }
            }
            _ => {
                if v < 0.1 { "negligible" }
                else if v < 0.3 { "small" }
                else if v < 0.5 { "medium" }
                else { "large" }
            }
        }
    }
}

impl fmt::Display for EffectSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}={:.4} ({})", self.measure, self.value, self.interpretation())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Hypothesis test result
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Complete result of a hypothesis test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    pub test_id: TestId,
    pub test_name: String,
    pub test_statistic: TestStatistic,
    pub p_value: PValue,
    pub reject: bool,
    pub alpha: f64,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub effect_size: Option<EffectSize>,
    pub sample_size: usize,
    pub description: String,
}

impl HypothesisTestResult {
    pub fn new(
        test_name: impl Into<String>,
        test_statistic: TestStatistic,
        p_value: PValue,
        alpha: f64,
    ) -> Self {
        let reject = p_value.is_significant(alpha);
        HypothesisTestResult {
            test_id: TestId::new(),
            test_name: test_name.into(),
            test_statistic,
            p_value,
            reject,
            alpha,
            confidence_interval: None,
            effect_size: None,
            sample_size: 0,
            description: String::new(),
        }
    }

    pub fn with_ci(mut self, ci: ConfidenceInterval) -> Self {
        self.confidence_interval = Some(ci); self
    }

    pub fn with_effect_size(mut self, es: EffectSize) -> Self {
        self.effect_size = Some(es); self
    }

    pub fn with_sample_size(mut self, n: usize) -> Self {
        self.sample_size = n; self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into(); self
    }

    pub fn is_significant(&self) -> bool { self.reject }

    /// Whether the evidence is practically meaningful (effect size above threshold).
    pub fn is_practically_significant(&self, min_effect: f64) -> bool {
        self.reject && self.effect_size.as_ref()
            .map(|es| es.value.abs() >= min_effect)
            .unwrap_or(false)
    }
}

impl fmt::Display for HypothesisTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} | {} | reject={}",
            self.test_name, self.test_statistic, self.p_value, self.reject)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test battery
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A collection of hypothesis test results run together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestBattery {
    pub name: String,
    pub results: Vec<HypothesisTestResult>,
    pub fwer_method: Option<FWERControl>,
}

impl TestBattery {
    pub fn new(name: impl Into<String>) -> Self {
        TestBattery { name: name.into(), results: Vec::new(), fwer_method: None }
    }

    pub fn add_result(&mut self, result: HypothesisTestResult) {
        self.results.push(result);
    }

    pub fn add_test(&mut self, result: HypothesisTestResult) {
        self.add_result(result);
    }

    pub fn num_tests(&self) -> usize { self.results.len() }
    pub fn num_rejected(&self) -> usize { self.results.iter().filter(|r| r.reject).count() }
    pub fn num_significant(&self) -> usize { self.num_rejected() }

    pub fn rejection_rate(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.num_rejected() as f64 / self.results.len() as f64
    }

    pub fn min_p_value(&self) -> PValue {
        self.results.iter().map(|r| r.p_value)
            .min_by(|a, b| a.value().partial_cmp(&b.value()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_default()
    }

    pub fn max_p_value(&self) -> PValue {
        self.results.iter().map(|r| r.p_value)
            .max_by(|a, b| a.value().partial_cmp(&b.value()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_default()
    }

    /// Apply FWER correction and return adjusted results.
    pub fn apply_fwer(&mut self, method: FWERControl) {
        self.fwer_method = Some(method);
        let adjusted = apply_fwer_correction(&self.results, method);
        for (result, adj_p) in self.results.iter_mut().zip(adjusted.iter()) {
            result.p_value = *adj_p;
            result.reject = adj_p.is_significant(result.alpha);
        }
    }

    pub fn significant_results(&self) -> Vec<&HypothesisTestResult> {
        self.results.iter().filter(|r| r.reject).collect()
    }
}

impl fmt::Display for TestBattery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Battery '{}': {}/{} rejected (min p={})",
            self.name, self.num_rejected(), self.num_tests(), self.min_p_value())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Alpha budget
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tracks alpha spending across multiple tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaBudget {
    pub total_alpha: f64,
    pub spent_alpha: f64,
    pub allocations: Vec<(String, f64)>,
}

impl AlphaBudget {
    pub fn new(total_alpha: f64) -> Self {
        AlphaBudget { total_alpha, spent_alpha: 0.0, allocations: Vec::new() }
    }

    /// Remaining alpha budget.
    pub fn remaining(&self) -> f64 {
        (self.total_alpha - self.spent_alpha).max(0.0)
    }

    /// Whether the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() < 1e-15
    }

    /// Allocate alpha for a test. Returns `None` if insufficient budget.
    pub fn allocate(&mut self, test_name: impl Into<String>, amount: f64) -> Option<f64> {
        if amount > self.remaining() + 1e-15 {
            return None;
        }
        let name = test_name.into();
        let actual = amount.min(self.remaining());
        self.spent_alpha += actual;
        self.allocations.push((name, actual));
        Some(actual)
    }

    /// Allocate equally among n upcoming tests.
    pub fn allocate_equal(&mut self, names: &[String]) -> Vec<f64> {
        let n = names.len();
        if n == 0 { return vec![]; }
        let per_test = self.remaining() / n as f64;
        names.iter().map(|name| {
            self.allocate(name.clone(), per_test).unwrap_or(0.0)
        }).collect()
    }

    /// Allocate using Holm-Bonferroni step-down spending.
    pub fn allocate_holm(&mut self, num_tests: usize) -> Vec<f64> {
        (0..num_tests).map(|i| {
            self.total_alpha / (num_tests - i) as f64
        }).collect()
    }

    pub fn fraction_spent(&self) -> f64 {
        if self.total_alpha <= 0.0 { return 1.0; }
        self.spent_alpha / self.total_alpha
    }
}

impl fmt::Display for AlphaBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlphaBudget(spent={:.4}/{:.4}, {:.1}% remaining)",
            self.spent_alpha, self.total_alpha, self.remaining() / self.total_alpha * 100.0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FWER control
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Family-wise error rate control methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FWERControl {
    /// Holm–Bonferroni step-down method.
    HolmBonferroni,
    /// Hochberg step-up method.
    Hochberg,
    /// Hommel method (more powerful but complex).
    Hommel,
    /// Standard Bonferroni (most conservative).
    Bonferroni,
    /// Benjamini-Hochberg FDR control.
    BenjaminiHochberg,
}

impl fmt::Display for FWERControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FWERControl::HolmBonferroni => write!(f, "Holm-Bonferroni"),
            FWERControl::Hochberg => write!(f, "Hochberg"),
            FWERControl::Hommel => write!(f, "Hommel"),
            FWERControl::Bonferroni => write!(f, "Bonferroni"),
            FWERControl::BenjaminiHochberg => write!(f, "Benjamini-Hochberg"),
        }
    }
}

/// Apply FWER correction to a set of test results. Returns adjusted p-values.
pub fn apply_fwer_correction(
    results: &[HypothesisTestResult],
    method: FWERControl,
) -> Vec<PValue> {
    let n = results.len();
    if n == 0 { return vec![]; }

    let mut indexed_p: Vec<(usize, f64)> = results.iter()
        .enumerate()
        .map(|(i, r)| (i, r.p_value.value()))
        .collect();
    indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0_f64; n];

    match method {
        FWERControl::Bonferroni => {
            for (orig_idx, p) in &indexed_p {
                adjusted[*orig_idx] = (*p * n as f64).min(1.0);
            }
        }
        FWERControl::HolmBonferroni => {
            let mut max_so_far = 0.0_f64;
            for (rank, (orig_idx, p)) in indexed_p.iter().enumerate() {
                let factor = (n - rank) as f64;
                let adj = (p * factor).min(1.0);
                max_so_far = max_so_far.max(adj);
                adjusted[*orig_idx] = max_so_far;
            }
        }
        FWERControl::Hochberg => {
            let mut min_so_far = 1.0_f64;
            for (rank, (orig_idx, p)) in indexed_p.iter().enumerate().rev() {
                let factor = (n - rank) as f64;
                let adj = (p * factor).min(1.0);
                min_so_far = min_so_far.min(adj);
                adjusted[*orig_idx] = min_so_far;
            }
        }
        FWERControl::Hommel => {
            // Hommel's method: iteratively check subsets.
            // Simplified implementation using Hochberg as a base.
            let mut min_so_far = 1.0_f64;
            for (rank, (orig_idx, p)) in indexed_p.iter().enumerate().rev() {
                let factor = (n - rank) as f64;
                let adj = (p * factor).min(1.0);
                min_so_far = min_so_far.min(adj);
                adjusted[*orig_idx] = min_so_far;
            }
            // Additional Hommel refinement: check if we can improve
            for j in (2..=n).rev() {
                let cj: f64 = (1..=j).map(|k| 1.0 / k as f64).sum();
                let p_j = indexed_p[j - 1].1;
                let threshold = p_j * cj;
                for (rank, (orig_idx, _)) in indexed_p.iter().enumerate() {
                    if rank < j - 1 {
                        adjusted[*orig_idx] = adjusted[*orig_idx].min(threshold.min(1.0));
                    }
                }
            }
        }
        FWERControl::BenjaminiHochberg => {
            let mut min_so_far = 1.0_f64;
            for (rank, (orig_idx, p)) in indexed_p.iter().enumerate().rev() {
                let adj = (p * n as f64 / (rank + 1) as f64).min(1.0);
                min_so_far = min_so_far.min(adj);
                adjusted[*orig_idx] = min_so_far;
            }
        }
    }

    adjusted.into_iter()
        .map(|p| PValue::new_unchecked(p))
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bootstrap and permutation results
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Result of a bootstrap analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub point_estimate: f64,
    pub standard_error: f64,
    pub confidence_interval: ConfidenceInterval,
    pub num_iterations: usize,
    pub bias: f64,
    pub bootstrap_distribution: Vec<f64>,
}

impl BootstrapResult {
    pub fn new(
        point_estimate: f64,
        standard_error: f64,
        ci: ConfidenceInterval,
        num_iterations: usize,
        distribution: Vec<f64>,
    ) -> Self {
        let bias = if distribution.is_empty() {
            0.0
        } else {
            let mean: f64 = distribution.iter().sum::<f64>() / distribution.len() as f64;
            mean - point_estimate
        };
        BootstrapResult {
            point_estimate, standard_error,
            confidence_interval: ci, num_iterations, bias,
            bootstrap_distribution: distribution,
        }
    }

    /// Bias-corrected estimate.
    pub fn bias_corrected(&self) -> f64 {
        self.point_estimate - self.bias
    }

    /// Coefficient of variation of the bootstrap distribution.
    pub fn cv(&self) -> f64 {
        if self.point_estimate.abs() < 1e-15 { return f64::INFINITY; }
        self.standard_error / self.point_estimate.abs()
    }
}

impl fmt::Display for BootstrapResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bootstrap(est={:.4}, se={:.4}, CI={}, n={})",
            self.point_estimate, self.standard_error,
            self.confidence_interval, self.num_iterations)
    }
}

/// Result of a permutation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationResult {
    pub observed_statistic: f64,
    pub p_value: PValue,
    pub num_permutations: usize,
    pub null_distribution: Vec<f64>,
    pub num_exceedances: usize,
}

impl PermutationResult {
    pub fn new(
        observed: f64,
        null_distribution: Vec<f64>,
    ) -> Self {
        let n = null_distribution.len();
        let exceedances = null_distribution.iter()
            .filter(|x| x.abs() >= observed.abs())
            .count();
        let p = if n > 0 { (exceedances + 1) as f64 / (n + 1) as f64 } else { 1.0 };
        PermutationResult {
            observed_statistic: observed,
            p_value: PValue::new_unchecked(p),
            num_permutations: n,
            null_distribution,
            num_exceedances: exceedances,
        }
    }

    /// Percentile rank of the observed statistic.
    pub fn percentile_rank(&self) -> f64 {
        if self.null_distribution.is_empty() { return 0.5; }
        let below = self.null_distribution.iter()
            .filter(|x| **x < self.observed_statistic).count();
        below as f64 / self.null_distribution.len() as f64
    }
}

impl fmt::Display for PermutationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Permutation(stat={:.4}, {}, n={})",
            self.observed_statistic, self.p_value, self.num_permutations)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_value_validation() {
        assert!(PValue::new(0.05).is_some());
        assert!(PValue::new(0.0).is_some());
        assert!(PValue::new(1.0).is_some());
        assert!(PValue::new(-0.1).is_none());
        assert!(PValue::new(1.1).is_none());
    }

    #[test]
    fn test_p_value_significance() {
        let p = PValue::new(0.03).unwrap();
        assert!(p.is_significant(0.05));
        assert!(!p.is_significant(0.01));
    }

    #[test]
    fn test_p_value_stars() {
        assert_eq!(PValue::new(0.0001).unwrap().stars(), "***");
        assert_eq!(PValue::new(0.005).unwrap().stars(), "**");
        assert_eq!(PValue::new(0.03).unwrap().stars(), "*");
        assert_eq!(PValue::new(0.07).unwrap().stars(), ".");
        assert_eq!(PValue::new(0.5).unwrap().stars(), "");
    }

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::symmetric(5.0, 1.0, 0.95);
        assert_eq!(ci.lower, 4.0);
        assert_eq!(ci.upper, 6.0);
        assert!(ci.contains(5.0));
        assert!(!ci.contains(7.0));
        assert!(ci.excludes_zero());
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert_eq!(EffectSize::cohen_d(0.1).interpretation(), "negligible");
        assert_eq!(EffectSize::cohen_d(0.3).interpretation(), "small");
        assert_eq!(EffectSize::cohen_d(0.6).interpretation(), "medium");
        assert_eq!(EffectSize::cohen_d(1.0).interpretation(), "large");
    }

    #[test]
    fn test_hypothesis_test_result() {
        let ts = TestStatistic::z_score(2.5);
        let p = PValue::new(0.012).unwrap();
        let r = HypothesisTestResult::new("test1", ts, p, 0.05);
        assert!(r.is_significant());
    }

    #[test]
    fn test_test_battery() {
        let mut battery = TestBattery::new("collusion tests");
        battery.add_result(HypothesisTestResult::new(
            "t1", TestStatistic::z_score(2.0), PValue::new(0.04).unwrap(), 0.05));
        battery.add_result(HypothesisTestResult::new(
            "t2", TestStatistic::z_score(1.0), PValue::new(0.3).unwrap(), 0.05));
        assert_eq!(battery.num_tests(), 2);
        assert_eq!(battery.num_rejected(), 1);
        assert!((battery.rejection_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_budget() {
        let mut budget = AlphaBudget::new(0.05);
        assert!(!budget.is_exhausted());
        assert!(budget.allocate("t1", 0.02).is_some());
        assert!(budget.allocate("t2", 0.02).is_some());
        assert!((budget.remaining() - 0.01).abs() < 1e-10);
        assert!(budget.allocate("t3", 0.02).is_none());
    }

    #[test]
    fn test_bonferroni_correction() {
        let results: Vec<HypothesisTestResult> = (0..3).map(|i| {
            let p = PValue::new(0.01 + 0.01 * i as f64).unwrap();
            HypothesisTestResult::new(format!("t{}", i), TestStatistic::z_score(2.0), p, 0.05)
        }).collect();
        let adjusted = apply_fwer_correction(&results, FWERControl::Bonferroni);
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0].value() >= results[0].p_value.value());
    }

    #[test]
    fn test_holm_bonferroni() {
        let results: Vec<HypothesisTestResult> = vec![0.01, 0.04, 0.03].into_iter().map(|p| {
            HypothesisTestResult::new("t", TestStatistic::z_score(2.0),
                PValue::new(p).unwrap(), 0.05)
        }).collect();
        let adjusted = apply_fwer_correction(&results, FWERControl::HolmBonferroni);
        assert_eq!(adjusted.len(), 3);
        // Holm: sorted p-values multiplied by (n-rank)
        for adj in &adjusted {
            assert!(adj.value() <= 1.0);
        }
    }

    #[test]
    fn test_bootstrap_result() {
        let dist: Vec<f64> = (0..100).map(|i| 5.0 + 0.1 * (i as f64 - 50.0)).collect();
        let ci = ConfidenceInterval::symmetric(5.0, 0.5, 0.95);
        let br = BootstrapResult::new(5.0, 0.3, ci, 100, dist);
        assert!((br.bias).abs() < 1.0);
        assert!(br.cv() > 0.0);
    }

    #[test]
    fn test_permutation_result() {
        let null: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();
        let pr = PermutationResult::new(3.0, null);
        assert!(pr.p_value.value() > 0.0);
        assert!(pr.p_value.value() < 1.0);
    }

    #[test]
    fn test_fisher_combine() {
        let ps = vec![PValue::new(0.01).unwrap(), PValue::new(0.05).unwrap()];
        let stat = PValue::fisher_combine(&ps);
        assert!(stat > 0.0);
    }

    #[test]
    fn test_bonferroni_correct() {
        let p = PValue::new(0.01).unwrap();
        let adj = p.bonferroni_correct(5);
        assert!((adj.value() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_practically_significant() {
        let ts = TestStatistic::z_score(3.0);
        let p = PValue::new(0.001).unwrap();
        let r = HypothesisTestResult::new("t", ts, p, 0.05)
            .with_effect_size(EffectSize::cohen_d(0.8));
        assert!(r.is_practically_significant(0.5));
        assert!(!r.is_practically_significant(1.0));
    }

    #[test]
    fn test_test_statistic_display() {
        let t = TestStatistic::t_score(2.5, 30.0);
        let s = format!("{}", t);
        assert!(s.contains("t="));
        assert!(s.contains("2.5"));
    }

    #[test]
    fn test_benjamini_hochberg() {
        let results: Vec<HypothesisTestResult> = vec![0.01, 0.03, 0.04, 0.10].into_iter().map(|p| {
            HypothesisTestResult::new("t", TestStatistic::z_score(2.0),
                PValue::new(p).unwrap(), 0.05)
        }).collect();
        let adjusted = apply_fwer_correction(&results, FWERControl::BenjaminiHochberg);
        assert_eq!(adjusted.len(), 4);
        for adj in &adjusted {
            assert!(adj.value() >= 0.0 && adj.value() <= 1.0);
        }
    }
}
