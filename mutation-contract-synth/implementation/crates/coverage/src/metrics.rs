//! Coverage metrics and reporting.
//!
//! Aggregates all coverage metrics (scores, subsumption stats, dominator stats,
//! equivalence stats) into a unified report with trend analysis, thresholds,
//! and alerting.

use crate::dominator::DominatorStats;
use crate::equivalence::EquivalenceStats;
use crate::scoring::MutationScore;
use crate::subsumption::SubsumptionStats;
use crate::{CoverageError, KillMatrix, MutantId, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ---------------------------------------------------------------------------
// CoverageMetrics
// ---------------------------------------------------------------------------

/// Aggregated coverage metrics from all analysis phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    /// Overall mutation score.
    pub mutation_score: MutationScore,
    /// Subsumption analysis stats (if run).
    pub subsumption: Option<SubsumptionStats>,
    /// Dominator set stats (if computed).
    pub dominator: Option<DominatorStats>,
    /// Equivalence detection stats (if run).
    pub equivalence: Option<EquivalenceStats>,
    /// Total mutants analyzed.
    pub total_mutants: usize,
    /// Total tests executed.
    pub total_tests: usize,
    /// Computation timestamp.
    pub timestamp: String,
    /// Run identifier.
    pub run_id: String,
    /// Custom key-value annotations.
    pub annotations: BTreeMap<String, String>,
}

impl CoverageMetrics {
    /// Create basic metrics from a kill matrix.
    pub fn from_kill_matrix(km: &KillMatrix, equivalent: &BTreeSet<usize>) -> Self {
        Self {
            mutation_score: MutationScore::from_kill_matrix(km, equivalent),
            subsumption: None,
            dominator: None,
            equivalence: None,
            total_mutants: km.num_mutants(),
            total_tests: km.num_tests(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            run_id: uuid::Uuid::new_v4().to_string(),
            annotations: BTreeMap::new(),
        }
    }

    pub fn with_subsumption(mut self, stats: SubsumptionStats) -> Self {
        self.subsumption = Some(stats);
        self
    }

    pub fn with_dominator(mut self, stats: DominatorStats) -> Self {
        self.dominator = Some(stats);
        self
    }

    pub fn with_equivalence(mut self, stats: EquivalenceStats) -> Self {
        self.equivalence = Some(stats);
        self
    }

    pub fn annotate(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    /// Overall score as a percentage.
    pub fn score_percent(&self) -> f64 {
        self.mutation_score.score() * 100.0
    }

    /// Dominator reduction factor (if available).
    pub fn reduction_factor(&self) -> Option<f64> {
        self.dominator.as_ref().map(|d| d.reduction_factor)
    }

    /// Equivalence rate (if available).
    pub fn equivalence_rate(&self) -> Option<f64> {
        self.equivalence.as_ref().map(|e| e.equivalence_rate())
    }
}

// ---------------------------------------------------------------------------
// MetricDelta
// ---------------------------------------------------------------------------

/// Difference between two metric snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub from_run: String,
    pub to_run: String,
    pub score_delta: f64,
    pub killed_delta: i64,
    pub total_delta: i64,
    pub equivalent_delta: i64,
    pub test_delta: i64,
    pub subsumption_edge_delta: Option<i64>,
    pub dominator_size_delta: Option<i64>,
}

impl MetricDelta {
    /// Compute the delta between two metrics.
    pub fn compute(from: &CoverageMetrics, to: &CoverageMetrics) -> Self {
        let sub_delta = match (&from.subsumption, &to.subsumption) {
            (Some(a), Some(b)) => Some(b.reduced_edges as i64 - a.reduced_edges as i64),
            _ => None,
        };
        let dom_delta = match (&from.dominator, &to.dominator) {
            (Some(a), Some(b)) => Some(b.dominator_size as i64 - a.dominator_size as i64),
            _ => None,
        };

        MetricDelta {
            from_run: from.run_id.clone(),
            to_run: to.run_id.clone(),
            score_delta: to.mutation_score.score() - from.mutation_score.score(),
            killed_delta: to.mutation_score.killed as i64 - from.mutation_score.killed as i64,
            total_delta: to.total_mutants as i64 - from.total_mutants as i64,
            equivalent_delta: to.mutation_score.equivalent as i64
                - from.mutation_score.equivalent as i64,
            test_delta: to.total_tests as i64 - from.total_tests as i64,
            subsumption_edge_delta: sub_delta,
            dominator_size_delta: dom_delta,
        }
    }

    /// Is the score improving?
    pub fn is_improving(&self) -> bool {
        self.score_delta > 0.001
    }

    /// Is the score declining?
    pub fn is_declining(&self) -> bool {
        self.score_delta < -0.001
    }
}

// ---------------------------------------------------------------------------
// MetricThreshold
// ---------------------------------------------------------------------------

/// Threshold configuration for metric alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricThreshold {
    pub name: String,
    pub metric: ThresholdMetric,
    pub operator: ThresholdOp,
    pub value: f64,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMetric {
    MutationScore,
    EquivalenceRate,
    ReductionFactor,
    TestCount,
    MutantCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdOp {
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl MetricThreshold {
    /// Check if a metric value violates this threshold.
    pub fn check(&self, metrics: &CoverageMetrics) -> Option<MetricAlert> {
        let actual = match self.metric {
            ThresholdMetric::MutationScore => metrics.mutation_score.score(),
            ThresholdMetric::EquivalenceRate => metrics.equivalence_rate().unwrap_or(0.0),
            ThresholdMetric::ReductionFactor => metrics.reduction_factor().unwrap_or(1.0),
            ThresholdMetric::TestCount => metrics.total_tests as f64,
            ThresholdMetric::MutantCount => metrics.total_mutants as f64,
        };

        let violated = match self.operator {
            ThresholdOp::LessThan => actual < self.value,
            ThresholdOp::GreaterThan => actual > self.value,
            ThresholdOp::LessOrEqual => actual <= self.value,
            ThresholdOp::GreaterOrEqual => actual >= self.value,
        };

        if violated {
            Some(MetricAlert {
                threshold_name: self.name.clone(),
                severity: self.severity,
                message: format!(
                    "{}: actual={:.4}, threshold={:.4}",
                    self.name, actual, self.value
                ),
                actual_value: actual,
                threshold_value: self.value,
            })
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// MetricAlert
// ---------------------------------------------------------------------------

/// An alert triggered by a threshold violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAlert {
    pub threshold_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub actual_value: f64,
    pub threshold_value: f64,
}

impl fmt::Display for MetricAlert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.message)
    }
}

// ---------------------------------------------------------------------------
// Metric checker
// ---------------------------------------------------------------------------

/// Checks metrics against a set of thresholds.
pub struct MetricChecker {
    thresholds: Vec<MetricThreshold>,
}

impl MetricChecker {
    pub fn new() -> Self {
        Self {
            thresholds: Vec::new(),
        }
    }

    pub fn add_threshold(&mut self, threshold: MetricThreshold) {
        self.thresholds.push(threshold);
    }

    /// Add a common "minimum score" threshold.
    pub fn require_minimum_score(&mut self, min_score: f64) {
        self.thresholds.push(MetricThreshold {
            name: "minimum_mutation_score".into(),
            metric: ThresholdMetric::MutationScore,
            operator: ThresholdOp::LessThan,
            value: min_score,
            severity: AlertSeverity::Error,
        });
    }

    /// Check all thresholds, returning triggered alerts.
    pub fn check(&self, metrics: &CoverageMetrics) -> Vec<MetricAlert> {
        self.thresholds
            .iter()
            .filter_map(|t| t.check(metrics))
            .collect()
    }

    /// Check and partition by severity.
    pub fn check_by_severity(
        &self,
        metrics: &CoverageMetrics,
    ) -> BTreeMap<String, Vec<MetricAlert>> {
        let alerts = self.check(metrics);
        let mut by_sev: BTreeMap<String, Vec<MetricAlert>> = BTreeMap::new();
        for alert in alerts {
            by_sev
                .entry(format!("{}", alert.severity))
                .or_default()
                .push(alert);
        }
        by_sev
    }
}

impl Default for MetricChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trend analysis
// ---------------------------------------------------------------------------

/// Trend analysis over multiple metric snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub snapshots: Vec<CoverageMetrics>,
}

impl MetricTrend {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn add(&mut self, metrics: CoverageMetrics) {
        self.snapshots.push(metrics);
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Compute deltas between consecutive snapshots.
    pub fn deltas(&self) -> Vec<MetricDelta> {
        self.snapshots
            .windows(2)
            .map(|w| MetricDelta::compute(&w[0], &w[1]))
            .collect()
    }

    /// Latest metrics.
    pub fn latest(&self) -> Option<&CoverageMetrics> {
        self.snapshots.last()
    }

    /// Score values over time.
    pub fn scores(&self) -> Vec<f64> {
        self.snapshots
            .iter()
            .map(|m| m.mutation_score.score())
            .collect()
    }

    /// Average score across all snapshots.
    pub fn average_score(&self) -> f64 {
        let s = self.scores();
        if s.is_empty() {
            0.0
        } else {
            s.iter().sum::<f64>() / s.len() as f64
        }
    }

    /// Score trend slope (linear regression).
    pub fn score_slope(&self) -> f64 {
        let scores = self.scores();
        let n = scores.len();
        if n < 2 {
            return 0.0;
        }
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let xm = xs.iter().sum::<f64>() / n as f64;
        let ym = scores.iter().sum::<f64>() / n as f64;
        let num: f64 = xs
            .iter()
            .zip(scores.iter())
            .map(|(x, y)| (x - xm) * (y - ym))
            .sum();
        let den: f64 = xs.iter().map(|x| (x - xm).powi(2)).sum();
        if den.abs() < 1e-12 {
            0.0
        } else {
            num / den
        }
    }
}

impl Default for MetricTrend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Summary report
// ---------------------------------------------------------------------------

/// A summary report of coverage metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryReport {
    pub title: String,
    pub sections: Vec<ReportSection>,
}

/// A section of a summary report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub heading: String,
    pub lines: Vec<String>,
}

impl SummaryReport {
    /// Generate a summary report from coverage metrics.
    pub fn generate(metrics: &CoverageMetrics) -> Self {
        let mut sections = Vec::new();

        // Overall score section.
        let mut score_lines = vec![
            format!("Mutation Score: {:.1}%", metrics.score_percent()),
            format!(
                "Killed: {}/{}",
                metrics.mutation_score.killed,
                metrics.mutation_score.killable()
            ),
            format!("Equivalent: {}", metrics.mutation_score.equivalent),
            format!("Total Mutants: {}", metrics.total_mutants),
            format!("Total Tests: {}", metrics.total_tests),
        ];
        sections.push(ReportSection {
            heading: "Overall".into(),
            lines: score_lines,
        });

        // Subsumption section.
        if let Some(ref sub) = metrics.subsumption {
            sections.push(ReportSection {
                heading: "Subsumption".into(),
                lines: vec![
                    format!("Edges (reduced): {}", sub.reduced_edges),
                    format!("Equivalence classes: {}", sub.equivalence_classes),
                    format!("Roots: {}", sub.root_count),
                    format!("Max depth: {}", sub.max_depth),
                ],
            });
        }

        // Dominator section.
        if let Some(ref dom) = metrics.dominator {
            sections.push(ReportSection {
                heading: "Dominator Set".into(),
                lines: vec![
                    format!("Size: {}", dom.dominator_size),
                    format!("Reduction: {:.1}x", dom.reduction_factor),
                    format!("Algorithm: {}", dom.algorithm),
                ],
            });
        }

        // Equivalence section.
        if let Some(ref eq) = metrics.equivalence {
            sections.push(ReportSection {
                heading: "Equivalence".into(),
                lines: vec![
                    format!("Checked: {}", eq.total_checked),
                    format!("Equivalent: {}", eq.equivalent_count),
                    format!("Rate: {:.1}%", eq.equivalence_rate() * 100.0),
                ],
            });
        }

        SummaryReport {
            title: "Coverage Analysis Report".into(),
            sections,
        }
    }

    /// Render as plain text.
    pub fn to_text(&self) -> String {
        let mut out = format!("=== {} ===\n\n", self.title);
        for section in &self.sections {
            out.push_str(&format!("--- {} ---\n", section.heading));
            for line in &section.lines {
                out.push_str(&format!("  {}\n", line));
            }
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_kill_matrix, MutantId};

    fn basic_metrics() -> CoverageMetrics {
        let km = make_test_kill_matrix(3, 5, &[(0, 0), (1, 1), (2, 2)]);
        CoverageMetrics::from_kill_matrix(&km, &BTreeSet::new())
    }

    #[test]
    fn test_metrics_from_km() {
        let m = basic_metrics();
        assert_eq!(m.total_mutants, 5);
        assert_eq!(m.total_tests, 3);
        assert_eq!(m.mutation_score.killed, 3);
    }

    #[test]
    fn test_score_percent() {
        let m = basic_metrics();
        assert!((m.score_percent() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_annotate() {
        let m = basic_metrics().annotate("branch", "main");
        assert_eq!(m.annotations.get("branch").unwrap(), "main");
    }

    #[test]
    fn test_delta() {
        let km1 = make_test_kill_matrix(3, 5, &[(0, 0), (1, 1)]);
        let km2 = make_test_kill_matrix(3, 5, &[(0, 0), (1, 1), (2, 2)]);
        let m1 = CoverageMetrics::from_kill_matrix(&km1, &BTreeSet::new());
        let m2 = CoverageMetrics::from_kill_matrix(&km2, &BTreeSet::new());
        let delta = MetricDelta::compute(&m1, &m2);
        assert!(delta.is_improving());
        assert_eq!(delta.killed_delta, 1);
    }

    #[test]
    fn test_delta_declining() {
        let km1 = make_test_kill_matrix(3, 5, &[(0, 0), (1, 1), (2, 2)]);
        let km2 = make_test_kill_matrix(3, 5, &[(0, 0)]);
        let m1 = CoverageMetrics::from_kill_matrix(&km1, &BTreeSet::new());
        let m2 = CoverageMetrics::from_kill_matrix(&km2, &BTreeSet::new());
        let delta = MetricDelta::compute(&m1, &m2);
        assert!(delta.is_declining());
    }

    #[test]
    fn test_threshold_check_triggered() {
        let m = basic_metrics(); // 60% score
        let threshold = MetricThreshold {
            name: "min_score".into(),
            metric: ThresholdMetric::MutationScore,
            operator: ThresholdOp::LessThan,
            value: 0.80,
            severity: AlertSeverity::Error,
        };
        let alert = threshold.check(&m);
        assert!(alert.is_some());
    }

    #[test]
    fn test_threshold_check_not_triggered() {
        let m = basic_metrics();
        let threshold = MetricThreshold {
            name: "min_score".into(),
            metric: ThresholdMetric::MutationScore,
            operator: ThresholdOp::LessThan,
            value: 0.50,
            severity: AlertSeverity::Warning,
        };
        assert!(threshold.check(&m).is_none());
    }

    #[test]
    fn test_metric_checker() {
        let m = basic_metrics();
        let mut checker = MetricChecker::new();
        checker.require_minimum_score(0.80);
        let alerts = checker.check(&m);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_metric_checker_by_severity() {
        let m = basic_metrics();
        let mut checker = MetricChecker::new();
        checker.require_minimum_score(0.80);
        let by_sev = checker.check_by_severity(&m);
        assert!(by_sev.contains_key("ERROR"));
    }

    #[test]
    fn test_trend() {
        let mut trend = MetricTrend::new();
        for i in 0..5 {
            let km = make_test_kill_matrix(3, 5, &(0..=i).map(|j| (j % 3, j)).collect::<Vec<_>>());
            trend.add(CoverageMetrics::from_kill_matrix(&km, &BTreeSet::new()));
        }
        assert_eq!(trend.len(), 5);
        assert!(trend.score_slope() > 0.0);
    }

    #[test]
    fn test_trend_deltas() {
        let mut trend = MetricTrend::new();
        let km1 = make_test_kill_matrix(2, 3, &[(0, 0)]);
        let km2 = make_test_kill_matrix(2, 3, &[(0, 0), (1, 1)]);
        trend.add(CoverageMetrics::from_kill_matrix(&km1, &BTreeSet::new()));
        trend.add(CoverageMetrics::from_kill_matrix(&km2, &BTreeSet::new()));
        let deltas = trend.deltas();
        assert_eq!(deltas.len(), 1);
        assert!(deltas[0].is_improving());
    }

    #[test]
    fn test_trend_average() {
        let mut trend = MetricTrend::new();
        let km = make_test_kill_matrix(2, 2, &[(0, 0), (1, 1)]);
        trend.add(CoverageMetrics::from_kill_matrix(&km, &BTreeSet::new()));
        assert!((trend.average_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_report() {
        let m = basic_metrics();
        let report = SummaryReport::generate(&m);
        assert!(!report.sections.is_empty());
        let text = report.to_text();
        assert!(text.contains("Mutation Score"));
        assert!(text.contains("60.0%"));
    }

    #[test]
    fn test_summary_with_all_sections() {
        let mut m = basic_metrics();
        m.subsumption = Some(SubsumptionStats {
            total_mutants: 5,
            total_edges: 10,
            reduced_edges: 5,
            equivalence_classes: 3,
            max_class_size: 2,
            root_count: 2,
            leaf_count: 3,
            max_depth: 2,
            dynamic_detections: 10,
            static_confirmations: 0,
            static_refutations: 0,
            smt_unknowns: 0,
            per_operator: BTreeMap::new(),
        });
        m.dominator = Some(DominatorStats {
            algorithm: "greedy".into(),
            input_killed: 3,
            dominator_size: 2,
            reduction_factor: 1.5,
            computation_time_ms: 10,
            per_operator: BTreeMap::new(),
            quality: crate::dominator::DominatorQuality {
                dominator_size: 2,
                total_killed: 3,
                reduction_factor: 1.5,
                is_valid: true,
                spec_equivalent: true,
                test_coverage: 1.0,
                avg_representation: 0.5,
                max_representation: 1,
                min_representation: 0,
            },
        });
        let report = SummaryReport::generate(&m);
        assert!(report.sections.len() >= 3);
    }

    #[test]
    fn test_alert_display() {
        let alert = MetricAlert {
            threshold_name: "test".into(),
            severity: AlertSeverity::Warning,
            message: "score too low".into(),
            actual_value: 0.5,
            threshold_value: 0.8,
        };
        let s = format!("{}", alert);
        assert!(s.contains("WARNING"));
    }

    #[test]
    fn test_empty_trend() {
        let trend = MetricTrend::new();
        assert!(trend.is_empty());
        assert_eq!(trend.average_score(), 0.0);
        assert_eq!(trend.score_slope(), 0.0);
    }
}
