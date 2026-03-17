//! Mutation score computation and analysis.
//!
//! Provides various mutation score metrics including traditional mutation score,
//! per-operator scores, per-function scores, test effectiveness, and score
//! trend analysis over time.

use crate::{
    CoverageError, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result, TestId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

// ---------------------------------------------------------------------------
// MutationScore
// ---------------------------------------------------------------------------

/// Core mutation score and related metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationScore {
    /// Total number of mutants generated.
    pub total_mutants: usize,
    /// Number of mutants detected (killed) by the test suite.
    pub killed: usize,
    /// Number of surviving (alive) mutants.
    pub survived: usize,
    /// Number of equivalent mutants (cannot be detected).
    pub equivalent: usize,
    /// Number of timed-out mutants.
    pub timed_out: usize,
    /// Number of error mutants.
    pub errors: usize,
}

impl MutationScore {
    /// Create a score from raw counts.
    pub fn new(total: usize, killed: usize, equivalent: usize) -> Self {
        let survived = total.saturating_sub(killed).saturating_sub(equivalent);
        Self {
            total_mutants: total,
            killed,
            survived,
            equivalent,
            timed_out: 0,
            errors: 0,
        }
    }

    /// Compute from a kill matrix and known equivalent mutant indices.
    pub fn from_kill_matrix(km: &KillMatrix, equivalent: &BTreeSet<usize>) -> Self {
        let total = km.num_mutants();
        let killed = km.killed_set().len();
        let eq_count = equivalent.len();
        let survived = total.saturating_sub(killed).saturating_sub(eq_count);
        Self {
            total_mutants: total,
            killed,
            survived,
            equivalent: eq_count,
            timed_out: 0,
            errors: 0,
        }
    }

    /// Traditional mutation score: killed / (total - equivalent).
    pub fn score(&self) -> f64 {
        let denominator = self.total_mutants.saturating_sub(self.equivalent);
        if denominator == 0 {
            1.0
        } else {
            self.killed as f64 / denominator as f64
        }
    }

    /// Raw score without accounting for equivalents: killed / total.
    pub fn raw_score(&self) -> f64 {
        if self.total_mutants == 0 {
            1.0
        } else {
            self.killed as f64 / self.total_mutants as f64
        }
    }

    /// Number of killable mutants (total - equivalent).
    pub fn killable(&self) -> usize {
        self.total_mutants.saturating_sub(self.equivalent)
    }

    /// Whether the test suite achieves the given minimum score.
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score() >= threshold
    }

    /// Whether mutation-adequate (score == 1.0).
    pub fn is_adequate(&self) -> bool {
        self.killed >= self.killable()
    }

    /// Score as a percentage string.
    pub fn score_percent(&self) -> String {
        format!("{:.1}%", self.score() * 100.0)
    }
}

impl fmt::Display for MutationScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}/{} killed, {} equiv)",
            self.score_percent(),
            self.killed,
            self.killable(),
            self.equivalent
        )
    }
}

// ---------------------------------------------------------------------------
// ScoreBreakdown
// ---------------------------------------------------------------------------

/// Detailed breakdown of mutation scores by various dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub overall: MutationScore,
    pub per_operator: BTreeMap<String, OperatorScore>,
    pub per_function: BTreeMap<String, FunctionScore>,
    pub per_test: Vec<TestEffectiveness>,
}

impl ScoreBreakdown {
    /// Compute a full breakdown from kill matrix, descriptors, and equivalents.
    pub fn compute(
        km: &KillMatrix,
        descriptors: &HashMap<MutantId, MutantDescriptor>,
        equivalent: &BTreeSet<usize>,
    ) -> Self {
        let overall = MutationScore::from_kill_matrix(km, equivalent);
        let per_operator = compute_operator_scores(km, descriptors, equivalent);
        let per_function = compute_function_scores(km, descriptors, equivalent);
        let per_test = compute_test_effectiveness(km, descriptors);
        ScoreBreakdown {
            overall,
            per_operator,
            per_function,
            per_test,
        }
    }

    /// Get the operator with the lowest score.
    pub fn weakest_operator(&self) -> Option<(&str, f64)> {
        self.per_operator
            .iter()
            .min_by(|a, b| a.1.score.score().partial_cmp(&b.1.score.score()).unwrap())
            .map(|(name, os)| (name.as_str(), os.score.score()))
    }

    /// Get the function with the lowest score.
    pub fn weakest_function(&self) -> Option<(&str, f64)> {
        self.per_function
            .iter()
            .min_by(|a, b| a.1.score.score().partial_cmp(&b.1.score.score()).unwrap())
            .map(|(name, fs)| (name.as_str(), fs.score.score()))
    }

    /// Get the most effective test.
    pub fn most_effective_test(&self) -> Option<&TestEffectiveness> {
        self.per_test.iter().max_by_key(|te| te.unique_kills)
    }

    /// Get the least effective tests (those that add no unique kills).
    pub fn redundant_tests(&self) -> Vec<&TestEffectiveness> {
        self.per_test
            .iter()
            .filter(|te| te.unique_kills == 0)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// OperatorScore
// ---------------------------------------------------------------------------

/// Score for a specific mutation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorScore {
    pub operator: String,
    pub score: MutationScore,
    /// Fraction of all mutants from this operator.
    pub fraction_of_total: f64,
    /// This operator's contribution to the overall score gap.
    pub score_gap_contribution: f64,
}

fn compute_operator_scores(
    km: &KillMatrix,
    descriptors: &HashMap<MutantId, MutantDescriptor>,
    equivalent: &BTreeSet<usize>,
) -> BTreeMap<String, OperatorScore> {
    let mut by_op: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new(); // (total, killed, equiv)
    let killed_set = km.killed_set();

    for (idx, mid) in km.mutants.iter().enumerate() {
        if let Some(desc) = descriptors.get(mid) {
            let name = desc.operator.short_name().to_string();
            let entry = by_op.entry(name).or_insert((0, 0, 0));
            entry.0 += 1;
            if killed_set.contains(&idx) {
                entry.1 += 1;
            }
            if equivalent.contains(&idx) {
                entry.2 += 1;
            }
        }
    }

    let total = km.num_mutants().max(1) as f64;
    let overall_gap = {
        let ks = MutationScore::from_kill_matrix(km, equivalent);
        1.0 - ks.score()
    };

    by_op
        .into_iter()
        .map(|(name, (t, k, e))| {
            let score = MutationScore::new(t, k, e);
            let fraction = t as f64 / total;
            let op_gap = 1.0 - score.score();
            let contribution = if overall_gap > 0.0 {
                (op_gap * fraction) / overall_gap
            } else {
                0.0
            };
            (
                name.clone(),
                OperatorScore {
                    operator: name,
                    score,
                    fraction_of_total: fraction,
                    score_gap_contribution: contribution,
                },
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FunctionScore
// ---------------------------------------------------------------------------

/// Score for a specific function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionScore {
    pub function_name: String,
    pub score: MutationScore,
    pub mutant_ids: Vec<MutantId>,
}

fn compute_function_scores(
    km: &KillMatrix,
    descriptors: &HashMap<MutantId, MutantDescriptor>,
    equivalent: &BTreeSet<usize>,
) -> BTreeMap<String, FunctionScore> {
    let mut by_func: BTreeMap<String, (Vec<usize>, Vec<MutantId>)> = BTreeMap::new();
    let killed_set = km.killed_set();

    for (idx, mid) in km.mutants.iter().enumerate() {
        if let Some(desc) = descriptors.get(mid) {
            let fname = desc.site.function_name.clone();
            let entry = by_func
                .entry(fname)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(idx);
            entry.1.push(mid.clone());
        }
    }

    by_func
        .into_iter()
        .map(|(fname, (indices, ids))| {
            let total = indices.len();
            let k = indices.iter().filter(|i| killed_set.contains(i)).count();
            let e = indices.iter().filter(|i| equivalent.contains(i)).count();
            let score = MutationScore::new(total, k, e);
            (
                fname.clone(),
                FunctionScore {
                    function_name: fname,
                    score,
                    mutant_ids: ids,
                },
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// TestEffectiveness
// ---------------------------------------------------------------------------

/// How effective a single test is at mutation detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEffectiveness {
    pub test_id: TestId,
    /// Total mutants this test detects.
    pub total_kills: usize,
    /// Mutants detected ONLY by this test.
    pub unique_kills: usize,
    /// Fraction of all killed mutants this test detects.
    pub kill_fraction: f64,
    /// Operators this test is most effective against.
    pub effective_operators: Vec<String>,
}

fn compute_test_effectiveness(
    km: &KillMatrix,
    descriptors: &HashMap<MutantId, MutantDescriptor>,
) -> Vec<TestEffectiveness> {
    let total_killed = km.killed_set().len().max(1) as f64;
    let kill_sets = km.kill_sets();

    (0..km.num_tests())
        .map(|t| {
            let kills = km.killed_mutants(t);
            let total = kills.len();

            // Unique: mutants detected only by this test.
            let unique = kills.iter().filter(|&&m| kill_sets[m].len() == 1).count();

            let kill_fraction = total as f64 / total_killed;

            // Most effective operators: count kills per operator.
            let mut op_counts: BTreeMap<String, usize> = BTreeMap::new();
            for &m in &kills {
                if let Some(desc) = descriptors.get(&km.mutants[m]) {
                    *op_counts
                        .entry(desc.operator.short_name().to_string())
                        .or_default() += 1;
                }
            }
            let mut effective_ops: Vec<(String, usize)> = op_counts.into_iter().collect();
            effective_ops.sort_by(|a, b| b.1.cmp(&a.1));
            let effective_operators: Vec<String> =
                effective_ops.into_iter().take(3).map(|(n, _)| n).collect();

            TestEffectiveness {
                test_id: km.tests[t].clone(),
                total_kills: total,
                unique_kills: unique,
                kill_fraction,
                effective_operators,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Score trends
// ---------------------------------------------------------------------------

/// A snapshot of mutation score at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreSnapshot {
    pub label: String,
    pub timestamp: Option<String>,
    pub score: MutationScore,
}

/// Trend analysis over multiple score snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreTrend {
    pub snapshots: Vec<ScoreSnapshot>,
}

impl ScoreTrend {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, label: impl Into<String>, score: MutationScore) {
        self.snapshots.push(ScoreSnapshot {
            label: label.into(),
            timestamp: None,
            score,
        });
    }

    pub fn add_snapshot_with_time(
        &mut self,
        label: impl Into<String>,
        timestamp: impl Into<String>,
        score: MutationScore,
    ) {
        self.snapshots.push(ScoreSnapshot {
            label: label.into(),
            timestamp: Some(timestamp.into()),
            score,
        });
    }

    /// Linear regression slope of score over snapshots.
    pub fn trend_slope(&self) -> f64 {
        let n = self.snapshots.len();
        if n < 2 {
            return 0.0;
        }
        let scores: Vec<f64> = self.snapshots.iter().map(|s| s.score.score()).collect();
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x_mean = xs.iter().sum::<f64>() / n as f64;
        let y_mean = scores.iter().sum::<f64>() / n as f64;
        let num: f64 = xs
            .iter()
            .zip(scores.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let den: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
        if den.abs() < 1e-12 {
            0.0
        } else {
            num / den
        }
    }

    /// Is the score trend improving?
    pub fn is_improving(&self) -> bool {
        self.trend_slope() > 0.001
    }

    /// Is the score trend declining?
    pub fn is_declining(&self) -> bool {
        self.trend_slope() < -0.001
    }

    /// Latest score.
    pub fn latest(&self) -> Option<&ScoreSnapshot> {
        self.snapshots.last()
    }

    /// Delta between first and last snapshot.
    pub fn overall_delta(&self) -> Option<f64> {
        if self.snapshots.len() < 2 {
            return None;
        }
        let first = self.snapshots.first().unwrap().score.score();
        let last = self.snapshots.last().unwrap().score.score();
        Some(last - first)
    }
}

impl Default for ScoreTrend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Score comparison
// ---------------------------------------------------------------------------

/// Comparison between two score computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComparison {
    pub label_a: String,
    pub label_b: String,
    pub score_a: f64,
    pub score_b: f64,
    pub delta: f64,
    pub killed_delta: i64,
    pub new_kills: Vec<MutantId>,
    pub lost_kills: Vec<MutantId>,
}

impl ScoreComparison {
    /// Compare two kill matrices (e.g., before and after test suite changes).
    pub fn compare(
        label_a: impl Into<String>,
        km_a: &KillMatrix,
        label_b: impl Into<String>,
        km_b: &KillMatrix,
        equivalent: &BTreeSet<usize>,
    ) -> Self {
        let score_a_obj = MutationScore::from_kill_matrix(km_a, equivalent);
        let score_b_obj = MutationScore::from_kill_matrix(km_b, equivalent);
        let sa = score_a_obj.score();
        let sb = score_b_obj.score();

        let killed_a = km_a.killed_set();
        let killed_b = km_b.killed_set();

        // New kills: in B but not in A (by mutant ID).
        let id_killed_a: BTreeSet<&MutantId> = killed_a.iter().map(|&i| &km_a.mutants[i]).collect();
        let id_killed_b: BTreeSet<&MutantId> = killed_b.iter().map(|&i| &km_b.mutants[i]).collect();

        let new_kills: Vec<MutantId> = id_killed_b
            .difference(&id_killed_a)
            .map(|&&ref id| id.clone())
            .collect();
        let lost_kills: Vec<MutantId> = id_killed_a
            .difference(&id_killed_b)
            .map(|&&ref id| id.clone())
            .collect();

        ScoreComparison {
            label_a: label_a.into(),
            label_b: label_b.into(),
            score_a: sa,
            score_b: sb,
            delta: sb - sa,
            killed_delta: score_b_obj.killed as i64 - score_a_obj.killed as i64,
            new_kills,
            lost_kills,
        }
    }
}

// ---------------------------------------------------------------------------
// Score decomposition
// ---------------------------------------------------------------------------

/// Decomposition of the score gap by operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDecomposition {
    pub overall_gap: f64,
    pub contributions: Vec<GapContribution>,
}

/// How much one factor contributes to the overall score gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapContribution {
    pub label: String,
    pub gap: f64,
    pub weight: f64,
    pub contribution: f64,
}

impl ScoreDecomposition {
    /// Decompose the score gap by operator.
    pub fn by_operator(breakdown: &ScoreBreakdown) -> Self {
        let overall_gap = 1.0 - breakdown.overall.score();
        let mut contributions = Vec::new();
        for (name, os) in &breakdown.per_operator {
            let op_gap = 1.0 - os.score.score();
            let weight = os.fraction_of_total;
            let contribution = op_gap * weight;
            contributions.push(GapContribution {
                label: name.clone(),
                gap: op_gap,
                weight,
                contribution,
            });
        }
        contributions.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap());
        ScoreDecomposition {
            overall_gap,
            contributions,
        }
    }

    /// Decompose the score gap by function.
    pub fn by_function(breakdown: &ScoreBreakdown) -> Self {
        let overall_gap = 1.0 - breakdown.overall.score();
        let total = breakdown.overall.total_mutants.max(1) as f64;
        let mut contributions = Vec::new();
        for (name, fs) in &breakdown.per_function {
            let func_gap = 1.0 - fs.score.score();
            let weight = fs.score.total_mutants as f64 / total;
            let contribution = func_gap * weight;
            contributions.push(GapContribution {
                label: name.clone(),
                gap: func_gap,
                weight,
                contribution,
            });
        }
        contributions.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap());
        ScoreDecomposition {
            overall_gap,
            contributions,
        }
    }

    /// Top N contributors to the score gap.
    pub fn top_contributors(&self, n: usize) -> Vec<&GapContribution> {
        self.contributions.iter().take(n).collect()
    }
}

// ---------------------------------------------------------------------------
// Visualization data
// ---------------------------------------------------------------------------

/// Data point for score visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreChartData {
    pub labels: Vec<String>,
    pub scores: Vec<f64>,
    pub killed_counts: Vec<usize>,
    pub total_counts: Vec<usize>,
}

impl ScoreChartData {
    /// Build chart data from per-operator scores.
    pub fn from_operator_scores(scores: &BTreeMap<String, OperatorScore>) -> Self {
        let mut labels = Vec::new();
        let mut score_vals = Vec::new();
        let mut killed = Vec::new();
        let mut totals = Vec::new();
        for (name, os) in scores {
            labels.push(name.clone());
            score_vals.push(os.score.score());
            killed.push(os.score.killed);
            totals.push(os.score.total_mutants);
        }
        ScoreChartData {
            labels,
            scores: score_vals,
            killed_counts: killed,
            total_counts: totals,
        }
    }

    /// Build chart data from a trend.
    pub fn from_trend(trend: &ScoreTrend) -> Self {
        let labels: Vec<String> = trend.snapshots.iter().map(|s| s.label.clone()).collect();
        let scores: Vec<f64> = trend.snapshots.iter().map(|s| s.score.score()).collect();
        let killed: Vec<usize> = trend.snapshots.iter().map(|s| s.score.killed).collect();
        let totals: Vec<usize> = trend
            .snapshots
            .iter()
            .map(|s| s.score.total_mutants)
            .collect();
        ScoreChartData {
            labels,
            scores,
            killed_counts: killed,
            total_counts: totals,
        }
    }
}

// ---------------------------------------------------------------------------
// Minimum adequate score
// ---------------------------------------------------------------------------

/// Configuration for minimum mutation score requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdequacyThreshold {
    pub minimum_overall: f64,
    pub minimum_per_operator: Option<BTreeMap<String, f64>>,
    pub minimum_per_function: Option<f64>,
}

impl Default for AdequacyThreshold {
    fn default() -> Self {
        Self {
            minimum_overall: 0.80,
            minimum_per_operator: None,
            minimum_per_function: None,
        }
    }
}

impl AdequacyThreshold {
    /// Check if a breakdown meets the threshold.
    pub fn check(&self, breakdown: &ScoreBreakdown) -> ThresholdResult {
        let mut violations = Vec::new();

        if breakdown.overall.score() < self.minimum_overall {
            violations.push(ThresholdViolation {
                category: "overall".into(),
                name: "overall".into(),
                expected: self.minimum_overall,
                actual: breakdown.overall.score(),
            });
        }

        if let Some(ref per_op) = self.minimum_per_operator {
            for (op, &min) in per_op {
                if let Some(os) = breakdown.per_operator.get(op) {
                    if os.score.score() < min {
                        violations.push(ThresholdViolation {
                            category: "operator".into(),
                            name: op.clone(),
                            expected: min,
                            actual: os.score.score(),
                        });
                    }
                }
            }
        }

        if let Some(min_func) = self.minimum_per_function {
            for (fname, fs) in &breakdown.per_function {
                if fs.score.score() < min_func {
                    violations.push(ThresholdViolation {
                        category: "function".into(),
                        name: fname.clone(),
                        expected: min_func,
                        actual: fs.score.score(),
                    });
                }
            }
        }

        ThresholdResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

/// Result of checking a score against thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdResult {
    pub passed: bool,
    pub violations: Vec<ThresholdViolation>,
}

/// A single threshold violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdViolation {
    pub category: String,
    pub name: String,
    pub expected: f64,
    pub actual: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator, MutationSite,
    };

    fn descriptors_for(km: &KillMatrix) -> HashMap<MutantId, MutantDescriptor> {
        km.mutants
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let op = match i % 3 {
                    0 => MutationOperator::AOR,
                    1 => MutationOperator::ROR,
                    _ => MutationOperator::COR,
                };
                let func = format!("func_{}", i % 2);
                let desc = MutantDescriptor::new(
                    id.clone(),
                    op,
                    MutationSite::new("test.c", func, i + 1, 1),
                    "orig",
                    "repl",
                );
                (id.clone(), desc)
            })
            .collect()
    }

    #[test]
    fn test_score_basic() {
        let s = MutationScore::new(100, 80, 10);
        assert!((s.score() - 80.0 / 90.0).abs() < 1e-9);
        assert_eq!(s.killable(), 90);
        assert!(!s.is_adequate());
    }

    #[test]
    fn test_score_adequate() {
        let s = MutationScore::new(10, 8, 2);
        assert!(s.is_adequate());
        assert!((s.score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_empty() {
        let s = MutationScore::new(0, 0, 0);
        assert!((s.score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_all_equivalent() {
        let s = MutationScore::new(5, 0, 5);
        assert!((s.score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_kill_matrix() {
        let km = make_test_kill_matrix(3, 4, &[(0, 0), (1, 1), (2, 2)]);
        let eq = BTreeSet::from([3]);
        let s = MutationScore::from_kill_matrix(&km, &eq);
        assert_eq!(s.total_mutants, 4);
        assert_eq!(s.killed, 3);
        assert_eq!(s.equivalent, 1);
        assert!(s.is_adequate());
    }

    #[test]
    fn test_meets_threshold() {
        let s = MutationScore::new(100, 85, 0);
        assert!(s.meets_threshold(0.80));
        assert!(!s.meets_threshold(0.90));
    }

    #[test]
    fn test_score_display() {
        let s = MutationScore::new(100, 80, 0);
        let display = format!("{}", s);
        assert!(display.contains("80.0%"));
    }

    #[test]
    fn test_breakdown() {
        let km = make_test_kill_matrix(3, 6, &[(0, 0), (0, 1), (1, 2), (1, 3), (2, 4)]);
        let descs = descriptors_for(&km);
        let eq = BTreeSet::new();
        let bd = ScoreBreakdown::compute(&km, &descs, &eq);
        assert_eq!(bd.overall.total_mutants, 6);
        assert_eq!(bd.overall.killed, 5);
        assert!(!bd.per_operator.is_empty());
        assert!(!bd.per_function.is_empty());
        assert_eq!(bd.per_test.len(), 3);
    }

    #[test]
    fn test_weakest_operator() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (1, 1)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        let (_, score) = bd.weakest_operator().unwrap();
        assert!(score <= 1.0);
    }

    #[test]
    fn test_test_effectiveness() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let descs = descriptors_for(&km);
        let eff = compute_test_effectiveness(&km, &descs);
        assert_eq!(eff.len(), 3);
        for te in &eff {
            assert_eq!(te.total_kills, 1);
            assert_eq!(te.unique_kills, 1);
        }
    }

    #[test]
    fn test_redundant_tests() {
        // t0 and t1 detect same mutants
        let km = make_test_kill_matrix(2, 2, &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        // Neither test has unique kills
        let redundant = bd.redundant_tests();
        assert_eq!(redundant.len(), 2);
    }

    #[test]
    fn test_trend_improving() {
        let mut trend = ScoreTrend::new();
        trend.add_snapshot("v1", MutationScore::new(100, 60, 0));
        trend.add_snapshot("v2", MutationScore::new(100, 70, 0));
        trend.add_snapshot("v3", MutationScore::new(100, 80, 0));
        assert!(trend.is_improving());
        assert!(!trend.is_declining());
        assert!(trend.trend_slope() > 0.0);
    }

    #[test]
    fn test_trend_declining() {
        let mut trend = ScoreTrend::new();
        trend.add_snapshot("v1", MutationScore::new(100, 80, 0));
        trend.add_snapshot("v2", MutationScore::new(100, 70, 0));
        trend.add_snapshot("v3", MutationScore::new(100, 60, 0));
        assert!(trend.is_declining());
    }

    #[test]
    fn test_trend_flat() {
        let mut trend = ScoreTrend::new();
        trend.add_snapshot("v1", MutationScore::new(100, 80, 0));
        trend.add_snapshot("v2", MutationScore::new(100, 80, 0));
        assert!(!trend.is_improving());
        assert!(!trend.is_declining());
    }

    #[test]
    fn test_trend_delta() {
        let mut trend = ScoreTrend::new();
        trend.add_snapshot("v1", MutationScore::new(100, 50, 0));
        trend.add_snapshot("v2", MutationScore::new(100, 100, 0));
        let delta = trend.overall_delta().unwrap();
        assert!((delta - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_comparison() {
        let km_a = make_test_kill_matrix(2, 3, &[(0, 0)]);
        let km_b = make_test_kill_matrix(2, 3, &[(0, 0), (1, 1)]);
        let cmp = ScoreComparison::compare("before", &km_a, "after", &km_b, &BTreeSet::new());
        assert!(cmp.delta > 0.0);
        assert_eq!(cmp.killed_delta, 1);
        assert!(!cmp.new_kills.is_empty());
    }

    #[test]
    fn test_decomposition_by_operator() {
        let km = make_test_kill_matrix(3, 6, &[(0, 0), (1, 2), (2, 4)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        let decomp = ScoreDecomposition::by_operator(&bd);
        assert!(decomp.overall_gap >= 0.0);
        assert!(!decomp.contributions.is_empty());
    }

    #[test]
    fn test_chart_data_from_operator() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (1, 1), (0, 2)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        let chart = ScoreChartData::from_operator_scores(&bd.per_operator);
        assert_eq!(chart.labels.len(), chart.scores.len());
    }

    #[test]
    fn test_chart_data_from_trend() {
        let mut trend = ScoreTrend::new();
        trend.add_snapshot("v1", MutationScore::new(10, 5, 0));
        trend.add_snapshot("v2", MutationScore::new(10, 8, 0));
        let chart = ScoreChartData::from_trend(&trend);
        assert_eq!(chart.labels.len(), 2);
    }

    #[test]
    fn test_threshold_pass() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (0, 1), (1, 2), (1, 3)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        let thresh = AdequacyThreshold {
            minimum_overall: 0.80,
            ..Default::default()
        };
        let result = thresh.check(&bd);
        assert!(result.passed);
    }

    #[test]
    fn test_threshold_fail() {
        let km = make_test_kill_matrix(2, 10, &[(0, 0)]);
        let descs = descriptors_for(&km);
        let bd = ScoreBreakdown::compute(&km, &descs, &BTreeSet::new());
        let thresh = AdequacyThreshold {
            minimum_overall: 0.80,
            ..Default::default()
        };
        let result = thresh.check(&bd);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_raw_score() {
        let s = MutationScore::new(100, 80, 10);
        assert!((s.raw_score() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_score_percent_format() {
        let s = MutationScore::new(3, 2, 0);
        let pct = s.score_percent();
        assert!(pct.contains('%'));
    }
}
