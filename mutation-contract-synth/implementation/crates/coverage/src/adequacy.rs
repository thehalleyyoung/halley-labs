//! Test suite adequacy analysis.
//!
//! Determines whether a test suite T is mutation-adequate: does T detect all
//! detectable (non-equivalent) mutants? Identifies adequacy gaps and suggests
//! improvements.

use crate::scoring::MutationScore;
use crate::{
    CoverageError, KillMatrix, MutantDescriptor, MutantId, MutationOperator, Result, TestId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// AdequacyCertificate
// ---------------------------------------------------------------------------

/// Certificate attesting that a test suite meets mutation adequacy criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdequacyCertificate {
    /// Whether the test suite is mutation-adequate.
    pub is_adequate: bool,
    /// Overall mutation score.
    pub score: f64,
    /// Number of mutants detected.
    pub killed: usize,
    /// Number of non-equivalent surviving mutants.
    pub surviving_non_equivalent: usize,
    /// Number of equivalent mutants excluded.
    pub equivalent_excluded: usize,
    /// Per-operator adequacy.
    pub per_operator: BTreeMap<String, OperatorAdequacy>,
    /// Identified gaps.
    pub gaps: Vec<AdequacyGap>,
    /// Timestamp of certification.
    pub timestamp: String,
    /// Test suite identifier.
    pub test_suite_id: String,
}

impl AdequacyCertificate {
    /// Summary line for display.
    pub fn summary(&self) -> String {
        if self.is_adequate {
            format!(
                "ADEQUATE: score={:.1}%, all killable mutants detected",
                self.score * 100.0
            )
        } else {
            format!(
                "INADEQUATE: score={:.1}%, {} gaps found",
                self.score * 100.0,
                self.gaps.len()
            )
        }
    }

    /// Operators that are individually adequate.
    pub fn adequate_operators(&self) -> Vec<&str> {
        self.per_operator
            .iter()
            .filter(|(_, a)| a.is_adequate)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Operators that are individually inadequate.
    pub fn inadequate_operators(&self) -> Vec<&str> {
        self.per_operator
            .iter()
            .filter(|(_, a)| !a.is_adequate)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

/// Per-operator adequacy status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorAdequacy {
    pub operator: String,
    pub is_adequate: bool,
    pub score: f64,
    pub killed: usize,
    pub total_killable: usize,
    pub surviving: Vec<MutantId>,
}

// ---------------------------------------------------------------------------
// AdequacyGap
// ---------------------------------------------------------------------------

/// A gap in test suite adequacy: a killable mutant that survives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdequacyGap {
    /// The surviving mutant.
    pub mutant_id: MutantId,
    /// Operator that produced this mutant.
    pub operator: Option<String>,
    /// Function where the mutation is located.
    pub function: Option<String>,
    /// Source location.
    pub location: Option<String>,
    /// Description of the mutation.
    pub description: Option<String>,
    /// Suggested test strategy to cover this gap.
    pub suggestion: String,
    /// Priority (higher = more important to fix).
    pub priority: GapPriority,
}

/// Priority of an adequacy gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GapPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for GapPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Improvement suggestion
// ---------------------------------------------------------------------------

/// A suggestion for improving test suite adequacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementSuggestion {
    /// Which gaps this suggestion addresses.
    pub addresses: Vec<MutantId>,
    /// Human-readable description of the test to add.
    pub test_description: String,
    /// Estimated number of new mutant detections.
    pub estimated_new_kills: usize,
    /// Impact on the overall score.
    pub estimated_score_delta: f64,
    /// Priority of this suggestion.
    pub priority: GapPriority,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for adequacy analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdequacyConfig {
    /// Minimum overall score for adequacy.
    pub min_score: f64,
    /// Whether to require per-operator adequacy.
    pub require_per_operator: bool,
    /// Minimum per-operator score (if required).
    pub min_operator_score: f64,
    /// Whether to generate improvement suggestions.
    pub generate_suggestions: bool,
    /// Maximum number of suggestions to generate.
    pub max_suggestions: usize,
}

impl Default for AdequacyConfig {
    fn default() -> Self {
        Self {
            min_score: 1.0,
            require_per_operator: false,
            min_operator_score: 0.80,
            generate_suggestions: true,
            max_suggestions: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// AdequacyAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes test suite adequacy with respect to mutation testing.
pub struct AdequacyAnalyzer {
    config: AdequacyConfig,
    descriptors: HashMap<MutantId, MutantDescriptor>,
    equivalent_mutants: BTreeSet<MutantId>,
}

impl AdequacyAnalyzer {
    pub fn new() -> Self {
        Self {
            config: AdequacyConfig::default(),
            descriptors: HashMap::new(),
            equivalent_mutants: BTreeSet::new(),
        }
    }

    pub fn with_config(config: AdequacyConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    pub fn register_descriptors(&mut self, descs: Vec<MutantDescriptor>) {
        for d in descs {
            self.descriptors.insert(d.id.clone(), d);
        }
    }

    pub fn set_equivalent_mutants(&mut self, eq: BTreeSet<MutantId>) {
        self.equivalent_mutants = eq;
    }

    // -- Core analysis -----------------------------------------------------

    /// Run the full adequacy analysis.
    pub fn analyze(
        &self,
        km: &KillMatrix,
        test_suite_id: impl Into<String>,
    ) -> Result<AdequacyCertificate> {
        let equivalent_indices: BTreeSet<usize> = km
            .mutants
            .iter()
            .enumerate()
            .filter(|(_, id)| self.equivalent_mutants.contains(id))
            .map(|(i, _)| i)
            .collect();

        let score = MutationScore::from_kill_matrix(km, &equivalent_indices);
        let per_operator = self.check_per_operator(km, &equivalent_indices);
        let gaps = self.identify_gaps(km, &equivalent_indices);

        let is_adequate = score.score() >= self.config.min_score
            && (!self.config.require_per_operator || per_operator.values().all(|a| a.is_adequate));

        Ok(AdequacyCertificate {
            is_adequate,
            score: score.score(),
            killed: score.killed,
            surviving_non_equivalent: score.survived,
            equivalent_excluded: score.equivalent,
            per_operator,
            gaps,
            timestamp: chrono::Utc::now().to_rfc3339(),
            test_suite_id: test_suite_id.into(),
        })
    }

    // -- Per-operator adequacy ---------------------------------------------

    fn check_per_operator(
        &self,
        km: &KillMatrix,
        equivalent: &BTreeSet<usize>,
    ) -> BTreeMap<String, OperatorAdequacy> {
        let mut by_op: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, mid) in km.mutants.iter().enumerate() {
            if let Some(desc) = self.descriptors.get(mid) {
                by_op
                    .entry(desc.operator.short_name().to_string())
                    .or_default()
                    .push(idx);
            }
        }

        let killed_set = km.killed_set();
        by_op
            .into_iter()
            .map(|(op, indices)| {
                let total = indices.len();
                let eq_count = indices.iter().filter(|i| equivalent.contains(i)).count();
                let killable = total - eq_count;
                let killed_count = indices.iter().filter(|i| killed_set.contains(i)).count();
                let sc = if killable == 0 {
                    1.0
                } else {
                    killed_count as f64 / killable as f64
                };
                let is_adequate = sc >= self.config.min_operator_score;

                let surviving: Vec<MutantId> = indices
                    .iter()
                    .filter(|&&i| !killed_set.contains(&i) && !equivalent.contains(&i))
                    .map(|&i| km.mutants[i].clone())
                    .collect();

                (
                    op.clone(),
                    OperatorAdequacy {
                        operator: op,
                        is_adequate,
                        score: sc,
                        killed: killed_count,
                        total_killable: killable,
                        surviving,
                    },
                )
            })
            .collect()
    }

    // -- Gap identification ------------------------------------------------

    /// Identify all adequacy gaps (surviving non-equivalent mutants).
    pub fn identify_gaps(&self, km: &KillMatrix, equivalent: &BTreeSet<usize>) -> Vec<AdequacyGap> {
        let killed_set = km.killed_set();
        let mut gaps = Vec::new();

        for (idx, mid) in km.mutants.iter().enumerate() {
            if killed_set.contains(&idx) || equivalent.contains(&idx) {
                continue;
            }

            let desc = self.descriptors.get(mid);
            let operator = desc.map(|d| d.operator.short_name().to_string());
            let function = desc.map(|d| d.site.function_name.clone());
            let location = desc.map(|d| format!("{}", d.site));
            let description = desc.map(|d| d.description.clone());

            let priority = self.assess_gap_priority(mid, km);
            let suggestion = self.generate_gap_suggestion(mid, desc);

            gaps.push(AdequacyGap {
                mutant_id: mid.clone(),
                operator,
                function,
                location,
                description,
                suggestion,
                priority,
            });
        }

        gaps.sort_by(|a, b| b.priority.cmp(&a.priority));
        gaps
    }

    fn assess_gap_priority(&self, id: &MutantId, km: &KillMatrix) -> GapPriority {
        let desc = match self.descriptors.get(id) {
            Some(d) => d,
            None => return GapPriority::Medium,
        };

        // Critical operators get higher priority.
        match desc.operator {
            MutationOperator::ROR | MutationOperator::COR => GapPriority::High,
            MutationOperator::SDL | MutationOperator::BOMB => GapPriority::Critical,
            MutationOperator::AOR | MutationOperator::UOI => GapPriority::Medium,
            _ => GapPriority::Low,
        }
    }

    fn generate_gap_suggestion(&self, id: &MutantId, desc: Option<&MutantDescriptor>) -> String {
        match desc {
            Some(d) => match d.operator {
                MutationOperator::AOR => format!(
                    "Add a test that exercises the arithmetic at {} with boundary values",
                    d.site
                ),
                MutationOperator::ROR => format!(
                    "Add a test at the boundary of the relational condition at {}",
                    d.site
                ),
                MutationOperator::COR => {
                    format!("Add tests for each branch of the conditional at {}", d.site)
                }
                MutationOperator::SDL => format!(
                    "Add a test that depends on the statement at {} being executed",
                    d.site
                ),
                MutationOperator::CR => {
                    format!("Add a test sensitive to the constant value at {}", d.site)
                }
                MutationOperator::VR => {
                    format!("Add a test with distinct variable values at {}", d.site)
                }
                _ => format!(
                    "Add a test targeting the mutation {} at {}",
                    d.operator, d.site
                ),
            },
            None => format!("Add a test targeting mutant {}", id),
        }
    }

    // -- Improvement suggestions -------------------------------------------

    /// Generate suggestions for improving test suite adequacy.
    pub fn suggest_improvements(
        &self,
        km: &KillMatrix,
        equivalent: &BTreeSet<usize>,
    ) -> Vec<ImprovementSuggestion> {
        let gaps = self.identify_gaps(km, equivalent);
        if gaps.is_empty() {
            return Vec::new();
        }

        let total_score = MutationScore::from_kill_matrix(km, equivalent);

        // Group gaps by function for multi-mutant suggestions.
        let mut by_function: BTreeMap<String, Vec<&AdequacyGap>> = BTreeMap::new();
        for gap in &gaps {
            let func = gap.function.clone().unwrap_or_else(|| "unknown".into());
            by_function.entry(func).or_default().push(gap);
        }

        let mut suggestions = Vec::new();
        let killable = total_score.killable().max(1);

        for (func, func_gaps) in &by_function {
            let addresses: Vec<MutantId> = func_gaps.iter().map(|g| g.mutant_id.clone()).collect();
            let est = addresses.len();
            let delta = est as f64 / killable as f64;
            let priority = func_gaps
                .iter()
                .map(|g| g.priority)
                .max()
                .unwrap_or(GapPriority::Low);

            suggestions.push(ImprovementSuggestion {
                addresses,
                test_description: format!("Add boundary-value tests for function '{}'", func),
                estimated_new_kills: est,
                estimated_score_delta: delta,
                priority,
            });
        }

        suggestions.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.estimated_new_kills.cmp(&a.estimated_new_kills))
        });
        suggestions.truncate(self.config.max_suggestions);
        suggestions
    }

    // -- Minimal adequate subset -------------------------------------------

    /// Compute a minimal subset of tests that achieves the same mutation score.
    pub fn minimal_adequate_subset(&self, km: &KillMatrix) -> Vec<TestId> {
        let killed_set = km.killed_set();
        if killed_set.is_empty() {
            return Vec::new();
        }

        let mut uncovered: BTreeSet<usize> = killed_set.clone();
        let mut selected = Vec::new();
        let mut selected_set = HashSet::new();

        while !uncovered.is_empty() {
            let best = (0..km.num_tests())
                .filter(|t| !selected_set.contains(t))
                .max_by_key(|&t| km.killed_mutants(t).intersection(&uncovered).count());
            match best {
                Some(t) => {
                    selected.push(t);
                    selected_set.insert(t);
                    for m in km.killed_mutants(t) {
                        uncovered.remove(&m);
                    }
                }
                None => break,
            }
        }

        selected.iter().map(|&t| km.tests[t].clone()).collect()
    }

    /// Compute the redundancy ratio: 1 - (minimal / total).
    pub fn test_redundancy(&self, km: &KillMatrix) -> f64 {
        let minimal = self.minimal_adequate_subset(km);
        if km.num_tests() == 0 {
            0.0
        } else {
            1.0 - (minimal.len() as f64 / km.num_tests() as f64)
        }
    }
}

impl Default for AdequacyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_test_kill_matrix, make_test_mutant, MutantId, MutationOperator};

    fn setup_analyzer(km: &KillMatrix) -> AdequacyAnalyzer {
        let mut analyzer = AdequacyAnalyzer::new();
        let descs: Vec<MutantDescriptor> = km
            .mutants
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let op = match i % 4 {
                    0 => MutationOperator::AOR,
                    1 => MutationOperator::ROR,
                    2 => MutationOperator::COR,
                    _ => MutationOperator::SDL,
                };
                make_test_mutant(id.as_str(), op)
            })
            .collect();
        analyzer.register_descriptors(descs);
        analyzer
    }

    #[test]
    fn test_adequate() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let analyzer = setup_analyzer(&km);
        let cert = analyzer.analyze(&km, "test_suite_1").unwrap();
        assert!(cert.is_adequate);
        assert!((cert.score - 1.0).abs() < 1e-9);
        assert!(cert.gaps.is_empty());
    }

    #[test]
    fn test_inadequate() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (1, 1)]);
        let analyzer = setup_analyzer(&km);
        let cert = analyzer.analyze(&km, "test_suite_2").unwrap();
        assert!(!cert.is_adequate);
        assert_eq!(cert.gaps.len(), 2);
    }

    #[test]
    fn test_with_equivalents() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (1, 1)]);
        let mut analyzer = setup_analyzer(&km);
        analyzer.set_equivalent_mutants(BTreeSet::from([MutantId::new("m2"), MutantId::new("m3")]));
        let cert = analyzer.analyze(&km, "ts").unwrap();
        assert!(cert.is_adequate);
        assert_eq!(cert.equivalent_excluded, 2);
    }

    #[test]
    fn test_per_operator_adequacy() {
        let km = make_test_kill_matrix(3, 4, &[(0, 0), (1, 1), (2, 2)]);
        let analyzer = setup_analyzer(&km);
        let cert = analyzer.analyze(&km, "ts").unwrap();
        assert!(!cert.per_operator.is_empty());
    }

    #[test]
    fn test_gap_priority() {
        let km = make_test_kill_matrix(1, 4, &[(0, 0)]);
        let analyzer = setup_analyzer(&km);
        let gaps = analyzer.identify_gaps(&km, &BTreeSet::new());
        assert_eq!(gaps.len(), 3);
        // Gaps should be sorted by priority (highest first).
        assert!(gaps[0].priority >= gaps[gaps.len() - 1].priority);
    }

    #[test]
    fn test_suggestions() {
        let km = make_test_kill_matrix(2, 6, &[(0, 0), (1, 1)]);
        let analyzer = setup_analyzer(&km);
        let suggestions = analyzer.suggest_improvements(&km, &BTreeSet::new());
        assert!(!suggestions.is_empty());
        for s in &suggestions {
            assert!(!s.addresses.is_empty());
            assert!(s.estimated_new_kills > 0);
        }
    }

    #[test]
    fn test_no_suggestions_when_adequate() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let analyzer = setup_analyzer(&km);
        let suggestions = analyzer.suggest_improvements(&km, &BTreeSet::new());
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_minimal_adequate_subset() {
        // t0 and t1 both detect m0; t1 also detects m1.
        let km = make_test_kill_matrix(3, 2, &[(0, 0), (1, 0), (1, 1), (2, 1)]);
        let analyzer = AdequacyAnalyzer::new();
        let minimal = analyzer.minimal_adequate_subset(&km);
        // t1 alone covers both m0 and m1.
        assert!(minimal.len() <= 2);
    }

    #[test]
    fn test_minimal_adequate_empty() {
        let km = make_test_kill_matrix(3, 3, &[]);
        let analyzer = AdequacyAnalyzer::new();
        assert!(analyzer.minimal_adequate_subset(&km).is_empty());
    }

    #[test]
    fn test_test_redundancy() {
        let km = make_test_kill_matrix(4, 2, &[(0, 0), (1, 0), (2, 1), (3, 1)]);
        let analyzer = AdequacyAnalyzer::new();
        let redundancy = analyzer.test_redundancy(&km);
        assert!(redundancy >= 0.0 && redundancy <= 1.0);
        assert!(redundancy > 0.0); // Some tests are redundant.
    }

    #[test]
    fn test_certificate_summary() {
        let km = make_test_kill_matrix(2, 2, &[(0, 0), (1, 1)]);
        let analyzer = setup_analyzer(&km);
        let cert = analyzer.analyze(&km, "ts").unwrap();
        let summary = cert.summary();
        assert!(summary.contains("ADEQUATE"));
    }

    #[test]
    fn test_adequate_operators() {
        let km = make_test_kill_matrix(4, 4, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
        let analyzer = setup_analyzer(&km);
        let cert = analyzer.analyze(&km, "ts").unwrap();
        assert!(!cert.adequate_operators().is_empty());
        assert!(cert.inadequate_operators().is_empty());
    }

    #[test]
    fn test_config_custom() {
        let config = AdequacyConfig {
            min_score: 0.90,
            require_per_operator: true,
            min_operator_score: 0.75,
            ..Default::default()
        };
        let km = make_test_kill_matrix(3, 4, &[(0, 0), (1, 1), (2, 2)]);
        let mut analyzer = AdequacyAnalyzer::with_config(config);
        let descs: Vec<MutantDescriptor> = km
            .mutants
            .iter()
            .enumerate()
            .map(|(i, id)| make_test_mutant(id.as_str(), MutationOperator::AOR))
            .collect();
        analyzer.register_descriptors(descs);
        let cert = analyzer.analyze(&km, "ts").unwrap();
        // 3/4 = 75% < 90% threshold.
        assert!(!cert.is_adequate);
    }

    #[test]
    fn test_gap_suggestion_content() {
        let km = make_test_kill_matrix(1, 2, &[(0, 0)]);
        let analyzer = setup_analyzer(&km);
        let gaps = analyzer.identify_gaps(&km, &BTreeSet::new());
        assert_eq!(gaps.len(), 1);
        assert!(!gaps[0].suggestion.is_empty());
    }

    #[test]
    fn test_empty_matrix_adequacy() {
        let km = make_test_kill_matrix(0, 0, &[]);
        let analyzer = AdequacyAnalyzer::new();
        let cert = analyzer.analyze(&km, "empty").unwrap();
        assert!(cert.is_adequate);
    }

    #[test]
    fn test_suggestion_max_limit() {
        let config = AdequacyConfig {
            max_suggestions: 2,
            ..Default::default()
        };
        let km = make_test_kill_matrix(1, 10, &[(0, 0)]);
        let mut analyzer = AdequacyAnalyzer::with_config(config);
        let descs: Vec<MutantDescriptor> = km
            .mutants
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let func = format!("func_{}", i);
                crate::MutantDescriptor::new(
                    id.clone(),
                    MutationOperator::AOR,
                    crate::MutationSite::new("t.c", func, i + 1, 1),
                    "a",
                    "b",
                )
            })
            .collect();
        analyzer.register_descriptors(descs);
        let suggestions = analyzer.suggest_improvements(&km, &BTreeSet::new());
        assert!(suggestions.len() <= 2);
    }
}
