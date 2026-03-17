//! Clinical significance filter pipeline.
//!
//! Orchestrates the full scoring and filtering workflow: DrugBank → Beers →
//! FAERS → Comorbidity → Composite → Rank → Filter → Report. Provides
//! deduplication and configurable thresholds.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::composite::{
    classify_severity, ClinicalSeverity, SignificanceScorer,
    ScoredConflict, SignificanceReport, generate_significance_report,
    rank_conflicts,
};
use crate::{ClinicalConfig, ConfirmedConflict, PatientProfile};

// ─────────────────────────── FilterConfig ────────────────────────────────

/// Configuration for the significance filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Minimum severity to include in output.
    pub min_severity: ClinicalSeverity,
    /// Whether to include Beers Criteria evaluation.
    pub include_beers: bool,
    /// Whether to include FAERS signal evaluation.
    pub include_faers: bool,
    /// Whether to include comorbidity prevalence scoring.
    pub include_comorbidity: bool,
    /// Custom weights (overrides defaults if Some).
    pub custom_weights: Option<CustomWeights>,
    /// Deduplicate overlapping findings.
    pub deduplicate: bool,
    /// Maximum number of conflicts to return (0 = unlimited).
    pub max_results: usize,
}

/// Custom weight overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomWeights {
    pub drugbank: f64,
    pub beers: f64,
    pub faers: f64,
    pub comorbidity: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        FilterConfig {
            min_severity: ClinicalSeverity::Minor,
            include_beers: true,
            include_faers: true,
            include_comorbidity: true,
            custom_weights: None,
            deduplicate: true,
            max_results: 0,
        }
    }
}

impl FilterConfig {
    pub fn new(min_severity: ClinicalSeverity) -> Self {
        FilterConfig {
            min_severity,
            ..Default::default()
        }
    }

    pub fn with_weights(mut self, drugbank: f64, beers: f64, faers: f64, comorbidity: f64) -> Self {
        self.custom_weights = Some(CustomWeights { drugbank, beers, faers, comorbidity });
        self
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    pub fn drugbank_only() -> Self {
        FilterConfig {
            include_beers: false,
            include_faers: false,
            include_comorbidity: false,
            custom_weights: Some(CustomWeights {
                drugbank: 1.0,
                beers: 0.0,
                faers: 0.0,
                comorbidity: 0.0,
            }),
            ..Default::default()
        }
    }

    pub fn critical_only() -> Self {
        FilterConfig {
            min_severity: ClinicalSeverity::Critical,
            ..Default::default()
        }
    }
}

// ─────────────────────────── FilterResult ────────────────────────────────

/// The output of the significance filter pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    /// Conflicts classified as Critical.
    pub critical: Vec<ScoredConflict>,
    /// Conflicts classified as Significant.
    pub significant: Vec<ScoredConflict>,
    /// Conflicts classified as Moderate.
    pub moderate: Vec<ScoredConflict>,
    /// Conflicts classified as Minor.
    pub minor: Vec<ScoredConflict>,
    /// Conflicts that were below the minimum severity threshold.
    pub filtered_out: Vec<ScoredConflict>,
    /// Summary statistics.
    pub statistics: FilterStatistics,
    /// Full significance report.
    pub report: SignificanceReport,
}

impl FilterResult {
    /// All conflicts that passed the filter (above min severity).
    pub fn all_passing(&self) -> Vec<&ScoredConflict> {
        self.critical.iter()
            .chain(self.significant.iter())
            .chain(self.moderate.iter())
            .chain(self.minor.iter())
            .collect()
    }

    /// Total number of conflicts that passed.
    pub fn passing_count(&self) -> usize {
        self.critical.len() + self.significant.len() + self.moderate.len() + self.minor.len()
    }

    /// Whether any critical conflicts were found.
    pub fn has_critical(&self) -> bool {
        !self.critical.is_empty()
    }

    /// Whether any significant-or-above conflicts were found.
    pub fn has_significant_or_above(&self) -> bool {
        !self.critical.is_empty() || !self.significant.is_empty()
    }

    /// Get the highest-ranked conflict.
    pub fn top_conflict(&self) -> Option<&ScoredConflict> {
        self.critical.first()
            .or(self.significant.first())
            .or(self.moderate.first())
            .or(self.minor.first())
    }
}

impl fmt::Display for FilterResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FilterResult: {} passing ({} critical, {} significant, {} moderate, {} minor), {} filtered out",
            self.passing_count(),
            self.critical.len(),
            self.significant.len(),
            self.moderate.len(),
            self.minor.len(),
            self.filtered_out.len(),
        )
    }
}

// ─────────────────────────── FilterStatistics ────────────────────────────

/// Statistics about the filtering process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStatistics {
    pub total_input: usize,
    pub total_output: usize,
    pub filtered_count: usize,
    pub duplicates_removed: usize,
    pub by_severity: SeverityBreakdown,
    pub by_source: SourceBreakdown,
}

/// Breakdown by severity classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityBreakdown {
    pub critical: usize,
    pub significant: usize,
    pub moderate: usize,
    pub minor: usize,
    pub negligible: usize,
}

/// Breakdown by which evidence source contributed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceBreakdown {
    pub drugbank_hits: usize,
    pub beers_hits: usize,
    pub faers_hits: usize,
    pub comorbidity_weighted: usize,
}

impl fmt::Display for FilterStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input:{} → Output:{} (filtered:{}, deduped:{})",
            self.total_input, self.total_output,
            self.filtered_count, self.duplicates_removed,
        )
    }
}

// ─────────────────────────── SignificanceFilter ──────────────────────────

/// The main significance filter pipeline.
#[derive(Debug, Clone)]
pub struct SignificanceFilter {
    config: FilterConfig,
    scorer: SignificanceScorer,
}

impl SignificanceFilter {
    /// Create a filter from a ClinicalConfig.
    pub fn new(config: &ClinicalConfig) -> Self {
        let scorer = SignificanceScorer::from_config(config);
        SignificanceFilter {
            config: FilterConfig {
                min_severity: classify_severity(config.min_severity_threshold),
                include_beers: config.include_beers,
                include_faers: config.include_faers,
                include_comorbidity: config.include_comorbidity,
                ..Default::default()
            },
            scorer,
        }
    }

    /// Create with explicit FilterConfig and scorer.
    pub fn with_config(config: FilterConfig, scorer: SignificanceScorer) -> Self {
        SignificanceFilter { config, scorer }
    }

    /// Create with all defaults.
    pub fn with_defaults() -> Self {
        Self::new(&ClinicalConfig::default())
    }

    /// Access the scorer.
    pub fn scorer(&self) -> &SignificanceScorer {
        &self.scorer
    }

    /// Access the config.
    pub fn config(&self) -> &FilterConfig {
        &self.config
    }

    /// Run the full filter pipeline.
    pub fn filter(
        &self,
        conflicts: &[ConfirmedConflict],
        patient: &PatientProfile,
    ) -> FilterResult {
        // 1. Score all conflicts
        let mut scored = self.scorer.score_all_conflicts(conflicts, patient);

        // 2. Adjust scores based on config (zero out disabled components)
        if !self.config.include_beers {
            for sc in &mut scored {
                sc.score.beers_component = 0.0;
            }
        }
        if !self.config.include_faers {
            for sc in &mut scored {
                sc.score.faers_component = 0.0;
            }
        }
        if !self.config.include_comorbidity {
            for sc in &mut scored {
                sc.score.comorbidity_component = 0.0;
            }
        }

        // If components were zeroed out, recalculate total scores
        if !self.config.include_beers || !self.config.include_faers || !self.config.include_comorbidity {
            for sc in &mut scored {
                let w = self.scorer.drugbank_weight
                    + if self.config.include_beers { self.scorer.beers_weight } else { 0.0 }
                    + if self.config.include_faers { self.scorer.faers_weight } else { 0.0 }
                    + if self.config.include_comorbidity { self.scorer.comorbidity_weight } else { 0.0 };
                let w = if w == 0.0 { 1.0 } else { w };

                let total = (self.scorer.drugbank_weight * sc.score.drugbank_component
                    + if self.config.include_beers { self.scorer.beers_weight * sc.score.beers_component } else { 0.0 }
                    + if self.config.include_faers { self.scorer.faers_weight * sc.score.faers_component } else { 0.0 }
                    + if self.config.include_comorbidity { self.scorer.comorbidity_weight * sc.score.comorbidity_component } else { 0.0 })
                    / w;

                sc.score.total_score = total.min(1.0).max(0.0);
                sc.score.severity_classification = classify_severity(sc.score.total_score);
            }
        }

        // 3. Re-rank after adjustments
        rank_conflicts(&mut scored);

        // 4. Deduplicate
        let duplicates_removed;
        if self.config.deduplicate {
            let before = scored.len();
            scored = deduplicate_conflicts(scored);
            duplicates_removed = before - scored.len();
            rank_conflicts(&mut scored);
        } else {
            duplicates_removed = 0;
        }

        // 5. Collect source statistics
        let by_source = compute_source_breakdown(&scored);

        // 6. Partition by severity
        let total_before_filter = scored.len();
        let min_sev = self.config.min_severity;

        let mut critical = Vec::new();
        let mut significant = Vec::new();
        let mut moderate = Vec::new();
        let mut minor = Vec::new();
        let mut filtered_out = Vec::new();

        for sc in scored {
            let sev = sc.severity();
            if sev.at_least(min_sev) {
                match sev {
                    ClinicalSeverity::Critical => critical.push(sc),
                    ClinicalSeverity::Significant => significant.push(sc),
                    ClinicalSeverity::Moderate => moderate.push(sc),
                    ClinicalSeverity::Minor => minor.push(sc),
                    ClinicalSeverity::Negligible => {
                        if min_sev == ClinicalSeverity::Negligible {
                            minor.push(sc); // group with minor
                        } else {
                            filtered_out.push(sc);
                        }
                    }
                }
            } else {
                filtered_out.push(sc);
            }
        }

        // 7. Apply max_results limit
        if self.config.max_results > 0 {
            let mut remaining = self.config.max_results;
            critical.truncate(remaining);
            remaining = remaining.saturating_sub(critical.len());
            significant.truncate(remaining);
            remaining = remaining.saturating_sub(significant.len());
            moderate.truncate(remaining);
            remaining = remaining.saturating_sub(moderate.len());
            minor.truncate(remaining);
        }

        let total_output = critical.len() + significant.len() + moderate.len() + minor.len();
        let filtered_count = total_before_filter - total_output;

        let by_severity = SeverityBreakdown {
            critical: critical.len(),
            significant: significant.len(),
            moderate: moderate.len(),
            minor: minor.len(),
            negligible: filtered_out.iter().filter(|s| s.severity() == ClinicalSeverity::Negligible).count(),
        };

        let statistics = FilterStatistics {
            total_input: conflicts.len(),
            total_output,
            filtered_count,
            duplicates_removed,
            by_severity,
            by_source,
        };

        // 8. Generate report from all passing conflicts
        let all_passing: Vec<ScoredConflict> = critical.iter()
            .chain(significant.iter())
            .chain(moderate.iter())
            .chain(minor.iter())
            .cloned()
            .collect();
        let report = generate_significance_report(&all_passing);

        FilterResult {
            critical,
            significant,
            moderate,
            minor,
            filtered_out,
            statistics,
            report,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────

/// Remove duplicate drug-pair conflicts, keeping the highest-scored one.
fn deduplicate_conflicts(scored: Vec<ScoredConflict>) -> Vec<ScoredConflict> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut result = Vec::with_capacity(scored.len());

    // scored is already sorted by descending score, so the first occurrence is the best
    for sc in scored {
        let key = sc.conflict.pair_key();
        if seen.insert(key) {
            result.push(sc);
        }
    }

    result
}

/// Compute how many conflicts had hits from each evidence source.
fn compute_source_breakdown(scored: &[ScoredConflict]) -> SourceBreakdown {
    let mut drugbank_hits = 0;
    let mut beers_hits = 0;
    let mut faers_hits = 0;
    let mut comorbidity_weighted = 0;

    for sc in scored {
        if sc.score.drugbank_component > 0.0 {
            drugbank_hits += 1;
        }
        if sc.score.beers_component > 0.0 {
            beers_hits += 1;
        }
        if sc.score.faers_component > 0.0 {
            faers_hits += 1;
        }
        if sc.score.comorbidity_component > 0.0 {
            comorbidity_weighted += 1;
        }
    }

    SourceBreakdown {
        drugbank_hits,
        beers_hits,
        faers_hits,
        comorbidity_weighted,
    }
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sex;
    use crate::Condition;
    use crate::Medication;
    use crate::Severity;

    fn elderly_patient() -> PatientProfile {
        PatientProfile::new(
            78.0, 70.0, Sex::Male,
        )
        .with_conditions(vec![
            Condition::new("I10", "Hypertension"),
            Condition::new("E11.9", "Type 2 Diabetes"),
            Condition::new("N18.3", "CKD stage 3"),
        ])
        .with_medications(vec![
            Medication::new("Warfarin", "anticoagulant", 5.0),
            Medication::new("Aspirin", "antiplatelet", 81.0),
            Medication::new("Lisinopril", "ACE inhibitor", 10.0),
            Medication::new("Simvastatin", "statin", 20.0),
            Medication::new("Metformin", "biguanide", 500.0),
        ])
        .with_egfr(45.0)
    }

    fn sample_conflicts() -> Vec<ConfirmedConflict> {
        vec![
            ConfirmedConflict::new(
                "Warfarin", "Aspirin", Severity::Major,
                "Additive anticoagulation", "Increased bleeding risk",
            ),
            ConfirmedConflict::new(
                "Simvastatin", "Amiodarone", Severity::Major,
                "CYP3A4 inhibition", "Rhabdomyolysis risk",
            ),
            ConfirmedConflict::new(
                "Metformin", "Lisinopril", Severity::Minor,
                "Minor PD interaction", "Modest effect",
            ),
        ]
    }

    #[test]
    fn test_filter_basic() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        assert_eq!(result.statistics.total_input, 3);
        assert!(result.passing_count() > 0);
    }

    #[test]
    fn test_filter_has_critical_or_significant() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        assert!(
            result.has_significant_or_above(),
            "Warfarin+aspirin should produce significant/critical"
        );
    }

    #[test]
    fn test_filter_empty_input() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&[], &patient);

        assert_eq!(result.passing_count(), 0);
        assert_eq!(result.statistics.total_input, 0);
    }

    #[test]
    fn test_filter_critical_only() {
        let config = ClinicalConfig {
            min_severity_threshold: 0.8,
            ..ClinicalConfig::default()
        };
        let f = SignificanceFilter::new(&config);
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        for sc in result.all_passing() {
            assert!(
                sc.severity().at_least(ClinicalSeverity::Critical),
                "Only critical should pass, got {:?}", sc.severity()
            );
        }
    }

    #[test]
    fn test_filter_deduplication() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();

        // Create duplicate pair
        let conflicts = vec![
            ConfirmedConflict::new(
                "Warfarin", "Aspirin", Severity::Major,
                "Mechanism A", "Description A",
            ),
            ConfirmedConflict::new(
                "Warfarin", "Aspirin", Severity::Major,
                "Mechanism B", "Description B",
            ),
        ];

        let result = f.filter(&conflicts, &patient);
        assert_eq!(result.statistics.duplicates_removed, 1);
        assert_eq!(result.passing_count(), 1);
    }

    #[test]
    fn test_filter_no_dedup() {
        let config = FilterConfig {
            deduplicate: false,
            ..Default::default()
        };
        let scorer = SignificanceScorer::with_defaults();
        let f = SignificanceFilter::with_config(config, scorer);
        let patient = elderly_patient();

        let conflicts = vec![
            ConfirmedConflict::new("Warfarin", "Aspirin", Severity::Major, "A", "B"),
            ConfirmedConflict::new("Warfarin", "Aspirin", Severity::Major, "C", "D"),
        ];

        let result = f.filter(&conflicts, &patient);
        assert_eq!(result.statistics.duplicates_removed, 0);
    }

    #[test]
    fn test_filter_max_results() {
        let config = FilterConfig {
            max_results: 1,
            ..Default::default()
        };
        let scorer = SignificanceScorer::with_defaults();
        let f = SignificanceFilter::with_config(config, scorer);
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        assert!(result.passing_count() <= 1, "Max results should be 1");
    }

    #[test]
    fn test_filter_statistics() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        let stats = &result.statistics;
        assert_eq!(stats.total_input, 3);
        assert_eq!(stats.total_output + stats.filtered_count + stats.duplicates_removed, stats.total_input);
        assert!(stats.by_source.drugbank_hits > 0);
    }

    #[test]
    fn test_filter_drugbank_only() {
        let config = FilterConfig::drugbank_only();
        let scorer = SignificanceScorer::new(1.0, 0.0, 0.0, 0.0);
        let f = SignificanceFilter::with_config(config, scorer);
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        for sc in result.all_passing() {
            assert_eq!(sc.score.beers_component, 0.0);
            assert_eq!(sc.score.faers_component, 0.0);
            assert_eq!(sc.score.comorbidity_component, 0.0);
        }
    }

    #[test]
    fn test_filter_result_display() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        let s = format!("{}", result);
        assert!(s.contains("passing"));
        assert!(s.contains("filtered out"));
    }

    #[test]
    fn test_filter_report_generated() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        assert_eq!(result.report.total_conflicts, result.passing_count());
    }

    #[test]
    fn test_filter_top_conflict() {
        let f = SignificanceFilter::with_defaults();
        let patient = elderly_patient();
        let result = f.filter(&sample_conflicts(), &patient);

        let top = result.top_conflict();
        assert!(top.is_some());
    }

    #[test]
    fn test_filter_config_defaults() {
        let config = FilterConfig::default();
        assert_eq!(config.min_severity, ClinicalSeverity::Minor);
        assert!(config.include_beers);
        assert!(config.include_faers);
        assert!(config.include_comorbidity);
        assert!(config.deduplicate);
        assert_eq!(config.max_results, 0);
    }

    #[test]
    fn test_deduplicate_preserves_highest() {
        let scored = vec![
            ScoredConflict {
                conflict: ConfirmedConflict::new("A", "B", Severity::Minor, "low", "low"),
                score: crate::composite::SignificanceScore {
                    total_score: 0.3,
                    severity_classification: ClinicalSeverity::Minor,
                    ..crate::composite::SignificanceScore::zero()
                },
                rank: 2,
            },
            ScoredConflict {
                conflict: ConfirmedConflict::new("B", "A", Severity::Major, "high", "high"),
                score: crate::composite::SignificanceScore {
                    total_score: 0.9,
                    severity_classification: ClinicalSeverity::Critical,
                    ..crate::composite::SignificanceScore::zero()
                },
                rank: 1,
            },
        ];

        // Sort by descending score first (as the pipeline does)
        let mut sorted = scored;
        sorted.sort_by(|a, b| b.score.total_score.partial_cmp(&a.score.total_score).unwrap());

        let deduped = deduplicate_conflicts(sorted);
        assert_eq!(deduped.len(), 1);
        assert!((deduped[0].score.total_score - 0.9).abs() < 1e-10, "Should keep highest score");
    }

    #[test]
    fn test_statistics_display() {
        let stats = FilterStatistics {
            total_input: 10,
            total_output: 7,
            filtered_count: 2,
            duplicates_removed: 1,
            by_severity: SeverityBreakdown {
                critical: 1,
                significant: 2,
                moderate: 3,
                minor: 1,
                negligible: 0,
            },
            by_source: SourceBreakdown {
                drugbank_hits: 5,
                beers_hits: 3,
                faers_hits: 4,
                comorbidity_weighted: 6,
            },
        };
        let s = format!("{}", stats);
        assert!(s.contains("10"));
        assert!(s.contains("7"));
    }

    #[test]
    fn test_filter_severity_partition() {
        let f = SignificanceFilter::new(&ClinicalConfig {
            min_severity_threshold: 0.0,
            ..ClinicalConfig::default()
        });
        let patient = elderly_patient();
        let conflicts = sample_conflicts();
        let result = f.filter(&conflicts, &patient);

        // All should be in one of the buckets
        let total = result.critical.len()
            + result.significant.len()
            + result.moderate.len()
            + result.minor.len()
            + result.filtered_out.len();
        assert!(total > 0);
    }
}
