//! Confidence scoring for compatibility assessments.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Source of a confidence assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceSource {
    SchemaDiff,
    SemverAnalysis,
    UserOverride,
    HistoricalData,
    Heuristic,
    Combined,
}

impl fmt::Display for ConfidenceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaDiff => write!(f, "schema_diff"),
            Self::SemverAnalysis => write!(f, "semver_analysis"),
            Self::UserOverride => write!(f, "user_override"),
            Self::HistoricalData => write!(f, "historical_data"),
            Self::Heuristic => write!(f, "heuristic"),
            Self::Combined => write!(f, "combined"),
        }
    }
}

/// A confidence score for a compatibility assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Value between 0.0 and 1.0.
    pub value: f64,
    /// Source of this confidence score.
    pub source: ConfidenceSource,
    /// Human-readable rationale.
    pub rationale: String,
    /// Number of evidence items backing this score.
    pub evidence_count: usize,
}

impl ConfidenceScore {
    pub fn new(value: f64, source: ConfidenceSource, rationale: &str) -> Self {
        let value = value.clamp(0.0, 1.0);
        Self {
            value,
            source,
            rationale: rationale.to_string(),
            evidence_count: 1,
        }
    }

    /// High-confidence score from a schema diff analysis.
    pub fn from_schema_diff(breaking_changes: usize, total_changes: usize) -> Self {
        let value = if total_changes == 0 {
            0.95
        } else if breaking_changes == 0 {
            0.90
        } else {
            let ratio = breaking_changes as f64 / total_changes as f64;
            (0.85 - ratio * 0.3).max(0.5)
        };
        Self {
            value,
            source: ConfidenceSource::SchemaDiff,
            rationale: format!(
                "Schema diff: {} breaking out of {} total changes",
                breaking_changes, total_changes
            ),
            evidence_count: total_changes.max(1),
        }
    }

    /// Medium-confidence score from semver analysis.
    pub fn from_semver_analysis(major_changed: bool, minor_changed: bool, patch_changed: bool) -> Self {
        let (value, rationale) = if major_changed {
            (0.6, "Major version bump suggests breaking changes")
        } else if minor_changed {
            (0.75, "Minor version bump suggests backward-compatible additions")
        } else if patch_changed {
            (0.85, "Patch version bump suggests bug fixes only")
        } else {
            (0.9, "No version change detected")
        };
        Self {
            value,
            source: ConfidenceSource::SemverAnalysis,
            rationale: rationale.to_string(),
            evidence_count: 1,
        }
    }

    /// User-provided override confidence.
    pub fn from_user_override(value: f64, rationale: &str) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            source: ConfidenceSource::UserOverride,
            rationale: rationale.to_string(),
            evidence_count: 1,
        }
    }

    /// Confidence from historical deployment data.
    pub fn from_historical_data(
        successful_deployments: usize,
        total_deployments: usize,
        rationale: &str,
    ) -> Self {
        let value = if total_deployments == 0 {
            0.5
        } else {
            let raw = successful_deployments as f64 / total_deployments as f64;
            // Apply a slight downward adjustment for small samples
            let sample_factor = 1.0 - (1.0 / (total_deployments as f64 + 1.0));
            raw * sample_factor + 0.5 * (1.0 - sample_factor)
        };
        Self {
            value: value.clamp(0.0, 1.0),
            source: ConfidenceSource::HistoricalData,
            rationale: rationale.to_string(),
            evidence_count: total_deployments,
        }
    }

    /// Create a heuristic-based confidence.
    pub fn from_heuristic(value: f64, rationale: &str) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            source: ConfidenceSource::Heuristic,
            rationale: rationale.to_string(),
            evidence_count: 1,
        }
    }

    pub fn is_high(&self) -> bool {
        self.value >= 0.8
    }

    pub fn is_medium(&self) -> bool {
        self.value >= 0.5 && self.value < 0.8
    }

    pub fn is_low(&self) -> bool {
        self.value < 0.5
    }

    pub fn label(&self) -> &'static str {
        if self.is_high() {
            "high"
        } else if self.is_medium() {
            "medium"
        } else {
            "low"
        }
    }

    /// Combine with another score, yielding a weighted average.
    pub fn combine(&self, other: &ConfidenceScore) -> ConfidenceScore {
        let total_evidence = self.evidence_count + other.evidence_count;
        let w1 = self.evidence_count as f64 / total_evidence as f64;
        let w2 = other.evidence_count as f64 / total_evidence as f64;
        let combined_value = self.value * w1 + other.value * w2;
        ConfidenceScore {
            value: combined_value.clamp(0.0, 1.0),
            source: ConfidenceSource::Combined,
            rationale: format!(
                "Combined: [{}] ({:.2}) + [{}] ({:.2})",
                self.source, self.value, other.source, other.value
            ),
            evidence_count: total_evidence,
        }
    }
}

impl fmt::Display for ConfidenceScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} ({}, {})", self.value, self.label(), self.source)
    }
}

/// Combines multiple confidence sources via Bayesian-style updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceModel {
    pub prior: f64,
    pub source_weights: Vec<(ConfidenceSource, f64)>,
}

impl Default for ConfidenceModel {
    fn default() -> Self {
        Self {
            prior: 0.5,
            source_weights: vec![
                (ConfidenceSource::SchemaDiff, 1.0),
                (ConfidenceSource::SemverAnalysis, 0.7),
                (ConfidenceSource::UserOverride, 0.9),
                (ConfidenceSource::HistoricalData, 0.8),
                (ConfidenceSource::Heuristic, 0.5),
            ],
        }
    }
}

impl ConfidenceModel {
    pub fn new(prior: f64) -> Self {
        Self {
            prior: prior.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    pub fn with_weight(mut self, source: ConfidenceSource, weight: f64) -> Self {
        if let Some(entry) = self.source_weights.iter_mut().find(|(s, _)| *s == source) {
            entry.1 = weight;
        } else {
            self.source_weights.push((source, weight));
        }
        self
    }

    fn weight_for(&self, source: &ConfidenceSource) -> f64 {
        self.source_weights
            .iter()
            .find(|(s, _)| s == source)
            .map(|(_, w)| *w)
            .unwrap_or(0.5)
    }

    /// Bayesian update: given a prior probability and new evidence,
    /// compute a posterior probability.
    pub fn bayesian_update(&self, prior: f64, evidence: &ConfidenceScore) -> f64 {
        let weight = self.weight_for(&evidence.source);
        // Treat evidence.value as the likelihood ratio scaled by source weight
        let likelihood_if_true = evidence.value * weight + (1.0 - weight) * 0.5;
        let likelihood_if_false = (1.0 - evidence.value) * weight + (1.0 - weight) * 0.5;

        let numerator = likelihood_if_true * prior;
        let denominator = numerator + likelihood_if_false * (1.0 - prior);

        if denominator < 1e-12 {
            prior
        } else {
            (numerator / denominator).clamp(0.0, 1.0)
        }
    }

    /// Aggregate multiple scores by sequential Bayesian update.
    pub fn aggregate(&self, scores: &[ConfidenceScore]) -> ConfidenceScore {
        if scores.is_empty() {
            return ConfidenceScore::new(
                self.prior,
                ConfidenceSource::Combined,
                "No evidence provided, returning prior",
            );
        }
        if scores.len() == 1 {
            return scores[0].clone();
        }

        let mut posterior = self.prior;
        let mut total_evidence = 0usize;
        let mut rationale_parts = Vec::new();

        for score in scores {
            posterior = self.bayesian_update(posterior, score);
            total_evidence += score.evidence_count;
            rationale_parts.push(format!("{}:{:.2}", score.source, score.value));
        }

        ConfidenceScore {
            value: posterior,
            source: ConfidenceSource::Combined,
            rationale: format!("Bayesian aggregate of [{}]", rationale_parts.join(", ")),
            evidence_count: total_evidence,
        }
    }

    /// Simple weighted average aggregation (alternative to Bayesian).
    pub fn weighted_average(&self, scores: &[ConfidenceScore]) -> ConfidenceScore {
        if scores.is_empty() {
            return ConfidenceScore::new(
                self.prior,
                ConfidenceSource::Combined,
                "No scores to average",
            );
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        let mut total_evidence = 0usize;

        for score in scores {
            let w = self.weight_for(&score.source) * score.evidence_count as f64;
            weighted_sum += score.value * w;
            total_weight += w;
            total_evidence += score.evidence_count;
        }

        let value = if total_weight < 1e-12 {
            self.prior
        } else {
            weighted_sum / total_weight
        };

        ConfidenceScore {
            value: value.clamp(0.0, 1.0),
            source: ConfidenceSource::Combined,
            rationale: format!("Weighted average of {} scores", scores.len()),
            evidence_count: total_evidence,
        }
    }

    /// Most conservative: take the minimum confidence.
    pub fn conservative(&self, scores: &[ConfidenceScore]) -> ConfidenceScore {
        if scores.is_empty() {
            return ConfidenceScore::new(
                self.prior,
                ConfidenceSource::Combined,
                "No scores to evaluate",
            );
        }

        let min_score = scores
            .iter()
            .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        ConfidenceScore {
            value: min_score.value,
            source: ConfidenceSource::Combined,
            rationale: format!(
                "Conservative (min) of {} scores, driven by {}",
                scores.len(),
                min_score.source
            ),
            evidence_count: scores.iter().map(|s| s.evidence_count).sum(),
        }
    }
}

/// Red-tagging system for low-confidence assessments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedTagging {
    pub threshold: f64,
}

/// A tagged assessment, either high-confidence or red-tagged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedAssessment {
    pub score: ConfidenceScore,
    pub is_red_tagged: bool,
    pub tag_reason: Option<String>,
}

impl Default for RedTagging {
    fn default() -> Self {
        Self { threshold: 0.7 }
    }
}

impl RedTagging {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Tag a single assessment.
    pub fn tag_one(&self, score: &ConfidenceScore) -> TaggedAssessment {
        let is_red = score.value < self.threshold;
        let reason = if is_red {
            Some(format!(
                "Confidence {:.2} below threshold {:.2} (source: {})",
                score.value, self.threshold, score.source
            ))
        } else {
            None
        };
        TaggedAssessment {
            score: score.clone(),
            is_red_tagged: is_red,
            tag_reason: reason,
        }
    }

    /// Partition a slice of scores into (high_confidence, red_tagged).
    pub fn tag(
        &self,
        assessments: &[ConfidenceScore],
    ) -> (Vec<TaggedAssessment>, Vec<TaggedAssessment>) {
        let mut high = Vec::new();
        let mut red = Vec::new();

        for score in assessments {
            let tagged = self.tag_one(score);
            if tagged.is_red_tagged {
                red.push(tagged);
            } else {
                high.push(tagged);
            }
        }

        (high, red)
    }

    /// Get the fraction of assessments that are red-tagged.
    pub fn red_tag_ratio(&self, assessments: &[ConfidenceScore]) -> f64 {
        if assessments.is_empty() {
            return 0.0;
        }
        let red_count = assessments
            .iter()
            .filter(|s| s.value < self.threshold)
            .count();
        red_count as f64 / assessments.len() as f64
    }

    /// Returns true if any assessment in the list is red-tagged.
    pub fn has_red_tags(&self, assessments: &[ConfidenceScore]) -> bool {
        assessments.iter().any(|s| s.value < self.threshold)
    }

    /// Summarize red-tagging results.
    pub fn summary(&self, assessments: &[ConfidenceScore]) -> RedTagSummary {
        let (high, red) = self.tag(assessments);
        RedTagSummary {
            total: assessments.len(),
            high_confidence_count: high.len(),
            red_tagged_count: red.len(),
            average_confidence: if assessments.is_empty() {
                0.0
            } else {
                assessments.iter().map(|s| s.value).sum::<f64>() / assessments.len() as f64
            },
            min_confidence: assessments
                .iter()
                .map(|s| s.value)
                .fold(f64::INFINITY, f64::min),
            max_confidence: assessments
                .iter()
                .map(|s| s.value)
                .fold(f64::NEG_INFINITY, f64::max),
            threshold: self.threshold,
        }
    }
}

/// Summary of red-tagging results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedTagSummary {
    pub total: usize,
    pub high_confidence_count: usize,
    pub red_tagged_count: usize,
    pub average_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub threshold: f64,
}

impl fmt::Display for RedTagSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} high-confidence (threshold={:.2}), avg={:.2}, range=[{:.2}, {:.2}]",
            self.high_confidence_count,
            self.total,
            self.threshold,
            self.average_confidence,
            self.min_confidence,
            self.max_confidence,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_from_schema_diff_no_changes() {
        let score = ConfidenceScore::from_schema_diff(0, 0);
        assert!(score.value > 0.9);
        assert!(score.is_high());
    }

    #[test]
    fn test_confidence_from_schema_diff_no_breaking() {
        let score = ConfidenceScore::from_schema_diff(0, 5);
        assert!(score.value >= 0.85);
        assert!(score.is_high());
    }

    #[test]
    fn test_confidence_from_schema_diff_some_breaking() {
        let score = ConfidenceScore::from_schema_diff(3, 10);
        let score2 = ConfidenceScore::from_schema_diff(0, 10);
        assert!(score.value < score2.value);
    }

    #[test]
    fn test_confidence_from_semver_major() {
        let score = ConfidenceScore::from_semver_analysis(true, false, false);
        assert!(score.value < 0.7);
        assert_eq!(score.source, ConfidenceSource::SemverAnalysis);
    }

    #[test]
    fn test_confidence_from_semver_minor() {
        let score = ConfidenceScore::from_semver_analysis(false, true, false);
        assert!(score.value >= 0.7);
        assert!(score.value < 0.85);
    }

    #[test]
    fn test_confidence_from_semver_patch() {
        let score = ConfidenceScore::from_semver_analysis(false, false, true);
        assert!(score.value >= 0.8);
    }

    #[test]
    fn test_user_override() {
        let score = ConfidenceScore::from_user_override(0.99, "Manual verification");
        assert!((score.value - 0.99).abs() < 1e-9);
        assert_eq!(score.source, ConfidenceSource::UserOverride);
    }

    #[test]
    fn test_historical_data_empty() {
        let score = ConfidenceScore::from_historical_data(0, 0, "No history");
        assert!((score.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_historical_data_good() {
        let score = ConfidenceScore::from_historical_data(100, 100, "All green");
        assert!(score.value > 0.8);
    }

    #[test]
    fn test_combine_scores() {
        let s1 = ConfidenceScore::from_schema_diff(0, 5);
        let s2 = ConfidenceScore::from_semver_analysis(false, true, false);
        let combined = s1.combine(&s2);
        assert_eq!(combined.source, ConfidenceSource::Combined);
        assert!(combined.value > 0.0 && combined.value < 1.0);
    }

    #[test]
    fn test_confidence_model_bayesian() {
        let model = ConfidenceModel::default();
        let scores = vec![
            ConfidenceScore::from_schema_diff(0, 5),
            ConfidenceScore::from_semver_analysis(false, true, false),
        ];
        let result = model.aggregate(&scores);
        assert_eq!(result.source, ConfidenceSource::Combined);
        assert!(result.value > 0.0);
    }

    #[test]
    fn test_confidence_model_empty() {
        let model = ConfidenceModel::new(0.6);
        let result = model.aggregate(&[]);
        assert!((result.value - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_model_single() {
        let model = ConfidenceModel::default();
        let score = ConfidenceScore::from_schema_diff(0, 3);
        let result = model.aggregate(&[score.clone()]);
        assert!((result.value - score.value).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_average() {
        let model = ConfidenceModel::default();
        let scores = vec![
            ConfidenceScore::from_schema_diff(0, 10),
            ConfidenceScore::from_semver_analysis(false, false, true),
        ];
        let result = model.weighted_average(&scores);
        assert!(result.value > 0.5);
    }

    #[test]
    fn test_conservative() {
        let model = ConfidenceModel::default();
        let s1 = ConfidenceScore::new(0.9, ConfidenceSource::SchemaDiff, "high");
        let s2 = ConfidenceScore::new(0.3, ConfidenceSource::Heuristic, "low");
        let result = model.conservative(&[s1, s2]);
        assert!((result.value - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_red_tagging_basic() {
        let rt = RedTagging::new(0.7);
        let high = ConfidenceScore::new(0.9, ConfidenceSource::SchemaDiff, "good");
        let low = ConfidenceScore::new(0.4, ConfidenceSource::Heuristic, "bad");

        let (h, r) = rt.tag(&[high, low]);
        assert_eq!(h.len(), 1);
        assert_eq!(r.len(), 1);
        assert!(r[0].is_red_tagged);
        assert!(r[0].tag_reason.is_some());
    }

    #[test]
    fn test_red_tagging_all_high() {
        let rt = RedTagging::new(0.5);
        let scores = vec![
            ConfidenceScore::new(0.8, ConfidenceSource::SchemaDiff, "a"),
            ConfidenceScore::new(0.9, ConfidenceSource::SemverAnalysis, "b"),
        ];
        let (h, r) = rt.tag(&scores);
        assert_eq!(h.len(), 2);
        assert_eq!(r.len(), 0);
        assert!(!rt.has_red_tags(&scores));
    }

    #[test]
    fn test_red_tag_ratio() {
        let rt = RedTagging::new(0.7);
        let scores = vec![
            ConfidenceScore::new(0.9, ConfidenceSource::SchemaDiff, "a"),
            ConfidenceScore::new(0.3, ConfidenceSource::Heuristic, "b"),
            ConfidenceScore::new(0.5, ConfidenceSource::HistoricalData, "c"),
            ConfidenceScore::new(0.8, ConfidenceSource::UserOverride, "d"),
        ];
        let ratio = rt.red_tag_ratio(&scores);
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_red_tag_summary() {
        let rt = RedTagging::new(0.7);
        let scores = vec![
            ConfidenceScore::new(0.9, ConfidenceSource::SchemaDiff, "a"),
            ConfidenceScore::new(0.3, ConfidenceSource::Heuristic, "b"),
        ];
        let summary = rt.summary(&scores);
        assert_eq!(summary.total, 2);
        assert_eq!(summary.high_confidence_count, 1);
        assert_eq!(summary.red_tagged_count, 1);
        assert!((summary.average_confidence - 0.6).abs() < 1e-9);
        assert!((summary.min_confidence - 0.3).abs() < 1e-9);
        assert!((summary.max_confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_display() {
        let score = ConfidenceScore::new(0.85, ConfidenceSource::SchemaDiff, "test");
        let display = format!("{}", score);
        assert!(display.contains("0.85"));
        assert!(display.contains("high"));
    }

    #[test]
    fn test_label() {
        assert_eq!(
            ConfidenceScore::new(0.9, ConfidenceSource::SchemaDiff, "").label(),
            "high"
        );
        assert_eq!(
            ConfidenceScore::new(0.6, ConfidenceSource::SchemaDiff, "").label(),
            "medium"
        );
        assert_eq!(
            ConfidenceScore::new(0.3, ConfidenceSource::SchemaDiff, "").label(),
            "low"
        );
    }

    #[test]
    fn test_clamping() {
        let score = ConfidenceScore::new(1.5, ConfidenceSource::SchemaDiff, "over");
        assert!((score.value - 1.0).abs() < 1e-9);
        let score2 = ConfidenceScore::new(-0.5, ConfidenceSource::SchemaDiff, "under");
        assert!(score2.value.abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_update_high_evidence() {
        let model = ConfidenceModel::default();
        let evidence = ConfidenceScore::new(0.95, ConfidenceSource::SchemaDiff, "strong");
        let posterior = model.bayesian_update(0.5, &evidence);
        assert!(posterior > 0.5);
    }

    #[test]
    fn test_bayesian_update_low_evidence() {
        let model = ConfidenceModel::default();
        let evidence = ConfidenceScore::new(0.1, ConfidenceSource::SchemaDiff, "weak");
        let posterior = model.bayesian_update(0.5, &evidence);
        assert!(posterior < 0.5);
    }
}
