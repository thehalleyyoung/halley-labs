//! Cascade classification and severity scoring.
//!
//! Takes raw findings from Tier 1 / Tier 2 and assigns a structured
//! [`Classification`] with primary and secondary cascade types, a confidence
//! score, and a human-readable description.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::tier1::{AmplificationRisk, FanInRisk, TimeoutViolation};

// ---------------------------------------------------------------------------
// CascadeType
// ---------------------------------------------------------------------------

/// Enumeration of cascade failure patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CascadeType {
    RetryAmplification,
    TimeoutChainViolation,
    FanInStorm,
    CircularDependency,
    ResourceExhaustion,
    MultiFailureCombination,
}

impl fmt::Display for CascadeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CascadeType::RetryAmplification => write!(f, "Retry Amplification"),
            CascadeType::TimeoutChainViolation => write!(f, "Timeout Chain Violation"),
            CascadeType::FanInStorm => write!(f, "Fan-In Storm"),
            CascadeType::CircularDependency => write!(f, "Circular Dependency"),
            CascadeType::ResourceExhaustion => write!(f, "Resource Exhaustion"),
            CascadeType::MultiFailureCombination => write!(f, "Multi-Failure Combination"),
        }
    }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// A fully classified finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub primary_type: CascadeType,
    pub secondary_types: Vec<CascadeType>,
    pub confidence: f64,
    pub description: String,
}

// ---------------------------------------------------------------------------
// CascadeClassifier
// ---------------------------------------------------------------------------

/// Stateless classifier that maps raw analysis findings to cascade types.
#[derive(Debug, Clone)]
pub struct CascadeClassifier;

impl CascadeClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Classify an amplification risk.
    pub fn classify_amplification_risk(risk: &AmplificationRisk) -> Classification {
        let mut secondary = Vec::new();
        if risk.capacity < (risk.amplification_factor as u64) {
            secondary.push(CascadeType::ResourceExhaustion);
        }
        if risk.path.len() > 4 {
            secondary.push(CascadeType::TimeoutChainViolation);
        }

        let confidence = compute_amplification_confidence(risk.amplification_factor, risk.path.len());

        let description = Self::generate_amplification_description(risk);

        Classification {
            primary_type: CascadeType::RetryAmplification,
            secondary_types: secondary,
            confidence,
            description,
        }
    }

    /// Classify a timeout violation.
    pub fn classify_timeout_violation(violation: &TimeoutViolation) -> Classification {
        let mut secondary = Vec::new();
        if violation.path.len() > 5 {
            secondary.push(CascadeType::RetryAmplification);
        }

        let confidence = compute_timeout_confidence(violation.excess_ms, violation.deadline_ms);

        let description = format!(
            "Timeout chain along {} exceeds deadline by {}ms ({} total vs {} deadline)",
            format_path(&violation.path),
            violation.excess_ms,
            violation.total_timeout_ms,
            violation.deadline_ms,
        );

        Classification {
            primary_type: CascadeType::TimeoutChainViolation,
            secondary_types: secondary,
            confidence,
            description,
        }
    }

    /// Classify a fan-in risk.
    pub fn classify_fan_in_risk(risk: &FanInRisk) -> Classification {
        let mut secondary = Vec::new();
        if risk.combined_amplification > 50.0 {
            secondary.push(CascadeType::RetryAmplification);
        }
        if risk.capacity < (risk.combined_amplification as u64) {
            secondary.push(CascadeType::ResourceExhaustion);
        }

        let confidence = compute_fan_in_confidence(
            risk.incoming_paths.len(),
            risk.combined_amplification,
        );

        let description = format!(
            "Service '{}' receives {} incoming paths with combined amplification {:.1}x (capacity: {})",
            risk.service,
            risk.incoming_paths.len(),
            risk.combined_amplification,
            risk.capacity,
        );

        Classification {
            primary_type: CascadeType::FanInStorm,
            secondary_types: secondary,
            confidence,
            description,
        }
    }

    /// Compute a numeric severity score for a classification.
    pub fn compute_severity_score(classification: &Classification) -> f64 {
        let base = match classification.primary_type {
            CascadeType::RetryAmplification => 7.0,
            CascadeType::TimeoutChainViolation => 6.0,
            CascadeType::FanInStorm => 8.0,
            CascadeType::CircularDependency => 9.0,
            CascadeType::ResourceExhaustion => 7.5,
            CascadeType::MultiFailureCombination => 8.5,
        };
        let secondary_bonus = classification.secondary_types.len() as f64 * 0.5;
        let confidence_factor = classification.confidence;
        (base + secondary_bonus) * confidence_factor
    }

    /// Rank a list of classifications by severity score (descending).
    pub fn rank_findings(findings: Vec<Classification>) -> Vec<(Classification, f64)> {
        let mut scored: Vec<(Classification, f64)> = findings
            .into_iter()
            .map(|c| {
                let score = Self::compute_severity_score(&c);
                (c, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Generate a human-readable description for any classification.
    pub fn generate_description(classification: &Classification) -> String {
        let secondary_desc = if classification.secondary_types.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = classification
                .secondary_types
                .iter()
                .map(|t| t.to_string())
                .collect();
            format!(" (also: {})", names.join(", "))
        };
        format!(
            "[{:.0}% confidence] {}: {}{}",
            classification.confidence * 100.0,
            classification.primary_type,
            classification.description,
            secondary_desc,
        )
    }

    // ----- private -----

    fn generate_amplification_description(risk: &AmplificationRisk) -> String {
        format!(
            "Retry amplification of {:.1}x along {} (min capacity: {}; severity: {})",
            risk.amplification_factor,
            format_path(&risk.path),
            risk.capacity,
            risk.severity,
        )
    }
}

impl Default for CascadeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SeverityScorer
// ---------------------------------------------------------------------------

/// Configurable severity scoring with adjustable weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityScorer {
    pub amplification_weight: f64,
    pub timeout_weight: f64,
    pub fan_in_weight: f64,
    pub secondary_bonus: f64,
}

impl Default for SeverityScorer {
    fn default() -> Self {
        Self {
            amplification_weight: 1.0,
            timeout_weight: 0.8,
            fan_in_weight: 1.2,
            secondary_bonus: 0.5,
        }
    }
}

impl SeverityScorer {
    pub fn new(
        amplification_weight: f64,
        timeout_weight: f64,
        fan_in_weight: f64,
        secondary_bonus: f64,
    ) -> Self {
        Self {
            amplification_weight,
            timeout_weight,
            fan_in_weight,
            secondary_bonus,
        }
    }

    /// Score a classification using the configured weights.
    pub fn score(&self, classification: &Classification) -> f64 {
        let weight = match classification.primary_type {
            CascadeType::RetryAmplification => self.amplification_weight,
            CascadeType::TimeoutChainViolation => self.timeout_weight,
            CascadeType::FanInStorm => self.fan_in_weight,
            CascadeType::CircularDependency => 1.5,
            CascadeType::ResourceExhaustion => 1.0,
            CascadeType::MultiFailureCombination => 1.3,
        };
        let base = match classification.primary_type {
            CascadeType::RetryAmplification => 7.0,
            CascadeType::TimeoutChainViolation => 6.0,
            CascadeType::FanInStorm => 8.0,
            CascadeType::CircularDependency => 9.0,
            CascadeType::ResourceExhaustion => 7.5,
            CascadeType::MultiFailureCombination => 8.5,
        };
        let secondary = classification.secondary_types.len() as f64 * self.secondary_bonus;
        (base * weight + secondary) * classification.confidence
    }

    /// Score and rank a batch of classifications.
    pub fn rank(&self, findings: Vec<Classification>) -> Vec<(Classification, f64)> {
        let mut scored: Vec<(Classification, f64)> = findings
            .into_iter()
            .map(|c| {
                let s = self.score(&c);
                (c, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_path(path: &[String]) -> String {
    path.join(" → ")
}

fn compute_amplification_confidence(amplification: f64, path_len: usize) -> f64 {
    // Higher amplification and shorter paths give higher confidence.
    let amp_factor = (amplification.log2().max(0.0) / 10.0).min(1.0);
    let len_factor = if path_len <= 3 { 1.0 } else { 0.9_f64.powi(path_len as i32 - 3) };
    (amp_factor * 0.7 + len_factor * 0.3).clamp(0.1, 1.0)
}

fn compute_timeout_confidence(excess_ms: u64, deadline_ms: u64) -> f64 {
    if deadline_ms == 0 {
        return 0.5;
    }
    let ratio = excess_ms as f64 / deadline_ms as f64;
    (ratio / (ratio + 1.0)).clamp(0.1, 1.0)
}

fn compute_fan_in_confidence(num_paths: usize, combined_amp: f64) -> f64 {
    let path_factor = (num_paths as f64 / 5.0).min(1.0);
    let amp_factor = (combined_amp.log2().max(0.0) / 10.0).min(1.0);
    (path_factor * 0.5 + amp_factor * 0.5).clamp(0.1, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::{AmplificationRisk, FanInRisk, TimeoutViolation};

    fn sample_amplification_risk() -> AmplificationRisk {
        AmplificationRisk {
            path: vec!["A".into(), "B".into(), "C".into()],
            amplification_factor: 27.0,
            capacity: 100,
            severity: "high".into(),
        }
    }

    fn sample_timeout_violation() -> TimeoutViolation {
        TimeoutViolation {
            path: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            total_timeout_ms: 15_000,
            deadline_ms: 5_000,
            excess_ms: 10_000,
        }
    }

    fn sample_fan_in_risk() -> FanInRisk {
        FanInRisk {
            service: "D".into(),
            incoming_paths: vec![
                vec!["A".into(), "B".into(), "D".into()],
                vec!["A".into(), "C".into(), "D".into()],
            ],
            combined_amplification: 18.0,
            capacity: 50,
        }
    }

    #[test]
    fn test_classify_amplification() {
        let cls = CascadeClassifier::classify_amplification_risk(&sample_amplification_risk());
        assert_eq!(cls.primary_type, CascadeType::RetryAmplification);
        assert!(cls.confidence > 0.0 && cls.confidence <= 1.0);
    }

    #[test]
    fn test_classify_timeout() {
        let cls = CascadeClassifier::classify_timeout_violation(&sample_timeout_violation());
        assert_eq!(cls.primary_type, CascadeType::TimeoutChainViolation);
        assert!(cls.description.contains("exceeds deadline"));
    }

    #[test]
    fn test_classify_fan_in() {
        let cls = CascadeClassifier::classify_fan_in_risk(&sample_fan_in_risk());
        assert_eq!(cls.primary_type, CascadeType::FanInStorm);
        assert!(cls.description.contains("D"));
    }

    #[test]
    fn test_amplification_secondary_resource_exhaustion() {
        let risk = AmplificationRisk {
            path: vec!["A".into(), "B".into()],
            amplification_factor: 200.0,
            capacity: 10, // capacity < amplification
            severity: "critical".into(),
        };
        let cls = CascadeClassifier::classify_amplification_risk(&risk);
        assert!(cls.secondary_types.contains(&CascadeType::ResourceExhaustion));
    }

    #[test]
    fn test_severity_score_positive() {
        let cls = CascadeClassifier::classify_amplification_risk(&sample_amplification_risk());
        let score = CascadeClassifier::compute_severity_score(&cls);
        assert!(score > 0.0);
    }

    #[test]
    fn test_severity_score_secondary_bonus() {
        let mut cls = CascadeClassifier::classify_amplification_risk(&sample_amplification_risk());
        let score_without = CascadeClassifier::compute_severity_score(&cls);
        cls.secondary_types.push(CascadeType::ResourceExhaustion);
        let score_with = CascadeClassifier::compute_severity_score(&cls);
        assert!(score_with > score_without);
    }

    #[test]
    fn test_rank_findings() {
        let findings = vec![
            CascadeClassifier::classify_amplification_risk(&sample_amplification_risk()),
            CascadeClassifier::classify_timeout_violation(&sample_timeout_violation()),
            CascadeClassifier::classify_fan_in_risk(&sample_fan_in_risk()),
        ];
        let ranked = CascadeClassifier::rank_findings(findings);
        assert_eq!(ranked.len(), 3);
        // Scores should be in descending order.
        for i in 0..ranked.len() - 1 {
            assert!(ranked[i].1 >= ranked[i + 1].1);
        }
    }

    #[test]
    fn test_generate_description_non_empty() {
        let cls = CascadeClassifier::classify_amplification_risk(&sample_amplification_risk());
        let desc = CascadeClassifier::generate_description(&cls);
        assert!(!desc.is_empty());
        assert!(desc.contains("Retry Amplification"));
    }

    #[test]
    fn test_severity_scorer_custom_weights() {
        let scorer = SeverityScorer::new(2.0, 0.5, 1.0, 0.3);
        let cls = CascadeClassifier::classify_amplification_risk(&sample_amplification_risk());
        let score = scorer.score(&cls);
        assert!(score > 0.0);
    }

    #[test]
    fn test_severity_scorer_rank() {
        let scorer = SeverityScorer::default();
        let findings = vec![
            CascadeClassifier::classify_amplification_risk(&sample_amplification_risk()),
            CascadeClassifier::classify_timeout_violation(&sample_timeout_violation()),
        ];
        let ranked = scorer.rank(findings);
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn test_cascade_type_display() {
        assert_eq!(CascadeType::RetryAmplification.to_string(), "Retry Amplification");
        assert_eq!(CascadeType::FanInStorm.to_string(), "Fan-In Storm");
    }

    #[test]
    fn test_confidence_bounds() {
        let c1 = compute_amplification_confidence(1.0, 2);
        assert!(c1 >= 0.1 && c1 <= 1.0);
        let c2 = compute_timeout_confidence(0, 1000);
        assert!(c2 >= 0.1 && c2 <= 1.0);
        let c3 = compute_fan_in_confidence(1, 1.0);
        assert!(c3 >= 0.1 && c3 <= 1.0);
    }
}
