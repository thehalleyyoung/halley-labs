//! Futility certificate: empirically calibrated prediction.
//!
//! NOT a formal proof but an empirically calibrated prediction about whether
//! decomposition will be futile. Uses spectral features and calibrated thresholds.

use crate::error::{CertificateError, CertificateResult};
use crate::futility::{CertificateType, FutilityPrediction};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Spectral features used for futility prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub spectral_gap: f64,
    pub spectral_ratio: f64,
    pub algebraic_connectivity: f64,
    pub condition_number: f64,
    pub num_blocks: usize,
    pub balance_ratio: f64,
    pub crossing_density: f64,
    pub eigenvalue_decay_rate: f64,
    pub normalized_cut_value: f64,
    pub modularity: f64,
    pub additional: IndexMap<String, f64>,
}

impl SpectralFeatures {
    pub fn new(
        spectral_gap: f64,
        spectral_ratio: f64,
        algebraic_connectivity: f64,
        condition_number: f64,
        num_blocks: usize,
    ) -> Self {
        Self {
            spectral_gap,
            spectral_ratio,
            algebraic_connectivity,
            condition_number,
            num_blocks,
            balance_ratio: 1.0,
            crossing_density: 0.0,
            eigenvalue_decay_rate: 0.0,
            normalized_cut_value: 0.0,
            modularity: 0.0,
            additional: IndexMap::new(),
        }
    }

    /// Convert features to a vector for model input.
    pub fn to_vector(&self) -> Vec<f64> {
        let mut v = vec![
            self.spectral_gap,
            self.spectral_ratio,
            self.algebraic_connectivity,
            self.condition_number,
            self.num_blocks as f64,
            self.balance_ratio,
            self.crossing_density,
            self.eigenvalue_decay_rate,
            self.normalized_cut_value,
            self.modularity,
        ];
        for val in self.additional.values() {
            v.push(*val);
        }
        v
    }

    /// Feature names in the same order as to_vector().
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = vec![
            "spectral_gap".to_string(),
            "spectral_ratio".to_string(),
            "algebraic_connectivity".to_string(),
            "condition_number".to_string(),
            "num_blocks".to_string(),
            "balance_ratio".to_string(),
            "crossing_density".to_string(),
            "eigenvalue_decay_rate".to_string(),
            "normalized_cut_value".to_string(),
            "modularity".to_string(),
        ];
        for key in self.additional.keys() {
            names.push(key.clone());
        }
        names
    }

    /// Validate that features are reasonable.
    pub fn validate(&self) -> CertificateResult<()> {
        if self.spectral_gap.is_nan() || self.spectral_gap.is_infinite() {
            return Err(CertificateError::numerical_precision(
                "spectral_gap is non-finite",
                self.spectral_gap,
                0.0,
            ));
        }
        if self.condition_number < 1.0 {
            return Err(CertificateError::numerical_precision(
                "condition number < 1",
                self.condition_number,
                1.0,
            ));
        }
        if self.num_blocks == 0 {
            return Err(CertificateError::invalid_partition("zero blocks in features"));
        }
        Ok(())
    }
}

/// Threshold configuration for futility classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityThresholds {
    pub spectral_ratio_threshold: f64,
    pub condition_number_threshold: f64,
    pub spectral_gap_threshold: f64,
    pub combined_score_threshold: f64,
    pub uncertainty_margin: f64,
}

impl Default for FutilityThresholds {
    fn default() -> Self {
        Self {
            spectral_ratio_threshold: 0.3,
            condition_number_threshold: 1e3,
            spectral_gap_threshold: 0.01,
            combined_score_threshold: 0.5,
            uncertainty_margin: 0.1,
        }
    }
}

/// Calibration data reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReference {
    pub calibration_id: String,
    pub num_training_instances: usize,
    pub calibration_date: String,
    pub brier_score: f64,
    pub reliability_score: f64,
    pub num_features_used: usize,
}

/// Futility certificate — empirical, NOT formal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutilityCertificate {
    pub id: String,
    pub created_at: String,
    /// Always Empirical — this is NOT a formal proof.
    pub certificate_type: CertificateType,
    pub features: SpectralFeatures,
    pub thresholds: FutilityThresholds,
    pub prediction: FutilityPrediction,
    pub futility_score: f64,
    pub confidence: f64,
    pub per_feature_scores: IndexMap<String, f64>,
    pub calibration_ref: Option<CalibrationReference>,
    pub metadata: IndexMap<String, String>,
}

impl FutilityCertificate {
    /// Generate a futility certificate from spectral features.
    ///
    /// NOTE: This is an EMPIRICAL prediction, not a formal proof.
    pub fn generate(
        features: SpectralFeatures,
        thresholds: FutilityThresholds,
        calibration_ref: Option<CalibrationReference>,
    ) -> CertificateResult<Self> {
        features.validate()?;

        let mut per_feature_scores = IndexMap::new();

        // Score each feature: higher = more likely to be futile
        let ratio_score = Self::sigmoid(
            features.spectral_ratio - thresholds.spectral_ratio_threshold,
            10.0,
        );
        per_feature_scores.insert("spectral_ratio".to_string(), ratio_score);

        let kappa_score = Self::sigmoid(
            (features.condition_number.log10() - thresholds.condition_number_threshold.log10())
                / thresholds.condition_number_threshold.log10(),
            5.0,
        );
        per_feature_scores.insert("condition_number".to_string(), kappa_score);

        let gap_score = Self::sigmoid(
            thresholds.spectral_gap_threshold - features.spectral_gap,
            20.0,
        );
        per_feature_scores.insert("spectral_gap".to_string(), gap_score);

        let balance_score = Self::sigmoid(0.5 - features.balance_ratio, 5.0);
        per_feature_scores.insert("balance_ratio".to_string(), balance_score);

        let crossing_score = Self::sigmoid(features.crossing_density - 0.5, 5.0);
        per_feature_scores.insert("crossing_density".to_string(), crossing_score);

        let modularity_score = Self::sigmoid(-features.modularity, 5.0);
        per_feature_scores.insert("modularity".to_string(), modularity_score);

        // Combined futility score (weighted average)
        let weights = [0.25, 0.20, 0.20, 0.10, 0.15, 0.10];
        let scores = [
            ratio_score,
            kappa_score,
            gap_score,
            balance_score,
            crossing_score,
            modularity_score,
        ];
        let futility_score: f64 =
            weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();

        // Determine prediction
        let prediction = if futility_score > thresholds.combined_score_threshold + thresholds.uncertainty_margin {
            FutilityPrediction::Futile
        } else if futility_score < thresholds.combined_score_threshold - thresholds.uncertainty_margin {
            FutilityPrediction::NotFutile
        } else {
            FutilityPrediction::Uncertain
        };

        // Confidence: distance from threshold, calibrated
        let distance_from_threshold =
            (futility_score - thresholds.combined_score_threshold).abs();
        let raw_confidence = (distance_from_threshold / thresholds.uncertainty_margin).min(1.0);
        let confidence = raw_confidence * raw_confidence;

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            certificate_type: CertificateType::Empirical,
            features,
            thresholds,
            prediction,
            futility_score,
            confidence,
            per_feature_scores,
            calibration_ref: calibration_ref,
            metadata: IndexMap::new(),
        })
    }

    fn sigmoid(x: f64, steepness: f64) -> f64 {
        1.0 / (1.0 + (-steepness * x).exp())
    }

    /// Confidence level based on distance from threshold.
    pub fn confidence_level(&self) -> &str {
        if self.confidence > 0.8 {
            "high"
        } else if self.confidence > 0.5 {
            "medium"
        } else if self.confidence > 0.2 {
            "low"
        } else {
            "very_low"
        }
    }

    /// Calibration quality: how well was the threshold calibrated?
    pub fn calibration_quality(&self) -> IndexMap<String, f64> {
        let mut quality = IndexMap::new();

        if let Some(ref cal) = self.calibration_ref {
            quality.insert("brier_score".to_string(), cal.brier_score);
            quality.insert("reliability_score".to_string(), cal.reliability_score);
            quality.insert(
                "training_instances".to_string(),
                cal.num_training_instances as f64,
            );
            quality.insert("num_features".to_string(), cal.num_features_used as f64);

            // Overall calibration quality
            let cal_quality = if cal.brier_score < 0.1 && cal.num_training_instances > 100 {
                0.9
            } else if cal.brier_score < 0.2 && cal.num_training_instances > 50 {
                0.7
            } else if cal.brier_score < 0.3 {
                0.5
            } else {
                0.3
            };
            quality.insert("overall_quality".to_string(), cal_quality);
        } else {
            quality.insert("overall_quality".to_string(), 0.0);
            quality.insert("note".to_string(), 0.0); // "no calibration data"
        }

        quality
    }

    /// Which features contribute most to the futility prediction?
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let mut importance: Vec<(String, f64)> = self
            .per_feature_scores
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        importance.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        importance
    }

    /// How would the prediction change if each feature changed by some fraction?
    pub fn sensitivity_analysis(&self, perturbation_fraction: f64) -> IndexMap<String, f64> {
        let mut sensitivity = IndexMap::new();
        let base_score = self.futility_score;

        let features_vec = self.features.to_vector();
        let feature_names = self.features.feature_names();

        for (i, name) in feature_names.iter().enumerate() {
            if i >= features_vec.len() {
                break;
            }
            let original = features_vec[i];
            let delta = original.abs() * perturbation_fraction;
            if delta < 1e-15 {
                sensitivity.insert(name.clone(), 0.0);
                continue;
            }

            // Approximate sensitivity: how much does feature score change?
            if let Some(score) = self.per_feature_scores.get(name) {
                let approx_sensitivity = score / (base_score.abs().max(1e-10));
                sensitivity.insert(name.clone(), approx_sensitivity);
            }
        }

        sensitivity
    }

    /// Summary as displayable text.
    pub fn summary_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push("═══════════════════════════════════════".to_string());
        lines.push("  FUTILITY CERTIFICATE (EMPIRICAL)".to_string());
        lines.push("═══════════════════════════════════════".to_string());
        lines.push(format!("  ID: {}", self.id));
        lines.push(format!("  Prediction: {}", self.prediction));
        lines.push(format!("  Futility Score: {:.4}", self.futility_score));
        lines.push(format!(
            "  Confidence: {:.4} ({})",
            self.confidence,
            self.confidence_level()
        ));
        lines.push(String::new());
        lines.push("  ⚠ This is an EMPIRICAL prediction,".to_string());
        lines.push("    NOT a formal mathematical proof.".to_string());
        lines.push(String::new());
        lines.push("  Feature Scores:".to_string());
        for (name, score) in &self.per_feature_scores {
            lines.push(format!("    {}: {:.4}", name, score));
        }
        if let Some(ref cal) = self.calibration_ref {
            lines.push(String::new());
            lines.push(format!("  Calibration: {}", cal.calibration_id));
            lines.push(format!("  Brier Score: {:.4}", cal.brier_score));
            lines.push(format!("  Training N: {}", cal.num_training_instances));
        }
        lines.push("═══════════════════════════════════════".to_string());
        lines.join("\n")
    }

    /// Summary statistics.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("futility_score".to_string(), self.futility_score);
        stats.insert("confidence".to_string(), self.confidence);
        stats.insert(
            "is_futile".to_string(),
            match self.prediction {
                FutilityPrediction::Futile => 1.0,
                FutilityPrediction::NotFutile => 0.0,
                FutilityPrediction::Uncertain => 0.5,
            },
        );
        for (k, v) in &self.per_feature_scores {
            stats.insert(format!("feature_{}", k), *v);
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_features() -> SpectralFeatures {
        SpectralFeatures {
            spectral_gap: 0.5,
            spectral_ratio: 0.1,
            algebraic_connectivity: 0.3,
            condition_number: 10.0,
            num_blocks: 3,
            balance_ratio: 0.8,
            crossing_density: 0.2,
            eigenvalue_decay_rate: 0.5,
            normalized_cut_value: 0.3,
            modularity: 0.6,
            additional: IndexMap::new(),
        }
    }

    #[test]
    fn test_generate_not_futile() {
        let features = make_test_features();
        let thresholds = FutilityThresholds::default();
        let cert = FutilityCertificate::generate(features, thresholds, None).unwrap();
        assert_eq!(cert.certificate_type, CertificateType::Empirical);
        assert_eq!(cert.prediction, FutilityPrediction::NotFutile);
    }

    #[test]
    fn test_generate_futile() {
        let mut features = make_test_features();
        features.spectral_gap = 0.001;
        features.spectral_ratio = 0.8;
        features.condition_number = 1e5;
        features.crossing_density = 0.8;
        features.modularity = -0.1;
        features.balance_ratio = 0.1;
        let thresholds = FutilityThresholds::default();
        let cert = FutilityCertificate::generate(features, thresholds, None).unwrap();
        assert_eq!(cert.prediction, FutilityPrediction::Futile);
    }

    #[test]
    fn test_invalid_features_fail() {
        let mut features = make_test_features();
        features.condition_number = 0.5;
        let result = FutilityCertificate::generate(features, FutilityThresholds::default(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_confidence_level() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let level = cert.confidence_level();
        assert!(
            level == "high" || level == "medium" || level == "low" || level == "very_low"
        );
    }

    #[test]
    fn test_calibration_quality_no_ref() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let quality = cert.calibration_quality();
        assert!((quality["overall_quality"] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_quality_with_ref() {
        let features = make_test_features();
        let cal_ref = CalibrationReference {
            calibration_id: "cal-001".to_string(),
            num_training_instances: 200,
            calibration_date: "2024-01-01".to_string(),
            brier_score: 0.08,
            reliability_score: 0.95,
            num_features_used: 10,
        };
        let cert = FutilityCertificate::generate(
            features,
            FutilityThresholds::default(),
            Some(cal_ref),
        )
        .unwrap();
        let quality = cert.calibration_quality();
        assert!(quality["overall_quality"] > 0.5);
    }

    #[test]
    fn test_feature_importance() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let importance = cert.feature_importance();
        assert!(!importance.is_empty());
        // Should be sorted descending
        for i in 1..importance.len() {
            assert!(importance[i - 1].1 >= importance[i].1);
        }
    }

    #[test]
    fn test_sensitivity_analysis() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let sens = cert.sensitivity_analysis(0.1);
        assert!(!sens.is_empty());
    }

    #[test]
    fn test_summary_text() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let text = cert.summary_text();
        assert!(text.contains("FUTILITY CERTIFICATE"));
        assert!(text.contains("EMPIRICAL"));
    }

    #[test]
    fn test_summary_stats() {
        let features = make_test_features();
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let stats = cert.summary_stats();
        assert!(stats.contains_key("futility_score"));
        assert!(stats.contains_key("confidence"));
    }

    #[test]
    fn test_spectral_features_to_vector() {
        let features = make_test_features();
        let vec = features.to_vector();
        assert_eq!(vec.len(), 10);
        assert!((vec[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_features_names() {
        let features = make_test_features();
        let names = features.feature_names();
        assert_eq!(names.len(), features.to_vector().len());
        assert_eq!(names[0], "spectral_gap");
    }

    #[test]
    fn test_sigmoid() {
        let mid = FutilityCertificate::sigmoid(0.0, 1.0);
        assert!((mid - 0.5).abs() < 1e-10);
        let high = FutilityCertificate::sigmoid(10.0, 1.0);
        assert!(high > 0.99);
        let low = FutilityCertificate::sigmoid(-10.0, 1.0);
        assert!(low < 0.01);
    }
}
