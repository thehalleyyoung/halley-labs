//! Spectral bound certificates.
//!
//! This module provides certificates based on spectral analysis of the constraint
//! matrix structure. The T2 spectral scaling law bounds the decomposition gap
//! using spectral properties, while Davis-Kahan certificates bound eigenspace
//! perturbation, and partition quality certificates assess clustering output.

pub mod davis_kahan;
pub mod partition_quality;
pub mod scaling_law;

pub use davis_kahan::DavisKahanCertificate;
pub use partition_quality::PartitionQualityCertificate;
pub use scaling_law::SpectralScalingCertificate;

use serde::{Deserialize, Serialize};

/// Summary of spectral bound analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralBoundSummary {
    pub scaling_law_bound: Option<f64>,
    pub is_vacuous: bool,
    pub davis_kahan_angle_bound: Option<f64>,
    pub partition_quality_score: Option<f64>,
    pub spectral_gap: Option<f64>,
    pub condition_number: Option<f64>,
}

impl SpectralBoundSummary {
    /// Create a summary from optional certificates.
    pub fn from_certificates(
        scaling: Option<&SpectralScalingCertificate>,
        dk: Option<&DavisKahanCertificate>,
        pq: Option<&PartitionQualityCertificate>,
    ) -> Self {
        Self {
            scaling_law_bound: scaling.map(|s| s.bound_value),
            is_vacuous: scaling.map_or(true, |s| s.is_vacuous),
            davis_kahan_angle_bound: dk.map(|d| d.angle_bound),
            partition_quality_score: pq.map(|p| p.overall_quality_score),
            spectral_gap: scaling.map(|s| s.gamma_squared),
            condition_number: scaling.map(|s| s.kappa),
        }
    }

    /// Whether the spectral analysis provides useful information.
    pub fn is_informative(&self) -> bool {
        !self.is_vacuous && self.scaling_law_bound.is_some()
    }

    /// Combined confidence: how trustworthy is the spectral analysis?
    pub fn combined_confidence(&self) -> f64 {
        let mut score = 0.0;
        let mut count = 0.0;

        if let Some(bound) = self.scaling_law_bound {
            if !self.is_vacuous && bound.is_finite() {
                score += 0.8;
            } else {
                score += 0.2;
            }
            count += 1.0;
        }

        if let Some(angle) = self.davis_kahan_angle_bound {
            if angle < 0.5 {
                score += 0.9;
            } else if angle < 1.0 {
                score += 0.5;
            } else {
                score += 0.1;
            }
            count += 1.0;
        }

        if let Some(quality) = self.partition_quality_score {
            score += quality;
            count += 1.0;
        }

        if count > 0.0 {
            score / count
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_empty() {
        let summary = SpectralBoundSummary::from_certificates(None, None, None);
        assert!(summary.is_vacuous);
        assert!(!summary.is_informative());
    }

    #[test]
    fn test_combined_confidence_empty() {
        let summary = SpectralBoundSummary::from_certificates(None, None, None);
        assert!((summary.combined_confidence() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_summary_informative() {
        let summary = SpectralBoundSummary {
            scaling_law_bound: Some(5.0),
            is_vacuous: false,
            davis_kahan_angle_bound: Some(0.1),
            partition_quality_score: Some(0.8),
            spectral_gap: Some(0.5),
            condition_number: Some(10.0),
        };
        assert!(summary.is_informative());
        assert!(summary.combined_confidence() > 0.5);
    }

    #[test]
    fn test_summary_vacuous() {
        let summary = SpectralBoundSummary {
            scaling_law_bound: Some(1e15),
            is_vacuous: true,
            davis_kahan_angle_bound: None,
            partition_quality_score: None,
            spectral_gap: Some(1e-10),
            condition_number: Some(1e6),
        };
        assert!(!summary.is_informative());
    }
}
