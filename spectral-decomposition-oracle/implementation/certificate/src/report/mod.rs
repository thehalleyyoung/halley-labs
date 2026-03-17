//! Report generation module.
//!
//! Combines all certificates for an instance into comprehensive reports,
//! cross-instance comparisons, and visualization data.

pub mod comparison;
pub mod generator;
pub mod visualization;

pub use comparison::ComparisonReport;
pub use generator::CertificateReport;
pub use visualization::VisualizationData;

use serde::{Deserialize, Serialize};

/// Report format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Text,
    Latex,
    Summary,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::Text => write!(f, "TEXT"),
            Self::Latex => write!(f, "LaTeX"),
            Self::Summary => write!(f, "SUMMARY"),
        }
    }
}

/// Decomposability tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum DecomposabilityTier {
    Easy,
    Medium,
    Hard,
    Intractable,
}

impl std::fmt::Display for DecomposabilityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Easy => write!(f, "EASY"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Hard => write!(f, "HARD"),
            Self::Intractable => write!(f, "INTRACTABLE"),
        }
    }
}

/// Classify an instance tier based on certificates.
pub fn classify_tier(
    l3_bound: Option<f64>,
    t2_vacuous: bool,
    futility_score: Option<f64>,
    actual_gap: Option<f64>,
) -> DecomposabilityTier {
    if let Some(gap) = actual_gap {
        if gap < 0.01 {
            return DecomposabilityTier::Easy;
        }
    }
    if let Some(l3) = l3_bound {
        if l3 < 1.0 {
            return DecomposabilityTier::Easy;
        }
        if l3 < 10.0 && !t2_vacuous {
            return DecomposabilityTier::Medium;
        }
    }
    if let Some(fs) = futility_score {
        if fs > 0.8 {
            return DecomposabilityTier::Intractable;
        }
    }
    if t2_vacuous {
        return DecomposabilityTier::Hard;
    }
    DecomposabilityTier::Medium
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_easy() {
        let tier = classify_tier(Some(0.5), false, None, Some(0.005));
        assert_eq!(tier, DecomposabilityTier::Easy);
    }

    #[test]
    fn test_classify_intractable() {
        let tier = classify_tier(Some(100.0), true, Some(0.9), None);
        assert_eq!(tier, DecomposabilityTier::Intractable);
    }

    #[test]
    fn test_classify_hard_vacuous() {
        let tier = classify_tier(Some(50.0), true, Some(0.5), None);
        assert_eq!(tier, DecomposabilityTier::Hard);
    }

    #[test]
    fn test_tier_ordering() {
        assert!(DecomposabilityTier::Easy < DecomposabilityTier::Medium);
        assert!(DecomposabilityTier::Medium < DecomposabilityTier::Hard);
        assert!(DecomposabilityTier::Hard < DecomposabilityTier::Intractable);
    }

    #[test]
    fn test_report_format_display() {
        assert_eq!(format!("{}", ReportFormat::Json), "JSON");
        assert_eq!(format!("{}", ReportFormat::Latex), "LaTeX");
    }
}
