//! # cascade-analysis
//!
//! Two-tier cascade analysis engine for the CascadeVerify project.
//!
//! Provides fast graph-based risk detection (Tier 1), deep bounded model
//! checking (Tier 2), orchestration of the two tiers, incremental analysis
//! for configuration diffs, cascade classification, and statistics collection.

pub mod benchmark;
pub mod classification;
pub mod incremental;
pub mod orchestrator;
pub mod statistics;
pub mod tier1;
pub mod tier2;

pub use classification::{CascadeClassifier, CascadeType, Classification, SeverityScorer};
pub use incremental::{
    CachedResult, IncrementalAnalyzer, IncrementalResult, TopologyDiff,
};
pub use orchestrator::{
    AnalysisConfig, AnalysisMode, AnalysisOrchestrator, AnalysisResult,
};
pub use statistics::{
    AnalysisStatistics, HistogramBuilder, PerformanceMetrics, RiskSummary, TopologyStats,
};
pub use tier1::{
    AmplificationRisk, FanInRisk, Tier1Analyzer, Tier1Config, Tier1Result, TimeoutViolation,
};
pub use tier2::{MinimalFailureSetInfo, Tier2Analyzer, Tier2Config, Tier2Result};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_config_default() {
        let config = AnalysisConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_analysis_mode_variants() {
        let modes = vec![AnalysisMode::Fast, AnalysisMode::Full, AnalysisMode::Incremental];
        for mode in modes {
            let _ = format!("{:?}", mode);
        }
    }

    #[test]
    fn test_cascade_classifier_reexport() {
        let classifier = CascadeClassifier::new();
        let _ = format!("{:?}", classifier);
    }

    #[test]
    fn test_severity_scorer_reexport() {
        let scorer = SeverityScorer::new();
        let _ = format!("{:?}", scorer);
    }

    #[test]
    fn test_tier1_config_default() {
        let config = Tier1Config::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_tier2_config_default() {
        let config = Tier2Config::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_incremental_analyzer_reexport() {
        let analyzer = IncrementalAnalyzer::new();
        let _ = format!("{:?}", analyzer);
    }

    #[test]
    fn test_analysis_statistics_reexport() {
        let stats = AnalysisStatistics::default();
        let _ = format!("{:?}", stats);
    }

    #[test]
    fn test_classification_serde() {
        let c = Classification {
            cascade_type: CascadeType::RetryAmplification,
            severity: cascade_types::CascadeSeverity::High,
            confidence: 0.95,
        };
        let json = serde_json::to_string(&c).unwrap();
        let deser: Classification = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.confidence, 0.95);
    }

    #[test]
    fn test_histogram_builder_reexport() {
        let mut hb = HistogramBuilder::new(0.0, 100.0, 10);
        hb.add(42.0);
        hb.add(73.0);
        let hist = hb.build();
        assert_eq!(hist.bucket_count(), 10);
    }

    #[test]
    fn test_topology_diff_reexport() {
        let diff = TopologyDiff {
            added_services: vec!["new-svc".into()],
            removed_services: vec![],
            modified_edges: vec![],
        };
        assert_eq!(diff.added_services.len(), 1);
    }
}
