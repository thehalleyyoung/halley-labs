//! # cascade-verify
//!
//! Verification pipeline, reporting, and CI/CD integration for the CascadeVerify
//! project. This crate orchestrates the end-to-end analysis flow: parsing
//! configuration files, building the Retry-Timeout Interaction Graph (RTIG),
//! running tiered cascade analysis, synthesising repairs, and producing
//! machine-readable reports (SARIF, JUnit) as well as human-friendly output.
//!
//! ## Modules
//!
//! - [`pipeline`] – Main verification pipeline that ties every stage together.
//! - [`sarif`] – SARIF 2.1.0 report generation for GitHub Advanced Security / GHAS.
//! - [`junit`] – JUnit XML report generation for CI test-result dashboards.
//! - [`reporter`] – Human-readable plain-text and Markdown reporting.
//! - [`cicd`] – CI/CD gating, annotation output, and diff-mode analysis.
//! - [`cache`] – Analysis result caching (in-memory LRU and filesystem).

pub mod cache;
pub mod cicd;
pub mod junit;
pub mod pipeline;
pub mod reporter;
pub mod sarif;

// Re-export the most commonly used public API surface.

pub use cache::{AnalysisCache, CacheConfig, CacheEntry, CacheKey, CacheStats};
pub use cicd::{CiCdIntegration, GatingAction, GatingPolicy, GatingRule, GitHubAnnotation};
pub use junit::{JUnitGenerator, JUnitTestReport, TestCase, TestStatus, TestSuite};
pub use pipeline::{
    AnalysisMode, OutputConfig, PipelineConfig, PipelineResult, PipelineStage, PipelineStats,
    VerificationPipeline,
};
pub use reporter::{ReportFormat, Reporter};
pub use sarif::{SarifGenerator, SarifReport};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_analysis_mode_variants() {
        let modes = vec![
            AnalysisMode::Quick,
            AnalysisMode::Standard,
            AnalysisMode::Deep,
        ];
        for mode in modes {
            let _ = format!("{:?}", mode);
        }
    }

    #[test]
    fn test_pipeline_stage_ordering() {
        let stages = vec![
            PipelineStage::Parse,
            PipelineStage::BuildGraph,
            PipelineStage::Analyze,
            PipelineStage::Repair,
            PipelineStage::Report,
        ];
        for (i, stage) in stages.iter().enumerate() {
            assert_eq!(*stage as usize, i);
        }
    }

    #[test]
    fn test_output_config_default() {
        let config = OutputConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_cache_stats_initial() {
        let stats = CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            size: 0,
        };
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_gating_action_variants() {
        let actions = vec![
            GatingAction::Pass,
            GatingAction::Warn,
            GatingAction::Block,
        ];
        for a in actions {
            let _ = format!("{:?}", a);
        }
    }

    #[test]
    fn test_gating_policy_default() {
        let policy = GatingPolicy::default();
        let _ = format!("{:?}", policy);
    }

    #[test]
    fn test_report_format_variants() {
        let formats = vec![
            ReportFormat::Text,
            ReportFormat::Json,
            ReportFormat::Sarif,
            ReportFormat::JUnit,
        ];
        for f in formats {
            let _ = format!("{:?}", f);
        }
    }

    #[test]
    fn test_test_status_variants() {
        let statuses = vec![
            TestStatus::Passed,
            TestStatus::Failed,
            TestStatus::Skipped,
        ];
        for s in statuses {
            let _ = format!("{:?}", s);
        }
    }

    #[test]
    fn test_junit_test_case() {
        let tc = TestCase {
            name: "test_retry_amplification".into(),
            class_name: "cascade.analysis".into(),
            time_seconds: 0.42,
            status: TestStatus::Passed,
            failure_message: None,
        };
        assert_eq!(tc.status, TestStatus::Passed);
        assert!(tc.failure_message.is_none());
    }

    #[test]
    fn test_junit_test_case_failed() {
        let tc = TestCase {
            name: "test_timeout_chain".into(),
            class_name: "cascade.analysis".into(),
            time_seconds: 1.2,
            status: TestStatus::Failed,
            failure_message: Some("Timeout chain violation detected".into()),
        };
        assert_eq!(tc.status, TestStatus::Failed);
        assert!(tc.failure_message.is_some());
    }

    #[test]
    fn test_github_annotation() {
        let ann = GitHubAnnotation {
            level: "warning".into(),
            file: "envoy.yaml".into(),
            line: 42,
            message: "High retry count".into(),
        };
        assert_eq!(ann.level, "warning");
        assert_eq!(ann.line, 42);
    }

    #[test]
    fn test_pipeline_stats_default() {
        let stats = PipelineStats::default();
        let _ = format!("{:?}", stats);
    }

    #[test]
    fn test_sarif_generator_reexport() {
        let gen = SarifGenerator::new();
        let _ = format!("{:?}", gen);
    }

    #[test]
    fn test_junit_generator_reexport() {
        let gen = JUnitGenerator::new();
        let _ = format!("{:?}", gen);
    }

    #[test]
    fn test_cicd_integration_reexport() {
        let ci = CiCdIntegration::new();
        let _ = format!("{:?}", ci);
    }
}
