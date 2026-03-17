//! # cascade-cli
//!
//! Command-line interface for the CascadeVerify project.
//! Provides verify, repair, check, analyze, diff, report, and benchmark
//! subcommands for detecting retry-amplification cascades in microservice
//! configuration files.

pub mod commands;
pub mod config;
pub mod handlers;
pub mod output;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::commands::*;
    use super::config::*;
    use super::output::*;

    // -- OutputFormat -------------------------------------------------------

    #[test]
    fn test_output_format_variants() {
        let formats = [
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::Sarif,
            OutputFormat::JUnit,
            OutputFormat::Markdown,
        ];
        for f in formats {
            let _ = format!("{:?}", f);
        }
    }

    // -- AnalysisTier -------------------------------------------------------

    #[test]
    fn test_analysis_tier_variants() {
        let tiers = [AnalysisTier::Tier1, AnalysisTier::Tier2, AnalysisTier::Both];
        for t in tiers {
            let _ = format!("{:?}", t);
        }
    }

    // -- FailOnLevel --------------------------------------------------------

    #[test]
    fn test_fail_on_level_variants() {
        let levels = [
            FailOnLevel::Error,
            FailOnLevel::Warning,
            FailOnLevel::Info,
            FailOnLevel::Never,
        ];
        for l in levels {
            let _ = format!("{:?}", l);
        }
    }

    // -- BenchmarkTopology --------------------------------------------------

    #[test]
    fn test_benchmark_topology_variants() {
        let topos = [
            BenchmarkTopology::Chain,
            BenchmarkTopology::Diamond,
            BenchmarkTopology::Star,
            BenchmarkTopology::Mesh,
            BenchmarkTopology::All,
        ];
        for t in topos {
            let _ = format!("{:?}", t);
        }
    }

    // -- CliOutput ----------------------------------------------------------

    #[test]
    fn test_cli_output_reexport() {
        let _ = format!("{:?}", CliOutput::default());
    }

    // -- VerifyArgs validation ----------------------------------------------

    #[test]
    fn test_verify_args_validation_empty_paths() {
        let args = VerifyArgs {
            paths: vec![],
            output: OutputArgs::default(),
            mode: ModeArgs::default(),
            policy: PolicyArgs::default(),
        };
        let result = args.validate();
        assert!(result.is_err());
    }

    // -- OutputArgs ---------------------------------------------------------

    #[test]
    fn test_output_args_default() {
        let args = OutputArgs::default();
        assert_eq!(args.format, OutputFormat::Text);
    }

    // -- ModeArgs -----------------------------------------------------------

    #[test]
    fn test_mode_args_default() {
        let args = ModeArgs::default();
        assert_eq!(args.tier, AnalysisTier::Both);
    }

    // -- PolicyArgs ---------------------------------------------------------

    #[test]
    fn test_policy_args_default() {
        let args = PolicyArgs::default();
        assert_eq!(args.fail_on, FailOnLevel::Error);
    }

    // -- FindingSummary -----------------------------------------------------

    #[test]
    fn test_finding_summary_empty() {
        let summary = FindingSummary::default();
        assert_eq!(summary.total(), 0);
    }

    #[test]
    fn test_finding_summary_counts() {
        let summary = FindingSummary {
            critical: 1,
            error: 2,
            warning: 3,
            info: 4,
        };
        assert_eq!(summary.total(), 10);
    }

    // -- AnalysisSummary ----------------------------------------------------

    #[test]
    fn test_analysis_summary_default() {
        let summary = AnalysisSummary::default();
        assert_eq!(summary.total_files, 0);
        assert_eq!(summary.findings.total(), 0);
    }

    // -- RepairSummary ------------------------------------------------------

    #[test]
    fn test_repair_summary_default() {
        let summary = RepairSummary::default();
        assert_eq!(summary.actions_count, 0);
    }

    // -- BenchmarkResult ----------------------------------------------------

    #[test]
    fn test_benchmark_result_default() {
        let result = BenchmarkResult::default();
        assert_eq!(result.topology, "");
        assert_eq!(result.service_count, 0);
    }

    // -- ProgressDisplay ----------------------------------------------------

    #[test]
    fn test_progress_display_new() {
        let pd = ProgressDisplay::new(false);
        let _ = format!("{:?}", pd);
    }
}
