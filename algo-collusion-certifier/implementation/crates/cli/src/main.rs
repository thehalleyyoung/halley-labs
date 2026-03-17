//! CollusionProof CLI — entry point.
//!
//! Parses command-line arguments, sets up logging, dispatches to the appropriate
//! subcommand handler, and manages global error handling.

mod commands;
mod config_loader;
mod evaluation;
mod logging;
mod output;
mod runner;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

use commands::{
    CollusionProofCli, Command, ConfigAction, OutputFormat,
};
use config_loader::{
    CliConfig, ConfigBuilder, ConfigProfile, generate_default_config, load_config,
    merge_cli_args, validate_config, write_default_config,
};
use evaluation::{
    Classification, EvalScenario, EvaluationRunner, get_builtin_scenarios,
};
use logging::{
    AuditEventType, AuditLog, VerbosityLevel, setup_logging,
};
use output::{
    CertificateView, DetectionResultView, EvaluationResultsView, ResultFormatter,
    ScenarioView, VerificationView, create_formatter,
};
use runner::{
    CheckpointingRunner, ParallelRunner, PipelineResult, PipelineRunner,
    run_simulation_only, scenarios_to_views,
};

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let cli = CollusionProofCli::parse();

    let verbosity = VerbosityLevel::from_flags(cli.quiet, cli.verbose);
    setup_logging(verbosity);

    log::debug!("CollusionProof CLI starting");
    log::debug!("Verbosity: {}", verbosity);

    match run(cli, verbosity) {
        Ok(()) => {
            log::debug!("CollusionProof CLI completed successfully");
            ExitCode::SUCCESS
        }
        Err(e) => {
            log::error!("Error: {:#}", e);
            // Print error chain for debugging
            if verbosity >= VerbosityLevel::Verbose {
                for (i, cause) in e.chain().enumerate().skip(1) {
                    log::error!("  Caused by [{}]: {}", i, cause);
                }
            }
            eprintln!("Error: {:#}", e);
            ExitCode::FAILURE
        }
    }
}

/// Central dispatch: load config and run the selected subcommand.
fn run(cli: CollusionProofCli, verbosity: VerbosityLevel) -> Result<()> {
    let use_color = !cli.no_color;
    let formatter = create_formatter(&cli.format, use_color);

    match cli.command {
        Command::Run(ref args) => cmd_run(args, &cli, verbosity, formatter.as_ref()),
        Command::Simulate(ref args) => cmd_simulate(args, &cli, verbosity, formatter.as_ref()),
        Command::Analyze(ref args) => cmd_analyze(args, &cli, verbosity, formatter.as_ref()),
        Command::Certify(ref args) => cmd_certify(args, &cli, verbosity, formatter.as_ref()),
        Command::Verify(ref args) => cmd_verify(args, &cli, verbosity, formatter.as_ref()),
        Command::Evaluate(ref args) => cmd_evaluate(args, &cli, verbosity, formatter.as_ref()),
        Command::Scenarios(ref args) => cmd_scenarios(args, &cli, verbosity, formatter.as_ref()),
        Command::Config(ref args) => cmd_config(args, &cli, verbosity),
    }
}

// ── Subcommand handlers ─────────────────────────────────────────────────────

/// `run` — Run the full certification pipeline on a scenario.
fn cmd_run(
    args: &commands::RunArgs,
    cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    let config_path = cli.config.as_ref().map(|p| p.to_string_lossy().to_string());
    let mut config = load_config(config_path.as_deref())?;
    merge_cli_args(&mut config, &args);
    validate_config(&config)?;

    let oracle_level = args.oracle_level.to_oracle_level();

    if args.all {
        // Run all scenarios
        let scenarios = get_builtin_scenarios();
        let ids: Vec<String> = scenarios.iter().map(|s| s.id.clone()).collect();

        if args.jobs != 1 {
            let parallel = ParallelRunner::new(
                config,
                oracle_level,
                args.seed,
                verbosity,
                args.jobs,
            );
            let results = parallel.run_scenarios(&ids);
            for result in results {
                match result {
                    Ok(r) => {
                        let view = r.to_detection_view();
                        println!("{}", formatter.format_detection(&view));
                    }
                    Err(e) => {
                        eprintln!("Scenario failed: {:#}", e);
                    }
                }
            }
        } else {
            for id in &ids {
                let mut runner = PipelineRunner::new(
                    config.clone(),
                    oracle_level,
                    args.seed,
                    verbosity,
                );
                match runner.run_scenario_by_id(id) {
                    Ok(r) => {
                        let view = r.to_detection_view();
                        println!("{}", formatter.format_detection(&view));
                        save_result_if_needed(&args, &r)?;
                    }
                    Err(e) => {
                        eprintln!("Scenario {} failed: {:#}", id, e);
                    }
                }
            }
        }
    } else if let Some(scenario_id) = &args.scenario {
        // Run single scenario
        let mut runner = if args.checkpoint {
            let inner = PipelineRunner::new(
                config,
                oracle_level,
                args.seed,
                verbosity,
            );
            let scenarios = get_builtin_scenarios();
            let scenario = scenarios
                .iter()
                .find(|s| s.id == *scenario_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown scenario: {}", scenario_id))?
                .clone();

            let mut ckpt = CheckpointingRunner::new(inner, args.checkpoint_dir.clone());
            let result = ckpt.run_scenario(&scenario)?;
            let view = result.to_detection_view();
            println!("{}", formatter.format_detection(&view));
            save_result_if_needed(&args, &result)?;
            return Ok(());
        } else {
            PipelineRunner::new(config, oracle_level, args.seed, verbosity)
        };

        let result = runner.run_scenario_by_id(scenario_id)?;
        let view = result.to_detection_view();
        println!("{}", formatter.format_detection(&view));

        if verbosity >= VerbosityLevel::Verbose {
            println!("\n{}", runner.timing_summary());
        }

        save_result_if_needed(&args, &result)?;
    }

    Ok(())
}

/// Save pipeline result to output directory if requested.
fn save_result_if_needed(args: &commands::RunArgs, result: &PipelineResult) -> Result<()> {
    if args.save_intermediates {
        std::fs::create_dir_all(&args.output_dir)?;
        let path = args
            .output_dir
            .join(format!("{}_result.json", result.scenario_id));
        let json = serde_json::to_string_pretty(result)?;
        std::fs::write(&path, json)?;
        log::info!("Saved result to: {}", path.display());
    }
    Ok(())
}

/// `simulate` — Run market simulation only.
fn cmd_simulate(
    args: &commands::SimulateArgs,
    _cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    _formatter: &dyn ResultFormatter,
) -> Result<()> {
    log::info!(
        "Running simulation: {} market, {} players, {} rounds",
        args.market_type,
        args.num_players,
        args.rounds,
    );

    let trajectory = run_simulation_only(args)?;

    // Save trajectory
    let json = shared_types::serialization::to_json(&trajectory)
        .map_err(|e| anyhow::anyhow!(e))?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("Failed to write trajectory to: {}", args.output.display()))?;

    if verbosity >= VerbosityLevel::Normal {
        println!("{}", output::format_trajectory_summary(&trajectory));
        println!("Trajectory saved to: {}", args.output.display());
    }

    Ok(())
}

/// `analyze` — Run statistical analysis on trajectory data.
fn cmd_analyze(
    args: &commands::AnalyzeArgs,
    cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    log::info!("Analyzing trajectory: {}", args.input.display());

    let content = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read trajectory: {}", args.input.display()))?;
    let trajectory: shared_types::PriceTrajectory = shared_types::serialization::from_json(&content)
        .map_err(|e| anyhow::anyhow!(e))?;

    log::info!(
        "Loaded trajectory: {} rounds, {} players",
        trajectory.len(),
        trajectory.num_players
    );

    // Run pipeline as a lightweight analysis
    let config = ConfigBuilder::new()
        .significance_level(args.alpha)
        .bootstrap_resamples(args.bootstrap_resamples)
        .seed(args.seed)
        .num_rounds(trajectory.len())
        .num_players(trajectory.num_players)
        .build();

    // Build a synthetic scenario for analysis
    let scenario = EvalScenario {
        id: "analysis".into(),
        name: "Analysis Input".into(),
        description: format!("Analysis of {}", args.input.display()),
        market_type: shared_types::MarketType::Bertrand,
        demand_system: shared_types::DemandSystem::Linear {
            max_quantity: 10.0,
            slope: 1.0,
        },
        num_players: trajectory.num_players,
        algorithm: "Unknown".into(),
        num_rounds: trajectory.len(),
        ground_truth: Classification::Inconclusive,
        difficulty: evaluation::ScenarioDifficulty::Medium,
    };

    let oracle_level = args.oracle_level.to_oracle_level();
    let mut runner = PipelineRunner::new(config, oracle_level, args.seed, verbosity);
    let result = runner.run_scenario(&scenario)?;

    let view = result.to_detection_view();
    println!("{}", formatter.format_detection(&view));

    // Save analysis results
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("Failed to write analysis results: {}", args.output.display()))?;
    log::info!("Analysis results saved to: {}", args.output.display());

    Ok(())
}

/// `certify` — Generate a certificate from analysis results.
fn cmd_certify(
    args: &commands::CertifyArgs,
    _cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    log::info!("Generating certificate from: {}", args.analysis_results.display());

    let content = std::fs::read_to_string(&args.analysis_results)
        .with_context(|| format!("Failed to read: {}", args.analysis_results.display()))?;
    let result: PipelineResult = serde_json::from_str(&content)
        .context("Failed to parse analysis results")?;

    // Build certificate view
    let cert_view = CertificateView {
        id: result.certificate_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        scenario_id: result.scenario_id.clone(),
        classification: result.classification.to_string(),
        confidence: result.confidence,
        collusion_premium: result.collusion_premium,
        timestamp: chrono::Utc::now().to_rfc3339(),
        hash: format!("{:016x}", hash_result(&result)),
        num_evidence: result.layer_results.len(),
        num_tests: result.layer_results.iter().map(|l| l.test_count).sum(),
    };

    println!("{}", formatter.format_certificate(&cert_view));

    let json = serde_json::to_string_pretty(&cert_view)?;
    std::fs::write(&args.output, &json)
        .with_context(|| format!("Failed to write certificate: {}", args.output.display()))?;
    log::info!("Certificate saved to: {}", args.output.display());

    Ok(())
}

/// Simple hash for a pipeline result.
fn hash_result(result: &PipelineResult) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    result.scenario_id.hash(&mut hasher);
    result.collusion_premium.to_bits().hash(&mut hasher);
    result.confidence.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// `verify` — Verify an existing certificate.
fn cmd_verify(
    args: &commands::VerifyArgs,
    _cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    log::info!("Verifying certificate: {}", args.certificate.display());

    let content = std::fs::read_to_string(&args.certificate)
        .with_context(|| format!("Failed to read: {}", args.certificate.display()))?;
    let cert: CertificateView = serde_json::from_str(&content)
        .context("Failed to parse certificate")?;

    let mut issues = Vec::new();

    // Validate hash
    let hash_valid = !cert.hash.is_empty();
    if !hash_valid {
        issues.push("Certificate hash is empty".into());
    }

    // Validate classification
    let valid_classifications = ["Collusive", "Competitive", "Inconclusive"];
    let class_valid = valid_classifications.contains(&cert.classification.as_str());
    if !class_valid {
        issues.push(format!("Invalid classification: {}", cert.classification));
    }

    // Validate confidence range
    if cert.confidence < 0.0 || cert.confidence > 1.0 {
        issues.push(format!("Confidence out of range: {}", cert.confidence));
    }

    // Validate premium range
    if cert.collusion_premium < 0.0 || cert.collusion_premium > 1.0 {
        issues.push(format!("Collusion premium out of range: {}", cert.collusion_premium));
    }

    // Validate evidence consistency
    let evidence_sufficient = cert.num_evidence > 0 || cert.classification == "Inconclusive";
    if !evidence_sufficient {
        issues.push("No evidence for non-inconclusive classification".into());
    }

    // Check evidence bundle if provided
    if let Some(bundle_path) = &args.evidence_bundle {
        let bundle_content = std::fs::read_to_string(bundle_path)
            .with_context(|| format!("Failed to read evidence: {}", bundle_path.display()))?;
        // Verify bundle is valid JSON
        if serde_json::from_str::<serde_json::Value>(&bundle_content).is_err() {
            issues.push("Evidence bundle is not valid JSON".into());
        }
    }

    if args.strict && !issues.is_empty() {
        // In strict mode, any issue is a failure
    }

    let valid = issues.is_empty();
    let view = VerificationView {
        valid,
        hash_valid,
        tests_consistent: class_valid,
        evidence_sufficient,
        issues,
    };

    println!("{}", formatter.format_verification(&view));

    if !valid {
        anyhow::bail!("Certificate verification failed");
    }

    Ok(())
}

/// `evaluate` — Run the evaluation benchmark suite.
fn cmd_evaluate(
    args: &commands::EvaluateArgs,
    cli: &CollusionProofCli,
    verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    let config_path = cli.config.as_ref().map(|p| p.to_string_lossy().to_string());
    let mut config = load_config(config_path.as_deref())?;
    config.jobs = args.jobs;

    let eval_runner = EvaluationRunner::new(config, verbosity);

    let results = match args.mode {
        commands::EvalModeArg::Smoke => eval_runner.run_smoke_evaluation()?,
        commands::EvalModeArg::Standard => eval_runner.run_standard_evaluation()?,
        commands::EvalModeArg::Full => eval_runner.run_full_evaluation()?,
    };

    println!("{}", formatter.format_evaluation(&results));

    // Save results
    std::fs::create_dir_all(&args.output_dir)?;
    let path = args.output_dir.join("evaluation_results.json");
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&path, &json)?;
    log::info!("Evaluation results saved to: {}", path.display());

    Ok(())
}

/// `scenarios` — List available scenarios.
fn cmd_scenarios(
    args: &commands::ScenariosArgs,
    _cli: &CollusionProofCli,
    _verbosity: VerbosityLevel,
    formatter: &dyn ResultFormatter,
) -> Result<()> {
    let mut scenarios = get_builtin_scenarios();

    // Apply filters
    if let Some(difficulty) = &args.difficulty {
        let diff_str = difficulty.to_string();
        scenarios.retain(|s| s.difficulty.to_string() == diff_str);
    }

    if let Some(market_type) = &args.market_type {
        let mt_str = market_type.to_string();
        scenarios.retain(|s| format!("{:?}", s.market_type) == mt_str);
    }

    if let Some(gt) = &args.ground_truth {
        let gt_str = gt.to_string();
        scenarios.retain(|s| s.ground_truth.to_string() == gt_str);
    }

    let views = scenarios_to_views(&scenarios);

    if args.detailed {
        for view in &views {
            println!("━━━ {} ━━━", view.id);
            println!("  Name:        {}", view.name);
            println!("  Description: {}", view.description);
            println!("  Market:      {}", view.market_type);
            println!("  Players:     {}", view.num_players);
            println!("  Algorithm:   {}", view.algorithm);
            println!("  Rounds:      {}", view.num_rounds);
            println!("  Truth:       {}", view.ground_truth);
            println!("  Difficulty:  {}", view.difficulty);
            println!();
        }
    } else {
        println!("{}", formatter.format_scenarios(&views));
    }

    Ok(())
}

/// `config` — Show or generate configuration.
fn cmd_config(
    args: &commands::ConfigArgs,
    cli: &CollusionProofCli,
    _verbosity: VerbosityLevel,
) -> Result<()> {
    match args.action {
        ConfigAction::Show => {
            let config_path = cli.config.as_ref().map(|p| p.to_string_lossy().to_string());
            let config = load_config(config_path.as_deref())?;
            let json = serde_json::to_string_pretty(&config)?;
            println!("{}", json);
        }
        ConfigAction::Generate => {
            let profile = args.profile.map(ConfigProfile::from);
            write_default_config(&args.output, profile)?;
            println!("Configuration written to: {}", args.output.display());
        }
        ConfigAction::Validate => {
            let config_path = cli.config.as_ref().map(|p| p.to_string_lossy().to_string());
            let config = load_config(config_path.as_deref())?;
            match validate_config(&config) {
                Ok(()) => println!("Configuration is valid."),
                Err(e) => {
                    eprintln!("Configuration validation failed: {:#}", e);
                    anyhow::bail!("Invalid configuration");
                }
            }
        }
    }
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_parse_run() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "run",
            "--scenario",
            "bertrand_qlearning_2p",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_evaluate_smoke() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "evaluate",
            "--mode",
            "smoke",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_config_show() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "config",
            "show",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_scenarios() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "scenarios",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_simulate() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "simulate",
            "--rounds",
            "100",
            "--num-players",
            "3",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_verify() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "verify",
            "--certificate",
            "cert.json",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_global_flags() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "-q",
            "--no-color",
            "--format",
            "json",
            "config",
            "show",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert!(cli.quiet);
        assert!(cli.no_color);
        assert_eq!(cli.format, OutputFormat::Json);
    }

    #[test]
    fn test_verbosity_from_flags() {
        assert_eq!(
            VerbosityLevel::from_flags(true, 0),
            VerbosityLevel::Quiet
        );
        assert_eq!(
            VerbosityLevel::from_flags(false, 0),
            VerbosityLevel::Normal
        );
        assert_eq!(
            VerbosityLevel::from_flags(false, 1),
            VerbosityLevel::Verbose
        );
        assert_eq!(
            VerbosityLevel::from_flags(false, 2),
            VerbosityLevel::Debug
        );
    }

    #[test]
    fn test_hash_result_deterministic() {
        let result = PipelineResult {
            scenario_id: "test".into(),
            classification: Classification::Collusive,
            confidence: 0.95,
            collusion_premium: 0.8,
            evidence_strength: shared_types::EvidenceStrength::Strong,
            oracle_level: shared_types::OracleAccessLevel::Layer0,
            layer_results: vec![],
            certificate_id: None,
            verification_passed: None,
            timing: std::collections::HashMap::new(),
            num_rounds: 100,
            num_players: 2,
        };
        let h1 = hash_result(&result);
        let h2 = hash_result(&result);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_pearson_ext() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let corr = evaluation::pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cli_rejects_invalid_alpha() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "run",
            "--scenario",
            "test",
            "--alpha",
            "1.5",
        ]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_rejects_invalid_rounds() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "run",
            "--scenario",
            "test",
            "--rounds",
            "5",
        ]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_rejects_invalid_players() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "simulate",
            "--num-players",
            "1",
        ]);
        assert!(cli.is_err());
    }

    #[test]
    fn test_cli_parse_run_all() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "run",
            "--all",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_analyze() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "analyze",
            "--input",
            "traj.json",
            "--alpha",
            "0.01",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_certify() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "certify",
            "--analysis-results",
            "results.json",
            "--output",
            "cert.json",
        ]);
        assert!(cli.is_ok());
    }
}
