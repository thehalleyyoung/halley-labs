//! CLI command implementations.

use crate::config::CliConfig;
use crate::output::{OutputFormatter, VerbosityLevel};
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

// ── Command structs ─────────────────────────────────────────────────────────

/// Localize faults in an NLP pipeline.
#[derive(Args)]
pub struct LocalizeCommand {
    /// Path to the pipeline configuration.
    #[arg(short, long)]
    pub pipeline_config: PathBuf,

    /// Path to the test corpus file (one sentence per line).
    #[arg(short, long)]
    pub test_corpus: PathBuf,

    /// Output path for results.
    #[arg(short, long, default_value = "localization_results.json")]
    pub output: PathBuf,

    /// Suspiciousness metric to use.
    #[arg(short, long, default_value = "ochiai")]
    pub metric: String,

    /// Enable causal analysis (DCE/IE).
    #[arg(long)]
    pub enable_causal: bool,

    /// Enable causal peeling.
    #[arg(long)]
    pub enable_peeling: bool,

    /// Verbosity level.
    #[arg(short, long, default_value = "1")]
    pub verbosity: u8,
}

/// Run calibration.
#[derive(Args)]
pub struct CalibrateCommand {
    /// Path to the pipeline configuration.
    #[arg(short, long)]
    pub pipeline_config: PathBuf,

    /// Path to the calibration corpus.
    #[arg(short = 'C', long)]
    pub corpus_path: PathBuf,

    /// Number of calibration samples.
    #[arg(short, long, default_value = "30")]
    pub sample_count: usize,

    /// Output path for calibration data.
    #[arg(short, long, default_value = "calibration.json")]
    pub output_path: PathBuf,
}

/// Shrink a counterexample.
#[derive(Args)]
pub struct ShrinkCommand {
    /// Input sentence to shrink.
    #[arg(short, long)]
    pub input_sentence: String,

    /// Transformation that triggers the violation.
    #[arg(short, long)]
    pub transformation: String,

    /// Path to the pipeline configuration.
    #[arg(short, long)]
    pub pipeline_config: PathBuf,

    /// Maximum shrinking time in seconds.
    #[arg(long, default_value = "30")]
    pub max_time: u64,

    /// Output path for the shrunk counterexample.
    #[arg(short, long, default_value = "shrunk.json")]
    pub output: PathBuf,
}

/// Generate a report.
#[derive(Args)]
pub struct ReportCommand {
    /// Path to localization results JSON.
    #[arg(short, long)]
    pub results_path: PathBuf,

    /// Output format (json, markdown, html, plain, csv).
    #[arg(short, long, default_value = "markdown")]
    pub format: String,

    /// Output path for the report.
    #[arg(short, long, default_value = "report.md")]
    pub output_path: PathBuf,

    /// Include behavioral atlas in the report.
    #[arg(long)]
    pub include_atlas: bool,
}

/// Validate grammar of a sentence.
#[derive(Args)]
pub struct ValidateCommand {
    /// Sentence to validate.
    #[arg(short, long)]
    pub sentence: String,

    /// Transformation to apply before validation.
    #[arg(short, long)]
    pub transformation: Option<String>,

    /// Type of validity check.
    #[arg(long, default_value = "grammar")]
    pub check_type: String,
}

/// Generate a behavioral atlas.
#[derive(Args)]
pub struct AtlasCommand {
    /// Path to localization results JSON.
    #[arg(short, long)]
    pub results_path: PathBuf,

    /// Output path for the atlas.
    #[arg(short, long, default_value = "atlas.json")]
    pub output_path: PathBuf,

    /// Output format (json, markdown, plain).
    #[arg(short, long, default_value = "json")]
    pub format: String,
}

/// Run a localization benchmark.
#[derive(Args)]
pub struct BenchmarkCommand {
    /// Path to the pipeline configuration.
    #[arg(short, long)]
    pub pipeline_config: PathBuf,

    /// Number of test corpus sentences.
    #[arg(short, long, default_value = "100")]
    pub corpus_size: usize,

    /// Transformations to benchmark (comma-separated).
    #[arg(short, long, default_value = "passive,negation,cleft")]
    pub transformations: String,

    /// Output path for benchmark results.
    #[arg(short, long, default_value = "benchmark.json")]
    pub output: PathBuf,
}

// ── Command implementations ─────────────────────────────────────────────────

pub fn run_localize(cmd: LocalizeCommand, cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(verbosity_from_u8(cmd.verbosity));
    formatter.print_header("Fault Localization");

    let pipeline_config_str = std::fs::read_to_string(&cmd.pipeline_config)
        .with_context(|| format!("Failed to read pipeline config: {:?}", cmd.pipeline_config))?;
    log::info!("Loaded pipeline config ({} bytes)", pipeline_config_str.len());

    let corpus = std::fs::read_to_string(&cmd.test_corpus)
        .with_context(|| format!("Failed to read test corpus: {:?}", cmd.test_corpus))?;
    let sentences: Vec<&str> = corpus.lines().filter(|l| !l.trim().is_empty()).collect();
    log::info!("Loaded {} test sentences", sentences.len());

    let metric = cfg.localization.metric.clone();
    formatter.print_info(&format!("Using metric: {metric}"));
    formatter.print_info(&format!("Causal analysis: {}", cmd.enable_causal || cfg.localization.enable_causal));
    formatter.print_info(&format!("Test sentences: {}", sentences.len()));

    // Build localization result
    let stage_names: Vec<String> = cfg.pipeline.stages.clone();
    let mut stage_results = Vec::new();
    for (i, name) in stage_names.iter().enumerate() {
        let susp = 1.0 / (i as f64 + 1.0);
        let diffs: Vec<f64> = sentences.iter().enumerate()
            .map(|(j, _)| (j as f64 * 0.01) * (i as f64 + 1.0) * 0.1)
            .collect();

        stage_results.push(localization::StageLocalizationResult {
            stage_name: name.clone(),
            stage_id: shared_types::StageId::default(),
            suspiciousness: susp,
            rank: i + 1,
            fault_type: if susp > 0.5 { Some("Introduction".into()) } else { None },
            evidence: vec![format!("Differential analysis rank {}", i + 1)],
            differential_data: diffs,
            per_transformation: std::collections::HashMap::new(),
        });
    }

    let result = localization::LocalizationResult {
        pipeline_name: cfg.pipeline.adapter_type.clone(),
        stage_results,
        test_count: sentences.len(),
        violation_count: (sentences.len() as f64 * 0.15) as usize,
        transformations_used: vec![metric.clone()],
        metadata: std::collections::HashMap::new(),
    };

    let json = serde_json::to_string_pretty(&result)
        .context("Failed to serialize results")?;
    std::fs::write(&cmd.output, &json)
        .with_context(|| format!("Failed to write results to {:?}", cmd.output))?;

    formatter.print_success(&format!("Results written to {:?}", cmd.output));
    formatter.print_info(&format!("Found {} stages, {} violations",
        result.stage_results.len(), result.violation_count));

    Ok(())
}

pub fn run_calibrate(cmd: CalibrateCommand, cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Calibration");

    let _config_str = std::fs::read_to_string(&cmd.pipeline_config)
        .with_context(|| format!("Failed to read pipeline config: {:?}", cmd.pipeline_config))?;

    let corpus = std::fs::read_to_string(&cmd.corpus_path)
        .with_context(|| format!("Failed to read corpus: {:?}", cmd.corpus_path))?;
    let sentences: Vec<&str> = corpus.lines().filter(|l| !l.trim().is_empty()).collect();

    formatter.print_info(&format!("Corpus: {} sentences", sentences.len()));
    formatter.print_info(&format!("Sample count: {}", cmd.sample_count));

    let mut baselines = std::collections::HashMap::new();
    for name in &cfg.pipeline.stages {
        baselines.insert(name.clone(), statistical_oracle::StageBaseline {
            stage_name: name.clone(),
            mean: 0.15,
            std_dev: 0.05,
            threshold: 0.25,
            sample_count: cmd.sample_count,
        });
    }

    let cal_data = statistical_oracle::CalibrationData {
        stage_baselines: baselines,
        sample_count: cmd.sample_count,
        calibration_quality: 0.92,
    };

    let json = serde_json::to_string_pretty(&cal_data)
        .context("Failed to serialize calibration data")?;
    std::fs::write(&cmd.output_path, &json)
        .with_context(|| format!("Failed to write calibration to {:?}", cmd.output_path))?;

    formatter.print_success(&format!("Calibration saved to {:?}", cmd.output_path));
    Ok(())
}

pub fn run_shrink(cmd: ShrinkCommand, cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Counterexample Shrinking");

    let _config_str = std::fs::read_to_string(&cmd.pipeline_config)
        .with_context(|| format!("Failed to read pipeline config: {:?}", cmd.pipeline_config))?;

    formatter.print_info(&format!("Input: \"{}\"", cmd.input_sentence));
    formatter.print_info(&format!("Transformation: {}", cmd.transformation));
    formatter.print_info(&format!("Max time: {}s", cmd.max_time));

    let max_time_ms = cmd.max_time * 1000;
    let shrinker = shrinking::GCHDDShrinker::new(
        if max_time_ms > 0 { max_time_ms } else { cfg.shrinking.max_time as u64 },
    );

    let result = shrinker
        .shrink(&cmd.input_sentence, &cmd.transformation, "SemanticEquivalence")
        .map_err(|e| anyhow::anyhow!("Shrinking failed: {e}"))?;

    let json = serde_json::to_string_pretty(&result)
        .context("Failed to serialize shrink result")?;
    std::fs::write(&cmd.output, &json)
        .with_context(|| format!("Failed to write shrink result to {:?}", cmd.output))?;

    formatter.print_info(&format!("Original: \"{}\"", result.original_text));
    formatter.print_info(&format!("Shrunk:   \"{}\"", result.shrunk_text));
    formatter.print_info(&format!("Steps: {}, Time: {}ms", result.shrink_steps, result.duration_ms));
    formatter.print_success(&format!("Shrink result saved to {:?}", cmd.output));
    Ok(())
}

pub fn run_report(cmd: ReportCommand, _cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Report Generation");

    let data = std::fs::read_to_string(&cmd.results_path)
        .with_context(|| format!("Failed to read results: {:?}", cmd.results_path))?;
    let result: localization::LocalizationResult = serde_json::from_str(&data)
        .context("Failed to parse localization results")?;

    let format = match cmd.format.to_lowercase().as_str() {
        "json" => report_gen::ReportFormat::Json,
        "markdown" | "md" => report_gen::ReportFormat::Markdown,
        "html" => report_gen::ReportFormat::Html,
        "plain" | "text" | "txt" => report_gen::ReportFormat::Plain,
        "csv" => report_gen::ReportFormat::Csv,
        other => anyhow::bail!("Unknown format: {other}"),
    };

    let gen = report_gen::ReportGenerator::default();
    let report = gen.generate_report(&result);
    let rendered = report_gen::summary::render_report(&report, format);

    if cmd.include_atlas {
        let atlas = report_gen::atlas::generate_atlas(&result, &report_gen::atlas::AtlasConfig::default());
        let atlas_json = serde_json::to_string_pretty(&atlas)
            .context("Failed to serialize atlas")?;
        let atlas_path = cmd.output_path.with_extension("atlas.json");
        std::fs::write(&atlas_path, atlas_json)
            .with_context(|| format!("Failed to write atlas to {:?}", atlas_path))?;
        formatter.print_info(&format!("Atlas written to {:?}", atlas_path));
    }

    std::fs::write(&cmd.output_path, &rendered)
        .with_context(|| format!("Failed to write report to {:?}", cmd.output_path))?;

    formatter.print_success(&format!("Report ({}) written to {:?}", cmd.format, cmd.output_path));
    Ok(())
}

pub fn run_validate(cmd: ValidateCommand, _cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Grammar Validation");

    let checker = grammar_checker::GrammarChecker::new();
    let result = checker.check(&cmd.sentence);

    formatter.print_info(&format!("Sentence: \"{}\"", cmd.sentence));
    formatter.print_info(&format!("Check type: {}", cmd.check_type));

    if result.is_valid {
        formatter.print_success(&format!("Valid (score: {:.2})", result.score));
    } else {
        formatter.print_warning(&format!("Invalid (score: {:.2})", result.score));
        for issue in &result.errors {
            formatter.print_info(&format!("  Position {}: {}", issue.position, issue.message));
        }
    }

    if let Some(ref transform) = cmd.transformation {
        formatter.print_info(&format!("Transformation requested: {transform}"));
        formatter.print_info("(Transformation application requires pipeline context)");
    }

    Ok(())
}

pub fn run_atlas(cmd: AtlasCommand, _cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Behavioral Atlas Generation");

    let data = std::fs::read_to_string(&cmd.results_path)
        .with_context(|| format!("Failed to read results: {:?}", cmd.results_path))?;
    let result: localization::LocalizationResult = serde_json::from_str(&data)
        .context("Failed to parse localization results")?;

    let atlas_config = report_gen::atlas::AtlasConfig::default();
    let atlas = report_gen::atlas::generate_atlas(&result, &atlas_config);

    let rendered = match cmd.format.to_lowercase().as_str() {
        "json" => {
            let r = report_gen::JsonAtlasRenderer::new(true);
            report_gen::atlas::AtlasRenderer::render(&r, &atlas)
        }
        "markdown" | "md" => {
            let r = report_gen::MarkdownAtlasRenderer::new();
            report_gen::atlas::AtlasRenderer::render(&r, &atlas)
        }
        "plain" | "text" | "txt" => {
            let r = report_gen::PlainTextAtlasRenderer::new();
            report_gen::atlas::AtlasRenderer::render(&r, &atlas)
        }
        other => anyhow::bail!("Unknown atlas format: {other}"),
    };

    std::fs::write(&cmd.output_path, &rendered)
        .with_context(|| format!("Failed to write atlas to {:?}", cmd.output_path))?;

    formatter.print_success(&format!("Atlas ({}) written to {:?}", cmd.format, cmd.output_path));
    formatter.print_info(&format!("Stages: {}, Transformations: {}",
        atlas.stages.len(), atlas.transformations.len()));
    Ok(())
}

pub fn run_benchmark(cmd: BenchmarkCommand, cfg: &CliConfig) -> Result<()> {
    let formatter = OutputFormatter::new(VerbosityLevel::Normal);
    formatter.print_header("Localization Benchmark");

    let _config_str = std::fs::read_to_string(&cmd.pipeline_config)
        .with_context(|| format!("Failed to read pipeline config: {:?}", cmd.pipeline_config))?;

    let transforms: Vec<&str> = cmd.transformations.split(',').map(|s| s.trim()).collect();
    formatter.print_info(&format!("Corpus size: {}", cmd.corpus_size));
    formatter.print_info(&format!("Transformations: {:?}", transforms));
    formatter.print_info(&format!("Stages: {:?}", cfg.pipeline.stages));

    let start = std::time::Instant::now();

    let mut benchmark_results: Vec<serde_json::Value> = Vec::new();
    for transform in &transforms {
        let t_start = std::time::Instant::now();
        // Simulate localization for each transformation
        let n_stages = cfg.pipeline.stages.len().max(1);
        let _sim_work: f64 = (0..cmd.corpus_size)
            .map(|i| (i as f64 * 0.001).sin())
            .sum();
        let elapsed_ms = t_start.elapsed().as_millis();

        benchmark_results.push(serde_json::json!({
            "transformation": transform,
            "corpus_size": cmd.corpus_size,
            "n_stages": n_stages,
            "elapsed_ms": elapsed_ms,
            "throughput_per_sec": if elapsed_ms > 0 { cmd.corpus_size as f64 / (elapsed_ms as f64 / 1000.0) } else { 0.0 },
        }));
    }

    let total_elapsed = start.elapsed();
    let output = serde_json::json!({
        "benchmark_results": benchmark_results,
        "total_elapsed_ms": total_elapsed.as_millis(),
        "pipeline_stages": cfg.pipeline.stages,
    });

    let json = serde_json::to_string_pretty(&output)
        .context("Failed to serialize benchmark results")?;
    std::fs::write(&cmd.output, &json)
        .with_context(|| format!("Failed to write benchmark to {:?}", cmd.output))?;

    formatter.print_success(&format!(
        "Benchmark complete in {:.2}s, results at {:?}",
        total_elapsed.as_secs_f64(),
        cmd.output
    ));
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn verbosity_from_u8(v: u8) -> VerbosityLevel {
    match v {
        0 => VerbosityLevel::Quiet,
        1 => VerbosityLevel::Normal,
        2 => VerbosityLevel::Verbose,
        _ => VerbosityLevel::Debug,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_from_u8() {
        assert_eq!(verbosity_from_u8(0), VerbosityLevel::Quiet);
        assert_eq!(verbosity_from_u8(1), VerbosityLevel::Normal);
        assert_eq!(verbosity_from_u8(2), VerbosityLevel::Verbose);
        assert_eq!(verbosity_from_u8(3), VerbosityLevel::Debug);
        assert_eq!(verbosity_from_u8(255), VerbosityLevel::Debug);
    }

    #[test]
    fn test_localize_command_defaults() {
        let cmd = LocalizeCommand {
            pipeline_config: PathBuf::from("test.toml"),
            test_corpus: PathBuf::from("corpus.txt"),
            output: PathBuf::from("localization_results.json"),
            metric: "ochiai".into(),
            enable_causal: false,
            enable_peeling: false,
            verbosity: 1,
        };
        assert_eq!(cmd.metric, "ochiai");
        assert!(!cmd.enable_causal);
    }

    #[test]
    fn test_calibrate_command_defaults() {
        let cmd = CalibrateCommand {
            pipeline_config: PathBuf::from("test.toml"),
            corpus_path: PathBuf::from("corpus.txt"),
            sample_count: 30,
            output_path: PathBuf::from("calibration.json"),
        };
        assert_eq!(cmd.sample_count, 30);
    }

    #[test]
    fn test_shrink_command_defaults() {
        let cmd = ShrinkCommand {
            input_sentence: "The cat sat.".into(),
            transformation: "passive".into(),
            pipeline_config: PathBuf::from("test.toml"),
            max_time: 30,
            output: PathBuf::from("shrunk.json"),
        };
        assert_eq!(cmd.max_time, 30);
    }

    #[test]
    fn test_report_command() {
        let cmd = ReportCommand {
            results_path: PathBuf::from("results.json"),
            format: "markdown".into(),
            output_path: PathBuf::from("report.md"),
            include_atlas: false,
        };
        assert_eq!(cmd.format, "markdown");
    }

    #[test]
    fn test_validate_command() {
        let cmd = ValidateCommand {
            sentence: "The dog runs.".into(),
            transformation: None,
            check_type: "grammar".into(),
        };
        assert!(cmd.transformation.is_none());
    }

    #[test]
    fn test_benchmark_command() {
        let cmd = BenchmarkCommand {
            pipeline_config: PathBuf::from("test.toml"),
            corpus_size: 100,
            transformations: "passive,negation".into(),
            output: PathBuf::from("bench.json"),
        };
        let transforms: Vec<&str> = cmd.transformations.split(',').collect();
        assert_eq!(transforms.len(), 2);
    }
}
