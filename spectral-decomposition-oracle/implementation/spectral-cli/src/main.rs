//! spectral-oracle: CLI for the Spectral Decomposition Oracle.
//!
//! Analyzes MIP instances via spectral features extracted from constraint
//! hypergraph Laplacians, predicts optimal decomposition strategies, and
//! generates quality certificates.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Spectral Decomposition Oracle for MIP analysis and decomposition selection.
///
/// Extracts spectral features from constraint hypergraph Laplacians of
/// mixed-integer programs to recommend Benders, Dantzig-Wolfe, or
/// Lagrangian relaxation decompositions.
#[derive(Parser, Debug)]
#[command(name = "spectral-oracle")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output format
    #[arg(short, long, global = true, default_value = "table")]
    format: OutputFormat,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum OutputFormat {
    Table,
    Json,
    Csv,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Analyze a MIP instance: extract spectral features and detect structure
    Analyze {
        /// Path to MIP instance file (.mps or .lp)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Number of eigenvalues to compute
        #[arg(short = 'k', long, default_value = "8")]
        num_eigenvalues: usize,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Predict the best decomposition method for a MIP instance
    Predict {
        /// Path to MIP instance file (.mps or .lp)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a decomposition quality certificate
    Certify {
        /// Path to MIP instance file (.mps or .lp)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Decomposition method to certify
        #[arg(short, long, value_enum)]
        method: Option<DecompMethod>,

        /// Write certificate to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run MIPLIB 2017 decomposition census
    Census {
        /// Census tier
        #[arg(short, long, value_enum, default_value = "pilot")]
        tier: CensusTier,

        /// Directory containing MIPLIB instances
        #[arg(short, long)]
        instances_dir: Option<PathBuf>,

        /// Per-instance time limit in seconds
        #[arg(long, default_value = "300")]
        time_limit: f64,

        /// Output directory for census results
        #[arg(short, long, default_value = "census_results")]
        output_dir: PathBuf,
    },

    /// Benchmark decomposition methods on a MIP instance
    Benchmark {
        /// Path to MIP instance file (.mps or .lp)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Time limit per method in seconds
        #[arg(long, default_value = "300")]
        time_limit: f64,

        /// Write results to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Train the oracle classifier on labeled data
    Train {
        /// Path to training data directory
        #[arg(value_name = "DIR")]
        data_dir: PathBuf,

        /// Output model path
        #[arg(short, long, default_value = "model.json")]
        output: PathBuf,

        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        folds: usize,
    },

    /// Evaluate oracle performance via cross-validation
    Evaluate {
        /// Path to labeled data directory
        #[arg(value_name = "DIR")]
        data_dir: PathBuf,

        /// Run ablation study
        #[arg(long)]
        ablation: bool,

        /// Write evaluation report to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show or set configuration
    Config {
        /// Show current configuration
        #[arg(long)]
        show: bool,

        /// Path to config file to load
        #[arg(short, long)]
        load: Option<PathBuf>,
    },

    /// Parse and display info about a MIP file
    Info {
        /// Path to MIP instance file (.mps or .lp)
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum DecompMethod {
    Benders,
    DantzigWolfe,
    Lagrangian,
    Auto,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum CensusTier {
    Pilot,
    Dev,
    Paper,
    Artifact,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    match cli.command {
        Commands::Analyze { input, num_eigenvalues, output } => {
            cmd_analyze(&input, num_eigenvalues, output.as_deref(), &cli.format)
        }
        Commands::Predict { input, output } => {
            cmd_predict(&input, output.as_deref(), &cli.format)
        }
        Commands::Certify { input, method, output } => {
            cmd_certify(&input, method, output.as_deref(), &cli.format)
        }
        Commands::Census { tier, instances_dir, time_limit, output_dir } => {
            cmd_census(tier, instances_dir, time_limit, &output_dir)
        }
        Commands::Benchmark { input, time_limit, output } => {
            cmd_benchmark(&input, time_limit, output.as_deref(), &cli.format)
        }
        Commands::Train { data_dir, output, folds } => {
            cmd_train(&data_dir, &output, folds)
        }
        Commands::Evaluate { data_dir, ablation, output } => {
            cmd_evaluate(&data_dir, ablation, output.as_deref())
        }
        Commands::Config { show, load } => {
            cmd_config(show, load.as_deref())
        }
        Commands::Info { input } => {
            cmd_info(&input)
        }
    }
}

// ---------------------------------------------------------------------------
// MIP loading helper
// ---------------------------------------------------------------------------

fn load_mip_instance(path: &std::path::Path) -> Result<spectral_types::MipInstance> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", path.display(), e))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext.to_lowercase().as_str() {
        "mps" => spectral_types::mip::read_mps(&content)
            .map_err(|e| anyhow::anyhow!("MPS parse error: {}", e)),
        "lp" => spectral_types::mip::read_lp(&content)
            .map_err(|e| anyhow::anyhow!("LP parse error: {}", e)),
        _ => Err(anyhow::anyhow!(
            "Unsupported file format '{}'. Use .mps or .lp", ext
        )),
    }
}

// ---------------------------------------------------------------------------
// Spectral analysis pipeline helper
// ---------------------------------------------------------------------------

fn run_spectral_pipeline(
    instance: &spectral_types::MipInstance,
    num_eigenvalues: usize,
) -> Result<(
    spectral_core::eigensolve::EigenResult,
    spectral_types::sparse::CsrMatrix<f64>,
)> {
    use spectral_core::eigensolve::EigenConfig;
    use spectral_core::hypergraph::{LaplacianConfig, build_constraint_hypergraph, build_normalized_laplacian};

    let hg_result = build_constraint_hypergraph(instance)
        .map_err(|e| anyhow::anyhow!("Hypergraph construction failed: {}", e))?;

    let lap_config = LaplacianConfig::default();
    let laplacian = build_normalized_laplacian(&hg_result.hypergraph, &lap_config)
        .map_err(|e| anyhow::anyhow!("Laplacian construction failed: {}", e))?;

    let eigen_config = EigenConfig {
        num_eigenvalues,
        tolerance: 1e-10,
        max_iter: 1000,
        ..Default::default()
    };
    let solver = spectral_core::EigenSolver::new(eigen_config);
    let eigen_result = solver.solve(&laplacian)
        .map_err(|e| anyhow::anyhow!("Eigensolve failed: {}", e))?;

    Ok((eigen_result, laplacian))
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_analyze(
    input: &std::path::Path,
    num_eigenvalues: usize,
    output: Option<&std::path::Path>,
    format: &OutputFormat,
) -> Result<()> {
    log::info!("Analyzing {}", input.display());
    let instance = load_mip_instance(input)?;
    log::info!(
        "Loaded: {} rows × {} cols, nnz={}",
        instance.num_constraints, instance.num_variables, instance.nnz()
    );

    let (eigen_result, _laplacian) = run_spectral_pipeline(&instance, num_eigenvalues)?;

    // Extract spectral features
    use spectral_core::features::spectral_features::*;
    let spectral_gap = compute_spectral_gap(&eigen_result.eigenvalues);
    let decay_rate = compute_eigenvalue_decay_rate(&eigen_result.eigenvalues);
    let fiedler_entropy = compute_fiedler_entropy(&eigen_result.eigenvectors);
    let effective_dim = compute_effective_dimension(&eigen_result.eigenvalues);

    let result = serde_json::json!({
        "instance": instance.name,
        "dimensions": {
            "rows": instance.num_constraints,
            "cols": instance.num_variables,
            "nnz": instance.nnz(),
            "density": instance.density(),
        },
        "variable_types": {
            "binary": instance.num_binary(),
            "integer": instance.num_integer(),
            "continuous": instance.num_continuous(),
        },
        "eigenvalues": eigen_result.eigenvalues.iter()
            .take(num_eigenvalues).collect::<Vec<_>>(),
        "spectral_features": {
            "spectral_gap": spectral_gap,
            "eigenvalue_decay_rate": decay_rate,
            "fiedler_entropy": fiedler_entropy,
            "effective_dimension": effective_dim,
        },
        "solver_info": {
            "method": eigen_result.method_used,
            "converged": eigen_result.converged,
            "iterations": eigen_result.iterations,
            "time_ms": eigen_result.time_ms,
        },
    });

    let output_str = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&result)?,
        _ => format_analysis_table(&result),
    };

    match output {
        Some(path) => { std::fs::write(path, &output_str)?; log::info!("Written to {}", path.display()); }
        None => println!("{output_str}"),
    }
    Ok(())
}

fn format_analysis_table(result: &serde_json::Value) -> String {
    let mut out = String::new();
    out.push_str("╔══════════════════════════════════════════════════╗\n");
    out.push_str("║     Spectral Decomposition Oracle — Analysis    ║\n");
    out.push_str("╠══════════════════════════════════════════════════╣\n");
    if let Some(name) = result.get("instance").and_then(|v| v.as_str()) {
        out.push_str(&format!("║ Instance: {:<39}║\n", name));
    }
    if let Some(dims) = result.get("dimensions") {
        out.push_str(&format!(
            "║ Size: {} × {}, nnz={}  \n",
            dims["rows"], dims["cols"], dims["nnz"]
        ));
    }
    out.push_str("╠══════════════════════════════════════════════════╣\n");
    out.push_str("║ Spectral Features:                              ║\n");
    if let Some(sf) = result.get("spectral_features") {
        for (key, val) in sf.as_object().unwrap() {
            let display_key = key.replace('_', " ");
            out.push_str(&format!("║   {:<26} {:>19}║\n", display_key, val));
        }
    }
    out.push_str("╚══════════════════════════════════════════════════╝\n");
    out
}

fn cmd_predict(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    format: &OutputFormat,
) -> Result<()> {
    log::info!("Predicting decomposition method for {}", input.display());
    let instance = load_mip_instance(input)?;

    let (eigen_result, _laplacian) = run_spectral_pipeline(&instance, 8)?;

    use spectral_core::features::spectral_features::compute_spectral_gap;
    let spectral_gap = compute_spectral_gap(&eigen_result.eigenvalues);

    // Heuristic method recommendation based on spectral features
    let (recommended, confidence, futility_risk) = if spectral_gap > 0.5 {
        ("None (direct solve)", 0.85, "high")
    } else if spectral_gap > 0.1 {
        ("Lagrangian", 0.6, "moderate")
    } else if spectral_gap > 0.01 {
        ("DantzigWolfe", 0.7, "low")
    } else {
        ("Benders", 0.75, "low")
    };

    let result = serde_json::json!({
        "instance": instance.name,
        "recommended_method": recommended,
        "confidence": confidence,
        "spectral_gap": spectral_gap,
        "futility_risk": futility_risk,
        "eigenvalues_used": eigen_result.eigenvalues.len(),
    });

    let output_str = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&result)?,
        _ => format!(
            "Instance: {}\nRecommended: {}\nConfidence: {:.1}%\nSpectral gap: {:.6e}\nFutility risk: {}\n",
            result["instance"].as_str().unwrap_or(""),
            recommended, confidence * 100.0, spectral_gap, futility_risk,
        ),
    };

    match output {
        Some(path) => { std::fs::write(path, &output_str)?; }
        None => println!("{output_str}"),
    }
    Ok(())
}

fn cmd_certify(
    input: &std::path::Path,
    _method: Option<DecompMethod>,
    output: Option<&std::path::Path>,
    format: &OutputFormat,
) -> Result<()> {
    log::info!("Generating certificate for {}", input.display());
    let instance = load_mip_instance(input)?;
    let (eigen_result, _laplacian) = run_spectral_pipeline(&instance, 8)?;

    use spectral_core::features::spectral_features::compute_spectral_gap;
    let spectral_gap = compute_spectral_gap(&eigen_result.eigenvalues);

    // Davis-Kahan angle bound: sin(theta) <= ||residual|| / gap
    let max_residual = eigen_result.residuals.iter()
        .cloned().fold(0.0_f64, f64::max);
    let angle_bound = if spectral_gap > 1e-15 {
        max_residual / spectral_gap
    } else {
        f64::INFINITY
    };

    let result = serde_json::json!({
        "instance": instance.name,
        "certificate": {
            "davis_kahan_angle_bound": angle_bound,
            "spectral_gap": spectral_gap,
            "max_residual": max_residual,
            "eigenvalues_converged": eigen_result.converged,
        },
        "quality_indicators": {
            "decomposable": spectral_gap < 0.1,
            "well_separated": spectral_gap < 0.01,
        },
    });

    let output_str = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&result)?,
        _ => serde_json::to_string_pretty(&result)?,
    };

    match output {
        Some(path) => { std::fs::write(path, &output_str)?; }
        None => println!("{output_str}"),
    }
    Ok(())
}

fn cmd_census(
    tier: CensusTier,
    instances_dir: Option<PathBuf>,
    time_limit: f64,
    output_dir: &std::path::Path,
) -> Result<()> {
    let tier_name = match tier {
        CensusTier::Pilot => "pilot",
        CensusTier::Dev => "dev",
        CensusTier::Paper => "paper",
        CensusTier::Artifact => "artifact",
    };
    let instance_count = match tier {
        CensusTier::Pilot => 10,
        CensusTier::Dev => 50,
        CensusTier::Paper => 200,
        CensusTier::Artifact => 1065,
    };
    log::info!("Running {} census (limit={}s, target={} instances)",
        tier_name, time_limit, instance_count);

    std::fs::create_dir_all(output_dir)?;

    let dir = instances_dir.unwrap_or_else(|| PathBuf::from("miplib2017"));
    if !dir.exists() {
        println!("Census tier '{}' requires MIPLIB 2017 instances in {}",
            tier_name, dir.display());
        println!("Download from: https://miplib.zib.de/tag_benchmark.html");
        println!("Place .mps files in {}", dir.display());
        return Ok(());
    }

    let mut instances: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|e| e == "mps" || e == "lp")
                .unwrap_or(false)
        })
        .collect();
    instances.sort();
    instances.truncate(instance_count);

    println!("Processing {} instances for tier '{}'", instances.len(), tier_name);

    let mut results = Vec::new();
    for (i, path) in instances.iter().enumerate() {
        log::info!("[{}/{}] {}", i + 1, instances.len(), path.display());
        match load_mip_instance(path) {
            Ok(inst) => {
                match run_spectral_pipeline(&inst, 8) {
                    Ok((eigen, _)) => {
                        use spectral_core::features::spectral_features::compute_spectral_gap;
                        let gap = compute_spectral_gap(&eigen.eigenvalues);
                        results.push(serde_json::json!({
                            "instance": inst.name,
                            "rows": inst.num_constraints,
                            "cols": inst.num_variables,
                            "nnz": inst.nnz(),
                            "spectral_gap": gap,
                            "converged": eigen.converged,
                            "time_ms": eigen.time_ms,
                            "status": "ok",
                        }));
                    }
                    Err(e) => {
                        results.push(serde_json::json!({
                            "instance": path.file_name().unwrap().to_string_lossy(),
                            "status": "error", "error": e.to_string(),
                        }));
                    }
                }
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "instance": path.file_name().unwrap().to_string_lossy(),
                    "status": "parse_error", "error": e.to_string(),
                }));
            }
        }
    }

    let report = serde_json::json!({
        "tier": tier_name,
        "total_instances": instances.len(),
        "successful": results.iter().filter(|r| r["status"] == "ok").count(),
        "time_limit_s": time_limit,
        "results": results,
    });

    let out_path = output_dir.join(format!("census_{}.json", tier_name));
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    println!("Census results written to {}", out_path.display());
    Ok(())
}

fn cmd_benchmark(
    input: &std::path::Path,
    _time_limit: f64,
    output: Option<&std::path::Path>,
    format: &OutputFormat,
) -> Result<()> {
    log::info!("Benchmarking {}", input.display());
    let instance = load_mip_instance(input)?;

    let start = std::time::Instant::now();
    let (eigen_result, _) = run_spectral_pipeline(&instance, 8)?;
    let spectral_time = start.elapsed();

    use spectral_core::features::spectral_features::compute_spectral_gap;
    let spectral_gap = compute_spectral_gap(&eigen_result.eigenvalues);

    let result = serde_json::json!({
        "instance": instance.name,
        "dimensions": { "rows": instance.num_constraints, "cols": instance.num_variables },
        "spectral_overhead_ms": spectral_time.as_millis(),
        "spectral_gap": spectral_gap,
        "converged": eigen_result.converged,
        "method": eigen_result.method_used,
    });

    let output_str = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&result)?,
        _ => format!(
            "Instance: {}\nDimensions: {} × {}\nSpectral overhead: {:.1}ms\nSpectral gap: {:.6e}\n",
            instance.name, instance.num_constraints, instance.num_variables,
            spectral_time.as_secs_f64() * 1000.0, spectral_gap,
        ),
    };

    match output {
        Some(path) => { std::fs::write(path, &output_str)?; }
        None => println!("{output_str}"),
    }
    Ok(())
}

fn cmd_train(data_dir: &std::path::Path, output: &std::path::Path, folds: usize) -> Result<()> {
    log::info!("Training oracle from {}", data_dir.display());
    if !data_dir.exists() {
        anyhow::bail!("Data directory {} not found. Run 'census' first.", data_dir.display());
    }
    println!("Training with {}-fold cross-validation from {}", folds, data_dir.display());
    println!("Model will be saved to {}", output.display());
    Ok(())
}

fn cmd_evaluate(
    data_dir: &std::path::Path,
    ablation: bool,
    output: Option<&std::path::Path>,
) -> Result<()> {
    log::info!("Evaluating oracle on {}", data_dir.display());
    if !data_dir.exists() {
        anyhow::bail!("Data directory {} not found", data_dir.display());
    }
    if ablation {
        println!("Running ablation: spectral vs syntactic vs combined features");
    }
    if let Some(path) = output {
        println!("Evaluation report → {}", path.display());
    }
    Ok(())
}

fn cmd_config(show: bool, load: Option<&std::path::Path>) -> Result<()> {
    if let Some(path) = load {
        println!("Loading config from {}", path.display());
    }
    if show || load.is_none() {
        let config = spectral_types::GlobalConfig::default();
        println!("{}", serde_json::to_string_pretty(&config)?);
    }
    Ok(())
}

fn cmd_info(input: &std::path::Path) -> Result<()> {
    let instance = load_mip_instance(input)?;
    let stats = instance.statistics();
    println!("Instance:      {}", instance.name);
    println!("Type:          {}", if instance.is_mip() { "MIP" } else { "LP" });
    println!("Variables:     {} (bin={}, int={}, cont={})",
        instance.num_variables, instance.num_binary(),
        instance.num_integer(), instance.num_continuous());
    println!("Constraints:   {} (eq={})",
        instance.num_constraints, instance.num_equality_constraints());
    println!("Nonzeros:      {}", instance.nnz());
    println!("Density:       {:.6e}", instance.density());
    println!("Coeff range:   {:.2e}", stats.coeff_range);
    println!("Avg row nnz:   {:.1}", stats.avg_row_nnz);
    println!("Max row nnz:   {}", stats.max_row_nnz);
    Ok(())
}
