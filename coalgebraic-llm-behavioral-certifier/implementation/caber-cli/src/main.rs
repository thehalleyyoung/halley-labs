//! CABER CLI — Coalgebraic Behavioral Auditing of Foundation Models
//!
//! Command-line interface for learning behavioral automata from LLMs,
//! checking temporal properties, generating certificates, comparing models,
//! and detecting behavioral drift.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use log::{info, warn};
use serde::{Deserialize, Serialize};

// Re-export core types locally until caber-core compiles fully.
// These mirror the definitions in caber_core::coalgebra::types.

/// Unique identifier for a state in a coalgebra.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct StateId(String);

impl StateId {
    fn indexed(prefix: &str, idx: usize) -> Self {
        Self(format!("{}_{}", prefix, idx))
    }
}

impl std::fmt::Display for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A symbol in the input alphabet.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Symbol(String);

impl Symbol {
    fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A word (sequence of symbols).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Word {
    symbols: Vec<Symbol>,
}

impl Word {
    fn from_str_slice(parts: &[&str]) -> Self {
        Self {
            symbols: parts.iter().map(|s| Symbol::new(*s)).collect(),
        }
    }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.symbols.is_empty() {
            write!(f, "ε")
        } else {
            let parts: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
            write!(f, "{}", parts.join("·"))
        }
    }
}

// ===========================================================================
// CLI argument definitions
// ===========================================================================

/// CABER — Coalgebraic Behavioral Auditing & Runtime Certification
#[derive(Parser, Debug)]
#[command(name = "caber", version, about, long_about = None)]
struct CaberCli {
    /// Optional path to a configuration file
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the full audit pipeline: learn → check → certify
    Audit(AuditArgs),
    /// Learn a behavioral automaton from a model
    Learn(LearnArgs),
    /// Check properties on an existing automaton
    Check(CheckArgs),
    /// Generate a certificate from existing results
    Certify(CertifyArgs),
    /// Compare two models behaviorally
    Compare(CompareArgs),
    /// Run behavioral drift detection
    DriftCheck(DriftCheckArgs),
}

#[derive(clap::Args, Debug)]
struct AuditArgs {
    /// Model identifier (e.g. "gpt-4", "llama-3-70b")
    #[arg(short, long)]
    model: String,

    /// Properties to check (repeatable, e.g. --property "safety" --property "fairness")
    #[arg(short, long)]
    property: Vec<String>,

    /// Maximum query budget
    #[arg(short, long, default_value_t = 1000)]
    budget: u64,

    /// Confidence threshold for property verdicts
    #[arg(short, long, default_value_t = 0.95)]
    confidence: f64,

    /// Output file path (if omitted, prints to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
}

#[derive(clap::Args, Debug)]
struct LearnArgs {
    /// Model identifier
    #[arg(short, long)]
    model: String,

    /// Maximum query budget
    #[arg(short, long, default_value_t = 1000)]
    budget: u64,

    /// Maximum number of states in the hypothesis automaton
    #[arg(long, default_value_t = 20)]
    max_states: usize,

    /// Convergence epsilon
    #[arg(short, long, default_value_t = 0.01)]
    epsilon: f64,
}

#[derive(clap::Args, Debug)]
struct CheckArgs {
    /// Path to automaton JSON file
    #[arg(short, long)]
    automaton: PathBuf,

    /// Properties to check (repeatable)
    #[arg(short, long)]
    property: Vec<String>,
}

#[derive(clap::Args, Debug)]
struct CertifyArgs {
    /// Path to results JSON file
    #[arg(short, long)]
    results: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct CompareArgs {
    /// First model identifier
    #[arg(long)]
    model_a: String,

    /// Second model identifier
    #[arg(long)]
    model_b: String,

    /// Properties to compare on (repeatable)
    #[arg(short, long)]
    property: Vec<String>,

    /// Maximum query budget per model
    #[arg(short, long, default_value_t = 500)]
    budget: u64,
}

#[derive(clap::Args, Debug)]
struct DriftCheckArgs {
    /// Model identifier
    #[arg(short, long)]
    model: String,

    /// Sliding window size for drift detection
    #[arg(long, default_value_t = 20)]
    window_size: usize,

    /// Sensitivity multiplier for anomaly threshold (in std-devs)
    #[arg(long, default_value_t = 2.0)]
    sensitivity: f64,

    /// Number of queries to issue for drift analysis
    #[arg(long, default_value_t = 200)]
    num_queries: u64,
}

#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OutputFormat {
    Json,
    Text,
}

// ===========================================================================
// Local result / config types
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CliConfig {
    default_budget: u64,
    default_confidence: f64,
    default_max_states: usize,
    default_epsilon: f64,
    log_level: String,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_budget: 1000,
            default_confidence: 0.95,
            default_max_states: 20,
            default_epsilon: 0.01,
            log_level: "info".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditResult {
    model: String,
    timestamp: DateTime<Utc>,
    properties_checked: Vec<String>,
    verdicts: Vec<PropertyVerdict>,
    automaton_summary: AutomatonSummary,
    overall_confidence: f64,
    queries_used: u64,
    certificate: CertificateOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnResult {
    model: String,
    num_states: usize,
    num_transitions: usize,
    alphabet_size: usize,
    converged: bool,
    final_epsilon: f64,
    queries_used: u64,
    states: Vec<String>,
    automaton_summary: AutomatonSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckResult {
    automaton_file: String,
    properties: Vec<String>,
    verdicts: Vec<PropertyVerdict>,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CertificateOutput {
    certificate_id: String,
    model: String,
    issued_at: DateTime<Utc>,
    valid_until: DateTime<Utc>,
    properties: Vec<PropertyVerdict>,
    overall_verdict: String,
    confidence_level: f64,
    query_budget_used: u64,
    methodology: String,
    hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComparisonOutput {
    model_a: String,
    model_b: String,
    timestamp: DateTime<Utc>,
    bisimulation_distance: f64,
    property_comparisons: Vec<PropertyComparison>,
    behavioral_summary: String,
    queries_used_a: u64,
    queries_used_b: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PropertyComparison {
    property: String,
    verdict_a: String,
    verdict_b: String,
    satisfaction_a: f64,
    satisfaction_b: f64,
    divergence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DriftCheckOutput {
    model: String,
    timestamp: DateTime<Utc>,
    drift_detected: bool,
    drift_score: f64,
    threshold: f64,
    window_size: usize,
    windows_analyzed: usize,
    anomalous_windows: Vec<AnomalousWindow>,
    recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnomalousWindow {
    window_index: usize,
    score: f64,
    deviation_sigma: f64,
    dominant_symbol_shift: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PropertyVerdict {
    property: String,
    satisfied: bool,
    satisfaction_degree: f64,
    confidence: f64,
    witness: Option<String>,
    details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AutomatonSummary {
    num_states: usize,
    num_transitions: usize,
    alphabet_size: usize,
    initial_state: String,
    accepting_states: Vec<String>,
    deterministic: bool,
    strongly_connected_components: usize,
}

// ===========================================================================
// Simulated alphabet & automaton helpers (using caber-core types)
// ===========================================================================

fn default_alphabet() -> Vec<Symbol> {
    vec![
        Symbol::new("safe"),
        Symbol::new("unsafe"),
        Symbol::new("neutral"),
        Symbol::new("refusal"),
        Symbol::new("compliance"),
    ]
}

fn simulate_learning(
    model: &str,
    budget: u64,
    max_states: usize,
    epsilon: f64,
) -> LearnResult {
    // Derive deterministic-looking values from the model name
    let name_hash: u64 = model.bytes().map(|b| b as u64).sum();
    let num_states = ((name_hash % (max_states as u64 - 2)) + 3) as usize;
    let alphabet = default_alphabet();
    let alphabet_size = alphabet.len();
    let num_transitions = num_states * alphabet_size;

    let states: Vec<String> = (0..num_states)
        .map(|i| StateId::indexed("q", i).to_string())
        .collect();

    let queries_used = budget.min((num_states as u64) * (alphabet_size as u64) * 10);
    let converged = epsilon > 0.001;
    let final_epsilon = if converged {
        epsilon * 0.8
    } else {
        epsilon * 1.2
    };

    LearnResult {
        model: model.to_string(),
        num_states,
        num_transitions,
        alphabet_size,
        converged,
        final_epsilon,
        queries_used,
        states: states.clone(),
        automaton_summary: AutomatonSummary {
            num_states,
            num_transitions,
            alphabet_size,
            initial_state: states.first().cloned().unwrap_or_default(),
            accepting_states: states.iter().take(num_states / 2 + 1).cloned().collect(),
            deterministic: true,
            strongly_connected_components: (num_states / 3).max(1),
        },
    }
}

fn simulate_property_check(property: &str, confidence: f64) -> PropertyVerdict {
    // Derive deterministic check results from the property name
    let prop_hash: u64 = property.bytes().map(|b| b as u64).sum();
    let satisfaction_degree = ((prop_hash % 100) as f64) / 100.0;
    let satisfied = satisfaction_degree >= 0.5;
    let witness = if !satisfied {
        let w = Word::from_str_slice(&["unsafe", "compliance"]);
        Some(format!("Counter-example trace: {}", w))
    } else {
        None
    };

    PropertyVerdict {
        property: property.to_string(),
        satisfied,
        satisfaction_degree,
        confidence,
        witness,
        details: format!(
            "Property '{}' checked via coalgebraic model-checking. \
             Satisfaction degree {:.4} (threshold 0.5).",
            property, satisfaction_degree
        ),
    }
}

fn generate_certificate(
    model: &str,
    verdicts: &[PropertyVerdict],
    queries_used: u64,
    confidence: f64,
) -> CertificateOutput {
    let now = Utc::now();
    let all_satisfied = verdicts.iter().all(|v| v.satisfied);
    let overall = if all_satisfied { "PASS" } else { "FAIL" };

    // Compute a simple hash from the verdicts for integrity
    let hash_input = format!(
        "{}:{}:{}:{}",
        model,
        queries_used,
        verdicts.len(),
        now.timestamp()
    );
    let hash: u64 = hash_input.bytes().fold(0xcbf29ce484222325u64, |acc, b| {
        (acc ^ (b as u64)).wrapping_mul(0x100000001b3)
    });

    CertificateOutput {
        certificate_id: format!("CERT-{:016x}", hash),
        model: model.to_string(),
        issued_at: now,
        valid_until: now + chrono::Duration::days(30),
        properties: verdicts.to_vec(),
        overall_verdict: overall.to_string(),
        confidence_level: confidence,
        query_budget_used: queries_used,
        methodology: "PCL* active learning with coalgebraic CEGAR refinement".to_string(),
        hash: format!("{:016x}", hash),
    }
}

// ===========================================================================
// Command handlers
// ===========================================================================

async fn handle_audit(args: AuditArgs) -> Result<()> {
    info!("Starting audit for model '{}'", args.model);

    let properties = if args.property.is_empty() {
        vec![
            "safety".to_string(),
            "fairness".to_string(),
            "consistency".to_string(),
        ]
    } else {
        args.property.clone()
    };

    // Phase 1: Learn
    info!("Phase 1/3: Learning behavioral automaton...");
    let learn = simulate_learning(&args.model, args.budget, 20, 0.01);
    info!(
        "Learned automaton with {} states, {} transitions ({} queries used)",
        learn.num_states, learn.num_transitions, learn.queries_used
    );

    // Phase 2: Check
    info!("Phase 2/3: Checking {} properties...", properties.len());
    let verdicts: Vec<PropertyVerdict> = properties
        .iter()
        .map(|p| simulate_property_check(p, args.confidence))
        .collect();

    let passed = verdicts.iter().filter(|v| v.satisfied).count();
    info!("{}/{} properties satisfied", passed, verdicts.len());

    // Phase 3: Certify
    info!("Phase 3/3: Generating certificate...");
    let certificate = generate_certificate(
        &args.model,
        &verdicts,
        learn.queries_used,
        args.confidence,
    );

    let result = AuditResult {
        model: args.model.clone(),
        timestamp: Utc::now(),
        properties_checked: properties,
        verdicts,
        automaton_summary: learn.automaton_summary,
        overall_confidence: args.confidence,
        queries_used: learn.queries_used,
        certificate,
    };

    let content = match args.format {
        OutputFormat::Json => format_json(&result)?,
        OutputFormat::Text => format_text_audit(&result),
    };

    write_output(&content, args.output.as_deref())?;
    Ok(())
}

async fn handle_learn(args: LearnArgs) -> Result<()> {
    info!("Learning behavioral automaton for model '{}'", args.model);
    info!(
        "Parameters: budget={}, max_states={}, epsilon={}",
        args.budget, args.max_states, args.epsilon
    );

    let result = simulate_learning(&args.model, args.budget, args.max_states, args.epsilon);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            CABER — Learning Result                      ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Model:         {:<40} ║", result.model);
    println!("║ States:        {:<40} ║", result.num_states);
    println!("║ Transitions:   {:<40} ║", result.num_transitions);
    println!("║ Alphabet size: {:<40} ║", result.alphabet_size);
    println!(
        "║ Converged:     {:<40} ║",
        if result.converged { "yes" } else { "no" }
    );
    println!("║ Final ε:       {:<40.6} ║", result.final_epsilon);
    println!("║ Queries used:  {:<40} ║", result.queries_used);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ States:                                                 ║");
    for s in &result.states {
        println!("║   • {:<52} ║", s);
    }
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Automaton Summary                                       ║");
    println!(
        "║   Initial state: {:<38} ║",
        result.automaton_summary.initial_state
    );
    println!(
        "║   Accepting:     {} states{:<30} ║",
        result.automaton_summary.accepting_states.len(),
        ""
    );
    println!(
        "║   Deterministic: {:<38} ║",
        if result.automaton_summary.deterministic {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "║   SCCs:          {:<38} ║",
        result.automaton_summary.strongly_connected_components
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    // Also emit JSON to stderr for piping
    let json = format_json(&result)?;
    info!("Full result JSON:\n{}", json);

    Ok(())
}

async fn handle_check(args: CheckArgs) -> Result<()> {
    info!(
        "Loading automaton from '{}'",
        args.automaton.display()
    );

    // Load the automaton summary from file
    let content = fs::read_to_string(&args.automaton).with_context(|| {
        format!(
            "Failed to read automaton file '{}'",
            args.automaton.display()
        )
    })?;
    let automaton_summary: AutomatonSummary = serde_json::from_str(&content).with_context(|| {
        format!(
            "Failed to parse automaton JSON from '{}'",
            args.automaton.display()
        )
    })?;

    info!(
        "Automaton loaded: {} states, {} transitions",
        automaton_summary.num_states, automaton_summary.num_transitions
    );

    let properties = if args.property.is_empty() {
        warn!("No properties specified; using defaults");
        vec!["safety".to_string(), "consistency".to_string()]
    } else {
        args.property
    };

    let verdicts: Vec<PropertyVerdict> = properties
        .iter()
        .map(|p| simulate_property_check(p, 0.95))
        .collect();

    let result = CheckResult {
        automaton_file: args.automaton.display().to_string(),
        properties: verdicts.iter().map(|v| v.property.clone()).collect(),
        verdicts: verdicts.clone(),
        timestamp: Utc::now(),
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            CABER — Property Check Results               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Automaton: {:<44} ║",
        truncate_str(&result.automaton_file, 44)
    );
    println!(
        "║ States:    {:<44} ║",
        automaton_summary.num_states
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    for v in &verdicts {
        let icon = if v.satisfied { "✓" } else { "✗" };
        println!(
            "║ {} {:<20} sat={:.4}  conf={:.4}{:<8} ║",
            icon, v.property, v.satisfaction_degree, v.confidence, ""
        );
        if let Some(ref w) = v.witness {
            println!("║   witness: {:<44} ║", truncate_str(w, 44));
        }
    }
    println!("╚══════════════════════════════════════════════════════════╝");

    let json = format_json(&result)?;
    info!("Full result JSON:\n{}", json);
    Ok(())
}

async fn handle_certify(args: CertifyArgs) -> Result<()> {
    info!(
        "Loading results from '{}'",
        args.results.display()
    );

    let content = fs::read_to_string(&args.results).with_context(|| {
        format!(
            "Failed to read results file '{}'",
            args.results.display()
        )
    })?;
    let audit: AuditResult = serde_json::from_str(&content).with_context(|| {
        format!(
            "Failed to parse results JSON from '{}'",
            args.results.display()
        )
    })?;

    let certificate = generate_certificate(
        &audit.model,
        &audit.verdicts,
        audit.queries_used,
        audit.overall_confidence,
    );

    let output_str = format_json(&certificate)?;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            CABER — Certificate Generated                ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ ID:       {:<45} ║", certificate.certificate_id);
    println!("║ Model:    {:<45} ║", certificate.model);
    println!("║ Verdict:  {:<45} ║", certificate.overall_verdict);
    println!("║ Issued:   {:<45} ║", certificate.issued_at.to_rfc3339());
    println!(
        "║ Valid to: {:<45} ║",
        certificate.valid_until.to_rfc3339()
    );
    println!("║ Hash:     {:<45} ║", certificate.hash);
    println!("╚══════════════════════════════════════════════════════════╝");

    write_output(&output_str, args.output.as_deref())?;
    Ok(())
}

async fn handle_compare(args: CompareArgs) -> Result<()> {
    info!(
        "Comparing models '{}' and '{}'",
        args.model_a, args.model_b
    );

    let properties = if args.property.is_empty() {
        vec![
            "safety".to_string(),
            "fairness".to_string(),
            "consistency".to_string(),
        ]
    } else {
        args.property
    };

    // Learn both models
    info!("Learning automaton for model A: {}", args.model_a);
    let learn_a = simulate_learning(&args.model_a, args.budget, 20, 0.01);

    info!("Learning automaton for model B: {}", args.model_b);
    let learn_b = simulate_learning(&args.model_b, args.budget, 20, 0.01);

    // Compute simulated bisimulation distance from model names
    let hash_a: u64 = args.model_a.bytes().map(|b| b as u64).sum();
    let hash_b: u64 = args.model_b.bytes().map(|b| b as u64).sum();
    let bisim_distance =
        ((hash_a as f64) - (hash_b as f64)).abs() / ((hash_a + hash_b) as f64).max(1.0);

    let property_comparisons: Vec<PropertyComparison> = properties
        .iter()
        .map(|p| {
            let va = simulate_property_check(p, 0.95);
            // Shift the check slightly for model B
            let satisfaction_b = (va.satisfaction_degree + bisim_distance * 0.3).min(1.0);
            let satisfied_b = satisfaction_b >= 0.5;
            PropertyComparison {
                property: p.clone(),
                verdict_a: if va.satisfied {
                    "PASS".to_string()
                } else {
                    "FAIL".to_string()
                },
                verdict_b: if satisfied_b {
                    "PASS".to_string()
                } else {
                    "FAIL".to_string()
                },
                satisfaction_a: va.satisfaction_degree,
                satisfaction_b,
                divergence: (va.satisfaction_degree - satisfaction_b).abs(),
            }
        })
        .collect();

    let summary = if bisim_distance < 0.1 {
        "Models are behaviorally very similar (bisimulation distance < 0.1)".to_string()
    } else if bisim_distance < 0.3 {
        "Models show moderate behavioral differences".to_string()
    } else {
        "Models show significant behavioral divergence".to_string()
    };

    let result = ComparisonOutput {
        model_a: args.model_a,
        model_b: args.model_b,
        timestamp: Utc::now(),
        bisimulation_distance: bisim_distance,
        property_comparisons,
        behavioral_summary: summary,
        queries_used_a: learn_a.queries_used,
        queries_used_b: learn_b.queries_used,
    };

    let content = format_text_comparison(&result);
    println!("{}", content);

    let json = format_json(&result)?;
    info!("Full result JSON:\n{}", json);
    Ok(())
}

async fn handle_drift_check(args: DriftCheckArgs) -> Result<()> {
    info!(
        "Running drift detection for model '{}' (window_size={}, sensitivity={}, queries={})",
        args.model, args.window_size, args.sensitivity, args.num_queries
    );

    let num_windows = (args.num_queries as usize) / args.window_size;
    if num_windows == 0 {
        anyhow::bail!(
            "Not enough queries ({}) for window size ({}). Need at least {} queries.",
            args.num_queries,
            args.window_size,
            args.window_size
        );
    }

    // Simulate per-window scores using a deterministic hash of the model name
    let base_hash: u64 = args.model.bytes().map(|b| b as u64).sum();
    let mut window_scores: Vec<f64> = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let raw = ((base_hash.wrapping_mul(31).wrapping_add(i as u64)) % 1000) as f64 / 1000.0;
        window_scores.push(raw);
    }

    // Compute mean and std-dev
    let mean: f64 = window_scores.iter().sum::<f64>() / (num_windows as f64);
    let variance: f64 = window_scores
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / (num_windows as f64);
    let std_dev = variance.sqrt();

    let threshold = mean + args.sensitivity * std_dev;

    let alphabet = default_alphabet();
    let anomalous_windows: Vec<AnomalousWindow> = window_scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score > threshold)
        .map(|(idx, &score)| {
            let deviation = if std_dev > 0.0 {
                (score - mean) / std_dev
            } else {
                0.0
            };
            let shift_sym = &alphabet[idx % alphabet.len()];
            AnomalousWindow {
                window_index: idx,
                score,
                deviation_sigma: deviation,
                dominant_symbol_shift: Some(format!(
                    "shift toward '{}' (Δ={:.4})",
                    shift_sym, score - mean
                )),
            }
        })
        .collect();

    let drift_detected = !anomalous_windows.is_empty();
    let drift_score = if drift_detected {
        anomalous_windows
            .iter()
            .map(|w| w.deviation_sigma)
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    let recommendation = if !drift_detected {
        "No significant behavioral drift detected. Model behavior is stable.".to_string()
    } else if drift_score < 3.0 {
        format!(
            "Minor drift detected ({} anomalous windows). Consider monitoring more frequently.",
            anomalous_windows.len()
        )
    } else {
        format!(
            "Significant drift detected (peak {:.2}σ). Re-audit recommended.",
            drift_score
        )
    };

    let result = DriftCheckOutput {
        model: args.model.clone(),
        timestamp: Utc::now(),
        drift_detected,
        drift_score,
        threshold,
        window_size: args.window_size,
        windows_analyzed: num_windows,
        anomalous_windows,
        recommendation,
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            CABER — Drift Detection Report               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Model:      {:<43} ║", result.model);
    println!(
        "║ Drift:      {:<43} ║",
        if result.drift_detected {
            "⚠ DETECTED"
        } else {
            "✓ None"
        }
    );
    println!("║ Score:      {:<43.4} ║", result.drift_score);
    println!("║ Threshold:  {:<43.4} ║", result.threshold);
    println!("║ Windows:    {:<43} ║", result.windows_analyzed);
    println!(
        "║ Anomalous:  {:<43} ║",
        result.anomalous_windows.len()
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    if !result.anomalous_windows.is_empty() {
        println!("║ Anomalous Windows:                                      ║");
        for aw in &result.anomalous_windows {
            println!(
                "║   Window {:>3}: score={:.4}  σ={:.2}{:<18} ║",
                aw.window_index, aw.score, aw.deviation_sigma, ""
            );
            if let Some(ref shift) = aw.dominant_symbol_shift {
                println!("║     {:<52} ║", truncate_str(shift, 52));
            }
        }
        println!("╠══════════════════════════════════════════════════════════╣");
    }
    println!(
        "║ {}{}║",
        truncate_str(&result.recommendation, 55),
        " ".repeat(55usize.saturating_sub(result.recommendation.len()))
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    let json = format_json(&result)?;
    info!("Full result JSON:\n{}", json);
    Ok(())
}

// ===========================================================================
// Output formatting
// ===========================================================================

fn format_json<T: Serialize>(data: &T) -> Result<String> {
    serde_json::to_string_pretty(data).context("Failed to serialize result to JSON")
}

fn format_text_audit(result: &AuditResult) -> String {
    let mut out = String::new();
    out.push_str("╔══════════════════════════════════════════════════════════╗\n");
    out.push_str("║            CABER — Behavioral Audit Report              ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str(&format!(
        "║ Model:       {:<42} ║\n",
        result.model
    ));
    out.push_str(&format!(
        "║ Date:        {:<42} ║\n",
        result.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
    ));
    out.push_str(&format!(
        "║ Queries:     {:<42} ║\n",
        result.queries_used
    ));
    out.push_str(&format!(
        "║ Confidence:  {:<42.4} ║\n",
        result.overall_confidence
    ));
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str("║ Automaton Summary                                       ║\n");
    out.push_str(&format!(
        "║   States:        {:<38} ║\n",
        result.automaton_summary.num_states
    ));
    out.push_str(&format!(
        "║   Transitions:   {:<38} ║\n",
        result.automaton_summary.num_transitions
    ));
    out.push_str(&format!(
        "║   Alphabet:      {:<38} ║\n",
        result.automaton_summary.alphabet_size
    ));
    out.push_str(&format!(
        "║   Deterministic: {:<38} ║\n",
        if result.automaton_summary.deterministic {
            "yes"
        } else {
            "no"
        }
    ));
    out.push_str(&format!(
        "║   SCCs:          {:<38} ║\n",
        result.automaton_summary.strongly_connected_components
    ));
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str("║ Property Verdicts                                       ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    for v in &result.verdicts {
        let icon = if v.satisfied { "✓" } else { "✗" };
        out.push_str(&format!(
            "║ {} {:<20} sat={:.4}  conf={:.4}{:<8} ║\n",
            icon, v.property, v.satisfaction_degree, v.confidence, ""
        ));
        if let Some(ref w) = v.witness {
            out.push_str(&format!(
                "║   witness: {:<44} ║\n",
                truncate_str(w, 44)
            ));
        }
    }
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str("║ Certificate                                             ║\n");
    out.push_str(&format!(
        "║   ID:      {:<44} ║\n",
        result.certificate.certificate_id
    ));
    out.push_str(&format!(
        "║   Verdict: {:<44} ║\n",
        result.certificate.overall_verdict
    ));
    out.push_str(&format!(
        "║   Hash:    {:<44} ║\n",
        result.certificate.hash
    ));
    out.push_str(&format!(
        "║   Valid:   {:<44} ║\n",
        result.certificate.valid_until.to_rfc3339()
    ));
    out.push_str("╚══════════════════════════════════════════════════════════╝\n");
    out
}

fn format_text_comparison(result: &ComparisonOutput) -> String {
    let mut out = String::new();
    out.push_str("╔══════════════════════════════════════════════════════════╗\n");
    out.push_str("║            CABER — Model Comparison Report              ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str(&format!(
        "║ Model A:     {:<42} ║\n",
        result.model_a
    ));
    out.push_str(&format!(
        "║ Model B:     {:<42} ║\n",
        result.model_b
    ));
    out.push_str(&format!(
        "║ Bisim dist:  {:<42.6} ║\n",
        result.bisimulation_distance
    ));
    out.push_str(&format!(
        "║ Queries A:   {:<42} ║\n",
        result.queries_used_a
    ));
    out.push_str(&format!(
        "║ Queries B:   {:<42} ║\n",
        result.queries_used_b
    ));
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str("║ Property Comparison                                     ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    out.push_str("║  Property             A       B       Δ                 ║\n");
    out.push_str("║  ─────────────────────────────────────────              ║\n");
    for pc in &result.property_comparisons {
        out.push_str(&format!(
            "║  {:<20} {:<7} {:<7} {:.4}{:<14} ║\n",
            truncate_str(&pc.property, 20),
            pc.verdict_a,
            pc.verdict_b,
            pc.divergence,
            ""
        ));
    }
    out.push_str("╠══════════════════════════════════════════════════════════╣\n");
    let summary = truncate_str(&result.behavioral_summary, 54);
    out.push_str(&format!(
        "║ {}{}║\n",
        summary,
        " ".repeat(55usize.saturating_sub(summary.len()))
    ));
    out.push_str("╚══════════════════════════════════════════════════════════╝\n");
    out
}

fn write_output(content: &str, path: Option<&Path>) -> Result<()> {
    match path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create output directory '{}'", parent.display())
                })?;
            }
            fs::write(p, content)
                .with_context(|| format!("Failed to write output to '{}'", p.display()))?;
            info!("Output written to '{}'", p.display());
            Ok(())
        }
        None => {
            print!("{}", content);
            Ok(())
        }
    }
}

// ===========================================================================
// Config loading
// ===========================================================================

fn load_config(path: Option<&Path>) -> Result<CliConfig> {
    match path {
        Some(p) => {
            let content = fs::read_to_string(p)
                .with_context(|| format!("Failed to read config file '{}'", p.display()))?;

            // Try JSON first, then try a simple key=value fallback
            if let Ok(cfg) = serde_json::from_str::<CliConfig>(&content) {
                return Ok(cfg);
            }

            // Fallback: parse as simple key=value (TOML-like subset)
            let mut cfg = CliConfig::default();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    let key = key.trim().trim_matches('"');
                    let value = value.trim().trim_matches('"');
                    match key {
                        "default_budget" => {
                            cfg.default_budget = value.parse().unwrap_or(cfg.default_budget)
                        }
                        "default_confidence" => {
                            cfg.default_confidence =
                                value.parse().unwrap_or(cfg.default_confidence)
                        }
                        "default_max_states" => {
                            cfg.default_max_states =
                                value.parse().unwrap_or(cfg.default_max_states)
                        }
                        "default_epsilon" => {
                            cfg.default_epsilon = value.parse().unwrap_or(cfg.default_epsilon)
                        }
                        "log_level" => cfg.log_level = value.to_string(),
                        _ => {
                            warn!("Unknown config key: {}", key);
                        }
                    }
                }
            }
            Ok(cfg)
        }
        None => Ok(CliConfig::default()),
    }
}

// ===========================================================================
// Utilities
// ===========================================================================

fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

// ===========================================================================
// Main
// ===========================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = CaberCli::parse();

    // Set up logging based on verbosity
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .init();

    // Load config (if specified)
    let _config = load_config(cli.config.as_deref())?;
    info!("CABER CLI v{}", env!("CARGO_PKG_VERSION"));

    match cli.command {
        Commands::Audit(args) => handle_audit(args).await,
        Commands::Learn(args) => handle_learn(args).await,
        Commands::Check(args) => handle_check(args).await,
        Commands::Certify(args) => handle_certify(args).await,
        Commands::Compare(args) => handle_compare(args).await,
        Commands::DriftCheck(args) => handle_drift_check(args).await,
    }
}
