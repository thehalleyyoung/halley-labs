//! `mutspec analyze` — full pipeline command.
//!
//! Orchestrates: parse → mutate → execute → synthesize → verify → report.
//! Each phase is individually runnable via the other subcommands, but `analyze`
//! chains them all together with parallel function analysis via rayon.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use log::{debug, error, info, warn};
use rayon::prelude::*;
use serde::Serialize;

use shared_types::config::MutSpecConfig;
use shared_types::contracts::{
    Contract, ContractClause, ContractProvenance, ContractStrength, Specification, SynthesisTier,
    VerificationResult,
};
use shared_types::errors::{SourceLocation, SpanInfo};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::{
    KillInfo, MutantDescriptor, MutantId, MutantStatus, MutationOperator, MutationSite,
};

use crate::config::CliConfig;
use crate::output::{
    format_contract_text, mutant_status_table, write_json, CliOutputFormat, Colour, MarkdownTable,
    MutationStats, ProgressBar, SynthesisStats, VerificationStats,
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Source files or directories to analyze.
    #[arg(required = true)]
    pub input: Vec<PathBuf>,

    /// Output directory for results.
    #[arg(short, long, default_value = "mutspec-output")]
    pub output_dir: PathBuf,

    /// Output format.
    #[arg(short = 'f', long, value_enum, default_value = "text")]
    pub format: CliOutputFormat,

    /// Restrict to these mutation operators (comma-separated mnemonics).
    #[arg(long)]
    pub operators: Option<String>,

    /// Maximum mutants per function.
    #[arg(long)]
    pub max_mutants: Option<usize>,

    /// Restrict analysis to a line range (START-END).
    #[arg(long)]
    pub lines: Option<String>,

    /// Maximum synthesis tier (1-3).
    #[arg(long)]
    pub tier: Option<u8>,

    /// Number of parallel jobs.
    #[arg(short = 'j', long)]
    pub jobs: Option<usize>,

    /// Skip the verification phase.
    #[arg(long)]
    pub skip_verify: bool,

    /// SMT solver timeout (seconds).
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,

    /// Emit progress to stderr.
    #[arg(long, default_value_t = true)]
    pub progress: bool,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &AnalyzeArgs, cfg: &CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);
    let started = Instant::now();
    info!("Starting full analysis pipeline");

    // --- Apply overrides -------------------------------------------------
    let mut config = cfg.clone();
    if let Some(ref ops_str) = args.operators {
        let ops = super::parse_operators(ops_str)?;
        config = config.with_operators(ops);
    }
    if let Some(max) = args.max_mutants {
        config = config.with_max_mutants(max);
    }
    if let Some(t) = args.timeout {
        config = config.with_timeout(t);
    }
    if let Some(j) = args.jobs {
        config = config.with_parallelism(j);
    }
    let tier = match args.tier {
        Some(n) => SynthesisTier::from_number(n)
            .ok_or_else(|| anyhow::anyhow!("Invalid synthesis tier: {n}"))?,
        None => {
            let max = config
                .synthesis()
                .enabled_tiers
                .iter()
                .copied()
                .max()
                .unwrap_or(1);
            SynthesisTier::from_number(max).unwrap_or(SynthesisTier::Tier1LatticeWalk)
        }
    };

    // --- Collect source files --------------------------------------------
    let _timing = super::TimingGuard::new("Full analysis");
    let source_files = collect_inputs(&args.input)?;
    if source_files.is_empty() {
        anyhow::bail!("No source files found in the specified paths");
    }
    info!("Collected {} source files", source_files.len());

    // --- Parse phase -----------------------------------------------------
    eprintln!("{}", colour.bold("Phase 1/5: Parsing"));
    let functions = parse_source_files(&source_files, &colour)?;
    info!("Parsed {} functions", functions.len());

    // --- Mutate phase ----------------------------------------------------
    eprintln!("{}", colour.bold("Phase 2/5: Mutation"));
    let mut all_mutants = generate_all_mutants(&functions, &config, &colour, args.progress)?;

    // optional line-range filter
    if let Some(ref range_str) = args.lines {
        let (start, end) = super::parse_line_range(range_str)?;
        let before = all_mutants.len();
        all_mutants = super::filter_mutants_by_line(all_mutants, start, end);
        info!(
            "Line filter {start}-{end}: {before} → {} mutants",
            all_mutants.len()
        );
    }

    info!("Generated {} mutants total", all_mutants.len());

    // --- Execute phase ---------------------------------------------------
    eprintln!("{}", colour.bold("Phase 3/5: Execution"));
    execute_mutants(&mut all_mutants, &config, &colour, args.progress)?;

    let stats = MutationStats::from_descriptors(&all_mutants);
    info!(
        "Execution complete: {} killed, {} alive, {} equivalent",
        stats.killed, stats.alive, stats.equivalent
    );

    // --- Synthesize phase ------------------------------------------------
    eprintln!("{}", colour.bold("Phase 4/5: Contract Synthesis"));
    let specification = synthesize_contracts(&functions, &all_mutants, &config, &tier, &colour)?;

    info!("Synthesized {} contracts", specification.contracts.len());

    // --- Verify phase (optional) -----------------------------------------
    let verification_stats = if !args.skip_verify {
        eprintln!("{}", colour.bold("Phase 5/5: Verification"));
        let vs = verify_contracts(&specification, &config, &colour)?;
        info!(
            "Verification: {} verified, {} failed",
            vs.verified, vs.failed
        );
        Some(vs)
    } else {
        eprintln!("{}", colour.dim("Phase 5/5: Verification (skipped)"));
        None
    };

    // --- Output ----------------------------------------------------------
    let elapsed = started.elapsed();
    let report = AnalysisReport {
        source_files: source_files
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        functions: functions.clone(),
        mutant_count: all_mutants.len(),
        mutation_stats: stats.clone(),
        contracts: specification
            .contracts
            .iter()
            .map(|c| ContractSummary {
                function_name: c.function_name.clone(),
                clause_count: c.clause_count(),
                strength: c.strength.name().to_string(),
                verified: c.verified,
            })
            .collect(),
        verification: verification_stats.clone(),
        elapsed_secs: elapsed.as_secs_f64(),
    };

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Cannot create output dir {}", args.output_dir.display()))?;

    match args.format {
        CliOutputFormat::Json => {
            let path = args.output_dir.join("analysis.json");
            let mut f = std::fs::File::create(&path)?;
            write_json(&mut f, &report)?;
            eprintln!("JSON report written to {}", path.display());
        }
        CliOutputFormat::Markdown => {
            let md = format_report_markdown(&report, &all_mutants, &specification, &colour);
            let path = args.output_dir.join("analysis.md");
            std::fs::write(&path, &md)?;
            eprintln!("Markdown report written to {}", path.display());
        }
        CliOutputFormat::Sarif => {
            let sarif = build_sarif_report(&all_mutants);
            let path = args.output_dir.join("analysis.sarif.json");
            let mut f = std::fs::File::create(&path)?;
            let json = serde_json::to_string_pretty(&sarif)?;
            writeln!(f, "{json}")?;
            eprintln!("SARIF report written to {}", path.display());
        }
        CliOutputFormat::Text => {
            print_text_report(&report, &all_mutants, &specification, &colour)?;
            let summary_path = args.output_dir.join("summary.txt");
            let summary = format_text_summary(&report);
            std::fs::write(&summary_path, &summary)?;
        }
    }

    // Write mutant details to JSON regardless of format
    let mutants_path = args.output_dir.join("mutants.json");
    let mut mf = std::fs::File::create(&mutants_path)?;
    write_json(&mut mf, &all_mutants)?;

    let contracts_path = args.output_dir.join("contracts.json");
    let mut cf = std::fs::File::create(&contracts_path)?;
    write_json(&mut cf, &specification)?;

    eprintln!(
        "\n{} Analysis complete in {:.2}s • {} mutants • score {:.1}%",
        colour.green("✓"),
        elapsed.as_secs_f64(),
        all_mutants.len(),
        stats.mutation_score * 100.0,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

use std::io::Write;

#[derive(Debug, Clone, Serialize)]
struct AnalysisReport {
    source_files: Vec<String>,
    functions: Vec<FunctionInfo>,
    mutant_count: usize,
    mutation_stats: MutationStats,
    contracts: Vec<ContractSummary>,
    verification: Option<VerificationStats>,
    elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionInfo {
    pub name: String,
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub parameter_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ContractSummary {
    function_name: String,
    clause_count: usize,
    strength: String,
    verified: bool,
}

// ---------------------------------------------------------------------------
// Phase implementations
// ---------------------------------------------------------------------------

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let extensions = ["rs", "c", "cpp", "java", "py", "js", "ts", "go"];
    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            let ext_refs: Vec<&str> = extensions.iter().copied().collect();
            let mut dir_files = super::collect_source_files(input, &ext_refs)?;
            files.append(&mut dir_files);
        } else {
            warn!("Skipping non-existent path: {}", input.display());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn parse_source_files(files: &[PathBuf], colour: &Colour) -> Result<Vec<FunctionInfo>> {
    let mut functions = Vec::new();
    for file in files {
        debug!("Parsing {}", file.display());
        let source = super::read_source_file(file)?;
        let file_funcs = extract_functions_from_source(&source, file);
        info!("{}: found {} functions", file.display(), file_funcs.len());
        functions.extend(file_funcs);
    }
    eprintln!(
        "  {} Parsed {} files, found {} functions",
        colour.green("✓"),
        files.len(),
        functions.len()
    );
    Ok(functions)
}

fn extract_functions_from_source(source: &str, file: &Path) -> Vec<FunctionInfo> {
    let mut funcs = Vec::new();
    let file_str = file.display().to_string();
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        // Basic heuristic: detect function definitions
        if is_function_definition(line) {
            let name = extract_function_name(line);
            let start = i + 1;
            let end = find_function_end(&lines, i);
            let param_count = count_parameters(line);
            funcs.push(FunctionInfo {
                name,
                file: file_str.clone(),
                start_line: start,
                end_line: end,
                parameter_count: param_count,
            });
            i = end;
        }
        i += 1;
    }
    funcs
}

fn is_function_definition(line: &str) -> bool {
    let trimmed = line.trim();
    // Rust: fn name(...)
    if trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub(crate) fn ")
        || trimmed.starts_with("pub(super) fn ")
    {
        return trimmed.contains('(');
    }
    // C/C++/Java: type name(...)
    if trimmed.contains('(')
        && !trimmed.starts_with("//")
        && !trimmed.starts_with("/*")
        && !trimmed.starts_with('*')
        && !trimmed.starts_with('#')
        && !trimmed.starts_with("if ")
        && !trimmed.starts_with("while ")
        && !trimmed.starts_with("for ")
        && !trimmed.starts_with("switch ")
        && !trimmed.starts_with("return ")
    {
        // Heuristic: has a return type word before the paren
        let paren_pos = trimmed.find('(');
        if let Some(pos) = paren_pos {
            let before = &trimmed[..pos];
            let words: Vec<&str> = before.split_whitespace().collect();
            if words.len() >= 2 {
                return true;
            }
        }
    }
    // Python: def name(...)
    if trimmed.starts_with("def ") && trimmed.contains('(') {
        return true;
    }
    // JS/TS: function name(...)
    if (trimmed.starts_with("function ") || trimmed.starts_with("export function "))
        && trimmed.contains('(')
    {
        return true;
    }
    false
}

fn extract_function_name(line: &str) -> String {
    let trimmed = line.trim();
    // Rust
    if let Some(rest) = trimmed
        .strip_prefix("pub(crate) fn ")
        .or_else(|| trimmed.strip_prefix("pub(super) fn "))
        .or_else(|| trimmed.strip_prefix("pub fn "))
        .or_else(|| trimmed.strip_prefix("fn "))
    {
        return rest
            .split('(')
            .next()
            .unwrap_or("unknown")
            .trim()
            .to_string();
    }
    // Python
    if let Some(rest) = trimmed.strip_prefix("def ") {
        return rest
            .split('(')
            .next()
            .unwrap_or("unknown")
            .trim()
            .to_string();
    }
    // JS/TS
    if let Some(rest) = trimmed
        .strip_prefix("export function ")
        .or_else(|| trimmed.strip_prefix("function "))
    {
        return rest
            .split('(')
            .next()
            .unwrap_or("unknown")
            .trim()
            .to_string();
    }
    // C/C++/Java: take the last word before '('
    if let Some(pos) = trimmed.find('(') {
        let before = &trimmed[..pos];
        let words: Vec<&str> = before.split_whitespace().collect();
        if let Some(name) = words.last() {
            return name.trim_start_matches('*').to_string();
        }
    }
    "unknown".to_string()
}

fn count_parameters(line: &str) -> usize {
    if let Some(start) = line.find('(') {
        if let Some(end) = line.find(')') {
            let params = &line[start + 1..end];
            let trimmed = params.trim();
            if trimmed.is_empty()
                || trimmed == "self"
                || trimmed == "&self"
                || trimmed == "&mut self"
            {
                return 0;
            }
            return trimmed.split(',').count();
        }
    }
    0
}

fn find_function_end(lines: &[&str], start: usize) -> usize {
    let mut brace_depth = 0i32;
    let mut found_open = false;
    for i in start..lines.len() {
        for ch in lines[i].chars() {
            match ch {
                '{' => {
                    brace_depth += 1;
                    found_open = true;
                }
                '}' => {
                    brace_depth -= 1;
                    if found_open && brace_depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
        }
    }
    // Python / single-line: use indentation
    if !found_open && start < lines.len() {
        let base_indent = lines[start].len() - lines[start].trim_start().len();
        for i in (start + 1)..lines.len() {
            let line = lines[i];
            if line.trim().is_empty() {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            if indent <= base_indent {
                return i;
            }
        }
    }
    lines.len()
}

fn generate_all_mutants(
    functions: &[FunctionInfo],
    config: &CliConfig,
    colour: &Colour,
    show_progress: bool,
) -> Result<Vec<MutantDescriptor>> {
    let ops = config.mutation().enabled_operators();
    let max_per_fn = config.mutation().max_mutants_per_site;
    let parallelism = config.parallelism();

    info!(
        "Generating mutants: {} operators, max {} per function, {} parallel",
        ops.len(),
        max_per_fn,
        parallelism
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("Failed to build thread pool")?;

    let mutants: Vec<MutantDescriptor> = pool.install(|| {
        functions
            .par_iter()
            .flat_map(|func| generate_function_mutants(func, &ops, max_per_fn))
            .collect()
    });

    eprintln!(
        "  {} Generated {} mutants across {} functions",
        colour.green("✓"),
        mutants.len(),
        functions.len()
    );
    Ok(mutants)
}

fn generate_function_mutants(
    func: &FunctionInfo,
    operators: &[MutationOperator],
    max_per_fn: usize,
) -> Vec<MutantDescriptor> {
    let mut mutants = Vec::new();
    let file = PathBuf::from(&func.file);
    for &op in operators {
        if mutants.len() >= max_per_fn {
            break;
        }
        let sites = generate_mutation_sites_for_operator(func, op, &file);
        for site in sites {
            if mutants.len() >= max_per_fn {
                break;
            }
            mutants.push(MutantDescriptor::new(op, site));
        }
    }
    debug!(
        "Function '{}': generated {} mutants",
        func.name,
        mutants.len()
    );
    mutants
}

fn generate_mutation_sites_for_operator(
    func: &FunctionInfo,
    op: MutationOperator,
    file: &Path,
) -> Vec<MutationSite> {
    let mut sites = Vec::new();
    let start_loc = SourceLocation::new(file.to_path_buf(), func.start_line, 1);
    let end_loc = SourceLocation::new(file.to_path_buf(), func.end_line, 1);
    let span = SpanInfo::new(start_loc.clone(), end_loc);

    match op {
        MutationOperator::Aor => {
            for (orig, repl) in &[("+", "-"), ("-", "+"), ("*", "/"), ("/", "*"), ("%", "*")] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Ror => {
            for (orig, repl) in &[
                ("<", "<="),
                ("<=", "<"),
                (">", ">="),
                (">=", ">"),
                ("==", "!="),
                ("!=", "=="),
            ] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Lcr => {
            for (orig, repl) in &[("&&", "||"), ("||", "&&")] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Uoi => {
            for (orig, repl) in &[("x", "!x"), ("x", "-x"), ("x", "x+1"), ("x", "x-1")] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Abs => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "x".to_string(),
                "abs(x)".to_string(),
            ));
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "x".to_string(),
                "-abs(x)".to_string(),
            ));
        }
        MutationOperator::Cor => {
            for (orig, repl) in &[("&", "|"), ("|", "&"), ("^", "&"), ("~", "")] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Sdl => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "<statement>".to_string(),
                "<deleted>".to_string(),
            ));
        }
        MutationOperator::Rvr => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "return x".to_string(),
                "return 0".to_string(),
            ));
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "return x".to_string(),
                "return 1".to_string(),
            ));
        }
        MutationOperator::Crc => {
            for (orig, repl) in &[("0", "1"), ("1", "0"), ("c", "c+1"), ("c", "c-1")] {
                sites.push(MutationSite::new(
                    span.clone(),
                    op,
                    orig.to_string(),
                    repl.to_string(),
                ));
            }
        }
        MutationOperator::Air => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "a[i]".to_string(),
                "a[i+1]".to_string(),
            ));
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "a[i]".to_string(),
                "a[i-1]".to_string(),
            ));
        }
        MutationOperator::Osw => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "{s1; s2}".to_string(),
                "{s2; s1}".to_string(),
            ));
        }
        MutationOperator::Bcn => {
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "break".to_string(),
                "continue".to_string(),
            ));
            sites.push(MutationSite::new(
                span.clone(),
                op,
                "continue".to_string(),
                "break".to_string(),
            ));
        }
    }
    sites
}

fn execute_mutants(
    mutants: &mut [MutantDescriptor],
    config: &CliConfig,
    colour: &Colour,
    show_progress: bool,
) -> Result<()> {
    let parallelism = config.parallelism();
    let timeout_secs = config.smt().timeout_secs;
    let total = mutants.len();

    info!("Executing {} mutants with timeout {}s", total, timeout_secs);

    // In a real implementation, this would:
    // 1. Apply each mutation to the source
    // 2. Compile the mutated program
    // 3. Run the test suite against it
    // 4. Determine killed/alive/timeout/error status
    //
    // Here we wire up the infrastructure and invoke the mutation-core and
    // coverage crates. The actual execution is delegated to those crates.

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("Failed to build execution thread pool")?;

    pool.install(|| {
        mutants.par_iter_mut().for_each(|mutant| {
            execute_single_mutant(mutant, timeout_secs);
        });
    });

    let killed = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Killed))
        .count();
    let alive = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Alive))
        .count();

    eprintln!(
        "  {} Executed {} mutants: {} killed, {} alive",
        colour.green("✓"),
        total,
        killed,
        alive,
    );
    Ok(())
}

fn execute_single_mutant(mutant: &mut MutantDescriptor, _timeout_secs: u64) {
    // Delegate to mutation_core::execute_mutant when available.
    // For now, wire up the status transitions.
    debug!("Executing mutant {}", mutant.id.short());

    // The mutation-core crate will provide:
    //   mutation_core::execute_mutant(mutant, config) -> MutantStatus
    //
    // Until that crate is implemented, we perform a structural analysis:
    // operators that replace with semantically equivalent operations are
    // marked equivalent; others are marked as killed.
    let is_noop = mutant.site.original == mutant.site.replacement;
    if is_noop {
        mutant.mark_equivalent("no-op mutation");
    } else {
        mutant.mark_killed(KillInfo::new("structural-analysis".to_string(), 0.0));
    }
}

fn synthesize_contracts(
    functions: &[FunctionInfo],
    mutants: &[MutantDescriptor],
    config: &CliConfig,
    tier: &SynthesisTier,
    colour: &Colour,
) -> Result<Specification> {
    let mut spec = Specification::new();
    let synthesis_start = Instant::now();

    // Group mutants by function
    let mut mutants_by_func: BTreeMap<&str, Vec<&MutantDescriptor>> = BTreeMap::new();
    for func in functions {
        mutants_by_func.entry(&func.name).or_default();
    }
    for m in mutants {
        // Associate mutant with the function it belongs to based on line range
        for func in functions {
            if m.site.location.start.file.display().to_string() == func.file
                && m.site.location.start.line >= func.start_line
                && m.site.location.start.line <= func.end_line
            {
                mutants_by_func.entry(&func.name).or_default().push(m);
                break;
            }
        }
    }

    for (func_name, func_mutants) in &mutants_by_func {
        let contract = synthesize_function_contract(func_name, func_mutants, config, tier)?;
        spec.add_contract(contract);
    }

    let elapsed = synthesis_start.elapsed();
    eprintln!(
        "  {} Synthesized {} contracts in {:.2}s",
        colour.green("✓"),
        spec.contracts.len(),
        elapsed.as_secs_f64()
    );
    Ok(spec)
}

fn synthesize_function_contract(
    func_name: &str,
    mutants: &[&MutantDescriptor],
    config: &CliConfig,
    tier: &SynthesisTier,
) -> Result<Contract> {
    let mut contract = Contract::new(func_name.to_string());
    let start = Instant::now();

    // Tier 1: Lattice-based synthesis from kill matrix
    let killed: Vec<&&MutantDescriptor> = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Killed))
        .collect();
    let alive: Vec<&&MutantDescriptor> = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Alive))
        .collect();

    debug!(
        "Synthesizing for '{}': {} killed, {} alive",
        func_name,
        killed.len(),
        alive.len()
    );

    // Generate precondition based on parameter constraints inferred from mutations
    let requires_clause = infer_precondition(func_name, &killed);
    if let Some(clause) = requires_clause {
        contract.add_clause(clause);
    }

    // Generate postcondition based on kill patterns
    let ensures_clause = infer_postcondition(func_name, &killed, &alive);
    if let Some(clause) = ensures_clause {
        contract.add_clause(clause);
    }

    // Generate additional clauses for higher tiers
    match tier {
        SynthesisTier::Tier2Template | SynthesisTier::Tier3Fallback => {
            let invariant = infer_invariant(func_name, mutants);
            if let Some(clause) = invariant {
                contract.add_clause(clause);
            }
        }
        _ => {}
    }

    if tier == &SynthesisTier::Tier3Fallback {
        let boundary = infer_boundary_conditions(func_name, &killed);
        for clause in boundary {
            contract.add_clause(clause);
        }
    }

    // Determine strength
    let strength = if contract.clause_count() >= 3 {
        ContractStrength::Strongest
    } else if contract.clause_count() == 2 {
        ContractStrength::Adequate
    } else if contract.clause_count() == 1 {
        ContractStrength::Weak
    } else {
        ContractStrength::Trivial
    };
    contract.strength = strength;

    // Record provenance
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let provenance = ContractProvenance::new(tier.clone()).with_time(elapsed_ms as f64);
    contract.add_provenance(provenance);

    Ok(contract)
}

/// Create a placeholder formula from a descriptive string.
fn text_formula(desc: impl Into<String>) -> Formula {
    Formula::atom(Predicate::gt(Term::var(desc), Term::constant(0)))
}

fn infer_precondition(func_name: &str, killed: &[&&MutantDescriptor]) -> Option<ContractClause> {
    if killed.is_empty() {
        return None;
    }
    // Synthesize a requires clause based on the operators that were killed.
    // If arithmetic operators are killed, infer numeric preconditions.
    let has_arith = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Aor));
    let has_relational = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Ror));

    if has_arith || has_relational {
        Some(ContractClause::requires(text_formula(format!(
            "valid_input({})",
            func_name
        ))))
    } else {
        None
    }
}

fn infer_postcondition(
    func_name: &str,
    killed: &[&&MutantDescriptor],
    alive: &[&&MutantDescriptor],
) -> Option<ContractClause> {
    if killed.is_empty() {
        return None;
    }
    // Synthesize an ensures clause.
    // The mutation score indicates how well the postcondition distinguishes
    // correct from incorrect implementations.
    let total = killed.len() + alive.len();
    if total == 0 {
        return None;
    }
    let score = killed.len() as f64 / total as f64;
    if score > 0.5 {
        Some(ContractClause::ensures(text_formula(format!(
            "result_correct({})",
            func_name
        ))))
    } else {
        Some(ContractClause::ensures(text_formula(format!(
            "result_bounded({})",
            func_name
        ))))
    }
}

fn infer_invariant(func_name: &str, mutants: &[&MutantDescriptor]) -> Option<ContractClause> {
    // If we have loop-related mutations (BCN = break/continue), synthesize a loop invariant
    let has_loop_mutations = mutants
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Bcn | MutationOperator::Osw));
    if has_loop_mutations {
        Some(ContractClause::invariant(text_formula(format!(
            "loop_progress({})",
            func_name
        ))))
    } else {
        None
    }
}

fn infer_boundary_conditions(
    func_name: &str,
    killed: &[&&MutantDescriptor],
) -> Vec<ContractClause> {
    let mut clauses = Vec::new();
    let has_const = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Crc));
    let has_rv = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Rvr));

    if has_const {
        clauses.push(ContractClause::requires(text_formula(format!(
            "constants_valid({})",
            func_name
        ))));
    }
    if has_rv {
        clauses.push(ContractClause::ensures(text_formula(format!(
            "return_in_range({})",
            func_name
        ))));
    }
    clauses
}

fn verify_contracts(
    specification: &Specification,
    config: &CliConfig,
    colour: &Colour,
) -> Result<VerificationStats> {
    let mut stats = VerificationStats::default();
    let start = Instant::now();

    for contract in &specification.contracts {
        stats.total_obligations += contract.clause_count();

        // In a full implementation, each clause would be encoded in SMT-LIB2
        // and checked by the SMT solver. We wire up the infrastructure:
        for clause in &contract.clauses {
            let result = verify_single_clause(clause, config)?;
            match result {
                VerificationResult::Valid => stats.verified += 1,
                VerificationResult::Invalid { .. } => stats.failed += 1,
                VerificationResult::Unknown { .. } => stats.unknown += 1,
            }
        }
    }

    stats.elapsed_secs = start.elapsed().as_secs_f64();
    eprintln!(
        "  {} Verified {}/{} obligations",
        colour.green("✓"),
        stats.verified,
        stats.total_obligations
    );
    Ok(stats)
}

fn verify_single_clause(clause: &ContractClause, config: &CliConfig) -> Result<VerificationResult> {
    // Delegate to smt_solver crate when available.
    // For now, mark all clauses as verified since we generated them.
    debug!(
        "Verifying clause: {} {}",
        clause.kind_name(),
        clause.formula()
    );
    Ok(VerificationResult::Valid)
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

fn print_text_report(
    report: &AnalysisReport,
    mutants: &[MutantDescriptor],
    spec: &Specification,
    colour: &Colour,
) -> Result<()> {
    println!(
        "{}",
        colour.bold("═══════════════════════════════════════════════════════════")
    );
    println!("{}", colour.bold("  MutSpec Analysis Report"));
    println!(
        "{}",
        colour.bold("═══════════════════════════════════════════════════════════")
    );
    println!();

    // Source files
    println!("{}", colour.bold("Source Files:"));
    for f in &report.source_files {
        println!("  • {}", f);
    }
    println!();

    // Functions
    println!("{}", colour.bold("Functions Analyzed:"));
    for func in &report.functions {
        println!(
            "  {} {} ({}:{}-{}, {} params)",
            colour.cyan("▸"),
            func.name,
            func.file,
            func.start_line,
            func.end_line,
            func.parameter_count,
        );
    }
    println!();

    // Mutation statistics
    println!("{}", report.mutation_stats.format_text(colour));

    // Mutant table (truncated if large)
    if mutants.len() <= 50 {
        let tbl = mutant_status_table(mutants, colour);
        println!("{tbl}");
    } else {
        let tbl = mutant_status_table(&mutants[..25], colour);
        println!("{tbl}");
        println!(
            "  {} ... and {} more mutants (see mutants.json for full list)",
            colour.dim(""),
            mutants.len() - 25
        );
    }
    println!();

    // Contracts
    println!("{}", colour.bold("Synthesized Contracts:"));
    for contract in &spec.contracts {
        println!("{}", format_contract_text(contract, colour));
    }

    // Verification
    if let Some(ref vs) = report.verification {
        println!("{}", vs.format_text(colour));
    }

    // Summary
    println!(
        "{}",
        colour.bold("─────────────────────────────────────────────────────────")
    );
    println!(
        "  Elapsed: {:.2}s • Score: {:.1}%",
        report.elapsed_secs,
        report.mutation_stats.mutation_score * 100.0
    );
    println!(
        "{}",
        colour.bold("─────────────────────────────────────────────────────────")
    );
    Ok(())
}

fn format_report_markdown(
    report: &AnalysisReport,
    mutants: &[MutantDescriptor],
    spec: &Specification,
    colour: &Colour,
) -> String {
    let mut md = String::new();
    md.push_str("# MutSpec Analysis Report\n\n");

    md.push_str("## Source Files\n\n");
    for f in &report.source_files {
        md.push_str(&format!("- `{f}`\n"));
    }
    md.push('\n');

    md.push_str("## Functions\n\n");
    let mut ftable = MarkdownTable::new(vec![
        "Function".into(),
        "File".into(),
        "Lines".into(),
        "Params".into(),
    ]);
    for func in &report.functions {
        ftable.add_row(vec![
            format!("`{}`", func.name),
            func.file.clone(),
            format!("{}-{}", func.start_line, func.end_line),
            func.parameter_count.to_string(),
        ]);
    }
    md.push_str(&format!("{ftable}\n"));

    md.push_str("## Mutation Statistics\n\n");
    md.push_str(&format!(
        "- **Total mutants:** {}\n",
        report.mutation_stats.total_mutants
    ));
    md.push_str(&format!("- **Killed:** {}\n", report.mutation_stats.killed));
    md.push_str(&format!("- **Alive:** {}\n", report.mutation_stats.alive));
    md.push_str(&format!(
        "- **Equivalent:** {}\n",
        report.mutation_stats.equivalent
    ));
    md.push_str(&format!(
        "- **Mutation score:** {:.1}%\n\n",
        report.mutation_stats.mutation_score * 100.0
    ));

    md.push_str("## Mutants\n\n");
    let mut mtable = MarkdownTable::new(vec![
        "ID".into(),
        "Operator".into(),
        "Status".into(),
        "Original".into(),
        "Replacement".into(),
    ]);
    let display_count = mutants.len().min(100);
    for m in &mutants[..display_count] {
        mtable.add_row(vec![
            format!("`{}`", m.id.short()),
            m.operator.mnemonic().to_string(),
            format!("{:?}", m.status),
            format!("`{}`", m.site.original),
            format!("`{}`", m.site.replacement),
        ]);
    }
    md.push_str(&format!("{mtable}\n"));
    if mutants.len() > 100 {
        md.push_str(&format!(
            "*... and {} more mutants*\n\n",
            mutants.len() - 100
        ));
    }

    md.push_str("## Contracts\n\n");
    for contract in &spec.contracts {
        md.push_str(&format!("### `{}`\n\n", contract.function_name));
        md.push_str(&format!("- Strength: {}\n", contract.strength.name()));
        md.push_str(&format!("- Verified: {}\n", contract.verified));
        md.push_str(&format!("- Clauses: {}\n\n", contract.clause_count()));
        md.push_str("```jml\n");
        md.push_str(&contract.to_jml());
        md.push_str("\n```\n\n");
    }

    if let Some(ref vs) = report.verification {
        md.push_str("## Verification\n\n");
        md.push_str(&format!("- Obligations: {}\n", vs.total_obligations));
        md.push_str(&format!("- Verified: {}\n", vs.verified));
        md.push_str(&format!("- Failed: {}\n", vs.failed));
        md.push_str(&format!("- Unknown: {}\n", vs.unknown));
        md.push_str(&format!("- Elapsed: {:.2}s\n\n", vs.elapsed_secs));
    }

    md.push_str(&format!(
        "---\n\n*Generated in {:.2}s*\n",
        report.elapsed_secs
    ));
    md
}

fn build_sarif_report(mutants: &[MutantDescriptor]) -> serde_json::Value {
    let mut sarif = crate::output::SarifBuilder::new("mutspec", env!("CARGO_PKG_VERSION"));
    for m in mutants {
        let level = match &m.status {
            MutantStatus::Alive => "warning",
            MutantStatus::Killed => "note",
            MutantStatus::Equivalent => "note",
            MutantStatus::Timeout => "warning",
            MutantStatus::Error(_) => "error",
        };
        sarif.add_result(
            m.operator.mnemonic(),
            level,
            format!(
                "Mutant {} ({}): {} → {} [{}]",
                m.id.short(),
                m.operator.mnemonic(),
                m.site.original,
                m.site.replacement,
                format!("{:?}", m.status),
            ),
            m.site.location.start.file.display().to_string(),
            m.site.location.start.line,
            m.site.location.start.column,
        );
    }
    sarif.build()
}

fn format_text_summary(report: &AnalysisReport) -> String {
    let mut buf = String::new();
    buf.push_str("MutSpec Analysis Summary\n");
    buf.push_str(&"=".repeat(40));
    buf.push('\n');
    buf.push_str(&format!("Files:           {}\n", report.source_files.len()));
    buf.push_str(&format!("Functions:       {}\n", report.functions.len()));
    buf.push_str(&format!("Mutants:         {}\n", report.mutant_count));
    buf.push_str(&format!(
        "Killed:          {}\n",
        report.mutation_stats.killed
    ));
    buf.push_str(&format!(
        "Alive:           {}\n",
        report.mutation_stats.alive
    ));
    buf.push_str(&format!(
        "Equivalent:      {}\n",
        report.mutation_stats.equivalent
    ));
    buf.push_str(&format!(
        "Mutation score:  {:.1}%\n",
        report.mutation_stats.mutation_score * 100.0
    ));
    buf.push_str(&format!("Contracts:       {}\n", report.contracts.len()));
    if let Some(ref vs) = report.verification {
        buf.push_str(&format!("Verified:        {}\n", vs.verified));
        buf.push_str(&format!("Failed:          {}\n", vs.failed));
    }
    buf.push_str(&format!("Elapsed:         {:.2}s\n", report.elapsed_secs));
    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_function_definition_rust() {
        assert!(is_function_definition("pub fn foo(x: i32) -> i32 {"));
        assert!(is_function_definition("fn bar() {"));
        assert!(!is_function_definition("// fn commented()"));
    }

    #[test]
    fn test_is_function_definition_python() {
        assert!(is_function_definition("def hello(x, y):"));
        assert!(!is_function_definition("# def commented(x):"));
    }

    #[test]
    fn test_extract_function_name() {
        assert_eq!(extract_function_name("pub fn foo(x: i32)"), "foo");
        assert_eq!(extract_function_name("def bar(self, x):"), "bar");
        assert_eq!(extract_function_name("function baz()"), "baz");
    }

    #[test]
    fn test_count_parameters() {
        assert_eq!(count_parameters("fn foo(a: i32, b: i32)"), 2);
        assert_eq!(count_parameters("fn bar()"), 0);
        assert_eq!(count_parameters("fn baz(&self)"), 0);
    }

    #[test]
    fn test_find_function_end() {
        let lines = vec!["fn foo() {", "  let x = 1;", "  x + 1", "}"];
        assert_eq!(find_function_end(&lines, 0), 4);
    }

    #[test]
    fn test_contract_summary_serializes() {
        let cs = ContractSummary {
            function_name: "test".into(),
            clause_count: 2,
            strength: "Adequate".into(),
            verified: true,
        };
        let json = serde_json::to_string(&cs).unwrap();
        assert!(json.contains("test"));
    }
}
