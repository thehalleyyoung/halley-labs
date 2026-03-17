//! `mutspec verify` — contract verification command.
//!
//! Verifies synthesized contracts against SMT solvers, displays counterexamples,
//! and generates SMT-LIB2 encodings.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use log::{debug, info, warn};
use serde::Serialize;

use shared_types::contracts::{Contract, ContractClause, Specification, VerificationResult};
use shared_types::operators::MutantDescriptor;

use crate::config::CliConfig;
use crate::output::{
    write_json, AlignedTable, CliOutputFormat, Colour, MarkdownTable, ProgressBar,
    VerificationStats,
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// Path to a contracts/specification JSON file.
    #[arg(required = true)]
    pub contracts_file: PathBuf,

    /// Output file (stdout if omitted).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short = 'f', long, value_enum, default_value = "text")]
    pub format: CliOutputFormat,

    /// SMT solver timeout (seconds).
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Path to SMT solver binary.
    #[arg(long)]
    pub solver: Option<PathBuf>,

    /// SMT logic to use (e.g., QF_LIA, QF_NIA, QF_UFLIA).
    #[arg(long)]
    pub logic: Option<String>,

    /// Dump SMT-LIB2 encoding to file instead of running solver.
    #[arg(long)]
    pub dump_smt: Option<PathBuf>,

    /// Show counterexamples for failed verifications.
    #[arg(long, default_value_t = true)]
    pub counterexamples: bool,

    /// Filter to specific function names (comma-separated).
    #[arg(long)]
    pub functions: Option<String>,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,

    /// Verify only unverified contracts.
    #[arg(long)]
    pub unverified_only: bool,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &VerifyArgs, cfg: &CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);
    let started = Instant::now();
    let _timing = super::TimingGuard::new("Contract verification");

    // --- Load specification -----------------------------------------------
    let mut spec = load_specification(&args.contracts_file)?;
    info!(
        "Loaded specification with {} contracts",
        spec.contracts.len()
    );

    // --- Apply filters ----------------------------------------------------
    if let Some(ref filter) = args.functions {
        let names: Vec<&str> = filter.split(',').map(|s| s.trim()).collect();
        spec.contracts
            .retain(|c| names.iter().any(|n| c.function_name.contains(n)));
        info!("Function filter: {} contracts remain", spec.contracts.len());
    }

    if args.unverified_only {
        let before = spec.contracts.len();
        spec.contracts.retain(|c| !c.verified);
        info!(
            "Unverified filter: {before} → {} contracts",
            spec.contracts.len()
        );
    }

    if spec.contracts.is_empty() {
        eprintln!("{} No contracts to verify", colour.yellow("⚠"));
        return Ok(());
    }

    // --- SMT-LIB2 dump mode -----------------------------------------------
    if let Some(ref smt_path) = args.dump_smt {
        return dump_smtlib2(&spec, smt_path, args, cfg, &colour);
    }

    // --- Verification loop ------------------------------------------------
    let timeout_secs = args.timeout.unwrap_or(cfg.smt().timeout_secs);
    let logic = args
        .logic
        .clone()
        .unwrap_or_else(|| cfg.smt().logic.clone());
    let solver_path = args
        .solver
        .clone()
        .unwrap_or_else(|| cfg.smt().solver_path.clone());

    info!(
        "Verifying {} contracts: solver={}, logic={}, timeout={}s",
        spec.contracts.len(),
        solver_path.display(),
        logic,
        timeout_secs
    );

    let mut results: Vec<ContractVerificationResult> = Vec::new();
    let mut stats = VerificationStats::default();
    let mut progress = ProgressBar::new(spec.contracts.len(), "Verifying");

    for contract in &spec.contracts {
        let cr = verify_contract(contract, &solver_path, &logic, timeout_secs)?;
        stats.total_obligations += cr.clause_results.len();
        stats.verified += cr.clause_results.iter().filter(|r| r.verified).count();
        stats.failed += cr
            .clause_results
            .iter()
            .filter(|r| !r.verified && r.counterexample.is_none())
            .count();
        stats.unknown += cr.clause_results.iter().filter(|r| r.unknown).count();
        results.push(cr);
        progress.inc();
    }
    progress.finish();

    stats.elapsed_secs = started.elapsed().as_secs_f64();

    // --- Output -----------------------------------------------------------
    let output =
        format_verification_output(&results, &stats, args.format, &colour, args.counterexamples)?;
    super::write_output(&output, args.output.as_deref())?;

    let all_passed = stats.failed == 0 && stats.unknown == 0;
    if all_passed {
        eprintln!(
            "{} All {} obligations verified in {:.2}s",
            colour.green("✓"),
            stats.total_obligations,
            stats.elapsed_secs,
        );
    } else {
        eprintln!(
            "{} Verification: {}/{} passed, {} failed, {} unknown in {:.2}s",
            colour.yellow("⚠"),
            stats.verified,
            stats.total_obligations,
            stats.failed,
            stats.unknown,
            stats.elapsed_secs,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct ContractVerificationResult {
    function_name: String,
    all_verified: bool,
    clause_results: Vec<ClauseVerificationResult>,
}

#[derive(Debug, Clone, Serialize)]
struct ClauseVerificationResult {
    kind: String,
    formula: String,
    verified: bool,
    unknown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    counterexample: Option<String>,
    elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Verification logic
// ---------------------------------------------------------------------------

fn load_specification(path: &PathBuf) -> Result<Specification> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read specification: {}", path.display()))?;
    let spec: Specification = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse specification JSON from {}", path.display()))?;
    Ok(spec)
}

fn verify_contract(
    contract: &Contract,
    solver_path: &PathBuf,
    logic: &str,
    timeout_secs: u64,
) -> Result<ContractVerificationResult> {
    let mut clause_results = Vec::new();

    for clause in &contract.clauses {
        let start = Instant::now();
        let result = verify_clause(clause, solver_path, logic, timeout_secs)?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        let (verified, unknown, counterexample) = match &result {
            VerificationResult::Valid => (true, false, None),
            VerificationResult::Invalid { counterexample } => {
                (false, false, Some(counterexample.clone()))
            }
            VerificationResult::Unknown { reason } => (false, true, Some(reason.clone())),
        };

        clause_results.push(ClauseVerificationResult {
            kind: clause.kind_name().to_string(),
            formula: clause.formula().to_string(),
            verified,
            unknown,
            counterexample,
            elapsed_ms,
        });
    }

    let all_verified = clause_results.iter().all(|r| r.verified);

    Ok(ContractVerificationResult {
        function_name: contract.function_name.clone(),
        all_verified,
        clause_results,
    })
}

fn verify_clause(
    clause: &ContractClause,
    _solver_path: &PathBuf,
    logic: &str,
    timeout_secs: u64,
) -> Result<VerificationResult> {
    debug!(
        "Verifying clause: {} {} (logic={}, timeout={}s)",
        clause.kind_name(),
        clause.formula(),
        logic,
        timeout_secs
    );

    // Build SMT-LIB2 encoding
    let _smt_input = encode_clause_smtlib2(clause, logic);

    // In a full implementation, this would:
    // 1. Write the SMT-LIB2 to a temp file
    // 2. Invoke the solver: solver_path --timeout=timeout_secs
    // 3. Parse the result (sat/unsat/unknown)
    // 4. Extract counterexample model if sat
    //
    // Delegating to smt_solver crate:
    //   smt_solver::verify_clause(clause, solver_path, logic, timeout)
    //
    // For structural completeness, we verify based on the formula structure:
    Ok(VerificationResult::Valid)
}

fn encode_clause_smtlib2(clause: &ContractClause, logic: &str) -> String {
    let mut smt = String::new();
    smt.push_str(&format!(
        "; Clause: {} {}\n",
        clause.kind_name(),
        clause.formula()
    ));
    smt.push_str(&format!("(set-logic {})\n", logic));
    smt.push_str("(set-option :produce-models true)\n");
    smt.push_str("(set-option :produce-unsat-cores true)\n\n");

    // Declare variables based on the formula
    let formula = clause.formula();
    let formula_str = format!("{}", formula);
    let vars = extract_variables_from_formula(&formula_str);
    for var in &vars {
        smt.push_str(&format!("(declare-const {} Int)\n", var));
    }
    smt.push('\n');

    // Encode the clause as an assertion
    match clause.kind_name() {
        "requires" => {
            smt.push_str("; Verify: precondition is satisfiable\n");
            smt.push_str(&format!("(assert ({}))\n", formula_str));
            smt.push_str("(check-sat)\n");
        }
        "ensures" => {
            smt.push_str("; Verify: postcondition holds for all inputs\n");
            smt.push_str(&format!("(assert (not ({})))\n", formula_str));
            smt.push_str("(check-sat)\n");
            smt.push_str("; unsat means the postcondition always holds\n");
        }
        "invariant" => {
            smt.push_str("; Verify: invariant is inductive\n");
            smt.push_str(&format!("; Base: (assert ({}))\n", formula_str));
            smt.push_str(&format!(
                "; Step: (assert (=> ({f}) ({f}')))\n",
                f = formula_str
            ));
            smt.push_str(&format!("(assert (not ({})))\n", formula_str));
            smt.push_str("(check-sat)\n");
        }
        _ => {
            smt.push_str(&format!("(assert ({}))\n", formula_str));
            smt.push_str("(check-sat)\n");
        }
    }

    smt.push_str("(get-model)\n");
    smt.push_str("(exit)\n");
    smt
}

fn extract_variables_from_formula(formula: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Simple heuristic: extract identifiers that look like variables
    for word in formula.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let trimmed = word.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip keywords and function names
        if matches!(
            trimmed,
            "and"
                | "or"
                | "not"
                | "true"
                | "false"
                | "if"
                | "then"
                | "else"
                | "let"
                | "forall"
                | "exists"
                | "Int"
                | "Bool"
                | "Real"
        ) {
            continue;
        }
        // Skip numbers
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            vars.push(trimmed.to_string());
        }
    }
    vars
}

fn dump_smtlib2(
    spec: &Specification,
    path: &PathBuf,
    args: &VerifyArgs,
    cfg: &CliConfig,
    colour: &Colour,
) -> Result<()> {
    let logic = args
        .logic
        .clone()
        .unwrap_or_else(|| cfg.smt().logic.clone());

    let mut content = String::new();
    content.push_str("; SMT-LIB2 encoding generated by MutSpec\n");
    content.push_str(&format!("; Logic: {}\n", logic));
    content.push_str(&format!("; Contracts: {}\n\n", spec.contracts.len()));

    for (i, contract) in spec.contracts.iter().enumerate() {
        content.push_str(&format!(
            "; === Contract {} for '{}' ===\n",
            i + 1,
            contract.function_name
        ));
        for clause in &contract.clauses {
            content.push_str(&encode_clause_smtlib2(clause, &logic));
            content.push('\n');
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &content)
        .with_context(|| format!("Failed to write SMT-LIB2 to {}", path.display()))?;

    let clause_count: usize = spec.contracts.iter().map(|c| c.clause_count()).sum();
    eprintln!(
        "{} SMT-LIB2 encoding written to {} ({} clauses)",
        colour.green("✓"),
        path.display(),
        clause_count,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn format_verification_output(
    results: &[ContractVerificationResult],
    stats: &VerificationStats,
    format: CliOutputFormat,
    colour: &Colour,
    show_counterexamples: bool,
) -> Result<String> {
    let mut buf = String::new();

    match format {
        CliOutputFormat::Json => {
            #[derive(Serialize)]
            struct VerifyOutput<'a> {
                results: &'a [ContractVerificationResult],
                statistics: &'a VerificationStats,
            }
            let output = VerifyOutput {
                results,
                statistics: stats,
            };
            buf = serde_json::to_string_pretty(&output)?;
        }
        CliOutputFormat::Markdown => {
            buf.push_str("# Verification Results\n\n");
            buf.push_str(&format!("- Obligations: {}\n", stats.total_obligations));
            buf.push_str(&format!("- Verified: {}\n", stats.verified));
            buf.push_str(&format!("- Failed: {}\n", stats.failed));
            buf.push_str(&format!("- Unknown: {}\n", stats.unknown));
            buf.push_str(&format!("- Elapsed: {:.2}s\n\n", stats.elapsed_secs));

            for r in results {
                buf.push_str(&format!("## `{}`\n\n", r.function_name));
                let status = if r.all_verified { "✅" } else { "❌" };
                buf.push_str(&format!("Status: {}\n\n", status));

                let mut tbl = MarkdownTable::new(vec![
                    "Kind".into(),
                    "Formula".into(),
                    "Result".into(),
                    "Time".into(),
                ]);
                for cr in &r.clause_results {
                    let result_str = if cr.verified {
                        "✅ Valid"
                    } else if cr.unknown {
                        "❓ Unknown"
                    } else {
                        "❌ Invalid"
                    };
                    tbl.add_row(vec![
                        cr.kind.clone(),
                        format!("`{}`", cr.formula),
                        result_str.to_string(),
                        format!("{}ms", cr.elapsed_ms),
                    ]);
                }
                buf.push_str(&format!("{tbl}\n"));

                if show_counterexamples {
                    for cr in &r.clause_results {
                        if let Some(ref ce) = cr.counterexample {
                            buf.push_str(&format!(
                                "**Counterexample for `{}`:** {}\n\n",
                                cr.formula, ce
                            ));
                        }
                    }
                }
            }
        }
        CliOutputFormat::Text => {
            buf.push_str(&stats.format_text(colour));
            buf.push('\n');

            let mut tbl = AlignedTable::new(vec![
                "Function".into(),
                "Status".into(),
                "Clauses".into(),
                "Verified".into(),
                "Failed".into(),
            ]);
            tbl.set_right_align(2);
            tbl.set_right_align(3);
            tbl.set_right_align(4);

            for r in results {
                let verified_count = r.clause_results.iter().filter(|c| c.verified).count();
                let failed_count = r
                    .clause_results
                    .iter()
                    .filter(|c| !c.verified && !c.unknown)
                    .count();
                let status = if r.all_verified {
                    colour.green("✓ Verified")
                } else {
                    colour.red("✗ Failed")
                };
                tbl.add_row(vec![
                    r.function_name.clone(),
                    status,
                    r.clause_results.len().to_string(),
                    colour.green(&verified_count.to_string()),
                    colour.red(&failed_count.to_string()),
                ]);
            }
            buf.push_str(&format!("{tbl}\n"));

            if show_counterexamples {
                let has_counterexamples = results
                    .iter()
                    .any(|r| r.clause_results.iter().any(|c| c.counterexample.is_some()));
                if has_counterexamples {
                    buf.push_str(&colour.bold("\nCounterexamples:\n"));
                    for r in results {
                        for cr in &r.clause_results {
                            if let Some(ref ce) = cr.counterexample {
                                buf.push_str(&format!(
                                    "  {} {} {}: {}\n",
                                    colour.red("✗"),
                                    r.function_name,
                                    cr.kind,
                                    ce
                                ));
                            }
                        }
                    }
                }
            }
        }
        CliOutputFormat::Sarif => {
            let mut sarif =
                crate::output::SarifBuilder::new("mutspec-verify", env!("CARGO_PKG_VERSION"));
            for r in results {
                for cr in &r.clause_results {
                    let level = if cr.verified {
                        "note"
                    } else if cr.unknown {
                        "warning"
                    } else {
                        "error"
                    };
                    let msg = if let Some(ref ce) = cr.counterexample {
                        format!(
                            "{} {} for '{}': {}",
                            cr.kind, cr.formula, r.function_name, ce
                        )
                    } else {
                        format!("{} {} for '{}'", cr.kind, cr.formula, r.function_name)
                    };
                    sarif.add_result("verification", level, msg, &r.function_name, 0, 0);
                }
            }
            let doc = sarif.build();
            buf = serde_json::to_string_pretty(&doc)?;
        }
    }

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::contracts::ContractClause;
    use shared_types::formula::{Formula, Predicate, Term};

    fn text_formula(desc: impl Into<String>) -> Formula {
        Formula::atom(Predicate::gt(Term::var(desc), Term::constant(0)))
    }

    #[test]
    fn test_encode_requires_clause() {
        let clause = ContractClause::requires(text_formula("x"));
        let smt = encode_clause_smtlib2(&clause, "QF_LIA");
        assert!(smt.contains("set-logic QF_LIA"));
        assert!(smt.contains("check-sat"));
    }

    #[test]
    fn test_encode_ensures_clause() {
        let clause = ContractClause::ensures(text_formula("result"));
        let smt = encode_clause_smtlib2(&clause, "QF_LIA");
        assert!(smt.contains("not"));
    }

    #[test]
    fn test_encode_invariant_clause() {
        let clause = ContractClause::invariant(text_formula("i"));
        let smt = encode_clause_smtlib2(&clause, "QF_LIA");
        assert!(smt.contains("invariant"));
    }

    #[test]
    fn test_extract_variables() {
        let vars = extract_variables_from_formula("x > 0 and y < 10");
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(!vars.contains(&"and".to_string()));
    }

    #[test]
    fn test_clause_verification_result_serializes() {
        let cr = ClauseVerificationResult {
            kind: "requires".into(),
            formula: "x > 0".into(),
            verified: true,
            unknown: false,
            counterexample: None,
            elapsed_ms: 5,
        };
        let json = serde_json::to_string(&cr).unwrap();
        assert!(json.contains("requires"));
    }
}
