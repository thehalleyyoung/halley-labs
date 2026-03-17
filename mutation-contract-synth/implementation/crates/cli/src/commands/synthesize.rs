//! `mutspec synthesize` — contract synthesis command.
//!
//! Synthesizes formal contracts (preconditions, postconditions, invariants) from
//! mutation kill matrices.  Supports tier-based synthesis, multiple output
//! formats, and JML contract generation.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;

use shared_types::contracts::{
    Contract, ContractClause, ContractProvenance, ContractStrength, Specification, SynthesisTier,
};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::{MutantDescriptor, MutantStatus, MutationOperator};

use crate::config::CliConfig;
use crate::output::{
    format_contract_jml, format_contract_text, write_json, AlignedTable, CliOutputFormat, Colour,
    MarkdownTable, ProgressBar, SynthesisStats,
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct SynthesizeArgs {
    /// Path to a mutants JSON file (output of `mutspec mutate` or `mutspec analyze`).
    #[arg(required = true)]
    pub mutants_file: PathBuf,

    /// Output file for contracts (stdout if omitted).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short = 'f', long, value_enum)]
    pub format: Option<CliOutputFormat>,

    /// Maximum synthesis tier (1-3).
    #[arg(short, long)]
    pub tier: Option<u8>,

    /// Maximum clauses per contract.
    #[arg(long)]
    pub max_clauses: Option<usize>,

    /// Enable weakening pass.
    #[arg(long)]
    pub weaken: bool,

    /// Filter to specific function names (comma-separated).
    #[arg(long)]
    pub functions: Option<String>,

    /// Number of parallel jobs.
    #[arg(short = 'j', long)]
    pub jobs: Option<usize>,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,

    /// Emit JML-format contracts.
    #[arg(long)]
    pub jml: bool,

    /// Minimum mutation score to synthesize a contract (0.0-1.0).
    #[arg(long)]
    pub min_score: Option<f64>,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &SynthesizeArgs, cfg: &CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);
    let output_format = super::resolve_output_format(args.format, args.output.as_deref());
    let started = Instant::now();
    let _timing = super::TimingGuard::new("Contract synthesis");

    // --- Load mutants -----------------------------------------------------
    let mutants = load_mutants(&args.mutants_file)?;
    info!(
        "Loaded {} mutants from {}",
        mutants.len(),
        args.mutants_file.display()
    );

    if mutants.is_empty() {
        eprintln!("{} No mutants found in input file", colour.yellow("⚠"));
        return Ok(());
    }

    // --- Determine tier ---------------------------------------------------
    let tier = match args.tier {
        Some(n) => SynthesisTier::from_number(n)
            .ok_or_else(|| anyhow::anyhow!("Invalid synthesis tier: {n}. Use 1, 2, or 3"))?,
        None => {
            let max = cfg
                .synthesis()
                .enabled_tiers
                .iter()
                .copied()
                .max()
                .unwrap_or(1);
            SynthesisTier::from_number(max).unwrap_or(SynthesisTier::Tier1LatticeWalk)
        }
    };
    let max_clauses = args
        .max_clauses
        .unwrap_or(cfg.synthesis().lattice_max_steps as usize);
    let parallelism = args.jobs.unwrap_or_else(|| cfg.parallelism());

    info!(
        "Synthesis tier: {}, max clauses: {}",
        tier.tier_number(),
        max_clauses
    );

    // --- Group mutants by function ----------------------------------------
    let mut by_function = group_by_function(&mutants);

    // Apply function filter
    if let Some(ref filter) = args.functions {
        let names: Vec<&str> = filter.split(',').map(|s| s.trim()).collect();
        by_function.retain(|name, _| names.iter().any(|n| name.contains(n)));
        info!("Function filter: {} functions selected", by_function.len());
    }

    // Apply minimum score filter
    if let Some(min_score) = args.min_score {
        by_function.retain(|name, descs| {
            let score = compute_score(descs);
            if score < min_score {
                debug!("Skipping '{}': score {:.2} < {:.2}", name, score, min_score);
                false
            } else {
                true
            }
        });
    }

    if by_function.is_empty() {
        eprintln!(
            "{} No functions match the filter criteria",
            colour.yellow("⚠")
        );
        return Ok(());
    }

    // --- Synthesize contracts ---------------------------------------------
    let mut spec = Specification::new();
    let func_names: Vec<String> = by_function.keys().cloned().collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("Failed to build thread pool")?;

    let contracts: Vec<Contract> = pool.install(|| {
        func_names
            .par_iter()
            .filter_map(|name| {
                let descs = by_function.get(name)?;
                match synthesize_contract(name, descs, &tier, max_clauses, args.weaken) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        warn!("Failed to synthesize contract for '{}': {}", name, e);
                        None
                    }
                }
            })
            .collect()
    });

    for c in contracts {
        spec.add_contract(c);
    }

    // --- Statistics -------------------------------------------------------
    let elapsed = started.elapsed();
    let stats = SynthesisStats {
        functions_analyzed: by_function.len(),
        contracts_synthesized: spec.contracts.len(),
        total_clauses: spec.contracts.iter().map(|c| c.clause_count()).sum(),
        tier: format!("Tier {}", tier.tier_number()),
        elapsed_secs: elapsed.as_secs_f64(),
    };

    // --- Output -----------------------------------------------------------
    if args.jml {
        let output = format_jml_output(&spec);
        super::write_output(&output, args.output.as_deref())?;
    } else {
        let output = format_synthesis_output(&spec, &stats, output_format, &colour)?;
        super::write_output(&output, args.output.as_deref())?;
    }

    eprintln!(
        "{} Synthesized {} contracts ({} clauses) in {:.2}s",
        colour.green("✓"),
        spec.contracts.len(),
        stats.total_clauses,
        elapsed.as_secs_f64(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal logic
// ---------------------------------------------------------------------------

fn load_mutants(path: &PathBuf) -> Result<Vec<MutantDescriptor>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read mutants file: {}", path.display()))?;

    if let Ok(mutants) = serde_json::from_str::<Vec<MutantDescriptor>>(&content) {
        return Ok(mutants);
    }

    #[derive(Debug, Deserialize)]
    struct MutantEnvelope {
        mutants: Vec<MutantDescriptor>,
    }

    let envelope: MutantEnvelope = serde_json::from_str(&content).with_context(|| {
        format!(
            "Failed to parse mutants JSON from {} as either a bare mutant array or {{\"mutants\": [...]}}",
            path.display()
        )
    })?;
    Ok(envelope.mutants)
}

fn group_by_function(mutants: &[MutantDescriptor]) -> BTreeMap<String, Vec<&MutantDescriptor>> {
    let mut map: BTreeMap<String, Vec<&MutantDescriptor>> = BTreeMap::new();
    for m in mutants {
        // Group by file:startline as a proxy for function identity
        let key = format!(
            "fn@{}:{}",
            m.site.location.start.file.display(),
            m.site.location.start.line,
        );
        map.entry(key).or_default().push(m);
    }
    map
}

fn compute_score(mutants: &[&MutantDescriptor]) -> f64 {
    let total = mutants.len();
    if total == 0 {
        return 0.0;
    }
    let killed = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Killed))
        .count();
    let equivalent = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Equivalent))
        .count();
    let denom = total.saturating_sub(equivalent);
    if denom == 0 {
        1.0
    } else {
        killed as f64 / denom as f64
    }
}

fn synthesize_contract(
    func_name: &str,
    mutants: &[&MutantDescriptor],
    tier: &SynthesisTier,
    max_clauses: usize,
    weaken: bool,
) -> Result<Contract> {
    let start = Instant::now();
    let mut contract = Contract::new(func_name.to_string());

    let killed: Vec<&&MutantDescriptor> = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Killed))
        .collect();
    let alive: Vec<&&MutantDescriptor> = mutants
        .iter()
        .filter(|m| matches!(m.status, MutantStatus::Alive))
        .collect();

    // --- Tier 1: Lattice walk ------------------------------------------
    let preconditions = lattice_walk_preconditions(func_name, &killed);
    for clause in preconditions {
        if contract.clause_count() < max_clauses {
            contract.add_clause(clause);
        }
    }

    let postconditions = lattice_walk_postconditions(func_name, &killed, &alive);
    for clause in postconditions {
        if contract.clause_count() < max_clauses {
            contract.add_clause(clause);
        }
    }

    // --- Tier 2: Template-based ----------------------------------------
    if matches!(
        tier,
        SynthesisTier::Tier2Template | SynthesisTier::Tier3Fallback
    ) {
        let templates = template_based_clauses(func_name, &killed);
        for clause in templates {
            if contract.clause_count() < max_clauses {
                contract.add_clause(clause);
            }
        }
    }

    // --- Tier 3: Fallback / heuristic ----------------------------------
    if matches!(tier, SynthesisTier::Tier3Fallback) {
        let fallback = fallback_clauses(func_name, mutants);
        for clause in fallback {
            if contract.clause_count() < max_clauses {
                contract.add_clause(clause);
            }
        }
    }

    // --- Weakening pass ------------------------------------------------
    if weaken && contract.clause_count() > 1 {
        weaken_contract(&mut contract);
    }

    // --- Determine strength --------------------------------------------
    contract.strength = determine_strength(&contract, mutants);

    // --- Provenance ----------------------------------------------------
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let provenance = ContractProvenance::new(tier.clone()).with_time(elapsed_ms as f64);
    contract.add_provenance(provenance);

    debug!(
        "Contract for '{}': {} clauses, strength={}",
        func_name,
        contract.clause_count(),
        contract.strength.name()
    );

    Ok(contract)
}

/// Create a placeholder formula from a descriptive string.
fn text_formula(desc: impl Into<String>) -> Formula {
    Formula::atom(Predicate::gt(Term::var(desc), Term::constant(0)))
}

fn lattice_walk_preconditions(
    func_name: &str,
    killed: &[&&MutantDescriptor],
) -> Vec<ContractClause> {
    let mut clauses = Vec::new();

    // Analyse killed operator patterns to infer input constraints
    let has_arith = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Aor));
    let has_rel = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Ror));
    let has_const = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Crc));

    if has_arith {
        clauses.push(ContractClause::requires(text_formula(format!(
            "numeric_inputs_valid({})",
            func_name
        ))));
    }

    if has_rel {
        clauses.push(ContractClause::requires(text_formula(format!(
            "comparison_bounds_hold({})",
            func_name
        ))));
    }

    if has_const {
        clauses.push(ContractClause::requires(text_formula(format!(
            "constants_in_range({})",
            func_name
        ))));
    }

    clauses
}

fn lattice_walk_postconditions(
    func_name: &str,
    killed: &[&&MutantDescriptor],
    alive: &[&&MutantDescriptor],
) -> Vec<ContractClause> {
    let mut clauses = Vec::new();

    if killed.is_empty() {
        return clauses;
    }

    let score = if killed.len() + alive.len() > 0 {
        killed.len() as f64 / (killed.len() + alive.len()) as f64
    } else {
        0.0
    };

    // Strong postcondition if high kill rate
    if score >= 0.8 {
        clauses.push(ContractClause::ensures(text_formula(format!(
            "result == expected_output({})",
            func_name
        ))));
    } else if score >= 0.5 {
        clauses.push(ContractClause::ensures(text_formula(format!(
            "result_in_valid_range({})",
            func_name
        ))));
    }

    // Return value constraint
    let has_rvr = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Rvr));
    if has_rvr {
        clauses.push(ContractClause::ensures(text_formula(format!(
            "return_value_meaningful({})",
            func_name
        ))));
    }

    clauses
}

fn template_based_clauses(func_name: &str, killed: &[&&MutantDescriptor]) -> Vec<ContractClause> {
    let mut clauses = Vec::new();

    // Template: if array index mutations are killed, infer bounds checking
    let has_air = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Air));
    if has_air {
        clauses.push(ContractClause::requires(text_formula(format!(
            "0 <= index && index < array_length({})",
            func_name
        ))));
    }

    // Template: if logical connector mutations are killed, infer condition structure
    let has_lcr = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Lcr));
    if has_lcr {
        clauses.push(ContractClause::requires(text_formula(format!(
            "conditions_independent({})",
            func_name
        ))));
    }

    // Template: if statement deletion is killed, all statements are necessary
    let has_sdl = killed
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Sdl));
    if has_sdl {
        clauses.push(ContractClause::invariant(text_formula(format!(
            "all_statements_necessary({})",
            func_name
        ))));
    }

    clauses
}

fn fallback_clauses(func_name: &str, mutants: &[&MutantDescriptor]) -> Vec<ContractClause> {
    let mut clauses = Vec::new();

    // Fallback: generate catch-all invariant if many operators are covered
    let operator_count = mutants
        .iter()
        .map(|m| m.operator)
        .collect::<std::collections::HashSet<_>>()
        .len();

    if operator_count >= 5 {
        clauses.push(ContractClause::invariant(text_formula(format!(
            "function_wellformed({})",
            func_name
        ))));
    }

    // Fallback: boundary conditions
    let has_uoi = mutants
        .iter()
        .any(|m| matches!(m.operator, MutationOperator::Uoi));
    if has_uoi {
        clauses.push(ContractClause::requires(text_formula(format!(
            "no_overflow({})",
            func_name
        ))));
    }

    clauses
}

fn weaken_contract(contract: &mut Contract) {
    // Weakening: remove redundant or overly-specific clauses.
    // In a full implementation, this would use the SMT solver to check
    // implication between clauses. Here we apply structural heuristics:
    // remove duplicate kind clauses (keep the first of each kind).
    let mut seen_kinds = std::collections::HashSet::new();
    let clauses: Vec<ContractClause> = contract
        .clauses
        .drain(..)
        .filter(|c| {
            let key = format!("{}:{}", c.kind_name(), c.formula());
            seen_kinds.insert(key)
        })
        .collect();
    contract.clauses = clauses;
    debug!(
        "After weakening: {} clauses remain for '{}'",
        contract.clause_count(),
        contract.function_name
    );
}

fn determine_strength(contract: &Contract, mutants: &[&MutantDescriptor]) -> ContractStrength {
    let clause_count = contract.clause_count();
    let score = compute_score(mutants);

    if clause_count >= 3 && score >= 0.9 {
        ContractStrength::Strongest
    } else if clause_count >= 2 && score >= 0.7 {
        ContractStrength::Adequate
    } else if clause_count >= 1 {
        ContractStrength::Weak
    } else {
        ContractStrength::Trivial
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn format_jml_output(spec: &Specification) -> String {
    let mut buf = String::new();
    buf.push_str("// JML Contracts generated by MutSpec\n");
    buf.push_str("// ====================================\n\n");
    for contract in &spec.contracts {
        buf.push_str(&format!("// Contract for: {}\n", contract.function_name));
        buf.push_str(&format_contract_jml(contract));
        buf.push_str("\n\n");
    }
    buf
}

fn format_synthesis_output(
    spec: &Specification,
    stats: &SynthesisStats,
    format: CliOutputFormat,
    colour: &Colour,
) -> Result<String> {
    let mut buf = String::new();

    match format {
        CliOutputFormat::Json => {
            buf = serde_json::to_string_pretty(spec)?;
        }
        CliOutputFormat::Markdown => {
            buf.push_str("# Synthesized Contracts\n\n");
            buf.push_str(&format!("- Functions: {}\n", stats.functions_analyzed));
            buf.push_str(&format!("- Contracts: {}\n", stats.contracts_synthesized));
            buf.push_str(&format!("- Clauses: {}\n", stats.total_clauses));
            buf.push_str(&format!("- Tier: {}\n", stats.tier));
            buf.push_str(&format!("- Elapsed: {:.2}s\n\n", stats.elapsed_secs));

            for contract in &spec.contracts {
                buf.push_str(&format!("## `{}`\n\n", contract.function_name));
                buf.push_str(&format!("- Strength: {}\n", contract.strength.name()));
                buf.push_str(&format!("- Clauses: {}\n\n", contract.clause_count()));
                buf.push_str("```jml\n");
                buf.push_str(&contract.to_jml());
                buf.push_str("\n```\n\n");
            }
        }
        CliOutputFormat::Text => {
            buf.push_str(&stats.format_text(colour));
            buf.push('\n');
            for contract in &spec.contracts {
                buf.push_str(&format_contract_text(contract, colour));
                buf.push('\n');
            }

            // Summary table
            let mut tbl = AlignedTable::new(vec![
                "Function".into(),
                "Clauses".into(),
                "Strength".into(),
                "Verified".into(),
            ]);
            tbl.set_right_align(1);
            for c in &spec.contracts {
                tbl.add_row(vec![
                    c.function_name.clone(),
                    c.clause_count().to_string(),
                    c.strength.name().to_string(),
                    if c.verified {
                        colour.green("✓")
                    } else {
                        colour.yellow("–")
                    },
                ]);
            }
            buf.push_str(&format!("{tbl}"));
        }
        CliOutputFormat::Sarif => {
            let mut sarif =
                crate::output::SarifBuilder::new("mutspec-synth", env!("CARGO_PKG_VERSION"));
            for c in &spec.contracts {
                for clause in &c.clauses {
                    sarif.add_result(
                        "contract-clause",
                        "note",
                        format!(
                            "{} {} for function '{}'",
                            clause.kind_name(),
                            clause.formula(),
                            c.function_name
                        ),
                        "synthesized",
                        0,
                        0,
                    );
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
    use shared_types::errors::{SourceLocation, SpanInfo};
    use shared_types::operators::MutationSite;
    use std::path::PathBuf;

    fn make_mutant(op: MutationOperator, status: MutantStatus) -> MutantDescriptor {
        let loc = SourceLocation::new(PathBuf::from("test.rs"), 1, 1);
        let span = SpanInfo::new(loc.clone(), loc);
        let site = MutationSite::new(span, op, "+", "-");
        let mut m = MutantDescriptor::new(op, site);
        match status {
            MutantStatus::Killed => m.mark_killed(shared_types::operators::KillInfo::new(
                "test".to_string(),
                0.0,
            )),
            MutantStatus::Alive => {}
            MutantStatus::Equivalent => m.mark_equivalent("test equivalent"),
            _ => {}
        }
        m
    }

    #[test]
    fn test_compute_score() {
        let m1 = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let m2 = make_mutant(MutationOperator::Ror, MutantStatus::Alive);
        let refs: Vec<&MutantDescriptor> = vec![&m1, &m2];
        let score = compute_score(&refs);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_score_all_killed() {
        let m1 = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let m2 = make_mutant(MutationOperator::Ror, MutantStatus::Killed);
        let refs: Vec<&MutantDescriptor> = vec![&m1, &m2];
        let score = compute_score(&refs);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_determine_strength_strongest() {
        let m = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let refs = vec![&m];
        let mut c = Contract::new("test");
        c.add_clause(ContractClause::requires(text_formula("a")));
        c.add_clause(ContractClause::ensures(text_formula("b")));
        c.add_clause(ContractClause::invariant(text_formula("c")));
        let strength = determine_strength(&c, &refs);
        assert!(matches!(strength, ContractStrength::Strongest));
    }

    #[test]
    fn test_lattice_walk_preconditions() {
        let m = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let r = &m;
        let clauses = lattice_walk_preconditions("foo", &[&r]);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_synthesize_contract() {
        let m1 = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let m2 = make_mutant(MutationOperator::Ror, MutantStatus::Killed);
        let refs: Vec<&MutantDescriptor> = vec![&m1, &m2];
        let contract = synthesize_contract(
            "test_fn",
            &refs,
            &SynthesisTier::Tier1LatticeWalk,
            10,
            false,
        )
        .unwrap();
        assert!(!contract.function_name.is_empty());
        assert!(contract.clause_count() > 0);
    }

    #[test]
    fn test_load_mutants_from_bare_array() {
        let mutant = make_mutant(MutationOperator::Aor, MutantStatus::Killed);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mutants.json");
        std::fs::write(&path, serde_json::to_string(&vec![mutant]).unwrap()).unwrap();

        let mutants = load_mutants(&path).unwrap();
        assert_eq!(mutants.len(), 1);
    }

    #[test]
    fn test_load_mutants_from_wrapped_output() {
        #[derive(Serialize)]
        struct WrappedMutants<'a> {
            mutants: &'a [MutantDescriptor],
        }

        let mutant = make_mutant(MutationOperator::Ror, MutantStatus::Alive);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mutants.json");
        let wrapped = WrappedMutants {
            mutants: std::slice::from_ref(&mutant),
        };
        std::fs::write(&path, serde_json::to_string(&wrapped).unwrap()).unwrap();

        let mutants = load_mutants(&path).unwrap();
        assert_eq!(mutants.len(), 1);
        assert!(matches!(mutants[0].operator, MutationOperator::Ror));
    }

    #[test]
    fn test_json_synthesis_output_is_plain_specification() {
        let mut spec = Specification::new();
        spec.add_contract(Contract::new("abs"));
        let stats = SynthesisStats {
            functions_analyzed: 1,
            contracts_synthesized: 1,
            total_clauses: 0,
            tier: "Tier 1".to_string(),
            elapsed_secs: 0.0,
        };

        let output =
            format_synthesis_output(&spec, &stats, CliOutputFormat::Json, &Colour::new(true))
                .unwrap();
        let parsed: Specification = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.contracts.len(), 1);
        assert_eq!(parsed.contracts[0].function_name, "abs");
    }
}
