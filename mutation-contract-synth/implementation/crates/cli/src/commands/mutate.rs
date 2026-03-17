//! `mutspec mutate` — mutation generation command.
//!
//! Generates mutants for the given source files using the configured operators.
//! Supports operator filtering, line-range restriction, and multiple output
//! formats.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::Serialize;

use shared_types::config::MutSpecConfig;
use shared_types::errors::{SourceLocation, SpanInfo};
use shared_types::operators::{
    MutantDescriptor, MutantId, MutantStatus, MutationOperator, MutationSite,
};

use crate::config::CliConfig;
use crate::output::{
    mutant_table, write_json, AlignedTable, CliOutputFormat, Colour, MarkdownTable, MutationStats,
    ProgressBar,
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct MutateArgs {
    /// Source files or directories to mutate.
    #[arg(required = true)]
    pub input: Vec<PathBuf>,

    /// Output file (stdout if omitted).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short = 'f', long, value_enum)]
    pub format: Option<CliOutputFormat>,

    /// Mutation operators to use (comma-separated mnemonics).
    #[arg(long)]
    pub operators: Option<String>,

    /// Maximum mutants per function (0 = unlimited).
    #[arg(long)]
    pub max_mutants: Option<usize>,

    /// Restrict mutations to a line range (START-END).
    #[arg(long)]
    pub lines: Option<String>,

    /// Only list mutation sites without generating full descriptors.
    #[arg(long)]
    pub dry_run: bool,

    /// Number of parallel jobs.
    #[arg(short = 'j', long)]
    pub jobs: Option<usize>,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,

    /// Include operator statistics in the output.
    #[arg(long)]
    pub stats: bool,

    /// List all available operators and exit.
    #[arg(long)]
    pub list_operators: bool,

    /// Sampling strategy: "uniform" or "weighted".
    #[arg(long)]
    pub sampling: Option<String>,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &MutateArgs, cfg: &CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);
    let output_format = super::resolve_output_format(args.format, args.output.as_deref());

    // Handle --list-operators early exit
    if args.list_operators {
        return list_all_operators(&colour);
    }

    let started = Instant::now();
    let _timing = super::TimingGuard::new("Mutation generation");

    // --- Build effective config ------------------------------------------
    let mut config = cfg.clone();
    if let Some(ref ops_str) = args.operators {
        let ops = super::parse_operators(ops_str)?;
        config = config.with_operators(ops);
    }
    if let Some(max) = args.max_mutants {
        config = config.with_max_mutants(max);
    }
    if let Some(j) = args.jobs {
        config = config.with_parallelism(j);
    }

    // --- Collect source files --------------------------------------------
    let source_files = collect_mutate_inputs(&args.input)?;
    if source_files.is_empty() {
        anyhow::bail!("No source files found");
    }
    info!("Collected {} source files", source_files.len());

    // --- Parse functions -------------------------------------------------
    let functions = parse_functions(&source_files)?;
    info!("Found {} functions", functions.len());

    // --- Generate mutants ------------------------------------------------
    let operators = config.mutation().enabled_operators();
    let max_per_fn = config.mutation().max_mutants_per_site;
    let parallelism = config.parallelism();

    let mut mutants = generate_mutants(&functions, &operators, max_per_fn, parallelism)?;

    // --- Apply line filter -----------------------------------------------
    if let Some(ref range_str) = args.lines {
        let (start, end) = super::parse_line_range(range_str)?;
        let before = mutants.len();
        mutants = super::filter_mutants_by_line(mutants, start, end);
        info!(
            "Line filter {start}-{end}: {before} → {} mutants",
            mutants.len()
        );
    }

    // --- Dry run mode ----------------------------------------------------
    if args.dry_run {
        return print_dry_run(&mutants, &colour, output_format);
    }

    // --- Output ----------------------------------------------------------
    let elapsed = started.elapsed();
    let output_text = format_mutant_output(&mutants, output_format, &colour, args.stats)?;
    super::write_output(&output_text, args.output.as_deref())?;

    // Summary to stderr
    eprintln!(
        "{} Generated {} mutants in {:.2}s using {} operators",
        colour.green("✓"),
        mutants.len(),
        elapsed.as_secs_f64(),
        operators.len(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn collect_mutate_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let extensions = ["rs", "c", "cpp", "java", "py", "js", "ts", "go"];
    let ext_refs: Vec<&str> = extensions.iter().copied().collect();
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            let mut df = super::collect_source_files(input, &ext_refs)?;
            files.append(&mut df);
        } else {
            warn!("Skipping non-existent path: {}", input.display());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

#[derive(Debug, Clone)]
struct ParsedFunction {
    name: String,
    file: PathBuf,
    start_line: usize,
    end_line: usize,
    body: String,
}

fn parse_functions(files: &[PathBuf]) -> Result<Vec<ParsedFunction>> {
    let mut functions = Vec::new();
    for file in files {
        let source = super::read_source_file(file)?;
        let lines: Vec<&str> = source.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if is_function_def(line) {
                let name = extract_fn_name(line);
                let start = i + 1;
                let end = find_fn_end(&lines, i);
                let body = lines[i..end].join("\n");
                functions.push(ParsedFunction {
                    name,
                    file: file.clone(),
                    start_line: start,
                    end_line: end,
                    body,
                });
                i = end;
            }
            i += 1;
        }
    }
    Ok(functions)
}

fn is_function_def(line: &str) -> bool {
    let t = line.trim();
    if (t.starts_with("fn ") || t.starts_with("pub fn ") || t.starts_with("pub(crate) fn "))
        && t.contains('(')
    {
        return true;
    }
    if t.starts_with("def ") && t.contains('(') {
        return true;
    }
    if (t.starts_with("function ") || t.starts_with("export function ")) && t.contains('(') {
        return true;
    }
    if t.contains('(') && !t.starts_with("//") && !t.starts_with("if ") && !t.starts_with("for ") {
        if let Some(pos) = t.find('(') {
            let before = &t[..pos];
            if before.split_whitespace().count() >= 2 {
                return true;
            }
        }
    }
    false
}

fn extract_fn_name(line: &str) -> String {
    let t = line.trim();
    if let Some(rest) = t
        .strip_prefix("pub(crate) fn ")
        .or_else(|| t.strip_prefix("pub fn "))
        .or_else(|| t.strip_prefix("fn "))
        .or_else(|| t.strip_prefix("def "))
        .or_else(|| t.strip_prefix("export function "))
        .or_else(|| t.strip_prefix("function "))
    {
        return rest
            .split('(')
            .next()
            .unwrap_or("unknown")
            .trim()
            .to_string();
    }
    if let Some(pos) = t.find('(') {
        let before = &t[..pos];
        if let Some(name) = before.split_whitespace().last() {
            return name.trim_start_matches('*').to_string();
        }
    }
    "unknown".to_string()
}

fn find_fn_end(lines: &[&str], start: usize) -> usize {
    let mut depth = 0i32;
    let mut found_brace = false;
    for i in start..lines.len() {
        for ch in lines[i].chars() {
            if ch == '{' {
                depth += 1;
                found_brace = true;
            } else if ch == '}' {
                depth -= 1;
                if found_brace && depth == 0 {
                    return i + 1;
                }
            }
        }
    }
    if !found_brace && start < lines.len() {
        let base = lines[start].len() - lines[start].trim_start().len();
        for i in (start + 1)..lines.len() {
            if lines[i].trim().is_empty() {
                continue;
            }
            let indent = lines[i].len() - lines[i].trim_start().len();
            if indent <= base {
                return i;
            }
        }
    }
    lines.len()
}

fn generate_mutants(
    functions: &[ParsedFunction],
    operators: &[MutationOperator],
    max_per_fn: usize,
    parallelism: usize,
) -> Result<Vec<MutantDescriptor>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("Failed to build thread pool")?;

    let mutants: Vec<MutantDescriptor> = pool.install(|| {
        functions
            .par_iter()
            .flat_map(|func| generate_fn_mutants(func, operators, max_per_fn))
            .collect()
    });

    Ok(mutants)
}

fn generate_fn_mutants(
    func: &ParsedFunction,
    operators: &[MutationOperator],
    max: usize,
) -> Vec<MutantDescriptor> {
    let mut mutants = Vec::new();

    for &op in operators {
        if mutants.len() >= max {
            break;
        }
        let pairs = operator_replacements(op);
        for (orig, repl) in &pairs {
            if mutants.len() >= max {
                break;
            }
            // Check if the original pattern exists in the function body
            if func.body.contains(orig.as_str()) {
                let loc_start = SourceLocation::new(func.file.clone(), func.start_line, 1);
                let loc_end = SourceLocation::new(func.file.clone(), func.end_line, 1);
                let span = SpanInfo::new(loc_start, loc_end);
                let site = MutationSite::new(span, op, orig.clone(), repl.clone());
                mutants.push(MutantDescriptor::new(op, site));
            }
        }
    }

    debug!("Function '{}': {} mutants", func.name, mutants.len());
    mutants
}

fn operator_replacements(op: MutationOperator) -> Vec<(String, String)> {
    match op {
        MutationOperator::Aor => vec![
            ("+".into(), "-".into()),
            ("-".into(), "+".into()),
            ("*".into(), "/".into()),
            ("/".into(), "*".into()),
            ("%".into(), "+".into()),
        ],
        MutationOperator::Ror => vec![
            ("<".into(), "<=".into()),
            ("<=".into(), "<".into()),
            (">".into(), ">=".into()),
            (">=".into(), ">".into()),
            ("==".into(), "!=".into()),
            ("!=".into(), "==".into()),
        ],
        MutationOperator::Lcr => vec![("&&".into(), "||".into()), ("||".into(), "&&".into())],
        MutationOperator::Uoi => vec![("x".into(), "-x".into()), ("x".into(), "!x".into())],
        MutationOperator::Abs => vec![
            ("x".into(), "abs(x)".into()),
            ("x".into(), "-abs(x)".into()),
        ],
        MutationOperator::Cor => vec![
            ("&".into(), "|".into()),
            ("|".into(), "&".into()),
            ("^".into(), "&".into()),
        ],
        MutationOperator::Sdl => vec![("<stmt>".into(), "/* deleted */".into())],
        MutationOperator::Rvr => vec![("return".into(), "return 0; //".into())],
        MutationOperator::Crc => vec![("0".into(), "1".into()), ("1".into(), "0".into())],
        MutationOperator::Air => vec![
            ("[i]".into(), "[i+1]".into()),
            ("[i]".into(), "[i-1]".into()),
        ],
        MutationOperator::Osw => vec![("{s1;s2}".into(), "{s2;s1}".into())],
        MutationOperator::Bcn => vec![
            ("break".into(), "continue".into()),
            ("continue".into(), "break".into()),
        ],
    }
}

fn list_all_operators(colour: &Colour) -> Result<()> {
    println!("{}", colour.bold("Available Mutation Operators"));
    println!();
    let mut tbl = AlignedTable::new(vec![
        "Mnemonic".into(),
        "Description".into(),
        "Category".into(),
    ]);

    let descriptions = [
        ("AOR", "Arithmetic Operator Replacement", "Expression"),
        ("ROR", "Relational Operator Replacement", "Expression"),
        ("LCR", "Logical Connector Replacement", "Expression"),
        ("UOI", "Unary Operator Insertion", "Expression"),
        ("ABS", "Absolute Value Insertion", "Expression"),
        (
            "COR",
            "Bitwise/Conditional Operator Replacement",
            "Expression",
        ),
        ("SDL", "Statement Deletion", "Statement"),
        ("RVR", "Return Value Replacement", "Statement"),
        ("CRC", "Constant Replacement", "Literal"),
        ("AIR", "Array Index Replacement", "Reference"),
        ("OSW", "Operator Statement Swap", "Statement"),
        ("BCN", "Break/Continue Interchange", "Control Flow"),
    ];

    for (mnemonic, desc, category) in &descriptions {
        tbl.add_row(vec![
            colour.cyan(mnemonic),
            desc.to_string(),
            colour.dim(category),
        ]);
    }

    println!("{tbl}");
    println!();
    println!(
        "Standard set: {}",
        colour.bold(
            &MutationOperator::standard_set()
                .iter()
                .map(|o| o.mnemonic())
                .collect::<Vec<_>>()
                .join(", ")
        )
    );
    println!(
        "All operators: {}",
        MutationOperator::all()
            .iter()
            .map(|o| o.mnemonic())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn print_dry_run(
    mutants: &[MutantDescriptor],
    colour: &Colour,
    format: CliOutputFormat,
) -> Result<()> {
    eprintln!(
        "{} Dry run: {} mutation sites found",
        colour.yellow("⚠"),
        mutants.len()
    );

    match format {
        CliOutputFormat::Json => {
            #[derive(Serialize)]
            struct DryRunSite {
                operator: String,
                original: String,
                replacement: String,
                line: usize,
                file: String,
            }
            let sites: Vec<DryRunSite> = mutants
                .iter()
                .map(|m| DryRunSite {
                    operator: m.operator.mnemonic().to_string(),
                    original: m.site.original.clone(),
                    replacement: m.site.replacement.clone(),
                    line: m.site.location.start.line,
                    file: m.site.location.start.file.display().to_string(),
                })
                .collect();
            let json = serde_json::to_string_pretty(&sites)?;
            println!("{json}");
        }
        _ => {
            let tbl = mutant_table(mutants, colour);
            println!("{tbl}");
        }
    }
    Ok(())
}

fn format_mutant_output(
    mutants: &[MutantDescriptor],
    format: CliOutputFormat,
    colour: &Colour,
    include_stats: bool,
) -> Result<String> {
    let mut buf = String::new();

    match format {
        CliOutputFormat::Json => {
            if include_stats {
                #[derive(Serialize)]
                struct MutantOutput<'a> {
                    mutants: &'a [MutantDescriptor],
                    statistics: MutationStats,
                }

                let output = MutantOutput {
                    mutants,
                    statistics: MutationStats::from_descriptors(mutants),
                };
                buf = serde_json::to_string_pretty(&output)?;
            } else {
                buf = serde_json::to_string_pretty(mutants)?;
            }
        }
        CliOutputFormat::Markdown => {
            buf.push_str("# Generated Mutants\n\n");
            let mut tbl = MarkdownTable::new(vec![
                "ID".into(),
                "Operator".into(),
                "Original".into(),
                "Replacement".into(),
                "File".into(),
                "Line".into(),
            ]);
            for m in mutants {
                tbl.add_row(vec![
                    format!("`{}`", m.id.short()),
                    m.operator.mnemonic().to_string(),
                    format!("`{}`", m.site.original),
                    format!("`{}`", m.site.replacement),
                    m.site.location.start.file.display().to_string(),
                    m.site.location.start.line.to_string(),
                ]);
            }
            buf.push_str(&format!("{tbl}"));
            if include_stats {
                let stats = MutationStats::from_descriptors(mutants);
                buf.push_str(&format!("\n**Total:** {}\n", stats.total_mutants));
            }
        }
        CliOutputFormat::Text => {
            let tbl = mutant_table(mutants, colour);
            buf.push_str(&format!("{tbl}"));
            if include_stats {
                buf.push('\n');
                let stats = MutationStats::from_descriptors(mutants);
                buf.push_str(&stats.format_text(colour));
            }
        }
        CliOutputFormat::Sarif => {
            let mut sarif =
                crate::output::SarifBuilder::new("mutspec-mutate", env!("CARGO_PKG_VERSION"));
            for m in mutants {
                sarif.add_result(
                    m.operator.mnemonic(),
                    "note",
                    format!("Mutation: {} → {}", m.site.original, m.site.replacement),
                    m.site.location.start.file.display().to_string(),
                    m.site.location.start.line,
                    m.site.location.start.column,
                );
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

    #[test]
    fn test_is_function_def() {
        assert!(is_function_def("fn main() {"));
        assert!(is_function_def("pub fn add(a: i32, b: i32) -> i32 {"));
        assert!(is_function_def("def hello():"));
        assert!(!is_function_def("// fn comment"));
        assert!(!is_function_def("if foo() {"));
    }

    #[test]
    fn test_extract_fn_name() {
        assert_eq!(extract_fn_name("fn main()"), "main");
        assert_eq!(extract_fn_name("pub fn add(a: i32)"), "add");
        assert_eq!(extract_fn_name("def foo(self):"), "foo");
    }

    #[test]
    fn test_operator_replacements() {
        let r = operator_replacements(MutationOperator::Aor);
        assert!(r.len() >= 4);
    }

    #[test]
    fn test_operator_replacements_all() {
        for op in MutationOperator::all() {
            let r = operator_replacements(op);
            assert!(!r.is_empty(), "No replacements for {:?}", op);
        }
    }

    #[test]
    fn test_json_output_without_stats_is_bare_array() {
        let func = ParsedFunction {
            name: "add".to_string(),
            file: PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 1,
            body: "fn add(a: i32, b: i32) -> i32 { return a + b; }".to_string(),
        };
        let mutants = generate_fn_mutants(&func, &[MutationOperator::Aor], 1);
        let output =
            format_mutant_output(&mutants, CliOutputFormat::Json, &Colour::new(true), false)
                .unwrap();
        let parsed: Vec<MutantDescriptor> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_json_output_with_stats_is_wrapped() {
        let func = ParsedFunction {
            name: "add".to_string(),
            file: PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 1,
            body: "fn add(a: i32, b: i32) -> i32 { return a + b; }".to_string(),
        };
        let mutants = generate_fn_mutants(&func, &[MutationOperator::Aor], 1);
        let output =
            format_mutant_output(&mutants, CliOutputFormat::Json, &Colour::new(true), true)
                .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.get("mutants").is_some());
        assert!(parsed.get("statistics").is_some());
    }
}
