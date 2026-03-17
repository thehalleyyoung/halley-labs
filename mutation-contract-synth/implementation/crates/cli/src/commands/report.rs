//! `mutspec report` — report generation command.
//!
//! Generates formatted reports from analysis results in text, JSON, Markdown,
//! and SARIF formats. Supports combining mutant and contract data.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use log::{debug, info, warn};
use serde::Serialize;

use shared_types::contracts::{Contract, Specification};
use shared_types::operators::{MutantDescriptor, MutantStatus, MutationOperator};

use crate::output::{
    format_contract_text, mutant_status_table, write_json, AlignedTable, CliOutputFormat, Colour,
    MarkdownTable, MutationStats, SarifBuilder, SynthesisStats, VerificationStats,
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct ReportArgs {
    /// Path to mutants JSON file.
    #[arg(long)]
    pub mutants: Option<PathBuf>,

    /// Path to contracts/specification JSON file.
    #[arg(long)]
    pub contracts: Option<PathBuf>,

    /// Output file (stdout if omitted).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format.
    #[arg(short = 'f', long, value_enum, default_value = "text")]
    pub format: CliOutputFormat,

    /// Title for the report.
    #[arg(long, default_value = "MutSpec Report")]
    pub title: String,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,

    /// Include detailed per-mutant information.
    #[arg(long)]
    pub detailed: bool,

    /// Sort mutants by this field.
    #[arg(long, value_enum)]
    pub sort_by: Option<SortField>,

    /// Include operator distribution breakdown.
    #[arg(long)]
    pub operator_stats: bool,

    /// Show only surviving (alive) mutants.
    #[arg(long)]
    pub survivors_only: bool,

    /// Maximum mutants to display in text output.
    #[arg(long, default_value = "100")]
    pub max_display: usize,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum SortField {
    Operator,
    Status,
    File,
    Line,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &ReportArgs, _cfg: &crate::config::CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);
    let started = Instant::now();
    let _timing = super::TimingGuard::new("Report generation");

    if args.mutants.is_none() && args.contracts.is_none() {
        anyhow::bail!("At least one of --mutants or --contracts must be provided");
    }

    // --- Load data --------------------------------------------------------
    let mutants = match &args.mutants {
        Some(path) => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let m: Vec<MutantDescriptor> = serde_json::from_str(&content)?;
            info!("Loaded {} mutants", m.len());
            Some(m)
        }
        None => None,
    };

    let specification = match &args.contracts {
        Some(path) => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let s: Specification = serde_json::from_str(&content)?;
            info!("Loaded specification with {} contracts", s.contracts.len());
            Some(s)
        }
        None => None,
    };

    // --- Apply filters and sorting ----------------------------------------
    let display_mutants = prepare_mutants(mutants.as_deref(), args);

    // --- Generate report --------------------------------------------------
    let output = format_report(
        &args.title,
        display_mutants.as_deref(),
        specification.as_ref(),
        args,
        &colour,
    )?;

    super::write_output(&output, args.output.as_deref())?;

    let elapsed = started.elapsed();
    eprintln!(
        "{} Report generated in {:.2}s",
        colour.green("✓"),
        elapsed.as_secs_f64()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Data preparation
// ---------------------------------------------------------------------------

fn prepare_mutants(
    mutants: Option<&[MutantDescriptor]>,
    args: &ReportArgs,
) -> Option<Vec<MutantDescriptor>> {
    let Some(mutants) = mutants else {
        return None;
    };

    let mut result: Vec<MutantDescriptor> = if args.survivors_only {
        mutants
            .iter()
            .filter(|m| matches!(m.status, MutantStatus::Alive))
            .cloned()
            .collect()
    } else {
        mutants.to_vec()
    };

    // Sort
    if let Some(sort) = args.sort_by {
        match sort {
            SortField::Operator => {
                result.sort_by(|a, b| a.operator.mnemonic().cmp(&b.operator.mnemonic()));
            }
            SortField::Status => {
                result.sort_by(|a, b| format!("{:?}", a.status).cmp(&format!("{:?}", b.status)));
            }
            SortField::File => {
                result.sort_by(|a, b| a.site.location.start.file.cmp(&b.site.location.start.file));
            }
            SortField::Line => {
                result.sort_by_key(|m| m.site.location.start.line);
            }
        }
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

fn format_report(
    title: &str,
    mutants: Option<&[MutantDescriptor]>,
    spec: Option<&Specification>,
    args: &ReportArgs,
    colour: &Colour,
) -> Result<String> {
    match args.format {
        CliOutputFormat::Json => format_json_report(title, mutants, spec),
        CliOutputFormat::Markdown => format_markdown_report(title, mutants, spec, args),
        CliOutputFormat::Sarif => format_sarif_report(mutants),
        CliOutputFormat::Text => format_text_report(title, mutants, spec, args, colour),
    }
}

fn format_json_report(
    title: &str,
    mutants: Option<&[MutantDescriptor]>,
    spec: Option<&Specification>,
) -> Result<String> {
    #[derive(Serialize)]
    struct Report<'a> {
        title: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        mutation_stats: Option<MutationStats>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mutants: Option<&'a [MutantDescriptor]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        specification: Option<&'a Specification>,
    }

    let stats = mutants.map(MutationStats::from_descriptors);
    let report = Report {
        title,
        mutation_stats: stats,
        mutants,
        specification: spec,
    };
    Ok(serde_json::to_string_pretty(&report)?)
}

fn format_markdown_report(
    title: &str,
    mutants: Option<&[MutantDescriptor]>,
    spec: Option<&Specification>,
    args: &ReportArgs,
) -> Result<String> {
    let mut md = String::new();
    md.push_str(&format!("# {title}\n\n"));

    if let Some(mutants) = mutants {
        let stats = MutationStats::from_descriptors(mutants);

        md.push_str("## Mutation Summary\n\n");
        md.push_str(&format!("| Metric | Value |\n"));
        md.push_str(&format!("| ------ | ----- |\n"));
        md.push_str(&format!("| Total mutants | {} |\n", stats.total_mutants));
        md.push_str(&format!("| Killed | {} |\n", stats.killed));
        md.push_str(&format!("| Alive | {} |\n", stats.alive));
        md.push_str(&format!("| Equivalent | {} |\n", stats.equivalent));
        md.push_str(&format!("| Timeout | {} |\n", stats.timeout));
        md.push_str(&format!("| Error | {} |\n", stats.error));
        md.push_str(&format!(
            "| **Mutation score** | **{:.1}%** |\n\n",
            stats.mutation_score * 100.0
        ));

        if args.operator_stats {
            md.push_str("### By Operator\n\n");
            let mut op_tbl = MarkdownTable::new(vec!["Operator".into(), "Count".into()]);
            for (op, count) in &stats.operators {
                op_tbl.add_row(vec![op.clone(), count.to_string()]);
            }
            md.push_str(&format!("{op_tbl}\n"));
        }

        if args.detailed {
            md.push_str("## Mutant Details\n\n");
            let display_count = mutants.len().min(args.max_display);
            let mut tbl = MarkdownTable::new(vec![
                "ID".into(),
                "Operator".into(),
                "Status".into(),
                "Original".into(),
                "Replacement".into(),
                "File".into(),
                "Line".into(),
            ]);
            for m in &mutants[..display_count] {
                tbl.add_row(vec![
                    format!("`{}`", m.id.short()),
                    m.operator.mnemonic().to_string(),
                    format!("{:?}", m.status),
                    format!("`{}`", m.site.original),
                    format!("`{}`", m.site.replacement),
                    m.site.location.start.file.display().to_string(),
                    m.site.location.start.line.to_string(),
                ]);
            }
            md.push_str(&format!("{tbl}\n"));
            if mutants.len() > display_count {
                md.push_str(&format!(
                    "*... and {} more mutants*\n\n",
                    mutants.len() - display_count
                ));
            }
        }
    }

    if let Some(spec) = spec {
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
    }

    md.push_str("---\n\n*Generated by MutSpec*\n");
    Ok(md)
}

fn format_sarif_report(mutants: Option<&[MutantDescriptor]>) -> Result<String> {
    let mut sarif = SarifBuilder::new("mutspec-report", env!("CARGO_PKG_VERSION"));

    if let Some(mutants) = mutants {
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
                    "{} → {} [{}]",
                    m.site.original,
                    m.site.replacement,
                    format!("{:?}", m.status),
                ),
                m.site.location.start.file.display().to_string(),
                m.site.location.start.line,
                m.site.location.start.column,
            );
        }
    }

    let doc = sarif.build();
    Ok(serde_json::to_string_pretty(&doc)?)
}

fn format_text_report(
    title: &str,
    mutants: Option<&[MutantDescriptor]>,
    spec: Option<&Specification>,
    args: &ReportArgs,
    colour: &Colour,
) -> Result<String> {
    let mut buf = String::new();

    let sep = "═".repeat(60);
    buf.push_str(&colour.bold(&sep));
    buf.push('\n');
    buf.push_str(&colour.bold(&format!("  {title}")));
    buf.push('\n');
    buf.push_str(&colour.bold(&sep));
    buf.push_str("\n\n");

    if let Some(mutants) = mutants {
        let stats = MutationStats::from_descriptors(mutants);
        buf.push_str(&stats.format_text(colour));
        buf.push('\n');

        if args.operator_stats {
            buf.push_str(&colour.bold("Operator Distribution:\n"));
            let mut op_tbl = AlignedTable::new(vec!["Operator".into(), "Count".into()]);
            op_tbl.set_right_align(1);
            for (op, count) in &stats.operators {
                op_tbl.add_row(vec![colour.cyan(op), count.to_string()]);
            }
            buf.push_str(&format!("{op_tbl}\n"));
        }

        if args.detailed {
            let display_count = mutants.len().min(args.max_display);
            let tbl = mutant_status_table(&mutants[..display_count], colour);
            buf.push_str(&format!("{tbl}"));
            if mutants.len() > display_count {
                buf.push_str(&format!(
                    "\n  {} ... and {} more mutants\n",
                    colour.dim(""),
                    mutants.len() - display_count
                ));
            }
            buf.push('\n');
        }

        // File breakdown
        let by_file = super::partition_by_file(mutants);
        if by_file.len() > 1 {
            buf.push_str(&colour.bold("By File:\n"));
            let mut ftbl = AlignedTable::new(vec![
                "File".into(),
                "Mutants".into(),
                "Killed".into(),
                "Alive".into(),
            ]);
            ftbl.set_right_align(1);
            ftbl.set_right_align(2);
            ftbl.set_right_align(3);
            for (file, file_mutants) in &by_file {
                let k = file_mutants
                    .iter()
                    .filter(|m| matches!(m.status, MutantStatus::Killed))
                    .count();
                let a = file_mutants
                    .iter()
                    .filter(|m| matches!(m.status, MutantStatus::Alive))
                    .count();
                ftbl.add_row(vec![
                    file.display().to_string(),
                    file_mutants.len().to_string(),
                    colour.green(&k.to_string()),
                    colour.red(&a.to_string()),
                ]);
            }
            buf.push_str(&format!("{ftbl}\n"));
        }
    }

    if let Some(spec) = spec {
        buf.push_str(&colour.bold("Contracts:\n\n"));
        for contract in &spec.contracts {
            buf.push_str(&format_contract_text(contract, colour));
            buf.push('\n');
        }
    }

    let thin_sep = "─".repeat(60);
    buf.push_str(&colour.bold(&thin_sep));
    buf.push('\n');
    buf.push_str(&colour.dim("Generated by MutSpec"));
    buf.push('\n');

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

    fn sample_mutants() -> Vec<MutantDescriptor> {
        let loc = SourceLocation::new(PathBuf::from("test.rs"), 10, 1);
        let span = SpanInfo::new(loc.clone(), loc);
        let site = MutationSite::new(span, MutationOperator::Aor, "+", "-");
        let mut m1 = MutantDescriptor::new(MutationOperator::Aor, site);
        m1.mark_killed(shared_types::operators::KillInfo::new(
            "test".to_string(),
            0.0,
        ));

        let loc2 = SourceLocation::new(PathBuf::from("test.rs"), 20, 1);
        let span2 = SpanInfo::new(loc2.clone(), loc2);
        let site2 = MutationSite::new(span2, MutationOperator::Ror, "<", "<=");
        let m2 = MutantDescriptor::new(MutationOperator::Ror, site2);

        vec![m1, m2]
    }

    #[test]
    fn test_format_json_report() {
        let mutants = sample_mutants();
        let result = format_json_report("Test", Some(&mutants), None).unwrap();
        assert!(result.contains("mutation_stats"));
        assert!(result.contains("Aor"));
    }

    #[test]
    fn test_format_markdown_report() {
        let mutants = sample_mutants();
        let args = ReportArgs {
            mutants: None,
            contracts: None,
            output: None,
            format: CliOutputFormat::Markdown,
            title: "Test".into(),
            no_color: true,
            detailed: true,
            sort_by: None,
            operator_stats: true,
            survivors_only: false,
            max_display: 50,
        };
        let result = format_markdown_report("Test", Some(&mutants), None, &args).unwrap();
        assert!(result.contains("# Test"));
        assert!(result.contains("Mutation Summary"));
    }

    #[test]
    fn test_prepare_mutants_survivors_only() {
        let mutants = sample_mutants();
        let args = ReportArgs {
            mutants: None,
            contracts: None,
            output: None,
            format: CliOutputFormat::Text,
            title: "Test".into(),
            no_color: true,
            detailed: false,
            sort_by: None,
            operator_stats: false,
            survivors_only: true,
            max_display: 50,
        };
        let result = prepare_mutants(Some(&mutants), &args).unwrap();
        // Only alive mutants
        assert!(result
            .iter()
            .all(|m| matches!(m.status, MutantStatus::Alive)));
    }

    #[test]
    fn test_prepare_mutants_sort_by_line() {
        let mutants = sample_mutants();
        let args = ReportArgs {
            mutants: None,
            contracts: None,
            output: None,
            format: CliOutputFormat::Text,
            title: "Test".into(),
            no_color: true,
            detailed: false,
            sort_by: Some(SortField::Line),
            operator_stats: false,
            survivors_only: false,
            max_display: 50,
        };
        let result = prepare_mutants(Some(&mutants), &args).unwrap();
        assert_eq!(result.len(), 2);
    }
}
