//! Report generation: CSV output, summary tables, LaTeX table generation for
//! papers, per-instance results, aggregate statistics, gap closure histograms,
//! and timing breakdowns.

use crate::instance::InstanceSet;
use crate::metrics::{
    arithmetic_mean, median, shifted_geometric_mean, AggregateMetrics, BenchmarkMetrics,
};
use crate::runner::{BatchSummary, RunResult, RunStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

// ---------------------------------------------------------------------------
// Report configuration
// ---------------------------------------------------------------------------

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Csv,
    Latex,
    Markdown,
    PlainText,
}

/// Configuration for report generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Output format.
    pub format: ReportFormat,
    /// Whether to include per-instance rows.
    pub per_instance: bool,
    /// Whether to include aggregate rows.
    pub aggregate: bool,
    /// Whether to include timing breakdown.
    pub timing_breakdown: bool,
    /// Whether to include cut statistics.
    pub cut_stats: bool,
    /// Decimal places for floating point numbers.
    pub decimal_places: usize,
    /// Title for the report / LaTeX caption.
    pub title: Option<String>,
    /// LaTeX label for cross-referencing.
    pub label: Option<String>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        ReportConfig {
            format: ReportFormat::Csv,
            per_instance: true,
            aggregate: true,
            timing_breakdown: false,
            cut_stats: false,
            decimal_places: 3,
            title: None,
            label: None,
        }
    }
}

impl ReportConfig {
    /// Create a LaTeX report config.
    pub fn latex(title: &str, label: &str) -> Self {
        ReportConfig {
            format: ReportFormat::Latex,
            per_instance: true,
            aggregate: true,
            timing_breakdown: true,
            cut_stats: true,
            decimal_places: 2,
            title: Some(title.to_string()),
            label: Some(label.to_string()),
        }
    }

    /// Create a minimal CSV config.
    pub fn csv() -> Self {
        ReportConfig {
            format: ReportFormat::Csv,
            ..Default::default()
        }
    }

    /// Format a float value using configured decimal places.
    pub fn fmt_float(&self, v: f64) -> String {
        format!("{:.*}", self.decimal_places, v)
    }
}

// ---------------------------------------------------------------------------
// Timing breakdown
// ---------------------------------------------------------------------------

/// Timing breakdown for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub instance_name: String,
    pub total_time_secs: f64,
    pub reformulation_secs: f64,
    pub reformulation_pct: f64,
    pub cut_generation_secs: f64,
    pub cut_generation_pct: f64,
    pub lp_solve_secs: f64,
    pub lp_solve_pct: f64,
    pub branching_secs: f64,
    pub branching_pct: f64,
    pub other_secs: f64,
    pub other_pct: f64,
}

impl TimingBreakdown {
    /// Extract from a RunResult with metrics.
    pub fn from_result(result: &RunResult) -> Option<Self> {
        let m = result.metrics.as_ref()?;
        let total = m.solve_time_secs.max(1e-12);
        Some(TimingBreakdown {
            instance_name: result.instance_name.clone(),
            total_time_secs: total,
            reformulation_secs: m.reformulation_time_secs,
            reformulation_pct: (m.reformulation_time_secs / total) * 100.0,
            cut_generation_secs: m.cut_generation_time_secs,
            cut_generation_pct: (m.cut_generation_time_secs / total) * 100.0,
            lp_solve_secs: m.lp_solve_time_secs,
            lp_solve_pct: (m.lp_solve_time_secs / total) * 100.0,
            branching_secs: m.branching_time_secs,
            branching_pct: (m.branching_time_secs / total) * 100.0,
            other_secs: m.other_time_secs,
            other_pct: (m.other_time_secs / total) * 100.0,
        })
    }

    /// Average timing breakdown from multiple results.
    pub fn average(breakdowns: &[TimingBreakdown]) -> Option<TimingBreakdown> {
        if breakdowns.is_empty() {
            return None;
        }
        let n = breakdowns.len() as f64;
        Some(TimingBreakdown {
            instance_name: "average".to_string(),
            total_time_secs: breakdowns.iter().map(|b| b.total_time_secs).sum::<f64>() / n,
            reformulation_secs: breakdowns.iter().map(|b| b.reformulation_secs).sum::<f64>() / n,
            reformulation_pct: breakdowns.iter().map(|b| b.reformulation_pct).sum::<f64>() / n,
            cut_generation_secs: breakdowns
                .iter()
                .map(|b| b.cut_generation_secs)
                .sum::<f64>()
                / n,
            cut_generation_pct: breakdowns.iter().map(|b| b.cut_generation_pct).sum::<f64>() / n,
            lp_solve_secs: breakdowns.iter().map(|b| b.lp_solve_secs).sum::<f64>() / n,
            lp_solve_pct: breakdowns.iter().map(|b| b.lp_solve_pct).sum::<f64>() / n,
            branching_secs: breakdowns.iter().map(|b| b.branching_secs).sum::<f64>() / n,
            branching_pct: breakdowns.iter().map(|b| b.branching_pct).sum::<f64>() / n,
            other_secs: breakdowns.iter().map(|b| b.other_secs).sum::<f64>() / n,
            other_pct: breakdowns.iter().map(|b| b.other_pct).sum::<f64>() / n,
        })
    }
}

// ---------------------------------------------------------------------------
// CSV Reporter
// ---------------------------------------------------------------------------

/// Generate CSV reports from benchmark results.
pub struct CsvReporter;

impl CsvReporter {
    /// Write per-instance results as CSV to a writer.
    pub fn write_results<W: Write>(
        writer: &mut W,
        results: &[RunResult],
        config: &ReportConfig,
    ) -> Result<(), crate::BenchError> {
        // Header.
        writeln!(
            writer,
            "instance,config,status,time_s,objective,bound,gap_pct,nodes,iterations,cuts"
        )
        .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;

        for r in results {
            let gap = r.gap_percent().unwrap_or(f64::NAN);
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{}",
                r.instance_name,
                r.config_label,
                r.status,
                config.fmt_float(r.wall_time_secs),
                r.objective
                    .map(|v| config.fmt_float(v))
                    .unwrap_or_else(|| "NA".to_string()),
                r.best_bound
                    .map(|v| config.fmt_float(v))
                    .unwrap_or_else(|| "NA".to_string()),
                if gap.is_nan() {
                    "NA".to_string()
                } else {
                    config.fmt_float(gap)
                },
                r.node_count,
                r.iteration_count,
                r.cuts_generated,
            )
            .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;
        }
        Ok(())
    }

    /// Write per-instance results to a CSV file.
    pub fn write_file(
        path: &Path,
        results: &[RunResult],
        config: &ReportConfig,
    ) -> Result<(), crate::BenchError> {
        let mut file = std::fs::File::create(path).map_err(crate::BenchError::Io)?;
        Self::write_results(&mut file, results, config)
    }

    /// Write aggregate metrics for multiple configs.
    pub fn write_aggregate<W: Write>(
        writer: &mut W,
        config_metrics: &HashMap<String, AggregateMetrics>,
        report_config: &ReportConfig,
    ) -> Result<(), crate::BenchError> {
        writeln!(
            writer,
            "config,count,solved,solve_rate,mean_time,sgm_time,median_time,mean_nodes,sgm_nodes,mean_gap_closure,total_cuts"
        )
        .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;

        let mut configs: Vec<_> = config_metrics.keys().collect();
        configs.sort();
        for label in configs {
            let m = &config_metrics[label];
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{}",
                label,
                m.count,
                m.solved,
                report_config.fmt_float(m.solve_rate()),
                report_config.fmt_float(m.mean_time),
                report_config.fmt_float(m.sgm_time),
                report_config.fmt_float(m.median_time),
                report_config.fmt_float(m.mean_nodes),
                report_config.fmt_float(m.sgm_nodes),
                report_config.fmt_float(m.mean_root_gap_closure),
                m.total_cuts,
            )
            .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;
        }
        Ok(())
    }

    /// Write timing breakdown as CSV.
    pub fn write_timing<W: Write>(
        writer: &mut W,
        results: &[RunResult],
        config: &ReportConfig,
    ) -> Result<(), crate::BenchError> {
        writeln!(
            writer,
            "instance,total_s,reform_s,reform_pct,cutgen_s,cutgen_pct,lp_s,lp_pct,branch_s,branch_pct,other_s,other_pct"
        )
        .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;

        for r in results {
            if let Some(tb) = TimingBreakdown::from_result(r) {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{},{},{},{}",
                    tb.instance_name,
                    config.fmt_float(tb.total_time_secs),
                    config.fmt_float(tb.reformulation_secs),
                    config.fmt_float(tb.reformulation_pct),
                    config.fmt_float(tb.cut_generation_secs),
                    config.fmt_float(tb.cut_generation_pct),
                    config.fmt_float(tb.lp_solve_secs),
                    config.fmt_float(tb.lp_solve_pct),
                    config.fmt_float(tb.branching_secs),
                    config.fmt_float(tb.branching_pct),
                    config.fmt_float(tb.other_secs),
                    config.fmt_float(tb.other_pct),
                )
                .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LaTeX Reporter
// ---------------------------------------------------------------------------

/// Generate LaTeX tables for papers.
pub struct LatexReporter;

impl LatexReporter {
    /// Generate a LaTeX per-instance results table.
    pub fn per_instance_table(results: &[RunResult], config: &ReportConfig) -> String {
        let dp = config.decimal_places;
        let mut out = String::new();
        let caption = config.title.as_deref().unwrap_or("Benchmark Results");
        let label = config.label.as_deref().unwrap_or("tab:results");

        out.push_str("\\begin{table}[htbp]\n");
        out.push_str("\\centering\n");
        out.push_str(&format!("\\caption{{{}}}\n", escape_latex(caption)));
        out.push_str(&format!("\\label{{{}}}\n", label));
        out.push_str("\\begin{tabular}{lrrrrrr}\n");
        out.push_str("\\toprule\n");
        out.push_str("Instance & Status & Time (s) & Obj & Gap (\\%) & Nodes & Cuts \\\\\n");
        out.push_str("\\midrule\n");

        for r in results {
            let gap_str = r
                .gap_percent()
                .map(|g| format!("{:.*}", dp, g))
                .unwrap_or_else(|| "--".to_string());
            let obj_str = r
                .objective
                .map(|o| format!("{:.*}", dp, o))
                .unwrap_or_else(|| "--".to_string());

            out.push_str(&format!(
                "{} & {} & {:.*} & {} & {} & {} & {} \\\\\n",
                escape_latex(&r.instance_name),
                r.status,
                dp,
                r.wall_time_secs,
                obj_str,
                gap_str,
                r.node_count,
                r.cuts_generated,
            ));
        }

        out.push_str("\\bottomrule\n");
        out.push_str("\\end{tabular}\n");
        out.push_str("\\end{table}\n");
        out
    }

    /// Generate a LaTeX aggregate comparison table (configs as columns).
    pub fn aggregate_table(
        config_metrics: &HashMap<String, AggregateMetrics>,
        config: &ReportConfig,
    ) -> String {
        let dp = config.decimal_places;
        let mut configs: Vec<&String> = config_metrics.keys().collect();
        configs.sort();
        let ncols = configs.len();
        let col_spec = format!("l{}", "r".repeat(ncols));

        let mut out = String::new();
        let caption = config
            .title
            .as_deref()
            .unwrap_or("Aggregate Benchmark Comparison");
        let label = config.label.as_deref().unwrap_or("tab:aggregate");

        out.push_str("\\begin{table}[htbp]\n");
        out.push_str("\\centering\n");
        out.push_str(&format!("\\caption{{{}}}\n", escape_latex(caption)));
        out.push_str(&format!("\\label{{{}}}\n", label));
        out.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        out.push_str("\\toprule\n");

        // Header row.
        let header: Vec<String> = std::iter::once("Metric".to_string())
            .chain(configs.iter().map(|c| escape_latex(c)))
            .collect();
        out.push_str(&format!("{} \\\\\n", header.join(" & ")));
        out.push_str("\\midrule\n");

        // Rows.
        let metrics = &configs;
        let row = |name: &str, f: &dyn Fn(&AggregateMetrics) -> String| -> String {
            let values: Vec<String> = metrics.iter().map(|c| f(&config_metrics[*c])).collect();
            let mut all = vec![name.to_string()];
            all.extend(values);
            format!("{} \\\\\n", all.join(" & "))
        };

        out.push_str(&row("Solved", &|m| format!("{}/{}", m.solved, m.count)));
        out.push_str(&row("Solve rate (\\%)", &|m| {
            format!("{:.*}", dp, m.solve_rate())
        }));
        out.push_str(&row("Mean time (s)", &|m| {
            format!("{:.*}", dp, m.mean_time)
        }));
        out.push_str(&row("SGM time (s)", &|m| format!("{:.*}", dp, m.sgm_time)));
        out.push_str(&row("Median time (s)", &|m| {
            format!("{:.*}", dp, m.median_time)
        }));
        out.push_str(&row("Mean nodes", &|m| format!("{:.*}", 1, m.mean_nodes)));
        out.push_str(&row("Root closure (\\%)", &|m| {
            format!("{:.*}", dp, m.mean_root_gap_closure)
        }));
        out.push_str(&row("Total cuts", &|m| format!("{}", m.total_cuts)));

        out.push_str("\\bottomrule\n");
        out.push_str("\\end{{tabular}}\n");
        out.push_str("\\end{table}\n");
        out
    }

    /// Generate a LaTeX timing breakdown table.
    pub fn timing_table(breakdowns: &[TimingBreakdown], config: &ReportConfig) -> String {
        let dp = config.decimal_places;
        let mut out = String::new();

        out.push_str("\\begin{table}[htbp]\n");
        out.push_str("\\centering\n");
        out.push_str("\\caption{Timing Breakdown}\n");
        out.push_str("\\label{tab:timing}\n");
        out.push_str("\\begin{tabular}{lrrrrrr}\n");
        out.push_str("\\toprule\n");
        out.push_str("Instance & Total (s) & Reform (\\%) & CutGen (\\%) & LP (\\%) & Branch (\\%) & Other (\\%) \\\\\n");
        out.push_str("\\midrule\n");

        for tb in breakdowns {
            out.push_str(&format!(
                "{} & {:.*} & {:.*} & {:.*} & {:.*} & {:.*} & {:.*} \\\\\n",
                escape_latex(&tb.instance_name),
                dp,
                tb.total_time_secs,
                dp,
                tb.reformulation_pct,
                dp,
                tb.cut_generation_pct,
                dp,
                tb.lp_solve_pct,
                dp,
                tb.branching_pct,
                dp,
                tb.other_pct,
            ));
        }

        out.push_str("\\bottomrule\n");
        out.push_str("\\end{tabular}\n");
        out.push_str("\\end{table}\n");
        out
    }
}

/// Escape special LaTeX characters in a string.
fn escape_latex(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('~', "\\textasciitilde{}")
        .replace('^', "\\textasciicircum{}")
}

// ---------------------------------------------------------------------------
// Summary reporter (plain text / markdown)
// ---------------------------------------------------------------------------

/// Generate human-readable summary reports.
pub struct SummaryReporter;

impl SummaryReporter {
    /// Generate a plain-text summary of results.
    pub fn text_summary(results: &[RunResult]) -> String {
        let summary = BatchSummary::from_results(results);
        let agg = AggregateMetrics::from_results(results);
        let mut out = String::new();

        out.push_str("=== Benchmark Summary ===\n");
        out.push_str(&format!("{}\n", summary));
        out.push_str(&format!("\n{}\n", agg));

        // Status breakdown.
        out.push_str("\nStatus breakdown:\n");
        let mut status_counts: HashMap<RunStatus, usize> = HashMap::new();
        for r in results {
            *status_counts.entry(r.status).or_default() += 1;
        }
        let mut statuses: Vec<_> = status_counts.iter().collect();
        statuses.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (status, count) in statuses {
            out.push_str(&format!("  {}: {}\n", status, count));
        }

        out
    }

    /// Generate a markdown summary.
    pub fn markdown_summary(
        results: &[RunResult],
        config_metrics: &HashMap<String, AggregateMetrics>,
    ) -> String {
        let mut out = String::new();
        out.push_str("# Benchmark Results\n\n");

        // Aggregate table.
        if !config_metrics.is_empty() {
            out.push_str("## Aggregate Comparison\n\n");
            out.push_str("| Config | Solved | Rate | Mean Time | SGM Time | Nodes |\n");
            out.push_str("|--------|--------|------|-----------|----------|-------|\n");
            let mut configs: Vec<_> = config_metrics.keys().collect();
            configs.sort();
            for label in configs {
                let m = &config_metrics[label];
                out.push_str(&format!(
                    "| {} | {}/{} | {:.1}% | {:.3}s | {:.3}s | {:.0} |\n",
                    label,
                    m.solved,
                    m.count,
                    m.solve_rate(),
                    m.mean_time,
                    m.sgm_time,
                    m.mean_nodes,
                ));
            }
            out.push('\n');
        }

        // Per-instance table (first 50).
        if !results.is_empty() {
            out.push_str("## Per-Instance Results\n\n");
            out.push_str("| Instance | Status | Time (s) | Gap (%) | Nodes | Cuts |\n");
            out.push_str("|----------|--------|----------|---------|-------|------|\n");
            for r in results.iter().take(50) {
                let gap_str = r
                    .gap_percent()
                    .map(|g| format!("{:.2}", g))
                    .unwrap_or_else(|| "--".to_string());
                out.push_str(&format!(
                    "| {} | {} | {:.3} | {} | {} | {} |\n",
                    r.instance_name,
                    r.status,
                    r.wall_time_secs,
                    gap_str,
                    r.node_count,
                    r.cuts_generated,
                ));
            }
            if results.len() > 50 {
                out.push_str(&format!(
                    "\n*({} more instances omitted)*\n",
                    results.len() - 50
                ));
            }
            out.push('\n');
        }

        out
    }

    /// Generate a compact one-line summary per configuration.
    pub fn one_line_summary(label: &str, metrics: &AggregateMetrics) -> String {
        format!(
            "{}: solved {}/{} ({:.1}%), sgm={:.3}s, nodes={:.0}, gap_closure={:.1}%",
            label,
            metrics.solved,
            metrics.count,
            metrics.solve_rate(),
            metrics.sgm_time,
            metrics.mean_nodes,
            metrics.mean_root_gap_closure,
        )
    }
}

// ---------------------------------------------------------------------------
// Gap closure histogram data
// ---------------------------------------------------------------------------

/// Data for a gap closure histogram (data only, no rendering).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapClosureHistogram {
    /// Bin edges (e.g., [0, 10, 20, ..., 100]).
    pub bin_edges: Vec<f64>,
    /// Count in each bin.
    pub counts: Vec<usize>,
    /// Configuration label.
    pub config_label: String,
}

impl GapClosureHistogram {
    /// Build a gap closure histogram from results.
    pub fn from_results(config_label: &str, results: &[RunResult], num_bins: usize) -> Self {
        let closures: Vec<f64> = results
            .iter()
            .filter_map(|r| r.metrics.as_ref())
            .map(|m| m.root_gap_closure_percent.clamp(0.0, 100.0))
            .collect();

        let bin_width = 100.0 / num_bins as f64;
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            bin_edges.push(i as f64 * bin_width);
        }

        let mut counts = vec![0usize; num_bins];
        for &c in &closures {
            let bin = ((c / bin_width) as usize).min(num_bins - 1);
            counts[bin] += 1;
        }

        GapClosureHistogram {
            bin_edges,
            counts,
            config_label: config_label.to_string(),
        }
    }

    /// Total number of instances.
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Fraction in each bin.
    pub fn fractions(&self) -> Vec<f64> {
        let total = self.total() as f64;
        if total == 0.0 {
            return vec![0.0; self.counts.len()];
        }
        self.counts.iter().map(|&c| c as f64 / total).collect()
    }
}

// ---------------------------------------------------------------------------
// Full report generation
// ---------------------------------------------------------------------------

/// Generate a complete set of report files.
pub fn generate_full_report(
    output_dir: &Path,
    results: &HashMap<String, Vec<RunResult>>,
    config: &ReportConfig,
) -> Result<(), crate::BenchError> {
    std::fs::create_dir_all(output_dir).map_err(crate::BenchError::Io)?;

    // Per-config CSV files.
    for (label, rs) in results {
        let safe_label = label.replace(|c: char| !c.is_alphanumeric() && c != '-', "_");
        let csv_path = output_dir.join(format!("{}.csv", safe_label));
        CsvReporter::write_file(&csv_path, rs, config)?;
    }

    // Aggregate CSV.
    let config_metrics: HashMap<String, AggregateMetrics> = results
        .iter()
        .map(|(label, rs)| (label.clone(), AggregateMetrics::from_results(rs)))
        .collect();
    let agg_path = output_dir.join("aggregate.csv");
    let mut agg_file = std::fs::File::create(&agg_path).map_err(crate::BenchError::Io)?;
    CsvReporter::write_aggregate(&mut agg_file, &config_metrics, config)?;

    // Summary text.
    let summary_path = output_dir.join("summary.txt");
    let mut summary_parts = Vec::new();
    for (label, rs) in results {
        summary_parts.push(format!(
            "=== {} ===\n{}",
            label,
            SummaryReporter::text_summary(rs)
        ));
    }
    std::fs::write(&summary_path, summary_parts.join("\n\n")).map_err(crate::BenchError::Io)?;

    // Markdown summary.
    let md_path = output_dir.join("summary.md");
    let all_results: Vec<RunResult> = results.values().flat_map(|v| v.iter().cloned()).collect();
    let md = SummaryReporter::markdown_summary(&all_results, &config_metrics);
    std::fs::write(&md_path, md).map_err(crate::BenchError::Io)?;

    // LaTeX tables.
    if config.format == ReportFormat::Latex {
        let latex_config = ReportConfig::latex(
            config.title.as_deref().unwrap_or("Results"),
            config.label.as_deref().unwrap_or("tab:results"),
        );
        let agg_latex = LatexReporter::aggregate_table(&config_metrics, &latex_config);
        let latex_path = output_dir.join("aggregate.tex");
        std::fs::write(&latex_path, agg_latex).map_err(crate::BenchError::Io)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::RunResult;

    fn sample_results() -> Vec<RunResult> {
        vec![
            RunResult::optimal("inst_a", "config1", 1.5, -10.0, 5, 100),
            RunResult::optimal("inst_b", "config1", 2.3, -8.0, 10, 200),
            RunResult::timeout("inst_c", "config1", 60.0, Some(-5.0), Some(-7.0)),
        ]
    }

    #[test]
    fn test_csv_write() {
        let results = sample_results();
        let config = ReportConfig::csv();
        let mut buf = Vec::new();
        CsvReporter::write_results(&mut buf, &results, &config).unwrap();
        let csv_str = String::from_utf8(buf).unwrap();
        assert!(csv_str.contains("inst_a"));
        assert!(csv_str.contains("inst_b"));
        assert!(csv_str.contains("Optimal"));
    }

    #[test]
    fn test_csv_aggregate() {
        let results = sample_results();
        let mut metrics = HashMap::new();
        metrics.insert("cfg1".to_string(), AggregateMetrics::from_results(&results));
        let config = ReportConfig::csv();
        let mut buf = Vec::new();
        CsvReporter::write_aggregate(&mut buf, &metrics, &config).unwrap();
        let csv_str = String::from_utf8(buf).unwrap();
        assert!(csv_str.contains("cfg1"));
    }

    #[test]
    fn test_latex_per_instance() {
        let results = sample_results();
        let config = ReportConfig::latex("Test Table", "tab:test");
        let latex = LatexReporter::per_instance_table(&results, &config);
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("inst\\_a"));
        assert!(latex.contains("\\end{table}"));
    }

    #[test]
    fn test_latex_aggregate() {
        let mut metrics = HashMap::new();
        metrics.insert(
            "cfg_a".to_string(),
            AggregateMetrics::from_results(&sample_results()),
        );
        let config = ReportConfig::latex("Agg", "tab:agg");
        let latex = LatexReporter::aggregate_table(&metrics, &config);
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("cfg\\_a"));
    }

    #[test]
    fn test_text_summary() {
        let results = sample_results();
        let txt = SummaryReporter::text_summary(&results);
        assert!(txt.contains("Benchmark Summary"));
        assert!(txt.contains("Optimal"));
    }

    #[test]
    fn test_markdown_summary() {
        let results = sample_results();
        let mut metrics = HashMap::new();
        metrics.insert("c".to_string(), AggregateMetrics::from_results(&results));
        let md = SummaryReporter::markdown_summary(&results, &metrics);
        assert!(md.contains("# Benchmark Results"));
        assert!(md.contains("inst_a"));
    }

    #[test]
    fn test_timing_breakdown() {
        let mut r = RunResult::optimal("test", "cfg", 2.0, -10.0, 5, 100);
        r.metrics = Some(BenchmarkMetrics {
            solve_time_secs: 2.0,
            node_count: 5,
            root_gap_percent: 10.0,
            final_gap_percent: 0.0,
            root_gap_closure_percent: 100.0,
            iteration_count: 100,
            cuts_by_type: HashMap::new(),
            reformulation_time_secs: 0.1,
            cut_generation_time_secs: 0.3,
            lp_solve_time_secs: 1.2,
            branching_time_secs: 0.3,
            other_time_secs: 0.1,
        });
        let tb = TimingBreakdown::from_result(&r).unwrap();
        assert!((tb.total_time_secs - 2.0).abs() < 1e-10);
        assert!(tb.lp_solve_pct > 50.0);
    }

    #[test]
    fn test_gap_closure_histogram() {
        let mut results = Vec::new();
        for i in 0..10 {
            let mut r = RunResult::optimal(&format!("i{}", i), "cfg", 1.0, 0.0, 0, 0);
            r.metrics = Some(BenchmarkMetrics {
                root_gap_closure_percent: (i as f64) * 10.0,
                ..BenchmarkMetrics::zero()
            });
            results.push(r);
        }
        let hist = GapClosureHistogram::from_results("test", &results, 10);
        assert_eq!(hist.total(), 10);
        assert_eq!(hist.counts.len(), 10);
    }

    #[test]
    fn test_escape_latex() {
        assert_eq!(escape_latex("a_b"), "a\\_b");
        assert_eq!(escape_latex("100%"), "100\\%");
        assert_eq!(escape_latex("a&b"), "a\\&b");
    }

    #[test]
    fn test_one_line_summary() {
        let metrics = AggregateMetrics::from_results(&sample_results());
        let line = SummaryReporter::one_line_summary("test", &metrics);
        assert!(line.contains("test"));
        assert!(line.contains("solved"));
    }
}
