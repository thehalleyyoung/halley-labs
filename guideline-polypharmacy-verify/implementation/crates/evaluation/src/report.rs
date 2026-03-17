//! Evaluation report generation.
//!
//! Produces structured evaluation reports in Markdown, LaTeX, and JSON formats
//! from benchmark results, baseline comparisons, and statistical metrics.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::baseline::BaselineComparison;
use crate::benchmark::{BenchmarkResult, SuiteResult, SuiteSummary};
use crate::metrics::{
    ConfusionMatrix, DescriptiveStats, PerformanceMetrics, ScalabilityMetrics,
    compute_descriptive,
};

// ═══════════════════════════════════════════════════════════════════════════
// Chart Data
// ═══════════════════════════════════════════════════════════════════════════

/// Chart type selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Bar,
    Line,
    Scatter,
    Heatmap,
    Histogram,
    BoxPlot,
}

impl fmt::Display for ChartType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bar => write!(f, "bar"),
            Self::Line => write!(f, "line"),
            Self::Scatter => write!(f, "scatter"),
            Self::Heatmap => write!(f, "heatmap"),
            Self::Histogram => write!(f, "histogram"),
            Self::BoxPlot => write!(f, "boxplot"),
        }
    }
}

/// Abstract chart data container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub chart_type: ChartType,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<ChartSeries>,
}

/// A single data series within a chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSeries {
    pub name: String,
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
}

impl ChartData {
    /// Create a simple bar chart.
    pub fn bar(title: &str, labels: &[&str], values: &[f64]) -> Self {
        Self {
            chart_type: ChartType::Bar,
            title: title.into(),
            x_label: "Category".into(),
            y_label: "Value".into(),
            series: vec![ChartSeries {
                name: "data".into(),
                x_values: (0..labels.len()).map(|i| i as f64).collect(),
                y_values: values.to_vec(),
            }],
        }
    }

    /// Create a line chart from x, y data.
    pub fn line(title: &str, x_label: &str, y_label: &str, xs: &[f64], ys: &[f64]) -> Self {
        Self {
            chart_type: ChartType::Line,
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            series: vec![ChartSeries {
                name: "data".into(),
                x_values: xs.to_vec(),
                y_values: ys.to_vec(),
            }],
        }
    }

    /// Create a scatter plot.
    pub fn scatter(title: &str, x_label: &str, y_label: &str, xs: &[f64], ys: &[f64]) -> Self {
        Self {
            chart_type: ChartType::Scatter,
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            series: vec![ChartSeries {
                name: "data".into(),
                x_values: xs.to_vec(),
                y_values: ys.to_vec(),
            }],
        }
    }

    /// Create a multi-series line chart.
    pub fn multi_line(title: &str, x_label: &str, y_label: &str, series: Vec<ChartSeries>) -> Self {
        Self {
            chart_type: ChartType::Line,
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            series,
        }
    }

    /// Add a series.
    pub fn add_series(&mut self, series: ChartSeries) {
        self.series.push(series);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Report Table
// ═══════════════════════════════════════════════════════════════════════════

/// A table within a report section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTable {
    pub caption: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub alignment: Vec<Alignment>,
}

/// Column alignment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

impl ReportTable {
    pub fn new(caption: &str, headers: Vec<String>) -> Self {
        let n = headers.len();
        Self {
            caption: caption.into(),
            headers,
            rows: Vec::new(),
            alignment: vec![Alignment::Left; n],
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        assert_eq!(row.len(), self.headers.len(), "Row width must match header count");
        self.rows.push(row);
    }

    pub fn with_alignment(mut self, alignment: Vec<Alignment>) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn column_count(&self) -> usize {
        self.headers.len()
    }

    /// Build a summary table from SuiteSummary.
    pub fn from_suite_summary(summary: &SuiteSummary) -> Self {
        let mut table = Self::new("Suite Summary", vec![
            "Metric".into(), "Value".into(),
        ]);
        table.add_row(vec!["Total benchmarks".into(), format!("{}", summary.total)]);
        table.add_row(vec!["Passed".into(), format!("{}", summary.passed)]);
        table.add_row(vec!["Failed".into(), format!("{}", summary.failed)]);
        table.add_row(vec!["Timed out".into(), format!("{}", summary.timed_out)]);
        table.add_row(vec!["Accuracy".into(), format!("{:.1}%", summary.accuracy * 100.0)]);
        table.add_row(vec!["Avg time (ms)".into(), format!("{:.2}", summary.avg_time_ms)]);
        table.add_row(vec!["Max time (ms)".into(), format!("{:.2}", summary.max_time_ms)]);
        table.add_row(vec!["Median time (ms)".into(), format!("{:.2}", summary.median_time_ms)]);
        table.add_row(vec!["Total time (ms)".into(), format!("{:.2}", summary.total_time_ms)]);
        table
    }

    /// Build a detailed results table.
    pub fn from_benchmark_results(results: &[BenchmarkResult]) -> Self {
        let mut table = Self::new("Benchmark Results", vec![
            "ID".into(), "Expected".into(), "Actual".into(),
            "Correct".into(), "T1 (ms)".into(), "T2 (ms)".into(),
            "Total (ms)".into(), "Memory (B)".into(), "Conflicts".into(),
        ]);
        for r in results {
            table.add_row(vec![
                r.benchmark_id.clone(),
                format!("{}", r.expected_verdict),
                format!("{}", r.verdict),
                if r.correct { "✓".into() } else { "✗".into() },
                format!("{:.2}", r.tier1_time_ms),
                format!("{:.2}", r.tier2_time_ms),
                format!("{:.2}", r.total_time_ms),
                format!("{}", r.memory_usage_bytes),
                format!("{}", r.conflicts_found),
            ]);
        }
        table
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Report Content & Section
// ═══════════════════════════════════════════════════════════════════════════

/// Content types within a report section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportContent {
    Text(String),
    Table(ReportTable),
    Metrics(DescriptiveStats),
    Chart(ChartData),
    ConfusionMatrixDisplay {
        tp: u64, fp: u64, fn_: u64, tn: u64,
        sensitivity: f64, specificity: f64, f1: f64,
    },
    KeyValue(Vec<(String, String)>),
}

/// A section in the evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub content: Vec<ReportContent>,
    pub subsections: Vec<ReportSection>,
}

impl ReportSection {
    pub fn new(title: &str) -> Self {
        Self { title: title.into(), content: Vec::new(), subsections: Vec::new() }
    }

    pub fn add_text(&mut self, text: &str) {
        self.content.push(ReportContent::Text(text.into()));
    }

    pub fn add_table(&mut self, table: ReportTable) {
        self.content.push(ReportContent::Table(table));
    }

    pub fn add_chart(&mut self, chart: ChartData) {
        self.content.push(ReportContent::Chart(chart));
    }

    pub fn add_metrics(&mut self, stats: DescriptiveStats) {
        self.content.push(ReportContent::Metrics(stats));
    }

    pub fn add_key_value(&mut self, pairs: Vec<(String, String)>) {
        self.content.push(ReportContent::KeyValue(pairs));
    }

    pub fn add_subsection(&mut self, section: ReportSection) {
        self.subsections.push(section);
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.add_text(text);
        self
    }

    pub fn with_table(mut self, table: ReportTable) -> Self {
        self.add_table(table);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation Report
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub title: String,
    pub authors: Vec<String>,
    pub date: String,
    pub abstract_text: String,
    pub sections: Vec<ReportSection>,
    pub metadata: ReportMetadata,
}

/// Report metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generator_version: String,
    pub total_benchmarks: usize,
    pub total_suites: usize,
    pub total_wall_clock_ms: f64,
    pub configuration: String,
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            generator_version: "0.1.0".into(),
            total_benchmarks: 0,
            total_suites: 0,
            total_wall_clock_ms: 0.0,
            configuration: String::new(),
        }
    }
}

impl EvaluationReport {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.into(),
            authors: Vec::new(),
            date: String::new(),
            abstract_text: String::new(),
            sections: Vec::new(),
            metadata: ReportMetadata::default(),
        }
    }

    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    pub fn with_author(mut self, author: &str) -> Self {
        self.authors.push(author.into());
        self
    }

    pub fn with_date(mut self, date: &str) -> Self {
        self.date = date.into();
        self
    }

    pub fn with_abstract(mut self, text: &str) -> Self {
        self.abstract_text = text.into();
        self
    }

    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Report Generator
// ═══════════════════════════════════════════════════════════════════════════

/// Generates evaluation reports from benchmark results and baseline comparisons.
pub struct ReportGenerator;

impl ReportGenerator {
    /// Generate a full evaluation report.
    pub fn generate_evaluation_report(
        suite_results: &[SuiteResult],
        baseline: &BaselineComparison,
    ) -> EvaluationReport {
        let mut report = EvaluationReport::new("GuardPharma Evaluation Report")
            .with_author("GuardPharma Evaluation Engine")
            .with_abstract(
                "This report presents the evaluation results of the GuardPharma two-tier \
                 formal verification engine for polypharmacy safety checking. Results are \
                 compared against a TMR-style atemporal baseline."
            );

        let total_benchmarks: usize = suite_results.iter().map(|s| s.results.len()).sum();
        let total_wall: f64 = suite_results.iter().map(|s| s.wall_clock_ms).sum();
        report.metadata = ReportMetadata {
            generator_version: "0.1.0".into(),
            total_benchmarks,
            total_suites: suite_results.len(),
            total_wall_clock_ms: total_wall,
            configuration: "default".into(),
        };

        // Section 1: Executive Summary.
        report.add_section(Self::executive_summary_section(suite_results, baseline));

        // Section 2: Baseline Comparison (E1).
        report.add_section(Self::baseline_comparison_section(baseline));

        // Section 3: Per-suite detailed results.
        for sr in suite_results {
            report.add_section(Self::suite_detail_section(sr));
        }

        // Section 4: Performance analysis.
        report.add_section(Self::performance_section(suite_results));

        // Section 5: Statistical summary.
        report.add_section(Self::statistical_summary_section(suite_results));

        report
    }

    fn executive_summary_section(
        suite_results: &[SuiteResult],
        baseline: &BaselineComparison,
    ) -> ReportSection {
        let mut section = ReportSection::new("Executive Summary");

        let total: usize = suite_results.iter().map(|s| s.summary.total).sum();
        let passed: usize = suite_results.iter().map(|s| s.summary.passed).sum();
        let accuracy = if total > 0 { passed as f64 / total as f64 } else { 0.0 };

        section.add_text(&format!(
            "Evaluated {} benchmarks across {} suites. Overall accuracy: {:.1}%. \
             Baseline comparison: {} true positives, {} additional conflicts found by \
             GuardPharma, {} false positives eliminated.",
            total, suite_results.len(), accuracy * 100.0,
            baseline.true_positives, baseline.false_negatives_baseline,
            baseline.false_positives_baseline,
        ));

        let summary_kv = vec![
            ("Total Benchmarks".into(), format!("{}", total)),
            ("Passed".into(), format!("{}", passed)),
            ("Accuracy".into(), format!("{:.2}%", accuracy * 100.0)),
            ("Suites".into(), format!("{}", suite_results.len())),
            ("Baseline TP".into(), format!("{}", baseline.true_positives)),
            ("Baseline FP (eliminated)".into(), format!("{}", baseline.false_positives_baseline)),
            ("Temporal misses (baseline)".into(), format!("{}", baseline.false_negatives_baseline)),
        ];
        section.add_key_value(summary_kv);

        section
    }

    fn baseline_comparison_section(baseline: &BaselineComparison) -> ReportSection {
        let mut section = ReportSection::new("E1: Baseline Comparison");
        section.add_text(
            "Comparison of GuardPharma verification results against the TMR-style \
             atemporal baseline interaction checker."
        );

        let cm = ConfusionMatrix::new(
            baseline.true_positives as u64,
            baseline.false_positives_baseline as u64,
            baseline.false_negatives_baseline as u64,
            baseline.true_negatives as u64,
        );

        section.content.push(ReportContent::ConfusionMatrixDisplay {
            tp: cm.tp, fp: cm.fp, fn_: cm.fn_, tn: cm.tn,
            sensitivity: cm.sensitivity(),
            specificity: cm.specificity(),
            f1: cm.f1_score(),
        });

        let mut table = ReportTable::new("Baseline Comparison Metrics", vec![
            "Metric".into(), "Value".into(),
        ]);
        table.add_row(vec!["True Positives".into(), format!("{}", baseline.true_positives)]);
        table.add_row(vec!["False Positives (baseline)".into(), format!("{}", baseline.false_positives_baseline)]);
        table.add_row(vec!["False Negatives (baseline)".into(), format!("{}", baseline.false_negatives_baseline)]);
        table.add_row(vec!["True Negatives".into(), format!("{}", baseline.true_negatives)]);
        table.add_row(vec!["Baseline Accuracy".into(), format!("{:.3}", baseline.baseline_accuracy())]);
        table.add_row(vec!["Baseline Sensitivity".into(), format!("{:.3}", baseline.baseline_sensitivity())]);
        table.add_row(vec!["Baseline Specificity".into(), format!("{:.3}", baseline.baseline_specificity())]);
        table.add_row(vec!["Severity Agreement".into(), format!("{:.3}", baseline.severity_agreement_rate())]);
        section.add_table(table);

        section
    }

    fn suite_detail_section(sr: &SuiteResult) -> ReportSection {
        let mut section = ReportSection::new(&format!("Suite: {}", sr.suite_name));

        section.add_text(&format!(
            "{}/{} benchmarks passed ({:.1}%). Wall-clock time: {:.1}ms.",
            sr.summary.passed, sr.summary.total,
            sr.summary.accuracy * 100.0, sr.wall_clock_ms,
        ));

        section.add_table(ReportTable::from_suite_summary(&sr.summary));
        section.add_table(ReportTable::from_benchmark_results(&sr.results));

        // Time distribution chart.
        let times: Vec<f64> = sr.results.iter().map(|r| r.total_time_ms).collect();
        let ids: Vec<f64> = (0..times.len()).map(|i| i as f64).collect();
        section.add_chart(ChartData::bar(
            &format!("{} — Time per Benchmark", sr.suite_name),
            &sr.results.iter().map(|r| r.benchmark_id.as_str()).collect::<Vec<_>>(),
            &times,
        ));

        section
    }

    fn performance_section(suite_results: &[SuiteResult]) -> ReportSection {
        let mut section = ReportSection::new("Performance Analysis");

        let all_times: Vec<f64> = suite_results
            .iter()
            .flat_map(|s| s.results.iter().map(|r| r.total_time_ms))
            .collect();
        let time_stats = compute_descriptive(&all_times);

        section.add_text(&format!(
            "Across all benchmarks: mean={:.2}ms, median={:.2}ms, std={:.2}ms, \
             min={:.2}ms, max={:.2}ms.",
            time_stats.mean, time_stats.median, time_stats.std_dev,
            time_stats.min, time_stats.max,
        ));
        section.add_metrics(time_stats);

        // Tier breakdown.
        let tier1_times: Vec<f64> = suite_results
            .iter()
            .flat_map(|s| s.results.iter().map(|r| r.tier1_time_ms))
            .collect();
        let tier2_times: Vec<f64> = suite_results
            .iter()
            .flat_map(|s| s.results.iter().map(|r| r.tier2_time_ms))
            .collect();

        let mut tier_table = ReportTable::new("Tier Timing Breakdown", vec![
            "Tier".into(), "Mean (ms)".into(), "Median (ms)".into(),
            "Std (ms)".into(), "Max (ms)".into(),
        ]);
        let t1_stats = compute_descriptive(&tier1_times);
        let t2_stats = compute_descriptive(&tier2_times);
        tier_table.add_row(vec![
            "Tier 1".into(), format!("{:.2}", t1_stats.mean),
            format!("{:.2}", t1_stats.median), format!("{:.2}", t1_stats.std_dev),
            format!("{:.2}", t1_stats.max),
        ]);
        tier_table.add_row(vec![
            "Tier 2".into(), format!("{:.2}", t2_stats.mean),
            format!("{:.2}", t2_stats.median), format!("{:.2}", t2_stats.std_dev),
            format!("{:.2}", t2_stats.max),
        ]);
        section.add_table(tier_table);

        section
    }

    fn statistical_summary_section(suite_results: &[SuiteResult]) -> ReportSection {
        let mut section = ReportSection::new("Statistical Summary");

        for sr in suite_results {
            let mut sub = ReportSection::new(&sr.suite_name);
            let times: Vec<f64> = sr.results.iter().map(|r| r.total_time_ms).collect();
            let stats = compute_descriptive(&times);
            sub.add_text(&format!("n={}, {}", stats.count, stats));
            sub.add_metrics(stats);
            section.add_subsection(sub);
        }

        section
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Markdown Formatter
// ═══════════════════════════════════════════════════════════════════════════

/// Render a report as Markdown.
pub fn format_markdown(report: &EvaluationReport) -> String {
    let mut out = String::with_capacity(8192);

    out.push_str(&format!("# {}\n\n", report.title));

    if !report.authors.is_empty() {
        out.push_str(&format!("**Authors:** {}\n\n", report.authors.join(", ")));
    }
    if !report.date.is_empty() {
        out.push_str(&format!("**Date:** {}\n\n", report.date));
    }
    if !report.abstract_text.is_empty() {
        out.push_str(&format!("## Abstract\n\n{}\n\n", report.abstract_text));
    }

    for section in &report.sections {
        format_section_md(&mut out, section, 2);
    }

    out
}

fn format_section_md(out: &mut String, section: &ReportSection, level: usize) {
    let hashes = "#".repeat(level);
    out.push_str(&format!("{} {}\n\n", hashes, section.title));

    for content in &section.content {
        match content {
            ReportContent::Text(t) => {
                out.push_str(t);
                out.push_str("\n\n");
            }
            ReportContent::Table(table) => {
                format_table_md(out, table);
            }
            ReportContent::Metrics(stats) => {
                out.push_str(&format!(
                    "- **Count:** {}\n- **Mean:** {:.4}\n- **Median:** {:.4}\n\
                     - **Std Dev:** {:.4}\n- **Min:** {:.4}\n- **Max:** {:.4}\n\
                     - **Q1:** {:.4}\n- **Q3:** {:.4}\n\n",
                    stats.count, stats.mean, stats.median, stats.std_dev,
                    stats.min, stats.max, stats.q1, stats.q3,
                ));
            }
            ReportContent::Chart(chart) => {
                out.push_str(&format!("*[Chart: {} ({})]*\n\n", chart.title, chart.chart_type));
            }
            ReportContent::ConfusionMatrixDisplay { tp, fp, fn_, tn, sensitivity, specificity, f1 } => {
                out.push_str("```\n");
                out.push_str("              Predicted+  Predicted-\n");
                out.push_str(&format!("Actual+       {:>8}    {:>8}\n", tp, fn_));
                out.push_str(&format!("Actual-       {:>8}    {:>8}\n", fp, tn));
                out.push_str("```\n\n");
                out.push_str(&format!(
                    "Sensitivity={:.3}, Specificity={:.3}, F1={:.3}\n\n",
                    sensitivity, specificity, f1,
                ));
            }
            ReportContent::KeyValue(pairs) => {
                for (k, v) in pairs {
                    out.push_str(&format!("- **{}:** {}\n", k, v));
                }
                out.push('\n');
            }
        }
    }

    for sub in &section.subsections {
        format_section_md(out, sub, level + 1);
    }
}

fn format_table_md(out: &mut String, table: &ReportTable) {
    if !table.caption.is_empty() {
        out.push_str(&format!("**{}**\n\n", table.caption));
    }

    // Header row.
    out.push_str("| ");
    for h in &table.headers {
        out.push_str(h);
        out.push_str(" | ");
    }
    out.push('\n');

    // Separator.
    out.push_str("| ");
    for (i, _) in table.headers.iter().enumerate() {
        let align = table.alignment.get(i).copied().unwrap_or(Alignment::Left);
        match align {
            Alignment::Left => out.push_str(":--- | "),
            Alignment::Center => out.push_str(":---: | "),
            Alignment::Right => out.push_str("---: | "),
        }
    }
    out.push('\n');

    // Data rows.
    for row in &table.rows {
        out.push_str("| ");
        for cell in row {
            out.push_str(cell);
            out.push_str(" | ");
        }
        out.push('\n');
    }
    out.push('\n');
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX Formatter
// ═══════════════════════════════════════════════════════════════════════════

/// Render a report as LaTeX.
pub fn format_latex(report: &EvaluationReport) -> String {
    let mut out = String::with_capacity(16384);

    out.push_str("\\documentclass{article}\n");
    out.push_str("\\usepackage{booktabs}\n");
    out.push_str("\\usepackage{graphicx}\n");
    out.push_str("\\usepackage{amsmath}\n\n");

    out.push_str(&format!("\\title{{{}}}\n", latex_escape(&report.title)));
    if !report.authors.is_empty() {
        out.push_str(&format!("\\author{{{}}}\n", report.authors.iter().map(|a| latex_escape(a)).collect::<Vec<_>>().join(" \\and ")));
    }
    if !report.date.is_empty() {
        out.push_str(&format!("\\date{{{}}}\n", latex_escape(&report.date)));
    }
    out.push_str("\n\\begin{document}\n\\maketitle\n\n");

    if !report.abstract_text.is_empty() {
        out.push_str("\\begin{abstract}\n");
        out.push_str(&latex_escape(&report.abstract_text));
        out.push_str("\n\\end{abstract}\n\n");
    }

    for section in &report.sections {
        format_section_latex(&mut out, section, 0);
    }

    out.push_str("\\end{document}\n");
    out
}

fn format_section_latex(out: &mut String, section: &ReportSection, depth: usize) {
    let cmd = match depth {
        0 => "section",
        1 => "subsection",
        _ => "subsubsection",
    };
    out.push_str(&format!("\\{}{{{}}}\n\n", cmd, latex_escape(&section.title)));

    for content in &section.content {
        match content {
            ReportContent::Text(t) => {
                out.push_str(&latex_escape(t));
                out.push_str("\n\n");
            }
            ReportContent::Table(table) => {
                format_table_latex(out, table);
            }
            ReportContent::Metrics(stats) => {
                out.push_str(&format!(
                    "Count: {}, Mean: {:.4}, Median: {:.4}, SD: {:.4}, Range: [{:.4}, {:.4}]\n\n",
                    stats.count, stats.mean, stats.median, stats.std_dev, stats.min, stats.max,
                ));
            }
            ReportContent::Chart(chart) => {
                out.push_str(&format!("% Chart placeholder: {} ({})\n\n", chart.title, chart.chart_type));
            }
            ReportContent::ConfusionMatrixDisplay { tp, fp, fn_, tn, sensitivity, specificity, f1 } => {
                out.push_str("\\begin{verbatim}\n");
                out.push_str(&format!("TP={} FP={} FN={} TN={}\n", tp, fp, fn_, tn));
                out.push_str(&format!("Sens={:.3} Spec={:.3} F1={:.3}\n", sensitivity, specificity, f1));
                out.push_str("\\end{verbatim}\n\n");
            }
            ReportContent::KeyValue(pairs) => {
                out.push_str("\\begin{description}\n");
                for (k, v) in pairs {
                    out.push_str(&format!("  \\item[{}] {}\n", latex_escape(k), latex_escape(v)));
                }
                out.push_str("\\end{description}\n\n");
            }
        }
    }

    for sub in &section.subsections {
        format_section_latex(out, sub, depth + 1);
    }
}

fn format_table_latex(out: &mut String, table: &ReportTable) {
    let ncols = table.headers.len();
    let col_spec: String = table.alignment.iter().map(|a| match a {
        Alignment::Left => 'l',
        Alignment::Center => 'c',
        Alignment::Right => 'r',
    }).collect();
    let col_spec = if col_spec.is_empty() {
        "l".repeat(ncols)
    } else {
        col_spec
    };

    out.push_str("\\begin{table}[h]\n\\centering\n");
    out.push_str(&format!("\\caption{{{}}}\n", latex_escape(&table.caption)));
    out.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
    out.push_str("\\toprule\n");

    // Headers.
    let headers: Vec<String> = table.headers.iter().map(|h| latex_escape(h)).collect();
    out.push_str(&headers.join(" & "));
    out.push_str(" \\\\\n\\midrule\n");

    // Rows.
    for row in &table.rows {
        let cells: Vec<String> = row.iter().map(|c| latex_escape(c)).collect();
        out.push_str(&cells.join(" & "));
        out.push_str(" \\\\\n");
    }

    out.push_str("\\bottomrule\n");
    out.push_str("\\end{tabular}\n\\end{table}\n\n");
}

fn latex_escape(s: &str) -> String {
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

// ═══════════════════════════════════════════════════════════════════════════
// JSON Formatter
// ═══════════════════════════════════════════════════════════════════════════

/// Render a report as JSON.
pub fn format_json(report: &EvaluationReport) -> serde_json::Value {
    serde_json::to_value(report).unwrap_or(serde_json::Value::Null)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{ActualVerdict, BenchmarkResult, ExpectedVerdict, SuiteResult};
    use crate::baseline::BaselineComparison;

    fn sample_results() -> Vec<BenchmarkResult> {
        vec![
            BenchmarkResult::success("b1", ActualVerdict::Safe, ExpectedVerdict::Safe, 10.0, 0.0, 1024, 0, 2),
            BenchmarkResult::success("b2", ActualVerdict::Unsafe, ExpectedVerdict::Unsafe, 8.0, 12.0, 2048, 3, 2),
            BenchmarkResult::success("b3", ActualVerdict::Safe, ExpectedVerdict::Unsafe, 5.0, 0.0, 512, 0, 1),
        ]
    }

    fn sample_suite_result() -> SuiteResult {
        SuiteResult::from_results("test-suite", sample_results(), 100.0)
    }

    fn sample_baseline() -> BaselineComparison {
        BaselineComparison {
            true_positives: 5,
            false_negatives_baseline: 2,
            false_positives_baseline: 1,
            true_negatives: 42,
            total_pairs: 50,
            pair_details: vec![],
        }
    }

    #[test]
    fn test_generate_report() {
        let report = ReportGenerator::generate_evaluation_report(
            &[sample_suite_result()],
            &sample_baseline(),
        );
        assert!(!report.sections.is_empty());
        assert!(report.section_count() >= 4);
    }

    #[test]
    fn test_format_markdown() {
        let report = ReportGenerator::generate_evaluation_report(
            &[sample_suite_result()],
            &sample_baseline(),
        );
        let md = format_markdown(&report);
        assert!(md.contains("# GuardPharma"));
        assert!(md.contains("Executive Summary"));
        assert!(md.contains("Baseline Comparison"));
    }

    #[test]
    fn test_format_latex() {
        let report = ReportGenerator::generate_evaluation_report(
            &[sample_suite_result()],
            &sample_baseline(),
        );
        let latex = format_latex(&report);
        assert!(latex.contains("\\documentclass"));
        assert!(latex.contains("\\begin{document}"));
        assert!(latex.contains("\\end{document}"));
        assert!(latex.contains("\\begin{tabular}"));
    }

    #[test]
    fn test_format_json() {
        let report = ReportGenerator::generate_evaluation_report(
            &[sample_suite_result()],
            &sample_baseline(),
        );
        let json = format_json(&report);
        assert!(json.is_object());
        assert!(json.get("title").is_some());
        assert!(json.get("sections").is_some());
    }

    #[test]
    fn test_report_table_from_summary() {
        let sr = sample_suite_result();
        let table = ReportTable::from_suite_summary(&sr.summary);
        assert!(table.row_count() >= 5);
    }

    #[test]
    fn test_report_table_from_results() {
        let results = sample_results();
        let table = ReportTable::from_benchmark_results(&results);
        assert_eq!(table.row_count(), 3);
        assert_eq!(table.column_count(), 9);
    }

    #[test]
    fn test_chart_data_bar() {
        let chart = ChartData::bar("Test", &["A", "B"], &[1.0, 2.0]);
        assert_eq!(chart.series.len(), 1);
        assert_eq!(chart.series[0].y_values.len(), 2);
    }

    #[test]
    fn test_chart_data_line() {
        let chart = ChartData::line("T", "x", "y", &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(chart.series.len(), 1);
    }

    #[test]
    fn test_chart_data_multi_line() {
        let s1 = ChartSeries { name: "A".into(), x_values: vec![1.0], y_values: vec![2.0] };
        let s2 = ChartSeries { name: "B".into(), x_values: vec![1.0], y_values: vec![3.0] };
        let chart = ChartData::multi_line("T", "x", "y", vec![s1, s2]);
        assert_eq!(chart.series.len(), 2);
    }

    #[test]
    fn test_section_builder() {
        let section = ReportSection::new("Test")
            .with_text("Hello")
            .with_table(ReportTable::new("T", vec!["A".into()]));
        assert_eq!(section.content.len(), 2);
    }

    #[test]
    fn test_latex_escape() {
        assert_eq!(latex_escape("a & b"), "a \\& b");
        assert_eq!(latex_escape("100%"), "100\\%");
        assert_eq!(latex_escape("x_1"), "x\\_1");
    }

    #[test]
    fn test_report_with_metadata() {
        let report = EvaluationReport::new("Test").with_author("Author").with_date("2024-01-01");
        assert_eq!(report.authors.len(), 1);
        assert_eq!(report.date, "2024-01-01");
    }

    #[test]
    fn test_markdown_contains_tables() {
        let report = ReportGenerator::generate_evaluation_report(
            &[sample_suite_result()],
            &sample_baseline(),
        );
        let md = format_markdown(&report);
        assert!(md.contains("| "), "Markdown should contain table separators");
        assert!(md.contains("---"), "Markdown should contain table alignment");
    }
}
