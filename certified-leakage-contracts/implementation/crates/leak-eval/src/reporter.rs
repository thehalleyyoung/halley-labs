//! Report generation for benchmark and comparison results.
//!
//! Supports multiple output formats: JSON, CSV, LaTeX tables, and structured
//! benchmark reports for integration into papers and dashboards.

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::comparator::ComparisonReport;

/// Configuration for report generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Title of the report.
    pub title: String,
    /// Whether to include raw data tables.
    pub include_raw_data: bool,
    /// Whether to include statistical summaries.
    pub include_statistics: bool,
    /// Decimal places for floating-point values.
    pub decimal_places: usize,
    /// Whether to sort results by analysis time.
    pub sort_by_time: bool,
}

impl ReportConfig {
    /// Create a default report configuration.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            include_raw_data: true,
            include_statistics: true,
            decimal_places: 3,
            sort_by_time: false,
        }
    }
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self::new("Benchmark Report")
    }
}

/// A structured benchmark report combining results and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report configuration.
    pub config: ReportConfig,
    /// Benchmark results included in this report.
    pub results: Vec<BenchmarkResult>,
    /// Timestamp of report generation (ISO 8601).
    pub generated_at: String,
    /// Optional notes or commentary.
    pub notes: Vec<String>,
}

impl BenchmarkReport {
    /// Create a new report from results.
    pub fn new(config: ReportConfig, results: Vec<BenchmarkResult>) -> Self {
        Self {
            config,
            results,
            generated_at: chrono::Utc::now().to_rfc3339(),
            notes: Vec::new(),
        }
    }

    /// Add a note to the report.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Number of results in the report.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Number of successful results.
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }
}

/// Comparison table for side-by-side tool evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTable {
    /// Column headers (tool names).
    pub headers: Vec<String>,
    /// Row labels (benchmark names).
    pub row_labels: Vec<String>,
    /// Cell values: rows × columns.
    pub cells: Vec<Vec<String>>,
}

impl ComparisonTable {
    /// Build a comparison table from a comparison report, using a value extractor.
    pub fn from_report<F>(report: &ComparisonReport, extract: F) -> Self
    where
        F: Fn(&BenchmarkResult) -> String,
    {
        let headers: Vec<String> = report.tools.iter().map(|t| t.name.clone()).collect();

        // Collect all benchmark names.
        let mut row_labels: Vec<String> = Vec::new();
        for results in report.results.values() {
            for r in results {
                if !row_labels.contains(&r.benchmark_name) {
                    row_labels.push(r.benchmark_name.clone());
                }
            }
        }
        row_labels.sort();

        let mut cells = Vec::new();
        for label in &row_labels {
            let mut row = Vec::new();
            for header in &headers {
                let val = report
                    .results
                    .get(header)
                    .and_then(|rs| rs.iter().find(|r| &r.benchmark_name == label))
                    .map(|r| extract(r))
                    .unwrap_or_else(|| "—".to_string());
                row.push(val);
            }
            cells.push(row);
        }

        Self {
            headers,
            row_labels,
            cells,
        }
    }
}

/// JSON report output.
#[derive(Debug)]
pub struct JsonReport;

impl JsonReport {
    /// Write a benchmark report as JSON.
    pub fn write<W: Write>(writer: &mut W, report: &BenchmarkReport) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.write_all(json.as_bytes())
    }
}

/// CSV report output.
#[derive(Debug)]
pub struct CsvReport;

impl CsvReport {
    /// Write benchmark results as CSV.
    pub fn write<W: Write>(writer: &mut W, results: &[BenchmarkResult]) -> std::io::Result<()> {
        writeln!(
            writer,
            "benchmark,category,elapsed_ms,leaking_sets,leakage_bits,success,timed_out"
        )?;
        for r in results {
            writeln!(
                writer,
                "{},{:?},{},{},{:.6},{},{}",
                r.benchmark_name,
                r.category,
                r.elapsed.as_millis(),
                r.reported_leaking_sets,
                r.reported_leakage_bits,
                r.success,
                r.timed_out,
            )?;
        }
        Ok(())
    }
}

/// LaTeX table output for inclusion in papers.
#[derive(Debug)]
pub struct LatexTable;

impl LatexTable {
    /// Render a comparison table as a LaTeX tabular environment.
    pub fn render(table: &ComparisonTable) -> String {
        let num_cols = table.headers.len() + 1; // +1 for row label column
        let col_spec = format!("l{}", "r".repeat(table.headers.len()));

        let mut out = String::new();
        out.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        out.push_str("\\toprule\n");

        // Header row.
        out.push_str("Benchmark");
        for h in &table.headers {
            out.push_str(&format!(" & {}", h));
        }
        out.push_str(" \\\\\n\\midrule\n");

        // Data rows.
        for (i, label) in table.row_labels.iter().enumerate() {
            out.push_str(label);
            if let Some(row) = table.cells.get(i) {
                for cell in row {
                    out.push_str(&format!(" & {}", cell));
                }
            }
            out.push_str(" \\\\\n");
        }

        out.push_str("\\bottomrule\n");
        out.push_str(&format!("\\end{{tabular}}\n"));
        let _ = num_cols; // used for col_spec
        out
    }
}
