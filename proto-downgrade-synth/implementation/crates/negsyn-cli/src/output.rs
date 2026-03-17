//! Output formatting for the NegSynth CLI.
//!
//! Provides a unified [`OutputFormat`] enum, an [`OutputWriter`] that
//! dispatches to format-specific writers, progress indicators, coloured
//! terminal helpers, table formatting, and JSON pretty-printing.

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Human-readable text (default).
    Text,
    /// JSON (pretty-printed).
    Json,
    /// SARIF (Static Analysis Results Interchange Format).
    Sarif,
    /// GraphViz DOT.
    Dot,
    /// Comma-separated values.
    Csv,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Text
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
            Self::Sarif => write!(f, "sarif"),
            Self::Dot => write!(f, "dot"),
            Self::Csv => write!(f, "csv"),
        }
    }
}

impl OutputFormat {
    /// File extension commonly associated with this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Text => "txt",
            Self::Json => "json",
            Self::Sarif => "sarif",
            Self::Dot => "dot",
            Self::Csv => "csv",
        }
    }
}

// ---------------------------------------------------------------------------
// OutputWriter
// ---------------------------------------------------------------------------

/// Writes analysis results to stdout or a file in the requested format.
pub struct OutputWriter {
    format: OutputFormat,
    no_color: bool,
    destination: Destination,
}

enum Destination {
    Stdout,
    File(std::fs::File),
}

impl OutputWriter {
    /// Create a writer targeting stdout.
    pub fn stdout(format: OutputFormat, no_color: bool) -> Self {
        Self {
            format,
            no_color,
            destination: Destination::Stdout,
        }
    }

    /// Create a writer targeting a file.
    pub fn file(path: &Path, format: OutputFormat, no_color: bool) -> Result<Self> {
        let f = std::fs::File::create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;
        Ok(Self {
            format,
            no_color,
            destination: Destination::File(f),
        })
    }

    pub fn format(&self) -> OutputFormat {
        self.format
    }

    /// Write an arbitrary `Serialize` value in the configured format.
    pub fn write_value<T: Serialize>(&mut self, value: &T) -> Result<()> {
        let text = match self.format {
            OutputFormat::Json | OutputFormat::Sarif => {
                serde_json::to_string_pretty(value).context("JSON serialization failed")?
            }
            OutputFormat::Text => {
                // Use Debug representation for plain text fallback.
                serde_json::to_string_pretty(value).context("serialization")?
            }
            OutputFormat::Csv | OutputFormat::Dot => {
                serde_json::to_string_pretty(value).context("serialization")?
            }
        };
        self.write_raw(&text)
    }

    /// Write a pre-formatted string.
    pub fn write_raw(&mut self, text: &str) -> Result<()> {
        match &mut self.destination {
            Destination::Stdout => {
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                handle.write_all(text.as_bytes())?;
                if !text.ends_with('\n') {
                    handle.write_all(b"\n")?;
                }
                handle.flush()?;
            }
            Destination::File(f) => {
                f.write_all(text.as_bytes())?;
                if !text.ends_with('\n') {
                    f.write_all(b"\n")?;
                }
                f.flush()?;
            }
        }
        Ok(())
    }

    /// Write a table (header + rows) in the configured format.
    pub fn write_table(&mut self, table: &Table) -> Result<()> {
        match self.format {
            OutputFormat::Text => self.write_raw(&table.render_text(self.no_color)),
            OutputFormat::Csv => self.write_raw(&table.render_csv()),
            OutputFormat::Json => {
                let json = table.to_json_value();
                let s = serde_json::to_string_pretty(&json)?;
                self.write_raw(&s)
            }
            _ => self.write_raw(&table.render_text(self.no_color)),
        }
    }

    /// Helper: `true` when colour codes should be emitted.
    pub fn use_color(&self) -> bool {
        !self.no_color && matches!(self.destination, Destination::Stdout)
    }
}

// ---------------------------------------------------------------------------
// Table formatting
// ---------------------------------------------------------------------------

/// A simple table with headers and rows.
#[derive(Debug, Clone)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub title: Option<String>,
}

impl Table {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
            title: None,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Compute column widths as max(header, all rows) for each column.
    fn column_widths(&self) -> Vec<usize> {
        let ncols = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < ncols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }
        widths
    }

    /// Render as an ASCII table.
    pub fn render_text(&self, _no_color: bool) -> String {
        let widths = self.column_widths();
        let mut out = String::new();

        if let Some(ref title) = self.title {
            writeln!(out, "\n  {title}").unwrap();
            writeln!(out, "  {}", "─".repeat(title.len())).unwrap();
        }

        // Header
        write!(out, "  ").unwrap();
        for (i, hdr) in self.headers.iter().enumerate() {
            if i > 0 {
                write!(out, "  │ ").unwrap();
            }
            write!(out, "{:<width$}", hdr, width = widths[i]).unwrap();
        }
        writeln!(out).unwrap();

        // Separator
        write!(out, "  ").unwrap();
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                write!(out, "──┼─").unwrap();
            }
            write!(out, "{}", "─".repeat(*w)).unwrap();
        }
        writeln!(out).unwrap();

        // Rows
        for row in &self.rows {
            write!(out, "  ").unwrap();
            for (i, cell) in row.iter().enumerate() {
                if i > 0 {
                    write!(out, "  │ ").unwrap();
                }
                let w = widths.get(i).copied().unwrap_or(cell.len());
                write!(out, "{:<width$}", cell, width = w).unwrap();
            }
            writeln!(out).unwrap();
        }

        out
    }

    /// Render as CSV.
    pub fn render_csv(&self) -> String {
        let mut out = String::new();
        out.push_str(&csv_escape_row(&self.headers));
        out.push('\n');
        for row in &self.rows {
            out.push_str(&csv_escape_row(row));
            out.push('\n');
        }
        out
    }

    /// Convert to a JSON array of objects.
    pub fn to_json_value(&self) -> serde_json::Value {
        let rows: Vec<serde_json::Value> = self
            .rows
            .iter()
            .map(|row| {
                let mut obj = serde_json::Map::new();
                for (i, hdr) in self.headers.iter().enumerate() {
                    let cell = row.get(i).cloned().unwrap_or_default();
                    obj.insert(hdr.clone(), serde_json::Value::String(cell));
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        serde_json::Value::Array(rows)
    }
}

fn csv_escape_row(cells: &[String]) -> String {
    cells
        .iter()
        .map(|c| {
            if c.contains(',') || c.contains('"') || c.contains('\n') {
                format!("\"{}\"", c.replace('"', "\"\""))
            } else {
                c.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// Progress indicator
// ---------------------------------------------------------------------------

/// A simple progress spinner for long-running operations.
pub struct ProgressSpinner {
    label: String,
    start: Instant,
    frames: &'static [&'static str],
    frame_idx: usize,
    enabled: bool,
}

impl ProgressSpinner {
    const FRAMES: &'static [&'static str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

    pub fn new(label: impl Into<String>, enabled: bool) -> Self {
        let label = label.into();
        if enabled {
            eprint!("  {} {}…", Self::FRAMES[0], label);
        }
        Self {
            label,
            start: Instant::now(),
            frames: Self::FRAMES,
            frame_idx: 0,
            enabled,
        }
    }

    /// Advance the spinner animation (call periodically).
    pub fn tick(&mut self) {
        if !self.enabled {
            return;
        }
        self.frame_idx = (self.frame_idx + 1) % self.frames.len();
        let elapsed = self.start.elapsed().as_secs_f64();
        eprint!(
            "\r  {} {}… ({:.1}s)",
            self.frames[self.frame_idx], self.label, elapsed
        );
        let _ = io::stderr().flush();
    }

    /// Finish the spinner, printing a completion message.
    pub fn finish(self, message: &str) {
        if self.enabled {
            let elapsed = self.start.elapsed().as_secs_f64();
            eprintln!("\r  ✓ {} — {} ({:.3}s)", self.label, message, elapsed);
        }
    }

    /// Finish with a failure marker.
    pub fn fail(self, message: &str) {
        if self.enabled {
            let elapsed = self.start.elapsed().as_secs_f64();
            eprintln!("\r  ✗ {} — {} ({:.3}s)", self.label, message, elapsed);
        }
    }
}

// ---------------------------------------------------------------------------
// Coloured helpers
// ---------------------------------------------------------------------------

pub fn red(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[31m{s}\x1b[0m") }
}

pub fn green(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[32m{s}\x1b[0m") }
}

pub fn yellow(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[33m{s}\x1b[0m") }
}

pub fn bold(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[1m{s}\x1b[0m") }
}

pub fn dim(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[2m{s}\x1b[0m") }
}

pub fn cyan(s: &str, no_color: bool) -> String {
    if no_color { s.to_string() } else { format!("\x1b[36m{s}\x1b[0m") }
}

// ---------------------------------------------------------------------------
// SARIF helpers
// ---------------------------------------------------------------------------

/// Minimal SARIF v2.1.0 envelope.
#[derive(Debug, Clone, Serialize)]
pub struct SarifReport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifDriver {
    pub name: String,
    pub version: String,
    #[serde(rename = "informationUri")]
    pub information_uri: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifMessage {
    pub text: String,
}

impl SarifReport {
    pub fn new() -> Self {
        Self {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json".into(),
            version: "2.1.0".into(),
            runs: vec![SarifRun {
                tool: SarifTool {
                    driver: SarifDriver {
                        name: "negsyn".into(),
                        version: env!("CARGO_PKG_VERSION").into(),
                        information_uri: "https://github.com/negsyn/negsyn".into(),
                    },
                },
                results: Vec::new(),
            }],
        }
    }

    pub fn add_result(&mut self, rule_id: &str, level: &str, message: &str) {
        if let Some(run) = self.runs.first_mut() {
            run.results.push(SarifResult {
                rule_id: rule_id.into(),
                level: level.into(),
                message: SarifMessage {
                    text: message.into(),
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// DOT graph helper
// ---------------------------------------------------------------------------

/// Build a DOT graph from labeled nodes and edges.
pub fn render_dot(
    name: &str,
    nodes: &[(String, BTreeMap<String, String>)],
    edges: &[(String, String, String)],
) -> String {
    let mut out = format!("digraph {name} {{\n");
    writeln!(out, "  rankdir=LR;").unwrap();
    writeln!(out, "  node [shape=record fontname=\"monospace\"];").unwrap();

    for (id, attrs) in nodes {
        let label_parts: Vec<String> = attrs
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect();
        let label = if label_parts.is_empty() {
            id.clone()
        } else {
            format!("{}|{}", id, label_parts.join("\\n"))
        };
        writeln!(out, "  \"{id}\" [label=\"{{{label}}}\"];").unwrap();
    }

    for (from, to, label) in edges {
        writeln!(out, "  \"{from}\" -> \"{to}\" [label=\"{label}\"];").unwrap();
    }

    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Text.to_string(), "text");
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
    }

    #[test]
    fn output_format_extension() {
        assert_eq!(OutputFormat::Sarif.extension(), "sarif");
        assert_eq!(OutputFormat::Dot.extension(), "dot");
    }

    #[test]
    fn table_render_text() {
        let mut t = Table::new(vec!["Name".into(), "Value".into()]);
        t.add_row(vec!["alpha".into(), "1".into()]);
        t.add_row(vec!["beta".into(), "22".into()]);
        let text = t.render_text(true);
        assert!(text.contains("Name"));
        assert!(text.contains("alpha"));
        assert!(text.contains("22"));
    }

    #[test]
    fn table_render_csv() {
        let mut t = Table::new(vec!["A".into(), "B".into()]);
        t.add_row(vec!["hello".into(), "world".into()]);
        t.add_row(vec!["with,comma".into(), "ok".into()]);
        let csv = t.render_csv();
        assert!(csv.contains("A,B"));
        assert!(csv.contains("\"with,comma\""));
    }

    #[test]
    fn table_to_json() {
        let mut t = Table::new(vec!["x".into(), "y".into()]);
        t.add_row(vec!["1".into(), "2".into()]);
        let val = t.to_json_value();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["x"], "1");
    }

    #[test]
    fn csv_escape_quotes() {
        let r = csv_escape_row(&["a\"b".to_string()]);
        assert_eq!(r, "\"a\"\"b\"");
    }

    #[test]
    fn colour_helpers_no_color() {
        assert_eq!(red("err", true), "err");
        assert_eq!(green("ok", true), "ok");
        assert_eq!(bold("b", true), "b");
    }

    #[test]
    fn colour_helpers_with_color() {
        let s = red("err", false);
        assert!(s.contains("\x1b[31m"));
    }

    #[test]
    fn sarif_report_add_result() {
        let mut r = SarifReport::new();
        r.add_result("DOWNGRADE-001", "error", "TLS downgrade found");
        assert_eq!(r.runs[0].results.len(), 1);
    }

    #[test]
    fn render_dot_basic() {
        let nodes = vec![
            ("s0".into(), BTreeMap::from([("phase".into(), "initial".into())])),
            ("s1".into(), BTreeMap::new()),
        ];
        let edges = vec![("s0".into(), "s1".into(), "hello".into())];
        let dot = render_dot("test", &nodes, &edges);
        assert!(dot.contains("digraph test"));
        assert!(dot.contains("s0"));
        assert!(dot.contains("-> \"s1\""));
    }

    #[test]
    fn progress_spinner_disabled() {
        let s = ProgressSpinner::new("test", false);
        s.finish("ok"); // should not panic
    }

    #[test]
    fn table_with_title() {
        let t = Table::new(vec!["X".into()])
            .with_title("My Table");
        let text = t.render_text(true);
        assert!(text.contains("My Table"));
    }

    #[test]
    fn column_widths_computed() {
        let mut t = Table::new(vec!["AB".into(), "C".into()]);
        t.add_row(vec!["ABCDE".into(), "xy".into()]);
        let w = t.column_widths();
        assert_eq!(w, vec![5, 2]);
    }
}
