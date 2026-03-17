//! Output formatting for the MutSpec CLI.
//!
//! Provides formatters for JSON, aligned text tables, Markdown, and SARIF, plus
//! progress-bar and colour helpers.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use clap::ValueEnum;
use serde::Serialize;

use shared_types::operators::{MutantDescriptor, MutantStatus, MutationOperator};

// ---------------------------------------------------------------------------
// Output format enum (CLI-level; avoids name collision with shared_types).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CliOutputFormat {
    Text,
    Json,
    Markdown,
    Sarif,
}

impl fmt::Display for CliOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
            Self::Markdown => write!(f, "markdown"),
            Self::Sarif => write!(f, "sarif"),
        }
    }
}

// ---------------------------------------------------------------------------
// Verbosity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
    Debug,
    Trace,
}

impl From<u8> for Verbosity {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::Verbose,
            2 => Self::Debug,
            _ => Self::Trace,
        }
    }
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

pub struct Colour {
    enabled: bool,
}

impl Colour {
    pub fn new(no_color: bool) -> Self {
        Self {
            enabled: !no_color && atty_stdout(),
        }
    }

    pub fn bold(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn green(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn red(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn yellow(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn cyan(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn dim(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn magenta(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[35m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn status_colour(&self, status: &MutantStatus) -> String {
        match status {
            MutantStatus::Killed => self.green("Killed"),
            MutantStatus::Alive => self.red("Alive"),
            MutantStatus::Equivalent => self.yellow("Equivalent"),
            MutantStatus::Timeout => self.yellow("Timeout"),
            MutantStatus::Error(_) => self.red("Error"),
        }
    }
}

fn atty_stdout() -> bool {
    std::env::var("TERM").is_ok()
}

// ---------------------------------------------------------------------------
// Aligned text table
// ---------------------------------------------------------------------------

pub struct AlignedTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    right_align: Vec<bool>,
}

impl AlignedTable {
    pub fn new(headers: Vec<String>) -> Self {
        let ncols = headers.len();
        Self {
            headers,
            rows: Vec::new(),
            right_align: vec![false; ncols],
        }
    }

    pub fn set_right_align(&mut self, col: usize) {
        if col < self.right_align.len() {
            self.right_align[col] = true;
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn column_widths(&self) -> Vec<usize> {
        let ncols = self.headers.len();
        let mut widths = vec![0usize; ncols];
        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = strip_ansi(h).len();
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < ncols {
                    widths[i] = widths[i].max(strip_ansi(cell).len());
                }
            }
        }
        widths
    }

    fn write_row(
        &self,
        f: &mut fmt::Formatter<'_>,
        row: &[String],
        widths: &[usize],
    ) -> fmt::Result {
        let ncols = self.headers.len();
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                write!(f, " | ")?;
            }
            let w = if i < ncols {
                widths[i]
            } else {
                strip_ansi(cell).len()
            };
            let visible_len = strip_ansi(cell).len();
            let padding = w.saturating_sub(visible_len);
            if i < self.right_align.len() && self.right_align[i] {
                write!(f, "{}{}", " ".repeat(padding), cell)?;
            } else {
                write!(f, "{}{}", cell, " ".repeat(padding))?;
            }
        }
        writeln!(f)
    }
}

impl fmt::Display for AlignedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = self.column_widths();
        self.write_row(f, &self.headers, &widths)?;
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                write!(f, "-+-")?;
            }
            write!(f, "{}", "-".repeat(*w))?;
        }
        writeln!(f)?;
        for row in &self.rows {
            self.write_row(f, row, &widths)?;
        }
        Ok(())
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            out.push(ch);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Markdown table
// ---------------------------------------------------------------------------

pub struct MarkdownTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl MarkdownTable {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

impl fmt::Display for MarkdownTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "|")?;
        for h in &self.headers {
            write!(f, " {} |", h)?;
        }
        writeln!(f)?;
        write!(f, "|")?;
        for h in &self.headers {
            write!(f, " {} |", "-".repeat(h.len().max(3)))?;
        }
        writeln!(f)?;
        for row in &self.rows {
            write!(f, "|")?;
            for (i, cell) in row.iter().enumerate() {
                let min_w = self.headers.get(i).map(|h| h.len().max(3)).unwrap_or(3);
                write!(f, " {:<width$} |", cell, width = min_w)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

pub fn write_json<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let json =
        serde_json::to_string_pretty(value).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writeln!(writer, "{json}")
}

pub fn write_json_compact<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let json = serde_json::to_string(value).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writeln!(writer, "{json}")
}

// ---------------------------------------------------------------------------
// SARIF builder
// ---------------------------------------------------------------------------

pub struct SarifBuilder {
    tool_name: String,
    tool_version: String,
    results: Vec<SarifResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifMessage {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    pub physical_location: SarifPhysicalLocation,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: SarifArtifactLocation,
    pub region: SarifRegion,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRegion {
    #[serde(rename = "startLine")]
    pub start_line: usize,
    #[serde(rename = "startColumn")]
    pub start_column: usize,
    #[serde(rename = "endLine", skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(rename = "endColumn", skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

impl SarifBuilder {
    pub fn new(tool_name: impl Into<String>, tool_version: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_version: tool_version.into(),
            results: Vec::new(),
        }
    }

    pub fn add_result(
        &mut self,
        rule_id: impl Into<String>,
        level: impl Into<String>,
        message: impl Into<String>,
        file: impl Into<String>,
        line: usize,
        column: usize,
    ) {
        self.results.push(SarifResult {
            rule_id: rule_id.into(),
            level: level.into(),
            message: SarifMessage {
                text: message.into(),
            },
            locations: vec![SarifLocation {
                physical_location: SarifPhysicalLocation {
                    artifact_location: SarifArtifactLocation { uri: file.into() },
                    region: SarifRegion {
                        start_line: line,
                        start_column: column,
                        end_line: None,
                        end_column: None,
                    },
                },
            }],
        });
    }

    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    pub fn build(&self) -> serde_json::Value {
        serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": &self.tool_name,
                        "version": &self.tool_version,
                        "informationUri": "https://github.com/mutspec/mutspec",
                        "rules": self.collect_rules(),
                    }
                },
                "results": &self.results,
            }]
        })
    }

    fn collect_rules(&self) -> Vec<serde_json::Value> {
        let mut seen = std::collections::HashSet::new();
        let mut rules = Vec::new();
        for r in &self.results {
            if seen.insert(r.rule_id.clone()) {
                rules.push(serde_json::json!({
                    "id": r.rule_id,
                    "shortDescription": { "text": format!("MutSpec rule {}", r.rule_id) },
                }));
            }
        }
        rules
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let doc = self.build();
        let json = serde_json::to_string_pretty(&doc)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        writeln!(writer, "{json}")
    }
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

pub struct ProgressBar {
    total: usize,
    current: usize,
    label: String,
    started: Instant,
    last_draw: Option<Instant>,
    bar_width: usize,
    enabled: bool,
}

impl ProgressBar {
    pub fn new(total: usize, label: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            label: label.into(),
            started: Instant::now(),
            last_draw: None,
            bar_width: 40,
            enabled: atty_stderr(),
        }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn set_total(&mut self, total: usize) {
        self.total = total;
    }

    pub fn inc(&mut self) {
        self.current += 1;
        self.draw();
    }

    pub fn set(&mut self, value: usize) {
        self.current = value;
        self.draw();
    }

    pub fn finish(&mut self) {
        self.current = self.total;
        self.draw();
        if self.enabled {
            eprintln!();
        }
    }

    pub fn finish_with_message(&mut self, msg: &str) {
        self.current = self.total;
        if self.enabled {
            eprint!("\r\x1b[K{msg}");
            eprintln!();
        }
    }

    fn draw(&mut self) {
        if !self.enabled || self.total == 0 {
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last_draw {
            if now.duration_since(last) < Duration::from_millis(50) && self.current < self.total {
                return;
            }
        }
        self.last_draw = Some(now);
        let pct = (self.current as f64 / self.total as f64).min(1.0);
        let filled = (pct * self.bar_width as f64) as usize;
        let empty = self.bar_width - filled;
        let elapsed = self.started.elapsed();
        let eta = if pct > 0.0 && pct < 1.0 {
            let remaining = elapsed.as_secs_f64() * (1.0 - pct) / pct;
            format_duration(Duration::from_secs_f64(remaining))
        } else {
            "–".to_string()
        };
        eprint!(
            "\r\x1b[K{} [{}>{}] {}/{} ({:.0}%) ETA {}",
            self.label,
            "=".repeat(filled),
            " ".repeat(empty),
            self.current,
            self.total,
            pct * 100.0,
            eta
        );
    }
}

fn atty_stderr() -> bool {
    std::env::var("TERM").is_ok()
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

// ---------------------------------------------------------------------------
// Mutant table helpers
// ---------------------------------------------------------------------------

pub fn mutant_table(mutants: &[MutantDescriptor], colour: &Colour) -> AlignedTable {
    let mut tbl = AlignedTable::new(vec![
        "ID".into(),
        "Operator".into(),
        "Location".into(),
        "Original".into(),
        "Replacement".into(),
    ]);
    for m in mutants {
        let loc_str = format!(
            "{}:{}",
            m.site.location.start.line, m.site.location.start.column
        );
        tbl.add_row(vec![
            colour.cyan(&m.id.short()),
            m.operator.mnemonic().to_string(),
            loc_str,
            colour.red(&m.site.original),
            colour.green(&m.site.replacement),
        ]);
    }
    tbl
}

pub fn mutant_status_table(mutants: &[MutantDescriptor], colour: &Colour) -> AlignedTable {
    let mut tbl = AlignedTable::new(vec![
        "ID".into(),
        "Operator".into(),
        "Status".into(),
        "Original".into(),
        "Replacement".into(),
    ]);
    for m in mutants {
        tbl.add_row(vec![
            colour.cyan(&m.id.short()),
            m.operator.mnemonic().to_string(),
            colour.status_colour(&m.status),
            m.site.original.clone(),
            m.site.replacement.clone(),
        ]);
    }
    tbl
}

// ---------------------------------------------------------------------------
// Contract formatting
// ---------------------------------------------------------------------------

pub fn format_contract_text(
    contract: &shared_types::contracts::Contract,
    colour: &Colour,
) -> String {
    let mut buf = String::new();
    buf.push_str(&colour.bold(&format!(
        "Contract for function `{}`",
        contract.function_name
    )));
    buf.push('\n');
    buf.push_str(&format!("  Strength: {}\n", contract.strength.name()));
    buf.push_str(&format!("  Verified: {}\n", contract.verified));
    buf.push_str(&format!("  Clauses:  {}\n", contract.clause_count()));
    for clause in &contract.clauses {
        buf.push_str(&format!(
            "    {} {}\n",
            clause.kind_name(),
            clause.formula()
        ));
    }
    buf
}

pub fn format_contract_jml(contract: &shared_types::contracts::Contract) -> String {
    contract.to_jml()
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct MutationStats {
    pub total_mutants: usize,
    pub killed: usize,
    pub alive: usize,
    pub equivalent: usize,
    pub timeout: usize,
    pub error: usize,
    pub mutation_score: f64,
    pub operators: BTreeMap<String, usize>,
}

impl MutationStats {
    pub fn from_descriptors(descs: &[MutantDescriptor]) -> Self {
        let mut stats = Self::default();
        stats.total_mutants = descs.len();
        for d in descs {
            let op_name = d.operator.mnemonic().to_string();
            *stats.operators.entry(op_name).or_insert(0) += 1;
            match &d.status {
                MutantStatus::Killed => stats.killed += 1,
                MutantStatus::Alive => stats.alive += 1,
                MutantStatus::Equivalent => stats.equivalent += 1,
                MutantStatus::Timeout => stats.timeout += 1,
                MutantStatus::Error(_) => stats.error += 1,
            }
        }
        let denom = stats.total_mutants.saturating_sub(stats.equivalent);
        stats.mutation_score = if denom > 0 {
            stats.killed as f64 / denom as f64
        } else {
            1.0
        };
        stats
    }

    pub fn format_text(&self, colour: &Colour) -> String {
        let mut buf = String::new();
        buf.push_str(&colour.bold("Mutation Statistics\n"));
        buf.push_str(&format!("  Total mutants:    {}\n", self.total_mutants));
        buf.push_str(&format!(
            "  Killed:           {}\n",
            colour.green(&self.killed.to_string())
        ));
        buf.push_str(&format!(
            "  Alive:            {}\n",
            colour.red(&self.alive.to_string())
        ));
        buf.push_str(&format!("  Equivalent:       {}\n", self.equivalent));
        buf.push_str(&format!("  Timeout:          {}\n", self.timeout));
        buf.push_str(&format!("  Error:            {}\n", self.error));
        buf.push_str(&format!(
            "  Mutation score:   {}\n",
            colour.bold(&format!("{:.1}%", self.mutation_score * 100.0))
        ));
        if !self.operators.is_empty() {
            buf.push_str("\n  By operator:\n");
            for (op, count) in &self.operators {
                buf.push_str(&format!("    {:<6} {}\n", op, count));
            }
        }
        buf
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SynthesisStats {
    pub functions_analyzed: usize,
    pub contracts_synthesized: usize,
    pub total_clauses: usize,
    pub tier: String,
    pub elapsed_secs: f64,
}

impl SynthesisStats {
    pub fn format_text(&self, colour: &Colour) -> String {
        let mut buf = String::new();
        buf.push_str(&colour.bold("Synthesis Statistics\n"));
        buf.push_str(&format!(
            "  Functions analyzed:     {}\n",
            self.functions_analyzed
        ));
        buf.push_str(&format!(
            "  Contracts synthesized:  {}\n",
            self.contracts_synthesized
        ));
        buf.push_str(&format!(
            "  Total clauses:          {}\n",
            self.total_clauses
        ));
        buf.push_str(&format!("  Synthesis tier:         {}\n", self.tier));
        buf.push_str(&format!(
            "  Elapsed:                {:.2}s\n",
            self.elapsed_secs
        ));
        buf
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VerificationStats {
    pub total_obligations: usize,
    pub verified: usize,
    pub failed: usize,
    pub unknown: usize,
    pub elapsed_secs: f64,
}

impl VerificationStats {
    pub fn format_text(&self, colour: &Colour) -> String {
        let mut buf = String::new();
        buf.push_str(&colour.bold("Verification Results\n"));
        buf.push_str(&format!("  Obligations:  {}\n", self.total_obligations));
        buf.push_str(&format!(
            "  Verified:     {}\n",
            colour.green(&self.verified.to_string())
        ));
        buf.push_str(&format!(
            "  Failed:       {}\n",
            colour.red(&self.failed.to_string())
        ));
        buf.push_str(&format!("  Unknown:      {}\n", self.unknown));
        buf.push_str(&format!("  Elapsed:      {:.2}s\n", self.elapsed_secs));
        buf
    }
}

pub fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_table_display() {
        let mut tbl = AlignedTable::new(vec!["Name".into(), "Value".into()]);
        tbl.add_row(vec!["foo".into(), "42".into()]);
        tbl.add_row(vec!["barbaz".into(), "1".into()]);
        let s = format!("{tbl}");
        assert!(s.contains("Name"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_markdown_table() {
        let mut tbl = MarkdownTable::new(vec!["A".into(), "B".into()]);
        tbl.add_row(vec!["1".into(), "2".into()]);
        let s = format!("{tbl}");
        assert!(s.contains("| A |"));
    }

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[32mhello\x1b[0m"), "hello");
    }

    #[test]
    fn test_colour_disabled() {
        let c = Colour::new(true);
        assert_eq!(c.green("hi"), "hi");
        assert_eq!(c.bold("hi"), "hi");
    }

    #[test]
    fn test_format_duration_fn() {
        assert_eq!(format_duration(Duration::from_secs(5)), "5s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m5s");
    }

    #[test]
    fn test_sarif_builder() {
        let mut sb = SarifBuilder::new("test-tool", "0.1.0");
        sb.add_result("R001", "warning", "test msg", "file.ms", 10, 5);
        assert_eq!(sb.result_count(), 1);
        let doc = sb.build();
        assert_eq!(doc["version"], "2.1.0");
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(format!("{}", CliOutputFormat::Json), "json");
    }

    #[test]
    fn test_progress_bar_no_panic() {
        let mut pb = ProgressBar::new(10, "test");
        pb.disable();
        pb.inc();
        pb.finish();
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world foo bar", 8), "hello...");
    }
}
