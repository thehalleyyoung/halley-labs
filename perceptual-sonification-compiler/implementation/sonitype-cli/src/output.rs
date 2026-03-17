//! Output formatting: terminal tables, WAV writing, JSON reports, and
//! Markdown/text report generation.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::io::Write;
use std::path::Path;

// ── Table Formatter ─────────────────────────────────────────────────────────

/// Minimalist terminal table formatter (no external crate dependency).
#[derive(Debug, Clone)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    alignments: Vec<Alignment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Right,
    Center,
}

impl Table {
    pub fn new(headers: Vec<impl Into<String>>) -> Self {
        let headers: Vec<String> = headers.into_iter().map(Into::into).collect();
        let alignments = vec![Alignment::Left; headers.len()];
        Self {
            headers,
            rows: Vec::new(),
            alignments,
        }
    }

    pub fn set_alignment(&mut self, col: usize, align: Alignment) {
        if col < self.alignments.len() {
            self.alignments[col] = align;
        }
    }

    pub fn add_row(&mut self, row: Vec<impl Into<String>>) {
        self.rows.push(row.into_iter().map(Into::into).collect());
    }

    /// Render the table as a formatted string.
    pub fn render(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        // Compute column widths.
        let ncols = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < ncols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let mut out = String::new();

        // Header.
        out.push_str(&self.format_row(&self.headers, &widths));
        // Separator.
        let sep: Vec<String> = widths.iter().map(|&w| "─".repeat(w)).collect();
        out.push_str(&self.format_row(&sep, &widths));
        // Rows.
        for row in &self.rows {
            out.push_str(&self.format_row(row, &widths));
        }

        out
    }

    fn format_row(&self, cells: &[String], widths: &[usize]) -> String {
        let mut parts = Vec::new();
        for (i, w) in widths.iter().enumerate() {
            let cell = cells.get(i).map(|s| s.as_str()).unwrap_or("");
            let align = self.alignments.get(i).copied().unwrap_or(Alignment::Left);
            let formatted = match align {
                Alignment::Left => format!("{:<width$}", cell, width = w),
                Alignment::Right => format!("{:>width$}", cell, width = w),
                Alignment::Center => {
                    let pad = w.saturating_sub(cell.len());
                    let left = pad / 2;
                    let right = pad - left;
                    format!("{}{}{}", " ".repeat(left), cell, " ".repeat(right))
                }
            };
            parts.push(formatted);
        }
        format!("  {}  \n", parts.join("  │  "))
    }
}

// ── OutputFormatter ─────────────────────────────────────────────────────────

/// High-level formatter for compilation / lint / info output.
pub struct OutputFormatter;

impl OutputFormatter {
    /// Format a compilation result summary for terminal display.
    pub fn compilation_summary(result: &CompilationSummary) -> String {
        let mut out = String::new();
        out.push_str("╭─ Compilation Summary ─────────────────────────────╮\n");
        out.push_str(&format!("│  Status:   {:>38} │\n", if result.success { "✓ success" } else { "✗ failed" }));
        out.push_str(&format!("│  Streams:  {:>38} │\n", result.stream_count));
        out.push_str(&format!("│  Nodes:    {:>38} │\n", result.node_count));
        out.push_str(&format!("│  WCET:     {:>35.2} ms │\n", result.wcet_ms));
        out.push_str(&format!(
            "│  Time:     {:>35.2} ms │\n",
            result.total_time_ms
        ));
        out.push_str("╰───────────────────────────────────────────────────╯\n");
        out
    }

    /// Format a type-check report.
    pub fn type_check_report(report: &TypeCheckReport) -> String {
        let mut out = String::new();
        out.push_str("Type Check Report\n");
        out.push_str(&"═".repeat(40));
        out.push('\n');
        out.push_str(&format!("  Errors:   {}\n", report.errors));
        out.push_str(&format!("  Warnings: {}\n", report.warnings));
        out.push_str(&format!(
            "  Constraints satisfied: {}/{}\n",
            report.constraints_satisfied, report.constraints_total
        ));
        if !report.constraint_details.is_empty() {
            out.push('\n');
            let mut table = Table::new(vec!["Constraint", "Status", "Detail"]);
            for c in &report.constraint_details {
                table.add_row(vec![
                    c.name.clone(),
                    if c.satisfied { "✓" } else { "✗" }.to_string(),
                    c.detail.clone(),
                ]);
            }
            out.push_str(&table.render());
        }
        out
    }

    /// Format lint results.
    pub fn lint_report(results: &[LintResult]) -> String {
        if results.is_empty() {
            return "No lint issues found. ✓\n".to_string();
        }
        let mut table = Table::new(vec!["Severity", "Code", "Message", "Location"]);
        for r in results {
            table.add_row(vec![
                r.severity.clone(),
                r.code.clone(),
                r.message.clone(),
                r.location.clone(),
            ]);
        }
        format!("Lint Results ({} issue{})\n{}", results.len(), if results.len() == 1 { "" } else { "s" }, table.render())
    }

    /// Format info summary.
    pub fn info_summary(info: &InfoSummary) -> String {
        let mut out = String::new();
        out.push_str(&format!("File: {}\n", info.file));
        out.push_str(&"─".repeat(40));
        out.push('\n');
        out.push_str(&format!("  Streams:          {}\n", info.stream_count));
        out.push_str(&format!("  Mappings:         {}\n", info.mapping_count));
        out.push_str(&format!(
            "  Cognitive load:   {:.1} / {}\n",
            info.estimated_cognitive_load, info.cognitive_budget
        ));
        out.push_str(&format!(
            "  WCET estimate:    {:.2} ms\n",
            info.wcet_estimate_ms
        ));
        if !info.spectral_layout.is_empty() {
            out.push_str("\n  Spectral Layout:\n");
            for (name, band) in &info.spectral_layout {
                out.push_str(&format!("    {}: Bark band {}\n", name, band));
            }
        }
        out
    }
}

// ── Data types for formatted output ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct CompilationSummary {
    pub success: bool,
    pub stream_count: usize,
    pub node_count: usize,
    pub wcet_ms: f64,
    pub total_time_ms: f64,
    pub errors: usize,
    pub warnings: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeCheckReport {
    pub errors: usize,
    pub warnings: usize,
    pub constraints_satisfied: usize,
    pub constraints_total: usize,
    pub constraint_details: Vec<ConstraintDetail>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintDetail {
    pub name: String,
    pub satisfied: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LintResult {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub location: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InfoSummary {
    pub file: String,
    pub stream_count: usize,
    pub mapping_count: usize,
    pub estimated_cognitive_load: f64,
    pub cognitive_budget: u8,
    pub wcet_estimate_ms: f64,
    pub spectral_layout: Vec<(String, u8)>,
}

// ── WAV Output ──────────────────────────────────────────────────────────────

/// Write PCM audio data as a WAV file.
pub struct WavOutput;

impl WavOutput {
    /// Write interleaved PCM samples as a WAV file.
    ///
    /// `samples` contains interleaved channel data. `channels` is the number
    /// of output channels. `sample_rate` is in Hz. `bit_depth` is 16, 24, or
    /// 32.
    pub fn write(
        path: &Path,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        bit_depth: u16,
    ) -> Result<()> {
        if ![16, 24, 32].contains(&bit_depth) {
            bail!("Unsupported bit depth: {}", bit_depth);
        }

        let bytes_per_sample = (bit_depth / 8) as u32;
        let num_samples = samples.len() as u32;
        let data_size = num_samples * bytes_per_sample;
        let block_align = channels as u32 * bytes_per_sample;

        let mut file = std::fs::File::create(path)
            .with_context(|| format!("creating WAV file {}", path.display()))?;

        // RIFF header.
        file.write_all(b"RIFF")?;
        file.write_all(&(36 + data_size).to_le_bytes())?;
        file.write_all(b"WAVE")?;

        // fmt sub-chunk.
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // sub-chunk size
        file.write_all(&1u16.to_le_bytes())?; // PCM format
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&(sample_rate * block_align).to_le_bytes())?; // byte rate
        file.write_all(&(block_align as u16).to_le_bytes())?; // block align
        file.write_all(&bit_depth.to_le_bytes())?;

        // data sub-chunk.
        file.write_all(b"data")?;
        file.write_all(&data_size.to_le_bytes())?;

        // Write sample data.
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            match bit_depth {
                16 => {
                    let v = (clamped * i16::MAX as f32) as i16;
                    file.write_all(&v.to_le_bytes())?;
                }
                24 => {
                    let v = (clamped * 8_388_607.0) as i32;
                    let bytes = v.to_le_bytes();
                    file.write_all(&bytes[..3])?;
                }
                32 => {
                    let v = (clamped * i32::MAX as f32) as i32;
                    file.write_all(&v.to_le_bytes())?;
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Write WAV with metadata chunks (LIST/INFO).
    pub fn write_with_metadata(
        path: &Path,
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
        bit_depth: u16,
        metadata: &WavMetadata,
    ) -> Result<()> {
        // Write base WAV first, then append INFO chunk.
        Self::write(path, samples, channels, sample_rate, bit_depth)?;

        // Build LIST/INFO chunk.
        let info_chunk = metadata.to_info_chunk();
        if info_chunk.is_empty() {
            return Ok(());
        }

        // Re-open and patch: append INFO and update RIFF size.
        let mut data = std::fs::read(path)?;
        let info_bytes = info_chunk;
        data.extend_from_slice(&info_bytes);

        // Update RIFF size (bytes 4..8).
        let riff_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_size.to_le_bytes());

        std::fs::write(path, &data)?;
        Ok(())
    }

    /// Compute the peak amplitude in a sample buffer.
    pub fn peak_level(samples: &[f32]) -> f32 {
        samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }

    /// Compute RMS level.
    pub fn rms_level(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / samples.len() as f64).sqrt() as f32
    }
}

/// WAV metadata fields.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WavMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub comment: Option<String>,
    pub software: Option<String>,
}

impl WavMetadata {
    fn to_info_chunk(&self) -> Vec<u8> {
        let mut entries: Vec<(&[u8; 4], &str)> = Vec::new();
        if let Some(ref v) = self.title {
            entries.push((b"INAM", v));
        }
        if let Some(ref v) = self.artist {
            entries.push((b"IART", v));
        }
        if let Some(ref v) = self.comment {
            entries.push((b"ICMT", v));
        }
        if let Some(ref v) = self.software {
            entries.push((b"ISFT", v));
        }
        if entries.is_empty() {
            return Vec::new();
        }

        let mut info_data: Vec<u8> = Vec::new();
        info_data.extend_from_slice(b"INFO");
        for (tag, val) in &entries {
            let bytes = val.as_bytes();
            let size = bytes.len() as u32 + 1; // include null terminator
            info_data.extend_from_slice(*tag);
            info_data.extend_from_slice(&size.to_le_bytes());
            info_data.extend_from_slice(bytes);
            info_data.push(0); // null terminator
            if size % 2 != 0 {
                info_data.push(0); // pad to even
            }
        }

        let mut chunk = Vec::new();
        chunk.extend_from_slice(b"LIST");
        chunk.extend_from_slice(&(info_data.len() as u32).to_le_bytes());
        chunk.extend_from_slice(&info_data);
        chunk
    }
}

// ── JSON Output ─────────────────────────────────────────────────────────────

/// Structured JSON output for programmatic consumption.
pub struct JsonOutput;

impl JsonOutput {
    /// Serialize any `Serialize` value to pretty JSON string.
    pub fn format<T: Serialize>(value: &T) -> Result<String> {
        serde_json::to_string_pretty(value).context("JSON serialisation")
    }

    /// Write JSON to a file.
    pub fn write_to_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
        let json = Self::format(value)?;
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
    }
}

// ── Report Generator ────────────────────────────────────────────────────────

/// Format for generated reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    PlainText,
    Markdown,
    Json,
}

/// Generates human-readable compilation reports.
pub struct ReportGenerator;

impl ReportGenerator {
    /// Generate a full compilation report.
    pub fn generate(
        summary: &CompilationSummary,
        type_report: &TypeCheckReport,
        lint_results: &[LintResult],
        timings: &[(String, f64)],
        format: ReportFormat,
    ) -> String {
        match format {
            ReportFormat::PlainText => Self::plain(summary, type_report, lint_results, timings),
            ReportFormat::Markdown => Self::markdown(summary, type_report, lint_results, timings),
            ReportFormat::Json => Self::json(summary, type_report, lint_results, timings),
        }
    }

    fn plain(
        summary: &CompilationSummary,
        type_report: &TypeCheckReport,
        lint_results: &[LintResult],
        timings: &[(String, f64)],
    ) -> String {
        let mut out = String::new();
        out.push_str("SoniType Compilation Report\n");
        out.push_str(&"=".repeat(40));
        out.push('\n');
        out.push_str(&OutputFormatter::compilation_summary(summary));
        out.push('\n');
        out.push_str(&OutputFormatter::type_check_report(type_report));
        out.push('\n');
        out.push_str(&OutputFormatter::lint_report(lint_results));
        out.push('\n');
        out.push_str("Phase Timings\n");
        for (name, ms) in timings {
            out.push_str(&format!("  {:<25} {:.2} ms\n", name, ms * 1000.0));
        }
        out
    }

    fn markdown(
        summary: &CompilationSummary,
        type_report: &TypeCheckReport,
        lint_results: &[LintResult],
        timings: &[(String, f64)],
    ) -> String {
        let mut out = String::new();
        out.push_str("# SoniType Compilation Report\n\n");
        out.push_str("## Summary\n\n");
        out.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        out.push_str(&format!(
            "| Status | {} |\n",
            if summary.success { "✓" } else { "✗" }
        ));
        out.push_str(&format!("| Streams | {} |\n", summary.stream_count));
        out.push_str(&format!("| Nodes | {} |\n", summary.node_count));
        out.push_str(&format!("| WCET | {:.2} ms |\n", summary.wcet_ms));
        out.push_str(&format!("| Total time | {:.2} ms |\n\n", summary.total_time_ms));

        out.push_str("## Type Check\n\n");
        out.push_str(&format!(
            "- Errors: {}\n- Warnings: {}\n- Constraints: {}/{}\n\n",
            type_report.errors,
            type_report.warnings,
            type_report.constraints_satisfied,
            type_report.constraints_total,
        ));

        if !lint_results.is_empty() {
            out.push_str("## Lint Results\n\n");
            out.push_str("| Severity | Code | Message |\n|----------|------|---------|\n");
            for r in lint_results {
                out.push_str(&format!("| {} | {} | {} |\n", r.severity, r.code, r.message));
            }
            out.push('\n');
        }

        out.push_str("## Timings\n\n");
        out.push_str("| Phase | Duration |\n|-------|----------|\n");
        for (name, secs) in timings {
            out.push_str(&format!("| {} | {:.2} ms |\n", name, secs * 1000.0));
        }

        out
    }

    fn json(
        summary: &CompilationSummary,
        type_report: &TypeCheckReport,
        lint_results: &[LintResult],
        timings: &[(String, f64)],
    ) -> String {
        let report = serde_json::json!({
            "summary": summary,
            "type_check": type_report,
            "lint": lint_results,
            "timings": timings.iter().map(|(n, t)| {
                serde_json::json!({"phase": n, "duration_s": t})
            }).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&report).unwrap_or_default()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_summary() -> CompilationSummary {
        CompilationSummary {
            success: true,
            stream_count: 3,
            node_count: 12,
            wcet_ms: 2.5,
            total_time_ms: 120.0,
            errors: 0,
            warnings: 1,
        }
    }

    fn sample_type_report() -> TypeCheckReport {
        TypeCheckReport {
            errors: 0,
            warnings: 1,
            constraints_satisfied: 5,
            constraints_total: 5,
            constraint_details: vec![ConstraintDetail {
                name: "bark_separation".into(),
                satisfied: true,
                detail: "all streams ≥1 band apart".into(),
            }],
        }
    }

    #[test]
    fn table_render() {
        let mut t = Table::new(vec!["Name", "Value"]);
        t.add_row(vec!["alpha", "1"]);
        t.add_row(vec!["beta", "2"]);
        let out = t.render();
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
    }

    #[test]
    fn table_alignment_right() {
        let mut t = Table::new(vec!["Key", "Num"]);
        t.set_alignment(1, Alignment::Right);
        t.add_row(vec!["x", "42"]);
        let out = t.render();
        assert!(out.contains("42"));
    }

    #[test]
    fn table_empty_headers() {
        let t = Table::new(Vec::<String>::new());
        assert!(t.render().is_empty());
    }

    #[test]
    fn compilation_summary_format() {
        let s = sample_summary();
        let out = OutputFormatter::compilation_summary(&s);
        assert!(out.contains("success"));
        assert!(out.contains("12"));
    }

    #[test]
    fn type_check_report_format() {
        let r = sample_type_report();
        let out = OutputFormatter::type_check_report(&r);
        assert!(out.contains("5/5"));
    }

    #[test]
    fn lint_report_empty() {
        let out = OutputFormatter::lint_report(&[]);
        assert!(out.contains("No lint issues"));
    }

    #[test]
    fn lint_report_non_empty() {
        let results = vec![LintResult {
            severity: "warning".into(),
            code: "L001".into(),
            message: "masking detected".into(),
            location: "test.soni:5:3".into(),
        }];
        let out = OutputFormatter::lint_report(&results);
        assert!(out.contains("masking"));
    }

    #[test]
    fn info_summary_format() {
        let info = InfoSummary {
            file: "test.soni".into(),
            stream_count: 2,
            mapping_count: 4,
            estimated_cognitive_load: 2.5,
            cognitive_budget: 4,
            wcet_estimate_ms: 1.2,
            spectral_layout: vec![("temp".into(), 8), ("pressure".into(), 15)],
        };
        let out = OutputFormatter::info_summary(&info);
        assert!(out.contains("temp"));
        assert!(out.contains("Bark band 8"));
    }

    #[test]
    fn wav_peak_level() {
        let samples = vec![0.5f32, -0.8, 0.3, 0.0];
        assert!((WavOutput::peak_level(&samples) - 0.8).abs() < 0.001);
    }

    #[test]
    fn wav_rms_level() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        assert!((WavOutput::rms_level(&samples) - 1.0).abs() < 0.001);
    }

    #[test]
    fn wav_rms_empty() {
        assert_eq!(WavOutput::rms_level(&[]), 0.0);
    }

    #[test]
    fn wav_write_and_read_back() {
        let dir = std::env::temp_dir();
        let path = dir.join("sonitype_test_output.wav");
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0 * 440.0 * std::f32::consts::TAU).sin() * 0.5).collect();
        WavOutput::write(&path, &samples, 1, 44100, 16).unwrap();

        // Check file starts with RIFF/WAVE.
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wav_metadata_chunk() {
        let meta = WavMetadata {
            title: Some("Test".into()),
            artist: Some("SoniType".into()),
            comment: None,
            software: Some("sonitype-cli".into()),
        };
        let chunk = meta.to_info_chunk();
        assert!(!chunk.is_empty());
        assert_eq!(&chunk[..4], b"LIST");
    }

    #[test]
    fn json_output_round_trip() {
        let summary = sample_summary();
        let json = JsonOutput::format(&summary).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["stream_count"], 3);
    }

    #[test]
    fn report_plain() {
        let r = ReportGenerator::generate(
            &sample_summary(),
            &sample_type_report(),
            &[],
            &[("Parsing".into(), 0.01)],
            ReportFormat::PlainText,
        );
        assert!(r.contains("Compilation Report"));
    }

    #[test]
    fn report_markdown() {
        let r = ReportGenerator::generate(
            &sample_summary(),
            &sample_type_report(),
            &[],
            &[("Parsing".into(), 0.01)],
            ReportFormat::Markdown,
        );
        assert!(r.contains("# SoniType"));
    }

    #[test]
    fn report_json() {
        let r = ReportGenerator::generate(
            &sample_summary(),
            &sample_type_report(),
            &[],
            &[("Parsing".into(), 0.01)],
            ReportFormat::Json,
        );
        let v: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(v["summary"]["success"].as_bool().unwrap());
    }

    #[test]
    fn wav_write_invalid_bit_depth() {
        let dir = std::env::temp_dir();
        let path = dir.join("sonitype_test_bad_bd.wav");
        let result = WavOutput::write(&path, &[0.0], 1, 44100, 8);
        assert!(result.is_err());
    }
}
