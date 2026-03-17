//! Output formatting and color management for SafeStep CLI.

use std::collections::HashMap;
use std::io::Write;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::cli::OutputFormat;

// ---------------------------------------------------------------------------
// ColorScheme
// ---------------------------------------------------------------------------

/// Terminal color codes for safety-annotated output.
#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub enabled: bool,
}

impl ColorScheme {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn safe(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[32m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn warning(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[33m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn error(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[31m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn pnr(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[1;31m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn info(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[36m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn bold(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[1m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn dim(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[2m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn header(&self, text: &str) -> String {
        if self.enabled { format!("\x1b[1;34m{}\x1b[0m", text) } else { text.to_string() }
    }

    pub fn step_number(&self, n: usize) -> String {
        let s = format!("[Step {}]", n);
        if self.enabled { format!("\x1b[1;36m{}\x1b[0m", s) } else { s }
    }
}

// ---------------------------------------------------------------------------
// OutputManager
// ---------------------------------------------------------------------------

/// Manages output formatting and destination.
pub struct OutputManager {
    format: OutputFormat,
    colors: ColorScheme,
    output_path: Option<std::path::PathBuf>,
    buffer: Vec<String>,
}

impl OutputManager {
    pub fn new(format: OutputFormat, color_enabled: bool) -> Self {
        Self {
            format,
            colors: ColorScheme::new(color_enabled),
            output_path: None,
            buffer: Vec::new(),
        }
    }

    pub fn with_output_path(mut self, path: std::path::PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    pub fn colors(&self) -> &ColorScheme {
        &self.colors
    }

    pub fn format(&self) -> OutputFormat {
        self.format
    }

    /// Write a line to the output buffer.
    pub fn writeln(&mut self, line: &str) {
        self.buffer.push(line.to_string());
    }

    /// Write a blank line.
    pub fn blank_line(&mut self) {
        self.buffer.push(String::new());
    }

    /// Write a section header.
    pub fn section(&mut self, title: &str) {
        self.buffer.push(String::new());
        self.buffer.push(self.colors.header(title));
        self.buffer.push(self.colors.dim(&"─".repeat(title.len().max(40))));
    }

    /// Render a serializable value in the selected format.
    pub fn render_value<T: Serialize>(&mut self, value: &T) -> Result<()> {
        let text = match self.format {
            OutputFormat::Json => serde_json::to_string_pretty(value)
                .context("failed to serialize to JSON")?,
            OutputFormat::Yaml => serde_yaml::to_string(value)
                .context("failed to serialize to YAML")?,
            OutputFormat::Text => format!("{:#?}", serde_json::to_value(value)
                .context("failed to serialize")?),
            OutputFormat::Markdown => self.to_markdown(value)?,
            OutputFormat::Html => self.to_html(value)?,
        };
        self.buffer.push(text);
        Ok(())
    }

    /// Render a key-value table.
    pub fn render_table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        match self.format {
            OutputFormat::Text => self.render_text_table(headers, rows),
            OutputFormat::Markdown => self.render_markdown_table(headers, rows),
            OutputFormat::Html => self.render_html_table(headers, rows),
            OutputFormat::Json => {
                let items: Vec<HashMap<String, String>> = rows
                    .iter()
                    .map(|row| {
                        headers.iter().zip(row.iter())
                            .map(|(h, v)| (h.to_string(), v.clone()))
                            .collect()
                    })
                    .collect();
                if let Ok(json) = serde_json::to_string_pretty(&items) {
                    self.buffer.push(json);
                }
            }
            OutputFormat::Yaml => {
                let items: Vec<HashMap<String, String>> = rows
                    .iter()
                    .map(|row| {
                        headers.iter().zip(row.iter())
                            .map(|(h, v)| (h.to_string(), v.clone()))
                            .collect()
                    })
                    .collect();
                if let Ok(yaml) = serde_yaml::to_string(&items) {
                    self.buffer.push(yaml);
                }
            }
        }
    }

    fn render_text_table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        let col_count = headers.len();
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < col_count {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let header_line: String = headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<width$}", h, width = widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        self.buffer.push(self.colors.bold(&header_line));

        let sep: String = widths.iter().map(|w| "─".repeat(*w)).collect::<Vec<_>>().join("──");
        self.buffer.push(self.colors.dim(&sep));

        for row in rows {
            let line: String = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = widths.get(i).copied().unwrap_or(cell.len());
                    format!("{:<width$}", cell, width = w)
                })
                .collect::<Vec<_>>()
                .join("  ");
            self.buffer.push(line);
        }
    }

    fn render_markdown_table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        let header = format!("| {} |", headers.join(" | "));
        let sep = format!("| {} |", headers.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
        self.buffer.push(header);
        self.buffer.push(sep);
        for row in rows {
            self.buffer.push(format!("| {} |", row.join(" | ")));
        }
    }

    fn render_html_table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        self.buffer.push("<table>".to_string());
        self.buffer.push("<thead><tr>".to_string());
        for h in headers {
            self.buffer.push(format!("  <th>{}</th>", html_escape(h)));
        }
        self.buffer.push("</tr></thead>".to_string());
        self.buffer.push("<tbody>".to_string());
        for row in rows {
            self.buffer.push("<tr>".to_string());
            for cell in row {
                self.buffer.push(format!("  <td>{}</td>", html_escape(cell)));
            }
            self.buffer.push("</tr>".to_string());
        }
        self.buffer.push("</tbody></table>".to_string());
    }

    fn to_markdown<T: Serialize>(&self, value: &T) -> Result<String> {
        let json = serde_json::to_value(value).context("serialize")?;
        let mut out = String::new();
        render_json_as_markdown(&json, &mut out, 0);
        Ok(out)
    }

    fn to_html<T: Serialize>(&self, value: &T) -> Result<String> {
        let json = serde_json::to_string_pretty(value).context("serialize")?;
        Ok(format!(
            "<!DOCTYPE html>\n<html><body><pre><code>{}</code></pre></body></html>",
            html_escape(&json)
        ))
    }

    /// Flush all buffered output to the configured destination.
    pub fn flush(&mut self) -> Result<()> {
        let content = self.buffer.join("\n");
        if let Some(ref path) = self.output_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create output dir: {}", parent.display()))?;
            }
            std::fs::write(path, &content)
                .with_context(|| format!("failed to write output: {}", path.display()))?;
            tracing::info!(path = %path.display(), "output written to file");
        } else {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(content.as_bytes()).context("write stdout")?;
            handle.write_all(b"\n").ok();
            handle.flush().ok();
        }
        self.buffer.clear();
        Ok(())
    }

    /// Get buffered content as a single string (for testing).
    pub fn get_buffer(&self) -> String {
        self.buffer.join("\n")
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render_json_as_markdown(value: &serde_json::Value, out: &mut String, depth: usize) {
    let indent = "  ".repeat(depth);
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        out.push_str(&format!("{}**{}**:\n", indent, k));
                        render_json_as_markdown(v, out, depth + 1);
                    }
                    _ => {
                        out.push_str(&format!("{}- **{}**: {}\n", indent, k, format_value(v)));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                out.push_str(&format!("{}{}. ", indent, i + 1));
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        out.push('\n');
                        render_json_as_markdown(v, out, depth + 1);
                    }
                    _ => {
                        out.push_str(&format_value(v));
                        out.push('\n');
                    }
                }
            }
        }
        _ => {
            out.push_str(&format!("{}{}\n", indent, format_value(value)));
        }
    }
}

fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => format!("{}", v),
    }
}

/// Create an OutputManager from CLI options.
pub fn create_output_manager(
    format: OutputFormat,
    color_enabled: bool,
    output_path: Option<std::path::PathBuf>,
) -> OutputManager {
    let mut mgr = OutputManager::new(format, color_enabled);
    if let Some(path) = output_path {
        mgr = mgr.with_output_path(path);
    }
    mgr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_scheme_safe() {
        let cs = ColorScheme::new(true);
        assert!(cs.safe("ok").contains("\x1b[32m"));

        let cs_no = ColorScheme::new(false);
        assert_eq!(cs_no.safe("ok"), "ok");
    }

    #[test]
    fn test_color_scheme_error() {
        let cs = ColorScheme::new(true);
        assert!(cs.error("err").contains("\x1b[31m"));
    }

    #[test]
    fn test_color_scheme_pnr() {
        let cs = ColorScheme::new(true);
        let pnr = cs.pnr("point of no return");
        assert!(pnr.contains("\x1b[1;31m"));
    }

    #[test]
    fn test_color_scheme_disabled() {
        let cs = ColorScheme::new(false);
        assert_eq!(cs.safe("x"), "x");
        assert_eq!(cs.warning("x"), "x");
        assert_eq!(cs.error("x"), "x");
        assert_eq!(cs.pnr("x"), "x");
        assert_eq!(cs.info("x"), "x");
        assert_eq!(cs.bold("x"), "x");
        assert_eq!(cs.dim("x"), "x");
    }

    #[test]
    fn test_output_manager_writeln() {
        let mut mgr = OutputManager::new(OutputFormat::Text, false);
        mgr.writeln("hello");
        mgr.writeln("world");
        assert_eq!(mgr.get_buffer(), "hello\nworld");
    }

    #[test]
    fn test_output_manager_section() {
        let mut mgr = OutputManager::new(OutputFormat::Text, false);
        mgr.section("Test Section");
        let buf = mgr.get_buffer();
        assert!(buf.contains("Test Section"));
    }

    #[test]
    fn test_output_manager_render_json() {
        let mut mgr = OutputManager::new(OutputFormat::Json, false);
        let data = serde_json::json!({"key": "value", "num": 42});
        mgr.render_value(&data).unwrap();
        let buf = mgr.get_buffer();
        assert!(buf.contains("\"key\""));
        assert!(buf.contains("42"));
    }

    #[test]
    fn test_output_manager_render_yaml() {
        let mut mgr = OutputManager::new(OutputFormat::Yaml, false);
        let data = serde_json::json!({"key": "value"});
        mgr.render_value(&data).unwrap();
        let buf = mgr.get_buffer();
        assert!(buf.contains("key:"));
    }

    #[test]
    fn test_output_manager_text_table() {
        let mut mgr = OutputManager::new(OutputFormat::Text, false);
        mgr.render_table(
            &["Name", "Status"],
            &[
                vec!["svc-a".into(), "safe".into()],
                vec!["svc-b".into(), "pnr".into()],
            ],
        );
        let buf = mgr.get_buffer();
        assert!(buf.contains("Name"));
        assert!(buf.contains("svc-a"));
        assert!(buf.contains("svc-b"));
    }

    #[test]
    fn test_output_manager_markdown_table() {
        let mut mgr = OutputManager::new(OutputFormat::Markdown, false);
        mgr.render_table(
            &["A", "B"],
            &[vec!["1".into(), "2".into()]],
        );
        let buf = mgr.get_buffer();
        assert!(buf.contains("| A | B |"));
        assert!(buf.contains("| 1 | 2 |"));
    }

    #[test]
    fn test_output_manager_html_table() {
        let mut mgr = OutputManager::new(OutputFormat::Html, false);
        mgr.render_table(
            &["X"],
            &[vec!["<b>bold</b>".into()]],
        );
        let buf = mgr.get_buffer();
        assert!(buf.contains("<table>"));
        assert!(buf.contains("&lt;b&gt;"));
    }

    #[test]
    fn test_output_manager_flush_to_file() {
        let dir = std::env::temp_dir().join("safestep_output_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_output.txt");

        let mut mgr = OutputManager::new(OutputFormat::Text, false)
            .with_output_path(path.clone());
        mgr.writeln("line1");
        mgr.writeln("line2");
        mgr.flush().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("line1"));
        assert!(content.contains("line2"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<b>&\""), "&lt;b&gt;&amp;&quot;");
    }

    #[test]
    fn test_create_output_manager() {
        let mgr = create_output_manager(OutputFormat::Json, true, None);
        assert_eq!(mgr.format(), OutputFormat::Json);
        assert!(mgr.colors().enabled);
    }

    #[test]
    fn test_render_markdown_value() {
        let mut mgr = OutputManager::new(OutputFormat::Markdown, false);
        let data = serde_json::json!({"name": "test", "count": 3});
        mgr.render_value(&data).unwrap();
        let buf = mgr.get_buffer();
        assert!(buf.contains("**name**"));
    }

    #[test]
    fn test_render_html_value() {
        let mut mgr = OutputManager::new(OutputFormat::Html, false);
        let data = serde_json::json!({"a": 1});
        mgr.render_value(&data).unwrap();
        let buf = mgr.get_buffer();
        assert!(buf.contains("<html>"));
    }

    #[test]
    fn test_blank_line() {
        let mut mgr = OutputManager::new(OutputFormat::Text, false);
        mgr.writeln("a");
        mgr.blank_line();
        mgr.writeln("b");
        assert_eq!(mgr.get_buffer(), "a\n\nb");
    }

    #[test]
    fn test_step_number() {
        let cs = ColorScheme::new(false);
        assert_eq!(cs.step_number(3), "[Step 3]");
        let cs2 = ColorScheme::new(true);
        assert!(cs2.step_number(1).contains("Step 1"));
    }
}
