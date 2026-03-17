//! Output formatting for SafeStep diagnostic reports.
//!
//! Provides formatters for JSON, plain text, Markdown, and HTML output,
//! plus a table renderer supporting both box-drawing and simple ASCII styles.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ColumnAlignment
// ---------------------------------------------------------------------------

/// Alignment for a table column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnAlignment {
    Left,
    Center,
    Right,
}

impl Default for ColumnAlignment {
    fn default() -> Self {
        Self::Left
    }
}

// ---------------------------------------------------------------------------
// TableRenderer
// ---------------------------------------------------------------------------

/// Renders tabular data in either Unicode box-drawing or simple ASCII style.
#[derive(Debug, Clone)]
pub struct TableRenderer {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    alignments: HashMap<usize, ColumnAlignment>,
}

impl TableRenderer {
    /// Create an empty table with no headers.
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
            rows: Vec::new(),
            alignments: HashMap::new(),
        }
    }

    /// Create a table pre-populated with column headers.
    pub fn with_headers(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
            alignments: HashMap::new(),
        }
    }

    /// Append a data row. Returns `&mut Self` for chaining.
    pub fn add_row(&mut self, row: Vec<String>) -> &mut Self {
        self.rows.push(row);
        self
    }

    /// Set the alignment for a specific column (0-indexed).
    pub fn set_alignment(&mut self, col: usize, align: ColumnAlignment) -> &mut Self {
        self.alignments.insert(col, align);
        self
    }

    // -- helpers -----------------------------------------------------------

    fn num_cols(&self) -> usize {
        let from_headers = self.headers.len();
        let from_rows = self.rows.iter().map(|r| r.len()).max().unwrap_or(0);
        from_headers.max(from_rows)
    }

    fn col_widths(&self) -> Vec<usize> {
        let cols = self.num_cols();
        let mut widths = vec![0usize; cols];
        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = widths[i].max(h.len());
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < cols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }
        // Ensure at least width 1 so borders render correctly.
        for w in &mut widths {
            if *w == 0 {
                *w = 1;
            }
        }
        widths
    }

    fn alignment_for(&self, col: usize) -> ColumnAlignment {
        self.alignments.get(&col).copied().unwrap_or_default()
    }

    fn pad_cell(&self, text: &str, width: usize, align: ColumnAlignment) -> String {
        if text.len() >= width {
            return text.to_string();
        }
        let padding = width - text.len();
        match align {
            ColumnAlignment::Left => format!("{}{}", text, " ".repeat(padding)),
            ColumnAlignment::Right => format!("{}{}", " ".repeat(padding), text),
            ColumnAlignment::Center => {
                let left = padding / 2;
                let right = padding - left;
                format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
            }
        }
    }

    fn cell_at<'a>(row: &'a [String], col: usize) -> &'a str {
        row.get(col).map(|s| s.as_str()).unwrap_or("")
    }

    // -- public renderers --------------------------------------------------

    /// Render using Unicode box-drawing characters.
    pub fn render(&self) -> String {
        let widths = self.col_widths();
        let cols = widths.len();
        if cols == 0 {
            return String::new();
        }

        let mut out = String::new();

        // Top border:  ┌───┬───┐
        out.push('\u{250C}');
        for (i, &w) in widths.iter().enumerate() {
            for _ in 0..w + 2 {
                out.push('\u{2500}');
            }
            if i + 1 < cols {
                out.push('\u{252C}');
            }
        }
        out.push('\u{2510}');
        out.push('\n');

        // Header row
        if !self.headers.is_empty() {
            out.push('\u{2502}');
            for (i, &w) in widths.iter().enumerate() {
                let text = self.headers.get(i).map(|s| s.as_str()).unwrap_or("");
                let padded = self.pad_cell(text, w, self.alignment_for(i));
                let _ = write!(out, " {} ", padded);
                out.push('\u{2502}');
            }
            out.push('\n');

            // Separator:  ├───┼───┤
            out.push('\u{251C}');
            for (i, &w) in widths.iter().enumerate() {
                for _ in 0..w + 2 {
                    out.push('\u{2500}');
                }
                if i + 1 < cols {
                    out.push('\u{253C}');
                }
            }
            out.push('\u{2524}');
            out.push('\n');
        }

        // Data rows
        for row in &self.rows {
            out.push('\u{2502}');
            for (i, &w) in widths.iter().enumerate() {
                let text = Self::cell_at(row, i);
                let padded = self.pad_cell(text, w, self.alignment_for(i));
                let _ = write!(out, " {} ", padded);
                out.push('\u{2502}');
            }
            out.push('\n');
        }

        // Bottom border:  └───┴───┘
        out.push('\u{2514}');
        for (i, &w) in widths.iter().enumerate() {
            for _ in 0..w + 2 {
                out.push('\u{2500}');
            }
            if i + 1 < cols {
                out.push('\u{2534}');
            }
        }
        out.push('\u{2518}');
        out.push('\n');

        out
    }

    /// Render using simple pipe and dash characters.
    pub fn render_simple(&self) -> String {
        let widths = self.col_widths();
        let cols = widths.len();
        if cols == 0 {
            return String::new();
        }

        let mut out = String::new();

        let separator = Self::simple_separator(&widths);

        out.push_str(&separator);
        out.push('\n');

        if !self.headers.is_empty() {
            out.push('|');
            for (i, &w) in widths.iter().enumerate() {
                let text = self.headers.get(i).map(|s| s.as_str()).unwrap_or("");
                let padded = self.pad_cell(text, w, self.alignment_for(i));
                let _ = write!(out, " {} |", padded);
            }
            out.push('\n');
            out.push_str(&separator);
            out.push('\n');
        }

        for row in &self.rows {
            out.push('|');
            for (i, &w) in widths.iter().enumerate() {
                let text = Self::cell_at(row, i);
                let padded = self.pad_cell(text, w, self.alignment_for(i));
                let _ = write!(out, " {} |", padded);
            }
            out.push('\n');
        }

        out.push_str(&separator);
        out.push('\n');

        out
    }

    fn simple_separator(widths: &[usize]) -> String {
        let mut sep = String::new();
        sep.push('+');
        for &w in widths {
            for _ in 0..w + 2 {
                sep.push('-');
            }
            sep.push('+');
        }
        sep
    }
}

impl Default for TableRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JsonFormatter
// ---------------------------------------------------------------------------

/// Formats `serde_json::Value` data as JSON strings.
pub struct JsonFormatter;

impl JsonFormatter {
    /// Pretty-print a JSON value with 2-space indentation.
    pub fn format(value: &serde_json::Value) -> String {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    }

    /// Compact single-line JSON.
    pub fn format_compact(value: &serde_json::Value) -> String {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    }

    /// Pretty-printed JSON with `//`-style comments injected before matching
    /// top-level keys. `comments` maps a JSON key name to the comment text.
    pub fn format_with_comments(
        value: &serde_json::Value,
        comments: &HashMap<String, String>,
    ) -> String {
        let pretty = Self::format(value);
        if comments.is_empty() {
            return pretty;
        }
        let mut out = String::with_capacity(pretty.len() + comments.len() * 40);
        for line in pretty.lines() {
            // Detect lines like `  "key":` and inject the comment above.
            let trimmed = line.trim_start();
            if trimmed.starts_with('"') {
                if let Some(key_end) = trimmed[1..].find('"') {
                    let key = &trimmed[1..1 + key_end];
                    if let Some(comment) = comments.get(key) {
                        let indent: String =
                            line.chars().take_while(|c| c.is_whitespace()).collect();
                        let _ = writeln!(out, "{}// {}", indent, comment);
                    }
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        // Remove trailing newline to match `format` behaviour.
        if out.ends_with('\n') {
            out.pop();
        }
        out
    }
}

// ---------------------------------------------------------------------------
// TextFormatter
// ---------------------------------------------------------------------------

/// Produces plain-text diagnostic output.
pub struct TextFormatter;

impl TextFormatter {
    /// Format a sequence of `(heading, body)` sections.
    /// Each heading is underlined with `=` characters.
    pub fn format_sections(sections: &[(String, String)]) -> String {
        let mut out = String::new();
        for (i, (heading, body)) in sections.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(heading);
            out.push('\n');
            for _ in 0..heading.len() {
                out.push('=');
            }
            out.push('\n');
            out.push_str(body);
            out.push('\n');
        }
        out
    }

    /// Render key-value pairs with aligned colons.
    pub fn format_key_value(pairs: &[(String, String)]) -> String {
        if pairs.is_empty() {
            return String::new();
        }
        let max_key = pairs.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let mut out = String::new();
        for (key, val) in pairs {
            let padding = max_key - key.len();
            let _ = writeln!(out, "{}{}  : {}", key, " ".repeat(padding), val);
        }
        out
    }

    /// Draw a simple ASCII bar chart line.
    ///
    /// Example output: `CPU  [########------] 57%`
    pub fn format_bar(label: &str, value: f64, max: f64, width: usize) -> String {
        let effective_max = if max <= 0.0 { 1.0 } else { max };
        let ratio = (value / effective_max).clamp(0.0, 1.0);
        let filled = (ratio * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        let pct = (ratio * 100.0).round() as u64;
        format!(
            "{:<6} [{}{}] {}%",
            label,
            "#".repeat(filled),
            "-".repeat(empty),
            pct,
        )
    }

    /// Render a Unicode sparkline from a slice of values.
    ///
    /// Uses the block characters ▁▂▃▄▅▆▇█ to represent relative magnitudes.
    pub fn format_sparkline(values: &[f64]) -> String {
        if values.is_empty() {
            return String::new();
        }
        let blocks = [
            '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
            '\u{2588}',
        ];
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        let mut out = String::with_capacity(values.len() * 4);
        for &v in values {
            let idx = if range == 0.0 {
                3 // middle
            } else {
                let normalised = (v - min) / range;
                ((normalised * 7.0).round() as usize).min(7)
            };
            out.push(blocks[idx]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MarkdownFormatter
// ---------------------------------------------------------------------------

/// Produces Markdown-formatted output.
pub struct MarkdownFormatter;

impl MarkdownFormatter {
    /// Render `(heading, body)` sections using `##` headings.
    pub fn format_sections(sections: &[(String, String)]) -> String {
        let mut out = String::new();
        for (i, (heading, body)) in sections.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "## {}", heading);
            out.push('\n');
            out.push_str(body);
            out.push('\n');
        }
        out
    }

    /// Render a Markdown table from headers and rows.
    pub fn format_table(headers: &[String], rows: &[Vec<String>]) -> String {
        if headers.is_empty() {
            return String::new();
        }
        let num_cols = headers.len();

        // Compute column widths for pretty alignment.
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len().max(3)).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < num_cols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let mut out = String::new();

        // Header row
        out.push('|');
        for (i, h) in headers.iter().enumerate() {
            let w = widths.get(i).copied().unwrap_or(3);
            let _ = write!(out, " {:<w$} |", h, w = w);
        }
        out.push('\n');

        // Separator
        out.push('|');
        for &w in &widths {
            let _ = write!(out, " {} |", "-".repeat(w));
        }
        out.push('\n');

        // Data rows
        for row in rows {
            out.push('|');
            for i in 0..num_cols {
                let w = widths.get(i).copied().unwrap_or(3);
                let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                let _ = write!(out, " {:<w$} |", cell, w = w);
            }
            out.push('\n');
        }
        out
    }

    /// Render a checklist. Each item is `(text, checked)`.
    pub fn format_checklist(items: &[(String, bool)]) -> String {
        let mut out = String::new();
        for (text, checked) in items {
            if *checked {
                let _ = writeln!(out, "- [x] {}", text);
            } else {
                let _ = writeln!(out, "- [ ] {}", text);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// HtmlFormatter
// ---------------------------------------------------------------------------

/// Produces HTML output with embedded CSS.
pub struct HtmlFormatter;

impl HtmlFormatter {
    /// Render a full HTML page containing the given sections.
    pub fn format_page(sections: &[(String, String)], title: &str) -> String {
        let mut body = String::new();
        for (heading, content) in sections {
            let _ = write!(
                body,
                "    <section>\n      <h2>{}</h2>\n      <p>{}</p>\n    </section>\n",
                Self::escape_html(heading),
                Self::escape_html(content),
            );
        }

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #1e1e2e;
      --fg: #cdd6f4;
      --accent: #89b4fa;
      --border: #45475a;
      --surface: #313244;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--fg);
      padding: 2rem;
      line-height: 1.6;
    }}
    h1 {{ color: var(--accent); margin-bottom: 1.5rem; }}
    h2 {{ color: var(--accent); margin-bottom: 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }}
    section {{ background: var(--surface); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }}
    p {{ white-space: pre-wrap; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
    th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: var(--surface); color: var(--accent); }}
    details {{ margin-bottom: 0.75rem; }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
  </style>
</head>
<body>
  <h1>{title}</h1>
{body}</body>
</html>
"#,
            title = Self::escape_html(title),
            body = body,
        )
    }

    /// Render an HTML `<table>` from headers and rows.
    pub fn format_table(headers: &[String], rows: &[Vec<String>]) -> String {
        let mut out = String::from("<table>\n  <thead>\n    <tr>\n");
        for h in headers {
            let _ = writeln!(out, "      <th>{}</th>", Self::escape_html(h));
        }
        out.push_str("    </tr>\n  </thead>\n  <tbody>\n");
        for row in rows {
            out.push_str("    <tr>\n");
            for cell in row {
                let _ = writeln!(out, "      <td>{}</td>", Self::escape_html(cell));
            }
            out.push_str("    </tr>\n");
        }
        out.push_str("  </tbody>\n</table>\n");
        out
    }

    /// Render a collapsible `<details>/<summary>` block.
    pub fn format_collapsible(title: &str, content: &str) -> String {
        format!(
            "<details>\n  <summary>{}</summary>\n  <div>{}</div>\n</details>\n",
            Self::escape_html(title),
            Self::escape_html(content),
        )
    }

    // -- helpers -----------------------------------------------------------

    fn escape_html(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '"' => out.push_str("&quot;"),
                '\'' => out.push_str("&#39;"),
                _ => out.push(c),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// OutputFormatter  (façade)
// ---------------------------------------------------------------------------

/// Convenience façade that delegates to the format-specific formatters.
pub struct OutputFormatter;

impl OutputFormatter {
    /// Pretty-print a JSON value.
    pub fn format_json(value: &serde_json::Value) -> String {
        JsonFormatter::format(value)
    }

    /// Format `(heading, body)` sections as plain text.
    pub fn format_text(sections: &[(String, String)]) -> String {
        TextFormatter::format_sections(sections)
    }

    /// Format `(heading, body)` sections as Markdown.
    pub fn format_markdown(sections: &[(String, String)]) -> String {
        MarkdownFormatter::format_sections(sections)
    }

    /// Format `(heading, body)` sections as a complete HTML page.
    pub fn format_html(sections: &[(String, String)], title: &str) -> String {
        HtmlFormatter::format_page(sections, title)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ColumnAlignment ---------------------------------------------------

    #[test]
    fn column_alignment_default_is_left() {
        assert_eq!(ColumnAlignment::default(), ColumnAlignment::Left);
    }

    // -- TableRenderer: Unicode -------------------------------------------

    #[test]
    fn table_render_empty() {
        let table = TableRenderer::new();
        assert_eq!(table.render(), "");
    }

    #[test]
    fn table_render_headers_only() {
        let table = TableRenderer::with_headers(vec!["A".into(), "B".into()]);
        let out = table.render();
        assert!(out.contains('\u{250C}')); // ┌
        assert!(out.contains('\u{2510}')); // ┐
        assert!(out.contains('\u{2514}')); // └
        assert!(out.contains('\u{2518}')); // ┘
        assert!(out.contains('\u{2502}')); // │
        assert!(out.contains("A"));
        assert!(out.contains("B"));
    }

    #[test]
    fn table_render_with_data() {
        let mut table = TableRenderer::with_headers(vec!["Name".into(), "Value".into()]);
        table.add_row(vec!["alpha".into(), "1".into()]);
        table.add_row(vec!["beta".into(), "2".into()]);
        let out = table.render();
        assert!(out.contains("Name"));
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        // Should have separator between header and rows.
        assert!(out.contains('\u{251C}')); // ├
        assert!(out.contains('\u{253C}')); // ┼
        assert!(out.contains('\u{2524}')); // ┤
    }

    #[test]
    fn table_render_alignment() {
        let mut table = TableRenderer::with_headers(vec!["Left".into(), "Center".into(), "Right".into()]);
        table.set_alignment(0, ColumnAlignment::Left);
        table.set_alignment(1, ColumnAlignment::Center);
        table.set_alignment(2, ColumnAlignment::Right);
        table.add_row(vec!["a".into(), "b".into(), "c".into()]);
        let out = table.render();
        // Find the data row and verify padding direction.
        let data_lines: Vec<&str> = out.lines().filter(|l| l.contains("a")).collect();
        assert!(!data_lines.is_empty());
    }

    #[test]
    fn table_render_no_headers() {
        let mut table = TableRenderer::new();
        table.add_row(vec!["x".into(), "y".into()]);
        let out = table.render();
        assert!(out.contains("x"));
        assert!(out.contains("y"));
        // Should NOT contain the header/data separator (├).
        assert!(!out.contains('\u{251C}'));
    }

    // -- TableRenderer: Simple ---------------------------------------------

    #[test]
    fn table_render_simple_empty() {
        let table = TableRenderer::new();
        assert_eq!(table.render_simple(), "");
    }

    #[test]
    fn table_render_simple_with_data() {
        let mut table = TableRenderer::with_headers(vec!["H1".into(), "H2".into()]);
        table.add_row(vec!["a".into(), "b".into()]);
        let out = table.render_simple();
        assert!(out.contains("|"));
        assert!(out.contains("+"));
        assert!(out.contains("-"));
        assert!(out.contains("H1"));
        assert!(out.contains("a"));
    }

    #[test]
    fn table_render_simple_no_headers() {
        let mut table = TableRenderer::new();
        table.add_row(vec!["1".into(), "2".into(), "3".into()]);
        table.add_row(vec!["4".into(), "5".into(), "6".into()]);
        let out = table.render_simple();
        let lines: Vec<&str> = out.lines().collect();
        // separator, row, row, separator => 4 lines
        assert_eq!(lines.len(), 4);
    }

    // -- TableRenderer: edge cases ----------------------------------------

    #[test]
    fn table_ragged_rows() {
        let mut table = TableRenderer::with_headers(vec!["A".into(), "B".into(), "C".into()]);
        table.add_row(vec!["1".into()]); // short row
        table.add_row(vec!["a".into(), "b".into(), "c".into(), "d".into()]); // long row
        let out = table.render();
        // Should not panic and should contain all headers.
        assert!(out.contains("A"));
        assert!(out.contains("B"));
        assert!(out.contains("C"));
    }

    #[test]
    fn table_alignment_with_long_text() {
        let mut table = TableRenderer::with_headers(vec!["Short".into(), "Header".into()]);
        table.set_alignment(1, ColumnAlignment::Right);
        table.add_row(vec!["x".into(), "a really long cell value".into()]);
        let out = table.render();
        assert!(out.contains("a really long cell value"));
    }

    // -- JsonFormatter -----------------------------------------------------

    #[test]
    fn json_format_pretty() {
        let val = json!({"name": "test", "count": 42});
        let out = JsonFormatter::format(&val);
        assert!(out.contains("\"name\""));
        assert!(out.contains("\"test\""));
        assert!(out.contains("42"));
        // Pretty should have newlines.
        assert!(out.contains('\n'));
    }

    #[test]
    fn json_format_compact() {
        let val = json!({"a": 1, "b": 2});
        let out = JsonFormatter::format_compact(&val);
        assert!(!out.contains('\n'));
        assert!(out.contains("\"a\""));
    }

    #[test]
    fn json_format_with_empty_comments() {
        let val = json!({"key": "val"});
        let comments = HashMap::new();
        let out = JsonFormatter::format_with_comments(&val, &comments);
        // Should be identical to pretty-printed output.
        assert_eq!(out, JsonFormatter::format(&val));
    }

    #[test]
    fn json_format_with_comments() {
        let val = json!({"name": "alice", "age": 30});
        let mut comments = HashMap::new();
        comments.insert("name".into(), "The user name".into());
        let out = JsonFormatter::format_with_comments(&val, &comments);
        assert!(out.contains("// The user name"));
        assert!(out.contains("\"name\""));
    }

    #[test]
    fn json_format_nested_with_comments() {
        let val = json!({"outer": {"inner": 1}});
        let mut comments = HashMap::new();
        comments.insert("inner".into(), "nested comment".into());
        let out = JsonFormatter::format_with_comments(&val, &comments);
        assert!(out.contains("// nested comment"));
    }

    // -- TextFormatter -----------------------------------------------------

    #[test]
    fn text_format_sections_empty() {
        let out = TextFormatter::format_sections(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn text_format_sections_single() {
        let sections = vec![("Title".into(), "Body text.".into())];
        let out = TextFormatter::format_sections(&sections);
        assert!(out.contains("Title"));
        assert!(out.contains("====="));
        assert!(out.contains("Body text."));
    }

    #[test]
    fn text_format_sections_multiple() {
        let sections = vec![
            ("First".into(), "AAA".into()),
            ("Second".into(), "BBB".into()),
        ];
        let out = TextFormatter::format_sections(&sections);
        let first_pos = out.find("First").unwrap();
        let second_pos = out.find("Second").unwrap();
        assert!(first_pos < second_pos);
        // Second section should have a blank line before it.
        let between = &out[first_pos..second_pos];
        assert!(between.contains("\n\n"));
    }

    #[test]
    fn text_format_sections_underline_length() {
        let heading = "Hello World";
        let sections = vec![(heading.into(), "body".into())];
        let out = TextFormatter::format_sections(&sections);
        let underline = "=".repeat(heading.len());
        assert!(out.contains(&underline));
    }

    #[test]
    fn text_format_key_value_empty() {
        assert_eq!(TextFormatter::format_key_value(&[]), "");
    }

    #[test]
    fn text_format_key_value() {
        let pairs = vec![
            ("Name".into(), "Alice".into()),
            ("Age".into(), "30".into()),
            ("Location".into(), "NYC".into()),
        ];
        let out = TextFormatter::format_key_value(&pairs);
        // All colons should be vertically aligned.
        let colon_positions: Vec<usize> = out
            .lines()
            .filter_map(|l| l.find(':'))
            .collect();
        assert!(!colon_positions.is_empty());
        let first = colon_positions[0];
        for pos in &colon_positions {
            assert_eq!(*pos, first, "colons should be aligned");
        }
    }

    #[test]
    fn text_format_bar_zero() {
        let out = TextFormatter::format_bar("mem", 0.0, 100.0, 20);
        assert!(out.contains("["));
        assert!(out.contains("]"));
        assert!(out.contains("0%"));
        assert!(out.contains("--------------------")); // 20 dashes
    }

    #[test]
    fn text_format_bar_full() {
        let out = TextFormatter::format_bar("cpu", 100.0, 100.0, 10);
        assert!(out.contains("##########")); // 10 hashes
        assert!(out.contains("100%"));
    }

    #[test]
    fn text_format_bar_half() {
        let out = TextFormatter::format_bar("io", 50.0, 100.0, 20);
        assert!(out.contains("50%"));
    }

    #[test]
    fn text_format_bar_clamps_over_max() {
        let out = TextFormatter::format_bar("x", 200.0, 100.0, 10);
        assert!(out.contains("100%"));
    }

    #[test]
    fn text_format_bar_zero_max() {
        // Should not panic; treat max as 1.
        let out = TextFormatter::format_bar("z", 0.0, 0.0, 10);
        assert!(out.contains("0%"));
    }

    #[test]
    fn text_format_sparkline_empty() {
        assert_eq!(TextFormatter::format_sparkline(&[]), "");
    }

    #[test]
    fn text_format_sparkline_single() {
        let out = TextFormatter::format_sparkline(&[5.0]);
        assert_eq!(out.len(), 3); // one UTF-8 block char (3 bytes in ▄)
    }

    #[test]
    fn text_format_sparkline_ascending() {
        let out = TextFormatter::format_sparkline(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        // First char should be the lowest block, last the highest.
        let chars: Vec<char> = out.chars().collect();
        assert_eq!(chars.len(), 8);
        assert_eq!(chars[0], '\u{2581}'); // ▁
        assert_eq!(chars[7], '\u{2588}'); // █
    }

    #[test]
    fn text_format_sparkline_constant() {
        let out = TextFormatter::format_sparkline(&[5.0, 5.0, 5.0]);
        let chars: Vec<char> = out.chars().collect();
        // All same value → all mid-range block (index 3 = ▄).
        assert!(chars.iter().all(|&c| c == '\u{2584}'));
    }

    // -- MarkdownFormatter -------------------------------------------------

    #[test]
    fn markdown_format_sections_empty() {
        let out = MarkdownFormatter::format_sections(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn markdown_format_sections() {
        let sections = vec![("Overview".into(), "Some text.".into())];
        let out = MarkdownFormatter::format_sections(&sections);
        assert!(out.contains("## Overview"));
        assert!(out.contains("Some text."));
    }

    #[test]
    fn markdown_format_table_empty_headers() {
        assert_eq!(MarkdownFormatter::format_table(&[], &[]), "");
    }

    #[test]
    fn markdown_format_table() {
        let headers = vec!["Col1".into(), "Col2".into()];
        let rows = vec![
            vec!["a".into(), "b".into()],
            vec!["c".into(), "d".into()],
        ];
        let out = MarkdownFormatter::format_table(&headers, &rows);
        // Should start with a pipe.
        assert!(out.starts_with('|'));
        // Should have separator line with dashes.
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.len() >= 3);
        assert!(lines[1].contains("---"));
        assert!(out.contains("Col1"));
        assert!(out.contains("a"));
    }

    #[test]
    fn markdown_format_table_alignment() {
        let headers = vec!["Name".into(), "Score".into()];
        let rows = vec![vec!["alice".into(), "100".into()]];
        let out = MarkdownFormatter::format_table(&headers, &rows);
        // Verify columns are pipe-separated.
        for line in out.lines() {
            let pipe_count = line.chars().filter(|&c| c == '|').count();
            assert!(pipe_count >= 3); // |col1|col2|
        }
    }

    #[test]
    fn markdown_format_checklist() {
        let items = vec![
            ("Run tests".into(), true),
            ("Deploy staging".into(), false),
            ("Deploy prod".into(), false),
        ];
        let out = MarkdownFormatter::format_checklist(&items);
        assert!(out.contains("- [x] Run tests"));
        assert!(out.contains("- [ ] Deploy staging"));
        assert!(out.contains("- [ ] Deploy prod"));
    }

    #[test]
    fn markdown_format_checklist_empty() {
        assert_eq!(MarkdownFormatter::format_checklist(&[]), "");
    }

    // -- HtmlFormatter -----------------------------------------------------

    #[test]
    fn html_format_page_structure() {
        let sections = vec![("Heading".into(), "Content".into())];
        let out = HtmlFormatter::format_page(&sections, "My Report");
        assert!(out.contains("<!DOCTYPE html>"));
        assert!(out.contains("<title>My Report</title>"));
        assert!(out.contains("<h1>My Report</h1>"));
        assert!(out.contains("<h2>Heading</h2>"));
        assert!(out.contains("Content"));
        assert!(out.contains("</html>"));
    }

    #[test]
    fn html_format_page_escapes() {
        let sections = vec![("A <b>bold</b> heading".into(), "x & y".into())];
        let out = HtmlFormatter::format_page(&sections, "Test <script>");
        assert!(out.contains("&lt;script&gt;"));
        assert!(out.contains("&lt;b&gt;bold&lt;/b&gt;"));
        assert!(out.contains("x &amp; y"));
    }

    #[test]
    fn html_format_page_has_css() {
        let out = HtmlFormatter::format_page(&[], "T");
        assert!(out.contains("<style>"));
        assert!(out.contains("font-family"));
    }

    #[test]
    fn html_format_table() {
        let headers = vec!["X".into(), "Y".into()];
        let rows = vec![vec!["1".into(), "2".into()]];
        let out = HtmlFormatter::format_table(&headers, &rows);
        assert!(out.contains("<table>"));
        assert!(out.contains("<th>X</th>"));
        assert!(out.contains("<td>1</td>"));
        assert!(out.contains("</table>"));
    }

    #[test]
    fn html_format_table_escapes() {
        let headers = vec!["<h>".into()];
        let rows = vec![vec!["a&b".into()]];
        let out = HtmlFormatter::format_table(&headers, &rows);
        assert!(out.contains("&lt;h&gt;"));
        assert!(out.contains("a&amp;b"));
    }

    #[test]
    fn html_format_collapsible() {
        let out = HtmlFormatter::format_collapsible("Click me", "Hidden content");
        assert!(out.contains("<details>"));
        assert!(out.contains("<summary>Click me</summary>"));
        assert!(out.contains("Hidden content"));
        assert!(out.contains("</details>"));
    }

    #[test]
    fn html_format_collapsible_escapes() {
        let out = HtmlFormatter::format_collapsible("<script>", "a & b");
        assert!(out.contains("&lt;script&gt;"));
        assert!(out.contains("a &amp; b"));
    }

    // -- OutputFormatter (façade) ------------------------------------------

    #[test]
    fn output_formatter_json() {
        let val = json!({"ok": true});
        let out = OutputFormatter::format_json(&val);
        assert!(out.contains("\"ok\""));
        assert!(out.contains("true"));
    }

    #[test]
    fn output_formatter_text() {
        let sections = vec![("S".into(), "Body".into())];
        let out = OutputFormatter::format_text(&sections);
        assert!(out.contains("S"));
        assert!(out.contains("="));
    }

    #[test]
    fn output_formatter_markdown() {
        let sections = vec![("S".into(), "Body".into())];
        let out = OutputFormatter::format_markdown(&sections);
        assert!(out.contains("## S"));
    }

    #[test]
    fn output_formatter_html() {
        let sections = vec![("S".into(), "Body".into())];
        let out = OutputFormatter::format_html(&sections, "Title");
        assert!(out.contains("<title>Title</title>"));
    }

    // -- Integration: roundtrip table through different renderers ----------

    #[test]
    fn table_unicode_vs_simple_same_data() {
        let mut table = TableRenderer::with_headers(vec![
            "Service".into(),
            "From".into(),
            "To".into(),
        ]);
        table.add_row(vec!["api".into(), "1.0.0".into(), "2.0.0".into()]);
        table.add_row(vec!["web".into(), "3.1.0".into(), "3.2.0".into()]);

        let unicode = table.render();
        let simple = table.render_simple();

        // Both should contain the same data.
        for word in &["Service", "api", "web", "1.0.0", "2.0.0", "3.1.0", "3.2.0"] {
            assert!(unicode.contains(word), "unicode missing {}", word);
            assert!(simple.contains(word), "simple missing {}", word);
        }

        // Unicode uses box chars, simple uses ASCII.
        assert!(unicode.contains('\u{2500}'));
        assert!(!simple.contains('\u{2500}'));
    }

    #[test]
    fn table_single_column() {
        let mut table = TableRenderer::with_headers(vec!["Items".into()]);
        table.add_row(vec!["one".into()]);
        table.add_row(vec!["two".into()]);
        let out = table.render();
        assert!(out.contains("Items"));
        assert!(out.contains("one"));
        // Single column should have no column separators (┬/┼/┴).
        assert!(!out.contains('\u{252C}'));
        assert!(!out.contains('\u{253C}'));
        assert!(!out.contains('\u{2534}'));
    }

    #[test]
    fn text_bar_visual_width() {
        let out = TextFormatter::format_bar("net", 75.0, 100.0, 20);
        // 75% of 20 = 15 filled, 5 empty.
        assert!(out.contains(&"#".repeat(15)));
        assert!(out.contains(&"-".repeat(5)));
    }

    #[test]
    fn sparkline_descending() {
        let vals = vec![100.0, 75.0, 50.0, 25.0, 0.0];
        let out = TextFormatter::format_sparkline(&vals);
        let chars: Vec<char> = out.chars().collect();
        assert_eq!(chars.len(), 5);
        assert_eq!(chars[0], '\u{2588}'); // █  highest
        assert_eq!(chars[4], '\u{2581}'); // ▁  lowest
    }

    #[test]
    fn markdown_table_wide_cells() {
        let headers = vec!["Short".into(), "A Much Longer Header".into()];
        let rows = vec![vec!["x".into(), "y".into()]];
        let out = MarkdownFormatter::format_table(&headers, &rows);
        // The separator dashes should be at least as wide as the header.
        let sep_line = out.lines().nth(1).unwrap();
        let dash_segments: Vec<&str> = sep_line.split('|').filter(|s| !s.is_empty()).collect();
        for seg in &dash_segments {
            assert!(seg.trim().chars().all(|c| c == '-'));
        }
    }

    #[test]
    fn html_page_multiple_sections() {
        let sections = vec![
            ("A".into(), "aaa".into()),
            ("B".into(), "bbb".into()),
            ("C".into(), "ccc".into()),
        ];
        let out = HtmlFormatter::format_page(&sections, "Multi");
        let section_count = out.matches("<section>").count();
        assert_eq!(section_count, 3);
    }

    #[test]
    fn html_table_empty() {
        let out = HtmlFormatter::format_table(&[], &[]);
        assert!(out.contains("<table>"));
        assert!(out.contains("</table>"));
    }

    #[test]
    fn json_format_array() {
        let val = json!([1, 2, 3]);
        let pretty = JsonFormatter::format(&val);
        let compact = JsonFormatter::format_compact(&val);
        assert!(pretty.contains('\n'));
        assert!(!compact.contains('\n'));
        assert!(compact.contains("[1,2,3]"));
    }

    #[test]
    fn json_format_with_multiple_comments() {
        let val = json!({"alpha": 1, "beta": 2, "gamma": 3});
        let mut comments = HashMap::new();
        comments.insert("alpha".into(), "first".into());
        comments.insert("gamma".into(), "third".into());
        let out = JsonFormatter::format_with_comments(&val, &comments);
        assert!(out.contains("// first"));
        assert!(out.contains("// third"));
        assert!(!out.contains("// second")); // beta has no comment
    }
}
