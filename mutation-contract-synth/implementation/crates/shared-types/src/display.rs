//! Shared display utilities for the MutSpec system.
//!
//! Provides [`IndentWriter`] for pretty-printing with indentation,
//! [`TableFormatter`] for tabular CLI output, and miscellaneous
//! formatting helpers.

use std::fmt;

// ---------------------------------------------------------------------------
// IndentWriter
// ---------------------------------------------------------------------------

/// A writer that tracks indentation level and emits properly indented lines.
#[derive(Debug, Clone)]
pub struct IndentWriter {
    buf: String,
    indent: usize,
    indent_str: String,
    at_line_start: bool,
}

impl IndentWriter {
    /// Create a new `IndentWriter` using the given per-level indent string.
    pub fn new(indent_str: impl Into<String>) -> Self {
        Self {
            buf: String::new(),
            indent: 0,
            indent_str: indent_str.into(),
            at_line_start: true,
        }
    }

    /// Create an `IndentWriter` that uses `n` spaces per level.
    pub fn with_spaces(n: usize) -> Self {
        Self::new(" ".repeat(n))
    }

    /// Create an `IndentWriter` that uses tab characters.
    pub fn with_tabs() -> Self {
        Self::new("\t")
    }

    /// Increase indentation by one level.
    pub fn indent(&mut self) {
        self.indent += 1;
    }

    /// Decrease indentation by one level (saturating).
    pub fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Set indentation to an absolute level.
    pub fn set_indent(&mut self, level: usize) {
        self.indent = level;
    }

    /// Current indentation level.
    pub fn level(&self) -> usize {
        self.indent
    }

    /// Write a string, inserting indentation after every newline.
    pub fn write(&mut self, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                self.buf.push('\n');
                self.at_line_start = true;
            } else {
                if self.at_line_start {
                    for _ in 0..self.indent {
                        self.buf.push_str(&self.indent_str);
                    }
                    self.at_line_start = false;
                }
                self.buf.push(ch);
            }
        }
    }

    /// Write a complete line (appends a newline).
    pub fn write_line(&mut self, s: &str) {
        self.write(s);
        self.write("\n");
    }

    /// Write a blank line.
    pub fn blank_line(&mut self) {
        self.buf.push('\n');
        self.at_line_start = true;
    }

    /// Write an opening brace and increase indent.
    pub fn open_brace(&mut self) {
        self.write_line("{");
        self.indent();
    }

    /// Decrease indent and write a closing brace.
    pub fn close_brace(&mut self) {
        self.dedent();
        self.write_line("}");
    }

    /// Consume the writer and return the accumulated string.
    pub fn finish(self) -> String {
        self.buf
    }

    /// Return a reference to the current buffer.
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Clear the buffer, keeping settings.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.at_line_start = true;
    }

    /// Current length of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl fmt::Display for IndentWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.buf)
    }
}

impl fmt::Write for IndentWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write(s);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TableFormatter
// ---------------------------------------------------------------------------

/// Simple fixed-width table formatter for CLI output.
#[derive(Debug, Clone)]
pub struct TableFormatter {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    col_widths: Vec<usize>,
    separator: char,
}

impl TableFormatter {
    /// Create a table with the given column headers.
    pub fn new(headers: Vec<String>) -> Self {
        let col_widths = headers.iter().map(|h| h.len()).collect();
        Self {
            headers,
            rows: Vec::new(),
            col_widths,
            separator: '|',
        }
    }

    /// Create a table from string-slice headers.
    pub fn from_strs(headers: &[&str]) -> Self {
        Self::new(headers.iter().map(|s| s.to_string()).collect())
    }

    /// Set the column separator character.
    pub fn with_separator(mut self, sep: char) -> Self {
        self.separator = sep;
        self
    }

    /// Add a row of values.
    pub fn add_row(&mut self, row: Vec<String>) {
        for (i, cell) in row.iter().enumerate() {
            if i < self.col_widths.len() {
                self.col_widths[i] = self.col_widths[i].max(cell.len());
            }
        }
        self.rows.push(row);
    }

    /// Add a row from string slices.
    pub fn add_row_strs(&mut self, row: &[&str]) {
        self.add_row(row.iter().map(|s| s.to_string()).collect());
    }

    /// Number of rows (excluding header).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn col_count(&self) -> usize {
        self.headers.len()
    }

    /// Render the table as a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_row(&self.headers, &mut out);
        let sep_line: Vec<String> = self.col_widths.iter().map(|w| "-".repeat(*w)).collect();
        self.render_row(&sep_line, &mut out);
        for row in &self.rows {
            self.render_row(row, &mut out);
        }
        out
    }

    fn render_row(&self, row: &[String], out: &mut String) {
        let sep = self.separator;
        out.push(sep);
        for (i, width) in self.col_widths.iter().enumerate() {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            for _ in cell.len()..*width {
                out.push(' ');
            }
            out.push(' ');
            out.push(sep);
        }
        out.push('\n');
    }
}

impl fmt::Display for TableFormatter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

// ---------------------------------------------------------------------------
// Colour support detection
// ---------------------------------------------------------------------------

/// Returns `true` if the terminal likely supports ANSI colour codes.
pub fn color_supported() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if let Ok(term) = std::env::var("TERM") {
        return term != "dumb";
    }
    false
}

// ---------------------------------------------------------------------------
// Truncation utilities
// ---------------------------------------------------------------------------

/// Truncate a string to at most `max_len` characters, appending an ellipsis
/// if truncation occurred.
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        ".".repeat(max_len)
    } else {
        let mut result = String::with_capacity(max_len);
        for (i, ch) in s.chars().enumerate() {
            if i >= max_len - 3 {
                result.push_str("...");
                break;
            }
            result.push(ch);
        }
        result
    }
}

/// Truncate lines, keeping the first `max_lines` and appending a summary.
pub fn truncate_lines(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= max_lines {
        return s.to_string();
    }
    let kept: Vec<&str> = lines[..max_lines].to_vec();
    let omitted = lines.len() - max_lines;
    format!("{}\n... ({omitted} more lines)", kept.join("\n"))
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// Format a large number with thousands separators.
pub fn format_number(n: i64) -> String {
    if n < 0 {
        return format!("-{}", format_number(-n));
    }
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

/// Format a duration in seconds to a human-readable string.
pub fn format_duration_secs(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.0}\u{00B5}s", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.1}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else if secs < 3600.0 {
        let m = (secs / 60.0).floor() as u64;
        let s = secs - (m as f64 * 60.0);
        format!("{m}m {s:.1}s")
    } else {
        let h = (secs / 3600.0).floor() as u64;
        let m = ((secs - h as f64 * 3600.0) / 60.0).floor() as u64;
        format!("{h}h {m}m")
    }
}

/// Format a byte count to a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indent_writer_basic() {
        let mut w = IndentWriter::with_spaces(4);
        w.write_line("hello");
        w.indent();
        w.write_line("world");
        w.dedent();
        w.write_line("done");
        let s = w.finish();
        assert!(s.contains("hello"));
        assert!(s.contains("    world"));
        assert!(s.contains("done"));
    }

    #[test]
    fn test_indent_writer_braces() {
        let mut w = IndentWriter::with_spaces(2);
        w.write_line("fn foo()");
        w.open_brace();
        w.write_line("return 1;");
        w.close_brace();
        let s = w.finish();
        assert!(s.contains("{"));
        assert!(s.contains("  return 1;"));
        assert!(s.contains("}"));
    }

    #[test]
    fn test_indent_writer_clear() {
        let mut w = IndentWriter::with_spaces(2);
        w.write_line("hello");
        assert!(!w.is_empty());
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn test_indent_writer_display() {
        let mut w = IndentWriter::with_spaces(2);
        w.write_line("test");
        let displayed = format!("{}", w);
        assert!(displayed.contains("test"));
    }

    #[test]
    fn test_indent_writer_fmt_write() {
        use std::fmt::Write;
        let mut w = IndentWriter::with_spaces(2);
        w.indent();
        write!(w, "hello {}", 42).unwrap();
        let s = w.finish();
        assert!(s.contains("  hello 42"));
    }

    #[test]
    fn test_table_formatter() {
        let mut t = TableFormatter::from_strs(&["Name", "Value"]);
        t.add_row_strs(&["x", "42"]);
        t.add_row_strs(&["flag", "true"]);
        let s = t.render();
        assert!(s.contains("Name"));
        assert!(s.contains("42"));
        assert!(s.contains("flag"));
        assert_eq!(t.row_count(), 2);
        assert_eq!(t.col_count(), 2);
    }

    #[test]
    fn test_table_display() {
        let mut t = TableFormatter::from_strs(&["A", "B"]);
        t.add_row_strs(&["1", "2"]);
        let displayed = format!("{}", t);
        assert!(displayed.contains("A"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("hi", 2), "hi");
        assert_eq!(truncate("hello", 3), "...");
    }

    #[test]
    fn test_truncate_lines() {
        let text = "a\nb\nc\nd\ne";
        let result = truncate_lines(text, 2);
        assert!(result.contains("a"));
        assert!(result.contains("b"));
        assert!(result.contains("3 more lines"));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(-42000), "-42,000");
    }

    #[test]
    fn test_format_duration() {
        let us = format_duration_secs(0.0001);
        assert!(us.contains("\u{00B5}s"));
        let ms = format_duration_secs(0.5);
        assert!(ms.contains("ms"));
        let sec = format_duration_secs(5.0);
        assert!(sec.contains("s"));
        let min = format_duration_secs(90.0);
        assert!(min.contains("m"));
        let hr = format_duration_secs(7200.0);
        assert!(hr.contains("h"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert!(format_bytes(2048).contains("KB"));
        assert!(format_bytes(5_000_000).contains("MB"));
        assert!(format_bytes(5_000_000_000).contains("GB"));
    }

    #[test]
    fn test_color_supported_no_color() {
        // NO_COLOR=1 should disable colours; just test the function exists.
        let _ = color_supported();
    }
}
