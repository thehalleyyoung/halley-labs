//! Source-location types.
//!
//! Lightweight types for tracking where in a source program a given
//! floating-point operation lives.  Used throughout diagnosis and repair
//! reports so that users can navigate back to the offending code.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SourceSpan ─────────────────────────────────────────────────────────────

/// A span inside a source file, identified by line and column ranges.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    /// Path to the source file (relative or absolute).
    pub file: String,
    /// Starting line (1-based).
    pub line_start: u32,
    /// Ending line (1-based, inclusive).
    pub line_end: u32,
    /// Starting column (1-based).
    pub col_start: u32,
    /// Ending column (1-based, inclusive).
    pub col_end: u32,
}

impl SourceSpan {
    /// Create a span.
    pub fn new(
        file: impl Into<String>,
        line_start: u32,
        line_end: u32,
        col_start: u32,
        col_end: u32,
    ) -> Self {
        Self {
            file: file.into(),
            line_start,
            line_end,
            col_start,
            col_end,
        }
    }

    /// Create a single-line span.
    pub fn line(file: impl Into<String>, line: u32, col_start: u32, col_end: u32) -> Self {
        Self::new(file, line, line, col_start, col_end)
    }

    /// Create a point span (single character).
    pub fn point(file: impl Into<String>, line: u32, col: u32) -> Self {
        Self::new(file, line, line, col, col)
    }

    /// Whether the span covers exactly one line.
    pub fn is_single_line(&self) -> bool {
        self.line_start == self.line_end
    }

    /// Number of lines covered.
    pub fn line_count(&self) -> u32 {
        self.line_end.saturating_sub(self.line_start) + 1
    }

    /// Whether this span contains the given line number.
    pub fn contains_line(&self, line: u32) -> bool {
        line >= self.line_start && line <= self.line_end
    }

    /// Merge two spans into the smallest span covering both.
    pub fn merge(&self, other: &Self) -> Option<Self> {
        if self.file != other.file {
            return None;
        }
        Some(Self {
            file: self.file.clone(),
            line_start: self.line_start.min(other.line_start),
            line_end: self.line_end.max(other.line_end),
            col_start: if self.line_start <= other.line_start {
                self.col_start
            } else {
                other.col_start
            },
            col_end: if self.line_end >= other.line_end {
                self.col_end
            } else {
                other.col_end
            },
        })
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_single_line() {
            write!(
                f,
                "{}:{}:{}-{}",
                self.file, self.line_start, self.col_start, self.col_end
            )
        } else {
            write!(
                f,
                "{}:{}:{}-{}:{}",
                self.file, self.line_start, self.col_start, self.line_end, self.col_end
            )
        }
    }
}

// ─── SourceFile ─────────────────────────────────────────────────────────────

/// Metadata about a source file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceFile {
    /// File path.
    pub path: String,
    /// SHA-256 hex digest of the file content (for cache validation).
    pub content_hash: String,
}

impl SourceFile {
    /// Create a new source-file descriptor.
    pub fn new(path: impl Into<String>, content_hash: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content_hash: content_hash.into(),
        }
    }
}

impl fmt::Display for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.path,
            &self.content_hash[..8.min(self.content_hash.len())]
        )
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_single_line() {
        let s = SourceSpan::line("foo.c", 10, 5, 15);
        assert!(s.is_single_line());
        assert_eq!(s.line_count(), 1);
        assert!(s.contains_line(10));
        assert!(!s.contains_line(11));
    }

    #[test]
    fn span_multi_line() {
        let s = SourceSpan::new("bar.c", 10, 20, 1, 30);
        assert!(!s.is_single_line());
        assert_eq!(s.line_count(), 11);
        assert!(s.contains_line(15));
    }

    #[test]
    fn span_point() {
        let s = SourceSpan::point("baz.c", 5, 3);
        assert!(s.is_single_line());
        assert_eq!(s.col_start, 3);
        assert_eq!(s.col_end, 3);
    }

    #[test]
    fn span_merge_same_file() {
        let a = SourceSpan::line("f.c", 10, 1, 10);
        let b = SourceSpan::line("f.c", 20, 5, 15);
        let m = a.merge(&b).unwrap();
        assert_eq!(m.line_start, 10);
        assert_eq!(m.line_end, 20);
    }

    #[test]
    fn span_merge_different_files() {
        let a = SourceSpan::line("a.c", 1, 1, 1);
        let b = SourceSpan::line("b.c", 1, 1, 1);
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn span_display_single_line() {
        let s = SourceSpan::line("test.c", 42, 5, 10);
        assert_eq!(s.to_string(), "test.c:42:5-10");
    }

    #[test]
    fn span_display_multi_line() {
        let s = SourceSpan::new("test.c", 10, 20, 1, 30);
        let d = s.to_string();
        assert!(d.contains("test.c"));
        assert!(d.contains("10"));
    }

    #[test]
    fn source_file_display() {
        let sf = SourceFile::new("main.c", "abcdef0123456789");
        let d = sf.to_string();
        assert!(d.contains("main.c"));
        assert!(d.contains("abcdef01"));
    }

    #[test]
    fn serde_source_span() {
        let s = SourceSpan::line("x.c", 1, 1, 5);
        let json = serde_json::to_string(&s).unwrap();
        let s2: SourceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn serde_source_file() {
        let sf = SourceFile::new("y.c", "hash123");
        let json = serde_json::to_string(&sf).unwrap();
        let sf2: SourceFile = serde_json::from_str(&json).unwrap();
        assert_eq!(sf, sf2);
    }
}
