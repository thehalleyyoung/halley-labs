use std::fmt;

/// A span in source code, tracking byte offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn empty() -> Self {
        Self { start: 0, end: 0 }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// Human-readable source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Registry of source files and their contents for mapping spans to locations.
#[derive(Debug, Clone)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

#[derive(Debug, Clone)]
struct SourceFile {
    name: String,
    source: String,
    /// Byte offset of each line start.
    line_starts: Vec<usize>,
    /// Global byte offset where this file starts.
    global_offset: usize,
}

impl SourceFile {
    fn new(name: String, source: String, global_offset: usize) -> Self {
        let mut line_starts = vec![0];
        for (i, ch) in source.char_indices() {
            if ch == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            name,
            source,
            line_starts,
            global_offset,
        }
    }

    fn contains(&self, offset: usize) -> bool {
        offset >= self.global_offset && offset < self.global_offset + self.source.len()
    }

    fn local_offset(&self, global: usize) -> usize {
        global - self.global_offset
    }

    fn line_col(&self, local_offset: usize) -> (usize, usize) {
        let line = match self.line_starts.binary_search(&local_offset) {
            Ok(l) => l,
            Err(l) => l.saturating_sub(1),
        };
        let col = local_offset - self.line_starts[line];
        (line + 1, col + 1) // 1-indexed
    }

    fn get_line(&self, line_1indexed: usize) -> Option<&str> {
        let idx = line_1indexed.checked_sub(1)?;
        let start = *self.line_starts.get(idx)?;
        let end = self
            .line_starts
            .get(idx + 1)
            .copied()
            .unwrap_or(self.source.len());
        let text = &self.source[start..end];
        Some(text.trim_end_matches('\n').trim_end_matches('\r'))
    }
}

impl SourceMap {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Register a source file. Returns the global offset assigned to this file.
    pub fn add_file(&mut self, name: impl Into<String>, source: impl Into<String>) -> usize {
        let global_offset = self
            .files
            .last()
            .map(|f| f.global_offset + f.source.len())
            .unwrap_or(0);
        self.files
            .push(SourceFile::new(name.into(), source.into(), global_offset));
        global_offset
    }

    /// Look up the source location for a byte offset.
    pub fn lookup(&self, offset: usize) -> Option<SourceLocation> {
        for file in &self.files {
            if file.contains(offset) {
                let local = file.local_offset(offset);
                let (line, column) = file.line_col(local);
                return Some(SourceLocation {
                    file: file.name.clone(),
                    line,
                    column,
                });
            }
        }
        // If offset is at the very end of the last file
        if let Some(file) = self.files.last() {
            if offset == file.global_offset + file.source.len() {
                let local = file.source.len().saturating_sub(1);
                let (line, column) = file.line_col(local);
                return Some(SourceLocation {
                    file: file.name.clone(),
                    line,
                    column,
                });
            }
        }
        None
    }

    /// Look up the source location for a span start.
    pub fn lookup_span(&self, span: Span) -> Option<SourceLocation> {
        self.lookup(span.start)
    }

    /// Get the source line text for a given span.
    pub fn get_source_line(&self, span: Span) -> Option<String> {
        let loc = self.lookup(span.start)?;
        for file in &self.files {
            if file.name == loc.file {
                return file.get_line(loc.line).map(String::from);
            }
        }
        None
    }

    /// Get the source text for a span.
    pub fn get_source_text(&self, span: Span) -> Option<&str> {
        for file in &self.files {
            if file.contains(span.start) {
                let local_start = file.local_offset(span.start);
                let local_end =
                    file.local_offset(span.end.min(file.global_offset + file.source.len()));
                return Some(&file.source[local_start..local_end]);
            }
        }
        None
    }

    /// Format a span as a human-readable location string.
    pub fn format_span(&self, span: Span) -> String {
        match self.lookup(span.start) {
            Some(loc) => format!("{}", loc),
            None => format!("offset {}", span.start),
        }
    }

    /// Format a span with source context for error display.
    pub fn format_span_with_context(&self, span: Span) -> String {
        let loc = match self.lookup(span.start) {
            Some(l) => l,
            None => return format!("at offset {}", span.start),
        };
        let mut out = format!("--> {}\n", loc);
        if let Some(line_text) = self.get_source_line(span) {
            let line_num = format!("{}", loc.line);
            let padding = " ".repeat(line_num.len());
            out.push_str(&format!("{} |\n", padding));
            out.push_str(&format!("{} | {}\n", line_num, line_text));
            out.push_str(&format!("{} |", padding));
            let col_offset = loc.column.saturating_sub(1);
            let underline_len = span.len().max(1).min(line_text.len().saturating_sub(col_offset));
            out.push_str(&" ".repeat(col_offset + 1));
            out.push_str(&"^".repeat(underline_len));
            out.push('\n');
        }
        out
    }

    /// Merge two spans.
    pub fn merge_spans(&self, a: Span, b: Span) -> Span {
        a.merge(b)
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let a = Span::new(5, 10);
        let b = Span::new(8, 20);
        let merged = a.merge(b);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_source_map_single_file() {
        let mut sm = SourceMap::new();
        sm.add_file("test.rsl", "line one\nline two\nline three\n");
        let loc = sm.lookup(0).unwrap();
        assert_eq!(loc.file, "test.rsl");
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        let loc2 = sm.lookup(9).unwrap();
        assert_eq!(loc2.line, 2);
        assert_eq!(loc2.column, 1);

        let loc3 = sm.lookup(14).unwrap();
        assert_eq!(loc3.line, 2);
        assert_eq!(loc3.column, 6);
    }

    #[test]
    fn test_source_map_multi_file() {
        let mut sm = SourceMap::new();
        sm.add_file("a.rsl", "abc\n");
        sm.add_file("b.rsl", "def\n");
        let loc_a = sm.lookup(2).unwrap();
        assert_eq!(loc_a.file, "a.rsl");
        let loc_b = sm.lookup(5).unwrap();
        assert_eq!(loc_b.file, "b.rsl");
    }

    #[test]
    fn test_get_source_line() {
        let mut sm = SourceMap::new();
        sm.add_file("test.rsl", "alpha\nbeta\ngamma\n");
        let line = sm.get_source_line(Span::new(6, 10)).unwrap();
        assert_eq!(line, "beta");
    }

    #[test]
    fn test_format_span_with_context() {
        let mut sm = SourceMap::new();
        sm.add_file("test.rsl", "let x = 42;\nlet y = bad;\n");
        let ctx = sm.format_span_with_context(Span::new(20, 23));
        assert!(ctx.contains("test.rsl"));
        assert!(ctx.contains("let y = bad;"));
    }
}
