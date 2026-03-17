//! Source location tracking for mapping analysis results back to code.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// A location in source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

impl SourceLocation {
    pub fn new(file: impl Into<PathBuf>, line: u32, column: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            end_line: None,
            end_column: None,
        }
    }

    pub fn with_end(mut self, end_line: u32, end_column: u32) -> Self {
        self.end_line = Some(end_line);
        self.end_column = Some(end_column);
        self
    }

    pub fn span_lines(&self) -> u32 {
        self.end_line.unwrap_or(self.line) - self.line + 1
    }

    pub fn contains_line(&self, line: u32) -> bool {
        line >= self.line && line <= self.end_line.unwrap_or(self.line)
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file.display(), self.line, self.column)?;
        if let (Some(el), Some(ec)) = (self.end_line, self.end_column) {
            write!(f, "-{}:{}", el, ec)?;
        }
        Ok(())
    }
}

/// A span of source code with its text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpan {
    pub location: SourceLocation,
    pub text: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

impl SourceSpan {
    pub fn new(location: SourceLocation, text: impl Into<String>) -> Self {
        Self {
            location,
            text: text.into(),
            context_before: Vec::new(),
            context_after: Vec::new(),
        }
    }

    pub fn with_context(mut self, before: Vec<String>, after: Vec<String>) -> Self {
        self.context_before = before;
        self.context_after = after;
        self
    }
}

/// A mapping from analysis objects to their source locations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceMap {
    entries: Vec<SourceMapEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMapEntry {
    pub id: String,
    pub kind: SourceEntityKind,
    pub location: SourceLocation,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceEntityKind {
    Function,
    Loop,
    Assignment,
    Expression,
    ForceComputation,
    IntegratorStep,
    BoundaryCondition,
    LibraryCall,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, id: impl Into<String>, kind: SourceEntityKind, location: SourceLocation, description: impl Into<String>) {
        self.entries.push(SourceMapEntry {
            id: id.into(),
            kind,
            location,
            description: description.into(),
        });
    }

    pub fn find_by_line(&self, file: &str, line: u32) -> Vec<&SourceMapEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.location.file.to_str() == Some(file) && e.location.contains_line(line)
            })
            .collect()
    }

    pub fn find_by_kind(&self, kind: SourceEntityKind) -> Vec<&SourceMapEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("test.py", 42, 8);
        assert_eq!(format!("{}", loc), "test.py:42:8");
    }

    #[test]
    fn test_source_location_span() {
        let loc = SourceLocation::new("test.py", 10, 1).with_end(15, 20);
        assert_eq!(loc.span_lines(), 6);
        assert!(loc.contains_line(12));
        assert!(!loc.contains_line(16));
    }

    #[test]
    fn test_source_map() {
        let mut map = SourceMap::new();
        map.add_entry("f1", SourceEntityKind::ForceComputation,
            SourceLocation::new("sim.py", 10, 1).with_end(20, 1),
            "Gravitational force");
        map.add_entry("s1", SourceEntityKind::IntegratorStep,
            SourceLocation::new("sim.py", 25, 1).with_end(30, 1),
            "Leapfrog step");
        assert_eq!(map.find_by_line("sim.py", 15).len(), 1);
        assert_eq!(map.find_by_kind(SourceEntityKind::IntegratorStep).len(), 1);
    }
}
