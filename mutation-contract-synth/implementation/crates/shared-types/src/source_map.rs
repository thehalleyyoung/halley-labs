//! Source-location mapping from IR positions back to original source.
//!
//! The [`SourceMap`] is populated during AST-to-IR lowering so that later
//! phases (mutation, contract synthesis, reporting) can attribute results
//! to the original program text.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::errors::SpanInfo;

// ---------------------------------------------------------------------------
// SourceRange
// ---------------------------------------------------------------------------

/// A contiguous range in the original source text, identified by file, start
/// and end offsets (byte-based), and line/column for human display.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRange {
    pub file: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl SourceRange {
    pub fn new(
        file: impl Into<String>,
        start_offset: usize,
        end_offset: usize,
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
    ) -> Self {
        Self {
            file: file.into(),
            start_offset,
            end_offset,
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// Create a zero-width range at the given position.
    pub fn point(file: impl Into<String>, line: usize, col: usize, offset: usize) -> Self {
        Self {
            file: file.into(),
            start_offset: offset,
            end_offset: offset,
            start_line: line,
            start_col: col,
            end_line: line,
            end_col: col,
        }
    }

    /// Byte length of this range.
    pub fn byte_len(&self) -> usize {
        self.end_offset.saturating_sub(self.start_offset)
    }

    /// Number of lines covered.
    pub fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.start_line) + 1
    }

    /// Returns true if `other` is entirely within this range.
    pub fn contains(&self, other: &SourceRange) -> bool {
        self.file == other.file
            && self.start_offset <= other.start_offset
            && self.end_offset >= other.end_offset
    }

    /// Returns true if the two ranges overlap.
    pub fn overlaps(&self, other: &SourceRange) -> bool {
        self.file == other.file
            && self.start_offset < other.end_offset
            && other.start_offset < self.end_offset
    }

    /// Merge with another range from the same file, producing the smallest
    /// range that covers both.
    pub fn merge(&self, other: &SourceRange) -> Option<SourceRange> {
        if self.file != other.file {
            return None;
        }
        let start_offset = self.start_offset.min(other.start_offset);
        let end_offset = self.end_offset.max(other.end_offset);
        let (start_line, start_col) = if self.start_offset <= other.start_offset {
            (self.start_line, self.start_col)
        } else {
            (other.start_line, other.start_col)
        };
        let (end_line, end_col) = if self.end_offset >= other.end_offset {
            (self.end_line, self.end_col)
        } else {
            (other.end_line, other.end_col)
        };
        Some(SourceRange {
            file: self.file.clone(),
            start_offset,
            end_offset,
            start_line,
            start_col,
            end_line,
            end_col,
        })
    }

    /// Convert to a [`SpanInfo`].
    pub fn to_span_info(&self) -> SpanInfo {
        use crate::errors::SourceLocation;
        SpanInfo::new(
            SourceLocation::new(&self.file, self.start_line, self.start_col),
            SourceLocation::new(&self.file, self.end_line, self.end_col),
        )
    }
}

impl fmt::Display for SourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start_line == self.end_line {
            write!(
                f,
                "{}:{}:{}-{}",
                self.file, self.start_line, self.start_col, self.end_col
            )
        } else {
            write!(
                f,
                "{}:{}:{}-{}:{}",
                self.file, self.start_line, self.start_col, self.end_line, self.end_col
            )
        }
    }
}

// ---------------------------------------------------------------------------
// MappingEntry
// ---------------------------------------------------------------------------

/// A single entry linking an IR location (basic-block id + statement index)
/// back to a source range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingEntry {
    pub block_id: usize,
    pub stmt_index: usize,
    pub source: SourceRange,
    pub description: Option<String>,
}

impl MappingEntry {
    pub fn new(block_id: usize, stmt_index: usize, source: SourceRange) -> Self {
        Self {
            block_id,
            stmt_index,
            source,
            description: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Key used for internal indexing: `(block_id, stmt_index)`.
    pub fn ir_key(&self) -> (usize, usize) {
        (self.block_id, self.stmt_index)
    }
}

impl fmt::Display for MappingEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BB{}[{}] -> {}",
            self.block_id, self.stmt_index, self.source
        )?;
        if let Some(ref desc) = self.description {
            write!(f, " ({desc})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SourceMap
// ---------------------------------------------------------------------------

/// Maps IR positions (basic-block id, statement index) to source locations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceMap {
    entries: BTreeMap<(usize, usize), MappingEntry>,
    /// Mapping from function name to its overall source range.
    function_ranges: BTreeMap<String, SourceRange>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping entry.
    pub fn add(&mut self, entry: MappingEntry) {
        self.entries.insert(entry.ir_key(), entry);
    }

    /// Add a mapping from position to source range.
    pub fn add_mapping(&mut self, block_id: usize, stmt_index: usize, source: SourceRange) {
        self.add(MappingEntry::new(block_id, stmt_index, source));
    }

    /// Record the source range of a function.
    pub fn add_function_range(&mut self, name: impl Into<String>, range: SourceRange) {
        self.function_ranges.insert(name.into(), range);
    }

    /// Look up the source range for an IR position.
    pub fn lookup(&self, block_id: usize, stmt_index: usize) -> Option<&SourceRange> {
        self.entries.get(&(block_id, stmt_index)).map(|e| &e.source)
    }

    /// Look up the full mapping entry.
    pub fn lookup_entry(&self, block_id: usize, stmt_index: usize) -> Option<&MappingEntry> {
        self.entries.get(&(block_id, stmt_index))
    }

    /// Look up the source range for a function by name.
    pub fn function_range(&self, name: &str) -> Option<&SourceRange> {
        self.function_ranges.get(name)
    }

    /// Iterate all mapping entries.
    pub fn entries(&self) -> impl Iterator<Item = &MappingEntry> {
        self.entries.values()
    }

    /// Number of IR-position entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the source map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find all IR positions that map to locations within the given source range.
    pub fn find_in_range(&self, range: &SourceRange) -> Vec<&MappingEntry> {
        self.entries
            .values()
            .filter(|e| range.contains(&e.source) || range.overlaps(&e.source))
            .collect()
    }

    /// Find all entries for a given basic block.
    pub fn entries_for_block(&self, block_id: usize) -> Vec<&MappingEntry> {
        self.entries
            .range((block_id, 0)..=(block_id, usize::MAX))
            .map(|(_, e)| e)
            .collect()
    }

    /// Return a merged source range covering all entries in a block.
    pub fn block_range(&self, block_id: usize) -> Option<SourceRange> {
        let entries = self.entries_for_block(block_id);
        let mut merged: Option<SourceRange> = None;
        for entry in entries {
            merged = Some(match merged {
                None => entry.source.clone(),
                Some(prev) => prev.merge(&entry.source).unwrap_or(prev),
            });
        }
        merged
    }

    /// Return the overall source range covering the entire program.
    pub fn global_range(&self) -> Option<SourceRange> {
        let mut merged: Option<SourceRange> = None;
        for entry in self.entries.values() {
            merged = Some(match merged {
                None => entry.source.clone(),
                Some(prev) => prev.merge(&entry.source).unwrap_or(prev),
            });
        }
        merged
    }

    /// Clear all mappings.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.function_ranges.clear();
    }

    /// Merge another source map into this one.
    pub fn merge_from(&mut self, other: &SourceMap) {
        for (key, entry) in &other.entries {
            self.entries.insert(*key, entry.clone());
        }
        for (name, range) in &other.function_ranges {
            self.function_ranges.insert(name.clone(), range.clone());
        }
    }
}

impl fmt::Display for SourceMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SourceMap ({} entries):", self.entries.len())?;
        for entry in self.entries.values() {
            writeln!(f, "  {entry}")?;
        }
        for (name, range) in &self.function_ranges {
            writeln!(f, "  fn {name} -> {range}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_range(line: usize) -> SourceRange {
        SourceRange::new("test.ms", line * 10, line * 10 + 20, line, 1, line, 20)
    }

    #[test]
    fn test_source_range_byte_len() {
        let r = SourceRange::new("a.ms", 0, 50, 1, 1, 3, 10);
        assert_eq!(r.byte_len(), 50);
    }

    #[test]
    fn test_source_range_line_count() {
        let r = SourceRange::new("a.ms", 0, 50, 1, 1, 3, 10);
        assert_eq!(r.line_count(), 3);
    }

    #[test]
    fn test_source_range_point() {
        let r = SourceRange::point("a.ms", 5, 10, 42);
        assert_eq!(r.byte_len(), 0);
        assert_eq!(r.line_count(), 1);
    }

    #[test]
    fn test_source_range_contains() {
        let outer = SourceRange::new("a.ms", 0, 100, 1, 1, 10, 20);
        let inner = SourceRange::new("a.ms", 10, 50, 2, 1, 5, 10);
        let outside = SourceRange::new("a.ms", 90, 120, 9, 1, 12, 5);
        assert!(outer.contains(&inner));
        assert!(!outer.contains(&outside));
    }

    #[test]
    fn test_source_range_overlaps() {
        let a = SourceRange::new("a.ms", 0, 50, 1, 1, 5, 10);
        let b = SourceRange::new("a.ms", 30, 80, 3, 1, 8, 10);
        let c = SourceRange::new("a.ms", 60, 90, 6, 1, 9, 10);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_source_range_merge() {
        let a = SourceRange::new("a.ms", 10, 50, 2, 1, 5, 10);
        let b = SourceRange::new("a.ms", 30, 80, 4, 1, 8, 10);
        let merged = a.merge(&b).unwrap();
        assert_eq!(merged.start_offset, 10);
        assert_eq!(merged.end_offset, 80);
    }

    #[test]
    fn test_source_range_merge_different_files() {
        let a = SourceRange::new("a.ms", 10, 50, 2, 1, 5, 10);
        let b = SourceRange::new("b.ms", 30, 80, 4, 1, 8, 10);
        assert!(a.merge(&b).is_none());
    }

    #[test]
    fn test_source_range_display_same_line() {
        let r = SourceRange::new("test.ms", 0, 10, 5, 1, 5, 11);
        let s = r.to_string();
        assert!(s.contains("test.ms:5:1-11"));
    }

    #[test]
    fn test_source_range_display_multi_line() {
        let r = SourceRange::new("test.ms", 0, 100, 5, 1, 10, 20);
        let s = r.to_string();
        assert!(s.contains("5:1-10:20"));
    }

    #[test]
    fn test_source_range_to_span_info() {
        let r = SourceRange::new("a.ms", 0, 10, 1, 1, 1, 10);
        let span = r.to_span_info();
        assert_eq!(span.start.line, 1);
        assert_eq!(span.end.column, 10);
    }

    #[test]
    fn test_mapping_entry() {
        let entry = MappingEntry::new(0, 2, sample_range(5)).with_description("assignment");
        assert_eq!(entry.ir_key(), (0, 2));
        let s = entry.to_string();
        assert!(s.contains("BB0[2]"));
        assert!(s.contains("assignment"));
    }

    #[test]
    fn test_source_map_add_lookup() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, sample_range(1));
        sm.add_mapping(0, 1, sample_range(2));
        sm.add_mapping(1, 0, sample_range(3));
        assert_eq!(sm.len(), 3);
        assert!(sm.lookup(0, 0).is_some());
        assert!(sm.lookup(0, 1).is_some());
        assert!(sm.lookup(2, 0).is_none());
    }

    #[test]
    fn test_source_map_entries_for_block() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, sample_range(1));
        sm.add_mapping(0, 1, sample_range(2));
        sm.add_mapping(1, 0, sample_range(3));
        let entries = sm.entries_for_block(0);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_source_map_function_range() {
        let mut sm = SourceMap::new();
        sm.add_function_range("foo", sample_range(1));
        assert!(sm.function_range("foo").is_some());
        assert!(sm.function_range("bar").is_none());
    }

    #[test]
    fn test_source_map_block_range() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, SourceRange::new("a.ms", 0, 20, 1, 1, 2, 10));
        sm.add_mapping(0, 1, SourceRange::new("a.ms", 15, 40, 2, 5, 4, 10));
        let range = sm.block_range(0).unwrap();
        assert_eq!(range.start_offset, 0);
        assert_eq!(range.end_offset, 40);
    }

    #[test]
    fn test_source_map_find_in_range() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, SourceRange::new("a.ms", 10, 20, 2, 1, 2, 10));
        sm.add_mapping(0, 1, SourceRange::new("a.ms", 50, 60, 5, 1, 5, 10));
        let search = SourceRange::new("a.ms", 0, 30, 1, 1, 3, 10);
        let found = sm.find_in_range(&search);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].stmt_index, 0);
    }

    #[test]
    fn test_source_map_merge_from() {
        let mut sm1 = SourceMap::new();
        sm1.add_mapping(0, 0, sample_range(1));
        let mut sm2 = SourceMap::new();
        sm2.add_mapping(1, 0, sample_range(2));
        sm2.add_function_range("bar", sample_range(3));
        sm1.merge_from(&sm2);
        assert_eq!(sm1.len(), 2);
        assert!(sm1.function_range("bar").is_some());
    }

    #[test]
    fn test_source_map_clear() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, sample_range(1));
        sm.add_function_range("f", sample_range(2));
        sm.clear();
        assert!(sm.is_empty());
        assert!(sm.function_range("f").is_none());
    }

    #[test]
    fn test_source_map_display() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, sample_range(1));
        let s = format!("{sm}");
        assert!(s.contains("1 entries"));
        assert!(s.contains("BB0[0]"));
    }

    #[test]
    fn test_source_map_global_range_empty() {
        let sm = SourceMap::new();
        assert!(sm.global_range().is_none());
    }

    #[test]
    fn test_source_map_global_range() {
        let mut sm = SourceMap::new();
        sm.add_mapping(0, 0, SourceRange::new("a.ms", 0, 20, 1, 1, 2, 10));
        sm.add_mapping(1, 0, SourceRange::new("a.ms", 30, 60, 4, 1, 6, 10));
        let range = sm.global_range().unwrap();
        assert_eq!(range.start_offset, 0);
        assert_eq!(range.end_offset, 60);
    }
}
