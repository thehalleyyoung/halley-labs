//! PIT XML report parser and CSV kill-matrix reader.
//!
//! PIT (<https://pitest.org>) produces mutation reports in two main formats:
//!
//! 1. **XML** — the default `mutations.xml` file containing one `<mutation>`
//!    element per generated mutant, nested under a `<mutations>` root.
//! 2. **CSV** — an optional `kill_matrix.csv` where each row records whether a
//!    specific test killed a specific mutant.
//!
//! This module provides [`PitXmlParser`] and [`PitCsvParser`] for reading both
//! formats into structured [`PitMutation`] and [`CsvKillEntry`] values that
//! downstream code can convert into MutSpec internal types.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::errors::{PitError, PitResult};

// ---------------------------------------------------------------------------
// PIT mutation detection status (from PIT's DetectionStatus enum)
// ---------------------------------------------------------------------------

/// Detection status reported by PIT for a single mutant.
///
/// Mirrors PIT's `org.pitest.mutationtest.DetectionStatus` enum.  The
/// discriminants are matched case-insensitively when parsing XML/CSV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PitDetectionStatus {
    /// The mutant was killed by at least one test.
    Killed,
    /// The mutant survived — no test detected the change.
    Survived,
    /// The mutant caused a timeout (infinite loop, deadlock).
    TimedOut,
    /// The mutant could not be loaded by the JVM (e.g. verification error).
    NonViable,
    /// There was a JVM-level error applying the mutant.
    MemoryError,
    /// The mutant was not covered by any test.
    NoCoverage,
    /// PIT determined the mutant was not started (internal skip).
    NotStarted,
    /// PIT reports the run was started but no result was recorded.
    Started,
    /// PIT explicitly marks the mutant as run-error.
    RunError,
}

impl PitDetectionStatus {
    /// Parse a status string as produced by PIT (case-insensitive).
    pub fn from_pit_string(s: &str) -> PitResult<Self> {
        match s.to_uppercase().as_str() {
            "KILLED" => Ok(PitDetectionStatus::Killed),
            "SURVIVED" | "LIVED" => Ok(PitDetectionStatus::Survived),
            "TIMED_OUT" | "TIMEDOUT" => Ok(PitDetectionStatus::TimedOut),
            "NON_VIABLE" | "NONVIABLE" => Ok(PitDetectionStatus::NonViable),
            "MEMORY_ERROR" | "MEMORYERROR" => Ok(PitDetectionStatus::MemoryError),
            "NO_COVERAGE" | "NOCOVERAGE" => Ok(PitDetectionStatus::NoCoverage),
            "NOT_STARTED" | "NOTSTARTED" => Ok(PitDetectionStatus::NotStarted),
            "STARTED" => Ok(PitDetectionStatus::Started),
            "RUN_ERROR" | "RUNERROR" => Ok(PitDetectionStatus::RunError),
            _ => Err(PitError::UnknownStatus {
                path: PathBuf::new(),
                status: s.to_string(),
            }),
        }
    }

    /// Returns `true` if this status means the mutant was detected (killed / timed-out).
    pub fn is_detected(&self) -> bool {
        matches!(
            self,
            PitDetectionStatus::Killed | PitDetectionStatus::TimedOut
        )
    }

    /// Returns `true` if the mutant survived (was NOT detected).
    pub fn is_survived(&self) -> bool {
        matches!(self, PitDetectionStatus::Survived)
    }

    /// Returns `true` if the status represents an error or non-viable condition.
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            PitDetectionStatus::NonViable
                | PitDetectionStatus::MemoryError
                | PitDetectionStatus::RunError
        )
    }
}

impl fmt::Display for PitDetectionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            PitDetectionStatus::Killed => "KILLED",
            PitDetectionStatus::Survived => "SURVIVED",
            PitDetectionStatus::TimedOut => "TIMED_OUT",
            PitDetectionStatus::NonViable => "NON_VIABLE",
            PitDetectionStatus::MemoryError => "MEMORY_ERROR",
            PitDetectionStatus::NoCoverage => "NO_COVERAGE",
            PitDetectionStatus::NotStarted => "NOT_STARTED",
            PitDetectionStatus::Started => "STARTED",
            PitDetectionStatus::RunError => "RUN_ERROR",
        };
        write!(f, "{label}")
    }
}

// ---------------------------------------------------------------------------
// PitMutation — one <mutation> element from the XML report
// ---------------------------------------------------------------------------

/// A single mutation as parsed from PIT's XML `<mutation>` element.
///
/// # XML structure
///
/// ```xml
/// <mutation detected="true" status="KILLED" numberOfTestsRun="3">
///   <sourceFile>Calculator.java</sourceFile>
///   <mutatedClass>com.example.Calculator</mutatedClass>
///   <mutatedMethod>add</mutatedMethod>
///   <methodDescription>(II)I</methodDescription>
///   <lineNumber>42</lineNumber>
///   <mutator>org.pitest.mutationtest.engine.gregor.mutators.MathMutator</mutator>
///   <indexes><index>12</index></indexes>
///   <blocks><block>3</block></blocks>
///   <killingTest>com.example.CalculatorTest::testAdd</killingTest>
///   <description>replaced int return with 0 for com/example/Calculator::add</description>
/// </mutation>
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PitMutation {
    /// Whether PIT considers this mutant detected.
    pub detected: bool,
    /// PIT detection status string (KILLED, SURVIVED, etc.).
    pub status: PitDetectionStatus,
    /// Number of tests PIT ran against this mutant.
    pub number_of_tests_run: u32,
    /// Relative path to the source file (e.g. `Calculator.java`).
    pub source_file: String,
    /// Fully-qualified mutated class name (e.g. `com.example.Calculator`).
    pub mutated_class: String,
    /// Name of the mutated method (e.g. `add`).
    pub mutated_method: String,
    /// JVM method descriptor (e.g. `(II)I`).
    pub method_description: String,
    /// Source line where the mutation was applied.
    pub line_number: u32,
    /// Fully-qualified PIT mutator class name.
    pub mutator: String,
    /// Bytecode instruction indices touched by the mutation.
    pub indexes: Vec<u32>,
    /// Basic-block indices touched by the mutation.
    pub blocks: Vec<u32>,
    /// The fully-qualified test that killed this mutant, if any.
    pub killing_test: Option<String>,
    /// Human-readable description of the mutation.
    pub description: Option<String>,
}

impl PitMutation {
    /// Returns the simple class name (last segment after the final `.`).
    pub fn simple_class_name(&self) -> &str {
        self.mutated_class
            .rsplit('.')
            .next()
            .unwrap_or(&self.mutated_class)
    }

    /// Returns the package path (everything before the last `.`).
    pub fn package(&self) -> Option<&str> {
        self.mutated_class.rsplit_once('.').map(|(pkg, _)| pkg)
    }

    /// Returns the short mutator name (last segment of the fully-qualified class).
    pub fn short_mutator_name(&self) -> &str {
        self.mutator.rsplit('.').next().unwrap_or(&self.mutator)
    }

    /// Returns a compact identifier string usable as a map key.
    pub fn canonical_key(&self) -> String {
        format!(
            "{}::{}:{}_{}",
            self.mutated_class,
            self.mutated_method,
            self.line_number,
            self.short_mutator_name()
        )
    }
}

impl fmt::Display for PitMutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{status}] {class}::{method} L{line} ({mutator})",
            status = self.status,
            class = self.simple_class_name(),
            method = self.mutated_method,
            line = self.line_number,
            mutator = self.short_mutator_name(),
        )?;
        if let Some(ref test) = self.killing_test {
            write!(f, " killed by {test}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PitXmlParser
// ---------------------------------------------------------------------------

/// Parser for PIT's standard XML mutation report (`mutations.xml`).
///
/// The parser is intentionally lenient: it skips individual `<mutation>`
/// elements that contain unrecognised statuses (logging a warning) rather
/// than aborting the entire parse.  Structured errors are returned only for
/// fatal problems (I/O, malformed XML).
pub struct PitXmlParser {
    /// Path to the XML file currently being parsed (for error messages).
    path: PathBuf,
    /// Whether to skip mutations with unrecognised statuses instead of erroring.
    lenient: bool,
    /// Accumulated warnings generated during lenient parsing.
    warnings: Vec<String>,
}

impl PitXmlParser {
    /// Create a new parser that will read from `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lenient: true,
            warnings: Vec::new(),
        }
    }

    /// Set strict mode — any unrecognised element/status causes an error.
    pub fn strict(mut self) -> Self {
        self.lenient = false;
        self
    }

    /// Set lenient mode (default) — skip unrecognised mutations with warnings.
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Returns warnings accumulated during the last parse.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Parse the XML file and return all mutations.
    ///
    /// This is a hand-rolled SAX-style parser that reads the XML as text and
    /// extracts `<mutation>` elements without requiring a full XML crate
    /// dependency.  It handles the simple, predictable structure that PIT
    /// produces.
    pub fn parse(&mut self) -> PitResult<Vec<PitMutation>> {
        let content = std::fs::read_to_string(&self.path).map_err(|e| PitError::Io {
            path: self.path.clone(),
            source: e,
        })?;
        self.warnings.clear();
        self.parse_str(&content)
    }

    /// Parse from an in-memory XML string (useful for testing).
    pub fn parse_str(&mut self, xml: &str) -> PitResult<Vec<PitMutation>> {
        self.warnings.clear();
        let mut mutations = Vec::new();

        // Split on <mutation to find each mutation element.
        let chunks: Vec<&str> = xml.split("<mutation").collect();

        // First chunk is everything before the first <mutation (the preamble).
        for (idx, chunk) in chunks.iter().enumerate().skip(1) {
            // Re-add the tag prefix for attribute parsing.
            let element = format!("<mutation{chunk}");
            match self.parse_mutation_element(&element, idx) {
                Ok(m) => mutations.push(m),
                Err(e) if self.lenient && e.is_recoverable() => {
                    let warning = format!("Skipping mutation #{idx}: {e}");
                    log::warn!("{warning}");
                    self.warnings.push(warning);
                }
                Err(e) => return Err(e),
            }
        }

        log::info!(
            "Parsed {} mutations from `{}`{}",
            mutations.len(),
            self.path.display(),
            if self.warnings.is_empty() {
                String::new()
            } else {
                format!(" ({} warnings)", self.warnings.len())
            }
        );
        Ok(mutations)
    }

    /// Parse a single `<mutation ...> ... </mutation>` element.
    fn parse_mutation_element(&self, element: &str, idx: usize) -> PitResult<PitMutation> {
        let detected = self
            .extract_attribute(element, "detected")
            .unwrap_or_else(|| "false".to_string());
        let detected = detected == "true";

        let status_str = self
            .extract_attribute(element, "status")
            .unwrap_or_else(|| "SURVIVED".to_string());
        let status = PitDetectionStatus::from_pit_string(&status_str).map_err(|_| {
            PitError::UnknownStatus {
                path: self.path.clone(),
                status: status_str.clone(),
            }
        })?;

        let number_of_tests_run = self
            .extract_attribute(element, "numberOfTestsRun")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let source_file = self
            .extract_child_text(element, "sourceFile")
            .ok_or_else(|| PitError::MissingElement {
                path: self.path.clone(),
                element: "sourceFile".into(),
                line: idx,
            })?;

        let mutated_class = self
            .extract_child_text(element, "mutatedClass")
            .ok_or_else(|| PitError::MissingElement {
                path: self.path.clone(),
                element: "mutatedClass".into(),
                line: idx,
            })?;

        let mutated_method = self
            .extract_child_text(element, "mutatedMethod")
            .ok_or_else(|| PitError::MissingElement {
                path: self.path.clone(),
                element: "mutatedMethod".into(),
                line: idx,
            })?;

        let method_description = self
            .extract_child_text(element, "methodDescription")
            .unwrap_or_default();

        let line_number_str = self
            .extract_child_text(element, "lineNumber")
            .ok_or_else(|| PitError::MissingElement {
                path: self.path.clone(),
                element: "lineNumber".into(),
                line: idx,
            })?;
        let line_number =
            line_number_str
                .parse::<u32>()
                .map_err(|_| PitError::InvalidAttribute {
                    path: self.path.clone(),
                    attribute: "lineNumber".into(),
                    value: line_number_str.clone(),
                    reason: "expected u32".into(),
                })?;

        let mutator = self.extract_child_text(element, "mutator").ok_or_else(|| {
            PitError::MissingElement {
                path: self.path.clone(),
                element: "mutator".into(),
                line: idx,
            }
        })?;

        let indexes = self.extract_child_list(element, "indexes", "index");
        let blocks = self.extract_child_list(element, "blocks", "block");

        let killing_test = self
            .extract_child_text(element, "killingTest")
            .filter(|s| !s.is_empty() && s != "none" && s != "null");

        let description = self
            .extract_child_text(element, "description")
            .filter(|s| !s.is_empty());

        Ok(PitMutation {
            detected,
            status,
            number_of_tests_run,
            source_file,
            mutated_class,
            mutated_method,
            method_description,
            line_number,
            mutator,
            indexes,
            blocks,
            killing_test,
            description,
        })
    }

    /// Extract the value of an XML attribute from the opening tag.
    fn extract_attribute(&self, element: &str, attr: &str) -> Option<String> {
        // Match patterns like `attr="value"` or `attr='value'`.
        let needle_dq = format!("{attr}=\"");
        let needle_sq = format!("{attr}='");

        if let Some(start) = element.find(&needle_dq) {
            let value_start = start + needle_dq.len();
            let rest = &element[value_start..];
            rest.find('"').map(|end| rest[..end].to_string())
        } else if let Some(start) = element.find(&needle_sq) {
            let value_start = start + needle_sq.len();
            let rest = &element[value_start..];
            rest.find('\'').map(|end| rest[..end].to_string())
        } else {
            None
        }
    }

    /// Extract the text content of a child element `<tag>text</tag>`.
    fn extract_child_text(&self, element: &str, tag: &str) -> Option<String> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        if let Some(start) = element.find(&open) {
            let text_start = start + open.len();
            let rest = &element[text_start..];
            rest.find(&close).map(|end| rest[..end].trim().to_string())
        } else {
            None
        }
    }

    /// Extract a list of integer values from a nested repeated element.
    ///
    /// E.g. `<indexes><index>12</index><index>34</index></indexes>` → `[12, 34]`.
    fn extract_child_list(&self, element: &str, wrapper: &str, item: &str) -> Vec<u32> {
        let open_wrapper = format!("<{wrapper}>");
        let close_wrapper = format!("</{wrapper}>");
        let Some(start) = element.find(&open_wrapper) else {
            return Vec::new();
        };
        let inner_start = start + open_wrapper.len();
        let rest = &element[inner_start..];
        let Some(end) = rest.find(&close_wrapper) else {
            return Vec::new();
        };
        let inner = &rest[..end];

        let open_item = format!("<{item}>");
        let close_item = format!("</{item}>");
        let mut values = Vec::new();
        let mut search_from = 0;
        while let Some(item_start) = inner[search_from..].find(&open_item) {
            let text_start = search_from + item_start + open_item.len();
            if let Some(item_end) = inner[text_start..].find(&close_item) {
                let text = inner[text_start..text_start + item_end].trim();
                if let Ok(v) = text.parse::<u32>() {
                    values.push(v);
                }
                search_from = text_start + item_end + close_item.len();
            } else {
                break;
            }
        }
        values
    }
}

// ---------------------------------------------------------------------------
// CsvKillEntry — one row from the CSV kill matrix
// ---------------------------------------------------------------------------

/// A single row from PIT's CSV kill-matrix export.
///
/// The CSV typically has columns like:
/// `mutant_id,mutated_class,mutated_method,line_number,mutator,killing_test,status`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvKillEntry {
    /// Opaque mutant identifier (may be an integer or hash).
    pub mutant_id: String,
    /// Fully-qualified mutated class.
    pub mutated_class: String,
    /// Method name.
    pub mutated_method: String,
    /// Source line number.
    pub line_number: u32,
    /// Mutator class name.
    pub mutator: String,
    /// Test that killed this mutant (empty if survived).
    pub killing_test: Option<String>,
    /// Detection status string.
    pub status: PitDetectionStatus,
}

impl fmt::Display for CsvKillEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{status}] {cls}::{method} L{line} ({mutator})",
            status = self.status,
            cls = self.mutated_class,
            method = self.mutated_method,
            line = self.line_number,
            mutator = self.mutator,
        )
    }
}

// ---------------------------------------------------------------------------
// PitCsvParser
// ---------------------------------------------------------------------------

/// Parser for PIT's CSV kill-matrix files.
///
/// The expected CSV format uses a header row followed by data rows:
/// ```text
/// mutant_id,mutated_class,mutated_method,line_number,mutator,killing_test,status
/// 1,com.example.Calc,add,42,MathMutator,com.example.CalcTest::testAdd,KILLED
/// ```
pub struct PitCsvParser {
    path: PathBuf,
    /// Column-name-to-index mapping (built from the header row).
    column_map: HashMap<String, usize>,
}

impl PitCsvParser {
    /// Create a new CSV parser for the given file.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            column_map: HashMap::new(),
        }
    }

    /// Parse the CSV file and return all kill entries.
    pub fn parse(&mut self) -> PitResult<Vec<CsvKillEntry>> {
        let content = std::fs::read_to_string(&self.path).map_err(|e| PitError::Io {
            path: self.path.clone(),
            source: e,
        })?;
        self.parse_str(&content)
    }

    /// Parse from an in-memory CSV string.
    pub fn parse_str(&mut self, csv: &str) -> PitResult<Vec<CsvKillEntry>> {
        let mut lines = csv.lines();
        let header = lines.next().ok_or_else(|| PitError::CsvFormat {
            path: self.path.clone(),
            record: 0,
            message: "empty CSV — no header row".into(),
        })?;

        self.build_column_map(header);
        let mut entries = Vec::new();

        for (row_idx, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record_num = row_idx + 1;
            let entry = self.parse_csv_row(line, record_num)?;
            entries.push(entry);
        }

        log::info!(
            "Parsed {} CSV kill-matrix entries from `{}`",
            entries.len(),
            self.path.display()
        );
        Ok(entries)
    }

    /// Build the column-name-to-index mapping from the header row.
    fn build_column_map(&mut self, header: &str) {
        self.column_map.clear();
        for (i, col) in header.split(',').enumerate() {
            let col = col.trim().to_lowercase();
            self.column_map.insert(col, i);
        }
    }

    /// Look up a column value by name in a split row.
    fn column_value<'a>(&self, fields: &[&'a str], name: &str) -> Option<&'a str> {
        self.column_map
            .get(&name.to_lowercase())
            .and_then(|&idx| fields.get(idx))
            .map(|s| s.trim())
    }

    /// Parse a single CSV data row.
    fn parse_csv_row(&self, line: &str, record: usize) -> PitResult<CsvKillEntry> {
        let fields: Vec<&str> = line.split(',').collect();

        // If we have a column map, use named lookups; otherwise positional.
        let mutant_id = self
            .column_value(&fields, "mutant_id")
            .or_else(|| fields.first().copied())
            .unwrap_or("")
            .to_string();

        let mutated_class = self
            .column_value(&fields, "mutated_class")
            .or_else(|| fields.get(1).copied())
            .unwrap_or("")
            .to_string();

        let mutated_method = self
            .column_value(&fields, "mutated_method")
            .or_else(|| fields.get(2).copied())
            .unwrap_or("")
            .to_string();

        let line_number_str = self
            .column_value(&fields, "line_number")
            .or_else(|| fields.get(3).copied())
            .unwrap_or("0");
        let line_number = line_number_str
            .parse::<u32>()
            .map_err(|_| PitError::CsvFormat {
                path: self.path.clone(),
                record,
                message: format!("invalid line_number `{line_number_str}`"),
            })?;

        let mutator = self
            .column_value(&fields, "mutator")
            .or_else(|| fields.get(4).copied())
            .unwrap_or("")
            .to_string();

        let killing_test_str = self
            .column_value(&fields, "killing_test")
            .or_else(|| fields.get(5).copied())
            .unwrap_or("")
            .trim()
            .to_string();
        let killing_test = if killing_test_str.is_empty()
            || killing_test_str == "none"
            || killing_test_str == "null"
        {
            None
        } else {
            Some(killing_test_str)
        };

        let status_str = self
            .column_value(&fields, "status")
            .or_else(|| fields.get(6).copied())
            .unwrap_or("SURVIVED");
        let status = PitDetectionStatus::from_pit_string(status_str)?;

        Ok(CsvKillEntry {
            mutant_id,
            mutated_class,
            mutated_method,
            line_number,
            mutator,
            killing_test,
            status,
        })
    }
}

// ---------------------------------------------------------------------------
// PitReportSummary — aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics computed from a parsed PIT report.
///
/// Useful for quick status displays and validation checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PitReportSummary {
    /// Total number of mutations generated.
    pub total_mutations: usize,
    /// Number of mutations detected (killed + timed-out).
    pub detected: usize,
    /// Number of mutations that survived.
    pub survived: usize,
    /// Number of mutations with no test coverage.
    pub no_coverage: usize,
    /// Number of non-viable / error mutations.
    pub errors: usize,
    /// Per-mutator counts.
    pub per_mutator: HashMap<String, usize>,
    /// Per-class counts.
    pub per_class: HashMap<String, usize>,
    /// Distinct killing tests.
    pub killing_tests: Vec<String>,
}

impl PitReportSummary {
    /// Build a summary from a slice of parsed mutations.
    pub fn from_mutations(mutations: &[PitMutation]) -> Self {
        let total_mutations = mutations.len();
        let mut detected = 0usize;
        let mut survived = 0usize;
        let mut no_coverage = 0usize;
        let mut errors = 0usize;
        let mut per_mutator: HashMap<String, usize> = HashMap::new();
        let mut per_class: HashMap<String, usize> = HashMap::new();
        let mut killing_tests_set: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for m in mutations {
            match m.status {
                PitDetectionStatus::Killed | PitDetectionStatus::TimedOut => detected += 1,
                PitDetectionStatus::Survived => survived += 1,
                PitDetectionStatus::NoCoverage => no_coverage += 1,
                _ if m.status.is_error() => errors += 1,
                _ => {}
            }
            *per_mutator
                .entry(m.short_mutator_name().to_string())
                .or_insert(0) += 1;
            *per_class
                .entry(m.simple_class_name().to_string())
                .or_insert(0) += 1;
            if let Some(ref test) = m.killing_test {
                killing_tests_set.insert(test.clone());
            }
        }

        let mut killing_tests: Vec<String> = killing_tests_set.into_iter().collect();
        killing_tests.sort();

        PitReportSummary {
            total_mutations,
            detected,
            survived,
            no_coverage,
            errors,
            per_mutator,
            per_class,
            killing_tests,
        }
    }

    /// Mutation score as a percentage (detected / (total - errors - no_coverage)).
    pub fn mutation_score(&self) -> f64 {
        let testable = self.total_mutations - self.errors - self.no_coverage;
        if testable == 0 {
            return 0.0;
        }
        (self.detected as f64 / testable as f64) * 100.0
    }

    /// Test strength: detected / total.
    pub fn test_strength(&self) -> f64 {
        if self.total_mutations == 0 {
            return 0.0;
        }
        (self.detected as f64 / self.total_mutations as f64) * 100.0
    }

    /// Number of distinct mutator types encountered.
    pub fn mutator_count(&self) -> usize {
        self.per_mutator.len()
    }

    /// Number of distinct classes that were mutated.
    pub fn class_count(&self) -> usize {
        self.per_class.len()
    }
}

impl fmt::Display for PitReportSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PIT Report Summary")?;
        writeln!(f, "  Total mutations:  {}", self.total_mutations)?;
        writeln!(f, "  Detected:         {}", self.detected)?;
        writeln!(f, "  Survived:         {}", self.survived)?;
        writeln!(f, "  No coverage:      {}", self.no_coverage)?;
        writeln!(f, "  Errors:           {}", self.errors)?;
        writeln!(f, "  Mutation score:   {:.1}%", self.mutation_score())?;
        writeln!(f, "  Test strength:    {:.1}%", self.test_strength())?;
        writeln!(f, "  Killing tests:    {}", self.killing_tests.len())?;
        writeln!(f, "  Mutator types:    {}", self.mutator_count())?;
        write!(f, "  Classes mutated:  {}", self.class_count())
    }
}

// ---------------------------------------------------------------------------
// Utility: group mutations by class
// ---------------------------------------------------------------------------

/// Groups mutations by their fully-qualified class name.
pub fn group_by_class(mutations: &[PitMutation]) -> HashMap<String, Vec<&PitMutation>> {
    let mut map: HashMap<String, Vec<&PitMutation>> = HashMap::new();
    for m in mutations {
        map.entry(m.mutated_class.clone()).or_default().push(m);
    }
    map
}

/// Groups mutations by their mutator short name.
pub fn group_by_mutator(mutations: &[PitMutation]) -> HashMap<String, Vec<&PitMutation>> {
    let mut map: HashMap<String, Vec<&PitMutation>> = HashMap::new();
    for m in mutations {
        map.entry(m.short_mutator_name().to_string())
            .or_default()
            .push(m);
    }
    map
}

/// Groups mutations by source file name.
pub fn group_by_source_file(mutations: &[PitMutation]) -> HashMap<String, Vec<&PitMutation>> {
    let mut map: HashMap<String, Vec<&PitMutation>> = HashMap::new();
    for m in mutations {
        map.entry(m.source_file.clone()).or_default().push(m);
    }
    map
}

/// Filter mutations to only those that survived.
pub fn surviving_mutations(mutations: &[PitMutation]) -> Vec<&PitMutation> {
    mutations
        .iter()
        .filter(|m| m.status.is_survived())
        .collect()
}

/// Filter mutations to only those that were detected (killed / timed-out).
pub fn detected_mutations(mutations: &[PitMutation]) -> Vec<&PitMutation> {
    mutations
        .iter()
        .filter(|m| m.status.is_detected())
        .collect()
}

/// Read and parse a PIT XML report from the given file path.
///
/// Convenience wrapper around [`PitXmlParser`].
pub fn parse_pit_xml(path: impl AsRef<Path>) -> PitResult<Vec<PitMutation>> {
    PitXmlParser::new(path.as_ref()).parse()
}

/// Read and parse a PIT CSV kill matrix from the given file path.
///
/// Convenience wrapper around [`PitCsvParser`].
pub fn parse_pit_csv(path: impl AsRef<Path>) -> PitResult<Vec<CsvKillEntry>> {
    PitCsvParser::new(path.as_ref()).parse()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<mutations>
  <mutation detected="true" status="KILLED" numberOfTestsRun="5">
    <sourceFile>Calculator.java</sourceFile>
    <mutatedClass>com.example.Calculator</mutatedClass>
    <mutatedMethod>add</mutatedMethod>
    <methodDescription>(II)I</methodDescription>
    <lineNumber>42</lineNumber>
    <mutator>org.pitest.mutationtest.engine.gregor.mutators.MathMutator</mutator>
    <indexes><index>12</index></indexes>
    <blocks><block>3</block></blocks>
    <killingTest>com.example.CalculatorTest::testAdd</killingTest>
    <description>replaced int return with 0</description>
  </mutation>
  <mutation detected="false" status="SURVIVED" numberOfTestsRun="3">
    <sourceFile>Calculator.java</sourceFile>
    <mutatedClass>com.example.Calculator</mutatedClass>
    <mutatedMethod>subtract</mutatedMethod>
    <methodDescription>(II)I</methodDescription>
    <lineNumber>56</lineNumber>
    <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
    <indexes><index>8</index></indexes>
    <blocks><block>1</block></blocks>
    <killingTest></killingTest>
    <description>negated conditional</description>
  </mutation>
  <mutation detected="true" status="TIMED_OUT" numberOfTestsRun="2">
    <sourceFile>Sorter.java</sourceFile>
    <mutatedClass>com.example.Sorter</mutatedClass>
    <mutatedMethod>sort</mutatedMethod>
    <methodDescription>([I)V</methodDescription>
    <lineNumber>101</lineNumber>
    <mutator>org.pitest.mutationtest.engine.gregor.mutators.IncrementsMutator</mutator>
    <indexes><index>20</index><index>21</index></indexes>
    <blocks><block>5</block><block>6</block></blocks>
    <killingTest>com.example.SorterTest::testSort</killingTest>
    <description>Changed increment</description>
  </mutation>
</mutations>"#;

    const SAMPLE_CSV: &str = "\
mutant_id,mutated_class,mutated_method,line_number,mutator,killing_test,status
1,com.example.Calculator,add,42,MathMutator,com.example.CalculatorTest::testAdd,KILLED
2,com.example.Calculator,subtract,56,NegateConditionalsMutator,,SURVIVED
3,com.example.Sorter,sort,101,IncrementsMutator,com.example.SorterTest::testSort,TIMED_OUT
";

    #[test]
    fn test_detection_status_from_string() {
        assert_eq!(
            PitDetectionStatus::from_pit_string("KILLED").unwrap(),
            PitDetectionStatus::Killed
        );
        assert_eq!(
            PitDetectionStatus::from_pit_string("survived").unwrap(),
            PitDetectionStatus::Survived
        );
        assert_eq!(
            PitDetectionStatus::from_pit_string("TIMED_OUT").unwrap(),
            PitDetectionStatus::TimedOut
        );
        assert_eq!(
            PitDetectionStatus::from_pit_string("NO_COVERAGE").unwrap(),
            PitDetectionStatus::NoCoverage
        );
        assert!(PitDetectionStatus::from_pit_string("BOGUS").is_err());
    }

    #[test]
    fn test_detection_status_display() {
        assert_eq!(PitDetectionStatus::Killed.to_string(), "KILLED");
        assert_eq!(PitDetectionStatus::TimedOut.to_string(), "TIMED_OUT");
    }

    #[test]
    fn test_detection_status_predicates() {
        assert!(PitDetectionStatus::Killed.is_detected());
        assert!(PitDetectionStatus::TimedOut.is_detected());
        assert!(!PitDetectionStatus::Survived.is_detected());
        assert!(PitDetectionStatus::Survived.is_survived());
        assert!(PitDetectionStatus::NonViable.is_error());
        assert!(PitDetectionStatus::MemoryError.is_error());
        assert!(!PitDetectionStatus::Killed.is_error());
    }

    #[test]
    fn test_parse_xml_mutations() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        assert_eq!(mutations.len(), 3);

        let m0 = &mutations[0];
        assert!(m0.detected);
        assert_eq!(m0.status, PitDetectionStatus::Killed);
        assert_eq!(m0.mutated_class, "com.example.Calculator");
        assert_eq!(m0.mutated_method, "add");
        assert_eq!(m0.method_description, "(II)I");
        assert_eq!(m0.line_number, 42);
        assert_eq!(m0.source_file, "Calculator.java");
        assert_eq!(
            m0.killing_test.as_deref(),
            Some("com.example.CalculatorTest::testAdd")
        );
        assert_eq!(m0.indexes, vec![12]);
        assert_eq!(m0.blocks, vec![3]);
        assert_eq!(m0.number_of_tests_run, 5);

        let m1 = &mutations[1];
        assert!(!m1.detected);
        assert_eq!(m1.status, PitDetectionStatus::Survived);
        assert!(m1.killing_test.is_none());
        assert_eq!(m1.line_number, 56);

        let m2 = &mutations[2];
        assert_eq!(m2.status, PitDetectionStatus::TimedOut);
        assert_eq!(m2.indexes, vec![20, 21]);
        assert_eq!(m2.blocks, vec![5, 6]);
    }

    #[test]
    fn test_pit_mutation_helpers() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let m = &mutations[0];

        assert_eq!(m.simple_class_name(), "Calculator");
        assert_eq!(m.package(), Some("com.example"));
        assert_eq!(m.short_mutator_name(), "MathMutator");
        assert!(m.canonical_key().contains("Calculator"));
        assert!(m.canonical_key().contains("add"));
    }

    #[test]
    fn test_pit_mutation_display() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let display = mutations[0].to_string();
        assert!(display.contains("KILLED"));
        assert!(display.contains("Calculator"));
        assert!(display.contains("add"));
    }

    #[test]
    fn test_parse_csv_entries() {
        let mut parser = PitCsvParser::new("test.csv");
        let entries = parser.parse_str(SAMPLE_CSV).unwrap();
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].mutant_id, "1");
        assert_eq!(entries[0].mutated_class, "com.example.Calculator");
        assert_eq!(entries[0].status, PitDetectionStatus::Killed);
        assert_eq!(
            entries[0].killing_test.as_deref(),
            Some("com.example.CalculatorTest::testAdd")
        );

        assert_eq!(entries[1].status, PitDetectionStatus::Survived);
        assert!(entries[1].killing_test.is_none());

        assert_eq!(entries[2].status, PitDetectionStatus::TimedOut);
    }

    #[test]
    fn test_report_summary() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let summary = PitReportSummary::from_mutations(&mutations);

        assert_eq!(summary.total_mutations, 3);
        assert_eq!(summary.detected, 2);
        assert_eq!(summary.survived, 1);
        assert_eq!(summary.killing_tests.len(), 2);
        assert!(summary.mutation_score() > 60.0);
    }

    #[test]
    fn test_report_summary_display() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let summary = PitReportSummary::from_mutations(&mutations);
        let display = summary.to_string();
        assert!(display.contains("PIT Report Summary"));
        assert!(display.contains("Total mutations"));
    }

    #[test]
    fn test_group_by_class() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let groups = group_by_class(&mutations);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["com.example.Calculator"].len(), 2);
        assert_eq!(groups["com.example.Sorter"].len(), 1);
    }

    #[test]
    fn test_group_by_mutator() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let groups = group_by_mutator(&mutations);
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_group_by_source_file() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let groups = group_by_source_file(&mutations);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["Calculator.java"].len(), 2);
    }

    #[test]
    fn test_surviving_mutations() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let survived = surviving_mutations(&mutations);
        assert_eq!(survived.len(), 1);
        assert_eq!(survived[0].mutated_method, "subtract");
    }

    #[test]
    fn test_detected_mutations() {
        let mut parser = PitXmlParser::new("test.xml");
        let mutations = parser.parse_str(SAMPLE_XML).unwrap();
        let detected = detected_mutations(&mutations);
        assert_eq!(detected.len(), 2);
    }

    #[test]
    fn test_empty_xml() {
        let mut parser = PitXmlParser::new("empty.xml");
        let mutations = parser.parse_str("<mutations></mutations>").unwrap();
        assert!(mutations.is_empty());
    }

    #[test]
    fn test_empty_csv() {
        let mut parser = PitCsvParser::new("empty.csv");
        let entries = parser
            .parse_str(
                "mutant_id,mutated_class,mutated_method,line_number,mutator,killing_test,status\n",
            )
            .unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_summary_zero_mutations() {
        let summary = PitReportSummary::from_mutations(&[]);
        assert_eq!(summary.mutation_score(), 0.0);
        assert_eq!(summary.test_strength(), 0.0);
    }

    #[test]
    fn test_csv_entry_display() {
        let entry = CsvKillEntry {
            mutant_id: "1".into(),
            mutated_class: "com.example.Calc".into(),
            mutated_method: "add".into(),
            line_number: 42,
            mutator: "MathMutator".into(),
            killing_test: Some("testAdd".into()),
            status: PitDetectionStatus::Killed,
        };
        let s = entry.to_string();
        assert!(s.contains("KILLED"));
        assert!(s.contains("Calc"));
    }

    #[test]
    fn test_lenient_parsing_unknown_status() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mutations>
  <mutation detected="true" status="KILLED" numberOfTestsRun="1">
    <sourceFile>A.java</sourceFile>
    <mutatedClass>com.A</mutatedClass>
    <mutatedMethod>foo</mutatedMethod>
    <lineNumber>10</lineNumber>
    <mutator>SomeMutator</mutator>
  </mutation>
  <mutation detected="false" status="WEIRD_STATUS" numberOfTestsRun="1">
    <sourceFile>B.java</sourceFile>
    <mutatedClass>com.B</mutatedClass>
    <mutatedMethod>bar</mutatedMethod>
    <lineNumber>20</lineNumber>
    <mutator>OtherMutator</mutator>
  </mutation>
</mutations>"#;

        let mut parser = PitXmlParser::new("test.xml").lenient();
        let mutations = parser.parse_str(xml).unwrap();
        assert_eq!(mutations.len(), 1);
        assert_eq!(parser.warnings().len(), 1);
    }

    #[test]
    fn test_extract_attribute_single_quotes() {
        let parser = PitXmlParser::new("test.xml");
        let element = "<mutation detected='true' status='KILLED'>";
        let val = parser.extract_attribute(element, "detected");
        assert_eq!(val.as_deref(), Some("true"));
    }
}
