//! Diagnostic engine: error formatting, colour-coded output, and
//! machine-readable export (JSON / SARIF).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

// ── Severity ────────────────────────────────────────────────────────────────

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Hint,
    Info,
    Warning,
    Error,
}

impl Severity {
    /// ANSI colour code for terminal output.
    pub fn colour_code(self) -> &'static str {
        match self {
            Severity::Error => "\x1b[1;31m",   // bold red
            Severity::Warning => "\x1b[1;33m", // bold yellow
            Severity::Info => "\x1b[1;34m",    // bold blue
            Severity::Hint => "\x1b[1;36m",    // bold cyan
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
            Severity::Hint => "hint",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ── Source location ─────────────────────────────────────────────────────────

/// A location inside a source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
    pub end_line: Option<usize>,
    pub end_column: Option<usize>,
}

impl SourceLocation {
    pub fn new(file: impl Into<PathBuf>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            end_line: None,
            end_column: None,
        }
    }

    pub fn with_end(mut self, end_line: usize, end_column: usize) -> Self {
        self.end_line = Some(end_line);
        self.end_column = Some(end_column);
        self
    }

    /// Is this a multi-line span?
    pub fn is_multiline(&self) -> bool {
        self.end_line.map_or(false, |el| el > self.line)
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file.display(), self.line, self.column)
    }
}

// ── Diagnostic ──────────────────────────────────────────────────────────────

/// A single diagnostic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub location: Option<SourceLocation>,
    pub notes: Vec<String>,
    pub suggestion: Option<String>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: None,
            message: message.into(),
            location: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            code: None,
            message: message.into(),
            location: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            code: None,
            message: message.into(),
            location: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    pub fn hint(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Hint,
            code: None,
            message: message.into(),
            location: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.location = Some(loc);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

// ── Diagnostic format enum ──────────────────────────────────────────────────

/// Which output format to use for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticFormat {
    Plain,
    Json,
    Sarif,
}

// ── Diagnostic engine ───────────────────────────────────────────────────────

/// Collects diagnostics and formats them for the terminal or machine consumption.
#[derive(Debug)]
pub struct DiagnosticEngine {
    format: DiagnosticFormat,
    diagnostics: Vec<Diagnostic>,
    use_colour: bool,
}

impl DiagnosticEngine {
    pub fn new(format: DiagnosticFormat) -> Self {
        let use_colour = format == DiagnosticFormat::Plain && atty_stdout();
        Self {
            format,
            diagnostics: Vec::new(),
            use_colour,
        }
    }

    /// Force colour on/off regardless of terminal detection.
    pub fn set_colour(&mut self, on: bool) {
        self.use_colour = on;
    }

    /// Add a diagnostic.
    pub fn emit(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Number of errors emitted so far.
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    /// Number of warnings emitted so far.
    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }

    /// Whether any errors have been emitted.
    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }

    /// All collected diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Clear all diagnostics.
    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    /// Render all diagnostics to a string using the configured format.
    pub fn render_all(&self, sources: &SourceCache) -> String {
        match self.format {
            DiagnosticFormat::Plain => self.render_plain(sources),
            DiagnosticFormat::Json => self.render_json(),
            DiagnosticFormat::Sarif => self.render_sarif(),
        }
    }

    /// Render a summary line: "N error(s), M warning(s)".
    pub fn summary(&self) -> String {
        let e = self.error_count();
        let w = self.warning_count();
        format!(
            "{} error{}, {} warning{}",
            e,
            if e == 1 { "" } else { "s" },
            w,
            if w == 1 { "" } else { "s" },
        )
    }

    // ── Plain text rendering ──────────────────────────────────

    fn render_plain(&self, sources: &SourceCache) -> String {
        let mut out = String::new();
        for diag in &self.diagnostics {
            out.push_str(&self.format_plain_diagnostic(diag, sources));
            out.push('\n');
        }
        out
    }

    fn format_plain_diagnostic(&self, diag: &Diagnostic, sources: &SourceCache) -> String {
        let reset = if self.use_colour { "\x1b[0m" } else { "" };
        let colour = if self.use_colour {
            diag.severity.colour_code()
        } else {
            ""
        };

        let mut out = String::new();

        // Location prefix.
        if let Some(ref loc) = diag.location {
            out.push_str(&format!("{}: ", loc));
        }

        // Severity + code + message.
        let code_str = diag
            .code
            .as_ref()
            .map(|c| format!("[{}] ", c))
            .unwrap_or_default();
        out.push_str(&format!(
            "{}{}{}{}: {}\n",
            colour,
            diag.severity.label(),
            reset,
            code_str,
            diag.message
        ));

        // Source snippet.
        if let Some(ref loc) = diag.location {
            if let Some(snippet) = sources.snippet(loc) {
                out.push_str(&snippet);
            }
        }

        // Notes.
        for note in &diag.notes {
            out.push_str(&format!("  = note: {}\n", note));
        }

        // Suggestion.
        if let Some(ref sug) = diag.suggestion {
            out.push_str(&format!(
                "  = {}suggestion{}: {}\n",
                if self.use_colour { "\x1b[1;32m" } else { "" },
                reset,
                sug
            ));
        }

        out
    }

    // ── JSON rendering ────────────────────────────────────────

    fn render_json(&self) -> String {
        serde_json::to_string_pretty(&self.diagnostics).unwrap_or_default()
    }

    // ── SARIF rendering ───────────────────────────────────────

    fn render_sarif(&self) -> String {
        let results: Vec<serde_json::Value> = self
            .diagnostics
            .iter()
            .map(|d| {
                let mut result = serde_json::json!({
                    "ruleId": d.code.as_deref().unwrap_or("unknown"),
                    "level": match d.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                        Severity::Info => "note",
                        Severity::Hint => "note",
                    },
                    "message": { "text": d.message },
                });
                if let Some(ref loc) = d.location {
                    result["locations"] = serde_json::json!([{
                        "physicalLocation": {
                            "artifactLocation": { "uri": loc.file.display().to_string() },
                            "region": {
                                "startLine": loc.line,
                                "startColumn": loc.column,
                                "endLine": loc.end_line.unwrap_or(loc.line),
                                "endColumn": loc.end_column.unwrap_or(loc.column),
                            }
                        }
                    }]);
                }
                result
            })
            .collect();

        let sarif = serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "sonitype",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                },
                "results": results,
            }]
        });
        serde_json::to_string_pretty(&sarif).unwrap_or_default()
    }
}

// ── Source cache ─────────────────────────────────────────────────────────────

/// Cache of source files for snippet rendering.
#[derive(Debug, Default)]
pub struct SourceCache {
    files: Vec<(PathBuf, String)>,
}

impl SourceCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a source file to the cache.
    pub fn add(&mut self, path: impl Into<PathBuf>, source: impl Into<String>) {
        self.files.push((path.into(), source.into()));
    }

    /// Load a source file from disk into the cache.
    pub fn load(&mut self, path: &Path) -> std::io::Result<()> {
        let source = std::fs::read_to_string(path)?;
        self.files.push((path.to_path_buf(), source));
        Ok(())
    }

    /// Lookup source text for a path.
    pub fn get(&self, path: &Path) -> Option<&str> {
        self.files
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, s)| s.as_str())
    }

    /// Render a source snippet with a caret for a given location.
    pub fn snippet(&self, loc: &SourceLocation) -> Option<String> {
        let source = self.get(&loc.file)?;
        let lines: Vec<&str> = source.lines().collect();
        if loc.line == 0 || loc.line > lines.len() {
            return None;
        }

        let mut out = String::new();
        let line_num_width = format!("{}", loc.line).len().max(4);

        if loc.is_multiline() {
            let end_line = loc.end_line.unwrap_or(loc.line).min(lines.len());
            for ln in loc.line..=end_line {
                let text = lines.get(ln - 1).unwrap_or(&"");
                out.push_str(&format!("{:>width$} │ {}\n", ln, text, width = line_num_width));
            }
        } else {
            let text = lines[loc.line - 1];
            out.push_str(&format!(
                "{:>width$} │ {}\n",
                loc.line,
                text,
                width = line_num_width
            ));
            // Caret line.
            let col = loc.column.saturating_sub(1);
            let end_col = loc
                .end_column
                .unwrap_or(loc.column + 1)
                .saturating_sub(1)
                .max(col + 1);
            let span_len = end_col - col;
            out.push_str(&format!(
                "{:>width$} │ {}{}\n",
                "",
                " ".repeat(col),
                "^".repeat(span_len),
                width = line_num_width
            ));
        }

        Some(out)
    }
}

// ── Error Explainer ─────────────────────────────────────────────────────────

/// Provides detailed explanations for well-known error codes.
pub struct ErrorExplainer;

impl ErrorExplainer {
    /// Return a detailed explanation and suggested fix for an error code.
    pub fn explain(code: &str) -> Option<ErrorExplanation> {
        ERROR_EXPLANATIONS
            .iter()
            .find(|e| e.code == code)
            .cloned()
    }

    /// List all known error codes.
    pub fn all_codes() -> &'static [ErrorExplanation] {
        ERROR_EXPLANATIONS
    }
}

/// A detailed explanation for a specific error code.
#[derive(Debug, Clone)]
pub struct ErrorExplanation {
    pub code: &'static str,
    pub title: &'static str,
    pub explanation: &'static str,
    pub suggestion: &'static str,
    pub doc_url: &'static str,
}

static ERROR_EXPLANATIONS: &[ErrorExplanation] = &[
    ErrorExplanation {
        code: "E0001",
        title: "Syntax error",
        explanation: "The DSL source contains a syntax error that prevents parsing. \
                      Check for missing braces, unmatched parentheses, or invalid tokens.",
        suggestion: "Review the indicated line and ensure proper DSL syntax.",
        doc_url: "https://sonitype.dev/errors/E0001",
    },
    ErrorExplanation {
        code: "E0002",
        title: "Unknown identifier",
        explanation: "A name was used that has not been defined in the current scope. \
                      This can happen if a variable, stream, or mapping is referenced \
                      before it is declared.",
        suggestion: "Declare the identifier before use, or check for typos.",
        doc_url: "https://sonitype.dev/errors/E0002",
    },
    ErrorExplanation {
        code: "E0003",
        title: "Type mismatch",
        explanation: "The types of two expressions are incompatible. SoniType uses a \
                      perceptual type system that includes psychoacoustic qualifiers; \
                      ensure that Bark bands, cognitive loads, and JNDs are compatible.",
        suggestion: "Add explicit type annotations or restructure the mapping.",
        doc_url: "https://sonitype.dev/errors/E0003",
    },
    ErrorExplanation {
        code: "E0004",
        title: "Bark band conflict",
        explanation: "Two or more streams are mapped to overlapping Bark critical bands, \
                      which may cause auditory masking and information loss.",
        suggestion: "Separate the streams by at least one critical band, or reduce their \
                     temporal overlap.",
        doc_url: "https://sonitype.dev/errors/E0004",
    },
    ErrorExplanation {
        code: "E0005",
        title: "Cognitive load exceeded",
        explanation: "The number of simultaneous auditory streams exceeds the configured \
                      cognitive load budget (default: 4, based on Cowan's 4±1 rule).",
        suggestion: "Reduce the number of simultaneous streams, or increase the budget.",
        doc_url: "https://sonitype.dev/errors/E0005",
    },
    ErrorExplanation {
        code: "E0006",
        title: "JND violation",
        explanation: "A parameter mapping produces differences below the Just Noticeable \
                      Difference threshold. Listeners will be unable to perceive the \
                      intended data variation.",
        suggestion: "Increase the parameter range or use a more discriminable dimension.",
        doc_url: "https://sonitype.dev/errors/E0006",
    },
    ErrorExplanation {
        code: "E0007",
        title: "WCET budget exceeded",
        explanation: "The estimated worst-case execution time for one audio buffer exceeds \
                      the real-time deadline. The sonification will not render in real time.",
        suggestion: "Simplify the audio graph, reduce the number of nodes, or increase \
                     the buffer size.",
        doc_url: "https://sonitype.dev/errors/E0007",
    },
    ErrorExplanation {
        code: "E0008",
        title: "Masking warning",
        explanation: "A stream may be partially masked by another nearby stream. While not \
                      an error, this can reduce perceptual clarity.",
        suggestion: "Increase spectral separation or adjust amplitudes.",
        doc_url: "https://sonitype.dev/errors/E0008",
    },
    ErrorExplanation {
        code: "E0009",
        title: "Unused stream",
        explanation: "A stream is declared but never referenced in a composition or output.",
        suggestion: "Remove the unused stream or connect it to an output.",
        doc_url: "https://sonitype.dev/errors/E0009",
    },
    ErrorExplanation {
        code: "E0010",
        title: "Invalid hearing profile",
        explanation: "The specified hearing profile name is not recognised, or the custom \
                      audiogram data is malformed.",
        suggestion: "Use a built-in profile (e.g. 'normal', 'mild_loss') or provide valid \
                     frequency/threshold pairs.",
        doc_url: "https://sonitype.dev/errors/E0010",
    },
    ErrorExplanation {
        code: "E0011",
        title: "Duplicate identifier",
        explanation: "A stream, mapping, or variable name is defined more than once in the \
                      same scope.",
        suggestion: "Rename one of the conflicting definitions.",
        doc_url: "https://sonitype.dev/errors/E0011",
    },
    ErrorExplanation {
        code: "E0012",
        title: "Invalid parameter range",
        explanation: "A mapping parameter has a range that is invalid, e.g. min > max, \
                      or a frequency outside the audible range (20 Hz–20 kHz).",
        suggestion: "Correct the parameter bounds.",
        doc_url: "https://sonitype.dev/errors/E0012",
    },
];

// ── Diagnostic Formatter (thin wrappers) ────────────────────────────────────

/// Convenience wrapper that formats a single diagnostic in a given format.
pub struct DiagnosticFormatter;

impl DiagnosticFormatter {
    pub fn format_plain(diag: &Diagnostic, sources: &SourceCache) -> String {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Plain);
        engine.set_colour(false);
        engine.emit(diag.clone());
        engine.render_all(sources)
    }

    pub fn format_json(diag: &Diagnostic) -> String {
        serde_json::to_string_pretty(diag).unwrap_or_default()
    }

    pub fn format_sarif(diags: &[Diagnostic]) -> String {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Sarif);
        for d in diags {
            engine.emit(d.clone());
        }
        engine.render_all(&SourceCache::new())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Best-effort TTY detection (does not depend on the `atty` crate).
fn atty_stdout() -> bool {
    // On Unix we can check via libc; fallback to assuming colour is fine.
    #[cfg(unix)]
    {
        unsafe { libc_isatty(1) != 0 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

#[cfg(unix)]
extern "C" {
    #[link_name = "isatty"]
    fn libc_isatty(fd: i32) -> i32;
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_source() -> SourceCache {
        let mut cache = SourceCache::new();
        cache.add(
            "test.soni",
            "stream temperature {\n  pitch: 200..800 Hz\n  pan: -1..1\n}\n",
        );
        cache
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Hint < Severity::Info);
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Error.to_string(), "error");
        assert_eq!(Severity::Warning.to_string(), "warning");
    }

    #[test]
    fn source_location_display() {
        let loc = SourceLocation::new("test.soni", 3, 5);
        assert_eq!(loc.to_string(), "test.soni:3:5");
    }

    #[test]
    fn source_location_multiline() {
        let loc = SourceLocation::new("test.soni", 1, 1).with_end(3, 5);
        assert!(loc.is_multiline());
    }

    #[test]
    fn diagnostic_builder() {
        let d = Diagnostic::error("bad thing")
            .with_code("E0001")
            .with_location(SourceLocation::new("x.soni", 1, 1))
            .with_note("see docs")
            .with_suggestion("try this");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.code.as_deref(), Some("E0001"));
        assert_eq!(d.notes.len(), 1);
        assert!(d.suggestion.is_some());
    }

    #[test]
    fn engine_error_count() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Plain);
        engine.emit(Diagnostic::error("e1"));
        engine.emit(Diagnostic::warning("w1"));
        engine.emit(Diagnostic::error("e2"));
        assert_eq!(engine.error_count(), 2);
        assert_eq!(engine.warning_count(), 1);
        assert!(engine.has_errors());
    }

    #[test]
    fn engine_summary() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Plain);
        engine.emit(Diagnostic::error("e"));
        assert_eq!(engine.summary(), "1 error, 0 warnings");
    }

    #[test]
    fn engine_clear() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Plain);
        engine.emit(Diagnostic::error("e"));
        engine.clear();
        assert_eq!(engine.error_count(), 0);
    }

    #[test]
    fn plain_rendering_contains_message() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Plain);
        engine.set_colour(false);
        engine.emit(
            Diagnostic::error("missing semicolon")
                .with_location(SourceLocation::new("test.soni", 2, 8)),
        );
        let out = engine.render_all(&sample_source());
        assert!(out.contains("missing semicolon"));
        assert!(out.contains("test.soni:2:8"));
    }

    #[test]
    fn json_rendering_is_valid_json() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Json);
        engine.emit(Diagnostic::warning("w").with_code("E0008"));
        let out = engine.render_all(&SourceCache::new());
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn sarif_rendering_has_schema() {
        let mut engine = DiagnosticEngine::new(DiagnosticFormat::Sarif);
        engine.emit(Diagnostic::error("e").with_code("E0001"));
        let out = engine.render_all(&SourceCache::new());
        assert!(out.contains("sarif-schema"));
    }

    #[test]
    fn snippet_single_line() {
        let cache = sample_source();
        let loc = SourceLocation::new("test.soni", 2, 3).with_end(2, 8);
        let snip = cache.snippet(&loc).unwrap();
        assert!(snip.contains("pitch"));
        assert!(snip.contains("^"));
    }

    #[test]
    fn snippet_multiline() {
        let cache = sample_source();
        let loc = SourceLocation::new("test.soni", 1, 1).with_end(3, 10);
        let snip = cache.snippet(&loc).unwrap();
        assert!(snip.contains("stream"));
        assert!(snip.contains("pan"));
    }

    #[test]
    fn error_explainer_known_code() {
        let expl = ErrorExplainer::explain("E0001").unwrap();
        assert_eq!(expl.title, "Syntax error");
        assert!(!expl.explanation.is_empty());
    }

    #[test]
    fn error_explainer_unknown_code() {
        assert!(ErrorExplainer::explain("E9999").is_none());
    }

    #[test]
    fn error_explainer_all_codes() {
        let all = ErrorExplainer::all_codes();
        assert!(all.len() >= 10);
    }

    #[test]
    fn diagnostic_formatter_plain() {
        let d = Diagnostic::error("oops");
        let out = DiagnosticFormatter::format_plain(&d, &SourceCache::new());
        assert!(out.contains("oops"));
    }

    #[test]
    fn diagnostic_formatter_json() {
        let d = Diagnostic::warning("warn");
        let out = DiagnosticFormatter::format_json(&d);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["message"], "warn");
    }

    #[test]
    fn snippet_out_of_bounds_returns_none() {
        let cache = sample_source();
        let loc = SourceLocation::new("test.soni", 999, 1);
        assert!(cache.snippet(&loc).is_none());
    }

    #[test]
    fn snippet_unknown_file_returns_none() {
        let cache = sample_source();
        let loc = SourceLocation::new("unknown.soni", 1, 1);
        assert!(cache.snippet(&loc).is_none());
    }
}
