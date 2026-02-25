//! Structured logging module for the CABER (Coalgebraic Auditing of Behavioral
//! Equivalence Relations) project. Provides audit logging, performance timing,
//! query tracking, and append-only audit trails with JSON serialization support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity levels for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl LogLevel {
    /// Returns a numeric severity useful for filtering and ordering.
    pub fn severity(&self) -> u8 {
        match self {
            LogLevel::Debug => 0,
            LogLevel::Info => 1,
            LogLevel::Warn => 2,
            LogLevel::Error => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// LogEntry
// ---------------------------------------------------------------------------

/// A single structured log entry carrying a timestamp, severity, originating
/// component, human-readable message, and arbitrary key-value context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub component: String,
    pub message: String,
    pub context: HashMap<String, String>,
}

impl LogEntry {
    /// Creates a new `LogEntry` with the current UTC timestamp in ISO 8601.
    pub fn new(level: LogLevel, component: &str, message: &str) -> Self {
        Self {
            timestamp: iso8601_now(),
            level,
            component: component.to_string(),
            message: message.to_string(),
            context: HashMap::new(),
        }
    }

    /// Attaches a key-value pair of contextual metadata to this entry.
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Serializes the entry to a compact JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| format!("{{\"error\":\"serialization failed for: {}\"}}", self.message))
    }
}

impl std::fmt::Display for LogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} [{}] {}",
            self.timestamp, self.level, self.component, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// AuditLogger
// ---------------------------------------------------------------------------

/// Accumulates `LogEntry` items produced during an audit session.  All entries
/// are kept in memory so that they can be inspected, filtered, or serialized
/// after the audit completes.
#[derive(Debug, Clone)]
pub struct AuditLogger {
    component: String,
    entries: Vec<LogEntry>,
}

impl AuditLogger {
    /// Creates a new logger scoped to the given component name.
    pub fn new(component: &str) -> Self {
        Self {
            component: component.to_string(),
            entries: Vec::new(),
        }
    }

    // -- convenience methods for each severity level -------------------------

    /// Logs a message at `Info` level.
    pub fn log_info(&mut self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Logs a message at `Warn` level.
    pub fn log_warn(&mut self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Logs a message at `Error` level.
    pub fn log_error(&mut self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    /// Logs a message at `Debug` level.
    pub fn log_debug(&mut self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    // -- querying ------------------------------------------------------------

    /// Returns a slice over every entry recorded so far.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Returns references to entries that match the requested `level`.
    pub fn entries_at_level(&self, level: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level == level).collect()
    }

    /// Returns the total number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns entries whose severity is at least `min_level`.
    pub fn entries_at_or_above(&self, min_level: LogLevel) -> Vec<&LogEntry> {
        let threshold = min_level.severity();
        self.entries
            .iter()
            .filter(|e| e.level.severity() >= threshold)
            .collect()
    }

    /// Logs a message with extra contextual key-value pairs.
    pub fn log_with_context(
        &mut self,
        level: LogLevel,
        message: &str,
        context: HashMap<String, String>,
    ) {
        let mut entry = LogEntry::new(level, &self.component, message);
        entry.context = context;
        self.entries.push(entry);
    }

    /// Serializes the full log to a JSON array string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.entries)
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Clears all recorded entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    // -- internal ------------------------------------------------------------

    fn log(&mut self, level: LogLevel, message: &str) {
        let entry = LogEntry::new(level, &self.component, message);
        self.entries.push(entry);
    }
}

// ---------------------------------------------------------------------------
// PerformanceRecord
// ---------------------------------------------------------------------------

/// The result of a completed performance measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub label: String,
    pub duration_ms: f64,
    pub timestamp: String,
}

impl PerformanceRecord {
    /// Serializes to a compact JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

impl std::fmt::Display for PerformanceRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} completed in {:.3}ms",
            self.timestamp, self.label, self.duration_ms
        )
    }
}

// ---------------------------------------------------------------------------
// PerformanceTimer
// ---------------------------------------------------------------------------

/// A lightweight wall-clock timer that measures elapsed time between
/// construction (`start`) and consumption (`stop`).
#[derive(Debug)]
pub struct PerformanceTimer {
    label: String,
    start_instant: Instant,
    start_timestamp: String,
}

impl PerformanceTimer {
    /// Starts the timer with the given human-readable label.
    pub fn start(label: &str) -> Self {
        Self {
            label: label.to_string(),
            start_instant: Instant::now(),
            start_timestamp: iso8601_now(),
        }
    }

    /// Returns the number of milliseconds elapsed since the timer was started,
    /// without consuming the timer.
    pub fn elapsed_ms(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64() * 1000.0
    }

    /// Consumes the timer and returns a `PerformanceRecord` capturing the
    /// label, total duration, and the timestamp at which timing began.
    pub fn stop(self) -> PerformanceRecord {
        let duration_ms = self.start_instant.elapsed().as_secs_f64() * 1000.0;
        PerformanceRecord {
            label: self.label,
            duration_ms,
            timestamp: self.start_timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryRecord  (internal to QueryLogger)
// ---------------------------------------------------------------------------

/// A single recorded query made during an audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    pub query_type: String,
    pub input: String,
    pub output: String,
    pub latency_ms: f64,
    pub timestamp: String,
}

// ---------------------------------------------------------------------------
// QueryLogger
// ---------------------------------------------------------------------------

/// Tracks every query issued during an audit, providing aggregate statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLogger {
    queries: Vec<QueryRecord>,
}

impl QueryLogger {
    /// Creates an empty query log.
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    /// Records a single query with its type label, raw input, output, and
    /// measured latency in milliseconds.
    pub fn log_query(
        &mut self,
        query_type: &str,
        input: &str,
        output: &str,
        latency_ms: f64,
    ) {
        self.queries.push(QueryRecord {
            query_type: query_type.to_string(),
            input: input.to_string(),
            output: output.to_string(),
            latency_ms,
            timestamp: iso8601_now(),
        });
    }

    /// Total number of queries logged so far.
    pub fn total_queries(&self) -> usize {
        self.queries.len()
    }

    /// Arithmetic mean of all recorded latencies, or `0.0` when the log is
    /// empty.
    pub fn average_latency(&self) -> f64 {
        if self.queries.is_empty() {
            return 0.0;
        }
        let total: f64 = self.queries.iter().map(|q| q.latency_ms).sum();
        total / self.queries.len() as f64
    }

    /// Returns a map from query type to the number of queries of that type.
    pub fn queries_by_type(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for q in &self.queries {
            *counts.entry(q.query_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the maximum latency observed, or `0.0` if no queries exist.
    pub fn max_latency(&self) -> f64 {
        self.queries
            .iter()
            .map(|q| q.latency_ms)
            .fold(0.0_f64, f64::max)
    }

    /// Returns the minimum latency observed, or `0.0` if no queries exist.
    pub fn min_latency(&self) -> f64 {
        if self.queries.is_empty() {
            return 0.0;
        }
        self.queries
            .iter()
            .map(|q| q.latency_ms)
            .fold(f64::INFINITY, f64::min)
    }

    /// Serializes the full query log as a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for QueryLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AuditAction  (internal to AuditTrail)
// ---------------------------------------------------------------------------

/// A single entry in the append-only audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAction {
    pub timestamp: String,
    pub kind: String,
    pub description: String,
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// AuditTrail
// ---------------------------------------------------------------------------

/// An append-only sequence of actions and results captured over the lifetime
/// of an audit.  Entries cannot be modified or removed once recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    actions: Vec<AuditAction>,
}

impl AuditTrail {
    /// Creates a new empty audit trail.
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }

    /// Records a generic action with a short label and a longer description.
    pub fn record_action(&mut self, action: &str, details: &str) {
        self.actions.push(AuditAction {
            timestamp: iso8601_now(),
            kind: action.to_string(),
            description: details.to_string(),
            metadata: HashMap::new(),
        });
    }

    /// Records the result of a property check with its confidence score.
    pub fn record_result(&mut self, property: &str, result: &str, confidence: f64) {
        let mut metadata = HashMap::new();
        metadata.insert("property".to_string(), property.to_string());
        metadata.insert("confidence".to_string(), format!("{:.6}", confidence));

        self.actions.push(AuditAction {
            timestamp: iso8601_now(),
            kind: "result".to_string(),
            description: result.to_string(),
            metadata,
        });
    }

    /// Serializes the entire trail to a pretty-printed JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.actions)
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Produces a human-readable summary: action count broken down by kind,
    /// plus the number of result entries.
    pub fn summary(&self) -> String {
        let total = self.actions.len();
        let results = self
            .actions
            .iter()
            .filter(|a| a.kind == "result")
            .count();
        let non_result = total - results;

        let mut kind_counts: HashMap<&str, usize> = HashMap::new();
        for a in &self.actions {
            if a.kind != "result" {
                *kind_counts.entry(a.kind.as_str()).or_insert(0) += 1;
            }
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Audit trail: {} total entries", total));
        lines.push(format!("  Actions : {}", non_result));
        for (kind, count) in &kind_counts {
            lines.push(format!("    - {}: {}", kind, count));
        }
        lines.push(format!("  Results : {}", results));

        lines.join("\n")
    }

    /// Total number of entries in the trail.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Returns a slice over the raw actions.
    pub fn actions(&self) -> &[AuditAction] {
        &self.actions
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Produces an ISO 8601 timestamp string for the current moment in UTC.
/// Falls back to a fixed placeholder only when the system clock is
/// unavailable (essentially never on supported platforms).
fn iso8601_now() -> String {
    let now = std::time::SystemTime::now();
    match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(dur) => {
            let secs = dur.as_secs();
            let days = secs / 86400;
            let remaining = secs % 86400;
            let hours = remaining / 3600;
            let minutes = (remaining % 3600) / 60;
            let seconds = remaining % 60;
            let millis = dur.subsec_millis();

            // Convert days since epoch to year-month-day via a simple
            // calendar computation (handles leap years correctly).
            let (year, month, day) = days_to_ymd(days);

            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
                year, month, day, hours, minutes, seconds, millis
            )
        }
        Err(_) => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

/// Converts a count of days since the Unix epoch (1970-01-01) to
/// `(year, month, day)`.
fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    // Algorithm adapted from Howard Hinnant's `civil_from_days`.
    let z = days_since_epoch as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as u64, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_logger_basic() {
        let mut logger = AuditLogger::new("test-component");
        assert!(logger.is_empty());

        logger.log_info("system started");
        logger.log_warn("disk space low");
        logger.log_error("connection lost");
        logger.log_debug("handshake bytes received");

        assert_eq!(logger.len(), 4);
        assert!(!logger.is_empty());

        let entries = logger.entries();
        assert_eq!(entries[0].level, LogLevel::Info);
        assert_eq!(entries[0].message, "system started");
        assert_eq!(entries[0].component, "test-component");
        assert_eq!(entries[1].level, LogLevel::Warn);
        assert_eq!(entries[2].level, LogLevel::Error);
        assert_eq!(entries[3].level, LogLevel::Debug);
    }

    #[test]
    fn test_audit_logger_entries_at_level() {
        let mut logger = AuditLogger::new("filter-test");
        logger.log_info("a");
        logger.log_info("b");
        logger.log_warn("c");
        logger.log_error("d");
        logger.log_debug("e");

        let infos = logger.entries_at_level(LogLevel::Info);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].message, "a");
        assert_eq!(infos[1].message, "b");

        let warns = logger.entries_at_level(LogLevel::Warn);
        assert_eq!(warns.len(), 1);

        let errors = logger.entries_at_level(LogLevel::Error);
        assert_eq!(errors.len(), 1);

        let debugs = logger.entries_at_level(LogLevel::Debug);
        assert_eq!(debugs.len(), 1);
    }

    #[test]
    fn test_audit_logger_entries_at_or_above() {
        let mut logger = AuditLogger::new("severity");
        logger.log_debug("d");
        logger.log_info("i");
        logger.log_warn("w");
        logger.log_error("e");

        let warn_and_above = logger.entries_at_or_above(LogLevel::Warn);
        assert_eq!(warn_and_above.len(), 2);
        assert!(warn_and_above.iter().all(|e| e.level.severity() >= LogLevel::Warn.severity()));
    }

    #[test]
    fn test_log_entry_serialization() {
        let entry = LogEntry::new(LogLevel::Info, "serde-test", "hello world")
            .with_context("key1", "value1")
            .with_context("key2", "value2");

        let json = entry.to_json();
        assert!(json.contains("\"level\":\"Info\""));
        assert!(json.contains("\"message\":\"hello world\""));
        assert!(json.contains("\"component\":\"serde-test\""));
        assert!(json.contains("\"key1\":\"value1\""));

        // Round-trip deserialization
        let parsed: LogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.level, LogLevel::Info);
        assert_eq!(parsed.message, "hello world");
        assert_eq!(parsed.context.get("key2").unwrap(), "value2");
    }

    #[test]
    fn test_performance_timer() {
        let timer = PerformanceTimer::start("test-op");
        // Perform a tiny amount of work to ensure some time passes.
        let mut _sum = 0u64;
        for i in 0..1000 {
            _sum += i;
        }
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 0.0);

        let record = timer.stop();
        assert_eq!(record.label, "test-op");
        assert!(record.duration_ms >= 0.0);
        assert!(!record.timestamp.is_empty());

        let json = record.to_json();
        assert!(json.contains("\"label\":\"test-op\""));
    }

    #[test]
    fn test_query_logger() {
        let mut ql = QueryLogger::new();
        assert_eq!(ql.total_queries(), 0);
        assert_eq!(ql.average_latency(), 0.0);

        ql.log_query("embedding", "hello", "[0.1, 0.2]", 12.5);
        ql.log_query("embedding", "world", "[0.3, 0.4]", 15.0);
        ql.log_query("completion", "prompt", "response", 120.0);

        assert_eq!(ql.total_queries(), 3);

        let avg = ql.average_latency();
        let expected = (12.5 + 15.0 + 120.0) / 3.0;
        assert!((avg - expected).abs() < 1e-9);

        let by_type = ql.queries_by_type();
        assert_eq!(by_type.get("embedding"), Some(&2));
        assert_eq!(by_type.get("completion"), Some(&1));

        assert!((ql.max_latency() - 120.0).abs() < 1e-9);
        assert!((ql.min_latency() - 12.5).abs() < 1e-9);

        let json = ql.to_json();
        assert!(json.contains("embedding"));
        assert!(json.contains("completion"));
    }

    #[test]
    fn test_audit_trail_actions_and_results() {
        let mut trail = AuditTrail::new();
        assert!(trail.is_empty());

        trail.record_action("start", "beginning behavioral audit");
        trail.record_action("query", "sent equivalence prompt");
        trail.record_result("monotonicity", "pass", 0.95);
        trail.record_result("compositionality", "fail", 0.42);

        assert_eq!(trail.len(), 4);

        let json = trail.to_json();
        assert!(json.contains("monotonicity"));
        assert!(json.contains("compositionality"));
        assert!(json.contains("0.950000"));

        let summary = trail.summary();
        assert!(summary.contains("4 total entries"));
        assert!(summary.contains("Results : 2"));
    }

    #[test]
    fn test_audit_trail_summary_breakdown() {
        let mut trail = AuditTrail::new();
        trail.record_action("init", "initialize pipeline");
        trail.record_action("init", "load config");
        trail.record_action("query", "LLM prompt");
        trail.record_result("safety", "pass", 0.99);

        let summary = trail.summary();
        assert!(summary.contains("4 total entries"));
        assert!(summary.contains("Actions : 3"));
        assert!(summary.contains("Results : 1"));
        assert!(summary.contains("init: 2"));
        assert!(summary.contains("query: 1"));
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Debug.severity() < LogLevel::Info.severity());
        assert!(LogLevel::Info.severity() < LogLevel::Warn.severity());
        assert!(LogLevel::Warn.severity() < LogLevel::Error.severity());
    }

    #[test]
    fn test_audit_logger_clear() {
        let mut logger = AuditLogger::new("clear-test");
        logger.log_info("one");
        logger.log_info("two");
        assert_eq!(logger.len(), 2);

        logger.clear();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_iso8601_now_format() {
        let ts = iso8601_now();
        // Basic structural assertions on the timestamp.
        assert!(ts.ends_with('Z'), "timestamp should end with Z");
        assert!(ts.contains('T'), "timestamp should contain T separator");
        assert_eq!(ts.len(), 24, "ISO 8601 with millis should be 24 chars");
    }

    #[test]
    fn test_query_logger_default() {
        let ql = QueryLogger::default();
        assert_eq!(ql.total_queries(), 0);
        assert_eq!(ql.average_latency(), 0.0);
    }
}
