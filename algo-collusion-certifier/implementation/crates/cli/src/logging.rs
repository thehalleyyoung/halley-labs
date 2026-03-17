//! Logging setup and structured logging for the CollusionProof CLI.
//!
//! Provides configurable log levels, structured context, timing measurement,
//! and an append-only audit log for certification runs.

use std::fmt;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use log::{Level, LevelFilter};
use serde::{Deserialize, Serialize};

// ── Verbosity level ─────────────────────────────────────────────────────────

/// Controls how much output the CLI produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VerbosityLevel {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

impl VerbosityLevel {
    /// Convert to a log LevelFilter.
    pub fn to_level_filter(self) -> LevelFilter {
        match self {
            VerbosityLevel::Quiet => LevelFilter::Error,
            VerbosityLevel::Normal => LevelFilter::Info,
            VerbosityLevel::Verbose => LevelFilter::Debug,
            VerbosityLevel::Debug => LevelFilter::Trace,
        }
    }

    /// Parse from CLI flags: quiet flag + verbose count.
    pub fn from_flags(quiet: bool, verbose: u8) -> Self {
        if quiet {
            VerbosityLevel::Quiet
        } else {
            match verbose {
                0 => VerbosityLevel::Normal,
                1 => VerbosityLevel::Verbose,
                _ => VerbosityLevel::Debug,
            }
        }
    }
}

impl fmt::Display for VerbosityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerbosityLevel::Quiet => write!(f, "quiet"),
            VerbosityLevel::Normal => write!(f, "normal"),
            VerbosityLevel::Verbose => write!(f, "verbose"),
            VerbosityLevel::Debug => write!(f, "debug"),
        }
    }
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        VerbosityLevel::Normal
    }
}

// ── Logging setup ───────────────────────────────────────────────────────────

/// Initialize the global logger with the given verbosity level.
pub fn setup_logging(verbosity: VerbosityLevel) {
    let level_filter = verbosity.to_level_filter();

    env_logger::Builder::new()
        .filter_level(level_filter)
        .format(|buf, record| {
            let level_style = match record.level() {
                Level::Error => "\x1b[31m",   // red
                Level::Warn => "\x1b[33m",    // yellow
                Level::Info => "\x1b[32m",     // green
                Level::Debug => "\x1b[36m",    // cyan
                Level::Trace => "\x1b[90m",    // gray
            };
            let reset = "\x1b[0m";
            let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");

            writeln!(
                buf,
                "{} {}{:>5}{} [{}] {}",
                timestamp,
                level_style,
                record.level(),
                reset,
                record.target(),
                record.args()
            )
        })
        .init();

    log::debug!("Logging initialized at level: {}", verbosity);
}

/// Initialize logging for tests (ignores multiple-init errors).
pub fn setup_test_logging() {
    let _ = env_logger::Builder::new()
        .filter_level(LevelFilter::Debug)
        .is_test(true)
        .try_init();
}

// ── Structured log context ──────────────────────────────────────────────────

/// Structured context attached to log entries for pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    pub scenario_id: String,
    pub stage: String,
    pub round: Option<usize>,
    pub metadata: Vec<(String, String)>,
}

impl LogContext {
    pub fn new(scenario_id: impl Into<String>, stage: impl Into<String>) -> Self {
        Self {
            scenario_id: scenario_id.into(),
            stage: stage.into(),
            round: None,
            metadata: Vec::new(),
        }
    }

    pub fn with_round(mut self, round: usize) -> Self {
        self.round = Some(round);
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Emit an info log with this context.
    pub fn info(&self, message: &str) {
        log::info!("[{}:{}] {}", self.scenario_id, self.stage, message);
    }

    /// Emit a debug log with this context.
    pub fn debug(&self, message: &str) {
        log::debug!("[{}:{}] {}", self.scenario_id, self.stage, message);
    }

    /// Emit a warning log with this context.
    pub fn warn(&self, message: &str) {
        log::warn!("[{}:{}] {}", self.scenario_id, self.stage, message);
    }

    /// Emit an error log with this context.
    pub fn error(&self, message: &str) {
        log::error!("[{}:{}] {}", self.scenario_id, self.stage, message);
    }
}

impl fmt::Display for LogContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.scenario_id, self.stage)?;
        if let Some(r) = self.round {
            write!(f, " (round {})", r)?;
        }
        for (k, v) in &self.metadata {
            write!(f, " {}={}", k, v)?;
        }
        Ok(())
    }
}

// ── Timing measurement ──────────────────────────────────────────────────────

/// Measures elapsed time for a named stage.
#[derive(Debug, Clone)]
pub struct StageTimer {
    pub name: String,
    pub start: Instant,
    pub end: Option<Instant>,
}

impl StageTimer {
    pub fn start(name: impl Into<String>) -> Self {
        let name = name.into();
        log::debug!("⏱ Starting stage: {}", name);
        Self {
            name,
            start: Instant::now(),
            end: None,
        }
    }

    pub fn stop(&mut self) -> Duration {
        let elapsed = self.start.elapsed();
        self.end = Some(Instant::now());
        log::debug!("⏱ Stage '{}' completed in {:.3}s", self.name, elapsed.as_secs_f64());
        elapsed
    }

    pub fn elapsed(&self) -> Duration {
        match self.end {
            Some(end) => end.duration_since(self.start),
            None => self.start.elapsed(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.end.is_none()
    }
}

/// Logs the start and end of a named pipeline stage.
pub fn log_stage_start(scenario_id: &str, stage: &str) {
    log::info!("▶ [{}] Starting stage: {}", scenario_id, stage);
}

pub fn log_stage_end(scenario_id: &str, stage: &str, duration: Duration) {
    log::info!(
        "✔ [{}] Completed stage: {} ({:.3}s)",
        scenario_id,
        stage,
        duration.as_secs_f64()
    );
}

pub fn log_stage_error(scenario_id: &str, stage: &str, error: &str) {
    log::error!("✘ [{}] Failed stage: {} — {}", scenario_id, stage, error);
}

// ── Timing collector ────────────────────────────────────────────────────────

/// Collected timing entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingEntry {
    pub stage: String,
    pub duration_secs: f64,
    pub timestamp: DateTime<Utc>,
}

/// Collects per-stage timing information across a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct TimingCollector {
    entries: Vec<TimingEntry>,
    active_timer: Option<(String, Instant)>,
}

impl TimingCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin timing a stage.
    pub fn start_stage(&mut self, stage: impl Into<String>) {
        let name = stage.into();
        if let Some((prev, start)) = self.active_timer.take() {
            let elapsed = start.elapsed();
            self.entries.push(TimingEntry {
                stage: prev,
                duration_secs: elapsed.as_secs_f64(),
                timestamp: Utc::now(),
            });
        }
        self.active_timer = Some((name, Instant::now()));
    }

    /// Finish timing the current stage.
    pub fn finish_stage(&mut self) {
        if let Some((name, start)) = self.active_timer.take() {
            let elapsed = start.elapsed();
            self.entries.push(TimingEntry {
                stage: name,
                duration_secs: elapsed.as_secs_f64(),
                timestamp: Utc::now(),
            });
        }
    }

    /// Get all collected timing entries.
    pub fn entries(&self) -> &[TimingEntry] {
        &self.entries
    }

    /// Total elapsed time across all stages.
    pub fn total_duration(&self) -> f64 {
        self.entries.iter().map(|e| e.duration_secs).sum()
    }

    /// Format timing information as a summary string.
    pub fn summary(&self) -> String {
        let mut s = String::from("Pipeline Timing Summary:\n");
        for entry in &self.entries {
            s.push_str(&format!("  {:30} {:>8.3}s\n", entry.stage, entry.duration_secs));
        }
        s.push_str(&format!("  {:30} {:>8.3}s\n", "TOTAL", self.total_duration()));
        s
    }
}

// ── Audit log ───────────────────────────────────────────────────────────────

/// An append-only audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub scenario_id: String,
    pub details: String,
    pub metadata: serde_json::Value,
}

/// Types of auditable events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditEventType {
    PipelineStarted,
    StageCompleted,
    StageFailed,
    TestExecuted,
    CertificateGenerated,
    CertificateVerified,
    ConfigLoaded,
    EvaluationStarted,
    EvaluationCompleted,
}

impl fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditEventType::PipelineStarted => write!(f, "PIPELINE_STARTED"),
            AuditEventType::StageCompleted => write!(f, "STAGE_COMPLETED"),
            AuditEventType::StageFailed => write!(f, "STAGE_FAILED"),
            AuditEventType::TestExecuted => write!(f, "TEST_EXECUTED"),
            AuditEventType::CertificateGenerated => write!(f, "CERTIFICATE_GENERATED"),
            AuditEventType::CertificateVerified => write!(f, "CERTIFICATE_VERIFIED"),
            AuditEventType::ConfigLoaded => write!(f, "CONFIG_LOADED"),
            AuditEventType::EvaluationStarted => write!(f, "EVALUATION_STARTED"),
            AuditEventType::EvaluationCompleted => write!(f, "EVALUATION_COMPLETED"),
        }
    }
}

/// Thread-safe append-only audit log for certification runs.
#[derive(Debug, Clone)]
pub struct AuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Append an entry to the audit log.
    pub fn log(
        &self,
        event_type: AuditEventType,
        scenario_id: impl Into<String>,
        details: impl Into<String>,
    ) {
        self.log_with_metadata(event_type, scenario_id, details, serde_json::Value::Null);
    }

    /// Append an entry with metadata.
    pub fn log_with_metadata(
        &self,
        event_type: AuditEventType,
        scenario_id: impl Into<String>,
        details: impl Into<String>,
        metadata: serde_json::Value,
    ) {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            event_type,
            scenario_id: scenario_id.into(),
            details: details.into(),
            metadata,
        };
        if let Ok(mut entries) = self.entries.lock() {
            entries.push(entry);
        }
    }

    /// Get all entries.
    pub fn entries(&self) -> Vec<AuditEntry> {
        self.entries.lock().map(|e| e.clone()).unwrap_or_default()
    }

    /// Get entries filtered by scenario.
    pub fn entries_for_scenario(&self, scenario_id: &str) -> Vec<AuditEntry> {
        self.entries()
            .into_iter()
            .filter(|e| e.scenario_id == scenario_id)
            .collect()
    }

    /// Write audit log to a JSON file.
    pub fn write_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let entries = self.entries();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_level_from_flags() {
        assert_eq!(VerbosityLevel::from_flags(true, 0), VerbosityLevel::Quiet);
        assert_eq!(VerbosityLevel::from_flags(false, 0), VerbosityLevel::Normal);
        assert_eq!(VerbosityLevel::from_flags(false, 1), VerbosityLevel::Verbose);
        assert_eq!(VerbosityLevel::from_flags(false, 2), VerbosityLevel::Debug);
        assert_eq!(VerbosityLevel::from_flags(false, 10), VerbosityLevel::Debug);
    }

    #[test]
    fn test_verbosity_level_ordering() {
        assert!(VerbosityLevel::Quiet < VerbosityLevel::Normal);
        assert!(VerbosityLevel::Normal < VerbosityLevel::Verbose);
        assert!(VerbosityLevel::Verbose < VerbosityLevel::Debug);
    }

    #[test]
    fn test_verbosity_to_level_filter() {
        assert_eq!(VerbosityLevel::Quiet.to_level_filter(), LevelFilter::Error);
        assert_eq!(VerbosityLevel::Normal.to_level_filter(), LevelFilter::Info);
        assert_eq!(VerbosityLevel::Verbose.to_level_filter(), LevelFilter::Debug);
        assert_eq!(VerbosityLevel::Debug.to_level_filter(), LevelFilter::Trace);
    }

    #[test]
    fn test_verbosity_display() {
        assert_eq!(VerbosityLevel::Quiet.to_string(), "quiet");
        assert_eq!(VerbosityLevel::Normal.to_string(), "normal");
        assert_eq!(VerbosityLevel::Verbose.to_string(), "verbose");
        assert_eq!(VerbosityLevel::Debug.to_string(), "debug");
    }

    #[test]
    fn test_log_context_creation() {
        let ctx = LogContext::new("scenario_1", "simulation");
        assert_eq!(ctx.scenario_id, "scenario_1");
        assert_eq!(ctx.stage, "simulation");
        assert!(ctx.round.is_none());
    }

    #[test]
    fn test_log_context_with_round() {
        let ctx = LogContext::new("s1", "test").with_round(42);
        assert_eq!(ctx.round, Some(42));
    }

    #[test]
    fn test_log_context_with_metadata() {
        let ctx = LogContext::new("s1", "test")
            .with_meta("alpha", "0.05")
            .with_meta("layer", "0");
        assert_eq!(ctx.metadata.len(), 2);
    }

    #[test]
    fn test_log_context_display() {
        let ctx = LogContext::new("s1", "sim").with_round(5);
        let s = ctx.to_string();
        assert!(s.contains("s1"));
        assert!(s.contains("sim"));
        assert!(s.contains("round 5"));
    }

    #[test]
    fn test_stage_timer() {
        let mut timer = StageTimer::start("test_stage");
        assert!(timer.is_running());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let dur = timer.stop();
        assert!(!timer.is_running());
        assert!(dur.as_millis() >= 5);
    }

    #[test]
    fn test_timing_collector() {
        let mut collector = TimingCollector::new();
        collector.start_stage("stage_a");
        std::thread::sleep(std::time::Duration::from_millis(10));
        collector.finish_stage();
        collector.start_stage("stage_b");
        std::thread::sleep(std::time::Duration::from_millis(10));
        collector.finish_stage();

        assert_eq!(collector.entries().len(), 2);
        assert!(collector.total_duration() > 0.01);
    }

    #[test]
    fn test_timing_collector_auto_close_previous() {
        let mut collector = TimingCollector::new();
        collector.start_stage("a");
        collector.start_stage("b"); // auto-closes "a"
        collector.finish_stage();

        assert_eq!(collector.entries().len(), 2);
    }

    #[test]
    fn test_timing_collector_summary() {
        let mut collector = TimingCollector::new();
        collector.start_stage("init");
        collector.finish_stage();
        let s = collector.summary();
        assert!(s.contains("init"));
        assert!(s.contains("TOTAL"));
    }

    #[test]
    fn test_audit_log_basic() {
        let log = AuditLog::new();
        assert!(log.is_empty());

        log.log(AuditEventType::PipelineStarted, "s1", "Pipeline started");
        assert_eq!(log.len(), 1);

        log.log(AuditEventType::StageCompleted, "s1", "Simulation done");
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_audit_log_with_metadata() {
        let log = AuditLog::new();
        log.log_with_metadata(
            AuditEventType::TestExecuted,
            "s1",
            "Price convergence test",
            serde_json::json!({"p_value": 0.03}),
        );
        let entries = log.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].event_type, AuditEventType::TestExecuted);
    }

    #[test]
    fn test_audit_log_filter_by_scenario() {
        let log = AuditLog::new();
        log.log(AuditEventType::PipelineStarted, "s1", "start");
        log.log(AuditEventType::PipelineStarted, "s2", "start");
        log.log(AuditEventType::StageCompleted, "s1", "done");

        let s1_entries = log.entries_for_scenario("s1");
        assert_eq!(s1_entries.len(), 2);
    }

    #[test]
    fn test_audit_log_thread_safety() {
        let log = AuditLog::new();
        let log_clone = log.clone();

        let handle = std::thread::spawn(move || {
            for i in 0..10 {
                log_clone.log(
                    AuditEventType::StageCompleted,
                    "s1",
                    format!("Stage {}", i),
                );
            }
        });

        for i in 0..10 {
            log.log(AuditEventType::StageCompleted, "s2", format!("Stage {}", i));
        }

        handle.join().unwrap();
        assert_eq!(log.len(), 20);
    }

    #[test]
    fn test_audit_event_type_display() {
        assert_eq!(AuditEventType::PipelineStarted.to_string(), "PIPELINE_STARTED");
        assert_eq!(AuditEventType::CertificateGenerated.to_string(), "CERTIFICATE_GENERATED");
    }

    #[test]
    fn test_audit_log_write_to_file() {
        let log = AuditLog::new();
        log.log(AuditEventType::PipelineStarted, "s1", "test");

        let tmp = std::env::temp_dir().join("collusion_proof_test_audit.json");
        log.write_to_file(&tmp).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("PIPELINE_STARTED"));
        let _ = std::fs::remove_file(&tmp);
    }
}
