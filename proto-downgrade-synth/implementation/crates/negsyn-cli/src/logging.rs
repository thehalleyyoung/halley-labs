//! Logging configuration for the NegSynth CLI.
//!
//! Provides custom log formatting with timestamps, verbosity level mapping,
//! optional log file output, structured logging fields, and performance
//! timing decorators.

use env_logger::fmt::Formatter;
use log::{Level, LevelFilter, Record};
use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Logging configuration derived from CLI flags.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Verbosity: 0 = errors only, 1 = warn, 2 = info, 3 = debug, 4+ = trace.
    pub verbosity: u8,
    /// Optional path to a log file (receives all messages regardless of level).
    pub log_file: Option<PathBuf>,
    /// If true, strip ANSI colour codes.
    pub no_color: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            verbosity: 1,
            log_file: None,
            no_color: false,
        }
    }
}

impl LogConfig {
    /// Map numeric verbosity to a [`LevelFilter`].
    pub fn level_filter(&self) -> LevelFilter {
        match self.verbosity {
            0 => LevelFilter::Error,
            1 => LevelFilter::Warn,
            2 => LevelFilter::Info,
            3 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    }
}

// ---------------------------------------------------------------------------
// File logger
// ---------------------------------------------------------------------------

/// Thread-safe log-file writer that receives a copy of every log record.
static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

/// Attempt to open (or create) a log file and store it in the global cell.
fn open_log_file(path: &Path) -> io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    LOG_FILE
        .set(Mutex::new(file))
        .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "log file already set"))
}

/// Write a formatted record to the log file, if one is open.
fn write_to_log_file(record: &Record<'_>) {
    if let Some(mtx) = LOG_FILE.get() {
        if let Ok(mut f) = mtx.lock() {
            let now = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f");
            let _ = writeln!(
                f,
                "{} [{:<5}] {}: {}",
                now,
                record.level(),
                record.target(),
                record.args()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/// Initialise the global logger according to `config`.
///
/// Must be called exactly once (typically from `main`).
/// Falls back to `env_logger` defaults if `RUST_LOG` is set.
pub fn init(config: &LogConfig) -> Result<(), anyhow::Error> {
    if let Some(ref path) = config.log_file {
        open_log_file(path)?;
    }

    let filter = config.level_filter();
    let no_color = config.no_color;

    let mut builder = env_logger::Builder::new();
    builder.filter_level(filter);

    // Allow RUST_LOG to override if present.
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        builder.parse_filters(&rust_log);
    }

    builder.format(move |buf: &mut Formatter, record: &Record<'_>| {
        // Also write to log file.
        write_to_log_file(record);

        let now = chrono::Local::now().format("%H:%M:%S");
        let level_str = level_token(record.level(), no_color);
        let target = short_target(record.target());

        writeln!(buf, "{now} {level_str} [{target}] {}", record.args())
    });

    builder.try_init().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Coloured (or plain) level prefix.
fn level_token(level: Level, no_color: bool) -> String {
    if no_color {
        return format!("{:<5}", level);
    }
    match level {
        Level::Error => format!("\x1b[1;31m{:<5}\x1b[0m", level),
        Level::Warn => format!("\x1b[1;33m{:<5}\x1b[0m", level),
        Level::Info => format!("\x1b[1;32m{:<5}\x1b[0m", level),
        Level::Debug => format!("\x1b[1;36m{:<5}\x1b[0m", level),
        Level::Trace => format!("\x1b[1;35m{:<5}\x1b[0m", level),
    }
}

/// Trim crate-prefix to keep log lines short.
fn short_target(target: &str) -> &str {
    target
        .strip_prefix("negsyn_cli::")
        .or_else(|| target.strip_prefix("negsyn_"))
        .unwrap_or(target)
}

// ---------------------------------------------------------------------------
// Structured fields helper
// ---------------------------------------------------------------------------

/// A set of key-value fields that can be appended to a log message.
#[derive(Debug, Clone, Default)]
pub struct LogFields {
    pairs: Vec<(String, String)>,
}

impl LogFields {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    pub fn field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.pairs.push((key.into(), value.into()));
        self
    }

    /// Format as ` key=value key=value …`.
    pub fn format(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.pairs {
            if v.contains(' ') {
                write!(out, " {k}=\"{v}\"").unwrap();
            } else {
                write!(out, " {k}={v}").unwrap();
            }
        }
        out
    }
}

impl std::fmt::Display for LogFields {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format())
    }
}

// ---------------------------------------------------------------------------
// Timing guard
// ---------------------------------------------------------------------------

/// RAII guard that logs elapsed wall-clock time when dropped.
///
/// ```ignore
/// let _t = TimingGuard::new("extraction");
/// // … expensive work …
/// // prints: "extraction completed in 1.234s"
/// ```
pub struct TimingGuard {
    label: String,
    start: Instant,
    level: Level,
}

impl TimingGuard {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            level: Level::Info,
        }
    }

    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    /// Consume the guard and return elapsed duration *without* logging.
    pub fn finish_silent(self) -> std::time::Duration {
        let d = self.start.elapsed();
        std::mem::forget(self); // skip Drop
        d
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let secs = elapsed.as_secs_f64();
        let msg = if secs < 1.0 {
            format!("{} completed in {:.1}ms", self.label, secs * 1000.0)
        } else {
            format!("{} completed in {:.3}s", self.label, secs)
        };
        log::log!(self.level, "{msg}");
    }
}

// ---------------------------------------------------------------------------
// Scoped timing utility (non-RAII)
// ---------------------------------------------------------------------------

/// Measure wall-clock time for a closure and return `(result, duration)`.
pub fn timed<F, T>(f: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

/// Measure and log at info level.
pub fn timed_log<F, T>(label: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let _g = TimingGuard::new(label);
    f()
}

// ---------------------------------------------------------------------------
// Phase progress helper
// ---------------------------------------------------------------------------

/// Emit a phase-start message to stderr.
pub fn phase_start(phase: &str) {
    eprint_phase(phase, "started");
}

/// Emit a phase-end message to stderr.
pub fn phase_end(phase: &str) {
    eprint_phase(phase, "done");
}

fn eprint_phase(phase: &str, status: &str) {
    let now = chrono::Local::now().format("%H:%M:%S");
    eprintln!("[{now}] {phase}: {status}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbosity_mapping() {
        assert_eq!(LogConfig { verbosity: 0, ..Default::default() }.level_filter(), LevelFilter::Error);
        assert_eq!(LogConfig { verbosity: 1, ..Default::default() }.level_filter(), LevelFilter::Warn);
        assert_eq!(LogConfig { verbosity: 2, ..Default::default() }.level_filter(), LevelFilter::Info);
        assert_eq!(LogConfig { verbosity: 3, ..Default::default() }.level_filter(), LevelFilter::Debug);
        assert_eq!(LogConfig { verbosity: 4, ..Default::default() }.level_filter(), LevelFilter::Trace);
        assert_eq!(LogConfig { verbosity: 255, ..Default::default() }.level_filter(), LevelFilter::Trace);
    }

    #[test]
    fn short_target_strips_prefix() {
        assert_eq!(short_target("negsyn_cli::commands::analyze"), "commands::analyze");
        assert_eq!(short_target("negsyn_types"), "types");
        assert_eq!(short_target("something_else"), "something_else");
    }

    #[test]
    fn level_token_no_color() {
        let tok = level_token(Level::Error, true);
        assert_eq!(tok, "ERROR");
    }

    #[test]
    fn level_token_with_color() {
        let tok = level_token(Level::Error, false);
        assert!(tok.contains("\x1b["));
        assert!(tok.contains("ERROR"));
    }

    #[test]
    fn log_fields_formatting() {
        let f = LogFields::new()
            .field("library", "openssl")
            .field("version", "3.0.1")
            .field("path", "/usr/lib/libssl.so");
        let s = f.format();
        assert!(s.contains("library=openssl"));
        assert!(s.contains("version=3.0.1"));
        assert!(s.contains("path=/usr/lib/libssl.so"));
    }

    #[test]
    fn log_fields_quotes_spaces() {
        let f = LogFields::new().field("desc", "hello world");
        let s = f.format();
        assert!(s.contains("desc=\"hello world\""));
    }

    #[test]
    fn timing_guard_elapsed() {
        let g = TimingGuard::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let d = g.finish_silent();
        assert!(d.as_millis() >= 5);
    }

    #[test]
    fn timed_returns_result() {
        let (val, dur) = timed(|| 42);
        assert_eq!(val, 42);
        assert!(dur.as_nanos() > 0);
    }

    #[test]
    fn default_log_config() {
        let c = LogConfig::default();
        assert_eq!(c.verbosity, 1);
        assert!(!c.no_color);
        assert!(c.log_file.is_none());
    }
}
