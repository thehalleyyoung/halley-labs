//! SMT solver interface and process-based solver implementation.
//!
//! Provides the [`SmtSolver`] trait and a concrete implementation
//! [`ProcessSolver`] that communicates with an external solver (e.g. Z3)
//! via stdin/stdout.

use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::ast::{SmtExpr, SmtScript};
use crate::model::SmtModel;
use crate::sexp_parser::SExp;

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Configuration for an SMT solver process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Path to the solver binary (default: `"z3"`).
    pub solver_path: PathBuf,
    /// Timeout in seconds per query.
    pub timeout_secs: u64,
    /// Use incremental mode (stdin pipe).
    pub incremental: bool,
    /// Memory limit in MB (0 = unlimited).
    pub memory_limit_mb: u64,
    /// SMT-LIB logic to set.
    pub logic: String,
    /// Extra command-line flags.
    pub extra_flags: Vec<String>,
    /// Dump all queries to files for debugging.
    pub dump_queries: bool,
    /// Directory for query dumps.
    pub dump_dir: Option<PathBuf>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            solver_path: PathBuf::from("z3"),
            timeout_secs: 30,
            incremental: false,
            memory_limit_mb: 4096,
            logic: "QF_LIA".to_string(),
            extra_flags: vec![],
            dump_queries: false,
            dump_dir: None,
        }
    }
}

impl SolverConfig {
    /// Create a config for Z3.
    pub fn z3() -> Self {
        SolverConfig {
            solver_path: PathBuf::from("z3"),
            extra_flags: vec!["-in".to_string(), "-smt2".to_string()],
            ..Default::default()
        }
    }

    /// Create a config for CVC5.
    pub fn cvc5() -> Self {
        SolverConfig {
            solver_path: PathBuf::from("cvc5"),
            extra_flags: vec![
                "--lang".to_string(),
                "smt2".to_string(),
                "--incremental".to_string(),
            ],
            ..Default::default()
        }
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Enable query dumping.
    pub fn with_dump(mut self, dir: PathBuf) -> Self {
        self.dump_queries = true;
        self.dump_dir = Some(dir);
        self
    }

    /// Build the command-line arguments for the solver.
    pub fn solver_args(&self) -> Vec<String> {
        let mut args = self.extra_flags.clone();
        if self.timeout_secs > 0 && self.solver_path.to_string_lossy().contains("z3") {
            args.push(format!("-t:{}", self.timeout_secs * 1000));
        }
        if self.memory_limit_mb > 0 && self.solver_path.to_string_lossy().contains("z3") {
            args.push(format!("-memory:{}", self.memory_limit_mb));
        }
        args
    }
}

// ---------------------------------------------------------------------------
// Solver result
// ---------------------------------------------------------------------------

/// Result of a `check-sat` query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverResult {
    /// Satisfiable — model may be available.
    Sat(Option<SmtModel>),
    /// Unsatisfiable.
    Unsat,
    /// Unknown (solver gave up or timed out).
    Unknown(String),
    /// Error from solver.
    Error(String),
}

impl SolverResult {
    /// Whether the result is SAT.
    pub fn is_sat(&self) -> bool {
        matches!(self, SolverResult::Sat(_))
    }

    /// Whether the result is UNSAT.
    pub fn is_unsat(&self) -> bool {
        matches!(self, SolverResult::Unsat)
    }

    /// Whether the result is unknown.
    pub fn is_unknown(&self) -> bool {
        matches!(self, SolverResult::Unknown(_))
    }

    /// Whether the result is an error.
    pub fn is_error(&self) -> bool {
        matches!(self, SolverResult::Error(_))
    }

    /// Extract model if SAT.
    pub fn model(&self) -> Option<&SmtModel> {
        match self {
            SolverResult::Sat(Some(m)) => Some(m),
            _ => None,
        }
    }

    /// Extract unsat core names (placeholder — real implementation reads from solver).
    pub fn unsat_core(&self) -> Option<Vec<String>> {
        None
    }
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverResult::Sat(Some(m)) => write!(f, "sat\n{}", m),
            SolverResult::Sat(None) => write!(f, "sat"),
            SolverResult::Unsat => write!(f, "unsat"),
            SolverResult::Unknown(msg) => write!(f, "unknown: {}", msg),
            SolverResult::Error(msg) => write!(f, "error: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Solver trait
// ---------------------------------------------------------------------------

/// Trait for SMT solver implementations.
pub trait SmtSolver: Send {
    /// Run a complete script (non-incremental).
    fn run_script(&mut self, script: &SmtScript) -> SolverResult;

    /// Check satisfiability of the current assertion stack.
    fn check_sat_with_text(&mut self, smt_text: &str) -> SolverResult;

    /// Send a command and read the response (incremental mode).
    fn send_command(&mut self, cmd: &str) -> Result<String, String>;

    /// Get the solver name.
    fn name(&self) -> &str;

    /// Get solver statistics (if available).
    fn statistics(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Reset the solver state.
    fn reset(&mut self);

    /// Check if the solver process is alive.
    fn is_alive(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Process solver
// ---------------------------------------------------------------------------

/// An SMT solver that runs as an external process.
pub struct ProcessSolver {
    config: SolverConfig,
    /// The child process (for incremental mode).
    child: Option<Child>,
    /// Query counter for dump file naming.
    query_counter: u64,
    /// Total solver time in microseconds.
    total_time_us: u64,
    /// Number of queries.
    num_queries: u64,
}

impl ProcessSolver {
    /// Create a new process solver.
    pub fn new(config: SolverConfig) -> Self {
        ProcessSolver {
            config,
            child: None,
            query_counter: 0,
            total_time_us: 0,
            num_queries: 0,
        }
    }

    /// Create with default Z3 config.
    pub fn z3() -> Self {
        Self::new(SolverConfig::z3())
    }

    /// Create with default CVC5 config.
    pub fn cvc5() -> Self {
        Self::new(SolverConfig::cvc5())
    }

    /// Get the configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Get timing statistics.
    pub fn timing_stats(&self) -> (u64, u64, f64) {
        let avg = if self.num_queries > 0 {
            self.total_time_us as f64 / self.num_queries as f64
        } else {
            0.0
        };
        (self.num_queries, self.total_time_us, avg)
    }

    /// Optionally dump a query to a file.
    fn dump_query(&mut self, text: &str) {
        if self.dump_queries_enabled() {
            if let Some(ref dir) = self.config.dump_dir {
                let path = dir.join(format!("query_{:04}.smt2", self.query_counter));
                let _ = std::fs::write(&path, text);
            }
        }
        self.query_counter += 1;
    }

    fn dump_queries_enabled(&self) -> bool {
        self.config.dump_queries && self.config.dump_dir.is_some()
    }

    /// Start the solver process for incremental mode.
    fn start_process(&mut self) -> Result<(), String> {
        let child = Command::new(&self.config.solver_path)
            .args(&self.config.solver_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to start solver: {}", e))?;
        self.child = Some(child);
        Ok(())
    }

    /// Run solver in one-shot mode: write script to temp file, execute, parse output.
    fn run_oneshot(&mut self, smt_text: &str) -> SolverResult {
        let start = Instant::now();
        self.dump_query(smt_text);

        let tmp = match tempfile::NamedTempFile::new() {
            Ok(t) => t,
            Err(e) => return SolverResult::Error(format!("temp file error: {}", e)),
        };
        if let Err(e) = std::fs::write(tmp.path(), smt_text) {
            return SolverResult::Error(format!("write error: {}", e));
        }

        let mut args = self.config.solver_args();
        args.push(tmp.path().to_string_lossy().to_string());

        let output = Command::new(&self.config.solver_path).args(&args).output();

        let elapsed = start.elapsed().as_micros() as u64;
        self.total_time_us += elapsed;
        self.num_queries += 1;

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                parse_solver_output(&stdout, &stderr)
            }
            Err(e) => SolverResult::Error(format!("execution error: {}", e)),
        }
    }
}

impl SmtSolver for ProcessSolver {
    fn run_script(&mut self, script: &SmtScript) -> SolverResult {
        let text = script.render();
        self.run_oneshot(&text)
    }

    fn check_sat_with_text(&mut self, smt_text: &str) -> SolverResult {
        self.run_oneshot(smt_text)
    }

    fn send_command(&mut self, cmd: &str) -> Result<String, String> {
        if self.child.is_none() {
            self.start_process()?;
        }
        let child = self.child.as_mut().ok_or("no solver process")?;
        let stdin = child.stdin.as_mut().ok_or("no stdin")?;
        stdin
            .write_all(cmd.as_bytes())
            .map_err(|e| format!("write error: {}", e))?;
        stdin
            .write_all(b"\n")
            .map_err(|e| format!("write error: {}", e))?;
        stdin.flush().map_err(|e| format!("flush error: {}", e))?;

        let stdout = child.stdout.as_mut().ok_or("no stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| format!("read error: {}", e))?;
        Ok(line.trim().to_string())
    }

    fn name(&self) -> &str {
        self.config
            .solver_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    }

    fn statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("queries".into(), self.num_queries.to_string());
        stats.insert("total_time_us".into(), self.total_time_us.to_string());
        stats
    }

    fn reset(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
        }
        self.child = None;
    }

    fn is_alive(&self) -> bool {
        self.child.is_some()
    }
}

impl Drop for ProcessSolver {
    fn drop(&mut self) {
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Parse the raw stdout/stderr from a solver invocation.
fn parse_solver_output(stdout: &str, stderr: &str) -> SolverResult {
    let lines: Vec<&str> = stdout.lines().collect();
    if lines.is_empty() {
        if !stderr.is_empty() {
            return SolverResult::Error(stderr.to_string());
        }
        return SolverResult::Error("no output from solver".to_string());
    }

    let first = lines[0].trim();
    match first {
        "sat" => {
            let model_text = if lines.len() > 1 {
                let rest = lines[1..].join("\n");
                parse_model_text(&rest)
            } else {
                None
            };
            SolverResult::Sat(model_text)
        }
        "unsat" => SolverResult::Unsat,
        "unknown" => {
            let reason = if lines.len() > 1 {
                lines[1..].join(" ")
            } else {
                "unknown".to_string()
            };
            SolverResult::Unknown(reason)
        }
        _ => {
            if first.starts_with("(error") {
                SolverResult::Error(first.to_string())
            } else if !stderr.is_empty() {
                SolverResult::Error(stderr.to_string())
            } else {
                SolverResult::Unknown(first.to_string())
            }
        }
    }
}

/// Try to parse model text from solver output.
fn parse_model_text(text: &str) -> Option<SmtModel> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    match crate::sexp_parser::parse_sexp(trimmed) {
        Ok(sexp) => Some(crate::model::parse_model_from_sexp(&sexp)),
        Err(_) => {
            let mut model = SmtModel::new();
            for line in trimmed.lines() {
                let line = line.trim();
                if line.starts_with("(define-fun") {
                    // Simple heuristic parse for common patterns
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 5 {
                        let name = parts[1].to_string();
                        let val_str = parts.last().unwrap().trim_end_matches(')');
                        if let Ok(v) = val_str.parse::<i64>() {
                            model.insert(name, crate::model::ModelValue::Int(v));
                        }
                    }
                }
            }
            if model.is_empty() {
                None
            } else {
                Some(model)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mock solver (for testing)
// ---------------------------------------------------------------------------

/// A mock solver that returns pre-configured results.
pub struct MockSolver {
    /// Queue of results to return.
    results: Vec<SolverResult>,
    /// Index into results.
    index: usize,
    /// Queries received.
    pub queries: Vec<String>,
}

impl MockSolver {
    /// Create a mock that always returns SAT.
    pub fn always_sat() -> Self {
        MockSolver {
            results: vec![SolverResult::Sat(None)],
            index: 0,
            queries: Vec::new(),
        }
    }

    /// Create a mock that always returns UNSAT.
    pub fn always_unsat() -> Self {
        MockSolver {
            results: vec![SolverResult::Unsat],
            index: 0,
            queries: Vec::new(),
        }
    }

    /// Create with a sequence of results.
    pub fn with_results(results: Vec<SolverResult>) -> Self {
        MockSolver {
            results,
            index: 0,
            queries: Vec::new(),
        }
    }

    /// Create a mock that returns SAT with a specific model.
    pub fn with_model(model: SmtModel) -> Self {
        MockSolver {
            results: vec![SolverResult::Sat(Some(model))],
            index: 0,
            queries: Vec::new(),
        }
    }
}

impl SmtSolver for MockSolver {
    fn run_script(&mut self, script: &SmtScript) -> SolverResult {
        self.queries.push(script.render());
        let result = if self.results.is_empty() {
            SolverResult::Unknown("no mock result".to_string())
        } else {
            let idx = self.index % self.results.len();
            self.index += 1;
            self.results[idx].clone()
        };
        result
    }

    fn check_sat_with_text(&mut self, smt_text: &str) -> SolverResult {
        self.queries.push(smt_text.to_string());
        let result = if self.results.is_empty() {
            SolverResult::Unknown("no mock result".to_string())
        } else {
            let idx = self.index % self.results.len();
            self.index += 1;
            self.results[idx].clone()
        };
        result
    }

    fn send_command(&mut self, cmd: &str) -> Result<String, String> {
        self.queries.push(cmd.to_string());
        Ok("success".to_string())
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn reset(&mut self) {
        self.index = 0;
        self.queries.clear();
    }

    fn is_alive(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_config_default() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.logic, "QF_LIA");
        assert_eq!(cfg.timeout_secs, 30);
    }

    #[test]
    fn test_solver_config_z3() {
        let cfg = SolverConfig::z3();
        assert!(cfg.extra_flags.contains(&"-in".to_string()));
    }

    #[test]
    fn test_solver_result_is_sat() {
        assert!(SolverResult::Sat(None).is_sat());
        assert!(!SolverResult::Unsat.is_sat());
        assert!(SolverResult::Unsat.is_unsat());
    }

    #[test]
    fn test_solver_result_display() {
        assert_eq!(format!("{}", SolverResult::Unsat), "unsat");
        assert_eq!(format!("{}", SolverResult::Sat(None)), "sat");
    }

    #[test]
    fn test_parse_solver_output_sat() {
        let result = parse_solver_output("sat\n", "");
        assert!(result.is_sat());
    }

    #[test]
    fn test_parse_solver_output_unsat() {
        let result = parse_solver_output("unsat\n", "");
        assert!(result.is_unsat());
    }

    #[test]
    fn test_parse_solver_output_unknown() {
        let result = parse_solver_output("unknown\ntimeout", "");
        assert!(result.is_unknown());
    }

    #[test]
    fn test_parse_solver_output_error() {
        let result = parse_solver_output("(error \"invalid\")\n", "");
        assert!(result.is_error());
    }

    #[test]
    fn test_mock_solver_always_sat() {
        let mut mock = MockSolver::always_sat();
        let script = SmtScript::new();
        let result = mock.run_script(&script);
        assert!(result.is_sat());
        assert_eq!(mock.queries.len(), 1);
    }

    #[test]
    fn test_mock_solver_always_unsat() {
        let mut mock = MockSolver::always_unsat();
        let result = mock.check_sat_with_text("(check-sat)");
        assert!(result.is_unsat());
    }

    #[test]
    fn test_mock_solver_sequence() {
        let mut mock = MockSolver::with_results(vec![SolverResult::Sat(None), SolverResult::Unsat]);
        let script = SmtScript::new();
        assert!(mock.run_script(&script).is_sat());
        assert!(mock.run_script(&script).is_unsat());
        // Wraps around
        assert!(mock.run_script(&script).is_sat());
    }

    #[test]
    fn test_mock_solver_with_model() {
        let mut model = SmtModel::new();
        model.insert("x".to_string(), crate::model::ModelValue::Int(42));
        let mut mock = MockSolver::with_model(model);
        let result = mock.run_script(&SmtScript::new());
        assert!(result.is_sat());
        let m = result.model().unwrap();
        assert_eq!(m.get_int("x"), Some(42));
    }

    #[test]
    fn test_mock_solver_reset() {
        let mut mock = MockSolver::always_sat();
        mock.check_sat_with_text("query1");
        mock.check_sat_with_text("query2");
        assert_eq!(mock.queries.len(), 2);
        mock.reset();
        assert!(mock.queries.is_empty());
    }

    #[test]
    fn test_process_solver_creation() {
        let solver = ProcessSolver::z3();
        assert_eq!(solver.name(), "z3");
        assert!(!solver.is_alive());
    }

    #[test]
    fn test_solver_config_with_timeout() {
        let cfg = SolverConfig::z3().with_timeout(60);
        assert_eq!(cfg.timeout_secs, 60);
    }

    #[test]
    fn test_solver_timing_stats() {
        let solver = ProcessSolver::z3();
        let (queries, total, avg) = solver.timing_stats();
        assert_eq!(queries, 0);
        assert_eq!(total, 0);
        assert_eq!(avg, 0.0);
    }
}
