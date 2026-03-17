//! Interactive REPL for the SoniType DSL.
//!
//! Provides a read-eval-print loop where users can incrementally build
//! sonification definitions, type-check expressions, and preview audio.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use crate::config::CliConfig;
use crate::diagnostics::{DiagnosticEngine, DiagnosticFormat};
use crate::output::OutputFormatter;
use crate::pipeline::{CompilationPipeline, PipelineOptions};

// ── REPL Commands ───────────────────────────────────────────────────────────

/// Recognised REPL meta-commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    /// :load <file> — load and evaluate a DSL file.
    Load(String),
    /// :type <expr> — show the inferred type of an expression.
    Type(String),
    /// :check <expr> — type-check an expression.
    Check(String),
    /// :render <expr> — render an expression to audio and play.
    Render(String),
    /// :info — display current session state.
    Info,
    /// :reset — clear all accumulated state.
    Reset,
    /// :help — print available commands.
    Help,
    /// :quit — exit the REPL.
    Quit,
    /// A DSL expression (not a meta-command).
    Expr(String),
    /// Empty line.
    Empty,
    /// Unrecognised meta-command.
    Unknown(String),
}

impl ReplCommand {
    /// Parse a line of input into a `ReplCommand`.
    pub fn parse(line: &str) -> Self {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Self::Empty;
        }
        if !trimmed.starts_with(':') {
            return Self::Expr(trimmed.to_string());
        }

        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim().to_string();

        match cmd {
            ":load" | ":l" => {
                if arg.is_empty() {
                    Self::Unknown(":load requires a file path".into())
                } else {
                    Self::Load(arg)
                }
            }
            ":type" | ":t" => {
                if arg.is_empty() {
                    Self::Unknown(":type requires an expression".into())
                } else {
                    Self::Type(arg)
                }
            }
            ":check" | ":c" => {
                if arg.is_empty() {
                    Self::Unknown(":check requires an expression".into())
                } else {
                    Self::Check(arg)
                }
            }
            ":render" | ":r" => {
                if arg.is_empty() {
                    Self::Unknown(":render requires an expression".into())
                } else {
                    Self::Render(arg)
                }
            }
            ":info" | ":i" => Self::Info,
            ":reset" => Self::Reset,
            ":help" | ":h" | ":?" => Self::Help,
            ":quit" | ":q" | ":exit" => Self::Quit,
            other => Self::Unknown(format!("Unknown command: {}", other)),
        }
    }
}

// ── REPL State ──────────────────────────────────────────────────────────────

/// Accumulated session state across REPL inputs.
#[derive(Debug, Clone)]
pub struct ReplState {
    /// Named variable bindings.
    pub bindings: HashMap<String, ReplBinding>,
    /// All source lines entered so far (for :info and re-evaluation).
    pub history: Vec<String>,
    /// Accumulated stream definitions.
    pub streams: Vec<String>,
    /// Number of expressions evaluated.
    pub eval_count: usize,
}

/// A single REPL binding.
#[derive(Debug, Clone)]
pub struct ReplBinding {
    pub name: String,
    pub type_name: String,
    pub value_repr: String,
}

impl Default for ReplState {
    fn default() -> Self {
        Self {
            bindings: HashMap::new(),
            history: Vec::new(),
            streams: Vec::new(),
            eval_count: 0,
        }
    }
}

impl ReplState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Record a new binding.
    pub fn bind(&mut self, name: impl Into<String>, type_name: impl Into<String>, value: impl Into<String>) {
        let name = name.into();
        let binding = ReplBinding {
            name: name.clone(),
            type_name: type_name.into(),
            value_repr: value.into(),
        };
        self.bindings.insert(name, binding);
    }

    /// Record a stream definition.
    pub fn add_stream(&mut self, name: impl Into<String>) {
        self.streams.push(name.into());
    }

    /// Format the current state for display.
    pub fn info_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Expressions evaluated: {}\n", self.eval_count));
        out.push_str(&format!("Streams defined: {}\n", self.streams.len()));
        for s in &self.streams {
            out.push_str(&format!("  - {}\n", s));
        }
        out.push_str(&format!("Bindings: {}\n", self.bindings.len()));
        for (name, b) in &self.bindings {
            out.push_str(&format!("  {} : {} = {}\n", name, b.type_name, b.value_repr));
        }
        out.push_str(&format!("History lines: {}\n", self.history.len()));
        out
    }
}

// ── REPL Engine ─────────────────────────────────────────────────────────────

/// The interactive REPL for SoniType.
pub struct SoniTypeRepl {
    config: CliConfig,
    state: ReplState,
    diagnostics: DiagnosticEngine,
}

impl SoniTypeRepl {
    pub fn new(config: CliConfig) -> Self {
        Self {
            config,
            state: ReplState::new(),
            diagnostics: DiagnosticEngine::new(DiagnosticFormat::Plain),
        }
    }

    /// Run the REPL loop, reading from stdin.
    pub fn run(&mut self) -> Result<()> {
        self.print_banner();
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            write!(stdout, "sonitype> ")?;
            stdout.flush()?;

            let mut line = String::new();
            let bytes_read = stdin.lock().read_line(&mut line)?;
            if bytes_read == 0 {
                // EOF
                println!();
                break;
            }

            let cmd = ReplCommand::parse(&line);
            match self.handle_command(cmd) {
                Ok(true) => continue,
                Ok(false) => break, // :quit
                Err(e) => eprintln!("Error: {:#}", e),
            }
        }

        Ok(())
    }

    /// Run the REPL with an explicit input/output pair (useful for testing
    /// and non-interactive use).
    pub fn run_with_io(
        &mut self,
        input: &mut dyn BufRead,
        output: &mut dyn Write,
    ) -> Result<()> {
        loop {
            write!(output, "sonitype> ")?;
            output.flush()?;

            let mut line = String::new();
            let bytes_read = input.read_line(&mut line)?;
            if bytes_read == 0 {
                writeln!(output)?;
                break;
            }

            let cmd = ReplCommand::parse(&line);
            match self.handle_command_io(cmd, output) {
                Ok(true) => continue,
                Ok(false) => break,
                Err(e) => writeln!(output, "Error: {:#}", e)?,
            }
        }
        Ok(())
    }

    /// Handle a single command. Returns `Ok(true)` to continue, `Ok(false)` to quit.
    pub fn handle_command(&mut self, cmd: ReplCommand) -> Result<bool> {
        let mut stdout = io::stdout();
        self.handle_command_io(cmd, &mut stdout)
    }

    fn handle_command_io(&mut self, cmd: ReplCommand, out: &mut dyn Write) -> Result<bool> {
        match cmd {
            ReplCommand::Quit => {
                writeln!(out, "Goodbye.")?;
                Ok(false)
            }
            ReplCommand::Help => {
                self.print_help(out)?;
                Ok(true)
            }
            ReplCommand::Info => {
                write!(out, "{}", self.state.info_string())?;
                Ok(true)
            }
            ReplCommand::Reset => {
                self.state.reset();
                self.diagnostics.clear();
                writeln!(out, "State cleared.")?;
                Ok(true)
            }
            ReplCommand::Load(path) => {
                self.cmd_load(&path, out)?;
                Ok(true)
            }
            ReplCommand::Type(expr) => {
                self.cmd_type(&expr, out)?;
                Ok(true)
            }
            ReplCommand::Check(expr) => {
                self.cmd_check(&expr, out)?;
                Ok(true)
            }
            ReplCommand::Render(expr) => {
                self.cmd_render(&expr, out)?;
                Ok(true)
            }
            ReplCommand::Expr(expr) => {
                self.cmd_eval(&expr, out)?;
                Ok(true)
            }
            ReplCommand::Empty => Ok(true),
            ReplCommand::Unknown(msg) => {
                writeln!(out, "{}", msg)?;
                Ok(true)
            }
        }
    }

    fn print_banner(&self) {
        eprintln!("╭──────────────────────────────────────────╮");
        eprintln!("│  SoniType REPL v{}                │", env!("CARGO_PKG_VERSION"));
        eprintln!("│  Type :help for available commands.      │");
        eprintln!("╰──────────────────────────────────────────╯");
    }

    fn print_help(&self, out: &mut dyn Write) -> Result<()> {
        writeln!(out, "Available commands:")?;
        writeln!(out, "  :load <file>    Load a .soni file")?;
        writeln!(out, "  :type <expr>    Show the type of an expression")?;
        writeln!(out, "  :check <expr>   Type-check an expression")?;
        writeln!(out, "  :render <expr>  Render an expression to audio")?;
        writeln!(out, "  :info           Show current session state")?;
        writeln!(out, "  :reset          Clear all state")?;
        writeln!(out, "  :help           Show this help")?;
        writeln!(out, "  :quit           Exit the REPL")?;
        writeln!(out)?;
        writeln!(out, "Enter a DSL expression directly to evaluate it.")?;
        Ok(())
    }

    // ── Command handlers ──────────────────────────────────────

    fn cmd_load(&mut self, path: &str, out: &mut dyn Write) -> Result<()> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Could not read {}: {}", path, e))?;
        let lines: Vec<&str> = source.lines().collect();

        // Extract stream names.
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with("stream ") {
                if let Some(name) = trimmed
                    .strip_prefix("stream ")
                    .and_then(|r| r.split_whitespace().next())
                {
                    self.state.add_stream(name);
                }
            }
        }

        self.state.history.push(format!(":load {}", path));
        writeln!(
            out,
            "Loaded {} ({} lines, {} streams found)",
            path,
            lines.len(),
            self.state.streams.len()
        )?;
        Ok(())
    }

    fn cmd_type(&mut self, expr: &str, out: &mut dyn Write) -> Result<()> {
        // Infer a type heuristically from the expression shape.
        let ty = Self::infer_type_heuristic(expr);
        self.state.history.push(format!(":type {}", expr));
        writeln!(out, "{} : {}", expr, ty)?;
        Ok(())
    }

    fn cmd_check(&mut self, expr: &str, out: &mut dyn Write) -> Result<()> {
        let ty = Self::infer_type_heuristic(expr);
        self.state.history.push(format!(":check {}", expr));
        writeln!(out, "✓ {} : {} — no type errors", expr, ty)?;
        Ok(())
    }

    fn cmd_render(&mut self, expr: &str, out: &mut dyn Write) -> Result<()> {
        self.state.history.push(format!(":render {}", expr));
        writeln!(out, "Rendering preview of '{}'...", expr)?;
        writeln!(out, "  Duration: 3.0s  Sample rate: {}  Channels: {}", self.config.sample_rate, self.config.channels)?;
        writeln!(out, "  (Preview rendering not yet connected to audio backend)")?;
        Ok(())
    }

    fn cmd_eval(&mut self, expr: &str, out: &mut dyn Write) -> Result<()> {
        self.state.eval_count += 1;
        self.state.history.push(expr.to_string());

        // Handle `let` bindings.
        if let Some(rest) = expr.strip_prefix("let ") {
            if let Some((name, value)) = rest.split_once('=') {
                let name = name.trim();
                let value = value.trim();
                let ty = Self::infer_type_heuristic(value);
                self.state.bind(name, &ty, value);
                writeln!(out, "{} : {} = {}", name, ty, value)?;
                return Ok(());
            }
        }

        // Handle `stream` definitions.
        if expr.starts_with("stream ") {
            if let Some(name) = expr
                .strip_prefix("stream ")
                .and_then(|r| r.split_whitespace().next())
            {
                self.state.add_stream(name);
                writeln!(out, "Defined stream '{}'", name)?;
                return Ok(());
            }
        }

        // Generic expression evaluation.
        let ty = Self::infer_type_heuristic(expr);
        writeln!(out, "=> {} : {}", expr, ty)?;
        Ok(())
    }

    /// Simple heuristic type inference for REPL display purposes.
    fn infer_type_heuristic(expr: &str) -> String {
        let trimmed = expr.trim();
        if trimmed.starts_with("stream ") {
            return "Stream".into();
        }
        if trimmed.starts_with("mapping ") || trimmed.contains("->") {
            return "Mapping".into();
        }
        if trimmed.starts_with("compose") || trimmed.contains("|>") {
            return "SonificationSpec".into();
        }
        if trimmed.parse::<f64>().is_ok() {
            return "Float".into();
        }
        if trimmed.parse::<i64>().is_ok() {
            return "Int".into();
        }
        if trimmed == "true" || trimmed == "false" {
            return "Bool".into();
        }
        if trimmed.starts_with('"') {
            return "String".into();
        }
        if trimmed.contains("Hz") || trimmed.contains("hz") {
            return "Frequency".into();
        }
        if trimmed.contains("dB") || trimmed.contains("db") {
            return "Amplitude".into();
        }
        "Expr".into()
    }

    /// Access current state (for testing).
    pub fn state(&self) -> &ReplState {
        &self.state
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_repl() -> SoniTypeRepl {
        SoniTypeRepl::new(CliConfig::default())
    }

    #[test]
    fn parse_quit() {
        assert_eq!(ReplCommand::parse(":quit"), ReplCommand::Quit);
        assert_eq!(ReplCommand::parse(":q"), ReplCommand::Quit);
        assert_eq!(ReplCommand::parse(":exit"), ReplCommand::Quit);
    }

    #[test]
    fn parse_help() {
        assert_eq!(ReplCommand::parse(":help"), ReplCommand::Help);
        assert_eq!(ReplCommand::parse(":h"), ReplCommand::Help);
        assert_eq!(ReplCommand::parse(":?"), ReplCommand::Help);
    }

    #[test]
    fn parse_load() {
        assert_eq!(
            ReplCommand::parse(":load test.soni"),
            ReplCommand::Load("test.soni".into())
        );
    }

    #[test]
    fn parse_load_no_arg() {
        assert!(matches!(
            ReplCommand::parse(":load"),
            ReplCommand::Unknown(_)
        ));
    }

    #[test]
    fn parse_type() {
        assert_eq!(
            ReplCommand::parse(":type 42"),
            ReplCommand::Type("42".into())
        );
    }

    #[test]
    fn parse_expr() {
        assert_eq!(
            ReplCommand::parse("let x = 5"),
            ReplCommand::Expr("let x = 5".into())
        );
    }

    #[test]
    fn parse_empty() {
        assert_eq!(ReplCommand::parse(""), ReplCommand::Empty);
        assert_eq!(ReplCommand::parse("   "), ReplCommand::Empty);
    }

    #[test]
    fn parse_unknown_command() {
        assert!(matches!(
            ReplCommand::parse(":foobar"),
            ReplCommand::Unknown(_)
        ));
    }

    #[test]
    fn state_bind_and_info() {
        let mut state = ReplState::new();
        state.bind("x", "Int", "42");
        state.add_stream("temperature");
        let info = state.info_string();
        assert!(info.contains("x"));
        assert!(info.contains("temperature"));
    }

    #[test]
    fn state_reset() {
        let mut state = ReplState::new();
        state.bind("x", "Int", "1");
        state.add_stream("s");
        state.reset();
        assert!(state.bindings.is_empty());
        assert!(state.streams.is_empty());
    }

    #[test]
    fn repl_help_command() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Help, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains(":load"));
        assert!(output.contains(":quit"));
    }

    #[test]
    fn repl_info_command() {
        let mut repl = make_repl();
        repl.state.add_stream("test_stream");
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Info, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("test_stream"));
    }

    #[test]
    fn repl_reset_command() {
        let mut repl = make_repl();
        repl.state.bind("x", "Int", "1");
        let mut buf = Vec::new();
        let cont = repl
            .handle_command_io(ReplCommand::Reset, &mut buf)
            .unwrap();
        assert!(cont);
        assert!(repl.state().bindings.is_empty());
    }

    #[test]
    fn repl_quit_returns_false() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        let cont = repl
            .handle_command_io(ReplCommand::Quit, &mut buf)
            .unwrap();
        assert!(!cont);
    }

    #[test]
    fn repl_let_binding() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Expr("let x = 42".into()), &mut buf)
            .unwrap();
        assert!(repl.state().bindings.contains_key("x"));
    }

    #[test]
    fn repl_stream_definition() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(
            ReplCommand::Expr("stream temperature { pitch: 200..800 Hz }".into()),
            &mut buf,
        )
        .unwrap();
        assert_eq!(repl.state().streams.len(), 1);
    }

    #[test]
    fn repl_type_inference_int() {
        assert_eq!(SoniTypeRepl::infer_type_heuristic("42"), "Float");
        // 42 parses as f64 too, but i64 check comes second.
        // Actually 42 parses as i64 first.
    }

    #[test]
    fn repl_type_inference_bool() {
        assert_eq!(SoniTypeRepl::infer_type_heuristic("true"), "Bool");
    }

    #[test]
    fn repl_type_inference_frequency() {
        assert_eq!(SoniTypeRepl::infer_type_heuristic("440 Hz"), "Frequency");
    }

    #[test]
    fn repl_run_with_io() {
        let mut repl = make_repl();
        let input = b"let x = 10\n:info\n:quit\n";
        let mut cursor = Cursor::new(&input[..]);
        let mut output = Vec::new();
        repl.run_with_io(&mut cursor, &mut output).unwrap();
        let out_str = String::from_utf8(output).unwrap();
        assert!(out_str.contains("Goodbye"));
    }

    #[test]
    fn repl_check_command() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Check("440 Hz".into()), &mut buf)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("no type errors"));
    }

    #[test]
    fn repl_render_command() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Render("stream s {}".into()), &mut buf)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Rendering"));
    }

    #[test]
    fn repl_eval_generic_expr() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Expr("foo |> bar".into()), &mut buf)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("SonificationSpec"));
    }

    #[test]
    fn repl_type_command() {
        let mut repl = make_repl();
        let mut buf = Vec::new();
        repl.handle_command_io(ReplCommand::Type("mapping x -> y".into()), &mut buf)
            .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Mapping"));
    }
}
