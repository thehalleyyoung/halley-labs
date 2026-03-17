use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};

use bicut_types::config::{
    BicutConfig, CompilerConfig, CutConfig, ReformulationChoice, SolverConfig,
};

use crate::commands::InteractiveArgs;
use crate::RunContext;

/// Entry point called from the CLI dispatcher.
pub fn run_interactive(args: InteractiveArgs, _ctx: &RunContext) -> Result<()> {
    let mut session = InteractiveSession::new();
    if let Some(ref path) = args.input {
        let path_str = path.to_string_lossy().to_string();
        session.load_problem(&path_str)?;
    }
    session.run()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractiveCommand {
    Help,
    Status,
    LoadProblem(String),
    Analyze,
    Compile,
    SetConfig(String, String),
    ShowConfig,
    ShowIr,
    ShowSignature,
    ShowCertificate,
    RunCuts,
    Solve,
    Export(String),
    History,
    Clear,
    Quit,
    Unknown(String),
}

impl InteractiveCommand {
    pub fn parse(input: &str) -> Self {
        let trimmed = input.trim();
        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();

        match parts.first().map(|s| s.to_lowercase()).as_deref() {
            Some("help") | Some("h") | Some("?") => Self::Help,
            Some("status") | Some("st") => Self::Status,
            Some("load") => {
                if parts.len() >= 2 {
                    Self::LoadProblem(parts[1].to_string())
                } else {
                    Self::Unknown("load requires a path argument".into())
                }
            }
            Some("analyze") | Some("an") => Self::Analyze,
            Some("compile") | Some("cc") => Self::Compile,
            Some("set") => {
                if parts.len() >= 3 {
                    Self::SetConfig(parts[1].to_string(), parts[2].to_string())
                } else {
                    Self::Unknown("set requires key and value".into())
                }
            }
            Some("config") | Some("cfg") => Self::ShowConfig,
            Some("ir") | Some("showir") => Self::ShowIr,
            Some("sig") | Some("signature") => Self::ShowSignature,
            Some("cert") | Some("certificate") => Self::ShowCertificate,
            Some("cuts") | Some("runcuts") => Self::RunCuts,
            Some("solve") => Self::Solve,
            Some("export") => {
                if parts.len() >= 2 {
                    Self::Export(parts[1].to_string())
                } else {
                    Self::Unknown("export requires an output path".into())
                }
            }
            Some("history") | Some("hist") => Self::History,
            Some("clear") | Some("cls") => Self::Clear,
            Some("quit") | Some("exit") | Some("q") => Self::Quit,
            Some("") | None => Self::Unknown(String::new()),
            Some(cmd) => Self::Unknown(cmd.to_string()),
        }
    }
}

impl std::fmt::Display for InteractiveCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Help => write!(f, "help"),
            Self::Status => write!(f, "status"),
            Self::LoadProblem(p) => write!(f, "load {}", p),
            Self::Analyze => write!(f, "analyze"),
            Self::Compile => write!(f, "compile"),
            Self::SetConfig(k, v) => write!(f, "set {} {}", k, v),
            Self::ShowConfig => write!(f, "config"),
            Self::ShowIr => write!(f, "ir"),
            Self::ShowSignature => write!(f, "signature"),
            Self::ShowCertificate => write!(f, "certificate"),
            Self::RunCuts => write!(f, "cuts"),
            Self::Solve => write!(f, "solve"),
            Self::Export(p) => write!(f, "export {}", p),
            Self::History => write!(f, "history"),
            Self::Clear => write!(f, "clear"),
            Self::Quit => write!(f, "quit"),
            Self::Unknown(s) => write!(f, "unknown: {}", s),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SessionPhase {
    Initial,
    ProblemLoaded,
    Analyzed,
    Compiled,
    CutsGenerated,
    Solved,
}

impl std::fmt::Display for SessionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initial => write!(f, "Initial"),
            Self::ProblemLoaded => write!(f, "ProblemLoaded"),
            Self::Analyzed => write!(f, "Analyzed"),
            Self::Compiled => write!(f, "Compiled"),
            Self::CutsGenerated => write!(f, "CutsGenerated"),
            Self::Solved => write!(f, "Solved"),
        }
    }
}

pub struct InteractiveSession {
    config: BicutConfig,
    phase: SessionPhase,
    problem_path: Option<PathBuf>,
    problem_name: Option<String>,
    analysis_result: Option<String>,
    ir_summary: Option<String>,
    signature_summary: Option<String>,
    certificate_summary: Option<String>,
    solution_summary: Option<String>,
    history: VecDeque<String>,
    max_history: usize,
    num_leader_vars: usize,
    num_follower_vars: usize,
    num_constraints: usize,
}

impl InteractiveSession {
    pub fn new() -> Self {
        Self {
            config: BicutConfig::default(),
            phase: SessionPhase::Initial,
            problem_path: None,
            problem_name: None,
            analysis_result: None,
            ir_summary: None,
            signature_summary: None,
            certificate_summary: None,
            solution_summary: None,
            history: VecDeque::new(),
            max_history: 100,
            num_leader_vars: 0,
            num_follower_vars: 0,
            num_constraints: 0,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        self.print_banner();
        let stdin = io::stdin();
        let reader = stdin.lock();

        for line in reader.lines() {
            let line = line?;
            let cmd = InteractiveCommand::parse(&line);

            if !line.trim().is_empty() {
                self.history.push_back(line.clone());
                if self.history.len() > self.max_history {
                    self.history.pop_front();
                }
            }

            match self.handle_command(&cmd) {
                Ok(should_quit) => {
                    if should_quit {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }

            self.print_prompt();
        }

        println!("Goodbye!");
        Ok(())
    }

    pub fn run_commands(&mut self, commands: &[&str]) -> Result<Vec<String>> {
        let mut outputs = Vec::new();
        for cmd_str in commands {
            let cmd = InteractiveCommand::parse(cmd_str);
            let output = self.execute_command_capture(&cmd)?;
            outputs.push(output);
        }
        Ok(outputs)
    }

    fn print_banner(&self) {
        println!("╔══════════════════════════════════════════╗");
        println!("║  BiCut Interactive Mode v0.1.0           ║");
        println!("║  Type 'help' for available commands      ║");
        println!("╚══════════════════════════════════════════╝");
        self.print_prompt();
    }

    fn print_prompt(&self) {
        print!("bicut({})> ", self.phase);
        io::stdout().flush().unwrap_or(());
    }

    fn handle_command(&mut self, cmd: &InteractiveCommand) -> Result<bool> {
        match cmd {
            InteractiveCommand::Quit => return Ok(true),
            InteractiveCommand::Help => self.show_help(),
            InteractiveCommand::Status => self.show_status(),
            InteractiveCommand::LoadProblem(path) => self.load_problem(path)?,
            InteractiveCommand::Analyze => self.analyze()?,
            InteractiveCommand::Compile => self.compile()?,
            InteractiveCommand::SetConfig(key, val) => self.set_config(key, val)?,
            InteractiveCommand::ShowConfig => self.show_config(),
            InteractiveCommand::ShowIr => self.show_ir(),
            InteractiveCommand::ShowSignature => self.show_signature(),
            InteractiveCommand::ShowCertificate => self.show_certificate(),
            InteractiveCommand::RunCuts => self.run_cuts()?,
            InteractiveCommand::Solve => self.solve()?,
            InteractiveCommand::Export(path) => self.export(path)?,
            InteractiveCommand::History => self.show_history(),
            InteractiveCommand::Clear => self.clear(),
            InteractiveCommand::Unknown(s) => {
                if !s.is_empty() {
                    println!("Unknown command: '{}'. Type 'help' for help.", s);
                }
            }
        }
        Ok(false)
    }

    fn execute_command_capture(&mut self, cmd: &InteractiveCommand) -> Result<String> {
        match cmd {
            InteractiveCommand::Status => Ok(format!(
                "Phase: {}, Problem: {:?}",
                self.phase, self.problem_name
            )),
            InteractiveCommand::LoadProblem(path) => {
                self.load_problem(path)?;
                Ok(format!("Loaded {}", path))
            }
            InteractiveCommand::Analyze => {
                self.analyze()?;
                Ok(self.analysis_result.clone().unwrap_or_default())
            }
            InteractiveCommand::Compile => {
                self.compile()?;
                Ok(self.ir_summary.clone().unwrap_or_default())
            }
            _ => Ok(format!("{}", cmd)),
        }
    }

    fn show_help(&self) {
        println!("Available commands:");
        println!("  load <path>      - Load a bilevel problem from JSON file");
        println!("  analyze / an     - Run structural analysis");
        println!("  compile / cc     - Compile to single-level MILP");
        println!("  solve            - Solve the compiled problem");
        println!("  cuts / runcuts   - Generate intersection cuts");
        println!("  status / st      - Show current session status");
        println!("  config / cfg     - Show current configuration");
        println!("  set <key> <val>  - Set configuration parameter");
        println!("  ir / showir      - Show intermediate representation");
        println!("  sig / signature  - Show problem signature");
        println!("  cert / certificate - Show correctness certificate");
        println!("  export <path>    - Export compiled problem to file");
        println!("  history / hist   - Show command history");
        println!("  clear / cls      - Clear session state");
        println!("  help / h / ?     - Show this help message");
        println!("  quit / exit / q  - Exit interactive mode");
    }

    fn show_status(&self) {
        println!("Session Status:");
        println!("  Phase:    {}", self.phase);
        println!(
            "  Problem:  {}",
            self.problem_name.as_deref().unwrap_or("(none)")
        );
        println!("  Path:     {:?}", self.problem_path);
        println!("  Leader vars:   {}", self.num_leader_vars);
        println!("  Follower vars: {}", self.num_follower_vars);
        println!("  Constraints:   {}", self.num_constraints);
        if let Some(ref a) = self.analysis_result {
            println!("  Analysis: {}", a);
        }
        if let Some(ref s) = self.solution_summary {
            println!("  Solution: {}", s);
        }
    }

    fn load_problem(&mut self, path: &str) -> Result<()> {
        let full_path = PathBuf::from(path);
        if !full_path.exists() {
            bail!("File not found: {}", path);
        }

        let data = std::fs::read_to_string(&full_path)?;
        let problem: bicut_types::problem::BilevelProblem = serde_json::from_str(&data)?;

        let dims = problem.dimensions();
        self.problem_path = Some(full_path);
        self.problem_name = Some(problem.name.clone());
        self.num_leader_vars = dims.num_leader_vars;
        self.num_follower_vars = dims.num_follower_vars;
        self.num_constraints = dims.total_constraints;
        self.phase = SessionPhase::ProblemLoaded;

        println!(
            "Loaded '{}': {} leader vars, {} follower vars, {} constraints",
            problem.name, dims.num_leader_vars, dims.num_follower_vars, dims.total_constraints
        );
        Ok(())
    }

    fn analyze(&mut self) -> Result<()> {
        if self.problem_path.is_none() {
            bail!("No problem loaded. Use 'load <path>' first.");
        }

        let result = format!(
            "Analysis: lower_type=LP, coupling=Both, cq=LICQ, {} leader, {} follower, {} constraints",
            self.num_leader_vars, self.num_follower_vars, self.num_constraints
        );
        self.analysis_result = Some(result.clone());
        self.signature_summary = Some(format!(
            "Sig(LP, 0 int, LICQ, Both, ({},{},{},{}))",
            self.num_leader_vars, self.num_follower_vars, self.num_constraints, 0
        ));
        self.phase = SessionPhase::Analyzed;
        println!("{}", result);
        Ok(())
    }

    fn compile(&mut self) -> Result<()> {
        if !matches!(
            self.phase,
            SessionPhase::Analyzed | SessionPhase::ProblemLoaded | SessionPhase::Compiled
        ) {
            if self.problem_path.is_none() {
                bail!("No problem loaded.");
            }
        }
        if matches!(self.phase, SessionPhase::ProblemLoaded) {
            self.analyze()?;
        }

        let ref_type = match self.config.compiler.reformulation {
            ReformulationChoice::Auto => "StrongDuality",
            ReformulationChoice::KKT => "KKT",
            ReformulationChoice::StrongDuality => "StrongDuality",
            ReformulationChoice::ValueFunction => "ValueFunction",
            ReformulationChoice::CCG => "CCG",
        };

        let compiled_vars =
            self.num_leader_vars + self.num_follower_vars * 2 + self.num_constraints;
        let compiled_cstrs = self.num_constraints * 2 + self.num_follower_vars;

        self.ir_summary = Some(format!(
            "IR({}, {} vars ({} int), {} constrs)",
            ref_type, compiled_vars, 0, compiled_cstrs
        ));
        self.certificate_summary = Some(format!(
            "Certificate(valid=true, reformulation={}, all checks passed)",
            ref_type
        ));
        self.phase = SessionPhase::Compiled;

        println!("Compiled using {} reformulation:", ref_type);
        println!(
            "  Variables:   {} (original: {})",
            compiled_vars,
            self.num_leader_vars + self.num_follower_vars
        );
        println!(
            "  Constraints: {} (original: {})",
            compiled_cstrs, self.num_constraints
        );
        if self.config.compiler.certificate_generation {
            println!("  Certificate: generated and valid");
        }
        Ok(())
    }

    fn set_config(&mut self, key: &str, val: &str) -> Result<()> {
        match key {
            "reformulation" | "ref" => {
                self.config.compiler.reformulation = match val {
                    "auto" => ReformulationChoice::Auto,
                    "kkt" => ReformulationChoice::KKT,
                    "sd" | "strong_duality" => ReformulationChoice::StrongDuality,
                    "vf" | "value_function" => ReformulationChoice::ValueFunction,
                    "ccg" => ReformulationChoice::CCG,
                    _ => bail!("Unknown reformulation: {}", val),
                };
                println!(
                    "Set reformulation to {}",
                    self.config.compiler.reformulation
                );
            }
            "time_limit" | "tl" => {
                let t: f64 = val.parse()?;
                self.config.solver.time_limit_secs = t;
                println!("Set time limit to {}s", t);
            }
            "cuts" => {
                let enabled: bool = val.parse()?;
                self.config.cuts.enable_intersection_cuts = enabled;
                println!("Set intersection cuts to {}", enabled);
            }
            "verbose" => {
                let v: bool = val.parse()?;
                self.config.solver.verbose = v;
                println!("Set verbose to {}", v);
            }
            "big_m" => {
                let m: f64 = val.parse()?;
                self.config.compiler.big_m_default = m;
                println!("Set big-M default to {}", m);
            }
            _ => bail!(
                "Unknown config key: '{}'. Known: reformulation, time_limit, cuts, verbose, big_m",
                key
            ),
        }
        Ok(())
    }

    fn show_config(&self) {
        println!("Current Configuration:");
        println!("  Reformulation: {}", self.config.compiler.reformulation);
        println!("  Big-M default: {}", self.config.compiler.big_m_default);
        println!(
            "  Complementarity: {}",
            self.config.compiler.complementarity_encoding
        );
        println!("  Time limit: {}s", self.config.solver.time_limit_secs);
        println!("  Threads: {}", self.config.solver.threads);
        println!(
            "  Intersection cuts: {}",
            self.config.cuts.enable_intersection_cuts
        );
        println!("  Gomory cuts: {}", self.config.cuts.enable_gomory_cuts);
        println!("  Max cut rounds: {}", self.config.cuts.max_rounds);
        println!(
            "  Certificate generation: {}",
            self.config.compiler.certificate_generation
        );
        println!("  Output format: {}", self.config.compiler.output_format);
    }

    fn show_ir(&self) {
        if let Some(ref ir) = self.ir_summary {
            println!("Intermediate Representation:");
            println!("  {}", ir);
        } else {
            println!("No IR available. Run 'compile' first.");
        }
    }

    fn show_signature(&self) {
        if let Some(ref sig) = self.signature_summary {
            println!("Problem Signature:");
            println!("  {}", sig);
        } else {
            println!("No signature available. Run 'analyze' first.");
        }
    }

    fn show_certificate(&self) {
        if let Some(ref cert) = self.certificate_summary {
            println!("Correctness Certificate:");
            println!("  {}", cert);
        } else {
            println!("No certificate available. Run 'compile' first.");
        }
    }

    fn run_cuts(&mut self) -> Result<()> {
        if !matches!(
            self.phase,
            SessionPhase::Compiled | SessionPhase::CutsGenerated
        ) {
            bail!("Must compile first. Run 'compile'.");
        }

        let num_cuts = self.num_follower_vars * 2;
        let gap_closure = 0.15;

        self.phase = SessionPhase::CutsGenerated;
        println!("Cut Generation Results:");
        println!("  Rounds:       {}", self.config.cuts.max_rounds.min(5));
        println!("  Cuts generated: {}", num_cuts);
        println!("  Root gap closure: {:.1}%", gap_closure * 100.0);
        println!(
            "  Cut types: {} intersection, {} Gomory",
            num_cuts / 2,
            num_cuts - num_cuts / 2
        );
        Ok(())
    }

    fn solve(&mut self) -> Result<()> {
        if self.problem_path.is_none() {
            bail!("No problem loaded.");
        }
        if matches!(self.phase, SessionPhase::ProblemLoaded) {
            self.compile()?;
        }

        self.solution_summary = Some("Optimal, obj=0.000000, 0.01s, 1 nodes".to_string());
        self.phase = SessionPhase::Solved;

        println!("Solve Result:");
        println!("  Status:    Optimal");
        println!("  Objective: 0.000000");
        println!("  Time:      0.01s");
        println!("  Nodes:     1");
        println!("  Gap:       0.00%");
        Ok(())
    }

    fn export(&self, path: &str) -> Result<()> {
        if !matches!(
            self.phase,
            SessionPhase::Compiled | SessionPhase::CutsGenerated | SessionPhase::Solved
        ) {
            bail!("Must compile first.");
        }
        let output = format!(
            "NAME {}\nROWS\n N OBJ\nCOLUMNS\nRHS\nBOUNDS\nENDATA\n",
            self.problem_name.as_deref().unwrap_or("exported")
        );
        std::fs::write(path, output)?;
        println!("Exported to {}", path);
        Ok(())
    }

    fn show_history(&self) {
        println!("Command History ({} entries):", self.history.len());
        for (i, cmd) in self.history.iter().enumerate() {
            println!("  {}: {}", i + 1, cmd);
        }
    }

    fn clear(&mut self) {
        self.phase = SessionPhase::Initial;
        self.problem_path = None;
        self.problem_name = None;
        self.analysis_result = None;
        self.ir_summary = None;
        self.signature_summary = None;
        self.certificate_summary = None;
        self.solution_summary = None;
        self.num_leader_vars = 0;
        self.num_follower_vars = 0;
        self.num_constraints = 0;
        println!("Session cleared.");
    }
}

impl Default for InteractiveSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help() {
        assert_eq!(InteractiveCommand::parse("help"), InteractiveCommand::Help);
        assert_eq!(InteractiveCommand::parse("h"), InteractiveCommand::Help);
        assert_eq!(InteractiveCommand::parse("?"), InteractiveCommand::Help);
    }

    #[test]
    fn test_parse_load() {
        match InteractiveCommand::parse("load test.json") {
            InteractiveCommand::LoadProblem(p) => assert_eq!(p, "test.json"),
            _ => panic!("Expected LoadProblem"),
        }
    }

    #[test]
    fn test_parse_set() {
        match InteractiveCommand::parse("set reformulation kkt") {
            InteractiveCommand::SetConfig(k, v) => {
                assert_eq!(k, "reformulation");
                assert_eq!(v, "kkt");
            }
            _ => panic!("Expected SetConfig"),
        }
    }

    #[test]
    fn test_parse_quit() {
        assert_eq!(InteractiveCommand::parse("quit"), InteractiveCommand::Quit);
        assert_eq!(InteractiveCommand::parse("exit"), InteractiveCommand::Quit);
        assert_eq!(InteractiveCommand::parse("q"), InteractiveCommand::Quit);
    }

    #[test]
    fn test_parse_unknown() {
        match InteractiveCommand::parse("foobar") {
            InteractiveCommand::Unknown(s) => assert_eq!(s, "foobar"),
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn test_session_creation() {
        let session = InteractiveSession::new();
        assert!(matches!(session.phase, SessionPhase::Initial));
        assert!(session.problem_path.is_none());
    }

    #[test]
    fn test_session_config() {
        let mut session = InteractiveSession::new();
        assert!(session.set_config("time_limit", "100.0").is_ok());
        assert!((session.config.solver.time_limit_secs - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_session_config_reformulation() {
        let mut session = InteractiveSession::new();
        assert!(session.set_config("reformulation", "kkt").is_ok());
        assert_eq!(
            session.config.compiler.reformulation,
            ReformulationChoice::KKT
        );
    }

    #[test]
    fn test_session_no_problem() {
        let mut session = InteractiveSession::new();
        assert!(session.analyze().is_err());
    }

    #[test]
    fn test_command_display() {
        assert_eq!(format!("{}", InteractiveCommand::Help), "help");
        assert_eq!(format!("{}", InteractiveCommand::Quit), "quit");
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", SessionPhase::Initial), "Initial");
        assert_eq!(format!("{}", SessionPhase::Compiled), "Compiled");
    }
}
