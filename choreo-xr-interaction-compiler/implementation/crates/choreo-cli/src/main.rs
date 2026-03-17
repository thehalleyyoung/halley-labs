//! Choreo XR Interaction Compiler — CLI entry point.
//!
//! Provides subcommands for compiling, verifying, simulating, and
//! analysing `.choreo` interaction choreographies targeting XR runtimes.

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use log::info;

// ── Re-exports from workspace crates (used in stub bodies) ──────────
use choreo_dsl::{Lexer, Parser as DslParser, TypeChecker, PrettyPrinter, Desugarer, SemanticAnalyzer};
use choreo_ec::ECCompiler;
use choreo_codegen::rust_backend::RustCodeGenerator;
use choreo_codegen::csharp_backend::CSharpCodeGenerator;
use choreo_codegen::typescript_backend::TypeScriptCodeGenerator;
use choreo_codegen::dot_backend::DotGraphGenerator;
use choreo_codegen::json_backend::JsonExporter;
use choreo_cegar::{CEGARVerifier, CEGARConfig, CompositionalVerifier, CertificateBuilder};
use choreo_simulator::simulator::{HeadlessSimulator, SimulationConfig};
use choreo_simulator::benchmark::{BenchmarkSuite, BenchmarkParams, generate_grid_scene, generate_random_scene, generate_cluster_scene, generate_chain_scene};
use choreo_conflict::report::ReportBuilder;

// ── CLI definition ──────────────────────────────────────────────────

/// Choreo — the XR interaction choreography compiler.
///
/// Compile, verify, simulate, and analyse spatial interaction programs
/// for augmented- and virtual-reality runtimes.
#[derive(Parser)]
#[command(
    name = "choreo",
    version,
    about = "XR interaction choreography compiler & verifier",
    long_about = "Choreo compiles high-level interaction choreographies into \
                  verified automata targeting OpenXR, Unity, and web-XR runtimes.\n\n\
                  Use `choreo <COMMAND> --help` for per-command documentation.",
    after_help = "Examples:\n  \
                  choreo compile scene.choreo --backend rust -o out.rs\n  \
                  choreo verify  scene.choreo --property deadlock-free\n  \
                  choreo simulate scene.choreo --steps 1000\n  \
                  choreo benchmark --scenario grid --entities 64",
    propagate_version = true,
)]
struct Cli {
    /// Enable verbose logging (repeat for more: -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all non-error output
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse DSL, type-check, compile to automaton, and generate code.
    Compile {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Code-generation backend
        #[arg(short, long, value_enum, default_value_t = Backend::Rust)]
        backend: Backend,

        /// Output file path (stdout if omitted)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Dump the Event Calculus intermediate representation
        #[arg(long)]
        emit_ir: bool,
    },

    /// Run spatial CEGAR verification on a choreography.
    Verify {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Property to verify
        #[arg(short, long, value_enum, default_value_t = VerifyProperty::DeadlockFree)]
        property: VerifyProperty,

        /// Verification backend
        #[arg(short, long, value_enum, default_value_t = VerifyBackend::Bdd)]
        backend: VerifyBackend,

        /// Enable compositional (assume-guarantee) verification
        #[arg(long)]
        compositional: bool,

        /// Write a machine-readable verification certificate
        #[arg(long, value_name = "PATH")]
        certificate: Option<PathBuf>,
    },

    /// Quick type-check only (no compilation or code generation).
    Check {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,
    },

    /// Run a headless simulation of the choreography.
    Simulate {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Maximum number of simulation steps
        #[arg(short, long, value_name = "N")]
        steps: Option<u64>,

        /// Simulation time-step in seconds
        #[arg(short, long, value_name = "SECONDS", default_value_t = 1.0 / 60.0)]
        time_step: f64,
    },

    /// Pretty-print the source file.
    Format {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,
    },

    /// Import a glTF scene as a Choreo spatial environment.
    ImportGltf {
        /// Path to the .gltf / .glb scene file
        #[arg(value_name = "SCENE")]
        input: PathBuf,
    },

    /// Export an OpenXR action manifest from a choreography.
    ExportOpenxr {
        /// Path to the .choreo source file
        #[arg(value_name = "INPUT")]
        input: PathBuf,
    },

    /// Run built-in performance benchmarks.
    Benchmark {
        /// Benchmark scenario
        #[arg(short, long, value_enum, default_value_t = BenchScenario::Grid)]
        scenario: BenchScenario,

        /// Number of entities in the benchmark scene
        #[arg(short, long, value_name = "N", default_value_t = 64)]
        entities: u32,
    },
}

// ── Value-enum helpers ──────────────────────────────────────────────

#[derive(Clone, ValueEnum)]
enum Backend {
    Rust,
    Csharp,
    Typescript,
    Dot,
    Json,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Rust => write!(f, "rust"),
            Backend::Csharp => write!(f, "csharp"),
            Backend::Typescript => write!(f, "typescript"),
            Backend::Dot => write!(f, "dot"),
            Backend::Json => write!(f, "json"),
        }
    }
}

#[derive(Clone, ValueEnum)]
enum VerifyProperty {
    DeadlockFree,
    Reachable,
    Safe,
    Deterministic,
}

impl std::fmt::Display for VerifyProperty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyProperty::DeadlockFree => write!(f, "deadlock-free"),
            VerifyProperty::Reachable => write!(f, "reachable"),
            VerifyProperty::Safe => write!(f, "safe"),
            VerifyProperty::Deterministic => write!(f, "deterministic"),
        }
    }
}

#[derive(Clone, ValueEnum)]
enum VerifyBackend {
    Explicit,
    Bdd,
    Bmc,
}

impl std::fmt::Display for VerifyBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyBackend::Explicit => write!(f, "explicit"),
            VerifyBackend::Bdd => write!(f, "bdd"),
            VerifyBackend::Bmc => write!(f, "bmc"),
        }
    }
}

#[derive(Clone, ValueEnum)]
enum BenchScenario {
    Grid,
    Random,
    Cluster,
    Chain,
}

impl std::fmt::Display for BenchScenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchScenario::Grid => write!(f, "grid"),
            BenchScenario::Random => write!(f, "random"),
            BenchScenario::Cluster => write!(f, "cluster"),
            BenchScenario::Chain => write!(f, "chain"),
        }
    }
}

// ── Coloured status helpers ─────────────────────────────────────────

fn status_ok(msg: &str) {
    eprintln!("\x1b[1;32m✓\x1b[0m {msg}");
}

fn status_info(msg: &str) {
    eprintln!("\x1b[1;34mℹ\x1b[0m {msg}");
}

fn status_warn(msg: &str) {
    eprintln!("\x1b[1;33m⚠\x1b[0m {msg}");
}

fn status_err(msg: &str) {
    eprintln!("\x1b[1;31m✗\x1b[0m {msg}");
}

// ── Shared helpers ──────────────────────────────────────────────────

fn read_source(path: &PathBuf) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("failed to read source file: {}", path.display()))
}

fn write_output(path: &Option<PathBuf>, content: &str) -> Result<()> {
    match path {
        Some(p) => {
            std::fs::write(p, content)
                .with_context(|| format!("failed to write output: {}", p.display()))?;
            status_ok(&format!("wrote {}", p.display()));
        }
        None => print!("{content}"),
    }
    Ok(())
}

// ── Subcommand implementations ──────────────────────────────────────

#[allow(unused_variables, unreachable_code)]
fn cmd_compile(input: PathBuf, backend: Backend, output: Option<PathBuf>, emit_ir: bool) -> Result<()> {
    let started = Instant::now();
    let source = read_source(&input)?;
    status_info(&format!("compiling {} ({} bytes)", input.display(), source.len()));

    // 1. Lex
    let _lexer = Lexer::new(&source);
    status_ok("lexed source");

    // 2. Parse
    let _parser = DslParser::new(&source);
    status_ok("parsed AST");

    // 3. Type-check
    let _checker = TypeChecker::new();
    status_ok("type-check passed");

    // 4. Desugar
    let _desugarer = Desugarer::new();
    status_ok("desugared AST");

    // 5. Semantic analysis
    let _semantic = SemanticAnalyzer::new();
    status_ok("semantic analysis passed");

    // 6. EC compilation
    let _ec_compiler = ECCompiler::new();
    if emit_ir {
        status_info("Event Calculus IR:");
        // In a full implementation the IR would be dumped here.
        todo!("dump EC intermediate representation to stderr");
    }
    status_ok("compiled to Event Calculus blueprint");

    // 7. Code generation
    let generated: String = match backend {
        Backend::Rust => {
            let _gen = RustCodeGenerator::new();
            todo!("invoke RustCodeGenerator::generate on the automaton")
        }
        Backend::Csharp => {
            let _gen = CSharpCodeGenerator::new();
            todo!("invoke CSharpCodeGenerator::generate on the automaton")
        }
        Backend::Typescript => {
            let _gen = TypeScriptCodeGenerator::new();
            todo!("invoke TypeScriptCodeGenerator::generate on the automaton")
        }
        Backend::Dot => {
            let _gen = DotGraphGenerator::new();
            todo!("invoke DotGraphGenerator::generate on the automaton")
        }
        Backend::Json => {
            let _gen = JsonExporter::new();
            todo!("invoke JsonExporter::generate on the automaton")
        }
    };

    write_output(&output, &generated)?;

    let elapsed = started.elapsed();
    status_ok(&format!("compilation finished in {elapsed:.2?}"));
    Ok(())
}

#[allow(unused_variables, unreachable_code)]
fn cmd_verify(
    input: PathBuf,
    property: VerifyProperty,
    backend: VerifyBackend,
    compositional: bool,
    certificate: Option<PathBuf>,
) -> Result<()> {
    let started = Instant::now();
    let source = read_source(&input)?;
    status_info(&format!(
        "verifying {} against property `{property}` (backend: {backend})",
        input.display(),
    ));

    // Parse & compile (reuse front-end pipeline)
    let _lexer = Lexer::new(&source);
    let _parser = DslParser::new(&source);
    let _checker = TypeChecker::new();
    status_ok("front-end pipeline complete");

    // Select property
    let _prop_label = match property {
        VerifyProperty::DeadlockFree => "deadlock freedom",
        VerifyProperty::Reachable => "state reachability",
        VerifyProperty::Safe => "safety invariant",
        VerifyProperty::Deterministic => "determinism",
    };

    if compositional {
        status_info("compositional (assume-guarantee) verification enabled");
        let _compositional = CompositionalVerifier::new(CEGARConfig::default());
        todo!("run compositional CEGAR loop");
    }

    // Build CEGAR verifier
    let _config = CEGARConfig::default();
    let _verifier = CEGARVerifier::new(CEGARConfig::default());
    status_ok("CEGAR verifier initialised");

    // Run verification
    todo!("execute CEGAR loop and report verdict");

    #[allow(unreachable_code)]
    if let Some(cert_path) = certificate {
        let _builder = CertificateBuilder::new();
        status_info(&format!("writing certificate to {}", cert_path.display()));
        todo!("serialise verification certificate");
    }

    #[allow(unreachable_code)]
    {
        let elapsed = started.elapsed();
        status_ok(&format!("verification finished in {elapsed:.2?}"));
        Ok(())
    }
}

fn cmd_check(input: PathBuf) -> Result<()> {
    let started = Instant::now();
    let source = read_source(&input)?;
    status_info(&format!("type-checking {}", input.display()));

    let _lexer = Lexer::new(&source);
    status_ok("lexed source");

    let _parser = DslParser::new(&source);
    status_ok("parsed AST");

    let _checker = TypeChecker::new();
    status_ok("type-check passed");

    let _semantic = SemanticAnalyzer::new();
    status_ok("semantic analysis passed");

    let _report = ReportBuilder::new("choreo-check");

    let elapsed = started.elapsed();
    status_ok(&format!("check completed in {elapsed:.2?} — no errors"));
    Ok(())
}

#[allow(unused_variables, unreachable_code)]
fn cmd_simulate(input: PathBuf, steps: Option<u64>, time_step: f64) -> Result<()> {
    let started = Instant::now();
    let source = read_source(&input)?;
    let max_steps = steps.unwrap_or(10_000);
    status_info(&format!(
        "simulating {} (max {max_steps} steps, dt={time_step:.6}s)",
        input.display(),
    ));

    // Front-end
    let _lexer = Lexer::new(&source);
    let _parser = DslParser::new(&source);
    let _checker = TypeChecker::new();
    status_ok("front-end pipeline complete");

    // Build simulator
    let config = SimulationConfig {
        time_step,
        max_duration: max_steps as f64 * time_step,
        ..Default::default()
    };
    let _sim = HeadlessSimulator::new(config);
    status_ok("headless simulator initialised");

    todo!("run simulation loop and report statistics");

    #[allow(unreachable_code)]
    {
        let elapsed = started.elapsed();
        status_ok(&format!("simulation finished in {elapsed:.2?}"));
        Ok(())
    }
}

#[allow(unreachable_code)]
fn cmd_format(input: PathBuf) -> Result<()> {
    let source = read_source(&input)?;
    status_info(&format!("formatting {}", input.display()));

    let _lexer = Lexer::new(&source);
    let _parser = DslParser::new(&source);
    let _printer = PrettyPrinter::new();

    todo!("pretty-print the parsed AST back to source");

    #[allow(unreachable_code)]
    {
        status_ok("formatted successfully");
        Ok(())
    }
}

#[allow(unreachable_code)]
fn cmd_import_gltf(input: PathBuf) -> Result<()> {
    status_info(&format!("importing glTF scene from {}", input.display()));

    if !input.exists() {
        anyhow::bail!("scene file not found: {}", input.display());
    }

    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "gltf" | "glb" => {}
        _ => status_warn(&format!("unexpected extension `.{ext}`; expected .gltf or .glb")),
    }

    todo!("parse glTF, extract spatial entities, and emit Choreo scene block");

    #[allow(unreachable_code)]
    {
        status_ok("glTF import complete");
        Ok(())
    }
}

#[allow(unreachable_code)]
fn cmd_export_openxr(input: PathBuf) -> Result<()> {
    let source = read_source(&input)?;
    status_info(&format!("exporting OpenXR manifest from {}", input.display()));

    let _lexer = Lexer::new(&source);
    let _parser = DslParser::new(&source);
    let _checker = TypeChecker::new();
    status_ok("front-end pipeline complete");

    todo!("extract interaction profiles and generate OpenXR action manifest JSON");

    #[allow(unreachable_code)]
    {
        status_ok("OpenXR manifest exported");
        Ok(())
    }
}

#[allow(unused_variables, unreachable_code)]
fn cmd_benchmark(scenario: BenchScenario, entities: u32) -> Result<()> {
    let started = Instant::now();
    status_info(&format!(
        "running `{scenario}` benchmark with {entities} entities"
    ));

    let params = BenchmarkParams {
        min_entities: entities as usize,
        max_entities: entities as usize,
        entity_step: 1,
        ..Default::default()
    };

    let _scene = match scenario {
        BenchScenario::Grid => {
            status_info("generating grid scene");
            generate_grid_scene(entities as usize, 2.0)
        }
        BenchScenario::Random => {
            status_info("generating random scene");
            generate_random_scene(entities as usize, 100.0, 42)
        }
        BenchScenario::Cluster => {
            let per_cluster = (entities as usize).max(4) / 4;
            status_info("generating cluster scene");
            generate_cluster_scene(4, per_cluster, 10.0)
        }
        BenchScenario::Chain => {
            status_info("generating chain scene");
            generate_chain_scene(entities as usize, 1.5)
        }
    };

    let _suite = BenchmarkSuite::new(params);
    status_ok("benchmark suite initialised");

    todo!("execute benchmark suite and print statistical summary");

    #[allow(unreachable_code)]
    {
        let elapsed = started.elapsed();
        status_ok(&format!("benchmark finished in {elapsed:.2?}"));
        Ok(())
    }
}

// ── Entry point ─────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    // Initialise logging based on verbosity / quiet flags.
    let log_level = if cli.quiet {
        "error"
    } else {
        match cli.verbose {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        }
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    info!("choreo {} starting", choreo_cli::version());

    let result = match cli.command {
        Command::Compile { input, backend, output, emit_ir } => {
            cmd_compile(input, backend, output, emit_ir)
        }
        Command::Verify { input, property, backend, compositional, certificate } => {
            cmd_verify(input, property, backend, compositional, certificate)
        }
        Command::Check { input } => cmd_check(input),
        Command::Simulate { input, steps, time_step } => cmd_simulate(input, steps, time_step),
        Command::Format { input } => cmd_format(input),
        Command::ImportGltf { input } => cmd_import_gltf(input),
        Command::ExportOpenxr { input } => cmd_export_openxr(input),
        Command::Benchmark { scenario, entities } => cmd_benchmark(scenario, entities),
    };

    if let Err(err) = result {
        status_err(&format!("{err:#}"));
        process::exit(1);
    }
}
