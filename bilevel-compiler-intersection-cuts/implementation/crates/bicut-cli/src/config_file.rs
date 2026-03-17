//! Configuration file handling for the BiCut CLI.
//!
//! Supports TOML-based configuration files with hierarchical defaults,
//! CLI argument merging, validation, and automatic discovery of
//! `~/.bicut/config.toml`.

use anyhow::{bail, Context, Result};
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Paths ──────────────────────────────────────────────────────────

/// Return the default configuration directory (`~/.bicut`).
pub fn default_config_dir() -> PathBuf {
    dirs_home().join(".bicut")
}

/// Return the default configuration file path.
pub fn default_config_path() -> PathBuf {
    default_config_dir().join("config.toml")
}

/// Best-effort home directory.
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Search for a config file by walking up the directory tree.
pub fn discover_config(start: &Path) -> Option<PathBuf> {
    let mut dir = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        let candidate = dir.join(".bicut").join("config.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        let candidate2 = dir.join("bicut.toml");
        if candidate2.is_file() {
            return Some(candidate2);
        }
        if !dir.pop() {
            break;
        }
    }
    let global = default_config_path();
    if global.is_file() {
        return Some(global);
    }
    None
}

// ── Top-level config ───────────────────────────────────────────────

/// Root configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BiCutConfig {
    /// Solver settings.
    pub solver: SolverConfig,

    /// Compiler settings.
    pub compiler: CompilerConfig,

    /// Cut generation settings.
    pub cuts: CutConfig,

    /// Benchmark settings.
    pub benchmark: BenchmarkConfig,

    /// Output settings.
    pub output: OutputConfig,

    /// Tolerance settings.
    pub tolerances: ToleranceConfig,

    /// Extra key-value pairs for extensions.
    #[serde(flatten)]
    pub extra: HashMap<String, toml::Value>,
}

impl Default for BiCutConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            compiler: CompilerConfig::default(),
            cuts: CutConfig::default(),
            benchmark: BenchmarkConfig::default(),
            output: OutputConfig::default(),
            tolerances: ToleranceConfig::default(),
            extra: HashMap::new(),
        }
    }
}

impl BiCutConfig {
    /// Validate the entire configuration, returning an error on the first
    /// invalid field.
    pub fn validate(&mut self) -> Result<()> {
        self.solver.validate().context("solver config")?;
        self.compiler.validate().context("compiler config")?;
        self.cuts.validate().context("cuts config")?;
        self.benchmark.validate().context("benchmark config")?;
        self.tolerances.validate().context("tolerance config")?;
        Ok(())
    }

    /// Merge another config on top of this one (non-default fields win).
    pub fn merge_override(&mut self, other: &BiCutConfig) {
        if other.solver.max_iterations != SolverConfig::default().max_iterations {
            self.solver.max_iterations = other.solver.max_iterations;
        }
        if other.solver.time_limit_secs != SolverConfig::default().time_limit_secs {
            self.solver.time_limit_secs = other.solver.time_limit_secs;
        }
        if other.solver.lp_solver != SolverConfig::default().lp_solver {
            self.solver.lp_solver.clone_from(&other.solver.lp_solver);
        }
        if other.compiler.reformulation != CompilerConfig::default().reformulation {
            self.compiler
                .reformulation
                .clone_from(&other.compiler.reformulation);
        }
        if other.compiler.big_m != CompilerConfig::default().big_m {
            self.compiler.big_m = other.compiler.big_m;
        }
        if other.compiler.presolve != CompilerConfig::default().presolve {
            self.compiler.presolve = other.compiler.presolve;
        }
        if other.cuts.max_rounds != CutConfig::default().max_rounds {
            self.cuts.max_rounds = other.cuts.max_rounds;
        }
        if other.cuts.max_cuts_per_round != CutConfig::default().max_cuts_per_round {
            self.cuts.max_cuts_per_round = other.cuts.max_cuts_per_round;
        }
        if other.benchmark.num_instances != BenchmarkConfig::default().num_instances {
            self.benchmark.num_instances = other.benchmark.num_instances;
        }
        if other.benchmark.warmup_runs != BenchmarkConfig::default().warmup_runs {
            self.benchmark.warmup_runs = other.benchmark.warmup_runs;
        }
        for (k, v) in &other.extra {
            self.extra.insert(k.clone(), v.clone());
        }
    }

    /// Write this config to the default path, creating directories as needed.
    pub fn write_default_file(&self) -> Result<PathBuf> {
        let dir = default_config_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating config dir {}", dir.display()))?;
        let path = default_config_path();
        let text = toml::to_string_pretty(self).context("serialising config")?;
        std::fs::write(&path, &text)
            .with_context(|| format!("writing config to {}", path.display()))?;
        Ok(path)
    }

    /// Load from a TOML string.
    pub fn from_toml(text: &str) -> Result<Self> {
        let cfg: BiCutConfig = toml::from_str(text).context("parsing TOML config")?;
        Ok(cfg)
    }

    /// Serialise to a TOML string.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serialising config to TOML")
    }
}

// ── Solver config ──────────────────────────────────────────────────

/// LP / MIP solver tunables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// Which LP solver back-end to use.
    pub lp_solver: String,

    /// Maximum simplex iterations per LP solve.
    pub max_iterations: u64,

    /// Wall-clock time limit in seconds for the overall solve.
    pub time_limit_secs: f64,

    /// Numerical pivot tolerance for the LP solver.
    pub pivot_tolerance: f64,

    /// Whether to use warm-starting between successive LP solves.
    pub warm_start: bool,

    /// Number of threads (0 = auto).
    pub threads: usize,

    /// Enable presolve in the LP solver.
    pub lp_presolve: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            lp_solver: "simplex".to_string(),
            max_iterations: 100_000,
            time_limit_secs: 3600.0,
            pivot_tolerance: 1e-8,
            warm_start: true,
            threads: 0,
            lp_presolve: true,
        }
    }
}

impl SolverConfig {
    fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            bail!("max_iterations must be > 0");
        }
        if self.time_limit_secs <= 0.0 {
            bail!("time_limit_secs must be positive");
        }
        if self.pivot_tolerance <= 0.0 || self.pivot_tolerance >= 1.0 {
            bail!("pivot_tolerance must be in (0, 1)");
        }
        let known = ["simplex", "dual-simplex", "barrier", "auto"];
        if !known.contains(&self.lp_solver.as_str()) {
            bail!(
                "unknown lp_solver '{}'; expected one of {:?}",
                self.lp_solver,
                known
            );
        }
        Ok(())
    }
}

// ── Compiler config ────────────────────────────────────────────────

/// Settings for the bilevel-to-single-level compilation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CompilerConfig {
    /// Reformulation strategy.
    pub reformulation: String,

    /// Big-M value for KKT complementarity linearisation.
    pub big_m: f64,

    /// Enable presolve / problem reduction before compilation.
    pub presolve: bool,

    /// Whether to compute the value function during compilation.
    pub compute_value_function: bool,

    /// Whether to generate intersection cuts during branch-and-cut.
    pub intersection_cuts: bool,

    /// Strategy for strengthening cuts.
    pub cut_strengthening: String,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            reformulation: "kkt".to_string(),
            big_m: 1e6,
            presolve: true,
            compute_value_function: false,
            intersection_cuts: true,
            cut_strengthening: "default".to_string(),
        }
    }
}

impl CompilerConfig {
    fn validate(&self) -> Result<()> {
        let known = ["kkt", "sip", "value-function", "hpf"];
        if !known.contains(&self.reformulation.as_str()) {
            bail!(
                "unknown reformulation '{}'; expected one of {:?}",
                self.reformulation,
                known
            );
        }
        if self.big_m <= 0.0 {
            bail!("big_m must be positive");
        }
        let known_strength = ["none", "default", "aggressive"];
        if !known_strength.contains(&self.cut_strengthening.as_str()) {
            bail!(
                "unknown cut_strengthening '{}'; expected one of {:?}",
                self.cut_strengthening,
                known_strength
            );
        }
        Ok(())
    }
}

// ── Cut config ─────────────────────────────────────────────────────

/// Settings for the cut generation (separation) loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CutConfig {
    /// Max separation rounds.
    pub max_rounds: usize,

    /// Max cuts added per round.
    pub max_cuts_per_round: usize,

    /// Minimum violation required to add a cut.
    pub min_violation: f64,

    /// Enable intersection cuts.
    pub intersection_cuts: bool,

    /// Enable lift-and-project cuts.
    pub lift_and_project: bool,

    /// Purge cuts whose slack exceeds this threshold.
    pub purge_threshold: f64,
}

impl Default for CutConfig {
    fn default() -> Self {
        Self {
            max_rounds: 50,
            max_cuts_per_round: 100,
            min_violation: 1e-6,
            intersection_cuts: true,
            lift_and_project: false,
            purge_threshold: 1e-4,
        }
    }
}

impl CutConfig {
    fn validate(&self) -> Result<()> {
        if self.max_rounds == 0 {
            bail!("max_rounds must be > 0");
        }
        if self.max_cuts_per_round == 0 {
            bail!("max_cuts_per_round must be > 0");
        }
        if self.min_violation < 0.0 {
            bail!("min_violation must be >= 0");
        }
        if self.purge_threshold < 0.0 {
            bail!("purge_threshold must be >= 0");
        }
        Ok(())
    }
}

// ── Benchmark config ───────────────────────────────────────────────

/// Settings for the benchmark runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchmarkConfig {
    /// Number of random instances to generate.
    pub num_instances: usize,

    /// Number of warm-up runs before measurement.
    pub warmup_runs: usize,

    /// Number of measurement runs per instance.
    pub measurement_runs: usize,

    /// Output directory for benchmark results.
    pub output_dir: String,

    /// Random seed (0 = use system entropy).
    pub seed: u64,

    /// Time limit per instance in seconds.
    pub per_instance_limit_secs: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_instances: 10,
            warmup_runs: 1,
            measurement_runs: 3,
            output_dir: "benchmark_output".to_string(),
            seed: 42,
            per_instance_limit_secs: 60.0,
        }
    }
}

impl BenchmarkConfig {
    fn validate(&self) -> Result<()> {
        if self.measurement_runs == 0 {
            bail!("measurement_runs must be > 0");
        }
        if self.per_instance_limit_secs <= 0.0 {
            bail!("per_instance_limit_secs must be positive");
        }
        Ok(())
    }
}

// ── Output config ──────────────────────────────────────────────────

/// Settings for output formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Whether to use colour in terminal output.
    pub color: bool,

    /// Decimal precision for floating-point numbers.
    pub precision: usize,

    /// Maximum column width for table output.
    pub max_column_width: usize,

    /// Whether to show progress bars.
    pub progress_bars: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            color: true,
            precision: 6,
            max_column_width: 40,
            progress_bars: true,
        }
    }
}

// ── Tolerance config ───────────────────────────────────────────────

/// Numerical tolerance settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToleranceConfig {
    /// Feasibility tolerance.
    pub feasibility: f64,

    /// Optimality tolerance (reduced cost).
    pub optimality: f64,

    /// Integrality tolerance.
    pub integrality: f64,

    /// Cut violation tolerance.
    pub cut_violation: f64,

    /// General zero tolerance.
    pub zero: f64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            feasibility: 1e-8,
            optimality: 1e-8,
            integrality: 1e-5,
            cut_violation: 1e-6,
            zero: 1e-10,
        }
    }
}

impl ToleranceConfig {
    fn validate(&self) -> Result<()> {
        if self.feasibility <= 0.0 || self.feasibility >= 1.0 {
            bail!("feasibility tolerance must be in (0, 1)");
        }
        if self.optimality <= 0.0 || self.optimality >= 1.0 {
            bail!("optimality tolerance must be in (0, 1)");
        }
        if self.integrality <= 0.0 || self.integrality >= 1.0 {
            bail!("integrality tolerance must be in (0, 1)");
        }
        if self.zero <= 0.0 || self.zero >= 1e-3 {
            bail!("zero tolerance must be in (0, 1e-3)");
        }
        Ok(())
    }
}

// ── CLI override merging ───────────────────────────────────────────

/// Keys that can be overridden from the CLI via `--set key=value`.
pub fn apply_overrides(cfg: &mut BiCutConfig, overrides: &[(String, String)]) -> Result<()> {
    for (key, value) in overrides {
        debug!("config override: {} = {}", key, value);
        match key.as_str() {
            "solver.max_iterations" => {
                cfg.solver.max_iterations =
                    value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "solver.time_limit_secs" => {
                cfg.solver.time_limit_secs =
                    value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "solver.lp_solver" => {
                cfg.solver.lp_solver = value.to_string();
            }
            "compiler.reformulation" => {
                cfg.compiler.reformulation = value.to_string();
            }
            "compiler.big_m" => {
                cfg.compiler.big_m = value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "compiler.presolve" => {
                cfg.compiler.presolve =
                    parse_bool(value).with_context(|| format!("parsing {key}"))?;
            }
            "cuts.max_rounds" => {
                cfg.cuts.max_rounds = value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "cuts.max_cuts_per_round" => {
                cfg.cuts.max_cuts_per_round =
                    value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "cuts.intersection_cuts" => {
                cfg.cuts.intersection_cuts =
                    parse_bool(value).with_context(|| format!("parsing {key}"))?;
            }
            "tolerances.feasibility" => {
                cfg.tolerances.feasibility =
                    value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "tolerances.optimality" => {
                cfg.tolerances.optimality =
                    value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "benchmark.seed" => {
                cfg.benchmark.seed = value.parse().with_context(|| format!("parsing {key}"))?;
            }
            "output.precision" => {
                cfg.output.precision = value.parse().with_context(|| format!("parsing {key}"))?;
            }
            other => {
                bail!("unknown config key '{other}'");
            }
        }
    }
    cfg.validate()?;
    Ok(())
}

fn parse_bool(s: &str) -> Result<bool> {
    match s.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => bail!("expected boolean, got '{s}'"),
    }
}

/// Parse a `key=value` string into a tuple.
pub fn parse_key_value(s: &str) -> Result<(String, String)> {
    let pos = s
        .find('=')
        .with_context(|| format!("expected KEY=VALUE, got '{s}'"))?;
    let key = s[..pos].trim().to_string();
    let val = s[pos + 1..].trim().to_string();
    if key.is_empty() {
        bail!("empty key in '{s}'");
    }
    Ok((key, val))
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_roundtrip() {
        let cfg = BiCutConfig::default();
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let parsed: BiCutConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.solver.max_iterations, 100_000);
        assert_eq!(parsed.compiler.reformulation, "kkt");
    }

    #[test]
    fn test_default_config_validates() {
        let mut cfg = BiCutConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_invalid_solver_config() {
        let mut cfg = BiCutConfig::default();
        cfg.solver.max_iterations = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_reformulation() {
        let mut cfg = BiCutConfig::default();
        cfg.compiler.reformulation = "bogus".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_merge_override() {
        let mut base = BiCutConfig::default();
        let mut overlay = BiCutConfig::default();
        overlay.solver.max_iterations = 999;
        base.merge_override(&overlay);
        assert_eq!(base.solver.max_iterations, 999);
    }

    #[test]
    fn test_apply_overrides() {
        let mut cfg = BiCutConfig::default();
        let overrides = vec![
            ("solver.max_iterations".to_string(), "500".to_string()),
            ("compiler.big_m".to_string(), "1e4".to_string()),
        ];
        apply_overrides(&mut cfg, &overrides).unwrap();
        assert_eq!(cfg.solver.max_iterations, 500);
        assert!((cfg.compiler.big_m - 1e4).abs() < 1e-6);
    }

    #[test]
    fn test_apply_overrides_unknown_key() {
        let mut cfg = BiCutConfig::default();
        let overrides = vec![("not.a.real.key".to_string(), "42".to_string())];
        assert!(apply_overrides(&mut cfg, &overrides).is_err());
    }

    #[test]
    fn test_parse_key_value() {
        let (k, v) = parse_key_value("solver.time_limit_secs=120").unwrap();
        assert_eq!(k, "solver.time_limit_secs");
        assert_eq!(v, "120");
    }

    #[test]
    fn test_parse_key_value_missing_eq() {
        assert!(parse_key_value("noequalshere").is_err());
    }

    #[test]
    fn test_parse_bool() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("yes").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("0").unwrap());
        assert!(parse_bool("maybe").is_err());
    }

    #[test]
    fn test_tolerance_validation() {
        let mut cfg = BiCutConfig::default();
        cfg.tolerances.feasibility = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_discover_config_returns_none_for_missing() {
        let result = discover_config(Path::new("/nonexistent/path/bicut_test_xyz"));
        // May or may not find ~/.bicut/config.toml; just ensure no panic.
        let _ = result;
    }

    #[test]
    fn test_from_toml_string() {
        let toml_str = r#"
            [solver]
            max_iterations = 5000
            time_limit_secs = 120.0

            [compiler]
            reformulation = "sip"
        "#;
        let cfg = BiCutConfig::from_toml(toml_str).unwrap();
        assert_eq!(cfg.solver.max_iterations, 5000);
        assert_eq!(cfg.compiler.reformulation, "sip");
    }

    #[test]
    fn test_default_config_dir() {
        let dir = default_config_dir();
        // Should end with .bicut
        assert!(dir.ends_with(".bicut"));
    }
}
