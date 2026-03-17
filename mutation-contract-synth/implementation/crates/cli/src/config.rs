//! CLI configuration management.
//!
//! [`CliConfig`] wraps `shared_types::config::MutSpecConfig` and adds CLI-level
//! conveniences: file discovery, environment variable overrides, merge logic, and
//! validation diagnostics.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use shared_types::config::{
    AnalysisConfig, CoverageConfig, MutSpecConfig, MutationConfig, OutputFormat, SmtConfig,
    SynthesisConfig,
};
use shared_types::operators::MutationOperator;

// ---------------------------------------------------------------------------
// Configuration search paths
// ---------------------------------------------------------------------------

const CONFIG_FILE_NAMES: &[&str] = &[
    "mutspec.toml",
    ".mutspec.toml",
    ".mutspec/config.toml",
    "mutspec/config.toml",
];

const ENV_PREFIX: &str = "MUTSPEC_";

// ---------------------------------------------------------------------------
// ConfigDiagnostic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ConfigSeverity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for ConfigSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigDiagnostic {
    pub severity: ConfigSeverity,
    pub field: String,
    pub message: String,
}

impl fmt::Display for ConfigDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

// ---------------------------------------------------------------------------
// CliConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Underlying library config.
    #[serde(flatten)]
    pub inner: MutSpecConfig,

    /// Path to the config file that was loaded, if any.
    #[serde(skip)]
    pub config_path: Option<PathBuf>,

    /// Whether verbose logging was requested at the CLI level.
    #[serde(skip)]
    pub verbose: bool,

    /// Number of parallel workers (CLI-level; not in library config).
    #[serde(skip)]
    pub parallelism_count: usize,
}

impl CliConfig {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    pub fn default_config() -> Self {
        Self {
            inner: MutSpecConfig::test_config(),
            config_path: None,
            verbose: false,
            parallelism_count: 0,
        }
    }

    /// Load from an explicit path, falling back to discovery.
    pub fn load(path: Option<&Path>, verbose: bool) -> Result<Self> {
        let mut cfg = if let Some(p) = path {
            info!("Loading config from explicit path: {}", p.display());
            Self::load_file(p)?
        } else {
            match Self::discover()? {
                Some(c) => c,
                None => {
                    info!("No config file found; using defaults");
                    Self::default_config()
                }
            }
        };
        cfg.verbose = verbose;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    fn load_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file {}", path.display()))?;
        let inner: MutSpecConfig = MutSpecConfig::from_toml(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse TOML config: {e}"))?;
        Ok(Self {
            inner,
            config_path: Some(path.to_path_buf()),
            verbose: false,
            parallelism_count: 0,
        })
    }

    /// Walk up from the current directory looking for known config file names.
    fn discover() -> Result<Option<Self>> {
        let cwd = std::env::current_dir().context("Cannot determine cwd")?;
        let mut dir = cwd.as_path();
        loop {
            for name in CONFIG_FILE_NAMES {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    info!("Discovered config at {}", candidate.display());
                    return Self::load_file(&candidate).map(Some);
                }
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
        Ok(None)
    }

    // ------------------------------------------------------------------
    // Serialisation helpers
    // ------------------------------------------------------------------

    pub fn to_toml_string(&self) -> Result<String> {
        self.inner.to_toml().map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        let content = self.to_toml_string()?;
        std::fs::write(path, &content)
            .with_context(|| format!("Failed to write config to {}", path.display()))?;
        info!("Config saved to {}", path.display());
        Ok(())
    }

    // ------------------------------------------------------------------
    // Environment overrides
    // ------------------------------------------------------------------

    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}TIMEOUT_SECS")) {
            if let Ok(v) = val.parse::<u64>() {
                debug!("Applying env override MUTSPEC_TIMEOUT_SECS={v}");
                self.inner.smt.timeout_secs = v;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}MAX_MUTANTS")) {
            if let Ok(v) = val.parse::<usize>() {
                debug!("Applying env override MUTSPEC_MAX_MUTANTS={v}");
                self.inner.mutation.max_mutants_per_site = v;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}OPERATORS")) {
            let ops: Vec<String> = val
                .split(',')
                .filter_map(|s| {
                    let trimmed = s.trim();
                    MutationOperator::from_mnemonic(trimmed).map(|op| op.mnemonic().to_string())
                })
                .collect();
            if !ops.is_empty() {
                debug!(
                    "Applying env override MUTSPEC_OPERATORS with {} operators",
                    ops.len()
                );
                self.inner.mutation.operators = ops;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}OUTPUT_FORMAT")) {
            match val.to_lowercase().as_str() {
                "json" => self.inner.output.format = OutputFormat::Json,
                "text" => self.inner.output.format = OutputFormat::Text,
                "sarif" => self.inner.output.format = OutputFormat::Sarif,
                "jml" => self.inner.output.format = OutputFormat::Jml,
                other => warn!("Unknown MUTSPEC_OUTPUT_FORMAT value: {other}"),
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}SMT_SOLVER")) {
            debug!("Applying env override MUTSPEC_SMT_SOLVER={val}");
            self.inner.smt.solver_path = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}SYNTHESIS_TIER")) {
            if let Ok(n) = val.parse::<u8>() {
                if n >= 1 && n <= 3 {
                    debug!("Applying env override MUTSPEC_SYNTHESIS_TIER={n}");
                    self.inner.synthesis.enabled_tiers = (1..=n).collect();
                }
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}PARALLEL")) {
            if let Ok(v) = val.parse::<usize>() {
                debug!("Applying env override MUTSPEC_PARALLEL={v}");
                self.parallelism_count = v;
            }
        }
    }

    /// Return all overrides that were set via the environment.
    pub fn active_env_overrides(&self) -> Vec<(String, String)> {
        let mut overrides = Vec::new();
        let keys = [
            "TIMEOUT_SECS",
            "MAX_MUTANTS",
            "OPERATORS",
            "OUTPUT_FORMAT",
            "SMT_SOLVER",
            "SYNTHESIS_TIER",
            "PARALLEL",
        ];
        for key in keys {
            if let Ok(val) = std::env::var(format!("{ENV_PREFIX}{key}")) {
                overrides.push((format!("{ENV_PREFIX}{key}"), val));
            }
        }
        overrides
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    pub fn validate(&self) -> Vec<ConfigDiagnostic> {
        let mut diags = Vec::new();

        if self.inner.mutation.operators.is_empty() {
            diags.push(ConfigDiagnostic {
                severity: ConfigSeverity::Error,
                field: "mutation.operators".into(),
                message: "No mutation operators specified".into(),
            });
        }

        if self.inner.mutation.max_mutants_per_site == 0 {
            diags.push(ConfigDiagnostic {
                severity: ConfigSeverity::Warning,
                field: "mutation.max_mutants_per_site".into(),
                message: "Max mutants set to 0; no mutants will be generated".into(),
            });
        }

        if self.inner.smt.timeout_secs == 0 {
            diags.push(ConfigDiagnostic {
                severity: ConfigSeverity::Warning,
                field: "smt.timeout_secs".into(),
                message: "SMT timeout is 0; solver calls will immediately time out".into(),
            });
        }

        if !self.inner.smt.solver_path.as_os_str().is_empty()
            && !self.inner.smt.solver_path.exists()
        {
            diags.push(ConfigDiagnostic {
                severity: ConfigSeverity::Warning,
                field: "smt.solver_path".into(),
                message: format!(
                    "Solver path does not exist: {}",
                    self.inner.smt.solver_path.display()
                ),
            });
        }

        if self.parallelism_count == 0 {
            diags.push(ConfigDiagnostic {
                severity: ConfigSeverity::Warning,
                field: "parallelism".into(),
                message: "Parallelism set to 0; will use 1 thread".into(),
            });
        }

        diags
    }

    pub fn has_errors(&self) -> bool {
        self.validate()
            .iter()
            .any(|d| matches!(d.severity, ConfigSeverity::Error))
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn mutation(&self) -> &MutationConfig {
        &self.inner.mutation
    }

    pub fn synthesis(&self) -> &SynthesisConfig {
        &self.inner.synthesis
    }

    pub fn smt(&self) -> &SmtConfig {
        &self.inner.smt
    }

    pub fn analysis(&self) -> &AnalysisConfig {
        &self.inner.analysis
    }

    pub fn coverage(&self) -> &CoverageConfig {
        &self.inner.coverage
    }

    pub fn output_format(&self) -> &OutputFormat {
        &self.inner.output.format
    }

    pub fn parallelism(&self) -> usize {
        let p = self.parallelism_count;
        if p == 0 {
            1
        } else {
            p
        }
    }

    // ------------------------------------------------------------------
    // Merge helpers
    // ------------------------------------------------------------------

    pub fn with_operators(&self, ops: Vec<MutationOperator>) -> Self {
        let mut c = self.clone();
        c.inner.mutation.operators = ops.iter().map(|o| o.mnemonic().to_string()).collect();
        c
    }

    pub fn with_timeout(&self, secs: u64) -> Self {
        let mut c = self.clone();
        c.inner.smt.timeout_secs = secs;
        c
    }

    pub fn with_max_mutants(&self, max: usize) -> Self {
        let mut c = self.clone();
        c.inner.mutation.max_mutants_per_site = max;
        c
    }

    pub fn with_parallelism(&self, n: usize) -> Self {
        let mut c = self.clone();
        c.parallelism_count = n;
        c
    }
}

// ---------------------------------------------------------------------------
// Init template
// ---------------------------------------------------------------------------

pub fn default_config_template() -> String {
    let cfg = MutSpecConfig::test_config();
    cfg.to_toml().unwrap_or_default()
}

pub fn annotated_config_template() -> String {
    let mut buf = String::new();
    buf.push_str("# MutSpec configuration file\n");
    buf.push_str("# Place in project root as mutspec.toml\n\n");

    buf.push_str("[mutation]\n");
    buf.push_str(
        "# Mutation operators (AOR, ROR, LCR, UOI, ABS, COR, SDL, RVR, CRC, AIR, OSW, BCN)\n",
    );
    buf.push_str("operators = [\"AOR\", \"ROR\", \"LCR\", \"UOI\", \"ABS\"]\n");
    buf.push_str("# Maximum mutants per site\n");
    buf.push_str("max_mutants_per_site = 10\n");
    buf.push_str("# Generation timeout in seconds\n");
    buf.push_str("generation_timeout_secs = 30\n\n");

    buf.push_str("[synthesis]\n");
    buf.push_str("# Enabled synthesis tiers (1 = lattice, 2 = template, 3 = fallback)\n");
    buf.push_str("enabled_tiers = [1, 2, 3]\n");
    buf.push_str("# Timeout per tier in seconds\n");
    buf.push_str("tier_timeout_secs = 60\n");
    buf.push_str("# Enable contract minimisation\n");
    buf.push_str("minimise_contracts = true\n\n");

    buf.push_str("[smt]\n");
    buf.push_str("# Path to SMT solver binary (z3, cvc5, etc.)\n");
    buf.push_str("solver_path = \"z3\"\n");
    buf.push_str("# Solver timeout in seconds\n");
    buf.push_str("timeout_secs = 30\n");
    buf.push_str("# SMT logic to use\n");
    buf.push_str("logic = \"QF_LIA\"\n\n");

    buf.push_str("[analysis]\n");
    buf.push_str("# Maximum expression depth\n");
    buf.push_str("max_expr_depth = 50\n");
    buf.push_str("# Use SSA form\n");
    buf.push_str("use_ssa = true\n\n");

    buf.push_str("[coverage]\n");
    buf.push_str("# Enable subsumption analysis\n");
    buf.push_str("subsumption = true\n");
    buf.push_str("# Adequate score threshold (0.0-1.0)\n");
    buf.push_str("adequate_score_threshold = 0.8\n\n");

    buf.push_str("[output]\n");
    buf.push_str("# Output format: text, json, sarif, jml\n");
    buf.push_str("format = \"text\"\n");
    buf.push_str("# Output directory for reports\n");
    buf.push_str("output_dir = \"mutspec-output\"\n");
    buf.push_str("# Verbosity level (0-3)\n");
    buf.push_str("verbosity = 1\n");

    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = CliConfig::default_config();
        assert!(!cfg.inner.mutation.operators.is_empty());
    }

    #[test]
    fn test_validate_empty_operators() {
        let mut cfg = CliConfig::default_config();
        cfg.inner.mutation.operators.clear();
        let diags = cfg.validate();
        assert!(diags.iter().any(|d| d.field == "mutation.operators"));
    }

    #[test]
    fn test_with_timeout() {
        let cfg = CliConfig::default_config().with_timeout(99);
        assert_eq!(cfg.inner.smt.timeout_secs, 99);
    }

    #[test]
    fn test_with_parallelism() {
        let cfg = CliConfig::default_config().with_parallelism(8);
        assert_eq!(cfg.parallelism(), 8);
    }

    #[test]
    fn test_parallelism_zero_fallback() {
        let cfg = CliConfig::default_config().with_parallelism(0);
        assert_eq!(cfg.parallelism(), 1);
    }
    #[test]
    fn test_annotated_template() {
        let t = annotated_config_template();
        assert!(t.contains("[mutation]"));
        assert!(t.contains("[smt]"));
    }
}
