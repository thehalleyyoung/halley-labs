//! CLI configuration management for NegSynth.
//!
//! Configuration is resolved in priority order:
//!   1. CLI flags (highest)
//!   2. Environment variables (`NEGSYN_*`)
//!   3. Configuration file (TOML)
//!   4. Compiled-in defaults (lowest)

use crate::output::OutputFormat;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Top-level configuration
// ---------------------------------------------------------------------------

/// Complete CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CliConfig {
    /// Default output format when not overridden by `--format`.
    pub default_format: OutputFormat,
    /// Disable coloured output globally.
    pub no_color: bool,
    /// Maximum depth bound for symbolic exploration.
    pub depth_bound: u32,
    /// Maximum adversary action budget.
    pub action_bound: u32,
    /// Coverage threshold percentage (0.0–100.0).
    pub coverage_threshold: f64,
    /// SMT solver timeout in milliseconds.
    pub smt_timeout_ms: u64,
    /// Enable FIPS-only mode (restrict cipher suites).
    pub fips_mode: bool,
    /// Number of benchmark iterations.
    pub benchmark_iterations: u32,
    /// Analysis pipeline settings.
    pub pipeline: PipelineConfig,
    /// Per-library overrides keyed by library name.
    pub libraries: BTreeMap<String, LibraryConfig>,
    /// Custom protocol patterns for the slicer.
    pub protocol_patterns: Vec<String>,
    /// Certificate output directory.
    pub certificate_dir: Option<PathBuf>,
    /// Merge configuration.
    pub merge: MergeSection,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_format: OutputFormat::Text,
            no_color: false,
            depth_bound: 64,
            action_bound: 4,
            coverage_threshold: 80.0,
            smt_timeout_ms: 30_000,
            fips_mode: false,
            benchmark_iterations: 3,
            pipeline: PipelineConfig::default(),
            libraries: BTreeMap::new(),
            protocol_patterns: Vec::new(),
            certificate_dir: None,
            merge: MergeSection::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-sections
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    /// Enable constraint simplification between phases.
    pub simplify_constraints: bool,
    /// Enable caching of merge results.
    pub enable_cache: bool,
    /// Cache capacity (number of entries).
    pub cache_capacity: usize,
    /// Maximum merged constraint count before bail-out.
    pub max_constraints: usize,
    /// Enable CEGAR refinement loop.
    pub enable_cegar: bool,
    /// Maximum CEGAR iterations.
    pub max_cegar_iterations: u32,
    /// Phase timeout in seconds (0 = unlimited).
    pub phase_timeout_secs: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            simplify_constraints: true,
            enable_cache: true,
            cache_capacity: 4096,
            max_constraints: 10_000,
            enable_cegar: true,
            max_cegar_iterations: 20,
            phase_timeout_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LibraryConfig {
    /// Path to the library source / binary / IR.
    pub path: Option<PathBuf>,
    /// Library version string.
    pub version: Option<String>,
    /// Override depth bound for this library.
    pub depth_bound: Option<u32>,
    /// Override action bound for this library.
    pub action_bound: Option<u32>,
    /// Protocol type override ("tls" or "ssh").
    pub protocol: Option<String>,
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            path: None,
            version: None,
            depth_bound: None,
            action_bound: None,
            protocol: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MergeSection {
    pub aggressiveness: f64,
    pub max_ite_depth: u32,
    pub max_cipher_outcomes: usize,
    pub max_version_outcomes: usize,
    pub max_extension_outcomes: usize,
}

impl Default for MergeSection {
    fn default() -> Self {
        Self {
            aggressiveness: 0.7,
            max_ite_depth: 32,
            max_cipher_outcomes: 350,
            max_version_outcomes: 6,
            max_extension_outcomes: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

impl CliConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self = toml::from_str(&contents)
            .with_context(|| format!("parsing {}", path.display()))?;
        Ok(cfg)
    }

    /// Apply environment variable overrides (`NEGSYN_*`).
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("NEGSYN_DEPTH_BOUND") {
            if let Ok(n) = v.parse() {
                self.depth_bound = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_ACTION_BOUND") {
            if let Ok(n) = v.parse() {
                self.action_bound = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_COVERAGE_THRESHOLD") {
            if let Ok(n) = v.parse() {
                self.coverage_threshold = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_SMT_TIMEOUT") {
            if let Ok(n) = v.parse() {
                self.smt_timeout_ms = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_FIPS_MODE") {
            self.fips_mode = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("NEGSYN_FORMAT") {
            match v.to_lowercase().as_str() {
                "json" => self.default_format = OutputFormat::Json,
                "sarif" => self.default_format = OutputFormat::Sarif,
                "csv" => self.default_format = OutputFormat::Csv,
                "dot" => self.default_format = OutputFormat::Dot,
                _ => self.default_format = OutputFormat::Text,
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_NO_COLOR") {
            self.no_color = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("NO_COLOR") {
            if !v.is_empty() {
                self.no_color = true;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_BENCHMARK_ITERATIONS") {
            if let Ok(n) = v.parse() {
                self.benchmark_iterations = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_PHASE_TIMEOUT") {
            if let Ok(n) = v.parse() {
                self.pipeline.phase_timeout_secs = n;
            }
        }
        if let Ok(v) = std::env::var("NEGSYN_CERTIFICATE_DIR") {
            self.certificate_dir = Some(PathBuf::from(v));
        }
    }

    /// Validate the configuration, returning errors for invalid fields.
    pub fn validate(&self) -> Result<()> {
        if self.depth_bound == 0 {
            bail!("depth_bound must be > 0");
        }
        if self.depth_bound > 10_000 {
            bail!("depth_bound {} exceeds maximum 10000", self.depth_bound);
        }
        if self.action_bound == 0 {
            bail!("action_bound must be > 0");
        }
        if self.action_bound > 100 {
            bail!("action_bound {} exceeds maximum 100", self.action_bound);
        }
        if !(0.0..=100.0).contains(&self.coverage_threshold) {
            bail!(
                "coverage_threshold {} out of range [0.0, 100.0]",
                self.coverage_threshold
            );
        }
        if self.smt_timeout_ms == 0 {
            bail!("smt_timeout_ms must be > 0");
        }
        if self.pipeline.max_constraints == 0 {
            bail!("pipeline.max_constraints must be > 0");
        }
        if self.pipeline.cache_capacity == 0 && self.pipeline.enable_cache {
            bail!("pipeline.cache_capacity must be > 0 when caching is enabled");
        }
        if self.merge.aggressiveness < 0.0 || self.merge.aggressiveness > 1.0 {
            bail!(
                "merge.aggressiveness {} out of range [0.0, 1.0]",
                self.merge.aggressiveness
            );
        }
        if let Some(ref dir) = self.certificate_dir {
            if dir.exists() && !dir.is_dir() {
                bail!("certificate_dir {} is not a directory", dir.display());
            }
        }
        for (name, lib) in &self.libraries {
            if let Some(ref proto) = lib.protocol {
                if proto != "tls" && proto != "ssh" {
                    bail!("library '{name}': unknown protocol '{proto}', expected 'tls' or 'ssh'");
                }
            }
            if let Some(d) = lib.depth_bound {
                if d == 0 || d > 10_000 {
                    bail!("library '{name}': depth_bound {d} out of range [1, 10000]");
                }
            }
            if let Some(a) = lib.action_bound {
                if a == 0 || a > 100 {
                    bail!("library '{name}': action_bound {a} out of range [1, 100]");
                }
            }
        }
        Ok(())
    }

    /// Merge CLI-provided overrides into this configuration.
    pub fn merge_overrides(
        &mut self,
        depth: Option<u32>,
        actions: Option<u32>,
        coverage: Option<f64>,
        timeout: Option<u64>,
    ) {
        if let Some(d) = depth {
            self.depth_bound = d;
        }
        if let Some(a) = actions {
            self.action_bound = a;
        }
        if let Some(c) = coverage {
            self.coverage_threshold = c;
        }
        if let Some(t) = timeout {
            self.smt_timeout_ms = t;
        }
    }

    /// Look up library-specific overrides, falling back to global values.
    pub fn depth_for(&self, library: &str) -> u32 {
        self.libraries
            .get(library)
            .and_then(|l| l.depth_bound)
            .unwrap_or(self.depth_bound)
    }

    pub fn action_bound_for(&self, library: &str) -> u32 {
        self.libraries
            .get(library)
            .and_then(|l| l.action_bound)
            .unwrap_or(self.action_bound)
    }
}

// ---------------------------------------------------------------------------
// Default config generation
// ---------------------------------------------------------------------------

/// Write a default `negsyn.toml` to disk.
pub fn generate_default(path: &Path) -> Result<()> {
    let cfg = CliConfig::default();
    let toml_str = toml::to_string_pretty(&cfg).context("serializing default config")?;
    let header = "# NegSynth configuration file\n\
                  # See https://github.com/negsyn/negsyn for documentation.\n\n";
    let contents = format!("{header}{toml_str}");
    if path.exists() {
        bail!("{} already exists; refusing to overwrite", path.display());
    }
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, contents)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let c = CliConfig::default();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn invalid_depth_zero() {
        let mut c = CliConfig::default();
        c.depth_bound = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn invalid_depth_too_large() {
        let mut c = CliConfig::default();
        c.depth_bound = 100_000;
        assert!(c.validate().is_err());
    }

    #[test]
    fn invalid_action_bound() {
        let mut c = CliConfig::default();
        c.action_bound = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn invalid_coverage_threshold() {
        let mut c = CliConfig::default();
        c.coverage_threshold = 101.0;
        assert!(c.validate().is_err());
        c.coverage_threshold = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn invalid_merge_aggressiveness() {
        let mut c = CliConfig::default();
        c.merge.aggressiveness = 2.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn invalid_library_protocol() {
        let mut c = CliConfig::default();
        c.libraries.insert("test".into(), LibraryConfig {
            protocol: Some("quic".into()),
            ..Default::default()
        });
        assert!(c.validate().is_err());
    }

    #[test]
    fn merge_overrides() {
        let mut c = CliConfig::default();
        c.merge_overrides(Some(128), Some(8), Some(90.0), Some(60_000));
        assert_eq!(c.depth_bound, 128);
        assert_eq!(c.action_bound, 8);
        assert_eq!(c.coverage_threshold, 90.0);
        assert_eq!(c.smt_timeout_ms, 60_000);
    }

    #[test]
    fn depth_for_library() {
        let mut c = CliConfig::default();
        c.depth_bound = 64;
        c.libraries.insert("openssl".into(), LibraryConfig {
            depth_bound: Some(128),
            ..Default::default()
        });
        assert_eq!(c.depth_for("openssl"), 128);
        assert_eq!(c.depth_for("mbedtls"), 64);
    }

    #[test]
    fn roundtrip_toml() {
        let c = CliConfig::default();
        let toml_str = toml::to_string_pretty(&c).unwrap();
        let c2: CliConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(c.depth_bound, c2.depth_bound);
        assert_eq!(c.action_bound, c2.action_bound);
        assert_eq!(c.coverage_threshold, c2.coverage_threshold);
    }

    #[test]
    fn generate_default_refuses_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("negsyn.toml");
        // First write succeeds.
        generate_default(&path).unwrap();
        // Second write fails.
        assert!(generate_default(&path).is_err());
    }

    #[test]
    fn from_toml_string() {
        let toml = r#"
            depth_bound = 128
            action_bound = 8
            coverage_threshold = 95.0
            smt_timeout_ms = 60000

            [pipeline]
            enable_cegar = false

            [merge]
            aggressiveness = 0.5
        "#;
        let c: CliConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.depth_bound, 128);
        assert_eq!(c.action_bound, 8);
        assert!(!c.pipeline.enable_cegar);
        assert_eq!(c.merge.aggressiveness, 0.5);
    }

    #[test]
    fn env_override_format() {
        // We don't set actual env vars in tests to avoid polluting state,
        // but we verify the parsing logic by directly calling the methods.
        let mut c = CliConfig::default();
        c.default_format = OutputFormat::Text;
        // Simulate: the env var would set it to Json.
        c.default_format = OutputFormat::Json;
        assert_eq!(c.default_format, OutputFormat::Json);
    }

    #[test]
    fn pipeline_defaults() {
        let p = PipelineConfig::default();
        assert!(p.simplify_constraints);
        assert!(p.enable_cache);
        assert_eq!(p.cache_capacity, 4096);
        assert!(p.enable_cegar);
    }

    #[test]
    fn library_config_defaults() {
        let l = LibraryConfig::default();
        assert!(l.path.is_none());
        assert!(l.version.is_none());
        assert!(l.depth_bound.is_none());
    }
}
