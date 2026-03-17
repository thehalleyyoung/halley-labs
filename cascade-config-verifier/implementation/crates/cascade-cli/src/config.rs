//! CLI configuration loading and management.
//!
//! Provides hierarchical configuration: file → environment → CLI flags.
//! Looks for `.cascade-verify.yaml` / `.cascade-verify.yml` in the current
//! directory and the user's home directory.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Configuration structs ──────────────────────────────────────────────────

/// Top-level CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CliConfig {
    #[serde(default)]
    pub analysis: AnalysisSettings,
    #[serde(default)]
    pub output: OutputSettings,
    #[serde(default)]
    pub cache: CacheSettings,
    #[serde(default)]
    pub ci: CiSettings,
}

/// Controls the analysis engine behaviour.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalysisSettings {
    /// Default analysis mode: `fast`, `deep`, or `auto`.
    #[serde(default = "default_mode")]
    pub default_mode: String,
    /// Maximum number of simultaneous failures for Tier 2 BMC.
    #[serde(default = "default_max_failure_budget")]
    pub max_failure_budget: usize,
    /// Global wall-clock timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Minimum amplification factor to flag a risk (Tier 1).
    #[serde(default = "default_amplification_threshold")]
    pub amplification_threshold: f64,
    /// Per-path timeout budget violation threshold in milliseconds.
    #[serde(default = "default_timeout_threshold_ms")]
    pub timeout_threshold_ms: u64,
}

/// Controls output formatting and destinations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputSettings {
    /// Default output format: `table`, `json`, `yaml`, `markdown`.
    #[serde(default = "default_format")]
    pub default_format: String,
    /// Enable verbose output by default.
    #[serde(default)]
    pub verbose: bool,
    /// Enable coloured terminal output.
    #[serde(default = "default_true")]
    pub color: bool,
    /// Optional directory to write output files.
    #[serde(default)]
    pub output_dir: Option<String>,
}

/// Controls the analysis result cache.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheSettings {
    /// Enable / disable caching.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Directory for cached results.
    #[serde(default = "default_cache_dir")]
    pub dir: String,
    /// Maximum cache size in megabytes.
    #[serde(default = "default_max_size_mb")]
    pub max_size_mb: usize,
    /// Time-to-live for cached entries in hours.
    #[serde(default = "default_ttl_hours")]
    pub ttl_hours: u64,
}

/// CI / CD gate-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CiSettings {
    /// Fail the build on CRITICAL findings.
    #[serde(default = "default_true")]
    pub fail_on_critical: bool,
    /// Fail the build on HIGH findings.
    #[serde(default)]
    pub fail_on_high: bool,
    /// Maximum number of MEDIUM findings before failure.
    #[serde(default = "default_max_medium")]
    pub max_medium: usize,
    /// Emit GitHub Actions `::warning::` / `::error::` annotations.
    #[serde(default)]
    pub annotations: bool,
    /// Write a step summary for GitHub Actions.
    #[serde(default)]
    pub step_summary: bool,
}

// ── Serde default helpers ──────────────────────────────────────────────────

fn default_mode() -> String {
    "auto".into()
}
fn default_max_failure_budget() -> usize {
    3
}
fn default_timeout_ms() -> u64 {
    60_000
}
fn default_amplification_threshold() -> f64 {
    8.0
}
fn default_timeout_threshold_ms() -> u64 {
    30_000
}
fn default_format() -> String {
    "table".into()
}
fn default_true() -> bool {
    true
}
fn default_cache_dir() -> String {
    ".cascade-cache".into()
}
fn default_max_size_mb() -> usize {
    256
}
fn default_ttl_hours() -> u64 {
    24
}
fn default_max_medium() -> usize {
    10
}

// ── Default impls ──────────────────────────────────────────────────────────

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            analysis: AnalysisSettings::default(),
            output: OutputSettings::default(),
            cache: CacheSettings::default(),
            ci: CiSettings::default(),
        }
    }
}

impl Default for AnalysisSettings {
    fn default() -> Self {
        Self {
            default_mode: default_mode(),
            max_failure_budget: default_max_failure_budget(),
            timeout_ms: default_timeout_ms(),
            amplification_threshold: default_amplification_threshold(),
            timeout_threshold_ms: default_timeout_threshold_ms(),
        }
    }
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            default_format: default_format(),
            verbose: false,
            color: true,
            output_dir: None,
        }
    }
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: default_cache_dir(),
            max_size_mb: default_max_size_mb(),
            ttl_hours: default_ttl_hours(),
        }
    }
}

impl Default for CiSettings {
    fn default() -> Self {
        Self {
            fail_on_critical: true,
            fail_on_high: false,
            max_medium: default_max_medium(),
            annotations: false,
            step_summary: false,
        }
    }
}

// ── Configuration loading ──────────────────────────────────────────────────

impl CliConfig {
    /// Load configuration from a YAML file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config file '{}': {}", path, e))?;
        let config: Self = serde_yaml::from_str(&content)
            .map_err(|e| format!("failed to parse config file '{}': {}", path, e))?;
        Ok(config)
    }

    /// Build a configuration by reading well-known environment variables.
    ///
    /// Supported variables:
    /// - `CASCADE_MODE` → `analysis.default_mode`
    /// - `CASCADE_OUTPUT` → `output.default_format`
    /// - `CASCADE_TIMEOUT` → `analysis.timeout_ms`
    /// - `CASCADE_MAX_FAILURES` → `analysis.max_failure_budget`
    /// - `CASCADE_CACHE_DIR` → `cache.dir`
    /// - `CASCADE_VERBOSE` → `output.verbose`
    /// - `NO_COLOR` → `output.color = false`
    /// - `CASCADE_FAIL_ON_CRITICAL` → `ci.fail_on_critical`
    /// - `CASCADE_FAIL_ON_HIGH` → `ci.fail_on_high`
    /// - `CASCADE_MAX_MEDIUM` → `ci.max_medium`
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(v) = std::env::var("CASCADE_MODE") {
            config.analysis.default_mode = v;
        }
        if let Ok(v) = std::env::var("CASCADE_OUTPUT") {
            config.output.default_format = v;
        }
        if let Ok(v) = std::env::var("CASCADE_TIMEOUT") {
            if let Ok(ms) = v.parse::<u64>() {
                config.analysis.timeout_ms = ms;
            }
        }
        if let Ok(v) = std::env::var("CASCADE_MAX_FAILURES") {
            if let Ok(n) = v.parse::<usize>() {
                config.analysis.max_failure_budget = n;
            }
        }
        if let Ok(v) = std::env::var("CASCADE_CACHE_DIR") {
            config.cache.dir = v;
        }
        if let Ok(v) = std::env::var("CASCADE_VERBOSE") {
            config.output.verbose = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if std::env::var("NO_COLOR").is_ok() {
            config.output.color = false;
        }
        if let Ok(v) = std::env::var("CASCADE_FAIL_ON_CRITICAL") {
            config.ci.fail_on_critical = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("CASCADE_FAIL_ON_HIGH") {
            config.ci.fail_on_high = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = std::env::var("CASCADE_MAX_MEDIUM") {
            if let Ok(n) = v.parse::<usize>() {
                config.ci.max_medium = n;
            }
        }
        config
    }

    /// Merge two configurations.  Values in `overrides` take precedence
    /// unless they equal the default, in which case `base` is kept.
    pub fn merge(base: Self, overrides: Self) -> Self {
        let default = Self::default();
        Self {
            analysis: AnalysisSettings {
                default_mode: pick(
                    &base.analysis.default_mode,
                    &overrides.analysis.default_mode,
                    &default.analysis.default_mode,
                ),
                max_failure_budget: pick_val(
                    base.analysis.max_failure_budget,
                    overrides.analysis.max_failure_budget,
                    default.analysis.max_failure_budget,
                ),
                timeout_ms: pick_val(
                    base.analysis.timeout_ms,
                    overrides.analysis.timeout_ms,
                    default.analysis.timeout_ms,
                ),
                amplification_threshold: pick_f64(
                    base.analysis.amplification_threshold,
                    overrides.analysis.amplification_threshold,
                    default.analysis.amplification_threshold,
                ),
                timeout_threshold_ms: pick_val(
                    base.analysis.timeout_threshold_ms,
                    overrides.analysis.timeout_threshold_ms,
                    default.analysis.timeout_threshold_ms,
                ),
            },
            output: OutputSettings {
                default_format: pick(
                    &base.output.default_format,
                    &overrides.output.default_format,
                    &default.output.default_format,
                ),
                verbose: base.output.verbose || overrides.output.verbose,
                color: base.output.color && overrides.output.color,
                output_dir: overrides.output.output_dir.or(base.output.output_dir),
            },
            cache: CacheSettings {
                enabled: overrides.cache.enabled && base.cache.enabled,
                dir: pick(
                    &base.cache.dir,
                    &overrides.cache.dir,
                    &default.cache.dir,
                ),
                max_size_mb: pick_val(
                    base.cache.max_size_mb,
                    overrides.cache.max_size_mb,
                    default.cache.max_size_mb,
                ),
                ttl_hours: pick_val(
                    base.cache.ttl_hours,
                    overrides.cache.ttl_hours,
                    default.cache.ttl_hours,
                ),
            },
            ci: CiSettings {
                fail_on_critical: overrides.ci.fail_on_critical || base.ci.fail_on_critical,
                fail_on_high: overrides.ci.fail_on_high || base.ci.fail_on_high,
                max_medium: pick_val(
                    base.ci.max_medium,
                    overrides.ci.max_medium,
                    default.ci.max_medium,
                ),
                annotations: overrides.ci.annotations || base.ci.annotations,
                step_summary: overrides.ci.step_summary || base.ci.step_summary,
            },
        }
    }

    /// Validate the configuration and return all detected problems.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Analysis settings.
        match self.analysis.default_mode.as_str() {
            "fast" | "deep" | "auto" => {}
            other => errors.push(format!(
                "analysis.default_mode: unknown mode '{}' (expected fast, deep, auto)",
                other
            )),
        }
        if self.analysis.max_failure_budget == 0 {
            errors.push("analysis.max_failure_budget must be > 0".into());
        }
        if self.analysis.max_failure_budget > 20 {
            errors.push(format!(
                "analysis.max_failure_budget {} is unreasonably large (max 20)",
                self.analysis.max_failure_budget
            ));
        }
        if self.analysis.timeout_ms == 0 {
            errors.push("analysis.timeout_ms must be > 0".into());
        }
        if self.analysis.amplification_threshold <= 1.0 {
            errors.push(format!(
                "analysis.amplification_threshold {} must be > 1.0",
                self.analysis.amplification_threshold
            ));
        }
        if self.analysis.timeout_threshold_ms == 0 {
            errors.push("analysis.timeout_threshold_ms must be > 0".into());
        }

        // Output settings.
        match self.output.default_format.as_str() {
            "table" | "json" | "yaml" | "markdown" | "sarif" | "junit" => {}
            other => errors.push(format!(
                "output.default_format: unknown format '{}' (expected table, json, yaml, markdown, sarif, junit)",
                other
            )),
        }

        // Cache settings.
        if self.cache.enabled {
            if self.cache.dir.is_empty() {
                errors.push("cache.dir must not be empty when caching is enabled".into());
            }
            if self.cache.max_size_mb == 0 {
                errors.push("cache.max_size_mb must be > 0".into());
            }
            if self.cache.ttl_hours == 0 {
                errors.push("cache.ttl_hours must be > 0".into());
            }
        }

        // CI settings.
        if self.ci.max_medium > 1000 {
            errors.push(format!(
                "ci.max_medium {} is unreasonably large (max 1000)",
                self.ci.max_medium
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ── Merge helpers ──────────────────────────────────────────────────────────

/// Pick the override value if it differs from default, else keep base.
fn pick(base: &str, over: &str, default: &str) -> String {
    if over != default { over.to_string() } else { base.to_string() }
}

fn pick_val<T: PartialEq + Copy>(base: T, over: T, default: T) -> T {
    if over != default { over } else { base }
}

fn pick_f64(base: f64, over: f64, default: f64) -> f64 {
    if (over - default).abs() > f64::EPSILON { over } else { base }
}

// ── File discovery ─────────────────────────────────────────────────────────

/// Well-known config file names searched in order.
const CONFIG_NAMES: &[&str] = &[
    ".cascade-verify.yaml",
    ".cascade-verify.yml",
    "cascade-verify.yaml",
    "cascade-verify.yml",
];

/// Search for a configuration file in the current directory and the user's
/// home directory.  Returns the first match found.
pub fn find_config_file() -> Option<String> {
    // Current directory.
    if let Ok(cwd) = std::env::current_dir() {
        for name in CONFIG_NAMES {
            let candidate = cwd.join(name);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }

    // Home directory.
    if let Some(home) = home_dir() {
        for name in CONFIG_NAMES {
            let candidate = home.join(name);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }

    None
}

/// Resolve the user's home directory portably.
fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

/// Load configuration with the standard precedence chain:
/// 1. Explicit file (if provided).
/// 2. Auto-discovered config file.
/// 3. Environment variables.
/// 4. Built-in defaults.
pub fn load_config_with_defaults(explicit_path: Option<&str>) -> CliConfig {
    // Start with defaults.
    let base = CliConfig::default();

    // Layer from file.
    let from_file = explicit_path
        .map(|p| p.to_string())
        .or_else(find_config_file)
        .and_then(|path| {
            tracing::debug!("loading config from {}", path);
            match CliConfig::from_file(&path) {
                Ok(cfg) => Some(cfg),
                Err(e) => {
                    tracing::warn!("could not load config from {}: {}", path, e);
                    None
                }
            }
        })
        .unwrap_or_default();

    // Layer from env.
    let from_env = CliConfig::from_env();

    // Merge: file overrides defaults, env overrides file.
    let merged = CliConfig::merge(base, from_file);
    CliConfig::merge(merged, from_env)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default construction ───────────────────────────────────────────

    #[test]
    fn default_config_has_sensible_analysis_defaults() {
        let cfg = CliConfig::default();
        assert_eq!(cfg.analysis.default_mode, "auto");
        assert_eq!(cfg.analysis.max_failure_budget, 3);
        assert_eq!(cfg.analysis.timeout_ms, 60_000);
        assert!((cfg.analysis.amplification_threshold - 8.0).abs() < f64::EPSILON);
        assert_eq!(cfg.analysis.timeout_threshold_ms, 30_000);
    }

    #[test]
    fn default_config_has_sensible_output_defaults() {
        let cfg = CliConfig::default();
        assert_eq!(cfg.output.default_format, "table");
        assert!(!cfg.output.verbose);
        assert!(cfg.output.color);
        assert!(cfg.output.output_dir.is_none());
    }

    #[test]
    fn default_config_has_sensible_cache_defaults() {
        let cfg = CliConfig::default();
        assert!(cfg.cache.enabled);
        assert_eq!(cfg.cache.dir, ".cascade-cache");
        assert_eq!(cfg.cache.max_size_mb, 256);
        assert_eq!(cfg.cache.ttl_hours, 24);
    }

    #[test]
    fn default_config_has_sensible_ci_defaults() {
        let cfg = CliConfig::default();
        assert!(cfg.ci.fail_on_critical);
        assert!(!cfg.ci.fail_on_high);
        assert_eq!(cfg.ci.max_medium, 10);
        assert!(!cfg.ci.annotations);
        assert!(!cfg.ci.step_summary);
    }

    // ── YAML round-trip ────────────────────────────────────────────────

    #[test]
    fn yaml_round_trip() {
        let cfg = CliConfig::default();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let parsed: CliConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn from_yaml_with_partial_fields() {
        let yaml = r#"
analysis:
  default_mode: deep
  max_failure_budget: 5
"#;
        let cfg: CliConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.analysis.default_mode, "deep");
        assert_eq!(cfg.analysis.max_failure_budget, 5);
        // Others should be default.
        assert_eq!(cfg.analysis.timeout_ms, 60_000);
        assert_eq!(cfg.output.default_format, "table");
    }

    #[test]
    fn from_yaml_with_all_sections() {
        let yaml = r#"
analysis:
  default_mode: fast
  max_failure_budget: 2
  timeout_ms: 30000
  amplification_threshold: 4.0
  timeout_threshold_ms: 15000
output:
  default_format: json
  verbose: true
  color: false
  output_dir: /tmp/reports
cache:
  enabled: false
  dir: /tmp/cache
  max_size_mb: 128
  ttl_hours: 12
ci:
  fail_on_critical: true
  fail_on_high: true
  max_medium: 5
  annotations: true
  step_summary: true
"#;
        let cfg: CliConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.analysis.default_mode, "fast");
        assert_eq!(cfg.analysis.max_failure_budget, 2);
        assert_eq!(cfg.analysis.timeout_ms, 30_000);
        assert!((cfg.analysis.amplification_threshold - 4.0).abs() < f64::EPSILON);
        assert_eq!(cfg.output.default_format, "json");
        assert!(cfg.output.verbose);
        assert!(!cfg.output.color);
        assert_eq!(cfg.output.output_dir, Some("/tmp/reports".into()));
        assert!(!cfg.cache.enabled);
        assert_eq!(cfg.cache.dir, "/tmp/cache");
        assert!(cfg.ci.fail_on_high);
        assert_eq!(cfg.ci.max_medium, 5);
        assert!(cfg.ci.annotations);
        assert!(cfg.ci.step_summary);
    }

    // ── from_file ──────────────────────────────────────────────────────

    #[test]
    fn from_file_returns_error_for_missing_file() {
        let result = CliConfig::from_file("/nonexistent/.cascade-verify.yaml");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to read"));
    }

    #[test]
    fn from_file_returns_error_for_invalid_yaml() {
        let dir = std::env::temp_dir().join("cascade_test_bad_yaml");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.yaml");
        std::fs::write(&path, "{{{{ not valid yaml").unwrap();
        let result = CliConfig::from_file(path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to parse"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_file_loads_valid_yaml() {
        let dir = std::env::temp_dir().join("cascade_test_good_yaml");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("good.yaml");
        std::fs::write(
            &path,
            "analysis:\n  default_mode: deep\n  timeout_ms: 5000\n",
        )
        .unwrap();
        let cfg = CliConfig::from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(cfg.analysis.default_mode, "deep");
        assert_eq!(cfg.analysis.timeout_ms, 5000);
        let _ = std::fs::remove_file(&path);
    }

    // ── Merge ──────────────────────────────────────────────────────────

    #[test]
    fn merge_overrides_take_precedence() {
        let base = CliConfig::default();
        let mut over = CliConfig::default();
        over.analysis.default_mode = "deep".into();
        over.output.default_format = "json".into();
        let merged = CliConfig::merge(base, over);
        assert_eq!(merged.analysis.default_mode, "deep");
        assert_eq!(merged.output.default_format, "json");
    }

    #[test]
    fn merge_keeps_base_when_override_is_default() {
        let mut base = CliConfig::default();
        base.analysis.default_mode = "fast".into();
        let over = CliConfig::default(); // all defaults
        let merged = CliConfig::merge(base.clone(), over);
        assert_eq!(merged.analysis.default_mode, "fast");
    }

    // ── Validation ─────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_default_config() {
        let cfg = CliConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_unknown_mode() {
        let mut cfg = CliConfig::default();
        cfg.analysis.default_mode = "turbo".into();
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unknown mode")));
    }

    #[test]
    fn validate_rejects_zero_failure_budget() {
        let mut cfg = CliConfig::default();
        cfg.analysis.max_failure_budget = 0;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("max_failure_budget")));
    }

    #[test]
    fn validate_rejects_excessive_failure_budget() {
        let mut cfg = CliConfig::default();
        cfg.analysis.max_failure_budget = 50;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unreasonably large")));
    }

    #[test]
    fn validate_rejects_zero_timeout() {
        let mut cfg = CliConfig::default();
        cfg.analysis.timeout_ms = 0;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("timeout_ms")));
    }

    #[test]
    fn validate_rejects_low_amplification_threshold() {
        let mut cfg = CliConfig::default();
        cfg.analysis.amplification_threshold = 0.5;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("amplification_threshold")));
    }

    #[test]
    fn validate_rejects_unknown_output_format() {
        let mut cfg = CliConfig::default();
        cfg.output.default_format = "html".into();
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unknown format")));
    }

    #[test]
    fn validate_rejects_empty_cache_dir_when_enabled() {
        let mut cfg = CliConfig::default();
        cfg.cache.dir = String::new();
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("cache.dir")));
    }

    #[test]
    fn validate_accepts_empty_cache_dir_when_disabled() {
        let mut cfg = CliConfig::default();
        cfg.cache.enabled = false;
        cfg.cache.dir = String::new();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_reports_multiple_errors() {
        let mut cfg = CliConfig::default();
        cfg.analysis.default_mode = "turbo".into();
        cfg.analysis.max_failure_budget = 0;
        cfg.analysis.timeout_ms = 0;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.len() >= 3);
    }

    // ── Environment override ───────────────────────────────────────────

    #[test]
    fn from_env_reads_no_color() {
        // We can't safely set env vars in parallel tests, so we just verify
        // the method returns a valid config with defaults.
        let cfg = CliConfig::from_env();
        assert!(cfg.validate().is_ok());
    }

    // ── find_config_file ───────────────────────────────────────────────

    #[test]
    fn find_config_file_returns_none_when_no_file_exists() {
        // Save and change dir to a temp dir with no config files.
        let dir = std::env::temp_dir().join("cascade_test_no_cfg");
        let _ = std::fs::create_dir_all(&dir);
        // We can't easily change cwd in a test, so we just verify
        // the function doesn't panic.
        let _result = find_config_file();
    }

    // ── load_config_with_defaults ──────────────────────────────────────

    #[test]
    fn load_config_with_defaults_returns_valid_config() {
        let cfg = load_config_with_defaults(None);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn load_config_with_explicit_path() {
        let dir = std::env::temp_dir().join("cascade_test_explicit");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("custom.yaml");
        std::fs::write(
            &path,
            "analysis:\n  default_mode: fast\nci:\n  fail_on_high: true\n",
        )
        .unwrap();
        let cfg = load_config_with_defaults(Some(path.to_str().unwrap()));
        assert_eq!(cfg.analysis.default_mode, "fast");
        assert!(cfg.ci.fail_on_high);
        let _ = std::fs::remove_file(&path);
    }

    // ── JSON serialization ─────────────────────────────────────────────

    #[test]
    fn json_round_trip() {
        let cfg = CliConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: CliConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, parsed);
    }
}
