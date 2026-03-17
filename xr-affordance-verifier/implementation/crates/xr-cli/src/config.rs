//! CLI configuration management.
//!
//! Supports loading config from files, environment variables, and CLI
//! arguments, with a well-defined merge order and search path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use xr_types::config::VerifierConfig;
use xr_types::error::{VerifierError, VerifierResult};

use crate::OutputFormat;

// ─── Environment variable prefix ───────────────────────────────────────────

const ENV_PREFIX: &str = "XR_VERIFY";

// ─── Config file names ─────────────────────────────────────────────────────

const CONFIG_FILE_NAMES: &[&str] = &[
    "xr-verify.json",
    ".xr-verify.json",
    ".config/xr-verify/config.json",
];

// ─── CliConfig ─────────────────────────────────────────────────────────────

/// CLI-specific configuration wrapping the verifier config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Core verifier configuration.
    pub verifier: VerifierConfig,
    /// Output format preference.
    #[serde(default = "default_format")]
    pub format: OutputFormat,
    /// Verbosity level (0–4).
    #[serde(default = "default_verbosity")]
    pub verbosity: u8,
    /// Whether colored output is disabled.
    #[serde(default)]
    pub no_color: bool,
    /// Default output directory for reports and certificates.
    #[serde(default = "default_output_dir")]
    pub default_output_dir: PathBuf,
    /// Whether to show timing information.
    #[serde(default = "default_true")]
    pub show_timing: bool,
    /// Whether to show progress indicators.
    #[serde(default = "default_true")]
    pub show_progress: bool,
    /// Custom aliases for scene paths.
    #[serde(default)]
    pub scene_aliases: HashMap<String, PathBuf>,
}

fn default_format() -> OutputFormat {
    OutputFormat::Text
}

fn default_verbosity() -> u8 {
    2
}

fn default_output_dir() -> PathBuf {
    PathBuf::from(".")
}

fn default_true() -> bool {
    true
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            verifier: VerifierConfig::default(),
            format: OutputFormat::Text,
            verbosity: 2,
            no_color: false,
            default_output_dir: PathBuf::from("."),
            show_timing: true,
            show_progress: true,
            scene_aliases: HashMap::new(),
        }
    }
}

impl CliConfig {
    /// Load configuration with the following precedence (highest first):
    /// 1. CLI arguments
    /// 2. Environment variables
    /// 3. Explicit config file path
    /// 4. Auto-discovered config file
    /// 5. Default values
    pub fn load_with_overrides(
        config_path: Option<&Path>,
        format: OutputFormat,
        verbosity: u8,
        no_color: bool,
    ) -> Self {
        let mut config = if let Some(path) = config_path {
            Self::load_from_file(path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load config from {}: {}", path.display(), e);
                Self::auto_discover().unwrap_or_default()
            })
        } else {
            Self::auto_discover().unwrap_or_default()
        };

        // Apply environment variable overrides
        config.apply_env_overrides();

        // Apply CLI argument overrides
        config.format = format;
        config.verbosity = verbosity;
        if no_color {
            config.no_color = true;
        }

        // Check NO_COLOR environment variable (standard convention)
        if std::env::var("NO_COLOR").is_ok() {
            config.no_color = true;
        }

        config
    }

    /// Load configuration from a specific file.
    pub fn load_from_file(path: &Path) -> VerifierResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents).map_err(|e| {
            VerifierError::Configuration(format!(
                "Failed to parse config file {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(config)
    }

    /// Try to auto-discover a configuration file.
    pub fn auto_discover() -> Option<Self> {
        for path in Self::config_search_paths() {
            if path.exists() {
                tracing::debug!("Found config file: {}", path.display());
                match Self::load_from_file(&path) {
                    Ok(config) => return Some(config),
                    Err(e) => {
                        tracing::warn!(
                            "Found but failed to load {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }
        None
    }

    /// Return the ordered list of paths searched for configuration files.
    pub fn config_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // Current directory
        for name in CONFIG_FILE_NAMES {
            paths.push(PathBuf::from(name));
        }

        // Home directory
        if let Some(home) = dirs_home() {
            for name in CONFIG_FILE_NAMES {
                paths.push(home.join(name));
            }
        }

        // XDG config dir
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            paths.push(PathBuf::from(xdg).join("xr-verify/config.json"));
        }

        // System-wide (Unix)
        #[cfg(unix)]
        {
            paths.push(PathBuf::from("/etc/xr-verify/config.json"));
        }

        paths
    }

    /// Apply environment variable overrides.
    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_SAMPLES")) {
            if let Ok(n) = val.parse::<usize>() {
                self.verifier.sampling.num_samples = n;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_SMT_TIMEOUT")) {
            if let Ok(t) = val.parse::<f64>() {
                self.verifier.smt.timeout_s = t;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_VERBOSITY")) {
            if let Ok(v) = val.parse::<u8>() {
                self.verbosity = v;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_FORMAT")) {
            match val.to_lowercase().as_str() {
                "json" => self.format = OutputFormat::Json,
                "compact" => self.format = OutputFormat::Compact,
                _ => self.format = OutputFormat::Text,
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_OUTPUT_DIR")) {
            self.default_output_dir = PathBuf::from(val);
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_NO_COLOR")) {
            if val == "1" || val.eq_ignore_ascii_case("true") {
                self.no_color = true;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_TIER2")) {
            if val == "0" || val.eq_ignore_ascii_case("false") {
                self.verifier.tier2.enabled = false;
            }
        }

        if let Ok(val) = std::env::var(format!("{ENV_PREFIX}_SEED")) {
            if let Ok(s) = val.parse::<u64>() {
                self.verifier.sampling.seed = s;
            }
        }
    }

    /// Serialize configuration to pretty JSON.
    pub fn to_json(&self) -> VerifierResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| VerifierError::Configuration(format!("JSON serialization error: {e}")))
    }

    /// Save configuration to a file.
    pub fn save(&self, path: &Path) -> VerifierResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Generate a default configuration template with comments.
    pub fn generate_template() -> String {
        let config = Self::default();
        // We serialize with nice formatting
        serde_json::to_string_pretty(&config).unwrap_or_else(|_| "{}".into())
    }

    /// Merge another config into this one (other takes precedence for non-default fields).
    pub fn merge_from(&mut self, other: &CliConfig) {
        // Merge verifier config fields selectively
        if other.verifier.sampling.num_samples != VerifierConfig::default().sampling.num_samples {
            self.verifier.sampling.num_samples = other.verifier.sampling.num_samples;
        }
        if other.verifier.smt.timeout_s != VerifierConfig::default().smt.timeout_s {
            self.verifier.smt.timeout_s = other.verifier.smt.timeout_s;
        }
        if other.verbosity != default_verbosity() {
            self.verbosity = other.verbosity;
        }
        if other.no_color {
            self.no_color = true;
        }

        // Merge scene aliases
        for (k, v) in &other.scene_aliases {
            self.scene_aliases.insert(k.clone(), v.clone());
        }
    }

    /// Resolve a scene path, checking aliases first.
    pub fn resolve_scene_path(&self, input: &Path) -> PathBuf {
        if let Some(alias) = input.to_str().and_then(|s| self.scene_aliases.get(s)) {
            alias.clone()
        } else {
            input.to_path_buf()
        }
    }

    /// Get all environment variable names that are recognized.
    pub fn env_var_names() -> Vec<(&'static str, &'static str)> {
        vec![
            ("XR_VERIFY_SAMPLES", "Number of population samples"),
            ("XR_VERIFY_SMT_TIMEOUT", "SMT solver timeout in seconds"),
            ("XR_VERIFY_VERBOSITY", "Verbosity level (0-4)"),
            ("XR_VERIFY_FORMAT", "Output format (text/json/compact)"),
            ("XR_VERIFY_OUTPUT_DIR", "Default output directory"),
            ("XR_VERIFY_NO_COLOR", "Disable colored output (1/true)"),
            ("XR_VERIFY_TIER2", "Enable Tier 2 verification (0/false to disable)"),
            ("XR_VERIFY_SEED", "Random seed for sampling"),
            ("NO_COLOR", "Standard no-color convention"),
        ]
    }
}

/// Get the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    // Simple cross-platform home directory detection
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .ok()
}

// Custom serialize/deserialize for OutputFormat since it's defined in main
impl Serialize for OutputFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            OutputFormat::Text => serializer.serialize_str("text"),
            OutputFormat::Json => serializer.serialize_str("json"),
            OutputFormat::Compact => serializer.serialize_str("compact"),
        }
    }
}

impl<'de> Deserialize<'de> for OutputFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "compact" => Ok(OutputFormat::Compact),
            _ => Ok(OutputFormat::Text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cli_config() {
        let config = CliConfig::default();
        assert_eq!(config.format, OutputFormat::Text);
        assert_eq!(config.verbosity, 2);
        assert!(!config.no_color);
        assert!(config.show_timing);
        assert!(config.show_progress);
    }

    #[test]
    fn test_config_search_paths() {
        let paths = CliConfig::config_search_paths();
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.ends_with("xr-verify.json")));
    }

    #[test]
    fn test_config_serialization() {
        let config = CliConfig::default();
        let json = config.to_json().unwrap();
        assert!(json.contains("verifier"));
        assert!(json.contains("format"));

        let parsed: CliConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.format, config.format);
        assert_eq!(parsed.verbosity, config.verbosity);
    }

    #[test]
    fn test_generate_template() {
        let template = CliConfig::generate_template();
        assert!(template.contains("verifier"));
        assert!(template.contains("format"));
    }

    #[test]
    fn test_merge_configs() {
        let mut base = CliConfig::default();
        let mut override_cfg = CliConfig::default();
        override_cfg.no_color = true;
        override_cfg.scene_aliases.insert(
            "test".into(),
            PathBuf::from("/tmp/test.json"),
        );

        base.merge_from(&override_cfg);
        assert!(base.no_color);
        assert!(base.scene_aliases.contains_key("test"));
    }

    #[test]
    fn test_resolve_scene_path() {
        let mut config = CliConfig::default();
        config.scene_aliases.insert(
            "demo".into(),
            PathBuf::from("/scenes/demo.json"),
        );

        let resolved = config.resolve_scene_path(Path::new("demo"));
        assert_eq!(resolved, PathBuf::from("/scenes/demo.json"));

        let resolved = config.resolve_scene_path(Path::new("other.json"));
        assert_eq!(resolved, PathBuf::from("other.json"));
    }

    #[test]
    fn test_env_var_names() {
        let vars = CliConfig::env_var_names();
        assert!(!vars.is_empty());
        assert!(vars.iter().any(|(name, _)| *name == "XR_VERIFY_SAMPLES"));
        assert!(vars.iter().any(|(name, _)| *name == "NO_COLOR"));
    }

    #[test]
    fn test_load_with_overrides_defaults() {
        let config = CliConfig::load_with_overrides(
            None,
            OutputFormat::Json,
            3,
            true,
        );
        assert_eq!(config.format, OutputFormat::Json);
        assert_eq!(config.verbosity, 3);
        assert!(config.no_color);
    }

    #[test]
    fn test_output_format_serde() {
        let json = serde_json::to_string(&OutputFormat::Json).unwrap();
        assert_eq!(json, "\"json\"");

        let parsed: OutputFormat = serde_json::from_str("\"compact\"").unwrap();
        assert_eq!(parsed, OutputFormat::Compact);

        let parsed: OutputFormat = serde_json::from_str("\"text\"").unwrap();
        assert_eq!(parsed, OutputFormat::Text);
    }
}
