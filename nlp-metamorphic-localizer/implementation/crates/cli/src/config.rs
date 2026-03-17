//! CLI configuration loading from TOML files with command-line overrides.

use serde::{Deserialize, Serialize};

// ── Configuration structs ───────────────────────────────────────────────────

/// Top-level CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(default)]
    pub pipeline: CliPipelineConfig,
    #[serde(default)]
    pub localization: CliLocalizationConfig,
    #[serde(default)]
    pub shrinking: CliShrinkingConfig,
    #[serde(default)]
    pub output: CliOutputConfig,
}

impl CliConfig {
    /// Load configuration from a TOML file.
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", path, e))?;
        let config: CliConfig = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file '{}': {}", path, e))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.localization.threshold < 0.0 || self.localization.threshold > 1.0 {
            anyhow::bail!(
                "Localization threshold must be in [0, 1], got {}",
                self.localization.threshold
            );
        }
        if self.localization.max_suspects == 0 {
            anyhow::bail!("max_suspects must be > 0");
        }
        if self.shrinking.max_time == 0 {
            anyhow::bail!("shrinking max_time must be > 0");
        }
        Ok(())
    }

    /// Generate a default configuration as a TOML string.
    pub fn generate_default_toml() -> String {
        let default = Self::default();
        toml::to_string_pretty(&default).unwrap_or_else(|_| "# Failed to generate".into())
    }

    /// Merge with CLI arguments (CLI takes precedence over file config).
    pub fn merge_with_args(
        &mut self,
        metric: Option<&str>,
        threshold: Option<f64>,
        max_suspects: Option<usize>,
        enable_causal: Option<bool>,
        enable_peeling: Option<bool>,
        verbosity: Option<u8>,
        format: Option<&str>,
    ) {
        if let Some(m) = metric {
            self.localization.metric = m.to_string();
        }
        if let Some(t) = threshold {
            self.localization.threshold = t;
        }
        if let Some(ms) = max_suspects {
            self.localization.max_suspects = ms;
        }
        if let Some(c) = enable_causal {
            self.localization.enable_causal = c;
        }
        if let Some(p) = enable_peeling {
            self.localization.enable_peeling = p;
        }
        if let Some(v) = verbosity {
            self.output.verbosity = v;
        }
        if let Some(f) = format {
            self.output.format = f.to_string();
        }
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            pipeline: CliPipelineConfig::default(),
            localization: CliLocalizationConfig::default(),
            shrinking: CliShrinkingConfig::default(),
            output: CliOutputConfig::default(),
        }
    }
}

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliPipelineConfig {
    pub adapter_type: String,
    pub stages: Vec<String>,
    pub model_path: Option<String>,
}

impl Default for CliPipelineConfig {
    fn default() -> Self {
        Self {
            adapter_type: "spacy-like".into(),
            stages: vec![
                "tokenizer".into(),
                "pos_tagger".into(),
                "dependency_parser".into(),
                "ner".into(),
                "sentiment".into(),
            ],
            model_path: None,
        }
    }
}

/// Localization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliLocalizationConfig {
    pub metric: String,
    pub threshold: f64,
    pub max_suspects: usize,
    pub enable_causal: bool,
    pub enable_peeling: bool,
}

impl Default for CliLocalizationConfig {
    fn default() -> Self {
        Self {
            metric: "ochiai".into(),
            threshold: 0.1,
            max_suspects: 3,
            enable_causal: false,
            enable_peeling: false,
        }
    }
}

/// Shrinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliShrinkingConfig {
    pub max_time: u64,
    pub enable_binary_search: bool,
    pub min_size: usize,
}

impl Default for CliShrinkingConfig {
    fn default() -> Self {
        Self {
            max_time: 30_000,
            enable_binary_search: true,
            min_size: 3,
        }
    }
}

/// Output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliOutputConfig {
    pub format: String,
    pub verbosity: u8,
    pub color: bool,
}

impl Default for CliOutputConfig {
    fn default() -> Self {
        Self {
            format: "json".into(),
            verbosity: 1,
            color: true,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = CliConfig::default();
        assert_eq!(cfg.localization.metric, "ochiai");
        assert_eq!(cfg.localization.max_suspects, 3);
        assert_eq!(cfg.pipeline.stages.len(), 5);
        assert!(cfg.pipeline.model_path.is_none());
    }

    #[test]
    fn test_validate_valid() {
        let cfg = CliConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_threshold() {
        let mut cfg = CliConfig::default();
        cfg.localization.threshold = 2.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_suspects() {
        let mut cfg = CliConfig::default();
        cfg.localization.max_suspects = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_max_time() {
        let mut cfg = CliConfig::default();
        cfg.shrinking.max_time = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_merge_with_args() {
        let mut cfg = CliConfig::default();
        cfg.merge_with_args(
            Some("dstar"),
            Some(0.2),
            Some(5),
            Some(true),
            None,
            Some(2),
            Some("markdown"),
        );
        assert_eq!(cfg.localization.metric, "dstar");
        assert_eq!(cfg.localization.threshold, 0.2);
        assert_eq!(cfg.localization.max_suspects, 5);
        assert!(cfg.localization.enable_causal);
        assert_eq!(cfg.output.verbosity, 2);
        assert_eq!(cfg.output.format, "markdown");
    }

    #[test]
    fn test_merge_with_none_args() {
        let mut cfg = CliConfig::default();
        let original_metric = cfg.localization.metric.clone();
        cfg.merge_with_args(None, None, None, None, None, None, None);
        assert_eq!(cfg.localization.metric, original_metric);
    }

    #[test]
    fn test_generate_default_toml() {
        let toml_str = CliConfig::generate_default_toml();
        assert!(toml_str.contains("ochiai"));
        assert!(toml_str.contains("tokenizer"));
    }

    #[test]
    fn test_roundtrip_toml() {
        let cfg = CliConfig::default();
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let parsed: CliConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.localization.metric, cfg.localization.metric);
        assert_eq!(parsed.pipeline.stages.len(), cfg.pipeline.stages.len());
    }

    #[test]
    fn test_load_from_invalid_path() {
        let result = CliConfig::load_from_file("/nonexistent/path.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let pc = CliPipelineConfig::default();
        assert_eq!(pc.adapter_type, "spacy-like");
        assert!(pc.stages.contains(&"tokenizer".to_string()));
    }
}
