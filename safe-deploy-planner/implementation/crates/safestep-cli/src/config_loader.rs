//! Configuration loading with layered precedence: CLI > env > file > defaults.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Full SafeStep configuration with all sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeStepConfig {
    pub planner: PlannerSection,
    pub solver: SolverSection,
    pub output: OutputSection,
    pub kubernetes: KubernetesSection,
    pub schema: SchemaSection,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Planner-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerSection {
    pub max_depth: usize,
    #[serde(with = "duration_secs")]
    pub timeout: Duration,
    pub use_cegar: bool,
    pub use_parallel: bool,
    pub max_cegar_iterations: usize,
    pub completeness_check: bool,
    pub treewidth_threshold: usize,
    pub optimization_objectives: Vec<String>,
}

/// Solver tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverSection {
    pub restart_strategy: String,
    pub restart_base_interval: u64,
    pub clause_deletion_threshold: f64,
    pub max_learned_clauses: usize,
    pub phase_policy: String,
    pub var_selection: String,
}

/// Output formatting defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
    pub format: String,
    pub color: String,
    pub pager: bool,
    pub max_width: usize,
}

/// Kubernetes integration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesSection {
    pub namespace: String,
    pub context: Option<String>,
    pub dry_run: bool,
    pub health_check_timeout_secs: u64,
}

/// Schema analysis settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSection {
    pub default_format: String,
    pub min_confidence: f64,
    pub ignore_deprecated: bool,
}

impl Default for SafeStepConfig {
    fn default() -> Self {
        Self {
            planner: PlannerSection::default(),
            solver: SolverSection::default(),
            output: OutputSection::default(),
            kubernetes: KubernetesSection::default(),
            schema: SchemaSection::default(),
            extra: HashMap::new(),
        }
    }
}

impl Default for PlannerSection {
    fn default() -> Self {
        Self {
            max_depth: 100,
            timeout: Duration::from_secs(300),
            use_cegar: true,
            use_parallel: true,
            max_cegar_iterations: 50,
            completeness_check: true,
            treewidth_threshold: 4,
            optimization_objectives: vec!["steps".into()],
        }
    }
}

impl Default for SolverSection {
    fn default() -> Self {
        Self {
            restart_strategy: "luby".into(),
            restart_base_interval: 100,
            clause_deletion_threshold: 0.5,
            max_learned_clauses: 100_000,
            phase_policy: "phase_saving".into(),
            var_selection: "vsids".into(),
        }
    }
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            format: "text".into(),
            color: "auto".into(),
            pager: false,
            max_width: 120,
        }
    }
}

impl Default for KubernetesSection {
    fn default() -> Self {
        Self {
            namespace: "default".into(),
            context: None,
            dry_run: false,
            health_check_timeout_secs: 120,
        }
    }
}

impl Default for SchemaSection {
    fn default() -> Self {
        Self {
            default_format: "openapi".into(),
            min_confidence: 0.8,
            ignore_deprecated: false,
        }
    }
}

mod duration_secs {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Loads and merges configuration from multiple sources.
pub struct ConfigLoader {
    file_config: Option<SafeStepConfig>,
    env_overrides: HashMap<String, String>,
    cli_overrides: HashMap<String, String>,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self {
            file_config: None,
            env_overrides: HashMap::new(),
            cli_overrides: HashMap::new(),
        }
    }

    /// Attempt to load configuration from a file path.
    pub fn load_file(&mut self, path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;

        let config: SafeStepConfig = if path
            .extension()
            .map_or(false, |e| e == "yaml" || e == "yml")
        {
            serde_yaml::from_str(&content)
                .with_context(|| format!("failed to parse YAML config: {}", path.display()))?
        } else {
            serde_json::from_str(&content)
                .with_context(|| format!("failed to parse JSON config: {}", path.display()))?
        };

        self.validate_config(&config)?;
        info!(path = %path.display(), "loaded configuration file");
        self.file_config = Some(config);
        Ok(())
    }

    /// Scan environment variables for SAFESTEP_* overrides.
    pub fn load_env(&mut self) {
        let prefix = "SAFESTEP_";
        for (key, value) in std::env::vars() {
            if let Some(stripped) = key.strip_prefix(prefix) {
                let config_key = stripped.to_lowercase().replace('_', ".");
                debug!(key = %config_key, value = %value, "found env override");
                self.env_overrides.insert(config_key, value);
            }
        }
    }

    /// Register CLI argument overrides.
    pub fn set_cli_override(&mut self, key: &str, value: String) {
        self.cli_overrides.insert(key.to_string(), value);
    }

    /// Build the final merged configuration.
    pub fn build(self) -> SafeStepConfig {
        let mut config = self.file_config.unwrap_or_default();
        Self::apply_overrides_static(&mut config, &self.env_overrides, "env");
        Self::apply_overrides_static(&mut config, &self.cli_overrides, "cli");
        config
    }

    fn apply_overrides_static(
        config: &mut SafeStepConfig,
        overrides: &HashMap<String, String>,
        source: &str,
    ) {
        for (key, value) in overrides {
            let applied = match key.as_str() {
                "planner.max_depth" | "max.depth" => {
                    value.parse::<usize>().map(|v| config.planner.max_depth = v).is_ok()
                }
                "planner.timeout" | "timeout" => {
                    value.parse::<u64>().map(|v| config.planner.timeout = Duration::from_secs(v)).is_ok()
                }
                "planner.use_cegar" | "cegar" => {
                    value.parse::<bool>().map(|v| config.planner.use_cegar = v).is_ok()
                }
                "planner.treewidth_threshold" | "treewidth.threshold" => {
                    value.parse::<usize>().map(|v| config.planner.treewidth_threshold = v).is_ok()
                }
                "solver.restart_strategy" | "restart.strategy" => {
                    config.solver.restart_strategy = value.clone();
                    true
                }
                "solver.phase_policy" | "phase.policy" => {
                    config.solver.phase_policy = value.clone();
                    true
                }
                "output.format" | "format" => {
                    config.output.format = value.clone();
                    true
                }
                "output.color" | "color" => {
                    config.output.color = value.clone();
                    true
                }
                "kubernetes.namespace" | "namespace" => {
                    config.kubernetes.namespace = value.clone();
                    true
                }
                "schema.default_format" | "schema.format" => {
                    config.schema.default_format = value.clone();
                    true
                }
                "schema.min_confidence" | "min.confidence" => {
                    value.parse::<f64>().map(|v| config.schema.min_confidence = v).is_ok()
                }
                _ => {
                    config.extra.insert(key.clone(), serde_json::Value::String(value.clone()));
                    true
                }
            };
            if applied {
                debug!(source, key, value, "applied config override");
            } else {
                warn!(source, key, value, "failed to apply config override");
            }
        }
    }

    /// Validate a loaded configuration for semantic correctness.
    fn validate_config(&self, config: &SafeStepConfig) -> Result<()> {
        if config.planner.max_depth == 0 {
            anyhow::bail!("planner.max_depth must be > 0");
        }
        if config.planner.timeout.as_secs() == 0 {
            anyhow::bail!("planner.timeout must be > 0");
        }
        if config.planner.treewidth_threshold == 0 {
            anyhow::bail!("planner.treewidth_threshold must be > 0");
        }
        if config.solver.clause_deletion_threshold < 0.0
            || config.solver.clause_deletion_threshold > 1.0
        {
            anyhow::bail!("solver.clause_deletion_threshold must be in [0.0, 1.0]");
        }
        if config.schema.min_confidence < 0.0 || config.schema.min_confidence > 1.0 {
            anyhow::bail!("schema.min_confidence must be in [0.0, 1.0]");
        }
        let valid_restarts = ["luby", "geometric", "fixed", "never"];
        if !valid_restarts.contains(&config.solver.restart_strategy.as_str()) {
            anyhow::bail!(
                "solver.restart_strategy must be one of: {}",
                valid_restarts.join(", ")
            );
        }
        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to auto-discover a config file in standard locations.
pub fn discover_config_file() -> Option<PathBuf> {
    let candidates = [
        "safestep.yaml", "safestep.yml", "safestep.json",
        ".safestep.yaml", ".safestep.yml", ".safestep.json",
    ];
    for name in &candidates {
        let p = PathBuf::from(name);
        if p.exists() {
            debug!(path = %p.display(), "discovered config file");
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let home_config = PathBuf::from(home).join(".config").join("safestep").join("config.yaml");
        if home_config.exists() {
            return Some(home_config);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SafeStepConfig::default();
        assert_eq!(config.planner.max_depth, 100);
        assert_eq!(config.planner.timeout, Duration::from_secs(300));
        assert!(config.planner.use_cegar);
        assert_eq!(config.solver.restart_strategy, "luby");
        assert_eq!(config.output.format, "text");
        assert_eq!(config.kubernetes.namespace, "default");
        assert_eq!(config.schema.min_confidence, 0.8);
    }

    #[test]
    fn test_config_loader_defaults() {
        let loader = ConfigLoader::new();
        let config = loader.build();
        assert_eq!(config.planner.max_depth, 100);
    }

    #[test]
    fn test_config_loader_cli_overrides() {
        let mut loader = ConfigLoader::new();
        loader.set_cli_override("planner.max_depth", "50".to_string());
        loader.set_cli_override("planner.timeout", "120".to_string());
        loader.set_cli_override("output.format", "json".to_string());
        let config = loader.build();
        assert_eq!(config.planner.max_depth, 50);
        assert_eq!(config.planner.timeout, Duration::from_secs(120));
        assert_eq!(config.output.format, "json");
    }

    #[test]
    fn test_config_loader_json_file() {
        let dir = std::env::temp_dir().join("safestep_test_json_cl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let json = serde_json::to_string_pretty(&SafeStepConfig::default()).unwrap();
        std::fs::write(&path, &json).unwrap();

        let mut loader = ConfigLoader::new();
        loader.load_file(&path).unwrap();
        let config = loader.build();
        assert_eq!(config.planner.max_depth, 100);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_loader_yaml_file() {
        let dir = std::env::temp_dir().join("safestep_test_yaml_cl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        let mut config = SafeStepConfig::default();
        config.planner.max_depth = 42;
        let yaml = serde_yaml::to_string(&config).unwrap();
        std::fs::write(&path, &yaml).unwrap();

        let mut loader = ConfigLoader::new();
        loader.load_file(&path).unwrap();
        let loaded = loader.build();
        assert_eq!(loaded.planner.max_depth, 42);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_loader_precedence() {
        let dir = std::env::temp_dir().join("safestep_test_prec_cl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let mut config = SafeStepConfig::default();
        config.planner.max_depth = 10;
        std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let mut loader = ConfigLoader::new();
        loader.load_file(&path).unwrap();
        loader.set_cli_override("planner.max_depth", "99".to_string());
        let result = loader.build();
        assert_eq!(result.planner.max_depth, 99);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_validation_max_depth_zero() {
        let mut config = SafeStepConfig::default();
        config.planner.max_depth = 0;
        let loader = ConfigLoader::new();
        assert!(loader.validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_restart() {
        let mut config = SafeStepConfig::default();
        config.solver.restart_strategy = "invalid".into();
        let loader = ConfigLoader::new();
        assert!(loader.validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_confidence() {
        let mut config = SafeStepConfig::default();
        config.schema.min_confidence = 1.5;
        let loader = ConfigLoader::new();
        assert!(loader.validate_config(&config).is_err());
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = SafeStepConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SafeStepConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.planner.max_depth, config.planner.max_depth);
        assert_eq!(parsed.solver.restart_strategy, config.solver.restart_strategy);
    }

    #[test]
    fn test_unknown_override_stored_in_extra() {
        let mut loader = ConfigLoader::new();
        loader.set_cli_override("custom.key", "custom_value".to_string());
        let config = loader.build();
        assert_eq!(
            config.extra.get("custom.key"),
            Some(&serde_json::Value::String("custom_value".to_string()))
        );
    }

    #[test]
    fn test_config_loader_nonexistent_file() {
        let mut loader = ConfigLoader::new();
        assert!(loader.load_file(Path::new("/tmp/nonexistent_safestep_config_xyz.json")).is_err());
    }

    #[test]
    fn test_config_loader_invalid_json() {
        let dir = std::env::temp_dir().join("safestep_test_invalid_cl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "not valid json {{{").unwrap();
        let mut loader = ConfigLoader::new();
        assert!(loader.load_file(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
