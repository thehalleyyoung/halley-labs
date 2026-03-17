use serde::{Deserialize, Serialize};
use std::path::Path;

/// Full application configuration, merging CLI args and TOML config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub solver: SolverConfig,
    pub encoding: EncodingConfig,
    pub pareto: ParetoConfig,
    pub planner: PlannerConfig,
    pub certificate: CertificateConfig,
    pub output: OutputConfig,
    pub benchmark: BenchmarkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub backend: String,
    pub timeout_seconds: u64,
    pub max_iterations: u64,
    pub extract_conflicts: bool,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub target: String,
    pub include_soft_constraints: bool,
    pub normalize_costs: bool,
    pub cost_currency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    pub epsilon: f64,
    pub max_iterations: usize,
    pub objectives: Vec<String>,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    pub max_parallel_tasks: usize,
    pub working_days_per_week: u32,
    pub default_effort_days: f64,
    pub budget_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    pub issuer: String,
    pub include_proof_steps: bool,
    pub hash_algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub color: bool,
    pub table_width: usize,
    pub progress_bar: bool,
    pub decimal_places: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub collect_memory: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            backend: "smt".into(),
            timeout_seconds: 300,
            max_iterations: 10000,
            extract_conflicts: true,
            seed: None,
        }
    }
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            target: "smt".into(),
            include_soft_constraints: false,
            normalize_costs: true,
            cost_currency: "USD".into(),
        }
    }
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.001,
            max_iterations: 100,
            objectives: vec!["cost".into(), "compliance".into(), "risk".into()],
            normalize: true,
        }
    }
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: 4,
            working_days_per_week: 5,
            default_effort_days: 10.0,
            budget_limit: None,
        }
    }
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            issuer: "regsynth".into(),
            include_proof_steps: true,
            hash_algorithm: "sha256".into(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            color: true,
            table_width: 80,
            progress_bar: true,
            decimal_places: 4,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 1,
            collect_memory: false,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            encoding: EncodingConfig::default(),
            pareto: ParetoConfig::default(),
            planner: PlannerConfig::default(),
            certificate: CertificateConfig::default(),
            output: OutputConfig::default(),
            benchmark: BenchmarkConfig::default(),
        }
    }
}

impl AppConfig {
    /// Load configuration: first from TOML file (if specified), then override with CLI args.
    pub fn load(config_path: Option<&Path>, cli: &crate::Cli) -> anyhow::Result<Self> {
        let mut cfg = if let Some(path) = config_path {
            log::info!("Loading configuration from {}", path.display());
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", path.display(), e))?;
            toml::from_str::<AppConfig>(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse config file {}: {}", path.display(), e))?
        } else {
            AppConfig::default()
        };

        // Override solver backend from CLI
        if let Some(ref solver) = cli.solver {
            cfg.solver.backend = solver.to_string().to_lowercase();
        }

        // Override timeout from CLI
        if let Some(timeout) = cli.timeout {
            cfg.solver.timeout_seconds = timeout;
        }

        cfg.validate()?;

        Ok(cfg)
    }

    /// Validate configuration values.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.solver.timeout_seconds == 0 {
            anyhow::bail!("Solver timeout must be greater than 0");
        }
        if self.pareto.epsilon <= 0.0 || self.pareto.epsilon >= 1.0 {
            anyhow::bail!("Pareto epsilon must be in (0, 1), got {}", self.pareto.epsilon);
        }
        if self.pareto.max_iterations == 0 {
            anyhow::bail!("Pareto max_iterations must be greater than 0");
        }
        if self.planner.max_parallel_tasks == 0 {
            anyhow::bail!("Planner max_parallel_tasks must be greater than 0");
        }
        if self.output.table_width < 40 {
            anyhow::bail!("Output table_width must be at least 40");
        }
        Ok(())
    }

    /// Generate default config as TOML string for `regsynth init`.
    pub fn generate_default_toml() -> String {
        let cfg = AppConfig::default();
        toml::to_string_pretty(&cfg).unwrap_or_else(|_| "# Failed to generate config".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = AppConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_invalid_epsilon() {
        let mut cfg = AppConfig::default();
        cfg.pareto.epsilon = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_timeout() {
        let mut cfg = AppConfig::default();
        cfg.solver.timeout_seconds = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_generate_default_toml() {
        let toml_str = AppConfig::generate_default_toml();
        assert!(toml_str.contains("[solver]"));
        assert!(toml_str.contains("[encoding]"));
        assert!(toml_str.contains("[pareto]"));
    }

    #[test]
    fn test_roundtrip_toml() {
        let cfg = AppConfig::default();
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let parsed: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.solver.timeout_seconds, cfg.solver.timeout_seconds);
        assert_eq!(parsed.pareto.epsilon, cfg.pareto.epsilon);
    }

    #[test]
    fn test_invalid_table_width() {
        let mut cfg = AppConfig::default();
        cfg.output.table_width = 10;
        assert!(cfg.validate().is_err());
    }
}
