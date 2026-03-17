use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub solver: SolverConfig,
    pub pareto: ParetoConfig,
    pub temporal: TemporalConfig,
    pub output: OutputConfig,
    pub pipeline: PipelineOptions,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            solver: SolverConfig::default(),
            pareto: ParetoConfig::default(),
            temporal: TemporalConfig::default(),
            output: OutputConfig::default(),
            pipeline: PipelineOptions::default(),
        }
    }
}

impl PipelineConfig {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.solver.timeout_seconds == 0 {
            errors.push("Solver timeout must be > 0".to_string());
        }
        if self.pareto.epsilon <= 0.0 || self.pareto.epsilon >= 1.0 {
            errors.push("Pareto epsilon must be in (0, 1)".to_string());
        }
        if self.temporal.horizon == 0 {
            errors.push("Temporal horizon must be > 0".to_string());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub backend: SolverBackend,
    pub timeout_seconds: u64,
    pub memory_limit_mb: u64,
    pub random_seed: u64,
    pub verbosity: u32,
    pub incremental: bool,
    pub produce_proofs: bool,
    pub produce_unsat_cores: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            backend: SolverBackend::MaxSmt,
            timeout_seconds: 900,
            memory_limit_mb: 4096,
            random_seed: 42,
            verbosity: 1,
            incremental: true,
            produce_proofs: true,
            produce_unsat_cores: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverBackend {
    Sat,
    Smt,
    MaxSmt,
    Ilp,
    Combined,
}

impl fmt::Display for SolverBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverBackend::Sat => write!(f, "SAT"),
            SolverBackend::Smt => write!(f, "SMT"),
            SolverBackend::MaxSmt => write!(f, "MaxSMT"),
            SolverBackend::Ilp => write!(f, "ILP"),
            SolverBackend::Combined => write!(f, "Combined"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    pub epsilon: f64,
    pub max_iterations: usize,
    pub max_frontier_size: usize,
    pub scalarization_method: ScalarizationMethod,
    pub weight_generation: WeightGeneration,
    pub use_incremental: bool,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        ParetoConfig {
            epsilon: 0.01,
            max_iterations: 500,
            max_frontier_size: 200,
            scalarization_method: ScalarizationMethod::EpsilonConstraint,
            weight_generation: WeightGeneration::Uniform,
            use_incremental: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarizationMethod {
    WeightedSum,
    EpsilonConstraint,
    Chebyshev,
    NormalBoundary,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightGeneration {
    Uniform,
    Random,
    NBI,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub horizon: usize,
    pub timestep_months: usize,
    pub max_budget_per_step: f64,
    pub max_changes_per_step: usize,
    pub enable_trajectory_optimization: bool,
    pub enable_bisimulation_reduction: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        TemporalConfig {
            horizon: 10,
            timestep_months: 3,
            max_budget_per_step: 1_000_000.0,
            max_changes_per_step: 20,
            enable_trajectory_optimization: true,
            enable_bisimulation_reduction: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub include_certificates: bool,
    pub include_proofs: bool,
    pub include_visualizations: bool,
    pub output_directory: String,
    pub verbose: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            format: OutputFormat::Json,
            include_certificates: true,
            include_proofs: true,
            include_visualizations: false,
            output_directory: "output".to_string(),
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Text,
    Csv,
    Html,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "JSON"),
            OutputFormat::Text => write!(f, "Text"),
            OutputFormat::Csv => write!(f, "CSV"),
            OutputFormat::Html => write!(f, "HTML"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOptions {
    pub skip_type_check: bool,
    pub skip_pareto: bool,
    pub skip_certificates: bool,
    pub skip_roadmap: bool,
    pub enable_caching: bool,
    pub parallel_encoding: bool,
    pub log_level: String,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            skip_type_check: false,
            skip_pareto: false,
            skip_certificates: false,
            skip_roadmap: false,
            enable_caching: true,
            parallel_encoding: false,
            log_level: "info".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = PipelineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = PipelineConfig::default();
        config.solver.timeout_seconds = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_solver_backend_display() {
        assert_eq!(SolverBackend::MaxSmt.to_string(), "MaxSMT");
    }
}
