//! Optimizer configuration: solver parameters, strategy selection, decomposition
//! settings, multi-objective weights, logging, and progress callbacks.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Primary configuration for the optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub solver: SolverParams,
    pub strategy: StrategyConfig,
    pub decomposition: DecompositionConfig,
    pub multi_objective: MultiObjectiveConfig,
    pub logging: LoggingConfig,
    pub progress_callback: Option<ProgressCallbackConfig>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            solver: SolverParams::default(),
            strategy: StrategyConfig::default(),
            decomposition: DecompositionConfig::default(),
            multi_objective: MultiObjectiveConfig::default(),
            logging: LoggingConfig::default(),
            progress_callback: None,
        }
    }
}

impl OptimizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.solver.time_limit = Some(limit);
        self
    }

    pub fn with_node_limit(mut self, limit: usize) -> Self {
        self.solver.node_limit = Some(limit);
        self
    }

    pub fn with_gap_tolerance(mut self, gap: f64) -> Self {
        self.solver.gap_tolerance = gap;
        self
    }

    pub fn with_strategy(mut self, strategy: SolverStrategy) -> Self {
        self.strategy.solver_strategy = strategy;
        self
    }

    pub fn with_decomposition(mut self, enabled: bool) -> Self {
        self.decomposition.enabled = enabled;
        self
    }

    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.multi_objective.weights = weights;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.logging.verbose = verbose;
        self
    }

    /// Validate the configuration, returning errors for invalid settings.
    pub fn validate(&self) -> Result<(), Vec<ConfigError>> {
        let mut errors = Vec::new();

        if self.solver.gap_tolerance < 0.0 || self.solver.gap_tolerance > 1.0 {
            errors.push(ConfigError::InvalidValue {
                field: "gap_tolerance".into(),
                message: "Must be in [0, 1]".into(),
            });
        }

        if self.solver.feasibility_tolerance < 0.0 {
            errors.push(ConfigError::InvalidValue {
                field: "feasibility_tolerance".into(),
                message: "Must be non-negative".into(),
            });
        }

        if let Some(limit) = self.solver.node_limit {
            if limit == 0 {
                errors.push(ConfigError::InvalidValue {
                    field: "node_limit".into(),
                    message: "Must be positive".into(),
                });
            }
        }

        if self.strategy.beam_width == 0 {
            errors.push(ConfigError::InvalidValue {
                field: "beam_width".into(),
                message: "Must be positive".into(),
            });
        }

        if self.strategy.sa_initial_temperature <= 0.0 {
            errors.push(ConfigError::InvalidValue {
                field: "sa_initial_temperature".into(),
                message: "Must be positive".into(),
            });
        }

        if self.strategy.sa_cooling_rate <= 0.0 || self.strategy.sa_cooling_rate >= 1.0 {
            errors.push(ConfigError::InvalidValue {
                field: "sa_cooling_rate".into(),
                message: "Must be in (0, 1)".into(),
            });
        }

        let weight_sum: f64 = self.multi_objective.weights.values().sum();
        if !self.multi_objective.weights.is_empty() && (weight_sum - 1.0).abs() > 0.01 {
            errors.push(ConfigError::InvalidValue {
                field: "multi_objective.weights".into(),
                message: format!("Weights should sum to ~1.0, got {:.4}", weight_sum),
            });
        }

        if self.decomposition.min_band_streams == 0 {
            errors.push(ConfigError::InvalidValue {
                field: "min_band_streams".into(),
                message: "Must be positive".into(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Solver-level parameters controlling search termination and tolerance.
#[derive(Debug, Clone)]
pub struct SolverParams {
    pub time_limit: Option<Duration>,
    pub node_limit: Option<usize>,
    pub gap_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
    pub max_stall_iterations: usize,
    pub random_seed: Option<u64>,
    pub num_threads: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        SolverParams {
            time_limit: Some(Duration::from_secs(60)),
            node_limit: Some(100_000),
            gap_tolerance: 0.01,
            feasibility_tolerance: 1e-6,
            absolute_tolerance: 1e-9,
            max_iterations: 10_000,
            max_stall_iterations: 500,
            random_seed: None,
            num_threads: 1,
        }
    }
}

/// Strategy-level configuration selecting exact vs approximate methods.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub solver_strategy: SolverStrategy,
    pub branching: BranchingConfig,
    pub beam_width: usize,
    pub sa_initial_temperature: f64,
    pub sa_cooling_rate: f64,
    pub sa_min_temperature: f64,
    pub sa_steps_per_temperature: usize,
    pub num_random_restarts: usize,
    pub greedy_first: bool,
    pub use_hybrid: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        StrategyConfig {
            solver_strategy: SolverStrategy::BranchAndBound,
            branching: BranchingConfig::default(),
            beam_width: 10,
            sa_initial_temperature: 100.0,
            sa_cooling_rate: 0.995,
            sa_min_temperature: 0.01,
            sa_steps_per_temperature: 50,
            num_random_restarts: 5,
            greedy_first: true,
            use_hybrid: false,
        }
    }
}

/// High-level solver strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStrategy {
    /// Exact branch-and-bound search.
    BranchAndBound,
    /// Greedy heuristic (fast, approximate).
    Greedy,
    /// Simulated annealing (stochastic local search).
    SimulatedAnnealing,
    /// Beam search (top-k partial solutions).
    BeamSearch,
    /// Random restart local search.
    RandomRestart,
    /// Hybrid: branch-and-bound with local refinement.
    Hybrid,
    /// Multi-objective Pareto optimization.
    Pareto,
}

impl fmt::Display for SolverStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverStrategy::BranchAndBound => write!(f, "BranchAndBound"),
            SolverStrategy::Greedy => write!(f, "Greedy"),
            SolverStrategy::SimulatedAnnealing => write!(f, "SimulatedAnnealing"),
            SolverStrategy::BeamSearch => write!(f, "BeamSearch"),
            SolverStrategy::RandomRestart => write!(f, "RandomRestart"),
            SolverStrategy::Hybrid => write!(f, "Hybrid"),
            SolverStrategy::Pareto => write!(f, "Pareto"),
        }
    }
}

/// Branching configuration for branch-and-bound.
#[derive(Debug, Clone)]
pub struct BranchingConfig {
    pub strategy: BranchingStrategyType,
    pub min_domain_width: f64,
    pub max_depth: usize,
}

impl Default for BranchingConfig {
    fn default() -> Self {
        BranchingConfig {
            strategy: BranchingStrategyType::LargestDomain,
            min_domain_width: 1e-4,
            max_depth: 100,
        }
    }
}

/// Branching strategy type for configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategyType {
    LargestDomain,
    MostConstrained,
    MaxImpact,
}

/// Decomposition configuration.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    pub enabled: bool,
    pub bark_band_decomposition: bool,
    pub temporal_decomposition: bool,
    pub stream_group_decomposition: bool,
    pub min_band_streams: usize,
    pub temporal_window_ms: f64,
    pub merge_strategy: MergeStrategy,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        DecompositionConfig {
            enabled: true,
            bark_band_decomposition: true,
            temporal_decomposition: false,
            stream_group_decomposition: false,
            min_band_streams: 1,
            temporal_window_ms: 100.0,
            merge_strategy: MergeStrategy::BestFit,
        }
    }
}

/// Strategy for merging decomposed subproblem solutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    BestFit,
    FirstFit,
    WeightedAverage,
}

/// Multi-objective optimization configuration.
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    pub weights: HashMap<String, f64>,
    pub population_size: usize,
    pub num_generations: usize,
    pub archive_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub reference_point: Option<Vec<f64>>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("mutual_information".into(), 0.5);
        weights.insert("latency".into(), 0.2);
        weights.insert("cognitive_load".into(), 0.15);
        weights.insert("spectral_clarity".into(), 0.15);

        MultiObjectiveConfig {
            weights,
            population_size: 100,
            num_generations: 200,
            archive_size: 50,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            reference_point: None,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub verbose: bool,
    pub log_level: LogLevel,
    pub log_nodes: bool,
    pub log_bounds: bool,
    pub log_incumbents: bool,
    pub report_interval_ms: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            verbose: false,
            log_level: LogLevel::Info,
            log_nodes: false,
            log_bounds: true,
            log_incumbents: true,
            report_interval_ms: 1000,
        }
    }
}

/// Log level for optimizer output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Progress callback configuration.
#[derive(Debug, Clone)]
pub struct ProgressCallbackConfig {
    pub report_interval_nodes: usize,
    pub report_interval_ms: u64,
    pub include_bounds: bool,
    pub include_incumbent: bool,
}

impl Default for ProgressCallbackConfig {
    fn default() -> Self {
        ProgressCallbackConfig {
            report_interval_nodes: 100,
            report_interval_ms: 500,
            include_bounds: true,
            include_incumbent: true,
        }
    }
}

/// Progress report sent via callback.
#[derive(Debug, Clone)]
pub struct ProgressReport {
    pub nodes_explored: usize,
    pub nodes_pruned: usize,
    pub best_bound: f64,
    pub best_objective: f64,
    pub gap: f64,
    pub elapsed_ms: f64,
    pub status: SearchStatus,
}

/// Current search status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStatus {
    Running,
    Optimal,
    Feasible,
    Infeasible,
    TimeLimitReached,
    NodeLimitReached,
    Stalled,
}

impl fmt::Display for SearchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchStatus::Running => write!(f, "Running"),
            SearchStatus::Optimal => write!(f, "Optimal"),
            SearchStatus::Feasible => write!(f, "Feasible"),
            SearchStatus::Infeasible => write!(f, "Infeasible"),
            SearchStatus::TimeLimitReached => write!(f, "TimeLimitReached"),
            SearchStatus::NodeLimitReached => write!(f, "NodeLimitReached"),
            SearchStatus::Stalled => write!(f, "Stalled"),
        }
    }
}

/// Configuration validation error.
#[derive(Debug, Clone)]
pub enum ConfigError {
    InvalidValue { field: String, message: String },
    MissingField(String),
    Incompatible { field1: String, field2: String, message: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidValue { field, message } => {
                write!(f, "Invalid value for '{}': {}", field, message)
            }
            ConfigError::MissingField(field) => write!(f, "Missing field: {}", field),
            ConfigError::Incompatible { field1, field2, message } => {
                write!(f, "Incompatible '{}' and '{}': {}", field1, field2, message)
            }
        }
    }
}

/// Builder for OptimizerConfig with fluent API.
pub struct OptimizerConfigBuilder {
    config: OptimizerConfig,
}

impl OptimizerConfigBuilder {
    pub fn new() -> Self {
        OptimizerConfigBuilder {
            config: OptimizerConfig::default(),
        }
    }

    pub fn time_limit(mut self, limit: Duration) -> Self {
        self.config.solver.time_limit = Some(limit);
        self
    }

    pub fn node_limit(mut self, limit: usize) -> Self {
        self.config.solver.node_limit = Some(limit);
        self
    }

    pub fn gap_tolerance(mut self, gap: f64) -> Self {
        self.config.solver.gap_tolerance = gap;
        self
    }

    pub fn strategy(mut self, strategy: SolverStrategy) -> Self {
        self.config.strategy.solver_strategy = strategy;
        self
    }

    pub fn beam_width(mut self, width: usize) -> Self {
        self.config.strategy.beam_width = width;
        self
    }

    pub fn sa_temperature(mut self, temp: f64) -> Self {
        self.config.strategy.sa_initial_temperature = temp;
        self
    }

    pub fn sa_cooling(mut self, rate: f64) -> Self {
        self.config.strategy.sa_cooling_rate = rate;
        self
    }

    pub fn random_restarts(mut self, n: usize) -> Self {
        self.config.strategy.num_random_restarts = n;
        self
    }

    pub fn decomposition(mut self, enabled: bool) -> Self {
        self.config.decomposition.enabled = enabled;
        self
    }

    pub fn bark_bands(mut self, enabled: bool) -> Self {
        self.config.decomposition.bark_band_decomposition = enabled;
        self
    }

    pub fn weight(mut self, name: &str, w: f64) -> Self {
        self.config.multi_objective.weights.insert(name.into(), w);
        self
    }

    pub fn verbose(mut self, v: bool) -> Self {
        self.config.logging.verbose = v;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.solver.random_seed = Some(seed);
        self
    }

    pub fn build(self) -> Result<OptimizerConfig, Vec<ConfigError>> {
        self.config.validate()?;
        Ok(self.config)
    }

    pub fn build_unchecked(self) -> OptimizerConfig {
        self.config
    }
}

impl Default for OptimizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = OptimizerConfig::default();
        assert!(cfg.solver.time_limit.is_some());
        assert_eq!(cfg.solver.gap_tolerance, 0.01);
    }

    #[test]
    fn test_config_validation_ok() {
        let cfg = OptimizerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_gap() {
        let mut cfg = OptimizerConfig::default();
        cfg.solver.gap_tolerance = -0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_beam_width() {
        let mut cfg = OptimizerConfig::default();
        cfg.strategy.beam_width = 0;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ConfigError::InvalidValue { field, .. } if field == "beam_width")));
    }

    #[test]
    fn test_config_validation_bad_sa_temp() {
        let mut cfg = OptimizerConfig::default();
        cfg.strategy.sa_initial_temperature = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_cooling_rate() {
        let mut cfg = OptimizerConfig::default();
        cfg.strategy.sa_cooling_rate = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_builder_default() {
        let cfg = OptimizerConfigBuilder::new().build_unchecked();
        assert_eq!(cfg.strategy.solver_strategy, SolverStrategy::BranchAndBound);
    }

    #[test]
    fn test_builder_with_params() {
        let cfg = OptimizerConfigBuilder::new()
            .time_limit(Duration::from_secs(120))
            .node_limit(5000)
            .gap_tolerance(0.05)
            .strategy(SolverStrategy::Greedy)
            .build()
            .unwrap();
        assert_eq!(cfg.solver.time_limit, Some(Duration::from_secs(120)));
        assert_eq!(cfg.solver.node_limit, Some(5000));
        assert_eq!(cfg.strategy.solver_strategy, SolverStrategy::Greedy);
    }

    #[test]
    fn test_builder_validation_fails() {
        let result = OptimizerConfigBuilder::new()
            .gap_tolerance(2.0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_fluent_config() {
        let cfg = OptimizerConfig::new()
            .with_time_limit(Duration::from_secs(30))
            .with_node_limit(1000)
            .with_gap_tolerance(0.02)
            .with_strategy(SolverStrategy::SimulatedAnnealing)
            .with_verbose(true);
        assert_eq!(cfg.solver.node_limit, Some(1000));
        assert!(cfg.logging.verbose);
    }

    #[test]
    fn test_solver_strategy_display() {
        assert_eq!(format!("{}", SolverStrategy::BranchAndBound), "BranchAndBound");
        assert_eq!(format!("{}", SolverStrategy::Hybrid), "Hybrid");
    }

    #[test]
    fn test_search_status_display() {
        assert_eq!(format!("{}", SearchStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SearchStatus::Infeasible), "Infeasible");
    }

    #[test]
    fn test_progress_report() {
        let report = ProgressReport {
            nodes_explored: 100,
            nodes_pruned: 30,
            best_bound: 5.0,
            best_objective: 4.5,
            gap: 0.1,
            elapsed_ms: 1234.0,
            status: SearchStatus::Running,
        };
        assert_eq!(report.nodes_explored, 100);
        assert_eq!(report.status, SearchStatus::Running);
    }

    #[test]
    fn test_multi_objective_default_weights() {
        let cfg = MultiObjectiveConfig::default();
        let sum: f64 = cfg.weights.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
