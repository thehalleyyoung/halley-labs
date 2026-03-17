//! Configuration types for all subsystems.
//!
//! Provides typed configuration for spectral analysis, oracle prediction,
//! census collection, optimization, and global settings.

use serde::{Deserialize, Serialize};
use crate::decomposition::DecompositionMethod;

/// Laplacian type for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LaplacianType {
    Combinatorial,
    Normalized,
    SignlessLaplacian,
    RandomWalk,
}

impl Default for LaplacianType {
    fn default() -> Self {
        LaplacianType::Normalized
    }
}

/// Eigenvalue solver method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EigensolverMethod {
    LanczosSymmetric,
    ArnoldiNonSymmetric,
    PowerIteration,
    QR,
}

impl Default for EigensolverMethod {
    fn default() -> Self {
        EigensolverMethod::LanczosSymmetric
    }
}

/// Configuration for spectral analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConfig {
    pub laplacian_type: LaplacianType,
    pub normalization: bool,
    pub num_eigenvalues: usize,
    pub eigensolver: EigensolverMethod,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub use_shift_invert: bool,
    pub sigma: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            laplacian_type: LaplacianType::Normalized,
            normalization: true,
            num_eigenvalues: 20,
            eigensolver: EigensolverMethod::LanczosSymmetric,
            tolerance: 1e-10,
            max_iterations: 1000,
            use_shift_invert: false,
            sigma: 0.0,
        }
    }
}

/// Classifier type for the oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassifierType {
    RandomForest,
    GradientBoosting,
    SVM,
    NeuralNetwork,
    Ensemble,
}

impl Default for ClassifierType {
    fn default() -> Self {
        ClassifierType::RandomForest
    }
}

/// Configuration for the oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    pub classifier: ClassifierType,
    pub confidence_threshold: f64,
    pub use_spectral_features: bool,
    pub use_syntactic_features: bool,
    pub use_graph_features: bool,
    pub feature_selection_k: usize,
    pub calibrate_probabilities: bool,
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            classifier: ClassifierType::RandomForest,
            confidence_threshold: 0.6,
            use_spectral_features: true,
            use_syntactic_features: true,
            use_graph_features: true,
            feature_selection_k: 20,
            calibrate_probabilities: true,
            n_estimators: 100,
            max_depth: Some(10),
            min_samples_leaf: 5,
        }
    }
}

/// Census tier for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CensusTier {
    Quick,
    Standard,
    Deep,
    Exhaustive,
}

impl CensusTier {
    pub fn time_limit(self) -> f64 {
        match self {
            CensusTier::Quick => 60.0,
            CensusTier::Standard => 300.0,
            CensusTier::Deep => 1800.0,
            CensusTier::Exhaustive => 7200.0,
        }
    }
}

impl Default for CensusTier {
    fn default() -> Self {
        CensusTier::Standard
    }
}

/// Configuration for the census (benchmark collection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CensusConfig {
    pub tier: CensusTier,
    pub time_limit: f64,
    pub methods: Vec<DecompositionMethod>,
    pub parallelism: usize,
    pub retry_on_failure: bool,
    pub max_retries: usize,
    pub output_dir: String,
    pub save_intermediate: bool,
}

impl Default for CensusConfig {
    fn default() -> Self {
        Self {
            tier: CensusTier::Standard,
            time_limit: 300.0,
            methods: DecompositionMethod::all().to_vec(),
            parallelism: 4,
            retry_on_failure: true,
            max_retries: 2,
            output_dir: "census_output".to_string(),
            save_intermediate: true,
        }
    }
}

/// Solver selection for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverType {
    Cplex,
    Gurobi,
    Scip,
    HiGHS,
    Cbc,
    Glpk,
}

impl Default for SolverType {
    fn default() -> Self {
        SolverType::Scip
    }
}

/// Configuration for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub solver: SolverType,
    pub time_limit: f64,
    pub gap_tolerance: f64,
    pub thread_count: usize,
    pub presolve: bool,
    pub cuts: bool,
    pub heuristics: bool,
    pub verbose: bool,
    pub node_limit: Option<usize>,
    pub solution_limit: Option<usize>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            solver: SolverType::Scip,
            time_limit: 3600.0,
            gap_tolerance: 1e-4,
            thread_count: 4,
            presolve: true,
            cuts: true,
            heuristics: true,
            verbose: false,
            node_limit: None,
            solution_limit: None,
        }
    }
}

/// Global configuration combining all subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub spectral: SpectralConfig,
    pub oracle: OracleConfig,
    pub census: CensusConfig,
    pub optimization: OptimizationConfig,
    pub log_level: LogLevel,
    pub random_seed: u64,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            spectral: SpectralConfig::default(),
            oracle: OracleConfig::default(),
            census: CensusConfig::default(),
            optimization: OptimizationConfig::default(),
            log_level: LogLevel::Info,
            random_seed: 42,
            schema_version: 1,
        }
    }
}

impl GlobalConfig {
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            crate::error::SpectralError::Io(crate::error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }

    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            crate::error::SpectralError::Io(crate::error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.spectral.num_eigenvalues == 0 {
            return Err(crate::error::SpectralError::Config(
                crate::error::ConfigError::InvalidValue {
                    field: "spectral.num_eigenvalues".into(),
                    value: "0".into(),
                    expected: "> 0".into(),
                },
            ));
        }
        if self.spectral.tolerance <= 0.0 {
            return Err(crate::error::SpectralError::Config(
                crate::error::ConfigError::InvalidValue {
                    field: "spectral.tolerance".into(),
                    value: self.spectral.tolerance.to_string(),
                    expected: "> 0".into(),
                },
            ));
        }
        if self.oracle.confidence_threshold < 0.0 || self.oracle.confidence_threshold > 1.0 {
            return Err(crate::error::SpectralError::Config(
                crate::error::ConfigError::OutOfRange {
                    field: "oracle.confidence_threshold".into(),
                    value: self.oracle.confidence_threshold,
                    min: 0.0,
                    max: 1.0,
                },
            ));
        }
        if self.optimization.time_limit <= 0.0 {
            return Err(crate::error::SpectralError::Config(
                crate::error::ConfigError::InvalidValue {
                    field: "optimization.time_limit".into(),
                    value: self.optimization.time_limit.to_string(),
                    expected: "> 0".into(),
                },
            ));
        }
        Ok(())
    }
}

/// Builder pattern for GlobalConfig.
pub struct GlobalConfigBuilder {
    config: GlobalConfig,
}

impl GlobalConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: GlobalConfig::default(),
        }
    }

    pub fn spectral(mut self, cfg: SpectralConfig) -> Self {
        self.config.spectral = cfg;
        self
    }

    pub fn oracle(mut self, cfg: OracleConfig) -> Self {
        self.config.oracle = cfg;
        self
    }

    pub fn census(mut self, cfg: CensusConfig) -> Self {
        self.config.census = cfg;
        self
    }

    pub fn optimization(mut self, cfg: OptimizationConfig) -> Self {
        self.config.optimization = cfg;
        self
    }

    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.config.log_level = level;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    pub fn laplacian_type(mut self, lt: LaplacianType) -> Self {
        self.config.spectral.laplacian_type = lt;
        self
    }

    pub fn num_eigenvalues(mut self, n: usize) -> Self {
        self.config.spectral.num_eigenvalues = n;
        self
    }

    pub fn confidence_threshold(mut self, t: f64) -> Self {
        self.config.oracle.confidence_threshold = t;
        self
    }

    pub fn solver(mut self, s: SolverType) -> Self {
        self.config.optimization.solver = s;
        self
    }

    pub fn time_limit(mut self, t: f64) -> Self {
        self.config.optimization.time_limit = t;
        self
    }

    pub fn parallelism(mut self, p: usize) -> Self {
        self.config.census.parallelism = p;
        self
    }

    pub fn build(self) -> crate::error::Result<GlobalConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    pub fn build_unchecked(self) -> GlobalConfig {
        self.config
    }
}

impl Default for GlobalConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let _ = SpectralConfig::default();
        let _ = OracleConfig::default();
        let _ = CensusConfig::default();
        let _ = OptimizationConfig::default();
        let _ = GlobalConfig::default();
    }

    #[test]
    fn test_global_config_validate() {
        let cfg = GlobalConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_global_config_json_roundtrip() {
        let cfg = GlobalConfig::default();
        let json = cfg.to_json().unwrap();
        let back = GlobalConfig::from_json(&json).unwrap();
        assert_eq!(back.random_seed, 42);
    }

    #[test]
    fn test_invalid_eigenvalues() {
        let mut cfg = GlobalConfig::default();
        cfg.spectral.num_eigenvalues = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_confidence() {
        let mut cfg = GlobalConfig::default();
        cfg.oracle.confidence_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let cfg = GlobalConfigBuilder::new()
            .seed(123)
            .num_eigenvalues(30)
            .confidence_threshold(0.7)
            .solver(SolverType::Gurobi)
            .build()
            .unwrap();
        assert_eq!(cfg.random_seed, 123);
        assert_eq!(cfg.spectral.num_eigenvalues, 30);
    }

    #[test]
    fn test_builder_invalid() {
        let result = GlobalConfigBuilder::new()
            .num_eigenvalues(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_census_tier_time_limit() {
        assert_eq!(CensusTier::Quick.time_limit(), 60.0);
        assert_eq!(CensusTier::Exhaustive.time_limit(), 7200.0);
    }

    #[test]
    fn test_builder_unchecked() {
        let cfg = GlobalConfigBuilder::new()
            .num_eigenvalues(0)
            .build_unchecked();
        assert_eq!(cfg.spectral.num_eigenvalues, 0);
    }

    #[test]
    fn test_laplacian_type_default() {
        assert_eq!(LaplacianType::default(), LaplacianType::Normalized);
    }

    #[test]
    fn test_solver_type_default() {
        assert_eq!(SolverType::default(), SolverType::Scip);
    }

    #[test]
    fn test_log_level_default() {
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }
}
