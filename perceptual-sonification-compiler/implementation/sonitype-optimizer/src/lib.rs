//! # SoniType Optimizer
//!
//! Information-theoretic optimizer with constraint propagation and branch-and-bound
//! search for the SoniType perceptual sonification compiler.
//!
//! Maximizes psychoacoustically-constrained mutual information I_ψ(D; A) over the
//! space of mapping parameters, subject to masking, JND, segregation, and cognitive
//! load constraints.

pub mod mutual_information;
pub mod constraints;
pub mod propagation;
pub mod branch_and_bound;
pub mod pareto;
pub mod search;
pub mod decomposition;
pub mod objective;
pub mod config;

pub use mutual_information::{
    MutualInformationEstimator, PsychoacousticChannel, InformationLossBound,
    DiscriminabilityEstimator,
};
pub use constraints::{
    Constraint, ConstraintSet, FeasibleRegion, ConstraintReport, ConstraintSatisfaction,
};
pub use propagation::{
    ConstraintPropagator, Domain, DomainStore, PropagationResult,
};
pub use branch_and_bound::{
    BranchAndBoundOptimizer, SearchNode, SearchTree, BranchingStrategy,
    BoundingStrategy, SearchStatistics,
};
pub use pareto::{
    ParetoOptimizer, ParetoFront, ObjectiveVector, WeightedSum,
};
pub use search::{
    GreedySearch, SimulatedAnnealing, BeamSearch, RandomRestart, HybridSearch,
};
pub use decomposition::{
    BarkBandDecomposition, TemporalDecomposition, StreamGroupDecomposition,
};
pub use objective::{
    MutualInformationObjective, DiscriminabilityObjective, LatencyObjective,
    CognitiveLoadObjective, SpectralClarityObjective, CompositeObjective,
    ObjectiveFn,
};
pub use config::OptimizerConfig;

use std::collections::HashMap;
use std::fmt;

// ── Local type equivalents for sonitype-core / sonitype-psychoacoustic / sonitype-ir ──

/// Unique identifier for an audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub u32);

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream_{}", self.0)
    }
}

/// Unique identifier for a mapping parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParameterId(pub String);

impl fmt::Display for ParameterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ParameterId {
    fn from(s: &str) -> Self {
        ParameterId(s.to_string())
    }
}

/// Bark band index (0..23 for 24 critical bands).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarkBand(pub u8);

impl BarkBand {
    pub const NUM_BANDS: usize = 24;

    pub fn center_frequency(&self) -> f64 {
        // Approximate Bark-band center frequencies in Hz
        const CENTERS: [f64; 24] = [
            50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0,
            1000.0, 1170.0, 1370.0, 1600.0, 1850.0, 2150.0, 2500.0, 2900.0,
            3400.0, 4000.0, 4800.0, 5800.0, 7000.0, 8500.0, 10500.0, 13500.0,
        ];
        CENTERS[self.0 as usize]
    }

    pub fn bandwidth(&self) -> f64 {
        // Critical bandwidth approximation
        let fc = self.center_frequency();
        25.0 + 75.0 * (1.0 + 1.4 * (fc / 1000.0).powi(2)).powf(0.69)
    }
}

/// Auditory dimension for JND comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditoryDimension {
    Pitch,
    Loudness,
    Timbre,
    SpatialAzimuth,
    SpatialElevation,
    Duration,
    AttackTime,
}

/// Full mapping configuration: parameter assignments for all streams.
#[derive(Debug, Clone)]
pub struct MappingConfig {
    pub stream_params: HashMap<StreamId, StreamMapping>,
    pub global_params: HashMap<String, f64>,
}

impl MappingConfig {
    pub fn new() -> Self {
        MappingConfig {
            stream_params: HashMap::new(),
            global_params: HashMap::new(),
        }
    }

    pub fn stream_count(&self) -> usize {
        self.stream_params.len()
    }

    pub fn get_param(&self, param: &ParameterId) -> Option<f64> {
        self.global_params.get(&param.0).copied()
    }

    pub fn set_param(&mut self, param: &ParameterId, value: f64) {
        self.global_params.insert(param.0.clone(), value);
    }
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-stream mapping parameters.
#[derive(Debug, Clone)]
pub struct StreamMapping {
    pub stream_id: StreamId,
    pub frequency_hz: f64,
    pub amplitude_db: f64,
    pub bark_band: BarkBand,
    pub dimension_values: HashMap<AuditoryDimension, f64>,
}

impl StreamMapping {
    pub fn new(id: StreamId, freq: f64, amp: f64) -> Self {
        let bark = Self::freq_to_bark_band(freq);
        StreamMapping {
            stream_id: id,
            frequency_hz: freq,
            amplitude_db: amp,
            bark_band: bark,
            dimension_values: HashMap::new(),
        }
    }

    fn freq_to_bark_band(freq: f64) -> BarkBand {
        let bark = 13.0 * (0.00076 * freq).atan() + 3.5 * (freq / 7500.0).powi(2).atan();
        BarkBand((bark.round() as u8).min(23))
    }
}

/// Segregation predicate for stream separation requirements.
#[derive(Debug, Clone)]
pub enum SegregationPredicate {
    MinFrequencySeparation(f64),
    DifferentBarkBands,
    MinOnsetAsynchrony(f64),
    HarmonicSeparation,
}

/// Result type for the optimizer.
pub type OptimizerResult<T> = Result<T, OptimizerError>;

/// Errors from the optimizer.
#[derive(Debug, Clone)]
pub enum OptimizerError {
    Infeasible(String),
    Timeout(String),
    NumericalError(String),
    InvalidConfig(String),
    ConvergenceFailure(String),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::Infeasible(msg) => write!(f, "Infeasible: {}", msg),
            OptimizerError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            OptimizerError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            OptimizerError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            OptimizerError::ConvergenceFailure(msg) => write!(f, "Convergence failure: {}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

/// Solution from the optimizer.
#[derive(Debug, Clone)]
pub struct OptimizationSolution {
    pub config: MappingConfig,
    pub objective_value: f64,
    pub objective_values: HashMap<String, f64>,
    pub constraint_satisfaction: f64,
    pub solve_time_ms: f64,
    pub nodes_explored: usize,
}
