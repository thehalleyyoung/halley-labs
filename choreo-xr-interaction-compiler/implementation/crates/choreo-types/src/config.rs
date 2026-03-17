//! Configuration types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    pub epsilon: f64,
    pub max_gjk_iterations: usize,
    pub max_epa_iterations: usize,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            max_gjk_iterations: 64,
            max_epa_iterations: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerificationConfig {
    pub max_depth: usize,
    pub timeout_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompilationConfig {
    pub optimization_level: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationConfig {
    pub time_step: f64,
    pub max_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    pub format: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChoreoConfig {
    pub spatial: SpatialConfig,
    pub verification: VerificationConfig,
    pub compilation: CompilationConfig,
    pub simulation: SimulationConfig,
    pub output: OutputConfig,
}
