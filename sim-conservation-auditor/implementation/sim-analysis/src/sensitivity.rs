//! Sensitivity analysis methods.
use serde::{Serialize, Deserialize};

/// Parameter sensitivity result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivity { pub parameter: String, pub sensitivity: f64 }

/// Morris method screening result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorrisResult { pub mu_star: Vec<f64>, pub sigma: Vec<f64>, pub parameter_names: Vec<String> }

/// Sobol sensitivity indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SobolIndices { pub first_order: Vec<f64>, pub total_order: Vec<f64> }

/// Sensitivity analysis engine.
#[derive(Debug, Clone, Default)]
pub struct SensitivityAnalyzer;
