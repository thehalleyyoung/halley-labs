//! Distance metrics and stage-level distance types.

use crate::types::StageId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which distance metric to use when comparing intermediate representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Exact,
    Jaccard,
    EditDistance,
    TreeEditDistance,
    Cosine,
    Custom(String),
}

/// A single distance measurement with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceValue {
    pub value: f64,
    pub metric: DistanceMetric,
    pub normalized: bool,
}

impl DistanceValue {
    pub fn new(value: f64, metric: DistanceMetric) -> Self {
        Self {
            value,
            metric,
            normalized: false,
        }
    }

    pub fn normalized(value: f64, metric: DistanceMetric) -> Self {
        Self {
            value,
            metric,
            normalized: true,
        }
    }
}

/// Distance for a single named sub-component of a stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentDistance {
    pub component_name: String,
    pub distance: f64,
}

/// Aggregated distance for one pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDistance {
    pub stage_id: StageId,
    pub stage_name: String,
    pub overall_distance: f64,
    pub component_distances: Vec<ComponentDistance>,
}

/// Configuration for distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceConfig {
    pub metric: DistanceMetric,
    pub normalize: bool,
    pub weights: HashMap<String, f64>,
}

impl Default for DistanceConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Exact,
            normalize: true,
            weights: HashMap::new(),
        }
    }
}
