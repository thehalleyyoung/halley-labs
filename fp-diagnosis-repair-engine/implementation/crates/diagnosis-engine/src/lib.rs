//! Diagnosis engine for Penumbra floating-point error detection.

use eag_builder::NodeIndex;
use penumbra_types::DiagnosisCategory;
use serde::{Deserialize, Serialize};

/// A diagnosis of a floating-point error at a specific EAG node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub node_index: NodeIndex,
    pub category: DiagnosisCategory,
    pub severity: f64,
    pub description: String,
    pub bits_lost: f64,
}

impl Diagnosis {
    pub fn new(node_index: NodeIndex, category: DiagnosisCategory) -> Self {
        let severity = category.typical_severity();
        let description = category.description().to_string();
        Self {
            node_index,
            category,
            severity,
            description,
            bits_lost: 0.0,
        }
    }
}
