//! Shadow execution engine for Penumbra.
//!
//! Provides higher-precision "shadow" execution alongside normal floating-point
//! computation to detect and quantify numerical errors.

use serde::{Deserialize, Serialize};
use penumbra_types::{FpOperation, OpId, SourceSpan};

/// A shadow value that pairs a computed f64 with a higher-precision reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowValue {
    pub computed: f64,
    pub shadow: f64,
}

impl ShadowValue {
    pub fn new(computed: f64, shadow: f64) -> Self {
        Self { computed, shadow }
    }

    pub fn error(&self) -> f64 {
        (self.computed - self.shadow).abs()
    }

    pub fn relative_error(&self) -> f64 {
        if self.shadow == 0.0 {
            if self.computed == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            ((self.computed - self.shadow) / self.shadow).abs()
        }
    }
}

/// Result of shadow-executing a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowResult {
    pub op_id: OpId,
    pub operation: FpOperation,
    pub inputs: Vec<ShadowValue>,
    pub output: ShadowValue,
    pub source_location: Option<SourceSpan>,
}

/// Configuration for the shadow execution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    pub track_provenance: bool,
    pub error_threshold: f64,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            track_provenance: true,
            error_threshold: 1e-10,
        }
    }
}
