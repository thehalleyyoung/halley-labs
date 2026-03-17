//! Violation classification.
use serde::{Serialize, Deserialize};

/// Types of conservation violations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType { EnergyDrift, MomentumDrift, AngularMomentumDrift, SymplecticDrift, NumericalNoise, SystematicBias, Catastrophic }

/// Classifies detected violations.
#[derive(Debug, Clone, Default)]
pub struct ViolationClassifier;
impl ViolationClassifier {
    /// Classify a violation from its characteristics.
    pub fn classify(&self, drift_rate: f64, magnitude: f64) -> ClassificationResult {
        let vtype = if magnitude > 0.1 { ViolationType::Catastrophic }
                    else if drift_rate.abs() > 1e-6 { ViolationType::SystematicBias }
                    else { ViolationType::NumericalNoise };
        ClassificationResult { violation_type: vtype, confidence: 0.9 }
    }
}

/// Result of violation classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult { pub violation_type: ViolationType, pub confidence: f64 }
