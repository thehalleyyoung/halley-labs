//! Fault classification and causal verdict types for localization results.

use serde::{Deserialize, Serialize};

/// Classification of a detected fault's causal role at a pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CausalVerdict {
    /// The stage introduced the fault (violation disappears when stage output is replaced).
    Introduced,
    /// The stage amplified a pre-existing fault from an upstream stage.
    Amplified {
        amplification_factor: f64,
    },
    /// The stage contributed to the fault but is not the primary cause.
    Contributing,
    /// The stage is not causally related to the fault.
    NotCausal,
}

impl CausalVerdict {
    /// Human-readable description of the verdict.
    pub fn description(&self) -> String {
        match self {
            CausalVerdict::Introduced => "Fault was introduced at this stage".to_string(),
            CausalVerdict::Amplified { amplification_factor } => {
                format!(
                    "Fault was amplified {:.1}× by this stage",
                    amplification_factor
                )
            }
            CausalVerdict::Contributing => "Stage contributed to the fault".to_string(),
            CausalVerdict::NotCausal => "Stage is not causally related".to_string(),
        }
    }

    /// Whether this verdict indicates the stage is at least partially responsible.
    pub fn is_responsible(&self) -> bool {
        !matches!(self, CausalVerdict::NotCausal)
    }

    /// Severity score from 0.0 (not causal) to 1.0 (introduced).
    pub fn severity(&self) -> f64 {
        match self {
            CausalVerdict::Introduced => 1.0,
            CausalVerdict::Amplified { amplification_factor } => {
                (0.5 + 0.5 * (1.0 - 1.0 / amplification_factor.max(1.0))).min(0.95)
            }
            CausalVerdict::Contributing => 0.3,
            CausalVerdict::NotCausal => 0.0,
        }
    }
}

/// High-level fault classification for a pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FaultClassification {
    /// The stage consistently mishandles a specific transformation type.
    TransformationSpecific {
        transformation: String,
        error_rate: f64,
    },
    /// The stage introduces cascading errors that amplify downstream.
    CascadingAmplification {
        downstream_stages: Vec<String>,
        amplification_chain: Vec<f64>,
    },
    /// The stage has a consistent bias in one direction.
    SystematicBias {
        direction: String,
        magnitude: f64,
    },
    /// The stage intermittently fails under specific input patterns.
    IntermittentFailure {
        trigger_pattern: String,
        failure_rate: f64,
    },
    /// The stage interacts badly with another specific stage.
    StageInteraction {
        interacting_stage: String,
        interaction_type: String,
    },
    /// Unclassified fault.
    Unclassified {
        description: String,
    },
}

impl FaultClassification {
    /// Human-readable summary of the fault classification.
    pub fn summary(&self) -> String {
        match self {
            FaultClassification::TransformationSpecific {
                transformation,
                error_rate,
            } => format!(
                "Specific to '{}' transformation (error rate: {:.1}%)",
                transformation,
                error_rate * 100.0
            ),
            FaultClassification::CascadingAmplification {
                downstream_stages,
                amplification_chain,
            } => format!(
                "Cascading through {} downstream stages (max amplification: {:.1}×)",
                downstream_stages.len(),
                amplification_chain
                    .iter()
                    .cloned()
                    .fold(0.0f64, f64::max)
            ),
            FaultClassification::SystematicBias {
                direction,
                magnitude,
            } => format!("Systematic {} bias (magnitude: {:.3})", direction, magnitude),
            FaultClassification::IntermittentFailure {
                trigger_pattern,
                failure_rate,
            } => format!(
                "Intermittent failure on '{}' (rate: {:.1}%)",
                trigger_pattern,
                failure_rate * 100.0
            ),
            FaultClassification::StageInteraction {
                interacting_stage,
                interaction_type,
            } => format!(
                "{} interaction with '{}'",
                interaction_type, interacting_stage
            ),
            FaultClassification::Unclassified { description } => description.clone(),
        }
    }

    /// Severity score from 0.0 to 1.0.
    pub fn severity(&self) -> f64 {
        match self {
            FaultClassification::TransformationSpecific { error_rate, .. } => *error_rate,
            FaultClassification::CascadingAmplification {
                amplification_chain,
                ..
            } => {
                let max_amp = amplification_chain
                    .iter()
                    .cloned()
                    .fold(0.0f64, f64::max);
                (max_amp / (max_amp + 1.0)).min(1.0)
            }
            FaultClassification::SystematicBias { magnitude, .. } => magnitude.min(1.0).abs(),
            FaultClassification::IntermittentFailure { failure_rate, .. } => *failure_rate,
            FaultClassification::StageInteraction { .. } => 0.5,
            FaultClassification::Unclassified { .. } => 0.3,
        }
    }
}

/// Result of an interventional analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionResult {
    pub stage_name: String,
    pub original_violation: f64,
    pub post_intervention_violation: f64,
    pub attenuation: f64,
    pub verdict: CausalVerdict,
    pub confidence: f64,
}

impl InterventionResult {
    /// Whether the intervention resolved the violation.
    pub fn resolved(&self) -> bool {
        self.post_intervention_violation < f64::EPSILON
    }

    /// Whether the intervention significantly reduced the violation.
    pub fn significantly_attenuated(&self) -> bool {
        self.attenuation > 0.5
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_verdict_severity() {
        assert!((CausalVerdict::Introduced.severity() - 1.0).abs() < f64::EPSILON);
        assert!(CausalVerdict::NotCausal.severity() < f64::EPSILON);
        assert!(CausalVerdict::Contributing.severity() > 0.0);
        assert!(
            CausalVerdict::Amplified {
                amplification_factor: 4.0
            }
            .severity()
                > 0.5
        );
    }

    #[test]
    fn test_causal_verdict_responsibility() {
        assert!(CausalVerdict::Introduced.is_responsible());
        assert!(CausalVerdict::Contributing.is_responsible());
        assert!(!CausalVerdict::NotCausal.is_responsible());
    }

    #[test]
    fn test_fault_classification_severity() {
        let ts = FaultClassification::TransformationSpecific {
            transformation: "passivization".to_string(),
            error_rate: 0.8,
        };
        assert!((ts.severity() - 0.8).abs() < 0.01);

        let ca = FaultClassification::CascadingAmplification {
            downstream_stages: vec!["parser".into(), "ner".into()],
            amplification_chain: vec![2.0, 4.0],
        };
        assert!(ca.severity() > 0.5);

        let sb = FaultClassification::SystematicBias {
            direction: "left".to_string(),
            magnitude: 0.3,
        };
        assert!((sb.severity() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_intervention_result() {
        let ir = InterventionResult {
            stage_name: "tagger".to_string(),
            original_violation: 0.8,
            post_intervention_violation: 0.0,
            attenuation: 1.0,
            verdict: CausalVerdict::Introduced,
            confidence: 0.95,
        };
        assert!(ir.resolved());
        assert!(ir.significantly_attenuated());
    }

    #[test]
    fn test_fault_classification_summaries() {
        let classifications = vec![
            FaultClassification::TransformationSpecific {
                transformation: "passivization".to_string(),
                error_rate: 0.75,
            },
            FaultClassification::CascadingAmplification {
                downstream_stages: vec!["parser".into()],
                amplification_chain: vec![3.0],
            },
            FaultClassification::SystematicBias {
                direction: "positive".to_string(),
                magnitude: 0.4,
            },
            FaultClassification::IntermittentFailure {
                trigger_pattern: "passive gerunds".to_string(),
                failure_rate: 0.2,
            },
            FaultClassification::StageInteraction {
                interacting_stage: "parser".to_string(),
                interaction_type: "cascading".to_string(),
            },
            FaultClassification::Unclassified {
                description: "unknown issue".to_string(),
            },
        ];

        for fc in &classifications {
            let summary = fc.summary();
            assert!(!summary.is_empty());
            let sev = fc.severity();
            assert!(sev >= 0.0 && sev <= 1.0);
        }
    }
}
