//! # fpdiag-diagnosis
//!
//! Taxonomic diagnosis engine for the Penumbra floating-point error
//! analysis pipeline.
//!
//! Implements the five-classifier taxonomy (T3) that classifies every
//! high-error EAG node into a root-cause category:
//! 1. Catastrophic cancellation
//! 2. Absorption
//! 3. Smearing
//! 4. Amplified rounding
//! 5. Ill-conditioned subproblem
//!
//! Each classifier operates on EAG subgraphs and produces structured
//! diagnoses with confidence scores and repair recommendations.

use fpdiag_types::{
    config::DiagnosisConfig,
    diagnosis::{Diagnosis, DiagnosisCategory, DiagnosisReport, DiagnosisSeverity},
    eag::{EagNode, EagNodeId, ErrorAmplificationGraph},
    expression::FpOp,
    source::SourceSpan,
};
use thiserror::Error;

/// Errors from the diagnosis engine.
#[derive(Debug, Error)]
pub enum DiagnosisError {
    #[error("EAG is empty")]
    EmptyEag,
    #[error("node {0} not found in EAG")]
    NodeNotFound(EagNodeId),
    #[error("classifier failed for node {node}: {reason}")]
    ClassifierFailed { node: EagNodeId, reason: String },
}

/// The main diagnosis engine.
///
/// Runs all five classifiers over the EAG and produces a
/// [`DiagnosisReport`].
pub struct DiagnosisEngine {
    config: DiagnosisConfig,
}

impl DiagnosisEngine {
    /// Create a new engine with the given config.
    pub fn new(config: DiagnosisConfig) -> Self {
        Self { config }
    }

    /// Create an engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DiagnosisConfig::default())
    }

    /// Run the full diagnosis pipeline on an EAG.
    pub fn diagnose(
        &self,
        eag: &ErrorAmplificationGraph,
    ) -> Result<DiagnosisReport, DiagnosisError> {
        if eag.node_count() == 0 {
            return Err(DiagnosisError::EmptyEag);
        }

        let mut report = DiagnosisReport::new();
        report.total_nodes = eag.node_count();

        let start = std::time::Instant::now();

        // Analyze each high-error node
        let high_error_nodes = eag.high_error_nodes(self.config.error_threshold_ulps);

        for node in &high_error_nodes {
            if let Some(diag) = self.classify_node(eag, node)? {
                if diag.confidence >= self.config.min_confidence {
                    report.add(diag);
                }
            }
        }

        report.diagnosis_time_ms = start.elapsed().as_millis() as u64;
        report.sort();
        Ok(report)
    }

    /// Classify a single node using all five classifiers.
    fn classify_node(
        &self,
        eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Result<Option<Diagnosis>, DiagnosisError> {
        // Try classifiers in order of specificity
        let classifiers: Vec<(
            &str,
            fn(&Self, &ErrorAmplificationGraph, &EagNode) -> Option<Diagnosis>,
        )> = vec![
            ("cancellation", Self::classify_cancellation),
            ("absorption", Self::classify_absorption),
            ("smearing", Self::classify_smearing),
            ("amplified_rounding", Self::classify_amplified_rounding),
            ("ill_conditioned", Self::classify_ill_conditioned),
        ];

        let mut best: Option<Diagnosis> = None;

        for (_name, classifier) in &classifiers {
            if let Some(diag) = classifier(self, eag, node) {
                match &best {
                    None => best = Some(diag),
                    Some(current) => {
                        if diag.confidence > current.confidence {
                            best = Some(diag);
                        }
                    }
                }
                if !self.config.exhaustive {
                    break;
                }
            }
        }

        Ok(best)
    }

    /// Classifier 1: Catastrophic cancellation.
    ///
    /// Detects subtraction of nearly equal values where relative error
    /// in the result is much larger than in the operands.
    ///
    /// Heuristic: node is a Sub operation with condition number > 10²
    /// (operands are within ~1% of each other).
    fn classify_cancellation(
        &self,
        eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Option<Diagnosis> {
        if node.op != FpOp::Sub && node.op != FpOp::Add {
            return None;
        }

        let condition = node.condition_number?;
        if condition < 100.0 {
            return None;
        }

        let severity = if condition > 1e10 {
            DiagnosisSeverity::Critical
        } else if condition > 1e6 {
            DiagnosisSeverity::Error
        } else if condition > 1e3 {
            DiagnosisSeverity::Warning
        } else {
            DiagnosisSeverity::Info
        };

        let confidence = (condition.log10() / 16.0).clamp(0.5, 0.99);

        let mut diag = Diagnosis::new(
            node.id,
            DiagnosisCategory::CatastrophicCancellation,
            severity,
            confidence,
        );
        diag.explanation = format!(
            "Subtraction of nearly equal values (condition number: {:.2e}). \
             The result loses approximately {:.0} bits of precision.",
            condition,
            condition.log2().min(53.0)
        );
        diag.source = node.source.clone();

        // Compute error contribution via incoming edges
        let incoming = eag.incoming(node.id);
        let total_incoming_error: f64 = incoming.iter().map(|e| e.weight.0).sum();
        if total_incoming_error > 0.0 {
            diag.error_contribution = node.local_error / total_incoming_error.max(node.local_error);
        }

        Some(diag)
    }

    /// Classifier 2: Absorption.
    ///
    /// Detects addition where a small addend is absorbed by a much
    /// larger accumulator.
    fn classify_absorption(
        &self,
        _eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Option<Diagnosis> {
        if node.op != FpOp::Add && node.op != FpOp::Sum {
            return None;
        }

        // For absorption, the condition number should be moderate
        // but the relative error should be high for what should be
        // a "safe" addition
        if node.relative_error < 1e-14 {
            return None;
        }

        // Absorption is indicated when computed == one operand exactly
        // (the small addend was completely lost)
        if node.computed_value == node.shadow_value {
            return None;
        }

        let bits_lost = if node.shadow_value.abs() > 0.0 {
            -((node.computed_value - node.shadow_value).abs() / node.shadow_value.abs()).log2()
        } else {
            0.0
        };

        if bits_lost < 1.0 {
            return None;
        }

        let severity = if bits_lost > 26.0 {
            DiagnosisSeverity::Critical
        } else if bits_lost > 10.0 {
            DiagnosisSeverity::Error
        } else if bits_lost > 4.0 {
            DiagnosisSeverity::Warning
        } else {
            DiagnosisSeverity::Info
        };

        let confidence = (bits_lost / 53.0).clamp(0.3, 0.95);

        let mut diag = Diagnosis::new(node.id, DiagnosisCategory::Absorption, severity, confidence);
        diag.explanation = format!(
            "Small addend absorbed by large accumulator. \
             Approximately {:.1} bits of the addend were lost.",
            bits_lost
        );
        diag.source = node.source.clone();

        Some(diag)
    }

    /// Classifier 3: Smearing.
    ///
    /// Detects alternating-sign additions with gradual error growth.
    fn classify_smearing(
        &self,
        eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Option<Diagnosis> {
        if node.op != FpOp::Sum && node.op != FpOp::Add {
            return None;
        }

        // Look at the chain of incoming operations for sign alternation
        let incoming = eag.incoming(node.id);
        if incoming.len() < 2 {
            return None;
        }

        // Check if multiple incoming edges have similar weights (gradual growth)
        let weights: Vec<f64> = incoming.iter().map(|e| e.weight.0).collect();
        let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
        let variance: f64 = weights
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>()
            / weights.len() as f64;

        // Smearing: relatively uniform error contribution from multiple sources
        if variance / mean_weight.powi(2) > 0.5 {
            return None; // Too much variation; not smearing
        }

        if node.relative_error < 1e-13 {
            return None;
        }

        let severity = if node.relative_error > 1e-6 {
            DiagnosisSeverity::Error
        } else {
            DiagnosisSeverity::Warning
        };

        let confidence = 0.6;

        let mut diag = Diagnosis::new(node.id, DiagnosisCategory::Smearing, severity, confidence);
        diag.explanation = format!(
            "Gradual error accumulation from {} incoming error sources \
             with mean amplification {:.2}×.",
            incoming.len(),
            mean_weight
        );
        diag.source = node.source.clone();

        Some(diag)
    }

    /// Classifier 4: Amplified rounding.
    ///
    /// High condition number amplifies ordinary per-operation rounding.
    fn classify_amplified_rounding(
        &self,
        eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Option<Diagnosis> {
        // Skip subtraction (handled by cancellation classifier)
        if node.op == FpOp::Sub {
            return None;
        }

        let condition = node.condition_number?;
        if condition < 10.0 {
            return None;
        }

        // Check that error is proportional to condition × machine epsilon
        let expected_error = condition * f64::EPSILON * node.shadow_value.abs();
        if node.local_error < expected_error * 0.1 {
            return None; // Error is smaller than expected; not amplified rounding
        }

        let severity = if condition > 1e8 {
            DiagnosisSeverity::Error
        } else if condition > 1e4 {
            DiagnosisSeverity::Warning
        } else {
            DiagnosisSeverity::Info
        };

        let confidence = (condition.log10() / 16.0).clamp(0.4, 0.9);

        let mut diag = Diagnosis::new(
            node.id,
            DiagnosisCategory::AmplifiedRounding,
            severity,
            confidence,
        );
        diag.explanation = format!(
            "Operation {} has condition number {:.2e}, amplifying \
             rounding error by the same factor.",
            node.op, condition
        );
        diag.source = node.source.clone();

        Some(diag)
    }

    /// Classifier 5: Ill-conditioned subproblem.
    ///
    /// Detects black-box library calls with high error amplification.
    fn classify_ill_conditioned(
        &self,
        _eag: &ErrorAmplificationGraph,
        node: &EagNode,
    ) -> Option<Diagnosis> {
        if !node.is_black_box {
            return None;
        }

        let amplification = node.condition_number?;
        if amplification < 100.0 {
            return None;
        }

        let severity = if amplification > 1e10 {
            DiagnosisSeverity::Critical
        } else if amplification > 1e6 {
            DiagnosisSeverity::Error
        } else {
            DiagnosisSeverity::Warning
        };

        let confidence = (amplification.log10() / 16.0).clamp(0.5, 0.95);

        let label = node.label.as_deref().unwrap_or("unknown");
        let mut diag = Diagnosis::new(
            node.id,
            DiagnosisCategory::IllConditionedSubproblem,
            severity,
            confidence,
        );
        diag.explanation = format!(
            "Library call '{}' amplifies input error by {:.2e}×. \
             The subproblem is numerically ill-conditioned at these inputs.",
            label, amplification
        );
        diag.source = node.source.clone();

        Some(diag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph};

    fn make_cancellation_eag() -> ErrorAmplificationGraph {
        let mut eag = ErrorAmplificationGraph::new();
        let mut node = EagNode::new(EagNodeId(0), FpOp::Sub, 1e-10, 1e-15);
        node.condition_number = Some(1e10);
        eag.add_node(node);
        eag
    }

    #[test]
    fn diagnose_cancellation() {
        let eag = make_cancellation_eag();
        let engine = DiagnosisEngine::with_defaults();
        let report = engine.diagnose(&eag).unwrap();
        assert!(!report.diagnoses.is_empty());
        assert_eq!(
            report.diagnoses[0].category,
            DiagnosisCategory::CatastrophicCancellation
        );
    }

    #[test]
    fn empty_eag_error() {
        let eag = ErrorAmplificationGraph::new();
        let engine = DiagnosisEngine::with_defaults();
        assert!(engine.diagnose(&eag).is_err());
    }
}
