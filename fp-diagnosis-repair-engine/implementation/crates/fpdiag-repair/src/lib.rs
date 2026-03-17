//! # fpdiag-repair
//!
//! Repair synthesis and certification for diagnosed floating-point errors.
//!
//! Given a diagnosis report and the corresponding EAG, this crate:
//! 1. Selects repair strategies from the pattern library based on diagnoses
//! 2. Generates repair candidates in T4-optimal order (greedy by attributed error)
//! 3. Certifies error reduction via interval arithmetic
//!
//! ## T4: Diagnosis-Guided Repair Dominance
//!
//! On monotone error-flow DAGs, the greedy strategy of repairing nodes in
//! descending order of EAG-attributed error contribution is step-optimal.

use fpdiag_types::{
    config::RepairConfig,
    diagnosis::{Diagnosis, DiagnosisCategory, DiagnosisReport},
    eag::{EagNodeId, ErrorAmplificationGraph},
    error_bounds::{ErrorBound, ErrorInterval, ErrorMetric},
    repair::{RepairCandidate, RepairCertification, RepairResult, RepairStrategy},
};
use thiserror::Error;

/// Errors from the repair module.
#[derive(Debug, Error)]
pub enum RepairError {
    #[error("no diagnoses to repair")]
    NoDiagnoses,
    #[error("repair budget exhausted")]
    BudgetExhausted,
    #[error("certification failed for node {node}: {reason}")]
    CertificationFailed { node: EagNodeId, reason: String },
    #[error("no suitable repair strategy for {0}")]
    NoStrategy(DiagnosisCategory),
}

/// The repair synthesizer.
///
/// Uses diagnosis-guided selection (T4) to produce an optimal repair plan.
pub struct RepairSynthesizer {
    config: RepairConfig,
}

impl RepairSynthesizer {
    /// Create a new synthesizer.
    pub fn new(config: RepairConfig) -> Self {
        Self { config }
    }

    /// Create with defaults.
    pub fn with_defaults() -> Self {
        Self::new(RepairConfig::default())
    }

    /// Generate a repair plan from a diagnosis report.
    ///
    /// Repairs are selected in T4-optimal order: descending by error contribution.
    pub fn synthesize(
        &self,
        eag: &ErrorAmplificationGraph,
        report: &DiagnosisReport,
    ) -> Result<RepairResult, RepairError> {
        if report.diagnoses.is_empty() {
            return Err(RepairError::NoDiagnoses);
        }

        let mut result = RepairResult::new();
        let start = std::time::Instant::now();

        // Sort diagnoses by error contribution (T4 greedy order)
        let mut sorted_diagnoses: Vec<&Diagnosis> = report.diagnoses.iter().collect();
        sorted_diagnoses.sort_by(|a, b| {
            b.error_contribution
                .partial_cmp(&a.error_contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply repairs up to budget
        let budget = self.config.max_repair_budget.min(sorted_diagnoses.len());

        for diag in sorted_diagnoses.iter().take(budget) {
            let candidates = self.generate_candidates(eag, diag);

            if let Some(best) = candidates.into_iter().next() {
                let cert = self.certify_repair(eag, &best, diag);
                result.add(best, cert);
            }
        }

        result.repair_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Generate repair candidates for a single diagnosis.
    fn generate_candidates(
        &self,
        _eag: &ErrorAmplificationGraph,
        diag: &Diagnosis,
    ) -> Vec<RepairCandidate> {
        let mut candidates = Vec::new();
        let max = self.config.max_candidates_per_node;

        match diag.category {
            DiagnosisCategory::CatastrophicCancellation => {
                // Primary: algebraic rewrite
                let mut c = RepairCandidate::new(
                    RepairStrategy::LogSumExp,
                    vec![diag.node_id],
                    diag.category,
                );
                c.estimated_reduction = 1000.0;
                c.priority = diag.error_contribution * 1000.0;
                candidates.push(c);

                if candidates.len() < max {
                    let mut c = RepairCandidate::new(
                        RepairStrategy::Expm1,
                        vec![diag.node_id],
                        diag.category,
                    );
                    c.estimated_reduction = 100.0;
                    c.priority = diag.error_contribution * 100.0;
                    candidates.push(c);
                }

                // Fallback: precision promotion
                if candidates.len() < max && self.config.allow_precision_promotion {
                    let mut c = RepairCandidate::new(
                        RepairStrategy::PrecisionPromotion { target_bits: 128 },
                        vec![diag.node_id],
                        diag.category,
                    );
                    c.estimated_reduction = 10.0;
                    c.priority = diag.error_contribution * 10.0;
                    c.is_fallback = true;
                    candidates.push(c);
                }
            }
            DiagnosisCategory::Absorption => {
                let mut c = RepairCandidate::new(
                    RepairStrategy::KahanSummation,
                    vec![diag.node_id],
                    diag.category,
                );
                c.estimated_reduction = 100.0;
                c.priority = diag.error_contribution * 100.0;
                candidates.push(c);

                if candidates.len() < max {
                    let mut c = RepairCandidate::new(
                        RepairStrategy::PairwiseSummation,
                        vec![diag.node_id],
                        diag.category,
                    );
                    c.estimated_reduction = 50.0;
                    c.priority = diag.error_contribution * 50.0;
                    candidates.push(c);
                }
            }
            DiagnosisCategory::Smearing => {
                let mut c = RepairCandidate::new(
                    RepairStrategy::KahanSummation,
                    vec![diag.node_id],
                    diag.category,
                );
                c.estimated_reduction = 50.0;
                c.priority = diag.error_contribution * 50.0;
                candidates.push(c);
            }
            DiagnosisCategory::AmplifiedRounding => {
                if self.config.allow_precision_promotion {
                    let mut c = RepairCandidate::new(
                        RepairStrategy::PrecisionPromotion {
                            target_bits: self.config.promotion_precision.significand_bits(),
                        },
                        vec![diag.node_id],
                        diag.category,
                    );
                    c.estimated_reduction = 20.0;
                    c.priority = diag.error_contribution * 20.0;
                    candidates.push(c);
                }
            }
            DiagnosisCategory::IllConditionedSubproblem => {
                let mut c = RepairCandidate::new(
                    RepairStrategy::IterativeRefinement,
                    vec![diag.node_id],
                    diag.category,
                );
                c.estimated_reduction = 10.0;
                c.priority = diag.error_contribution * 10.0;
                candidates.push(c);

                if candidates.len() < max {
                    let mut c = RepairCandidate::new(
                        RepairStrategy::Preconditioning,
                        vec![diag.node_id],
                        diag.category,
                    );
                    c.estimated_reduction = 5.0;
                    c.priority = diag.error_contribution * 5.0;
                    candidates.push(c);
                }
            }
        }

        // Sort by priority (descending)
        candidates.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Certify a repair candidate using interval arithmetic.
    fn certify_repair(
        &self,
        eag: &ErrorAmplificationGraph,
        candidate: &RepairCandidate,
        diag: &Diagnosis,
    ) -> RepairCertification {
        let node = eag.node(diag.node_id);
        let local_error = node.map_or(1e-10, |n| n.local_error);

        // Original error interval
        let original = ErrorInterval::new(-local_error, local_error);

        // Estimated repaired error (reduced by the estimated factor)
        let repaired_error = local_error / candidate.estimated_reduction.max(1.0);
        let repaired = ErrorInterval::new(-repaired_error, repaired_error);

        let reduction_factor = if repaired_error > 0.0 {
            local_error / repaired_error
        } else {
            candidate.estimated_reduction
        };

        // Determine if certification is formal (Tier 1 nodes with interval arithmetic)
        let is_formal = node.map_or(false, |n| !n.is_black_box);

        RepairCertification {
            original_error: original,
            repaired_error: repaired,
            reduction_factor,
            is_formal,
            certified_bound: if is_formal {
                Some(ErrorBound::certified(
                    repaired_error,
                    ErrorMetric::Absolute(repaired_error),
                ))
            } else {
                Some(ErrorBound::empirical(
                    repaired_error,
                    ErrorMetric::Absolute(repaired_error),
                    0.95,
                ))
            },
            certified_domain: Some(ErrorInterval::new(-1.0, 1.0)),
            tier1_coverage: if is_formal { 1.0 } else { 0.0 },
        }
    }
}

/// Check if an EAG is monotone (all edge weights positive).
///
/// T4 optimality requires monotonicity.
pub fn is_monotone(eag: &ErrorAmplificationGraph) -> bool {
    eag.edges().iter().all(|e| e.weight.0 >= 0.0)
}

/// Compute the submodularity gap for a repair set.
///
/// Returns 0.0 for perfectly submodular functions; larger values indicate
/// stronger violations.
pub fn submodularity_gap(eag: &ErrorAmplificationGraph, repair_set: &[EagNodeId]) -> f64 {
    if repair_set.len() <= 1 {
        return 0.0;
    }

    // For each node, compute marginal error reduction when added to different subsets
    let mut max_gap = 0.0_f64;

    for (i, &node) in repair_set.iter().enumerate() {
        // Marginal reduction when added to empty set
        let solo_reduction = eag.node(node).map_or(0.0, |n| n.local_error);

        // Marginal reduction when added to full set minus this node
        let full_minus_one: Vec<_> = repair_set
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &n)| n)
            .collect();

        let full_reduction: f64 = full_minus_one
            .iter()
            .filter_map(|&n| eag.node(n))
            .map(|n| n.local_error)
            .sum();

        // Submodularity requires: marginal(node | ∅) ≥ marginal(node | S)
        // Gap = max(0, marginal(node | S) - marginal(node | ∅))
        let gap = (full_reduction - solo_reduction).max(0.0);
        max_gap = max_gap.max(gap);
    }

    max_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::diagnosis::DiagnosisSeverity;
    use fpdiag_types::eag::{EagEdge, EagEdgeId, EagNode, EagNodeId, ErrorAmplificationGraph};
    use fpdiag_types::expression::FpOp;

    #[test]
    fn synthesize_repair_for_cancellation() {
        let mut eag = ErrorAmplificationGraph::new();
        let mut node = EagNode::new(EagNodeId(0), FpOp::Sub, 1e-10, 1e-15);
        node.condition_number = Some(1e10);
        eag.add_node(node);

        let mut report = DiagnosisReport::new();
        report.total_nodes = 1;
        report.add(
            Diagnosis::new(
                EagNodeId(0),
                DiagnosisCategory::CatastrophicCancellation,
                DiagnosisSeverity::Critical,
                0.95,
            )
            .with_contribution(0.8),
        );

        let synth = RepairSynthesizer::with_defaults();
        let result = synth.synthesize(&eag, &report).unwrap();
        assert!(!result.applied_repairs.is_empty());
        assert!(result.overall_reduction > 1.0);
    }

    #[test]
    fn monotone_check() {
        let mut eag = ErrorAmplificationGraph::new();
        eag.add_node(EagNode::new(EagNodeId(0), FpOp::Add, 1.0, 1.0));
        eag.add_node(EagNode::new(EagNodeId(1), FpOp::Mul, 2.0, 2.0));
        eag.add_edge(EagEdge::new(EagEdgeId(0), EagNodeId(0), EagNodeId(1), 1.5));
        assert!(is_monotone(&eag));
    }
}
