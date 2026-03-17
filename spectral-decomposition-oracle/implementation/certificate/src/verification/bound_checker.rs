//! Bound verification: independently verify L3, T2, and futility certificates.

use crate::futility::certificate::FutilityCertificate;
use crate::l3_bound::partition_bound::{ConstraintInfo, L3PartitionCertificate};
use crate::spectral_bound::scaling_law::SpectralScalingCertificate;
use crate::verification::{CheckSeverity, VerificationCheck, VerificationResult};
use serde::{Deserialize, Serialize};

/// Configuration for bound checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundCheckerConfig {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_condition_number: f64,
    pub max_bound_ratio: f64,
}

impl Default for BoundCheckerConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            max_condition_number: 1e12,
            max_bound_ratio: 1e10,
        }
    }
}

/// Independent bound verifier.
#[derive(Debug, Clone)]
pub struct BoundChecker {
    pub config: BoundCheckerConfig,
}

impl BoundChecker {
    pub fn new(config: BoundCheckerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self {
            config: BoundCheckerConfig::default(),
        }
    }

    /// Verify an L3 certificate by recomputing crossing weight, checking dual
    /// feasibility, and verifying partition covers all variables.
    pub fn verify_l3_certificate(
        &self,
        cert: &L3PartitionCertificate,
        constraints: &[ConstraintInfo],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check 1: Partition covers all variables
        result.add_check(self.check_partition_coverage(cert));

        // Check 2: No empty blocks
        result.add_check(self.check_no_empty_blocks(cert));

        // Check 3: Recompute crossing weight
        result.add_check(self.check_crossing_weight_recomputation(cert, constraints));

        // Check 4: Per-edge contributions
        result.add_check(self.check_per_edge_contributions(cert));

        // Check 5: Total bound is sum of contributions
        result.add_check(self.check_total_bound_sum(cert));

        // Check 6: Dual values are finite
        result.add_check(self.check_dual_values_finite(cert));

        // Check 7: Bound is non-negative
        result.add_check(self.check_bound_nonnegative(cert));

        // Check 8: If actual gap known, check bound >= gap
        if let Some(gap) = cert.actual_gap {
            result.add_check(self.check_bound_exceeds_gap(cert.total_bound, gap));
        }

        // Check 9: Crossing edges reference valid constraints
        result.add_check(self.check_crossing_edge_validity(cert, constraints.len()));

        result
    }

    fn check_partition_coverage(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        let n = cert.partition.num_variables;
        let assigned = cert.partition.block_assignments.len();
        VerificationCheck {
            name: "partition_coverage".to_string(),
            passed: assigned == n,
            severity: CheckSeverity::Error,
            message: format!("{}/{} variables assigned", assigned, n),
            value: Some(assigned as f64),
            threshold: Some(n as f64),
        }
    }

    fn check_no_empty_blocks(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        let empty_blocks: Vec<usize> = cert
            .partition
            .block_sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == 0)
            .map(|(i, _)| i)
            .collect();
        VerificationCheck {
            name: "no_empty_blocks".to_string(),
            passed: empty_blocks.is_empty(),
            severity: CheckSeverity::Error,
            message: if empty_blocks.is_empty() {
                "all blocks non-empty".to_string()
            } else {
                format!("empty blocks: {:?}", empty_blocks)
            },
            value: Some(empty_blocks.len() as f64),
            threshold: Some(0.0),
        }
    }

    fn check_crossing_weight_recomputation(
        &self,
        cert: &L3PartitionCertificate,
        constraints: &[ConstraintInfo],
    ) -> VerificationCheck {
        let mut recomputed = 0.0;
        for (ci, constraint) in constraints.iter().enumerate() {
            if ci >= cert.dual_values.len() {
                break;
            }
            let blocks_touched = cert.partition.blocks_touched_by(&constraint.variable_indices);
            if blocks_touched.len() > 1 {
                recomputed +=
                    cert.dual_values[ci].abs() * (blocks_touched.len() as f64 - 1.0);
            }
        }

        let diff = (recomputed - cert.total_bound).abs();
        let relative = if cert.total_bound.abs() > 1e-15 {
            diff / cert.total_bound.abs()
        } else {
            diff
        };

        VerificationCheck {
            name: "crossing_weight_recomputation".to_string(),
            passed: relative < self.config.relative_tolerance || diff < self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!(
                "recomputed={:.6e}, stored={:.6e}, diff={:.6e}",
                recomputed, cert.total_bound, diff
            ),
            value: Some(diff),
            threshold: Some(self.config.absolute_tolerance),
        }
    }

    fn check_per_edge_contributions(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        let mut max_error = 0.0f64;
        for edge in &cert.crossing_edges {
            let expected = edge.dual_value.abs() * (edge.num_blocks_touched as f64 - 1.0);
            let error = (expected - edge.contribution).abs();
            max_error = max_error.max(error);
        }
        VerificationCheck {
            name: "per_edge_contributions".to_string(),
            passed: max_error < self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("max per-edge error: {:.6e}", max_error),
            value: Some(max_error),
            threshold: Some(self.config.absolute_tolerance),
        }
    }

    fn check_total_bound_sum(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        let sum: f64 = cert.crossing_edges.iter().map(|e| e.contribution).sum();
        let diff = (sum - cert.total_bound).abs();
        VerificationCheck {
            name: "total_bound_sum".to_string(),
            passed: diff < self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("sum={:.6e}, total={:.6e}, diff={:.6e}", sum, cert.total_bound, diff),
            value: Some(diff),
            threshold: Some(self.config.absolute_tolerance),
        }
    }

    fn check_dual_values_finite(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        let non_finite: Vec<usize> = cert
            .dual_values
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.is_finite())
            .map(|(i, _)| i)
            .collect();
        VerificationCheck {
            name: "dual_values_finite".to_string(),
            passed: non_finite.is_empty(),
            severity: CheckSeverity::Error,
            message: if non_finite.is_empty() {
                "all dual values finite".to_string()
            } else {
                format!("{} non-finite dual values at {:?}", non_finite.len(), &non_finite[..non_finite.len().min(5)])
            },
            value: Some(non_finite.len() as f64),
            threshold: Some(0.0),
        }
    }

    fn check_bound_nonnegative(&self, cert: &L3PartitionCertificate) -> VerificationCheck {
        VerificationCheck {
            name: "bound_nonnegative".to_string(),
            passed: cert.total_bound >= -self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("bound = {:.6e}", cert.total_bound),
            value: Some(cert.total_bound),
            threshold: Some(0.0),
        }
    }

    fn check_bound_exceeds_gap(&self, bound: f64, gap: f64) -> VerificationCheck {
        let passed = bound >= gap - self.config.absolute_tolerance;
        VerificationCheck {
            name: "bound_exceeds_gap".to_string(),
            passed,
            severity: if passed {
                CheckSeverity::Info
            } else {
                CheckSeverity::Error
            },
            message: format!("bound={:.6e}, gap={:.6e}", bound, gap),
            value: Some(bound - gap),
            threshold: Some(-self.config.absolute_tolerance),
        }
    }

    fn check_crossing_edge_validity(
        &self,
        cert: &L3PartitionCertificate,
        num_constraints: usize,
    ) -> VerificationCheck {
        let invalid: Vec<usize> = cert
            .crossing_edges
            .iter()
            .filter(|e| e.constraint_index >= num_constraints)
            .map(|e| e.constraint_index)
            .collect();
        VerificationCheck {
            name: "crossing_edge_validity".to_string(),
            passed: invalid.is_empty(),
            severity: CheckSeverity::Error,
            message: if invalid.is_empty() {
                "all crossing edges reference valid constraints".to_string()
            } else {
                format!("{} invalid constraint references", invalid.len())
            },
            value: Some(invalid.len() as f64),
            threshold: Some(0.0),
        }
    }

    /// Verify a T2 spectral scaling law certificate.
    pub fn verify_t2_certificate(
        &self,
        cert: &SpectralScalingCertificate,
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check 1: Parameters are valid
        result.add_check(VerificationCheck {
            name: "delta_squared_nonneg".to_string(),
            passed: cert.delta_squared >= 0.0,
            severity: CheckSeverity::Error,
            message: format!("δ² = {:.6e}", cert.delta_squared),
            value: Some(cert.delta_squared),
            threshold: Some(0.0),
        });

        result.add_check(VerificationCheck {
            name: "gamma_squared_positive".to_string(),
            passed: cert.gamma_squared > 0.0,
            severity: CheckSeverity::Error,
            message: format!("γ² = {:.6e}", cert.gamma_squared),
            value: Some(cert.gamma_squared),
            threshold: Some(0.0),
        });

        result.add_check(VerificationCheck {
            name: "kappa_valid".to_string(),
            passed: cert.kappa >= 1.0,
            severity: CheckSeverity::Error,
            message: format!("κ = {:.6e}", cert.kappa),
            value: Some(cert.kappa),
            threshold: Some(1.0),
        });

        // Check 2: Recompute constant C
        let expected_c = cert.k as f64 * cert.kappa.powi(4) * cert.c_inf_norm;
        let c_diff = (expected_c - cert.constant_c).abs();
        result.add_check(VerificationCheck {
            name: "constant_c_recompute".to_string(),
            passed: c_diff < self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("expected C={:.6e}, stored={:.6e}", expected_c, cert.constant_c),
            value: Some(c_diff),
            threshold: Some(self.config.absolute_tolerance),
        });

        // Check 3: Recompute bound
        let expected_bound = expected_c * cert.delta_squared / cert.gamma_squared;
        let bound_diff = (expected_bound - cert.bound_value).abs();
        result.add_check(VerificationCheck {
            name: "bound_recompute".to_string(),
            passed: bound_diff < self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!(
                "expected bound={:.6e}, stored={:.6e}",
                expected_bound, cert.bound_value
            ),
            value: Some(bound_diff),
            threshold: Some(self.config.absolute_tolerance),
        });

        // Check 4: Condition number warning
        result.add_check(VerificationCheck {
            name: "condition_number_reasonable".to_string(),
            passed: cert.kappa < self.config.max_condition_number,
            severity: CheckSeverity::Warning,
            message: format!("κ = {:.6e}", cert.kappa),
            value: Some(cert.kappa),
            threshold: Some(self.config.max_condition_number),
        });

        // Check 5: Vacuousness consistency
        let should_be_vacuous = cert.kappa > 1e3 || cert.gamma_squared < 1e-8;
        result.add_check(VerificationCheck {
            name: "vacuousness_consistency".to_string(),
            passed: cert.is_vacuous == should_be_vacuous || !should_be_vacuous,
            severity: CheckSeverity::Warning,
            message: format!(
                "is_vacuous={}, κ={:.2e}, γ²={:.2e}",
                cert.is_vacuous, cert.kappa, cert.gamma_squared
            ),
            value: None,
            threshold: None,
        });

        // Check 6: If empirical gap known, bound should be valid
        if let Some(emp) = cert.empirical_gap {
            result.add_check(VerificationCheck {
                name: "bound_exceeds_empirical".to_string(),
                passed: cert.bound_value >= emp - self.config.absolute_tolerance,
                severity: CheckSeverity::Error,
                message: format!("bound={:.6e}, empirical gap={:.6e}", cert.bound_value, emp),
                value: Some(cert.bound_value - emp),
                threshold: Some(-self.config.absolute_tolerance),
            });
        }

        result
    }

    /// Verify consistency between L3 and T2 bounds (should give compatible bounds).
    pub fn verify_consistency(
        &self,
        l3: &L3PartitionCertificate,
        t2: &SpectralScalingCertificate,
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Both bounds should be non-negative
        result.add_check(VerificationCheck {
            name: "l3_nonneg".to_string(),
            passed: l3.total_bound >= -self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("L3 bound = {:.6e}", l3.total_bound),
            value: Some(l3.total_bound),
            threshold: Some(0.0),
        });

        result.add_check(VerificationCheck {
            name: "t2_nonneg".to_string(),
            passed: t2.bound_value >= -self.config.absolute_tolerance,
            severity: CheckSeverity::Error,
            message: format!("T2 bound = {:.6e}", t2.bound_value),
            value: Some(t2.bound_value),
            threshold: Some(0.0),
        });

        // If both are non-vacuous, they should be in a reasonable ratio
        if !t2.is_vacuous && l3.total_bound > 1e-15 {
            let ratio = t2.bound_value / l3.total_bound;
            result.add_check(VerificationCheck {
                name: "bound_ratio_reasonable".to_string(),
                passed: ratio < self.config.max_bound_ratio,
                severity: CheckSeverity::Warning,
                message: format!(
                    "T2/L3 ratio = {:.2e} (T2={:.6e}, L3={:.6e})",
                    ratio, t2.bound_value, l3.total_bound
                ),
                value: Some(ratio),
                threshold: Some(self.config.max_bound_ratio),
            });
        }

        // If actual gaps are known, they should agree
        if let (Some(l3_gap), Some(t2_gap)) = (l3.actual_gap, t2.empirical_gap) {
            let gap_diff = (l3_gap - t2_gap).abs();
            result.add_check(VerificationCheck {
                name: "actual_gaps_agree".to_string(),
                passed: gap_diff < self.config.absolute_tolerance,
                severity: CheckSeverity::Warning,
                message: format!("L3 gap={:.6e}, T2 gap={:.6e}", l3_gap, t2_gap),
                value: Some(gap_diff),
                threshold: Some(self.config.absolute_tolerance),
            });
        }

        result
    }

    /// Basic verification of a futility certificate.
    pub fn verify_futility_certificate(
        &self,
        cert: &FutilityCertificate,
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check: score in valid range
        result.add_check(VerificationCheck {
            name: "score_range".to_string(),
            passed: cert.futility_score >= 0.0 && cert.futility_score <= 1.0,
            severity: CheckSeverity::Error,
            message: format!("futility_score = {:.4}", cert.futility_score),
            value: Some(cert.futility_score),
            threshold: None,
        });

        // Check: confidence in valid range
        result.add_check(VerificationCheck {
            name: "confidence_range".to_string(),
            passed: cert.confidence >= 0.0 && cert.confidence <= 1.0,
            severity: CheckSeverity::Error,
            message: format!("confidence = {:.4}", cert.confidence),
            value: Some(cert.confidence),
            threshold: None,
        });

        // Check: prediction consistent with score and threshold
        let expected_futile = cert.futility_score > cert.thresholds.combined_score_threshold + cert.thresholds.uncertainty_margin;
        let expected_not_futile = cert.futility_score < cert.thresholds.combined_score_threshold - cert.thresholds.uncertainty_margin;
        let prediction_consistent = match cert.prediction {
            crate::futility::FutilityPrediction::Futile => expected_futile,
            crate::futility::FutilityPrediction::NotFutile => expected_not_futile,
            crate::futility::FutilityPrediction::Uncertain => !expected_futile && !expected_not_futile,
        };
        result.add_check(VerificationCheck {
            name: "prediction_consistency".to_string(),
            passed: prediction_consistent,
            severity: CheckSeverity::Error,
            message: format!(
                "prediction={}, score={:.4}, threshold={:.4}±{:.4}",
                cert.prediction,
                cert.futility_score,
                cert.thresholds.combined_score_threshold,
                cert.thresholds.uncertainty_margin,
            ),
            value: Some(cert.futility_score),
            threshold: Some(cert.thresholds.combined_score_threshold),
        });

        // Check: certificate type is empirical
        result.add_check(VerificationCheck {
            name: "certificate_type".to_string(),
            passed: cert.certificate_type == crate::futility::CertificateType::Empirical,
            severity: CheckSeverity::Error,
            message: format!("type = {}", cert.certificate_type),
            value: None,
            threshold: None,
        });

        // Check: features are valid
        result.add_check(VerificationCheck {
            name: "features_valid".to_string(),
            passed: cert.features.validate().is_ok(),
            severity: CheckSeverity::Error,
            message: "feature validation".to_string(),
            value: None,
            threshold: None,
        });

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l3_bound::partition_bound::{Partition, PartitionMethod};

    fn make_test_l3() -> (L3PartitionCertificate, Vec<ConstraintInfo>) {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let constraints = vec![
            ConstraintInfo {
                index: 0,
                name: "c0".into(),
                variable_indices: vec![0, 2],
            },
            ConstraintInfo {
                index: 1,
                name: "c1".into(),
                variable_indices: vec![0, 1],
            },
        ];
        let duals = vec![3.0, 2.0];
        let cert =
            L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();
        (cert, constraints)
    }

    #[test]
    fn test_verify_l3_valid() {
        let (cert, constraints) = make_test_l3();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_l3_certificate(&cert, &constraints);
        assert!(result.all_passed);
    }

    #[test]
    fn test_verify_l3_all_checks_present() {
        let (cert, constraints) = make_test_l3();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_l3_certificate(&cert, &constraints);
        assert!(result.num_checks >= 8);
    }

    #[test]
    fn test_verify_t2_valid() {
        let cert =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_t2_certificate(&cert);
        assert!(result.all_passed);
    }

    #[test]
    fn test_verify_t2_high_kappa_warning() {
        let cert =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 1e13, 10.0, vec![]).unwrap();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_t2_certificate(&cert);
        assert!(result.num_warnings > 0);
    }

    #[test]
    fn test_verify_consistency() {
        let (l3, _) = make_test_l3();
        let t2 =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_consistency(&l3, &t2);
        assert!(result.num_checks >= 2);
    }

    #[test]
    fn test_verify_futility() {
        use crate::futility::certificate::{FutilityCertificate, FutilityThresholds, SpectralFeatures};
        let features = SpectralFeatures::new(0.5, 0.1, 0.3, 10.0, 3);
        let cert =
            FutilityCertificate::generate(features, FutilityThresholds::default(), None).unwrap();
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_futility_certificate(&cert);
        assert!(result.all_passed);
    }

    #[test]
    fn test_bound_checker_config() {
        let config = BoundCheckerConfig {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            max_condition_number: 1e6,
            max_bound_ratio: 1e8,
        };
        let checker = BoundChecker::new(config);
        assert!((checker.config.absolute_tolerance - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_verify_l3_with_gap() {
        let (mut cert, constraints) = make_test_l3();
        cert.set_objectives(100.0, 98.0);
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_l3_certificate(&cert, &constraints);
        let gap_check = result.details.iter().find(|c| c.name == "bound_exceeds_gap");
        assert!(gap_check.is_some());
    }

    #[test]
    fn test_verify_t2_with_empirical() {
        let mut cert =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        cert.compare_with_empirical(5.0);
        let checker = BoundChecker::with_defaults();
        let result = checker.verify_t2_certificate(&cert);
        let emp_check = result.details.iter().find(|c| c.name == "bound_exceeds_empirical");
        assert!(emp_check.is_some());
    }
}
