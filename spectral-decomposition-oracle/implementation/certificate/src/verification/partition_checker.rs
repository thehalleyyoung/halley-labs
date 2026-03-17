//! Partition verification.
//!
//! Checks partition integrity: all variables assigned, no overlaps, block count
//! matches, crossing edges correctly identified, crossing weight correct.

use crate::l3_bound::partition_bound::{ConstraintInfo, L3PartitionCertificate, Partition};
use crate::verification::{CheckSeverity, VerificationCheck, VerificationResult};
use serde::{Deserialize, Serialize};

/// Issues found during partition verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionIssues {
    pub unassigned_variables: Vec<usize>,
    pub duplicate_assignments: Vec<(usize, Vec<usize>)>,
    pub empty_blocks: Vec<usize>,
    pub block_count_mismatch: Option<(usize, usize)>,
    pub crossing_edge_errors: Vec<String>,
    pub crossing_weight_error: Option<f64>,
    pub total_issues: usize,
}

impl PartitionIssues {
    pub fn new() -> Self {
        Self {
            unassigned_variables: Vec::new(),
            duplicate_assignments: Vec::new(),
            empty_blocks: Vec::new(),
            block_count_mismatch: None,
            crossing_edge_errors: Vec::new(),
            crossing_weight_error: None,
            total_issues: 0,
        }
    }

    pub fn has_issues(&self) -> bool {
        self.total_issues > 0
    }

    fn count_issues(&mut self) {
        self.total_issues = self.unassigned_variables.len()
            + self.duplicate_assignments.len()
            + self.empty_blocks.len()
            + if self.block_count_mismatch.is_some() { 1 } else { 0 }
            + self.crossing_edge_errors.len()
            + if self.crossing_weight_error.is_some() { 1 } else { 0 };
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.unassigned_variables.is_empty() {
            parts.push(format!("{} unassigned variables", self.unassigned_variables.len()));
        }
        if !self.duplicate_assignments.is_empty() {
            parts.push(format!(
                "{} duplicate assignments",
                self.duplicate_assignments.len()
            ));
        }
        if !self.empty_blocks.is_empty() {
            parts.push(format!("{} empty blocks", self.empty_blocks.len()));
        }
        if let Some((claimed, actual)) = self.block_count_mismatch {
            parts.push(format!(
                "block count: claimed={}, actual={}",
                claimed, actual
            ));
        }
        if !self.crossing_edge_errors.is_empty() {
            parts.push(format!(
                "{} crossing edge errors",
                self.crossing_edge_errors.len()
            ));
        }
        if let Some(err) = self.crossing_weight_error {
            parts.push(format!("crossing weight error: {:.6e}", err));
        }
        if parts.is_empty() {
            "no issues found".to_string()
        } else {
            parts.join("; ")
        }
    }
}

impl Default for PartitionIssues {
    fn default() -> Self {
        Self::new()
    }
}

/// Partition integrity checker.
#[derive(Debug, Clone)]
pub struct PartitionChecker {
    pub tolerance: f64,
}

impl PartitionChecker {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn with_defaults() -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Full partition verification.
    pub fn verify_partition(
        &self,
        partition: &Partition,
        expected_num_variables: Option<usize>,
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check 1: All variables assigned
        if let Some(expected) = expected_num_variables {
            result.add_check(VerificationCheck {
                name: "variable_count".to_string(),
                passed: partition.num_variables == expected,
                severity: CheckSeverity::Error,
                message: format!(
                    "expected {} variables, partition has {}",
                    expected, partition.num_variables
                ),
                value: Some(partition.num_variables as f64),
                threshold: Some(expected as f64),
            });
        }

        // Check 2: Assignments are valid (within 0..num_blocks)
        let invalid_assignments: Vec<usize> = partition
            .block_assignments
            .iter()
            .enumerate()
            .filter(|(_, &b)| b >= partition.num_blocks)
            .map(|(i, _)| i)
            .collect();

        result.add_check(VerificationCheck {
            name: "valid_assignments".to_string(),
            passed: invalid_assignments.is_empty(),
            severity: CheckSeverity::Error,
            message: if invalid_assignments.is_empty() {
                "all assignments valid".to_string()
            } else {
                format!(
                    "{} invalid assignments: {:?}",
                    invalid_assignments.len(),
                    &invalid_assignments[..invalid_assignments.len().min(5)]
                )
            },
            value: Some(invalid_assignments.len() as f64),
            threshold: Some(0.0),
        });

        // Check 3: No empty blocks
        let empty_blocks: Vec<usize> = partition
            .block_sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == 0)
            .map(|(i, _)| i)
            .collect();

        result.add_check(VerificationCheck {
            name: "no_empty_blocks".to_string(),
            passed: empty_blocks.is_empty(),
            severity: CheckSeverity::Warning,
            message: if empty_blocks.is_empty() {
                format!("all {} blocks non-empty", partition.num_blocks)
            } else {
                format!("empty blocks: {:?}", empty_blocks)
            },
            value: Some(empty_blocks.len() as f64),
            threshold: Some(0.0),
        });

        // Check 4: Block sizes sum to num_variables
        let size_sum: usize = partition.block_sizes.iter().sum();
        result.add_check(VerificationCheck {
            name: "block_sizes_sum".to_string(),
            passed: size_sum == partition.num_variables,
            severity: CheckSeverity::Error,
            message: format!(
                "block sizes sum = {}, num_variables = {}",
                size_sum, partition.num_variables
            ),
            value: Some(size_sum as f64),
            threshold: Some(partition.num_variables as f64),
        });

        // Check 5: Block sizes are consistent with assignments
        let mut computed_sizes = vec![0usize; partition.num_blocks];
        for &b in &partition.block_assignments {
            if b < partition.num_blocks {
                computed_sizes[b] += 1;
            }
        }
        let sizes_match = computed_sizes == partition.block_sizes;

        result.add_check(VerificationCheck {
            name: "block_sizes_consistent".to_string(),
            passed: sizes_match,
            severity: CheckSeverity::Error,
            message: if sizes_match {
                "block sizes consistent with assignments".to_string()
            } else {
                format!(
                    "computed {:?} vs stored {:?}",
                    computed_sizes, partition.block_sizes
                )
            },
            value: None,
            threshold: None,
        });

        // Check 6: Reasonable balance
        let balance = partition.balance_ratio();
        result.add_check(VerificationCheck {
            name: "balance_ratio".to_string(),
            passed: balance > 0.01,
            severity: CheckSeverity::Warning,
            message: format!("balance ratio = {:.4}", balance),
            value: Some(balance),
            threshold: Some(0.01),
        });

        result
    }

    /// Verify crossing edges in a certificate.
    pub fn verify_crossing_edges(
        &self,
        cert: &L3PartitionCertificate,
        constraints: &[ConstraintInfo],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check each claimed crossing edge
        for edge in &cert.crossing_edges {
            if edge.constraint_index >= constraints.len() {
                result.add_check(VerificationCheck {
                    name: format!("crossing_edge_{}_valid_index", edge.constraint_index),
                    passed: false,
                    severity: CheckSeverity::Error,
                    message: format!(
                        "constraint index {} out of range [0, {})",
                        edge.constraint_index,
                        constraints.len()
                    ),
                    value: None,
                    threshold: None,
                });
                continue;
            }

            let constraint = &constraints[edge.constraint_index];
            let actual_blocks = cert.partition.blocks_touched_by(&constraint.variable_indices);

            // Verify it's actually crossing
            let is_crossing = actual_blocks.len() > 1;
            result.add_check(VerificationCheck {
                name: format!("crossing_edge_{}_is_crossing", edge.constraint_index),
                passed: is_crossing,
                severity: CheckSeverity::Error,
                message: format!(
                    "edge {} touches {} blocks: {:?}",
                    edge.constraint_index,
                    actual_blocks.len(),
                    actual_blocks
                ),
                value: Some(actual_blocks.len() as f64),
                threshold: Some(2.0),
            });

            // Verify blocks_touched matches
            let blocks_match = edge.num_blocks_touched == actual_blocks.len();
            result.add_check(VerificationCheck {
                name: format!("crossing_edge_{}_blocks_count", edge.constraint_index),
                passed: blocks_match,
                severity: CheckSeverity::Error,
                message: format!(
                    "claimed {} blocks, actual {} blocks",
                    edge.num_blocks_touched,
                    actual_blocks.len()
                ),
                value: Some(edge.num_blocks_touched as f64),
                threshold: Some(actual_blocks.len() as f64),
            });
        }

        // Check: are there missing crossing edges?
        let mut expected_crossing_indices = std::collections::BTreeSet::new();
        for (ci, constraint) in constraints.iter().enumerate() {
            let blocks = cert.partition.blocks_touched_by(&constraint.variable_indices);
            if blocks.len() > 1 {
                expected_crossing_indices.insert(ci);
            }
        }

        let cert_crossing_indices: std::collections::BTreeSet<usize> = cert
            .crossing_edges
            .iter()
            .map(|e| e.constraint_index)
            .collect();

        let missing: Vec<usize> = expected_crossing_indices
            .difference(&cert_crossing_indices)
            .cloned()
            .collect();

        let extra: Vec<usize> = cert_crossing_indices
            .difference(&expected_crossing_indices)
            .cloned()
            .collect();

        result.add_check(VerificationCheck {
            name: "no_missing_crossing_edges".to_string(),
            passed: missing.is_empty(),
            severity: CheckSeverity::Error,
            message: if missing.is_empty() {
                "no missing crossing edges".to_string()
            } else {
                format!("missing {} crossing edges: {:?}", missing.len(), &missing[..missing.len().min(5)])
            },
            value: Some(missing.len() as f64),
            threshold: Some(0.0),
        });

        result.add_check(VerificationCheck {
            name: "no_extra_crossing_edges".to_string(),
            passed: extra.is_empty(),
            severity: CheckSeverity::Warning,
            message: if extra.is_empty() {
                "no extra crossing edges".to_string()
            } else {
                format!("extra {} crossing edges: {:?}", extra.len(), &extra[..extra.len().min(5)])
            },
            value: Some(extra.len() as f64),
            threshold: Some(0.0),
        });

        result
    }

    /// Verify crossing weight computation.
    pub fn verify_crossing_weight(
        &self,
        cert: &L3PartitionCertificate,
        constraints: &[ConstraintInfo],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        let mut recomputed_weight = 0.0;
        for (ci, constraint) in constraints.iter().enumerate() {
            if ci >= cert.dual_values.len() {
                break;
            }
            let blocks = cert.partition.blocks_touched_by(&constraint.variable_indices);
            if blocks.len() > 1 {
                recomputed_weight +=
                    cert.dual_values[ci].abs() * (blocks.len() as f64 - 1.0);
            }
        }

        let diff = (recomputed_weight - cert.total_bound).abs();

        result.add_check(VerificationCheck {
            name: "crossing_weight_correct".to_string(),
            passed: diff < self.tolerance,
            severity: CheckSeverity::Error,
            message: format!(
                "recomputed={:.6e}, stored={:.6e}, diff={:.6e}",
                recomputed_weight, cert.total_bound, diff
            ),
            value: Some(diff),
            threshold: Some(self.tolerance),
        });

        result
    }

    /// Generate a comprehensive partition report.
    pub fn generate_issues_report(
        &self,
        partition: &Partition,
        cert: Option<&L3PartitionCertificate>,
        constraints: Option<&[ConstraintInfo]>,
    ) -> PartitionIssues {
        let mut issues = PartitionIssues::new();

        // Check for empty blocks
        for (i, &sz) in partition.block_sizes.iter().enumerate() {
            if sz == 0 {
                issues.empty_blocks.push(i);
            }
        }

        // Check block sizes consistency
        let mut computed_sizes = vec![0usize; partition.num_blocks];
        for &b in &partition.block_assignments {
            if b < partition.num_blocks {
                computed_sizes[b] += 1;
            }
        }
        if computed_sizes != partition.block_sizes {
            issues.block_count_mismatch = Some((
                partition.block_sizes.iter().sum(),
                computed_sizes.iter().sum(),
            ));
        }

        // Check crossing edges if certificate and constraints provided
        if let (Some(cert), Some(constraints)) = (cert, constraints) {
            for edge in &cert.crossing_edges {
                if edge.constraint_index >= constraints.len() {
                    issues
                        .crossing_edge_errors
                        .push(format!("edge {} out of range", edge.constraint_index));
                    continue;
                }

                let actual_blocks = partition
                    .blocks_touched_by(&constraints[edge.constraint_index].variable_indices);
                if actual_blocks.len() != edge.num_blocks_touched {
                    issues.crossing_edge_errors.push(format!(
                        "edge {} claimed {} blocks, actual {}",
                        edge.constraint_index,
                        edge.num_blocks_touched,
                        actual_blocks.len()
                    ));
                }
            }

            // Verify total crossing weight
            let mut recomputed = 0.0;
            for (ci, constraint) in constraints.iter().enumerate() {
                if ci >= cert.dual_values.len() {
                    break;
                }
                let blocks = partition.blocks_touched_by(&constraint.variable_indices);
                if blocks.len() > 1 {
                    recomputed += cert.dual_values[ci].abs() * (blocks.len() as f64 - 1.0);
                }
            }
            let diff = (recomputed - cert.total_bound).abs();
            if diff > self.tolerance {
                issues.crossing_weight_error = Some(diff);
            }
        }

        issues.count_issues();
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l3_bound::partition_bound::PartitionMethod;

    fn make_good_partition() -> Partition {
        Partition::new(vec![0, 0, 1, 1, 2, 2], 3, PartitionMethod::Manual).unwrap()
    }

    #[test]
    fn test_verify_good_partition() {
        let p = make_good_partition();
        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_partition(&p, Some(6));
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_verify_wrong_variable_count() {
        let p = make_good_partition();
        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_partition(&p, Some(10));
        assert!(!result.all_passed);
    }

    #[test]
    fn test_verify_no_expected_count() {
        let p = make_good_partition();
        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_partition(&p, None);
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_verify_crossing_edges() {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let constraints = vec![
            ConstraintInfo { index: 0, name: "cross".into(), variable_indices: vec![0, 2] },
            ConstraintInfo { index: 1, name: "internal".into(), variable_indices: vec![0, 1] },
        ];
        let duals = vec![3.0, 2.0];
        let cert = L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();

        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_crossing_edges(&cert, &constraints);
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_verify_crossing_weight() {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let constraints = vec![
            ConstraintInfo { index: 0, name: "cross".into(), variable_indices: vec![0, 2] },
        ];
        let duals = vec![5.0];
        let cert = L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();

        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_crossing_weight(&cert, &constraints);
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_generate_issues_report_clean() {
        let p = make_good_partition();
        let checker = PartitionChecker::with_defaults();
        let issues = checker.generate_issues_report(&p, None, None);
        assert!(!issues.has_issues());
    }

    #[test]
    fn test_generate_issues_report_with_cert() {
        let partition = Partition::new(vec![0, 0, 1, 1], 2, PartitionMethod::Manual).unwrap();
        let constraints = vec![
            ConstraintInfo { index: 0, name: "cross".into(), variable_indices: vec![0, 2] },
        ];
        let duals = vec![5.0];
        let cert = L3PartitionCertificate::compute_bound(&partition, &constraints, &duals).unwrap();

        let checker = PartitionChecker::with_defaults();
        let issues = checker.generate_issues_report(&partition, Some(&cert), Some(&constraints));
        assert!(!issues.has_issues());
    }

    #[test]
    fn test_partition_issues_summary() {
        let mut issues = PartitionIssues::new();
        assert_eq!(issues.summary(), "no issues found");
        issues.empty_blocks.push(2);
        issues.count_issues();
        assert!(issues.summary().contains("empty blocks"));
    }

    #[test]
    fn test_balance_warning() {
        let p = Partition::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2, PartitionMethod::Manual).unwrap();
        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_partition(&p, None);
        // Should have a balance warning but not an error
        let balance_check = result.details.iter().find(|c| c.name == "balance_ratio");
        assert!(balance_check.is_some());
    }

    #[test]
    fn test_block_sizes_sum() {
        let p = make_good_partition();
        let checker = PartitionChecker::with_defaults();
        let result = checker.verify_partition(&p, None);
        let sum_check = result.details.iter().find(|c| c.name == "block_sizes_sum");
        assert!(sum_check.is_some());
        assert!(sum_check.unwrap().passed);
    }
}
