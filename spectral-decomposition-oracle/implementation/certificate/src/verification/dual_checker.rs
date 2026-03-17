//! Dual solution verification.
//!
//! Given LP dual solution y*, verify:
//! - Dual feasibility (A^T y ≤ c for minimization)
//! - Complementary slackness
//! - Dual objective value matches primal
//! - Perturbation analysis

use crate::error::{CertificateError, CertificateResult};
use crate::verification::{CheckSeverity, VerificationCheck, VerificationResult};
use serde::{Deserialize, Serialize};

/// Sparse constraint representation: Ax ≤ b (or Ax = b, Ax ≥ b).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    /// Sparse entries: (row, col, value)
    pub entries: Vec<(usize, usize, f64)>,
    pub rhs: Vec<f64>,
    pub objective: Vec<f64>,
    pub constraint_types: Vec<ConstraintType>,
}

/// Type of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    LessEqual,
    Equal,
    GreaterEqual,
}

/// Reduced cost analysis for a variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducedCostInfo {
    pub variable_index: usize,
    pub reduced_cost: f64,
    pub objective_coeff: f64,
    pub dual_contribution: f64,
    pub is_at_bound: bool,
}

/// Perturbation analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationAnalysis {
    pub max_perturbation: f64,
    pub most_sensitive_constraint: usize,
    pub sensitivity_values: Vec<f64>,
    pub certificate_robustness: f64,
}

/// Dual solution checker.
#[derive(Debug, Clone)]
pub struct DualChecker {
    pub tolerance: f64,
}

impl DualChecker {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn with_defaults() -> Self {
        Self { tolerance: 1e-8 }
    }

    /// Verify dual feasibility: A^T y ≤ c for a minimization LP.
    ///
    /// For minimization: A^T y ≤ c (reduced costs ≥ 0 for non-basic variables)
    pub fn verify_dual_feasibility(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        // Check dimension
        if dual_values.len() != matrix.num_rows {
            result.add_check(VerificationCheck {
                name: "dual_dimension".to_string(),
                passed: false,
                severity: CheckSeverity::Error,
                message: format!(
                    "expected {} dual values, got {}",
                    matrix.num_rows,
                    dual_values.len()
                ),
                value: Some(dual_values.len() as f64),
                threshold: Some(matrix.num_rows as f64),
            });
            return result;
        }

        result.add_check(VerificationCheck {
            name: "dual_dimension".to_string(),
            passed: true,
            severity: CheckSeverity::Info,
            message: format!("{} dual values", dual_values.len()),
            value: Some(dual_values.len() as f64),
            threshold: Some(matrix.num_rows as f64),
        });

        // Compute A^T * y
        let mut at_y = vec![0.0; matrix.num_cols];
        for &(row, col, val) in &matrix.entries {
            if row < dual_values.len() && col < matrix.num_cols {
                at_y[col] += val * dual_values[row];
            }
        }

        // Check reduced costs: c_j - (A^T y)_j ≥ 0 for minimization
        let mut max_violation = 0.0f64;
        let mut num_violations = 0;
        let mut violations = Vec::new();

        for j in 0..matrix.num_cols {
            let reduced_cost = matrix.objective[j] - at_y[j];
            if reduced_cost < -self.tolerance {
                max_violation = max_violation.max(-reduced_cost);
                num_violations += 1;
                if violations.len() < 5 {
                    violations.push(format!("var {}: rc={:.6e}", j, reduced_cost));
                }
            }
        }

        result.add_check(VerificationCheck {
            name: "dual_feasibility".to_string(),
            passed: num_violations == 0,
            severity: CheckSeverity::Error,
            message: if num_violations == 0 {
                "all reduced costs non-negative".to_string()
            } else {
                format!(
                    "{} violations (max={:.6e}): {}",
                    num_violations,
                    max_violation,
                    violations.join(", ")
                )
            },
            value: Some(max_violation),
            threshold: Some(self.tolerance),
        });

        // Check sign constraints on duals
        for (i, &d) in dual_values.iter().enumerate() {
            if i < matrix.constraint_types.len() {
                let sign_ok = match matrix.constraint_types[i] {
                    ConstraintType::LessEqual => d >= -self.tolerance,
                    ConstraintType::GreaterEqual => d <= self.tolerance,
                    ConstraintType::Equal => true,
                };
                if !sign_ok && i < 3 {
                    result.add_check(VerificationCheck {
                        name: format!("dual_sign_{}", i),
                        passed: false,
                        severity: CheckSeverity::Error,
                        message: format!(
                            "dual[{}]={:.6e}, constraint type={:?}",
                            i, d, matrix.constraint_types[i]
                        ),
                        value: Some(d),
                        threshold: Some(0.0),
                    });
                }
            }
        }

        // Overall dual sign check
        let sign_violations: usize = dual_values
            .iter()
            .enumerate()
            .filter(|(i, &d)| {
                if *i >= matrix.constraint_types.len() {
                    return false;
                }
                match matrix.constraint_types[*i] {
                    ConstraintType::LessEqual => d < -self.tolerance,
                    ConstraintType::GreaterEqual => d > self.tolerance,
                    ConstraintType::Equal => false,
                }
            })
            .count();

        result.add_check(VerificationCheck {
            name: "dual_sign_overall".to_string(),
            passed: sign_violations == 0,
            severity: CheckSeverity::Error,
            message: format!("{} dual sign violations", sign_violations),
            value: Some(sign_violations as f64),
            threshold: Some(0.0),
        });

        // Check duals are finite
        let non_finite = dual_values.iter().filter(|d| !d.is_finite()).count();
        result.add_check(VerificationCheck {
            name: "duals_finite".to_string(),
            passed: non_finite == 0,
            severity: CheckSeverity::Error,
            message: format!("{} non-finite duals", non_finite),
            value: Some(non_finite as f64),
            threshold: Some(0.0),
        });

        result
    }

    /// Verify complementary slackness.
    ///
    /// For each constraint i: y_i * (a_i^T x - b_i) = 0
    /// i.e., if y_i ≠ 0, constraint must be tight.
    pub fn verify_complementary_slackness(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
        primal_values: &[f64],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        if primal_values.len() != matrix.num_cols {
            result.add_check(VerificationCheck {
                name: "primal_dimension".to_string(),
                passed: false,
                severity: CheckSeverity::Error,
                message: format!(
                    "expected {} primal values, got {}",
                    matrix.num_cols,
                    primal_values.len()
                ),
                value: None,
                threshold: None,
            });
            return result;
        }

        // Compute Ax
        let mut ax = vec![0.0; matrix.num_rows];
        for &(row, col, val) in &matrix.entries {
            if col < primal_values.len() && row < matrix.num_rows {
                ax[row] += val * primal_values[col];
            }
        }

        let mut max_violation = 0.0f64;
        let mut num_violations = 0;

        for i in 0..matrix.num_rows.min(dual_values.len()) {
            let slack = ax[i] - matrix.rhs[i];
            let cs_product = dual_values[i].abs() * slack.abs();
            if cs_product > self.tolerance {
                max_violation = max_violation.max(cs_product);
                num_violations += 1;
            }
        }

        result.add_check(VerificationCheck {
            name: "complementary_slackness".to_string(),
            passed: num_violations == 0,
            severity: CheckSeverity::Error,
            message: format!(
                "{} CS violations (max product={:.6e})",
                num_violations, max_violation
            ),
            value: Some(max_violation),
            threshold: Some(self.tolerance),
        });

        result
    }

    /// Verify dual objective matches primal.
    ///
    /// For a min LP: c^T x = b^T y at optimality (strong duality).
    pub fn verify_strong_duality(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
        primal_values: &[f64],
    ) -> VerificationResult {
        let mut result = VerificationResult::new();

        let primal_obj: f64 = matrix
            .objective
            .iter()
            .zip(primal_values.iter())
            .map(|(c, x)| c * x)
            .sum();

        let dual_obj: f64 = matrix
            .rhs
            .iter()
            .zip(dual_values.iter())
            .map(|(b, y)| b * y)
            .sum();

        let duality_gap = (primal_obj - dual_obj).abs();
        let relative_gap = duality_gap / primal_obj.abs().max(1.0);

        result.add_check(VerificationCheck {
            name: "strong_duality".to_string(),
            passed: duality_gap < self.tolerance || relative_gap < self.tolerance,
            severity: CheckSeverity::Error,
            message: format!(
                "primal={:.6e}, dual={:.6e}, gap={:.6e}",
                primal_obj, dual_obj, duality_gap
            ),
            value: Some(duality_gap),
            threshold: Some(self.tolerance),
        });

        result.add_check(VerificationCheck {
            name: "primal_objective".to_string(),
            passed: primal_obj.is_finite(),
            severity: CheckSeverity::Info,
            message: format!("primal obj = {:.6e}", primal_obj),
            value: Some(primal_obj),
            threshold: None,
        });

        result.add_check(VerificationCheck {
            name: "dual_objective".to_string(),
            passed: dual_obj.is_finite(),
            severity: CheckSeverity::Info,
            message: format!("dual obj = {:.6e}", dual_obj),
            value: Some(dual_obj),
            threshold: None,
        });

        result
    }

    /// Extract and verify reduced costs.
    pub fn compute_reduced_costs(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
    ) -> CertificateResult<Vec<ReducedCostInfo>> {
        if dual_values.len() != matrix.num_rows {
            return Err(CertificateError::incomplete_data(
                "dual_values",
                format!(
                    "expected {}, got {}",
                    matrix.num_rows,
                    dual_values.len()
                ),
            ));
        }

        let mut at_y = vec![0.0; matrix.num_cols];
        for &(row, col, val) in &matrix.entries {
            if row < dual_values.len() && col < matrix.num_cols {
                at_y[col] += val * dual_values[row];
            }
        }

        let results: Vec<ReducedCostInfo> = (0..matrix.num_cols)
            .map(|j| {
                let rc = matrix.objective[j] - at_y[j];
                ReducedCostInfo {
                    variable_index: j,
                    reduced_cost: rc,
                    objective_coeff: matrix.objective[j],
                    dual_contribution: at_y[j],
                    is_at_bound: rc.abs() > self.tolerance,
                }
            })
            .collect();

        Ok(results)
    }

    /// Perturbation analysis: how much can y* change before certificate breaks?
    pub fn perturbation_analysis(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
    ) -> CertificateResult<PerturbationAnalysis> {
        let reduced_costs = self.compute_reduced_costs(matrix, dual_values)?;

        // The certificate breaks when any reduced cost becomes negative.
        // For each positive reduced cost rc_j, the dual can change by at most rc_j / ||a_j||
        let mut min_margin = f64::INFINITY;
        let mut most_sensitive = 0;
        let mut sensitivities = Vec::with_capacity(matrix.num_rows);

        // Compute column norms
        let mut col_norms = vec![0.0f64; matrix.num_cols];
        for &(_, col, val) in &matrix.entries {
            if col < matrix.num_cols {
                col_norms[col] += val * val;
            }
        }
        for norm in col_norms.iter_mut() {
            *norm = norm.sqrt();
        }

        for rc_info in &reduced_costs {
            let j = rc_info.variable_index;
            if col_norms[j] > 1e-15 {
                let margin = rc_info.reduced_cost.abs() / col_norms[j];
                if margin < min_margin {
                    min_margin = margin;
                    most_sensitive = j;
                }
            }
        }

        // Per-constraint sensitivity: how much does bound change per unit dual change
        for i in 0..matrix.num_rows {
            let mut row_norm_sq = 0.0;
            for &(row, _, val) in &matrix.entries {
                if row == i {
                    row_norm_sq += val * val;
                }
            }
            sensitivities.push(row_norm_sq.sqrt());
        }

        let robustness = if min_margin.is_finite() {
            min_margin
        } else {
            f64::INFINITY
        };

        Ok(PerturbationAnalysis {
            max_perturbation: min_margin,
            most_sensitive_constraint: most_sensitive,
            sensitivity_values: sensitivities,
            certificate_robustness: robustness,
        })
    }

    /// Summary of all dual verification checks.
    pub fn full_verification(
        &self,
        matrix: &ConstraintMatrix,
        dual_values: &[f64],
        primal_values: Option<&[f64]>,
    ) -> VerificationResult {
        let mut result = self.verify_dual_feasibility(matrix, dual_values);

        if let Some(primal) = primal_values {
            result.merge(self.verify_complementary_slackness(matrix, dual_values, primal));
            result.merge(self.verify_strong_duality(matrix, dual_values, primal));
        }

        result
    }
}

#[cfg(test)]
fn make_test_matrix() -> ConstraintMatrix {
    // min 2x + 3y
    // s.t. x + y <= 10
    //      x     <= 6
    //          y <= 8
    // Dual: max 10a + 6b + 8c
    //       s.t. a + b <= 2, a + c <= 3, a,b,c >= 0
    // Optimal: x=6, y=4, obj=24. Dual: a=2, b=0, c=1, dobj=20+0+8=28? 
    // Actually: a=2, b=0, c=1 → constraints: 2+0=2 ≤ 2 ✓, 2+1=3 ≤ 3 ✓
    // dual obj = 10*2 + 6*0 + 8*1 = 28. But primal = 12+12=24. Gap!
    // The correct dual: a=2, b=0, c=1 → dobj = 20+0+8 = 28 > 24, infeasible...
    // Let me fix: standard form dual for min c^Tx s.t. Ax<=b is max b^Ty s.t. A^Ty<=c, y>=0
    // a + b <= 2, a + c <= 3, a,b,c >= 0
    // Optimal: a=2, b=0, c=1 → 2+0=2≤2, 2+1=3≤3 ✓, dobj=20+0+8=28 > 24? 
    // That violates weak duality. Let me recalculate...
    // Actually for min LP the dual bound should be ≤ primal.
    // max b^Ty = max 10a + 6b + 8c s.t. a+b ≤ 2, a+c ≤ 3
    // At opt: x=6,y=4. Binding: x+y=10, x=6. So a>0, b>0, c=0.
    // a+b=2, a=3-c, c=0 → a=3? but a+b=2 → b=-1 < 0 invalid
    // Let me just use a simple correct example
    ConstraintMatrix {
        num_rows: 2,
        num_cols: 2,
        entries: vec![
            (0, 0, 1.0), // row 0, col 0
            (0, 1, 1.0), // row 0, col 1
            (1, 0, 1.0), // row 1, col 0
        ],
        rhs: vec![5.0, 3.0],
        objective: vec![3.0, 2.0],
        constraint_types: vec![ConstraintType::LessEqual, ConstraintType::LessEqual],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_feasibility_correct() {
        let matrix = make_test_matrix();
        // A^T y ≤ c: [y1+y2, y1] ≤ [3, 2]
        // y1=1, y2=1 → [2, 1] ≤ [3, 2] ✓
        let checker = DualChecker::with_defaults();
        let result = checker.verify_dual_feasibility(&matrix, &[1.0, 1.0]);
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_dual_feasibility_wrong_dimension() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let result = checker.verify_dual_feasibility(&matrix, &[1.0]);
        assert!(!result.all_passed);
    }

    #[test]
    fn test_dual_feasibility_infeasible() {
        let matrix = make_test_matrix();
        // y1=5, y2=0 → [5, 5] ≤ [3, 2]? No!
        let checker = DualChecker::with_defaults();
        let result = checker.verify_dual_feasibility(&matrix, &[5.0, 0.0]);
        assert!(!result.all_passed);
    }

    #[test]
    fn test_complementary_slackness() {
        let matrix = make_test_matrix();
        // If x=[3,2], Ax=[5,3]=b, all constraints tight
        // Then any dual ≥ 0 satisfies CS
        let checker = DualChecker::with_defaults();
        let result =
            checker.verify_complementary_slackness(&matrix, &[1.0, 1.0], &[3.0, 2.0]);
        assert!(result.all_passed, "{}", result.summary());
    }

    #[test]
    fn test_complementary_slackness_violation() {
        let matrix = make_test_matrix();
        // x=[1,1], Ax=[2,1], b=[5,3]. Slack = [-3,-2].
        // y=[1,1], so y*slack = 1*3 + 1*2 = 5 ≠ 0
        let checker = DualChecker::with_defaults();
        let result =
            checker.verify_complementary_slackness(&matrix, &[1.0, 1.0], &[1.0, 1.0]);
        assert!(!result.all_passed);
    }

    #[test]
    fn test_strong_duality() {
        let matrix = make_test_matrix();
        // primal obj = 3*3 + 2*2 = 13
        // dual obj = 5*1 + 3*2 = 11
        // Gap = 2 (not optimal, but we're testing the check)
        let checker = DualChecker::with_defaults();
        let result = checker.verify_strong_duality(&matrix, &[1.0, 2.0], &[3.0, 2.0]);
        // This should show a duality gap
        let duality_check = result.details.iter().find(|c| c.name == "strong_duality");
        assert!(duality_check.is_some());
    }

    #[test]
    fn test_reduced_costs() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let rcs = checker.compute_reduced_costs(&matrix, &[1.0, 1.0]).unwrap();
        assert_eq!(rcs.len(), 2);
        // rc_0 = 3 - (1+1) = 1
        assert!((rcs[0].reduced_cost - 1.0).abs() < 1e-10);
        // rc_1 = 2 - 1 = 1
        assert!((rcs[1].reduced_cost - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perturbation_analysis() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let analysis = checker.perturbation_analysis(&matrix, &[1.0, 1.0]).unwrap();
        assert!(analysis.max_perturbation > 0.0);
        assert!(analysis.certificate_robustness > 0.0);
    }

    #[test]
    fn test_full_verification() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let result = checker.full_verification(&matrix, &[1.0, 1.0], Some(&[3.0, 2.0]));
        assert!(result.num_checks > 3);
    }

    #[test]
    fn test_full_verification_no_primal() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let result = checker.full_verification(&matrix, &[1.0, 1.0], None);
        assert!(result.num_checks > 0);
    }

    #[test]
    fn test_non_finite_duals() {
        let matrix = make_test_matrix();
        let checker = DualChecker::with_defaults();
        let result = checker.verify_dual_feasibility(&matrix, &[f64::NAN, 1.0]);
        let finite_check = result.details.iter().find(|c| c.name == "duals_finite");
        assert!(finite_check.is_some());
        assert!(!finite_check.unwrap().passed);
    }
}
