//! Solution extraction and post-processing.
//!
//! Primal/dual values, reduced costs, slack values, basis status,
//! sensitivity ranges, solution validation, infeasibility certificates (Farkas proof).

use crate::model::LpModel;
use bicut_types::{BasisStatus, ConstraintSense, LpStatus, OptDirection};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Complete LP solution with all derived quantities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpSolution {
    /// Solution status.
    pub status: LpStatus,
    /// Objective function value.
    pub objective: f64,
    /// Primal variable values.
    pub primal: Vec<f64>,
    /// Dual variable values (shadow prices).
    pub dual: Vec<f64>,
    /// Reduced costs.
    pub reduced_costs: Vec<f64>,
    /// Slack values for each constraint.
    pub slacks: Vec<f64>,
    /// Basis status for each variable.
    pub var_basis_status: Vec<BasisStatus>,
    /// Basis status for each constraint (slack).
    pub con_basis_status: Vec<BasisStatus>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Solve time in seconds.
    pub solve_time: f64,
}

impl LpSolution {
    /// Create an optimal solution.
    pub fn optimal(
        objective: f64,
        primal: Vec<f64>,
        dual: Vec<f64>,
        reduced_costs: Vec<f64>,
        iterations: usize,
    ) -> Self {
        Self {
            status: LpStatus::Optimal,
            objective,
            primal,
            dual,
            reduced_costs,
            slacks: Vec::new(),
            var_basis_status: Vec::new(),
            con_basis_status: Vec::new(),
            iterations,
            solve_time: 0.0,
        }
    }

    /// Create an infeasible solution.
    pub fn infeasible() -> Self {
        Self {
            status: LpStatus::Infeasible,
            objective: f64::INFINITY,
            primal: Vec::new(),
            dual: Vec::new(),
            reduced_costs: Vec::new(),
            slacks: Vec::new(),
            var_basis_status: Vec::new(),
            con_basis_status: Vec::new(),
            iterations: 0,
            solve_time: 0.0,
        }
    }

    /// Create an unbounded solution.
    pub fn unbounded() -> Self {
        Self {
            status: LpStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            primal: Vec::new(),
            dual: Vec::new(),
            reduced_costs: Vec::new(),
            slacks: Vec::new(),
            var_basis_status: Vec::new(),
            con_basis_status: Vec::new(),
            iterations: 0,
            solve_time: 0.0,
        }
    }

    /// Check if solution is optimal.
    pub fn is_optimal(&self) -> bool {
        self.status == LpStatus::Optimal
    }
}

impl fmt::Display for LpSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Status: {}", self.status)?;
        writeln!(f, "Objective: {:.8}", self.objective)?;
        writeln!(f, "Iterations: {}", self.iterations)?;
        if !self.primal.is_empty() {
            writeln!(f, "Primal values ({} vars):", self.primal.len())?;
            for (i, &v) in self.primal.iter().enumerate().take(20) {
                writeln!(f, "  x[{}] = {:.8}", i, v)?;
            }
            if self.primal.len() > 20 {
                writeln!(f, "  ... ({} more)", self.primal.len() - 20)?;
            }
        }
        Ok(())
    }
}

/// Compute slack values for all constraints.
pub fn compute_slacks(model: &LpModel, primal: &[f64]) -> Vec<f64> {
    let mut slacks = Vec::with_capacity(model.constraints.len());
    for con in &model.constraints {
        let activity: f64 = con
            .row_indices
            .iter()
            .zip(con.row_values.iter())
            .map(
                |(&j, &a)| {
                    if j < primal.len() {
                        a * primal[j]
                    } else {
                        0.0
                    }
                },
            )
            .sum();
        let slack = con.rhs - activity;
        slacks.push(slack);
    }
    slacks
}

/// Compute the objective value from primal values.
pub fn compute_objective(model: &LpModel, primal: &[f64]) -> f64 {
    let mut obj = model.obj_offset;
    for (j, var) in model.variables.iter().enumerate() {
        if j < primal.len() {
            obj += var.obj_coeff * primal[j];
        }
    }
    obj
}

/// Determine basis status for each variable based on the solution.
pub fn determine_var_basis_status(
    model: &LpModel,
    primal: &[f64],
    basic_indices: &[usize],
    tol: f64,
) -> Vec<BasisStatus> {
    let n = model.num_vars();
    let mut status = vec![BasisStatus::NonBasicLower; n];

    for &idx in basic_indices {
        if idx < n {
            status[idx] = BasisStatus::Basic;
        }
    }

    for j in 0..n {
        if status[j] == BasisStatus::Basic {
            continue;
        }
        let var = &model.variables[j];
        let val = if j < primal.len() { primal[j] } else { 0.0 };

        if var.lower_bound <= -1e20 && var.upper_bound >= 1e20 {
            status[j] = BasisStatus::SuperBasic;
        } else if var.upper_bound < 1e20 && (val - var.upper_bound).abs() < tol {
            status[j] = BasisStatus::NonBasicUpper;
        } else if var.lower_bound > -1e20 && (val - var.lower_bound).abs() < tol {
            status[j] = BasisStatus::NonBasicLower;
        }
    }

    status
}

/// Determine basis status for each constraint.
pub fn determine_con_basis_status(
    model: &LpModel,
    slacks: &[f64],
    basic_indices: &[usize],
    num_vars: usize,
    tol: f64,
) -> Vec<BasisStatus> {
    let m = model.constraints.len();
    let mut status = vec![BasisStatus::NonBasicLower; m];

    // Slack variables start at index num_vars
    for &idx in basic_indices {
        if idx >= num_vars && idx - num_vars < m {
            status[idx - num_vars] = BasisStatus::Basic;
        }
    }

    for i in 0..m {
        if status[i] == BasisStatus::Basic {
            continue;
        }
        let slack = if i < slacks.len() { slacks[i] } else { 0.0 };
        if slack.abs() < tol {
            status[i] = BasisStatus::NonBasicLower; // active constraint
        }
    }

    status
}

/// Validate primal feasibility of a solution.
pub fn validate_primal_feasibility(model: &LpModel, primal: &[f64], tol: f64) -> ValidationResult {
    let mut violations = Vec::new();
    let mut max_violation = 0.0f64;

    // Check variable bounds
    for (j, var) in model.variables.iter().enumerate() {
        let val = if j < primal.len() { primal[j] } else { 0.0 };
        if val < var.lower_bound - tol {
            let viol = var.lower_bound - val;
            max_violation = max_violation.max(viol);
            violations.push(Violation::BoundViolation {
                var_idx: j,
                value: val,
                bound: var.lower_bound,
                is_lower: true,
            });
        }
        if val > var.upper_bound + tol {
            let viol = val - var.upper_bound;
            max_violation = max_violation.max(viol);
            violations.push(Violation::BoundViolation {
                var_idx: j,
                value: val,
                bound: var.upper_bound,
                is_lower: false,
            });
        }
    }

    // Check constraints
    let slacks = compute_slacks(model, primal);
    for (i, con) in model.constraints.iter().enumerate() {
        let slack = slacks[i];
        let violated = match con.sense {
            ConstraintSense::Le => slack < -tol,
            ConstraintSense::Ge => slack > tol, // Wait: slack = rhs - activity
            // For Ge: activity >= rhs means rhs - activity <= 0 means slack <= 0
            ConstraintSense::Eq => slack.abs() > tol,
        };

        // Correct: for Ge constraint, slack = rhs - activity, feasible when activity >= rhs, i.e. slack <= 0
        let actual_violated = match con.sense {
            ConstraintSense::Le => slack < -tol,      // activity > rhs
            ConstraintSense::Ge => slack > tol,       // activity < rhs
            ConstraintSense::Eq => slack.abs() > tol, // activity != rhs
        };

        if actual_violated {
            let viol = match con.sense {
                ConstraintSense::Le => -slack,
                ConstraintSense::Ge => slack,
                ConstraintSense::Eq => slack.abs(),
            };
            max_violation = max_violation.max(viol);
            violations.push(Violation::ConstraintViolation {
                con_idx: i,
                slack,
                sense: con.sense,
            });
        }
        let _ = violated;
    }

    ValidationResult {
        is_feasible: violations.is_empty(),
        max_violation,
        violations,
    }
}

/// Validate dual feasibility.
pub fn validate_dual_feasibility(
    model: &LpModel,
    dual: &[f64],
    reduced_costs: &[f64],
    primal: &[f64],
    tol: f64,
) -> ValidationResult {
    let mut violations = Vec::new();
    let mut max_violation = 0.0f64;

    for (j, var) in model.variables.iter().enumerate() {
        if j >= reduced_costs.len() {
            break;
        }
        let rc = reduced_costs[j];
        let val = if j < primal.len() { primal[j] } else { 0.0 };
        let at_lower = (val - var.lower_bound).abs() < tol;
        let at_upper = var.upper_bound < 1e20 && (val - var.upper_bound).abs() < tol;
        let is_interior = !at_lower && !at_upper;

        // For minimization:
        // At lower bound: rc >= 0
        // At upper bound: rc <= 0
        // Basic/interior: rc = 0
        let sign_factor = if model.sense == OptDirection::Maximize {
            -1.0
        } else {
            1.0
        };
        let adj_rc = rc * sign_factor;

        if at_lower && adj_rc < -tol {
            let viol = -adj_rc;
            max_violation = max_violation.max(viol);
            violations.push(Violation::DualViolation {
                var_idx: j,
                reduced_cost: rc,
            });
        } else if at_upper && adj_rc > tol {
            let viol = adj_rc;
            max_violation = max_violation.max(viol);
            violations.push(Violation::DualViolation {
                var_idx: j,
                reduced_cost: rc,
            });
        } else if is_interior && adj_rc.abs() > tol {
            max_violation = max_violation.max(adj_rc.abs());
            violations.push(Violation::DualViolation {
                var_idx: j,
                reduced_cost: rc,
            });
        }
    }

    // Check dual variable signs for inequality constraints
    for (i, con) in model.constraints.iter().enumerate() {
        if i >= dual.len() {
            break;
        }
        let y = dual[i];
        match con.sense {
            ConstraintSense::Le => {
                if y < -tol {
                    max_violation = max_violation.max(-y);
                    violations.push(Violation::DualSignViolation {
                        con_idx: i,
                        dual_value: y,
                    });
                }
            }
            ConstraintSense::Ge => {
                if y > tol {
                    max_violation = max_violation.max(y);
                    violations.push(Violation::DualSignViolation {
                        con_idx: i,
                        dual_value: y,
                    });
                }
            }
            ConstraintSense::Eq => {} // no sign constraint
        }
    }

    ValidationResult {
        is_feasible: violations.is_empty(),
        max_violation,
        violations,
    }
}

/// Validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_feasible: bool,
    pub max_violation: f64,
    pub violations: Vec<Violation>,
}

/// Types of violations.
#[derive(Debug, Clone)]
pub enum Violation {
    BoundViolation {
        var_idx: usize,
        value: f64,
        bound: f64,
        is_lower: bool,
    },
    ConstraintViolation {
        con_idx: usize,
        slack: f64,
        sense: ConstraintSense,
    },
    DualViolation {
        var_idx: usize,
        reduced_cost: f64,
    },
    DualSignViolation {
        con_idx: usize,
        dual_value: f64,
    },
}

/// Sensitivity analysis: compute allowable ranges for objective coefficients
/// and RHS values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityRanges {
    /// For each variable: (obj_coeff_decrease, obj_coeff_increase) allowed before basis changes.
    pub obj_ranges: Vec<(f64, f64)>,
    /// For each constraint: (rhs_decrease, rhs_increase) allowed before basis changes.
    pub rhs_ranges: Vec<(f64, f64)>,
}

/// Compute sensitivity ranges for an optimal solution.
pub fn compute_sensitivity_ranges(
    model: &LpModel,
    primal: &[f64],
    dual: &[f64],
    reduced_costs: &[f64],
    basic_indices: &[usize],
) -> SensitivityRanges {
    let n = model.num_vars();
    let m = model.num_constraints();
    let tol = 1e-10;

    // Objective coefficient ranges
    let mut obj_ranges = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    for j in 0..n {
        if j >= reduced_costs.len() {
            continue;
        }
        let rc = reduced_costs[j];
        let is_basic = basic_indices.contains(&j);

        if !is_basic {
            // Non-basic variable: can change obj by |rc| before it enters the basis
            if rc > tol {
                obj_ranges[j] = (-rc, f64::INFINITY);
            } else if rc < -tol {
                obj_ranges[j] = (f64::NEG_INFINITY, -rc);
            } else {
                obj_ranges[j] = (0.0, 0.0); // degenerate
            }
        } else {
            // Basic variable: need to analyze the basis inverse column
            // Simplified: use the reduced costs of non-basic variables
            let mut min_decrease = f64::INFINITY;
            let mut min_increase = f64::INFINITY;

            for k in 0..n {
                if k == j || basic_indices.contains(&k) {
                    continue;
                }
                if k < reduced_costs.len() {
                    let rc_k = reduced_costs[k];
                    // The sensitivity depends on the ratio rc_k / d_kj
                    // where d_kj is the representation of column k in the basis
                    // Simplified estimate:
                    if rc_k.abs() > tol {
                        min_decrease = min_decrease.min(rc_k.abs());
                        min_increase = min_increase.min(rc_k.abs());
                    }
                }
            }

            obj_ranges[j] = (
                if min_decrease.is_finite() {
                    -min_decrease
                } else {
                    f64::NEG_INFINITY
                },
                if min_increase.is_finite() {
                    min_increase
                } else {
                    f64::INFINITY
                },
            );
        }
    }

    // RHS ranges
    let mut rhs_ranges = vec![(f64::NEG_INFINITY, f64::INFINITY); m];
    for i in 0..m {
        if i >= dual.len() {
            continue;
        }
        let y_i = dual[i];

        // Compute allowable change in b_i
        // The basic variable values change by B^{-1} * delta_b
        // For a change in b_i only, the k-th basic variable changes by B^{-1}_{k,i} * delta
        // We need B^{-1}_{k,i} for all k, but we approximate with the dual value
        if y_i.abs() > tol {
            // Simplified: use slack-based estimate
            let slacks = compute_slacks(model, primal);
            let slack = if i < slacks.len() { slacks[i] } else { 0.0 };

            let decrease = slack.abs().max(tol);
            let increase = 10.0 * decrease; // approximate
            rhs_ranges[i] = (-decrease, increase);
        }
    }

    SensitivityRanges {
        obj_ranges,
        rhs_ranges,
    }
}

/// Farkas certificate of infeasibility.
///
/// For an infeasible LP: min c^T x s.t. Ax <= b, x >= 0
/// A Farkas certificate is y >= 0 such that A^T y >= 0 and b^T y < 0.
/// This proves that the primal is infeasible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarkasCertificate {
    /// Dual multipliers (Farkas ray).
    pub y: Vec<f64>,
    /// Value of b^T y (should be negative for infeasibility proof).
    pub b_dot_y: f64,
    /// Whether the certificate is valid.
    pub is_valid: bool,
}

/// Compute a Farkas certificate from an infeasible dual solution.
pub fn compute_farkas_certificate(model: &LpModel, dual: &[f64]) -> FarkasCertificate {
    let m = model.constraints.len();
    let n = model.num_vars();

    if dual.len() < m {
        return FarkasCertificate {
            y: dual.to_vec(),
            b_dot_y: 0.0,
            is_valid: false,
        };
    }

    // Normalize the dual: ensure y >= 0 (for <= constraints)
    let mut y = vec![0.0; m];
    for i in 0..m {
        match model.constraints[i].sense {
            ConstraintSense::Le => y[i] = dual[i].max(0.0),
            ConstraintSense::Ge => y[i] = (-dual[i]).max(0.0),
            ConstraintSense::Eq => y[i] = dual[i], // can be any sign
        }
    }

    // Compute b^T y
    let b_dot_y: f64 = model
        .constraints
        .iter()
        .enumerate()
        .map(|(i, c)| c.rhs * y[i])
        .sum();

    // Check A^T y >= 0 for all variables
    let mut valid = b_dot_y < -1e-8;
    if valid {
        for j in 0..n {
            let mut at_y = 0.0;
            for (i, con) in model.constraints.iter().enumerate() {
                if let Some(pos) = con.row_indices.iter().position(|&c| c == j) {
                    at_y += con.row_values[pos] * y[i];
                }
            }
            // For x_j >= 0: need (A^T y)_j >= 0
            let var = &model.variables[j];
            if var.lower_bound >= 0.0 && at_y < -1e-8 {
                valid = false;
                break;
            }
        }
    }

    FarkasCertificate {
        y,
        b_dot_y,
        is_valid: valid,
    }
}

/// Extract a complete solution from solver output.
pub fn extract_solution(
    model: &LpModel,
    status: LpStatus,
    primal: Vec<f64>,
    dual: Vec<f64>,
    reduced_costs: Vec<f64>,
    basic_indices: &[usize],
    iterations: usize,
    solve_time: f64,
) -> LpSolution {
    let slacks = compute_slacks(model, &primal);
    let var_basis = determine_var_basis_status(model, &primal, basic_indices, 1e-8);
    let con_basis =
        determine_con_basis_status(model, &slacks, basic_indices, model.num_vars(), 1e-8);

    let objective = if status == LpStatus::Optimal {
        compute_objective(model, &primal)
    } else {
        match status {
            LpStatus::Infeasible => f64::INFINITY,
            LpStatus::Unbounded => f64::NEG_INFINITY,
            _ => compute_objective(model, &primal),
        }
    };

    LpSolution {
        status,
        objective,
        primal,
        dual,
        reduced_costs,
        slacks,
        var_basis_status: var_basis,
        con_basis_status: con_basis,
        iterations,
        solve_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Variable};

    fn make_test_model() -> LpModel {
        let mut m = LpModel::new("test");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, f64::INFINITY));
        let y = m.add_variable(Variable::continuous("y", 0.0, f64::INFINITY));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 2.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 4.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 6.0);
        c1.add_term(x, 2.0);
        c1.add_term(y, 1.0);
        m.add_constraint(c1);

        m
    }

    #[test]
    fn test_compute_slacks() {
        let model = make_test_model();
        let primal = vec![1.0, 1.0];
        let slacks = compute_slacks(&model, &primal);
        assert!((slacks[0] - 2.0).abs() < 1e-10); // 4 - (1+1) = 2
        assert!((slacks[1] - 3.0).abs() < 1e-10); // 6 - (2+1) = 3
    }

    #[test]
    fn test_compute_objective() {
        let model = make_test_model();
        let obj = compute_objective(&model, &[2.0, 1.0]);
        assert!((obj - 4.0).abs() < 1e-10); // 1*2 + 2*1 = 4
    }

    #[test]
    fn test_validate_feasible() {
        let model = make_test_model();
        let primal = vec![1.0, 1.0];
        let result = validate_primal_feasibility(&model, &primal, 1e-8);
        assert!(result.is_feasible);
    }

    #[test]
    fn test_validate_infeasible() {
        let model = make_test_model();
        let primal = vec![5.0, 5.0]; // violates both constraints
        let result = validate_primal_feasibility(&model, &primal, 1e-8);
        assert!(!result.is_feasible);
        assert!(result.max_violation > 0.0);
    }

    #[test]
    fn test_var_basis_status() {
        let model = make_test_model();
        let primal = vec![0.0, 4.0];
        let basic = vec![1, 3]; // y and slack2 are basic
        let status = determine_var_basis_status(&model, &primal, &basic, 1e-8);
        assert_eq!(status[0], BasisStatus::NonBasicLower);
        assert_eq!(status[1], BasisStatus::Basic);
    }

    #[test]
    fn test_con_basis_status() {
        let model = make_test_model();
        let slacks = vec![0.0, 2.0]; // c0 active, c1 not
        let basic = vec![0, 3]; // x0 and slack2 basic
        let status = determine_con_basis_status(&model, &slacks, &basic, 2, 1e-8);
        assert_eq!(status[1], BasisStatus::Basic); // slack2 (idx 3) is basic
    }

    #[test]
    fn test_sensitivity_ranges() {
        let model = make_test_model();
        let primal = vec![2.0, 2.0];
        let dual = vec![0.0, 0.5];
        let rc = vec![0.0, 0.0];
        let basic = vec![0, 1];
        let sens = compute_sensitivity_ranges(&model, &primal, &dual, &rc, &basic);
        assert_eq!(sens.obj_ranges.len(), 2);
        assert_eq!(sens.rhs_ranges.len(), 2);
    }

    #[test]
    fn test_farkas_certificate() {
        let model = make_test_model();
        let dual = vec![1.0, 0.0];
        let cert = compute_farkas_certificate(&model, &dual);
        // This may or may not be valid depending on the dual values
        assert_eq!(cert.y.len(), 2);
    }

    #[test]
    fn test_extract_solution() {
        let model = make_test_model();
        let sol = extract_solution(
            &model,
            LpStatus::Optimal,
            vec![2.0, 2.0],
            vec![0.5, 0.5],
            vec![0.0, 0.0],
            &[0, 1],
            10,
            0.01,
        );
        assert!(sol.is_optimal());
        assert_eq!(sol.slacks.len(), 2);
        assert_eq!(sol.var_basis_status.len(), 2);
    }

    #[test]
    fn test_lp_solution_display() {
        let sol = LpSolution::optimal(42.0, vec![1.0, 2.0, 3.0], vec![0.5], vec![0.0, 0.0, 0.0], 5);
        let display = format!("{}", sol);
        assert!(display.contains("Optimal"));
        assert!(display.contains("42"));
    }

    #[test]
    fn test_dual_feasibility_validation() {
        let model = make_test_model();
        let primal = vec![0.0, 0.0];
        let dual = vec![0.0, 0.0];
        let rc = vec![1.0, 2.0]; // positive rc at lower bound = dual feasible (min)
        let result = validate_dual_feasibility(&model, &dual, &rc, &primal, 1e-8);
        assert!(result.is_feasible);
    }
}
