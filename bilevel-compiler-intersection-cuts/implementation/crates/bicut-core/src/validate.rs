//! Problem validation for bilevel optimization.
//!
//! Checks well-formedness of bilevel programs: dimension consistency,
//! constraint feasibility, type correctness, and structural validity.

use bicut_types::{BilevelProblem, SparseMatrix, SparseMatrixCsr, DEFAULT_TOLERANCE};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// A single validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub code: String,
    pub message: String,
    pub location: String,
}

impl ValidationIssue {
    pub fn error(
        code: impl Into<String>,
        message: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            severity: IssueSeverity::Error,
            code: code.into(),
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn warning(
        code: impl Into<String>,
        message: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            code: code.into(),
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn info(
        code: impl Into<String>,
        message: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            severity: IssueSeverity::Info,
            code: code.into(),
            message: message.into(),
            location: location.into(),
        }
    }
}

/// Complete validation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub num_errors: usize,
    pub num_warnings: usize,
    pub num_info: usize,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            is_valid: true,
            issues: Vec::new(),
            num_errors: 0,
            num_warnings: 0,
            num_info: 0,
        }
    }

    fn add(&mut self, issue: ValidationIssue) {
        match issue.severity {
            IssueSeverity::Error => {
                self.num_errors += 1;
                self.is_valid = false;
            }
            IssueSeverity::Warning => {
                self.num_warnings += 1;
            }
            IssueSeverity::Info => {
                self.num_info += 1;
            }
        }
        self.issues.push(issue);
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Configuration for validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    pub tolerance: f64,
    pub check_feasibility: bool,
    pub check_numerical: bool,
    pub max_coefficient: f64,
    pub warn_large_coefficient: f64,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            check_feasibility: true,
            check_numerical: true,
            max_coefficient: 1e15,
            warn_large_coefficient: 1e8,
        }
    }
}

/// Problem validator.
pub struct ProblemValidator {
    config: ValidatorConfig,
}

impl ProblemValidator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Run all validation checks.
    pub fn validate(&self, problem: &BilevelProblem) -> ValidationReport {
        let mut report = ValidationReport::new();

        self.check_dimensions(&mut report, problem);
        self.check_matrix_consistency(&mut report, problem);
        self.check_objective_consistency(&mut report, problem);
        self.check_bounds_consistency(&mut report, problem);
        self.check_sparse_matrix_validity(&mut report, problem);

        if self.config.check_numerical {
            self.check_numerical_issues(&mut report, problem);
        }

        if self.config.check_feasibility {
            self.check_trivial_infeasibility(&mut report, problem);
        }

        self.check_structural_issues(&mut report, problem);

        debug!(
            "Validation complete: {} errors, {} warnings",
            report.num_errors, report.num_warnings
        );

        report
    }

    /// Check that all declared dimensions are consistent.
    fn check_dimensions(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        // Lower objective dimension
        if problem.lower_obj_c.len() != problem.num_lower_vars {
            report.add(ValidationIssue::error(
                "DIM001",
                format!(
                    "Lower objective vector length ({}) != num_lower_vars ({})",
                    problem.lower_obj_c.len(),
                    problem.num_lower_vars
                ),
                "lower_obj_c",
            ));
        }

        // Lower RHS dimension
        if problem.lower_b.len() != problem.num_lower_constraints {
            report.add(ValidationIssue::error(
                "DIM002",
                format!(
                    "Lower RHS vector length ({}) != num_lower_constraints ({})",
                    problem.lower_b.len(),
                    problem.num_lower_constraints
                ),
                "lower_b",
            ));
        }

        // Upper objective x dimension
        if problem.upper_obj_c_x.len() != problem.num_upper_vars {
            report.add(ValidationIssue::error(
                "DIM003",
                format!(
                    "Upper objective x vector length ({}) != num_upper_vars ({})",
                    problem.upper_obj_c_x.len(),
                    problem.num_upper_vars
                ),
                "upper_obj_c_x",
            ));
        }

        // Upper objective y dimension
        if problem.upper_obj_c_y.len() != problem.num_lower_vars {
            report.add(ValidationIssue::error(
                "DIM004",
                format!(
                    "Upper objective y vector length ({}) != num_lower_vars ({})",
                    problem.upper_obj_c_y.len(),
                    problem.num_lower_vars
                ),
                "upper_obj_c_y",
            ));
        }

        // Upper RHS dimension
        if problem.upper_constraints_b.len() != problem.num_upper_constraints {
            report.add(ValidationIssue::error(
                "DIM005",
                format!(
                    "Upper RHS vector length ({}) != num_upper_constraints ({})",
                    problem.upper_constraints_b.len(),
                    problem.num_upper_constraints
                ),
                "upper_constraints_b",
            ));
        }

        // Non-negative dimensions
        if problem.num_upper_vars == 0 && problem.num_lower_vars == 0 {
            report.add(ValidationIssue::error(
                "DIM006",
                "Problem has zero variables in both levels".to_string(),
                "dimensions",
            ));
        }

        if problem.num_lower_vars == 0 {
            report.add(ValidationIssue::warning(
                "DIM007",
                "Lower level has zero variables".to_string(),
                "num_lower_vars",
            ));
        }
    }

    /// Check matrix dimension consistency.
    fn check_matrix_consistency(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        // Lower A matrix
        if problem.lower_a.rows != problem.num_lower_constraints {
            report.add(ValidationIssue::error(
                "MAT001",
                format!(
                    "lower_a.rows ({}) != num_lower_constraints ({})",
                    problem.lower_a.rows, problem.num_lower_constraints
                ),
                "lower_a",
            ));
        }
        if problem.lower_a.cols != problem.num_lower_vars {
            report.add(ValidationIssue::error(
                "MAT002",
                format!(
                    "lower_a.cols ({}) != num_lower_vars ({})",
                    problem.lower_a.cols, problem.num_lower_vars
                ),
                "lower_a",
            ));
        }

        // Linking matrix
        if problem.lower_linking_b.rows != problem.num_lower_constraints
            && !problem.lower_linking_b.entries.is_empty()
        {
            report.add(ValidationIssue::error(
                "MAT003",
                format!(
                    "lower_linking_b.rows ({}) != num_lower_constraints ({})",
                    problem.lower_linking_b.rows, problem.num_lower_constraints
                ),
                "lower_linking_b",
            ));
        }
        if problem.lower_linking_b.cols != problem.num_upper_vars
            && !problem.lower_linking_b.entries.is_empty()
        {
            report.add(ValidationIssue::warning(
                "MAT004",
                format!(
                    "lower_linking_b.cols ({}) != num_upper_vars ({})",
                    problem.lower_linking_b.cols, problem.num_upper_vars
                ),
                "lower_linking_b",
            ));
        }

        // Upper constraints matrix
        if problem.upper_constraints_a.rows != problem.num_upper_constraints {
            report.add(ValidationIssue::error(
                "MAT005",
                format!(
                    "upper_constraints_a.rows ({}) != num_upper_constraints ({})",
                    problem.upper_constraints_a.rows, problem.num_upper_constraints
                ),
                "upper_constraints_a",
            ));
        }
    }

    /// Check objective vector consistency.
    fn check_objective_consistency(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        // Check for all-zero objectives
        let lower_all_zero = problem
            .lower_obj_c
            .iter()
            .all(|&c| c.abs() < self.config.tolerance);
        if lower_all_zero && problem.num_lower_vars > 0 {
            report.add(ValidationIssue::warning(
                "OBJ001",
                "Lower-level objective is all zeros; any feasible point is optimal".to_string(),
                "lower_obj_c",
            ));
        }

        let upper_all_zero = problem
            .upper_obj_c_x
            .iter()
            .all(|&c| c.abs() < self.config.tolerance)
            && problem
                .upper_obj_c_y
                .iter()
                .all(|&c| c.abs() < self.config.tolerance);
        if upper_all_zero {
            report.add(ValidationIssue::warning(
                "OBJ002",
                "Upper-level objective is all zeros; any bilevel-feasible point is optimal"
                    .to_string(),
                "upper_obj",
            ));
        }
    }

    /// Check variable bounds for consistency.
    fn check_bounds_consistency(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        for (j, bound) in lp.var_bounds.iter().enumerate() {
            if bound.lower > bound.upper + self.config.tolerance {
                report.add(ValidationIssue::error(
                    "BND001",
                    format!(
                        "Variable {} has infeasible bounds: lb={} > ub={}",
                        j, bound.lower, bound.upper
                    ),
                    format!("var_bounds[{}]", j),
                ));
            }
        }
    }

    /// Validate sparse matrix entries.
    fn check_sparse_matrix_validity(
        &self,
        report: &mut ValidationReport,
        problem: &BilevelProblem,
    ) {
        check_sparse_entries(
            report,
            &problem.lower_a,
            "lower_a",
            problem.num_lower_constraints,
            problem.num_lower_vars,
        );
        check_sparse_entries(
            report,
            &problem.lower_linking_b,
            "lower_linking_b",
            problem.lower_linking_b.rows,
            problem.lower_linking_b.cols,
        );
        check_sparse_entries(
            report,
            &problem.upper_constraints_a,
            "upper_constraints_a",
            problem.upper_constraints_a.rows,
            problem.upper_constraints_a.cols,
        );
    }

    /// Check for numerical issues (extreme coefficients, near-zeros, etc.).
    fn check_numerical_issues(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        let max_coeff = self.config.max_coefficient;
        let warn_coeff = self.config.warn_large_coefficient;
        let tol = self.config.tolerance;

        // Check lower_a
        for entry in &problem.lower_a.entries {
            if entry.value.abs() > max_coeff {
                report.add(ValidationIssue::error(
                    "NUM001",
                    format!(
                        "Extreme coefficient {:.2e} at ({}, {})",
                        entry.value, entry.row, entry.col
                    ),
                    "lower_a",
                ));
            } else if entry.value.abs() > warn_coeff {
                report.add(ValidationIssue::warning(
                    "NUM002",
                    format!(
                        "Large coefficient {:.2e} at ({}, {})",
                        entry.value, entry.row, entry.col
                    ),
                    "lower_a",
                ));
            }

            if entry.value.abs() > 0.0 && entry.value.abs() < tol * 100.0 {
                report.add(ValidationIssue::warning(
                    "NUM003",
                    format!(
                        "Near-zero coefficient {:.2e} at ({}, {})",
                        entry.value, entry.row, entry.col
                    ),
                    "lower_a",
                ));
            }

            if entry.value.is_nan() || entry.value.is_infinite() {
                report.add(ValidationIssue::error(
                    "NUM004",
                    format!("NaN/Inf coefficient at ({}, {})", entry.row, entry.col),
                    "lower_a",
                ));
            }
        }

        // Check RHS values
        for (i, &b) in problem.lower_b.iter().enumerate() {
            if b.is_nan() || b.is_infinite() {
                report.add(ValidationIssue::error(
                    "NUM005",
                    format!("NaN/Inf in lower RHS at index {}", i),
                    "lower_b",
                ));
            }
        }

        // Check objective values
        for (j, &c) in problem.lower_obj_c.iter().enumerate() {
            if c.is_nan() || c.is_infinite() {
                report.add(ValidationIssue::error(
                    "NUM006",
                    format!("NaN/Inf in lower objective at index {}", j),
                    "lower_obj_c",
                ));
            }
        }

        // Coefficient range warning
        let abs_vals: Vec<f64> = problem
            .lower_a
            .entries
            .iter()
            .map(|e| e.value.abs())
            .filter(|&v| v > tol)
            .collect();
        if abs_vals.len() >= 2 {
            let min_v = abs_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_v = abs_vals.iter().cloned().fold(0.0f64, f64::max);
            if min_v > tol {
                let range = max_v / min_v;
                if range > 1e6 {
                    report.add(ValidationIssue::warning(
                        "NUM007",
                        format!("Large coefficient range: {:.2e}; consider scaling", range),
                        "lower_a",
                    ));
                }
            }
        }
    }

    /// Check for trivially infeasible problems.
    fn check_trivial_infeasibility(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        let tol = self.config.tolerance;

        // Check if any lower-level constraint has an empty row but negative RHS
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        for i in 0..problem.num_lower_constraints {
            let entries = csr.row_entries(i);
            let all_nonneg = entries.iter().all(|(_, v)| *v >= -tol);
            let all_nonpos = entries.iter().all(|(_, v)| *v <= tol);

            // If all coefficients are non-negative and RHS < 0, then for y >= 0:
            // sum(a_j * y_j) >= 0 > b_i → infeasible
            if all_nonneg && !entries.is_empty() && problem.lower_b[i] < -tol {
                report.add(ValidationIssue::warning(
                    "FEAS001",
                    format!("Constraint {} may be infeasible: all non-negative coefficients but b={:.4}",
                        i, problem.lower_b[i]),
                    format!("lower_constraint[{}]", i),
                ));
            }

            // Empty row with negative RHS
            if entries.is_empty() && problem.lower_b[i] < -tol {
                report.add(ValidationIssue::error(
                    "FEAS002",
                    format!(
                        "Constraint {} is infeasible: 0 <= {:.4} (empty row, negative RHS)",
                        i, problem.lower_b[i]
                    ),
                    format!("lower_constraint[{}]", i),
                ));
            }
        }
    }

    /// Check for structural issues.
    fn check_structural_issues(&self, report: &mut ValidationReport, problem: &BilevelProblem) {
        // Check for unused variables (not in any constraint or objective)
        let mut used_vars: HashSet<usize> = HashSet::new();
        for entry in &problem.lower_a.entries {
            if entry.value.abs() > self.config.tolerance {
                used_vars.insert(entry.col);
            }
        }
        for (j, &c) in problem.lower_obj_c.iter().enumerate() {
            if c.abs() > self.config.tolerance {
                used_vars.insert(j);
            }
        }

        let n = problem.num_lower_vars;
        let unused: Vec<usize> = (0..n).filter(|j| !used_vars.contains(j)).collect();
        if !unused.is_empty() {
            report.add(ValidationIssue::info(
                "STR001",
                format!(
                    "{} unused lower-level variables: {:?}",
                    unused.len(),
                    &unused[..unused.len().min(5)]
                ),
                "lower_vars",
            ));
        }

        // Check for empty constraints
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let empty_rows: Vec<usize> = (0..problem.num_lower_constraints)
            .filter(|&i| csr.row_entries(i).is_empty())
            .collect();
        if !empty_rows.is_empty() {
            report.add(ValidationIssue::warning(
                "STR002",
                format!(
                    "{} empty constraint rows: {:?}",
                    empty_rows.len(),
                    &empty_rows[..empty_rows.len().min(5)]
                ),
                "lower_a",
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_sparse_entries(
    report: &mut ValidationReport,
    sm: &SparseMatrix,
    name: &str,
    max_rows: usize,
    max_cols: usize,
) {
    for (idx, entry) in sm.entries.iter().enumerate() {
        if entry.row >= max_rows {
            report.add(ValidationIssue::error(
                "SPR001",
                format!(
                    "Entry {} has row {} >= declared rows {}",
                    idx, entry.row, max_rows
                ),
                name.to_string(),
            ));
        }
        if entry.col >= max_cols {
            report.add(ValidationIssue::error(
                "SPR002",
                format!(
                    "Entry {} has col {} >= declared cols {}",
                    idx, entry.col, max_cols
                ),
                name.to_string(),
            ));
        }
    }
}

/// Quick validation: returns Ok(()) or the first error.
pub fn quick_validate(problem: &BilevelProblem) -> Result<(), String> {
    if problem.lower_obj_c.len() != problem.num_lower_vars {
        return Err(format!(
            "Lower objective dimension mismatch: {} vs {}",
            problem.lower_obj_c.len(),
            problem.num_lower_vars
        ));
    }
    if problem.lower_b.len() != problem.num_lower_constraints {
        return Err(format!(
            "Lower RHS dimension mismatch: {} vs {}",
            problem.lower_b.len(),
            problem.num_lower_constraints
        ));
    }
    if problem.lower_a.rows != problem.num_lower_constraints {
        return Err("Lower constraint matrix row mismatch".to_string());
    }
    if problem.lower_a.cols != problem.num_lower_vars {
        return Err("Lower constraint matrix col mismatch".to_string());
    }
    if problem.upper_obj_c_x.len() != problem.num_upper_vars {
        return Err(format!(
            "Upper objective x dimension mismatch: {} vs {}",
            problem.upper_obj_c_x.len(),
            problem.num_upper_vars
        ));
    }
    if problem.upper_obj_c_y.len() != problem.num_lower_vars {
        return Err(format!(
            "Upper objective y dimension mismatch: {} vs {}",
            problem.upper_obj_c_y.len(),
            problem.num_lower_vars
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_valid_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    fn make_invalid_dimensions() -> BilevelProblem {
        let lower_a = SparseMatrix::new(2, 2);
        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],         // Wrong: should be 2
            lower_obj_c: vec![1.0, 1.0, 1.0], // Wrong: should be 2
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_valid_problem_passes() {
        let p = make_valid_problem();
        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        assert!(report.is_valid);
        assert_eq!(report.num_errors, 0);
    }

    #[test]
    fn test_dimension_mismatch_detected() {
        let p = make_invalid_dimensions();
        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        assert!(!report.is_valid);
        assert!(report.num_errors >= 2);
    }

    #[test]
    fn test_quick_validate_valid() {
        let p = make_valid_problem();
        assert!(quick_validate(&p).is_ok());
    }

    #[test]
    fn test_quick_validate_invalid() {
        let p = make_invalid_dimensions();
        assert!(quick_validate(&p).is_err());
    }

    #[test]
    fn test_nan_detection() {
        let mut p = make_valid_problem();
        p.lower_a.entries.push(bicut_types::SparseEntry {
            row: 0,
            col: 0,
            value: f64::NAN,
        });
        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        let has_nan_error = report.issues.iter().any(|i| i.code == "NUM004");
        assert!(has_nan_error);
    }

    #[test]
    fn test_infeasible_empty_row() {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        // Row 1 is empty

        let p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, -1.0], // Negative RHS with empty row
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        };

        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        let has_feas_error = report.issues.iter().any(|i| i.code == "FEAS002");
        assert!(has_feas_error);
    }

    #[test]
    fn test_large_coefficient_warning() {
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(0, 0, 1e10);

        let p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix::new(1, 1),
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        };

        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        let has_num_warn = report.issues.iter().any(|i| i.code == "NUM002");
        assert!(has_num_warn);
    }

    #[test]
    fn test_out_of_bounds_entry() {
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(5, 0, 1.0); // row 5 >= rows=1

        let p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![5.0],
            lower_linking_b: SparseMatrix::new(1, 1),
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        };

        let v = ProblemValidator::with_defaults();
        let report = v.validate(&p);
        let has_spr = report.issues.iter().any(|i| i.code == "SPR001");
        assert!(has_spr);
    }

    #[test]
    fn test_issue_severity() {
        let e = ValidationIssue::error("E1", "test", "loc");
        let w = ValidationIssue::warning("W1", "test", "loc");
        let i = ValidationIssue::info("I1", "test", "loc");
        assert_eq!(e.severity, IssueSeverity::Error);
        assert_eq!(w.severity, IssueSeverity::Warning);
        assert_eq!(i.severity, IssueSeverity::Info);
    }
}
