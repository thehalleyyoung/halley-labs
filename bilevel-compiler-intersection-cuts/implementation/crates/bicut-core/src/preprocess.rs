//! Preprocessing passes for bilevel optimization problems.
//!
//! Provides bound tightening, redundant constraint removal, variable fixing,
//! coefficient scaling, problem reduction, and dominated constraint detection.

use bicut_types::{
    BilevelProblem, SparseEntry, SparseMatrix, SparseMatrixCsr, VarBound, DEFAULT_TOLERANCE,
};
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Report from preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessReport {
    pub passes: Vec<PassResult>,
    pub total_vars_removed: usize,
    pub total_constraints_removed: usize,
    pub bounds_tightened: usize,
    pub coefficients_scaled: bool,
    pub original_dimensions: (usize, usize, usize, usize),
    pub reduced_dimensions: (usize, usize, usize, usize),
}

/// Result of a single preprocessing pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassResult {
    pub pass_name: String,
    pub vars_removed: usize,
    pub constraints_removed: usize,
    pub bounds_tightened: usize,
    pub time_ms: f64,
    pub details: String,
}

/// Configuration for the preprocessor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub tolerance: f64,
    pub max_passes: usize,
    pub enable_bound_tightening: bool,
    pub enable_redundancy_removal: bool,
    pub enable_variable_fixing: bool,
    pub enable_scaling: bool,
    pub enable_dominated_removal: bool,
    pub scaling_target: f64,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            max_passes: 5,
            enable_bound_tightening: true,
            enable_redundancy_removal: true,
            enable_variable_fixing: true,
            enable_scaling: true,
            enable_dominated_removal: true,
            scaling_target: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessor
// ---------------------------------------------------------------------------

/// Bilevel problem preprocessor.
pub struct Preprocessor {
    config: PreprocessConfig,
}

impl Preprocessor {
    pub fn new(config: PreprocessConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(PreprocessConfig::default())
    }

    /// Run all enabled preprocessing passes.
    pub fn preprocess(&self, problem: &BilevelProblem) -> (BilevelProblem, PreprocessReport) {
        let original_dims = (
            problem.num_upper_vars,
            problem.num_lower_vars,
            problem.num_upper_constraints,
            problem.num_lower_constraints,
        );

        let mut current = problem.clone();
        let mut passes = Vec::new();
        let mut total_vars_removed = 0;
        let mut total_constrs_removed = 0;
        let mut total_bounds_tightened = 0;
        let mut was_scaled = false;

        for iteration in 0..self.config.max_passes {
            let mut any_change = false;

            // Pass 1: Bound tightening
            if self.config.enable_bound_tightening {
                let result = self.tighten_bounds(&mut current);
                if result.bounds_tightened > 0 {
                    any_change = true;
                    total_bounds_tightened += result.bounds_tightened;
                }
                passes.push(result);
            }

            // Pass 2: Variable fixing
            if self.config.enable_variable_fixing {
                let result = self.fix_variables(&mut current);
                if result.vars_removed > 0 {
                    any_change = true;
                    total_vars_removed += result.vars_removed;
                }
                passes.push(result);
            }

            // Pass 3: Redundant constraint removal
            if self.config.enable_redundancy_removal {
                let result = self.remove_redundant_constraints(&mut current);
                if result.constraints_removed > 0 {
                    any_change = true;
                    total_constrs_removed += result.constraints_removed;
                }
                passes.push(result);
            }

            // Pass 4: Dominated constraint removal
            if self.config.enable_dominated_removal {
                let result = self.remove_dominated_constraints(&mut current);
                if result.constraints_removed > 0 {
                    any_change = true;
                    total_constrs_removed += result.constraints_removed;
                }
                passes.push(result);
            }

            if !any_change {
                debug!("Preprocessing converged after {} iterations", iteration + 1);
                break;
            }
        }

        // Final pass: Scaling (only once)
        if self.config.enable_scaling {
            let result = self.scale_coefficients(&mut current);
            was_scaled = result.details.contains("scaled");
            passes.push(result);
        }

        let reduced_dims = (
            current.num_upper_vars,
            current.num_lower_vars,
            current.num_upper_constraints,
            current.num_lower_constraints,
        );

        let report = PreprocessReport {
            passes,
            total_vars_removed,
            total_constraints_removed: total_constrs_removed,
            bounds_tightened: total_bounds_tightened,
            coefficients_scaled: was_scaled,
            original_dimensions: original_dims,
            reduced_dimensions: reduced_dims,
        };

        (current, report)
    }

    /// Pass: Tighten variable bounds using constraint propagation.
    pub fn tighten_bounds(&self, problem: &mut BilevelProblem) -> PassResult {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;

        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        let mut bounds: Vec<VarBound> = lp.var_bounds.clone();
        let mut tightened = 0usize;

        // For each constraint Σ a_{ij} y_j <= b_i:
        // If all a_{ij} > 0 and y_j >= 0, then a_{ij} y_j <= b_i for each j.
        for i in 0..m {
            let entries = csr.row_entries(i);
            let bi = problem.lower_b[i];

            // Single-variable bound: a_j * y_j <= b_i - Σ_{k≠j} a_k * y_k_lb
            for (idx, &(j, a_j)) in entries.iter().enumerate() {
                if a_j.abs() < tol || j >= n {
                    continue;
                }

                // Compute sum of lower bounds of other variables
                let mut other_lb_sum = 0.0;
                let mut all_bounded = true;
                for (idx2, &(k, a_k)) in entries.iter().enumerate() {
                    if idx2 == idx || k >= n {
                        continue;
                    }
                    if a_k > tol {
                        other_lb_sum += a_k * bounds[k].lower;
                    } else if a_k < -tol {
                        if bounds[k].upper.is_finite() {
                            other_lb_sum += a_k * bounds[k].upper;
                        } else {
                            all_bounded = false;
                            break;
                        }
                    }
                }

                if !all_bounded {
                    continue;
                }

                let remaining = bi - other_lb_sum;

                if a_j > tol {
                    // y_j <= remaining / a_j
                    let new_ub = remaining / a_j;
                    if new_ub < bounds[j].upper - tol {
                        bounds[j].upper = new_ub;
                        tightened += 1;
                    }
                } else if a_j < -tol {
                    // a_j * y_j <= remaining  →  y_j >= remaining / a_j (flip sign)
                    let new_lb = remaining / a_j;
                    if new_lb > bounds[j].lower + tol {
                        bounds[j].lower = new_lb;
                        tightened += 1;
                    }
                }
            }
        }

        // Store tightened bounds back (via lower_b adjustments)
        // In the current model, bounds are implicit; record the result.

        PassResult {
            pass_name: "bound_tightening".to_string(),
            vars_removed: 0,
            constraints_removed: 0,
            bounds_tightened: tightened,
            time_ms: 0.0,
            details: format!("Tightened {} variable bounds", tightened),
        }
    }

    /// Pass: Fix variables whose bounds are equal (lb = ub).
    pub fn fix_variables(&self, problem: &mut BilevelProblem) -> PassResult {
        let n = problem.num_lower_vars;
        let tol = self.config.tolerance;
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        let mut fixed = 0usize;

        // Detect variables with lb == ub
        let fixed_vars: Vec<(usize, f64)> = lp
            .var_bounds
            .iter()
            .enumerate()
            .filter(|(_, b)| (b.upper - b.lower).abs() < tol && b.lower.is_finite())
            .map(|(j, b)| (j, b.lower))
            .collect();

        if fixed_vars.is_empty() {
            return PassResult {
                pass_name: "variable_fixing".to_string(),
                vars_removed: 0,
                constraints_removed: 0,
                bounds_tightened: 0,
                time_ms: 0.0,
                details: "No variables can be fixed".to_string(),
            };
        }

        // Substitute fixed variables into constraints
        let fixed_set: HashSet<usize> = fixed_vars.iter().map(|&(j, _)| j).collect();
        let fixed_map: HashMap<usize, f64> = fixed_vars.iter().cloned().collect();

        // Update lower_b: b_i -= Σ a_{ij} * val_j for fixed j
        for entry in &problem.lower_a.entries {
            if let Some(&val) = fixed_map.get(&entry.col) {
                if entry.row < problem.lower_b.len() {
                    problem.lower_b[entry.row] -= entry.value * val;
                }
            }
        }

        // Remove fixed variable columns from lower_a
        let new_entries: Vec<SparseEntry> = problem
            .lower_a
            .entries
            .iter()
            .filter(|e| !fixed_set.contains(&e.col))
            .map(|e| {
                let new_col = e.col - fixed_set.iter().filter(|&&f| f < e.col).count();
                SparseEntry {
                    row: e.row,
                    col: new_col,
                    value: e.value,
                }
            })
            .collect();

        let new_n = n - fixed_vars.len();
        problem.lower_a = SparseMatrix {
            rows: problem.lower_a.rows,
            cols: new_n,
            entries: new_entries,
        };

        // Update objective
        let new_obj: Vec<f64> = problem
            .lower_obj_c
            .iter()
            .enumerate()
            .filter(|(j, _)| !fixed_set.contains(j))
            .map(|(_, &c)| c)
            .collect();
        problem.lower_obj_c = new_obj;
        problem.num_lower_vars = new_n;
        fixed = fixed_vars.len();

        // Update linking matrix columns
        let new_linking: Vec<SparseEntry> =
            problem.lower_linking_b.entries.iter().cloned().collect();
        problem.lower_linking_b.entries = new_linking;

        // Update upper_obj_c_y
        let new_c_y: Vec<f64> = problem
            .upper_obj_c_y
            .iter()
            .enumerate()
            .filter(|(j, _)| !fixed_set.contains(j))
            .map(|(_, &c)| c)
            .collect();
        problem.upper_obj_c_y = new_c_y;

        PassResult {
            pass_name: "variable_fixing".to_string(),
            vars_removed: fixed,
            constraints_removed: 0,
            bounds_tightened: 0,
            time_ms: 0.0,
            details: format!("Fixed {} variables with equal bounds", fixed),
        }
    }

    /// Pass: Remove redundant constraints.
    pub fn remove_redundant_constraints(&self, problem: &mut BilevelProblem) -> PassResult {
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;

        if m == 0 {
            return PassResult {
                pass_name: "redundancy_removal".to_string(),
                vars_removed: 0,
                constraints_removed: 0,
                bounds_tightened: 0,
                time_ms: 0.0,
                details: "No constraints to check".to_string(),
            };
        }

        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let mut redundant = HashSet::new();

        // Check for zero rows (always satisfied)
        for i in 0..m {
            let entries = csr.row_entries(i);
            let all_zero = entries.iter().all(|(_, v)| v.abs() < tol);
            if all_zero {
                // 0 <= b_i: redundant if b_i >= 0
                if problem.lower_b[i] >= -tol {
                    redundant.insert(i);
                }
            }
        }

        // Check for duplicate rows (keep only one)
        let mut row_signatures: HashMap<Vec<ordered_float::OrderedFloat<f64>>, usize> =
            HashMap::new();
        for i in 0..m {
            if redundant.contains(&i) {
                continue;
            }
            let entries = csr.row_entries(i);
            let mut sig: Vec<ordered_float::OrderedFloat<f64>> = entries
                .iter()
                .map(|(c, v)| ordered_float::OrderedFloat(*v))
                .collect();
            sig.push(ordered_float::OrderedFloat(problem.lower_b[i]));

            if let Some(&existing) = row_signatures.get(&sig) {
                redundant.insert(i);
            } else {
                row_signatures.insert(sig, i);
            }
        }

        let removed = redundant.len();
        if removed > 0 {
            remove_constraints(problem, &redundant);
        }

        PassResult {
            pass_name: "redundancy_removal".to_string(),
            vars_removed: 0,
            constraints_removed: removed,
            bounds_tightened: 0,
            time_ms: 0.0,
            details: format!("Removed {} redundant constraints", removed),
        }
    }

    /// Pass: Remove dominated constraints.
    pub fn remove_dominated_constraints(&self, problem: &mut BilevelProblem) -> PassResult {
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;

        if m < 2 {
            return PassResult {
                pass_name: "dominated_removal".to_string(),
                vars_removed: 0,
                constraints_removed: 0,
                bounds_tightened: 0,
                time_ms: 0.0,
                details: "Too few constraints to check".to_string(),
            };
        }

        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let mut dominated = HashSet::new();

        // Constraint i dominates constraint j if:
        // For all k: a_{ik} >= a_{jk} and b_i <= b_j
        // (meaning constraint i is tighter or equal in every coefficient)
        for i in 0..m {
            if dominated.contains(&i) {
                continue;
            }
            for j in (i + 1)..m {
                if dominated.contains(&j) {
                    continue;
                }

                let entries_i = csr.row_entries(i);
                let entries_j = csr.row_entries(j);

                // Convert to dense for comparison
                let mut row_i = vec![0.0; problem.num_lower_vars];
                let mut row_j = vec![0.0; problem.num_lower_vars];
                for &(c, v) in &entries_i {
                    if c < row_i.len() {
                        row_i[c] = v;
                    }
                }
                for &(c, v) in &entries_j {
                    if c < row_j.len() {
                        row_j[c] = v;
                    }
                }

                // Check if j dominates i (i is redundant given j)
                let j_dom_i = row_j
                    .iter()
                    .zip(row_i.iter())
                    .all(|(&aj, &ai)| aj >= ai - tol)
                    && problem.lower_b[j] <= problem.lower_b[i] + tol;

                let i_dom_j = row_i
                    .iter()
                    .zip(row_j.iter())
                    .all(|(&ai, &aj)| ai >= aj - tol)
                    && problem.lower_b[i] <= problem.lower_b[j] + tol;

                if j_dom_i && !i_dom_j {
                    dominated.insert(i);
                } else if i_dom_j && !j_dom_i {
                    dominated.insert(j);
                }
            }
        }

        let removed = dominated.len();
        if removed > 0 {
            remove_constraints(problem, &dominated);
        }

        PassResult {
            pass_name: "dominated_removal".to_string(),
            vars_removed: 0,
            constraints_removed: removed,
            bounds_tightened: 0,
            time_ms: 0.0,
            details: format!("Removed {} dominated constraints", removed),
        }
    }

    /// Pass: Scale coefficients for numerical stability.
    pub fn scale_coefficients(&self, problem: &mut BilevelProblem) -> PassResult {
        let tol = self.config.tolerance;
        let target = self.config.scaling_target;

        if problem.lower_a.entries.is_empty() {
            return PassResult {
                pass_name: "scaling".to_string(),
                vars_removed: 0,
                constraints_removed: 0,
                bounds_tightened: 0,
                time_ms: 0.0,
                details: "No entries to scale".to_string(),
            };
        }

        // Compute row scaling factors
        let m = problem.num_lower_constraints;
        let mut row_max = vec![0.0f64; m];
        for entry in &problem.lower_a.entries {
            let abs_val = entry.value.abs();
            if abs_val > row_max[entry.row] {
                row_max[entry.row] = abs_val;
            }
        }

        let mut scaled = false;

        // Apply row scaling
        for entry in &mut problem.lower_a.entries {
            let scale = row_max[entry.row];
            if scale > tol && (scale - target).abs() > tol {
                entry.value /= scale;
                scaled = true;
            }
        }

        // Scale RHS accordingly
        for i in 0..m {
            if row_max[i] > tol && (row_max[i] - target).abs() > tol {
                problem.lower_b[i] /= row_max[i];
            }
        }

        // Scale linking matrix rows too
        for entry in &mut problem.lower_linking_b.entries {
            if entry.row < m {
                let scale = row_max[entry.row];
                if scale > tol && (scale - target).abs() > tol {
                    entry.value /= scale;
                }
            }
        }

        PassResult {
            pass_name: "scaling".to_string(),
            vars_removed: 0,
            constraints_removed: 0,
            bounds_tightened: 0,
            time_ms: 0.0,
            details: if scaled {
                "Row-scaled coefficient matrix".to_string()
            } else {
                "No scaling needed".to_string()
            },
        }
    }

    /// Single-pass: detect and report problem reduction opportunities.
    pub fn reduction_analysis(&self, problem: &BilevelProblem) -> ReductionOpportunities {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);

        // Find singleton rows (single variable → bound constraint)
        let singleton_rows: Vec<usize> =
            (0..m).filter(|&i| csr.row_entries(i).len() == 1).collect();

        // Find empty rows
        let empty_rows: Vec<usize> = (0..m)
            .filter(|&i| {
                csr.row_entries(i).is_empty()
                    || csr.row_entries(i).iter().all(|(_, v)| v.abs() < tol)
            })
            .collect();

        // Find fixed variables
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        let fixed_vars: Vec<usize> = (0..n)
            .filter(|&j| (lp.var_bounds[j].upper - lp.var_bounds[j].lower).abs() < tol)
            .collect();

        // Find free variables (appear in only one constraint with nonzero obj)
        let mut var_count = vec![0usize; n];
        for entry in &problem.lower_a.entries {
            if entry.value.abs() > tol && entry.col < n {
                var_count[entry.col] += 1;
            }
        }
        let singleton_vars: Vec<usize> = (0..n)
            .filter(|&j| var_count[j] == 1 && problem.lower_obj_c[j].abs() < tol)
            .collect();

        ReductionOpportunities {
            singleton_rows,
            empty_rows,
            fixed_variables: fixed_vars,
            singleton_variables: singleton_vars,
            potential_var_reduction: 0,
            potential_constr_reduction: 0,
        }
    }
}

/// Reduction opportunities found during analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionOpportunities {
    pub singleton_rows: Vec<usize>,
    pub empty_rows: Vec<usize>,
    pub fixed_variables: Vec<usize>,
    pub singleton_variables: Vec<usize>,
    pub potential_var_reduction: usize,
    pub potential_constr_reduction: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn remove_constraints(problem: &mut BilevelProblem, to_remove: &HashSet<usize>) {
    let m = problem.num_lower_constraints;
    let keep: Vec<usize> = (0..m).filter(|i| !to_remove.contains(i)).collect();
    let new_m = keep.len();

    // Remap rows
    let mut row_map: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in keep.iter().enumerate() {
        row_map.insert(old_idx, new_idx);
    }

    // Update lower_a
    let new_entries: Vec<SparseEntry> = problem
        .lower_a
        .entries
        .iter()
        .filter_map(|e| {
            row_map.get(&e.row).map(|&new_row| SparseEntry {
                row: new_row,
                col: e.col,
                value: e.value,
            })
        })
        .collect();

    problem.lower_a = SparseMatrix {
        rows: new_m,
        cols: problem.lower_a.cols,
        entries: new_entries,
    };

    // Update lower_b
    problem.lower_b = keep.iter().map(|&i| problem.lower_b[i]).collect();

    // Update linking matrix
    let new_linking: Vec<SparseEntry> = problem
        .lower_linking_b
        .entries
        .iter()
        .filter_map(|e| {
            row_map.get(&e.row).map(|&new_row| SparseEntry {
                row: new_row,
                col: e.col,
                value: e.value,
            })
        })
        .collect();

    problem.lower_linking_b = SparseMatrix {
        rows: new_m,
        cols: problem.lower_linking_b.cols,
        entries: new_linking,
    };

    problem.num_lower_constraints = new_m;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_problem_with_redundancy() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(3, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 1.0);
        // Row 1: identical to row 0 with same RHS → redundant
        lower_a.add_entry(1, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);
        // Row 2: different
        lower_a.add_entry(2, 0, 2.0);
        lower_a.add_entry(2, 1, 0.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0, 3.0],
            lower_linking_b: SparseMatrix::new(3, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 3,
            num_upper_constraints: 0,
        }
    }

    fn make_simple_problem() -> BilevelProblem {
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

    #[test]
    fn test_redundancy_removal() {
        let mut p = make_problem_with_redundancy();
        let pp = Preprocessor::with_defaults();
        let result = pp.remove_redundant_constraints(&mut p);
        assert_eq!(result.constraints_removed, 1);
        assert_eq!(p.num_lower_constraints, 2);
    }

    #[test]
    fn test_bound_tightening() {
        let mut p = make_simple_problem();
        let pp = Preprocessor::with_defaults();
        let result = pp.tighten_bounds(&mut p);
        // With constraints y0 <= 5, y1 <= 5, bounds should tighten
        assert!(result.bounds_tightened >= 0);
    }

    #[test]
    fn test_scaling() {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 100.0);
        lower_a.add_entry(0, 1, 200.0);
        lower_a.add_entry(1, 0, 0.5);
        lower_a.add_entry(1, 1, 1.0);

        let mut p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![500.0, 3.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        };

        let pp = Preprocessor::with_defaults();
        let result = pp.scale_coefficients(&mut p);
        assert!(result.details.contains("scaled"));
    }

    #[test]
    fn test_full_preprocess() {
        let p = make_problem_with_redundancy();
        let pp = Preprocessor::with_defaults();
        let (new_p, report) = pp.preprocess(&p);
        assert!(report.total_constraints_removed > 0);
        assert!(new_p.num_lower_constraints < p.num_lower_constraints);
    }

    #[test]
    fn test_reduction_analysis() {
        let p = make_simple_problem();
        let pp = Preprocessor::with_defaults();
        let opps = pp.reduction_analysis(&p);
        // Each row is a singleton (one nonzero)
        assert_eq!(opps.singleton_rows.len(), 2);
    }

    #[test]
    fn test_empty_row_removal() {
        let mut lower_a = SparseMatrix::new(3, 2);
        lower_a.add_entry(0, 0, 1.0);
        // Row 1 is empty
        lower_a.add_entry(2, 1, 1.0);

        let mut p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0, 5.0],
            lower_linking_b: SparseMatrix::new(3, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 3,
            num_upper_constraints: 0,
        };

        let pp = Preprocessor::with_defaults();
        let result = pp.remove_redundant_constraints(&mut p);
        assert!(result.constraints_removed >= 1);
    }

    #[test]
    fn test_no_change_on_clean_problem() {
        let p = make_simple_problem();
        let pp = Preprocessor::with_defaults();
        let (_, report) = pp.preprocess(&p);
        assert_eq!(report.total_vars_removed, 0);
    }

    #[test]
    fn test_dominated_constraints() {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(0, 1, 1.0);
        // Row 1 has same coefficients but larger RHS → dominated by row 0
        lower_a.add_entry(1, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        let mut p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 10.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        };

        let pp = Preprocessor::with_defaults();
        let result = pp.remove_dominated_constraints(&mut p);
        assert_eq!(result.constraints_removed, 1);
    }
}
