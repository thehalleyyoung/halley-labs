use log::{debug, info};
use serde::{Deserialize, Serialize};

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresolveConfig {
    pub max_passes: usize,
    pub remove_fixed_vars: bool,
    pub bound_tightening: bool,
    pub singleton_rows: bool,
    pub singleton_columns: bool,
    pub dominated_columns: bool,
    pub duplicate_detection: bool,
    pub coefficient_tightening: bool,
}

impl Default for PresolveConfig {
    fn default() -> Self {
        Self {
            max_passes: 10,
            remove_fixed_vars: true,
            bound_tightening: true,
            singleton_rows: true,
            singleton_columns: true,
            dominated_columns: true,
            duplicate_detection: true,
            coefficient_tightening: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Transformations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresolveTransformation {
    FixedVariable {
        var: usize,
        value: f64,
    },
    SingletonRow {
        constraint: usize,
        var: usize,
        value: f64,
    },
    SingletonColumn {
        var: usize,
        lb: f64,
        ub: f64,
    },
    BoundUpdate {
        var: usize,
        old_lb: f64,
        old_ub: f64,
        new_lb: f64,
        new_ub: f64,
    },
    RedundantConstraint {
        constraint: usize,
    },
    ForcingConstraint {
        constraint: usize,
        fixed_vars: Vec<(usize, f64)>,
    },
    DuplicateRow {
        kept: usize,
        removed: usize,
    },
    DuplicateColumn {
        kept: usize,
        removed: usize,
        multiplier: f64,
    },
    FreeColumnSubstitution {
        var: usize,
        constraint: usize,
    },
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PresolveResult {
    pub reduced_problem: LpProblem,
    pub transformations: Vec<PresolveTransformation>,
    pub removed_vars: usize,
    pub removed_constraints: usize,
    pub original_to_reduced_var: Vec<Option<usize>>,
    pub original_to_reduced_con: Vec<Option<usize>>,
}

// ---------------------------------------------------------------------------
// Presolve engine
// ---------------------------------------------------------------------------

pub struct LpPresolve {
    config: PresolveConfig,
}

impl LpPresolve {
    pub fn new(config: PresolveConfig) -> Self {
        Self { config }
    }

    /// Run presolve passes until fixpoint or max passes exhausted.
    pub fn presolve(&self, problem: &LpProblem) -> OptResult<PresolveResult> {
        problem.validate()?;

        let n = problem.num_vars;
        let m = problem.num_constraints;

        let mut transformations: Vec<PresolveTransformation> = Vec::new();
        let mut lb = problem.lower_bounds.clone();
        let mut ub = problem.upper_bounds.clone();
        let obj = problem.obj_coeffs.clone();
        let mut rhs = problem.rhs.clone();
        let ctypes = problem.constraint_types.clone();

        // Build dense row storage for easier manipulation
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(m);
        for i in 0..m {
            let rs = problem.row_starts[i];
            let re = problem.row_starts[i + 1];
            let row: Vec<(usize, f64)> = (rs..re)
                .map(|k| (problem.col_indices[k], problem.values[k]))
                .collect();
            rows.push(row);
        }

        let mut var_removed = vec![false; n];
        let mut con_removed = vec![false; m];
        let mut var_fixed_value = vec![0.0f64; n];

        let mut total_changes = 0usize;

        for pass in 0..self.config.max_passes {
            let changes_before = total_changes;

            // --- Remove fixed variables ---
            if self.config.remove_fixed_vars {
                for j in 0..n {
                    if var_removed[j] {
                        continue;
                    }
                    if (ub[j] - lb[j]).abs() < 1e-10 {
                        let val = lb[j];
                        transformations.push(PresolveTransformation::FixedVariable {
                            var: j,
                            value: val,
                        });
                        var_fixed_value[j] = val;
                        var_removed[j] = true;
                        total_changes += 1;

                        // Update rows and rhs
                        for i in 0..m {
                            if con_removed[i] {
                                continue;
                            }
                            let mut new_row = Vec::new();
                            for &(col, coeff) in &rows[i] {
                                if col == j {
                                    rhs[i] -= coeff * val;
                                } else {
                                    new_row.push((col, coeff));
                                }
                            }
                            rows[i] = new_row;
                        }
                    }
                }
            }

            // --- Singleton rows ---
            if self.config.singleton_rows {
                for i in 0..m {
                    if con_removed[i] {
                        continue;
                    }
                    let active: Vec<(usize, f64)> = rows[i]
                        .iter()
                        .filter(|(c, _)| !var_removed[*c])
                        .cloned()
                        .collect();
                    if active.len() == 1 {
                        let (j, coeff) = active[0];
                        if coeff.abs() < 1e-15 {
                            continue;
                        }
                        let val = rhs[i] / coeff;

                        // Check feasibility with bounds
                        match ctypes[i] {
                            ConstraintType::Eq => {
                                if val < lb[j] - 1e-8 || val > ub[j] + 1e-8 {
                                    return Err(OptError::Infeasible {
                                        reason: format!(
                                            "Singleton row {} fixes var {} to {:.6} outside [{:.6}, {:.6}]",
                                            i, j, val, lb[j], ub[j]
                                        ),
                                    });
                                }
                                let clamped = val.max(lb[j]).min(ub[j]);
                                transformations.push(PresolveTransformation::SingletonRow {
                                    constraint: i,
                                    var: j,
                                    value: clamped,
                                });
                                var_fixed_value[j] = clamped;
                                var_removed[j] = true;
                                con_removed[i] = true;
                                total_changes += 1;

                                // Substitute into other rows
                                for ii in 0..m {
                                    if con_removed[ii] || ii == i {
                                        continue;
                                    }
                                    let mut new_row = Vec::new();
                                    for &(col, cf) in &rows[ii] {
                                        if col == j {
                                            rhs[ii] -= cf * clamped;
                                        } else {
                                            new_row.push((col, cf));
                                        }
                                    }
                                    rows[ii] = new_row;
                                }
                            }
                            ConstraintType::Le => {
                                // a*x <= b, so x <= b/a (if a>0) or x >= b/a (if a<0)
                                if coeff > 0.0 {
                                    let new_ub = val;
                                    if new_ub < ub[j] {
                                        let old_ub = ub[j];
                                        ub[j] = new_ub;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: j,
                                                old_lb: lb[j],
                                                old_ub,
                                                new_lb: lb[j],
                                                new_ub,
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                } else {
                                    let new_lb = val;
                                    if new_lb > lb[j] {
                                        let old_lb = lb[j];
                                        lb[j] = new_lb;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: j,
                                                old_lb,
                                                old_ub: ub[j],
                                                new_lb,
                                                new_ub: ub[j],
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                }
                                con_removed[i] = true;
                                total_changes += 1;
                            }
                            ConstraintType::Ge => {
                                if coeff > 0.0 {
                                    let new_lb = val;
                                    if new_lb > lb[j] {
                                        let old_lb = lb[j];
                                        lb[j] = new_lb;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: j,
                                                old_lb,
                                                old_ub: ub[j],
                                                new_lb,
                                                new_ub: ub[j],
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                } else {
                                    let new_ub = val;
                                    if new_ub < ub[j] {
                                        let old_ub = ub[j];
                                        ub[j] = new_ub;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: j,
                                                old_lb: lb[j],
                                                old_ub,
                                                new_lb: lb[j],
                                                new_ub,
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                }
                                con_removed[i] = true;
                                total_changes += 1;
                            }
                        }
                    }
                }
            }

            // --- Bound tightening ---
            if self.config.bound_tightening {
                for i in 0..m {
                    if con_removed[i] {
                        continue;
                    }
                    let active: Vec<(usize, f64)> = rows[i]
                        .iter()
                        .filter(|(c, _)| !var_removed[*c])
                        .cloned()
                        .collect();
                    if active.is_empty() {
                        continue;
                    }

                    for &(target_j, target_coeff) in &active {
                        if target_coeff.abs() < 1e-15 {
                            continue;
                        }

                        // Compute activity range of sum excluding target_j
                        let mut min_rest = 0.0f64;
                        let mut max_rest = 0.0f64;
                        let mut bounded = true;
                        for &(j2, c2) in &active {
                            if j2 == target_j {
                                continue;
                            }
                            if c2 > 0.0 {
                                if lb[j2].is_finite() {
                                    min_rest += c2 * lb[j2];
                                } else {
                                    bounded = false;
                                }
                                if ub[j2].is_finite() {
                                    max_rest += c2 * ub[j2];
                                } else {
                                    bounded = false;
                                }
                            } else {
                                if ub[j2].is_finite() {
                                    min_rest += c2 * ub[j2];
                                } else {
                                    bounded = false;
                                }
                                if lb[j2].is_finite() {
                                    max_rest += c2 * lb[j2];
                                } else {
                                    bounded = false;
                                }
                            }
                        }
                        if !bounded {
                            continue;
                        }

                        // Derive new bound for target_j from constraint
                        match ctypes[i] {
                            ConstraintType::Le => {
                                // sum <= rhs  →  target_coeff * x_j <= rhs - min_rest
                                if target_coeff > 0.0 {
                                    let new_ub = (rhs[i] - min_rest) / target_coeff;
                                    if new_ub < ub[target_j] - 1e-8 {
                                        let old_ub = ub[target_j];
                                        ub[target_j] = new_ub;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: target_j,
                                                old_lb: lb[target_j],
                                                old_ub,
                                                new_lb: lb[target_j],
                                                new_ub,
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                } else {
                                    let new_lb = (rhs[i] - min_rest) / target_coeff;
                                    if new_lb > lb[target_j] + 1e-8 {
                                        let old_lb = lb[target_j];
                                        lb[target_j] = new_lb;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: target_j,
                                                old_lb,
                                                old_ub: ub[target_j],
                                                new_lb,
                                                new_ub: ub[target_j],
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                }
                            }
                            ConstraintType::Ge => {
                                if target_coeff > 0.0 {
                                    let new_lb = (rhs[i] - max_rest) / target_coeff;
                                    if new_lb > lb[target_j] + 1e-8 {
                                        let old_lb = lb[target_j];
                                        lb[target_j] = new_lb;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: target_j,
                                                old_lb,
                                                old_ub: ub[target_j],
                                                new_lb,
                                                new_ub: ub[target_j],
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                } else {
                                    let new_ub = (rhs[i] - max_rest) / target_coeff;
                                    if new_ub < ub[target_j] - 1e-8 {
                                        let old_ub = ub[target_j];
                                        ub[target_j] = new_ub;
                                        transformations.push(
                                            PresolveTransformation::BoundUpdate {
                                                var: target_j,
                                                old_lb: lb[target_j],
                                                old_ub,
                                                new_lb: lb[target_j],
                                                new_ub,
                                            },
                                        );
                                        total_changes += 1;
                                    }
                                }
                            }
                            ConstraintType::Eq => {
                                // equality gives both upper and lower
                                let implied = (rhs[i] - max_rest) / target_coeff;
                                let implied2 = (rhs[i] - min_rest) / target_coeff;
                                let new_lb = implied.min(implied2);
                                let new_ub = implied.max(implied2);
                                let mut changed = false;
                                let old_lb = lb[target_j];
                                let old_ub = ub[target_j];
                                if new_lb > lb[target_j] + 1e-8 {
                                    lb[target_j] = new_lb;
                                    changed = true;
                                }
                                if new_ub < ub[target_j] - 1e-8 {
                                    ub[target_j] = new_ub;
                                    changed = true;
                                }
                                if changed {
                                    transformations.push(
                                        PresolveTransformation::BoundUpdate {
                                            var: target_j,
                                            old_lb,
                                            old_ub,
                                            new_lb: lb[target_j],
                                            new_ub: ub[target_j],
                                        },
                                    );
                                    total_changes += 1;
                                }
                            }
                        }

                        // Check infeasibility
                        if lb[target_j] > ub[target_j] + 1e-8 {
                            return Err(OptError::Infeasible {
                                reason: format!(
                                    "Bound tightening made var {} infeasible: lb={:.6} > ub={:.6}",
                                    target_j, lb[target_j], ub[target_j]
                                ),
                            });
                        }
                    }
                }
            }

            // --- Redundant constraint removal ---
            for i in 0..m {
                if con_removed[i] {
                    continue;
                }
                let active: Vec<(usize, f64)> = rows[i]
                    .iter()
                    .filter(|(c, _)| !var_removed[*c])
                    .cloned()
                    .collect();

                if active.is_empty() {
                    // Empty row: check feasibility of 0 {<=,=,>=} rhs
                    let feasible = match ctypes[i] {
                        ConstraintType::Le => rhs[i] >= -1e-8,
                        ConstraintType::Ge => rhs[i] <= 1e-8,
                        ConstraintType::Eq => rhs[i].abs() <= 1e-8,
                    };
                    if !feasible {
                        return Err(OptError::Infeasible {
                            reason: format!(
                                "Empty constraint {} is infeasible: 0 {} {:.6}",
                                i, ctypes[i], rhs[i]
                            ),
                        });
                    }
                    con_removed[i] = true;
                    transformations.push(PresolveTransformation::RedundantConstraint {
                        constraint: i,
                    });
                    total_changes += 1;
                    continue;
                }

                // Compute activity bounds
                let mut min_act = 0.0f64;
                let mut max_act = 0.0f64;
                let mut all_bounded = true;
                for &(j, c) in &active {
                    if c > 0.0 {
                        if lb[j].is_finite() {
                            min_act += c * lb[j];
                        } else {
                            all_bounded = false;
                        }
                        if ub[j].is_finite() {
                            max_act += c * ub[j];
                        } else {
                            all_bounded = false;
                        }
                    } else {
                        if ub[j].is_finite() {
                            min_act += c * ub[j];
                        } else {
                            all_bounded = false;
                        }
                        if lb[j].is_finite() {
                            max_act += c * lb[j];
                        } else {
                            all_bounded = false;
                        }
                    }
                }

                if all_bounded {
                    let redundant = match ctypes[i] {
                        ConstraintType::Le => max_act <= rhs[i] + 1e-8,
                        ConstraintType::Ge => min_act >= rhs[i] - 1e-8,
                        ConstraintType::Eq => false,
                    };
                    if redundant {
                        con_removed[i] = true;
                        transformations.push(PresolveTransformation::RedundantConstraint {
                            constraint: i,
                        });
                        total_changes += 1;
                        continue;
                    }

                    // Forcing constraint: if activity at bounds meets rhs exactly
                    if ctypes[i] == ConstraintType::Le && (min_act - rhs[i]).abs() < 1e-8 {
                        // All vars forced to the bound that minimises activity
                        let mut fixed_vars = Vec::new();
                        for &(j, c) in &active {
                            let val = if c > 0.0 { lb[j] } else { ub[j] };
                            if val.is_finite() {
                                fixed_vars.push((j, val));
                            }
                        }
                        if fixed_vars.len() == active.len() {
                            for &(j, val) in &fixed_vars {
                                var_fixed_value[j] = val;
                                var_removed[j] = true;
                                // Update other rows
                                for ii in 0..m {
                                    if con_removed[ii] || ii == i {
                                        continue;
                                    }
                                    let mut new_row = Vec::new();
                                    for &(col, cf) in &rows[ii] {
                                        if col == j {
                                            rhs[ii] -= cf * val;
                                        } else {
                                            new_row.push((col, cf));
                                        }
                                    }
                                    rows[ii] = new_row;
                                }
                            }
                            con_removed[i] = true;
                            transformations.push(PresolveTransformation::ForcingConstraint {
                                constraint: i,
                                fixed_vars,
                            });
                            total_changes += 1;
                        }
                    }
                }
            }

            // --- Singleton columns ---
            if self.config.singleton_columns {
                // Count column appearances
                let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
                for i in 0..m {
                    if con_removed[i] {
                        continue;
                    }
                    for &(j, _) in &rows[i] {
                        if !var_removed[j] {
                            col_rows[j].push(i);
                        }
                    }
                }

                for j in 0..n {
                    if var_removed[j] || col_rows[j].len() != 1 {
                        continue;
                    }
                    let i = col_rows[j][0];
                    if con_removed[i] {
                        continue;
                    }

                    // Variable j appears only in constraint i
                    let coeff = rows[i]
                        .iter()
                        .find(|(c, _)| *c == j)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    if coeff.abs() < 1e-15 {
                        continue;
                    }

                    // For a minimisation obj, the singleton column's optimal value
                    // depends on its objective coefficient sign and constraint type.
                    let c_j = if problem.maximize {
                        -obj[j]
                    } else {
                        obj[j]
                    };

                    // Determine which bound is optimal
                    let optimal_val = match ctypes[i] {
                        ConstraintType::Le => {
                            if (c_j > 0.0 && coeff > 0.0) || (c_j < 0.0 && coeff < 0.0) {
                                lb[j]
                            } else if c_j.abs() < 1e-15 {
                                lb[j]
                            } else {
                                ub[j]
                            }
                        }
                        ConstraintType::Ge => {
                            if (c_j > 0.0 && coeff < 0.0) || (c_j < 0.0 && coeff > 0.0) {
                                lb[j]
                            } else if c_j.abs() < 1e-15 {
                                lb[j]
                            } else {
                                ub[j]
                            }
                        }
                        ConstraintType::Eq => {
                            // Can't eliminate singleton column from equality easily
                            // unless the constraint can be used to substitute
                            continue;
                        }
                    };

                    if !optimal_val.is_finite() {
                        if c_j < -1e-15 {
                            return Err(OptError::Unbounded {
                                reason: format!("Singleton column {} is unbounded", j),
                            });
                        }
                        continue;
                    }

                    transformations.push(PresolveTransformation::SingletonColumn {
                        var: j,
                        lb: lb[j],
                        ub: ub[j],
                    });
                    var_fixed_value[j] = optimal_val;
                    var_removed[j] = true;
                    rhs[i] -= coeff * optimal_val;
                    rows[i].retain(|(c, _)| *c != j);
                    total_changes += 1;
                }
            }

            // --- Dominated columns ---
            if self.config.dominated_columns {
                self.detect_dominated_columns(
                    n,
                    m,
                    &rows,
                    &obj,
                    &lb,
                    &ub,
                    &var_removed,
                    &con_removed,
                    &mut transformations,
                    &mut total_changes,
                );
            }

            // --- Duplicate row detection ---
            if self.config.duplicate_detection {
                self.detect_duplicate_rows(
                    m,
                    &rows,
                    &rhs,
                    &ctypes,
                    &var_removed,
                    &mut con_removed,
                    &mut transformations,
                    &mut total_changes,
                );
            }

            debug!(
                "Presolve pass {}: {} changes",
                pass,
                total_changes - changes_before
            );

            if total_changes == changes_before {
                break;
            }
        }

        // Build reduced problem
        let remaining_vars: Vec<usize> = (0..n).filter(|j| !var_removed[*j]).collect();
        let remaining_cons: Vec<usize> = (0..m).filter(|i| !con_removed[*i]).collect();

        let mut original_to_reduced_var = vec![None; n];
        for (new_j, &old_j) in remaining_vars.iter().enumerate() {
            original_to_reduced_var[old_j] = Some(new_j);
        }
        let mut original_to_reduced_con = vec![None; m];
        for (new_i, &old_i) in remaining_cons.iter().enumerate() {
            original_to_reduced_con[old_i] = Some(new_i);
        }

        let mut reduced = LpProblem::new(problem.maximize);
        for &old_j in &remaining_vars {
            reduced.add_variable(
                obj[old_j],
                lb[old_j],
                ub[old_j],
                Some(problem.var_names[old_j].clone()),
            );
        }
        for &old_i in &remaining_cons {
            let active: Vec<(usize, f64)> = rows[old_i]
                .iter()
                .filter(|(c, _)| !var_removed[*c])
                .map(|(c, v)| (original_to_reduced_var[*c].unwrap(), *v))
                .collect();
            let indices: Vec<usize> = active.iter().map(|(i, _)| *i).collect();
            let coeffs: Vec<f64> = active.iter().map(|(_, v)| *v).collect();
            reduced.add_constraint(&indices, &coeffs, ctypes[old_i], rhs[old_i])?;
        }

        let removed_vars = n - remaining_vars.len();
        let removed_constraints = m - remaining_cons.len();

        info!(
            "Presolve complete: removed {} vars, {} constraints ({} transformations)",
            removed_vars,
            removed_constraints,
            transformations.len()
        );

        Ok(PresolveResult {
            reduced_problem: reduced,
            transformations,
            removed_vars,
            removed_constraints,
            original_to_reduced_var,
            original_to_reduced_con,
        })
    }

    /// Undo presolve transformations to recover solution for original problem.
    pub fn postsolve(
        &self,
        result: &PresolveResult,
        reduced_solution: &LpSolution,
    ) -> OptResult<LpSolution> {
        let n_orig = result.original_to_reduced_var.len();
        let m_orig = result.original_to_reduced_con.len();

        let mut primal = vec![0.0; n_orig];
        let mut dual = vec![0.0; m_orig];

        // Fill in reduced-problem solution values
        for (old_j, mapped) in result.original_to_reduced_var.iter().enumerate() {
            if let Some(new_j) = mapped {
                if *new_j < reduced_solution.primal_values.len() {
                    primal[old_j] = reduced_solution.primal_values[*new_j];
                }
            }
        }
        for (old_i, mapped) in result.original_to_reduced_con.iter().enumerate() {
            if let Some(new_i) = mapped {
                if *new_i < reduced_solution.dual_values.len() {
                    dual[old_i] = reduced_solution.dual_values[*new_i];
                }
            }
        }

        // Apply transformations in reverse order
        for t in result.transformations.iter().rev() {
            match t {
                PresolveTransformation::FixedVariable { var, value } => {
                    primal[*var] = *value;
                }
                PresolveTransformation::SingletonRow { var, value, .. } => {
                    primal[*var] = *value;
                }
                PresolveTransformation::SingletonColumn { var, lb, .. } => {
                    // Use lb as default (the value was determined during presolve)
                    if primal[*var] == 0.0 {
                        primal[*var] = *lb;
                    }
                }
                PresolveTransformation::BoundUpdate { var, .. } => {
                    // Clamp to original bounds (already set)
                    let _ = var;
                }
                PresolveTransformation::RedundantConstraint { constraint } => {
                    // Dual value is 0 for redundant constraints
                    dual[*constraint] = 0.0;
                }
                PresolveTransformation::ForcingConstraint {
                    fixed_vars, ..
                } => {
                    for &(j, val) in fixed_vars {
                        primal[j] = val;
                    }
                }
                PresolveTransformation::DuplicateRow { removed, kept } => {
                    dual[*removed] = dual[*kept];
                }
                PresolveTransformation::DuplicateColumn {
                    kept,
                    removed,
                    multiplier,
                } => {
                    primal[*removed] = primal[*kept] * multiplier;
                }
                PresolveTransformation::FreeColumnSubstitution { var, .. } => {
                    let _ = var; // value already set from reduced solution mapping
                }
            }
        }

        let basis_status = vec![BasisStatus::Free; n_orig];

        Ok(LpSolution {
            status: reduced_solution.status,
            objective_value: reduced_solution.objective_value,
            primal_values: primal,
            dual_values: dual,
            reduced_costs: vec![0.0; n_orig],
            basis_status,
            iterations: reduced_solution.iterations,
            time_seconds: reduced_solution.time_seconds,
        })
    }

    /// Detect dominated columns: column j dominates column k if in every
    /// constraint a_{ij} >= a_{ik} (for Le constraints) and c_j <= c_k.
    fn detect_dominated_columns(
        &self,
        n: usize,
        _m: usize,
        _rows: &[Vec<(usize, f64)>],
        obj: &[f64],
        lb: &[f64],
        _ub: &[f64],
        var_removed: &[bool],
        _con_removed: &[bool],
        transformations: &mut Vec<PresolveTransformation>,
        total_changes: &mut usize,
    ) {
        // Simple pairwise check for small problems (quadratic in n, but practical
        // with the column-count filter we apply below).
        let active: Vec<usize> = (0..n).filter(|j| !var_removed[*j]).collect();
        if active.len() > 200 {
            return; // skip for large problems
        }

        for idx_a in 0..active.len() {
            let j = active[idx_a];
            for idx_b in (idx_a + 1)..active.len() {
                let k = active[idx_b];
                // Check if j dominates k: c_j <= c_k and both have lb >= 0
                if obj[j] <= obj[k] + 1e-10 && lb[j] >= 0.0 && lb[k] >= 0.0 {
                    // Just record the domination; actual elimination requires more
                    // careful checking of all constraints, which we skip for very
                    // large problems. For small ones, log it.
                    debug!("Column {} potentially dominates column {}", j, k);
                }
            }
        }
        // Dominated column detection is heuristic — we record but don't remove
        // to keep the implementation safe for all problem types.
        let _ = transformations;
        let _ = total_changes;
    }

    /// Detect duplicate (proportional) rows using a fingerprinting approach.
    fn detect_duplicate_rows(
        &self,
        m: usize,
        rows: &[Vec<(usize, f64)>],
        rhs: &[f64],
        ctypes: &[ConstraintType],
        var_removed: &[bool],
        con_removed: &mut [bool],
        transformations: &mut Vec<PresolveTransformation>,
        total_changes: &mut usize,
    ) {
        use std::collections::HashMap;

        // Fingerprint each row by its sorted column indices
        let mut buckets: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for i in 0..m {
            if con_removed[i] {
                continue;
            }
            let mut cols: Vec<usize> = rows[i]
                .iter()
                .filter(|(c, _)| !var_removed[*c])
                .map(|(c, _)| *c)
                .collect();
            cols.sort_unstable();
            buckets.entry(cols).or_default().push(i);
        }

        for (_cols, group) in &buckets {
            if group.len() < 2 {
                continue;
            }
            // Pairwise check for proportionality
            for idx_a in 0..group.len() {
                let i = group[idx_a];
                if con_removed[i] {
                    continue;
                }
                for idx_b in (idx_a + 1)..group.len() {
                    let k = group[idx_b];
                    if con_removed[k] {
                        continue;
                    }
                    if let Some(ratio) = row_proportional(
                        &rows[i],
                        &rows[k],
                        var_removed,
                    ) {
                        // Rows are proportional with factor `ratio`: row_i = ratio * row_k
                        // Check if one is redundant
                        let rhs_i = rhs[i];
                        let rhs_k_scaled = ratio * rhs[k];

                        let same_type = ctypes[i] == ctypes[k]
                            || (ratio < 0.0
                                && ((ctypes[i] == ConstraintType::Le
                                    && ctypes[k] == ConstraintType::Ge)
                                    || (ctypes[i] == ConstraintType::Ge
                                        && ctypes[k] == ConstraintType::Le)));

                        if same_type && (rhs_i - rhs_k_scaled).abs() < 1e-8 {
                            con_removed[k] = true;
                            transformations.push(PresolveTransformation::DuplicateRow {
                                kept: i,
                                removed: k,
                            });
                            *total_changes += 1;
                            debug!("Duplicate rows {} and {}, removed {}", i, k, k);
                        }
                    }
                }
            }
        }
    }
}

/// Check if two sparse rows are proportional. Returns the ratio if so.
fn row_proportional(
    row_a: &[(usize, f64)],
    row_b: &[(usize, f64)],
    var_removed: &[bool],
) -> Option<f64> {
    let a: Vec<(usize, f64)> = row_a
        .iter()
        .filter(|(c, _)| !var_removed[*c])
        .cloned()
        .collect();
    let b: Vec<(usize, f64)> = row_b
        .iter()
        .filter(|(c, _)| !var_removed[*c])
        .cloned()
        .collect();

    if a.len() != b.len() || a.is_empty() {
        return None;
    }

    // Sort both by column index
    let mut a_sorted = a;
    let mut b_sorted = b;
    a_sorted.sort_by_key(|(c, _)| *c);
    b_sorted.sort_by_key(|(c, _)| *c);

    let mut ratio: Option<f64> = None;
    for (&(ca, va), &(cb, vb)) in a_sorted.iter().zip(b_sorted.iter()) {
        if ca != cb {
            return None;
        }
        if vb.abs() < 1e-15 {
            if va.abs() > 1e-15 {
                return None;
            }
            continue;
        }
        let r = va / vb;
        match ratio {
            None => ratio = Some(r),
            Some(prev) => {
                if (r - prev).abs() > 1e-8 * (1.0 + prev.abs()) {
                    return None;
                }
            }
        }
    }
    ratio
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{ConstraintType, LpProblem, SolverStatus};

    fn make_presolve() -> LpPresolve {
        LpPresolve::new(PresolveConfig::default())
    }

    #[test]
    fn test_empty_problem() {
        let lp = LpProblem::new(false);
        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert_eq!(res.removed_vars, 0);
        assert_eq!(res.removed_constraints, 0);
    }

    #[test]
    fn test_fixed_variable_removal() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 5.0, 5.0, Some("x0".into())); // fixed at 5
        lp.add_variable(2.0, 0.0, 10.0, Some("x1".into()));
        // x0 + x1 <= 10
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 10.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert!(res.removed_vars >= 1);
        assert_eq!(res.reduced_problem.num_vars, 1);
        // rhs adjusted: 10 - 5 = 5
        assert!((res.reduced_problem.rhs[0] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_singleton_row_eq() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 10.0, None);
        lp.add_variable(2.0, 0.0, 10.0, None);
        // x0 = 3 (singleton row)
        lp.add_constraint(&[0], &[1.0], ConstraintType::Eq, 3.0)
            .unwrap();
        // x0 + x1 <= 7
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 7.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        // x0 should be fixed to 3, constraint removed
        assert!(res.removed_vars >= 1);
        assert!(res.removed_constraints >= 1);
    }

    #[test]
    fn test_singleton_row_infeasible() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 2.0, None);
        // x0 = 5, but x0 <= 2 → infeasible
        lp.add_constraint(&[0], &[1.0], ConstraintType::Eq, 5.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp);
        assert!(res.is_err());
    }

    #[test]
    fn test_redundant_constraint() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 3.0, None);
        lp.add_variable(1.0, 0.0, 3.0, None);
        // x0 + x1 <= 100 (always satisfied since max is 6)
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 100.0)
            .unwrap();
        // x0 + x1 <= 5
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert!(res.removed_constraints >= 1);
    }

    #[test]
    fn test_bound_tightening() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 100.0, None);
        lp.add_variable(1.0, 0.0, 100.0, None);
        // x0 + x1 <= 5
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        // Upper bounds should be tightened to 5
        assert!(res.reduced_problem.upper_bounds[0] <= 5.0 + 1e-6);
        assert!(res.reduced_problem.upper_bounds[1] <= 5.0 + 1e-6);
    }

    #[test]
    fn test_duplicate_row_detection() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 10.0, None);
        lp.add_variable(1.0, 0.0, 10.0, None);
        // Two identical constraints: x0 + x1 <= 5
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert!(res.removed_constraints >= 1);
    }

    #[test]
    fn test_proportional_row_detection() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 10.0, None);
        lp.add_variable(1.0, 0.0, 10.0, None);
        // x0 + x1 <= 5
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();
        // 2*x0 + 2*x1 <= 10 (same constraint scaled by 2)
        lp.add_constraint(&[0, 1], &[2.0, 2.0], ConstraintType::Le, 10.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert!(res.removed_constraints >= 1);
    }

    #[test]
    fn test_postsolve_restores_fixed() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 5.0, 5.0, None); // fixed at 5
        lp.add_variable(2.0, 0.0, 10.0, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 10.0)
            .unwrap();

        let ps = make_presolve();
        let pres = ps.presolve(&lp).unwrap();

        // Simulate a solution to the reduced problem
        let reduced_sol = LpSolution {
            status: SolverStatus::Optimal,
            objective_value: 15.0,
            primal_values: vec![5.0], // x1 = 5
            dual_values: vec![2.0],
            reduced_costs: vec![0.0],
            basis_status: vec![BasisStatus::Basic],
            iterations: 1,
            time_seconds: 0.1,
        };

        let orig_sol = ps.postsolve(&pres, &reduced_sol).unwrap();
        assert!((orig_sol.primal_values[0] - 5.0).abs() < 1e-8);
        assert!((orig_sol.primal_values[1] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_forcing_constraint() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 3.0, None);
        lp.add_variable(1.0, 0.0, 3.0, None);
        // x0 + x1 <= 0 → forces x0=0, x1=0
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 0.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp).unwrap();
        assert!(res.removed_vars >= 2);
    }

    #[test]
    fn test_presolve_config_disable_all() {
        let config = PresolveConfig {
            max_passes: 0,
            ..Default::default()
        };
        let ps = LpPresolve::new(config);
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 5.0, 5.0, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 10.0)
            .unwrap();

        let res = ps.presolve(&lp).unwrap();
        // With 0 passes, nothing should be removed
        assert_eq!(res.removed_vars, 0);
    }

    #[test]
    fn test_empty_row_infeasible() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 5.0, 5.0, None);
        // 1*x0 = 5 will be processed, leaving an empty row in further passes
        // But let's directly test: after fixing x0=5, the constraint 1*x0 <= 3
        // becomes 0 <= -2, which is infeasible
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();

        let ps = make_presolve();
        let res = ps.presolve(&lp);
        // After fixing x0=5, constraint becomes 0 <= 3-5 = -2 which is infeasible
        assert!(res.is_err());
    }
}
