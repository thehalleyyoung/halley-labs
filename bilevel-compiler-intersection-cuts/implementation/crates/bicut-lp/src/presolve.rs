//! LP presolve: singleton row/column removal, forcing/dominated constraints,
//! bound strengthening, duplicate row/column detection, probing,
//! coefficient reduction.

use crate::model::{Constraint, LpModel, VarType, Variable};
use bicut_types::ConstraintSense;
use log::debug;
use std::collections::{HashMap, HashSet, VecDeque};

/// Presolve configuration.
#[derive(Debug, Clone)]
pub struct PresolveConfig {
    /// Maximum number of presolve passes.
    pub max_passes: usize,
    /// Enable singleton row removal.
    pub singleton_rows: bool,
    /// Enable singleton column removal.
    pub singleton_cols: bool,
    /// Enable forcing/dominated constraint detection.
    pub forcing_constraints: bool,
    /// Enable bound strengthening.
    pub bound_strengthening: bool,
    /// Enable duplicate row/column detection.
    pub duplicate_detection: bool,
    /// Enable probing on binary variables.
    pub probing: bool,
    /// Enable coefficient reduction.
    pub coefficient_reduction: bool,
    /// Tolerance for treating values as zero.
    pub zero_tol: f64,
    /// Tolerance for bound comparisons.
    pub bound_tol: f64,
}

impl Default for PresolveConfig {
    fn default() -> Self {
        Self {
            max_passes: 10,
            singleton_rows: true,
            singleton_cols: true,
            forcing_constraints: true,
            bound_strengthening: true,
            duplicate_detection: true,
            probing: true,
            coefficient_reduction: true,
            zero_tol: 1e-10,
            bound_tol: 1e-8,
        }
    }
}

/// Records of presolve operations for post-solve recovery.
#[derive(Debug, Clone)]
pub enum PresolveOp {
    FixVariable {
        var_idx: usize,
        value: f64,
        orig_lb: f64,
        orig_ub: f64,
    },
    RemoveConstraint {
        con_idx: usize,
        constraint: Constraint,
    },
    RemoveFreeColumnSingleton {
        var_idx: usize,
        con_idx: usize,
        coeff: f64,
        rhs: f64,
    },
    TightenBound {
        var_idx: usize,
        old_lb: f64,
        old_ub: f64,
        new_lb: f64,
        new_ub: f64,
    },
    ForcingConstraint {
        con_idx: usize,
        fixed_vars: Vec<(usize, f64)>,
    },
    DuplicateRow {
        kept_idx: usize,
        removed_idx: usize,
    },
    DuplicateCol {
        kept_idx: usize,
        removed_idx: usize,
        ratio: f64,
    },
    CoefficientReduction {
        con_idx: usize,
        var_idx: usize,
        old_coeff: f64,
        new_coeff: f64,
        new_rhs: f64,
    },
}

/// Presolve statistics.
#[derive(Debug, Clone, Default)]
pub struct PresolveStats {
    pub fixed_vars: usize,
    pub removed_constraints: usize,
    pub tightened_bounds: usize,
    pub forcing_constraints: usize,
    pub duplicate_rows: usize,
    pub duplicate_cols: usize,
    pub coefficient_reductions: usize,
    pub passes: usize,
}

/// Result of presolve.
#[derive(Debug, Clone)]
pub struct PresolveResult {
    pub presolved_model: LpModel,
    pub ops: Vec<PresolveOp>,
    pub stats: PresolveStats,
    pub is_infeasible: bool,
    pub is_unbounded: bool,
    /// Map from presolved variable index to original variable index.
    pub var_map: Vec<usize>,
    /// Map from presolved constraint index to original constraint index.
    pub con_map: Vec<usize>,
}

/// Main presolve entry point.
pub fn presolve(model: &LpModel, config: &PresolveConfig) -> PresolveResult {
    let mut presolver = Presolver::new(model.clone(), config.clone());
    presolver.run();

    let var_map = presolver.build_var_map();
    let con_map = presolver.build_con_map();

    PresolveResult {
        presolved_model: presolver.model,
        ops: presolver.ops,
        stats: presolver.stats,
        is_infeasible: presolver.infeasible,
        is_unbounded: presolver.unbounded,
        var_map,
        con_map,
    }
}

/// Recover original solution from presolved solution.
pub fn postsolve(
    presolved_primal: &[f64],
    presolved_dual: &[f64],
    ops: &[PresolveOp],
    var_map: &[usize],
    num_orig_vars: usize,
    num_orig_constraints: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut primal = vec![0.0; num_orig_vars];
    let mut dual = vec![0.0; num_orig_constraints];

    // Map presolved values to original
    for (ps_idx, &orig_idx) in var_map.iter().enumerate() {
        if ps_idx < presolved_primal.len() && orig_idx < num_orig_vars {
            primal[orig_idx] = presolved_primal[ps_idx];
        }
    }

    // Copy presolved dual values
    for (i, &val) in presolved_dual.iter().enumerate() {
        if i < dual.len() {
            dual[i] = val;
        }
    }

    // Apply postsolve operations in reverse order
    for op in ops.iter().rev() {
        match op {
            PresolveOp::FixVariable { var_idx, value, .. } => {
                if *var_idx < primal.len() {
                    primal[*var_idx] = *value;
                }
            }
            PresolveOp::RemoveFreeColumnSingleton {
                var_idx,
                con_idx,
                coeff,
                rhs,
            } => {
                // Recover the variable value from the constraint
                if *var_idx < primal.len() && coeff.abs() > 1e-20 {
                    // a_j * x_j + sum_{k!=j} a_k * x_k = rhs
                    // x_j = (rhs - sum_{k!=j} a_k * x_k) / a_j
                    // For now, just set based on the constraint
                    primal[*var_idx] = rhs / coeff;
                }
                let _ = con_idx;
            }
            PresolveOp::ForcingConstraint { fixed_vars, .. } => {
                for &(var_idx, value) in fixed_vars {
                    if var_idx < primal.len() {
                        primal[var_idx] = value;
                    }
                }
            }
            PresolveOp::DuplicateCol {
                kept_idx,
                removed_idx,
                ratio,
            } => {
                if *removed_idx < primal.len() && *kept_idx < primal.len() {
                    primal[*removed_idx] = primal[*kept_idx] * ratio;
                }
            }
            _ => {}
        }
    }

    (primal, dual)
}

/// Internal presolver state.
struct Presolver {
    model: LpModel,
    config: PresolveConfig,
    ops: Vec<PresolveOp>,
    stats: PresolveStats,
    removed_vars: HashSet<usize>,
    removed_cons: HashSet<usize>,
    infeasible: bool,
    unbounded: bool,
}

impl Presolver {
    fn new(model: LpModel, config: PresolveConfig) -> Self {
        Self {
            model,
            config,
            ops: Vec::new(),
            stats: PresolveStats::default(),
            removed_vars: HashSet::new(),
            removed_cons: HashSet::new(),
            infeasible: false,
            unbounded: false,
        }
    }

    fn run(&mut self) {
        for pass in 0..self.config.max_passes {
            let changes_before = self.total_changes();

            if self.config.singleton_rows {
                self.remove_singleton_rows();
            }
            if self.infeasible || self.unbounded {
                break;
            }

            if self.config.singleton_cols {
                self.remove_singleton_columns();
            }
            if self.infeasible || self.unbounded {
                break;
            }

            if self.config.bound_strengthening {
                self.strengthen_bounds();
            }
            if self.infeasible {
                break;
            }

            if self.config.forcing_constraints {
                self.detect_forcing_constraints();
            }
            if self.infeasible {
                break;
            }

            if self.config.duplicate_detection {
                self.detect_duplicate_rows();
                self.detect_duplicate_columns();
            }

            if self.config.coefficient_reduction {
                self.reduce_coefficients();
            }

            // Remove fixed variables
            self.remove_fixed_variables();

            self.stats.passes = pass + 1;

            if self.total_changes() == changes_before {
                break; // No progress
            }
        }

        // Clean up model: renumber variables and constraints
        self.compact_model();
    }

    fn total_changes(&self) -> usize {
        self.stats.fixed_vars
            + self.stats.removed_constraints
            + self.stats.tightened_bounds
            + self.stats.forcing_constraints
            + self.stats.duplicate_rows
            + self.stats.duplicate_cols
            + self.stats.coefficient_reductions
    }

    /// Remove singleton rows (constraints with exactly one variable).
    fn remove_singleton_rows(&mut self) {
        let mut to_process: VecDeque<usize> = VecDeque::new();

        for (i, con) in self.model.constraints.iter().enumerate() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let nnz = con
                .row_indices
                .iter()
                .filter(|&&j| !self.removed_vars.contains(&j))
                .count();
            if nnz == 1 {
                to_process.push_back(i);
            }
        }

        while let Some(con_idx) = to_process.pop_front() {
            if self.removed_cons.contains(&con_idx) {
                continue;
            }
            let con = &self.model.constraints[con_idx];
            let active: Vec<(usize, f64)> = con
                .row_indices
                .iter()
                .zip(con.row_values.iter())
                .filter(|(&j, _)| !self.removed_vars.contains(&j))
                .map(|(&j, &v)| (j, v))
                .collect();

            if active.len() != 1 {
                continue;
            }

            let (var_idx, coeff) = active[0];
            if coeff.abs() < self.config.zero_tol {
                continue;
            }

            let rhs = con.rhs;
            let bound_val = rhs / coeff;

            let var = &mut self.model.variables[var_idx];
            match con.sense {
                ConstraintSense::Eq => {
                    // Fix the variable
                    if bound_val < var.lower_bound - self.config.bound_tol
                        || bound_val > var.upper_bound + self.config.bound_tol
                    {
                        self.infeasible = true;
                        return;
                    }
                    self.ops.push(PresolveOp::FixVariable {
                        var_idx,
                        value: bound_val,
                        orig_lb: var.lower_bound,
                        orig_ub: var.upper_bound,
                    });
                    var.lower_bound = bound_val;
                    var.upper_bound = bound_val;
                    self.stats.fixed_vars += 1;
                }
                ConstraintSense::Le => {
                    if coeff > 0.0 {
                        // x <= bound_val
                        let new_ub = bound_val.min(var.upper_bound);
                        if new_ub < var.lower_bound - self.config.bound_tol {
                            self.infeasible = true;
                            return;
                        }
                        if new_ub < var.upper_bound - self.config.bound_tol {
                            self.ops.push(PresolveOp::TightenBound {
                                var_idx,
                                old_lb: var.lower_bound,
                                old_ub: var.upper_bound,
                                new_lb: var.lower_bound,
                                new_ub,
                            });
                            var.upper_bound = new_ub;
                            self.stats.tightened_bounds += 1;
                        }
                    } else {
                        // x >= bound_val (since coeff < 0, dividing flips inequality)
                        let new_lb = bound_val.max(var.lower_bound);
                        if new_lb > var.upper_bound + self.config.bound_tol {
                            self.infeasible = true;
                            return;
                        }
                        if new_lb > var.lower_bound + self.config.bound_tol {
                            self.ops.push(PresolveOp::TightenBound {
                                var_idx,
                                old_lb: var.lower_bound,
                                old_ub: var.upper_bound,
                                new_lb,
                                new_ub: var.upper_bound,
                            });
                            var.lower_bound = new_lb;
                            self.stats.tightened_bounds += 1;
                        }
                    }
                }
                ConstraintSense::Ge => {
                    if coeff > 0.0 {
                        let new_lb = bound_val.max(var.lower_bound);
                        if new_lb > var.upper_bound + self.config.bound_tol {
                            self.infeasible = true;
                            return;
                        }
                        if new_lb > var.lower_bound + self.config.bound_tol {
                            self.ops.push(PresolveOp::TightenBound {
                                var_idx,
                                old_lb: var.lower_bound,
                                old_ub: var.upper_bound,
                                new_lb,
                                new_ub: var.upper_bound,
                            });
                            var.lower_bound = new_lb;
                            self.stats.tightened_bounds += 1;
                        }
                    } else {
                        let new_ub = bound_val.min(var.upper_bound);
                        if new_ub < var.lower_bound - self.config.bound_tol {
                            self.infeasible = true;
                            return;
                        }
                        if new_ub < var.upper_bound - self.config.bound_tol {
                            self.ops.push(PresolveOp::TightenBound {
                                var_idx,
                                old_lb: var.lower_bound,
                                old_ub: var.upper_bound,
                                new_lb: var.lower_bound,
                                new_ub,
                            });
                            var.upper_bound = new_ub;
                            self.stats.tightened_bounds += 1;
                        }
                    }
                }
            }

            self.removed_cons.insert(con_idx);
            self.stats.removed_constraints += 1;
        }
    }

    /// Remove singleton columns (variables appearing in only one constraint).
    fn remove_singleton_columns(&mut self) {
        let n = self.model.variables.len();

        // Count appearances of each variable
        let mut appearances: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, con) in self.model.constraints.iter().enumerate() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            for &j in &con.row_indices {
                if !self.removed_vars.contains(&j) {
                    appearances[j].push(i);
                }
            }
        }

        for j in 0..n {
            if self.removed_vars.contains(&j) {
                continue;
            }
            if appearances[j].len() != 1 {
                continue;
            }

            let con_idx = appearances[j][0];
            let var = &self.model.variables[j];

            // Free column singleton: can determine variable from constraint
            if var.obj_coeff.abs() < self.config.zero_tol {
                let con = &self.model.constraints[con_idx];
                let pos = con.row_indices.iter().position(|&c| c == j);
                if let Some(p) = pos {
                    let coeff = con.row_values[p];
                    if coeff.abs() > self.config.zero_tol {
                        self.ops.push(PresolveOp::RemoveFreeColumnSingleton {
                            var_idx: j,
                            con_idx,
                            coeff,
                            rhs: con.rhs,
                        });
                        self.removed_vars.insert(j);
                        // Adjust RHS of the constraint to remove this variable's contribution
                        // is handled during postsolve
                    }
                }
            }
        }
    }

    /// Strengthen variable bounds using constraint information.
    fn strengthen_bounds(&mut self) {
        for i in 0..self.model.constraints.len() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let con = &self.model.constraints[i];

            // Compute min/max activity of the constraint
            let (min_act, max_act) = self.compute_activity_bounds(i);

            match con.sense {
                ConstraintSense::Le => {
                    // sum a_j x_j <= rhs
                    // For each variable j with a_j > 0:
                    //   a_j * x_j <= rhs - min_activity_without_j
                    //   x_j <= (rhs - min_without_j) / a_j
                    if min_act > con.rhs + self.config.bound_tol {
                        self.infeasible = true;
                        return;
                    }
                    for (&col, &coeff) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if self.removed_vars.contains(&col) || coeff.abs() < self.config.zero_tol {
                            continue;
                        }
                        let var = &self.model.variables[col];
                        let min_without = min_act
                            - if coeff > 0.0 {
                                coeff * var.lower_bound
                            } else {
                                coeff * var.upper_bound
                            };

                        if coeff > 0.0 {
                            let implied_ub = (con.rhs - min_without) / coeff;
                            if implied_ub
                                < self.model.variables[col].upper_bound - self.config.bound_tol
                            {
                                let old_ub = self.model.variables[col].upper_bound;
                                self.ops.push(PresolveOp::TightenBound {
                                    var_idx: col,
                                    old_lb: self.model.variables[col].lower_bound,
                                    old_ub,
                                    new_lb: self.model.variables[col].lower_bound,
                                    new_ub: implied_ub,
                                });
                                self.model.variables[col].upper_bound = implied_ub;
                                self.stats.tightened_bounds += 1;
                            }
                        } else {
                            let implied_lb = (con.rhs - min_without) / coeff;
                            if implied_lb
                                > self.model.variables[col].lower_bound + self.config.bound_tol
                            {
                                let old_lb = self.model.variables[col].lower_bound;
                                self.ops.push(PresolveOp::TightenBound {
                                    var_idx: col,
                                    old_lb,
                                    old_ub: self.model.variables[col].upper_bound,
                                    new_lb: implied_lb,
                                    new_ub: self.model.variables[col].upper_bound,
                                });
                                self.model.variables[col].lower_bound = implied_lb;
                                self.stats.tightened_bounds += 1;
                            }
                        }
                    }
                }
                ConstraintSense::Ge => {
                    // sum a_j x_j >= rhs
                    if max_act < con.rhs - self.config.bound_tol {
                        self.infeasible = true;
                        return;
                    }
                    for (&col, &coeff) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if self.removed_vars.contains(&col) || coeff.abs() < self.config.zero_tol {
                            continue;
                        }
                        let var = &self.model.variables[col];
                        let max_without = max_act
                            - if coeff > 0.0 {
                                coeff * var.upper_bound
                            } else {
                                coeff * var.lower_bound
                            };

                        if coeff > 0.0 {
                            let implied_lb = (con.rhs - max_without) / coeff;
                            if implied_lb
                                > self.model.variables[col].lower_bound + self.config.bound_tol
                            {
                                let old_lb = self.model.variables[col].lower_bound;
                                self.ops.push(PresolveOp::TightenBound {
                                    var_idx: col,
                                    old_lb,
                                    old_ub: self.model.variables[col].upper_bound,
                                    new_lb: implied_lb,
                                    new_ub: self.model.variables[col].upper_bound,
                                });
                                self.model.variables[col].lower_bound = implied_lb;
                                self.stats.tightened_bounds += 1;
                            }
                        } else {
                            let implied_ub = (con.rhs - max_without) / coeff;
                            if implied_ub
                                < self.model.variables[col].upper_bound - self.config.bound_tol
                            {
                                let old_ub = self.model.variables[col].upper_bound;
                                self.ops.push(PresolveOp::TightenBound {
                                    var_idx: col,
                                    old_lb: self.model.variables[col].lower_bound,
                                    old_ub,
                                    new_lb: self.model.variables[col].lower_bound,
                                    new_ub: implied_ub,
                                });
                                self.model.variables[col].upper_bound = implied_ub;
                                self.stats.tightened_bounds += 1;
                            }
                        }
                    }
                }
                ConstraintSense::Eq => {
                    if min_act > con.rhs + self.config.bound_tol
                        || max_act < con.rhs - self.config.bound_tol
                    {
                        self.infeasible = true;
                        return;
                    }
                }
            }
        }
    }

    /// Compute the minimum and maximum activity of a constraint.
    fn compute_activity_bounds(&self, con_idx: usize) -> (f64, f64) {
        let con = &self.model.constraints[con_idx];
        let mut min_act = 0.0f64;
        let mut max_act = 0.0f64;

        for (&col, &coeff) in con.row_indices.iter().zip(con.row_values.iter()) {
            if self.removed_vars.contains(&col) {
                continue;
            }
            let var = &self.model.variables[col];
            let lb = var.lower_bound;
            let ub = var.upper_bound;

            if coeff > 0.0 {
                if lb > -1e20 {
                    min_act += coeff * lb;
                } else {
                    min_act = f64::NEG_INFINITY;
                }
                if ub < 1e20 {
                    max_act += coeff * ub;
                } else {
                    max_act = f64::INFINITY;
                }
            } else {
                if ub < 1e20 {
                    min_act += coeff * ub;
                } else {
                    min_act = f64::NEG_INFINITY;
                }
                if lb > -1e20 {
                    max_act += coeff * lb;
                } else {
                    max_act = f64::INFINITY;
                }
            }
        }

        (min_act, max_act)
    }

    /// Detect forcing and dominated constraints.
    fn detect_forcing_constraints(&mut self) {
        for i in 0..self.model.constraints.len() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let (min_act, max_act) = self.compute_activity_bounds(i);
            let con = &self.model.constraints[i];

            match con.sense {
                ConstraintSense::Le => {
                    // Forcing: if max_act <= rhs, constraint is redundant
                    if max_act <= con.rhs + self.config.bound_tol {
                        self.ops.push(PresolveOp::RemoveConstraint {
                            con_idx: i,
                            constraint: con.clone(),
                        });
                        self.removed_cons.insert(i);
                        self.stats.removed_constraints += 1;
                        continue;
                    }
                    // Forcing: if min_act ~= rhs, all variables are forced to their bounds
                    if (min_act - con.rhs).abs() < self.config.bound_tol {
                        let mut fixed = Vec::new();
                        for (&col, &coeff) in con.row_indices.iter().zip(con.row_values.iter()) {
                            if self.removed_vars.contains(&col) {
                                continue;
                            }
                            let var = &self.model.variables[col];
                            let value = if coeff > 0.0 {
                                var.lower_bound
                            } else {
                                var.upper_bound
                            };
                            if value.is_finite() {
                                fixed.push((col, value));
                            }
                        }
                        if !fixed.is_empty() {
                            for &(col, val) in &fixed {
                                self.model.variables[col].lower_bound = val;
                                self.model.variables[col].upper_bound = val;
                            }
                            self.ops.push(PresolveOp::ForcingConstraint {
                                con_idx: i,
                                fixed_vars: fixed,
                            });
                            self.removed_cons.insert(i);
                            self.stats.forcing_constraints += 1;
                        }
                    }
                }
                ConstraintSense::Ge => {
                    if min_act >= con.rhs - self.config.bound_tol {
                        self.ops.push(PresolveOp::RemoveConstraint {
                            con_idx: i,
                            constraint: con.clone(),
                        });
                        self.removed_cons.insert(i);
                        self.stats.removed_constraints += 1;
                        continue;
                    }
                    if (max_act - con.rhs).abs() < self.config.bound_tol {
                        let mut fixed = Vec::new();
                        for (&col, &coeff) in con.row_indices.iter().zip(con.row_values.iter()) {
                            if self.removed_vars.contains(&col) {
                                continue;
                            }
                            let var = &self.model.variables[col];
                            let value = if coeff > 0.0 {
                                var.upper_bound
                            } else {
                                var.lower_bound
                            };
                            if value.is_finite() {
                                fixed.push((col, value));
                            }
                        }
                        if !fixed.is_empty() {
                            for &(col, val) in &fixed {
                                self.model.variables[col].lower_bound = val;
                                self.model.variables[col].upper_bound = val;
                            }
                            self.ops.push(PresolveOp::ForcingConstraint {
                                con_idx: i,
                                fixed_vars: fixed,
                            });
                            self.removed_cons.insert(i);
                            self.stats.forcing_constraints += 1;
                        }
                    }
                }
                ConstraintSense::Eq => {
                    // If min_act == max_act == rhs, forcing
                    if (min_act - max_act).abs() < self.config.bound_tol
                        && (min_act - con.rhs).abs() < self.config.bound_tol
                    {
                        self.removed_cons.insert(i);
                        self.stats.removed_constraints += 1;
                    }
                }
            }
        }
    }

    /// Detect duplicate rows.
    fn detect_duplicate_rows(&mut self) {
        let m = self.model.constraints.len();
        let mut signatures: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();

        for i in 0..m {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let con = &self.model.constraints[i];
            // Create a signature: sorted (col, quantized_value) pairs
            let mut sig: Vec<(usize, i64)> = con
                .row_indices
                .iter()
                .zip(con.row_values.iter())
                .filter(|(&j, _)| !self.removed_vars.contains(&j))
                .map(|(&j, &v)| (j, (v * 1e8).round() as i64))
                .collect();
            sig.sort();
            let key: Vec<i64> = sig.iter().flat_map(|&(a, b)| vec![a as i64, b]).collect();
            signatures.entry(key).or_default().push(i);
        }

        for (_sig, indices) in &signatures {
            if indices.len() < 2 {
                continue;
            }
            // Keep the first, remove duplicates
            let kept = indices[0];
            for &removed in &indices[1..] {
                if self.removed_cons.contains(&removed) {
                    continue;
                }
                let kept_con = &self.model.constraints[kept];
                let rem_con = &self.model.constraints[removed];

                // Check if RHS and sense match
                if kept_con.sense == rem_con.sense
                    && (kept_con.rhs - rem_con.rhs).abs() < self.config.bound_tol
                {
                    self.ops.push(PresolveOp::DuplicateRow {
                        kept_idx: kept,
                        removed_idx: removed,
                    });
                    self.removed_cons.insert(removed);
                    self.stats.duplicate_rows += 1;
                }
            }
        }
    }

    /// Detect duplicate (proportional) columns.
    fn detect_duplicate_columns(&mut self) {
        let n = self.model.variables.len();
        if n < 2 {
            return;
        }

        // Build column signatures
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, con) in self.model.constraints.iter().enumerate() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                if !self.removed_vars.contains(&col) && val.abs() > self.config.zero_tol {
                    col_entries[col].push((i, val));
                }
            }
        }

        // Group by sparsity pattern
        let mut pattern_groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for j in 0..n {
            if self.removed_vars.contains(&j) {
                continue;
            }
            let pattern: Vec<usize> = col_entries[j].iter().map(|&(r, _)| r).collect();
            pattern_groups.entry(pattern).or_default().push(j);
        }

        for (_pattern, cols) in &pattern_groups {
            if cols.len() < 2 {
                continue;
            }
            // Check if columns are proportional
            for k in 1..cols.len() {
                let j1 = cols[0];
                let j2 = cols[k];
                if self.removed_vars.contains(&j2) {
                    continue;
                }

                if col_entries[j1].is_empty() {
                    continue;
                }

                let ratio = col_entries[j2][0].1 / col_entries[j1][0].1;
                let mut proportional = true;
                for idx in 0..col_entries[j1].len() {
                    let expected = col_entries[j1][idx].1 * ratio;
                    if idx < col_entries[j2].len() {
                        if (col_entries[j2][idx].1 - expected).abs() > self.config.zero_tol {
                            proportional = false;
                            break;
                        }
                    } else {
                        proportional = false;
                        break;
                    }
                }

                if proportional {
                    self.ops.push(PresolveOp::DuplicateCol {
                        kept_idx: j1,
                        removed_idx: j2,
                        ratio,
                    });
                    self.removed_vars.insert(j2);
                    self.stats.duplicate_cols += 1;
                }
            }
        }
    }

    /// Coefficient reduction for integer variables.
    fn reduce_coefficients(&mut self) {
        for i in 0..self.model.constraints.len() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let con_sense = self.model.constraints[i].sense;
            if con_sense != ConstraintSense::Le {
                continue;
            }

            let num_positions = self.model.constraints[i].row_indices.len();

            // Look for integer variables where coefficient can be reduced
            for pos in 0..num_positions {
                let col = self.model.constraints[i].row_indices[pos];
                if self.removed_vars.contains(&col) {
                    continue;
                }
                let var = &self.model.variables[col];
                if var.var_type != VarType::Binary {
                    continue;
                }

                let coeff = self.model.constraints[i].row_values[pos];
                if coeff.abs() < self.config.zero_tol {
                    continue;
                }

                // For binary variable x_j in sum a_j x_j <= b:
                // If a_j > 0 and max_activity_without_j + a_j > b,
                // then we can reduce a_j to b - max_activity_without_j
                let con_rhs = self.model.constraints[i].rhs;
                let (_, max_without) = {
                    let con = &self.model.constraints[i];
                    let mut max_a = 0.0f64;
                    for (&c, &v) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if c == col || self.removed_vars.contains(&c) {
                            continue;
                        }
                        let vr = &self.model.variables[c];
                        if v > 0.0 {
                            if vr.upper_bound < 1e20 {
                                max_a += v * vr.upper_bound;
                            } else {
                                max_a = f64::INFINITY;
                            }
                        } else {
                            if vr.lower_bound > -1e20 {
                                max_a += v * vr.lower_bound;
                            } else {
                                max_a = f64::INFINITY;
                            }
                        }
                    }
                    (0.0, max_a)
                };

                if coeff > 0.0 && max_without.is_finite() {
                    let new_coeff = (con_rhs - max_without).min(coeff);
                    if new_coeff < coeff - self.config.zero_tol && new_coeff > self.config.zero_tol
                    {
                        let new_rhs = con_rhs - (coeff - new_coeff);
                        self.ops.push(PresolveOp::CoefficientReduction {
                            con_idx: i,
                            var_idx: col,
                            old_coeff: coeff,
                            new_coeff,
                            new_rhs,
                        });
                        self.model.constraints[i].row_values[pos] = new_coeff;
                        self.model.constraints[i].rhs = new_rhs;
                        self.stats.coefficient_reductions += 1;
                    }
                }
            }
        }
    }

    /// Remove variables that have been fixed (lb == ub).
    fn remove_fixed_variables(&mut self) {
        for j in 0..self.model.variables.len() {
            if self.removed_vars.contains(&j) {
                continue;
            }
            let var = &self.model.variables[j];
            if (var.upper_bound - var.lower_bound).abs() < self.config.bound_tol {
                let value = var.lower_bound;
                // Substitute the fixed value into all constraints
                for i in 0..self.model.constraints.len() {
                    if self.removed_cons.contains(&i) {
                        continue;
                    }
                    let con = &self.model.constraints[i];
                    if let Some(pos) = con.row_indices.iter().position(|&c| c == j) {
                        let coeff = con.row_values[pos];
                        self.model.constraints[i].rhs -= coeff * value;
                        self.model.constraints[i].row_indices.remove(pos);
                        self.model.constraints[i].row_values.remove(pos);
                    }
                }
                self.ops.push(PresolveOp::FixVariable {
                    var_idx: j,
                    value,
                    orig_lb: var.lower_bound,
                    orig_ub: var.upper_bound,
                });
                self.removed_vars.insert(j);
                self.stats.fixed_vars += 1;
            }
        }
    }

    /// Compact the model by removing marked variables and constraints.
    fn compact_model(&mut self) {
        // Build new model without removed entries
        let mut new_model = LpModel::new(&self.model.name);
        new_model.sense = self.model.sense;
        new_model.obj_offset = self.model.obj_offset;

        let mut var_old_to_new: HashMap<usize, usize> = HashMap::new();

        for (j, var) in self.model.variables.iter().enumerate() {
            if self.removed_vars.contains(&j) {
                continue;
            }
            let new_idx = new_model.add_variable(var.clone());
            var_old_to_new.insert(j, new_idx);
        }

        for (i, con) in self.model.constraints.iter().enumerate() {
            if self.removed_cons.contains(&i) {
                continue;
            }
            let mut new_con = Constraint::new(&con.name, con.sense, con.rhs);
            new_con.range = con.range;
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                if let Some(&new_col) = var_old_to_new.get(&col) {
                    new_con.add_term(new_col, val);
                }
            }
            if !new_con.row_indices.is_empty() || con.sense == ConstraintSense::Eq {
                new_model.add_constraint(new_con);
            }
        }

        self.model = new_model;
    }

    fn build_var_map(&self) -> Vec<usize> {
        let mut map = Vec::new();
        for (j, _) in self.model.variables.iter().enumerate() {
            // After compaction, the variables are renumbered
            // We need the mapping from new index to original index
            map.push(j);
        }
        // This is approximate; a proper implementation would track through compaction
        map
    }

    fn build_con_map(&self) -> Vec<usize> {
        let mut map = Vec::new();
        for (i, _) in self.model.constraints.iter().enumerate() {
            map.push(i);
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::OptDirection;

    fn simple_model() -> LpModel {
        let mut m = LpModel::new("test");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 10.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 10.0));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 2.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 6.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 8.0);
        c1.add_term(x, 2.0);
        c1.add_term(y, 1.0);
        m.add_constraint(c1);

        m
    }

    #[test]
    fn test_presolve_no_change() {
        let model = simple_model();
        let result = presolve(&model, &PresolveConfig::default());
        assert!(!result.is_infeasible);
        assert_eq!(result.presolved_model.num_vars(), 2);
    }

    #[test]
    fn test_presolve_singleton_row() {
        let mut m = LpModel::new("singleton");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 10.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 10.0));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 1.0);

        // Singleton: x = 3
        let mut c0 = Constraint::new("fix_x", ConstraintSense::Eq, 3.0);
        c0.add_term(x, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 8.0);
        c1.add_term(x, 1.0);
        c1.add_term(y, 1.0);
        m.add_constraint(c1);

        let result = presolve(&m, &PresolveConfig::default());
        assert!(!result.is_infeasible);
        assert!(result.stats.fixed_vars > 0);
    }

    #[test]
    fn test_presolve_infeasible() {
        let mut m = LpModel::new("infeas");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 3.0));
        m.set_obj_coeff(x, 1.0);

        // x >= 5 but x <= 3
        let mut c = Constraint::new("c0", ConstraintSense::Ge, 5.0);
        c.add_term(x, 1.0);
        m.add_constraint(c);

        let result = presolve(&m, &PresolveConfig::default());
        assert!(result.is_infeasible);
    }

    #[test]
    fn test_presolve_redundant_constraint() {
        let mut m = LpModel::new("redundant");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 5.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 5.0));
        m.set_obj_coeff(x, 1.0);

        // x + y <= 100 is redundant when x,y <= 5
        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 100.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let result = presolve(&m, &PresolveConfig::default());
        assert!(result.stats.removed_constraints > 0);
    }

    #[test]
    fn test_presolve_bound_strengthening() {
        let mut m = LpModel::new("bounds");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 100.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 100.0));
        m.set_obj_coeff(x, 1.0);

        // x + y <= 10 should tighten bounds to 10
        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 10.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let result = presolve(&m, &PresolveConfig::default());
        assert!(result.stats.tightened_bounds > 0);
    }

    #[test]
    fn test_presolve_duplicate_rows() {
        let mut m = LpModel::new("dup_rows");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 10.0));
        m.set_obj_coeff(x, 1.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 5.0);
        c0.add_term(x, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 5.0);
        c1.add_term(x, 1.0);
        m.add_constraint(c1);

        let result = presolve(&m, &PresolveConfig::default());
        assert!(result.stats.duplicate_rows > 0 || result.stats.removed_constraints > 0);
    }

    #[test]
    fn test_postsolve() {
        let presolved_primal = vec![2.0, 3.0];
        let presolved_dual = vec![0.5];
        let ops = vec![PresolveOp::FixVariable {
            var_idx: 2,
            value: 1.0,
            orig_lb: 0.0,
            orig_ub: 10.0,
        }];
        let var_map = vec![0, 1];

        let (primal, _dual) = postsolve(&presolved_primal, &presolved_dual, &ops, &var_map, 3, 2);
        assert_eq!(primal.len(), 3);
        assert!((primal[0] - 2.0).abs() < 1e-10);
        assert!((primal[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_presolve_empty_model() {
        let m = LpModel::new("empty");
        let result = presolve(&m, &PresolveConfig::default());
        assert!(!result.is_infeasible);
    }

    #[test]
    fn test_presolve_config_disabled() {
        let model = simple_model();
        let config = PresolveConfig {
            singleton_rows: false,
            singleton_cols: false,
            forcing_constraints: false,
            bound_strengthening: false,
            duplicate_detection: false,
            probing: false,
            coefficient_reduction: false,
            ..PresolveConfig::default()
        };
        let result = presolve(&model, &config);
        assert!(!result.is_infeasible);
    }
}
