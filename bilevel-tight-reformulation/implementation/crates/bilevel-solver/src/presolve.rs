//! Presolve routines for LP/MIP simplification.

use crate::model::SolverModel;
use bilevel_types::ConstraintSense;
use serde::{Deserialize, Serialize};
use log::{debug, info, trace};

/// Presolve configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresolveConfig {
    pub max_rounds: u32,
    pub enable_singleton_row: bool,
    pub enable_singleton_col: bool,
    pub enable_implied_bounds: bool,
    pub enable_fixed_vars: bool,
    pub enable_redundant_constraints: bool,
    pub enable_dominated_columns: bool,
    pub tolerance: f64,
}

impl Default for PresolveConfig {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            enable_singleton_row: true,
            enable_singleton_col: true,
            enable_implied_bounds: true,
            enable_fixed_vars: true,
            enable_redundant_constraints: true,
            enable_dominated_columns: true,
            tolerance: 1e-8,
        }
    }
}

/// Result of presolve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresolveResult {
    pub variables_fixed: usize,
    pub constraints_removed: usize,
    pub bounds_tightened: usize,
    pub rounds_performed: u32,
    pub reductions: Vec<PresolveReduction>,
}

/// A single presolve reduction for postsolve recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresolveReduction {
    FixVariable { var_index: usize, value: f64 },
    RemoveConstraint { con_index: usize },
    TightenLower { var_index: usize, old_lb: f64, new_lb: f64 },
    TightenUpper { var_index: usize, old_ub: f64, new_ub: f64 },
    SubstituteVariable { var_index: usize, con_index: usize, coeff: f64, rhs: f64 },
    RemoveRedundant { con_index: usize },
}

/// Postsolve information for recovering original solution.
#[derive(Debug, Clone)]
pub struct PostsolveInfo {
    reductions: Vec<PresolveReduction>,
    original_num_vars: usize,
    original_num_cons: usize,
}

impl PostsolveInfo {
    pub fn new(reductions: Vec<PresolveReduction>, num_vars: usize, num_cons: usize) -> Self {
        Self {
            reductions,
            original_num_vars: num_vars,
            original_num_cons: num_cons,
        }
    }

    pub fn recover_solution(&self, presolved_values: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.original_num_vars];
        let n_presolved = presolved_values.len().min(self.original_num_vars);
        for i in 0..n_presolved {
            values[i] = presolved_values[i];
        }

        for reduction in self.reductions.iter().rev() {
            match reduction {
                PresolveReduction::FixVariable { var_index, value } => {
                    if *var_index < values.len() {
                        values[*var_index] = *value;
                    }
                }
                PresolveReduction::SubstituteVariable { var_index, con_index: _, coeff, rhs } => {
                    if *var_index < values.len() && coeff.abs() > 1e-15 {
                        values[*var_index] = rhs / coeff;
                    }
                }
                _ => {}
            }
        }
        values
    }
}

/// The presolver.
#[derive(Debug)]
pub struct Presolver {
    config: PresolveConfig,
    reductions: Vec<PresolveReduction>,
    removed_constraints: Vec<bool>,
    fixed_variables: Vec<bool>,
}

impl Presolver {
    pub fn new() -> Self {
        Self::with_config(PresolveConfig::default())
    }

    pub fn with_config(config: PresolveConfig) -> Self {
        Self {
            config,
            reductions: Vec::new(),
            removed_constraints: Vec::new(),
            fixed_variables: Vec::new(),
        }
    }

    pub fn presolve(&mut self, model: &mut SolverModel) -> PresolveResult {
        let n = model.num_variables();
        let m = model.num_constraints();
        self.removed_constraints = vec![false; m];
        self.fixed_variables = vec![false; n];
        self.reductions.clear();

        let mut total_fixed = 0usize;
        let mut total_removed = 0usize;
        let mut total_tightened = 0usize;

        for round in 0..self.config.max_rounds {
            let mut changed = false;

            if self.config.enable_fixed_vars {
                let fixed = self.detect_fixed_variables(model);
                if fixed > 0 {
                    changed = true;
                    total_fixed += fixed;
                }
            }

            if self.config.enable_singleton_row {
                let removed = self.detect_singleton_rows(model);
                if removed > 0 {
                    changed = true;
                    total_removed += removed;
                }
            }

            if self.config.enable_implied_bounds {
                let tightened = self.propagate_implied_bounds(model);
                if tightened > 0 {
                    changed = true;
                    total_tightened += tightened;
                }
            }

            if self.config.enable_redundant_constraints {
                let removed = self.detect_redundant_constraints(model);
                if removed > 0 {
                    changed = true;
                    total_removed += removed;
                }
            }

            if !changed {
                break;
            }
        }

        PresolveResult {
            variables_fixed: total_fixed,
            constraints_removed: total_removed,
            bounds_tightened: total_tightened,
            rounds_performed: self.config.max_rounds,
            reductions: self.reductions.clone(),
        }
    }

    fn detect_fixed_variables(&mut self, model: &mut SolverModel) -> usize {
        let mut count = 0;
        for i in 0..model.variables.len() {
            if self.fixed_variables[i] {
                continue;
            }
            let lb = model.variables[i].lower;
            let ub = model.variables[i].upper;
            if (ub - lb).abs() < self.config.tolerance {
                let value = (lb + ub) / 2.0;
                self.fixed_variables[i] = true;
                self.reductions.push(PresolveReduction::FixVariable {
                    var_index: i,
                    value,
                });
                model.variables[i].lower = value;
                model.variables[i].upper = value;
                count += 1;
            }
        }
        count
    }

    fn detect_singleton_rows(&mut self, model: &mut SolverModel) -> usize {
        let mut count = 0;
        for c in 0..model.constraints.len() {
            if self.removed_constraints[c] {
                continue;
            }
            let con = &model.constraints[c];
            let active_count = con.coefficients.iter()
                .filter(|(idx, _)| idx.raw() < model.variables.len() && !self.fixed_variables[idx.raw()])
                .count();
            if active_count == 1 {
                if let Some((col, coeff)) = con.coefficients.iter()
                    .find(|(idx, _)| idx.raw() < model.variables.len() && !self.fixed_variables[idx.raw()])
                    .map(|(idx, c)| (*idx, *c))
                {
                    if coeff.abs() > self.config.tolerance {
                        let bound_val = con.rhs / coeff;
                        let col_raw = col.raw();
                        let var = &mut model.variables[col_raw];
                        let sense = con.sense;
                        match sense {
                            ConstraintSense::Le => {
                                if coeff > 0.0 {
                                    let new_ub = bound_val;
                                    if new_ub < var.upper - self.config.tolerance {
                                        let old_ub = var.upper;
                                        var.upper = new_ub;
                                        self.reductions.push(PresolveReduction::TightenUpper {
                                            var_index: col_raw, old_ub, new_ub,
                                        });
                                    }
                                } else {
                                    let new_lb = bound_val;
                                    if new_lb > var.lower + self.config.tolerance {
                                        let old_lb = var.lower;
                                        var.lower = new_lb;
                                        self.reductions.push(PresolveReduction::TightenLower {
                                            var_index: col_raw, old_lb, new_lb,
                                        });
                                    }
                                }
                            }
                            ConstraintSense::Ge => {
                                if coeff > 0.0 {
                                    let new_lb = bound_val;
                                    if new_lb > var.lower + self.config.tolerance {
                                        let old_lb = var.lower;
                                        var.lower = new_lb;
                                        self.reductions.push(PresolveReduction::TightenLower {
                                            var_index: col_raw, old_lb, new_lb,
                                        });
                                    }
                                } else {
                                    let new_ub = bound_val;
                                    if new_ub < var.upper - self.config.tolerance {
                                        let old_ub = var.upper;
                                        var.upper = new_ub;
                                        self.reductions.push(PresolveReduction::TightenUpper {
                                            var_index: col_raw, old_ub, new_ub,
                                        });
                                    }
                                }
                            }
                            ConstraintSense::Eq => {
                                let value = bound_val;
                                self.fixed_variables[col_raw] = true;
                                model.variables[col_raw].lower = value;
                                model.variables[col_raw].upper = value;
                                self.reductions.push(PresolveReduction::SubstituteVariable {
                                    var_index: col_raw, con_index: c, coeff, rhs: con.rhs,
                                });
                            }
                        }
                        self.removed_constraints[c] = true;
                        self.reductions.push(PresolveReduction::RemoveConstraint { con_index: c });
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn propagate_implied_bounds(&mut self, model: &mut SolverModel) -> usize {
        let mut count = 0;
        for c in 0..model.constraints.len() {
            if self.removed_constraints[c] {
                continue;
            }
            let con = &model.constraints[c];
            if con.sense != ConstraintSense::Le {
                continue;
            }

            for k in 0..con.coefficients.len() {
                let var_idx = con.coefficients[k].0.raw();
                if var_idx >= model.variables.len() || self.fixed_variables[var_idx] {
                    continue;
                }
                let a_k = con.coefficients[k].1;
                if a_k.abs() < self.config.tolerance {
                    continue;
                }

                let mut sum_min = 0.0;
                let mut sum_max = 0.0;
                let mut computable = true;

                for j in 0..con.coefficients.len() {
                    if j == k {
                        continue;
                    }
                    let idx = con.coefficients[j].0.raw();
                    if idx >= model.variables.len() {
                        continue;
                    }
                    let a_j = con.coefficients[j].1;
                    let lb = model.variables[idx].lower;
                    let ub = model.variables[idx].upper;
                    if !lb.is_finite() || !ub.is_finite() {
                        computable = false;
                        break;
                    }
                    if a_j > 0.0 {
                        sum_min += a_j * lb;
                        sum_max += a_j * ub;
                    } else {
                        sum_min += a_j * ub;
                        sum_max += a_j * lb;
                    }
                }

                if !computable {
                    continue;
                }

                let rhs = con.rhs;
                if a_k > self.config.tolerance {
                    let implied_ub = (rhs - sum_min) / a_k;
                    if implied_ub < model.variables[var_idx].upper - self.config.tolerance {
                        let old_ub = model.variables[var_idx].upper;
                        model.variables[var_idx].upper = implied_ub;
                        self.reductions.push(PresolveReduction::TightenUpper {
                            var_index: var_idx, old_ub, new_ub: implied_ub,
                        });
                        count += 1;
                    }
                    let implied_lb = (rhs - sum_max) / a_k;
                    if implied_lb > model.variables[var_idx].lower + self.config.tolerance {
                        let old_lb = model.variables[var_idx].lower;
                        model.variables[var_idx].lower = implied_lb;
                        self.reductions.push(PresolveReduction::TightenLower {
                            var_index: var_idx, old_lb, new_lb: implied_lb,
                        });
                        count += 1;
                    }
                } else if a_k < -self.config.tolerance {
                    let implied_lb = (rhs - sum_min) / a_k;
                    if implied_lb > model.variables[var_idx].lower + self.config.tolerance {
                        let old_lb = model.variables[var_idx].lower;
                        model.variables[var_idx].lower = implied_lb;
                        self.reductions.push(PresolveReduction::TightenLower {
                            var_index: var_idx, old_lb, new_lb: implied_lb,
                        });
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn detect_redundant_constraints(&mut self, model: &SolverModel) -> usize {
        let mut count = 0;
        for c in 0..model.constraints.len() {
            if self.removed_constraints[c] {
                continue;
            }
            let con = &model.constraints[c];
            if con.sense != ConstraintSense::Le {
                continue;
            }

            let mut max_lhs = 0.0;
            let mut computable = true;
            for (idx, coeff) in con.coefficients.iter().map(|(v, c)| (v.raw(), *c)) {
                if idx >= model.variables.len() {
                    continue;
                }
                let lb = model.variables[idx].lower;
                let ub = model.variables[idx].upper;
                if !lb.is_finite() || !ub.is_finite() {
                    computable = false;
                    break;
                }
                if coeff > 0.0 {
                    max_lhs += coeff * ub;
                } else {
                    max_lhs += coeff * lb;
                }
            }

            if computable && max_lhs <= con.rhs + self.config.tolerance {
                self.removed_constraints[c] = true;
                self.reductions.push(PresolveReduction::RemoveRedundant { con_index: c });
                count += 1;
            }
        }
        count
    }

    pub fn postsolve_info(&self, model: &SolverModel) -> PostsolveInfo {
        PostsolveInfo::new(
            self.reductions.clone(),
            model.num_variables(),
            model.num_constraints(),
        )
    }
}

impl Default for Presolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate presolve statistics across multiple rounds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PresolveStatistics {
    pub total_variables_fixed: usize,
    pub total_constraints_removed: usize,
    pub total_bounds_tightened: usize,
    pub total_rounds: u32,
    pub per_round: Vec<RoundStatistics>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoundStatistics {
    pub round: u32,
    pub variables_fixed: usize,
    pub constraints_removed: usize,
    pub bounds_tightened: usize,
}

impl PresolveStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_round(&mut self, result: &PresolveResult) {
        self.total_variables_fixed += result.variables_fixed;
        self.total_constraints_removed += result.constraints_removed;
        self.total_bounds_tightened += result.bounds_tightened;
        self.total_rounds += 1;
        self.per_round.push(RoundStatistics {
            round: self.total_rounds,
            variables_fixed: result.variables_fixed,
            constraints_removed: result.constraints_removed,
            bounds_tightened: result.bounds_tightened,
        });
    }

    pub fn has_reductions(&self) -> bool {
        self.total_variables_fixed > 0
            || self.total_constraints_removed > 0
            || self.total_bounds_tightened > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SolverModel;
use bilevel_types::ConstraintSense;

    fn make_test_model() -> SolverModel {
        let mut model = SolverModel::new();
        model.add_variable("x1", 0.0, 10.0, 1.0, false);
        model.add_variable("x2", 0.0, 10.0, 2.0, false);
        model.add_variable("x3", 5.0, 5.0, 3.0, false); // fixed variable
        model.add_constraint("c1", vec![0, 1], vec![1.0, 1.0], "<=", 20.0); // redundant
        model.add_constraint("c2", vec![0], vec![1.0], "<=", 8.0);
        model.add_constraint("c3", vec![0, 1], vec![1.0, 2.0], "<=", 12.0);
        model.set_minimize(true);
        model
    }

    #[test]
    fn test_presolve_config() {
        let config = PresolveConfig::default();
        assert!(config.max_rounds > 0);
        assert!(config.enable_singleton_row);
    }

    #[test]
    fn test_detect_fixed_variables() {
        let mut model = make_test_model();
        let mut presolver = Presolver::new();
        presolver.removed_constraints = vec![false; model.num_constraints()];
        presolver.fixed_variables = vec![false; model.num_variables()];
        let count = presolver.detect_fixed_variables(&mut model);
        assert_eq!(count, 1); // x3 is fixed
    }

    #[test]
    fn test_presolve_full() {
        let mut model = make_test_model();
        let mut presolver = Presolver::new();
        let result = presolver.presolve(&mut model);
        assert!(result.variables_fixed >= 1);
    }

    #[test]
    fn test_postsolve_recovery() {
        let info = PostsolveInfo::new(
            vec![PresolveReduction::FixVariable { var_index: 2, value: 5.0 }],
            3, 2,
        );
        let presolved = vec![1.0, 2.0, 0.0];
        let recovered = info.recover_solution(&presolved);
        assert_eq!(recovered.len(), 3);
        assert!((recovered[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_presolve_statistics() {
        let mut stats = PresolveStatistics::new();
        let result = PresolveResult {
            variables_fixed: 3,
            constraints_removed: 2,
            bounds_tightened: 5,
            rounds_performed: 1,
            reductions: Vec::new(),
        };
        stats.add_round(&result);
        assert!(stats.has_reductions());
        assert_eq!(stats.total_variables_fixed, 3);
    }

    #[test]
    fn test_detect_redundant() {
        let mut model = SolverModel::new();
        model.add_variable("x", 0.0, 5.0, 1.0, false);
        model.add_constraint("c1", vec![0], vec![1.0], "<=", 100.0); // redundant
        model.set_minimize(true);

        let mut presolver = Presolver::new();
        presolver.removed_constraints = vec![false; 1];
        presolver.fixed_variables = vec![false; 1];
        let count = presolver.detect_redundant_constraints(&model);
        assert_eq!(count, 1);
    }
}
