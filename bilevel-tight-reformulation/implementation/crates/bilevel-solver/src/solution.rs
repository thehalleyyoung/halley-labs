//! Solution extraction: primal values, dual values, reduced costs, basis status, sensitivity.

use std::fmt;
use serde::{Deserialize, Serialize};
use bilevel_types::{VarIdx, ConIdx};

/// Status of a variable or constraint in the basis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Superbasic,
    Fixed,
}

impl Default for BasisStatus {
    fn default() -> Self { BasisStatus::AtLower }
}

impl fmt::Display for BasisStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Basic => write!(f, "BS"),
            Self::AtLower => write!(f, "LB"),
            Self::AtUpper => write!(f, "UB"),
            Self::Superbasic => write!(f, "SB"),
            Self::Fixed => write!(f, "FX"),
        }
    }
}

/// Sensitivity analysis range for a coefficient
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensitivityRange {
    pub current: f64,
    pub lower: f64,
    pub upper: f64,
}

impl SensitivityRange {
    pub fn new(current: f64, lower: f64, upper: f64) -> Self {
        SensitivityRange { current, lower, upper }
    }

    pub fn allowable_decrease(&self) -> f64 { self.current - self.lower }
    pub fn allowable_increase(&self) -> f64 { self.upper - self.current }

    pub fn is_degenerate(&self) -> bool {
        (self.upper - self.lower).abs() < 1e-10
    }
}

impl fmt::Display for SensitivityRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6}, {:.6}] (current: {:.6})", self.lower, self.upper, self.current)
    }
}

/// Complete solution from a solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub num_variables: usize,
    pub num_constraints: usize,
    primal: Vec<f64>,
    dual: Vec<f64>,
    reduced_costs: Vec<f64>,
    slacks: Vec<f64>,
    var_basis: Vec<BasisStatus>,
    con_basis: Vec<BasisStatus>,
    pub objective_value: f64,
    obj_sensitivity: Vec<SensitivityRange>,
    rhs_sensitivity: Vec<SensitivityRange>,
}

impl Solution {
    pub fn new(
        num_variables: usize,
        num_constraints: usize,
        primal: Vec<f64>,
        dual: Vec<f64>,
        reduced_costs: Vec<f64>,
        slacks: Vec<f64>,
        var_basis: Vec<BasisStatus>,
        con_basis: Vec<BasisStatus>,
        objective_value: f64,
    ) -> Self {
        Solution {
            num_variables, num_constraints, primal, dual, reduced_costs,
            slacks, var_basis, con_basis, objective_value,
            obj_sensitivity: Vec::new(), rhs_sensitivity: Vec::new(),
        }
    }

    pub fn from_primal(primal: Vec<f64>, objective_value: f64) -> Self {
        let n = primal.len();
        Solution {
            num_variables: n, num_constraints: 0, primal,
            dual: Vec::new(), reduced_costs: vec![0.0; n],
            slacks: Vec::new(), var_basis: vec![BasisStatus::Basic; n],
            con_basis: Vec::new(), objective_value,
            obj_sensitivity: Vec::new(), rhs_sensitivity: Vec::new(),
        }
    }

    pub fn primal_values(&self) -> &[f64] { &self.primal }
    pub fn primal_value(&self, var: VarIdx) -> f64 { self.primal[var] }
    pub fn dual_values(&self) -> &[f64] { &self.dual }
    pub fn dual_value(&self, con: ConIdx) -> f64 { self.dual[con] }
    pub fn reduced_costs(&self) -> &[f64] { &self.reduced_costs }
    pub fn reduced_cost(&self, var: VarIdx) -> f64 { self.reduced_costs[var] }
    pub fn slacks(&self) -> &[f64] { &self.slacks }
    pub fn slack(&self, con: ConIdx) -> f64 { self.slacks[con] }
    pub fn var_basis_statuses(&self) -> &[BasisStatus] { &self.var_basis }
    pub fn var_basis_status(&self, var: VarIdx) -> BasisStatus { self.var_basis[var] }
    pub fn con_basis_statuses(&self) -> &[BasisStatus] { &self.con_basis }
    pub fn con_basis_status(&self, con: ConIdx) -> BasisStatus { self.con_basis[con] }

    pub fn set_obj_sensitivity(&mut self, ranges: Vec<SensitivityRange>) {
        self.obj_sensitivity = ranges;
    }

    pub fn set_rhs_sensitivity(&mut self, ranges: Vec<SensitivityRange>) {
        self.rhs_sensitivity = ranges;
    }

    pub fn obj_sensitivity(&self) -> &[SensitivityRange] { &self.obj_sensitivity }
    pub fn rhs_sensitivity(&self) -> &[SensitivityRange] { &self.rhs_sensitivity }

    pub fn is_basic(&self, var: VarIdx) -> bool {
        self.var_basis[var] == BasisStatus::Basic
    }

    pub fn basic_variables(&self) -> Vec<VarIdx> {
        self.var_basis.iter().enumerate()
            .filter(|(_, &s)| s == BasisStatus::Basic)
            .map(|(i, _)| VarIdx::new(i)).collect()
    }

    pub fn nonbasic_variables(&self) -> Vec<VarIdx> {
        self.var_basis.iter().enumerate()
            .filter(|(_, &s)| s != BasisStatus::Basic)
            .map(|(i, _)| VarIdx::new(i)).collect()
    }

    pub fn compute_activities(&self, a_matrix: &[Vec<(usize, f64)>]) -> Vec<f64> {
        let mut activities = vec![0.0; a_matrix.len()];
        for (i, row) in a_matrix.iter().enumerate() {
            for &(j, coeff) in row {
                if j < self.primal.len() {
                    activities[i] += coeff * self.primal[j];
                }
            }
        }
        activities
    }

    pub fn check_primal_feasibility(
        &self, activities: &[f64], rhs: &[f64],
        senses: &[bilevel_types::ConstraintSense], tol: f64,
    ) -> bool {
        for i in 0..activities.len().min(rhs.len()) {
            let violation = match senses[i] {
                bilevel_types::ConstraintSense::Le => (activities[i] - rhs[i]).max(0.0),
                bilevel_types::ConstraintSense::Ge => (rhs[i] - activities[i]).max(0.0),
                bilevel_types::ConstraintSense::Eq => (activities[i] - rhs[i]).abs(),
            };
            if violation > tol { return false; }
        }
        true
    }

    pub fn check_bound_feasibility(&self, lower: &[f64], upper: &[f64], tol: f64) -> bool {
        for i in 0..self.primal.len() {
            if self.primal[i] < lower[i] - tol || self.primal[i] > upper[i] + tol {
                return false;
            }
        }
        true
    }

    pub fn max_primal_infeasibility(
        &self, activities: &[f64], rhs: &[f64],
        senses: &[bilevel_types::ConstraintSense],
    ) -> f64 {
        let mut max_viol = 0.0f64;
        for i in 0..activities.len().min(rhs.len()) {
            let violation = match senses[i] {
                bilevel_types::ConstraintSense::Le => (activities[i] - rhs[i]).max(0.0),
                bilevel_types::ConstraintSense::Ge => (rhs[i] - activities[i]).max(0.0),
                bilevel_types::ConstraintSense::Eq => (activities[i] - rhs[i]).abs(),
            };
            max_viol = max_viol.max(violation);
        }
        max_viol
    }

    pub fn max_dual_infeasibility(&self) -> f64 {
        let mut max_viol = 0.0f64;
        for i in 0..self.reduced_costs.len() {
            let viol = match self.var_basis.get(i).copied().unwrap_or(BasisStatus::AtLower) {
                BasisStatus::AtLower => (-self.reduced_costs[i]).max(0.0),
                BasisStatus::AtUpper => self.reduced_costs[i].max(0.0),
                BasisStatus::Basic | BasisStatus::Superbasic => self.reduced_costs[i].abs(),
                BasisStatus::Fixed => 0.0,
            };
            max_viol = max_viol.max(viol);
        }
        max_viol
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Solution (objective = {:.10}):", self.objective_value)?;
        for i in 0..self.primal.len().min(20) {
            write!(f, "  x[{}] = {:>12.6}", i, self.primal[i])?;
            if i < self.reduced_costs.len() {
                write!(f, "  rc = {:>12.6}", self.reduced_costs[i])?;
            }
            if i < self.var_basis.len() {
                write!(f, "  [{}]", self.var_basis[i])?;
            }
            writeln!(f)?;
        }
        if self.primal.len() > 20 {
            writeln!(f, "  ... ({} more)", self.primal.len() - 20)?;
        }
        Ok(())
    }
}

/// Builder for constructing solutions incrementally
#[derive(Debug, Clone)]
pub struct SolutionBuilder {
    num_variables: usize,
    num_constraints: usize,
    primal: Vec<f64>,
    dual: Vec<f64>,
    reduced_costs: Vec<f64>,
    slacks: Vec<f64>,
    var_basis: Vec<BasisStatus>,
    con_basis: Vec<BasisStatus>,
    objective_value: f64,
}

impl SolutionBuilder {
    pub fn new(num_variables: usize, num_constraints: usize) -> Self {
        SolutionBuilder {
            num_variables, num_constraints,
            primal: vec![0.0; num_variables],
            dual: vec![0.0; num_constraints],
            reduced_costs: vec![0.0; num_variables],
            slacks: vec![0.0; num_constraints],
            var_basis: vec![BasisStatus::AtLower; num_variables],
            con_basis: vec![BasisStatus::Basic; num_constraints],
            objective_value: 0.0,
        }
    }

    pub fn set_primal(mut self, v: Vec<f64>) -> Self { self.primal = v; self }
    pub fn set_dual(mut self, v: Vec<f64>) -> Self { self.dual = v; self }
    pub fn set_reduced_costs(mut self, v: Vec<f64>) -> Self { self.reduced_costs = v; self }
    pub fn set_slacks(mut self, v: Vec<f64>) -> Self { self.slacks = v; self }
    pub fn set_var_basis(mut self, v: Vec<BasisStatus>) -> Self { self.var_basis = v; self }
    pub fn set_con_basis(mut self, v: Vec<BasisStatus>) -> Self { self.con_basis = v; self }
    pub fn set_objective(mut self, v: f64) -> Self { self.objective_value = v; self }

    pub fn build(self) -> Solution {
        Solution::new(
            self.num_variables, self.num_constraints,
            self.primal, self.dual, self.reduced_costs, self.slacks,
            self.var_basis, self.con_basis, self.objective_value,
        )
    }
}

/// Compute objective coefficient sensitivity ranges
pub fn compute_obj_sensitivity(
    basic_vars: &[usize], nonbasic_vars: &[usize],
    reduced_costs: &[f64], basis_inverse_cols: &[Vec<f64>],
    obj_coeffs: &[f64], num_vars: usize,
) -> Vec<SensitivityRange> {
    let mut ranges = Vec::with_capacity(num_vars);
    for j in 0..num_vars {
        let is_basic = basic_vars.contains(&j);
        if is_basic {
            let mut min_inc = f64::INFINITY;
            let mut max_dec = f64::NEG_INFINITY;
            if let Some(basis_pos) = basic_vars.iter().position(|&v| v == j) {
                for &k in nonbasic_vars {
                    if k < basis_inverse_cols.len() && basis_pos < basis_inverse_cols[k].len() {
                        let a_entry = basis_inverse_cols[k][basis_pos];
                        if a_entry.abs() > 1e-12 {
                            let ratio = -reduced_costs[k] / a_entry;
                            if a_entry > 0.0 { min_inc = min_inc.min(ratio); }
                            else { max_dec = max_dec.max(ratio); }
                        }
                    }
                }
            }
            ranges.push(SensitivityRange::new(
                obj_coeffs[j], obj_coeffs[j] + max_dec, obj_coeffs[j] + min_inc));
        } else {
            let rc = if j < reduced_costs.len() { reduced_costs[j] } else { 0.0 };
            ranges.push(SensitivityRange::new(
                obj_coeffs[j], obj_coeffs[j] - rc.abs(), obj_coeffs[j] + rc.abs()));
        }
    }
    ranges
}

/// Compute RHS sensitivity ranges
pub fn compute_rhs_sensitivity(
    basic_vars: &[usize], basic_values: &[f64],
    lower_bounds: &[f64], upper_bounds: &[f64],
    basis_inverse_row: &[Vec<f64>], rhs: &[f64], num_cons: usize,
) -> Vec<SensitivityRange> {
    let mut ranges = Vec::with_capacity(num_cons);
    for i in 0..num_cons {
        let mut min_inc = f64::INFINITY;
        let mut max_dec = f64::NEG_INFINITY;
        if i < basis_inverse_row.len() {
            for (k, &bv) in basic_vars.iter().enumerate() {
                if k < basis_inverse_row[i].len() {
                    let beta = basis_inverse_row[i][k];
                    if beta.abs() > 1e-12 {
                        let val = basic_values[k];
                        let lb = if bv < lower_bounds.len() { lower_bounds[bv] } else { 0.0 };
                        let ub = if bv < upper_bounds.len() { upper_bounds[bv] } else { f64::INFINITY };
                        if beta > 0.0 {
                            min_inc = min_inc.min((ub - val) / beta);
                            max_dec = max_dec.max((lb - val) / beta);
                        } else {
                            min_inc = min_inc.min((lb - val) / beta);
                            max_dec = max_dec.max((ub - val) / beta);
                        }
                    }
                }
            }
        }
        ranges.push(SensitivityRange::new(rhs[i], rhs[i] + max_dec, rhs[i] + min_inc));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_status_display() {
        assert_eq!(format!("{}", BasisStatus::Basic), "BS");
        assert_eq!(format!("{}", BasisStatus::AtLower), "LB");
    }

    #[test]
    fn test_sensitivity_range() {
        let range = SensitivityRange::new(5.0, 3.0, 8.0);
        assert_eq!(range.allowable_decrease(), 2.0);
        assert_eq!(range.allowable_increase(), 3.0);
        assert!(!range.is_degenerate());
    }

    #[test]
    fn test_solution_builder() {
        let sol = SolutionBuilder::new(3, 2)
            .set_primal(vec![1.0, 2.0, 3.0])
            .set_dual(vec![0.5, 0.0])
            .set_objective(10.0)
            .build();
        assert_eq!(sol.primal_value(0), 1.0);
        assert_eq!(sol.dual_value(0), 0.5);
        assert_eq!(sol.objective_value, 10.0);
    }

    #[test]
    fn test_solution_from_primal() {
        let sol = Solution::from_primal(vec![1.0, 2.0], 5.0);
        assert_eq!(sol.num_variables, 2);
        assert_eq!(sol.objective_value, 5.0);
    }

    #[test]
    fn test_basic_nonbasic_variables() {
        let sol = SolutionBuilder::new(4, 2)
            .set_var_basis(vec![BasisStatus::Basic, BasisStatus::AtLower, BasisStatus::Basic, BasisStatus::AtUpper])
            .build();
        assert_eq!(sol.basic_variables(), vec![0, 2]);
        assert_eq!(sol.nonbasic_variables(), vec![1, 3]);
    }

    #[test]
    fn test_feasibility_check() {
        let sol = SolutionBuilder::new(2, 1).set_primal(vec![3.0, 4.0]).build();
        let activities = sol.compute_activities(&[vec![(0, 1.0), (1, 1.0)]]);
        assert_eq!(activities, vec![7.0]);
        assert!(sol.check_primal_feasibility(&activities, &[10.0],
            &[bilevel_types::ConstraintSense::Le], 1e-8));
        assert!(!sol.check_primal_feasibility(&activities, &[5.0],
            &[bilevel_types::ConstraintSense::Le], 1e-8));
    }
}
