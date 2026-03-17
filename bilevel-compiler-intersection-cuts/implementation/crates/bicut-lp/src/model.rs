//! LP model representation for the BiCut LP solver.
//!
//! Provides the core data structures for representing linear programs:
//! variable bounds, objective function, constraint matrix, modification API,
//! model statistics, and variable/constraint naming.

use bicut_types::{ConstraintSense, OptDirection, SparseMatrix, VarBound};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Variable information within the LP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub obj_coeff: f64,
    pub var_type: VarType,
    pub index: usize,
}

impl Variable {
    pub fn continuous(name: &str, lb: f64, ub: f64) -> Self {
        Self {
            name: name.to_string(),
            lower_bound: lb,
            upper_bound: ub,
            obj_coeff: 0.0,
            var_type: VarType::Continuous,
            index: 0,
        }
    }

    pub fn binary(name: &str) -> Self {
        Self {
            name: name.to_string(),
            lower_bound: 0.0,
            upper_bound: 1.0,
            obj_coeff: 0.0,
            var_type: VarType::Binary,
            index: 0,
        }
    }

    pub fn integer(name: &str, lb: f64, ub: f64) -> Self {
        Self {
            name: name.to_string(),
            lower_bound: lb,
            upper_bound: ub,
            obj_coeff: 0.0,
            var_type: VarType::Integer,
            index: 0,
        }
    }

    pub fn is_fixed(&self) -> bool {
        (self.upper_bound - self.lower_bound).abs() < 1e-10
    }

    pub fn is_free(&self) -> bool {
        self.lower_bound <= -1e20 && self.upper_bound >= 1e20
    }
}

/// Variable type in the LP model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarType {
    Continuous,
    Integer,
    Binary,
    SemiContinuous,
    SemiInteger,
}

impl Default for VarType {
    fn default() -> Self {
        VarType::Continuous
    }
}

/// Constraint information within the LP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub sense: ConstraintSense,
    pub rhs: f64,
    pub range: Option<f64>,
    pub row_indices: Vec<usize>,
    pub row_values: Vec<f64>,
    pub index: usize,
}

impl Constraint {
    pub fn new(name: &str, sense: ConstraintSense, rhs: f64) -> Self {
        Self {
            name: name.to_string(),
            sense,
            rhs,
            range: None,
            row_indices: Vec::new(),
            row_values: Vec::new(),
            index: 0,
        }
    }

    pub fn add_term(&mut self, col: usize, value: f64) {
        if let Some(pos) = self.row_indices.iter().position(|&c| c == col) {
            self.row_values[pos] += value;
        } else {
            self.row_indices.push(col);
            self.row_values.push(value);
        }
    }

    pub fn nnz(&self) -> usize {
        self.row_values.iter().filter(|&&v| v.abs() > 1e-20).count()
    }
}

/// SOS (Special Ordered Set) constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SosConstraint {
    pub sos_type: SosType,
    pub name: String,
    pub members: Vec<usize>,
    pub weights: Vec<f64>,
}

/// SOS type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SosType {
    Type1,
    Type2,
}

/// Indicator constraint: if binary_var == active_value then constraint is enforced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConstraint {
    pub name: String,
    pub binary_var: usize,
    pub active_value: bool,
    pub constraint_indices: Vec<usize>,
    pub constraint_values: Vec<f64>,
    pub sense: ConstraintSense,
    pub rhs: f64,
}

/// LP model statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub num_vars: usize,
    pub num_constraints: usize,
    pub num_nonzeros: usize,
    pub num_integer_vars: usize,
    pub num_binary_vars: usize,
    pub num_continuous_vars: usize,
    pub num_eq_constraints: usize,
    pub num_le_constraints: usize,
    pub num_ge_constraints: usize,
    pub num_range_constraints: usize,
    pub num_free_vars: usize,
    pub num_fixed_vars: usize,
    pub num_bounded_vars: usize,
    pub obj_offset: f64,
    pub density: f64,
    pub max_abs_coeff: f64,
    pub min_abs_nonzero_coeff: f64,
}

/// The core LP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpModel {
    pub name: String,
    pub sense: OptDirection,
    pub obj_offset: f64,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
    pub sos_constraints: Vec<SosConstraint>,
    pub indicator_constraints: Vec<IndicatorConstraint>,
    var_name_map: HashMap<String, usize>,
    con_name_map: HashMap<String, usize>,
}

impl LpModel {
    /// Create a new empty LP model.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sense: OptDirection::Minimize,
            obj_offset: 0.0,
            variables: Vec::new(),
            constraints: Vec::new(),
            sos_constraints: Vec::new(),
            indicator_constraints: Vec::new(),
            var_name_map: HashMap::new(),
            con_name_map: HashMap::new(),
        }
    }

    /// Add a variable to the model, returning its index.
    pub fn add_variable(&mut self, mut var: Variable) -> usize {
        let idx = self.variables.len();
        var.index = idx;
        self.var_name_map.insert(var.name.clone(), idx);
        self.variables.push(var);
        idx
    }

    /// Add a constraint to the model, returning its index.
    pub fn add_constraint(&mut self, mut con: Constraint) -> usize {
        let idx = self.constraints.len();
        con.index = idx;
        self.con_name_map.insert(con.name.clone(), idx);
        self.constraints.push(con);
        idx
    }

    /// Add an SOS constraint.
    pub fn add_sos_constraint(&mut self, sos: SosConstraint) {
        self.sos_constraints.push(sos);
    }

    /// Add an indicator constraint.
    pub fn add_indicator_constraint(&mut self, ind: IndicatorConstraint) {
        self.indicator_constraints.push(ind);
    }

    /// Set the objective coefficient for a variable.
    pub fn set_obj_coeff(&mut self, var_idx: usize, coeff: f64) {
        self.variables[var_idx].obj_coeff = coeff;
    }

    /// Set variable bounds.
    pub fn set_var_bounds(&mut self, var_idx: usize, lb: f64, ub: f64) {
        self.variables[var_idx].lower_bound = lb;
        self.variables[var_idx].upper_bound = ub;
    }

    /// Set constraint RHS.
    pub fn set_rhs(&mut self, con_idx: usize, rhs: f64) {
        self.constraints[con_idx].rhs = rhs;
    }

    /// Set constraint sense.
    pub fn set_sense(&mut self, con_idx: usize, sense: ConstraintSense) {
        self.constraints[con_idx].sense = sense;
    }

    /// Set a coefficient in the constraint matrix.
    pub fn set_coeff(&mut self, con_idx: usize, var_idx: usize, value: f64) {
        let con = &mut self.constraints[con_idx];
        if let Some(pos) = con.row_indices.iter().position(|&c| c == var_idx) {
            if value.abs() < 1e-20 {
                con.row_indices.swap_remove(pos);
                con.row_values.swap_remove(pos);
            } else {
                con.row_values[pos] = value;
            }
        } else if value.abs() >= 1e-20 {
            con.row_indices.push(var_idx);
            con.row_values.push(value);
        }
    }

    /// Get objective coefficients as a vector.
    pub fn obj_coeffs(&self) -> Vec<f64> {
        self.variables.iter().map(|v| v.obj_coeff).collect()
    }

    /// Get variable lower bounds.
    pub fn lower_bounds(&self) -> Vec<f64> {
        self.variables.iter().map(|v| v.lower_bound).collect()
    }

    /// Get variable upper bounds.
    pub fn upper_bounds(&self) -> Vec<f64> {
        self.variables.iter().map(|v| v.upper_bound).collect()
    }

    /// Get RHS vector.
    pub fn rhs_vec(&self) -> Vec<f64> {
        self.constraints.iter().map(|c| c.rhs).collect()
    }

    /// Get senses vector.
    pub fn senses_vec(&self) -> Vec<ConstraintSense> {
        self.constraints.iter().map(|c| c.sense).collect()
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.variables.len()
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Number of nonzero entries.
    pub fn num_nonzeros(&self) -> usize {
        self.constraints.iter().map(|c| c.nnz()).sum()
    }

    /// Lookup variable index by name.
    pub fn var_index(&self, name: &str) -> Option<usize> {
        self.var_name_map.get(name).copied()
    }

    /// Lookup constraint index by name.
    pub fn con_index(&self, name: &str) -> Option<usize> {
        self.con_name_map.get(name).copied()
    }

    /// Remove a variable by index (adjusts all references).
    pub fn remove_variable(&mut self, var_idx: usize) {
        let name = self.variables[var_idx].name.clone();
        self.var_name_map.remove(&name);
        self.variables.remove(var_idx);
        // Re-index
        for (i, v) in self.variables.iter_mut().enumerate() {
            v.index = i;
            self.var_name_map.insert(v.name.clone(), i);
        }
        // Update constraint references
        for con in &mut self.constraints {
            let mut to_remove = Vec::new();
            for (pos, idx) in con.row_indices.iter_mut().enumerate() {
                if *idx == var_idx {
                    to_remove.push(pos);
                } else if *idx > var_idx {
                    *idx -= 1;
                }
            }
            for &pos in to_remove.iter().rev() {
                con.row_indices.swap_remove(pos);
                con.row_values.swap_remove(pos);
            }
        }
    }

    /// Remove a constraint by index.
    pub fn remove_constraint(&mut self, con_idx: usize) {
        let name = self.constraints[con_idx].name.clone();
        self.con_name_map.remove(&name);
        self.constraints.remove(con_idx);
        for (i, c) in self.constraints.iter_mut().enumerate() {
            c.index = i;
            self.con_name_map.insert(c.name.clone(), i);
        }
    }

    /// Build a sparse constraint matrix in COO format.
    pub fn constraint_matrix(&self) -> SparseMatrix {
        let m = self.constraints.len();
        let n = self.variables.len();
        let mut mat = SparseMatrix::new(m, n);
        for (i, con) in self.constraints.iter().enumerate() {
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                if val.abs() > 1e-20 {
                    mat.add_entry(i, col, val);
                }
            }
        }
        mat
    }

    /// Build a column-major dense matrix (for small problems).
    pub fn dense_matrix(&self) -> Vec<Vec<f64>> {
        let m = self.constraints.len();
        let n = self.variables.len();
        let mut mat = vec![vec![0.0; n]; m];
        for (i, con) in self.constraints.iter().enumerate() {
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                mat[i][col] = val;
            }
        }
        mat
    }

    /// Get model statistics.
    pub fn stats(&self) -> ModelStats {
        let num_vars = self.variables.len();
        let num_constraints = self.constraints.len();
        let num_nonzeros = self.num_nonzeros();
        let mut num_integer = 0;
        let mut num_binary = 0;
        let mut num_continuous = 0;
        let mut num_free = 0;
        let mut num_fixed = 0;
        let mut num_bounded = 0;

        for v in &self.variables {
            match v.var_type {
                VarType::Integer | VarType::SemiInteger => num_integer += 1,
                VarType::Binary => num_binary += 1,
                VarType::Continuous | VarType::SemiContinuous => num_continuous += 1,
            }
            if v.is_free() {
                num_free += 1;
            } else if v.is_fixed() {
                num_fixed += 1;
            } else {
                num_bounded += 1;
            }
        }

        let num_eq = self
            .constraints
            .iter()
            .filter(|c| c.sense == ConstraintSense::Eq)
            .count();
        let num_le = self
            .constraints
            .iter()
            .filter(|c| c.sense == ConstraintSense::Le)
            .count();
        let num_ge = self
            .constraints
            .iter()
            .filter(|c| c.sense == ConstraintSense::Ge)
            .count();
        let num_range = self
            .constraints
            .iter()
            .filter(|c| c.range.is_some())
            .count();

        let max_coeff = self.max_abs_coeff();
        let min_coeff = self.min_abs_nonzero_coeff();

        let density = if num_vars > 0 && num_constraints > 0 {
            num_nonzeros as f64 / (num_vars as f64 * num_constraints as f64)
        } else {
            0.0
        };

        ModelStats {
            num_vars,
            num_constraints,
            num_nonzeros,
            num_integer_vars: num_integer,
            num_binary_vars: num_binary,
            num_continuous_vars: num_continuous,
            num_eq_constraints: num_eq,
            num_le_constraints: num_le,
            num_ge_constraints: num_ge,
            num_range_constraints: num_range,
            num_free_vars: num_free,
            num_fixed_vars: num_fixed,
            num_bounded_vars: num_bounded,
            obj_offset: self.obj_offset,
            density,
            max_abs_coeff: max_coeff,
            min_abs_nonzero_coeff: min_coeff,
        }
    }

    /// Maximum absolute coefficient value.
    fn max_abs_coeff(&self) -> f64 {
        let mut max_val = 0.0f64;
        for con in &self.constraints {
            for &v in &con.row_values {
                max_val = max_val.max(v.abs());
            }
        }
        for v in &self.variables {
            max_val = max_val.max(v.obj_coeff.abs());
        }
        max_val
    }

    /// Minimum absolute nonzero coefficient value.
    fn min_abs_nonzero_coeff(&self) -> f64 {
        let mut min_val = f64::INFINITY;
        for con in &self.constraints {
            for &v in &con.row_values {
                if v.abs() > 1e-20 {
                    min_val = min_val.min(v.abs());
                }
            }
        }
        for v in &self.variables {
            if v.obj_coeff.abs() > 1e-20 {
                min_val = min_val.min(v.obj_coeff.abs());
            }
        }
        if min_val == f64::INFINITY {
            0.0
        } else {
            min_val
        }
    }

    /// Scale the model so coefficient range is more uniform.
    pub fn scale(&mut self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraints.len();
        let n = self.variables.len();
        let mut row_scale = vec![1.0; m];
        let mut col_scale = vec![1.0; n];

        // Geometric mean scaling: 2 passes
        for _pass in 0..2 {
            // Row scaling
            for (i, con) in self.constraints.iter().enumerate() {
                let mut max_v = 0.0f64;
                for &v in &con.row_values {
                    max_v = max_v.max(
                        v.abs()
                            * col_scale[con.row_indices[con
                                .row_values
                                .iter()
                                .position(|&x| (x - v).abs() < 1e-30)
                                .unwrap_or(0)]],
                    );
                }
                // Use max absolute value in row
                let mut abs_vals: Vec<f64> = con
                    .row_values
                    .iter()
                    .map(|&v| v.abs())
                    .filter(|&v| v > 1e-20)
                    .collect();
                if !abs_vals.is_empty() {
                    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let geo_mean = (abs_vals.first().unwrap() * abs_vals.last().unwrap()).sqrt();
                    if geo_mean > 1e-20 {
                        row_scale[i] = 1.0 / geo_mean;
                    }
                }
                let _ = max_v; // suppress unused
            }
            // Column scaling
            for j in 0..n {
                let mut abs_vals = Vec::new();
                for con in &self.constraints {
                    for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if col == j && val.abs() > 1e-20 {
                            abs_vals.push(val.abs());
                        }
                    }
                }
                if !abs_vals.is_empty() {
                    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let geo_mean = (abs_vals.first().unwrap() * abs_vals.last().unwrap()).sqrt();
                    if geo_mean > 1e-20 {
                        col_scale[j] = 1.0 / geo_mean;
                    }
                }
            }
        }

        // Apply scaling
        for (i, con) in self.constraints.iter_mut().enumerate() {
            for (pos, &col) in con.row_indices.iter().enumerate() {
                con.row_values[pos] *= row_scale[i] * col_scale[col];
            }
            con.rhs *= row_scale[i];
        }
        for (j, var) in self.variables.iter_mut().enumerate() {
            var.obj_coeff *= col_scale[j];
            if var.lower_bound > -1e20 {
                var.lower_bound /= col_scale[j];
            }
            if var.upper_bound < 1e20 {
                var.upper_bound /= col_scale[j];
            }
        }

        (row_scale, col_scale)
    }

    /// Unscale a solution vector.
    pub fn unscale_primal(&self, x: &[f64], col_scale: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(col_scale.iter())
            .map(|(&xi, &s)| xi * s)
            .collect()
    }

    /// Unscale dual values.
    pub fn unscale_dual(&self, y: &[f64], row_scale: &[f64]) -> Vec<f64> {
        y.iter()
            .zip(row_scale.iter())
            .map(|(&yi, &s)| yi * s)
            .collect()
    }

    /// Convert to standard form (all <= constraints, non-negative vars).
    /// Returns the transformed model and a mapping.
    pub fn to_standard_form(&self) -> (LpModel, StandardFormMap) {
        let mut std_model = LpModel::new(&format!("{}_std", self.name));
        std_model.sense = self.sense;
        std_model.obj_offset = self.obj_offset;

        let mut col_map = Vec::new(); // maps std var -> (orig var, sign)
        let mut free_var_split = HashMap::new();

        // Handle variables
        for (j, var) in self.variables.iter().enumerate() {
            if var.is_free() {
                // Split free var into x+ - x-
                let idx_pos = std_model.add_variable(Variable::continuous(
                    &format!("{}_pos", var.name),
                    0.0,
                    f64::INFINITY,
                ));
                let idx_neg = std_model.add_variable(Variable::continuous(
                    &format!("{}_neg", var.name),
                    0.0,
                    f64::INFINITY,
                ));
                std_model.variables[idx_pos].obj_coeff = var.obj_coeff;
                std_model.variables[idx_neg].obj_coeff = -var.obj_coeff;
                col_map.push((j, 1.0));
                col_map.push((j, -1.0));
                free_var_split.insert(j, (idx_pos, idx_neg));
            } else if var.lower_bound != 0.0 && var.lower_bound > -1e20 {
                // Shift: x' = x - lb, so x = x' + lb
                let idx = std_model.add_variable(Variable::continuous(
                    &var.name,
                    0.0,
                    if var.upper_bound < 1e20 {
                        var.upper_bound - var.lower_bound
                    } else {
                        f64::INFINITY
                    },
                ));
                std_model.variables[idx].obj_coeff = var.obj_coeff;
                col_map.push((j, 1.0));
            } else {
                let idx = std_model.add_variable(Variable::continuous(
                    &var.name,
                    var.lower_bound.max(0.0),
                    var.upper_bound,
                ));
                std_model.variables[idx].obj_coeff = var.obj_coeff;
                col_map.push((j, 1.0));
            }
        }

        // Handle constraints
        for (i, con) in self.constraints.iter().enumerate() {
            match con.sense {
                ConstraintSense::Le => {
                    let mut new_con = Constraint::new(&con.name, ConstraintSense::Le, con.rhs);
                    for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if let Some(&(idx_pos, idx_neg)) = free_var_split.get(&col) {
                            new_con.add_term(idx_pos, val);
                            new_con.add_term(idx_neg, -val);
                        } else {
                            // Find the std column for this original column
                            let std_col = col_map.iter().position(|&(orig, _)| orig == col);
                            if let Some(sc) = std_col {
                                new_con.add_term(sc, val);
                            }
                        }
                    }
                    std_model.add_constraint(new_con);
                }
                ConstraintSense::Ge => {
                    // Flip: -a^T x <= -rhs
                    let mut new_con = Constraint::new(&con.name, ConstraintSense::Le, -con.rhs);
                    for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if let Some(&(idx_pos, idx_neg)) = free_var_split.get(&col) {
                            new_con.add_term(idx_pos, -val);
                            new_con.add_term(idx_neg, val);
                        } else {
                            let std_col = col_map.iter().position(|&(orig, _)| orig == col);
                            if let Some(sc) = std_col {
                                new_con.add_term(sc, -val);
                            }
                        }
                    }
                    std_model.add_constraint(new_con);
                }
                ConstraintSense::Eq => {
                    // Two inequalities: a^T x <= b and -a^T x <= -b
                    let mut con_le =
                        Constraint::new(&format!("{}_le", con.name), ConstraintSense::Le, con.rhs);
                    let mut con_ge =
                        Constraint::new(&format!("{}_ge", con.name), ConstraintSense::Le, -con.rhs);
                    for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                        if let Some(&(idx_pos, idx_neg)) = free_var_split.get(&col) {
                            con_le.add_term(idx_pos, val);
                            con_le.add_term(idx_neg, -val);
                            con_ge.add_term(idx_pos, -val);
                            con_ge.add_term(idx_neg, val);
                        } else {
                            let std_col = col_map.iter().position(|&(orig, _)| orig == col);
                            if let Some(sc) = std_col {
                                con_le.add_term(sc, val);
                                con_ge.add_term(sc, -val);
                            }
                        }
                    }
                    std_model.add_constraint(con_le);
                    std_model.add_constraint(con_ge);
                }
            }
            let _ = i; // suppress unused
        }

        let map = StandardFormMap {
            col_map,
            free_var_split,
            num_orig_vars: self.variables.len(),
            num_orig_constraints: self.constraints.len(),
        };

        (std_model, map)
    }

    /// Check if this is a pure LP (no integer variables).
    pub fn is_pure_lp(&self) -> bool {
        self.variables
            .iter()
            .all(|v| v.var_type == VarType::Continuous)
    }

    /// Check if this is a MIP.
    pub fn is_mip(&self) -> bool {
        self.variables
            .iter()
            .any(|v| v.var_type != VarType::Continuous)
    }

    /// Get a formatted summary string.
    pub fn summary(&self) -> String {
        let s = self.stats();
        format!(
            "LP Model '{}': {} vars ({} int, {} bin), {} constraints, {} nonzeros, density {:.4}",
            self.name,
            s.num_vars,
            s.num_integer_vars,
            s.num_binary_vars,
            s.num_constraints,
            s.num_nonzeros,
            s.density
        )
    }

    /// Transpose the model: swap rows and columns.
    pub fn transpose(&self) -> LpModel {
        let mut t = LpModel::new(&format!("{}_T", self.name));
        // Old constraints become variables
        for con in &self.constraints {
            let lb = match con.sense {
                ConstraintSense::Le => f64::NEG_INFINITY,
                ConstraintSense::Ge => 0.0,
                ConstraintSense::Eq => f64::NEG_INFINITY,
            };
            let ub = match con.sense {
                ConstraintSense::Le => 0.0,
                ConstraintSense::Ge => f64::INFINITY,
                ConstraintSense::Eq => f64::INFINITY,
            };
            let mut var = Variable::continuous(&con.name, lb, ub);
            var.obj_coeff = con.rhs;
            t.add_variable(var);
        }
        // Old variables become constraints
        for var in &self.variables {
            let new_con = Constraint::new(&var.name, ConstraintSense::Le, var.obj_coeff);
            t.add_constraint(new_con);
        }
        // Fill transposed coefficients
        for (i, con) in self.constraints.iter().enumerate() {
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                t.set_coeff(col, i, val);
            }
        }
        t
    }
}

/// Mapping from standard form back to original model.
#[derive(Debug, Clone)]
pub struct StandardFormMap {
    pub col_map: Vec<(usize, f64)>,
    pub free_var_split: HashMap<usize, (usize, usize)>,
    pub num_orig_vars: usize,
    pub num_orig_constraints: usize,
}

impl StandardFormMap {
    /// Map a standard form solution back to the original variables.
    pub fn recover_primal(&self, std_x: &[f64]) -> Vec<f64> {
        let mut orig_x = vec![0.0; self.num_orig_vars];
        for (std_col, &(orig_col, sign)) in self.col_map.iter().enumerate() {
            if std_col < std_x.len() {
                orig_x[orig_col] += sign * std_x[std_col];
            }
        }
        orig_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> LpModel {
        let mut m = LpModel::new("test");
        m.sense = OptDirection::Minimize;
        let x0 = m.add_variable(Variable::continuous("x0", 0.0, 10.0));
        let x1 = m.add_variable(Variable::continuous("x1", 0.0, 10.0));
        m.set_obj_coeff(x0, 1.0);
        m.set_obj_coeff(x1, 2.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 4.0);
        c0.add_term(x0, 1.0);
        c0.add_term(x1, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 6.0);
        c1.add_term(x0, 2.0);
        c1.add_term(x1, 1.0);
        m.add_constraint(c1);

        m
    }

    #[test]
    fn test_model_creation() {
        let m = make_test_model();
        assert_eq!(m.num_vars(), 2);
        assert_eq!(m.num_constraints(), 2);
        assert_eq!(m.num_nonzeros(), 4);
    }

    #[test]
    fn test_variable_lookup() {
        let m = make_test_model();
        assert_eq!(m.var_index("x0"), Some(0));
        assert_eq!(m.var_index("x1"), Some(1));
        assert_eq!(m.var_index("x2"), None);
    }

    #[test]
    fn test_constraint_lookup() {
        let m = make_test_model();
        assert_eq!(m.con_index("c0"), Some(0));
        assert_eq!(m.con_index("c1"), Some(1));
    }

    #[test]
    fn test_obj_coeffs() {
        let m = make_test_model();
        assert_eq!(m.obj_coeffs(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_model_stats() {
        let m = make_test_model();
        let s = m.stats();
        assert_eq!(s.num_vars, 2);
        assert_eq!(s.num_constraints, 2);
        assert_eq!(s.num_le_constraints, 2);
        assert_eq!(s.num_continuous_vars, 2);
        assert!(s.density > 0.9);
    }

    #[test]
    fn test_remove_variable() {
        let mut m = make_test_model();
        m.remove_variable(0);
        assert_eq!(m.num_vars(), 1);
        assert_eq!(m.variables[0].name, "x1");
    }

    #[test]
    fn test_remove_constraint() {
        let mut m = make_test_model();
        m.remove_constraint(0);
        assert_eq!(m.num_constraints(), 1);
        assert_eq!(m.constraints[0].name, "c1");
    }

    #[test]
    fn test_dense_matrix() {
        let m = make_test_model();
        let mat = m.dense_matrix();
        assert_eq!(mat.len(), 2);
        assert!((mat[0][0] - 1.0).abs() < 1e-10);
        assert!((mat[1][0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_pure_lp() {
        let m = make_test_model();
        assert!(m.is_pure_lp());
        assert!(!m.is_mip());
    }

    #[test]
    fn test_standard_form() {
        let mut m = LpModel::new("test_std");
        let x0 = m.add_variable(Variable::continuous("x0", 0.0, f64::INFINITY));
        m.set_obj_coeff(x0, 1.0);
        let mut c = Constraint::new("c0", ConstraintSense::Ge, 2.0);
        c.add_term(x0, 1.0);
        m.add_constraint(c);
        let (std_m, _map) = m.to_standard_form();
        assert!(std_m
            .constraints
            .iter()
            .all(|c| c.sense == ConstraintSense::Le));
    }
}
