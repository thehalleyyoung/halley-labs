//! Simplex tableau operations: column generation, row operations, pivot operations,
//! reduced cost computation, basic variable values, entering/leaving variable selection.

use crate::basis::{Basis, BasisError};
use crate::model::LpModel;
use bicut_types::ConstraintSense;

/// Tolerance for numerical comparisons.
const TOLERANCE: f64 = 1e-8;

/// Pricing strategy for entering variable selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PricingStrategy {
    /// Dantzig's rule: most negative reduced cost.
    Dantzig,
    /// Steepest edge: best improvement per unit step.
    SteepestEdge,
    /// Devex: approximate steepest edge.
    Devex,
    /// Partial pricing: only scan a subset of variables.
    Partial,
}

/// Ratio test strategy for leaving variable selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatioTestStrategy {
    /// Standard minimum ratio test.
    Standard,
    /// Harris ratio test (relaxed for degeneracy).
    Harris,
    /// Bound-flipping ratio test.
    BoundFlipping,
}

/// The simplex tableau encapsulates the current state of the simplex algorithm,
/// including basis, working data, and operations.
#[derive(Debug, Clone)]
pub struct SimplexTableau {
    /// Number of original variables.
    pub num_vars: usize,
    /// Number of constraints (= number of rows).
    pub num_rows: usize,
    /// Total number of columns (original vars + slacks).
    pub num_cols: usize,
    /// Constraint matrix in dense column-major form (column j = col_matrix[j]).
    pub col_matrix: Vec<Vec<f64>>,
    /// Objective coefficients for all columns.
    pub obj: Vec<f64>,
    /// Right-hand side values.
    pub rhs: Vec<f64>,
    /// Variable lower bounds.
    pub lower_bounds: Vec<f64>,
    /// Variable upper bounds.
    pub upper_bounds: Vec<f64>,
    /// Current basis.
    pub basis: Basis,
    /// Current basic variable values (B^{-1} b).
    pub basic_values: Vec<f64>,
    /// Current reduced costs for non-basic variables.
    pub reduced_costs: Vec<f64>,
    /// Dual variables (simplex multipliers).
    pub dual_values: Vec<f64>,
    /// Steepest edge weights (if using steepest edge pricing).
    pub edge_weights: Vec<f64>,
    /// Devex reference frame weights.
    pub devex_weights: Vec<f64>,
    /// Pricing strategy.
    pub pricing: PricingStrategy,
    /// Ratio test strategy.
    pub ratio_test: RatioTestStrategy,
    /// Whether the problem is maximization.
    pub is_maximization: bool,
    /// Iteration counter.
    pub iteration: usize,
    /// Partial pricing start index.
    partial_price_idx: usize,
    /// Partial pricing batch size.
    partial_price_batch: usize,
}

impl SimplexTableau {
    /// Build a tableau from an LP model. Adds slack variables for inequality constraints.
    pub fn from_model(model: &LpModel) -> Self {
        let m = model.num_constraints();
        let n = model.num_vars();

        // Count slacks needed
        let mut num_slacks = 0;
        for con in &model.constraints {
            if con.sense != ConstraintSense::Eq {
                num_slacks += 1;
            }
        }
        let total_cols = n + num_slacks;

        // Build column matrix
        let mut col_matrix = vec![vec![0.0; m]; total_cols];
        for (i, con) in model.constraints.iter().enumerate() {
            for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
                col_matrix[col][i] = val;
            }
        }

        // Add slack columns
        let mut slack_idx = n;
        let mut rhs = vec![0.0; m];
        for (i, con) in model.constraints.iter().enumerate() {
            rhs[i] = con.rhs;
            match con.sense {
                ConstraintSense::Le => {
                    col_matrix[slack_idx][i] = 1.0;
                    slack_idx += 1;
                }
                ConstraintSense::Ge => {
                    // Multiply row by -1 to convert to <=, then add slack
                    for j in 0..n {
                        col_matrix[j][i] = -col_matrix[j][i];
                    }
                    rhs[i] = -rhs[i];
                    col_matrix[slack_idx][i] = 1.0;
                    slack_idx += 1;
                }
                ConstraintSense::Eq => {
                    // No slack for equality
                }
            }
        }

        // Objective: original vars have model coefficients, slacks have 0
        let mut obj = vec![0.0; total_cols];
        let is_max = model.sense == bicut_types::OptDirection::Maximize;
        for (j, var) in model.variables.iter().enumerate() {
            obj[j] = if is_max {
                -var.obj_coeff
            } else {
                var.obj_coeff
            };
        }

        // Bounds
        let mut lower = vec![0.0; total_cols];
        let mut upper = vec![f64::INFINITY; total_cols];
        for (j, var) in model.variables.iter().enumerate() {
            lower[j] = var.lower_bound;
            upper[j] = var.upper_bound;
        }

        // Initial basis: slacks are basic
        let basic_vars: Vec<usize> = (n..total_cols).collect();
        let basis = Basis::new(m, basic_vars);

        let mut tableau = Self {
            num_vars: n,
            num_rows: m,
            num_cols: total_cols,
            col_matrix,
            obj,
            rhs,
            lower_bounds: lower,
            upper_bounds: upper,
            basis,
            basic_values: vec![0.0; m],
            reduced_costs: vec![0.0; total_cols],
            dual_values: vec![0.0; m],
            edge_weights: vec![1.0; total_cols],
            devex_weights: vec![1.0; total_cols],
            pricing: PricingStrategy::Dantzig,
            ratio_test: RatioTestStrategy::Standard,
            is_maximization: is_max,
            iteration: 0,
            partial_price_idx: 0,
            partial_price_batch: 50,
        };

        // Factorize initial basis
        let cols = tableau.col_matrix.clone();
        let _ = tableau.basis.factorize(|var| {
            if var < cols.len() {
                cols[var].clone()
            } else {
                vec![0.0; m]
            }
        });

        tableau.compute_basic_values();
        tableau.compute_dual_values();
        tableau.compute_reduced_costs();

        tableau
    }

    /// Get a column of the constraint matrix.
    pub fn get_column(&self, j: usize) -> &[f64] {
        &self.col_matrix[j]
    }

    /// Compute FTRAN: solve B * d = a_j to get the representation of column j in the basis.
    pub fn ftran(&self, col: &[f64]) -> Result<Vec<f64>, BasisError> {
        self.basis.solve(col)
    }

    /// Compute BTRAN: solve B^T * y = c_B to get dual variables.
    pub fn btran(&self, rhs: &[f64]) -> Result<Vec<f64>, BasisError> {
        self.basis.solve_transpose(rhs)
    }

    /// Compute basic variable values: x_B = B^{-1} b.
    pub fn compute_basic_values(&mut self) {
        match self.basis.solve(&self.rhs) {
            Ok(vals) => self.basic_values = vals,
            Err(_) => {
                // If solve fails, set to rhs as fallback
                self.basic_values = self.rhs.clone();
            }
        }
    }

    /// Compute dual variables (simplex multipliers): y = B^{-T} c_B.
    pub fn compute_dual_values(&mut self) {
        let m = self.num_rows;
        let mut cb = vec![0.0; m];
        for (i, &var) in self.basis.basic_vars.iter().enumerate() {
            if var < self.obj.len() {
                cb[i] = self.obj[var];
            }
        }
        match self.btran(&cb) {
            Ok(y) => self.dual_values = y,
            Err(_) => self.dual_values = vec![0.0; m],
        }
    }

    /// Compute reduced costs for all non-basic variables.
    pub fn compute_reduced_costs(&mut self) {
        self.reduced_costs = vec![0.0; self.num_cols];
        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                self.reduced_costs[j] = 0.0;
            } else {
                let mut rc = self.obj[j];
                let col = &self.col_matrix[j];
                for (i, &val) in col.iter().enumerate() {
                    rc -= self.dual_values[i] * val;
                }
                self.reduced_costs[j] = rc;
            }
        }
    }

    /// Compute the reduced cost for a single variable.
    pub fn reduced_cost(&self, j: usize) -> f64 {
        if self.basis.is_basic(j) {
            return 0.0;
        }
        let mut rc = self.obj[j];
        let col = &self.col_matrix[j];
        for (i, &val) in col.iter().enumerate() {
            rc -= self.dual_values[i] * val;
        }
        rc
    }

    /// Select the entering variable using the current pricing strategy.
    pub fn select_entering(&mut self) -> Option<usize> {
        match self.pricing {
            PricingStrategy::Dantzig => self.dantzig_pricing(),
            PricingStrategy::SteepestEdge => self.steepest_edge_pricing(),
            PricingStrategy::Devex => self.devex_pricing(),
            PricingStrategy::Partial => self.partial_pricing(),
        }
    }

    /// Dantzig's rule: select the variable with the most negative reduced cost.
    fn dantzig_pricing(&self) -> Option<usize> {
        let mut best_j = None;
        let mut best_rc = -TOLERANCE;

        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                continue;
            }
            let rc = self.reduced_costs[j];

            // Check if variable can improve: at lower bound with negative rc,
            // or at upper bound with positive rc
            let at_lower = self.is_at_lower(j);
            let at_upper = self.is_at_upper(j);

            if at_lower && rc < best_rc {
                best_rc = rc;
                best_j = Some(j);
            } else if at_upper && -rc < best_rc {
                best_rc = -rc;
                best_j = Some(j);
            }
        }
        best_j
    }

    /// Steepest edge pricing: select based on rc^2 / ||d||^2.
    fn steepest_edge_pricing(&self) -> Option<usize> {
        let mut best_j = None;
        let mut best_ratio = -TOLERANCE;

        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                continue;
            }
            let rc = self.reduced_costs[j];
            let at_lower = self.is_at_lower(j);
            let at_upper = self.is_at_upper(j);

            let effective_rc = if at_lower {
                -rc
            } else if at_upper {
                rc
            } else {
                -rc.abs()
            };

            if effective_rc > TOLERANCE {
                let weight = self.edge_weights[j].max(1e-12);
                let ratio = effective_rc * effective_rc / weight;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_j = Some(j);
                }
            }
        }
        best_j
    }

    /// Devex pricing: approximate steepest edge.
    fn devex_pricing(&self) -> Option<usize> {
        let mut best_j = None;
        let mut best_ratio = -TOLERANCE;

        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                continue;
            }
            let rc = self.reduced_costs[j];
            let at_lower = self.is_at_lower(j);
            let at_upper = self.is_at_upper(j);
            let effective_rc = if at_lower && rc < -TOLERANCE {
                -rc
            } else if at_upper && rc > TOLERANCE {
                rc
            } else {
                continue;
            };

            let weight = self.devex_weights[j].max(1e-12);
            let ratio = effective_rc * effective_rc / weight;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_j = Some(j);
            }
        }
        best_j
    }

    /// Partial pricing: scan only a batch of variables.
    fn partial_pricing(&mut self) -> Option<usize> {
        let mut best_j = None;
        let mut best_rc = -TOLERANCE;
        let start = self.partial_price_idx;
        let batch = self.partial_price_batch.min(self.num_cols);

        for offset in 0..self.num_cols {
            let j = (start + offset) % self.num_cols;
            if self.basis.is_basic(j) {
                continue;
            }
            let rc = self.reduced_costs[j];
            let at_lower = self.is_at_lower(j);

            if at_lower && rc < best_rc {
                best_rc = rc;
                best_j = Some(j);
            }

            if offset >= batch && best_j.is_some() {
                break;
            }
        }

        self.partial_price_idx = (start + batch) % self.num_cols;
        best_j
    }

    /// Select the leaving variable using the current ratio test strategy.
    /// Returns (leaving_row, step_size, bound_flip) or None if unbounded.
    pub fn select_leaving(&self, entering: usize) -> Option<(usize, f64, bool)> {
        let col = &self.col_matrix[entering];
        let pivot_col = match self.ftran(col) {
            Ok(d) => d,
            Err(_) => return None,
        };

        let at_lower = self.is_at_lower(entering);

        match self.ratio_test {
            RatioTestStrategy::Standard => self.standard_ratio_test(&pivot_col, at_lower),
            RatioTestStrategy::Harris => self.harris_ratio_test(&pivot_col, at_lower),
            RatioTestStrategy::BoundFlipping => {
                self.bound_flipping_ratio_test(&pivot_col, at_lower, entering)
            }
        }
    }

    /// Standard minimum ratio test.
    fn standard_ratio_test(
        &self,
        pivot_col: &[f64],
        entering_at_lower: bool,
    ) -> Option<(usize, f64, bool)> {
        let m = self.num_rows;
        let mut min_ratio = f64::INFINITY;
        let mut leave_row = None;

        for i in 0..m {
            let d_i = pivot_col[i];
            let basic_var = self.basis.basic_vars[i];
            let x_i = self.basic_values[i];
            let lb = self.lower_bounds[basic_var];
            let ub = self.upper_bounds[basic_var];

            if entering_at_lower {
                // Step increases: look at d_i > 0 (hits upper) or d_i < 0 (hits lower)
                if d_i > TOLERANCE {
                    let ratio = (x_i - lb) / d_i;
                    if ratio < min_ratio - TOLERANCE
                        || (ratio < min_ratio + TOLERANCE
                            && d_i > pivot_col[leave_row.unwrap_or(0)].abs())
                    {
                        min_ratio = ratio;
                        leave_row = Some(i);
                    }
                } else if d_i < -TOLERANCE {
                    let ratio = (x_i - ub) / d_i;
                    if ratio < min_ratio - TOLERANCE {
                        min_ratio = ratio;
                        leave_row = Some(i);
                    }
                }
            } else {
                // Step decreases
                if d_i < -TOLERANCE {
                    let ratio = (x_i - lb) / (-d_i);
                    if ratio < min_ratio - TOLERANCE {
                        min_ratio = ratio;
                        leave_row = Some(i);
                    }
                } else if d_i > TOLERANCE {
                    let ratio = (ub - x_i) / d_i;
                    if ratio < min_ratio - TOLERANCE {
                        min_ratio = ratio;
                        leave_row = Some(i);
                    }
                }
            }
        }

        leave_row.map(|r| (r, min_ratio, false))
    }

    /// Harris ratio test: allows small infeasibilities to handle degeneracy.
    fn harris_ratio_test(
        &self,
        pivot_col: &[f64],
        entering_at_lower: bool,
    ) -> Option<(usize, f64, bool)> {
        let m = self.num_rows;
        let harris_tol = 1e-7;

        // Pass 1: find the Harris ratio (relaxed minimum)
        let mut harris_ratio = f64::INFINITY;
        for i in 0..m {
            let d_i = pivot_col[i];
            let basic_var = self.basis.basic_vars[i];
            let x_i = self.basic_values[i];
            let lb = self.lower_bounds[basic_var];
            let ub = self.upper_bounds[basic_var];

            if entering_at_lower && d_i > TOLERANCE {
                let ratio = (x_i - lb + harris_tol) / d_i;
                harris_ratio = harris_ratio.min(ratio);
            } else if entering_at_lower && d_i < -TOLERANCE {
                let ratio = (x_i - ub - harris_tol) / d_i;
                harris_ratio = harris_ratio.min(ratio);
            } else if !entering_at_lower && d_i < -TOLERANCE {
                let ratio = (x_i - lb + harris_tol) / (-d_i);
                harris_ratio = harris_ratio.min(ratio);
            } else if !entering_at_lower && d_i > TOLERANCE {
                let ratio = (ub - x_i + harris_tol) / d_i;
                harris_ratio = harris_ratio.min(ratio);
            }
        }

        if harris_ratio == f64::INFINITY {
            return None; // unbounded
        }

        // Pass 2: among candidates within harris_ratio, pick largest pivot
        let mut best_row = None;
        let mut best_pivot = 0.0f64;

        for i in 0..m {
            let d_i = pivot_col[i];
            let basic_var = self.basis.basic_vars[i];
            let x_i = self.basic_values[i];
            let lb = self.lower_bounds[basic_var];
            let ub = self.upper_bounds[basic_var];

            let ratio = if entering_at_lower && d_i > TOLERANCE {
                (x_i - lb) / d_i
            } else if entering_at_lower && d_i < -TOLERANCE {
                (x_i - ub) / d_i
            } else if !entering_at_lower && d_i < -TOLERANCE {
                (x_i - lb) / (-d_i)
            } else if !entering_at_lower && d_i > TOLERANCE {
                (ub - x_i) / d_i
            } else {
                continue;
            };

            if ratio <= harris_ratio + harris_tol && d_i.abs() > best_pivot {
                best_pivot = d_i.abs();
                best_row = Some(i);
            }
        }

        let actual_ratio = if let Some(row) = best_row {
            let d_i = pivot_col[row];
            let basic_var = self.basis.basic_vars[row];
            let x_i = self.basic_values[row];
            let lb = self.lower_bounds[basic_var];
            let ub = self.upper_bounds[basic_var];
            if entering_at_lower {
                if d_i > 0.0 {
                    (x_i - lb) / d_i
                } else {
                    (x_i - ub) / d_i
                }
            } else {
                if d_i < 0.0 {
                    (x_i - lb) / (-d_i)
                } else {
                    (ub - x_i) / d_i
                }
            }
        } else {
            0.0
        };

        best_row.map(|r| (r, actual_ratio.max(0.0), false))
    }

    /// Bound-flipping ratio test: allows non-basic variables to flip bounds.
    fn bound_flipping_ratio_test(
        &self,
        pivot_col: &[f64],
        entering_at_lower: bool,
        entering: usize,
    ) -> Option<(usize, f64, bool)> {
        // First check if entering variable has a finite opposite bound
        let enter_lb = self.lower_bounds[entering];
        let enter_ub = self.upper_bounds[entering];
        let enter_range = enter_ub - enter_lb;

        // Standard ratio test first
        let std_result = self.standard_ratio_test(pivot_col, entering_at_lower);

        if let Some((row, ratio, _)) = std_result {
            // Check if flipping the entering variable to its other bound is better
            if enter_range < f64::INFINITY && enter_range < ratio {
                return Some((row, enter_range, true));
            }
            return Some((row, ratio, false));
        }

        // If standard test finds nothing, check bound flip
        if enter_range < f64::INFINITY {
            return Some((0, enter_range, true));
        }

        None
    }

    /// Perform a pivot operation: enter variable `entering` at row `leaving_row`.
    pub fn pivot(
        &mut self,
        entering: usize,
        leaving_row: usize,
        step: f64,
    ) -> Result<(), BasisError> {
        let col = self.col_matrix[entering].clone();
        let pivot_col = self.ftran(&col)?;
        let pivot_val = pivot_col[leaving_row];

        if pivot_val.abs() < 1e-12 {
            return Err(BasisError::NumericalInstability);
        }

        // Update basic variable values
        let m = self.num_rows;
        for i in 0..m {
            if i == leaving_row {
                self.basic_values[i] = self.lower_bounds[entering] + step;
            } else {
                self.basic_values[i] -= pivot_col[i] * step;
            }
        }

        // Update the basis
        self.basis.update(leaving_row, entering, &pivot_col)?;

        // Update steepest edge weights if applicable
        if self.pricing == PricingStrategy::SteepestEdge {
            self.update_steepest_edge_weights(entering, &pivot_col, leaving_row);
        } else if self.pricing == PricingStrategy::Devex {
            self.update_devex_weights(entering, &pivot_col, leaving_row);
        }

        // Recompute duals and reduced costs
        self.compute_dual_values();
        self.compute_reduced_costs();

        self.iteration += 1;
        Ok(())
    }

    /// Update steepest edge weights after a pivot.
    fn update_steepest_edge_weights(
        &mut self,
        entering: usize,
        pivot_col: &[f64],
        leaving_row: usize,
    ) {
        let pivot_val = pivot_col[leaving_row];
        if pivot_val.abs() < 1e-12 {
            return;
        }

        let rho = self.edge_weights[entering];
        let tau = 1.0 / (pivot_val * pivot_val);

        for j in 0..self.num_cols {
            if self.basis.is_basic(j) || j == entering {
                continue;
            }
            let col_j = &self.col_matrix[j];
            if let Ok(d_j) = self.ftran(col_j) {
                let alpha_j = d_j[leaving_row];
                let sigma = alpha_j / pivot_val;
                let new_weight = self.edge_weights[j]
                    - 2.0
                        * sigma
                        * d_j
                            .iter()
                            .zip(pivot_col.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                    + sigma * sigma * rho;
                self.edge_weights[j] = new_weight.max(1e-4);
            }
        }

        self.edge_weights[entering] = tau * rho;
    }

    /// Update Devex weights after a pivot.
    fn update_devex_weights(&mut self, _entering: usize, pivot_col: &[f64], leaving_row: usize) {
        let pivot_val = pivot_col[leaving_row];
        if pivot_val.abs() < 1e-12 {
            return;
        }

        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                continue;
            }
            let col_j = &self.col_matrix[j];
            if let Ok(d_j) = self.ftran(col_j) {
                let alpha_j = d_j[leaving_row];
                let ratio = alpha_j / pivot_val;
                let candidate = 0.999 * self.devex_weights[j] + ratio * ratio;
                self.devex_weights[j] = candidate.max(1e-4);
            }
        }
    }

    /// Check if variable j is at its lower bound.
    pub fn is_at_lower(&self, j: usize) -> bool {
        if let Some(pos) = self.basis.position_of(j) {
            (self.basic_values[pos] - self.lower_bounds[j]).abs() < TOLERANCE
        } else {
            true // non-basic at lower by default
        }
    }

    /// Check if variable j is at its upper bound.
    pub fn is_at_upper(&self, j: usize) -> bool {
        if let Some(pos) = self.basis.position_of(j) {
            (self.basic_values[pos] - self.upper_bounds[j]).abs() < TOLERANCE
        } else {
            false
        }
    }

    /// Get current objective value.
    pub fn objective_value(&self) -> f64 {
        let mut val = 0.0;
        for (i, &var) in self.basis.basic_vars.iter().enumerate() {
            val += self.obj[var] * self.basic_values[i];
        }
        // Add contribution from non-basic variables at their bounds
        for j in 0..self.num_cols {
            if !self.basis.is_basic(j) {
                if self.is_at_upper(j) {
                    val += self.obj[j] * self.upper_bounds[j];
                } else {
                    val += self.obj[j] * self.lower_bounds[j];
                }
            }
        }
        if self.is_maximization {
            -val
        } else {
            val
        }
    }

    /// Get current primal solution (all variables).
    pub fn primal_solution(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.num_cols];
        for (i, &var) in self.basis.basic_vars.iter().enumerate() {
            x[var] = self.basic_values[i];
        }
        for j in 0..self.num_cols {
            if !self.basis.is_basic(j) {
                x[j] = if self.is_at_upper(j) {
                    self.upper_bounds[j]
                } else {
                    self.lower_bounds[j]
                };
            }
        }
        x
    }

    /// Check primal feasibility of the current solution.
    pub fn is_primal_feasible(&self) -> bool {
        for (i, &var) in self.basis.basic_vars.iter().enumerate() {
            let x = self.basic_values[i];
            if x < self.lower_bounds[var] - TOLERANCE || x > self.upper_bounds[var] + TOLERANCE {
                return false;
            }
        }
        true
    }

    /// Check dual feasibility (optimality conditions).
    pub fn is_dual_feasible(&self) -> bool {
        for j in 0..self.num_cols {
            if self.basis.is_basic(j) {
                continue;
            }
            let rc = self.reduced_costs[j];
            if self.is_at_lower(j) && rc < -TOLERANCE {
                return false;
            }
            if self.is_at_upper(j) && rc > TOLERANCE {
                return false;
            }
        }
        true
    }

    /// Compute the sum of infeasibilities for Phase I.
    pub fn sum_infeasibilities(&self) -> f64 {
        let mut sum = 0.0;
        for (i, &var) in self.basis.basic_vars.iter().enumerate() {
            let x = self.basic_values[i];
            if x < self.lower_bounds[var] - TOLERANCE {
                sum += self.lower_bounds[var] - x;
            } else if x > self.upper_bounds[var] + TOLERANCE {
                sum += x - self.upper_bounds[var];
            }
        }
        sum
    }

    /// Refactorize the basis.
    pub fn refactorize(&mut self) -> Result<(), BasisError> {
        let cols = self.col_matrix.clone();
        let m = self.num_rows;
        self.basis.factorize(|var| {
            if var < cols.len() {
                cols[var].clone()
            } else {
                vec![0.0; m]
            }
        })?;
        self.compute_basic_values();
        self.compute_dual_values();
        self.compute_reduced_costs();
        Ok(())
    }

    /// Apply perturbation for degeneracy handling.
    pub fn apply_perturbation(&mut self) {
        let m = self.num_rows;
        for i in 0..m {
            let var = self.basis.basic_vars[i];
            let eps = 1e-6 * (1.0 + (i as f64) * 1e-8);
            // Perturb bounds slightly to break ties
            if self.basic_values[i] <= self.lower_bounds[var] + TOLERANCE {
                self.basic_values[i] = self.lower_bounds[var] + eps;
            }
        }
    }

    /// Remove perturbation and recompute.
    pub fn remove_perturbation(&mut self) {
        self.compute_basic_values();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, LpModel, Variable};

    fn make_simple_lp() -> LpModel {
        // min -x0 - x1 s.t. x0 + x1 <= 4, 2*x0 + x1 <= 6, x0,x1 >= 0
        let mut m = LpModel::new("test");
        m.sense = bicut_types::OptDirection::Minimize;
        let x0 = m.add_variable(Variable::continuous("x0", 0.0, f64::INFINITY));
        let x1 = m.add_variable(Variable::continuous("x1", 0.0, f64::INFINITY));
        m.set_obj_coeff(x0, -1.0);
        m.set_obj_coeff(x1, -1.0);

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
    fn test_tableau_creation() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        assert_eq!(tab.num_vars, 2);
        assert_eq!(tab.num_rows, 2);
        assert_eq!(tab.num_cols, 4); // 2 vars + 2 slacks
    }

    #[test]
    fn test_basic_values() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        // Initial: slacks basic, so x_B = rhs = [4, 6]
        assert!((tab.basic_values[0] - 4.0).abs() < 1e-8);
        assert!((tab.basic_values[1] - 6.0).abs() < 1e-8);
    }

    #[test]
    fn test_reduced_costs() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        // At initial basis, rc for x0 = -1, rc for x1 = -1
        assert!((tab.reduced_costs[0] - (-1.0)).abs() < 1e-8);
        assert!((tab.reduced_costs[1] - (-1.0)).abs() < 1e-8);
    }

    #[test]
    fn test_entering_selection() {
        let lp = make_simple_lp();
        let mut tab = SimplexTableau::from_model(&lp);
        let entering = tab.select_entering();
        assert!(entering.is_some());
        let j = entering.unwrap();
        assert!(j == 0 || j == 1); // both have rc = -1
    }

    #[test]
    fn test_leaving_selection() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        let result = tab.select_leaving(0);
        assert!(result.is_some());
    }

    #[test]
    fn test_primal_feasibility() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        assert!(tab.is_primal_feasible());
    }

    #[test]
    fn test_objective_value() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        // Initial: all originals at lower bound (0), obj = 0
        assert!((tab.objective_value()).abs() < 1e-8);
    }

    #[test]
    fn test_ftran() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        let col = tab.get_column(0).to_vec();
        let d = tab.ftran(&col).unwrap();
        // B is identity (slacks), so d = col
        assert!((d[0] - col[0]).abs() < 1e-8);
        assert!((d[1] - col[1]).abs() < 1e-8);
    }

    #[test]
    fn test_primal_solution() {
        let lp = make_simple_lp();
        let tab = SimplexTableau::from_model(&lp);
        let x = tab.primal_solution();
        assert_eq!(x.len(), 4);
        assert!((x[0]).abs() < 1e-8); // x0 at lower
        assert!((x[1]).abs() < 1e-8); // x1 at lower
    }
}
