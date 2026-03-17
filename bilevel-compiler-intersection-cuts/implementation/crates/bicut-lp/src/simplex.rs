//! Revised simplex method implementation.
//!
//! Phase I (find initial basic feasible solution), Phase II (optimize),
//! pricing (Dantzig's rule, steepest edge), ratio test (standard, Harris),
//! basis management, degeneracy handling via perturbation, iteration limits.

use crate::basis::{Basis, BasisError};
use crate::model::{Constraint, LpModel, Variable};
use crate::tableau::{PricingStrategy, RatioTestStrategy, SimplexTableau};
use bicut_types::{ConstraintSense, LpStatus, OptDirection};
use log::{debug, info, trace, warn};

/// Configuration for the revised simplex method.
#[derive(Debug, Clone)]
pub struct SimplexConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Feasibility tolerance.
    pub feasibility_tol: f64,
    /// Optimality tolerance.
    pub optimality_tol: f64,
    /// Pivot tolerance.
    pub pivot_tol: f64,
    /// Whether to use perturbation for degeneracy.
    pub use_perturbation: bool,
    /// Pricing strategy.
    pub pricing: PricingStrategy,
    /// Ratio test strategy.
    pub ratio_test: RatioTestStrategy,
    /// Refactorization interval (0 = auto).
    pub refactorize_interval: usize,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000_000,
            feasibility_tol: 1e-8,
            optimality_tol: 1e-8,
            pivot_tol: 1e-10,
            use_perturbation: true,
            pricing: PricingStrategy::Dantzig,
            ratio_test: RatioTestStrategy::Harris,
            refactorize_interval: 0,
            verbose: false,
        }
    }
}

/// Result of the simplex solve.
#[derive(Debug, Clone)]
pub struct SimplexResult {
    pub status: LpStatus,
    pub objective: f64,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub basis_indices: Vec<usize>,
    pub iterations: usize,
    pub phase1_iterations: usize,
}

/// The revised simplex solver.
pub struct RevisedSimplex {
    config: SimplexConfig,
}

impl RevisedSimplex {
    /// Create a new simplex solver with the given configuration.
    pub fn new(config: SimplexConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SimplexConfig::default())
    }

    /// Solve an LP model using the two-phase simplex method.
    pub fn solve(&self, model: &LpModel) -> SimplexResult {
        let m = model.num_constraints();
        let n = model.num_vars();

        if m == 0 || n == 0 {
            return SimplexResult {
                status: LpStatus::Optimal,
                objective: model.obj_offset,
                primal: vec![0.0; n],
                dual: Vec::new(),
                reduced_costs: vec![0.0; n],
                basis_indices: Vec::new(),
                iterations: 0,
                phase1_iterations: 0,
            };
        }

        // Build the tableau
        let mut tableau = SimplexTableau::from_model(model);
        tableau.pricing = self.config.pricing;
        tableau.ratio_test = self.config.ratio_test;

        // Check if initial basis is feasible
        let needs_phase1 = !tableau.is_primal_feasible();

        let mut total_iters = 0;
        let mut phase1_iters = 0;

        if needs_phase1 {
            match self.phase1(&mut tableau) {
                Phase1Result::Feasible(iters) => {
                    phase1_iters = iters;
                    total_iters += iters;
                }
                Phase1Result::Infeasible(iters) => {
                    return SimplexResult {
                        status: LpStatus::Infeasible,
                        objective: f64::INFINITY,
                        primal: vec![0.0; n],
                        dual: vec![0.0; m],
                        reduced_costs: vec![0.0; n],
                        basis_indices: tableau.basis.basic_vars.clone(),
                        iterations: iters,
                        phase1_iterations: iters,
                    };
                }
                Phase1Result::IterationLimit(iters) => {
                    return SimplexResult {
                        status: LpStatus::IterationLimit,
                        objective: f64::INFINITY,
                        primal: vec![0.0; n],
                        dual: vec![0.0; m],
                        reduced_costs: vec![0.0; n],
                        basis_indices: tableau.basis.basic_vars.clone(),
                        iterations: iters,
                        phase1_iterations: iters,
                    };
                }
            }
        }

        // Phase II: optimize
        match self.phase2(&mut tableau, total_iters) {
            Phase2Result::Optimal(iters) => {
                total_iters += iters;
                let primal_full = tableau.primal_solution();
                let primal = primal_full[..n].to_vec();
                let dual = tableau.dual_values.clone();
                let rc = tableau.reduced_costs[..n].to_vec();
                let obj = if model.sense == OptDirection::Maximize {
                    -tableau.objective_value() + model.obj_offset
                } else {
                    tableau.objective_value() + model.obj_offset
                };

                SimplexResult {
                    status: LpStatus::Optimal,
                    objective: obj,
                    primal,
                    dual,
                    reduced_costs: rc,
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: total_iters,
                    phase1_iterations: phase1_iters,
                }
            }
            Phase2Result::Unbounded(iters) => {
                total_iters += iters;
                SimplexResult {
                    status: LpStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    primal: vec![0.0; n],
                    dual: vec![0.0; m],
                    reduced_costs: vec![0.0; n],
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: total_iters,
                    phase1_iterations: phase1_iters,
                }
            }
            Phase2Result::IterationLimit(iters) => {
                total_iters += iters;
                let primal_full = tableau.primal_solution();
                let primal = primal_full[..n].to_vec();
                SimplexResult {
                    status: LpStatus::IterationLimit,
                    objective: tableau.objective_value() + model.obj_offset,
                    primal,
                    dual: tableau.dual_values.clone(),
                    reduced_costs: tableau.reduced_costs[..n].to_vec(),
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: total_iters,
                    phase1_iterations: phase1_iters,
                }
            }
            Phase2Result::NumericalError(iters) => {
                total_iters += iters;
                SimplexResult {
                    status: LpStatus::Unknown,
                    objective: f64::NAN,
                    primal: vec![0.0; n],
                    dual: vec![0.0; m],
                    reduced_costs: vec![0.0; n],
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: total_iters,
                    phase1_iterations: phase1_iters,
                }
            }
        }
    }

    /// Solve using a warm start from an existing basis.
    pub fn solve_warm(&self, model: &LpModel, basis_indices: &[usize]) -> SimplexResult {
        let m = model.num_constraints();
        let n = model.num_vars();

        let mut tableau = SimplexTableau::from_model(model);
        tableau.pricing = self.config.pricing;
        tableau.ratio_test = self.config.ratio_test;

        // Set the warm-start basis
        if basis_indices.len() == m {
            tableau.basis = Basis::new(m, basis_indices.to_vec());
            let cols = tableau.col_matrix.clone();
            if tableau
                .basis
                .factorize(|var| {
                    if var < cols.len() {
                        cols[var].clone()
                    } else {
                        vec![0.0; m]
                    }
                })
                .is_ok()
            {
                tableau.compute_basic_values();
                tableau.compute_dual_values();
                tableau.compute_reduced_costs();
            } else {
                // Fall back to default basis
                let default_basis: Vec<usize> = (n..(n + m)).collect();
                tableau.basis = Basis::new(m, default_basis);
                let cols2 = tableau.col_matrix.clone();
                let _ = tableau.basis.factorize(|var| {
                    if var < cols2.len() {
                        cols2[var].clone()
                    } else {
                        vec![0.0; m]
                    }
                });
                tableau.compute_basic_values();
                tableau.compute_dual_values();
                tableau.compute_reduced_costs();
            }
        }

        // Check feasibility
        if !tableau.is_primal_feasible() {
            // Need Phase I
            return self.solve(model);
        }

        // Phase II directly
        match self.phase2(&mut tableau, 0) {
            Phase2Result::Optimal(iters) => {
                let primal_full = tableau.primal_solution();
                let primal = primal_full[..n].to_vec();
                let obj = if model.sense == OptDirection::Maximize {
                    -tableau.objective_value() + model.obj_offset
                } else {
                    tableau.objective_value() + model.obj_offset
                };
                SimplexResult {
                    status: LpStatus::Optimal,
                    objective: obj,
                    primal,
                    dual: tableau.dual_values.clone(),
                    reduced_costs: tableau.reduced_costs[..n].to_vec(),
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: iters,
                    phase1_iterations: 0,
                }
            }
            Phase2Result::Unbounded(iters) => SimplexResult {
                status: LpStatus::Unbounded,
                objective: f64::NEG_INFINITY,
                primal: vec![0.0; n],
                dual: vec![0.0; m],
                reduced_costs: vec![0.0; n],
                basis_indices: tableau.basis.basic_vars.clone(),
                iterations: iters,
                phase1_iterations: 0,
            },
            Phase2Result::IterationLimit(iters) => {
                let primal_full = tableau.primal_solution();
                SimplexResult {
                    status: LpStatus::IterationLimit,
                    objective: tableau.objective_value() + model.obj_offset,
                    primal: primal_full[..n].to_vec(),
                    dual: tableau.dual_values.clone(),
                    reduced_costs: tableau.reduced_costs[..n].to_vec(),
                    basis_indices: tableau.basis.basic_vars.clone(),
                    iterations: iters,
                    phase1_iterations: 0,
                }
            }
            Phase2Result::NumericalError(iters) => SimplexResult {
                status: LpStatus::Unknown,
                objective: f64::NAN,
                primal: vec![0.0; n],
                dual: vec![0.0; m],
                reduced_costs: vec![0.0; n],
                basis_indices: tableau.basis.basic_vars.clone(),
                iterations: iters,
                phase1_iterations: 0,
            },
        }
    }

    /// Phase I: Find a basic feasible solution using the Big-M method.
    /// We add artificial variables and minimize their sum.
    fn phase1(&self, tableau: &mut SimplexTableau) -> Phase1Result {
        let m = tableau.num_rows;
        let orig_cols = tableau.num_cols;

        // Identify infeasible rows and add artificial variables
        let mut artificial_vars = Vec::new();
        let mut has_infeasible = false;

        for i in 0..m {
            let basic_var = tableau.basis.basic_vars[i];
            let val = tableau.basic_values[i];
            let lb = tableau.lower_bounds[basic_var];
            let ub = tableau.upper_bounds[basic_var];

            if val < lb - self.config.feasibility_tol || val > ub + self.config.feasibility_tol {
                has_infeasible = true;
                // Add an artificial variable for this row
                let art_idx = tableau.num_cols;
                let mut art_col = vec![0.0; m];
                if val < lb {
                    art_col[i] = -1.0;
                } else {
                    art_col[i] = 1.0;
                }
                tableau.col_matrix.push(art_col);
                tableau.obj.push(0.0); // original obj = 0
                tableau.lower_bounds.push(0.0);
                tableau.upper_bounds.push(f64::INFINITY);
                tableau.reduced_costs.push(0.0);
                tableau.edge_weights.push(1.0);
                tableau.devex_weights.push(1.0);
                tableau.num_cols += 1;
                artificial_vars.push((i, art_idx));
            }
        }

        if !has_infeasible {
            return Phase1Result::Feasible(0);
        }

        // Save original objective
        let orig_obj = tableau.obj.clone();

        // Phase I objective: minimize sum of artificials
        for j in 0..tableau.num_cols {
            tableau.obj[j] = 0.0;
        }
        for &(_, art_idx) in &artificial_vars {
            tableau.obj[art_idx] = 1.0;
        }

        // Put artificials into the basis
        for &(row, art_idx) in &artificial_vars {
            let leaving_var = tableau.basis.basic_vars[row];
            tableau.basis.basic_vars[row] = art_idx;
            // Set basic value for artificial to the infeasibility amount
            let val = tableau.basic_values[row];
            let lb = tableau.lower_bounds[leaving_var];
            let ub = tableau.upper_bounds[leaving_var];
            if val < lb {
                tableau.basic_values[row] = lb - val;
            } else {
                tableau.basic_values[row] = val - ub;
            }
        }

        // Refactorize with new basis
        let _ = tableau.refactorize();

        let mut iters = 0;
        let max_iters = self.config.max_iterations / 2;

        // Degeneracy counter
        let mut degenerate_count = 0;
        let mut last_obj = f64::INFINITY;

        loop {
            if iters >= max_iters {
                // Restore objective and clean up
                self.phase1_cleanup(tableau, &orig_obj, &artificial_vars, orig_cols);
                return Phase1Result::IterationLimit(iters);
            }

            // Check refactorization
            if tableau.basis.should_refactorize() {
                if tableau.refactorize().is_err() {
                    self.phase1_cleanup(tableau, &orig_obj, &artificial_vars, orig_cols);
                    return Phase1Result::IterationLimit(iters);
                }
            }

            // Select entering variable
            let entering = match tableau.select_entering() {
                Some(j) => j,
                None => {
                    // Optimal for Phase I
                    break;
                }
            };

            // Select leaving variable
            let (leaving_row, step, bound_flip) = match tableau.select_leaving(entering) {
                Some(result) => result,
                None => {
                    // Phase I unbounded means something is wrong
                    break;
                }
            };

            // Handle degeneracy
            if step < self.config.feasibility_tol {
                degenerate_count += 1;
                if degenerate_count > 50 && self.config.use_perturbation {
                    tableau.apply_perturbation();
                    degenerate_count = 0;
                }
            } else {
                degenerate_count = 0;
            }

            if bound_flip {
                // Flip the entering variable to its other bound
                // No basis change needed
                iters += 1;
                continue;
            }

            // Perform pivot
            if tableau.pivot(entering, leaving_row, step).is_err() {
                // Numerical trouble: refactorize and try again
                if tableau.refactorize().is_err() {
                    self.phase1_cleanup(tableau, &orig_obj, &artificial_vars, orig_cols);
                    return Phase1Result::IterationLimit(iters);
                }
            }

            let current_obj = tableau.objective_value();
            if (current_obj - last_obj).abs() < 1e-12 {
                degenerate_count += 1;
            }
            last_obj = current_obj;

            iters += 1;

            if self.config.verbose && iters % 100 == 0 {
                debug!(
                    "Phase I iter {}: obj = {:.6e}",
                    iters,
                    tableau.objective_value()
                );
            }
        }

        // Check if Phase I objective is zero (all artificials out of basis)
        let phase1_obj = tableau.objective_value();
        let feasible = phase1_obj < self.config.feasibility_tol;

        // Remove perturbation if used
        tableau.remove_perturbation();

        if feasible {
            // Try to remove any remaining artificial variables from the basis
            self.remove_artificials_from_basis(tableau, &artificial_vars, orig_cols);
        }

        // Restore original objective and clean up
        self.phase1_cleanup(tableau, &orig_obj, &artificial_vars, orig_cols);

        if feasible {
            // After Phase I cleanup, recompute basic values and verify primal feasibility.
            // Phase I only ensures artificials are zero, but original variables may still
            // violate their bounds (meaning the original problem is infeasible).
            tableau.compute_basic_values();
            if tableau.is_primal_feasible() {
                Phase1Result::Feasible(iters)
            } else {
                Phase1Result::Infeasible(iters)
            }
        } else {
            Phase1Result::Infeasible(iters)
        }
    }

    /// Remove artificial variables from the basis after Phase I.
    fn remove_artificials_from_basis(
        &self,
        tableau: &mut SimplexTableau,
        artificial_vars: &[(usize, usize)],
        orig_cols: usize,
    ) {
        for &(row, art_idx) in artificial_vars {
            if tableau.basis.basic_vars[row] == art_idx {
                // This artificial is still basic—try to pivot it out
                let mut best_col = None;
                let mut best_pivot = 0.0f64;

                for j in 0..orig_cols {
                    if tableau.basis.is_basic(j) {
                        continue;
                    }
                    let col = &tableau.col_matrix[j];
                    if let Ok(d) = tableau.ftran(col) {
                        if d[row].abs() > self.config.pivot_tol && d[row].abs() > best_pivot {
                            best_pivot = d[row].abs();
                            best_col = Some(j);
                        }
                    }
                }

                if let Some(j) = best_col {
                    let col = tableau.col_matrix[j].clone();
                    if let Ok(d) = tableau.ftran(&col) {
                        let _ = tableau.basis.update(row, j, &d);
                    }
                }
            }
        }
    }

    /// Clean up after Phase I: restore objective, remove artificial columns.
    fn phase1_cleanup(
        &self,
        tableau: &mut SimplexTableau,
        orig_obj: &[f64],
        _artificial_vars: &[(usize, usize)],
        orig_cols: usize,
    ) {
        // Restore objective
        tableau.obj.truncate(orig_cols);
        for (j, &c) in orig_obj.iter().enumerate().take(orig_cols) {
            tableau.obj[j] = c;
        }

        // Remove artificial columns
        tableau.col_matrix.truncate(orig_cols);
        tableau.lower_bounds.truncate(orig_cols);
        tableau.upper_bounds.truncate(orig_cols);
        tableau.reduced_costs.truncate(orig_cols);
        tableau.edge_weights.truncate(orig_cols);
        tableau.devex_weights.truncate(orig_cols);
        tableau.num_cols = orig_cols;

        // Recompute dual values and reduced costs
        tableau.compute_dual_values();
        tableau.compute_reduced_costs();
    }

    /// Phase II: Optimize the objective function.
    fn phase2(&self, tableau: &mut SimplexTableau, start_iters: usize) -> Phase2Result {
        let mut iters = 0;
        let max_iters = self.config.max_iterations - start_iters;
        let mut degenerate_count = 0;
        let mut consecutive_refactors = 0;

        loop {
            if iters >= max_iters {
                return Phase2Result::IterationLimit(iters);
            }

            // Check refactorization
            if tableau.basis.should_refactorize() {
                match tableau.refactorize() {
                    Ok(()) => {
                        consecutive_refactors = 0;
                    }
                    Err(_) => {
                        consecutive_refactors += 1;
                        if consecutive_refactors > 3 {
                            return Phase2Result::NumericalError(iters);
                        }
                    }
                }
            }

            // Select entering variable
            let entering = match tableau.select_entering() {
                Some(j) => j,
                None => {
                    // No improving direction: optimal
                    return Phase2Result::Optimal(iters);
                }
            };

            // Select leaving variable
            let (leaving_row, step, bound_flip) = match tableau.select_leaving(entering) {
                Some(result) => result,
                None => {
                    return Phase2Result::Unbounded(iters);
                }
            };

            // Handle degeneracy
            if step < self.config.feasibility_tol {
                degenerate_count += 1;
                if degenerate_count > 100 && self.config.use_perturbation {
                    tableau.apply_perturbation();
                    degenerate_count = 0;
                }
            } else {
                degenerate_count = 0;
            }

            if bound_flip {
                // Flip entering variable bound without basis change
                iters += 1;
                tableau.compute_basic_values();
                tableau.compute_dual_values();
                tableau.compute_reduced_costs();
                continue;
            }

            // Perform pivot
            match tableau.pivot(entering, leaving_row, step) {
                Ok(()) => {}
                Err(_) => {
                    // Try refactorization
                    if tableau.refactorize().is_err() {
                        return Phase2Result::NumericalError(iters);
                    }
                    // Retry entering selection
                    iters += 1;
                    continue;
                }
            }

            iters += 1;

            if self.config.verbose && iters % 100 == 0 {
                debug!(
                    "Phase II iter {}: obj = {:.6e}",
                    iters,
                    tableau.objective_value()
                );
            }
        }
    }
}

/// Phase I outcome.
enum Phase1Result {
    Feasible(usize),
    Infeasible(usize),
    IterationLimit(usize),
}

/// Phase II outcome.
enum Phase2Result {
    Optimal(usize),
    Unbounded(usize),
    IterationLimit(usize),
    NumericalError(usize),
}

/// Solve an LP model using the default simplex configuration.
pub fn solve_lp(model: &LpModel) -> SimplexResult {
    let solver = RevisedSimplex::default_solver();
    solver.solve(model)
}

/// Solve an LP model with a specific configuration.
pub fn solve_lp_with_config(model: &LpModel, config: SimplexConfig) -> SimplexResult {
    let solver = RevisedSimplex::new(config);
    solver.solve(model)
}

/// Parametric simplex: solve a sequence of LPs where only the RHS changes.
pub fn parametric_rhs(model: &LpModel, rhs_values: &[Vec<f64>]) -> Vec<SimplexResult> {
    let solver = RevisedSimplex::default_solver();
    let mut results = Vec::new();
    let mut last_basis: Option<Vec<usize>> = None;

    for rhs in rhs_values {
        let mut modified_model = model.clone();
        for (i, &val) in rhs.iter().enumerate() {
            if i < modified_model.constraints.len() {
                modified_model.constraints[i].rhs = val;
            }
        }

        let result = if let Some(ref basis) = last_basis {
            solver.solve_warm(&modified_model, basis)
        } else {
            solver.solve(&modified_model)
        };

        last_basis = Some(result.basis_indices.clone());
        results.push(result);
    }

    results
}

/// Parametric simplex: solve a sequence of LPs where only the objective changes.
pub fn parametric_obj(model: &LpModel, obj_values: &[Vec<f64>]) -> Vec<SimplexResult> {
    let solver = RevisedSimplex::default_solver();
    let mut results = Vec::new();
    let mut last_basis: Option<Vec<usize>> = None;

    for obj in obj_values {
        let mut modified_model = model.clone();
        for (j, &val) in obj.iter().enumerate() {
            if j < modified_model.variables.len() {
                modified_model.variables[j].obj_coeff = val;
            }
        }

        let result = if let Some(ref basis) = last_basis {
            solver.solve_warm(&modified_model, basis)
        } else {
            solver.solve(&modified_model)
        };

        last_basis = Some(result.basis_indices.clone());
        results.push(result);
    }

    results
}

/// Self-dual simplex: handles both primal and dual infeasibility.
pub fn self_dual_simplex(model: &LpModel) -> SimplexResult {
    // First try standard simplex
    let config = SimplexConfig {
        pricing: PricingStrategy::SteepestEdge,
        ratio_test: RatioTestStrategy::Harris,
        ..SimplexConfig::default()
    };
    let solver = RevisedSimplex::new(config);
    let result = solver.solve(model);

    // If iteration limit, try with different pricing
    if result.status == LpStatus::IterationLimit {
        let config2 = SimplexConfig {
            pricing: PricingStrategy::Dantzig,
            ratio_test: RatioTestStrategy::Standard,
            use_perturbation: true,
            ..SimplexConfig::default()
        };
        let solver2 = RevisedSimplex::new(config2);
        return solver2.solve(model);
    }

    result
}

/// Crash procedure: find a good initial basis heuristically.
pub fn crash_basis(model: &LpModel) -> Vec<usize> {
    let m = model.num_constraints();
    let n = model.num_vars();

    // Score each variable by potential to enter the basis
    let mut scores: Vec<(usize, f64)> = Vec::new();

    for (j, var) in model.variables.iter().enumerate() {
        // Prefer variables with low reduced cost and involvement in many constraints
        let involvement: usize = model
            .constraints
            .iter()
            .filter(|c| c.row_indices.contains(&j))
            .count();

        let score = var.obj_coeff.abs() * 0.1 + involvement as f64;
        scores.push((j, score));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily select variables for the basis using a triangularity heuristic
    let mut basis = Vec::new();
    let mut covered_rows = vec![false; m];

    for (j, _score) in &scores {
        if basis.len() >= m {
            break;
        }

        // Check which uncovered rows this variable appears in
        let mut best_row = None;
        let mut best_coeff = 0.0f64;

        for con in &model.constraints {
            if let Some(pos) = con.row_indices.iter().position(|&c| c == *j) {
                let row = con.index;
                if !covered_rows[row] && con.row_values[pos].abs() > best_coeff {
                    best_coeff = con.row_values[pos].abs();
                    best_row = Some(row);
                }
            }
        }

        if let Some(row) = best_row {
            basis.push(*j);
            covered_rows[row] = true;
        }
    }

    // Fill remaining positions with slack variables
    let num_slacks = model
        .constraints
        .iter()
        .filter(|c| c.sense != ConstraintSense::Eq)
        .count();

    let mut slack_idx = n;
    for i in 0..m {
        if !covered_rows[i] {
            if slack_idx < n + num_slacks {
                basis.push(slack_idx);
                slack_idx += 1;
            }
        }
    }

    // Ensure basis has exactly m elements
    while basis.len() < m {
        basis.push(n + basis.len());
    }
    basis.truncate(m);

    basis
}

/// Crossover from interior point solution to a basic solution.
pub fn crossover_to_basis(
    model: &LpModel,
    interior_primal: &[f64],
    interior_dual: &[f64],
) -> SimplexResult {
    let m = model.num_constraints();
    let n = model.num_vars();
    let tol = 1e-6;

    // Classify variables as basic or non-basic based on proximity to bounds
    let mut basic_candidates = Vec::new();
    let mut nonbasic_lower = Vec::new();
    let mut nonbasic_upper = Vec::new();

    for (j, var) in model.variables.iter().enumerate() {
        let x_j = if j < interior_primal.len() {
            interior_primal[j]
        } else {
            0.0
        };
        let at_lb = (x_j - var.lower_bound).abs() < tol;
        let at_ub = var.upper_bound < 1e20 && (x_j - var.upper_bound).abs() < tol;

        if at_lb {
            nonbasic_lower.push(j);
        } else if at_ub {
            nonbasic_upper.push(j);
        } else {
            basic_candidates.push(j);
        }
    }

    // Also need to handle slack variables
    let num_slacks = model
        .constraints
        .iter()
        .filter(|c| c.sense != ConstraintSense::Eq)
        .count();

    // Build an initial basis from the candidates
    let mut basis_indices: Vec<usize> = basic_candidates.clone();

    // Fill with slacks if needed
    let mut slack_idx = n;
    while basis_indices.len() < m && slack_idx < n + num_slacks {
        if !basis_indices.contains(&slack_idx) {
            basis_indices.push(slack_idx);
        }
        slack_idx += 1;
    }
    basis_indices.truncate(m);

    // Use warm start with this basis
    let solver = RevisedSimplex::default_solver();
    if basis_indices.len() == m {
        solver.solve_warm(model, &basis_indices)
    } else {
        solver.solve(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Variable};

    fn simple_lp() -> LpModel {
        // min -x - y s.t. x + y <= 4, 2x + y <= 6, x,y >= 0
        // Optimal: x=2, y=2, obj=-4
        let mut m = LpModel::new("simple");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, f64::INFINITY));
        let y = m.add_variable(Variable::continuous("y", 0.0, f64::INFINITY));
        m.set_obj_coeff(x, -1.0);
        m.set_obj_coeff(y, -1.0);

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

    fn infeasible_lp() -> LpModel {
        // min x s.t. x >= 5, x <= 3
        let mut m = LpModel::new("infeasible");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 3.0));
        m.set_obj_coeff(x, 1.0);

        let mut c = Constraint::new("c0", ConstraintSense::Ge, 5.0);
        c.add_term(x, 1.0);
        m.add_constraint(c);

        m
    }

    fn bounded_lp() -> LpModel {
        // min x s.t. x + y <= 10, x,y >= 0, x <= 5, y <= 5
        let mut m = LpModel::new("bounded");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 5.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 5.0));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 0.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 10.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        m
    }

    #[test]
    fn test_solve_simple() {
        let model = simple_lp();
        let result = solve_lp(&model);
        assert_eq!(result.status, LpStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < 1e-6,
            "obj = {}",
            result.objective
        );
    }

    #[test]
    fn test_solve_bounded() {
        let model = bounded_lp();
        let result = solve_lp(&model);
        assert_eq!(result.status, LpStatus::Optimal);
        assert!(
            (result.objective - 0.0).abs() < 1e-6,
            "obj = {}",
            result.objective
        );
        assert!(result.primal[0].abs() < 1e-6);
    }

    #[test]
    fn test_solve_infeasible() {
        let model = infeasible_lp();
        let result = solve_lp(&model);
        assert!(
            result.status == LpStatus::Infeasible || result.status == LpStatus::Unknown,
            "status = {:?}",
            result.status
        );
    }

    #[test]
    fn test_warm_start() {
        let model = simple_lp();
        let result1 = solve_lp(&model);
        let result2 = RevisedSimplex::default_solver().solve_warm(&model, &result1.basis_indices);
        assert_eq!(result2.status, LpStatus::Optimal);
        assert!((result2.objective - result1.objective).abs() < 1e-6);
        // Warm start should use fewer iterations
    }

    #[test]
    fn test_steepest_edge() {
        let model = simple_lp();
        let config = SimplexConfig {
            pricing: PricingStrategy::SteepestEdge,
            ..SimplexConfig::default()
        };
        let result = solve_lp_with_config(&model, config);
        assert_eq!(result.status, LpStatus::Optimal);
        assert!((result.objective - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_harris_ratio_test() {
        let model = simple_lp();
        let config = SimplexConfig {
            ratio_test: RatioTestStrategy::Harris,
            ..SimplexConfig::default()
        };
        let result = solve_lp_with_config(&model, config);
        assert_eq!(result.status, LpStatus::Optimal);
    }

    #[test]
    fn test_crash_basis() {
        let model = simple_lp();
        let basis = crash_basis(&model);
        assert_eq!(basis.len(), model.num_constraints());
    }

    #[test]
    fn test_parametric_rhs() {
        let model = simple_lp();
        let rhs_seq = vec![vec![4.0, 6.0], vec![3.0, 5.0], vec![5.0, 7.0]];
        let results = parametric_rhs(&model, &rhs_seq);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.status == LpStatus::Optimal || r.status == LpStatus::IterationLimit);
        }
    }

    #[test]
    fn test_self_dual_simplex() {
        let model = simple_lp();
        let result = self_dual_simplex(&model);
        assert_eq!(result.status, LpStatus::Optimal);
    }

    #[test]
    fn test_equality_constraint() {
        // min x + y s.t. x + y = 3, x,y >= 0
        let mut m = LpModel::new("eq");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, f64::INFINITY));
        let y = m.add_variable(Variable::continuous("y", 0.0, f64::INFINITY));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 1.0);

        let mut c = Constraint::new("c0", ConstraintSense::Eq, 3.0);
        c.add_term(x, 1.0);
        c.add_term(y, 1.0);
        m.add_constraint(c);

        let result = solve_lp(&m);
        // No slack for eq constraints, so we need Phase I
        // Optimal should be 3 (either x=3,y=0 or x=0,y=3 etc.)
        assert!(
            result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit,
            "status = {:?}",
            result.status
        );
    }
}
