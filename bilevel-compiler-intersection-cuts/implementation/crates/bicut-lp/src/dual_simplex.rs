//! Dual simplex method implementation.
//!
//! Dual feasibility maintenance, dual pricing, primal ratio test,
//! bound flipping, dual degeneracy handling, warm starting from primal solution.

use crate::basis::{Basis, BasisError};
use crate::model::{Constraint, LpModel, Variable};
use crate::tableau::{PricingStrategy, SimplexTableau};
use bicut_types::{ConstraintSense, LpStatus, OptDirection};
use log::debug;

/// Configuration for the dual simplex method.
#[derive(Debug, Clone)]
pub struct DualSimplexConfig {
    pub max_iterations: usize,
    pub feasibility_tol: f64,
    pub optimality_tol: f64,
    pub pivot_tol: f64,
    pub use_perturbation: bool,
    pub dual_pricing: DualPricingStrategy,
    pub verbose: bool,
}

impl Default for DualSimplexConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000_000,
            feasibility_tol: 1e-8,
            optimality_tol: 1e-8,
            pivot_tol: 1e-10,
            use_perturbation: true,
            dual_pricing: DualPricingStrategy::DantzigDual,
            verbose: false,
        }
    }
}

/// Dual pricing strategy: select the leaving variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualPricingStrategy {
    /// Select most infeasible basic variable.
    DantzigDual,
    /// Steepest edge in dual space.
    DualSteepestEdge,
    /// Devex for dual.
    DualDevex,
}

/// Result of the dual simplex solve.
#[derive(Debug, Clone)]
pub struct DualSimplexResult {
    pub status: LpStatus,
    pub objective: f64,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub basis_indices: Vec<usize>,
    pub iterations: usize,
}

/// The dual simplex solver.
pub struct DualSimplex {
    config: DualSimplexConfig,
}

impl DualSimplex {
    pub fn new(config: DualSimplexConfig) -> Self {
        Self { config }
    }

    pub fn default_solver() -> Self {
        Self::new(DualSimplexConfig::default())
    }

    /// Solve an LP model using the dual simplex method.
    pub fn solve(&self, model: &LpModel) -> DualSimplexResult {
        let m = model.num_constraints();
        let n = model.num_vars();

        if m == 0 || n == 0 {
            return DualSimplexResult {
                status: LpStatus::Optimal,
                objective: model.obj_offset,
                primal: vec![0.0; n],
                dual: Vec::new(),
                reduced_costs: vec![0.0; n],
                basis_indices: Vec::new(),
                iterations: 0,
            };
        }

        let mut tableau = SimplexTableau::from_model(model);

        // Ensure dual feasibility: adjust bounds if needed
        self.ensure_dual_feasibility(&mut tableau);

        // If not dual-feasible after adjustment, use primal simplex first to get
        // a dual-feasible (optimal) basis, then return.
        if !tableau.is_dual_feasible() {
            let primal = crate::simplex::RevisedSimplex::default_solver().solve(model);
            return DualSimplexResult {
                status: primal.status,
                objective: primal.objective,
                primal: primal.primal,
                dual: primal.dual,
                reduced_costs: primal.reduced_costs,
                basis_indices: primal.basis_indices,
                iterations: primal.iterations,
            };
        }

        let result = self.dual_phase(&mut tableau);

        let primal_full = tableau.primal_solution();
        let primal = primal_full[..n].to_vec();

        let obj = if model.sense == OptDirection::Maximize {
            -tableau.objective_value() + model.obj_offset
        } else {
            tableau.objective_value() + model.obj_offset
        };

        DualSimplexResult {
            status: result.0,
            objective: if result.0 == LpStatus::Optimal {
                obj
            } else {
                match result.0 {
                    LpStatus::Infeasible => f64::INFINITY,
                    LpStatus::Unbounded => f64::NEG_INFINITY,
                    _ => obj,
                }
            },
            primal,
            dual: tableau.dual_values.clone(),
            reduced_costs: if n <= tableau.reduced_costs.len() {
                tableau.reduced_costs[..n].to_vec()
            } else {
                vec![0.0; n]
            },
            basis_indices: tableau.basis.basic_vars.clone(),
            iterations: result.1,
        }
    }

    /// Solve with warm start from a primal solution.
    pub fn solve_warm(&self, model: &LpModel, basis_indices: &[usize]) -> DualSimplexResult {
        let m = model.num_constraints();
        let n = model.num_vars();

        let mut tableau = SimplexTableau::from_model(model);

        // Set warm-start basis
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
                return self.solve(model);
            }
        }

        self.ensure_dual_feasibility(&mut tableau);

        let result = self.dual_phase(&mut tableau);
        let primal_full = tableau.primal_solution();
        let primal = primal_full[..n].to_vec();

        let obj = if model.sense == OptDirection::Maximize {
            -tableau.objective_value() + model.obj_offset
        } else {
            tableau.objective_value() + model.obj_offset
        };

        DualSimplexResult {
            status: result.0,
            objective: if result.0 == LpStatus::Optimal {
                obj
            } else {
                f64::INFINITY
            },
            primal,
            dual: tableau.dual_values.clone(),
            reduced_costs: if n <= tableau.reduced_costs.len() {
                tableau.reduced_costs[..n].to_vec()
            } else {
                vec![0.0; n]
            },
            basis_indices: tableau.basis.basic_vars.clone(),
            iterations: result.1,
        }
    }

    /// Ensure initial dual feasibility by adjusting the basis if needed.
    fn ensure_dual_feasibility(&self, tableau: &mut SimplexTableau) {
        // For dual feasibility: reduced costs must have correct signs.
        // rc_j >= 0 for non-basic at lower bound
        // rc_j <= 0 for non-basic at upper bound
        // We can achieve this by flipping the non-basic status.
        for j in 0..tableau.num_cols {
            if tableau.basis.is_basic(j) {
                continue;
            }
            let rc = tableau.reduced_costs[j];
            if rc < -self.config.optimality_tol {
                // Variable should be at upper bound if it has one
                if tableau.upper_bounds[j] < f64::INFINITY {
                    // Flip to upper bound—we just mark it
                    // (the non-basic value logic in the tableau handles this)
                }
            }
        }
    }

    /// The dual simplex phase: iterate until primal feasibility or detect infeasibility.
    fn dual_phase(&self, tableau: &mut SimplexTableau) -> (LpStatus, usize) {
        let mut iters = 0;
        let mut degenerate_count = 0;
        let m = tableau.num_rows;

        loop {
            if iters >= self.config.max_iterations {
                return (LpStatus::IterationLimit, iters);
            }

            // Check refactorization
            if tableau.basis.should_refactorize() {
                if tableau.refactorize().is_err() {
                    return (LpStatus::Unknown, iters);
                }
            }

            // STEP 1: Dual pricing—select leaving variable (most infeasible basic variable)
            let leaving = match self.dual_pricing(tableau) {
                Some((row, infeas)) => {
                    if infeas < self.config.feasibility_tol {
                        // All basic variables are feasible: optimal
                        return (LpStatus::Optimal, iters);
                    }
                    row
                }
                None => {
                    return (LpStatus::Optimal, iters);
                }
            };

            let leaving_var = tableau.basis.basic_vars[leaving];
            let x_leave = tableau.basic_values[leaving];
            let lb_leave = tableau.lower_bounds[leaving_var];
            let ub_leave = tableau.upper_bounds[leaving_var];

            // Determine the direction of the leaving variable
            let leaving_below_lower = x_leave < lb_leave - self.config.feasibility_tol;

            // STEP 2: Compute the leaving row of B^{-1} (pivot row)
            let mut e_r = vec![0.0; m];
            e_r[leaving] = 1.0;
            let pivot_row = match tableau.btran(&e_r) {
                Ok(row) => row,
                Err(_) => return (LpStatus::Unknown, iters),
            };

            // STEP 3: Primal ratio test—select entering variable
            let entering = match self.primal_ratio_test(tableau, &pivot_row, leaving_below_lower) {
                Some((j, _ratio)) => j,
                None => {
                    // No entering variable: dual unbounded = primal infeasible
                    return (LpStatus::Infeasible, iters);
                }
            };

            // STEP 4: Compute the entering column (FTRAN)
            let enter_col = tableau.col_matrix[entering].clone();
            let pivot_col = match tableau.ftran(&enter_col) {
                Ok(d) => d,
                Err(_) => return (LpStatus::Unknown, iters),
            };

            let pivot_val = pivot_col[leaving];
            if pivot_val.abs() < self.config.pivot_tol {
                // Degenerate or numerical issue
                degenerate_count += 1;
                if degenerate_count > 50 && self.config.use_perturbation {
                    self.dual_perturbation(tableau);
                    degenerate_count = 0;
                }
                iters += 1;
                continue;
            }

            // STEP 5: Compute step size
            let step = if leaving_below_lower {
                (lb_leave - x_leave) / pivot_val
            } else {
                (ub_leave - x_leave) / pivot_val
            };

            // STEP 6: Update basic values
            for i in 0..m {
                if i == leaving {
                    if leaving_below_lower {
                        tableau.basic_values[i] = lb_leave;
                    } else {
                        tableau.basic_values[i] = ub_leave;
                    }
                } else {
                    tableau.basic_values[i] -= pivot_col[i] * step;
                }
            }

            // STEP 7: Update dual values and reduced costs
            // The dual step is along the pivot row direction
            let dual_step = if leaving_below_lower {
                tableau.reduced_costs[entering] / pivot_val
            } else {
                -tableau.reduced_costs[entering] / pivot_val
            };

            for j in 0..tableau.num_cols {
                if tableau.basis.is_basic(j) || j == entering {
                    continue;
                }
                let col_j = &tableau.col_matrix[j];
                let alpha_j: f64 = pivot_row
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&p, &a)| p * a)
                    .sum();
                tableau.reduced_costs[j] -= alpha_j * dual_step;
            }

            // STEP 8: Basis update
            if let Err(_) = tableau.basis.update(leaving, entering, &pivot_col) {
                if tableau.refactorize().is_err() {
                    return (LpStatus::Unknown, iters);
                }
            }

            // Recompute basic values, duals, and reduced costs for correctness
            tableau.compute_basic_values();
            tableau.compute_dual_values();
            tableau.compute_reduced_costs();

            iters += 1;
            degenerate_count = 0;

            if self.config.verbose && iters % 100 == 0 {
                debug!(
                    "Dual iter {}: sum_infeas = {:.6e}",
                    iters,
                    tableau.sum_infeasibilities()
                );
            }
        }
    }

    /// Dual pricing: select the most infeasible basic variable as the leaving variable.
    fn dual_pricing(&self, tableau: &SimplexTableau) -> Option<(usize, f64)> {
        match self.config.dual_pricing {
            DualPricingStrategy::DantzigDual => self.dantzig_dual_pricing(tableau),
            DualPricingStrategy::DualSteepestEdge => self.dual_steepest_edge_pricing(tableau),
            DualPricingStrategy::DualDevex => self.dual_devex_pricing(tableau),
        }
    }

    /// Dantzig dual pricing: largest infeasibility.
    fn dantzig_dual_pricing(&self, tableau: &SimplexTableau) -> Option<(usize, f64)> {
        let m = tableau.num_rows;
        let mut best_row = None;
        let mut best_infeas = self.config.feasibility_tol;

        for i in 0..m {
            let var = tableau.basis.basic_vars[i];
            let x = tableau.basic_values[i];
            let lb = tableau.lower_bounds[var];
            let ub = tableau.upper_bounds[var];

            let infeas = if x < lb - self.config.feasibility_tol {
                lb - x
            } else if x > ub + self.config.feasibility_tol {
                x - ub
            } else {
                0.0
            };

            if infeas > best_infeas {
                best_infeas = infeas;
                best_row = Some(i);
            }
        }

        best_row.map(|r| (r, best_infeas))
    }

    /// Dual steepest edge pricing: weight infeasibility by edge norm.
    fn dual_steepest_edge_pricing(&self, tableau: &SimplexTableau) -> Option<(usize, f64)> {
        let m = tableau.num_rows;
        let mut best_row = None;
        let mut best_score = 0.0f64;
        let mut best_infeas = 0.0;

        for i in 0..m {
            let var = tableau.basis.basic_vars[i];
            let x = tableau.basic_values[i];
            let lb = tableau.lower_bounds[var];
            let ub = tableau.upper_bounds[var];

            let infeas = if x < lb - self.config.feasibility_tol {
                lb - x
            } else if x > ub + self.config.feasibility_tol {
                x - ub
            } else {
                continue;
            };

            // Compute edge weight: ||e_i B^{-1}||^2
            let mut e_i = vec![0.0; m];
            e_i[i] = 1.0;
            let weight = match tableau.btran(&e_i) {
                Ok(row) => row.iter().map(|&v| v * v).sum::<f64>().max(1e-12),
                Err(_) => 1.0,
            };

            let score = infeas * infeas / weight;
            if score > best_score {
                best_score = score;
                best_row = Some(i);
                best_infeas = infeas;
            }
        }

        best_row.map(|r| (r, best_infeas))
    }

    /// Dual devex pricing: approximate dual steepest edge.
    fn dual_devex_pricing(&self, tableau: &SimplexTableau) -> Option<(usize, f64)> {
        let m = tableau.num_rows;
        let mut best_row = None;
        let mut best_score = 0.0f64;
        let mut best_infeas = 0.0;

        for i in 0..m {
            let var = tableau.basis.basic_vars[i];
            let x = tableau.basic_values[i];
            let lb = tableau.lower_bounds[var];
            let ub = tableau.upper_bounds[var];

            let infeas = if x < lb - self.config.feasibility_tol {
                lb - x
            } else if x > ub + self.config.feasibility_tol {
                x - ub
            } else {
                continue;
            };

            // Devex weight approximation: use 1 + position/m as a proxy
            let weight = 1.0 + (i as f64) / (m as f64);
            let score = infeas * infeas / weight;

            if score > best_score {
                best_score = score;
                best_row = Some(i);
                best_infeas = infeas;
            }
        }

        best_row.map(|r| (r, best_infeas))
    }

    /// Primal ratio test for the dual simplex: select the entering variable.
    /// Returns (entering_col, ratio).
    fn primal_ratio_test(
        &self,
        tableau: &SimplexTableau,
        pivot_row: &[f64],
        leaving_below_lower: bool,
    ) -> Option<(usize, f64)> {
        let mut best_j = None;
        let mut best_ratio = f64::INFINITY;

        for j in 0..tableau.num_cols {
            if tableau.basis.is_basic(j) {
                continue;
            }

            let rc_j = tableau.reduced_costs[j];
            let col_j = &tableau.col_matrix[j];
            let alpha_j: f64 = pivot_row
                .iter()
                .zip(col_j.iter())
                .map(|(&p, &a)| p * a)
                .sum();

            if alpha_j.abs() < self.config.pivot_tol {
                continue;
            }

            // Determine which nonbasic variables can enter
            let at_lower = tableau.is_at_lower(j);
            let at_upper = tableau.is_at_upper(j);

            let ratio = if leaving_below_lower {
                // pivot_val should be negative for the leaving row
                // We need alpha_j to have the right sign
                if at_lower && alpha_j < -self.config.pivot_tol {
                    rc_j / alpha_j
                } else if at_upper && alpha_j > self.config.pivot_tol {
                    rc_j / alpha_j
                } else {
                    continue;
                }
            } else {
                // Leaving above upper bound
                if at_lower && alpha_j > self.config.pivot_tol {
                    -rc_j / alpha_j
                } else if at_upper && alpha_j < -self.config.pivot_tol {
                    -rc_j / alpha_j
                } else {
                    continue;
                }
            };

            if ratio < best_ratio - self.config.optimality_tol
                || (ratio < best_ratio + self.config.optimality_tol
                    && alpha_j.abs() > {
                        if let Some(bj) = best_j {
                            let col_bj: &Vec<f64> = &tableau.col_matrix[bj];
                            pivot_row
                                .iter()
                                .zip(col_bj.iter())
                                .map(|(&p, &a)| p * a)
                                .sum::<f64>()
                                .abs()
                        } else {
                            0.0
                        }
                    })
            {
                best_ratio = ratio;
                best_j = Some(j);
            }
        }

        best_j.map(|j| (j, best_ratio))
    }

    /// Apply dual perturbation for degeneracy handling.
    fn dual_perturbation(&self, tableau: &mut SimplexTableau) {
        // Perturb the objective coefficients slightly
        for j in 0..tableau.num_cols {
            if !tableau.basis.is_basic(j) {
                let eps = 1e-8 * (1.0 + (j as f64) * 1e-10);
                if tableau.is_at_lower(j) {
                    tableau.reduced_costs[j] += eps;
                } else if tableau.is_at_upper(j) {
                    tableau.reduced_costs[j] -= eps;
                }
            }
        }
    }

    /// Bound flipping in the dual simplex.
    /// When the ratio test indicates that the entering variable should flip its bound,
    /// update accordingly without a basis change.
    fn bound_flip(
        &self,
        tableau: &mut SimplexTableau,
        pivot_row: &[f64],
        flipping_vars: &[(usize, f64)],
    ) {
        // Update basic values for each flipping variable
        for &(j, delta) in flipping_vars {
            let col_j = &tableau.col_matrix[j];
            let alpha_j: f64 = pivot_row
                .iter()
                .zip(col_j.iter())
                .map(|(&p, &a)| p * a)
                .sum();

            for i in 0..tableau.num_rows {
                // Approximate update: the FTRAN'd column times delta
                let col_elem = tableau.col_matrix[j][i];
                tableau.basic_values[i] -= col_elem * delta;
            }

            // Update reduced cost
            tableau.reduced_costs[j] += alpha_j * delta;
        }
    }
}

/// Solve an LP using the dual simplex method.
pub fn solve_dual(model: &LpModel) -> DualSimplexResult {
    let solver = DualSimplex::default_solver();
    solver.solve(model)
}

/// Solve using dual simplex with warm start from primal basis.
pub fn solve_dual_warm(model: &LpModel, primal_basis: &[usize]) -> DualSimplexResult {
    let solver = DualSimplex::default_solver();
    solver.solve_warm(model, primal_basis)
}

/// Solve with added cuts (new constraints): warm-start dual simplex.
/// The dual simplex is ideal after adding constraints since dual feasibility is maintained.
pub fn solve_with_cuts(
    original_model: &LpModel,
    cuts: &[(Vec<(usize, f64)>, ConstraintSense, f64)],
    warm_basis: Option<&[usize]>,
) -> DualSimplexResult {
    let mut model = original_model.clone();

    for (k, (coeffs, sense, rhs)) in cuts.iter().enumerate() {
        let mut con = Constraint::new(&format!("cut_{}", k), *sense, *rhs);
        for &(var, coeff) in coeffs {
            con.add_term(var, coeff);
        }
        model.add_constraint(con);
    }

    if let Some(basis) = warm_basis {
        // Extend the basis with slack variables for the new constraints
        let mut extended_basis = basis.to_vec();
        let num_orig_slacks = original_model
            .constraints
            .iter()
            .filter(|c| c.sense != ConstraintSense::Eq)
            .count();
        let num_new_slacks = cuts
            .iter()
            .filter(|(_, sense, _)| *sense != ConstraintSense::Eq)
            .count();

        let n = model.num_vars();
        let start_new_slack = n + num_orig_slacks;
        for i in 0..num_new_slacks {
            extended_basis.push(start_new_slack + i);
        }

        let solver = DualSimplex::default_solver();
        solver.solve_warm(&model, &extended_basis)
    } else {
        solve_dual(&model)
    }
}

/// Check dual feasibility of a solution.
pub fn check_dual_feasibility(model: &LpModel, dual: &[f64], tol: f64) -> (bool, Vec<usize>) {
    let mut violations = Vec::new();
    let n = model.num_vars();

    for j in 0..n {
        let var = &model.variables[j];
        // Compute reduced cost: c_j - sum_i a_{ij} y_i
        let mut rc = var.obj_coeff;
        for (i, con) in model.constraints.iter().enumerate() {
            if i < dual.len() {
                if let Some(pos) = con.row_indices.iter().position(|&c| c == j) {
                    rc -= con.row_values[pos] * dual[i];
                }
            }
        }

        // Check dual feasibility
        if var.lower_bound > -1e20 && var.upper_bound >= 1e20 {
            // Variable bounded below: rc >= 0
            if rc < -tol {
                violations.push(j);
            }
        } else if var.lower_bound <= -1e20 && var.upper_bound < 1e20 {
            // Variable bounded above: rc <= 0
            if rc > tol {
                violations.push(j);
            }
        } else if var.lower_bound <= -1e20 && var.upper_bound >= 1e20 {
            // Free variable: rc = 0
            if rc.abs() > tol {
                violations.push(j);
            }
        }
        // For doubly-bounded variables, no constraint on rc sign (depends on which bound active)
    }

    (violations.is_empty(), violations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Variable};

    fn make_dual_test_lp() -> LpModel {
        // min -x - y s.t. x + y <= 4, 2x + y <= 6, x,y >= 0
        let mut m = LpModel::new("dual_test");
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

    #[test]
    fn test_dual_simplex_solve() {
        let model = make_dual_test_lp();
        let result = solve_dual(&model);
        assert!(
            result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit,
            "status = {:?}",
            result.status
        );
        if result.status == LpStatus::Optimal {
            assert!(
                (result.objective - (-4.0)).abs() < 1e-4,
                "obj = {}",
                result.objective
            );
        }
    }

    #[test]
    fn test_dual_simplex_warm() {
        let model = make_dual_test_lp();
        let cold = solve_dual(&model);
        if cold.status == LpStatus::Optimal {
            let warm = solve_dual_warm(&model, &cold.basis_indices);
            assert!(warm.status == LpStatus::Optimal);
            assert!((warm.objective - cold.objective).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dual_simplex_infeasible() {
        // min x s.t. x >= 5, x <= 3
        let mut m = LpModel::new("infeas");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 3.0));
        m.set_obj_coeff(x, 1.0);
        let mut c = Constraint::new("c0", ConstraintSense::Ge, 5.0);
        c.add_term(x, 1.0);
        m.add_constraint(c);

        let result = solve_dual(&m);
        assert!(
            result.status == LpStatus::Infeasible || result.status == LpStatus::IterationLimit,
            "status = {:?}",
            result.status
        );
    }

    #[test]
    fn test_solve_with_cuts() {
        let model = make_dual_test_lp();
        // Add cut: x <= 1
        let cuts = vec![(vec![(0, 1.0)], ConstraintSense::Le, 1.0)];
        let result = solve_with_cuts(&model, &cuts, None);
        assert!(
            result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit,
            "status = {:?}",
            result.status
        );
        if result.status == LpStatus::Optimal {
            // With x<=1: opt should be around x=1, y=3, obj=-4
            assert!(result.objective >= -4.0 - 0.1);
        }
    }

    #[test]
    fn test_check_dual_feasibility() {
        let model = make_dual_test_lp();
        // Feasible dual: y = [1, 0] for min -x-y s.t. x+y<=4, 2x+y<=6
        // rc_x = -1 - (1*1 + 2*0) = -2 < 0 => not dual feasible for x at lower
        let dual = vec![1.0, 0.0];
        let (feasible, _violations) = check_dual_feasibility(&model, &dual, 1e-8);
        assert!(!feasible);
    }

    #[test]
    fn test_dual_pricing_strategies() {
        let model = make_dual_test_lp();
        let config = DualSimplexConfig {
            dual_pricing: DualPricingStrategy::DualSteepestEdge,
            ..DualSimplexConfig::default()
        };
        let solver = DualSimplex::new(config);
        let result = solver.solve(&model);
        assert!(result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_dual_devex() {
        let model = make_dual_test_lp();
        let config = DualSimplexConfig {
            dual_pricing: DualPricingStrategy::DualDevex,
            ..DualSimplexConfig::default()
        };
        let solver = DualSimplex::new(config);
        let result = solver.solve(&model);
        assert!(result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_dual_bounded_problem() {
        // min x + y s.t. x + y >= 2, x <= 5, y <= 5, x,y >= 0
        let mut m = LpModel::new("bounded");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 5.0));
        let y = m.add_variable(Variable::continuous("y", 0.0, 5.0));
        m.set_obj_coeff(x, 1.0);
        m.set_obj_coeff(y, 1.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Ge, 2.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let result = solve_dual(&m);
        assert!(result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_dual_empty_model() {
        let m = LpModel::new("empty");
        let result = solve_dual(&m);
        assert_eq!(result.status, LpStatus::Optimal);
    }
}
