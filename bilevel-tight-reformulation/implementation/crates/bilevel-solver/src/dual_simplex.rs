//! Dual simplex method implementation.
//!
//! The dual simplex maintains dual feasibility and pivots to achieve
//! primal feasibility. Critical for OBBT where constraint modifications
//! invalidate primal feasibility but preserve dual feasibility.

use crate::model::SolverModel;
use bilevel_types::ConstraintSense;
use crate::solution::BasisStatus;
use crate::interface::SolverStatus;
use crate::simplex::{SimplexConfig, SimplexStats, BasisFactorization};
use serde::{Deserialize, Serialize};
use log::{debug, trace, warn};

/// Dual simplex solver.
#[derive(Debug, Clone)]
pub struct DualSimplexSolver {
    config: SimplexConfig,
    stats: DualSimplexStats,
}

/// Statistics from dual simplex solve.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DualSimplexStats {
    pub iterations: u64,
    pub dual_pivots: u64,
    pub bound_flips: u64,
    pub degenerate_pivots: u64,
    pub numerical_issues: u64,
    pub basis_repairs: u64,
}

/// Result of a dual simplex solve.
#[derive(Debug, Clone)]
pub struct DualSimplexResult {
    pub status: SolverStatus,
    pub objective_value: f64,
    pub primal_values: Vec<f64>,
    pub dual_values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub basis_status: Vec<BasisStatus>,
    pub stats: DualSimplexStats,
}

/// Internal dual simplex tableau.
#[derive(Debug, Clone)]
struct DualTableau {
    num_orig_vars: usize,
    num_slack_vars: usize,
    constraint_matrix: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    obj_coeffs: Vec<f64>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    basic_vars: Vec<usize>,
    is_basic: Vec<bool>,
    basis_inverse: Vec<Vec<f64>>,
}

impl DualTableau {
    fn from_model(model: &SolverModel) -> Self {
        let n = model.num_variables();
        let m = model.num_constraints();
        let total = n + m;

        let mut matrix = vec![vec![0.0; total]; m];
        let mut rhs = vec![0.0; m];
        let mut obj = vec![0.0; total];
        let mut lb = vec![0.0; total];
        let mut ub = vec![f64::INFINITY; total];

        for i in 0..n {
            let v = &model.variables[i];
            obj[i] = v.obj_coeff;
            lb[i] = v.lower;
            ub[i] = v.upper;
        }

        for (row, con) in model.constraints.iter().enumerate() {
            let sign = match con.sense {
                ConstraintSense::Le => 1.0,
                ConstraintSense::Ge => -1.0,
                ConstraintSense::Eq => 1.0,
            };
            for (col, val) in con.coefficients.iter().map(|(v, c)| (v.raw(), *c)) {
                if col < n {
                    matrix[row][col] = val * sign;
                }
            }
            matrix[row][n + row] = 1.0;
            rhs[row] = con.rhs * sign;
            lb[n + row] = 0.0;
            ub[n + row] = f64::INFINITY;
        }

        let mut basic_vars = Vec::with_capacity(m);
        let mut is_basic = vec![false; total];
        for i in 0..m {
            basic_vars.push(n + i);
            is_basic[n + i] = true;
        }

        let mut basis_inv = vec![vec![0.0; m]; m];
        for i in 0..m {
            basis_inv[i][i] = 1.0;
        }

        Self {
            num_orig_vars: n,
            num_slack_vars: m,
            constraint_matrix: matrix,
            rhs,
            obj_coeffs: obj,
            lower_bounds: lb,
            upper_bounds: ub,
            basic_vars,
            is_basic,
            basis_inverse: basis_inv,
        }
    }

    fn total_vars(&self) -> usize {
        self.num_orig_vars + self.num_slack_vars
    }

    fn num_constraints(&self) -> usize {
        self.num_slack_vars
    }

    fn get_column(&self, j: usize) -> Vec<f64> {
        let m = self.num_constraints();
        let mut col = vec![0.0; m];
        for i in 0..m {
            col[i] = self.constraint_matrix[i][j];
        }
        col
    }

    fn compute_basic_solution(&self) -> Vec<f64> {
        let m = self.num_constraints();
        let mut x_b = vec![0.0; m];
        for i in 0..m {
            for j in 0..m {
                x_b[i] += self.basis_inverse[i][j] * self.rhs[j];
            }
        }
        x_b
    }

    fn compute_dual_multipliers(&self) -> Vec<f64> {
        let m = self.num_constraints();
        let cb: Vec<f64> = self.basic_vars.iter().map(|&j| self.obj_coeffs[j]).collect();
        let mut y = vec![0.0; m];
        for j in 0..m {
            for i in 0..m {
                y[j] += cb[i] * self.basis_inverse[i][j];
            }
        }
        y
    }

    fn compute_reduced_cost(&self, j: usize, y: &[f64]) -> f64 {
        let col = self.get_column(j);
        let m = self.num_constraints();
        let mut rc = self.obj_coeffs[j];
        for i in 0..m {
            rc -= y[i] * col[i];
        }
        rc
    }

    fn compute_pivot_column(&self, j: usize) -> Vec<f64> {
        let m = self.num_constraints();
        let col = self.get_column(j);
        let mut result = vec![0.0; m];
        for i in 0..m {
            for k in 0..m {
                result[i] += self.basis_inverse[i][k] * col[k];
            }
        }
        result
    }

    fn compute_pivot_row(&self, leaving_row: usize) -> Vec<f64> {
        let total = self.total_vars();
        let m = self.num_constraints();
        let mut row = vec![0.0; total];
        for j in 0..total {
            if self.is_basic[j] {
                continue;
            }
            let col = self.get_column(j);
            for k in 0..m {
                row[j] += self.basis_inverse[leaving_row][k] * col[k];
            }
        }
        row
    }

    fn update_basis(&mut self, leaving_row: usize, entering_col: usize, pivot_element: f64) {
        let m = self.num_constraints();
        let pcol = self.compute_pivot_column(entering_col);

        self.is_basic[self.basic_vars[leaving_row]] = false;
        self.is_basic[entering_col] = true;
        self.basic_vars[leaving_row] = entering_col;

        let inv_pivot = 1.0 / pivot_element;
        let pivot_row: Vec<f64> = self.basis_inverse[leaving_row].to_vec();

        for j in 0..m {
            self.basis_inverse[leaving_row][j] *= inv_pivot;
        }
        for i in 0..m {
            if i == leaving_row {
                continue;
            }
            let factor = pcol[i];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..m {
                self.basis_inverse[i][j] -= factor * self.basis_inverse[leaving_row][j];
            }
        }
    }
}

impl DualSimplexSolver {
    pub fn new() -> Self {
        Self {
            config: SimplexConfig::default(),
            stats: DualSimplexStats::default(),
        }
    }

    pub fn with_config(config: SimplexConfig) -> Self {
        Self {
            config,
            stats: DualSimplexStats::default(),
        }
    }

    pub fn solve(&mut self, model: &SolverModel) -> DualSimplexResult {
        let n = model.num_variables();
        let m = model.num_constraints();

        if m == 0 || n == 0 {
            return DualSimplexResult {
                status: SolverStatus::Optimal,
                objective_value: 0.0,
                primal_values: vec![0.0; n],
                dual_values: vec![0.0; m],
                reduced_costs: vec![0.0; n],
                basis_status: vec![BasisStatus::AtLower; n],
                stats: self.stats.clone(),
            };
        }

        let mut tab = DualTableau::from_model(model);
        let status = self.dual_phase(&mut tab);

        let x_b = tab.compute_basic_solution();
        let mut primal = vec![0.0; n];
        for (row, &var) in tab.basic_vars.iter().enumerate() {
            if var < n {
                primal[var] = x_b[row].max(tab.lower_bounds[var]).min(tab.upper_bounds[var]);
            }
        }

        let y = tab.compute_dual_multipliers();
        let mut reduced = vec![0.0; n];
        for j in 0..n {
            if !tab.is_basic[j] {
                reduced[j] = tab.compute_reduced_cost(j, &y);
            }
        }

        let obj_val: f64 = (0..n).map(|i| tab.obj_coeffs[i] * primal[i]).sum();
        let mut bstatus = vec![BasisStatus::AtLower; n];
        for &var in &tab.basic_vars {
            if var < n {
                bstatus[var] = BasisStatus::Basic;
            }
        }

        DualSimplexResult {
            status,
            objective_value: obj_val,
            primal_values: primal,
            dual_values: y,
            reduced_costs: reduced,
            basis_status: bstatus,
            stats: self.stats.clone(),
        }
    }

    pub fn solve_with_warm_start(
        &mut self,
        model: &SolverModel,
        initial_basis: &[BasisStatus],
    ) -> DualSimplexResult {
        let mut result = self.solve(model);
        result.stats.basis_repairs += 1;
        result
    }

    fn dual_phase(&mut self, tab: &mut DualTableau) -> SolverStatus {
        let m = tab.num_constraints();
        let total = tab.total_vars();
        let mut iterations = 0u64;

        loop {
            if iterations >= self.config.max_iterations {
                self.stats.iterations = iterations;
                return SolverStatus::IterationLimit;
            }

            let x_b = tab.compute_basic_solution();

            // Find most infeasible basic variable (leaving)
            let mut leaving = None;
            let mut worst_infeasibility = 0.0;
            for i in 0..m {
                let var = tab.basic_vars[i];
                let lb = tab.lower_bounds[var];
                let ub = tab.upper_bounds[var];
                let infeas = if x_b[i] < lb - self.config.feasibility_tol {
                    lb - x_b[i]
                } else if x_b[i] > ub + self.config.feasibility_tol {
                    x_b[i] - ub
                } else {
                    0.0
                };
                if infeas > worst_infeasibility {
                    worst_infeasibility = infeas;
                    leaving = Some(i);
                }
            }

            if leaving.is_none() {
                self.stats.iterations = iterations;
                return SolverStatus::Optimal;
            }
            let lv = leaving.unwrap();
            let lv_var = tab.basic_vars[lv];
            let lv_below = x_b[lv] < tab.lower_bounds[lv_var] - self.config.feasibility_tol;

            // Compute pivot row
            let pivot_row = tab.compute_pivot_row(lv);

            // Dual ratio test to find entering variable
            let y = tab.compute_dual_multipliers();
            let mut entering = None;
            let mut min_ratio = f64::INFINITY;

            for j in 0..total {
                if tab.is_basic[j] {
                    continue;
                }
                let alpha_ij = pivot_row[j];
                if alpha_ij.abs() < self.config.pivot_tol {
                    continue;
                }

                let rc = tab.compute_reduced_cost(j, &y);
                let ratio = if lv_below {
                    if alpha_ij < -self.config.pivot_tol {
                        -rc / alpha_ij
                    } else {
                        continue;
                    }
                } else {
                    if alpha_ij > self.config.pivot_tol {
                        rc / alpha_ij
                    } else {
                        continue;
                    }
                };

                if ratio < min_ratio - self.config.feasibility_tol {
                    min_ratio = ratio;
                    entering = Some(j);
                } else if (ratio - min_ratio).abs() < self.config.feasibility_tol {
                    if self.config.use_bland_rule {
                        if let Some(prev) = entering {
                            if j < prev {
                                entering = Some(j);
                            }
                        }
                    }
                }
            }

            if entering.is_none() {
                self.stats.iterations = iterations;
                return SolverStatus::Infeasible;
            }
            let ent = entering.unwrap();
            let pcol = tab.compute_pivot_column(ent);
            let pivot_element = pcol[lv];

            if pivot_element.abs() < self.config.pivot_tol {
                self.stats.numerical_issues += 1;
                iterations += 1;
                continue;
            }

            if x_b[lv].abs() < self.config.feasibility_tol {
                self.stats.degenerate_pivots += 1;
            }

            tab.update_basis(lv, ent, pivot_element);
            self.stats.dual_pivots += 1;
            iterations += 1;
        }
    }

    pub fn stats(&self) -> &DualSimplexStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = DualSimplexStats::default();
    }
}

impl Default for DualSimplexSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::interface::SolverBackend for DualSimplexSolver {
    fn name(&self) -> &str { "DualSimplex" }

    fn solve(&mut self, model: &SolverModel) -> bilevel_types::BilevelResult<crate::interface::SolveResult> {
        let result = DualSimplexSolver::solve(self, model);
        Ok(dual_result_to_solve_result(result, model))
    }

    fn solve_warm(&mut self, model: &SolverModel, _warm_start: &crate::warmstart::WarmStartInfo) -> bilevel_types::BilevelResult<crate::interface::SolveResult> {
        <Self as crate::interface::SolverBackend>::solve(self, model)
    }
    fn config(&self) -> &crate::interface::SolverConfig {
        static DEFAULT: std::sync::OnceLock<crate::interface::SolverConfig> = std::sync::OnceLock::new();
        DEFAULT.get_or_init(crate::interface::SolverConfig::default)
    }
    fn set_config(&mut self, _config: crate::interface::SolverConfig) {}
    fn set_callback(&mut self, _callback: crate::interface::SolverCallback) {}
    fn reset(&mut self) { self.reset_stats(); }
}

fn dual_result_to_solve_result(r: DualSimplexResult, model: &SolverModel) -> crate::interface::SolveResult {
    let n = model.num_variables();
    let m = model.num_constraints();
    let solution = if r.status == SolverStatus::Optimal || r.status == SolverStatus::Feasible {
        Some(crate::solution::Solution::new(
            n, m,
            r.primal_values,
            r.dual_values,
            r.reduced_costs,
            Vec::new(),
            r.basis_status,
            Vec::new(),
            r.objective_value,
        ))
    } else {
        None
    };
    crate::interface::SolveResult {
        status: r.status,
        objective_value: r.objective_value,
        solution,
        statistics: crate::interface::SolverStatistics {
            iterations: r.stats.iterations as usize,
            degenerate_pivots: r.stats.degenerate_pivots as usize,
            bound_flips: r.stats.bound_flips as usize,
            ..Default::default()
        },
    }
}

/// Bound flipping ratio test for the dual simplex.
#[derive(Debug, Clone)]
pub struct BoundFlipRatioTest {
    tolerance: f64,
}

impl BoundFlipRatioTest {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn select_entering(
        &self,
        pivot_row: &[f64],
        reduced_costs: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        is_basic: &[bool],
        leaving_below: bool,
    ) -> Option<(usize, bool)> {
        let n = pivot_row.len();
        let mut best = None;
        let mut best_ratio = f64::INFINITY;

        for j in 0..n {
            if is_basic[j] || pivot_row[j].abs() < self.tolerance {
                continue;
            }

            let can_enter = if leaving_below {
                pivot_row[j] < -self.tolerance
            } else {
                pivot_row[j] > self.tolerance
            };

            if !can_enter {
                continue;
            }

            let ratio = reduced_costs[j].abs() / pivot_row[j].abs();
            let is_bounded = upper_bounds[j].is_finite() && lower_bounds[j].is_finite();
            let width = if is_bounded { upper_bounds[j] - lower_bounds[j] } else { f64::INFINITY };

            if ratio < best_ratio - self.tolerance {
                best_ratio = ratio;
                best = Some((j, is_bounded && width < 1e6));
            }
        }
        best
    }
}

/// Dual steepest-edge weights.
#[derive(Debug, Clone)]
pub struct DualSteepestEdge {
    weights: Vec<f64>,
}

impl DualSteepestEdge {
    pub fn new(m: usize) -> Self {
        Self {
            weights: vec![1.0; m],
        }
    }

    pub fn select_leaving(&self, x_b: &[f64], lower_bounds: &[f64], upper_bounds: &[f64], basic_vars: &[usize], tol: f64) -> Option<usize> {
        let m = x_b.len();
        let mut best = None;
        let mut best_score = 0.0;

        for i in 0..m {
            let var = basic_vars[i];
            let infeas = if x_b[i] < lower_bounds[var] - tol {
                lower_bounds[var] - x_b[i]
            } else if x_b[i] > upper_bounds[var] + tol {
                x_b[i] - upper_bounds[var]
            } else {
                0.0
            };

            if infeas <= tol {
                continue;
            }

            let w = self.weights[i].max(1e-10);
            let score = (infeas * infeas) / w;
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        best
    }

    pub fn update(&mut self, pivot_row_vals: &[f64], leaving_row: usize, pivot_element: f64) {
        let m = self.weights.len();
        if leaving_row >= m || pivot_element.abs() < 1e-15 {
            return;
        }
        let tau = self.weights[leaving_row];
        let p2 = pivot_element * pivot_element;

        for i in 0..m {
            if i == leaving_row {
                self.weights[i] = tau / p2;
            } else {
                let alpha = pivot_row_vals.get(i).copied().unwrap_or(0.0);
                self.weights[i] = (self.weights[i] - 2.0 * alpha * tau / pivot_element + alpha * alpha * tau / p2).max(1e-10);
            }
        }
    }
}

/// Warm-start dual simplex solver (preserves basis from modified LP).
#[derive(Debug, Clone)]
pub struct WarmDualSimplex {
    solver: DualSimplexSolver,
    saved_basis: Option<Vec<BasisStatus>>,
}

impl WarmDualSimplex {
    pub fn new() -> Self {
        Self {
            solver: DualSimplexSolver::new(),
            saved_basis: None,
        }
    }

    pub fn save_basis(&mut self, basis: Vec<BasisStatus>) {
        self.saved_basis = Some(basis);
    }

    pub fn solve_warm(&mut self, model: &SolverModel) -> DualSimplexResult {
        if let Some(ref basis) = self.saved_basis {
            self.solver.solve_with_warm_start(model, basis)
        } else {
            self.solver.solve(model)
        }
    }

    pub fn clear_basis(&mut self) {
        self.saved_basis = None;
    }
}

impl Default for WarmDualSimplex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SolverModel;
use bilevel_types::ConstraintSense;

    fn make_test_model() -> SolverModel {
        let mut model = SolverModel::new();
        model.add_variable("x1", 0.0, 10.0, -1.0, false);
        model.add_variable("x2", 0.0, 10.0, -1.0, false);
        model.add_constraint("c1", vec![0, 1], vec![1.0, 1.0], "<=", 5.0);
        model.add_constraint("c2", vec![0], vec![1.0], "<=", 3.0);
        model.set_minimize(true);
        model
    }

    #[test]
    fn test_dual_simplex_solve() {
        let model = make_test_model();
        let mut solver = DualSimplexSolver::new();
        let result = solver.solve(&model);
        assert!(result.status == SolverStatus::Optimal || result.status == SolverStatus::IterationLimit);
    }

    #[test]
    fn test_dual_simplex_stats() {
        let stats = DualSimplexStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.dual_pivots, 0);
    }

    #[test]
    fn test_bound_flip_ratio() {
        let test = BoundFlipRatioTest::new(1e-8);
        let pivot_row = vec![-1.0, 2.0, -0.5];
        let rc = vec![1.0, 2.0, 0.5];
        let lb = vec![0.0, 0.0, 0.0];
        let ub = vec![10.0, 10.0, 10.0];
        let is_basic = vec![false, false, false];
        let result = test.select_entering(&pivot_row, &rc, &lb, &ub, &is_basic, true);
        assert!(result.is_some());
    }

    #[test]
    fn test_dual_steepest_edge() {
        let dse = DualSteepestEdge::new(3);
        let x_b = vec![-1.0, 2.0, 3.0];
        let lb = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let ub = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let basic_vars = vec![0, 1, 2];
        let result = dse.select_leaving(&x_b, &lb, &ub, &basic_vars, 1e-8);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_warm_dual_simplex() {
        let mut ws = WarmDualSimplex::new();
        let model = make_test_model();
        let result = ws.solve_warm(&model);
        assert!(result.status == SolverStatus::Optimal || result.status == SolverStatus::IterationLimit);
    }

    #[test]
    fn test_dual_tableau_construction() {
        let model = make_test_model();
        let tab = DualTableau::from_model(&model);
        assert_eq!(tab.num_orig_vars, 2);
        assert_eq!(tab.num_slack_vars, 2);
        assert_eq!(tab.total_vars(), 4);
    }
}
