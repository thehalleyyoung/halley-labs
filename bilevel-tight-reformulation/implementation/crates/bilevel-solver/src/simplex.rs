//! Simplex method implementation for linear programming.
//!
//! Implements both Phase I (feasibility) and Phase II (optimality) of the
//! revised simplex method with Bland's anti-cycling rule.

use crate::model::{SolverModel, ColumnData};
use bilevel_types::ConstraintSense;
use crate::solution::{Solution, BasisStatus};
use crate::interface::{SolverStatus, SolverStatistics};
use serde::{Deserialize, Serialize};
use log::{debug, trace, warn};

/// Configuration for the simplex solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplexConfig {
    pub max_iterations: u64,
    pub feasibility_tol: f64,
    pub optimality_tol: f64,
    pub pivot_tol: f64,
    pub use_bland_rule: bool,
    pub perturbation: f64,
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            feasibility_tol: 1e-8,
            optimality_tol: 1e-8,
            pivot_tol: 1e-10,
            use_bland_rule: true,
            perturbation: 1e-12,
        }
    }
}

/// Internal basis representation for the simplex method.
#[derive(Debug, Clone)]
struct Basis {
    basic_vars: Vec<usize>,
    basis_matrix_inv: Vec<Vec<f64>>,
    is_basic: Vec<bool>,
    basic_costs: Vec<f64>,
}

impl Basis {
    fn new(m: usize, n: usize) -> Self {
        let mut is_basic = vec![false; n + m];
        let mut basic_vars = Vec::with_capacity(m);
        for i in 0..m {
            is_basic[n + i] = true;
            basic_vars.push(n + i);
        }
        let mut basis_inv = vec![vec![0.0; m]; m];
        for i in 0..m {
            basis_inv[i][i] = 1.0;
        }
        Self {
            basic_vars,
            basis_matrix_inv: basis_inv,
            is_basic,
            basic_costs: vec![0.0; m],
        }
    }

    fn update(&mut self, leaving_row: usize, entering_col: usize, pivot_column: &[f64], pivot_element: f64) {
        let m = self.basic_vars.len();
        self.is_basic[self.basic_vars[leaving_row]] = false;
        self.is_basic[entering_col] = true;
        self.basic_vars[leaving_row] = entering_col;

        let inv_pivot = 1.0 / pivot_element;
        let pivot_row_old: Vec<f64> = self.basis_matrix_inv[leaving_row].clone();
        for j in 0..m {
            self.basis_matrix_inv[leaving_row][j] *= inv_pivot;
        }
        for i in 0..m {
            if i == leaving_row {
                continue;
            }
            let factor = pivot_column[i];
            if factor.abs() < 1e-15 {
                continue;
            }
            for j in 0..m {
                self.basis_matrix_inv[i][j] -= factor * self.basis_matrix_inv[leaving_row][j];
            }
        }
    }

    fn compute_basic_solution(&self, rhs: &[f64]) -> Vec<f64> {
        let m = self.basic_vars.len();
        let mut x_b = vec![0.0; m];
        for i in 0..m {
            let mut val = 0.0;
            for j in 0..m {
                val += self.basis_matrix_inv[i][j] * rhs[j];
            }
            x_b[i] = val;
        }
        x_b
    }

    fn compute_dual(&self, cb: &[f64]) -> Vec<f64> {
        let m = self.basic_vars.len();
        let mut y = vec![0.0; m];
        for j in 0..m {
            let mut val = 0.0;
            for i in 0..m {
                val += cb[i] * self.basis_matrix_inv[i][j];
            }
            y[j] = val;
        }
        y
    }

    fn compute_pivot_column(&self, col: &[f64]) -> Vec<f64> {
        let m = self.basic_vars.len();
        let mut result = vec![0.0; m];
        for i in 0..m {
            let mut val = 0.0;
            for j in 0..m {
                val += self.basis_matrix_inv[i][j] * col[j];
            }
            result[i] = val;
        }
        result
    }
}

/// Tableau representation for the simplex method.
#[derive(Debug, Clone)]
struct Tableau {
    num_vars: usize,
    num_constraints: usize,
    constraint_matrix: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    obj_coeffs: Vec<f64>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl Tableau {
    fn from_model(model: &SolverModel) -> Self {
        let n = model.num_variables();
        let m = model.num_constraints();
        let total_cols = n + m;
        let mut matrix = vec![vec![0.0; total_cols]; m];
        let mut rhs = vec![0.0; m];
        let mut obj = vec![0.0; total_cols];
        let mut lb = vec![0.0; total_cols];
        let mut ub = vec![f64::INFINITY; total_cols];

        for i in 0..n {
            let v = &model.variables[i];
            obj[i] = v.obj_coeff;
            lb[i] = v.lower;
            ub[i] = v.upper;
        }
        for i in 0..m {
            lb[n + i] = 0.0;
            ub[n + i] = f64::INFINITY;
        }

        for (row_idx, con) in model.constraints.iter().enumerate() {
            let sign = match con.sense {
                ConstraintSense::Le => 1.0,
                ConstraintSense::Ge => -1.0,
                ConstraintSense::Eq => 1.0,
            };
            for (col, val) in con.coefficients.iter().map(|(v, c)| (v.raw(), *c)) {
                if col < n {
                    matrix[row_idx][col] = val * sign;
                }
            }
            matrix[row_idx][n + row_idx] = 1.0;
            rhs[row_idx] = con.rhs * sign;
            if sign < 0.0 {
                matrix[row_idx][n + row_idx] = -1.0;
                rhs[row_idx] = -rhs[row_idx];
                lb[n + row_idx] = 0.0;
                ub[n + row_idx] = f64::INFINITY;
            }
        }

        Self {
            num_vars: total_cols,
            num_constraints: m,
            constraint_matrix: matrix,
            rhs,
            obj_coeffs: obj,
            lower_bounds: lb,
            upper_bounds: ub,
        }
    }

    fn get_column(&self, j: usize) -> Vec<f64> {
        let m = self.num_constraints;
        let mut col = vec![0.0; m];
        for i in 0..m {
            col[i] = self.constraint_matrix[i][j];
        }
        col
    }
}

/// Primal simplex method solver.
#[derive(Debug, Clone)]
pub struct SimplexSolver {
    config: SimplexConfig,
    stats: SimplexStats,
}

/// Statistics from a simplex solve.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimplexStats {
    pub iterations: u64,
    pub phase1_iterations: u64,
    pub phase2_iterations: u64,
    pub degenerate_pivots: u64,
    pub numerical_issues: u64,
}

/// Result of a simplex solve.
#[derive(Debug, Clone)]
pub struct SimplexResult {
    pub status: SolverStatus,
    pub objective_value: f64,
    pub primal_values: Vec<f64>,
    pub dual_values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub basis_status: Vec<BasisStatus>,
    pub stats: SimplexStats,
}

impl SimplexSolver {
    pub fn new() -> Self {
        Self {
            config: SimplexConfig::default(),
            stats: SimplexStats::default(),
        }
    }

    pub fn with_config(config: SimplexConfig) -> Self {
        Self {
            config,
            stats: SimplexStats::default(),
        }
    }

    pub fn solve(&mut self, model: &SolverModel) -> SimplexResult {
        let n_orig = model.num_variables();
        let m = model.num_constraints();
        if m == 0 || n_orig == 0 {
            return SimplexResult {
                status: SolverStatus::Optimal,
                objective_value: 0.0,
                primal_values: vec![0.0; n_orig],
                dual_values: vec![0.0; m],
                reduced_costs: vec![0.0; n_orig],
                basis_status: vec![BasisStatus::AtLower; n_orig],
                stats: self.stats.clone(),
            };
        }

        let mut tableau = Tableau::from_model(model);
        let n = tableau.num_vars;

        // Phase I: find basic feasible solution
        let phase1_result = self.phase1(&mut tableau);
        if phase1_result.is_none() {
            return SimplexResult {
                status: SolverStatus::Infeasible,
                objective_value: f64::NAN,
                primal_values: vec![0.0; n_orig],
                dual_values: vec![0.0; m],
                reduced_costs: vec![0.0; n_orig],
                basis_status: vec![BasisStatus::AtLower; n_orig],
                stats: self.stats.clone(),
            };
        }
        let mut basis = phase1_result.unwrap();

        // Phase II: optimize
        let result = self.phase2(&tableau, &mut basis);
        let x_b = basis.compute_basic_solution(&tableau.rhs);
        let mut primal = vec![0.0; n_orig];
        for (row, &var) in basis.basic_vars.iter().enumerate() {
            if var < n_orig {
                primal[var] = x_b[row];
            }
        }
        for i in 0..n_orig {
            primal[i] = primal[i].max(tableau.lower_bounds[i]).min(tableau.upper_bounds[i]);
        }

        let cb: Vec<f64> = basis.basic_vars.iter().map(|&j| tableau.obj_coeffs[j]).collect();
        let dual = basis.compute_dual(&cb);

        let mut reduced = vec![0.0; n_orig];
        for j in 0..n_orig {
            if !basis.is_basic[j] {
                let col = tableau.get_column(j);
                let mut rc = tableau.obj_coeffs[j];
                for i in 0..m {
                    rc -= dual[i] * col[i];
                }
                reduced[j] = rc;
            }
        }

        let mut obj_val = 0.0;
        for i in 0..n_orig {
            obj_val += tableau.obj_coeffs[i] * primal[i];
        }

        let mut bstatus = vec![BasisStatus::AtLower; n_orig];
        for (row, &var) in basis.basic_vars.iter().enumerate() {
            if var < n_orig {
                bstatus[var] = BasisStatus::Basic;
            }
        }

        SimplexResult {
            status: result,
            objective_value: obj_val,
            primal_values: primal,
            dual_values: dual,
            reduced_costs: reduced,
            basis_status: bstatus,
            stats: self.stats.clone(),
        }
    }

    fn phase1(&mut self, tableau: &mut Tableau) -> Option<Basis> {
        let m = tableau.num_constraints;
        let n = tableau.num_vars;

        let mut rhs_positive = true;
        for i in 0..m {
            if tableau.rhs[i] < -self.config.feasibility_tol {
                for j in 0..n {
                    tableau.constraint_matrix[i][j] = -tableau.constraint_matrix[i][j];
                }
                tableau.rhs[i] = -tableau.rhs[i];
            }
        }

        let mut basis = Basis::new(m, n);
        let x_b = basis.compute_basic_solution(&tableau.rhs);
        let mut all_nonneg = true;
        for i in 0..m {
            if x_b[i] < -self.config.feasibility_tol {
                all_nonneg = false;
                break;
            }
        }

        if all_nonneg {
            return Some(basis);
        }

        let n_art = m;
        let total = n + n_art;
        let mut art_obj = vec![0.0; total];
        for i in 0..n_art {
            art_obj[n + i] = 1.0;
        }

        let mut extended_matrix = vec![vec![0.0; total]; m];
        for i in 0..m {
            for j in 0..n {
                extended_matrix[i][j] = tableau.constraint_matrix[i][j];
            }
            extended_matrix[i][n + i] = 1.0;
        }

        let mut art_basis = Basis::new(m, total);
        for i in 0..m {
            art_basis.basic_vars[i] = n + i;
            art_basis.is_basic[n + i] = true;
        }

        let art_tableau = Tableau {
            num_vars: total,
            num_constraints: m,
            constraint_matrix: extended_matrix.clone(),
            rhs: tableau.rhs.clone(),
            obj_coeffs: art_obj,
            lower_bounds: vec![0.0; total],
            upper_bounds: vec![f64::INFINITY; total],
        };

        let mut iterations = 0u64;
        loop {
            if iterations >= self.config.max_iterations {
                return None;
            }
            let x_b = art_basis.compute_basic_solution(&art_tableau.rhs);
            let cb: Vec<f64> = art_basis.basic_vars.iter().map(|&j| art_tableau.obj_coeffs[j]).collect();
            let dual = art_basis.compute_dual(&cb);

            let mut entering = None;
            let mut best_rc = -self.config.optimality_tol;
            for j in 0..total {
                if art_basis.is_basic[j] {
                    continue;
                }
                let col = art_tableau.get_column(j);
                let mut rc = art_tableau.obj_coeffs[j];
                for i in 0..m {
                    rc -= dual[i] * col[i];
                }
                if self.config.use_bland_rule {
                    if rc < -self.config.optimality_tol {
                        entering = Some(j);
                        break;
                    }
                } else if rc < best_rc {
                    best_rc = rc;
                    entering = Some(j);
                }
            }

            if entering.is_none() {
                break;
            }
            let ent = entering.unwrap();

            let col = art_tableau.get_column(ent);
            let pivot_col = art_basis.compute_pivot_column(&col);

            let leaving = self.ratio_test(&x_b, &pivot_col, m);
            if leaving.is_none() {
                return None;
            }
            let lv = leaving.unwrap();
            art_basis.update(lv, ent, &pivot_col, pivot_col[lv]);
            iterations += 1;
        }
        self.stats.phase1_iterations = iterations;

        let x_b = art_basis.compute_basic_solution(&art_tableau.rhs);
        let mut sum_art = 0.0;
        for (row, &var) in art_basis.basic_vars.iter().enumerate() {
            if var >= n {
                sum_art += x_b[row].abs();
            }
        }
        if sum_art > self.config.feasibility_tol * 10.0 {
            return None;
        }

        let mut new_basis = Basis::new(m, n);
        for i in 0..m {
            let var = art_basis.basic_vars[i];
            if var < n {
                new_basis.basic_vars[i] = var;
                new_basis.is_basic[var] = true;
            } else {
                let mut found = false;
                for j in 0..n {
                    if !new_basis.is_basic[j] {
                        let col = tableau.get_column(j);
                        let pcol = new_basis.compute_pivot_column(&col);
                        if pcol[i].abs() > self.config.pivot_tol {
                            new_basis.update(i, j, &pcol, pcol[i]);
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    new_basis.basic_vars[i] = n - 1;
                }
            }
        }

        for j in n..n + m {
            new_basis.is_basic.truncate(n);
        }

        Some(new_basis)
    }

    fn phase2(&mut self, tableau: &Tableau, basis: &mut Basis) -> SolverStatus {
        let m = tableau.num_constraints;
        let n = tableau.num_vars;
        let mut iterations = 0u64;

        loop {
            if iterations >= self.config.max_iterations {
                self.stats.phase2_iterations = iterations;
                return SolverStatus::IterationLimit;
            }

            let x_b = basis.compute_basic_solution(&tableau.rhs);
            let cb: Vec<f64> = basis.basic_vars.iter().map(|&j| tableau.obj_coeffs[j]).collect();
            let dual = basis.compute_dual(&cb);

            let mut entering = None;
            let mut best_rc = -self.config.optimality_tol;
            for j in 0..n {
                if basis.is_basic[j] {
                    continue;
                }
                let col = tableau.get_column(j);
                let mut rc = tableau.obj_coeffs[j];
                for i in 0..m {
                    rc -= dual[i] * col[i];
                }
                if self.config.use_bland_rule {
                    if rc < -self.config.optimality_tol {
                        entering = Some(j);
                        break;
                    }
                } else if rc < best_rc {
                    best_rc = rc;
                    entering = Some(j);
                }
            }

            if entering.is_none() {
                self.stats.phase2_iterations = iterations;
                return SolverStatus::Optimal;
            }
            let ent = entering.unwrap();

            let col = tableau.get_column(ent);
            let pivot_col = basis.compute_pivot_column(&col);

            let leaving = self.ratio_test(&x_b, &pivot_col, m);
            if leaving.is_none() {
                self.stats.phase2_iterations = iterations;
                return SolverStatus::Unbounded;
            }
            let lv = leaving.unwrap();

            if pivot_col[lv].abs() < self.config.pivot_tol {
                self.stats.numerical_issues += 1;
                self.stats.phase2_iterations = iterations;
                return SolverStatus::NumericalError;
            }

            if x_b[lv].abs() < self.config.feasibility_tol {
                self.stats.degenerate_pivots += 1;
            }

            basis.update(lv, ent, &pivot_col, pivot_col[lv]);
            iterations += 1;
        }
    }

    fn ratio_test(&self, x_b: &[f64], pivot_col: &[f64], m: usize) -> Option<usize> {
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..m {
            if pivot_col[i] <= self.config.pivot_tol {
                continue;
            }
            let ratio = x_b[i] / pivot_col[i];
            if ratio < min_ratio - self.config.feasibility_tol {
                min_ratio = ratio;
                leaving = Some(i);
            } else if (ratio - min_ratio).abs() < self.config.feasibility_tol {
                if self.config.use_bland_rule {
                    if let Some(prev) = leaving {
                        if i < prev {
                            leaving = Some(i);
                        }
                    }
                }
            }
        }
        leaving
    }

    pub fn stats(&self) -> &SimplexStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = SimplexStats::default();
    }

    pub fn config(&self) -> &SimplexConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: SimplexConfig) {
        self.config = config;
    }
}

impl Default for SimplexSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::interface::SolverBackend for SimplexSolver {
    fn name(&self) -> &str { "PrimalSimplex" }

    fn solve(&mut self, model: &SolverModel) -> bilevel_types::BilevelResult<crate::interface::SolveResult> {
        let result = SimplexSolver::solve(self, model);
        Ok(simplex_result_to_solve_result(result, model))
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

fn simplex_result_to_solve_result(r: SimplexResult, model: &SolverModel) -> crate::interface::SolveResult {
    let n = model.num_variables();
    let m = model.num_constraints();
    let solution = if r.status == SolverStatus::Optimal || r.status == SolverStatus::Feasible {
        Some(Solution::new(
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
            phase1_iterations: r.stats.phase1_iterations as usize,
            phase2_iterations: r.stats.phase2_iterations as usize,
            degenerate_pivots: r.stats.degenerate_pivots as usize,
            ..Default::default()
        },
    }
}

/// Steepest-edge pricing strategy.
#[derive(Debug, Clone)]
pub struct SteepestEdgePricer {
    weights: Vec<f64>,
    initialized: bool,
}

impl SteepestEdgePricer {
    pub fn new(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
            initialized: false,
        }
    }

    pub fn initialize(&mut self, n: usize) {
        self.weights = vec![1.0; n];
        self.initialized = true;
    }

    pub fn select_entering(&self, reduced_costs: &[f64], is_basic: &[bool]) -> Option<usize> {
        let mut best_idx = None;
        let mut best_score = 0.0f64;

        for j in 0..reduced_costs.len() {
            if is_basic[j] || j >= self.weights.len() {
                continue;
            }
            let rc = reduced_costs[j];
            if rc >= -1e-8 {
                continue;
            }
            let w = self.weights[j].max(1e-10);
            let score = (rc * rc) / w;
            if score > best_score {
                best_score = score;
                best_idx = Some(j);
            }
        }
        best_idx
    }

    pub fn update_weights(&mut self, pivot_col: &[f64], leaving_row: usize, entering_col: usize, basis_inv_row: &[f64]) {
        let pivot = pivot_col[leaving_row];
        if pivot.abs() < 1e-12 || entering_col >= self.weights.len() {
            return;
        }
        let pivot_sq = pivot * pivot;
        let alpha_sq_sum: f64 = pivot_col.iter().map(|&a| a * a).sum();
        let new_weight = alpha_sq_sum / pivot_sq;
        self.weights[entering_col] = new_weight.max(1e-10);
    }
}

/// Harris ratio test with tolerance.
#[derive(Debug, Clone)]
pub struct HarrisRatioTest {
    tolerance: f64,
}

impl HarrisRatioTest {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn select_leaving(&self, x_b: &[f64], pivot_col: &[f64]) -> Option<(usize, f64)> {
        let m = x_b.len();
        let mut candidates = Vec::new();

        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if pivot_col[i] <= 1e-10 {
                continue;
            }
            let ratio = (x_b[i] + self.tolerance) / pivot_col[i];
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }

        if min_ratio == f64::INFINITY {
            return None;
        }

        for i in 0..m {
            if pivot_col[i] <= 1e-10 {
                continue;
            }
            let ratio = x_b[i] / pivot_col[i];
            if ratio <= min_ratio + self.tolerance {
                candidates.push((i, ratio));
            }
        }

        candidates.into_iter().min_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Perturbation-based anti-cycling mechanism.
#[derive(Debug, Clone)]
pub struct PerturbationAntiCycling {
    perturbations: Vec<f64>,
    active: bool,
}

impl PerturbationAntiCycling {
    pub fn new(m: usize, epsilon: f64) -> Self {
        let perturbations: Vec<f64> = (0..m)
            .map(|i| epsilon * (1.0 + (i as f64) * 0.01))
            .collect();
        Self {
            perturbations,
            active: false,
        }
    }

    pub fn activate(&mut self) {
        self.active = true;
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn perturbed_rhs(&self, rhs: &[f64]) -> Vec<f64> {
        if !self.active {
            return rhs.to_vec();
        }
        rhs.iter()
            .zip(self.perturbations.iter())
            .map(|(&r, &p)| r + p)
            .collect()
    }
}

/// Basis factorization using product form of inverse.
#[derive(Debug, Clone)]
pub struct BasisFactorization {
    eta_vectors: Vec<EtaVector>,
    initial_inverse: Vec<Vec<f64>>,
    num_updates: usize,
    refactor_threshold: usize,
}

#[derive(Debug, Clone)]
struct EtaVector {
    pivot_row: usize,
    column: Vec<f64>,
}

impl BasisFactorization {
    pub fn new(m: usize) -> Self {
        let mut inv = vec![vec![0.0; m]; m];
        for i in 0..m {
            inv[i][i] = 1.0;
        }
        Self {
            eta_vectors: Vec::new(),
            initial_inverse: inv,
            num_updates: 0,
            refactor_threshold: 100,
        }
    }

    pub fn apply_ftran(&self, rhs: &mut [f64]) {
        let m = rhs.len();
        let mut temp = vec![0.0; m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..m {
                s += self.initial_inverse[i][j] * rhs[j];
            }
            temp[i] = s;
        }
        rhs.copy_from_slice(&temp);

        for eta in &self.eta_vectors {
            let pivot_val = rhs[eta.pivot_row];
            for i in 0..m {
                if i == eta.pivot_row {
                    rhs[i] = pivot_val * eta.column[i];
                } else {
                    rhs[i] += pivot_val * eta.column[i];
                }
            }
        }
    }

    pub fn apply_btran(&self, rhs: &mut [f64]) {
        let m = rhs.len();
        for eta in self.eta_vectors.iter().rev() {
            let mut s = 0.0;
            for i in 0..m {
                s += rhs[i] * eta.column[i];
            }
            rhs[eta.pivot_row] = s;
        }

        let mut temp = vec![0.0; m];
        for j in 0..m {
            let mut s = 0.0;
            for i in 0..m {
                s += rhs[i] * self.initial_inverse[i][j];
            }
            temp[j] = s;
        }
        rhs.copy_from_slice(&temp);
    }

    pub fn update(&mut self, pivot_col: &[f64], pivot_row: usize) {
        let m = pivot_col.len();
        let pivot_element = pivot_col[pivot_row];
        if pivot_element.abs() < 1e-15 {
            return;
        }
        let mut eta = vec![0.0; m];
        let inv_pivot = 1.0 / pivot_element;
        for i in 0..m {
            if i == pivot_row {
                eta[i] = inv_pivot;
            } else {
                eta[i] = -pivot_col[i] * inv_pivot;
            }
        }
        self.eta_vectors.push(EtaVector {
            pivot_row,
            column: eta,
        });
        self.num_updates += 1;
    }

    pub fn needs_refactorization(&self) -> bool {
        self.num_updates >= self.refactor_threshold
    }

    pub fn refactorize(&mut self, basis_columns: &[Vec<f64>]) {
        let m = basis_columns.len();
        if m == 0 {
            return;
        }
        let mut inv = vec![vec![0.0; m]; m];
        for i in 0..m {
            inv[i][i] = 1.0;
        }
        for col in 0..m {
            let mut pivot_row = col;
            let mut max_val = basis_columns[col][col].abs();
            for row in (col + 1)..m {
                if basis_columns[col][row].abs() > max_val {
                    max_val = basis_columns[col][row].abs();
                    pivot_row = row;
                }
            }
            if max_val < 1e-12 {
                continue;
            }
            if pivot_row != col {
                inv.swap(col, pivot_row);
            }
            let pivot = basis_columns[col][col];
            if pivot.abs() < 1e-15 {
                continue;
            }
            let inv_pivot = 1.0 / pivot;
            for j in 0..m {
                inv[col][j] *= inv_pivot;
            }
            for i in 0..m {
                if i == col {
                    continue;
                }
                let factor = basis_columns[col][i];
                for j in 0..m {
                    inv[i][j] -= factor * inv[col][j];
                }
            }
        }
        self.initial_inverse = inv;
        self.eta_vectors.clear();
        self.num_updates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SolverModel;

    fn make_simple_lp() -> SolverModel {
        let mut model = SolverModel::new();
        model.add_variable("x1", 0.0, f64::INFINITY, 1.0, false);
        model.add_variable("x2", 0.0, f64::INFINITY, 2.0, false);
        model.add_constraint("c1", vec![0, 1], vec![1.0, 1.0], "<=", 4.0);
        model.add_constraint("c2", vec![0, 1], vec![1.0, 3.0], "<=", 6.0);
        model.set_minimize(true);
        model
    }

    #[test]
    fn test_simplex_simple_lp() {
        let model = make_simple_lp();
        let mut solver = SimplexSolver::new();
        let result = solver.solve(&model);
        assert_eq!(result.status, SolverStatus::Optimal);
    }

    #[test]
    fn test_simplex_config() {
        let cfg = SimplexConfig::default();
        assert!(cfg.max_iterations > 0);
        assert!(cfg.feasibility_tol > 0.0);
    }

    #[test]
    fn test_steepest_edge() {
        let pricer = SteepestEdgePricer::new(5);
        let rc = vec![-1.0, 0.5, -2.0, 0.0, -0.5];
        let is_basic = vec![false; 5];
        let idx = pricer.select_entering(&rc, &is_basic);
        assert!(idx.is_some());
    }

    #[test]
    fn test_harris_ratio() {
        let test = HarrisRatioTest::new(1e-8);
        let x_b = vec![4.0, 6.0, 2.0];
        let pivot_col = vec![1.0, 2.0, 0.5];
        let result = test.select_leaving(&x_b, &pivot_col);
        assert!(result.is_some());
        let (row, ratio) = result.unwrap();
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_perturbation() {
        let mut pac = PerturbationAntiCycling::new(3, 1e-8);
        assert!(!pac.is_active());
        pac.activate();
        assert!(pac.is_active());
        let rhs = vec![1.0, 2.0, 3.0];
        let perturbed = pac.perturbed_rhs(&rhs);
        assert!(perturbed[0] > 1.0);
    }

    #[test]
    fn test_basis_factorization_identity() {
        let fact = BasisFactorization::new(3);
        let mut rhs = vec![1.0, 2.0, 3.0];
        fact.apply_ftran(&mut rhs);
        assert!((rhs[0] - 1.0).abs() < 1e-10);
        assert!((rhs[1] - 2.0).abs() < 1e-10);
        assert!((rhs[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplex_stats_default() {
        let stats = SimplexStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.phase1_iterations, 0);
    }

    #[test]
    fn test_basis_new() {
        let basis = Basis::new(3, 5);
        assert_eq!(basis.basic_vars.len(), 3);
        assert!(basis.is_basic[5]);
        assert!(basis.is_basic[6]);
        assert!(basis.is_basic[7]);
    }
}
