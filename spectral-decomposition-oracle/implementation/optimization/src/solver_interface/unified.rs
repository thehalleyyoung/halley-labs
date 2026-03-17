//! Unified solver implementation.
//!
//! Dispatches to internal simplex or interior point solvers and provides
//! a simple LP solver implementation for self-contained use.

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};
use crate::solver_interface::{SolverConfig, SolverInterface, SolverType};
use log::warn;
use std::time::Instant;

/// Unified solver that dispatches to internal implementations.
pub struct UnifiedSolver {
    config: SolverConfig,
    problem: Option<LpProblem>,
    last_solution: Option<LpSolution>,
    status: SolverStatus,
}

impl UnifiedSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            problem: None,
            last_solution: None,
            status: SolverStatus::NumericalError,
        }
    }

    pub fn from_problem(problem: LpProblem, config: SolverConfig) -> Self {
        Self {
            config,
            problem: Some(problem),
            last_solution: None,
            status: SolverStatus::NumericalError,
        }
    }

    /// Solve using internal simplex method.
    fn solve_with_simplex(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        let n = problem.num_vars;
        let m = problem.num_constraints;

        if n == 0 {
            return Ok(LpSolution {
                status: SolverStatus::Optimal,
                objective_value: 0.0,
                primal_values: Vec::new(),
                dual_values: Vec::new(),
                reduced_costs: Vec::new(),
                basis_status: Vec::new(),
                iterations: 0,
                time_seconds: 0.0,
            });
        }

        // Add slack/surplus/artificial variables for standard form
        let n_slack = m;
        let n_total = n + n_slack;
        let sign = if problem.maximize { -1.0 } else { 1.0 };

        // Extended objective
        let mut c = vec![0.0; n_total];
        for i in 0..n {
            c[i] = sign * problem.obj_coeffs[i];
        }

        // Constraint matrix in dense form
        let mut a = vec![vec![0.0; n_total]; m];
        let mut b = problem.rhs.clone();
        b.resize(m, 0.0);

        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    let col = problem.col_indices[idx];
                    let val = problem.values[idx];
                    if col < n {
                        a[i][col] = val;
                    }
                }
            }

            // Add slack variables
            let slack_idx = n + i;
            match problem.constraint_types.get(i).unwrap_or(&ConstraintType::Le) {
                ConstraintType::Le => {
                    a[i][slack_idx] = 1.0;
                    if b[i] < 0.0 {
                        for j in 0..n_total {
                            a[i][j] = -a[i][j];
                        }
                        b[i] = -b[i];
                    }
                }
                ConstraintType::Ge => {
                    a[i][slack_idx] = -1.0;
                    if b[i] < 0.0 {
                        for j in 0..n_total {
                            a[i][j] = -a[i][j];
                        }
                        b[i] = -b[i];
                        a[i][slack_idx] = 1.0;
                    }
                }
                ConstraintType::Eq => {
                    a[i][slack_idx] = 1.0; // Artificial
                    if b[i] < 0.0 {
                        for j in 0..n_total {
                            a[i][j] = -a[i][j];
                        }
                        b[i] = -b[i];
                    }
                }
            }
        }

        // Initial basis: slack variables
        let mut basis: Vec<usize> = (n..n_total).collect();
        let mut x = vec![0.0; n_total];
        for i in 0..m {
            x[basis[i]] = b[i];
        }

        // Phase II simplex iterations
        let max_iter = 1000 * (n + m + 1);
        let mut iterations = 0;
        let tol = 1e-8;

        for _iter in 0..max_iter {
            if start.elapsed().as_secs_f64() > self.config.time_limit {
                return Ok(LpSolution {
                    status: SolverStatus::TimeLimit,
                    objective_value: compute_obj(&c, &x, n, sign),
                    primal_values: x[..n].to_vec(),
                    dual_values: vec![0.0; m],
                    reduced_costs: vec![0.0; n],
                    basis_status: self.compute_basis_status(n, &basis),
                    iterations,
                    time_seconds: start.elapsed().as_secs_f64(),
                });
            }

            iterations += 1;

            // Compute reduced costs: c_j - c_B * B^{-1} * a_j
            let cb: Vec<f64> = basis.iter().map(|&bi| c[bi]).collect();
            let mut entering = None;
            let mut best_rc = -tol;

            for j in 0..n_total {
                if basis.contains(&j) {
                    continue;
                }
                // Compute c_B * column_j (simplified - use column directly)
                let col_j: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
                let rc = c[j] - dot(&cb, &col_j);

                if rc < best_rc {
                    best_rc = rc;
                    entering = Some(j);
                }
            }

            let ej = match entering {
                Some(j) => j,
                None => break, // Optimal
            };

            // Compute direction: d = B^{-1} * a_j (simplified using current tableau)
            let col_ej: Vec<f64> = (0..m).map(|i| a[i][ej]).collect();

            // Ratio test
            let mut best_ratio = f64::INFINITY;
            let mut leaving = None;

            for i in 0..m {
                if col_ej[i] > tol {
                    let ratio = x[basis[i]] / col_ej[i];
                    if ratio < best_ratio - tol {
                        best_ratio = ratio;
                        leaving = Some(i);
                    }
                }
            }

            let li = match leaving {
                Some(i) => i,
                None => {
                    // Unbounded
                    return Err(OptError::unbounded("Simplex detected unbounded problem"));
                }
            };

            // Pivot
            let pivot_val = a[li][ej];
            if pivot_val.abs() < 1e-14 {
                warn!("Near-zero pivot: {}", pivot_val);
                continue;
            }

            // Scale pivot row
            let inv_pivot = 1.0 / pivot_val;
            for j in 0..n_total {
                a[li][j] *= inv_pivot;
            }
            b[li] *= inv_pivot;

            // Eliminate from other rows
            for i in 0..m {
                if i == li {
                    continue;
                }
                let factor = a[i][ej];
                if factor.abs() > 1e-14 {
                    for j in 0..n_total {
                        a[i][j] -= factor * a[li][j];
                    }
                    b[i] -= factor * b[li];
                }
            }

            // Update basis and solution
            x[basis[li]] = 0.0;
            basis[li] = ej;
            for i in 0..m {
                x[basis[i]] = b[i].max(0.0);
            }
        }

        // Enforce original variable bounds
        for i in 0..n {
            x[i] = x[i]
                .max(problem.lower_bounds[i])
                .min(problem.upper_bounds[i]);
        }

        let obj_val = compute_obj(&c, &x, n, sign);

        // Check feasibility
        let mut feasible = true;
        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };
            let mut lhs = 0.0;
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    let col = problem.col_indices[idx];
                    if col < n {
                        lhs += problem.values[idx] * x[col];
                    }
                }
            }
            let viol = match problem.constraint_types.get(i).unwrap_or(&ConstraintType::Le) {
                ConstraintType::Le => (lhs - problem.rhs[i]).max(0.0),
                ConstraintType::Ge => (problem.rhs[i] - lhs).max(0.0),
                ConstraintType::Eq => (lhs - problem.rhs[i]).abs(),
            };
            if viol > 1e-4 {
                feasible = false;
                break;
            }
        }

        let status = if feasible {
            SolverStatus::Optimal
        } else {
            SolverStatus::Infeasible
        };

        // Compute dual values (from final basis)
        let dual_values = self.compute_duals(m, n, &basis, &c, &a);
        let reduced_costs = self.compute_reduced_costs(n, &c, &dual_values, problem);

        Ok(LpSolution {
            status,
            objective_value: obj_val,
            primal_values: x[..n].to_vec(),
            dual_values,
            reduced_costs,
            basis_status: self.compute_basis_status(n, &basis),
            iterations,
            time_seconds: start.elapsed().as_secs_f64(),
        })
    }

    /// Solve using a basic interior point method.
    fn solve_with_interior_point(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        let n = problem.num_vars;
        let m = problem.num_constraints;

        if n == 0 {
            return Ok(LpSolution {
                status: SolverStatus::Optimal,
                objective_value: 0.0,
                primal_values: Vec::new(),
                dual_values: Vec::new(),
                reduced_costs: Vec::new(),
                basis_status: Vec::new(),
                iterations: 0,
                time_seconds: 0.0,
            });
        }

        let sign = if problem.maximize { -1.0 } else { 1.0 };
        let n_slack = m;
        let n_total = n + n_slack;

        // Build dense constraint matrix
        let mut a_mat = vec![vec![0.0; n_total]; m];
        let mut b_vec = problem.rhs.clone();
        b_vec.resize(m, 0.0);

        for i in 0..m {
            let row_start = problem.row_starts[i];
            let row_end = if i + 1 < problem.row_starts.len() {
                problem.row_starts[i + 1]
            } else {
                problem.col_indices.len()
            };
            for idx in row_start..row_end {
                if idx < problem.col_indices.len() {
                    let col = problem.col_indices[idx];
                    if col < n {
                        a_mat[i][col] = problem.values[idx];
                    }
                }
            }
            a_mat[i][n + i] = 1.0; // slack
        }

        let mut c_vec = vec![0.0; n_total];
        for i in 0..n {
            c_vec[i] = sign * problem.obj_coeffs[i];
        }

        // Initialize: x > 0, s > 0
        let mut x = vec![1.0; n_total];
        let mut s = vec![1.0; n_total];
        let mut y = vec![0.0; m];

        let max_iter = 100;
        let tol = 1e-8;
        let mut iterations = 0;

        for iter in 0..max_iter {
            iterations = iter + 1;

            if start.elapsed().as_secs_f64() > self.config.time_limit {
                break;
            }

            // Compute residuals
            // rp = b - Ax
            let mut rp = b_vec.clone();
            for i in 0..m {
                for j in 0..n_total {
                    rp[i] -= a_mat[i][j] * x[j];
                }
            }

            // rd = c - A^T y - s
            let mut rd = c_vec.clone();
            for j in 0..n_total {
                for i in 0..m {
                    rd[j] -= a_mat[i][j] * y[i];
                }
                rd[j] -= s[j];
            }

            // mu = x^T s / n
            let mu: f64 = x.iter().zip(s.iter()).map(|(xi, si)| xi * si).sum::<f64>()
                / n_total as f64;

            // Check convergence
            let rp_norm: f64 = rp.iter().map(|r| r * r).sum::<f64>().sqrt();
            let rd_norm: f64 = rd.iter().map(|r| r * r).sum::<f64>().sqrt();

            if rp_norm < tol && rd_norm < tol && mu < tol {
                break;
            }

            // Centering parameter
            let sigma = 0.3;

            // Solve Newton system (simplified direct approach)
            // Compute D^2 = X/S
            let d2: Vec<f64> = (0..n_total)
                .map(|j| x[j] / s[j].max(1e-14))
                .collect();

            // Normal equations: A D^2 A^T dy = rp + A D^2 (rd + sigma*mu*X^{-1}*e - S*e)
            let mut rhs_mod = vec![0.0; n_total];
            for j in 0..n_total {
                rhs_mod[j] = rd[j] + sigma * mu / x[j].max(1e-14);
            }

            // Build ADA^T
            let mut ada = vec![vec![0.0; m]; m];
            for i in 0..m {
                for k in 0..m {
                    for j in 0..n_total {
                        ada[i][k] += a_mat[i][j] * d2[j] * a_mat[k][j];
                    }
                }
                ada[i][i] += 1e-12; // regularization
            }

            // RHS for normal equations
            let mut ne_rhs = rp.clone();
            for i in 0..m {
                for j in 0..n_total {
                    ne_rhs[i] += a_mat[i][j] * d2[j] * rhs_mod[j];
                }
            }

            // Solve ADA^T * dy = ne_rhs via Cholesky
            let dy = solve_symmetric_system(&ada, &ne_rhs);

            // Back-substitute for dx, ds
            let mut dx = vec![0.0; n_total];
            let mut ds = vec![0.0; n_total];
            for j in 0..n_total {
                let mut atdy = 0.0;
                for i in 0..m {
                    atdy += a_mat[i][j] * dy[i];
                }
                dx[j] = d2[j] * (atdy - rhs_mod[j]);
                ds[j] = -s[j] - (s[j] / x[j].max(1e-14)) * dx[j] + sigma * mu / x[j].max(1e-14);
            }

            // Step length (fraction to boundary)
            let mut alpha_p: f64 = 1.0;
            let mut alpha_d: f64 = 1.0;
            let tau = 0.995;

            for j in 0..n_total {
                if dx[j] < -1e-14 {
                    alpha_p = alpha_p.min(-tau * x[j] / dx[j]);
                }
                if ds[j] < -1e-14 {
                    alpha_d = alpha_d.min(-tau * s[j] / ds[j]);
                }
            }

            alpha_p = alpha_p.min(1.0).max(0.0);
            alpha_d = alpha_d.min(1.0).max(0.0);

            // Update
            for j in 0..n_total {
                x[j] += alpha_p * dx[j];
                s[j] += alpha_d * ds[j];
                x[j] = x[j].max(1e-14);
                s[j] = s[j].max(1e-14);
            }
            for i in 0..m {
                y[i] += alpha_d * dy[i];
            }
        }

        // Enforce bounds
        for i in 0..n {
            x[i] = x[i]
                .max(problem.lower_bounds[i])
                .min(problem.upper_bounds[i]);
        }

        let obj_val = (0..n)
            .map(|i| problem.obj_coeffs[i] * x[i])
            .sum::<f64>();

        Ok(LpSolution {
            status: SolverStatus::Optimal,
            objective_value: obj_val,
            primal_values: x[..n].to_vec(),
            dual_values: y,
            reduced_costs: (0..n)
                .map(|i| {
                    let mut rc = sign * problem.obj_coeffs[i];
                    rc -= s[i];
                    sign * rc
                })
                .collect(),
            basis_status: (0..n)
                .map(|i| {
                    if (x[i] - problem.lower_bounds[i]).abs() < 1e-6 {
                        BasisStatus::AtLower
                    } else if (x[i] - problem.upper_bounds[i]).abs() < 1e-6 {
                        BasisStatus::AtUpper
                    } else {
                        BasisStatus::Basic
                    }
                })
                .collect(),
            iterations,
            time_seconds: start.elapsed().as_secs_f64(),
        })
    }

    fn compute_duals(
        &self,
        m: usize,
        _n: usize,
        basis: &[usize],
        c: &[f64],
        _a: &[Vec<f64>],
    ) -> Vec<f64> {
        let mut duals = vec![0.0; m];
        for i in 0..m.min(basis.len()) {
            if basis[i] < c.len() {
                duals[i] = c[basis[i]];
            }
        }
        duals
    }

    fn compute_reduced_costs(
        &self,
        n: usize,
        _c: &[f64],
        duals: &[f64],
        problem: &LpProblem,
    ) -> Vec<f64> {
        let sign = if problem.maximize { -1.0 } else { 1.0 };
        (0..n)
            .map(|j| {
                let mut rc = sign * problem.obj_coeffs[j];
                for (i, &d) in duals.iter().enumerate() {
                    let row_start = problem.row_starts[i];
                    let row_end = if i + 1 < problem.row_starts.len() {
                        problem.row_starts[i + 1]
                    } else {
                        problem.col_indices.len()
                    };
                    for idx in row_start..row_end {
                        if idx < problem.col_indices.len() && problem.col_indices[idx] == j
                        {
                            rc -= d * problem.values[idx];
                        }
                    }
                }
                sign * rc
            })
            .collect()
    }

    fn compute_basis_status(&self, n: usize, basis: &[usize]) -> Vec<BasisStatus> {
        (0..n)
            .map(|j| {
                if basis.contains(&j) {
                    BasisStatus::Basic
                } else {
                    BasisStatus::AtLower
                }
            })
            .collect()
    }
}

impl SolverInterface for UnifiedSolver {
    fn solve_lp(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        self.problem = Some(problem.clone());

        let result = match self.config.solver_type {
            SolverType::InternalSimplex | SolverType::ScipEmulation | SolverType::GcgEmulation
            | SolverType::HiGHS | SolverType::Scip => {
                self.solve_with_simplex(problem)
            }
            SolverType::InternalInteriorPoint => self.solve_with_interior_point(problem),
        };

        match &result {
            Ok(sol) => {
                self.status = sol.status;
                self.last_solution = Some(sol.clone());
            }
            Err(_) => {
                self.status = SolverStatus::NumericalError;
            }
        }

        result
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn get_dual_values(&self) -> OptResult<Vec<f64>> {
        self.last_solution
            .as_ref()
            .map(|s| s.dual_values.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn get_basis(&self) -> OptResult<Vec<BasisStatus>> {
        self.last_solution
            .as_ref()
            .map(|s| s.basis_status.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn add_constraint(
        &mut self,
        coeffs: &[(usize, f64)],
        ctype: ConstraintType,
        rhs: f64,
    ) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_constraints;
        let (indices, vals): (Vec<usize>, Vec<f64>) = coeffs.iter().copied().unzip();
        problem.add_constraint(&indices, &vals, ctype, rhs)?;
        Ok(idx)
    }

    fn add_variable(&mut self, obj: f64, lb: f64, ub: f64) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_vars;
        problem.add_variable(obj, lb, ub, None);
        Ok(idx)
    }

    fn set_objective(&mut self, coeffs: &[(usize, f64)]) -> OptResult<()> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        for &(i, val) in coeffs {
            if i < problem.obj_coeffs.len() {
                problem.obj_coeffs[i] = val;
            }
        }
        Ok(())
    }

    fn set_time_limit(&mut self, seconds: f64) {
        self.config.time_limit = seconds;
    }

    fn name(&self) -> &str {
        "UnifiedSolver"
    }

    fn reset(&mut self) {
        self.problem = None;
        self.last_solution = None;
        self.status = SolverStatus::NumericalError;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn compute_obj(c: &[f64], x: &[f64], n: usize, sign: f64) -> f64 {
    sign * (0..n).map(|i| c[i] * x[i]).sum::<f64>()
}

/// Solve a symmetric positive definite system via Cholesky.
fn solve_symmetric_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }

    // Cholesky: A = L * L^T
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = if sum > 0.0 { sum.sqrt() } else { 1e-10 };
            } else {
                l[i][j] = sum / l[j][j].max(1e-14);
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i].max(1e-14);
    }

    // Backward substitution: L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i].max(1e-14);
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lp() -> LpProblem {
        // min x + y s.t. x + y <= 10, x >= 0, y >= 0, x <= 7, y <= 7
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 1.0];
        lp.lower_bounds = vec![0.0, 0.0];
        lp.upper_bounds = vec![7.0, 7.0];
        lp.row_starts = vec![0, 2];
        lp.col_indices = vec![0, 1];
        lp.values = vec![1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le];
        lp.rhs = vec![10.0];
        lp.num_constraints = 1;
        lp
    }

    #[test]
    fn test_unified_solver_creation() {
        let config = SolverConfig::default();
        let solver = UnifiedSolver::new(config);
        assert_eq!(solver.name(), "UnifiedSolver");
    }

    #[test]
    fn test_unified_solve_simple() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        let result = solver.solve_lp(&lp);
        assert!(result.is_ok());
        let sol = result.unwrap();
        assert!(sol.objective_value >= -0.01);
    }

    #[test]
    fn test_unified_solve_empty() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = LpProblem::new(false);
        let sol = solver.solve_lp(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert_eq!(sol.objective_value, 0.0);
    }

    #[test]
    fn test_unified_interior_point() {
        let config = SolverConfig::default().with_type(SolverType::InternalInteriorPoint);
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        let result = solver.solve_lp(&lp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unified_get_status() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        solver.solve_lp(&lp).unwrap();
        let status = solver.get_status();
        assert!(matches!(status, SolverStatus::Optimal | SolverStatus::Infeasible));
    }

    #[test]
    fn test_unified_get_duals() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        solver.solve_lp(&lp).unwrap();
        let duals = solver.get_dual_values();
        assert!(duals.is_ok());
    }

    #[test]
    fn test_unified_get_basis() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        solver.solve_lp(&lp).unwrap();
        let basis = solver.get_basis();
        assert!(basis.is_ok());
    }

    #[test]
    fn test_unified_no_solution() {
        let config = SolverConfig::default();
        let solver = UnifiedSolver::new(config);
        assert!(solver.get_dual_values().is_err());
        assert!(solver.get_basis().is_err());
    }

    #[test]
    fn test_unified_reset() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        let lp = make_test_lp();
        solver.solve_lp(&lp).unwrap();
        solver.reset();
        assert!(solver.get_dual_values().is_err());
    }

    #[test]
    fn test_unified_add_constraint() {
        let config = SolverConfig::default();
        let lp = make_test_lp();
        let mut solver = UnifiedSolver::from_problem(lp, config);
        let idx = solver.add_constraint(&[(0, 1.0)], ConstraintType::Le, 5.0);
        assert!(idx.is_ok());
    }

    #[test]
    fn test_unified_set_time_limit() {
        let config = SolverConfig::default();
        let mut solver = UnifiedSolver::new(config);
        solver.set_time_limit(30.0);
        assert!((solver.config.time_limit - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_symmetric_system() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![8.0, 7.0];
        let x = solve_symmetric_system(&a, &b);
        // 4x + 2y = 8, 2x + 3y = 7 => x = 1.25, y = 1.5
        assert!((x[0] - 1.25).abs() < 1e-4);
        assert!((x[1] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_from_problem() {
        let config = SolverConfig::default();
        let lp = make_test_lp();
        let mut solver = UnifiedSolver::from_problem(lp, config);
        let result = solver.solve_lp(&solver.problem.clone().unwrap());
        assert!(result.is_ok());
    }
}
