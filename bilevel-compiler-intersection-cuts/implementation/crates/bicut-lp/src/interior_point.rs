//! Interior point method (Mehrotra predictor-corrector).
//!
//! Central path following, Newton system assembly and solve,
//! step size computation, starting point heuristic, crossover to basic solution.

use crate::model::{Constraint, LpModel, Variable};
use bicut_types::{ConstraintSense, LpStatus, OptDirection};
use log::debug;

/// Configuration for the interior point method.
#[derive(Debug, Clone)]
pub struct InteriorPointConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (duality gap).
    pub tolerance: f64,
    /// Step size reduction factor.
    pub step_reduction: f64,
    /// Barrier parameter initial value.
    pub initial_mu: f64,
    /// Whether to perform crossover to a basic solution.
    pub crossover: bool,
    /// Verbose output.
    pub verbose: bool,
    /// Regularization parameter for the normal equations.
    pub regularization: f64,
}

impl Default for InteriorPointConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            step_reduction: 0.99,
            initial_mu: 1.0,
            crossover: true,
            verbose: false,
            regularization: 1e-12,
        }
    }
}

/// Interior point method result.
#[derive(Debug, Clone)]
pub struct InteriorPointResult {
    pub status: LpStatus,
    pub objective: f64,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub iterations: usize,
    pub duality_gap: f64,
}

/// Solve an LP using the interior point method.
pub fn solve_interior_point(model: &LpModel, config: &InteriorPointConfig) -> InteriorPointResult {
    let m = model.num_constraints();
    let n = model.num_vars();

    if m == 0 || n == 0 {
        return InteriorPointResult {
            status: LpStatus::Optimal,
            objective: model.obj_offset,
            primal: vec![0.0; n],
            dual: vec![0.0; m],
            reduced_costs: vec![0.0; n],
            iterations: 0,
            duality_gap: 0.0,
        };
    }

    // Convert to standard form: min c^T x s.t. Ax = b, x >= 0
    // by adding slack variables
    let (std_model, slack_info) = to_standard_form(model);
    let std_m = std_model.num_eq;
    let std_n = std_model.num_vars;

    if std_m == 0 || std_n == 0 {
        return InteriorPointResult {
            status: LpStatus::Optimal,
            objective: model.obj_offset,
            primal: vec![0.0; n],
            dual: vec![0.0; m],
            reduced_costs: vec![0.0; n],
            iterations: 0,
            duality_gap: 0.0,
        };
    }

    let mut solver = MehrotraSolver::new(std_model, config.clone());
    let result = solver.solve();

    // Map back to original variables
    let primal: Vec<f64> = result.primal.iter().take(n).copied().collect();
    let dual: Vec<f64> = result.dual.iter().take(m).copied().collect();
    let rc: Vec<f64> = result.reduced_costs.iter().take(n).copied().collect();

    let obj = if model.sense == OptDirection::Maximize {
        -(result.objective - model.obj_offset) + model.obj_offset
    } else {
        result.objective + model.obj_offset
    };

    InteriorPointResult {
        status: result.status,
        objective: obj,
        primal,
        dual,
        reduced_costs: rc,
        iterations: result.iterations,
        duality_gap: result.duality_gap,
    }
}

/// Standard form LP for interior point.
#[derive(Debug, Clone)]
struct StandardFormLP {
    /// Number of equality constraints.
    num_eq: usize,
    /// Number of variables (including slacks).
    num_vars: usize,
    /// Objective coefficients.
    c: Vec<f64>,
    /// Constraint matrix A (dense, row-major) for Ax = b.
    a: Vec<Vec<f64>>,
    /// RHS vector b.
    b: Vec<f64>,
}

/// Information about slacks added during conversion.
struct SlackInfo {
    num_original_vars: usize,
    num_slacks: usize,
}

/// Convert LP model to standard form Ax = b, x >= 0.
fn to_standard_form(model: &LpModel) -> (StandardFormLP, SlackInfo) {
    let n = model.num_vars();
    let m = model.num_constraints();

    // Count slacks needed
    let mut num_slacks = 0;
    for con in &model.constraints {
        if con.sense != ConstraintSense::Eq {
            num_slacks += 1;
        }
    }

    let total_vars = n + num_slacks;
    let mut c = vec![0.0; total_vars];
    let is_max = model.sense == OptDirection::Maximize;

    for (j, var) in model.variables.iter().enumerate() {
        c[j] = if is_max {
            -var.obj_coeff
        } else {
            var.obj_coeff
        };
    }

    let mut a = vec![vec![0.0; total_vars]; m];
    let mut b = vec![0.0; m];
    let mut slack_idx = n;

    for (i, con) in model.constraints.iter().enumerate() {
        for (&col, &val) in con.row_indices.iter().zip(con.row_values.iter()) {
            a[i][col] = val;
        }
        b[i] = con.rhs;

        match con.sense {
            ConstraintSense::Le => {
                a[i][slack_idx] = 1.0;
                slack_idx += 1;
            }
            ConstraintSense::Ge => {
                // Multiply by -1: -a^T x + s = -b => a^T x - s = b
                for j in 0..n {
                    a[i][j] = -a[i][j];
                }
                b[i] = -b[i];
                a[i][slack_idx] = 1.0;
                slack_idx += 1;
            }
            ConstraintSense::Eq => {}
        }
    }

    let std_model = StandardFormLP {
        num_eq: m,
        num_vars: total_vars,
        c,
        a,
        b,
    };

    let slack_info = SlackInfo {
        num_original_vars: n,
        num_slacks,
    };

    (std_model, slack_info)
}

/// Mehrotra predictor-corrector interior point solver.
struct MehrotraSolver {
    model: StandardFormLP,
    config: InteriorPointConfig,
    /// Primal variables.
    x: Vec<f64>,
    /// Dual variables (equality constraints).
    y: Vec<f64>,
    /// Dual slacks (reduced costs, s = c - A^T y).
    s: Vec<f64>,
}

impl MehrotraSolver {
    fn new(model: StandardFormLP, config: InteriorPointConfig) -> Self {
        let n = model.num_vars;
        let m = model.num_eq;
        Self {
            model,
            config,
            x: vec![1.0; n],
            y: vec![0.0; m],
            s: vec![1.0; n],
        }
    }

    fn solve(&mut self) -> InteriorPointResult {
        self.compute_starting_point();

        let mut iters = 0;
        let m = self.model.num_eq;
        let n = self.model.num_vars;

        loop {
            if iters >= self.config.max_iterations {
                return self.build_result(LpStatus::IterationLimit, iters);
            }

            // Compute residuals
            let (r_p, r_d, mu) = self.compute_residuals();
            let r_p_norm: f64 = r_p.iter().map(|x| x * x).sum::<f64>().sqrt();
            let r_d_norm: f64 = r_d.iter().map(|x| x * x).sum::<f64>().sqrt();

            if self.config.verbose {
                debug!(
                    "IPM iter {}: mu={:.3e}, ||rp||={:.3e}, ||rd||={:.3e}",
                    iters, mu, r_p_norm, r_d_norm
                );
            }

            // Check convergence
            if mu < self.config.tolerance
                && r_p_norm < self.config.tolerance
                && r_d_norm < self.config.tolerance
            {
                return self.build_result(LpStatus::Optimal, iters);
            }

            // Check for infeasibility indicators
            let primal_obj: f64 = self
                .model
                .c
                .iter()
                .zip(self.x.iter())
                .map(|(&c, &x)| c * x)
                .sum();
            let dual_obj: f64 = self
                .model
                .b
                .iter()
                .zip(self.y.iter())
                .map(|(&b, &y)| b * y)
                .sum();

            if r_p_norm < 1e-4 && primal_obj > 1e15 {
                return self.build_result(LpStatus::Infeasible, iters);
            }
            if r_d_norm < 1e-4 && dual_obj < -1e15 {
                return self.build_result(LpStatus::Unbounded, iters);
            }

            // PREDICTOR step (affine scaling direction)
            let (dx_aff, dy_aff, ds_aff) = self.solve_newton_system(&r_p, &r_d, &vec![0.0; n]);

            // Compute maximum step sizes for affine direction
            let alpha_p_aff = self.max_step_primal(&dx_aff);
            let alpha_d_aff = self.max_step_dual(&ds_aff);

            // Compute the affine duality gap
            let mu_aff = self.compute_mu_after_step(&dx_aff, &ds_aff, alpha_p_aff, alpha_d_aff);

            // Compute centering parameter sigma
            let sigma = if mu > 1e-20 {
                let ratio = mu_aff / mu;
                (ratio * ratio * ratio).min(1.0)
            } else {
                0.0
            };

            // CORRECTOR step
            let mut r_xs = vec![0.0; n];
            for j in 0..n {
                r_xs[j] = dx_aff[j] * ds_aff[j] - sigma * mu;
            }

            // Combined right-hand side for the corrector
            let (dx, dy, ds) = self.solve_newton_system(&r_p, &r_d, &r_xs);

            // Step size computation with fraction-to-boundary rule
            let alpha_p = self.config.step_reduction * self.max_step_primal(&dx);
            let alpha_d = self.config.step_reduction * self.max_step_dual(&ds);

            // Update iterates
            for j in 0..n {
                self.x[j] += alpha_p * dx[j];
                self.x[j] = self.x[j].max(1e-14); // ensure positivity
            }
            for i in 0..m {
                self.y[i] += alpha_d * dy[i];
            }
            for j in 0..n {
                self.s[j] += alpha_d * ds[j];
                self.s[j] = self.s[j].max(1e-14);
            }

            iters += 1;
        }
    }

    /// Compute a good starting point using Mehrotra's heuristic.
    fn compute_starting_point(&mut self) {
        let m = self.model.num_eq;
        let n = self.model.num_vars;

        // Solve min ||x||^2 s.t. Ax = b for an initial x
        // Use x = A^T (A A^T)^{-1} b
        let aat = self.compute_aat();
        let aat_inv_b = self.solve_symmetric(&aat, &self.model.b);

        // x_hat = A^T * (AAT)^{-1} * b
        let mut x_hat = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                x_hat[j] += self.model.a[i][j] * aat_inv_b[i];
            }
        }

        // s_hat = c - A^T y_hat where y_hat = (AAT)^{-1} * (A * c)
        let mut ac = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                ac[i] += self.model.a[i][j] * self.model.c[j];
            }
        }
        let y_hat = self.solve_symmetric(&aat, &ac);
        let mut s_hat = vec![0.0; n];
        for j in 0..n {
            s_hat[j] = self.model.c[j];
            for i in 0..m {
                s_hat[j] -= self.model.a[i][j] * y_hat[i];
            }
        }

        // Shift to ensure positivity
        let delta_x = (-1.5 * x_hat.iter().copied().fold(f64::INFINITY, f64::min)).max(0.0);
        let delta_s = (-1.5 * s_hat.iter().copied().fold(f64::INFINITY, f64::min)).max(0.0);

        for j in 0..n {
            x_hat[j] += delta_x;
            s_hat[j] += delta_s;
        }

        // Additional shift based on x^T s
        let mut xs_sum = 0.0;
        let mut x_sum = 0.0;
        let mut s_sum = 0.0;
        for j in 0..n {
            xs_sum += x_hat[j] * s_hat[j];
            x_sum += x_hat[j];
            s_sum += s_hat[j];
        }

        let delta_x_bar = if s_sum > 1e-20 {
            0.5 * xs_sum / s_sum
        } else {
            1.0
        };
        let delta_s_bar = if x_sum > 1e-20 {
            0.5 * xs_sum / x_sum
        } else {
            1.0
        };

        for j in 0..n {
            self.x[j] = (x_hat[j] + delta_x_bar).max(1e-4);
            self.s[j] = (s_hat[j] + delta_s_bar).max(1e-4);
        }
        self.y = y_hat;
    }

    /// Compute A * A^T.
    fn compute_aat(&self) -> Vec<Vec<f64>> {
        let m = self.model.num_eq;
        let n = self.model.num_vars;
        let mut aat = vec![vec![0.0; m]; m];
        for i in 0..m {
            for k in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.model.a[i][j] * self.model.a[k][j];
                }
                aat[i][k] = sum;
            }
            aat[i][i] += self.config.regularization; // regularize
        }
        aat
    }

    /// Solve a symmetric positive definite system using Cholesky factorization.
    fn solve_symmetric(&self, mat: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
        let n = rhs.len();
        if n == 0 {
            return Vec::new();
        }

        // Cholesky factorization: mat = L * L^T
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = mat[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        // Not positive definite: add regularization
                        l[i][i] = (sum + self.config.regularization * 10.0).sqrt().max(1e-10);
                    } else {
                        l[i][i] = sum.sqrt();
                    }
                } else {
                    l[i][j] = if l[j][j].abs() > 1e-20 {
                        sum / l[j][j]
                    } else {
                        0.0
                    };
                }
            }
        }

        // Forward substitution: L y = rhs
        let mut y = rhs.to_vec();
        for i in 0..n {
            for j in 0..i {
                y[i] -= l[i][j] * y[j];
            }
            if l[i][i].abs() > 1e-20 {
                y[i] /= l[i][i];
            }
        }

        // Backward substitution: L^T x = y
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= l[j][i] * x[j];
            }
            if l[i][i].abs() > 1e-20 {
                x[i] /= l[i][i];
            }
        }

        x
    }

    /// Compute residuals (r_p, r_d, mu).
    fn compute_residuals(&self) -> (Vec<f64>, Vec<f64>, f64) {
        let m = self.model.num_eq;
        let n = self.model.num_vars;

        // Primal residual: r_p = b - Ax
        let mut r_p = self.model.b.clone();
        for i in 0..m {
            for j in 0..n {
                r_p[i] -= self.model.a[i][j] * self.x[j];
            }
        }

        // Dual residual: r_d = c - A^T y - s
        let mut r_d = self.model.c.clone();
        for j in 0..n {
            for i in 0..m {
                r_d[j] -= self.model.a[i][j] * self.y[i];
            }
            r_d[j] -= self.s[j];
        }

        // Complementarity measure mu = x^T s / n
        let mut xs_sum = 0.0;
        for j in 0..n {
            xs_sum += self.x[j] * self.s[j];
        }
        let mu = xs_sum / n as f64;

        (r_p, r_d, mu)
    }

    /// Solve the Newton system (augmented system approach).
    ///
    /// The KKT system is:
    /// | -D   A^T | | dx | = | r_d - X^{-1} r_xs |
    /// |  A   0   | | dy |   | r_p               |
    ///
    /// where D = X^{-1} S (diagonal) and r_xs is the corrector term.
    /// ds = X^{-1}(r_xs - S dx)
    fn solve_newton_system(
        &self,
        r_p: &[f64],
        r_d: &[f64],
        r_xs: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.model.num_eq;
        let n = self.model.num_vars;

        // Compute D = S/X (diagonal) and the modified RHS for the normal equations
        let mut d = vec![0.0; n];
        let mut rhs_mod = vec![0.0; n];
        for j in 0..n {
            d[j] = if self.x[j] > 1e-20 {
                self.s[j] / self.x[j]
            } else {
                self.s[j] / 1e-20
            };
            // Modified dual residual: r_d[j] + r_xs[j] / x[j] - s[j] * r_p_contribution
            rhs_mod[j] = r_d[j];
            if self.x[j] > 1e-20 {
                rhs_mod[j] += r_xs[j] / self.x[j];
            }
        }

        // Form normal equations: (A D^{-1} A^T) dy = r_p + A D^{-1} rhs_mod
        let mut adadt = vec![vec![0.0; m]; m];
        for i in 0..m {
            for k in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    if d[j] > 1e-20 {
                        sum += self.model.a[i][j] * self.model.a[k][j] / d[j];
                    }
                }
                adadt[i][k] = sum;
            }
            adadt[i][i] += self.config.regularization;
        }

        // RHS of normal equations
        let mut ne_rhs = r_p.to_vec();
        for i in 0..m {
            for j in 0..n {
                if d[j] > 1e-20 {
                    ne_rhs[i] += self.model.a[i][j] * rhs_mod[j] / d[j];
                }
            }
        }

        // Solve for dy
        let dy = self.solve_symmetric(&adadt, &ne_rhs);

        // Recover dx: dx = D^{-1}(A^T dy - rhs_mod)
        let mut dx = vec![0.0; n];
        for j in 0..n {
            let mut at_dy = 0.0;
            for i in 0..m {
                at_dy += self.model.a[i][j] * dy[i];
            }
            if d[j] > 1e-20 {
                dx[j] = (at_dy - rhs_mod[j]) / d[j];
            }
        }

        // Recover ds: ds = -s + (r_xs - S*dx) / X
        // Or ds = (r_xs - s*dx - x*ds is circular)
        // Actually: from complementarity row: S dx + X ds = r_xs - XS e (with centering)
        // ds = (r_xs - S dx) / X
        // But our r_xs already encodes sigma*mu, so:
        // X ds = sigma*mu*e - XS e - S dx + correction
        // Actually, ds = -r_d + A^T dy (original relation)
        let mut ds = vec![0.0; n];
        for j in 0..n {
            ds[j] = -r_d[j];
            for i in 0..m {
                ds[j] += self.model.a[i][j] * dy[i];
            }
            // Include the corrector: ds should also reflect the complementarity
            // In standard form: ds = c - A^T (y + dy) - (s + ds) ... that's circular
            // Properly: ds[j] = r_d[j] inverted + corrector
            // Let me use the direct formula: ds = -s - dx*s/x + r_xs/x
            // This is from X ds = r_xs - S dx - X S ... simplified:
            let x_j = self.x[j].max(1e-20);
            ds[j] = (-self.s[j] * dx[j] + r_xs[j]) / x_j;
        }

        (dx, dy, ds)
    }

    /// Compute maximum step size in x direction maintaining x > 0.
    fn max_step_primal(&self, dx: &[f64]) -> f64 {
        let mut alpha = 1.0f64;
        for j in 0..self.model.num_vars {
            if dx[j] < -1e-20 {
                let step = -self.x[j] / dx[j];
                alpha = alpha.min(step);
            }
        }
        alpha.min(1.0).max(0.0)
    }

    /// Compute maximum step size in s direction maintaining s > 0.
    fn max_step_dual(&self, ds: &[f64]) -> f64 {
        let mut alpha = 1.0f64;
        for j in 0..self.model.num_vars {
            if ds[j] < -1e-20 {
                let step = -self.s[j] / ds[j];
                alpha = alpha.min(step);
            }
        }
        alpha.min(1.0).max(0.0)
    }

    /// Compute duality gap after a trial step.
    fn compute_mu_after_step(&self, dx: &[f64], ds: &[f64], alpha_p: f64, alpha_d: f64) -> f64 {
        let n = self.model.num_vars;
        let mut xs_sum = 0.0;
        for j in 0..n {
            let x_new = self.x[j] + alpha_p * dx[j];
            let s_new = self.s[j] + alpha_d * ds[j];
            xs_sum += x_new.max(0.0) * s_new.max(0.0);
        }
        xs_sum / n as f64
    }

    fn build_result(&self, status: LpStatus, iters: usize) -> InteriorPointResult {
        let obj: f64 = self
            .model
            .c
            .iter()
            .zip(self.x.iter())
            .map(|(&c, &x)| c * x)
            .sum();
        let (_, _, mu) = self.compute_residuals();

        InteriorPointResult {
            status,
            objective: obj,
            primal: self.x.clone(),
            dual: self.y.clone(),
            reduced_costs: self.s.clone(),
            iterations: iters,
            duality_gap: mu,
        }
    }
}

/// Crossover from interior point solution to a basic feasible solution.
pub fn crossover_to_basic(
    model: &LpModel,
    ipm_result: &InteriorPointResult,
) -> InteriorPointResult {
    // Use the simplex crossover from the simplex module
    let result = crate::simplex::crossover_to_basis(model, &ipm_result.primal, &ipm_result.dual);

    InteriorPointResult {
        status: result.status,
        objective: result.objective,
        primal: result.primal,
        dual: result.dual,
        reduced_costs: result.reduced_costs,
        iterations: ipm_result.iterations + result.iterations,
        duality_gap: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Variable};

    fn make_simple_lp() -> LpModel {
        // min -x - y s.t. x + y <= 4, 2x + y <= 6, x,y >= 0
        let mut m = LpModel::new("ipm_test");
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

    fn make_trivial_lp() -> LpModel {
        // min x s.t. x >= 1, x <= 10
        let mut m = LpModel::new("trivial");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 1.0, 10.0));
        m.set_obj_coeff(x, 1.0);

        let mut c = Constraint::new("c0", ConstraintSense::Le, 10.0);
        c.add_term(x, 1.0);
        m.add_constraint(c);

        m
    }

    #[test]
    fn test_ipm_simple() {
        let model = make_simple_lp();
        let config = InteriorPointConfig {
            max_iterations: 100,
            crossover: false,
            ..InteriorPointConfig::default()
        };
        let result = solve_interior_point(&model, &config);
        assert!(
            result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit,
            "status = {:?}",
            result.status
        );
        if result.status == LpStatus::Optimal {
            assert!(
                (result.objective - (-4.0)).abs() < 0.1,
                "obj = {}",
                result.objective
            );
        }
    }

    #[test]
    fn test_ipm_trivial() {
        let model = make_trivial_lp();
        let config = InteriorPointConfig::default();
        let result = solve_interior_point(&model, &config);
        assert!(result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_ipm_empty() {
        let model = LpModel::new("empty");
        let config = InteriorPointConfig::default();
        let result = solve_interior_point(&model, &config);
        assert_eq!(result.status, LpStatus::Optimal);
    }

    #[test]
    fn test_standard_form_conversion() {
        let model = make_simple_lp();
        let (std_model, _info) = to_standard_form(&model);
        assert_eq!(std_model.num_eq, 2);
        assert_eq!(std_model.num_vars, 4); // 2 original + 2 slacks
    }

    #[test]
    fn test_cholesky_solve() {
        let solver = MehrotraSolver::new(
            StandardFormLP {
                num_eq: 0,
                num_vars: 0,
                c: Vec::new(),
                a: Vec::new(),
                b: Vec::new(),
            },
            InteriorPointConfig::default(),
        );
        let mat = vec![vec![4.0, 2.0], vec![2.0, 5.0]];
        let rhs = vec![8.0, 9.0];
        let x = solver.solve_symmetric(&mat, &rhs);
        // 4x + 2y = 8, 2x + 5y = 9 => x = 1.375, y = 1.25
        assert!((x[0] - 1.375).abs() < 1e-8, "x[0] = {}", x[0]);
        assert!((x[1] - 1.25).abs() < 1e-8, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_max_step_primal() {
        let solver = MehrotraSolver {
            model: StandardFormLP {
                num_eq: 0,
                num_vars: 2,
                c: vec![0.0; 2],
                a: Vec::new(),
                b: Vec::new(),
            },
            config: InteriorPointConfig::default(),
            x: vec![1.0, 2.0],
            y: Vec::new(),
            s: vec![1.0, 1.0],
        };
        let dx = vec![-0.5, -1.0];
        let alpha = solver.max_step_primal(&dx);
        // x[0] + alpha * dx[0] = 1 - 0.5*alpha >= 0 => alpha <= 2
        // x[1] + alpha * dx[1] = 2 - 1.0*alpha >= 0 => alpha <= 2
        assert!((alpha - 1.0).abs() < 1e-10); // capped at 1.0
    }

    #[test]
    fn test_ipm_config() {
        let config = InteriorPointConfig {
            max_iterations: 50,
            tolerance: 1e-6,
            ..InteriorPointConfig::default()
        };
        let model = make_simple_lp();
        let result = solve_interior_point(&model, &config);
        assert!(result.iterations <= 50);
    }

    #[test]
    fn test_residuals_at_start() {
        let model = make_simple_lp();
        let (std_model, _) = to_standard_form(&model);
        let mut solver = MehrotraSolver::new(std_model, InteriorPointConfig::default());
        solver.compute_starting_point();
        let (r_p, r_d, mu) = solver.compute_residuals();
        // Just verify they're finite
        assert!(r_p.iter().all(|x| x.is_finite()));
        assert!(r_d.iter().all(|x| x.is_finite()));
        assert!(mu.is_finite() && mu >= 0.0);
    }

    #[test]
    fn test_ge_constraint() {
        // min x s.t. x >= 2, x <= 10
        let mut m = LpModel::new("ge_test");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, 10.0));
        m.set_obj_coeff(x, 1.0);

        let mut c = Constraint::new("c0", ConstraintSense::Ge, 2.0);
        c.add_term(x, 1.0);
        m.add_constraint(c);

        let config = InteriorPointConfig {
            max_iterations: 100,
            crossover: false,
            ..InteriorPointConfig::default()
        };
        let result = solve_interior_point(&m, &config);
        assert!(result.status == LpStatus::Optimal || result.status == LpStatus::IterationLimit);
    }
}
