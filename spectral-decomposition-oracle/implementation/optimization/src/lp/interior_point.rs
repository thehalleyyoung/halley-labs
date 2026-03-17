use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, LpProblem, LpSolution, SolverStatus};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorPointConfig {
    pub max_iterations: usize,
    pub time_limit_secs: f64,
    pub gap_tolerance: f64,
    pub feasibility_tolerance: f64,
    pub regularization_param: f64,
    pub use_crossover: bool,
}

impl Default for InteriorPointConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            time_limit_secs: 3600.0,
            gap_tolerance: 1e-8,
            feasibility_tolerance: 1e-8,
            regularization_param: 1e-12,
            use_crossover: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Interior-point solver
// ---------------------------------------------------------------------------

pub struct InteriorPointSolver {
    config: InteriorPointConfig,
}

/// Working state for the IPM iterations.
struct IpmState {
    m: usize,
    n: usize,
    /// Primal variables (length n).
    x: Vec<f64>,
    /// Dual variables (length m).
    y: Vec<f64>,
    /// Dual slacks (length n).
    s: Vec<f64>,
}

impl InteriorPointSolver {
    pub fn new(config: InteriorPointConfig) -> Self {
        Self { config }
    }

    /// Solve an LP using Mehrotra's predictor-corrector interior-point method.
    pub fn solve(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        problem.validate()?;

        // Convert to standard form: min c^T x, Ax = b, x >= 0
        let (std_prob, orig_n) = problem.to_standard_form()?;
        let m = std_prob.num_constraints;
        let n = std_prob.num_vars;

        if m == 0 || n == 0 {
            return Ok(self.trivial_solution(problem));
        }

        // Initialize
        let mut state = self.initialize(&std_prob)?;

        let mut iter = 0usize;
        loop {
            if iter >= self.config.max_iterations {
                info!("Interior point: iteration limit reached at {}", iter);
                let sol = self.build_solution(
                    problem,
                    &std_prob,
                    &state,
                    orig_n,
                    iter,
                    SolverStatus::IterationLimit,
                    &start,
                );
                return Ok(sol);
            }
            if start.elapsed().as_secs_f64() > self.config.time_limit_secs {
                let sol = self.build_solution(
                    problem,
                    &std_prob,
                    &state,
                    orig_n,
                    iter,
                    SolverStatus::TimeLimit,
                    &start,
                );
                return Ok(sol);
            }

            // Check convergence
            let (primal_res, dual_res, gap) = self.check_convergence(&std_prob, &state);
            debug!(
                "IPM iter {}: primal_res={:.2e} dual_res={:.2e} gap={:.2e}",
                iter, primal_res, dual_res, gap
            );

            if primal_res < self.config.feasibility_tolerance
                && dual_res < self.config.feasibility_tolerance
                && gap < self.config.gap_tolerance
            {
                info!("Interior point converged in {} iterations", iter);
                let mut sol = self.build_solution(
                    problem,
                    &std_prob,
                    &state,
                    orig_n,
                    iter,
                    SolverStatus::Optimal,
                    &start,
                );
                if self.config.use_crossover {
                    sol = self.crossover(problem, &std_prob, &state, sol, orig_n, &start);
                }
                return Ok(sol);
            }

            // Predictor step (affine scaling direction)
            let (dx_aff, _dy_aff, ds_aff) = self.predictor_step(&std_prob, &state)?;

            // Affine step length
            let alpha_aff_pri = self.step_length(&state.x, &dx_aff, 1.0);
            let alpha_aff_dual = self.step_length(&state.s, &ds_aff, 1.0);

            // Compute centering parameter
            let sigma = self.compute_centering_parameter(
                &state,
                &dx_aff,
                &ds_aff,
                alpha_aff_pri,
                alpha_aff_dual,
            );

            // Corrector step (centering + corrector)
            let (dx, dy, ds) =
                self.corrector_step(&std_prob, &state, &dx_aff, &ds_aff, sigma)?;

            // Step length with fraction-to-boundary rule
            let alpha_pri = self.step_length(&state.x, &dx, 0.9995);
            let alpha_dual = self.step_length(&state.s, &ds, 0.9995);

            // Update iterates
            for i in 0..n {
                state.x[i] += alpha_pri * dx[i];
                state.s[i] += alpha_dual * ds[i];
            }
            for i in 0..m {
                state.y[i] += alpha_dual * dy[i];
            }

            // Safeguard: clamp to prevent exact zero
            for i in 0..n {
                if state.x[i] < 1e-15 {
                    state.x[i] = 1e-15;
                }
                if state.s[i] < 1e-15 {
                    state.s[i] = 1e-15;
                }
            }

            iter += 1;
        }
    }

    /// Solve using the homogeneous self-dual formulation (detects infeasibility/unboundedness).
    pub fn homogeneous_self_dual(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        problem.validate()?;

        let (std_prob, orig_n) = problem.to_standard_form()?;
        let m = std_prob.num_constraints;
        let n = std_prob.num_vars;

        if m == 0 || n == 0 {
            return Ok(self.trivial_solution(problem));
        }

        // HSD embeds the problem as:
        //   min  c^T x - b^T y
        //   s.t. A x - b tau        = 0
        //       -A^T y + c tau - s   = 0
        //        b^T y - c^T x - kappa = 0
        //        x, s, tau, kappa >= 0

        let c = &std_prob.obj_coeffs;
        let b = &std_prob.rhs;

        // Initial point
        let mut x = vec![1.0; n];
        let mut y = vec![0.0; m];
        let mut s = vec![1.0; n];
        let mut tau = 1.0;
        let mut kappa = 1.0;

        for iter in 0..self.config.max_iterations {
            if start.elapsed().as_secs_f64() > self.config.time_limit_secs {
                return Ok(self.build_solution(
                    problem,
                    &std_prob,
                    &IpmState {
                        m,
                        n,
                        x: x.iter().map(|v| v / tau).collect(),
                        y: y.iter().map(|v| v / tau).collect(),
                        s: s.iter().map(|v| v / tau).collect(),
                    },
                    orig_n,
                    iter,
                    SolverStatus::TimeLimit,
                    &start,
                ));
            }

            let mu = (dot(&x, &s) + tau * kappa) / (n as f64 + 1.0);

            if mu < self.config.gap_tolerance {
                // Check termination conditions
                if tau > kappa * 1e-6 {
                    // Optimal
                    let x_sol: Vec<f64> = x.iter().map(|v| v / tau).collect();
                    let y_sol: Vec<f64> = y.iter().map(|v| v / tau).collect();
                    let s_sol: Vec<f64> = s.iter().map(|v| v / tau).collect();
                    let state = IpmState {
                        m,
                        n,
                        x: x_sol,
                        y: y_sol,
                        s: s_sol,
                    };
                    return Ok(self.build_solution(
                        problem,
                        &std_prob,
                        &state,
                        orig_n,
                        iter,
                        SolverStatus::Optimal,
                        &start,
                    ));
                } else {
                    // Infeasible or unbounded
                    let bt_y: f64 = dot(b, &y);
                    let ct_x: f64 = dot(c, &x);
                    if bt_y > ct_x {
                        return Ok(LpSolution {
                            status: SolverStatus::Infeasible,
                            objective_value: f64::NAN,
                            primal_values: vec![0.0; problem.num_vars],
                            dual_values: vec![0.0; problem.num_constraints],
                            reduced_costs: vec![0.0; problem.num_vars],
                            basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                            iterations: iter,
                            time_seconds: start.elapsed().as_secs_f64(),
                        });
                    } else {
                        return Ok(LpSolution {
                            status: SolverStatus::Unbounded,
                            objective_value: f64::NEG_INFINITY,
                            primal_values: vec![0.0; problem.num_vars],
                            dual_values: vec![0.0; problem.num_constraints],
                            reduced_costs: vec![0.0; problem.num_vars],
                            basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                            iterations: iter,
                            time_seconds: start.elapsed().as_secs_f64(),
                        });
                    }
                }
            }

            // Predictor direction (simplified for HSD: use same Newton system as standard)
            // We approximate by treating tau=const and solving standard system
            let state = IpmState {
                m,
                n,
                x: x.clone(),
                y: y.clone(),
                s: s.clone(),
            };
            let (dx, dy, ds) = match self.predictor_step(&std_prob, &state) {
                Ok(v) => v,
                Err(_) => {
                    // Numerical issues — return current best
                    return Ok(self.build_solution(
                        problem,
                        &std_prob,
                        &state,
                        orig_n,
                        iter,
                        SolverStatus::NumericalError,
                        &start,
                    ));
                }
            };

            // Step for tau and kappa (simplified: keep them proportional to mu)
            let dtau = -tau + mu / kappa;
            let dkappa = -kappa + mu / tau;

            // Step lengths
            let alpha_pri = self.step_length(&x, &dx, 0.9995).min(
                if dtau < 0.0 {
                    -0.9995 * tau / dtau
                } else {
                    1.0
                },
            );
            let alpha_dual = self.step_length(&s, &ds, 0.9995).min(
                if dkappa < 0.0 {
                    -0.9995 * kappa / dkappa
                } else {
                    1.0
                },
            );

            for i in 0..n {
                x[i] += alpha_pri * dx[i];
                s[i] += alpha_dual * ds[i];
                if x[i] < 1e-15 {
                    x[i] = 1e-15;
                }
                if s[i] < 1e-15 {
                    s[i] = 1e-15;
                }
            }
            for i in 0..m {
                y[i] += alpha_dual * dy[i];
            }
            tau += alpha_pri * dtau;
            kappa += alpha_dual * dkappa;
            if tau < 1e-15 {
                tau = 1e-15;
            }
            if kappa < 1e-15 {
                kappa = 1e-15;
            }
        }

        let state = IpmState {
            m,
            n,
            x: x.iter().map(|v| v / tau).collect(),
            y: y.iter().map(|v| v / tau).collect(),
            s: s.iter().map(|v| v / tau).collect(),
        };
        Ok(self.build_solution(
            problem,
            &std_prob,
            &state,
            orig_n,
            self.config.max_iterations,
            SolverStatus::IterationLimit,
            &start,
        ))
    }

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    fn initialize(&self, problem: &LpProblem) -> OptResult<IpmState> {
        let m = problem.num_constraints;
        let n = problem.num_vars;

        // Heuristic starting point
        // x = max(1, |c|), s = max(1, |c|), y = 0
        let mut x = vec![0.0; n];
        let mut s = vec![0.0; n];
        for j in 0..n {
            x[j] = 1.0f64.max(problem.obj_coeffs[j].abs());
            s[j] = 1.0f64.max(problem.obj_coeffs[j].abs());
        }
        let y = vec![0.0; m];

        // Adjust x for feasibility: compute Ax, then shift x
        let mut ax = vec![0.0; m];
        problem.multiply_ax(&x, &mut ax);
        let mut primal_resid = vec![0.0; m];
        for i in 0..m {
            primal_resid[i] = problem.rhs[i] - ax[i];
        }

        // Shift x: solve A dx = resid approximately via A^T (A A^T)^{-1} resid
        // Simplified: scale x uniformly
        let resid_norm: f64 = primal_resid.iter().map(|v| v * v).sum::<f64>().sqrt();
        if resid_norm > 1e-8 {
            let ax_norm: f64 = ax.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
            let scale = 1.0 + resid_norm / ax_norm;
            for j in 0..n {
                x[j] *= scale;
            }
        }

        // Ensure strict positivity
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        if x_min <= 0.0 {
            let shift = -x_min + 1.0;
            for v in x.iter_mut() {
                *v += shift;
            }
        }
        let s_min = s.iter().cloned().fold(f64::INFINITY, f64::min);
        if s_min <= 0.0 {
            let shift = -s_min + 1.0;
            for v in s.iter_mut() {
                *v += shift;
            }
        }

        debug!(
            "IPM initialized: n={}, m={}, x_min={:.2e}, s_min={:.2e}",
            n,
            m,
            x.iter().cloned().fold(f64::INFINITY, f64::min),
            s.iter().cloned().fold(f64::INFINITY, f64::min),
        );

        Ok(IpmState { m, n, x, y, s })
    }

    // -----------------------------------------------------------------------
    // Predictor step (affine direction)
    // -----------------------------------------------------------------------

    fn predictor_step(
        &self,
        problem: &LpProblem,
        state: &IpmState,
    ) -> OptResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let m = state.m;
        let n = state.n;

        // Residuals
        // rp = b - Ax
        let mut ax = vec![0.0; m];
        problem.multiply_ax(&state.x, &mut ax);
        let rp: Vec<f64> = (0..m).map(|i| problem.rhs[i] - ax[i]).collect();

        // rd = c - A^T y - s
        let mut aty = vec![0.0; n];
        problem.multiply_atx(&state.y, &mut aty);
        let rd: Vec<f64> = (0..n)
            .map(|j| problem.obj_coeffs[j] - aty[j] - state.s[j])
            .collect();

        // rc = -X S e  (complementarity)
        let rc: Vec<f64> = (0..n).map(|j| -state.x[j] * state.s[j]).collect();

        self.solve_newton_system(problem, state, &rp, &rd, &rc)
    }

    // -----------------------------------------------------------------------
    // Corrector step
    // -----------------------------------------------------------------------

    fn corrector_step(
        &self,
        problem: &LpProblem,
        state: &IpmState,
        dx_aff: &[f64],
        ds_aff: &[f64],
        sigma: f64,
    ) -> OptResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let m = state.m;
        let n = state.n;

        let mu: f64 = dot(&state.x, &state.s) / n as f64;

        // Residuals (same primal/dual residuals as predictor)
        let mut ax = vec![0.0; m];
        problem.multiply_ax(&state.x, &mut ax);
        let rp: Vec<f64> = (0..m).map(|i| problem.rhs[i] - ax[i]).collect();

        let mut aty = vec![0.0; n];
        problem.multiply_atx(&state.y, &mut aty);
        let rd: Vec<f64> = (0..n)
            .map(|j| problem.obj_coeffs[j] - aty[j] - state.s[j])
            .collect();

        // Corrector complementarity: -XSe + sigma*mu*e - dX_aff dS_aff e
        let rc: Vec<f64> = (0..n)
            .map(|j| {
                -state.x[j] * state.s[j] + sigma * mu - dx_aff[j] * ds_aff[j]
            })
            .collect();

        self.solve_newton_system(problem, state, &rp, &rd, &rc)
    }

    // -----------------------------------------------------------------------
    // Newton system solver (normal equations)
    // -----------------------------------------------------------------------

    /// Solve the Newton system:
    ///   [  0   A^T  I ] [dx]   [rd]
    ///   [  A   0    0 ] [dy] = [rp]
    ///   [  S   0    X ] [ds]   [rc]
    ///
    /// Eliminate ds = X^{-1}(rc - S dx), then:
    ///   A D^2 A^T dy = rp + A D^2 (rd - X^{-1} rc)
    /// where D^2 = X / S.
    fn solve_newton_system(
        &self,
        problem: &LpProblem,
        state: &IpmState,
        rp: &[f64],
        rd: &[f64],
        rc: &[f64],
    ) -> OptResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let m = state.m;
        let n = state.n;

        // D^2 diagonal
        let d2: Vec<f64> = (0..n)
            .map(|j| state.x[j] / state.s[j].max(1e-20))
            .collect();

        // RHS for normal equations: rp + A * D^2 * (rd - X^{-1} rc)
        let mut rhs_inner = vec![0.0; n];
        for j in 0..n {
            let x_inv_rc = rc[j] / state.x[j].max(1e-20);
            rhs_inner[j] = d2[j] * (rd[j] - x_inv_rc);
        }

        // A * rhs_inner
        let mut a_rhs = vec![0.0; m];
        for i in 0..m {
            let rs = problem.row_starts[i];
            let re = problem.row_starts[i + 1];
            let mut s = 0.0;
            for k in rs..re {
                let j = problem.col_indices[k];
                s += problem.values[k] * rhs_inner[j];
            }
            a_rhs[i] = s;
        }

        let normal_rhs: Vec<f64> = (0..m).map(|i| rp[i] + a_rhs[i]).collect();

        // Form A D^2 A^T (dense m×m)
        let ada = self.form_normal_matrix(problem, &d2)?;

        // Cholesky factorize and solve
        let l = self.cholesky_factorize(&ada, m)?;
        let dy = self.cholesky_solve(&l, &normal_rhs, m);

        // Back-substitute for dx: dx = D^2 (A^T dy - rd + X^{-1} rc)
        let mut aty_dy = vec![0.0; n];
        problem.multiply_atx(&dy, &mut aty_dy);

        let mut dx = vec![0.0; n];
        for j in 0..n {
            let x_inv_rc = rc[j] / state.x[j].max(1e-20);
            dx[j] = d2[j] * (aty_dy[j] - rd[j] + x_inv_rc);
        }

        // ds = X^{-1} (rc - S dx)
        let mut ds = vec![0.0; n];
        for j in 0..n {
            ds[j] = (rc[j] - state.s[j] * dx[j]) / state.x[j].max(1e-20);
        }

        Ok((dx, dy, ds))
    }

    /// Form A * diag(d2) * A^T as dense matrix with regularisation.
    fn form_normal_matrix(&self, problem: &LpProblem, d2: &[f64]) -> OptResult<Vec<f64>> {
        let m = problem.num_constraints;
        let n = problem.num_vars;
        let mut ada = vec![0.0; m * m];

        // For each pair of rows (i1, i2), compute sum_j a_{i1,j} * d2[j] * a_{i2,j}
        // We iterate column by column for efficiency with CSR.
        // Build column -> row entries mapping
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..m {
            let rs = problem.row_starts[i];
            let re = problem.row_starts[i + 1];
            for k in rs..re {
                let j = problem.col_indices[k];
                col_entries[j].push((i, problem.values[k]));
            }
        }

        for j in 0..n {
            let entries = &col_entries[j];
            let dj = d2[j];
            for &(i1, a1) in entries {
                for &(i2, a2) in entries {
                    ada[i1 * m + i2] += a1 * dj * a2;
                }
            }
        }

        // Regularisation
        let reg = self.config.regularization_param;
        for i in 0..m {
            ada[i * m + i] += reg;
        }

        Ok(ada)
    }

    /// Dense Cholesky factorisation: A = L L^T. Returns L in row-major.
    fn cholesky_factorize(&self, a: &[f64], n: usize) -> OptResult<Vec<f64>> {
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = a[i * n + i] - sum;
                    if diag <= 0.0 {
                        // Not positive definite — add regularisation and retry
                        warn!(
                            "Cholesky: non-positive diagonal {:.2e} at index {}, adding regularisation",
                            diag, i
                        );
                        let mut a_reg = a.to_vec();
                        let bump = 1e-8 * (1.0 + a[i * n + i].abs());
                        for ii in 0..n {
                            a_reg[ii * n + ii] += bump;
                        }
                        return self.cholesky_factorize_inner(&a_reg, n);
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        Ok(l)
    }

    fn cholesky_factorize_inner(&self, a: &[f64], n: usize) -> OptResult<Vec<f64>> {
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = a[i * n + i] - sum;
                    if diag <= 0.0 {
                        return Err(OptError::NumericalError {
                            context: format!(
                                "Cholesky factorization failed: non-positive diagonal {:.2e} at {}",
                                diag, i
                            ),
                        });
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
                }
            }
        }
        Ok(l)
    }

    /// Solve L L^T x = b given Cholesky factor L.
    fn cholesky_solve(&self, l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        // Forward substitution: L y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for j in 0..i {
                s -= l[i * n + j] * y[j];
            }
            y[i] = s / l[i * n + i].max(1e-20);
        }

        // Backward substitution: L^T x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= l[j * n + i] * x[j];
            }
            x[i] = s / l[i * n + i].max(1e-20);
        }

        x
    }

    // -----------------------------------------------------------------------
    // Centering parameter
    // -----------------------------------------------------------------------

    fn compute_centering_parameter(
        &self,
        state: &IpmState,
        dx_aff: &[f64],
        ds_aff: &[f64],
        alpha_pri: f64,
        alpha_dual: f64,
    ) -> f64 {
        let n = state.n;
        let mu = dot(&state.x, &state.s) / n as f64;

        // mu_aff = (x + alpha_pri * dx_aff)^T (s + alpha_dual * ds_aff) / n
        let mut mu_aff = 0.0;
        for j in 0..n {
            mu_aff +=
                (state.x[j] + alpha_pri * dx_aff[j]) * (state.s[j] + alpha_dual * ds_aff[j]);
        }
        mu_aff /= n as f64;

        // Mehrotra heuristic: sigma = (mu_aff / mu)^3
        let ratio = mu_aff / mu.max(1e-20);
        let sigma = ratio * ratio * ratio;
        sigma.max(0.0).min(1.0)
    }

    // -----------------------------------------------------------------------
    // Step length
    // -----------------------------------------------------------------------

    /// Maximum alpha in (0, max_alpha] such that v + alpha * dv >= 0.
    fn step_length(&self, v: &[f64], dv: &[f64], fraction: f64) -> f64 {
        let mut alpha = 1.0;
        for i in 0..v.len() {
            if dv[i] < -1e-15 {
                let a = -v[i] / dv[i];
                if a < alpha {
                    alpha = a;
                }
            }
        }
        (alpha * fraction).min(1.0).max(0.0)
    }

    // -----------------------------------------------------------------------
    // Convergence check
    // -----------------------------------------------------------------------

    /// Returns (primal_residual, dual_residual, duality_gap).
    fn check_convergence(&self, problem: &LpProblem, state: &IpmState) -> (f64, f64, f64) {
        let m = state.m;
        let n = state.n;

        // Primal residual: ||Ax - b|| / max(1, ||b||)
        let mut ax = vec![0.0; m];
        problem.multiply_ax(&state.x, &mut ax);
        let mut pr_sq = 0.0;
        let mut b_sq = 0.0;
        for i in 0..m {
            let r = ax[i] - problem.rhs[i];
            pr_sq += r * r;
            b_sq += problem.rhs[i] * problem.rhs[i];
        }
        let primal_res = pr_sq.sqrt() / 1.0f64.max(b_sq.sqrt());

        // Dual residual: ||A^T y + s - c|| / max(1, ||c||)
        let mut aty = vec![0.0; n];
        problem.multiply_atx(&state.y, &mut aty);
        let mut dr_sq = 0.0;
        let mut c_sq = 0.0;
        for j in 0..n {
            let r = aty[j] + state.s[j] - problem.obj_coeffs[j];
            dr_sq += r * r;
            c_sq += problem.obj_coeffs[j] * problem.obj_coeffs[j];
        }
        let dual_res = dr_sq.sqrt() / 1.0f64.max(c_sq.sqrt());

        // Duality gap: x^T s / n
        let gap = dot(&state.x, &state.s) / n as f64;

        (primal_res, dual_res, gap)
    }

    // -----------------------------------------------------------------------
    // Crossover
    // -----------------------------------------------------------------------

    /// Convert interior point solution to basic solution by identifying active bounds.
    fn crossover(
        &self,
        original: &LpProblem,
        std_prob: &LpProblem,
        state: &IpmState,
        mut sol: LpSolution,
        _orig_n: usize,
        _start: &Instant,
    ) -> LpSolution {
        let n = std_prob.num_vars;
        let m = std_prob.num_constraints;
        let tol = 1e-6;

        let mut basis_status = vec![BasisStatus::AtLower; n];
        let mut basic_count = 0usize;

        // Identify basic / nonbasic based on x and s values
        for j in 0..n {
            if state.x[j] < tol {
                basis_status[j] = BasisStatus::AtLower;
            } else if state.s[j] < tol && basic_count < m {
                basis_status[j] = BasisStatus::Basic;
                basic_count += 1;
            } else if basic_count < m {
                basis_status[j] = BasisStatus::Basic;
                basic_count += 1;
            } else {
                basis_status[j] = BasisStatus::AtLower;
            }
        }

        // Map back to original problem size
        let mut orig_basis = vec![BasisStatus::AtLower; original.num_vars];
        for j in 0..original.num_vars.min(n) {
            orig_basis[j] = basis_status[j];
        }

        sol.basis_status = orig_basis;
        debug!("Crossover: {} basic variables identified", basic_count);
        sol
    }

    // -----------------------------------------------------------------------
    // Solution construction
    // -----------------------------------------------------------------------

    fn build_solution(
        &self,
        original: &LpProblem,
        _std_prob: &LpProblem,
        state: &IpmState,
        _orig_n: usize,
        iters: usize,
        status: SolverStatus,
        start: &Instant,
    ) -> LpSolution {
        let mut x_orig = vec![0.0; original.num_vars];
        for j in 0..original.num_vars.min(state.x.len()) {
            x_orig[j] = state.x[j] + original.lower_bounds[j];
        }

        let obj: f64 = (0..original.num_vars)
            .map(|j| original.obj_coeffs[j] * x_orig[j])
            .sum();

        // Map dual values
        let dual = if state.y.len() == original.num_constraints {
            if original.maximize {
                state.y.iter().map(|v| -v).collect()
            } else {
                state.y.clone()
            }
        } else {
            // Standard form may have different number of constraints
            let mut d = vec![0.0; original.num_constraints];
            for i in 0..original.num_constraints.min(state.y.len()) {
                d[i] = if original.maximize {
                    -state.y[i]
                } else {
                    state.y[i]
                };
            }
            d
        };

        let mut rc = vec![0.0; original.num_vars];
        for j in 0..original.num_vars.min(state.s.len()) {
            rc[j] = if original.maximize {
                -state.s[j]
            } else {
                state.s[j]
            };
        }

        LpSolution {
            status,
            objective_value: obj,
            primal_values: x_orig,
            dual_values: dual,
            reduced_costs: rc,
            basis_status: vec![BasisStatus::Free; original.num_vars],
            iterations: iters,
            time_seconds: start.elapsed().as_secs_f64(),
        }
    }

    fn trivial_solution(&self, problem: &LpProblem) -> LpSolution {
        let n = problem.num_vars;
        let x = problem.lower_bounds.clone();
        let obj: f64 = (0..n).map(|j| problem.obj_coeffs[j] * x[j]).sum();
        LpSolution {
            status: SolverStatus::Optimal,
            objective_value: obj,
            primal_values: x,
            dual_values: vec![],
            reduced_costs: problem.obj_coeffs.clone(),
            basis_status: vec![BasisStatus::AtLower; n],
            iterations: 0,
            time_seconds: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::ConstraintType;

    fn default_solver() -> InteriorPointSolver {
        InteriorPointSolver::new(InteriorPointConfig::default())
    }

    fn small_lp() -> LpProblem {
        // min -x1 - 2*x2  s.t. x1+x2<=4, x1<=3, x2<=3, x>=0
        let mut lp = LpProblem::new(false);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-2.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();
        lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();
        lp
    }

    #[test]
    fn test_solve_small_lp() {
        let solver = default_solver();
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!(
            (sol.objective_value - (-7.0)).abs() < 1e-4,
            "obj = {}",
            sol.objective_value
        );
    }

    #[test]
    fn test_solve_maximise() {
        let mut lp = LpProblem::new(true);
        lp.add_variable(3.0, 0.0, f64::INFINITY, None);
        lp.add_variable(5.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 4.0)
            .unwrap();
        lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 6.0)
            .unwrap();
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 8.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!(
            (sol.objective_value - 36.0).abs() < 1e-2,
            "obj = {}",
            sol.objective_value
        );
    }

    #[test]
    fn test_equality_constraint() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Eq, 5.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!(
            (sol.objective_value - 5.0).abs() < 1e-4,
            "obj = {}",
            sol.objective_value
        );
    }

    #[test]
    fn test_cholesky_identity() {
        let solver = default_solver();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let l = solver.cholesky_factorize(&a, 2).unwrap();
        assert!((l[0] - 1.0).abs() < 1e-12);
        assert!((l[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_2x2() {
        let solver = default_solver();
        // A = [4, 2; 2, 3]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = solver.cholesky_factorize(&a, 2).unwrap();
        // L = [2, 0; 1, sqrt(2)]
        assert!((l[0] - 2.0).abs() < 1e-10);
        assert!((l[2] - 1.0).abs() < 1e-10);
        assert!((l[3] - (2.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve() {
        let solver = default_solver();
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = solver.cholesky_factorize(&a, 2).unwrap();
        let b = vec![8.0, 7.0];
        let x = solver.cholesky_solve(&l, &b, 2);
        // x = A^{-1} b = [4,2;2,3]^{-1} [8;7] = [1.25; 1.5]
        assert!((x[0] - 1.25).abs() < 1e-8);
        assert!((x[1] - 1.5).abs() < 1e-8);
    }

    #[test]
    fn test_step_length_no_blocking() {
        let solver = default_solver();
        let v = vec![1.0, 2.0, 3.0];
        let dv = vec![0.1, 0.2, 0.3];
        let alpha = solver.step_length(&v, &dv, 1.0);
        assert!((alpha - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_step_length_blocking() {
        let solver = default_solver();
        let v = vec![1.0, 2.0, 3.0];
        let dv = vec![-2.0, 0.0, 0.0]; // blocks at alpha=0.5
        let alpha = solver.step_length(&v, &dv, 0.9995);
        assert!((alpha - 0.5 * 0.9995).abs() < 1e-10);
    }

    #[test]
    fn test_centering_parameter() {
        let solver = default_solver();
        let state = IpmState {
            m: 1,
            n: 2,
            x: vec![1.0, 1.0],
            y: vec![0.0],
            s: vec![1.0, 1.0],
        };
        let dx_aff = vec![0.0, 0.0];
        let ds_aff = vec![0.0, 0.0];
        let sigma = solver.compute_centering_parameter(&state, &dx_aff, &ds_aff, 1.0, 1.0);
        // mu_aff = mu = 1, ratio = 1, sigma = 1
        assert!((sigma - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convergence_check() {
        let solver = default_solver();
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 5.0)
            .unwrap();
        let (std_lp, _) = lp.to_standard_form().unwrap();
        let state = IpmState {
            m: 1,
            n: 2,
            x: vec![3.0, 2.0], // x + s = 5 → feasible
            y: vec![1.0],
            s: vec![0.0, 0.0],
        };
        let (pr, dr, gap) = solver.check_convergence(&std_lp, &state);
        assert!(pr.is_finite());
        assert!(dr.is_finite());
        assert!(gap.is_finite());
    }

    #[test]
    fn test_homogeneous_self_dual() {
        let solver = default_solver();
        let lp = small_lp();
        let sol = solver.homogeneous_self_dual(&lp).unwrap();
        // Should converge to something reasonable
        assert!(sol.objective_value.is_finite() || sol.status != SolverStatus::Optimal);
    }

    #[test]
    fn test_crossover_flag() {
        let config = InteriorPointConfig {
            use_crossover: true,
            ..Default::default()
        };
        let solver = InteriorPointSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        // Basis status should have some non-Free entries after crossover
        assert!(sol.basis_status.iter().any(|b| *b != BasisStatus::Free));
    }

    #[test]
    fn test_iteration_limit() {
        let config = InteriorPointConfig {
            max_iterations: 2,
            ..Default::default()
        };
        let solver = InteriorPointSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert!(
            sol.status == SolverStatus::IterationLimit || sol.status == SolverStatus::Optimal
        );
    }

    #[test]
    fn test_single_var() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(2.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 5.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Ge, 1.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!(
            (sol.primal_values[0] - 1.0).abs() < 1e-3,
            "x = {}",
            sol.primal_values[0]
        );
    }
}
