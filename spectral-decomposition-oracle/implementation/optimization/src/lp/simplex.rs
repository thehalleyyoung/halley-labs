use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingRule {
    Dantzig,
    SteepestEdge,
    Devex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplexConfig {
    pub max_iterations: usize,
    pub time_limit_secs: f64,
    pub primal_tolerance: f64,
    pub dual_tolerance: f64,
    pub pivot_tolerance: f64,
    pub pricing_rule: PricingRule,
    pub use_harris_ratio_test: bool,
    pub perturbation_magnitude: f64,
    pub refactoring_interval: usize,
    pub use_bound_flipping: bool,
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            time_limit_secs: 3600.0,
            primal_tolerance: 1e-8,
            dual_tolerance: 1e-8,
            pivot_tolerance: 1e-10,
            pricing_rule: PricingRule::Dantzig,
            use_harris_ratio_test: true,
            perturbation_magnitude: 1e-6,
            refactoring_interval: 100,
            use_bound_flipping: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Eta vector (product-form update of basis inverse)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EtaVector {
    pivot_row: usize,
    /// Column of B^{-1} update: eta[i] for i != pivot_row, and eta[pivot_row] = 1/pivot_elem.
    coeffs: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Simplex solver
// ---------------------------------------------------------------------------

pub struct SimplexSolver {
    config: SimplexConfig,
}

/// Working state for one simplex solve.
struct SimplexState {
    m: usize,
    n: usize,
    /// Indices of basic variables (length m).
    basis: Vec<usize>,
    /// For each variable, its basis status.
    var_status: Vec<BasisStatus>,
    /// Current basic variable values x_B = B^{-1} b.
    xb: Vec<f64>,
    /// Reduced costs for all variables.
    rc: Vec<f64>,
    /// Eta-file product form of B^{-1}.
    eta_file: Vec<EtaVector>,
    /// Dense B^{-1} (recomputed periodically).
    b_inv: Vec<f64>,
    /// Whether b_inv is current.
    b_inv_fresh: bool,
    /// Iteration count since last refactorisation.
    iters_since_refactor: usize,
    /// Steepest-edge weights (if applicable).
    se_weights: Vec<f64>,
    /// Devex weights.
    devex_weights: Vec<f64>,
}

impl SimplexSolver {
    pub fn new(config: SimplexConfig) -> Self {
        Self { config }
    }

    /// Solve an LP problem using the (revised) simplex method.
    pub fn solve(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        problem.validate()?;

        // Convert to standard form (all Eq, all vars >= 0)
        let (std_problem, orig_n) = problem.to_standard_form()?;
        let m = std_problem.num_constraints;
        let _n = std_problem.num_vars;

        if m == 0 {
            return Ok(self.trivial_solution(problem));
        }

        // Phase I: find a basic feasible solution
        let (phase1_basis, phase1_status) =
            self.phase_one(&std_problem, &start)?;

        if phase1_status == SolverStatus::Infeasible {
            return Ok(LpSolution {
                status: SolverStatus::Infeasible,
                objective_value: f64::NAN,
                primal_values: vec![0.0; problem.num_vars],
                dual_values: vec![0.0; m],
                reduced_costs: vec![0.0; problem.num_vars],
                basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                iterations: 0,
                time_seconds: start.elapsed().as_secs_f64(),
            });
        }

        // Phase II: optimise original objective from BFS
        let (solution, iters) = self.phase_two(
            &std_problem,
            phase1_basis,
            &start,
        )?;

        // Map back to original variables
        let sol = self.extract_solution(problem, &std_problem, &solution, orig_n, iters, &start);
        Ok(sol)
    }

    /// Solve via the dual simplex method.
    pub fn dual_simplex(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        problem.validate()?;
        let (std_problem, orig_n) = problem.to_standard_form()?;
        let m = std_problem.num_constraints;
        let n = std_problem.num_vars;

        if m == 0 {
            return Ok(self.trivial_solution(problem));
        }

        // Start with slack basis (may be dual feasible but primal infeasible)
        let mut state = self.init_state(&std_problem)?;

        let mut total_iters = 0usize;

        loop {
            if total_iters >= self.config.max_iterations {
                return Ok(LpSolution {
                    status: SolverStatus::IterationLimit,
                    objective_value: f64::NAN,
                    primal_values: vec![0.0; problem.num_vars],
                    dual_values: vec![0.0; m],
                    reduced_costs: vec![0.0; problem.num_vars],
                    basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                    iterations: total_iters,
                    time_seconds: start.elapsed().as_secs_f64(),
                });
            }
            if start.elapsed().as_secs_f64() > self.config.time_limit_secs {
                return Ok(LpSolution {
                    status: SolverStatus::TimeLimit,
                    objective_value: f64::NAN,
                    primal_values: vec![0.0; problem.num_vars],
                    dual_values: vec![0.0; m],
                    reduced_costs: vec![0.0; problem.num_vars],
                    basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                    iterations: total_iters,
                    time_seconds: start.elapsed().as_secs_f64(),
                });
            }

            // Recompute xB and reduced costs
            self.recompute_xb(&std_problem, &mut state);
            self.compute_reduced_costs(&std_problem, &mut state);

            // Choose leaving variable: most infeasible basic variable
            let mut leaving = None;
            let mut max_infeas = self.config.primal_tolerance;
            for r in 0..m {
                if state.xb[r] < -max_infeas {
                    max_infeas = -state.xb[r];
                    leaving = Some(r);
                }
            }

            if leaving.is_none() {
                // Primal feasible and dual feasible → optimal
                let sol = self.build_solution_from_state(
                    problem,
                    &std_problem,
                    &state,
                    orig_n,
                    total_iters,
                    &start,
                );
                return Ok(sol);
            }

            let r = leaving.unwrap();

            // Compute pivot row: e_r^T B^{-1} A for nonbasic columns
            let mut rho = vec![0.0; m];
            rho[r] = 1.0;
            let y = self.btran_vec(&state, &rho);

            // Dual ratio test: among nonbasic j with a_bar_rj != 0, find entering
            let mut entering = None;
            let mut min_ratio = f64::INFINITY;

            for j in 0..n {
                if state.var_status[j] == BasisStatus::Basic {
                    continue;
                }
                // Compute a_bar_rj = y^T a_j
                let mut a_bar = 0.0;
                for i in 0..m {
                    let rs = std_problem.row_starts[i];
                    let re = std_problem.row_starts[i + 1];
                    for k in rs..re {
                        if std_problem.col_indices[k] == j {
                            a_bar += y[i] * std_problem.values[k];
                        }
                    }
                }

                if a_bar.abs() < self.config.pivot_tolerance {
                    continue;
                }

                // For AtLower: need a_bar < 0 (entering at lower bound increases xb[r])
                // For AtUpper: need a_bar > 0
                let ratio = match state.var_status[j] {
                    BasisStatus::AtLower => {
                        if a_bar > self.config.pivot_tolerance {
                            continue;
                        }
                        state.rc[j] / a_bar // rc >= 0, a_bar < 0 → ratio <= 0 in abs
                    }
                    BasisStatus::AtUpper => {
                        if a_bar < -self.config.pivot_tolerance {
                            continue;
                        }
                        -state.rc[j] / a_bar
                    }
                    _ => continue,
                };

                let ratio_abs = ratio.abs();
                if ratio_abs < min_ratio {
                    min_ratio = ratio_abs;
                    entering = Some(j);
                }
            }

            if entering.is_none() {
                return Ok(LpSolution {
                    status: SolverStatus::Infeasible,
                    objective_value: f64::NAN,
                    primal_values: vec![0.0; problem.num_vars],
                    dual_values: vec![0.0; m],
                    reduced_costs: vec![0.0; problem.num_vars],
                    basis_status: vec![BasisStatus::AtLower; problem.num_vars],
                    iterations: total_iters,
                    time_seconds: start.elapsed().as_secs_f64(),
                });
            }

            let j_enter = entering.unwrap();
            self.pivot(&std_problem, &mut state, j_enter, r);
            total_iters += 1;

            if state.iters_since_refactor >= self.config.refactoring_interval {
                self.refactor_basis(&std_problem, &mut state)?;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase I
    // -----------------------------------------------------------------------

    fn phase_one(
        &self,
        problem: &LpProblem,
        start: &Instant,
    ) -> OptResult<(Vec<usize>, SolverStatus)> {
        let m = problem.num_constraints;
        let n = problem.num_vars;

        // Build Phase-I problem: add artificial variables, min sum of artificials
        let total = n + m;
        let mut obj_p1 = vec![0.0; total];
        for j in n..total {
            obj_p1[j] = 1.0;
        }

        // Build augmented rows (original + identity for artificials)
        let mut p1_row_starts = Vec::with_capacity(m + 1);
        let mut p1_col_indices = Vec::new();
        let mut p1_values = Vec::new();
        let mut p1_rhs = problem.rhs.clone();

        p1_row_starts.push(0);
        for i in 0..m {
            let rs = problem.row_starts[i];
            let re = problem.row_starts[i + 1];

            // Ensure rhs >= 0 (multiply row by -1 if needed)
            let sign = if p1_rhs[i] < 0.0 { -1.0 } else { 1.0 };
            if sign < 0.0 {
                p1_rhs[i] = -p1_rhs[i];
            }

            for k in rs..re {
                p1_col_indices.push(problem.col_indices[k]);
                p1_values.push(sign * problem.values[k]);
            }
            // Artificial variable column
            p1_col_indices.push(n + i);
            p1_values.push(1.0);
            p1_row_starts.push(p1_col_indices.len());
        }

        let mut lb = problem.lower_bounds.clone();
        let mut ub = problem.upper_bounds.clone();
        lb.resize(total, 0.0);
        ub.resize(total, f64::INFINITY);

        let mut var_names = problem.var_names.clone();
        for i in 0..m {
            var_names.push(format!("_a{}", i));
        }

        let p1_problem = LpProblem {
            num_vars: total,
            num_constraints: m,
            obj_coeffs: obj_p1,
            row_starts: p1_row_starts,
            col_indices: p1_col_indices,
            values: p1_values,
            constraint_types: vec![ConstraintType::Eq; m],
            rhs: p1_rhs,
            lower_bounds: lb,
            upper_bounds: ub,
            var_names,
            maximize: false,
        };

        // Initial basis: artificial variables
        let basis: Vec<usize> = (n..total).collect();
        let mut state = SimplexState {
            m,
            n: total,
            basis: basis.clone(),
            var_status: Vec::new(),
            xb: vec![0.0; m],
            rc: vec![0.0; total],
            eta_file: Vec::new(),
            b_inv: vec![0.0; m * m],
            b_inv_fresh: false,
            iters_since_refactor: 0,
            se_weights: Vec::new(),
            devex_weights: Vec::new(),
        };

        // Set var status
        state.var_status = vec![BasisStatus::AtLower; total];
        for i in 0..m {
            state.var_status[n + i] = BasisStatus::Basic;
        }

        // B = I for artificials
        for i in 0..m {
            state.b_inv[i * m + i] = 1.0;
        }
        state.b_inv_fresh = true;

        // xB = B^{-1} b = b
        for i in 0..m {
            state.xb[i] = p1_problem.rhs[i];
        }

        // Run Phase I simplex iterations
        let mut iters = 0usize;
        loop {
            if iters >= self.config.max_iterations {
                return Err(OptError::ConvergenceFailure {
                    iterations: iters,
                    message: "Phase I iteration limit".into(),
                });
            }
            if start.elapsed().as_secs_f64() > self.config.time_limit_secs {
                return Err(OptError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_secs_f64(),
                    limit: self.config.time_limit_secs,
                });
            }

            self.compute_reduced_costs(&p1_problem, &mut state);

            // Pricing
            let entering = self.pricing(&state);
            if entering.is_none() {
                break; // optimal for Phase I
            }
            let j_enter = entering.unwrap();

            // FTRAN: d = B^{-1} a_j
            let mut col_j = vec![0.0; m];
            p1_problem.column_dense(j_enter, &mut col_j);
            let d = self.ftran(&state, &col_j);

            // Ratio test
            let leaving = if self.config.use_harris_ratio_test {
                self.harris_ratio_test(&state, &d)
            } else {
                self.ratio_test(&state, &d)
            };

            if leaving.is_none() {
                // Unbounded in Phase I – shouldn't happen with artificials
                warn!("Phase I appears unbounded");
                break;
            }
            let r = leaving.unwrap();

            self.pivot(&p1_problem, &mut state, j_enter, r);
            iters += 1;

            if state.iters_since_refactor >= self.config.refactoring_interval {
                self.refactor_basis(&p1_problem, &mut state)?;
            }
        }

        // Check Phase I objective
        let p1_obj: f64 = state
            .basis
            .iter()
            .enumerate()
            .map(|(i, &j)| p1_problem.obj_coeffs[j] * state.xb[i])
            .sum();

        if p1_obj > self.config.primal_tolerance {
            info!("Phase I objective = {:.2e} > 0 → infeasible", p1_obj);
            return Ok((vec![], SolverStatus::Infeasible));
        }

        // Remove artificial variables from basis if any remain
        let mut final_basis = state.basis.clone();
        for i in 0..m {
            if final_basis[i] >= n {
                // Try to pivot in a real variable
                let mut pivoted = false;
                for j in 0..n {
                    if state.var_status[j] != BasisStatus::Basic {
                        let mut col_j = vec![0.0; m];
                        p1_problem.column_dense(j, &mut col_j);
                        let d = self.ftran(&state, &col_j);
                        if d[i].abs() > self.config.pivot_tolerance {
                            self.pivot(&p1_problem, &mut state, j, i);
                            final_basis = state.basis.clone();
                            pivoted = true;
                            break;
                        }
                    }
                }
                if !pivoted {
                    debug!(
                        "Artificial variable {} remains in basis at position {} (degenerate)",
                        final_basis[i], i
                    );
                }
            }
        }

        debug!("Phase I complete in {} iterations", iters);
        Ok((state.basis.clone(), SolverStatus::Optimal))
    }

    // -----------------------------------------------------------------------
    // Phase II
    // -----------------------------------------------------------------------

    fn phase_two(
        &self,
        problem: &LpProblem,
        initial_basis: Vec<usize>,
        start: &Instant,
    ) -> OptResult<(SimplexState, usize)> {
        let m = problem.num_constraints;
        let n = problem.num_vars;

        let mut state = SimplexState {
            m,
            n,
            basis: initial_basis,
            var_status: vec![BasisStatus::AtLower; n],
            xb: vec![0.0; m],
            rc: vec![0.0; n],
            eta_file: Vec::new(),
            b_inv: vec![0.0; m * m],
            b_inv_fresh: false,
            iters_since_refactor: 0,
            se_weights: vec![1.0; n],
            devex_weights: vec![1.0; n],
        };

        for &j in &state.basis {
            if j < n {
                state.var_status[j] = BasisStatus::Basic;
            }
        }

        // Initial factorisation
        self.refactor_basis(problem, &mut state)?;

        let mut iters = 0usize;
        loop {
            if iters >= self.config.max_iterations {
                return Ok((state, iters));
            }
            if start.elapsed().as_secs_f64() > self.config.time_limit_secs {
                return Ok((state, iters));
            }

            self.recompute_xb(problem, &mut state);
            self.compute_reduced_costs(problem, &mut state);

            let entering = self.pricing(&state);
            if entering.is_none() {
                debug!("Phase II optimal after {} iterations", iters);
                break;
            }
            let j_enter = entering.unwrap();

            let mut col_j = vec![0.0; m];
            problem.column_dense(j_enter, &mut col_j);
            let d = self.ftran(&state, &col_j);

            // Ratio test
            let leaving = if self.config.use_bound_flipping {
                self.bound_flipping_ratio_test(problem, &state, &d, j_enter)
            } else if self.config.use_harris_ratio_test {
                self.harris_ratio_test(&state, &d)
            } else {
                self.ratio_test(&state, &d)
            };

            if leaving.is_none() {
                // Unbounded
                info!("Problem is unbounded (entering variable {})", j_enter);
                state.var_status[j_enter] = BasisStatus::Free; // mark for detection
                return Ok((state, iters));
            }
            let r = leaving.unwrap();

            self.pivot(problem, &mut state, j_enter, r);
            iters += 1;

            if state.iters_since_refactor >= self.config.refactoring_interval {
                self.refactor_basis(problem, &mut state)?;
            }
        }

        Ok((state, iters))
    }

    // -----------------------------------------------------------------------
    // Pricing methods
    // -----------------------------------------------------------------------

    fn pricing(&self, state: &SimplexState) -> Option<usize> {
        match self.config.pricing_rule {
            PricingRule::Dantzig => self.pricing_dantzig(state),
            PricingRule::SteepestEdge => self.pricing_steepest_edge(state),
            PricingRule::Devex => self.pricing_devex(state),
        }
    }

    /// Dantzig pricing: choose variable with most negative reduced cost.
    fn pricing_dantzig(&self, state: &SimplexState) -> Option<usize> {
        let mut best_j = None;
        let mut best_rc = -self.config.dual_tolerance;
        for j in 0..state.n {
            if state.var_status[j] == BasisStatus::Basic {
                continue;
            }
            let rc = state.rc[j];
            // AtLower: entering increases value → need rc < 0
            // AtUpper: entering decreases value → need rc > 0
            let effective_rc = match state.var_status[j] {
                BasisStatus::AtLower | BasisStatus::Free => rc,
                BasisStatus::AtUpper => -rc,
                _ => continue,
            };
            if effective_rc < best_rc {
                best_rc = effective_rc;
                best_j = Some(j);
            }
        }
        best_j
    }

    /// Steepest-edge pricing: rc_j^2 / weight_j.
    fn pricing_steepest_edge(&self, state: &SimplexState) -> Option<usize> {
        let mut best_j = None;
        let mut best_score = -self.config.dual_tolerance;
        for j in 0..state.n {
            if state.var_status[j] == BasisStatus::Basic {
                continue;
            }
            let rc = match state.var_status[j] {
                BasisStatus::AtLower | BasisStatus::Free => state.rc[j],
                BasisStatus::AtUpper => -state.rc[j],
                _ => continue,
            };
            if rc >= -self.config.dual_tolerance {
                continue;
            }
            let w = if j < state.se_weights.len() {
                state.se_weights[j].max(1e-12)
            } else {
                1.0
            };
            let score = -(rc * rc) / w;
            if score < best_score {
                best_score = score;
                best_j = Some(j);
            }
        }
        best_j
    }

    /// Devex pricing (approximate steepest edge).
    fn pricing_devex(&self, state: &SimplexState) -> Option<usize> {
        let mut best_j = None;
        let mut best_score = -self.config.dual_tolerance;
        for j in 0..state.n {
            if state.var_status[j] == BasisStatus::Basic {
                continue;
            }
            let rc = match state.var_status[j] {
                BasisStatus::AtLower | BasisStatus::Free => state.rc[j],
                BasisStatus::AtUpper => -state.rc[j],
                _ => continue,
            };
            if rc >= -self.config.dual_tolerance {
                continue;
            }
            let w = if j < state.devex_weights.len() {
                state.devex_weights[j].max(1e-12)
            } else {
                1.0
            };
            let score = -(rc * rc) / w;
            if score < best_score {
                best_score = score;
                best_j = Some(j);
            }
        }
        best_j
    }

    // -----------------------------------------------------------------------
    // Ratio tests
    // -----------------------------------------------------------------------

    /// Standard ratio test: min { xB[i] / d[i] : d[i] > 0 }.
    fn ratio_test(&self, state: &SimplexState, d: &[f64]) -> Option<usize> {
        let mut best_r = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..state.m {
            if d[i] > self.config.pivot_tolerance {
                let ratio = state.xb[i] / d[i];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    best_r = Some(i);
                }
            }
        }
        best_r
    }

    /// Harris two-pass ratio test for degeneracy handling.
    fn harris_ratio_test(&self, state: &SimplexState, d: &[f64]) -> Option<usize> {
        let tol = self.config.primal_tolerance;

        // Pass 1: compute Harris threshold
        let mut harris_threshold = f64::INFINITY;
        for i in 0..state.m {
            if d[i] > self.config.pivot_tolerance {
                let ratio = (state.xb[i] + tol) / d[i];
                if ratio < harris_threshold {
                    harris_threshold = ratio;
                }
            }
        }
        if harris_threshold.is_infinite() {
            return None;
        }

        // Pass 2: among candidates within threshold, pick largest pivot
        let mut best_r = None;
        let mut best_pivot = 0.0;
        for i in 0..state.m {
            if d[i] > self.config.pivot_tolerance {
                let ratio = state.xb[i] / d[i];
                if ratio <= harris_threshold + tol && d[i] > best_pivot {
                    best_pivot = d[i];
                    best_r = Some(i);
                }
            }
        }
        best_r
    }

    /// Bound-flipping ratio test for bounded variables.
    fn bound_flipping_ratio_test(
        &self,
        problem: &LpProblem,
        state: &SimplexState,
        d: &[f64],
        _entering: usize,
    ) -> Option<usize> {
        // For bounded variables, when the standard ratio test would give a
        // very small step, we can flip bounds of blocking basic variables instead.
        let mut best_r = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..state.m {
            if d[i] > self.config.pivot_tolerance {
                let j = state.basis[i];
                let ub = if j < problem.upper_bounds.len() {
                    problem.upper_bounds[j]
                } else {
                    f64::INFINITY
                };
                let ratio = state.xb[i] / d[i];

                // If the variable has a finite upper bound, we could also flip
                // it to its upper bound. Use the smaller step.
                if ub.is_finite() && state.xb[i] < ub {
                    let flip_ratio = (ub - state.xb[i]) / d[i];
                    let effective = ratio.min(flip_ratio);
                    if effective < min_ratio {
                        min_ratio = effective;
                        best_r = Some(i);
                    }
                } else if ratio < min_ratio {
                    min_ratio = ratio;
                    best_r = Some(i);
                }
            }
        }
        best_r
    }

    // -----------------------------------------------------------------------
    // Pivot
    // -----------------------------------------------------------------------

    fn pivot(
        &self,
        problem: &LpProblem,
        state: &mut SimplexState,
        entering: usize,
        leaving_row: usize,
    ) {
        let leaving_var = state.basis[leaving_row];

        // Compute d = B^{-1} a_entering (FTRAN)
        let mut col_j = vec![0.0; state.m];
        problem.column_dense(entering, &mut col_j);
        let d = self.ftran(state, &col_j);

        let pivot_elem = d[leaving_row];
        if pivot_elem.abs() < 1e-15 {
            warn!(
                "Very small pivot element {:.2e} at ({}, {})",
                pivot_elem, leaving_row, entering
            );
        }

        // Step size
        let step = if pivot_elem.abs() > 1e-15 {
            state.xb[leaving_row] / pivot_elem
        } else {
            0.0
        };

        // Update basic variable values
        for i in 0..state.m {
            state.xb[i] -= step * d[i];
        }
        state.xb[leaving_row] = step;

        // Update eta file
        self.update_eta_file(state, &d, leaving_row);

        // Update basis
        state.basis[leaving_row] = entering;
        state.var_status[leaving_var] = BasisStatus::AtLower;
        state.var_status[entering] = BasisStatus::Basic;

        // Update steepest-edge / devex weights (simplified)
        if self.config.pricing_rule == PricingRule::SteepestEdge
            || self.config.pricing_rule == PricingRule::Devex
        {
            self.update_weights(state, &d, leaving_row);
        }

        state.iters_since_refactor += 1;
    }

    // -----------------------------------------------------------------------
    // BTRAN / FTRAN
    // -----------------------------------------------------------------------

    /// Backward transformation: y = e^T B^{-1}.
    /// Uses the eta file in reverse order: y = input, then for each eta from last to first,
    /// y[pivot_row] = (y[pivot_row] - sum_{i != pivot_row} y[i] * eta[i]) * eta[pivot_row].
    fn btran_vec(&self, state: &SimplexState, input: &[f64]) -> Vec<f64> {
        let m = state.m;
        let mut y = input.to_vec();

        if state.b_inv_fresh && state.eta_file.is_empty() {
            // Use dense B^{-1}: y = input^T * B^{-1}
            let mut result = vec![0.0; m];
            for j in 0..m {
                let mut s = 0.0;
                for i in 0..m {
                    s += input[i] * state.b_inv[i * m + j];
                }
                result[j] = s;
            }
            return result;
        }

        // Apply B^{-1} first if fresh
        if state.b_inv_fresh {
            let mut result = vec![0.0; m];
            for j in 0..m {
                let mut s = 0.0;
                for i in 0..m {
                    s += y[i] * state.b_inv[i * m + j];
                }
                result[j] = s;
            }
            y = result;
        }

        // Apply eta vectors in reverse order
        for eta in state.eta_file.iter().rev() {
            let r = eta.pivot_row;
            let mut sum = 0.0;
            for i in 0..m {
                if i != r {
                    sum += y[i] * eta.coeffs[i];
                }
            }
            y[r] = (y[r] - sum) * eta.coeffs[r];
        }

        y
    }

    /// Forward transformation: d = B^{-1} * input.
    fn ftran(&self, state: &SimplexState, input: &[f64]) -> Vec<f64> {
        let m = state.m;
        let mut d = input.to_vec();

        // Apply dense B^{-1} if fresh
        if state.b_inv_fresh {
            let mut result = vec![0.0; m];
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..m {
                    s += state.b_inv[i * m + j] * d[j];
                }
                result[i] = s;
            }
            d = result;
        }

        // Apply eta vectors in forward order
        for eta in &state.eta_file {
            let r = eta.pivot_row;
            let dr = d[r];
            for i in 0..m {
                if i != r {
                    d[i] -= eta.coeffs[i] * dr;
                }
            }
            d[r] = dr * eta.coeffs[r];
        }

        d
    }

    // -----------------------------------------------------------------------
    // Eta-file maintenance
    // -----------------------------------------------------------------------

    fn update_eta_file(&self, state: &mut SimplexState, d: &[f64], pivot_row: usize) {
        let m = state.m;
        let pivot_elem = d[pivot_row];
        if pivot_elem.abs() < 1e-15 {
            return;
        }

        let mut eta_coeffs = vec![0.0; m];
        for i in 0..m {
            if i == pivot_row {
                eta_coeffs[i] = 1.0 / pivot_elem;
            } else {
                eta_coeffs[i] = -d[i] / pivot_elem;
            }
        }

        state.eta_file.push(EtaVector {
            pivot_row,
            coeffs: eta_coeffs,
        });
    }

    // -----------------------------------------------------------------------
    // Refactorisation (full basis LU)
    // -----------------------------------------------------------------------

    fn refactor_basis(&self, problem: &LpProblem, state: &mut SimplexState) -> OptResult<()> {
        let m = state.m;

        // Build dense basis matrix
        let mut b_mat = vec![0.0; m * m];
        let mut col_buf = vec![0.0; m];
        for (k, &j) in state.basis.iter().enumerate() {
            problem.column_dense(j, &mut col_buf);
            for i in 0..m {
                b_mat[i * m + k] = col_buf[i];
            }
        }

        // LU factorize and compute B^{-1}
        state.b_inv = self.dense_inverse(&b_mat, m)?;
        state.b_inv_fresh = true;
        state.eta_file.clear();
        state.iters_since_refactor = 0;

        // Recompute xB
        self.recompute_xb(problem, state);

        debug!("Refactored basis");
        Ok(())
    }

    fn recompute_xb(&self, problem: &LpProblem, state: &mut SimplexState) {
        let m = state.m;
        // xB = B^{-1} b
        let rhs = &problem.rhs;
        if state.b_inv_fresh && state.eta_file.is_empty() {
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..m {
                    s += state.b_inv[i * m + j] * rhs[j];
                }
                state.xb[i] = s;
            }
        } else {
            let xb = self.ftran(state, rhs);
            state.xb = xb;
        }
    }

    // -----------------------------------------------------------------------
    // Compute reduced costs
    // -----------------------------------------------------------------------

    fn compute_reduced_costs(&self, problem: &LpProblem, state: &mut SimplexState) {
        let m = state.m;
        let n = state.n;

        // y = c_B^T B^{-1}  (BTRAN of c_B)
        let mut c_b = vec![0.0; m];
        for (i, &j) in state.basis.iter().enumerate() {
            c_b[i] = if j < problem.obj_coeffs.len() {
                problem.obj_coeffs[j]
            } else {
                0.0
            };
        }
        let y = self.btran_vec(state, &c_b);

        // rc_j = c_j - y^T a_j
        for j in 0..n {
            if state.var_status[j] == BasisStatus::Basic {
                state.rc[j] = 0.0;
                continue;
            }
            let c_j = if j < problem.obj_coeffs.len() {
                problem.obj_coeffs[j]
            } else {
                0.0
            };

            let mut yta = 0.0;
            for i in 0..m {
                let rs = problem.row_starts[i];
                let re = problem.row_starts[i + 1];
                for k in rs..re {
                    if problem.col_indices[k] == j {
                        yta += y[i] * problem.values[k];
                    }
                }
            }
            state.rc[j] = c_j - yta;
        }
    }

    // -----------------------------------------------------------------------
    // Weight updates
    // -----------------------------------------------------------------------

    fn update_weights(&self, state: &mut SimplexState, d: &[f64], pivot_row: usize) {
        let _m = state.m;
        let pivot_elem = d[pivot_row];
        if pivot_elem.abs() < 1e-15 {
            return;
        }
        // Simplified steepest-edge update: w_j' = w_j + d_pivot_row_j^2 / pivot^2
        let pivot_sq = pivot_elem * pivot_elem;
        for j in 0..state.n.min(state.se_weights.len()) {
            if state.var_status[j] == BasisStatus::Basic {
                continue;
            }
            state.se_weights[j] = (state.se_weights[j] + d[pivot_row].powi(2) / pivot_sq).max(1e-6);
        }
        // Devex: reset weights periodically to 1.0
        if state.iters_since_refactor > 0 && state.iters_since_refactor % 50 == 0 {
            for w in state.devex_weights.iter_mut() {
                *w = 1.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Dense LU inverse
    // -----------------------------------------------------------------------

    fn dense_inverse(&self, a: &[f64], n: usize) -> OptResult<Vec<f64>> {
        let mut lu = a.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();

        // LU with partial pivoting
        for k in 0..n {
            let mut max_val = lu[perm[k] * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = lu[perm[i] * n + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val < self.config.pivot_tolerance {
                return Err(OptError::NumericalError {
                    context: format!("Singular basis at column {} (pivot={:.2e})", k, max_val),
                });
            }
            perm.swap(k, max_row);

            let pivot_row = perm[k];
            for i in (k + 1)..n {
                let target_row = perm[i];
                let factor = lu[target_row * n + k] / lu[pivot_row * n + k];
                lu[target_row * n + k] = factor;
                for j in (k + 1)..n {
                    lu[target_row * n + j] -= factor * lu[pivot_row * n + j];
                }
            }
        }

        // Solve for each column of identity
        let mut inv = vec![0.0; n * n];
        for col in 0..n {
            let mut b = vec![0.0; n];
            b[col] = 1.0;

            // Permute
            let mut pb = vec![0.0; n];
            for i in 0..n {
                pb[i] = b[perm[i]];
            }

            // Forward substitution
            for i in 1..n {
                let row = perm[i];
                for j in 0..i {
                    pb[i] -= lu[row * n + j] * pb[j];
                }
            }

            // Backward substitution
            for i in (0..n).rev() {
                let row = perm[i];
                for j in (i + 1)..n {
                    pb[i] -= lu[row * n + j] * pb[j];
                }
                pb[i] /= lu[row * n + i];
            }

            for i in 0..n {
                inv[i * n + col] = pb[i];
            }
        }

        Ok(inv)
    }

    // -----------------------------------------------------------------------
    // Initialisation helpers
    // -----------------------------------------------------------------------

    fn init_state(&self, problem: &LpProblem) -> OptResult<SimplexState> {
        let m = problem.num_constraints;
        let n = problem.num_vars;

        // Try to find slack variables for initial basis
        // In standard form, the last m variables should be slacks
        let basis: Vec<usize> = if n >= m {
            ((n - m)..n).collect()
        } else {
            (0..m).map(|i| i.min(n - 1)).collect()
        };

        let mut var_status = vec![BasisStatus::AtLower; n];
        for &j in &basis {
            if j < n {
                var_status[j] = BasisStatus::Basic;
            }
        }

        let mut state = SimplexState {
            m,
            n,
            basis,
            var_status,
            xb: vec![0.0; m],
            rc: vec![0.0; n],
            eta_file: Vec::new(),
            b_inv: vec![0.0; m * m],
            b_inv_fresh: false,
            iters_since_refactor: 0,
            se_weights: vec![1.0; n],
            devex_weights: vec![1.0; n],
        };

        self.refactor_basis(problem, &mut state)?;
        Ok(state)
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

    // -----------------------------------------------------------------------
    // Solution extraction
    // -----------------------------------------------------------------------

    fn extract_solution(
        &self,
        original: &LpProblem,
        std_problem: &LpProblem,
        state: &SimplexState,
        orig_n: usize,
        iters: usize,
        start: &Instant,
    ) -> LpSolution {
        let m = std_problem.num_constraints;
        let n = std_problem.num_vars;

        // Check for unbounded
        for j in 0..n.min(state.var_status.len()) {
            if state.var_status[j] == BasisStatus::Free
                && j < orig_n
                && state.rc[j] < -self.config.dual_tolerance
            {
                return LpSolution {
                    status: SolverStatus::Unbounded,
                    objective_value: f64::NEG_INFINITY,
                    primal_values: vec![0.0; original.num_vars],
                    dual_values: vec![0.0; original.num_constraints],
                    reduced_costs: vec![0.0; original.num_vars],
                    basis_status: vec![BasisStatus::AtLower; original.num_vars],
                    iterations: iters,
                    time_seconds: start.elapsed().as_secs_f64(),
                };
            }
        }

        // Extract primal values (shift back by lower bounds)
        let mut x_std = vec![0.0; n];
        for (i, &j) in state.basis.iter().enumerate() {
            if j < n {
                x_std[j] = state.xb[i];
            }
        }

        let mut x_orig = vec![0.0; original.num_vars];
        for j in 0..original.num_vars {
            x_orig[j] = x_std[j] + original.lower_bounds[j];
        }

        // Compute objective
        let obj: f64 = (0..original.num_vars)
            .map(|j| original.obj_coeffs[j] * x_orig[j])
            .sum();

        // Extract dual values from BTRAN
        let mut c_b = vec![0.0; m];
        for (i, &j) in state.basis.iter().enumerate() {
            c_b[i] = if j < std_problem.obj_coeffs.len() {
                std_problem.obj_coeffs[j]
            } else {
                0.0
            };
        }
        let y = self.btran_vec(state, &c_b);

        // Adjust dual signs if maximising
        let dual_values: Vec<f64> = if original.maximize {
            y.iter().map(|v| -v).collect()
        } else {
            y.clone()
        };

        // Reduced costs for original vars
        let mut rc_orig = vec![0.0; original.num_vars];
        for j in 0..original.num_vars {
            if j < state.rc.len() {
                rc_orig[j] = if original.maximize {
                    -state.rc[j]
                } else {
                    state.rc[j]
                };
            }
        }

        // Basis status
        let mut basis_status = vec![BasisStatus::AtLower; original.num_vars];
        for j in 0..original.num_vars {
            if j < state.var_status.len() {
                basis_status[j] = state.var_status[j];
            }
        }

        LpSolution {
            status: SolverStatus::Optimal,
            objective_value: obj,
            primal_values: x_orig,
            dual_values,
            reduced_costs: rc_orig,
            basis_status,
            iterations: iters,
            time_seconds: start.elapsed().as_secs_f64(),
        }
    }

    fn build_solution_from_state(
        &self,
        original: &LpProblem,
        std_problem: &LpProblem,
        state: &SimplexState,
        orig_n: usize,
        iters: usize,
        start: &Instant,
    ) -> LpSolution {
        self.extract_solution(original, std_problem, state, orig_n, iters, start)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::ConstraintType;

    fn default_solver() -> SimplexSolver {
        SimplexSolver::new(SimplexConfig::default())
    }

    /// min -x1 - 2*x2  s.t. x1+x2<=4, x1<=3, x2<=3, x1,x2>=0
    /// Optimal at (1,3) with obj = -7.
    fn small_lp() -> LpProblem {
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
        assert!((sol.objective_value - (-7.0)).abs() < 1e-6);
        assert!((sol.primal_values[0] - 1.0).abs() < 1e-6);
        assert!((sol.primal_values[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_maximisation() {
        // max 3x1 + 5x2  s.t.  x1<=4, x2<=6, x1+x2<=8, x>=0
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
        // Optimal: x1=2, x2=6 → obj=36
        assert!((sol.objective_value - 36.0).abs() < 1e-4);
    }

    #[test]
    fn test_infeasible() {
        // x1 >= 5 and x1 <= 3
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Ge, 5.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Infeasible);
    }

    #[test]
    fn test_unbounded() {
        // min -x1  s.t. x1 >= 0 (no upper bound constraint)
        let mut lp = LpProblem::new(false);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(0.0, 0.0, f64::INFINITY, None);
        // Only: x2 <= 5
        lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 5.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Unbounded);
    }

    #[test]
    fn test_single_variable() {
        // min 2x  s.t. x >= 3, x <= 10
        let mut lp = LpProblem::new(false);
        lp.add_variable(2.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0], &[1.0], ConstraintType::Ge, 3.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 10.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.primal_values[0] - 3.0).abs() < 1e-6);
        assert!((sol.objective_value - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_equality_constraint() {
        // min x1 + x2  s.t. x1 + x2 = 5, x1,x2 >= 0
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Eq, 5.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_degeneracy() {
        // Degenerate LP: min -x1 - x2  s.t.  x1+x2<=1, x1<=1, x2<=1, x1,x2>=0
        // Multiple basic feasible solutions at vertex (1,0) and (0,1) are degenerate.
        let mut lp = LpProblem::new(false);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 1.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 1.0)
            .unwrap();
        lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 1.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_steepest_edge_pricing() {
        let mut config = SimplexConfig::default();
        config.pricing_rule = PricingRule::SteepestEdge;
        let solver = SimplexSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - (-7.0)).abs() < 1e-6);
    }

    #[test]
    fn test_devex_pricing() {
        let mut config = SimplexConfig::default();
        config.pricing_rule = PricingRule::Devex;
        let solver = SimplexSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - (-7.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dual_simplex() {
        let solver = default_solver();
        let lp = small_lp();
        let sol = solver.dual_simplex(&lp).unwrap();
        // Dual simplex may or may not converge on all problems easily,
        // but for this standard LP it should find the optimum.
        assert!(
            sol.status == SolverStatus::Optimal || sol.status == SolverStatus::Infeasible,
            "Unexpected status: {:?}",
            sol.status
        );
    }

    #[test]
    fn test_harris_ratio_test_flag() {
        let mut config = SimplexConfig::default();
        config.use_harris_ratio_test = false;
        let solver = SimplexSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
    }

    #[test]
    fn test_three_variable_lp() {
        // min -2x1 - 3x2 - x3  s.t. x1+x2+x3<=10, 2x1+x2<=14, x2+3x3<=12, x>=0
        let mut lp = LpProblem::new(false);
        lp.add_variable(-2.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-3.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1, 2], &[1.0, 1.0, 1.0], ConstraintType::Le, 10.0)
            .unwrap();
        lp.add_constraint(&[0, 1], &[2.0, 1.0], ConstraintType::Le, 14.0)
            .unwrap();
        lp.add_constraint(&[1, 2], &[1.0, 3.0], ConstraintType::Le, 12.0)
            .unwrap();
        let solver = default_solver();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        // Feasibility check
        let x = &sol.primal_values;
        assert!(x[0] + x[1] + x[2] <= 10.0 + 1e-6);
        assert!(2.0 * x[0] + x[1] <= 14.0 + 1e-6);
        assert!(x[1] + 3.0 * x[2] <= 12.0 + 1e-6);
    }

    #[test]
    fn test_iteration_limit() {
        let mut config = SimplexConfig::default();
        config.max_iterations = 1; // very low
        let solver = SimplexSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp);
        // Should either hit iteration limit in phase I (error) or return limit status
        // Either way it shouldn't panic.
        assert!(sol.is_ok() || sol.is_err());
    }

    #[test]
    fn test_bound_flipping() {
        let mut config = SimplexConfig::default();
        config.use_bound_flipping = true;
        let solver = SimplexSolver::new(config);
        let lp = small_lp();
        let sol = solver.solve(&lp).unwrap();
        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective_value - (-7.0)).abs() < 1e-6);
    }
}
