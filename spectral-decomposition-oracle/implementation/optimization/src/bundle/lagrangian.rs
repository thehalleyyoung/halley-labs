//! Lagrangian relaxation framework.
//!
//! Given an LP with complicating constraints, this module dualises a selected
//! subset of constraints (the *relaxed* constraints) and solves the resulting
//! Lagrangian dual via bundle or subgradient methods. The LP is optionally
//! decomposed into independent block subproblems so that each evaluation of
//! the Lagrangian function is cheap.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{OptError, OptResult};
use crate::lp::{ConstraintType, LpProblem};
use crate::bundle::{
    BundleConfig, BundleMethod, LagrangianResult, SubgradientInfo,
};
use crate::bundle::subgradient::{SubgradientConfig, SubgradientSolver, SubgradientStepRule};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which dual method to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LagrangianMethod {
    Bundle,
    Subgradient,
    Volume,
}

/// Configuration for the Lagrangian relaxation solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianConfig {
    pub max_iterations: usize,
    pub gap_tolerance: f64,
    pub time_limit: f64,
    pub method: LagrangianMethod,
    pub heuristic_frequency: usize,
    pub initial_multipliers: Option<Vec<f64>>,
    pub verbose: bool,
}

impl Default for LagrangianConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            gap_tolerance: 1e-6,
            time_limit: 3600.0,
            method: LagrangianMethod::Bundle,
            heuristic_frequency: 10,
            initial_multipliers: None,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Problem description
// ---------------------------------------------------------------------------

/// Describes which constraints to relax and how to partition variables into
/// independent blocks.
#[derive(Debug, Clone)]
pub struct LagrangianProblem {
    /// The original LP.
    pub original: LpProblem,
    /// Indices of constraints to relax (dualise).
    pub relaxed_constraints: Vec<usize>,
    /// Number of independent blocks after relaxation.
    pub num_blocks: usize,
    /// Block assignment for each variable (0-indexed).
    pub variable_partition: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Lagrangian relaxation solver.
#[allow(dead_code)]
pub struct LagrangianRelaxation {
    problem: LagrangianProblem,
    config: LagrangianConfig,
    /// Which constraints are *not* relaxed (kept in the subproblems).
    kept_constraints: Vec<usize>,
    /// Best primal feasible solution found.
    best_primal: Option<Vec<f64>>,
    best_primal_value: f64,
    /// Best dual bound.
    best_dual_bound: f64,
}

impl LagrangianRelaxation {
    /// Build a new Lagrangian relaxation solver.
    pub fn new(problem: LagrangianProblem, config: LagrangianConfig) -> OptResult<Self> {
        // Validate.
        let m = problem.original.num_constraints;
        let n = problem.original.num_vars;

        for &r in &problem.relaxed_constraints {
            if r >= m {
                return Err(OptError::InvalidProblem {
                    reason: format!(
                        "relaxed constraint index {} out of range (problem has {} constraints)",
                        r, m
                    ),
                });
            }
        }
        if problem.variable_partition.len() != n {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "variable_partition length {} != num_vars {}",
                    problem.variable_partition.len(),
                    n
                ),
            });
        }
        for &b in &problem.variable_partition {
            if b >= problem.num_blocks {
                return Err(OptError::InvalidProblem {
                    reason: format!(
                        "block index {} >= num_blocks {}",
                        b, problem.num_blocks
                    ),
                });
            }
        }

        // Determine kept constraints.
        let relaxed_set: std::collections::HashSet<usize> =
            problem.relaxed_constraints.iter().copied().collect();
        let kept_constraints: Vec<usize> = (0..m).filter(|i| !relaxed_set.contains(i)).collect();

        Ok(Self {
            problem,
            config,
            kept_constraints,
            best_primal: None,
            best_primal_value: f64::INFINITY,
            best_dual_bound: f64::NEG_INFINITY,
        })
    }

    // -----------------------------------------------------------------------
    // Main solve
    // -----------------------------------------------------------------------

    /// Solve the Lagrangian dual.
    pub fn solve(&mut self) -> OptResult<LagrangianResult> {
        let num_relaxed = self.problem.relaxed_constraints.len();
        if num_relaxed == 0 {
            return Err(OptError::InvalidProblem {
                reason: "no constraints selected for relaxation".into(),
            });
        }

        let start = Instant::now();

        // Initial multipliers.
        let init_mult = match &self.config.initial_multipliers {
            Some(m) => {
                if m.len() != num_relaxed {
                    return Err(OptError::InvalidProblem {
                        reason: format!(
                            "initial_multipliers length {} != relaxed constraints count {}",
                            m.len(),
                            num_relaxed
                        ),
                    });
                }
                m.clone()
            }
            None => vec![0.0; num_relaxed],
        };

        // Choose a solving method and run.
        // We build a closure oracle that wraps evaluate_lagrangian.
        // Because we need &mut self inside the closure and also call self.solve,
        // we pull out the pieces needed by the oracle into local state.

        let result = match self.config.method {
            LagrangianMethod::Bundle => self.solve_with_bundle(&init_mult, &start)?,
            LagrangianMethod::Subgradient => self.solve_with_subgradient(&init_mult, &start)?,
            LagrangianMethod::Volume => self.solve_with_volume(&init_mult, &start)?,
        };

        Ok(result)
    }

    /// Solve using the proximal bundle method.
    fn solve_with_bundle(
        &mut self,
        init_mult: &[f64],
        _start: &Instant,
    ) -> OptResult<LagrangianResult> {
        let num_relaxed = self.problem.relaxed_constraints.len();

        let bundle_cfg = BundleConfig {
            max_iterations: self.config.max_iterations,
            gap_tolerance: self.config.gap_tolerance,
            time_limit: self.config.time_limit,
            verbose: self.config.verbose,
            ..BundleConfig::default()
        };
        let mut bm = BundleMethod::new(bundle_cfg, num_relaxed);

        // Evaluate at initial multipliers for warm-start.
        let init_info = self.evaluate_lagrangian(init_mult)?;
        bm.warm_start(init_mult.to_vec(), init_info.value);

        let _heuristic_freq = self.config.heuristic_frequency;
        let mut _iteration_counter = 0usize;

        // The bundle method minimises, but the Lagrangian dual is a *max* problem.
        // L(λ) = min_x { c^Tx + λ^T(Ax − b) }  is concave in λ.
        // We negate: minimise −L(λ), which is convex.
        let mut oracle = |multipliers: &[f64]| -> OptResult<SubgradientInfo> {
            let info = self.evaluate_lagrangian(multipliers)?;
            _iteration_counter += 1;

            // Store subproblem solutions for primal recovery.
            // Extract from info.point (we stash solutions via the oracle).
            // Actually, the oracle returns the subgradient and value;
            // subproblem solutions are captured inside evaluate_lagrangian.
            // We re-evaluate to grab them if needed.

            // Negate: we want to minimise −L(λ).
            Ok(SubgradientInfo {
                point: info.point,
                value: -info.value,
                subgradient: info.subgradient.iter().map(|g| -g).collect(),
            })
        };

        let bres = bm.solve(&mut oracle)?;

        // Final evaluation to get subproblem solutions.
        let final_mult = bres.optimal_point.clone();
        let final_info = self.evaluate_lagrangian_with_solutions(&final_mult)?;
        let last_subproblem_solutions = final_info.1;

        let dual_bound = -bres.optimal_value;
        self.best_dual_bound = dual_bound;

        // Primal recovery.
        let primal_estimate = self
            .primal_recovery(&last_subproblem_solutions, &final_mult)
            .unwrap_or_else(|_| vec![0.0; self.problem.original.num_vars]);

        Ok(LagrangianResult {
            dual_bound,
            multipliers: final_mult,
            primal_estimate,
            subproblem_solutions: last_subproblem_solutions,
            iterations: bres.iterations,
            gap: bres.gap,
        })
    }

    /// Solve using the subgradient method.
    fn solve_with_subgradient(
        &mut self,
        init_mult: &[f64],
        _start: &Instant,
    ) -> OptResult<LagrangianResult> {
        let num_relaxed = self.problem.relaxed_constraints.len();

        let sub_cfg = SubgradientConfig {
            max_iterations: self.config.max_iterations,
            step_rule: SubgradientStepRule::Polyak,
            initial_step_size: 2.0,
            best_bound_estimate: 0.0,
            use_averaging: true,
            verbose: self.config.verbose,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(sub_cfg, num_relaxed);
        solver.set_initial_point(init_mult.to_vec());

        let mut oracle = |multipliers: &[f64]| -> OptResult<SubgradientInfo> {
            let info = self.evaluate_lagrangian(multipliers)?;
            // Negate for minimisation.
            Ok(SubgradientInfo {
                point: info.point,
                value: -info.value,
                subgradient: info.subgradient.iter().map(|g| -g).collect(),
            })
        };

        let bres = solver.solve(&mut oracle)?;
        let final_mult = bres.optimal_point.clone();
        let final_info = self.evaluate_lagrangian_with_solutions(&final_mult)?;
        let dual_bound = -bres.optimal_value;
        self.best_dual_bound = dual_bound;

        let primal_estimate = self
            .primal_recovery(&final_info.1, &final_mult)
            .unwrap_or_else(|_| vec![0.0; self.problem.original.num_vars]);

        Ok(LagrangianResult {
            dual_bound,
            multipliers: final_mult,
            primal_estimate,
            subproblem_solutions: final_info.1,
            iterations: bres.iterations,
            gap: bres.gap,
        })
    }

    /// Solve using the volume algorithm variant.
    fn solve_with_volume(
        &mut self,
        init_mult: &[f64],
        _start: &Instant,
    ) -> OptResult<LagrangianResult> {
        let num_relaxed = self.problem.relaxed_constraints.len();

        let sub_cfg = SubgradientConfig {
            max_iterations: self.config.max_iterations,
            step_rule: SubgradientStepRule::Adaptive,
            initial_step_size: 2.0,
            best_bound_estimate: 0.0,
            use_averaging: true,
            verbose: self.config.verbose,
            ..SubgradientConfig::default()
        };
        let mut solver = SubgradientSolver::new(sub_cfg, num_relaxed);
        solver.set_initial_point(init_mult.to_vec());

        let mut oracle = |multipliers: &[f64]| -> OptResult<SubgradientInfo> {
            let info = self.evaluate_lagrangian(multipliers)?;
            Ok(SubgradientInfo {
                point: info.point,
                value: -info.value,
                subgradient: info.subgradient.iter().map(|g| -g).collect(),
            })
        };

        let bres = solver.volume_algorithm(&mut oracle)?;
        let final_mult = bres.optimal_point.clone();
        let final_info = self.evaluate_lagrangian_with_solutions(&final_mult)?;
        let dual_bound = -bres.optimal_value;
        self.best_dual_bound = dual_bound;

        let primal_estimate = self
            .primal_recovery(&final_info.1, &final_mult)
            .unwrap_or_else(|_| vec![0.0; self.problem.original.num_vars]);

        Ok(LagrangianResult {
            dual_bound,
            multipliers: final_mult,
            primal_estimate,
            subproblem_solutions: final_info.1,
            iterations: bres.iterations,
            gap: bres.gap,
        })
    }

    // -----------------------------------------------------------------------
    // Lagrangian evaluation
    // -----------------------------------------------------------------------

    /// Evaluate the Lagrangian function at multipliers λ:
    ///
    ///   L(λ) = min_x { c^Tx + λ^T(Ax − b) }  subject to kept constraints.
    ///
    /// Decompose into block subproblems and return value + subgradient.
    pub fn evaluate_lagrangian(&self, multipliers: &[f64]) -> OptResult<SubgradientInfo> {
        let (info, _solutions) = self.evaluate_lagrangian_impl(multipliers)?;
        Ok(info)
    }

    /// Evaluate and also return per-block solutions.
    fn evaluate_lagrangian_with_solutions(
        &self,
        multipliers: &[f64],
    ) -> OptResult<(SubgradientInfo, Vec<Vec<f64>>)> {
        self.evaluate_lagrangian_impl(multipliers)
    }

    fn evaluate_lagrangian_impl(
        &self,
        multipliers: &[f64],
    ) -> OptResult<(SubgradientInfo, Vec<Vec<f64>>)> {
        let lp = &self.problem.original;
        let n = lp.num_vars;
        let num_relaxed = self.problem.relaxed_constraints.len();
        assert_eq!(multipliers.len(), num_relaxed);

        // Build modified objective: c̃_j = c_j + Σ_i λ_i A_{r_i,j}
        // where r_i are the relaxed constraint indices.
        let mut modified_obj = lp.obj_coeffs.clone();
        if lp.maximize {
            // Convert max c^Tx to min -c^Tx for internal handling.
            for v in modified_obj.iter_mut() {
                *v = -*v;
            }
        }

        for (idx, &ri) in self.problem.relaxed_constraints.iter().enumerate() {
            let rs = lp.row_starts[ri];
            let re = lp.row_starts[ri + 1];
            for k in rs..re {
                let j = lp.col_indices[k];
                let a_ij = lp.values[k];
                modified_obj[j] += multipliers[idx] * a_ij;
            }
        }

        // Constant term: −λ^T b  (for relaxed constraints).
        let mut constant = 0.0;
        for (idx, &ri) in self.problem.relaxed_constraints.iter().enumerate() {
            constant -= multipliers[idx] * lp.rhs[ri];
        }

        // Solve each block subproblem independently.
        let mut total_value = constant;
        let mut full_solution = vec![0.0; n];
        let mut block_solutions = Vec::with_capacity(self.problem.num_blocks);

        for block in 0..self.problem.num_blocks {
            let (block_val, block_sol) = self.solve_subproblem(block, &modified_obj)?;
            total_value += block_val;

            // Write block solution into full solution vector.
            let mut bsol = Vec::new();
            for j in 0..n {
                if self.problem.variable_partition[j] == block {
                    full_solution[j] = block_sol.get(bsol.len()).copied().unwrap_or(0.0);
                    bsol.push(full_solution[j]);
                }
            }
            block_solutions.push(bsol);
        }

        // Compute subgradient: g_idx = (Ax*)_{r_idx} − b_{r_idx}.
        let subgradient = self.compute_subgradient(multipliers, &full_solution);

        Ok((
            SubgradientInfo {
                point: multipliers.to_vec(),
                value: total_value,
                subgradient,
            },
            block_solutions,
        ))
    }

    // -----------------------------------------------------------------------
    // Block subproblem
    // -----------------------------------------------------------------------

    /// Solve the LP subproblem for one block with a modified objective.
    ///
    /// The subproblem includes only variables in `block` and only the kept
    /// (non-relaxed) constraints that reference those variables.
    pub fn solve_subproblem(
        &self,
        block: usize,
        modified_obj: &[f64],
    ) -> OptResult<(f64, Vec<f64>)> {
        let lp = &self.problem.original;
        let n = lp.num_vars;

        // Variables in this block.
        let block_vars: Vec<usize> = (0..n)
            .filter(|&j| self.problem.variable_partition[j] == block)
            .collect();

        if block_vars.is_empty() {
            return Ok((0.0, Vec::new()));
        }

        let block_n = block_vars.len();
        // Mapping from original variable index to block-local index.
        let mut var_to_local = vec![usize::MAX; n];
        for (local, &orig) in block_vars.iter().enumerate() {
            var_to_local[orig] = local;
        }

        // Identify kept constraints that involve block variables.
        let mut relevant_constraints = Vec::new();
        for &ci in &self.kept_constraints {
            let rs = lp.row_starts[ci];
            let re = lp.row_starts[ci + 1];
            let involves_block = (rs..re).any(|k| {
                let j = lp.col_indices[k];
                self.problem.variable_partition[j] == block
            });
            // Only include if all variables in this constraint belong to this block
            // (otherwise it's a linking constraint that was supposed to be relaxed).
            if involves_block {
                let all_in_block = (rs..re)
                    .all(|k| self.problem.variable_partition[lp.col_indices[k]] == block);
                if all_in_block {
                    relevant_constraints.push(ci);
                }
            }
        }

        // For each variable, the bounds give box constraints.
        // We solve a bounded LP:  min c̃^T x  s.t.  local constraints + bounds.
        //
        // Use a simple bounded-variable optimisation:
        // If there are no linking constraints, each variable can be set independently
        // to its bound that minimises the objective (greedy for LP with only bounds).
        // If there are constraints, we use a simple primal feasibility approach.

        if relevant_constraints.is_empty() {
            // Pure box-constrained LP: each variable independently set to its optimal bound.
            let mut sol = vec![0.0; block_n];
            let mut obj_val = 0.0;
            for (local, &orig) in block_vars.iter().enumerate() {
                let c = modified_obj[orig];
                let lb = lp.lower_bounds[orig];
                let ub = lp.upper_bounds[orig];

                // Minimise c * x_j with lb <= x_j <= ub.
                let x = if c > 0.0 {
                    if lb.is_finite() { lb } else { 0.0 }
                } else if c < 0.0 {
                    if ub.is_finite() { ub } else { 0.0 }
                } else {
                    // c == 0: any feasible value; pick lower bound.
                    if lb.is_finite() { lb } else { 0.0 }
                };
                sol[local] = x;
                obj_val += c * x;
            }
            return Ok((obj_val, sol));
        }

        // Build a small LP for this block and solve via a simple bounded-variable
        // approach (iterate over constraints using a projected gradient method).
        let mut sol = vec![0.0; block_n];
        // Initialise at feasible point (lower bounds).
        for (local, &orig) in block_vars.iter().enumerate() {
            let lb = lp.lower_bounds[orig];
            sol[local] = if lb.is_finite() { lb } else { 0.0 };
        }

        // Simple projected gradient descent for the block LP.
        let max_pg_iter = 2000;
        let step = 0.01;

        for _pg_iter in 0..max_pg_iter {
            // Gradient = modified_obj (for the block variables) + penalty for constraint violation.
            let mut grad = vec![0.0; block_n];
            for (local, &orig) in block_vars.iter().enumerate() {
                grad[local] = modified_obj[orig];
            }

            // Add quadratic penalty for constraint violations.
            let penalty_weight = 100.0;
            for &ci in &relevant_constraints {
                let rs = lp.row_starts[ci];
                let re = lp.row_starts[ci + 1];
                // Compute Ax for this constraint.
                let mut ax = 0.0;
                for k in rs..re {
                    let j = lp.col_indices[k];
                    let local = var_to_local[j];
                    if local < block_n {
                        ax += lp.values[k] * sol[local];
                    }
                }
                let rhs_val = lp.rhs[ci];
                let violation = match lp.constraint_types[ci] {
                    ConstraintType::Le => (ax - rhs_val).max(0.0),
                    ConstraintType::Ge => (rhs_val - ax).max(0.0),
                    ConstraintType::Eq => ax - rhs_val,
                };
                if violation.abs() > 1e-12 {
                    for k in rs..re {
                        let j = lp.col_indices[k];
                        let local = var_to_local[j];
                        if local < block_n {
                            grad[local] += penalty_weight * violation * lp.values[k];
                        }
                    }
                }
            }

            // Gradient step with projection onto bounds.
            let mut max_change = 0.0_f64;
            for (local, &orig) in block_vars.iter().enumerate() {
                let lb = lp.lower_bounds[orig];
                let ub = lp.upper_bounds[orig];
                let x_new = sol[local] - step * grad[local];
                let x_proj = x_new
                    .max(if lb.is_finite() { lb } else { x_new - 1.0 })
                    .min(if ub.is_finite() { ub } else { x_new + 1.0 });
                let change = (x_proj - sol[local]).abs();
                if change > max_change {
                    max_change = change;
                }
                sol[local] = x_proj;
            }

            if max_change < 1e-10 {
                break;
            }
        }

        let mut obj_val = 0.0;
        for (local, &orig) in block_vars.iter().enumerate() {
            obj_val += modified_obj[orig] * sol[local];
        }

        Ok((obj_val, sol))
    }

    // -----------------------------------------------------------------------
    // Subgradient computation
    // -----------------------------------------------------------------------

    /// Compute the subgradient of L(λ) at the current multipliers given the
    /// full primal solution x*.
    ///
    /// g_idx = (Ax*)_{r_idx} − b_{r_idx}  for each relaxed constraint r_idx.
    pub fn compute_subgradient(&self, _multipliers: &[f64], primal_solution: &[f64]) -> Vec<f64> {
        let lp = &self.problem.original;
        let num_relaxed = self.problem.relaxed_constraints.len();
        let mut g = vec![0.0; num_relaxed];

        for (idx, &ri) in self.problem.relaxed_constraints.iter().enumerate() {
            let rs = lp.row_starts[ri];
            let re = lp.row_starts[ri + 1];
            let mut ax = 0.0;
            for k in rs..re {
                ax += lp.values[k] * primal_solution[lp.col_indices[k]];
            }
            g[idx] = ax - lp.rhs[ri];
        }

        g
    }

    // -----------------------------------------------------------------------
    // Primal recovery
    // -----------------------------------------------------------------------

    /// Heuristic to recover a feasible primal solution from the block
    /// subproblem solutions.
    ///
    /// Strategy: start from the concatenated subproblem solution, then project
    /// violated relaxed constraints by shifting variables proportionally.
    pub fn primal_recovery(
        &self,
        block_solutions: &[Vec<f64>],
        _multipliers: &[f64],
    ) -> OptResult<Vec<f64>> {
        let lp = &self.problem.original;
        let n = lp.num_vars;

        // Reconstruct full solution from blocks.
        let mut x = vec![0.0; n];
        for block in 0..self.problem.num_blocks {
            let block_sol = block_solutions.get(block).cloned().unwrap_or_default();
            let mut local_idx = 0;
            for j in 0..n {
                if self.problem.variable_partition[j] == block {
                    if local_idx < block_sol.len() {
                        x[j] = block_sol[local_idx];
                    }
                    local_idx += 1;
                }
            }
        }

        // Project onto feasible region for relaxed constraints.
        let max_repair_iters = 50;
        for _repair in 0..max_repair_iters {
            let mut max_violation = 0.0_f64;

            for &ri in &self.problem.relaxed_constraints {
                let rs = lp.row_starts[ri];
                let re = lp.row_starts[ri + 1];
                let mut ax = 0.0;
                for k in rs..re {
                    ax += lp.values[k] * x[lp.col_indices[k]];
                }
                let rhs_val = lp.rhs[ri];

                let violation = match lp.constraint_types[ri] {
                    ConstraintType::Le => ax - rhs_val,
                    ConstraintType::Ge => rhs_val - ax,
                    ConstraintType::Eq => ax - rhs_val,
                };

                if violation > 1e-8 {
                    if violation > max_violation {
                        max_violation = violation;
                    }
                    // Proportional adjustment: shift each variable to reduce violation.
                    let nnz = (re - rs).max(1) as f64;
                    for k in rs..re {
                        let j = lp.col_indices[k];
                        let a_val = lp.values[k];
                        if a_val.abs() > 1e-15 {
                            let shift = violation / (nnz * a_val);
                            x[j] -= shift;
                            // Enforce bounds.
                            x[j] = x[j]
                                .max(if lp.lower_bounds[j].is_finite() {
                                    lp.lower_bounds[j]
                                } else {
                                    x[j]
                                })
                                .min(if lp.upper_bounds[j].is_finite() {
                                    lp.upper_bounds[j]
                                } else {
                                    x[j]
                                });
                        }
                    }
                }
            }

            if max_violation < 1e-6 {
                break;
            }
        }

        Ok(x)
    }

    // -----------------------------------------------------------------------
    // Dual bound
    // -----------------------------------------------------------------------

    /// The Lagrangian value at any λ is a valid dual bound.
    pub fn compute_dual_bound(&self, lagrangian_value: f64) -> f64 {
        lagrangian_value
    }

    // -----------------------------------------------------------------------
    // Reporting
    // -----------------------------------------------------------------------

    /// Format a quality report with bounds and gap.
    pub fn report_quality(&self, dual_bound: f64, primal_bound: f64) -> String {
        let gap = if primal_bound.abs() > 1e-10 {
            (primal_bound - dual_bound).abs() / primal_bound.abs() * 100.0
        } else {
            (primal_bound - dual_bound).abs() * 100.0
        };
        format!(
            "Dual bound: {:.6}, Primal bound: {:.6}, Gap: {:.4}%",
            dual_bound, primal_bound, gap
        )
    }

    // -----------------------------------------------------------------------
    // Problem decomposition
    // -----------------------------------------------------------------------

    /// Split the original problem into independent block subproblems
    /// (considering only kept constraints).
    pub fn decompose_problem(&self) -> Vec<LpProblem> {
        let lp = &self.problem.original;
        let n = lp.num_vars;
        let mut block_lps = Vec::with_capacity(self.problem.num_blocks);

        for block in 0..self.problem.num_blocks {
            let block_vars: Vec<usize> = (0..n)
                .filter(|&j| self.problem.variable_partition[j] == block)
                .collect();

            let block_n = block_vars.len();
            let mut var_to_local = vec![usize::MAX; n];
            for (local, &orig) in block_vars.iter().enumerate() {
                var_to_local[orig] = local;
            }

            let mut sub_lp = LpProblem::new(lp.maximize);

            // Add variables.
            for &orig in &block_vars {
                sub_lp.add_variable(
                    lp.obj_coeffs[orig],
                    lp.lower_bounds[orig],
                    lp.upper_bounds[orig],
                    Some(lp.var_names[orig].clone()),
                );
            }

            // Add kept constraints that involve only this block's variables.
            for &ci in &self.kept_constraints {
                let rs = lp.row_starts[ci];
                let re = lp.row_starts[ci + 1];

                let all_in_block = (rs..re)
                    .all(|k| self.problem.variable_partition[lp.col_indices[k]] == block);

                if !all_in_block {
                    continue;
                }

                let mut indices = Vec::new();
                let mut coeffs = Vec::new();
                for k in rs..re {
                    let j = lp.col_indices[k];
                    let local = var_to_local[j];
                    if local < block_n {
                        indices.push(local);
                        coeffs.push(lp.values[k]);
                    }
                }
                if !indices.is_empty() {
                    let _ = sub_lp.add_constraint(
                        &indices,
                        &coeffs,
                        lp.constraint_types[ci],
                        lp.rhs[ci],
                    );
                }
            }

            block_lps.push(sub_lp);
        }

        block_lps
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::LpProblem;

    /// Helper: build a small LP with 2 blocks linked by one constraint.
    ///
    ///  min -x1 - 2*x2 - x3 - x4
    ///  s.t. x1 + x2          <= 4   (block 0 constraint)
    ///              x3 + x4   <= 5   (block 1 constraint)
    ///       x1 + x3          <= 3   (linking / to relax)
    ///       x_i >= 0, x_i <= 10
    fn make_test_lp() -> (LpProblem, Vec<usize>, Vec<usize>) {
        let mut lp = LpProblem::new(false);
        lp.add_variable(-1.0, 0.0, 10.0, Some("x1".into()));
        lp.add_variable(-2.0, 0.0, 10.0, Some("x2".into()));
        lp.add_variable(-1.0, 0.0, 10.0, Some("x3".into()));
        lp.add_variable(-1.0, 0.0, 10.0, Some("x4".into()));

        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0)
            .unwrap();
        lp.add_constraint(&[2, 3], &[1.0, 1.0], ConstraintType::Le, 5.0)
            .unwrap();
        lp.add_constraint(&[0, 2], &[1.0, 1.0], ConstraintType::Le, 3.0)
            .unwrap();

        // Relax the linking constraint (index 2).
        let relaxed = vec![2];
        // Partition: x1,x2 → block 0; x3,x4 → block 1.
        let partition = vec![0, 0, 1, 1];
        (lp, relaxed, partition)
    }

    #[test]
    fn test_lagrangian_config_default() {
        let cfg = LagrangianConfig::default();
        assert_eq!(cfg.max_iterations, 500);
        assert!((cfg.gap_tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(cfg.heuristic_frequency, 10);
    }

    #[test]
    fn test_new_validates_relaxed_constraints() {
        let (lp, _, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: vec![99],
            num_blocks: 2,
            variable_partition: partition,
        };
        let result = LagrangianRelaxation::new(problem, LagrangianConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_new_validates_partition_length() {
        let (lp, relaxed, _) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: vec![0, 0], // wrong length
        };
        let result = LagrangianRelaxation::new(problem, LagrangianConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_new_validates_block_index() {
        let (lp, relaxed, _) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: vec![0, 0, 5, 1], // block 5 invalid
        };
        let result = LagrangianRelaxation::new(problem, LagrangianConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_new_success() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default());
        assert!(lr.is_ok());
    }

    #[test]
    fn test_evaluate_lagrangian_zero_multipliers() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        let info = lr.evaluate_lagrangian(&[0.0]).unwrap();
        // With λ=0, subproblems minimise original objective independently.
        // Block 0: min -x1 - 2x2 s.t. x1+x2<=4, 0<=xi<=10 → x2=4, x1=0, val=-8
        // Block 1: min -x3 - x4  s.t. x3+x4<=5, 0<=xi<=10 → x3=5 or x4=5, val=-5
        // Total L(0) = -8 + -5 = -13 (ish, depends on solver)
        assert!(info.value < 0.0, "val={}", info.value);
    }

    #[test]
    fn test_compute_subgradient() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        // x = [1, 2, 3, 4]
        // Relaxed constraint 2: x1 + x3 <= 3, so Ax - b = 1+3 - 3 = 1
        let g = lr.compute_subgradient(&[0.0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(g.len(), 1);
        assert!((g[0] - 1.0).abs() < 1e-12, "g={}", g[0]);
    }

    #[test]
    fn test_compute_dual_bound() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        assert!((lr.compute_dual_bound(-10.0) - (-10.0)).abs() < 1e-12);
    }

    #[test]
    fn test_report_quality() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        let report = lr.report_quality(-10.0, -8.0);
        assert!(report.contains("Dual bound"));
        assert!(report.contains("Primal bound"));
        assert!(report.contains("Gap"));
    }

    #[test]
    fn test_decompose_problem() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        let block_lps = lr.decompose_problem();
        assert_eq!(block_lps.len(), 2);
        // Block 0 has x1, x2; block 1 has x3, x4.
        assert_eq!(block_lps[0].num_vars, 2);
        assert_eq!(block_lps[1].num_vars, 2);
        // Block 0 has constraint x1+x2<=4.
        assert_eq!(block_lps[0].num_constraints, 1);
        // Block 1 has constraint x3+x4<=5.
        assert_eq!(block_lps[1].num_constraints, 1);
    }

    #[test]
    fn test_solve_no_relaxed() {
        let (lp, _, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: vec![], // no constraints relaxed
            num_blocks: 2,
            variable_partition: partition,
        };
        let mut lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        let result = lr.solve();
        assert!(result.is_err()); // should error: nothing to relax
    }

    #[test]
    fn test_solve_subgradient_method() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let cfg = LagrangianConfig {
            max_iterations: 50,
            method: LagrangianMethod::Subgradient,
            verbose: false,
            ..LagrangianConfig::default()
        };
        let mut lr = LagrangianRelaxation::new(problem, cfg).unwrap();
        let result = lr.solve().unwrap();
        assert!(result.iterations > 0);
        assert_eq!(result.multipliers.len(), 1);
    }

    #[test]
    fn test_solve_bundle_method() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let cfg = LagrangianConfig {
            max_iterations: 50,
            method: LagrangianMethod::Bundle,
            verbose: false,
            ..LagrangianConfig::default()
        };
        let mut lr = LagrangianRelaxation::new(problem, cfg).unwrap();
        let result = lr.solve().unwrap();
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_volume_method() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let cfg = LagrangianConfig {
            max_iterations: 50,
            method: LagrangianMethod::Volume,
            verbose: false,
            ..LagrangianConfig::default()
        };
        let mut lr = LagrangianRelaxation::new(problem, cfg).unwrap();
        let result = lr.solve().unwrap();
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_primal_recovery() {
        let (lp, relaxed, partition) = make_test_lp();
        let problem = LagrangianProblem {
            original: lp,
            relaxed_constraints: relaxed,
            num_blocks: 2,
            variable_partition: partition,
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();
        // Block 0: [x1=2, x2=2], block 1: [x3=2, x4=3]
        let block_sols = vec![vec![2.0, 2.0], vec![2.0, 3.0]];
        let primal = lr.primal_recovery(&block_sols, &[0.0]).unwrap();
        assert_eq!(primal.len(), 4);
        // Relaxed constraint: x1+x3<=3, with x1=2, x3=2 → violation=1.
        // Recovery should try to fix this.
    }

    #[test]
    fn test_solve_subproblem_box_only() {
        let (lp, relaxed, partition) = make_test_lp();
        // Build a problem where all constraints are relaxed → only box constraints remain.
        let all_relaxed = vec![0, 1, 2];
        let problem = LagrangianProblem {
            original: lp.clone(),
            relaxed_constraints: all_relaxed,
            num_blocks: 2,
            variable_partition: partition.clone(),
        };
        let lr = LagrangianRelaxation::new(problem, LagrangianConfig::default()).unwrap();

        let modified_obj = vec![-1.0, -2.0, -1.0, -1.0];
        let (val, sol) = lr.solve_subproblem(0, &modified_obj).unwrap();
        // Block 0: min -x1 - 2x2 with 0<=x<=10 → x1=10, x2=10, val=-30.
        assert!((val - (-30.0)).abs() < 1e-6, "val={}", val);
        assert!((sol[0] - 10.0).abs() < 1e-6);
        assert!((sol[1] - 10.0).abs() < 1e-6);
    }
}
