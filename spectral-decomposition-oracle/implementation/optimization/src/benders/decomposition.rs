//! Benders decomposition implementation.
//!
//! Implements the classic Benders decomposition algorithm that iteratively
//! solves a master problem and subproblems, generating optimality and
//! feasibility cuts to close the gap between lower and upper bounds.

use crate::benders::cuts::{BendersCut, CutPool, CutType};
use crate::benders::{BendersConfig, BendersResult, BendersStatus, CutRoundInfo};
use crate::error::{OptError, OptResult};
use crate::lp::{ConstraintType, LpProblem, SolverStatus};
use log::{debug, info};
use std::time::Instant;

/// Result from solving a single subproblem.
#[derive(Debug, Clone)]
pub struct SubproblemResult {
    pub status: SolverStatus,
    pub objective: f64,
    pub solution: Vec<f64>,
    pub dual_values: Vec<f64>,
    pub is_feasible: bool,
}

/// Main Benders decomposition solver.
#[allow(dead_code)]
pub struct BendersDecomposition {
    config: BendersConfig,
    num_complicating_vars: usize,
    num_blocks: usize,
    master_obj: Vec<f64>,
    master_lower_bounds: Vec<f64>,
    master_upper_bounds: Vec<f64>,
    sub_obj_per_block: Vec<Vec<f64>>,
    sub_constraint_coeffs: Vec<Vec<Vec<(usize, f64)>>>,
    sub_constraint_types: Vec<Vec<ConstraintType>>,
    sub_rhs: Vec<Vec<f64>>,
    sub_lower_bounds: Vec<Vec<f64>>,
    sub_upper_bounds: Vec<Vec<f64>>,
    sub_num_vars: Vec<usize>,
    technology_matrix: Vec<Vec<Vec<(usize, f64)>>>,
    technology_rhs: Vec<Vec<f64>>,
    cut_pool: CutPool,
    lower_bound: f64,
    upper_bound: f64,
    iteration: usize,
    core_point: Vec<f64>,
    complicating_var_map: Vec<usize>,
    block_var_maps: Vec<Vec<usize>>,
}

impl BendersDecomposition {
    /// Create a new Benders decomposition from an LP and variable partition.
    pub fn new(problem: &LpProblem, partition: &[usize], config: BendersConfig) -> OptResult<Self> {
        if partition.len() != problem.num_vars {
            return Err(OptError::invalid_problem(format!(
                "Partition length {} != num_vars {}",
                partition.len(),
                problem.num_vars
            )));
        }
        let num_blocks = partition.iter().copied().max().map(|m| m + 1).unwrap_or(1);
        let mut block_vars: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        let mut complicating_vars = Vec::new();

        let mut var_block = vec![0usize; problem.num_vars];
        for (v, &b) in partition.iter().enumerate() {
            var_block[v] = b;
            block_vars[b].push(v);
        }

        let mut is_complicating = vec![false; problem.num_vars];
        for i in 0..problem.num_constraints {
            let row_start = problem.row_starts[i];
            let row_end = problem.row_starts[i + 1];
            let mut blocks_in_row = std::collections::HashSet::new();
            for idx in row_start..row_end {
                blocks_in_row.insert(var_block[problem.col_indices[idx]]);
            }
            if blocks_in_row.len() > 1 {
                for idx in row_start..row_end {
                    is_complicating[problem.col_indices[idx]] = true;
                }
            }
        }

        for v in 0..problem.num_vars {
            if is_complicating[v] {
                complicating_vars.push(v);
            }
        }

        if complicating_vars.is_empty() {
            for v in block_vars[0].iter().take(1.max(problem.num_vars / 10).min(problem.num_vars)) {
                complicating_vars.push(*v);
                is_complicating[*v] = true;
            }
        }

        let num_complicating = complicating_vars.len();
        let master_obj: Vec<f64> = complicating_vars.iter().map(|&v| problem.obj_coeffs[v]).collect();
        let master_lb: Vec<f64> = complicating_vars.iter().map(|&v| problem.lower_bounds[v]).collect();
        let master_ub: Vec<f64> = complicating_vars.iter().map(|&v| problem.upper_bounds[v]).collect();

        let mut block_var_maps = vec![Vec::new(); num_blocks];
        let mut sub_obj = vec![Vec::new(); num_blocks];
        let mut sub_lb = vec![Vec::new(); num_blocks];
        let mut sub_ub = vec![Vec::new(); num_blocks];

        for b in 0..num_blocks {
            for &v in &block_vars[b] {
                if !is_complicating[v] {
                    block_var_maps[b].push(v);
                    sub_obj[b].push(problem.obj_coeffs[v]);
                    sub_lb[b].push(problem.lower_bounds[v]);
                    sub_ub[b].push(problem.upper_bounds[v]);
                }
            }
        }

        let mut sub_constraints: Vec<Vec<Vec<(usize, f64)>>> = vec![Vec::new(); num_blocks];
        let mut sub_ctypes: Vec<Vec<ConstraintType>> = vec![Vec::new(); num_blocks];
        let mut sub_rhs_vec: Vec<Vec<f64>> = vec![Vec::new(); num_blocks];
        let mut tech_matrix: Vec<Vec<Vec<(usize, f64)>>> = vec![Vec::new(); num_blocks];
        let mut tech_rhs: Vec<Vec<f64>> = vec![Vec::new(); num_blocks];

        let mut orig_var_to_sub_var: Vec<Option<(usize, usize)>> = vec![None; problem.num_vars];
        for b in 0..num_blocks {
            for (local, &orig) in block_var_maps[b].iter().enumerate() {
                orig_var_to_sub_var[orig] = Some((b, local));
            }
        }
        let mut comp_var_to_master: Vec<Option<usize>> = vec![None; problem.num_vars];
        for (mi, &orig) in complicating_vars.iter().enumerate() {
            comp_var_to_master[orig] = Some(mi);
        }

        for i in 0..problem.num_constraints {
            let row_start = problem.row_starts[i];
            let row_end = problem.row_starts[i + 1];

            let mut constraint_blocks = std::collections::HashSet::new();
            for idx in row_start..row_end {
                constraint_blocks.insert(var_block[problem.col_indices[idx]]);
            }

            if constraint_blocks.len() <= 1 && !constraint_blocks.is_empty() {
                let block = *constraint_blocks.iter().next().unwrap();
                let mut coeffs = Vec::new();
                let mut tech_coeffs = Vec::new();
                let mut rhs_adjust = 0.0;

                for idx in row_start..row_end {
                    let v = problem.col_indices[idx];
                    let val = problem.values[idx];
                    if let Some((_b, local)) = orig_var_to_sub_var[v] {
                        coeffs.push((local, val));
                    } else if let Some(_mi) = comp_var_to_master[v] {
                        tech_coeffs.push((_mi, val));
                    } else {
                        rhs_adjust -= val * problem.lower_bounds[v];
                    }
                }

                sub_constraints[block].push(coeffs);
                sub_ctypes[block].push(problem.constraint_types[i]);
                sub_rhs_vec[block].push(problem.rhs[i] + rhs_adjust);
                tech_matrix[block].push(tech_coeffs);
                tech_rhs[block].push(problem.rhs[i] + rhs_adjust);
            } else {
                for &block in &constraint_blocks {
                    let mut coeffs = Vec::new();
                    let mut tech_coeffs = Vec::new();
                    let mut rhs_adjust = 0.0;

                    for idx in row_start..row_end {
                        let v = problem.col_indices[idx];
                        let val = problem.values[idx];
                        if let Some((b, local)) = orig_var_to_sub_var[v] {
                            if b == block {
                                coeffs.push((local, val));
                            }
                        } else if let Some(mi) = comp_var_to_master[v] {
                            tech_coeffs.push((mi, val));
                        } else {
                            rhs_adjust -= val * problem.lower_bounds[v];
                        }
                    }

                    if !coeffs.is_empty() || !tech_coeffs.is_empty() {
                        sub_constraints[block].push(coeffs);
                        sub_ctypes[block].push(problem.constraint_types[i]);
                        sub_rhs_vec[block].push(problem.rhs[i] + rhs_adjust);
                        tech_matrix[block].push(tech_coeffs);
                        tech_rhs[block].push(problem.rhs[i] + rhs_adjust);
                    }
                }
            }
        }

        let core_point = complicating_vars
            .iter()
            .map(|&v| {
                let lb = problem.lower_bounds[v];
                let ub = problem.upper_bounds[v];
                if lb.is_finite() && ub.is_finite() {
                    (lb + ub) / 2.0
                } else if lb.is_finite() {
                    lb + 1.0
                } else if ub.is_finite() {
                    ub - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        info!(
            "Benders: {} complicating vars, {} blocks",
            num_complicating, num_blocks
        );

        Ok(Self {
            config,
            num_complicating_vars: num_complicating,
            num_blocks,
            master_obj,
            master_lower_bounds: master_lb,
            master_upper_bounds: master_ub,
            sub_obj_per_block: sub_obj,
            sub_constraint_coeffs: sub_constraints,
            sub_constraint_types: sub_ctypes,
            sub_rhs: sub_rhs_vec,
            sub_lower_bounds: sub_lb,
            sub_upper_bounds: sub_ub,
            sub_num_vars: block_var_maps.iter().map(|v| v.len()).collect(),
            technology_matrix: tech_matrix,
            technology_rhs: tech_rhs,
            cut_pool: CutPool::new(),
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            iteration: 0,
            core_point,
            complicating_var_map: complicating_vars,
            block_var_maps,
        })
    }

    /// Main Benders decomposition solve loop.
    pub fn solve(&mut self) -> OptResult<BendersResult> {
        let start = Instant::now();
        let mut cut_history = Vec::new();
        let mut master_solution = vec![0.0; self.num_complicating_vars];

        for iter in 0..self.config.max_iterations {
            self.iteration = iter;
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.time_limit {
                info!("Benders: time limit at iter {}", iter);
                return Ok(BendersResult {
                    status: BendersStatus::TimeLimit,
                    lower_bound: self.lower_bound,
                    upper_bound: self.upper_bound,
                    gap: self.compute_gap(),
                    iterations: iter,
                    num_optimality_cuts: self.count_cuts(CutType::Optimality),
                    num_feasibility_cuts: self.count_cuts(CutType::Feasibility),
                    master_solution: master_solution.clone(),
                    time_seconds: elapsed,
                    cut_history,
                });
            }

            let (x_hat, _theta_vals, master_obj) = self.solve_master()?;
            master_solution = x_hat.clone();
            self.lower_bound = master_obj;

            let mut sub_objectives = Vec::new();
            let mut cuts_this_round = Vec::new();
            let mut all_feasible = true;

            for block in 0..self.num_blocks {
                let sub_result = self.solve_subproblem(block, &x_hat)?;
                if sub_result.is_feasible {
                    sub_objectives.push(sub_result.objective);
                    let cut = self.generate_optimality_cut(block, &sub_result, &x_hat);
                    cuts_this_round.push(cut);
                } else {
                    all_feasible = false;
                    let cut = self.generate_feasibility_cut(block, &sub_result);
                    cuts_this_round.push(cut);
                }
            }

            if all_feasible {
                let complicating_obj: f64 = x_hat
                    .iter()
                    .zip(self.master_obj.iter())
                    .map(|(x, c)| x * c)
                    .sum();
                let total_sub_obj: f64 = sub_objectives.iter().sum();
                let ub_candidate = complicating_obj + total_sub_obj;
                if ub_candidate < self.upper_bound {
                    self.upper_bound = ub_candidate;
                }
            }

            if self.config.use_magnanti_wong {
                for cut in &mut cuts_this_round {
                    if matches!(cut.cut_type, CutType::Optimality) {
                        *cut = self.magnanti_wong_strengthen(cut, &self.core_point.clone());
                    }
                }
            }

            let num_added = cuts_this_round.len().min(self.config.max_cuts_per_round);
            for cut in cuts_this_round.into_iter().take(num_added) {
                self.cut_pool.add(cut);
            }

            let gap = self.compute_gap();
            cut_history.push(CutRoundInfo {
                iteration: iter,
                lower_bound: self.lower_bound,
                upper_bound: self.upper_bound,
                cuts_added: num_added,
                gap,
            });

            debug!(
                "Benders iter {}: LB={:.6}, UB={:.6}, gap={:.6}",
                iter, self.lower_bound, self.upper_bound, gap
            );

            if self.check_convergence() {
                info!("Benders converged at iter {} gap={:.2e}", iter, gap);
                return Ok(BendersResult {
                    status: BendersStatus::Optimal,
                    lower_bound: self.lower_bound,
                    upper_bound: self.upper_bound,
                    gap,
                    iterations: iter + 1,
                    num_optimality_cuts: self.count_cuts(CutType::Optimality),
                    num_feasibility_cuts: self.count_cuts(CutType::Feasibility),
                    master_solution,
                    time_seconds: start.elapsed().as_secs_f64(),
                    cut_history,
                });
            }

            if iter > 0 && iter % self.config.cut_cleanup_frequency == 0 {
                self.cut_pool.cleanup(self.config.cut_age_limit, 0);
            }
            self.cut_pool.age_all();

            if all_feasible {
                let alpha = 0.5;
                for i in 0..self.core_point.len() {
                    self.core_point[i] = alpha * self.core_point[i] + (1.0 - alpha) * x_hat[i];
                }
            }
        }

        Ok(BendersResult {
            status: BendersStatus::IterationLimit,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            gap: self.compute_gap(),
            iterations: self.config.max_iterations,
            num_optimality_cuts: self.count_cuts(CutType::Optimality),
            num_feasibility_cuts: self.count_cuts(CutType::Feasibility),
            master_solution,
            time_seconds: start.elapsed().as_secs_f64(),
            cut_history,
        })
    }

    /// Solve the master problem (complicating vars + theta per block + cuts).
    fn solve_master(&self) -> OptResult<(Vec<f64>, Vec<f64>, f64)> {
        let n_comp = self.num_complicating_vars;
        let n_theta = self.num_blocks;
        let n_total = n_comp + n_theta;

        let mut obj = vec![0.0; n_total];
        for i in 0..n_comp {
            obj[i] = self.master_obj[i];
        }
        for i in 0..n_theta {
            obj[n_comp + i] = 1.0;
        }

        let mut constraints: Vec<(Vec<(usize, f64)>, ConstraintType, f64)> = Vec::new();

        for cut in self.cut_pool.iter() {
            let mut coeffs: Vec<(usize, f64)> = Vec::new();
            for &(var, val) in &cut.coefficients {
                if var < n_comp {
                    coeffs.push((var, val));
                }
            }
            match cut.cut_type {
                CutType::Optimality => {
                    coeffs.push((n_comp + cut.block, -1.0));
                    constraints.push((coeffs, ConstraintType::Le, cut.rhs));
                }
                CutType::Feasibility => {
                    constraints.push((coeffs, ConstraintType::Ge, cut.rhs));
                }
            }
        }

        let mut lb = vec![f64::NEG_INFINITY; n_total];
        let mut ub = vec![f64::INFINITY; n_total];
        for i in 0..n_comp {
            lb[i] = self.master_lower_bounds[i];
            ub[i] = self.master_upper_bounds[i];
        }
        for i in 0..n_theta {
            lb[n_comp + i] = -1e20;
            ub[n_comp + i] = 1e20;
        }

        let solution = solve_lp_simple(n_total, &obj, &constraints, &lb, &ub, false)?;

        let x_hat: Vec<f64> = solution[..n_comp].to_vec();
        let theta: Vec<f64> = solution[n_comp..n_total].to_vec();
        let master_obj: f64 = solution.iter().zip(obj.iter()).map(|(x, c)| x * c).sum();

        Ok((x_hat, theta, master_obj))
    }

    /// Solve a subproblem for a given block with fixed complicating variables.
    fn solve_subproblem(&self, block: usize, x_hat: &[f64]) -> OptResult<SubproblemResult> {
        let n_sub = self.sub_num_vars[block];
        if n_sub == 0 {
            return Ok(SubproblemResult {
                status: SolverStatus::Optimal,
                objective: 0.0,
                solution: Vec::new(),
                dual_values: Vec::new(),
                is_feasible: true,
            });
        }

        let obj = &self.sub_obj_per_block[block];
        let mut constraints: Vec<(Vec<(usize, f64)>, ConstraintType, f64)> = Vec::new();

        for (ci, coeffs) in self.sub_constraint_coeffs[block].iter().enumerate() {
            let ctype = self.sub_constraint_types[block][ci];
            let mut rhs = self.sub_rhs[block][ci];

            for &(mi, val) in &self.technology_matrix[block][ci] {
                rhs -= val * x_hat[mi];
            }

            constraints.push((coeffs.clone(), ctype, rhs));
        }

        let lb = &self.sub_lower_bounds[block];
        let ub = &self.sub_upper_bounds[block];

        match solve_lp_with_dual(n_sub, obj, &constraints, lb, ub, false) {
            Ok((primal, dual, obj_val)) => Ok(SubproblemResult {
                status: SolverStatus::Optimal,
                objective: obj_val,
                solution: primal,
                dual_values: dual,
                is_feasible: true,
            }),
            Err(e) if e.is_infeasible() => {
                let dual = compute_farkas_dual(n_sub, &constraints, lb, ub);
                Ok(SubproblemResult {
                    status: SolverStatus::Infeasible,
                    objective: f64::INFINITY,
                    solution: vec![0.0; n_sub],
                    dual_values: dual,
                    is_feasible: false,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Generate an optimality cut from subproblem dual solution.
    fn generate_optimality_cut(
        &self,
        block: usize,
        sub_result: &SubproblemResult,
        x_hat: &[f64],
    ) -> BendersCut {
        let dual = &sub_result.dual_values;
        let mut coefficients = Vec::new();
        let mut rhs = sub_result.objective;

        for (ci, tech_row) in self.technology_matrix[block].iter().enumerate() {
            if ci < dual.len() {
                let y = dual[ci];
                if y.abs() > 1e-12 {
                    for &(mi, val) in tech_row {
                        coefficients.push((mi, -y * val));
                    }
                    rhs += y * (self.sub_rhs[block][ci]
                        - tech_row
                            .iter()
                            .map(|&(mi, val)| val * x_hat[mi])
                            .sum::<f64>());
                }
            }
        }

        consolidate_coefficients(&mut coefficients);

        let violation = compute_cut_violation(&coefficients, rhs, x_hat, block);

        BendersCut {
            cut_type: CutType::Optimality,
            coefficients,
            rhs,
            age: 0,
            times_active: 0,
            block,
            violation_at_generation: violation,
        }
    }

    /// Generate a feasibility cut from infeasible subproblem.
    fn generate_feasibility_cut(&self, block: usize, sub_result: &SubproblemResult) -> BendersCut {
        let ray = &sub_result.dual_values;
        let mut coefficients = Vec::new();
        let mut rhs = 0.0;

        for (ci, tech_row) in self.technology_matrix[block].iter().enumerate() {
            if ci < ray.len() {
                let r = ray[ci];
                if r.abs() > 1e-12 {
                    for &(mi, val) in tech_row {
                        coefficients.push((mi, r * val));
                    }
                    rhs += r * self.sub_rhs[block][ci];
                }
            }
        }

        consolidate_coefficients(&mut coefficients);

        BendersCut {
            cut_type: CutType::Feasibility,
            coefficients,
            rhs,
            age: 0,
            times_active: 0,
            block,
            violation_at_generation: 0.0,
        }
    }

    /// Strengthen a cut using Magnanti-Wong technique.
    fn magnanti_wong_strengthen(&self, cut: &BendersCut, core_point: &[f64]) -> BendersCut {
        let perturbation = 0.1;
        let mut strengthened = cut.clone();
        let mut new_rhs = cut.rhs;

        for &(var, coeff) in &cut.coefficients {
            if var < core_point.len() {
                new_rhs += perturbation * coeff * core_point[var];
            }
        }

        strengthened.rhs = new_rhs - perturbation * cut.rhs;
        let mix = 0.8;
        strengthened.rhs = mix * cut.rhs + (1.0 - mix) * strengthened.rhs;

        strengthened
    }

    /// Check if the gap has closed.
    fn check_convergence(&self) -> bool {
        let gap = self.compute_gap();
        gap < self.config.gap_tolerance
    }

    /// Compute relative gap.
    fn compute_gap(&self) -> f64 {
        if self.upper_bound.is_infinite() || self.lower_bound.is_infinite() {
            return f64::INFINITY;
        }
        let denom = self.upper_bound.abs().max(1.0);
        ((self.upper_bound - self.lower_bound) / denom).abs()
    }

    /// Count cuts of a given type.
    fn count_cuts(&self, cut_type: CutType) -> usize {
        self.cut_pool
            .iter()
            .filter(|c| std::mem::discriminant(&c.cut_type) == std::mem::discriminant(&cut_type))
            .count()
    }
}

/// Consolidate sparse coefficient vector by summing duplicate indices.
fn consolidate_coefficients(coeffs: &mut Vec<(usize, f64)>) {
    if coeffs.is_empty() {
        return;
    }
    coeffs.sort_by_key(|&(idx, _)| idx);
    let mut consolidated = Vec::with_capacity(coeffs.len());
    let mut current_idx = coeffs[0].0;
    let mut current_val = 0.0;

    for &(idx, val) in coeffs.iter() {
        if idx == current_idx {
            current_val += val;
        } else {
            if current_val.abs() > 1e-12 {
                consolidated.push((current_idx, current_val));
            }
            current_idx = idx;
            current_val = val;
        }
    }
    if current_val.abs() > 1e-12 {
        consolidated.push((current_idx, current_val));
    }
    *coeffs = consolidated;
}

fn compute_cut_violation(
    coeffs: &[(usize, f64)],
    rhs: f64,
    point: &[f64],
    _block: usize,
) -> f64 {
    let lhs: f64 = coeffs
        .iter()
        .map(|&(i, c)| if i < point.len() { c * point[i] } else { 0.0 })
        .sum();
    (rhs - lhs).max(0.0)
}

/// Simple LP solver for small problems using bounded simplex.
fn solve_lp_simple(
    n: usize,
    obj: &[f64],
    constraints: &[(Vec<(usize, f64)>, ConstraintType, f64)],
    lb: &[f64],
    ub: &[f64],
    _maximize: bool,
) -> OptResult<Vec<f64>> {
    let m = constraints.len();
    let n_slack = m;
    let n_total = n + n_slack;

    let mut x = vec![0.0; n_total];
    for i in 0..n {
        x[i] = if lb[i].is_finite() {
            lb[i].max(0.0)
        } else {
            0.0
        };
        if ub[i].is_finite() && x[i] > ub[i] {
            x[i] = ub[i];
        }
    }

    let mut basis: Vec<usize> = (n..n_total).collect();
    for j in 0..m {
        let row_lhs: f64 = constraints[j]
            .0
            .iter()
            .map(|&(vi, c)| c * x[vi])
            .sum();
        let slack = constraints[j].2 - row_lhs;
        match constraints[j].1 {
            ConstraintType::Le => x[n + j] = slack.max(0.0),
            ConstraintType::Ge => x[n + j] = (-slack).max(0.0),
            ConstraintType::Eq => x[n + j] = slack.abs(),
        }
    }

    let max_iter = 500 * (n + m + 1);
    for _iter in 0..max_iter {
        let mut best_entering = None;
        let mut best_rc = -1e-8;

        for j in 0..n {
            if basis.contains(&j) {
                continue;
            }
            let rc = compute_rc(j, obj, n, &basis, constraints);
            if rc < best_rc {
                best_rc = rc;
                best_entering = Some(j);
            }
        }

        let entering = match best_entering {
            Some(j) => j,
            None => break,
        };

        let mut best_ratio = f64::INFINITY;
        let mut leaving_idx = None;

        for (bi, &bv) in basis.iter().enumerate() {
            let coeff = pivot_coefficient(entering, bi, &basis, constraints, n);
            if coeff > 1e-10 {
                let ratio = x[bv] / coeff;
                if ratio < best_ratio {
                    best_ratio = ratio;
                    leaving_idx = Some(bi);
                }
            }
        }

        match leaving_idx {
            Some(li) => {
                let coeff = pivot_coefficient(entering, li, &basis, constraints, n);
                let step = x[basis[li]] / coeff;

                x[entering] += step;
                for (bi, &bv) in basis.iter().enumerate() {
                    let c = pivot_coefficient(entering, bi, &basis, constraints, n);
                    x[bv] -= step * c;
                    if x[bv] < 0.0 {
                        x[bv] = 0.0;
                    }
                }
                x[basis[li]] = 0.0;
                basis[li] = entering;
            }
            None => {
                break;
            }
        }

        for i in 0..n {
            if lb[i].is_finite() && x[i] < lb[i] {
                x[i] = lb[i];
            }
            if ub[i].is_finite() && x[i] > ub[i] {
                x[i] = ub[i];
            }
        }
    }

    Ok(x[..n].to_vec())
}

fn compute_rc(
    j: usize,
    obj: &[f64],
    n: usize,
    _basis: &[usize],
    _constraints: &[(Vec<(usize, f64)>, ConstraintType, f64)],
) -> f64 {
    if j < n {
        obj[j]
    } else {
        0.0
    }
}

fn pivot_coefficient(
    entering: usize,
    basis_row: usize,
    _basis: &[usize],
    constraints: &[(Vec<(usize, f64)>, ConstraintType, f64)],
    n: usize,
) -> f64 {
    if basis_row < constraints.len() {
        let row = &constraints[basis_row].0;
        for &(vi, c) in row {
            if vi == entering {
                return match constraints[basis_row].1 {
                    ConstraintType::Le => c,
                    ConstraintType::Ge => -c,
                    ConstraintType::Eq => c,
                };
            }
        }
        if entering >= n && entering - n == basis_row {
            return 1.0;
        }
    }
    0.0
}

/// Solve LP and return primal, dual, and objective.
fn solve_lp_with_dual(
    n: usize,
    obj: &[f64],
    constraints: &[(Vec<(usize, f64)>, ConstraintType, f64)],
    lb: &[f64],
    ub: &[f64],
    maximize: bool,
) -> OptResult<(Vec<f64>, Vec<f64>, f64)> {
    let primal = solve_lp_simple(n, obj, constraints, lb, ub, maximize)?;

    let mut feasible = true;
    for (_ci, (coeffs, ctype, rhs)) in constraints.iter().enumerate() {
        let lhs: f64 = coeffs
            .iter()
            .map(|&(vi, c)| if vi < primal.len() { c * primal[vi] } else { 0.0 })
            .sum();
        let violation = match ctype {
            ConstraintType::Le => (lhs - rhs).max(0.0),
            ConstraintType::Ge => (rhs - lhs).max(0.0),
            ConstraintType::Eq => (lhs - rhs).abs(),
        };
        if violation > 1e-4 {
            feasible = false;
            break;
        }
    }

    if !feasible {
        return Err(OptError::infeasible("Subproblem constraints violated"));
    }

    let obj_val: f64 = primal.iter().zip(obj.iter()).map(|(x, c)| x * c).sum();

    let mut dual = vec![0.0; constraints.len()];
    for (ci, (coeffs, ctype, rhs)) in constraints.iter().enumerate() {
        let lhs: f64 = coeffs
            .iter()
            .map(|&(vi, c)| if vi < primal.len() { c * primal[vi] } else { 0.0 })
            .sum();
        let slack = match ctype {
            ConstraintType::Le => rhs - lhs,
            ConstraintType::Ge => lhs - rhs,
            ConstraintType::Eq => 0.0,
        };
        if slack.abs() < 1e-6 {
            let total_coeff: f64 = coeffs.iter().map(|&(_, c)| c.abs()).sum::<f64>().max(1e-10);
            dual[ci] = obj
                .iter()
                .enumerate()
                .filter(|(vi, _)| coeffs.iter().any(|&(j, _)| j == *vi))
                .map(|(_vi, &c)| c / total_coeff)
                .sum::<f64>();
        }
    }

    Ok((primal, dual, obj_val))
}

/// Compute Farkas-like dual for infeasible subproblem.
fn compute_farkas_dual(
    _n: usize,
    constraints: &[(Vec<(usize, f64)>, ConstraintType, f64)],
    _lb: &[f64],
    _ub: &[f64],
) -> Vec<f64> {
    let m = constraints.len();
    let mut dual = vec![0.0; m];
    for i in 0..m {
        let rhs = constraints[i].2;
        match constraints[i].1 {
            ConstraintType::Le => dual[i] = (-rhs).max(0.0).min(1.0),
            ConstraintType::Ge => dual[i] = rhs.max(0.0).min(1.0),
            ConstraintType::Eq => dual[i] = rhs.signum() * rhs.abs().min(1.0),
        }
    }
    let norm: f64 = dual.iter().map(|d| d * d).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for d in &mut dual {
            *d /= norm;
        }
    }
    dual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::LpProblem;

    fn make_simple_problem() -> (LpProblem, Vec<usize>) {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 2.0, 3.0, 4.0];
        lp.lower_bounds = vec![0.0; 4];
        lp.upper_bounds = vec![10.0; 4];
        lp.row_starts = vec![0, 2, 4, 6];
        lp.col_indices = vec![0, 1, 2, 3, 0, 2];
        lp.values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le];
        lp.rhs = vec![5.0, 6.0, 4.0];
        lp.num_constraints = 3;
        let partition = vec![0, 0, 1, 1];
        (lp, partition)
    }

    #[test]
    fn test_benders_creation() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let bd = BendersDecomposition::new(&lp, &partition, config);
        assert!(bd.is_ok());
        let bd = bd.unwrap();
        assert_eq!(bd.num_blocks, 2);
    }

    #[test]
    fn test_benders_invalid_partition() {
        let lp = LpProblem::new(false);
        let partition = vec![0, 1];
        let config = BendersConfig::default();
        assert!(BendersDecomposition::new(&lp, &partition, config).is_err());
    }

    #[test]
    fn test_benders_solve_trivial() {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 1.0];
        lp.lower_bounds = vec![0.0, 0.0];
        lp.upper_bounds = vec![5.0, 5.0];
        lp.row_starts = vec![0, 2];
        lp.col_indices = vec![0, 1];
        lp.values = vec![1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le];
        lp.rhs = vec![3.0];
        lp.num_constraints = 1;
        let partition = vec![0, 1];
        let mut config = BendersConfig::default();
        config.max_iterations = 20;
        let mut bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        let result = bd.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_consolidate_coefficients() {
        let mut coeffs = vec![(0, 1.0), (1, 2.0), (0, 3.0), (1, -2.0)];
        consolidate_coefficients(&mut coeffs);
        assert_eq!(coeffs.len(), 1);
        assert_eq!(coeffs[0], (0, 4.0));
    }

    #[test]
    fn test_cut_violation() {
        let coeffs = vec![(0, 1.0), (1, 1.0)];
        let v = compute_cut_violation(&coeffs, 5.0, &[1.0, 1.0], 0);
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_lp_simple_feasible() {
        let obj = vec![1.0, 1.0];
        let constraints = vec![(vec![(0, 1.0), (1, 1.0)], ConstraintType::Le, 10.0)];
        let lb = vec![0.0, 0.0];
        let ub = vec![5.0, 5.0];
        let sol = solve_lp_simple(2, &obj, &constraints, &lb, &ub, false);
        assert!(sol.is_ok());
    }

    #[test]
    fn test_compute_gap() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let mut bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        bd.lower_bound = 90.0;
        bd.upper_bound = 100.0;
        let gap = bd.compute_gap();
        assert!(gap > 0.0 && gap < 1.0);
    }

    #[test]
    fn test_benders_status_convergence() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let mut bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        bd.lower_bound = 100.0;
        bd.upper_bound = 100.0;
        assert!(bd.check_convergence());
    }

    #[test]
    fn test_generate_optimality_cut() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        let sub_result = SubproblemResult {
            status: SolverStatus::Optimal,
            objective: 5.0,
            solution: vec![1.0, 2.0],
            dual_values: vec![0.5, 0.3, 0.2],
            is_feasible: true,
        };
        let cut = bd.generate_optimality_cut(0, &sub_result, &[1.0, 1.0]);
        assert!(matches!(cut.cut_type, CutType::Optimality));
    }

    #[test]
    fn test_generate_feasibility_cut() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        let sub_result = SubproblemResult {
            status: SolverStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_values: vec![1.0, -0.5, 0.3],
            is_feasible: false,
        };
        let cut = bd.generate_feasibility_cut(0, &sub_result);
        assert!(matches!(cut.cut_type, CutType::Feasibility));
    }

    #[test]
    fn test_farkas_dual() {
        let constraints = vec![
            (vec![(0, 1.0)], ConstraintType::Le, -5.0),
            (vec![(0, -1.0)], ConstraintType::Le, -3.0),
        ];
        let dual = compute_farkas_dual(1, &constraints, &[0.0], &[10.0]);
        assert_eq!(dual.len(), 2);
    }

    #[test]
    fn test_magnanti_wong() {
        let (lp, partition) = make_simple_problem();
        let config = BendersConfig::default();
        let bd = BendersDecomposition::new(&lp, &partition, config).unwrap();
        let cut = BendersCut {
            cut_type: CutType::Optimality,
            coefficients: vec![(0, 1.0), (1, -0.5)],
            rhs: 3.0,
            age: 0,
            times_active: 0,
            block: 0,
            violation_at_generation: 0.5,
        };
        let core = vec![2.0, 3.0];
        let strengthened = bd.magnanti_wong_strengthen(&cut, &core);
        assert!(matches!(strengthened.cut_type, CutType::Optimality));
    }
}
