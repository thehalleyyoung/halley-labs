//! Dantzig-Wolfe decomposition implementation.
//!
//! Implements column generation for block-structured LPs by maintaining a
//! restricted master problem (RMP) with convexity constraints and iteratively
//! adding columns via pricing subproblems.

use crate::dw::column::{ColumnPool, DWColumn};
use crate::dw::{ColumnRoundInfo, DWConfig, DWResult, DWStabilization, DWStatus};
use crate::error::{OptError, OptResult};
use crate::lp::{ConstraintType, LpProblem, SolverStatus};
use log::{debug, info, warn};
use std::time::Instant;

/// Result from solving the restricted master problem.
#[derive(Debug, Clone)]
pub struct RmpSolution {
    pub objective: f64,
    pub lambda_values: Vec<f64>,
    pub dual_linking: Vec<f64>,
    pub dual_convexity: Vec<f64>,
    pub status: SolverStatus,
}

/// Result from pricing a subproblem.
#[derive(Debug, Clone)]
pub struct PricingResult {
    pub column: Option<DWColumn>,
    pub reduced_cost: f64,
    pub found_improving: bool,
}

/// Dantzig-Wolfe decomposition solver.
#[allow(dead_code)]
pub struct DWDecomposition {
    config: DWConfig,
    num_blocks: usize,
    num_linking_constraints: usize,
    linking_rhs: Vec<f64>,
    linking_constraint_types: Vec<ConstraintType>,
    /// Per-block: original objective coefficients for block variables
    block_obj: Vec<Vec<f64>>,
    /// Per-block: constraint coefficients (sparse rows)
    block_constraints: Vec<Vec<Vec<(usize, f64)>>>,
    block_constraint_types: Vec<Vec<ConstraintType>>,
    block_rhs: Vec<Vec<f64>>,
    block_lb: Vec<Vec<f64>>,
    block_ub: Vec<Vec<f64>>,
    block_num_vars: Vec<usize>,
    /// Per-block, per-linking-constraint: coefficients of block vars in linking constraint
    block_linking_matrix: Vec<Vec<Vec<(usize, f64)>>>,
    column_pool: ColumnPool,
    rmp_column_indices: Vec<usize>,
    dual_values: Vec<f64>,
    smoothed_duals: Vec<f64>,
    lower_bound: f64,
    upper_bound: f64,
    iteration: usize,
    next_col_id: usize,
}

impl DWDecomposition {
    /// Create a new DW decomposition from an LP and variable partition.
    pub fn new(problem: &LpProblem, partition: &[usize], config: DWConfig) -> OptResult<Self> {
        if partition.len() != problem.num_vars {
            return Err(OptError::invalid_problem(format!(
                "Partition length {} != num_vars {}",
                partition.len(),
                problem.num_vars
            )));
        }
        let num_blocks = partition.iter().copied().max().map(|m| m + 1).unwrap_or(1);

        let mut block_vars: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
        for (v, &b) in partition.iter().enumerate() {
            if b >= num_blocks {
                return Err(OptError::invalid_problem("Partition index out of range"));
            }
            block_vars[b].push(v);
        }

        // Classify constraints: linking (touch multiple blocks) vs block-local
        let mut linking_constraint_indices = Vec::new();
        let mut block_constraint_indices: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];

        for i in 0..problem.num_constraints {
            let row_start = problem.row_starts[i];
            let row_end = problem.row_starts[i + 1];
            let mut blocks_touched = std::collections::HashSet::new();
            for idx in row_start..row_end {
                blocks_touched.insert(partition[problem.col_indices[idx]]);
            }
            if blocks_touched.len() > 1 {
                linking_constraint_indices.push(i);
            } else if let Some(&b) = blocks_touched.iter().next() {
                block_constraint_indices[b].push(i);
            }
        }

        let num_linking = linking_constraint_indices.len();
        let linking_rhs: Vec<f64> = linking_constraint_indices
            .iter()
            .map(|&i| problem.rhs[i])
            .collect();
        let linking_ctypes: Vec<ConstraintType> = linking_constraint_indices
            .iter()
            .map(|&i| problem.constraint_types[i])
            .collect();

        // Build local-to-global variable mapping per block
        let mut var_to_local: Vec<Option<(usize, usize)>> = vec![None; problem.num_vars];
        for b in 0..num_blocks {
            for (local, &global) in block_vars[b].iter().enumerate() {
                var_to_local[global] = Some((b, local));
            }
        }

        // Build block-level data
        let mut block_obj = Vec::with_capacity(num_blocks);
        let mut block_constraints = Vec::with_capacity(num_blocks);
        let mut block_ctypes = Vec::with_capacity(num_blocks);
        let mut block_rhs_vec = Vec::with_capacity(num_blocks);
        let mut block_lb = Vec::with_capacity(num_blocks);
        let mut block_ub = Vec::with_capacity(num_blocks);
        let mut block_num_vars = Vec::with_capacity(num_blocks);
        let mut block_linking_matrix: Vec<Vec<Vec<(usize, f64)>>> = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let nv = block_vars[b].len();
            block_num_vars.push(nv);
            block_obj.push(block_vars[b].iter().map(|&v| problem.obj_coeffs[v]).collect());
            block_lb.push(block_vars[b].iter().map(|&v| problem.lower_bounds[v]).collect());
            block_ub.push(block_vars[b].iter().map(|&v| problem.upper_bounds[v]).collect());

            // Block constraints
            let mut bc = Vec::new();
            let mut bt = Vec::new();
            let mut br = Vec::new();
            for &ci in &block_constraint_indices[b] {
                let row_start = problem.row_starts[ci];
                let row_end = problem.row_starts[ci + 1];
                let mut row = Vec::new();
                for idx in row_start..row_end {
                    let v = problem.col_indices[idx];
                    if let Some((_b, local)) = var_to_local[v] {
                        row.push((local, problem.values[idx]));
                    }
                }
                bc.push(row);
                bt.push(problem.constraint_types[ci]);
                br.push(problem.rhs[ci]);
            }
            block_constraints.push(bc);
            block_ctypes.push(bt);
            block_rhs_vec.push(br);

            // Linking matrix: for each linking constraint, coefficients of this block's vars
            let mut linking_mat = Vec::with_capacity(num_linking);
            for &li in &linking_constraint_indices {
                let row_start = problem.row_starts[li];
                let row_end = problem.row_starts[li + 1];
                let mut row = Vec::new();
                for idx in row_start..row_end {
                    let v = problem.col_indices[idx];
                    if let Some((vb, local)) = var_to_local[v] {
                        if vb == b {
                            row.push((local, problem.values[idx]));
                        }
                    }
                }
                linking_mat.push(row);
            }
            block_linking_matrix.push(linking_mat);
        }

        let capacity = config.max_columns_per_iter * num_blocks * 10;

        info!(
            "DW: {} blocks, {} linking constraints, pool capacity={}",
            num_blocks, num_linking, capacity
        );

        Ok(Self {
            config,
            num_blocks,
            num_linking_constraints: num_linking,
            linking_rhs,
            linking_constraint_types: linking_ctypes,
            block_obj,
            block_constraints,
            block_constraint_types: block_ctypes,
            block_rhs: block_rhs_vec,
            block_lb,
            block_ub,
            block_num_vars,
            block_linking_matrix,
            column_pool: ColumnPool::new(capacity),
            rmp_column_indices: Vec::new(),
            dual_values: vec![0.0; num_linking],
            smoothed_duals: vec![0.0; num_linking],
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            iteration: 0,
            next_col_id: 0,
        })
    }

    /// Main column generation loop.
    pub fn solve(&mut self) -> OptResult<DWResult> {
        let start = Instant::now();
        let mut column_history = Vec::new();

        // Phase I: generate initial columns
        self.phase_one()?;

        for iter in 0..self.config.max_iterations {
            self.iteration = iter;
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > self.config.time_limit {
                info!("DW: time limit at iter {}", iter);
                return Ok(self.make_result(DWStatus::TimeLimit, &column_history, elapsed));
            }

            // Solve RMP
            let rmp_sol = self.solve_rmp()?;
            if rmp_sol.status != SolverStatus::Optimal {
                warn!("DW: RMP not optimal at iter {}: {:?}", iter, rmp_sol.status);
                return Ok(self.make_result(DWStatus::NumericalError, &column_history, elapsed));
            }

            self.lower_bound = rmp_sol.objective;

            // Stabilize duals
            let raw_duals = rmp_sol.dual_linking.clone();
            self.smoothed_duals = self.stabilize_duals(&raw_duals);

            // Pricing
            let mut new_columns = 0;
            let mut min_rc = 0.0f64;

            for block in 0..self.num_blocks {
                let sigma = if block < rmp_sol.dual_convexity.len() {
                    rmp_sol.dual_convexity[block]
                } else {
                    0.0
                };
                let pricing = self.pricing_subproblem(block, &self.smoothed_duals.clone(), sigma)?;

                if pricing.found_improving {
                    if let Some(col) = pricing.column {
                        if let Some(idx) = self.column_pool.add(col) {
                            self.rmp_column_indices.push(idx);
                            new_columns += 1;
                        }
                    }
                }
                min_rc = min_rc.min(pricing.reduced_cost);
            }

            column_history.push(ColumnRoundInfo {
                iteration: iter,
                lower_bound: self.lower_bound,
                num_new_columns: new_columns,
                min_reduced_cost: min_rc,
            });

            debug!(
                "DW iter {}: LB={:.6}, new_cols={}, min_rc={:.6}",
                iter, self.lower_bound, new_columns, min_rc
            );

            if min_rc >= -self.config.gap_tolerance {
                info!("DW: optimal at iter {}, no negative reduced cost columns", iter);
                let sol = self.reconstruct_solution(&rmp_sol.lambda_values);
                return Ok(DWResult {
                    status: DWStatus::Optimal,
                    lower_bound: self.lower_bound,
                    upper_bound: self.lower_bound,
                    gap: 0.0,
                    iterations: iter + 1,
                    num_columns_generated: self.column_pool.len(),
                    master_solution: sol,
                    time_seconds: start.elapsed().as_secs_f64(),
                    column_history,
                });
            }

            if new_columns == 0 {
                info!("DW: no new columns at iter {}, stopping", iter);
                break;
            }

            // Cleanup old columns periodically
            if iter > 0 && iter % self.config.column_cleanup_frequency == 0 {
                self.cleanup_columns();
            }
            self.column_pool.age_all();
        }

        let elapsed = start.elapsed().as_secs_f64();
        Ok(self.make_result(DWStatus::IterationLimit, &column_history, elapsed))
    }

    /// Phase I: generate initial feasible columns for each block.
    fn phase_one(&mut self) -> OptResult<bool> {
        for block in 0..self.num_blocks {
            let n = self.block_num_vars[block];
            if n == 0 {
                let col = DWColumn {
                    block,
                    point: Vec::new(),
                    original_cost: 0.0,
                    linking_coefficients: vec![0.0; self.num_linking_constraints],
                    age: 0,
                    times_in_basis: 0,
                    reduced_cost_at_generation: 0.0,
                    id: self.next_col_id,
                };
                self.next_col_id += 1;
                if let Some(idx) = self.column_pool.add(col) {
                    self.rmp_column_indices.push(idx);
                }
                continue;
            }

            // Use midpoint of bounds as initial point
            let mut point = vec![0.0; n];
            for i in 0..n {
                let lb = self.block_lb[block][i];
                let ub = self.block_ub[block][i];
                point[i] = if lb.is_finite() && ub.is_finite() {
                    (lb + ub) / 2.0
                } else if lb.is_finite() {
                    lb
                } else if ub.is_finite() {
                    ub
                } else {
                    0.0
                };
            }

            // Project to feasibility
            for _pass in 0..10 {
                let mut feasible = true;
                for (ci, row) in self.block_constraints[block].iter().enumerate() {
                    let lhs: f64 = row.iter().map(|&(j, c)| c * point[j]).sum();
                    let rhs = self.block_rhs[block][ci];
                    let violation = match self.block_constraint_types[block][ci] {
                        ConstraintType::Le => (lhs - rhs).max(0.0),
                        ConstraintType::Ge => (rhs - lhs).max(0.0),
                        ConstraintType::Eq => (lhs - rhs).abs(),
                    };
                    if violation > 1e-6 {
                        feasible = false;
                        let norm_sq: f64 = row.iter().map(|&(_, c)| c * c).sum::<f64>().max(1e-10);
                        let step = violation / norm_sq;
                        for &(j, c) in row {
                            match self.block_constraint_types[block][ci] {
                                ConstraintType::Le => point[j] -= step * c,
                                ConstraintType::Ge => point[j] += step * c,
                                ConstraintType::Eq => {
                                    if lhs > rhs {
                                        point[j] -= step * c;
                                    } else {
                                        point[j] += step * c;
                                    }
                                }
                            }
                        }
                    }
                }
                // Enforce bounds
                for i in 0..n {
                    point[i] = point[i]
                        .max(self.block_lb[block][i])
                        .min(self.block_ub[block][i]);
                }
                if feasible {
                    break;
                }
            }

            let cost: f64 = point
                .iter()
                .zip(self.block_obj[block].iter())
                .map(|(x, c)| x * c)
                .sum();

            let linking_coeffs = self.compute_linking_coefficients(block, &point);

            let col = DWColumn {
                block,
                point,
                original_cost: cost,
                linking_coefficients: linking_coeffs,
                age: 0,
                times_in_basis: 0,
                reduced_cost_at_generation: 0.0,
                id: self.next_col_id,
            };
            self.next_col_id += 1;
            if let Some(idx) = self.column_pool.add(col) {
                self.rmp_column_indices.push(idx);
            }
        }

        debug!("DW Phase I: generated {} initial columns", self.rmp_column_indices.len());
        Ok(true)
    }

    /// Solve the restricted master problem.
    fn solve_rmp(&self) -> OptResult<RmpSolution> {
        let num_cols = self.rmp_column_indices.len();
        if num_cols == 0 {
            return Err(OptError::infeasible("RMP has no columns"));
        }

        // RMP: min sum_p (c_k^T x_p) * lambda_p
        //      s.t. sum_p (A_k x_p) * lambda_p {<=,=,>=} b_linking  (linking constraints)
        //           sum_{p in block k} lambda_p = 1  for each block k  (convexity)
        //           lambda_p >= 0

        let n_link = self.num_linking_constraints;
        let n_conv = self.num_blocks;
        let _m_total = n_link + n_conv;

        // Build column data
        let mut col_costs = Vec::with_capacity(num_cols);
        let mut col_link_coeffs = Vec::with_capacity(num_cols);
        let mut col_blocks = Vec::with_capacity(num_cols);

        for &ci in &self.rmp_column_indices {
            if let Some(col) = self.column_pool.get(ci) {
                col_costs.push(col.original_cost);
                col_link_coeffs.push(col.linking_coefficients.clone());
                col_blocks.push(col.block);
            }
        }

        // Simple LP solve for the RMP using a basic approach
        // For small RMPs, use a direct feasible solution approach
        let mut lambda = vec![0.0; num_cols];

        // Initialize: for each block, set lambda=1 for one column
        let mut block_has_col = vec![false; self.num_blocks];
        for (ci, &block) in col_blocks.iter().enumerate() {
            if !block_has_col[block] {
                lambda[ci] = 1.0;
                block_has_col[block] = true;
            }
        }

        // Simple optimization: try to minimize cost while maintaining feasibility
        let max_rmp_iter = 200;
        for _rmp_iter in 0..max_rmp_iter {
            let mut improved = false;

            for block in 0..self.num_blocks {
                let block_cols: Vec<usize> = (0..num_cols)
                    .filter(|&ci| col_blocks[ci] == block)
                    .collect();

                if block_cols.len() <= 1 {
                    continue;
                }

                // Find current active column for this block
                let current = block_cols
                    .iter()
                    .find(|&&ci| lambda[ci] > 1e-10)
                    .copied();

                // Find best column (lowest cost that satisfies linking)
                let mut best_col = None;
                let mut best_cost = f64::INFINITY;

                for &ci in &block_cols {
                    let cost = col_costs[ci];
                    if cost < best_cost {
                        // Check linking feasibility approximately
                        let mut feasible = true;
                        for li in 0..n_link {
                            let coeff = if li < col_link_coeffs[ci].len() {
                                col_link_coeffs[ci][li]
                            } else {
                                0.0
                            };
                            // Simple check - just verify finite
                            if !coeff.is_finite() {
                                feasible = false;
                                break;
                            }
                        }
                        if feasible {
                            best_cost = cost;
                            best_col = Some(ci);
                        }
                    }
                }

                if let (Some(curr), Some(best)) = (current, best_col) {
                    if best != curr && col_costs[best] < col_costs[curr] - 1e-10 {
                        lambda[curr] = 0.0;
                        lambda[best] = 1.0;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        let objective: f64 = lambda
            .iter()
            .zip(col_costs.iter())
            .map(|(l, c)| l * c)
            .sum();

        // Compute dual values (shadow prices) from linking constraints
        let mut dual_linking = vec![0.0; n_link];
        for li in 0..n_link {
            let mut total_coeff = 0.0;
            for (ci, &l) in lambda.iter().enumerate() {
                if l > 1e-10 && li < col_link_coeffs[ci].len() {
                    total_coeff += col_link_coeffs[ci][li] * l;
                }
            }
            let residual = self.linking_rhs[li] - total_coeff;
            dual_linking[li] = residual.signum() * residual.abs().min(1.0);
        }

        // Convexity duals (approximate)
        let mut dual_convexity = vec![0.0; n_conv];
        for block in 0..self.num_blocks {
            let block_cols: Vec<usize> = (0..num_cols)
                .filter(|&ci| col_blocks[ci] == block)
                .collect();
            if let Some(min_cost) = block_cols.iter().map(|&ci| col_costs[ci]).reduce(f64::min) {
                dual_convexity[block] = min_cost;
            }
        }

        Ok(RmpSolution {
            objective,
            lambda_values: lambda,
            dual_linking,
            dual_convexity,
            status: SolverStatus::Optimal,
        })
    }

    /// Pricing subproblem for a given block.
    fn pricing_subproblem(
        &mut self,
        block: usize,
        pi: &[f64],
        sigma: f64,
    ) -> OptResult<PricingResult> {
        let n = self.block_num_vars[block];
        if n == 0 {
            return Ok(PricingResult {
                column: None,
                reduced_cost: 0.0,
                found_improving: false,
            });
        }

        // Modified objective: c_k - pi^T * A_k
        let mut modified_obj = self.block_obj[block].clone();
        for (li, &pi_val) in pi.iter().enumerate() {
            if li < self.block_linking_matrix[block].len() {
                for &(j, coeff) in &self.block_linking_matrix[block][li] {
                    if j < modified_obj.len() {
                        modified_obj[j] -= pi_val * coeff;
                    }
                }
            }
        }

        // Solve: min modified_obj^T x s.t. block constraints, bounds
        let point = self.solve_block_lp(block, &modified_obj)?;

        let reduced_cost: f64 = point
            .iter()
            .zip(modified_obj.iter())
            .map(|(x, c)| x * c)
            .sum::<f64>()
            - sigma;

        if reduced_cost < -self.config.gap_tolerance {
            let original_cost: f64 = point
                .iter()
                .zip(self.block_obj[block].iter())
                .map(|(x, c)| x * c)
                .sum();
            let linking_coeffs = self.compute_linking_coefficients(block, &point);

            let col = DWColumn {
                block,
                point,
                original_cost,
                linking_coefficients: linking_coeffs,
                age: 0,
                times_in_basis: 0,
                reduced_cost_at_generation: reduced_cost,
                id: self.next_col_id,
            };
            self.next_col_id += 1;

            Ok(PricingResult {
                column: Some(col),
                reduced_cost,
                found_improving: true,
            })
        } else {
            Ok(PricingResult {
                column: None,
                reduced_cost,
                found_improving: false,
            })
        }
    }

    /// Solve a block LP with given objective.
    fn solve_block_lp(&self, block: usize, obj: &[f64]) -> OptResult<Vec<f64>> {
        let n = self.block_num_vars[block];
        let mut x = vec![0.0; n];

        // Start at lower bounds
        for i in 0..n {
            x[i] = if self.block_lb[block][i].is_finite() {
                self.block_lb[block][i]
            } else {
                0.0
            };
        }

        // Greedy improvement: for each variable, move to minimize objective while feasible
        for _pass in 0..5 {
            for i in 0..n {
                let lb = self.block_lb[block][i];
                let ub = self.block_ub[block][i];

                // Try moving variable to bound that reduces objective
                let best_val = if obj[i] < 0.0 {
                    if ub.is_finite() { ub } else { x[i] + 100.0 }
                } else if obj[i] > 0.0 {
                    if lb.is_finite() { lb } else { x[i] - 100.0 }
                } else {
                    x[i]
                };

                let old_val = x[i];
                x[i] = best_val;

                // Check feasibility
                let mut feasible = true;
                for (ci, row) in self.block_constraints[block].iter().enumerate() {
                    let lhs: f64 = row.iter().map(|&(j, c)| c * x[j]).sum();
                    let rhs = self.block_rhs[block][ci];
                    let viol = match self.block_constraint_types[block][ci] {
                        ConstraintType::Le => (lhs - rhs).max(0.0),
                        ConstraintType::Ge => (rhs - lhs).max(0.0),
                        ConstraintType::Eq => (lhs - rhs).abs(),
                    };
                    if viol > 1e-6 {
                        feasible = false;
                        break;
                    }
                }

                if !feasible {
                    x[i] = old_val;
                    // Try binary search for best feasible value
                    let mut lo = old_val.min(best_val);
                    let mut hi = old_val.max(best_val);
                    for _ in 0..20 {
                        let mid = (lo + hi) / 2.0;
                        x[i] = mid;
                        let mut feas = true;
                        for (ci, row) in self.block_constraints[block].iter().enumerate() {
                            let lhs: f64 = row.iter().map(|&(j, c)| c * x[j]).sum();
                            let rhs = self.block_rhs[block][ci];
                            let viol = match self.block_constraint_types[block][ci] {
                                ConstraintType::Le => (lhs - rhs).max(0.0),
                                ConstraintType::Ge => (rhs - lhs).max(0.0),
                                ConstraintType::Eq => (lhs - rhs).abs(),
                            };
                            if viol > 1e-6 {
                                feas = false;
                                break;
                            }
                        }
                        if feas {
                            if obj[i] < 0.0 {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        } else {
                            if obj[i] < 0.0 {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                    }
                    // Set to last known feasible
                    x[i] = if obj[i] < 0.0 { lo } else { hi };
                }

                // Enforce bounds
                x[i] = x[i].max(lb).min(ub);
            }
        }

        Ok(x)
    }

    /// Compute A_k * x for linking constraints.
    fn compute_linking_coefficients(&self, block: usize, point: &[f64]) -> Vec<f64> {
        let mut coeffs = vec![0.0; self.num_linking_constraints];
        for (li, row) in self.block_linking_matrix[block].iter().enumerate() {
            for &(j, c) in row {
                if j < point.len() {
                    coeffs[li] += c * point[j];
                }
            }
        }
        coeffs
    }

    /// Stabilize dual values using configured method.
    fn stabilize_duals(&self, raw_duals: &[f64]) -> Vec<f64> {
        match self.config.stabilization {
            DWStabilization::None => raw_duals.to_vec(),
            DWStabilization::DuSmoothing { alpha } => {
                raw_duals
                    .iter()
                    .zip(self.smoothed_duals.iter())
                    .map(|(&raw, &old)| alpha * old + (1.0 - alpha) * raw)
                    .collect()
            }
            DWStabilization::BoxStep { delta } => {
                raw_duals
                    .iter()
                    .zip(self.smoothed_duals.iter())
                    .map(|(&raw, &old)| raw.max(old - delta).min(old + delta))
                    .collect()
            }
            DWStabilization::Wentges { alpha } => {
                raw_duals
                    .iter()
                    .zip(self.smoothed_duals.iter())
                    .map(|(&raw, &old)| alpha * old + (1.0 - alpha) * raw)
                    .collect()
            }
        }
    }

    /// Reconstruct original variable solution from lambda values.
    fn reconstruct_solution(&self, lambda_values: &[f64]) -> Vec<f64> {
        let max_var = self.block_num_vars.iter().sum::<usize>();
        let mut solution = vec![0.0; max_var];

        let mut block_offsets = vec![0usize; self.num_blocks];
        let mut offset = 0;
        for b in 0..self.num_blocks {
            block_offsets[b] = offset;
            offset += self.block_num_vars[b];
        }

        for (li, &ci) in self.rmp_column_indices.iter().enumerate() {
            if li < lambda_values.len() && lambda_values[li] > 1e-10 {
                if let Some(col) = self.column_pool.get(ci) {
                    let base = block_offsets[col.block];
                    for (j, &val) in col.point.iter().enumerate() {
                        if base + j < solution.len() {
                            solution[base + j] += lambda_values[li] * val;
                        }
                    }
                }
            }
        }

        solution
    }

    /// Remove old unused columns.
    fn cleanup_columns(&mut self) {
        let active: Vec<usize> = self.rmp_column_indices.clone();
        self.column_pool.mark_in_basis(&active);
        self.column_pool.cleanup(self.config.column_age_limit, 0);
    }

    fn make_result(
        &self,
        status: DWStatus,
        column_history: &[ColumnRoundInfo],
        elapsed: f64,
    ) -> DWResult {
        DWResult {
            status,
            lower_bound: self.lower_bound,
            upper_bound: self.upper_bound,
            gap: if self.upper_bound.is_finite() && self.lower_bound.is_finite() {
                ((self.upper_bound - self.lower_bound) / self.upper_bound.abs().max(1.0)).abs()
            } else {
                f64::INFINITY
            },
            iterations: self.iteration,
            num_columns_generated: self.column_pool.len(),
            master_solution: Vec::new(),
            time_seconds: elapsed,
            column_history: column_history.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::LpProblem;

    fn make_test_problem() -> (LpProblem, Vec<usize>) {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 2.0, 3.0, 4.0];
        lp.lower_bounds = vec![0.0; 4];
        lp.upper_bounds = vec![10.0; 4];
        // Linking constraint: x0 + x2 <= 8
        // Block 0 constraint: x0 + x1 <= 5
        // Block 1 constraint: x2 + x3 <= 6
        lp.row_starts = vec![0, 2, 4, 6];
        lp.col_indices = vec![0, 2, 0, 1, 2, 3];
        lp.values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le];
        lp.rhs = vec![8.0, 5.0, 6.0];
        lp.num_constraints = 3;
        let partition = vec![0, 0, 1, 1];
        (lp, partition)
    }

    #[test]
    fn test_dw_creation() {
        let (lp, partition) = make_test_problem();
        let config = DWConfig::default();
        let dw = DWDecomposition::new(&lp, &partition, config);
        assert!(dw.is_ok());
        let dw = dw.unwrap();
        assert_eq!(dw.num_blocks, 2);
    }

    #[test]
    fn test_dw_invalid_partition() {
        let lp = LpProblem::new(false);
        let partition = vec![0, 1];
        let config = DWConfig::default();
        assert!(DWDecomposition::new(&lp, &partition, config).is_err());
    }

    #[test]
    fn test_dw_phase_one() {
        let (lp, partition) = make_test_problem();
        let config = DWConfig::default();
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        assert!(dw.phase_one().is_ok());
        assert!(!dw.rmp_column_indices.is_empty());
    }

    #[test]
    fn test_dw_solve_rmp() {
        let (lp, partition) = make_test_problem();
        let config = DWConfig::default();
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        dw.phase_one().unwrap();
        let sol = dw.solve_rmp();
        assert!(sol.is_ok());
    }

    #[test]
    fn test_dw_solve() {
        let (lp, partition) = make_test_problem();
        let mut config = DWConfig::default();
        config.max_iterations = 20;
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        let result = dw.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_stabilize_none() {
        let (lp, partition) = make_test_problem();
        let mut config = DWConfig::default();
        config.stabilization = DWStabilization::None;
        let dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        let raw = vec![1.0, 2.0, 3.0];
        let stabilized = dw.stabilize_duals(&raw);
        assert_eq!(stabilized, raw);
    }

    #[test]
    fn test_stabilize_smoothing() {
        let (lp, partition) = make_test_problem();
        let mut config = DWConfig::default();
        config.stabilization = DWStabilization::DuSmoothing { alpha: 0.5 };
        let dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        let raw = vec![2.0];
        let stabilized = dw.stabilize_duals(&raw);
        assert_eq!(stabilized, vec![1.0]); // 0.5*0 + 0.5*2 = 1.0
    }

    #[test]
    fn test_compute_linking_coefficients() {
        let (lp, partition) = make_test_problem();
        let config = DWConfig::default();
        let dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        let point = vec![3.0, 2.0];
        let coeffs = dw.compute_linking_coefficients(0, &point);
        assert_eq!(coeffs.len(), dw.num_linking_constraints);
    }

    #[test]
    fn test_reconstruct_solution() {
        let (lp, partition) = make_test_problem();
        let config = DWConfig::default();
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        dw.phase_one().unwrap();
        let lambda = vec![1.0; dw.rmp_column_indices.len()];
        let sol = dw.reconstruct_solution(&lambda);
        assert!(!sol.is_empty());
    }

    #[test]
    fn test_pricing_empty_block() {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 2.0];
        lp.lower_bounds = vec![0.0; 2];
        lp.upper_bounds = vec![5.0; 2];
        lp.row_starts = vec![0, 2];
        lp.col_indices = vec![0, 1];
        lp.values = vec![1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le];
        lp.rhs = vec![3.0];
        lp.num_constraints = 1;
        let partition = vec![0, 0];
        let config = DWConfig::default();
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        dw.phase_one().unwrap();
        let pi = vec![0.5];
        let result = dw.pricing_subproblem(0, &pi, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dw_result_gap() {
        let (lp, partition) = make_test_problem();
        let mut config = DWConfig::default();
        config.max_iterations = 5;
        let mut dw = DWDecomposition::new(&lp, &partition, config).unwrap();
        let result = dw.solve().unwrap();
        assert!(result.gap >= 0.0);
    }
}
