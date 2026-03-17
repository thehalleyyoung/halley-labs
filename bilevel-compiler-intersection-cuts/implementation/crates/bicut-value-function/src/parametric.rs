//! Parametric LP solver for the lower-level value function.
//!
//! Solves LPs parametrically as the right-hand side varies with x:
//!   min c^T y  s.t.  Ay ≤ b + Bx,  y ≥ 0
//!
//! Tracks basis changes, enumerates optimal bases, and performs sensitivity analysis.

use std::collections::{HashMap, HashSet};

use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BasisStatus, BilevelProblem, LpProblem, LpSolution, LpStatus};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Basis representation
// ---------------------------------------------------------------------------

/// Identifies a basis by which variables/slacks are basic.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BasisInfo {
    /// Indices of basic variables (original + slack).
    pub basic_indices: Vec<usize>,
    /// Basis status per original variable.
    pub var_status: Vec<BasisStatus>,
    /// Whether the basis is degenerate.
    pub is_degenerate: bool,
    /// Whether the basis is dual feasible.
    pub is_dual_feasible: bool,
    /// Whether the basis is primal feasible.
    pub is_primal_feasible: bool,
}

impl BasisInfo {
    pub fn size(&self) -> usize {
        self.basic_indices.len()
    }

    pub fn same_basis(&self, other: &BasisInfo) -> bool {
        if self.basic_indices.len() != other.basic_indices.len() {
            return false;
        }
        let s1: HashSet<_> = self.basic_indices.iter().collect();
        let s2: HashSet<_> = other.basic_indices.iter().collect();
        s1 == s2
    }
}

/// Sensitivity range for a single RHS coefficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityRange {
    pub constraint_index: usize,
    pub lower: f64,
    pub upper: f64,
    pub current_value: f64,
    pub shadow_price: f64,
}

/// Result from a parametric LP solve along a single direction.
#[derive(Debug, Clone)]
pub struct ParametricResult {
    pub breakpoints: Vec<f64>,
    pub segment_bases: Vec<BasisInfo>,
    pub segment_slopes: Vec<(f64, f64)>,
    pub total_pivots: u64,
}

/// Result from a full multi-dimensional parametric analysis.
#[derive(Debug, Clone)]
pub struct FullParametricAnalysis {
    pub dimension_results: Vec<ParametricResult>,
    pub unique_bases: Vec<BasisInfo>,
    pub sensitivity_ranges: Vec<SensitivityRange>,
}

// ---------------------------------------------------------------------------
// Parametric solver
// ---------------------------------------------------------------------------

/// Parametric LP solver with basis tracking.
pub struct ParametricSolver {
    problem: BilevelProblem,
    solver: SimplexSolver,
    tolerance: f64,
    max_bases: usize,
}

impl ParametricSolver {
    pub fn new(problem: BilevelProblem) -> Self {
        Self {
            problem,
            solver: SimplexSolver::default(),
            tolerance: TOLERANCE,
            max_bases: 1000,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_max_bases(mut self, max: usize) -> Self {
        self.max_bases = max;
        self
    }

    /// Solve the lower-level LP at a specific x and extract basis information.
    pub fn solve_at(&self, x: &[f64]) -> VFResult<(LpSolution, BasisInfo)> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        let basis = self.extract_basis(&sol, &lp);
        Ok((sol, basis))
    }

    fn extract_basis(&self, sol: &LpSolution, lp: &LpProblem) -> BasisInfo {
        let n = lp.num_vars;
        let m = lp.num_constraints;

        let mut basic_indices = Vec::new();
        let mut var_status = vec![BasisStatus::NonBasicLower; n];

        for (j, &bs) in sol.basis.iter().enumerate() {
            if j < n {
                var_status[j] = bs;
                if bs == BasisStatus::Basic {
                    basic_indices.push(j);
                }
            }
        }

        let a_dense = lp.a_matrix.to_dense();
        for i in 0..m {
            let ay: f64 = (0..n.min(sol.primal.len()))
                .map(|j| a_dense[(i, j)] * sol.primal[j])
                .sum();
            let slack = lp.b_rhs[i] - ay;
            if slack.abs() > self.tolerance {
                basic_indices.push(n + i);
            }
        }

        let is_degenerate = sol.primal.iter().enumerate().any(|(j, &v)| {
            sol.basis.get(j) == Some(&BasisStatus::Basic) && v.abs() < self.tolerance
        });

        let is_dual_feasible = sol.dual.iter().all(|&d| d >= -self.tolerance);
        let is_primal_feasible = sol.primal.iter().all(|&v| v >= -self.tolerance);

        BasisInfo {
            basic_indices,
            var_status,
            is_degenerate,
            is_dual_feasible,
            is_primal_feasible,
        }
    }

    /// Parametric simplex along a single direction in x-space.
    pub fn parametric_solve_direction(
        &self,
        x0: &[f64],
        direction: &[f64],
        t_min: f64,
        t_max: f64,
        num_steps: usize,
    ) -> VFResult<ParametricResult> {
        if x0.len() != direction.len() || x0.len() != self.problem.num_upper_vars {
            return Err(VFError::DimensionMismatch {
                expected: self.problem.num_upper_vars,
                got: x0.len(),
            });
        }

        let mut breakpoints = Vec::new();
        let mut segment_bases = Vec::new();
        let mut segment_slopes = Vec::new();
        let mut total_pivots = 0u64;

        let step = if num_steps > 0 {
            (t_max - t_min) / num_steps as f64
        } else {
            t_max - t_min
        };
        let mut prev_basis: Option<BasisInfo> = None;
        let mut segment_start_t = t_min;
        let mut segment_start_val = 0.0f64;

        for k in 0..=num_steps {
            let t = t_min + k as f64 * step;
            let x_t: Vec<f64> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&a, &d)| a + t * d)
                .collect();

            let result = self.solve_at(&x_t);
            let (sol, basis) = match result {
                Ok(pair) => pair,
                Err(_) => continue,
            };

            total_pivots += sol.iterations;

            match &prev_basis {
                None => {
                    segment_start_t = t;
                    segment_start_val = sol.objective;
                    prev_basis = Some(basis);
                }
                Some(pb) => {
                    if !pb.same_basis(&basis) {
                        let bp_t = if k > 0 {
                            t_min + (k as f64 - 0.5) * step
                        } else {
                            t
                        };
                        breakpoints.push(bp_t);

                        let dt = bp_t - segment_start_t;
                        let slope = if dt.abs() > self.tolerance {
                            (sol.objective - segment_start_val) / dt
                        } else {
                            0.0
                        };
                        let intercept = segment_start_val - slope * segment_start_t;
                        segment_slopes.push((slope, intercept));
                        segment_bases.push(pb.clone());

                        segment_start_t = bp_t;
                        segment_start_val = sol.objective;
                        prev_basis = Some(basis);
                    } else if k == num_steps {
                        let dt = t - segment_start_t;
                        let slope = if dt.abs() > self.tolerance {
                            (sol.objective - segment_start_val) / dt
                        } else {
                            0.0
                        };
                        let intercept = segment_start_val - slope * segment_start_t;
                        segment_slopes.push((slope, intercept));
                        segment_bases.push(basis);
                    }
                }
            }
        }

        if segment_bases.is_empty() {
            if let Some(pb) = prev_basis {
                segment_bases.push(pb);
                segment_slopes.push((0.0, segment_start_val));
            }
        }

        Ok(ParametricResult {
            breakpoints,
            segment_bases,
            segment_slopes,
            total_pivots,
        })
    }

    /// Enumerate all optimal bases reachable by varying x in the given box.
    pub fn enumerate_bases(
        &self,
        x_lower: &[f64],
        x_upper: &[f64],
        samples_per_dim: usize,
    ) -> VFResult<Vec<BasisInfo>> {
        let nx = self.problem.num_upper_vars;
        if x_lower.len() != nx || x_upper.len() != nx {
            return Err(VFError::DimensionMismatch {
                expected: nx,
                got: x_lower.len(),
            });
        }

        let mut seen_bases: Vec<BasisInfo> = Vec::new();
        let total_samples = samples_per_dim.pow(nx.min(10) as u32).min(self.max_bases);
        let points = generate_grid_points(x_lower, x_upper, samples_per_dim, total_samples);

        for x in &points {
            if seen_bases.len() >= self.max_bases {
                break;
            }

            match self.solve_at(x) {
                Ok((_sol, basis)) => {
                    let is_new = !seen_bases.iter().any(|b| b.same_basis(&basis));
                    if is_new {
                        seen_bases.push(basis);
                    }
                }
                Err(_) => continue,
            }
        }

        if seen_bases.is_empty() {
            return Err(VFError::NoCriticalRegions);
        }

        Ok(seen_bases)
    }

    /// Compute sensitivity ranges for all RHS coefficients at a given x.
    pub fn sensitivity_analysis(&self, x: &[f64]) -> VFResult<Vec<SensitivityRange>> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        let m = lp.num_constraints;
        let mut ranges = Vec::with_capacity(m);

        for i in 0..m {
            let shadow_price = if i < sol.dual.len() { sol.dual[i] } else { 0.0 };
            let (lower, upper) = self.compute_rhs_range(&lp, &sol, i);

            ranges.push(SensitivityRange {
                constraint_index: i,
                lower,
                upper,
                current_value: lp.b_rhs[i],
                shadow_price,
            });
        }

        Ok(ranges)
    }

    fn compute_rhs_range(
        &self,
        lp: &LpProblem,
        sol: &LpSolution,
        constraint_idx: usize,
    ) -> (f64, f64) {
        let current_b = lp.b_rhs[constraint_idx];
        let base_basis: Vec<BasisStatus> = sol.basis.clone();

        // Binary search downward
        let mut lo = current_b - 100.0;
        let mut hi = current_b;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let mut test_lp = lp.clone();
            test_lp.b_rhs[constraint_idx] = mid;
            match self.solver.solve(&test_lp) {
                Ok(test_sol) if test_sol.status == LpStatus::Optimal => {
                    if self.same_basis_status(&test_sol.basis, &base_basis) {
                        hi = mid;
                        lo = mid - (current_b - mid).abs().max(1.0);
                    } else {
                        lo = mid;
                    }
                }
                _ => lo = mid,
            }
            if (hi - lo).abs() < self.tolerance {
                break;
            }
        }
        let lower_bound = lo;

        // Binary search upward
        let mut lo = current_b;
        let mut hi = current_b + 100.0;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let mut test_lp = lp.clone();
            test_lp.b_rhs[constraint_idx] = mid;
            match self.solver.solve(&test_lp) {
                Ok(test_sol) if test_sol.status == LpStatus::Optimal => {
                    if self.same_basis_status(&test_sol.basis, &base_basis) {
                        lo = mid;
                        hi = mid + (mid - current_b).abs().max(1.0);
                    } else {
                        hi = mid;
                    }
                }
                _ => hi = mid,
            }
            if (hi - lo).abs() < self.tolerance {
                break;
            }
        }
        let upper_bound = hi;

        (lower_bound, upper_bound)
    }

    fn same_basis_status(&self, a: &[BasisStatus], b: &[BasisStatus]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    /// Full parametric analysis along each coordinate direction.
    pub fn full_parametric_analysis(
        &self,
        x0: &[f64],
        box_radius: f64,
        steps_per_dim: usize,
    ) -> VFResult<FullParametricAnalysis> {
        let nx = self.problem.num_upper_vars;
        let mut dimension_results = Vec::with_capacity(nx);
        let mut all_bases: Vec<BasisInfo> = Vec::new();

        for d in 0..nx {
            let mut direction = vec![0.0; nx];
            direction[d] = 1.0;

            let result = self.parametric_solve_direction(
                x0,
                &direction,
                -box_radius,
                box_radius,
                steps_per_dim,
            )?;

            for basis in &result.segment_bases {
                let is_new = !all_bases.iter().any(|b| b.same_basis(basis));
                if is_new {
                    all_bases.push(basis.clone());
                }
            }

            dimension_results.push(result);
        }

        let sensitivity_ranges = self.sensitivity_analysis(x0)?;

        Ok(FullParametricAnalysis {
            dimension_results,
            unique_bases: all_bases,
            sensitivity_ranges,
        })
    }

    /// Compute the basis inverse B^{-1} for the given basis indices.
    pub fn compute_basis_inverse(
        &self,
        x: &[f64],
        basis_indices: &[usize],
    ) -> VFResult<DMatrix<f64>> {
        let lp = self.problem.lower_level_lp(x);
        let n = lp.num_vars;
        let m = lp.num_constraints;
        let a_dense = lp.a_matrix.to_dense();

        if basis_indices.len() != m {
            return Err(VFError::DimensionMismatch {
                expected: m,
                got: basis_indices.len(),
            });
        }

        let mut b_mat = DMatrix::zeros(m, m);
        for (col, &idx) in basis_indices.iter().enumerate() {
            if idx < n {
                for row in 0..m {
                    b_mat[(row, col)] = a_dense[(row, idx)];
                }
            } else {
                let slack_row = idx - n;
                if slack_row < m {
                    b_mat[(slack_row, col)] = 1.0;
                }
            }
        }

        b_mat
            .try_inverse()
            .ok_or_else(|| VFError::NumericalError("Singular basis matrix".into()))
    }

    /// Compute reduced costs for non-basic variables given a basis.
    pub fn compute_reduced_costs(&self, x: &[f64], basis_indices: &[usize]) -> VFResult<Vec<f64>> {
        let lp = self.problem.lower_level_lp(x);
        let n = lp.num_vars;
        let m = lp.num_constraints;
        let a_dense = lp.a_matrix.to_dense();

        let b_inv = self.compute_basis_inverse(x, basis_indices)?;

        let c_b = DVector::from_iterator(
            m,
            basis_indices
                .iter()
                .map(|&idx| if idx < n { lp.c[idx] } else { 0.0 }),
        );

        let pi = &b_inv.transpose() * &c_b;

        let total = n + m;
        let mut reduced_costs = Vec::with_capacity(total);

        for j in 0..total {
            let c_j = if j < n { lp.c[j] } else { 0.0 };
            let mut a_j = DVector::zeros(m);
            if j < n {
                for i in 0..m {
                    a_j[i] = a_dense[(i, j)];
                }
            } else {
                let slack_row = j - n;
                if slack_row < m {
                    a_j[slack_row] = 1.0;
                }
            }

            let rc = c_j - pi.dot(&a_j);
            reduced_costs.push(rc);
        }

        Ok(reduced_costs)
    }

    /// Determine the leaving variable for a given entering variable.
    pub fn ratio_test(
        &self,
        x: &[f64],
        basis_indices: &[usize],
        entering_idx: usize,
    ) -> VFResult<Option<(usize, f64)>> {
        let lp = self.problem.lower_level_lp(x);
        let n = lp.num_vars;
        let m = lp.num_constraints;
        let a_dense = lp.a_matrix.to_dense();

        let b_inv = self.compute_basis_inverse(x, basis_indices)?;

        let mut a_entering = DVector::zeros(m);
        if entering_idx < n {
            for i in 0..m {
                a_entering[i] = a_dense[(i, entering_idx)];
            }
        } else {
            let slack_row = entering_idx - n;
            if slack_row < m {
                a_entering[slack_row] = 1.0;
            }
        }

        let d = &b_inv * &a_entering;
        let b_rhs = DVector::from_column_slice(&lp.b_rhs);
        let x_b = &b_inv * &b_rhs;

        let mut min_ratio = f64::INFINITY;
        let mut leaving_pos = None;

        for i in 0..m {
            if d[i] > self.tolerance {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio - self.tolerance {
                    min_ratio = ratio;
                    leaving_pos = Some(i);
                }
            }
        }

        match leaving_pos {
            Some(pos) => Ok(Some((basis_indices[pos], min_ratio))),
            None => Ok(None),
        }
    }

    /// Perform one parametric simplex pivot.
    pub fn parametric_pivot(
        &self,
        x: &[f64],
        basis_indices: &mut Vec<usize>,
        entering_idx: usize,
    ) -> VFResult<Option<usize>> {
        let result = self.ratio_test(x, basis_indices, entering_idx)?;

        match result {
            Some((leaving_idx, _ratio)) => {
                if let Some(pos) = basis_indices.iter().position(|&b| b == leaving_idx) {
                    basis_indices[pos] = entering_idx;
                    Ok(Some(leaving_idx))
                } else {
                    Err(VFError::NumericalError(
                        "Leaving variable not in basis".into(),
                    ))
                }
            }
            None => Ok(None),
        }
    }

    /// Track how the basis changes as we move from x0 to x1.
    pub fn track_basis_path(
        &self,
        x0: &[f64],
        x1: &[f64],
        num_steps: usize,
    ) -> VFResult<Vec<(f64, BasisInfo)>> {
        let direction: Vec<f64> = x0.iter().zip(x1.iter()).map(|(a, b)| b - a).collect();
        let mut path = Vec::new();

        for k in 0..=num_steps {
            let t = k as f64 / num_steps.max(1) as f64;
            let x_t: Vec<f64> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&a, &d)| a + t * d)
                .collect();

            match self.solve_at(&x_t) {
                Ok((_sol, basis)) => {
                    let should_add = path.is_empty()
                        || !path
                            .last()
                            .map(|(_, b): &(f64, BasisInfo)| b.same_basis(&basis))
                            .unwrap_or(false);
                    if should_add {
                        path.push((t, basis));
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(path)
    }

    pub fn problem(&self) -> &BilevelProblem {
        &self.problem
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn generate_grid_points(
    lb: &[f64],
    ub: &[f64],
    samples_per_dim: usize,
    max_total: usize,
) -> Vec<Vec<f64>> {
    let n = lb.len();
    if n == 0 {
        return vec![vec![]];
    }

    let mut points = Vec::new();
    let mut indices = vec![0usize; n];

    loop {
        if points.len() >= max_total {
            break;
        }

        let point: Vec<f64> = (0..n)
            .map(|d| {
                if samples_per_dim <= 1 {
                    (lb[d] + ub[d]) / 2.0
                } else {
                    lb[d] + (ub[d] - lb[d]) * indices[d] as f64 / (samples_per_dim - 1) as f64
                }
            })
            .collect();
        points.push(point);

        let mut carry = true;
        for d in (0..n).rev() {
            if carry {
                indices[d] += 1;
                if indices[d] >= samples_per_dim {
                    indices[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }

    points
}

/// Compute the objective value for a given basis at parameter point x.
pub fn basis_objective(
    problem: &BilevelProblem,
    basis_indices: &[usize],
    x: &[f64],
) -> VFResult<f64> {
    let lp = problem.lower_level_lp(x);
    let n = lp.num_vars;
    let m = lp.num_constraints;
    let a_dense = lp.a_matrix.to_dense();

    let mut b_mat = DMatrix::zeros(m, m);
    for (col, &idx) in basis_indices.iter().enumerate() {
        if idx < n {
            for row in 0..m {
                b_mat[(row, col)] = a_dense[(row, idx)];
            }
        } else {
            let slack_row = idx - n;
            if slack_row < m {
                b_mat[(slack_row, col)] = 1.0;
            }
        }
    }

    let b_inv = b_mat
        .try_inverse()
        .ok_or_else(|| VFError::NumericalError("Singular basis".into()))?;

    let rhs = DVector::from_column_slice(&lp.b_rhs);
    let x_b = &b_inv * &rhs;

    let mut obj = 0.0;
    for (col, &idx) in basis_indices.iter().enumerate() {
        let c_j = if idx < n { lp.c[idx] } else { 0.0 };
        obj += c_j * x_b[col];
    }

    Ok(obj)
}

/// Determine the affine expression of the objective as a function of x
/// for a fixed basis.
pub fn basis_affine_expression(
    problem: &BilevelProblem,
    basis_indices: &[usize],
    x_ref: &[f64],
) -> VFResult<(Vec<f64>, f64)> {
    let nx = problem.num_upper_vars;
    let val_ref = basis_objective(problem, basis_indices, x_ref)?;

    let step = 1e-6;
    let mut gradient = vec![0.0; nx];

    for d in 0..nx {
        let mut x_plus = x_ref.to_vec();
        x_plus[d] += step;
        let val_plus = basis_objective(problem, basis_indices, &x_plus)?;
        gradient[d] = (val_plus - val_ref) / step;
    }

    let constant = val_ref
        - gradient
            .iter()
            .zip(x_ref.iter())
            .map(|(g, x)| g * x)
            .sum::<f64>();

    Ok((gradient, constant))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn test_bilevel() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 1);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 0, 1.0);

        let mut linking_b = SparseMatrix::new(2, 1);
        linking_b.add_entry(0, 0, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![2.0, 3.0],
            lower_linking_b: linking_b,
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_solve_at() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let (sol, basis) = solver.solve_at(&[0.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(sol.objective >= -TOLERANCE);
        assert!(!basis.basic_indices.is_empty());
    }

    #[test]
    fn test_parametric_direction() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let result = solver
            .parametric_solve_direction(&[0.0], &[1.0], -2.0, 2.0, 20)
            .unwrap();
        assert!(!result.segment_bases.is_empty());
        assert!(!result.segment_slopes.is_empty());
    }

    #[test]
    fn test_enumerate_bases() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let bases = solver.enumerate_bases(&[-2.0], &[2.0], 10).unwrap();
        assert!(!bases.is_empty());
    }

    #[test]
    fn test_sensitivity_analysis() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let ranges = solver.sensitivity_analysis(&[0.0]).unwrap();
        assert_eq!(ranges.len(), 2);
        for r in &ranges {
            assert!(r.lower <= r.current_value + TOLERANCE);
        }
    }

    #[test]
    fn test_basis_same() {
        let b1 = BasisInfo {
            basic_indices: vec![0, 2],
            var_status: vec![BasisStatus::Basic],
            is_degenerate: false,
            is_dual_feasible: true,
            is_primal_feasible: true,
        };
        let b2 = BasisInfo {
            basic_indices: vec![2, 0],
            var_status: vec![BasisStatus::Basic],
            is_degenerate: false,
            is_dual_feasible: true,
            is_primal_feasible: true,
        };
        assert!(b1.same_basis(&b2));
    }

    #[test]
    fn test_full_parametric_analysis() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let analysis = solver.full_parametric_analysis(&[0.0], 2.0, 10).unwrap();
        assert!(!analysis.unique_bases.is_empty());
        assert!(!analysis.sensitivity_ranges.is_empty());
    }

    #[test]
    fn test_track_basis_path() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let path = solver.track_basis_path(&[-1.0], &[1.0], 10).unwrap();
        assert!(!path.is_empty());
        assert!(path[0].0 >= 0.0);
        assert!(path[0].0 <= 1.0);
    }

    #[test]
    fn test_generate_grid_points() {
        let pts = generate_grid_points(&[0.0, 0.0], &[1.0, 1.0], 3, 100);
        assert_eq!(pts.len(), 9);
    }

    #[test]
    fn test_compute_reduced_costs() {
        let problem = test_bilevel();
        let solver = ParametricSolver::new(problem);
        let (_sol, basis) = solver.solve_at(&[0.0]).unwrap();
        let _ = solver.compute_reduced_costs(&[0.0], &basis.basic_indices);
    }
}
