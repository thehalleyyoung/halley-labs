//! Dual analysis for the value function.
//!
//! Tracks dual multipliers, computes shadow prices, characterizes the dual
//! feasible region, performs complementary slackness analysis, and assesses
//! the stability of optimal dual solutions.

use std::collections::HashMap;

use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BasisStatus, BilevelProblem, LpStatus, Polyhedron};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::oracle::ValueFunctionOracle;
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Shadow price info
// ---------------------------------------------------------------------------

/// Shadow price information for each constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowPriceInfo {
    /// Shadow prices (dual multipliers π).
    pub prices: Vec<f64>,
    /// Whether each shadow price is valid (i.e., the constraint is active).
    pub is_active: Vec<bool>,
    /// Validity flag.
    pub valid: bool,
    /// The x-point at which these prices were computed.
    pub at_point: Vec<f64>,
}

impl ShadowPriceInfo {
    /// Get the shadow price for constraint i.
    pub fn price(&self, i: usize) -> f64 {
        self.prices.get(i).copied().unwrap_or(0.0)
    }

    /// Number of active constraints.
    pub fn num_active(&self) -> usize {
        self.is_active.iter().filter(|&&a| a).count()
    }

    /// Indices of active constraints.
    pub fn active_indices(&self) -> Vec<usize> {
        self.is_active
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(|(i, _)| i)
            .collect()
    }

    /// Sum of all shadow prices.
    pub fn total_sensitivity(&self) -> f64 {
        self.prices.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Dual stability info
// ---------------------------------------------------------------------------

/// Information about the stability of the optimal dual solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualStabilityInfo {
    /// Whether the dual solution is stable (unique and non-degenerate).
    pub stable: bool,
    /// Condition number of the basis matrix.
    pub condition_number: f64,
    /// Minimum reduced cost (smaller = closer to degeneracy).
    pub min_reduced_cost: f64,
    /// Number of near-degenerate variables.
    pub near_degenerate_count: usize,
    /// Dual perturbation tolerance (max perturbation keeping basis optimal).
    pub perturbation_tolerance: f64,
}

impl DualStabilityInfo {
    /// Is the dual solution numerically well-conditioned?
    pub fn is_well_conditioned(&self, threshold: f64) -> bool {
        self.condition_number < threshold
    }

    /// Stability score ∈ [0, 1] (higher is better).
    pub fn stability_score(&self) -> f64 {
        let cond_score = 1.0 / (1.0 + self.condition_number.log10().max(0.0));
        let rc_score = self.min_reduced_cost.min(1.0).max(0.0);
        (cond_score + rc_score) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Complementary slackness result
// ---------------------------------------------------------------------------

/// Result of complementary slackness verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplementarySlacknessResult {
    /// Whether CS conditions hold.
    pub holds: bool,
    /// Maximum CS violation.
    pub max_violation: f64,
    /// Per-constraint violation.
    pub constraint_violations: Vec<f64>,
    /// Per-variable violation.
    pub variable_violations: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Dual feasible region
// ---------------------------------------------------------------------------

/// Characterization of the dual feasible region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualFeasibleRegion {
    /// The dual polyhedron: { π : A^T π ≥ c, π ≥ 0 }.
    pub polyhedron: Polyhedron,
    /// Extreme points (vertices) of the dual polyhedron.
    pub vertices: Vec<Vec<f64>>,
    /// Extreme rays (if unbounded).
    pub extreme_rays: Vec<Vec<f64>>,
    /// Whether the region is bounded.
    pub is_bounded: bool,
}

// ---------------------------------------------------------------------------
// Dual analyzer
// ---------------------------------------------------------------------------

/// Analyzes dual properties of the value function.
pub struct DualAnalyzer {
    problem: BilevelProblem,
    solver: SimplexSolver,
    tolerance: f64,
}

impl DualAnalyzer {
    pub fn new(problem: BilevelProblem) -> Self {
        Self {
            problem,
            solver: SimplexSolver::default(),
            tolerance: TOLERANCE,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Compute shadow prices at a given x.
    pub fn shadow_prices(&self, x: &[f64]) -> VFResult<ShadowPriceInfo> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        let m = lp.num_constraints;
        let n = lp.num_vars;
        let a_dense = lp.a_matrix.to_dense();

        // Determine which constraints are active
        let mut is_active = vec![false; m];
        for i in 0..m {
            let ay: f64 = (0..n.min(sol.primal.len()))
                .map(|j| a_dense[(i, j)] * sol.primal[j])
                .sum();
            let slack = lp.b_rhs[i] - ay;
            is_active[i] = slack.abs() < self.tolerance * 10.0;
        }

        Ok(ShadowPriceInfo {
            prices: sol.dual.clone(),
            is_active,
            valid: true,
            at_point: x.to_vec(),
        })
    }

    /// Track shadow prices as x varies along a direction.
    pub fn track_shadow_prices(
        &self,
        x0: &[f64],
        direction: &[f64],
        t_min: f64,
        t_max: f64,
        num_steps: usize,
    ) -> VFResult<Vec<(f64, ShadowPriceInfo)>> {
        let mut results = Vec::new();

        for k in 0..=num_steps {
            let t = t_min + (t_max - t_min) * k as f64 / num_steps.max(1) as f64;
            let x: Vec<f64> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&a, &d)| a + t * d)
                .collect();

            match self.shadow_prices(&x) {
                Ok(sp) => results.push((t, sp)),
                Err(_) => continue,
            }
        }

        Ok(results)
    }

    /// Verify complementary slackness conditions at a point.
    pub fn verify_complementary_slackness(
        &self,
        x: &[f64],
    ) -> VFResult<ComplementarySlacknessResult> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        let m = lp.num_constraints;
        let n = lp.num_vars;
        let a_dense = lp.a_matrix.to_dense();

        let mut constraint_violations = Vec::with_capacity(m);
        let mut max_violation = 0.0f64;

        for i in 0..m {
            let ay: f64 = (0..n.min(sol.primal.len()))
                .map(|j| a_dense[(i, j)] * sol.primal[j])
                .sum();
            let slack = lp.b_rhs[i] - ay;
            let dual_val = sol.dual.get(i).copied().unwrap_or(0.0);
            let violation = (dual_val * slack).abs();
            constraint_violations.push(violation);
            max_violation = max_violation.max(violation);
        }

        let mut variable_violations = Vec::with_capacity(n);
        for j in 0..n.min(sol.primal.len()) {
            let reduced_cost = lp.c[j]
                - (0..m)
                    .map(|i| {
                        let d = sol.dual.get(i).copied().unwrap_or(0.0);
                        a_dense[(i, j)] * d
                    })
                    .sum::<f64>();
            let violation = (sol.primal[j] * reduced_cost).abs();
            variable_violations.push(violation);
            max_violation = max_violation.max(violation);
        }

        let holds = max_violation < self.tolerance * 100.0;

        Ok(ComplementarySlacknessResult {
            holds,
            max_violation,
            constraint_violations,
            variable_violations,
        })
    }

    /// Assess dual stability at x (basis condition number, reduced cost gap).
    pub fn dual_stability(&self, x: &[f64]) -> VFResult<DualStabilityInfo> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        let n = lp.num_vars;
        let m = lp.num_constraints;
        let a_dense = lp.a_matrix.to_dense();

        // Find basic variables
        let mut basic_cols: Vec<usize> = Vec::new();
        for (j, &bs) in sol.basis.iter().enumerate() {
            if bs == BasisStatus::Basic && j < n {
                basic_cols.push(j);
            }
        }

        // Determine which slacks are basic
        for i in 0..m {
            if basic_cols.len() >= m {
                break;
            }
            let ay: f64 = (0..n.min(sol.primal.len()))
                .map(|j| a_dense[(i, j)] * sol.primal[j])
                .sum();
            let slack = lp.b_rhs[i] - ay;
            if slack.abs() > self.tolerance {
                basic_cols.push(n + i);
            }
        }

        // Pad if needed
        while basic_cols.len() < m {
            basic_cols.push(n + basic_cols.len());
        }

        // Build basis matrix
        let mut b_mat = DMatrix::zeros(m, m);
        for (col, &idx) in basic_cols.iter().enumerate() {
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

        // Condition number estimate (ratio of largest to smallest singular value)
        let svd = b_mat.svd(false, false);
        let singular_values = &svd.singular_values;
        let max_sv = singular_values.iter().cloned().fold(0.0f64, f64::max);
        let min_sv = singular_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let condition_number = if min_sv > TOLERANCE {
            max_sv / min_sv
        } else {
            f64::INFINITY
        };

        // Minimum reduced cost for non-basic variables
        let mut min_rc = f64::INFINITY;
        let mut near_degenerate_count = 0;

        for j in 0..n {
            if sol.basis.get(j) == Some(&BasisStatus::NonBasicLower)
                || sol.basis.get(j) == Some(&BasisStatus::NonBasicUpper)
            {
                let rc = lp.c[j]
                    - (0..m)
                        .map(|i| {
                            let d = sol.dual.get(i).copied().unwrap_or(0.0);
                            a_dense[(i, j)] * d
                        })
                        .sum::<f64>();
                let rc_abs = rc.abs();
                if rc_abs < min_rc {
                    min_rc = rc_abs;
                }
                if rc_abs < self.tolerance * 1000.0 {
                    near_degenerate_count += 1;
                }
            }
        }

        if min_rc == f64::INFINITY {
            min_rc = 0.0;
        }

        // Perturbation tolerance: how much can dual be perturbed?
        let perturbation_tolerance = min_rc * 0.5;

        let stable = condition_number < 1e10 && min_rc > self.tolerance * 100.0;

        Ok(DualStabilityInfo {
            stable,
            condition_number,
            min_reduced_cost: min_rc,
            near_degenerate_count,
            perturbation_tolerance,
        })
    }

    /// Characterize the dual feasible region for the LP at x.
    pub fn dual_feasible_region(&self, x: &[f64]) -> VFResult<DualFeasibleRegion> {
        let lp = self.problem.lower_level_lp(x);
        let n = lp.num_vars;
        let m = lp.num_constraints;
        let a_dense = lp.a_matrix.to_dense();

        // Dual feasible region: { π ≥ 0 : A^T π ≤ c }  (for minimization with ≤ constraints)
        let mut poly = Polyhedron::new(m);

        // Non-negativity: -π_i ≤ 0
        for i in 0..m {
            let mut normal = vec![0.0; m];
            normal[i] = -1.0;
            poly.add_halfspace(normal, 0.0);
        }

        // A^T π ≤ c  →  for each j: sum_i A_{ij} π_i ≤ c_j
        for j in 0..n {
            let mut normal = vec![0.0; m];
            for i in 0..m {
                normal[i] = a_dense[(i, j)];
            }
            poly.add_halfspace(normal, lp.c[j]);
        }

        // Find the optimal dual vertex
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        let mut vertices = Vec::new();
        if sol.status == LpStatus::Optimal {
            vertices.push(sol.dual.clone());
        }

        Ok(DualFeasibleRegion {
            polyhedron: poly,
            vertices,
            extreme_rays: Vec::new(),
            is_bounded: true,
        })
    }

    /// Compute the subgradient of φ at x from dual multipliers.
    ///
    /// For the parametric LP  φ(x) = min{ cᵀy : Ay ≤ b + Bx, y ≥ 0 },
    /// the subgradient is  g = Bᵀ sp  where sp are the LP shadow prices
    /// (∂φ/∂b_i ≤ 0 for binding ≤ constraints in a minimisation problem).
    /// LP solvers (HiGHS, etc.) return shadow prices in this convention.
    pub fn compute_subgradient(&self, x: &[f64]) -> VFResult<Vec<f64>> {
        let sp = self.shadow_prices(x)?;
        let nx = self.problem.num_upper_vars;
        let mut grad = vec![0.0; nx];

        for entry in &self.problem.lower_linking_b.entries {
            if entry.row < sp.prices.len() && entry.col < nx {
                grad[entry.col] += entry.value * sp.prices[entry.row];
            }
        }

        Ok(grad)
    }

    /// Compare dual solutions at two nearby points.
    pub fn dual_continuity_check(&self, x1: &[f64], x2: &[f64]) -> VFResult<f64> {
        let sp1 = self.shadow_prices(x1)?;
        let sp2 = self.shadow_prices(x2)?;

        let dual_diff: f64 = sp1
            .prices
            .iter()
            .zip(sp2.prices.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let x_diff: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        if x_diff > TOLERANCE {
            Ok(dual_diff / x_diff)
        } else {
            Ok(0.0)
        }
    }

    /// Detect dual degeneracy at x.
    pub fn detect_degeneracy(&self, x: &[f64]) -> VFResult<bool> {
        let lp = self.problem.lower_level_lp(x);
        let sol = self
            .solver
            .solve(&lp)
            .map_err(|e| VFError::LpError(format!("{}", e)))?;

        if sol.status != LpStatus::Optimal {
            return Err(VFError::LpError(format!("Status: {}", sol.status)));
        }

        // Dual degeneracy: a basic variable has zero value
        for (j, &bs) in sol.basis.iter().enumerate() {
            if bs == BasisStatus::Basic && j < sol.primal.len() {
                if sol.primal[j].abs() < self.tolerance {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::SparseMatrix;

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
    fn test_shadow_prices() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let sp = analyzer.shadow_prices(&[0.0]).unwrap();
        assert!(sp.valid);
        assert_eq!(sp.prices.len(), 2);
    }

    #[test]
    fn test_complementary_slackness() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let cs = analyzer.verify_complementary_slackness(&[0.0]).unwrap();
        assert!(cs.holds);
        assert!(cs.max_violation < 1.0);
    }

    #[test]
    fn test_dual_stability() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let stability = analyzer.dual_stability(&[0.0]).unwrap();
        assert!(stability.condition_number >= 1.0);
        assert!(stability.stability_score() >= 0.0);
        assert!(stability.stability_score() <= 1.0);
    }

    #[test]
    fn test_dual_feasible_region() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let region = analyzer.dual_feasible_region(&[0.0]).unwrap();
        assert!(!region.polyhedron.halfspaces.is_empty());
    }

    #[test]
    fn test_compute_subgradient() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let sg = analyzer.compute_subgradient(&[0.0]).unwrap();
        assert_eq!(sg.len(), 1);
    }

    #[test]
    fn test_track_shadow_prices() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let track = analyzer
            .track_shadow_prices(&[0.0], &[1.0], -1.0, 1.0, 5)
            .unwrap();
        assert!(!track.is_empty());
    }

    #[test]
    fn test_dual_continuity() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let ratio = analyzer.dual_continuity_check(&[0.0], &[0.1]).unwrap();
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_detect_degeneracy() {
        let problem = test_bilevel();
        let analyzer = DualAnalyzer::new(problem);
        let _ = analyzer.detect_degeneracy(&[0.0]);
    }

    #[test]
    fn test_shadow_price_info_methods() {
        let sp = ShadowPriceInfo {
            prices: vec![1.0, 0.0, 2.0],
            is_active: vec![true, false, true],
            valid: true,
            at_point: vec![0.0],
        };
        assert_eq!(sp.num_active(), 2);
        assert_eq!(sp.active_indices(), vec![0, 2]);
        assert!((sp.total_sensitivity() - 3.0).abs() < 1e-10);
    }
}
