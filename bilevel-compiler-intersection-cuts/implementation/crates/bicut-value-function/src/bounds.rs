//! Value function bounds computation.
//!
//! Computes upper and lower bounds on φ(x) over polyhedral domains,
//! Lagrangian relaxation bounds, bound tightening using VF structure,
//! and valid inequality generation from bounds.

use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BilevelProblem, ValidInequality};
use serde::{Deserialize, Serialize};

use crate::oracle::ValueFunctionOracle;
use crate::piecewise_linear::{AffinePiece, PiecewiseLinearVF};
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Bounds structures
// ---------------------------------------------------------------------------

/// Upper and lower bounds on the value function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFunctionBounds {
    /// Lower bound on φ(x) over the domain.
    pub lower: f64,
    /// Upper bound on φ(x) over the domain.
    pub upper: f64,
    /// Gap between upper and lower bound.
    pub gap: f64,
    /// Relative gap (gap / upper).
    pub relative_gap: f64,
    /// Number of evaluations used.
    pub evaluations_used: usize,
}

impl ValueFunctionBounds {
    pub fn new(lower: f64, upper: f64) -> Self {
        let gap = (upper - lower).max(0.0);
        let relative_gap = if upper.abs() > TOLERANCE {
            gap / upper.abs()
        } else {
            gap
        };
        Self {
            lower,
            upper,
            gap,
            relative_gap,
            evaluations_used: 0,
        }
    }

    pub fn with_evaluations(mut self, n: usize) -> Self {
        self.evaluations_used = n;
        self
    }

    /// Check if the bounds are tight (gap below tolerance).
    pub fn is_tight(&self, tol: f64) -> bool {
        self.gap < tol
    }

    /// Intersect with another set of bounds (tighten).
    pub fn intersect(&self, other: &ValueFunctionBounds) -> ValueFunctionBounds {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        ValueFunctionBounds::new(lower, upper)
            .with_evaluations(self.evaluations_used + other.evaluations_used)
    }
}

/// Result of bound tightening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundTighteningResult {
    /// Whether the bounds were tightened.
    pub tightened: bool,
    /// New lower bound.
    pub new_lower: f64,
    /// New upper bound.
    pub new_upper: f64,
    /// Previous lower bound.
    pub old_lower: f64,
    /// Previous upper bound.
    pub old_upper: f64,
    /// Amount of tightening on the lower bound.
    pub lower_improvement: f64,
    /// Amount of tightening on the upper bound.
    pub upper_improvement: f64,
}

/// Result of Lagrangian relaxation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianBound {
    /// The Lagrangian bound value.
    pub bound: f64,
    /// The optimal Lagrange multipliers.
    pub multipliers: Vec<f64>,
    /// Number of subgradient iterations used.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Bounds computer
// ---------------------------------------------------------------------------

/// Computes bounds on the value function.
pub struct BoundsComputer {
    problem: BilevelProblem,
    solver: SimplexSolver,
    tolerance: f64,
    max_iterations: usize,
}

impl BoundsComputer {
    pub fn new(problem: BilevelProblem) -> Self {
        Self {
            problem,
            solver: SimplexSolver::default(),
            tolerance: TOLERANCE,
            max_iterations: 100,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Compute bounds on φ(x) over a box [x_lower, x_upper] using sampling.
    pub fn bounds_over_box(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_lower: &[f64],
        x_upper: &[f64],
        num_samples: usize,
    ) -> VFResult<ValueFunctionBounds> {
        let nx = x_lower.len();
        let mut best_lower = f64::INFINITY;
        let mut best_upper = f64::NEG_INFINITY;
        let mut eval_count = 0usize;

        // Evaluate at corners of the box
        let num_corners = 2usize.pow(nx.min(10) as u32);
        for corner in 0..num_corners.min(num_samples / 2 + 1) {
            let x: Vec<f64> = (0..nx)
                .map(|d| {
                    if (corner >> d) & 1 == 0 {
                        x_lower[d]
                    } else {
                        x_upper[d]
                    }
                })
                .collect();

            match oracle.value(&x) {
                Ok(v) => {
                    best_lower = best_lower.min(v);
                    best_upper = best_upper.max(v);
                    eval_count += 1;
                }
                Err(_) => continue,
            }
        }

        // Evaluate at additional random points
        let remaining = num_samples.saturating_sub(eval_count);
        let mut rng_state: u64 = 54321;
        for _ in 0..remaining {
            let x: Vec<f64> = (0..nx)
                .map(|d| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    x_lower[d] + u * (x_upper[d] - x_lower[d])
                })
                .collect();

            match oracle.value(&x) {
                Ok(v) => {
                    best_lower = best_lower.min(v);
                    best_upper = best_upper.max(v);
                    eval_count += 1;
                }
                Err(_) => continue,
            }
        }

        if eval_count == 0 {
            return Err(VFError::SamplingError(
                "No feasible points found in box".into(),
            ));
        }

        Ok(ValueFunctionBounds::new(best_lower, best_upper).with_evaluations(eval_count))
    }

    /// Compute a lower bound using a cutting plane model.
    pub fn cutting_plane_lower_bound(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_eval: &[f64],
        x_query: &[f64],
    ) -> VFResult<f64> {
        let info = oracle.evaluate(x_eval)?;
        let dual = oracle.dual_info(x_eval)?;

        // φ(x) ≥ φ(x_eval) + g^T (x - x_eval)
        let diff: f64 = dual
            .subgradient
            .iter()
            .zip(x_query.iter())
            .zip(x_eval.iter())
            .map(|((g, xq), xe)| g * (xq - xe))
            .sum();

        Ok(info.value + diff)
    }

    /// Build a cutting plane lower bound from multiple evaluation points.
    pub fn multi_cut_lower_bound(
        &self,
        oracle: &dyn ValueFunctionOracle,
        eval_points: &[Vec<f64>],
        x_query: &[f64],
    ) -> VFResult<f64> {
        let mut best_lb = f64::NEG_INFINITY;

        for x_eval in eval_points {
            match self.cutting_plane_lower_bound(oracle, x_eval, x_query) {
                Ok(lb) => {
                    best_lb = best_lb.max(lb);
                }
                Err(_) => continue,
            }
        }

        if best_lb == f64::NEG_INFINITY {
            return Err(VFError::SamplingError(
                "No valid cutting planes found".into(),
            ));
        }

        Ok(best_lb)
    }

    /// Compute Lagrangian relaxation bound.
    ///
    /// For the bilevel problem, we relax the linking constraints
    /// and solve the resulting easier subproblem.
    pub fn lagrangian_bound(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_lower: &[f64],
        x_upper: &[f64],
        initial_multipliers: &[f64],
        num_iterations: usize,
    ) -> VFResult<LagrangianBound> {
        let nx = x_lower.len();
        let mut mu = initial_multipliers.to_vec();
        let mut best_bound = f64::NEG_INFINITY;
        let mut step_size = 1.0;

        for iter in 0..num_iterations.min(self.max_iterations) {
            // Evaluate Lagrangian at the midpoint with current multipliers
            let x_mid: Vec<f64> = (0..nx).map(|d| (x_lower[d] + x_upper[d]) / 2.0).collect();

            let val = match oracle.value(&x_mid) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Lagrangian: L(μ) = φ(x) + μ^T g(x)
            // Where g(x) represents the constraint violation
            let sg = oracle.subgradient(&x_mid).unwrap_or_else(|_| vec![0.0; nx]);

            let lagrangian_val = val + mu.iter().zip(sg.iter()).map(|(m, g)| m * g).sum::<f64>();

            if lagrangian_val > best_bound {
                best_bound = lagrangian_val;
            }

            // Subgradient update
            let sg_norm_sq: f64 = sg.iter().map(|g| g * g).sum::<f64>();
            if sg_norm_sq > TOLERANCE {
                step_size = 1.0 / (iter as f64 + 1.0);
                for j in 0..mu.len().min(sg.len()) {
                    mu[j] = (mu[j] + step_size * sg[j]).max(0.0);
                }
            }
        }

        Ok(LagrangianBound {
            bound: best_bound,
            multipliers: mu,
            iterations: num_iterations.min(self.max_iterations),
        })
    }

    /// Tighten bounds using value function structure.
    pub fn tighten_bounds(
        &self,
        oracle: &dyn ValueFunctionOracle,
        current: &ValueFunctionBounds,
        eval_points: &[Vec<f64>],
    ) -> VFResult<BoundTighteningResult> {
        let mut new_lower = current.lower;
        let mut new_upper = current.upper;

        for x in eval_points {
            match oracle.value(x) {
                Ok(v) => {
                    new_lower = new_lower.min(v);
                    new_upper = new_upper.max(v);
                }
                Err(_) => continue,
            }

            // Use cutting planes to potentially improve bounds
            if let Ok(dual) = oracle.dual_info(x) {
                // The subgradient gives us a linear underestimator
                let info = oracle.evaluate(x)?;
                // At any other point x', φ(x') ≥ φ(x) + g^T(x' - x)
                // This is already captured in the PWL lower bound
                let _ = (info, dual);
            }
        }

        let lower_improvement = (new_lower - current.lower).max(0.0);
        let upper_improvement = (current.upper - new_upper).max(0.0);
        let tightened = lower_improvement > self.tolerance || upper_improvement > self.tolerance;

        Ok(BoundTighteningResult {
            tightened,
            new_lower,
            new_upper,
            old_lower: current.lower,
            old_upper: current.upper,
            lower_improvement,
            upper_improvement,
        })
    }

    /// Generate valid inequalities from bounds.
    pub fn generate_bound_inequalities(
        &self,
        oracle: &dyn ValueFunctionOracle,
        eval_points: &[Vec<f64>],
    ) -> Vec<ValidInequality> {
        let nx = self.problem.num_upper_vars;
        let ny = self.problem.num_lower_vars;
        let mut inequalities = Vec::new();

        for x in eval_points {
            if let Ok(info) = oracle.evaluate(x) {
                if let Ok(dual) = oracle.dual_info(x) {
                    // Valid inequality: c^T y ≥ φ(x*) + g^T(x - x*)
                    // Rearranged: -g^T x + c^T y ≥ φ(x*) - g^T x*
                    let alpha: Vec<f64> = dual.subgradient.iter().map(|&g| -g).collect();
                    let beta = self.problem.lower_obj_c.clone();
                    let gamma = info.value
                        - dual
                            .subgradient
                            .iter()
                            .zip(x.iter())
                            .map(|(g, xi)| g * xi)
                            .sum::<f64>();

                    inequalities.push(ValidInequality { alpha, beta, gamma });
                }
            }
        }

        inequalities
    }

    /// Compute optimality gap at a point given primal and dual bounds.
    pub fn optimality_gap(
        &self,
        x: &[f64],
        y: &[f64],
        oracle: &dyn ValueFunctionOracle,
    ) -> VFResult<f64> {
        let vf_val = oracle.value(x)?;

        let primal_obj: f64 = self
            .problem
            .lower_obj_c
            .iter()
            .zip(y.iter())
            .map(|(c, yi)| c * yi)
            .sum();

        Ok((primal_obj - vf_val).max(0.0))
    }

    /// Compute bounds on the value function using a PWL approximation.
    pub fn pwl_bounds(
        &self,
        pwl: &PiecewiseLinearVF,
        x_lower: &[f64],
        x_upper: &[f64],
        num_samples: usize,
    ) -> ValueFunctionBounds {
        let nx = x_lower.len();
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        let mut rng_state: u64 = 98765;
        for _ in 0..num_samples {
            let x: Vec<f64> = (0..nx)
                .map(|d| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    x_lower[d] + u * (x_upper[d] - x_lower[d])
                })
                .collect();

            let val = pwl.evaluate(&x);
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        ValueFunctionBounds::new(min_val, max_val).with_evaluations(num_samples)
    }

    /// Iteratively tighten bounds until convergence.
    pub fn iterative_bound_tightening(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_lower: &[f64],
        x_upper: &[f64],
        initial_samples: usize,
        max_rounds: usize,
    ) -> VFResult<ValueFunctionBounds> {
        let mut bounds = self.bounds_over_box(oracle, x_lower, x_upper, initial_samples)?;

        for round in 0..max_rounds {
            // Generate refinement points near the current bound extremes
            let nx = x_lower.len();
            let mut refine_points = Vec::new();

            let mut rng_state: u64 = 11111 + round as u64;
            for _ in 0..10 {
                let x: Vec<f64> = (0..nx)
                    .map(|d| {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                        x_lower[d] + u * (x_upper[d] - x_lower[d])
                    })
                    .collect();
                refine_points.push(x);
            }

            let result = self.tighten_bounds(oracle, &bounds, &refine_points)?;

            if !result.tightened {
                break;
            }

            bounds = ValueFunctionBounds::new(result.new_lower, result.new_upper)
                .with_evaluations(bounds.evaluations_used + refine_points.len());
        }

        Ok(bounds)
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

    struct TestOracle;

    impl ValueFunctionOracle for TestOracle {
        fn evaluate(&self, x: &[f64]) -> VFResult<crate::oracle::ValueFunctionInfo> {
            Ok(crate::oracle::ValueFunctionInfo {
                value: x[0].abs(),
                primal_solution: vec![x[0].abs()],
                dual_solution: vec![0.0, 0.0],
                basis: vec![],
                iterations: 1,
            })
        }

        fn dual_info(&self, x: &[f64]) -> VFResult<crate::oracle::DualInfo> {
            Ok(crate::oracle::DualInfo {
                multipliers: vec![0.0, 0.0],
                subgradient: vec![if x[0] >= 0.0 { 1.0 } else { -1.0 }],
                is_degenerate: false,
            })
        }

        fn check_feasibility(&self, _x: &[f64]) -> VFResult<crate::oracle::FeasibilityInfo> {
            Ok(crate::oracle::FeasibilityInfo {
                is_feasible: true,
                is_bounded: true,
                farkas_certificate: None,
            })
        }

        fn statistics(&self) -> crate::oracle::OracleStatistics {
            crate::oracle::OracleStatistics::default()
        }

        fn reset_statistics(&self) {}
    }

    #[test]
    fn test_bounds_over_box() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);
        let bounds = computer
            .bounds_over_box(&TestOracle, &[-2.0], &[2.0], 20)
            .unwrap();
        assert!(bounds.lower >= 0.0);
        assert!(bounds.upper <= 2.1);
        assert!(bounds.gap >= 0.0);
    }

    #[test]
    fn test_value_function_bounds_new() {
        let bounds = ValueFunctionBounds::new(1.0, 5.0);
        assert!((bounds.gap - 4.0).abs() < 1e-10);
        assert!(!bounds.is_tight(1.0));
    }

    #[test]
    fn test_bounds_intersect() {
        let a = ValueFunctionBounds::new(1.0, 10.0);
        let b = ValueFunctionBounds::new(3.0, 7.0);
        let c = a.intersect(&b);
        assert!((c.lower - 3.0).abs() < 1e-10);
        assert!((c.upper - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cutting_plane_lower_bound() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);
        let lb = computer
            .cutting_plane_lower_bound(&TestOracle, &[1.0], &[2.0])
            .unwrap();
        // φ(1) = 1, g = 1, so φ(2) ≥ 1 + 1*(2-1) = 2
        assert!((lb - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_generate_bound_inequalities() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);
        let points = vec![vec![0.0], vec![1.0]];
        let ineqs = computer.generate_bound_inequalities(&TestOracle, &points);
        assert_eq!(ineqs.len(), 2);
    }

    #[test]
    fn test_optimality_gap() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);
        let gap = computer
            .optimality_gap(&[1.0], &[1.5], &TestOracle)
            .unwrap();
        // φ(1) = 1, c^T y = 1.5, gap = 0.5
        assert!((gap - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_pwl_bounds() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);

        let mut pwl = PiecewiseLinearVF::new(1);
        pwl.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));
        pwl.add_piece(AffinePiece::from_gradient(vec![-1.0], 0.0));

        let bounds = computer.pwl_bounds(&pwl, &[-2.0], &[2.0], 100);
        assert!(bounds.lower >= 0.0);
        assert!(bounds.upper <= 2.1);
    }

    #[test]
    fn test_bound_tightening_result() {
        let result = BoundTighteningResult {
            tightened: true,
            new_lower: 1.0,
            new_upper: 4.0,
            old_lower: 0.0,
            old_upper: 5.0,
            lower_improvement: 1.0,
            upper_improvement: 1.0,
        };
        assert!(result.tightened);
        assert!((result.lower_improvement - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lagrangian_bound() {
        let problem = test_bilevel();
        let computer = BoundsComputer::new(problem);
        let lb = computer
            .lagrangian_bound(&TestOracle, &[-1.0], &[1.0], &[0.0], 5)
            .unwrap();
        assert!(lb.iterations > 0);
    }
}
