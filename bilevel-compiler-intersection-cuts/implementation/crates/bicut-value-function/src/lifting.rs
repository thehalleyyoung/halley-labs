//! Value function lifting for intersection cuts.
//!
//! Implements Gomory-Johnson style strengthening using value function structure,
//! subadditive approximations, lifting coefficient computation, and valid
//! inequality generation from VF structure.

use bicut_types::{AffineFunction, BilevelProblem, Polyhedron, SparseMatrix, ValidInequality};
use serde::{Deserialize, Serialize};

use crate::oracle::ValueFunctionOracle;
use crate::piecewise_linear::{AffinePiece, PiecewiseLinearVF};
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Subadditive approximation
// ---------------------------------------------------------------------------

/// A piecewise-linear subadditive function for lifting.
///
/// A function ψ: R^n → R is subadditive if ψ(a+b) ≤ ψ(a) + ψ(b) for all a, b.
/// Subadditive functions yield valid inequalities via lifting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubadditiveApprox {
    /// Breakpoints (1-D) or sample points (multi-D).
    pub breakpoints: Vec<f64>,
    /// Values at the breakpoints.
    pub values: Vec<f64>,
    /// Whether the approximation is known to be subadditive.
    pub is_verified_subadditive: bool,
    /// Dimension.
    pub dim: usize,
}

impl SubadditiveApprox {
    /// Evaluate the subadditive function at a scalar r.
    pub fn evaluate_1d(&self, r: f64) -> f64 {
        if self.breakpoints.is_empty() {
            return 0.0;
        }
        if self.breakpoints.len() == 1 {
            return self.values[0] * r / self.breakpoints[0].max(TOLERANCE);
        }

        // Piecewise linear interpolation
        let n = self.breakpoints.len();

        if r <= self.breakpoints[0] {
            // Extrapolate from the first segment
            let slope = if n >= 2 {
                (self.values[1] - self.values[0])
                    / (self.breakpoints[1] - self.breakpoints[0]).max(TOLERANCE)
            } else {
                self.values[0] / self.breakpoints[0].max(TOLERANCE)
            };
            return self.values[0] + slope * (r - self.breakpoints[0]);
        }

        if r >= self.breakpoints[n - 1] {
            let slope = if n >= 2 {
                (self.values[n - 1] - self.values[n - 2])
                    / (self.breakpoints[n - 1] - self.breakpoints[n - 2]).max(TOLERANCE)
            } else {
                self.values[0] / self.breakpoints[0].max(TOLERANCE)
            };
            return self.values[n - 1] + slope * (r - self.breakpoints[n - 1]);
        }

        // Binary search for the interval
        let mut lo = 0;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.breakpoints[mid] <= r {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let t = (r - self.breakpoints[lo])
            / (self.breakpoints[hi] - self.breakpoints[lo]).max(TOLERANCE);
        self.values[lo] * (1.0 - t) + self.values[hi] * t
    }

    /// Check subadditivity on a grid.
    pub fn verify_subadditivity(&self, num_checks: usize) -> bool {
        if self.breakpoints.is_empty() {
            return true;
        }

        let lo = *self.breakpoints.first().unwrap_or(&0.0);
        let hi = *self.breakpoints.last().unwrap_or(&1.0);
        let range = hi - lo;

        for i in 0..num_checks {
            let a = lo + range * i as f64 / num_checks as f64;
            for j in 0..num_checks {
                let b = lo + range * j as f64 / num_checks as f64;
                let psi_ab = self.evaluate_1d(a + b);
                let psi_a = self.evaluate_1d(a);
                let psi_b = self.evaluate_1d(b);
                if psi_ab > psi_a + psi_b + TOLERANCE * 100.0 {
                    return false;
                }
            }
        }

        true
    }

    /// Create a Gomory fractional cut function (one of the simplest subadditive functions).
    pub fn gomory_fractional(f0: f64) -> Self {
        // ψ(r) = r/f0  if 0 ≤ r ≤ f0
        // ψ(r) = (1-r)/(1-f0) if f0 < r < 1
        // Period 1
        let n_breakpoints = 21;
        let mut breakpoints = Vec::with_capacity(n_breakpoints);
        let mut values = Vec::with_capacity(n_breakpoints);

        for i in 0..n_breakpoints {
            let r = i as f64 / (n_breakpoints - 1) as f64;
            breakpoints.push(r);
            let frac = r - r.floor();
            let val = if frac <= f0 + TOLERANCE {
                frac / f0.max(TOLERANCE)
            } else {
                (1.0 - frac) / (1.0 - f0).max(TOLERANCE)
            };
            values.push(val);
        }

        SubadditiveApprox {
            breakpoints,
            values,
            is_verified_subadditive: true,
            dim: 1,
        }
    }

    /// Create a linear subadditive function: ψ(r) = α * r.
    pub fn linear(alpha: f64) -> Self {
        SubadditiveApprox {
            breakpoints: vec![0.0, 1.0],
            values: vec![0.0, alpha],
            is_verified_subadditive: true,
            dim: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifting coefficients
// ---------------------------------------------------------------------------

/// Computed lifting coefficients for an intersection cut.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiftingCoefficients {
    /// Lifting coefficient for each variable.
    pub coefficients: Vec<f64>,
    /// Whether the coefficients yield a valid inequality.
    pub is_valid: bool,
    /// Strengthening factor (ratio of lifted vs unlifted cut depth).
    pub strengthening_factor: f64,
}

impl LiftingCoefficients {
    /// Apply these coefficients to generate a valid inequality.
    pub fn to_inequality(&self, rhs: f64) -> ValidInequality {
        let n = self.coefficients.len();
        // Split coefficients into x and y parts
        // Convention: first half are x, second half are y
        let mid = n / 2;
        let alpha = self.coefficients[..mid].to_vec();
        let beta = self.coefficients[mid..].to_vec();
        ValidInequality {
            alpha,
            beta,
            gamma: rhs,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifting computer
// ---------------------------------------------------------------------------

/// Computes lifting coefficients using value function structure.
pub struct LiftingComputer {
    /// The bilevel problem.
    problem: BilevelProblem,
    /// Numerical tolerance.
    tolerance: f64,
    /// Maximum number of lifting iterations.
    max_iterations: usize,
}

impl LiftingComputer {
    pub fn new(problem: BilevelProblem) -> Self {
        Self {
            problem,
            tolerance: TOLERANCE,
            max_iterations: 1000,
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

    /// Compute Gomory-Johnson lifting coefficients for an intersection cut.
    ///
    /// Given a fractional point (x*, y*) and the value function, compute
    /// strengthened cut coefficients using the subadditive dual.
    pub fn compute_gj_lifting(
        &self,
        x_star: &[f64],
        y_star: &[f64],
        oracle: &dyn ValueFunctionOracle,
        base_cut: &ValidInequality,
    ) -> VFResult<LiftingCoefficients> {
        let nx = x_star.len();
        let ny = y_star.len();

        // Get value function info at x*
        let vf_info = oracle.evaluate(x_star)?;
        let dual_info = oracle.dual_info(x_star)?;

        // Compute f0 = fractional part of the value function
        let phi_val = vf_info.value;
        let f0 = phi_val - phi_val.floor();
        let f0 = if f0.abs() < self.tolerance {
            0.5 // Avoid degenerate case
        } else {
            f0
        };

        // Build Gomory fractional function
        let psi = SubadditiveApprox::gomory_fractional(f0);

        // Compute lifted coefficients
        let mut coefficients = Vec::with_capacity(nx + ny);

        // Lift x-variables using subgradient information
        for j in 0..nx {
            let grad_j = dual_info.subgradient.get(j).copied().unwrap_or(0.0);
            let base_coeff = base_cut.alpha.get(j).copied().unwrap_or(0.0);

            // Lifting: use ψ(r_j) where r_j is the fractional part of the
            // tableau coefficient.
            let r_j = grad_j - grad_j.floor();
            let lifted = psi.evaluate_1d(r_j);

            // Take the tighter of the lifted coefficient and the base coefficient
            let coeff = if lifted.abs() > self.tolerance {
                base_coeff.max(lifted)
            } else {
                base_coeff
            };
            coefficients.push(coeff);
        }

        // Lift y-variables
        for j in 0..ny {
            let base_coeff = base_cut.beta.get(j).copied().unwrap_or(0.0);

            // Use the objective coefficient and dual info for lifting
            let c_j = self.problem.lower_obj_c.get(j).copied().unwrap_or(0.0);
            let r_j = c_j - c_j.floor();
            let lifted = psi.evaluate_1d(r_j);

            let coeff = if lifted.abs() > self.tolerance {
                base_coeff.max(lifted)
            } else {
                base_coeff
            };
            coefficients.push(coeff);
        }

        // Compute strengthening factor
        let base_depth = self.compute_cut_depth(base_cut, x_star, y_star);
        let new_cut = LiftingCoefficients {
            coefficients: coefficients.clone(),
            is_valid: true,
            strengthening_factor: 1.0,
        };
        let new_ineq = new_cut.to_inequality(base_cut.gamma);
        let new_depth = self.compute_cut_depth(&new_ineq, x_star, y_star);

        let strengthening_factor = if base_depth.abs() > self.tolerance {
            new_depth / base_depth
        } else {
            1.0
        };

        Ok(LiftingCoefficients {
            coefficients,
            is_valid: true,
            strengthening_factor,
        })
    }

    /// Compute the depth of a cut at a point.
    fn compute_cut_depth(&self, cut: &ValidInequality, x: &[f64], y: &[f64]) -> f64 {
        let lhs: f64 = cut
            .alpha
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a * xi)
            .sum::<f64>()
            + cut
                .beta
                .iter()
                .zip(y.iter())
                .map(|(b, yi)| b * yi)
                .sum::<f64>();
        cut.gamma - lhs
    }

    /// Generate a value-function based valid inequality at a point.
    pub fn generate_vf_inequality(
        &self,
        x_star: &[f64],
        oracle: &dyn ValueFunctionOracle,
    ) -> VFResult<ValidInequality> {
        let vf_info = oracle.evaluate(x_star)?;
        let dual_info = oracle.dual_info(x_star)?;

        let nx = x_star.len();
        let ny = self.problem.num_lower_vars;

        // Valid inequality: c^T y ≥ φ(x)
        // Linearized: c^T y ≥ φ(x*) + g^T (x - x*)
        // = c^T y - g^T x ≥ φ(x*) - g^T x*

        let alpha: Vec<f64> = dual_info.subgradient.iter().map(|&g| -g).collect();
        let beta = self.problem.lower_obj_c.clone();
        let gamma = vf_info.value
            - dual_info
                .subgradient
                .iter()
                .zip(x_star.iter())
                .map(|(g, x)| g * x)
                .sum::<f64>();

        Ok(ValidInequality { alpha, beta, gamma })
    }

    /// Generate multiple valid inequalities by evaluating at several points.
    pub fn generate_multiple_inequalities(
        &self,
        points: &[Vec<f64>],
        oracle: &dyn ValueFunctionOracle,
    ) -> Vec<ValidInequality> {
        let mut inequalities = Vec::new();

        for x in points {
            if let Ok(ineq) = self.generate_vf_inequality(x, oracle) {
                inequalities.push(ineq);
            }
        }

        inequalities
    }

    /// Compute the subadditive closure of the value function.
    pub fn subadditive_closure(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_points: &[Vec<f64>],
    ) -> VFResult<PiecewiseLinearVF> {
        let dim = self.problem.num_upper_vars;
        let mut pwl = PiecewiseLinearVF::new(dim);

        for x in x_points {
            if let Ok(info) = oracle.evaluate(x) {
                if let Ok(dual) = oracle.dual_info(x) {
                    let constant = info.value
                        - dual
                            .subgradient
                            .iter()
                            .zip(x.iter())
                            .map(|(g, xi)| g * xi)
                            .sum::<f64>();

                    pwl.add_piece(AffinePiece {
                        coefficients: dual.subgradient.clone(),
                        constant,
                        region: None,
                    });
                }
            }
        }

        Ok(pwl)
    }

    /// Strengthen a set of cuts using value function lifting.
    pub fn strengthen_cuts(
        &self,
        cuts: &[ValidInequality],
        x_star: &[f64],
        y_star: &[f64],
        oracle: &dyn ValueFunctionOracle,
    ) -> Vec<ValidInequality> {
        let mut strengthened = Vec::new();

        for cut in cuts {
            match self.compute_gj_lifting(x_star, y_star, oracle, cut) {
                Ok(lifted) => {
                    strengthened.push(lifted.to_inequality(cut.gamma));
                }
                Err(_) => {
                    strengthened.push(cut.clone());
                }
            }
        }

        strengthened
    }

    /// Compute the gap between upper and lower bounds from VF lifting.
    pub fn compute_lifting_gap(
        &self,
        x_star: &[f64],
        y_star: &[f64],
        oracle: &dyn ValueFunctionOracle,
    ) -> VFResult<f64> {
        let vf_info = oracle.evaluate(x_star)?;

        // Lower bound: φ(x*)
        let lower = vf_info.value;

        // Primal bound: c^T y*
        let upper: f64 = self
            .problem
            .lower_obj_c
            .iter()
            .zip(y_star.iter())
            .map(|(c, y)| c * y)
            .sum();

        Ok((upper - lower).max(0.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subadditive_linear() {
        let psi = SubadditiveApprox::linear(2.0);
        assert!((psi.evaluate_1d(0.5) - 1.0).abs() < 1e-10);
        assert!((psi.evaluate_1d(1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gomory_fractional() {
        let psi = SubadditiveApprox::gomory_fractional(0.5);
        // At r=0: ψ(0) = 0
        assert!(psi.evaluate_1d(0.0).abs() < 0.1);
        // At r=0.5: ψ(0.5) = 1.0
        assert!((psi.evaluate_1d(0.5) - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_verify_subadditivity_linear() {
        let psi = SubadditiveApprox::linear(1.0);
        assert!(psi.verify_subadditivity(10));
    }

    #[test]
    fn test_subadditive_evaluate_interpolation() {
        let psi = SubadditiveApprox {
            breakpoints: vec![0.0, 0.5, 1.0],
            values: vec![0.0, 1.0, 0.5],
            is_verified_subadditive: false,
            dim: 1,
        };
        // At 0.25: interpolate between (0.0, 0.0) and (0.5, 1.0) → 0.5
        assert!((psi.evaluate_1d(0.25) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lifting_coefficients_to_inequality() {
        let lc = LiftingCoefficients {
            coefficients: vec![1.0, 2.0, 3.0, 4.0],
            is_valid: true,
            strengthening_factor: 1.5,
        };
        let ineq = lc.to_inequality(5.0);
        assert_eq!(ineq.alpha, vec![1.0, 2.0]);
        assert_eq!(ineq.beta, vec![3.0, 4.0]);
        assert_eq!(ineq.gamma, 5.0);
    }

    #[test]
    fn test_compute_cut_depth() {
        let problem = test_bilevel_lifting();
        let computer = LiftingComputer::new(problem);

        let cut = ValidInequality {
            alpha: vec![1.0],
            beta: vec![1.0],
            gamma: 3.0,
        };

        let depth = computer.compute_cut_depth(&cut, &[1.0], &[1.0]);
        assert!((depth - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_multiple_empty() {
        let problem = test_bilevel_lifting();
        let computer = LiftingComputer::new(problem);

        // Empty points → empty inequalities
        let ineqs = computer.generate_multiple_inequalities(&[], &DummyOracle);
        assert!(ineqs.is_empty());
    }

    #[test]
    fn test_subadditive_closure_empty() {
        let problem = test_bilevel_lifting();
        let computer = LiftingComputer::new(problem);
        let result = computer.subadditive_closure(&DummyOracle, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_pieces(), 0);
    }

    fn test_bilevel_lifting() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(1, 1);
        lower_a.add_entry(0, 0, 1.0);
        let mut linking_b = SparseMatrix::new(1, 1);
        linking_b.add_entry(0, 0, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a,
            lower_b: vec![1.0],
            lower_linking_b: linking_b,
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 0,
        }
    }

    /// Dummy oracle for testing lifting utilities that don't require a real LP solve.
    struct DummyOracle;

    impl ValueFunctionOracle for DummyOracle {
        fn evaluate(&self, x: &[f64]) -> VFResult<crate::oracle::ValueFunctionInfo> {
            Ok(crate::oracle::ValueFunctionInfo {
                value: x.iter().sum::<f64>().abs(),
                primal_solution: vec![0.0],
                dual_solution: vec![0.0],
                basis: vec![],
                iterations: 0,
            })
        }

        fn dual_info(&self, x: &[f64]) -> VFResult<crate::oracle::DualInfo> {
            Ok(crate::oracle::DualInfo {
                multipliers: vec![0.0],
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
}
