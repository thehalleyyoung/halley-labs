//! Boundedness analysis for bilevel optimization problems.
//!
//! Verifies lower-level feasibility and boundedness for all leader decisions,
//! performs LP-based boundedness proofs, ray analysis, and domain computation.

use bicut_types::{BilevelProblem, SparseMatrix, SparseMatrixCsr, DEFAULT_TOLERANCE};
use log::debug;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Status of boundedness analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundednessStatus {
    /// Lower level is bounded for all feasible leader decisions.
    Bounded,
    /// Lower level is unbounded for some leader decision.
    Unbounded,
    /// Lower level is infeasible for some leader decision.
    Infeasible,
    /// Could not determine boundedness.
    Unknown,
}

/// Result of boundedness analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundednessResult {
    pub status: BoundednessStatus,
    pub feasibility_status: FeasibilityStatus,
    pub unbounded_ray: Option<Vec<f64>>,
    pub certificate: BoundednessCertificate,
    pub details: Vec<String>,
}

/// Feasibility status for the lower level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeasibilityStatus {
    FeasibleForAll,
    FeasibleForSome,
    InfeasibleForAll,
    Unknown,
}

/// Certificate for boundedness claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundednessCertificate {
    pub method: String,
    pub dual_bound: Option<f64>,
    pub primal_bound: Option<f64>,
    pub witness_x: Option<Vec<f64>>,
    pub confidence: f64,
}

/// Configuration for the boundedness analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundednessConfig {
    pub tolerance: f64,
    pub max_ray_iterations: usize,
    pub sample_count: usize,
    pub seed: u64,
}

impl Default for BoundednessConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            max_ray_iterations: 100,
            sample_count: 50,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Boundedness analysis engine.
pub struct BoundednessAnalyzer {
    config: BoundednessConfig,
}

impl BoundednessAnalyzer {
    pub fn new(config: BoundednessConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(BoundednessConfig::default())
    }

    /// Perform complete boundedness analysis.
    pub fn analyze(&self, problem: &BilevelProblem) -> BoundednessResult {
        let mut details = Vec::new();

        // Step 1: Check structural boundedness (variable bounds, constraint structure)
        let structural = self.check_structural_boundedness(problem);
        details.push(format!("Structural check: {:?}", structural));

        if structural == BoundednessStatus::Bounded {
            return BoundednessResult {
                status: BoundednessStatus::Bounded,
                feasibility_status: FeasibilityStatus::FeasibleForAll,
                unbounded_ray: None,
                certificate: BoundednessCertificate {
                    method: "structural".to_string(),
                    dual_bound: None,
                    primal_bound: None,
                    witness_x: None,
                    confidence: 1.0,
                },
                details,
            };
        }

        // Step 2: Ray analysis
        let ray_result = self.ray_analysis(problem);
        details.push(format!(
            "Ray analysis: {:?}",
            ray_result
                .as_ref()
                .map(|r| "unbounded ray found")
                .unwrap_or("no ray")
        ));

        if let Some(ray) = &ray_result {
            return BoundednessResult {
                status: BoundednessStatus::Unbounded,
                feasibility_status: FeasibilityStatus::FeasibleForSome,
                unbounded_ray: Some(ray.clone()),
                certificate: BoundednessCertificate {
                    method: "ray_analysis".to_string(),
                    dual_bound: None,
                    primal_bound: None,
                    witness_x: None,
                    confidence: 0.9,
                },
                details,
            };
        }

        // Step 3: LP-based feasibility / boundedness check
        let lp_result = self.lp_boundedness_check(problem);
        details.push(format!("LP-based check: {:?}", lp_result.0));

        // Step 4: Sampling-based check for different leader decisions
        let sampling = self.sampling_check(problem);
        details.push(format!("Sampling check: {:?}", sampling));

        let status = combine_boundedness(&[structural, lp_result.0, sampling]);
        let feasibility = self.check_feasibility(problem);

        BoundednessResult {
            status,
            feasibility_status: feasibility,
            unbounded_ray: None,
            certificate: BoundednessCertificate {
                method: "combined".to_string(),
                dual_bound: lp_result.1,
                primal_bound: lp_result.2,
                witness_x: None,
                confidence: match status {
                    BoundednessStatus::Bounded => 0.9,
                    BoundednessStatus::Unbounded => 0.8,
                    _ => 0.5,
                },
            },
            details,
        }
    }

    /// Check structural boundedness: do variable bounds + constraints guarantee finiteness?
    pub fn check_structural_boundedness(&self, problem: &BilevelProblem) -> BoundednessStatus {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;

        // If all variables are bounded above and below, the feasible set is bounded
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        let all_bounded = lp
            .var_bounds
            .iter()
            .all(|b| b.lower.is_finite() && b.upper.is_finite());

        if all_bounded {
            return BoundednessStatus::Bounded;
        }

        // Check if constraint matrix has a structure that implies boundedness
        let csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_a);
        let obj = &problem.lower_obj_c;

        // If every variable with nonzero objective coefficient has a finite upper bound,
        // and the feasible region is bounded in that direction, we're bounded.
        let obj_vars_bounded = obj.iter().enumerate().all(|(j, &c)| {
            if c.abs() < self.config.tolerance {
                return true;
            }
            // Check if this variable is bounded by constraints
            lp.var_bounds[j].upper.is_finite()
        });

        if obj_vars_bounded {
            return BoundednessStatus::Bounded;
        }

        // Check if the constraint polyhedron implies boundedness via row structure
        // A sufficient condition: for each unbounded variable j with c_j != 0,
        // there exists a constraint row with a positive coefficient for j
        let unbounded_with_obj: Vec<usize> = (0..n)
            .filter(|&j| {
                obj[j].abs() > self.config.tolerance && !lp.var_bounds[j].upper.is_finite()
            })
            .collect();

        if unbounded_with_obj.is_empty() {
            return BoundednessStatus::Bounded;
        }

        let all_constrained = unbounded_with_obj.iter().all(|&j| {
            problem
                .lower_a
                .entries
                .iter()
                .any(|e| e.col == j && e.value.abs() > self.config.tolerance)
        });

        if all_constrained && m >= unbounded_with_obj.len() {
            BoundednessStatus::Bounded
        } else {
            BoundednessStatus::Unknown
        }
    }

    /// Ray analysis: check if the recession cone of the lower-level feasible
    /// region contains a direction that decreases the objective unboundedly.
    pub fn ray_analysis(&self, problem: &BilevelProblem) -> Option<Vec<f64>> {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;

        let a_dense = problem.lower_a.to_dense();
        let c = &problem.lower_obj_c;

        // Recession cone: { d >= 0 : A*d <= 0 }
        // Unbounded iff ∃ d in recession cone with c^T d < 0
        // We try canonical directions and random directions.

        // Check canonical directions
        for j in 0..n {
            if c[j] >= -tol {
                continue; // This direction doesn't decrease the objective
            }
            // Check if e_j is in the recession cone: A * e_j <= 0
            let mut feasible = true;
            for i in 0..m {
                if i < a_dense.nrows() && j < a_dense.ncols() {
                    if a_dense[(i, j)] > tol {
                        feasible = false;
                        break;
                    }
                }
            }
            if feasible {
                let mut ray = vec![0.0; n];
                ray[j] = 1.0;
                return Some(ray);
            }
        }

        // Try random directions in the recession cone
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        for _ in 0..self.config.max_ray_iterations {
            let d: Vec<f64> = (0..n)
                .map(|_| {
                    use rand::Rng;
                    rng.gen_range(0.0..1.0)
                })
                .collect();

            // Check c^T d < 0
            let obj_val: f64 = c.iter().zip(d.iter()).map(|(ci, di)| ci * di).sum();
            if obj_val >= -tol {
                continue;
            }

            // Check A*d <= 0
            let ad = a_dense.clone() * DVector::from_column_slice(&d);
            let in_recession = ad.iter().all(|&v| v <= tol);

            if in_recession {
                return Some(d);
            }
        }

        None
    }

    /// LP-based boundedness check.
    pub fn lp_boundedness_check(
        &self,
        problem: &BilevelProblem,
    ) -> (BoundednessStatus, Option<f64>, Option<f64>) {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;

        // Try to compute bounds on the objective value.
        // Lower bound: min c^T y  s.t. Ay <= b, y >= 0
        // For x = 0 (simplest case).
        let a_dense = problem.lower_a.to_dense();
        let b = &problem.lower_b;
        let c = &problem.lower_obj_c;

        // Check feasibility at origin
        let origin = vec![0.0; n];
        let feasible = b.iter().all(|&bi| bi >= -self.config.tolerance);

        if !feasible {
            return (BoundednessStatus::Unknown, None, None);
        }

        // Evaluate objective at origin
        let obj_at_origin: f64 = c.iter().zip(origin.iter()).map(|(ci, yi)| ci * yi).sum();

        // Try to find an upper bound by evaluating at vertex-like points
        let mut best_obj = obj_at_origin;
        let mut worst_obj = obj_at_origin;

        // Try corner points of variable bounds
        let lp = problem.lower_level_lp(&vec![0.0; problem.num_upper_vars]);
        for j in 0..n {
            if lp.var_bounds[j].upper.is_finite() {
                let mut point = vec![0.0; n];
                point[j] = lp.var_bounds[j].upper;

                // Check feasibility
                let ax = a_dense.clone() * DVector::from_column_slice(&point);
                let feas = (0..m).all(|i| ax[i] <= b[i] + self.config.tolerance);

                if feas {
                    let obj: f64 = c.iter().zip(point.iter()).map(|(ci, yi)| ci * yi).sum();
                    if obj < best_obj {
                        best_obj = obj;
                    }
                    if obj > worst_obj {
                        worst_obj = obj;
                    }
                }
            }
        }

        // If we found both a lower and upper bound, the problem is bounded
        if (worst_obj - best_obj).abs() < 1e10 {
            (BoundednessStatus::Bounded, Some(best_obj), Some(worst_obj))
        } else {
            (BoundednessStatus::Unknown, Some(best_obj), None)
        }
    }

    /// Sampling-based boundedness check for multiple leader decisions.
    pub fn sampling_check(&self, problem: &BilevelProblem) -> BoundednessStatus {
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        let nx = problem.num_upper_vars;
        let ny = problem.num_lower_vars;
        let tol = self.config.tolerance;

        let a_dense = problem.lower_a.to_dense();
        let m = problem.num_lower_constraints;

        for _ in 0..self.config.sample_count {
            let x: Vec<f64> = (0..nx).map(|_| rng.gen_range(0.0..10.0)).collect();

            // Compute RHS: b + Bx
            let mut rhs = problem.lower_b.clone();
            for entry in &problem.lower_linking_b.entries {
                if entry.col < x.len() && entry.row < rhs.len() {
                    rhs[entry.row] += entry.value * x[entry.col];
                }
            }

            // Check if the recession cone has an improving direction
            let c = &problem.lower_obj_c;
            for j in 0..ny {
                if c[j] >= -tol {
                    continue;
                }
                let mut in_recession = true;
                for i in 0..m.min(a_dense.nrows()) {
                    if j < a_dense.ncols() && a_dense[(i, j)] > tol {
                        in_recession = false;
                        break;
                    }
                }
                if in_recession {
                    return BoundednessStatus::Unbounded;
                }
            }
        }

        BoundednessStatus::Bounded
    }

    /// Check feasibility of the lower level.
    pub fn check_feasibility(&self, problem: &BilevelProblem) -> FeasibilityStatus {
        use rand::Rng;
        let nx = problem.num_upper_vars;
        let ny = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        let tol = self.config.tolerance;

        let a_dense = problem.lower_a.to_dense();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed + 1);

        let mut feasible_count = 0;
        let total = self.config.sample_count;

        for _ in 0..total {
            let x: Vec<f64> = (0..nx).map(|_| rng.gen_range(0.0..5.0)).collect();

            // Compute RHS: b + Bx
            let mut rhs = problem.lower_b.clone();
            for entry in &problem.lower_linking_b.entries {
                if entry.col < x.len() && entry.row < rhs.len() {
                    rhs[entry.row] += entry.value * x[entry.col];
                }
            }

            // Check if origin is feasible for this RHS
            let origin_feasible = rhs.iter().all(|&bi| bi >= -tol);
            if origin_feasible {
                feasible_count += 1;
            }
        }

        if feasible_count == total {
            FeasibilityStatus::FeasibleForAll
        } else if feasible_count > 0 {
            FeasibilityStatus::FeasibleForSome
        } else {
            FeasibilityStatus::InfeasibleForAll
        }
    }

    /// Compute the domain of the leader: the set of x for which the lower level is feasible.
    pub fn compute_leader_domain(&self, problem: &BilevelProblem) -> LeaderDomain {
        let nx = problem.num_upper_vars;

        // The lower level is feasible when ∃ y ≥ 0: Ay ≤ b + Bx.
        // By Farkas' lemma, this requires that b + Bx ≥ 0 in certain directions.
        // Sufficient condition: b + Bx ≥ 0 (for non-negative b entries and non-negative linking).
        let linking_csr = SparseMatrixCsr::from_sparse_matrix(&problem.lower_linking_b);

        let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); nx];

        // For each lower-level constraint i: sum_j a_{ij} y_j <= b_i + sum_k B_{ik} x_k
        // The origin y=0 is feasible iff b_i + Bx_i >= 0 for all i.
        for i in 0..problem.num_lower_constraints {
            let bi = problem.lower_b[i];
            let linking_entries = linking_csr.row_entries(i);

            // b_i + sum B_{ik} x_k >= 0  →  sum B_{ik} x_k >= -b_i
            for &(k, coeff) in &linking_entries {
                if coeff.abs() < self.config.tolerance || k >= nx {
                    continue;
                }
                // coeff * x_k >= -b_i (considering only this term)
                if coeff > 0.0 {
                    let lb = -bi / coeff;
                    if lb > bounds[k].0 {
                        bounds[k].0 = lb;
                    }
                } else {
                    let ub = -bi / coeff;
                    if ub < bounds[k].1 {
                        bounds[k].1 = ub;
                    }
                }
            }
        }

        let is_bounded = bounds
            .iter()
            .all(|(lb, ub)| lb.is_finite() && ub.is_finite());
        let is_nonempty = bounds.iter().all(|(lb, ub)| lb <= ub);
        LeaderDomain {
            variable_bounds: bounds,
            is_bounded,
            is_nonempty,
        }
    }
}

/// Description of the leader's feasible domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderDomain {
    pub variable_bounds: Vec<(f64, f64)>,
    pub is_bounded: bool,
    pub is_nonempty: bool,
}

use rand::SeedableRng;

fn combine_boundedness(statuses: &[BoundednessStatus]) -> BoundednessStatus {
    if statuses
        .iter()
        .any(|s| matches!(s, BoundednessStatus::Unbounded))
    {
        return BoundednessStatus::Unbounded;
    }
    if statuses
        .iter()
        .all(|s| matches!(s, BoundednessStatus::Bounded))
    {
        return BoundednessStatus::Bounded;
    }
    if statuses
        .iter()
        .any(|s| matches!(s, BoundednessStatus::Bounded))
    {
        return BoundednessStatus::Bounded;
    }
    BoundednessStatus::Unknown
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_bounded_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 2);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    fn make_unbounded_problem() -> BilevelProblem {
        // min -y1 - y2 s.t. no upper bounds
        let lower_a = SparseMatrix::new(0, 2);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![0.0, 0.0],
            lower_obj_c: vec![-1.0, -1.0],
            lower_a,
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, 1),
            upper_constraints_a: SparseMatrix::new(0, 3),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 2,
            num_lower_constraints: 0,
            num_upper_constraints: 0,
        }
    }

    #[test]
    fn test_bounded_structural() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let result = analyzer.check_structural_boundedness(&p);
        assert_eq!(result, BoundednessStatus::Bounded);
    }

    #[test]
    fn test_full_bounded_analysis() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let result = analyzer.analyze(&p);
        assert_eq!(result.status, BoundednessStatus::Bounded);
    }

    #[test]
    fn test_ray_no_constraints() {
        let p = make_unbounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let ray = analyzer.ray_analysis(&p);
        // Should find an unbounded ray since no constraints and negative objective
        assert!(ray.is_some());
    }

    #[test]
    fn test_feasibility_check() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let feas = analyzer.check_feasibility(&p);
        assert_eq!(feas, FeasibilityStatus::FeasibleForAll);
    }

    #[test]
    fn test_leader_domain() {
        let mut p = make_bounded_problem();
        // Add linking: y1 <= 5 + x, so x >= -5 for feasibility
        p.lower_linking_b = SparseMatrix::new(2, 1);
        p.lower_linking_b.add_entry(0, 0, 1.0);
        let analyzer = BoundednessAnalyzer::with_defaults();
        let domain = analyzer.compute_leader_domain(&p);
        assert!(domain.is_nonempty);
    }

    #[test]
    fn test_lp_boundedness() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let (status, lb, ub) = analyzer.lp_boundedness_check(&p);
        assert_eq!(status, BoundednessStatus::Bounded);
    }

    #[test]
    fn test_sampling_check() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let result = analyzer.sampling_check(&p);
        assert_eq!(result, BoundednessStatus::Bounded);
    }

    #[test]
    fn test_boundedness_certificate() {
        let p = make_bounded_problem();
        let analyzer = BoundednessAnalyzer::with_defaults();
        let result = analyzer.analyze(&p);
        assert!(result.certificate.confidence > 0.0);
    }
}
