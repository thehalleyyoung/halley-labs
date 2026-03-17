//! Constraint qualification verification for the lower-level problem.
//!
//! Provides LICQ, MFCQ, and Slater condition checking with a tiered
//! verification approach: syntactic → LP-based → sampling-based.

use bicut_types::{
    BilevelProblem, CqResult, CqStatus, SparseMatrix, SparseMatrixCsr, DEFAULT_TOLERANCE,
};
use log::debug;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The tier at which a CQ was verified (or failed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationTier {
    /// Pure syntactic / structural reasoning (cheapest).
    Syntactic,
    /// LP-based verification.
    LpBased,
    /// Monte-Carlo / sampling-based verification.
    SamplingBased,
    /// Conservative approximation (soundly over-approximate).
    Conservative,
}

/// Configuration for the CQ verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CqVerifierConfig {
    pub tolerance: f64,
    pub max_samples: usize,
    pub seed: u64,
    pub enable_lp_tier: bool,
    pub enable_sampling_tier: bool,
    pub conservative_mode: bool,
}

impl Default for CqVerifierConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            max_samples: 100,
            seed: 42,
            enable_lp_tier: true,
            enable_sampling_tier: true,
            conservative_mode: false,
        }
    }
}

/// Full report from CQ verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CqVerificationReport {
    pub licq: CqResult,
    pub mfcq: CqResult,
    pub slater: CqResult,
    pub tier_used: VerificationTier,
    pub overall_status: CqStatus,
    pub num_active_constraints: usize,
    pub rank_of_active_jacobian: usize,
    pub details: Vec<String>,
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// CQ verification engine.
pub struct CqVerifier {
    config: CqVerifierConfig,
}

impl CqVerifier {
    pub fn new(config: CqVerifierConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(CqVerifierConfig::default())
    }

    /// Run the full tiered verification pipeline.
    pub fn verify(&self, problem: &BilevelProblem) -> CqVerificationReport {
        let mut details = Vec::new();

        // Tier 1: syntactic
        let syntactic = self.verify_syntactic(problem);
        details.push(format!(
            "Syntactic tier: LICQ={}, MFCQ={}, Slater={}",
            syntactic.licq.status, syntactic.mfcq.status, syntactic.slater.status
        ));

        if is_conclusive(&syntactic) {
            return syntactic;
        }

        // Tier 2: LP-based
        if self.config.enable_lp_tier {
            let lp_result = self.verify_lp_based(problem);
            details.push(format!(
                "LP tier: LICQ={}, MFCQ={}, Slater={}",
                lp_result.licq.status, lp_result.mfcq.status, lp_result.slater.status
            ));
            if is_conclusive(&lp_result) {
                let mut result = lp_result;
                result.details = details;
                return result;
            }
        }

        // Tier 3: sampling
        if self.config.enable_sampling_tier {
            let sampling_result = self.verify_sampling(problem);
            details.push(format!(
                "Sampling tier: LICQ={}, MFCQ={}, Slater={}",
                sampling_result.licq.status,
                sampling_result.mfcq.status,
                sampling_result.slater.status
            ));
            let mut result = sampling_result;
            result.details = details;
            return result;
        }

        // Conservative fallback
        let mut report = syntactic;
        report.details = details;
        if self.config.conservative_mode {
            report.overall_status = CqStatus::Unknown;
        }
        report
    }

    /// Tier 1: Syntactic / structural verification.
    pub fn verify_syntactic(&self, problem: &BilevelProblem) -> CqVerificationReport {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;
        let tol = self.config.tolerance;

        // Build the dense constraint matrix
        let a_dense = problem.lower_a.to_dense();

        // If m <= n, LICQ could hold; check row rank
        let (licq, rank) = if m == 0 {
            (
                CqResult::satisfied("LICQ", "No active constraints (vacuously satisfied)"),
                0,
            )
        } else if m > n {
            (
                CqResult::violated(
                    "LICQ",
                    format!(
                        "More constraints ({}) than variables ({}); LICQ cannot hold",
                        m, n
                    ),
                ),
                0,
            )
        } else {
            let r = compute_rank(&a_dense, tol);
            if r == m {
                (
                    CqResult::satisfied(
                        "LICQ",
                        format!("Constraint matrix has full row rank: rank={} = m={}", r, m),
                    ),
                    r,
                )
            } else {
                (
                    CqResult::violated("LICQ", format!("Rank deficient: rank={} < m={}", r, m)),
                    r,
                )
            }
        };

        // MFCQ: relaxed version – requires existence of a direction d such that
        // A_eq d = 0 and A_ineq d < 0 for active inequalities.
        // Syntactic check: if no equality constraints and the matrix has full row rank,
        // MFCQ holds.
        let mfcq = if m == 0 {
            CqResult::satisfied("MFCQ", "No constraints; MFCQ vacuously satisfied")
        } else if rank == m {
            CqResult::satisfied(
                "MFCQ",
                "Full row rank implies MFCQ for inequality-constrained problems",
            )
        } else {
            CqResult::unknown(
                "MFCQ",
                0.5,
                "Cannot determine MFCQ syntactically with rank-deficient constraints",
            )
        };

        // Slater: for LP, Slater condition ≡ strict feasibility of Ay < b.
        let slater = self.check_slater_syntactic(problem);

        let overall = combine_statuses(&[licq.status, mfcq.status, slater.status]);

        CqVerificationReport {
            licq,
            mfcq,
            slater,
            tier_used: VerificationTier::Syntactic,
            overall_status: overall,
            num_active_constraints: m,
            rank_of_active_jacobian: rank,
            details: vec!["Syntactic verification".to_string()],
        }
    }

    /// Tier 2: LP-based verification.
    pub fn verify_lp_based(&self, problem: &BilevelProblem) -> CqVerificationReport {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;
        let tol = self.config.tolerance;

        let a_dense = problem.lower_a.to_dense();
        let b = &problem.lower_b;

        // Try to find a strictly feasible point (for Slater).
        // We check the origin and scaled interior points.
        let slater = self.check_slater_lp_heuristic(problem);

        // For LICQ: evaluate at a candidate interior point and check rank of active constraints.
        let (candidate, active_set) = find_candidate_and_active_set(&a_dense, b, n, tol);
        let rank = if active_set.is_empty() {
            0
        } else {
            let sub = extract_active_rows(&a_dense, &active_set);
            compute_rank(&sub, tol)
        };

        let licq = if active_set.len() > n {
            CqResult::violated(
                "LICQ",
                format!(
                    "At candidate point: {} active constraints > {} variables",
                    active_set.len(),
                    n
                ),
            )
        } else if active_set.is_empty() {
            CqResult::satisfied("LICQ", "No active constraints at candidate interior point")
        } else if rank == active_set.len() {
            CqResult::satisfied(
                "LICQ",
                format!(
                    "Active Jacobian at LP-derived point has full rank: {}/{}",
                    rank,
                    active_set.len()
                ),
            )
        } else {
            CqResult::violated(
                "LICQ",
                format!(
                    "Active Jacobian rank {} < {} active constraints",
                    rank,
                    active_set.len()
                ),
            )
        };

        // MFCQ at the candidate point
        let mfcq = if active_set.is_empty() {
            CqResult::satisfied("MFCQ", "No active constraints → MFCQ trivially holds")
        } else {
            check_mfcq_at_point(&a_dense, &active_set, n, tol)
        };

        let overall = combine_statuses(&[licq.status, mfcq.status, slater.status]);

        CqVerificationReport {
            licq,
            mfcq,
            slater,
            tier_used: VerificationTier::LpBased,
            overall_status: overall,
            num_active_constraints: active_set.len(),
            rank_of_active_jacobian: rank,
            details: vec!["LP-based verification".to_string()],
        }
    }

    /// Tier 3: Sampling-based verification.
    pub fn verify_sampling(&self, problem: &BilevelProblem) -> CqVerificationReport {
        let m = problem.num_lower_constraints;
        let n = problem.num_lower_vars;
        let tol = self.config.tolerance;

        let a_dense = problem.lower_a.to_dense();
        let b = &problem.lower_b;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        let mut best_active = m; // worst case
        let mut best_rank = 0usize;
        let mut slater_found = false;
        let mut licq_violated_ever = false;

        for _ in 0..self.config.max_samples {
            // Generate a random point in [0, 10]^n
            let point: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..10.0)).collect();

            // Compute active set
            let active = compute_active_set_at(&a_dense, b, &point, tol);

            // Check strict feasibility
            if is_strictly_feasible(&a_dense, b, &point, tol) {
                slater_found = true;
            }

            if active.is_empty() {
                continue;
            }

            if active.len() < best_active {
                best_active = active.len();
            }

            // Check LICQ at this point
            if active.len() > n {
                licq_violated_ever = true;
                continue;
            }

            let sub = extract_active_rows(&a_dense, &active);
            let r = compute_rank(&sub, tol);
            if r > best_rank {
                best_rank = r;
            }
            if r < active.len() {
                licq_violated_ever = true;
            }
        }

        let licq = if licq_violated_ever {
            CqResult::unknown(
                "LICQ",
                0.3,
                format!(
                    "LICQ violation observed in {} samples",
                    self.config.max_samples
                ),
            )
        } else if best_active == 0 || best_active == m {
            CqResult::unknown("LICQ", 0.5, "No informative active sets found in sampling")
        } else {
            CqResult::unknown(
                "LICQ",
                0.7,
                format!(
                    "No LICQ violation in {} samples (probabilistic)",
                    self.config.max_samples
                ),
            )
        };

        let slater = if slater_found {
            CqResult::satisfied(
                "Slater",
                format!(
                    "Strictly feasible point found via sampling ({} samples)",
                    self.config.max_samples
                ),
            )
        } else {
            CqResult::unknown(
                "Slater",
                0.4,
                format!(
                    "No strictly feasible point in {} samples",
                    self.config.max_samples
                ),
            )
        };

        let mfcq = if slater_found {
            CqResult::satisfied(
                "MFCQ",
                "Slater condition implies MFCQ for purely inequality-constrained LPs",
            )
        } else {
            CqResult::unknown(
                "MFCQ",
                0.4,
                "Cannot verify MFCQ without Slater point or LP solver",
            )
        };

        let overall = combine_statuses(&[licq.status, mfcq.status, slater.status]);

        CqVerificationReport {
            licq,
            mfcq,
            slater,
            tier_used: VerificationTier::SamplingBased,
            overall_status: overall,
            num_active_constraints: best_active,
            rank_of_active_jacobian: best_rank,
            details: vec![format!(
                "Sampling verification ({} samples)",
                self.config.max_samples
            )],
        }
    }

    /// Slater syntactic check: if there are only inequality constraints and
    /// the RHS has room, strict feasibility is plausible.
    fn check_slater_syntactic(&self, problem: &BilevelProblem) -> CqResult {
        let m = problem.num_lower_constraints;
        if m == 0 {
            return CqResult::satisfied("Slater", "No constraints; Slater vacuously holds");
        }

        // Check origin feasibility: A*0 = 0 ≤ b iff b ≥ 0
        let all_b_positive = problem.lower_b.iter().all(|&bi| bi > self.config.tolerance);
        if all_b_positive {
            return CqResult::satisfied("Slater", "Origin is strictly feasible: all b_i > 0");
        }

        let all_b_nonneg = problem
            .lower_b
            .iter()
            .all(|&bi| bi >= -self.config.tolerance);
        if all_b_nonneg {
            return CqResult::unknown(
                "Slater",
                0.6,
                "Origin is feasible but not strictly; Slater status uncertain",
            );
        }

        CqResult::unknown(
            "Slater",
            0.3,
            "Cannot determine Slater condition syntactically",
        )
    }

    /// Heuristic LP-based Slater check: try several candidate points.
    fn check_slater_lp_heuristic(&self, problem: &BilevelProblem) -> CqResult {
        let n = problem.num_lower_vars;
        let a_dense = problem.lower_a.to_dense();
        let b = &problem.lower_b;
        let tol = self.config.tolerance;

        // Try the origin
        let origin = vec![0.0; n];
        if is_strictly_feasible(&a_dense, b, &origin, tol) {
            return CqResult::satisfied(
                "Slater",
                "Origin is strictly feasible for lower-level constraints",
            );
        }

        // Try centroid of bound box: (0.5, ..., 0.5)
        let centroid = vec![0.5; n];
        if is_strictly_feasible(&a_dense, b, &centroid, tol) {
            return CqResult::satisfied(
                "Slater",
                "Centroid point is strictly feasible for lower-level constraints",
            );
        }

        // Try small positive point
        let small = vec![0.01; n];
        if is_strictly_feasible(&a_dense, b, &small, tol) {
            return CqResult::satisfied("Slater", "Small positive point is strictly feasible");
        }

        // Try to find a strictly feasible point by scaling the RHS
        let min_b = problem
            .lower_b
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_b > tol {
            // b > 0, so y=0 is strictly feasible
            return CqResult::satisfied(
                "Slater",
                "All RHS values strictly positive → origin strictly feasible",
            );
        }

        CqResult::unknown(
            "Slater",
            0.5,
            "LP-based heuristic could not find strictly feasible point",
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use rand::SeedableRng;

fn compute_rank(mat: &DMatrix<f64>, tol: f64) -> usize {
    if mat.nrows() == 0 || mat.ncols() == 0 {
        return 0;
    }
    let svd = mat.clone().svd(false, false);
    svd.singular_values.iter().filter(|&&s| s > tol).count()
}

fn extract_active_rows(a: &DMatrix<f64>, active: &[usize]) -> DMatrix<f64> {
    let n = a.ncols();
    let m = active.len();
    let mut sub = DMatrix::zeros(m, n);
    for (i, &row) in active.iter().enumerate() {
        if row < a.nrows() {
            for j in 0..n {
                sub[(i, j)] = a[(row, j)];
            }
        }
    }
    sub
}

fn compute_active_set_at(a: &DMatrix<f64>, b: &[f64], point: &[f64], tol: f64) -> Vec<usize> {
    let mut active = Vec::new();
    for i in 0..a.nrows().min(b.len()) {
        let mut val = 0.0;
        for j in 0..a.ncols().min(point.len()) {
            val += a[(i, j)] * point[j];
        }
        if (val - b[i]).abs() < tol {
            active.push(i);
        }
    }
    active
}

fn is_strictly_feasible(a: &DMatrix<f64>, b: &[f64], point: &[f64], tol: f64) -> bool {
    for i in 0..a.nrows().min(b.len()) {
        let mut val = 0.0;
        for j in 0..a.ncols().min(point.len()) {
            val += a[(i, j)] * point[j];
        }
        if val >= b[i] - tol {
            return false;
        }
    }
    // Also check non-negativity strictly
    point.iter().all(|&v| v > tol)
}

fn find_candidate_and_active_set(
    a: &DMatrix<f64>,
    b: &[f64],
    n: usize,
    tol: f64,
) -> (Vec<f64>, Vec<usize>) {
    // Try the analytic center heuristic: a small positive point
    let point: Vec<f64> = vec![0.1; n];
    let active = compute_active_set_at(a, b, &point, tol);
    (point, active)
}

fn check_mfcq_at_point(a: &DMatrix<f64>, active: &[usize], n: usize, tol: f64) -> CqResult {
    // MFCQ: ∃ d such that A_{active} d < 0 (for inequality constraints).
    // This requires that the active constraint normals do not positively span R^n.
    // Sufficient condition: rank of active Jacobian < n, or a feasible direction exists.
    let sub = extract_active_rows(a, active);
    let rank = compute_rank(&sub, tol);

    if active.len() < n {
        // Fewer active constraints than variables → a direction always exists
        return CqResult::satisfied(
            "MFCQ",
            format!(
                "{} active constraints < {} variables → MFCQ holds",
                active.len(),
                n
            ),
        );
    }

    if rank < active.len() {
        // Rank deficient active Jacobian might still allow MFCQ
        // Check if −1 vector is in the row space
        let ones = DVector::from_element(n, -1.0);
        let product = &sub * &ones;
        let all_negative = product.iter().all(|&v| v < -tol);
        if all_negative {
            return CqResult::satisfied(
                "MFCQ",
                "Direction d = -1 yields A_active * d < 0 for all active rows",
            );
        }
    }

    // Try random directions
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..50 {
        let d: DVector<f64> = DVector::from_fn(n, |_, _| rng.gen_range(-1.0..1.0));
        let product = &sub * &d;
        if product.iter().all(|&v| v < -tol) {
            return CqResult::satisfied(
                "MFCQ",
                "Random direction found satisfying A_active * d < 0",
            );
        }
    }

    CqResult::unknown(
        "MFCQ",
        0.4,
        "Could not find a direction d with A_active * d < 0",
    )
}

fn is_conclusive(report: &CqVerificationReport) -> bool {
    matches!(
        report.overall_status,
        CqStatus::Satisfied | CqStatus::Violated
    )
}

fn combine_statuses(statuses: &[CqStatus]) -> CqStatus {
    // If any is Satisfied, overall is at least partially good.
    // If all are Satisfied, overall is Satisfied.
    // If any is Violated, overall is Violated (conservative).
    let any_violated = statuses.iter().any(|s| matches!(s, CqStatus::Violated));
    let all_satisfied = statuses.iter().all(|s| matches!(s, CqStatus::Satisfied));
    let any_satisfied = statuses.iter().any(|s| matches!(s, CqStatus::Satisfied));

    if any_violated {
        CqStatus::Violated
    } else if all_satisfied {
        CqStatus::Satisfied
    } else if any_satisfied {
        CqStatus::Satisfied
    } else {
        CqStatus::Unknown
    }
}

// ---------------------------------------------------------------------------
// Standalone helpers (public API)
// ---------------------------------------------------------------------------

/// Check LICQ at a specific point for the lower-level constraints.
pub fn check_licq_at_point(problem: &BilevelProblem, point: &[f64], tol: f64) -> CqResult {
    let a_dense = problem.lower_a.to_dense();
    let active = compute_active_set_at(&a_dense, &problem.lower_b, point, tol);

    if active.is_empty() {
        return CqResult::satisfied("LICQ", "No active constraints at given point");
    }

    let n = problem.num_lower_vars;
    if active.len() > n {
        return CqResult::violated(
            "LICQ",
            format!("{} active constraints > {} variables", active.len(), n),
        );
    }

    let sub = extract_active_rows(&a_dense, &active);
    let rank = compute_rank(&sub, tol);

    if rank == active.len() {
        CqResult::satisfied(
            "LICQ",
            format!("Active Jacobian has full rank {} at point", rank),
        )
    } else {
        CqResult::violated(
            "LICQ",
            format!("Rank {} < {} active constraints", rank, active.len()),
        )
    }
}

/// Check MFCQ at a specific point.
pub fn check_mfcq_at(problem: &BilevelProblem, point: &[f64], tol: f64) -> CqResult {
    let a_dense = problem.lower_a.to_dense();
    let active = compute_active_set_at(&a_dense, &problem.lower_b, point, tol);

    if active.is_empty() {
        return CqResult::satisfied("MFCQ", "No active constraints at given point");
    }

    check_mfcq_at_point(&a_dense, &active, problem.num_lower_vars, tol)
}

/// Conservative CQ approximation: assume worst case if uncertain.
pub fn conservative_cq_check(problem: &BilevelProblem) -> CqVerificationReport {
    let verifier = CqVerifier::new(CqVerifierConfig {
        conservative_mode: true,
        ..Default::default()
    });
    verifier.verify(problem)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_problem_with_full_rank() -> BilevelProblem {
        // 2 constraints, 3 variables, full row rank
        let mut a = SparseMatrix::new(2, 3);
        a.add_entry(0, 0, 1.0);
        a.add_entry(0, 1, 0.0);
        a.add_entry(0, 2, 0.0);
        a.add_entry(1, 0, 0.0);
        a.add_entry(1, 1, 1.0);
        a.add_entry(1, 2, 0.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0, 1.0],
            lower_a: a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 1),
            upper_constraints_a: SparseMatrix::new(0, 4),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 3,
            num_lower_constraints: 2,
            num_upper_constraints: 0,
        }
    }

    fn make_rank_deficient_problem() -> BilevelProblem {
        // 2 identical constraints → rank deficient
        let mut a = SparseMatrix::new(2, 2);
        a.add_entry(0, 0, 1.0);
        a.add_entry(0, 1, 1.0);
        a.add_entry(1, 0, 1.0);
        a.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0, 1.0],
            lower_obj_c: vec![1.0, 1.0],
            lower_a: a,
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

    #[test]
    fn test_syntactic_full_rank_licq() {
        let p = make_problem_with_full_rank();
        let verifier = CqVerifier::with_defaults();
        let report = verifier.verify_syntactic(&p);
        assert_eq!(report.licq.status, CqStatus::Satisfied);
    }

    #[test]
    fn test_syntactic_rank_deficient_licq() {
        let p = make_rank_deficient_problem();
        let verifier = CqVerifier::with_defaults();
        let report = verifier.verify_syntactic(&p);
        assert_eq!(report.licq.status, CqStatus::Violated);
    }

    #[test]
    fn test_slater_origin_feasible() {
        let p = make_problem_with_full_rank();
        let verifier = CqVerifier::with_defaults();
        let report = verifier.verify_syntactic(&p);
        assert_eq!(report.slater.status, CqStatus::Satisfied);
    }

    #[test]
    fn test_full_verification() {
        let p = make_problem_with_full_rank();
        let verifier = CqVerifier::with_defaults();
        let report = verifier.verify(&p);
        assert_eq!(report.overall_status, CqStatus::Satisfied);
    }

    #[test]
    fn test_check_licq_at_point() {
        let p = make_problem_with_full_rank();
        let result = check_licq_at_point(&p, &[0.5, 0.5, 0.5], 1e-8);
        // Point is interior → no active constraints → LICQ satisfied
        assert_eq!(result.status, CqStatus::Satisfied);
    }

    #[test]
    fn test_conservative_mode() {
        let p = make_problem_with_full_rank();
        let report = conservative_cq_check(&p);
        assert!(matches!(
            report.overall_status,
            CqStatus::Satisfied | CqStatus::Unknown
        ));
    }

    #[test]
    fn test_empty_constraints() {
        let p = BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a: SparseMatrix::new(0, 1),
            lower_b: vec![],
            lower_linking_b: SparseMatrix::new(0, 1),
            upper_constraints_a: SparseMatrix::new(0, 2),
            upper_constraints_b: vec![],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 0,
            num_upper_constraints: 0,
        };
        let verifier = CqVerifier::with_defaults();
        let report = verifier.verify(&p);
        assert_eq!(report.overall_status, CqStatus::Satisfied);
    }

    #[test]
    fn test_sampling_tier() {
        let p = make_problem_with_full_rank();
        let verifier = CqVerifier::new(CqVerifierConfig {
            max_samples: 20,
            ..Default::default()
        });
        let report = verifier.verify_sampling(&p);
        assert!(report.details.len() > 0);
    }

    #[test]
    fn test_verification_tier_enum() {
        assert_ne!(VerificationTier::Syntactic, VerificationTier::LpBased);
        assert_ne!(VerificationTier::LpBased, VerificationTier::SamplingBased);
    }
}
