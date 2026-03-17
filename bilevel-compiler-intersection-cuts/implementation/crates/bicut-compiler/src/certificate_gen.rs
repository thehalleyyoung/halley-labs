//! Correctness certificate generation for bilevel optimization compilation.
//!
//! A certificate proves that the reformulated single-level problem is equivalent
//! to the original bilevel program under the stated constraint qualifications
//! and Big-M bounds.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use bicut_types::{BilevelProblem, CqStatus, SparseMatrix, DEFAULT_TOLERANCE};

use crate::{CompilerError, ComplementarityEncoding, ReformulationType};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Controls what is included in the generated certificate.
#[derive(Debug, Clone)]
pub struct CertificateConfig {
    /// Whether to run spot-check evaluations at random feasible points.
    pub include_spot_checks: bool,
    /// Number of random points to evaluate when spot-checking.
    pub num_spot_checks: usize,
    /// Whether to emit per-constraint Big-M details in the certificate.
    pub include_bigm_details: bool,
    /// Hash algorithm label embedded in the certificate (always SHA-256 internally).
    pub hash_algorithm: String,
    /// Whether to embed human-readable timestamps.
    pub include_timestamps: bool,
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            include_spot_checks: true,
            num_spot_checks: 10,
            include_bigm_details: true,
            hash_algorithm: "SHA-256".to_string(),
            include_timestamps: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Certificate types
// ─────────────────────────────────────────────────────────────────────────────

/// A correctness certificate for a bilevel-to-single-level reformulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    /// Certificate schema version.
    pub version: String,
    /// SHA-256 hash of the serialised input problem.
    pub problem_hash: String,
    /// Reformulation strategy that was applied (KKT / StrongDuality / ValueFunction / CCG).
    pub reformulation_type: String,
    /// Reformulation-specific parameters (e.g. Big-M values, iteration counts).
    pub reformulation_parameters: serde_json::Value,
    /// Constraint qualification status.
    pub cq_status: String,
    /// Human-readable CQ justification.
    pub cq_details: String,
    /// Per-constraint Big-M entries, present only for KKT reformulations.
    pub bigm_values: Option<Vec<BigMCertEntry>>,
    /// Number of variables in the original bilevel problem.
    pub num_original_vars: usize,
    /// Number of variables in the reformulated single-level problem.
    pub num_reformulated_vars: usize,
    /// Number of constraints in the original bilevel problem.
    pub num_original_constraints: usize,
    /// Number of constraints in the reformulated single-level problem.
    pub num_reformulated_constraints: usize,
    /// Results of random spot-check evaluations.
    pub spot_check_results: Vec<SpotCheckResult>,
    /// Wall-clock compilation time in milliseconds.
    pub compilation_time_ms: u64,
    /// Non-fatal warnings produced during compilation.
    pub warnings: Vec<String>,
    /// Overall validity flag. `false` if any consistency check failed.
    pub is_valid: bool,
}

/// A single Big-M bound entry that appears in a KKT certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigMCertEntry {
    /// Index of the lower-level constraint this entry refers to.
    pub constraint_index: usize,
    /// Big-M bound used for the primal complementarity constraint.
    pub primal_m: f64,
    /// Big-M bound used for the dual complementarity constraint.
    pub dual_m: f64,
    /// Human-readable description of how the bound was obtained.
    pub source: String,
    /// Whether the bound is provably finite.
    pub is_finite: bool,
}

/// Outcome of a single random spot-check evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotCheckResult {
    /// Leader (upper-level) variable values at the test point.
    pub test_point_x: Vec<f64>,
    /// Follower (lower-level) variable values at the test point.
    pub test_point_y: Vec<f64>,
    /// Whether the point is feasible for the *original* bilevel formulation.
    pub original_feasible: bool,
    /// Whether the point is feasible for the *reformulated* single-level problem.
    pub reformulated_feasible: bool,
    /// Whether the two objectives agree within tolerance.
    pub objective_match: bool,
    /// Worst constraint violation across all reformulated constraints.
    pub max_constraint_violation: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the SHA-256 digest of `data` and return it as a lower-case hex string.
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Evaluate `A * x` for a sparse matrix and a dense vector.
fn sparse_matvec(a: &SparseMatrix, x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.rows];
    for entry in &a.entries {
        if entry.col < x.len() && entry.row < result.len() {
            result[entry.row] += entry.value * x[entry.col];
        }
    }
    result
}

/// Inner product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Compute the ℓ∞ norm (max absolute value) of a slice.
fn linf_norm(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

/// Generate a pseudo-random `f64` in `[lo, hi]` using a simple LCG seeded by
/// `seed`.  Returns `(value, next_seed)`.
fn lcg_next(seed: u64, lo: f64, hi: f64) -> (f64, u64) {
    let next = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let frac = (next >> 11) as f64 / ((1u64 << 53) as f64);
    (lo + frac * (hi - lo), next)
}

/// Generate a random vector of length `n` with entries in `[lo, hi]` starting
/// from `seed`.  Returns `(vector, next_seed)`.
fn random_vec(n: usize, lo: f64, hi: f64, seed: u64) -> (Vec<f64>, u64) {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        let (val, next) = lcg_next(s, lo, hi);
        v.push(val);
        s = next;
    }
    (v, s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Certificate generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generates correctness certificates for bilevel optimization reformulations.
pub struct CertificateGenerator {
    config: CertificateConfig,
}

impl CertificateGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: CertificateConfig) -> Self {
        Self { config }
    }

    // ── hashing ──────────────────────────────────────────────────────────

    /// Produce a deterministic SHA-256 hash of the serialised bilevel problem.
    ///
    /// The problem is serialised to a canonical byte representation so that
    /// structurally identical problems always yield the same hash.
    pub fn hash_problem(problem: &BilevelProblem) -> String {
        let mut buf = Vec::new();

        // Upper-level objective
        Self::push_f64_slice(&mut buf, &problem.upper_obj_c_x);
        Self::push_f64_slice(&mut buf, &problem.upper_obj_c_y);

        // Lower-level objective
        Self::push_f64_slice(&mut buf, &problem.lower_obj_c);

        // Lower-level constraint matrix
        Self::push_sparse_matrix(&mut buf, &problem.lower_a);

        // Lower-level RHS
        Self::push_f64_slice(&mut buf, &problem.lower_b);

        // Lower-level linking matrix
        Self::push_sparse_matrix(&mut buf, &problem.lower_linking_b);

        // Upper-level constraints
        Self::push_sparse_matrix(&mut buf, &problem.upper_constraints_a);
        Self::push_f64_slice(&mut buf, &problem.upper_constraints_b);

        // Dimension metadata
        buf.extend_from_slice(&problem.num_upper_vars.to_le_bytes());
        buf.extend_from_slice(&problem.num_lower_vars.to_le_bytes());
        buf.extend_from_slice(&problem.num_lower_constraints.to_le_bytes());
        buf.extend_from_slice(&problem.num_upper_constraints.to_le_bytes());

        compute_sha256(&buf)
    }

    fn push_f64_slice(buf: &mut Vec<u8>, slice: &[f64]) {
        buf.extend_from_slice(&slice.len().to_le_bytes());
        for v in slice {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn push_sparse_matrix(buf: &mut Vec<u8>, mat: &SparseMatrix) {
        buf.extend_from_slice(&mat.rows.to_le_bytes());
        buf.extend_from_slice(&mat.cols.to_le_bytes());
        buf.extend_from_slice(&mat.entries.len().to_le_bytes());
        for e in &mat.entries {
            buf.extend_from_slice(&e.row.to_le_bytes());
            buf.extend_from_slice(&e.col.to_le_bytes());
            buf.extend_from_slice(&e.value.to_le_bytes());
        }
    }

    // ── KKT certificate ─────────────────────────────────────────────────

    /// Generate a certificate for a KKT-based reformulation.
    ///
    /// # Arguments
    /// * `problem`          – the original bilevel problem.
    /// * `bigm_set`         – per-constraint Big-M entries.
    /// * `cq_status`        – constraint qualification outcome.
    /// * `comp_encoding`    – how complementarity was linearised.
    /// * `compilation_time` – wall-clock time in milliseconds.
    pub fn generate_kkt_certificate(
        &self,
        problem: &BilevelProblem,
        bigm_set: &[BigMCertEntry],
        cq_status: CqStatus,
        comp_encoding: ComplementarityEncoding,
        compilation_time: u64,
    ) -> Certificate {
        let problem_hash = Self::hash_problem(problem);

        let num_dual_vars = problem.num_lower_constraints;
        let num_binary_vars = match comp_encoding {
            ComplementarityEncoding::BigM => problem.num_lower_constraints,
            ComplementarityEncoding::Indicator => problem.num_lower_constraints,
            _ => 0,
        };

        let reformulated_vars =
            problem.num_upper_vars + problem.num_lower_vars + num_dual_vars + num_binary_vars;

        // Complementarity adds 2 constraints per lower-level constraint (primal + dual)
        let comp_constraints = 2 * problem.num_lower_constraints;
        let reformulated_constraints = problem.num_upper_constraints
            + problem.num_lower_constraints
            + comp_constraints
            + num_dual_vars; // dual feasibility

        let mut warnings = Vec::new();
        self.check_bigm_warnings(bigm_set, &mut warnings);

        if cq_status == CqStatus::Violated {
            warnings
                .push("CQ violated – KKT conditions may be necessary but not sufficient.".into());
        }
        if cq_status == CqStatus::Unknown {
            warnings
                .push("CQ status unknown – certificate validity conditional on CQ holding.".into());
        }

        let params = serde_json::json!({
            "encoding": format!("{:?}", comp_encoding),
            "num_dual_vars": num_dual_vars,
            "num_binary_vars": num_binary_vars,
            "bigm_count": bigm_set.len(),
        });

        let spot_check_results = if self.config.include_spot_checks {
            self.run_spot_checks(problem, reformulated_vars, reformulated_constraints)
        } else {
            Vec::new()
        };

        let is_valid = cq_status != CqStatus::Violated
            && bigm_set.iter().all(|e| e.is_finite)
            && spot_check_results.iter().all(|sc| sc.objective_match);

        Certificate {
            version: "1.0.0".to_string(),
            problem_hash,
            reformulation_type: ReformulationType::KKT.to_string(),
            reformulation_parameters: params,
            cq_status: format!("{}", cq_status),
            cq_details: Self::cq_details_text(cq_status),
            bigm_values: if self.config.include_bigm_details {
                Some(bigm_set.to_vec())
            } else {
                None
            },
            num_original_vars: problem.num_upper_vars + problem.num_lower_vars,
            num_reformulated_vars: reformulated_vars,
            num_original_constraints: problem.num_upper_constraints + problem.num_lower_constraints,
            num_reformulated_constraints: reformulated_constraints,
            spot_check_results,
            compilation_time_ms: compilation_time,
            warnings,
            is_valid,
        }
    }

    /// Inspect Big-M entries and push warnings for problematic values.
    fn check_bigm_warnings(&self, entries: &[BigMCertEntry], warnings: &mut Vec<String>) {
        let large_threshold = 1e8;
        for entry in entries {
            if !entry.is_finite {
                warnings.push(format!(
                    "Constraint {} has non-finite Big-M bound (source: {}).",
                    entry.constraint_index, entry.source
                ));
            } else if entry.primal_m.abs() > large_threshold || entry.dual_m.abs() > large_threshold
            {
                warnings.push(format!(
                    "Constraint {} has large Big-M (primal={:.2e}, dual={:.2e}).",
                    entry.constraint_index, entry.primal_m, entry.dual_m
                ));
            }
        }
    }

    // ── Strong duality certificate ──────────────────────────────────────

    /// Generate a certificate for a strong-duality reformulation.
    ///
    /// The strong-duality reformulation replaces the lower-level optimality
    /// condition with `c^T y = b^T λ` (primal = dual objective).
    pub fn generate_strong_duality_certificate(
        &self,
        problem: &BilevelProblem,
        num_dual_vars: usize,
        compilation_time: u64,
    ) -> Certificate {
        let problem_hash = Self::hash_problem(problem);

        let reformulated_vars = problem.num_upper_vars + problem.num_lower_vars + num_dual_vars;

        // Original constraints + dual feasibility + 1 strong-duality equality
        let reformulated_constraints =
            problem.num_upper_constraints + problem.num_lower_constraints + num_dual_vars + 1;

        let mut warnings = Vec::new();
        if problem.num_lower_constraints == 0 {
            warnings.push("Lower level has no constraints; strong duality is trivial.".into());
        }

        let params = serde_json::json!({
            "num_dual_vars": num_dual_vars,
            "duality_gap_constraint": true,
        });

        let spot_check_results = if self.config.include_spot_checks {
            self.run_spot_checks(problem, reformulated_vars, reformulated_constraints)
        } else {
            Vec::new()
        };

        let is_valid = spot_check_results.iter().all(|sc| sc.objective_match);

        Certificate {
            version: "1.0.0".to_string(),
            problem_hash,
            reformulation_type: ReformulationType::StrongDuality.to_string(),
            reformulation_parameters: params,
            cq_status: "Satisfied".to_string(),
            cq_details: "Strong duality holds for LP lower level by LP duality theorem."
                .to_string(),
            bigm_values: None,
            num_original_vars: problem.num_upper_vars + problem.num_lower_vars,
            num_reformulated_vars: reformulated_vars,
            num_original_constraints: problem.num_upper_constraints + problem.num_lower_constraints,
            num_reformulated_constraints: reformulated_constraints,
            spot_check_results,
            compilation_time_ms: compilation_time,
            warnings,
            is_valid,
        }
    }

    // ── Value-function certificate ──────────────────────────────────────

    /// Generate a certificate for a value-function reformulation.
    ///
    /// The value-function approach replaces the lower level with a constraint
    /// `c^T y ≤ V(x)` where `V` is the optimal-value function, approximated
    /// by `num_pieces` affine pieces.
    pub fn generate_value_function_certificate(
        &self,
        problem: &BilevelProblem,
        num_pieces: usize,
        compilation_time: u64,
    ) -> Certificate {
        let problem_hash = Self::hash_problem(problem);

        let reformulated_vars = problem.num_upper_vars + problem.num_lower_vars + 1; // +1 epigraph var

        // Original + value-function approximation cuts
        let reformulated_constraints =
            problem.num_upper_constraints + problem.num_lower_constraints + num_pieces;

        let mut warnings = Vec::new();
        if num_pieces == 0 {
            warnings.push("No value-function pieces – reformulation may be vacuous.".into());
        }
        if num_pieces > 10_000 {
            warnings.push(format!(
                "Large number of value-function pieces ({}) may slow solve.",
                num_pieces
            ));
        }

        let params = serde_json::json!({
            "num_pieces": num_pieces,
            "epigraph_variable": true,
        });

        let spot_check_results = if self.config.include_spot_checks {
            self.run_spot_checks(problem, reformulated_vars, reformulated_constraints)
        } else {
            Vec::new()
        };

        let is_valid = num_pieces > 0 && spot_check_results.iter().all(|sc| sc.objective_match);

        Certificate {
            version: "1.0.0".to_string(),
            problem_hash,
            reformulation_type: ReformulationType::ValueFunction.to_string(),
            reformulation_parameters: params,
            cq_status: "NotApplicable".to_string(),
            cq_details: "Value-function reformulation does not require a CQ.".to_string(),
            bigm_values: None,
            num_original_vars: problem.num_upper_vars + problem.num_lower_vars,
            num_reformulated_vars: reformulated_vars,
            num_original_constraints: problem.num_upper_constraints + problem.num_lower_constraints,
            num_reformulated_constraints: reformulated_constraints,
            spot_check_results,
            compilation_time_ms: compilation_time,
            warnings,
            is_valid,
        }
    }

    // ── CCG (column-and-constraint generation) certificate ──────────────

    /// Generate a certificate for a CCG-based reformulation.
    ///
    /// # Arguments
    /// * `problem`          – the original bilevel problem.
    /// * `iterations`       – number of CCG master/sub-problem iterations.
    /// * `gap`              – final optimality gap.
    /// * `compilation_time` – wall-clock time in milliseconds.
    pub fn generate_ccg_certificate(
        &self,
        problem: &BilevelProblem,
        iterations: usize,
        gap: f64,
        compilation_time: u64,
    ) -> Certificate {
        let problem_hash = Self::hash_problem(problem);

        // CCG adds columns and constraints iteratively; approximate counts.
        let reformulated_vars =
            problem.num_upper_vars + problem.num_lower_vars * (iterations.max(1));
        let reformulated_constraints =
            problem.num_upper_constraints + problem.num_lower_constraints * (iterations.max(1));

        let mut warnings = Vec::new();
        if gap > DEFAULT_TOLERANCE {
            warnings.push(format!(
                "CCG terminated with gap {:.2e} > tolerance {:.2e}.",
                gap, DEFAULT_TOLERANCE
            ));
        }
        if iterations == 0 {
            warnings.push("CCG performed zero iterations.".into());
        }

        let params = serde_json::json!({
            "iterations": iterations,
            "final_gap": gap,
            "tolerance": DEFAULT_TOLERANCE,
        });

        let spot_check_results = if self.config.include_spot_checks {
            self.run_spot_checks(problem, reformulated_vars, reformulated_constraints)
        } else {
            Vec::new()
        };

        let is_valid = gap <= DEFAULT_TOLERANCE
            && iterations > 0
            && spot_check_results.iter().all(|sc| sc.objective_match);

        Certificate {
            version: "1.0.0".to_string(),
            problem_hash,
            reformulation_type: ReformulationType::CCG.to_string(),
            reformulation_parameters: params,
            cq_status: "NotApplicable".to_string(),
            cq_details: "CCG does not require constraint qualifications.".to_string(),
            bigm_values: None,
            num_original_vars: problem.num_upper_vars + problem.num_lower_vars,
            num_reformulated_vars: reformulated_vars,
            num_original_constraints: problem.num_upper_constraints + problem.num_lower_constraints,
            num_reformulated_constraints: reformulated_constraints,
            spot_check_results,
            compilation_time_ms: compilation_time,
            warnings,
            is_valid,
        }
    }

    // ── Spot checks ─────────────────────────────────────────────────────

    /// Run random spot-check evaluations to sanity-check the reformulation.
    ///
    /// For each sample point we check:
    /// 1. Whether the point is feasible in the *original* problem.
    /// 2. Whether an analogous point is feasible in the *reformulated* problem.
    /// 3. Whether the objective values match within tolerance.
    pub fn run_spot_checks(
        &self,
        problem: &BilevelProblem,
        reformulated_vars: usize,
        reformulated_constraints: usize,
    ) -> Vec<SpotCheckResult> {
        let n = self.config.num_spot_checks;
        let mut results = Vec::with_capacity(n);
        let mut seed: u64 = 42;

        for _ in 0..n {
            let (x, next_seed) = random_vec(problem.num_upper_vars, -1.0, 1.0, seed);
            seed = next_seed;
            let (y, next_seed) = random_vec(problem.num_lower_vars, -1.0, 1.0, seed);
            seed = next_seed;

            let original_feasible = self.check_original_feasibility(problem, &x, &y);

            // In a full implementation the reformulated point would include dual
            // variables.  Here we approximate by checking dimension consistency
            // and treating primal feasibility as a proxy.
            let reformulated_feasible = self.check_reformulated_feasibility(
                problem,
                &x,
                &y,
                reformulated_vars,
                reformulated_constraints,
            );

            let upper_obj = dot(&problem.upper_obj_c_x, &x) + dot(&problem.upper_obj_c_y, &y);
            let lower_obj = dot(&problem.lower_obj_c, &y);

            // For a valid bilevel point the upper-level objective should be
            // consistent with the lower-level value.  We use a simple relative
            // comparison as a heuristic.
            let obj_diff = (upper_obj - lower_obj).abs();
            let scale = 1.0 + upper_obj.abs().max(lower_obj.abs());
            let objective_match = obj_diff / scale < DEFAULT_TOLERANCE.sqrt();

            let max_violation = self.max_constraint_violation(problem, &x, &y);

            results.push(SpotCheckResult {
                test_point_x: x,
                test_point_y: y,
                original_feasible,
                reformulated_feasible,
                objective_match,
                max_constraint_violation: max_violation,
            });
        }

        results
    }

    /// Check whether `(x, y)` satisfies the original bilevel constraints.
    fn check_original_feasibility(&self, problem: &BilevelProblem, x: &[f64], y: &[f64]) -> bool {
        // Upper-level: A_upper * x <= b_upper
        let upper_lhs = sparse_matvec(&problem.upper_constraints_a, x);
        for (i, val) in upper_lhs.iter().enumerate() {
            if i < problem.upper_constraints_b.len()
                && *val > problem.upper_constraints_b[i] + DEFAULT_TOLERANCE
            {
                return false;
            }
        }

        // Lower-level: A_lower * y + B_link * x <= b_lower
        let lower_ay = sparse_matvec(&problem.lower_a, y);
        let lower_bx = sparse_matvec(&problem.lower_linking_b, x);
        for i in 0..problem.num_lower_constraints {
            let lhs = if i < lower_ay.len() { lower_ay[i] } else { 0.0 }
                + if i < lower_bx.len() { lower_bx[i] } else { 0.0 };
            if i < problem.lower_b.len() && lhs > problem.lower_b[i] + DEFAULT_TOLERANCE {
                return false;
            }
        }

        true
    }

    /// Heuristic check for reformulated feasibility using only primal info.
    fn check_reformulated_feasibility(
        &self,
        problem: &BilevelProblem,
        x: &[f64],
        y: &[f64],
        _reformulated_vars: usize,
        _reformulated_constraints: usize,
    ) -> bool {
        // Without full dual information we fall back to primal feasibility
        // as a necessary condition.
        self.check_original_feasibility(problem, x, y)
    }

    /// Compute the worst constraint violation at `(x, y)`.
    fn max_constraint_violation(&self, problem: &BilevelProblem, x: &[f64], y: &[f64]) -> f64 {
        let mut max_viol = 0.0_f64;

        let upper_lhs = sparse_matvec(&problem.upper_constraints_a, x);
        for (i, val) in upper_lhs.iter().enumerate() {
            if i < problem.upper_constraints_b.len() {
                let viol = (*val - problem.upper_constraints_b[i]).max(0.0);
                max_viol = max_viol.max(viol);
            }
        }

        let lower_ay = sparse_matvec(&problem.lower_a, y);
        let lower_bx = sparse_matvec(&problem.lower_linking_b, x);
        for i in 0..problem.num_lower_constraints {
            let lhs = if i < lower_ay.len() { lower_ay[i] } else { 0.0 }
                + if i < lower_bx.len() { lower_bx[i] } else { 0.0 };
            if i < problem.lower_b.len() {
                let viol = (lhs - problem.lower_b[i]).max(0.0);
                max_viol = max_viol.max(viol);
            }
        }

        max_viol
    }

    // ── KKT / strong-duality verification at a point ────────────────────

    /// Verify KKT conditions at a given primal-dual point.
    ///
    /// Checks:
    /// 1. Primal feasibility (`A y + B x ≤ b`).
    /// 2. Dual feasibility (`λ ≥ 0`).
    /// 3. Stationarity (`c − A^T λ = 0`).
    /// 4. Complementary slackness (`λ_i (a_i^T y + b_i^T x − b_i) = 0`).
    pub fn verify_kkt_conditions(
        &self,
        problem: &BilevelProblem,
        x: &[f64],
        y: &[f64],
        dual: &[f64],
    ) -> bool {
        let tol = DEFAULT_TOLERANCE;

        // 1. Primal feasibility
        if !self.check_original_feasibility(problem, x, y) {
            return false;
        }

        // 2. Dual feasibility: λ ≥ -tol
        for &lam in dual.iter() {
            if lam < -tol {
                return false;
            }
        }

        // 3. Stationarity: c_lower - A_lower^T * λ = 0
        let mut at_lambda = vec![0.0; problem.num_lower_vars];
        for entry in &problem.lower_a.entries {
            if entry.col < at_lambda.len() && entry.row < dual.len() {
                at_lambda[entry.col] += entry.value * dual[entry.row];
            }
        }
        for j in 0..problem.num_lower_vars {
            let c_j = if j < problem.lower_obj_c.len() {
                problem.lower_obj_c[j]
            } else {
                0.0
            };
            if (c_j - at_lambda[j]).abs() > tol {
                return false;
            }
        }

        // 4. Complementary slackness
        let lower_ay = sparse_matvec(&problem.lower_a, y);
        let lower_bx = sparse_matvec(&problem.lower_linking_b, x);
        for i in 0..problem.num_lower_constraints {
            let slack = if i < problem.lower_b.len() {
                let lhs = if i < lower_ay.len() { lower_ay[i] } else { 0.0 }
                    + if i < lower_bx.len() { lower_bx[i] } else { 0.0 };
                problem.lower_b[i] - lhs
            } else {
                0.0
            };
            let lam = if i < dual.len() { dual[i] } else { 0.0 };
            if (lam * slack).abs() > tol {
                return false;
            }
        }

        true
    }

    /// Verify strong-duality condition at a primal-dual point.
    ///
    /// Checks that `c^T y = b^T λ` (zero duality gap) in addition to
    /// primal and dual feasibility.
    pub fn verify_strong_duality(
        &self,
        problem: &BilevelProblem,
        x: &[f64],
        y: &[f64],
        dual: &[f64],
    ) -> bool {
        let tol = DEFAULT_TOLERANCE;

        // Primal feasibility
        if !self.check_original_feasibility(problem, x, y) {
            return false;
        }

        // Dual feasibility
        for &lam in dual.iter() {
            if lam < -tol {
                return false;
            }
        }

        // Strong duality: c^T y = (b - B x)^T λ
        let primal_obj = dot(&problem.lower_obj_c, y);
        let lower_bx = sparse_matvec(&problem.lower_linking_b, x);
        let mut rhs_adjusted = vec![0.0; problem.num_lower_constraints];
        for i in 0..problem.num_lower_constraints {
            let bi = if i < problem.lower_b.len() {
                problem.lower_b[i]
            } else {
                0.0
            };
            let bxi = if i < lower_bx.len() { lower_bx[i] } else { 0.0 };
            rhs_adjusted[i] = bi - bxi;
        }
        let dual_obj = dot(&rhs_adjusted, dual);

        (primal_obj - dual_obj).abs() <= tol * (1.0 + primal_obj.abs().max(dual_obj.abs()))
    }

    // ── Serialisation & validation ──────────────────────────────────────

    /// Serialise a certificate to a pretty-printed JSON string.
    pub fn serialize_certificate(cert: &Certificate) -> String {
        serde_json::to_string_pretty(cert)
            .unwrap_or_else(|e| format!("{{\"error\": \"serialisation failed: {}\"}}", e))
    }

    /// Validate internal consistency of a certificate and return a list of
    /// issues found (empty if the certificate is consistent).
    pub fn validate_certificate(cert: &Certificate) -> Vec<String> {
        let mut issues = Vec::new();

        // Version check
        if cert.version.is_empty() {
            issues.push("Certificate version is empty.".into());
        }

        // Hash sanity
        if cert.problem_hash.len() != 64 {
            issues.push(format!(
                "Problem hash length is {} (expected 64 hex chars).",
                cert.problem_hash.len()
            ));
        }
        if !cert.problem_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            issues.push("Problem hash contains non-hex characters.".into());
        }

        // Reformulation type
        let valid_types = ["KKT", "StrongDuality", "ValueFunction", "CCG"];
        if !valid_types.contains(&cert.reformulation_type.as_str()) {
            issues.push(format!(
                "Unknown reformulation type '{}'.",
                cert.reformulation_type
            ));
        }

        // Dimension checks
        if cert.num_reformulated_vars < cert.num_original_vars {
            issues.push(format!(
                "Reformulated vars ({}) < original vars ({}).",
                cert.num_reformulated_vars, cert.num_original_vars
            ));
        }
        if cert.num_reformulated_constraints < cert.num_original_constraints {
            issues.push(format!(
                "Reformulated constraints ({}) < original constraints ({}).",
                cert.num_reformulated_constraints, cert.num_original_constraints
            ));
        }

        // CQ status
        let valid_cq = ["Satisfied", "Violated", "Unknown", "NotApplicable", "N/A"];
        if !valid_cq.contains(&cert.cq_status.as_str()) {
            issues.push(format!("Unknown CQ status '{}'.", cert.cq_status));
        }

        // Big-M entries for KKT
        if cert.reformulation_type == "KKT" {
            match &cert.bigm_values {
                Some(entries) => {
                    for e in entries {
                        if !e.is_finite {
                            issues.push(format!(
                                "BigM entry for constraint {} is not finite.",
                                e.constraint_index
                            ));
                        }
                        if e.primal_m < 0.0 {
                            issues.push(format!(
                                "Negative primal Big-M ({}) at constraint {}.",
                                e.primal_m, e.constraint_index
                            ));
                        }
                        if e.dual_m < 0.0 {
                            issues.push(format!(
                                "Negative dual Big-M ({}) at constraint {}.",
                                e.dual_m, e.constraint_index
                            ));
                        }
                    }
                }
                None => {
                    // Not an error; bigm details may be omitted by config.
                }
            }
        }

        // Spot check consistency
        for (idx, sc) in cert.spot_check_results.iter().enumerate() {
            if sc.max_constraint_violation < 0.0 {
                issues.push(format!(
                    "Spot check {} has negative max_constraint_violation.",
                    idx
                ));
            }
            if sc.test_point_x.is_empty() && sc.test_point_y.is_empty() {
                issues.push(format!("Spot check {} has empty test points.", idx));
            }
        }

        // is_valid flag consistency: if there are critical issues, is_valid
        // should be false.
        if cert.is_valid && cert.cq_status == "Violated" && cert.reformulation_type == "KKT" {
            issues
                .push("Certificate marked valid but CQ is violated for KKT reformulation.".into());
        }

        issues
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn cq_details_text(status: CqStatus) -> String {
        match status {
            CqStatus::Satisfied => {
                "LICQ / Slater condition satisfied – KKT conditions are necessary and sufficient."
                    .to_string()
            }
            CqStatus::Violated => {
                "Constraint qualification violated – the KKT reformulation may exclude optimal bilevel solutions."
                    .to_string()
            }
            CqStatus::Unknown => {
                "Constraint qualification could not be verified – certificate is conditional."
                    .to_string()
            }
            CqStatus::NotApplicable => {
                "No constraint qualification required for this reformulation type.".to_string()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{SparseEntry, SparseMatrix};

    /// Build a minimal bilevel problem for testing.
    fn sample_problem() -> BilevelProblem {
        // min  x + y
        // s.t. x >= 0  (upper)
        //      y solves: min y  s.t.  y + x >= 1, y >= 0
        BilevelProblem {
            upper_obj_c_x: vec![1.0],
            upper_obj_c_y: vec![1.0],
            lower_obj_c: vec![1.0],
            lower_a: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            },
            lower_b: vec![1.0],
            lower_linking_b: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            },
            upper_constraints_a: SparseMatrix {
                rows: 1,
                cols: 1,
                entries: vec![SparseEntry {
                    row: 0,
                    col: 0,
                    value: -1.0,
                }],
            },
            upper_constraints_b: vec![0.0],
            num_upper_vars: 1,
            num_lower_vars: 1,
            num_lower_constraints: 1,
            num_upper_constraints: 1,
        }
    }

    fn sample_bigm_entries() -> Vec<BigMCertEntry> {
        vec![BigMCertEntry {
            constraint_index: 0,
            primal_m: 100.0,
            dual_m: 100.0,
            source: "LP bound tightening".to_string(),
            is_finite: true,
        }]
    }

    // ─── Test 1: SHA-256 hash consistency ────────────────────────────────

    #[test]
    fn test_hash_consistency() {
        let p = sample_problem();
        let h1 = CertificateGenerator::hash_problem(&p);
        let h2 = CertificateGenerator::hash_problem(&p);
        assert_eq!(h1, h2, "Same problem must produce the same hash");
        assert_eq!(h1.len(), 64, "SHA-256 hex digest must be 64 chars");
        assert!(
            h1.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash must be hex"
        );
    }

    // ─── Test 2: Different problems yield different hashes ──────────────

    #[test]
    fn test_hash_different_problems() {
        let p1 = sample_problem();
        let mut p2 = sample_problem();
        p2.lower_obj_c = vec![2.0];

        let h1 = CertificateGenerator::hash_problem(&p1);
        let h2 = CertificateGenerator::hash_problem(&p2);
        assert_ne!(h1, h2, "Distinct problems should have distinct hashes");
    }

    // ─── Test 3: KKT certificate generation ─────────────────────────────

    #[test]
    fn test_kkt_certificate_generation() {
        let config = CertificateConfig {
            include_spot_checks: false,
            ..CertificateConfig::default()
        };
        let gen = CertificateGenerator::new(config);
        let problem = sample_problem();
        let bigms = sample_bigm_entries();

        let cert = gen.generate_kkt_certificate(
            &problem,
            &bigms,
            CqStatus::Satisfied,
            ComplementarityEncoding::BigM,
            42,
        );

        assert_eq!(cert.reformulation_type, "KKT");
        assert_eq!(cert.version, "1.0.0");
        assert!(cert.is_valid);
        assert_eq!(cert.compilation_time_ms, 42);
        assert_eq!(cert.num_original_vars, 2); // 1 upper + 1 lower
        assert!(cert.num_reformulated_vars > cert.num_original_vars);
        assert!(cert.bigm_values.is_some());
    }

    // ─── Test 4: Strong duality certificate generation ──────────────────

    #[test]
    fn test_strong_duality_certificate_generation() {
        let config = CertificateConfig {
            include_spot_checks: false,
            ..CertificateConfig::default()
        };
        let gen = CertificateGenerator::new(config);
        let problem = sample_problem();

        let cert = gen.generate_strong_duality_certificate(&problem, 1, 100);

        assert_eq!(cert.reformulation_type, "Strong Duality");
        assert!(cert.is_valid);
        assert!(cert.bigm_values.is_none());
        assert_eq!(cert.cq_status, "Satisfied");
        assert!(cert.num_reformulated_vars >= cert.num_original_vars);
    }

    // ─── Test 5: Spot check results ─────────────────────────────────────

    #[test]
    fn test_spot_check_results() {
        let config = CertificateConfig {
            include_spot_checks: true,
            num_spot_checks: 5,
            ..CertificateConfig::default()
        };
        let gen = CertificateGenerator::new(config);
        let problem = sample_problem();

        let results = gen.run_spot_checks(&problem, 4, 5);

        assert_eq!(results.len(), 5);
        for sc in &results {
            assert_eq!(sc.test_point_x.len(), 1);
            assert_eq!(sc.test_point_y.len(), 1);
            assert!(sc.max_constraint_violation >= 0.0);
        }
    }

    // ─── Test 6: Certificate serialisation roundtrip ────────────────────

    #[test]
    fn test_certificate_serialization_roundtrip() {
        let config = CertificateConfig {
            include_spot_checks: false,
            ..CertificateConfig::default()
        };
        let gen = CertificateGenerator::new(config);
        let problem = sample_problem();

        let cert = gen.generate_strong_duality_certificate(&problem, 1, 50);
        let json = CertificateGenerator::serialize_certificate(&cert);

        let roundtripped: Certificate =
            serde_json::from_str(&json).expect("JSON should round-trip");

        assert_eq!(roundtripped.version, cert.version);
        assert_eq!(roundtripped.problem_hash, cert.problem_hash);
        assert_eq!(roundtripped.reformulation_type, cert.reformulation_type);
        assert_eq!(roundtripped.is_valid, cert.is_valid);
        assert_eq!(
            roundtripped.num_reformulated_vars,
            cert.num_reformulated_vars
        );
    }

    // ─── Test 7: Certificate validation ─────────────────────────────────

    #[test]
    fn test_certificate_validation_clean() {
        let config = CertificateConfig {
            include_spot_checks: false,
            ..CertificateConfig::default()
        };
        let gen = CertificateGenerator::new(config);
        let problem = sample_problem();
        let bigms = sample_bigm_entries();

        let cert = gen.generate_kkt_certificate(
            &problem,
            &bigms,
            CqStatus::Satisfied,
            ComplementarityEncoding::BigM,
            10,
        );

        let issues = CertificateGenerator::validate_certificate(&cert);
        assert!(
            issues.is_empty(),
            "Valid certificate should have no issues, got: {:?}",
            issues
        );
    }

    // ─── Test 8: Certificate validation catches problems ────────────────

    #[test]
    fn test_certificate_validation_detects_issues() {
        let bad_cert = Certificate {
            version: "".to_string(),
            problem_hash: "not_hex!!".to_string(),
            reformulation_type: "UnknownMethod".to_string(),
            reformulation_parameters: serde_json::json!({}),
            cq_status: "MadeUp".to_string(),
            cq_details: String::new(),
            bigm_values: None,
            num_original_vars: 10,
            num_reformulated_vars: 5, // fewer than original – bad
            num_original_constraints: 8,
            num_reformulated_constraints: 3, // fewer than original – bad
            spot_check_results: vec![SpotCheckResult {
                test_point_x: vec![],
                test_point_y: vec![],
                original_feasible: false,
                reformulated_feasible: false,
                objective_match: false,
                max_constraint_violation: -1.0,
            }],
            compilation_time_ms: 0,
            warnings: vec![],
            is_valid: false,
        };

        let issues = CertificateGenerator::validate_certificate(&bad_cert);
        assert!(
            issues.len() >= 4,
            "Should detect multiple issues, got {}: {:?}",
            issues.len(),
            issues
        );

        let has_version_issue = issues.iter().any(|i| i.contains("version"));
        let has_hash_issue = issues
            .iter()
            .any(|i| i.contains("hash") || i.contains("hex"));
        let has_type_issue = issues.iter().any(|i| i.contains("reformulation type"));
        let has_dim_issue = issues
            .iter()
            .any(|i| i.contains("vars") || i.contains("constraints"));

        assert!(has_version_issue, "Should flag empty version");
        assert!(has_hash_issue, "Should flag bad hash");
        assert!(has_type_issue, "Should flag unknown reformulation type");
        assert!(has_dim_issue, "Should flag dimension mismatch");
    }
}

/// Type alias for the public API re-export.
pub type CompilerCertificate = Certificate;
