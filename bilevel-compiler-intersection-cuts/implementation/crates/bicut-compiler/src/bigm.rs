//! Big-M computation for bilevel optimisation.
//!
//! When the lower-level LP is reformulated via KKT conditions the
//! complementarity constraints  `s_i · λ_i = 0`  are linearised with
//! binary indicators:
//!
//! ```text
//!   s_i  ≤  M_primal · (1 − z_i)
//!   λ_i  ≤  M_dual   · z_i
//! ```
//!
//! This module computes *tight* values for `M_primal` and `M_dual` using
//! interval arithmetic, LP-based bound tightening, or user-specified
//! constants, in order to strengthen the resulting MIP relaxation.

use crate::*;

// ───────────────────────────── configuration ─────────────────────────────

/// Tunables for Big-M computation.
#[derive(Debug, Clone)]
pub struct BigMConfig {
    /// Fallback M when no tighter bound can be derived.
    pub default_m: f64,
    /// If `true`, attempt LP-based bound tightening.
    pub use_bound_tightening: bool,
    /// Multiplicative safety factor applied *after* tightening (≥ 1.0).
    pub safety_margin_factor: f64,
    /// Maximum LP-based tightening iterations per constraint.
    pub max_tightening_iters: usize,
    /// If `true`, use interval arithmetic as a quick first pass.
    pub interval_arithmetic: bool,
    /// Numerical zero tolerance.
    pub tol: f64,
}

impl Default for BigMConfig {
    fn default() -> Self {
        Self {
            default_m: 1e6,
            use_bound_tightening: true,
            safety_margin_factor: 1.1,
            max_tightening_iters: 20,
            interval_arithmetic: true,
            tol: DEFAULT_TOLERANCE,
        }
    }
}

// ───────────────────────────── result types ──────────────────────────────

/// How a particular Big-M value was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigMSource {
    /// Supplied explicitly by the user / problem data.
    UserSpecified,
    /// Derived via interval arithmetic on variable bounds.
    IntervalArithmetic,
    /// Tightened by solving an LP sub-problem.
    BoundTightening,
    /// Fallback constant from [`BigMConfig::default_m`].
    Default,
}

/// Big-M result for a single complementarity pair.
#[derive(Debug, Clone)]
pub struct BigMResult {
    /// Index of the lower-level constraint this relates to.
    pub constraint_index: usize,
    /// M for the primal slack:  s_i ≤ M_primal · (1 − z_i).
    pub primal_bigm: f64,
    /// M for the dual variable: λ_i ≤ M_dual · z_i.
    pub dual_bigm: f64,
    /// How the values were derived.
    pub source: BigMSource,
    /// `true` when both M values are finite.
    pub is_finite: bool,
    /// Number of LP iterations used during tightening (0 if none).
    pub tightening_iterations: usize,
}

impl BigMResult {
    fn new_default(constraint_index: usize, m: f64) -> Self {
        Self {
            constraint_index,
            primal_bigm: m,
            dual_bigm: m,
            source: BigMSource::Default,
            is_finite: m.is_finite(),
            tightening_iterations: 0,
        }
    }
}

// ───────────────────────────── BigMSet ────────────────────────────────────

/// Collection of Big-M values for every complementarity pair in a problem.
#[derive(Debug, Clone)]
pub struct BigMSet {
    pub primal_ms: Vec<BigMResult>,
    pub dual_ms: Vec<BigMResult>,
    pub max_primal_m: f64,
    pub max_dual_m: f64,
    pub all_finite: bool,
}

impl BigMSet {
    /// Build from parallel vectors of primal and dual results.
    pub fn from_results(primal_ms: Vec<BigMResult>, dual_ms: Vec<BigMResult>) -> Self {
        let max_primal_m = primal_ms
            .iter()
            .map(|r| r.primal_bigm)
            .fold(0.0_f64, f64::max);
        let max_dual_m = dual_ms.iter().map(|r| r.dual_bigm).fold(0.0_f64, f64::max);
        let all_finite =
            primal_ms.iter().all(|r| r.is_finite) && dual_ms.iter().all(|r| r.is_finite);
        Self {
            primal_ms,
            dual_ms,
            max_primal_m,
            max_dual_m,
            all_finite,
        }
    }

    /// Retrieve the primal-side Big-M for constraint `idx`.
    pub fn get_primal_m(&self, idx: usize) -> Option<&BigMResult> {
        self.primal_ms.get(idx)
    }

    /// Retrieve the dual-side Big-M for constraint `idx`.
    pub fn get_dual_m(&self, idx: usize) -> Option<&BigMResult> {
        self.dual_ms.get(idx)
    }

    /// Overall tightest M (minimum across both sides and all constraints).
    pub fn tightest_m(&self) -> f64 {
        let p = self
            .primal_ms
            .iter()
            .map(|r| r.primal_bigm)
            .fold(f64::INFINITY, f64::min);
        let d = self
            .dual_ms
            .iter()
            .map(|r| r.dual_bigm)
            .fold(f64::INFINITY, f64::min);
        f64::min(p, d)
    }

    /// Number of complementarity pairs.
    pub fn num_pairs(&self) -> usize {
        self.primal_ms.len()
    }

    /// Average primal M.
    pub fn avg_primal_m(&self) -> f64 {
        if self.primal_ms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.primal_ms.iter().map(|r| r.primal_bigm).sum();
        sum / self.primal_ms.len() as f64
    }

    /// Average dual M.
    pub fn avg_dual_m(&self) -> f64 {
        if self.dual_ms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.dual_ms.iter().map(|r| r.dual_bigm).sum();
        sum / self.dual_ms.len() as f64
    }

    /// Fraction of constraints where both M values are finite.
    pub fn finite_fraction(&self) -> f64 {
        let total = self.primal_ms.len().max(1);
        let n_finite = self
            .primal_ms
            .iter()
            .zip(self.dual_ms.iter())
            .filter(|(p, d)| p.is_finite && d.is_finite)
            .count();
        n_finite as f64 / total as f64
    }

    /// Count how many M values came from each source.
    pub fn source_counts(&self) -> [usize; 4] {
        let mut counts = [0usize; 4];
        for r in self.primal_ms.iter().chain(self.dual_ms.iter()) {
            let idx = match r.source {
                BigMSource::UserSpecified => 0,
                BigMSource::IntervalArithmetic => 1,
                BigMSource::BoundTightening => 2,
                BigMSource::Default => 3,
            };
            counts[idx] += 1;
        }
        counts
    }
}

// ──────────────────────── interval-arithmetic helpers ─────────────────────

/// Multiply two intervals `[lo_a, hi_a] × [lo_b, hi_b]`.
pub fn interval_multiply(lo_a: f64, hi_a: f64, lo_b: f64, hi_b: f64) -> (f64, f64) {
    let products = [lo_a * lo_b, lo_a * hi_b, hi_a * lo_b, hi_a * hi_b];
    let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

/// Sum a slice of intervals, returning the enclosing interval.
pub fn interval_sum(intervals: &[(f64, f64)]) -> (f64, f64) {
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    for &(a, b) in intervals {
        lo += a;
        hi += b;
    }
    (lo, hi)
}

/// Negate an interval: −[a, b] = [−b, −a].
fn interval_negate(lo: f64, hi: f64) -> (f64, f64) {
    (-hi, -lo)
}

/// Derive variable-bound intervals from the constraint matrix `A y ≤ b`
/// together with explicit variable bounds.
///
/// Returns a vector of `(lower, upper)` pairs, one per variable.
pub fn compute_variable_bounds_from_constraints(
    a: &SparseMatrix,
    b: &[f64],
    senses: &[ConstraintSense],
    var_bounds: &[VarBound],
) -> Vec<(f64, f64)> {
    let n = var_bounds.len();
    let mut bounds: Vec<(f64, f64)> = var_bounds.iter().map(|vb| (vb.lower, vb.upper)).collect();

    // Single-variable constraints can directly tighten bounds.
    // Group entries by row.
    let mut row_entries: Vec<Vec<&SparseEntry>> = vec![Vec::new(); a.rows];
    for e in &a.entries {
        if e.row < a.rows {
            row_entries[e.row].push(e);
        }
    }

    for (i, entries) in row_entries.iter().enumerate() {
        if entries.len() != 1 {
            continue;
        }
        let e = entries[0];
        let coeff = e.value;
        if coeff.abs() < 1e-15 {
            continue;
        }
        let j = e.col;
        if j >= n {
            continue;
        }
        let rhs = b[i];
        let sense = if i < senses.len() {
            senses[i]
        } else {
            ConstraintSense::Le
        };

        match sense {
            ConstraintSense::Le => {
                // coeff * y_j <= rhs
                if coeff > 0.0 {
                    bounds[j].1 = bounds[j].1.min(rhs / coeff);
                } else {
                    bounds[j].0 = bounds[j].0.max(rhs / coeff);
                }
            }
            ConstraintSense::Ge => {
                // coeff * y_j >= rhs
                if coeff > 0.0 {
                    bounds[j].0 = bounds[j].0.max(rhs / coeff);
                } else {
                    bounds[j].1 = bounds[j].1.min(rhs / coeff);
                }
            }
            ConstraintSense::Eq => {
                let val = rhs / coeff;
                bounds[j].0 = bounds[j].0.max(val);
                bounds[j].1 = bounds[j].1.min(val);
            }
        }
    }

    // Multi-variable rows: try to derive a bound on each variable from the
    // bounds of the other variables in that row.
    for (i, entries) in row_entries.iter().enumerate() {
        if entries.len() < 2 {
            continue;
        }
        let rhs = b[i];
        let sense = if i < senses.len() {
            senses[i]
        } else {
            ConstraintSense::Le
        };

        for &target in entries.iter() {
            let j = target.col;
            let a_j = target.value;
            if a_j.abs() < 1e-15 || j >= n {
                continue;
            }

            // Compute the interval  Σ_{k≠j} a_k · [lb_k, ub_k].
            let mut rest_lo = 0.0_f64;
            let mut rest_hi = 0.0_f64;
            let mut can_compute = true;
            for &other in entries.iter() {
                if other.col == j {
                    continue;
                }
                let k = other.col;
                if k >= n {
                    can_compute = false;
                    break;
                }
                let (lo_k, hi_k) = bounds[k];
                if lo_k == f64::NEG_INFINITY || hi_k == f64::INFINITY {
                    can_compute = false;
                    break;
                }
                let (prod_lo, prod_hi) = interval_multiply(other.value, other.value, lo_k, hi_k);
                rest_lo += prod_lo;
                rest_hi += prod_hi;
            }
            if !can_compute {
                continue;
            }

            // From   a_j y_j + [rest_lo, rest_hi] (sense) rhs
            // isolate y_j.
            match sense {
                ConstraintSense::Le => {
                    // a_j y_j <= rhs - rest_lo  (worst case for rest)
                    let residual = rhs - rest_lo;
                    if a_j > 0.0 {
                        bounds[j].1 = bounds[j].1.min(residual / a_j);
                    } else {
                        bounds[j].0 = bounds[j].0.max(residual / a_j);
                    }
                }
                ConstraintSense::Ge => {
                    let residual = rhs - rest_hi;
                    if a_j > 0.0 {
                        bounds[j].0 = bounds[j].0.max(residual / a_j);
                    } else {
                        bounds[j].1 = bounds[j].1.min(residual / a_j);
                    }
                }
                ConstraintSense::Eq => {
                    // a_j y_j = rhs - [rest_lo, rest_hi]
                    if a_j > 0.0 {
                        bounds[j].1 = bounds[j].1.min((rhs - rest_lo) / a_j);
                        bounds[j].0 = bounds[j].0.max((rhs - rest_hi) / a_j);
                    } else {
                        bounds[j].0 = bounds[j].0.max((rhs - rest_lo) / a_j);
                        bounds[j].1 = bounds[j].1.min((rhs - rest_hi) / a_j);
                    }
                }
            }
        }
    }

    bounds
}

// ──────────────────────── BigMComputer ────────────────────────────────────

/// Main entry-point for computing Big-M values.
pub struct BigMComputer {
    config: BigMConfig,
}

impl BigMComputer {
    pub fn new(config: BigMConfig) -> Self {
        Self { config }
    }

    // ─── public API ──────────────────────────────────────────────────

    /// Compute Big-M values for every complementarity pair in `problem`.
    pub fn compute_all_bigms(&self, problem: &BilevelProblem) -> BigMSet {
        let m = problem.num_lower_constraints;
        let mut primal_ms = Vec::with_capacity(m);
        let mut dual_ms = Vec::with_capacity(m);
        for i in 0..m {
            primal_ms.push(self.compute_primal_bigm(problem, i));
            dual_ms.push(self.compute_dual_bigm(problem, i));
        }
        BigMSet::from_results(primal_ms, dual_ms)
    }

    /// M for the primal slack of lower-level constraint `constraint_idx`:
    ///
    /// ```text
    ///   max  a_i^T y − b_i   s.t. y ∈ Y
    /// ```
    ///
    /// where Y is the lower-level feasible set (ignoring the leader's
    /// decisions for now, i.e. treating x as fixed at its bounds).
    pub fn compute_primal_bigm(
        &self,
        problem: &BilevelProblem,
        constraint_idx: usize,
    ) -> BigMResult {
        if constraint_idx >= problem.num_lower_constraints {
            return BigMResult::new_default(constraint_idx, self.config.default_m);
        }

        // ── 1. try interval arithmetic ──
        if self.config.interval_arithmetic {
            let (ia_lo, ia_hi) = self.interval_arithmetic_bound(problem, constraint_idx);
            let candidate = ia_hi; // max of  a_i^T y − b_i
            if candidate.is_finite() && candidate > 0.0 {
                let m = self.apply_safety_margin(candidate);
                if self.validate_bigm(m) {
                    return BigMResult {
                        constraint_index: constraint_idx,
                        primal_bigm: m,
                        dual_bigm: self.config.default_m,
                        source: BigMSource::IntervalArithmetic,
                        is_finite: true,
                        tightening_iterations: 0,
                    };
                }
            }
        }

        // ── 2. try LP-based bound tightening ──
        if self.config.use_bound_tightening {
            if let Some(m_raw) = self.bound_tightening_lp(problem, constraint_idx, false) {
                let m = self.apply_safety_margin(m_raw.max(0.0));
                if self.validate_bigm(m) {
                    return BigMResult {
                        constraint_index: constraint_idx,
                        primal_bigm: m,
                        dual_bigm: self.config.default_m,
                        source: BigMSource::BoundTightening,
                        is_finite: true,
                        tightening_iterations: 1,
                    };
                }
            }
        }

        // ── 3. fall back to default ──
        BigMResult::new_default(constraint_idx, self.config.default_m)
    }

    /// M for the dual variable of lower-level constraint `constraint_idx`:
    ///
    /// ```text
    ///   max  λ_i   s.t. λ ∈ Λ
    /// ```
    ///
    /// where Λ is the dual feasible set of the lower level.
    pub fn compute_dual_bigm(&self, problem: &BilevelProblem, constraint_idx: usize) -> BigMResult {
        if constraint_idx >= problem.num_lower_constraints {
            return BigMResult::new_default(constraint_idx, self.config.default_m);
        }

        // ── 1. try LP-based tightening first (more precise for duals) ──
        if self.config.use_bound_tightening {
            if let Some(m_raw) = self.bound_tightening_lp(problem, constraint_idx, true) {
                let m = self.apply_safety_margin(m_raw.max(0.0));
                if self.validate_bigm(m) {
                    return BigMResult {
                        constraint_index: constraint_idx,
                        primal_bigm: self.config.default_m,
                        dual_bigm: m,
                        source: BigMSource::BoundTightening,
                        is_finite: true,
                        tightening_iterations: 1,
                    };
                }
            }
        }

        // ── 2. try interval arithmetic on the dual polyhedron ──
        if self.config.interval_arithmetic {
            let dual_bound = self.dual_interval_bound(problem, constraint_idx);
            if dual_bound.is_finite() && dual_bound > 0.0 {
                let m = self.apply_safety_margin(dual_bound);
                if self.validate_bigm(m) {
                    return BigMResult {
                        constraint_index: constraint_idx,
                        primal_bigm: self.config.default_m,
                        dual_bigm: m,
                        source: BigMSource::IntervalArithmetic,
                        is_finite: true,
                        tightening_iterations: 0,
                    };
                }
            }
        }

        BigMResult::new_default(constraint_idx, self.config.default_m)
    }

    /// Quick interval-arithmetic bound on `a_i^T y − b_i` over the
    /// lower-level feasible region.
    ///
    /// Returns `(lower_bound, upper_bound)` of that expression.
    pub fn interval_arithmetic_bound(
        &self,
        problem: &BilevelProblem,
        constraint_idx: usize,
    ) -> (f64, f64) {
        let n = problem.num_lower_vars;
        let a = &problem.lower_a;
        let b = &problem.lower_b;

        // Gather variable bounds, tightened by the constraint system.
        let base_bounds: Vec<VarBound> = (0..n)
            .map(|_| VarBound {
                lower: 0.0,
                upper: f64::INFINITY,
            })
            .collect();
        let senses: Vec<ConstraintSense> = vec![ConstraintSense::Le; problem.num_lower_constraints];
        let var_bounds = compute_variable_bounds_from_constraints(a, b, &senses, &base_bounds);

        // Collect entries of row `constraint_idx`.
        let row_entries: Vec<&SparseEntry> = a
            .entries
            .iter()
            .filter(|e| e.row == constraint_idx)
            .collect();

        let mut intervals: Vec<(f64, f64)> = Vec::with_capacity(row_entries.len());
        for e in &row_entries {
            let j = e.col;
            if j >= n {
                continue;
            }
            let (lb, ub) = var_bounds[j];
            // Skip if variable is unbounded – can't do interval arithmetic.
            if lb == f64::NEG_INFINITY || ub == f64::INFINITY {
                return (f64::NEG_INFINITY, f64::INFINITY);
            }
            intervals.push(interval_multiply(e.value, e.value, lb, ub));
        }

        let (sum_lo, sum_hi) = interval_sum(&intervals);
        let rhs = if constraint_idx < b.len() {
            b[constraint_idx]
        } else {
            0.0
        };
        // a_i^T y − b_i  ∈  [sum_lo − b_i, sum_hi − b_i]
        (sum_lo - rhs, sum_hi - rhs)
    }

    /// Solve an LP to obtain a tight bound on either the primal slack or
    /// the dual variable for the given constraint.
    ///
    /// When `is_dual` is `false` we maximise `a_i^T y − b_i` over the
    /// primal feasible set.  When `is_dual` is `true` we maximise `λ_i`
    /// over the dual feasible set.
    pub fn bound_tightening_lp(
        &self,
        problem: &BilevelProblem,
        constraint_idx: usize,
        is_dual: bool,
    ) -> Option<f64> {
        if is_dual {
            self.dual_bound_lp(problem, constraint_idx)
        } else {
            self.primal_bound_lp(problem, constraint_idx)
        }
    }

    /// Check that `m_value` is a usable Big-M constant.
    pub fn validate_bigm(&self, m_value: f64) -> bool {
        if !m_value.is_finite() {
            return false;
        }
        if m_value < -self.config.tol {
            return false;
        }
        // Reject absurdly large values that would destroy LP numerics.
        if m_value > 1e12 {
            return false;
        }
        true
    }

    /// Multiply the raw bound by the safety margin factor.
    pub fn apply_safety_margin(&self, m_value: f64) -> f64 {
        let factor = if self.config.safety_margin_factor >= 1.0 {
            self.config.safety_margin_factor
        } else {
            1.0
        };
        let result = m_value * factor;
        // Ensure we never return something smaller than the tolerance.
        if result < self.config.tol {
            self.config.tol
        } else {
            result
        }
    }

    // ─── private helpers ─────────────────────────────────────────────

    /// Maximise `a_i^T y − b_i` over `{ y : A y ≤ b, y ≥ 0 }`.
    fn primal_bound_lp(&self, problem: &BilevelProblem, constraint_idx: usize) -> Option<f64> {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        if n == 0 || m == 0 {
            return None;
        }

        // Objective: maximise a_i^T y.
        // We build a maximisation LP with the lower-level constraints.
        let mut obj = vec![0.0; n];
        for e in &problem.lower_a.entries {
            if e.row == constraint_idx && e.col < n {
                obj[e.col] = e.value;
            }
        }

        let senses = vec![ConstraintSense::Le; m];
        let var_bounds: Vec<VarBound> = (0..n)
            .map(|_| VarBound {
                lower: 0.0,
                upper: 1e8,
            })
            .collect();

        let lp = LpProblem {
            direction: OptDirection::Maximize,
            c: obj,
            a_matrix: problem.lower_a.clone(),
            b_rhs: problem.lower_b.clone(),
            senses,
            var_bounds,
            num_vars: n,
            num_constraints: m,
        };

        let solver = SimplexSolver::new();
        match solver.solve(&lp) {
            Ok(sol) if sol.status == LpStatus::Optimal => {
                let rhs = if constraint_idx < problem.lower_b.len() {
                    problem.lower_b[constraint_idx]
                } else {
                    0.0
                };
                Some(sol.objective - rhs)
            }
            Ok(sol) if sol.status == LpStatus::Unbounded => None,
            _ => None,
        }
    }

    /// Maximise `λ_i` over the dual feasible set
    /// `{ λ ≥ 0 : A^T λ ≥ c }` (dual of the lower-level LP).
    fn dual_bound_lp(&self, problem: &BilevelProblem, constraint_idx: usize) -> Option<f64> {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        if n == 0 || m == 0 {
            return None;
        }

        // Build A^T as a SparseMatrix (transpose of lower_a).
        let mut at_entries: Vec<SparseEntry> = Vec::with_capacity(problem.lower_a.entries.len());
        for e in &problem.lower_a.entries {
            at_entries.push(SparseEntry {
                row: e.col,
                col: e.row,
                value: e.value,
            });
        }
        let a_transpose = SparseMatrix {
            rows: n,
            cols: m,
            entries: at_entries,
        };

        // Dual constraints: A^T λ ≥ c  ↔  −A^T λ ≤ −c
        let mut neg_at_entries: Vec<SparseEntry> = Vec::with_capacity(a_transpose.entries.len());
        for e in &a_transpose.entries {
            neg_at_entries.push(SparseEntry {
                row: e.row,
                col: e.col,
                value: -e.value,
            });
        }
        let neg_a_transpose = SparseMatrix {
            rows: n,
            cols: m,
            entries: neg_at_entries,
        };

        let neg_c: Vec<f64> = problem.lower_obj_c.iter().map(|&v| -v).collect();
        let senses = vec![ConstraintSense::Le; n];
        let var_bounds: Vec<VarBound> = (0..m)
            .map(|_| VarBound {
                lower: 0.0,
                upper: 1e8,
            })
            .collect();

        // Objective: maximise λ_{constraint_idx}, i.e. e_i^T λ.
        let mut obj = vec![0.0; m];
        if constraint_idx < m {
            obj[constraint_idx] = 1.0;
        }

        let lp = LpProblem {
            direction: OptDirection::Maximize,
            c: obj,
            a_matrix: neg_a_transpose,
            b_rhs: neg_c,
            senses,
            var_bounds,
            num_vars: m,
            num_constraints: n,
        };

        let solver = SimplexSolver::new();
        match solver.solve(&lp) {
            Ok(sol) if sol.status == LpStatus::Optimal => Some(sol.objective),
            Ok(sol) if sol.status == LpStatus::Unbounded => None,
            _ => None,
        }
    }

    /// Interval-arithmetic bound on a single dual variable.
    ///
    /// Uses the dual feasibility condition  A^T λ ≥ c  together with λ ≥ 0
    /// to derive an upper bound on λ_i.
    fn dual_interval_bound(&self, problem: &BilevelProblem, constraint_idx: usize) -> f64 {
        let n = problem.num_lower_vars;
        let m = problem.num_lower_constraints;
        if n == 0 || m == 0 {
            return f64::INFINITY;
        }

        // For each primal variable j, the dual constraint is:
        //   Σ_i  A_{j,i} λ_i  ≥  c_j
        //
        // If A_{j, constraint_idx} > 0, then
        //   λ_{constraint_idx} ≤ (c_j - Σ_{k≠constraint_idx} A_{j,k} λ_k_min) / A_{j, constraint_idx}
        // and since λ_k ≥ 0, the minimum of each λ_k is 0 when A_{j,k} > 0.
        //
        // We keep the tightest such upper bound across all j.

        // Build a column-view of A for `constraint_idx`.
        let mut col_entries: Vec<(usize, f64)> = Vec::new();
        for e in &problem.lower_a.entries {
            if e.col == constraint_idx {
                col_entries.push((e.row, e.value));
            }
        }

        // Also build per-row lookup.
        let mut row_map: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for e in &problem.lower_a.entries {
            if e.row < n {
                // A^T: row j, col i <=> A: row i, col j
                // We need the j-th dual constraint, which uses column j of A^T
                // which is row j of A... but lower_a is m×n.
                // Actually lower_a has rows = constraints, cols = variables.
                // A^T has rows = variables, cols = constraints.
                // Dual constraint j: Σ_i A_{i,j} λ_i >= c_j
                // So for variable j, gather entries where col == j.
            }
        }

        // Rebuild: for dual constraint j, we need all (i, A_{i,j}) with A_{i,j} != 0.
        let mut dual_constraints: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for e in &problem.lower_a.entries {
            if e.col < n && e.row < m {
                dual_constraints[e.col].push((e.row, e.value));
            }
        }

        let mut best_ub = f64::INFINITY;

        for j in 0..n {
            // Find the coefficient of λ_{constraint_idx} in dual constraint j.
            let a_j_target = dual_constraints[j]
                .iter()
                .find(|(i, _)| *i == constraint_idx)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);

            if a_j_target <= self.config.tol {
                // This constraint does not bound λ_{constraint_idx} from above.
                continue;
            }

            let c_j = if j < problem.lower_obj_c.len() {
                problem.lower_obj_c[j]
            } else {
                0.0
            };

            // The dual constraint is:
            //   a_j_target · λ_{constraint_idx} + Σ_{k≠constraint_idx} A_{j,k} λ_k ≥ c_j
            //
            // Since λ_k ≥ 0, for an upper bound on λ_{constraint_idx} we set
            // the positive-coefficient λ_k to 0 and cannot bound the
            // negative-coefficient terms.  We get:
            //   λ_{constraint_idx} ≤ c_j / a_j_target   (when all other terms are 0)
            //
            // But if c_j < 0 this gives a negative bound which is redundant
            // with λ ≥ 0, so we ignore it.
            let mut rest_min = 0.0_f64;
            let mut feasible = true;
            for &(k, a_jk) in &dual_constraints[j] {
                if k == constraint_idx {
                    continue;
                }
                // λ_k ≥ 0.  When a_jk > 0, minimum contribution is 0.
                // When a_jk < 0, contribution is unbounded below (λ_k → ∞).
                if a_jk < -self.config.tol {
                    // Negative coefficient with λ_k ≥ 0 means the sum can
                    // go to −∞, so this row doesn't bound our target.
                    feasible = false;
                    break;
                }
                // a_jk >= 0: minimum contribution when λ_k = 0 → 0.
                // (rest_min stays 0)
            }
            if !feasible {
                continue;
            }

            let ub = (c_j - rest_min) / a_j_target;
            if ub >= 0.0 && ub < best_ub {
                best_ub = ub;
            }
        }

        best_ub
    }
}

// ──────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──

    /// Build a tiny lower-level LP:
    ///   min  c^T y
    ///   s.t. A y ≤ b, y ≥ 0
    fn simple_bilevel(
        a_dense: &[Vec<f64>],
        b: &[f64],
        c: &[f64],
        num_upper: usize,
    ) -> BilevelProblem {
        let m = a_dense.len();
        let n = if m > 0 { a_dense[0].len() } else { 0 };

        let mut entries = Vec::new();
        for (i, row) in a_dense.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v.abs() > 1e-15 {
                    entries.push(SparseEntry {
                        row: i,
                        col: j,
                        value: v,
                    });
                }
            }
        }
        let lower_a = SparseMatrix {
            rows: m,
            cols: n,
            entries,
        };

        BilevelProblem {
            upper_obj_c_x: vec![0.0; num_upper],
            upper_obj_c_y: vec![0.0; n],
            lower_obj_c: c.to_vec(),
            lower_a,
            lower_b: b.to_vec(),
            lower_linking_b: SparseMatrix {
                rows: m,
                cols: num_upper,
                entries: vec![],
            },
            upper_constraints_a: SparseMatrix {
                rows: 0,
                cols: num_upper + n,
                entries: vec![],
            },
            upper_constraints_b: vec![],
            num_upper_vars: num_upper,
            num_lower_vars: n,
            num_lower_constraints: m,
            num_upper_constraints: 0,
        }
    }

    // ── interval arithmetic ──

    #[test]
    fn test_interval_multiply_positive() {
        let (lo, hi) = interval_multiply(1.0, 3.0, 2.0, 4.0);
        assert!((lo - 2.0).abs() < 1e-12);
        assert!((hi - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_multiply_mixed_sign() {
        let (lo, hi) = interval_multiply(-2.0, 3.0, -1.0, 4.0);
        // products: 2, -8, -3, 12
        assert!((lo - (-8.0)).abs() < 1e-12);
        assert!((hi - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sum_basic() {
        let intervals = vec![(1.0, 2.0), (3.0, 5.0), (-1.0, 0.0)];
        let (lo, hi) = interval_sum(&intervals);
        assert!((lo - 3.0).abs() < 1e-12);
        assert!((hi - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_sum_empty() {
        let (lo, hi) = interval_sum(&[]);
        assert!((lo - 0.0).abs() < 1e-12);
        assert!((hi - 0.0).abs() < 1e-12);
    }

    // ── safety margin ──

    #[test]
    fn test_safety_margin_application() {
        let comp = BigMComputer::new(BigMConfig {
            safety_margin_factor: 1.5,
            ..Default::default()
        });
        let m = comp.apply_safety_margin(10.0);
        assert!((m - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_safety_margin_clamp_at_one() {
        let comp = BigMComputer::new(BigMConfig {
            safety_margin_factor: 0.5, // invalid, should clamp to 1.0
            ..Default::default()
        });
        let m = comp.apply_safety_margin(10.0);
        assert!((m - 10.0).abs() < 1e-12);
    }

    // ── validation ──

    #[test]
    fn test_validate_bigm_rejects_infinity() {
        let comp = BigMComputer::new(BigMConfig::default());
        assert!(!comp.validate_bigm(f64::INFINITY));
        assert!(!comp.validate_bigm(f64::NAN));
    }

    #[test]
    fn test_validate_bigm_rejects_negative() {
        let comp = BigMComputer::new(BigMConfig::default());
        assert!(!comp.validate_bigm(-1.0));
    }

    #[test]
    fn test_validate_bigm_accepts_reasonable() {
        let comp = BigMComputer::new(BigMConfig::default());
        assert!(comp.validate_bigm(100.0));
        assert!(comp.validate_bigm(0.0));
        assert!(comp.validate_bigm(1e6));
    }

    // ── BigMSet operations ──

    #[test]
    fn test_bigm_set_from_results() {
        let primal = vec![
            BigMResult {
                constraint_index: 0,
                primal_bigm: 10.0,
                dual_bigm: 100.0,
                source: BigMSource::IntervalArithmetic,
                is_finite: true,
                tightening_iterations: 0,
            },
            BigMResult {
                constraint_index: 1,
                primal_bigm: 20.0,
                dual_bigm: 100.0,
                source: BigMSource::BoundTightening,
                is_finite: true,
                tightening_iterations: 1,
            },
        ];
        let dual = vec![
            BigMResult {
                constraint_index: 0,
                primal_bigm: 100.0,
                dual_bigm: 5.0,
                source: BigMSource::BoundTightening,
                is_finite: true,
                tightening_iterations: 1,
            },
            BigMResult {
                constraint_index: 1,
                primal_bigm: 100.0,
                dual_bigm: 15.0,
                source: BigMSource::Default,
                is_finite: true,
                tightening_iterations: 0,
            },
        ];

        let set = BigMSet::from_results(primal, dual);
        assert_eq!(set.num_pairs(), 2);
        assert!((set.max_primal_m - 20.0).abs() < 1e-12);
        assert!((set.max_dual_m - 15.0).abs() < 1e-12);
        assert!(set.all_finite);
        assert!((set.tightest_m() - 5.0).abs() < 1e-12);
        assert!((set.avg_primal_m() - 15.0).abs() < 1e-12);
        assert!((set.avg_dual_m() - 10.0).abs() < 1e-12);
        assert!((set.finite_fraction() - 1.0).abs() < 1e-12);

        let counts = set.source_counts();
        // primal: 1 IA + 1 BT; dual: 1 BT + 1 Default
        assert_eq!(counts[0], 0); // UserSpecified
        assert_eq!(counts[1], 1); // IntervalArithmetic
        assert_eq!(counts[2], 2); // BoundTightening
        assert_eq!(counts[3], 1); // Default

        assert!(set.get_primal_m(0).is_some());
        assert!(set.get_primal_m(2).is_none());
        assert!(set.get_dual_m(1).is_some());
    }

    // ── interval-arithmetic bound on a simple problem ──

    #[test]
    fn test_interval_arithmetic_bound_simple() {
        // Lower level: y1, y2 ≥ 0
        //   y1 + y2 ≤ 10
        //   y1      ≤  6
        //        y2 ≤  8
        // min y1 + y2
        let problem = simple_bilevel(
            &[vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            &[10.0, 6.0, 8.0],
            &[1.0, 1.0],
            1,
        );

        let comp = BigMComputer::new(BigMConfig::default());
        // For constraint 0 (y1 + y2 ≤ 10):
        //   a_0^T y - b_0 = y1 + y2 - 10
        //   y1 ∈ [0,6], y2 ∈ [0,8]  →  y1+y2 ∈ [0,14]  →  expr ∈ [-10, 4]
        let (lo, hi) = comp.interval_arithmetic_bound(&problem, 0);
        assert!((lo - (-10.0)).abs() < 1e-8);
        assert!((hi - 4.0).abs() < 1e-8);
    }

    // ── compute_all_bigms smoke test ──

    #[test]
    fn test_compute_all_bigms_smoke() {
        let problem = simple_bilevel(
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[5.0, 5.0],
            &[1.0, 1.0],
            1,
        );

        let comp = BigMComputer::new(BigMConfig {
            use_bound_tightening: false,
            ..Default::default()
        });
        let set = comp.compute_all_bigms(&problem);
        assert_eq!(set.num_pairs(), 2);
        // With only interval arithmetic on simple bound constraints both
        // should produce finite M values.
        for r in &set.primal_ms {
            assert!(r.primal_bigm > 0.0);
        }
    }

    // ── edge case: zero-coefficient row ──

    #[test]
    fn test_zero_coefficient_row() {
        // Constraint row of all zeros → expression is constant 0 − b.
        let mut problem = simple_bilevel(
            &[vec![0.0, 0.0], vec![1.0, 0.0]],
            &[0.0, 5.0],
            &[1.0, 1.0],
            0,
        );
        // Manually ensure the sparse matrix has no entries for row 0.
        problem.lower_a.entries.retain(|e| e.row != 0);

        let comp = BigMComputer::new(BigMConfig::default());
        let (lo, hi) = comp.interval_arithmetic_bound(&problem, 0);
        // 0 - 0 = 0 on both ends
        assert!((lo - 0.0).abs() < 1e-12);
        assert!((hi - 0.0).abs() < 1e-12);
    }

    // ── variable bounds from constraints ──

    #[test]
    fn test_compute_variable_bounds_from_constraints() {
        // y1 ≤ 4, y2 ≤ 7
        let a = SparseMatrix {
            rows: 2,
            cols: 2,
            entries: vec![
                SparseEntry {
                    row: 0,
                    col: 0,
                    value: 1.0,
                },
                SparseEntry {
                    row: 1,
                    col: 1,
                    value: 1.0,
                },
            ],
        };
        let b = vec![4.0, 7.0];
        let senses = vec![ConstraintSense::Le, ConstraintSense::Le];
        let vb = vec![
            VarBound {
                lower: 0.0,
                upper: f64::INFINITY,
            },
            VarBound {
                lower: 0.0,
                upper: f64::INFINITY,
            },
        ];

        let bounds = compute_variable_bounds_from_constraints(&a, &b, &senses, &vb);
        assert!((bounds[0].0 - 0.0).abs() < 1e-12);
        assert!((bounds[0].1 - 4.0).abs() < 1e-12);
        assert!((bounds[1].0 - 0.0).abs() < 1e-12);
        assert!((bounds[1].1 - 7.0).abs() < 1e-12);
    }
}

/// Type alias for backward compatibility with re-exports.
pub type BigMEstimate = BigMResult;
