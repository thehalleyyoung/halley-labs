//! Value function oracle trait and implementations.
//!
//! Provides the [`ValueFunctionOracle`] trait for evaluating φ(x) = min{c^T y : Ay ≤ b + Bx, y ≥ 0}
//! along with concrete implementations: [`ExactLpOracle`] (solves an LP per query)
//! and [`CachedOracle`] (wraps any oracle with an LRU cache).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{
    AffineFunction, BasisStatus, BilevelProblem, LpProblem, LpSolution, LpStatus, OrdF64,
    Polyhedron, SparseMatrix,
};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Oracle trait
// ---------------------------------------------------------------------------

/// Information returned alongside a value-function evaluation.
#[derive(Debug, Clone)]
pub struct ValueFunctionInfo {
    /// The optimal objective value φ(x).
    pub value: f64,
    /// Optimal primal solution y*.
    pub primal_solution: Vec<f64>,
    /// Optimal dual solution π*.
    pub dual_solution: Vec<f64>,
    /// Basis status of each variable at optimality.
    pub basis: Vec<BasisStatus>,
    /// Number of simplex iterations used.
    pub iterations: u64,
}

/// Dual information at a given evaluation point.
#[derive(Debug, Clone)]
pub struct DualInfo {
    /// Dual multipliers π for the constraints Ay ≤ b + Bx.
    pub multipliers: Vec<f64>,
    /// Subgradient of φ at x: g = -B^T π  (when φ is differentiable).
    pub subgradient: Vec<f64>,
    /// Whether the dual solution is degenerate.
    pub is_degenerate: bool,
}

/// Feasibility information for the lower level.
#[derive(Debug, Clone)]
pub struct FeasibilityInfo {
    pub is_feasible: bool,
    pub is_bounded: bool,
    /// If infeasible, a Farkas certificate (dual ray).
    pub farkas_certificate: Option<Vec<f64>>,
}

/// The core oracle trait for evaluating the lower-level value function.
pub trait ValueFunctionOracle: Send + Sync {
    /// Evaluate φ(x) and return full solution information.
    fn evaluate(&self, x: &[f64]) -> VFResult<ValueFunctionInfo>;

    /// Get dual information at x.
    fn dual_info(&self, x: &[f64]) -> VFResult<DualInfo>;

    /// Check feasibility of the lower-level problem for given x.
    fn check_feasibility(&self, x: &[f64]) -> VFResult<FeasibilityInfo>;

    /// Evaluate just the optimal value φ(x).
    fn value(&self, x: &[f64]) -> VFResult<f64> {
        self.evaluate(x).map(|info| info.value)
    }

    /// Evaluate the subgradient of φ at x.
    fn subgradient(&self, x: &[f64]) -> VFResult<Vec<f64>> {
        self.dual_info(x).map(|d| d.subgradient)
    }

    /// Get statistics about oracle usage.
    fn statistics(&self) -> OracleStatistics;

    /// Reset oracle statistics.
    fn reset_statistics(&self);
}

// ---------------------------------------------------------------------------
// Oracle statistics
// ---------------------------------------------------------------------------

/// Statistics about oracle usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OracleStatistics {
    pub total_evaluations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_lp_solves: u64,
    pub total_iterations: u64,
    pub feasible_count: u64,
    pub infeasible_count: u64,
    pub avg_iterations: f64,
}

/// Thread-safe statistics tracker.
pub struct OracleStatsTracker {
    total_evaluations: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_lp_solves: AtomicU64,
    total_iterations: AtomicU64,
    feasible_count: AtomicU64,
    infeasible_count: AtomicU64,
}

impl OracleStatsTracker {
    pub fn new() -> Self {
        Self {
            total_evaluations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_lp_solves: AtomicU64::new(0),
            total_iterations: AtomicU64::new(0),
            feasible_count: AtomicU64::new(0),
            infeasible_count: AtomicU64::new(0),
        }
    }

    pub fn record_evaluation(&self) {
        self.total_evaluations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_lp_solve(&self, iterations: u64) {
        self.total_lp_solves.fetch_add(1, Ordering::Relaxed);
        self.total_iterations
            .fetch_add(iterations, Ordering::Relaxed);
    }

    pub fn record_feasible(&self) {
        self.feasible_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_infeasible(&self) {
        self.infeasible_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> OracleStatistics {
        let total_lp = self.total_lp_solves.load(Ordering::Relaxed);
        let total_iter = self.total_iterations.load(Ordering::Relaxed);
        let avg = if total_lp > 0 {
            total_iter as f64 / total_lp as f64
        } else {
            0.0
        };
        OracleStatistics {
            total_evaluations: self.total_evaluations.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            total_lp_solves: total_lp,
            total_iterations: total_iter,
            feasible_count: self.feasible_count.load(Ordering::Relaxed),
            infeasible_count: self.infeasible_count.load(Ordering::Relaxed),
            avg_iterations: avg,
        }
    }

    pub fn reset(&self) {
        self.total_evaluations.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.total_lp_solves.store(0, Ordering::Relaxed);
        self.total_iterations.store(0, Ordering::Relaxed);
        self.feasible_count.store(0, Ordering::Relaxed);
        self.infeasible_count.store(0, Ordering::Relaxed);
    }
}

impl Default for OracleStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Exact LP oracle
// ---------------------------------------------------------------------------

/// Oracle that solves a fresh LP for each evaluation of φ(x).
pub struct ExactLpOracle {
    /// The bilevel problem specification.
    problem: BilevelProblem,
    /// The LP solver to use.
    solver: Arc<dyn LpSolver>,
    /// Statistics tracker.
    stats: OracleStatsTracker,
}

impl ExactLpOracle {
    pub fn new(problem: BilevelProblem, solver: Arc<dyn LpSolver>) -> Self {
        Self {
            problem,
            solver,
            stats: OracleStatsTracker::new(),
        }
    }

    pub fn with_default_solver(problem: BilevelProblem) -> Self {
        Self::new(problem, Arc::new(SimplexSolver::default()))
    }

    /// Build the lower-level LP for a given x and solve it.
    fn solve_lower_level(&self, x: &[f64]) -> VFResult<LpSolution> {
        if x.len() != self.problem.num_upper_vars {
            return Err(VFError::DimensionMismatch {
                expected: self.problem.num_upper_vars,
                got: x.len(),
            });
        }

        let lp = self.problem.lower_level_lp(x);
        match self.solver.solve(&lp) {
            Ok(sol) => {
                self.stats.record_lp_solve(sol.iterations);
                Ok(sol)
            }
            Err(e) => Err(VFError::LpError(format!("{}", e))),
        }
    }

    /// Compute the subgradient g = Bᵀ sp from LP shadow prices.
    ///
    /// For φ(x) = min{ cᵀy : Ay ≤ b + Bx, y ≥ 0 }, the subgradient is
    /// g = Bᵀ sp where sp_i = ∂φ/∂b_i (shadow price convention, ≤ 0 for
    /// binding ≤ constraints in minimisation).
    fn compute_subgradient(&self, dual: &[f64]) -> Vec<f64> {
        let nx = self.problem.num_upper_vars;
        let mut grad = vec![0.0; nx];
        for entry in &self.problem.lower_linking_b.entries {
            if entry.row < dual.len() && entry.col < nx {
                grad[entry.col] += entry.value * dual[entry.row];
            }
        }
        grad
    }

    /// Determine if the dual solution is degenerate.
    fn check_degeneracy(&self, sol: &LpSolution) -> bool {
        for (i, &bs) in sol.basis.iter().enumerate() {
            if bs == BasisStatus::Basic && i < sol.primal.len() {
                if sol.primal[i].abs() < TOLERANCE {
                    return true;
                }
            }
        }
        false
    }

    /// Access the problem specification.
    pub fn problem(&self) -> &BilevelProblem {
        &self.problem
    }
}

impl ValueFunctionOracle for ExactLpOracle {
    fn evaluate(&self, x: &[f64]) -> VFResult<ValueFunctionInfo> {
        self.stats.record_evaluation();
        let sol = self.solve_lower_level(x)?;

        match sol.status {
            LpStatus::Optimal => {
                self.stats.record_feasible();
                Ok(ValueFunctionInfo {
                    value: sol.objective,
                    primal_solution: sol.primal,
                    dual_solution: sol.dual,
                    basis: sol.basis,
                    iterations: sol.iterations,
                })
            }
            LpStatus::Infeasible => {
                self.stats.record_infeasible();
                Err(VFError::InfeasibleLowerLevel)
            }
            LpStatus::Unbounded => Err(VFError::UnboundedLowerLevel),
            _ => Err(VFError::LpError(format!(
                "Unexpected LP status: {}",
                sol.status
            ))),
        }
    }

    fn dual_info(&self, x: &[f64]) -> VFResult<DualInfo> {
        self.stats.record_evaluation();
        let sol = self.solve_lower_level(x)?;

        match sol.status {
            LpStatus::Optimal => {
                let subgradient = self.compute_subgradient(&sol.dual);
                let is_degenerate = self.check_degeneracy(&sol);
                self.stats.record_feasible();
                Ok(DualInfo {
                    multipliers: sol.dual,
                    subgradient,
                    is_degenerate,
                })
            }
            LpStatus::Infeasible => {
                self.stats.record_infeasible();
                Err(VFError::InfeasibleLowerLevel)
            }
            _ => Err(VFError::LpError(format!("LP status: {}", sol.status))),
        }
    }

    fn check_feasibility(&self, x: &[f64]) -> VFResult<FeasibilityInfo> {
        self.stats.record_evaluation();
        let sol = self.solve_lower_level(x)?;
        match sol.status {
            LpStatus::Optimal => {
                self.stats.record_feasible();
                Ok(FeasibilityInfo {
                    is_feasible: true,
                    is_bounded: true,
                    farkas_certificate: None,
                })
            }
            LpStatus::Infeasible => {
                self.stats.record_infeasible();
                let farkas = if sol.dual.is_empty() {
                    None
                } else {
                    Some(sol.dual.clone())
                };
                Ok(FeasibilityInfo {
                    is_feasible: false,
                    is_bounded: true,
                    farkas_certificate: farkas,
                })
            }
            LpStatus::Unbounded => Ok(FeasibilityInfo {
                is_feasible: true,
                is_bounded: false,
                farkas_certificate: None,
            }),
            _ => Err(VFError::LpError(format!("LP status: {}", sol.status))),
        }
    }

    fn statistics(&self) -> OracleStatistics {
        self.stats.snapshot()
    }

    fn reset_statistics(&self) {
        self.stats.reset();
    }
}

// ---------------------------------------------------------------------------
// LRU cache internals
// ---------------------------------------------------------------------------

fn discretize_key(x: &[f64], precision: u32) -> Vec<i64> {
    let scale = 10f64.powi(precision as i32);
    x.iter().map(|&v| (v * scale).round() as i64).collect()
}

#[derive(Debug, Clone)]
struct CacheEntry {
    info: ValueFunctionInfo,
    dual: DualInfo,
    access_count: u64,
    last_access: u64,
}

struct LruCache {
    entries: HashMap<Vec<i64>, CacheEntry>,
    capacity: usize,
    access_counter: u64,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            access_counter: 0,
        }
    }

    fn get(&mut self, key: &[i64]) -> Option<&CacheEntry> {
        self.access_counter += 1;
        let counter = self.access_counter;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access = counter;
            Some(&*entry)
        } else {
            None
        }
    }

    fn insert(&mut self, key: Vec<i64>, entry: CacheEntry) {
        if self.entries.len() >= self.capacity {
            self.evict();
        }
        self.entries.insert(key, entry);
    }

    fn evict(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, _)| k.clone());
        if let Some(key) = lru_key {
            self.entries.remove(&key);
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
    }
}

// ---------------------------------------------------------------------------
// Cached oracle
// ---------------------------------------------------------------------------

/// Oracle wrapper that caches results with an LRU eviction policy.
pub struct CachedOracle {
    inner: Arc<dyn ValueFunctionOracle>,
    cache: Mutex<LruCache>,
    precision: u32,
    stats: OracleStatsTracker,
}

impl CachedOracle {
    pub fn new(inner: Arc<dyn ValueFunctionOracle>, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(capacity)),
            precision: 8,
            stats: OracleStatsTracker::new(),
        }
    }

    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision;
        self
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    fn cache_key(&self, x: &[f64]) -> Vec<i64> {
        discretize_key(x, self.precision)
    }

    fn lookup_or_compute(&self, x: &[f64]) -> VFResult<(ValueFunctionInfo, DualInfo)> {
        let key = self.cache_key(x);

        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(entry) = cache.get(&key) {
                self.stats.record_cache_hit();
                return Ok((entry.info.clone(), entry.dual.clone()));
            }
        }

        self.stats.record_cache_miss();

        let info = self.inner.evaluate(x)?;
        let dual = self.inner.dual_info(x)?;

        {
            let entry = CacheEntry {
                info: info.clone(),
                dual: dual.clone(),
                access_count: 1,
                last_access: 0,
            };
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, entry);
        }

        Ok((info, dual))
    }
}

impl ValueFunctionOracle for CachedOracle {
    fn evaluate(&self, x: &[f64]) -> VFResult<ValueFunctionInfo> {
        self.stats.record_evaluation();
        let (info, _dual) = self.lookup_or_compute(x)?;
        Ok(info)
    }

    fn dual_info(&self, x: &[f64]) -> VFResult<DualInfo> {
        self.stats.record_evaluation();
        let (_info, dual) = self.lookup_or_compute(x)?;
        Ok(dual)
    }

    fn check_feasibility(&self, x: &[f64]) -> VFResult<FeasibilityInfo> {
        self.stats.record_evaluation();
        self.inner.check_feasibility(x)
    }

    fn statistics(&self) -> OracleStatistics {
        let mut stats = self.stats.snapshot();
        let inner_stats = self.inner.statistics();
        stats.total_lp_solves = inner_stats.total_lp_solves;
        stats.total_iterations = inner_stats.total_iterations;
        stats.avg_iterations = inner_stats.avg_iterations;
        stats
    }

    fn reset_statistics(&self) {
        self.stats.reset();
        self.inner.reset_statistics();
        self.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Batch evaluation utilities
// ---------------------------------------------------------------------------

/// Evaluate the oracle at multiple points.
pub fn batch_evaluate(
    oracle: &dyn ValueFunctionOracle,
    points: &[Vec<f64>],
) -> Vec<VFResult<ValueFunctionInfo>> {
    points.iter().map(|x| oracle.evaluate(x)).collect()
}

/// Evaluate the oracle at multiple points, collecting only the values.
pub fn batch_values(oracle: &dyn ValueFunctionOracle, points: &[Vec<f64>]) -> Vec<VFResult<f64>> {
    points.iter().map(|x| oracle.value(x)).collect()
}

/// Finite-difference gradient approximation of φ at x.
pub fn finite_difference_gradient(
    oracle: &dyn ValueFunctionOracle,
    x: &[f64],
    step_size: f64,
) -> VFResult<Vec<f64>> {
    let n = x.len();
    let mut grad = vec![0.0; n];

    for j in 0..n {
        let mut x_plus = x.to_vec();
        x_plus[j] += step_size;
        let val_plus = oracle.value(&x_plus)?;

        let mut x_minus = x.to_vec();
        x_minus[j] -= step_size;
        let val_minus = oracle.value(&x_minus)?;

        grad[j] = (val_plus - val_minus) / (2.0 * step_size);
    }

    Ok(grad)
}

/// Compute a numeric Hessian approximation of φ at x.
pub fn finite_difference_hessian(
    oracle: &dyn ValueFunctionOracle,
    x: &[f64],
    step_size: f64,
) -> VFResult<Vec<Vec<f64>>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec();
            xpp[i] += step_size;
            xpp[j] += step_size;

            let mut xpm = x.to_vec();
            xpm[i] += step_size;
            xpm[j] -= step_size;

            let mut xmp = x.to_vec();
            xmp[i] -= step_size;
            xmp[j] += step_size;

            let mut xmm = x.to_vec();
            xmm[i] -= step_size;
            xmm[j] -= step_size;

            let fpp = oracle.value(&xpp)?;
            let fpm = oracle.value(&xpm)?;
            let fmp = oracle.value(&xmp)?;
            let fmm = oracle.value(&xmm)?;

            let val = (fpp - fpm - fmp + fmm) / (4.0 * step_size * step_size);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }

    Ok(hess)
}

/// Check whether the value function appears convex around a point.
pub fn check_local_convexity(
    oracle: &dyn ValueFunctionOracle,
    x: &[f64],
    step_size: f64,
) -> VFResult<bool> {
    let hess = finite_difference_hessian(oracle, x, step_size)?;
    let n = hess.len();

    for i in 0..n {
        let diag = hess[i][i];
        let off_diag_sum: f64 = (0..n).filter(|&j| j != i).map(|j| hess[i][j].abs()).sum();
        if diag < -off_diag_sum - TOLERANCE {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Multi-point feasibility classification.
pub fn classify_feasibility(
    oracle: &dyn ValueFunctionOracle,
    points: &[Vec<f64>],
) -> (Vec<usize>, Vec<usize>) {
    let mut feasible = Vec::new();
    let mut infeasible = Vec::new();

    for (i, x) in points.iter().enumerate() {
        match oracle.check_feasibility(x) {
            Ok(info) if info.is_feasible => feasible.push(i),
            _ => infeasible.push(i),
        }
    }

    (feasible, infeasible)
}

/// Estimate the Lipschitz constant of φ from sample evaluations.
pub fn estimate_lipschitz_constant(
    oracle: &dyn ValueFunctionOracle,
    points: &[Vec<f64>],
) -> VFResult<f64> {
    if points.len() < 2 {
        return Ok(0.0);
    }

    let mut values: Vec<(usize, f64)> = Vec::new();
    for (i, x) in points.iter().enumerate() {
        if let Ok(v) = oracle.value(x) {
            values.push((i, v));
        }
    }

    let mut max_ratio = 0.0f64;
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            let (idx_i, val_i) = values[i];
            let (idx_j, val_j) = values[j];

            let dist: f64 = points[idx_i]
                .iter()
                .zip(points[idx_j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist > TOLERANCE {
                let ratio = (val_i - val_j).abs() / dist;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
    }

    Ok(max_ratio)
}

/// Compute a lower bound on φ(x) using weak duality.
pub fn weak_duality_bound(problem: &BilevelProblem, x: &[f64], dual_multipliers: &[f64]) -> f64 {
    let mut rhs = problem.lower_b.clone();
    for entry in &problem.lower_linking_b.entries {
        if entry.col < x.len() && entry.row < rhs.len() {
            rhs[entry.row] += entry.value * x[entry.col];
        }
    }
    rhs.iter()
        .zip(dual_multipliers.iter())
        .map(|(b, pi)| b * pi)
        .sum()
}

/// Verify complementary slackness conditions for a primal-dual pair.
pub fn verify_complementary_slackness(
    problem: &BilevelProblem,
    x: &[f64],
    y: &[f64],
    dual: &[f64],
    tol: f64,
) -> bool {
    let lp = problem.lower_level_lp(x);
    let a_dense = lp.a_matrix.to_dense();
    let m = lp.num_constraints;
    let n = lp.num_vars;

    for i in 0..m {
        let ay: f64 = (0..n.min(y.len())).map(|j| a_dense[(i, j)] * y[j]).sum();
        let slack = lp.b_rhs[i] - ay;
        if i < dual.len() {
            if dual[i].abs() > tol && slack.abs() > tol {
                return false;
            }
        }
    }

    for j in 0..n.min(y.len()) {
        let reduced_cost = lp.c[j]
            - (0..m)
                .map(|i| {
                    let d = if i < dual.len() { dual[i] } else { 0.0 };
                    a_dense[(i, j)] * d
                })
                .sum::<f64>();

        if y[j].abs() > tol && reduced_cost.abs() > tol {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};
    use std::sync::Arc;

    fn simple_bilevel() -> BilevelProblem {
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

    #[test]
    fn test_exact_oracle_evaluate() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let info = oracle.evaluate(&[1.0]).unwrap();
        assert_eq!(info.primal_solution.len(), 1);
        assert!(info.value >= -TOLERANCE);
    }

    #[test]
    fn test_exact_oracle_dual_info() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let dual = oracle.dual_info(&[1.0]).unwrap();
        assert_eq!(dual.subgradient.len(), 1);
    }

    #[test]
    fn test_exact_oracle_feasibility() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let feas = oracle.check_feasibility(&[1.0]).unwrap();
        assert!(feas.is_feasible);
    }

    #[test]
    fn test_exact_oracle_dimension_mismatch() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let result = oracle.evaluate(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cached_oracle() {
        let problem = simple_bilevel();
        let inner = Arc::new(ExactLpOracle::with_default_solver(problem));
        let cached = CachedOracle::new(inner, 100);

        let _info1 = cached.evaluate(&[1.0]).unwrap();
        let _info2 = cached.evaluate(&[1.0]).unwrap();

        let stats = cached.statistics();
        assert!(stats.cache_hits >= 1);
    }

    #[test]
    fn test_cached_oracle_clear() {
        let problem = simple_bilevel();
        let inner = Arc::new(ExactLpOracle::with_default_solver(problem));
        let cached = CachedOracle::new(inner, 100);

        cached.evaluate(&[1.0]).unwrap();
        assert_eq!(cached.cache_size(), 1);

        cached.clear_cache();
        assert_eq!(cached.cache_size(), 0);
    }

    #[test]
    fn test_batch_evaluate() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let results = batch_evaluate(&oracle, &points);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok());
        }
    }

    #[test]
    fn test_finite_difference_gradient() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let grad = finite_difference_gradient(&oracle, &[1.0], 1e-5).unwrap();
        assert_eq!(grad.len(), 1);
        assert!(grad[0].abs() < 0.1);
    }

    #[test]
    fn test_estimate_lipschitz() {
        let problem = simple_bilevel();
        let oracle = ExactLpOracle::with_default_solver(problem);
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let lip = estimate_lipschitz_constant(&oracle, &points).unwrap();
        assert!(lip >= 0.0);
    }

    #[test]
    fn test_statistics_tracking() {
        let tracker = OracleStatsTracker::new();
        tracker.record_evaluation();
        tracker.record_evaluation();
        tracker.record_lp_solve(10);
        tracker.record_cache_hit();
        tracker.record_cache_miss();

        let snap = tracker.snapshot();
        assert_eq!(snap.total_evaluations, 2);
        assert_eq!(snap.cache_hits, 1);
        assert_eq!(snap.cache_misses, 1);

        tracker.reset();
        let snap2 = tracker.snapshot();
        assert_eq!(snap2.total_evaluations, 0);
    }
}
