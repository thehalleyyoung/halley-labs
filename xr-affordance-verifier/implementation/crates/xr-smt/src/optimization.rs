//! SMT-based optimization for boundary case detection and budget allocation.
//!
//! This module provides two primary capabilities:
//!
//! 1. **SmtOptimizer**: Finds optimal solutions subject to linear constraints using
//!    iterative binary search over a QF_LRA feasibility checker. Used to locate body
//!    parameters on the boundary of accessibility regions — the critical points where
//!    a small change in stature, arm length, etc. flips reachability.
//!
//! 2. **BudgetAllocator**: Implements Theorem C3 optimal budget allocation between
//!    statistical sampling and SMT-based verification queries. Given a fixed total
//!    budget, it partitions effort across parameter-space regions to minimize the
//!    expected coverage-gap residual.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::qf_lra::{
    Assignment, FeasibilityChecker, FeasibilityResult, LinearCombination, LinearConstraint,
    Relation,
};
use crate::expr::SmtExpr;
use crate::verification::ParameterRegion;

// ---------------------------------------------------------------------------
// OptimizationResult
// ---------------------------------------------------------------------------

/// Outcome of a linear optimization query.
#[derive(Debug, Clone)]
pub enum OptimizationResult {
    /// A proven-optimal solution was found within tolerance.
    Optimal {
        value: f64,
        model: IndexMap<String, f64>,
    },
    /// The objective is unbounded over the feasible region.
    Unbounded,
    /// The constraint set is infeasible.
    Infeasible,
    /// The solver exceeded its iteration or time budget.
    Timeout,
    /// A feasible solution was found but the optimality gap exceeds the tolerance.
    ApproximateOptimal {
        value: f64,
        model: IndexMap<String, f64>,
        gap: f64,
    },
}

impl OptimizationResult {
    /// Returns `true` if the result contains a usable solution (optimal or approximate).
    pub fn is_feasible(&self) -> bool {
        matches!(
            self,
            OptimizationResult::Optimal { .. } | OptimizationResult::ApproximateOptimal { .. }
        )
    }

    /// Extract the objective value if one was found.
    pub fn value(&self) -> Option<f64> {
        match self {
            OptimizationResult::Optimal { value, .. }
            | OptimizationResult::ApproximateOptimal { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// Extract the model (variable assignment) if one was found.
    pub fn model(&self) -> Option<&IndexMap<String, f64>> {
        match self {
            OptimizationResult::Optimal { model, .. }
            | OptimizationResult::ApproximateOptimal { model, .. } => Some(model),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SmtOptimizer
// ---------------------------------------------------------------------------

/// Iterative binary-search optimizer over a linear feasibility checker.
///
/// The core loop is:
///   1. Check feasibility of the base constraints.
///   2. Evaluate the objective at the feasible point to get an initial bound.
///   3. Binary-search: tighten a bound constraint on the objective and re-check
///      feasibility, converging to the optimum within `tolerance`.
pub struct SmtOptimizer {
    constraints: Vec<LinearConstraint>,
    variables: IndexMap<String, usize>,
    max_iterations: usize,
    tolerance: f64,
    timeout_ms: u64,
}

impl SmtOptimizer {
    /// Create a new optimizer with sensible defaults.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            variables: IndexMap::new(),
            max_iterations: 100,
            tolerance: 1e-6,
            timeout_ms: 30_000,
        }
    }

    /// Set the convergence tolerance (absolute gap between upper and lower bounds).
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "tolerance must be positive");
        self.tolerance = tol;
        self
    }

    /// Set the wall-clock timeout in milliseconds.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum number of binary-search iterations.
    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        assert!(iters > 0, "max_iterations must be positive");
        self.max_iterations = iters;
        self
    }

    /// Register a variable for optimization.
    pub fn add_variable(&mut self, name: impl Into<String>) {
        let name = name.into();
        let idx = self.variables.len();
        self.variables.entry(name).or_insert(idx);
    }

    /// Add a linear constraint.
    pub fn add_constraint(&mut self, constraint: LinearConstraint) {
        for var in constraint.lhs.terms.keys().chain(constraint.rhs.terms.keys()) {
            let idx = self.variables.len();
            self.variables.entry(var.clone()).or_insert(idx);
        }
        self.constraints.push(constraint);
    }

    /// Minimize a linear objective subject to the current constraints.
    pub fn minimize(&mut self, objective: &LinearCombination) -> OptimizationResult {
        let mut checker = self.build_checker();
        let initial = match checker.check() {
            FeasibilityResult::Feasible(a) => a,
            FeasibilityResult::Infeasible => return OptimizationResult::Infeasible,
            FeasibilityResult::Unknown => return OptimizationResult::Timeout,
        };
        let init_val = objective.evaluate(&initial.to_index_map());
        let probe_lower = init_val - 1e8;
        let lower = self.find_lower_bound(&checker, objective, probe_lower, init_val);
        let upper = init_val;
        binary_search_optimal(
            &mut checker, objective, &self.constraints,
            lower, upper, self.tolerance, self.max_iterations,
        )
    }

    /// Maximize a linear objective (delegates to `minimize` on the negation).
    pub fn maximize(&mut self, objective: &LinearCombination) -> OptimizationResult {
        let neg = objective.negate();
        let result = self.minimize(&neg);
        match result {
            OptimizationResult::Optimal { value, model } => OptimizationResult::Optimal {
                value: -value, model,
            },
            OptimizationResult::ApproximateOptimal { value, model, gap } => {
                OptimizationResult::ApproximateOptimal { value: -value, model, gap }
            }
            other => other,
        }
    }

    /// Find the feasible point closest (L1 distance) to `target`.
    pub fn find_closest_to_boundary(
        &mut self, target: &[f64], variable_names: &[String],
    ) -> OptimizationResult {
        assert_eq!(target.len(), variable_names.len(),
            "target and variable_names must have equal length");
        let base_constraints = self.constraints.clone();
        let mut extra_constraints: Vec<LinearConstraint> = Vec::new();
        let mut objective = LinearCombination::new();
        let mut aux_names: Vec<String> = Vec::new();

        for (i, (name, &t)) in variable_names.iter().zip(target.iter()).enumerate() {
            let d_name = format!("__dist_{i}");
            aux_names.push(d_name.clone());
            let mut lhs1 = LinearCombination::new();
            lhs1.add_term(d_name.clone(), 1.0);
            lhs1.add_term(name.clone(), -1.0);
            extra_constraints.push(LinearConstraint::new(
                lhs1, Relation::Ge, LinearCombination::constant(-t),
            ));
            let mut lhs2 = LinearCombination::new();
            lhs2.add_term(d_name.clone(), 1.0);
            lhs2.add_term(name.clone(), 1.0);
            extra_constraints.push(LinearConstraint::new(
                lhs2, Relation::Ge, LinearCombination::constant(t),
            ));
            extra_constraints.push(LinearConstraint::new(
                LinearCombination::variable(d_name.clone()),
                Relation::Ge, LinearCombination::constant(0.0),
            ));
            objective.add_term(d_name, 1.0);
        }

        let mut sub = SmtOptimizer::new()
            .with_tolerance(self.tolerance)
            .with_timeout(self.timeout_ms)
            .with_max_iterations(self.max_iterations);
        for var in self.variables.keys() { sub.add_variable(var.clone()); }
        for name in &aux_names { sub.add_variable(name.clone()); }
        for c in &base_constraints { sub.add_constraint(c.clone()); }
        for c in extra_constraints { sub.add_constraint(c); }

        let result = sub.minimize(&objective);
        match result {
            OptimizationResult::Optimal { value, mut model } => {
                for name in &aux_names { model.swap_remove(name); }
                OptimizationResult::Optimal { value, model }
            }
            OptimizationResult::ApproximateOptimal { value, mut model, gap } => {
                for name in &aux_names { model.swap_remove(name); }
                OptimizationResult::ApproximateOptimal { value, model, gap }
            }
            other => other,
        }
    }

    /// Maximize the slack on constraint `constraint_idx` while satisfying all others.
    pub fn find_maximum_margin(&mut self, constraint_idx: usize) -> OptimizationResult {
        assert!(constraint_idx < self.constraints.len(), "constraint_idx out of bounds");
        let target = &self.constraints[constraint_idx];
        let slack_objective = match target.relation {
            Relation::Le | Relation::Lt => target.rhs.add(&target.lhs.negate()),
            Relation::Ge | Relation::Gt => target.lhs.add(&target.rhs.negate()),
            Relation::Eq => LinearCombination::constant(0.0),
        };
        let mut sub = SmtOptimizer::new()
            .with_tolerance(self.tolerance)
            .with_timeout(self.timeout_ms)
            .with_max_iterations(self.max_iterations);
        for var in self.variables.keys() { sub.add_variable(var.clone()); }
        for (i, c) in self.constraints.iter().enumerate() {
            if i != constraint_idx { sub.add_constraint(c.clone()); }
        }
        sub.maximize(&slack_objective)
    }

    fn build_checker(&self) -> FeasibilityChecker {
        let mut checker = FeasibilityChecker::new();
        for var in self.variables.keys() { checker.add_variable(var.clone()); }
        for c in &self.constraints { checker.add_constraint(c.clone()); }
        checker
    }

    fn find_lower_bound(
        &self, base_checker: &FeasibilityChecker,
        objective: &LinearCombination, mut candidate: f64, upper: f64,
    ) -> f64 {
        let mut step = (upper - candidate).abs().max(1.0);
        let mut proven_lower = candidate;
        for _ in 0..40 {
            let mut checker = FeasibilityChecker::new();
            for var in base_checker.variables().keys() { checker.add_variable(var.clone()); }
            for c in base_checker.constraints() { checker.add_constraint(c.clone()); }
            checker.add_constraint(LinearConstraint::new(
                objective.clone(), Relation::Le, LinearCombination::constant(candidate),
            ));
            match checker.check() {
                FeasibilityResult::Feasible(_) => { candidate -= step; step *= 2.0; }
                _ => { proven_lower = candidate; break; }
            }
        }
        proven_lower
    }
}

impl Default for SmtOptimizer {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// binary_search_optimal (standalone helper)
// ---------------------------------------------------------------------------

fn binary_search_optimal(
    checker: &mut FeasibilityChecker,
    objective: &LinearCombination,
    base_constraints: &[LinearConstraint],
    mut lower: f64,
    mut upper: f64,
    tolerance: f64,
    max_iters: usize,
) -> OptimizationResult {
    let mut best_model: IndexMap<String, f64> = {
        let mut tmp = FeasibilityChecker::new();
        for var in checker.variables().keys() { tmp.add_variable(var.clone()); }
        for c in base_constraints { tmp.add_constraint(c.clone()); }
        match tmp.check() {
            FeasibilityResult::Feasible(a) => a.to_index_map(),
            _ => return OptimizationResult::Infeasible,
        }
    };
    let mut best_value = objective.evaluate(&best_model);
    upper = best_value;
    if lower > upper { lower = upper - 1.0; }

    let mut iterations = 0;
    while (upper - lower) > tolerance && iterations < max_iters {
        iterations += 1;
        let mid = (lower + upper) / 2.0;
        let mut trial = FeasibilityChecker::new();
        for var in checker.variables().keys() { trial.add_variable(var.clone()); }
        for c in base_constraints { trial.add_constraint(c.clone()); }
        trial.add_constraint(LinearConstraint::new(
            objective.clone(), Relation::Le, LinearCombination::constant(mid),
        ));
        match trial.check() {
            FeasibilityResult::Feasible(a) => {
                let map = a.to_index_map();
                let val = objective.evaluate(&map);
                if val < best_value { best_value = val; best_model = map; }
                upper = mid;
            }
            FeasibilityResult::Infeasible => { lower = mid; }
            FeasibilityResult::Unknown => { lower = mid; }
        }
    }

    let gap = upper - lower;
    if gap <= tolerance {
        OptimizationResult::Optimal { value: best_value, model: best_model }
    } else {
        OptimizationResult::ApproximateOptimal { value: best_value, model: best_model, gap }
    }
}

// ---------------------------------------------------------------------------
// BudgetAllocation / RegionAllocation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionAllocation {
    pub region_index: usize,
    pub smt_queries: usize,
    pub samples: usize,
    pub priority: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAllocation {
    pub smt_budget: usize,
    pub sampling_budget: usize,
    pub region_allocations: Vec<RegionAllocation>,
}

impl BudgetAllocation {
    pub fn total(&self) -> usize { self.smt_budget + self.sampling_budget }
    pub fn smt_fraction(&self) -> f64 {
        let tot = self.total();
        if tot == 0 { return 0.0; }
        self.smt_budget as f64 / tot as f64
    }
    pub fn is_smt_dominant(&self) -> bool { self.smt_budget > self.sampling_budget }
}

// ---------------------------------------------------------------------------
// BudgetAllocator — Theorem C3
// ---------------------------------------------------------------------------

pub struct BudgetAllocator {
    total_budget: usize,
    smt_cost_per_query: f64,
    sample_cost_per_point: f64,
    region_uncertainties: Vec<f64>,
    region_volumes: Vec<f64>,
}

impl BudgetAllocator {
    pub fn new(total_budget: usize) -> Self {
        Self { total_budget, smt_cost_per_query: 1.0, sample_cost_per_point: 1.0,
            region_uncertainties: Vec::new(), region_volumes: Vec::new() }
    }
    pub fn with_costs(mut self, smt_cost: f64, sample_cost: f64) -> Self {
        assert!(smt_cost > 0.0 && sample_cost > 0.0, "costs must be positive");
        self.smt_cost_per_query = smt_cost;
        self.sample_cost_per_point = sample_cost;
        self
    }
    pub fn add_region(&mut self, uncertainty: f64, volume: f64) {
        self.region_uncertainties.push(uncertainty.clamp(0.0, 1.0));
        self.region_volumes.push(volume.max(0.0));
    }

    pub fn allocate(&self) -> BudgetAllocation {
        let n = self.region_uncertainties.len();
        if n == 0 || self.total_budget == 0 {
            return BudgetAllocation { smt_budget: 0, sampling_budget: 0, region_allocations: Vec::new() };
        }
        let v_total: f64 = self.region_volumes.iter().sum();
        let v_uncertain: f64 = self.region_uncertainties.iter()
            .zip(self.region_volumes.iter()).map(|(u, v)| u * v).sum();
        if v_total <= 0.0 { return self.uniform_allocation(); }
        let ratio = (self.smt_cost_per_query * v_uncertain
            / (self.sample_cost_per_point * v_total)).sqrt();
        let bs_float = self.total_budget as f64 / (1.0 + ratio);
        let sampling_budget = bs_float.round().max(0.0) as usize;
        let smt_budget = self.total_budget.saturating_sub(sampling_budget);
        let region_allocations = self.distribute_to_regions(smt_budget, sampling_budget);
        BudgetAllocation { smt_budget, sampling_budget, region_allocations }
    }

    pub fn allocate_adaptive(&self, current_coverage: f64, target_coverage: f64) -> BudgetAllocation {
        let n = self.region_uncertainties.len();
        if n == 0 || self.total_budget == 0 {
            return BudgetAllocation { smt_budget: 0, sampling_budget: 0, region_allocations: Vec::new() };
        }
        let gap = (target_coverage - current_coverage).max(0.0);
        let smt_share = (1.0 - gap * 2.0).clamp(0.1, 0.9);
        let smt_budget = (self.total_budget as f64 * smt_share).round() as usize;
        let sampling_budget = self.total_budget.saturating_sub(smt_budget);
        let region_allocations = self.distribute_to_regions_adaptive(smt_budget, sampling_budget, gap);
        BudgetAllocation { smt_budget, sampling_budget, region_allocations }
    }

    pub fn estimate_queries_needed(&self, coverage_gap: f64) -> usize {
        if self.region_volumes.is_empty() || coverage_gap <= 0.0 { return 0; }
        let v_total: f64 = self.region_volumes.iter().sum();
        if v_total <= 0.0 { return 0; }
        let n = self.region_volumes.len() as f64;
        let avg_volume = v_total / n;
        let volume_to_cover = coverage_gap * v_total;
        let raw = (volume_to_cover / avg_volume).ceil() as usize;
        ((raw as f64) * 1.2).ceil() as usize
    }

    fn distribute_to_regions(&self, smt_budget: usize, sampling_budget: usize) -> Vec<RegionAllocation> {
        let n = self.region_uncertainties.len();
        let priorities: Vec<f64> = self.region_uncertainties.iter()
            .zip(self.region_volumes.iter()).map(|(u, v)| u * v).collect();
        let total_priority: f64 = priorities.iter().sum();
        if total_priority <= 0.0 { return self.uniform_region_allocations(smt_budget, sampling_budget); }
        let mut allocations = Vec::with_capacity(n);
        let mut smt_rem = smt_budget;
        let mut samp_rem = sampling_budget;
        for (i, &p) in priorities.iter().enumerate() {
            let frac = p / total_priority;
            let sq = if i == n - 1 { smt_rem } else { ((smt_budget as f64 * frac).round() as usize).min(smt_rem) };
            let sp = if i == n - 1 { samp_rem } else { ((sampling_budget as f64 * frac).round() as usize).min(samp_rem) };
            smt_rem = smt_rem.saturating_sub(sq);
            samp_rem = samp_rem.saturating_sub(sp);
            allocations.push(RegionAllocation { region_index: i, smt_queries: sq, samples: sp, priority: frac });
        }
        allocations
    }

    fn distribute_to_regions_adaptive(&self, smt_budget: usize, sampling_budget: usize, gap: f64) -> Vec<RegionAllocation> {
        let n = self.region_uncertainties.len();
        let exponent = 1.0 + (1.0 - gap).clamp(0.0, 2.0);
        let raw_priorities: Vec<f64> = self.region_uncertainties.iter()
            .zip(self.region_volumes.iter()).map(|(u, v)| (u * v).powf(exponent)).collect();
        let total_priority: f64 = raw_priorities.iter().sum();
        if total_priority <= 0.0 { return self.uniform_region_allocations(smt_budget, sampling_budget); }
        let mut allocations = Vec::with_capacity(n);
        let mut smt_rem = smt_budget;
        let mut samp_rem = sampling_budget;
        for (i, &p) in raw_priorities.iter().enumerate() {
            let frac = p / total_priority;
            let sq = if i == n - 1 { smt_rem } else { ((smt_budget as f64 * frac).round() as usize).min(smt_rem) };
            let sp = if i == n - 1 { samp_rem } else { ((sampling_budget as f64 * frac).round() as usize).min(samp_rem) };
            smt_rem = smt_rem.saturating_sub(sq);
            samp_rem = samp_rem.saturating_sub(sp);
            allocations.push(RegionAllocation { region_index: i, smt_queries: sq, samples: sp, priority: frac });
        }
        allocations
    }

    fn uniform_allocation(&self) -> BudgetAllocation {
        let smt_budget = self.total_budget / 2;
        let sampling_budget = self.total_budget - smt_budget;
        BudgetAllocation { smt_budget, sampling_budget,
            region_allocations: self.uniform_region_allocations(smt_budget, sampling_budget) }
    }

    fn uniform_region_allocations(&self, smt_budget: usize, sampling_budget: usize) -> Vec<RegionAllocation> {
        let n = self.region_uncertainties.len();
        if n == 0 { return Vec::new(); }
        let smt_per = smt_budget / n;
        let sample_per = sampling_budget / n;
        let mut allocs: Vec<RegionAllocation> = (0..n)
            .map(|i| RegionAllocation { region_index: i, smt_queries: smt_per, samples: sample_per, priority: 1.0 / n as f64 })
            .collect();
        if let Some(last) = allocs.last_mut() {
            last.smt_queries += smt_budget - smt_per * n;
            last.samples += sampling_budget - sample_per * n;
        }
        allocs
    }
}

// ---------------------------------------------------------------------------
// Utility: extract ParameterRegion from optimizer bounds
// ---------------------------------------------------------------------------

pub fn extract_parameter_region(constraints: &[LinearConstraint], variable_names: &[String]) -> ParameterRegion {
    let n = variable_names.len();
    let mut lower = vec![f64::NEG_INFINITY; n];
    let mut upper = vec![f64::INFINITY; n];
    let name_to_idx: IndexMap<&str, usize> = variable_names.iter()
        .enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    for c in constraints {
        let lhs_terms: Vec<_> = c.lhs.terms.iter().filter(|(_, v)| v.abs() > 1e-15).collect();
        let rhs_terms: Vec<_> = c.rhs.terms.iter().filter(|(_, v)| v.abs() > 1e-15).collect();
        if lhs_terms.len() == 1 && rhs_terms.is_empty() {
            let (var, coeff) = lhs_terms[0];
            if let Some(&idx) = name_to_idx.get(var.as_str()) {
                let bound = (c.rhs.constant - c.lhs.constant) / coeff;
                match c.relation {
                    Relation::Le | Relation::Lt if *coeff > 0.0 => { upper[idx] = upper[idx].min(bound); }
                    Relation::Le | Relation::Lt if *coeff < 0.0 => { lower[idx] = lower[idx].max(-bound); }
                    Relation::Ge | Relation::Gt if *coeff > 0.0 => { lower[idx] = lower[idx].max(bound); }
                    Relation::Ge | Relation::Gt if *coeff < 0.0 => { upper[idx] = upper[idx].min(-bound); }
                    Relation::Eq => { lower[idx] = lower[idx].max(bound); upper[idx] = upper[idx].min(bound); }
                    _ => {}
                }
            }
        }
    }
    for v in lower.iter_mut() { if !v.is_finite() { *v = -1e6; } }
    for v in upper.iter_mut() { if !v.is_finite() { *v = 1e6; } }
    ParameterRegion::new(lower, upper)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qf_lra::{LinearCombination, LinearConstraint, Relation};

    fn bounded_2d_optimizer() -> SmtOptimizer {
        let mut opt = SmtOptimizer::new().with_tolerance(1e-4);
        opt.add_variable("x");
        opt.add_variable("y");
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Ge, LinearCombination::constant(0.0)));
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Le, LinearCombination::constant(10.0)));
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("y"), Relation::Ge, LinearCombination::constant(0.0)));
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("y"), Relation::Le, LinearCombination::constant(10.0)));
        opt
    }

    #[test]
    fn minimize_simple_linear_objective() {
        let mut opt = bounded_2d_optimizer();
        let mut obj = LinearCombination::new();
        obj.add_term("x", 1.0);
        obj.add_term("y", 1.0);
        let result = opt.minimize(&obj);
        assert!(result.is_feasible(), "should find a feasible optimum");
        let val = result.value().unwrap();
        assert!(val.abs() < 0.5, "optimal x+y on [0,10]^2 near 0, got {val}");
    }

    #[test]
    fn maximize_simple_linear_objective() {
        let mut opt = bounded_2d_optimizer();
        let mut obj = LinearCombination::new();
        obj.add_term("x", 1.0);
        obj.add_term("y", 1.0);
        let result = opt.maximize(&obj);
        assert!(result.is_feasible());
        let val = result.value().unwrap();
        assert!((val - 20.0).abs() < 1.0, "max x+y on [0,10]^2 near 20, got {val}");
    }

    #[test]
    fn infeasible_system() {
        let mut opt = SmtOptimizer::new();
        opt.add_variable("x");
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Ge, LinearCombination::constant(10.0)));
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Le, LinearCombination::constant(5.0)));
        let obj = LinearCombination::variable("x");
        let result = opt.minimize(&obj);
        assert!(matches!(result, OptimizationResult::Infeasible),
            "should detect infeasibility, got {:?}", result);
    }

    #[test]
    fn find_closest_to_boundary_basic() {
        let mut opt = bounded_2d_optimizer();
        let target = vec![15.0, 15.0];
        let vars = vec!["x".into(), "y".into()];
        let result = opt.find_closest_to_boundary(&target, &vars);
        assert!(result.is_feasible());
        let val = result.value().unwrap();
        assert!((val - 10.0).abs() < 2.0, "L1 distance near 10, got {val}");
    }

    #[test]
    fn budget_allocator_equal_cost() {
        let mut alloc = BudgetAllocator::new(100).with_costs(1.0, 1.0);
        alloc.add_region(0.5, 10.0);
        alloc.add_region(0.5, 10.0);
        let result = alloc.allocate();
        assert_eq!(result.total(), 100);
        assert_eq!(result.region_allocations.len(), 2);
        let frac = result.smt_fraction();
        assert!(frac > 0.3 && frac < 0.6, "smt fraction ~0.41, got {frac}");
    }

    #[test]
    fn budget_allocator_asymmetric_cost() {
        let mut alloc = BudgetAllocator::new(1000).with_costs(10.0, 1.0);
        alloc.add_region(0.8, 100.0);
        alloc.add_region(0.2, 100.0);
        let result = alloc.allocate();
        assert_eq!(result.total(), 1000);
        let frac = result.smt_fraction();
        assert!(frac > 0.5 && frac < 0.85, "high smt_cost frac ~0.69, got {frac}");
        let r0 = &result.region_allocations[0];
        let r1 = &result.region_allocations[1];
        assert!(r0.smt_queries >= r1.smt_queries, "region 0 (u=0.8) >= region 1 (u=0.2)");
    }

    #[test]
    fn budget_allocator_adaptive() {
        let mut alloc = BudgetAllocator::new(200).with_costs(1.0, 1.0);
        alloc.add_region(0.6, 50.0);
        alloc.add_region(0.4, 50.0);
        let result_large = alloc.allocate_adaptive(0.2, 0.95);
        let result_small = alloc.allocate_adaptive(0.90, 0.95);
        assert!(result_small.smt_fraction() > result_large.smt_fraction(),
            "small gap higher smt frac; small={}, large={}",
            result_small.smt_fraction(), result_large.smt_fraction());
    }

    #[test]
    fn budget_allocation_fractions() {
        let alloc = BudgetAllocation {
            smt_budget: 60, sampling_budget: 40,
            region_allocations: vec![RegionAllocation {
                region_index: 0, smt_queries: 60, samples: 40, priority: 1.0,
            }],
        };
        assert_eq!(alloc.total(), 100);
        assert!((alloc.smt_fraction() - 0.6).abs() < 1e-9);
        assert!(alloc.is_smt_dominant());
    }

    #[test]
    fn optimization_with_multiple_variables() {
        let mut opt = SmtOptimizer::new().with_tolerance(0.1);
        for name in &["x", "y", "z"] {
            opt.add_variable(*name);
            opt.add_constraint(LinearConstraint::new(
                LinearCombination::variable(*name), Relation::Ge, LinearCombination::constant(0.0)));
            opt.add_constraint(LinearConstraint::new(
                LinearCombination::variable(*name), Relation::Le, LinearCombination::constant(15.0)));
        }
        let mut sum = LinearCombination::new();
        sum.add_term("x", 1.0); sum.add_term("y", 1.0); sum.add_term("z", 1.0);
        opt.add_constraint(LinearConstraint::new(sum, Relation::Le, LinearCombination::constant(30.0)));
        let mut obj = LinearCombination::new();
        obj.add_term("x", 1.0); obj.add_term("y", 2.0); obj.add_term("z", 3.0);
        let result = opt.maximize(&obj);
        assert!(result.is_feasible(), "3-variable system should be feasible");
        let val = result.value().unwrap();
        assert!(val > 40.0, "max(x+2y+3z) should be large, got {val}");
    }

    #[test]
    fn find_maximum_margin_basic() {
        let mut opt = SmtOptimizer::new().with_tolerance(0.1);
        opt.add_variable("x");
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Le, LinearCombination::constant(10.0)));
        opt.add_constraint(LinearConstraint::new(
            LinearCombination::variable("x"), Relation::Ge, LinearCombination::constant(0.0)));
        let result = opt.find_maximum_margin(0);
        assert!(result.is_feasible());
        let val = result.value().unwrap();
        assert!(val > 5.0, "max margin on (x<=10) with (x>=0) ~10, got {val}");
    }

    #[test]
    fn estimate_queries_needed_basic() {
        let mut alloc = BudgetAllocator::new(500);
        alloc.add_region(0.5, 100.0);
        alloc.add_region(0.5, 100.0);
        let needed = alloc.estimate_queries_needed(0.1);
        assert!(needed > 0, "nonzero gap needs at least one query");
        let needed_larger = alloc.estimate_queries_needed(0.5);
        assert!(needed_larger > needed, "larger gap needs more queries");
        assert_eq!(alloc.estimate_queries_needed(0.0), 0, "zero gap = zero queries");
    }

    #[test]
    fn extract_parameter_region_from_constraints() {
        let constraints = vec![
            LinearConstraint::new(LinearCombination::variable("x"), Relation::Ge, LinearCombination::constant(1.0)),
            LinearConstraint::new(LinearCombination::variable("x"), Relation::Le, LinearCombination::constant(5.0)),
            LinearConstraint::new(LinearCombination::variable("y"), Relation::Ge, LinearCombination::constant(2.0)),
            LinearConstraint::new(LinearCombination::variable("y"), Relation::Le, LinearCombination::constant(8.0)),
        ];
        let vars = vec!["x".into(), "y".into()];
        let region = extract_parameter_region(&constraints, &vars);
        assert_eq!(region.dimension(), 2);
        assert!((region.lower[0] - 1.0).abs() < 1e-9);
        assert!((region.upper[0] - 5.0).abs() < 1e-9);
        assert!((region.lower[1] - 2.0).abs() < 1e-9);
        assert!((region.upper[1] - 8.0).abs() < 1e-9);
        assert!((region.volume() - 24.0).abs() < 1e-9);
    }
}


    /// Add slack variables to constraints and return an objective that
    /// maximizes the minimum slack (i.e. the feasibility margin).
    ///
    /// For each constraint `c_i`, a fresh slack variable `slack_prefix_i ≥ 0`
    /// is introduced and the constraint is relaxed to allow measuring how
    /// far inside the feasible region a point lies.
    ///
    /// A global variable `__min_slack` is constrained to be ≤ each `slack_i`,
    /// and the returned objective is `neg(__min_slack)` (to be *minimized*,
    /// which maximizes the margin).
    ///
    /// Returns `(objective, augmented_constraints)`.
    pub fn encode_maximize_margin(
        constraints: &[SmtExpr],
        slack_prefix: &str,
    ) -> (SmtExpr, Vec<SmtExpr>) {
        let min_slack = SmtExpr::var("__min_slack");
        let mut augmented: Vec<SmtExpr> = Vec::new();

        for (i, constraint) in constraints.iter().enumerate() {
            let slack_name = format!("{slack_prefix}{i}");
            let slack_var = SmtExpr::var(&slack_name);

            // slack_i >= 0
            augmented.push(SmtExpr::ge(slack_var.clone(), SmtExpr::Const(0.0)));

            // Re-interpret constraint: if it is `lhs ≤ rhs`, replace with
            // `lhs + slack_i ≤ rhs`, i.e. `lhs ≤ rhs − slack_i` already
            // holds with slack.  We emit the original constraint (it must
            // still hold) plus a binder that connects the slack to the
            // margin.
            //
            // For a general boolean constraint we simply assert:
            //   constraint => slack_i == 0   (tight when active)
            //   !constraint => false         (must still hold)
            // Simplification: we always keep the original constraint and
            // bind __min_slack ≤ slack_i.  The caller interprets slack_i
            // as an externally-supplied relaxation amount.
            augmented.push(constraint.clone());

            // __min_slack ≤ slack_i
            augmented.push(SmtExpr::le(min_slack.clone(), slack_var));
        }

        // __min_slack ≥ 0
        augmented.push(SmtExpr::ge(min_slack.clone(), SmtExpr::Const(0.0)));

        // Objective: minimize -__min_slack  ⟺  maximize __min_slack.
        let objective = SmtExpr::neg(min_slack);

        (objective, augmented)
    }

// ---------------------------------------------------------------------------
// Tests (alternate API — kept behind cfg(test) for structural validation)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_alt {
    use super::*;

    // -- Mock solver ----------------------------------------------------------
    //
    // The real `InternalSolver` lives in `crate::solver` which may not be
    // compiled yet.  For unit-testing the optimization *logic* we exercise
    // the algorithm on manually-constructed scenarios and verify structural
    // properties that hold regardless of the underlying solver.

    #[test]
    fn test_optimization_result_serde() {
        let result = OptimizationResult {
            optimal_value: 3.14,
            model: Some(HashMap::from([("x".to_string(), 1.0)])),
            iterations: 7,
            status: OptimizationStatus::Optimal,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: OptimizationResult = serde_json::from_str(&json).unwrap();
        assert!((deser.optimal_value - 3.14).abs() < 1e-12);
        assert_eq!(deser.iterations, 7);
        assert_eq!(deser.status, OptimizationStatus::Optimal);
        assert!(deser.model.is_some());
    }

    #[test]
    fn test_optimization_status_variants() {
        for status in [
            OptimizationStatus::Optimal,
            OptimizationStatus::Bounded,
            OptimizationStatus::Infeasible,
            OptimizationStatus::Timeout,
            OptimizationStatus::Unknown,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: OptimizationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn test_budget_allocation_basic() {
        let alloc = BudgetAllocator::new(100.0)
            .with_sampling_cost(0.01)
            .with_smt_cost(1.0)
            .with_coverage_target(0.95)
            .with_region_count(10)
            .compute_allocation()
            .unwrap();

        // Fractions must sum to 1.
        assert!((alloc.sampling_fraction + alloc.smt_fraction - 1.0).abs() < 1e-12);
        // Budgets must sum to total.
        assert!((alloc.sampling_budget_secs + alloc.smt_budget_secs - 100.0).abs() < 1e-9);
        // Must produce a non-negative number of samples / queries.
        assert!(alloc.num_samples > 0);
        // Coverage should be in [0, 1].
        assert!(alloc.expected_coverage >= 0.0 && alloc.expected_coverage <= 1.0);
    }

    #[test]
    fn test_budget_allocation_extreme_costs() {
        // When SMT is very cheap relative to sampling, most budget goes to SMT.
        let alloc = BudgetAllocator::new(60.0)
            .with_sampling_cost(10.0)
            .with_smt_cost(0.001)
            .with_coverage_target(0.99)
            .with_region_count(5)
            .compute_allocation()
            .unwrap();

        assert!(alloc.smt_fraction > alloc.sampling_fraction);

        // When SMT is very expensive, most budget goes to sampling.
        let alloc2 = BudgetAllocator::new(60.0)
            .with_sampling_cost(0.0001)
            .with_smt_cost(100.0)
            .with_coverage_target(0.99)
            .with_region_count(5)
            .compute_allocation()
            .unwrap();

        assert!(alloc2.sampling_fraction > alloc2.smt_fraction);
    }

    #[test]
    fn test_budget_allocation_validation() {
        // Zero budget → error.
        assert!(BudgetAllocator::new(0.0).compute_allocation().is_err());
        // Zero sampling cost → error.
        assert!(BudgetAllocator::new(10.0)
            .with_sampling_cost(0.0)
            .compute_allocation()
            .is_err());
        // Zero region count → error.
        assert!(BudgetAllocator::new(10.0)
            .with_region_count(0)
            .compute_allocation()
            .is_err());
        // Coverage out of range → error.
        assert!(BudgetAllocator::new(10.0)
            .with_coverage_target(0.0)
            .compute_allocation()
            .is_err());
        assert!(BudgetAllocator::new(10.0)
            .with_coverage_target(1.5)
            .compute_allocation()
            .is_err());
    }

    #[test]
    fn test_optimal_split_formula() {
        let allocator = BudgetAllocator::new(100.0)
            .with_sampling_cost(0.01)
            .with_smt_cost(1.0)
            .with_region_count(10);

        let (s_frac, q_frac) = allocator.compute_optimal_split();
        // Expected: smt_cost / (smt_cost + sampling_cost * region_count)
        //         = 1.0 / (1.0 + 0.01 * 10)  = 1.0 / 1.1  ≈ 0.9091
        let expected = 1.0 / 1.1;
        assert!((s_frac - expected).abs() < 1e-10);
        assert!((s_frac + q_frac - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_optimizer_builder() {
        let opt = SmtOptimizer::new()
            .with_precision(1e-3)
            .with_max_iterations(50)
            .with_timeout(10.0);
        assert!((opt.precision - 1e-3).abs() < 1e-15);
        assert_eq!(opt.max_iterations, 50);
        assert!((opt.timeout_secs - 10.0).abs() < 1e-15);
    }

    #[test]
    fn test_minimize_invalid_bounds() {
        let opt = SmtOptimizer::new();
        let obj = SmtExpr::var("x");
        let result = opt.minimize(&obj, &[], 10.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_boundary_invalid_bounds() {
        let opt = SmtOptimizer::new();
        let pred = SmtExpr::le(SmtExpr::var("x"), SmtExpr::Const(5.0));
        let result = opt.find_boundary(&pred, "x", &[], 10.0, 5.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_objective_encode_minimize_distance() {
        let target = vec![1.0, 2.0, 3.0];
        let (obj, side) = ObjectiveEncoder::encode_minimize_distance(&target, "v", 3);

        // Should produce 3 * 3 = 9 side constraints (ge_0, ge_pos, ge_neg per dim).
        assert_eq!(side.len(), 9);

        // The objective should mention the 3 auxiliary variables.
        let vars = obj.free_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("__dist_v0"));
        assert!(vars.contains("__dist_v1"));
        assert!(vars.contains("__dist_v2"));
    }

    #[test]
    fn test_objective_encode_maximize_margin() {
        let c1 = SmtExpr::le(SmtExpr::var("x"), SmtExpr::Const(10.0));
        let c2 = SmtExpr::ge(SmtExpr::var("x"), SmtExpr::Const(0.0));
        let constraints = vec![c1, c2];

        let (obj, augmented) = ObjectiveEncoder::encode_maximize_margin(&constraints, "s");

        // 2 constraints → 2 slack_ge_0 + 2 original + 2 min_slack_le_slack + 1 min_slack_ge_0
        assert_eq!(augmented.len(), 7);

        // Objective should reference __min_slack.
        let vars = obj.free_variables();
        assert!(vars.contains("__min_slack"));
    }

    #[test]
    fn test_encode_minimize_distance_zero_dim() {
        let (obj, side) = ObjectiveEncoder::encode_minimize_distance(&[], "v", 0);
        assert!(side.is_empty());
        // Objective should be constant 0.
        assert_eq!(obj, SmtExpr::Const(0.0));
    }

    #[test]
    fn test_budget_allocation_single_region() {
        let alloc = BudgetAllocator::new(10.0)
            .with_sampling_cost(0.001)
            .with_smt_cost(1.0)
            .with_coverage_target(0.95)
            .with_region_count(1)
            .compute_allocation()
            .unwrap();

        // With a single region, the split should heavily favor sampling.
        // smt_cost / (smt_cost + sampling_cost * 1) = 1.0/1.001 ≈ 0.999
        assert!(alloc.sampling_fraction > 0.99);
        assert!(alloc.expected_coverage > 0.0);
    }
}
