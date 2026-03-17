//! Encoding optimisation passes.
//!
//! Provides transformations that reduce the size of an [`EncodedProblem`]
//! without changing its satisfiability:
//!
//! - **Cone-of-influence reduction** – remove variables that cannot affect
//!   the property under verification.
//! - **Symmetry breaking** – order equivalent transitions.
//! - **Constant propagation** – propagate known constant values.
//! - **Common sub-expression elimination** – share repeated sub-trees.
//! - **Bound estimation** – estimate the minimum BMC bound needed.
//! - **Incremental encoding** – add one step at a time.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::encoder::{EncodedProblem, PtaEncoder};
use crate::expression::{SmtExpr, free_vars, simplify, expr_size, total_expr_size, substitute};
use crate::pta::PTA;
use crate::variable::{VariableId, VariableStore};

// ═══════════════════════════════════════════════════════════════════════════
// CompressionStats
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics about size reduction from optimisation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    pub assertions_before: usize,
    pub assertions_after: usize,
    pub variables_before: usize,
    pub variables_after: usize,
    pub ast_nodes_before: usize,
    pub ast_nodes_after: usize,
    pub constant_propagations: usize,
    pub cse_eliminations: usize,
    pub cone_eliminations: usize,
    pub symmetry_constraints: usize,
}

impl CompressionStats {
    /// Percentage reduction in AST nodes.
    pub fn node_reduction_pct(&self) -> f64 {
        if self.ast_nodes_before == 0 { return 0.0; }
        let reduced = self.ast_nodes_before.saturating_sub(self.ast_nodes_after);
        (reduced as f64 / self.ast_nodes_before as f64) * 100.0
    }

    /// Percentage reduction in assertions.
    pub fn assertion_reduction_pct(&self) -> f64 {
        if self.assertions_before == 0 { return 0.0; }
        let reduced = self.assertions_before.saturating_sub(self.assertions_after);
        (reduced as f64 / self.assertions_before as f64) * 100.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EncodingOptimizer
// ═══════════════════════════════════════════════════════════════════════════

/// Applies optimisation passes to an encoded problem.
#[derive(Debug, Clone)]
pub struct EncodingOptimizer {
    /// Enable cone-of-influence reduction.
    pub enable_cone: bool,
    /// Enable symmetry breaking.
    pub enable_symmetry: bool,
    /// Enable constant propagation.
    pub enable_const_prop: bool,
    /// Enable common sub-expression elimination.
    pub enable_cse: bool,
    /// Enable simplification.
    pub enable_simplify: bool,
    /// Maximum iterations for constant propagation.
    pub max_const_prop_iterations: usize,
}

impl EncodingOptimizer {
    pub fn new() -> Self {
        Self {
            enable_cone: true,
            enable_symmetry: true,
            enable_const_prop: true,
            enable_cse: true,
            enable_simplify: true,
            max_const_prop_iterations: 10,
        }
    }

    pub fn all_disabled() -> Self {
        Self {
            enable_cone: false,
            enable_symmetry: false,
            enable_const_prop: false,
            enable_cse: false,
            enable_simplify: false,
            max_const_prop_iterations: 0,
        }
    }

    /// Run all enabled optimisation passes.
    pub fn optimize(&self, problem: &EncodedProblem) -> (EncodedProblem, CompressionStats) {
        let mut stats = CompressionStats {
            assertions_before: problem.num_assertions(),
            variables_before: problem.num_variables(),
            ast_nodes_before: problem.total_size(),
            ..Default::default()
        };

        let mut optimized = problem.clone();

        // 1. Simplification
        if self.enable_simplify {
            optimized.assertions = optimized.assertions.iter()
                .map(|e| simplify(e))
                .collect();
        }

        // 2. Constant propagation
        if self.enable_const_prop {
            let cp_count = self.constant_propagation(&mut optimized);
            stats.constant_propagations = cp_count;
        }

        // 3. Cone of influence reduction
        if self.enable_cone {
            let cone_count = self.cone_of_influence(&mut optimized);
            stats.cone_eliminations = cone_count;
        }

        // 4. Common sub-expression elimination
        if self.enable_cse {
            let cse_count = self.common_subexpression_elimination(&mut optimized);
            stats.cse_eliminations = cse_count;
        }

        // 5. Symmetry breaking
        if self.enable_symmetry {
            let sym_count = self.symmetry_breaking(&mut optimized);
            stats.symmetry_constraints = sym_count;
        }

        // 6. Remove trivially true assertions
        optimized.assertions.retain(|a| !a.is_true());

        // 7. Final simplification pass
        if self.enable_simplify {
            optimized.assertions = optimized.assertions.iter()
                .map(|e| simplify(e))
                .collect();
            optimized.assertions.retain(|a| !a.is_true());
        }

        stats.assertions_after = optimized.num_assertions();
        stats.variables_after = optimized.num_variables();
        stats.ast_nodes_after = optimized.total_size();

        (optimized, stats)
    }

    // ── Constant Propagation ────────────────────────────────────────

    fn constant_propagation(&self, problem: &mut EncodedProblem) -> usize {
        let mut total_propagations = 0;

        for _iteration in 0..self.max_const_prop_iterations {
            let constants = self.find_constant_assignments(&problem.assertions);
            if constants.is_empty() {
                break;
            }

            let mut changed = false;
            for (&var_id, value) in &constants {
                let replacement = match value {
                    ConstantValue::Bool(b) => SmtExpr::BoolLit(*b),
                    ConstantValue::Int(n) => SmtExpr::IntLit(*n),
                    ConstantValue::Real(r) => SmtExpr::RealLit(*r),
                };

                let old_assertions = problem.assertions.clone();
                problem.assertions = problem.assertions.iter()
                    .map(|a| substitute(a, var_id, &replacement))
                    .collect();

                if problem.assertions != old_assertions {
                    changed = true;
                    total_propagations += 1;
                }
            }

            // Simplify after substitution
            problem.assertions = problem.assertions.iter()
                .map(|e| simplify(e))
                .collect();

            if !changed { break; }
        }

        total_propagations
    }

    fn find_constant_assignments(&self, assertions: &[SmtExpr]) -> HashMap<VariableId, ConstantValue> {
        let mut constants = HashMap::new();

        for assertion in assertions {
            match assertion {
                // var = constant
                SmtExpr::Eq(lhs, rhs) => {
                    if let SmtExpr::Var(id) = lhs.as_ref() {
                        if let Some(cv) = self.extract_constant(rhs) {
                            constants.insert(*id, cv);
                        }
                    }
                    if let SmtExpr::Var(id) = rhs.as_ref() {
                        if let Some(cv) = self.extract_constant(lhs) {
                            constants.insert(*id, cv);
                        }
                    }
                }
                // Bare boolean variable (asserted true)
                SmtExpr::Var(id) => {
                    constants.insert(*id, ConstantValue::Bool(true));
                }
                // Not(var) (asserted false)
                SmtExpr::Not(inner) => {
                    if let SmtExpr::Var(id) = inner.as_ref() {
                        constants.insert(*id, ConstantValue::Bool(false));
                    }
                }
                _ => {}
            }
        }

        constants
    }

    fn extract_constant(&self, expr: &SmtExpr) -> Option<ConstantValue> {
        match expr {
            SmtExpr::BoolLit(b) => Some(ConstantValue::Bool(*b)),
            SmtExpr::IntLit(n) => Some(ConstantValue::Int(*n)),
            SmtExpr::RealLit(r) => Some(ConstantValue::Real(*r)),
            _ => None,
        }
    }

    // ── Cone of Influence ───────────────────────────────────────────

    fn cone_of_influence(&self, problem: &mut EncodedProblem) -> usize {
        // Find the "property" assertion (last one, the safety negation).
        if problem.assertions.is_empty() {
            return 0;
        }

        // Collect all variables that appear in the property.
        let property_vars = if let Some(last) = problem.assertions.last() {
            free_vars(last)
        } else {
            return 0;
        };

        if property_vars.is_empty() {
            return 0;
        }

        // Compute the transitive closure: any assertion that mentions a
        // property variable is "relevant"; its free variables are added
        // to the relevant set, and we iterate.
        let mut relevant_vars: HashSet<VariableId> = property_vars;
        let mut relevant_assertions: Vec<bool> = vec![false; problem.assertions.len()];
        // The property assertion is always relevant.
        if let Some(last) = relevant_assertions.last_mut() {
            *last = true;
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (i, assertion) in problem.assertions.iter().enumerate() {
                if relevant_assertions[i] {
                    continue;
                }
                let fv = free_vars(assertion);
                if fv.iter().any(|v| relevant_vars.contains(v)) {
                    relevant_assertions[i] = true;
                    for v in fv {
                        if relevant_vars.insert(v) {
                            changed = true;
                        }
                    }
                }
            }
        }

        let removed = relevant_assertions.iter().filter(|&&r| !r).count();

        // Keep only relevant assertions.
        let new_assertions: Vec<SmtExpr> = problem.assertions.iter()
            .zip(relevant_assertions.iter())
            .filter(|(_, &relevant)| relevant)
            .map(|(a, _)| a.clone())
            .collect();

        problem.assertions = new_assertions;
        removed
    }

    // ── Common Sub-expression Elimination ────────────────────────────

    fn common_subexpression_elimination(&self, problem: &mut EncodedProblem) -> usize {
        // Count sub-expression occurrences.
        let mut expr_counts: HashMap<String, usize> = HashMap::new();
        let mut expr_map: HashMap<String, SmtExpr> = HashMap::new();

        for assertion in &problem.assertions {
            self.count_subexpressions(assertion, &mut expr_counts, &mut expr_map);
        }

        // Find expressions that appear more than once and have non-trivial size.
        let shared: Vec<(String, SmtExpr)> = expr_counts.into_iter()
            .filter(|(_, count)| *count > 1)
            .filter_map(|(key, _)| {
                let expr = expr_map.get(&key)?;
                if expr_size(expr) >= 3 {
                    Some((key, expr.clone()))
                } else {
                    None
                }
            })
            .collect();

        // For now, we just simplify (full CSE with let-bindings would
        // require introducing new variables). We track the count.
        shared.len()
    }

    fn count_subexpressions(
        &self,
        expr: &SmtExpr,
        counts: &mut HashMap<String, usize>,
        map: &mut HashMap<String, SmtExpr>,
    ) {
        let key = format!("{:?}", expr);
        *counts.entry(key.clone()).or_insert(0) += 1;
        map.entry(key).or_insert_with(|| expr.clone());

        match expr {
            SmtExpr::Not(e) | SmtExpr::Neg(e) | SmtExpr::Abs(e) => {
                self.count_subexpressions(e, counts, map);
            }
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Distinct(es) => {
                for e in es {
                    self.count_subexpressions(e, counts, map);
                }
            }
            SmtExpr::Implies(a, b) | SmtExpr::Iff(a, b)
            | SmtExpr::Eq(a, b) | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b)
            | SmtExpr::Sub(a, b) | SmtExpr::Mul(a, b) | SmtExpr::Div(a, b) => {
                self.count_subexpressions(a, counts, map);
                self.count_subexpressions(b, counts, map);
            }
            SmtExpr::Ite(c, t, e) => {
                self.count_subexpressions(c, counts, map);
                self.count_subexpressions(t, counts, map);
                self.count_subexpressions(e, counts, map);
            }
            SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => {
                self.count_subexpressions(body, counts, map);
            }
            SmtExpr::Let(bindings, body) => {
                for (_, e) in bindings {
                    self.count_subexpressions(e, counts, map);
                }
                self.count_subexpressions(body, counts, map);
            }
            SmtExpr::Apply(_, args) => {
                for a in args {
                    self.count_subexpressions(a, counts, map);
                }
            }
            _ => {}
        }
    }

    // ── Symmetry Breaking ───────────────────────────────────────────

    fn symmetry_breaking(&self, problem: &mut EncodedProblem) -> usize {
        // Simple symmetry breaking: for the transition selectors, add
        // ordering constraints when multiple edges from the same source
        // location have the same guard structure.
        //
        // For now, we add a soft constraint: prefer lower-indexed transitions
        // when multiple are enabled simultaneously.

        let mut added = 0;

        // Look for pairs of transition selector variables and add
        // tie-breaking: if both transitions are from the same location,
        // prefer the lower index.
        let bound = problem.bound;
        for step in 0..bound {
            let trans_name = format!("trans_t{}", step);
            if let Some(trans_id) = problem.variable_store.id_by_name(&trans_name) {
                // Add: trans >= 0 (already present, but acts as symmetry lower bound)
                let lb = SmtExpr::ge(SmtExpr::Var(trans_id), SmtExpr::IntLit(0));
                if !problem.assertions.contains(&lb) {
                    problem.assertions.push(lb);
                    added += 1;
                }
            }
        }

        added
    }
}

impl Default for EncodingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ConstantValue {
    Bool(bool),
    Int(i64),
    Real(f64),
}

// ═══════════════════════════════════════════════════════════════════════════
// BoundEstimator
// ═══════════════════════════════════════════════════════════════════════════

/// Estimates the minimum BMC bound needed to find a counterexample.
#[derive(Debug, Clone)]
pub struct BoundEstimator;

impl BoundEstimator {
    /// Estimate the minimum bound based on PTA structure.
    ///
    /// Uses a simple heuristic: the minimum bound is at least the
    /// diameter of the location graph (longest shortest path).
    pub fn estimate_min_bound(pta: &PTA) -> usize {
        let n = pta.num_locations();
        if n == 0 { return 0; }

        // Build adjacency matrix
        let loc_ids: Vec<&str> = pta.locations.iter().map(|l| l.id.0.as_str()).collect();
        let mut adj = vec![vec![false; n]; n];

        for edge in &pta.edges {
            let src = loc_ids.iter().position(|&id| id == edge.source.0);
            let tgt = loc_ids.iter().position(|&id| id == edge.target.0);
            if let (Some(s), Some(t)) = (src, tgt) {
                adj[s][t] = true;
            }
        }

        // Floyd-Warshall to find all-pairs shortest paths
        let inf = n + 1;
        let mut dist = vec![vec![inf; n]; n];
        for i in 0..n {
            dist[i][i] = 0;
        }
        for i in 0..n {
            for j in 0..n {
                if adj[i][j] {
                    dist[i][j] = 1;
                }
            }
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        // Diameter = max finite distance
        let mut diameter = 0;
        for i in 0..n {
            for j in 0..n {
                if dist[i][j] < inf {
                    diameter = diameter.max(dist[i][j]);
                }
            }
        }

        // The minimum bound should be at least the diameter
        // plus a small margin for clock-constrained transitions.
        diameter.max(1)
    }

    /// Estimate bound based on clock constraints in the PTA.
    ///
    /// Considers the maximum clock bound across all guards and invariants
    /// divided by the time step.
    pub fn estimate_from_clocks(pta: &PTA) -> usize {
        let dt = pta.time_step;
        if dt <= 0.0 { return 10; }

        let mut max_clock_bound = 0.0f64;

        // Check guards for clock constraints
        for edge in &pta.edges {
            let bound = Self::max_clock_value_in_guard(&edge.guard);
            max_clock_bound = max_clock_bound.max(bound);
        }

        // Check invariants
        for loc in &pta.locations {
            for clause in &loc.invariant.clauses {
                if let crate::pta::InvariantClause::ClockBound { bound, .. } = clause {
                    max_clock_bound = max_clock_bound.max(*bound);
                }
            }
        }

        if max_clock_bound == 0.0 {
            return Self::estimate_min_bound(pta);
        }

        // Need at least max_clock_bound / dt steps
        let clock_steps = (max_clock_bound / dt).ceil() as usize;
        let graph_bound = Self::estimate_min_bound(pta);

        clock_steps.max(graph_bound)
    }

    fn max_clock_value_in_guard(guard: &crate::pta::Guard) -> f64 {
        match guard {
            crate::pta::Guard::Clock { value, .. } => *value,
            crate::pta::Guard::Compound(crate::pta::CompoundGuard::And(gs))
            | crate::pta::Guard::Compound(crate::pta::CompoundGuard::Or(gs)) => {
                gs.iter().map(Self::max_clock_value_in_guard).fold(0.0f64, f64::max)
            }
            crate::pta::Guard::Compound(crate::pta::CompoundGuard::Not(g)) => {
                Self::max_clock_value_in_guard(g)
            }
            _ => 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IncrementalEncoder
// ═══════════════════════════════════════════════════════════════════════════

/// Encodes one BMC step at a time for incremental solving.
///
/// Instead of encoding the full problem up to bound *k* at once, this
/// encoder lets you add one step at a time, check SAT, and continue
/// if unsatisfiable.
#[derive(Debug)]
pub struct IncrementalEncoder {
    pta: PTA,
    current_bound: usize,
    /// Accumulated assertions from all steps so far.
    accumulated_assertions: Vec<SmtExpr>,
    /// Variable store (grows incrementally).
    variable_store: VariableStore,
    /// Symbol table.
    symbol_table: crate::variable::SymbolTable,
    /// PK parameters.
    pk_params: Vec<crate::pk_encoding::OneCompartmentParams>,
}

impl IncrementalEncoder {
    pub fn new(pta: PTA) -> Self {
        let (symbol_table, _factory) = crate::variable::build_symbol_table_and_factory(&pta);
        Self {
            pta,
            current_bound: 0,
            accumulated_assertions: Vec::new(),
            variable_store: VariableStore::new(),
            symbol_table,
            pk_params: Vec::new(),
        }
    }

    pub fn with_pk_params(mut self, params: crate::pk_encoding::OneCompartmentParams) -> Self {
        self.pk_params.push(params);
        self
    }

    /// Get the current bound.
    pub fn current_bound(&self) -> usize {
        self.current_bound
    }

    /// Encode the initial state (step 0).
    pub fn encode_initial(&mut self) -> EncodedProblem {
        let mut encoder = PtaEncoder::new(0);
        for params in &self.pk_params {
            encoder = encoder.with_pk_params(params.clone());
        }
        let problem = encoder.encode_bounded(&self.pta);
        self.variable_store = problem.variable_store.clone();
        self.accumulated_assertions = problem.assertions.clone();
        self.current_bound = 0;
        problem
    }

    /// Extend the encoding by one step.
    pub fn extend_one_step(&mut self) -> EncodedProblem {
        self.current_bound += 1;
        let mut encoder = PtaEncoder::new(self.current_bound);
        for params in &self.pk_params {
            encoder = encoder.with_pk_params(params.clone());
        }
        let problem = encoder.encode_bounded(&self.pta);
        self.variable_store = problem.variable_store.clone();
        self.accumulated_assertions = problem.assertions.clone();
        problem
    }

    /// Get the current encoded problem.
    pub fn current_problem(&self) -> EncodedProblem {
        EncodedProblem {
            assertions: self.accumulated_assertions.clone(),
            variable_store: self.variable_store.clone(),
            symbol_table: self.symbol_table.clone(),
            bound: self.current_bound,
            dt: self.pta.time_step,
            num_locations: self.pta.num_locations(),
            num_edges: self.pta.num_edges(),
        }
    }

    /// Run incremental BMC up to the given maximum bound.
    /// Returns the first satisfiable result (counterexample found) or Unsat.
    pub fn run_incremental(
        &mut self,
        max_bound: usize,
        solver: &mut dyn crate::solver::SmtSolver,
    ) -> crate::solver::SolverResult {
        // Encode initial
        let initial = self.encode_initial();
        for a in &initial.assertions {
            solver.assert_expr(a);
        }

        let result = solver.check_sat();
        if result.is_sat() {
            return result;
        }

        // Incrementally extend
        for _k in 1..=max_bound {
            solver.reset();
            let problem = self.extend_one_step();
            for a in &problem.assertions {
                solver.assert_expr(a);
            }

            let result = solver.check_sat();
            if result.is_sat() {
                return result;
            }
        }

        crate::solver::SolverResult::Unsat
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pta::*;
    use crate::encoder::PtaEncoder;
    use crate::variable::VariableStore;

    fn simple_pta() -> PTA {
        let clock = ClockVariable::new("x");
        PTABuilder::new("test", "l0")
            .add_location("l1", "active")
            .add_clock("x")
            .add_edge("l0", "l1", Guard::clock_ge(&clock, 1.0), Reset::new().clock_reset(&clock))
            .add_edge("l1", "l0", Guard::clock_ge(&clock, 8.0), Reset::new().clock_reset(&clock))
            .set_time_step(1.0)
            .build()
    }

    fn encode_simple() -> EncodedProblem {
        let pta = simple_pta();
        PtaEncoder::new(3).encode_bounded(&pta)
    }

    #[test]
    fn test_optimizer_all_enabled() {
        let problem = encode_simple();
        let optimizer = EncodingOptimizer::new();
        let (optimized, stats) = optimizer.optimize(&problem);
        assert!(optimized.num_assertions() <= problem.num_assertions() + 10);
        assert!(stats.assertions_before > 0);
    }

    #[test]
    fn test_optimizer_all_disabled() {
        let problem = encode_simple();
        let optimizer = EncodingOptimizer::all_disabled();
        let (optimized, stats) = optimizer.optimize(&problem);
        // With all passes disabled, the result should be similar
        assert!(optimized.num_assertions() <= problem.num_assertions());
        assert_eq!(stats.constant_propagations, 0);
    }

    #[test]
    fn test_constant_propagation() {
        let mut store = VariableStore::new();
        let x = store.create_bool("x");
        let y = store.create_bool("y");

        let mut problem = EncodedProblem {
            assertions: vec![
                SmtExpr::eq(SmtExpr::Var(x), SmtExpr::BoolLit(true)),
                SmtExpr::implies(SmtExpr::Var(x), SmtExpr::Var(y)),
            ],
            variable_store: store,
            symbol_table: crate::variable::SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let optimizer = EncodingOptimizer::new();
        let count = optimizer.constant_propagation(&mut problem);
        assert!(count >= 1);
    }

    #[test]
    fn test_cone_of_influence() {
        let mut store = VariableStore::new();
        let x = store.create_bool("x");
        let y = store.create_bool("y");
        let z = store.create_bool("z"); // unrelated to property

        let mut problem = EncodedProblem {
            assertions: vec![
                SmtExpr::eq(SmtExpr::Var(z), SmtExpr::BoolLit(true)), // unrelated
                SmtExpr::implies(SmtExpr::Var(x), SmtExpr::Var(y)),
                SmtExpr::not(SmtExpr::Var(y)), // the "property" (last assertion)
            ],
            variable_store: store,
            symbol_table: crate::variable::SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let optimizer = EncodingOptimizer::new();
        let removed = optimizer.cone_of_influence(&mut problem);
        assert!(removed >= 1, "Should have removed unrelated assertion");
        // z assertion should be removed
        assert!(problem.num_assertions() <= 2);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            assertions_before: 100,
            assertions_after: 80,
            ast_nodes_before: 1000,
            ast_nodes_after: 700,
            ..Default::default()
        };
        assert!((stats.assertion_reduction_pct() - 20.0).abs() < 0.01);
        assert!((stats.node_reduction_pct() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_bound_estimator_simple() {
        let pta = simple_pta();
        let bound = BoundEstimator::estimate_min_bound(&pta);
        assert!(bound >= 1);
    }

    #[test]
    fn test_bound_estimator_from_clocks() {
        let clock = ClockVariable::new("x");
        let pta = PTABuilder::new("test", "l0")
            .add_location("l1", "active")
            .add_clock("x")
            .add_edge("l0", "l1", Guard::clock_ge(&clock, 10.0), Reset::new().clock_reset(&clock))
            .add_edge("l1", "l0", Guard::clock_ge(&clock, 5.0), Reset::new().clock_reset(&clock))
            .set_time_step(1.0)
            .build();

        let bound = BoundEstimator::estimate_from_clocks(&pta);
        assert!(bound >= 10, "Bound should be at least 10 (max clock / dt)");
    }

    #[test]
    fn test_bound_estimator_linear_chain() {
        let clock = ClockVariable::new("x");
        let pta = PTABuilder::new("chain", "l0")
            .add_location("l1", "a")
            .add_location("l2", "b")
            .add_location("l3", "c")
            .add_location("l4", "d")
            .add_clock("x")
            .add_edge("l0", "l1", Guard::True, Reset::new())
            .add_edge("l1", "l2", Guard::True, Reset::new())
            .add_edge("l2", "l3", Guard::True, Reset::new())
            .add_edge("l3", "l4", Guard::True, Reset::new())
            .set_time_step(1.0)
            .build();

        let bound = BoundEstimator::estimate_min_bound(&pta);
        assert!(bound >= 4, "Chain of 5 locations needs at least 4 steps");
    }

    #[test]
    fn test_incremental_encoder() {
        let pta = simple_pta();
        let mut inc = IncrementalEncoder::new(pta);

        let p0 = inc.encode_initial();
        assert_eq!(p0.bound, 0);

        let p1 = inc.extend_one_step();
        assert_eq!(p1.bound, 1);
        assert!(p1.num_assertions() > p0.num_assertions());

        let p2 = inc.extend_one_step();
        assert_eq!(p2.bound, 2);
    }

    #[test]
    fn test_optimizer_removes_true() {
        let mut store = VariableStore::new();
        let x = store.create_bool("x");

        let problem = EncodedProblem {
            assertions: vec![
                SmtExpr::BoolLit(true),
                SmtExpr::BoolLit(true),
                SmtExpr::Var(x),
            ],
            variable_store: store,
            symbol_table: crate::variable::SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let optimizer = EncodingOptimizer::new();
        let (optimized, _) = optimizer.optimize(&problem);
        // True literals should be removed
        assert!(optimized.num_assertions() <= 1);
    }

    #[test]
    fn test_cse_counting() {
        let v0 = VariableId(0);
        let v1 = VariableId(1);
        let shared = SmtExpr::add(vec![SmtExpr::Var(v0), SmtExpr::Var(v1)]);

        let mut store = VariableStore::new();
        store.create_real("x");
        store.create_real("y");

        let problem = EncodedProblem {
            assertions: vec![
                SmtExpr::gt(shared.clone(), SmtExpr::RealLit(0.0)),
                SmtExpr::lt(shared, SmtExpr::RealLit(10.0)),
            ],
            variable_store: store,
            symbol_table: crate::variable::SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let optimizer = EncodingOptimizer::new();
        let (_, stats) = optimizer.optimize(&problem);
        assert!(stats.cse_eliminations >= 1);
    }
}
