//! Constraint normalization and management for the XR accessibility verifier.
//!
//! Provides [`BoundedVariable`] for variables with interval bounds,
//! [`ConstraintSet`] for collecting and scoping SMT constraints,
//! [`ConstraintNormalizer`] for canonical-form transformations,
//! [`BoundPropagator`] for tightening bounds via constraint propagation,
//! and [`VariableScope`] for nested lexical scoping of declarations.

use serde::{Deserialize, Serialize};

use indexmap::IndexMap;
use xr_types::VerifierError;

use crate::expr::{SmtExpr, SmtSort};

// ---------------------------------------------------------------------------
// BoundedVariable
// ---------------------------------------------------------------------------

/// A typed variable with optional finite lower/upper bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedVariable {
    /// Variable name (must match the corresponding SMT declaration).
    pub name: String,
    /// Inclusive lower bound (defaults to −∞).
    pub lower: f64,
    /// Inclusive upper bound (defaults to +∞).
    pub upper: f64,
    /// SMT sort of the variable.
    pub sort: SmtSort,
}

impl BoundedVariable {
    /// Create an unbounded variable with the given name and sort.
    pub fn new(name: impl Into<String>, sort: SmtSort) -> Self {
        Self {
            name: name.into(),
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            sort,
        }
    }

    /// Builder helper — set both bounds at once.
    pub fn with_bounds(mut self, lo: f64, hi: f64) -> Self {
        self.lower = lo;
        self.upper = hi;
        self
    }

    /// Try to raise the lower bound. Returns `true` if the bound actually changed.
    pub fn tighten_lower(&mut self, lo: f64) -> bool {
        if lo > self.lower {
            self.lower = lo;
            true
        } else {
            false
        }
    }

    /// Try to lower the upper bound. Returns `true` if the bound actually changed.
    pub fn tighten_upper(&mut self, hi: f64) -> bool {
        if hi < self.upper {
            self.upper = hi;
            true
        } else {
            false
        }
    }

    /// Returns `true` when the interval is non-empty (lower ≤ upper).
    pub fn is_feasible(&self) -> bool {
        self.lower <= self.upper
    }

    /// Width of the bounding interval.
    pub fn range(&self) -> f64 {
        self.upper - self.lower
    }

    /// Midpoint of the interval, handling infinite bounds gracefully.
    pub fn midpoint(&self) -> f64 {
        match (self.lower.is_finite(), self.upper.is_finite()) {
            (true, true) => (self.lower + self.upper) / 2.0,
            (true, false) => self.lower + 1.0,
            (false, true) => self.upper - 1.0,
            (false, false) => 0.0,
        }
    }

    /// Check whether `val` lies within the closed interval [lower, upper].
    pub fn contains(&self, val: f64) -> bool {
        val >= self.lower && val <= self.upper
    }

    /// Both bounds are finite.
    pub fn is_bounded(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite()
    }

    /// Generate SMT constraints encoding the variable's current bounds.
    ///
    /// Produces `(>= name lower)` and/or `(<= name upper)` when each bound
    /// is finite.
    pub fn to_smt_constraints(&self) -> Vec<SmtExpr> {
        let var = SmtExpr::var(&self.name);
        let mut out = Vec::new();
        if self.lower.is_finite() {
            out.push(SmtExpr::ge(var.clone(), SmtExpr::real(self.lower)));
        }
        if self.upper.is_finite() {
            out.push(SmtExpr::le(var, SmtExpr::real(self.upper)));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ConstraintSet
// ---------------------------------------------------------------------------

/// A collection of SMT constraints together with typed, bounded variables and
/// a push/pop scope stack for incremental solving.
#[derive(Debug, Clone)]
pub struct ConstraintSet {
    /// Named variables with their current bound information.
    variables: IndexMap<String, BoundedVariable>,
    /// Accumulated SMT constraint expressions.
    constraints: Vec<SmtExpr>,
    /// Stack of constraint-count snapshots for push/pop scoping.
    scope_stack: Vec<usize>,
}

impl ConstraintSet {
    /// Create an empty constraint set.
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
            constraints: Vec::new(),
            scope_stack: Vec::new(),
        }
    }

    /// Register a variable (overwrites any previous entry with the same name).
    pub fn add_variable(&mut self, var: BoundedVariable) {
        self.variables.insert(var.name.clone(), var);
    }

    /// Look up a variable by name.
    pub fn get_variable(&self, name: &str) -> Option<&BoundedVariable> {
        self.variables.get(name)
    }

    /// Look up a variable by name (mutable).
    pub fn get_variable_mut(&mut self, name: &str) -> Option<&mut BoundedVariable> {
        self.variables.get_mut(name)
    }

    /// Append a constraint expression.
    pub fn add_constraint(&mut self, constraint: SmtExpr) {
        self.constraints.push(constraint);
    }

    /// Tighten the bounds of an existing variable and add corresponding SMT
    /// constraints. Returns an error if the variable has not been declared.
    pub fn add_bound_constraint(
        &mut self,
        var_name: &str,
        lo: f64,
        hi: f64,
    ) -> Result<(), VerifierError> {
        let var = self
            .variables
            .get_mut(var_name)
            .ok_or_else(|| VerifierError::SmtEncoding(format!("unknown variable: {var_name}")))?;

        var.tighten_lower(lo);
        var.tighten_upper(hi);

        let v = SmtExpr::var(var_name);
        if lo.is_finite() {
            self.constraints
                .push(SmtExpr::ge(v.clone(), SmtExpr::real(lo)));
        }
        if hi.is_finite() {
            self.constraints.push(SmtExpr::le(v, SmtExpr::real(hi)));
        }
        Ok(())
    }

    /// Conjunction (AND) of every constraint in the set.
    pub fn conjunction(&self) -> SmtExpr {
        if self.constraints.is_empty() {
            return SmtExpr::BoolConst(true);
        }
        if self.constraints.len() == 1 {
            return self.constraints[0].clone();
        }
        SmtExpr::And(self.constraints.clone())
    }

    /// Disjunction of this set's conjunction with another set's conjunction.
    pub fn disjunction_with(&self, other: &ConstraintSet) -> SmtExpr {
        SmtExpr::Or(vec![self.conjunction(), other.conjunction()])
    }

    /// Save the current constraint count so it can be restored later.
    pub fn push(&mut self) {
        self.scope_stack.push(self.constraints.len());
    }

    /// Restore the constraint list to the most recently pushed snapshot.
    pub fn pop(&mut self) {
        if let Some(n) = self.scope_stack.pop() {
            self.constraints.truncate(n);
        }
    }

    /// Number of registered variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of accumulated constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns `true` when there are no constraints at all.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Iterate over all registered variables.
    pub fn variables(&self) -> impl Iterator<Item = &BoundedVariable> {
        self.variables.values()
    }

    /// Slice view of all constraints.
    pub fn constraints(&self) -> &[SmtExpr] {
        &self.constraints
    }

    /// Returns `true` when every registered variable has finite bounds.
    pub fn all_variables_bounded(&self) -> bool {
        self.variables.values().all(|v| v.is_bounded())
    }

    /// Simplify constraints, remove tautologies, and extract simple variable
    /// bounds from inequality constraints.
    pub fn normalize(&mut self) {
        // 1. Simplify each constraint and drop boolean-true tautologies.
        let simplified: Vec<SmtExpr> = self
            .constraints
            .iter()
            .map(|c| ConstraintNormalizer::normalize_constraint(c))
            .filter(|c| *c != SmtExpr::BoolConst(true))
            .collect();

        self.constraints = simplified;

        // 2. Extract simple bounds and tighten variables.
        let mut extracted: Vec<(String, f64, f64)> = Vec::new();
        for c in &self.constraints {
            extracted.extend(ConstraintNormalizer::extract_variable_bounds(c));
        }
        for (name, lo, hi) in extracted {
            if let Some(var) = self.variables.get_mut(&name) {
                var.tighten_lower(lo);
                var.tighten_upper(hi);
            }
        }
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConstraintNormalizer
// ---------------------------------------------------------------------------

/// Stateless utilities for rewriting constraints into canonical form.
pub struct ConstraintNormalizer;

impl ConstraintNormalizer {
    /// Simplify an expression and rewrite into canonical form:
    /// - `Ge(a, b)` → `Le(b, a)`
    /// - `Gt(a, b)` → `Lt(b, a)`
    /// - `Eq(a, b)` → `And(Le(a, b), Le(b, a))`
    pub fn normalize_constraint(expr: &SmtExpr) -> SmtExpr {
        let simplified = expr.simplify();
        Self::rewrite(&simplified)
    }

    /// Recursive rewrite pass that converts Ge→Le, Gt→Lt, Eq→And(Le,Le).
    fn rewrite(expr: &SmtExpr) -> SmtExpr {
        match expr {
            SmtExpr::Ge(a, b) => SmtExpr::le(Self::rewrite(b), Self::rewrite(a)),
            SmtExpr::Gt(a, b) => SmtExpr::lt(Self::rewrite(b), Self::rewrite(a)),
            SmtExpr::Eq(a, b) => {
                let ra = Self::rewrite(a);
                let rb = Self::rewrite(b);
                SmtExpr::And(vec![
                    SmtExpr::le(ra.clone(), rb.clone()),
                    SmtExpr::le(rb, ra),
                ])
            }
            SmtExpr::And(children) => {
                SmtExpr::And(children.iter().map(|c| Self::rewrite(c)).collect())
            }
            SmtExpr::Or(children) => {
                SmtExpr::Or(children.iter().map(|c| Self::rewrite(c)).collect())
            }
            SmtExpr::Not(inner) => SmtExpr::not(Self::rewrite(inner)),
            SmtExpr::Le(a, b) => SmtExpr::le(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Lt(a, b) => SmtExpr::lt(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Add(a, b) => SmtExpr::add(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Sub(a, b) => SmtExpr::sub(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Mul(a, b) => SmtExpr::mul(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Div(a, b) => SmtExpr::div(Self::rewrite(a), Self::rewrite(b)),
            SmtExpr::Neg(a) => SmtExpr::neg(Self::rewrite(a)),
            SmtExpr::Ite(c, t, e) => {
                SmtExpr::ite(Self::rewrite(c), Self::rewrite(t), Self::rewrite(e))
            }
            SmtExpr::Let(name, val, body) => {
                SmtExpr::let_bind(name.clone(), Self::rewrite(val), Self::rewrite(body))
            }
            other => other.clone(),
        }
    }

    /// Extract simple variable bounds from a constraint expression.
    ///
    /// Recognises patterns of the form:
    /// - `Le(Var(x), Const(c))` → (x, −∞, c)
    /// - `Le(Const(c), Var(x))` → (x, c, +∞)
    /// - `Lt(Var(x), Const(c))` → (x, −∞, c)  (slightly loose but safe)
    /// - `Lt(Const(c), Var(x))` → (x, c, +∞)
    /// - `Ge(Var(x), Const(c))` → (x, c, +∞)
    /// - `Ge(Const(c), Var(x))` → (x, −∞, c)
    /// - `Gt(Var(x), Const(c))` → (x, c, +∞)
    /// - `Gt(Const(c), Var(x))` → (x, −∞, c)
    /// - `And(...)` — recurse into children.
    pub fn extract_variable_bounds(expr: &SmtExpr) -> Vec<(String, f64, f64)> {
        let mut results = Vec::new();
        Self::extract_bounds_inner(expr, &mut results);
        results
    }

    fn extract_bounds_inner(expr: &SmtExpr, out: &mut Vec<(String, f64, f64)>) {
        match expr {
            // x <= c  or  x < c
            SmtExpr::Le(a, b) | SmtExpr::Lt(a, b) => {
                if let (SmtExpr::Var(name), SmtExpr::Const(c)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), f64::NEG_INFINITY, *c));
                }
                if let (SmtExpr::Const(c), SmtExpr::Var(name)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), *c, f64::INFINITY));
                }
            }
            // x >= c  or  x > c
            SmtExpr::Ge(a, b) | SmtExpr::Gt(a, b) => {
                if let (SmtExpr::Var(name), SmtExpr::Const(c)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), *c, f64::INFINITY));
                }
                if let (SmtExpr::Const(c), SmtExpr::Var(name)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), f64::NEG_INFINITY, *c));
                }
            }
            // x == c  →  both bounds
            SmtExpr::Eq(a, b) => {
                if let (SmtExpr::Var(name), SmtExpr::Const(c)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), *c, *c));
                }
                if let (SmtExpr::Const(c), SmtExpr::Var(name)) = (a.as_ref(), b.as_ref()) {
                    out.push((name.clone(), *c, *c));
                }
            }
            SmtExpr::And(children) => {
                for child in children {
                    Self::extract_bounds_inner(child, out);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// BoundPropagator
// ---------------------------------------------------------------------------

/// Result of bound propagation.
#[derive(Debug, Clone)]
pub enum PropagationResult {
    /// Propagation finished and all variables remain feasible.
    Feasible {
        /// Number of propagation iterations performed.
        iterations: usize,
        /// Total number of bound tightenings applied.
        tightened: usize,
    },
    /// A variable's bounds became contradictory (lower > upper).
    Infeasible {
        /// Name of the first variable found infeasible.
        variable: String,
    },
    /// Bounds converged to a fixed point without further change.
    FixedPoint {
        /// Number of iterations to reach the fixed point.
        iterations: usize,
    },
}

/// Iterative bound propagator that tightens variable intervals using the
/// constraints in a [`ConstraintSet`].
pub struct BoundPropagator {
    /// Maximum number of propagation rounds.
    pub max_iterations: usize,
    /// Absolute tolerance for considering a bound change significant.
    pub tolerance: f64,
}

impl BoundPropagator {
    /// Create a propagator with default settings (100 iterations, 1e-10
    /// tolerance).
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }

    /// Run bound propagation on a constraint set.
    ///
    /// Repeatedly scans every constraint, attempting to tighten variable
    /// bounds. Stops when a fixed point is reached, a variable becomes
    /// infeasible, or the iteration limit is hit.
    pub fn propagate(
        &self,
        constraint_set: &mut ConstraintSet,
    ) -> Result<PropagationResult, VerifierError> {
        let mut total_tightened: usize = 0;

        for iteration in 0..self.max_iterations {
            let mut changed = false;

            // Snapshot constraint vec so we can iterate while mutating variables.
            let constraints: Vec<SmtExpr> = constraint_set.constraints.clone();
            for c in &constraints {
                if self.propagate_single(c, &mut constraint_set.variables) {
                    changed = true;
                    total_tightened += 1;
                }
            }

            // Check feasibility after each round.
            for var in constraint_set.variables.values() {
                if !var.is_feasible() {
                    return Ok(PropagationResult::Infeasible {
                        variable: var.name.clone(),
                    });
                }
            }

            if !changed {
                return Ok(PropagationResult::FixedPoint {
                    iterations: iteration + 1,
                });
            }
        }

        Ok(PropagationResult::Feasible {
            iterations: self.max_iterations,
            tightened: total_tightened,
        })
    }

    /// Attempt to tighten variable bounds from a single constraint.
    ///
    /// Returns `true` if any bound was tightened by more than `self.tolerance`.
    pub fn propagate_single(
        &self,
        expr: &SmtExpr,
        variables: &mut IndexMap<String, BoundedVariable>,
    ) -> bool {
        let bounds = ConstraintNormalizer::extract_variable_bounds(expr);
        let mut any_changed = false;

        for (name, lo, hi) in bounds {
            if let Some(var) = variables.get_mut(&name) {
                if lo.is_finite() && lo > var.lower + self.tolerance {
                    var.lower = lo;
                    any_changed = true;
                }
                if hi.is_finite() && hi < var.upper - self.tolerance {
                    var.upper = hi;
                    any_changed = true;
                }
            }
        }

        any_changed
    }
}

impl Default for BoundPropagator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VariableScope
// ---------------------------------------------------------------------------

/// Lexical scope stack for variable declarations.
///
/// Inner scopes shadow outer ones; [`lookup`](VariableScope::lookup) searches
/// from the innermost scope outward.
pub struct VariableScope {
    scopes: Vec<IndexMap<String, BoundedVariable>>,
}

impl VariableScope {
    /// Create a scope manager with a single (global) scope.
    pub fn new() -> Self {
        Self {
            scopes: vec![IndexMap::new()],
        }
    }

    /// Push a new empty scope onto the stack.
    pub fn push_scope(&mut self) {
        self.scopes.push(IndexMap::new());
    }

    /// Pop the innermost scope. Returns `None` if only the global scope
    /// remains (the global scope is never popped).
    pub fn pop_scope(&mut self) -> Option<IndexMap<String, BoundedVariable>> {
        if self.scopes.len() > 1 {
            self.scopes.pop()
        } else {
            None
        }
    }

    /// Declare a variable in the current (innermost) scope.
    pub fn declare(&mut self, var: BoundedVariable) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(var.name.clone(), var);
        }
    }

    /// Look up a variable by name, searching from the innermost scope outward.
    pub fn lookup(&self, name: &str) -> Option<&BoundedVariable> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }
}

impl Default for VariableScope {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- BoundedVariable tightening -----------------------------------------

    #[test]
    fn test_bounded_variable_tightening() {
        let mut v = BoundedVariable::new("x", SmtSort::Real).with_bounds(-10.0, 10.0);

        assert!(v.is_feasible());
        assert!(v.is_bounded());
        assert_eq!(v.range(), 20.0);
        assert!((v.midpoint() - 0.0).abs() < 1e-12);
        assert!(v.contains(0.0));
        assert!(!v.contains(11.0));

        // Tighten lower
        assert!(v.tighten_lower(-5.0));
        assert_eq!(v.lower, -5.0);
        // Attempting to loosen should not change
        assert!(!v.tighten_lower(-8.0));
        assert_eq!(v.lower, -5.0);

        // Tighten upper
        assert!(v.tighten_upper(3.0));
        assert_eq!(v.upper, 3.0);
        assert!(!v.tighten_upper(7.0));
        assert_eq!(v.upper, 3.0);

        assert!(v.is_feasible());
        assert_eq!(v.range(), 8.0);

        // Tighten until infeasible
        v.tighten_lower(5.0);
        assert!(!v.is_feasible());
    }

    // -- ConstraintSet push / pop -------------------------------------------

    #[test]
    fn test_constraint_set_push_pop() {
        let mut cs = ConstraintSet::new();
        cs.add_variable(BoundedVariable::new("x", SmtSort::Real));

        cs.add_constraint(SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(10.0)));
        assert_eq!(cs.num_constraints(), 1);

        cs.push();
        cs.add_constraint(SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(0.0)));
        assert_eq!(cs.num_constraints(), 2);

        cs.push();
        cs.add_constraint(SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)));
        assert_eq!(cs.num_constraints(), 3);

        cs.pop();
        assert_eq!(cs.num_constraints(), 2);

        cs.pop();
        assert_eq!(cs.num_constraints(), 1);

        // Popping with nothing on the stack is harmless.
        cs.pop();
        assert_eq!(cs.num_constraints(), 1);
    }

    // -- Constraint normalization -------------------------------------------

    #[test]
    fn test_constraint_normalization() {
        // Ge(x, 2) should become Le(2, x)
        let expr = SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(2.0));
        let norm = ConstraintNormalizer::normalize_constraint(&expr);
        assert_eq!(norm, SmtExpr::le(SmtExpr::real(2.0), SmtExpr::var("x")));

        // Gt(x, 3) should become Lt(3, x)
        let expr = SmtExpr::gt(SmtExpr::var("x"), SmtExpr::real(3.0));
        let norm = ConstraintNormalizer::normalize_constraint(&expr);
        assert_eq!(norm, SmtExpr::lt(SmtExpr::real(3.0), SmtExpr::var("x")));

        // Eq(x, 5) should become And(Le(x,5), Le(5,x))
        let expr = SmtExpr::eq(SmtExpr::var("x"), SmtExpr::real(5.0));
        let norm = ConstraintNormalizer::normalize_constraint(&expr);
        assert_eq!(
            norm,
            SmtExpr::And(vec![
                SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)),
                SmtExpr::le(SmtExpr::real(5.0), SmtExpr::var("x")),
            ])
        );
    }

    // -- Bound extraction ---------------------------------------------------

    #[test]
    fn test_bound_extraction() {
        // x <= 5
        let expr = SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0));
        let bounds = ConstraintNormalizer::extract_variable_bounds(&expr);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].0, "x");
        assert!(bounds[0].1.is_infinite() && bounds[0].1 < 0.0);
        assert_eq!(bounds[0].2, 5.0);

        // 2 <= x  (equivalent to x >= 2)
        let expr = SmtExpr::le(SmtExpr::real(2.0), SmtExpr::var("x"));
        let bounds = ConstraintNormalizer::extract_variable_bounds(&expr);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].0, "x");
        assert_eq!(bounds[0].1, 2.0);
        assert!(bounds[0].2.is_infinite() && bounds[0].2 > 0.0);

        // And(x >= 0, x <= 10) → two bounds for x
        let expr = SmtExpr::And(vec![
            SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(0.0)),
            SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(10.0)),
        ]);
        let bounds = ConstraintNormalizer::extract_variable_bounds(&expr);
        assert_eq!(bounds.len(), 2);
    }

    // -- Simple bound propagation -------------------------------------------

    #[test]
    fn test_bound_propagation_simple() {
        let mut cs = ConstraintSet::new();
        cs.add_variable(BoundedVariable::new("x", SmtSort::Real));

        // x <= 5
        cs.add_constraint(SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)));
        // x >= -3
        cs.add_constraint(SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(-3.0)));

        let prop = BoundPropagator::new();
        let result = prop.propagate(&mut cs).unwrap();

        // Should reach a fixed point.
        match &result {
            PropagationResult::FixedPoint { .. } => {}
            PropagationResult::Feasible { .. } => {}
            other => panic!("unexpected result: {other:?}"),
        }

        let v = cs.get_variable("x").unwrap();
        assert!(v.lower >= -3.0 - 1e-9);
        assert!(v.upper <= 5.0 + 1e-9);
    }

    // -- Infeasible propagation ---------------------------------------------

    #[test]
    fn test_bound_propagation_infeasible() {
        let mut cs = ConstraintSet::new();
        cs.add_variable(BoundedVariable::new("y", SmtSort::Real));

        // y >= 10
        cs.add_constraint(SmtExpr::ge(SmtExpr::var("y"), SmtExpr::real(10.0)));
        // y <= 5   — contradicts the above
        cs.add_constraint(SmtExpr::le(SmtExpr::var("y"), SmtExpr::real(5.0)));

        let prop = BoundPropagator::new();
        let result = prop.propagate(&mut cs).unwrap();

        match result {
            PropagationResult::Infeasible { variable } => {
                assert_eq!(variable, "y");
            }
            other => panic!("expected Infeasible, got {other:?}"),
        }
    }

    // -- Variable scoping ---------------------------------------------------

    #[test]
    fn test_variable_scope() {
        let mut scope = VariableScope::new();

        // Declare in global scope
        scope.declare(BoundedVariable::new("a", SmtSort::Real).with_bounds(0.0, 1.0));
        assert!(scope.lookup("a").is_some());

        // Push inner scope and shadow "a"
        scope.push_scope();
        scope.declare(BoundedVariable::new("a", SmtSort::Real).with_bounds(2.0, 3.0));
        scope.declare(BoundedVariable::new("b", SmtSort::Real).with_bounds(4.0, 5.0));

        let a = scope.lookup("a").unwrap();
        assert_eq!(a.lower, 2.0); // inner shadow

        // Pop inner scope — "a" reverts, "b" gone
        scope.pop_scope();
        let a = scope.lookup("a").unwrap();
        assert_eq!(a.lower, 0.0);
        assert!(scope.lookup("b").is_none());

        // Cannot pop the global scope
        assert!(scope.pop_scope().is_none());
    }

    // -- Conjunction generation ---------------------------------------------

    #[test]
    fn test_conjunction_generation() {
        let mut cs = ConstraintSet::new();
        assert_eq!(cs.conjunction(), SmtExpr::BoolConst(true));

        cs.add_constraint(SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(1.0)));
        // Single constraint → no wrapping And
        assert_eq!(
            cs.conjunction(),
            SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(1.0)),
        );

        cs.add_constraint(SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(0.0)));
        match cs.conjunction() {
            SmtExpr::And(children) => assert_eq!(children.len(), 2),
            other => panic!("expected And, got {other:?}"),
        }

        // disjunction_with
        let mut other = ConstraintSet::new();
        other.add_constraint(SmtExpr::le(SmtExpr::var("y"), SmtExpr::real(2.0)));
        match cs.disjunction_with(&other) {
            SmtExpr::Or(children) => assert_eq!(children.len(), 2),
            other => panic!("expected Or, got {other:?}"),
        }
    }
}
