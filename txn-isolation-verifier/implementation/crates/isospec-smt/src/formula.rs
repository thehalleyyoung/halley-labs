//! SMT formula construction helpers.
//!
//! Provides a builder API for constructing complex SMT formulas,
//! quantifier handling, array theory helpers, and simplification passes.

use std::collections::HashMap;

use isospec_types::constraint::{SmtExpr, SmtSort};

// ---------------------------------------------------------------------------
// FormulaBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing complex SMT formulas.
#[derive(Debug, Clone)]
pub struct FormulaBuilder {
    /// Stack of partial expressions being assembled.
    stack: Vec<SmtExpr>,
}

impl FormulaBuilder {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Push a constant reference onto the stack.
    pub fn var(mut self, name: &str) -> Self {
        self.stack.push(SmtExpr::Const(name.to_string()));
        self
    }

    /// Push a boolean literal.
    pub fn bool_lit(mut self, b: bool) -> Self {
        self.stack.push(SmtExpr::BoolLit(b));
        self
    }

    /// Push an integer literal.
    pub fn int_lit(mut self, i: i64) -> Self {
        self.stack.push(SmtExpr::IntLit(i));
        self
    }

    /// Push an arbitrary expression.
    pub fn expr(mut self, e: SmtExpr) -> Self {
        self.stack.push(e);
        self
    }

    /// Pop two expressions and push their conjunction.
    pub fn and2(mut self) -> Self {
        let b = self.stack.pop().unwrap_or(SmtExpr::BoolLit(true));
        let a = self.stack.pop().unwrap_or(SmtExpr::BoolLit(true));
        self.stack.push(SmtExpr::And(vec![a, b]));
        self
    }

    /// Pop two expressions and push their disjunction.
    pub fn or2(mut self) -> Self {
        let b = self.stack.pop().unwrap_or(SmtExpr::BoolLit(false));
        let a = self.stack.pop().unwrap_or(SmtExpr::BoolLit(false));
        self.stack.push(SmtExpr::Or(vec![a, b]));
        self
    }

    /// Pop one expression and push its negation.
    pub fn not(mut self) -> Self {
        let a = self.stack.pop().unwrap_or(SmtExpr::BoolLit(false));
        self.stack.push(SmtExpr::Not(Box::new(a)));
        self
    }

    /// Pop two expressions and push lhs => rhs.
    pub fn implies(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::BoolLit(true));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::BoolLit(true));
        self.stack
            .push(SmtExpr::Implies(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs = rhs.
    pub fn eq(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack.push(SmtExpr::Eq(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs < rhs.
    pub fn lt(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack.push(SmtExpr::Lt(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs <= rhs.
    pub fn le(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack.push(SmtExpr::Le(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs > rhs.
    pub fn gt(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack.push(SmtExpr::Gt(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs >= rhs.
    pub fn ge(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack.push(SmtExpr::Ge(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs + rhs.
    pub fn add(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack
            .push(SmtExpr::Add(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs - rhs.
    pub fn sub(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack
            .push(SmtExpr::Sub(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop two expressions and push lhs * rhs.
    pub fn mul(mut self) -> Self {
        let rhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let lhs = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        self.stack
            .push(SmtExpr::Mul(Box::new(lhs), Box::new(rhs)));
        self
    }

    /// Pop three expressions: condition, then, else and push ite.
    pub fn ite(mut self) -> Self {
        let else_branch = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let then_branch = self.stack.pop().unwrap_or(SmtExpr::IntLit(0));
        let cond = self.stack.pop().unwrap_or(SmtExpr::BoolLit(true));
        self.stack.push(SmtExpr::Ite(
            Box::new(cond),
            Box::new(then_branch),
            Box::new(else_branch),
        ));
        self
    }

    /// Consume N items from the stack, create AND.
    pub fn and_n(mut self, n: usize) -> Self {
        let actual = n.min(self.stack.len());
        let split = self.stack.len() - actual;
        let children: Vec<SmtExpr> = self.stack.drain(split..).collect();
        self.stack.push(SmtExpr::And(children));
        self
    }

    /// Consume N items from the stack, create OR.
    pub fn or_n(mut self, n: usize) -> Self {
        let actual = n.min(self.stack.len());
        let split = self.stack.len() - actual;
        let children: Vec<SmtExpr> = self.stack.drain(split..).collect();
        self.stack.push(SmtExpr::Or(children));
        self
    }

    /// Consume N items from the stack, create DISTINCT.
    pub fn distinct_n(mut self, n: usize) -> Self {
        let actual = n.min(self.stack.len());
        let split = self.stack.len() - actual;
        let children: Vec<SmtExpr> = self.stack.drain(split..).collect();
        self.stack.push(SmtExpr::Distinct(children));
        self
    }

    /// Get the top expression (the result) without consuming.
    pub fn peek(&self) -> Option<&SmtExpr> {
        self.stack.last()
    }

    /// Consume the builder and return the final expression.
    pub fn build(mut self) -> SmtExpr {
        self.stack.pop().unwrap_or(SmtExpr::BoolLit(true))
    }

    /// Return the current stack size.
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
}

impl Default for FormulaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Quantifier helpers
// ---------------------------------------------------------------------------

/// Helper for building quantified formulas.
pub struct QuantifierBuilder;

impl QuantifierBuilder {
    /// Build a forall quantifier.
    pub fn forall(vars: &[(&str, &str)], body: SmtExpr) -> SmtExpr {
        let bindings: Vec<(String, SmtSort)> = vars
            .iter()
            .map(|(n, s)| (n.to_string(), SmtSort::from_name(s)))
            .collect();
        SmtExpr::ForAll(bindings, Box::new(body))
    }

    /// Build an exists quantifier.
    pub fn exists(vars: &[(&str, &str)], body: SmtExpr) -> SmtExpr {
        let bindings: Vec<(String, SmtSort)> = vars
            .iter()
            .map(|(n, s)| (n.to_string(), SmtSort::from_name(s)))
            .collect();
        SmtExpr::Exists(bindings, Box::new(body))
    }

    /// Build a bounded forall: forall x in [lo, hi). body(x)
    pub fn bounded_forall(var: &str, lo: i64, hi: i64, body_fn: impl Fn(SmtExpr) -> SmtExpr) -> SmtExpr {
        let var_expr = SmtExpr::Const(var.to_string());
        let body = body_fn(var_expr.clone());
        let bounds = SmtExpr::And(vec![
            SmtExpr::Ge(Box::new(var_expr.clone()), Box::new(SmtExpr::IntLit(lo))),
            SmtExpr::Lt(Box::new(var_expr), Box::new(SmtExpr::IntLit(hi))),
        ]);
        let guarded = SmtExpr::Implies(Box::new(bounds), Box::new(body));
        Self::forall(&[(var, "Int")], guarded)
    }

    /// Build a bounded exists: exists x in [lo, hi). body(x)
    pub fn bounded_exists(var: &str, lo: i64, hi: i64, body_fn: impl Fn(SmtExpr) -> SmtExpr) -> SmtExpr {
        let var_expr = SmtExpr::Const(var.to_string());
        let body = body_fn(var_expr.clone());
        let bounds = SmtExpr::And(vec![
            SmtExpr::Ge(Box::new(var_expr.clone()), Box::new(SmtExpr::IntLit(lo))),
            SmtExpr::Lt(Box::new(var_expr), Box::new(SmtExpr::IntLit(hi))),
        ]);
        let guarded = SmtExpr::And(vec![bounds, body]);
        Self::exists(&[(var, "Int")], guarded)
    }
}

// ---------------------------------------------------------------------------
// Array theory helpers
// ---------------------------------------------------------------------------

/// Helpers for working with the SMT array theory.
pub struct ArrayHelper;

impl ArrayHelper {
    /// Select a value from an array: (select arr idx).
    pub fn select(arr: SmtExpr, idx: SmtExpr) -> SmtExpr {
        SmtExpr::Select(Box::new(arr), Box::new(idx))
    }

    /// Store a value into an array: (store arr idx val).
    pub fn store(arr: SmtExpr, idx: SmtExpr, val: SmtExpr) -> SmtExpr {
        SmtExpr::Store(Box::new(arr), Box::new(idx), Box::new(val))
    }

    /// Build a constant array by repeated stores.
    pub fn const_array(base: SmtExpr, entries: &[(i64, SmtExpr)]) -> SmtExpr {
        let mut arr = base;
        for (idx, val) in entries {
            arr = Self::store(arr, SmtExpr::IntLit(*idx), val.clone());
        }
        arr
    }

    /// Assert that two arrays are equal on a range of indices.
    pub fn arrays_equal_on_range(
        arr1: &str,
        arr2: &str,
        lo: i64,
        hi: i64,
    ) -> SmtExpr {
        let mut conjuncts = Vec::new();
        for i in lo..hi {
            let sel1 = Self::select(
                SmtExpr::Const(arr1.to_string()),
                SmtExpr::IntLit(i),
            );
            let sel2 = Self::select(
                SmtExpr::Const(arr2.to_string()),
                SmtExpr::IntLit(i),
            );
            conjuncts.push(SmtExpr::Eq(Box::new(sel1), Box::new(sel2)));
        }
        SmtExpr::And(conjuncts)
    }

    /// Build a "memory update" pattern: arr' = store(arr, idx, val),
    /// and all other indices unchanged.
    pub fn point_update(
        arr_name: &str,
        arr_prime_name: &str,
        idx: SmtExpr,
        val: SmtExpr,
        num_indices: i64,
    ) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();

        // arr' at idx = val
        let updated = SmtExpr::Eq(
            Box::new(Self::select(
                SmtExpr::Const(arr_prime_name.to_string()),
                idx.clone(),
            )),
            Box::new(val),
        );
        constraints.push(updated);

        // Frame condition: arr' at other indices = arr at those indices
        for i in 0..num_indices {
            let i_expr = SmtExpr::IntLit(i);
            let not_idx = SmtExpr::Not(Box::new(SmtExpr::Eq(
                Box::new(i_expr.clone()),
                Box::new(idx.clone()),
            )));
            let frame = SmtExpr::Implies(
                Box::new(not_idx),
                Box::new(SmtExpr::Eq(
                    Box::new(Self::select(
                        SmtExpr::Const(arr_prime_name.to_string()),
                        i_expr.clone(),
                    )),
                    Box::new(Self::select(
                        SmtExpr::Const(arr_name.to_string()),
                        i_expr,
                    )),
                )),
            );
            constraints.push(frame);
        }

        constraints
    }
}

// ---------------------------------------------------------------------------
// Simplification passes
// ---------------------------------------------------------------------------

/// Performs syntactic simplification of SMT expressions.
pub struct Simplifier;

impl Simplifier {
    /// Run all simplification passes.
    pub fn simplify(expr: &SmtExpr) -> SmtExpr {
        let mut current = expr.clone();
        // Iterate until fixpoint (max 10 passes to prevent infinite loops)
        for _ in 0..10 {
            let next = Self::simplify_one_pass(&current);
            if next == current {
                break;
            }
            current = next;
        }
        current
    }

    fn simplify_one_pass(expr: &SmtExpr) -> SmtExpr {
        match expr {
            // not(not(x)) => x
            SmtExpr::Not(inner) => {
                let simplified_inner = Self::simplify_one_pass(inner);
                match simplified_inner {
                    SmtExpr::Not(double_inner) => *double_inner,
                    SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
                    other => SmtExpr::Not(Box::new(other)),
                }
            }

            // and(...) with flattening and identity elimination
            SmtExpr::And(children) => {
                let mut simplified: Vec<SmtExpr> = Vec::new();
                for child in children {
                    let s = Self::simplify_one_pass(child);
                    match s {
                        SmtExpr::BoolLit(true) => {} // identity
                        SmtExpr::BoolLit(false) => return SmtExpr::BoolLit(false), // absorbing
                        SmtExpr::And(inner) => simplified.extend(inner), // flatten
                        other => simplified.push(other),
                    }
                }
                match simplified.len() {
                    0 => SmtExpr::BoolLit(true),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::And(simplified),
                }
            }

            // or(...) with flattening and identity elimination
            SmtExpr::Or(children) => {
                let mut simplified: Vec<SmtExpr> = Vec::new();
                for child in children {
                    let s = Self::simplify_one_pass(child);
                    match s {
                        SmtExpr::BoolLit(false) => {} // identity
                        SmtExpr::BoolLit(true) => return SmtExpr::BoolLit(true), // absorbing
                        SmtExpr::Or(inner) => simplified.extend(inner), // flatten
                        other => simplified.push(other),
                    }
                }
                match simplified.len() {
                    0 => SmtExpr::BoolLit(false),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::Or(simplified),
                }
            }

            // implies(true, rhs) => rhs; implies(false, _) => true; implies(_, true) => true
            SmtExpr::Implies(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::BoolLit(true), _) => sr,
                    (SmtExpr::BoolLit(false), _) => SmtExpr::BoolLit(true),
                    (_, SmtExpr::BoolLit(true)) => SmtExpr::BoolLit(true),
                    _ => SmtExpr::Implies(Box::new(sl), Box::new(sr)),
                }
            }

            // eq(x, x) => true (syntactic)
            SmtExpr::Eq(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                if sl == sr {
                    SmtExpr::BoolLit(true)
                } else {
                    SmtExpr::Eq(Box::new(sl), Box::new(sr))
                }
            }

            // ite(true, t, _) => t; ite(false, _, e) => e
            SmtExpr::Ite(cond, t, e) => {
                let sc = Self::simplify_one_pass(cond);
                let st = Self::simplify_one_pass(t);
                let se = Self::simplify_one_pass(e);
                match sc {
                    SmtExpr::BoolLit(true) => st,
                    SmtExpr::BoolLit(false) => se,
                    _ => {
                        if st == se {
                            st
                        } else {
                            SmtExpr::Ite(Box::new(sc), Box::new(st), Box::new(se))
                        }
                    }
                }
            }

            // Arithmetic: x + 0 => x, 0 + x => x
            SmtExpr::Add(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(0), _) => sr,
                    (_, SmtExpr::IntLit(0)) => sl,
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::IntLit(a + b),
                    _ => SmtExpr::Add(Box::new(sl), Box::new(sr)),
                }
            }

            SmtExpr::Sub(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (_, SmtExpr::IntLit(0)) => sl,
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::IntLit(a - b),
                    _ => {
                        if sl == sr {
                            SmtExpr::IntLit(0)
                        } else {
                            SmtExpr::Sub(Box::new(sl), Box::new(sr))
                        }
                    }
                }
            }

            SmtExpr::Mul(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(0), _) | (_, SmtExpr::IntLit(0)) => SmtExpr::IntLit(0),
                    (SmtExpr::IntLit(1), _) => sr,
                    (_, SmtExpr::IntLit(1)) => sl,
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::IntLit(a * b),
                    _ => SmtExpr::Mul(Box::new(sl), Box::new(sr)),
                }
            }

            // Comparisons with constant folding
            SmtExpr::Lt(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::BoolLit(a < b),
                    _ => SmtExpr::Lt(Box::new(sl), Box::new(sr)),
                }
            }

            SmtExpr::Le(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::BoolLit(a <= b),
                    _ => SmtExpr::Le(Box::new(sl), Box::new(sr)),
                }
            }

            SmtExpr::Gt(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::BoolLit(a > b),
                    _ => SmtExpr::Gt(Box::new(sl), Box::new(sr)),
                }
            }

            SmtExpr::Ge(lhs, rhs) => {
                let sl = Self::simplify_one_pass(lhs);
                let sr = Self::simplify_one_pass(rhs);
                match (&sl, &sr) {
                    (SmtExpr::IntLit(a), SmtExpr::IntLit(b)) => SmtExpr::BoolLit(a >= b),
                    _ => SmtExpr::Ge(Box::new(sl), Box::new(sr)),
                }
            }

            // Recurse into other expressions
            SmtExpr::Select(arr, idx) => SmtExpr::Select(
                Box::new(Self::simplify_one_pass(arr)),
                Box::new(Self::simplify_one_pass(idx)),
            ),
            SmtExpr::Store(arr, idx, val) => SmtExpr::Store(
                Box::new(Self::simplify_one_pass(arr)),
                Box::new(Self::simplify_one_pass(idx)),
                Box::new(Self::simplify_one_pass(val)),
            ),
            SmtExpr::ForAll(vars, body) => {
                SmtExpr::ForAll(vars.clone(), Box::new(Self::simplify_one_pass(body)))
            }
            SmtExpr::Exists(vars, body) => {
                SmtExpr::Exists(vars.clone(), Box::new(Self::simplify_one_pass(body)))
            }
            SmtExpr::Distinct(children) => {
                let simplified: Vec<SmtExpr> =
                    children.iter().map(|c| Self::simplify_one_pass(c)).collect();
                if simplified.len() <= 1 {
                    SmtExpr::BoolLit(true)
                } else {
                    SmtExpr::Distinct(simplified)
                }
            }

            // Leaves: no simplification
            other => other.clone(),
        }
    }

    /// Count the total number of nodes in an expression tree.
    pub fn node_count(expr: &SmtExpr) -> usize {
        match expr {
            SmtExpr::Not(inner) => 1 + Self::node_count(inner),
            SmtExpr::And(children) | SmtExpr::Or(children) | SmtExpr::Distinct(children) => {
                1 + children.iter().map(|c| Self::node_count(c)).sum::<usize>()
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b) => 1 + Self::node_count(a) + Self::node_count(b),
            SmtExpr::Ite(c, t, e) => {
                1 + Self::node_count(c) + Self::node_count(t) + Self::node_count(e)
            }
            SmtExpr::Select(a, i) => 1 + Self::node_count(a) + Self::node_count(i),
            SmtExpr::Store(a, i, v) => {
                1 + Self::node_count(a) + Self::node_count(i) + Self::node_count(v)
            }
            SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => 1 + Self::node_count(body),
            _ => 1,
        }
    }

    /// Estimate the depth of the expression tree.
    pub fn depth(expr: &SmtExpr) -> usize {
        match expr {
            SmtExpr::Not(inner) => 1 + Self::depth(inner),
            SmtExpr::And(children) | SmtExpr::Or(children) | SmtExpr::Distinct(children) => {
                1 + children.iter().map(|c| Self::depth(c)).max().unwrap_or(0)
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b) => 1 + Self::depth(a).max(Self::depth(b)),
            SmtExpr::Ite(c, t, e) => {
                1 + Self::depth(c).max(Self::depth(t)).max(Self::depth(e))
            }
            SmtExpr::Select(a, i) => 1 + Self::depth(a).max(Self::depth(i)),
            SmtExpr::Store(a, i, v) => {
                1 + Self::depth(a).max(Self::depth(i)).max(Self::depth(v))
            }
            SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => 1 + Self::depth(body),
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// LetBinding – common subexpression elimination helper
// ---------------------------------------------------------------------------

/// Identifies and extracts common subexpressions into let bindings.
pub struct LetExtractor;

impl LetExtractor {
    /// Count occurrences of each subexpression (by string representation).
    pub fn count_subexpressions(expr: &SmtExpr) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        Self::count_rec(expr, &mut counts);
        counts
    }

    fn count_rec(expr: &SmtExpr, counts: &mut HashMap<String, usize>) {
        let repr = format!("{:?}", expr);
        *counts.entry(repr).or_insert(0) += 1;

        match expr {
            SmtExpr::Not(inner) => Self::count_rec(inner, counts),
            SmtExpr::And(children) | SmtExpr::Or(children) | SmtExpr::Distinct(children) => {
                for child in children {
                    Self::count_rec(child, counts);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b) => {
                Self::count_rec(a, counts);
                Self::count_rec(b, counts);
            }
            SmtExpr::Ite(c, t, e) => {
                Self::count_rec(c, counts);
                Self::count_rec(t, counts);
                Self::count_rec(e, counts);
            }
            SmtExpr::Select(a, i) => {
                Self::count_rec(a, counts);
                Self::count_rec(i, counts);
            }
            SmtExpr::Store(a, i, v) => {
                Self::count_rec(a, counts);
                Self::count_rec(i, counts);
                Self::count_rec(v, counts);
            }
            SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => {
                Self::count_rec(body, counts);
            }
            _ => {}
        }
    }

    /// Return the number of subexpressions that appear more than once.
    pub fn shared_count(expr: &SmtExpr) -> usize {
        let counts = Self::count_subexpressions(expr);
        counts.values().filter(|&&c| c > 1).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_builder_basic() {
        let formula = FormulaBuilder::new()
            .var("x")
            .var("y")
            .lt()
            .build();
        assert_eq!(
            formula,
            SmtExpr::Lt(
                Box::new(SmtExpr::Const("x".into())),
                Box::new(SmtExpr::Const("y".into())),
            )
        );
    }

    #[test]
    fn test_formula_builder_complex() {
        // (x < y) AND (y > 0)
        let formula = FormulaBuilder::new()
            .var("x")
            .var("y")
            .lt()
            .var("y")
            .int_lit(0)
            .gt()
            .and2()
            .build();

        match &formula {
            SmtExpr::And(children) => assert_eq!(children.len(), 2),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn test_formula_builder_and_n() {
        let formula = FormulaBuilder::new()
            .bool_lit(true)
            .bool_lit(false)
            .var("x")
            .and_n(3)
            .build();
        match formula {
            SmtExpr::And(children) => assert_eq!(children.len(), 3),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn test_formula_builder_ite() {
        let formula = FormulaBuilder::new()
            .var("cond")
            .int_lit(1)
            .int_lit(2)
            .ite()
            .build();
        match formula {
            SmtExpr::Ite(_, _, _) => {}
            _ => panic!("expected Ite"),
        }
    }

    #[test]
    fn test_quantifier_forall() {
        let body = SmtExpr::Gt(
            Box::new(SmtExpr::Const("x".into())),
            Box::new(SmtExpr::IntLit(0)),
        );
        let q = QuantifierBuilder::forall(&[("x", "Int")], body);
        match q {
            SmtExpr::ForAll(vars, _) => {
                assert_eq!(vars.len(), 1);
                assert_eq!(vars[0].0, "x");
            }
            _ => panic!("expected ForAll"),
        }
    }

    #[test]
    fn test_quantifier_bounded_exists() {
        let q = QuantifierBuilder::bounded_exists("i", 0, 10, |i| {
            SmtExpr::Eq(Box::new(i), Box::new(SmtExpr::IntLit(5)))
        });
        match q {
            SmtExpr::Exists(vars, _) => assert_eq!(vars[0].0, "i"),
            _ => panic!("expected Exists"),
        }
    }

    #[test]
    fn test_array_select_store() {
        let arr = SmtExpr::Const("A".into());
        let stored = ArrayHelper::store(arr.clone(), SmtExpr::IntLit(0), SmtExpr::IntLit(42));
        let selected = ArrayHelper::select(stored, SmtExpr::IntLit(0));
        match selected {
            SmtExpr::Select(_, _) => {}
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_array_const_array() {
        let base = SmtExpr::Const("empty_arr".into());
        let arr = ArrayHelper::const_array(
            base,
            &[(0, SmtExpr::IntLit(10)), (1, SmtExpr::IntLit(20))],
        );
        // Should be store(store(empty_arr, 0, 10), 1, 20)
        match arr {
            SmtExpr::Store(_, _, _) => {}
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn test_simplify_double_negation() {
        let expr = SmtExpr::Not(Box::new(SmtExpr::Not(Box::new(SmtExpr::Const("x".into())))));
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::Const("x".into()));
    }

    #[test]
    fn test_simplify_and_identity() {
        let expr = SmtExpr::And(vec![SmtExpr::BoolLit(true), SmtExpr::Const("x".into())]);
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::Const("x".into()));
    }

    #[test]
    fn test_simplify_and_absorbing() {
        let expr = SmtExpr::And(vec![SmtExpr::BoolLit(false), SmtExpr::Const("x".into())]);
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_simplify_or_identity() {
        let expr = SmtExpr::Or(vec![SmtExpr::BoolLit(false), SmtExpr::Const("x".into())]);
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::Const("x".into()));
    }

    #[test]
    fn test_simplify_implies_true_lhs() {
        let expr = SmtExpr::Implies(
            Box::new(SmtExpr::BoolLit(true)),
            Box::new(SmtExpr::Const("x".into())),
        );
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::Const("x".into()));
    }

    #[test]
    fn test_simplify_arithmetic() {
        let expr = SmtExpr::Add(Box::new(SmtExpr::IntLit(3)), Box::new(SmtExpr::IntLit(4)));
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::IntLit(7));
    }

    #[test]
    fn test_simplify_mul_by_zero() {
        let expr = SmtExpr::Mul(
            Box::new(SmtExpr::IntLit(0)),
            Box::new(SmtExpr::Const("x".into())),
        );
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::IntLit(0));
    }

    #[test]
    fn test_simplify_eq_reflexive() {
        let expr = SmtExpr::Eq(
            Box::new(SmtExpr::Const("x".into())),
            Box::new(SmtExpr::Const("x".into())),
        );
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_simplify_ite_true_cond() {
        let expr = SmtExpr::Ite(
            Box::new(SmtExpr::BoolLit(true)),
            Box::new(SmtExpr::IntLit(1)),
            Box::new(SmtExpr::IntLit(2)),
        );
        let simplified = Simplifier::simplify(&expr);
        assert_eq!(simplified, SmtExpr::IntLit(1));
    }

    #[test]
    fn test_node_count() {
        let expr = SmtExpr::And(vec![
            SmtExpr::Const("x".into()),
            SmtExpr::Not(Box::new(SmtExpr::Const("y".into()))),
        ]);
        assert_eq!(Simplifier::node_count(&expr), 4); // And, x, Not, y
    }

    #[test]
    fn test_depth() {
        let expr = SmtExpr::Not(Box::new(SmtExpr::Not(Box::new(SmtExpr::Const("x".into())))));
        assert_eq!(Simplifier::depth(&expr), 3);
    }

    #[test]
    fn test_let_extractor_shared() {
        let sub = SmtExpr::Add(
            Box::new(SmtExpr::Const("x".into())),
            Box::new(SmtExpr::IntLit(1)),
        );
        let expr = SmtExpr::And(vec![
            SmtExpr::Gt(Box::new(sub.clone()), Box::new(SmtExpr::IntLit(0))),
            SmtExpr::Lt(Box::new(sub), Box::new(SmtExpr::IntLit(10))),
        ]);
        let shared = LetExtractor::shared_count(&expr);
        assert!(shared > 0);
    }

    #[test]
    fn test_arrays_equal_on_range() {
        let eq = ArrayHelper::arrays_equal_on_range("A", "B", 0, 3);
        match eq {
            SmtExpr::And(children) => assert_eq!(children.len(), 3),
            _ => panic!("expected And with 3 children"),
        }
    }

    #[test]
    fn test_point_update() {
        let constraints = ArrayHelper::point_update(
            "mem",
            "mem_prime",
            SmtExpr::IntLit(2),
            SmtExpr::IntLit(99),
            4,
        );
        // 1 update constraint + 4 frame conditions
        assert_eq!(constraints.len(), 5);
    }
}
