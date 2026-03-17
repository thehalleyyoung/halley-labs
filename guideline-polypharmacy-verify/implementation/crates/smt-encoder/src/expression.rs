//! SMT expression AST with simplification, substitution, and SMT-LIB2 output.
//!
//! Provides a rich expression tree mirroring the SMT-LIB2 language, plus
//! builder helpers, constant-folding simplification, variable substitution,
//! free-variable collection, node counting, and pretty-printing.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::variable::{VariableId, VariableStore};

// ═══════════════════════════════════════════════════════════════════════════
// SmtExpr
// ═══════════════════════════════════════════════════════════════════════════

/// An SMT expression (abstract syntax tree node).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    // ── Atoms ────────────────────────────────────────────────────────
    Var(VariableId),
    BoolLit(bool),
    IntLit(i64),
    RealLit(f64),

    // ── Boolean connectives ─────────────────────────────────────────
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Iff(Box<SmtExpr>, Box<SmtExpr>),

    // ── Comparison ──────────────────────────────────────────────────
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),
    Distinct(Vec<SmtExpr>),

    // ── Arithmetic ──────────────────────────────────────────────────
    Add(Vec<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Box<SmtExpr>, Box<SmtExpr>),
    Div(Box<SmtExpr>, Box<SmtExpr>),
    Neg(Box<SmtExpr>),
    Abs(Box<SmtExpr>),

    // ── Conditional ─────────────────────────────────────────────────
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // ── Quantifiers ─────────────────────────────────────────────────
    ForAll(Vec<VariableId>, Box<SmtExpr>),
    Exists(Vec<VariableId>, Box<SmtExpr>),

    // ── Let bindings ────────────────────────────────────────────────
    Let(Vec<(VariableId, Box<SmtExpr>)>, Box<SmtExpr>),

    // ── Uninterpreted function application ──────────────────────────
    Apply(String, Vec<SmtExpr>),
}

impl SmtExpr {
    // ── Convenience constructors ────────────────────────────────────

    pub fn var(id: VariableId) -> Self {
        SmtExpr::Var(id)
    }

    pub fn bool_lit(b: bool) -> Self {
        SmtExpr::BoolLit(b)
    }

    pub fn int_lit(n: i64) -> Self {
        SmtExpr::IntLit(n)
    }

    pub fn real_lit(r: f64) -> Self {
        SmtExpr::RealLit(r)
    }

    pub fn not(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::And(exprs)
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Or(exprs)
    }

    pub fn implies(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(lhs), Box::new(rhs))
    }

    pub fn iff(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Iff(Box::new(lhs), Box::new(rhs))
    }

    pub fn eq(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(lhs), Box::new(rhs))
    }

    pub fn lt(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Lt(Box::new(lhs), Box::new(rhs))
    }

    pub fn le(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Le(Box::new(lhs), Box::new(rhs))
    }

    pub fn gt(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Gt(Box::new(lhs), Box::new(rhs))
    }

    pub fn ge(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Ge(Box::new(lhs), Box::new(rhs))
    }

    pub fn add(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Add(exprs)
    }

    pub fn sub(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Sub(Box::new(lhs), Box::new(rhs))
    }

    pub fn mul(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Mul(Box::new(lhs), Box::new(rhs))
    }

    pub fn div(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Div(Box::new(lhs), Box::new(rhs))
    }

    pub fn neg(e: SmtExpr) -> Self {
        SmtExpr::Neg(Box::new(e))
    }

    pub fn abs(e: SmtExpr) -> Self {
        SmtExpr::Abs(Box::new(e))
    }

    pub fn ite(cond: SmtExpr, then_e: SmtExpr, else_e: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(cond), Box::new(then_e), Box::new(else_e))
    }

    pub fn forall(vars: Vec<VariableId>, body: SmtExpr) -> Self {
        SmtExpr::ForAll(vars, Box::new(body))
    }

    pub fn exists(vars: Vec<VariableId>, body: SmtExpr) -> Self {
        SmtExpr::Exists(vars, Box::new(body))
    }

    pub fn let_bind(bindings: Vec<(VariableId, SmtExpr)>, body: SmtExpr) -> Self {
        SmtExpr::Let(
            bindings.into_iter().map(|(v, e)| (v, Box::new(e))).collect(),
            Box::new(body),
        )
    }

    pub fn apply(name: &str, args: Vec<SmtExpr>) -> Self {
        SmtExpr::Apply(name.to_string(), args)
    }

    pub fn distinct(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Distinct(exprs)
    }

    // ── Predicates ──────────────────────────────────────────────────

    pub fn is_true(&self) -> bool {
        matches!(self, SmtExpr::BoolLit(true))
    }

    pub fn is_false(&self) -> bool {
        matches!(self, SmtExpr::BoolLit(false))
    }

    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_)
        )
    }

    pub fn is_var(&self) -> bool {
        matches!(self, SmtExpr::Var(_))
    }

    /// Extract the numeric value if this is a numeric literal.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SmtExpr::IntLit(n) => Some(*n as f64),
            SmtExpr::RealLit(r) => Some(*r),
            _ => None,
        }
    }

    /// Extract the integer value if this is an int literal.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SmtExpr::IntLit(n) => Some(*n),
            _ => None,
        }
    }

    /// Extract the bool value if this is a bool literal.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SmtExpr::BoolLit(b) => Some(*b),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExprBuilder — fluent API
// ═══════════════════════════════════════════════════════════════════════════

/// Fluent builder for constructing SMT expressions.
#[derive(Debug, Clone)]
pub struct ExprBuilder {
    store: *const VariableStore,
}

// Safety: ExprBuilder only holds a read-only pointer used for name lookups
// during the lifetime of the builder; it is neither Send nor Sync.
impl ExprBuilder {
    /// Create a builder referencing the given variable store.
    ///
    /// # Safety
    /// The caller must ensure the store outlives the builder.
    pub fn new(store: &VariableStore) -> Self {
        Self { store: store as *const VariableStore }
    }

    fn store(&self) -> &VariableStore {
        unsafe { &*self.store }
    }

    /// Variable reference by id.
    pub fn var(&self, id: VariableId) -> SmtExpr {
        SmtExpr::Var(id)
    }

    /// Variable reference by name.
    pub fn var_by_name(&self, name: &str) -> Option<SmtExpr> {
        self.store().id_by_name(name).map(SmtExpr::Var)
    }

    /// Variable at a time step.
    pub fn var_at_step(&self, base: &str, step: usize) -> Option<SmtExpr> {
        self.store().id_at_step(base, step).map(SmtExpr::Var)
    }

    pub fn bool_lit(&self, b: bool) -> SmtExpr { SmtExpr::BoolLit(b) }
    pub fn int_lit(&self, n: i64) -> SmtExpr { SmtExpr::IntLit(n) }
    pub fn real_lit(&self, r: f64) -> SmtExpr { SmtExpr::RealLit(r) }

    pub fn and_all(&self, exprs: Vec<SmtExpr>) -> SmtExpr {
        let filtered: Vec<_> = exprs.into_iter().filter(|e| !e.is_true()).collect();
        if filtered.is_empty() {
            return SmtExpr::BoolLit(true);
        }
        if filtered.iter().any(|e| e.is_false()) {
            return SmtExpr::BoolLit(false);
        }
        if filtered.len() == 1 {
            return filtered.into_iter().next().unwrap();
        }
        SmtExpr::And(filtered)
    }

    pub fn or_all(&self, exprs: Vec<SmtExpr>) -> SmtExpr {
        let filtered: Vec<_> = exprs.into_iter().filter(|e| !e.is_false()).collect();
        if filtered.is_empty() {
            return SmtExpr::BoolLit(false);
        }
        if filtered.iter().any(|e| e.is_true()) {
            return SmtExpr::BoolLit(true);
        }
        if filtered.len() == 1 {
            return filtered.into_iter().next().unwrap();
        }
        SmtExpr::Or(filtered)
    }

    pub fn implies(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::implies(lhs, rhs)
    }

    pub fn eq(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::eq(lhs, rhs)
    }

    pub fn lt(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::lt(lhs, rhs)
    }

    pub fn le(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::le(lhs, rhs)
    }

    pub fn gt(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::gt(lhs, rhs)
    }

    pub fn ge(&self, lhs: SmtExpr, rhs: SmtExpr) -> SmtExpr {
        SmtExpr::ge(lhs, rhs)
    }

    pub fn add(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::add(vec![a, b])
    }

    pub fn add_many(&self, exprs: Vec<SmtExpr>) -> SmtExpr {
        SmtExpr::add(exprs)
    }

    pub fn sub(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::sub(a, b)
    }

    pub fn mul(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::mul(a, b)
    }

    pub fn div(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        SmtExpr::div(a, b)
    }

    pub fn neg(&self, e: SmtExpr) -> SmtExpr { SmtExpr::neg(e) }

    pub fn ite(&self, cond: SmtExpr, then_e: SmtExpr, else_e: SmtExpr) -> SmtExpr {
        SmtExpr::ite(cond, then_e, else_e)
    }

    pub fn not(&self, e: SmtExpr) -> SmtExpr { SmtExpr::not(e) }

    pub fn distinct(&self, exprs: Vec<SmtExpr>) -> SmtExpr {
        SmtExpr::distinct(exprs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification
// ═══════════════════════════════════════════════════════════════════════════

/// Simplify an expression (constant folding, double negation, trivial
/// conjunctions/disjunctions, identity arithmetic).
pub fn simplify(expr: &SmtExpr) -> SmtExpr {
    match expr {
        // ── Double negation ─────────────────────────────────────────
        SmtExpr::Not(inner) => {
            let s = simplify(inner);
            match s {
                SmtExpr::Not(inner2) => *inner2,
                SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
                other => SmtExpr::Not(Box::new(other)),
            }
        }

        // ── And ─────────────────────────────────────────────────────
        SmtExpr::And(exprs) => {
            let mut simplified: Vec<SmtExpr> = Vec::new();
            for e in exprs {
                let s = simplify(e);
                match s {
                    SmtExpr::BoolLit(true) => continue,
                    SmtExpr::BoolLit(false) => return SmtExpr::BoolLit(false),
                    SmtExpr::And(inner) => simplified.extend(inner),
                    other => simplified.push(other),
                }
            }
            match simplified.len() {
                0 => SmtExpr::BoolLit(true),
                1 => simplified.into_iter().next().unwrap(),
                _ => SmtExpr::And(simplified),
            }
        }

        // ── Or ──────────────────────────────────────────────────────
        SmtExpr::Or(exprs) => {
            let mut simplified: Vec<SmtExpr> = Vec::new();
            for e in exprs {
                let s = simplify(e);
                match s {
                    SmtExpr::BoolLit(false) => continue,
                    SmtExpr::BoolLit(true) => return SmtExpr::BoolLit(true),
                    SmtExpr::Or(inner) => simplified.extend(inner),
                    other => simplified.push(other),
                }
            }
            match simplified.len() {
                0 => SmtExpr::BoolLit(false),
                1 => simplified.into_iter().next().unwrap(),
                _ => SmtExpr::Or(simplified),
            }
        }

        // ── Implies ─────────────────────────────────────────────────
        SmtExpr::Implies(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (SmtExpr::BoolLit(false), _) => SmtExpr::BoolLit(true),
                (_, SmtExpr::BoolLit(true)) => SmtExpr::BoolLit(true),
                (SmtExpr::BoolLit(true), _) => sb,
                (_, SmtExpr::BoolLit(false)) => SmtExpr::Not(Box::new(sa)),
                _ => SmtExpr::Implies(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Iff ─────────────────────────────────────────────────────
        SmtExpr::Iff(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (SmtExpr::BoolLit(true), _) => sb,
                (_, SmtExpr::BoolLit(true)) => sa,
                (SmtExpr::BoolLit(false), _) => SmtExpr::Not(Box::new(sb)),
                (_, SmtExpr::BoolLit(false)) => SmtExpr::Not(Box::new(sa)),
                _ => SmtExpr::Iff(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Add ─────────────────────────────────────────────────────
        SmtExpr::Add(exprs) => {
            let simplified: Vec<_> = exprs.iter().map(simplify).collect();
            // Constant fold
            let mut constant_sum: f64 = 0.0;
            let mut has_int_only = true;
            let mut int_sum: i64 = 0;
            let mut non_const: Vec<SmtExpr> = Vec::new();
            for s in simplified {
                match &s {
                    SmtExpr::IntLit(n) => {
                        int_sum += n;
                        constant_sum += *n as f64;
                    }
                    SmtExpr::RealLit(r) => {
                        constant_sum += r;
                        has_int_only = false;
                    }
                    _ => {
                        has_int_only = false;
                        non_const.push(s);
                    }
                }
            }
            if non_const.is_empty() {
                if has_int_only {
                    return SmtExpr::IntLit(int_sum);
                }
                return SmtExpr::RealLit(constant_sum);
            }
            if constant_sum != 0.0 {
                if has_int_only && int_sum != 0 {
                    non_const.insert(0, SmtExpr::IntLit(int_sum));
                } else if constant_sum != 0.0 {
                    non_const.insert(0, SmtExpr::RealLit(constant_sum));
                }
            }
            if non_const.len() == 1 {
                return non_const.into_iter().next().unwrap();
            }
            SmtExpr::Add(non_const)
        }

        // ── Sub ─────────────────────────────────────────────────────
        SmtExpr::Sub(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::IntLit(x - y),
                (SmtExpr::RealLit(x), SmtExpr::RealLit(y)) => SmtExpr::RealLit(x - y),
                (_, SmtExpr::IntLit(0)) => sa,
                (_, SmtExpr::RealLit(r)) if *r == 0.0 => sa,
                _ => SmtExpr::Sub(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Mul ─────────────────────────────────────────────────────
        SmtExpr::Mul(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::IntLit(x * y),
                (SmtExpr::RealLit(x), SmtExpr::RealLit(y)) => SmtExpr::RealLit(x * y),
                (SmtExpr::IntLit(0), _) | (_, SmtExpr::IntLit(0)) => SmtExpr::IntLit(0),
                (SmtExpr::RealLit(r), _) | (_, SmtExpr::RealLit(r)) if *r == 0.0 => SmtExpr::RealLit(0.0),
                (SmtExpr::IntLit(1), _) => sb,
                (_, SmtExpr::IntLit(1)) => sa,
                (SmtExpr::RealLit(r), _) if *r == 1.0 => sb,
                (_, SmtExpr::RealLit(r)) if *r == 1.0 => sa,
                _ => SmtExpr::Mul(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Div ─────────────────────────────────────────────────────
        SmtExpr::Div(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb) {
                (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) if *y != 0 => SmtExpr::IntLit(x / y),
                (SmtExpr::RealLit(x), SmtExpr::RealLit(y)) if *y != 0.0 => SmtExpr::RealLit(x / y),
                (_, SmtExpr::IntLit(1)) => sa,
                (_, SmtExpr::RealLit(r)) if *r == 1.0 => sa,
                _ => SmtExpr::Div(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Neg ─────────────────────────────────────────────────────
        SmtExpr::Neg(inner) => {
            let s = simplify(inner);
            match s {
                SmtExpr::IntLit(n) => SmtExpr::IntLit(-n),
                SmtExpr::RealLit(r) => SmtExpr::RealLit(-r),
                SmtExpr::Neg(inner2) => *inner2,
                other => SmtExpr::Neg(Box::new(other)),
            }
        }

        // ── Abs ─────────────────────────────────────────────────────
        SmtExpr::Abs(inner) => {
            let s = simplify(inner);
            match s {
                SmtExpr::IntLit(n) => SmtExpr::IntLit(n.abs()),
                SmtExpr::RealLit(r) => SmtExpr::RealLit(r.abs()),
                other => SmtExpr::Abs(Box::new(other)),
            }
        }

        // ── Ite ─────────────────────────────────────────────────────
        SmtExpr::Ite(c, t, e) => {
            let sc = simplify(c);
            let st = simplify(t);
            let se = simplify(e);
            match &sc {
                SmtExpr::BoolLit(true) => st,
                SmtExpr::BoolLit(false) => se,
                _ if st == se => st,
                _ => SmtExpr::Ite(Box::new(sc), Box::new(st), Box::new(se)),
            }
        }

        // ── Eq ──────────────────────────────────────────────────────
        SmtExpr::Eq(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            if sa == sb { return SmtExpr::BoolLit(true); }
            match (&sa, &sb) {
                (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::BoolLit(x == y),
                (SmtExpr::RealLit(x), SmtExpr::RealLit(y)) => SmtExpr::BoolLit(x == y),
                (SmtExpr::BoolLit(x), SmtExpr::BoolLit(y)) => SmtExpr::BoolLit(x == y),
                _ => SmtExpr::Eq(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Lt / Le / Gt / Ge ───────────────────────────────────────
        SmtExpr::Lt(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (sa.as_f64(), sb.as_f64()) {
                (Some(x), Some(y)) => SmtExpr::BoolLit(x < y),
                _ => SmtExpr::Lt(Box::new(sa), Box::new(sb)),
            }
        }
        SmtExpr::Le(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (sa.as_f64(), sb.as_f64()) {
                (Some(x), Some(y)) => SmtExpr::BoolLit(x <= y),
                _ => SmtExpr::Le(Box::new(sa), Box::new(sb)),
            }
        }
        SmtExpr::Gt(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (sa.as_f64(), sb.as_f64()) {
                (Some(x), Some(y)) => SmtExpr::BoolLit(x > y),
                _ => SmtExpr::Gt(Box::new(sa), Box::new(sb)),
            }
        }
        SmtExpr::Ge(a, b) => {
            let sa = simplify(a);
            let sb = simplify(b);
            match (sa.as_f64(), sb.as_f64()) {
                (Some(x), Some(y)) => SmtExpr::BoolLit(x >= y),
                _ => SmtExpr::Ge(Box::new(sa), Box::new(sb)),
            }
        }

        // ── Pass-through ────────────────────────────────────────────
        SmtExpr::Distinct(exprs) => {
            let simplified: Vec<_> = exprs.iter().map(simplify).collect();
            SmtExpr::Distinct(simplified)
        }

        SmtExpr::ForAll(vars, body) => {
            SmtExpr::ForAll(vars.clone(), Box::new(simplify(body)))
        }
        SmtExpr::Exists(vars, body) => {
            SmtExpr::Exists(vars.clone(), Box::new(simplify(body)))
        }
        SmtExpr::Let(bindings, body) => {
            let sb: Vec<_> = bindings.iter()
                .map(|(v, e)| (*v, Box::new(simplify(e))))
                .collect();
            SmtExpr::Let(sb, Box::new(simplify(body)))
        }
        SmtExpr::Apply(name, args) => {
            SmtExpr::Apply(name.clone(), args.iter().map(simplify).collect())
        }

        // Atoms are already simplified.
        other => other.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Substitute every occurrence of `var` with `replacement` in `expr`.
pub fn substitute(expr: &SmtExpr, var: VariableId, replacement: &SmtExpr) -> SmtExpr {
    match expr {
        SmtExpr::Var(id) if *id == var => replacement.clone(),
        SmtExpr::Var(_) | SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) => {
            expr.clone()
        }

        SmtExpr::Not(e) => SmtExpr::Not(Box::new(substitute(e, var, replacement))),
        SmtExpr::And(es) => SmtExpr::And(es.iter().map(|e| substitute(e, var, replacement)).collect()),
        SmtExpr::Or(es) => SmtExpr::Or(es.iter().map(|e| substitute(e, var, replacement)).collect()),
        SmtExpr::Implies(a, b) => SmtExpr::Implies(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Iff(a, b) => SmtExpr::Iff(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),

        SmtExpr::Eq(a, b) => SmtExpr::Eq(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Lt(a, b) => SmtExpr::Lt(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Le(a, b) => SmtExpr::Le(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Gt(a, b) => SmtExpr::Gt(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Ge(a, b) => SmtExpr::Ge(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Distinct(es) => SmtExpr::Distinct(
            es.iter().map(|e| substitute(e, var, replacement)).collect(),
        ),

        SmtExpr::Add(es) => SmtExpr::Add(es.iter().map(|e| substitute(e, var, replacement)).collect()),
        SmtExpr::Sub(a, b) => SmtExpr::Sub(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Mul(a, b) => SmtExpr::Mul(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Div(a, b) => SmtExpr::Div(
            Box::new(substitute(a, var, replacement)),
            Box::new(substitute(b, var, replacement)),
        ),
        SmtExpr::Neg(e) => SmtExpr::Neg(Box::new(substitute(e, var, replacement))),
        SmtExpr::Abs(e) => SmtExpr::Abs(Box::new(substitute(e, var, replacement))),

        SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
            Box::new(substitute(c, var, replacement)),
            Box::new(substitute(t, var, replacement)),
            Box::new(substitute(e, var, replacement)),
        ),

        SmtExpr::ForAll(vs, body) => {
            if vs.contains(&var) {
                expr.clone() // var is bound
            } else {
                SmtExpr::ForAll(vs.clone(), Box::new(substitute(body, var, replacement)))
            }
        }
        SmtExpr::Exists(vs, body) => {
            if vs.contains(&var) {
                expr.clone()
            } else {
                SmtExpr::Exists(vs.clone(), Box::new(substitute(body, var, replacement)))
            }
        }
        SmtExpr::Let(bindings, body) => {
            let bound: HashSet<_> = bindings.iter().map(|(v, _)| *v).collect();
            let new_bindings: Vec<_> = bindings.iter()
                .map(|(v, e)| (*v, Box::new(substitute(e, var, replacement))))
                .collect();
            if bound.contains(&var) {
                SmtExpr::Let(new_bindings, body.clone())
            } else {
                SmtExpr::Let(new_bindings, Box::new(substitute(body, var, replacement)))
            }
        }
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter().map(|e| substitute(e, var, replacement)).collect(),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free Variables
// ═══════════════════════════════════════════════════════════════════════════

/// Collect the set of free (unbound) variables in an expression.
pub fn free_vars(expr: &SmtExpr) -> HashSet<VariableId> {
    let mut result = HashSet::new();
    free_vars_inner(expr, &HashSet::new(), &mut result);
    result
}

fn free_vars_inner(
    expr: &SmtExpr,
    bound: &HashSet<VariableId>,
    acc: &mut HashSet<VariableId>,
) {
    match expr {
        SmtExpr::Var(id) => {
            if !bound.contains(id) {
                acc.insert(*id);
            }
        }
        SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) => {}

        SmtExpr::Not(e) | SmtExpr::Neg(e) | SmtExpr::Abs(e) => {
            free_vars_inner(e, bound, acc);
        }

        SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Distinct(es) => {
            for e in es {
                free_vars_inner(e, bound, acc);
            }
        }

        SmtExpr::Implies(a, b) | SmtExpr::Iff(a, b)
        | SmtExpr::Eq(a, b) | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
        | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b)
        | SmtExpr::Sub(a, b) | SmtExpr::Mul(a, b) | SmtExpr::Div(a, b) => {
            free_vars_inner(a, bound, acc);
            free_vars_inner(b, bound, acc);
        }

        SmtExpr::Ite(c, t, e) => {
            free_vars_inner(c, bound, acc);
            free_vars_inner(t, bound, acc);
            free_vars_inner(e, bound, acc);
        }

        SmtExpr::ForAll(vs, body) | SmtExpr::Exists(vs, body) => {
            let mut new_bound = bound.clone();
            for v in vs {
                new_bound.insert(*v);
            }
            free_vars_inner(body, &new_bound, acc);
        }

        SmtExpr::Let(bindings, body) => {
            let mut new_bound = bound.clone();
            for (v, e) in bindings {
                free_vars_inner(e, bound, acc);
                new_bound.insert(*v);
            }
            free_vars_inner(body, &new_bound, acc);
        }

        SmtExpr::Apply(_, args) => {
            for a in args {
                free_vars_inner(a, bound, acc);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression Size
// ═══════════════════════════════════════════════════════════════════════════

/// Count the number of nodes in an expression tree.
pub fn expr_size(expr: &SmtExpr) -> usize {
    match expr {
        SmtExpr::Var(_) | SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) => 1,

        SmtExpr::Not(e) | SmtExpr::Neg(e) | SmtExpr::Abs(e) => 1 + expr_size(e),

        SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Distinct(es) => {
            1 + es.iter().map(expr_size).sum::<usize>()
        }

        SmtExpr::Implies(a, b) | SmtExpr::Iff(a, b)
        | SmtExpr::Eq(a, b) | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
        | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b)
        | SmtExpr::Sub(a, b) | SmtExpr::Mul(a, b) | SmtExpr::Div(a, b) => {
            1 + expr_size(a) + expr_size(b)
        }

        SmtExpr::Ite(c, t, e) => 1 + expr_size(c) + expr_size(t) + expr_size(e),

        SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => 1 + expr_size(body),

        SmtExpr::Let(bindings, body) => {
            1 + bindings.iter().map(|(_, e)| expr_size(e)).sum::<usize>() + expr_size(body)
        }

        SmtExpr::Apply(_, args) => 1 + args.iter().map(expr_size).sum::<usize>(),
    }
}

/// Total size of a collection of expressions.
pub fn total_expr_size(exprs: &[SmtExpr]) -> usize {
    exprs.iter().map(expr_size).sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// SMT-LIB2 Pretty Printing
// ═══════════════════════════════════════════════════════════════════════════

/// Format an expression as an SMT-LIB2 string.
pub fn to_smtlib2(expr: &SmtExpr, store: &VariableStore) -> String {
    let mut buf = String::new();
    write_smtlib2(expr, store, &mut buf);
    buf
}

fn write_smtlib2(expr: &SmtExpr, store: &VariableStore, buf: &mut String) {
    match expr {
        SmtExpr::Var(id) => {
            if let Some(v) = store.get(*id) {
                buf.push_str(&v.qualified_name());
            } else {
                buf.push_str(&format!("?v{}", id.0));
            }
        }
        SmtExpr::BoolLit(true) => buf.push_str("true"),
        SmtExpr::BoolLit(false) => buf.push_str("false"),
        SmtExpr::IntLit(n) => {
            if *n < 0 {
                buf.push_str(&format!("(- {})", -n));
            } else {
                buf.push_str(&n.to_string());
            }
        }
        SmtExpr::RealLit(r) => {
            if *r < 0.0 {
                buf.push_str(&format!("(- {})", format_real(-r)));
            } else {
                buf.push_str(&format_real(*r));
            }
        }

        SmtExpr::Not(e) => {
            buf.push_str("(not ");
            write_smtlib2(e, store, buf);
            buf.push(')');
        }

        SmtExpr::And(es) => write_nary("and", es, store, buf),
        SmtExpr::Or(es) => write_nary("or", es, store, buf),

        SmtExpr::Implies(a, b) => write_binary("=>", a, b, store, buf),
        SmtExpr::Iff(a, b) => write_binary("=", a, b, store, buf),
        SmtExpr::Eq(a, b) => write_binary("=", a, b, store, buf),
        SmtExpr::Lt(a, b) => write_binary("<", a, b, store, buf),
        SmtExpr::Le(a, b) => write_binary("<=", a, b, store, buf),
        SmtExpr::Gt(a, b) => write_binary(">", a, b, store, buf),
        SmtExpr::Ge(a, b) => write_binary(">=", a, b, store, buf),

        SmtExpr::Add(es) => write_nary("+", es, store, buf),
        SmtExpr::Sub(a, b) => write_binary("-", a, b, store, buf),
        SmtExpr::Mul(a, b) => write_binary("*", a, b, store, buf),
        SmtExpr::Div(a, b) => write_binary("/", a, b, store, buf),

        SmtExpr::Neg(e) => {
            buf.push_str("(- ");
            write_smtlib2(e, store, buf);
            buf.push(')');
        }
        SmtExpr::Abs(e) => {
            buf.push_str("(abs ");
            write_smtlib2(e, store, buf);
            buf.push(')');
        }

        SmtExpr::Ite(c, t, e) => {
            buf.push_str("(ite ");
            write_smtlib2(c, store, buf);
            buf.push(' ');
            write_smtlib2(t, store, buf);
            buf.push(' ');
            write_smtlib2(e, store, buf);
            buf.push(')');
        }

        SmtExpr::Distinct(es) => write_nary("distinct", es, store, buf),

        SmtExpr::ForAll(vs, body) => {
            buf.push_str("(forall (");
            for (i, v) in vs.iter().enumerate() {
                if i > 0 { buf.push(' '); }
                if let Some(var) = store.get(*v) {
                    buf.push_str(&format!("({} {})", var.qualified_name(), var.sort.to_smtlib2()));
                } else {
                    buf.push_str(&format!("(?v{} Int)", v.0));
                }
            }
            buf.push_str(") ");
            write_smtlib2(body, store, buf);
            buf.push(')');
        }

        SmtExpr::Exists(vs, body) => {
            buf.push_str("(exists (");
            for (i, v) in vs.iter().enumerate() {
                if i > 0 { buf.push(' '); }
                if let Some(var) = store.get(*v) {
                    buf.push_str(&format!("({} {})", var.qualified_name(), var.sort.to_smtlib2()));
                } else {
                    buf.push_str(&format!("(?v{} Int)", v.0));
                }
            }
            buf.push_str(") ");
            write_smtlib2(body, store, buf);
            buf.push(')');
        }

        SmtExpr::Let(bindings, body) => {
            buf.push_str("(let (");
            for (i, (v, e)) in bindings.iter().enumerate() {
                if i > 0 { buf.push(' '); }
                buf.push('(');
                if let Some(var) = store.get(*v) {
                    buf.push_str(&var.qualified_name());
                } else {
                    buf.push_str(&format!("?v{}", v.0));
                }
                buf.push(' ');
                write_smtlib2(e, store, buf);
                buf.push(')');
            }
            buf.push_str(") ");
            write_smtlib2(body, store, buf);
            buf.push(')');
        }

        SmtExpr::Apply(name, args) => {
            buf.push('(');
            buf.push_str(name);
            for a in args {
                buf.push(' ');
                write_smtlib2(a, store, buf);
            }
            buf.push(')');
        }
    }
}

fn write_binary(op: &str, a: &SmtExpr, b: &SmtExpr, store: &VariableStore, buf: &mut String) {
    buf.push('(');
    buf.push_str(op);
    buf.push(' ');
    write_smtlib2(a, store, buf);
    buf.push(' ');
    write_smtlib2(b, store, buf);
    buf.push(')');
}

fn write_nary(op: &str, es: &[SmtExpr], store: &VariableStore, buf: &mut String) {
    buf.push('(');
    buf.push_str(op);
    for e in es {
        buf.push(' ');
        write_smtlib2(e, store, buf);
    }
    buf.push(')');
}

fn format_real(r: f64) -> String {
    if r == r.floor() && r.abs() < 1e15 {
        format!("{:.1}", r)
    } else {
        format!("{}", r)
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Lightweight display without variable store (uses raw ids).
        let store = VariableStore::new();
        write!(f, "{}", to_smtlib2(self, &store))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::VariableStore;

    fn make_store() -> VariableStore {
        let mut s = VariableStore::new();
        s.create_bool("p");    // id 0
        s.create_bool("q");    // id 1
        s.create_real("x");    // id 2
        s.create_real("y");    // id 3
        s.create_int("n");     // id 4
        s
    }

    #[test]
    fn test_literal_predicates() {
        assert!(SmtExpr::BoolLit(true).is_true());
        assert!(SmtExpr::BoolLit(false).is_false());
        assert!(SmtExpr::IntLit(42).is_literal());
        assert!(!SmtExpr::Var(VariableId(0)).is_literal());
    }

    #[test]
    fn test_simplify_double_negation() {
        let e = SmtExpr::not(SmtExpr::not(SmtExpr::Var(VariableId(0))));
        assert_eq!(simplify(&e), SmtExpr::Var(VariableId(0)));
    }

    #[test]
    fn test_simplify_and_true() {
        let e = SmtExpr::and(vec![
            SmtExpr::BoolLit(true),
            SmtExpr::Var(VariableId(0)),
            SmtExpr::BoolLit(true),
        ]);
        assert_eq!(simplify(&e), SmtExpr::Var(VariableId(0)));
    }

    #[test]
    fn test_simplify_and_false() {
        let e = SmtExpr::and(vec![
            SmtExpr::Var(VariableId(0)),
            SmtExpr::BoolLit(false),
        ]);
        assert_eq!(simplify(&e), SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_simplify_or_false() {
        let e = SmtExpr::or(vec![
            SmtExpr::BoolLit(false),
            SmtExpr::Var(VariableId(1)),
        ]);
        assert_eq!(simplify(&e), SmtExpr::Var(VariableId(1)));
    }

    #[test]
    fn test_simplify_or_true() {
        let e = SmtExpr::or(vec![
            SmtExpr::Var(VariableId(0)),
            SmtExpr::BoolLit(true),
        ]);
        assert_eq!(simplify(&e), SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_simplify_arithmetic() {
        let e = SmtExpr::add(vec![SmtExpr::IntLit(2), SmtExpr::IntLit(3)]);
        assert_eq!(simplify(&e), SmtExpr::IntLit(5));

        let e2 = SmtExpr::mul(SmtExpr::IntLit(4), SmtExpr::IntLit(5));
        assert_eq!(simplify(&e2), SmtExpr::IntLit(20));

        let e3 = SmtExpr::sub(SmtExpr::IntLit(10), SmtExpr::IntLit(0));
        assert_eq!(simplify(&e3), SmtExpr::IntLit(10));
    }

    #[test]
    fn test_simplify_mul_identity() {
        let v = SmtExpr::Var(VariableId(2));
        let e = SmtExpr::mul(SmtExpr::IntLit(1), v.clone());
        assert_eq!(simplify(&e), v);
    }

    #[test]
    fn test_simplify_mul_zero() {
        let v = SmtExpr::Var(VariableId(2));
        let e = SmtExpr::mul(SmtExpr::IntLit(0), v);
        assert_eq!(simplify(&e), SmtExpr::IntLit(0));
    }

    #[test]
    fn test_simplify_implies_false_antecedent() {
        let e = SmtExpr::implies(SmtExpr::BoolLit(false), SmtExpr::Var(VariableId(0)));
        assert_eq!(simplify(&e), SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_simplify_ite_const_cond() {
        let e = SmtExpr::ite(
            SmtExpr::BoolLit(true),
            SmtExpr::IntLit(1),
            SmtExpr::IntLit(2),
        );
        assert_eq!(simplify(&e), SmtExpr::IntLit(1));
    }

    #[test]
    fn test_simplify_eq_same() {
        let v = SmtExpr::Var(VariableId(0));
        let e = SmtExpr::eq(v.clone(), v);
        assert_eq!(simplify(&e), SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_simplify_comparison_constants() {
        let e = SmtExpr::lt(SmtExpr::IntLit(3), SmtExpr::IntLit(5));
        assert_eq!(simplify(&e), SmtExpr::BoolLit(true));

        let e2 = SmtExpr::ge(SmtExpr::IntLit(2), SmtExpr::IntLit(10));
        assert_eq!(simplify(&e2), SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_substitute_basic() {
        let v0 = VariableId(0);
        let v1 = VariableId(1);
        let e = SmtExpr::and(vec![SmtExpr::Var(v0), SmtExpr::Var(v1)]);
        let result = substitute(&e, v0, &SmtExpr::BoolLit(true));
        assert_eq!(result, SmtExpr::and(vec![SmtExpr::BoolLit(true), SmtExpr::Var(v1)]));
    }

    #[test]
    fn test_substitute_bound() {
        let v0 = VariableId(0);
        let e = SmtExpr::forall(vec![v0], SmtExpr::Var(v0));
        let result = substitute(&e, v0, &SmtExpr::BoolLit(true));
        assert_eq!(result, e); // bound variable not substituted
    }

    #[test]
    fn test_free_vars() {
        let v0 = VariableId(0);
        let v1 = VariableId(1);
        let v2 = VariableId(2);
        let e = SmtExpr::and(vec![
            SmtExpr::Var(v0),
            SmtExpr::forall(vec![v1], SmtExpr::implies(SmtExpr::Var(v1), SmtExpr::Var(v2))),
        ]);
        let fv = free_vars(&e);
        assert!(fv.contains(&v0));
        assert!(!fv.contains(&v1)); // bound
        assert!(fv.contains(&v2));
    }

    #[test]
    fn test_expr_size() {
        let e = SmtExpr::and(vec![
            SmtExpr::Var(VariableId(0)),
            SmtExpr::not(SmtExpr::Var(VariableId(1))),
        ]);
        assert_eq!(expr_size(&e), 4); // and, var0, not, var1
    }

    #[test]
    fn test_smtlib2_output() {
        let store = make_store();
        let p = VariableId(0);
        let x = VariableId(2);

        let e = SmtExpr::implies(
            SmtExpr::Var(p),
            SmtExpr::gt(SmtExpr::Var(x), SmtExpr::RealLit(0.0)),
        );
        let s = to_smtlib2(&e, &store);
        assert_eq!(s, "(=> p (> x 0.0))");
    }

    #[test]
    fn test_smtlib2_negative() {
        let store = VariableStore::new();
        let e = SmtExpr::IntLit(-5);
        assert_eq!(to_smtlib2(&e, &store), "(- 5)");
    }

    #[test]
    fn test_expr_builder_and_all() {
        let store = make_store();
        let builder = ExprBuilder::new(&store);
        let e = builder.and_all(vec![
            SmtExpr::BoolLit(true),
            SmtExpr::Var(VariableId(0)),
        ]);
        assert_eq!(e, SmtExpr::Var(VariableId(0)));
    }

    #[test]
    fn test_expr_builder_or_all_false() {
        let store = make_store();
        let builder = ExprBuilder::new(&store);
        let e = builder.or_all(vec![
            SmtExpr::BoolLit(false),
            SmtExpr::BoolLit(false),
        ]);
        assert_eq!(e, SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_as_accessors() {
        assert_eq!(SmtExpr::IntLit(42).as_i64(), Some(42));
        assert_eq!(SmtExpr::RealLit(3.14).as_f64(), Some(3.14));
        assert_eq!(SmtExpr::BoolLit(true).as_bool(), Some(true));
        assert_eq!(SmtExpr::Var(VariableId(0)).as_f64(), None);
    }

    #[test]
    fn test_simplify_nested_and() {
        let e = SmtExpr::and(vec![
            SmtExpr::and(vec![SmtExpr::Var(VariableId(0)), SmtExpr::Var(VariableId(1))]),
            SmtExpr::Var(VariableId(2)),
        ]);
        let s = simplify(&e);
        // Flattened
        if let SmtExpr::And(inner) = &s {
            assert_eq!(inner.len(), 3);
        } else {
            panic!("Expected And");
        }
    }

    #[test]
    fn test_simplify_neg_neg() {
        let e = SmtExpr::neg(SmtExpr::neg(SmtExpr::Var(VariableId(0))));
        assert_eq!(simplify(&e), SmtExpr::Var(VariableId(0)));
    }

    #[test]
    fn test_simplify_abs_literal() {
        let e = SmtExpr::abs(SmtExpr::IntLit(-7));
        assert_eq!(simplify(&e), SmtExpr::IntLit(7));
    }

    #[test]
    fn test_total_expr_size() {
        let exprs = vec![
            SmtExpr::Var(VariableId(0)),
            SmtExpr::and(vec![SmtExpr::Var(VariableId(1)), SmtExpr::Var(VariableId(2))]),
        ];
        assert_eq!(total_expr_size(&exprs), 4); // 1 + (1 + 1 + 1)
    }
}
