//! SMT-related types for constraint encoding and solving.
//!
//! Provides the expression language, sorts, formulas, models, proofs,
//! and SMT-LIB2 pretty-printing used by the encoding and verification phases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── SMT Sorts ────────────────────────────────────────────────────────────

/// SMT-LIB sorts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
    BitVec(u32),
    Array(Box<SmtSort>, Box<SmtSort>),
    Uninterpreted(String),
}

impl SmtSort {
    pub fn bv(width: u32) -> Self {
        SmtSort::BitVec(width)
    }

    pub fn array(index: SmtSort, element: SmtSort) -> Self {
        SmtSort::Array(Box::new(index), Box::new(element))
    }

    /// Whether this sort is a bitvector.
    pub fn is_bitvec(&self) -> bool {
        matches!(self, SmtSort::BitVec(_))
    }

    /// Bitvector width (None if not a bitvector).
    pub fn bv_width(&self) -> Option<u32> {
        match self {
            SmtSort::BitVec(w) => Some(*w),
            _ => None,
        }
    }

    /// SMT-LIB2 sort string.
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtSort::Bool => "Bool".into(),
            SmtSort::Int => "Int".into(),
            SmtSort::Real => "Real".into(),
            SmtSort::BitVec(w) => format!("(_ BitVec {})", w),
            SmtSort::Array(idx, elem) => {
                format!("(Array {} {})", idx.to_smtlib(), elem.to_smtlib())
            }
            SmtSort::Uninterpreted(name) => name.clone(),
        }
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ── SMT Expressions ──────────────────────────────────────────────────────

/// SMT expressions (terms and formulas).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtExpr {
    // Atoms
    Var(String, SmtSort),
    BoolConst(bool),
    IntConst(i64),
    RealConst(i64, u64), // numerator, denominator
    BvConst(u64, u32),   // value, width

    // Boolean
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Xor(Box<SmtExpr>, Box<SmtExpr>),
    Iff(Box<SmtExpr>, Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // Equality and comparison
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Distinct(Vec<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),

    // Arithmetic
    Add(Vec<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Vec<SmtExpr>),
    Div(Box<SmtExpr>, Box<SmtExpr>),
    Mod(Box<SmtExpr>, Box<SmtExpr>),
    Neg(Box<SmtExpr>),
    Abs(Box<SmtExpr>),

    // Bitvector arithmetic
    BvAdd(Box<SmtExpr>, Box<SmtExpr>),
    BvSub(Box<SmtExpr>, Box<SmtExpr>),
    BvMul(Box<SmtExpr>, Box<SmtExpr>),
    BvUdiv(Box<SmtExpr>, Box<SmtExpr>),
    BvSdiv(Box<SmtExpr>, Box<SmtExpr>),
    BvUrem(Box<SmtExpr>, Box<SmtExpr>),
    BvSrem(Box<SmtExpr>, Box<SmtExpr>),

    // Bitvector bitwise
    BvAnd(Box<SmtExpr>, Box<SmtExpr>),
    BvOr(Box<SmtExpr>, Box<SmtExpr>),
    BvXor(Box<SmtExpr>, Box<SmtExpr>),
    BvNot(Box<SmtExpr>),
    BvNeg(Box<SmtExpr>),
    BvShl(Box<SmtExpr>, Box<SmtExpr>),
    BvLshr(Box<SmtExpr>, Box<SmtExpr>),
    BvAshr(Box<SmtExpr>, Box<SmtExpr>),

    // Bitvector comparison
    BvUlt(Box<SmtExpr>, Box<SmtExpr>),
    BvUle(Box<SmtExpr>, Box<SmtExpr>),
    BvUgt(Box<SmtExpr>, Box<SmtExpr>),
    BvUge(Box<SmtExpr>, Box<SmtExpr>),
    BvSlt(Box<SmtExpr>, Box<SmtExpr>),
    BvSle(Box<SmtExpr>, Box<SmtExpr>),
    BvSgt(Box<SmtExpr>, Box<SmtExpr>),
    BvSge(Box<SmtExpr>, Box<SmtExpr>),

    // Bitvector manipulation
    Concat(Box<SmtExpr>, Box<SmtExpr>),
    Extract(u32, u32, Box<SmtExpr>), // high, low, expr
    ZeroExt(u32, Box<SmtExpr>),
    SignExt(u32, Box<SmtExpr>),

    // Array
    Select(Box<SmtExpr>, Box<SmtExpr>),
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // Quantifiers
    ForAll(Vec<(String, SmtSort)>, Box<SmtExpr>),
    Exists(Vec<(String, SmtSort)>, Box<SmtExpr>),

    // Let binding
    Let(Vec<(String, SmtExpr)>, Box<SmtExpr>),

    // Uninterpreted function application
    Apply(String, Vec<SmtExpr>),
}

// ── Expression Builder Helpers ───────────────────────────────────────────

impl SmtExpr {
    pub fn bool_var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into(), SmtSort::Bool)
    }

    pub fn int_var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into(), SmtSort::Int)
    }

    pub fn bv_var(name: impl Into<String>, width: u32) -> Self {
        SmtExpr::Var(name.into(), SmtSort::BitVec(width))
    }

    pub fn array_var(name: impl Into<String>, idx: SmtSort, elem: SmtSort) -> Self {
        SmtExpr::Var(name.into(), SmtSort::Array(Box::new(idx), Box::new(elem)))
    }

    // Boolean builders
    pub fn not(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        if exprs.len() == 1 {
            return exprs.into_iter().next().unwrap();
        }
        SmtExpr::And(exprs)
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        if exprs.len() == 1 {
            return exprs.into_iter().next().unwrap();
        }
        SmtExpr::Or(exprs)
    }

    pub fn implies(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(lhs), Box::new(rhs))
    }

    pub fn eq(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(lhs), Box::new(rhs))
    }

    pub fn ite(cond: SmtExpr, then_e: SmtExpr, else_e: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(cond), Box::new(then_e), Box::new(else_e))
    }

    // Bitvector builders
    pub fn bvadd(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::BvAdd(Box::new(a), Box::new(b))
    }

    pub fn bvsub(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::BvSub(Box::new(a), Box::new(b))
    }

    pub fn bvand(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::BvAnd(Box::new(a), Box::new(b))
    }

    pub fn bvor(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::BvOr(Box::new(a), Box::new(b))
    }

    pub fn extract(high: u32, low: u32, e: SmtExpr) -> Self {
        SmtExpr::Extract(high, low, Box::new(e))
    }

    pub fn concat(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Concat(Box::new(a), Box::new(b))
    }

    pub fn zero_ext(bits: u32, e: SmtExpr) -> Self {
        SmtExpr::ZeroExt(bits, Box::new(e))
    }

    pub fn sign_ext(bits: u32, e: SmtExpr) -> Self {
        SmtExpr::SignExt(bits, Box::new(e))
    }

    // Array builders
    pub fn select(array: SmtExpr, index: SmtExpr) -> Self {
        SmtExpr::Select(Box::new(array), Box::new(index))
    }

    pub fn store(array: SmtExpr, index: SmtExpr, value: SmtExpr) -> Self {
        SmtExpr::Store(Box::new(array), Box::new(index), Box::new(value))
    }

    // ── Analysis ─────────────────────────────────────────────────

    /// Node count of the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            SmtExpr::Var(_, _) | SmtExpr::BoolConst(_) | SmtExpr::IntConst(_)
            | SmtExpr::RealConst(_, _) | SmtExpr::BvConst(_, _) => 1,
            SmtExpr::Not(e) | SmtExpr::Neg(e) | SmtExpr::Abs(e)
            | SmtExpr::BvNot(e) | SmtExpr::BvNeg(e) => 1 + e.node_count(),
            SmtExpr::ZeroExt(_, e) | SmtExpr::SignExt(_, e) | SmtExpr::Extract(_, _, e) => {
                1 + e.node_count()
            }
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es)
            | SmtExpr::Distinct(es) => {
                1 + es.iter().map(|e| e.node_count()).sum::<usize>()
            }
            SmtExpr::Implies(a, b) | SmtExpr::Xor(a, b) | SmtExpr::Iff(a, b)
            | SmtExpr::Eq(a, b) | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b) | SmtExpr::Sub(a, b)
            | SmtExpr::Div(a, b) | SmtExpr::Mod(a, b)
            | SmtExpr::BvAdd(a, b) | SmtExpr::BvSub(a, b) | SmtExpr::BvMul(a, b)
            | SmtExpr::BvUdiv(a, b) | SmtExpr::BvSdiv(a, b)
            | SmtExpr::BvUrem(a, b) | SmtExpr::BvSrem(a, b)
            | SmtExpr::BvAnd(a, b) | SmtExpr::BvOr(a, b) | SmtExpr::BvXor(a, b)
            | SmtExpr::BvShl(a, b) | SmtExpr::BvLshr(a, b) | SmtExpr::BvAshr(a, b)
            | SmtExpr::BvUlt(a, b) | SmtExpr::BvUle(a, b) | SmtExpr::BvUgt(a, b) | SmtExpr::BvUge(a, b)
            | SmtExpr::BvSlt(a, b) | SmtExpr::BvSle(a, b) | SmtExpr::BvSgt(a, b) | SmtExpr::BvSge(a, b)
            | SmtExpr::Concat(a, b) | SmtExpr::Select(a, b) => {
                1 + a.node_count() + b.node_count()
            }
            SmtExpr::Ite(c, t, e) | SmtExpr::Store(c, t, e) => {
                1 + c.node_count() + t.node_count() + e.node_count()
            }
            SmtExpr::ForAll(_, body) | SmtExpr::Exists(_, body) => 1 + body.node_count(),
            SmtExpr::Let(bindings, body) => {
                1 + bindings.iter().map(|(_, e)| e.node_count()).sum::<usize>() + body.node_count()
            }
            SmtExpr::Apply(_, args) => {
                1 + args.iter().map(|e| e.node_count()).sum::<usize>()
            }
        }
    }

    /// Collect all free variable names.
    pub fn free_variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            SmtExpr::Var(name, _) => { vars.insert(name.clone()); }
            SmtExpr::Not(e) | SmtExpr::Neg(e) | SmtExpr::Abs(e)
            | SmtExpr::BvNot(e) | SmtExpr::BvNeg(e)
            | SmtExpr::ZeroExt(_, e) | SmtExpr::SignExt(_, e) | SmtExpr::Extract(_, _, e) => {
                e.collect_vars(vars);
            }
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es)
            | SmtExpr::Distinct(es) => {
                for e in es { e.collect_vars(vars); }
            }
            SmtExpr::Implies(a, b) | SmtExpr::Xor(a, b) | SmtExpr::Iff(a, b)
            | SmtExpr::Eq(a, b) | SmtExpr::Lt(a, b) | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b) | SmtExpr::Ge(a, b) | SmtExpr::Sub(a, b)
            | SmtExpr::Div(a, b) | SmtExpr::Mod(a, b)
            | SmtExpr::BvAdd(a, b) | SmtExpr::BvSub(a, b) | SmtExpr::BvMul(a, b)
            | SmtExpr::BvUdiv(a, b) | SmtExpr::BvSdiv(a, b)
            | SmtExpr::BvUrem(a, b) | SmtExpr::BvSrem(a, b)
            | SmtExpr::BvAnd(a, b) | SmtExpr::BvOr(a, b) | SmtExpr::BvXor(a, b)
            | SmtExpr::BvShl(a, b) | SmtExpr::BvLshr(a, b) | SmtExpr::BvAshr(a, b)
            | SmtExpr::BvUlt(a, b) | SmtExpr::BvUle(a, b) | SmtExpr::BvUgt(a, b) | SmtExpr::BvUge(a, b)
            | SmtExpr::BvSlt(a, b) | SmtExpr::BvSle(a, b) | SmtExpr::BvSgt(a, b) | SmtExpr::BvSge(a, b)
            | SmtExpr::Concat(a, b) | SmtExpr::Select(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            SmtExpr::Ite(c, t, e) | SmtExpr::Store(c, t, e) => {
                c.collect_vars(vars);
                t.collect_vars(vars);
                e.collect_vars(vars);
            }
            SmtExpr::ForAll(bound, body) | SmtExpr::Exists(bound, body) => {
                let mut inner = std::collections::HashSet::new();
                body.collect_vars(&mut inner);
                for (name, _) in bound { inner.remove(name); }
                vars.extend(inner);
            }
            SmtExpr::Let(bindings, body) => {
                for (_, e) in bindings { e.collect_vars(vars); }
                let mut inner = std::collections::HashSet::new();
                body.collect_vars(&mut inner);
                for (name, _) in bindings { inner.remove(name); }
                vars.extend(inner);
            }
            SmtExpr::Apply(_, args) => {
                for a in args { a.collect_vars(vars); }
            }
            _ => {}
        }
    }

    /// Simplify this expression with basic rewriting rules.
    pub fn simplify(&self) -> SmtExpr {
        match self {
            SmtExpr::Not(inner) => {
                let s = inner.simplify();
                match s {
                    SmtExpr::BoolConst(b) => SmtExpr::BoolConst(!b),
                    SmtExpr::Not(inner2) => *inner2,
                    other => SmtExpr::not(other),
                }
            }
            SmtExpr::And(exprs) => {
                let simplified: Vec<SmtExpr> = exprs.iter()
                    .map(|e| e.simplify())
                    .filter(|e| *e != SmtExpr::BoolConst(true))
                    .collect();
                if simplified.iter().any(|e| *e == SmtExpr::BoolConst(false)) {
                    return SmtExpr::BoolConst(false);
                }
                if simplified.is_empty() {
                    SmtExpr::BoolConst(true)
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    SmtExpr::And(simplified)
                }
            }
            SmtExpr::Or(exprs) => {
                let simplified: Vec<SmtExpr> = exprs.iter()
                    .map(|e| e.simplify())
                    .filter(|e| *e != SmtExpr::BoolConst(false))
                    .collect();
                if simplified.iter().any(|e| *e == SmtExpr::BoolConst(true)) {
                    return SmtExpr::BoolConst(true);
                }
                if simplified.is_empty() {
                    SmtExpr::BoolConst(false)
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    SmtExpr::Or(simplified)
                }
            }
            SmtExpr::Ite(c, t, e) => {
                let sc = c.simplify();
                let st = t.simplify();
                let se = e.simplify();
                match &sc {
                    SmtExpr::BoolConst(true) => st,
                    SmtExpr::BoolConst(false) => se,
                    _ if st == se => st,
                    _ => SmtExpr::ite(sc, st, se),
                }
            }
            SmtExpr::Eq(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if sa == sb {
                    SmtExpr::BoolConst(true)
                } else {
                    SmtExpr::eq(sa, sb)
                }
            }
            SmtExpr::Implies(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (SmtExpr::BoolConst(false), _) => SmtExpr::BoolConst(true),
                    (_, SmtExpr::BoolConst(true)) => SmtExpr::BoolConst(true),
                    (SmtExpr::BoolConst(true), _) => sb,
                    _ => SmtExpr::implies(sa, sb),
                }
            }
            other => other.clone(),
        }
    }

    /// Pretty-print to SMT-LIB2 format.
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtExpr::Var(name, _) => name.clone(),
            SmtExpr::BoolConst(true) => "true".into(),
            SmtExpr::BoolConst(false) => "false".into(),
            SmtExpr::IntConst(n) => {
                if *n < 0 { format!("(- {})", -n) } else { format!("{}", n) }
            }
            SmtExpr::RealConst(num, den) => {
                if *den == 1 { format!("{}.0", num) } else { format!("(/ {}.0 {}.0)", num, den) }
            }
            SmtExpr::BvConst(val, width) => {
                format!("(_ bv{} {})", val, width)
            }
            SmtExpr::Not(e) => format!("(not {})", e.to_smtlib()),
            SmtExpr::And(es) => {
                let parts: Vec<_> = es.iter().map(|e| e.to_smtlib()).collect();
                format!("(and {})", parts.join(" "))
            }
            SmtExpr::Or(es) => {
                let parts: Vec<_> = es.iter().map(|e| e.to_smtlib()).collect();
                format!("(or {})", parts.join(" "))
            }
            SmtExpr::Implies(a, b) => format!("(=> {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Xor(a, b) => format!("(xor {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Iff(a, b) => format!("(= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Ite(c, t, e) => format!("(ite {} {} {})", c.to_smtlib(), t.to_smtlib(), e.to_smtlib()),
            SmtExpr::Eq(a, b) => format!("(= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Distinct(es) => {
                let parts: Vec<_> = es.iter().map(|e| e.to_smtlib()).collect();
                format!("(distinct {})", parts.join(" "))
            }
            SmtExpr::Lt(a, b) => format!("(< {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Le(a, b) => format!("(<= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Gt(a, b) => format!("(> {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Ge(a, b) => format!("(>= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Add(es) => {
                let parts: Vec<_> = es.iter().map(|e| e.to_smtlib()).collect();
                format!("(+ {})", parts.join(" "))
            }
            SmtExpr::Sub(a, b) => format!("(- {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Mul(es) => {
                let parts: Vec<_> = es.iter().map(|e| e.to_smtlib()).collect();
                format!("(* {})", parts.join(" "))
            }
            SmtExpr::Div(a, b) => format!("(div {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Mod(a, b) => format!("(mod {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Neg(e) => format!("(- {})", e.to_smtlib()),
            SmtExpr::Abs(e) => format!("(abs {})", e.to_smtlib()),
            SmtExpr::BvAdd(a, b) => format!("(bvadd {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSub(a, b) => format!("(bvsub {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvMul(a, b) => format!("(bvmul {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUdiv(a, b) => format!("(bvudiv {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSdiv(a, b) => format!("(bvsdiv {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUrem(a, b) => format!("(bvurem {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSrem(a, b) => format!("(bvsrem {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvAnd(a, b) => format!("(bvand {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvOr(a, b) => format!("(bvor {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvXor(a, b) => format!("(bvxor {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvNot(e) => format!("(bvnot {})", e.to_smtlib()),
            SmtExpr::BvNeg(e) => format!("(bvneg {})", e.to_smtlib()),
            SmtExpr::BvShl(a, b) => format!("(bvshl {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvLshr(a, b) => format!("(bvlshr {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvAshr(a, b) => format!("(bvashr {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUlt(a, b) => format!("(bvult {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUle(a, b) => format!("(bvule {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUgt(a, b) => format!("(bvugt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvUge(a, b) => format!("(bvuge {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSlt(a, b) => format!("(bvslt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSle(a, b) => format!("(bvsle {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSgt(a, b) => format!("(bvsgt {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::BvSge(a, b) => format!("(bvsge {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Concat(a, b) => format!("(concat {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Extract(h, l, e) => format!("((_ extract {} {}) {})", h, l, e.to_smtlib()),
            SmtExpr::ZeroExt(n, e) => format!("((_ zero_extend {}) {})", n, e.to_smtlib()),
            SmtExpr::SignExt(n, e) => format!("((_ sign_extend {}) {})", n, e.to_smtlib()),
            SmtExpr::Select(a, i) => format!("(select {} {})", a.to_smtlib(), i.to_smtlib()),
            SmtExpr::Store(a, i, v) => format!("(store {} {} {})", a.to_smtlib(), i.to_smtlib(), v.to_smtlib()),
            SmtExpr::ForAll(vars, body) => {
                let vdecl: Vec<_> = vars.iter().map(|(n, s)| format!("({} {})", n, s.to_smtlib())).collect();
                format!("(forall ({}) {})", vdecl.join(" "), body.to_smtlib())
            }
            SmtExpr::Exists(vars, body) => {
                let vdecl: Vec<_> = vars.iter().map(|(n, s)| format!("({} {})", n, s.to_smtlib())).collect();
                format!("(exists ({}) {})", vdecl.join(" "), body.to_smtlib())
            }
            SmtExpr::Let(bindings, body) => {
                let bdecl: Vec<_> = bindings.iter().map(|(n, e)| format!("({} {})", n, e.to_smtlib())).collect();
                format!("(let ({}) {})", bdecl.join(" "), body.to_smtlib())
            }
            SmtExpr::Apply(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let parts: Vec<_> = args.iter().map(|e| e.to_smtlib()).collect();
                    format!("({} {})", name, parts.join(" "))
                }
            }
        }
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ── SMT Formula ──────────────────────────────────────────────────────────

/// An SMT formula with declarations and assertions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtFormula {
    pub logic: Option<String>,
    pub declarations: Vec<SmtDeclaration>,
    pub assertions: Vec<SmtExpr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtDeclaration {
    DeclareSort(String, u32),
    DeclareFun(String, Vec<SmtSort>, SmtSort),
    DeclareConst(String, SmtSort),
    DefineFun(String, Vec<(String, SmtSort)>, SmtSort, SmtExpr),
}

impl SmtFormula {
    pub fn new() -> Self {
        SmtFormula {
            logic: None,
            declarations: Vec::new(),
            assertions: Vec::new(),
        }
    }

    pub fn with_logic(logic: impl Into<String>) -> Self {
        SmtFormula {
            logic: Some(logic.into()),
            declarations: Vec::new(),
            assertions: Vec::new(),
        }
    }

    pub fn declare_const(&mut self, name: impl Into<String>, sort: SmtSort) {
        self.declarations.push(SmtDeclaration::DeclareConst(name.into(), sort));
    }

    pub fn declare_fun(&mut self, name: impl Into<String>, args: Vec<SmtSort>, ret: SmtSort) {
        self.declarations.push(SmtDeclaration::DeclareFun(name.into(), args, ret));
    }

    pub fn assert(&mut self, expr: SmtExpr) {
        self.assertions.push(expr);
    }

    pub fn assert_all(&mut self, exprs: impl IntoIterator<Item = SmtExpr>) {
        self.assertions.extend(exprs);
    }

    /// Render to complete SMT-LIB2 script.
    pub fn to_smtlib2(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref logic) = self.logic {
            lines.push(format!("(set-logic {})", logic));
        }

        for decl in &self.declarations {
            match decl {
                SmtDeclaration::DeclareSort(name, arity) => {
                    lines.push(format!("(declare-sort {} {})", name, arity));
                }
                SmtDeclaration::DeclareFun(name, args, ret) => {
                    let arg_strs: Vec<_> = args.iter().map(|s| s.to_smtlib()).collect();
                    lines.push(format!(
                        "(declare-fun {} ({}) {})",
                        name,
                        arg_strs.join(" "),
                        ret.to_smtlib()
                    ));
                }
                SmtDeclaration::DeclareConst(name, sort) => {
                    lines.push(format!("(declare-const {} {})", name, sort.to_smtlib()));
                }
                SmtDeclaration::DefineFun(name, params, ret, body) => {
                    let param_strs: Vec<_> = params
                        .iter()
                        .map(|(n, s)| format!("({} {})", n, s.to_smtlib()))
                        .collect();
                    lines.push(format!(
                        "(define-fun {} ({}) {} {})",
                        name,
                        param_strs.join(" "),
                        ret.to_smtlib(),
                        body.to_smtlib()
                    ));
                }
            }
        }

        for assertion in &self.assertions {
            lines.push(format!("(assert {})", assertion.to_smtlib()));
        }

        lines.push("(check-sat)".into());
        lines.push("(get-model)".into());
        lines.join("\n")
    }

    /// Number of assertions.
    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }

    /// Total nodes across all assertions.
    pub fn total_nodes(&self) -> usize {
        self.assertions.iter().map(|a| a.node_count()).sum()
    }
}

impl Default for SmtFormula {
    fn default() -> Self {
        Self::new()
    }
}

// ── SMT Result ───────────────────────────────────────────────────────────

/// Result of an SMT solver check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtResult {
    Sat(SmtModel),
    Unsat(SmtProof),
    Unknown(String),
}

impl SmtResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat(_))
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, SmtResult::Unknown(_))
    }

    pub fn model(&self) -> Option<&SmtModel> {
        match self {
            SmtResult::Sat(m) => Some(m),
            _ => None,
        }
    }

    pub fn proof(&self) -> Option<&SmtProof> {
        match self {
            SmtResult::Unsat(p) => Some(p),
            _ => None,
        }
    }
}

impl fmt::Display for SmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtResult::Sat(model) => write!(f, "sat ({} assignments)", model.assignments.len()),
            SmtResult::Unsat(proof) => write!(f, "unsat ({} core clauses)", proof.unsat_core.len()),
            SmtResult::Unknown(reason) => write!(f, "unknown ({})", reason),
        }
    }
}

// ── SMT Model ────────────────────────────────────────────────────────────

/// A satisfying assignment (model) from the SMT solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: HashMap<String, SmtValue>,
}

/// Concrete values in a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Int(i64),
    BitVec { value: u64, width: u32 },
    Real(i64, u64),
    Array(Vec<(SmtValue, SmtValue)>, Box<SmtValue>),
}

impl SmtModel {
    pub fn new() -> Self {
        SmtModel {
            assignments: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: SmtValue) {
        self.assignments.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&SmtValue> {
        self.assignments.get(name)
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.get(name) {
            Some(SmtValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.get(name) {
            Some(SmtValue::Int(n)) => Some(*n),
            _ => None,
        }
    }

    pub fn get_bv(&self, name: &str) -> Option<(u64, u32)> {
        match self.get(name) {
            Some(SmtValue::BitVec { value, width }) => Some((*value, *width)),
            _ => None,
        }
    }

    /// Number of variable assignments.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Render to SMT-LIB2 model format.
    pub fn to_smtlib2(&self) -> String {
        let mut lines = vec!["(model".to_string()];
        let mut keys: Vec<&String> = self.assignments.keys().collect();
        keys.sort();
        for name in keys {
            let val = &self.assignments[name];
            lines.push(format!("  (define-fun {} () {} {})", name, val.sort_str(), val.to_smtlib()));
        }
        lines.push(")".to_string());
        lines.join("\n")
    }
}

impl Default for SmtModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtValue {
    fn sort_str(&self) -> String {
        match self {
            SmtValue::Bool(_) => "Bool".into(),
            SmtValue::Int(_) => "Int".into(),
            SmtValue::BitVec { width, .. } => format!("(_ BitVec {})", width),
            SmtValue::Real(_, _) => "Real".into(),
            SmtValue::Array(_, _) => "Array".into(),
        }
    }

    fn to_smtlib(&self) -> String {
        match self {
            SmtValue::Bool(true) => "true".into(),
            SmtValue::Bool(false) => "false".into(),
            SmtValue::Int(n) => {
                if *n < 0 { format!("(- {})", -n) } else { format!("{}", n) }
            }
            SmtValue::BitVec { value, width } => format!("(_ bv{} {})", value, width),
            SmtValue::Real(num, den) => {
                if *den == 1 { format!("{}.0", num) } else { format!("(/ {}.0 {}.0)", num, den) }
            }
            SmtValue::Array(entries, default) => {
                let mut result = format!("((as const) {})", default.to_smtlib());
                for (k, v) in entries {
                    result = format!("(store {} {} {})", result, k.to_smtlib(), v.to_smtlib());
                }
                result
            }
        }
    }
}

impl fmt::Display for SmtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ── SMT Proof ────────────────────────────────────────────────────────────

/// An UNSAT certificate from the SMT solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtProof {
    pub unsat_core: Vec<String>,
    pub proof_steps: Vec<ProofStep>,
}

/// A step in an UNSAT proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    pub id: usize,
    pub rule: ProofRule,
    pub premises: Vec<usize>,
    pub conclusion: SmtExpr,
}

/// Proof rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofRule {
    Assumption,
    Resolution,
    UnitPropagation,
    TheoryLemma(String),
    Rewrite,
    Congruence,
    Transitivity,
    Reflexivity,
    Symmetry,
    ModusPonens,
    AndElim(usize),
    OrIntro(usize),
    NotElim,
    Custom(String),
}

impl SmtProof {
    pub fn new() -> Self {
        SmtProof {
            unsat_core: Vec::new(),
            proof_steps: Vec::new(),
        }
    }

    pub fn with_core(core: Vec<String>) -> Self {
        SmtProof {
            unsat_core: core,
            proof_steps: Vec::new(),
        }
    }

    pub fn add_step(&mut self, rule: ProofRule, premises: Vec<usize>, conclusion: SmtExpr) -> usize {
        let id = self.proof_steps.len();
        self.proof_steps.push(ProofStep {
            id,
            rule,
            premises,
            conclusion,
        });
        id
    }

    /// Number of proof steps.
    pub fn step_count(&self) -> usize {
        self.proof_steps.len()
    }

    /// Validate that all premise references are valid.
    pub fn is_well_formed(&self) -> bool {
        for step in &self.proof_steps {
            for &premise in &step.premises {
                if premise >= step.id {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for SmtProof {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_display() {
        assert_eq!(SmtSort::Bool.to_smtlib(), "Bool");
        assert_eq!(SmtSort::BitVec(32).to_smtlib(), "(_ BitVec 32)");
        let arr = SmtSort::array(SmtSort::BitVec(32), SmtSort::BitVec(8));
        assert_eq!(arr.to_smtlib(), "(Array (_ BitVec 32) (_ BitVec 8))");
    }

    #[test]
    fn test_expr_smtlib() {
        let x = SmtExpr::bv_var("x", 32);
        assert_eq!(x.to_smtlib(), "x");

        let bv = SmtExpr::BvConst(42, 32);
        assert_eq!(bv.to_smtlib(), "(_ bv42 32)");

        let add = SmtExpr::bvadd(x.clone(), bv.clone());
        assert_eq!(add.to_smtlib(), "(bvadd x (_ bv42 32))");
    }

    #[test]
    fn test_expr_simplify() {
        let e = SmtExpr::And(vec![SmtExpr::BoolConst(true), SmtExpr::bool_var("p")]);
        let s = e.simplify();
        assert_eq!(s, SmtExpr::Var("p".into(), SmtSort::Bool));

        let e = SmtExpr::Or(vec![SmtExpr::BoolConst(true), SmtExpr::bool_var("p")]);
        let s = e.simplify();
        assert_eq!(s, SmtExpr::BoolConst(true));
    }

    #[test]
    fn test_not_simplify() {
        let e = SmtExpr::not(SmtExpr::BoolConst(true));
        assert_eq!(e.simplify(), SmtExpr::BoolConst(false));

        let e = SmtExpr::not(SmtExpr::not(SmtExpr::bool_var("p")));
        let s = e.simplify();
        assert_eq!(s, SmtExpr::Var("p".into(), SmtSort::Bool));
    }

    #[test]
    fn test_ite_simplify() {
        let e = SmtExpr::ite(SmtExpr::BoolConst(true), SmtExpr::IntConst(1), SmtExpr::IntConst(2));
        assert_eq!(e.simplify(), SmtExpr::IntConst(1));
    }

    #[test]
    fn test_eq_simplify() {
        let x = SmtExpr::int_var("x");
        let e = SmtExpr::eq(x.clone(), x.clone());
        assert_eq!(e.simplify(), SmtExpr::BoolConst(true));
    }

    #[test]
    fn test_formula_smtlib2() {
        let mut formula = SmtFormula::with_logic("QF_BV");
        formula.declare_const("x", SmtSort::BitVec(32));
        formula.declare_const("y", SmtSort::BitVec(32));
        formula.assert(SmtExpr::eq(
            SmtExpr::bvadd(SmtExpr::bv_var("x", 32), SmtExpr::bv_var("y", 32)),
            SmtExpr::BvConst(42, 32),
        ));
        let output = formula.to_smtlib2();
        assert!(output.contains("(set-logic QF_BV)"));
        assert!(output.contains("(declare-const x (_ BitVec 32))"));
        assert!(output.contains("(check-sat)"));
    }

    #[test]
    fn test_node_count() {
        let leaf = SmtExpr::IntConst(1);
        assert_eq!(leaf.node_count(), 1);

        let expr = SmtExpr::eq(SmtExpr::int_var("x"), SmtExpr::IntConst(5));
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_free_variables() {
        let e = SmtExpr::eq(SmtExpr::int_var("x"), SmtExpr::Add(vec![SmtExpr::int_var("y"), SmtExpr::IntConst(1)]));
        let vars = e.free_variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_forall_bound_vars() {
        let body = SmtExpr::eq(SmtExpr::int_var("x"), SmtExpr::int_var("y"));
        let forall = SmtExpr::ForAll(
            vec![("x".into(), SmtSort::Int)],
            Box::new(body),
        );
        let vars = forall.free_variables();
        assert!(!vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_smt_model() {
        let mut model = SmtModel::new();
        model.set("x", SmtValue::BitVec { value: 42, width: 32 });
        model.set("flag", SmtValue::Bool(true));

        assert_eq!(model.get_bool("flag"), Some(true));
        assert_eq!(model.get_bv("x"), Some((42, 32)));
        assert_eq!(model.len(), 2);
    }

    #[test]
    fn test_smt_result() {
        let mut model = SmtModel::new();
        model.set("x", SmtValue::Int(42));
        let result = SmtResult::Sat(model);
        assert!(result.is_sat());
        assert!(result.model().is_some());

        let result = SmtResult::Unknown("timeout".into());
        assert!(result.is_unknown());
    }

    #[test]
    fn test_proof_well_formed() {
        let mut proof = SmtProof::new();
        proof.add_step(ProofRule::Assumption, vec![], SmtExpr::bool_var("p"));
        proof.add_step(ProofRule::Assumption, vec![], SmtExpr::not(SmtExpr::bool_var("p")));
        proof.add_step(ProofRule::Resolution, vec![0, 1], SmtExpr::BoolConst(false));
        assert!(proof.is_well_formed());
    }

    #[test]
    fn test_proof_not_well_formed() {
        let mut proof = SmtProof::new();
        proof.proof_steps.push(ProofStep {
            id: 0,
            rule: ProofRule::Resolution,
            premises: vec![1], // references future step
            conclusion: SmtExpr::BoolConst(false),
        });
        assert!(!proof.is_well_formed());
    }

    #[test]
    fn test_implies_simplify() {
        let e = SmtExpr::implies(SmtExpr::BoolConst(false), SmtExpr::bool_var("q"));
        assert_eq!(e.simplify(), SmtExpr::BoolConst(true));

        let e = SmtExpr::implies(SmtExpr::BoolConst(true), SmtExpr::bool_var("q"));
        assert_eq!(e.simplify(), SmtExpr::Var("q".into(), SmtSort::Bool));
    }

    #[test]
    fn test_model_smtlib2() {
        let mut model = SmtModel::new();
        model.set("x", SmtValue::Int(42));
        let output = model.to_smtlib2();
        assert!(output.contains("(model"));
        assert!(output.contains("42"));
    }
}
