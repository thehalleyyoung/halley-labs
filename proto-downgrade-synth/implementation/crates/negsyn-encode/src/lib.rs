//! # negsyn-encode: Dolev-Yao + SMT Constraint Encoding
//!
//! Implements ALG4: DYENCODE — the core encoding of protocol negotiation
//! state machines into SMT formulas for automated downgrade attack synthesis.
//!
//! The encoding combines:
//! - **Dolev-Yao message algebra** (Definition D4): symbolic message
//!   construction and deconstruction with adversary deduction rules.
//! - **SMT bitvector theory**: efficient encoding of cipher suite IDs,
//!   protocol versions, and security orderings.
//! - **Bounded model checking**: unrolling the negotiation LTS up to
//!   depth k with adversary budget n.
//! - **Property negation**: encoding downgrade freedom (Definition D5)
//!   as a satisfiability query.
//!
//! # Architecture
//!
//! ```text
//! NegotiationLTS + AdversaryBudget
//!         │
//!         ▼
//!    ┌──────────┐
//!    │ DYEncoder│── orchestrates ──┐
//!    └──────────┘                  │
//!         │                        │
//!    ┌────┴────┐    ┌──────────┐   │
//!    │Unrolling│    │Dolev-Yao │   │
//!    │ Engine  │    │  Algebra │   │
//!    └────┬────┘    └────┬─────┘   │
//!         │              │         │
//!    ┌────┴────┐    ┌────┴─────┐   │
//!    │Bitvector│    │Adversary │   │
//!    │Encoding │    │ Encoding │   │
//!    └────┬────┘    └────┬─────┘   │
//!         │              │         │
//!    ┌────┴──────────────┴─────┐   │
//!    │   Property Encoding     │◄──┘
//!    └────────────┬────────────┘
//!                 │
//!            ┌────┴────┐
//!            │SmtLib2  │
//!            │ Writer  │
//!            └─────────┘
//! ```

pub mod adversary_encoding;
pub mod bitvector;
pub mod dolev_yao;
pub mod encoder;
pub mod optimization;
pub mod property;
pub mod smtlib;
pub mod unrolling;

pub use adversary_encoding::{AdversaryEncoder, AdversaryEncoderConfig};
pub use bitvector::{BvEncoder, BvSort};
pub use dolev_yao::{DYTermAlgebra, DeductionRules, KnowledgeEncoder, TermEncoder};
pub use encoder::{DYEncoder, DyEncodeResult, EncoderConfig};
pub use optimization::{EncodingOptimizer, OptimizationConfig};
pub use property::{
    DowngradeProperty, ExtensionStripping, HonestOutcome, PropertyEncoder, SecurityOrdering,
    VersionDowngrade,
};
pub use smtlib::{SmtLib2Writer, WriterConfig};
pub use unrolling::{TimeStep, UnrollingEngine};

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ─── SMT sort declarations ──────────────────────────────────────────────

/// SMT sort (type) declarations used throughout the encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    BitVec(u32),
    Int,
    Array(Box<SmtSort>, Box<SmtSort>),
    Uninterpreted(String),
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::BitVec(w) => write!(f, "(_ BitVec {})", w),
            SmtSort::Int => write!(f, "Int"),
            SmtSort::Array(idx, elem) => write!(f, "(Array {} {})", idx, elem),
            SmtSort::Uninterpreted(name) => write!(f, "{}", name),
        }
    }
}

// ─── SMT expression AST ─────────────────────────────────────────────────

/// An SMT expression node forming the core formula representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtExpr {
    BoolLit(bool),
    BvLit(u64, u32),
    IntLit(i64),
    Var(String),
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Distinct(Vec<SmtExpr>),
    BvUlt(Box<SmtExpr>, Box<SmtExpr>),
    BvUle(Box<SmtExpr>, Box<SmtExpr>),
    BvUgt(Box<SmtExpr>, Box<SmtExpr>),
    BvSlt(Box<SmtExpr>, Box<SmtExpr>),
    BvAdd(Box<SmtExpr>, Box<SmtExpr>),
    BvSub(Box<SmtExpr>, Box<SmtExpr>),
    BvMul(Box<SmtExpr>, Box<SmtExpr>),
    BvAnd(Box<SmtExpr>, Box<SmtExpr>),
    BvOr(Box<SmtExpr>, Box<SmtExpr>),
    BvXor(Box<SmtExpr>, Box<SmtExpr>),
    BvNot(Box<SmtExpr>),
    BvShl(Box<SmtExpr>, Box<SmtExpr>),
    BvLShr(Box<SmtExpr>, Box<SmtExpr>),
    Extract(Box<SmtExpr>, u32, u32),
    Concat(Box<SmtExpr>, Box<SmtExpr>),
    ZeroExtend(Box<SmtExpr>, u32),
    SignExtend(Box<SmtExpr>, u32),
    IntAdd(Vec<SmtExpr>),
    IntSub(Box<SmtExpr>, Box<SmtExpr>),
    IntMul(Box<SmtExpr>, Box<SmtExpr>),
    IntLe(Box<SmtExpr>, Box<SmtExpr>),
    IntLt(Box<SmtExpr>, Box<SmtExpr>),
    IntGe(Box<SmtExpr>, Box<SmtExpr>),
    Select(Box<SmtExpr>, Box<SmtExpr>),
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Apply(String, Vec<SmtExpr>),
    Let(Vec<(String, SmtExpr)>, Box<SmtExpr>),
    Exists(Vec<(String, SmtSort)>, Box<SmtExpr>),
    ForAll(Vec<(String, SmtSort)>, Box<SmtExpr>),
}

impl SmtExpr {
    pub fn bool_lit(b: bool) -> Self {
        SmtExpr::BoolLit(b)
    }

    pub fn bv_lit(val: u64, width: u32) -> Self {
        SmtExpr::BvLit(val, width)
    }

    pub fn var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into())
    }

    pub fn not(e: SmtExpr) -> Self {
        match e {
            SmtExpr::Not(inner) => *inner,
            SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
            other => SmtExpr::Not(Box::new(other)),
        }
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        let mut flat = Vec::with_capacity(exprs.len());
        for e in exprs {
            match e {
                SmtExpr::BoolLit(true) => continue,
                SmtExpr::BoolLit(false) => return SmtExpr::BoolLit(false),
                SmtExpr::And(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => SmtExpr::BoolLit(true),
            1 => flat.into_iter().next().unwrap(),
            _ => SmtExpr::And(flat),
        }
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        let mut flat = Vec::with_capacity(exprs.len());
        for e in exprs {
            match e {
                SmtExpr::BoolLit(false) => continue,
                SmtExpr::BoolLit(true) => return SmtExpr::BoolLit(true),
                SmtExpr::Or(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        match flat.len() {
            0 => SmtExpr::BoolLit(false),
            1 => flat.into_iter().next().unwrap(),
            _ => SmtExpr::Or(flat),
        }
    }

    pub fn implies(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        match (&lhs, &rhs) {
            (SmtExpr::BoolLit(false), _) => SmtExpr::BoolLit(true),
            (SmtExpr::BoolLit(true), _) => rhs,
            (_, SmtExpr::BoolLit(true)) => SmtExpr::BoolLit(true),
            _ => SmtExpr::Implies(Box::new(lhs), Box::new(rhs)),
        }
    }

    pub fn ite(c: SmtExpr, t: SmtExpr, e: SmtExpr) -> Self {
        match &c {
            SmtExpr::BoolLit(true) => t,
            SmtExpr::BoolLit(false) => e,
            _ => {
                if t == e {
                    t
                } else {
                    SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e))
                }
            }
        }
    }

    pub fn eq(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        if lhs == rhs {
            SmtExpr::BoolLit(true)
        } else {
            SmtExpr::Eq(Box::new(lhs), Box::new(rhs))
        }
    }

    pub fn bv_ult(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvUlt(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_ule(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvUle(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_add(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvAdd(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_sub(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvSub(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_and(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvAnd(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_or(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvOr(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_xor(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::BvXor(Box::new(lhs), Box::new(rhs))
    }

    pub fn bv_not(e: SmtExpr) -> Self {
        SmtExpr::BvNot(Box::new(e))
    }

    pub fn select(arr: SmtExpr, idx: SmtExpr) -> Self {
        SmtExpr::Select(Box::new(arr), Box::new(idx))
    }

    pub fn store(arr: SmtExpr, idx: SmtExpr, val: SmtExpr) -> Self {
        SmtExpr::Store(Box::new(arr), Box::new(idx), Box::new(val))
    }

    pub fn int_add(exprs: Vec<SmtExpr>) -> Self {
        let filtered: Vec<_> = exprs
            .into_iter()
            .filter(|e| !matches!(e, SmtExpr::IntLit(0)))
            .collect();
        match filtered.len() {
            0 => SmtExpr::IntLit(0),
            1 => filtered.into_iter().next().unwrap(),
            _ => SmtExpr::IntAdd(filtered),
        }
    }

    pub fn int_le(lhs: SmtExpr, rhs: SmtExpr) -> Self {
        SmtExpr::IntLe(Box::new(lhs), Box::new(rhs))
    }

    /// Count nodes in the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            SmtExpr::BoolLit(_) | SmtExpr::BvLit(..) | SmtExpr::IntLit(_) | SmtExpr::Var(_) => 1,
            SmtExpr::Not(a)
            | SmtExpr::BvNot(a)
            | SmtExpr::Extract(a, _, _)
            | SmtExpr::ZeroExtend(a, _)
            | SmtExpr::SignExtend(a, _) => 1 + a.node_count(),
            SmtExpr::And(es)
            | SmtExpr::Or(es)
            | SmtExpr::IntAdd(es)
            | SmtExpr::Distinct(es) => 1 + es.iter().map(|e| e.node_count()).sum::<usize>(),
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::BvUlt(a, b)
            | SmtExpr::BvUle(a, b)
            | SmtExpr::BvUgt(a, b)
            | SmtExpr::BvSlt(a, b)
            | SmtExpr::BvAdd(a, b)
            | SmtExpr::BvSub(a, b)
            | SmtExpr::BvMul(a, b)
            | SmtExpr::BvAnd(a, b)
            | SmtExpr::BvOr(a, b)
            | SmtExpr::BvXor(a, b)
            | SmtExpr::BvShl(a, b)
            | SmtExpr::BvLShr(a, b)
            | SmtExpr::Concat(a, b)
            | SmtExpr::IntSub(a, b)
            | SmtExpr::IntMul(a, b)
            | SmtExpr::IntLe(a, b)
            | SmtExpr::IntLt(a, b)
            | SmtExpr::IntGe(a, b)
            | SmtExpr::Select(a, b) => 1 + a.node_count() + b.node_count(),
            SmtExpr::Ite(c, t, e) | SmtExpr::Store(c, t, e) => {
                1 + c.node_count() + t.node_count() + e.node_count()
            }
            SmtExpr::Apply(_, args) => 1 + args.iter().map(|e| e.node_count()).sum::<usize>(),
            SmtExpr::Let(bindings, body) => {
                1 + bindings.iter().map(|(_, e)| e.node_count()).sum::<usize>() + body.node_count()
            }
            SmtExpr::Exists(_, body) | SmtExpr::ForAll(_, body) => 1 + body.node_count(),
        }
    }

    /// Collect all free variable names in the expression.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_free_vars(&mut vars, &BTreeSet::new());
        vars
    }

    fn collect_free_vars(&self, free: &mut BTreeSet<String>, bound: &BTreeSet<String>) {
        match self {
            SmtExpr::Var(name) => {
                if !bound.contains(name) {
                    free.insert(name.clone());
                }
            }
            SmtExpr::BoolLit(_) | SmtExpr::BvLit(..) | SmtExpr::IntLit(_) => {}
            SmtExpr::Not(a)
            | SmtExpr::BvNot(a)
            | SmtExpr::Extract(a, _, _)
            | SmtExpr::ZeroExtend(a, _)
            | SmtExpr::SignExtend(a, _) => a.collect_free_vars(free, bound),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::IntAdd(es) | SmtExpr::Distinct(es) => {
                for e in es {
                    e.collect_free_vars(free, bound);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::BvUlt(a, b)
            | SmtExpr::BvUle(a, b)
            | SmtExpr::BvUgt(a, b)
            | SmtExpr::BvSlt(a, b)
            | SmtExpr::BvAdd(a, b)
            | SmtExpr::BvSub(a, b)
            | SmtExpr::BvMul(a, b)
            | SmtExpr::BvAnd(a, b)
            | SmtExpr::BvOr(a, b)
            | SmtExpr::BvXor(a, b)
            | SmtExpr::BvShl(a, b)
            | SmtExpr::BvLShr(a, b)
            | SmtExpr::Concat(a, b)
            | SmtExpr::IntSub(a, b)
            | SmtExpr::IntMul(a, b)
            | SmtExpr::IntLe(a, b)
            | SmtExpr::IntLt(a, b)
            | SmtExpr::IntGe(a, b)
            | SmtExpr::Select(a, b) => {
                a.collect_free_vars(free, bound);
                b.collect_free_vars(free, bound);
            }
            SmtExpr::Ite(c, t, e) | SmtExpr::Store(c, t, e) => {
                c.collect_free_vars(free, bound);
                t.collect_free_vars(free, bound);
                e.collect_free_vars(free, bound);
            }
            SmtExpr::Apply(_, args) => {
                for a in args {
                    a.collect_free_vars(free, bound);
                }
            }
            SmtExpr::Let(bindings, body) => {
                let mut new_bound = bound.clone();
                for (name, expr) in bindings {
                    expr.collect_free_vars(free, bound);
                    new_bound.insert(name.clone());
                }
                body.collect_free_vars(free, &new_bound);
            }
            SmtExpr::Exists(vars, body) | SmtExpr::ForAll(vars, body) => {
                let mut new_bound = bound.clone();
                for (name, _) in vars {
                    new_bound.insert(name.clone());
                }
                body.collect_free_vars(free, &new_bound);
            }
        }
    }

    /// Substitute a variable with an expression throughout.
    pub fn substitute(&self, var: &str, replacement: &SmtExpr) -> SmtExpr {
        match self {
            SmtExpr::Var(name) if name == var => replacement.clone(),
            SmtExpr::Var(_) | SmtExpr::BoolLit(_) | SmtExpr::BvLit(..) | SmtExpr::IntLit(_) => {
                self.clone()
            }
            SmtExpr::Not(a) => SmtExpr::not(a.substitute(var, replacement)),
            SmtExpr::And(es) => {
                SmtExpr::and(es.iter().map(|e| e.substitute(var, replacement)).collect())
            }
            SmtExpr::Or(es) => {
                SmtExpr::or(es.iter().map(|e| e.substitute(var, replacement)).collect())
            }
            SmtExpr::Implies(a, b) => SmtExpr::implies(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::Eq(a, b) => SmtExpr::eq(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::Ite(c, t, e) => SmtExpr::ite(
                c.substitute(var, replacement),
                t.substitute(var, replacement),
                e.substitute(var, replacement),
            ),
            SmtExpr::BvUlt(a, b) => SmtExpr::bv_ult(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvUle(a, b) => SmtExpr::bv_ule(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvAdd(a, b) => SmtExpr::bv_add(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvSub(a, b) => SmtExpr::bv_sub(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvAnd(a, b) => SmtExpr::bv_and(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvOr(a, b) => SmtExpr::bv_or(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvXor(a, b) => SmtExpr::bv_xor(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::BvNot(a) => SmtExpr::bv_not(a.substitute(var, replacement)),
            SmtExpr::Extract(a, hi, lo) => {
                SmtExpr::Extract(Box::new(a.substitute(var, replacement)), *hi, *lo)
            }
            SmtExpr::Concat(a, b) => SmtExpr::Concat(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            SmtExpr::ZeroExtend(a, w) => {
                SmtExpr::ZeroExtend(Box::new(a.substitute(var, replacement)), *w)
            }
            SmtExpr::SignExtend(a, w) => {
                SmtExpr::SignExtend(Box::new(a.substitute(var, replacement)), *w)
            }
            SmtExpr::IntAdd(es) => {
                SmtExpr::int_add(es.iter().map(|e| e.substitute(var, replacement)).collect())
            }
            SmtExpr::IntLe(a, b) => SmtExpr::int_le(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::Select(a, b) => SmtExpr::select(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
            ),
            SmtExpr::Store(a, b, c) => SmtExpr::store(
                a.substitute(var, replacement),
                b.substitute(var, replacement),
                c.substitute(var, replacement),
            ),
            SmtExpr::Apply(name, args) => SmtExpr::Apply(
                name.clone(),
                args.iter().map(|a| a.substitute(var, replacement)).collect(),
            ),
            SmtExpr::Let(bindings, body) => {
                let new_bindings: Vec<_> = bindings
                    .iter()
                    .map(|(n, e)| (n.clone(), e.substitute(var, replacement)))
                    .collect();
                if bindings.iter().any(|(n, _)| n == var) {
                    SmtExpr::Let(new_bindings, body.clone())
                } else {
                    SmtExpr::Let(new_bindings, Box::new(body.substitute(var, replacement)))
                }
            }
            SmtExpr::Exists(vars, body) | SmtExpr::ForAll(vars, body) => {
                if vars.iter().any(|(n, _)| n == var) {
                    self.clone()
                } else {
                    let new_body = body.substitute(var, replacement);
                    match self {
                        SmtExpr::Exists(..) => SmtExpr::Exists(vars.clone(), Box::new(new_body)),
                        _ => SmtExpr::ForAll(vars.clone(), Box::new(new_body)),
                    }
                }
            }
            _ => {
                // Remaining binary ops
                self.clone()
            }
        }
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtExpr::BoolLit(true) => write!(f, "true"),
            SmtExpr::BoolLit(false) => write!(f, "false"),
            SmtExpr::BvLit(v, w) => write!(f, "(_ bv{} {})", v, w),
            SmtExpr::IntLit(v) => {
                if *v < 0 {
                    write!(f, "(- {})", -v)
                } else {
                    write!(f, "{}", v)
                }
            }
            SmtExpr::Var(name) => write!(f, "{}", name),
            SmtExpr::Not(a) => write!(f, "(not {})", a),
            SmtExpr::And(es) => {
                write!(f, "(and")?;
                for e in es {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            }
            SmtExpr::Or(es) => {
                write!(f, "(or")?;
                for e in es {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            }
            SmtExpr::Implies(a, b) => write!(f, "(=> {} {})", a, b),
            SmtExpr::Ite(c, t, e) => write!(f, "(ite {} {} {})", c, t, e),
            SmtExpr::Eq(a, b) => write!(f, "(= {} {})", a, b),
            SmtExpr::Distinct(es) => {
                write!(f, "(distinct")?;
                for e in es {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            }
            SmtExpr::BvUlt(a, b) => write!(f, "(bvult {} {})", a, b),
            SmtExpr::BvUle(a, b) => write!(f, "(bvule {} {})", a, b),
            SmtExpr::BvUgt(a, b) => write!(f, "(bvugt {} {})", a, b),
            SmtExpr::BvSlt(a, b) => write!(f, "(bvslt {} {})", a, b),
            SmtExpr::BvAdd(a, b) => write!(f, "(bvadd {} {})", a, b),
            SmtExpr::BvSub(a, b) => write!(f, "(bvsub {} {})", a, b),
            SmtExpr::BvMul(a, b) => write!(f, "(bvmul {} {})", a, b),
            SmtExpr::BvAnd(a, b) => write!(f, "(bvand {} {})", a, b),
            SmtExpr::BvOr(a, b) => write!(f, "(bvor {} {})", a, b),
            SmtExpr::BvXor(a, b) => write!(f, "(bvxor {} {})", a, b),
            SmtExpr::BvNot(a) => write!(f, "(bvnot {})", a),
            SmtExpr::BvShl(a, b) => write!(f, "(bvshl {} {})", a, b),
            SmtExpr::BvLShr(a, b) => write!(f, "(bvlshr {} {})", a, b),
            SmtExpr::Extract(a, hi, lo) => write!(f, "((_ extract {} {}) {})", hi, lo, a),
            SmtExpr::Concat(a, b) => write!(f, "(concat {} {})", a, b),
            SmtExpr::ZeroExtend(a, w) => write!(f, "((_ zero_extend {}) {})", w, a),
            SmtExpr::SignExtend(a, w) => write!(f, "((_ sign_extend {}) {})", w, a),
            SmtExpr::IntAdd(es) => {
                write!(f, "(+")?;
                for e in es {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            }
            SmtExpr::IntSub(a, b) => write!(f, "(- {} {})", a, b),
            SmtExpr::IntMul(a, b) => write!(f, "(* {} {})", a, b),
            SmtExpr::IntLe(a, b) => write!(f, "(<= {} {})", a, b),
            SmtExpr::IntLt(a, b) => write!(f, "(< {} {})", a, b),
            SmtExpr::IntGe(a, b) => write!(f, "(>= {} {})", a, b),
            SmtExpr::Select(a, i) => write!(f, "(select {} {})", a, i),
            SmtExpr::Store(a, i, v) => write!(f, "(store {} {} {})", a, i, v),
            SmtExpr::Apply(name, args) => {
                write!(f, "({}", name)?;
                for a in args {
                    write!(f, " {}", a)?;
                }
                write!(f, ")")
            }
            SmtExpr::Let(bindings, body) => {
                write!(f, "(let (")?;
                for (i, (n, e)) in bindings.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", n, e)?;
                }
                write!(f, ") {})", body)
            }
            SmtExpr::Exists(vars, body) => {
                write!(f, "(exists (")?;
                for (i, (n, s)) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", n, s)?;
                }
                write!(f, ") {})", body)
            }
            SmtExpr::ForAll(vars, body) => {
                write!(f, "(forall (")?;
                for (i, (n, s)) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", n, s)?;
                }
                write!(f, ") {})", body)
            }
        }
    }
}

// ─── Constraint origin tracking ─────────────────────────────────────────

/// Origin of an SMT constraint for debugging and tracing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintOrigin {
    InitialState,
    Transition { from: usize, to: usize, step: u32 },
    AdversaryAction { action_idx: u32, step: u32 },
    KnowledgeAccumulation { step: u32 },
    PropertyNegation,
    FrameCondition { step: u32 },
    BudgetBound,
    DepthBound,
    SymmetryBreaking,
}

/// An annotated SMT constraint carrying its origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtConstraint {
    pub formula: SmtExpr,
    pub origin: ConstraintOrigin,
    pub label: String,
}

impl SmtConstraint {
    pub fn new(formula: SmtExpr, origin: ConstraintOrigin, label: impl Into<String>) -> Self {
        SmtConstraint {
            formula,
            origin,
            label: label.into(),
        }
    }
}

// ─── SMT declarations ───────────────────────────────────────────────────

/// An SMT-LIB2 declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtDeclaration {
    DeclareSort { name: String, arity: u32 },
    DeclareFun { name: String, args: Vec<SmtSort>, ret: SmtSort },
    DefineFun { name: String, args: Vec<(String, SmtSort)>, ret: SmtSort, body: SmtExpr },
    DeclareConst { name: String, sort: SmtSort },
}

// ─── Complete SMT formula ───────────────────────────────────────────────

/// The complete SMT formula produced by DYENCODE (Algorithm 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtFormula {
    pub constraints: Vec<SmtConstraint>,
    pub declarations: Vec<SmtDeclaration>,
    pub depth_bound: u32,
    pub adversary_budget: u32,
    pub library_name: String,
    pub library_version: String,
}

impl SmtFormula {
    pub fn new(depth_bound: u32, adversary_budget: u32) -> Self {
        SmtFormula {
            constraints: Vec::new(),
            declarations: Vec::new(),
            depth_bound,
            adversary_budget,
            library_name: String::new(),
            library_version: String::new(),
        }
    }

    pub fn add_constraint(&mut self, constraint: SmtConstraint) {
        self.constraints.push(constraint);
    }

    pub fn add_declaration(&mut self, decl: SmtDeclaration) {
        self.declarations.push(decl);
    }

    pub fn total_nodes(&self) -> usize {
        self.constraints.iter().map(|c| c.formula.node_count()).sum()
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn declaration_count(&self) -> usize {
        self.declarations.len()
    }

    pub fn merge(&mut self, other: SmtFormula) {
        self.declarations.extend(other.declarations);
        self.constraints.extend(other.constraints);
    }
}

// ─── Adversary budget ───────────────────────────────────────────────────

/// Configuration for the bounded Dolev-Yao adversary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryBudget {
    pub max_actions: u32,
    pub max_drops: u32,
    pub max_injects: u32,
    pub max_modifies: u32,
    pub max_intercepts: u32,
    pub max_replays: u32,
}

impl AdversaryBudget {
    pub fn new(max_actions: u32) -> Self {
        AdversaryBudget {
            max_actions,
            max_drops: max_actions,
            max_injects: max_actions,
            max_modifies: max_actions,
            max_intercepts: max_actions,
            max_replays: max_actions,
        }
    }

    pub fn with_per_action_limits(
        max_actions: u32,
        drops: u32,
        injects: u32,
        modifies: u32,
        intercepts: u32,
    ) -> Self {
        AdversaryBudget {
            max_actions,
            max_drops: drops,
            max_injects: injects,
            max_modifies: modifies,
            max_intercepts: intercepts,
            max_replays: 0,
        }
    }

    pub fn total_budget(&self) -> u32 {
        self.max_actions
    }
}

impl Default for AdversaryBudget {
    fn default() -> Self {
        Self::new(5)
    }
}

// ─── Encoding statistics ────────────────────────────────────────────────

/// Statistics collected during the DYENCODE process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncodingStats {
    pub total_constraints: usize,
    pub total_variables: usize,
    pub total_nodes: usize,
    pub unrolling_depth: u32,
    pub adversary_budget: u32,
    pub encoding_time_ms: u64,
    pub optimization_time_ms: u64,
    pub state_vars: usize,
    pub transition_constraints: usize,
    pub knowledge_constraints: usize,
    pub property_constraints: usize,
}

impl fmt::Display for EncodingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncodingStats {{ constraints: {}, vars: {}, nodes: {}, depth: {}, budget: {} }}",
            self.total_constraints,
            self.total_variables,
            self.total_nodes,
            self.unrolling_depth,
            self.adversary_budget,
        )
    }
}

// ─── LTS types for encoding ─────────────────────────────────────────────

/// State identifier in the negotiation LTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateId(pub u64);

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Handshake phase for use in encoding (mirrors the protocol types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    Init,
    ClientHelloSent,
    ServerHelloReceived,
    Negotiated,
    Done,
    Abort,
}

impl Phase {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Phase::Done | Phase::Abort)
    }

    pub fn to_index(&self) -> u32 {
        match self {
            Phase::Init => 0,
            Phase::ClientHelloSent => 1,
            Phase::ServerHelloReceived => 2,
            Phase::Negotiated => 3,
            Phase::Done => 4,
            Phase::Abort => 5,
        }
    }

    pub fn from_index(idx: u32) -> Option<Phase> {
        match idx {
            0 => Some(Phase::Init),
            1 => Some(Phase::ClientHelloSent),
            2 => Some(Phase::ServerHelloReceived),
            3 => Some(Phase::Negotiated),
            4 => Some(Phase::Done),
            5 => Some(Phase::Abort),
            _ => None,
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Adversary action kind for encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdvAction {
    Drop,
    Intercept,
    Inject { payload_id: u32 },
    Modify { field_id: u32 },
    Replay { message_idx: u32 },
    NoOp,
}

impl AdvAction {
    pub fn action_index(&self) -> u32 {
        match self {
            AdvAction::Drop => 0,
            AdvAction::Intercept => 1,
            AdvAction::Inject { .. } => 2,
            AdvAction::Modify { .. } => 3,
            AdvAction::Replay { .. } => 4,
            AdvAction::NoOp => 5,
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, AdvAction::NoOp)
    }

    pub fn num_action_types() -> u32 {
        6
    }
}

/// Transition label in the encoding LTS.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncTransitionLabel {
    ClientAction { action_id: u32, ciphers: Vec<u16>, version: u16 },
    ServerAction { action_id: u32, cipher: u16, version: u16 },
    Adversary(AdvAction),
    Tau,
}

impl EncTransitionLabel {
    pub fn is_adversary(&self) -> bool {
        matches!(self, EncTransitionLabel::Adversary(_))
    }
}

/// An LTS state with observable outcome for the encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncState {
    pub id: StateId,
    pub phase: Phase,
    pub offered_ciphers: BTreeSet<u16>,
    pub selected_cipher: Option<u16>,
    pub selected_version: Option<u16>,
    pub active_extensions: BTreeSet<u16>,
    pub is_terminal: bool,
}

impl EncState {
    pub fn new(id: StateId, phase: Phase) -> Self {
        EncState {
            id,
            phase,
            offered_ciphers: BTreeSet::new(),
            selected_cipher: None,
            selected_version: None,
            active_extensions: BTreeSet::new(),
            is_terminal: phase.is_terminal(),
        }
    }
}

/// A transition in the encoding LTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncTransition {
    pub source: StateId,
    pub target: StateId,
    pub label: EncTransitionLabel,
    pub guard: Option<SmtExpr>,
}

/// The negotiation LTS used directly by the encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncNegotiationLTS {
    pub states: BTreeMap<StateId, EncState>,
    pub transitions: Vec<EncTransition>,
    pub initial_state: StateId,
}

impl EncNegotiationLTS {
    pub fn new(initial: EncState) -> Self {
        let id = initial.id;
        let mut states = BTreeMap::new();
        states.insert(id, initial);
        EncNegotiationLTS {
            states,
            transitions: Vec::new(),
            initial_state: id,
        }
    }

    pub fn add_state(&mut self, state: EncState) {
        self.states.insert(state.id, state);
    }

    pub fn add_transition(&mut self, trans: EncTransition) {
        self.transitions.push(trans);
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn terminal_states(&self) -> Vec<StateId> {
        self.states
            .values()
            .filter(|s| s.is_terminal)
            .map(|s| s.id)
            .collect()
    }

    pub fn transitions_from(&self, state: StateId) -> Vec<&EncTransition> {
        self.transitions.iter().filter(|t| t.source == state).collect()
    }

    pub fn adversary_transitions(&self) -> Vec<&EncTransition> {
        self.transitions.iter().filter(|t| t.label.is_adversary()).collect()
    }

    pub fn reachable_states(&self) -> BTreeSet<StateId> {
        let mut visited = BTreeSet::new();
        let mut stack = vec![self.initial_state];
        while let Some(s) = stack.pop() {
            if visited.insert(s) {
                for t in self.transitions_from(s) {
                    if !visited.contains(&t.target) {
                        stack.push(t.target);
                    }
                }
            }
        }
        visited
    }

    /// Get all unique cipher suite IDs mentioned anywhere in the LTS.
    pub fn all_cipher_ids(&self) -> BTreeSet<u16> {
        let mut ids = BTreeSet::new();
        for state in self.states.values() {
            ids.extend(&state.offered_ciphers);
            if let Some(c) = state.selected_cipher {
                ids.insert(c);
            }
        }
        for t in &self.transitions {
            match &t.label {
                EncTransitionLabel::ClientAction { ciphers, .. } => {
                    ids.extend(ciphers);
                }
                EncTransitionLabel::ServerAction { cipher, .. } => {
                    ids.insert(*cipher);
                }
                _ => {}
            }
        }
        ids
    }

    /// Get all unique version values mentioned in the LTS.
    pub fn all_version_ids(&self) -> BTreeSet<u16> {
        let mut vers = BTreeSet::new();
        for state in self.states.values() {
            if let Some(v) = state.selected_version {
                vers.insert(v);
            }
        }
        for t in &self.transitions {
            match &t.label {
                EncTransitionLabel::ClientAction { version, .. }
                | EncTransitionLabel::ServerAction { version, .. } => {
                    vers.insert(*version);
                }
                _ => {}
            }
        }
        vers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_sort_display() {
        assert_eq!(format!("{}", SmtSort::Bool), "Bool");
        assert_eq!(format!("{}", SmtSort::BitVec(16)), "(_ BitVec 16)");
        let arr = SmtSort::Array(Box::new(SmtSort::BitVec(8)), Box::new(SmtSort::Bool));
        assert_eq!(format!("{}", arr), "(Array (_ BitVec 8) Bool)");
    }

    #[test]
    fn test_smt_expr_and_simplification() {
        let t = SmtExpr::BoolLit(true);
        let a = SmtExpr::var("a");
        let result = SmtExpr::and(vec![t, a.clone()]);
        assert_eq!(result, SmtExpr::var("a"));

        let f = SmtExpr::BoolLit(false);
        let result = SmtExpr::and(vec![f, a]);
        assert_eq!(result, SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_smt_expr_or_simplification() {
        let f = SmtExpr::BoolLit(false);
        let a = SmtExpr::var("a");
        let result = SmtExpr::or(vec![f, a.clone()]);
        assert_eq!(result, SmtExpr::var("a"));

        let t = SmtExpr::BoolLit(true);
        let result = SmtExpr::or(vec![t, a]);
        assert_eq!(result, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_smt_expr_double_negation() {
        let a = SmtExpr::var("a");
        let neg_neg = SmtExpr::not(SmtExpr::not(a.clone()));
        assert_eq!(neg_neg, a);
    }

    #[test]
    fn test_smt_expr_node_count() {
        let e = SmtExpr::and(vec![
            SmtExpr::var("a"),
            SmtExpr::eq(SmtExpr::var("b"), SmtExpr::bv_lit(42, 16)),
        ]);
        assert_eq!(e.node_count(), 5);
    }

    #[test]
    fn test_smt_expr_free_vars() {
        let e = SmtExpr::and(vec![
            SmtExpr::var("x"),
            SmtExpr::eq(SmtExpr::var("y"), SmtExpr::var("z")),
        ]);
        let fv = e.free_vars();
        assert_eq!(fv.len(), 3);
        assert!(fv.contains("x"));
        assert!(fv.contains("y"));
        assert!(fv.contains("z"));
    }

    #[test]
    fn test_smt_expr_substitute() {
        let e = SmtExpr::and(vec![SmtExpr::var("x"), SmtExpr::var("y")]);
        let result = e.substitute("x", &SmtExpr::BoolLit(true));
        assert_eq!(result, SmtExpr::var("y"));
    }

    #[test]
    fn test_smt_expr_display() {
        let e = SmtExpr::bv_ult(SmtExpr::var("x"), SmtExpr::bv_lit(10, 16));
        assert_eq!(format!("{}", e), "(bvult x (_ bv10 16))");
    }

    #[test]
    fn test_smt_formula_stats() {
        let mut f = SmtFormula::new(10, 3);
        f.add_constraint(SmtConstraint::new(
            SmtExpr::var("x"),
            ConstraintOrigin::InitialState,
            "test",
        ));
        assert_eq!(f.constraint_count(), 1);
        assert_eq!(f.total_nodes(), 1);
    }

    #[test]
    fn test_adversary_budget() {
        let b = AdversaryBudget::new(5);
        assert_eq!(b.total_budget(), 5);

        let b2 = AdversaryBudget::with_per_action_limits(5, 2, 2, 1, 3);
        assert_eq!(b2.max_drops, 2);
        assert_eq!(b2.max_injects, 2);
    }

    #[test]
    fn test_and_flatten() {
        let inner = SmtExpr::and(vec![SmtExpr::var("a"), SmtExpr::var("b")]);
        let outer = SmtExpr::and(vec![inner, SmtExpr::var("c")]);
        match outer {
            SmtExpr::And(es) => assert_eq!(es.len(), 3),
            _ => panic!("expected flattened And"),
        }
    }

    #[test]
    fn test_implies_simplification() {
        let r = SmtExpr::implies(SmtExpr::BoolLit(false), SmtExpr::var("x"));
        assert_eq!(r, SmtExpr::BoolLit(true));

        let r = SmtExpr::implies(SmtExpr::BoolLit(true), SmtExpr::var("x"));
        assert_eq!(r, SmtExpr::var("x"));
    }

    #[test]
    fn test_ite_simplification() {
        let r = SmtExpr::ite(SmtExpr::BoolLit(true), SmtExpr::var("a"), SmtExpr::var("b"));
        assert_eq!(r, SmtExpr::var("a"));

        let r = SmtExpr::ite(SmtExpr::var("c"), SmtExpr::var("x"), SmtExpr::var("x"));
        assert_eq!(r, SmtExpr::var("x"));
    }

    #[test]
    fn test_eq_simplification() {
        let r = SmtExpr::eq(SmtExpr::var("a"), SmtExpr::var("a"));
        assert_eq!(r, SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_enc_lts_basic() {
        let init = EncState::new(StateId(0), Phase::Init);
        let mut lts = EncNegotiationLTS::new(init);

        let mut done = EncState::new(StateId(1), Phase::Done);
        done.selected_cipher = Some(0x002F);
        lts.add_state(done);

        lts.add_transition(EncTransition {
            source: StateId(0),
            target: StateId(1),
            label: EncTransitionLabel::Tau,
            guard: None,
        });

        assert_eq!(lts.state_count(), 2);
        assert_eq!(lts.transition_count(), 1);
        assert_eq!(lts.terminal_states().len(), 1);
    }

    #[test]
    fn test_phase_roundtrip() {
        for i in 0..6 {
            let phase = Phase::from_index(i).unwrap();
            assert_eq!(phase.to_index(), i);
        }
        assert!(Phase::from_index(99).is_none());
    }

    #[test]
    fn test_adv_action_index() {
        assert_eq!(AdvAction::Drop.action_index(), 0);
        assert_eq!(AdvAction::Intercept.action_index(), 1);
        assert!(AdvAction::Drop.is_active());
        assert!(!AdvAction::NoOp.is_active());
    }
}
