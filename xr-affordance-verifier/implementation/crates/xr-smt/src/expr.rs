//! SMT expression AST for QF_LRA (Quantifier-Free Linear Real Arithmetic).
//!
//! Provides the core expression type [`SmtExpr`] with variants for arithmetic,
//! boolean connectives, comparisons, and let bindings. Includes simplification,
//! substitution, free-variable collection, and SMT-LIB2 pretty-printing.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// SMT sort (type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Real,
    Int,
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::Real => write!(f, "Real"),
            SmtSort::Int => write!(f, "Int"),
        }
    }
}

/// Variable declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtDecl {
    pub name: String,
    pub sort: SmtSort,
}

impl SmtDecl {
    pub fn new(name: impl Into<String>, sort: SmtSort) -> Self {
        Self {
            name: name.into(),
            sort,
        }
    }

    pub fn to_smtlib2(&self) -> String {
        format!("(declare-fun {} () {})", self.name, self.sort)
    }
}

/// SMT expression AST.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    Const(f64),
    BoolConst(bool),
    Var(String),
    Add(Box<SmtExpr>, Box<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Box<SmtExpr>, Box<SmtExpr>),
    Div(Box<SmtExpr>, Box<SmtExpr>),
    Neg(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Not(Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Let(String, Box<SmtExpr>, Box<SmtExpr>),
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl SmtExpr {
    pub fn real(v: f64) -> Self {
        SmtExpr::Const(v)
    }

    pub fn bool_const(v: bool) -> Self {
        SmtExpr::BoolConst(v)
    }

    pub fn var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into())
    }

    pub fn add(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Add(Box::new(a), Box::new(b))
    }

    pub fn sub(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Sub(Box::new(a), Box::new(b))
    }

    pub fn mul(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Mul(Box::new(a), Box::new(b))
    }

    pub fn div(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Div(Box::new(a), Box::new(b))
    }

    pub fn neg(a: SmtExpr) -> Self {
        SmtExpr::Neg(Box::new(a))
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::And(exprs)
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Or(exprs)
    }

    pub fn not(a: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(a))
    }

    pub fn le(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Le(Box::new(a), Box::new(b))
    }

    pub fn lt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Lt(Box::new(a), Box::new(b))
    }

    pub fn ge(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Ge(Box::new(a), Box::new(b))
    }

    pub fn gt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Gt(Box::new(a), Box::new(b))
    }

    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(a), Box::new(b))
    }

    pub fn ite(cond: SmtExpr, then_: SmtExpr, else_: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(cond), Box::new(then_), Box::new(else_))
    }

    pub fn let_bind(name: impl Into<String>, val: SmtExpr, body: SmtExpr) -> Self {
        SmtExpr::Let(name.into(), Box::new(val), Box::new(body))
    }
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Simplify via constant folding, identity elimination, and boolean laws.
    pub fn simplify(&self) -> SmtExpr {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) | SmtExpr::Var(_) => self.clone(),

            SmtExpr::Neg(a) => {
                let a = a.simplify();
                match a {
                    SmtExpr::Const(v) => SmtExpr::Const(-v),
                    SmtExpr::Neg(inner) => *inner,
                    other => SmtExpr::neg(other),
                }
            }

            SmtExpr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::Const(x + y),
                    (SmtExpr::Const(v), _) if *v == 0.0 => b,
                    (_, SmtExpr::Const(v)) if *v == 0.0 => a,
                    _ => SmtExpr::add(a, b),
                }
            }

            SmtExpr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::Const(x - y),
                    (_, SmtExpr::Const(v)) if *v == 0.0 => a,
                    (SmtExpr::Const(v), _) if *v == 0.0 => SmtExpr::neg(b),
                    _ => SmtExpr::sub(a, b),
                }
            }

            SmtExpr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::Const(x * y),
                    (SmtExpr::Const(v), _) | (_, SmtExpr::Const(v)) if *v == 0.0 => {
                        SmtExpr::Const(0.0)
                    }
                    (SmtExpr::Const(v), _) if *v == 1.0 => b,
                    (_, SmtExpr::Const(v)) if *v == 1.0 => a,
                    (SmtExpr::Const(v), _) if *v == -1.0 => SmtExpr::neg(b),
                    (_, SmtExpr::Const(v)) if *v == -1.0 => SmtExpr::neg(a),
                    _ => SmtExpr::mul(a, b),
                }
            }

            SmtExpr::Div(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) if *y != 0.0 => {
                        SmtExpr::Const(x / y)
                    }
                    (SmtExpr::Const(v), _) if *v == 0.0 => SmtExpr::Const(0.0),
                    (_, SmtExpr::Const(v)) if *v == 1.0 => a,
                    _ => SmtExpr::div(a, b),
                }
            }

            SmtExpr::And(children) => {
                let mut simplified: Vec<SmtExpr> = Vec::new();
                for c in children {
                    let s = c.simplify();
                    match s {
                        SmtExpr::BoolConst(false) => return SmtExpr::BoolConst(false),
                        SmtExpr::BoolConst(true) => {}
                        SmtExpr::And(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                match simplified.len() {
                    0 => SmtExpr::BoolConst(true),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::And(simplified),
                }
            }

            SmtExpr::Or(children) => {
                let mut simplified: Vec<SmtExpr> = Vec::new();
                for c in children {
                    let s = c.simplify();
                    match s {
                        SmtExpr::BoolConst(true) => return SmtExpr::BoolConst(true),
                        SmtExpr::BoolConst(false) => {}
                        SmtExpr::Or(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                match simplified.len() {
                    0 => SmtExpr::BoolConst(false),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::Or(simplified),
                }
            }

            SmtExpr::Not(a) => {
                let a = a.simplify();
                match a {
                    SmtExpr::BoolConst(v) => SmtExpr::BoolConst(!v),
                    SmtExpr::Not(inner) => *inner,
                    other => SmtExpr::not(other),
                }
            }

            SmtExpr::Le(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::BoolConst(x <= y),
                    _ => SmtExpr::le(a, b),
                }
            }

            SmtExpr::Lt(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::BoolConst(x < y),
                    _ => SmtExpr::lt(a, b),
                }
            }

            SmtExpr::Ge(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::BoolConst(x >= y),
                    _ => SmtExpr::ge(a, b),
                }
            }

            SmtExpr::Gt(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::BoolConst(x > y),
                    _ => SmtExpr::gt(a, b),
                }
            }

            SmtExpr::Eq(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (SmtExpr::Const(x), SmtExpr::Const(y)) => SmtExpr::BoolConst((x - y).abs() < 1e-15),
                    (SmtExpr::BoolConst(x), SmtExpr::BoolConst(y)) => SmtExpr::BoolConst(x == y),
                    _ => SmtExpr::eq(a, b),
                }
            }

            SmtExpr::Ite(c, t, e) => {
                let c = c.simplify();
                let t = t.simplify();
                let e = e.simplify();
                match c {
                    SmtExpr::BoolConst(true) => t,
                    SmtExpr::BoolConst(false) => e,
                    _ => {
                        if t == e {
                            t
                        } else {
                            SmtExpr::ite(c, t, e)
                        }
                    }
                }
            }

            SmtExpr::Let(name, val, body) => {
                let val = val.simplify();
                let body = body.simplify();
                if !body.free_variables().contains(name.as_str()) {
                    body
                } else if let SmtExpr::Const(_) | SmtExpr::BoolConst(_) = &val {
                    body.substitute(name, &val).simplify()
                } else {
                    SmtExpr::let_bind(name.clone(), val, body)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Substitution
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Replace all free occurrences of `var` with `replacement`.
    pub fn substitute(&self, var: &str, replacement: &SmtExpr) -> SmtExpr {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) => self.clone(),
            SmtExpr::Var(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            SmtExpr::Add(a, b) => SmtExpr::add(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Sub(a, b) => SmtExpr::sub(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Mul(a, b) => SmtExpr::mul(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Div(a, b) => SmtExpr::div(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Neg(a) => SmtExpr::neg(a.substitute(var, replacement)),
            SmtExpr::And(cs) => SmtExpr::and(cs.iter().map(|c| c.substitute(var, replacement)).collect()),
            SmtExpr::Or(cs) => SmtExpr::or(cs.iter().map(|c| c.substitute(var, replacement)).collect()),
            SmtExpr::Not(a) => SmtExpr::not(a.substitute(var, replacement)),
            SmtExpr::Le(a, b) => SmtExpr::le(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Lt(a, b) => SmtExpr::lt(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Ge(a, b) => SmtExpr::ge(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Gt(a, b) => SmtExpr::gt(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Eq(a, b) => SmtExpr::eq(a.substitute(var, replacement), b.substitute(var, replacement)),
            SmtExpr::Ite(c, t, e) => SmtExpr::ite(
                c.substitute(var, replacement),
                t.substitute(var, replacement),
                e.substitute(var, replacement),
            ),
            SmtExpr::Let(name, val, body) => {
                let new_val = val.substitute(var, replacement);
                if name == var {
                    // The let binding shadows var, so don't substitute in body
                    SmtExpr::let_bind(name.clone(), new_val, *body.clone())
                } else {
                    SmtExpr::let_bind(name.clone(), new_val, body.substitute(var, replacement))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free variables
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Collect all free variable names respecting Let-binding scoping.
    pub fn free_variables(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_free_vars(&mut vars, &BTreeSet::new());
        vars
    }

    fn collect_free_vars(&self, out: &mut BTreeSet<String>, bound: &BTreeSet<String>) {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) => {}
            SmtExpr::Var(name) => {
                if !bound.contains(name) {
                    out.insert(name.clone());
                }
            }
            SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Eq(a, b) => {
                a.collect_free_vars(out, bound);
                b.collect_free_vars(out, bound);
            }
            SmtExpr::Neg(a) | SmtExpr::Not(a) => {
                a.collect_free_vars(out, bound);
            }
            SmtExpr::And(cs) | SmtExpr::Or(cs) => {
                for c in cs {
                    c.collect_free_vars(out, bound);
                }
            }
            SmtExpr::Ite(c, t, e) => {
                c.collect_free_vars(out, bound);
                t.collect_free_vars(out, bound);
                e.collect_free_vars(out, bound);
            }
            SmtExpr::Let(name, val, body) => {
                val.collect_free_vars(out, bound);
                let mut inner_bound = bound.clone();
                inner_bound.insert(name.clone());
                body.collect_free_vars(out, &inner_bound);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB2 output
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Pretty-print to SMT-LIB2 format.
    pub fn to_smtlib2(&self) -> String {
        match self {
            SmtExpr::Const(v) => {
                if *v < 0.0 {
                    format!("(- {})", format_decimal(-v))
                } else {
                    format_decimal(*v)
                }
            }
            SmtExpr::BoolConst(b) => (if *b { "true" } else { "false" }).to_string(),
            SmtExpr::Var(name) => name.clone(),
            SmtExpr::Add(a, b) => format!("(+ {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Sub(a, b) => format!("(- {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Mul(a, b) => format!("(* {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Div(a, b) => format!("(/ {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Neg(a) => format!("(- {})", a.to_smtlib2()),
            SmtExpr::And(cs) if cs.is_empty() => "true".to_string(),
            SmtExpr::And(cs) if cs.len() == 1 => cs[0].to_smtlib2(),
            SmtExpr::And(cs) => {
                let args: Vec<String> = cs.iter().map(|c| c.to_smtlib2()).collect();
                format!("(and {})", args.join(" "))
            }
            SmtExpr::Or(cs) if cs.is_empty() => "false".to_string(),
            SmtExpr::Or(cs) if cs.len() == 1 => cs[0].to_smtlib2(),
            SmtExpr::Or(cs) => {
                let args: Vec<String> = cs.iter().map(|c| c.to_smtlib2()).collect();
                format!("(or {})", args.join(" "))
            }
            SmtExpr::Not(a) => format!("(not {})", a.to_smtlib2()),
            SmtExpr::Le(a, b) => format!("(<= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Lt(a, b) => format!("(< {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Ge(a, b) => format!("(>= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Gt(a, b) => format!("(> {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Eq(a, b) => format!("(= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            SmtExpr::Ite(c, t, e) => {
                format!("(ite {} {} {})", c.to_smtlib2(), t.to_smtlib2(), e.to_smtlib2())
            }
            SmtExpr::Let(name, val, body) => {
                format!("(let (({} {})) {})", name, val.to_smtlib2(), body.to_smtlib2())
            }
        }
    }
}

fn format_decimal(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib2())
    }
}

// ---------------------------------------------------------------------------
// Metrics & queries
// ---------------------------------------------------------------------------

impl SmtExpr {
    /// Count total AST nodes.
    pub fn size(&self) -> usize {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) | SmtExpr::Var(_) => 1,
            SmtExpr::Neg(a) | SmtExpr::Not(a) => 1 + a.size(),
            SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Eq(a, b) => 1 + a.size() + b.size(),
            SmtExpr::And(cs) | SmtExpr::Or(cs) => 1 + cs.iter().map(|c| c.size()).sum::<usize>(),
            SmtExpr::Ite(c, t, e) => 1 + c.size() + t.size() + e.size(),
            SmtExpr::Let(_, v, b) => 1 + v.size() + b.size(),
        }
    }

    /// Maximum nesting depth.
    pub fn depth(&self) -> usize {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) | SmtExpr::Var(_) => 0,
            SmtExpr::Neg(a) | SmtExpr::Not(a) => 1 + a.depth(),
            SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Eq(a, b) => 1 + a.depth().max(b.depth()),
            SmtExpr::And(cs) | SmtExpr::Or(cs) => {
                1 + cs.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
            SmtExpr::Ite(c, t, e) => 1 + c.depth().max(t.depth()).max(e.depth()),
            SmtExpr::Let(_, v, b) => 1 + v.depth().max(b.depth()),
        }
    }

    /// Check if the expression is linear (QF_LRA compatible).
    pub fn is_linear(&self) -> bool {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) | SmtExpr::Var(_) => true,
            SmtExpr::Neg(a) | SmtExpr::Not(a) => a.is_linear(),
            SmtExpr::Add(a, b) | SmtExpr::Sub(a, b) => a.is_linear() && b.is_linear(),
            SmtExpr::Mul(a, b) => {
                let a_const = matches!(a.as_ref(), SmtExpr::Const(_));
                let b_const = matches!(b.as_ref(), SmtExpr::Const(_));
                (a_const || b_const) && a.is_linear() && b.is_linear()
            }
            SmtExpr::Div(a, b) => {
                let b_const = matches!(b.as_ref(), SmtExpr::Const(_));
                b_const && a.is_linear() && b.is_linear()
            }
            SmtExpr::And(cs) | SmtExpr::Or(cs) => cs.iter().all(|c| c.is_linear()),
            SmtExpr::Le(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Eq(a, b) => a.is_linear() && b.is_linear(),
            SmtExpr::Ite(c, t, e) => c.is_linear() && t.is_linear() && e.is_linear(),
            SmtExpr::Let(_, v, b) => v.is_linear() && b.is_linear(),
        }
    }

    /// Map a function over all variable references.
    pub fn map_vars<F: Fn(&str) -> SmtExpr>(&self, f: &F) -> SmtExpr {
        match self {
            SmtExpr::Const(_) | SmtExpr::BoolConst(_) => self.clone(),
            SmtExpr::Var(name) => f(name),
            SmtExpr::Add(a, b) => SmtExpr::add(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Sub(a, b) => SmtExpr::sub(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Mul(a, b) => SmtExpr::mul(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Div(a, b) => SmtExpr::div(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Neg(a) => SmtExpr::neg(a.map_vars(f)),
            SmtExpr::And(cs) => SmtExpr::and(cs.iter().map(|c| c.map_vars(f)).collect()),
            SmtExpr::Or(cs) => SmtExpr::or(cs.iter().map(|c| c.map_vars(f)).collect()),
            SmtExpr::Not(a) => SmtExpr::not(a.map_vars(f)),
            SmtExpr::Le(a, b) => SmtExpr::le(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Lt(a, b) => SmtExpr::lt(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Ge(a, b) => SmtExpr::ge(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Gt(a, b) => SmtExpr::gt(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Eq(a, b) => SmtExpr::eq(a.map_vars(f), b.map_vars(f)),
            SmtExpr::Ite(c, t, e) => SmtExpr::ite(c.map_vars(f), t.map_vars(f), e.map_vars(f)),
            SmtExpr::Let(name, val, body) => {
                SmtExpr::let_bind(name.clone(), val.map_vars(f), body.map_vars(f))
            }
        }
    }

    /// Evaluate numerically given variable assignments.
    /// Returns `None` if a variable is missing or the expression is purely boolean.
    pub fn eval(&self, assignment: &HashMap<String, f64>) -> Option<f64> {
        match self {
            SmtExpr::Const(v) => Some(*v),
            SmtExpr::BoolConst(_) => None,
            SmtExpr::Var(name) => assignment.get(name).copied(),
            SmtExpr::Add(a, b) => Some(a.eval(assignment)? + b.eval(assignment)?),
            SmtExpr::Sub(a, b) => Some(a.eval(assignment)? - b.eval(assignment)?),
            SmtExpr::Mul(a, b) => Some(a.eval(assignment)? * b.eval(assignment)?),
            SmtExpr::Div(a, b) => {
                let bv = b.eval(assignment)?;
                if bv.abs() < 1e-15 {
                    None
                } else {
                    Some(a.eval(assignment)? / bv)
                }
            }
            SmtExpr::Neg(a) => Some(-a.eval(assignment)?),
            SmtExpr::Ite(c, t, e) => {
                if c.eval_bool(assignment)? {
                    t.eval(assignment)
                } else {
                    e.eval(assignment)
                }
            }
            SmtExpr::Let(name, val, body) => {
                let v = val.eval(assignment)?;
                let mut extended = assignment.clone();
                extended.insert(name.clone(), v);
                body.eval(&extended)
            }
            _ => None,
        }
    }

    /// Evaluate boolean expressions.
    pub fn eval_bool(&self, assignment: &HashMap<String, f64>) -> Option<bool> {
        match self {
            SmtExpr::BoolConst(b) => Some(*b),
            SmtExpr::And(cs) => {
                for c in cs {
                    if !c.eval_bool(assignment)? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            SmtExpr::Or(cs) => {
                for c in cs {
                    if c.eval_bool(assignment)? {
                        return Some(true);
                    }
                }
                Some(false)
            }
            SmtExpr::Not(a) => Some(!a.eval_bool(assignment)?),
            SmtExpr::Le(a, b) => Some(a.eval(assignment)? <= b.eval(assignment)?),
            SmtExpr::Lt(a, b) => Some(a.eval(assignment)? < b.eval(assignment)?),
            SmtExpr::Ge(a, b) => Some(a.eval(assignment)? >= b.eval(assignment)?),
            SmtExpr::Gt(a, b) => Some(a.eval(assignment)? > b.eval(assignment)?),
            SmtExpr::Eq(a, b) => {
                let av = a.eval(assignment)?;
                let bv = b.eval(assignment)?;
                Some((av - bv).abs() < 1e-10)
            }
            SmtExpr::Ite(c, t, e) => {
                if c.eval_bool(assignment)? {
                    t.eval_bool(assignment)
                } else {
                    e.eval_bool(assignment)
                }
            }
            SmtExpr::Let(name, val, body) => {
                let v = val.eval(assignment)?;
                let mut extended = assignment.clone();
                extended.insert(name.clone(), v);
                body.eval_bool(&extended)
            }
            _ => None,
        }
    }

    /// Check if this is a constant expression.
    pub fn is_const(&self) -> bool {
        matches!(self, SmtExpr::Const(_) | SmtExpr::BoolConst(_))
    }

    /// Try to extract a constant f64 value.
    pub fn as_const(&self) -> Option<f64> {
        if let SmtExpr::Const(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Check if this is a variable.
    pub fn is_var(&self) -> bool {
        matches!(self, SmtExpr::Var(_))
    }

    /// Try to extract a variable name.
    pub fn as_var(&self) -> Option<&str> {
        if let SmtExpr::Var(name) = self {
            Some(name)
        } else {
            None
        }
    }

    /// Sum a list of expressions.
    pub fn sum(exprs: Vec<SmtExpr>) -> SmtExpr {
        if exprs.is_empty() {
            return SmtExpr::Const(0.0);
        }
        let mut iter = exprs.into_iter();
        let mut acc = iter.next().unwrap();
        for e in iter {
            acc = SmtExpr::add(acc, e);
        }
        acc
    }

    /// Create a linear combination: Σ coeffs[i] * vars[i].
    pub fn linear_combination(coeffs: &[f64], vars: &[SmtExpr]) -> SmtExpr {
        assert_eq!(coeffs.len(), vars.len());
        let terms: Vec<SmtExpr> = coeffs
            .iter()
            .zip(vars.iter())
            .filter(|(c, _)| c.abs() > 1e-15)
            .map(|(c, v)| {
                if (*c - 1.0).abs() < 1e-15 {
                    v.clone()
                } else if (*c + 1.0).abs() < 1e-15 {
                    SmtExpr::neg(v.clone())
                } else {
                    SmtExpr::mul(SmtExpr::Const(*c), v.clone())
                }
            })
            .collect();
        SmtExpr::sum(terms)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_display() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(3.0));
        assert_eq!(e.to_smtlib2(), "(+ x 3.0)");
    }

    #[test]
    fn test_negation_display() {
        let e = SmtExpr::real(-2.5);
        assert_eq!(e.to_smtlib2(), "(- 2.5)");
    }

    #[test]
    fn test_and_display() {
        let e = SmtExpr::and(vec![
            SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)),
            SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(0.0)),
        ]);
        assert_eq!(e.to_smtlib2(), "(and (<= x 5.0) (>= x 0.0))");
    }

    #[test]
    fn test_let_display() {
        let e = SmtExpr::let_bind(
            "y",
            SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0)),
            SmtExpr::mul(SmtExpr::var("y"), SmtExpr::real(2.0)),
        );
        assert_eq!(
            e.to_smtlib2(),
            "(let ((y (+ x 1.0))) (* y 2.0))"
        );
    }

    #[test]
    fn test_simplify_constant_folding() {
        let e = SmtExpr::add(SmtExpr::real(2.0), SmtExpr::real(3.0));
        assert_eq!(e.simplify(), SmtExpr::Const(5.0));
    }

    #[test]
    fn test_simplify_identity_add() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(0.0));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn test_simplify_identity_mul_one() {
        let e = SmtExpr::mul(SmtExpr::real(1.0), SmtExpr::var("x"));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let e = SmtExpr::mul(SmtExpr::real(0.0), SmtExpr::var("x"));
        assert_eq!(e.simplify(), SmtExpr::Const(0.0));
    }

    #[test]
    fn test_simplify_double_neg() {
        let e = SmtExpr::neg(SmtExpr::neg(SmtExpr::var("x")));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn test_simplify_double_not() {
        let e = SmtExpr::not(SmtExpr::not(SmtExpr::var("p")));
        assert_eq!(e.simplify(), SmtExpr::var("p"));
    }

    #[test]
    fn test_simplify_and_false() {
        let e = SmtExpr::and(vec![SmtExpr::var("p"), SmtExpr::BoolConst(false)]);
        assert_eq!(e.simplify(), SmtExpr::BoolConst(false));
    }

    #[test]
    fn test_simplify_or_true() {
        let e = SmtExpr::or(vec![SmtExpr::var("p"), SmtExpr::BoolConst(true)]);
        assert_eq!(e.simplify(), SmtExpr::BoolConst(true));
    }

    #[test]
    fn test_simplify_and_single() {
        let e = SmtExpr::and(vec![SmtExpr::var("p")]);
        assert_eq!(e.simplify(), SmtExpr::var("p"));
    }

    #[test]
    fn test_simplify_flatten_and() {
        let inner = SmtExpr::and(vec![SmtExpr::var("a"), SmtExpr::var("b")]);
        let e = SmtExpr::and(vec![inner, SmtExpr::var("c")]);
        let s = e.simplify();
        if let SmtExpr::And(cs) = s {
            assert_eq!(cs.len(), 3);
        } else {
            panic!("Expected And");
        }
    }

    #[test]
    fn test_simplify_comparison() {
        let e = SmtExpr::le(SmtExpr::real(1.0), SmtExpr::real(2.0));
        assert_eq!(e.simplify(), SmtExpr::BoolConst(true));
        let e2 = SmtExpr::gt(SmtExpr::real(1.0), SmtExpr::real(2.0));
        assert_eq!(e2.simplify(), SmtExpr::BoolConst(false));
    }

    #[test]
    fn test_simplify_ite_true() {
        let e = SmtExpr::ite(SmtExpr::BoolConst(true), SmtExpr::real(1.0), SmtExpr::real(2.0));
        assert_eq!(e.simplify(), SmtExpr::Const(1.0));
    }

    #[test]
    fn test_simplify_ite_same_branches() {
        let e = SmtExpr::ite(SmtExpr::var("p"), SmtExpr::real(5.0), SmtExpr::real(5.0));
        assert_eq!(e.simplify(), SmtExpr::Const(5.0));
    }

    #[test]
    fn test_substitute() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::var("y"));
        let result = e.substitute("x", &SmtExpr::real(3.0));
        assert_eq!(result, SmtExpr::add(SmtExpr::real(3.0), SmtExpr::var("y")));
    }

    #[test]
    fn test_substitute_let_shadow() {
        let e = SmtExpr::let_bind(
            "x",
            SmtExpr::real(10.0),
            SmtExpr::add(SmtExpr::var("x"), SmtExpr::var("y")),
        );
        let result = e.substitute("x", &SmtExpr::real(99.0));
        // x is shadowed in body, so only val is substituted (but val doesn't contain x)
        if let SmtExpr::Let(_, _, body) = &result {
            // body should still be x + y (x is bound by let)
            assert_eq!(body.to_smtlib2(), "(+ x y)");
        } else {
            panic!("Expected Let");
        }
    }

    #[test]
    fn test_free_variables() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::var("y"));
        let fv = e.free_variables();
        assert!(fv.contains("x"));
        assert!(fv.contains("y"));
        assert_eq!(fv.len(), 2);
    }

    #[test]
    fn test_free_variables_let_scoping() {
        let e = SmtExpr::let_bind(
            "x",
            SmtExpr::var("a"),
            SmtExpr::add(SmtExpr::var("x"), SmtExpr::var("b")),
        );
        let fv = e.free_variables();
        assert!(fv.contains("a"), "a should be free (in let value)");
        assert!(fv.contains("b"), "b should be free (in body)");
        assert!(!fv.contains("x"), "x should be bound by let");
        assert_eq!(fv.len(), 2);
    }

    #[test]
    fn test_size() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0));
        assert_eq!(e.size(), 3);
    }

    #[test]
    fn test_depth() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0));
        assert_eq!(e.depth(), 1);
        let deep = SmtExpr::add(SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0)), SmtExpr::real(2.0));
        assert_eq!(deep.depth(), 2);
    }

    #[test]
    fn test_is_linear() {
        let linear = SmtExpr::add(
            SmtExpr::mul(SmtExpr::real(2.0), SmtExpr::var("x")),
            SmtExpr::var("y"),
        );
        assert!(linear.is_linear());

        let nonlinear = SmtExpr::mul(SmtExpr::var("x"), SmtExpr::var("y"));
        assert!(!nonlinear.is_linear());
    }

    #[test]
    fn test_eval() {
        let e = SmtExpr::add(
            SmtExpr::mul(SmtExpr::real(2.0), SmtExpr::var("x")),
            SmtExpr::var("y"),
        );
        let mut a = HashMap::new();
        a.insert("x".to_string(), 3.0);
        a.insert("y".to_string(), 1.0);
        assert_eq!(e.eval(&a), Some(7.0));
    }

    #[test]
    fn test_eval_missing_var() {
        let e = SmtExpr::var("z");
        let a = HashMap::new();
        assert_eq!(e.eval(&a), None);
    }

    #[test]
    fn test_eval_bool() {
        let e = SmtExpr::and(vec![
            SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)),
            SmtExpr::ge(SmtExpr::var("x"), SmtExpr::real(0.0)),
        ]);
        let mut a = HashMap::new();
        a.insert("x".to_string(), 3.0);
        assert_eq!(e.eval_bool(&a), Some(true));
        a.insert("x".to_string(), -1.0);
        assert_eq!(e.eval_bool(&a), Some(false));
    }

    #[test]
    fn test_linear_combination() {
        let vars = vec![SmtExpr::var("x"), SmtExpr::var("y"), SmtExpr::var("z")];
        let e = SmtExpr::linear_combination(&[2.0, -1.0, 0.0], &vars);
        let mut a = HashMap::new();
        a.insert("x".to_string(), 3.0);
        a.insert("y".to_string(), 1.0);
        a.insert("z".to_string(), 100.0);
        assert_eq!(e.eval(&a), Some(5.0));
    }

    #[test]
    fn test_smtdecl() {
        let d = SmtDecl::new("x", SmtSort::Real);
        assert_eq!(d.to_smtlib2(), "(declare-fun x () Real)");
    }

    #[test]
    fn test_map_vars() {
        let e = SmtExpr::add(SmtExpr::var("x"), SmtExpr::var("y"));
        let renamed = e.map_vars(&|name: &str| SmtExpr::var(format!("v_{}", name)));
        assert_eq!(renamed.to_smtlib2(), "(+ v_x v_y)");
    }

    #[test]
    fn test_simplify_let_unused() {
        let e = SmtExpr::let_bind("x", SmtExpr::real(5.0), SmtExpr::var("y"));
        let s = e.simplify();
        assert_eq!(s, SmtExpr::var("y"));
    }

    #[test]
    fn test_simplify_let_inline_const() {
        let e = SmtExpr::let_bind(
            "x",
            SmtExpr::real(5.0),
            SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0)),
        );
        let s = e.simplify();
        assert_eq!(s, SmtExpr::Const(6.0));
    }

    #[test]
    fn test_sum_empty() {
        assert_eq!(SmtExpr::sum(vec![]), SmtExpr::Const(0.0));
    }

    #[test]
    fn test_as_const_and_as_var() {
        assert_eq!(SmtExpr::real(3.0).as_const(), Some(3.0));
        assert_eq!(SmtExpr::var("x").as_var(), Some("x"));
        assert_eq!(SmtExpr::var("x").as_const(), None);
    }

    #[test]
    fn test_sub_identity() {
        let e = SmtExpr::sub(SmtExpr::var("x"), SmtExpr::real(0.0));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn test_div_identity() {
        let e = SmtExpr::div(SmtExpr::var("x"), SmtExpr::real(1.0));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn test_mul_neg_one() {
        let e = SmtExpr::mul(SmtExpr::real(-1.0), SmtExpr::var("x"));
        let s = e.simplify();
        assert_eq!(s, SmtExpr::neg(SmtExpr::var("x")));
    }

    #[test]
    fn test_eq_constants() {
        let e1 = SmtExpr::eq(SmtExpr::real(3.0), SmtExpr::real(3.0));
        assert_eq!(e1.simplify(), SmtExpr::BoolConst(true));
        let e2 = SmtExpr::eq(SmtExpr::real(3.0), SmtExpr::real(4.0));
        assert_eq!(e2.simplify(), SmtExpr::BoolConst(false));
    }

    #[test]
    fn test_or_empty() {
        let e = SmtExpr::or(vec![]);
        assert_eq!(e.simplify(), SmtExpr::BoolConst(false));
    }

    #[test]
    fn test_and_empty() {
        let e = SmtExpr::and(vec![]);
        assert_eq!(e.simplify(), SmtExpr::BoolConst(true));
    }

    #[test]
    fn test_ite_display() {
        let e = SmtExpr::ite(SmtExpr::var("p"), SmtExpr::real(1.0), SmtExpr::real(0.0));
        assert_eq!(e.to_smtlib2(), "(ite p 1.0 0.0)");
    }

    #[test]
    fn test_eval_let() {
        let e = SmtExpr::let_bind(
            "y",
            SmtExpr::add(SmtExpr::var("x"), SmtExpr::real(1.0)),
            SmtExpr::mul(SmtExpr::var("y"), SmtExpr::real(2.0)),
        );
        let mut a = HashMap::new();
        a.insert("x".to_string(), 4.0);
        assert_eq!(e.eval(&a), Some(10.0));
    }

    #[test]
    fn test_eval_bool_not() {
        let e = SmtExpr::not(SmtExpr::le(SmtExpr::var("x"), SmtExpr::real(5.0)));
        let mut a = HashMap::new();
        a.insert("x".to_string(), 6.0);
        assert_eq!(e.eval_bool(&a), Some(true));
    }

    #[test]
    fn test_nested_simplification() {
        let e = SmtExpr::add(
            SmtExpr::mul(SmtExpr::real(0.0), SmtExpr::var("x")),
            SmtExpr::add(SmtExpr::real(1.0), SmtExpr::real(2.0)),
        );
        assert_eq!(e.simplify(), SmtExpr::Const(3.0));
    }
}
