//! SMT-LIB2 abstract syntax tree.
//!
//! Provides a Rust-native representation of SMT-LIB2 commands, expressions,
//! sorts, and scripts. The AST can be pretty-printed to valid SMT-LIB2 text
//! for consumption by any conforming solver.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sorts
// ---------------------------------------------------------------------------

/// SMT-LIB2 sort (type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    /// Boolean sort.
    Bool,
    /// Fixed-width integer sort (bit-vectors are out of scope; we use Int).
    Int,
    /// Real sort (unused in QF_LIA but kept for generality).
    Real,
    /// Array sort `(Array <index> <element>)`.
    Array(Box<SmtSort>, Box<SmtSort>),
    /// Uninterpreted sort with a given name.
    Uninterpreted(String),
    /// Parameterized sort `(name arg1 arg2 ...)`.
    Parameterized(String, Vec<SmtSort>),
}

impl SmtSort {
    /// Shorthand for `(Array Int Int)`.
    pub fn int_array() -> Self {
        SmtSort::Array(Box::new(SmtSort::Int), Box::new(SmtSort::Int))
    }

    /// Whether this sort is numeric (`Int` or `Real`).
    pub fn is_numeric(&self) -> bool {
        matches!(self, SmtSort::Int | SmtSort::Real)
    }

    /// Whether this sort is `Bool`.
    pub fn is_bool(&self) -> bool {
        matches!(self, SmtSort::Bool)
    }

    /// Whether this sort is an array.
    pub fn is_array(&self) -> bool {
        matches!(self, SmtSort::Array(..))
    }

    /// Return the element sort if this is an array, else `None`.
    pub fn array_element(&self) -> Option<&SmtSort> {
        match self {
            SmtSort::Array(_, elem) => Some(elem),
            _ => None,
        }
    }

    /// Return the index sort if this is an array, else `None`.
    pub fn array_index(&self) -> Option<&SmtSort> {
        match self {
            SmtSort::Array(idx, _) => Some(idx),
            _ => None,
        }
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::Int => write!(f, "Int"),
            SmtSort::Real => write!(f, "Real"),
            SmtSort::Array(idx, elem) => write!(f, "(Array {} {})", idx, elem),
            SmtSort::Uninterpreted(name) => write!(f, "{}", name),
            SmtSort::Parameterized(name, args) => {
                write!(f, "({}", name)?;
                for a in args {
                    write!(f, " {}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// SMT-LIB2 expression (term).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtExpr {
    /// Integer literal.
    IntLit(i64),
    /// Boolean literal.
    BoolLit(bool),
    /// Symbol (variable or function name).
    Symbol(String),
    /// Quoted symbol `|...|`.
    QuotedSymbol(String),
    /// String literal.
    StringLit(String),

    // -- Core theory --
    /// `true`
    True,
    /// `false`
    False,
    /// `(not e)`
    Not(Box<SmtExpr>),
    /// `(and e1 e2 ...)`
    And(Vec<SmtExpr>),
    /// `(or e1 e2 ...)`
    Or(Vec<SmtExpr>),
    /// `(=> e1 e2)`
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    /// `(= e1 e2)`
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    /// `(distinct e1 e2 ...)`
    Distinct(Vec<SmtExpr>),
    /// `(ite cond then else)`
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // -- Arithmetic --
    /// `(+ e1 e2 ...)`
    Add(Vec<SmtExpr>),
    /// `(- e1 e2)` or `(- e)` for unary
    Sub(Vec<SmtExpr>),
    /// `(* e1 e2 ...)`
    Mul(Vec<SmtExpr>),
    /// `(div e1 e2)` — integer division
    Div(Box<SmtExpr>, Box<SmtExpr>),
    /// `(mod e1 e2)`
    Mod(Box<SmtExpr>, Box<SmtExpr>),
    /// `(abs e)`
    Abs(Box<SmtExpr>),

    // -- Comparison --
    /// `(< e1 e2)`
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    /// `(<= e1 e2)`
    Le(Box<SmtExpr>, Box<SmtExpr>),
    /// `(> e1 e2)`
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    /// `(>= e1 e2)`
    Ge(Box<SmtExpr>, Box<SmtExpr>),

    // -- Array theory --
    /// `(select array index)`
    Select(Box<SmtExpr>, Box<SmtExpr>),
    /// `(store array index value)`
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),

    // -- Quantifiers --
    /// `(forall ((x1 s1) (x2 s2) ...) body)`
    Forall(Vec<(String, SmtSort)>, Box<SmtExpr>),
    /// `(exists ((x1 s1) (x2 s2) ...) body)`
    Exists(Vec<(String, SmtSort)>, Box<SmtExpr>),

    // -- Let bindings --
    /// `(let ((x1 e1) (x2 e2) ...) body)`
    Let(Vec<(String, SmtExpr)>, Box<SmtExpr>),

    // -- Generic application --
    /// `(f arg1 arg2 ...)`
    Apply(String, Vec<SmtExpr>),

    // -- Annotations --
    /// `(! expr :named name)` or other attributes
    Annotated(Box<SmtExpr>, Vec<(String, String)>),
}

impl SmtExpr {
    // -- Constructors --

    /// Create an integer literal.
    pub fn int(v: i64) -> Self {
        SmtExpr::IntLit(v)
    }

    /// Create a boolean literal.
    pub fn bool_lit(v: bool) -> Self {
        SmtExpr::BoolLit(v)
    }

    /// Create a symbol reference.
    pub fn sym(name: impl Into<String>) -> Self {
        SmtExpr::Symbol(name.into())
    }

    /// Create `(not e)`.
    pub fn not(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    /// Create `(and e1 e2)`.
    pub fn and(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::And(vec![a, b])
    }

    /// Create a multi-way `(and e1 e2 ...)`.
    pub fn and_many(es: Vec<SmtExpr>) -> Self {
        match es.len() {
            0 => SmtExpr::True,
            1 => es.into_iter().next().unwrap(),
            _ => SmtExpr::And(es),
        }
    }

    /// Create `(or e1 e2)`.
    pub fn or(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Or(vec![a, b])
    }

    /// Create a multi-way `(or e1 e2 ...)`.
    pub fn or_many(es: Vec<SmtExpr>) -> Self {
        match es.len() {
            0 => SmtExpr::False,
            1 => es.into_iter().next().unwrap(),
            _ => SmtExpr::Or(es),
        }
    }

    /// Create `(=> a b)`.
    pub fn implies(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(a), Box::new(b))
    }

    /// Create `(= a b)`.
    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(a), Box::new(b))
    }

    /// Create `(ite c t e)`.
    pub fn ite(c: SmtExpr, t: SmtExpr, e: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e))
    }

    /// Create `(+ a b)`.
    pub fn add(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Add(vec![a, b])
    }

    /// Create `(- a b)`.
    pub fn sub(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Sub(vec![a, b])
    }

    /// Create `(* a b)`.
    pub fn mul(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Mul(vec![a, b])
    }

    /// Create `(< a b)`.
    pub fn lt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Lt(Box::new(a), Box::new(b))
    }

    /// Create `(<= a b)`.
    pub fn le(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Le(Box::new(a), Box::new(b))
    }

    /// Create `(> a b)`.
    pub fn gt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Gt(Box::new(a), Box::new(b))
    }

    /// Create `(>= a b)`.
    pub fn ge(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Ge(Box::new(a), Box::new(b))
    }

    /// Create `(select arr idx)`.
    pub fn select(arr: SmtExpr, idx: SmtExpr) -> Self {
        SmtExpr::Select(Box::new(arr), Box::new(idx))
    }

    /// Create `(store arr idx val)`.
    pub fn store(arr: SmtExpr, idx: SmtExpr, val: SmtExpr) -> Self {
        SmtExpr::Store(Box::new(arr), Box::new(idx), Box::new(val))
    }

    /// Create `(div a b)`.
    pub fn div(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Div(Box::new(a), Box::new(b))
    }

    /// Create `(mod a b)`.
    pub fn modulo(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Mod(Box::new(a), Box::new(b))
    }

    /// Create `(abs e)`.
    pub fn abs(e: SmtExpr) -> Self {
        SmtExpr::Abs(Box::new(e))
    }

    /// Create `(f arg1 arg2 ...)`.
    pub fn apply(f: impl Into<String>, args: Vec<SmtExpr>) -> Self {
        SmtExpr::Apply(f.into(), args)
    }

    /// Annotate with `:named`.
    pub fn named(self, name: impl Into<String>) -> Self {
        SmtExpr::Annotated(Box::new(self), vec![("named".to_string(), name.into())])
    }

    // -- Queries --

    /// Collect all free symbols (variables) in this expression.
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms = Vec::new();
        self.collect_symbols(&mut syms);
        syms.sort();
        syms.dedup();
        syms
    }

    fn collect_symbols(&self, out: &mut Vec<String>) {
        match self {
            SmtExpr::Symbol(s) => out.push(s.clone()),
            SmtExpr::QuotedSymbol(s) => out.push(s.clone()),
            SmtExpr::Not(e) | SmtExpr::Abs(e) => e.collect_symbols(out),
            SmtExpr::And(es)
            | SmtExpr::Or(es)
            | SmtExpr::Distinct(es)
            | SmtExpr::Add(es)
            | SmtExpr::Sub(es)
            | SmtExpr::Mul(es) => {
                for e in es {
                    e.collect_symbols(out);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Mod(a, b)
            | SmtExpr::Select(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            SmtExpr::Store(a, b, c) | SmtExpr::Ite(a, b, c) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
                c.collect_symbols(out);
            }
            SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
                let bound: std::collections::HashSet<_> =
                    bindings.iter().map(|(n, _)| n.as_str()).collect();
                let mut inner = Vec::new();
                body.collect_symbols(&mut inner);
                for s in inner {
                    if !bound.contains(s.as_str()) {
                        out.push(s);
                    }
                }
            }
            SmtExpr::Let(bindings, body) => {
                for (_, e) in bindings {
                    e.collect_symbols(out);
                }
                let bound: std::collections::HashSet<_> =
                    bindings.iter().map(|(n, _)| n.as_str()).collect();
                let mut inner = Vec::new();
                body.collect_symbols(&mut inner);
                for s in inner {
                    if !bound.contains(s.as_str()) {
                        out.push(s);
                    }
                }
            }
            SmtExpr::Apply(_, args) => {
                for a in args {
                    a.collect_symbols(out);
                }
            }
            SmtExpr::Annotated(e, _) => e.collect_symbols(out),
            SmtExpr::IntLit(_)
            | SmtExpr::BoolLit(_)
            | SmtExpr::StringLit(_)
            | SmtExpr::True
            | SmtExpr::False => {}
        }
    }

    /// Compute the AST depth.
    pub fn depth(&self) -> usize {
        match self {
            SmtExpr::IntLit(_)
            | SmtExpr::BoolLit(_)
            | SmtExpr::StringLit(_)
            | SmtExpr::Symbol(_)
            | SmtExpr::QuotedSymbol(_)
            | SmtExpr::True
            | SmtExpr::False => 1,
            SmtExpr::Not(e) | SmtExpr::Abs(e) => 1 + e.depth(),
            SmtExpr::And(es)
            | SmtExpr::Or(es)
            | SmtExpr::Distinct(es)
            | SmtExpr::Add(es)
            | SmtExpr::Sub(es)
            | SmtExpr::Mul(es) => 1 + es.iter().map(|e| e.depth()).max().unwrap_or(0),
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Mod(a, b)
            | SmtExpr::Select(a, b) => 1 + a.depth().max(b.depth()),
            SmtExpr::Store(a, b, c) | SmtExpr::Ite(a, b, c) => {
                1 + a.depth().max(b.depth()).max(c.depth())
            }
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => 1 + body.depth(),
            SmtExpr::Let(bindings, body) => {
                let bd = bindings.iter().map(|(_, e)| e.depth()).max().unwrap_or(0);
                1 + bd.max(body.depth())
            }
            SmtExpr::Apply(_, args) => 1 + args.iter().map(|a| a.depth()).max().unwrap_or(0),
            SmtExpr::Annotated(e, _) => e.depth(),
        }
    }

    /// Count total AST nodes.
    pub fn node_count(&self) -> usize {
        match self {
            SmtExpr::IntLit(_)
            | SmtExpr::BoolLit(_)
            | SmtExpr::StringLit(_)
            | SmtExpr::Symbol(_)
            | SmtExpr::QuotedSymbol(_)
            | SmtExpr::True
            | SmtExpr::False => 1,
            SmtExpr::Not(e) | SmtExpr::Abs(e) => 1 + e.node_count(),
            SmtExpr::And(es)
            | SmtExpr::Or(es)
            | SmtExpr::Distinct(es)
            | SmtExpr::Add(es)
            | SmtExpr::Sub(es)
            | SmtExpr::Mul(es) => 1 + es.iter().map(|e| e.node_count()).sum::<usize>(),
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Mod(a, b)
            | SmtExpr::Select(a, b) => 1 + a.node_count() + b.node_count(),
            SmtExpr::Store(a, b, c) | SmtExpr::Ite(a, b, c) => {
                1 + a.node_count() + b.node_count() + c.node_count()
            }
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => 1 + body.node_count(),
            SmtExpr::Let(bindings, body) => {
                1 + bindings.iter().map(|(_, e)| e.node_count()).sum::<usize>() + body.node_count()
            }
            SmtExpr::Apply(_, args) => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
            SmtExpr::Annotated(e, _) => e.node_count(),
        }
    }

    /// Substitute a symbol with an expression throughout.
    pub fn substitute(&self, name: &str, replacement: &SmtExpr) -> SmtExpr {
        match self {
            SmtExpr::Symbol(s) if s == name => replacement.clone(),
            SmtExpr::Symbol(_)
            | SmtExpr::QuotedSymbol(_)
            | SmtExpr::IntLit(_)
            | SmtExpr::BoolLit(_)
            | SmtExpr::StringLit(_)
            | SmtExpr::True
            | SmtExpr::False => self.clone(),
            SmtExpr::Not(e) => SmtExpr::Not(Box::new(e.substitute(name, replacement))),
            SmtExpr::Abs(e) => SmtExpr::Abs(Box::new(e.substitute(name, replacement))),
            SmtExpr::And(es) => {
                SmtExpr::And(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Or(es) => {
                SmtExpr::Or(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Distinct(es) => {
                SmtExpr::Distinct(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Add(es) => {
                SmtExpr::Add(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Sub(es) => {
                SmtExpr::Sub(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Mul(es) => {
                SmtExpr::Mul(es.iter().map(|e| e.substitute(name, replacement)).collect())
            }
            SmtExpr::Implies(a, b) => SmtExpr::Implies(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Eq(a, b) => SmtExpr::Eq(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Lt(a, b) => SmtExpr::Lt(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Le(a, b) => SmtExpr::Le(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Gt(a, b) => SmtExpr::Gt(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Ge(a, b) => SmtExpr::Ge(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Div(a, b) => SmtExpr::Div(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Mod(a, b) => SmtExpr::Mod(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Select(a, b) => SmtExpr::Select(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
            ),
            SmtExpr::Store(a, b, c) => SmtExpr::Store(
                Box::new(a.substitute(name, replacement)),
                Box::new(b.substitute(name, replacement)),
                Box::new(c.substitute(name, replacement)),
            ),
            SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
                Box::new(c.substitute(name, replacement)),
                Box::new(t.substitute(name, replacement)),
                Box::new(e.substitute(name, replacement)),
            ),
            SmtExpr::Forall(binds, body) => {
                if binds.iter().any(|(n, _)| n == name) {
                    self.clone()
                } else {
                    SmtExpr::Forall(binds.clone(), Box::new(body.substitute(name, replacement)))
                }
            }
            SmtExpr::Exists(binds, body) => {
                if binds.iter().any(|(n, _)| n == name) {
                    self.clone()
                } else {
                    SmtExpr::Exists(binds.clone(), Box::new(body.substitute(name, replacement)))
                }
            }
            SmtExpr::Let(binds, body) => {
                let new_binds: Vec<_> = binds
                    .iter()
                    .map(|(n, e)| (n.clone(), e.substitute(name, replacement)))
                    .collect();
                if binds.iter().any(|(n, _)| n == name) {
                    SmtExpr::Let(new_binds, body.clone())
                } else {
                    SmtExpr::Let(new_binds, Box::new(body.substitute(name, replacement)))
                }
            }
            SmtExpr::Apply(f, args) => SmtExpr::Apply(
                f.clone(),
                args.iter()
                    .map(|a| a.substitute(name, replacement))
                    .collect(),
            ),
            SmtExpr::Annotated(e, attrs) => {
                SmtExpr::Annotated(Box::new(e.substitute(name, replacement)), attrs.clone())
            }
        }
    }

    /// Simple constant folding for arithmetic.
    pub fn simplify(&self) -> SmtExpr {
        match self {
            SmtExpr::Add(es) => {
                let simplified: Vec<_> = es.iter().map(|e| e.simplify()).collect();
                let (consts, rest): (Vec<_>, Vec<_>) = simplified
                    .into_iter()
                    .partition(|e| matches!(e, SmtExpr::IntLit(_)));
                let sum: i64 = consts
                    .iter()
                    .map(|e| match e {
                        SmtExpr::IntLit(v) => *v,
                        _ => 0,
                    })
                    .sum();
                let mut result = rest;
                if sum != 0 || result.is_empty() {
                    result.push(SmtExpr::IntLit(sum));
                }
                if result.len() == 1 {
                    result.into_iter().next().unwrap()
                } else {
                    SmtExpr::Add(result)
                }
            }
            SmtExpr::And(es) => {
                let simplified: Vec<_> = es.iter().map(|e| e.simplify()).collect();
                let mut result = Vec::new();
                for e in simplified {
                    if e == SmtExpr::False {
                        return SmtExpr::False;
                    }
                    if e != SmtExpr::True {
                        result.push(e);
                    }
                }
                SmtExpr::and_many(result)
            }
            SmtExpr::Or(es) => {
                let simplified: Vec<_> = es.iter().map(|e| e.simplify()).collect();
                let mut result = Vec::new();
                for e in simplified {
                    if e == SmtExpr::True {
                        return SmtExpr::True;
                    }
                    if e != SmtExpr::False {
                        result.push(e);
                    }
                }
                SmtExpr::or_many(result)
            }
            SmtExpr::Not(e) => {
                let inner = e.simplify();
                match inner {
                    SmtExpr::True => SmtExpr::False,
                    SmtExpr::False => SmtExpr::True,
                    SmtExpr::Not(e2) => *e2,
                    other => SmtExpr::Not(Box::new(other)),
                }
            }
            SmtExpr::Ite(c, t, e) => {
                let cs = c.simplify();
                let ts = t.simplify();
                let es = e.simplify();
                match cs {
                    SmtExpr::True | SmtExpr::BoolLit(true) => ts,
                    SmtExpr::False | SmtExpr::BoolLit(false) => es,
                    _ => {
                        if ts == es {
                            ts
                        } else {
                            SmtExpr::Ite(Box::new(cs), Box::new(ts), Box::new(es))
                        }
                    }
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtExpr::IntLit(v) => {
                if *v < 0 {
                    write!(f, "(- {})", -v)
                } else {
                    write!(f, "{}", v)
                }
            }
            SmtExpr::BoolLit(b) => write!(f, "{}", b),
            SmtExpr::Symbol(s) => write!(f, "{}", s),
            SmtExpr::QuotedSymbol(s) => write!(f, "|{}|", s),
            SmtExpr::StringLit(s) => write!(f, "\"{}\"", s),
            SmtExpr::True => write!(f, "true"),
            SmtExpr::False => write!(f, "false"),
            SmtExpr::Not(e) => write!(f, "(not {})", e),
            SmtExpr::And(es) => write_nary(f, "and", es),
            SmtExpr::Or(es) => write_nary(f, "or", es),
            SmtExpr::Implies(a, b) => write!(f, "(=> {} {})", a, b),
            SmtExpr::Eq(a, b) => write!(f, "(= {} {})", a, b),
            SmtExpr::Distinct(es) => write_nary(f, "distinct", es),
            SmtExpr::Ite(c, t, e) => write!(f, "(ite {} {} {})", c, t, e),
            SmtExpr::Add(es) => write_nary(f, "+", es),
            SmtExpr::Sub(es) => write_nary(f, "-", es),
            SmtExpr::Mul(es) => write_nary(f, "*", es),
            SmtExpr::Div(a, b) => write!(f, "(div {} {})", a, b),
            SmtExpr::Mod(a, b) => write!(f, "(mod {} {})", a, b),
            SmtExpr::Abs(e) => write!(f, "(abs {})", e),
            SmtExpr::Lt(a, b) => write!(f, "(< {} {})", a, b),
            SmtExpr::Le(a, b) => write!(f, "(<= {} {})", a, b),
            SmtExpr::Gt(a, b) => write!(f, "(> {} {})", a, b),
            SmtExpr::Ge(a, b) => write!(f, "(>= {} {})", a, b),
            SmtExpr::Select(a, i) => write!(f, "(select {} {})", a, i),
            SmtExpr::Store(a, i, v) => write!(f, "(store {} {} {})", a, i, v),
            SmtExpr::Forall(binds, body) => {
                write!(f, "(forall (")?;
                for (i, (name, sort)) in binds.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", name, sort)?;
                }
                write!(f, ") {})", body)
            }
            SmtExpr::Exists(binds, body) => {
                write!(f, "(exists (")?;
                for (i, (name, sort)) in binds.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", name, sort)?;
                }
                write!(f, ") {})", body)
            }
            SmtExpr::Let(binds, body) => {
                write!(f, "(let (")?;
                for (i, (name, expr)) in binds.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", name, expr)?;
                }
                write!(f, ") {})", body)
            }
            SmtExpr::Apply(fun, args) => {
                write!(f, "({}", fun)?;
                for a in args {
                    write!(f, " {}", a)?;
                }
                write!(f, ")")
            }
            SmtExpr::Annotated(e, attrs) => {
                write!(f, "(! {}", e)?;
                for (k, v) in attrs {
                    write!(f, " :{} {}", k, v)?;
                }
                write!(f, ")")
            }
        }
    }
}

fn write_nary(f: &mut fmt::Formatter<'_>, op: &str, es: &[SmtExpr]) -> fmt::Result {
    write!(f, "({}", op)?;
    for e in es {
        write!(f, " {}", e)?;
    }
    write!(f, ")")
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// SMT-LIB2 command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtCommand {
    /// `(set-logic <logic>)`
    SetLogic(String),
    /// `(set-option <key> <value>)`
    SetOption(String, String),
    /// `(set-info <key> <value>)`
    SetInfo(String, String),
    /// `(declare-sort <name> <arity>)`
    DeclareSort(String, usize),
    /// `(define-sort <name> (<params>) <sort>)`
    DefineSort(String, Vec<String>, SmtSort),
    /// `(declare-fun <name> (<arg-sorts>) <return-sort>)`
    DeclareFun(String, Vec<SmtSort>, SmtSort),
    /// `(define-fun <name> (<args>) <return-sort> <body>)`
    DefineFun(String, Vec<(String, SmtSort)>, SmtSort, SmtExpr),
    /// `(declare-const <name> <sort>)`
    DeclareConst(String, SmtSort),
    /// `(assert <expr>)`
    Assert(SmtExpr),
    /// `(check-sat)`
    CheckSat,
    /// `(get-model)`
    GetModel,
    /// `(get-value (<terms>))`
    GetValue(Vec<SmtExpr>),
    /// `(get-unsat-core)`
    GetUnsatCore,
    /// `(push <n>)`
    Push(u32),
    /// `(pop <n>)`
    Pop(u32),
    /// `(reset)`
    Reset,
    /// `(exit)`
    Exit,
    /// `(echo <string>)`
    Echo(String),
    /// Raw SMT-LIB2 text (escape hatch).
    Raw(String),
}

impl fmt::Display for SmtCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtCommand::SetLogic(l) => write!(f, "(set-logic {})", l),
            SmtCommand::SetOption(k, v) => write!(f, "(set-option :{} {})", k, v),
            SmtCommand::SetInfo(k, v) => write!(f, "(set-info :{} {})", k, v),
            SmtCommand::DeclareSort(n, a) => write!(f, "(declare-sort {} {})", n, a),
            SmtCommand::DefineSort(n, params, s) => {
                write!(f, "(define-sort {} (", n)?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") {})", s)
            }
            SmtCommand::DeclareFun(n, args, ret) => {
                write!(f, "(declare-fun {} (", n)?;
                for (i, s) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", s)?;
                }
                write!(f, ") {})", ret)
            }
            SmtCommand::DefineFun(n, args, ret, body) => {
                write!(f, "(define-fun {} (", n)?;
                for (i, (name, sort)) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", name, sort)?;
                }
                write!(f, ") {} {})", ret, body)
            }
            SmtCommand::DeclareConst(n, s) => write!(f, "(declare-const {} {})", n, s),
            SmtCommand::Assert(e) => write!(f, "(assert {})", e),
            SmtCommand::CheckSat => write!(f, "(check-sat)"),
            SmtCommand::GetModel => write!(f, "(get-model)"),
            SmtCommand::GetValue(terms) => {
                write!(f, "(get-value (")?;
                for (i, t) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, "))")
            }
            SmtCommand::GetUnsatCore => write!(f, "(get-unsat-core)"),
            SmtCommand::Push(n) => write!(f, "(push {})", n),
            SmtCommand::Pop(n) => write!(f, "(pop {})", n),
            SmtCommand::Reset => write!(f, "(reset)"),
            SmtCommand::Exit => write!(f, "(exit)"),
            SmtCommand::Echo(s) => write!(f, "(echo \"{}\")", s),
            SmtCommand::Raw(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Script
// ---------------------------------------------------------------------------

/// An SMT-LIB2 script — an ordered sequence of commands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmtScript {
    pub commands: Vec<SmtCommand>,
}

impl SmtScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        SmtScript {
            commands: Vec::with_capacity(cap),
        }
    }

    pub fn push(&mut self, cmd: SmtCommand) {
        self.commands.push(cmd);
    }

    pub fn set_logic(&mut self, logic: &str) {
        self.commands.push(SmtCommand::SetLogic(logic.to_string()));
    }

    pub fn declare_const(&mut self, name: &str, sort: SmtSort) {
        self.commands
            .push(SmtCommand::DeclareConst(name.to_string(), sort));
    }

    pub fn declare_fun(&mut self, name: &str, args: Vec<SmtSort>, ret: SmtSort) {
        self.commands
            .push(SmtCommand::DeclareFun(name.to_string(), args, ret));
    }

    pub fn assert(&mut self, expr: SmtExpr) {
        self.commands.push(SmtCommand::Assert(expr));
    }

    pub fn check_sat(&mut self) {
        self.commands.push(SmtCommand::CheckSat);
    }

    pub fn get_model(&mut self) {
        self.commands.push(SmtCommand::GetModel);
    }

    pub fn get_unsat_core(&mut self) {
        self.commands.push(SmtCommand::GetUnsatCore);
    }

    pub fn push_scope(&mut self, n: u32) {
        self.commands.push(SmtCommand::Push(n));
    }

    pub fn pop_scope(&mut self, n: u32) {
        self.commands.push(SmtCommand::Pop(n));
    }

    pub fn exit(&mut self) {
        self.commands.push(SmtCommand::Exit);
    }

    /// Render the complete script to a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for cmd in &self.commands {
            out.push_str(&format!("{}\n", cmd));
        }
        out
    }

    /// Count the number of assertions.
    pub fn assertion_count(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| matches!(c, SmtCommand::Assert(_)))
            .count()
    }

    /// Get all declared constant names.
    pub fn declared_consts(&self) -> Vec<String> {
        self.commands
            .iter()
            .filter_map(|c| match c {
                SmtCommand::DeclareConst(n, _) => Some(n.clone()),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for SmtScript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_display() {
        assert_eq!(SmtSort::Int.to_string(), "Int");
        assert_eq!(SmtSort::Bool.to_string(), "Bool");
        assert_eq!(SmtSort::int_array().to_string(), "(Array Int Int)");
    }

    #[test]
    fn test_expr_display_int_lit() {
        assert_eq!(SmtExpr::int(42).to_string(), "42");
        assert_eq!(SmtExpr::int(-5).to_string(), "(- 5)");
    }

    #[test]
    fn test_expr_display_comparison() {
        let e = SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10));
        assert_eq!(e.to_string(), "(<= x 10)");
    }

    #[test]
    fn test_expr_display_nested() {
        let e = SmtExpr::and(
            SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)),
            SmtExpr::gt(SmtExpr::sym("y"), SmtExpr::int(0)),
        );
        assert_eq!(e.to_string(), "(and (<= x 10) (> y 0))");
    }

    #[test]
    fn test_free_symbols() {
        let e = SmtExpr::and(
            SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)),
            SmtExpr::gt(SmtExpr::sym("y"), SmtExpr::sym("x")),
        );
        let syms = e.free_symbols();
        assert_eq!(syms, vec!["x", "y"]);
    }

    #[test]
    fn test_free_symbols_with_quantifier() {
        let body = SmtExpr::le(SmtExpr::sym("x"), SmtExpr::sym("y"));
        let e = SmtExpr::Forall(vec![("x".to_string(), SmtSort::Int)], Box::new(body));
        let syms = e.free_symbols();
        assert_eq!(syms, vec!["y"]);
    }

    #[test]
    fn test_depth() {
        let e = SmtExpr::and(
            SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)),
            SmtExpr::True,
        );
        assert_eq!(e.depth(), 3);
    }

    #[test]
    fn test_node_count() {
        let e = SmtExpr::add(SmtExpr::sym("x"), SmtExpr::int(1));
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_substitute() {
        let e = SmtExpr::add(SmtExpr::sym("x"), SmtExpr::int(1));
        let replaced = e.substitute("x", &SmtExpr::int(42));
        assert_eq!(replaced, SmtExpr::add(SmtExpr::int(42), SmtExpr::int(1)));
    }

    #[test]
    fn test_simplify_and_true() {
        let e = SmtExpr::And(vec![SmtExpr::True, SmtExpr::sym("p"), SmtExpr::True]);
        let s = e.simplify();
        assert_eq!(s, SmtExpr::sym("p"));
    }

    #[test]
    fn test_simplify_and_false() {
        let e = SmtExpr::And(vec![SmtExpr::sym("p"), SmtExpr::False]);
        let s = e.simplify();
        assert_eq!(s, SmtExpr::False);
    }

    #[test]
    fn test_simplify_double_negation() {
        let e = SmtExpr::not(SmtExpr::not(SmtExpr::sym("p")));
        let s = e.simplify();
        assert_eq!(s, SmtExpr::sym("p"));
    }

    #[test]
    fn test_simplify_ite_true_cond() {
        let e = SmtExpr::ite(SmtExpr::True, SmtExpr::int(1), SmtExpr::int(2));
        assert_eq!(e.simplify(), SmtExpr::int(1));
    }

    #[test]
    fn test_script_render() {
        let mut s = SmtScript::new();
        s.set_logic("QF_LIA");
        s.declare_const("x", SmtSort::Int);
        s.assert(SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)));
        s.check_sat();
        let rendered = s.render();
        assert!(rendered.contains("(set-logic QF_LIA)"));
        assert!(rendered.contains("(declare-const x Int)"));
        assert!(rendered.contains("(assert (<= x 10))"));
        assert!(rendered.contains("(check-sat)"));
    }

    #[test]
    fn test_script_assertion_count() {
        let mut s = SmtScript::new();
        s.assert(SmtExpr::True);
        s.assert(SmtExpr::False);
        s.check_sat();
        assert_eq!(s.assertion_count(), 2);
    }

    #[test]
    fn test_command_display_declare_fun() {
        let cmd = SmtCommand::DeclareFun(
            "f".to_string(),
            vec![SmtSort::Int, SmtSort::Int],
            SmtSort::Bool,
        );
        assert_eq!(cmd.to_string(), "(declare-fun f (Int Int) Bool)");
    }

    #[test]
    fn test_and_many_empty() {
        assert_eq!(SmtExpr::and_many(vec![]), SmtExpr::True);
    }

    #[test]
    fn test_or_many_single() {
        assert_eq!(SmtExpr::or_many(vec![SmtExpr::sym("p")]), SmtExpr::sym("p"));
    }

    #[test]
    fn test_sort_properties() {
        assert!(SmtSort::Int.is_numeric());
        assert!(!SmtSort::Bool.is_numeric());
        assert!(SmtSort::Bool.is_bool());
        assert!(SmtSort::int_array().is_array());
        assert_eq!(SmtSort::int_array().array_element(), Some(&SmtSort::Int));
    }
}
