//! Constraint types for SMT encoding.
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
    BitVec(u32),
    Array(Box<SmtSort>, Box<SmtSort>),
    Uninterpreted(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    Var(String, SmtSort),
    /// Sort-agnostic constant reference (inferred by solver).
    Const(String),
    IntLit(i64),
    BoolLit(bool),
    RealLit(f64),
    BitVecLit(u64, u32),
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Iff(Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Ne(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),
    Add(Box<SmtExpr>, Box<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Box<SmtExpr>, Box<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Select(Box<SmtExpr>, Box<SmtExpr>),
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Apply(String, Vec<SmtExpr>),
    Forall(Vec<(String, SmtSort)>, Box<SmtExpr>),
    Exists(Vec<(String, SmtSort)>, Box<SmtExpr>),
    /// Alias for Forall (alternate casing accepted by isospec-smt).
    ForAll(Vec<(String, SmtSort)>, Box<SmtExpr>),
    Distinct(Vec<SmtExpr>),
    Let(Vec<(String, SmtExpr)>, Box<SmtExpr>),
    Comment(String),
    // Bit-vector operations
    BvAnd(Box<SmtExpr>, Box<SmtExpr>),
    BvOr(Box<SmtExpr>, Box<SmtExpr>),
    BvAdd(Box<SmtExpr>, Box<SmtExpr>),
    BvSub(Box<SmtExpr>, Box<SmtExpr>),
    BvMul(Box<SmtExpr>, Box<SmtExpr>),
    BvUlt(Box<SmtExpr>, Box<SmtExpr>),
    BvSlt(Box<SmtExpr>, Box<SmtExpr>),
    Extract(u32, u32, Box<SmtExpr>),
    ZeroExtend(u32, Box<SmtExpr>),
    SignExtend(u32, Box<SmtExpr>),
    Concat(Box<SmtExpr>, Box<SmtExpr>),
}

impl SmtExpr {
    pub fn var(name: impl Into<String>, sort: SmtSort) -> Self {
        Self::Var(name.into(), sort)
    }
    pub fn bool_var(name: impl Into<String>) -> Self {
        Self::Var(name.into(), SmtSort::Bool)
    }
    pub fn int_var(name: impl Into<String>) -> Self {
        Self::Var(name.into(), SmtSort::Int)
    }
    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        let filtered: Vec<SmtExpr> = exprs.into_iter().filter(|e| !matches!(e, SmtExpr::BoolLit(true))).collect();
        if filtered.is_empty() { SmtExpr::BoolLit(true) }
        else if filtered.len() == 1 { filtered.into_iter().next().unwrap() }
        else if filtered.iter().any(|e| matches!(e, SmtExpr::BoolLit(false))) { SmtExpr::BoolLit(false) }
        else { SmtExpr::And(filtered) }
    }
    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        let filtered: Vec<SmtExpr> = exprs.into_iter().filter(|e| !matches!(e, SmtExpr::BoolLit(false))).collect();
        if filtered.is_empty() { SmtExpr::BoolLit(false) }
        else if filtered.len() == 1 { filtered.into_iter().next().unwrap() }
        else if filtered.iter().any(|e| matches!(e, SmtExpr::BoolLit(true))) { SmtExpr::BoolLit(true) }
        else { SmtExpr::Or(filtered) }
    }
    pub fn implies(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(a), Box::new(b))
    }
    pub fn not(e: SmtExpr) -> Self {
        match e {
            SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
            SmtExpr::Not(inner) => *inner,
            other => SmtExpr::Not(Box::new(other)),
        }
    }
    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self { SmtExpr::Eq(Box::new(a), Box::new(b)) }
    pub fn lt(a: SmtExpr, b: SmtExpr) -> Self { SmtExpr::Lt(Box::new(a), Box::new(b)) }
    pub fn le(a: SmtExpr, b: SmtExpr) -> Self { SmtExpr::Le(Box::new(a), Box::new(b)) }

    pub fn to_smtlib2(&self) -> String {
        match self {
            Self::Var(name, _) | Self::Const(name) => name.clone(),
            Self::IntLit(i) => if *i < 0 { format!("(- {})", -i) } else { i.to_string() },
            Self::BoolLit(b) => if *b { "true".into() } else { "false".into() },
            Self::RealLit(r) => format!("{:.6}", r),
            Self::BitVecLit(val, width) => format!("(_ bv{} {})", val, width),
            Self::Not(e) => format!("(not {})", e.to_smtlib2()),
            Self::And(es) => { let parts: Vec<_> = es.iter().map(|e| e.to_smtlib2()).collect(); format!("(and {})", parts.join(" ")) }
            Self::Or(es) => { let parts: Vec<_> = es.iter().map(|e| e.to_smtlib2()).collect(); format!("(or {})", parts.join(" ")) }
            Self::Distinct(es) => { let parts: Vec<_> = es.iter().map(|e| e.to_smtlib2()).collect(); format!("(distinct {})", parts.join(" ")) }
            Self::Implies(a, b) => format!("(=> {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Iff(a, b) => format!("(= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Eq(a, b) => format!("(= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Ne(a, b) => format!("(not (= {} {}))", a.to_smtlib2(), b.to_smtlib2()),
            Self::Lt(a, b) => format!("(< {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Le(a, b) => format!("(<= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Gt(a, b) => format!("(> {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Ge(a, b) => format!("(>= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Add(a, b) => format!("(+ {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Sub(a, b) => format!("(- {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Mul(a, b) => format!("(* {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvAnd(a, b) => format!("(bvand {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvOr(a, b) => format!("(bvor {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvAdd(a, b) => format!("(bvadd {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvSub(a, b) => format!("(bvsub {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvMul(a, b) => format!("(bvmul {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvUlt(a, b) => format!("(bvult {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::BvSlt(a, b) => format!("(bvslt {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Concat(a, b) => format!("(concat {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Self::Ite(c, t, e) => format!("(ite {} {} {})", c.to_smtlib2(), t.to_smtlib2(), e.to_smtlib2()),
            Self::Select(a, i) => format!("(select {} {})", a.to_smtlib2(), i.to_smtlib2()),
            Self::Store(a, i, v) => format!("(store {} {} {})", a.to_smtlib2(), i.to_smtlib2(), v.to_smtlib2()),
            Self::Extract(hi, lo, e) => format!("((_ extract {} {}) {})", hi, lo, e.to_smtlib2()),
            Self::ZeroExtend(n, e) => format!("((_ zero_extend {}) {})", n, e.to_smtlib2()),
            Self::SignExtend(n, e) => format!("((_ sign_extend {}) {})", n, e.to_smtlib2()),
            Self::Apply(f, args) => { let parts: Vec<_> = args.iter().map(|e| e.to_smtlib2()).collect(); format!("({} {})", f, parts.join(" ")) }
            Self::Forall(vars, body) | Self::ForAll(vars, body) => {
                let vs: Vec<_> = vars.iter().map(|(n, s)| format!("({} {})", n, s.to_smtlib2())).collect();
                format!("(forall ({}) {})", vs.join(" "), body.to_smtlib2())
            }
            Self::Exists(vars, body) => {
                let vs: Vec<_> = vars.iter().map(|(n, s)| format!("({} {})", n, s.to_smtlib2())).collect();
                format!("(exists ({}) {})", vs.join(" "), body.to_smtlib2())
            }
            Self::Let(bindings, body) => {
                let bs: Vec<_> = bindings.iter().map(|(n, e)| format!("({} {})", n, e.to_smtlib2())).collect();
                format!("(let ({}) {})", bs.join(" "), body.to_smtlib2())
            }
            Self::Comment(text) => format!("; {}", text),
        }
    }

    pub fn variable_count(&self) -> usize {
        match self {
            Self::Var(_, _) | Self::Const(_) => 1,
            Self::Not(e) => e.variable_count(),
            Self::And(es) | Self::Or(es) | Self::Distinct(es) => es.iter().map(|e| e.variable_count()).sum(),
            Self::Implies(a, b) | Self::Iff(a, b) | Self::Eq(a, b) | Self::Ne(a, b)
            | Self::Lt(a, b) | Self::Le(a, b) | Self::Gt(a, b) | Self::Ge(a, b)
            | Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b)
            | Self::BvAnd(a, b) | Self::BvOr(a, b) | Self::BvAdd(a, b) | Self::BvSub(a, b)
            | Self::BvMul(a, b) | Self::BvUlt(a, b) | Self::BvSlt(a, b) | Self::Concat(a, b)
            => a.variable_count() + b.variable_count(),
            Self::Ite(c, t, e) => c.variable_count() + t.variable_count() + e.variable_count(),
            _ => 0,
        }
    }
}

impl SmtSort {
    pub fn to_smtlib2(&self) -> String {
        match self {
            Self::Bool => "Bool".into(),
            Self::Int => "Int".into(),
            Self::Real => "Real".into(),
            Self::BitVec(n) => format!("(_ BitVec {})", n),
            Self::Array(k, v) => format!("(Array {} {})", k.to_smtlib2(), v.to_smtlib2()),
            Self::Uninterpreted(name) => name.clone(),
        }
    }

    /// Parse a sort name string into an SmtSort.
    pub fn from_name(name: &str) -> Self {
        match name {
            "Bool" => Self::Bool,
            "Int" => Self::Int,
            "Real" => Self::Real,
            _ => Self::Uninterpreted(name.to_string()),
        }
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib2())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtConstraintSet {
    pub declarations: Vec<(String, SmtSort)>,
    pub assertions: Vec<SmtExpr>,
    pub soft_assertions: Vec<(SmtExpr, u32, String)>,
    pub logic: String,
}

impl SmtConstraintSet {
    pub fn new(logic: impl Into<String>) -> Self {
        Self { declarations: Vec::new(), assertions: Vec::new(), soft_assertions: Vec::new(), logic: logic.into() }
    }
    pub fn declare(&mut self, name: impl Into<String>, sort: SmtSort) {
        self.declarations.push((name.into(), sort));
    }
    pub fn add_declaration(&mut self, name: impl Into<String>, sort_name: String) {
        self.declarations.push((name.into(), SmtSort::from_name(&sort_name)));
    }
    pub fn assert(&mut self, expr: SmtExpr) { self.assertions.push(expr); }
    pub fn add_assertion(&mut self, expr: SmtExpr) { self.assertions.push(expr); }
    pub fn assert_soft(&mut self, expr: SmtExpr, weight: u32, group: impl Into<String>) {
        self.soft_assertions.push((expr, weight, group.into()));
    }
    pub fn to_smtlib2(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("(set-logic {})", self.logic));
        for (name, sort) in &self.declarations {
            lines.push(format!("(declare-const {} {})", name, sort.to_smtlib2()));
        }
        for assertion in &self.assertions {
            lines.push(format!("(assert {})", assertion.to_smtlib2()));
        }
        for (expr, weight, group) in &self.soft_assertions {
            lines.push(format!("(assert-soft {} :weight {} :id {})", expr.to_smtlib2(), weight, group));
        }
        lines.push("(check-sat)".into());
        lines.push("(get-model)".into());
        lines.join("
")
    }
    pub fn constraint_count(&self) -> usize { self.assertions.len() + self.soft_assertions.len() }
    pub fn variable_count(&self) -> usize { self.declarations.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_smt_expr_simplification() {
        let e = SmtExpr::and(vec![SmtExpr::BoolLit(true), SmtExpr::bool_var("x")]);
        assert!(matches!(e, SmtExpr::Var(_, _)));
        let e2 = SmtExpr::and(vec![SmtExpr::BoolLit(false), SmtExpr::bool_var("x")]);
        assert!(matches!(e2, SmtExpr::BoolLit(false)));
    }
    #[test]
    fn test_smtlib2_output() {
        let e = SmtExpr::and(vec![
            SmtExpr::lt(SmtExpr::int_var("x"), SmtExpr::IntLit(10)),
            SmtExpr::ge(SmtExpr::int_var("x"), SmtExpr::IntLit(0)),
        ]);
        let s = e.to_smtlib2();
        assert!(s.contains("and"));
        assert!(s.contains("< x 10"));
    }
    #[test]
    fn test_constraint_set() {
        let mut cs = SmtConstraintSet::new("QF_LIA");
        cs.declare("x", SmtSort::Int);
        cs.assert(SmtExpr::lt(SmtExpr::int_var("x"), SmtExpr::IntLit(100)));
        let output = cs.to_smtlib2();
        assert!(output.contains("set-logic QF_LIA"));
        assert!(output.contains("declare-const x Int"));
        assert!(output.contains("check-sat"));
    }
}
