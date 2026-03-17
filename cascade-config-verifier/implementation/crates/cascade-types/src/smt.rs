//! SMT expression and formula types for cascade service-mesh analysis.
//!
//! This module defines the *data* layer — sorts, expressions, constraints,
//! formulas, models, and variable encodings — used to build SMT-LIB2 queries.
//! It intentionally does **not** contain solver integration; that lives in a
//! separate crate.

use std::collections::BTreeSet;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::service::ServiceId;

// ---------------------------------------------------------------------------
// SmtSort
// ---------------------------------------------------------------------------

/// SMT-LIB2 sort (type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    Real,
    BitVec(u32),
}

impl SmtSort {
    /// Render the sort in SMT-LIB2 syntax.
    pub fn smtlib_name(&self) -> String {
        match self {
            SmtSort::Bool => "Bool".to_string(),
            SmtSort::Int => "Int".to_string(),
            SmtSort::Real => "Real".to_string(),
            SmtSort::BitVec(n) => format!("(_ BitVec {})", n),
        }
    }

    /// Returns `true` for Int and Real sorts.
    pub fn is_numeric(&self) -> bool {
        matches!(self, SmtSort::Int | SmtSort::Real)
    }
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.smtlib_name())
    }
}

// ---------------------------------------------------------------------------
// SmtVariable
// ---------------------------------------------------------------------------

/// A typed SMT variable, optionally scoped to a service and/or time-step.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SmtVariable {
    pub name: String,
    pub sort: SmtSort,
    pub time_step: Option<u32>,
    pub service: Option<ServiceId>,
}

impl SmtVariable {
    pub fn new(name: impl Into<String>, sort: SmtSort) -> Self {
        Self {
            name: name.into(),
            sort,
            time_step: None,
            service: None,
        }
    }

    pub fn with_time_step(mut self, t: u32) -> Self {
        self.time_step = Some(t);
        self
    }

    pub fn with_service(mut self, s: ServiceId) -> Self {
        self.service = Some(s);
        self
    }

    /// Name that embeds the time-step suffix when present (e.g. `"load_t3"`).
    pub fn qualified_name(&self) -> String {
        match self.time_step {
            Some(t) => format!("{}_t{}", self.name, t),
            None => self.name.clone(),
        }
    }

    /// SMT-LIB2 `declare-const` statement.
    pub fn to_smtlib(&self) -> String {
        format!(
            "(declare-const {} {})",
            self.qualified_name(),
            self.sort.smtlib_name()
        )
    }
}

impl fmt::Display for SmtVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.qualified_name())
    }
}

// ---------------------------------------------------------------------------
// SmtExpr
// ---------------------------------------------------------------------------

/// Recursive SMT expression tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    Var(String),
    BoolConst(bool),
    IntConst(i64),
    RealConst(f64),

    // Arithmetic
    Add(Box<SmtExpr>, Box<SmtExpr>),
    Sub(Box<SmtExpr>, Box<SmtExpr>),
    Mul(Box<SmtExpr>, Box<SmtExpr>),
    Div(Box<SmtExpr>, Box<SmtExpr>),

    // Boolean
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Not(Box<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),

    // Comparison
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Gt(Box<SmtExpr>, Box<SmtExpr>),
    Ge(Box<SmtExpr>, Box<SmtExpr>),

    // Conditional
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
}

impl SmtExpr {
    // ---- helper constructors ------------------------------------------------

    pub fn var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into())
    }

    pub fn bool_const(b: bool) -> Self {
        SmtExpr::BoolConst(b)
    }

    pub fn int_const(i: i64) -> Self {
        SmtExpr::IntConst(i)
    }

    pub fn real_const(f: f64) -> Self {
        SmtExpr::RealConst(f)
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::And(exprs)
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Or(exprs)
    }

    pub fn not(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(a), Box::new(b))
    }

    pub fn lt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Lt(Box::new(a), Box::new(b))
    }

    pub fn le(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Le(Box::new(a), Box::new(b))
    }

    pub fn gt(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Gt(Box::new(a), Box::new(b))
    }

    pub fn ge(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Ge(Box::new(a), Box::new(b))
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

    pub fn ite(cond: SmtExpr, then: SmtExpr, else_: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(cond), Box::new(then), Box::new(else_))
    }

    pub fn implies(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(a), Box::new(b))
    }

    // ---- queries -------------------------------------------------------------

    pub fn is_const(&self) -> bool {
        matches!(
            self,
            SmtExpr::BoolConst(_) | SmtExpr::IntConst(_) | SmtExpr::RealConst(_)
        )
    }

    pub fn is_var(&self) -> bool {
        matches!(self, SmtExpr::Var(_))
    }

    /// Collect all `Var` names that appear in the expression.
    pub fn free_variables(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        self.collect_vars(&mut set);
        set
    }

    fn collect_vars(&self, set: &mut BTreeSet<String>) {
        match self {
            SmtExpr::Var(n) => {
                set.insert(n.clone());
            }
            SmtExpr::BoolConst(_) | SmtExpr::IntConst(_) | SmtExpr::RealConst(_) => {}
            SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Implies(a, b) => {
                a.collect_vars(set);
                b.collect_vars(set);
            }
            SmtExpr::Not(e) => e.collect_vars(set),
            SmtExpr::And(es) | SmtExpr::Or(es) => {
                for e in es {
                    e.collect_vars(set);
                }
            }
            SmtExpr::Ite(c, t, f) => {
                c.collect_vars(set);
                t.collect_vars(set);
                f.collect_vars(set);
            }
        }
    }

    /// Maximum nesting depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            SmtExpr::Var(_)
            | SmtExpr::BoolConst(_)
            | SmtExpr::IntConst(_)
            | SmtExpr::RealConst(_) => 1,
            SmtExpr::Add(a, b)
            | SmtExpr::Sub(a, b)
            | SmtExpr::Mul(a, b)
            | SmtExpr::Div(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Implies(a, b) => 1 + a.depth().max(b.depth()),
            SmtExpr::Not(e) => 1 + e.depth(),
            SmtExpr::And(es) | SmtExpr::Or(es) => {
                1 + es.iter().map(|e| e.depth()).max().unwrap_or(0)
            }
            SmtExpr::Ite(c, t, f) => 1 + c.depth().max(t.depth()).max(f.depth()),
        }
    }

    /// Basic algebraic/boolean simplification (single pass).
    pub fn simplify(&self) -> SmtExpr {
        match self {
            // -- And ----------------------------------------------------------
            SmtExpr::And(es) => {
                let simplified: Vec<SmtExpr> = es
                    .iter()
                    .map(|e| e.simplify())
                    .filter(|e| *e != SmtExpr::BoolConst(true))
                    .collect();
                if simplified.iter().any(|e| *e == SmtExpr::BoolConst(false)) {
                    return SmtExpr::BoolConst(false);
                }
                match simplified.len() {
                    0 => SmtExpr::BoolConst(true),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::And(simplified),
                }
            }

            // -- Or -----------------------------------------------------------
            SmtExpr::Or(es) => {
                let simplified: Vec<SmtExpr> = es
                    .iter()
                    .map(|e| e.simplify())
                    .filter(|e| *e != SmtExpr::BoolConst(false))
                    .collect();
                if simplified.iter().any(|e| *e == SmtExpr::BoolConst(true)) {
                    return SmtExpr::BoolConst(true);
                }
                match simplified.len() {
                    0 => SmtExpr::BoolConst(false),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::Or(simplified),
                }
            }

            // -- Not(Not(x)) -> x ---------------------------------------------
            SmtExpr::Not(inner) => {
                let inner = inner.simplify();
                match inner {
                    SmtExpr::Not(x) => *x,
                    SmtExpr::BoolConst(b) => SmtExpr::BoolConst(!b),
                    other => SmtExpr::Not(Box::new(other)),
                }
            }

            // -- Add identity: x + 0 = 0 + x = x -----------------------------
            SmtExpr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if a == SmtExpr::IntConst(0) {
                    return b;
                }
                if b == SmtExpr::IntConst(0) {
                    return a;
                }
                SmtExpr::Add(Box::new(a), Box::new(b))
            }

            // -- Sub identity: x - 0 = x -------------------------------------
            SmtExpr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if b == SmtExpr::IntConst(0) {
                    return a;
                }
                SmtExpr::Sub(Box::new(a), Box::new(b))
            }

            // -- Mul identities: x * 1 = x, x * 0 = 0 -----------------------
            SmtExpr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if a == SmtExpr::IntConst(1) {
                    return b;
                }
                if b == SmtExpr::IntConst(1) {
                    return a;
                }
                if a == SmtExpr::IntConst(0) || b == SmtExpr::IntConst(0) {
                    return SmtExpr::IntConst(0);
                }
                SmtExpr::Mul(Box::new(a), Box::new(b))
            }

            // -- Leaves and everything else: just clone -----------------------
            other => other.clone(),
        }
    }

    /// Render as an SMT-LIB2 s-expression.
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtExpr::Var(n) => n.clone(),
            SmtExpr::BoolConst(b) => if *b { "true" } else { "false" }.to_string(),
            SmtExpr::IntConst(i) => {
                if *i < 0 {
                    format!("(- {})", -i)
                } else {
                    i.to_string()
                }
            }
            SmtExpr::RealConst(f) => format!("{:.1}", f),

            SmtExpr::Add(a, b) => format!("(+ {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Sub(a, b) => format!("(- {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Mul(a, b) => format!("(* {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Div(a, b) => format!("(div {} {})", a.to_smtlib(), b.to_smtlib()),

            SmtExpr::And(es) => {
                if es.is_empty() {
                    "true".to_string()
                } else {
                    let inner: Vec<String> = es.iter().map(|e| e.to_smtlib()).collect();
                    format!("(and {})", inner.join(" "))
                }
            }
            SmtExpr::Or(es) => {
                if es.is_empty() {
                    "false".to_string()
                } else {
                    let inner: Vec<String> = es.iter().map(|e| e.to_smtlib()).collect();
                    format!("(or {})", inner.join(" "))
                }
            }
            SmtExpr::Not(e) => format!("(not {})", e.to_smtlib()),
            SmtExpr::Implies(a, b) => format!("(=> {} {})", a.to_smtlib(), b.to_smtlib()),

            SmtExpr::Eq(a, b) => format!("(= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Lt(a, b) => format!("(< {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Le(a, b) => format!("(<= {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Gt(a, b) => format!("(> {} {})", a.to_smtlib(), b.to_smtlib()),
            SmtExpr::Ge(a, b) => format!("(>= {} {})", a.to_smtlib(), b.to_smtlib()),

            SmtExpr::Ite(c, t, f) => {
                format!(
                    "(ite {} {} {})",
                    c.to_smtlib(),
                    t.to_smtlib(),
                    f.to_smtlib()
                )
            }
        }
    }
}

impl fmt::Display for SmtExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ---------------------------------------------------------------------------
// SmtConstraint
// ---------------------------------------------------------------------------

/// A named, sourced assertion wrapping an [`SmtExpr`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtConstraint {
    pub expr: SmtExpr,
    pub name: Option<String>,
    pub source: Option<String>,
}

impl SmtConstraint {
    pub fn new(expr: SmtExpr) -> Self {
        Self {
            expr,
            name: None,
            source: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Render as `(assert ...)` or `(assert (! ... :named N))`.
    pub fn to_smtlib(&self) -> String {
        let body = self.expr.to_smtlib();
        match &self.name {
            Some(n) => format!("(assert (! {} :named {}))", body, n),
            None => format!("(assert {})", body),
        }
    }
}

// ---------------------------------------------------------------------------
// SmtFormula
// ---------------------------------------------------------------------------

/// A complete SMT-LIB2 formula (declarations + assertions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtFormula {
    pub constraints: Vec<SmtConstraint>,
    pub declarations: Vec<SmtVariable>,
}

impl SmtFormula {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            declarations: Vec::new(),
        }
    }

    pub fn add_constraint(&mut self, c: SmtConstraint) {
        self.constraints.push(c);
    }

    pub fn add_declaration(&mut self, v: SmtVariable) {
        self.declarations.push(v);
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn variable_count(&self) -> usize {
        self.declarations.len()
    }

    /// Union of free variables across all constraints.
    pub fn free_variables(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        for c in &self.constraints {
            set.extend(c.expr.free_variables());
        }
        set
    }

    /// Full SMT-LIB2 script: set-logic, declarations, asserts, check-sat.
    pub fn to_smtlib(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("(set-logic ALL)".to_string());
        for d in &self.declarations {
            lines.push(d.to_smtlib());
        }
        for c in &self.constraints {
            lines.push(c.to_smtlib());
        }
        lines.push("(check-sat)".to_string());
        lines.join("\n")
    }
}

impl Default for SmtFormula {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SmtValue
// ---------------------------------------------------------------------------

/// A concrete value returned in a satisfying model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Int(i64),
    Real(f64),
    BitVec(Vec<bool>),
}

impl SmtValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SmtValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            SmtValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_real(&self) -> Option<f64> {
        match self {
            SmtValue::Real(f) => Some(*f),
            _ => None,
        }
    }

    pub fn to_smtlib(&self) -> String {
        match self {
            SmtValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            SmtValue::Int(i) => {
                if *i < 0 {
                    format!("(- {})", -i)
                } else {
                    i.to_string()
                }
            }
            SmtValue::Real(f) => format!("{:.1}", f),
            SmtValue::BitVec(bits) => {
                let s: String = bits.iter().map(|b| if *b { '1' } else { '0' }).collect();
                format!("#b{}", s)
            }
        }
    }
}

impl fmt::Display for SmtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

// ---------------------------------------------------------------------------
// SmtModel
// ---------------------------------------------------------------------------

/// A satisfying assignment mapping variable names to values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: IndexMap<String, SmtValue>,
}

impl SmtModel {
    pub fn new() -> Self {
        Self {
            assignments: IndexMap::new(),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: SmtValue) {
        self.assignments.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&SmtValue> {
        self.assignments.get(name)
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.get(name).and_then(SmtValue::as_bool)
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.get(name).and_then(SmtValue::as_int)
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, SmtValue> {
        self.assignments.iter()
    }
}

impl Default for SmtModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SmtModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(model")?;
        for (name, val) in &self.assignments {
            match val {
                SmtValue::BitVec(bits) => {
                    writeln!(
                        f,
                        "  (define-fun {} () (_ BitVec {}) {})",
                        name,
                        bits.len(),
                        val.to_smtlib()
                    )?;
                }
                _ => {
                    let sort_name = match val {
                        SmtValue::Bool(_) => "Bool",
                        SmtValue::Int(_) => "Int",
                        SmtValue::Real(_) => "Real",
                        SmtValue::BitVec(_) => unreachable!(),
                    };
                    writeln!(
                        f,
                        "  (define-fun {} () {} {})",
                        name,
                        sort_name,
                        val.to_smtlib()
                    )?;
                }
            }
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// SmtResult
// ---------------------------------------------------------------------------

/// Outcome of an SMT solver invocation.
#[derive(Debug, Clone)]
pub enum SmtResult {
    Sat(SmtModel),
    Unsat(Vec<String>),
    Unknown(String),
    Timeout(u64),
}

impl SmtResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat(_))
    }

    pub fn model(&self) -> Option<&SmtModel> {
        match self {
            SmtResult::Sat(m) => Some(m),
            _ => None,
        }
    }

    pub fn unsat_core(&self) -> Option<&[String]> {
        match self {
            SmtResult::Unsat(core) => Some(core),
            _ => None,
        }
    }
}

impl fmt::Display for SmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtResult::Sat(m) => write!(f, "sat ({} assignments)", m.len()),
            SmtResult::Unsat(core) => {
                if core.is_empty() {
                    write!(f, "unsat")
                } else {
                    write!(f, "unsat (core: {})", core.join(", "))
                }
            }
            SmtResult::Unknown(reason) => write!(f, "unknown ({})", reason),
            SmtResult::Timeout(ms) => write!(f, "timeout ({}ms)", ms),
        }
    }
}

// ---------------------------------------------------------------------------
// VariableEncoding
// ---------------------------------------------------------------------------

/// Mapping from services to their SMT variables across time-steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableEncoding {
    pub service_vars: IndexMap<ServiceId, Vec<SmtVariable>>,
    pub time_steps: u32,
}

impl VariableEncoding {
    pub fn new(time_steps: u32) -> Self {
        Self {
            service_vars: IndexMap::new(),
            time_steps,
        }
    }

    pub fn add_service_var(&mut self, service: ServiceId, var: SmtVariable) {
        self.service_vars.entry(service).or_default().push(var);
    }

    pub fn vars_for_service(&self, service: &ServiceId) -> &[SmtVariable] {
        self.service_vars
            .get(service)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn all_variables(&self) -> Vec<&SmtVariable> {
        self.service_vars.values().flat_map(|v| v.iter()).collect()
    }

    /// Create a `load_SERVICE_tT` Int variable.
    pub fn load_var(&self, service: &ServiceId, time_step: u32) -> SmtVariable {
        SmtVariable::new(format!("load_{}", service), SmtSort::Int)
            .with_time_step(time_step)
            .with_service(service.clone())
    }

    /// Create a `state_SERVICE_tT` Bool variable.
    pub fn state_var(&self, service: &ServiceId, time_step: u32) -> SmtVariable {
        SmtVariable::new(format!("state_{}", service), SmtSort::Bool)
            .with_time_step(time_step)
            .with_service(service.clone())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- SmtSort -------------------------------------------------------------

    #[test]
    fn sort_smtlib_names() {
        assert_eq!(SmtSort::Bool.smtlib_name(), "Bool");
        assert_eq!(SmtSort::Int.smtlib_name(), "Int");
        assert_eq!(SmtSort::Real.smtlib_name(), "Real");
        assert_eq!(SmtSort::BitVec(32).smtlib_name(), "(_ BitVec 32)");
    }

    #[test]
    fn sort_is_numeric() {
        assert!(SmtSort::Int.is_numeric());
        assert!(SmtSort::Real.is_numeric());
        assert!(!SmtSort::Bool.is_numeric());
        assert!(!SmtSort::BitVec(8).is_numeric());
    }

    // -- SmtVariable ---------------------------------------------------------

    #[test]
    fn variable_qualified_name_no_timestep() {
        let v = SmtVariable::new("x", SmtSort::Int);
        assert_eq!(v.qualified_name(), "x");
    }

    #[test]
    fn variable_qualified_name_with_timestep() {
        let v = SmtVariable::new("load", SmtSort::Int).with_time_step(3);
        assert_eq!(v.qualified_name(), "load_t3");
    }

    #[test]
    fn variable_to_smtlib() {
        let v = SmtVariable::new("ready", SmtSort::Bool).with_time_step(0);
        assert_eq!(v.to_smtlib(), "(declare-const ready_t0 Bool)");
    }

    #[test]
    fn variable_display() {
        let v = SmtVariable::new("x", SmtSort::Real).with_time_step(7);
        assert_eq!(format!("{}", v), "x_t7");
    }

    // -- SmtExpr to_smtlib ---------------------------------------------------

    #[test]
    fn expr_var_and_const() {
        assert_eq!(SmtExpr::var("x").to_smtlib(), "x");
        assert_eq!(SmtExpr::bool_const(true).to_smtlib(), "true");
        assert_eq!(SmtExpr::int_const(42).to_smtlib(), "42");
        assert_eq!(SmtExpr::int_const(-5).to_smtlib(), "(- 5)");
        assert_eq!(SmtExpr::real_const(3.5).to_smtlib(), "3.5");
    }

    #[test]
    fn expr_arithmetic() {
        let e = SmtExpr::add(SmtExpr::var("a"), SmtExpr::int_const(1));
        assert_eq!(e.to_smtlib(), "(+ a 1)");

        let e = SmtExpr::mul(SmtExpr::var("x"), SmtExpr::var("y"));
        assert_eq!(e.to_smtlib(), "(* x y)");
    }

    #[test]
    fn expr_boolean_ops() {
        let e = SmtExpr::and(vec![
            SmtExpr::var("a"),
            SmtExpr::var("b"),
            SmtExpr::var("c"),
        ]);
        assert_eq!(e.to_smtlib(), "(and a b c)");

        let e = SmtExpr::or(vec![SmtExpr::var("x"), SmtExpr::var("y")]);
        assert_eq!(e.to_smtlib(), "(or x y)");

        let e = SmtExpr::not(SmtExpr::var("p"));
        assert_eq!(e.to_smtlib(), "(not p)");
    }

    #[test]
    fn expr_comparison() {
        assert_eq!(
            SmtExpr::eq(SmtExpr::var("a"), SmtExpr::var("b")).to_smtlib(),
            "(= a b)"
        );
        assert_eq!(
            SmtExpr::lt(SmtExpr::var("x"), SmtExpr::int_const(10)).to_smtlib(),
            "(< x 10)"
        );
        assert_eq!(
            SmtExpr::ge(SmtExpr::var("y"), SmtExpr::int_const(0)).to_smtlib(),
            "(>= y 0)"
        );
    }

    #[test]
    fn expr_ite_and_implies() {
        let e = SmtExpr::ite(
            SmtExpr::var("c"),
            SmtExpr::int_const(1),
            SmtExpr::int_const(0),
        );
        assert_eq!(e.to_smtlib(), "(ite c 1 0)");

        let e = SmtExpr::implies(SmtExpr::var("p"), SmtExpr::var("q"));
        assert_eq!(e.to_smtlib(), "(=> p q)");
    }

    // -- SmtExpr free_variables ----------------------------------------------

    #[test]
    fn expr_free_variables() {
        let e = SmtExpr::and(vec![
            SmtExpr::lt(SmtExpr::var("x"), SmtExpr::var("y")),
            SmtExpr::eq(SmtExpr::var("y"), SmtExpr::int_const(5)),
        ]);
        let vars = e.free_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    // -- SmtExpr depth -------------------------------------------------------

    #[test]
    fn expr_depth() {
        assert_eq!(SmtExpr::var("x").depth(), 1);
        assert_eq!(
            SmtExpr::add(SmtExpr::var("a"), SmtExpr::var("b")).depth(),
            2
        );
        // not -> and -> lt -> var/const = depth 4
        let deep = SmtExpr::not(SmtExpr::and(vec![SmtExpr::lt(
            SmtExpr::var("x"),
            SmtExpr::int_const(1),
        )]));
        assert_eq!(deep.depth(), 4);
    }

    // -- SmtExpr simplify ----------------------------------------------------

    #[test]
    fn simplify_and_single() {
        let e = SmtExpr::and(vec![SmtExpr::var("x")]);
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn simplify_or_single() {
        let e = SmtExpr::or(vec![SmtExpr::var("y")]);
        assert_eq!(e.simplify(), SmtExpr::var("y"));
    }

    #[test]
    fn simplify_double_negation() {
        let e = SmtExpr::not(SmtExpr::not(SmtExpr::var("p")));
        assert_eq!(e.simplify(), SmtExpr::var("p"));
    }

    #[test]
    fn simplify_and_with_true() {
        let e = SmtExpr::and(vec![SmtExpr::bool_const(true), SmtExpr::var("x")]);
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn simplify_and_with_false() {
        let e = SmtExpr::and(vec![SmtExpr::var("x"), SmtExpr::bool_const(false)]);
        assert_eq!(e.simplify(), SmtExpr::bool_const(false));
    }

    #[test]
    fn simplify_or_with_true() {
        let e = SmtExpr::or(vec![SmtExpr::var("x"), SmtExpr::bool_const(true)]);
        assert_eq!(e.simplify(), SmtExpr::bool_const(true));
    }

    #[test]
    fn simplify_add_zero() {
        let e = SmtExpr::add(SmtExpr::int_const(0), SmtExpr::var("x"));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn simplify_mul_one() {
        let e = SmtExpr::mul(SmtExpr::var("x"), SmtExpr::int_const(1));
        assert_eq!(e.simplify(), SmtExpr::var("x"));
    }

    #[test]
    fn simplify_mul_zero() {
        let e = SmtExpr::mul(SmtExpr::var("x"), SmtExpr::int_const(0));
        assert_eq!(e.simplify(), SmtExpr::int_const(0));
    }

    // -- SmtConstraint -------------------------------------------------------

    #[test]
    fn constraint_to_smtlib_unnamed() {
        let c = SmtConstraint::new(SmtExpr::var("p"));
        assert_eq!(c.to_smtlib(), "(assert p)");
    }

    #[test]
    fn constraint_to_smtlib_named() {
        let c = SmtConstraint::new(SmtExpr::var("p")).with_name("c1");
        assert_eq!(c.to_smtlib(), "(assert (! p :named c1))");
    }

    // -- SmtFormula ----------------------------------------------------------

    #[test]
    fn formula_to_smtlib() {
        let mut f = SmtFormula::new();
        f.add_declaration(SmtVariable::new("x", SmtSort::Int));
        f.add_constraint(SmtConstraint::new(SmtExpr::gt(
            SmtExpr::var("x"),
            SmtExpr::int_const(0),
        )));
        let script = f.to_smtlib();
        assert!(script.contains("(set-logic ALL)"));
        assert!(script.contains("(declare-const x Int)"));
        assert!(script.contains("(assert (> x 0))"));
        assert!(script.contains("(check-sat)"));
        assert_eq!(f.constraint_count(), 1);
        assert_eq!(f.variable_count(), 1);
    }

    #[test]
    fn formula_free_variables() {
        let mut f = SmtFormula::new();
        f.add_constraint(SmtConstraint::new(SmtExpr::and(vec![
            SmtExpr::var("a"),
            SmtExpr::var("b"),
        ])));
        f.add_constraint(SmtConstraint::new(SmtExpr::var("c")));
        let fv = f.free_variables();
        assert_eq!(fv.len(), 3);
    }

    // -- SmtModel ------------------------------------------------------------

    #[test]
    fn model_get_operations() {
        let mut m = SmtModel::new();
        m.set("x", SmtValue::Int(42));
        m.set("flag", SmtValue::Bool(true));

        assert_eq!(m.get_int("x"), Some(42));
        assert_eq!(m.get_bool("flag"), Some(true));
        assert_eq!(m.get_bool("x"), None);
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn model_display() {
        let mut m = SmtModel::new();
        m.set("a", SmtValue::Bool(false));
        let out = format!("{}", m);
        assert!(out.contains("model"));
        assert!(out.contains("a"));
    }

    // -- SmtValue ------------------------------------------------------------

    #[test]
    fn value_conversions() {
        assert_eq!(SmtValue::Bool(true).as_bool(), Some(true));
        assert_eq!(SmtValue::Int(7).as_int(), Some(7));
        assert_eq!(SmtValue::Real(1.5).as_real(), Some(1.5));
        assert_eq!(SmtValue::Bool(false).as_int(), None);
    }

    #[test]
    fn value_to_smtlib() {
        assert_eq!(SmtValue::Bool(true).to_smtlib(), "true");
        assert_eq!(SmtValue::Int(-3).to_smtlib(), "(- 3)");
        assert_eq!(
            SmtValue::BitVec(vec![true, false, true]).to_smtlib(),
            "#b101"
        );
    }

    // -- SmtResult -----------------------------------------------------------

    #[test]
    fn result_queries() {
        let sat = SmtResult::Sat(SmtModel::new());
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert!(sat.model().is_some());

        let unsat = SmtResult::Unsat(vec!["c1".into(), "c2".into()]);
        assert!(unsat.is_unsat());
        assert_eq!(unsat.unsat_core().unwrap().len(), 2);

        let unk = SmtResult::Unknown("incomplete".into());
        assert!(!unk.is_sat());
        assert!(unk.model().is_none());

        let to = SmtResult::Timeout(5000);
        assert!(!to.is_sat());
        assert_eq!(format!("{}", to), "timeout (5000ms)");
    }

    // -- VariableEncoding ----------------------------------------------------

    #[test]
    fn encoding_load_and_state_vars() {
        let enc = VariableEncoding::new(5);
        let svc = ServiceId::new("gateway");

        let lv = enc.load_var(&svc, 2);
        assert_eq!(lv.qualified_name(), "load_gateway_t2");
        assert_eq!(lv.sort, SmtSort::Int);

        let sv = enc.state_var(&svc, 0);
        assert_eq!(sv.qualified_name(), "state_gateway_t0");
        assert_eq!(sv.sort, SmtSort::Bool);
    }

    #[test]
    fn encoding_add_and_query() {
        let mut enc = VariableEncoding::new(3);
        let svc = ServiceId::new("api");
        let v = SmtVariable::new("latency", SmtSort::Real).with_service(svc.clone());
        enc.add_service_var(svc.clone(), v);

        assert_eq!(enc.vars_for_service(&svc).len(), 1);
        assert_eq!(enc.all_variables().len(), 1);
        assert!(enc.vars_for_service(&ServiceId::new("none")).is_empty());
    }

    // -- Helper constructors -------------------------------------------------

    #[test]
    fn helper_constructors_smoke() {
        let _ = SmtExpr::var("x");
        let _ = SmtExpr::bool_const(false);
        let _ = SmtExpr::int_const(0);
        let _ = SmtExpr::real_const(0.0);
        let _ = SmtExpr::and(vec![]);
        let _ = SmtExpr::or(vec![]);
        let _ = SmtExpr::not(SmtExpr::var("p"));
        let _ = SmtExpr::eq(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::lt(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::le(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::gt(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::ge(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::add(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::sub(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::mul(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::div(SmtExpr::var("a"), SmtExpr::var("b"));
        let _ = SmtExpr::ite(SmtExpr::var("c"), SmtExpr::var("t"), SmtExpr::var("f"));
        let _ = SmtExpr::implies(SmtExpr::var("a"), SmtExpr::var("b"));
    }

    // -- Serialization -------------------------------------------------------

    #[test]
    fn serde_roundtrip_expr() {
        let expr = SmtExpr::and(vec![
            SmtExpr::lt(SmtExpr::var("x"), SmtExpr::int_const(10)),
            SmtExpr::ge(SmtExpr::var("y"), SmtExpr::real_const(0.0)),
        ]);
        let json = serde_json::to_string(&expr).unwrap();
        let back: SmtExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn serde_roundtrip_formula() {
        let mut f = SmtFormula::new();
        f.add_declaration(SmtVariable::new("x", SmtSort::Int));
        f.add_constraint(SmtConstraint::new(SmtExpr::var("x")).with_name("c1"));
        let json = serde_json::to_string(&f).unwrap();
        let back: SmtFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(back.constraint_count(), 1);
        assert_eq!(back.variable_count(), 1);
    }

    #[test]
    fn serde_roundtrip_variable_encoding() {
        let mut enc = VariableEncoding::new(4);
        enc.add_service_var(
            ServiceId::new("web"),
            SmtVariable::new("load", SmtSort::Int).with_time_step(0),
        );
        let json = serde_json::to_string(&enc).unwrap();
        let back: VariableEncoding = serde_json::from_str(&json).unwrap();
        assert_eq!(back.time_steps, 4);
        assert_eq!(back.all_variables().len(), 1);
    }
}
