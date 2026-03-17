use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstraintId(pub String);

impl ConstraintId {
    pub fn new(id: impl Into<String>) -> Self { ConstraintId(id.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConstraintKind {
    Hard,
    Soft { weight: f64 },
}

impl ConstraintKind {
    pub fn is_hard(&self) -> bool { matches!(self, ConstraintKind::Hard) }
    pub fn weight(&self) -> f64 {
        match self {
            ConstraintKind::Hard => f64::INFINITY,
            ConstraintKind::Soft { weight } => *weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarId(pub String);

impl VarId {
    pub fn new(name: impl Into<String>) -> Self { VarId(name.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
    pub fn timestamped(&self, t: usize) -> Self { VarId(format!("{}@{}", self.0, t)) }
}

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "="),
            CompareOp::Ne => write!(f, "≠"),
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "≤"),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, "≥"),
        }
    }
}

impl CompareOp {
    pub fn evaluate_f64(&self, a: f64, b: f64) -> bool {
        match self {
            CompareOp::Eq => (a - b).abs() < 1e-10,
            CompareOp::Ne => (a - b).abs() >= 1e-10,
            CompareOp::Lt => a < b,
            CompareOp::Le => a <= b,
            CompareOp::Gt => a > b,
            CompareOp::Ge => a >= b,
        }
    }

    pub fn negate(&self) -> Self {
        match self {
            CompareOp::Eq => CompareOp::Ne,
            CompareOp::Ne => CompareOp::Eq,
            CompareOp::Lt => CompareOp::Ge,
            CompareOp::Le => CompareOp::Gt,
            CompareOp::Gt => CompareOp::Le,
            CompareOp::Ge => CompareOp::Lt,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintExpr {
    BoolConst(bool),
    Var(VarId),
    Not(Box<ConstraintExpr>),
    And(Vec<ConstraintExpr>),
    Or(Vec<ConstraintExpr>),
    Implies(Box<ConstraintExpr>, Box<ConstraintExpr>),
    Iff(Box<ConstraintExpr>, Box<ConstraintExpr>),
    Compare(CompareOp, Box<ArithExpr>, Box<ArithExpr>),
    Ite(Box<ConstraintExpr>, Box<ConstraintExpr>, Box<ConstraintExpr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArithExpr {
    Const(f64),
    Var(VarId),
    Add(Box<ArithExpr>, Box<ArithExpr>),
    Sub(Box<ArithExpr>, Box<ArithExpr>),
    Mul(Box<ArithExpr>, Box<ArithExpr>),
    Neg(Box<ArithExpr>),
}

impl ArithExpr {
    pub fn constant(v: f64) -> Self { ArithExpr::Const(v) }
    pub fn var(name: &str) -> Self { ArithExpr::Var(VarId::new(name)) }

    pub fn evaluate(&self, env: &HashMap<String, f64>) -> Option<f64> {
        match self {
            ArithExpr::Const(v) => Some(*v),
            ArithExpr::Var(id) => env.get(&id.0).copied(),
            ArithExpr::Add(l, r) => {
                let lv = l.evaluate(env)?;
                let rv = r.evaluate(env)?;
                Some(lv + rv)
            }
            ArithExpr::Sub(l, r) => {
                let lv = l.evaluate(env)?;
                let rv = r.evaluate(env)?;
                Some(lv - rv)
            }
            ArithExpr::Mul(l, r) => {
                let lv = l.evaluate(env)?;
                let rv = r.evaluate(env)?;
                Some(lv * rv)
            }
            ArithExpr::Neg(e) => e.evaluate(env).map(|v| -v),
        }
    }

    pub fn free_variables(&self) -> HashSet<VarId> {
        match self {
            ArithExpr::Const(_) => HashSet::new(),
            ArithExpr::Var(id) => { let mut s = HashSet::new(); s.insert(id.clone()); s }
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) | ArithExpr::Mul(l, r) => {
                let mut s = l.free_variables();
                s.extend(r.free_variables());
                s
            }
            ArithExpr::Neg(e) => e.free_variables(),
        }
    }
}

impl ConstraintExpr {
    pub fn bool_const(v: bool) -> Self { ConstraintExpr::BoolConst(v) }
    pub fn var(name: &str) -> Self { ConstraintExpr::Var(VarId::new(name)) }

    pub fn not(e: ConstraintExpr) -> Self { ConstraintExpr::Not(Box::new(e)) }
    pub fn and(exprs: Vec<ConstraintExpr>) -> Self { ConstraintExpr::And(exprs) }
    pub fn or(exprs: Vec<ConstraintExpr>) -> Self { ConstraintExpr::Or(exprs) }
    pub fn implies(a: ConstraintExpr, b: ConstraintExpr) -> Self {
        ConstraintExpr::Implies(Box::new(a), Box::new(b))
    }

    pub fn evaluate(&self, bool_env: &HashMap<String, bool>, arith_env: &HashMap<String, f64>) -> Option<bool> {
        match self {
            ConstraintExpr::BoolConst(v) => Some(*v),
            ConstraintExpr::Var(id) => bool_env.get(&id.0).copied(),
            ConstraintExpr::Not(e) => e.evaluate(bool_env, arith_env).map(|v| !v),
            ConstraintExpr::And(es) => {
                let mut result = true;
                for e in es {
                    result = result && e.evaluate(bool_env, arith_env)?;
                }
                Some(result)
            }
            ConstraintExpr::Or(es) => {
                let mut result = false;
                for e in es {
                    result = result || e.evaluate(bool_env, arith_env)?;
                }
                Some(result)
            }
            ConstraintExpr::Implies(a, b) => {
                let av = a.evaluate(bool_env, arith_env)?;
                let bv = b.evaluate(bool_env, arith_env)?;
                Some(!av || bv)
            }
            ConstraintExpr::Iff(a, b) => {
                let av = a.evaluate(bool_env, arith_env)?;
                let bv = b.evaluate(bool_env, arith_env)?;
                Some(av == bv)
            }
            ConstraintExpr::Compare(op, l, r) => {
                let lv = l.evaluate(arith_env)?;
                let rv = r.evaluate(arith_env)?;
                Some(op.evaluate_f64(lv, rv))
            }
            ConstraintExpr::Ite(cond, then_br, else_br) => {
                if cond.evaluate(bool_env, arith_env)? {
                    then_br.evaluate(bool_env, arith_env)
                } else {
                    else_br.evaluate(bool_env, arith_env)
                }
            }
        }
    }

    pub fn free_variables(&self) -> HashSet<VarId> {
        match self {
            ConstraintExpr::BoolConst(_) => HashSet::new(),
            ConstraintExpr::Var(id) => { let mut s = HashSet::new(); s.insert(id.clone()); s }
            ConstraintExpr::Not(e) => e.free_variables(),
            ConstraintExpr::And(es) | ConstraintExpr::Or(es) => {
                es.iter().flat_map(|e| e.free_variables()).collect()
            }
            ConstraintExpr::Implies(a, b) | ConstraintExpr::Iff(a, b) => {
                let mut s = a.free_variables();
                s.extend(b.free_variables());
                s
            }
            ConstraintExpr::Compare(_, l, r) => {
                let mut s = l.free_variables();
                s.extend(r.free_variables());
                s
            }
            ConstraintExpr::Ite(c, t, e) => {
                let mut s = c.free_variables();
                s.extend(t.free_variables());
                s.extend(e.free_variables());
                s
            }
        }
    }

    pub fn simplify(&self) -> ConstraintExpr {
        match self {
            ConstraintExpr::Not(inner) => {
                let s = inner.simplify();
                match s {
                    ConstraintExpr::BoolConst(v) => ConstraintExpr::BoolConst(!v),
                    ConstraintExpr::Not(e) => *e,
                    other => ConstraintExpr::Not(Box::new(other)),
                }
            }
            ConstraintExpr::And(es) => {
                let simplified: Vec<_> = es.iter().map(|e| e.simplify()).collect();
                if simplified.iter().any(|e| matches!(e, ConstraintExpr::BoolConst(false))) {
                    return ConstraintExpr::BoolConst(false);
                }
                let filtered: Vec<_> = simplified.into_iter()
                    .filter(|e| !matches!(e, ConstraintExpr::BoolConst(true)))
                    .collect();
                if filtered.is_empty() { ConstraintExpr::BoolConst(true) }
                else if filtered.len() == 1 { filtered.into_iter().next().unwrap() }
                else { ConstraintExpr::And(filtered) }
            }
            ConstraintExpr::Or(es) => {
                let simplified: Vec<_> = es.iter().map(|e| e.simplify()).collect();
                if simplified.iter().any(|e| matches!(e, ConstraintExpr::BoolConst(true))) {
                    return ConstraintExpr::BoolConst(true);
                }
                let filtered: Vec<_> = simplified.into_iter()
                    .filter(|e| !matches!(e, ConstraintExpr::BoolConst(false)))
                    .collect();
                if filtered.is_empty() { ConstraintExpr::BoolConst(false) }
                else if filtered.len() == 1 { filtered.into_iter().next().unwrap() }
                else { ConstraintExpr::Or(filtered) }
            }
            ConstraintExpr::Implies(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (ConstraintExpr::BoolConst(false), _) => ConstraintExpr::BoolConst(true),
                    (_, ConstraintExpr::BoolConst(true)) => ConstraintExpr::BoolConst(true),
                    (ConstraintExpr::BoolConst(true), _) => sb,
                    _ => ConstraintExpr::Implies(Box::new(sa), Box::new(sb)),
                }
            }
            other => other.clone(),
        }
    }

    pub fn to_cnf(&self) -> Vec<Clause> {
        let simplified = self.simplify();
        let mut clauses = Vec::new();
        Self::collect_cnf(&simplified, &mut clauses);
        clauses
    }

    fn collect_cnf(expr: &ConstraintExpr, clauses: &mut Vec<Clause>) {
        match expr {
            ConstraintExpr::And(es) => {
                for e in es { Self::collect_cnf(e, clauses); }
            }
            ConstraintExpr::Or(es) => {
                let mut lits = Vec::new();
                for e in es {
                    match e {
                        ConstraintExpr::Var(id) => lits.push(Literal::Pos(id.clone())),
                        ConstraintExpr::Not(inner) => {
                            if let ConstraintExpr::Var(id) = inner.as_ref() {
                                lits.push(Literal::Neg(id.clone()));
                            } else {
                                lits.push(Literal::Pos(VarId::new(format!("aux_{}", clauses.len()))));
                            }
                        }
                        _ => {
                            lits.push(Literal::Pos(VarId::new(format!("aux_{}", clauses.len()))));
                        }
                    }
                }
                clauses.push(Clause { literals: lits });
            }
            ConstraintExpr::Var(id) => {
                clauses.push(Clause { literals: vec![Literal::Pos(id.clone())] });
            }
            ConstraintExpr::Not(inner) => {
                if let ConstraintExpr::Var(id) = inner.as_ref() {
                    clauses.push(Clause { literals: vec![Literal::Neg(id.clone())] });
                }
            }
            ConstraintExpr::BoolConst(true) => {}
            ConstraintExpr::BoolConst(false) => {
                clauses.push(Clause { literals: Vec::new() });
            }
            _ => {
                clauses.push(Clause { literals: vec![Literal::Pos(VarId::new(format!("tseitin_{}", clauses.len())))] });
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            ConstraintExpr::BoolConst(_) | ConstraintExpr::Var(_) => 1,
            ConstraintExpr::Not(e) => 1 + e.size(),
            ConstraintExpr::And(es) | ConstraintExpr::Or(es) => 1 + es.iter().map(|e| e.size()).sum::<usize>(),
            ConstraintExpr::Implies(a, b) | ConstraintExpr::Iff(a, b) => 1 + a.size() + b.size(),
            ConstraintExpr::Compare(_, l, r) => 1 + l.size_inner() + r.size_inner(),
            ConstraintExpr::Ite(c, t, e) => 1 + c.size() + t.size() + e.size(),
        }
    }
}

impl ArithExpr {
    fn size_inner(&self) -> usize {
        match self {
            ArithExpr::Const(_) | ArithExpr::Var(_) => 1,
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) | ArithExpr::Mul(l, r) => 1 + l.size_inner() + r.size_inner(),
            ArithExpr::Neg(e) => 1 + e.size_inner(),
        }
    }
}

impl fmt::Display for ConstraintExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintExpr::BoolConst(v) => write!(f, "{}", v),
            ConstraintExpr::Var(id) => write!(f, "{}", id),
            ConstraintExpr::Not(e) => write!(f, "¬({})", e),
            ConstraintExpr::And(es) => {
                let parts: Vec<String> = es.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            ConstraintExpr::Or(es) => {
                let parts: Vec<String> = es.iter().map(|e| format!("{}", e)).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            ConstraintExpr::Implies(a, b) => write!(f, "({} → {})", a, b),
            ConstraintExpr::Iff(a, b) => write!(f, "({} ↔ {})", a, b),
            ConstraintExpr::Compare(op, l, r) => write!(f, "({} {} {})", l, op, r),
            ConstraintExpr::Ite(c, t, e) => write!(f, "(if {} then {} else {})", c, t, e),
        }
    }
}

impl fmt::Display for ArithExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithExpr::Const(v) => write!(f, "{}", v),
            ArithExpr::Var(id) => write!(f, "{}", id),
            ArithExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            ArithExpr::Sub(l, r) => write!(f, "({} - {})", l, r),
            ArithExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            ArithExpr::Neg(e) => write!(f, "(-{})", e),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Pos(VarId),
    Neg(VarId),
}

impl Literal {
    pub fn var_id(&self) -> &VarId {
        match self { Literal::Pos(id) | Literal::Neg(id) => id }
    }
    pub fn is_positive(&self) -> bool { matches!(self, Literal::Pos(_)) }
    pub fn negate(&self) -> Literal {
        match self {
            Literal::Pos(id) => Literal::Neg(id.clone()),
            Literal::Neg(id) => Literal::Pos(id.clone()),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Pos(id) => write!(f, "{}", id),
            Literal::Neg(id) => write!(f, "¬{}", id),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self { Clause { literals } }
    pub fn unit(lit: Literal) -> Self { Clause { literals: vec![lit] } }
    pub fn is_empty(&self) -> bool { self.literals.is_empty() }
    pub fn is_unit(&self) -> bool { self.literals.len() == 1 }
    pub fn len(&self) -> usize { self.literals.len() }

    pub fn variables(&self) -> HashSet<VarId> {
        self.literals.iter().map(|l| l.var_id().clone()).collect()
    }

    pub fn resolve(&self, other: &Clause, pivot: &VarId) -> Option<Clause> {
        let self_has_pos = self.literals.iter().any(|l| l.var_id() == pivot && l.is_positive());
        let self_has_neg = self.literals.iter().any(|l| l.var_id() == pivot && !l.is_positive());
        let other_has_pos = other.literals.iter().any(|l| l.var_id() == pivot && l.is_positive());
        let other_has_neg = other.literals.iter().any(|l| l.var_id() == pivot && !l.is_positive());

        if (self_has_pos && other_has_neg) || (self_has_neg && other_has_pos) {
            let mut new_lits: Vec<Literal> = Vec::new();
            for l in &self.literals {
                if l.var_id() != pivot { new_lits.push(l.clone()); }
            }
            for l in &other.literals {
                if l.var_id() != pivot { new_lits.push(l.clone()); }
            }
            Some(Clause::new(new_lits))
        } else {
            None
        }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.literals.is_empty() { return write!(f, "⊥"); }
        let parts: Vec<String> = self.literals.iter().map(|l| format!("{}", l)).collect();
        write!(f, "({})", parts.join(" ∨ "))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub id: ConstraintId,
    pub kind: ConstraintKind,
    pub expr: ConstraintExpr,
    pub description: String,
    pub source_obligation: Option<String>,
    pub source_jurisdiction: Option<String>,
}

impl Constraint {
    pub fn hard(id: &str, expr: ConstraintExpr) -> Self {
        Constraint {
            id: ConstraintId::new(id), kind: ConstraintKind::Hard,
            expr, description: String::new(),
            source_obligation: None, source_jurisdiction: None,
        }
    }

    pub fn soft(id: &str, expr: ConstraintExpr, weight: f64) -> Self {
        Constraint {
            id: ConstraintId::new(id), kind: ConstraintKind::Soft { weight },
            expr, description: String::new(),
            source_obligation: None, source_jurisdiction: None,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_source(mut self, obl: &str, jur: &str) -> Self {
        self.source_obligation = Some(obl.to_string());
        self.source_jurisdiction = Some(jur.to_string());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
    hard_indices: Vec<usize>,
    soft_indices: Vec<usize>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        ConstraintSet { constraints: Vec::new(), hard_indices: Vec::new(), soft_indices: Vec::new() }
    }

    pub fn add(&mut self, constraint: Constraint) {
        let idx = self.constraints.len();
        if constraint.kind.is_hard() { self.hard_indices.push(idx); }
        else { self.soft_indices.push(idx); }
        self.constraints.push(constraint);
    }

    pub fn hard_constraints(&self) -> Vec<&Constraint> {
        self.hard_indices.iter().map(|&i| &self.constraints[i]).collect()
    }

    pub fn soft_constraints(&self) -> Vec<&Constraint> {
        self.soft_indices.iter().map(|&i| &self.constraints[i]).collect()
    }

    pub fn all(&self) -> &[Constraint] { &self.constraints }
    pub fn len(&self) -> usize { self.constraints.len() }
    pub fn is_empty(&self) -> bool { self.constraints.is_empty() }
    pub fn hard_count(&self) -> usize { self.hard_indices.len() }
    pub fn soft_count(&self) -> usize { self.soft_indices.len() }

    pub fn total_soft_weight(&self) -> f64 {
        self.soft_constraints().iter().map(|c| c.kind.weight()).sum()
    }

    pub fn variables(&self) -> HashSet<VarId> {
        self.constraints.iter().flat_map(|c| c.expr.free_variables()).collect()
    }

    pub fn merge(&self, other: &ConstraintSet) -> ConstraintSet {
        let mut merged = self.clone();
        for c in &other.constraints { merged.add(c.clone()); }
        merged
    }
}

impl Default for ConstraintSet {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_expr_eval() {
        let expr = ConstraintExpr::and(vec![
            ConstraintExpr::var("x"),
            ConstraintExpr::var("y"),
        ]);
        let mut env = HashMap::new();
        env.insert("x".to_string(), true);
        env.insert("y".to_string(), true);
        assert_eq!(expr.evaluate(&env, &HashMap::new()), Some(true));
        env.insert("y".to_string(), false);
        assert_eq!(expr.evaluate(&env, &HashMap::new()), Some(false));
    }

    #[test]
    fn test_simplify() {
        let expr = ConstraintExpr::and(vec![
            ConstraintExpr::BoolConst(true),
            ConstraintExpr::var("x"),
        ]);
        let simplified = expr.simplify();
        assert!(matches!(simplified, ConstraintExpr::Var(_)));
    }

    #[test]
    fn test_clause_resolve() {
        let c1 = Clause::new(vec![Literal::Pos(VarId::new("x")), Literal::Pos(VarId::new("y"))]);
        let c2 = Clause::new(vec![Literal::Neg(VarId::new("x")), Literal::Pos(VarId::new("z"))]);
        let resolved = c1.resolve(&c2, &VarId::new("x"));
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().len(), 2);
    }

    #[test]
    fn test_constraint_set() {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::hard("h1", ConstraintExpr::var("x")));
        cs.add(Constraint::soft("s1", ConstraintExpr::var("y"), 0.5));
        assert_eq!(cs.hard_count(), 1);
        assert_eq!(cs.soft_count(), 1);
    }
}
