//! Logical formula representation for specifications.
//!
//! Provides [`Formula`], [`Term`], and [`Predicate`] types for building,
//! simplifying, and printing logical formulas used in contracts and
//! weakest-precondition computations over QF-LIA.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// Relational operator for atomic predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Relation {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Relation {
    /// The negation of this relation.
    pub fn negate(&self) -> Relation {
        match self {
            Relation::Eq => Relation::Ne,
            Relation::Ne => Relation::Eq,
            Relation::Lt => Relation::Ge,
            Relation::Le => Relation::Gt,
            Relation::Gt => Relation::Le,
            Relation::Ge => Relation::Lt,
        }
    }

    /// The "flipped" relation (swap operands).
    pub fn flip(&self) -> Relation {
        match self {
            Relation::Eq => Relation::Eq,
            Relation::Ne => Relation::Ne,
            Relation::Lt => Relation::Gt,
            Relation::Le => Relation::Ge,
            Relation::Gt => Relation::Lt,
            Relation::Ge => Relation::Le,
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            Relation::Eq => "==",
            Relation::Ne => "!=",
            Relation::Lt => "<",
            Relation::Le => "<=",
            Relation::Gt => ">",
            Relation::Ge => ">=",
        }
    }

    pub fn smt_symbol(&self) -> &'static str {
        match self {
            Relation::Eq => "=",
            Relation::Ne => "distinct",
            Relation::Lt => "<",
            Relation::Le => "<=",
            Relation::Gt => ">",
            Relation::Ge => ">=",
        }
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

// ---------------------------------------------------------------------------
// Term
// ---------------------------------------------------------------------------

/// An arithmetic term in a formula.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    /// Integer constant.
    Const(i64),
    /// Variable reference.
    Var(String),
    /// Addition: left + right.
    Add(Box<Term>, Box<Term>),
    /// Subtraction: left - right.
    Sub(Box<Term>, Box<Term>),
    /// Multiplication by a constant: coeff * term.
    Mul(i64, Box<Term>),
    /// Unary negation.
    Neg(Box<Term>),
    /// Array select: array[index].
    ArraySelect(Box<Term>, Box<Term>),
    /// If-then-else: if cond then t else e.
    Ite(Box<Formula>, Box<Term>, Box<Term>),
    /// Old-value reference for postconditions.
    Old(Box<Term>),
    /// Result value reference in postconditions.
    Result,
}

impl Term {
    // -- Constructors -------------------------------------------------------

    pub fn constant(v: i64) -> Self {
        Term::Const(v)
    }

    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    pub fn add(l: Term, r: Term) -> Self {
        Term::Add(Box::new(l), Box::new(r))
    }

    pub fn sub(l: Term, r: Term) -> Self {
        Term::Sub(Box::new(l), Box::new(r))
    }

    pub fn mul(coeff: i64, t: Term) -> Self {
        Term::Mul(coeff, Box::new(t))
    }

    pub fn neg(t: Term) -> Self {
        Term::Neg(Box::new(t))
    }

    pub fn array_select(arr: Term, idx: Term) -> Self {
        Term::ArraySelect(Box::new(arr), Box::new(idx))
    }

    pub fn ite(cond: Formula, t: Term, e: Term) -> Self {
        Term::Ite(Box::new(cond), Box::new(t), Box::new(e))
    }

    pub fn old(t: Term) -> Self {
        Term::Old(Box::new(t))
    }

    // -- Queries ------------------------------------------------------------

    /// Collect all free variable names in this term.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut BTreeSet<String>) {
        match self {
            Term::Const(_) | Term::Result => {}
            Term::Var(name) => {
                vars.insert(name.clone());
            }
            Term::Add(l, r) | Term::Sub(l, r) | Term::ArraySelect(l, r) => {
                l.collect_vars(vars);
                r.collect_vars(vars);
            }
            Term::Mul(_, t) | Term::Neg(t) | Term::Old(t) => {
                t.collect_vars(vars);
            }
            Term::Ite(cond, t, e) => {
                cond.collect_vars(vars);
                t.collect_vars(vars);
                e.collect_vars(vars);
            }
        }
    }

    /// Size of the term (number of nodes).
    pub fn size(&self) -> usize {
        match self {
            Term::Const(_) | Term::Var(_) | Term::Result => 1,
            Term::Add(l, r) | Term::Sub(l, r) | Term::ArraySelect(l, r) => 1 + l.size() + r.size(),
            Term::Mul(_, t) | Term::Neg(t) | Term::Old(t) => 1 + t.size(),
            Term::Ite(c, t, e) => 1 + c.size() + t.size() + e.size(),
        }
    }

    /// Substitute a variable with a term.
    pub fn substitute(&self, var: &str, replacement: &Term) -> Term {
        match self {
            Term::Const(_) | Term::Result => self.clone(),
            Term::Var(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Term::Add(l, r) => Term::add(
                l.substitute(var, replacement),
                r.substitute(var, replacement),
            ),
            Term::Sub(l, r) => Term::sub(
                l.substitute(var, replacement),
                r.substitute(var, replacement),
            ),
            Term::Mul(c, t) => Term::mul(*c, t.substitute(var, replacement)),
            Term::Neg(t) => Term::neg(t.substitute(var, replacement)),
            Term::ArraySelect(a, i) => Term::array_select(
                a.substitute(var, replacement),
                i.substitute(var, replacement),
            ),
            Term::Ite(cond, t, e) => Term::ite(
                cond.substitute(var, replacement),
                t.substitute(var, replacement),
                e.substitute(var, replacement),
            ),
            Term::Old(t) => Term::old(t.substitute(var, replacement)),
        }
    }

    /// Attempt constant-folding on this term.
    pub fn simplify(&self) -> Term {
        match self {
            Term::Add(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Term::Const(a), Term::Const(b)) => Term::Const(a + b),
                    (Term::Const(0), _) => rs,
                    (_, Term::Const(0)) => ls,
                    _ => Term::add(ls, rs),
                }
            }
            Term::Sub(l, r) => {
                let ls = l.simplify();
                let rs = r.simplify();
                match (&ls, &rs) {
                    (Term::Const(a), Term::Const(b)) => Term::Const(a - b),
                    (_, Term::Const(0)) => ls,
                    _ if ls == rs => Term::Const(0),
                    _ => Term::sub(ls, rs),
                }
            }
            Term::Mul(c, t) => {
                let ts = t.simplify();
                match (c, &ts) {
                    (0, _) => Term::Const(0),
                    (1, _) => ts,
                    (-1, _) => Term::neg(ts),
                    (c, Term::Const(v)) => Term::Const(c * v),
                    _ => Term::mul(*c, ts),
                }
            }
            Term::Neg(t) => {
                let ts = t.simplify();
                match &ts {
                    Term::Const(v) => Term::Const(-v),
                    Term::Neg(inner) => *inner.clone(),
                    _ => Term::neg(ts),
                }
            }
            _ => self.clone(),
        }
    }

    /// Is this a constant term?
    pub fn is_const(&self) -> bool {
        matches!(self, Term::Const(_))
    }

    /// Extract constant value if this is a Const.
    pub fn as_const(&self) -> Option<i64> {
        match self {
            Term::Const(v) => Some(*v),
            _ => None,
        }
    }

    /// Is this a simple variable reference?
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Const(v) => write!(f, "{v}"),
            Term::Var(name) => write!(f, "{name}"),
            Term::Add(l, r) => write!(f, "({l} + {r})"),
            Term::Sub(l, r) => write!(f, "({l} - {r})"),
            Term::Mul(c, t) => write!(f, "({c} * {t})"),
            Term::Neg(t) => write!(f, "(-{t})"),
            Term::ArraySelect(a, i) => write!(f, "{a}[{i}]"),
            Term::Ite(c, t, e) => write!(f, "(if {c} then {t} else {e})"),
            Term::Old(t) => write!(f, "\\old({t})"),
            Term::Result => write!(f, "\\result"),
        }
    }
}

// ---------------------------------------------------------------------------
// Predicate
// ---------------------------------------------------------------------------

/// An atomic predicate: `left relation right`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Predicate {
    pub relation: Relation,
    pub left: Term,
    pub right: Term,
}

impl Predicate {
    pub fn new(relation: Relation, left: Term, right: Term) -> Self {
        Self {
            relation,
            left,
            right,
        }
    }

    pub fn eq(left: Term, right: Term) -> Self {
        Self::new(Relation::Eq, left, right)
    }

    pub fn ne(left: Term, right: Term) -> Self {
        Self::new(Relation::Ne, left, right)
    }

    pub fn lt(left: Term, right: Term) -> Self {
        Self::new(Relation::Lt, left, right)
    }

    pub fn le(left: Term, right: Term) -> Self {
        Self::new(Relation::Le, left, right)
    }

    pub fn gt(left: Term, right: Term) -> Self {
        Self::new(Relation::Gt, left, right)
    }

    pub fn ge(left: Term, right: Term) -> Self {
        Self::new(Relation::Ge, left, right)
    }

    /// Negate this predicate (flip relation).
    pub fn negate(&self) -> Predicate {
        Predicate {
            relation: self.relation.negate(),
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }

    /// Flip operands.
    pub fn flip(&self) -> Predicate {
        Predicate {
            relation: self.relation.flip(),
            left: self.right.clone(),
            right: self.left.clone(),
        }
    }

    /// Substitute variable in both sides.
    pub fn substitute(&self, var: &str, replacement: &Term) -> Predicate {
        Predicate {
            relation: self.relation,
            left: self.left.substitute(var, replacement),
            right: self.right.substitute(var, replacement),
        }
    }

    /// Collect free variables.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut vars = self.left.free_vars();
        vars.extend(self.right.free_vars());
        vars
    }

    /// Size.
    pub fn size(&self) -> usize {
        1 + self.left.size() + self.right.size()
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.left, self.relation, self.right)
    }
}

// ---------------------------------------------------------------------------
// Formula
// ---------------------------------------------------------------------------

/// A logical formula in the specification language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Formula {
    /// Constant true.
    True,
    /// Constant false.
    False,
    /// Atomic predicate.
    Atom(Predicate),
    /// Conjunction.
    And(Vec<Formula>),
    /// Disjunction.
    Or(Vec<Formula>),
    /// Negation.
    Not(Box<Formula>),
    /// Implication: antecedent => consequent.
    Implies(Box<Formula>, Box<Formula>),
    /// Bi-implication: left <=> right.
    Iff(Box<Formula>, Box<Formula>),
    /// Universal quantification.
    Forall(Vec<String>, Box<Formula>),
    /// Existential quantification.
    Exists(Vec<String>, Box<Formula>),
}

impl Formula {
    // -- Constructors -------------------------------------------------------

    pub fn atom(pred: Predicate) -> Self {
        Formula::Atom(pred)
    }

    pub fn and(conjuncts: Vec<Formula>) -> Self {
        let filtered: Vec<_> = conjuncts
            .into_iter()
            .filter(|f| *f != Formula::True)
            .collect();
        if filtered.is_empty() {
            Formula::True
        } else if filtered.contains(&Formula::False) {
            Formula::False
        } else if filtered.len() == 1 {
            filtered.into_iter().next().unwrap()
        } else {
            Formula::And(filtered)
        }
    }

    pub fn or(disjuncts: Vec<Formula>) -> Self {
        let filtered: Vec<_> = disjuncts
            .into_iter()
            .filter(|f| *f != Formula::False)
            .collect();
        if filtered.is_empty() {
            Formula::False
        } else if filtered.contains(&Formula::True) {
            Formula::True
        } else if filtered.len() == 1 {
            filtered.into_iter().next().unwrap()
        } else {
            Formula::Or(filtered)
        }
    }

    pub fn not(f: Formula) -> Self {
        match f {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Not(inner) => *inner,
            other => Formula::Not(Box::new(other)),
        }
    }

    pub fn implies(antecedent: Formula, consequent: Formula) -> Self {
        match (&antecedent, &consequent) {
            (Formula::False, _) | (_, Formula::True) => Formula::True,
            (Formula::True, _) => consequent,
            _ => Formula::Implies(Box::new(antecedent), Box::new(consequent)),
        }
    }

    pub fn iff(left: Formula, right: Formula) -> Self {
        if left == right {
            Formula::True
        } else {
            Formula::Iff(Box::new(left), Box::new(right))
        }
    }

    pub fn forall(vars: Vec<String>, body: Formula) -> Self {
        if vars.is_empty() {
            body
        } else {
            Formula::Forall(vars, Box::new(body))
        }
    }

    pub fn exists(vars: Vec<String>, body: Formula) -> Self {
        if vars.is_empty() {
            body
        } else {
            Formula::Exists(vars, Box::new(body))
        }
    }

    // -- Queries ------------------------------------------------------------

    /// Collect all free variables in this formula.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut BTreeSet<String>) {
        match self {
            Formula::True | Formula::False => {}
            Formula::Atom(pred) => {
                vars.extend(pred.free_vars());
            }
            Formula::And(fs) | Formula::Or(fs) => {
                for f in fs {
                    f.collect_vars(vars);
                }
            }
            Formula::Not(f) => f.collect_vars(vars),
            Formula::Implies(a, b) | Formula::Iff(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            Formula::Forall(bound, body) | Formula::Exists(bound, body) => {
                let mut body_vars = BTreeSet::new();
                body.collect_vars(&mut body_vars);
                for v in &body_vars {
                    if !bound.contains(v) {
                        vars.insert(v.clone());
                    }
                }
            }
        }
    }

    /// Size of the formula (number of nodes).
    pub fn size(&self) -> usize {
        match self {
            Formula::True | Formula::False => 1,
            Formula::Atom(p) => p.size(),
            Formula::And(fs) | Formula::Or(fs) => 1 + fs.iter().map(|f| f.size()).sum::<usize>(),
            Formula::Not(f) => 1 + f.size(),
            Formula::Implies(a, b) | Formula::Iff(a, b) => 1 + a.size() + b.size(),
            Formula::Forall(_, body) | Formula::Exists(_, body) => 1 + body.size(),
        }
    }

    /// Depth of the formula tree.
    pub fn depth(&self) -> usize {
        match self {
            Formula::True | Formula::False | Formula::Atom(_) => 1,
            Formula::And(fs) | Formula::Or(fs) => {
                1 + fs.iter().map(|f| f.depth()).max().unwrap_or(0)
            }
            Formula::Not(f) => 1 + f.depth(),
            Formula::Implies(a, b) | Formula::Iff(a, b) => 1 + a.depth().max(b.depth()),
            Formula::Forall(_, body) | Formula::Exists(_, body) => 1 + body.depth(),
        }
    }

    /// Is this a constant (True or False)?
    pub fn is_constant(&self) -> bool {
        matches!(self, Formula::True | Formula::False)
    }

    /// Is this trivially true?
    pub fn is_true(&self) -> bool {
        matches!(self, Formula::True)
    }

    /// Is this trivially false?
    pub fn is_false(&self) -> bool {
        matches!(self, Formula::False)
    }

    // -- Simplification -----------------------------------------------------

    /// Simplify the formula by flattening nested And/Or, eliminating
    /// double negation, and constant folding.
    pub fn simplify(&self) -> Formula {
        match self {
            Formula::True | Formula::False => self.clone(),
            Formula::Atom(p) => Formula::Atom(Predicate {
                relation: p.relation,
                left: p.left.simplify(),
                right: p.right.simplify(),
            }),
            Formula::And(fs) => {
                let mut simplified: Vec<Formula> = Vec::new();
                for f in fs {
                    let s = f.simplify();
                    match s {
                        Formula::True => continue,
                        Formula::False => return Formula::False,
                        Formula::And(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                simplified.dedup();
                Formula::and(simplified)
            }
            Formula::Or(fs) => {
                let mut simplified: Vec<Formula> = Vec::new();
                for f in fs {
                    let s = f.simplify();
                    match s {
                        Formula::False => continue,
                        Formula::True => return Formula::True,
                        Formula::Or(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                simplified.dedup();
                Formula::or(simplified)
            }
            Formula::Not(f) => {
                let s = f.simplify();
                Formula::not(s)
            }
            Formula::Implies(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                Formula::implies(sa, sb)
            }
            Formula::Iff(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                Formula::iff(sa, sb)
            }
            Formula::Forall(vars, body) => {
                let sb = body.simplify();
                Formula::forall(vars.clone(), sb)
            }
            Formula::Exists(vars, body) => {
                let sb = body.simplify();
                Formula::exists(vars.clone(), sb)
            }
        }
    }

    /// Substitute a variable in all contained terms and predicates.
    pub fn substitute(&self, var: &str, replacement: &Term) -> Formula {
        match self {
            Formula::True | Formula::False => self.clone(),
            Formula::Atom(p) => Formula::Atom(p.substitute(var, replacement)),
            Formula::And(fs) => {
                Formula::And(fs.iter().map(|f| f.substitute(var, replacement)).collect())
            }
            Formula::Or(fs) => {
                Formula::Or(fs.iter().map(|f| f.substitute(var, replacement)).collect())
            }
            Formula::Not(f) => Formula::Not(Box::new(f.substitute(var, replacement))),
            Formula::Implies(a, b) => Formula::Implies(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Formula::Iff(a, b) => Formula::Iff(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Formula::Forall(vars, body) => {
                if vars.contains(&var.to_string()) {
                    self.clone()
                } else {
                    Formula::Forall(vars.clone(), Box::new(body.substitute(var, replacement)))
                }
            }
            Formula::Exists(vars, body) => {
                if vars.contains(&var.to_string()) {
                    self.clone()
                } else {
                    Formula::Exists(vars.clone(), Box::new(body.substitute(var, replacement)))
                }
            }
        }
    }

    // -- Conversion helpers -------------------------------------------------

    /// Convert to conjunctive normal form (basic, may blow up).
    pub fn to_cnf(&self) -> Formula {
        let simplified = self.simplify();
        Self::cnf_inner(&simplified)
    }

    fn cnf_inner(formula: &Formula) -> Formula {
        match formula {
            Formula::True | Formula::False | Formula::Atom(_) => formula.clone(),
            Formula::Not(f) => match f.as_ref() {
                Formula::Or(fs) => {
                    let negated: Vec<_> = fs.iter().map(|g| Formula::not(g.clone())).collect();
                    Self::cnf_inner(&Formula::And(negated))
                }
                Formula::And(fs) => {
                    let negated: Vec<_> = fs.iter().map(|g| Formula::not(g.clone())).collect();
                    Self::cnf_inner(&Formula::Or(negated))
                }
                Formula::Not(inner) => Self::cnf_inner(inner),
                _ => formula.clone(),
            },
            Formula::And(fs) => {
                let converted: Vec<_> = fs.iter().map(Self::cnf_inner).collect();
                Formula::And(converted)
            }
            Formula::Or(fs) => {
                let converted: Vec<_> = fs.iter().map(Self::cnf_inner).collect();
                Self::distribute_or(&converted)
            }
            Formula::Implies(a, b) => {
                Self::cnf_inner(&Formula::or(vec![Formula::not(*a.clone()), *b.clone()]))
            }
            Formula::Iff(a, b) => {
                let fwd = Formula::implies(*a.clone(), *b.clone());
                let bwd = Formula::implies(*b.clone(), *a.clone());
                Self::cnf_inner(&Formula::and(vec![fwd, bwd]))
            }
            _ => formula.clone(),
        }
    }

    fn distribute_or(disjuncts: &[Formula]) -> Formula {
        if disjuncts.is_empty() {
            return Formula::False;
        }
        if disjuncts.len() == 1 {
            return disjuncts[0].clone();
        }
        let first = &disjuncts[0];
        let rest = Self::distribute_or(&disjuncts[1..]);
        match (first, &rest) {
            (Formula::And(cs), _) => {
                let distributed: Vec<_> = cs
                    .iter()
                    .map(|c| Self::distribute_or(&[c.clone(), rest.clone()]))
                    .collect();
                Formula::And(distributed)
            }
            (_, Formula::And(cs)) => {
                let distributed: Vec<_> = cs
                    .iter()
                    .map(|c| Self::distribute_or(&[first.clone(), c.clone()]))
                    .collect();
                Formula::And(distributed)
            }
            _ => Formula::Or(vec![first.clone(), rest]),
        }
    }

    /// Convert to disjunctive normal form (basic, may blow up).
    pub fn to_dnf(&self) -> Formula {
        let simplified = self.simplify();
        Self::dnf_inner(&simplified)
    }

    fn dnf_inner(formula: &Formula) -> Formula {
        match formula {
            Formula::True | Formula::False | Formula::Atom(_) => formula.clone(),
            Formula::Not(f) => match f.as_ref() {
                Formula::And(fs) => {
                    let negated: Vec<_> = fs.iter().map(|g| Formula::not(g.clone())).collect();
                    Self::dnf_inner(&Formula::Or(negated))
                }
                Formula::Or(fs) => {
                    let negated: Vec<_> = fs.iter().map(|g| Formula::not(g.clone())).collect();
                    Self::dnf_inner(&Formula::And(negated))
                }
                Formula::Not(inner) => Self::dnf_inner(inner),
                _ => formula.clone(),
            },
            Formula::Or(fs) => {
                let converted: Vec<_> = fs.iter().map(Self::dnf_inner).collect();
                Formula::Or(converted)
            }
            Formula::And(fs) => {
                let converted: Vec<_> = fs.iter().map(Self::dnf_inner).collect();
                Self::distribute_and(&converted)
            }
            Formula::Implies(a, b) => {
                Self::dnf_inner(&Formula::or(vec![Formula::not(*a.clone()), *b.clone()]))
            }
            _ => formula.clone(),
        }
    }

    fn distribute_and(conjuncts: &[Formula]) -> Formula {
        if conjuncts.is_empty() {
            return Formula::True;
        }
        if conjuncts.len() == 1 {
            return conjuncts[0].clone();
        }
        let first = &conjuncts[0];
        let rest = Self::distribute_and(&conjuncts[1..]);
        match (first, &rest) {
            (Formula::Or(ds), _) => {
                let distributed: Vec<_> = ds
                    .iter()
                    .map(|d| Self::distribute_and(&[d.clone(), rest.clone()]))
                    .collect();
                Formula::Or(distributed)
            }
            (_, Formula::Or(ds)) => {
                let distributed: Vec<_> = ds
                    .iter()
                    .map(|d| Self::distribute_and(&[first.clone(), d.clone()]))
                    .collect();
                Formula::Or(distributed)
            }
            _ => Formula::And(vec![first.clone(), rest]),
        }
    }

    /// Structural equivalence check (not semantic).
    pub fn structurally_equal(&self, other: &Formula) -> bool {
        self == other
    }

    /// Count the number of atomic predicates.
    pub fn atom_count(&self) -> usize {
        match self {
            Formula::True | Formula::False => 0,
            Formula::Atom(_) => 1,
            Formula::And(fs) | Formula::Or(fs) => fs.iter().map(|f| f.atom_count()).sum(),
            Formula::Not(f) => f.atom_count(),
            Formula::Implies(a, b) | Formula::Iff(a, b) => a.atom_count() + b.atom_count(),
            Formula::Forall(_, body) | Formula::Exists(_, body) => body.atom_count(),
        }
    }

    /// SMT-LIB2 string representation.
    pub fn to_smt(&self) -> String {
        match self {
            Formula::True => "true".to_string(),
            Formula::False => "false".to_string(),
            Formula::Atom(p) => {
                format!(
                    "({} {} {})",
                    p.relation.smt_symbol(),
                    term_to_smt(&p.left),
                    term_to_smt(&p.right)
                )
            }
            Formula::And(fs) => {
                if fs.is_empty() {
                    "true".to_string()
                } else {
                    let parts: Vec<_> = fs.iter().map(|f| f.to_smt()).collect();
                    format!("(and {})", parts.join(" "))
                }
            }
            Formula::Or(fs) => {
                if fs.is_empty() {
                    "false".to_string()
                } else {
                    let parts: Vec<_> = fs.iter().map(|f| f.to_smt()).collect();
                    format!("(or {})", parts.join(" "))
                }
            }
            Formula::Not(f) => format!("(not {})", f.to_smt()),
            Formula::Implies(a, b) => format!("(=> {} {})", a.to_smt(), b.to_smt()),
            Formula::Iff(a, b) => format!("(= {} {})", a.to_smt(), b.to_smt()),
            Formula::Forall(vars, body) => {
                let bindings: Vec<_> = vars.iter().map(|v| format!("({v} Int)")).collect();
                format!("(forall ({}) {})", bindings.join(" "), body.to_smt())
            }
            Formula::Exists(vars, body) => {
                let bindings: Vec<_> = vars.iter().map(|v| format!("({v} Int)")).collect();
                format!("(exists ({}) {})", bindings.join(" "), body.to_smt())
            }
        }
    }
}

/// Convert a term to SMT-LIB2 format.
fn term_to_smt(t: &Term) -> String {
    match t {
        Term::Const(v) => {
            if *v < 0 {
                format!("(- {})", -v)
            } else {
                v.to_string()
            }
        }
        Term::Var(name) => name.clone(),
        Term::Add(l, r) => format!("(+ {} {})", term_to_smt(l), term_to_smt(r)),
        Term::Sub(l, r) => format!("(- {} {})", term_to_smt(l), term_to_smt(r)),
        Term::Mul(c, t) => format!("(* {} {})", c, term_to_smt(t)),
        Term::Neg(t) => format!("(- {})", term_to_smt(t)),
        Term::ArraySelect(a, i) => format!("(select {} {})", term_to_smt(a), term_to_smt(i)),
        Term::Ite(c, t, e) => format!("(ite {} {} {})", c.to_smt(), term_to_smt(t), term_to_smt(e)),
        Term::Old(t) => format!("old_{}", term_to_smt(t)),
        Term::Result => "result".to_string(),
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "true"),
            Formula::False => write!(f, "false"),
            Formula::Atom(p) => write!(f, "{p}"),
            Formula::And(fs) => {
                write!(f, "(")?;
                for (i, formula) in fs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " && ")?;
                    }
                    write!(f, "{formula}")?;
                }
                write!(f, ")")
            }
            Formula::Or(fs) => {
                write!(f, "(")?;
                for (i, formula) in fs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " || ")?;
                    }
                    write!(f, "{formula}")?;
                }
                write!(f, ")")
            }
            Formula::Not(inner) => write!(f, "!{inner}"),
            Formula::Implies(a, b) => write!(f, "({a} ==> {b})"),
            Formula::Iff(a, b) => write!(f, "({a} <==> {b})"),
            Formula::Forall(vars, body) => {
                write!(f, "(forall {} . {body})", vars.join(", "))
            }
            Formula::Exists(vars, body) => {
                write!(f, "(exists {} . {body})", vars.join(", "))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn var_term(name: &str) -> Term {
        Term::var(name)
    }

    fn x_gt_0() -> Formula {
        Formula::atom(Predicate::gt(var_term("x"), Term::constant(0)))
    }

    fn y_lt_10() -> Formula {
        Formula::atom(Predicate::lt(var_term("y"), Term::constant(10)))
    }

    // -- Relation tests --

    #[test]
    fn test_relation_negate() {
        assert_eq!(Relation::Eq.negate(), Relation::Ne);
        assert_eq!(Relation::Lt.negate(), Relation::Ge);
        assert_eq!(Relation::Gt.negate(), Relation::Le);
    }

    #[test]
    fn test_relation_flip() {
        assert_eq!(Relation::Lt.flip(), Relation::Gt);
        assert_eq!(Relation::Le.flip(), Relation::Ge);
        assert_eq!(Relation::Eq.flip(), Relation::Eq);
    }

    #[test]
    fn test_relation_display() {
        assert_eq!(Relation::Eq.to_string(), "==");
        assert_eq!(Relation::Ne.to_string(), "!=");
        assert_eq!(Relation::Lt.to_string(), "<");
    }

    // -- Term tests --

    #[test]
    fn test_term_const() {
        let t = Term::constant(42);
        assert!(t.is_const());
        assert_eq!(t.as_const(), Some(42));
        assert_eq!(t.size(), 1);
    }

    #[test]
    fn test_term_var() {
        let t = Term::var("x");
        assert!(t.is_var());
        assert_eq!(t.free_vars().len(), 1);
    }

    #[test]
    fn test_term_add() {
        let t = Term::add(Term::var("x"), Term::constant(1));
        assert_eq!(t.size(), 3);
        assert!(t.free_vars().contains("x"));
    }

    #[test]
    fn test_term_display() {
        let t = Term::add(Term::var("x"), Term::constant(1));
        assert_eq!(t.to_string(), "(x + 1)");
    }

    #[test]
    fn test_term_simplify_add_zero() {
        let t = Term::add(Term::var("x"), Term::constant(0));
        assert_eq!(t.simplify(), Term::var("x"));
    }

    #[test]
    fn test_term_simplify_sub_self() {
        let t = Term::sub(Term::var("x"), Term::var("x"));
        assert_eq!(t.simplify(), Term::constant(0));
    }

    #[test]
    fn test_term_simplify_mul_zero() {
        let t = Term::mul(0, Term::var("x"));
        assert_eq!(t.simplify(), Term::constant(0));
    }

    #[test]
    fn test_term_simplify_mul_one() {
        let t = Term::mul(1, Term::var("x"));
        assert_eq!(t.simplify(), Term::var("x"));
    }

    #[test]
    fn test_term_simplify_double_neg() {
        let t = Term::neg(Term::neg(Term::var("x")));
        assert_eq!(t.simplify(), Term::var("x"));
    }

    #[test]
    fn test_term_simplify_const_fold() {
        let t = Term::add(Term::constant(3), Term::constant(4));
        assert_eq!(t.simplify(), Term::constant(7));
    }

    #[test]
    fn test_term_substitute() {
        let t = Term::add(Term::var("x"), Term::constant(1));
        let result = t.substitute("x", &Term::constant(5));
        assert_eq!(result.simplify(), Term::constant(6));
    }

    #[test]
    fn test_term_result_display() {
        assert_eq!(Term::Result.to_string(), "\\result");
    }

    #[test]
    fn test_term_old_display() {
        let t = Term::old(Term::var("x"));
        assert_eq!(t.to_string(), "\\old(x)");
    }

    #[test]
    fn test_term_array_select_display() {
        let t = Term::array_select(Term::var("a"), Term::constant(0));
        assert_eq!(t.to_string(), "a[0]");
    }

    #[test]
    fn test_term_ite() {
        let t = Term::ite(Formula::True, Term::constant(1), Term::constant(0));
        assert!(t.size() > 1);
    }

    // -- Predicate tests --

    #[test]
    fn test_predicate_basic() {
        let p = Predicate::gt(Term::var("x"), Term::constant(0));
        assert_eq!(p.relation, Relation::Gt);
        assert_eq!(p.to_string(), "x > 0");
    }

    #[test]
    fn test_predicate_negate() {
        let p = Predicate::gt(Term::var("x"), Term::constant(0));
        let neg = p.negate();
        assert_eq!(neg.relation, Relation::Le);
    }

    #[test]
    fn test_predicate_flip() {
        let p = Predicate::lt(Term::var("x"), Term::var("y"));
        let flipped = p.flip();
        assert_eq!(flipped.relation, Relation::Gt);
        assert_eq!(flipped.left, Term::var("y"));
    }

    #[test]
    fn test_predicate_substitute() {
        let p = Predicate::eq(Term::var("x"), Term::constant(5));
        let p2 = p.substitute("x", &Term::var("y"));
        assert_eq!(p2.left, Term::var("y"));
    }

    #[test]
    fn test_predicate_free_vars() {
        let p = Predicate::lt(Term::var("x"), Term::var("y"));
        let vars = p.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    // -- Formula tests --

    #[test]
    fn test_formula_true_false() {
        assert!(Formula::True.is_true());
        assert!(Formula::False.is_false());
        assert!(Formula::True.is_constant());
    }

    #[test]
    fn test_formula_and_simplification() {
        assert_eq!(Formula::and(vec![Formula::True, x_gt_0()]), x_gt_0());
        assert_eq!(Formula::and(vec![Formula::False, x_gt_0()]), Formula::False);
        assert_eq!(
            Formula::and(vec![Formula::True, Formula::True]),
            Formula::True
        );
    }

    #[test]
    fn test_formula_or_simplification() {
        assert_eq!(Formula::or(vec![Formula::False, x_gt_0()]), x_gt_0());
        assert_eq!(Formula::or(vec![Formula::True, x_gt_0()]), Formula::True);
        assert_eq!(
            Formula::or(vec![Formula::False, Formula::False]),
            Formula::False
        );
    }

    #[test]
    fn test_formula_not_double_negation() {
        assert_eq!(Formula::not(Formula::not(x_gt_0())), x_gt_0());
        assert_eq!(Formula::not(Formula::True), Formula::False);
        assert_eq!(Formula::not(Formula::False), Formula::True);
    }

    #[test]
    fn test_formula_implies() {
        assert_eq!(Formula::implies(Formula::False, x_gt_0()), Formula::True);
        assert_eq!(Formula::implies(x_gt_0(), Formula::True), Formula::True);
        assert_eq!(Formula::implies(Formula::True, x_gt_0()), x_gt_0());
    }

    #[test]
    fn test_formula_iff_same() {
        assert_eq!(Formula::iff(x_gt_0(), x_gt_0()), Formula::True);
    }

    #[test]
    fn test_formula_size() {
        assert_eq!(Formula::True.size(), 1);
        assert!(x_gt_0().size() > 1);
        let conjunction = Formula::and(vec![x_gt_0(), y_lt_10()]);
        assert!(conjunction.size() > x_gt_0().size());
    }

    #[test]
    fn test_formula_depth() {
        assert_eq!(Formula::True.depth(), 1);
        let nested = Formula::and(vec![Formula::or(vec![x_gt_0(), y_lt_10()]), Formula::True]);
        assert!(nested.depth() >= 2);
    }

    #[test]
    fn test_formula_free_vars() {
        let f = Formula::and(vec![x_gt_0(), y_lt_10()]);
        let vars = f.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_formula_forall_bound_vars() {
        let f = Formula::forall(vec!["x".into()], x_gt_0());
        let vars = f.free_vars();
        assert!(!vars.contains("x"));
    }

    #[test]
    fn test_formula_substitute() {
        let f = x_gt_0();
        let result = f.substitute("x", &Term::constant(5));
        // After substitution, should be 5 > 0.
        match &result {
            Formula::Atom(p) => {
                assert_eq!(p.left, Term::constant(5));
            }
            _ => panic!("expected atom"),
        }
    }

    #[test]
    fn test_formula_substitute_bound() {
        let f = Formula::forall(vec!["x".into()], x_gt_0());
        let result = f.substitute("x", &Term::constant(5));
        // Should not substitute bound variable.
        assert_eq!(result, f);
    }

    #[test]
    fn test_formula_simplify_nested_and() {
        let inner = Formula::And(vec![x_gt_0(), y_lt_10()]);
        let outer = Formula::And(vec![inner, Formula::True]);
        let simplified = outer.simplify();
        // Should flatten the nested And and remove True.
        match &simplified {
            Formula::And(fs) => assert_eq!(fs.len(), 2),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn test_formula_simplify_nested_or() {
        let inner = Formula::Or(vec![x_gt_0(), y_lt_10()]);
        let outer = Formula::Or(vec![inner, Formula::False]);
        let simplified = outer.simplify();
        match &simplified {
            Formula::Or(fs) => assert_eq!(fs.len(), 2),
            _ => panic!("expected Or"),
        }
    }

    #[test]
    fn test_formula_atom_count() {
        let f = Formula::and(vec![x_gt_0(), y_lt_10()]);
        assert_eq!(f.atom_count(), 2);
    }

    #[test]
    fn test_formula_display() {
        assert_eq!(Formula::True.to_string(), "true");
        assert_eq!(Formula::False.to_string(), "false");
        let f = x_gt_0();
        let s = f.to_string();
        assert!(s.contains("x") && s.contains(">") && s.contains("0"));
    }

    #[test]
    fn test_formula_and_display() {
        let f = Formula::And(vec![x_gt_0(), y_lt_10()]);
        let s = f.to_string();
        assert!(s.contains("&&"));
    }

    #[test]
    fn test_formula_or_display() {
        let f = Formula::Or(vec![x_gt_0(), y_lt_10()]);
        let s = f.to_string();
        assert!(s.contains("||"));
    }

    #[test]
    fn test_formula_not_display() {
        let f = Formula::Not(Box::new(x_gt_0()));
        let s = f.to_string();
        assert!(s.starts_with('!'));
    }

    #[test]
    fn test_formula_to_smt() {
        let f = x_gt_0();
        let smt = f.to_smt();
        assert!(smt.contains(">"));
        assert!(smt.contains("x"));
    }

    #[test]
    fn test_formula_to_smt_and() {
        let f = Formula::And(vec![x_gt_0(), y_lt_10()]);
        let smt = f.to_smt();
        assert!(smt.starts_with("(and"));
    }

    #[test]
    fn test_formula_to_smt_not() {
        let f = Formula::Not(Box::new(Formula::True));
        let smt = f.to_smt();
        assert!(smt.contains("not"));
    }

    #[test]
    fn test_formula_to_smt_implies() {
        let f = Formula::Implies(Box::new(x_gt_0()), Box::new(y_lt_10()));
        let smt = f.to_smt();
        assert!(smt.contains("=>"));
    }

    #[test]
    fn test_formula_to_smt_forall() {
        let f = Formula::Forall(vec!["x".into()], Box::new(x_gt_0()));
        let smt = f.to_smt();
        assert!(smt.contains("forall"));
    }

    #[test]
    fn test_formula_to_smt_negative_const() {
        let f = Formula::atom(Predicate::gt(Term::var("x"), Term::constant(-5)));
        let smt = f.to_smt();
        assert!(smt.contains("(- 5)"));
    }

    #[test]
    fn test_formula_cnf_basic() {
        let f = Formula::or(vec![Formula::and(vec![x_gt_0(), y_lt_10()]), x_gt_0()]);
        let cnf = f.to_cnf();
        // CNF should be an And of Ors (or simpler).
        fn is_cnf(f: &Formula) -> bool {
            match f {
                Formula::And(fs) => fs.iter().all(|g| is_clause(g)),
                other => is_clause(other),
            }
        }
        fn is_clause(f: &Formula) -> bool {
            match f {
                Formula::Or(fs) => fs.iter().all(|g| is_literal(g)),
                other => is_literal(other),
            }
        }
        fn is_literal(f: &Formula) -> bool {
            match f {
                Formula::Atom(_) | Formula::True | Formula::False => true,
                Formula::Not(inner) => matches!(inner.as_ref(), Formula::Atom(_)),
                _ => false,
            }
        }
        assert!(is_cnf(&cnf));
    }

    #[test]
    fn test_formula_dnf_basic() {
        let f = Formula::and(vec![Formula::or(vec![x_gt_0(), y_lt_10()]), x_gt_0()]);
        let _dnf = f.to_dnf();
        // Just ensure it doesn't panic.
    }

    #[test]
    fn test_formula_structural_equality() {
        let a = x_gt_0();
        let b = x_gt_0();
        assert!(a.structurally_equal(&b));
        assert!(!a.structurally_equal(&y_lt_10()));
    }

    #[test]
    fn test_formula_exists_empty_vars() {
        let f = Formula::exists(vec![], x_gt_0());
        assert_eq!(f, x_gt_0());
    }

    #[test]
    fn test_formula_forall_empty_vars() {
        let f = Formula::forall(vec![], x_gt_0());
        assert_eq!(f, x_gt_0());
    }

    #[test]
    fn test_formula_serialization() {
        let f = Formula::and(vec![x_gt_0(), y_lt_10()]);
        let json = serde_json::to_string(&f).unwrap();
        let f2: Formula = serde_json::from_str(&json).unwrap();
        assert_eq!(f, f2);
    }

    #[test]
    fn test_term_serialization() {
        let t = Term::add(Term::var("x"), Term::constant(1));
        let json = serde_json::to_string(&t).unwrap();
        let t2: Term = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
