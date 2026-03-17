// smt_ast.rs — SMT expression utilities: simplification, negation, variable
// collection, SMTLIB2 serialisation, sort inference, substitution, and script
// generation.

use crate::{SmtExpr, SmtSort};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

// ─── SmtVar ─────────────────────────────────────────────────────────────────

/// A named SMT variable together with its sort.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SmtVar {
    pub name: String,
    pub sort: SmtSort,
}

impl SmtVar {
    pub fn new(name: impl Into<String>, sort: SmtSort) -> Self {
        Self {
            name: name.into(),
            sort,
        }
    }

    pub fn bool_var(name: impl Into<String>) -> Self {
        Self::new(name, SmtSort::Bool)
    }

    pub fn int_var(name: impl Into<String>) -> Self {
        Self::new(name, SmtSort::Int)
    }

    pub fn real_var(name: impl Into<String>) -> Self {
        Self::new(name, SmtSort::Real)
    }

    /// Convert to an `SmtExpr::Var`.
    pub fn to_expr(&self) -> SmtExpr {
        SmtExpr::Var(self.name.clone(), self.sort.clone())
    }
}

impl fmt::Display for SmtVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.name, self.sort)
    }
}

// ─── SmtExpr helper implementations ─────────────────────────────────────────

impl SmtExpr {
    // ── variable collection ─────────────────────────────────────────────

    /// Collect every free variable name occurring in the expression (sorted).
    pub fn free_variables(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_var_names(&mut vars);
        vars
    }

    fn collect_var_names(&self, vars: &mut BTreeSet<String>) {
        match self {
            SmtExpr::Var(name, _) => {
                vars.insert(name.clone());
            }
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) => {}
            SmtExpr::Not(e) | SmtExpr::Neg(e) => e.collect_var_names(vars),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                for e in es {
                    e.collect_var_names(vars);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Sub(a, b) => {
                a.collect_var_names(vars);
                b.collect_var_names(vars);
            }
            SmtExpr::Ite(c, t, e) => {
                c.collect_var_names(vars);
                t.collect_var_names(vars);
                e.collect_var_names(vars);
            }
            SmtExpr::Apply(_, args) => {
                for a in args {
                    a.collect_var_names(vars);
                }
            }
        }
    }

    /// Collect typed variables as `(name, sort)` pairs.
    pub fn typed_variables(&self) -> HashMap<String, SmtSort> {
        let mut map = HashMap::new();
        self.collect_typed_vars(&mut map);
        map
    }

    fn collect_typed_vars(&self, vars: &mut HashMap<String, SmtSort>) {
        match self {
            SmtExpr::Var(name, sort) => {
                vars.insert(name.clone(), sort.clone());
            }
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_) => {}
            SmtExpr::Not(e) | SmtExpr::Neg(e) => e.collect_typed_vars(vars),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                for e in es {
                    e.collect_typed_vars(vars);
                }
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Sub(a, b) => {
                a.collect_typed_vars(vars);
                b.collect_typed_vars(vars);
            }
            SmtExpr::Ite(c, t, e) => {
                c.collect_typed_vars(vars);
                t.collect_typed_vars(vars);
                e.collect_typed_vars(vars);
            }
            SmtExpr::Apply(_, args) => {
                for a in args {
                    a.collect_typed_vars(vars);
                }
            }
        }
    }

    // ── negation ────────────────────────────────────────────────────────

    /// Logical negation, pushing negation inward via De Morgan where possible.
    pub fn negate(&self) -> SmtExpr {
        match self {
            SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
            SmtExpr::Not(inner) => (**inner).clone(),
            SmtExpr::And(es) => SmtExpr::Or(es.iter().map(|e| e.negate()).collect()),
            SmtExpr::Or(es) => SmtExpr::And(es.iter().map(|e| e.negate()).collect()),
            SmtExpr::Implies(a, b) => {
                // ¬(a → b) ≡ a ∧ ¬b
                SmtExpr::And(vec![(**a).clone(), b.negate()])
            }
            other => SmtExpr::Not(Box::new(other.clone())),
        }
    }

    // ── simplification ──────────────────────────────────────────────────

    /// Simplify using constant-folding, double-negation elimination, identity
    /// removal, zero/one absorption, and flattening of nested And/Or/Add/Mul.
    pub fn simplify(&self) -> SmtExpr {
        match self {
            // ── Not ─────────────────────────────────────────────────────
            SmtExpr::Not(inner) => {
                let s = inner.simplify();
                match s {
                    SmtExpr::Not(e) => *e,
                    SmtExpr::BoolLit(b) => SmtExpr::BoolLit(!b),
                    other => SmtExpr::Not(Box::new(other)),
                }
            }
            // ── And ─────────────────────────────────────────────────────
            SmtExpr::And(es) => {
                let simplified: Vec<SmtExpr> = es
                    .iter()
                    .map(|e| e.simplify())
                    .flat_map(|e| match e {
                        SmtExpr::And(inner) => inner,
                        other => vec![other],
                    })
                    .filter(|e| !matches!(e, SmtExpr::BoolLit(true)))
                    .collect();
                if simplified
                    .iter()
                    .any(|e| matches!(e, SmtExpr::BoolLit(false)))
                {
                    return SmtExpr::BoolLit(false);
                }
                match simplified.len() {
                    0 => SmtExpr::BoolLit(true),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::And(simplified),
                }
            }
            // ── Or ──────────────────────────────────────────────────────
            SmtExpr::Or(es) => {
                let simplified: Vec<SmtExpr> = es
                    .iter()
                    .map(|e| e.simplify())
                    .flat_map(|e| match e {
                        SmtExpr::Or(inner) => inner,
                        other => vec![other],
                    })
                    .filter(|e| !matches!(e, SmtExpr::BoolLit(false)))
                    .collect();
                if simplified
                    .iter()
                    .any(|e| matches!(e, SmtExpr::BoolLit(true)))
                {
                    return SmtExpr::BoolLit(true);
                }
                match simplified.len() {
                    0 => SmtExpr::BoolLit(false),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => SmtExpr::Or(simplified),
                }
            }
            // ── Implies ─────────────────────────────────────────────────
            SmtExpr::Implies(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (SmtExpr::BoolLit(false), _) => SmtExpr::BoolLit(true),
                    (SmtExpr::BoolLit(true), _) => sb,
                    (_, SmtExpr::BoolLit(true)) => SmtExpr::BoolLit(true),
                    (_, SmtExpr::BoolLit(false)) => sa.negate(),
                    _ => SmtExpr::Implies(Box::new(sa), Box::new(sb)),
                }
            }
            // ── Eq ──────────────────────────────────────────────────────
            SmtExpr::Eq(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                if sa == sb {
                    SmtExpr::BoolLit(true)
                } else {
                    match (&sa, &sb) {
                        (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::BoolLit(x == y),
                        (SmtExpr::BoolLit(x), SmtExpr::BoolLit(y)) => SmtExpr::BoolLit(x == y),
                        _ => SmtExpr::Eq(Box::new(sa), Box::new(sb)),
                    }
                }
            }
            // ── Neg ─────────────────────────────────────────────────────
            SmtExpr::Neg(inner) => {
                let s = inner.simplify();
                match s {
                    SmtExpr::Neg(e) => *e,
                    SmtExpr::IntLit(n) => SmtExpr::IntLit(-n),
                    SmtExpr::RealLit(r) => SmtExpr::RealLit(-r),
                    other => SmtExpr::Neg(Box::new(other)),
                }
            }
            // ── Add ─────────────────────────────────────────────────────
            SmtExpr::Add(es) => simplify_add(es),
            // ── Mul ─────────────────────────────────────────────────────
            SmtExpr::Mul(es) => simplify_mul(es),
            // ── Sub ─────────────────────────────────────────────────────
            SmtExpr::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::IntLit(x - y),
                    (SmtExpr::RealLit(x), SmtExpr::RealLit(y)) => SmtExpr::RealLit(x - y),
                    _ if sa == sb => SmtExpr::IntLit(0),
                    _ => SmtExpr::Sub(Box::new(sa), Box::new(sb)),
                }
            }
            // ── Ite ─────────────────────────────────────────────────────
            SmtExpr::Ite(c, t, e) => {
                let sc = c.simplify();
                let st = t.simplify();
                let se = e.simplify();
                match &sc {
                    SmtExpr::BoolLit(true) => st,
                    SmtExpr::BoolLit(false) => se,
                    _ if st == se => st,
                    _ => SmtExpr::Ite(Box::new(sc), Box::new(st), Box::new(se)),
                }
            }
            // ── Comparison constant-folding ─────────────────────────────
            SmtExpr::Lt(a, b) => simplify_cmp(a, b, |x, y| x < y, SmtExpr::Lt),
            SmtExpr::Le(a, b) => simplify_cmp_eq(a, b, |x, y| x <= y, SmtExpr::Le),
            SmtExpr::Gt(a, b) => simplify_cmp(a, b, |x, y| x > y, SmtExpr::Gt),
            SmtExpr::Ge(a, b) => simplify_cmp_eq(a, b, |x, y| x >= y, SmtExpr::Ge),
            // ── Leaf / Apply ────────────────────────────────────────────
            other => other.clone(),
        }
    }

    // ── SMTLIB2 serialisation ───────────────────────────────────────────

    /// Produce an SMTLIB2 s-expression string.
    pub fn to_smtlib2_string(&self) -> String {
        match self {
            SmtExpr::BoolLit(true) => "true".into(),
            SmtExpr::BoolLit(false) => "false".into(),
            SmtExpr::IntLit(n) => {
                if *n < 0 {
                    format!("(- {})", -n)
                } else {
                    n.to_string()
                }
            }
            SmtExpr::RealLit(r) => {
                if *r < 0.0 {
                    format!("(- {})", format_real(-r))
                } else {
                    format_real(*r)
                }
            }
            SmtExpr::Var(name, _) => name.clone(),
            SmtExpr::Not(e) => format!("(not {})", e.to_smtlib2_string()),
            SmtExpr::Neg(e) => format!("(- {})", e.to_smtlib2_string()),
            SmtExpr::And(es) => nary_smtlib2("and", es, "true"),
            SmtExpr::Or(es) => nary_smtlib2("or", es, "false"),
            SmtExpr::Add(es) => nary_smtlib2("+", es, "0"),
            SmtExpr::Mul(es) => nary_smtlib2("*", es, "1"),
            SmtExpr::Implies(a, b) => {
                format!("(=> {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Eq(a, b) => {
                format!("(= {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Lt(a, b) => {
                format!("(< {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Le(a, b) => {
                format!("(<= {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Gt(a, b) => {
                format!("(> {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Ge(a, b) => {
                format!("(>= {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Sub(a, b) => {
                format!("(- {} {})", a.to_smtlib2_string(), b.to_smtlib2_string())
            }
            SmtExpr::Ite(c, t, e) => format!(
                "(ite {} {} {})",
                c.to_smtlib2_string(),
                t.to_smtlib2_string(),
                e.to_smtlib2_string()
            ),
            SmtExpr::Apply(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let arg_strs: Vec<String> =
                        args.iter().map(|a| a.to_smtlib2_string()).collect();
                    format!("({} {})", name, arg_strs.join(" "))
                }
            }
        }
    }

    /// Build a full SMTLIB2 script from a set of constraint expressions:
    /// logic declaration, variable declarations, assertions, check-sat.
    pub fn to_smtlib2_script(constraints: &[SmtExpr]) -> String {
        let mut script = String::new();
        script.push_str("(set-logic QF_LRA)\n");

        let mut all_vars: HashMap<String, SmtSort> = HashMap::new();
        for c in constraints {
            c.collect_typed_vars(&mut all_vars);
        }

        let mut sorted_names: Vec<&String> = all_vars.keys().collect();
        sorted_names.sort();
        for name in sorted_names {
            let sort = &all_vars[name];
            script.push_str(&format!("(declare-fun {} () {})\n", name, sort));
        }

        script.push('\n');
        for c in constraints {
            script.push_str(&format!("(assert {})\n", c.to_smtlib2_string()));
        }

        script.push_str("\n(check-sat)\n(get-model)\n");
        script
    }

    // ── sort inference ──────────────────────────────────────────────────

    /// Infer the sort (type) of the expression.
    pub fn infer_sort(&self) -> SmtSort {
        match self {
            SmtExpr::BoolLit(_)
            | SmtExpr::Not(_)
            | SmtExpr::And(_)
            | SmtExpr::Or(_)
            | SmtExpr::Implies(_, _)
            | SmtExpr::Eq(_, _)
            | SmtExpr::Lt(_, _)
            | SmtExpr::Le(_, _)
            | SmtExpr::Gt(_, _)
            | SmtExpr::Ge(_, _) => SmtSort::Bool,
            SmtExpr::IntLit(_) => SmtSort::Int,
            SmtExpr::RealLit(_) => SmtSort::Real,
            SmtExpr::Var(_, sort) => sort.clone(),
            SmtExpr::Neg(e) | SmtExpr::Sub(e, _) => e.infer_sort(),
            SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                if es.iter().any(|e| matches!(e.infer_sort(), SmtSort::Real)) {
                    SmtSort::Real
                } else {
                    SmtSort::Int
                }
            }
            SmtExpr::Ite(_, t, _) => t.infer_sort(),
            SmtExpr::Apply(_, _) => SmtSort::Bool,
        }
    }

    // ── metrics ─────────────────────────────────────────────────────────

    /// Number of AST nodes.
    pub fn node_count(&self) -> usize {
        match self {
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_)
            | SmtExpr::Var(_, _) => 1,
            SmtExpr::Not(e) | SmtExpr::Neg(e) => 1 + e.node_count(),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                1 + es.iter().map(|e| e.node_count()).sum::<usize>()
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Sub(a, b) => 1 + a.node_count() + b.node_count(),
            SmtExpr::Ite(c, t, e) => 1 + c.node_count() + t.node_count() + e.node_count(),
            SmtExpr::Apply(_, args) => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        }
    }

    /// Maximum nesting depth.
    pub fn depth(&self) -> usize {
        match self {
            SmtExpr::BoolLit(_) | SmtExpr::IntLit(_) | SmtExpr::RealLit(_)
            | SmtExpr::Var(_, _) => 0,
            SmtExpr::Not(e) | SmtExpr::Neg(e) => 1 + e.depth(),
            SmtExpr::And(es) | SmtExpr::Or(es) | SmtExpr::Add(es) | SmtExpr::Mul(es) => {
                1 + es.iter().map(|e| e.depth()).max().unwrap_or(0)
            }
            SmtExpr::Implies(a, b)
            | SmtExpr::Eq(a, b)
            | SmtExpr::Lt(a, b)
            | SmtExpr::Le(a, b)
            | SmtExpr::Gt(a, b)
            | SmtExpr::Ge(a, b)
            | SmtExpr::Sub(a, b) => 1 + a.depth().max(b.depth()),
            SmtExpr::Ite(c, t, e) => 1 + c.depth().max(t.depth()).max(e.depth()),
            SmtExpr::Apply(_, args) => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
        }
    }

    /// `true` when the expression contains no variables.
    pub fn is_constant(&self) -> bool {
        self.free_variables().is_empty()
    }

    // ── substitution ────────────────────────────────────────────────────

    /// Replace every occurrence of the variable `var_name` with `replacement`.
    pub fn substitute(&self, var_name: &str, replacement: &SmtExpr) -> SmtExpr {
        match self {
            SmtExpr::Var(name, _) if name == var_name => replacement.clone(),
            SmtExpr::Var(_, _) | SmtExpr::BoolLit(_) | SmtExpr::IntLit(_)
            | SmtExpr::RealLit(_) => self.clone(),
            SmtExpr::Not(e) => SmtExpr::Not(Box::new(e.substitute(var_name, replacement))),
            SmtExpr::Neg(e) => SmtExpr::Neg(Box::new(e.substitute(var_name, replacement))),
            SmtExpr::And(es) => {
                SmtExpr::And(es.iter().map(|e| e.substitute(var_name, replacement)).collect())
            }
            SmtExpr::Or(es) => {
                SmtExpr::Or(es.iter().map(|e| e.substitute(var_name, replacement)).collect())
            }
            SmtExpr::Add(es) => {
                SmtExpr::Add(es.iter().map(|e| e.substitute(var_name, replacement)).collect())
            }
            SmtExpr::Mul(es) => {
                SmtExpr::Mul(es.iter().map(|e| e.substitute(var_name, replacement)).collect())
            }
            SmtExpr::Implies(a, b) => SmtExpr::Implies(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Eq(a, b) => SmtExpr::Eq(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Lt(a, b) => SmtExpr::Lt(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Le(a, b) => SmtExpr::Le(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Gt(a, b) => SmtExpr::Gt(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Ge(a, b) => SmtExpr::Ge(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Sub(a, b) => SmtExpr::Sub(
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            SmtExpr::Ite(c, t, e) => SmtExpr::Ite(
                Box::new(c.substitute(var_name, replacement)),
                Box::new(t.substitute(var_name, replacement)),
                Box::new(e.substitute(var_name, replacement)),
            ),
            SmtExpr::Apply(name, args) => SmtExpr::Apply(
                name.clone(),
                args.iter()
                    .map(|a| a.substitute(var_name, replacement))
                    .collect(),
            ),
        }
    }

    /// Bulk substitution of several variables at once.
    pub fn substitute_many(&self, bindings: &HashMap<String, SmtExpr>) -> SmtExpr {
        if bindings.is_empty() {
            return self.clone();
        }
        let mut result = self.clone();
        for (var, replacement) in bindings {
            result = result.substitute(var, replacement);
        }
        result
    }
}

// ─── private helpers ────────────────────────────────────────────────────────

fn format_real(r: f64) -> String {
    if r.fract() == 0.0 {
        format!("{}.0", r as i64)
    } else {
        format!("{}", r)
    }
}

fn nary_smtlib2(op: &str, es: &[SmtExpr], identity: &str) -> String {
    if es.is_empty() {
        identity.to_string()
    } else if es.len() == 1 {
        es[0].to_smtlib2_string()
    } else {
        let args: Vec<String> = es.iter().map(|e| e.to_smtlib2_string()).collect();
        format!("({} {})", op, args.join(" "))
    }
}

fn simplify_add(es: &[SmtExpr]) -> SmtExpr {
    let simplified: Vec<SmtExpr> = es
        .iter()
        .map(|e| e.simplify())
        .flat_map(|e| match e {
            SmtExpr::Add(inner) => inner,
            other => vec![other],
        })
        .collect();

    let mut int_sum: i64 = 0;
    let mut real_sum: f64 = 0.0;
    let mut has_real = false;
    let mut others = Vec::new();

    for e in simplified {
        match e {
            SmtExpr::IntLit(n) => int_sum += n,
            SmtExpr::RealLit(r) => {
                real_sum += r;
                has_real = true;
            }
            other => others.push(other),
        }
    }

    let const_val = real_sum + int_sum as f64;
    if const_val != 0.0 || others.is_empty() {
        if has_real {
            others.push(SmtExpr::RealLit(const_val));
        } else {
            others.push(SmtExpr::IntLit(int_sum));
        }
    }

    match others.len() {
        0 => SmtExpr::IntLit(0),
        1 => others.into_iter().next().unwrap(),
        _ => SmtExpr::Add(others),
    }
}

fn simplify_mul(es: &[SmtExpr]) -> SmtExpr {
    let simplified: Vec<SmtExpr> = es
        .iter()
        .map(|e| e.simplify())
        .flat_map(|e| match e {
            SmtExpr::Mul(inner) => inner,
            other => vec![other],
        })
        .collect();

    // Short-circuit on zero.
    for e in &simplified {
        match e {
            SmtExpr::IntLit(0) => return SmtExpr::IntLit(0),
            SmtExpr::RealLit(r) if *r == 0.0 => return SmtExpr::RealLit(0.0),
            _ => {}
        }
    }

    let mut int_prod: i64 = 1;
    let mut real_prod: f64 = 1.0;
    let mut has_real = false;
    let mut others = Vec::new();

    for e in simplified {
        match e {
            SmtExpr::IntLit(n) => int_prod *= n,
            SmtExpr::RealLit(r) => {
                real_prod *= r;
                has_real = true;
            }
            other => others.push(other),
        }
    }

    let const_val = real_prod * int_prod as f64;
    if const_val != 1.0 || others.is_empty() {
        if has_real {
            others.insert(0, SmtExpr::RealLit(const_val));
        } else {
            others.insert(0, SmtExpr::IntLit(int_prod));
        }
    }

    match others.len() {
        0 => SmtExpr::IntLit(1),
        1 => others.into_iter().next().unwrap(),
        _ => SmtExpr::Mul(others),
    }
}

/// Simplify a strict comparison (< or >) with constant folding.
fn simplify_cmp(
    a: &SmtExpr,
    b: &SmtExpr,
    int_cmp: impl Fn(i64, i64) -> bool,
    ctor: fn(Box<SmtExpr>, Box<SmtExpr>) -> SmtExpr,
) -> SmtExpr {
    let sa = a.simplify();
    let sb = b.simplify();
    match (&sa, &sb) {
        (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::BoolLit(int_cmp(*x, *y)),
        _ => ctor(Box::new(sa), Box::new(sb)),
    }
}

/// Simplify a non-strict comparison (<= or >=), also yielding `true` for
/// structurally equal operands.
fn simplify_cmp_eq(
    a: &SmtExpr,
    b: &SmtExpr,
    int_cmp: impl Fn(i64, i64) -> bool,
    ctor: fn(Box<SmtExpr>, Box<SmtExpr>) -> SmtExpr,
) -> SmtExpr {
    let sa = a.simplify();
    let sb = b.simplify();
    if sa == sb {
        return SmtExpr::BoolLit(true);
    }
    match (&sa, &sb) {
        (SmtExpr::IntLit(x), SmtExpr::IntLit(y)) => SmtExpr::BoolLit(int_cmp(*x, *y)),
        _ => ctor(Box::new(sa), Box::new(sb)),
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bvar(name: &str) -> SmtExpr {
        SmtExpr::Var(name.into(), SmtSort::Bool)
    }
    fn ivar(name: &str) -> SmtExpr {
        SmtExpr::Var(name.into(), SmtSort::Int)
    }

    #[test]
    fn test_free_variables() {
        let expr = SmtExpr::And(vec![bvar("x"), SmtExpr::Or(vec![bvar("y"), bvar("x")])]);
        let vars = expr.free_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_negate_de_morgan() {
        let expr = SmtExpr::And(vec![bvar("a"), bvar("b")]);
        let neg = expr.negate();
        match neg {
            SmtExpr::Or(es) => {
                assert_eq!(es.len(), 2);
                assert!(matches!(&es[0], SmtExpr::Not(_)));
            }
            _ => panic!("expected Or"),
        }
    }

    #[test]
    fn test_double_negation_simplify() {
        let expr = SmtExpr::Not(Box::new(SmtExpr::Not(Box::new(bvar("x")))));
        assert_eq!(expr.simplify(), bvar("x"));
    }

    #[test]
    fn test_and_true_removal() {
        let expr = SmtExpr::And(vec![SmtExpr::BoolLit(true), bvar("x")]);
        assert_eq!(expr.simplify(), bvar("x"));
    }

    #[test]
    fn test_and_false_short_circuit() {
        let expr = SmtExpr::And(vec![bvar("x"), SmtExpr::BoolLit(false)]);
        assert_eq!(expr.simplify(), SmtExpr::BoolLit(false));
    }

    #[test]
    fn test_or_identity() {
        let expr = SmtExpr::Or(vec![SmtExpr::BoolLit(false), bvar("y")]);
        assert_eq!(expr.simplify(), bvar("y"));
    }

    #[test]
    fn test_add_constant_folding() {
        let expr = SmtExpr::Add(vec![SmtExpr::IntLit(3), SmtExpr::IntLit(7), ivar("z")]);
        let s = expr.simplify();
        match s {
            SmtExpr::Add(es) => {
                assert!(es.contains(&ivar("z")));
                assert!(es.contains(&SmtExpr::IntLit(10)));
            }
            _ => panic!("expected Add, got {:?}", s),
        }
    }

    #[test]
    fn test_mul_zero() {
        let expr = SmtExpr::Mul(vec![ivar("x"), SmtExpr::IntLit(0)]);
        assert_eq!(expr.simplify(), SmtExpr::IntLit(0));
    }

    #[test]
    fn test_implies_constant_folding() {
        let expr = SmtExpr::Implies(Box::new(SmtExpr::BoolLit(true)), Box::new(bvar("p")));
        assert_eq!(expr.simplify(), bvar("p"));
    }

    #[test]
    fn test_ite_constant_condition() {
        let expr = SmtExpr::Ite(
            Box::new(SmtExpr::BoolLit(false)),
            Box::new(SmtExpr::IntLit(1)),
            Box::new(SmtExpr::IntLit(2)),
        );
        assert_eq!(expr.simplify(), SmtExpr::IntLit(2));
    }

    #[test]
    fn test_smtlib2_string() {
        let expr = SmtExpr::And(vec![
            bvar("x"),
            SmtExpr::Implies(Box::new(bvar("y")), Box::new(SmtExpr::BoolLit(true))),
        ]);
        let s = expr.to_smtlib2_string();
        assert_eq!(s, "(and x (=> y true))");
    }

    #[test]
    fn test_smtlib2_negative_int() {
        let e = SmtExpr::IntLit(-5);
        assert_eq!(e.to_smtlib2_string(), "(- 5)");
    }

    #[test]
    fn test_script_generation() {
        let constraints = vec![SmtExpr::Le(
            Box::new(ivar("cost")),
            Box::new(SmtExpr::IntLit(100)),
        )];
        let script = SmtExpr::to_smtlib2_script(&constraints);
        assert!(script.contains("(declare-fun cost () Int)"));
        assert!(script.contains("(assert (<= cost 100))"));
        assert!(script.contains("(check-sat)"));
    }

    #[test]
    fn test_substitute() {
        let expr = SmtExpr::Add(vec![ivar("x"), SmtExpr::IntLit(1)]);
        let result = expr.substitute("x", &SmtExpr::IntLit(42));
        assert_eq!(
            result,
            SmtExpr::Add(vec![SmtExpr::IntLit(42), SmtExpr::IntLit(1)])
        );
    }

    #[test]
    fn test_infer_sort() {
        assert_eq!(SmtExpr::BoolLit(true).infer_sort(), SmtSort::Bool);
        assert_eq!(SmtExpr::IntLit(0).infer_sort(), SmtSort::Int);
        assert_eq!(SmtExpr::RealLit(1.0).infer_sort(), SmtSort::Real);
        let cmp = SmtExpr::Lt(Box::new(ivar("x")), Box::new(ivar("y")));
        assert_eq!(cmp.infer_sort(), SmtSort::Bool);
    }

    #[test]
    fn test_node_count_and_depth() {
        let leaf = SmtExpr::IntLit(1);
        assert_eq!(leaf.node_count(), 1);
        assert_eq!(leaf.depth(), 0);

        let nested = SmtExpr::And(vec![
            bvar("a"),
            SmtExpr::Or(vec![bvar("b"), bvar("c")]),
        ]);
        assert_eq!(nested.node_count(), 5);
        assert_eq!(nested.depth(), 2);
    }

    #[test]
    fn test_smtvar_to_expr() {
        let v = SmtVar::bool_var("comply_art6");
        let e = v.to_expr();
        assert_eq!(e, SmtExpr::Var("comply_art6".into(), SmtSort::Bool));
    }

    #[test]
    fn test_eq_simplify_same_operands() {
        let expr = SmtExpr::Eq(Box::new(ivar("x")), Box::new(ivar("x")));
        assert_eq!(expr.simplify(), SmtExpr::BoolLit(true));
    }

    #[test]
    fn test_sub_same_operands() {
        let expr = SmtExpr::Sub(Box::new(ivar("x")), Box::new(ivar("x")));
        assert_eq!(expr.simplify(), SmtExpr::IntLit(0));
    }

    #[test]
    fn test_nested_and_flattening() {
        let inner = SmtExpr::And(vec![bvar("a"), bvar("b")]);
        let outer = SmtExpr::And(vec![inner, bvar("c")]);
        match outer.simplify() {
            SmtExpr::And(es) => assert_eq!(es.len(), 3),
            _ => panic!("expected flattened And"),
        }
    }
}
