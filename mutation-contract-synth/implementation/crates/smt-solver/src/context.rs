//! SMT context — assertion management, declaration tracking, and formula
//! conversion from the shared `Formula` type to `SmtExpr`.

use std::collections::HashMap;

use shared_types::{Formula, Predicate, Relation, Term};

use crate::ast::{SmtCommand, SmtExpr, SmtScript, SmtSort};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Tracks declared symbols, scoping, and assertion state.
#[derive(Debug, Clone)]
pub struct SmtContext {
    /// The SMT-LIB logic (e.g. `"QF_LIA"`).
    logic: String,
    /// Declared constants: name → sort.
    declarations: HashMap<String, SmtSort>,
    /// Named assertions for unsat-core extraction.
    named_assertions: HashMap<String, SmtExpr>,
    /// The script being built.
    script: SmtScript,
    /// Current scope depth (incremented by push, decremented by pop).
    scope_depth: u32,
    /// Stack of declaration snapshots for scope management.
    scope_stack: Vec<HashMap<String, SmtSort>>,
    /// Whether produce-unsat-cores is enabled.
    unsat_cores_enabled: bool,
    /// Whether produce-models is enabled.
    models_enabled: bool,
    /// Counter for generating unique assertion names.
    assertion_counter: u64,
}

impl SmtContext {
    /// Create a new context with the given logic.
    pub fn new(logic: &str) -> Self {
        let mut script = SmtScript::new();
        script.set_logic(logic);
        SmtContext {
            logic: logic.to_string(),
            declarations: HashMap::new(),
            named_assertions: HashMap::new(),
            script,
            scope_depth: 0,
            scope_stack: Vec::new(),
            unsat_cores_enabled: false,
            models_enabled: false,
            assertion_counter: 0,
        }
    }

    /// Create a QF_LIA context (default for MutSpec).
    pub fn qf_lia() -> Self {
        let mut ctx = Self::new("QF_LIA");
        ctx.enable_models();
        ctx
    }

    /// Create a QF_LIA context with unsat-core support.
    pub fn qf_lia_with_cores() -> Self {
        let mut ctx = Self::new("QF_LIA");
        ctx.enable_models();
        ctx.enable_unsat_cores();
        ctx
    }

    /// Enable `(set-option :produce-models true)`.
    pub fn enable_models(&mut self) {
        if !self.models_enabled {
            self.models_enabled = true;
            self.script.commands.insert(
                0,
                SmtCommand::SetOption("produce-models".into(), "true".into()),
            );
        }
    }

    /// Enable `(set-option :produce-unsat-cores true)`.
    pub fn enable_unsat_cores(&mut self) {
        if !self.unsat_cores_enabled {
            self.unsat_cores_enabled = true;
            self.script.commands.insert(
                0,
                SmtCommand::SetOption("produce-unsat-cores".into(), "true".into()),
            );
        }
    }

    /// Get the logic name.
    pub fn logic(&self) -> &str {
        &self.logic
    }

    /// Get the current scope depth.
    pub fn scope_depth(&self) -> u32 {
        self.scope_depth
    }

    // -- Declaration management --

    /// Declare an integer constant.
    pub fn declare_int(&mut self, name: &str) {
        self.declare_const(name, SmtSort::Int);
    }

    /// Declare a boolean constant.
    pub fn declare_bool(&mut self, name: &str) {
        self.declare_const(name, SmtSort::Bool);
    }

    /// Declare a constant of arbitrary sort.
    pub fn declare_const(&mut self, name: &str, sort: SmtSort) {
        if !self.declarations.contains_key(name) {
            self.declarations.insert(name.to_string(), sort.clone());
            self.script.declare_const(name, sort);
        }
    }

    /// Declare a function.
    pub fn declare_fun(&mut self, name: &str, args: Vec<SmtSort>, ret: SmtSort) {
        if !self.declarations.contains_key(name) {
            self.declarations.insert(name.to_string(), ret.clone());
            self.script.declare_fun(name, args, ret);
        }
    }

    /// Check if a symbol has been declared.
    pub fn is_declared(&self, name: &str) -> bool {
        self.declarations.contains_key(name)
    }

    /// Get the sort of a declared symbol.
    pub fn sort_of(&self, name: &str) -> Option<&SmtSort> {
        self.declarations.get(name)
    }

    /// Auto-declare all free variables in an expression as Int constants.
    pub fn auto_declare_ints(&mut self, expr: &SmtExpr) {
        for sym in expr.free_symbols() {
            if !self.is_declared(&sym) {
                self.declare_int(&sym);
            }
        }
    }

    // -- Assertion management --

    /// Assert an expression.
    pub fn assert(&mut self, expr: SmtExpr) {
        self.script.assert(expr);
    }

    /// Assert a named expression (for unsat-core tracking).
    pub fn assert_named(&mut self, name: &str, expr: SmtExpr) {
        let named = expr.clone().named(name);
        self.named_assertions.insert(name.to_string(), expr);
        self.script.assert(named);
    }

    /// Assert with auto-generated name, returns the generated name.
    pub fn assert_tracked(&mut self, expr: SmtExpr) -> String {
        let name = format!("a{}", self.assertion_counter);
        self.assertion_counter += 1;
        self.assert_named(&name, expr);
        name
    }

    // -- Scope management --

    /// Push a new scope level.
    pub fn push(&mut self) {
        self.scope_stack.push(self.declarations.clone());
        self.scope_depth += 1;
        self.script.push_scope(1);
    }

    /// Pop a scope level.
    pub fn pop(&mut self) {
        if let Some(prev) = self.scope_stack.pop() {
            self.declarations = prev;
            self.scope_depth -= 1;
            self.script.pop_scope(1);
        }
    }

    // -- Script generation --

    /// Add a check-sat command.
    pub fn check_sat(&mut self) {
        self.script.check_sat();
    }

    /// Add a get-model command.
    pub fn get_model(&mut self) {
        self.script.get_model();
    }

    /// Add a get-unsat-core command.
    pub fn get_unsat_core(&mut self) {
        self.script.get_unsat_core();
    }

    /// Finalize and return the script (consuming the context).
    pub fn into_script(mut self) -> SmtScript {
        self.script.exit();
        self.script
    }

    /// Get a reference to the current script.
    pub fn script(&self) -> &SmtScript {
        &self.script
    }

    /// Render the script to a string.
    pub fn render(&self) -> String {
        self.script.render()
    }

    /// Get total assertion count.
    pub fn assertion_count(&self) -> usize {
        self.script.assertion_count()
    }

    /// Get all declared constant names.
    pub fn declared_names(&self) -> Vec<String> {
        self.declarations.keys().cloned().collect()
    }

    // -- Formula conversion --

    /// Convert a `shared_types::Formula` to an `SmtExpr`, auto-declaring
    /// any free variables found.
    pub fn formula_to_smt(&mut self, formula: &Formula) -> SmtExpr {
        let expr = convert_formula(formula);
        self.auto_declare_ints(&expr);
        expr
    }

    /// Assert a `shared_types::Formula`, converting and auto-declaring.
    pub fn assert_formula(&mut self, formula: &Formula) {
        let expr = self.formula_to_smt(formula);
        self.assert(expr);
    }

    /// Assert a formula with a given name.
    pub fn assert_formula_named(&mut self, name: &str, formula: &Formula) {
        let expr = self.formula_to_smt(formula);
        self.assert_named(name, expr);
    }

    /// Build an SMT validity check: assert ¬φ, check-sat.
    /// If UNSAT, φ is valid.
    pub fn build_validity_check(&mut self, formula: &Formula) {
        let expr = self.formula_to_smt(formula);
        self.assert(SmtExpr::not(expr));
        self.check_sat();
    }

    /// Build an SMT satisfiability check: assert φ, check-sat.
    pub fn build_sat_check(&mut self, formula: &Formula) {
        let expr = self.formula_to_smt(formula);
        self.assert(expr);
        self.check_sat();
        if self.models_enabled {
            self.get_model();
        }
    }

    /// Build an entailment check: φ ⊢ ψ ≡ ¬(φ ∧ ¬ψ) is UNSAT.
    pub fn build_entailment_check(&mut self, premise: &Formula, conclusion: &Formula) {
        let p = self.formula_to_smt(premise);
        let c = self.formula_to_smt(conclusion);
        let negated = SmtExpr::and(p, SmtExpr::not(c));
        self.assert(negated);
        self.check_sat();
    }

    /// Build an equivalence check: φ ↔ ψ.
    pub fn build_equivalence_check(&mut self, lhs: &Formula, rhs: &Formula) {
        let l = self.formula_to_smt(lhs);
        let r = self.formula_to_smt(rhs);
        let not_equiv = SmtExpr::not(SmtExpr::Eq(Box::new(l), Box::new(r)));
        self.assert(not_equiv);
        self.check_sat();
    }
}

// ---------------------------------------------------------------------------
// Formula → SmtExpr conversion
// ---------------------------------------------------------------------------

/// Convert a `shared_types::Formula` to an `SmtExpr`.
pub fn convert_formula(formula: &Formula) -> SmtExpr {
    match formula {
        Formula::True => SmtExpr::True,
        Formula::False => SmtExpr::False,
        Formula::Atom(pred) => convert_predicate(pred),
        Formula::And(conjuncts) => {
            let exprs: Vec<_> = conjuncts.iter().map(convert_formula).collect();
            SmtExpr::and_many(exprs)
        }
        Formula::Or(disjuncts) => {
            let exprs: Vec<_> = disjuncts.iter().map(convert_formula).collect();
            SmtExpr::or_many(exprs)
        }
        Formula::Not(inner) => SmtExpr::not(convert_formula(inner)),
        Formula::Implies(a, b) => SmtExpr::implies(convert_formula(a), convert_formula(b)),
        Formula::Iff(a, b) => {
            let la = convert_formula(a);
            let lb = convert_formula(b);
            SmtExpr::and(
                SmtExpr::implies(la.clone(), lb.clone()),
                SmtExpr::implies(lb, la),
            )
        }
        Formula::Forall(vars, body) => SmtExpr::Forall(
            vars.iter().map(|v| (v.clone(), SmtSort::Int)).collect(),
            Box::new(convert_formula(body)),
        ),
        Formula::Exists(vars, body) => SmtExpr::Exists(
            vars.iter().map(|v| (v.clone(), SmtSort::Int)).collect(),
            Box::new(convert_formula(body)),
        ),
    }
}

/// Convert a `shared_types::Predicate` to an `SmtExpr`.
pub fn convert_predicate(pred: &Predicate) -> SmtExpr {
    let lhs = convert_term(&pred.left);
    let rhs = convert_term(&pred.right);
    match pred.relation {
        Relation::Eq => SmtExpr::eq(lhs, rhs),
        Relation::Ne => SmtExpr::not(SmtExpr::eq(lhs, rhs)),
        Relation::Lt => SmtExpr::lt(lhs, rhs),
        Relation::Le => SmtExpr::le(lhs, rhs),
        Relation::Gt => SmtExpr::gt(lhs, rhs),
        Relation::Ge => SmtExpr::ge(lhs, rhs),
    }
}

/// Convert a `shared_types::Term` to an `SmtExpr`.
pub fn convert_term(term: &Term) -> SmtExpr {
    match term {
        Term::Const(v) => SmtExpr::int(*v),
        Term::Var(name) => SmtExpr::sym(name.as_str()),
        Term::Add(a, b) => SmtExpr::add(convert_term(a), convert_term(b)),
        Term::Sub(a, b) => SmtExpr::sub(convert_term(a), convert_term(b)),
        Term::Mul(coeff, inner) => SmtExpr::mul(SmtExpr::int(*coeff), convert_term(inner)),
        Term::Neg(inner) => SmtExpr::Sub(vec![SmtExpr::int(0), convert_term(inner)]),
        Term::ArraySelect(arr, idx) => SmtExpr::select(convert_term(arr), convert_term(idx)),
        Term::Ite(cond, t, e) => {
            SmtExpr::ite(convert_formula(cond), convert_term(t), convert_term(e))
        }
        Term::Old(inner) => {
            let inner_expr = convert_term(inner);
            match &inner_expr {
                SmtExpr::Symbol(name) => SmtExpr::sym(format!("{}_old", name)),
                _ => inner_expr,
            }
        }
        Term::Result => SmtExpr::sym("__result"),
    }
}

/// Convert an `SmtExpr` back to a `shared_types::Formula` (best-effort).
pub fn smt_to_formula(expr: &SmtExpr) -> Option<Formula> {
    match expr {
        SmtExpr::True | SmtExpr::BoolLit(true) => Some(Formula::True),
        SmtExpr::False | SmtExpr::BoolLit(false) => Some(Formula::False),
        SmtExpr::Not(e) => smt_to_formula(e).map(|f| Formula::not(f)),
        SmtExpr::And(es) => {
            let fs: Option<Vec<_>> = es.iter().map(smt_to_formula).collect();
            fs.map(Formula::And)
        }
        SmtExpr::Or(es) => {
            let fs: Option<Vec<_>> = es.iter().map(smt_to_formula).collect();
            fs.map(Formula::Or)
        }
        SmtExpr::Implies(a, b) => {
            let fa = smt_to_formula(a)?;
            let fb = smt_to_formula(b)?;
            Some(Formula::implies(fa, fb))
        }
        SmtExpr::Eq(a, b) => {
            let ta = smt_to_term(a)?;
            let tb = smt_to_term(b)?;
            Some(Formula::Atom(Predicate {
                relation: Relation::Eq,
                left: ta,
                right: tb,
            }))
        }
        SmtExpr::Lt(a, b) => {
            let ta = smt_to_term(a)?;
            let tb = smt_to_term(b)?;
            Some(Formula::Atom(Predicate {
                relation: Relation::Lt,
                left: ta,
                right: tb,
            }))
        }
        SmtExpr::Le(a, b) => {
            let ta = smt_to_term(a)?;
            let tb = smt_to_term(b)?;
            Some(Formula::Atom(Predicate {
                relation: Relation::Le,
                left: ta,
                right: tb,
            }))
        }
        SmtExpr::Gt(a, b) => {
            let ta = smt_to_term(a)?;
            let tb = smt_to_term(b)?;
            Some(Formula::Atom(Predicate {
                relation: Relation::Gt,
                left: ta,
                right: tb,
            }))
        }
        SmtExpr::Ge(a, b) => {
            let ta = smt_to_term(a)?;
            let tb = smt_to_term(b)?;
            Some(Formula::Atom(Predicate {
                relation: Relation::Ge,
                left: ta,
                right: tb,
            }))
        }
        _ => None,
    }
}

/// Convert an `SmtExpr` to a `shared_types::Term` (best-effort).
pub fn smt_to_term(expr: &SmtExpr) -> Option<Term> {
    match expr {
        SmtExpr::IntLit(v) => Some(Term::Const(*v)),
        SmtExpr::Symbol(s) => Some(Term::Var(s.clone())),
        SmtExpr::Add(es) if es.len() == 2 => {
            let a = smt_to_term(&es[0])?;
            let b = smt_to_term(&es[1])?;
            Some(Term::Add(Box::new(a), Box::new(b)))
        }
        SmtExpr::Sub(es) if es.len() == 2 => {
            let a = smt_to_term(&es[0])?;
            let b = smt_to_term(&es[1])?;
            Some(Term::Sub(Box::new(a), Box::new(b)))
        }
        SmtExpr::Sub(es) if es.len() == 1 => {
            let inner = smt_to_term(&es[0])?;
            Some(Term::Neg(Box::new(inner)))
        }
        SmtExpr::Select(a, i) => {
            let arr = smt_to_term(a)?;
            let idx = smt_to_term(i)?;
            Some(Term::ArraySelect(Box::new(arr), Box::new(idx)))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qf_lia_context() {
        let ctx = SmtContext::qf_lia();
        assert_eq!(ctx.logic(), "QF_LIA");
        assert_eq!(ctx.scope_depth(), 0);
    }

    #[test]
    fn test_declare_and_assert() {
        let mut ctx = SmtContext::qf_lia();
        ctx.declare_int("x");
        ctx.declare_int("y");
        assert!(ctx.is_declared("x"));
        assert!(ctx.is_declared("y"));
        assert!(!ctx.is_declared("z"));
        ctx.assert(SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)));
        assert_eq!(ctx.assertion_count(), 1);
    }

    #[test]
    fn test_scope_management() {
        let mut ctx = SmtContext::qf_lia();
        ctx.declare_int("x");
        ctx.push();
        ctx.declare_int("y");
        assert!(ctx.is_declared("y"));
        ctx.pop();
        assert!(!ctx.is_declared("y"));
        assert!(ctx.is_declared("x"));
    }

    #[test]
    fn test_formula_conversion_atom() {
        let f = Formula::Atom(Predicate {
            relation: Relation::Le,
            lhs: Term::Var("x".into()),
            rhs: Term::Const(10),
        });
        let mut ctx = SmtContext::qf_lia();
        let expr = ctx.formula_to_smt(&f);
        assert_eq!(expr.to_string(), "(<= x 10)");
    }

    #[test]
    fn test_formula_conversion_conjunction() {
        let f = Formula::And(vec![
            Formula::Atom(Predicate {
                relation: Relation::Ge,
                lhs: Term::Var("x".into()),
                rhs: Term::Const(0),
            }),
            Formula::Atom(Predicate {
                relation: Relation::Lt,
                lhs: Term::Var("x".into()),
                rhs: Term::Const(100),
            }),
        ]);
        let mut ctx = SmtContext::qf_lia();
        let expr = ctx.formula_to_smt(&f);
        assert!(ctx.is_declared("x"));
        let s = expr.to_string();
        assert!(s.contains("and"));
        assert!(s.contains(">= x 0"));
        assert!(s.contains("< x 100"));
    }

    #[test]
    fn test_formula_conversion_negation() {
        let f = Formula::Not(Box::new(Formula::Atom(Predicate {
            relation: Relation::Eq,
            lhs: Term::Var("a".into()),
            rhs: Term::Var("b".into()),
        })));
        let mut ctx = SmtContext::qf_lia();
        let expr = ctx.formula_to_smt(&f);
        assert_eq!(expr.to_string(), "(not (= a b))");
    }

    #[test]
    fn test_assert_tracked() {
        let mut ctx = SmtContext::qf_lia_with_cores();
        let n1 = ctx.assert_tracked(SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(5)));
        let n2 = ctx.assert_tracked(SmtExpr::ge(SmtExpr::sym("x"), SmtExpr::int(10)));
        assert_eq!(n1, "a0");
        assert_eq!(n2, "a1");
        assert_eq!(ctx.assertion_count(), 2);
    }

    #[test]
    fn test_entailment_script() {
        let premise = Formula::Atom(Predicate {
            relation: Relation::Le,
            lhs: Term::Var("x".into()),
            rhs: Term::Const(5),
        });
        let conclusion = Formula::Atom(Predicate {
            relation: Relation::Le,
            lhs: Term::Var("x".into()),
            rhs: Term::Const(10),
        });
        let mut ctx = SmtContext::qf_lia();
        ctx.build_entailment_check(&premise, &conclusion);
        let rendered = ctx.render();
        assert!(rendered.contains("check-sat"));
    }

    #[test]
    fn test_roundtrip_atom() {
        let f = Formula::Atom(Predicate {
            relation: Relation::Le,
            lhs: Term::Var("x".into()),
            rhs: Term::Const(42),
        });
        let smt = convert_formula(&f);
        let back = smt_to_formula(&smt).unwrap();
        assert_eq!(f, back);
    }

    #[test]
    fn test_convert_term_arithmetic() {
        let t = Term::Add(Box::new(Term::Var("x".into())), Box::new(Term::Const(1)));
        let expr = convert_term(&t);
        assert_eq!(expr.to_string(), "(+ x 1)");
    }

    #[test]
    fn test_auto_declare() {
        let mut ctx = SmtContext::qf_lia();
        let expr = SmtExpr::le(SmtExpr::sym("a"), SmtExpr::sym("b"));
        ctx.auto_declare_ints(&expr);
        assert!(ctx.is_declared("a"));
        assert!(ctx.is_declared("b"));
    }

    #[test]
    fn test_named_assertion() {
        let mut ctx = SmtContext::qf_lia_with_cores();
        ctx.assert_named("my_assert", SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(5)));
        let rendered = ctx.render();
        assert!(rendered.contains(":named my_assert"));
    }
}
