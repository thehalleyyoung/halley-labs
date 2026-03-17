//! Arithmetic Operator Replacement (AOR).
//!
//! Replaces each arithmetic binary operator (+, -, *, /, %) with every other
//! arithmetic operator.  For the QF-LIA fragment only +/- replacements are
//! guaranteed to preserve linearity; the `preserves_qf_lia` flag reflects this.

use shared_types::ast::{ArithOp, Expression, Function, Statement};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::MutationSite;
use shared_types::MutSpecError;

use super::{as_stmts, collect_expr_sites, replace_in_statement, MutationOperatorTrait};

/// Concrete AOR operator.
pub struct ArithmeticOperatorReplacement {
    /// When true, only generate +↔− replacements (stays in QF-LIA).
    pub qf_lia_only: bool,
}

impl ArithmeticOperatorReplacement {
    pub fn new() -> Self {
        Self { qf_lia_only: false }
    }

    pub fn qf_lia_strict() -> Self {
        Self { qf_lia_only: true }
    }

    fn replacements_for(&self, op: ArithOp) -> Vec<ArithOp> {
        let all = [
            ArithOp::Add,
            ArithOp::Sub,
            ArithOp::Mul,
            ArithOp::Div,
            ArithOp::Mod,
        ];
        if self.qf_lia_only {
            // Only linear replacements: +↔−
            match op {
                ArithOp::Add => vec![ArithOp::Sub],
                ArithOp::Sub => vec![ArithOp::Add],
                _ => vec![], // Mul/Div/Mod have no linear replacement
            }
        } else {
            all.iter().copied().filter(|&r| r != op).collect()
        }
    }

    fn build_error_predicate(
        &self,
        from: ArithOp,
        to: ArithOp,
        left: &Expression,
        right: &Expression,
    ) -> Option<Formula> {
        // E(m)(x) characterises the inputs where mutant differs from original.
        // For a op b  →  a op' b  the error predicate is (a op b) ≠ (a op' b).
        let lhs = expr_to_term(left)?;
        let rhs = expr_to_term(right)?;
        let orig = arith_term(from, lhs.clone(), rhs.clone());
        let mutd = arith_term(to, lhs, rhs);
        Some(Formula::Atom(Predicate::ne(orig, mutd)))
    }
}

impl Default for ArithmeticOperatorReplacement {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperatorTrait for ArithmeticOperatorReplacement {
    fn name(&self) -> &str {
        "AOR"
    }

    fn description(&self) -> &str {
        "Arithmetic Operator Replacement – replaces each arithmetic binary operator with alternatives"
    }

    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite> {
        let sites = collect_expr_sites(&function.body);
        let mut result = Vec::new();
        let mut _site_counter = 0usize;

        for es in &sites {
            if let Expression::BinaryArith {
                op,
                lhs: _left,
                rhs: _right,
            } = &es.expression
            {
                let replacements = self.replacements_for(*op);
                for rep in replacements {
                    let _desc = format!("{} -> {}", op, rep);
                    let site = MutationSite {
                        location: shared_types::errors::SpanInfo::unknown(),
                        operator: shared_types::operators::MutationOperator::Aor,
                        original: format!("{}", op),
                        replacement: format!("{}", rep),
                        function_name: None,
                        expression_type: None,
                    };
                    result.push(site);
                    _site_counter += 1;
                }
            }
        }
        result
    }

    fn apply(&self, function: &Function, site: &MutationSite) -> Result<Function, MutSpecError> {
        // Parse the original/replacement to find from→to
        let from_op = str_to_arith(&site.original)
            .ok_or_else(|| MutSpecError::internal("cannot parse AOR site original"))?;
        let to_op = str_to_arith(&site.replacement)
            .ok_or_else(|| MutSpecError::internal("cannot parse AOR site replacement"))?;

        let expr_sites = collect_expr_sites(&function.body);

        // Find the matching arith-bin site
        for es in &expr_sites {
            if let Expression::BinaryArith { op, lhs, rhs } = &es.expression {
                if *op == from_op {
                    let mutated_expr = Expression::BinaryArith {
                        op: to_op,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    };
                    let new_body = apply_mutation_to_body(
                        as_stmts(&function.body),
                        es.stmt_flat_idx,
                        &es.expr_path,
                        &mutated_expr,
                    );
                    return Ok(Function {
                        name: function.name.clone(),
                        params: function.params.clone(),
                        return_type: function.return_type.clone(),
                        body: Statement::Block(new_body),
                        span: function.span.clone(),
                    });
                }
            }
        }

        Err(MutSpecError::internal(
            "AOR: no matching site found in function",
        ))
    }

    fn error_predicate(&self, site: &MutationSite) -> Option<Formula> {
        let from_op = str_to_arith(&site.original)?;
        let to_op = str_to_arith(&site.replacement)?;
        // Generic predicate: (a from b) != (a to b) – expressed symbolically.
        let a = Term::var("__lhs");
        let b = Term::var("__rhs");
        let orig = arith_term(from_op, a.clone(), b.clone());
        let mutd = arith_term(to_op, a, b);
        Some(Formula::Atom(Predicate::ne(orig, mutd)))
    }

    fn preserves_qf_lia(&self) -> bool {
        self.qf_lia_only
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn str_to_arith(s: &str) -> Option<ArithOp> {
    match s.trim() {
        "+" => Some(ArithOp::Add),
        "-" => Some(ArithOp::Sub),
        "*" => Some(ArithOp::Mul),
        "/" => Some(ArithOp::Div),
        "%" => Some(ArithOp::Mod),
        _ => None,
    }
}

fn arith_term(op: ArithOp, a: Term, b: Term) -> Term {
    match op {
        ArithOp::Add => Term::Add(Box::new(a), Box::new(b)),
        ArithOp::Sub => Term::Sub(Box::new(a), Box::new(b)),
        // For Mul/Div/Mod we approximate with Add (these are non-linear anyway)
        _ => Term::Add(Box::new(a), Box::new(b)),
    }
}

fn expr_to_term(expr: &Expression) -> Option<Term> {
    match expr {
        Expression::IntLiteral(n) => Some(Term::Const(*n)),
        Expression::Var(name) => Some(Term::var(name.as_str())),
        Expression::BinaryArith { op, lhs, rhs } => {
            let lt = expr_to_term(lhs)?;
            let rt = expr_to_term(rhs)?;
            Some(arith_term(*op, lt, rt))
        }
        Expression::UnaryArith(inner) => {
            let t = expr_to_term(inner)?;
            Some(Term::Neg(Box::new(t)))
        }
        _ => None,
    }
}

/// Apply a mutation to the function body, returning the new body.
fn apply_mutation_to_body(
    body: &[Statement],
    stmt_flat_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
) -> Vec<Statement> {
    let mut counter = 0usize;
    apply_mut_recursive(body, stmt_flat_idx, expr_path, replacement, &mut counter)
}

fn apply_mut_recursive(
    stmts: &[Statement],
    target_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
    counter: &mut usize,
) -> Vec<Statement> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        let cur = *counter;
        *counter += 1;
        if cur == target_idx {
            out.push(replace_in_statement(stmt, cur, expr_path, replacement));
            skip_stmts_inner(stmt, counter);
        } else {
            match stmt {
                Statement::IfElse {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let nt = apply_mut_recursive(
                        as_stmts(then_branch),
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    );
                    let ne = else_branch.as_ref().map(|eb| {
                        apply_mut_recursive(
                            as_stmts(eb),
                            target_idx,
                            expr_path,
                            replacement,
                            counter,
                        )
                    });
                    out.push(Statement::IfElse {
                        condition: condition.clone(),
                        then_branch: Box::new(Statement::Block(nt)),
                        else_branch: ne.map(|v| Box::new(Statement::Block(v))),
                        span: span.clone(),
                    });
                }
                Statement::Block(inner) => {
                    let ni =
                        apply_mut_recursive(inner, target_idx, expr_path, replacement, counter);
                    out.push(Statement::Block(ni));
                }
                Statement::Sequence(inner) => {
                    let ni =
                        apply_mut_recursive(inner, target_idx, expr_path, replacement, counter);
                    out.push(Statement::Sequence(ni));
                }
                _ => out.push(stmt.clone()),
            }
        }
    }
    out
}

fn skip_stmts_inner(stmt: &Statement, counter: &mut usize) {
    match stmt {
        Statement::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            skip_stmts(as_stmts(then_branch), counter);
            if let Some(eb) = else_branch {
                skip_stmts(as_stmts(eb), counter);
            }
        }
        Statement::Block(inner) | Statement::Sequence(inner) => skip_stmts(inner, counter),
        _ => {}
    }
}

fn skip_stmts(stmts: &[Statement], counter: &mut usize) {
    for stmt in stmts {
        *counter += 1;
        skip_stmts_inner(stmt, counter);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::types::*;

    fn make_add_func() -> Function {
        // f(x, y) { return x + y; }
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: ArithOp::Add,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::Var("y".into())),
            })),
        )
    }

    fn make_complex_func() -> Function {
        // f(x, y, z) { return (x + y) * z - 1; }
        let x_plus_y = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(Expression::Var("x".into())),
            rhs: Box::new(Expression::Var("y".into())),
        };
        let times_z = Expression::BinaryArith {
            op: ArithOp::Mul,
            lhs: Box::new(x_plus_y),
            rhs: Box::new(Expression::Var("z".into())),
        };
        let minus_one = Expression::BinaryArith {
            op: ArithOp::Sub,
            lhs: Box::new(times_z),
            rhs: Box::new(Expression::IntLiteral(1)),
        };
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
                Variable::param("z", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::ret(Some(minus_one)),
        )
    }

    #[test]
    fn test_aor_applicable_sites_simple() {
        let aor = ArithmeticOperatorReplacement::new();
        let func = make_add_func();
        let sites = aor.applicable_sites(&func);
        // + can be replaced by -, *, /, % → 4 sites
        assert_eq!(sites.len(), 4);
    }

    #[test]
    fn test_aor_qf_lia_only() {
        let aor = ArithmeticOperatorReplacement::qf_lia_strict();
        let func = make_add_func();
        let sites = aor.applicable_sites(&func);
        // Only + -> - is linear
        assert_eq!(sites.len(), 1);
    }

    #[test]
    fn test_aor_complex_function() {
        let aor = ArithmeticOperatorReplacement::new();
        let func = make_complex_func();
        let sites = aor.applicable_sites(&func);
        // 3 operators (+, *, -), each with 4 replacements → 12
        assert_eq!(sites.len(), 12);
    }

    #[test]
    fn test_aor_apply_simple() {
        let aor = ArithmeticOperatorReplacement::new();
        let func = make_add_func();
        let sites = aor.applicable_sites(&func);
        let sub_site = sites
            .iter()
            .find(|s| s.original == "+" && s.replacement == "-")
            .unwrap();
        let mutated = aor.apply(&func, sub_site).unwrap();
        // The return expression should now contain x - y
        let body_stmts = as_stmts(&mutated.body);
        if let Statement::Return {
            value: Some(Expression::BinaryArith { op, .. }),
            ..
        } = &body_stmts[0]
        {
            assert_eq!(*op, ArithOp::Sub);
        } else {
            panic!("expected BinaryArith in return");
        }
    }

    #[test]
    fn test_aor_error_predicate() {
        let aor = ArithmeticOperatorReplacement::new();
        let func = make_add_func();
        let sites = aor.applicable_sites(&func);
        let site = &sites[0];
        let pred = aor.error_predicate(site);
        assert!(pred.is_some());
    }

    #[test]
    fn test_aor_name_description() {
        let aor = ArithmeticOperatorReplacement::new();
        assert_eq!(aor.name(), "AOR");
        assert!(!aor.description().is_empty());
    }

    #[test]
    fn test_aor_preserves_qf_lia() {
        let strict = ArithmeticOperatorReplacement::qf_lia_strict();
        assert!(strict.preserves_qf_lia());
        let loose = ArithmeticOperatorReplacement::new();
        assert!(!loose.preserves_qf_lia());
    }

    #[test]
    fn test_aor_nested_expressions() {
        // f(a,b) { return (a + b) + (a - b); }
        let lhs = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(Expression::Var("a".into())),
            rhs: Box::new(Expression::Var("b".into())),
        };
        let rhs = Expression::BinaryArith {
            op: ArithOp::Sub,
            lhs: Box::new(Expression::Var("a".into())),
            rhs: Box::new(Expression::Var("b".into())),
        };
        let sum = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        let func = Function::new(
            "f",
            vec![
                Variable::param("a", QfLiaType::Int),
                Variable::param("b", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::ret(Some(sum)),
        );
        let aor = ArithmeticOperatorReplacement::new();
        let sites = aor.applicable_sites(&func);
        // 3 operators, each with 4 replacements → 12
        assert_eq!(sites.len(), 12);
    }

    #[test]
    fn test_aor_with_if_else() {
        let func = Function::new(
            "abs_diff",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::if_else(
                Expression::Relational {
                    op: shared_types::ast::RelOp::Lt,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                },
                Statement::ret(Some(Expression::BinaryArith {
                    op: ArithOp::Sub,
                    lhs: Box::new(Expression::Var("y".into())),
                    rhs: Box::new(Expression::Var("x".into())),
                })),
                Some(Statement::ret(Some(Expression::BinaryArith {
                    op: ArithOp::Sub,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                }))),
            ),
        );
        let aor = ArithmeticOperatorReplacement::new();
        let sites = aor.applicable_sites(&func);
        // 2 subtraction sites, each with 4 replacements → 8
        assert_eq!(sites.len(), 8);
    }

    #[test]
    fn test_str_to_arith() {
        assert_eq!(str_to_arith("+"), Some(ArithOp::Add));
        assert_eq!(str_to_arith("-"), Some(ArithOp::Sub));
        assert_eq!(str_to_arith("*"), Some(ArithOp::Mul));
        assert_eq!(str_to_arith("/"), Some(ArithOp::Div));
        assert_eq!(str_to_arith("%"), Some(ArithOp::Mod));
        assert_eq!(str_to_arith("?"), None);
    }

    #[test]
    fn test_parse_aor_description() {
        let from = str_to_arith("+").unwrap();
        let to = str_to_arith("-").unwrap();
        assert_eq!(from, ArithOp::Add);
        assert_eq!(to, ArithOp::Sub);
    }

    #[test]
    fn test_expr_to_term_literal() {
        let t = expr_to_term(&Expression::IntLiteral(42));
        assert_eq!(t, Some(Term::Const(42)));
    }

    #[test]
    fn test_expr_to_term_var() {
        let t = expr_to_term(&Expression::Var("x".into()));
        assert_eq!(t, Some(Term::var("x")));
    }

    #[test]
    fn test_expr_to_term_complex() {
        let e = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(Expression::Var("x".into())),
            rhs: Box::new(Expression::IntLiteral(1)),
        };
        let t = expr_to_term(&e);
        assert!(t.is_some());
    }

    #[test]
    fn test_no_sites_for_pure_relational() {
        let func = Function::new(
            "cmp",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::Relational {
                op: shared_types::ast::RelOp::Lt,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(0)),
            })),
        );
        let aor = ArithmeticOperatorReplacement::new();
        let sites = aor.applicable_sites(&func);
        assert!(
            sites.is_empty(),
            "no arith ops in a purely relational function"
        );
    }
}
