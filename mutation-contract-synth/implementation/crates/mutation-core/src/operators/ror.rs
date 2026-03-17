//! Relational Operator Replacement (ROR).
//!
//! Replaces each relational operator (<, <=, ==, !=, >=, >) with every other
//! relational operator.  For each replacement the error predicate captures the
//! boundary condition that distinguishes the mutant from the original.

use shared_types::ast::{Expression, Function, RelOp, Statement};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::MutationSite;
use shared_types::MutSpecError;

use super::{as_stmts, collect_expr_sites, replace_in_statement, MutationOperatorTrait};

/// Concrete ROR operator.
pub struct RelationalOperatorReplacement {
    _private: (),
}

impl RelationalOperatorReplacement {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// All five alternative operators for a given relational operator.
    fn replacements_for(op: RelOp) -> Vec<RelOp> {
        let all = [
            RelOp::Lt,
            RelOp::Le,
            RelOp::Eq,
            RelOp::Ne,
            RelOp::Gt,
            RelOp::Ge,
        ];
        all.iter().copied().filter(|&r| r != op).collect()
    }

    /// Compute a symbolic error predicate for `from → to` replacement.
    ///
    /// The error predicate characterises inputs where the truth-value of the
    /// comparison differs between original and mutant.
    ///
    /// Examples:
    ///  * `< → <=`  ⟹  E(m)(x) = (a == b)          (boundary: equality)
    ///  * `< → >`   ⟹  E(m)(x) = (a != b)           (always differs when not equal)
    ///  * `== → !=`  ⟹  E(m)(x) = true               (always differs)
    ///  * `< → ==`   ⟹  E(m)(x) = (a < b) XOR (a == b) = (a <= b)
    fn compute_error_predicate(from: RelOp, to: RelOp, left: &Term, right: &Term) -> Formula {
        // The error predicate is the XOR of the two conditions:
        //   E = (orig XOR mutant) = (orig AND NOT mutant) OR (NOT orig AND mutant)
        let orig = rel_formula(from, left, right);
        let mutd = rel_formula(to, left, right);
        // XOR = (A ∧ ¬B) ∨ (¬A ∧ B)
        let a_and_not_b = Formula::And(vec![orig.clone(), Formula::not(mutd.clone())]);
        let not_a_and_b = Formula::And(vec![Formula::not(orig), mutd]);
        Formula::Or(vec![a_and_not_b, not_a_and_b])
    }
}

impl Default for RelationalOperatorReplacement {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperatorTrait for RelationalOperatorReplacement {
    fn name(&self) -> &str {
        "ROR"
    }

    fn description(&self) -> &str {
        "Relational Operator Replacement – replaces each relational operator with alternatives"
    }

    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite> {
        let sites = collect_expr_sites(&function.body);
        let mut result = Vec::new();

        for es in &sites {
            if let Expression::Relational { op, .. } = &es.expression {
                let replacements = Self::replacements_for(*op);
                for rep in replacements {
                    let site = MutationSite {
                        location: shared_types::errors::SpanInfo::unknown(),
                        operator: shared_types::operators::MutationOperator::Ror,
                        original: format!("{}", op),
                        replacement: format!("{}", rep),
                        function_name: None,
                        expression_type: None,
                    };
                    result.push(site);
                }
            }
        }
        result
    }

    fn apply(&self, function: &Function, site: &MutationSite) -> Result<Function, MutSpecError> {
        let from_op = str_to_relop(&site.original)
            .ok_or_else(|| MutSpecError::internal("cannot parse ROR site original"))?;
        let to_op = str_to_relop(&site.replacement)
            .ok_or_else(|| MutSpecError::internal("cannot parse ROR site replacement"))?;

        let expr_sites = collect_expr_sites(&function.body);

        for es in &expr_sites {
            if let Expression::Relational { op, lhs, rhs } = &es.expression {
                if *op == from_op {
                    let mutated_expr = Expression::Relational {
                        op: to_op,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    };
                    let new_body = apply_mutation_body(
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
            "ROR: no matching site found in function",
        ))
    }

    fn error_predicate(&self, site: &MutationSite) -> Option<Formula> {
        let from_op = str_to_relop(&site.original)?;
        let to_op = str_to_relop(&site.replacement)?;
        let a = Term::var("__lhs");
        let b = Term::var("__rhs");
        Some(Self::compute_error_predicate(from_op, to_op, &a, &b))
    }

    fn preserves_qf_lia(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn rel_formula(op: RelOp, left: &Term, right: &Term) -> Formula {
    let pred = match op {
        RelOp::Lt => Predicate::lt(left.clone(), right.clone()),
        RelOp::Le => Predicate::le(left.clone(), right.clone()),
        RelOp::Eq => Predicate::eq(left.clone(), right.clone()),
        RelOp::Ne => Predicate::ne(left.clone(), right.clone()),
        RelOp::Gt => Predicate::gt(left.clone(), right.clone()),
        RelOp::Ge => Predicate::ge(left.clone(), right.clone()),
    };
    Formula::Atom(pred)
}

fn str_to_relop(s: &str) -> Option<RelOp> {
    match s.trim() {
        "<" => Some(RelOp::Lt),
        "<=" => Some(RelOp::Le),
        "==" => Some(RelOp::Eq),
        "!=" => Some(RelOp::Ne),
        ">=" => Some(RelOp::Ge),
        ">" => Some(RelOp::Gt),
        _ => None,
    }
}

fn apply_mutation_body(
    body: &[Statement],
    stmt_flat_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
) -> Vec<Statement> {
    let mut counter = 0usize;
    apply_ror_recursive(body, stmt_flat_idx, expr_path, replacement, &mut counter)
}

fn apply_ror_recursive(
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
            skip_ror_inner(stmt, counter);
        } else {
            match stmt {
                Statement::IfElse {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let nt = apply_ror_recursive(
                        as_stmts(then_branch),
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    );
                    let ne = else_branch.as_ref().map(|eb| {
                        apply_ror_recursive(
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
                        apply_ror_recursive(inner, target_idx, expr_path, replacement, counter);
                    out.push(Statement::Block(ni));
                }
                Statement::Sequence(inner) => {
                    let ni =
                        apply_ror_recursive(inner, target_idx, expr_path, replacement, counter);
                    out.push(Statement::Sequence(ni));
                }
                _ => out.push(stmt.clone()),
            }
        }
    }
    out
}

fn skip_ror_inner(stmt: &Statement, counter: &mut usize) {
    match stmt {
        Statement::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            skip_ror(as_stmts(then_branch), counter);
            if let Some(eb) = else_branch {
                skip_ror(as_stmts(eb), counter);
            }
        }
        Statement::Block(inner) | Statement::Sequence(inner) => skip_ror(inner, counter),
        _ => {}
    }
}

fn skip_ror(stmts: &[Statement], counter: &mut usize) {
    for s in stmts {
        *counter += 1;
        skip_ror_inner(s, counter);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::ast::ArithOp;
    use shared_types::types::*;

    fn make_lt_func() -> Function {
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::Relational {
                op: RelOp::Lt,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::Var("y".into())),
            })),
        )
    }

    fn make_eq_func() -> Function {
        Function::new(
            "eq",
            vec![
                Variable::param("a", QfLiaType::Int),
                Variable::param("b", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::Relational {
                op: RelOp::Eq,
                lhs: Box::new(Expression::Var("a".into())),
                rhs: Box::new(Expression::Var("b".into())),
            })),
        )
    }

    fn make_complex_cond_func() -> Function {
        // f(x) { if (x >= 0) { return x; } else { return -x; } }
        Function::new(
            "abs",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::if_else(
                Expression::Relational {
                    op: RelOp::Ge,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                },
                Statement::ret(Some(Expression::Var("x".into()))),
                Some(Statement::ret(Some(Expression::UnaryArith(Box::new(
                    Expression::Var("x".into()),
                ))))),
            ),
        )
    }

    #[test]
    fn test_ror_applicable_sites_lt() {
        let ror = RelationalOperatorReplacement::new();
        let func = make_lt_func();
        let sites = ror.applicable_sites(&func);
        assert_eq!(sites.len(), 5);
    }

    #[test]
    fn test_ror_applicable_sites_eq() {
        let ror = RelationalOperatorReplacement::new();
        let func = make_eq_func();
        let sites = ror.applicable_sites(&func);
        assert_eq!(sites.len(), 5);
    }

    #[test]
    fn test_ror_apply() {
        let ror = RelationalOperatorReplacement::new();
        let func = make_lt_func();
        let sites = ror.applicable_sites(&func);
        let le_site = sites
            .iter()
            .find(|s| s.original == "<" && s.replacement == "<=")
            .unwrap();
        let mutated = ror.apply(&func, le_site).unwrap();
        let body_stmts = as_stmts(&mutated.body);
        if let Statement::Return {
            value: Some(Expression::Relational { op, .. }),
            ..
        } = &body_stmts[0]
        {
            assert_eq!(*op, RelOp::Le);
        } else {
            panic!("expected Relational");
        }
    }

    #[test]
    fn test_ror_complex_cond() {
        let ror = RelationalOperatorReplacement::new();
        let func = make_complex_cond_func();
        let sites = ror.applicable_sites(&func);
        assert_eq!(sites.len(), 5);
    }

    #[test]
    fn test_ror_error_predicate() {
        let ror = RelationalOperatorReplacement::new();
        let func = make_lt_func();
        let sites = ror.applicable_sites(&func);
        for site in &sites {
            let pred = ror.error_predicate(site);
            assert!(pred.is_some());
        }
    }

    #[test]
    fn test_ror_name_description() {
        let ror = RelationalOperatorReplacement::new();
        assert_eq!(ror.name(), "ROR");
        assert!(!ror.description().is_empty());
    }

    #[test]
    fn test_ror_preserves_qf_lia() {
        let ror = RelationalOperatorReplacement::new();
        assert!(ror.preserves_qf_lia());
    }

    #[test]
    fn test_ror_all_replacements() {
        for op in &[
            RelOp::Lt,
            RelOp::Le,
            RelOp::Eq,
            RelOp::Ne,
            RelOp::Ge,
            RelOp::Gt,
        ] {
            let reps = RelationalOperatorReplacement::replacements_for(*op);
            assert_eq!(reps.len(), 5);
            assert!(!reps.contains(op));
        }
    }

    #[test]
    fn test_str_to_relop() {
        assert_eq!(str_to_relop("<"), Some(RelOp::Lt));
        assert_eq!(str_to_relop("<="), Some(RelOp::Le));
        assert_eq!(str_to_relop("=="), Some(RelOp::Eq));
        assert_eq!(str_to_relop("!="), Some(RelOp::Ne));
        assert_eq!(str_to_relop(">="), Some(RelOp::Ge));
        assert_eq!(str_to_relop(">"), Some(RelOp::Gt));
        assert_eq!(str_to_relop("??"), None);
    }

    #[test]
    fn test_ror_no_sites_for_arith_only() {
        let func = Function::new(
            "f",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: ArithOp::Add,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(1)),
            })),
        );
        let ror = RelationalOperatorReplacement::new();
        let sites = ror.applicable_sites(&func);
        assert!(sites.is_empty());
    }

    #[test]
    fn test_ror_nested_relational() {
        // f(x, y) { return (x < y) && (x != 0); }
        let func = Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::LogicalAnd(
                Box::new(Expression::Relational {
                    op: RelOp::Lt,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                }),
                Box::new(Expression::Relational {
                    op: RelOp::Ne,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
            ))),
        );
        let ror = RelationalOperatorReplacement::new();
        let sites = ror.applicable_sites(&func);
        // 2 relational sites × 5 replacements = 10
        assert_eq!(sites.len(), 10);
    }

    #[test]
    fn test_error_predicate_lt_to_le() {
        let a = Term::var("a");
        let b = Term::var("b");
        let pred =
            RelationalOperatorReplacement::compute_error_predicate(RelOp::Lt, RelOp::Le, &a, &b);
        assert!(!pred.is_true());
        assert!(!pred.is_false());
    }

    #[test]
    fn test_error_predicate_eq_to_ne() {
        let a = Term::var("a");
        let b = Term::var("b");
        let pred =
            RelationalOperatorReplacement::compute_error_predicate(RelOp::Eq, RelOp::Ne, &a, &b);
        assert!(!pred.is_false());
    }
}
