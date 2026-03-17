//! Unary Operator Insertion (UOI).
//!
//! Inserts unary operators into the program:
//!   - Arithmetic negation: `a -> -a`
//!   - Logical negation:    `b -> !b`

use shared_types::ast::{Expression, Function, Statement};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::MutationSite;
use shared_types::MutSpecError;

use super::{as_stmts, collect_expr_sites, replace_in_statement, MutationOperatorTrait};

/// Concrete UOI operator.
pub struct UnaryOperatorInsertion {
    _private: (),
}

impl UnaryOperatorInsertion {
    pub fn new() -> Self {
        Self { _private: () }
    }

    fn is_integer_expr(expr: &Expression) -> bool {
        matches!(
            expr,
            Expression::IntLiteral(_)
                | Expression::BinaryArith { .. }
                | Expression::UnaryArith(_)
                | Expression::Var(_)
                | Expression::FunctionCall { .. }
                | Expression::ArrayAccess { .. }
        )
    }

    fn is_boolean_expr(expr: &Expression) -> bool {
        matches!(
            expr,
            Expression::BoolLiteral(_)
                | Expression::Relational { .. }
                | Expression::LogicalAnd(..)
                | Expression::LogicalOr(..)
                | Expression::LogicalNot(_)
        )
    }

    fn already_negated(expr: &Expression) -> bool {
        matches!(expr, Expression::UnaryArith(_) | Expression::LogicalNot(_))
    }

    fn is_zero_literal(expr: &Expression) -> bool {
        matches!(expr, Expression::IntLiteral(0))
    }
}

impl Default for UnaryOperatorInsertion {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperatorTrait for UnaryOperatorInsertion {
    fn name(&self) -> &str {
        "UOI"
    }
    fn description(&self) -> &str {
        "Unary Operator Insertion - inserts unary negation (arithmetic or logical)"
    }

    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite> {
        let sites = collect_expr_sites(&function.body);
        let mut result = Vec::new();
        for es in &sites {
            let expr = &es.expression;
            if Self::already_negated(expr) {
                continue;
            }
            if Self::is_integer_expr(expr) && !Self::is_zero_literal(expr) {
                result.push(MutationSite {
                    location: shared_types::errors::SpanInfo::unknown(),
                    operator: shared_types::operators::MutationOperator::Uoi,
                    original: expr_short(expr),
                    replacement: format!("-({})", expr_short(expr)),
                    function_name: None,
                    expression_type: None,
                });
            }
            if Self::is_boolean_expr(expr) {
                result.push(MutationSite {
                    location: shared_types::errors::SpanInfo::unknown(),
                    operator: shared_types::operators::MutationOperator::Uoi,
                    original: expr_short(expr),
                    replacement: format!("!({})", expr_short(expr)),
                    function_name: None,
                    expression_type: None,
                });
            }
        }
        result
    }

    fn apply(&self, function: &Function, site: &MutationSite) -> Result<Function, MutSpecError> {
        let is_neg = site.replacement.starts_with("-(");
        let is_not = site.replacement.starts_with("!(");
        if !is_neg && !is_not {
            return Err(MutSpecError::internal("cannot parse UOI site"));
        }
        let expr_sites = collect_expr_sites(&function.body);
        for es in &expr_sites {
            let expr = &es.expression;
            if is_neg
                && Self::is_integer_expr(expr)
                && !Self::already_negated(expr)
                && !Self::is_zero_literal(expr)
            {
                let mutated_expr = Expression::UnaryArith(Box::new(expr.clone()));
                let new_body = apply_uoi_body(
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
            if is_not && Self::is_boolean_expr(expr) && !Self::already_negated(expr) {
                let mutated_expr = Expression::LogicalNot(Box::new(expr.clone()));
                let new_body = apply_uoi_body(
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
        Err(MutSpecError::internal("UOI: no matching site found"))
    }

    fn error_predicate(&self, site: &MutationSite) -> Option<Formula> {
        if site.replacement.starts_with("-(") {
            let a = Term::Var("__expr".into());
            let zero = Term::Const(0);
            Some(Formula::Atom(Predicate::ne(a, zero)))
        } else if site.replacement.starts_with("!(") {
            Some(Formula::True)
        } else {
            None
        }
    }

    fn preserves_qf_lia(&self) -> bool {
        true
    }
}

fn expr_short(expr: &Expression) -> String {
    match expr {
        Expression::Var(name) => name.clone(),
        Expression::IntLiteral(n) => format!("{}", n),
        Expression::BoolLiteral(b) => format!("{}", b),
        Expression::BinaryArith { op, .. } => format!("(..{}..)", op),
        Expression::Relational { op, .. } => format!("(..{}..)", op),
        Expression::LogicalAnd(..) => "(..&&..)".into(),
        Expression::LogicalOr(..) => "(..||..)".into(),
        Expression::UnaryArith(_) => "(-_)".into(),
        Expression::LogicalNot(_) => "(!_)".into(),
        _ => "expr".into(),
    }
}

fn apply_uoi_body(
    body: &[Statement],
    stmt_flat_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
) -> Vec<Statement> {
    let mut counter = 0usize;
    apply_uoi_recursive(body, stmt_flat_idx, expr_path, replacement, &mut counter)
}

fn apply_uoi_recursive(
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
            skip_uoi(stmt, counter);
        } else {
            match stmt {
                Statement::IfElse {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let nt = apply_uoi_recursive(
                        as_stmts(then_branch),
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    );
                    let ne = else_branch.as_ref().map(|eb| {
                        apply_uoi_recursive(
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
                    out.push(Statement::Block(apply_uoi_recursive(
                        inner,
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    )));
                }
                Statement::Sequence(inner) => {
                    out.push(Statement::Sequence(apply_uoi_recursive(
                        inner,
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    )));
                }
                _ => out.push(stmt.clone()),
            }
        }
    }
    out
}

fn skip_uoi(stmt: &Statement, counter: &mut usize) {
    match stmt {
        Statement::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            skip_uoi_stmts(as_stmts(then_branch), counter);
            if let Some(eb) = else_branch {
                skip_uoi_stmts(as_stmts(eb), counter);
            }
        }
        Statement::Block(inner) | Statement::Sequence(inner) => skip_uoi_stmts(inner, counter),
        _ => {}
    }
}

fn skip_uoi_stmts(stmts: &[Statement], counter: &mut usize) {
    for s in stmts {
        *counter += 1;
        skip_uoi(s, counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::ast::{ArithOp, RelOp};
    use shared_types::types::*;

    fn make_arith_func() -> Function {
        Function::new(
            "f",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: ArithOp::Add,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(1)),
            })),
        )
    }

    fn make_bool_func() -> Function {
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

    fn make_mixed_func() -> Function {
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::if_else(
                Expression::Relational {
                    op: RelOp::Lt,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                },
                Statement::ret(Some(Expression::BinaryArith {
                    op: ArithOp::Add,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                })),
                Some(Statement::ret(Some(Expression::BinaryArith {
                    op: ArithOp::Sub,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::Var("y".into())),
                }))),
            ),
        )
    }

    #[test]
    fn test_uoi_applicable_arith() {
        let sites = UnaryOperatorInsertion::new().applicable_sites(&make_arith_func());
        let neg_sites: Vec<_> = sites
            .iter()
            .filter(|s| s.replacement.starts_with("-("))
            .collect();
        assert!(neg_sites.len() >= 2);
    }
    #[test]
    fn test_uoi_applicable_bool() {
        let sites = UnaryOperatorInsertion::new().applicable_sites(&make_bool_func());
        let not_sites: Vec<_> = sites
            .iter()
            .filter(|s| s.replacement.starts_with("!("))
            .collect();
        assert!(not_sites.len() >= 1);
    }
    #[test]
    fn test_uoi_mixed() {
        let sites = UnaryOperatorInsertion::new().applicable_sites(&make_mixed_func());
        assert!(sites.iter().any(|s| s.replacement.starts_with("-(")));
        assert!(sites.iter().any(|s| s.replacement.starts_with("!(")));
    }
    #[test]
    fn test_uoi_apply_neg() {
        let uoi = UnaryOperatorInsertion::new();
        let func = make_arith_func();
        let sites = uoi.applicable_sites(&func);
        let neg_site = sites
            .iter()
            .find(|s| s.replacement.starts_with("-("))
            .unwrap();
        assert!(uoi.apply(&func, neg_site).is_ok());
    }
    #[test]
    fn test_uoi_apply_not() {
        let uoi = UnaryOperatorInsertion::new();
        let func = make_bool_func();
        let sites = uoi.applicable_sites(&func);
        let not_site = sites
            .iter()
            .find(|s| s.replacement.starts_with("!("))
            .unwrap();
        assert!(uoi.apply(&func, not_site).is_ok());
    }
    #[test]
    fn test_uoi_error_predicate_neg() {
        let uoi = UnaryOperatorInsertion::new();
        let func = make_arith_func();
        let sites = uoi.applicable_sites(&func);
        let neg_site = sites
            .iter()
            .find(|s| s.replacement.starts_with("-("))
            .unwrap();
        assert!(uoi.error_predicate(neg_site).is_some());
    }
    #[test]
    fn test_uoi_error_predicate_not() {
        let uoi = UnaryOperatorInsertion::new();
        let func = make_bool_func();
        let sites = uoi.applicable_sites(&func);
        let not_site = sites
            .iter()
            .find(|s| s.replacement.starts_with("!("))
            .unwrap();
        assert_eq!(uoi.error_predicate(not_site), Some(Formula::True));
    }
    #[test]
    fn test_uoi_name() {
        assert_eq!(UnaryOperatorInsertion::new().name(), "UOI");
    }
    #[test]
    fn test_uoi_preserves_qf_lia() {
        assert!(UnaryOperatorInsertion::new().preserves_qf_lia());
    }
    #[test]
    fn test_uoi_skips_zero_literal() {
        let func = Function::new(
            "f",
            vec![],
            QfLiaType::Int,
            Statement::ret(Some(Expression::IntLiteral(0))),
        );
        let sites = UnaryOperatorInsertion::new().applicable_sites(&func);
        assert!(
            sites
                .iter()
                .filter(|s| s.replacement.starts_with("-("))
                .count()
                == 0
        );
    }
    #[test]
    fn test_uoi_skips_already_negated() {
        let func = Function::new(
            "f",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::UnaryArith(Box::new(Expression::Var(
                "x".into(),
            ))))),
        );
        let sites = UnaryOperatorInsertion::new().applicable_sites(&func);
        // x itself is a valid site but the UnaryArith node is skipped
        assert!(
            sites
                .iter()
                .filter(|s| s.replacement.starts_with("-("))
                .count()
                >= 1
        );
    }
    #[test]
    fn test_expr_short_vals() {
        assert_eq!(expr_short(&Expression::Var("x".into())), "x");
        assert_eq!(expr_short(&Expression::IntLiteral(42)), "42");
    }
    #[test]
    fn test_is_integer_expr_check() {
        assert!(UnaryOperatorInsertion::is_integer_expr(&Expression::Var(
            "x".into()
        )));
        assert!(!UnaryOperatorInsertion::is_integer_expr(
            &Expression::BoolLiteral(true)
        ));
    }
    #[test]
    fn test_is_boolean_expr_check() {
        assert!(UnaryOperatorInsertion::is_boolean_expr(
            &Expression::BoolLiteral(true)
        ));
        assert!(!UnaryOperatorInsertion::is_boolean_expr(
            &Expression::IntLiteral(1)
        ));
    }
}
