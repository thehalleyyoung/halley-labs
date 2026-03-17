//! Logical Connector Replacement (LCR).
//!
//! Replaces logical connectives: AND ↔ OR.
//! The error predicate captures when the truth-value of the connective changes.

use shared_types::ast::{Expression, Function, LogicOp, Statement};
use shared_types::formula::{Formula, Predicate, Term};
use shared_types::operators::MutationSite;
use shared_types::MutSpecError;

use super::{as_stmts, collect_expr_sites, replace_in_statement, MutationOperatorTrait};

/// Concrete LCR operator.
pub struct LogicalConnectorReplacement {
    _private: (),
}

impl LogicalConnectorReplacement {
    pub fn new() -> Self {
        Self { _private: () }
    }

    fn replacements_for(op: LogicOp) -> Vec<LogicOp> {
        match op {
            LogicOp::And => vec![LogicOp::Or],
            LogicOp::Or => vec![LogicOp::And],
        }
    }

    fn compute_error_predicate(from: LogicOp, to: LogicOp) -> Formula {
        let p = Formula::atom(Predicate::ne(Term::Var("__p".into()), Term::Const(0)));
        let q = Formula::atom(Predicate::ne(Term::Var("__q".into()), Term::Const(0)));
        let orig = logic_formula(from, &p, &q);
        let mutd = logic_formula(to, &p, &q);
        let a_and_not_b = Formula::And(vec![orig.clone(), Formula::not(mutd.clone())]);
        let not_a_and_b = Formula::And(vec![Formula::not(orig), mutd]);
        Formula::Or(vec![a_and_not_b, not_a_and_b])
    }
}

impl Default for LogicalConnectorReplacement {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperatorTrait for LogicalConnectorReplacement {
    fn name(&self) -> &str {
        "LCR"
    }
    fn description(&self) -> &str {
        "Logical Connector Replacement - replaces logical connectives (AND<->OR)"
    }

    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite> {
        let sites = collect_expr_sites(&function.body);
        let mut result = Vec::new();
        for es in &sites {
            let op = match &es.expression {
                Expression::LogicalAnd(..) => Some(LogicOp::And),
                Expression::LogicalOr(..) => Some(LogicOp::Or),
                _ => None,
            };
            if let Some(op) = op {
                for rep in Self::replacements_for(op) {
                    result.push(MutationSite {
                        location: shared_types::errors::SpanInfo::unknown(),
                        operator: shared_types::operators::MutationOperator::Lcr,
                        original: format!("{}", op),
                        replacement: format!("{}", rep),
                        function_name: None,
                        expression_type: None,
                    });
                }
            }
        }
        result
    }

    fn apply(&self, function: &Function, site: &MutationSite) -> Result<Function, MutSpecError> {
        let (from_op, to_op) =
            parse_lcr_site(site).ok_or_else(|| MutSpecError::internal("cannot parse LCR site"))?;
        let expr_sites = collect_expr_sites(&function.body);
        for es in &expr_sites {
            let matched = match (&es.expression, from_op) {
                (Expression::LogicalAnd(left, right), LogicOp::And) => Some((left, right)),
                (Expression::LogicalOr(left, right), LogicOp::Or) => Some((left, right)),
                _ => None,
            };
            if let Some((left, right)) = matched {
                let mutated_expr = match to_op {
                    LogicOp::And => Expression::LogicalAnd(left.clone(), right.clone()),
                    LogicOp::Or => Expression::LogicalOr(left.clone(), right.clone()),
                };
                let new_body = apply_lcr_body(
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
        Err(MutSpecError::internal("LCR: no matching site found"))
    }

    fn error_predicate(&self, site: &MutationSite) -> Option<Formula> {
        let (from_op, to_op) = parse_lcr_site(site)?;
        Some(Self::compute_error_predicate(from_op, to_op))
    }

    fn preserves_qf_lia(&self) -> bool {
        true
    }
}

fn logic_formula(op: LogicOp, p: &Formula, q: &Formula) -> Formula {
    match op {
        LogicOp::And => Formula::and(vec![p.clone(), q.clone()]),
        LogicOp::Or => Formula::or(vec![p.clone(), q.clone()]),
    }
}

fn parse_lcr_site(site: &MutationSite) -> Option<(LogicOp, LogicOp)> {
    let from = str_to_logicop(&site.original)?;
    let to = str_to_logicop(&site.replacement)?;
    Some((from, to))
}

fn str_to_logicop(s: &str) -> Option<LogicOp> {
    match s.trim() {
        "&&" | "And" => Some(LogicOp::And),
        "||" | "Or" => Some(LogicOp::Or),
        _ => None,
    }
}

fn apply_lcr_body(
    body: &[Statement],
    stmt_flat_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
) -> Vec<Statement> {
    let mut counter = 0usize;
    apply_lcr_recursive(body, stmt_flat_idx, expr_path, replacement, &mut counter)
}

fn apply_lcr_recursive(
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
            skip_lcr(stmt, counter);
        } else {
            match stmt {
                Statement::IfElse {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let nt = apply_lcr_recursive(
                        as_stmts(then_branch),
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    );
                    let ne = else_branch.as_ref().map(|eb| {
                        apply_lcr_recursive(
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
                    out.push(Statement::Block(apply_lcr_recursive(
                        inner,
                        target_idx,
                        expr_path,
                        replacement,
                        counter,
                    )));
                }
                Statement::Sequence(inner) => {
                    out.push(Statement::Sequence(apply_lcr_recursive(
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

fn skip_lcr(stmt: &Statement, counter: &mut usize) {
    match stmt {
        Statement::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            skip_lcr_stmts(as_stmts(then_branch), counter);
            if let Some(eb) = else_branch {
                skip_lcr_stmts(as_stmts(eb), counter);
            }
        }
        Statement::Block(inner) | Statement::Sequence(inner) => skip_lcr_stmts(inner, counter),
        _ => {}
    }
}

fn skip_lcr_stmts(stmts: &[Statement], counter: &mut usize) {
    for s in stmts {
        *counter += 1;
        skip_lcr(s, counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::ast::RelOp;
    use shared_types::types::*;

    fn make_and_func() -> Function {
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::LogicalAnd(
                Box::new(Expression::Relational {
                    op: RelOp::Gt,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
                Box::new(Expression::Relational {
                    op: RelOp::Gt,
                    lhs: Box::new(Expression::Var("y".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
            ))),
        )
    }

    fn make_or_func() -> Function {
        Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::LogicalOr(
                Box::new(Expression::Relational {
                    op: RelOp::Lt,
                    lhs: Box::new(Expression::Var("x".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
                Box::new(Expression::Relational {
                    op: RelOp::Lt,
                    lhs: Box::new(Expression::Var("y".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
            ))),
        )
    }

    fn make_nested_logic_func() -> Function {
        let inner_or = Expression::LogicalOr(
            Box::new(Expression::Relational {
                op: RelOp::Gt,
                lhs: Box::new(Expression::Var("b".into())),
                rhs: Box::new(Expression::IntLiteral(0)),
            }),
            Box::new(Expression::Relational {
                op: RelOp::Gt,
                lhs: Box::new(Expression::Var("c".into())),
                rhs: Box::new(Expression::IntLiteral(0)),
            }),
        );
        Function::new(
            "f",
            vec![
                Variable::param("a", QfLiaType::Int),
                Variable::param("b", QfLiaType::Int),
                Variable::param("c", QfLiaType::Int),
            ],
            QfLiaType::Boolean,
            Statement::ret(Some(Expression::LogicalAnd(
                Box::new(Expression::Relational {
                    op: RelOp::Gt,
                    lhs: Box::new(Expression::Var("a".into())),
                    rhs: Box::new(Expression::IntLiteral(0)),
                }),
                Box::new(inner_or),
            ))),
        )
    }

    #[test]
    fn test_lcr_applicable_and() {
        assert_eq!(
            LogicalConnectorReplacement::new()
                .applicable_sites(&make_and_func())
                .len(),
            1
        );
    }
    #[test]
    fn test_lcr_applicable_or() {
        assert_eq!(
            LogicalConnectorReplacement::new()
                .applicable_sites(&make_or_func())
                .len(),
            1
        );
    }
    #[test]
    fn test_lcr_nested() {
        assert_eq!(
            LogicalConnectorReplacement::new()
                .applicable_sites(&make_nested_logic_func())
                .len(),
            2
        );
    }
    #[test]
    fn test_lcr_apply() {
        let lcr = LogicalConnectorReplacement::new();
        let f = make_and_func();
        let s = lcr.applicable_sites(&f);
        assert!(lcr.apply(&f, &s[0]).is_ok());
    }
    #[test]
    fn test_lcr_error_predicate() {
        let lcr = LogicalConnectorReplacement::new();
        let f = make_and_func();
        let s = lcr.applicable_sites(&f);
        assert!(lcr.error_predicate(&s[0]).is_some());
    }
    #[test]
    fn test_lcr_name() {
        assert_eq!(LogicalConnectorReplacement::new().name(), "LCR");
    }
    #[test]
    fn test_lcr_preserves_qf_lia() {
        assert!(LogicalConnectorReplacement::new().preserves_qf_lia());
    }
    #[test]
    fn test_str_to_logicop_vals() {
        assert_eq!(str_to_logicop("&&"), Some(LogicOp::And));
        assert_eq!(str_to_logicop("||"), Some(LogicOp::Or));
        assert_eq!(str_to_logicop("???"), None);
    }
    #[test]
    fn test_lcr_no_sites_for_arith() {
        let func = Function::new(
            "f",
            vec![Variable::param("x", QfLiaType::Int)],
            QfLiaType::Int,
            Statement::ret(Some(Expression::BinaryArith {
                op: shared_types::ast::ArithOp::Add,
                lhs: Box::new(Expression::Var("x".into())),
                rhs: Box::new(Expression::IntLiteral(1)),
            })),
        );
        assert!(LogicalConnectorReplacement::new()
            .applicable_sites(&func)
            .is_empty());
    }
    #[test]
    fn test_lcr_in_if_condition() {
        let func = Function::new(
            "f",
            vec![
                Variable::param("x", QfLiaType::Int),
                Variable::param("y", QfLiaType::Int),
            ],
            QfLiaType::Int,
            Statement::if_else(
                Expression::LogicalAnd(
                    Box::new(Expression::Relational {
                        op: RelOp::Gt,
                        lhs: Box::new(Expression::Var("x".into())),
                        rhs: Box::new(Expression::IntLiteral(0)),
                    }),
                    Box::new(Expression::Relational {
                        op: RelOp::Gt,
                        lhs: Box::new(Expression::Var("y".into())),
                        rhs: Box::new(Expression::IntLiteral(0)),
                    }),
                ),
                Statement::ret(Some(Expression::IntLiteral(1))),
                Some(Statement::ret(Some(Expression::IntLiteral(0)))),
            ),
        );
        assert_eq!(
            LogicalConnectorReplacement::new()
                .applicable_sites(&func)
                .len(),
            1
        );
    }
}
