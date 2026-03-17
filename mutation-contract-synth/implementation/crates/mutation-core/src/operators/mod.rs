//! Mutation operator trait, registry, and factory.

pub mod aor;
pub mod lcr;
pub mod ror;
pub mod uoi;

use shared_types::ast::{Expression, Function, Statement};
use shared_types::operators::MutationSite;

/// Extract a `&[Statement]` view from a `Statement` (Block/Sequence → inner,
/// otherwise a one-element slice).
pub(crate) fn as_stmts(stmt: &Statement) -> &[Statement] {
    match stmt {
        Statement::Block(s) | Statement::Sequence(s) => s.as_slice(),
        other => std::slice::from_ref(other),
    }
}

/// Trait that every concrete mutation operator must implement.
pub trait MutationOperatorTrait: Send + Sync {
    /// Human-readable name, e.g. "AOR".
    fn name(&self) -> &str;

    /// One-sentence description.
    fn description(&self) -> &str;

    /// Scan `function` and return every site where this operator *could* apply.
    fn applicable_sites(&self, function: &Function) -> Vec<MutationSite>;

    /// Apply the mutation at `site`, returning a **new** function with the
    /// replacement made.  Returns `Err` if the site is invalid.
    fn apply(
        &self,
        function: &Function,
        site: &MutationSite,
    ) -> Result<Function, shared_types::MutSpecError>;

    /// Compute the *error predicate* formula for this mutation.
    /// The error predicate characterises inputs that distinguish the mutant.
    fn error_predicate(&self, site: &MutationSite) -> Option<shared_types::formula::Formula> {
        let _ = site;
        None
    }

    /// Whether this operator is guaranteed to stay within QF-LIA.
    fn preserves_qf_lia(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// OperatorRegistry
// ---------------------------------------------------------------------------

/// A collection of registered mutation operators, keyed by name.
pub struct OperatorRegistry {
    operators: Vec<Box<dyn MutationOperatorTrait>>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    pub fn register(&mut self, op: Box<dyn MutationOperatorTrait>) {
        self.operators.push(op);
    }

    pub fn get(&self, name: &str) -> Option<&dyn MutationOperatorTrait> {
        self.operators
            .iter()
            .find(|o| o.name() == name)
            .map(|b| b.as_ref())
    }

    pub fn all(&self) -> &[Box<dyn MutationOperatorTrait>] {
        &self.operators
    }

    pub fn names(&self) -> Vec<&str> {
        self.operators.iter().map(|o| o.name()).collect()
    }

    pub fn len(&self) -> usize {
        self.operators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Find all applicable sites across every registered operator.
    pub fn all_sites(&self, function: &Function) -> Vec<(usize, MutationSite)> {
        let mut result = Vec::new();
        for (idx, op) in self.operators.iter().enumerate() {
            for site in op.applicable_sites(function) {
                result.push((idx, site));
            }
        }
        result
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a registry pre-loaded with all four standard operators.
pub fn create_standard_operators() -> OperatorRegistry {
    let mut reg = OperatorRegistry::new();
    reg.register(Box::new(aor::ArithmeticOperatorReplacement::new()));
    reg.register(Box::new(ror::RelationalOperatorReplacement::new()));
    reg.register(Box::new(lcr::LogicalConnectorReplacement::new()));
    reg.register(Box::new(uoi::UnaryOperatorInsertion::new()));
    reg
}

// ---------------------------------------------------------------------------
// Helpers shared across operator implementations
// ---------------------------------------------------------------------------

/// Node-path based expression replacement.  Given a root expression and a
/// `path` of child indices, replace the sub-expression at that path with
/// `replacement`.
pub(crate) fn replace_expr_at_path(
    root: &Expression,
    path: &[usize],
    replacement: &Expression,
) -> Expression {
    if path.is_empty() {
        return replacement.clone();
    }
    let idx = path[0];
    let rest = &path[1..];
    match root {
        Expression::BinaryArith { op, lhs, rhs } => match idx {
            0 => Expression::BinaryArith {
                op: *op,
                lhs: Box::new(replace_expr_at_path(lhs, rest, replacement)),
                rhs: rhs.clone(),
            },
            1 => Expression::BinaryArith {
                op: *op,
                lhs: lhs.clone(),
                rhs: Box::new(replace_expr_at_path(rhs, rest, replacement)),
            },
            _ => root.clone(),
        },
        Expression::Relational { op, lhs, rhs } => match idx {
            0 => Expression::Relational {
                op: *op,
                lhs: Box::new(replace_expr_at_path(lhs, rest, replacement)),
                rhs: rhs.clone(),
            },
            1 => Expression::Relational {
                op: *op,
                lhs: lhs.clone(),
                rhs: Box::new(replace_expr_at_path(rhs, rest, replacement)),
            },
            _ => root.clone(),
        },
        Expression::LogicalAnd(left, right) => match idx {
            0 => Expression::LogicalAnd(
                Box::new(replace_expr_at_path(left, rest, replacement)),
                right.clone(),
            ),
            1 => Expression::LogicalAnd(
                left.clone(),
                Box::new(replace_expr_at_path(right, rest, replacement)),
            ),
            _ => root.clone(),
        },
        Expression::LogicalOr(left, right) => match idx {
            0 => Expression::LogicalOr(
                Box::new(replace_expr_at_path(left, rest, replacement)),
                right.clone(),
            ),
            1 => Expression::LogicalOr(
                left.clone(),
                Box::new(replace_expr_at_path(right, rest, replacement)),
            ),
            _ => root.clone(),
        },
        Expression::UnaryArith(inner) if idx == 0 => {
            Expression::UnaryArith(Box::new(replace_expr_at_path(inner, rest, replacement)))
        }
        Expression::LogicalNot(inner) if idx == 0 => {
            Expression::LogicalNot(Box::new(replace_expr_at_path(inner, rest, replacement)))
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => match idx {
            0 => Expression::Conditional {
                condition: Box::new(replace_expr_at_path(condition, rest, replacement)),
                then_expr: then_expr.clone(),
                else_expr: else_expr.clone(),
            },
            1 => Expression::Conditional {
                condition: condition.clone(),
                then_expr: Box::new(replace_expr_at_path(then_expr, rest, replacement)),
                else_expr: else_expr.clone(),
            },
            2 => Expression::Conditional {
                condition: condition.clone(),
                then_expr: then_expr.clone(),
                else_expr: Box::new(replace_expr_at_path(else_expr, rest, replacement)),
            },
            _ => root.clone(),
        },
        Expression::FunctionCall { name, args } => {
            if idx < args.len() {
                let mut new_args = args.clone();
                new_args[idx] = replace_expr_at_path(&args[idx], rest, replacement);
                Expression::FunctionCall {
                    name: name.clone(),
                    args: new_args,
                }
            } else {
                root.clone()
            }
        }
        Expression::ArrayAccess { array, index } => match idx {
            0 => Expression::ArrayAccess {
                array: Box::new(replace_expr_at_path(array, rest, replacement)),
                index: index.clone(),
            },
            1 => Expression::ArrayAccess {
                array: array.clone(),
                index: Box::new(replace_expr_at_path(index, rest, replacement)),
            },
            _ => root.clone(),
        },
        _ => root.clone(),
    }
}

/// Apply an expression-level mutation to a statement, returning the new
/// statement with the replacement at the given statement-index + path.
pub(crate) fn replace_in_statement(
    stmt: &shared_types::ast::Statement,
    stmt_idx: usize,
    expr_path: &[usize],
    replacement: &Expression,
) -> shared_types::ast::Statement {
    let _ = stmt_idx;
    match stmt {
        shared_types::ast::Statement::Assign {
            target,
            value,
            span,
        } => shared_types::ast::Statement::Assign {
            target: target.clone(),
            value: replace_expr_at_path(value, expr_path, replacement),
            span: span.clone(),
        },
        shared_types::ast::Statement::Return {
            value: Some(expr),
            span,
        } => shared_types::ast::Statement::Return {
            value: Some(replace_expr_at_path(expr, expr_path, replacement)),
            span: span.clone(),
        },
        shared_types::ast::Statement::Assert {
            condition,
            message,
            span,
        } => shared_types::ast::Statement::Assert {
            condition: replace_expr_at_path(condition, expr_path, replacement),
            message: message.clone(),
            span: span.clone(),
        },
        shared_types::ast::Statement::VarDecl {
            var,
            init: Some(expr),
            span,
        } => shared_types::ast::Statement::VarDecl {
            var: var.clone(),
            init: Some(replace_expr_at_path(expr, expr_path, replacement)),
            span: span.clone(),
        },
        shared_types::ast::Statement::IfElse {
            condition,
            then_branch,
            else_branch,
            span,
        } => shared_types::ast::Statement::IfElse {
            condition: replace_expr_at_path(condition, expr_path, replacement),
            then_branch: then_branch.clone(),
            else_branch: else_branch.clone(),
            span: span.clone(),
        },
        other => other.clone(),
    }
}

/// Replace statement at a given index in the body of a function.
#[allow(dead_code)]
pub(crate) fn replace_statement_in_body(
    body: &[shared_types::ast::Statement],
    flat_idx: usize,
    new_stmt: &shared_types::ast::Statement,
) -> Vec<shared_types::ast::Statement> {
    let mut counter = 0usize;
    replace_stmt_recursive(body, flat_idx, new_stmt, &mut counter)
}

fn replace_stmt_recursive(
    stmts: &[shared_types::ast::Statement],
    target_idx: usize,
    new_stmt: &shared_types::ast::Statement,
    counter: &mut usize,
) -> Vec<shared_types::ast::Statement> {
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        let current = *counter;
        *counter += 1;
        if current == target_idx {
            out.push(new_stmt.clone());
            // Still recurse into children to keep counter consistent
            count_stmts_in(stmt, counter);
        } else {
            match stmt {
                shared_types::ast::Statement::IfElse {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let new_then = replace_stmt_recursive(
                        as_stmts(then_branch),
                        target_idx,
                        new_stmt,
                        counter,
                    );
                    let new_else = else_branch.as_ref().map(|eb| {
                        replace_stmt_recursive(as_stmts(eb), target_idx, new_stmt, counter)
                    });
                    out.push(shared_types::ast::Statement::IfElse {
                        condition: condition.clone(),
                        then_branch: Box::new(shared_types::ast::Statement::Block(new_then)),
                        else_branch: new_else
                            .map(|v| Box::new(shared_types::ast::Statement::Block(v))),
                        span: span.clone(),
                    });
                }
                shared_types::ast::Statement::Block(inner) => {
                    let new_inner = replace_stmt_recursive(inner, target_idx, new_stmt, counter);
                    out.push(shared_types::ast::Statement::Block(new_inner));
                }
                shared_types::ast::Statement::Sequence(inner) => {
                    let new_inner = replace_stmt_recursive(inner, target_idx, new_stmt, counter);
                    out.push(shared_types::ast::Statement::Sequence(new_inner));
                }
                _ => {
                    out.push(stmt.clone());
                }
            }
        }
    }
    out
}

fn count_stmts_in(stmt: &shared_types::ast::Statement, counter: &mut usize) {
    match stmt {
        shared_types::ast::Statement::IfElse {
            then_branch,
            else_branch,
            ..
        } => {
            count_stmts(as_stmts(then_branch), counter);
            if let Some(eb) = else_branch {
                count_stmts(as_stmts(eb), counter);
            }
        }
        shared_types::ast::Statement::Block(inner)
        | shared_types::ast::Statement::Sequence(inner) => {
            count_stmts(inner, counter);
        }
        _ => {}
    }
}

fn count_stmts(stmts: &[shared_types::ast::Statement], counter: &mut usize) {
    for stmt in stmts {
        *counter += 1;
        count_stmts_in(stmt, counter);
    }
}

// ---------------------------------------------------------------------------
// Flat indexing helpers – walk statement list in pre-order, collecting
// (flat_index, &Statement, &Expression, expr_child_path).
// ---------------------------------------------------------------------------

pub(crate) struct ExprSite {
    pub stmt_flat_idx: usize,
    pub expr_path: Vec<usize>,
    pub expression: Expression,
}

pub(crate) fn collect_expr_sites(body: &shared_types::ast::Statement) -> Vec<ExprSite> {
    let mut sites = Vec::new();
    let mut counter = 0usize;
    collect_expr_recursive(as_stmts(body), &mut counter, &mut sites);
    sites
}

fn collect_expr_recursive(
    stmts: &[shared_types::ast::Statement],
    counter: &mut usize,
    sites: &mut Vec<ExprSite>,
) {
    for stmt in stmts {
        let idx = *counter;
        *counter += 1;
        match stmt {
            shared_types::ast::Statement::Assign { value, .. } => {
                collect_sub_exprs(value, &[], idx, sites);
            }
            shared_types::ast::Statement::Return {
                value: Some(expr), ..
            } => {
                collect_sub_exprs(expr, &[], idx, sites);
            }
            shared_types::ast::Statement::Assert { condition, .. } => {
                collect_sub_exprs(condition, &[], idx, sites);
            }
            shared_types::ast::Statement::VarDecl {
                init: Some(expr), ..
            } => {
                collect_sub_exprs(expr, &[], idx, sites);
            }
            shared_types::ast::Statement::IfElse {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                collect_sub_exprs(condition, &[], idx, sites);
                collect_expr_recursive(as_stmts(then_branch), counter, sites);
                if let Some(eb) = else_branch {
                    collect_expr_recursive(as_stmts(eb), counter, sites);
                }
            }
            shared_types::ast::Statement::Block(inner)
            | shared_types::ast::Statement::Sequence(inner) => {
                collect_expr_recursive(inner, counter, sites);
            }
            _ => {}
        }
    }
}

fn collect_sub_exprs(
    expr: &Expression,
    prefix: &[usize],
    stmt_idx: usize,
    sites: &mut Vec<ExprSite>,
) {
    sites.push(ExprSite {
        stmt_flat_idx: stmt_idx,
        expr_path: prefix.to_vec(),
        expression: expr.clone(),
    });
    match expr {
        Expression::BinaryArith { lhs, rhs, .. } | Expression::Relational { lhs, rhs, .. } => {
            let mut lp = prefix.to_vec();
            lp.push(0);
            collect_sub_exprs(lhs, &lp, stmt_idx, sites);
            let mut rp = prefix.to_vec();
            rp.push(1);
            collect_sub_exprs(rhs, &rp, stmt_idx, sites);
        }
        Expression::LogicalAnd(left, right) | Expression::LogicalOr(left, right) => {
            let mut lp = prefix.to_vec();
            lp.push(0);
            collect_sub_exprs(left, &lp, stmt_idx, sites);
            let mut rp = prefix.to_vec();
            rp.push(1);
            collect_sub_exprs(right, &rp, stmt_idx, sites);
        }
        Expression::UnaryArith(inner) | Expression::LogicalNot(inner) => {
            let mut p = prefix.to_vec();
            p.push(0);
            collect_sub_exprs(inner, &p, stmt_idx, sites);
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            let mut cp = prefix.to_vec();
            cp.push(0);
            collect_sub_exprs(condition, &cp, stmt_idx, sites);
            let mut tp = prefix.to_vec();
            tp.push(1);
            collect_sub_exprs(then_expr, &tp, stmt_idx, sites);
            let mut ep = prefix.to_vec();
            ep.push(2);
            collect_sub_exprs(else_expr, &ep, stmt_idx, sites);
        }
        Expression::FunctionCall { args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                let mut p = prefix.to_vec();
                p.push(i);
                collect_sub_exprs(arg, &p, stmt_idx, sites);
            }
        }
        Expression::ArrayAccess { array, index } => {
            let mut ap = prefix.to_vec();
            ap.push(0);
            collect_sub_exprs(array, &ap, stmt_idx, sites);
            let mut ip = prefix.to_vec();
            ip.push(1);
            collect_sub_exprs(index, &ip, stmt_idx, sites);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::ast::*;
    use shared_types::types::*;

    #[test]
    fn test_registry_creation() {
        let reg = create_standard_operators();
        assert_eq!(reg.len(), 4);
        assert!(reg.get("AOR").is_some());
        assert!(reg.get("ROR").is_some());
        assert!(reg.get("LCR").is_some());
        assert!(reg.get("UOI").is_some());
    }

    #[test]
    fn test_registry_names() {
        let reg = create_standard_operators();
        let names = reg.names();
        assert!(names.contains(&"AOR"));
        assert!(names.contains(&"ROR"));
        assert!(names.contains(&"LCR"));
        assert!(names.contains(&"UOI"));
    }

    #[test]
    fn test_replace_expr_at_path_root() {
        let orig = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(Expression::Var("x".into())),
            rhs: Box::new(Expression::IntLiteral(1)),
        };
        let replaced = replace_expr_at_path(&orig, &[], &Expression::IntLiteral(42));
        assert_eq!(replaced, Expression::IntLiteral(42));
    }

    #[test]
    fn test_replace_expr_at_path_left() {
        let orig = Expression::BinaryArith {
            op: ArithOp::Add,
            lhs: Box::new(Expression::Var("x".into())),
            rhs: Box::new(Expression::IntLiteral(1)),
        };
        let replaced = replace_expr_at_path(&orig, &[0], &Expression::Var("y".into()));
        if let Expression::BinaryArith {
            op: ArithOp::Add,
            lhs,
            ..
        } = &replaced
        {
            assert_eq!(**lhs, Expression::Var("y".into()));
        } else {
            panic!("wrong shape");
        }
    }

    #[test]
    fn test_all_sites_for_simple_function() {
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
        let reg = create_standard_operators();
        let sites = reg.all_sites(&func);
        assert!(!sites.is_empty());
    }
}
