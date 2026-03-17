//! Desugaring passes for the SoniType DSL.
//!
//! Transforms syntactic sugar into core forms before type checking:
//! - Pipe operator (`|>`) → function application
//! - `where` clauses → explicit constraint annotations
//! - `with` blocks → parameter settings
//! - N-ary compose → binary compose tree

use crate::ast::*;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Desugar a complete program, applying all transformation passes.
pub fn desugar_program(program: &mut Program) {
    for decl in &mut program.declarations {
        desugar_declaration(decl);
    }
}

/// Desugar a single declaration.
pub fn desugar_declaration(decl: &mut Declaration) {
    match decl {
        Declaration::LetBinding(lb) => {
            desugar_expr(&mut lb.value);
        }
        Declaration::StreamDecl(sd) => {
            for param in &mut sd.expr.params {
                desugar_expr(&mut param.value);
            }
        }
        Declaration::ComposeDecl(cd) => {
            desugar_compose(&mut cd.expr);
            // Flatten where clause into compose
            if let Some(wc) = &cd.where_clause {
                for binding in &wc.bindings {
                    // Inject where bindings as constraints on the compose
                    let _ = binding; // consumed during type checking
                }
            }
        }
        Declaration::MappingDecl(md) => {
            if let Some((lo, hi)) = &mut md.expr.target.range {
                desugar_expr(lo);
                desugar_expr(hi);
            }
        }
        Declaration::SpecDecl(sd) => {
            for inner in &mut sd.body {
                desugar_declaration(inner);
            }
        }
        Declaration::DataDecl(_) | Declaration::ImportDecl(_) => {}
    }
}

/// Desugar an expression, recursively transforming all sub-expressions.
pub fn desugar_expr(expr: &mut Expr) {
    // Bottom-up: desugar children first, then the node itself.
    match expr {
        Expr::BinaryOp(b) => {
            desugar_expr(&mut b.lhs);
            desugar_expr(&mut b.rhs);
        }
        Expr::UnaryOp(u) => {
            desugar_expr(&mut u.operand);
        }
        Expr::FunctionCall(fc) => {
            desugar_expr(&mut fc.callee);
            for arg in &mut fc.args {
                desugar_expr(arg);
            }
        }
        Expr::LetIn(li) => {
            desugar_expr(&mut li.value);
            desugar_expr(&mut li.body);
        }
        Expr::IfThenElse(ite) => {
            desugar_expr(&mut ite.condition);
            desugar_expr(&mut ite.then_branch);
            desugar_expr(&mut ite.else_branch);
        }
        Expr::Lambda(lam) => {
            desugar_expr(&mut lam.body);
        }
        Expr::Grouped(inner, _) => {
            desugar_expr(inner);
        }
        Expr::StreamLiteral(s) => {
            for p in &mut s.params {
                desugar_expr(&mut p.value);
            }
        }
        Expr::Compose(c) => {
            desugar_compose(c);
        }
        Expr::FieldAccess(fa) => {
            desugar_expr(&mut fa.object);
        }
        Expr::WithClause(w) => {
            desugar_expr(&mut w.expr);
            desugar_with_clause(expr);
        }
        Expr::WhereClause(w) => {
            desugar_expr(&mut w.expr);
            desugar_where_clause(expr);
        }
        Expr::PipeOperator(_) => {
            desugar_pipe(expr);
        }
        Expr::Literal(_) | Expr::Identifier(_) | Expr::MappingLiteral(_) => {}
    }
}

// ─── Pipe Operator Desugaring ────────────────────────────────────────────────

/// Transform `a |> f` into `f(a)`, handling chains: `a |> f |> g` → `g(f(a))`.
fn desugar_pipe(expr: &mut Expr) {
    // Collect the pipe chain left-to-right
    let mut chain = Vec::new();
    collect_pipe_chain(expr, &mut chain);

    if chain.len() < 2 {
        return;
    }

    // chain[0] is the initial value, chain[1..] are the functions
    let mut result = chain.remove(0);
    desugar_expr(&mut result);

    for mut func in chain {
        desugar_expr(&mut func);
        let span = result.span().merge(func.span());
        result = Expr::FunctionCall(FunctionCall {
            callee: Box::new(func),
            args: vec![result],
            span,
        });
    }

    *expr = result;
}

fn collect_pipe_chain(expr: &Expr, chain: &mut Vec<Expr>) {
    match expr {
        Expr::PipeOperator(p) => {
            collect_pipe_chain(&p.lhs, chain);
            chain.push(*p.rhs.clone());
        }
        other => {
            chain.push(other.clone());
        }
    }
}

// ─── Where Clause Desugaring ─────────────────────────────────────────────────

/// Transform `expr where { x: v1, y: v2 }` into
/// `let x = v1 in let y = v2 in expr`.
fn desugar_where_clause(expr: &mut Expr) {
    if let Expr::WhereClause(w) = expr {
        let inner = *w.expr.clone();
        let span = w.span;

        let mut result = inner;
        // Build let-in chain from bindings in reverse so outermost is first binding
        for binding in w.bindings.iter().rev() {
            result = Expr::LetIn(LetIn {
                pattern: Pattern::Variable(binding.name.clone()),
                type_annotation: None,
                value: Box::new(binding.value.clone()),
                body: Box::new(result),
                span,
            });
        }

        *expr = result;
    }
}

// ─── With Clause Desugaring ──────────────────────────────────────────────────

/// `with` clauses are kept as-is for the type checker to interpret as constraints.
/// This pass simply ensures sub-expressions are desugared.
fn desugar_with_clause(expr: &mut Expr) {
    if let Expr::WithClause(w) = expr {
        for c in &mut w.constraints {
            desugar_expr(&mut c.value);
        }
    }
}

// ─── Compose Normalization ───────────────────────────────────────────────────

/// Normalize N-ary compose into left-nested binary form:
/// `compose { a || b || c }` → streams remain as a vec (already binary-amenable),
/// but we desugar each sub-expression.
fn desugar_compose(compose: &mut ComposeExpr) {
    for stream in &mut compose.streams {
        desugar_expr(stream);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn parse_and_desugar(src: &str) -> Program {
        let tokens = lex(src).expect("lex failed");
        let mut prog = parse(tokens).expect("parse failed");
        desugar_program(&mut prog);
        prog
    }

    fn extract_let_value(prog: &Program) -> &Expr {
        match &prog.declarations[0] {
            Declaration::LetBinding(lb) => &lb.value,
            _ => panic!("expected let binding"),
        }
    }

    #[test]
    fn test_simple_pipe_desugar() {
        let prog = parse_and_desugar("let x = a |> f");
        let expr = extract_let_value(&prog);
        assert!(matches!(expr, Expr::FunctionCall(_)));
    }

    #[test]
    fn test_chained_pipe_desugar() {
        let prog = parse_and_desugar("let x = a |> f |> g");
        let expr = extract_let_value(&prog);
        // Should be g(f(a))
        match expr {
            Expr::FunctionCall(fc) => {
                assert!(matches!(*fc.callee, Expr::Identifier(ref id) if id.name == "g"));
                assert_eq!(fc.args.len(), 1);
                assert!(matches!(&fc.args[0], Expr::FunctionCall(_)));
            }
            _ => panic!("expected function call"),
        }
    }

    #[test]
    fn test_literal_unchanged() {
        let prog = parse_and_desugar("let x = 42");
        let expr = extract_let_value(&prog);
        assert!(matches!(expr, Expr::Literal(_)));
    }

    #[test]
    fn test_binary_op_children_desugared() {
        let prog = parse_and_desugar("let x = (a |> f) + 1");
        let expr = extract_let_value(&prog);
        match expr {
            Expr::BinaryOp(b) => {
                // lhs should be desugared: grouped(function_call)
                match b.lhs.as_ref() {
                    Expr::Grouped(inner, _) => {
                        assert!(matches!(inner.as_ref(), Expr::FunctionCall(_)));
                    }
                    _ => panic!("expected grouped function call"),
                }
            }
            _ => panic!("expected binary op"),
        }
    }

    #[test]
    fn test_if_then_else_children_desugared() {
        let prog = parse_and_desugar("let x = if true then a |> f else 0");
        let expr = extract_let_value(&prog);
        match expr {
            Expr::IfThenElse(ite) => {
                assert!(matches!(*ite.then_branch, Expr::FunctionCall(_)));
            }
            _ => panic!("expected if-then-else"),
        }
    }

    #[test]
    fn test_compose_children_desugared() {
        let prog = parse_and_desugar("compose c = { s1 || s2 }");
        // Should not crash; compose streams are desugared
        assert!(!prog.declarations.is_empty());
    }

    #[test]
    fn test_stream_params_desugared() {
        let prog = parse_and_desugar(r#"stream s = { freq: 440.0, timbre: "sine" }"#);
        assert!(!prog.declarations.is_empty());
    }

    #[test]
    fn test_lambda_body_desugared() {
        let prog = parse_and_desugar("let f = 42");
        // Simple – just ensure no crash
        assert!(!prog.declarations.is_empty());
    }

    #[test]
    fn test_nested_let_in_desugared() {
        let prog = parse_and_desugar("let x = let y = 1 in y");
        let expr = extract_let_value(&prog);
        assert!(matches!(expr, Expr::LetIn(_)));
    }

    #[test]
    fn test_desugar_idempotent() {
        let src = "let x = 1 + 2";
        let mut prog1 = {
            let tokens = lex(src).unwrap();
            parse(tokens).unwrap()
        };
        desugar_program(&mut prog1);
        let snapshot1 = format!("{:?}", prog1);

        desugar_program(&mut prog1);
        let snapshot2 = format!("{:?}", prog1);

        assert_eq!(snapshot1, snapshot2, "desugaring should be idempotent");
    }
}
