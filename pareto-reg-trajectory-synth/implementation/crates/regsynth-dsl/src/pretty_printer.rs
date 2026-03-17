use crate::ast::*;
use std::fmt::Write;

/// Pretty printer for the regulatory DSL AST.
pub struct PrettyPrinter {
    output: String,
    indent: usize,
    indent_str: String,
}

impl PrettyPrinter {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            indent_str: "    ".to_string(),
        }
    }

    pub fn with_indent(mut self, indent_str: &str) -> Self {
        self.indent_str = indent_str.to_string();
        self
    }

    /// Pretty print a program and return the result.
    pub fn print_program(&mut self, program: &Program) -> String {
        self.output.clear();
        for (i, decl) in program.declarations.iter().enumerate() {
            if i > 0 {
                self.newline();
            }
            self.print_declaration(decl);
        }
        self.output.clone()
    }

    /// Pretty print a single declaration.
    pub fn print_declaration(&mut self, decl: &Declaration) {
        match &decl.kind {
            DeclarationKind::Jurisdiction(j) => self.print_jurisdiction(j),
            DeclarationKind::Obligation(o) => self.print_obligation_decl(o),
            DeclarationKind::Permission(p) => self.print_permission_decl(p),
            DeclarationKind::Prohibition(p) => self.print_prohibition_decl(p),
            DeclarationKind::Framework(f) => self.print_framework(f),
            DeclarationKind::Strategy(s) => self.print_strategy(s),
            DeclarationKind::Cost(c) => self.print_cost(c),
            DeclarationKind::Temporal(t) => self.print_temporal(t),
            DeclarationKind::Mapping(m) => self.print_mapping(m),
        }
    }

    fn print_jurisdiction(&mut self, j: &JurisdictionDecl) {
        self.write_indent();
        let _ = write!(self.output, "jurisdiction \"{}\"", j.name);
        if let Some(parent) = &j.parent {
            let _ = write!(self.output, " : \"{}\"", parent);
        }
        self.output.push_str(" {");
        self.newline();
        self.indent += 1;
        for child in &j.body {
            self.print_declaration(child);
        }
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_obligation_decl(&mut self, o: &ObligationDecl) {
        self.write_indent();
        let _ = write!(self.output, "obligation {}", o.name);
        if let Some(j) = &o.jurisdiction {
            let _ = write!(self.output, " : \"{}\"", j);
        }
        self.output.push_str(" {");
        self.newline();
        self.indent += 1;
        self.print_obligation_body(&o.body);
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_permission_decl(&mut self, p: &PermissionDecl) {
        self.write_indent();
        let _ = write!(self.output, "permission {}", p.name);
        if let Some(j) = &p.jurisdiction {
            let _ = write!(self.output, " : \"{}\"", j);
        }
        self.output.push_str(" {");
        self.newline();
        self.indent += 1;
        self.print_obligation_body(&p.body);
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_prohibition_decl(&mut self, p: &ProhibitionDecl) {
        self.write_indent();
        let _ = write!(self.output, "prohibition {}", p.name);
        if let Some(j) = &p.jurisdiction {
            let _ = write!(self.output, " : \"{}\"", j);
        }
        self.output.push_str(" {");
        self.newline();
        self.indent += 1;
        self.print_obligation_body(&p.body);
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_framework(&mut self, f: &FrameworkDecl) {
        self.write_indent();
        let _ = write!(self.output, "framework \"{}\"", f.name);
        if let Some(j) = &f.jurisdiction {
            let _ = write!(self.output, " : \"{}\"", j);
        }
        self.output.push_str(" {");
        self.newline();
        self.indent += 1;
        for child in &f.body {
            self.print_declaration(child);
        }
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_strategy(&mut self, s: &StrategyDecl) {
        self.write_indent();
        let _ = write!(self.output, "strategy {} {{", s.name);
        self.newline();
        self.indent += 1;
        for item in &s.body {
            self.write_indent();
            let _ = write!(self.output, "{}: ", item.key);
            self.print_expression(&item.value);
            self.output.push(';');
            self.newline();
        }
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    fn print_cost(&mut self, c: &CostDecl) {
        self.write_indent();
        let _ = write!(self.output, "cost {} {{ ", c.name);
        self.print_expression(&c.amount);
        if let Some(currency) = &c.currency {
            let _ = write!(self.output, " {}", currency);
        }
        self.output.push_str("; }");
        self.newline();
    }

    fn print_temporal(&mut self, t: &TemporalDecl) {
        self.write_indent();
        let _ = write!(self.output, "temporal {} {{ ", t.name);
        if let Some(start) = &t.start {
            let _ = write!(self.output, "#{}", start);
        }
        if let Some(end) = &t.end {
            let _ = write!(self.output, " -> #{}", end);
        }
        self.output.push_str("; }");
        self.newline();
    }

    fn print_mapping(&mut self, m: &MappingDecl) {
        self.write_indent();
        let _ = write!(self.output, "mapping {} {{", m.name);
        self.newline();
        self.indent += 1;
        for entry in &m.entries {
            self.write_indent();
            self.print_expression(&entry.from);
            self.output.push_str(" => ");
            self.print_expression(&entry.to);
            self.output.push(';');
            self.newline();
        }
        self.indent -= 1;
        self.write_indent();
        self.output.push('}');
        self.newline();
    }

    /// Print an obligation body (shared by obligation, permission, prohibition).
    pub fn print_obligation_body(&mut self, body: &ObligationBody) {
        if let Some(risk) = &body.risk_level {
            self.write_indent();
            let _ = write!(self.output, "risk_level: {};", risk);
            self.newline();
        }

        if let Some(grade) = &body.formalizability {
            self.write_indent();
            let _ = write!(self.output, "formalizability: {};", grade.as_u8());
            self.newline();
        }

        if let Some(domain) = &body.domain {
            self.write_indent();
            let _ = write!(self.output, "domain: \"{}\";", domain);
            self.newline();
        }

        if body.temporal_start.is_some() || body.temporal_end.is_some() {
            self.write_indent();
            self.output.push_str("temporal: ");
            if let Some(start) = &body.temporal_start {
                let _ = write!(self.output, "#{}", start);
            }
            if let Some(end) = &body.temporal_end {
                let _ = write!(self.output, " -> #{}", end);
            }
            self.output.push(';');
            self.newline();
        }

        for cond in &body.conditions {
            self.write_indent();
            self.output.push_str("requires: ");
            self.print_expression(cond);
            self.output.push(';');
            self.newline();
        }

        for exempt in &body.exemptions {
            self.write_indent();
            self.output.push_str("exempts: ");
            self.print_expression(exempt);
            self.output.push(';');
            self.newline();
        }

        for article in &body.article_refs {
            self.write_indent();
            let _ = write!(
                self.output,
                "article \"{}\" \"{}\"",
                article.framework, article.article
            );
            if let Some(p) = &article.paragraph {
                let _ = write!(self.output, " (\"{}\")", p);
            }
            self.output.push(';');
            self.newline();
        }

        for comp in &body.compositions {
            self.write_indent();
            let _ = write!(self.output, "compose: {} {};", comp.op, comp.operand);
            self.newline();
        }

        for field in &body.extra_fields {
            self.write_indent();
            let _ = write!(self.output, "{}: ", field.key);
            self.print_expression(&field.value);
            self.output.push(';');
            self.newline();
        }
    }

    /// Pretty print an expression.
    pub fn print_expression(&mut self, expr: &Expression) {
        match &expr.kind {
            ExpressionKind::Literal(lit) => {
                let _ = write!(self.output, "{}", lit);
            }
            ExpressionKind::Variable(name) => {
                self.output.push_str(name);
            }
            ExpressionKind::BinaryOp { op, left, right } => {
                let needs_parens_left = matches!(
                    &left.kind,
                    ExpressionKind::BinaryOp { op: inner_op, .. }
                    if inner_op.precedence() < op.precedence()
                );
                let needs_parens_right = matches!(
                    &right.kind,
                    ExpressionKind::BinaryOp { op: inner_op, .. }
                    if inner_op.precedence() <= op.precedence() && !op.is_right_assoc()
                );

                if needs_parens_left {
                    self.output.push('(');
                }
                self.print_expression(left);
                if needs_parens_left {
                    self.output.push(')');
                }

                let _ = write!(self.output, " {} ", op);

                if needs_parens_right {
                    self.output.push('(');
                }
                self.print_expression(right);
                if needs_parens_right {
                    self.output.push(')');
                }
            }
            ExpressionKind::UnaryOp { op, operand } => {
                let _ = write!(self.output, "{} ", op);
                let needs_parens = matches!(
                    &operand.kind,
                    ExpressionKind::BinaryOp { .. }
                );
                if needs_parens {
                    self.output.push('(');
                }
                self.print_expression(operand);
                if needs_parens {
                    self.output.push(')');
                }
            }
            ExpressionKind::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.output.push_str("if ");
                self.print_expression(condition);
                self.output.push_str(" then ");
                self.print_expression(then_branch);
                self.output.push_str(" else ");
                self.print_expression(else_branch);
            }
            ExpressionKind::Quantifier {
                kind,
                variable,
                domain,
                body,
            } => {
                let _ = write!(self.output, "{} {}", kind, variable);
                if let Some(d) = domain {
                    self.output.push_str(": ");
                    self.print_expression(d);
                }
                self.output.push_str(". ");
                self.print_expression(body);
            }
            ExpressionKind::FunctionCall { function, args } => {
                self.output.push_str(function);
                self.output.push('(');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expression(arg);
                }
                self.output.push(')');
            }
            ExpressionKind::ArticleRef {
                framework,
                article,
                paragraph,
            } => {
                let _ = write!(self.output, "article \"{}\" \"{}\"", framework, article);
                if let Some(p) = paragraph {
                    let _ = write!(self.output, " (\"{}\")", p);
                }
            }
            ExpressionKind::FieldAccess { object, field } => {
                self.print_expression(object);
                let _ = write!(self.output, ".{}", field);
            }
            ExpressionKind::Composition { op, left, right } => {
                self.print_expression(left);
                let _ = write!(self.output, " {} ", op);
                self.print_expression(right);
            }
        }
    }

    // ─── Helpers ────────────────────────────────────────────────

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str(&self.indent_str);
        }
    }

    fn newline(&mut self) {
        self.output.push('\n');
    }

    /// Get the current output.
    pub fn output(&self) -> &str {
        &self.output
    }
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: pretty print a program.
pub fn pretty_print(program: &Program) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_program(program)
}

/// Convenience function: pretty print an expression.
pub fn pretty_print_expr(expr: &Expression) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_expression(expr);
    pp.output().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;
    use crate::source_map::Span;

    fn roundtrip(src: &str) -> (String, String) {
        let (tokens, lex_errors) = lex(src);
        assert!(lex_errors.is_empty(), "lex errors: {:?}", lex_errors);
        let (program, parse_errors) = parse(tokens);
        assert!(
            parse_errors.is_empty(),
            "parse errors: {:?}",
            parse_errors
        );
        let printed = pretty_print(&program);

        // Parse the printed output again
        let (tokens2, lex_errors2) = lex(&printed);
        assert!(lex_errors2.is_empty(), "roundtrip lex errors: {:?}", lex_errors2);
        let (program2, parse_errors2) = parse(tokens2);
        assert!(
            parse_errors2.is_empty(),
            "roundtrip parse errors: {:?}\nprinted:\n{}",
            parse_errors2,
            printed
        );
        let printed2 = pretty_print(&program2);

        (printed, printed2)
    }

    #[test]
    fn test_roundtrip_obligation() {
        let (first, second) = roundtrip(r#"
            obligation transparency {
                risk_level: high;
                formalizability: 3;
                requires: true;
                exempts: false;
            }
        "#);
        assert_eq!(first, second, "roundtrip mismatch:\nfirst:\n{}\nsecond:\n{}", first, second);
    }

    #[test]
    fn test_roundtrip_jurisdiction() {
        let (first, second) = roundtrip(r#"
            jurisdiction "EU" {
                obligation gdpr {
                    requires: true;
                }
            }
        "#);
        assert_eq!(first, second);
    }

    #[test]
    fn test_roundtrip_expression() {
        let (first, second) = roundtrip(r#"
            obligation test {
                requires: a and b or c;
            }
        "#);
        assert_eq!(first, second);
    }

    #[test]
    fn test_roundtrip_if_then_else() {
        let (first, second) = roundtrip(r#"
            obligation test {
                requires: if x then y else z;
            }
        "#);
        assert_eq!(first, second);
    }

    #[test]
    fn test_roundtrip_temporal() {
        let (first, second) = roundtrip(r#"
            obligation test {
                temporal: #2024-01-01 -> #2025-12-31;
                requires: true;
            }
        "#);
        assert_eq!(first, second);
    }

    #[test]
    fn test_pretty_print_expr_simple() {
        let expr = Expression {
            kind: ExpressionKind::BinaryOp {
                op: BinOp::And,
                left: Box::new(Expression {
                    kind: ExpressionKind::Variable("a".into()),
                    span: Span::empty(),
                }),
                right: Box::new(Expression {
                    kind: ExpressionKind::Variable("b".into()),
                    span: Span::empty(),
                }),
            },
            span: Span::empty(),
        };
        let result = pretty_print_expr(&expr);
        assert_eq!(result, "a and b");
    }

    #[test]
    fn test_pretty_print_permission_prohibition() {
        let (first, second) = roundtrip(r#"
            permission data_use {
                domain: "research";
                requires: true;
            }
            prohibition scoring {
                risk_level: unacceptable;
                requires: true;
            }
        "#);
        assert_eq!(first, second);
    }

    #[test]
    fn test_roundtrip_strategy() {
        let (first, second) = roundtrip(r#"
            strategy mitigation {
                approach: "risk_based";
                threshold: 42;
            }
        "#);
        assert_eq!(first, second);
    }
}
