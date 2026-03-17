//! Pretty printer for the SoniType DSL AST.
//!
//! Formats AST nodes back to human-readable source code with proper
//! indentation, type annotations, and qualifier printing.

use crate::ast::*;
use crate::type_system::{PerceptualType, Qualifier};

// ─── Pretty Printer ──────────────────────────────────────────────────────────

/// Pretty printer with configurable indentation.
pub struct PrettyPrinter {
    indent: usize,
    indent_str: String,
    output: String,
}

impl PrettyPrinter {
    pub fn new() -> Self {
        Self {
            indent: 0,
            indent_str: "  ".to_string(),
            output: String::new(),
        }
    }

    pub fn with_indent(indent_str: &str) -> Self {
        Self {
            indent: 0,
            indent_str: indent_str.to_string(),
            output: String::new(),
        }
    }

    fn push(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn push_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str(&self.indent_str);
        }
    }

    fn newline(&mut self) {
        self.output.push('\n');
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Produce the final formatted output.
    pub fn finish(self) -> String {
        self.output
    }

    // ── Program ──────────────────────────────────────────────────────────────

    pub fn print_program(&mut self, program: &Program) {
        for (i, decl) in program.declarations.iter().enumerate() {
            if i > 0 {
                self.newline();
            }
            self.print_declaration(decl);
            self.newline();
        }
    }

    // ── Declarations ─────────────────────────────────────────────────────────

    pub fn print_declaration(&mut self, decl: &Declaration) {
        self.push_indent();
        match decl {
            Declaration::StreamDecl(sd) => {
                if sd.exported { self.push("export "); }
                self.push("stream ");
                self.push(&sd.name.name);
                if let Some(ann) = &sd.type_annotation {
                    self.push(": ");
                    self.print_type_annotation(ann);
                }
                self.push(" = ");
                self.print_stream_expr(&sd.expr);
            }
            Declaration::MappingDecl(md) => {
                if md.exported { self.push("export "); }
                self.push("mapping ");
                self.push(&md.name.name);
                if let Some(ann) = &md.type_annotation {
                    self.push(": ");
                    self.print_type_annotation(ann);
                }
                self.push(" = ");
                self.print_mapping_expr(&md.expr);
            }
            Declaration::ComposeDecl(cd) => {
                if cd.exported { self.push("export "); }
                self.push("compose ");
                self.push(&cd.name.name);
                self.push(" = ");
                self.print_compose_expr(&cd.expr);
                if let Some(wc) = &cd.where_clause {
                    self.push(" where ");
                    self.print_where_clause(wc);
                }
                if let Some(wc) = &cd.with_clause {
                    self.push(" with ");
                    self.print_with_clause(wc);
                }
            }
            Declaration::DataDecl(dd) => {
                self.push("data ");
                self.push(&dd.name.name);
                self.push(" = {");
                self.newline();
                self.indent();
                for (i, field) in dd.fields.iter().enumerate() {
                    self.push_indent();
                    self.push(&field.name.name);
                    self.push(": ");
                    self.print_type(&field.ty);
                    if i < dd.fields.len() - 1 {
                        self.push(",");
                    }
                    self.newline();
                }
                self.dedent();
                self.push_indent();
                self.push("}");
            }
            Declaration::LetBinding(lb) => {
                if lb.exported { self.push("export "); }
                self.push("let ");
                self.print_pattern(&lb.pattern);
                if let Some(ann) = &lb.type_annotation {
                    self.push(": ");
                    self.print_type_annotation(ann);
                }
                self.push(" = ");
                self.print_expr(&lb.value);
            }
            Declaration::SpecDecl(sd) => {
                self.push("spec ");
                self.push(&sd.name.name);
                self.push(" {");
                self.newline();
                self.indent();
                for inner in &sd.body {
                    self.print_declaration(inner);
                    self.newline();
                }
                self.dedent();
                self.push_indent();
                self.push("}");
            }
            Declaration::ImportDecl(id) => {
                self.push("import ");
                if !id.names.is_empty() {
                    for (i, name) in id.names.iter().enumerate() {
                        if i > 0 { self.push(", "); }
                        self.push(&name.name);
                    }
                    self.push(" from ");
                }
                self.push("\"");
                self.push(&id.path);
                self.push("\"");
            }
        }
    }

    // ── Expressions ──────────────────────────────────────────────────────────

    pub fn print_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal(lit) => self.print_literal_value(&lit.value),
            Expr::Identifier(id) => self.push(&id.name),
            Expr::BinaryOp(b) => {
                self.push("(");
                self.print_expr(&b.lhs);
                self.push(&format!(" {} ", b.op));
                self.print_expr(&b.rhs);
                self.push(")");
            }
            Expr::UnaryOp(u) => {
                self.push(&format!("{}", u.op));
                self.print_expr(&u.operand);
            }
            Expr::FunctionCall(fc) => {
                self.print_expr(&fc.callee);
                self.push("(");
                for (i, arg) in fc.args.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.print_expr(arg);
                }
                self.push(")");
            }
            Expr::LetIn(li) => {
                self.push("let ");
                self.print_pattern(&li.pattern);
                if let Some(ann) = &li.type_annotation {
                    self.push(": ");
                    self.print_type_annotation(ann);
                }
                self.push(" = ");
                self.print_expr(&li.value);
                self.push(" in ");
                self.print_expr(&li.body);
            }
            Expr::IfThenElse(ite) => {
                self.push("if ");
                self.print_expr(&ite.condition);
                self.push(" then ");
                self.print_expr(&ite.then_branch);
                self.push(" else ");
                self.print_expr(&ite.else_branch);
            }
            Expr::Lambda(lam) => {
                self.push("\\(");
                for (i, p) in lam.params.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.push(&p.name.name);
                    if let Some(ann) = &p.type_annotation {
                        self.push(": ");
                        self.print_type_annotation(ann);
                    }
                }
                self.push(") -> ");
                self.print_expr(&lam.body);
            }
            Expr::StreamLiteral(s) => self.print_stream_expr(s),
            Expr::MappingLiteral(m) => self.print_mapping_expr(m),
            Expr::Compose(c) => self.print_compose_expr(c),
            Expr::PipeOperator(p) => {
                self.print_expr(&p.lhs);
                self.push(" |> ");
                self.print_expr(&p.rhs);
            }
            Expr::FieldAccess(fa) => {
                self.print_expr(&fa.object);
                self.push(".");
                self.push(&fa.field.name);
            }
            Expr::Grouped(inner, _) => {
                self.push("(");
                self.print_expr(inner);
                self.push(")");
            }
            Expr::WithClause(w) => {
                self.print_expr(&w.expr);
                self.push(" with { ");
                for (i, c) in w.constraints.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.push(&c.name.name);
                    self.push(": ");
                    self.print_expr(&c.value);
                }
                self.push(" }");
            }
            Expr::WhereClause(w) => {
                self.print_expr(&w.expr);
                self.push(" where { ");
                for (i, b) in w.bindings.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.push(&b.name.name);
                    self.push(": ");
                    self.print_expr(&b.value);
                }
                self.push(" }");
            }
        }
    }

    fn print_literal_value(&mut self, value: &LiteralValue) {
        match value {
            LiteralValue::Int(v) => self.push(&v.to_string()),
            LiteralValue::Float(v) => {
                let s = format!("{v}");
                self.push(&s);
                // Ensure there's a decimal point for floats
                if !s.contains('.') && !s.contains('e') {
                    self.push(".0");
                }
            }
            LiteralValue::String(s) => {
                self.push("\"");
                self.push(&s.replace('\\', "\\\\").replace('"', "\\\""));
                self.push("\"");
            }
            LiteralValue::Bool(b) => self.push(if *b { "true" } else { "false" }),
        }
    }

    fn print_stream_expr(&mut self, s: &StreamExpr) {
        self.push("stream {");
        self.newline();
        self.indent();
        for (i, param) in s.params.iter().enumerate() {
            self.push_indent();
            self.push(&param.name.name);
            self.push(": ");
            self.print_expr(&param.value);
            if i < s.params.len() - 1 {
                self.push(",");
            }
            self.newline();
        }
        self.dedent();
        self.push_indent();
        self.push("}");
    }

    fn print_mapping_expr(&mut self, m: &MappingExpr) {
        self.push(&m.source.source.name);
        self.push(".");
        self.push(&m.source.field.name);
        self.push(" -> ");
        self.push(&format!("{}", m.target.param));
        if let Some((lo, hi)) = &m.target.range {
            self.push("(");
            self.print_expr(lo);
            self.push("..");
            self.print_expr(hi);
            self.push(")");
        }
    }

    fn print_compose_expr(&mut self, c: &ComposeExpr) {
        self.push("compose {");
        self.newline();
        self.indent();
        self.push_indent();
        for (i, stream) in c.streams.iter().enumerate() {
            if i > 0 {
                self.push(" || ");
            }
            self.print_expr(stream);
        }
        self.newline();
        self.dedent();
        self.push_indent();
        self.push("}");
    }

    fn print_where_clause(&mut self, wc: &WhereClause) {
        self.push("{");
        for (i, b) in wc.bindings.iter().enumerate() {
            if i > 0 { self.push(", "); }
            self.push(&b.name.name);
            self.push(": ");
            self.print_expr(&b.value);
        }
        self.push("}");
    }

    fn print_with_clause(&mut self, wc: &WithClause) {
        self.push("{");
        for (i, c) in wc.constraints.iter().enumerate() {
            if i > 0 { self.push(", "); }
            self.push(&c.name.name);
            self.push(": ");
            self.print_expr(&c.value);
        }
        self.push("}");
    }

    // ── Patterns ─────────────────────────────────────────────────────────────

    pub fn print_pattern(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Variable(id) => self.push(&id.name),
            Pattern::Tuple(pats, _) => {
                self.push("(");
                for (i, p) in pats.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.print_pattern(p);
                }
                self.push(")");
            }
            Pattern::Wildcard(_) => self.push("_"),
        }
    }

    // ── Types ────────────────────────────────────────────────────────────────

    pub fn print_type(&mut self, ty: &Type) {
        match ty {
            Type::Named(name, _) => self.push(name),
            Type::Parameterized(name, args, _) => {
                self.push(name);
                self.push("<");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.print_type(arg);
                }
                self.push(">");
            }
            Type::Function(from, to, _) => {
                self.print_type(from);
                self.push(" -> ");
                self.print_type(to);
            }
            Type::Tuple(elems, _) => {
                self.push("(");
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.print_type(e);
                }
                self.push(")");
            }
            Type::Qualified(base, ann, _) => {
                self.print_type(base);
                self.push("⟨");
                self.print_perceptual_annotation(ann);
                self.push("⟩");
            }
            Type::Variable(name, _) => {
                self.push("'");
                self.push(name);
            }
        }
    }

    pub fn print_type_annotation(&mut self, ann: &TypeAnnotation) {
        self.print_type(&ann.ty);
        if let Some(qual) = &ann.qualifier {
            self.push("⟨");
            self.print_perceptual_annotation(qual);
            self.push("⟩");
        }
    }

    pub fn print_perceptual_annotation(&mut self, ann: &PerceptualAnnotation) {
        for (i, pred) in ann.predicates.iter().enumerate() {
            if i > 0 { self.push(", "); }
            self.print_perceptual_predicate(pred);
        }
    }

    fn print_perceptual_predicate(&mut self, pred: &PerceptualPredicate) {
        match pred {
            PerceptualPredicate::BandOccupancy(bands) => {
                self.push("band: {");
                for (i, b) in bands.iter().enumerate() {
                    if i > 0 { self.push(", "); }
                    self.push(&b.to_string());
                }
                self.push("}");
            }
            PerceptualPredicate::CognitiveLoad(v) => {
                self.push(&format!("load: {v}"));
            }
            PerceptualPredicate::MaskingMargin(v) => {
                self.push(&format!("masking: {v}"));
            }
            PerceptualPredicate::Segregation(b) => {
                self.push(&format!("segregation: {b}"));
            }
            PerceptualPredicate::Jnd(param, v) => {
                self.push(&format!("jnd({param}): {v}"));
            }
        }
    }

    // ── Perceptual types (from type_system) ──────────────────────────────────

    pub fn print_perceptual_type(&mut self, ty: &PerceptualType) {
        self.push(&format!("{}", ty.base));
        if !ty.qualifier.is_trivial() {
            self.push("⟨");
            self.print_qualifier(&ty.qualifier);
            self.push("⟩");
        }
    }

    pub fn print_qualifier(&mut self, q: &Qualifier) {
        self.push(&format!("{q}"));
    }
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Convenience function ────────────────────────────────────────────────────

/// Pretty-print a program to a string.
pub fn pretty_print(program: &Program) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_program(program);
    pp.finish()
}

/// Pretty-print a single expression to a string.
pub fn pretty_print_expr(expr: &Expr) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_expr(expr);
    pp.finish()
}

/// Pretty-print a type to a string.
pub fn pretty_print_type(ty: &Type) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_type(ty);
    pp.finish()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;
    use crate::token::Span;

    fn roundtrip(src: &str) -> String {
        let tokens = lex(src).expect("lex failed");
        let prog = parse(tokens).expect("parse failed");
        pretty_print(&prog)
    }

    #[test]
    fn test_pp_let_binding() {
        let s = roundtrip("let x = 42");
        assert!(s.contains("let x = 42"));
    }

    #[test]
    fn test_pp_binary_op() {
        let s = roundtrip("let x = 1 + 2");
        assert!(s.contains("(1 + 2)"));
    }

    #[test]
    fn test_pp_boolean() {
        let s = roundtrip("let x = true");
        assert!(s.contains("true"));
    }

    #[test]
    fn test_pp_string_literal() {
        let s = roundtrip(r#"let x = "hello""#);
        assert!(s.contains(r#""hello""#));
    }

    #[test]
    fn test_pp_if_then_else() {
        let s = roundtrip("let x = if true then 1 else 0");
        assert!(s.contains("if true then 1 else 0"));
    }

    #[test]
    fn test_pp_stream_decl() {
        let s = roundtrip(r#"stream s = { freq: 440.0, timbre: "sine" }"#);
        assert!(s.contains("stream s"));
        assert!(s.contains("freq: 440"));
    }

    #[test]
    fn test_pp_data_decl() {
        let s = roundtrip("data d = { temperature: Float, pressure: Float }");
        assert!(s.contains("data d"));
        assert!(s.contains("temperature: Float"));
    }

    #[test]
    fn test_pp_mapping_decl() {
        let s = roundtrip("mapping m = data.temperature -> pitch(200..800)");
        assert!(s.contains("mapping m"));
        assert!(s.contains("pitch"));
    }

    #[test]
    fn test_pp_compose_decl() {
        let s = roundtrip("compose c = { s1 || s2 }");
        assert!(s.contains("compose c"));
        assert!(s.contains("||"));
    }

    #[test]
    fn test_pp_import() {
        let s = roundtrip(r#"import "stdlib""#);
        assert!(s.contains(r#"import "stdlib""#));
    }

    #[test]
    fn test_pp_export_let() {
        let s = roundtrip("export let x = 42");
        assert!(s.contains("export let x = 42"));
    }

    #[test]
    fn test_pp_pipe() {
        let s = roundtrip("let x = a |> f");
        assert!(s.contains("|>"));
    }

    #[test]
    fn test_pp_type_display() {
        let ty = Type::Function(
            Box::new(Type::Named("Stream".into(), Span::dummy())),
            Box::new(Type::Named("Mapping".into(), Span::dummy())),
            Span::dummy(),
        );
        let s = pretty_print_type(&ty);
        assert_eq!(s, "Stream -> Mapping");
    }

    #[test]
    fn test_pp_perceptual_type() {
        use crate::type_system::{BaseType, Qualifier};
        use std::collections::BTreeSet;

        let ty = PerceptualType::new(
            BaseType::Stream,
            Qualifier {
                band_occupancy: BTreeSet::from([3, 4]),
                cognitive_load: 1.0,
                masking_margin: f64::INFINITY,
                ..Default::default()
            },
        );
        let mut pp = PrettyPrinter::new();
        pp.print_perceptual_type(&ty);
        let s = pp.finish();
        assert!(s.contains("Stream"));
        assert!(s.contains("band"));
    }

    #[test]
    fn test_pp_indentation() {
        let s = roundtrip("data d = { x: Float, y: Int }");
        // Should have indented fields
        assert!(s.contains("  x: Float"));
    }
}
