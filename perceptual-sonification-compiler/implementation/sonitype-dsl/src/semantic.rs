//! Semantic analysis for the SoniType DSL.
//!
//! Performs name resolution, scope analysis, symbol table construction,
//! import resolution, and validation of data schemas against mappings.

use crate::ast::*;
use crate::token::Span;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─── Symbols ─────────────────────────────────────────────────────────────────

/// Kinds of symbols in the SoniType DSL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Symbol {
    Variable {
        name: String,
        ty: Option<Type>,
        mutable: bool,
        span: Span,
    },
    Stream {
        name: String,
        params: Vec<String>,
        span: Span,
    },
    Mapping {
        name: String,
        source_field: String,
        target_param: String,
        span: Span,
    },
    DataSource {
        name: String,
        fields: Vec<(String, Type)>,
        span: Span,
    },
    Function {
        name: String,
        param_count: usize,
        span: Span,
    },
}

impl Symbol {
    pub fn name(&self) -> &str {
        match self {
            Symbol::Variable { name, .. }
            | Symbol::Stream { name, .. }
            | Symbol::Mapping { name, .. }
            | Symbol::DataSource { name, .. }
            | Symbol::Function { name, .. } => name,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Symbol::Variable { span, .. }
            | Symbol::Stream { span, .. }
            | Symbol::Mapping { span, .. }
            | Symbol::DataSource { span, .. }
            | Symbol::Function { span, .. } => *span,
        }
    }
}

// ─── Symbol Table ────────────────────────────────────────────────────────────

/// Scoped symbol table for name resolution.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self { scopes: vec![HashMap::new()] }
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn define(&mut self, name: &str, symbol: Symbol) -> Option<Symbol> {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), symbol)
        } else {
            None
        }
    }

    pub fn resolve(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.get(name) {
                return Some(sym);
            }
        }
        None
    }

    pub fn is_defined_in_current_scope(&self, name: &str) -> bool {
        self.scopes.last().map_or(false, |s| s.contains_key(name))
    }

    /// All symbols visible at the current scope level.
    pub fn all_visible(&self) -> Vec<&Symbol> {
        let mut seen = HashMap::new();
        for scope in self.scopes.iter().rev() {
            for (name, sym) in scope {
                seen.entry(name.clone()).or_insert(sym);
            }
        }
        seen.into_values().collect()
    }

    pub fn depth(&self) -> usize {
        self.scopes.len()
    }
}

// ─── Semantic Errors ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticError {
    pub kind: SemanticErrorKind,
    pub message: String,
    pub span: Span,
}

impl SemanticError {
    pub fn new(kind: SemanticErrorKind, message: impl Into<String>, span: Span) -> Self {
        Self { kind, message: message.into(), span }
    }
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Semantic error at {}: [{}] {}", self.span, self.kind, self.message)
    }
}

impl std::error::Error for SemanticError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticErrorKind {
    UndefinedVariable,
    DuplicateDefinition,
    InvalidDataField,
    InvalidStreamParam,
    MappingSchemaViolation,
    UnresolvedImport,
    InvalidRange,
    TypeAnnotationMismatch,
}

impl fmt::Display for SemanticErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticErrorKind::UndefinedVariable => write!(f, "undefined variable"),
            SemanticErrorKind::DuplicateDefinition => write!(f, "duplicate definition"),
            SemanticErrorKind::InvalidDataField => write!(f, "invalid data field"),
            SemanticErrorKind::InvalidStreamParam => write!(f, "invalid stream param"),
            SemanticErrorKind::MappingSchemaViolation => write!(f, "mapping schema violation"),
            SemanticErrorKind::UnresolvedImport => write!(f, "unresolved import"),
            SemanticErrorKind::InvalidRange => write!(f, "invalid range"),
            SemanticErrorKind::TypeAnnotationMismatch => write!(f, "type annotation mismatch"),
        }
    }
}

// ─── Semantic Analyzer ───────────────────────────────────────────────────────

/// Performs semantic analysis on a parsed SoniType program.
pub struct SemanticAnalyzer {
    symbols: SymbolTable,
    errors: Vec<SemanticError>,
    /// Data schemas indexed by name for mapping validation.
    data_schemas: HashMap<String, Vec<(String, Type)>>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            errors: Vec::new(),
            data_schemas: HashMap::new(),
        }
    }

    /// Analyze a program, returning errors found.
    pub fn analyze(&mut self, program: &Program) -> Result<SymbolTable, Vec<SemanticError>> {
        // First pass: collect data declarations
        for decl in &program.declarations {
            if let Declaration::DataDecl(dd) = decl {
                self.register_data_decl(dd);
            }
        }

        // Second pass: analyze all declarations
        for decl in &program.declarations {
            self.analyze_declaration(decl);
        }

        if self.errors.is_empty() {
            Ok(self.symbols.clone())
        } else {
            Err(self.errors.clone())
        }
    }

    fn register_data_decl(&mut self, dd: &DataDecl) {
        let fields: Vec<(String, Type)> = dd
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), f.ty.clone()))
            .collect();

        let prev = self.symbols.define(
            &dd.name.name,
            Symbol::DataSource {
                name: dd.name.name.clone(),
                fields: fields.clone(),
                span: dd.span,
            },
        );

        if prev.is_some() {
            self.errors.push(SemanticError::new(
                SemanticErrorKind::DuplicateDefinition,
                format!("duplicate data declaration: {}", dd.name.name),
                dd.span,
            ));
        }

        self.data_schemas.insert(dd.name.name.clone(), fields);
    }

    fn analyze_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::StreamDecl(sd) => self.analyze_stream_decl(sd),
            Declaration::MappingDecl(md) => self.analyze_mapping_decl(md),
            Declaration::ComposeDecl(cd) => self.analyze_compose_decl(cd),
            Declaration::DataDecl(_) => {} // already handled
            Declaration::LetBinding(lb) => self.analyze_let_binding(lb),
            Declaration::SpecDecl(sd) => self.analyze_spec_decl(sd),
            Declaration::ImportDecl(id) => self.analyze_import_decl(id),
        }
    }

    fn analyze_stream_decl(&mut self, sd: &StreamDecl) {
        // Check for duplicate
        if self.symbols.is_defined_in_current_scope(&sd.name.name) {
            self.errors.push(SemanticError::new(
                SemanticErrorKind::DuplicateDefinition,
                format!("duplicate stream: {}", sd.name.name),
                sd.span,
            ));
            return;
        }

        let valid_params = ["freq", "frequency", "timbre", "pan", "amplitude", "envelope", "duration"];
        let mut param_names = Vec::new();

        for param in &sd.expr.params {
            if !valid_params.contains(&param.name.name.as_str()) {
                self.errors.push(SemanticError::new(
                    SemanticErrorKind::InvalidStreamParam,
                    format!("unknown stream parameter: {}", param.name.name),
                    param.span,
                ));
            }
            param_names.push(param.name.name.clone());
            self.analyze_expr(&param.value);
        }

        self.symbols.define(
            &sd.name.name,
            Symbol::Stream {
                name: sd.name.name.clone(),
                params: param_names,
                span: sd.span,
            },
        );
    }

    fn analyze_mapping_decl(&mut self, md: &MappingDecl) {
        if self.symbols.is_defined_in_current_scope(&md.name.name) {
            self.errors.push(SemanticError::new(
                SemanticErrorKind::DuplicateDefinition,
                format!("duplicate mapping: {}", md.name.name),
                md.span,
            ));
            return;
        }

        // Validate source against schema
        let source_name = &md.expr.source.source.name;
        let field_name = &md.expr.source.field.name;

        if let Some(schema) = self.data_schemas.get(source_name) {
            let has_field = schema.iter().any(|(f, _)| f == field_name);
            if !has_field {
                self.errors.push(SemanticError::new(
                    SemanticErrorKind::MappingSchemaViolation,
                    format!(
                        "data source '{}' has no field '{}'; available: {:?}",
                        source_name,
                        field_name,
                        schema.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>()
                    ),
                    md.expr.source.span,
                ));
            }
        }
        // If data source not found, it might be defined later or imported

        // Validate mapping range
        if let Some((lo, hi)) = &md.expr.target.range {
            self.analyze_expr(lo);
            self.analyze_expr(hi);

            // Check range makes sense for numeric literals
            if let (
                Expr::Literal(Literal { value: LiteralValue::Float(lo_v), .. }),
                Expr::Literal(Literal { value: LiteralValue::Float(hi_v), .. }),
            ) = (lo.as_ref(), hi.as_ref()) {
                if lo_v >= hi_v {
                    self.errors.push(SemanticError::new(
                        SemanticErrorKind::InvalidRange,
                        format!("mapping range is empty or inverted: {lo_v}..{hi_v}"),
                        md.expr.target.span,
                    ));
                }
            }

            if let (
                Expr::Literal(Literal { value: LiteralValue::Int(lo_v), .. }),
                Expr::Literal(Literal { value: LiteralValue::Int(hi_v), .. }),
            ) = (lo.as_ref(), hi.as_ref()) {
                if lo_v >= hi_v {
                    self.errors.push(SemanticError::new(
                        SemanticErrorKind::InvalidRange,
                        format!("mapping range is empty or inverted: {lo_v}..{hi_v}"),
                        md.expr.target.span,
                    ));
                }
            }
        }

        self.symbols.define(
            &md.name.name,
            Symbol::Mapping {
                name: md.name.name.clone(),
                source_field: format!("{}.{}", source_name, field_name),
                target_param: format!("{}", md.expr.target.param),
                span: md.span,
            },
        );
    }

    fn analyze_compose_decl(&mut self, cd: &ComposeDecl) {
        if self.symbols.is_defined_in_current_scope(&cd.name.name) {
            self.errors.push(SemanticError::new(
                SemanticErrorKind::DuplicateDefinition,
                format!("duplicate compose: {}", cd.name.name),
                cd.span,
            ));
            return;
        }

        for stream_expr in &cd.expr.streams {
            self.analyze_expr(stream_expr);
        }

        self.symbols.define(
            &cd.name.name,
            Symbol::Variable {
                name: cd.name.name.clone(),
                ty: None,
                mutable: false,
                span: cd.span,
            },
        );
    }

    fn analyze_let_binding(&mut self, lb: &LetBinding) {
        // Analyze the value first
        self.analyze_expr(&lb.value);

        // Then bind the pattern
        self.define_pattern(&lb.pattern);
    }

    fn analyze_spec_decl(&mut self, sd: &SpecDecl) {
        self.symbols.enter_scope();
        for decl in &sd.body {
            self.analyze_declaration(decl);
        }
        self.symbols.exit_scope();
    }

    fn analyze_import_decl(&mut self, id: &ImportDecl) {
        // Just record; actual resolution requires a module system
        for name in &id.names {
            self.symbols.define(
                &name.name,
                Symbol::Variable {
                    name: name.name.clone(),
                    ty: None,
                    mutable: false,
                    span: name.span,
                },
            );
        }
    }

    fn analyze_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Identifier(id) => {
                if self.symbols.resolve(&id.name).is_none() {
                    self.errors.push(SemanticError::new(
                        SemanticErrorKind::UndefinedVariable,
                        format!("undefined variable: {}", id.name),
                        id.span,
                    ));
                }
            }
            Expr::BinaryOp(b) => {
                self.analyze_expr(&b.lhs);
                self.analyze_expr(&b.rhs);
            }
            Expr::UnaryOp(u) => {
                self.analyze_expr(&u.operand);
            }
            Expr::FunctionCall(fc) => {
                self.analyze_expr(&fc.callee);
                for arg in &fc.args {
                    self.analyze_expr(arg);
                }
            }
            Expr::LetIn(li) => {
                self.analyze_expr(&li.value);
                self.symbols.enter_scope();
                self.define_pattern(&li.pattern);
                self.analyze_expr(&li.body);
                self.symbols.exit_scope();
            }
            Expr::IfThenElse(ite) => {
                self.analyze_expr(&ite.condition);
                self.analyze_expr(&ite.then_branch);
                self.analyze_expr(&ite.else_branch);
            }
            Expr::Lambda(lam) => {
                self.symbols.enter_scope();
                for p in &lam.params {
                    self.symbols.define(
                        &p.name.name,
                        Symbol::Variable {
                            name: p.name.name.clone(),
                            ty: None,
                            mutable: false,
                            span: p.name.span,
                        },
                    );
                }
                self.analyze_expr(&lam.body);
                self.symbols.exit_scope();
            }
            Expr::PipeOperator(p) => {
                self.analyze_expr(&p.lhs);
                self.analyze_expr(&p.rhs);
            }
            Expr::FieldAccess(fa) => {
                self.analyze_expr(&fa.object);
            }
            Expr::Grouped(inner, _) => {
                self.analyze_expr(inner);
            }
            Expr::StreamLiteral(s) => {
                for p in &s.params {
                    self.analyze_expr(&p.value);
                }
            }
            Expr::Compose(c) => {
                for s in &c.streams {
                    self.analyze_expr(s);
                }
            }
            Expr::WithClause(w) => {
                self.analyze_expr(&w.expr);
                for c in &w.constraints {
                    self.analyze_expr(&c.value);
                }
            }
            Expr::WhereClause(w) => {
                self.analyze_expr(&w.expr);
                for b in &w.bindings {
                    self.analyze_expr(&b.value);
                }
            }
            Expr::Literal(_) | Expr::MappingLiteral(_) => {}
        }
    }

    fn define_pattern(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Variable(id) => {
                if self.symbols.is_defined_in_current_scope(&id.name) {
                    self.errors.push(SemanticError::new(
                        SemanticErrorKind::DuplicateDefinition,
                        format!("duplicate binding: {}", id.name),
                        id.span,
                    ));
                }
                self.symbols.define(
                    &id.name,
                    Symbol::Variable {
                        name: id.name.clone(),
                        ty: None,
                        mutable: false,
                        span: id.span,
                    },
                );
            }
            Pattern::Tuple(pats, _) => {
                for p in pats {
                    self.define_pattern(p);
                }
            }
            Pattern::Wildcard(_) => {}
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn analyze_ok(src: &str) -> SymbolTable {
        let tokens = lex(src).expect("lex failed");
        let prog = parse(tokens).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&prog).expect("semantic analysis failed")
    }

    fn analyze_err(src: &str) -> Vec<SemanticError> {
        let tokens = lex(src).expect("lex failed");
        let prog = parse(tokens).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&prog).unwrap_err()
    }

    #[test]
    fn test_simple_let_binding() {
        let st = analyze_ok("let x = 42");
        assert!(st.resolve("x").is_some());
    }

    #[test]
    fn test_undefined_variable() {
        let errors = analyze_err("let x = y");
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::UndefinedVariable));
    }

    #[test]
    fn test_duplicate_definition() {
        let errors = analyze_err("let x = 1; let x = 2");
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::DuplicateDefinition));
    }

    #[test]
    fn test_data_decl_creates_symbol() {
        let st = analyze_ok("data temps = { temperature: Float, pressure: Float }");
        assert!(st.resolve("temps").is_some());
    }

    #[test]
    fn test_stream_decl() {
        let st = analyze_ok(r#"stream s = { freq: 440.0, timbre: "sine" }"#);
        assert!(st.resolve("s").is_some());
    }

    #[test]
    fn test_invalid_stream_param() {
        let errors = analyze_err(r#"stream s = { invalid_param: 42 }"#);
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::InvalidStreamParam));
    }

    #[test]
    fn test_mapping_schema_violation() {
        let errors = analyze_err(
            "data d = { temperature: Float }; mapping m = d.nonexistent -> pitch(200..800)",
        );
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::MappingSchemaViolation));
    }

    #[test]
    fn test_mapping_valid() {
        let st = analyze_ok(
            "data d = { temperature: Float }; mapping m = d.temperature -> pitch(200..800)",
        );
        assert!(st.resolve("m").is_some());
    }

    #[test]
    fn test_inverted_range() {
        let errors = analyze_err(
            "data d = { temperature: Float }; mapping m = d.temperature -> pitch(800..200)",
        );
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::InvalidRange));
    }

    #[test]
    fn test_let_in_scoping() {
        let st = analyze_ok("let x = let y = 1 in y");
        assert!(st.resolve("x").is_some());
        // y should not be visible outside the let-in
        assert!(st.resolve("y").is_none());
    }

    #[test]
    fn test_symbol_table_scope_depth() {
        let mut st = SymbolTable::new();
        assert_eq!(st.depth(), 1);
        st.enter_scope();
        assert_eq!(st.depth(), 2);
        st.exit_scope();
        assert_eq!(st.depth(), 1);
    }

    #[test]
    fn test_symbol_table_shadowing() {
        let mut st = SymbolTable::new();
        st.define("x", Symbol::Variable {
            name: "x".into(),
            ty: None,
            mutable: false,
            span: Span::dummy(),
        });
        st.enter_scope();
        st.define("x", Symbol::Variable {
            name: "x".into(),
            ty: Some(Type::Named("Int".into(), Span::dummy())),
            mutable: false,
            span: Span::dummy(),
        });
        // Inner scope x
        let sym = st.resolve("x").unwrap();
        match sym {
            Symbol::Variable { ty: Some(_), .. } => {} // has type annotation (inner)
            _ => panic!("expected inner x"),
        }
        st.exit_scope();
        // Back to outer x
        let sym = st.resolve("x").unwrap();
        match sym {
            Symbol::Variable { ty: None, .. } => {} // no type annotation (outer)
            _ => panic!("expected outer x"),
        }
    }

    #[test]
    fn test_compose_references_checked() {
        // s1 and s2 not defined → should produce errors
        let errors = analyze_err("compose c = { s1 || s2 }");
        assert!(errors.iter().any(|e| e.kind == SemanticErrorKind::UndefinedVariable));
    }

    #[test]
    fn test_import_defines_names() {
        let st = analyze_ok(r#"import foo from "bar""#);
        assert!(st.resolve("foo").is_some());
    }
}
