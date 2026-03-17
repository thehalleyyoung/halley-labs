//! Recursive-descent parser for the SoniType DSL.
//!
//! Consumes a token stream and produces an AST. Uses Pratt parsing for
//! operator precedence in expressions. Includes error recovery by
//! synchronising on statement boundaries.

use crate::ast::*;
use crate::token::{Span, Token, TokenKind};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Parse Errors ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self { message: message.into(), span }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}

// ─── Parser ──────────────────────────────────────────────────────────────────

/// Recursive-descent parser for the SoniType DSL.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, errors: Vec::new() }
    }

    // ── Token navigation ─────────────────────────────────────────────────────

    fn peek(&self) -> &TokenKind {
        self.tokens.get(self.pos).map(|t| &t.kind).unwrap_or(&TokenKind::Eof)
    }

    fn peek_token(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| self.tokens.last().unwrap())
    }

    fn current_span(&self) -> Span {
        self.peek_token().span
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError::new(
                format!("expected {kind}, found {}", self.peek()),
                self.current_span(),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<Identifier, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let span = self.current_span();
                self.advance();
                Ok(Identifier::new(name, span))
            }
            _ => Err(ParseError::new(
                format!("expected identifier, found {}", self.peek()),
                self.current_span(),
            )),
        }
    }

    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(kind)
    }

    fn at_end(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    fn error(&mut self, msg: impl Into<String>) -> ParseError {
        let e = ParseError::new(msg, self.current_span());
        self.errors.push(e.clone());
        e
    }

    /// Synchronise: skip tokens until we find a statement boundary.
    fn synchronize(&mut self) {
        while !self.at_end() {
            match self.peek() {
                TokenKind::Stream
                | TokenKind::Mapping
                | TokenKind::Compose
                | TokenKind::Data
                | TokenKind::Let
                | TokenKind::Import
                | TokenKind::Export
                | TokenKind::Spec
                | TokenKind::Semicolon => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // ── Program ──────────────────────────────────────────────────────────────

    fn parse_program(&mut self) -> Program {
        let start = self.current_span();
        let mut declarations = Vec::new();

        while !self.at_end() {
            match self.parse_declaration() {
                Ok(decl) => declarations.push(decl),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
            // Optional semicolons between declarations
            while self.check(&TokenKind::Semicolon) {
                self.advance();
            }
        }

        let end = self.current_span();
        Program::new(declarations, start.merge(end))
    }

    // ── Declarations ─────────────────────────────────────────────────────────

    fn parse_declaration(&mut self) -> Result<Declaration, ParseError> {
        let exported = if self.check(&TokenKind::Export) {
            self.advance();
            true
        } else {
            false
        };

        match self.peek() {
            TokenKind::Stream => self.parse_stream_decl(exported).map(Declaration::StreamDecl),
            TokenKind::Mapping => self.parse_mapping_decl(exported).map(Declaration::MappingDecl),
            TokenKind::Compose => self.parse_compose_decl(exported).map(Declaration::ComposeDecl),
            TokenKind::Data => self.parse_data_decl().map(Declaration::DataDecl),
            TokenKind::Let => self.parse_let_binding(exported).map(Declaration::LetBinding),
            TokenKind::Spec => self.parse_spec_decl().map(Declaration::SpecDecl),
            TokenKind::Import => self.parse_import_decl().map(Declaration::ImportDecl),
            _ => Err(self.error(format!("expected declaration, found {}", self.peek()))),
        }
    }

    fn parse_stream_decl(&mut self, exported: bool) -> Result<StreamDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Stream)?;
        let name = self.expect_ident()?;

        let type_annotation = self.try_parse_type_annotation()?;
        self.expect(&TokenKind::Eq)?;
        let expr = self.parse_stream_expr()?;

        let end_span = expr.span;
        Ok(StreamDecl { name, expr, type_annotation, exported, span: start.merge(end_span) })
    }

    fn parse_mapping_decl(&mut self, exported: bool) -> Result<MappingDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Mapping)?;
        let name = self.expect_ident()?;

        let type_annotation = self.try_parse_type_annotation()?;
        self.expect(&TokenKind::Eq)?;
        let expr = self.parse_mapping_expr()?;

        let end_span = expr.span;
        Ok(MappingDecl { name, expr, type_annotation, exported, span: start.merge(end_span) })
    }

    fn parse_compose_decl(&mut self, exported: bool) -> Result<ComposeDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Compose)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Eq)?;
        let expr = self.parse_compose_expr()?;

        let where_clause = if self.check(&TokenKind::Where) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        let with_clause = if self.check(&TokenKind::With) {
            Some(self.parse_with_clause()?)
        } else {
            None
        };

        let end = self.current_span();
        Ok(ComposeDecl {
            name,
            expr,
            where_clause,
            with_clause,
            exported,
            span: start.merge(end),
        })
    }

    fn parse_data_decl(&mut self) -> Result<DataDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Data)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Eq)?;
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_end() {
            let fname = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_type()?;
            fields.push(DataField {
                span: fname.span.merge(ty.span()),
                name: fname,
                ty,
            });
            if !self.check(&TokenKind::RBrace) {
                self.expect(&TokenKind::Comma)?;
            }
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;

        Ok(DataDecl { name, fields, source: None, span: start.merge(end) })
    }

    fn parse_let_binding(&mut self, exported: bool) -> Result<LetBinding, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Let)?;
        let pattern = self.parse_pattern()?;
        let type_annotation = self.try_parse_type_annotation()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expression()?;
        let end_span = value.span();
        Ok(LetBinding {
            pattern,
            type_annotation,
            value: Box::new(value),
            exported,
            span: start.merge(end_span),
        })
    }

    fn parse_spec_decl(&mut self) -> Result<SpecDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Spec)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_end() {
            match self.parse_declaration() {
                Ok(decl) => body.push(decl),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
            while self.check(&TokenKind::Semicolon) {
                self.advance();
            }
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;
        Ok(SpecDecl { name, body, span: start.merge(end) })
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Import)?;

        let mut names = Vec::new();
        // `import name from "path"` or `import "path"`
        if self.check(&TokenKind::Ident("".into())) {
            names.push(self.expect_ident()?);
            while self.check(&TokenKind::Comma) {
                self.advance();
                names.push(self.expect_ident()?);
            }
            // expect "from" as ident
            let from_kw = self.expect_ident()?;
            if from_kw.name != "from" {
                return Err(ParseError::new("expected 'from'", from_kw.span));
            }
        }

        let path_tok = self.expect(&TokenKind::StringLit(String::new()))?;
        let path = match &path_tok.kind {
            TokenKind::StringLit(s) => s.clone(),
            _ => unreachable!(),
        };

        let end = path_tok.span;
        Ok(ImportDecl { path, names, span: start.merge(end) })
    }

    // ── Stream expression ────────────────────────────────────────────────────

    fn parse_stream_expr(&mut self) -> Result<StreamExpr, ParseError> {
        let start = self.current_span();

        // Accept both `stream { ... }` (keyword already consumed) and `{ ... }`
        if self.check(&TokenKind::Stream) {
            self.advance();
        }
        self.expect(&TokenKind::LBrace)?;

        let mut params = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_end() {
            let pname = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expression()?;
            let pspan = pname.span.merge(value.span());
            params.push(StreamParam { name: pname, value, span: pspan });
            if !self.check(&TokenKind::RBrace) {
                self.expect(&TokenKind::Comma)?;
            }
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;
        Ok(StreamExpr { params, span: start.merge(end) })
    }

    // ── Mapping expression ───────────────────────────────────────────────────

    fn parse_mapping_expr(&mut self) -> Result<MappingExpr, ParseError> {
        let start = self.current_span();

        // Parse source: `data.field` or `ident.field`
        let source_name = self.expect_ident().or_else(|_| {
            // could also be the `data` keyword
            if self.check(&TokenKind::Data) {
                let span = self.current_span();
                self.advance();
                Ok(Identifier::new("data", span))
            } else {
                Err(ParseError::new("expected data source", self.current_span()))
            }
        })?;
        self.expect(&TokenKind::Dot)?;
        let field_name = self.expect_ident()?;
        let source = DataRef {
            span: source_name.span.merge(field_name.span),
            source: source_name,
            field: field_name,
        };

        self.expect(&TokenKind::Arrow)?;

        // Parse target: `param(lo..hi)` or just `param`
        let target = self.parse_mapping_target()?;

        let end_span = target.span;
        Ok(MappingExpr { source, target, span: start.merge(end_span) })
    }

    fn parse_mapping_target(&mut self) -> Result<MappingTarget, ParseError> {
        let start = self.current_span();
        let param_name = self.expect_ident()?;

        let param = match param_name.name.as_str() {
            "pitch" => AudioParamKind::Pitch,
            "timbre" => AudioParamKind::Timbre,
            "pan" => AudioParamKind::Pan,
            "amplitude" => AudioParamKind::Amplitude,
            "duration" => AudioParamKind::Duration,
            _ => return Err(ParseError::new(
                format!("unknown audio parameter: {}", param_name.name),
                param_name.span,
            )),
        };

        let range = if self.check(&TokenKind::LParen) {
            self.advance();
            let lo = self.parse_additive_expr()?;
            self.expect(&TokenKind::DotDot)?;
            let hi = self.parse_additive_expr()?;
            self.expect(&TokenKind::RParen)?;
            Some((Box::new(lo), Box::new(hi)))
        } else {
            None
        };

        let end = self.current_span();
        Ok(MappingTarget { param, range, span: start.merge(end) })
    }

    // ── Compose expression ───────────────────────────────────────────────────

    fn parse_compose_expr(&mut self) -> Result<ComposeExpr, ParseError> {
        let start = self.current_span();

        if self.check(&TokenKind::Compose) {
            self.advance();
        }
        self.expect(&TokenKind::LBrace)?;

        // Parse stream expressions separated by ||.
        // Use parse_and_expr so || is not consumed as logical OR.
        let mut streams = vec![self.parse_and_expr()?];
        while self.check(&TokenKind::PipePipe) {
            self.advance();
            streams.push(self.parse_and_expr()?);
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;
        Ok(ComposeExpr { streams, span: start.merge(end) })
    }

    // ── Where / With clauses ─────────────────────────────────────────────────

    fn parse_where_clause(&mut self) -> Result<WhereClause, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Where)?;
        self.expect(&TokenKind::LBrace)?;

        let mut bindings = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_end() {
            let name = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expression()?;
            let bspan = name.span.merge(value.span());
            bindings.push(WhereBinding { name, value, span: bspan });
            if !self.check(&TokenKind::RBrace) {
                self.expect(&TokenKind::Comma)?;
            }
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;
        Ok(WhereClause { bindings, span: start.merge(end) })
    }

    fn parse_with_clause(&mut self) -> Result<WithClause, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::With)?;
        self.expect(&TokenKind::LBrace)?;

        let mut constraints = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_end() {
            let name = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expression()?;
            let cspan = name.span.merge(value.span());
            constraints.push(Constraint { name, value, span: cspan });
            if !self.check(&TokenKind::RBrace) {
                self.expect(&TokenKind::Comma)?;
            }
        }

        let end = self.current_span();
        self.expect(&TokenKind::RBrace)?;
        Ok(WithClause { constraints, span: start.merge(end) })
    }

    // ── Patterns ─────────────────────────────────────────────────────────────

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        if self.check(&TokenKind::LParen) {
            let start = self.current_span();
            self.advance();
            let mut pats = Vec::new();
            if !self.check(&TokenKind::RParen) {
                pats.push(self.parse_pattern()?);
                while self.check(&TokenKind::Comma) {
                    self.advance();
                    pats.push(self.parse_pattern()?);
                }
            }
            let end = self.current_span();
            self.expect(&TokenKind::RParen)?;
            Ok(Pattern::Tuple(pats, start.merge(end)))
        } else if let TokenKind::Ident(name) = self.peek().clone() {
            if name == "_" {
                let span = self.current_span();
                self.advance();
                Ok(Pattern::Wildcard(span))
            } else {
                let id = self.expect_ident()?;
                Ok(Pattern::Variable(id))
            }
        } else {
            Err(self.error(format!("expected pattern, found {}", self.peek())))
        }
    }

    // ── Types ────────────────────────────────────────────────────────────────

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let base = self.parse_type_atom()?;

        // Function type: T -> U
        if self.check(&TokenKind::Arrow) {
            self.advance();
            let ret = self.parse_type()?;
            let span = base.span().merge(ret.span());
            return Ok(Type::Function(Box::new(base), Box::new(ret), span));
        }

        Ok(base)
    }

    fn parse_type_atom(&mut self) -> Result<Type, ParseError> {
        match self.peek().clone() {
            TokenKind::LParen => {
                let start = self.current_span();
                self.advance();
                let mut types = Vec::new();
                if !self.check(&TokenKind::RParen) {
                    types.push(self.parse_type()?);
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        types.push(self.parse_type()?);
                    }
                }
                let end = self.current_span();
                self.expect(&TokenKind::RParen)?;
                if types.len() == 1 {
                    Ok(types.into_iter().next().unwrap())
                } else {
                    Ok(Type::Tuple(types, start.merge(end)))
                }
            }
            kind if kind.is_type_keyword() => {
                let span = self.current_span();
                let name = format!("{}", self.peek());
                self.advance();

                // Parameterized: `Stream<Pitch>`
                if self.check(&TokenKind::Lt) {
                    self.advance();
                    let mut params = vec![self.parse_type()?];
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        params.push(self.parse_type()?);
                    }
                    let end = self.current_span();
                    self.expect(&TokenKind::Gt)?;
                    Ok(Type::Parameterized(name, params, span.merge(end)))
                } else {
                    Ok(Type::Named(name, span))
                }
            }
            TokenKind::Ident(ref name) if name.starts_with('\'') => {
                let span = self.current_span();
                let name = name.clone();
                self.advance();
                Ok(Type::Variable(name, span))
            }
            TokenKind::Ident(_) => {
                let id = self.expect_ident()?;
                Ok(Type::Named(id.name, id.span))
            }
            _ => Err(self.error(format!("expected type, found {}", self.peek()))),
        }
    }

    fn try_parse_type_annotation(&mut self) -> Result<Option<TypeAnnotation>, ParseError> {
        if self.check(&TokenKind::Colon) {
            let start = self.current_span();
            self.advance();
            let ty = self.parse_type()?;
            let span = start.merge(ty.span());
            Ok(Some(TypeAnnotation { ty, qualifier: None, span }))
        } else {
            Ok(None)
        }
    }

    // ── Expressions (Pratt parser) ───────────────────────────────────────────

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_pipe_expr()
    }

    fn parse_pipe_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_or_expr()?;
        while self.check(&TokenKind::PipeGt) {
            self.advance();
            let rhs = self.parse_or_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::PipeOperator(PipeExpr {
                lhs: Box::new(expr),
                rhs: Box::new(rhs),
                span,
            });
        }
        Ok(expr)
    }

    fn parse_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_and_expr()?;
        while self.check(&TokenKind::PipePipe) {
            self.advance();
            let rhs = self.parse_and_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op: BinOp::Or,
                lhs: Box::new(expr),
                rhs: Box::new(rhs),
                span,
            });
        }
        Ok(expr)
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_equality_expr()?;
        while self.check(&TokenKind::AmpAmp) {
            self.advance();
            let rhs = self.parse_equality_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op: BinOp::And,
                lhs: Box::new(expr),
                rhs: Box::new(rhs),
                span,
            });
        }
        Ok(expr)
    }

    fn parse_equality_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_comparison_expr()?;
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::Neq,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_comparison_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op, lhs: Box::new(expr), rhs: Box::new(rhs), span,
            });
        }
        Ok(expr)
    }

    fn parse_comparison_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_range_expr()?;
        loop {
            let op = match self.peek() {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::LtEq => BinOp::Lte,
                TokenKind::GtEq => BinOp::Gte,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_range_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op, lhs: Box::new(expr), rhs: Box::new(rhs), span,
            });
        }
        Ok(expr)
    }

    fn parse_range_expr(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_additive_expr()?;
        if self.check(&TokenKind::DotDot) {
            self.advance();
            let rhs = self.parse_additive_expr()?;
            let span = expr.span().merge(rhs.span());
            return Ok(Expr::BinaryOp(BinaryOp {
                op: BinOp::Range,
                lhs: Box::new(expr),
                rhs: Box::new(rhs),
                span,
            }));
        }
        Ok(expr)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_multiplicative_expr()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_multiplicative_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op, lhs: Box::new(expr), rhs: Box::new(rhs), span,
            });
        }
        Ok(expr)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary_expr()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary_expr()?;
            let span = expr.span().merge(rhs.span());
            expr = Expr::BinaryOp(BinaryOp {
                op, lhs: Box::new(expr), rhs: Box::new(rhs), span,
            });
        }
        Ok(expr)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            TokenKind::Minus => {
                let start = self.current_span();
                self.advance();
                let operand = self.parse_unary_expr()?;
                let span = start.merge(operand.span());
                Ok(Expr::UnaryOp(UnaryOp {
                    op: UnOp::Neg,
                    operand: Box::new(operand),
                    span,
                }))
            }
            TokenKind::Bang => {
                let start = self.current_span();
                self.advance();
                let operand = self.parse_unary_expr()?;
                let span = start.merge(operand.span());
                Ok(Expr::UnaryOp(UnaryOp {
                    op: UnOp::Not,
                    operand: Box::new(operand),
                    span,
                }))
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            if self.check(&TokenKind::LParen) {
                // Function call
                self.advance();
                let mut args = Vec::new();
                if !self.check(&TokenKind::RParen) {
                    args.push(self.parse_expression()?);
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        args.push(self.parse_expression()?);
                    }
                }
                let end = self.current_span();
                self.expect(&TokenKind::RParen)?;
                let span = expr.span().merge(end);
                expr = Expr::FunctionCall(FunctionCall {
                    callee: Box::new(expr),
                    args,
                    span,
                });
            } else if self.check(&TokenKind::Dot) {
                // Field access
                self.advance();
                let field = self.expect_ident()?;
                let span = expr.span().merge(field.span);
                expr = Expr::FieldAccess(FieldAccess {
                    object: Box::new(expr),
                    field,
                    span,
                });
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            TokenKind::IntLit(v) => {
                let span = self.current_span();
                self.advance();
                Ok(Expr::Literal(Literal {
                    value: LiteralValue::Int(v),
                    span,
                }))
            }
            TokenKind::FloatLit(v) => {
                let span = self.current_span();
                self.advance();
                Ok(Expr::Literal(Literal {
                    value: LiteralValue::Float(v),
                    span,
                }))
            }
            TokenKind::StringLit(ref v) => {
                let span = self.current_span();
                let val = v.clone();
                self.advance();
                Ok(Expr::Literal(Literal {
                    value: LiteralValue::String(val),
                    span,
                }))
            }
            TokenKind::True => {
                let span = self.current_span();
                self.advance();
                Ok(Expr::Literal(Literal {
                    value: LiteralValue::Bool(true),
                    span,
                }))
            }
            TokenKind::False => {
                let span = self.current_span();
                self.advance();
                Ok(Expr::Literal(Literal {
                    value: LiteralValue::Bool(false),
                    span,
                }))
            }
            TokenKind::Ident(_) => {
                let id = self.expect_ident()?;
                Ok(Expr::Identifier(id))
            }
            // Allow `data` keyword to be used as an identifier in expressions
            // (e.g., `data.temperature`)
            TokenKind::Data => {
                let span = self.current_span();
                self.advance();
                Ok(Expr::Identifier(Identifier::new("data", span)))
            }
            TokenKind::LParen => {
                let start = self.current_span();
                self.advance();
                let expr = self.parse_expression()?;
                let end = self.current_span();
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::Grouped(Box::new(expr), start.merge(end)))
            }
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Let => self.parse_let_in_expr(),
            _ => Err(self.error(format!("expected expression, found {}", self.peek()))),
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::If)?;
        let condition = self.parse_expression()?;
        self.expect(&TokenKind::Then)?;
        let then_branch = self.parse_expression()?;
        self.expect(&TokenKind::Else)?;
        let else_branch = self.parse_expression()?;
        let span = start.merge(else_branch.span());
        Ok(Expr::IfThenElse(IfThenElse {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            span,
        }))
    }

    fn parse_let_in_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        self.expect(&TokenKind::Let)?;
        let pattern = self.parse_pattern()?;
        let type_annotation = self.try_parse_type_annotation()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expression()?;
        self.expect(&TokenKind::In)?;
        let body = self.parse_expression()?;
        let span = start.merge(body.span());
        Ok(Expr::LetIn(LetIn {
            pattern,
            type_annotation,
            value: Box::new(value),
            body: Box::new(body),
            span,
        }))
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Parse a token stream into a Program AST.
pub fn parse(tokens: Vec<Token>) -> Result<Program, Vec<ParseError>> {
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program();
    if parser.errors.is_empty() {
        Ok(program)
    } else {
        Err(parser.errors)
    }
}

/// Parse a source string (lex + parse) into a Program AST.
pub fn parse_source(source: &str) -> Result<Program, Vec<String>> {
    let tokens = crate::lexer::lex(source).map_err(|errs| {
        errs.into_iter().map(|e| e.to_string()).collect::<Vec<_>>()
    })?;
    parse(tokens).map_err(|errs| errs.into_iter().map(|e| e.to_string()).collect())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse_ok(src: &str) -> Program {
        let tokens = lex(src).expect("lex failed");
        parse(tokens).expect("parse failed")
    }

    fn parse_expr_ok(src: &str) -> Expr {
        // Wrap in a let binding to make it a valid program
        let full = format!("let _x = {src}");
        let prog = parse_ok(&full);
        match &prog.declarations[0] {
            Declaration::LetBinding(lb) => *lb.value.clone(),
            _ => panic!("expected let binding"),
        }
    }

    #[test]
    fn test_parse_let_binding() {
        let prog = parse_ok("let x = 42");
        assert_eq!(prog.declarations.len(), 1);
        match &prog.declarations[0] {
            Declaration::LetBinding(lb) => {
                assert!(matches!(&lb.pattern, Pattern::Variable(id) if id.name == "x"));
            }
            _ => panic!("expected let binding"),
        }
    }

    #[test]
    fn test_parse_integer_literal() {
        let expr = parse_expr_ok("42");
        assert!(matches!(expr, Expr::Literal(Literal { value: LiteralValue::Int(42), .. })));
    }

    #[test]
    fn test_parse_float_literal() {
        let expr = parse_expr_ok("3.14");
        match expr {
            Expr::Literal(Literal { value: LiteralValue::Float(v), .. }) => {
                assert!((v - 3.14).abs() < 1e-10);
            }
            _ => panic!("expected float literal"),
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let expr = parse_expr_ok(r#""hello""#);
        assert!(
            matches!(expr, Expr::Literal(Literal { value: LiteralValue::String(ref s), .. }) if s == "hello")
        );
    }

    #[test]
    fn test_parse_binary_op() {
        let expr = parse_expr_ok("1 + 2 * 3");
        // Should parse as 1 + (2 * 3) due to precedence
        match expr {
            Expr::BinaryOp(b) => {
                assert_eq!(b.op, BinOp::Add);
                match *b.rhs {
                    Expr::BinaryOp(ref inner) => assert_eq!(inner.op, BinOp::Mul),
                    _ => panic!("expected multiplication on rhs"),
                }
            }
            _ => panic!("expected binary op"),
        }
    }

    #[test]
    fn test_parse_unary_negation() {
        let expr = parse_expr_ok("-5");
        assert!(matches!(expr, Expr::UnaryOp(UnaryOp { op: UnOp::Neg, .. })));
    }

    #[test]
    fn test_parse_if_then_else() {
        let expr = parse_expr_ok("if true then 1 else 0");
        assert!(matches!(expr, Expr::IfThenElse(_)));
    }

    #[test]
    fn test_parse_let_in() {
        let expr = parse_expr_ok("let y = 10 in y");
        assert!(matches!(expr, Expr::LetIn(_)));
    }

    #[test]
    fn test_parse_function_call() {
        let expr = parse_expr_ok("foo(1, 2, 3)");
        match expr {
            Expr::FunctionCall(fc) => assert_eq!(fc.args.len(), 3),
            _ => panic!("expected function call"),
        }
    }

    #[test]
    fn test_parse_field_access() {
        let expr = parse_expr_ok("data.temperature");
        assert!(matches!(expr, Expr::FieldAccess(_)));
    }

    #[test]
    fn test_parse_pipe_operator() {
        let expr = parse_expr_ok("x |> f |> g");
        match &expr {
            Expr::PipeOperator(p) => {
                assert!(matches!(*p.lhs, Expr::PipeOperator(_)));
            }
            _ => panic!("expected pipe operator"),
        }
    }

    #[test]
    fn test_parse_comparison() {
        let expr = parse_expr_ok("x > 0");
        assert!(matches!(expr, Expr::BinaryOp(BinaryOp { op: BinOp::Gt, .. })));
    }

    #[test]
    fn test_parse_stream_decl() {
        let prog = parse_ok(r#"stream s = { freq: 440.0, timbre: "sine" }"#);
        assert!(matches!(&prog.declarations[0], Declaration::StreamDecl(_)));
    }

    #[test]
    fn test_parse_data_decl() {
        let prog = parse_ok("data temps = { temperature: Float, pressure: Float }");
        match &prog.declarations[0] {
            Declaration::DataDecl(d) => {
                assert_eq!(d.name.name, "temps");
                assert_eq!(d.fields.len(), 2);
            }
            _ => panic!("expected data decl"),
        }
    }

    #[test]
    fn test_parse_mapping_decl() {
        let prog = parse_ok("mapping m = data.temperature -> pitch(200..800)");
        assert!(matches!(&prog.declarations[0], Declaration::MappingDecl(_)));
    }

    #[test]
    fn test_parse_compose_decl() {
        let prog = parse_ok("compose c = { s1 || s2 }");
        match &prog.declarations[0] {
            Declaration::ComposeDecl(d) => {
                assert_eq!(d.expr.streams.len(), 2);
            }
            _ => panic!("expected compose decl"),
        }
    }

    #[test]
    fn test_parse_import() {
        let prog = parse_ok(r#"import "stdlib/filters""#);
        assert!(matches!(&prog.declarations[0], Declaration::ImportDecl(_)));
    }

    #[test]
    fn test_parse_export_let() {
        let prog = parse_ok("export let x = 42");
        match &prog.declarations[0] {
            Declaration::LetBinding(lb) => assert!(lb.exported),
            _ => panic!("expected let binding"),
        }
    }

    #[test]
    fn test_parse_multiple_declarations() {
        let prog = parse_ok("let x = 1; let y = 2; let z = 3");
        assert_eq!(prog.declarations.len(), 3);
    }

    #[test]
    fn test_parse_boolean_literal() {
        let expr = parse_expr_ok("true");
        assert!(matches!(
            expr,
            Expr::Literal(Literal { value: LiteralValue::Bool(true), .. })
        ));
    }

    #[test]
    fn test_parse_grouped_expression() {
        let expr = parse_expr_ok("(1 + 2) * 3");
        match expr {
            Expr::BinaryOp(b) => {
                assert_eq!(b.op, BinOp::Mul);
                assert!(matches!(*b.lhs, Expr::Grouped(..)));
            }
            _ => panic!("expected mul"),
        }
    }

    #[test]
    fn test_parse_error_recovery() {
        let tokens = lex("let x = ; let y = 42").unwrap();
        let result = parse(tokens);
        // Should fail but parse the second declaration in error recovery
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_range_expr() {
        let expr = parse_expr_ok("1..100");
        assert!(matches!(expr, Expr::BinaryOp(BinaryOp { op: BinOp::Range, .. })));
    }

    #[test]
    fn test_parse_logical_and() {
        let expr = parse_expr_ok("a && b");
        assert!(matches!(expr, Expr::BinaryOp(BinaryOp { op: BinOp::And, .. })));
    }

    #[test]
    fn test_parse_spec_decl() {
        let prog = parse_ok("spec mySpec { let x = 1 }");
        assert!(matches!(&prog.declarations[0], Declaration::SpecDecl(_)));
    }
}
