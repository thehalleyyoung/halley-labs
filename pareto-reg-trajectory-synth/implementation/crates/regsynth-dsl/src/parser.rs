use crate::ast::*;
use crate::error::ParseError;
use crate::source_map::Span;
use crate::token::{Token, TokenKind};
use regsynth_types::{CompositionOp, FormalizabilityGrade, RiskLevel};

/// Recursive descent parser for the regulatory DSL.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    /// Parse the full program.
    pub fn parse_program(&mut self) -> (Program, Vec<ParseError>) {
        let start = self.current_span();
        let mut declarations = Vec::new();

        while !self.at_eof() {
            match self.parse_declaration() {
                Ok(decl) => declarations.push(decl),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
        }

        let end = self.current_span();
        let program = Program {
            declarations,
            span: start.merge(end),
        };
        (program, std::mem::take(&mut self.errors))
    }

    // ─── Declaration Parsing ────────────────────────────────────

    pub fn parse_declaration(&mut self) -> Result<Declaration, ParseError> {
        let start = self.current_span();
        match self.current_kind() {
            TokenKind::Jurisdiction => self.parse_jurisdiction_decl(start),
            TokenKind::Obligation => self.parse_obligation_decl(start),
            TokenKind::Permission => self.parse_permission_decl(start),
            TokenKind::Prohibition => self.parse_prohibition_decl(start),
            TokenKind::Framework => self.parse_framework_decl(start),
            TokenKind::Strategy => self.parse_strategy_decl(start),
            TokenKind::Cost => self.parse_cost_decl(start),
            TokenKind::Temporal => self.parse_temporal_decl(start),
            TokenKind::Mapping => self.parse_mapping_decl(start),
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "declaration (jurisdiction, obligation, permission, prohibition, framework, strategy, cost, temporal, mapping)",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    fn parse_jurisdiction_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Jurisdiction)?;
        let name = self.expect_string_or_ident()?;

        let parent = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.expect_string_or_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            match self.parse_declaration() {
                Ok(decl) => body.push(decl),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize_to_brace();
                }
            }
        }
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Jurisdiction(JurisdictionDecl {
                name,
                parent,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_obligation_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Obligation)?;
        let name = self.expect_ident()?;

        let jurisdiction = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.expect_string_or_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_obligation_body()?;
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Obligation(ObligationDecl {
                name,
                jurisdiction,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_permission_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Permission)?;
        let name = self.expect_ident()?;

        let jurisdiction = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.expect_string_or_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_obligation_body()?;
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Permission(PermissionDecl {
                name,
                jurisdiction,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_prohibition_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Prohibition)?;
        let name = self.expect_ident()?;

        let jurisdiction = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.expect_string_or_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_obligation_body()?;
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Prohibition(ProhibitionDecl {
                name,
                jurisdiction,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_framework_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Framework)?;
        let name = self.expect_string_or_ident()?;

        let jurisdiction = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.expect_string_or_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            match self.parse_declaration() {
                Ok(decl) => body.push(decl),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize_to_brace();
                }
            }
        }
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Framework(FrameworkDecl {
                name,
                jurisdiction,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_strategy_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Strategy)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut body = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            let item_start = self.current_span();
            let key = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expression()?;
            self.eat(TokenKind::Semicolon);
            let item_end = value.span;
            body.push(StrategyItem {
                key,
                value,
                span: item_start.merge(item_end),
            });
        }
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Strategy(StrategyDecl {
                name,
                body,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_cost_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Cost)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;
        let amount = self.parse_expression()?;
        let currency = if self.check_ident() {
            Some(self.expect_ident()?)
        } else {
            None
        };
        self.eat(TokenKind::Semicolon);
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Cost(CostDecl {
                name,
                amount,
                currency,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_temporal_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Temporal)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let start_date = if self.check_date() {
            Some(self.expect_date()?)
        } else {
            None
        };

        let end_date = if self.eat(TokenKind::Arrow) {
            if self.check_date() {
                Some(self.expect_date()?)
            } else {
                None
            }
        } else {
            None
        };

        self.eat(TokenKind::Semicolon);
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Temporal(TemporalDecl {
                name,
                start: start_date,
                end: end_date,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    fn parse_mapping_decl(&mut self, start: Span) -> Result<Declaration, ParseError> {
        self.expect(TokenKind::Mapping)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut entries = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            let entry_start = self.current_span();
            let from = self.parse_expression()?;
            self.expect(TokenKind::FatArrow)?;
            let to = self.parse_expression()?;
            self.eat(TokenKind::Semicolon);
            entries.push(MappingEntry {
                from,
                to,
                span: entry_start.merge(self.prev_span()),
            });
        }
        let end_span = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Declaration {
            kind: DeclarationKind::Mapping(MappingDecl {
                name,
                entries,
                span: start.merge(end_span),
            }),
            span: start.merge(end_span),
        })
    }

    // ─── Obligation Body ────────────────────────────────────────

    pub fn parse_obligation_body(&mut self) -> Result<ObligationBody, ParseError> {
        let start = self.current_span();
        let mut body = ObligationBody {
            span: start,
            ..Default::default()
        };

        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            match self.current_kind() {
                TokenKind::Requires => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    let expr = self.parse_expression()?;
                    body.conditions.push(expr);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Exempts => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    let expr = self.parse_expression()?;
                    body.exemptions.push(expr);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Temporal => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    let (ts, te) = self.parse_temporal_annotation()?;
                    body.temporal_start = ts;
                    body.temporal_end = te;
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::RiskLevel => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    body.risk_level = Some(self.parse_risk_level()?);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Domain => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    body.domain = Some(self.expect_string_or_ident()?);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Formalizability => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    body.formalizability = Some(self.parse_formalizability()?);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Article => {
                    body.article_refs.push(self.parse_article_ref_node()?);
                    self.eat(TokenKind::Semicolon);
                }
                TokenKind::Compose => {
                    self.advance();
                    self.eat(TokenKind::Colon);
                    let comp = self.parse_composition_node()?;
                    body.compositions.push(comp);
                    self.eat(TokenKind::Semicolon);
                }
                _ => {
                    // Try to parse as key: value field
                    if self.check_ident() {
                        let field_start = self.current_span();
                        let key = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let value = self.parse_expression()?;
                        self.eat(TokenKind::Semicolon);
                        body.extra_fields.push(FieldNode {
                            key,
                            value,
                            span: field_start.merge(self.prev_span()),
                        });
                    } else {
                        return Err(ParseError::unexpected_token(
                            self.current_span(),
                            "obligation body field",
                            &format!("{}", self.current_kind()),
                        ));
                    }
                }
            }
        }
        body.span = start.merge(self.current_span());
        Ok(body)
    }

    fn parse_temporal_annotation(&mut self) -> Result<(Option<String>, Option<String>), ParseError> {
        let start_date = if self.check_date() {
            Some(self.expect_date()?)
        } else {
            None
        };

        let end_date = if self.eat(TokenKind::Arrow) {
            if self.check_date() {
                Some(self.expect_date()?)
            } else {
                None
            }
        } else {
            None
        };

        Ok((start_date, end_date))
    }

    fn parse_risk_level(&mut self) -> Result<RiskLevel, ParseError> {
        let name = self.expect_ident()?;
        match name.as_str() {
            "minimal" => Ok(RiskLevel::Minimal),
            "limited" => Ok(RiskLevel::Limited),
            "high" => Ok(RiskLevel::High),
            "unacceptable" => Ok(RiskLevel::Unacceptable),
            _ => Err(ParseError::new(
                self.prev_span(),
                format!(
                    "invalid risk level '{}', expected minimal/limited/high/unacceptable",
                    name
                ),
            )),
        }
    }

    pub fn parse_formalizability(&mut self) -> Result<FormalizabilityGrade, ParseError> {
        match self.current_kind() {
            TokenKind::IntLit(n) => {
                let n = n;
                self.advance();
                FormalizabilityGrade::from_u8(n as u8).ok_or_else(|| {
                    ParseError::new(
                        self.prev_span(),
                        format!("invalid formalizability grade {}, expected 1-5", n),
                    )
                })
            }
            TokenKind::Ident(ref s) if s.starts_with('F') || s.starts_with('f') => {
                let s = s.clone();
                self.advance();
                let num_str = &s[1..];
                let n: u8 = num_str.parse().map_err(|_| {
                    ParseError::new(self.prev_span(), format!("invalid formalizability grade '{}'", s))
                })?;
                FormalizabilityGrade::from_u8(n).ok_or_else(|| {
                    ParseError::new(self.prev_span(), format!("invalid formalizability grade '{}'", s))
                })
            }
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "formalizability grade (1-5 or F1-F5)",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    fn parse_article_ref_node(&mut self) -> Result<ArticleRefNode, ParseError> {
        let start = self.current_span();
        self.expect(TokenKind::Article)?;
        let framework = self.expect_string_or_ident()?;
        let article = self.expect_string_or_ident()?;
        let paragraph = if self.check(TokenKind::LParen) {
            self.advance();
            let p = self.expect_string_or_ident()?;
            self.expect(TokenKind::RParen)?;
            Some(p)
        } else {
            None
        };
        Ok(ArticleRefNode {
            framework,
            article,
            paragraph,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_composition_node(&mut self) -> Result<CompositionNode, ParseError> {
        let start = self.current_span();
        let op = self.parse_composition_op()?;
        let operand = self.expect_ident()?;
        Ok(CompositionNode {
            op,
            operand,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_composition_op(&mut self) -> Result<CompositionOp, ParseError> {
        match self.current_kind() {
            TokenKind::Conjunction => {
                self.advance();
                Ok(CompositionOp::Conjunction)
            }
            TokenKind::Disjunction => {
                self.advance();
                Ok(CompositionOp::Disjunction)
            }
            TokenKind::OverrideOp => {
                self.advance();
                Ok(CompositionOp::Override)
            }
            TokenKind::ExceptionOp => {
                self.advance();
                Ok(CompositionOp::Exception)
            }
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "composition operator (⊗, ⊕, ▷, ⊘)",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    // ─── Expression Parsing ─────────────────────────────────────

    pub fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_binary_expr(0)
    }

    /// Precedence climbing parser for binary expressions.
    fn parse_binary_expr(&mut self, min_prec: u8) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary_expr()?;

        loop {
            // Check for composition operators first
            if let Some(cop) = self.try_composition_op() {
                self.advance();
                let right = self.parse_unary_expr()?;
                let span = left.span.merge(right.span);
                left = Expression {
                    kind: ExpressionKind::Composition {
                        op: cop,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                };
                continue;
            }

            let Some(op) = self.try_bin_op() else {
                break;
            };
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            self.advance();
            let next_min = if op.is_right_assoc() { prec } else { prec + 1 };
            let right = self.parse_binary_expr(next_min)?;
            let span = left.span.merge(right.span);
            left = Expression {
                kind: ExpressionKind::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }

        Ok(left)
    }

    fn try_bin_op(&self) -> Option<BinOp> {
        match self.current_kind() {
            TokenKind::Plus => Some(BinOp::Add),
            TokenKind::Minus => Some(BinOp::Sub),
            TokenKind::Star => Some(BinOp::Mul),
            TokenKind::Slash => Some(BinOp::Div),
            TokenKind::EqEq => Some(BinOp::Eq),
            TokenKind::NotEq => Some(BinOp::NotEq),
            TokenKind::Lt => Some(BinOp::Lt),
            TokenKind::Gt => Some(BinOp::Gt),
            TokenKind::LtEq => Some(BinOp::LtEq),
            TokenKind::GtEq => Some(BinOp::GtEq),
            TokenKind::And => Some(BinOp::And),
            TokenKind::Or => Some(BinOp::Or),
            TokenKind::Implies => Some(BinOp::Implies),
            _ => None,
        }
    }

    fn try_composition_op(&self) -> Option<CompositionOp> {
        match self.current_kind() {
            TokenKind::Conjunction => Some(CompositionOp::Conjunction),
            TokenKind::Disjunction => Some(CompositionOp::Disjunction),
            TokenKind::OverrideOp => Some(CompositionOp::Override),
            TokenKind::ExceptionOp => Some(CompositionOp::Exception),
            _ => None,
        }
    }

    fn parse_unary_expr(&mut self) -> Result<Expression, ParseError> {
        let start = self.current_span();
        match self.current_kind() {
            TokenKind::Not => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                let span = start.merge(operand.span);
                Ok(Expression {
                    kind: ExpressionKind::UnaryOp {
                        op: UnOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                let span = start.merge(operand.span);
                Ok(Expression {
                    kind: ExpressionKind::UnaryOp {
                        op: UnOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expression, ParseError> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            if self.check(TokenKind::Dot) {
                self.advance();
                let field = self.expect_ident()?;
                let span = expr.span.merge(self.prev_span());
                expr = Expression {
                    kind: ExpressionKind::FieldAccess {
                        object: Box::new(expr),
                        field,
                    },
                    span,
                };
            } else if self.check(TokenKind::LParen)
                && matches!(expr.kind, ExpressionKind::Variable(_))
            {
                // Function call
                let fn_name = if let ExpressionKind::Variable(name) = &expr.kind {
                    name.clone()
                } else {
                    unreachable!()
                };
                self.advance(); // skip (
                let mut args = Vec::new();
                while !self.check(TokenKind::RParen) && !self.at_eof() {
                    args.push(self.parse_expression()?);
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                let end = self.current_span();
                self.expect(TokenKind::RParen)?;
                expr = Expression {
                    kind: ExpressionKind::FunctionCall {
                        function: fn_name,
                        args,
                    },
                    span: expr.span.merge(end),
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> Result<Expression, ParseError> {
        let start = self.current_span();

        match self.current_kind() {
            TokenKind::IntLit(n) => {
                let n = n;
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Int(n)),
                    span: start,
                })
            }
            TokenKind::FloatLit(n) => {
                let n = n;
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Float(n)),
                    span: start,
                })
            }
            TokenKind::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Str(s)),
                    span: start,
                })
            }
            TokenKind::DateLit(ref d) => {
                let d = d.clone();
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Date(d)),
                    span: start,
                })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Bool(true)),
                    span: start,
                })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Bool(false)),
                    span: start,
                })
            }
            TokenKind::If => self.parse_if_then_else(),
            TokenKind::Forall => self.parse_quantifier(QuantifierKind::Forall),
            TokenKind::Exists => self.parse_quantifier(QuantifierKind::Exists),
            TokenKind::Article => self.parse_article_ref_expr(),
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(Expression {
                    kind: ExpressionKind::Variable(name),
                    span: start,
                })
            }
            _ => Err(ParseError::unexpected_token(
                start,
                "expression",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    fn parse_if_then_else(&mut self) -> Result<Expression, ParseError> {
        let start = self.current_span();
        self.expect(TokenKind::If)?;
        let condition = self.parse_expression()?;
        self.expect(TokenKind::Then)?;
        let then_branch = self.parse_expression()?;
        self.expect(TokenKind::Else)?;
        let else_branch = self.parse_expression()?;
        let span = start.merge(else_branch.span);
        Ok(Expression {
            kind: ExpressionKind::IfThenElse {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            span,
        })
    }

    fn parse_quantifier(&mut self, kind: QuantifierKind) -> Result<Expression, ParseError> {
        let start = self.current_span();
        self.advance(); // skip forall/exists
        let variable = self.expect_ident()?;

        let domain = if self.eat(TokenKind::Colon) {
            Some(Box::new(self.parse_primary_expr()?))
        } else {
            None
        };

        self.expect(TokenKind::Dot)?;
        let body = self.parse_expression()?;
        let span = start.merge(body.span);

        Ok(Expression {
            kind: ExpressionKind::Quantifier {
                kind,
                variable,
                domain,
                body: Box::new(body),
            },
            span,
        })
    }

    fn parse_article_ref_expr(&mut self) -> Result<Expression, ParseError> {
        let start = self.current_span();
        self.expect(TokenKind::Article)?;
        let framework = self.expect_string_or_ident()?;
        let article = self.expect_string_or_ident()?;
        let paragraph = if self.check(TokenKind::LParen) {
            self.advance();
            let p = self.expect_string_or_ident()?;
            self.expect(TokenKind::RParen)?;
            Some(p)
        } else {
            None
        };
        Ok(Expression {
            kind: ExpressionKind::ArticleRef {
                framework,
                article,
                paragraph,
            },
            span: start.merge(self.prev_span()),
        })
    }

    // ─── Token Helpers ──────────────────────────────────────────

    fn current_kind(&self) -> TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|t| t.span)
            .unwrap_or(Span::empty())
    }

    fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            Span::empty()
        }
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len() || self.current_kind() == TokenKind::Eof
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        std::mem::discriminant(&self.current_kind()) == std::mem::discriminant(&kind)
    }

    fn check_ident(&self) -> bool {
        matches!(self.current_kind(), TokenKind::Ident(_))
    }

    fn check_date(&self) -> bool {
        matches!(self.current_kind(), TokenKind::DateLit(_))
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<(), ParseError> {
        if self.check(kind.clone()) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::unexpected_token(
                self.current_span(),
                kind.description(),
                &format!("{}", self.current_kind()),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.current_kind() {
            TokenKind::Ident(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "identifier",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    fn expect_string_or_ident(&mut self) -> Result<String, ParseError> {
        match self.current_kind() {
            TokenKind::Ident(name) => {
                self.advance();
                Ok(name)
            }
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "string or identifier",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    fn expect_date(&mut self) -> Result<String, ParseError> {
        match self.current_kind() {
            TokenKind::DateLit(d) => {
                self.advance();
                Ok(d)
            }
            _ => Err(ParseError::unexpected_token(
                self.current_span(),
                "date literal",
                &format!("{}", self.current_kind()),
            )),
        }
    }

    /// Error recovery: skip tokens until we find a synchronization point.
    fn synchronize(&mut self) {
        while !self.at_eof() {
            if self.check(TokenKind::Semicolon) {
                self.advance();
                return;
            }
            match self.current_kind() {
                TokenKind::Jurisdiction
                | TokenKind::Obligation
                | TokenKind::Permission
                | TokenKind::Prohibition
                | TokenKind::Framework
                | TokenKind::Strategy
                | TokenKind::Cost
                | TokenKind::Temporal
                | TokenKind::Mapping => return,
                _ => self.advance(),
            }
        }
    }

    /// Error recovery: skip to closing brace.
    fn synchronize_to_brace(&mut self) {
        let mut depth = 0;
        while !self.at_eof() {
            match self.current_kind() {
                TokenKind::LBrace => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::RBrace => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                    self.advance();
                }
                _ => self.advance(),
            }
        }
    }
}

/// Convenience function: parse a source string (after lexing).
pub fn parse(tokens: Vec<Token>) -> (Program, Vec<ParseError>) {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse_src(src: &str) -> (Program, Vec<ParseError>) {
        let (tokens, lex_errors) = lex(src);
        assert!(lex_errors.is_empty(), "unexpected lex errors: {:?}", lex_errors);
        parse(tokens)
    }

    #[test]
    fn test_empty_program() {
        let (prog, errors) = parse_src("");
        assert!(errors.is_empty());
        assert!(prog.declarations.is_empty());
    }

    #[test]
    fn test_jurisdiction_decl() {
        let (prog, errors) = parse_src(r#"jurisdiction "EU" { }"#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        assert_eq!(prog.declarations.len(), 1);
        match &prog.declarations[0].kind {
            DeclarationKind::Jurisdiction(j) => {
                assert_eq!(j.name, "EU");
                assert!(j.parent.is_none());
            }
            other => panic!("expected jurisdiction, got {:?}", other),
        }
    }

    #[test]
    fn test_obligation_with_body() {
        let src = r#"
            obligation transparency {
                risk_level: high;
                formalizability: 3;
                requires: x and y;
                exempts: z;
                domain: "healthcare";
                temporal: #2024-01-01 -> #2025-12-31;
            }
        "#;
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        assert_eq!(prog.declarations.len(), 1);
        match &prog.declarations[0].kind {
            DeclarationKind::Obligation(o) => {
                assert_eq!(o.name, "transparency");
                assert_eq!(o.body.risk_level, Some(RiskLevel::High));
                assert_eq!(o.body.formalizability, Some(FormalizabilityGrade::F3));
                assert_eq!(o.body.conditions.len(), 1);
                assert_eq!(o.body.exemptions.len(), 1);
                assert_eq!(o.body.domain.as_deref(), Some("healthcare"));
                assert_eq!(o.body.temporal_start.as_deref(), Some("2024-01-01"));
                assert_eq!(o.body.temporal_end.as_deref(), Some("2025-12-31"));
            }
            other => panic!("expected obligation, got {:?}", other),
        }
    }

    #[test]
    fn test_expression_precedence() {
        let src = "obligation test { requires: a and b or c; }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            let expr = &o.body.conditions[0];
            // 'and' binds tighter than 'or', so this should be (a and b) or c
            match &expr.kind {
                ExpressionKind::BinaryOp { op: BinOp::Or, left, .. } => {
                    assert!(matches!(left.kind, ExpressionKind::BinaryOp { op: BinOp::And, .. }));
                }
                other => panic!("expected Or at top level, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_if_then_else() {
        let src = "obligation test { requires: if x then y else z; }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            assert!(matches!(o.body.conditions[0].kind, ExpressionKind::IfThenElse { .. }));
        }
    }

    #[test]
    fn test_quantifier() {
        let src = "obligation test { requires: forall x. x and true; }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            assert!(matches!(o.body.conditions[0].kind, ExpressionKind::Quantifier { .. }));
        }
    }

    #[test]
    fn test_function_call() {
        let src = "obligation test { requires: check(a, b); }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            match &o.body.conditions[0].kind {
                ExpressionKind::FunctionCall { function, args } => {
                    assert_eq!(function, "check");
                    assert_eq!(args.len(), 2);
                }
                other => panic!("expected fn call, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_field_access() {
        let src = "obligation test { requires: entity.name == high; }";
        let (_prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
    }

    #[test]
    fn test_composition_in_body() {
        let src = "obligation test { compose: ⊗ other_obl; }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            assert_eq!(o.body.compositions.len(), 1);
            assert_eq!(o.body.compositions[0].op, CompositionOp::Conjunction);
        }
    }

    #[test]
    fn test_nested_jurisdiction() {
        let src = r#"
            jurisdiction "EU" {
                obligation gdpr_consent {
                    formalizability: 2;
                }
            }
        "#;
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Jurisdiction(j) = &prog.declarations[0].kind {
            assert_eq!(j.body.len(), 1);
        }
    }

    #[test]
    fn test_strategy_decl() {
        let src = r#"
            strategy mitigation {
                approach: "risk_based";
                threshold: 0.5;
            }
        "#;
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        assert!(matches!(prog.declarations[0].kind, DeclarationKind::Strategy(_)));
    }

    #[test]
    fn test_error_recovery() {
        let src = r#"
            obligation good { requires: true; }
            999 bad stuff here ;
            obligation also_good { requires: false; }
        "#;
        let (prog, errors) = parse_src(src);
        // Should recover and parse at least the valid declarations
        assert!(!errors.is_empty());
        assert!(prog.declarations.len() >= 1);
    }

    #[test]
    fn test_permission_and_prohibition() {
        let src = r#"
            permission data_use { domain: "research"; }
            prohibition social_scoring { risk_level: unacceptable; }
        "#;
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        assert_eq!(prog.declarations.len(), 2);
        assert!(matches!(prog.declarations[0].kind, DeclarationKind::Permission(_)));
        assert!(matches!(prog.declarations[1].kind, DeclarationKind::Prohibition(_)));
    }

    #[test]
    fn test_formalizability_f_notation() {
        let src = "obligation test { formalizability: F2; }";
        let (prog, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        if let DeclarationKind::Obligation(o) = &prog.declarations[0].kind {
            assert_eq!(o.body.formalizability, Some(FormalizabilityGrade::F2));
        }
    }
}
