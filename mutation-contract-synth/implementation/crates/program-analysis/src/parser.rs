//! Lexer and recursive-descent parser for the loop-free imperative language.
//!
//! The language supports:
//! - Integer and boolean types, arrays
//! - Arithmetic, relational, and logical operators
//! - Variable declarations, assignments, if-else, return, assert
//! - Function definitions with typed parameters
//! - Programs as collections of functions

use shared_types::{
    ArithOp, Expression, Function, MutSpecError, Program, QfLiaType, RelOp, SourceLocation,
    SpanInfo, Statement, Variable,
};

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

/// Token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    IntLiteral(i64),
    BoolLiteral(bool),
    Identifier(String),
    Fn,
    Let,
    Var,
    If,
    Else,
    Return,
    Assert,
    Int,
    Bool,
    Void,
    Array,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    AmpAmp,
    PipePipe,
    Bang,
    ImpliesArrow,
    Eq,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    Arrow,
    Question,
    Eof,
}

impl Token {
    pub fn is_eof(&self) -> bool {
        matches!(self, Token::Eof)
    }

    pub fn describe(&self) -> &'static str {
        match self {
            Token::IntLiteral(_) => "integer literal",
            Token::BoolLiteral(_) => "boolean literal",
            Token::Identifier(_) => "identifier",
            Token::Fn => "'fn'",
            Token::Let => "'let'",
            Token::Var => "'var'",
            Token::If => "'if'",
            Token::Else => "'else'",
            Token::Return => "'return'",
            Token::Assert => "'assert'",
            Token::Int => "'int'",
            Token::Bool => "'bool'",
            Token::Void => "'void'",
            Token::Array => "'array'",
            Token::Plus => "'+'",
            Token::Minus => "'-'",
            Token::Star => "'*'",
            Token::Slash => "'/'",
            Token::Percent => "'%'",
            Token::EqEq => "'=='",
            Token::BangEq => "'!='",
            Token::Lt => "'<'",
            Token::Le => "'<='",
            Token::Gt => "'>'",
            Token::Ge => "'>='",
            Token::AmpAmp => "'&&'",
            Token::PipePipe => "'||'",
            Token::Bang => "'!'",
            Token::ImpliesArrow => "'==>'",
            Token::Eq => "'='",
            Token::LParen => "'('",
            Token::RParen => "')'",
            Token::LBrace => "'{'",
            Token::RBrace => "'}'",
            Token::LBracket => "'['",
            Token::RBracket => "']'",
            Token::Comma => "','",
            Token::Semicolon => "';'",
            Token::Colon => "':'",
            Token::Arrow => "'->'",
            Token::Question => "'?'",
            Token::Eof => "end of input",
        }
    }
}

/// A token with its source location.
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: SpanInfo,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// Tokenizer that converts source text into a stream of tokens.
pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    tokens: Vec<SpannedToken>,
    errors: Vec<MutSpecError>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            tokens: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Tokenize the entire input.
    pub fn tokenize(mut self) -> (Vec<SpannedToken>, Vec<MutSpecError>) {
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.source.len() {
                let loc = self.current_location();
                self.tokens.push(SpannedToken {
                    token: Token::Eof,
                    span: SpanInfo::new(loc.clone(), loc),
                });
                break;
            }
            match self.next_token() {
                Ok(st) => self.tokens.push(st),
                Err(e) => {
                    self.errors.push(e);
                    self.advance();
                }
            }
        }
        (self.tokens, self.errors)
    }

    fn current_location(&self) -> SourceLocation {
        SourceLocation::new("input", self.line, self.column)
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }
    fn peek_ahead(&self, n: usize) -> Option<char> {
        self.source.get(self.pos + n).copied()
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos < self.source.len() {
            let ch = self.source[self.pos];
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.peek().map_or(false, |c| c.is_whitespace()) {
                self.advance();
            }
            if self.peek() == Some('/') && self.peek_ahead(1) == Some('/') {
                while self.peek().map_or(false, |c| c != '\n') {
                    self.advance();
                }
                continue;
            }
            if self.peek() == Some('/') && self.peek_ahead(1) == Some('*') {
                self.advance();
                self.advance();
                let mut depth = 1;
                while depth > 0 {
                    match self.advance() {
                        Some('/') if self.peek() == Some('*') => {
                            self.advance();
                            depth += 1;
                        }
                        Some('*') if self.peek() == Some('/') => {
                            self.advance();
                            depth -= 1;
                        }
                        None => break,
                        _ => {}
                    }
                }
                continue;
            }
            break;
        }
    }

    fn next_token(&mut self) -> shared_types::Result<SpannedToken> {
        let start = self.current_location();
        let ch = self.peek().unwrap();

        if ch.is_ascii_digit() {
            return self.lex_number(start);
        }
        if ch.is_alphabetic() || ch == '_' {
            return self.lex_identifier(start);
        }

        match ch {
            '+' => {
                self.advance();
                Ok(self.mk(Token::Plus, start))
            }
            '-' => {
                self.advance();
                if self.peek() == Some('>') {
                    self.advance();
                    Ok(self.mk(Token::Arrow, start))
                } else {
                    Ok(self.mk(Token::Minus, start))
                }
            }
            '*' => {
                self.advance();
                Ok(self.mk(Token::Star, start))
            }
            '/' => {
                self.advance();
                Ok(self.mk(Token::Slash, start))
            }
            '%' => {
                self.advance();
                Ok(self.mk(Token::Percent, start))
            }
            '=' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        Ok(self.mk(Token::ImpliesArrow, start))
                    } else {
                        Ok(self.mk(Token::EqEq, start))
                    }
                } else {
                    Ok(self.mk(Token::Eq, start))
                }
            }
            '!' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(self.mk(Token::BangEq, start))
                } else {
                    Ok(self.mk(Token::Bang, start))
                }
            }
            '<' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(self.mk(Token::Le, start))
                } else {
                    Ok(self.mk(Token::Lt, start))
                }
            }
            '>' => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(self.mk(Token::Ge, start))
                } else {
                    Ok(self.mk(Token::Gt, start))
                }
            }
            '&' => {
                self.advance();
                if self.peek() == Some('&') {
                    self.advance();
                    Ok(self.mk(Token::AmpAmp, start))
                } else {
                    Err(MutSpecError::parse("Expected '&&'"))
                }
            }
            '|' => {
                self.advance();
                if self.peek() == Some('|') {
                    self.advance();
                    Ok(self.mk(Token::PipePipe, start))
                } else {
                    Err(MutSpecError::parse("Expected '||'"))
                }
            }
            '(' => {
                self.advance();
                Ok(self.mk(Token::LParen, start))
            }
            ')' => {
                self.advance();
                Ok(self.mk(Token::RParen, start))
            }
            '{' => {
                self.advance();
                Ok(self.mk(Token::LBrace, start))
            }
            '}' => {
                self.advance();
                Ok(self.mk(Token::RBrace, start))
            }
            '[' => {
                self.advance();
                Ok(self.mk(Token::LBracket, start))
            }
            ']' => {
                self.advance();
                Ok(self.mk(Token::RBracket, start))
            }
            ',' => {
                self.advance();
                Ok(self.mk(Token::Comma, start))
            }
            ';' => {
                self.advance();
                Ok(self.mk(Token::Semicolon, start))
            }
            ':' => {
                self.advance();
                Ok(self.mk(Token::Colon, start))
            }
            '?' => {
                self.advance();
                Ok(self.mk(Token::Question, start))
            }
            _ => Err(MutSpecError::parse(format!(
                "Unexpected character '{}'",
                ch
            ))),
        }
    }

    fn lex_number(&mut self, start: SourceLocation) -> shared_types::Result<SpannedToken> {
        let mut s = String::new();
        while self.peek().map_or(false, |c| c.is_ascii_digit()) {
            s.push(self.peek().unwrap());
            self.advance();
        }
        if self.peek().map_or(false, |c| c.is_alphabetic() || c == '_') {
            return Err(MutSpecError::parse(format!(
                "Invalid suffix on numeric literal"
            )));
        }
        let value: i64 = s
            .parse()
            .map_err(|_| MutSpecError::parse(format!("Integer '{}' out of range", s)))?;
        Ok(self.mk(Token::IntLiteral(value), start))
    }

    fn lex_identifier(&mut self, start: SourceLocation) -> shared_types::Result<SpannedToken> {
        let mut s = String::new();
        while self
            .peek()
            .map_or(false, |c| c.is_alphanumeric() || c == '_')
        {
            s.push(self.peek().unwrap());
            self.advance();
        }
        let token = match s.as_str() {
            "fn" => Token::Fn,
            "let" => Token::Let,
            "var" => Token::Var,
            "if" => Token::If,
            "else" => Token::Else,
            "return" => Token::Return,
            "assert" => Token::Assert,
            "int" => Token::Int,
            "bool" => Token::Bool,
            "void" => Token::Void,
            "array" => Token::Array,
            "true" => Token::BoolLiteral(true),
            "false" => Token::BoolLiteral(false),
            _ => Token::Identifier(s),
        };
        Ok(self.mk(token, start))
    }

    fn mk(&self, token: Token, start: SourceLocation) -> SpannedToken {
        SpannedToken {
            token,
            span: SpanInfo::new(start, self.current_location()),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Recursive-descent parser using Pratt-style precedence climbing for expressions.
pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    errors: Vec<MutSpecError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    None = 0,
    Implies = 1,
    Or = 2,
    And = 3,
    Equality = 4,
    Comparison = 5,
    Addition = 6,
    Multiplication = 7,
    Unary = 8,
    Postfix = 9,
}

fn infix_precedence(token: &Token) -> Option<Precedence> {
    match token {
        Token::ImpliesArrow | Token::Question => Some(Precedence::Implies),
        Token::PipePipe => Some(Precedence::Or),
        Token::AmpAmp => Some(Precedence::And),
        Token::EqEq | Token::BangEq => Some(Precedence::Equality),
        Token::Lt | Token::Le | Token::Gt | Token::Ge => Some(Precedence::Comparison),
        Token::Plus | Token::Minus => Some(Precedence::Addition),
        Token::Star | Token::Slash | Token::Percent => Some(Precedence::Multiplication),
        Token::LBracket | Token::LParen => Some(Precedence::Postfix),
        _ => None,
    }
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Parser {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    /// Parse from a source string. Lexes then parses.
    pub fn parse_source(source: &str) -> shared_types::Result<Program> {
        let (tokens, lex_errors) = Lexer::new(source).tokenize();
        if let Some(e) = lex_errors.into_iter().next() {
            return Err(e);
        }
        let mut parser = Parser::new(tokens);
        parser.parse_program()
    }

    // -- Helpers ------------------------------------------------------------

    fn cur(&self) -> &SpannedToken {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }
    fn tok(&self) -> &Token {
        &self.cur().token
    }
    fn span(&self) -> SpanInfo {
        self.cur().span.clone()
    }
    fn at(&self, t: &Token) -> bool {
        std::mem::discriminant(self.tok()) == std::mem::discriminant(t)
    }
    fn eof(&self) -> bool {
        self.tok().is_eof()
    }

    fn bump(&mut self) -> SpannedToken {
        let t = self.cur().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> shared_types::Result<SpannedToken> {
        if self.at(expected) {
            Ok(self.bump())
        } else {
            Err(MutSpecError::parse(format!(
                "Expected {}, found {}",
                expected.describe(),
                self.tok().describe()
            )))
        }
    }

    fn expect_id(&mut self) -> shared_types::Result<(String, SpanInfo)> {
        if let Token::Identifier(_) = self.tok() {
            let st = self.bump();
            if let Token::Identifier(n) = st.token {
                Ok((n, st.span))
            } else {
                unreachable!()
            }
        } else {
            Err(MutSpecError::parse(format!(
                "Expected identifier, found {}",
                self.tok().describe()
            )))
        }
    }

    fn semi(&mut self) -> shared_types::Result<()> {
        self.expect(&Token::Semicolon)?;
        Ok(())
    }

    fn synchronize(&mut self) {
        while !self.eof() {
            match self.tok() {
                Token::Semicolon => {
                    self.bump();
                    return;
                }
                Token::Fn
                | Token::Let
                | Token::Var
                | Token::If
                | Token::Return
                | Token::Assert
                | Token::RBrace => return,
                _ => {
                    self.bump();
                }
            }
        }
    }

    // -- Top-level ----------------------------------------------------------

    pub fn parse_program(&mut self) -> shared_types::Result<Program> {
        let mut functions = Vec::new();
        while !self.eof() {
            match self.parse_function() {
                Ok(f) => functions.push(f),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                    while !self.eof() && !self.at(&Token::Fn) {
                        self.bump();
                    }
                }
            }
        }
        if !self.errors.is_empty() {
            return Err(self.errors.remove(0));
        }
        Ok(Program::new(functions))
    }

    pub fn parse_function(&mut self) -> shared_types::Result<Function> {
        let start = self.span();
        self.expect(&Token::Fn)?;
        let (name, _) = self.expect_id()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        let ret = if self.at(&Token::Arrow) {
            self.bump();
            self.parse_type()?
        } else {
            QfLiaType::Void
        };
        self.expect(&Token::LBrace)?;
        let body_stmts = self.parse_stmts()?;
        let end = self.span();
        self.expect(&Token::RBrace)?;
        let body = Statement::Sequence(body_stmts);
        Ok(Function::new(name, params, ret, body).with_span(SpanInfo::new(start.start, end.end)))
    }

    fn parse_params(&mut self) -> shared_types::Result<Vec<Variable>> {
        let mut ps = Vec::new();
        if self.at(&Token::RParen) {
            return Ok(ps);
        }
        ps.push(self.parse_param()?);
        while self.at(&Token::Comma) {
            self.bump();
            ps.push(self.parse_param()?);
        }
        Ok(ps)
    }

    fn parse_param(&mut self) -> shared_types::Result<Variable> {
        let (name, _span) = self.expect_id()?;
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        Ok(Variable::param(name, ty))
    }

    fn parse_type(&mut self) -> shared_types::Result<QfLiaType> {
        match self.tok().clone() {
            Token::Int => {
                self.bump();
                Ok(QfLiaType::Int)
            }
            Token::Bool => {
                self.bump();
                Ok(QfLiaType::Boolean)
            }
            Token::Void => {
                self.bump();
                Ok(QfLiaType::Void)
            }
            Token::Array => {
                self.bump();
                self.expect(&Token::Lt)?;
                let _inner = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(QfLiaType::IntArray)
            }
            _ => Err(MutSpecError::parse(format!(
                "Expected type, found {}",
                self.tok().describe()
            ))),
        }
    }

    // -- Statements ---------------------------------------------------------

    fn parse_stmts(&mut self) -> shared_types::Result<Vec<Statement>> {
        let mut stmts = Vec::new();
        while !self.at(&Token::RBrace) && !self.eof() {
            match self.parse_stmt() {
                Ok(s) => stmts.push(s),
                Err(e) => {
                    self.errors.push(e);
                    self.synchronize();
                }
            }
        }
        Ok(stmts)
    }

    pub fn parse_stmt(&mut self) -> shared_types::Result<Statement> {
        match self.tok().clone() {
            Token::Let | Token::Var => self.parse_var_decl(),
            Token::If => self.parse_if(),
            Token::Return => self.parse_return(),
            Token::Assert => self.parse_assert(),
            Token::LBrace => self.parse_block(),
            Token::Identifier(_) => self.parse_assign_or_call(),
            _ => Err(MutSpecError::parse(format!(
                "Expected statement, found {}",
                self.tok().describe()
            ))),
        }
    }

    fn parse_var_decl(&mut self) -> shared_types::Result<Statement> {
        let start = self.span();
        self.bump(); // let/var
        let (name, _) = self.expect_id()?;
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        let init = if self.at(&Token::Eq) {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end = self.span();
        self.semi()?;
        Ok(Statement::VarDecl {
            var: Variable::local(name, ty),
            init,
            span: Some(SpanInfo::new(start.start, end.end)),
        })
    }

    fn parse_if(&mut self) -> shared_types::Result<Statement> {
        let start = self.span();
        self.expect(&Token::If)?;
        self.expect(&Token::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;
        let then_stmts = self.parse_stmts()?;
        self.expect(&Token::RBrace)?;
        let else_b = if self.at(&Token::Else) {
            self.bump();
            if self.at(&Token::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                self.expect(&Token::LBrace)?;
                let s = self.parse_stmts()?;
                self.expect(&Token::RBrace)?;
                Some(Box::new(Statement::Sequence(s)))
            }
        } else {
            None
        };
        let end = self.span();
        Ok(Statement::IfElse {
            condition: cond,
            then_branch: Box::new(Statement::Sequence(then_stmts)),
            else_branch: else_b,
            span: Some(SpanInfo::new(start.start, end.end)),
        })
    }

    fn parse_return(&mut self) -> shared_types::Result<Statement> {
        let start = self.span();
        self.expect(&Token::Return)?;
        let val = if self.at(&Token::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        let end = self.span();
        self.semi()?;
        Ok(Statement::Return {
            value: val,
            span: Some(SpanInfo::new(start.start, end.end)),
        })
    }

    fn parse_assert(&mut self) -> shared_types::Result<Statement> {
        let start = self.span();
        self.expect(&Token::Assert)?;
        self.expect(&Token::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        let end = self.span();
        self.semi()?;
        Ok(Statement::Assert {
            condition: cond,
            message: None,
            span: Some(SpanInfo::new(start.start, end.end)),
        })
    }

    fn parse_block(&mut self) -> shared_types::Result<Statement> {
        self.expect(&Token::LBrace)?;
        let stmts = self.parse_stmts()?;
        self.expect(&Token::RBrace)?;
        Ok(Statement::Block(stmts))
    }

    fn parse_assign_or_call(&mut self) -> shared_types::Result<Statement> {
        let start = self.span();
        let (name, _) = self.expect_id()?;
        if self.at(&Token::LBracket) {
            // Array assignment: encode as Assign { target: "name[...]", value: ... }
            // Since there's no ArrayAssign variant, we encode as a regular assignment
            // with the array name as target and wrap value in context
            self.bump();
            let idx = self.parse_expr()?;
            self.expect(&Token::RBracket)?;
            self.expect(&Token::Eq)?;
            let val = self.parse_expr()?;
            let end = self.span();
            self.semi()?;
            Ok(Statement::Assign {
                target: name,
                value: Expression::FunctionCall {
                    name: "__array_store".to_string(),
                    args: vec![idx, val],
                },
                span: Some(SpanInfo::new(start.start, end.end)),
            })
        } else if self.at(&Token::Eq) {
            self.bump();
            let val = self.parse_expr()?;
            let end = self.span();
            self.semi()?;
            Ok(Statement::Assign {
                target: name,
                value: val,
                span: Some(SpanInfo::new(start.start, end.end)),
            })
        } else if self.at(&Token::LParen) {
            self.bump();
            let args = self.parse_args()?;
            self.expect(&Token::RParen)?;
            let end = self.span();
            self.semi()?;
            let call_span = SpanInfo::new(start.start, end.end);
            Ok(Statement::Assign {
                target: "_".into(),
                value: Expression::FunctionCall { name, args },
                span: Some(call_span),
            })
        } else {
            Err(MutSpecError::parse(format!(
                "Expected '=', '[', or '(' after '{}'",
                name
            )))
        }
    }

    // -- Expressions (Pratt) ------------------------------------------------

    pub fn parse_expr(&mut self) -> shared_types::Result<Expression> {
        self.pratt(Precedence::None)
    }

    fn pratt(&mut self, min_prec: Precedence) -> shared_types::Result<Expression> {
        let mut left = self.prefix()?;
        loop {
            if self.eof() {
                break;
            }
            let prec = match infix_precedence(self.tok()) {
                Some(p) if p > min_prec => p,
                _ => break,
            };
            left = self.infix(left, prec)?;
        }
        Ok(left)
    }

    fn prefix(&mut self) -> shared_types::Result<Expression> {
        match self.tok().clone() {
            Token::IntLiteral(v) => {
                self.bump();
                Ok(Expression::IntLiteral(v))
            }
            Token::BoolLiteral(v) => {
                self.bump();
                Ok(Expression::BoolLiteral(v))
            }
            Token::Identifier(name) => {
                self.bump();
                if self.at(&Token::LParen) {
                    self.bump();
                    let args = self.parse_args()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expression::FunctionCall { name, args })
                } else if self.at(&Token::LBracket) {
                    self.bump();
                    let idx = self.parse_expr()?;
                    self.expect(&Token::RBracket)?;
                    Ok(Expression::ArrayAccess {
                        array: Box::new(Expression::Var(name)),
                        index: Box::new(idx),
                    })
                } else {
                    Ok(Expression::Var(name))
                }
            }
            Token::Minus => {
                self.bump();
                let operand = self.pratt(Precedence::Unary)?;
                Ok(Expression::UnaryArith(Box::new(operand)))
            }
            Token::Bang => {
                self.bump();
                let operand = self.pratt(Precedence::Unary)?;
                Ok(Expression::LogicalNot(Box::new(operand)))
            }
            Token::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            _ => Err(MutSpecError::parse(format!(
                "Expected expression, found {}",
                self.tok().describe()
            ))),
        }
    }

    fn infix(&mut self, left: Expression, prec: Precedence) -> shared_types::Result<Expression> {
        let token = self.tok().clone();
        match token {
            Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent => {
                self.bump();
                let right = self.pratt(prec)?;
                let op = match token {
                    Token::Plus => ArithOp::Add,
                    Token::Minus => ArithOp::Sub,
                    Token::Star => ArithOp::Mul,
                    Token::Slash => ArithOp::Div,
                    Token::Percent => ArithOp::Mod,
                    _ => unreachable!(),
                };
                Ok(Expression::BinaryArith {
                    op,
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                })
            }
            Token::EqEq | Token::BangEq | Token::Lt | Token::Le | Token::Gt | Token::Ge => {
                self.bump();
                let right = self.pratt(prec)?;
                let op = match token {
                    Token::EqEq => RelOp::Eq,
                    Token::BangEq => RelOp::Ne,
                    Token::Lt => RelOp::Lt,
                    Token::Le => RelOp::Le,
                    Token::Gt => RelOp::Gt,
                    Token::Ge => RelOp::Ge,
                    _ => unreachable!(),
                };
                Ok(Expression::Relational {
                    op,
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                })
            }
            Token::AmpAmp => {
                self.bump();
                let right = self.pratt(prec)?;
                Ok(Expression::LogicalAnd(Box::new(left), Box::new(right)))
            }
            Token::PipePipe => {
                self.bump();
                let right = self.pratt(prec)?;
                Ok(Expression::LogicalOr(Box::new(left), Box::new(right)))
            }
            Token::ImpliesArrow => {
                // Desugar a ==> b into !a || b
                self.bump();
                let right = self.pratt(Precedence::None)?;
                Ok(Expression::LogicalOr(
                    Box::new(Expression::LogicalNot(Box::new(left))),
                    Box::new(right),
                ))
            }
            Token::Question => {
                self.bump();
                let then_e = self.parse_expr()?;
                self.expect(&Token::Colon)?;
                let else_e = self.pratt(Precedence::None)?;
                Ok(Expression::Conditional {
                    condition: Box::new(left),
                    then_expr: Box::new(then_e),
                    else_expr: Box::new(else_e),
                })
            }
            Token::LBracket => {
                self.bump();
                let idx = self.parse_expr()?;
                self.expect(&Token::RBracket)?;
                Ok(Expression::ArrayAccess {
                    array: Box::new(left),
                    index: Box::new(idx),
                })
            }
            Token::LParen => {
                if let Expression::Var(name) = left {
                    self.bump();
                    let args = self.parse_args()?;
                    self.expect(&Token::RParen)?;
                    Ok(Expression::FunctionCall { name, args })
                } else {
                    Err(MutSpecError::parse("Function call requires identifier"))
                }
            }
            _ => Err(MutSpecError::parse(format!(
                "Unexpected infix operator {}",
                token.describe()
            ))),
        }
    }

    fn parse_args(&mut self) -> shared_types::Result<Vec<Expression>> {
        let mut args = Vec::new();
        if self.at(&Token::RParen) {
            return Ok(args);
        }
        args.push(self.parse_expr()?);
        while self.at(&Token::Comma) {
            self.bump();
            args.push(self.parse_expr()?);
        }
        Ok(args)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<Token> {
        let (toks, errs) = Lexer::new(src).tokenize();
        assert!(errs.is_empty(), "lex errors: {:?}", errs);
        toks.into_iter().map(|t| t.token).collect()
    }

    fn parse(src: &str) -> Program {
        Parser::parse_source(src).unwrap()
    }
    fn parse_err(src: &str) -> MutSpecError {
        Parser::parse_source(src).unwrap_err()
    }

    /// Helper: extract the inner Vec<Statement> from a function body (which is always Sequence).
    fn body_stmts(func: &Function) -> &Vec<Statement> {
        match &func.body {
            Statement::Sequence(stmts) => stmts,
            _ => panic!("expected Sequence body"),
        }
    }

    #[test]
    fn lex_empty() {
        assert_eq!(lex(""), vec![Token::Eof]);
    }

    #[test]
    fn lex_ints() {
        assert_eq!(
            lex("0 42 1000"),
            vec![
                Token::IntLiteral(0),
                Token::IntLiteral(42),
                Token::IntLiteral(1000),
                Token::Eof
            ]
        );
    }

    #[test]
    fn lex_bools() {
        assert_eq!(
            lex("true false"),
            vec![
                Token::BoolLiteral(true),
                Token::BoolLiteral(false),
                Token::Eof
            ]
        );
    }

    #[test]
    fn lex_keywords() {
        let t = lex("fn let var if else return assert int bool void array foo x1");
        assert_eq!(t[0], Token::Fn);
        assert_eq!(t[11], Token::Identifier("foo".into()));
        assert_eq!(t[12], Token::Identifier("x1".into()));
    }

    #[test]
    fn lex_ops() {
        let t = lex("+ - * / % == != < <= > >= && || ! ==> = -> ?");
        assert_eq!(t.len(), 19); // 18 ops + Eof
        assert_eq!(t[14], Token::ImpliesArrow);
    }

    #[test]
    fn lex_delims() {
        let t = lex("( ) { } [ ] , ; :");
        assert_eq!(t[0], Token::LParen);
        assert_eq!(t[8], Token::Colon);
    }

    #[test]
    fn lex_line_comment() {
        assert_eq!(
            lex("x // comment\ny"),
            vec![
                Token::Identifier("x".into()),
                Token::Identifier("y".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn lex_block_comment() {
        assert_eq!(
            lex("x /* c */ y"),
            vec![
                Token::Identifier("x".into()),
                Token::Identifier("y".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn lex_nested_comment() {
        assert_eq!(
            lex("a /* /* inner */ */ b"),
            vec![
                Token::Identifier("a".into()),
                Token::Identifier("b".into()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn lex_locations() {
        let (t, _) = Lexer::new("x + 42").tokenize();
        assert_eq!(t[0].span.start.column, 1);
        assert_eq!(t[1].span.start.column, 3);
        assert_eq!(t[2].span.start.column, 5);
    }

    #[test]
    fn lex_multiline() {
        let (t, _) = Lexer::new("x\ny\nz").tokenize();
        assert_eq!(t[0].span.start.line, 1);
        assert_eq!(t[1].span.start.line, 2);
        assert_eq!(t[2].span.start.line, 3);
    }

    #[test]
    fn parse_empty_fn() {
        let p = parse("fn main() -> void { }");
        assert_eq!(p.functions[0].name, "main");
        assert!(body_stmts(&p.functions[0]).is_empty());
    }

    #[test]
    fn parse_fn_params() {
        let p = parse("fn add(x: int, y: int) -> int { return x + y; }");
        assert_eq!(p.functions[0].params.len(), 2);
        assert_eq!(p.functions[0].return_type, QfLiaType::Int);
    }

    #[test]
    fn parse_var_decl() {
        let p = parse("fn f() -> void { let x: int = 42; }");
        let stmts = body_stmts(&p.functions[0]);
        assert!(matches!(&stmts[0], Statement::VarDecl { var, .. } if var.name == "x"));
    }

    #[test]
    fn parse_var_no_init() {
        let p = parse("fn f() -> void { var x: int; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::VarDecl { init, .. } => assert!(init.is_none()),
            _ => panic!("expected VarDecl"),
        }
    }

    #[test]
    fn parse_assign() {
        let p = parse("fn f() -> void { x = 10; }");
        let stmts = body_stmts(&p.functions[0]);
        assert!(matches!(&stmts[0], Statement::Assign { target, .. } if target == "x"));
    }

    #[test]
    fn parse_if_else() {
        let p = parse("fn f(x: int) -> int { if (x > 0) { return x; } else { return 0; } }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::IfElse {
                then_branch,
                else_branch,
                ..
            } => {
                // then_branch is a Box<Statement> (Sequence with 1 element)
                match then_branch.as_ref() {
                    Statement::Sequence(s) => assert_eq!(s.len(), 1),
                    _ => panic!("expected Sequence in then_branch"),
                }
                assert!(else_branch.is_some());
            }
            _ => panic!("expected IfElse"),
        }
    }

    #[test]
    fn parse_nested_if() {
        let p = parse("fn f(x: int) -> int { if (x > 0) { return 1; } else if (x < 0) { return -1; } else { return 0; } }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::IfElse { else_branch, .. } => {
                let eb = else_branch.as_ref().unwrap();
                assert!(matches!(eb.as_ref(), Statement::IfElse { .. }));
            }
            _ => panic!("expected IfElse"),
        }
    }

    #[test]
    fn parse_assert_stmt() {
        let p = parse("fn f(x: int) -> void { assert(x > 0); }");
        let stmts = body_stmts(&p.functions[0]);
        assert!(matches!(&stmts[0], Statement::Assert { .. }));
    }

    #[test]
    fn parse_return_void() {
        let p = parse("fn f() -> void { return; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value, .. } => assert!(value.is_none()),
            _ => panic!(),
        }
    }

    #[test]
    fn parse_block() {
        let p = parse("fn f() -> void { { let x: int = 1; x = 2; } }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Block(inner) => assert_eq!(inner.len(), 2),
            _ => panic!(),
        }
    }

    #[test]
    fn parse_arr_assign() {
        let p = parse("fn f() -> void { a[0] = 42; }");
        let stmts = body_stmts(&p.functions[0]);
        // Array assign is encoded as Assign with target "a"
        assert!(matches!(&stmts[0], Statement::Assign { target, .. } if target == "a"));
    }

    #[test]
    fn parse_arith_prec() {
        let p = parse("fn f(x: int, y: int, z: int) -> int { return x + y * z; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return {
                value:
                    Some(Expression::BinaryArith {
                        op: ArithOp::Add,
                        rhs,
                        ..
                    }),
                ..
            } => {
                assert!(matches!(
                    rhs.as_ref(),
                    Expression::BinaryArith {
                        op: ArithOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("expected Add(_, Mul)"),
        }
    }

    #[test]
    fn parse_cmp_and_logic() {
        let p = parse("fn f(x: int, y: int) -> bool { return x > 0 && y < 10; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::LogicalAnd(..)))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_implies() {
        // Implies is desugared to !a || b
        let p = parse("fn f(a: bool, b: bool) -> bool { return a ==> b; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::LogicalOr(..)))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_neg() {
        let p = parse("fn f(x: int) -> int { return -x; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::UnaryArith(..)))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_not() {
        let p = parse("fn f(b: bool) -> bool { return !b; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::LogicalNot(..)))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_call() {
        let p = parse("fn f(x: int) -> int { return add(x, 1); }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return {
                value: Some(Expression::FunctionCall { name, args, .. }),
                ..
            } => {
                assert_eq!(name, "add");
                assert_eq!(args.len(), 2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_ternary() {
        let p = parse("fn f(x: int) -> int { return x > 0 ? x : 0; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::Conditional { .. }))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_arr_access() {
        let p = parse("fn f(a: array<int>) -> int { return a[0]; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::ArrayAccess { .. }))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_parens() {
        let p = parse("fn f(x: int, y: int) -> int { return (x + y) * 2; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return {
                value:
                    Some(Expression::BinaryArith {
                        op: ArithOp::Mul,
                        lhs,
                        ..
                    }),
                ..
            } => {
                assert!(matches!(
                    lhs.as_ref(),
                    Expression::BinaryArith {
                        op: ArithOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_multi_fn() {
        let p = parse("fn a(x: int) -> int { return x; } fn b() -> int { return a(1); }");
        assert_eq!(p.functions.len(), 2);
    }

    #[test]
    fn parse_complex() {
        let p = parse(
            r#"
            fn abs(x: int) -> int { if (x >= 0) { return x; } else { return -x; } }
            fn max(a: int, b: int) -> int { if (a >= b) { return a; } else { return b; } }
            fn clamp(x: int, lo: int, hi: int) -> int {
                if (x < lo) { return lo; } else if (x > hi) { return hi; } else { return x; }
            }
        "#,
        );
        assert_eq!(p.functions.len(), 3);
    }

    #[test]
    fn parse_deep_expr() {
        let p =
            parse("fn f(a: int, b: int, c: int, d: int) -> int { return a + b * c - d / 2 % 3; }");
        assert_eq!(body_stmts(&p.functions[0]).len(), 1);
    }

    #[test]
    fn parse_logic_chain() {
        let p = parse("fn f(a: bool, b: bool, c: bool) -> bool { return a && b || c; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                assert!(matches!(e, Expression::LogicalOr(..)))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_arr_type() {
        let p = parse("fn f(a: array<int>) -> int { return a[0]; }");
        assert_eq!(p.functions[0].params[0].ty, QfLiaType::IntArray);
    }

    #[test]
    fn err_missing_semi() {
        assert!(matches!(
            parse_err("fn f() -> void { return 42 }"),
            MutSpecError::Parse { .. }
        ));
    }

    #[test]
    fn err_missing_paren() {
        assert!(matches!(
            parse_err("fn f( -> void { }"),
            MutSpecError::Parse { .. }
        ));
    }

    #[test]
    fn err_invalid_char() {
        assert!(matches!(
            parse_err("fn f() -> void { @ }"),
            MutSpecError::Parse { .. }
        ));
    }

    #[test]
    fn parse_multi_stmt() {
        let p = parse("fn f(x: int) -> int { let y: int = x + 1; let z: int = y * 2; assert(z > 0); return z; }");
        assert_eq!(body_stmts(&p.functions[0]).len(), 4);
    }

    #[test]
    fn parse_nested_calls() {
        let p = parse("fn f(x: int) -> int { return add(mul(x, 2), 1); }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return {
                value: Some(Expression::FunctionCall { args, .. }),
                ..
            } => {
                assert!(matches!(&args[0], Expression::FunctionCall { .. }));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_free_vars() {
        let p = parse("fn f(x: int, y: int) -> int { return x + y * 2; }");
        let stmts = body_stmts(&p.functions[0]);
        match &stmts[0] {
            Statement::Return { value: Some(e), .. } => {
                let v = e.referenced_vars();
                assert!(v.contains(&"x".to_string()) && v.contains(&"y".to_string()));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn parse_seq_assign() {
        let p = parse("fn swap(x: int, y: int) -> int { let t: int = x; x = y; y = t; return x; }");
        assert_eq!(body_stmts(&p.functions[0]).len(), 4);
    }
}
