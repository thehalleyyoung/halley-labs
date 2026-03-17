//! DSL AST types for property specification.
//!
//! Defines a small domain-specific language for expressing accessibility
//! properties such as:
//!
//! ```text
//! forall body in population(5%, 95%):
//!   forall device in [Quest3, VisionPro]:
//!     reachable(button_a, body, device)
//! ```
//!
//! This module provides the AST, token types, a basic parser,
//! and type-checking structures.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::error::{VerifierError, VerifierResult};

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

/// Token kinds produced by the lexer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenKind {
    // Keywords
    ForAll,
    Exists,
    In,
    Let,
    If,
    Then,
    Else,
    And,
    Or,
    Not,
    True,
    False,

    // Identifiers & literals
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Dot,
    Arrow,

    // Operators
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Special
    Eof,
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::ForAll => write!(f, "forall"),
            TokenKind::Exists => write!(f, "exists"),
            TokenKind::In => write!(f, "in"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Then => write!(f, "then"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::And => write!(f, "and"),
            TokenKind::Or => write!(f, "or"),
            TokenKind::Not => write!(f, "not"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Ident(s) => write!(f, "{s}"),
            TokenKind::IntLit(n) => write!(f, "{n}"),
            TokenKind::FloatLit(n) => write!(f, "{n}"),
            TokenKind::StringLit(s) => write!(f, "\"{s}\""),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::Eq => write!(f, "=="),
            TokenKind::Ne => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Le => write!(f, "<="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Ge => write!(f, ">="),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::Eof => write!(f, "<eof>"),
        }
    }
}

/// A token with its source location.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, column: usize) -> Self {
        Self { kind, line, column }
    }
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// Simple lexer for the DSL.
pub struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    /// Create a new lexer from source text.
    pub fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Tokenize the entire source.
    pub fn tokenize(&mut self) -> VerifierResult<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        if let Some(c) = ch {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '#' {
                // Line comment.
                while let Some(c) = self.advance() {
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> VerifierResult<Token> {
        self.skip_whitespace();
        let line = self.line;
        let col = self.col;

        let ch = match self.peek() {
            Some(c) => c,
            None => return Ok(Token::new(TokenKind::Eof, line, col)),
        };

        // Single-char tokens.
        let simple = match ch {
            '(' => Some(TokenKind::LParen),
            ')' => Some(TokenKind::RParen),
            '[' => Some(TokenKind::LBracket),
            ']' => Some(TokenKind::RBracket),
            ',' => Some(TokenKind::Comma),
            ':' => Some(TokenKind::Colon),
            '.' => Some(TokenKind::Dot),
            '+' => Some(TokenKind::Plus),
            '*' => Some(TokenKind::Star),
            '/' => Some(TokenKind::Slash),
            '%' => Some(TokenKind::Percent),
            _ => None,
        };
        if let Some(kind) = simple {
            self.advance();
            return Ok(Token::new(kind, line, col));
        }

        // Two-char tokens.
        if ch == '-' {
            self.advance();
            if self.peek() == Some('>') {
                self.advance();
                return Ok(Token::new(TokenKind::Arrow, line, col));
            }
            return Ok(Token::new(TokenKind::Minus, line, col));
        }
        if ch == '=' {
            self.advance();
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token::new(TokenKind::Eq, line, col));
            }
            return Err(VerifierError::DslParse {
                line,
                column: col,
                message: "Expected '==' but got '='".into(),
            });
        }
        if ch == '!' {
            self.advance();
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token::new(TokenKind::Ne, line, col));
            }
            return Err(VerifierError::DslParse {
                line,
                column: col,
                message: "Expected '!=' but got '!'".into(),
            });
        }
        if ch == '<' {
            self.advance();
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token::new(TokenKind::Le, line, col));
            }
            return Ok(Token::new(TokenKind::Lt, line, col));
        }
        if ch == '>' {
            self.advance();
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token::new(TokenKind::Ge, line, col));
            }
            return Ok(Token::new(TokenKind::Gt, line, col));
        }

        // String literal.
        if ch == '"' {
            self.advance();
            let mut s = String::new();
            loop {
                match self.advance() {
                    Some('"') => break,
                    Some('\\') => {
                        if let Some(esc) = self.advance() {
                            match esc {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                '\\' => s.push('\\'),
                                '"' => s.push('"'),
                                _ => s.push(esc),
                            }
                        }
                    }
                    Some(c) => s.push(c),
                    None => {
                        return Err(VerifierError::DslParse {
                            line,
                            column: col,
                            message: "Unterminated string literal".into(),
                        })
                    }
                }
            }
            return Ok(Token::new(TokenKind::StringLit(s), line, col));
        }

        // Number.
        if ch.is_ascii_digit() {
            let mut num_str = String::new();
            let mut is_float = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
                    if c == '.' || c == 'e' || c == 'E' {
                        is_float = true;
                    }
                    num_str.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            if is_float {
                let val: f64 = num_str.parse().map_err(|_| VerifierError::DslParse {
                    line,
                    column: col,
                    message: format!("Invalid float: {num_str}"),
                })?;
                return Ok(Token::new(TokenKind::FloatLit(val), line, col));
            } else {
                let val: i64 = num_str.parse().map_err(|_| VerifierError::DslParse {
                    line,
                    column: col,
                    message: format!("Invalid integer: {num_str}"),
                })?;
                return Ok(Token::new(TokenKind::IntLit(val), line, col));
            }
        }

        // Identifier or keyword.
        if ch.is_alphabetic() || ch == '_' {
            let mut ident = String::new();
            while let Some(c) = self.peek() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            let kind = match ident.as_str() {
                "forall" => TokenKind::ForAll,
                "exists" => TokenKind::Exists,
                "in" => TokenKind::In,
                "let" => TokenKind::Let,
                "if" => TokenKind::If,
                "then" => TokenKind::Then,
                "else" => TokenKind::Else,
                "and" => TokenKind::And,
                "or" => TokenKind::Or,
                "not" => TokenKind::Not,
                "true" => TokenKind::True,
                "false" => TokenKind::False,
                _ => TokenKind::Ident(ident),
            };
            return Ok(Token::new(kind, line, col));
        }

        Err(VerifierError::DslParse {
            line,
            column: col,
            message: format!("Unexpected character: '{ch}'"),
        })
    }
}

// ---------------------------------------------------------------------------
// AST types
// ---------------------------------------------------------------------------

/// Top-level property specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertySpec {
    /// Unique property ID.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Description.
    pub description: String,
    /// The property expression.
    pub expr: DslExpr,
    /// Target elements (empty = all).
    pub target_elements: Vec<String>,
    /// Required compliance standards.
    pub compliance_tags: Vec<String>,
}

impl PropertySpec {
    /// Create a new property spec.
    pub fn new(name: impl Into<String>, expr: DslExpr) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            expr,
            target_elements: Vec::new(),
            compliance_tags: Vec::new(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add target elements.
    pub fn with_targets(mut self, targets: Vec<String>) -> Self {
        self.target_elements = targets;
        self
    }

    /// Add compliance tags.
    pub fn with_compliance(mut self, tags: Vec<String>) -> Self {
        self.compliance_tags = tags;
        self
    }
}

impl std::fmt::Display for PropertySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Property '{}': {}", self.name, self.expr)
    }
}

/// DSL expression (AST node).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DslExpr {
    /// Boolean literal.
    BoolLit(bool),
    /// Integer literal.
    IntLit(i64),
    /// Float literal.
    FloatLit(f64),
    /// String literal.
    StringLit(String),
    /// Variable reference.
    Var(String),
    /// Predicate application: `name(args...)`.
    Predicate(DslPredicate),
    /// Quantified expression.
    Quantifier(DslQuantifier),
    /// Binary operation.
    BinOp {
        op: DslBinOp,
        left: Box<DslExpr>,
        right: Box<DslExpr>,
    },
    /// Unary operation (currently just `not`).
    UnaryOp {
        op: DslUnaryOp,
        operand: Box<DslExpr>,
    },
    /// Let-binding: `let name = value in body`.
    LetIn {
        name: String,
        value: Box<DslExpr>,
        body: Box<DslExpr>,
    },
    /// Conditional: `if cond then t else e`.
    IfThenElse {
        cond: Box<DslExpr>,
        then_branch: Box<DslExpr>,
        else_branch: Box<DslExpr>,
    },
    /// List literal.
    List(Vec<DslExpr>),
    /// Field access: `expr.field`.
    FieldAccess {
        object: Box<DslExpr>,
        field: String,
    },
    /// Function call: `name(args...)`.
    FunctionCall {
        name: String,
        args: Vec<DslExpr>,
    },
}

impl DslExpr {
    /// Convenience: create a numeric literal (backward compat).
    pub fn number(v: f64) -> Self {
        Self::FloatLit(v)
    }

    /// Convenience: create a variable reference (backward compat).
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Count AST nodes.
    pub fn node_count(&self) -> usize {
        match self {
            DslExpr::BoolLit(_)
            | DslExpr::IntLit(_)
            | DslExpr::FloatLit(_)
            | DslExpr::StringLit(_)
            | DslExpr::Var(_) => 1,
            DslExpr::Predicate(p) => 1 + p.args.iter().map(|a| a.node_count()).sum::<usize>(),
            DslExpr::Quantifier(q) => 1 + q.body.node_count(),
            DslExpr::BinOp { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
            DslExpr::UnaryOp { operand, .. } => 1 + operand.node_count(),
            DslExpr::LetIn { value, body, .. } => {
                1 + value.node_count() + body.node_count()
            }
            DslExpr::IfThenElse {
                cond,
                then_branch,
                else_branch,
            } => 1 + cond.node_count() + then_branch.node_count() + else_branch.node_count(),
            DslExpr::List(items) => 1 + items.iter().map(|i| i.node_count()).sum::<usize>(),
            DslExpr::FieldAccess { object, .. } => 1 + object.node_count(),
            DslExpr::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
        }
    }

    /// Collect free variables.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_free_vars(&mut vars, &[]);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_free_vars(&self, out: &mut Vec<String>, bound: &[String]) {
        match self {
            DslExpr::Var(name) => {
                if !bound.contains(name) {
                    out.push(name.clone());
                }
            }
            DslExpr::BoolLit(_)
            | DslExpr::IntLit(_)
            | DslExpr::FloatLit(_)
            | DslExpr::StringLit(_) => {}
            DslExpr::Predicate(p) => {
                for a in &p.args {
                    a.collect_free_vars(out, bound);
                }
            }
            DslExpr::Quantifier(q) => {
                q.domain.collect_free_vars(out, bound);
                let mut new_bound: Vec<String> = bound.to_vec();
                new_bound.push(q.variable.clone());
                q.body.collect_free_vars(out, &new_bound);
            }
            DslExpr::BinOp { left, right, .. } => {
                left.collect_free_vars(out, bound);
                right.collect_free_vars(out, bound);
            }
            DslExpr::UnaryOp { operand, .. } => {
                operand.collect_free_vars(out, bound);
            }
            DslExpr::LetIn { name, value, body } => {
                value.collect_free_vars(out, bound);
                let mut new_bound: Vec<String> = bound.to_vec();
                new_bound.push(name.clone());
                body.collect_free_vars(out, &new_bound);
            }
            DslExpr::IfThenElse {
                cond,
                then_branch,
                else_branch,
            } => {
                cond.collect_free_vars(out, bound);
                then_branch.collect_free_vars(out, bound);
                else_branch.collect_free_vars(out, bound);
            }
            DslExpr::List(items) => {
                for item in items {
                    item.collect_free_vars(out, bound);
                }
            }
            DslExpr::FieldAccess { object, .. } => {
                object.collect_free_vars(out, bound);
            }
            DslExpr::FunctionCall { args, .. } => {
                for a in args {
                    a.collect_free_vars(out, bound);
                }
            }
        }
    }
}

impl std::fmt::Display for DslExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslExpr::BoolLit(b) => write!(f, "{b}"),
            DslExpr::IntLit(n) => write!(f, "{n}"),
            DslExpr::FloatLit(n) => write!(f, "{n}"),
            DslExpr::StringLit(s) => write!(f, "\"{s}\""),
            DslExpr::Var(name) => write!(f, "{name}"),
            DslExpr::Predicate(p) => write!(f, "{p}"),
            DslExpr::Quantifier(q) => write!(f, "{q}"),
            DslExpr::BinOp { op, left, right } => write!(f, "({left} {op} {right})"),
            DslExpr::UnaryOp { op, operand } => write!(f, "({op} {operand})"),
            DslExpr::LetIn { name, value, body } => {
                write!(f, "(let {name} = {value} in {body})")
            }
            DslExpr::IfThenElse {
                cond,
                then_branch,
                else_branch,
            } => write!(f, "(if {cond} then {then_branch} else {else_branch})"),
            DslExpr::List(items) => {
                let strs: Vec<String> = items.iter().map(|i| format!("{i}")).collect();
                write!(f, "[{}]", strs.join(", "))
            }
            DslExpr::FieldAccess { object, field } => write!(f, "{object}.{field}"),
            DslExpr::FunctionCall { name, args } => {
                let strs: Vec<String> = args.iter().map(|a| format!("{a}")).collect();
                write!(f, "{}({})", name, strs.join(", "))
            }
        }
    }
}

/// A built-in predicate call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DslPredicate {
    /// Predicate name.
    pub name: String,
    /// Arguments.
    pub args: Vec<DslExpr>,
}

impl DslPredicate {
    /// Create a new predicate.
    pub fn new(name: impl Into<String>, args: Vec<DslExpr>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    /// "reachable(element, body, device)" predicate.
    pub fn reachable(element: DslExpr, body: DslExpr, device: DslExpr) -> Self {
        Self::new("reachable", vec![element, body, device])
    }

    /// "visible(element, body, device)" predicate.
    pub fn visible(element: DslExpr, body: DslExpr, device: DslExpr) -> Self {
        Self::new("visible", vec![element, body, device])
    }

    /// "in_tracking_volume(position, device)" predicate.
    pub fn in_tracking_volume(position: DslExpr, device: DslExpr) -> Self {
        Self::new("in_tracking_volume", vec![position, device])
    }

    /// "distance(a, b) < threshold" predicate.
    pub fn within_distance(a: DslExpr, b: DslExpr, threshold: DslExpr) -> Self {
        Self::new("within_distance", vec![a, b, threshold])
    }
}

impl std::fmt::Display for DslPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| format!("{a}")).collect();
        write!(f, "{}({})", self.name, args.join(", "))
    }
}

/// Quantifier kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantifierKind {
    /// Universal quantification (∀).
    ForAll,
    /// Existential quantification (∃).
    Exists,
}

impl std::fmt::Display for QuantifierKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantifierKind::ForAll => write!(f, "∀"),
            QuantifierKind::Exists => write!(f, "∃"),
        }
    }
}

/// A quantified expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DslQuantifier {
    /// Quantifier kind.
    pub kind: QuantifierKind,
    /// Bound variable name.
    pub variable: String,
    /// Domain expression.
    pub domain: Box<DslExpr>,
    /// Body expression.
    pub body: Box<DslExpr>,
}

impl DslQuantifier {
    /// Create a new universal quantifier.
    pub fn for_all(variable: impl Into<String>, domain: DslExpr, body: DslExpr) -> Self {
        Self {
            kind: QuantifierKind::ForAll,
            variable: variable.into(),
            domain: Box::new(domain),
            body: Box::new(body),
        }
    }

    /// Create a new existential quantifier.
    pub fn exists(variable: impl Into<String>, domain: DslExpr, body: DslExpr) -> Self {
        Self {
            kind: QuantifierKind::Exists,
            variable: variable.into(),
            domain: Box::new(domain),
            body: Box::new(body),
        }
    }
}

impl std::fmt::Display for DslQuantifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} in {}: {}",
            self.kind, self.variable, self.domain, self.body
        )
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DslBinOp {
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl std::fmt::Display for DslBinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslBinOp::And => write!(f, "and"),
            DslBinOp::Or => write!(f, "or"),
            DslBinOp::Eq => write!(f, "=="),
            DslBinOp::Ne => write!(f, "!="),
            DslBinOp::Lt => write!(f, "<"),
            DslBinOp::Le => write!(f, "<="),
            DslBinOp::Gt => write!(f, ">"),
            DslBinOp::Ge => write!(f, ">="),
            DslBinOp::Add => write!(f, "+"),
            DslBinOp::Sub => write!(f, "-"),
            DslBinOp::Mul => write!(f, "*"),
            DslBinOp::Div => write!(f, "/"),
            DslBinOp::Mod => write!(f, "%"),
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DslUnaryOp {
    Not,
    Neg,
}

impl std::fmt::Display for DslUnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslUnaryOp::Not => write!(f, "not"),
            DslUnaryOp::Neg => write!(f, "-"),
        }
    }
}

// ---------------------------------------------------------------------------
// Type system
// ---------------------------------------------------------------------------

/// Types in the DSL type system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DslType {
    Bool,
    Int,
    Float,
    String,
    /// A named domain type (e.g., "Body", "Device", "Element").
    Domain(String),
    /// List type.
    List(Box<DslType>),
    /// Population with percentile bounds.
    Population {
        low: ordered_float::OrderedFloat<f64>,
        high: ordered_float::OrderedFloat<f64>,
    },
    /// Function type.
    Function {
        params: Vec<DslType>,
        ret: Box<DslType>,
    },
    /// Type variable (for inference).
    TypeVar(String),
    /// Unknown / error type.
    Unknown,
}

impl std::fmt::Display for DslType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslType::Bool => write!(f, "Bool"),
            DslType::Int => write!(f, "Int"),
            DslType::Float => write!(f, "Float"),
            DslType::String => write!(f, "String"),
            DslType::Domain(name) => write!(f, "{name}"),
            DslType::List(inner) => write!(f, "[{inner}]"),
            DslType::Population { low, high } => {
                write!(f, "Population({:.0}%-{:.0}%)", low.0 * 100.0, high.0 * 100.0)
            }
            DslType::Function { params, ret } => {
                let ps: Vec<String> = params.iter().map(|p| format!("{p}")).collect();
                write!(f, "({}) -> {}", ps.join(", "), ret)
            }
            DslType::TypeVar(v) => write!(f, "'{v}"),
            DslType::Unknown => write!(f, "?"),
        }
    }
}

/// Type environment for type checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeEnv {
    /// Variable → type bindings.
    pub bindings: HashMap<String, DslType>,
    /// Predicate signatures.
    pub predicates: HashMap<String, Vec<DslType>>,
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::standard()
    }
}

impl TypeEnv {
    /// Create a standard type environment with built-in predicates.
    pub fn standard() -> Self {
        let mut predicates = HashMap::new();
        predicates.insert(
            "reachable".into(),
            vec![
                DslType::Domain("Element".into()),
                DslType::Domain("Body".into()),
                DslType::Domain("Device".into()),
            ],
        );
        predicates.insert(
            "visible".into(),
            vec![
                DslType::Domain("Element".into()),
                DslType::Domain("Body".into()),
                DslType::Domain("Device".into()),
            ],
        );
        predicates.insert(
            "in_tracking_volume".into(),
            vec![
                DslType::Domain("Position".into()),
                DslType::Domain("Device".into()),
            ],
        );
        predicates.insert(
            "within_distance".into(),
            vec![
                DslType::Domain("Position".into()),
                DslType::Domain("Position".into()),
                DslType::Float,
            ],
        );

        Self {
            bindings: HashMap::new(),
            predicates,
        }
    }

    /// Bind a variable to a type.
    pub fn bind(&mut self, name: impl Into<String>, ty: DslType) {
        self.bindings.insert(name.into(), ty);
    }

    /// Look up a variable's type.
    pub fn lookup(&self, name: &str) -> Option<&DslType> {
        self.bindings.get(name)
    }

    /// Look up a predicate signature.
    pub fn predicate_sig(&self, name: &str) -> Option<&Vec<DslType>> {
        self.predicates.get(name)
    }

    /// Type check a predicate call.
    pub fn check_predicate(&self, pred: &DslPredicate) -> Vec<String> {
        let mut errors = Vec::new();
        match self.predicates.get(&pred.name) {
            None => errors.push(format!("Unknown predicate: {}", pred.name)),
            Some(sig) => {
                if pred.args.len() != sig.len() {
                    errors.push(format!(
                        "Predicate '{}' expects {} args, got {}",
                        pred.name,
                        sig.len(),
                        pred.args.len()
                    ));
                }
            }
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Build a property: "all elements reachable for population(low, high) on device".
pub fn all_reachable_property(
    name: impl Into<String>,
    percentile_low: f64,
    percentile_high: f64,
    device_name: impl Into<String>,
) -> PropertySpec {
    let body_domain = DslExpr::FunctionCall {
        name: "population".into(),
        args: vec![
            DslExpr::FloatLit(percentile_low),
            DslExpr::FloatLit(percentile_high),
        ],
    };
    let device = DslExpr::Var(device_name.into());

    let inner = DslExpr::Predicate(DslPredicate::reachable(
        DslExpr::Var("element".into()),
        DslExpr::Var("body".into()),
        device,
    ));

    let body_quant = DslExpr::Quantifier(DslQuantifier::for_all("body", body_domain, inner));

    let element_domain = DslExpr::FunctionCall {
        name: "all_elements".into(),
        args: vec![],
    };
    let full = DslExpr::Quantifier(DslQuantifier::for_all("element", element_domain, body_quant));

    PropertySpec::new(name, full)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_simple() {
        let mut lexer = Lexer::new("forall x in pop: reachable(x)");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::ForAll);
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".into()));
        assert_eq!(tokens[2].kind, TokenKind::In);
    }

    #[test]
    fn test_lexer_numbers() {
        let mut lexer = Lexer::new("42 3.14");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::IntLit(42));
        assert_eq!(tokens[1].kind, TokenKind::FloatLit(3.14));
    }

    #[test]
    fn test_lexer_strings() {
        let mut lexer = Lexer::new("\"hello world\"");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::StringLit("hello world".into()));
    }

    #[test]
    fn test_lexer_operators() {
        let mut lexer = Lexer::new("< <= > >= == != + - * /");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Lt);
        assert_eq!(tokens[1].kind, TokenKind::Le);
        assert_eq!(tokens[2].kind, TokenKind::Gt);
        assert_eq!(tokens[3].kind, TokenKind::Ge);
        assert_eq!(tokens[4].kind, TokenKind::Eq);
        assert_eq!(tokens[5].kind, TokenKind::Ne);
    }

    #[test]
    fn test_lexer_arrow() {
        let mut lexer = Lexer::new("->");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Arrow);
    }

    #[test]
    fn test_lexer_comment() {
        let mut lexer = Lexer::new("x # comment\ny");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident("x".into()));
        assert_eq!(tokens[1].kind, TokenKind::Ident("y".into()));
    }

    #[test]
    fn test_lexer_brackets() {
        let mut lexer = Lexer::new("[a, b]");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::LBracket);
        assert_eq!(tokens[4].kind, TokenKind::RBracket);
    }

    #[test]
    fn test_lexer_keywords() {
        let mut lexer = Lexer::new("forall exists let if then else and or not true false");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].kind, TokenKind::ForAll);
        assert_eq!(tokens[1].kind, TokenKind::Exists);
        assert_eq!(tokens[2].kind, TokenKind::Let);
        assert_eq!(tokens[3].kind, TokenKind::If);
        assert_eq!(tokens[4].kind, TokenKind::Then);
        assert_eq!(tokens[5].kind, TokenKind::Else);
        assert_eq!(tokens[6].kind, TokenKind::And);
        assert_eq!(tokens[7].kind, TokenKind::Or);
        assert_eq!(tokens[8].kind, TokenKind::Not);
        assert_eq!(tokens[9].kind, TokenKind::True);
        assert_eq!(tokens[10].kind, TokenKind::False);
    }

    #[test]
    fn test_lexer_error() {
        let mut lexer = Lexer::new("@");
        assert!(lexer.tokenize().is_err());
    }

    #[test]
    fn test_dsl_expr_display() {
        let e = DslExpr::BinOp {
            op: DslBinOp::And,
            left: Box::new(DslExpr::Var("a".into())),
            right: Box::new(DslExpr::Var("b".into())),
        };
        assert_eq!(format!("{e}"), "(a and b)");
    }

    #[test]
    fn test_predicate_display() {
        let p = DslPredicate::reachable(
            DslExpr::Var("btn".into()),
            DslExpr::Var("body".into()),
            DslExpr::Var("quest3".into()),
        );
        let s = format!("{p}");
        assert!(s.contains("reachable"));
        assert!(s.contains("btn"));
    }

    #[test]
    fn test_quantifier_display() {
        let q = DslQuantifier::for_all(
            "body",
            DslExpr::FunctionCall {
                name: "population".into(),
                args: vec![DslExpr::FloatLit(0.05), DslExpr::FloatLit(0.95)],
            },
            DslExpr::Predicate(DslPredicate::reachable(
                DslExpr::Var("e".into()),
                DslExpr::Var("body".into()),
                DslExpr::Var("d".into()),
            )),
        );
        let s = format!("{q}");
        assert!(s.contains("∀"));
        assert!(s.contains("body"));
    }

    #[test]
    fn test_node_count() {
        let e = DslExpr::BinOp {
            op: DslBinOp::Add,
            left: Box::new(DslExpr::IntLit(1)),
            right: Box::new(DslExpr::IntLit(2)),
        };
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_free_vars() {
        let e = DslExpr::BinOp {
            op: DslBinOp::And,
            left: Box::new(DslExpr::Var("x".into())),
            right: Box::new(DslExpr::Var("y".into())),
        };
        let vars = e.free_vars();
        assert_eq!(vars, vec!["x", "y"]);
    }

    #[test]
    fn test_free_vars_bound() {
        let e = DslExpr::Quantifier(DslQuantifier::for_all(
            "x",
            DslExpr::Var("pop".into()),
            DslExpr::BinOp {
                op: DslBinOp::And,
                left: Box::new(DslExpr::Var("x".into())),
                right: Box::new(DslExpr::Var("y".into())),
            },
        ));
        let vars = e.free_vars();
        assert_eq!(vars, vec!["pop", "y"]);
    }

    #[test]
    fn test_free_vars_let() {
        let e = DslExpr::LetIn {
            name: "a".into(),
            value: Box::new(DslExpr::Var("b".into())),
            body: Box::new(DslExpr::BinOp {
                op: DslBinOp::Add,
                left: Box::new(DslExpr::Var("a".into())),
                right: Box::new(DslExpr::Var("c".into())),
            }),
        };
        let vars = e.free_vars();
        assert_eq!(vars, vec!["b", "c"]);
    }

    #[test]
    fn test_property_spec() {
        let prop = all_reachable_property("all_reachable", 0.05, 0.95, "Quest3");
        assert_eq!(prop.name, "all_reachable");
        let display = format!("{prop}");
        assert!(display.contains("all_reachable"));
    }

    #[test]
    fn test_type_env_standard() {
        let env = TypeEnv::standard();
        assert!(env.predicate_sig("reachable").is_some());
        assert!(env.predicate_sig("visible").is_some());
    }

    #[test]
    fn test_type_check_predicate_ok() {
        let env = TypeEnv::standard();
        let pred = DslPredicate::reachable(
            DslExpr::Var("e".into()),
            DslExpr::Var("b".into()),
            DslExpr::Var("d".into()),
        );
        let errors = env.check_predicate(&pred);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_type_check_predicate_wrong_arity() {
        let env = TypeEnv::standard();
        let pred = DslPredicate::new("reachable", vec![DslExpr::Var("e".into())]);
        let errors = env.check_predicate(&pred);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_type_check_unknown_predicate() {
        let env = TypeEnv::standard();
        let pred = DslPredicate::new("unknown_pred", vec![]);
        let errors = env.check_predicate(&pred);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_dsl_type_display() {
        assert_eq!(format!("{}", DslType::Bool), "Bool");
        assert_eq!(format!("{}", DslType::List(Box::new(DslType::Int))), "[Int]");
        let pop = DslType::Population {
            low: ordered_float::OrderedFloat(0.05),
            high: ordered_float::OrderedFloat(0.95),
        };
        assert!(format!("{pop}").contains("5%"));
    }

    #[test]
    fn test_token_kind_display() {
        assert_eq!(format!("{}", TokenKind::ForAll), "forall");
        assert_eq!(format!("{}", TokenKind::LParen), "(");
        assert_eq!(format!("{}", TokenKind::Arrow), "->");
    }

    #[test]
    fn test_serde_roundtrip_property() {
        let prop = all_reachable_property("test", 0.05, 0.95, "Q3");
        let json = serde_json::to_string(&prop).unwrap();
        let back: PropertySpec = serde_json::from_str(&json).unwrap();
        assert_eq!(prop.name, back.name);
    }

    #[test]
    fn test_serde_roundtrip_type() {
        let ty = DslType::Function {
            params: vec![DslType::Bool, DslType::Int],
            ret: Box::new(DslType::Float),
        };
        let json = serde_json::to_string(&ty).unwrap();
        let back: DslType = serde_json::from_str(&json).unwrap();
        assert_eq!(ty, back);
    }

    #[test]
    fn test_if_then_else_display() {
        let e = DslExpr::IfThenElse {
            cond: Box::new(DslExpr::BoolLit(true)),
            then_branch: Box::new(DslExpr::IntLit(1)),
            else_branch: Box::new(DslExpr::IntLit(2)),
        };
        let s = format!("{e}");
        assert!(s.contains("if"));
        assert!(s.contains("then"));
        assert!(s.contains("else"));
    }

    #[test]
    fn test_list_display() {
        let e = DslExpr::List(vec![DslExpr::IntLit(1), DslExpr::IntLit(2)]);
        assert_eq!(format!("{e}"), "[1, 2]");
    }

    #[test]
    fn test_field_access_display() {
        let e = DslExpr::FieldAccess {
            object: Box::new(DslExpr::Var("body".into())),
            field: "stature".into(),
        };
        assert_eq!(format!("{e}"), "body.stature");
    }

    #[test]
    fn test_type_env_bind_lookup() {
        let mut env = TypeEnv::standard();
        env.bind("x", DslType::Float);
        assert_eq!(env.lookup("x"), Some(&DslType::Float));
        assert_eq!(env.lookup("y"), None);
    }

    #[test]
    fn test_backward_compat() {
        let e = DslExpr::number(3.14);
        assert!(matches!(e, DslExpr::FloatLit(v) if (v - 3.14).abs() < 1e-12));
        let v = DslExpr::var("x");
        assert!(matches!(v, DslExpr::Var(name) if name == "x"));
    }
}
