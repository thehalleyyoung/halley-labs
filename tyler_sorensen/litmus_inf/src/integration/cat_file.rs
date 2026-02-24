//! Parser and translator for herd7 `.cat` memory model files.
//!
//! Provides [`CatFileParser`] for parsing `.cat` syntax into a [`CatModel`] AST,
//! and [`CatTranslator`] for bidirectional conversion between the Litmus∞
//! [`MemoryModel`] representation and `.cat` format.

use std::collections::HashMap;
use std::fmt;

use crate::checker::memory_model::{
    MemoryModel, RelationDef, RelationType, DerivedRelation, Constraint, RelationExpr,
};

// ---------------------------------------------------------------------------
// CatRelation — relation expression AST for .cat files
// ---------------------------------------------------------------------------

/// AST node for a relation expression in `.cat` syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatRelation {
    /// A named relation reference (e.g. `po`, `rf`, user-defined names).
    Named(String),
    /// A builtin relation keyword.
    Builtin(CatBuiltin),
    /// Union: `R1 | R2`.
    Union(Box<CatRelation>, Box<CatRelation>),
    /// Intersection: `R1 & R2`.
    Intersection(Box<CatRelation>, Box<CatRelation>),
    /// Sequence (composition): `R1 ; R2`.
    Sequence(Box<CatRelation>, Box<CatRelation>),
    /// Complement: `~R` (all pairs not in R).
    Complement(Box<CatRelation>),
    /// Inverse (transpose): `R^-1`.
    Inverse(Box<CatRelation>),
    /// Transitive closure: `R+`.
    TransitiveClosure(Box<CatRelation>),
    /// Reflexive-transitive closure: `R*`.
    ReflexiveTransitiveClosure(Box<CatRelation>),
    /// Set difference: `R1 \ R2`.
    Difference(Box<CatRelation>, Box<CatRelation>),
    /// Cartesian product of two sets: `S1 * S2`.
    Cartesian(Box<CatRelation>, Box<CatRelation>),
    /// Identity relation.
    Identity,
    /// Optional (reflexive closure): `R?`.
    Optional(Box<CatRelation>),
    /// Guard / filter: `[set]`.
    Guard(String),
    /// Empty relation: `0`.
    Empty,
}

impl fmt::Display for CatRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(n) => write!(f, "{}", n),
            Self::Builtin(b) => write!(f, "{}", b),
            Self::Union(a, b) => write!(f, "({} | {})", a, b),
            Self::Intersection(a, b) => write!(f, "({} & {})", a, b),
            Self::Sequence(a, b) => write!(f, "({} ; {})", a, b),
            Self::Complement(a) => write!(f, "~{}", a),
            Self::Inverse(a) => write!(f, "{}^-1", a),
            Self::TransitiveClosure(a) => write!(f, "{}+", a),
            Self::ReflexiveTransitiveClosure(a) => write!(f, "{}*", a),
            Self::Difference(a, b) => write!(f, "({} \\ {})", a, b),
            Self::Cartesian(a, b) => write!(f, "({} * {})", a, b),
            Self::Identity => write!(f, "id"),
            Self::Optional(a) => write!(f, "{}?", a),
            Self::Guard(s) => write!(f, "[{}]", s),
            Self::Empty => write!(f, "0"),
        }
    }
}

// ---------------------------------------------------------------------------
// CatBuiltin — well-known base relations
// ---------------------------------------------------------------------------

/// Builtin relations recognized in `.cat` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CatBuiltin {
    /// Program order.
    Po,
    /// Reads-from.
    Rf,
    /// Coherence order (write serialization).
    Co,
    /// From-reads.
    Fr,
    /// Identity.
    Id,
    /// External (cross-thread) restriction.
    Ext,
    /// Internal (same-thread) restriction.
    Int,
    /// Same-location restriction.
    Loc,
    /// Address dependency.
    Addr,
    /// Data dependency.
    Data,
    /// Control dependency.
    Ctrl,
    /// Read-modify-write atomicity.
    Rmw,
    /// Atomic memory operations.
    Amo,
}

impl CatBuiltin {
    /// Parse a string into a builtin, if it matches.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "po" => Some(Self::Po),
            "rf" => Some(Self::Rf),
            "co" | "ws" => Some(Self::Co),
            "fr" => Some(Self::Fr),
            "id" => Some(Self::Id),
            "ext" => Some(Self::Ext),
            "int" => Some(Self::Int),
            "loc" | "same-loc" => Some(Self::Loc),
            "addr" => Some(Self::Addr),
            "data" => Some(Self::Data),
            "ctrl" => Some(Self::Ctrl),
            "rmw" => Some(Self::Rmw),
            "amo" => Some(Self::Amo),
            _ => None,
        }
    }

    /// Canonical name string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Po => "po",
            Self::Rf => "rf",
            Self::Co => "co",
            Self::Fr => "fr",
            Self::Id => "id",
            Self::Ext => "ext",
            Self::Int => "int",
            Self::Loc => "loc",
            Self::Addr => "addr",
            Self::Data => "data",
            Self::Ctrl => "ctrl",
            Self::Rmw => "rmw",
            Self::Amo => "amo",
        }
    }
}

impl fmt::Display for CatBuiltin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// CatExpr — top-level expression wrapper
// ---------------------------------------------------------------------------

/// Top-level AST for a `.cat` expression (wraps [`CatRelation`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatExpr {
    pub relation: CatRelation,
}

impl CatExpr {
    pub fn new(relation: CatRelation) -> Self {
        Self { relation }
    }
}

impl fmt::Display for CatExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.relation)
    }
}

// ---------------------------------------------------------------------------
// CatAxiom — constraint declarations
// ---------------------------------------------------------------------------

/// An axiom (constraint) in a `.cat` model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatAxiom {
    /// `acyclic R as name`
    Acyclic { relation: CatRelation, name: Option<String> },
    /// `irreflexive R as name`
    Irreflexive { relation: CatRelation, name: Option<String> },
    /// `empty R as name`
    Empty { relation: CatRelation, name: Option<String> },
}

impl fmt::Display for CatAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acyclic { relation, name } => {
                write!(f, "acyclic {}", relation)?;
                if let Some(n) = name { write!(f, " as {}", n)?; }
                Ok(())
            }
            Self::Irreflexive { relation, name } => {
                write!(f, "irreflexive {}", relation)?;
                if let Some(n) = name { write!(f, " as {}", n)?; }
                Ok(())
            }
            Self::Empty { relation, name } => {
                write!(f, "empty {}", relation)?;
                if let Some(n) = name { write!(f, " as {}", n)?; }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CatBinding — let definitions
// ---------------------------------------------------------------------------

/// A `let` binding in a `.cat` file: `let name = expr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatBinding {
    pub name: String,
    pub expr: CatRelation,
    pub is_recursive: bool,
}

impl fmt::Display for CatBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_recursive {
            write!(f, "let rec {} = {}", self.name, self.expr)
        } else {
            write!(f, "let {} = {}", self.name, self.expr)
        }
    }
}

// ---------------------------------------------------------------------------
// CatModel — complete parsed model
// ---------------------------------------------------------------------------

/// A parsed `.cat` memory model.
#[derive(Debug, Clone)]
pub struct CatModel {
    /// Model name (from the `"name"` header line).
    pub name: String,
    /// Relation bindings in definition order.
    pub bindings: Vec<CatBinding>,
    /// Axioms (acyclic / irreflexive / empty constraints).
    pub axioms: Vec<CatAxiom>,
    /// Include directives encountered.
    pub includes: Vec<String>,
}

impl CatModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            bindings: Vec::new(),
            axioms: Vec::new(),
            includes: Vec::new(),
        }
    }

    /// Look up a binding by name.
    pub fn get_binding(&self, name: &str) -> Option<&CatBinding> {
        self.bindings.iter().find(|b| b.name == name)
    }

    /// All defined relation names.
    pub fn defined_names(&self) -> Vec<&str> {
        self.bindings.iter().map(|b| b.name.as_str()).collect()
    }

    /// All referenced builtin names.
    pub fn referenced_builtins(&self) -> Vec<CatBuiltin> {
        let mut builtins = Vec::new();
        for b in &self.bindings {
            collect_builtins(&b.expr, &mut builtins);
        }
        for a in &self.axioms {
            let rel = match a {
                CatAxiom::Acyclic { relation, .. } => relation,
                CatAxiom::Irreflexive { relation, .. } => relation,
                CatAxiom::Empty { relation, .. } => relation,
            };
            collect_builtins(rel, &mut builtins);
        }
        builtins.sort_by_key(|b| b.name());
        builtins.dedup();
        builtins
    }
}

fn collect_builtins(rel: &CatRelation, out: &mut Vec<CatBuiltin>) {
    match rel {
        CatRelation::Builtin(b) => { if !out.contains(b) { out.push(*b); } }
        CatRelation::Union(a, b)
        | CatRelation::Intersection(a, b)
        | CatRelation::Sequence(a, b)
        | CatRelation::Difference(a, b)
        | CatRelation::Cartesian(a, b) => {
            collect_builtins(a, out);
            collect_builtins(b, out);
        }
        CatRelation::Complement(a)
        | CatRelation::Inverse(a)
        | CatRelation::TransitiveClosure(a)
        | CatRelation::ReflexiveTransitiveClosure(a)
        | CatRelation::Optional(a) => {
            collect_builtins(a, out);
        }
        CatRelation::Named(_)
        | CatRelation::Identity
        | CatRelation::Guard(_)
        | CatRelation::Empty => {}
    }
}

impl fmt::Display for CatModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\"{}\"", self.name)?;
        for inc in &self.includes {
            writeln!(f, "include \"{}\"", inc)?;
        }
        writeln!(f)?;
        for b in &self.bindings {
            writeln!(f, "{}", b)?;
        }
        writeln!(f)?;
        for a in &self.axioms {
            writeln!(f, "{}", a)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token / Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    StringLit(String),
    Let,
    Rec,
    And,        // keyword `and` for multi-let
    Acyclic,
    Irreflexive,
    Empty,
    As,
    Include,
    Show,
    Unshow,
    Pipe,       // |
    Ampersand,  // &
    Semicolon,  // ;
    Tilde,      // ~
    Backslash,  // \ (difference)
    Star,       // *
    Plus,       // +
    Question,   // ?
    Caret,      // ^
    Minus,      // -
    LParen,
    RParen,
    LBracket,
    RBracket,
    Equals,
    Zero,       // literal 0
    One,        // literal 1
    Comma,
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    tokens: Vec<Token>,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            tokens: Vec::new(),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        while self.pos < self.chars.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.chars.len() {
                break;
            }
            let ch = self.chars[self.pos];
            match ch {
                '|' => { self.tokens.push(Token::Pipe); self.pos += 1; }
                '&' => { self.tokens.push(Token::Ampersand); self.pos += 1; }
                ';' => { self.tokens.push(Token::Semicolon); self.pos += 1; }
                '~' => { self.tokens.push(Token::Tilde); self.pos += 1; }
                '\\' => { self.tokens.push(Token::Backslash); self.pos += 1; }
                '*' => { self.tokens.push(Token::Star); self.pos += 1; }
                '+' => { self.tokens.push(Token::Plus); self.pos += 1; }
                '?' => { self.tokens.push(Token::Question); self.pos += 1; }
                '^' => { self.tokens.push(Token::Caret); self.pos += 1; }
                '-' => { self.tokens.push(Token::Minus); self.pos += 1; }
                '(' => { self.tokens.push(Token::LParen); self.pos += 1; }
                ')' => { self.tokens.push(Token::RParen); self.pos += 1; }
                '[' => { self.tokens.push(Token::LBracket); self.pos += 1; }
                ']' => { self.tokens.push(Token::RBracket); self.pos += 1; }
                '=' => { self.tokens.push(Token::Equals); self.pos += 1; }
                ',' => { self.tokens.push(Token::Comma); self.pos += 1; }
                '"' => { self.read_string()?; }
                '0' if !self.next_is_ident_char() => {
                    self.tokens.push(Token::Zero);
                    self.pos += 1;
                }
                '1' if !self.next_is_ident_char() => {
                    self.tokens.push(Token::One);
                    self.pos += 1;
                }
                _ if ch.is_alphanumeric() || ch == '_' => { self.read_ident(); }
                _ => {
                    return Err(format!("unexpected character '{}' at position {}", ch, self.pos));
                }
            }
        }
        self.tokens.push(Token::Eof);
        Ok(self.tokens.clone())
    }

    fn next_is_ident_char(&self) -> bool {
        if self.pos + 1 < self.chars.len() {
            let c = self.chars[self.pos + 1];
            c.is_alphanumeric() || c == '_' || c == '-'
        } else {
            false
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch.is_whitespace() {
                self.pos += 1;
            } else if self.pos + 1 < self.chars.len()
                && ch == '(' && self.chars[self.pos + 1] == '*'
            {
                // Block comment (* ... *)
                self.pos += 2;
                let mut depth = 1;
                while self.pos + 1 < self.chars.len() && depth > 0 {
                    if self.chars[self.pos] == '(' && self.chars[self.pos + 1] == '*' {
                        depth += 1;
                        self.pos += 2;
                    } else if self.chars[self.pos] == '*' && self.chars[self.pos + 1] == ')' {
                        depth -= 1;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<(), String> {
        self.pos += 1; // skip opening quote
        let start = self.pos;
        while self.pos < self.chars.len() && self.chars[self.pos] != '"' {
            self.pos += 1;
        }
        if self.pos >= self.chars.len() {
            return Err("unterminated string literal".to_string());
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        self.tokens.push(Token::StringLit(s));
        self.pos += 1; // skip closing quote
        Ok(())
    }

    fn read_ident(&mut self) {
        let start = self.pos;
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word: String = self.chars[start..self.pos].iter().collect();
        let token = match word.as_str() {
            "let" => Token::Let,
            "rec" => Token::Rec,
            "and" => Token::And,
            "acyclic" => Token::Acyclic,
            "irreflexive" => Token::Irreflexive,
            "empty" => Token::Empty,
            "as" => Token::As,
            "include" => Token::Include,
            "show" => Token::Show,
            "unshow" => Token::Unshow,
            _ => Token::Ident(word),
        };
        self.tokens.push(token);
    }
}

// ---------------------------------------------------------------------------
// CatFileParser
// ---------------------------------------------------------------------------

/// Parser for herd7 `.cat` memory model files.
///
/// Handles the subset of `.cat` syntax relevant for axiomatic memory model
/// definitions: `let` bindings, relation expressions, and axiom declarations.
///
/// # Example
/// ```ignore
/// let model = CatFileParser::parse_str(r#"
///     "SC"
///     let com = rf | co | fr
///     acyclic po | com
/// "#).unwrap();
/// assert_eq!(model.name, "SC");
/// ```
pub struct CatFileParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl CatFileParser {
    /// Parse a `.cat` file from a string.
    pub fn parse_str(input: &str) -> Result<CatModel, String> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Self { tokens, pos: 0 };
        parser.parse_model()
    }

    /// Parse a `.cat` file from a file path.
    pub fn parse_file(path: &str) -> Result<CatModel, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path, e))?;
        Self::parse_str(&content)
    }

    // -- helpers --

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        self.pos += 1;
        t
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            other => Err(format!("expected identifier, got {:?}", other)),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let t = self.advance();
        if &t == expected {
            Ok(())
        } else {
            Err(format!("expected {:?}, got {:?}", expected, t))
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    // -- model parsing --

    fn parse_model(&mut self) -> Result<CatModel, String> {
        // Optional model name as string literal.
        let name = match self.peek() {
            Token::StringLit(_) => {
                if let Token::StringLit(s) = self.advance() { s } else { unreachable!() }
            }
            _ => "unnamed".to_string(),
        };

        let mut model = CatModel::new(&name);

        while !self.at_eof() {
            match self.peek() {
                Token::Let => { self.parse_let_binding(&mut model)?; }
                Token::Acyclic => { self.parse_axiom_acyclic(&mut model)?; }
                Token::Irreflexive => { self.parse_axiom_irreflexive(&mut model)?; }
                Token::Empty => { self.parse_axiom_empty(&mut model)?; }
                Token::Include => { self.parse_include(&mut model)?; }
                Token::Show | Token::Unshow => { self.skip_show_unshow(); }
                Token::Eof => break,
                _ => {
                    // Skip unknown top-level tokens gracefully.
                    self.advance();
                }
            }
        }
        Ok(model)
    }

    fn parse_let_binding(&mut self, model: &mut CatModel) -> Result<(), String> {
        self.expect(&Token::Let)?;
        let is_recursive = if matches!(self.peek(), Token::Rec) {
            self.advance();
            true
        } else {
            false
        };
        let name = self.expect_ident()?;
        self.expect(&Token::Equals)?;
        let expr = self.parse_expr()?;
        model.bindings.push(CatBinding { name, expr, is_recursive });

        // Handle `and name = expr` continuation (multi-let).
        while matches!(self.peek(), Token::And) {
            self.advance();
            let name2 = self.expect_ident()?;
            self.expect(&Token::Equals)?;
            let expr2 = self.parse_expr()?;
            model.bindings.push(CatBinding { name: name2, expr: expr2, is_recursive });
        }

        Ok(())
    }

    fn parse_axiom_acyclic(&mut self, model: &mut CatModel) -> Result<(), String> {
        self.advance(); // consume `acyclic`
        let relation = self.parse_expr()?;
        let name = self.parse_optional_as_name();
        model.axioms.push(CatAxiom::Acyclic { relation, name });
        Ok(())
    }

    fn parse_axiom_irreflexive(&mut self, model: &mut CatModel) -> Result<(), String> {
        self.advance(); // consume `irreflexive`
        let relation = self.parse_expr()?;
        let name = self.parse_optional_as_name();
        model.axioms.push(CatAxiom::Irreflexive { relation, name });
        Ok(())
    }

    fn parse_axiom_empty(&mut self, model: &mut CatModel) -> Result<(), String> {
        self.advance(); // consume `empty`
        let relation = self.parse_expr()?;
        let name = self.parse_optional_as_name();
        model.axioms.push(CatAxiom::Empty { relation, name });
        Ok(())
    }

    fn parse_optional_as_name(&mut self) -> Option<String> {
        if matches!(self.peek(), Token::As) {
            self.advance();
            self.expect_ident().ok()
        } else {
            None
        }
    }

    fn parse_include(&mut self, model: &mut CatModel) -> Result<(), String> {
        self.advance(); // consume `include`
        match self.advance() {
            Token::StringLit(s) => { model.includes.push(s); Ok(()) }
            other => Err(format!("expected string after include, got {:?}", other)),
        }
    }

    fn skip_show_unshow(&mut self) {
        self.advance(); // consume show/unshow
        // Skip until next statement keyword or EOF.
        while !self.at_eof() {
            match self.peek() {
                Token::Let | Token::Acyclic | Token::Irreflexive
                | Token::Empty | Token::Include | Token::Show | Token::Unshow => break,
                _ => { self.advance(); }
            }
        }
    }

    // -- expression parsing (precedence climbing) --

    fn parse_expr(&mut self) -> Result<CatRelation, String> {
        self.parse_union()
    }

    // Lowest precedence: union `|`
    fn parse_union(&mut self) -> Result<CatRelation, String> {
        let mut left = self.parse_difference()?;
        while matches!(self.peek(), Token::Pipe) {
            self.advance();
            let right = self.parse_difference()?;
            left = CatRelation::Union(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // Difference `\`
    fn parse_difference(&mut self) -> Result<CatRelation, String> {
        let mut left = self.parse_intersection()?;
        while matches!(self.peek(), Token::Backslash) {
            self.advance();
            let right = self.parse_intersection()?;
            left = CatRelation::Difference(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // Intersection `&`
    fn parse_intersection(&mut self) -> Result<CatRelation, String> {
        let mut left = self.parse_sequence()?;
        while matches!(self.peek(), Token::Ampersand) {
            self.advance();
            let right = self.parse_sequence()?;
            left = CatRelation::Intersection(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // Sequence `;`
    fn parse_sequence(&mut self) -> Result<CatRelation, String> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek(), Token::Semicolon) {
            self.advance();
            let right = self.parse_unary()?;
            left = CatRelation::Sequence(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // Unary prefix: `~`
    fn parse_unary(&mut self) -> Result<CatRelation, String> {
        if matches!(self.peek(), Token::Tilde) {
            self.advance();
            let inner = self.parse_postfix()?;
            Ok(CatRelation::Complement(Box::new(inner)))
        } else {
            self.parse_postfix()
        }
    }

    // Postfix: `+`, `*`, `?`, `^-1`
    fn parse_postfix(&mut self) -> Result<CatRelation, String> {
        let mut expr = self.parse_atom()?;
        loop {
            match self.peek() {
                Token::Plus => {
                    self.advance();
                    expr = CatRelation::TransitiveClosure(Box::new(expr));
                }
                Token::Star => {
                    self.advance();
                    expr = CatRelation::ReflexiveTransitiveClosure(Box::new(expr));
                }
                Token::Question => {
                    self.advance();
                    expr = CatRelation::Optional(Box::new(expr));
                }
                Token::Caret => {
                    // ^-1
                    self.advance();
                    self.expect(&Token::Minus)?;
                    match self.advance() {
                        Token::Ident(s) if s == "1" => {}
                        Token::One => {}
                        other => return Err(format!("expected '1' after '^-', got {:?}", other)),
                    }
                    expr = CatRelation::Inverse(Box::new(expr));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    // Atoms: parenthesized expr, guard, ident, zero
    fn parse_atom(&mut self) -> Result<CatRelation, String> {
        match self.peek().clone() {
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&Token::RBracket)?;
                Ok(CatRelation::Guard(name))
            }
            Token::Zero => {
                self.advance();
                Ok(CatRelation::Empty)
            }
            Token::One => {
                self.advance();
                Ok(CatRelation::Identity)
            }
            Token::Ident(ref s) if s == "id" => {
                self.advance();
                Ok(CatRelation::Identity)
            }
            Token::Ident(ref s) => {
                let name = s.clone();
                self.advance();
                if let Some(builtin) = CatBuiltin::from_str_opt(&name) {
                    Ok(CatRelation::Builtin(builtin))
                } else {
                    Ok(CatRelation::Named(name))
                }
            }
            other => Err(format!("unexpected token in expression: {:?}", other)),
        }
    }
}

// ---------------------------------------------------------------------------
// CatTranslator — bidirectional conversion
// ---------------------------------------------------------------------------

/// Bidirectional translator between Litmus∞ [`MemoryModel`] and `.cat` format.
pub struct CatTranslator;

impl CatTranslator {
    /// Export a Litmus∞ [`MemoryModel`] to `.cat` format string.
    pub fn to_cat(model: &MemoryModel) -> String {
        let mut out = String::new();
        out.push_str(&format!("\"{}\"\n\n", model.name));

        for dr in &model.derived_relations {
            out.push_str(&format!("let {} = {}\n", dr.name, Self::expr_to_cat(&dr.expr)));
        }

        if !model.derived_relations.is_empty() && !model.constraints.is_empty() {
            out.push('\n');
        }

        for c in &model.constraints {
            match c {
                Constraint::Acyclic(e, _) => {
                    out.push_str(&format!("acyclic {}\n", Self::expr_to_cat(e)));
                }
                Constraint::Irreflexive(e, _) => {
                    out.push_str(&format!("irreflexive {}\n", Self::expr_to_cat(e)));
                }
                Constraint::Empty(e, _) => {
                    out.push_str(&format!("empty {}\n", Self::expr_to_cat(e)));
                }
            }
        }

        out
    }

    /// Convert a [`RelationExpr`] to `.cat` syntax string.
    fn expr_to_cat(expr: &RelationExpr) -> String {
        match expr {
            RelationExpr::Base(n) => n.clone(),
            RelationExpr::Seq(a, b) => format!("({} ; {})", Self::expr_to_cat(a), Self::expr_to_cat(b)),
            RelationExpr::Union(a, b) => format!("({} | {})", Self::expr_to_cat(a), Self::expr_to_cat(b)),
            RelationExpr::Inter(a, b) => format!("({} & {})", Self::expr_to_cat(a), Self::expr_to_cat(b)),
            RelationExpr::Diff(a, b) => format!("({} \\ {})", Self::expr_to_cat(a), Self::expr_to_cat(b)),
            RelationExpr::Inverse(a) => format!("{}^-1", Self::expr_to_cat(a)),
            RelationExpr::Plus(a) => format!("{}+", Self::expr_to_cat(a)),
            RelationExpr::Star(a) => format!("{}*", Self::expr_to_cat(a)),
            RelationExpr::Optional(a) => format!("{}?", Self::expr_to_cat(a)),
            RelationExpr::Identity => "id".to_string(),
            RelationExpr::Filter(p) => format!("[{}]", p),
            RelationExpr::Empty => "0".to_string(),
        }
    }

    /// Import a [`CatModel`] into a Litmus∞ [`MemoryModel`].
    pub fn from_cat(cat: &CatModel) -> MemoryModel {
        let mut model = MemoryModel::new(&cat.name);

        for binding in &cat.bindings {
            let expr = Self::cat_relation_to_expr(&binding.expr);
            model.add_derived(&binding.name, expr, "");
        }

        for axiom in &cat.axioms {
            match axiom {
                CatAxiom::Acyclic { relation, .. } => {
                    model.add_acyclic(Self::cat_relation_to_expr(relation));
                }
                CatAxiom::Irreflexive { relation, .. } => {
                    model.add_irreflexive(Self::cat_relation_to_expr(relation));
                }
                CatAxiom::Empty { relation, .. } => {
                    model.add_empty(Self::cat_relation_to_expr(relation));
                }
            }
        }

        model
    }

    /// Convert a [`CatRelation`] to a Litmus∞ [`RelationExpr`].
    fn cat_relation_to_expr(rel: &CatRelation) -> RelationExpr {
        match rel {
            CatRelation::Named(n) => RelationExpr::base(n),
            CatRelation::Builtin(b) => RelationExpr::base(b.name()),
            CatRelation::Union(a, b) => {
                RelationExpr::union(
                    Self::cat_relation_to_expr(a),
                    Self::cat_relation_to_expr(b),
                )
            }
            CatRelation::Intersection(a, b) => {
                RelationExpr::inter(
                    Self::cat_relation_to_expr(a),
                    Self::cat_relation_to_expr(b),
                )
            }
            CatRelation::Sequence(a, b) => {
                RelationExpr::seq(
                    Self::cat_relation_to_expr(a),
                    Self::cat_relation_to_expr(b),
                )
            }
            CatRelation::Complement(a) => {
                // .cat complement ~R is not directly in RelationExpr;
                // approximate as universal \ R.
                let inner = Self::cat_relation_to_expr(a);
                RelationExpr::diff(RelationExpr::Identity, inner)
            }
            CatRelation::Inverse(a) => {
                RelationExpr::inverse(Self::cat_relation_to_expr(a))
            }
            CatRelation::TransitiveClosure(a) => {
                RelationExpr::plus(Self::cat_relation_to_expr(a))
            }
            CatRelation::ReflexiveTransitiveClosure(a) => {
                RelationExpr::star(Self::cat_relation_to_expr(a))
            }
            CatRelation::Difference(a, b) => {
                RelationExpr::diff(
                    Self::cat_relation_to_expr(a),
                    Self::cat_relation_to_expr(b),
                )
            }
            CatRelation::Cartesian(a, b) => {
                // Approximate as sequence through identity.
                RelationExpr::seq(
                    Self::cat_relation_to_expr(a),
                    Self::cat_relation_to_expr(b),
                )
            }
            CatRelation::Identity => RelationExpr::Identity,
            CatRelation::Optional(a) => {
                RelationExpr::optional(Self::cat_relation_to_expr(a))
            }
            CatRelation::Guard(name) => {
                // Map common guard names to predicates.
                use crate::checker::memory_model::PredicateExpr;
                match name.as_str() {
                    "R" | "Reads" => RelationExpr::filter(PredicateExpr::IsRead),
                    "W" | "Writes" => RelationExpr::filter(PredicateExpr::IsWrite),
                    "F" | "Fences" => RelationExpr::filter(PredicateExpr::IsFence),
                    "RMW" => RelationExpr::filter(PredicateExpr::IsRMW),
                    _ => RelationExpr::base(name),
                }
            }
            CatRelation::Empty => RelationExpr::Empty,
        }
    }

    /// Compare a [`MemoryModel`] with a [`CatModel`] and return a list of
    /// differences as human-readable strings.
    pub fn compare(litmus_model: &MemoryModel, cat_model: &CatModel) -> Vec<String> {
        let mut diffs = Vec::new();

        if litmus_model.name != cat_model.name {
            diffs.push(format!(
                "name: litmus∞='{}' vs cat='{}'",
                litmus_model.name, cat_model.name
            ));
        }

        // Compare number of derived relations.
        let litmus_names: Vec<&str> = litmus_model
            .derived_relations
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        let cat_names = cat_model.defined_names();

        for name in &litmus_names {
            if !cat_names.contains(name) {
                diffs.push(format!("relation '{}' in litmus∞ but not in .cat", name));
            }
        }
        for name in &cat_names {
            if !litmus_names.contains(name) {
                diffs.push(format!("relation '{}' in .cat but not in litmus∞", name));
            }
        }

        // Compare constraint counts.
        let litmus_acyclic = litmus_model
            .constraints
            .iter()
            .filter(|c| matches!(c, Constraint::Acyclic(..)))
            .count();
        let cat_acyclic = cat_model
            .axioms
            .iter()
            .filter(|a| matches!(a, CatAxiom::Acyclic { .. }))
            .count();
        if litmus_acyclic != cat_acyclic {
            diffs.push(format!(
                "acyclicity constraints: litmus∞={} vs cat={}",
                litmus_acyclic, cat_acyclic
            ));
        }

        let litmus_irr = litmus_model
            .constraints
            .iter()
            .filter(|c| matches!(c, Constraint::Irreflexive(..)))
            .count();
        let cat_irr = cat_model
            .axioms
            .iter()
            .filter(|a| matches!(a, CatAxiom::Irreflexive { .. }))
            .count();
        if litmus_irr != cat_irr {
            diffs.push(format!(
                "irreflexivity constraints: litmus∞={} vs cat={}",
                litmus_irr, cat_irr
            ));
        }

        diffs
    }
}

// ---------------------------------------------------------------------------
// Standard .cat model definitions (built-in strings)
// ---------------------------------------------------------------------------

/// Built-in `.cat` model definitions for standard memory models.
pub struct StandardCatModels;

impl StandardCatModels {
    /// SC (Sequential Consistency) in `.cat` format.
    pub fn sc() -> &'static str {
        r#""SC"

let com = rf | co | fr

acyclic po | com as sc
"#
    }

    /// TSO (Total Store Order / x86) in `.cat` format.
    pub fn tso() -> &'static str {
        r#""TSO"

include "cos.cat"

let com = rf | co | fr
let ppo = [R];po;[R] | [R];po;[W] | [W];po;[W] | po-loc
let fence-order = po ; [F] ; po
let ghb = ppo | fence-order | rfe | co | fr

acyclic ghb as tso
irreflexive (fre ; rfe) as uniproc
"#
    }

    /// PSO (Partial Store Order / SPARC) in `.cat` format.
    pub fn pso() -> &'static str {
        r#""PSO"

let com = rf | co | fr
let ppo = [R];po;[R] | [R];po;[W] | po-loc
let fence-order = po ; [F] ; po
let ghb = ppo | fence-order | rfe | co | fr

acyclic ghb as pso
"#
    }

    /// ARMv8 in `.cat` format (simplified).
    pub fn arm() -> &'static str {
        r#""ARMv8"

include "cos.cat"

let com = rf | co | fr
let dp = addr | data
let rdw = po-loc & (fre ; rfe)
let detour = po-loc & (coe ; rfe)
let ii0 = dp | rdw | rfi
let ci0 = ctrl ; [W] | detour
let ic0 = 0
let cc0 = dp | po-loc | ctrl | (addr ; po)
let rec ii = ii0 | ci | (ic ; ci) | (ii ; ii)
and ic = ic0 | ii | cc | (ic ; cc)
and ci = ci0 | (ci ; ii) | (cc ; ci)
and cc = cc0 | ci | (ci ; ic) | (cc ; cc)
let ppo = [R] ; (ii | ic) | [W] ; (ci | cc)
let fence = po ; [F] ; po
let hb = ppo | fence | rfe

acyclic hb as no-thin-air
acyclic co | prop as observation
irreflexive (fre ; prop ; hb*) as propagation
"#
    }

    /// RISC-V (RVWMO) in `.cat` format (simplified).
    pub fn riscv() -> &'static str {
        r#""RISC-V"

include "cos.cat"

let com = rf | co | fr
let ppo = [R];po;[R] | [R];po;[W] | [W];po;[W] | [W];po;[R] & dep | po-loc | addr;po | data;po
let fence = po ; [F] ; po
let hb = ppo | fence | rfe

acyclic hb as rvwmo
"#
    }

    /// PTX (NVIDIA GPU) in `.cat` format (simplified).
    pub fn ptx() -> &'static str {
        r#""PTX"

let com = rf | co | fr
let ppo = po-loc
let morph = ppo | rfe | co | fr

acyclic morph as ptx
"#
    }

    /// Get a standard model by name.
    pub fn get(name: &str) -> Option<&'static str> {
        match name.to_uppercase().as_str() {
            "SC" => Some(Self::sc()),
            "TSO" => Some(Self::tso()),
            "PSO" => Some(Self::pso()),
            "ARM" | "ARMV8" => Some(Self::arm()),
            "RISCV" | "RISC-V" | "RVWMO" => Some(Self::riscv()),
            "PTX" => Some(Self::ptx()),
            _ => None,
        }
    }

    /// List available standard models.
    pub fn available() -> Vec<&'static str> {
        vec!["SC", "TSO", "PSO", "ARM", "RISC-V", "PTX"]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sc_model() {
        let model = CatFileParser::parse_str(StandardCatModels::sc()).unwrap();
        assert_eq!(model.name, "SC");
        assert_eq!(model.bindings.len(), 1);
        assert_eq!(model.bindings[0].name, "com");
        assert_eq!(model.axioms.len(), 1);
        assert!(matches!(&model.axioms[0], CatAxiom::Acyclic { name: Some(n), .. } if n == "sc"));
    }

    #[test]
    fn parse_tso_model() {
        let model = CatFileParser::parse_str(StandardCatModels::tso()).unwrap();
        assert_eq!(model.name, "TSO");
        assert!(model.bindings.len() >= 4);
        assert!(model.axioms.len() >= 2);
        assert!(model.includes.contains(&"cos.cat".to_string()));
    }

    #[test]
    fn parse_arm_model() {
        let model = CatFileParser::parse_str(StandardCatModels::arm()).unwrap();
        assert_eq!(model.name, "ARMv8");
        assert!(model.bindings.len() >= 5);
        assert_eq!(model.axioms.len(), 3);
    }

    #[test]
    fn parse_pso_model() {
        let model = CatFileParser::parse_str(StandardCatModels::pso()).unwrap();
        assert_eq!(model.name, "PSO");
        assert_eq!(model.bindings.len(), 4);
        assert_eq!(model.axioms.len(), 1);
    }

    #[test]
    fn parse_riscv_model() {
        let model = CatFileParser::parse_str(StandardCatModels::riscv()).unwrap();
        assert_eq!(model.name, "RISC-V");
        assert!(model.bindings.len() >= 3);
        assert_eq!(model.axioms.len(), 1);
    }

    #[test]
    fn parse_ptx_model() {
        let model = CatFileParser::parse_str(StandardCatModels::ptx()).unwrap();
        assert_eq!(model.name, "PTX");
        assert_eq!(model.axioms.len(), 1);
    }

    #[test]
    fn parse_empty_relation() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = 0
        "#).unwrap();
        assert_eq!(model.bindings[0].expr, CatRelation::Empty);
    }

    #[test]
    fn parse_union_expression() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let com = rf | co | fr
        "#).unwrap();
        // com should be Union(Union(rf, co), fr) due to left-associativity.
        let expr = &model.bindings[0].expr;
        match expr {
            CatRelation::Union(left, right) => {
                assert!(matches!(right.as_ref(), CatRelation::Builtin(CatBuiltin::Fr)));
                match left.as_ref() {
                    CatRelation::Union(a, b) => {
                        assert!(matches!(a.as_ref(), CatRelation::Builtin(CatBuiltin::Rf)));
                        assert!(matches!(b.as_ref(), CatRelation::Builtin(CatBuiltin::Co)));
                    }
                    _ => panic!("expected nested Union"),
                }
            }
            _ => panic!("expected Union, got {:?}", expr),
        }
    }

    #[test]
    fn parse_sequence_expression() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = po ; rf
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Sequence(a, b) => {
                assert!(matches!(a.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
                assert!(matches!(b.as_ref(), CatRelation::Builtin(CatBuiltin::Rf)));
            }
            other => panic!("expected Sequence, got {:?}", other),
        }
    }

    #[test]
    fn parse_inverse() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = rf^-1
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Inverse(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Builtin(CatBuiltin::Rf)));
            }
            other => panic!("expected Inverse, got {:?}", other),
        }
    }

    #[test]
    fn parse_transitive_closure() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = po+
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::TransitiveClosure(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
            }
            other => panic!("expected TransitiveClosure, got {:?}", other),
        }
    }

    #[test]
    fn parse_reflexive_transitive_closure() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = po*
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::ReflexiveTransitiveClosure(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
            }
            other => panic!("expected ReflexiveTransitiveClosure, got {:?}", other),
        }
    }

    #[test]
    fn parse_complement() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = ~po
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Complement(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
            }
            other => panic!("expected Complement, got {:?}", other),
        }
    }

    #[test]
    fn parse_difference() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = po \ rf
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Difference(a, b) => {
                assert!(matches!(a.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
                assert!(matches!(b.as_ref(), CatRelation::Builtin(CatBuiltin::Rf)));
            }
            other => panic!("expected Difference, got {:?}", other),
        }
    }

    #[test]
    fn parse_guard_expression() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let ppo = [R] ; po ; [W]
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Sequence(left, right) => {
                assert!(matches!(right.as_ref(), CatRelation::Guard(ref s) if s == "W"));
                match left.as_ref() {
                    CatRelation::Sequence(a, b) => {
                        assert!(matches!(a.as_ref(), CatRelation::Guard(ref s) if s == "R"));
                        assert!(matches!(b.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
                    }
                    _ => panic!("expected nested Sequence"),
                }
            }
            other => panic!("expected Sequence, got {:?}", other),
        }
    }

    #[test]
    fn parse_parenthesized() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = (po | rf) ; co
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Sequence(a, b) => {
                assert!(matches!(a.as_ref(), CatRelation::Union(..)));
                assert!(matches!(b.as_ref(), CatRelation::Builtin(CatBuiltin::Co)));
            }
            other => panic!("expected Sequence, got {:?}", other),
        }
    }

    #[test]
    fn parse_optional() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = po?
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::Optional(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Builtin(CatBuiltin::Po)));
            }
            other => panic!("expected Optional, got {:?}", other),
        }
    }

    #[test]
    fn parse_recursive_let() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let rec hb = po | rf | (hb ; hb)
        "#).unwrap();
        assert!(model.bindings[0].is_recursive);
        assert_eq!(model.bindings[0].name, "hb");
    }

    #[test]
    fn parse_multi_let_with_and() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let rec ii = ii0 | ci
            and ic = ic0 | ii
            and ci = ci0
            and cc = cc0 | ci
        "#).unwrap();
        assert_eq!(model.bindings.len(), 4);
        assert_eq!(model.bindings[0].name, "ii");
        assert_eq!(model.bindings[1].name, "ic");
        assert_eq!(model.bindings[2].name, "ci");
        assert_eq!(model.bindings[3].name, "cc");
    }

    #[test]
    fn parse_comment() {
        let model = CatFileParser::parse_str(r#"
            "test"
            (* this is a comment *)
            let r = po
        "#).unwrap();
        assert_eq!(model.bindings.len(), 1);
    }

    #[test]
    fn roundtrip_sc_model() {
        let cat_str = StandardCatModels::sc();
        let cat_model = CatFileParser::parse_str(cat_str).unwrap();
        let mem_model = CatTranslator::from_cat(&cat_model);
        assert_eq!(mem_model.name, "SC");
        assert_eq!(mem_model.derived_relations.len(), 1);
        assert_eq!(mem_model.constraints.len(), 1);
    }

    #[test]
    fn roundtrip_tso_model() {
        let cat_str = StandardCatModels::tso();
        let cat_model = CatFileParser::parse_str(cat_str).unwrap();
        let mem_model = CatTranslator::from_cat(&cat_model);
        assert_eq!(mem_model.name, "TSO");
        assert!(mem_model.derived_relations.len() >= 4);
    }

    #[test]
    fn to_cat_and_back() {
        let mut model = MemoryModel::new("Test");
        model.add_derived(
            "com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "communication",
        );
        model.add_acyclic(
            RelationExpr::union(RelationExpr::base("po"), RelationExpr::base("com")),
        );

        let cat_str = CatTranslator::to_cat(&model);
        assert!(cat_str.contains("\"Test\""));
        assert!(cat_str.contains("let com"));
        assert!(cat_str.contains("acyclic"));
    }

    #[test]
    fn compare_models() {
        let cat_model = CatFileParser::parse_str(StandardCatModels::sc()).unwrap();
        let mem_model = CatTranslator::from_cat(&cat_model);
        let diffs = CatTranslator::compare(&mem_model, &cat_model);
        assert!(diffs.is_empty(), "identical models should have no diffs: {:?}", diffs);
    }

    #[test]
    fn compare_models_with_differences() {
        let cat_model = CatFileParser::parse_str(StandardCatModels::tso()).unwrap();
        let mut mem_model = MemoryModel::new("SC");
        mem_model.add_derived("com", RelationExpr::base("rf"), "");
        mem_model.add_acyclic(RelationExpr::base("com"));

        let diffs = CatTranslator::compare(&mem_model, &cat_model);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn cat_model_display() {
        let model = CatFileParser::parse_str(StandardCatModels::sc()).unwrap();
        let display = format!("{}", model);
        assert!(display.contains("\"SC\""));
        assert!(display.contains("let com"));
    }

    #[test]
    fn cat_builtin_roundtrip() {
        for name in &["po", "rf", "co", "fr", "id", "ext", "int", "loc", "addr", "data", "ctrl", "rmw", "amo"] {
            let b = CatBuiltin::from_str_opt(name).unwrap();
            assert_eq!(b.name(), *name);
        }
    }

    #[test]
    fn standard_models_available() {
        let models = StandardCatModels::available();
        assert!(models.len() >= 6);
        for name in &models {
            assert!(StandardCatModels::get(name).is_some());
        }
    }

    #[test]
    fn cat_relation_display() {
        let r = CatRelation::Union(
            Box::new(CatRelation::Builtin(CatBuiltin::Rf)),
            Box::new(CatRelation::Builtin(CatBuiltin::Co)),
        );
        assert_eq!(format!("{}", r), "(rf | co)");
    }

    #[test]
    fn cat_expr_wrapper() {
        let expr = CatExpr::new(CatRelation::Builtin(CatBuiltin::Po));
        assert_eq!(format!("{}", expr), "po");
    }

    #[test]
    fn cat_axiom_display() {
        let axiom = CatAxiom::Acyclic {
            relation: CatRelation::Builtin(CatBuiltin::Po),
            name: Some("order".to_string()),
        };
        assert_eq!(format!("{}", axiom), "acyclic po as order");
    }

    #[test]
    fn cat_binding_display() {
        let binding = CatBinding {
            name: "com".to_string(),
            expr: CatRelation::Builtin(CatBuiltin::Rf),
            is_recursive: false,
        };
        assert_eq!(format!("{}", binding), "let com = rf");
    }

    #[test]
    fn cat_binding_display_recursive() {
        let binding = CatBinding {
            name: "hb".to_string(),
            expr: CatRelation::Builtin(CatBuiltin::Po),
            is_recursive: true,
        };
        assert_eq!(format!("{}", binding), "let rec hb = po");
    }

    #[test]
    fn parse_identity_relation() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = id
        "#).unwrap();
        assert_eq!(model.bindings[0].expr, CatRelation::Identity);
    }

    #[test]
    fn parse_nested_postfix() {
        let model = CatFileParser::parse_str(r#"
            "test"
            let r = (po | rf)+
        "#).unwrap();
        match &model.bindings[0].expr {
            CatRelation::TransitiveClosure(inner) => {
                assert!(matches!(inner.as_ref(), CatRelation::Union(..)));
            }
            other => panic!("expected TransitiveClosure, got {:?}", other),
        }
    }

    #[test]
    fn referenced_builtins() {
        let model = CatFileParser::parse_str(StandardCatModels::sc()).unwrap();
        let builtins = model.referenced_builtins();
        assert!(builtins.contains(&CatBuiltin::Rf));
        assert!(builtins.contains(&CatBuiltin::Co));
        assert!(builtins.contains(&CatBuiltin::Fr));
    }

    #[test]
    fn get_binding_lookup() {
        let model = CatFileParser::parse_str(StandardCatModels::sc()).unwrap();
        assert!(model.get_binding("com").is_some());
        assert!(model.get_binding("nonexistent").is_none());
    }
}
