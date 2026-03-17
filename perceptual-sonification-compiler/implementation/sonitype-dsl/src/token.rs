//! Lexer token types for the SoniType DSL.
//!
//! Defines the complete set of tokens produced by the lexer, including keywords,
//! operators, delimiters, literals, and type keywords used in perceptual
//! sonification specifications.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Source Position ─────────────────────────────────────────────────────────

/// A position in source text (1-indexed line and column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A span of source text, from `start` to `end` (inclusive of start, exclusive of end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: if self.start.offset <= other.start.offset { self.start } else { other.start },
            end: if self.end.offset >= other.end.offset { self.end } else { other.end },
        }
    }

    pub fn dummy() -> Self {
        Self::default()
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

// ─── Token Kind ──────────────────────────────────────────────────────────────

/// The kind of a lexer token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenKind {
    // ── Keywords ──
    Stream,
    Mapping,
    Compose,
    Data,
    Let,
    In,
    With,
    Where,
    If,
    Then,
    Else,
    Import,
    Export,
    Spec,
    True,
    False,

    // ── Type keywords ──
    TyStream,
    TyMapping,
    TyMultiStream,
    TyData,
    TySonificationSpec,
    TyPitch,
    TyTimbre,
    TyPan,
    TyAmplitude,
    TyDuration,
    TyFloat,
    TyInt,
    TyBool,
    TyString,

    // ── Literals ──
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),

    // ── Identifier ──
    Ident(String),

    // ── Arithmetic operators ──
    Plus,
    Minus,
    Star,
    Slash,

    // ── Logical operators ──
    PipePipe,   // ||
    AmpAmp,     // &&
    Bang,       // !

    // ── Comparison operators ──
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=

    // ── Special operators ──
    DotDot,     // ..
    PipeGt,     // |>
    At,         // @
    Eq,         // =
    Dot,        // .
    Arrow,      // ->

    // ── Delimiters ──
    LBrace,     // {
    RBrace,     // }
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    Comma,
    Semicolon,
    Colon,

    // ── Special ──
    Eof,
}

impl TokenKind {
    /// Returns `true` if this token is a keyword.
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Stream
                | TokenKind::Mapping
                | TokenKind::Compose
                | TokenKind::Data
                | TokenKind::Let
                | TokenKind::In
                | TokenKind::With
                | TokenKind::Where
                | TokenKind::If
                | TokenKind::Then
                | TokenKind::Else
                | TokenKind::Import
                | TokenKind::Export
                | TokenKind::Spec
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Returns `true` if this token is a type keyword.
    pub fn is_type_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::TyStream
                | TokenKind::TyMapping
                | TokenKind::TyMultiStream
                | TokenKind::TyData
                | TokenKind::TySonificationSpec
                | TokenKind::TyPitch
                | TokenKind::TyTimbre
                | TokenKind::TyPan
                | TokenKind::TyAmplitude
                | TokenKind::TyDuration
                | TokenKind::TyFloat
                | TokenKind::TyInt
                | TokenKind::TyBool
                | TokenKind::TyString
        )
    }

    /// Returns `true` if this token is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntLit(_) | TokenKind::FloatLit(_) | TokenKind::StringLit(_)
                | TokenKind::True | TokenKind::False
        )
    }

    /// Returns `true` if this token is an operator.
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::PipePipe
                | TokenKind::AmpAmp
                | TokenKind::Bang
                | TokenKind::EqEq
                | TokenKind::BangEq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::LtEq
                | TokenKind::GtEq
                | TokenKind::DotDot
                | TokenKind::PipeGt
                | TokenKind::At
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Keywords
            TokenKind::Stream => write!(f, "stream"),
            TokenKind::Mapping => write!(f, "mapping"),
            TokenKind::Compose => write!(f, "compose"),
            TokenKind::Data => write!(f, "data"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::In => write!(f, "in"),
            TokenKind::With => write!(f, "with"),
            TokenKind::Where => write!(f, "where"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Then => write!(f, "then"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::Export => write!(f, "export"),
            TokenKind::Spec => write!(f, "spec"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),

            // Type keywords
            TokenKind::TyStream => write!(f, "Stream"),
            TokenKind::TyMapping => write!(f, "Mapping"),
            TokenKind::TyMultiStream => write!(f, "MultiStream"),
            TokenKind::TyData => write!(f, "Data"),
            TokenKind::TySonificationSpec => write!(f, "SonificationSpec"),
            TokenKind::TyPitch => write!(f, "Pitch"),
            TokenKind::TyTimbre => write!(f, "Timbre"),
            TokenKind::TyPan => write!(f, "Pan"),
            TokenKind::TyAmplitude => write!(f, "Amplitude"),
            TokenKind::TyDuration => write!(f, "Duration"),
            TokenKind::TyFloat => write!(f, "Float"),
            TokenKind::TyInt => write!(f, "Int"),
            TokenKind::TyBool => write!(f, "Bool"),
            TokenKind::TyString => write!(f, "String"),

            // Literals
            TokenKind::IntLit(v) => write!(f, "{v}"),
            TokenKind::FloatLit(v) => write!(f, "{v}"),
            TokenKind::StringLit(v) => write!(f, "\"{v}\""),

            // Identifier
            TokenKind::Ident(s) => write!(f, "{s}"),

            // Operators
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::PipePipe => write!(f, "||"),
            TokenKind::AmpAmp => write!(f, "&&"),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::BangEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::PipeGt => write!(f, "|>"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Arrow => write!(f, "->"),

            // Delimiters
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),

            TokenKind::Eof => write!(f, "<EOF>"),
        }
    }
}

// ─── Token ───────────────────────────────────────────────────────────────────

/// A token produced by the lexer, with its source span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn eof(offset: usize, line: usize, column: usize) -> Self {
        let pos = Position::new(line, column, offset);
        Self { kind: TokenKind::Eof, span: Span::new(pos, pos) }
    }

    /// Returns `true` if this token matches the given kind (ignoring literal values).
    pub fn is(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(kind)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @ {}", self.kind, self.span)
    }
}

// ─── Keyword lookup ──────────────────────────────────────────────────────────

/// Attempt to match an identifier string to a keyword or type keyword.
pub fn lookup_keyword(ident: &str) -> Option<TokenKind> {
    match ident {
        // Language keywords
        "stream" => Some(TokenKind::Stream),
        "mapping" => Some(TokenKind::Mapping),
        "compose" => Some(TokenKind::Compose),
        "data" => Some(TokenKind::Data),
        "let" => Some(TokenKind::Let),
        "in" => Some(TokenKind::In),
        "with" => Some(TokenKind::With),
        "where" => Some(TokenKind::Where),
        "if" => Some(TokenKind::If),
        "then" => Some(TokenKind::Then),
        "else" => Some(TokenKind::Else),
        "import" => Some(TokenKind::Import),
        "export" => Some(TokenKind::Export),
        "spec" => Some(TokenKind::Spec),
        "true" => Some(TokenKind::True),
        "false" => Some(TokenKind::False),

        // Type keywords (capitalized)
        "Stream" => Some(TokenKind::TyStream),
        "Mapping" => Some(TokenKind::TyMapping),
        "MultiStream" => Some(TokenKind::TyMultiStream),
        "Data" => Some(TokenKind::TyData),
        "SonificationSpec" => Some(TokenKind::TySonificationSpec),
        "Pitch" => Some(TokenKind::TyPitch),
        "Timbre" => Some(TokenKind::TyTimbre),
        "Pan" => Some(TokenKind::TyPan),
        "Amplitude" => Some(TokenKind::TyAmplitude),
        "Duration" => Some(TokenKind::TyDuration),
        "Float" => Some(TokenKind::TyFloat),
        "Int" => Some(TokenKind::TyInt),
        "Bool" => Some(TokenKind::TyBool),
        "String" => Some(TokenKind::TyString),

        _ => None,
    }
}

/// Returns the list of all DSL keywords as strings.
pub fn all_keywords() -> &'static [&'static str] {
    &[
        "stream", "mapping", "compose", "data", "let", "in", "with", "where",
        "if", "then", "else", "import", "export", "spec", "true", "false",
    ]
}

/// Returns the list of all type keywords as strings.
pub fn all_type_keywords() -> &'static [&'static str] {
    &[
        "Stream", "Mapping", "MultiStream", "Data", "SonificationSpec",
        "Pitch", "Timbre", "Pan", "Amplitude", "Duration",
        "Float", "Int", "Bool", "String",
    ]
}

// ─── LexError ────────────────────────────────────────────────────────────────

/// An error encountered during lexing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl LexError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self { message: message.into(), span }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lex error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for LexError {}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        let pos = Position::new(1, 5, 4);
        assert_eq!(format!("{pos}"), "1:5");
    }

    #[test]
    fn test_span_display() {
        let span = Span::new(Position::new(1, 1, 0), Position::new(1, 5, 4));
        assert_eq!(format!("{span}"), "1:1-1:5");
    }

    #[test]
    fn test_span_merge() {
        let a = Span::new(Position::new(1, 1, 0), Position::new(1, 5, 4));
        let b = Span::new(Position::new(1, 8, 7), Position::new(1, 12, 11));
        let merged = a.merge(b);
        assert_eq!(merged.start.offset, 0);
        assert_eq!(merged.end.offset, 11);
    }

    #[test]
    fn test_keyword_lookup() {
        assert_eq!(lookup_keyword("stream"), Some(TokenKind::Stream));
        assert_eq!(lookup_keyword("let"), Some(TokenKind::Let));
        assert_eq!(lookup_keyword("true"), Some(TokenKind::True));
        assert_eq!(lookup_keyword("false"), Some(TokenKind::False));
        assert_eq!(lookup_keyword("foo"), None);
    }

    #[test]
    fn test_type_keyword_lookup() {
        assert_eq!(lookup_keyword("Stream"), Some(TokenKind::TyStream));
        assert_eq!(lookup_keyword("Mapping"), Some(TokenKind::TyMapping));
        assert_eq!(lookup_keyword("Pitch"), Some(TokenKind::TyPitch));
        assert_eq!(lookup_keyword("Float"), Some(TokenKind::TyFloat));
    }

    #[test]
    fn test_token_kind_is_keyword() {
        assert!(TokenKind::Stream.is_keyword());
        assert!(TokenKind::Let.is_keyword());
        assert!(!TokenKind::Plus.is_keyword());
        assert!(!TokenKind::TyStream.is_keyword());
    }

    #[test]
    fn test_token_kind_is_type_keyword() {
        assert!(TokenKind::TyStream.is_type_keyword());
        assert!(TokenKind::TyPitch.is_type_keyword());
        assert!(!TokenKind::Stream.is_type_keyword());
    }

    #[test]
    fn test_token_kind_is_literal() {
        assert!(TokenKind::IntLit(42).is_literal());
        assert!(TokenKind::FloatLit(3.14).is_literal());
        assert!(TokenKind::StringLit("hi".to_string()).is_literal());
        assert!(TokenKind::True.is_literal());
        assert!(!TokenKind::Ident("x".into()).is_literal());
    }

    #[test]
    fn test_token_kind_is_operator() {
        assert!(TokenKind::Plus.is_operator());
        assert!(TokenKind::PipePipe.is_operator());
        assert!(TokenKind::PipeGt.is_operator());
        assert!(!TokenKind::LBrace.is_operator());
    }

    #[test]
    fn test_token_kind_display() {
        assert_eq!(format!("{}", TokenKind::Stream), "stream");
        assert_eq!(format!("{}", TokenKind::Arrow), "->");
        assert_eq!(format!("{}", TokenKind::PipeGt), "|>");
        assert_eq!(format!("{}", TokenKind::IntLit(42)), "42");
        assert_eq!(format!("{}", TokenKind::Eof), "<EOF>");
    }

    #[test]
    fn test_token_is_discriminant() {
        let tok = Token::new(TokenKind::IntLit(42), Span::dummy());
        assert!(tok.is(&TokenKind::IntLit(0))); // matches discriminant, not value
        assert!(!tok.is(&TokenKind::FloatLit(0.0)));
    }

    #[test]
    fn test_lex_error_display() {
        let err = LexError::new("unexpected character", Span::dummy());
        assert!(format!("{err}").contains("unexpected character"));
    }

    #[test]
    fn test_all_keywords_non_empty() {
        assert!(!all_keywords().is_empty());
        assert!(!all_type_keywords().is_empty());
        // Every keyword string should resolve via lookup
        for kw in all_keywords() {
            assert!(lookup_keyword(kw).is_some(), "keyword '{kw}' not found");
        }
    }
}
