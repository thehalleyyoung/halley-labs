use crate::source_map::Span;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Token Kind ─────────────────────────────────────────────────

/// The kind of token, without span information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenKind {
    // ── Keywords ──
    Jurisdiction,
    Obligation,
    Permission,
    Prohibition,
    Requires,
    Exempts,
    Override,
    Compose,
    Temporal,
    RiskLevel,
    Domain,
    Article,
    Annex,
    Formalizability,
    Strategy,
    Cost,
    If,
    Then,
    Else,
    And,
    Or,
    Not,
    Implies,
    Forall,
    Exists,
    True,
    False,
    Framework,
    Mapping,

    // ── Punctuation ──
    LBrace,    // {
    RBrace,    // }
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    Semicolon, // ;
    Colon,     // :
    Comma,     // ,
    Dot,       // .
    Arrow,     // ->
    FatArrow,  // =>

    // ── Operators ──
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Eq,           // =
    EqEq,         // ==
    NotEq,        // !=
    Lt,           // <
    Gt,           // >
    LtEq,         // <=
    GtEq,         // >=
    Conjunction,   // ⊗  or  &*
    Disjunction,   // ⊕  or  |+
    OverrideOp,    // ▷  or  >>
    ExceptionOp,   // ⊘  or  \-

    // ── Literals ──
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    Ident(String),
    DateLit(String), // YYYY-MM-DD

    // ── Special ──
    Eof,
}

impl TokenKind {
    /// Returns a human-readable name for this token kind.
    pub fn description(&self) -> &str {
        match self {
            Self::Jurisdiction => "'jurisdiction'",
            Self::Obligation => "'obligation'",
            Self::Permission => "'permission'",
            Self::Prohibition => "'prohibition'",
            Self::Requires => "'requires'",
            Self::Exempts => "'exempts'",
            Self::Override => "'override'",
            Self::Compose => "'compose'",
            Self::Temporal => "'temporal'",
            Self::RiskLevel => "'risk_level'",
            Self::Domain => "'domain'",
            Self::Article => "'article'",
            Self::Annex => "'annex'",
            Self::Formalizability => "'formalizability'",
            Self::Strategy => "'strategy'",
            Self::Cost => "'cost'",
            Self::If => "'if'",
            Self::Then => "'then'",
            Self::Else => "'else'",
            Self::And => "'and'",
            Self::Or => "'or'",
            Self::Not => "'not'",
            Self::Implies => "'implies'",
            Self::Forall => "'forall'",
            Self::Exists => "'exists'",
            Self::True => "'true'",
            Self::False => "'false'",
            Self::Framework => "'framework'",
            Self::Mapping => "'mapping'",
            Self::LBrace => "'{'",
            Self::RBrace => "'}'",
            Self::LParen => "'('",
            Self::RParen => "')'",
            Self::LBracket => "'['",
            Self::RBracket => "']'",
            Self::Semicolon => "';'",
            Self::Colon => "':'",
            Self::Comma => "','",
            Self::Dot => "'.'",
            Self::Arrow => "'->'",
            Self::FatArrow => "'=>'",
            Self::Plus => "'+'",
            Self::Minus => "'-'",
            Self::Star => "'*'",
            Self::Slash => "'/'",
            Self::Eq => "'='",
            Self::EqEq => "'=='",
            Self::NotEq => "'!='",
            Self::Lt => "'<'",
            Self::Gt => "'>'",
            Self::LtEq => "'<='",
            Self::GtEq => "'>='",
            Self::Conjunction => "'⊗'",
            Self::Disjunction => "'⊕'",
            Self::OverrideOp => "'▷'",
            Self::ExceptionOp => "'⊘'",
            Self::StringLit(_) => "string literal",
            Self::IntLit(_) => "integer literal",
            Self::FloatLit(_) => "float literal",
            Self::Ident(_) => "identifier",
            Self::DateLit(_) => "date literal",
            Self::Eof => "end of file",
        }
    }

    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Self::Jurisdiction
                | Self::Obligation
                | Self::Permission
                | Self::Prohibition
                | Self::Requires
                | Self::Exempts
                | Self::Override
                | Self::Compose
                | Self::Temporal
                | Self::RiskLevel
                | Self::Domain
                | Self::Article
                | Self::Annex
                | Self::Formalizability
                | Self::Strategy
                | Self::Cost
                | Self::If
                | Self::Then
                | Self::Else
                | Self::And
                | Self::Or
                | Self::Not
                | Self::Implies
                | Self::Forall
                | Self::Exists
                | Self::True
                | Self::False
                | Self::Framework
                | Self::Mapping
        )
    }

    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::StringLit(_)
                | Self::IntLit(_)
                | Self::FloatLit(_)
                | Self::DateLit(_)
                | Self::True
                | Self::False
        )
    }

    pub fn is_composition_op(&self) -> bool {
        matches!(
            self,
            Self::Conjunction | Self::Disjunction | Self::OverrideOp | Self::ExceptionOp
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StringLit(s) => write!(f, "\"{}\"", s),
            Self::IntLit(n) => write!(f, "{}", n),
            Self::FloatLit(n) => write!(f, "{}", n),
            Self::Ident(s) => write!(f, "{}", s),
            Self::DateLit(s) => write!(f, "#{}", s),
            _ => write!(f, "{}", self.description()),
        }
    }
}

// ─── Token ──────────────────────────────────────────────────────

/// A token with its span in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn eof(pos: usize) -> Self {
        Self {
            kind: TokenKind::Eof,
            span: Span::new(pos, pos),
        }
    }

    pub fn is(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(kind)
    }

    pub fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// Keyword lookup table.
pub fn keyword_lookup(word: &str) -> Option<TokenKind> {
    match word {
        "jurisdiction" => Some(TokenKind::Jurisdiction),
        "obligation" => Some(TokenKind::Obligation),
        "permission" => Some(TokenKind::Permission),
        "prohibition" => Some(TokenKind::Prohibition),
        "requires" => Some(TokenKind::Requires),
        "exempts" => Some(TokenKind::Exempts),
        "override" => Some(TokenKind::Override),
        "compose" => Some(TokenKind::Compose),
        "temporal" => Some(TokenKind::Temporal),
        "risk_level" => Some(TokenKind::RiskLevel),
        "domain" => Some(TokenKind::Domain),
        "article" => Some(TokenKind::Article),
        "annex" => Some(TokenKind::Annex),
        "formalizability" => Some(TokenKind::Formalizability),
        "strategy" => Some(TokenKind::Strategy),
        "cost" => Some(TokenKind::Cost),
        "if" => Some(TokenKind::If),
        "then" => Some(TokenKind::Then),
        "else" => Some(TokenKind::Else),
        "and" => Some(TokenKind::And),
        "or" => Some(TokenKind::Or),
        "not" => Some(TokenKind::Not),
        "implies" => Some(TokenKind::Implies),
        "forall" => Some(TokenKind::Forall),
        "exists" => Some(TokenKind::Exists),
        "true" => Some(TokenKind::True),
        "false" => Some(TokenKind::False),
        "framework" => Some(TokenKind::Framework),
        "mapping" => Some(TokenKind::Mapping),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_lookup() {
        assert_eq!(keyword_lookup("jurisdiction"), Some(TokenKind::Jurisdiction));
        assert_eq!(keyword_lookup("if"), Some(TokenKind::If));
        assert_eq!(keyword_lookup("not_a_keyword"), None);
    }

    #[test]
    fn test_token_kind_description() {
        assert_eq!(TokenKind::Obligation.description(), "'obligation'");
        assert_eq!(TokenKind::LBrace.description(), "'{'");
        assert_eq!(TokenKind::Conjunction.description(), "'⊗'");
    }

    #[test]
    fn test_token_kind_flags() {
        assert!(TokenKind::Jurisdiction.is_keyword());
        assert!(!TokenKind::Plus.is_keyword());
        assert!(TokenKind::StringLit("hi".into()).is_literal());
        assert!(TokenKind::Conjunction.is_composition_op());
    }

    #[test]
    fn test_token_display() {
        let t = Token::new(TokenKind::Ident("foo".into()), Span::new(0, 3));
        assert_eq!(format!("{}", t), "foo");
    }
}
