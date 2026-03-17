//! Token definitions for the Choreo DSL lexer.
//!
//! Defines all token kinds, literal units, and the `Token` struct that carries
//! span information through the compilation pipeline.

use choreo_types::Span;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Duration and distance unit enums
// ---------------------------------------------------------------------------

/// Unit for duration literals (e.g., `500ms`, `2s`, `1.5min`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DurationUnit {
    /// Milliseconds
    Ms,
    /// Seconds
    S,
    /// Minutes
    Min,
}

impl DurationUnit {
    /// Convert a value with this unit to seconds.
    pub fn to_seconds(self, value: f64) -> f64 {
        match self {
            DurationUnit::Ms => value / 1000.0,
            DurationUnit::S => value,
            DurationUnit::Min => value * 60.0,
        }
    }
}

impl fmt::Display for DurationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DurationUnit::Ms => write!(f, "ms"),
            DurationUnit::S => write!(f, "s"),
            DurationUnit::Min => write!(f, "min"),
        }
    }
}

/// Unit for distance literals (e.g., `30cm`, `1.5m`, `250mm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceUnit {
    /// Millimeters
    Mm,
    /// Centimeters
    Cm,
    /// Meters
    M,
}

impl DistanceUnit {
    /// Convert a value with this unit to meters.
    pub fn to_meters(self, value: f64) -> f64 {
        match self {
            DistanceUnit::Mm => value / 1000.0,
            DistanceUnit::Cm => value / 100.0,
            DistanceUnit::M => value,
        }
    }
}

impl fmt::Display for DistanceUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceUnit::Mm => write!(f, "mm"),
            DistanceUnit::Cm => write!(f, "cm"),
            DistanceUnit::M => write!(f, "m"),
        }
    }
}

// ---------------------------------------------------------------------------
// TokenKind
// ---------------------------------------------------------------------------

/// Every kind of token the lexer can produce.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenKind {
    // ---- Keywords: declarations ----
    /// `region`
    Region,
    /// `interaction`
    Interaction,
    /// `scene`
    Scene,
    /// `entity`
    Entity,
    /// `zone`
    Zone,

    // ---- Keywords: gesture / spatial primitives ----
    /// `gaze`
    Gaze,
    /// `reach`
    Reach,
    /// `grab`
    Grab,
    /// `release`
    Release,
    /// `proximity`
    Proximity,
    /// `inside`
    Inside,
    /// `contains`
    Contains,
    /// `touch`
    Touch,

    // ---- Keywords: actions ----
    /// `activate`
    Activate,
    /// `deactivate`
    Deactivate,
    /// `emit`
    Emit,
    /// `spawn`
    Spawn,
    /// `destroy`
    Destroy,
    /// `set_timer`
    SetTimer,
    /// `cancel_timer`
    CancelTimer,

    // ---- Keywords: temporal ----
    /// `timeout`
    Timeout,
    /// `when`
    When,
    /// `then`
    Then,
    /// `within`
    Within,
    /// `after`
    After,

    // ---- Keywords: logical operators ----
    /// `and`
    And,
    /// `or`
    Or,
    /// `not`
    Not,

    // ---- Keywords: choreography combinators ----
    /// `par`
    Par,
    /// `seq`
    Seq,
    /// `choice`
    Choice,
    /// `loop`
    Loop,

    // ---- Keywords: bindings ----
    /// `let`
    Let,
    /// `in`
    In,
    /// `if`
    If,
    /// `else`
    Else,

    // ---- Keywords: import ----
    /// `import`
    Import,

    // ---- Keywords: geometry ----
    /// `box`
    Box_,
    /// `sphere`
    Sphere_,
    /// `capsule`
    Capsule_,
    /// `cylinder`
    Cylinder_,
    /// `convex_hull`
    ConvexHull,
    /// `union`
    Union,
    /// `intersection`
    Intersection,
    /// `difference`
    Difference,
    /// `transform`
    Transform,

    // ---- Keywords: booleans ----
    /// `true`
    True,
    /// `false`
    False,

    // ---- Punctuation ----
    /// `->`
    Arrow,
    /// `=>`
    FatArrow,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `=`
    Eq,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `!=`
    Ne,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `&`
    Ampersand,
    /// `|`
    Pipe,
    /// `@`
    At,
    /// `#`
    Hash,
    /// `..`
    DotDot,

    // ---- Literals ----
    /// An identifier like `my_region`
    Identifier(String),
    /// An integer literal like `42`
    IntLiteral(i64),
    /// A floating-point literal like `3.14`
    FloatLiteral(f64),
    /// A string literal like `"hello"`
    StringLiteral(String),
    /// A duration literal like `500ms`, `2s`, `1.5min`
    DurationLiteral(f64, DurationUnit),
    /// A distance literal like `30cm`, `1.5m`
    DistanceLiteral(f64, DistanceUnit),
    /// An angle literal like `45deg`
    AngleLiteral(f64),

    // ---- Special ----
    /// End of file
    Eof,
    /// A newline (tracked for line-sensitive contexts)
    Newline,
    /// A comment, either `//` line or `/* */` block
    Comment(String),
}

impl TokenKind {
    /// Returns `true` if this token is a keyword.
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Region
                | TokenKind::Interaction
                | TokenKind::Scene
                | TokenKind::Entity
                | TokenKind::Zone
                | TokenKind::Gaze
                | TokenKind::Reach
                | TokenKind::Grab
                | TokenKind::Release
                | TokenKind::Proximity
                | TokenKind::Inside
                | TokenKind::Contains
                | TokenKind::Touch
                | TokenKind::Activate
                | TokenKind::Deactivate
                | TokenKind::Emit
                | TokenKind::Spawn
                | TokenKind::Destroy
                | TokenKind::SetTimer
                | TokenKind::CancelTimer
                | TokenKind::Timeout
                | TokenKind::When
                | TokenKind::Then
                | TokenKind::Within
                | TokenKind::After
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Not
                | TokenKind::Par
                | TokenKind::Seq
                | TokenKind::Choice
                | TokenKind::Loop
                | TokenKind::Let
                | TokenKind::In
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Import
                | TokenKind::Box_
                | TokenKind::Sphere_
                | TokenKind::Capsule_
                | TokenKind::Cylinder_
                | TokenKind::ConvexHull
                | TokenKind::Union
                | TokenKind::Intersection
                | TokenKind::Difference
                | TokenKind::Transform
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Returns `true` if this token is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntLiteral(_)
                | TokenKind::FloatLiteral(_)
                | TokenKind::StringLiteral(_)
                | TokenKind::DurationLiteral(_, _)
                | TokenKind::DistanceLiteral(_, _)
                | TokenKind::AngleLiteral(_)
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Returns `true` for tokens that start a declaration.
    pub fn is_declaration_start(&self) -> bool {
        matches!(
            self,
            TokenKind::Region
                | TokenKind::Interaction
                | TokenKind::Scene
                | TokenKind::Entity
                | TokenKind::Zone
                | TokenKind::Let
                | TokenKind::Import
        )
    }

    /// Returns `true` for tokens that can synchronize after errors.
    pub fn is_sync_token(&self) -> bool {
        matches!(
            self,
            TokenKind::Region
                | TokenKind::Interaction
                | TokenKind::Scene
                | TokenKind::Entity
                | TokenKind::Zone
                | TokenKind::Let
                | TokenKind::Import
                | TokenKind::RBrace
                | TokenKind::Semicolon
                | TokenKind::Eof
        )
    }

    /// Returns the keyword string if this is a keyword, or `None`.
    pub fn keyword_str(&self) -> Option<&'static str> {
        match self {
            TokenKind::Region => Some("region"),
            TokenKind::Interaction => Some("interaction"),
            TokenKind::Scene => Some("scene"),
            TokenKind::Entity => Some("entity"),
            TokenKind::Zone => Some("zone"),
            TokenKind::Gaze => Some("gaze"),
            TokenKind::Reach => Some("reach"),
            TokenKind::Grab => Some("grab"),
            TokenKind::Release => Some("release"),
            TokenKind::Proximity => Some("proximity"),
            TokenKind::Inside => Some("inside"),
            TokenKind::Contains => Some("contains"),
            TokenKind::Touch => Some("touch"),
            TokenKind::Activate => Some("activate"),
            TokenKind::Deactivate => Some("deactivate"),
            TokenKind::Emit => Some("emit"),
            TokenKind::Spawn => Some("spawn"),
            TokenKind::Destroy => Some("destroy"),
            TokenKind::SetTimer => Some("set_timer"),
            TokenKind::CancelTimer => Some("cancel_timer"),
            TokenKind::Timeout => Some("timeout"),
            TokenKind::When => Some("when"),
            TokenKind::Then => Some("then"),
            TokenKind::Within => Some("within"),
            TokenKind::After => Some("after"),
            TokenKind::And => Some("and"),
            TokenKind::Or => Some("or"),
            TokenKind::Not => Some("not"),
            TokenKind::Par => Some("par"),
            TokenKind::Seq => Some("seq"),
            TokenKind::Choice => Some("choice"),
            TokenKind::Loop => Some("loop"),
            TokenKind::Let => Some("let"),
            TokenKind::In => Some("in"),
            TokenKind::If => Some("if"),
            TokenKind::Else => Some("else"),
            TokenKind::Import => Some("import"),
            TokenKind::Box_ => Some("box"),
            TokenKind::Sphere_ => Some("sphere"),
            TokenKind::Capsule_ => Some("capsule"),
            TokenKind::Cylinder_ => Some("cylinder"),
            TokenKind::ConvexHull => Some("convex_hull"),
            TokenKind::Union => Some("union"),
            TokenKind::Intersection => Some("intersection"),
            TokenKind::Difference => Some("difference"),
            TokenKind::Transform => Some("transform"),
            TokenKind::True => Some("true"),
            TokenKind::False => Some("false"),
            _ => None,
        }
    }

    /// Look up a keyword from an identifier string.
    pub fn from_keyword(s: &str) -> Option<TokenKind> {
        match s {
            "region" => Some(TokenKind::Region),
            "interaction" => Some(TokenKind::Interaction),
            "scene" => Some(TokenKind::Scene),
            "entity" => Some(TokenKind::Entity),
            "zone" => Some(TokenKind::Zone),
            "gaze" => Some(TokenKind::Gaze),
            "reach" => Some(TokenKind::Reach),
            "grab" => Some(TokenKind::Grab),
            "release" => Some(TokenKind::Release),
            "proximity" => Some(TokenKind::Proximity),
            "inside" => Some(TokenKind::Inside),
            "contains" => Some(TokenKind::Contains),
            "touch" => Some(TokenKind::Touch),
            "activate" => Some(TokenKind::Activate),
            "deactivate" => Some(TokenKind::Deactivate),
            "emit" => Some(TokenKind::Emit),
            "spawn" => Some(TokenKind::Spawn),
            "destroy" => Some(TokenKind::Destroy),
            "set_timer" => Some(TokenKind::SetTimer),
            "cancel_timer" => Some(TokenKind::CancelTimer),
            "timeout" => Some(TokenKind::Timeout),
            "when" => Some(TokenKind::When),
            "then" => Some(TokenKind::Then),
            "within" => Some(TokenKind::Within),
            "after" => Some(TokenKind::After),
            "and" => Some(TokenKind::And),
            "or" => Some(TokenKind::Or),
            "not" => Some(TokenKind::Not),
            "par" => Some(TokenKind::Par),
            "seq" => Some(TokenKind::Seq),
            "choice" => Some(TokenKind::Choice),
            "loop" => Some(TokenKind::Loop),
            "let" => Some(TokenKind::Let),
            "in" => Some(TokenKind::In),
            "if" => Some(TokenKind::If),
            "else" => Some(TokenKind::Else),
            "import" => Some(TokenKind::Import),
            "box" => Some(TokenKind::Box_),
            "sphere" => Some(TokenKind::Sphere_),
            "capsule" => Some(TokenKind::Capsule_),
            "cylinder" => Some(TokenKind::Cylinder_),
            "convex_hull" => Some(TokenKind::ConvexHull),
            "union" => Some(TokenKind::Union),
            "intersection" => Some(TokenKind::Intersection),
            "difference" => Some(TokenKind::Difference),
            "transform" => Some(TokenKind::Transform),
            "true" => Some(TokenKind::True),
            "false" => Some(TokenKind::False),
            _ => None,
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Identifier(s) => write!(f, "identifier `{}`", s),
            TokenKind::IntLiteral(n) => write!(f, "integer `{}`", n),
            TokenKind::FloatLiteral(n) => write!(f, "float `{}`", n),
            TokenKind::StringLiteral(s) => write!(f, "string \"{}\"", s),
            TokenKind::DurationLiteral(v, u) => write!(f, "duration `{}{}`", v, u),
            TokenKind::DistanceLiteral(v, u) => write!(f, "distance `{}{}`", v, u),
            TokenKind::AngleLiteral(v) => write!(f, "angle `{}deg`", v),
            TokenKind::Arrow => write!(f, "`->`"),
            TokenKind::FatArrow => write!(f, "`=>`"),
            TokenKind::LParen => write!(f, "`(`"),
            TokenKind::RParen => write!(f, "`)`"),
            TokenKind::LBrace => write!(f, "`{{`"),
            TokenKind::RBrace => write!(f, "`}}`"),
            TokenKind::LBracket => write!(f, "`[`"),
            TokenKind::RBracket => write!(f, "`]`"),
            TokenKind::Comma => write!(f, "`,`"),
            TokenKind::Semicolon => write!(f, "`;`"),
            TokenKind::Colon => write!(f, "`:`"),
            TokenKind::Dot => write!(f, "`.`"),
            TokenKind::DotDot => write!(f, "`..`"),
            TokenKind::Eq => write!(f, "`=`"),
            TokenKind::Lt => write!(f, "`<`"),
            TokenKind::Gt => write!(f, "`>`"),
            TokenKind::Le => write!(f, "`<=`"),
            TokenKind::Ge => write!(f, "`>=`"),
            TokenKind::Ne => write!(f, "`!=`"),
            TokenKind::Plus => write!(f, "`+`"),
            TokenKind::Minus => write!(f, "`-`"),
            TokenKind::Star => write!(f, "`*`"),
            TokenKind::Slash => write!(f, "`/`"),
            TokenKind::Ampersand => write!(f, "`&`"),
            TokenKind::Pipe => write!(f, "`|`"),
            TokenKind::At => write!(f, "`@`"),
            TokenKind::Hash => write!(f, "`#`"),
            TokenKind::Eof => write!(f, "end of file"),
            TokenKind::Newline => write!(f, "newline"),
            TokenKind::Comment(s) => write!(f, "comment `{}`", s),
            other => {
                if let Some(kw) = other.keyword_str() {
                    write!(f, "keyword `{}`", kw)
                } else {
                    write!(f, "{:?}", other)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Token struct
// ---------------------------------------------------------------------------

/// A single lexer token carrying its kind, source span, and original lexeme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, lexeme: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            lexeme: lexeme.into(),
        }
    }

    /// Create an EOF token.
    pub fn eof(offset: usize) -> Self {
        Self {
            kind: TokenKind::Eof,
            span: Span {
                start: offset,
                end: offset,
                file: None,
            },
            lexeme: String::new(),
        }
    }

    pub fn is_eof(&self) -> bool {
        self.kind == TokenKind::Eof
    }

    /// Check if this token matches the given kind.
    pub fn is(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(kind)
    }

    /// If this is an identifier token, return its name.
    pub fn as_identifier(&self) -> Option<&str> {
        match &self.kind {
            TokenKind::Identifier(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this is an int literal, return its value.
    pub fn as_int(&self) -> Option<i64> {
        match &self.kind {
            TokenKind::IntLiteral(n) => Some(*n),
            _ => None,
        }
    }

    /// If this is a float literal, return its value.
    pub fn as_float(&self) -> Option<f64> {
        match &self.kind {
            TokenKind::FloatLiteral(n) => Some(*n),
            _ => None,
        }
    }

    /// If this is a string literal, return its content.
    pub fn as_string(&self) -> Option<&str> {
        match &self.kind {
            TokenKind::StringLiteral(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_lookup() {
        assert_eq!(TokenKind::from_keyword("region"), Some(TokenKind::Region));
        assert_eq!(TokenKind::from_keyword("gaze"), Some(TokenKind::Gaze));
        assert_eq!(TokenKind::from_keyword("par"), Some(TokenKind::Par));
        assert_eq!(TokenKind::from_keyword("true"), Some(TokenKind::True));
        assert_eq!(TokenKind::from_keyword("false"), Some(TokenKind::False));
        assert_eq!(TokenKind::from_keyword("foobar"), None);
        assert_eq!(TokenKind::from_keyword("box"), Some(TokenKind::Box_));
        assert_eq!(TokenKind::from_keyword("convex_hull"), Some(TokenKind::ConvexHull));
    }

    #[test]
    fn test_keyword_roundtrip() {
        let keywords = vec![
            "region", "interaction", "scene", "entity", "zone", "gaze",
            "reach", "grab", "release", "proximity", "inside", "contains",
            "touch", "activate", "deactivate", "timeout", "when", "then",
            "and", "or", "not", "par", "seq", "choice", "loop",
            "let", "in", "if", "else", "true", "false",
        ];
        for kw in keywords {
            let kind = TokenKind::from_keyword(kw).unwrap();
            assert_eq!(kind.keyword_str(), Some(kw));
        }
    }

    #[test]
    fn test_duration_unit_to_seconds() {
        assert!((DurationUnit::Ms.to_seconds(500.0) - 0.5).abs() < 1e-9);
        assert!((DurationUnit::S.to_seconds(2.0) - 2.0).abs() < 1e-9);
        assert!((DurationUnit::Min.to_seconds(1.5) - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_unit_to_meters() {
        assert!((DistanceUnit::Mm.to_meters(1000.0) - 1.0).abs() < 1e-9);
        assert!((DistanceUnit::Cm.to_meters(100.0) - 1.0).abs() < 1e-9);
        assert!((DistanceUnit::M.to_meters(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_keyword() {
        assert!(TokenKind::Region.is_keyword());
        assert!(TokenKind::True.is_keyword());
        assert!(!TokenKind::Identifier("foo".into()).is_keyword());
        assert!(!TokenKind::IntLiteral(42).is_keyword());
        assert!(!TokenKind::Arrow.is_keyword());
    }

    #[test]
    fn test_is_literal() {
        assert!(TokenKind::IntLiteral(42).is_literal());
        assert!(TokenKind::FloatLiteral(3.14).is_literal());
        assert!(TokenKind::StringLiteral("hello".into()).is_literal());
        assert!(TokenKind::DurationLiteral(500.0, DurationUnit::Ms).is_literal());
        assert!(TokenKind::DistanceLiteral(1.5, DistanceUnit::M).is_literal());
        assert!(TokenKind::AngleLiteral(45.0).is_literal());
        assert!(TokenKind::True.is_literal());
        assert!(TokenKind::False.is_literal());
        assert!(!TokenKind::Identifier("x".into()).is_literal());
    }

    #[test]
    fn test_token_creation_and_accessors() {
        let span = Span { start: 0, end: 3, file: None };
        let tok = Token::new(TokenKind::Identifier("foo".into()), span.clone(), "foo");
        assert_eq!(tok.as_identifier(), Some("foo"));
        assert_eq!(tok.as_int(), None);
        assert!(!tok.is_eof());

        let tok2 = Token::new(TokenKind::IntLiteral(42), span.clone(), "42");
        assert_eq!(tok2.as_int(), Some(42));

        let tok3 = Token::new(TokenKind::FloatLiteral(3.14), span.clone(), "3.14");
        assert!((tok3.as_float().unwrap() - 3.14).abs() < 1e-9);

        let tok4 = Token::new(TokenKind::StringLiteral("hi".into()), span, "\"hi\"");
        assert_eq!(tok4.as_string(), Some("hi"));
    }

    #[test]
    fn test_eof_token() {
        let eof = Token::eof(100);
        assert!(eof.is_eof());
        assert_eq!(eof.span.start, 100);
        assert_eq!(eof.span.end, 100);
    }

    #[test]
    fn test_is_declaration_start() {
        assert!(TokenKind::Region.is_declaration_start());
        assert!(TokenKind::Entity.is_declaration_start());
        assert!(TokenKind::Let.is_declaration_start());
        assert!(TokenKind::Import.is_declaration_start());
        assert!(!TokenKind::When.is_declaration_start());
        assert!(!TokenKind::Gaze.is_declaration_start());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TokenKind::Arrow), "`->`");
        assert_eq!(format!("{}", TokenKind::Eof), "end of file");
        assert_eq!(format!("{}", TokenKind::Region), "keyword `region`");
        let tok = Token::new(
            TokenKind::IntLiteral(42),
            Span { start: 0, end: 2, file: None },
            "42",
        );
        assert_eq!(format!("{}", tok), "integer `42`");
    }

    #[test]
    fn test_duration_display() {
        assert_eq!(format!("{}", DurationUnit::Ms), "ms");
        assert_eq!(format!("{}", DurationUnit::S), "s");
        assert_eq!(format!("{}", DurationUnit::Min), "min");
    }

    #[test]
    fn test_distance_display() {
        assert_eq!(format!("{}", DistanceUnit::Mm), "mm");
        assert_eq!(format!("{}", DistanceUnit::Cm), "cm");
        assert_eq!(format!("{}", DistanceUnit::M), "m");
    }

    #[test]
    fn test_token_is_check() {
        let span = Span { start: 0, end: 1, file: None };
        let tok = Token::new(TokenKind::Identifier("x".into()), span, "x");
        assert!(tok.is(&TokenKind::Identifier(String::new())));
        assert!(!tok.is(&TokenKind::IntLiteral(0)));
    }
}
