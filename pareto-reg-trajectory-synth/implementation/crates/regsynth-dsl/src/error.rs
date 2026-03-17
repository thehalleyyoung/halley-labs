use crate::source_map::{SourceMap, Span};
use std::fmt;

/// All DSL error types.
#[derive(Debug, Clone)]
pub enum DslError {
    Lex(LexError),
    Parse(ParseError),
    Type(TypeError),
    Elaboration(ElaborationError),
}

impl DslError {
    pub fn span(&self) -> Span {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Type(e) => e.span,
            Self::Elaboration(e) => e.span,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Lex(e) => &e.message,
            Self::Parse(e) => &e.message,
            Self::Type(e) => &e.message,
            Self::Elaboration(e) => &e.message,
        }
    }

    /// Format this error with source context from a SourceMap.
    pub fn format_with_source(&self, source_map: &SourceMap) -> String {
        let kind = match self {
            Self::Lex(_) => "lex error",
            Self::Parse(_) => "parse error",
            Self::Type(_) => "type error",
            Self::Elaboration(_) => "elaboration error",
        };
        let mut out = format!("error[{}]: {}\n", kind, self.message());
        out.push_str(&source_map.format_span_with_context(self.span()));
        if let Some(suggestion) = self.suggestion() {
            out.push_str(&format!("help: {}\n", suggestion));
        }
        out
    }

    pub fn suggestion(&self) -> Option<&str> {
        match self {
            Self::Lex(e) => e.suggestion.as_deref(),
            Self::Parse(e) => e.suggestion.as_deref(),
            Self::Type(e) => e.suggestion.as_deref(),
            Self::Elaboration(e) => e.suggestion.as_deref(),
        }
    }
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for DslError {}

impl From<LexError> for DslError {
    fn from(e: LexError) -> Self {
        Self::Lex(e)
    }
}

impl From<ParseError> for DslError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

impl From<TypeError> for DslError {
    fn from(e: TypeError) -> Self {
        Self::Type(e)
    }
}

impl From<ElaborationError> for DslError {
    fn from(e: ElaborationError) -> Self {
        Self::Elaboration(e)
    }
}

// ─── Lex Error ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LexError {
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

impl LexError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    pub fn unexpected_char(span: Span, ch: char) -> Self {
        Self::new(span, format!("unexpected character '{}'", ch))
    }

    pub fn unterminated_string(span: Span) -> Self {
        Self::new(span, "unterminated string literal")
            .with_suggestion("add a closing '\"'")
    }

    pub fn unterminated_comment(span: Span) -> Self {
        Self::new(span, "unterminated block comment")
            .with_suggestion("add a closing '*/'")
    }

    pub fn invalid_date(span: Span, text: &str) -> Self {
        Self::new(span, format!("invalid date literal '{}'", text))
            .with_suggestion("use format YYYY-MM-DD")
    }

    pub fn invalid_number(span: Span, text: &str) -> Self {
        Self::new(span, format!("invalid number '{}'", text))
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ─── Parse Error ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ParseError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    pub fn unexpected_token(span: Span, expected: &str, found: &str) -> Self {
        Self::new(
            span,
            format!("expected {}, found {}", expected, found),
        )
    }

    pub fn unexpected_eof(span: Span) -> Self {
        Self::new(span, "unexpected end of input")
    }

    pub fn missing_field(span: Span, field: &str, context: &str) -> Self {
        Self::new(span, format!("missing '{}' in {}", field, context))
            .with_suggestion(format!("add '{}' field", field))
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ─── Type Error ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypeError {
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

impl TypeError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    pub fn type_mismatch(span: Span, expected: &str, found: &str) -> Self {
        Self::new(span, format!("type mismatch: expected {}, found {}", expected, found))
    }

    pub fn undefined_variable(span: Span, name: &str) -> Self {
        Self::new(span, format!("undefined variable '{}'", name))
    }

    pub fn undefined_jurisdiction(span: Span, name: &str) -> Self {
        Self::new(span, format!("undefined jurisdiction '{}'", name))
    }

    pub fn duplicate_definition(span: Span, name: &str) -> Self {
        Self::new(span, format!("duplicate definition of '{}'", name))
    }

    pub fn invalid_composition(span: Span, msg: &str) -> Self {
        Self::new(span, format!("invalid obligation composition: {}", msg))
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ─── Elaboration Error ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ElaborationError {
    pub span: Span,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ElaborationError {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            suggestion: None,
        }
    }

    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    pub fn unresolved_reference(span: Span, name: &str) -> Self {
        Self::new(span, format!("unresolved reference '{}'", name))
    }

    pub fn temporal_conflict(span: Span, msg: &str) -> Self {
        Self::new(span, format!("temporal conflict: {}", msg))
    }

    pub fn composition_error(span: Span, msg: &str) -> Self {
        Self::new(span, format!("composition error: {}", msg))
    }
}

impl fmt::Display for ElaborationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_error_display() {
        let e = LexError::unexpected_char(Span::new(0, 1), '@');
        assert!(e.to_string().contains("unexpected character '@'"));
    }

    #[test]
    fn test_parse_error_with_suggestion() {
        let e = ParseError::unexpected_token(Span::new(5, 10), "identifier", "'{'");
        assert!(e.to_string().contains("expected identifier"));
    }

    #[test]
    fn test_type_error_display() {
        let e = TypeError::type_mismatch(Span::new(0, 5), "Bool", "Int");
        assert!(e.to_string().contains("type mismatch"));
    }

    #[test]
    fn test_dsl_error_format_with_source() {
        let mut sm = SourceMap::new();
        sm.add_file("test.rsl", "let x = @bad;\n");
        let e = DslError::Lex(LexError::unexpected_char(Span::new(8, 9), '@'));
        let formatted = e.format_with_source(&sm);
        assert!(formatted.contains("lex error"));
        assert!(formatted.contains("unexpected character '@'"));
        assert!(formatted.contains("test.rsl"));
    }

    #[test]
    fn test_error_conversions() {
        let lex: DslError = LexError::new(Span::new(0, 1), "bad").into();
        assert!(matches!(lex, DslError::Lex(_)));
        let parse: DslError = ParseError::new(Span::new(0, 1), "bad").into();
        assert!(matches!(parse, DslError::Parse(_)));
        let ty: DslError = TypeError::new(Span::new(0, 1), "bad").into();
        assert!(matches!(ty, DslError::Type(_)));
        let elab: DslError = ElaborationError::new(Span::new(0, 1), "bad").into();
        assert!(matches!(elab, DslError::Elaboration(_)));
    }
}
